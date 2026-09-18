"""Exposure-aware multitype branching inference for censored cascade logs.

The complete-tree estimator used before Milestone 6 conditions on observed
extinction and cannot establish supercriticality.  This module instead models
children of type b from a type-a parent as an age-dependent Poisson process

    lambda_ab(u) = B_ab * gamma_ab * exp(-gamma_ab * u),  u >= 0.

If parent i is followed for C_i seconds, its integrated kernel exposure is
1 - exp(-gamma_ab * C_i).  With observed direct-child count N_ab, the
maximum-likelihood estimate is therefore

    B_hat_ab = N_ab / sum_i [1 - exp(-gamma_ab * C_i)].

The event parentage must come from causal attribution.  Event identifiers are
rollout-local, and all pooling/bootstrap operations preserve rollout
boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Iterable, Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq, minimize_scalar

from .capacity_theory import spectral_radius
from .simulation import InterventionEvent


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class TimedBranchingEvent:
    """One causally attributed event in an age-dependent branching forest."""

    event_id: int
    event_type: int
    birth_time: float
    parent_event_id: int | None
    root_event_id: int
    generation: int
    exposure_end_time: float | None = None


@dataclass(frozen=True)
class CensoredBranchingRollout:
    """One independent event forest observed until a finite stopping time."""

    events: tuple[TimedBranchingEvent, ...]
    observation_end: float
    requested_horizon: float
    population_cap: int | None = None
    population_cap_reached: bool = False

    def __post_init__(self) -> None:
        records = tuple(self.events)
        object.__setattr__(self, "events", records)
        if (
            not np.isfinite(self.observation_end)
            or self.observation_end <= 0.0
            or not np.isfinite(self.requested_horizon)
            or self.requested_horizon <= 0.0
            or self.observation_end > self.requested_horizon + 1e-12
        ):
            raise ValueError(
                "observation times must satisfy 0 < observation_end <= requested_horizon"
            )
        if self.population_cap is not None and self.population_cap <= 0:
            raise ValueError("population_cap must be positive when supplied")
        if self.population_cap_reached:
            if self.population_cap is None:
                raise ValueError("a reached population cap must be specified")
            if len(records) != self.population_cap:
                raise ValueError(
                    "a population-capped rollout must contain exactly population_cap events"
                )

        identifiers = [event.event_id for event in records]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("event_id values must be unique within each rollout")
        lookup = {event.event_id: event for event in records}
        for event in records:
            if event.event_id < 0 or event.event_type < 0 or event.generation < 0:
                raise ValueError("event identifiers, types, and generations must be nonnegative")
            if (
                not np.isfinite(event.birth_time)
                or event.birth_time < 0.0
                or event.birth_time > self.observation_end + 1e-12
            ):
                raise ValueError("event birth times must lie inside the observation window")
            if event.exposure_end_time is not None and (
                not np.isfinite(event.exposure_end_time)
                or event.exposure_end_time < event.birth_time - 1e-12
                or event.exposure_end_time > self.observation_end + 1e-12
            ):
                raise ValueError(
                    "event exposure ends must lie between birth and observation end"
                )
            if event.parent_event_id is None:
                if event.generation != 0 or event.root_event_id != event.event_id:
                    raise ValueError("each root must have generation zero and identify itself")
                continue
            if event.parent_event_id not in lookup:
                raise ValueError("every non-root event must retain its observed direct parent")
            parent = lookup[event.parent_event_id]
            if parent.birth_time > event.birth_time + 1e-12:
                raise ValueError("a child cannot precede its parent")
            parent_end = (
                self.observation_end
                if parent.exposure_end_time is None
                else parent.exposure_end_time
            )
            if event.birth_time > parent_end + 1e-12:
                raise ValueError("a child cannot occur after its parent's exposure ends")
            if event.generation != parent.generation + 1:
                raise ValueError("child generation must be parent generation plus one")
            if event.root_event_id != parent.root_event_id:
                raise ValueError("parent and child must belong to the same root tree")


def right_censor_branching_rollout(
    rollout: CensoredBranchingRollout,
    observation_end: float,
) -> CensoredBranchingRollout:
    """Return the prefix observed at a requested observation end.

    This supports paired horizon-sensitivity experiments: every horizon sees a
    nested prefix of the same realized forest. A population cap that occurred
    before the requested cutoff remains the actual chronological stopping rule.
    """

    cutoff = float(observation_end)
    if (
        not np.isfinite(cutoff)
        or cutoff <= 0.0
        or cutoff > rollout.requested_horizon + 1e-12
    ):
        raise ValueError(
            "observation_end must lie in (0, rollout.requested_horizon]"
        )
    cap_precedes_cutoff = bool(
        rollout.population_cap_reached
        and rollout.observation_end <= cutoff + 1e-12
    )
    actual_end = rollout.observation_end if cap_precedes_cutoff else cutoff
    events = tuple(
        TimedBranchingEvent(
            event_id=event.event_id,
            event_type=event.event_type,
            birth_time=event.birth_time,
            parent_event_id=event.parent_event_id,
            root_event_id=event.root_event_id,
            generation=event.generation,
            exposure_end_time=(
                None
                if event.exposure_end_time is None
                else min(event.exposure_end_time, actual_end)
            ),
        )
        for event in rollout.events
        if event.birth_time <= actual_end + 1e-12
    )
    return CensoredBranchingRollout(
        events=events,
        observation_end=float(actual_end),
        requested_horizon=cutoff,
        population_cap=rollout.population_cap,
        population_cap_reached=cap_precedes_cutoff,
    )


@dataclass(frozen=True)
class ExposureBranchingEstimate:
    """Sufficient statistics and fitted direct-offspring matrix."""

    matrix: FloatArray
    direct_child_counts: IntArray
    exposure_mass: FloatArray
    parent_counts: IntArray
    identifiable_rows: BoolArray
    spectral_radius: float
    rollout_count: int
    event_count: int
    temporally_censored_parent_count: int
    method: str


@dataclass(frozen=True)
class ExposureBootstrapResult:
    """Cluster-bootstrap uncertainty for the exposure-aware estimator."""

    point: ExposureBranchingEstimate
    matrix_lower: FloatArray
    matrix_upper: FloatArray
    spectral_radius_lower: float
    spectral_radius_upper: float
    probability_supercritical: float
    successful_replications: int
    requested_replications: int
    confidence_level: float


ObservationModel = Literal["continuous", "grouped"]


@dataclass(frozen=True)
class JointExposureBranchingEstimate:
    """Joint estimate of offspring means and recovery rates.

    ``continuous`` uses exact child ages. ``grouped`` uses the probability mass
    of a geometric sequence induced by sampling an exponential kernel on a
    fixed integration grid. The latter is the appropriate likelihood for the
    simulator, where a large fraction of direct children can share their
    parent's recorded start timestamp.
    """

    matrix: FloatArray
    recovery_rates: FloatArray
    direct_child_counts: IntArray
    exposure_mass: FloatArray
    parent_counts: IntArray
    identifiable_pairs: BoolArray
    recovery_at_bound: BoolArray
    spectral_radius: float
    rollout_count: int
    event_count: int
    edge_count: int
    zero_lag_edge_count: int
    log_likelihood: float
    observation_model: ObservationModel
    time_step: float | None


@dataclass(frozen=True)
class JointExposureBootstrapResult:
    """Rollout-cluster bootstrap for joint offspring/recovery inference."""

    point: JointExposureBranchingEstimate
    matrix_lower: FloatArray
    matrix_upper: FloatArray
    recovery_lower: FloatArray
    recovery_upper: FloatArray
    spectral_radius_lower: float
    spectral_radius_upper: float
    probability_supercritical: float
    successful_replications: int
    requested_replications: int
    confidence_level: float


@dataclass(frozen=True)
class TimestampAudit:
    """Audit of whether fixed-step event logs support an exact-time likelihood."""

    rollout_count: int
    event_count: int
    edge_count: int
    zero_lag_edge_count: int
    zero_lag_fraction: float
    off_grid_birth_count: int
    maximum_birth_grid_error: float
    minimum_positive_lag: float | None
    median_positive_lag: float | None
    exact_time_likelihood_supported: bool


def _event_exposure_end(
    event: TimedBranchingEvent, rollout: CensoredBranchingRollout
) -> float:
    return float(
        rollout.observation_end
        if event.exposure_end_time is None
        else event.exposure_end_time
    )


def _positive_square(name: str, value: ArrayLike) -> FloatArray:
    matrix = np.asarray(value, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] == 0
        or matrix.shape[0] != matrix.shape[1]
        or not np.all(np.isfinite(matrix))
        or np.any(matrix <= 0.0)
    ):
        raise ValueError(f"{name} must be a finite, strictly positive square matrix")
    return matrix


def _nonnegative_square(name: str, value: ArrayLike) -> FloatArray:
    matrix = np.asarray(value, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] == 0
        or matrix.shape[0] != matrix.shape[1]
        or not np.all(np.isfinite(matrix))
        or np.any(matrix < 0.0)
    ):
        raise ValueError(f"{name} must be a finite, nonnegative square matrix")
    return matrix


def exponential_kernel_exposure(age: ArrayLike, recovery_rate: ArrayLike):
    """Integrated exponential triggering kernel over an observed parent age."""

    follow_up = np.asarray(age, dtype=float)
    rate = np.asarray(recovery_rate, dtype=float)
    if (
        not np.all(np.isfinite(follow_up))
        or np.any(follow_up < 0.0)
        or not np.all(np.isfinite(rate))
        or np.any(rate <= 0.0)
    ):
        raise ValueError("age must be nonnegative and recovery_rate strictly positive")
    result = -np.expm1(-rate * follow_up)
    if np.ndim(age) == 0 and np.ndim(recovery_rate) == 0:
        return float(result)
    return result


def _validate_types(
    rollouts: Sequence[CensoredBranchingRollout], number_of_types: int
) -> None:
    for rollout in rollouts:
        for event in rollout.events:
            if event.event_type >= number_of_types:
                raise ValueError(
                    f"event type {event.event_type} is outside 0..{number_of_types - 1}"
                )


def estimate_censored_multitype_branching(
    rollouts: Iterable[CensoredBranchingRollout],
    recovery_rates: ArrayLike,
    *,
    exposure_tolerance: float = 1e-9,
) -> ExposureBranchingEstimate:
    """Fit B using every observed parent and its exact right-censored exposure."""

    records = tuple(rollouts)
    if not records:
        raise ValueError("at least one independent rollout is required")
    rates = _positive_square("recovery_rates", recovery_rates)
    if (
        not np.isfinite(exposure_tolerance)
        or exposure_tolerance <= 0.0
        or exposure_tolerance >= 1.0
    ):
        raise ValueError("exposure_tolerance must lie strictly between zero and one")
    type_count = rates.shape[0]
    _validate_types(records, type_count)

    children = np.zeros((type_count, type_count), dtype=np.int64)
    exposure = np.zeros((type_count, type_count), dtype=float)
    parents = np.zeros(type_count, dtype=np.int64)
    censored_parents = 0
    event_count = 0

    for rollout in records:
        lookup = {event.event_id: event for event in rollout.events}
        event_count += len(rollout.events)
        for event in rollout.events:
            parent_type = event.event_type
            parents[parent_type] += 1
            age = max(0.0, _event_exposure_end(event, rollout) - event.birth_time)
            parent_exposure = exponential_kernel_exposure(age, rates[parent_type])
            exposure[parent_type] += parent_exposure
            if np.any(parent_exposure < 1.0 - exposure_tolerance):
                censored_parents += 1
            if event.parent_event_id is not None:
                direct_parent = lookup[event.parent_event_id]
                children[direct_parent.event_type, event.event_type] += 1

    identifiable = np.all(exposure > exposure_tolerance, axis=1)
    matrix = np.zeros_like(exposure)
    np.divide(children, exposure, out=matrix, where=exposure > exposure_tolerance)
    return ExposureBranchingEstimate(
        matrix=matrix,
        direct_child_counts=children,
        exposure_mass=exposure,
        parent_counts=parents,
        identifiable_rows=identifiable,
        spectral_radius=float(spectral_radius(matrix)),
        rollout_count=len(records),
        event_count=event_count,
        temporally_censored_parent_count=censored_parents,
        method="exposure_likelihood",
    )


def estimate_naive_observed_tree(
    rollouts: Iterable[CensoredBranchingRollout],
    number_of_types: int,
) -> ExposureBranchingEstimate:
    """Complete-tree count estimator retained only as a censoring ablation."""

    records = tuple(rollouts)
    if not records:
        raise ValueError("at least one independent rollout is required")
    if number_of_types <= 0:
        raise ValueError("number_of_types must be positive")
    _validate_types(records, number_of_types)
    children = np.zeros((number_of_types, number_of_types), dtype=np.int64)
    parents = np.zeros(number_of_types, dtype=np.int64)
    event_count = 0
    for rollout in records:
        lookup = {event.event_id: event for event in rollout.events}
        event_count += len(rollout.events)
        for event in rollout.events:
            parents[event.event_type] += 1
            if event.parent_event_id is not None:
                direct_parent = lookup[event.parent_event_id]
                children[direct_parent.event_type, event.event_type] += 1
    denominator = np.repeat(parents[:, None], number_of_types, axis=1).astype(float)
    matrix = np.zeros_like(denominator)
    np.divide(children, denominator, out=matrix, where=denominator > 0.0)
    identifiable = parents > 0
    return ExposureBranchingEstimate(
        matrix=matrix,
        direct_child_counts=children,
        exposure_mass=denominator,
        parent_counts=parents,
        identifiable_rows=identifiable,
        spectral_radius=float(spectral_radius(matrix)),
        rollout_count=len(records),
        event_count=event_count,
        temporally_censored_parent_count=0,
        method="naive_observed_tree",
    )


def profile_log_likelihood(
    matrix: ArrayLike,
    estimate: ExposureBranchingEstimate,
) -> float:
    """B-dependent part of the Poisson-process log likelihood."""

    candidate = _nonnegative_square("matrix", matrix)
    if candidate.shape != estimate.matrix.shape:
        raise ValueError("matrix and sufficient statistics must have the same shape")
    counts = estimate.direct_child_counts.astype(float)
    positive_counts = counts > 0.0
    if np.any((candidate <= 0.0) & positive_counts):
        return float("-inf")
    log_term = np.zeros_like(candidate)
    log_term[positive_counts] = counts[positive_counts] * np.log(
        candidate[positive_counts]
    )
    return float(np.sum(log_term - candidate * estimate.exposure_mass))


def bootstrap_censored_multitype_branching(
    rollouts: Iterable[CensoredBranchingRollout],
    recovery_rates: ArrayLike,
    *,
    replications: int = 1_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> ExposureBootstrapResult:
    """Resample independent rollouts and propagate uncertainty through rho(B)."""

    records = tuple(rollouts)
    if not records:
        raise ValueError("at least one independent rollout is required")
    if replications <= 0:
        raise ValueError("replications must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    point = estimate_censored_multitype_branching(records, recovery_rates)
    active_rows = point.parent_counts > 0
    rng = np.random.default_rng(seed)
    matrices: list[FloatArray] = []
    radii: list[float] = []
    for _ in range(replications):
        indices = rng.integers(0, len(records), size=len(records))
        sample = tuple(records[int(index)] for index in indices)
        fitted = estimate_censored_multitype_branching(sample, recovery_rates)
        if np.all(fitted.identifiable_rows[active_rows]):
            matrices.append(fitted.matrix)
            radii.append(fitted.spectral_radius)
    if not matrices:
        raise RuntimeError("no identifiable cluster-bootstrap replication was obtained")
    alpha = (1.0 - confidence_level) / 2.0
    stack = np.stack(matrices)
    radius_array = np.asarray(radii, dtype=float)
    matrix_lower = np.minimum(np.quantile(stack, alpha, axis=0), point.matrix)
    matrix_upper = np.maximum(
        np.quantile(stack, 1.0 - alpha, axis=0), point.matrix
    )
    radius_lower = min(
        float(np.quantile(radius_array, alpha)), point.spectral_radius
    )
    radius_upper = max(
        float(np.quantile(radius_array, 1.0 - alpha)), point.spectral_radius
    )
    return ExposureBootstrapResult(
        point=point,
        matrix_lower=matrix_lower,
        matrix_upper=matrix_upper,
        spectral_radius_lower=radius_lower,
        spectral_radius_upper=radius_upper,
        probability_supercritical=float(np.mean(radius_array > 1.0)),
        successful_replications=len(matrices),
        requested_replications=replications,
        confidence_level=confidence_level,
    )


def intervention_events_to_censored_rollout(
    events: Iterable[InterventionEvent],
    *,
    observation_end: float,
    requested_horizon: float | None = None,
    population_cap: int | None = None,
    population_cap_reached: bool = False,
    exposure_policy: Literal["observation_window", "active_interval"] = (
        "observation_window"
    ),
) -> CensoredBranchingRollout:
    """Adapt one simulator event namespace to the censored-forest schema.

    Only events descending from an exogenous primary intervention are retained.
    The registered Hawkes model treats an event as a point whose causal kernel
    remains observable until the rollout horizon, so ``observation_window`` is
    the primary policy. ``active_interval`` is an explicit competing model for
    contact-only influence and uses the logged intervention end. Keeping the
    choice named prevents two different reproduction numbers from being mixed.
    """

    end = float(observation_end)
    requested = end if requested_horizon is None else float(requested_horizon)
    if not np.isfinite(end) or end <= 0.0:
        raise ValueError("observation_end must be finite and positive")
    if not np.isfinite(requested) or requested < end:
        raise ValueError("requested_horizon must be finite and at least observation_end")
    if exposure_policy not in ("observation_window", "active_interval"):
        raise ValueError("unknown exposure_policy")
    records = tuple(
        sorted(
            (
                event
                for event in events
                if event.is_causal and event.start_time <= end + 1e-12
            ),
            key=lambda item: (item.start_time, item.event_id),
        )
    )
    identifiers = {event.event_id for event in records}
    timed: list[TimedBranchingEvent] = []
    for event in records:
        if event.root_primary_event_id is None or event.generation is None:
            raise ValueError("causal simulator events must carry root and generation")
        if event.parent_event_id is not None and event.parent_event_id not in identifiers:
            raise ValueError("simulator log is missing a direct causal parent")
        event_end = (
            None
            if exposure_policy == "observation_window"
            else min(end, max(float(event.start_time), float(event.end_time)))
        )
        timed.append(
            TimedBranchingEvent(
                event_id=int(event.event_id),
                event_type=int(event.event_type),
                birth_time=float(event.start_time),
                parent_event_id=(
                    None
                    if event.parent_event_id is None
                    else int(event.parent_event_id)
                ),
                root_event_id=int(event.root_primary_event_id),
                generation=int(event.generation),
                exposure_end_time=event_end,
            )
        )
    return CensoredBranchingRollout(
        events=tuple(timed),
        observation_end=end,
        requested_horizon=requested,
        population_cap=population_cap,
        population_cap_reached=population_cap_reached,
    )


def audit_branching_timestamps(
    rollouts: Iterable[CensoredBranchingRollout],
    time_step: float,
    *,
    tolerance: float = 1e-9,
) -> TimestampAudit:
    """Quantify simultaneous edges and grid alignment before likelihood choice."""

    records = tuple(rollouts)
    if not records:
        raise ValueError("at least one rollout is required")
    if not np.isfinite(time_step) or time_step <= 0.0:
        raise ValueError("time_step must be finite and positive")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    edge_count = 0
    zero_count = 0
    event_count = 0
    off_grid = 0
    max_grid_error = 0.0
    positive_lags: list[float] = []
    for rollout in records:
        lookup = {event.event_id: event for event in rollout.events}
        event_count += len(rollout.events)
        for event in rollout.events:
            grid_error = abs(
                event.birth_time - round(event.birth_time / time_step) * time_step
            )
            max_grid_error = max(max_grid_error, grid_error)
            if grid_error > tolerance:
                off_grid += 1
            if event.parent_event_id is None:
                continue
            edge_count += 1
            lag = event.birth_time - lookup[event.parent_event_id].birth_time
            if lag <= tolerance:
                zero_count += 1
            else:
                positive_lags.append(float(lag))
    positive = np.asarray(positive_lags, dtype=float)
    return TimestampAudit(
        rollout_count=len(records),
        event_count=event_count,
        edge_count=edge_count,
        zero_lag_edge_count=zero_count,
        zero_lag_fraction=(zero_count / edge_count if edge_count else 0.0),
        off_grid_birth_count=off_grid,
        maximum_birth_grid_error=float(max_grid_error),
        minimum_positive_lag=(float(np.min(positive)) if positive.size else None),
        median_positive_lag=(float(np.median(positive)) if positive.size else None),
        exact_time_likelihood_supported=(zero_count == 0),
    )


def _validate_recovery_bounds(bounds: tuple[float, float]) -> tuple[float, float]:
    lower, upper = map(float, bounds)
    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or lower <= 0.0
        or upper <= lower
    ):
        raise ValueError("recovery_rate_bounds must satisfy 0 < lower < upper")
    return lower, upper


def _joint_pair_profile(
    parent_ages: FloatArray,
    child_lags: FloatArray,
    recovery_rate: float,
    observation_model: ObservationModel,
    time_step: float | None,
) -> tuple[float, float, float]:
    """Return profile log likelihood, B-hat, and integrated exposure."""

    gamma = float(recovery_rate)
    if observation_model == "continuous":
        exposure_by_parent = -np.expm1(-gamma * parent_ages)
        child_log_mass = (
            child_lags.size * np.log(gamma) - gamma * float(np.sum(child_lags))
        )
    else:
        assert time_step is not None
        # The simulator records intervention starts at grid instants. A parent
        # active at a grid instant has one opportunity in bin zero, including
        # when a child receives the same timestamp. Exposure is therefore
        # right-inclusive in the number of sampled bins.
        bin_count = np.floor(parent_ages / time_step + 1e-9).astype(int) + 1
        exposure_by_parent = -np.expm1(-gamma * bin_count * time_step)
        lag_bins = np.rint(child_lags / time_step).astype(int)
        if np.any(lag_bins < 0):
            return float("-inf"), float("nan"), float("nan")
        log_first_bin = np.log(-np.expm1(-gamma * time_step))
        child_log_mass = (
            child_lags.size * log_first_bin
            - gamma * time_step * float(np.sum(lag_bins))
        )
    exposure = float(np.sum(exposure_by_parent))
    count = int(child_lags.size)
    if count == 0:
        return 0.0, 0.0, exposure
    if exposure <= 0.0 or not np.isfinite(exposure):
        return float("-inf"), float("nan"), exposure
    offspring = count / exposure
    value = count * np.log(offspring) + child_log_mass - offspring * exposure
    return float(value), float(offspring), exposure


def estimate_joint_censored_multitype_branching(
    rollouts: Iterable[CensoredBranchingRollout],
    number_of_types: int,
    *,
    observation_model: ObservationModel = "continuous",
    time_step: float | None = None,
    recovery_rate_bounds: tuple[float, float] = (0.05, 20.0),
) -> JointExposureBranchingEstimate:
    """Profile-likelihood fit of both ``B`` and exponential recovery rates.

    Entries are separable by parent/child type. For a fixed recovery rate the
    offspring mean retains the closed form ``N / exposure``; a bounded
    one-dimensional optimisation then profiles each recovery rate. Bounds are
    explicit because a zero-heavy lag histogram can otherwise drive the
    continuous-time likelihood to an infinite recovery rate.
    """

    records = tuple(rollouts)
    if not records:
        raise ValueError("at least one independent rollout is required")
    if number_of_types <= 0:
        raise ValueError("number_of_types must be positive")
    if observation_model not in ("continuous", "grouped"):
        raise ValueError("observation_model must be 'continuous' or 'grouped'")
    if observation_model == "grouped":
        if time_step is None or not np.isfinite(time_step) or time_step <= 0.0:
            raise ValueError("grouped observations require a positive time_step")
        step: float | None = float(time_step)
    else:
        if time_step is not None:
            raise ValueError("time_step is only valid for grouped observations")
        step = None
    lower, upper = _validate_recovery_bounds(recovery_rate_bounds)
    _validate_types(records, number_of_types)

    parent_ages: list[list[float]] = [[] for _ in range(number_of_types)]
    lags: list[list[list[float]]] = [
        [[] for _ in range(number_of_types)] for _ in range(number_of_types)
    ]
    event_count = 0
    edge_count = 0
    zero_lag_count = 0
    for rollout in records:
        lookup = {event.event_id: event for event in rollout.events}
        event_count += len(rollout.events)
        for event in rollout.events:
            age = max(0.0, _event_exposure_end(event, rollout) - event.birth_time)
            parent_ages[event.event_type].append(age)
            if event.parent_event_id is None:
                continue
            parent = lookup[event.parent_event_id]
            lag = float(event.birth_time - parent.birth_time)
            if lag < -1e-12:
                raise ValueError("child lag cannot be negative")
            lag = max(0.0, lag)
            lags[parent.event_type][event.event_type].append(lag)
            edge_count += 1
            zero_lag_count += int(lag <= 1e-12)

    counts = np.zeros((number_of_types, number_of_types), dtype=np.int64)
    exposure = np.zeros((number_of_types, number_of_types), dtype=float)
    matrix = np.zeros((number_of_types, number_of_types), dtype=float)
    recovery = np.full(
        (number_of_types, number_of_types), np.sqrt(lower * upper), dtype=float
    )
    identifiable = np.zeros((number_of_types, number_of_types), dtype=bool)
    at_bound = np.zeros((number_of_types, number_of_types), dtype=bool)
    total_log_likelihood = 0.0
    log_bounds = (float(np.log(lower)), float(np.log(upper)))

    for parent_type in range(number_of_types):
        ages = np.asarray(parent_ages[parent_type], dtype=float)
        for child_type in range(number_of_types):
            child_lags = np.asarray(lags[parent_type][child_type], dtype=float)
            counts[parent_type, child_type] = child_lags.size
            if child_lags.size == 0 or ages.size == 0:
                _, _, mass = _joint_pair_profile(
                    ages,
                    child_lags,
                    recovery[parent_type, child_type],
                    observation_model,
                    step,
                )
                exposure[parent_type, child_type] = mass
                continue

            def objective(log_rate: float) -> float:
                value, _, _ = _joint_pair_profile(
                    ages,
                    child_lags,
                    float(np.exp(log_rate)),
                    observation_model,
                    step,
                )
                return -value if np.isfinite(value) else np.inf

            fitted = minimize_scalar(
                objective,
                bounds=log_bounds,
                method="bounded",
                options={"xatol": 1e-10, "maxiter": 500},
            )
            if not fitted.success or not np.isfinite(fitted.fun):
                raise RuntimeError(
                    f"joint recovery optimisation failed for ({parent_type}, {child_type})"
                )
            gamma = float(np.exp(fitted.x))
            value, offspring, mass = _joint_pair_profile(
                ages, child_lags, gamma, observation_model, step
            )
            recovery[parent_type, child_type] = gamma
            matrix[parent_type, child_type] = offspring
            exposure[parent_type, child_type] = mass
            identifiable[parent_type, child_type] = True
            log_position = (fitted.x - log_bounds[0]) / (
                log_bounds[1] - log_bounds[0]
            )
            at_bound[parent_type, child_type] = (
                log_position <= 1e-3 or log_position >= 1.0 - 1e-3
            )
            total_log_likelihood += value

    return JointExposureBranchingEstimate(
        matrix=matrix,
        recovery_rates=recovery,
        direct_child_counts=counts,
        exposure_mass=exposure,
        parent_counts=np.asarray([len(item) for item in parent_ages], dtype=np.int64),
        identifiable_pairs=identifiable,
        recovery_at_bound=at_bound,
        spectral_radius=float(spectral_radius(matrix)),
        rollout_count=len(records),
        event_count=event_count,
        edge_count=edge_count,
        zero_lag_edge_count=zero_lag_count,
        log_likelihood=float(total_log_likelihood),
        observation_model=observation_model,
        time_step=step,
    )


def bootstrap_joint_censored_multitype_branching(
    rollouts: Iterable[CensoredBranchingRollout],
    number_of_types: int,
    *,
    observation_model: ObservationModel = "continuous",
    time_step: float | None = None,
    recovery_rate_bounds: tuple[float, float] = (0.05, 20.0),
    replications: int = 1_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> JointExposureBootstrapResult:
    """Cluster bootstrap for joint recovery and branching inference."""

    records = tuple(rollouts)
    if not records:
        raise ValueError("at least one independent rollout is required")
    if replications <= 0:
        raise ValueError("replications must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    keyword = dict(
        observation_model=observation_model,
        time_step=time_step,
        recovery_rate_bounds=recovery_rate_bounds,
    )
    point = estimate_joint_censored_multitype_branching(
        records, number_of_types, **keyword
    )
    rng = np.random.default_rng(seed)
    matrices: list[FloatArray] = []
    recoveries: list[FloatArray] = []
    radii: list[float] = []
    required = point.identifiable_pairs
    for _ in range(replications):
        indices = rng.integers(0, len(records), size=len(records))
        sample = tuple(records[int(index)] for index in indices)
        fitted = estimate_joint_censored_multitype_branching(
            sample, number_of_types, **keyword
        )
        if np.all(fitted.identifiable_pairs[required]):
            matrices.append(fitted.matrix)
            recoveries.append(fitted.recovery_rates)
            radii.append(fitted.spectral_radius)
    if not matrices:
        raise RuntimeError("no identifiable joint-bootstrap replication was obtained")
    alpha = (1.0 - confidence_level) / 2.0
    matrix_stack = np.stack(matrices)
    recovery_stack = np.stack(recoveries)
    radius_array = np.asarray(radii, dtype=float)
    return JointExposureBootstrapResult(
        point=point,
        matrix_lower=np.minimum(np.quantile(matrix_stack, alpha, axis=0), point.matrix),
        matrix_upper=np.maximum(
            np.quantile(matrix_stack, 1.0 - alpha, axis=0), point.matrix
        ),
        recovery_lower=np.minimum(
            np.quantile(recovery_stack, alpha, axis=0), point.recovery_rates
        ),
        recovery_upper=np.maximum(
            np.quantile(recovery_stack, 1.0 - alpha, axis=0), point.recovery_rates
        ),
        spectral_radius_lower=min(
            float(np.quantile(radius_array, alpha)), point.spectral_radius
        ),
        spectral_radius_upper=max(
            float(np.quantile(radius_array, 1.0 - alpha)), point.spectral_radius
        ),
        probability_supercritical=float(np.mean(radius_array > 1.0)),
        successful_replications=len(matrices),
        requested_replications=replications,
        confidence_level=confidence_level,
    )


def classify_branching_regime(
    spectral_radius_lower: float,
    spectral_radius_upper: float,
    *,
    threshold: float = 1.0,
) -> Literal["certified_subcritical", "indeterminate", "certified_supercritical"]:
    """Fail-closed three-way decision from a spectral-radius interval."""

    lower = float(spectral_radius_lower)
    upper = float(spectral_radius_upper)
    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or not np.isfinite(threshold)
        or threshold <= 0.0
        or lower > upper
    ):
        raise ValueError("invalid spectral-radius interval or threshold")
    if upper < threshold:
        return "certified_subcritical"
    if lower > threshold:
        return "certified_supercritical"
    return "indeterminate"


def scale_branching_matrix(matrix: ArrayLike, target_radius: float) -> FloatArray:
    """Scale a nonzero branching matrix to an exact target spectral radius."""

    candidate = _nonnegative_square("matrix", matrix)
    if not np.isfinite(target_radius) or target_radius <= 0.0:
        raise ValueError("target_radius must be finite and strictly positive")
    radius = float(spectral_radius(candidate))
    if radius <= 0.0:
        raise ValueError("matrix must have positive spectral radius")
    return candidate * (target_radius / radius)


def simulate_censored_branching_rollout(
    matrix: ArrayLike,
    recovery_rates: ArrayLike,
    root_types: Sequence[int],
    *,
    horizon: float,
    population_cap: int,
    rng: np.random.Generator,
) -> CensoredBranchingRollout:
    """Simulate a censored age-dependent multitype branching forest.

    Population-cap censoring is a chronological stopping rule.  Events scheduled
    after the stopping time are never exposed to the estimator.
    """

    offspring = _nonnegative_square("matrix", matrix)
    rates = _positive_square("recovery_rates", recovery_rates)
    if offspring.shape != rates.shape:
        raise ValueError("matrix and recovery_rates must have the same shape")
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be finite and strictly positive")
    roots = tuple(int(value) for value in root_types)
    if not roots:
        raise ValueError("at least one root is required")
    if any(value < 0 or value >= offspring.shape[0] for value in roots):
        raise ValueError("root type outside branching matrix")
    if population_cap < len(roots):
        raise ValueError("population_cap must be at least the number of roots")

    # Candidate tuple: birth, tie-breaker, type, parent id, root id, generation.
    queue: list[tuple[float, int, int, int | None, int | None, int]] = []
    tie_breaker = 0
    for root_type in roots:
        heapq.heappush(queue, (0.0, tie_breaker, root_type, None, None, 0))
        tie_breaker += 1

    events: list[TimedBranchingEvent] = []
    reached_cap = False
    observation_end = float(horizon)
    while queue:
        birth, _, event_type, parent_id, root_id, generation = heapq.heappop(queue)
        if birth > horizon:
            break
        event_id = len(events)
        actual_root = event_id if parent_id is None else int(root_id)
        event = TimedBranchingEvent(
            event_id=event_id,
            event_type=event_type,
            birth_time=float(birth),
            parent_event_id=parent_id,
            root_event_id=actual_root,
            generation=generation,
        )
        events.append(event)
        if len(events) >= population_cap:
            reached_cap = True
            observation_end = float(birth)
            break

        for child_type in range(offspring.shape[0]):
            count = int(rng.poisson(offspring[event_type, child_type]))
            if count == 0:
                continue
            delays = rng.exponential(
                scale=1.0 / rates[event_type, child_type], size=count
            )
            for delay in delays:
                child_birth = float(birth + delay)
                if child_birth <= horizon:
                    heapq.heappush(
                        queue,
                        (
                            child_birth,
                            tie_breaker,
                            child_type,
                            event_id,
                            actual_root,
                            generation + 1,
                        ),
                    )
                    tie_breaker += 1

    return CensoredBranchingRollout(
        events=tuple(events),
        observation_end=observation_end,
        requested_horizon=float(horizon),
        population_cap=population_cap,
        population_cap_reached=reached_cap,
    )


def simulate_censored_branching_experiment(
    matrix: ArrayLike,
    recovery_rates: ArrayLike,
    *,
    rollout_count: int,
    roots_per_rollout: int,
    horizon: float,
    population_cap: int,
    root_type_probabilities: ArrayLike | None = None,
    seed: int = 0,
) -> tuple[CensoredBranchingRollout, ...]:
    """Generate independent rollout clusters for estimator validation."""

    offspring = _nonnegative_square("matrix", matrix)
    rates = _positive_square("recovery_rates", recovery_rates)
    if offspring.shape != rates.shape:
        raise ValueError("matrix and recovery_rates must have the same shape")
    if rollout_count <= 0 or roots_per_rollout <= 0:
        raise ValueError("rollout_count and roots_per_rollout must be positive")
    if population_cap < roots_per_rollout:
        raise ValueError("population_cap must be at least roots_per_rollout")
    type_count = offspring.shape[0]
    if root_type_probabilities is None:
        probabilities = np.full(type_count, 1.0 / type_count)
    else:
        probabilities = np.asarray(root_type_probabilities, dtype=float).reshape(-1)
        if (
            probabilities.size != type_count
            or not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0.0)
            or probabilities.sum() <= 0.0
        ):
            raise ValueError("invalid root_type_probabilities")
        probabilities = probabilities / probabilities.sum()

    seed_sequence = np.random.SeedSequence(seed)
    generators = [np.random.default_rng(item) for item in seed_sequence.spawn(rollout_count)]
    result: list[CensoredBranchingRollout] = []
    for generator in generators:
        root_types = generator.choice(
            type_count, size=roots_per_rollout, p=probabilities
        ).astype(int)
        result.append(
            simulate_censored_branching_rollout(
                offspring,
                rates,
                root_types,
                horizon=horizon,
                population_cap=population_cap,
                rng=generator,
            )
        )
    return tuple(result)


def simulate_depleting_branching_rollout(
    matrix: ArrayLike,
    recovery_rates: ArrayLike,
    root_types: Sequence[int],
    *,
    horizon: float,
    population_by_type: Sequence[int],
    rng: np.random.Generator,
) -> CensoredBranchingRollout:
    """Simulate branching with a genuine finite susceptible population.

    Each accepted event consumes one previously unaffected individual of its
    type. Candidate births compete chronologically for the remaining pool.
    This is different from merely stopping an otherwise unbounded process after
    a fixed number of logged events and directly exercises depletion.
    """

    offspring = _nonnegative_square("matrix", matrix)
    rates = _positive_square("recovery_rates", recovery_rates)
    if offspring.shape != rates.shape:
        raise ValueError("matrix and recovery_rates must have the same shape")
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be finite and positive")
    population = np.asarray(tuple(population_by_type), dtype=int)
    if (
        population.ndim != 1
        or population.size != offspring.shape[0]
        or np.any(population < 0)
        or int(np.sum(population)) <= 0
    ):
        raise ValueError("population_by_type must be nonnegative with positive total")
    roots = tuple(int(item) for item in root_types)
    if not roots:
        raise ValueError("at least one root is required")
    if any(item < 0 or item >= offspring.shape[0] for item in roots):
        raise ValueError("root type outside branching matrix")
    root_counts = np.bincount(roots, minlength=offspring.shape[0])
    if np.any(root_counts > population):
        raise ValueError("roots exceed the susceptible population of a type")

    remaining = population - root_counts
    population_cap = int(np.sum(population))
    # Candidate tuple: birth, tie, type, parent id, root id, generation, is_root.
    queue: list[tuple[float, int, int, int | None, int | None, int, bool]] = []
    tie = 0
    for root_type in roots:
        heapq.heappush(queue, (0.0, tie, root_type, None, None, 0, True))
        tie += 1

    events: list[TimedBranchingEvent] = []
    observation_end = float(horizon)
    reached_cap = False
    while queue:
        birth, _, event_type, parent_id, root_id, generation, is_root = heapq.heappop(
            queue
        )
        if birth > horizon:
            break
        if not is_root:
            if remaining[event_type] <= 0:
                continue
            remaining[event_type] -= 1
        event_id = len(events)
        actual_root = event_id if parent_id is None else int(root_id)
        events.append(
            TimedBranchingEvent(
                event_id=event_id,
                event_type=event_type,
                birth_time=float(birth),
                parent_event_id=parent_id,
                root_event_id=actual_root,
                generation=generation,
            )
        )
        if len(events) == population_cap:
            reached_cap = True
            observation_end = float(birth)
            break
        for child_type in range(offspring.shape[0]):
            count = int(rng.poisson(offspring[event_type, child_type]))
            if count <= 0:
                continue
            delays = rng.exponential(
                scale=1.0 / rates[event_type, child_type], size=count
            )
            for delay in delays:
                child_birth = float(birth + delay)
                if child_birth <= horizon:
                    heapq.heappush(
                        queue,
                        (
                            child_birth,
                            tie,
                            child_type,
                            event_id,
                            actual_root,
                            generation + 1,
                            False,
                        ),
                    )
                    tie += 1
    return CensoredBranchingRollout(
        events=tuple(events),
        observation_end=observation_end,
        requested_horizon=float(horizon),
        population_cap=population_cap,
        population_cap_reached=reached_cap,
    )


def simulate_depleting_branching_experiment(
    matrix: ArrayLike,
    recovery_rates: ArrayLike,
    *,
    rollout_count: int,
    roots_per_rollout: int,
    horizon: float,
    population_by_type: Sequence[int],
    root_type_probabilities: ArrayLike | None = None,
    seed: int = 0,
) -> tuple[CensoredBranchingRollout, ...]:
    """Generate independent forests with finite susceptible populations."""

    offspring = _nonnegative_square("matrix", matrix)
    rates = _positive_square("recovery_rates", recovery_rates)
    if offspring.shape != rates.shape:
        raise ValueError("matrix and recovery_rates must have the same shape")
    if rollout_count <= 0 or roots_per_rollout <= 0:
        raise ValueError("rollout_count and roots_per_rollout must be positive")
    population = np.asarray(tuple(population_by_type), dtype=int)
    if population.size != offspring.shape[0] or np.any(population < 0):
        raise ValueError("population_by_type does not match matrix")
    if roots_per_rollout > int(np.sum(population)):
        raise ValueError("roots_per_rollout exceeds total population")
    if root_type_probabilities is None:
        probabilities = population.astype(float)
    else:
        probabilities = np.asarray(root_type_probabilities, dtype=float).reshape(-1)
    if (
        probabilities.size != offspring.shape[0]
        or np.any(probabilities < 0.0)
        or not np.all(np.isfinite(probabilities))
        or probabilities.sum() <= 0.0
    ):
        raise ValueError("invalid root_type_probabilities")
    probabilities = probabilities / probabilities.sum()

    seed_sequence = np.random.SeedSequence(seed)
    result: list[CensoredBranchingRollout] = []
    for child_seed in seed_sequence.spawn(rollout_count):
        generator = np.random.default_rng(child_seed)
        available = population.copy()
        root_types: list[int] = []
        for _ in range(roots_per_rollout):
            local = probabilities * (available > 0)
            if local.sum() <= 0.0:
                raise ValueError("not enough typed population for requested roots")
            local = local / local.sum()
            selected = int(generator.choice(offspring.shape[0], p=local))
            root_types.append(selected)
            available[selected] -= 1
        result.append(
            simulate_depleting_branching_rollout(
                offspring,
                rates,
                root_types,
                horizon=horizon,
                population_by_type=population,
                rng=generator,
            )
        )
    return tuple(result)


def poisson_extinction_probability(mean_offspring: float) -> float:
    """Smallest solution of ``q = exp(m(q-1))`` for Poisson offspring."""

    mean = float(mean_offspring)
    if not np.isfinite(mean) or mean < 0.0:
        raise ValueError("mean_offspring must be finite and nonnegative")
    if mean <= 1.0:
        return 1.0
    function = lambda q: np.exp(mean * (q - 1.0)) - q
    return float(brentq(function, 0.0, 1.0 - 1e-12))


def simulate_extinct_poisson_trees(
    mean_offspring: float,
    *,
    rollout_count: int,
    population_guard: int = 100_000,
    seed: int = 0,
) -> tuple[int, int, int]:
    """Return pooled children, parents, and retained extinct trees.

    Trees that reach ``population_guard`` are discarded rather than called
    extinct. With a large guard, the retained complete trees isolate the
    classical extinction-conditioning duality without temporal censoring.
    """

    mean = float(mean_offspring)
    if not np.isfinite(mean) or mean < 0.0:
        raise ValueError("mean_offspring must be finite and nonnegative")
    if rollout_count <= 0 or population_guard <= 1:
        raise ValueError("rollout_count and population_guard must be valid")
    rng = np.random.default_rng(seed)
    pooled_children = 0
    pooled_parents = 0
    extinct = 0
    for _ in range(rollout_count):
        pending = 1
        parents = 0
        children = 0
        while pending and parents < population_guard:
            pending -= 1
            offspring = int(rng.poisson(mean))
            pending += offspring
            parents += 1
            children += offspring
            if parents + pending >= population_guard:
                break
        if pending == 0:
            extinct += 1
            pooled_parents += parents
            pooled_children += children
    return pooled_children, pooled_parents, extinct
