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
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .capacity_theory import spectral_radius


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
            if event.parent_event_id is None:
                if event.generation != 0 or event.root_event_id != event.event_id:
                    raise ValueError("each root must have generation zero and identify itself")
                continue
            if event.parent_event_id not in lookup:
                raise ValueError("every non-root event must retain its observed direct parent")
            parent = lookup[event.parent_event_id]
            if parent.birth_time > event.birth_time + 1e-12:
                raise ValueError("a child cannot precede its parent")
            if event.generation != parent.generation + 1:
                raise ValueError("child generation must be parent generation plus one")
            if event.root_event_id != parent.root_event_id:
                raise ValueError("parent and child must belong to the same root tree")


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
            age = max(0.0, rollout.observation_end - event.birth_time)
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
