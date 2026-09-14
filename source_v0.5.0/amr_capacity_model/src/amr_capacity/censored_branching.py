"""Exposure-aware branching likelihood for right-censored event forests.

The estimator treats each causal intervention as an individual in a
continuous-time multitype branching process. While active, a type-a event
produces type-b children at rate beta[a, b] and resolves at rate delta[a].
Administrative censoring contributes observed active time and observed births,
but not a false resolution. The next-generation matrix is B[a, b] =
beta[a, b] / delta[a].

This model is intentionally separate from complete-tree direct-offspring
estimation. Complete trees remain useful for held-out progeny checks, while
this likelihood is the fail-closed route for estimating criticality when some
selected roots are unresolved at the simulation horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from .capacity_theory import spectral_radius
from .simulation import InterventionEvent


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class ObservedRootCohortSelection:
    """All observed descendants of roots selected in a fixed start-time window."""

    events: tuple[InterventionEvent, ...]
    selected_primary_roots: int
    complete_primary_roots: int
    unresolved_primary_roots: int
    observed_root_sizes: tuple[int, ...]
    observed_root_max_generations: tuple[int, ...]


@dataclass(frozen=True)
class ExposureAwareBranchingEstimate:
    """Maximum-likelihood estimate for a censored continuous-time forest."""

    matrix: FloatArray
    exposure_time: FloatArray
    observed_parent_counts: IntArray
    completed_parent_counts: IntArray
    censored_parent_counts: IntArray
    direct_child_counts: IntArray
    birth_rate_matrix: FloatArray
    recovery_rates: FloatArray
    identified_rows: BoolArray
    spectral_radius: float | None
    log_likelihood_up_to_constant: float
    status: str


@dataclass(frozen=True)
class ExposureAwareBranchingBootstrap:
    """Percentile interval from resampling independent event forests."""

    point_estimate: ExposureAwareBranchingEstimate
    confidence_level: float
    requested_resamples: int
    valid_resamples: int
    matrix_lower: FloatArray
    matrix_upper: FloatArray
    spectral_radius_lower: float
    spectral_radius_upper: float


@dataclass(frozen=True)
class _ExposureStatistics:
    exposure_time: FloatArray
    observed_parent_counts: IntArray
    completed_parent_counts: IntArray
    censored_parent_counts: IntArray
    direct_child_counts: IntArray


def _validate_number_of_types(number_of_types: int) -> int:
    if (
        isinstance(number_of_types, (bool, np.bool_))
        or not isinstance(number_of_types, (int, np.integer))
        or number_of_types < 1
    ):
        raise ValueError("number_of_types must be a positive integer")
    return int(number_of_types)


def _validate_event_type(event: InterventionEvent, number_of_types: int) -> int:
    value = event.event_type
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not 0 <= int(value) < number_of_types
    ):
        raise ValueError("event_type lies outside number_of_types")
    return int(value)


def _require_unique_event_ids(records: tuple[InterventionEvent, ...]) -> None:
    identifiers = [event.event_id for event in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(
            "duplicate event_id values detected; event IDs are rollout-local, "
            "so pool independent event sets instead of concatenating them"
        )


def select_observed_root_cohorts(
    events: Iterable[InterventionEvent],
    root_start: float,
    root_end: float,
) -> ObservedRootCohortSelection:
    """Select roots by start time and retain complete and censored descendants.

    Unlike select_complete_root_cohorts, this function never drops an entire
    selected tree merely because an event is active at the observation horizon.
    Every non-root record must still have its proximate parent in the selected
    cohort, preventing left-truncated likelihood contributions.
    """

    if (
        not np.isfinite(root_start)
        or not np.isfinite(root_end)
        or root_start < 0.0
        or root_end <= root_start
    ):
        raise ValueError("root cohort bounds must be finite with 0 <= start < end")
    records = tuple(events)
    _require_unique_event_ids(records)
    causal = tuple(event for event in records if event.is_causal)
    roots = {
        event.event_id
        for event in causal
        if event.is_primary and root_start <= event.start_time < root_end
    }
    selected = tuple(
        event for event in causal if event.root_primary_event_id in roots
    )
    grouped: dict[int, list[InterventionEvent]] = {
        root_id: [] for root_id in roots
    }
    for event in selected:
        assert event.root_primary_event_id is not None
        grouped[event.root_primary_event_id].append(event)

    complete = 0
    unresolved = 0
    sizes: list[int] = []
    generations: list[int] = []
    for root_id, group in grouped.items():
        root_records = [event for event in group if event.event_id == root_id]
        if len(root_records) != 1 or not root_records[0].is_primary:
            raise ValueError("each selected cohort must contain its primary root")
        identifiers = {event.event_id for event in group}
        for event in group:
            if event.event_id == root_id:
                continue
            if event.parent_event_id not in identifiers:
                raise ValueError("selected cohort is missing a proximate parent")
        if any(event.censored for event in group):
            unresolved += 1
        else:
            complete += 1
        sizes.append(len(group))
        generations.append(
            max(
                (
                    int(event.generation)
                    for event in group
                    if event.generation is not None
                ),
                default=0,
            )
        )
    return ObservedRootCohortSelection(
        events=selected,
        selected_primary_roots=len(roots),
        complete_primary_roots=complete,
        unresolved_primary_roots=unresolved,
        observed_root_sizes=tuple(sizes),
        observed_root_max_generations=tuple(generations),
    )


def _event_statistics(
    events: Iterable[InterventionEvent],
    number_of_types: int,
) -> _ExposureStatistics:
    records = tuple(events)
    _require_unique_event_ids(records)
    causal = tuple(event for event in records if event.is_causal)
    by_id = {event.event_id: event for event in causal}
    exposure = np.zeros(number_of_types, dtype=float)
    observed = np.zeros(number_of_types, dtype=int)
    completed = np.zeros(number_of_types, dtype=int)
    censored = np.zeros(number_of_types, dtype=int)
    children = np.zeros((number_of_types, number_of_types), dtype=int)

    for event in causal:
        event_type = _validate_event_type(event, number_of_types)
        duration = float(event.end_time - event.start_time)
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("causal event exposure must be finite and positive")
        exposure[event_type] += duration
        observed[event_type] += 1
        if event.censored:
            censored[event_type] += 1
        else:
            completed[event_type] += 1

    for child in causal:
        child_type = _validate_event_type(child, number_of_types)
        if child.is_primary:
            if child.parent_event_id is not None:
                raise ValueError("a primary event cannot have a proximate parent")
            continue
        if child.parent_event_id not in by_id:
            raise ValueError("causal event is missing its proximate parent")
        assert child.parent_event_id is not None
        parent = by_id[child.parent_event_id]
        tolerance = 1e-9 * max(
            1.0,
            abs(parent.start_time),
            abs(parent.end_time),
            abs(child.start_time),
        )
        if (
            child.start_time < parent.start_time - tolerance
            or child.start_time > parent.end_time + tolerance
        ):
            raise ValueError(
                "child start_time must lie inside its parent's observed exposure"
            )
        parent_type = _validate_event_type(parent, number_of_types)
        children[parent_type, child_type] += 1

    return _ExposureStatistics(
        exposure_time=exposure,
        observed_parent_counts=observed,
        completed_parent_counts=completed,
        censored_parent_counts=censored,
        direct_child_counts=children,
    )


def _sum_statistics(
    statistics: Iterable[_ExposureStatistics],
    number_of_types: int,
) -> _ExposureStatistics:
    exposure = np.zeros(number_of_types, dtype=float)
    observed = np.zeros(number_of_types, dtype=int)
    completed = np.zeros(number_of_types, dtype=int)
    censored = np.zeros(number_of_types, dtype=int)
    children = np.zeros((number_of_types, number_of_types), dtype=int)
    count = 0
    for item in statistics:
        exposure += item.exposure_time
        observed += item.observed_parent_counts
        completed += item.completed_parent_counts
        censored += item.censored_parent_counts
        children += item.direct_child_counts
        count += 1
    if count == 0:
        raise ValueError("at least one event set is required")
    return _ExposureStatistics(exposure, observed, completed, censored, children)


def _poisson_log_likelihood(
    counts: FloatArray,
    rates: FloatArray,
    exposures: FloatArray,
) -> float:
    expected = rates * exposures
    value = -float(np.sum(expected))
    positive = counts > 0.0
    if np.any(positive):
        if np.any(rates[positive] <= 0.0):
            return -np.inf
        value += float(np.sum(counts[positive] * np.log(rates[positive])))
    return value


def _estimate_from_statistics(
    statistics: _ExposureStatistics,
) -> ExposureAwareBranchingEstimate:
    exposure = statistics.exposure_time
    if float(np.sum(exposure)) <= 0.0:
        raise ValueError("event sets contain no causal parent exposure")

    birth_rates = np.divide(
        statistics.direct_child_counts,
        exposure[:, None],
        out=np.zeros_like(statistics.direct_child_counts, dtype=float),
        where=exposure[:, None] > 0.0,
    )
    recovery_rates = np.divide(
        statistics.completed_parent_counts,
        exposure,
        out=np.zeros_like(exposure, dtype=float),
        where=exposure > 0.0,
    )
    matrix = np.zeros_like(birth_rates)
    completed_rows = statistics.completed_parent_counts > 0
    matrix[completed_rows] = (
        statistics.direct_child_counts[completed_rows]
        / statistics.completed_parent_counts[completed_rows, None]
    )
    unidentified_active = (statistics.observed_parent_counts > 0) & ~completed_rows
    matrix[unidentified_active] = np.nan
    rho = (
        None
        if np.any(unidentified_active)
        else spectral_radius(np.nan_to_num(matrix, nan=0.0))
    )

    birth_log_likelihood = _poisson_log_likelihood(
        statistics.direct_child_counts.astype(float),
        birth_rates,
        exposure[:, None],
    )
    recovery_log_likelihood = _poisson_log_likelihood(
        statistics.completed_parent_counts.astype(float),
        recovery_rates,
        exposure,
    )
    return ExposureAwareBranchingEstimate(
        matrix=matrix,
        exposure_time=exposure.copy(),
        observed_parent_counts=statistics.observed_parent_counts.copy(),
        completed_parent_counts=statistics.completed_parent_counts.copy(),
        censored_parent_counts=statistics.censored_parent_counts.copy(),
        direct_child_counts=statistics.direct_child_counts.copy(),
        birth_rate_matrix=birth_rates,
        recovery_rates=recovery_rates,
        identified_rows=completed_rows.copy(),
        spectral_radius=None if rho is None else float(rho),
        log_likelihood_up_to_constant=(
            birth_log_likelihood + recovery_log_likelihood
        ),
        status=(
            "identified"
            if rho is not None
            else "unidentified_active_type_without_resolution"
        ),
    )


def estimate_exposure_aware_branching(
    events: Iterable[InterventionEvent],
    *,
    number_of_types: int = 1,
) -> ExposureAwareBranchingEstimate:
    """Fit one event forest without treating horizon censoring as recovery."""

    count = _validate_number_of_types(number_of_types)
    statistics = _event_statistics(events, count)
    return _estimate_from_statistics(statistics)


def pool_exposure_aware_branching_events(
    event_sets: Iterable[Iterable[InterventionEvent]],
    *,
    number_of_types: int = 1,
) -> ExposureAwareBranchingEstimate:
    """Pool likelihood sufficient statistics across rollout-local namespaces."""

    count = _validate_number_of_types(number_of_types)
    sets = tuple(tuple(events) for events in event_sets)
    statistics = _sum_statistics(
        (_event_statistics(events, count) for events in sets),
        count,
    )
    return _estimate_from_statistics(statistics)


def bootstrap_exposure_aware_branching(
    event_sets: Iterable[Iterable[InterventionEvent]],
    *,
    number_of_types: int = 1,
    confidence_level: float = 0.95,
    resamples: int = 1_000,
    seed: int | None = None,
) -> ExposureAwareBranchingBootstrap:
    """Bootstrap whole independent forests, preserving within-tree dependence."""

    count = _validate_number_of_types(number_of_types)
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if (
        isinstance(resamples, (bool, np.bool_))
        or not isinstance(resamples, (int, np.integer))
        or resamples < 1
    ):
        raise ValueError("resamples must be a positive integer")
    sets = tuple(tuple(events) for events in event_sets)
    if len(sets) < 2:
        raise ValueError("bootstrap requires at least two independent event sets")
    per_set = tuple(_event_statistics(events, count) for events in sets)
    point = _estimate_from_statistics(_sum_statistics(per_set, count))
    if point.spectral_radius is None:
        raise ValueError("point estimate has an unidentified active parent type")

    rng = np.random.default_rng(seed)
    matrices: list[FloatArray] = []
    radii: list[float] = []
    active = point.observed_parent_counts > 0
    for _ in range(int(resamples)):
        indices = rng.integers(0, len(per_set), size=len(per_set))
        sampled = _sum_statistics((per_set[int(index)] for index in indices), count)
        estimate = _estimate_from_statistics(sampled)
        if np.any(active & ~estimate.identified_rows):
            continue
        if estimate.spectral_radius is None:
            continue
        matrices.append(estimate.matrix)
        radii.append(estimate.spectral_radius)
    if not matrices:
        raise ValueError("no bootstrap resample identified every active parent type")

    alpha = (1.0 - confidence_level) / 2.0
    stacked = np.asarray(matrices, dtype=float)
    radius_values = np.asarray(radii, dtype=float)
    return ExposureAwareBranchingBootstrap(
        point_estimate=point,
        confidence_level=float(confidence_level),
        requested_resamples=int(resamples),
        valid_resamples=len(matrices),
        matrix_lower=np.quantile(stacked, alpha, axis=0),
        matrix_upper=np.quantile(stacked, 1.0 - alpha, axis=0),
        spectral_radius_lower=float(np.quantile(radius_values, alpha)),
        spectral_radius_upper=float(np.quantile(radius_values, 1.0 - alpha)),
    )
