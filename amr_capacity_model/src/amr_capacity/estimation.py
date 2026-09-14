"""Causal and correlational cascade estimators for simulator event logs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Protocol

import numpy as np
from numpy.typing import NDArray

from .capacity_theory import (
    cascade_budget_direct,
    mean_cascade_loss,
    spectral_radius,
    truncated_cascade_loss,
)
from .simulation import InterventionEvent, PairedRolloutResult


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class BranchingEstimate:
    """Generation-level direct-offspring estimate."""

    matrix: FloatArray
    parent_counts: IntArray
    direct_child_counts: IntArray
    spectral_radius: float


@dataclass(frozen=True)
class CorridorCascadeMetrics:
    """Scalar statistics extracted from one paired corridor rollout."""

    primary_events: int
    causal_events: int
    control_traversals: float
    primary_events_per_traversal: float
    mean_event_severity_loss: float
    g_per_traversal: float
    direct_branching: BranchingEstimate
    measured_total_progeny_per_primary: float
    predicted_total_progeny_per_primary: float | None
    paired_severity_loss: float
    paired_traversal_loss: float


@dataclass(frozen=True)
class HeldOutCascadeValidation:
    """Prediction fitted on training rollouts and scored on separate rollouts."""

    training_branching: BranchingEstimate
    training_primary_distribution: FloatArray
    validation_primary_events: int
    validation_causal_events: int
    predicted_total_progeny_per_primary: float | None
    measured_total_progeny_per_primary: float
    relative_error: float | None
    max_generation: int | None
    truncated_predicted_total_progeny_per_primary: float | None
    truncated_relative_error: float | None


@dataclass(frozen=True)
class ChainForestDiagnostic:
    """Detect when scalar in-sample resolvent agreement is an identity.

    In a completely observed single-parent forest with ``E`` causal events and
    ``P`` primary roots, there are ``E-P`` direct edges. If all causal events
    are eligible parents, ``B=(E-P)/E`` and ``1/(1-B)=E/P`` by construction.
    That equality is bookkeeping, not predictive validation.
    """

    causal_events: int
    primary_events: int
    eligible_parent_events: int
    direct_edges: int
    rooted_single_parent_forest: bool
    measured_progeny_per_primary: float | None
    in_sample_resolvent: float | None
    identity_residual: float | None
    algebraic_identity: bool


@dataclass(frozen=True)
class BranchingForestDiagnostic:
    """Structural properties of one causally complete event forest."""

    causal_events: int
    primary_events: int
    direct_edges: int
    rooted_single_parent_forest: bool
    fanout_parent_events: int
    maximum_direct_children: int
    maximum_generation: int
    repeated_robot_events_within_root: int


@dataclass(frozen=True)
class MultitypeForestIdentityDiagnostic:
    """Detect the primary-mix resolvent identity in complete typed forests."""

    causal_events: int
    primary_events: int
    training_mean_progeny: float | None
    primary_mix_resolvent: float | None
    identity_residual: float | None
    algebraic_identity: bool
    training_spectral_radius: float | None
    complete_forest_rho_bound_verified: bool


@dataclass(frozen=True)
class RootCohortSelection:
    """Complete primary-root cohorts selected from a fixed observation window."""

    events: tuple[InterventionEvent, ...]
    selected_primary_roots: int
    complete_primary_roots: int
    unresolved_primary_roots: int
    complete_root_sizes: tuple[int, ...]
    complete_root_max_generations: tuple[int, ...]


@dataclass(frozen=True)
class FiniteChainEstimate:
    """Right-censoring-aware propagation estimate for a finite robot chain."""

    fleet_size: int
    propagation_successes: int
    observed_failures: int
    propagation_probability: float
    resolved_cascades: int
    fleet_saturated_cascades: int
    indeterminate_cascades: int
    algebraic_boundary_identity: bool
    validation_status: str


@dataclass(frozen=True)
class FiniteChainValidation:
    """Held-out validation of the depth-truncated scalar chain model."""

    training: FiniteChainEstimate
    validation_resolved_cascades: int
    validation_fleet_saturated_cascades: int
    validation_indeterminate_cascades: int
    measured_unique_robots_per_primary: float
    infinite_prediction: float | None
    infinite_relative_error: float | None
    finite_prediction: float
    finite_relative_error: float
    validation_status: str


class _EventResult(Protocol):
    events: tuple[InterventionEvent, ...]


class PairedEventRollout(Protocol):
    """Structural type shared by closed and open paired-rollout results."""

    treated: _EventResult


def _require_unique_event_ids(
    records: tuple[InterventionEvent, ...],
) -> None:
    identifiers = [event.event_id for event in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(
            "duplicate event_id values detected; event IDs are rollout-local, "
            "so use pool_direct_branching_events for independent logs instead "
            "of concatenating them"
        )


def select_complete_root_cohorts(
    events: Iterable[InterventionEvent],
    root_start: float,
    root_end: float,
) -> RootCohortSelection:
    """Select complete causal trees by primary-root time, without left truncation.

    Descendants are retained regardless of their own start times. A selected
    root is unresolved when any event in its tree is temporally censored.
    Event IDs remain rollout-local; selections from several rollouts must be
    passed separately to :func:`pool_direct_branching_events`.
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
        event.event_id: event
        for event in causal
        if event.is_primary and root_start <= event.start_time < root_end
    }
    grouped: dict[int, list[InterventionEvent]] = {
        root_id: [] for root_id in roots
    }
    for event in causal:
        if event.root_primary_event_id in grouped:
            assert event.root_primary_event_id is not None
            grouped[event.root_primary_event_id].append(event)

    complete_events: list[InterventionEvent] = []
    root_sizes: list[int] = []
    root_generations: list[int] = []
    unresolved = 0
    for root_id, group in grouped.items():
        root_records = [event for event in group if event.event_id == root_id]
        if len(root_records) != 1 or not root_records[0].is_primary:
            raise ValueError("each selected cohort must contain its primary root")
        if any(event.censored for event in group):
            unresolved += 1
            continue
        identifiers = {event.event_id for event in group}
        for event in group:
            if event.event_id == root_id:
                continue
            if event.parent_event_id not in identifiers:
                raise ValueError("selected cohort is missing a proximate parent")
        complete_events.extend(group)
        root_sizes.append(len(group))
        root_generations.append(
            max(
                (
                    int(event.generation)
                    for event in group
                    if event.generation is not None
                ),
                default=0,
            )
        )
    return RootCohortSelection(
        events=tuple(complete_events),
        selected_primary_roots=len(grouped),
        complete_primary_roots=len(grouped) - unresolved,
        unresolved_primary_roots=unresolved,
        complete_root_sizes=tuple(root_sizes),
        complete_root_max_generations=tuple(root_generations),
    )


def retype_events_by_minimum_speed(
    events: Iterable[InterventionEvent],
    *,
    stop_threshold: float = 1e-3,
) -> tuple[InterventionEvent, ...]:
    """Assign type 0 to protective stops and type 1 to reduced-speed events."""

    if not np.isfinite(stop_threshold) or stop_threshold < 0.0:
        raise ValueError("stop_threshold must be finite and nonnegative")
    records = tuple(events)
    if any(not np.isfinite(event.minimum_speed) for event in records):
        raise ValueError("minimum_speed must be finite for severity typing")
    return tuple(
        replace(
            event,
            event_type=0 if event.minimum_speed <= stop_threshold else 1,
        )
        for event in records
    )


def retype_events_by_position(
    events: Iterable[InterventionEvent],
    region_edges: Iterable[float],
) -> tuple[InterventionEvent, ...]:
    """Assign event types from their start position in fixed spatial regions."""

    edges = np.asarray(tuple(region_edges), dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("region_edges must contain at least two values")
    if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
        raise ValueError("region_edges must be finite and strictly increasing")
    records = tuple(events)
    typed: list[InterventionEvent] = []
    for event in records:
        if event.start_position is None or not np.isfinite(event.start_position):
            raise ValueError("start_position must be logged for spatial typing")
        position = float(event.start_position)
        tolerance = 1e-9 * max(1.0, abs(edges[0]), abs(edges[-1]))
        if position < edges[0] - tolerance or position > edges[-1] + tolerance:
            raise ValueError("event start_position lies outside region_edges")
        clipped = min(max(position, float(edges[0])), float(edges[-1]))
        event_type = int(np.searchsorted(edges[1:-1], clipped, side="right"))
        typed.append(replace(event, event_type=event_type))
    return tuple(typed)


def estimate_direct_branching(
    events: Iterable[InterventionEvent],
    *,
    number_of_types: int = 1,
) -> BranchingEstimate:
    """Estimate one-generation offspring using logged proximate parents only.

    Censored parent events are omitted because their offspring window is not
    complete. Only events descending from an exogenous primary are used.
    """

    if number_of_types < 1:
        raise ValueError("number_of_types must be at least one")
    records = tuple(events)
    _require_unique_event_ids(records)
    by_id = {event.event_id: event for event in records}
    eligible_parents = {
        event.event_id
        for event in records
        if event.is_causal and not event.censored
    }
    parent_counts = np.zeros(number_of_types, dtype=int)
    child_counts = np.zeros((number_of_types, number_of_types), dtype=int)
    for parent_id in eligible_parents:
        parent = by_id[parent_id]
        if not 0 <= parent.event_type < number_of_types:
            raise ValueError("event_type lies outside number_of_types")
        parent_counts[parent.event_type] += 1
    for child in records:
        if child.parent_event_id not in eligible_parents:
            continue
        parent = by_id[child.parent_event_id]
        if not 0 <= child.event_type < number_of_types:
            raise ValueError("event_type lies outside number_of_types")
        child_counts[parent.event_type, child.event_type] += 1
    matrix = np.divide(
        child_counts,
        parent_counts[:, None],
        out=np.zeros_like(child_counts, dtype=float),
        where=parent_counts[:, None] > 0,
    )
    return BranchingEstimate(
        matrix=matrix,
        parent_counts=parent_counts,
        direct_child_counts=child_counts,
        spectral_radius=spectral_radius(matrix),
    )


def estimate_horizon_branching(
    events: Iterable[InterventionEvent],
    horizon: float,
    *,
    number_of_types: int = 1,
) -> BranchingEstimate:
    """Correlational horizon estimator retained only as an ablation baseline.

    Every later causal event on another robot inside ``horizon`` is counted as
    a child. This intentionally mixes generations and can double-count the same
    event under several parents.
    """

    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be finite and strictly positive")
    if number_of_types < 1:
        raise ValueError("number_of_types must be at least one")
    records = tuple(event for event in events if event.is_causal)
    parents = tuple(event for event in records if not event.censored)
    parent_counts = np.zeros(number_of_types, dtype=int)
    child_counts = np.zeros((number_of_types, number_of_types), dtype=int)
    for parent in parents:
        parent_counts[parent.event_type] += 1
        for candidate in records:
            if candidate.robot_id == parent.robot_id:
                continue
            if parent.start_time < candidate.start_time <= parent.start_time + horizon:
                child_counts[parent.event_type, candidate.event_type] += 1
    matrix = np.divide(
        child_counts,
        parent_counts[:, None],
        out=np.zeros_like(child_counts, dtype=float),
        where=parent_counts[:, None] > 0,
    )
    return BranchingEstimate(
        matrix=matrix,
        parent_counts=parent_counts,
        direct_child_counts=child_counts,
        spectral_radius=spectral_radius(matrix),
    )


def diagnose_chain_forest(
    events: Iterable[InterventionEvent],
    *,
    tolerance: float = 1e-12,
) -> ChainForestDiagnostic:
    """Report whether an in-sample progeny comparison is algebraic.

    The sufficient statistics match :func:`estimate_direct_branching`, so the
    residual also exposes departures caused by horizon censoring or by a
    window filter that excludes a parent event.
    """

    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    records = tuple(events)
    causal = tuple(event for event in records if event.is_causal)
    primaries = tuple(event for event in causal if event.is_primary)
    causal_ids = {event.event_id for event in causal}
    nonroots = tuple(event for event in causal if not event.is_primary)
    rooted_forest = all(
        event.parent_event_id is not None
        and event.parent_event_id in causal_ids
        for event in nonroots
    )
    estimate = estimate_direct_branching(records)
    eligible = int(estimate.parent_counts[0])
    edges = int(estimate.direct_child_counts[0, 0])
    measured = len(causal) / len(primaries) if primaries else None
    coefficient = float(estimate.matrix[0, 0])
    resolvent = 1.0 / (1.0 - coefficient) if coefficient < 1.0 else None
    residual = (
        resolvent - measured
        if resolvent is not None and measured is not None
        else None
    )
    complete_identity_structure = (
        rooted_forest
        and len(causal) > 0
        and eligible == len(causal)
        and edges == len(causal) - len(primaries)
    )
    algebraic = (
        complete_identity_structure
        and residual is not None
        and abs(residual) <= tolerance
    )
    return ChainForestDiagnostic(
        causal_events=len(causal),
        primary_events=len(primaries),
        eligible_parent_events=eligible,
        direct_edges=edges,
        rooted_single_parent_forest=rooted_forest,
        measured_progeny_per_primary=measured,
        in_sample_resolvent=resolvent,
        identity_residual=residual,
        algebraic_identity=algebraic,
    )


def diagnose_branching_forest(
    events: Iterable[InterventionEvent],
) -> BranchingForestDiagnostic:
    """Measure fan-out, depth, and repeated-robot structure in one event log."""

    records = tuple(events)
    _require_unique_event_ids(records)
    causal = tuple(event for event in records if event.is_causal)
    identifiers = {event.event_id for event in causal}
    primaries = tuple(event for event in causal if event.is_primary)
    child_counts = {event.event_id: 0 for event in causal}
    rooted = True
    for event in causal:
        if event.is_primary:
            continue
        if event.parent_event_id not in identifiers:
            rooted = False
            continue
        assert event.parent_event_id is not None
        child_counts[event.parent_event_id] += 1
    fanout = sum(count > 1 for count in child_counts.values())
    maximum_children = max(child_counts.values(), default=0)
    maximum_generation = max(
        (
            int(event.generation)
            for event in causal
            if event.generation is not None
        ),
        default=0,
    )
    robots_by_root: dict[int, list[int]] = {}
    for event in causal:
        assert event.root_primary_event_id is not None
        robots_by_root.setdefault(event.root_primary_event_id, []).append(
            event.robot_id
        )
    repeated = sum(
        len(robot_ids) - len(set(robot_ids))
        for robot_ids in robots_by_root.values()
    )
    return BranchingForestDiagnostic(
        causal_events=len(causal),
        primary_events=len(primaries),
        direct_edges=sum(child_counts.values()),
        rooted_single_parent_forest=rooted,
        fanout_parent_events=fanout,
        maximum_direct_children=maximum_children,
        maximum_generation=maximum_generation,
        repeated_robot_events_within_root=repeated,
    )


def diagnose_multitype_forest_identity(
    event_sets: Iterable[Iterable[InterventionEvent]],
    *,
    number_of_types: int,
    tolerance: float = 1e-12,
) -> MultitypeForestIdentityDiagnostic:
    """Detect when a typed primary-mix resolvent restates training progeny.

    For complete single-parent forests, let ``n`` count events by type and
    ``r`` count primary roots. Direct-edge conservation gives
    ``n.T @ B = n.T - r.T``. Therefore

    ``(r/P).T @ (I-B)^-1 @ 1 = E/P``.

    The equality holds for any event typing, including non-normal matrices. It
    validates bookkeeping only; prediction still requires independent trees.
    """

    if number_of_types < 1:
        raise ValueError("number_of_types must be at least one")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    sets = tuple(tuple(events) for events in event_sets)
    if not sets:
        raise ValueError("at least one event set is required")
    causal_events = 0
    primary_counts = np.zeros(number_of_types, dtype=float)
    complete_structure = True
    for records in sets:
        structure = diagnose_branching_forest(records)
        causal = tuple(event for event in records if event.is_causal)
        causal_events += len(causal)
        complete_structure = bool(
            complete_structure
            and structure.rooted_single_parent_forest
            and structure.direct_edges
            == structure.causal_events - structure.primary_events
            and not any(event.censored for event in causal)
        )
        for event in causal:
            if event.is_primary:
                if not 0 <= event.event_type < number_of_types:
                    raise ValueError("event_type lies outside number_of_types")
                primary_counts[event.event_type] += 1.0
    primary_events = int(np.sum(primary_counts))
    if primary_events == 0:
        return MultitypeForestIdentityDiagnostic(
            causal_events=causal_events,
            primary_events=0,
            training_mean_progeny=None,
            primary_mix_resolvent=None,
            identity_residual=None,
            algebraic_identity=False,
            training_spectral_radius=None,
            complete_forest_rho_bound_verified=False,
        )
    estimate = pool_direct_branching_events(
        sets, number_of_types=number_of_types
    )
    rho_bound_verified = bool(
        complete_structure and estimate.spectral_radius <= 1.0 + tolerance
    )
    if complete_structure and not rho_bound_verified:
        raise AssertionError(
            "a complete single-parent forest cannot have spectral radius above one"
        )
    measured = causal_events / primary_events
    predicted: float | None = None
    if estimate.spectral_radius < 1.0:
        predicted = mean_cascade_loss(
            estimate.matrix,
            primary_counts / primary_events,
            np.ones(number_of_types),
        )
    residual = predicted - measured if predicted is not None else None
    algebraic = bool(
        complete_structure
        and residual is not None
        and abs(residual) <= tolerance
    )
    return MultitypeForestIdentityDiagnostic(
        causal_events=causal_events,
        primary_events=primary_events,
        training_mean_progeny=measured,
        primary_mix_resolvent=predicted,
        identity_residual=residual,
        algebraic_identity=algebraic,
        training_spectral_radius=estimate.spectral_radius,
        complete_forest_rho_bound_verified=rho_bound_verified,
    )


def pool_direct_branching_events(
    event_sets: Iterable[Iterable[InterventionEvent]],
    *,
    number_of_types: int = 1,
) -> BranchingEstimate:
    """Pool generation-level sufficient statistics across independent logs."""

    if number_of_types < 1:
        raise ValueError("number_of_types must be at least one")
    parent_counts = np.zeros(number_of_types, dtype=int)
    child_counts = np.zeros((number_of_types, number_of_types), dtype=int)
    observed = 0
    for events in event_sets:
        estimate = estimate_direct_branching(
            events, number_of_types=number_of_types
        )
        parent_counts += estimate.parent_counts
        child_counts += estimate.direct_child_counts
        observed += 1
    if observed == 0:
        raise ValueError("at least one rollout is required")
    matrix = np.divide(
        child_counts,
        parent_counts[:, None],
        out=np.zeros_like(child_counts, dtype=float),
        where=parent_counts[:, None] > 0,
    )
    return BranchingEstimate(
        matrix=matrix,
        parent_counts=parent_counts,
        direct_child_counts=child_counts,
        spectral_radius=spectral_radius(matrix),
    )


def pool_direct_branching(
    rollouts: Iterable[PairedEventRollout],
    *,
    number_of_types: int = 1,
) -> BranchingEstimate:
    """Pool direct-offspring statistics across closed or open paired runs."""

    return pool_direct_branching_events(
        (pair.treated.events for pair in rollouts),
        number_of_types=number_of_types,
    )


def pool_horizon_branching(
    rollouts: Iterable[PairedRolloutResult],
    horizon: float,
    *,
    number_of_types: int = 1,
) -> BranchingEstimate:
    """Pool the intentionally correlational horizon-ablation estimator."""

    if number_of_types < 1:
        raise ValueError("number_of_types must be at least one")
    parent_counts = np.zeros(number_of_types, dtype=int)
    child_counts = np.zeros((number_of_types, number_of_types), dtype=int)
    observed = 0
    for pair in rollouts:
        estimate = estimate_horizon_branching(
            pair.treated.events,
            horizon,
            number_of_types=number_of_types,
        )
        parent_counts += estimate.parent_counts
        child_counts += estimate.direct_child_counts
        observed += 1
    if observed == 0:
        raise ValueError("at least one rollout is required")
    matrix = np.divide(
        child_counts,
        parent_counts[:, None],
        out=np.zeros_like(child_counts, dtype=float),
        where=parent_counts[:, None] > 0,
    )
    return BranchingEstimate(
        matrix=matrix,
        parent_counts=parent_counts,
        direct_child_counts=child_counts,
        spectral_radius=spectral_radius(matrix),
    )


def validate_branching_out_of_sample(
    training_rollouts: Iterable[PairedEventRollout],
    validation_rollouts: Iterable[PairedEventRollout],
    *,
    number_of_types: int = 1,
    max_generation: int | None = None,
) -> HeldOutCascadeValidation:
    """Fit direct offspring on one set and predict progeny on a held-out set."""

    training = tuple(training_rollouts)
    validation = tuple(validation_rollouts)
    return validate_branching_event_sets(
        (pair.treated.events for pair in training),
        (pair.treated.events for pair in validation),
        number_of_types=number_of_types,
        max_generation=max_generation,
    )


def validate_branching_event_sets(
    training_event_sets: Iterable[Iterable[InterventionEvent]],
    validation_event_sets: Iterable[Iterable[InterventionEvent]],
    *,
    number_of_types: int = 1,
    max_generation: int | None = None,
) -> HeldOutCascadeValidation:
    """Fit direct offspring on event logs and score independent event logs."""

    training = tuple(tuple(events) for events in training_event_sets)
    validation = tuple(tuple(events) for events in validation_event_sets)
    if not training or not validation:
        raise ValueError("training and validation sets must both be nonempty")
    branching = pool_direct_branching_events(
        training, number_of_types=number_of_types
    )
    primary_type_counts = np.zeros(number_of_types, dtype=float)
    for events in training:
        for event in events:
            if event.is_primary and not event.censored:
                primary_type_counts[event.event_type] += 1.0
    if np.sum(primary_type_counts) <= 0.0:
        raise ValueError("training rollouts contain no completed primary events")
    primary_distribution = primary_type_counts / np.sum(primary_type_counts)

    validation_primary = 0
    validation_causal = 0
    for events in validation:
        completed = tuple(
            event
            for event in events
            if event.is_causal and not event.censored
        )
        validation_causal += len(completed)
        validation_primary += sum(event.is_primary for event in completed)
    if validation_primary == 0:
        raise ValueError("validation rollouts contain no completed primary events")
    measured = validation_causal / validation_primary

    if max_generation is not None and max_generation < 0:
        raise ValueError("max_generation must be nonnegative")
    predicted: float | None
    relative_error: float | None
    if branching.spectral_radius < 1.0:
        predicted = mean_cascade_loss(
            branching.matrix,
            primary_distribution,
            np.ones(number_of_types),
        )
        relative_error = abs(predicted - measured) / measured
    else:
        predicted = None
        relative_error = None
    truncated_predicted: float | None = None
    truncated_error: float | None = None
    if max_generation is not None:
        truncated_predicted = truncated_cascade_loss(
            branching.matrix,
            primary_distribution,
            np.ones(number_of_types),
            max_generation,
        )
        truncated_error = abs(truncated_predicted - measured) / measured
    return HeldOutCascadeValidation(
        training_branching=branching,
        training_primary_distribution=primary_distribution,
        validation_primary_events=validation_primary,
        validation_causal_events=validation_causal,
        predicted_total_progeny_per_primary=predicted,
        measured_total_progeny_per_primary=measured,
        relative_error=relative_error,
        max_generation=max_generation,
        truncated_predicted_total_progeny_per_primary=truncated_predicted,
        truncated_relative_error=truncated_error,
    )


@dataclass(frozen=True)
class _FiniteChainCohort:
    size: int
    status: str


def _finite_chain_cohorts(
    event_sets: Iterable[Iterable[InterventionEvent]],
    fleet_size: int,
) -> tuple[_FiniteChainCohort, ...]:
    if (
        isinstance(fleet_size, (bool, np.bool_))
        or not isinstance(fleet_size, (int, np.integer))
        or fleet_size < 1
    ):
        raise ValueError("fleet_size must be a positive integer")
    cohorts: list[_FiniteChainCohort] = []
    for event_set in event_sets:
        causal = tuple(event for event in event_set if event.is_causal)
        by_root: dict[int, list[InterventionEvent]] = {}
        for event in causal:
            assert event.root_primary_event_id is not None
            by_root.setdefault(event.root_primary_event_id, []).append(event)
        for root_id, records in by_root.items():
            identifiers = {event.event_id for event in records}
            roots = [event for event in records if event.event_id == root_id]
            if len(roots) != 1 or not roots[0].is_primary:
                raise ValueError("each finite chain must have one logged primary root")
            if len({event.robot_id for event in records}) != len(records):
                raise ValueError(
                    "finite-chain estimator requires at most one event per robot "
                    "within a primary cascade"
                )
            if len(records) > fleet_size:
                raise ValueError("cascade contains more unique robots than fleet_size")
            child_counts = {event.event_id: 0 for event in records}
            for event in records:
                if event.event_id == root_id:
                    continue
                if event.parent_event_id not in identifiers:
                    raise ValueError("finite-chain cascade is missing a proximate parent")
                assert event.parent_event_id is not None
                child_counts[event.parent_event_id] += 1
            if any(count > 1 for count in child_counts.values()):
                raise ValueError("finite-chain estimator does not accept fan-out trees")
            if sum(child_counts.values()) != len(records) - 1:
                raise ValueError("causal events do not form a rooted chain")
            if len(records) == fleet_size:
                status = "fleet_saturated"
            elif any(event.censored for event in records):
                status = "indeterminate"
            else:
                status = "resolved"
            cohorts.append(_FiniteChainCohort(size=len(records), status=status))
    return tuple(cohorts)


def estimate_finite_chain_propagation(
    event_sets: Iterable[Iterable[InterventionEvent]],
    fleet_size: int,
) -> FiniteChainEstimate:
    """Estimate chain propagation with the fleet boundary as right censoring.

    A resolved cascade of size ``L < N`` contributes ``L-1`` propagation
    successes and one observed failure. A cascade reaching all ``N`` robots
    contributes ``N-1`` successes and no failure because another propagation
    opportunity does not physically exist. Temporally right-censored cascades
    below ``N`` are reported and excluded rather than treated as failures.
    """

    cohorts = _finite_chain_cohorts(event_sets, fleet_size)
    resolved = tuple(item for item in cohorts if item.status == "resolved")
    saturated = tuple(item for item in cohorts if item.status == "fleet_saturated")
    indeterminate = tuple(item for item in cohorts if item.status == "indeterminate")
    usable = resolved + saturated
    if not usable:
        if indeterminate:
            raise ValueError(
                "all primary cascades are right-censored below the fleet boundary"
            )
        raise ValueError("no resolved or fleet-saturated primary cascades")
    successes = int(sum(item.size - 1 for item in usable))
    failures = len(resolved)
    trials = successes + failures
    probability = successes / trials if trials else 0.0
    boundary_identity = bool(
        np.isclose(probability, 1.0, rtol=0.0, atol=1e-12)
        and len(resolved) == 0
        and len(saturated) > 0
    )
    return FiniteChainEstimate(
        fleet_size=int(fleet_size),
        propagation_successes=successes,
        observed_failures=failures,
        propagation_probability=float(probability),
        resolved_cascades=len(resolved),
        fleet_saturated_cascades=len(saturated),
        indeterminate_cascades=len(indeterminate),
        algebraic_boundary_identity=boundary_identity,
        validation_status=(
            "descriptive_not_predictive"
            if boundary_identity
            else "eligible_for_held_out_prediction"
        ),
    )


def validate_finite_chain_event_sets(
    training_event_sets: Iterable[Iterable[InterventionEvent]],
    validation_event_sets: Iterable[Iterable[InterventionEvent]],
    fleet_size: int,
) -> FiniteChainValidation:
    """Fit a boundary-censored chain probability and score held-out cascades."""

    training_sets = tuple(tuple(events) for events in training_event_sets)
    validation_sets = tuple(tuple(events) for events in validation_event_sets)
    if not training_sets or not validation_sets:
        raise ValueError("training and validation sets must both be nonempty")
    estimate = estimate_finite_chain_propagation(training_sets, fleet_size)
    if estimate.indeterminate_cascades:
        raise ValueError(
            "training contains right-censored cascades below the fleet boundary"
        )
    validation = _finite_chain_cohorts(validation_sets, fleet_size)
    resolved = tuple(item for item in validation if item.status == "resolved")
    saturated = tuple(
        item for item in validation if item.status == "fleet_saturated"
    )
    indeterminate = tuple(
        item for item in validation if item.status == "indeterminate"
    )
    if indeterminate:
        raise ValueError(
            "validation contains right-censored cascades below the fleet boundary"
        )
    usable = resolved + saturated
    if not usable:
        raise ValueError("validation contains no determinate primary cascades")
    measured = float(np.mean([item.size for item in usable]))
    probability = estimate.propagation_probability
    infinite = 1.0 / (1.0 - probability) if probability < 1.0 else None
    infinite_error = (
        abs(infinite - measured) / measured if infinite is not None else None
    )
    finite = float(sum(probability**generation for generation in range(fleet_size)))
    finite_error = abs(finite - measured) / measured
    return FiniteChainValidation(
        training=estimate,
        validation_resolved_cascades=len(resolved),
        validation_fleet_saturated_cascades=len(saturated),
        validation_indeterminate_cascades=len(indeterminate),
        measured_unique_robots_per_primary=measured,
        infinite_prediction=infinite,
        infinite_relative_error=infinite_error,
        finite_prediction=finite,
        finite_relative_error=finite_error,
        validation_status=estimate.validation_status,
    )


def validate_finite_chain_out_of_sample(
    training_rollouts: Iterable[PairedEventRollout],
    validation_rollouts: Iterable[PairedEventRollout],
    fleet_size: int,
) -> FiniteChainValidation:
    """Paired-rollout wrapper for finite-chain held-out validation."""

    return validate_finite_chain_event_sets(
        (pair.treated.events for pair in training_rollouts),
        (pair.treated.events for pair in validation_rollouts),
        fleet_size,
    )


def summarize_paired_rollout(pair: PairedRolloutResult) -> CorridorCascadeMetrics:
    """Estimate scalar cascade and burden statistics from paired rollouts.

    ``predicted_total_progeny_per_primary`` is a plug-in self-consistency value,
    not an independent validation. Use ``validate_branching_out_of_sample`` for
    a paper-facing prediction test.
    """

    causal = tuple(
        event
        for event in pair.treated.events
        if event.is_causal and not event.censored
    )
    primary = tuple(event for event in causal if event.is_primary)
    if not primary:
        raise ValueError("treated rollout contains no completed primary intervention")
    exposure = pair.control.total_traversals
    if exposure <= 0.0:
        raise ValueError("control rollout has no traversal exposure")
    primary_rate = len(primary) / exposure
    mean_severity = float(np.mean([event.severity_loss for event in causal]))
    direct = estimate_direct_branching(pair.treated.events)
    measured_progeny = len(causal) / len(primary)
    predicted_progeny: float | None
    if direct.spectral_radius < 1.0:
        predicted_progeny = cascade_budget_direct(direct.matrix)[0]
    else:
        predicted_progeny = None
    return CorridorCascadeMetrics(
        primary_events=len(primary),
        causal_events=len(causal),
        control_traversals=exposure,
        primary_events_per_traversal=primary_rate,
        mean_event_severity_loss=mean_severity,
        g_per_traversal=primary_rate * mean_severity,
        direct_branching=direct,
        measured_total_progeny_per_primary=measured_progeny,
        predicted_total_progeny_per_primary=predicted_progeny,
        paired_severity_loss=pair.attributable_severity,
        paired_traversal_loss=pair.traversal_loss,
    )
