"""Statistical diagnostics for the open-system corridor simulator.

Queue stability is inferred from backlog drift, not from finite-horizon
throughput alone.  Drift uncertainty uses a Newey--West HAC covariance because
queue samples are serially correlated.  Latency for arrival cohorts is reported
with a fixed follow-up horizon, preventing the usual completed-job selection
bias close to overload.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Iterable, Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .open_system import LoadSegment, OpenPairedRolloutResult, OpenSimulationResult


FloatArray = NDArray[np.float64]
StabilityLabel = Literal[
    "no_growth_detected", "growth_detected", "indeterminate"
]


@dataclass(frozen=True)
class QueueDriftEstimate:
    """Linear backlog drift with autocorrelation-robust uncertainty."""

    slope: float
    standard_error: float
    lower_confidence: float
    upper_confidence: float
    confidence: float
    sample_count: int
    hac_lag: int

    @property
    def z_score(self) -> float:
        if self.standard_error == 0.0:
            if self.slope > 0.0:
                return np.inf
            if self.slope < 0.0:
                return -np.inf
            return 0.0
        return self.slope / self.standard_error


@dataclass(frozen=True)
class RegenerationDiagnostics:
    """Evidence that an upstream queue repeatedly returns to empty."""

    observation_duration: float
    empty_time_fraction: float
    zero_return_count: int
    last_empty_time: float | None
    terminal_busy_duration: float
    longest_observed_busy_duration: float


@dataclass(frozen=True)
class OperatingWindowMetrics:
    """Steady-window rates, state averages, latency, and stability label."""

    start_time: float
    end_time: float
    requested_arrival_rate: float | None
    observed_arrival_rate: float
    admission_rate: float
    completion_rate: float
    mean_queue_length: float
    final_queue_length: int
    mean_work_in_process: float
    mean_available_robots: float | None
    mean_returning_robots: float | None
    mean_fleet_utilization: float | None
    mean_speed: float | None
    intervention_time_rate: float
    crossing_occupancy_fraction: float
    mean_active_crossings: float
    mean_completed_travel_time: float | None
    mean_completed_sojourn_time: float | None
    completed_jobs: int
    drift: QueueDriftEstimate
    regeneration: RegenerationDiagnostics
    stability: StabilityLabel


@dataclass(frozen=True)
class ArrivalCohortMetrics:
    """Fixed-follow-up latency summary robust to right censoring."""

    cohort_start: float
    cohort_end: float
    followup: float
    jobs: int
    completed_within_followup: int
    completion_probability: float | None
    restricted_mean_sojourn_time: float | None
    restricted_mean_standard_error: float | None


@dataclass(frozen=True)
class LittleLawDiagnostic:
    """Finite-window check of the corridor-level Little's-law approximation."""

    mean_work_in_process: float
    completion_rate: float
    mean_completed_travel_time: float | None
    implied_work_in_process: float | None
    absolute_residual: float | None
    relative_residual: float | None


@dataclass(frozen=True)
class SegmentMetrics:
    """Operating metrics attached to one load-profile segment."""

    segment: LoadSegment
    retained_start_time: float
    metrics: OperatingWindowMetrics


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    sample_count: int
    resamples: int


@dataclass(frozen=True)
class RateReplicationSummary:
    """Across-seed uncertainty summary at one requested load."""

    requested_arrival_rate: float
    replications: int
    completion_rate: BootstrapInterval
    queue_drift: BootstrapInterval
    mean_queue_length: BootstrapInterval
    no_growth_fraction: float
    growth_fraction: float


@dataclass(frozen=True)
class CapacityBracket:
    """Interval between the last no-growth and first growth-detected load."""

    largest_no_growth_rate: float | None
    smallest_growth_rate: float | None

    @property
    def resolved(self) -> bool:
        return (
            self.largest_no_growth_rate is not None
            and self.smallest_growth_rate is not None
            and self.largest_no_growth_rate < self.smallest_growth_rate
        )


@dataclass(frozen=True)
class RampBranchComparison:
    """Same-load up/down comparison; a dynamic-memory diagnostic only."""

    arrival_rate: float
    up_mean_queue: float
    down_mean_queue: float
    queue_gap: float
    up_completion_rate: float
    down_completion_rate: float


@dataclass(frozen=True)
class OpenCounterfactualMetrics:
    """Fleet-level effect under common arrivals with crossings removed."""

    completion_loss: int
    terminal_backlog_increase: int
    queue_time_increase: float
    severity_increase: float
    common_completed_jobs: int
    mean_paired_sojourn_increase: float | None


def _validate_confidence(confidence: float) -> float:
    value = float(confidence)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    return value


def _window_series(
    time: FloatArray,
    values: FloatArray,
    start_time: float,
    end_time: float,
) -> tuple[FloatArray, FloatArray]:
    if not np.isfinite(start_time) or not np.isfinite(end_time):
        raise ValueError("window bounds must be finite")
    if start_time < time[0] or end_time > time[-1] or end_time <= start_time:
        raise ValueError("window must lie inside the recorded horizon")
    interior = (time > start_time) & (time < end_time)
    selected_time = np.concatenate(
        ([start_time], time[interior], [end_time])
    ).astype(float)
    selected_values = np.concatenate(
        (
            [np.interp(start_time, time, values)],
            values[interior],
            [np.interp(end_time, time, values)],
        )
    ).astype(float)
    return selected_time, selected_values


def _trapezoid(values: FloatArray, time: FloatArray) -> float:
    """Version-independent composite trapezoidal integral."""

    if values.size != time.size:
        raise ValueError("values and time must have equal length")
    if values.size < 2:
        return 0.0
    return float(np.sum(0.5 * (values[:-1] + values[1:]) * np.diff(time)))


def estimate_queue_drift(
    time: ArrayLike,
    queue_length: ArrayLike,
    *,
    confidence: float = 0.95,
    hac_lag: int | None = None,
) -> QueueDriftEstimate:
    """Estimate queue slope by OLS with a Bartlett-kernel Newey--West SE."""

    level = _validate_confidence(confidence)
    x = np.asarray(time, dtype=float).reshape(-1)
    y = np.asarray(queue_length, dtype=float).reshape(-1)
    if x.size != y.size or x.size < 3:
        raise ValueError("time and queue_length need at least three paired samples")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("time and queue_length must be finite")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("time must be strictly increasing")

    centered = x - np.mean(x)
    design = np.column_stack((np.ones(x.size, dtype=float), centered))
    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ y
    residual = y - design @ coefficients

    if hac_lag is None:
        lag = int(np.floor(4.0 * (x.size / 100.0) ** (2.0 / 9.0)))
    else:
        lag = int(hac_lag)
    if lag < 0 or lag >= x.size:
        raise ValueError("hac_lag must lie in [0, sample_count)")

    scores = design * residual[:, None]
    meat = scores.T @ scores
    for offset in range(1, lag + 1):
        weight = 1.0 - offset / (lag + 1.0)
        cross = scores[offset:].T @ scores[:-offset]
        meat += weight * (cross + cross.T)
    covariance = xtx_inverse @ meat @ xtx_inverse
    # Small-sample degrees-of-freedom correction.  Numerical round-off can
    # make a theoretically zero variance slightly negative.
    covariance *= x.size / (x.size - design.shape[1])
    standard_error = float(np.sqrt(max(0.0, covariance[1, 1])))
    slope = float(coefficients[1])
    critical = NormalDist().inv_cdf(0.5 + level / 2.0)
    return QueueDriftEstimate(
        slope=slope,
        standard_error=standard_error,
        lower_confidence=slope - critical * standard_error,
        upper_confidence=slope + critical * standard_error,
        confidence=level,
        sample_count=x.size,
        hac_lag=lag,
    )


def classify_queue_stability(
    drift: QueueDriftEstimate,
    observed_arrival_rate: float,
    completion_rate: float,
    *,
    regeneration: RegenerationDiagnostics | None = None,
    drift_tolerance: float = 0.002,
    rate_tolerance: float = 0.01,
) -> StabilityLabel:
    """Classify finite-run evidence without claiming positive recurrence.

    A finite trajectory cannot prove queue stability.  ``no_growth_detected``
    requires both flow balance and observed regeneration.  ``growth_detected``
    requires a positive drift confidence bound and a terminal busy period.
    Everything else remains explicitly indeterminate.
    """

    if not np.isfinite(drift_tolerance) or drift_tolerance < 0.0:
        raise ValueError("drift_tolerance must be finite and nonnegative")
    if not np.isfinite(rate_tolerance) or rate_tolerance < 0.0:
        raise ValueError("rate_tolerance must be finite and nonnegative")
    flow_balanced = completion_rate + rate_tolerance >= observed_arrival_rate
    if regeneration is None:
        if drift.lower_confidence > drift_tolerance:
            return "growth_detected"
        if drift.upper_confidence <= drift_tolerance and flow_balanced:
            return "no_growth_detected"
        return "indeterminate"

    sustained_terminal_busy_period = (
        regeneration.terminal_busy_duration > 0.0
        and regeneration.terminal_busy_duration
        >= 0.5 * regeneration.observation_duration
    )
    if (
        drift.lower_confidence > drift_tolerance
        and sustained_terminal_busy_period
        and regeneration.empty_time_fraction == 0.0
    ):
        return "growth_detected"
    regenerated = regeneration.empty_time_fraction >= 0.05 and (
        regeneration.zero_return_count >= 1
        or regeneration.longest_observed_busy_duration == 0.0
    )
    if regenerated and flow_balanced:
        return "no_growth_detected"
    return "indeterminate"


def queue_regeneration_diagnostics(
    time: ArrayLike,
    queue_length: ArrayLike,
) -> RegenerationDiagnostics:
    """Measure empty returns and observed busy-period durations."""

    x = np.asarray(time, dtype=float).reshape(-1)
    q = np.asarray(queue_length, dtype=float).reshape(-1)
    if x.size != q.size or x.size < 2:
        raise ValueError("time and queue_length need at least two paired samples")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(q)):
        raise ValueError("time and queue_length must be finite")
    if np.any(np.diff(x) <= 0.0) or np.any(q < 0.0):
        raise ValueError("time must increase and queue length must be nonnegative")
    empty = q < 0.5
    horizon = x[-1] - x[0]
    empty_fraction = _trapezoid(empty.astype(float), x) / horizon
    zero_returns = int(np.count_nonzero(empty[1:] & ~empty[:-1]))
    empty_indices = np.flatnonzero(empty)
    last_empty = float(x[empty_indices[-1]]) if empty_indices.size else None

    busy_durations: list[float] = []
    busy_start: float | None = None
    for index, is_empty in enumerate(empty):
        if not is_empty and busy_start is None:
            busy_start = float(x[index])
        if is_empty and busy_start is not None:
            busy_durations.append(float(x[index]) - busy_start)
            busy_start = None
    terminal_busy = 0.0
    if busy_start is not None:
        terminal_busy = float(x[-1]) - busy_start
        busy_durations.append(terminal_busy)
    return RegenerationDiagnostics(
        observation_duration=float(horizon),
        empty_time_fraction=float(empty_fraction),
        zero_return_count=zero_returns,
        last_empty_time=last_empty,
        terminal_busy_duration=terminal_busy,
        longest_observed_busy_duration=(max(busy_durations) if busy_durations else 0.0),
    )


def analyze_operating_window(
    result: OpenSimulationResult,
    start_time: float,
    end_time: float | None = None,
    *,
    requested_arrival_rate: float | None = None,
    confidence: float = 0.95,
    hac_lag: int | None = None,
    drift_tolerance: float = 0.002,
    rate_tolerance: float = 0.01,
) -> OperatingWindowMetrics:
    """Compute rates and a queue-stability decision on a retained window."""

    end = result.duration if end_time is None else float(end_time)
    time = result.trajectory.time
    queue_time, queue = _window_series(
        time,
        result.trajectory.queue_length.astype(float),
        float(start_time),
        end,
    )
    _, wip = _window_series(
        time,
        result.trajectory.work_in_process.astype(float),
        float(start_time),
        end,
    )
    available: FloatArray | None = None
    returning: FloatArray | None = None
    if result.config.fleet_size is not None:
        _, available = _window_series(
            time,
            result.trajectory.available_robots.astype(float),
            float(start_time),
            end,
        )
        _, returning = _window_series(
            time,
            result.trajectory.returning_robots.astype(float),
            float(start_time),
            end,
        )
    _, crossing = _window_series(
        time,
        result.trajectory.crossing_active.astype(float),
        float(start_time),
        end,
    )
    _, active_crossing_count = _window_series(
        time,
        result.trajectory.active_crossing_count.astype(float),
        float(start_time),
        end,
    )
    _, interventions = _window_series(
        time,
        result.trajectory.active_interventions.astype(float),
        float(start_time),
        end,
    )
    _, mean_speed_series = _window_series(
        time,
        result.trajectory.mean_speed,
        float(start_time),
        end,
    )
    horizon = end - float(start_time)

    arrivals = int(
        np.count_nonzero(
            (result.arrival_times >= start_time - 1e-12)
            & (result.arrival_times < end - 1e-12)
        )
    )
    all_admissions = np.asarray(
        [item.admission_time for item in result.completed_traversals]
        + result.final_state.admission_time.tolist(),
        dtype=float,
    )
    admissions = int(
        np.count_nonzero(
            (all_admissions >= start_time - 1e-12)
            & (all_admissions < end - 1e-12)
        )
    )
    window_completions = tuple(
        item
        for item in result.completed_traversals
        if start_time - 1e-12 <= item.completion_time < end - 1e-12
    )
    completion_count = len(window_completions)
    observed_arrival_rate = arrivals / horizon
    admission_rate = admissions / horizon
    completion_rate = completion_count / horizon

    drift = estimate_queue_drift(
        queue_time,
        queue,
        confidence=confidence,
        hac_lag=hac_lag,
    )
    regeneration = queue_regeneration_diagnostics(queue_time, queue)
    stability = classify_queue_stability(
        drift,
        observed_arrival_rate,
        completion_rate,
        regeneration=regeneration,
        drift_tolerance=drift_tolerance,
        rate_tolerance=rate_tolerance,
    )

    finite_speed = mean_speed_series[np.isfinite(mean_speed_series)]
    mean_wip = _trapezoid(wip, queue_time) / horizon
    mean_available = (
        _trapezoid(available, queue_time) / horizon
        if available is not None
        else None
    )
    mean_returning = (
        _trapezoid(returning, queue_time) / horizon
        if returning is not None
        else None
    )
    travel = np.asarray(
        [item.travel_time for item in window_completions], dtype=float
    )
    sojourn = np.asarray(
        [item.sojourn_time for item in window_completions], dtype=float
    )
    return OperatingWindowMetrics(
        start_time=float(start_time),
        end_time=end,
        requested_arrival_rate=requested_arrival_rate,
        observed_arrival_rate=observed_arrival_rate,
        admission_rate=admission_rate,
        completion_rate=completion_rate,
        mean_queue_length=_trapezoid(queue, queue_time) / horizon,
        final_queue_length=int(round(queue[-1])),
        mean_work_in_process=mean_wip,
        mean_available_robots=mean_available,
        mean_returning_robots=mean_returning,
        mean_fleet_utilization=(
            mean_wip / result.config.fleet_size
            if result.config.fleet_size is not None
            else None
        ),
        mean_speed=(float(np.mean(finite_speed)) if finite_speed.size else None),
        intervention_time_rate=_trapezoid(interventions, queue_time) / horizon,
        crossing_occupancy_fraction=_trapezoid(crossing, queue_time) / horizon,
        mean_active_crossings=(
            _trapezoid(active_crossing_count, queue_time) / horizon
        ),
        mean_completed_travel_time=(float(np.mean(travel)) if travel.size else None),
        mean_completed_sojourn_time=(
            float(np.mean(sojourn)) if sojourn.size else None
        ),
        completed_jobs=completion_count,
        drift=drift,
        regeneration=regeneration,
        stability=stability,
    )


def analyze_arrival_cohort(
    result: OpenSimulationResult,
    cohort_start: float,
    cohort_end: float,
    followup: float,
) -> ArrivalCohortMetrics:
    """Estimate completion and restricted mean sojourn at equal follow-up.

    Only arrivals with a full ``followup`` interval before the simulation
    horizon are accepted.  Therefore the restricted mean is simply the sample
    mean of ``min(sojourn, followup)`` and is not biased toward fast jobs.
    """

    if not np.isfinite(followup) or followup <= 0.0:
        raise ValueError("followup must be finite and strictly positive")
    if (
        not np.isfinite(cohort_start)
        or not np.isfinite(cohort_end)
        or cohort_start < 0.0
        or cohort_end <= cohort_start
        or cohort_end + followup > result.duration + 1e-12
    ):
        raise ValueError("cohort must have complete follow-up inside the run")
    identifiers = np.flatnonzero(
        (result.arrival_times >= cohort_start - 1e-12)
        & (result.arrival_times < cohort_end - 1e-12)
    )
    if identifiers.size == 0:
        return ArrivalCohortMetrics(
            cohort_start=cohort_start,
            cohort_end=cohort_end,
            followup=followup,
            jobs=0,
            completed_within_followup=0,
            completion_probability=None,
            restricted_mean_sojourn_time=None,
            restricted_mean_standard_error=None,
        )
    completion_by_job = {
        item.job_id: item.completion_time for item in result.completed_traversals
    }
    restricted: list[float] = []
    completed_count = 0
    for identifier in identifiers:
        arrival = float(result.arrival_times[identifier])
        completion = completion_by_job.get(int(identifier))
        if completion is not None and completion <= arrival + followup + 1e-12:
            restricted.append(completion - arrival)
            completed_count += 1
        else:
            restricted.append(followup)
    values = np.asarray(restricted, dtype=float)
    standard_error = (
        float(np.std(values, ddof=1) / np.sqrt(values.size))
        if values.size > 1
        else 0.0
    )
    return ArrivalCohortMetrics(
        cohort_start=cohort_start,
        cohort_end=cohort_end,
        followup=followup,
        jobs=int(identifiers.size),
        completed_within_followup=completed_count,
        completion_probability=completed_count / identifiers.size,
        restricted_mean_sojourn_time=float(np.mean(values)),
        restricted_mean_standard_error=standard_error,
    )


def little_law_diagnostic(
    metrics: OperatingWindowMetrics,
) -> LittleLawDiagnostic:
    """Compare time-average WIP with throughput times completed travel time."""

    travel = metrics.mean_completed_travel_time
    if travel is None:
        implied = None
        residual = None
        relative = None
    else:
        implied = metrics.completion_rate * travel
        residual = metrics.mean_work_in_process - implied
        relative = (
            residual / metrics.mean_work_in_process
            if metrics.mean_work_in_process > 0.0
            else None
        )
    return LittleLawDiagnostic(
        mean_work_in_process=metrics.mean_work_in_process,
        completion_rate=metrics.completion_rate,
        mean_completed_travel_time=travel,
        implied_work_in_process=implied,
        absolute_residual=residual,
        relative_residual=relative,
    )


def analyze_load_segments(
    result: OpenSimulationResult,
    segments: Sequence[LoadSegment],
    *,
    discard_fraction: float = 0.5,
    confidence: float = 0.95,
    drift_tolerance: float = 0.002,
    rate_tolerance: float = 0.01,
) -> tuple[SegmentMetrics, ...]:
    """Analyze the retained tail of every load-ramp dwell segment."""

    if not np.isfinite(discard_fraction) or not 0.0 <= discard_fraction < 1.0:
        raise ValueError("discard_fraction must lie in [0, 1)")
    summaries: list[SegmentMetrics] = []
    for segment in segments:
        retained_start = segment.start_time + discard_fraction * (
            segment.end_time - segment.start_time
        )
        summaries.append(
            SegmentMetrics(
                segment=segment,
                retained_start_time=retained_start,
                metrics=analyze_operating_window(
                    result,
                    retained_start,
                    segment.end_time,
                    requested_arrival_rate=segment.arrival_rate,
                    confidence=confidence,
                    drift_tolerance=drift_tolerance,
                    rate_tolerance=rate_tolerance,
                ),
            )
        )
    return tuple(summaries)


def bootstrap_mean_interval(
    values: ArrayLike,
    *,
    confidence: float = 0.95,
    resamples: int = 4000,
    seed: int | None = 0,
) -> BootstrapInterval:
    """Percentile bootstrap confidence interval for a replication mean."""

    level = _validate_confidence(confidence)
    data = np.asarray(values, dtype=float).reshape(-1)
    if data.size == 0 or not np.all(np.isfinite(data)):
        raise ValueError("values must be a nonempty finite vector")
    if resamples < 1:
        raise ValueError("resamples must be at least one")
    estimate = float(np.mean(data))
    if data.size == 1:
        lower = upper = estimate
    else:
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, data.size, size=(resamples, data.size))
        means = np.mean(data[indices], axis=1)
        alpha = (1.0 - level) / 2.0
        lower, upper = map(float, np.quantile(means, (alpha, 1.0 - alpha)))
    return BootstrapInterval(
        estimate=estimate,
        lower=lower,
        upper=upper,
        confidence=level,
        sample_count=int(data.size),
        resamples=resamples,
    )


def summarize_rate_replications(
    requested_arrival_rate: float,
    metrics: Iterable[OperatingWindowMetrics],
    *,
    confidence: float = 0.95,
    resamples: int = 4000,
    seed: int | None = 0,
) -> RateReplicationSummary:
    """Pool independent seeds without treating time samples as independent."""

    records = tuple(metrics)
    if not records:
        raise ValueError("at least one replication is required")
    labels = [item.stability for item in records]
    return RateReplicationSummary(
        requested_arrival_rate=float(requested_arrival_rate),
        replications=len(records),
        completion_rate=bootstrap_mean_interval(
            [item.completion_rate for item in records],
            confidence=confidence,
            resamples=resamples,
            seed=seed,
        ),
        queue_drift=bootstrap_mean_interval(
            [item.drift.slope for item in records],
            confidence=confidence,
            resamples=resamples,
            seed=None if seed is None else seed + 1,
        ),
        mean_queue_length=bootstrap_mean_interval(
            [item.mean_queue_length for item in records],
            confidence=confidence,
            resamples=resamples,
            seed=None if seed is None else seed + 2,
        ),
        no_growth_fraction=labels.count("no_growth_detected") / len(labels),
        growth_fraction=labels.count("growth_detected") / len(labels),
    )


def empirical_capacity_bracket(
    summaries: Iterable[RateReplicationSummary],
    *,
    decision_fraction: float = 0.8,
) -> CapacityBracket:
    """Bracket capacity; never interpolate across indeterminate loads."""

    if not np.isfinite(decision_fraction) or not 0.5 <= decision_fraction <= 1.0:
        raise ValueError("decision_fraction must lie in [0.5, 1]")
    records = tuple(sorted(summaries, key=lambda item: item.requested_arrival_rate))
    no_growth = [
        item.requested_arrival_rate
        for item in records
        if item.no_growth_fraction >= decision_fraction
    ]
    growth = [
        item.requested_arrival_rate
        for item in records
        if item.growth_fraction >= decision_fraction
    ]
    largest_no_growth = max(no_growth) if no_growth else None
    eligible_growth = [
        rate
        for rate in growth
        if largest_no_growth is None or rate > largest_no_growth
    ]
    return CapacityBracket(
        largest_no_growth_rate=largest_no_growth,
        smallest_growth_rate=(
            min(eligible_growth) if eligible_growth else None
        ),
    )


def compare_ramp_branches(
    up_segments: Sequence[SegmentMetrics],
    down_segments: Sequence[SegmentMetrics],
) -> tuple[RampBranchComparison, ...]:
    """Compare equal-load ramp branches without claiming static hysteresis.

    A nonzero gap can be caused entirely by finite dwell time and inherited
    backlog.  A drained-reset experiment is required before interpreting it as
    bistability.
    """

    up = {item.segment.arrival_rate: item for item in up_segments}
    down = {item.segment.arrival_rate: item for item in down_segments}
    comparisons: list[RampBranchComparison] = []
    for rate in sorted(up.keys() & down.keys()):
        upward = up[rate].metrics
        downward = down[rate].metrics
        comparisons.append(
            RampBranchComparison(
                arrival_rate=rate,
                up_mean_queue=upward.mean_queue_length,
                down_mean_queue=downward.mean_queue_length,
                queue_gap=downward.mean_queue_length - upward.mean_queue_length,
                up_completion_rate=upward.completion_rate,
                down_completion_rate=downward.completion_rate,
            )
        )
    return tuple(comparisons)


def summarize_open_counterfactual(
    pair: OpenPairedRolloutResult,
) -> OpenCounterfactualMetrics:
    """Compute work-conserving fleet effects from a paired open rollout."""

    treated_time = pair.treated.trajectory.time
    control_time = pair.control.trajectory.time
    if not np.array_equal(treated_time, control_time):
        raise ValueError("paired trajectories must share the same time grid")
    treated_backlog = (
        pair.treated.trajectory.queue_length
        + pair.treated.trajectory.work_in_process
    ).astype(float)
    control_backlog = (
        pair.control.trajectory.queue_length
        + pair.control.trajectory.work_in_process
    ).astype(float)
    queue_time_increase = _trapezoid(
        treated_backlog - control_backlog, treated_time
    )
    return OpenCounterfactualMetrics(
        completion_loss=pair.completion_loss,
        terminal_backlog_increase=pair.terminal_backlog_increase,
        queue_time_increase=queue_time_increase,
        severity_increase=pair.severity_increase,
        common_completed_jobs=int(pair.common_completed_job_ids.size),
        mean_paired_sojourn_increase=pair.mean_paired_sojourn_increase,
    )
