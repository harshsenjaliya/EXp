"""Open-system microscopic AMR corridor simulator.

Jobs arrive exogenously to an upstream FIFO queue.  A job is admitted as a
robot traversal only when the corridor entry satisfies the same stopping-
distance contract used inside the corridor.  Robots leave the state when they
complete the route, so work in process and downstream coupling are endogenous.

The simulator is intentionally one-dimensional in its collision logic: it is
the open-boundary counterpart of :mod:`amr_capacity.simulation`, not a free-
space navigation engine.  Fleet state is held in NumPy arrays ordered from the
front robot to the rear robot, giving O(N) leader lookup per integration step.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .capacity_theory import d_stop, v_safe_next_step
from .simulation import BlockerKind, EventCause, InterventionEvent


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
CrossingStatus = Literal["completed", "active_at_end", "unserved"]


@dataclass(frozen=True)
class OpenCorridorConfig:
    """Physical and numerical parameters for a one-way open corridor."""

    corridor_length: float
    desired_speed: float
    acceleration: float
    braking: float
    reaction_time: float
    dt: float
    robot_length: float
    safety_margin: float
    sensor_range: float
    crossing_x: float | tuple[float, ...]
    crossing_half_width: float = 0.4
    admission_speed: float = 0.0
    intervention_tolerance: float = 1e-4
    collision_tolerance: float = 1e-9
    fleet_size: int | None = None
    empty_return_time: float = 0.0

    def __post_init__(self) -> None:
        positive = {
            "corridor_length": self.corridor_length,
            "desired_speed": self.desired_speed,
            "acceleration": self.acceleration,
            "braking": self.braking,
            "dt": self.dt,
            "robot_length": self.robot_length,
            "sensor_range": self.sensor_range,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and strictly positive")
        nonnegative = {
            "reaction_time": self.reaction_time,
            "safety_margin": self.safety_margin,
            "crossing_half_width": self.crossing_half_width,
            "admission_speed": self.admission_speed,
            "intervention_tolerance": self.intervention_tolerance,
            "collision_tolerance": self.collision_tolerance,
            "empty_return_time": self.empty_return_time,
        }
        for name, value in nonnegative.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.reaction_time < self.dt:
            raise ValueError("reaction_time must be at least one integration step")
        if self.admission_speed > self.desired_speed:
            raise ValueError("admission_speed must not exceed desired_speed")
        if self.fleet_size is not None and (
            isinstance(self.fleet_size, (bool, np.bool_))
            or not isinstance(self.fleet_size, (int, np.integer))
            or self.fleet_size < 1
        ):
            raise ValueError("fleet_size must be a positive integer or None")
        crossing_positions = np.asarray(
            (self.crossing_x,)
            if np.isscalar(self.crossing_x)
            else self.crossing_x,
            dtype=float,
        ).reshape(-1)
        if crossing_positions.size == 0 or not np.all(np.isfinite(crossing_positions)):
            raise ValueError("crossing_x must contain at least one finite position")
        if np.any(np.diff(crossing_positions) <= 0.0):
            raise ValueError("crossing positions must be strictly increasing")
        if np.any(crossing_positions <= self.crossing_clearance) or np.any(
            crossing_positions >= self.corridor_length - self.crossing_clearance
        ):
            raise ValueError(
                "each crossing must leave its protected zone inside the corridor"
            )
        physical_diameter = self.robot_length + 2.0 * self.crossing_half_width
        if crossing_positions.size > 1 and np.any(
            np.diff(crossing_positions) <= physical_diameter
        ):
            raise ValueError("crossing protected zones must not overlap")
        normalized_crossing_x: float | tuple[float, ...]
        if crossing_positions.size == 1:
            normalized_crossing_x = float(crossing_positions[0])
        else:
            normalized_crossing_x = tuple(float(item) for item in crossing_positions)
        object.__setattr__(self, "crossing_x", normalized_crossing_x)
        required_sensor_range = max(
            float(
                d_stop(
                    self.desired_speed,
                    self.robot_clearance,
                    self.reaction_time,
                    self.braking,
                )
            ),
            float(
                d_stop(
                    self.desired_speed,
                    self.crossing_clearance,
                    self.reaction_time,
                    self.braking,
                )
            ),
        )
        if self.sensor_range + self.collision_tolerance < required_sensor_range:
            raise ValueError(
                "sensor_range must cover the desired-speed stopping distance"
            )

    @property
    def robot_clearance(self) -> float:
        """Static center-to-center clearance between consecutive robots."""

        return self.robot_length + self.safety_margin

    @property
    def crossing_clearance(self) -> float:
        """Static robot-center clearance from an occupied crossing."""

        return self.robot_length / 2.0 + self.crossing_half_width + self.safety_margin

    @property
    def crossing_positions(self) -> FloatArray:
        """Ordered disturbance locations as a one-dimensional array."""

        return np.asarray(
            (self.crossing_x,)
            if np.isscalar(self.crossing_x)
            else self.crossing_x,
            dtype=float,
        )

    @property
    def crossing_count(self) -> int:
        return int(self.crossing_positions.size)


@dataclass(frozen=True)
class CrossingRequest:
    """A request to occupy the crossing for a specified service duration.

    ``request_time`` is exogenous.  Actual activation is endogenous and occurs
    only when the protected zone can be occupied without violating the shield.
    """

    request_time: float
    duration: float
    source_id: int
    crossing_id: int = 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.request_time) or self.request_time < 0.0:
            raise ValueError("request_time must be finite and nonnegative")
        if not np.isfinite(self.duration) or self.duration <= 0.0:
            raise ValueError("duration must be finite and strictly positive")
        if not isinstance(self.source_id, (int, np.integer)):
            raise ValueError("source_id must be an integer")
        if (
            isinstance(self.crossing_id, (bool, np.bool_))
            or not isinstance(self.crossing_id, (int, np.integer))
            or self.crossing_id < 0
        ):
            raise ValueError("crossing_id must be a nonnegative integer")


@dataclass(frozen=True)
class CrossingServiceRecord:
    """Observed service of one exogenous crossing request."""

    source_id: int
    crossing_id: int
    request_time: float
    requested_duration: float
    activation_time: float | None
    release_time: float | None
    status: CrossingStatus

    @property
    def wait_time(self) -> float | None:
        if self.activation_time is None:
            return None
        return self.activation_time - self.request_time

    @property
    def occupied_time(self) -> float:
        if self.activation_time is None or self.release_time is None:
            return 0.0
        return self.release_time - self.activation_time


@dataclass(frozen=True)
class CompletedTraversal:
    """One completed job with queueing and in-corridor timing."""

    job_id: int
    robot_id: int
    arrival_time: float
    admission_time: float
    completion_time: float
    severity_loss: float

    @property
    def queue_wait(self) -> float:
        return self.admission_time - self.arrival_time

    @property
    def travel_time(self) -> float:
        return self.completion_time - self.admission_time

    @property
    def sojourn_time(self) -> float:
        return self.completion_time - self.arrival_time


@dataclass(frozen=True)
class OpenTrajectoryLog:
    """Strided aggregate state for an open-corridor realization."""

    time: FloatArray
    queue_length: IntArray
    work_in_process: IntArray
    available_robots: IntArray
    returning_robots: IntArray
    cumulative_arrivals: IntArray
    cumulative_admissions: IntArray
    cumulative_completions: IntArray
    cumulative_severity_loss: FloatArray
    mean_speed: FloatArray
    active_interventions: IntArray
    crossing_active: BoolArray
    crossing_reserved: BoolArray
    pending_crossings: IntArray
    active_crossing_count: IntArray
    reserved_crossing_count: IntArray


@dataclass(frozen=True)
class OpenFinalState:
    """Robots and queued jobs remaining at the simulation horizon."""

    robot_id: IntArray
    job_id: IntArray
    position: FloatArray
    speed: FloatArray
    arrival_time: FloatArray
    admission_time: FloatArray
    severity_loss: FloatArray
    queued_job_id: IntArray
    queued_arrival_time: FloatArray
    available_robot_id: IntArray
    returning_robot_id: IntArray
    return_ready_time: FloatArray


@dataclass(frozen=True)
class OpenSimulationResult:
    """Complete open-system result used by estimators and experiments."""

    config: OpenCorridorConfig
    duration: float
    arrival_times: FloatArray
    crossing_requests: tuple[CrossingRequest, ...]
    crossing_services: tuple[CrossingServiceRecord, ...]
    completed_traversals: tuple[CompletedTraversal, ...]
    events: tuple[InterventionEvent, ...]
    trajectory: OpenTrajectoryLog
    final_state: OpenFinalState
    total_admissions: int
    total_severity_loss: float
    collision_count: int
    crossing_violation_count: int

    @property
    def total_arrivals(self) -> int:
        return int(self.arrival_times.size)

    @property
    def total_completions(self) -> int:
        return len(self.completed_traversals)

    @property
    def final_queue_length(self) -> int:
        return int(self.final_state.queued_job_id.size)

    @property
    def final_work_in_process(self) -> int:
        return int(self.final_state.robot_id.size)

    @property
    def available_robot_count(self) -> int | None:
        if self.config.fleet_size is None:
            return None
        return int(self.final_state.available_robot_id.size)

    @property
    def returning_robot_count(self) -> int | None:
        if self.config.fleet_size is None:
            return None
        return int(self.final_state.returning_robot_id.size)

    @property
    def fleet_balance_residual(self) -> int | None:
        if self.config.fleet_size is None:
            return None
        return (
            self.config.fleet_size
            - self.final_work_in_process
            - int(self.final_state.available_robot_id.size)
            - int(self.final_state.returning_robot_id.size)
        )

    @property
    def throughput(self) -> float:
        return self.total_completions / self.duration

    @property
    def mass_balance_residual(self) -> int:
        """Arrivals minus queue, WIP, and completed jobs (must be zero)."""

        return (
            self.total_arrivals
            - self.final_queue_length
            - self.final_work_in_process
            - self.total_completions
        )


@dataclass(frozen=True)
class OpenPairedRolloutResult:
    """Crossing-request realization and matched no-crossing counterfactual."""

    treated: OpenSimulationResult
    control: OpenSimulationResult

    @property
    def completion_loss(self) -> int:
        return self.control.total_completions - self.treated.total_completions

    @property
    def terminal_backlog_increase(self) -> int:
        treated_backlog = (
            self.treated.final_queue_length + self.treated.final_work_in_process
        )
        control_backlog = (
            self.control.final_queue_length + self.control.final_work_in_process
        )
        return treated_backlog - control_backlog

    @property
    def severity_increase(self) -> float:
        return self.treated.total_severity_loss - self.control.total_severity_loss

    @property
    def common_completed_job_ids(self) -> IntArray:
        treated = {item.job_id for item in self.treated.completed_traversals}
        control = {item.job_id for item in self.control.completed_traversals}
        return np.asarray(sorted(treated & control), dtype=int)

    @property
    def mean_paired_sojourn_increase(self) -> float | None:
        identifiers = self.common_completed_job_ids
        if identifiers.size == 0:
            return None
        treated = {
            item.job_id: item.sojourn_time
            for item in self.treated.completed_traversals
        }
        control = {
            item.job_id: item.sojourn_time
            for item in self.control.completed_traversals
        }
        differences = np.asarray(
            [treated[int(key)] - control[int(key)] for key in identifiers],
            dtype=float,
        )
        return float(np.mean(differences))


@dataclass(frozen=True)
class LoadSegment:
    """Piecewise-constant arrival-rate segment for load-ramp experiments."""

    start_time: float
    end_time: float
    arrival_rate: float
    label: str = ""

    def __post_init__(self) -> None:
        if not np.isfinite(self.start_time) or self.start_time < 0.0:
            raise ValueError("start_time must be finite and nonnegative")
        if not np.isfinite(self.end_time) or self.end_time <= self.start_time:
            raise ValueError("end_time must be finite and exceed start_time")
        if not np.isfinite(self.arrival_rate) or self.arrival_rate < 0.0:
            raise ValueError("arrival_rate must be finite and nonnegative")


@dataclass
class _MutableEvent:
    event_id: int
    robot_id: int
    start_time: float
    cause: EventCause
    blocker_robot_id: int | None
    parent_event_id: int | None
    crossing_source_id: int | None
    start_position: float
    crossing_id: int | None
    end_time: float | None = None
    severity_loss: float = 0.0
    minimum_speed: float = np.inf
    censored: bool = False


def _validate_duration(duration: float, dt: float) -> int:
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be finite and strictly positive")
    step_count_float = duration / dt
    step_count = int(round(step_count_float))
    if not np.isclose(step_count, step_count_float, rtol=0.0, atol=1e-10):
        raise ValueError("duration must be an integer multiple of dt")
    return step_count


def _validate_arrival_times(arrival_times: ArrayLike, duration: float) -> FloatArray:
    values = np.asarray(arrival_times, dtype=float).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("arrival_times must be finite")
    if np.any(values < 0.0) or np.any(values >= duration):
        raise ValueError("arrival_times must lie in [0, duration)")
    if values.size > 1 and np.any(np.diff(values) < 0.0):
        raise ValueError("arrival_times must be sorted")
    return values.copy()


def _validate_requests(
    requests: Iterable[CrossingRequest], duration: float, crossing_count: int
) -> tuple[CrossingRequest, ...]:
    ordered = tuple(sorted(requests, key=lambda item: (item.request_time, item.source_id)))
    identifiers = [item.source_id for item in ordered]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("crossing source_id values must be unique")
    if any(item.request_time >= duration for item in ordered):
        raise ValueError("crossing request times must lie before duration")
    if any(item.crossing_id >= crossing_count for item in ordered):
        raise ValueError("crossing_id lies outside the configured crossings")
    return ordered


def deterministic_arrival_times(
    rate: float,
    duration: float,
    *,
    start_time: float = 0.0,
) -> FloatArray:
    """Generate a regular arrival stream with no arrival exactly at the start."""

    if not np.isfinite(rate) or rate < 0.0:
        raise ValueError("rate must be finite and nonnegative")
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be finite and strictly positive")
    if not np.isfinite(start_time) or not 0.0 <= start_time < duration:
        raise ValueError("start_time must lie in [0, duration)")
    if rate == 0.0:
        return np.empty(0, dtype=float)
    interval = 1.0 / rate
    count = max(0, int(np.ceil((duration - start_time) / interval)) - 1)
    values = start_time + interval * np.arange(1, count + 1, dtype=float)
    return values[values < duration - 1e-12]


def poisson_arrival_times(
    rate: float,
    duration: float,
    *,
    seed: int | None = None,
    start_time: float = 0.0,
) -> FloatArray:
    """Generate homogeneous Poisson arrival times from a local RNG."""

    segment = LoadSegment(start_time, duration, rate)
    return piecewise_poisson_arrival_times((segment,), seed=seed)


def piecewise_poisson_arrival_times(
    segments: Sequence[LoadSegment],
    *,
    seed: int | None = None,
) -> FloatArray:
    """Generate independent Poisson increments over non-overlapping segments."""

    ordered = tuple(sorted(segments, key=lambda item: item.start_time))
    for first, second in zip(ordered[:-1], ordered[1:], strict=True):
        if first.end_time > second.start_time + 1e-12:
            raise ValueError("load segments must not overlap")
    rng = np.random.default_rng(seed)
    arrivals: list[float] = []
    for segment in ordered:
        if segment.arrival_rate == 0.0:
            continue
        time = segment.start_time
        while True:
            time += float(rng.exponential(1.0 / segment.arrival_rate))
            if time >= segment.end_time:
                break
            arrivals.append(time)
    return np.asarray(arrivals, dtype=float)


def piecewise_deterministic_arrival_times(
    segments: Sequence[LoadSegment],
) -> FloatArray:
    """Generate regular arrivals separately inside each load segment."""

    ordered = tuple(sorted(segments, key=lambda item: item.start_time))
    for first, second in zip(ordered[:-1], ordered[1:], strict=True):
        if first.end_time > second.start_time + 1e-12:
            raise ValueError("load segments must not overlap")
    arrivals: list[float] = []
    for segment in ordered:
        if segment.arrival_rate == 0.0:
            continue
        interval = 1.0 / segment.arrival_rate
        time = segment.start_time + interval
        while time < segment.end_time - 1e-12:
            arrivals.append(time)
            time += interval
    return np.asarray(arrivals, dtype=float)


def poisson_crossing_requests(
    rate: float,
    duration: float,
    crossing_duration: float | tuple[float, float],
    *,
    seed: int | None = None,
    start_time: float = 0.0,
    source_id_start: int = 0,
    crossing_id: int = 0,
) -> tuple[CrossingRequest, ...]:
    """Generate Poisson crossing requests with fixed or uniform durations."""

    rng = np.random.default_rng(seed)
    if not np.isfinite(rate) or rate < 0.0:
        raise ValueError("rate must be finite and nonnegative")
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be finite and strictly positive")
    if not np.isfinite(start_time) or not 0.0 <= start_time < duration:
        raise ValueError("start_time must lie in [0, duration)")
    if (
        isinstance(crossing_id, (bool, np.bool_))
        or not isinstance(crossing_id, (int, np.integer))
        or crossing_id < 0
    ):
        raise ValueError("crossing_id must be a nonnegative integer")
    request_times: list[float] = []
    if rate > 0.0:
        request_time = start_time
        while True:
            request_time += float(rng.exponential(1.0 / rate))
            if request_time >= duration:
                break
            request_times.append(request_time)
    times = np.asarray(request_times, dtype=float)
    if isinstance(crossing_duration, tuple):
        low, high = map(float, crossing_duration)
        if not np.isfinite(low) or not np.isfinite(high) or not 0.0 < low <= high:
            raise ValueError("crossing_duration bounds must satisfy 0 < low <= high")
        durations = rng.uniform(low, high, size=times.size)
    else:
        value = float(crossing_duration)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("crossing_duration must be finite and positive")
        durations = np.full(times.size, value, dtype=float)
    return tuple(
        CrossingRequest(
            float(time),
            float(service),
            source_id_start + index,
            int(crossing_id),
        )
        for index, (time, service) in enumerate(
            zip(times, durations, strict=True)
        )
    )


def poisson_crossing_requests_multi(
    rate_per_crossing: float,
    duration: float,
    crossing_duration: float | tuple[float, float],
    crossing_count: int,
    *,
    seed: int | None = None,
    start_time: float = 0.0,
    source_id_start: int = 0,
) -> tuple[CrossingRequest, ...]:
    """Generate independent Poisson request streams at several crossings."""

    if (
        isinstance(crossing_count, (bool, np.bool_))
        or not isinstance(crossing_count, (int, np.integer))
        or crossing_count < 1
    ):
        raise ValueError("crossing_count must be a positive integer")
    seed_rng = np.random.default_rng(seed)
    requests: list[CrossingRequest] = []
    next_source = int(source_id_start)
    for crossing_id in range(int(crossing_count)):
        local_seed = int(seed_rng.integers(0, np.iinfo(np.int64).max))
        local = poisson_crossing_requests(
            rate_per_crossing,
            duration,
            crossing_duration,
            seed=local_seed,
            start_time=start_time,
            source_id_start=next_source,
            crossing_id=crossing_id,
        )
        requests.extend(local)
        next_source += len(local)
    return tuple(sorted(requests, key=lambda item: (item.request_time, item.source_id)))


def _resolve_lineage(
    mutable_events: dict[int, _MutableEvent],
) -> dict[int, tuple[int | None, int | None]]:
    resolved: dict[int, tuple[int | None, int | None]] = {}

    def visit(event_id: int, stack: set[int]) -> tuple[int | None, int | None]:
        if event_id in resolved:
            return resolved[event_id]
        if event_id in stack:
            resolved[event_id] = (None, None)
            return resolved[event_id]
        record = mutable_events[event_id]
        if record.cause == "crossing":
            resolved[event_id] = (event_id, 0)
            return resolved[event_id]
        if record.parent_event_id is None:
            resolved[event_id] = (None, None)
            return resolved[event_id]
        parent_root, parent_generation = visit(
            record.parent_event_id, stack | {event_id}
        )
        if parent_root is None or parent_generation is None:
            resolved[event_id] = (None, None)
        else:
            resolved[event_id] = (parent_root, parent_generation + 1)
        return resolved[event_id]

    for key in mutable_events:
        visit(key, set())
    return resolved


def _can_activate_crossing(
    config: OpenCorridorConfig,
    crossing_x: float,
    positions: FloatArray,
    speeds: FloatArray,
) -> bool:
    if positions.size == 0:
        return True
    physical_clearance = config.robot_length / 2.0 + config.crossing_half_width
    if np.any(np.abs(positions - crossing_x) < physical_clearance):
        return False
    approaching = np.flatnonzero(positions < crossing_x)
    if approaching.size == 0:
        return True
    nearest = int(approaching[np.argmax(positions[approaching])])
    gap = crossing_x - float(positions[nearest])
    required = float(
        d_stop(
            speeds[nearest],
            config.crossing_clearance,
            config.reaction_time,
            config.braking,
        )
    )
    return gap + config.collision_tolerance >= required


def _select_crossing_gate_robot(
    config: OpenCorridorConfig,
    crossing_x: float,
    robot_ids: IntArray,
    positions: FloatArray,
    speeds: FloatArray,
) -> int | None:
    """Select the closest upstream robot that can safely honor a reservation.

    Robots already too close to stop are allowed to clear the crossing.  A
    farther robot is reserved and brought to the stop line; its followers then
    respond through their ordinary leader shields.  This prevents pedestrian
    starvation under a dense platoon without inserting an unsafe obstacle.
    """

    approaching = np.flatnonzero(positions < crossing_x)
    if approaching.size == 0:
        return None
    gaps = crossing_x - positions[approaching]
    required = np.asarray(
        d_stop(
            speeds[approaching],
            config.crossing_clearance,
            config.reaction_time,
            config.braking,
        ),
        dtype=float,
    )
    eligible = approaching[
        gaps + config.collision_tolerance >= required
    ]
    if eligible.size == 0:
        return None
    selected = int(eligible[np.argmax(positions[eligible])])
    return int(robot_ids[selected])


def _can_admit(
    config: OpenCorridorConfig,
    positions: FloatArray,
    active_crossing_positions: FloatArray,
) -> bool:
    if np.any(
        active_crossing_positions
        <= config.robot_length / 2.0 + config.crossing_half_width
    ):
        return False
    if positions.size == 0:
        return True
    required = float(
        d_stop(
            config.admission_speed,
            config.robot_clearance,
            config.reaction_time,
            config.braking,
        )
    )
    return float(positions[-1]) + config.collision_tolerance >= required


def simulate_open_corridor(
    config: OpenCorridorConfig,
    duration: float,
    arrival_times: ArrayLike,
    *,
    crossing_requests: Iterable[CrossingRequest] = (),
    record_stride: int = 1,
) -> OpenSimulationResult:
    """Simulate a FIFO open corridor from an empty initial condition.

    The time loop is fixed step.  Arrival and crossing-request clocks are
    event driven.  All robot dynamics and leader constraints are vectorized.
    Every integration step asserts physical separation, stopping-distance
    invariance, crossing exclusion, and mass conservation.
    """

    step_count = _validate_duration(duration, config.dt)
    arrivals = _validate_arrival_times(arrival_times, duration)
    crossing_positions = config.crossing_positions
    crossing_count = config.crossing_count
    requests = _validate_requests(crossing_requests, duration, crossing_count)
    if record_stride < 1:
        raise ValueError("record_stride must be at least one")

    robot_ids = np.empty(0, dtype=int)
    job_ids = np.empty(0, dtype=int)
    positions = np.empty(0, dtype=float)
    speeds = np.empty(0, dtype=float)
    job_arrival_times = np.empty(0, dtype=float)
    admission_times = np.empty(0, dtype=float)
    severity_by_robot = np.empty(0, dtype=float)
    active_event_ids = np.empty(0, dtype=int)

    arrival_index = 0
    request_index = 0
    next_robot_id = 0
    next_event_id = 0
    queued_job_ids: list[int] = []
    queued_arrival_times: list[float] = []
    pending_requests: list[list[CrossingRequest]] = [
        [] for _ in range(crossing_count)
    ]
    active_crossings: list[CrossingRequest | None] = [
        None for _ in range(crossing_count)
    ]
    active_crossing_starts: list[float | None] = [
        None for _ in range(crossing_count)
    ]
    crossing_gate_robot_ids: list[int | None] = [
        None for _ in range(crossing_count)
    ]
    completed: list[CompletedTraversal] = []
    crossing_services: list[CrossingServiceRecord] = []
    mutable_events: dict[int, _MutableEvent] = {}
    available_robot_ids: deque[int] = deque(
        range(config.fleet_size) if config.fleet_size is not None else ()
    )
    returning_robots: list[tuple[float, int]] = []

    total_admissions = 0
    cumulative_severity = 0.0
    collision_count = 0
    crossing_violation_count = 0

    time_log: list[float] = [0.0]
    queue_log: list[int] = [0]
    wip_log: list[int] = [0]
    available_log: list[int] = [
        -1 if config.fleet_size is None else config.fleet_size
    ]
    returning_log: list[int] = [0]
    arrivals_log: list[int] = [0]
    admissions_log: list[int] = [0]
    completions_log: list[int] = [0]
    severity_log: list[float] = [0.0]
    mean_speed_log: list[float] = [np.nan]
    active_events_log: list[int] = [0]
    crossing_active_log: list[bool] = [False]
    crossing_reserved_log: list[bool] = [False]
    pending_crossings_log: list[int] = [0]
    active_crossing_count_log: list[int] = [0]
    reserved_crossing_count_log: list[int] = [0]

    for step in range(step_count):
        time = step * config.dt

        if config.fleet_size is not None:
            while returning_robots and returning_robots[0][0] <= time + 1e-12:
                _, returned_robot_id = heapq.heappop(returning_robots)
                available_robot_ids.append(returned_robot_id)

        while arrival_index < arrivals.size and arrivals[arrival_index] <= time + 1e-12:
            queued_job_ids.append(arrival_index)
            queued_arrival_times.append(float(arrivals[arrival_index]))
            arrival_index += 1

        while (
            request_index < len(requests)
            and requests[request_index].request_time <= time + 1e-12
        ):
            request = requests[request_index]
            pending_requests[request.crossing_id].append(request)
            request_index += 1

        for crossing_id, crossing_x in enumerate(crossing_positions):
            active = active_crossings[crossing_id]
            active_start = active_crossing_starts[crossing_id]
            if active is not None:
                assert active_start is not None
                planned_release = active_start + active.duration
                if time >= planned_release - 1e-12:
                    crossing_services.append(
                        CrossingServiceRecord(
                            source_id=active.source_id,
                            crossing_id=crossing_id,
                            request_time=active.request_time,
                            requested_duration=active.duration,
                            activation_time=active_start,
                            # Release is the first grid point not before plan.
                            release_time=time,
                            status="completed",
                        )
                    )
                    active_crossings[crossing_id] = None
                    active_crossing_starts[crossing_id] = None
                    crossing_gate_robot_ids[crossing_id] = None

            if active_crossings[crossing_id] is None and pending_requests[crossing_id]:
                if _can_activate_crossing(
                    config, float(crossing_x), positions, speeds
                ):
                    active_crossings[crossing_id] = pending_requests[crossing_id].pop(0)
                    active_crossing_starts[crossing_id] = time
                    crossing_gate_robot_ids[crossing_id] = None
                else:
                    gate_robot_id = crossing_gate_robot_ids[crossing_id]
                    gate_is_present = (
                        gate_robot_id is not None
                        and np.any(robot_ids == gate_robot_id)
                    )
                    if not gate_is_present:
                        crossing_gate_robot_ids[crossing_id] = (
                            _select_crossing_gate_robot(
                                config,
                                float(crossing_x),
                                robot_ids,
                                positions,
                                speeds,
                            )
                        )

        crossing_active_flags = np.asarray(
            [item is not None for item in active_crossings], dtype=bool
        )
        crossing_reserved_flags = crossing_active_flags | np.asarray(
            [item is not None for item in crossing_gate_robot_ids], dtype=bool
        )
        crossing_active = bool(np.any(crossing_active_flags))
        crossing_reserved = bool(np.any(crossing_reserved_flags))
        active_crossing_positions = crossing_positions[crossing_active_flags]

        robot_is_available = (
            config.fleet_size is None or bool(available_robot_ids)
        )
        if (
            queued_job_ids
            and robot_is_available
            and _can_admit(config, positions, active_crossing_positions)
        ):
            admitted_job = queued_job_ids.pop(0)
            admitted_arrival = queued_arrival_times.pop(0)
            if config.fleet_size is None:
                admitted_robot_id = next_robot_id
                next_robot_id += 1
            else:
                admitted_robot_id = available_robot_ids.popleft()
            robot_ids = np.append(robot_ids, admitted_robot_id)
            job_ids = np.append(job_ids, admitted_job)
            positions = np.append(positions, 0.0)
            speeds = np.append(speeds, config.admission_speed)
            job_arrival_times = np.append(job_arrival_times, admitted_arrival)
            admission_times = np.append(admission_times, time)
            severity_by_robot = np.append(severity_by_robot, 0.0)
            active_event_ids = np.append(active_event_ids, -1)
            total_admissions += 1

        count = positions.size
        free_next_speeds = np.minimum(
            config.desired_speed, speeds + config.acceleration * config.dt
        )
        safe_cap = np.full(count, config.desired_speed, dtype=float)
        blocker_kind = np.full(count, int(BlockerKind.NONE), dtype=np.int8)
        blocker_robot = np.full(count, -1, dtype=int)
        blocker_crossing_source = np.full(count, -1, dtype=int)
        blocker_crossing_id = np.full(count, -1, dtype=int)

        if count > 1:
            follower_index = np.arange(1, count, dtype=int)
            gaps = positions[:-1] - positions[1:]
            visible = gaps <= config.sensor_range
            constrained_followers = follower_index[visible]
            if constrained_followers.size:
                caps = np.asarray(
                    v_safe_next_step(
                        gaps[visible],
                        speeds[constrained_followers],
                        config.robot_clearance,
                        config.reaction_time,
                        config.braking,
                        config.dt,
                    ),
                    dtype=float,
                )
                safe_cap[constrained_followers] = np.minimum(
                    safe_cap[constrained_followers], caps
                )
                active_constraints = caps < (
                    free_next_speeds[constrained_followers]
                    - config.intervention_tolerance
                )
                selected = constrained_followers[active_constraints]
                blocker_kind[selected] = int(BlockerKind.ROBOT)
                blocker_robot[selected] = robot_ids[selected - 1]

        crossing_constraints: list[tuple[int, int, float]] = []
        for crossing_id, crossing_x in enumerate(crossing_positions):
            constraint_index: int | None = None
            if active_crossings[crossing_id] is not None and count:
                approaching = np.flatnonzero(positions < crossing_x)
                if approaching.size:
                    constraint_index = int(
                        approaching[np.argmax(positions[approaching])]
                    )
            elif crossing_gate_robot_ids[crossing_id] is not None and count:
                gate_matches = np.flatnonzero(
                    robot_ids == crossing_gate_robot_ids[crossing_id]
                )
                if gate_matches.size != 1:
                    raise AssertionError("reserved crossing gate robot was lost")
                constraint_index = int(gate_matches[0])
            if constraint_index is None:
                continue
            constraint_gap = float(crossing_x - positions[constraint_index])
            crossing_constraints.append(
                (crossing_id, constraint_index, constraint_gap)
            )
            if constraint_gap <= config.sensor_range:
                crossing_cap = float(
                    v_safe_next_step(
                        constraint_gap,
                        speeds[constraint_index],
                        config.crossing_clearance,
                        config.reaction_time,
                        config.braking,
                        config.dt,
                    )
                )
                if crossing_cap <= safe_cap[constraint_index]:
                    safe_cap[constraint_index] = crossing_cap
                    if crossing_cap < (
                        free_next_speeds[constraint_index]
                        - config.intervention_tolerance
                    ):
                        blocker_kind[constraint_index] = int(
                            BlockerKind.CROSSING
                        )
                        blocker_robot[constraint_index] = -1
                        blocker_crossing_id[constraint_index] = crossing_id
                        active = active_crossings[crossing_id]
                        source = (
                            active.source_id
                            if active is not None
                            else pending_requests[crossing_id][0].source_id
                        )
                        blocker_crossing_source[constraint_index] = source

        minimum_reachable = np.maximum(0.0, speeds - config.braking * config.dt)
        if np.any(
            safe_cap < minimum_reachable - config.collision_tolerance
        ):
            bad = int(
                np.flatnonzero(
                    safe_cap < minimum_reachable - config.collision_tolerance
                )[0]
            )
            raise AssertionError(
                "shield next-speed cap is unreachable under the configured "
                f"braking limit for robot {int(robot_ids[bad])}"
            )

        requested_acceleration = (safe_cap - speeds) / config.dt
        applied_acceleration = np.clip(
            requested_acceleration, -config.braking, config.acceleration
        )
        next_speeds = np.maximum(0.0, speeds + applied_acceleration * config.dt)
        movement = np.maximum(
            0.0,
            speeds * config.dt + 0.5 * applied_acceleration * config.dt**2,
        )

        next_positions = positions + movement
        physical_clearance = config.robot_length / 2.0 + config.crossing_half_width
        for crossing_x in active_crossing_positions:
            if np.any(
                np.abs(next_positions - crossing_x)
                < physical_clearance - config.collision_tolerance
            ):
                crossing_violation_count += 1
                raise AssertionError(
                    "occupied-crossing violation: a robot entered a protected zone"
                )
        if count > 1:
            next_gaps = next_positions[:-1] - next_positions[1:]
            required = np.asarray(
                d_stop(
                    next_speeds[1:],
                    config.robot_clearance,
                    config.reaction_time,
                    config.braking,
                ),
                dtype=float,
            )
            violations = next_gaps < required - config.collision_tolerance
            if np.any(violations):
                follower = int(np.flatnonzero(violations)[0] + 1)
                raise AssertionError(
                    "robot stopping-distance invariant violated after step "
                    f"{step}: robot {int(robot_ids[follower])}, "
                    f"gap={next_gaps[follower - 1]:.6f}, "
                    f"required={required[follower - 1]:.6f}"
                )
            overlaps = next_gaps < (
                config.robot_length - config.collision_tolerance
            )
            if np.any(overlaps):
                collision_count += int(np.count_nonzero(overlaps))
                follower = int(np.flatnonzero(overlaps)[0] + 1)
                raise AssertionError(
                    f"robot collision after step {step}: robot "
                    f"{int(robot_ids[follower])}"
                )

        for crossing_id, constraint_index, constraint_gap in crossing_constraints:
            if not crossing_reserved_flags[crossing_id]:
                continue
            gap_after = constraint_gap - float(movement[constraint_index])
            required_after = float(
                d_stop(
                    next_speeds[constraint_index],
                    config.crossing_clearance,
                    config.reaction_time,
                    config.braking,
                )
            )
            if gap_after < required_after - config.collision_tolerance:
                raise AssertionError(
                    "crossing reservation invariant violated after step "
                    f"{step} at crossing {crossing_id}: gap={gap_after:.6f}, "
                    f"required={required_after:.6f}"
                )

        constrained_now = safe_cap < (
            free_next_speeds - config.intervention_tolerance
        )
        step_intervening = np.zeros(count, dtype=bool)

        # Front-to-back processing makes same-step parent assignment well founded:
        # a follower's immediate leader has already been updated.
        for index in range(count):
            existing_id = int(active_event_ids[index])
            current_cause: EventCause | None = None
            current_blocker: int | None = None
            current_parent: int | None = None
            current_source: int | None = None
            current_crossing: int | None = None
            if constrained_now[index]:
                kind = BlockerKind(int(blocker_kind[index]))
                if kind == BlockerKind.CROSSING:
                    current_cause = "crossing"
                    source = int(blocker_crossing_source[index])
                    current_source = None if source < 0 else source
                    crossing_id = int(blocker_crossing_id[index])
                    current_crossing = None if crossing_id < 0 else crossing_id
                elif kind == BlockerKind.ROBOT:
                    current_blocker = int(blocker_robot[index])
                    leader_event = int(active_event_ids[index - 1])
                    if index > 0 and leader_event >= 0:
                        current_cause = "robot"
                        current_parent = leader_event
                    else:
                        current_cause = "background"
                else:
                    current_cause = "background"

            if existing_id >= 0 and constrained_now[index]:
                existing = mutable_events[existing_id]
                signature_changed = (
                    existing.cause != current_cause
                    or existing.blocker_robot_id != current_blocker
                    or existing.parent_event_id != current_parent
                    or existing.crossing_source_id != current_source
                    or existing.crossing_id != current_crossing
                )
                if signature_changed:
                    existing.end_time = time
                    active_event_ids[index] = -1
                    existing_id = -1

            if existing_id < 0 and constrained_now[index]:
                event_id = next_event_id
                next_event_id += 1
                active_event_ids[index] = event_id
                mutable_events[event_id] = _MutableEvent(
                    event_id=event_id,
                    robot_id=int(robot_ids[index]),
                    start_time=time,
                    cause="background" if current_cause is None else current_cause,
                    blocker_robot_id=current_blocker,
                    parent_event_id=current_parent,
                    crossing_source_id=current_source,
                    start_position=float(positions[index]),
                    crossing_id=current_crossing,
                )
                existing_id = event_id

            if existing_id >= 0:
                step_intervening[index] = True

        severity = np.clip(1.0 - next_speeds / config.desired_speed, 0.0, 1.0)
        severity_increment = severity * config.dt * step_intervening
        severity_by_robot += severity_increment
        cumulative_severity += float(np.sum(severity_increment))
        for index in np.flatnonzero(step_intervening):
            event_id = int(active_event_ids[index])
            record = mutable_events[event_id]
            record.severity_loss += float(severity[index] * config.dt)
            record.minimum_speed = min(record.minimum_speed, float(next_speeds[index]))

        recovered = (
            (active_event_ids >= 0)
            & ~constrained_now
            & (
                next_speeds
                >= config.desired_speed - config.intervention_tolerance
            )
        )
        for index in np.flatnonzero(recovered):
            event_id = int(active_event_ids[index])
            mutable_events[event_id].end_time = time + config.dt
            active_event_ids[index] = -1

        old_positions = positions
        positions = next_positions
        speeds = next_speeds

        completed_mask = positions >= config.corridor_length - 1e-12
        completed_count_this_step = int(np.count_nonzero(completed_mask))
        if completed_count_this_step:
            if not np.all(completed_mask[:completed_count_this_step]) or np.any(
                completed_mask[completed_count_this_step:]
            ):
                raise AssertionError("route order changed before corridor exit")
            for index in range(completed_count_this_step):
                distance_this_step = float(movement[index])
                if distance_this_step <= 0.0:
                    completion_time = time + config.dt
                else:
                    fraction = np.clip(
                        (config.corridor_length - float(old_positions[index]))
                        / distance_this_step,
                        0.0,
                        1.0,
                    )
                    completion_time = time + float(fraction) * config.dt
                completed.append(
                    CompletedTraversal(
                        job_id=int(job_ids[index]),
                        robot_id=int(robot_ids[index]),
                        arrival_time=float(job_arrival_times[index]),
                        admission_time=float(admission_times[index]),
                        completion_time=completion_time,
                        severity_loss=float(severity_by_robot[index]),
                    )
                )
                event_id = int(active_event_ids[index])
                if event_id >= 0:
                    mutable_events[event_id].end_time = completion_time
                    # Exit is a fully observed causal termination: the robot can
                    # no longer generate downstream offspring after departure.
                    mutable_events[event_id].censored = False
                if config.fleet_size is not None:
                    heapq.heappush(
                        returning_robots,
                        (
                            completion_time + config.empty_return_time,
                            int(robot_ids[index]),
                        ),
                    )

            retained = slice(completed_count_this_step, None)
            robot_ids = robot_ids[retained]
            job_ids = job_ids[retained]
            positions = positions[retained]
            speeds = speeds[retained]
            job_arrival_times = job_arrival_times[retained]
            admission_times = admission_times[retained]
            severity_by_robot = severity_by_robot[retained]
            active_event_ids = active_event_ids[retained]

        if config.fleet_size is not None:
            interval_end = (step + 1) * config.dt
            while (
                returning_robots
                and returning_robots[0][0] <= interval_end + 1e-12
            ):
                _, returned_robot_id = heapq.heappop(returning_robots)
                available_robot_ids.append(returned_robot_id)

        # Events arriving inside this integration interval join their queues at
        # the right endpoint.  They become service-eligible next step, but are
        # already part of the system population and final-time mass balance.
        interval_end = (step + 1) * config.dt
        while (
            arrival_index < arrivals.size
            and arrivals[arrival_index] <= interval_end + 1e-12
        ):
            queued_job_ids.append(arrival_index)
            queued_arrival_times.append(float(arrivals[arrival_index]))
            arrival_index += 1
        while (
            request_index < len(requests)
            and requests[request_index].request_time <= interval_end + 1e-12
        ):
            request = requests[request_index]
            pending_requests[request.crossing_id].append(request)
            request_index += 1

        expected_population = arrival_index - len(completed)
        observed_population = len(queued_job_ids) + positions.size
        if expected_population != observed_population:
            raise AssertionError(
                "job mass conservation failed inside the simulation: "
                f"expected {expected_population}, observed {observed_population}"
            )
        if config.fleet_size is not None:
            observed_fleet = (
                positions.size + len(available_robot_ids) + len(returning_robots)
            )
            if observed_fleet != config.fleet_size:
                raise AssertionError(
                    "physical fleet conservation failed inside the simulation: "
                    f"expected {config.fleet_size}, observed {observed_fleet}"
                )

        if (step + 1) % record_stride == 0 or step + 1 == step_count:
            log_time = interval_end
            visible_arrivals = int(np.searchsorted(arrivals, log_time, side="right"))
            time_log.append(log_time)
            queue_log.append(len(queued_job_ids))
            wip_log.append(int(positions.size))
            available_log.append(
                -1 if config.fleet_size is None else len(available_robot_ids)
            )
            returning_log.append(len(returning_robots))
            arrivals_log.append(visible_arrivals)
            admissions_log.append(total_admissions)
            completions_log.append(len(completed))
            severity_log.append(cumulative_severity)
            mean_speed_log.append(float(np.mean(speeds)) if speeds.size else np.nan)
            active_events_log.append(int(np.count_nonzero(active_event_ids >= 0)))
            crossing_active_log.append(crossing_active)
            crossing_reserved_log.append(crossing_reserved)
            pending_crossings_log.append(sum(map(len, pending_requests)))
            active_crossing_count_log.append(
                int(np.count_nonzero(crossing_active_flags))
            )
            reserved_crossing_count_log.append(
                int(np.count_nonzero(crossing_reserved_flags))
            )

    for crossing_id, active in enumerate(active_crossings):
        if active is None:
            continue
        active_start = active_crossing_starts[crossing_id]
        assert active_start is not None
        crossing_services.append(
            CrossingServiceRecord(
                source_id=active.source_id,
                crossing_id=crossing_id,
                request_time=active.request_time,
                requested_duration=active.duration,
                activation_time=active_start,
                release_time=duration,
                status="active_at_end",
            )
        )
    crossing_services.extend(
        CrossingServiceRecord(
            source_id=request.source_id,
            crossing_id=request.crossing_id,
            request_time=request.request_time,
            requested_duration=request.duration,
            activation_time=None,
            release_time=None,
            status="unserved",
        )
        for crossing_queue in pending_requests
        for request in crossing_queue
    )
    # Requests occurring inside the final integration interval are valid input
    # but have no sampled activation opportunity.
    crossing_services.extend(
        CrossingServiceRecord(
            source_id=request.source_id,
            crossing_id=request.crossing_id,
            request_time=request.request_time,
            requested_duration=request.duration,
            activation_time=None,
            release_time=None,
            status="unserved",
        )
        for request in requests[request_index:]
    )
    crossing_services.sort(key=lambda item: (item.request_time, item.source_id))

    if config.fleet_size is not None:
        while returning_robots and returning_robots[0][0] <= duration + 1e-12:
            _, returned_robot_id = heapq.heappop(returning_robots)
            available_robot_ids.append(returned_robot_id)

    for event_id in np.unique(active_event_ids[active_event_ids >= 0]):
        mutable_events[int(event_id)].end_time = duration
        mutable_events[int(event_id)].censored = True

    lineage = _resolve_lineage(mutable_events)
    finalized_events: list[InterventionEvent] = []
    for event_id in sorted(mutable_events):
        record = mutable_events[event_id]
        root, generation = lineage[event_id]
        finalized_events.append(
            InterventionEvent(
                event_id=event_id,
                robot_id=record.robot_id,
                event_type=0,
                start_time=record.start_time,
                end_time=duration if record.end_time is None else record.end_time,
                cause=record.cause,
                blocker_robot_id=record.blocker_robot_id,
                parent_event_id=record.parent_event_id,
                root_primary_event_id=root,
                generation=generation,
                crossing_source_id=record.crossing_source_id,
                severity_loss=record.severity_loss,
                minimum_speed=(
                    config.desired_speed
                    if not np.isfinite(record.minimum_speed)
                    else record.minimum_speed
                ),
                censored=record.censored,
                start_position=record.start_position,
                crossing_id=record.crossing_id,
            )
        )

    result = OpenSimulationResult(
        config=config,
        duration=duration,
        arrival_times=arrivals,
        crossing_requests=requests,
        crossing_services=tuple(crossing_services),
        completed_traversals=tuple(completed),
        events=tuple(finalized_events),
        trajectory=OpenTrajectoryLog(
            time=np.asarray(time_log, dtype=float),
            queue_length=np.asarray(queue_log, dtype=int),
            work_in_process=np.asarray(wip_log, dtype=int),
            available_robots=np.asarray(available_log, dtype=int),
            returning_robots=np.asarray(returning_log, dtype=int),
            cumulative_arrivals=np.asarray(arrivals_log, dtype=int),
            cumulative_admissions=np.asarray(admissions_log, dtype=int),
            cumulative_completions=np.asarray(completions_log, dtype=int),
            cumulative_severity_loss=np.asarray(severity_log, dtype=float),
            mean_speed=np.asarray(mean_speed_log, dtype=float),
            active_interventions=np.asarray(active_events_log, dtype=int),
            crossing_active=np.asarray(crossing_active_log, dtype=bool),
            crossing_reserved=np.asarray(crossing_reserved_log, dtype=bool),
            pending_crossings=np.asarray(pending_crossings_log, dtype=int),
            active_crossing_count=np.asarray(
                active_crossing_count_log, dtype=int
            ),
            reserved_crossing_count=np.asarray(
                reserved_crossing_count_log, dtype=int
            ),
        ),
        final_state=OpenFinalState(
            robot_id=robot_ids.copy(),
            job_id=job_ids.copy(),
            position=positions.copy(),
            speed=speeds.copy(),
            arrival_time=job_arrival_times.copy(),
            admission_time=admission_times.copy(),
            severity_loss=severity_by_robot.copy(),
            queued_job_id=np.asarray(queued_job_ids, dtype=int),
            queued_arrival_time=np.asarray(queued_arrival_times, dtype=float),
            available_robot_id=np.asarray(available_robot_ids, dtype=int),
            returning_robot_id=np.asarray(
                [item[1] for item in returning_robots], dtype=int
            ),
            return_ready_time=np.asarray(
                [item[0] for item in returning_robots], dtype=float
            ),
        ),
        total_admissions=total_admissions,
        total_severity_loss=cumulative_severity,
        collision_count=collision_count,
        crossing_violation_count=crossing_violation_count,
    )
    if result.mass_balance_residual != 0:
        raise AssertionError(
            f"final job mass balance residual is {result.mass_balance_residual}"
        )
    if result.fleet_balance_residual not in (None, 0):
        raise AssertionError(
            f"final fleet balance residual is {result.fleet_balance_residual}"
        )
    return result


def run_open_paired_rollout(
    config: OpenCorridorConfig,
    duration: float,
    arrival_times: ArrayLike,
    crossing_requests: Iterable[CrossingRequest],
    *,
    record_stride: int = 1,
) -> OpenPairedRolloutResult:
    """Run a disturbance realization and an exact no-crossing counterfactual."""

    arrivals = np.asarray(arrival_times, dtype=float).copy()
    requests = tuple(crossing_requests)
    treated = simulate_open_corridor(
        config,
        duration,
        arrivals,
        crossing_requests=requests,
        record_stride=record_stride,
    )
    control = simulate_open_corridor(
        config,
        duration,
        arrivals,
        crossing_requests=(),
        record_stride=record_stride,
    )
    return OpenPairedRolloutResult(treated=treated, control=control)
