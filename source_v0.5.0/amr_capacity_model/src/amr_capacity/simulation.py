"""Deterministic fixed-step simulation for a shield-coupled AMR corridor.

The first simulator is deliberately route constrained. Robot state is updated
as NumPy arrays, while a rectangular centerline provides a 2D embedding. This
isolates cascade mechanics before merges, routing, or deadlock resolution are
introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .capacity_theory import d_stop, v_safe_next_step
from .geometry import RectangularLoop


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
EventCause = Literal["crossing", "robot", "background"]


class BlockerKind(IntEnum):
    NONE = 0
    ROBOT = 1
    CROSSING = 2


class UnsafeCrossingScheduleError(ValueError):
    """Raised when an obstacle is introduced inside stopping distance."""


@dataclass(frozen=True)
class CorridorConfig:
    """Physical and numerical parameters for a one-way periodic corridor."""

    loop: RectangularLoop
    desired_speed: float
    acceleration: float
    braking: float
    reaction_time: float
    dt: float
    robot_length: float
    safety_margin: float
    sensor_range: float
    crossing_s: float
    crossing_half_width: float = 0.4
    intervention_tolerance: float = 1e-4
    collision_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        positive = {
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
            "intervention_tolerance": self.intervention_tolerance,
            "collision_tolerance": self.collision_tolerance,
        }
        for name, value in nonnegative.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.reaction_time < self.dt:
            raise ValueError("reaction_time must be at least one integration step")
        if not np.isfinite(self.crossing_s):
            raise ValueError("crossing_s must be finite")
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
    def route_length(self) -> float:
        return self.loop.perimeter

    @property
    def robot_clearance(self) -> float:
        """Required center-to-center static clearance for two robots."""

        return self.robot_length + self.safety_margin

    @property
    def crossing_clearance(self) -> float:
        """Required robot-center clearance from an occupied crossing."""

        return self.robot_length / 2.0 + self.crossing_half_width + self.safety_margin

    @property
    def normalized_crossing_s(self) -> float:
        return float(self.crossing_s % self.route_length)


@dataclass(frozen=True)
class CrossingEvent:
    """Exogenous interval during which the route crossing is occupied."""

    start_time: float
    duration: float
    source_id: int = 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.start_time) or self.start_time < 0.0:
            raise ValueError("start_time must be finite and nonnegative")
        if not np.isfinite(self.duration) or self.duration <= 0.0:
            raise ValueError("duration must be finite and strictly positive")
        if not isinstance(self.source_id, (int, np.integer)):
            raise ValueError("source_id must be an integer")

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration


@dataclass(frozen=True)
class FleetState:
    """Initial route positions and longitudinal speeds."""

    s: FloatArray
    speed: FloatArray


@dataclass(frozen=True)
class InterventionEvent:
    """One contiguous interval of shield-induced speed loss."""

    event_id: int
    robot_id: int
    event_type: int
    start_time: float
    end_time: float
    cause: EventCause
    blocker_robot_id: int | None
    parent_event_id: int | None
    root_primary_event_id: int | None
    generation: int | None
    crossing_source_id: int | None
    severity_loss: float
    minimum_speed: float
    censored: bool
    start_position: float | None = None
    crossing_id: int | None = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def is_primary(self) -> bool:
        return self.cause == "crossing"

    @property
    def is_causal(self) -> bool:
        return self.root_primary_event_id is not None


@dataclass(frozen=True)
class TrajectoryLog:
    """Strided simulation snapshots."""

    time: FloatArray
    s: FloatArray
    speed: FloatArray
    safe_speed: FloatArray
    intervening: BoolArray
    blocker_kind: NDArray[np.int8]
    blocker_robot_id: IntArray
    crossing_active: BoolArray

    def xy_heading(
        self, loop: RectangularLoop
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        x, y, heading = loop.to_xy_heading(self.s)
        return (
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            np.asarray(heading, dtype=float),
        )


@dataclass(frozen=True)
class SimulationResult:
    """Complete result used by estimators and experiment scripts."""

    config: CorridorConfig
    duration: float
    initial_state: FleetState
    final_state: FleetState
    events: tuple[InterventionEvent, ...]
    trajectory: TrajectoryLog
    distance_by_robot: FloatArray
    severity_loss_by_robot: FloatArray
    collision_count: int
    crossing_violation_count: int

    @property
    def robot_count(self) -> int:
        return int(self.distance_by_robot.size)

    @property
    def total_traversals(self) -> float:
        return float(np.sum(self.distance_by_robot) / self.config.route_length)

    @property
    def traversal_rate(self) -> float:
        return self.total_traversals / self.duration

    @property
    def total_severity_loss(self) -> float:
        return float(np.sum(self.severity_loss_by_robot))


@dataclass(frozen=True)
class PairedRolloutResult:
    """Event rollout paired with a no-crossing counterfactual."""

    treated: SimulationResult
    control: SimulationResult
    attributable_severity_by_robot: FloatArray
    distance_loss_by_robot: FloatArray

    @property
    def attributable_severity(self) -> float:
        return float(np.sum(self.attributable_severity_by_robot))

    @property
    def traversal_loss(self) -> float:
        return float(
            np.sum(self.distance_loss_by_robot) / self.treated.config.route_length
        )


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


def uniform_initial_state(
    config: CorridorConfig,
    robot_count: int,
    *,
    offset: float = 0.0,
    initial_speed: float | None = None,
) -> FleetState:
    """Create an equally spaced, desired-speed-safe fleet state."""

    if robot_count < 1:
        raise ValueError("robot_count must be at least one")
    if not np.isfinite(offset):
        raise ValueError("offset must be finite")
    speed_value = config.desired_speed if initial_speed is None else float(initial_speed)
    if not np.isfinite(speed_value) or speed_value < 0.0:
        raise ValueError("initial_speed must be finite and nonnegative")
    if speed_value > config.desired_speed + config.intervention_tolerance:
        raise ValueError("initial_speed must not exceed desired_speed")
    headway = config.route_length / robot_count
    required = float(
        d_stop(
            speed_value,
            config.robot_clearance,
            config.reaction_time,
            config.braking,
        )
    )
    if headway + config.collision_tolerance < required:
        raise ValueError(
            "uniform initial headway is below the static-leader stopping distance"
        )
    positions = np.mod(
        float(offset) + np.arange(robot_count, dtype=float) * headway,
        config.route_length,
    )
    order = np.argsort(positions)
    return FleetState(
        s=np.asarray(positions[order], dtype=float),
        speed=np.full(robot_count, speed_value, dtype=float),
    )


def _validate_initial_state(config: CorridorConfig, state: FleetState) -> FleetState:
    positions = np.asarray(state.s, dtype=float).reshape(-1)
    speeds = np.asarray(state.speed, dtype=float).reshape(-1)
    if positions.size == 0 or positions.size != speeds.size:
        raise ValueError("initial state arrays must be nonempty and have equal length")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(speeds)):
        raise ValueError("initial state must be finite")
    if np.any(speeds < 0.0) or np.any(
        speeds > config.desired_speed + config.intervention_tolerance
    ):
        raise ValueError("initial speeds must lie in [0, desired_speed]")
    positions = np.mod(positions, config.route_length)
    order = np.argsort(positions)
    positions = positions[order]
    speeds = speeds[order]
    leader = np.roll(np.arange(positions.size), -1)
    gaps = _forward_gaps(positions, leader, config.route_length)
    if np.min(gaps) < config.robot_length - config.collision_tolerance:
        raise ValueError("initial state contains overlapping robots")
    required = np.asarray(
        d_stop(
            speeds,
            config.robot_clearance,
            config.reaction_time,
            config.braking,
        )
    )
    if np.any(gaps + config.collision_tolerance < required):
        raise ValueError("initial state violates the stopping-distance invariant")
    return FleetState(s=positions.copy(), speed=speeds.copy())


def _forward_gaps(
    positions: FloatArray, leader: IntArray, route_length: float
) -> FloatArray:
    """Center-to-center forward gaps, including the one-robot ring case."""

    if positions.size == 1:
        return np.full(1, route_length, dtype=float)
    return np.mod(positions[leader] - positions, route_length)


def _validate_crossing_events(
    events: Iterable[CrossingEvent], duration: float, dt: float
) -> tuple[CrossingEvent, ...]:
    ordered = tuple(sorted(events, key=lambda item: item.start_time))
    previous_end = -np.inf
    seen_ids: set[int] = set()
    for event in ordered:
        if event.source_id in seen_ids:
            raise ValueError("crossing source_id values must be unique")
        seen_ids.add(event.source_id)
        if event.start_time < previous_end - 1e-12:
            raise ValueError("crossing events must not overlap")
        if event.end_time > duration + 1e-12:
            raise ValueError("crossing event extends beyond simulation duration")
        first_active_step = np.ceil((event.start_time - 1e-12) / dt) * dt
        if first_active_step >= event.end_time - 1e-12:
            raise ValueError("crossing event is shorter than its sampled time support")
        previous_end = event.end_time
    return ordered


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


def simulate_corridor(
    config: CorridorConfig,
    duration: float,
    *,
    crossing_events: Iterable[CrossingEvent] = (),
    initial_state: FleetState | None = None,
    robot_count: int | None = None,
    initial_offset: float = 0.0,
    record_stride: int = 1,
) -> SimulationResult:
    """Run a fixed-step, fleet-vectorized periodic-corridor simulation."""

    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be finite and strictly positive")
    step_count_float = duration / config.dt
    step_count = int(round(step_count_float))
    if not np.isclose(step_count, step_count_float, rtol=0.0, atol=1e-10):
        raise ValueError("duration must be an integer multiple of dt")
    if record_stride < 1:
        raise ValueError("record_stride must be at least one")
    if initial_state is None:
        if robot_count is None:
            raise ValueError("robot_count is required when initial_state is omitted")
        initial_state = uniform_initial_state(
            config, robot_count, offset=initial_offset
        )
    elif robot_count is not None and robot_count != len(initial_state.s):
        raise ValueError("robot_count disagrees with initial_state")
    state = _validate_initial_state(config, initial_state)
    events = _validate_crossing_events(crossing_events, duration, config.dt)

    positions = state.s.copy()
    speeds = state.speed.copy()
    initial_copy = FleetState(s=positions.copy(), speed=speeds.copy())
    count = positions.size
    leader = np.roll(np.arange(count, dtype=int), -1)
    active_event_id = np.full(count, -1, dtype=int)
    mutable_events: dict[int, _MutableEvent] = {}
    next_intervention_id = 0
    distance_by_robot = np.zeros(count, dtype=float)
    severity_by_robot = np.zeros(count, dtype=float)
    collision_count = 0
    crossing_violation_count = 0

    time_frames: list[float] = [0.0]
    position_frames: list[FloatArray] = [positions.copy()]
    speed_frames: list[FloatArray] = [speeds.copy()]
    safe_speed_frames: list[FloatArray] = [
        np.full(count, config.desired_speed, dtype=float)
    ]
    intervention_frames: list[BoolArray] = [np.zeros(count, dtype=bool)]
    blocker_frames: list[NDArray[np.int8]] = [np.zeros(count, dtype=np.int8)]
    blocker_robot_frames: list[IntArray] = [np.full(count, -1, dtype=int)]
    crossing_frames: list[bool] = [False]

    crossing_index = 0
    current_crossing: CrossingEvent | None = None
    previous_crossing_source: int | None = None

    for step in range(step_count):
        time = step * config.dt
        while crossing_index < len(events) and (
            time >= events[crossing_index].end_time - 1e-12
        ):
            crossing_index += 1
        if crossing_index < len(events):
            candidate = events[crossing_index]
            if candidate.start_time - 1e-12 <= time < candidate.end_time - 1e-12:
                current_crossing = candidate
            else:
                current_crossing = None
        else:
            current_crossing = None

        crossing_source = (
            None if current_crossing is None else current_crossing.source_id
        )
        crossing_active = current_crossing is not None

        gaps = _forward_gaps(positions, leader, config.route_length)
        leader_visible = gaps <= config.sensor_range
        leader_cap = np.full(count, config.desired_speed, dtype=float)
        if np.any(leader_visible):
            leader_cap[leader_visible] = np.asarray(
                v_safe_next_step(
                    gaps[leader_visible],
                    speeds[leader_visible],
                    config.robot_clearance,
                    config.reaction_time,
                    config.braking,
                    config.dt,
                )
            )

        safe_cap = np.minimum(config.desired_speed, leader_cap)
        blocker_kind = np.where(
            leader_cap < config.desired_speed - config.intervention_tolerance,
            int(BlockerKind.ROBOT),
            int(BlockerKind.NONE),
        ).astype(np.int8)
        blocker_robot = np.where(
            blocker_kind == int(BlockerKind.ROBOT), leader, -1
        ).astype(int)

        crossing_gaps = np.mod(
            config.normalized_crossing_s - positions, config.route_length
        )
        nearest_crossing_robot = int(np.argmin(crossing_gaps))
        nearest_crossing_gap = float(crossing_gaps[nearest_crossing_robot])

        if crossing_active and crossing_source != previous_crossing_source:
            required = float(
                d_stop(
                    speeds[nearest_crossing_robot],
                    config.crossing_clearance,
                    config.reaction_time,
                    config.braking,
                )
            )
            if nearest_crossing_gap + config.collision_tolerance < required:
                raise UnsafeCrossingScheduleError(
                    "crossing activated inside the nearest robot's stopping distance: "
                    f"gap={nearest_crossing_gap:.6f}, required={required:.6f}"
                )

        if crossing_active and nearest_crossing_gap <= config.sensor_range:
            crossing_cap = float(
                v_safe_next_step(
                    nearest_crossing_gap,
                    speeds[nearest_crossing_robot],
                    config.crossing_clearance,
                    config.reaction_time,
                    config.braking,
                    config.dt,
                )
            )
            if crossing_cap <= safe_cap[nearest_crossing_robot]:
                safe_cap[nearest_crossing_robot] = crossing_cap
                if crossing_cap < (
                    config.desired_speed - config.intervention_tolerance
                ):
                    blocker_kind[nearest_crossing_robot] = int(BlockerKind.CROSSING)
                    blocker_robot[nearest_crossing_robot] = -1

        minimum_reachable_speed = np.maximum(
            0.0, speeds - config.braking * config.dt
        )
        dynamically_infeasible = safe_cap < (
            minimum_reachable_speed - config.collision_tolerance
        )
        if np.any(dynamically_infeasible):
            bad_robot = int(np.flatnonzero(dynamically_infeasible)[0])
            raise AssertionError(
                "shield next-speed cap is unreachable under the configured "
                f"braking limit for robot {bad_robot}"
            )

        requested_acceleration = (safe_cap - speeds) / config.dt
        applied_acceleration = np.clip(
            requested_acceleration, -config.braking, config.acceleration
        )
        next_speeds = np.maximum(0.0, speeds + applied_acceleration * config.dt)
        movement = speeds * config.dt + 0.5 * applied_acceleration * config.dt**2
        movement = np.maximum(0.0, movement)

        if crossing_active:
            physical_clearance = (
                config.robot_length / 2.0 + config.crossing_half_width
            )
            allowed_movement = nearest_crossing_gap - physical_clearance
            if movement[nearest_crossing_robot] > (
                allowed_movement + config.collision_tolerance
            ):
                crossing_violation_count += 1
                raise AssertionError(
                    "occupied-crossing violation: the nearest robot entered "
                    "the protected crossing zone"
                )

        next_positions = np.mod(positions + movement, config.route_length)
        next_gaps = _forward_gaps(
            next_positions, leader, config.route_length
        )
        next_required_gaps = np.asarray(
            d_stop(
                next_speeds,
                config.robot_clearance,
                config.reaction_time,
                config.braking,
            )
        )
        invariant_violations = next_gaps < (
            next_required_gaps - config.collision_tolerance
        )
        if np.any(invariant_violations):
            bad_robot = int(np.flatnonzero(invariant_violations)[0])
            raise AssertionError(
                "robot stopping-distance invariant violated after step "
                f"{step}: robot {bad_robot}, gap={next_gaps[bad_robot]:.6f}, "
                f"required={next_required_gaps[bad_robot]:.6f}"
            )

        if crossing_active:
            next_crossing_gap = nearest_crossing_gap - float(
                movement[nearest_crossing_robot]
            )
            next_crossing_required = float(
                d_stop(
                    next_speeds[nearest_crossing_robot],
                    config.crossing_clearance,
                    config.reaction_time,
                    config.braking,
                )
            )
            if next_crossing_gap < (
                next_crossing_required - config.collision_tolerance
            ):
                raise AssertionError(
                    "crossing stopping-distance invariant violated after step "
                    f"{step}: gap={next_crossing_gap:.6f}, "
                    f"required={next_crossing_required:.6f}"
                )
        overlaps = next_gaps < (
            config.robot_length - config.collision_tolerance
        )
        if np.any(overlaps):
            collision_count += int(np.count_nonzero(overlaps))
            bad_robot = int(np.flatnonzero(overlaps)[0])
            raise AssertionError(
                f"robot collision after step {step}: robot {bad_robot} has "
                f"forward gap {next_gaps[bad_robot]:.6f}"
            )

        speed_deficit = next_speeds < (
            config.desired_speed - config.intervention_tolerance
        )
        constrained = safe_cap < (
            config.desired_speed - config.intervention_tolerance
        )
        intervening = constrained | speed_deficit
        previously_intervening = active_event_id >= 0
        starts = intervening & ~previously_intervening
        ends = ~intervening & previously_intervening

        start_robots = np.flatnonzero(starts)
        for robot_id in start_robots:
            active_event_id[robot_id] = next_intervention_id
            next_intervention_id += 1

        for robot_id in start_robots:
            event_id = int(active_event_id[robot_id])
            kind = BlockerKind(int(blocker_kind[robot_id]))
            parent_event_id: int | None = None
            blocker_id: int | None = None
            source_id: int | None = None
            if kind == BlockerKind.CROSSING:
                cause: EventCause = "crossing"
                source_id = crossing_source
            elif kind == BlockerKind.ROBOT:
                blocker_id = int(blocker_robot[robot_id])
                candidate_parent = int(active_event_id[blocker_id])
                if candidate_parent >= 0:
                    cause = "robot"
                    parent_event_id = candidate_parent
                else:
                    cause = "background"
            else:
                cause = "background"
            mutable_events[event_id] = _MutableEvent(
                event_id=event_id,
                robot_id=int(robot_id),
                start_time=time,
                cause=cause,
                blocker_robot_id=blocker_id,
                parent_event_id=parent_event_id,
                crossing_source_id=source_id,
                start_position=float(positions[robot_id]),
                crossing_id=0 if kind == BlockerKind.CROSSING else None,
            )

        active_robots = np.flatnonzero(intervening)
        severity = np.clip(1.0 - next_speeds / config.desired_speed, 0.0, 1.0)
        severity_by_robot += severity * config.dt
        for robot_id in active_robots:
            event_id = int(active_event_id[robot_id])
            record = mutable_events[event_id]
            record.severity_loss += float(severity[robot_id] * config.dt)
            record.minimum_speed = min(record.minimum_speed, float(next_speeds[robot_id]))

        for robot_id in np.flatnonzero(ends):
            event_id = int(active_event_id[robot_id])
            mutable_events[event_id].end_time = time + config.dt
            active_event_id[robot_id] = -1

        positions = next_positions
        speeds = next_speeds
        distance_by_robot += movement
        previous_crossing_source = crossing_source

        if (step + 1) % record_stride == 0 or step + 1 == step_count:
            time_frames.append((step + 1) * config.dt)
            position_frames.append(positions.copy())
            speed_frames.append(speeds.copy())
            safe_speed_frames.append(safe_cap.copy())
            intervention_frames.append(intervening.copy())
            blocker_frames.append(blocker_kind.copy())
            blocker_robot_frames.append(blocker_robot.copy())
            crossing_frames.append(crossing_active)

    for event_id in np.unique(active_event_id[active_event_id >= 0]):
        record = mutable_events[int(event_id)]
        record.end_time = duration
        record.censored = True

    lineage = _resolve_lineage(mutable_events)
    finalized_events: list[InterventionEvent] = []
    for event_id in sorted(mutable_events):
        record = mutable_events[event_id]
        root, generation = lineage[event_id]
        end_time = duration if record.end_time is None else record.end_time
        minimum_speed = (
            config.desired_speed
            if not np.isfinite(record.minimum_speed)
            else record.minimum_speed
        )
        finalized_events.append(
            InterventionEvent(
                event_id=event_id,
                robot_id=record.robot_id,
                event_type=0,
                start_time=record.start_time,
                end_time=end_time,
                cause=record.cause,
                blocker_robot_id=record.blocker_robot_id,
                parent_event_id=record.parent_event_id,
                root_primary_event_id=root,
                generation=generation,
                crossing_source_id=record.crossing_source_id,
                severity_loss=record.severity_loss,
                minimum_speed=minimum_speed,
                censored=record.censored,
                start_position=record.start_position,
                crossing_id=record.crossing_id,
            )
        )

    trajectory = TrajectoryLog(
        time=np.asarray(time_frames, dtype=float),
        s=np.asarray(position_frames, dtype=float),
        speed=np.asarray(speed_frames, dtype=float),
        safe_speed=np.asarray(safe_speed_frames, dtype=float),
        intervening=np.asarray(intervention_frames, dtype=bool),
        blocker_kind=np.asarray(blocker_frames, dtype=np.int8),
        blocker_robot_id=np.asarray(blocker_robot_frames, dtype=int),
        crossing_active=np.asarray(crossing_frames, dtype=bool),
    )
    return SimulationResult(
        config=config,
        duration=duration,
        initial_state=initial_copy,
        final_state=FleetState(s=positions.copy(), speed=speeds.copy()),
        events=tuple(finalized_events),
        trajectory=trajectory,
        distance_by_robot=distance_by_robot,
        severity_loss_by_robot=severity_by_robot,
        collision_count=collision_count,
        crossing_violation_count=crossing_violation_count,
    )


def run_paired_rollout(
    config: CorridorConfig,
    duration: float,
    crossing_events: Iterable[CrossingEvent],
    *,
    initial_state: FleetState | None = None,
    robot_count: int | None = None,
    initial_offset: float = 0.0,
    record_stride: int = 1,
) -> PairedRolloutResult:
    """Run event and no-event simulations from exactly the same initial state."""

    if initial_state is None:
        if robot_count is None:
            raise ValueError("robot_count is required when initial_state is omitted")
        initial_state = uniform_initial_state(
            config, robot_count, offset=initial_offset
        )
    treated = simulate_corridor(
        config,
        duration,
        crossing_events=crossing_events,
        initial_state=initial_state,
        record_stride=record_stride,
    )
    control = simulate_corridor(
        config,
        duration,
        crossing_events=(),
        initial_state=initial_state,
        record_stride=record_stride,
    )
    return PairedRolloutResult(
        treated=treated,
        control=control,
        attributable_severity_by_robot=np.maximum(
            0.0,
            treated.severity_loss_by_robot - control.severity_loss_by_robot,
        ),
        distance_loss_by_robot=np.maximum(
            0.0, control.distance_by_robot - treated.distance_by_robot
        ),
    )
