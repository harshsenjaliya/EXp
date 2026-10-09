"""Loaded multi-route AMR simulation with kinematics and conflict reservations.

Milestone 6 supplied route geometry and single-robot kinematic benchmarks but
did not run interacting robots on merge or intersection maps. This module is
the publication-facing bridge. Robots follow curvature-aware route profiles,
obey the same next-step stopping invariant as the corridor simulators, reserve
shared conflict zones, and build causal intervention forests from proximate
blocking contacts.

The model is deliberately route based. It is not a free-space planner and does
not claim industrial safety certification. Its purpose is controlled inference
and capacity experiments where every collision, conflict-zone intrusion, mass
balance, wheel-speed, yaw-rate, and braking invariant is executable.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .capacity_theory import d_stop, v_safe_next_step
from .kinematic_maps import (
    CircularObstacle,
    ConflictZone,
    DifferentialDriveLimits,
    KinematicMap,
    KinematicSpeedProfile,
    SampledPath,
    plan_kinematic_speed_profile,
)
from .simulation import InterventionEvent


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class NetworkJob:
    """One exogenous transport request assigned to a named route."""

    job_id: int
    arrival_time: float
    route_name: str

    def __post_init__(self) -> None:
        if self.job_id < 0:
            raise ValueError("job_id must be nonnegative")
        if not np.isfinite(self.arrival_time) or self.arrival_time < 0.0:
            raise ValueError("arrival_time must be finite and nonnegative")
        if not self.route_name:
            raise ValueError("route_name must be nonempty")


@dataclass(frozen=True)
class NetworkDisturbance:
    """Guarded temporary occupation of one route coordinate."""

    source_id: int
    route_name: str
    s: float
    request_time: float
    duration: float
    half_width: float = 0.35

    def __post_init__(self) -> None:
        if self.source_id < 0 or not self.route_name:
            raise ValueError("disturbance source and route must be valid")
        values = (self.s, self.request_time, self.duration, self.half_width)
        if any(not np.isfinite(item) for item in values):
            raise ValueError("disturbance values must be finite")
        if self.s <= 0.0 or self.request_time < 0.0:
            raise ValueError("disturbance position/time is invalid")
        if self.duration <= 0.0 or self.half_width <= 0.0:
            raise ValueError("disturbance duration/half_width must be positive")


@dataclass(frozen=True)
class NetworkDisturbanceService:
    source_id: int
    route_name: str
    request_time: float
    activation_time: float | None
    release_time: float | None
    status: str


@dataclass(frozen=True)
class NetworkCompletedTraversal:
    job_id: int
    robot_id: int
    route_name: str
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


@dataclass(frozen=True)
class NetworkTrajectoryLog:
    time: FloatArray
    queue_length: IntArray
    work_in_process: IntArray
    cumulative_completions: IntArray
    active_interventions: IntArray
    occupied_conflict_zones: IntArray
    mean_speed: FloatArray
    cumulative_severity_loss: FloatArray


@dataclass(frozen=True)
class NetworkConfig:
    map_spec: KinematicMap
    limits: DifferentialDriveLimits
    fleet_size: int
    dt: float = 0.05
    sensor_range: float = 8.0
    empty_return_time: float = 0.0
    admission_speed: float = 0.0
    intervention_tolerance: float = 1e-4
    collision_tolerance: float = 1e-8
    cross_route_lateral_gate: float | None = None

    def __post_init__(self) -> None:
        if self.fleet_size <= 0:
            raise ValueError("fleet_size must be positive")
        positive = (self.dt, self.sensor_range)
        if any(not np.isfinite(item) or item <= 0.0 for item in positive):
            raise ValueError("dt and sensor_range must be finite and positive")
        nonnegative = (
            self.empty_return_time,
            self.admission_speed,
            self.intervention_tolerance,
            self.collision_tolerance,
        )
        if any(not np.isfinite(item) or item < 0.0 for item in nonnegative):
            raise ValueError("network timing/tolerances must be nonnegative")
        if self.admission_speed > self.limits.max_speed:
            raise ValueError("admission speed exceeds robot maximum")
        required_sensor = float(
            d_stop(
                min(self.limits.max_speed, self.map_spec.speed_limit),
                self.robot_clearance,
                self.limits.reaction_time,
                self.limits.max_deceleration,
            )
        )
        if self.sensor_range + self.collision_tolerance < required_sensor:
            raise ValueError("sensor_range does not cover stopping distance")
        if self.cross_route_lateral_gate is not None and (
            not np.isfinite(self.cross_route_lateral_gate)
            or self.cross_route_lateral_gate <= 0.0
        ):
            raise ValueError("cross_route_lateral_gate must be positive")

    @property
    def robot_clearance(self) -> float:
        return self.limits.length + self.limits.safety_margin

    @property
    def obstacle_clearance(self) -> float:
        return 0.5 * self.limits.length + self.limits.safety_margin

    @property
    def lateral_gate(self) -> float:
        if self.cross_route_lateral_gate is not None:
            return float(self.cross_route_lateral_gate)
        return self.limits.width + self.limits.safety_margin


@dataclass(frozen=True)
class NetworkSimulationResult:
    config: NetworkConfig
    duration: float
    jobs: tuple[NetworkJob, ...]
    disturbances: tuple[NetworkDisturbance, ...]
    disturbance_services: tuple[NetworkDisturbanceService, ...]
    completed_traversals: tuple[NetworkCompletedTraversal, ...]
    events: tuple[InterventionEvent, ...]
    trajectory: NetworkTrajectoryLog
    total_admissions: int
    final_queue_length: int
    final_work_in_process: int
    available_robot_count: int
    returning_robot_count: int
    total_severity_loss: float
    collision_count: int
    conflict_violation_count: int
    disturbance_violation_count: int
    max_abs_yaw_rate: float
    max_abs_yaw_acceleration: float
    max_lateral_acceleration: float
    max_abs_wheel_speed: float

    @property
    def total_completions(self) -> int:
        return len(self.completed_traversals)

    @property
    def throughput(self) -> float:
        return self.total_completions / self.duration

    @property
    def mass_balance_residual(self) -> int:
        return (
            len(self.jobs)
            - self.final_queue_length
            - self.final_work_in_process
            - self.total_completions
        )

    @property
    def fleet_balance_residual(self) -> int:
        return (
            self.config.fleet_size
            - self.final_work_in_process
            - self.available_robot_count
            - self.returning_robot_count
        )


@dataclass(frozen=True)
class NetworkPairedRolloutResult:
    treated: NetworkSimulationResult
    control: NetworkSimulationResult

    @property
    def completion_loss(self) -> int:
        return self.control.total_completions - self.treated.total_completions

    @property
    def severity_increase(self) -> float:
        return self.treated.total_severity_loss - self.control.total_severity_loss


@dataclass
class _MutableEvent:
    event_id: int
    robot_id: int
    event_type: int
    start_time: float
    cause: str
    blocker_robot_id: int | None
    parent_event_id: int | None
    source_id: int | None
    start_position: float
    end_time: float | None = None
    severity_loss: float = 0.0
    minimum_speed: float = np.inf
    censored: bool = False


@dataclass
class _DisturbanceState:
    request: NetworkDisturbance
    status: str = "future"
    activation_time: float | None = None
    release_time: float | None = None
    gate_robot_id: int | None = None


@dataclass(frozen=True)
class _ZoneInterval:
    zone_id: str
    route_index: int
    entry: float
    exit: float


def _path_with_straight_feeders(path: SampledPath, target_length: float) -> SampledPath:
    """Extend route endpoints without scaling away its central curvature."""

    if target_length <= path.length + 1e-9:
        return path
    extension = 0.5 * (target_length - path.length)
    spacing = max(float(np.median(np.diff(path.s))), 0.08)
    count = max(3, int(np.ceil(extension / spacing)) + 1)
    start_heading = float(path.heading[0])
    end_heading = float(path.heading[-1])
    start_distance = np.linspace(extension, 0.0, count)
    end_distance = np.linspace(0.0, extension, count)
    start_x = path.x[0] - start_distance * np.cos(start_heading)
    start_y = path.y[0] - start_distance * np.sin(start_heading)
    end_x = path.x[-1] + end_distance * np.cos(end_heading)
    end_y = path.y[-1] + end_distance * np.sin(end_heading)
    x = np.concatenate((start_x[:-1], path.x, end_x[1:]))
    y = np.concatenate((start_y[:-1], path.y, end_y[1:]))
    ds = np.hypot(np.diff(x), np.diff(y))
    keep = np.concatenate(([True], ds > 1e-10))
    x = x[keep]
    y = y[keep]
    s = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))
    dx = np.gradient(x, s, edge_order=2)
    dy = np.gradient(y, s, edge_order=2)
    heading = np.unwrap(np.arctan2(dy, dx))
    curvature = np.gradient(heading, s, edge_order=2)
    curvature_rate = np.gradient(curvature, s, edge_order=2)
    return SampledPath(
        name=path.name,
        x=x,
        y=y,
        s=s,
        heading=heading,
        curvature=curvature,
        curvature_rate=curvature_rate,
    )


def extend_network_map(map_spec: KinematicMap, target_route_length: float) -> KinematicMap:
    """Add straight approach/exit storage while preserving central topology."""

    if not np.isfinite(target_route_length) or target_route_length <= 0.0:
        raise ValueError("target_route_length must be finite and positive")
    routes = tuple(
        _path_with_straight_feeders(route, target_route_length)
        for route in map_spec.routes
    )
    # The central geometry stays in world coordinates, so its zones and static
    # obstacles remain unchanged.
    return KinematicMap(
        name=f"{map_spec.name}_extended_{target_route_length:g}m",
        routes=routes,
        static_obstacles=tuple(map_spec.static_obstacles),
        conflict_zones=tuple(map_spec.conflict_zones),
        speed_limit=map_spec.speed_limit,
    )


def deterministic_network_jobs(
    rates_by_route: dict[str, float], duration: float
) -> tuple[NetworkJob, ...]:
    """Create deterministic per-route arrivals and merge them chronologically."""

    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be finite and positive")
    raw: list[tuple[float, str]] = []
    for route_name, rate in rates_by_route.items():
        if not route_name or not np.isfinite(rate) or rate < 0.0:
            raise ValueError("route rates must be finite and nonnegative")
        if rate == 0.0:
            continue
        times = np.arange(0.0, duration, 1.0 / rate, dtype=float)
        raw.extend((float(time), route_name) for time in times)
    raw.sort(key=lambda item: (item[0], item[1]))
    return tuple(NetworkJob(index, time, route) for index, (time, route) in enumerate(raw))


def poisson_network_disturbances(
    map_spec: KinematicMap,
    *,
    rate_per_route: float,
    duration: float,
    duration_range: tuple[float, float],
    seed: int,
    start_time: float = 0.0,
) -> tuple[NetworkDisturbance, ...]:
    """Generate disturbances near each route's first conflict-zone centre."""

    if rate_per_route < 0.0 or duration <= 0.0 or start_time < 0.0:
        raise ValueError("invalid disturbance process parameters")
    low, high = map(float, duration_range)
    if low <= 0.0 or high < low:
        raise ValueError("invalid duration_range")
    rng = np.random.default_rng(seed)
    zone_by_route: dict[str, ConflictZone] = {}
    for zone in map_spec.conflict_zones:
        for route_name in zone.route_names:
            zone_by_route.setdefault(route_name, zone)
    requests: list[NetworkDisturbance] = []
    source = 0
    for route in map_spec.routes:
        if route.name not in zone_by_route or rate_per_route == 0.0:
            continue
        zone = zone_by_route[route.name]
        distance = np.hypot(route.x - zone.x, route.y - zone.y)
        location = float(route.s[int(np.argmin(distance))])
        time = float(start_time + rng.exponential(1.0 / rate_per_route))
        while time < duration:
            requests.append(
                NetworkDisturbance(
                    source_id=source,
                    route_name=route.name,
                    s=location,
                    request_time=time,
                    duration=float(rng.uniform(low, high)),
                )
            )
            source += 1
            time += float(rng.exponential(1.0 / rate_per_route))
    return tuple(sorted(requests, key=lambda item: (item.request_time, item.source_id)))


def _zone_intervals(config: NetworkConfig) -> tuple[_ZoneInterval, ...]:
    route_lookup = {route.name: index for index, route in enumerate(config.map_spec.routes)}
    result: list[_ZoneInterval] = []
    for zone in config.map_spec.conflict_zones:
        for route_name in zone.route_names:
            route_index = route_lookup[route_name]
            route = config.map_spec.routes[route_index]
            distance = np.hypot(route.x - zone.x, route.y - zone.y)
            inside = np.flatnonzero(distance <= zone.radius + 1e-10)
            if inside.size == 0:
                raise ValueError(f"route {route_name} never intersects zone {zone.zone_id}")
            result.append(
                _ZoneInterval(
                    zone_id=zone.zone_id,
                    route_index=route_index,
                    entry=float(route.s[inside[0]]),
                    exit=float(route.s[inside[-1]]),
                )
            )
    return tuple(result)


def _resolve_lineage(
    events: dict[int, _MutableEvent],
) -> dict[int, tuple[int | None, int | None]]:
    resolved: dict[int, tuple[int | None, int | None]] = {}

    def visit(event_id: int, stack: set[int]) -> tuple[int | None, int | None]:
        if event_id in resolved:
            return resolved[event_id]
        if event_id in stack:
            resolved[event_id] = (None, None)
            return resolved[event_id]
        event = events[event_id]
        if event.cause == "crossing":
            resolved[event_id] = (event_id, 0)
        elif event.parent_event_id is None:
            resolved[event_id] = (None, None)
        else:
            root, generation = visit(event.parent_event_id, stack | {event_id})
            resolved[event_id] = (
                (None, None)
                if root is None or generation is None
                else (root, generation + 1)
            )
        return resolved[event_id]

    for event_id in events:
        visit(event_id, set())
    return resolved


def _validate_inputs(
    config: NetworkConfig,
    duration: float,
    jobs: Iterable[NetworkJob],
    disturbances: Iterable[NetworkDisturbance],
) -> tuple[tuple[NetworkJob, ...], tuple[NetworkDisturbance, ...], int]:
    steps_float = duration / config.dt
    steps = int(round(steps_float))
    if not np.isfinite(duration) or duration <= 0.0 or not np.isclose(
        steps_float, steps, atol=1e-10, rtol=0.0
    ):
        raise ValueError("duration must be a positive integer multiple of dt")
    route_names = {route.name for route in config.map_spec.routes}
    job_records = tuple(sorted(jobs, key=lambda item: (item.arrival_time, item.job_id)))
    if len({item.job_id for item in job_records}) != len(job_records):
        raise ValueError("job IDs must be unique")
    if any(item.route_name not in route_names for item in job_records):
        raise ValueError("job references an unknown route")
    if any(item.arrival_time >= duration + 1e-12 for item in job_records):
        raise ValueError("job arrival lies outside simulation horizon")
    disturbance_records = tuple(
        sorted(disturbances, key=lambda item: (item.request_time, item.source_id))
    )
    if len({item.source_id for item in disturbance_records}) != len(disturbance_records):
        raise ValueError("disturbance source IDs must be unique")
    route_by_name = {route.name: route for route in config.map_spec.routes}
    for item in disturbance_records:
        if item.route_name not in route_by_name:
            raise ValueError("disturbance references an unknown route")
        if item.s >= route_by_name[item.route_name].length:
            raise ValueError("disturbance lies outside its route")
        if item.request_time >= duration:
            raise ValueError("disturbance request lies outside horizon")
    return job_records, disturbance_records, steps


def simulate_route_network(
    config: NetworkConfig,
    duration: float,
    jobs: Iterable[NetworkJob],
    *,
    disturbances: Iterable[NetworkDisturbance] = (),
    record_stride: int = 10,
) -> NetworkSimulationResult:
    """Run an empty-start, finite-fleet multi-route simulation."""

    job_records, disturbance_records, step_count = _validate_inputs(
        config, duration, jobs, disturbances
    )
    if record_stride <= 0:
        raise ValueError("record_stride must be positive")
    routes = config.map_spec.routes
    route_lookup = {route.name: index for index, route in enumerate(routes)}
    cruise_speed = min(
        config.limits.max_speed,
        config.limits.max_wheel_speed,
        config.map_spec.speed_limit,
    )
    profiles: tuple[KinematicSpeedProfile, ...] = tuple(
        plan_kinematic_speed_profile(
            route,
            config.limits,
            map_speed_limit=config.map_spec.speed_limit,
            start_speed=cruise_speed,
            end_speed=cruise_speed,
        )
        for route in routes
    )
    intervals = _zone_intervals(config)
    interval_by_zone: dict[str, list[_ZoneInterval]] = {}
    for interval in intervals:
        interval_by_zone.setdefault(interval.zone_id, []).append(interval)
    zone_owner: dict[str, int | None] = {
        zone.zone_id: None for zone in config.map_spec.conflict_zones
    }

    disturbance_states = [_DisturbanceState(item) for item in disturbance_records]
    robot_ids = np.empty(0, dtype=int)
    job_ids = np.empty(0, dtype=int)
    route_indices = np.empty(0, dtype=int)
    progress = np.empty(0, dtype=float)
    speeds = np.empty(0, dtype=float)
    arrival_times = np.empty(0, dtype=float)
    admission_times = np.empty(0, dtype=float)
    severity_by_robot = np.empty(0, dtype=float)
    active_event_ids = np.empty(0, dtype=int)
    available: deque[int] = deque(range(config.fleet_size))
    returning: list[tuple[float, int]] = []
    queues: list[deque[NetworkJob]] = [deque() for _ in routes]
    job_index = 0
    next_event_id = 0
    admissions = 0
    completed: list[NetworkCompletedTraversal] = []
    mutable_events: dict[int, _MutableEvent] = {}
    total_severity = 0.0
    collisions = 0
    conflict_violations = 0
    disturbance_violations = 0
    max_yaw_rate = 0.0
    max_yaw_acceleration = 0.0
    max_lateral_acceleration = 0.0
    max_wheel_speed = 0.0
    previous_yaw_by_robot: dict[int, float] = {}

    time_log = [0.0]
    queue_log = [0]
    wip_log = [0]
    completion_log = [0]
    intervention_log = [0]
    zone_log = [0]
    mean_speed_log = [np.nan]
    severity_log = [0.0]

    for step in range(step_count):
        time = step * config.dt
        while returning and returning[0][0] <= time + 1e-12:
            _, returned_id = heapq.heappop(returning)
            available.append(returned_id)
        while job_index < len(job_records) and job_records[job_index].arrival_time <= time + 1e-12:
            job = job_records[job_index]
            queues[route_lookup[job.route_name]].append(job)
            job_index += 1

        # Release completed disturbance occupations and move arrived requests
        # into the guarded-pending state.
        for state in disturbance_states:
            if state.status == "future" and state.request.request_time <= time + 1e-12:
                state.status = "pending"
            if (
                state.status == "active"
                and state.activation_time is not None
                and time >= state.activation_time + state.request.duration - 1e-12
            ):
                state.status = "completed"
                state.release_time = time
                state.gate_robot_id = None

        # Release conflict-zone ownership after the owner exits or leaves.
        id_to_index = {int(value): index for index, value in enumerate(robot_ids)}
        for zone_id, owner in tuple(zone_owner.items()):
            if owner is None or owner not in id_to_index:
                zone_owner[zone_id] = None
                continue
            owner_index = id_to_index[owner]
            matching = [
                item
                for item in interval_by_zone[zone_id]
                if item.route_index == int(route_indices[owner_index])
            ]
            if not matching or progress[owner_index] > matching[0].exit + 1e-9:
                zone_owner[zone_id] = None

        # Grant each free resource to the closest upstream robot. Deterministic
        # robot-ID tie breaking keeps paired runs reproducible.
        for zone_id, owner in tuple(zone_owner.items()):
            if owner is not None:
                continue
            candidates: list[tuple[float, int]] = []
            for interval in interval_by_zone[zone_id]:
                indices = np.flatnonzero(route_indices == interval.route_index)
                upstream = indices[progress[indices] <= interval.entry + 1e-9]
                if upstream.size:
                    selected = int(upstream[np.argmax(progress[upstream])])
                    candidates.append(
                        (interval.entry - float(progress[selected]), int(robot_ids[selected]))
                    )
            if candidates:
                candidates.sort(key=lambda item: (item[0], item[1]))
                zone_owner[zone_id] = candidates[0][1]

        # Admit at most one job per route and step. An empty-start experiment
        # avoids hidden initial conflicts and reaches stationarity via warmup.
        for route_index, queue in enumerate(queues):
            if not queue or not available:
                continue
            route = routes[route_index]
            same = np.flatnonzero(route_indices == route_index)
            rear_gap = float(np.min(progress[same])) if same.size else np.inf
            required = float(
                d_stop(
                    config.admission_speed,
                    config.robot_clearance,
                    config.limits.reaction_time,
                    config.limits.max_deceleration,
                )
            )
            if rear_gap + config.collision_tolerance < required:
                continue
            start_x, start_y, _, _ = route.pose(0.0)
            if robot_ids.size:
                xy = np.asarray(
                    [routes[int(r)].pose(float(s))[:2] for r, s in zip(route_indices, progress, strict=True)]
                )
                if np.any(
                    np.hypot(xy[:, 0] - start_x, xy[:, 1] - start_y)
                    < max(config.limits.length, config.limits.width)
                    - config.collision_tolerance
                ):
                    continue
            job = queue.popleft()
            robot_id = available.popleft()
            robot_ids = np.append(robot_ids, robot_id)
            job_ids = np.append(job_ids, job.job_id)
            route_indices = np.append(route_indices, route_index)
            progress = np.append(progress, 0.0)
            speeds = np.append(speeds, config.admission_speed)
            arrival_times = np.append(arrival_times, job.arrival_time)
            admission_times = np.append(admission_times, time)
            severity_by_robot = np.append(severity_by_robot, 0.0)
            active_event_ids = np.append(active_event_ids, -1)
            admissions += 1

        count = robot_ids.size
        if count:
            x = np.empty(count, dtype=float)
            y = np.empty(count, dtype=float)
            heading = np.empty(count, dtype=float)
            curvature = np.empty(count, dtype=float)
            nominal = np.empty(count, dtype=float)
            for route_index, (route, profile) in enumerate(zip(routes, profiles, strict=True)):
                indices = np.flatnonzero(route_indices == route_index)
                if not indices.size:
                    continue
                pose = route.pose(progress[indices])
                x[indices], y[indices], heading[indices], curvature[indices] = pose
                nominal[indices] = np.interp(
                    progress[indices], route.s, profile.speed
                )
            safe_cap = nominal.copy()
            blocker_code = np.zeros(count, dtype=np.int8)  # 1 robot, 2 disturbance, 3 zone
            blocker_robot = np.full(count, -1, dtype=int)
            blocker_source = np.full(count, -1, dtype=int)

            # Guarded disturbance activation. Too-close robots clear; the
            # closest robot that can stop becomes the reservation gate.
            for state in disturbance_states:
                if state.status not in ("pending", "active"):
                    continue
                request = state.request
                route_index = route_lookup[request.route_name]
                indices = np.flatnonzero(route_indices == route_index)
                physical = 0.5 * config.limits.length + request.half_width
                if state.status == "pending":
                    occupied = bool(
                        np.any(np.abs(progress[indices] - request.s) < physical)
                    )
                    upstream = indices[progress[indices] < request.s]
                    nearest = (
                        None
                        if not upstream.size
                        else int(upstream[np.argmax(progress[upstream])])
                    )
                    safe_nearest = True
                    if nearest is not None:
                        gap = request.s - float(progress[nearest])
                        required = float(
                            d_stop(
                                speeds[nearest],
                                config.obstacle_clearance + request.half_width,
                                config.limits.reaction_time,
                                config.limits.max_deceleration,
                            )
                        )
                        safe_nearest = gap + config.collision_tolerance >= required
                    if not occupied and safe_nearest:
                        state.status = "active"
                        state.activation_time = time
                        state.gate_robot_id = None
                    elif state.gate_robot_id not in id_to_index:
                        eligible: list[int] = []
                        for candidate in upstream:
                            gap = request.s - float(progress[candidate])
                            required = float(
                                d_stop(
                                    speeds[candidate],
                                    config.obstacle_clearance + request.half_width,
                                    config.limits.reaction_time,
                                    config.limits.max_deceleration,
                                )
                            )
                            if gap + config.collision_tolerance >= required:
                                eligible.append(int(candidate))
                        if eligible:
                            chosen = max(eligible, key=lambda item: progress[item])
                            state.gate_robot_id = int(robot_ids[chosen])

                gate_index: int | None = None
                if state.status == "active":
                    upstream = indices[progress[indices] < request.s]
                    if upstream.size:
                        gate_index = int(upstream[np.argmax(progress[upstream])])
                elif state.gate_robot_id in id_to_index:
                    gate_index = id_to_index[int(state.gate_robot_id)]
                if gate_index is not None:
                    gap = request.s - float(progress[gate_index])
                    if 0.0 <= gap <= config.sensor_range:
                        cap = float(
                            v_safe_next_step(
                                gap,
                                speeds[gate_index],
                                config.obstacle_clearance + request.half_width,
                                config.limits.reaction_time,
                                config.limits.max_deceleration,
                                config.dt,
                            )
                        )
                        if cap < safe_cap[gate_index]:
                            safe_cap[gate_index] = cap
                            blocker_code[gate_index] = 2
                            blocker_source[gate_index] = request.source_id

            # Conflict-zone gates. Only the closest upstream robot on each
            # non-owner stream is stopped; its followers use ordinary contact.
            id_to_index = {int(value): index for index, value in enumerate(robot_ids)}
            for zone_id, zone_intervals in interval_by_zone.items():
                owner_id = zone_owner[zone_id]
                for interval in zone_intervals:
                    indices = np.flatnonzero(route_indices == interval.route_index)
                    upstream = indices[progress[indices] < interval.entry]
                    if not upstream.size:
                        continue
                    candidate = int(upstream[np.argmax(progress[upstream])])
                    if owner_id is not None and int(robot_ids[candidate]) == owner_id:
                        continue
                    gap = interval.entry - float(progress[candidate])
                    if gap <= config.sensor_range:
                        cap = float(
                            v_safe_next_step(
                                gap,
                                speeds[candidate],
                                config.obstacle_clearance,
                                config.limits.reaction_time,
                                config.limits.max_deceleration,
                                config.dt,
                            )
                        )
                        if cap < safe_cap[candidate]:
                            safe_cap[candidate] = cap
                            blocker_code[candidate] = 3
                            blocker_robot[candidate] = -1 if owner_id is None else owner_id

            # Same-route leaders. Apply after gates so immediate contact wins
            # ties and descendants are attributed one generation at a time.
            for route_index in range(len(routes)):
                indices = np.flatnonzero(route_indices == route_index)
                if indices.size < 2:
                    continue
                ordered = indices[np.argsort(progress[indices])]
                followers = ordered[:-1]
                leaders = ordered[1:]
                gaps = progress[leaders] - progress[followers]
                sensed = gaps <= config.sensor_range
                if not np.any(sensed):
                    continue
                caps = np.asarray(
                    v_safe_next_step(
                        gaps[sensed],
                        speeds[followers[sensed]],
                        config.robot_clearance,
                        config.limits.reaction_time,
                        config.limits.max_deceleration,
                        config.dt,
                    ),
                    dtype=float,
                )
                for follower, leader, cap in zip(
                    followers[sensed], leaders[sensed], caps, strict=True
                ):
                    if cap <= safe_cap[follower] + 1e-12:
                        safe_cap[follower] = min(safe_cap[follower], cap)
                        blocker_code[follower] = 1
                        blocker_robot[follower] = int(robot_ids[leader])

            # Aligned cross-route leaders capture the shared downstream lane of
            # a merge without imposing a one-robot capacity on the whole link.
            dx = x[None, :] - x[:, None]
            dy = y[None, :] - y[:, None]
            forward = dx * np.cos(heading)[:, None] + dy * np.sin(heading)[:, None]
            lateral = np.abs(-dx * np.sin(heading)[:, None] + dy * np.cos(heading)[:, None])
            distance = np.hypot(dx, dy)
            other_route = route_indices[:, None] != route_indices[None, :]
            candidates = (
                other_route
                & (forward > 0.0)
                & (forward <= config.sensor_range)
                & (lateral <= config.lateral_gate)
            )
            for follower in np.flatnonzero(np.any(candidates, axis=1)):
                choices = np.flatnonzero(candidates[follower])
                leader = int(choices[np.argmin(forward[follower, choices])])
                gap = float(distance[follower, leader])
                cap = float(
                    v_safe_next_step(
                        gap,
                        speeds[follower],
                        config.robot_clearance,
                        config.limits.reaction_time,
                        config.limits.max_deceleration,
                        config.dt,
                    )
                )
                if cap <= safe_cap[follower] + 1e-12:
                    safe_cap[follower] = min(safe_cap[follower], cap)
                    blocker_code[follower] = 1
                    blocker_robot[follower] = int(robot_ids[leader])

            minimum_reachable = np.maximum(
                0.0, speeds - config.limits.max_deceleration * config.dt
            )
            if np.any(
                safe_cap < minimum_reachable - config.collision_tolerance
            ):
                bad = int(
                    np.flatnonzero(
                        safe_cap < minimum_reachable - config.collision_tolerance
                    )[0]
                )
                raise AssertionError(
                    f"network shield cap is unreachable for robot {int(robot_ids[bad])}"
                )
            requested_acceleration = (safe_cap - speeds) / config.dt
            acceleration = np.clip(
                requested_acceleration,
                -config.limits.max_deceleration,
                config.limits.max_acceleration,
            )
            next_speeds = np.maximum(0.0, speeds + acceleration * config.dt)
            movement = np.maximum(
                0.0, speeds * config.dt + 0.5 * acceleration * config.dt**2
            )
            next_progress = progress + movement

            constrained = safe_cap < nominal - config.intervention_tolerance
            recovering = (active_event_ids >= 0) & (
                next_speeds < nominal - config.intervention_tolerance
            )
            intervening = constrained | recovering
            starts = constrained & (active_event_ids < 0)
            ends = ~intervening & (active_event_ids >= 0)
            for index in np.flatnonzero(starts):
                active_event_ids[index] = next_event_id
                next_event_id += 1
            id_to_index = {int(value): index for index, value in enumerate(robot_ids)}
            for index in np.flatnonzero(starts):
                code = int(blocker_code[index])
                parent: int | None = None
                blocker: int | None = None
                source: int | None = None
                if code == 2:
                    cause = "crossing"
                    source = int(blocker_source[index])
                elif code in (1, 3) and blocker_robot[index] in id_to_index:
                    blocker = int(blocker_robot[index])
                    blocker_index = id_to_index[blocker]
                    candidate_parent = int(active_event_ids[blocker_index])
                    if candidate_parent >= 0:
                        cause = "robot"
                        parent = candidate_parent
                    else:
                        cause = "background"
                else:
                    cause = "background"
                event_id = int(active_event_ids[index])
                mutable_events[event_id] = _MutableEvent(
                    event_id=event_id,
                    robot_id=int(robot_ids[index]),
                    event_type=int(route_indices[index]),
                    start_time=time,
                    cause=cause,
                    blocker_robot_id=blocker,
                    parent_event_id=parent,
                    source_id=source,
                    start_position=float(progress[index]),
                )

            severity = np.divide(
                nominal - next_speeds,
                nominal,
                out=np.zeros_like(nominal),
                where=nominal > 1e-9,
            )
            severity = np.clip(severity, 0.0, 1.0)
            severity_by_robot += severity * config.dt
            total_severity += float(np.sum(severity) * config.dt)
            for index in np.flatnonzero(active_event_ids >= 0):
                record = mutable_events[int(active_event_ids[index])]
                record.severity_loss += float(severity[index] * config.dt)
                record.minimum_speed = min(record.minimum_speed, float(next_speeds[index]))
            for index in np.flatnonzero(ends):
                mutable_events[int(active_event_ids[index])].end_time = time + config.dt
                active_event_ids[index] = -1

            # Post-step invariants: route headways, global physical overlap,
            # single-owner conflict zones, and occupied disturbance intervals.
            next_x = np.empty(count, dtype=float)
            next_y = np.empty(count, dtype=float)
            next_heading = np.empty(count, dtype=float)
            next_curvature = np.empty(count, dtype=float)
            for route_index, route in enumerate(routes):
                indices = np.flatnonzero(route_indices == route_index)
                if indices.size:
                    pose = route.pose(np.minimum(next_progress[indices], route.length))
                    next_x[indices], next_y[indices], next_heading[indices], next_curvature[indices] = pose
                if indices.size >= 2:
                    ordered = indices[np.argsort(next_progress[indices])]
                    gaps = np.diff(next_progress[ordered])
                    required = np.asarray(
                        d_stop(
                            next_speeds[ordered[:-1]],
                            config.robot_clearance,
                            config.limits.reaction_time,
                            config.limits.max_deceleration,
                        ),
                        dtype=float,
                    )
                    if np.any(gaps < required - config.collision_tolerance):
                        raise AssertionError("same-route stopping invariant violated")
            pair_distance = np.hypot(
                next_x[:, None] - next_x[None, :],
                next_y[:, None] - next_y[None, :],
            )
            np.fill_diagonal(pair_distance, np.inf)
            physical_diameter = min(config.limits.length, config.robot_clearance)
            collision_pairs = np.triu(
                pair_distance < physical_diameter - config.collision_tolerance, 1
            )
            if np.any(collision_pairs):
                collisions += int(np.count_nonzero(collision_pairs))
                raise AssertionError("global robot collision in route network")
            for zone_id, zone_intervals in interval_by_zone.items():
                occupants: list[int] = []
                for interval in zone_intervals:
                    indices = np.flatnonzero(route_indices == interval.route_index)
                    inside = indices[
                        (next_progress[indices] > interval.entry)
                        & (next_progress[indices] < interval.exit)
                    ]
                    occupants.extend(int(robot_ids[item]) for item in inside)
                if len(occupants) > 1 or (
                    occupants
                    and zone_owner[zone_id] is not None
                    and occupants[0] != zone_owner[zone_id]
                ):
                    conflict_violations += 1
                    raise AssertionError("conflict-zone reservation violated")
            for state in disturbance_states:
                if state.status != "active":
                    continue
                request = state.request
                route_index = route_lookup[request.route_name]
                indices = np.flatnonzero(route_indices == route_index)
                physical = 0.5 * config.limits.length + request.half_width
                if np.any(np.abs(next_progress[indices] - request.s) < physical - config.collision_tolerance):
                    disturbance_violations += 1
                    raise AssertionError("active disturbance interval was entered")

            yaw_rate = next_speeds * next_curvature
            lateral_acceleration = np.abs(next_speeds**2 * next_curvature)
            left_wheel = next_speeds * (1.0 - 0.5 * config.limits.wheel_track * next_curvature)
            right_wheel = next_speeds * (1.0 + 0.5 * config.limits.wheel_track * next_curvature)
            max_yaw_rate = max(max_yaw_rate, float(np.max(np.abs(yaw_rate))))
            max_lateral_acceleration = max(
                max_lateral_acceleration, float(np.max(lateral_acceleration))
            )
            max_wheel_speed = max(
                max_wheel_speed,
                float(np.max(np.abs(np.concatenate((left_wheel, right_wheel))))),
            )
            for index, robot_id in enumerate(robot_ids):
                old_yaw = previous_yaw_by_robot.get(int(robot_id), float(yaw_rate[index]))
                max_yaw_acceleration = max(
                    max_yaw_acceleration,
                    abs(float(yaw_rate[index]) - old_yaw) / config.dt,
                )
                previous_yaw_by_robot[int(robot_id)] = float(yaw_rate[index])

            # Complete routes, close active events, and return physical robots.
            completed_mask = np.asarray(
                [next_progress[i] >= routes[int(route_indices[i])].length for i in range(count)],
                dtype=bool,
            )
            for index in np.flatnonzero(completed_mask):
                route = routes[int(route_indices[index])]
                move = float(movement[index])
                fraction = (
                    1.0
                    if move <= 0.0
                    else float(
                        np.clip(
                            (route.length - float(progress[index])) / move, 0.0, 1.0
                        )
                    )
                )
                completion_time = time + fraction * config.dt
                completed.append(
                    NetworkCompletedTraversal(
                        job_id=int(job_ids[index]),
                        robot_id=int(robot_ids[index]),
                        route_name=route.name,
                        arrival_time=float(arrival_times[index]),
                        admission_time=float(admission_times[index]),
                        completion_time=completion_time,
                        severity_loss=float(severity_by_robot[index]),
                    )
                )
                if active_event_ids[index] >= 0:
                    record = mutable_events[int(active_event_ids[index])]
                    record.end_time = completion_time
                    record.censored = False
                heapq.heappush(
                    returning,
                    (completion_time + config.empty_return_time, int(robot_ids[index])),
                )
                previous_yaw_by_robot.pop(int(robot_ids[index]), None)
            keep = ~completed_mask
            robot_ids = robot_ids[keep]
            job_ids = job_ids[keep]
            route_indices = route_indices[keep]
            progress = next_progress[keep]
            speeds = next_speeds[keep]
            arrival_times = arrival_times[keep]
            admission_times = admission_times[keep]
            severity_by_robot = severity_by_robot[keep]
            active_event_ids = active_event_ids[keep]

        interval_end = (step + 1) * config.dt
        while job_index < len(job_records) and job_records[job_index].arrival_time <= interval_end + 1e-12:
            job = job_records[job_index]
            queues[route_lookup[job.route_name]].append(job)
            job_index += 1
        if (step + 1) % record_stride == 0 or step + 1 == step_count:
            time_log.append(interval_end)
            queue_log.append(sum(len(item) for item in queues))
            wip_log.append(int(robot_ids.size))
            completion_log.append(len(completed))
            intervention_log.append(int(np.count_nonzero(active_event_ids >= 0)))
            zone_log.append(sum(owner is not None for owner in zone_owner.values()))
            mean_speed_log.append(float(np.mean(speeds)) if speeds.size else np.nan)
            severity_log.append(total_severity)

        expected_jobs = job_index - len(completed)
        observed_jobs = sum(len(item) for item in queues) + int(robot_ids.size)
        if expected_jobs != observed_jobs:
            raise AssertionError("network job mass conservation failed")
        observed_fleet = int(robot_ids.size) + len(available) + len(returning)
        if observed_fleet != config.fleet_size:
            raise AssertionError("network physical fleet conservation failed")

    for event_id in np.unique(active_event_ids[active_event_ids >= 0]):
        event = mutable_events[int(event_id)]
        event.end_time = duration
        event.censored = True
    lineage = _resolve_lineage(mutable_events)
    finalized: list[InterventionEvent] = []
    for event_id in sorted(mutable_events):
        event = mutable_events[event_id]
        root, generation = lineage[event_id]
        finalized.append(
            InterventionEvent(
                event_id=event.event_id,
                robot_id=event.robot_id,
                event_type=event.event_type,
                start_time=event.start_time,
                end_time=duration if event.end_time is None else event.end_time,
                cause=event.cause,  # type: ignore[arg-type]
                blocker_robot_id=event.blocker_robot_id,
                parent_event_id=event.parent_event_id,
                root_primary_event_id=root,
                generation=generation,
                crossing_source_id=event.source_id,
                severity_loss=event.severity_loss,
                minimum_speed=(
                    0.0 if not np.isfinite(event.minimum_speed) else event.minimum_speed
                ),
                censored=event.censored,
                start_position=event.start_position,
                crossing_id=None,
            )
        )
    services = tuple(
        NetworkDisturbanceService(
            source_id=state.request.source_id,
            route_name=state.request.route_name,
            request_time=state.request.request_time,
            activation_time=state.activation_time,
            release_time=(duration if state.status == "active" else state.release_time),
            status=(
                "active_at_end"
                if state.status == "active"
                else "unserved"
                if state.status in ("future", "pending")
                else "completed"
            ),
        )
        for state in disturbance_states
    )
    while returning and returning[0][0] <= duration + 1e-12:
        _, returned_id = heapq.heappop(returning)
        available.append(returned_id)
    result = NetworkSimulationResult(
        config=config,
        duration=duration,
        jobs=job_records,
        disturbances=disturbance_records,
        disturbance_services=services,
        completed_traversals=tuple(completed),
        events=tuple(finalized),
        trajectory=NetworkTrajectoryLog(
            time=np.asarray(time_log, dtype=float),
            queue_length=np.asarray(queue_log, dtype=int),
            work_in_process=np.asarray(wip_log, dtype=int),
            cumulative_completions=np.asarray(completion_log, dtype=int),
            active_interventions=np.asarray(intervention_log, dtype=int),
            occupied_conflict_zones=np.asarray(zone_log, dtype=int),
            mean_speed=np.asarray(mean_speed_log, dtype=float),
            cumulative_severity_loss=np.asarray(severity_log, dtype=float),
        ),
        total_admissions=admissions,
        final_queue_length=sum(len(item) for item in queues),
        final_work_in_process=int(robot_ids.size),
        available_robot_count=len(available),
        returning_robot_count=len(returning),
        total_severity_loss=total_severity,
        collision_count=collisions,
        conflict_violation_count=conflict_violations,
        disturbance_violation_count=disturbance_violations,
        max_abs_yaw_rate=max_yaw_rate,
        max_abs_yaw_acceleration=max_yaw_acceleration,
        max_lateral_acceleration=max_lateral_acceleration,
        max_abs_wheel_speed=max_wheel_speed,
    )
    if result.mass_balance_residual != 0 or result.fleet_balance_residual != 0:
        raise AssertionError("final network conservation check failed")
    if result.max_abs_yaw_rate > config.limits.max_yaw_rate * 1.02:
        raise AssertionError("yaw-rate limit exceeded")
    if result.max_lateral_acceleration > config.limits.max_lateral_acceleration * 1.02:
        raise AssertionError("lateral-acceleration limit exceeded")
    if result.max_abs_wheel_speed > config.limits.max_wheel_speed * 1.02:
        raise AssertionError("wheel-speed limit exceeded")
    return result


def run_network_paired_rollout(
    config: NetworkConfig,
    duration: float,
    jobs: Iterable[NetworkJob],
    disturbances: Iterable[NetworkDisturbance],
    *,
    record_stride: int = 10,
) -> NetworkPairedRolloutResult:
    """Run matched treated/control network simulations."""

    jobs_tuple = tuple(jobs)
    disturbances_tuple = tuple(disturbances)
    treated = simulate_route_network(
        config,
        duration,
        jobs_tuple,
        disturbances=disturbances_tuple,
        record_stride=record_stride,
    )
    control = simulate_route_network(
        config,
        duration,
        jobs_tuple,
        disturbances=(),
        record_stride=record_stride,
    )
    return NetworkPairedRolloutResult(treated=treated, control=control)
