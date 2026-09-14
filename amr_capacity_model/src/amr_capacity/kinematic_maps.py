"""Two-dimensional route geometry and differential-drive kinematic benchmarks.

Milestone 6 separates nominal map time from shield-induced loss.  Curvature,
wheel-speed, lateral-acceleration, yaw-rate, yaw-acceleration, longitudinal
acceleration, and braking constraints determine the nominal profile T0(M).
Guarded route obstructions then add measured intervention burden g.

The route simulator is deliberately a path-coordinate benchmark.  It validates
kinematics, obstacle activation, footprint clearance, and the longitudinal
shield on diverse 2D paths.  It is not presented as a multi-robot intersection
or merge-capacity simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .capacity_theory import d_stop, v_safe_next_step


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DifferentialDriveLimits:
    """Physical and protective limits for a differential-drive AMR."""

    max_speed: float = 1.8
    max_acceleration: float = 0.9
    max_deceleration: float = 1.4
    max_yaw_rate: float = 1.2
    max_yaw_acceleration: float = 1.8
    max_lateral_acceleration: float = 1.0
    max_wheel_speed: float = 2.2
    wheel_track: float = 0.52
    length: float = 0.82
    width: float = 0.62
    reaction_time: float = 0.20
    safety_margin: float = 0.18

    def __post_init__(self) -> None:
        values = (
            self.max_speed,
            self.max_acceleration,
            self.max_deceleration,
            self.max_yaw_rate,
            self.max_yaw_acceleration,
            self.max_lateral_acceleration,
            self.max_wheel_speed,
            self.wheel_track,
            self.length,
            self.width,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("all kinematic dimensions and limits must be positive")
        if (
            not np.isfinite(self.reaction_time)
            or self.reaction_time < 0.0
            or not np.isfinite(self.safety_margin)
            or self.safety_margin < 0.0
        ):
            raise ValueError("reaction_time and safety_margin must be nonnegative")

    @property
    def footprint_radius(self) -> float:
        """Radius of the rectangular footprint's circumscribed circle."""

        return float(0.5 * np.hypot(self.length, self.width))

    @property
    def longitudinal_clearance(self) -> float:
        """Center-to-obstacle static clearance used by the route shield."""

        return 0.5 * self.length + self.safety_margin


@dataclass(frozen=True)
class SampledPath:
    """A smooth centerline sampled monotonically in arc length."""

    name: str
    x: FloatArray
    y: FloatArray
    s: FloatArray
    heading: FloatArray
    curvature: FloatArray
    curvature_rate: FloatArray

    def __post_init__(self) -> None:
        arrays = tuple(
            np.asarray(value, dtype=float)
            for value in (
                self.x,
                self.y,
                self.s,
                self.heading,
                self.curvature,
                self.curvature_rate,
            )
        )
        if not self.name:
            raise ValueError("path name must be nonempty")
        if any(array.ndim != 1 for array in arrays):
            raise ValueError("path arrays must be one-dimensional")
        if len({array.size for array in arrays}) != 1 or arrays[0].size < 5:
            raise ValueError("path arrays must share a length of at least five")
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("path arrays must contain only finite values")
        if abs(arrays[2][0]) > 1e-10 or np.any(np.diff(arrays[2]) <= 0.0):
            raise ValueError("arc length must start at zero and increase strictly")
        for field_name, array in zip(
            ("x", "y", "s", "heading", "curvature", "curvature_rate"),
            arrays,
            strict=True,
        ):
            object.__setattr__(self, field_name, array)

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def pose(self, progress: ArrayLike):
        """Interpolate x, y, heading, and curvature at route progress."""

        query = np.asarray(progress, dtype=float)
        if not np.all(np.isfinite(query)):
            raise ValueError("progress must be finite")
        clipped = np.clip(query, 0.0, self.length)
        values = tuple(
            np.interp(clipped, self.s, data)
            for data in (self.x, self.y, self.heading, self.curvature)
        )
        if query.ndim == 0:
            return tuple(float(value) for value in values)
        return values


@dataclass(frozen=True)
class CircularObstacle:
    """Static circular obstacle used for footprint-clearance checks."""

    obstacle_id: str
    x: float
    y: float
    radius: float

    def __post_init__(self) -> None:
        if not self.obstacle_id:
            raise ValueError("obstacle_id must be nonempty")
        if (
            not np.isfinite(self.x)
            or not np.isfinite(self.y)
            or not np.isfinite(self.radius)
            or self.radius <= 0.0
        ):
            raise ValueError("static obstacle geometry must be finite and positive")


@dataclass(frozen=True)
class ConflictZone:
    """A geometric reservation zone shared by named routes."""

    zone_id: str
    x: float
    y: float
    radius: float
    route_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.zone_id or len(self.route_names) < 2:
            raise ValueError("a conflict zone needs a name and at least two routes")
        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("conflict-zone radius must be positive")


@dataclass(frozen=True)
class KinematicMap:
    """Named collection of routes, obstacles, and reservation conflicts."""

    name: str
    routes: tuple[SampledPath, ...]
    static_obstacles: tuple[CircularObstacle, ...] = ()
    conflict_zones: tuple[ConflictZone, ...] = ()
    speed_limit: float = 1.8

    def __post_init__(self) -> None:
        if not self.name or not self.routes:
            raise ValueError("map name and at least one route are required")
        names = [route.name for route in self.routes]
        if len(set(names)) != len(names):
            raise ValueError("route names must be unique within a map")
        if not np.isfinite(self.speed_limit) or self.speed_limit <= 0.0:
            raise ValueError("map speed_limit must be positive")
        known = set(names)
        for zone in self.conflict_zones:
            if not set(zone.route_names).issubset(known):
                raise ValueError("conflict zone references an unknown route")


@dataclass(frozen=True)
class KinematicSpeedProfile:
    """Arc-length speed plan and constraint diagnostics."""

    path: SampledPath
    speed: FloatArray
    curvature_envelope: FloatArray
    nominal_time: float
    straight_reference_time: float
    turning_penalty: float
    max_abs_yaw_rate: float
    max_abs_yaw_acceleration: float
    max_lateral_acceleration: float
    max_abs_wheel_speed: float
    envelope_binding_fraction: float


@dataclass(frozen=True)
class RouteObstacle:
    """Time-requested hard occupation of an interval along one route."""

    obstacle_id: str
    kind: str
    s: float
    requested_start: float
    duration: float
    half_width: float = 0.35
    guarded: bool = True
    reaction_time_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not self.obstacle_id or not self.kind:
            raise ValueError("obstacle_id and kind must be nonempty")
        numeric = (
            self.s,
            self.requested_start,
            self.duration,
            self.half_width,
            self.reaction_time_multiplier,
        )
        if any(not np.isfinite(value) for value in numeric):
            raise ValueError("route-obstacle values must be finite")
        if (
            self.s <= 0.0
            or self.requested_start < 0.0
            or self.duration <= 0.0
            or self.half_width <= 0.0
            or self.reaction_time_multiplier < 1.0
        ):
            raise ValueError("invalid route-obstacle timing or geometry")


@dataclass(frozen=True)
class ObstacleActivation:
    obstacle_id: str
    kind: str
    requested_start: float
    activated_at: float
    released_at: float

    @property
    def activation_delay(self) -> float:
        return self.activated_at - self.requested_start


@dataclass(frozen=True)
class RouteTrajectory:
    time: FloatArray
    s: FloatArray
    x: FloatArray
    y: FloatArray
    heading: FloatArray
    speed: FloatArray
    yaw_rate: FloatArray
    lateral_acceleration: FloatArray


@dataclass(frozen=True)
class RouteSimulationResult:
    """One invariant-checked path traversal with route obstructions."""

    path_name: str
    nominal_time: float
    traversal_time: float
    delay: float
    severity_weighted_loss: float
    activations: tuple[ObstacleActivation, ...]
    collision_count: int
    obstacle_violation_count: int
    minimum_static_clearance: float
    max_abs_yaw_rate: float
    max_abs_yaw_acceleration: float
    max_lateral_acceleration: float
    max_abs_wheel_speed: float
    minimum_stopping_margin: float
    trajectory: RouteTrajectory


class UnsafeObstacleActivationError(RuntimeError):
    """Raised when an unguarded occupation violates the braking contract."""


def _line_segment(start: Sequence[float], end: Sequence[float]) -> FloatArray:
    p0 = np.asarray(start, dtype=float)
    p3 = np.asarray(end, dtype=float)
    delta = p3 - p0
    return np.vstack((p0, p0 + delta / 3.0, p0 + 2.0 * delta / 3.0, p3))


def _bezier_segment(control: FloatArray, parameter: FloatArray) -> FloatArray:
    one_minus = 1.0 - parameter
    return (
        one_minus[:, None] ** 3 * control[0]
        + 3.0 * one_minus[:, None] ** 2 * parameter[:, None] * control[1]
        + 3.0 * one_minus[:, None] * parameter[:, None] ** 2 * control[2]
        + parameter[:, None] ** 3 * control[3]
    )


def sample_bezier_route(
    name: str,
    segments: Sequence[ArrayLike],
    *,
    samples_per_segment: int = 100,
) -> SampledPath:
    """Sample a chain of cubic Bezier segments and derive arc geometry."""

    if samples_per_segment < 12:
        raise ValueError("samples_per_segment must be at least 12")
    pieces: list[FloatArray] = []
    previous_end: FloatArray | None = None
    for index, raw in enumerate(segments):
        control = np.asarray(raw, dtype=float)
        if control.shape != (4, 2) or not np.all(np.isfinite(control)):
            raise ValueError("each Bezier segment must have four finite 2D controls")
        if previous_end is not None and not np.allclose(
            control[0], previous_end, rtol=0.0, atol=1e-9
        ):
            raise ValueError("Bezier segments must be position-continuous")
        parameter = np.linspace(0.0, 1.0, samples_per_segment)
        points = _bezier_segment(control, parameter)
        pieces.append(points if index == 0 else points[1:])
        previous_end = control[-1]
    if not pieces:
        raise ValueError("at least one Bezier segment is required")
    points = np.vstack(pieces)
    step = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    if np.any(step <= 1e-10):
        raise ValueError("route sampling produced duplicate points")
    arc = np.concatenate(([0.0], np.cumsum(step)))
    dx = np.gradient(points[:, 0], arc, edge_order=2)
    dy = np.gradient(points[:, 1], arc, edge_order=2)
    heading = np.unwrap(np.arctan2(dy, dx))
    curvature = np.gradient(heading, arc, edge_order=2)
    curvature_rate = np.gradient(curvature, arc, edge_order=2)
    return SampledPath(
        name=name,
        x=points[:, 0],
        y=points[:, 1],
        s=arc,
        heading=heading,
        curvature=curvature,
        curvature_rate=curvature_rate,
    )


def straight_route(
    name: str = "straight",
    *,
    length: float = 30.0,
    samples: int = 160,
) -> SampledPath:
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("length must be positive")
    return sample_bezier_route(
        name, (_line_segment((0.0, 0.0), (length, 0.0)),),
        samples_per_segment=samples,
    )


def _map_straight() -> KinematicMap:
    route = straight_route("main")
    obstacles = (
        CircularObstacle("shelf_north", 10.0, 1.45, 0.45),
        CircularObstacle("shelf_south", 20.0, -1.45, 0.45),
    )
    return KinematicMap("straight_crossing", (route,), obstacles, speed_limit=1.8)


def _map_l_turn() -> KinematicMap:
    curve = np.array(
        [[10.0, 0.0], [12.209, 0.0], [14.0, 1.791], [14.0, 4.0]]
    )
    route = sample_bezier_route(
        "main",
        (
            _line_segment((0.0, 0.0), (10.0, 0.0)),
            curve,
            _line_segment((14.0, 4.0), (14.0, 20.0)),
        ),
        samples_per_segment=90,
    )
    obstacles = (
        CircularObstacle("inside_pillar", 10.8, 2.1, 0.42),
        CircularObstacle("outside_pillar", 16.0, 3.2, 0.55),
    )
    return KinematicMap("l_turn", (route,), obstacles, speed_limit=1.6)


def _map_s_curve() -> KinematicMap:
    route = sample_bezier_route(
        "main",
        (
            _line_segment((0.0, 0.0), (5.0, 0.0)),
            np.array([[5.0, 0.0], [9.0, 0.0], [11.0, 4.0], [15.0, 4.0]]),
            np.array([[15.0, 4.0], [19.0, 4.0], [21.0, 0.0], [25.0, 0.0]]),
            _line_segment((25.0, 0.0), (32.0, 0.0)),
        ),
        samples_per_segment=80,
    )
    obstacles = (
        CircularObstacle("chicane_north", 10.0, 5.4, 0.55),
        CircularObstacle("chicane_south", 21.0, -1.5, 0.50),
    )
    return KinematicMap("s_curve", (route,), obstacles, speed_limit=1.7)


def _map_merge() -> KinematicMap:
    north = sample_bezier_route(
        "north_in",
        (
            np.array([[0.0, 4.0], [6.0, 4.0], [8.0, 0.0], [14.0, 0.0]]),
            _line_segment((14.0, 0.0), (32.0, 0.0)),
        ),
        samples_per_segment=110,
    )
    south = sample_bezier_route(
        "south_in",
        (
            np.array([[0.0, -4.0], [6.0, -4.0], [8.0, 0.0], [14.0, 0.0]]),
            _line_segment((14.0, 0.0), (32.0, 0.0)),
        ),
        samples_per_segment=110,
    )
    zone = ConflictZone("merge_reservation", 12.0, 0.0, 2.1, ("north_in", "south_in"))
    obstacles = (
        CircularObstacle("merge_island", 8.0, 5.5, 0.48),
        CircularObstacle("merge_island_2", 8.0, -5.5, 0.48),
    )
    return KinematicMap("two_to_one_merge", (north, south), obstacles, (zone,), 1.5)


def _map_intersection() -> KinematicMap:
    east = sample_bezier_route(
        "eastbound",
        (_line_segment((-16.0, 0.0), (16.0, 0.0)),),
        samples_per_segment=180,
    )
    north = sample_bezier_route(
        "northbound",
        (_line_segment((0.0, -16.0), (0.0, 16.0)),),
        samples_per_segment=180,
    )
    zone = ConflictZone(
        "intersection_reservation", 0.0, 0.0, 1.4, ("eastbound", "northbound")
    )
    obstacles = (
        CircularObstacle("corner_ne", 2.0, 2.0, 0.42),
        CircularObstacle("corner_sw", -2.0, -2.0, 0.42),
    )
    return KinematicMap("four_way_intersection", (east, north), obstacles, (zone,), 1.4)


def _map_warehouse_grid() -> KinematicMap:
    route = sample_bezier_route(
        "serpentine",
        (
            _line_segment((0.0, 0.0), (12.0, 0.0)),
            np.array([[12.0, 0.0], [14.2, 0.0], [16.0, 1.8], [16.0, 3.0]]),
            np.array([[16.0, 3.0], [16.0, 4.2], [14.2, 6.0], [12.0, 6.0]]),
            _line_segment((12.0, 6.0), (0.0, 6.0)),
            np.array([[0.0, 6.0], [-2.2, 6.0], [-4.0, 7.8], [-4.0, 9.0]]),
            np.array([[-4.0, 9.0], [-4.0, 10.2], [-2.2, 12.0], [0.0, 12.0]]),
            _line_segment((0.0, 12.0), (16.0, 12.0)),
        ),
        samples_per_segment=75,
    )
    obstacles = tuple(
        CircularObstacle(f"shelf_{index}", x, y, 1.15)
        for index, (x, y) in enumerate(
            ((3.0, 3.0), (7.0, 3.0), (11.0, 3.0), (3.0, 9.0), (8.0, 9.0))
        )
    )
    return KinematicMap("warehouse_grid", (route,), obstacles, speed_limit=1.35)


def standard_map_catalogue() -> tuple[KinematicMap, ...]:
    """Six deterministic maps spanning turns, merges, conflicts, and aisles."""

    return (
        _map_straight(),
        _map_l_turn(),
        _map_s_curve(),
        _map_merge(),
        _map_intersection(),
        _map_warehouse_grid(),
    )


def minimum_static_clearance(
    map_spec: KinematicMap,
    limits: DifferentialDriveLimits,
) -> float:
    """Minimum obstacle-to-footprint clearance over every sampled route."""

    if not map_spec.static_obstacles:
        return float("inf")
    minimum = float("inf")
    padding = limits.footprint_radius + limits.safety_margin
    for route in map_spec.routes:
        for obstacle in map_spec.static_obstacles:
            distance = np.hypot(route.x - obstacle.x, route.y - obstacle.y)
            minimum = min(minimum, float(np.min(distance - obstacle.radius - padding)))
    return minimum


def assert_static_clearance(
    map_spec: KinematicMap,
    limits: DifferentialDriveLimits,
    *,
    tolerance: float = 1e-9,
) -> float:
    clearance = minimum_static_clearance(map_spec, limits)
    if clearance < -tolerance:
        raise ValueError(
            f"map {map_spec.name} intersects the robot footprint by {-clearance:.6g} m"
        )
    return clearance


def curvature_speed_envelope(
    path: SampledPath,
    limits: DifferentialDriveLimits,
    *,
    map_speed_limit: float | None = None,
    yaw_acceleration_split: float = 0.5,
) -> FloatArray:
    """Pointwise speed envelope from curvature and differential-drive limits."""

    if map_speed_limit is None:
        map_limit = limits.max_speed
    else:
        if not np.isfinite(map_speed_limit) or map_speed_limit <= 0.0:
            raise ValueError("map_speed_limit must be positive")
        map_limit = float(map_speed_limit)
    if not 0.0 < yaw_acceleration_split < 1.0:
        raise ValueError("yaw_acceleration_split must lie strictly between zero and one")
    abs_curvature = np.abs(path.curvature)
    abs_rate = np.abs(path.curvature_rate)
    epsilon = 1e-12
    base = min(limits.max_speed, limits.max_wheel_speed, map_limit)
    envelope = np.full(path.s.size, base, dtype=float)

    curved = abs_curvature > epsilon
    envelope[curved] = np.minimum(
        envelope[curved], limits.max_yaw_rate / abs_curvature[curved]
    )
    envelope[curved] = np.minimum(
        envelope[curved],
        np.sqrt(limits.max_lateral_acceleration / abs_curvature[curved]),
    )
    envelope[curved] = np.minimum(
        envelope[curved],
        limits.max_wheel_speed
        / (1.0 + 0.5 * limits.wheel_track * abs_curvature[curved]),
    )

    changing = abs_rate > epsilon
    curvature_rate_budget = (
        (1.0 - yaw_acceleration_split) * limits.max_yaw_acceleration
    )
    envelope[changing] = np.minimum(
        envelope[changing], np.sqrt(curvature_rate_budget / abs_rate[changing])
    )
    return np.maximum(envelope, 0.0)


def _segment_tangential_limits(
    path: SampledPath,
    limits: DifferentialDriveLimits,
    yaw_acceleration_split: float,
) -> tuple[FloatArray, FloatArray]:
    segment_curvature = np.maximum(
        np.abs(path.curvature[:-1]), np.abs(path.curvature[1:])
    )
    yaw_budget = yaw_acceleration_split * limits.max_yaw_acceleration
    curvature_cap = np.full_like(segment_curvature, np.inf)
    mask = segment_curvature > 1e-12
    curvature_cap[mask] = yaw_budget / segment_curvature[mask]
    return (
        np.minimum(limits.max_acceleration, curvature_cap),
        np.minimum(limits.max_deceleration, curvature_cap),
    )


def _solve_arc_speed(
    arc: FloatArray,
    envelope: FloatArray,
    acceleration: FloatArray,
    deceleration: FloatArray,
    start_speed: float,
    end_speed: float,
) -> FloatArray:
    speed = np.asarray(envelope, dtype=float).copy()
    speed[0] = min(speed[0], start_speed)
    speed[-1] = min(speed[-1], end_speed)
    distance = np.diff(arc)
    for _ in range(12):
        previous = speed.copy()
        for index, step in enumerate(distance):
            reachable = np.sqrt(
                max(0.0, speed[index] ** 2 + 2.0 * acceleration[index] * step)
            )
            speed[index + 1] = min(speed[index + 1], reachable)
        speed[-1] = min(speed[-1], end_speed)
        for index in range(speed.size - 2, -1, -1):
            reachable = np.sqrt(
                max(
                    0.0,
                    speed[index + 1] ** 2
                    + 2.0 * deceleration[index] * distance[index],
                )
            )
            speed[index] = min(speed[index], reachable)
        speed[0] = min(speed[0], start_speed)
        if np.max(np.abs(speed - previous)) < 1e-11:
            break
    return speed


def _integrate_profile_time(arc: FloatArray, speed: FloatArray) -> float:
    denominator = speed[:-1] + speed[1:]
    if np.any(denominator <= 1e-12):
        raise ValueError("speed profile contains an untraversable zero-speed segment")
    return float(np.sum(2.0 * np.diff(arc) / denominator))


def plan_kinematic_speed_profile(
    path: SampledPath,
    limits: DifferentialDriveLimits,
    *,
    map_speed_limit: float | None = None,
    start_speed: float = 0.0,
    end_speed: float = 0.0,
    yaw_acceleration_split: float = 0.5,
) -> KinematicSpeedProfile:
    """Compute a forward/backward arc-speed profile with hard turn limits."""

    if (
        not np.isfinite(start_speed)
        or not np.isfinite(end_speed)
        or start_speed < 0.0
        or end_speed < 0.0
    ):
        raise ValueError("endpoint speeds must be finite and nonnegative")
    envelope = curvature_speed_envelope(
        path,
        limits,
        map_speed_limit=map_speed_limit,
        yaw_acceleration_split=yaw_acceleration_split,
    )
    acceleration, deceleration = _segment_tangential_limits(
        path, limits, yaw_acceleration_split
    )
    speed = _solve_arc_speed(
        path.s, envelope, acceleration, deceleration, start_speed, end_speed
    )
    nominal_time = _integrate_profile_time(path.s, speed)

    straight_cap = min(
        limits.max_speed,
        limits.max_wheel_speed,
        limits.max_speed if map_speed_limit is None else float(map_speed_limit),
    )
    straight_envelope = np.full(path.s.size, straight_cap)
    straight_speed = _solve_arc_speed(
        path.s,
        straight_envelope,
        np.full(path.s.size - 1, limits.max_acceleration),
        np.full(path.s.size - 1, limits.max_deceleration),
        start_speed,
        end_speed,
    )
    straight_time = _integrate_profile_time(path.s, straight_speed)

    segment_distance = np.diff(path.s)
    acceleration_value = (
        speed[1:] ** 2 - speed[:-1] ** 2
    ) / (2.0 * segment_distance)
    midpoint_speed = 0.5 * (speed[:-1] + speed[1:])
    midpoint_curvature = 0.5 * (path.curvature[:-1] + path.curvature[1:])
    midpoint_rate = 0.5 * (path.curvature_rate[:-1] + path.curvature_rate[1:])
    yaw_acceleration = (
        acceleration_value * midpoint_curvature
        + midpoint_speed**2 * midpoint_rate
    )
    yaw_rate = speed * path.curvature
    lateral = speed**2 * np.abs(path.curvature)
    left_wheel = speed * (1.0 - 0.5 * limits.wheel_track * path.curvature)
    right_wheel = speed * (1.0 + 0.5 * limits.wheel_track * path.curvature)
    binding = np.isclose(speed, envelope, rtol=1e-5, atol=1e-7)
    return KinematicSpeedProfile(
        path=path,
        speed=speed,
        curvature_envelope=envelope,
        nominal_time=nominal_time,
        straight_reference_time=straight_time,
        turning_penalty=max(0.0, nominal_time - straight_time),
        max_abs_yaw_rate=float(np.max(np.abs(yaw_rate))),
        max_abs_yaw_acceleration=float(np.max(np.abs(yaw_acceleration))),
        max_lateral_acceleration=float(np.max(lateral)),
        max_abs_wheel_speed=float(
            max(np.max(np.abs(left_wheel)), np.max(np.abs(right_wheel)))
        ),
        envelope_binding_fraction=float(np.mean(binding)),
    )


def _local_braking_limit(
    path: SampledPath,
    limits: DifferentialDriveLimits,
    start_s: float,
    end_s: float,
    yaw_acceleration_split: float,
) -> float:
    lower, upper = sorted((start_s, end_s))
    mask = (path.s >= lower) & (path.s <= upper)
    curvature = np.abs(path.curvature[mask])
    if curvature.size == 0:
        curvature = np.array(
            [abs(float(np.interp(lower, path.s, path.curvature)))]
        )
    peak = float(np.max(curvature))
    if peak <= 1e-12:
        return limits.max_deceleration
    yaw_cap = (
        yaw_acceleration_split * limits.max_yaw_acceleration / peak
    )
    return float(min(limits.max_deceleration, yaw_cap))


def _safe_to_activate(
    obstacle: RouteObstacle,
    path: SampledPath,
    limits: DifferentialDriveLimits,
    progress: float,
    speed: float,
    yaw_acceleration_split: float,
) -> bool:
    entry = obstacle.s - obstacle.half_width
    exit_position = obstacle.s + obstacle.half_width
    if progress >= exit_position:
        return True
    if progress > entry:
        return False
    braking = _local_braking_limit(
        path, limits, progress, entry, yaw_acceleration_split
    )
    required = float(
        d_stop(
            speed,
            limits.longitudinal_clearance,
            limits.reaction_time * obstacle.reaction_time_multiplier,
            braking,
        )
    )
    return required <= entry - progress + 1e-10


def simulate_route_obstacles(
    map_spec: KinematicMap,
    route_name: str,
    limits: DifferentialDriveLimits,
    obstacles: Iterable[RouteObstacle] = (),
    *,
    dt: float = 0.02,
    sensor_range: float = 10.0,
    yaw_acceleration_split: float = 0.5,
    maximum_time: float | None = None,
) -> RouteSimulationResult:
    """Traverse one map route under guarded hard-occupation disturbances."""

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not np.isfinite(sensor_range) or sensor_range <= 0.0:
        raise ValueError("sensor_range must be finite and positive")
    route_lookup = {route.name: route for route in map_spec.routes}
    if route_name not in route_lookup:
        raise ValueError(f"unknown route {route_name}")
    path = route_lookup[route_name]
    records = tuple(obstacles)
    identifiers = [item.obstacle_id for item in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("route obstacle identifiers must be unique")
    if any(
        item.s - item.half_width <= 0.0
        or item.s + item.half_width >= path.length
        for item in records
    ):
        raise ValueError("route obstacles must lie strictly inside the path")

    static_clearance = assert_static_clearance(map_spec, limits)
    largest_reaction = max(
        (item.reaction_time_multiplier for item in records), default=1.0
    )
    required_sensor = float(
        d_stop(
            min(limits.max_speed, map_spec.speed_limit),
            limits.longitudinal_clearance,
            limits.reaction_time * largest_reaction,
            limits.max_deceleration,
        )
    )
    if sensor_range + 1e-12 < required_sensor:
        raise ValueError(
            "sensor_range must cover the maximum-speed stopping distance"
        )

    profile = plan_kinematic_speed_profile(
        path,
        limits,
        map_speed_limit=map_spec.speed_limit,
        yaw_acceleration_split=yaw_acceleration_split,
    )
    if maximum_time is None:
        maximum_time = (
            profile.nominal_time
            + sum(item.duration for item in records)
            + max((item.requested_start for item in records), default=0.0)
            + 30.0
        )
    if not np.isfinite(maximum_time) or maximum_time <= profile.nominal_time:
        raise ValueError("maximum_time must exceed nominal traversal time")

    pending = {item.obstacle_id: item for item in records}
    active: dict[str, tuple[RouteObstacle, float, float]] = {}
    activations: list[ObstacleActivation] = []
    progress = 0.0
    speed = 0.0
    time = 0.0
    severity = 0.0
    minimum_stopping_margin = float("inf")
    violation_count = 0

    times = [time]
    progresses = [progress]
    speeds = [speed]
    x0, y0, heading0, curvature0 = path.pose(progress)
    xs = [x0]
    ys = [y0]
    headings = [heading0]
    yaw_rates = [speed * curvature0]
    lateral_values = [0.0]

    while progress < path.length - 1e-9 and time < maximum_time:
        for obstacle_id, obstacle in tuple(pending.items()):
            if obstacle.requested_start > time + 1e-12:
                continue
            safe = _safe_to_activate(
                obstacle,
                path,
                limits,
                progress,
                speed,
                yaw_acceleration_split,
            )
            if obstacle.guarded and not safe:
                continue
            if not obstacle.guarded and not safe:
                raise UnsafeObstacleActivationError(
                    f"unguarded obstacle {obstacle_id} violates the braking contract"
                )
            released = time + obstacle.duration
            active[obstacle_id] = (obstacle, time, released)
            activations.append(
                ObstacleActivation(
                    obstacle_id,
                    obstacle.kind,
                    obstacle.requested_start,
                    time,
                    released,
                )
            )
            del pending[obstacle_id]

        for obstacle_id, (_, _, released) in tuple(active.items()):
            if time >= released - 1e-12:
                del active[obstacle_id]

        nearest: RouteObstacle | None = None
        nearest_distance = float("inf")
        for obstacle, _, _ in active.values():
            entry = obstacle.s - obstacle.half_width
            exit_position = obstacle.s + obstacle.half_width
            if entry < progress < exit_position:
                violation_count += 1
            if progress <= entry and entry - progress < nearest_distance:
                nearest = obstacle
                nearest_distance = entry - progress

        target_s = min(path.length, progress + max(speed, 0.15) * dt)
        nominal_cap = float(np.interp(target_s, path.s, profile.speed))
        safety_cap = float("inf")
        braking_floor = limits.max_deceleration
        reaction = limits.reaction_time
        if nearest is not None:
            braking_floor = _local_braking_limit(
                path,
                limits,
                progress,
                nearest.s - nearest.half_width,
                yaw_acceleration_split,
            )
            reaction = limits.reaction_time * nearest.reaction_time_multiplier
            stopping_margin = nearest_distance - float(
                d_stop(
                    speed,
                    limits.longitudinal_clearance,
                    reaction,
                    braking_floor,
                )
            )
            minimum_stopping_margin = min(
                minimum_stopping_margin, stopping_margin
            )
            if nearest_distance <= sensor_range:
                safety_cap = float(
                    v_safe_next_step(
                        nearest_distance,
                        speed,
                        limits.longitudinal_clearance,
                        reaction,
                        braking_floor,
                        dt,
                    )
                )

        local_braking = _local_braking_limit(
            path,
            limits,
            progress,
            min(path.length, progress + max(sensor_range, speed * dt)),
            yaw_acceleration_split,
        )
        reachable_floor = max(0.0, speed - local_braking * dt)
        reachable_ceiling = speed + limits.max_acceleration * dt
        requested_speed = min(nominal_cap, reachable_ceiling, safety_cap)
        if requested_speed < reachable_floor - 2e-8:
            if safety_cap < reachable_floor - 2e-8:
                raise UnsafeObstacleActivationError(
                    "the shield requested deceleration beyond the guaranteed limit"
                )
            next_speed = reachable_floor
        else:
            next_speed = max(0.0, requested_speed)

        movement = 0.5 * (speed + next_speed) * dt
        next_progress = min(path.length, progress + movement)
        if nearest is not None and next_progress > nearest.s - nearest.half_width + 1e-9:
            violation_count += 1
        reference = float(np.interp(progress, path.s, profile.speed))
        if reference > 1e-6:
            severity += max(0.0, 1.0 - speed / reference) * dt

        progress = next_progress
        speed = next_speed
        time += dt
        x, y, heading, curvature = path.pose(progress)
        times.append(time)
        progresses.append(progress)
        speeds.append(speed)
        xs.append(x)
        ys.append(y)
        headings.append(heading)
        yaw_rates.append(speed * curvature)
        lateral_values.append(speed**2 * abs(curvature))

    if progress < path.length - 1e-7:
        raise RuntimeError("route traversal did not finish before maximum_time")
    time_array = np.asarray(times, dtype=float)
    speed_array = np.asarray(speeds, dtype=float)
    yaw_array = np.asarray(yaw_rates, dtype=float)
    curvature_array = np.interp(np.asarray(progresses), path.s, path.curvature)
    left_wheel = speed_array * (
        1.0 - 0.5 * limits.wheel_track * curvature_array
    )
    right_wheel = speed_array * (
        1.0 + 0.5 * limits.wheel_track * curvature_array
    )
    yaw_acceleration = (
        np.diff(yaw_array) / np.diff(time_array)
        if time_array.size > 1
        else np.zeros(1)
    )
    if not np.isfinite(minimum_stopping_margin):
        minimum_stopping_margin = float("inf")
    return RouteSimulationResult(
        path_name=path.name,
        nominal_time=profile.nominal_time,
        traversal_time=float(time),
        delay=max(0.0, float(time - profile.nominal_time)),
        severity_weighted_loss=float(severity),
        activations=tuple(activations),
        collision_count=0 if static_clearance >= -1e-9 else 1,
        obstacle_violation_count=violation_count,
        minimum_static_clearance=static_clearance,
        max_abs_yaw_rate=float(np.max(np.abs(yaw_array))),
        max_abs_yaw_acceleration=float(np.max(np.abs(yaw_acceleration))),
        max_lateral_acceleration=float(np.max(lateral_values)),
        max_abs_wheel_speed=float(
            max(np.max(np.abs(left_wheel)), np.max(np.abs(right_wheel)))
        ),
        minimum_stopping_margin=minimum_stopping_margin,
        trajectory=RouteTrajectory(
            time=time_array,
            s=np.asarray(progresses, dtype=float),
            x=np.asarray(xs, dtype=float),
            y=np.asarray(ys, dtype=float),
            heading=np.asarray(headings, dtype=float),
            speed=speed_array,
            yaw_rate=yaw_array,
            lateral_acceleration=np.asarray(lateral_values, dtype=float),
        ),
    )


def _low_curvature_location(path: SampledPath, fraction: float) -> float:
    center = fraction * path.length
    width = 0.10 * path.length
    candidates = np.flatnonzero(
        (path.s >= center - width) & (path.s <= center + width)
    )
    if candidates.size == 0:
        return center
    scale = max(1.0, path.length)
    score = (
        np.abs(path.curvature[candidates])
        + np.abs(path.curvature_rate[candidates]) / scale
        + 0.02 * np.abs(path.s[candidates] - center) / scale
    )
    return float(path.s[candidates[int(np.argmin(score))]])


def standard_obstacle_suite(
    path: SampledPath,
    nominal_time: float,
) -> tuple[RouteObstacle, ...]:
    """Five reproducible obstruction mechanisms with comparable timing."""

    if not np.isfinite(nominal_time) or nominal_time <= 0.0:
        raise ValueError("nominal_time must be positive")
    return (
        RouteObstacle(
            "stationary_pallet",
            "unexpected_stationary",
            _low_curvature_location(path, 0.38),
            0.0,
            0.55 * nominal_time,
            guarded=False,
        ),
        RouteObstacle(
            "pedestrian",
            "pedestrian_crossing",
            _low_curvature_location(path, 0.54),
            0.32 * nominal_time,
            0.28 * nominal_time,
        ),
        RouteObstacle(
            "forklift",
            "moving_forklift_crossing",
            _low_curvature_location(path, 0.68),
            0.44 * nominal_time,
            0.30 * nominal_time,
            half_width=0.55,
        ),
        RouteObstacle(
            "aisle_closure",
            "temporary_aisle_closure",
            _low_curvature_location(path, 0.80),
            0.48 * nominal_time,
            0.42 * nominal_time,
            half_width=0.70,
        ),
        RouteObstacle(
            "occluded_pedestrian",
            "occlusion_delayed_detection",
            _low_curvature_location(path, 0.58),
            0.34 * nominal_time,
            0.30 * nominal_time,
            reaction_time_multiplier=1.8,
        ),
    )


def run_cross_map_obstacle_benchmark(
    limits: DifferentialDriveLimits | None = None,
    *,
    dt: float = 0.02,
    sensor_range: float = 10.0,
) -> tuple[tuple[str, str, str, RouteSimulationResult], ...]:
    """Run each obstacle mechanism independently on every catalogue route."""

    robot = DifferentialDriveLimits() if limits is None else limits
    rows: list[tuple[str, str, str, RouteSimulationResult]] = []
    for map_spec in standard_map_catalogue():
        assert_static_clearance(map_spec, robot)
        for route in map_spec.routes:
            profile = plan_kinematic_speed_profile(
                route, robot, map_speed_limit=map_spec.speed_limit
            )
            for obstacle in standard_obstacle_suite(route, profile.nominal_time):
                result = simulate_route_obstacles(
                    map_spec,
                    route.name,
                    robot,
                    (obstacle,),
                    dt=dt,
                    sensor_range=sensor_range,
                )
                if (
                    result.collision_count != 0
                    or result.obstacle_violation_count != 0
                    or result.minimum_stopping_margin < -1e-7
                ):
                    raise AssertionError(
                        f"safety invariant failed on {map_spec.name}/{route.name}"
                    )
                rows.append((map_spec.name, route.name, obstacle.kind, result))
    return tuple(rows)
