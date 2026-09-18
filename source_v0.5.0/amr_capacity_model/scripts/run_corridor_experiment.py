#!/usr/bin/env python3
"""Run the first reproducible paired-rollout corridor experiment."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from amr_capacity.capacity_theory import d_stop
from amr_capacity.estimation import (
    pool_direct_branching,
    pool_horizon_branching,
    summarize_paired_rollout,
    validate_branching_out_of_sample,
    validate_finite_chain_out_of_sample,
)
from amr_capacity.geometry import RectangularLoop
from amr_capacity.simulation import (
    CorridorConfig,
    CrossingEvent,
    PairedRolloutResult,
    run_paired_rollout,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
SEED = 20260914
SPEEDS = np.array([0.6, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0])
ROBOT_COUNT = 20
ROLLOUTS_PER_SPEED = 12
TRAINING_ROLLOUTS = ROLLOUTS_PER_SPEED // 2
DENSITY_ROBOT_COUNTS = (8, 12, 16, 20, 24, 28, 32, 33)
DENSITY_SWEEP_SPEED = 1.2
EVENT_START = 4.0
SIMULATION_DURATION = 120.0
HORIZONS = (1.0, 2.0, 4.0, 8.0)


@dataclass(frozen=True)
class SpeedExperiment:
    speed: float
    rollouts: tuple[PairedRolloutResult, ...]
    direct_B: float
    predicted_progeny: float
    truncated_predicted_progeny: float
    measured_progeny: float
    relative_error: float
    truncated_relative_error: float
    mean_g: float
    mean_severity_loss: float
    mean_traversal_loss: float
    mean_treated_traversal_rate: float
    mean_control_traversal_rate: float
    mean_attributable_rate_loss: float
    stopping_headway_margin: float
    finite_chain_phi: float
    finite_chain_infinite_prediction: float | None
    finite_chain_finite_prediction: float
    finite_chain_measured: float
    finite_chain_infinite_error: float | None
    finite_chain_finite_error: float
    training_resolved_cascades: int
    training_saturated_cascades: int
    finite_chain_boundary_identity: bool
    finite_chain_validation_status: str
    horizon_B: tuple[float, ...]


@dataclass(frozen=True)
class DensityExperiment:
    robot_count: int
    rollouts: tuple[PairedRolloutResult, ...]
    headway: float
    direct_B: float
    B_per_robot: float
    predicted_progeny: float
    truncated_predicted_progeny: float
    measured_progeny: float
    relative_error: float
    truncated_relative_error: float
    mean_g: float
    mean_treated_traversal_rate: float
    mean_control_traversal_rate: float
    mean_attributable_rate_loss: float
    stopping_headway_margin: float
    finite_chain_phi: float
    finite_chain_infinite_prediction: float | None
    finite_chain_finite_prediction: float
    finite_chain_measured: float
    finite_chain_infinite_error: float | None
    finite_chain_finite_error: float
    training_resolved_cascades: int
    training_saturated_cascades: int
    finite_chain_boundary_identity: bool
    finite_chain_validation_status: str


def make_config(speed: float) -> CorridorConfig:
    return CorridorConfig(
        loop=RectangularLoop(20.0, 10.0),
        desired_speed=speed,
        acceleration=1.0,
        braking=1.5,
        reaction_time=0.24,
        dt=0.02,
        robot_length=0.8,
        safety_margin=0.2,
        sensor_range=8.0,
        crossing_s=7.0,
        crossing_half_width=0.4,
    )


def generate_rollouts(
    speed: float,
    robot_count: int,
    rng: np.random.Generator,
) -> tuple[PairedRolloutResult, ...]:
    config = make_config(speed)
    headway = config.route_length / robot_count
    required_gap = float(
        d_stop(
            speed,
            config.crossing_clearance,
            config.reaction_time,
            config.braking,
        )
    )
    safety_slack = headway - required_gap
    if safety_slack <= config.collision_tolerance:
        raise RuntimeError(
            "robot count and speed violate the stopping-distance headway"
        )
    trigger_guard = min(0.12, 0.25 * safety_slack)
    lower_gap = required_gap + trigger_guard
    upper_gap = min(headway - trigger_guard, config.sensor_range - 0.2)
    if lower_gap >= upper_gap:
        raise RuntimeError("experiment has no safe crossing-trigger gap interval")

    rollouts: list[PairedRolloutResult] = []
    for replicate in range(ROLLOUTS_PER_SPEED):
        target_gap = float(rng.uniform(lower_gap, upper_gap))
        event_duration = float(rng.uniform(3.0, 7.0))
        # Choose only the initial phase; the physical crossing remains fixed.
        initial_offset = (
            config.normalized_crossing_s - speed * EVENT_START - target_gap
        ) % headway
        pair = run_paired_rollout(
            config,
            SIMULATION_DURATION,
            [CrossingEvent(EVENT_START, event_duration, replicate)],
            robot_count=robot_count,
            initial_offset=initial_offset,
            record_stride=25,
        )
        if pair.treated.collision_count or pair.treated.crossing_violation_count:
            raise AssertionError("safety invariant failed during experiment")
        rollouts.append(pair)
    return tuple(rollouts)


def run_speed_condition(
    speed: float, rng: np.random.Generator
) -> SpeedExperiment:
    rollouts = generate_rollouts(speed, ROBOT_COUNT, rng)

    training = rollouts[:TRAINING_ROLLOUTS]
    validation = rollouts[TRAINING_ROLLOUTS:]
    direct = pool_direct_branching(training)
    held_out = validate_branching_out_of_sample(
        training, validation, max_generation=ROBOT_COUNT - 1
    )
    finite_chain = validate_finite_chain_out_of_sample(
        training, validation, ROBOT_COUNT
    )
    if (
        held_out.predicted_total_progeny_per_primary is None
        or held_out.relative_error is None
        or held_out.truncated_predicted_total_progeny_per_primary is None
        or held_out.truncated_relative_error is None
    ):
        raise RuntimeError("training estimate was not subcritical")
    scalar_metrics = tuple(summarize_paired_rollout(pair) for pair in rollouts)
    treated_rates = np.asarray(
        [pair.treated.traversal_rate for pair in rollouts], dtype=float
    )
    control_rates = np.asarray(
        [pair.control.traversal_rate for pair in rollouts], dtype=float
    )
    config = make_config(speed)
    headway = config.route_length / ROBOT_COUNT
    required_headway = float(
        d_stop(
            speed,
            config.robot_clearance,
            config.reaction_time,
            config.braking,
        )
    )
    horizon_values = tuple(
        float(pool_horizon_branching(training, horizon).matrix[0, 0])
        for horizon in HORIZONS
    )
    return SpeedExperiment(
        speed=speed,
        rollouts=rollouts,
        direct_B=float(direct.matrix[0, 0]),
        predicted_progeny=held_out.predicted_total_progeny_per_primary,
        truncated_predicted_progeny=(
            held_out.truncated_predicted_total_progeny_per_primary
        ),
        measured_progeny=held_out.measured_total_progeny_per_primary,
        relative_error=held_out.relative_error,
        truncated_relative_error=held_out.truncated_relative_error,
        mean_g=float(np.mean([metric.g_per_traversal for metric in scalar_metrics])),
        mean_severity_loss=float(
            np.mean([pair.attributable_severity for pair in rollouts])
        ),
        mean_traversal_loss=float(
            np.mean([pair.traversal_loss for pair in rollouts])
        ),
        mean_treated_traversal_rate=float(np.mean(treated_rates)),
        mean_control_traversal_rate=float(np.mean(control_rates)),
        mean_attributable_rate_loss=float(np.mean(control_rates - treated_rates)),
        stopping_headway_margin=headway - required_headway,
        finite_chain_phi=finite_chain.training.propagation_probability,
        finite_chain_infinite_prediction=finite_chain.infinite_prediction,
        finite_chain_finite_prediction=finite_chain.finite_prediction,
        finite_chain_measured=finite_chain.measured_unique_robots_per_primary,
        finite_chain_infinite_error=finite_chain.infinite_relative_error,
        finite_chain_finite_error=finite_chain.finite_relative_error,
        training_resolved_cascades=finite_chain.training.resolved_cascades,
        training_saturated_cascades=(
            finite_chain.training.fleet_saturated_cascades
        ),
        finite_chain_boundary_identity=(
            finite_chain.training.algebraic_boundary_identity
        ),
        finite_chain_validation_status=finite_chain.validation_status,
        horizon_B=horizon_values,
    )


def run_density_condition(
    robot_count: int, rng: np.random.Generator
) -> DensityExperiment:
    rollouts = generate_rollouts(DENSITY_SWEEP_SPEED, robot_count, rng)
    training = rollouts[:TRAINING_ROLLOUTS]
    validation = rollouts[TRAINING_ROLLOUTS:]
    direct = pool_direct_branching(training)
    held_out = validate_branching_out_of_sample(
        training, validation, max_generation=robot_count - 1
    )
    finite_chain = validate_finite_chain_out_of_sample(
        training, validation, robot_count
    )
    if any(
        value is None
        for value in (
            held_out.predicted_total_progeny_per_primary,
            held_out.truncated_predicted_total_progeny_per_primary,
            held_out.relative_error,
            held_out.truncated_relative_error,
        )
    ):
        raise RuntimeError("density-sweep training estimate was not subcritical")
    scalar_metrics = tuple(summarize_paired_rollout(pair) for pair in rollouts)
    treated_rates = np.asarray(
        [pair.treated.traversal_rate for pair in rollouts], dtype=float
    )
    control_rates = np.asarray(
        [pair.control.traversal_rate for pair in rollouts], dtype=float
    )
    config = make_config(DENSITY_SWEEP_SPEED)
    headway = config.route_length / robot_count
    required_headway = float(
        d_stop(
            DENSITY_SWEEP_SPEED,
            config.robot_clearance,
            config.reaction_time,
            config.braking,
        )
    )
    direct_value = float(direct.matrix[0, 0])
    return DensityExperiment(
        robot_count=robot_count,
        rollouts=rollouts,
        headway=headway,
        direct_B=direct_value,
        B_per_robot=direct_value / robot_count,
        predicted_progeny=float(held_out.predicted_total_progeny_per_primary),
        truncated_predicted_progeny=float(
            held_out.truncated_predicted_total_progeny_per_primary
        ),
        measured_progeny=held_out.measured_total_progeny_per_primary,
        relative_error=float(held_out.relative_error),
        truncated_relative_error=float(held_out.truncated_relative_error),
        mean_g=float(np.mean([metric.g_per_traversal for metric in scalar_metrics])),
        mean_treated_traversal_rate=float(np.mean(treated_rates)),
        mean_control_traversal_rate=float(np.mean(control_rates)),
        mean_attributable_rate_loss=float(np.mean(control_rates - treated_rates)),
        stopping_headway_margin=headway - required_headway,
        finite_chain_phi=finite_chain.training.propagation_probability,
        finite_chain_infinite_prediction=finite_chain.infinite_prediction,
        finite_chain_finite_prediction=finite_chain.finite_prediction,
        finite_chain_measured=finite_chain.measured_unique_robots_per_primary,
        finite_chain_infinite_error=finite_chain.infinite_relative_error,
        finite_chain_finite_error=finite_chain.finite_relative_error,
        training_resolved_cascades=finite_chain.training.resolved_cascades,
        training_saturated_cascades=(
            finite_chain.training.fleet_saturated_cascades
        ),
        finite_chain_boundary_identity=(
            finite_chain.training.algebraic_boundary_identity
        ),
        finite_chain_validation_status=finite_chain.validation_status,
    )


def write_tables(
    results: tuple[SpeedExperiment, ...],
    density_results: tuple[DensityExperiment, ...],
) -> None:
    with (OUTPUT / "corridor_speed_sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "speed_m_per_s",
                "direct_B_training",
                "predicted_progeny_training",
                "truncated_predicted_progeny_training",
                "measured_progeny_held_out",
                "held_out_relative_error",
                "truncated_held_out_relative_error",
                "mean_g_seconds_per_traversal",
                "mean_paired_severity_loss_seconds",
                "mean_paired_traversal_loss",
                "mean_treated_traversal_rate_per_s",
                "mean_control_traversal_rate_per_s",
                "mean_attributable_rate_loss_per_s",
                "treated_rate_change_from_previous_speed",
                "stopping_headway_margin_m",
                "boundary_censored_phi_training",
                "boundary_infinite_prediction_training",
                "boundary_finite_prediction_training",
                "unique_robots_per_primary_held_out",
                "boundary_infinite_held_out_relative_error",
                "boundary_finite_held_out_relative_error",
                "training_resolved_cascades",
                "training_fleet_saturated_cascades",
                "finite_chain_boundary_identity",
                "finite_chain_validation_status",
                "robot_count",
                "rollouts",
            ]
        )
        previous_rate: float | None = None
        for result in results:
            rate_change = (
                ""
                if previous_rate is None
                else result.mean_treated_traversal_rate - previous_rate
            )
            writer.writerow(
                [
                    result.speed,
                    result.direct_B,
                    result.predicted_progeny,
                    result.truncated_predicted_progeny,
                    result.measured_progeny,
                    result.relative_error,
                    result.truncated_relative_error,
                    result.mean_g,
                    result.mean_severity_loss,
                    result.mean_traversal_loss,
                    result.mean_treated_traversal_rate,
                    result.mean_control_traversal_rate,
                    result.mean_attributable_rate_loss,
                    rate_change,
                    result.stopping_headway_margin,
                    result.finite_chain_phi,
                    (
                        ""
                        if result.finite_chain_infinite_prediction is None
                        else result.finite_chain_infinite_prediction
                    ),
                    result.finite_chain_finite_prediction,
                    result.finite_chain_measured,
                    (
                        ""
                        if result.finite_chain_infinite_error is None
                        else result.finite_chain_infinite_error
                    ),
                    result.finite_chain_finite_error,
                    result.training_resolved_cascades,
                    result.training_saturated_cascades,
                    result.finite_chain_boundary_identity,
                    result.finite_chain_validation_status,
                    ROBOT_COUNT,
                    ROLLOUTS_PER_SPEED,
                ]
            )
            previous_rate = result.mean_treated_traversal_rate

    with (OUTPUT / "corridor_horizon_sensitivity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["speed_m_per_s", "horizon_seconds", "correlational_B"])
        for result in results:
            for horizon, value in zip(HORIZONS, result.horizon_B, strict=True):
                writer.writerow([result.speed, horizon, value])

    with (OUTPUT / "corridor_density_sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "robot_count",
                "headway_m",
                "speed_m_per_s",
                "direct_B_training",
                "direct_B_per_robot",
                "predicted_progeny_training",
                "truncated_predicted_progeny_training",
                "measured_progeny_held_out",
                "held_out_relative_error",
                "truncated_held_out_relative_error",
                "mean_g_seconds_per_traversal",
                "mean_treated_traversal_rate_per_s",
                "mean_control_traversal_rate_per_s",
                "mean_attributable_rate_loss_per_s",
                "stopping_headway_margin_m",
                "boundary_censored_phi_training",
                "boundary_infinite_prediction_training",
                "boundary_finite_prediction_training",
                "unique_robots_per_primary_held_out",
                "boundary_infinite_held_out_relative_error",
                "boundary_finite_held_out_relative_error",
                "training_resolved_cascades",
                "training_fleet_saturated_cascades",
                "finite_chain_boundary_identity",
                "finite_chain_validation_status",
            ]
        )
        for result in density_results:
            writer.writerow(
                [
                    result.robot_count,
                    result.headway,
                    DENSITY_SWEEP_SPEED,
                    result.direct_B,
                    result.B_per_robot,
                    result.predicted_progeny,
                    result.truncated_predicted_progeny,
                    result.measured_progeny,
                    result.relative_error,
                    result.truncated_relative_error,
                    result.mean_g,
                    result.mean_treated_traversal_rate,
                    result.mean_control_traversal_rate,
                    result.mean_attributable_rate_loss,
                    result.stopping_headway_margin,
                    result.finite_chain_phi,
                    (
                        ""
                        if result.finite_chain_infinite_prediction is None
                        else result.finite_chain_infinite_prediction
                    ),
                    result.finite_chain_finite_prediction,
                    result.finite_chain_measured,
                    (
                        ""
                        if result.finite_chain_infinite_error is None
                        else result.finite_chain_infinite_error
                    ),
                    result.finite_chain_finite_error,
                    result.training_resolved_cascades,
                    result.training_saturated_cascades,
                    result.finite_chain_boundary_identity,
                    result.finite_chain_validation_status,
                ]
            )


def write_demo_events(pair: PairedRolloutResult) -> None:
    with (OUTPUT / "corridor_demo_events.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "event_id",
                "robot_id",
                "cause",
                "parent_event_id",
                "root_primary_event_id",
                "generation",
                "start_time_s",
                "end_time_s",
                "severity_loss_s",
                "minimum_speed_m_per_s",
                "censored",
            ]
        )
        for event in pair.treated.events:
            writer.writerow(
                [
                    event.event_id,
                    event.robot_id,
                    event.cause,
                    event.parent_event_id,
                    event.root_primary_event_id,
                    event.generation,
                    event.start_time,
                    event.end_time,
                    event.severity_loss,
                    event.minimum_speed,
                    event.censored,
                ]
            )


def write_rollout_table(
    speed_results: tuple[SpeedExperiment, ...],
    density_results: tuple[DensityExperiment, ...],
) -> None:
    """Write condition-level raw data rather than only aggregate rows."""

    with (OUTPUT / "corridor_rollout_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sweep",
                "condition_value",
                "speed_m_per_s",
                "robot_count",
                "replication",
                "split",
                "causal_events",
                "primary_events",
                "unique_causal_robots",
                "censored_causal_events",
                "treated_traversal_rate_per_s",
                "control_traversal_rate_per_s",
                "attributable_rate_loss_per_s",
                "attributable_severity_s",
                "traversal_loss",
                "collision_count",
                "crossing_violation_count",
            ]
        )
        families = (
            (
                "speed",
                (
                    (result.speed, result.speed, ROBOT_COUNT, result.rollouts)
                    for result in speed_results
                ),
            ),
            (
                "density",
                (
                    (
                        result.robot_count,
                        DENSITY_SWEEP_SPEED,
                        result.robot_count,
                        result.rollouts,
                    )
                    for result in density_results
                ),
            ),
        )
        for sweep, conditions in families:
            for condition, speed, robot_count, rollouts in conditions:
                for replication, pair in enumerate(rollouts):
                    causal = tuple(
                        event for event in pair.treated.events if event.is_causal
                    )
                    writer.writerow(
                        [
                            sweep,
                            condition,
                            speed,
                            robot_count,
                            replication,
                            (
                                "training"
                                if replication < TRAINING_ROLLOUTS
                                else "validation"
                            ),
                            len(causal),
                            sum(event.is_primary for event in causal),
                            len({event.robot_id for event in causal}),
                            sum(event.censored for event in causal),
                            pair.treated.traversal_rate,
                            pair.control.traversal_rate,
                            (
                                pair.control.traversal_rate
                                - pair.treated.traversal_rate
                            ),
                            pair.attributable_severity,
                            pair.traversal_loss,
                            pair.treated.collision_count,
                            pair.treated.crossing_violation_count,
                        ]
                    )


def write_figure(
    results: tuple[SpeedExperiment, ...], demo: PairedRolloutResult
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.5), constrained_layout=True)
    speed = np.array([result.speed for result in results])
    direct = np.array([result.direct_B for result in results])
    horizon = np.array([result.horizon_B[-1] for result in results])
    predicted = np.array(
        [
            np.nan
            if result.finite_chain_infinite_prediction is None
            else result.finite_chain_infinite_prediction
            for result in results
        ]
    )
    truncated = np.array(
        [result.finite_chain_finite_prediction for result in results]
    )
    measured = np.array([result.finite_chain_measured for result in results])

    axes[0].plot(speed, direct, "o-", linewidth=2.2, label="Direct contact")
    axes[0].plot(
        speed,
        horizon,
        "s--",
        linewidth=2.0,
        label=f"Time window ({HORIZONS[-1]:.0f} s)",
    )
    axes[0].axhline(1.0, color="#111827", linestyle=":", linewidth=1.3)
    axes[0].set_xlabel("Commanded speed (m/s)")
    axes[0].set_ylabel("Estimated offspring coefficient")
    axes[0].set_title("Attribution changes the estimate")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(speed, measured, "o-", linewidth=2.2, label="Held-out measured")
    axes[1].plot(speed, predicted, "s--", linewidth=2.0, label="Infinite-chain prediction")
    axes[1].plot(
        speed,
        truncated,
        "^:",
        linewidth=2.0,
        label=f"Boundary-censored finite depth ($D={ROBOT_COUNT - 1}$)",
    )
    axes[1].set_xlabel("Commanded speed (m/s)")
    axes[1].set_ylabel("Total interventions per primary")
    axes[1].set_title("Unique-robot, out-of-sample validation")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    trajectory = demo.treated.trajectory
    for robot_id in range(demo.treated.robot_count):
        route = trajectory.s[:, robot_id]
        intervening = trajectory.intervening[:, robot_id]
        axes[2].plot(trajectory.time, route, color="#94a3b8", linewidth=0.8)
        axes[2].scatter(
            trajectory.time[intervening],
            route[intervening],
            color="#dc2626",
            s=6,
            alpha=0.8,
        )
    crossing_times = trajectory.time[trajectory.crossing_active]
    if crossing_times.size:
        axes[2].plot(
            crossing_times,
            np.full_like(crossing_times, demo.treated.config.normalized_crossing_s),
            color="#111827",
            linewidth=3.0,
            label="Occupied crossing",
        )
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Route coordinate (m)")
    axes[2].set_title("Space–time cascade (red = intervention)")
    axes[2].grid(alpha=0.2)
    axes[2].legend(frameon=False, loc="upper right")

    figure.savefig(OUTPUT / "corridor_validation.png", dpi=200)
    plt.close(figure)


def write_density_figure(results: tuple[DensityExperiment, ...]) -> None:
    count = np.array([result.robot_count for result in results])
    direct = np.array([result.direct_B for result in results])
    per_robot = np.array([result.B_per_robot for result in results])
    predicted = np.array(
        [
            np.nan
            if result.finite_chain_infinite_prediction is None
            else result.finite_chain_infinite_prediction
            for result in results
        ]
    )
    truncated = np.array(
        [result.finite_chain_finite_prediction for result in results]
    )
    measured = np.array([result.finite_chain_measured for result in results])

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    axes[0].plot(count, direct, "o-", linewidth=2.2, label=r"Measured $B(N,v)$")
    second_axis = axes[0].twinx()
    second_axis.plot(
        count,
        per_robot,
        "s--",
        color="#d97706",
        linewidth=2.0,
        label=r"Empirical $B/N$",
    )
    axes[0].set_xlabel("Robots on 60 m loop")
    axes[0].set_ylabel("Direct offspring coefficient")
    second_axis.set_ylabel(r"Implied homogeneous coefficient $B/N$")
    axes[0].set_title("Testing homogeneous scaling at fixed speed")
    axes[0].grid(alpha=0.25)
    lines = axes[0].get_lines() + second_axis.get_lines()
    axes[0].legend(lines, [line.get_label() for line in lines], frameon=False)

    axes[1].plot(count, measured, "o-", linewidth=2.2, label="Held-out measured")
    axes[1].plot(count, predicted, "s--", linewidth=2.0, label="Infinite chain")
    axes[1].plot(
        count,
        truncated,
        "^:",
        linewidth=2.0,
        label="Boundary-censored finite depth",
    )
    axes[1].set_xlabel("Robots on 60 m loop")
    axes[1].set_ylabel("Total interventions per primary")
    axes[1].set_title(f"Density response at {DENSITY_SWEEP_SPEED:.1f} m/s")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    figure.savefig(OUTPUT / "corridor_density_validation.png", dpi=200)
    plt.close(figure)


def write_speed_throughput_figure(
    results: tuple[SpeedExperiment, ...],
) -> None:
    """Plot the previously missing closed-system circulation-rate response."""

    speed = np.asarray([result.speed for result in results], dtype=float)
    treated = np.asarray(
        [result.mean_treated_traversal_rate for result in results], dtype=float
    )
    control = np.asarray(
        [result.mean_control_traversal_rate for result in results], dtype=float
    )
    loss = np.asarray(
        [result.mean_attributable_rate_loss for result in results], dtype=float
    )
    decreasing = np.flatnonzero(np.diff(treated) < -1e-9)
    status = (
        "turnover observed"
        if decreasing.size
        else "no turnover in the stopping-distance-feasible sweep"
    )

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), constrained_layout=True)
    axes[0].plot(speed, control, "s--", linewidth=2.0, label="No-crossing control")
    axes[0].plot(speed, treated, "o-", linewidth=2.3, label="Crossing treatment")
    axes[0].set(
        xlabel="Commanded speed (m/s)",
        ylabel="Fleet circulation rate (traversals/s)",
        title=f"Closed system: {status}",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(speed, loss, "o-", color="#b42318", linewidth=2.2)
    axes[1].set(
        xlabel="Commanded speed (m/s)",
        ylabel="Attributable circulation-rate loss (traversals/s)",
        title="Disturbance penalty grows with speed",
    )
    axes[1].grid(alpha=0.25)
    figure.savefig(OUTPUT / "corridor_speed_throughput.png", dpi=200)
    plt.close(figure)


def write_truncation_figure(
    speed_results: tuple[SpeedExperiment, ...],
    density_results: tuple[DensityExperiment, ...],
) -> None:
    """Benchmark finite-chain and plain scalar errors without vacuous rows."""

    labeled = [
        (f"v={item.speed:g}", item)
        for item in speed_results
    ] + [
        (f"N={item.robot_count}", item)
        for item in density_results
    ]
    nonvacuous = [
        (label, item)
        for label, item in labeled
        if item.finite_chain_validation_status != "descriptive_not_predictive"
    ]
    vacuous = len(labeled) - len(nonvacuous)
    labels = [label for label, _ in nonvacuous]
    plain = np.asarray([item.relative_error for _, item in nonvacuous]) * 100.0
    finite = np.asarray(
        [item.finite_chain_finite_error for _, item in nonvacuous]
    ) * 100.0
    tolerance = 1.0
    wins = int(np.sum(finite < plain - tolerance))
    losses = int(np.sum(finite > plain + tolerance))
    ties = len(plain) - wins - losses

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    index = np.arange(len(labels))
    axes[0].plot(index, plain, "o--", label=r"Plain $1/(1-\hat B)$")
    axes[0].plot(index, finite, "s-", label=r"Boundary finite $M_N(\hat\phi)$")
    axes[0].set_xticks(index, labels, rotation=55, ha="right")
    axes[0].set(
        xlabel="Non-vacuous held-out condition",
        ylabel="Relative error (%)",
        title="Finite correction is not consistently better",
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)

    maximum = 1.08 * float(max(np.max(plain), np.max(finite), 1.0))
    diagonal = np.linspace(0.0, maximum, 200)
    axes[1].fill_between(
        diagonal,
        np.maximum(0.0, diagonal - tolerance),
        diagonal + tolerance,
        color="#e2e8f0",
        label="practical tie (±1 pp)",
    )
    axes[1].plot(diagonal, diagonal, "k:", linewidth=1.2)
    axes[1].scatter(plain, finite, color="#175cd3", s=42)
    axes[1].set(
        xlim=(0.0, maximum),
        ylim=(0.0, maximum),
        xlabel=r"Plain scalar error (%)",
        ylabel=r"Boundary-finite error (%)",
        title="Aggregate benchmark",
    )
    axes[1].text(
        0.97,
        0.05,
        (
            f"mean: {np.mean(plain):.2f}% vs {np.mean(finite):.2f}%\n"
            f"finite wins/losses/ties: {wins}/{losses}/{ties}\n"
            rf"{vacuous} $\hat\phi=1$ boundary identities excluded"
        ),
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    figure.suptitle("Finite-chain claim audit: no predictive advantage established")
    figure.savefig(OUTPUT / "finite_depth_truncation.png", dpi=200)
    plt.close(figure)


def write_2d_snapshots(demo: PairedRolloutResult) -> None:
    """Render the route-constrained 2D fleet before, during, and after impact."""

    trajectory = demo.treated.trajectory
    loop = demo.treated.config.loop
    selected_times = (3.5, 9.0, 18.0)
    route_s = np.linspace(0.0, loop.perimeter, 600)
    route_x, route_y, _ = loop.to_xy_heading(route_s)
    crossing_x, crossing_y, crossing_heading = loop.to_xy_heading(
        demo.treated.config.normalized_crossing_s
    )
    normal = np.array([-np.sin(crossing_heading), np.cos(crossing_heading)])

    figure, axes = plt.subplots(
        1, 3, figsize=(13.5, 3.8), constrained_layout=True, sharex=True, sharey=True
    )
    for axis, target_time in zip(axes, selected_times, strict=True):
        frame = int(np.argmin(np.abs(trajectory.time - target_time)))
        x, y, heading = loop.to_xy_heading(trajectory.s[frame])
        active = trajectory.intervening[frame]
        colors = np.where(active, "#dc2626", "#1769aa")
        axis.plot(route_x, route_y, color="#cbd5e1", linewidth=7.0, zorder=0)
        axis.plot(route_x, route_y, color="#475569", linewidth=1.0, zorder=1)
        axis.scatter(x, y, c=colors, s=48, edgecolor="white", linewidth=0.6, zorder=3)
        axis.quiver(
            x,
            y,
            np.cos(heading),
            np.sin(heading),
            color=colors,
            angles="xy",
            scale_units="xy",
            scale=2.5,
            width=0.006,
            zorder=4,
        )
        crossing_line = np.array([crossing_x, crossing_y])[:, None] + normal[:, None] * np.array(
            [-0.75, 0.75]
        )
        crossing_is_active = bool(trajectory.crossing_active[frame])
        axis.plot(
            crossing_line[0],
            crossing_line[1],
            color="#111827" if crossing_is_active else "#94a3b8",
            linewidth=5.0 if crossing_is_active else 2.0,
            linestyle="-" if crossing_is_active else "--",
            zorder=2,
        )
        axis.set_title(
            f"t = {trajectory.time[frame]:.1f} s\n"
            + ("crossing occupied" if crossing_is_active else "crossing clear")
        )
        axis.set_aspect("equal")
        axis.grid(alpha=0.15)
        axis.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    figure.suptitle("Route-constrained 2D corridor rollout (red = intervening)")
    figure.savefig(OUTPUT / "corridor_2d_snapshots.png", dpi=200)
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    results = tuple(run_speed_condition(float(speed), rng) for speed in SPEEDS)
    density_results = tuple(
        run_density_condition(robot_count, rng)
        for robot_count in DENSITY_ROBOT_COUNTS
    )

    demo_config = make_config(1.2)
    demo_headway = demo_config.route_length / ROBOT_COUNT
    target_gap = 2.2
    demo_offset = (
        demo_config.normalized_crossing_s - 1.2 * EVENT_START - target_gap
    ) % demo_headway
    demo = run_paired_rollout(
        demo_config,
        SIMULATION_DURATION,
        [CrossingEvent(EVENT_START, 6.0, 999)],
        robot_count=ROBOT_COUNT,
        initial_offset=demo_offset,
        record_stride=5,
    )

    write_tables(results, density_results)
    write_rollout_table(results, density_results)
    write_demo_events(demo)
    write_figure(results, demo)
    write_density_figure(density_results)
    write_speed_throughput_figure(results)
    write_truncation_figure(results, density_results)
    write_2d_snapshots(demo)

    print("Corridor experiment written to", OUTPUT)
    print(
        "speed  B_direct  progeny_inf  progeny_finite  progeny_test  "
        "err_inf  err_finite  treated_rate  control_rate  B_horizon_8s"
    )
    for result in results:
        print(
            f"{result.speed:4.1f}   {result.direct_B:8.4f}   "
            f"{result.predicted_progeny:11.4f}   "
            f"{result.truncated_predicted_progeny:14.4f}   "
            f"{result.measured_progeny:12.4f}   "
            f"{result.relative_error:7.4f}   "
            f"{result.truncated_relative_error:10.4f}   "
            f"{result.mean_treated_traversal_rate:12.4f}   "
            f"{result.mean_control_traversal_rate:12.4f}   "
            f"{result.horizon_B[-1]:13.4f}"
        )
    print("\nrobots  headway  B_direct  B_per_robot  progeny_test")
    for result in density_results:
        print(
            f"{result.robot_count:6d}  {result.headway:7.3f}  "
            f"{result.direct_B:8.4f}  {result.B_per_robot:11.5f}  "
            f"{result.measured_progeny:12.4f}"
        )


if __name__ == "__main__":
    main()
