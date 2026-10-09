#!/usr/bin/env python3
"""Milestone 7: grouped inference, calibration, depletion, and loaded networks.

The script deliberately separates software smoke checks from paper-facing
evidence. Every profile writes its complete configuration and fail-closed claim
ledger. A main-paper claim is never inferred from a missing or indeterminate
row.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from amr_capacity import (  # noqa: E402
    NetworkConfig,
    NetworkDisturbance,
    OpenCorridorConfig,
    audit_branching_timestamps,
    bootstrap_joint_censored_multitype_branching,
    cascade_budget_direct,
    classify_branching_regime,
    deterministic_arrival_times,
    deterministic_network_jobs,
    estimate_censored_multitype_branching,
    estimate_joint_censored_multitype_branching,
    extend_network_map,
    intervention_events_to_censored_rollout,
    poisson_crossing_requests_multi,
    poisson_extinction_probability,
    retype_events_by_minimum_speed,
    r_saddle_node_from_burden,
    run_network_paired_rollout,
    run_open_paired_rollout,
    scale_branching_matrix,
    simulate_censored_branching_experiment,
    simulate_depleting_branching_experiment,
    simulate_extinct_poisson_trees,
    standard_map_catalogue,
    standard_robot_catalogue,
)


PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {
        "outer_datasets": 4,
        "outer_rollouts": 60,
        "outer_bootstrap": 20,
        "bridge_rates": (0.25, 0.50),
        "bridge_duration": 90.0,
        "bridge_replications": 2,
        "network_fleets": (20,),
        "network_loads": (0.45,),
        "network_replications": 1,
        "network_min_duration": 75.0,
        "network_bootstrap": 20,
        "depletion_rollouts": 100,
        "extinction_rollouts": 3_000,
    },
    "quick": {
        "outer_datasets": 12,
        "outer_rollouts": 180,
        "outer_bootstrap": 80,
        "bridge_rates": (0.20, 0.35, 0.50, 0.70),
        "bridge_duration": 180.0,
        "bridge_replications": 6,
        "network_fleets": (50, 100, 200),
        "network_loads": (0.40, 0.70, 1.00),
        "network_replications": 2,
        "network_min_duration": 180.0,
        "network_bootstrap": 80,
        "depletion_rollouts": 500,
        "extinction_rollouts": 20_000,
    },
    "paper": {
        "outer_datasets": 200,
        "outer_rollouts": 1_000,
        "outer_bootstrap": 1_000,
        "bridge_rates": (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80),
        "bridge_duration": 1_800.0,
        "bridge_replications": 30,
        "network_fleets": (100, 200, 300),
        "network_loads": (0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.00),
        "network_replications": 20,
        "network_min_duration": 1_200.0,
        "network_bootstrap": 2_000,
        "depletion_rollouts": 5_000,
        "extinction_rollouts": 200_000,
    },
}


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    records = list(rows)
    if not records:
        raise ValueError(f"cannot write empty table {path}")
    columns: list[str] = []
    for record in records:
        for key in record:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def _non_normality(matrix: np.ndarray) -> float:
    norm = float(np.linalg.norm(matrix, ord="fro"))
    if norm <= 0.0:
        return 0.0
    commutator = matrix.T @ matrix - matrix @ matrix.T
    return float(np.linalg.norm(commutator, ord="fro") / norm**2)


def _optional(value: float | None) -> float | str:
    return "" if value is None or not np.isfinite(value) else float(value)


def run_outer_calibration(
    output: Path, profile: str, settings: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Repeated datasets, not bootstrap mass, estimate decision error rates."""

    base = np.array([[0.25, 0.80], [0.10, 0.30]], dtype=float)
    gamma = np.array([[1.0, 0.70], [0.90, 1.20]], dtype=float)
    targets = (0.85, 1.00, 1.15)
    rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(targets):
        truth = scale_branching_matrix(base, target)
        for dataset in range(int(settings["outer_datasets"])):
            seed = 2026097000 + 10_000 * target_index + dataset
            rollouts = simulate_censored_branching_experiment(
                truth,
                gamma,
                rollout_count=int(settings["outer_rollouts"]),
                roots_per_rollout=3,
                horizon=2.0,
                population_cap=500,
                root_type_probabilities=(0.55, 0.45),
                seed=seed,
            )
            interval = bootstrap_joint_censored_multitype_branching(
                rollouts,
                2,
                observation_model="continuous",
                recovery_rate_bounds=(0.20, 4.0),
                replications=int(settings["outer_bootstrap"]),
                seed=seed + 500_000,
            )
            known = estimate_censored_multitype_branching(rollouts, gamma)
            verdict = classify_branching_regime(
                interval.spectral_radius_lower, interval.spectral_radius_upper
            )
            rows.append(
                {
                    "profile": profile,
                    "true_rho": target,
                    "dataset": dataset,
                    "joint_rho": interval.point.spectral_radius,
                    "rho_ci_lower": interval.spectral_radius_lower,
                    "rho_ci_upper": interval.spectral_radius_upper,
                    "regime_verdict": verdict,
                    "interval_contains_truth": (
                        interval.spectral_radius_lower
                        <= target
                        <= interval.spectral_radius_upper
                    ),
                    "known_gamma_rho": known.spectral_radius,
                    "joint_abs_error": abs(interval.point.spectral_radius - target),
                    "known_gamma_abs_error": abs(known.spectral_radius - target),
                    "max_gamma_abs_error": float(
                        np.max(np.abs(interval.point.recovery_rates - gamma))
                    ),
                    "gamma_bound_entries": int(
                        np.count_nonzero(interval.point.recovery_at_bound)
                    ),
                    "bootstrap_successes": interval.successful_replications,
                }
            )
    summary: list[dict[str, Any]] = []
    for target in targets:
        selected = [row for row in rows if row["true_rho"] == target]
        false_supercritical = np.mean(
            [row["regime_verdict"] == "certified_supercritical" for row in selected]
        )
        false_subcritical = np.mean(
            [row["regime_verdict"] == "certified_subcritical" for row in selected]
        )
        summary.append(
            {
                "profile": profile,
                "true_rho": target,
                "outer_datasets": len(selected),
                "coverage_frequency": float(
                    np.mean([row["interval_contains_truth"] for row in selected])
                ),
                "mean_joint_rho": float(np.mean([row["joint_rho"] for row in selected])),
                "joint_mae": float(np.mean([row["joint_abs_error"] for row in selected])),
                "known_gamma_mae": float(
                    np.mean([row["known_gamma_abs_error"] for row in selected])
                ),
                "certified_subcritical_frequency": float(
                    np.mean(
                        [row["regime_verdict"] == "certified_subcritical" for row in selected]
                    )
                ),
                "indeterminate_frequency": float(
                    np.mean([row["regime_verdict"] == "indeterminate" for row in selected])
                ),
                "certified_supercritical_frequency": float(false_supercritical),
                "wrong_regime_frequency": float(
                    false_supercritical
                    if target < 1.0
                    else false_subcritical
                    if target > 1.0
                    else false_supercritical + false_subcritical
                ),
            }
        )
    _write_csv(output / "milestone7_outer_calibration_runs.csv", rows)
    _write_csv(output / "milestone7_outer_calibration_summary.csv", summary)
    return rows, summary


def _open_config() -> OpenCorridorConfig:
    return OpenCorridorConfig(
        corridor_length=30.0,
        desired_speed=1.2,
        acceleration=1.0,
        braking=1.5,
        reaction_time=0.2,
        dt=0.05,
        robot_length=0.8,
        safety_margin=0.2,
        sensor_range=8.0,
        crossing_x=tuple(float(item) for item in np.linspace(2.0, 28.0, 16)),
        crossing_half_width=0.4,
    )


def _heldout_progeny_prediction(
    matrix: np.ndarray,
    test_rollouts,
) -> tuple[float | None, float | None, int, int]:
    roots_by_type = np.zeros(matrix.shape[0], dtype=int)
    event_count = 0
    for rollout in test_rollouts:
        event_count += len(rollout.events)
        for event in rollout.events:
            if event.parent_event_id is None:
                roots_by_type[event.event_type] += 1
    roots = int(np.sum(roots_by_type))
    measured = None if roots == 0 else event_count / roots
    if roots == 0 or max(abs(np.linalg.eigvals(matrix))) >= 1.0:
        return None, measured, roots, event_count
    progeny = np.linalg.solve(np.eye(matrix.shape[0]) - matrix, np.ones(matrix.shape[0]))
    prediction = float(np.dot(roots_by_type / roots, progeny))
    return prediction, measured, roots, event_count


def run_simulator_bridge(
    output: Path, profile: str, settings: dict[str, Any]
) -> list[dict[str, Any]]:
    """Run grouped joint inference directly on fixed-step simulator logs."""

    config = _open_config()
    duration = float(settings["bridge_duration"])
    rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    for rate in settings["bridge_rates"]:
        rollouts = []
        active_interval_rollouts = []
        invariant_events = 0
        for replication in range(int(settings["bridge_replications"])):
            arrivals = deterministic_arrival_times(float(rate), duration)
            requests = poisson_crossing_requests_multi(
                0.015,
                duration,
                (2.0, 4.0),
                16,
                seed=2026098000 + replication,
                start_time=3.0,
            )
            pair = run_open_paired_rollout(
                config, duration, arrivals, requests, record_stride=40
            )
            if (
                pair.treated.collision_count
                or pair.treated.crossing_violation_count
                or pair.treated.mass_balance_residual
            ):
                raise AssertionError("simulator bridge violated a hard invariant")
            typed = retype_events_by_minimum_speed(pair.treated.events)
            rollout = intervention_events_to_censored_rollout(
                typed, observation_end=duration
            )
            rollouts.append(rollout)
            active_interval_rollouts.append(
                intervention_events_to_censored_rollout(
                    typed,
                    observation_end=duration,
                    exposure_policy="active_interval",
                )
            )
            invariant_events += len(typed)
        audit = audit_branching_timestamps(rollouts, config.dt)
        grouped = estimate_joint_censored_multitype_branching(
            rollouts,
            2,
            observation_model="grouped",
            time_step=config.dt,
            recovery_rate_bounds=(0.05, 20.0),
        )
        continuous = estimate_joint_censored_multitype_branching(
            rollouts,
            2,
            observation_model="continuous",
            recovery_rate_bounds=(0.05, 20.0),
        )
        active_interval = estimate_joint_censored_multitype_branching(
            active_interval_rollouts,
            2,
            observation_model="grouped",
            time_step=config.dt,
            recovery_rate_bounds=(0.05, 20.0),
        )
        split = max(1, len(rollouts) // 2)
        training = rollouts[:split]
        testing = rollouts[split:] or rollouts[-1:]
        train_fit = estimate_joint_censored_multitype_branching(
            training,
            2,
            observation_model="grouped",
            time_step=config.dt,
            recovery_rate_bounds=(0.05, 20.0),
        )
        prediction, measured, roots, heldout_events = _heldout_progeny_prediction(
            train_fit.matrix, testing
        )
        g_star: float | None = None
        if grouped.spectral_radius < 1.0:
            g_star = cascade_budget_direct(grouped.matrix)[0]
        rows.append(
            {
                "profile": profile,
                "arrival_rate_per_s": rate,
                "rollouts": len(rollouts),
                "logged_intervention_events": invariant_events,
                "causal_events": audit.event_count,
                "direct_edges": audit.edge_count,
                "zero_lag_edges": audit.zero_lag_edge_count,
                "zero_lag_fraction": audit.zero_lag_fraction,
                "grouped_rho": grouped.spectral_radius,
                "continuous_rho_ablation": continuous.spectral_radius,
                "active_interval_rho_ablation": active_interval.spectral_radius,
                "rho_model_difference": grouped.spectral_radius
                - continuous.spectral_radius,
                "gamma_bound_entries": int(np.count_nonzero(grouped.recovery_at_bound)),
                "non_normality": _non_normality(grouped.matrix),
                "G_star": _optional(g_star),
                "heldout_roots": roots,
                "heldout_events": heldout_events,
                "heldout_mean_progeny": _optional(measured),
                "training_resolvent_prediction": _optional(prediction),
                "heldout_relative_error": _optional(
                    None
                    if prediction is None or measured in (None, 0.0)
                    else abs(prediction - measured) / measured
                ),
            }
        )
        for parent_type in range(2):
            for child_type in range(2):
                matrix_rows.append(
                    {
                        "profile": profile,
                        "arrival_rate_per_s": rate,
                        "parent_type": parent_type,
                        "child_type": child_type,
                        "grouped_B": grouped.matrix[parent_type, child_type],
                        "grouped_gamma_per_s": grouped.recovery_rates[
                            parent_type, child_type
                        ],
                        "continuous_B": continuous.matrix[parent_type, child_type],
                        "continuous_gamma_per_s": continuous.recovery_rates[
                            parent_type, child_type
                        ],
                        "active_interval_B": active_interval.matrix[
                            parent_type, child_type
                        ],
                        "active_interval_gamma_per_s": (
                            active_interval.recovery_rates[parent_type, child_type]
                        ),
                        "direct_children": grouped.direct_child_counts[
                            parent_type, child_type
                        ],
                        "kernel_exposure": grouped.exposure_mass[
                            parent_type, child_type
                        ],
                        "identifiable": bool(
                            grouped.identifiable_pairs[parent_type, child_type]
                        ),
                    }
                )
    _write_csv(output / "milestone7_simulator_bridge.csv", rows)
    _write_csv(output / "milestone7_simulator_bridge_matrix.csv", matrix_rows)
    return rows


def run_finite_population_controls(
    output: Path, profile: str, settings: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base = np.array([[0.25, 0.80], [0.10, 0.30]], dtype=float)
    truth = scale_branching_matrix(base, 1.30)
    gamma = np.array([[1.0, 0.70], [0.90, 1.20]], dtype=float)
    depletion_rows: list[dict[str, Any]] = []
    for population_per_type in (10, 25, 50, 100):
        rollouts = simulate_depleting_branching_experiment(
            truth,
            gamma,
            rollout_count=int(settings["depletion_rollouts"]),
            roots_per_rollout=2,
            horizon=8.0,
            population_by_type=(population_per_type, population_per_type),
            root_type_probabilities=(0.55, 0.45),
            seed=2026099000 + population_per_type,
        )
        fitted = estimate_censored_multitype_branching(rollouts, gamma)
        depletion_rows.append(
            {
                "profile": profile,
                "population": 2 * population_per_type,
                "rollouts": len(rollouts),
                "population_capped_rollouts": sum(
                    item.population_cap_reached for item in rollouts
                ),
                "cap_fraction": float(
                    np.mean([item.population_cap_reached for item in rollouts])
                ),
                "mean_observed_events": float(
                    np.mean([len(item.events) for item in rollouts])
                ),
                "estimated_rho": fitted.spectral_radius,
                "true_uncapped_rho": 1.30,
                "depletion_bias": fitted.spectral_radius - 1.30,
            }
        )
    _write_csv(output / "milestone7_population_depletion.csv", depletion_rows)

    mean = 1.30
    q = poisson_extinction_probability(mean)
    children, parents, retained = simulate_extinct_poisson_trees(
        mean,
        rollout_count=int(settings["extinction_rollouts"]),
        population_guard=5_000,
        seed=2026100000,
    )
    duality = {
        "profile": profile,
        "true_mean": mean,
        "extinction_probability": q,
        "dual_mean_mq": mean * q,
        "empirical_extinct_tree_mean": children / parents,
        "absolute_error": abs(children / parents - mean * q),
        "retained_extinct_trees": retained,
        "simulated_trees": int(settings["extinction_rollouts"]),
    }
    _write_csv(output / "milestone7_extinction_duality.csv", [duality])
    return depletion_rows, duality


def _scheduled_network_disturbances(
    map_spec,
    duration: float,
    seed: int,
) -> tuple[NetworkDisturbance, ...]:
    rng = np.random.default_rng(seed)
    zone = map_spec.conflict_zones[0]
    rows: list[NetworkDisturbance] = []
    time = 0.15 * duration
    source = 0
    while time < 0.85 * duration:
        route = map_spec.routes[source % len(map_spec.routes)]
        location = float(
            route.s[int(np.argmin(np.hypot(route.x - zone.x, route.y - zone.y)))]
        )
        rows.append(
            NetworkDisturbance(
                source_id=source,
                route_name=route.name,
                s=location,
                request_time=float(time + rng.uniform(-1.0, 1.0)),
                duration=float(rng.uniform(6.0, 10.0)),
            )
        )
        source += 1
        time += 0.08 * duration
    return tuple(rows)


def run_loaded_networks(
    output: Path, profile: str, settings: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base_maps = {
        item.name: item
        for item in standard_map_catalogue()
        if item.name in ("two_to_one_merge", "four_way_intersection")
    }
    limits = dict(standard_robot_catalogue())["standard"]
    raw: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    for map_name, base_map in base_maps.items():
        for fleet_size in settings["network_fleets"]:
            target_length = max(60.0, float(fleet_size))
            map_spec = extend_network_map(base_map, target_length)
            duration = max(
                float(settings["network_min_duration"]),
                1.6 * target_length / map_spec.speed_limit,
            )
            duration = np.ceil(duration / 0.05) * 0.05
            config = NetworkConfig(
                map_spec=map_spec,
                limits=limits,
                fleet_size=int(fleet_size),
                dt=0.05,
                sensor_range=8.0,
            )
            for total_load in settings["network_loads"]:
                rollouts = []
                active_interval_rollouts = []
                treated_rates: list[float] = []
                control_rates: list[float] = []
                primary_x: list[float] = []
                total_x: list[float] = []
                postwarm_primary_x: list[float] = []
                postwarm_total_x: list[float] = []
                mean_wip_values: list[float] = []
                realized_density_values: list[float] = []
                mean_headway_values: list[float] = []
                completion_losses: list[int] = []
                nominal_times = []
                cruise = min(limits.max_speed, limits.max_wheel_speed, map_spec.speed_limit)
                from amr_capacity import plan_kinematic_speed_profile

                for route in map_spec.routes:
                    nominal_times.append(
                        plan_kinematic_speed_profile(
                            route,
                            limits,
                            map_speed_limit=map_spec.speed_limit,
                            start_speed=cruise,
                            end_speed=cruise,
                        ).nominal_time
                    )
                t0 = float(np.mean(nominal_times))
                for replication in range(int(settings["network_replications"])):
                    jobs = deterministic_network_jobs(
                        {
                            route.name: float(total_load) / len(map_spec.routes)
                            for route in map_spec.routes
                        },
                        duration,
                    )
                    disturbances = _scheduled_network_disturbances(
                        map_spec,
                        duration,
                        seed=(
                            2026101000
                            + 100_000 * list(base_maps).index(map_name)
                            + 1_000 * int(fleet_size)
                            + 100 * int(round(float(total_load) * 100))
                            + replication
                        ),
                    )
                    pair = run_network_paired_rollout(
                        config,
                        duration,
                        jobs,
                        disturbances,
                        record_stride=40,
                    )
                    for result in (pair.treated, pair.control):
                        if (
                            result.collision_count
                            or result.conflict_violation_count
                            or result.disturbance_violation_count
                            or result.mass_balance_residual
                            or result.fleet_balance_residual
                        ):
                            raise AssertionError("loaded network violated a hard invariant")
                    typed = retype_events_by_minimum_speed(pair.treated.events)
                    rollout = intervention_events_to_censored_rollout(
                        typed, observation_end=duration
                    )
                    rollouts.append(rollout)
                    active_interval_rollouts.append(
                        intervention_events_to_censored_rollout(
                            typed,
                            observation_end=duration,
                            exposure_policy="active_interval",
                        )
                    )
                    warmup = 0.35 * duration
                    treated_completions = sum(
                        item.completion_time >= warmup
                        for item in pair.treated.completed_traversals
                    )
                    control_completions = sum(
                        item.completion_time >= warmup
                        for item in pair.control.completed_traversals
                    )
                    window = duration - warmup
                    treated_rate = treated_completions / window
                    control_rate = control_completions / window
                    causal = [
                        event
                        for event in pair.treated.events
                        if event.is_causal and event.end_time > warmup
                    ]
                    def window_loss(event) -> float:
                        overlap = max(
                            0.0,
                            min(event.end_time, duration)
                            - max(event.start_time, warmup),
                        )
                        if event.duration <= 0.0:
                            return 0.0
                        return event.severity_loss * overlap / event.duration

                    postwarm_primary_loss = sum(
                        window_loss(event) for event in causal if event.is_primary
                    )
                    postwarm_total_loss = sum(window_loss(event) for event in causal)
                    denominator = max(treated_completions, 1)
                    postwarm_x_primary = (postwarm_primary_loss / denominator) / t0
                    postwarm_x_total = (postwarm_total_loss / denominator) / t0
                    all_causal = [event for event in pair.treated.events if event.is_causal]
                    all_completions = max(pair.treated.total_completions, 1)
                    x_primary = (
                        sum(
                            event.severity_loss
                            for event in all_causal
                            if event.is_primary
                        )
                        / all_completions
                        / t0
                    )
                    x_total = (
                        sum(event.severity_loss for event in all_causal)
                        / all_completions
                        / t0
                    )
                    retained = pair.treated.trajectory.time >= warmup
                    mean_wip = float(
                        np.mean(pair.treated.trajectory.work_in_process[retained])
                    )
                    lane_metres = float(sum(route.length for route in map_spec.routes))
                    realized_density = mean_wip / lane_metres
                    mean_headway = (
                        np.inf if mean_wip <= 0.0 else lane_metres / mean_wip
                    )
                    treated_rates.append(treated_rate)
                    control_rates.append(control_rate)
                    primary_x.append(x_primary)
                    total_x.append(x_total)
                    postwarm_primary_x.append(postwarm_x_primary)
                    postwarm_total_x.append(postwarm_x_total)
                    mean_wip_values.append(mean_wip)
                    realized_density_values.append(realized_density)
                    mean_headway_values.append(mean_headway)
                    completion_losses.append(pair.completion_loss)
                    raw.append(
                        {
                            "profile": profile,
                            "map": map_name,
                            "fleet_size": fleet_size,
                            "target_route_length_m": target_length,
                            "total_arrival_rate_per_s": total_load,
                            "replication": replication,
                            "duration_s": duration,
                            "treated_throughput_per_s": treated_rate,
                            "control_throughput_per_s": control_rate,
                            "completion_loss": pair.completion_loss,
                            "causal_events": len(rollout.events),
                            "primary_events": sum(
                                event.parent_event_id is None for event in rollout.events
                            ),
                            "x_primary_fleet": x_primary,
                            "x_total_fleet": x_total,
                            "x_primary_fleet_post_warmup": postwarm_x_primary,
                            "x_total_fleet_post_warmup": postwarm_x_total,
                            "mean_work_in_process": mean_wip,
                            "realized_density_robots_per_m": realized_density,
                            "mean_lane_headway_m": mean_headway,
                            "nominal_T0_s": t0,
                            "max_abs_yaw_rate": pair.treated.max_abs_yaw_rate,
                            "max_abs_yaw_acceleration": pair.treated.max_abs_yaw_acceleration,
                            "max_lateral_acceleration": pair.treated.max_lateral_acceleration,
                            "max_abs_wheel_speed": pair.treated.max_abs_wheel_speed,
                            "collision_count": pair.treated.collision_count,
                            "conflict_violation_count": pair.treated.conflict_violation_count,
                            "disturbance_violation_count": pair.treated.disturbance_violation_count,
                        }
                    )
                audit = audit_branching_timestamps(rollouts, config.dt)
                fitted = estimate_joint_censored_multitype_branching(
                    rollouts,
                    2,
                    observation_model="grouped",
                    time_step=config.dt,
                    recovery_rate_bounds=(0.05, 20.0),
                )
                active_interval_fit = estimate_joint_censored_multitype_branching(
                    active_interval_rollouts,
                    2,
                    observation_model="grouped",
                    time_step=config.dt,
                    recovery_rate_bounds=(0.05, 20.0),
                )
                interval = None
                if len(rollouts) >= 10 and np.any(fitted.identifiable_pairs):
                    try:
                        interval = bootstrap_joint_censored_multitype_branching(
                            rollouts,
                            2,
                            observation_model="grouped",
                            time_step=config.dt,
                            recovery_rate_bounds=(0.05, 20.0),
                            replications=int(settings["network_bootstrap"]),
                            seed=2026102000 + int(fleet_size),
                        )
                    except RuntimeError:
                        interval = None
                lower = None if interval is None else interval.spectral_radius_lower
                upper = None if interval is None else interval.spectral_radius_upper
                verdict = (
                    "insufficient_clusters"
                    if len(rollouts) < 10
                    else "indeterminate"
                    if lower is None or upper is None
                    else classify_branching_regime(lower, upper)
                )
                g_star = None
                if fitted.spectral_radius < 1.0:
                    g_star = cascade_budget_direct(fitted.matrix)[0]
                mean_primary_x = float(np.mean(primary_x))
                inference_status = (
                    "estimable"
                    if audit.edge_count >= 10 and audit.event_count >= 20
                    else "insufficient_events"
                )
                summary.append(
                    {
                        "profile": profile,
                        "map": map_name,
                        "fleet_size": fleet_size,
                        "target_route_length_m": target_length,
                        "total_arrival_rate_per_s": total_load,
                        "replications": len(rollouts),
                        "inference_status": inference_status,
                        "causal_events": audit.event_count,
                        "direct_edges": audit.edge_count,
                        "mean_treated_throughput_per_s": float(np.mean(treated_rates)),
                        "mean_control_throughput_per_s": float(np.mean(control_rates)),
                        "mean_completion_loss": float(np.mean(completion_losses)),
                        "grouped_rho": fitted.spectral_radius,
                        "active_interval_rho_ablation": (
                            active_interval_fit.spectral_radius
                        ),
                        "rho_ci_lower": _optional(lower),
                        "rho_ci_upper": _optional(upper),
                        "regime_verdict": verdict,
                        "G_star": _optional(g_star),
                        "non_normality": _non_normality(fitted.matrix),
                        "zero_lag_fraction": audit.zero_lag_fraction,
                        "gamma_bound_entries": int(np.count_nonzero(fitted.recovery_at_bound)),
                        "mean_x_primary_fleet": mean_primary_x,
                        "mean_x_total_fleet": float(np.mean(total_x)),
                        "mean_x_primary_fleet_post_warmup": float(
                            np.mean(postwarm_primary_x)
                        ),
                        "mean_x_total_fleet_post_warmup": float(
                            np.mean(postwarm_total_x)
                        ),
                        "mean_work_in_process": float(np.mean(mean_wip_values)),
                        "realized_density_robots_per_m": float(
                            np.mean(realized_density_values)
                        ),
                        "mean_lane_headway_m": float(np.mean(mean_headway_values)),
                        "fold_R_prediction": float(
                            r_saddle_node_from_burden(mean_primary_x)
                        ),
                        "all_safety_invariants_pass": True,
                    }
                )
                for parent_type in range(2):
                    for child_type in range(2):
                        matrix_rows.append(
                            {
                                "profile": profile,
                                "map": map_name,
                                "fleet_size": fleet_size,
                                "total_arrival_rate_per_s": total_load,
                                "parent_type": parent_type,
                                "child_type": child_type,
                                "grouped_B": fitted.matrix[parent_type, child_type],
                                "grouped_gamma_per_s": fitted.recovery_rates[
                                    parent_type, child_type
                                ],
                                "active_interval_B": active_interval_fit.matrix[
                                    parent_type, child_type
                                ],
                                "active_interval_gamma_per_s": (
                                    active_interval_fit.recovery_rates[
                                        parent_type, child_type
                                    ]
                                ),
                                "direct_children": fitted.direct_child_counts[
                                    parent_type, child_type
                                ],
                                "kernel_exposure": fitted.exposure_mass[
                                    parent_type, child_type
                                ],
                                "identifiable": bool(
                                    fitted.identifiable_pairs[parent_type, child_type]
                                ),
                            }
                        )
                print(
                    f"network {map_name} N={fleet_size} load={total_load:.2f} "
                    f"rho={fitted.spectral_radius:.3f}",
                    flush=True,
                )
    _write_csv(output / "milestone7_network_runs.csv", raw)
    _write_csv(output / "milestone7_network_summary.csv", summary)
    _write_csv(output / "milestone7_network_matrix.csv", matrix_rows)
    return raw, summary


def make_plots(
    output: Path,
    calibration: list[dict[str, Any]],
    bridge: list[dict[str, Any]],
    networks: list[dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), constrained_layout=True)
    targets = [row["true_rho"] for row in calibration]
    axes[0, 0].scatter(
        targets, [row["joint_rho"] for row in calibration], s=22, alpha=0.7
    )
    axes[0, 0].plot([0.8, 1.2], [0.8, 1.2], "k--", linewidth=1)
    axes[0, 0].axvline(1.0, color="#b42318", linewidth=1)
    axes[0, 0].axhline(1.0, color="#b42318", linewidth=1)
    axes[0, 0].set(
        xlabel="true rho", ylabel="joint estimate", title="Outer calibration"
    )

    rates = [row["arrival_rate_per_s"] for row in bridge]
    axes[0, 1].plot(
        rates, [row["grouped_rho"] for row in bridge], "o-", label="grouped"
    )
    axes[0, 1].plot(
        rates,
        [row["continuous_rho_ablation"] for row in bridge],
        "s--",
        label="exact-time ablation",
    )
    axes[0, 1].axhline(1.0, color="k", linewidth=1)
    axes[0, 1].set(
        xlabel="arrival rate [s^-1]",
        ylabel="estimated rho",
        title="Simulator bridge",
    )
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(
        rates,
        [float(row["G_star"]) for row in bridge],
        "o-",
        color="#7a5af8",
        label="G*",
    )
    axes[1, 0].set(
        xlabel="arrival rate [s^-1]",
        ylabel="exact cascade budget G*",
        title="Non-normal cumulative amplification",
    )
    heldout_axis = axes[1, 0].twinx()
    heldout_axis.plot(
        rates,
        [100.0 * float(row["heldout_relative_error"]) for row in bridge],
        "s--",
        color="#d55e00",
        label="held-out error",
    )
    heldout_axis.set_ylabel("held-out relative error [%]")

    for (map_name, fleet), selected in _group_rows(networks, ("map", "fleet_size")):
        ordered = sorted(selected, key=lambda row: row["total_arrival_rate_per_s"])
        axes[1, 1].plot(
            [row["total_arrival_rate_per_s"] for row in ordered],
            [row["mean_treated_throughput_per_s"] for row in ordered],
            "o-",
            label=f"{map_name}, N={fleet}",
        )
    axes[1, 1].set(
        xlabel="offered load [s^-1]",
        ylabel="throughput [s^-1]",
        title="Loaded networks",
    )
    axes[1, 1].legend(frameon=False, fontsize=7)
    figure.savefig(output / "milestone7_summary.png", dpi=220)
    plt.close(figure)

    estimable = [row for row in networks if row["inference_status"] == "estimable"]
    if estimable:
        figure, axis = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
        colors = {
            "two_to_one_merge": "#175cd3",
            "four_way_intersection": "#b42318",
        }
        for map_name in colors:
            selected = [row for row in estimable if row["map"] == map_name]
            if selected:
                axis.scatter(
                    [row["realized_density_robots_per_m"] for row in selected],
                    [row["grouped_rho"] for row in selected],
                    s=[35.0 + 45.0 * row["total_arrival_rate_per_s"] for row in selected],
                    color=colors[map_name],
                    alpha=0.8,
                    label=map_name,
                )
        axis.axhline(1.0, color="k", linewidth=1)
        axis.set(
            xlabel="realized mean density [robots/m of lane]",
            ylabel="grouped estimate of rho",
            title="Fleet count is not density",
        )
        axis.legend(frameon=False)
        figure.savefig(output / "milestone7_density_phase.png", dpi=220)
        plt.close(figure)


def _group_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]):
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[item] for item in keys)
        groups.setdefault(key, []).append(row)
    return groups.items()


def build_claim_ledger(
    profile: str,
    calibration: list[dict[str, Any]],
    bridge: list[dict[str, Any]],
    depletion: list[dict[str, Any]],
    duality: dict[str, Any],
    network_raw: list[dict[str, Any]],
    network_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    at_critical = [row for row in calibration if row["true_rho"] == 1.0]
    boundary_wrong_frequency = float(
        np.mean([row["regime_verdict"] != "indeterminate" for row in at_critical])
    )
    estimable_network = [
        row for row in network_summary if row["inference_status"] == "estimable"
    ]
    ledger = {
        "profile": profile,
        "software_gates": {
            "outer_calibration_executed": bool(calibration),
            "simulator_log_bridge_executed": bool(bridge),
            "grouped_model_used_when_zero_lags_exist": all(
                row["zero_lag_edges"] == 0 or row["zero_lag_fraction"] > 0
                for row in bridge
            ),
            "genuine_population_depletion_observed": any(
                row["population_capped_rollouts"] > 0 for row in depletion
            ),
            "extinction_duality_absolute_error_below_0_05": duality["absolute_error"] < 0.05,
            "network_safety_invariants_pass": all(
                row["collision_count"] == 0
                and row["conflict_violation_count"] == 0
                and row["disturbance_violation_count"] == 0
                for row in network_raw
            ),
            "merge_and_intersection_both_executed": {
                row["map"] for row in network_raw
            }
            == {"two_to_one_merge", "four_way_intersection"},
        },
        "evidence_gates": {
            "boundary_wrong_regime_frequency_at_most_0_05": (
                boundary_wrong_frequency <= 0.05
            ),
            "boundary_wrong_regime_frequency": boundary_wrong_frequency,
            "outer_coverage_target_0_90": float(
                np.mean([row["interval_contains_truth"] for row in calibration])
            )
            >= 0.90,
            "N200_executed": any(row["fleet_size"] == 200 for row in network_raw),
            "physical_point_rho_crossing_observed": (
                bool(estimable_network)
                and min(row["grouped_rho"] for row in estimable_network) < 1.0
                and max(row["grouped_rho"] for row in estimable_network) > 1.0
            ),
            "physical_confidence_certified_crossing_observed": any(
                {row["regime_verdict"] for row in group}
                >= {"certified_subcritical", "certified_supercritical"}
                for _, group in _group_rows(
                    network_summary, ("map", "fleet_size")
                )
            ),
            "throughput_turnover_observed": any(
                _has_turnover(group)
                for _, group in _group_rows(network_summary, ("map", "fleet_size"))
            ),
        },
        "scope": {
            "smoke_and_quick_profiles_are_not_paper_claims": profile != "paper",
            "burden_names": ["x_primary_fleet", "x_total_fleet"],
            "route_obstacle_burden_x_route_is_not_reused_here": True,
        },
    }
    return ledger


def _has_turnover(rows: list[dict[str, Any]]) -> bool:
    ordered = sorted(rows, key=lambda row: row["total_arrival_rate_per_s"])
    throughput = np.asarray(
        [row["mean_treated_throughput_per_s"] for row in ordered], dtype=float
    )
    return bool(throughput.size >= 3 and np.any(np.diff(throughput) < -1e-6))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    arguments = parser.parse_args()
    settings = PROFILES[arguments.profile]
    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)

    calibration, calibration_summary = run_outer_calibration(
        output, arguments.profile, settings
    )
    bridge = run_simulator_bridge(output, arguments.profile, settings)
    depletion, duality = run_finite_population_controls(
        output, arguments.profile, settings
    )
    network_raw, network_summary = run_loaded_networks(
        output, arguments.profile, settings
    )
    make_plots(output, calibration, bridge, network_summary)
    ledger = build_claim_ledger(
        arguments.profile,
        calibration,
        bridge,
        depletion,
        duality,
        network_raw,
        network_summary,
    )
    with (output / "milestone7_claim_ledger.json").open("w", encoding="utf-8") as stream:
        json.dump(ledger, stream, indent=2)
    manifest = {
        "profile": arguments.profile,
        "settings": settings,
        "outputs": sorted(path.name for path in output.glob("milestone7_*")),
        "calibration_summary": calibration_summary,
        "claim_ledger": ledger,
    }
    with (output / "milestone7_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    print(json.dumps(ledger, indent=2))


if __name__ == "__main__":
    main()
