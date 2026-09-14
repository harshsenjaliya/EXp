#!/usr/bin/env python3
"""Generate Milestone 6 estimator and cross-map kinematic evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from amr_capacity.branching_exposure import (
    bootstrap_censored_multitype_branching,
    estimate_naive_observed_tree,
    scale_branching_matrix,
    simulate_censored_branching_experiment,
)
from amr_capacity.kinematic_maps import (
    DifferentialDriveLimits,
    assert_static_clearance,
    plan_kinematic_speed_profile,
    run_cross_map_obstacle_benchmark,
    standard_map_catalogue,
    standard_robot_catalogue,
)


PROFILE = {
    "smoke": {
        "rollouts": 50,
        "bootstrap": 30,
        "dt": 0.05,
    },
    "quick": {
        "rollouts": 300,
        "bootstrap": 300,
        "dt": 0.025,
    },
    "paper": {
        "rollouts": 2_000,
        "bootstrap": 2_000,
        "dt": 0.01,
    },
}


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_branching_validation(
    output: Path,
    profile_name: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    settings = PROFILE[profile_name]
    base = np.array([[0.25, 0.80], [0.10, 0.30]], dtype=float)
    recovery = np.array([[1.0, 0.70], [0.90, 1.20]], dtype=float)
    targets = (0.60, 0.85, 1.00, 1.15, 1.30)
    summary: list[dict[str, object]] = []
    entries: list[dict[str, object]] = []

    for index, target in enumerate(targets):
        truth = scale_branching_matrix(base, target)
        rollouts = simulate_censored_branching_experiment(
            truth,
            recovery,
            rollout_count=int(settings["rollouts"]),
            roots_per_rollout=3,
            horizon=2.0,
            population_cap=500,
            root_type_probabilities=(0.55, 0.45),
            seed=2026091600 + index,
        )
        interval = bootstrap_censored_multitype_branching(
            rollouts,
            recovery,
            replications=int(settings["bootstrap"]),
            confidence_level=0.95,
            seed=2026092600 + index,
        )
        naive = estimate_naive_observed_tree(rollouts, 2)
        capped = sum(item.population_cap_reached for item in rollouts)
        summary.append(
            {
                "profile": profile_name,
                "true_rho": target,
                "exposure_rho": interval.point.spectral_radius,
                "rho_ci_lower": interval.spectral_radius_lower,
                "rho_ci_upper": interval.spectral_radius_upper,
                "p_rho_gt_1": interval.probability_supercritical,
                "naive_rho": naive.spectral_radius,
                "exposure_abs_error": abs(interval.point.spectral_radius - target),
                "naive_abs_error": abs(naive.spectral_radius - target),
                "rollouts": len(rollouts),
                "events": interval.point.event_count,
                "temporally_censored_parents": (
                    interval.point.temporally_censored_parent_count
                ),
                "population_capped_rollouts": capped,
                "bootstrap_successes": interval.successful_replications,
                "bootstrap_requested": interval.requested_replications,
            }
        )
        for parent_type in range(2):
            for child_type in range(2):
                entries.append(
                    {
                        "profile": profile_name,
                        "true_rho": target,
                        "parent_type": parent_type,
                        "child_type": child_type,
                        "true_B": truth[parent_type, child_type],
                        "exposure_B": interval.point.matrix[
                            parent_type, child_type
                        ],
                        "B_ci_lower": interval.matrix_lower[
                            parent_type, child_type
                        ],
                        "B_ci_upper": interval.matrix_upper[
                            parent_type, child_type
                        ],
                        "naive_B": naive.matrix[parent_type, child_type],
                        "direct_children": interval.point.direct_child_counts[
                            parent_type, child_type
                        ],
                        "kernel_exposure": interval.point.exposure_mass[
                            parent_type, child_type
                        ],
                        "parent_events": interval.point.parent_counts[parent_type],
                    }
                )

    write_rows(output / "milestone6_branching_validation.csv", summary)
    write_rows(output / "milestone6_branching_matrix_entries.csv", entries)

    true = np.array([float(row["true_rho"]) for row in summary])
    exposure = np.array([float(row["exposure_rho"]) for row in summary])
    naive_value = np.array([float(row["naive_rho"]) for row in summary])
    lower = np.array([float(row["rho_ci_lower"]) for row in summary])
    upper = np.array([float(row["rho_ci_upper"]) for row in summary])
    figure, axis = plt.subplots(figsize=(6.8, 5.0), constrained_layout=True)
    axis.plot([0.5, 1.4], [0.5, 1.4], color="0.25", linestyle="--", label="ideal")
    axis.axhline(1.0, color="#8c2d04", linewidth=1.0)
    axis.axvline(1.0, color="#8c2d04", linewidth=1.0)
    axis.errorbar(
        true,
        exposure,
        yerr=np.vstack((exposure - lower, upper - exposure)),
        fmt="o-",
        capsize=4,
        color="#005a9c",
        label="exposure-aware",
    )
    axis.plot(true, naive_value, "s--", color="#d55e00", label="complete-tree ablation")
    axis.set(
        xlabel="ground-truth spectral radius",
        ylabel="estimated spectral radius",
        title="Censored branching recovery across criticality",
        xlim=(0.52, 1.36),
        ylim=(0.45, 1.45),
    )
    axis.legend(frameon=False)
    figure.savefig(output / "milestone6_branching_recovery.png", dpi=220)
    plt.close(figure)
    return summary, entries


def run_kinematic_validation(
    output: Path,
    profile_name: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    settings = PROFILE[profile_name]
    platforms = standard_robot_catalogue()
    profile_rows: list[dict[str, object]] = []
    obstacle_rows: list[dict[str, object]] = []
    maps = standard_map_catalogue()
    for robot_name, limits in platforms:
        for map_spec in maps:
            clearance = assert_static_clearance(map_spec, limits)
            for route in map_spec.routes:
                planned = plan_kinematic_speed_profile(
                    route,
                    limits,
                    map_speed_limit=map_spec.speed_limit,
                )
                profile_rows.append(
                    {
                        "profile": profile_name,
                        "robot_class": robot_name,
                        "map": map_spec.name,
                        "route": route.name,
                        "route_length_m": route.length,
                        "map_speed_limit_mps": map_spec.speed_limit,
                        "nominal_time_s": planned.nominal_time,
                        "straight_reference_time_s": (
                            planned.straight_reference_time
                        ),
                        "turning_penalty_s": planned.turning_penalty,
                        "turning_penalty_fraction": (
                            planned.turning_penalty
                            / planned.straight_reference_time
                        ),
                        "max_yaw_rate_rps": planned.max_abs_yaw_rate,
                        "max_yaw_acceleration_rps2": (
                            planned.max_abs_yaw_acceleration
                        ),
                        "max_lateral_acceleration_mps2": (
                            planned.max_lateral_acceleration
                        ),
                        "max_wheel_speed_mps": planned.max_abs_wheel_speed,
                        "envelope_binding_fraction": (
                            planned.envelope_binding_fraction
                        ),
                        "minimum_static_clearance_m": clearance,
                        "conflict_zones": len(map_spec.conflict_zones),
                        "static_obstacles": len(map_spec.static_obstacles),
                    }
                )

        benchmark = run_cross_map_obstacle_benchmark(
            limits,
            dt=float(settings["dt"]),
            sensor_range=10.0,
        )
        for map_name, route_name, kind, intensity, result in benchmark:
            activation_delay = (
                result.activations[0].activation_delay
                if result.activations
                else np.nan
            )
            obstacle_rows.append(
                {
                    "profile": profile_name,
                    "robot_class": robot_name,
                    "map": map_name,
                    "route": route_name,
                    "obstacle_kind": kind,
                    "disturbance_level": intensity,
                    "nominal_time_s": result.nominal_time,
                    "traversal_time_s": result.traversal_time,
                    "delay_s": result.delay,
                    "severity_weighted_loss_s": (
                        result.severity_weighted_loss
                    ),
                    "dimensionless_burden_x": (
                        result.severity_weighted_loss / result.nominal_time
                    ),
                    "delay_fraction": result.delay / result.nominal_time,
                    "activation_delay_s": activation_delay,
                    "collision_count": result.collision_count,
                    "obstacle_violation_count": (
                        result.obstacle_violation_count
                    ),
                    "minimum_stopping_margin_m": (
                        result.minimum_stopping_margin
                    ),
                    "minimum_static_clearance_m": (
                        result.minimum_static_clearance
                    ),
                    "max_yaw_rate_rps": result.max_abs_yaw_rate,
                    "max_yaw_acceleration_rps2": (
                        result.max_abs_yaw_acceleration
                    ),
                    "max_lateral_acceleration_mps2": (
                        result.max_lateral_acceleration
                    ),
                    "max_wheel_speed_mps": result.max_abs_wheel_speed,
                    "yaw_rate_utilization": (
                        result.max_abs_yaw_rate / limits.max_yaw_rate
                    ),
                    "yaw_acceleration_utilization": (
                        result.max_abs_yaw_acceleration
                        / limits.max_yaw_acceleration
                    ),
                    "lateral_acceleration_utilization": (
                        result.max_lateral_acceleration
                        / limits.max_lateral_acceleration
                    ),
                    "wheel_speed_utilization": (
                        result.max_abs_wheel_speed / limits.max_wheel_speed
                    ),
                }
            )
    write_rows(output / "milestone6_kinematic_profiles.csv", profile_rows)
    write_rows(output / "milestone6_obstacle_benchmark.csv", obstacle_rows)

    figure, axes = plt.subplots(2, 3, figsize=(13.0, 8.0), constrained_layout=True)
    for axis, map_spec in zip(axes.flat, maps, strict=True):
        for route in map_spec.routes:
            axis.plot(route.x, route.y, linewidth=2.0, label=route.name)
        for obstacle in map_spec.static_obstacles:
            circle = plt.Circle(
                (obstacle.x, obstacle.y),
                obstacle.radius,
                color="0.25",
                alpha=0.45,
            )
            axis.add_patch(circle)
        for zone in map_spec.conflict_zones:
            circle = plt.Circle(
                (zone.x, zone.y),
                zone.radius,
                fill=False,
                color="#d55e00",
                linestyle="--",
                linewidth=1.5,
            )
            axis.add_patch(circle)
        axis.set_title(map_spec.name.replace("_", " "))
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.2)
        if len(map_spec.routes) > 1:
            axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Milestone 6 route and obstacle catalogue")
    figure.savefig(output / "milestone6_map_catalogue.png", dpi=220)
    plt.close(figure)

    kinds = sorted({str(row["obstacle_kind"]) for row in obstacle_rows})
    map_names = [item.name for item in maps]
    delay_grid = np.full((len(map_names), len(kinds)), np.nan)
    for map_index, map_name in enumerate(map_names):
        for kind_index, kind in enumerate(kinds):
            values = [
                float(row["delay_s"])
                for row in obstacle_rows
                if row["map"] == map_name and row["obstacle_kind"] == kind
            ]
            if values:
                delay_grid[map_index, kind_index] = float(np.mean(values))
    figure, axis = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    image = axis.imshow(delay_grid, aspect="auto", cmap="viridis")
    axis.set_xticks(np.arange(len(kinds)), [item.replace("_", " ") for item in kinds])
    axis.set_yticks(
        np.arange(len(map_names)), [item.replace("_", " ") for item in map_names]
    )
    axis.tick_params(axis="x", rotation=30)
    axis.set_title("Mean traversal delay by map and obstruction")
    figure.colorbar(image, ax=axis, label="delay (s)")
    figure.savefig(output / "milestone6_obstacle_delay.png", dpi=220)
    plt.close(figure)
    return profile_rows, obstacle_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE),
        default="quick",
        help="smoke is CI-sized; paper is the preregistered full run",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "milestone6",
    )
    parser.add_argument(
        "--skip-obstacles",
        action="store_true",
        help="generate branching and nominal kinematic results only",
    )
    arguments = parser.parse_args()
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)

    branching_rows, _ = run_branching_validation(output, arguments.profile)
    if arguments.skip_obstacles:
        kinematic_rows: list[dict[str, object]] = []
        obstacle_rows: list[dict[str, object]] = []
    else:
        kinematic_rows, obstacle_rows = run_kinematic_validation(
            output, arguments.profile
        )

    manifest = {
        "milestone": 6,
        "profile": arguments.profile,
        "settings": PROFILE[arguments.profile],
        "seed_family": "20260916xx estimator; 20260926xx bootstrap",
        "branching_model": (
            "lambda_ab(u)=B_ab*gamma_ab*exp(-gamma_ab*u); "
            "B_hat_ab=N_ab/sum_i(1-exp(-gamma_ab*C_i))"
        ),
        "kinematics": "unicycle centerline with differential-drive wheel limits",
        "nominal_time_rule": "turning and map limits belong to T0(M), not g",
        "maps": [
            "straight_crossing",
            "l_turn",
            "s_curve",
            "two_to_one_merge",
            "four_way_intersection",
            "warehouse_grid",
        ],
        "robot_classes": [
            name for name, _ in standard_robot_catalogue()
        ],
        "disturbance_levels": ["light", "medium", "heavy"],
        "obstacles": [
            "unexpected_stationary",
            "pedestrian_crossing",
            "moving_forklift_crossing",
            "temporary_aisle_closure",
            "occlusion_delayed_detection",
        ],
        "branching_rows": len(branching_rows),
        "kinematic_rows": len(kinematic_rows),
        "obstacle_rows": len(obstacle_rows),
        "hard_assertions": {
            "zero_static_collisions": all(
                int(row["collision_count"]) == 0 for row in obstacle_rows
            ),
            "zero_route_obstacle_violations": all(
                int(row["obstacle_violation_count"]) == 0
                for row in obstacle_rows
            ),
            "nonnegative_stopping_margin": all(
                float(row["minimum_stopping_margin_m"]) >= -1e-7
                for row in obstacle_rows
            ),
            "bounded_route_kinematics": all(
                float(row["yaw_rate_utilization"]) <= 1.001
                and float(row["lateral_acceleration_utilization"]) <= 1.001
                and float(row["wheel_speed_utilization"]) <= 1.001
                and float(row["yaw_acceleration_utilization"]) <= 1.02
                for row in obstacle_rows
            ),
        },
        "claim_boundary": (
            "Synthetic recovery validates the estimator implementation. "
            "Cross-map route runs validate kinematic and shield invariants. "
            "Neither alone establishes a physical multi-robot capacity fold."
        ),
    }
    with (output / "milestone6_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")

    branching_log = [
        {
            key: row[key]
            for key in (
                "true_rho",
                "exposure_rho",
                "rho_ci_lower",
                "rho_ci_upper",
                "p_rho_gt_1",
                "naive_rho",
                "events",
            )
        }
        for row in branching_rows
    ]
    obstacle_log = {
        "rows": len(obstacle_rows),
        "minimum_x": min(
            (float(row["dimensionless_burden_x"]) for row in obstacle_rows),
            default=None,
        ),
        "maximum_x": max(
            (float(row["dimensionless_burden_x"]) for row in obstacle_rows),
            default=None,
        ),
        "maximum_yaw_rate_utilization": max(
            (float(row["yaw_rate_utilization"]) for row in obstacle_rows),
            default=None,
        ),
        "maximum_yaw_acceleration_utilization": max(
            (
                float(row["yaw_acceleration_utilization"])
                for row in obstacle_rows
            ),
            default=None,
        ),
        "maximum_lateral_acceleration_utilization": max(
            (
                float(row["lateral_acceleration_utilization"])
                for row in obstacle_rows
            ),
            default=None,
        ),
        "maximum_wheel_speed_utilization": max(
            (float(row["wheel_speed_utilization"]) for row in obstacle_rows),
            default=None,
        ),
    }
    print(f"wrote Milestone 6 artifacts to {output}")
    print("BRANCHING_SUMMARY=" + json.dumps(branching_log, sort_keys=True))
    print("OBSTACLE_SUMMARY=" + json.dumps(obstacle_log, sort_keys=True))
    print("HARD_ASSERTIONS=" + json.dumps(manifest["hard_assertions"], sort_keys=True))


if __name__ == "__main__":
    main()
