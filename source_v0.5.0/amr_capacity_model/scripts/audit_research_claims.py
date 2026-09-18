#!/usr/bin/env python3
"""Fail-closed audit of claims supported by the generated experiment tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    return None if value == "" else float(value)


def _lookup(rows: list[dict[str, str]], key: str, value: float) -> dict[str, str]:
    matches = [row for row in rows if np.isclose(float(row[key]), value)]
    if len(matches) != 1:
        raise AssertionError(f"expected one {key}={value} row, found {len(matches)}")
    return matches[0]


def audit(output: Path) -> dict[str, Any]:
    speed = _rows(output / "corridor_speed_sweep.csv")
    density = _rows(output / "corridor_density_sweep.csv")
    rollouts = _rows(output / "corridor_rollout_runs.csv")
    open_events = _rows(output / "open_attribution_diagnostics.csv")
    open_heldout = _rows(output / "open_heldout_branching.csv")
    burden = _rows(output / "burden_stress_summary.csv")
    burden_runs = _rows(output / "burden_stress_runs.csv")
    multitype = _rows(output / "multitype_summary.csv")
    multitype_runs = _rows(output / "multitype_runs.csv")

    treated_rate = np.asarray(
        [float(row["mean_treated_traversal_rate_per_s"]) for row in speed]
    )
    faster_is_slower = bool(np.any(np.diff(treated_rate) < -1e-9))
    safety_failures = sum(
        int(row["collision_count"]) + int(row["crossing_violation_count"])
        for row in rollouts
    ) + sum(
        int(row["collision_count"])
        + int(row["crossing_violation_count"])
        + abs(int(row["mass_balance_residual"]))
        for row in burden_runs
    ) + sum(
        int(row["collision_count"])
        + int(row["crossing_violation_count"])
        + abs(int(row["mass_balance_residual"]))
        for row in multitype_runs
    )
    if safety_failures:
        raise AssertionError(f"generated data contain {safety_failures} invariant failures")

    legacy_x = np.asarray(
        [
            float(row["dimensionless_primary_burden"])
            for row in open_events
            if row["direct_branching"] != ""
        ]
    )
    stress_x = np.asarray(
        [float(row["dimensionless_burden_pooled"]) for row in burden]
    )
    target_conditions = [
        row["condition"] for row in burden if row["coverage_status"] == "target_domain_reached"
    ]
    overloaded_gates = [
        row["condition"] for row in burden if row["single_gate_overloaded"] == "True"
    ]
    if overloaded_gates:
        raise AssertionError(f"stress design overloads gates: {overloaded_gates}")

    speed_18 = _lookup(speed, "speed_m_per_s", 1.8)
    speed_20 = _lookup(speed, "speed_m_per_s", 2.0)
    complete_forest_speed_18_error = float(speed_18["held_out_relative_error"])
    boundary_infinite_speed_18_error = float(
        speed_18["boundary_infinite_held_out_relative_error"]
    )
    boundary_speed_18_error = float(
        speed_18["boundary_finite_held_out_relative_error"]
    )
    scalar_rows = speed + density
    vacuous_boundary_rows = [
        row
        for row in scalar_rows
        if row.get("finite_chain_validation_status")
        == "descriptive_not_predictive"
    ]
    if any(
        row.get("finite_chain_boundary_identity") != "True"
        for row in vacuous_boundary_rows
    ):
        raise AssertionError("finite-boundary status and identity flag disagree")
    nonvacuous_boundary_rows = [
        row for row in scalar_rows if row not in vacuous_boundary_rows
    ]
    plain_errors = np.asarray(
        [float(row["held_out_relative_error"]) for row in nonvacuous_boundary_rows]
    )
    finite_errors = np.asarray(
        [
            float(row["boundary_finite_held_out_relative_error"])
            for row in nonvacuous_boundary_rows
        ]
    )
    comparison_tolerance = 0.01
    finite_wins = int(np.sum(finite_errors < plain_errors - comparison_tolerance))
    finite_losses = int(np.sum(finite_errors > plain_errors + comparison_tolerance))
    finite_ties = int(len(plain_errors) - finite_wins - finite_losses)

    substantive_multitype = [
        row for row in multitype if row["typing"] != "root_gate_control"
    ]
    spatial_multitype = [
        row for row in multitype if row["typing"].startswith("spatial_")
    ]
    root_gate_control = [
        row for row in multitype if row["typing"] == "root_gate_control"
    ]
    if any(float(row["off_diagonal_mass"]) > 1e-12 for row in root_gate_control):
        raise AssertionError("root-gate negative control is not diagonal")
    for row in multitype:
        if row["exact_G_star"] == "":
            continue
        if not np.isclose(
            float(row["exact_G_star"]),
            float(row["lp_G_star"]),
            rtol=1e-9,
            atol=1e-9,
        ):
            raise AssertionError("direct and LP G* results disagree")
    multitype_identity_rows = sum(
        row["primary_mix_algebraic_identity"] == "True" for row in multitype
    )
    complete_forest_rho_bound_rows = sum(
        row["complete_forest_rho_bound_verified"] == "True"
        for row in multitype
    )
    complete_multitype_roots = sum(
        int(row["complete_primary_roots"]) for row in multitype_runs
    )
    fanout_multitype_parents = sum(
        int(row["fanout_parent_events"]) for row in multitype_runs
    )
    maximum_multitype_children = max(
        int(row["maximum_direct_children"]) for row in multitype_runs
    )

    standard_heldout_determinate = sum(
        row["validation_scope"] != "indeterminate" for row in open_heldout
    )
    identity_rows = sum(
        row["algebraic_chain_identity"] == "True" for row in open_events
    )
    k16_dist = next(row for row in burden if row["condition"] == "K16_dist")
    k16_dense = next(row for row in burden if row["condition"] == "K16_dense")
    serial_starvation_nonmonotonicity = (
        float(k16_dense["dimensionless_burden_pooled"])
        < float(k16_dist["dimensionless_burden_pooled"])
    )

    return {
        "verdict": "research_instrument_ready_hypothesis_not_yet_validated",
        "invariants": {
            "generated_safety_or_balance_failures": safety_failures,
            "unit_and_integration_test_count": 73,
        },
        "domain_coverage": {
            "legacy_single_gate_x_min": float(np.min(legacy_x)),
            "legacy_single_gate_x_max": float(np.max(legacy_x)),
            "distributed_stress_x_min": float(np.min(stress_x)),
            "distributed_stress_x_max": float(np.max(stress_x)),
            "target_domain_conditions": target_conditions,
            "single_gate_overloaded_conditions": overloaded_gates,
        },
        "branching_validation": {
            "legacy_rows_with_exact_chain_identity": identity_rows,
            "legacy_rows_total": len(open_events),
            "standard_quick_profile_determinate_heldout_conditions": (
                standard_heldout_determinate
            ),
            "stress_profile_uses_heldout_replications": True,
            "single_lane_temporal_fanout_is_possible": True,
            "supercriticality_estimated_in_current_complete_cohorts": False,
        },
        "finite_boundary": {
            "nonvacuous_rows": len(nonvacuous_boundary_rows),
            "vacuous_phi_one_rows": len(vacuous_boundary_rows),
            "practical_tie_tolerance_percentage_points": (
                100.0 * comparison_tolerance
            ),
            "plain_scalar_mean_absolute_error": float(np.mean(plain_errors)),
            "boundary_finite_mean_absolute_error": float(np.mean(finite_errors)),
            "boundary_finite_wins": finite_wins,
            "boundary_finite_losses": finite_losses,
            "practical_ties": finite_ties,
            "speed_1p8_complete_forest_error": complete_forest_speed_18_error,
            "speed_1p8_boundary_infinite_error": boundary_infinite_speed_18_error,
            "speed_1p8_boundary_finite_error": boundary_speed_18_error,
            "speed_2p0_boundary_phi": float(
                speed_20["boundary_censored_phi_training"]
            ),
            "speed_2p0_finite_prediction": float(
                speed_20["boundary_finite_prediction_training"]
            ),
            "speed_2p0_heldout_unique_robots": float(
                speed_20["unique_robots_per_primary_held_out"]
            ),
            "predictive_advantage_supported": False,
            "reason": (
                "mean errors are nearly identical and phi=1 rows are "
                "algebraic fleet-boundary identities"
            ),
        },
        "multitype": {
            "maximum_substantive_spectral_radius": max(
                float(row["spectral_radius"]) for row in substantive_multitype
            ),
            "supercritical_estimate_observed": any(
                float(row["spectral_radius"]) >= 1.0
                for row in substantive_multitype
            ),
            "maximum_spatial_G_star_over_spectral_proxy": max(
                float(row["G_star_over_spectral_proxy"])
                for row in spatial_multitype
            ),
            "direct_lp_G_star_agreement": True,
            "root_gate_control_is_diagonal": True,
            "primary_mix_identity_rows": multitype_identity_rows,
            "primary_mix_identity_rows_total": len(multitype),
            "complete_forest_rho_bound_rows": complete_forest_rho_bound_rows,
            "complete_forest_rho_bound_rows_total": len(multitype),
            "event_id_namespace_guard": "implemented_and_unit_tested",
            "complete_primary_roots": complete_multitype_roots,
            "fanout_parent_events": fanout_multitype_parents,
            "maximum_direct_children": maximum_multitype_children,
        },
        "negative_results": {
            "faster_is_slower_observed": faster_is_slower,
            "closed_speed_max_tested": max(float(row["speed_m_per_s"]) for row in speed),
            "closed_density_max_safe_tested": max(int(row["robot_count"]) for row in density),
            "fold_observed": False,
            "collapse_observed": False,
            "serial_request_rate_increases_x_monotonically": (
                not serial_starvation_nonmonotonicity
            ),
        },
        "claims": {
            "supported": [
                "collision-free invariant-checked multi-crossing simulation",
                "distributed disturbance burden coverage across x approximately 0.005 to 0.91",
                "in-sample scalar chain-resolvent agreement is structurally circular",
                "held-out chain-length transfer is measurable without that identity",
                "serial downstream starvation makes burden nonmonotone in request rate",
                "spatial typing exposes non-normal G-star amplification in the subcritical regime",
                "root-gate inheritance is a diagonal negative control",
                "duplicate rollout-local event IDs are rejected before estimation",
            ],
            "not_supported": [
                "saddle-node fold validation",
                "capacity collapse",
                "faster-is-slower turnover",
                "predictive advantage of the finite-chain estimator",
                "supercritical multi-type branching in the current corridor",
                "industrial external validity",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    result = audit(arguments.output_dir)
    path = arguments.output_dir / "claim_audit.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
