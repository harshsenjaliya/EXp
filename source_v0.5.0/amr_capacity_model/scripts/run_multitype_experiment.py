#!/usr/bin/env python3
"""Held-out multi-type branching audit for the multi-crossing corridor.

The experiment keeps each rollout's event-ID namespace separate, selects
complete trees by primary-root time, and compares severity and spatial event
typings.  It is designed to detect non-normal amplification without treating a
root-gate label or concatenated event IDs as topology evidence.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from amr_capacity import (
    OpenCorridorConfig,
    cascade_budget_direct,
    cascade_budget_lp,
    deterministic_arrival_times,
    diagnose_branching_forest,
    diagnose_multitype_forest_identity,
    pool_direct_branching_events,
    poisson_crossing_requests_multi,
    retype_events_by_minimum_speed,
    retype_events_by_position,
    run_open_paired_rollout,
    select_complete_root_cohorts,
    validate_branching_event_sets,
)
from amr_capacity.simulation import InterventionEvent


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs"


def _profile(name: str) -> dict[str, Any]:
    if name == "quick":
        return {
            "arrival_rates": (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80),
            "duration": 300.0,
            "followup": 60.0,
            "warmup_fraction": 0.30,
            "replications": 6,
            "crossing_count": 16,
            "rate_per_crossing": 0.015,
            "crossing_duration": (2.0, 4.0),
            "spatial_type_counts": (4, 8, 16),
        }
    if name == "paper":
        return {
            "arrival_rates": (
                0.10,
                0.15,
                0.20,
                0.25,
                0.30,
                0.35,
                0.40,
                0.50,
                0.60,
                0.80,
            ),
            "duration": 1800.0,
            "followup": 300.0,
            "warmup_fraction": 0.30,
            "replications": 30,
            "crossing_count": 16,
            "rate_per_crossing": 0.015,
            "crossing_duration": (2.0, 4.0),
            "spatial_type_counts": (4, 8, 16),
        }
    raise ValueError(f"unknown profile {name!r}")


def _config(crossing_count: int) -> OpenCorridorConfig:
    positions = tuple(float(item) for item in np.linspace(2.0, 28.0, crossing_count))
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
        crossing_x=positions,
        crossing_half_width=0.4,
    )


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    records = list(rows)
    if not records:
        raise ValueError(f"cannot write empty table {path}")
    keys: list[str] = []
    for record in records:
        for key in record:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def _root_gate_types(
    events: Iterable[InterventionEvent], crossing_count: int
) -> tuple[InterventionEvent, ...]:
    """Negative control: all descendants inherit one root-gate label.

    This necessarily produces a diagonal type matrix and therefore cannot
    encode directed propagation between gates.
    """

    records = tuple(events)
    root_gate = {
        event.event_id: event.crossing_id
        for event in records
        if event.is_primary and event.crossing_id is not None
    }
    typed: list[InterventionEvent] = []
    for event in records:
        if event.root_primary_event_id not in root_gate:
            raise ValueError("root gate is missing from a complete causal cohort")
        gate = int(root_gate[event.root_primary_event_id])
        if not 0 <= gate < crossing_count:
            raise ValueError("logged crossing_id lies outside the configured gates")
        typed.append(replace(event, event_type=gate))
    return tuple(typed)


def _typed_sets(
    event_sets: list[tuple[InterventionEvent, ...]],
    typing: str,
    type_count: int,
    corridor_length: float,
) -> list[tuple[InterventionEvent, ...]]:
    if typing == "scalar":
        return [
            tuple(replace(event, event_type=0) for event in events)
            for events in event_sets
        ]
    if typing == "severity_SR":
        return [retype_events_by_minimum_speed(events) for events in event_sets]
    if typing == "root_gate_control":
        return [_root_gate_types(events, type_count) for events in event_sets]
    if typing.startswith("spatial_"):
        edges = np.linspace(0.0, corridor_length, type_count + 1)
        return [retype_events_by_position(events, edges) for events in event_sets]
    raise ValueError(f"unknown typing {typing!r}")


def _non_normality(matrix: np.ndarray) -> float:
    norm = float(np.linalg.norm(matrix, ord="fro"))
    if norm == 0.0:
        return 0.0
    commutator = matrix.T @ matrix - matrix @ matrix.T
    return float(np.linalg.norm(commutator, ord="fro") / norm**2)


def _score_typing(
    *,
    arrival_rate: float,
    typing: str,
    type_count: int,
    event_sets: list[tuple[InterventionEvent, ...]],
    training_depth: int,
    corridor_length: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    typed = _typed_sets(event_sets, typing, type_count, corridor_length)
    split = len(typed) // 2
    training = typed[:split]
    validation = typed[split:]
    heldout = validate_branching_event_sets(
        training,
        validation,
        number_of_types=type_count,
        max_generation=training_depth,
    )
    estimate = heldout.training_branching
    identity = diagnose_multitype_forest_identity(
        training, number_of_types=type_count
    )
    matrix = np.asarray(estimate.matrix, dtype=float)
    rho = float(estimate.spectral_radius)
    spectral_proxy: float | str = ""
    exact_budget: float | str = ""
    non_normality_gap: float | str = ""
    lp_budget: float | str = ""
    lp_residual: float | str = ""
    if rho < 1.0:
        spectral_proxy = 1.0 / (1.0 - rho)
        exact_budget, _ = cascade_budget_direct(matrix)
        lp = cascade_budget_lp(matrix)
        if not lp.feasible or lp.budget is None:
            raise AssertionError("subcritical direct solve disagrees with LP feasibility")
        lp_budget = float(lp.budget)
        lp_residual = float(lp.minimum_residual or 0.0)
        if not np.isclose(exact_budget, lp_budget, rtol=1e-9, atol=1e-9):
            raise AssertionError("direct and LP cascade budgets disagree")
        non_normality_gap = exact_budget / spectral_proxy

    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    active_parent_counts = estimate.parent_counts[estimate.parent_counts > 0]
    row = {
        "arrival_rate_per_s": arrival_rate,
        "typing": typing,
        "type_count": type_count,
        "training_replications": split,
        "validation_replications": len(typed) - split,
        "training_primary_events": int(
            sum(event.is_primary for events in training for event in events)
        ),
        "training_causal_events": identity.causal_events,
        "training_mean_progeny": identity.training_mean_progeny,
        "primary_mix_identity_residual": identity.identity_residual,
        "primary_mix_algebraic_identity": identity.algebraic_identity,
        "complete_forest_rho_bound_verified": (
            identity.complete_forest_rho_bound_verified
        ),
        "training_primary_mix_status": (
            "descriptive_not_predictive"
            if identity.algebraic_identity
            else "not_an_algebraic_identity"
        ),
        "validation_primary_events": heldout.validation_primary_events,
        "validation_causal_events": heldout.validation_causal_events,
        "active_parent_types": int(active_parent_counts.size),
        "minimum_active_parent_count": (
            int(np.min(active_parent_counts)) if active_parent_counts.size else 0
        ),
        "spectral_radius": rho,
        "regime": (
            "subcritical_resolvent_defined"
            if rho < 1.0
            else "supercritical_resolvent_undefined"
        ),
        "spectral_proxy_1_over_1_minus_rho": spectral_proxy,
        "exact_G_star": exact_budget,
        "lp_G_star": lp_budget,
        "lp_minimum_residual": lp_residual,
        "G_star_over_spectral_proxy": non_normality_gap,
        "commutator_non_normality": _non_normality(matrix),
        "off_diagonal_mass": float(np.sum(off_diagonal)),
        "heldout_measured_progeny": heldout.measured_total_progeny_per_primary,
        "infinite_primary_mix_prediction": (
            ""
            if heldout.predicted_total_progeny_per_primary is None
            else heldout.predicted_total_progeny_per_primary
        ),
        "infinite_primary_mix_relative_error": (
            "" if heldout.relative_error is None else heldout.relative_error
        ),
        "training_derived_depth": training_depth,
        "finite_depth_primary_mix_prediction": (
            heldout.truncated_predicted_total_progeny_per_primary
        ),
        "finite_depth_primary_mix_relative_error": heldout.truncated_relative_error,
        "validation_scope": "independent_complete_root_cohorts",
    }
    entries: list[dict[str, Any]] = []
    for parent_type in range(type_count):
        for child_type in range(type_count):
            entries.append(
                {
                    "arrival_rate_per_s": arrival_rate,
                    "typing": typing,
                    "type_count": type_count,
                    "parent_type": parent_type,
                    "child_type": child_type,
                    "eligible_parent_events": int(
                        estimate.parent_counts[parent_type]
                    ),
                    "direct_child_events": int(
                        estimate.direct_child_counts[parent_type, child_type]
                    ),
                    "B_entry": float(matrix[parent_type, child_type]),
                }
            )
    return row, entries


def _run(
    settings: dict[str, Any], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    config = _config(int(settings["crossing_count"]))
    duration = float(settings["duration"])
    root_start = float(settings["warmup_fraction"]) * duration
    root_end = duration - float(settings["followup"])
    if root_end <= root_start:
        raise ValueError("profile leaves no primary-root cohort window")

    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    for rate_index, arrival_rate in enumerate(settings["arrival_rates"]):
        event_sets: list[tuple[InterventionEvent, ...]] = []
        selections = []
        for replication in range(int(settings["replications"])):
            arrivals = deterministic_arrival_times(arrival_rate, duration)
            # Common disturbance realizations across arrival-rate conditions.
            disturbance_seed = seed + 1_000_000 + replication
            requests = poisson_crossing_requests_multi(
                float(settings["rate_per_crossing"]),
                duration,
                settings["crossing_duration"],
                int(settings["crossing_count"]),
                seed=disturbance_seed,
                start_time=3.0,
            )
            pair = run_open_paired_rollout(
                config,
                duration,
                arrivals,
                requests,
                record_stride=20,
            )
            if (
                pair.treated.collision_count
                or pair.treated.crossing_violation_count
                or pair.treated.mass_balance_residual
            ):
                raise AssertionError("multi-type rollout violated a hard invariant")
            selection = select_complete_root_cohorts(
                pair.treated.events, root_start, root_end
            )
            event_sets.append(selection.events)
            selections.append(selection)
            structure = diagnose_branching_forest(selection.events)
            retained_time = pair.treated.trajectory.time >= root_start
            completed = sum(
                item.completion_time >= root_start
                for item in pair.treated.completed_traversals
            )
            raw_rows.append(
                {
                    "arrival_rate_per_s": arrival_rate,
                    "replication": replication,
                    "disturbance_seed": disturbance_seed,
                    "selected_primary_roots": selection.selected_primary_roots,
                    "complete_primary_roots": selection.complete_primary_roots,
                    "unresolved_primary_roots": selection.unresolved_primary_roots,
                    "causal_events": structure.causal_events,
                    "mean_complete_root_size": (
                        float(np.mean(selection.complete_root_sizes))
                        if selection.complete_root_sizes
                        else ""
                    ),
                    "maximum_generation": structure.maximum_generation,
                    "fanout_parent_events": structure.fanout_parent_events,
                    "maximum_direct_children": structure.maximum_direct_children,
                    "repeated_robot_events_within_root": (
                        structure.repeated_robot_events_within_root
                    ),
                    "mean_work_in_process": float(
                        np.mean(pair.treated.trajectory.work_in_process[retained_time])
                    ),
                    "completion_rate_per_s": completed / (duration - root_start),
                    "collision_count": pair.treated.collision_count,
                    "crossing_violation_count": pair.treated.crossing_violation_count,
                    "mass_balance_residual": pair.treated.mass_balance_residual,
                }
            )

        split = len(selections) // 2
        training_depth = max(
            (
                generation
                for selection in selections[:split]
                for generation in selection.complete_root_max_generations
            ),
            default=0,
        )
        typings = [("scalar", 1), ("severity_SR", 2)]
        typings.extend(
            (f"spatial_{count}", int(count))
            for count in settings["spatial_type_counts"]
        )
        typings.append(("root_gate_control", int(settings["crossing_count"])))
        for typing, type_count in typings:
            row, entries = _score_typing(
                arrival_rate=float(arrival_rate),
                typing=typing,
                type_count=type_count,
                event_sets=event_sets,
                training_depth=training_depth,
                corridor_length=config.corridor_length,
            )
            if typing == "root_gate_control" and float(row["off_diagonal_mass"]) > 1e-12:
                raise AssertionError("root-gate control must be diagonal by definition")
            summary_rows.append(row)
            matrix_rows.extend(entries)
        print(f"completed arrival-rate condition {arrival_rate:.2f}", flush=True)
    return raw_rows, summary_rows, matrix_rows


def _plot_summary(rows: list[dict[str, Any]], output: Path) -> None:
    def optional_number(row: dict[str, Any], key: str) -> float:
        value = row[key]
        return np.nan if value == "" or value is None else float(value)

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    colors = {
        "severity_SR": "#b42318",
        "spatial_4": "#175cd3",
        "spatial_8": "#067647",
        "spatial_16": "#7a5af8",
    }
    for typing, color in colors.items():
        selected = [row for row in rows if row["typing"] == typing]
        axes[0, 0].plot(
            [row["arrival_rate_per_s"] for row in selected],
            [row["spectral_radius"] for row in selected],
            "o-",
            label=typing,
            color=color,
        )
    axes[0, 0].axhline(1.0, color="black", linestyle=":", linewidth=1.2)
    axes[0, 0].set(
        xlabel="Arrival rate (jobs/s)",
        ylabel=r"Training spectral radius $\hat\rho(B)$",
        title="Training spectral radius after correct pooling",
    )
    axes[0, 0].legend(fontsize=8)

    spatial = [row for row in rows if row["typing"] == "spatial_16"]
    axes[0, 1].plot(
        [row["arrival_rate_per_s"] for row in spatial],
        [optional_number(row, "spectral_proxy_1_over_1_minus_rho") for row in spatial],
        "o--",
        label=r"spectral proxy $1/(1-\rho)$",
        color="#64748b",
    )
    axes[0, 1].plot(
        [row["arrival_rate_per_s"] for row in spatial],
        [optional_number(row, "exact_G_star") for row in spatial],
        "s-",
        label=r"exact $G^\star$",
        color="#175cd3",
    )
    axes[0, 1].set(
        xlabel="Arrival rate (jobs/s)",
        ylabel="Worst-case cumulative cascade budget",
        title="Non-normal topology penalty",
    )
    axes[0, 1].legend(fontsize=8)

    scalar = {
        row["arrival_rate_per_s"]: row for row in rows if row["typing"] == "scalar"
    }
    axes[1, 0].plot(
        [row["arrival_rate_per_s"] for row in spatial],
        [row["heldout_measured_progeny"] for row in spatial],
        "ko-",
        label="held-out measured",
    )
    axes[1, 0].plot(
        [row["arrival_rate_per_s"] for row in spatial],
        [
            optional_number(
                scalar[row["arrival_rate_per_s"]],
                "infinite_primary_mix_prediction",
            )
            for row in spatial
        ],
        "^--",
        label="scalar prediction",
        color="#b54708",
    )
    axes[1, 0].plot(
        [row["arrival_rate_per_s"] for row in spatial],
        [optional_number(row, "infinite_primary_mix_prediction") for row in spatial],
        "s-",
        label="spatial-16 prediction",
        color="#7a5af8",
    )
    axes[1, 0].set(
        xlabel="Arrival rate (jobs/s)",
        ylabel="Causal events per primary",
        title="Independent held-out prediction",
    )
    axes[1, 0].legend(fontsize=8)

    for typing, color in colors.items():
        if not typing.startswith("spatial_"):
            continue
        selected = [row for row in rows if row["typing"] == typing]
        axes[1, 1].plot(
            [row["arrival_rate_per_s"] for row in selected],
            [optional_number(row, "G_star_over_spectral_proxy") for row in selected],
            "o-",
            label=typing,
            color=color,
        )
    axes[1, 1].axhline(1.0, color="black", linestyle=":", linewidth=1.2)
    axes[1, 1].set(
        xlabel="Arrival rate (jobs/s)",
        ylabel=r"$G^\star/[1/(1-\rho)]$",
        title="Amplification missed by spectral radius",
    )
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.suptitle("Multi-type audit: non-normal amplification with fail-closed regime labels")
    figure.savefig(output, dpi=200)
    plt.close(figure)


def _plot_matrix(
    rows: list[dict[str, Any]], output: Path, *, typing: str = "spatial_16"
) -> None:
    selected = [row for row in rows if row["typing"] == typing]
    arrival_rate = max(float(row["arrival_rate_per_s"]) for row in selected)
    entries = [
        row
        for row in selected
        if np.isclose(float(row["arrival_rate_per_s"]), arrival_rate)
    ]
    type_count = int(entries[0]["type_count"])
    matrix = np.zeros((type_count, type_count), dtype=float)
    for row in entries:
        matrix[int(row["parent_type"]), int(row["child_type"])] = float(row["B_entry"])
    figure, axis = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    image = axis.imshow(matrix, origin="lower", cmap="magma", aspect="equal")
    figure.colorbar(image, ax=axis, label="Expected direct children")
    axis.set(
        xlabel="Child spatial region",
        ylabel="Parent spatial region",
        title=f"Spatial branching matrix at arrival rate {arrival_rate:.2f}/s",
    )
    figure.savefig(output, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("quick", "paper"), default="quick")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    settings = _profile(arguments.profile)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    raw, summary, matrices = _run(settings, arguments.seed)
    _write_csv(arguments.output_dir / "multitype_runs.csv", raw)
    _write_csv(arguments.output_dir / "multitype_summary.csv", summary)
    _write_csv(arguments.output_dir / "multitype_matrix_entries.csv", matrices)
    _plot_summary(summary, arguments.output_dir / "multitype_validation.png")
    _plot_matrix(matrices, arguments.output_dir / "multitype_spatial_matrix.png")

    substantive = [row for row in summary if row["typing"] != "root_gate_control"]
    spatial = [row for row in summary if row["typing"].startswith("spatial_")]
    spatial_gaps = [
        float(row["G_star_over_spectral_proxy"])
        for row in spatial
        if row["G_star_over_spectral_proxy"] != ""
    ]
    manifest = {
        "profile": arguments.profile,
        "seed": arguments.seed,
        "settings": settings,
        "config": asdict(_config(int(settings["crossing_count"]))),
        "findings": {
            "supercritical_training_estimate_observed": any(
                float(row["spectral_radius"]) >= 1.0 for row in substantive
            ),
            "rho_crossing_observed": (
                any(float(row["spectral_radius"]) < 1.0 for row in substantive)
                and any(float(row["spectral_radius"]) >= 1.0 for row in substantive)
            ),
            "maximum_spatial_non_normality_gap": max(
                spatial_gaps
            ) if spatial_gaps else None,
            "root_gate_control_is_diagonal": all(
                float(row["off_diagonal_mass"]) <= 1e-12
                for row in summary
                if row["typing"] == "root_gate_control"
            ),
            "complete_forest_rho_bound_verified": all(
                bool(row["complete_forest_rho_bound_verified"])
                for row in summary
            ),
            "training_primary_mix_rows_algebraic": sum(
                bool(row["primary_mix_algebraic_identity"])
                for row in summary
            ),
            "training_primary_mix_rows_total": len(summary),
            "complete_primary_roots": sum(
                int(row["complete_primary_roots"]) for row in raw
            ),
            "fanout_parent_events": sum(
                int(row["fanout_parent_events"]) for row in raw
            ),
            "maximum_direct_children": max(
                int(row["maximum_direct_children"]) for row in raw
            ),
        },
        "interpretation": {
            "pooling": (
                "event IDs are rollout-local; sufficient statistics are pooled "
                "without concatenating ID namespaces"
            ),
            "cohorts": (
                "roots are selected in a fixed window and their complete trees "
                "are retained; unresolved roots are excluded and counted"
            ),
            "root_gate_control": (
                "assigning every descendant its root gate makes B diagonal and "
                "cannot establish directed topology"
            ),
            "finite_depth": (
                "the truncation depth is the maximum generation in training only; "
                "it is not presented as a finite-fleet M_N validation"
            ),
            "complete_tree_bound": (
                "direct-edge conservation forces rho(B) <= 1 for the complete-"
                "tree MLE; supercritical inference requires censored exposure"
            ),
        },
    }
    with (arguments.output_dir / "multitype_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps(manifest["findings"], indent=2))


if __name__ == "__main__":
    main()
