#!/usr/bin/env python3
"""Map the dimensionless burden domain without overloading one crossing gate.

This experiment uses several independently guarded crossing locations.  The
per-gate offered load remains below one, so a large aggregate burden is not
created by making a single exogenous service queue unstable.  Results are a
coverage and stress audit only: no fold is claimed without an independently
detected loss of a stable operating branch.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from amr_capacity.estimation import (
    diagnose_chain_forest,
    estimate_direct_branching,
    validate_branching_event_sets,
)
from amr_capacity.open_metrics import (
    analyze_operating_window,
    bootstrap_mean_interval,
)
from amr_capacity.open_system import (
    OpenCorridorConfig,
    poisson_arrival_times,
    poisson_crossing_requests_multi,
    run_open_paired_rollout,
)
from amr_capacity.simulation import InterventionEvent


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs"


def _profile(name: str) -> dict[str, Any]:
    levels = (
        {"name": "K01_base", "crossings": 1, "rate_per_crossing": 0.025},
        {"name": "K02_dist", "crossings": 2, "rate_per_crossing": 0.05},
        {"name": "K04_dist", "crossings": 4, "rate_per_crossing": 0.05},
        {"name": "K08_dist", "crossings": 8, "rate_per_crossing": 0.05},
        {"name": "K12_dist", "crossings": 12, "rate_per_crossing": 0.05},
        {"name": "K16_dist", "crossings": 16, "rate_per_crossing": 0.05},
        {"name": "K16_dense", "crossings": 16, "rate_per_crossing": 0.10},
        {
            "name": "K16_midload",
            "crossings": 16,
            "rate_per_crossing": 0.05,
            "arrival_rate": 0.35,
        },
        {
            "name": "K16_long",
            "crossings": 16,
            "rate_per_crossing": 0.025,
            "arrival_rate": 0.20,
            "crossing_duration": (8.0, 12.0),
        },
        {
            "name": "K16_extreme",
            "crossings": 16,
            "rate_per_crossing": 0.025,
            "arrival_rate": 0.10,
            "crossing_duration": (8.0, 12.0),
        },
    )
    if name == "quick":
        return {
            "levels": levels,
            "duration": 300.0,
            "followup": 60.0,
            "replications": 6,
            "arrival_rate": 0.8,
            "crossing_duration": (2.0, 4.0),
            "bootstrap_resamples": 2000,
        }
    if name == "paper":
        return {
            "levels": levels,
            "duration": 1800.0,
            "followup": 300.0,
            "replications": 30,
            "arrival_rate": 0.8,
            "crossing_duration": (2.0, 4.0),
            "bootstrap_resamples": 10000,
        }
    raise ValueError(f"unknown profile {name!r}")


def _crossing_positions(count: int) -> tuple[float, ...]:
    if count == 1:
        return (15.0,)
    return tuple(float(item) for item in np.linspace(2.0, 28.0, count))


def _config(crossing_count: int) -> OpenCorridorConfig:
    positions = _crossing_positions(crossing_count)
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
        crossing_x=positions[0] if crossing_count == 1 else positions,
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


def _complete_root_cohorts(
    events: Iterable[InterventionEvent],
    root_start: float,
    root_end: float,
) -> tuple[tuple[InterventionEvent, ...], int, int]:
    """Return complete trees whose primary roots lie in a fixed cohort window."""

    causal = tuple(event for event in events if event.is_causal)
    roots = {
        event.event_id: event
        for event in causal
        if event.is_primary and root_start <= event.start_time < root_end
    }
    groups: dict[int, list[InterventionEvent]] = {key: [] for key in roots}
    for event in causal:
        if event.root_primary_event_id in groups:
            assert event.root_primary_event_id is not None
            groups[event.root_primary_event_id].append(event)
    complete: list[InterventionEvent] = []
    unresolved = 0
    for records in groups.values():
        if any(event.censored for event in records):
            unresolved += 1
        else:
            complete.extend(records)
    return tuple(complete), len(groups) - unresolved, unresolved


def _status(x_value: float) -> str:
    if 0.1 <= x_value <= 1.0:
        return "target_domain_reached"
    if x_value < 0.01:
        return "flat_shelf"
    if x_value < 0.1:
        return "extended_but_below_target"
    return "above_target_domain"


def _run(
    settings: dict[str, Any], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    duration = float(settings["duration"])
    warmup = duration / 2.0
    cohort_end = duration - float(settings["followup"])
    if cohort_end <= warmup:
        raise ValueError("followup leaves no root-cohort observation window")
    cohort_duration = cohort_end - warmup
    nominal_travel_time = 30.0 / 1.2
    raw_rows: list[dict[str, Any]] = []
    event_sets: dict[str, list[tuple[InterventionEvent, ...]]] = {}

    for level_index, level in enumerate(settings["levels"]):
        crossing_count = int(level["crossings"])
        request_rate = float(level["rate_per_crossing"])
        arrival_rate = float(level.get("arrival_rate", settings["arrival_rate"]))
        crossing_duration = level.get(
            "crossing_duration", settings["crossing_duration"]
        )
        label = str(level["name"])
        event_sets[label] = []
        config = _config(crossing_count)
        mean_service = float(np.mean(crossing_duration))
        offered_load = request_rate * mean_service
        for replication in range(int(settings["replications"])):
            arrival_seed = seed + replication
            disturbance_seed = seed + 1_000_000 + 10_000 * level_index + replication
            arrivals = poisson_arrival_times(
                arrival_rate, duration, seed=arrival_seed
            )
            requests = poisson_crossing_requests_multi(
                request_rate,
                duration,
                crossing_duration,
                crossing_count,
                seed=disturbance_seed,
                start_time=20.0,
            )
            pair = run_open_paired_rollout(
                config,
                duration,
                arrivals,
                requests,
                record_stride=5,
            )
            if pair.treated.mass_balance_residual != 0:
                raise AssertionError("job mass conservation failed")
            if pair.treated.collision_count or pair.treated.crossing_violation_count:
                raise AssertionError("multi-crossing safety invariant failed")
            treated = analyze_operating_window(
                pair.treated,
                warmup,
                duration,
                requested_arrival_rate=arrival_rate,
            )
            control = analyze_operating_window(
                pair.control,
                warmup,
                duration,
                requested_arrival_rate=arrival_rate,
            )
            control_cohort = analyze_operating_window(
                pair.control,
                warmup,
                cohort_end,
                requested_arrival_rate=arrival_rate,
            )
            complete_events, complete_roots, unresolved_roots = (
                _complete_root_cohorts(
                    pair.treated.events,
                    warmup,
                    cohort_end,
                )
            )
            event_sets[label].append(complete_events)
            primary = tuple(event for event in complete_events if event.is_primary)
            primary_severity = float(sum(event.severity_loss for event in primary))
            exposure = int(control_cohort.completed_jobs)
            burden_per_completion = primary_severity / exposure if exposure else np.nan
            x_value = burden_per_completion / nominal_travel_time
            direct = estimate_direct_branching(complete_events)
            direct_value = (
                float(direct.matrix[0, 0])
                if int(direct.parent_counts[0]) > 0
                else np.nan
            )
            identity = diagnose_chain_forest(complete_events)
            completed_services = sum(
                item.status == "completed"
                and warmup <= item.request_time < cohort_end
                for item in pair.treated.crossing_services
            )
            unserved_services = sum(
                item.status != "completed"
                and warmup <= item.request_time < cohort_end
                for item in pair.treated.crossing_services
            )
            ceiling = (
                crossing_count * cohort_duration / exposure / nominal_travel_time
                if exposure
                else np.nan
            )
            raw_rows.append(
                {
                    "condition": label,
                    "crossing_count": crossing_count,
                    "rate_per_crossing_per_s": request_rate,
                    "mean_service_duration_s": mean_service,
                    "per_crossing_offered_load": offered_load,
                    "single_gate_overloaded": offered_load >= 1.0,
                    "replication": replication,
                    "arrival_seed": arrival_seed,
                    "disturbance_seed": disturbance_seed,
                    "arrival_rate_per_s": arrival_rate,
                    "operating_window_start_s": warmup,
                    "operating_window_end_s": duration,
                    "root_cohort_end_s": cohort_end,
                    "control_cohort_completions": exposure,
                    "complete_primary_cohorts": complete_roots,
                    "unresolved_primary_cohorts": unresolved_roots,
                    "primary_events": len(primary),
                    "primary_severity_s": primary_severity,
                    "primary_burden_s_per_completion": burden_per_completion,
                    "dimensionless_primary_burden": x_value,
                    "analytic_x_ceiling": ceiling,
                    "coverage_status": _status(x_value) if np.isfinite(x_value) else "indeterminate",
                    "direct_branching_in_sample": direct_value,
                    "in_sample_resolvent": (
                        "" if identity.in_sample_resolvent is None else identity.in_sample_resolvent
                    ),
                    "in_sample_measured_progeny": (
                        ""
                        if identity.measured_progeny_per_primary is None
                        else identity.measured_progeny_per_primary
                    ),
                    "chain_identity_residual": (
                        "" if identity.identity_residual is None else identity.identity_residual
                    ),
                    "algebraic_chain_identity": identity.algebraic_identity,
                    "in_sample_validation_status": "descriptive_not_predictive",
                    "treated_completion_rate_per_s": treated.completion_rate,
                    "control_completion_rate_per_s": control.completion_rate,
                    "completion_rate_loss_per_s": (
                        control.completion_rate - treated.completion_rate
                    ),
                    "treated_queue_drift_per_s": treated.drift.slope,
                    "mean_active_crossings": treated.mean_active_crossings,
                    "any_crossing_occupancy_fraction": (
                        treated.crossing_occupancy_fraction
                    ),
                    "completed_crossing_requests": completed_services,
                    "unserved_or_censored_crossing_requests": unserved_services,
                    "fold_claim_status": "not_tested_no_independently_observed_fold",
                    "collision_count": pair.treated.collision_count,
                    "crossing_violation_count": pair.treated.crossing_violation_count,
                    "mass_balance_residual": pair.treated.mass_balance_residual,
                }
            )

    summary_rows: list[dict[str, Any]] = []
    heldout_rows: list[dict[str, Any]] = []
    for level_index, level in enumerate(settings["levels"]):
        crossing_count = int(level["crossings"])
        request_rate = float(level["rate_per_crossing"])
        label = str(level["name"])
        rows = [row for row in raw_rows if row["condition"] == label]
        valid_x = [
            float(row["dimensionless_primary_burden"])
            for row in rows
            if np.isfinite(row["dimensionless_primary_burden"])
        ]
        total_severity = float(sum(row["primary_severity_s"] for row in rows))
        total_exposure = int(sum(row["control_cohort_completions"] for row in rows))
        pooled_x = total_severity / total_exposure / nominal_travel_time
        treated_interval = bootstrap_mean_interval(
            [row["treated_completion_rate_per_s"] for row in rows],
            resamples=int(settings["bootstrap_resamples"]),
            seed=seed + level_index,
        )
        control_interval = bootstrap_mean_interval(
            [row["control_completion_rate_per_s"] for row in rows],
            resamples=int(settings["bootstrap_resamples"]),
            seed=seed + 10_000 + level_index,
        )
        x_interval = bootstrap_mean_interval(
            valid_x,
            resamples=int(settings["bootstrap_resamples"]),
            seed=seed + 20_000 + level_index,
        )
        summary_rows.append(
            {
                "condition": label,
                "crossing_count": crossing_count,
                "rate_per_crossing_per_s": request_rate,
                "arrival_rate_per_s": rows[0]["arrival_rate_per_s"],
                "mean_service_duration_s": rows[0]["mean_service_duration_s"],
                "per_crossing_offered_load": rows[0]["per_crossing_offered_load"],
                "single_gate_overloaded": rows[0]["single_gate_overloaded"],
                "replications": len(rows),
                "dimensionless_burden_pooled": pooled_x,
                "dimensionless_burden_mean": x_interval.estimate,
                "dimensionless_burden_ci_low": x_interval.lower,
                "dimensionless_burden_ci_high": x_interval.upper,
                "analytic_x_ceiling_mean": float(
                    np.mean([row["analytic_x_ceiling"] for row in rows])
                ),
                "coverage_status": _status(pooled_x),
                "treated_completion_rate_mean": treated_interval.estimate,
                "treated_completion_rate_ci_low": treated_interval.lower,
                "treated_completion_rate_ci_high": treated_interval.upper,
                "control_completion_rate_mean": control_interval.estimate,
                "control_completion_rate_ci_low": control_interval.lower,
                "control_completion_rate_ci_high": control_interval.upper,
                "completion_rate_loss_mean": float(
                    np.mean([row["completion_rate_loss_per_s"] for row in rows])
                ),
                "mean_active_crossings": float(
                    np.mean([row["mean_active_crossings"] for row in rows])
                ),
                "algebraic_identity_fraction": float(
                    np.mean([row["algebraic_chain_identity"] for row in rows])
                ),
                "complete_primary_cohorts": int(
                    sum(row["complete_primary_cohorts"] for row in rows)
                ),
                "unresolved_primary_cohorts": int(
                    sum(row["unresolved_primary_cohorts"] for row in rows)
                ),
                "fold_claim_status": "not_tested_no_independently_observed_fold",
            }
        )

        split = max(1, len(event_sets[label]) // 2)
        training_sets = event_sets[label][:split]
        validation_sets = event_sets[label][split:]
        try:
            heldout = validate_branching_event_sets(
                training_sets,
                validation_sets,
            )
            heldout_rows.append(
                {
                    "condition": label,
                    "crossing_count": crossing_count,
                    "rate_per_crossing_per_s": request_rate,
                    "arrival_rate_per_s": rows[0]["arrival_rate_per_s"],
                    "training_replications": len(training_sets),
                    "validation_replications": len(validation_sets),
                    "direct_branching_training": float(
                        heldout.training_branching.matrix[0, 0]
                    ),
                    "predicted_progeny_training": (
                        ""
                        if heldout.predicted_total_progeny_per_primary is None
                        else heldout.predicted_total_progeny_per_primary
                    ),
                    "measured_progeny_held_out": (
                        heldout.measured_total_progeny_per_primary
                    ),
                    "held_out_relative_error": (
                        "" if heldout.relative_error is None else heldout.relative_error
                    ),
                    "validation_primary_events": heldout.validation_primary_events,
                    "validation_causal_events": heldout.validation_causal_events,
                    "validation_scope": "independent_chain_length_transfer_only",
                    "supercriticality_identifiable": False,
                }
            )
        except ValueError as error:
            heldout_rows.append(
                {
                    "condition": label,
                    "crossing_count": crossing_count,
                    "rate_per_crossing_per_s": request_rate,
                    "arrival_rate_per_s": rows[0]["arrival_rate_per_s"],
                    "training_replications": len(training_sets),
                    "validation_replications": len(validation_sets),
                    "direct_branching_training": "",
                    "predicted_progeny_training": "",
                    "measured_progeny_held_out": "",
                    "held_out_relative_error": "",
                    "validation_primary_events": "",
                    "validation_causal_events": "",
                    "validation_scope": "indeterminate",
                    "supercriticality_identifiable": False,
                    "indeterminate_reason": str(error),
                }
            )
    return raw_rows, summary_rows, heldout_rows


def _plot(
    raw_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.5), constrained_layout=True)
    ordered = summary_rows
    condition_index = np.arange(len(ordered), dtype=float)
    condition_labels = [str(row["condition"]) for row in ordered]
    burden = np.asarray([row["dimensionless_burden_pooled"] for row in ordered])
    axes[0].axhspan(0.1, 1.0, color="#fef3c7", alpha=0.7, label="target domain")
    for row in raw_rows:
        index = condition_labels.index(str(row["condition"]))
        axes[0].scatter(
            index,
            row["dimensionless_primary_burden"],
            s=14,
            color="#94a3b8",
            alpha=0.45,
        )
    axes[0].plot(
        condition_index,
        burden,
        "o-",
        color="#175cd3",
        linewidth=2.2,
        label="pooled",
    )
    axes[0].set_xticks(condition_index, condition_labels, rotation=55, ha="right")
    axes[0].set(
        xlabel="Disturbance/load regime",
        ylabel=r"Dimensionless primary burden $x=g/T_0$",
        title="Burden-domain coverage",
        ylim=(0.0, max(1.0, 1.1 * float(np.max(burden)))),
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)

    treated = np.asarray([row["treated_completion_rate_mean"] for row in ordered])
    control = np.asarray([row["control_completion_rate_mean"] for row in ordered])
    throughput_ratio = np.divide(
        treated,
        control,
        out=np.full_like(treated, np.nan),
        where=control > 0.0,
    )
    scatter = axes[1].scatter(
        burden,
        throughput_ratio,
        c=np.asarray([row["arrival_rate_per_s"] for row in ordered]),
        cmap="plasma",
        s=58,
    )
    axes[1].axhline(1.0, color="black", linestyle=":", linewidth=1.2)
    figure.colorbar(scatter, ax=axes[1], label="requested arrival rate (jobs/s)")
    axes[1].set(
        xlabel=r"Dimensionless primary burden $x=g/T_0$",
        ylabel="Treated/control completion-rate ratio",
        title="Stress response, not fold identification",
    )
    axes[1].grid(alpha=0.2)

    valid = [
        row
        for row in heldout_rows
        if row["predicted_progeny_training"] != ""
        and row["measured_progeny_held_out"] != ""
    ]
    if valid:
        predicted = np.asarray(
            [float(row["predicted_progeny_training"]) for row in valid]
        )
        measured = np.asarray(
            [float(row["measured_progeny_held_out"]) for row in valid]
        )
        colors = np.asarray([row["crossing_count"] for row in valid], dtype=float)
        scatter = axes[2].scatter(
            measured, predicted, c=colors, cmap="viridis", s=55
        )
        maximum = 1.05 * float(max(np.max(measured), np.max(predicted)))
        axes[2].plot([0.0, maximum], [0.0, maximum], "k:", linewidth=1.2)
        axes[2].set(xlim=(0.0, maximum), ylim=(0.0, maximum))
        figure.colorbar(scatter, ax=axes[2], label="crossing locations")
    axes[2].set(
        xlabel="Held-out causal events per primary",
        ylabel="Training resolvent prediction",
        title="Held-out transfer (not in-sample identity)",
    )
    axes[2].grid(alpha=0.2)
    figure.suptitle("Multi-crossing burden audit: no fold claim without branch loss")
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
    raw_rows, summary_rows, heldout_rows = _run(settings, arguments.seed)
    _write_csv(arguments.output_dir / "burden_stress_runs.csv", raw_rows)
    _write_csv(arguments.output_dir / "burden_stress_summary.csv", summary_rows)
    _write_csv(arguments.output_dir / "burden_stress_heldout.csv", heldout_rows)
    _plot(
        raw_rows,
        summary_rows,
        heldout_rows,
        arguments.output_dir / "burden_domain_coverage.png",
    )
    manifest = {
        "profile": arguments.profile,
        "seed": arguments.seed,
        "settings": settings,
        "configs": {
            str(level["name"]): asdict(_config(int(level["crossings"])))
            for level in settings["levels"]
        },
        "interpretation": {
            "coverage": (
                "x in [0.1, 1] is the intended informative domain; reaching it "
                "does not by itself validate the saddle-node equation"
            ),
            "gate_load": (
                "per-crossing offered load stays below one; aggregate burden is "
                "not produced by an unstable single crossing-service queue"
            ),
            "branching": (
                "in-sample scalar resolvent agreement is flagged as an algebraic "
                "forest identity; only held-out transfer is scored"
            ),
            "fold": (
                "no fold is claimed unless a stable branch is independently "
                "observed to disappear under a quasi-static load protocol"
            ),
        },
    }
    with (arguments.output_dir / "burden_stress_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"summary": summary_rows, "heldout": heldout_rows}, indent=2))


if __name__ == "__main__":
    main()
