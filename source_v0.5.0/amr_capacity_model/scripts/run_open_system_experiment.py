"""Replicated open-system capacity and load-ramp experiments.

The quick profile is a deterministic smoke study suitable for CI and figure
inspection.  The paper profile increases horizon, replication count, and load
resolution; it is intentionally expensive and should be run before drawing
inferential conclusions.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from amr_capacity.capacity_theory import r_saddle_node_from_burden
from amr_capacity.estimation import (
    diagnose_chain_forest,
    estimate_direct_branching,
    estimate_horizon_branching,
    validate_branching_event_sets,
)
from amr_capacity.open_metrics import (
    OperatingWindowMetrics,
    analyze_load_segments,
    analyze_operating_window,
    bootstrap_mean_interval,
    compare_ramp_branches,
    empirical_capacity_bracket,
    summarize_open_counterfactual,
    summarize_rate_replications,
)
from amr_capacity.open_system import (
    CrossingRequest,
    LoadSegment,
    OpenCorridorConfig,
    piecewise_deterministic_arrival_times,
    poisson_arrival_times,
    poisson_crossing_requests,
    run_open_paired_rollout,
    simulate_open_corridor,
)


def _profile(name: str) -> dict[str, Any]:
    if name == "quick":
        return {
            "rates": (0.35, 0.50, 0.60, 0.65, 0.70, 0.80),
            "duration": 240.0,
            "replications": 3,
            "ramp_slow_dwell": 80.0,
            "ramp_fast_dwell": 30.0,
            "bootstrap_resamples": 1000,
            "fleet_sizes": (2, 3, 5, 8, 12, 20, 30),
            "fleet_replications": 2,
        }
    if name == "paper":
        return {
            "rates": tuple(np.round(np.arange(0.30, 0.851, 0.025), 3)),
            "duration": 1800.0,
            "replications": 30,
            "ramp_slow_dwell": 600.0,
            "ramp_fast_dwell": 120.0,
            "bootstrap_resamples": 10000,
            "fleet_sizes": tuple(range(2, 41, 2)),
            "fleet_replications": 20,
        }
    raise ValueError(f"unknown profile {name!r}")


def _config() -> OpenCorridorConfig:
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
        crossing_x=15.0,
        crossing_half_width=0.4,
    )


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    records = list(rows)
    if not records:
        raise ValueError(f"cannot write an empty table to {path}")
    keys: list[str] = []
    for record in records:
        for key in record:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _finite_or_blank(value: float | None) -> float | str:
    return "" if value is None or not np.isfinite(value) else float(value)


def _complete_root_cohort_events(
    events,
    root_start: float,
    root_end: float,
):
    """Select fully observed trees by primary-root cohort, not event start."""

    causal = tuple(event for event in events if event.is_causal)
    roots = {
        event.event_id: event
        for event in causal
        if event.is_primary and root_start <= event.start_time < root_end
    }
    groups = {key: [] for key in roots}
    for event in causal:
        if event.root_primary_event_id in groups:
            groups[event.root_primary_event_id].append(event)
    complete = tuple(
        event
        for records in groups.values()
        if not any(item.censored for item in records)
        for event in records
    )
    unresolved = sum(any(item.censored for item in records) for records in groups.values())
    return complete, len(groups) - unresolved, unresolved


def _metrics_row(
    condition: str,
    requested_rate: float,
    replication: int,
    metrics: OperatingWindowMetrics,
) -> dict[str, Any]:
    return {
        "condition": condition,
        "requested_arrival_rate": requested_rate,
        "replication": replication,
        "window_start": metrics.start_time,
        "window_end": metrics.end_time,
        "observed_arrival_rate": metrics.observed_arrival_rate,
        "admission_rate": metrics.admission_rate,
        "completion_rate": metrics.completion_rate,
        "mean_queue_length": metrics.mean_queue_length,
        "final_queue_length": metrics.final_queue_length,
        "mean_work_in_process": metrics.mean_work_in_process,
        "mean_available_robots": _finite_or_blank(metrics.mean_available_robots),
        "mean_returning_robots": _finite_or_blank(metrics.mean_returning_robots),
        "mean_fleet_utilization": _finite_or_blank(metrics.mean_fleet_utilization),
        "mean_speed": _finite_or_blank(metrics.mean_speed),
        "intervention_time_rate": metrics.intervention_time_rate,
        "crossing_occupancy_fraction": metrics.crossing_occupancy_fraction,
        "mean_active_crossings": metrics.mean_active_crossings,
        "mean_completed_travel_time": _finite_or_blank(
            metrics.mean_completed_travel_time
        ),
        "mean_completed_sojourn_time": _finite_or_blank(
            metrics.mean_completed_sojourn_time
        ),
        "completed_jobs": metrics.completed_jobs,
        "queue_drift": metrics.drift.slope,
        "queue_drift_hac_se": metrics.drift.standard_error,
        "queue_drift_ci_low": metrics.drift.lower_confidence,
        "queue_drift_ci_high": metrics.drift.upper_confidence,
        "hac_lag": metrics.drift.hac_lag,
        "empty_time_fraction": metrics.regeneration.empty_time_fraction,
        "zero_return_count": metrics.regeneration.zero_return_count,
        "terminal_busy_duration": metrics.regeneration.terminal_busy_duration,
        "longest_busy_duration": (
            metrics.regeneration.longest_observed_busy_duration
        ),
        "stability_evidence": metrics.stability,
    }


def _event_diagnostics(
    pair,
    start_time: float,
    end_time: float,
    control_metrics: OperatingWindowMetrics,
) -> dict[str, Any]:
    events = tuple(
        event
        for event in pair.treated.events
        if start_time <= event.start_time < end_time
    )
    direct = estimate_direct_branching(events)
    chain_identity = diagnose_chain_forest(events)
    horizon_2 = estimate_horizon_branching(events, 2.0)
    horizon_5 = estimate_horizon_branching(events, 5.0)
    horizon_10 = estimate_horizon_branching(events, 10.0)
    primaries = tuple(event for event in events if event.is_primary)
    causal = tuple(event for event in events if event.is_causal)
    exposure = max(1, control_metrics.completed_jobs)
    primary_burden = sum(item.severity_loss for item in primaries) / exposure
    nominal_travel = (
        control_metrics.mean_completed_travel_time
        if control_metrics.mean_completed_travel_time is not None
        else pair.control.config.corridor_length / pair.control.config.desired_speed
    )
    dimensionless_burden = primary_burden / nominal_travel
    parent_count = int(direct.parent_counts[0])
    direct_value = float(direct.matrix[0, 0]) if parent_count else np.nan
    fold_reference = float(r_saddle_node_from_burden(dimensionless_burden))
    completed_services = tuple(
        item
        for item in pair.treated.crossing_services
        if item.status == "completed" and start_time <= item.request_time < end_time
    )
    waits = [item.wait_time for item in completed_services if item.wait_time is not None]
    return {
        "primary_events": len(primaries),
        "causal_events": len(causal),
        "direct_parent_events": parent_count,
        "direct_branching": _finite_or_blank(direct_value),
        "direct_spectral_radius": _finite_or_blank(direct_value),
        "in_sample_resolvent": _finite_or_blank(
            chain_identity.in_sample_resolvent
        ),
        "in_sample_measured_progeny": _finite_or_blank(
            chain_identity.measured_progeny_per_primary
        ),
        "chain_identity_residual": _finite_or_blank(
            chain_identity.identity_residual
        ),
        "algebraic_chain_identity": chain_identity.algebraic_identity,
        "rooted_single_parent_forest": (
            chain_identity.rooted_single_parent_forest
        ),
        "in_sample_validation_status": "descriptive_not_predictive",
        "supercriticality_identifiable_in_this_layout": False,
        "horizon_branching_2s": float(horizon_2.matrix[0, 0]),
        "horizon_branching_5s": float(horizon_5.matrix[0, 0]),
        "horizon_branching_10s": float(horizon_10.matrix[0, 0]),
        "primary_burden_per_control_completion_s": primary_burden,
        "dimensionless_primary_burden": dimensionless_burden,
        "fold_curve_reference": fold_reference,
        "fold_test_status": "not_tested_no_independently_observed_fold",
        "completed_crossing_requests": len(completed_services),
        "mean_crossing_wait": _finite_or_blank(
            float(np.mean(waits)) if waits else None
        ),
    }


def _run_rate_sweep(
    config: OpenCorridorConfig,
    settings: dict[str, Any],
    seed: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[float, OperatingWindowMetrics]],
    dict[str, Any],
]:
    duration = float(settings["duration"])
    warmup = duration / 2.0
    replications = int(settings["replications"])
    raw_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    grouped: dict[str, dict[float, list[OperatingWindowMetrics]]] = {
        "crossings": {},
        "no_crossings": {},
    }
    event_sets_by_rate: dict[float, list[tuple[Any, ...]]] = {
        rate: [] for rate in settings["rates"]
    }
    for rate_index, rate in enumerate(settings["rates"]):
        grouped["crossings"][rate] = []
        grouped["no_crossings"][rate] = []
        for replication in range(replications):
            arrival_seed = seed + 10_000 * rate_index + replication
            # The same disturbance seed is reused across loads for a given
            # replication to reduce nuisance variation between rate points.
            crossing_seed = seed + 1_000_000 + replication
            arrivals = poisson_arrival_times(
                rate, duration, seed=arrival_seed
            )
            requests = poisson_crossing_requests(
                0.025,
                duration,
                (2.0, 4.0),
                seed=crossing_seed,
                start_time=20.0,
            )
            pair = run_open_paired_rollout(
                config,
                duration,
                arrivals,
                requests,
                record_stride=5,
            )
            treated = analyze_operating_window(
                pair.treated,
                warmup,
                duration,
                requested_arrival_rate=rate,
            )
            control = analyze_operating_window(
                pair.control,
                warmup,
                duration,
                requested_arrival_rate=rate,
            )
            grouped["crossings"][rate].append(treated)
            grouped["no_crossings"][rate].append(control)
            raw_rows.append(
                _metrics_row("crossings", rate, replication, treated)
            )
            raw_rows.append(
                _metrics_row("no_crossings", rate, replication, control)
            )
            counterfactual = summarize_open_counterfactual(pair)
            row = {
                "requested_arrival_rate": rate,
                "replication": replication,
                "arrival_seed": arrival_seed,
                "crossing_seed": crossing_seed,
                "arrivals": pair.treated.total_arrivals,
                "crossing_requests": len(requests),
                "collision_count": pair.treated.collision_count,
                "crossing_violation_count": pair.treated.crossing_violation_count,
                "mass_balance_residual": pair.treated.mass_balance_residual,
                "fleet_balance_residual": (
                    "" if pair.treated.fleet_balance_residual is None
                    else pair.treated.fleet_balance_residual
                ),
                **asdict(counterfactual),
                **_event_diagnostics(pair, warmup, duration, control),
            }
            event_rows.append(row)
            cohort_events, _, _ = _complete_root_cohort_events(
                pair.treated.events,
                warmup,
                duration - min(60.0, duration / 4.0),
            )
            event_sets_by_rate[rate].append(cohort_events)

    aggregate_rows: list[dict[str, Any]] = []
    summaries_by_condition: dict[str, list[Any]] = {}
    for condition, by_rate in grouped.items():
        summaries = []
        for rate_index, rate in enumerate(settings["rates"]):
            summary = summarize_rate_replications(
                rate,
                by_rate[rate],
                confidence=0.95,
                resamples=settings["bootstrap_resamples"],
                seed=seed + rate_index,
            )
            summaries.append(summary)
            aggregate_rows.append(
                {
                    "condition": condition,
                    "requested_arrival_rate": rate,
                    "replications": summary.replications,
                    "completion_rate_mean": summary.completion_rate.estimate,
                    "completion_rate_ci_low": summary.completion_rate.lower,
                    "completion_rate_ci_high": summary.completion_rate.upper,
                    "queue_drift_mean": summary.queue_drift.estimate,
                    "queue_drift_ci_low": summary.queue_drift.lower,
                    "queue_drift_ci_high": summary.queue_drift.upper,
                    "mean_queue_length": summary.mean_queue_length.estimate,
                    "mean_queue_ci_low": summary.mean_queue_length.lower,
                    "mean_queue_ci_high": summary.mean_queue_length.upper,
                    "no_growth_fraction": summary.no_growth_fraction,
                    "growth_fraction": summary.growth_fraction,
                }
            )
        summaries_by_condition[condition] = summaries

    brackets = {
        condition: asdict(
            empirical_capacity_bracket(summaries, decision_fraction=2.0 / 3.0)
        )
        for condition, summaries in summaries_by_condition.items()
    }
    heldout_rows: list[dict[str, Any]] = []
    for rate in settings["rates"]:
        event_sets = event_sets_by_rate[rate]
        split = max(1, len(event_sets) // 2)
        try:
            validation = validate_branching_event_sets(
                event_sets[:split], event_sets[split:]
            )
            heldout_rows.append(
                {
                    "requested_arrival_rate": rate,
                    "training_replications": split,
                    "validation_replications": len(event_sets) - split,
                    "direct_branching_training": float(
                        validation.training_branching.matrix[0, 0]
                    ),
                    "predicted_progeny_training": _finite_or_blank(
                        validation.predicted_total_progeny_per_primary
                    ),
                    "measured_progeny_held_out": (
                        validation.measured_total_progeny_per_primary
                    ),
                    "held_out_relative_error": _finite_or_blank(
                        validation.relative_error
                    ),
                    "validation_primary_events": validation.validation_primary_events,
                    "validation_causal_events": validation.validation_causal_events,
                    "validation_scope": "independent_chain_length_transfer_only",
                    "supercriticality_identifiable": False,
                }
            )
        except ValueError as error:
            heldout_rows.append(
                {
                    "requested_arrival_rate": rate,
                    "training_replications": split,
                    "validation_replications": len(event_sets) - split,
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
    representative = {
        condition: {
            rate: records[0] for rate, records in by_rate.items()
        }
        for condition, by_rate in grouped.items()
    }
    return raw_rows, aggregate_rows, representative, {
        "summaries": summaries_by_condition,
        "brackets": brackets,
        "event_rows": event_rows,
        "heldout_rows": heldout_rows,
    }


def _plot_rate_sweep(
    aggregate_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    styles = {
        "crossings": ("#b42318", "o", "crossing requests"),
        "no_crossings": ("#175cd3", "s", "no-crossing control"),
    }
    for condition, (color, marker, label) in styles.items():
        rows = sorted(
            (row for row in aggregate_rows if row["condition"] == condition),
            key=lambda row: row["requested_arrival_rate"],
        )
        rate = np.asarray([row["requested_arrival_rate"] for row in rows])
        completion = np.asarray([row["completion_rate_mean"] for row in rows])
        completion_error = np.vstack(
            (
                completion - np.asarray([row["completion_rate_ci_low"] for row in rows]),
                np.asarray([row["completion_rate_ci_high"] for row in rows]) - completion,
            )
        )
        drift = np.asarray([row["queue_drift_mean"] for row in rows])
        drift_error = np.vstack(
            (
                drift - np.asarray([row["queue_drift_ci_low"] for row in rows]),
                np.asarray([row["queue_drift_ci_high"] for row in rows]) - drift,
            )
        )
        queue = np.asarray([row["mean_queue_length"] for row in rows])
        axes[0, 0].errorbar(
            rate, completion, yerr=completion_error, color=color, marker=marker,
            capsize=3, label=label,
        )
        axes[0, 1].errorbar(
            rate, drift, yerr=drift_error, color=color, marker=marker, capsize=3,
            label=label,
        )
        axes[1, 0].plot(rate, queue, color=color, marker=marker, label=label)

    all_rates = np.asarray(
        sorted({row["requested_arrival_rate"] for row in aggregate_rows})
    )
    axes[0, 0].plot(all_rates, all_rates, "k--", linewidth=1, label="flow balance")
    axes[0, 0].set(xlabel="requested arrival rate [jobs/s]", ylabel="completion rate [jobs/s]")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].axhline(0.0, color="black", linewidth=1)
    axes[0, 1].set(xlabel="requested arrival rate [jobs/s]", ylabel="queue drift [jobs/s]")
    axes[1, 0].set(xlabel="requested arrival rate [jobs/s]", ylabel="mean upstream queue [jobs]")
    axes[1, 0].set_yscale("symlog", linthresh=0.1)

    event_rates = sorted({row["requested_arrival_rate"] for row in event_rows})
    direct_mean = []
    horizon_mean = []
    latency_mean = []
    for rate in event_rates:
        rows = [row for row in event_rows if row["requested_arrival_rate"] == rate]
        direct = [
            float(row["direct_branching"])
            for row in rows
            if row["direct_branching"] != ""
        ]
        direct_mean.append(float(np.mean(direct)) if direct else np.nan)
        horizon_mean.append(float(np.mean([row["horizon_branching_10s"] for row in rows])))
        latency = [
            row["mean_paired_sojourn_increase"]
            for row in rows
            if row["mean_paired_sojourn_increase"] is not None
        ]
        latency_mean.append(float(np.mean(latency)) if latency else np.nan)
    axes[1, 1].plot(event_rates, direct_mean, "o-", color="#067647", label="direct offspring")
    axes[1, 1].plot(event_rates, horizon_mean, "^--", color="#b54708", label="10 s horizon ablation")
    axes[1, 1].axhline(1.0, color="black", linewidth=1)
    axes[1, 1].set(xlabel="requested arrival rate [jobs/s]", ylabel="estimated offspring count")
    secondary = axes[1, 1].twinx()
    secondary.plot(event_rates, latency_mean, "d:", color="#7a5af8", label="paired sojourn increase")
    secondary.set_ylabel("paired sojourn increase [s]", color="#7a5af8")
    handles, labels = axes[1, 1].get_legend_handles_labels()
    handles2, labels2 = secondary.get_legend_handles_labels()
    axes[1, 1].legend(handles + handles2, labels + labels2, fontsize=8)
    fig.suptitle("Open-system capacity evidence (95% bootstrap intervals)")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_fold_diagnostic(event_rows: list[dict[str, Any]], output: Path) -> None:
    valid = [
        row
        for row in event_rows
        if row["direct_branching"] != ""
        and int(row["primary_events"]) > 0
        and float(row["dimensionless_primary_burden"]) > 0.0
    ]
    fig, axis = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    if valid:
        burden = np.asarray([row["dimensionless_primary_burden"] for row in valid])
        direct = np.asarray([float(row["direct_branching"]) for row in valid])
        rate = np.asarray([row["requested_arrival_rate"] for row in valid])
        scatter = axis.scatter(burden, direct, c=rate, cmap="viridis", s=45, alpha=0.85)
        maximum = max(1.0, float(np.max(burden)) * 1.15)
        curve_x = np.linspace(0.0, maximum, 400)
        axis.plot(
            curve_x,
            r_saddle_node_from_burden(curve_x),
            color="black",
            linewidth=2,
            label=r"fold prediction $R_{sn}(g/T_0)$",
        )
        fig.colorbar(scatter, ax=axis, label="requested arrival rate [jobs/s]")
        axis.axvspan(0.1, 1.0, color="#fef3c7", alpha=0.45, label="target coverage")
        axis.text(
            0.98,
            0.05,
            rf"observed positive-event range: $x\leq {np.max(burden):.5f}$",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
        )
        axis.legend()
    axis.set(
        xlabel=r"dimensionless primary burden $g/T_0$",
        ylabel=r"direct offspring estimate $\hat R$",
        title="Burden-domain audit (no observed fold; not a validation)",
    )
    axis.set_ylim(bottom=0.0)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _run_finite_fleet_sweep(
    base_config: OpenCorridorConfig,
    settings: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Measure the finite-fleet truncation under a saturated task source."""

    duration = float(settings["duration"])
    warmup = duration / 2.0
    return_time = 8.0
    arrival_rate = 1.2
    arrivals = piecewise_deterministic_arrival_times(
        (LoadSegment(0.0, duration, arrival_rate, "saturated"),)
    )
    raw: list[dict[str, Any]] = []
    for replication in range(int(settings["fleet_replications"])):
        requests = poisson_crossing_requests(
            0.025,
            duration,
            (2.0, 4.0),
            seed=seed + 4_000_000 + replication,
            start_time=20.0,
        )
        for fleet_size in settings["fleet_sizes"]:
            config = replace(
                base_config,
                fleet_size=int(fleet_size),
                empty_return_time=return_time,
            )
            pair = run_open_paired_rollout(
                config,
                duration,
                arrivals,
                requests,
                record_stride=5,
            )
            for condition, result in (
                ("crossings", pair.treated),
                ("no_crossings", pair.control),
            ):
                metrics = analyze_operating_window(
                    result,
                    warmup,
                    duration,
                    requested_arrival_rate=arrival_rate,
                )
                if result.fleet_balance_residual != 0:
                    raise AssertionError("finite-fleet experiment lost a robot")
                raw.append(
                    {
                        "condition": condition,
                        "fleet_size": fleet_size,
                        "replication": replication,
                        "empty_return_time": return_time,
                        "requested_arrival_rate": arrival_rate,
                        "completion_rate": metrics.completion_rate,
                        "mean_queue_length": metrics.mean_queue_length,
                        "mean_work_in_process": metrics.mean_work_in_process,
                        "mean_available_robots": metrics.mean_available_robots,
                        "mean_returning_robots": metrics.mean_returning_robots,
                        "mean_fleet_utilization": metrics.mean_fleet_utilization,
                        "queue_drift": metrics.drift.slope,
                        "stability_evidence": metrics.stability,
                        "mass_balance_residual": result.mass_balance_residual,
                        "fleet_balance_residual": result.fleet_balance_residual,
                        "collision_count": result.collision_count,
                        "crossing_violation_count": result.crossing_violation_count,
                    }
                )

    nominal_travel_time = (
        base_config.corridor_length / base_config.desired_speed
        + base_config.desired_speed / (2.0 * base_config.acceleration)
    )
    summary_rows: list[dict[str, Any]] = []
    for condition in ("crossings", "no_crossings"):
        for fleet_size in settings["fleet_sizes"]:
            records = [
                row
                for row in raw
                if row["condition"] == condition and row["fleet_size"] == fleet_size
            ]
            completion = bootstrap_mean_interval(
                [row["completion_rate"] for row in records],
                resamples=settings["bootstrap_resamples"],
                seed=seed + int(fleet_size),
            )
            utilization = bootstrap_mean_interval(
                [row["mean_fleet_utilization"] for row in records],
                resamples=settings["bootstrap_resamples"],
                seed=seed + 10_000 + int(fleet_size),
            )
            summary_rows.append(
                {
                    "condition": condition,
                    "fleet_size": fleet_size,
                    "replications": len(records),
                    "completion_rate_mean": completion.estimate,
                    "completion_rate_ci_low": completion.lower,
                    "completion_rate_ci_high": completion.upper,
                    "fleet_utilization_mean": utilization.estimate,
                    "fleet_utilization_ci_low": utilization.lower,
                    "fleet_utilization_ci_high": utilization.upper,
                    "nominal_circulation_ceiling": (
                        fleet_size / (nominal_travel_time + return_time)
                    ),
                }
            )
    return raw, summary_rows


def _plot_finite_fleet(summary_rows: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    styles = {
        "crossings": ("#b42318", "o", "crossing requests"),
        "no_crossings": ("#175cd3", "s", "no-crossing control"),
    }
    for condition, (color, marker, label) in styles.items():
        rows = sorted(
            (row for row in summary_rows if row["condition"] == condition),
            key=lambda row: row["fleet_size"],
        )
        size = np.asarray([row["fleet_size"] for row in rows])
        completion = np.asarray([row["completion_rate_mean"] for row in rows])
        error = np.vstack(
            (
                completion - np.asarray([row["completion_rate_ci_low"] for row in rows]),
                np.asarray([row["completion_rate_ci_high"] for row in rows]) - completion,
            )
        )
        utilization = np.asarray([row["fleet_utilization_mean"] for row in rows])
        axes[0].errorbar(
            size, completion, yerr=error, marker=marker, color=color,
            capsize=3, label=label,
        )
        axes[1].plot(size, utilization, marker=marker, color=color, label=label)
    reference = sorted(
        (row for row in summary_rows if row["condition"] == "no_crossings"),
        key=lambda row: row["fleet_size"],
    )
    axes[0].plot(
        [row["fleet_size"] for row in reference],
        [row["nominal_circulation_ceiling"] for row in reference],
        "k--",
        label=r"$N/(T_0+T_{return})$",
    )
    axes[0].set(
        xlabel="physical fleet size N",
        ylabel="saturated completion rate [jobs/s]",
        title="Finite-fleet truncation",
    )
    axes[1].set(
        xlabel="physical fleet size N",
        ylabel="mean active fraction",
        title="Bottleneck occupancy",
    )
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _make_ramp_segments(dwell: float) -> tuple[LoadSegment, ...]:
    rates = (0.35, 0.50, 0.65, 0.80, 0.65, 0.50, 0.35)
    peak = 3
    segments = []
    for index, rate in enumerate(rates):
        direction = "up" if index <= peak else "down"
        segments.append(
            LoadSegment(
                start_time=index * dwell,
                end_time=(index + 1) * dwell,
                arrival_rate=rate,
                label=f"{direction}-{rate:.2f}",
            )
        )
    return tuple(segments)


def _run_ramps(
    config: OpenCorridorConfig,
    settings: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    for ramp_index, ramp_name in enumerate(("fast", "slow")):
        dwell = float(settings[f"ramp_{ramp_name}_dwell"])
        segments = _make_ramp_segments(dwell)
        duration = segments[-1].end_time
        # A controlled load ramp uses regular task releases so the horizontal
        # axis is exact.  Stochastic arrivals remain the main replicated test.
        arrivals = piecewise_deterministic_arrival_times(segments)
        # Reuse the same local disturbance realization for equal-rate up and
        # down segments.  The empty-start control below replays it exactly.
        local_requests: dict[float, tuple[CrossingRequest, ...]] = {}
        for rate_index, rate in enumerate(sorted({item.arrival_rate for item in segments})):
            local_horizon = max(config.dt * 2.0, dwell - 5.0)
            local_start = min(5.0, local_horizon / 2.0)
            local_requests[rate] = poisson_crossing_requests(
                0.025,
                local_horizon,
                (2.0, 4.0),
                seed=seed + 3_000_000 + 100 * ramp_index + rate_index,
                start_time=local_start,
            )
        global_requests: list[CrossingRequest] = []
        source_id = 0
        for segment in segments:
            for request in local_requests[segment.arrival_rate]:
                global_requests.append(
                    CrossingRequest(
                        request_time=segment.start_time + request.request_time,
                        duration=request.duration,
                        source_id=source_id,
                    )
                )
                source_id += 1
        requests = tuple(global_requests)
        result = simulate_open_corridor(
            config,
            duration,
            arrivals,
            crossing_requests=requests,
            record_stride=5,
        )
        segment_metrics = analyze_load_segments(
            result, segments, discard_fraction=0.5
        )
        reset_metrics: dict[float, OperatingWindowMetrics] = {}
        for rate in sorted(local_requests):
            reset_segment = (LoadSegment(0.0, dwell, rate, f"reset-{rate:.2f}"),)
            reset_arrivals = piecewise_deterministic_arrival_times(reset_segment)
            reset_result = simulate_open_corridor(
                config,
                dwell,
                reset_arrivals,
                crossing_requests=local_requests[rate],
                record_stride=5,
            )
            reset_metrics[rate] = analyze_operating_window(
                reset_result,
                dwell / 2.0,
                dwell,
                requested_arrival_rate=rate,
            )
        up = tuple(item for item in segment_metrics if item.segment.label.startswith("up"))
        down = tuple(item for item in segment_metrics if item.segment.label.startswith("down"))
        comparisons = compare_ramp_branches(up, down)
        comparison_by_rate = {item.arrival_rate: item for item in comparisons}
        for item in segment_metrics:
            direction = "up" if item.segment.label.startswith("up") else "down"
            reset = reset_metrics[item.segment.arrival_rate]
            reset_value = reset.mean_queue_length
            comparison = comparison_by_rate.get(item.segment.arrival_rate)
            rows.append(
                {
                    "ramp": ramp_name,
                    "direction": direction,
                    "label": item.segment.label,
                    "requested_arrival_rate": item.segment.arrival_rate,
                    "segment_start": item.segment.start_time,
                    "segment_end": item.segment.end_time,
                    "retained_start": item.retained_start_time,
                    "observed_arrival_rate": item.metrics.observed_arrival_rate,
                    "completion_rate": item.metrics.completion_rate,
                    "mean_queue_length": item.metrics.mean_queue_length,
                    "queue_drift": item.metrics.drift.slope,
                    "stability_evidence": item.metrics.stability,
                    "empty_time_fraction": item.metrics.regeneration.empty_time_fraction,
                    "terminal_busy_duration": item.metrics.regeneration.terminal_busy_duration,
                    "matched_reset_mean_queue": reset_value,
                    "matched_reset_completion_rate": reset.completion_rate,
                    "matched_reset_queue_drift": reset.drift.slope,
                    "matched_reset_stability_evidence": reset.stability,
                    "excess_over_matched_reset": (
                        item.metrics.mean_queue_length - reset_value
                    ),
                    "down_minus_up_queue": (
                        _finite_or_blank(comparison.queue_gap)
                        if comparison is not None
                        else ""
                    ),
                    "collision_count": result.collision_count,
                    "crossing_violation_count": result.crossing_violation_count,
                    "mass_balance_residual": result.mass_balance_residual,
                }
            )
        results[ramp_name] = {
            "result": result,
            "segments": segment_metrics,
            "comparisons": comparisons,
            "reset_metrics": reset_metrics,
        }
    return rows, results


def _plot_ramps(ramp_results: dict[str, Any], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
    colors = {"fast": "#b42318", "slow": "#175cd3"}
    for column, name in enumerate(("fast", "slow")):
        result = ramp_results[name]["result"]
        axes[column].plot(
            result.trajectory.time,
            result.trajectory.queue_length,
            color=colors[name],
            linewidth=1.2,
        )
        for item in ramp_results[name]["segments"]:
            axes[column].axvline(item.segment.start_time, color="0.75", linewidth=0.6)
        axes[column].set(
            xlabel="time [s]",
            ylabel="upstream queue [jobs]",
            title=f"{name.capitalize()} ramp",
        )

        up = [item for item in ramp_results[name]["segments"] if item.segment.label.startswith("up")]
        down = [item for item in ramp_results[name]["segments"] if item.segment.label.startswith("down")]
        axes[2].plot(
            [item.segment.arrival_rate for item in up],
            [item.metrics.mean_queue_length for item in up],
            marker="o",
            color=colors[name],
            linestyle="-",
            label=f"{name} up",
        )
        axes[2].plot(
            [item.segment.arrival_rate for item in down],
            [item.metrics.mean_queue_length for item in down],
            marker="s",
            color=colors[name],
            linestyle="--",
            label=f"{name} down",
        )
        reset = ramp_results[name]["reset_metrics"]
        reset_rates = sorted(reset)
        axes[2].plot(
            reset_rates,
            [reset[rate].mean_queue_length for rate in reset_rates],
            marker="x",
            color=colors[name],
            linestyle=":",
            label=f"{name} matched empty-start",
        )
    axes[2].set(
        xlabel="requested arrival rate [jobs/s]",
        ylabel="mean upstream queue [jobs]",
        title="Matched state-memory test",
    )
    axes[2].set_yscale("symlog", linthresh=0.1)
    axes[2].legend(fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("quick", "paper"), default="quick")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs",
    )
    arguments = parser.parse_args()
    settings = _profile(arguments.profile)
    config = _config()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows, aggregate_rows, _, sweep = _run_rate_sweep(
        config, settings, arguments.seed
    )
    event_rows = sweep["event_rows"]
    _write_csv(arguments.output_dir / "open_capacity_runs.csv", raw_rows)
    _write_csv(arguments.output_dir / "open_capacity_summary.csv", aggregate_rows)
    _write_csv(arguments.output_dir / "open_attribution_diagnostics.csv", event_rows)
    _write_csv(
        arguments.output_dir / "open_heldout_branching.csv",
        sweep["heldout_rows"],
    )
    _plot_rate_sweep(
        aggregate_rows, event_rows, arguments.output_dir / "open_capacity_validation.png"
    )
    _plot_fold_diagnostic(
        event_rows, arguments.output_dir / "open_fold_diagnostic.png"
    )

    fleet_rows, fleet_summary = _run_finite_fleet_sweep(
        config, settings, arguments.seed
    )
    _write_csv(arguments.output_dir / "open_finite_fleet_runs.csv", fleet_rows)
    _write_csv(
        arguments.output_dir / "open_finite_fleet_summary.csv", fleet_summary
    )
    _plot_finite_fleet(
        fleet_summary, arguments.output_dir / "open_finite_fleet.png"
    )

    ramp_rows, ramp_results = _run_ramps(
        config, settings, arguments.seed
    )
    _write_csv(arguments.output_dir / "open_ramp_memory.csv", ramp_rows)
    _plot_ramps(ramp_results, arguments.output_dir / "open_ramp_memory.png")

    manifest = {
        "profile": arguments.profile,
        "seed": arguments.seed,
        "config": asdict(config),
        "settings": settings,
        "capacity_brackets": sweep["brackets"],
        "interpretation": {
            "finite_run_labels": (
                "no_growth_detected and growth_detected are evidence labels, "
                "not proofs of positive recurrence or instability"
            ),
            "ramp_gap": (
                "each equal-load segment reuses its task and disturbance pattern; "
                "up/down separation relative to the matched empty-start replay is "
                "dynamic memory, not evidence of static bistability"
            ),
            "fold_diagnostic": (
                "the parameter-free curve is shown only as a reference; the "
                "single-gate data neither cover its varying domain nor contain "
                "an independently detected fold"
            ),
            "finite_fleet": (
                "the physical-fleet sweep enforces active + returning + available "
                "= N at every integration step"
            ),
            "open_branching": (
                "the scalar in-sample resolvent is flagged as a chain-forest "
                "identity; open_heldout_branching.csv scores only independent "
                "replications and does not identify supercriticality"
            ),
        },
    }
    with (arguments.output_dir / "open_experiment_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
