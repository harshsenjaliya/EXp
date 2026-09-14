#!/usr/bin/env python3
"""Known-law validation of the exposure-aware censored branching estimator."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from amr_capacity import (
    bootstrap_exposure_aware_branching,
    pool_direct_branching_events,
    pool_exposure_aware_branching_events,
    select_complete_root_cohorts,
    simulate_censored_branching_forests,
    spectral_radius,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs"


def _profile(name: str) -> dict[str, Any]:
    if name == "quick":
        return {
            "root_count": 800,
            "observation_horizon": 4.0,
            "bootstrap_resamples": 400,
            "confidence_level": 0.95,
        }
    if name == "paper":
        return {
            "root_count": 5_000,
            "observation_horizon": 5.0,
            "bootstrap_resamples": 4_000,
            "confidence_level": 0.95,
        }
    raise ValueError(f"unknown profile {name!r}")


def _cases() -> tuple[dict[str, Any], ...]:
    return (
        {
            "name": "scalar_subcritical",
            "matrix": np.array([[0.65]]),
            "recovery": np.array([1.0]),
            "primary": np.array([1.0]),
        },
        {
            "name": "scalar_nearcritical",
            "matrix": np.array([[0.95]]),
            "recovery": np.array([1.0]),
            "primary": np.array([1.0]),
        },
        {
            "name": "scalar_supercritical",
            "matrix": np.array([[1.35]]),
            "recovery": np.array([1.0]),
            "primary": np.array([1.0]),
        },
        {
            "name": "nonnormal_subcritical",
            "matrix": np.array([[0.35, 1.10], [0.00, 0.75]]),
            "recovery": np.array([1.0, 0.8]),
            "primary": np.array([0.5, 0.5]),
        },
        {
            "name": "multitype_supercritical",
            "matrix": np.array([[0.70, 0.70], [0.20, 0.85]]),
            "recovery": np.array([1.0, 0.8]),
            "primary": np.array([0.5, 0.5]),
        },
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


def _complete_tree_event_sets(
    forests: tuple[tuple[Any, ...], ...],
) -> tuple[tuple[Any, ...], ...]:
    complete: list[tuple[Any, ...]] = []
    for forest in forests:
        selection = select_complete_root_cohorts(forest, 0.0, 0.5)
        if selection.events:
            complete.append(selection.events)
    return tuple(complete)


def _run(
    settings: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    for case_index, case in enumerate(_cases()):
        matrix = np.asarray(case["matrix"], dtype=float)
        type_count = matrix.shape[0]
        case_seed = seed + 100_000 * case_index
        forests = simulate_censored_branching_forests(
            matrix,
            case["recovery"],
            float(settings["observation_horizon"]),
            int(settings["root_count"]),
            primary_distribution=case["primary"],
            seed=case_seed,
        )
        exposure = pool_exposure_aware_branching_events(
            forests, number_of_types=type_count
        )
        if exposure.spectral_radius is None:
            raise AssertionError("known-law point estimate must be identifiable")
        interval = bootstrap_exposure_aware_branching(
            forests,
            number_of_types=type_count,
            confidence_level=float(settings["confidence_level"]),
            resamples=int(settings["bootstrap_resamples"]),
            seed=case_seed + 50_000,
        )
        complete_sets = _complete_tree_event_sets(forests)
        if not complete_sets:
            raise AssertionError("benchmark produced no complete trees")
        complete = pool_direct_branching_events(
            complete_sets, number_of_types=type_count
        )

        true_rho = float(spectral_radius(matrix))
        estimated_rho = float(exposure.spectral_radius)
        total_parents = int(np.sum(exposure.observed_parent_counts))
        total_censored = int(np.sum(exposure.censored_parent_counts))
        row = {
            "case": case["name"],
            "type_count": type_count,
            "root_count": int(settings["root_count"]),
            "complete_roots": len(complete_sets),
            "unresolved_roots": int(settings["root_count"]) - len(complete_sets),
            "complete_root_fraction": len(complete_sets) / int(settings["root_count"]),
            "observed_parent_events": total_parents,
            "completed_parent_events": int(
                np.sum(exposure.completed_parent_counts)
            ),
            "censored_parent_events": total_censored,
            "parent_censoring_fraction": total_censored / total_parents,
            "true_spectral_radius": true_rho,
            "exposure_spectral_radius": estimated_rho,
            "exposure_rho_ci_lower": interval.spectral_radius_lower,
            "exposure_rho_ci_upper": interval.spectral_radius_upper,
            "complete_tree_spectral_radius": float(complete.spectral_radius),
            "exposure_rho_absolute_error": abs(estimated_rho - true_rho),
            "complete_tree_rho_absolute_error": abs(
                float(complete.spectral_radius) - true_rho
            ),
            "matrix_mean_absolute_error": float(
                np.mean(np.abs(exposure.matrix - matrix))
            ),
            "bootstrap_valid_resamples": interval.valid_resamples,
            "bootstrap_requested_resamples": interval.requested_resamples,
            "true_regime": "subcritical" if true_rho < 1.0 else "supercritical",
            "exposure_regime": (
                "subcritical" if estimated_rho < 1.0 else "supercritical"
            ),
            "complete_tree_regime": (
                "subcritical"
                if complete.spectral_radius < 1.0
                else "supercritical"
            ),
        }
        summary.append(row)

        for parent_type in range(type_count):
            for child_type in range(type_count):
                entries.append(
                    {
                        "case": case["name"],
                        "parent_type": parent_type,
                        "child_type": child_type,
                        "true_B": float(matrix[parent_type, child_type]),
                        "exposure_B": float(
                            exposure.matrix[parent_type, child_type]
                        ),
                        "exposure_B_ci_lower": float(
                            interval.matrix_lower[parent_type, child_type]
                        ),
                        "exposure_B_ci_upper": float(
                            interval.matrix_upper[parent_type, child_type]
                        ),
                        "complete_tree_B": float(
                            complete.matrix[parent_type, child_type]
                        ),
                        "parent_exposure_time": float(
                            exposure.exposure_time[parent_type]
                        ),
                        "completed_parents": int(
                            exposure.completed_parent_counts[parent_type]
                        ),
                        "censored_parents": int(
                            exposure.censored_parent_counts[parent_type]
                        ),
                        "direct_children": int(
                            exposure.direct_child_counts[
                                parent_type, child_type
                            ]
                        ),
                    }
                )
        print(f"completed {case['name']}", flush=True)
    return summary, entries


def _plot(rows: list[dict[str, Any]], output: Path) -> None:
    labels = [str(row["case"]).replace("_", "\n") for row in rows]
    x = np.arange(len(rows), dtype=float)
    truth = np.array([row["true_spectral_radius"] for row in rows], dtype=float)
    exposure = np.array(
        [row["exposure_spectral_radius"] for row in rows], dtype=float
    )
    complete = np.array(
        [row["complete_tree_spectral_radius"] for row in rows], dtype=float
    )
    lower = exposure - np.array(
        [row["exposure_rho_ci_lower"] for row in rows], dtype=float
    )
    upper = np.array(
        [row["exposure_rho_ci_upper"] for row in rows], dtype=float
    ) - exposure

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=True)
    axes[0, 0].plot(x, truth, "ko", label="known truth")
    axes[0, 0].errorbar(
        x,
        exposure,
        yerr=np.vstack([lower, upper]),
        fmt="s",
        capsize=4,
        color="#175cd3",
        label="exposure likelihood (95% bootstrap)",
    )
    axes[0, 0].axhline(1.0, color="#b42318", linestyle=":")
    axes[0, 0].set(
        xticks=x,
        xticklabels=labels,
        ylabel="Spectral radius",
        title="Known-law recovery under administrative censoring",
    )
    axes[0, 0].legend(fontsize=8)

    width = 0.36
    axes[0, 1].bar(
        x - width / 2,
        [row["exposure_rho_absolute_error"] for row in rows],
        width,
        label="exposure likelihood",
        color="#175cd3",
    )
    axes[0, 1].bar(
        x + width / 2,
        [row["complete_tree_rho_absolute_error"] for row in rows],
        width,
        label="complete-tree selection",
        color="#f79009",
    )
    axes[0, 1].set(
        xticks=x,
        xticklabels=labels,
        ylabel="Absolute spectral-radius error",
        title="Conditioning on completed trees creates selection bias",
    )
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].bar(
        x,
        [row["parent_censoring_fraction"] for row in rows],
        color="#7a5af8",
    )
    axes[1, 0].set(
        xticks=x,
        xticklabels=labels,
        ylabel="Censored parent fraction",
        ylim=(0.0, 1.0),
        title="Administrative censoring retained by the likelihood",
    )

    axes[1, 1].plot(
        x, truth, "ko-", label="known truth"
    )
    axes[1, 1].plot(
        x, complete, "o--", color="#f79009", label="complete-tree estimate"
    )
    axes[1, 1].axhline(1.0, color="#b42318", linestyle=":")
    axes[1, 1].set(
        xticks=x,
        xticklabels=labels,
        ylabel="Spectral radius",
        title="Complete-tree MLE cannot demonstrate supercriticality",
    )
    axes[1, 1].legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(alpha=0.2, axis="y")
        axis.tick_params(axis="x", labelsize=8)
    figure.suptitle(
        "Milestone 6: exposure-aware branching identification benchmark"
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
    summary, entries = _run(settings, arguments.seed)
    _write_csv(
        arguments.output_dir / "censored_branching_summary.csv", summary
    )
    _write_csv(
        arguments.output_dir / "censored_branching_matrix_entries.csv", entries
    )
    _plot(
        summary,
        arguments.output_dir / "censored_branching_validation.png",
    )

    subcritical = [
        row for row in summary if row["true_regime"] == "subcritical"
    ]
    supercritical = [
        row for row in summary if row["true_regime"] == "supercritical"
    ]
    manifest = {
        "profile": arguments.profile,
        "seed": arguments.seed,
        "settings": settings,
        "findings": {
            "all_known_laws_within_matrix_mae_0_15": all(
                float(row["matrix_mean_absolute_error"]) <= 0.15
                for row in summary
            ),
            "all_subcritical_regimes_classified_correctly": all(
                row["exposure_regime"] == "subcritical"
                for row in subcritical
            ),
            "all_supercritical_regimes_classified_correctly": all(
                row["exposure_regime"] == "supercritical"
                for row in supercritical
            ),
            "all_complete_tree_estimates_at_or_below_one": all(
                float(row["complete_tree_spectral_radius"]) <= 1.0 + 1e-12
                for row in summary
            ),
            "supercritical_complete_tree_failure_observed": all(
                row["complete_tree_regime"] == "subcritical"
                for row in supercritical
            ),
            "supercritical_bootstrap_lower_bounds_above_one": all(
                float(row["exposure_rho_ci_lower"]) > 1.0
                for row in supercritical
            ),
            "minimum_bootstrap_valid_fraction": min(
                int(row["bootstrap_valid_resamples"])
                / int(row["bootstrap_requested_resamples"])
                for row in summary
            ),
        },
        "interpretation": {
            "estimand": (
                "B[a,b]=beta[a,b]/delta[a], where beta is the observed "
                "child-start rate and delta is the event-resolution rate"
            ),
            "censoring": (
                "active-at-horizon events contribute partial exposure and "
                "observed children but are not counted as resolutions"
            ),
            "resampling_unit": (
                "whole independent primary-root forests; event IDs remain local"
            ),
            "claim_boundary": (
                "known-law simulation validates estimator mechanics only; "
                "physical AMR criticality still requires independent layouts "
                "and data"
            ),
        },
    }
    with (
        arguments.output_dir / "censored_branching_manifest.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps(manifest["findings"], indent=2))


if __name__ == "__main__":
    main()
