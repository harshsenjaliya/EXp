#!/usr/bin/env python3
"""Generate deterministic tables and figures from the reference equations."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from amr_capacity.capacity_theory import (
    cascade_budget_lp,
    chain_matrix,
    fanout_matrix,
    lambda_saddle_node,
    n_saddle_node,
    r_saddle_node_from_burden,
    scalar_equilibria,
    spectral_radius,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"


def write_fold_curve() -> tuple[np.ndarray, np.ndarray]:
    x = np.logspace(-4, 3, 500)
    reproduction = r_saddle_node_from_burden(x)
    with (OUTPUT / "scalar_fold_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dimensionless_burden_g_over_T0", "R_at_fold"])
        writer.writerows(zip(x, reproduction, strict=True))
    return x, reproduction


def write_topology_table() -> list[tuple[str, int, float, float, float]]:
    rows: list[tuple[str, int, float, float, float]] = []
    for size in (4, 8, 16, 32):
        for beta in (0.7, 0.95, 0.99):
            for name, matrix in (
                ("chain", chain_matrix(size, beta)),
                ("fanout", fanout_matrix(size, beta)),
            ):
                result = cascade_budget_lp(matrix)
                if not result.feasible or result.budget is None:
                    raise RuntimeError(f"unexpected infeasible {name} matrix")
                rows.append((name, size, beta, spectral_radius(matrix), result.budget))
    with (OUTPUT / "non_normal_topologies.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["topology", "types", "beta", "spectral_radius", "G_star"])
        writer.writerows(rows)
    return rows


def write_figure(x: np.ndarray, reproduction: np.ndarray) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)

    axes[0].semilogx(x, reproduction, linewidth=2.4, color="#1769aa")
    axes[0].axhline(0.5, linestyle="--", linewidth=1.2, color="#6b7280")
    axes[0].set_xlabel(r"Dimensionless intervention burden $x=g/T_0$")
    axes[0].set_ylabel(r"Reproduction number at fold $\mathcal{R}_{sn}$")
    axes[0].set_title("Parameter-free scalar fold prediction")
    axes[0].grid(alpha=0.25)
    axes[0].set_ylim(0.48, 1.02)

    sizes = np.arange(2, 65)
    beta = 0.99
    chain_budgets = np.array(
        [cascade_budget_lp(chain_matrix(int(size), beta)).budget for size in sizes]
    )
    fanout_budgets = np.array(
        [cascade_budget_lp(fanout_matrix(int(size), beta)).budget for size in sizes]
    )
    axes[1].plot(sizes, chain_budgets, linewidth=2.2, label="Directed chain")
    axes[1].plot(sizes, fanout_budgets, linewidth=2.2, label="Fan-out")
    axes[1].plot(sizes, np.zeros_like(sizes), linestyle="--", color="#111827", label=r"$\rho(B)$")
    axes[1].set_xlabel("Number of intervention types")
    axes[1].set_ylabel(r"Cumulative cascade budget $G^\star$")
    axes[1].set_title(r"Non-normal amplification at $\rho(B)=0$")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    figure.savefig(OUTPUT / "reference_theory.png", dpi=200)
    plt.close(figure)


def write_equilibrium_branches() -> None:
    """Write and plot one normalized scalar saddle-node bifurcation."""

    nominal_time, burden, coupling, fraction = 8.0, 1.5, 0.12, 0.4
    capacity = lambda_saddle_node(nominal_time, burden, coupling, fraction)
    fold_density = n_saddle_node(nominal_time, burden, coupling)
    normalized_rates = np.linspace(0.001, 1.0, 300)
    stable_points: list[tuple[float, float]] = []
    unstable_points: list[tuple[float, float]] = []
    rows: list[tuple[float, float, str, float]] = []

    for normalized_rate in normalized_rates:
        rate = normalized_rate * capacity
        for equilibrium in scalar_equilibria(
            rate, nominal_time, burden, coupling, fraction
        ):
            branch = "stable" if equilibrium.stable else "unstable"
            normalized_density = equilibrium.density / fold_density
            rows.append(
                (
                    normalized_rate,
                    normalized_density,
                    branch,
                    equilibrium.fixed_point_slope,
                )
            )
            target = stable_points if equilibrium.stable else unstable_points
            target.append((normalized_rate, normalized_density))

    with (OUTPUT / "scalar_equilibrium_branches.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "arrival_rate_over_lambda_sn",
                "density_over_n_sn",
                "branch",
                "fixed_point_slope",
            ]
        )
        writer.writerows(rows)

    figure, axis = plt.subplots(figsize=(6.3, 4.5), constrained_layout=True)
    for points, label, color, linestyle in (
        (stable_points, "Stable free-flow branch", "#1769aa", "-"),
        (unstable_points, "Unstable threshold branch", "#d97706", "--"),
    ):
        values = np.asarray(points)
        axis.plot(
            values[:, 0],
            values[:, 1],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.4,
        )
    axis.scatter([1.0], [1.0], color="#b91c1c", s=45, zorder=4, label="Fold")
    axis.set_xlabel(r"Normalized arrival rate $\lambda/\lambda_{sn}$")
    axis.set_ylabel(r"Normalized coupled density $n/n_{sn}$")
    axis.set_title("Quasi-stationary saddle-node branches")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(OUTPUT / "scalar_equilibrium_branches.png", dpi=200)
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    x, reproduction = write_fold_curve()
    rows = write_topology_table()
    write_figure(x, reproduction)
    write_equilibrium_branches()

    print("Reference outputs written to", OUTPUT)
    print("\nSelected scalar-fold values")
    print("       x       R_sn")
    for burden in (0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0):
        print(f"  {burden:7.3g}   {r_saddle_node_from_burden(burden):.6f}")

    print("\nSelected non-normal result")
    selected = next(
        row for row in rows if row[0] == "chain" and row[1] == 32 and row[2] == 0.99
    )
    print(
        f"  topology={selected[0]}, K={selected[1]}, beta={selected[2]:.2f}, "
        f"rho={selected[3]:.1f}, G*={selected[4]:.6f}"
    )


if __name__ == "__main__":
    main()
