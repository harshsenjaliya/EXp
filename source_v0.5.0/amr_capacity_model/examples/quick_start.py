#!/usr/bin/env python3
"""Small executable example for the scalar speed-dependent fold model."""

from __future__ import annotations

import numpy as np

from amr_capacity.capacity_theory import (
    capacity_with_density_limit,
    scalar_equilibria,
)


def intervention_burden(speed: np.ndarray) -> np.ndarray:
    """Illustrative placeholder for measured ``g(v) = p(v)*D_bar(v)``.

    This curve is intentionally simple.  The 2D simulator will replace it with
    severity-weighted intervention measurements from paired rollouts.
    """

    return 0.08 + 0.04 * speed**2


def main() -> None:
    speed = np.linspace(0.2, 2.0, 400)
    path_length = 20.0
    chi, gamma = 0.32, 0.8
    spare_headway, reaction_time, braking = 2.0, 0.15, 1.1
    coupled_fraction = 0.4
    maximum_coupled_density = 20.0

    burden = intervention_burden(speed)
    result = capacity_with_density_limit(
        speed,
        path_length,
        burden,
        chi,
        gamma,
        spare_headway,
        reaction_time,
        braking,
        coupled_fraction,
        maximum_coupled_density,
    )
    capacity = np.asarray(result.arrival_capacity)
    best_index = int(np.argmax(capacity))
    best_speed = float(speed[best_index])
    best_capacity = float(capacity[best_index])
    best_burden = float(burden[best_index])
    nominal_time = path_length / best_speed
    phi = float(np.asarray(result.offspring_probability)[best_index])
    fold_limited = bool(np.asarray(result.fold_limited)[best_index])

    print(f"Best theoretical speed: {best_speed:.3f} m/s")
    print(f"Usable capacity: {best_capacity:.6f} jobs/s")
    print(f"Active boundary: {'cascade fold' if fold_limited else 'finite density'}")
    print(f"Direct-follower probability phi(v): {phi:.6f}")
    print("Equilibria at 95% of fold capacity:")
    for equilibrium in scalar_equilibria(
        0.95 * best_capacity,
        nominal_time,
        best_burden,
        phi,
        coupled_fraction,
    ):
        label = "stable" if equilibrium.stable else "unstable"
        print(
            f"  n={equilibrium.density:.6f}, "
            f"R={equilibrium.reproduction_number:.6f}, "
            f"F'(n)={equilibrium.fixed_point_slope:.6f}, {label}"
        )


if __name__ == "__main__":
    main()
