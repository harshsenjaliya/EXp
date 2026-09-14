"""Metrics adapters that delegate all equations to :mod:`capacity_theory`.

The future simulator should import this module instead of reimplementing any
closed form.  This keeps reported metrics and analytical reference values tied
to exactly the same tested functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .capacity_theory import (
    cascade_budget_lp,
    cascade_loss_derivative,
    fold_margin_general,
    lambda_saddle_node,
    mean_cascade_loss,
    n_saddle_node,
    r_saddle_node,
    r_saddle_node_from_burden,
    spectral_radius,
)


@dataclass(frozen=True)
class ScalarFoldMetrics:
    lambda_sn: float
    n_sn: float
    reproduction_at_fold: float
    dimensionless_burden: float


@dataclass(frozen=True)
class GeneralCascadeMetrics:
    spectral_radius: float
    worst_case_budget: float
    mean_severity_loss: float
    mean_severity_loss_derivative: float | None
    fold_margin: float | None


def scalar_fold_metrics(T0: float, g: float, phi: float, eta: float) -> ScalarFoldMetrics:
    """Return all scalar reference values from the canonical equations."""

    return ScalarFoldMetrics(
        lambda_sn=lambda_saddle_node(T0, g, phi, eta),
        n_sn=n_saddle_node(T0, g, phi),
        reproduction_at_fold=r_saddle_node(T0, g),
        dimensionless_burden=g / T0,
    )


def general_cascade_metrics(
    B: ArrayLike,
    primary_distribution: ArrayLike,
    severity: ArrayLike,
    *,
    dB_dn: ArrayLike | None = None,
    n: float | None = None,
    T0: float | None = None,
    primary_probability: float | None = None,
) -> GeneralCascadeMetrics:
    """Compute simulator-facing cascade and optional fold metrics."""

    matrix = np.asarray(B, dtype=float)
    certificate = cascade_budget_lp(matrix)
    if not certificate.feasible or certificate.budget is None:
        raise ValueError("the cascade matrix is not certifiably subcritical")
    loss = mean_cascade_loss(matrix, primary_distribution, severity)

    supplied = [dB_dn is not None, n is not None, T0 is not None, primary_probability is not None]
    if any(supplied) and not all(supplied):
        raise ValueError(
            "dB_dn, n, T0, and primary_probability must be supplied together"
        )

    loss_derivative = None
    margin = None
    if all(supplied):
        loss_derivative = cascade_loss_derivative(
            matrix, dB_dn, primary_distribution, severity
        )
        margin = fold_margin_general(
            n,
            T0,
            primary_probability,
            loss,
            loss_derivative,
        )

    return GeneralCascadeMetrics(
        spectral_radius=spectral_radius(matrix),
        worst_case_budget=certificate.budget,
        mean_severity_loss=loss,
        mean_severity_loss_derivative=loss_derivative,
        fold_margin=margin,
    )


def scalar_fold_curve(x: ArrayLike):
    """Reference curve used by scalar-layout validation figures."""

    return r_saddle_node_from_burden(x)


def assert_zero_collisions(collision_count: int) -> None:
    """Hard simulator invariant: the local shield must permit no collisions."""

    if collision_count != 0:
        raise AssertionError(
            f"shield invariant violated: expected zero collisions, observed {collision_count}"
        )

