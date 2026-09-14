"""Reference mathematics for shield-coupled AMR fleet capacity.

This module is the single source of truth for equations used by the simulator,
metrics, plots, and tests.  It deliberately contains no geometry, task logic,
or simulator state.

Conventions
-----------
* ``B[a, b]`` is the expected number of direct type-b child interventions
  caused by one type-a parent intervention (row-parent convention).
* Speeds are nonnegative and measured in m/s.
* Distances are measured in m and times in s.
* Every branching matrix must be finite, square, and elementwise nonnegative.

The stopping-distance equations are research abstractions.  They are not a
substitute for a certified protective-field calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CascadeBudgetResult:
    """Result of the additive resolvent-certificate linear program."""

    feasible: bool
    budget: float | None
    certificate: FloatArray | None
    spectral_radius: float
    minimum_residual: float | None
    solver_status: str


@dataclass(frozen=True)
class ScalarEquilibrium:
    """One quasi-stationary solution of the scalar open-system closure."""

    density: float
    reproduction_number: float
    fixed_point_slope: float
    stable: bool


@dataclass(frozen=True)
class SpeedCapacityResult:
    """Usable capacity after imposing a finite coupled-density limit."""

    arrival_capacity: float | FloatArray
    operating_density: float | FloatArray
    fold_density: float | FloatArray
    offspring_probability: float | FloatArray
    fold_limited: bool | NDArray[np.bool_]


def _float_array(value: ArrayLike) -> FloatArray:
    return np.asarray(value, dtype=float)


def _return_scalar_if_scalar_input(value: FloatArray, *inputs: object):
    if all(np.ndim(item) == 0 for item in inputs):
        return float(np.asarray(value))
    return value


def _require_finite(name: str, value: ArrayLike) -> FloatArray:
    array = _float_array(value)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _require_positive(name: str, value: ArrayLike) -> FloatArray:
    array = _require_finite(name, value)
    if np.any(array <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return array


def _require_nonnegative(name: str, value: ArrayLike) -> FloatArray:
    array = _require_finite(name, value)
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return array


def _validate_branching_matrix(B: ArrayLike) -> FloatArray:
    matrix = _require_nonnegative("B", B)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("B must be a square matrix")
    if matrix.shape[0] == 0:
        raise ValueError("B must be nonempty")
    return matrix


def _validate_vector(name: str, value: ArrayLike, size: int, *, positive: bool) -> FloatArray:
    vector = _require_finite(name, value).reshape(-1)
    if vector.size != size:
        raise ValueError(f"{name} must have length {size}")
    if positive and np.any(vector <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    if not positive and np.any(vector < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return vector


# ---------------------------------------------------------------------------
# 1. Local shield equations
# ---------------------------------------------------------------------------


def d_stop(v: ArrayLike, a0: ArrayLike, tau_r: ArrayLike, b: ArrayLike):
    """Conservative longitudinal stopping distance.

    ``a0`` combines static geometry and localization margins. ``tau_r`` is
    total response latency, and ``b`` is the guaranteed deceleration magnitude.
    Inputs broadcast according to NumPy rules.
    """

    speed = _require_nonnegative("v", v)
    margin = _require_nonnegative("a0", a0)
    latency = _require_nonnegative("tau_r", tau_r)
    braking = _require_positive("b", b)
    result = margin + speed * latency + speed**2 / (2.0 * braking)
    return _return_scalar_if_scalar_input(result, v, a0, tau_r, b)


def v_safe(d_free: ArrayLike, a0: ArrayLike, tau_r: ArrayLike, b: ArrayLike):
    """Largest nonnegative speed whose stopping distance fits in ``d_free``."""

    distance = _require_nonnegative("d_free", d_free)
    margin = _require_nonnegative("a0", a0)
    latency = _require_nonnegative("tau_r", tau_r)
    braking = _require_positive("b", b)
    slack = np.maximum(0.0, distance - margin)
    result = np.maximum(
        0.0,
        -braking * latency
        + np.sqrt((braking * latency) ** 2 + 2.0 * braking * slack),
    )
    return _return_scalar_if_scalar_input(result, d_free, a0, tau_r, b)


def v_safe_next_step(
    d_free: ArrayLike,
    v_current: ArrayLike,
    a0: ArrayLike,
    tau_r: ArrayLike,
    b: ArrayLike,
    dt: ArrayLike,
):
    """Largest next speed preserving the static-obstacle invariant after a step.

    With constant acceleration during one integration step, traveled distance
    is ``0.5*(v_current + v_next)*dt``. This function solves

    ``movement + d_stop(v_next) <= d_free``

    for the largest nonnegative ``v_next``. The simulator separately checks
    that this speed is reachable under its deceleration bound.
    """

    distance = _require_nonnegative("d_free", d_free)
    current = _require_nonnegative("v_current", v_current)
    margin = _require_nonnegative("a0", a0)
    latency = _require_nonnegative("tau_r", tau_r)
    braking = _require_positive("b", b)
    step = _require_positive("dt", dt)
    effective_latency = latency + step / 2.0
    slack = np.maximum(0.0, distance - margin - current * step / 2.0)
    result = np.maximum(
        0.0,
        -braking * effective_latency
        + np.sqrt(
            (braking * effective_latency) ** 2 + 2.0 * braking * slack
        ),
    )
    return _return_scalar_if_scalar_input(
        result, d_free, v_current, a0, tau_r, b, dt
    )


def tau_buffer(v: ArrayLike, A: ArrayLike, tau_r: ArrayLike, b: ArrayLike):
    """Temporal follower buffer for spare headway ``A = h - a0``."""

    speed = _require_positive("v", v)
    spare = _require_nonnegative("A", A)
    latency = _require_nonnegative("tau_r", tau_r)
    braking = _require_positive("b", b)
    result = np.maximum(0.0, spare / speed - latency - speed / (2.0 * braking))
    return _return_scalar_if_scalar_input(result, v, A, tau_r, b)


def buffer_loss_speed(A: float, tau_r: float, b: float) -> float:
    """Speed ``v0`` at which the temporal buffer first reaches zero."""

    spare = float(_require_nonnegative("A", A))
    latency = float(_require_nonnegative("tau_r", tau_r))
    braking = float(_require_positive("b", b))
    return float(
        -braking * latency
        + np.sqrt((braking * latency) ** 2 + 2.0 * braking * spare)
    )


# ---------------------------------------------------------------------------
# 2. Closed homogeneous intervention model
# ---------------------------------------------------------------------------


def reproduction_number(
    v: ArrayLike,
    n: ArrayLike,
    chi: ArrayLike,
    gamma: ArrayLike,
    A: ArrayLike,
    tau_r: ArrayLike,
    b: ArrayLike,
):
    """Homogeneous reproduction number under exponential recovery."""

    coupled = _require_nonnegative("n", n)
    probability = offspring_probability(v, chi, gamma, A, tau_r, b)
    result = coupled * probability
    return _return_scalar_if_scalar_input(result, v, n, chi, gamma, A, tau_r, b)


def offspring_probability(
    v: ArrayLike,
    chi: ArrayLike,
    gamma: ArrayLike,
    A: ArrayLike,
    tau_r: ArrayLike,
    b: ArrayLike,
):
    """Direct-follower coupling ``phi(v)`` under exponential recovery."""

    coupling = _require_nonnegative("chi", chi)
    if np.any(coupling > 1.0):
        raise ValueError("chi must not exceed one")
    recovery = _require_positive("gamma", gamma)
    buffer = tau_buffer(v, A, tau_r, b)
    result = coupling * np.exp(-recovery * buffer)
    return _return_scalar_if_scalar_input(result, v, chi, gamma, A, tau_r, b)


def critical_speed(
    n: float,
    chi: float,
    gamma: float,
    A: float,
    tau_r: float,
    b: float,
) -> float | None:
    """Closed-form positive speed solving ``R_I(v)=1``.

    Returns ``None`` when ``n * chi <= 1`` because there is no unique positive
    crossing below the buffer-loss speed.  The equality case becomes critical
    on the plateau ``v >= v0`` rather than at a unique crossing.
    """

    coupled = float(_require_nonnegative("n", n))
    coupling = float(_require_nonnegative("chi", chi))
    if coupling > 1.0:
        raise ValueError("chi must not exceed one")
    recovery = float(_require_positive("gamma", gamma))
    spare = float(_require_nonnegative("A", A))
    latency = float(_require_nonnegative("tau_r", tau_r))
    braking = float(_require_positive("b", b))
    if coupled * coupling <= 1.0 or spare == 0.0:
        return None
    kappa0 = latency + np.log(coupled * coupling) / recovery
    result = -braking * kappa0 + np.sqrt(
        (braking * kappa0) ** 2 + 2.0 * braking * spare
    )
    return float(result)


# ---------------------------------------------------------------------------
# 3. Scalar open-system fold
# ---------------------------------------------------------------------------


def mu_of_n(n: ArrayLike, T0: float, g: float, phi: float, eta: float):
    """Sustainable arrival rate at coupled work-in-process ``n``.

    The formula is valid only on the subcritical domain ``0 <= n * phi < 1``.
    """

    density = _require_nonnegative("n", n)
    nominal_time = float(_require_positive("T0", T0))
    burden = float(_require_nonnegative("g", g))
    coupling = float(_require_positive("phi", phi))
    fraction = float(_require_positive("eta", eta))
    if fraction > 1.0:
        raise ValueError("eta must not exceed one")
    slack = 1.0 - density * coupling
    if np.any(slack <= 0.0):
        raise ValueError("mu_of_n requires n * phi < 1")
    result = density * slack / (fraction * (nominal_time * slack + burden))
    return _return_scalar_if_scalar_input(result, n)


def lambda_saddle_node(T0: float, g: float, phi: float, eta: float) -> float:
    """Scalar open-system fold capacity.

    For ``g == 0`` this returns the limiting boundary value, but there is no
    genuine interior saddle-node.  A genuine fold requires ``g > 0``.
    """

    nominal_time = float(_require_positive("T0", T0))
    burden = float(_require_nonnegative("g", g))
    coupling = float(_require_positive("phi", phi))
    fraction = float(_require_positive("eta", eta))
    if fraction > 1.0:
        raise ValueError("eta must not exceed one")
    denominator = fraction * coupling * (
        np.sqrt(nominal_time + burden) + np.sqrt(burden)
    ) ** 2
    return float(1.0 / denominator)


def n_saddle_node(T0: float, g: float, phi: float) -> float:
    """Coupled work-in-process at the scalar fold."""

    nominal_time = float(_require_positive("T0", T0))
    burden = float(_require_nonnegative("g", g))
    coupling = float(_require_positive("phi", phi))
    numerator = np.sqrt(nominal_time + burden)
    denominator = coupling * (numerator + np.sqrt(burden))
    return float(numerator / denominator)


def r_saddle_node(T0: float, g: float) -> float:
    """Reproduction number at the scalar fold.

    For a genuine fold, ``g > 0`` and the result lies strictly in ``(1/2, 1)``.
    At ``g == 0`` the value one is a degenerate limiting boundary.
    """

    nominal_time = float(_require_positive("T0", T0))
    burden = float(_require_nonnegative("g", g))
    root_total = np.sqrt(nominal_time + burden)
    return float(root_total / (root_total + np.sqrt(burden)))


def r_saddle_node_from_burden(x: ArrayLike):
    """Parameter-free scalar fold curve for ``x = g / T0``."""

    burden = _require_nonnegative("x", x)
    result = np.sqrt(1.0 + burden) / (
        np.sqrt(1.0 + burden) + np.sqrt(burden)
    )
    return _return_scalar_if_scalar_input(result, x)


def fold_margin_scalar(n: ArrayLike, T0: float, g: float, phi: float):
    """Local free-flow stability margin ``1 - n*T'(n)/T(n)``."""

    density = _require_nonnegative("n", n)
    nominal_time = float(_require_positive("T0", T0))
    burden = float(_require_nonnegative("g", g))
    coupling = float(_require_positive("phi", phi))
    slack = 1.0 - density * coupling
    if np.any(slack <= 0.0):
        raise ValueError("fold_margin_scalar requires n * phi < 1")
    effective_time = nominal_time + burden / slack
    time_derivative = burden * coupling / slack**2
    result = 1.0 - density * time_derivative / effective_time
    return _return_scalar_if_scalar_input(result, n)


def scalar_equilibria(
    arrival_rate: float,
    T0: float,
    g: float,
    phi: float,
    eta: float,
    *,
    numerical_tolerance: float = 1e-12,
) -> tuple[ScalarEquilibrium, ...]:
    """Solve the quasi-stationary fixed point below ``n*phi = 1``.

    The closure is

    ``n = eta*lambda*(T0 + g/(1 - n*phi))``.

    A sub-fold arrival rate normally gives a stable lower-density solution and
    an unstable upper-density solution.  At the fold they coalesce into one
    neutral solution; above the fold no subcritical solution is returned.
    For the degenerate case ``g == 0``, multiplication by ``1-n*phi`` creates
    an extraneous boundary root, which this routine removes.
    """

    rate = float(_require_nonnegative("arrival_rate", arrival_rate))
    nominal_time = float(_require_positive("T0", T0))
    burden = float(_require_nonnegative("g", g))
    coupling = float(_require_positive("phi", phi))
    fraction = float(_require_positive("eta", eta))
    tolerance = float(_require_positive("numerical_tolerance", numerical_tolerance))
    if fraction > 1.0:
        raise ValueError("eta must not exceed one")

    offered_density = fraction * rate
    # phi*n^2 - (1 + eta*lambda*T0*phi)*n
    #     + eta*lambda*(T0+g) = 0.
    coefficient_b = -(1.0 + offered_density * nominal_time * coupling)
    coefficient_c = offered_density * (nominal_time + burden)
    discriminant = coefficient_b**2 - 4.0 * coupling * coefficient_c
    scale = max(1.0, coefficient_b**2, 4.0 * coupling * coefficient_c)
    if discriminant < -tolerance * scale:
        return ()
    discriminant = max(0.0, discriminant)

    root_discriminant = np.sqrt(discriminant)
    q_value = -0.5 * (
        coefficient_b + np.copysign(root_discriminant, coefficient_b)
    )
    candidate_roots = [q_value / coupling]
    if abs(q_value) > tolerance:
        candidate_roots.append(coefficient_c / q_value)
    else:
        candidate_roots.append(-coefficient_b / (2.0 * coupling))

    equilibria: list[ScalarEquilibrium] = []
    for root in sorted(candidate_roots):
        if root < -tolerance:
            continue
        density = max(0.0, float(root))
        reproduction = density * coupling
        if reproduction >= 1.0:
            continue
        if equilibria and np.isclose(
            density,
            equilibria[-1].density,
            rtol=10.0 * tolerance,
            atol=10.0 * tolerance,
        ):
            continue
        slack = 1.0 - reproduction
        slope = offered_density * burden * coupling / slack**2
        equilibria.append(
            ScalarEquilibrium(
                density=density,
                reproduction_number=reproduction,
                fixed_point_slope=float(slope),
                stable=bool(slope < 1.0 - 10.0 * tolerance),
            )
        )
    return tuple(equilibria)


def fold_capacity_at_speed(
    v: ArrayLike,
    path_length: ArrayLike,
    g: ArrayLike,
    chi: ArrayLike,
    gamma: ArrayLike,
    A: ArrayLike,
    tau_r: ArrayLike,
    b: ArrayLike,
    eta: ArrayLike,
):
    """Open-system fold capacity as a function of commanded speed.

    ``g`` is the severity-weighted primary intervention burden evaluated at the
    same speed(s).  Inputs broadcast according to NumPy rules.  A strictly
    positive ``chi`` is required because ``chi == 0`` removes this cascade-fold
    mechanism rather than creating a finite saddle-node.
    """

    speed = _require_positive("v", v)
    length = _require_positive("path_length", path_length)
    burden = _require_nonnegative("g", g)
    fraction = _require_positive("eta", eta)
    if np.any(fraction > 1.0):
        raise ValueError("eta must not exceed one")
    coupling = _require_positive("chi", chi)
    if np.any(coupling > 1.0):
        raise ValueError("chi must not exceed one")
    nominal_time = length / speed
    phi = offspring_probability(speed, coupling, gamma, A, tau_r, b)
    denominator = fraction * phi * (
        np.sqrt(nominal_time + burden) + np.sqrt(burden)
    ) ** 2
    result = 1.0 / denominator
    return _return_scalar_if_scalar_input(
        result, v, path_length, g, chi, gamma, A, tau_r, b, eta
    )


def capacity_with_density_limit(
    v: ArrayLike,
    path_length: ArrayLike,
    g: ArrayLike,
    chi: ArrayLike,
    gamma: ArrayLike,
    A: ArrayLike,
    tau_r: ArrayLike,
    b: ArrayLike,
    eta: ArrayLike,
    density_limit: ArrayLike,
) -> SpeedCapacityResult:
    """Usable speed-dependent capacity for a finite facility or fleet.

    The unbounded mean-field fold ``lambda_sn(v)`` is only the active capacity
    boundary when the facility can reach ``n_sn(v)``.  With a maximum coupled
    work-in-process ``n_max``, the usable boundary is

    ``mu(min(n_max, n_sn(v)))``.

    Therefore low-speed points with tiny coupling cannot claim arbitrarily high
    throughput by accumulating arbitrarily many robots.  ``g == 0`` is treated
    as a no-delay degenerate case and is always occupancy-limited.
    """

    speed = _require_positive("v", v)
    length = _require_positive("path_length", path_length)
    burden = _require_nonnegative("g", g)
    fraction = _require_positive("eta", eta)
    maximum_density = _require_positive("density_limit", density_limit)
    if np.any(fraction > 1.0):
        raise ValueError("eta must not exceed one")
    coupling = _require_positive("chi", chi)
    if np.any(coupling > 1.0):
        raise ValueError("chi must not exceed one")

    nominal_time = length / speed
    phi = offspring_probability(speed, coupling, gamma, A, tau_r, b)
    (
        nominal_time,
        burden,
        fraction,
        maximum_density,
        phi,
    ) = np.broadcast_arrays(
        nominal_time, burden, fraction, maximum_density, phi
    )
    positive_burden = burden > 0.0
    root_total = np.sqrt(nominal_time + burden)
    analytic_fold_density = root_total / (
        phi * (root_total + np.sqrt(burden))
    )
    fold_density = np.where(positive_burden, analytic_fold_density, np.inf)
    operating_density = np.minimum(maximum_density, fold_density)
    fold_limited = positive_burden & (maximum_density >= fold_density)

    slack = 1.0 - operating_density * phi
    positive_capacity = operating_density * slack / (
        fraction * (nominal_time * slack + burden)
    )
    no_delay_capacity = operating_density / (fraction * nominal_time)
    arrival_capacity = np.where(
        positive_burden, positive_capacity, no_delay_capacity
    )

    scalar_inputs = (v, path_length, g, chi, gamma, A, tau_r, b, eta, density_limit)
    return SpeedCapacityResult(
        arrival_capacity=_return_scalar_if_scalar_input(
            arrival_capacity, *scalar_inputs
        ),
        operating_density=_return_scalar_if_scalar_input(
            operating_density, *scalar_inputs
        ),
        fold_density=_return_scalar_if_scalar_input(fold_density, *scalar_inputs),
        offspring_probability=_return_scalar_if_scalar_input(phi, *scalar_inputs),
        fold_limited=(
            bool(np.asarray(fold_limited))
            if all(np.ndim(item) == 0 for item in scalar_inputs)
            else np.asarray(fold_limited, dtype=bool)
        ),
    )


# ---------------------------------------------------------------------------
# 4. General non-normal cascade and fold equations
# ---------------------------------------------------------------------------


def spectral_radius(B: ArrayLike) -> float:
    """Spectral radius of a finite nonnegative branching matrix."""

    matrix = _validate_branching_matrix(B)
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def cascade_budget_direct(
    B: ArrayLike, severity: ArrayLike | None = None
) -> tuple[float, FloatArray]:
    """Directly solve for cumulative progeny or severity-weighted loss.

    With ``severity=None``, solves ``(I-B) zeta = 1`` and returns
    ``max(zeta)``.  ``rho(B)`` must be strictly below one.
    """

    matrix = _validate_branching_matrix(B)
    rho = spectral_radius(matrix)
    if rho >= 1.0:
        raise ValueError("cascade resolvent exists only when rho(B) < 1")
    size = matrix.shape[0]
    rhs = (
        np.ones(size)
        if severity is None
        else _validate_vector("severity", severity, size, positive=False)
    )
    certificate = np.linalg.solve(np.eye(size) - matrix, rhs)
    if np.min(certificate) < -1e-10:
        raise FloatingPointError("nonnegative resolvent produced a negative certificate")
    certificate = np.maximum(certificate, 0.0)
    return float(np.max(certificate)), certificate


def cascade_budget_lp(
    B: ArrayLike,
    severity: ArrayLike | None = None,
    *,
    feasibility_tolerance: float = 1e-8,
) -> CascadeBudgetResult:
    """Compute the exact additive resolvent certificate by linear programming.

    The default strictly positive right-hand side is one.  In exact arithmetic,
    this LP is feasible exactly when ``rho(B) < 1``.  A custom severity vector
    computes a severity-weighted bound, but zero severity entries should not be
    used alone as a subcriticality test.
    """

    matrix = _validate_branching_matrix(B)
    size = matrix.shape[0]
    rho = spectral_radius(matrix)
    rhs = (
        np.ones(size)
        if severity is None
        else _validate_vector("severity", severity, size, positive=False)
    )

    # Variables are [zeta_0, ..., zeta_(K-1), G].
    objective = np.zeros(size + 1)
    objective[-1] = 1.0

    # rhs + B*zeta <= zeta  <=>  (B-I)*zeta <= -rhs.
    cascade_constraint = np.hstack(
        [matrix - np.eye(size), np.zeros((size, 1))]
    )
    cascade_bound = -rhs

    # zeta_j <= G.
    budget_constraint = np.hstack(
        [np.eye(size), -np.ones((size, 1))]
    )
    budget_bound = np.zeros(size)

    lower_zeta = 1.0 if severity is None else 0.0
    variable_bounds = [(lower_zeta, None)] * size + [(0.0, None)]
    result = linprog(
        objective,
        A_ub=np.vstack([cascade_constraint, budget_constraint]),
        b_ub=np.concatenate([cascade_bound, budget_bound]),
        bounds=variable_bounds,
        method="highs",
    )

    if not result.success:
        return CascadeBudgetResult(
            feasible=False,
            budget=None,
            certificate=None,
            spectral_radius=rho,
            minimum_residual=None,
            solver_status=result.message,
        )

    certificate = np.asarray(result.x[:-1], dtype=float)
    residual = (np.eye(size) - matrix) @ certificate - rhs
    minimum_residual = float(np.min(residual))
    if minimum_residual < -feasibility_tolerance:
        return CascadeBudgetResult(
            feasible=False,
            budget=None,
            certificate=None,
            spectral_radius=rho,
            minimum_residual=minimum_residual,
            solver_status="solver returned a certificate with excessive residual violation",
        )

    return CascadeBudgetResult(
        feasible=True,
        budget=float(result.x[-1]),
        certificate=certificate,
        spectral_radius=rho,
        minimum_residual=minimum_residual,
        solver_status=result.message,
    )


def _validate_primary_distribution(primary: ArrayLike, size: int) -> FloatArray:
    distribution = _validate_vector("primary_distribution", primary, size, positive=False)
    total = float(np.sum(distribution))
    if total <= 0.0:
        raise ValueError("primary_distribution must have positive mass")
    if not np.isclose(total, 1.0, rtol=1e-9, atol=1e-12):
        raise ValueError("primary_distribution must sum to one")
    return distribution


def mean_cascade_loss(
    B: ArrayLike,
    primary_distribution: ArrayLike,
    severity: ArrayLike,
) -> float:
    """Expected severity-weighted fleet loss caused by one primary event."""

    matrix = _validate_branching_matrix(B)
    size = matrix.shape[0]
    primary = _validate_primary_distribution(primary_distribution, size)
    severity_vector = _validate_vector("severity", severity, size, positive=False)
    _, cumulative = cascade_budget_direct(matrix, severity_vector)
    return float(primary @ cumulative)


def truncated_cascade_loss(
    B: ArrayLike,
    primary_distribution: ArrayLike,
    severity: ArrayLike,
    max_generation: int,
) -> float:
    """Finite-depth cascade loss ``a^T sum_{k=0}^D B^k d``.

    Unlike the infinite resolvent, this quantity is finite even when
    ``rho(B) >= 1``. ``max_generation=0`` counts only the primary generation.
    """

    matrix = _validate_branching_matrix(B)
    if max_generation < 0:
        raise ValueError("max_generation must be nonnegative")
    size = matrix.shape[0]
    primary = _validate_primary_distribution(primary_distribution, size)
    current = _validate_vector("severity", severity, size, positive=False)
    cumulative = current.copy()
    for _ in range(max_generation):
        current = matrix @ current
        cumulative += current
    return float(primary @ cumulative)


def truncated_cascade_budget(
    B: ArrayLike,
    max_generation: int,
    severity: ArrayLike | None = None,
) -> tuple[float, FloatArray]:
    """Worst-case finite-depth progeny or severity-weighted loss."""

    matrix = _validate_branching_matrix(B)
    if max_generation < 0:
        raise ValueError("max_generation must be nonnegative")
    size = matrix.shape[0]
    current = (
        np.ones(size)
        if severity is None
        else _validate_vector("severity", severity, size, positive=False)
    )
    cumulative = current.copy()
    for _ in range(max_generation):
        current = matrix @ current
        cumulative += current
    return float(np.max(cumulative)), cumulative


def cascade_loss_derivative(
    B: ArrayLike,
    dB_dn: ArrayLike,
    primary_distribution: ArrayLike,
    severity: ArrayLike,
) -> float:
    """Derivative of mean cascade loss with respect to coupled density.

    Implements ``a^T R (dB/dn) R d`` with ``R = (I-B)^-1``.
    """

    matrix = _validate_branching_matrix(B)
    derivative = _require_finite("dB_dn", dB_dn)
    if derivative.shape != matrix.shape:
        raise ValueError("dB_dn must have the same shape as B")
    size = matrix.shape[0]
    primary = _validate_primary_distribution(primary_distribution, size)
    severity_vector = _validate_vector("severity", severity, size, positive=False)
    rho = spectral_radius(matrix)
    if rho >= 1.0:
        raise ValueError("cascade loss derivative requires rho(B) < 1")
    resolvent = np.linalg.inv(np.eye(size) - matrix)
    return float(primary @ resolvent @ derivative @ resolvent @ severity_vector)


def fold_margin_general(
    n: float,
    T0: float,
    primary_probability: float,
    cascade_loss: float,
    cascade_loss_derivative_value: float,
) -> float:
    """Topology-aware fold margin ``1 - n*T_n/T``.

    Positive values describe the increasing/stable free-flow branch, zero is a
    smooth fold, and negative values describe the decreasing branch under the
    quasi-stationary mean-field closure.
    """

    density = float(_require_nonnegative("n", n))
    nominal_time = float(_require_positive("T0", T0))
    probability = float(_require_nonnegative("primary_probability", primary_probability))
    if probability > 1.0:
        raise ValueError("primary_probability must not exceed one")
    loss = float(_require_nonnegative("cascade_loss", cascade_loss))
    loss_derivative = float(
        _require_nonnegative("cascade_loss_derivative", cascade_loss_derivative_value)
    )
    effective_time = nominal_time + probability * loss
    return float(1.0 - density * probability * loss_derivative / effective_time)


def fold_elasticity_identity(
    n: float,
    T0: float,
    primary_probability: float,
    cascade_loss: float,
    cascade_loss_derivative_value: float,
) -> tuple[float, float]:
    """Return both sides of the general parameter-free fold identity.

    At a smooth fold the returned values satisfy
    ``n*H'(n)/H(n) == 1 + 1/x_eff``, where ``x_eff = p*H/T0``.
    """

    density = float(_require_nonnegative("n", n))
    nominal_time = float(_require_positive("T0", T0))
    probability = float(_require_positive("primary_probability", primary_probability))
    loss = float(_require_positive("cascade_loss", cascade_loss))
    loss_derivative = float(
        _require_nonnegative("cascade_loss_derivative", cascade_loss_derivative_value)
    )
    elasticity = density * loss_derivative / loss
    effective_burden = probability * loss / nominal_time
    target = 1.0 + 1.0 / effective_burden
    return float(elasticity), float(target)


# ---------------------------------------------------------------------------
# 5. Adversarial non-normal matrices used by tests and figures
# ---------------------------------------------------------------------------


def chain_matrix(size: int, beta: float) -> FloatArray:
    """Directed nilpotent chain with ``B[i, i+1] = beta``."""

    if size < 1:
        raise ValueError("size must be at least one")
    strength = float(_require_nonnegative("beta", beta))
    matrix = np.zeros((size, size), dtype=float)
    if size > 1:
        matrix[np.arange(size - 1), np.arange(1, size)] = strength
    return matrix


def fanout_matrix(size: int, beta: float) -> FloatArray:
    """Directed nilpotent fan-out with type zero triggering all other types."""

    if size < 1:
        raise ValueError("size must be at least one")
    strength = float(_require_nonnegative("beta", beta))
    matrix = np.zeros((size, size), dtype=float)
    if size > 1:
        matrix[0, 1:] = strength
    return matrix


def conditioned_bound(epsilon: float, weights: Iterable[float]) -> float:
    """Return ``kappa/epsilon`` for a positive-vector contraction certificate."""

    margin = float(_require_positive("epsilon", epsilon))
    if margin > 1.0:
        raise ValueError("epsilon must not exceed one")
    vector = _require_positive("weights", list(weights)).reshape(-1)
    return float(np.max(vector) / np.min(vector) / margin)
