from __future__ import annotations

import math
import unittest

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from amr_capacity.capacity_theory import (
    buffer_loss_speed,
    cascade_budget_direct,
    cascade_budget_lp,
    cascade_loss_derivative,
    capacity_with_density_limit,
    chain_matrix,
    conditioned_bound,
    critical_speed,
    d_stop,
    fanout_matrix,
    fold_capacity_at_speed,
    fold_elasticity_identity,
    fold_margin_general,
    fold_margin_scalar,
    lambda_saddle_node,
    mean_cascade_loss,
    mu_of_n,
    n_saddle_node,
    offspring_probability,
    reproduction_number,
    r_saddle_node,
    r_saddle_node_from_burden,
    scalar_equilibria,
    spectral_radius,
    tau_buffer,
    truncated_cascade_budget,
    truncated_cascade_loss,
    v_safe,
    v_safe_next_step,
)
from amr_capacity.metrics import (
    assert_zero_collisions,
    general_cascade_metrics,
    scalar_fold_curve,
    scalar_fold_metrics,
)


class ShieldEquationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(20260913)

    def test_safe_speed_inverts_stopping_distance(self) -> None:
        for _ in range(1_000):
            a0 = self.rng.uniform(0.05, 0.5)
            tau_r = self.rng.uniform(0.05, 0.6)
            b = self.rng.uniform(0.3, 3.0)
            d_free = self.rng.uniform(0.0, 12.0)
            speed = v_safe(d_free, a0, tau_r, b)
            if d_free >= a0:
                self.assertAlmostEqual(
                    d_stop(speed, a0, tau_r, b), d_free, places=9
                )
            else:
                self.assertEqual(speed, 0.0)

    def test_vectorized_safe_speed(self) -> None:
        distances = np.array([0.1, 0.5, 1.0, 4.0])
        speeds = v_safe(distances, a0=0.2, tau_r=0.1, b=1.0)
        self.assertEqual(speeds.shape, distances.shape)
        self.assertEqual(speeds[0], 0.0)
        np.testing.assert_allclose(
            d_stop(speeds[1:], 0.2, 0.1, 1.0), distances[1:], rtol=1e-12
        )

    def test_discrete_safe_speed_preserves_next_step_invariant(self) -> None:
        for _ in range(1_000):
            current = self.rng.uniform(0.0, 2.0)
            a0 = self.rng.uniform(0.3, 1.2)
            tau_r = self.rng.uniform(0.05, 0.5)
            braking = self.rng.uniform(0.5, 3.0)
            dt = self.rng.uniform(0.005, 0.05)
            minimum_distance = a0 + current * dt / 2.0
            distance = self.rng.uniform(minimum_distance, 8.0)
            next_speed = v_safe_next_step(
                distance, current, a0, tau_r, braking, dt
            )
            movement = 0.5 * (current + next_speed) * dt
            self.assertAlmostEqual(
                movement + d_stop(next_speed, a0, tau_r, braking),
                distance,
                places=9,
            )

    def test_invalid_physical_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            d_stop(-1.0, 0.2, 0.1, 1.0)
        with self.assertRaises(ValueError):
            v_safe(1.0, 0.2, 0.1, 0.0)
        with self.assertRaises(ValueError):
            tau_buffer(0.0, 1.0, 0.1, 1.0)


class ClosedSystemTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(17)

    def test_critical_speed_matches_numeric_root(self) -> None:
        tested = 0
        for _ in range(1_000):
            A = self.rng.uniform(0.5, 8.0)
            tau_r = self.rng.uniform(0.05, 0.5)
            b = self.rng.uniform(0.3, 3.0)
            gamma = self.rng.uniform(0.1, 3.0)
            chi = self.rng.uniform(0.2, 1.0)
            n = self.rng.uniform(0.5, 8.0)
            closed_form = critical_speed(n, chi, gamma, A, tau_r, b)
            if n * chi <= 1.0:
                self.assertIsNone(closed_form)
                continue
            tested += 1
            v0 = buffer_loss_speed(A, tau_r, b)
            self.assertGreater(closed_form, 0.0)
            self.assertLess(closed_form, v0)
            numeric = brentq(
                lambda speed: reproduction_number(
                    speed, n, chi, gamma, A, tau_r, b
                )
                - 1.0,
                1e-8,
                v0,
                xtol=1e-14,
            )
            self.assertAlmostEqual(closed_form, numeric, places=9)
        self.assertGreater(tested, 500)

    def test_three_regimes(self) -> None:
        A, tau_r, b, gamma = 2.0, 0.2, 1.0, 0.8
        v0 = buffer_loss_speed(A, tau_r, b)
        self.assertLess(
            reproduction_number(2.0 * v0, 2.0, 0.4, gamma, A, tau_r, b), 1.0
        )
        self.assertAlmostEqual(
            reproduction_number(2.0 * v0, 2.0, 0.5, gamma, A, tau_r, b),
            1.0,
            places=12,
        )
        vc = critical_speed(2.0, 0.8, gamma, A, tau_r, b)
        self.assertIsNotNone(vc)
        self.assertLess(vc, v0)

    def test_reproduction_number_factors_into_density_and_offspring_probability(self) -> None:
        parameters = dict(v=1.2, chi=0.7, gamma=0.9, A=2.1, tau_r=0.2, b=1.1)
        phi = offspring_probability(**parameters)
        self.assertAlmostEqual(
            reproduction_number(n=3.2, **parameters), 3.2 * phi, places=13
        )


class ScalarFoldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(29)

    def test_fold_closed_forms_match_numeric_maximum(self) -> None:
        for _ in range(750):
            T0 = self.rng.uniform(0.5, 60.0)
            g = self.rng.uniform(1e-4, 40.0)
            phi = self.rng.uniform(1e-3, 0.9)
            eta = self.rng.uniform(0.05, 1.0)
            expected_lambda = lambda_saddle_node(T0, g, phi, eta)
            expected_n = n_saddle_node(T0, g, phi)
            numeric = minimize_scalar(
                lambda density: -mu_of_n(density, T0, g, phi, eta),
                bounds=(1e-12, (1.0 - 1e-10) / phi),
                method="bounded",
                options={"xatol": 1e-13},
            )
            self.assertLess(
                abs(-numeric.fun - expected_lambda) / expected_lambda, 2e-7
            )
            self.assertLess(abs(numeric.x - expected_n) / expected_n, 2e-4)
            self.assertAlmostEqual(
                expected_n * phi, r_saddle_node(T0, g), places=11
            )
            self.assertAlmostEqual(
                fold_margin_scalar(expected_n, T0, g, phi), 0.0, places=10
            )

    def test_parameter_free_band_and_monotonicity(self) -> None:
        x = np.logspace(-10, 10, 10_000)
        result = r_saddle_node_from_burden(x)
        self.assertTrue(np.all(result > 0.5))
        self.assertTrue(np.all(result < 1.0))
        self.assertTrue(np.all(np.diff(result) < 0.0))
        self.assertEqual(r_saddle_node_from_burden(0.0), 1.0)
        self.assertAlmostEqual(result[-1], 0.5, places=5)

    def test_metrics_delegate_to_reference_functions(self) -> None:
        metric = scalar_fold_metrics(T0=5.0, g=1.2, phi=0.15, eta=0.4)
        self.assertEqual(metric.lambda_sn, lambda_saddle_node(5.0, 1.2, 0.15, 0.4))
        self.assertEqual(metric.n_sn, n_saddle_node(5.0, 1.2, 0.15))
        np.testing.assert_allclose(
            scalar_fold_curve(np.array([0.1, 1.0])),
            r_saddle_node_from_burden(np.array([0.1, 1.0])),
        )

    def test_two_branches_coalesce_and_disappear_at_fold(self) -> None:
        T0, g, phi, eta = 8.0, 1.5, 0.12, 0.4
        capacity = lambda_saddle_node(T0, g, phi, eta)

        below = scalar_equilibria(0.95 * capacity, T0, g, phi, eta)
        self.assertEqual(len(below), 2)
        self.assertTrue(below[0].stable)
        self.assertFalse(below[1].stable)
        for equilibrium in below:
            rhs = eta * 0.95 * capacity * (
                T0 + g / (1.0 - equilibrium.density * phi)
            )
            self.assertAlmostEqual(equilibrium.density, rhs, places=10)

        at_fold = scalar_equilibria(capacity, T0, g, phi, eta)
        self.assertEqual(len(at_fold), 1)
        self.assertAlmostEqual(at_fold[0].density, n_saddle_node(T0, g, phi), places=7)
        self.assertAlmostEqual(at_fold[0].fixed_point_slope, 1.0, places=7)
        self.assertFalse(at_fold[0].stable)

        above = scalar_equilibria(1.05 * capacity, T0, g, phi, eta)
        self.assertEqual(above, ())

    def test_degenerate_zero_burden_has_only_physical_root(self) -> None:
        equilibria = scalar_equilibria(
            arrival_rate=0.3, T0=4.0, g=0.0, phi=0.2, eta=0.5
        )
        self.assertEqual(len(equilibria), 1)
        self.assertAlmostEqual(equilibria[0].density, 0.6, places=12)
        self.assertTrue(equilibria[0].stable)

    def test_speed_wrapper_matches_scalar_formula(self) -> None:
        speed = np.array([0.5, 1.0, 1.5])
        length, burden, chi, gamma, A, tau_r, b, eta = (
            20.0,
            1.2,
            0.3,
            0.8,
            2.0,
            0.15,
            1.1,
            0.4,
        )
        computed = fold_capacity_at_speed(
            speed, length, burden, chi, gamma, A, tau_r, b, eta
        )
        expected = np.array(
            [
                lambda_saddle_node(
                    length / value,
                    burden,
                    offspring_probability(value, chi, gamma, A, tau_r, b),
                    eta,
                )
                for value in speed
            ]
        )
        np.testing.assert_allclose(computed, expected, rtol=1e-13, atol=0.0)

    def test_density_limit_selects_occupancy_or_fold_boundary(self) -> None:
        parameters = dict(
            v=1.0,
            path_length=20.0,
            g=1.2,
            chi=0.3,
            gamma=0.8,
            A=2.0,
            tau_r=0.15,
            b=1.1,
            eta=0.4,
        )
        phi = offspring_probability(
            parameters["v"],
            parameters["chi"],
            parameters["gamma"],
            parameters["A"],
            parameters["tau_r"],
            parameters["b"],
        )
        fold_density = n_saddle_node(
            parameters["path_length"] / parameters["v"], parameters["g"], phi
        )
        fold_capacity = fold_capacity_at_speed(**parameters)

        occupancy_limited = capacity_with_density_limit(
            **parameters, density_limit=0.5 * fold_density
        )
        self.assertFalse(occupancy_limited.fold_limited)
        self.assertAlmostEqual(
            occupancy_limited.arrival_capacity,
            mu_of_n(
                0.5 * fold_density,
                parameters["path_length"] / parameters["v"],
                parameters["g"],
                phi,
                parameters["eta"],
            ),
            places=12,
        )

        cascade_limited = capacity_with_density_limit(
            **parameters, density_limit=2.0 * fold_density
        )
        self.assertTrue(cascade_limited.fold_limited)
        self.assertAlmostEqual(cascade_limited.arrival_capacity, fold_capacity, places=12)
        self.assertAlmostEqual(cascade_limited.operating_density, fold_density, places=12)

    def test_zero_burden_capacity_is_occupancy_limited(self) -> None:
        result = capacity_with_density_limit(
            v=1.0,
            path_length=10.0,
            g=0.0,
            chi=0.5,
            gamma=1.0,
            A=0.0,
            tau_r=0.2,
            b=1.0,
            eta=0.5,
            density_limit=4.0,
        )
        self.assertFalse(result.fold_limited)
        self.assertTrue(math.isinf(result.fold_density))
        self.assertAlmostEqual(result.arrival_capacity, 0.8, places=12)


class CascadeCertificateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(41)

    def _random_subcritical_matrix(self, size: int) -> np.ndarray:
        matrix = self.rng.random((size, size))
        rho = spectral_radius(matrix)
        target = self.rng.uniform(0.05, 0.95)
        return matrix * target / rho

    def test_lp_matches_direct_resolvent(self) -> None:
        for _ in range(300):
            size = int(self.rng.integers(2, 9))
            B = self._random_subcritical_matrix(size)
            direct_budget, direct_certificate = cascade_budget_direct(B)
            result = cascade_budget_lp(B)
            self.assertTrue(result.feasible, result.solver_status)
            self.assertIsNotNone(result.budget)
            self.assertIsNotNone(result.certificate)
            self.assertLess(result.minimum_residual, 1e-6)
            self.assertGreaterEqual(result.minimum_residual, -1e-8)
            self.assertAlmostEqual(result.budget, direct_budget, places=7)
            np.testing.assert_allclose(
                result.certificate, direct_certificate, rtol=1e-7, atol=1e-8
            )

    def test_lp_infeasible_at_and_above_criticality(self) -> None:
        critical_reducible = np.array([[1.0, 0.0], [0.0, 0.2]])
        supercritical = np.array([[0.2, 1.2], [0.8, 0.1]])
        for B in (critical_reducible, supercritical):
            result = cascade_budget_lp(B)
            self.assertGreaterEqual(spectral_radius(B), 1.0)
            self.assertFalse(result.feasible)

    def test_near_critical_certificate_residual(self) -> None:
        B = np.array([[0.999, 0.0], [0.001, 0.4]])
        result = cascade_budget_lp(B)
        self.assertTrue(result.feasible, result.solver_status)
        self.assertGreater(result.budget, 999.0)
        self.assertGreaterEqual(result.minimum_residual, -1e-7)

    def test_chain_has_zero_spectral_radius_and_large_budget(self) -> None:
        size, beta = 32, 0.99
        B = chain_matrix(size, beta)
        expected = sum(beta**power for power in range(size))
        result = cascade_budget_lp(B)
        self.assertAlmostEqual(spectral_radius(B), 0.0, places=14)
        self.assertTrue(result.feasible)
        self.assertAlmostEqual(result.budget, expected, places=7)
        self.assertAlmostEqual(result.budget, 27.5019664, places=6)

    def test_fanout_exposes_non_normal_amplification(self) -> None:
        size, beta = 50, 0.3
        B = fanout_matrix(size, beta)
        expected = 1.0 + (size - 1) * beta
        result = cascade_budget_lp(B)
        self.assertAlmostEqual(spectral_radius(B), 0.0, places=14)
        self.assertTrue(result.feasible)
        self.assertAlmostEqual(result.budget, expected, places=7)

    def test_conditioned_certificate_bound_is_valid_but_loose(self) -> None:
        size, beta = 12, 0.8
        B = chain_matrix(size, beta)
        exact, _ = cascade_budget_direct(B)
        bound = conditioned_bound(1.0 - beta, np.ones(size))
        self.assertLessEqual(exact, bound)
        self.assertGreater(bound, exact)

    def test_finite_depth_matches_geometric_sum(self) -> None:
        B = np.array([[0.8]])
        depth = 12
        expected = sum(0.8**generation for generation in range(depth + 1))
        budget, vector = truncated_cascade_budget(B, depth)
        loss = truncated_cascade_loss(B, [1.0], [1.0], depth)
        self.assertAlmostEqual(budget, expected, places=13)
        self.assertAlmostEqual(vector[0], expected, places=13)
        self.assertAlmostEqual(loss, expected, places=13)

    def test_finite_depth_exists_for_supercritical_matrix(self) -> None:
        B = np.array([[1.2]])
        budget, _ = truncated_cascade_budget(B, max_generation=5)
        self.assertTrue(np.isfinite(budget))
        self.assertGreater(budget, 6.0)

    def test_chain_truncation_equals_infinite_resolvent_at_nilpotency_depth(self) -> None:
        B = chain_matrix(32, 0.99)
        infinite, _ = cascade_budget_direct(B)
        finite, _ = truncated_cascade_budget(B, max_generation=31)
        self.assertAlmostEqual(finite, infinite, places=12)


class GeneralFoldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(53)

    def test_general_loss_derivative_matches_finite_difference(self) -> None:
        for _ in range(100):
            size = int(self.rng.integers(2, 7))
            base = self.rng.random((size, size)) * 0.05
            direction = self.rng.random((size, size)) * 0.02
            density = self.rng.uniform(0.1, 1.0)
            B = base + density * direction
            if spectral_radius(B) >= 0.8:
                continue
            primary = self.rng.random(size)
            primary /= primary.sum()
            severity = self.rng.uniform(0.1, 3.0, size=size)
            analytic = cascade_loss_derivative(B, direction, primary, severity)
            step = 1e-6
            upper = mean_cascade_loss(B + step * direction, primary, severity)
            lower = mean_cascade_loss(B - step * direction, primary, severity)
            numeric = (upper - lower) / (2.0 * step)
            self.assertLess(abs(analytic - numeric) / max(1.0, abs(numeric)), 2e-7)

    def test_general_formula_reduces_to_scalar_formula(self) -> None:
        T0, g, phi, eta = 7.0, 1.3, 0.18, 0.4
        probability = 0.25
        severity = np.array([g / probability])
        density = n_saddle_node(T0, g, phi)
        B = np.array([[density * phi]])
        dB_dn = np.array([[phi]])
        primary = np.array([1.0])
        loss = mean_cascade_loss(B, primary, severity)
        derivative = cascade_loss_derivative(B, dB_dn, primary, severity)
        margin = fold_margin_general(density, T0, probability, loss, derivative)
        self.assertAlmostEqual(margin, 0.0, places=10)
        left, right = fold_elasticity_identity(
            density, T0, probability, loss, derivative
        )
        self.assertAlmostEqual(left, right, places=10)
        self.assertAlmostEqual(density * phi, r_saddle_node(T0, g), places=10)

    def test_general_metrics_use_reference_layer(self) -> None:
        B = np.array([[0.1, 0.2], [0.05, 0.15]])
        dB_dn = np.array([[0.01, 0.02], [0.005, 0.01]])
        primary = np.array([0.7, 0.3])
        severity = np.array([1.0, 1.5])
        result = general_cascade_metrics(
            B,
            primary,
            severity,
            dB_dn=dB_dn,
            n=2.0,
            T0=5.0,
            primary_probability=0.2,
        )
        self.assertAlmostEqual(result.spectral_radius, spectral_radius(B))
        self.assertGreater(result.worst_case_budget, 1.0)
        self.assertIsNotNone(result.fold_margin)


class SimulatorInvariantTests(unittest.TestCase):
    def test_zero_collision_assertion(self) -> None:
        assert_zero_collisions(0)
        with self.assertRaises(AssertionError):
            assert_zero_collisions(1)


if __name__ == "__main__":
    unittest.main()
