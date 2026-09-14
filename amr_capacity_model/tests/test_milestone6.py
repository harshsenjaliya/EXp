from __future__ import annotations

import unittest

import numpy as np

from amr_capacity.branching_exposure import (
    CensoredBranchingRollout,
    TimedBranchingEvent,
    bootstrap_censored_multitype_branching,
    estimate_censored_multitype_branching,
    estimate_naive_observed_tree,
    exponential_kernel_exposure,
    profile_log_likelihood,
    scale_branching_matrix,
    simulate_censored_branching_experiment,
)
from amr_capacity.kinematic_maps import (
    DifferentialDriveLimits,
    KinematicMap,
    RouteObstacle,
    UnsafeObstacleActivationError,
    assert_static_clearance,
    plan_kinematic_speed_profile,
    simulate_route_obstacles,
    standard_map_catalogue,
    straight_route,
)


class ExposureLikelihoodTests(unittest.TestCase):
    @staticmethod
    def _analytic_rollout() -> CensoredBranchingRollout:
        return CensoredBranchingRollout(
            events=(
                TimedBranchingEvent(0, 0, 0.0, None, 0, 0),
                TimedBranchingEvent(1, 1, 1.0, 0, 0, 1),
            ),
            observation_end=2.0,
            requested_horizon=2.0,
        )

    def test_exponential_kernel_exposure_is_stable_and_vectorized(self) -> None:
        self.assertEqual(exponential_kernel_exposure(0.0, 2.0), 0.0)
        self.assertAlmostEqual(
            exponential_kernel_exposure(2.0, 0.5),
            1.0 - np.exp(-1.0),
            places=14,
        )
        np.testing.assert_allclose(
            exponential_kernel_exposure(
                np.array([0.0, 1.0]), np.array([1.0, 2.0])
            ),
            [0.0, 1.0 - np.exp(-2.0)],
        )

    def test_closed_form_mle_uses_fractional_parent_exposure(self) -> None:
        rates = np.array([[1.0, 2.0], [1.5, 0.5]])
        fitted = estimate_censored_multitype_branching(
            (self._analytic_rollout(),), rates
        )
        self.assertAlmostEqual(
            fitted.exposure_mass[0, 1], 1.0 - np.exp(-4.0), places=14
        )
        self.assertAlmostEqual(
            fitted.matrix[0, 1], 1.0 / (1.0 - np.exp(-4.0)), places=14
        )
        self.assertEqual(fitted.direct_child_counts[0, 1], 1)
        self.assertEqual(fitted.parent_counts.tolist(), [1, 1])
        self.assertGreater(fitted.temporally_censored_parent_count, 0)
        self.assertGreaterEqual(
            profile_log_likelihood(fitted.matrix, fitted),
            profile_log_likelihood(fitted.matrix * 1.1, fitted),
        )

    def test_event_ids_are_local_to_each_rollout(self) -> None:
        rollout = self._analytic_rollout()
        duplicate_inside = CensoredBranchingRollout
        with self.assertRaisesRegex(ValueError, "unique"):
            duplicate_inside(
                events=(rollout.events[0], rollout.events[0]),
                observation_end=2.0,
                requested_horizon=2.0,
            )
        pooled = estimate_censored_multitype_branching(
            (rollout, rollout), np.ones((2, 2))
        )
        self.assertEqual(pooled.rollout_count, 2)
        self.assertEqual(pooled.event_count, 4)

    def test_exposure_likelihood_recovers_subcritical_radius(self) -> None:
        base = np.array([[0.25, 0.80], [0.10, 0.30]])
        truth = scale_branching_matrix(base, 0.70)
        rates = np.array([[1.0, 0.7], [0.9, 1.2]])
        rollouts = simulate_censored_branching_experiment(
            truth,
            rates,
            rollout_count=220,
            roots_per_rollout=3,
            horizon=1.8,
            population_cap=300,
            root_type_probabilities=(0.55, 0.45),
            seed=2026091401,
        )
        fitted = estimate_censored_multitype_branching(rollouts, rates)
        self.assertLess(abs(fitted.spectral_radius - 0.70), 0.15)
        self.assertLess(fitted.spectral_radius, 1.0)

    def test_supercritical_truth_is_not_forced_below_one(self) -> None:
        base = np.array([[0.25, 0.80], [0.10, 0.30]])
        truth = scale_branching_matrix(base, 1.24)
        rates = np.array([[1.0, 0.7], [0.9, 1.2]])
        rollouts = simulate_censored_branching_experiment(
            truth,
            rates,
            rollout_count=260,
            roots_per_rollout=3,
            horizon=2.0,
            population_cap=400,
            root_type_probabilities=(0.55, 0.45),
            seed=2026091402,
        )
        exposure = estimate_censored_multitype_branching(rollouts, rates)
        naive = estimate_naive_observed_tree(rollouts, 2)
        self.assertLess(abs(exposure.spectral_radius - 1.24), 0.16)
        self.assertGreater(exposure.spectral_radius, 1.05)
        self.assertLessEqual(naive.spectral_radius, 1.0 + 1e-10)
        self.assertGreater(exposure.spectral_radius, naive.spectral_radius)

        interval = bootstrap_censored_multitype_branching(
            rollouts,
            rates,
            replications=60,
            confidence_level=0.90,
            seed=2026091403,
        )
        self.assertLessEqual(
            interval.spectral_radius_lower, interval.point.spectral_radius
        )
        self.assertGreaterEqual(
            interval.spectral_radius_upper, interval.point.spectral_radius
        )
        self.assertGreater(interval.probability_supercritical, 0.70)


class KinematicMapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.limits = DifferentialDriveLimits()

    def test_catalogue_covers_six_distinct_environment_classes(self) -> None:
        maps = standard_map_catalogue()
        self.assertEqual(len(maps), 6)
        self.assertEqual(
            {item.name for item in maps},
            {
                "straight_crossing",
                "l_turn",
                "s_curve",
                "two_to_one_merge",
                "four_way_intersection",
                "warehouse_grid",
            },
        )
        self.assertGreaterEqual(
            sum(len(item.conflict_zones) for item in maps), 2
        )
        for map_spec in maps:
            self.assertGreaterEqual(assert_static_clearance(map_spec, self.limits), 0.0)

    def test_nominal_profile_respects_turn_and_wheel_envelopes(self) -> None:
        l_turn = next(
            item for item in standard_map_catalogue() if item.name == "l_turn"
        )
        profile = plan_kinematic_speed_profile(
            l_turn.routes[0],
            self.limits,
            map_speed_limit=l_turn.speed_limit,
        )
        self.assertGreater(profile.turning_penalty, 0.0)
        self.assertLess(
            np.max(np.abs(np.diff(l_turn.routes[0].curvature))), 0.08
        )
        self.assertLessEqual(
            profile.max_abs_yaw_rate, self.limits.max_yaw_rate * 1.001
        )
        self.assertLessEqual(
            profile.max_lateral_acceleration,
            self.limits.max_lateral_acceleration * 1.001,
        )
        self.assertLessEqual(
            profile.max_abs_wheel_speed, self.limits.max_wheel_speed * 1.001
        )
        self.assertLessEqual(
            profile.max_abs_yaw_acceleration,
            self.limits.max_yaw_acceleration * 1.15,
        )
        self.assertTrue(np.all(profile.speed <= profile.curvature_envelope + 1e-9))

    def test_straight_path_has_negligible_turning_penalty(self) -> None:
        path = straight_route(length=24.0)
        profile = plan_kinematic_speed_profile(path, self.limits)
        self.assertAlmostEqual(profile.turning_penalty, 0.0, places=10)
        self.assertLess(np.max(np.abs(path.curvature)), 1e-9)

    def test_long_stationary_obstacle_causes_safe_delay(self) -> None:
        map_spec = next(
            item
            for item in standard_map_catalogue()
            if item.name == "straight_crossing"
        )
        route = map_spec.routes[0]
        profile = plan_kinematic_speed_profile(
            route, self.limits, map_speed_limit=map_spec.speed_limit
        )
        obstacle = RouteObstacle(
            "pallet",
            "unexpected_stationary",
            0.42 * route.length,
            0.0,
            profile.nominal_time,
            guarded=False,
        )
        result = simulate_route_obstacles(
            map_spec,
            route.name,
            self.limits,
            (obstacle,),
            dt=0.04,
            sensor_range=10.0,
        )
        self.assertEqual(result.collision_count, 0)
        self.assertEqual(result.obstacle_violation_count, 0)
        self.assertGreater(result.delay, 1.0)
        self.assertGreater(result.severity_weighted_loss, 0.0)
        self.assertGreaterEqual(result.minimum_stopping_margin, -1e-7)
        self.assertLessEqual(
            result.max_abs_yaw_rate, self.limits.max_yaw_rate * 1.001
        )
        self.assertLessEqual(
            result.max_lateral_acceleration,
            self.limits.max_lateral_acceleration * 1.001,
        )
        self.assertLessEqual(
            result.max_abs_wheel_speed, self.limits.max_wheel_speed * 1.001
        )
        self.assertLessEqual(
            result.max_abs_yaw_acceleration,
            self.limits.max_yaw_acceleration * 1.02,
        )
        self.assertEqual(len(result.activations), 1)

    def test_unsafe_unguarded_activation_fails_closed(self) -> None:
        path = straight_route("main", length=12.0)
        map_spec = KinematicMap("test", (path,))
        obstacle = RouteObstacle(
            "too_close",
            "unexpected_stationary",
            0.40,
            0.0,
            2.0,
            half_width=0.10,
            guarded=False,
        )
        with self.assertRaises(UnsafeObstacleActivationError):
            simulate_route_obstacles(
                map_spec,
                path.name,
                self.limits,
                (obstacle,),
                dt=0.02,
                sensor_range=10.0,
            )


if __name__ == "__main__":
    unittest.main()
