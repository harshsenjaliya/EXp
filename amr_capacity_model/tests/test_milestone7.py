from __future__ import annotations

import unittest

import numpy as np

from amr_capacity.branching_exposure import (
    CensoredBranchingRollout,
    TimedBranchingEvent,
    audit_branching_timestamps,
    classify_branching_regime,
    estimate_joint_censored_multitype_branching,
    intervention_events_to_censored_rollout,
    poisson_extinction_probability,
    scale_branching_matrix,
    simulate_censored_branching_experiment,
    simulate_depleting_branching_experiment,
    simulate_extinct_poisson_trees,
)
from amr_capacity.simulation import InterventionEvent
from amr_capacity.kinematic_maps import standard_map_catalogue, standard_robot_catalogue
from amr_capacity.network_simulation import (
    NetworkConfig,
    NetworkDisturbance,
    deterministic_network_jobs,
    extend_network_map,
    simulate_route_network,
)


class JointRecoveryInferenceTests(unittest.TestCase):
    def test_joint_continuous_fit_recovers_matrix_and_recovery(self) -> None:
        base = np.array([[0.25, 0.80], [0.10, 0.30]])
        truth = scale_branching_matrix(base, 0.90)
        gamma = np.array([[1.0, 0.70], [0.90, 1.20]])
        rollouts = simulate_censored_branching_experiment(
            truth,
            gamma,
            rollout_count=500,
            roots_per_rollout=3,
            horizon=3.0,
            population_cap=500,
            root_type_probabilities=(0.55, 0.45),
            seed=2026091701,
        )
        fitted = estimate_joint_censored_multitype_branching(
            rollouts,
            2,
            observation_model="continuous",
            recovery_rate_bounds=(0.20, 4.0),
        )
        self.assertLess(abs(fitted.spectral_radius - 0.90), 0.12)
        self.assertLess(np.max(np.abs(fitted.recovery_rates - gamma)), 0.35)
        self.assertFalse(np.any(fitted.recovery_at_bound))

    def test_grouped_likelihood_handles_same_step_children(self) -> None:
        rollout = CensoredBranchingRollout(
            events=(
                TimedBranchingEvent(0, 0, 0.0, None, 0, 0),
                TimedBranchingEvent(1, 0, 0.0, 0, 0, 1),
                TimedBranchingEvent(2, 0, 0.1, 0, 0, 1),
            ),
            observation_end=1.0,
            requested_horizon=1.0,
        )
        grouped = estimate_joint_censored_multitype_branching(
            (rollout,),
            1,
            observation_model="grouped",
            time_step=0.1,
            recovery_rate_bounds=(0.05, 50.0),
        )
        continuous = estimate_joint_censored_multitype_branching(
            (rollout,),
            1,
            observation_model="continuous",
            recovery_rate_bounds=(0.05, 50.0),
        )
        self.assertEqual(grouped.zero_lag_edge_count, 1)
        self.assertTrue(np.isfinite(grouped.log_likelihood))
        self.assertFalse(grouped.recovery_at_bound[0, 0])
        self.assertGreater(
            continuous.recovery_rates[0, 0], grouped.recovery_rates[0, 0] * 1.5
        )

    def test_adapter_uses_active_event_duration_as_parent_exposure(self) -> None:
        events = (
            InterventionEvent(
                0, 4, 0, 1.0, 2.0, "crossing", None, None, 0, 0, 8,
                0.4, 0.0, False, 5.0, 0,
            ),
            InterventionEvent(
                1, 5, 0, 1.5, 1.8, "robot", 4, 0, 0, 1, None,
                0.2, 0.0, False, 4.0, None,
            ),
        )
        rollout = intervention_events_to_censored_rollout(
            events, observation_end=10.0, exposure_policy="active_interval"
        )
        registered = intervention_events_to_censored_rollout(
            events, observation_end=10.0
        )
        self.assertIsNone(registered.events[0].exposure_end_time)
        self.assertEqual(rollout.events[0].exposure_end_time, 2.0)
        self.assertEqual(rollout.events[1].exposure_end_time, 1.8)
        audit = audit_branching_timestamps((rollout,), 0.5)
        self.assertEqual(audit.edge_count, 1)
        self.assertEqual(audit.zero_lag_edge_count, 0)

    def test_timestamp_audit_rejects_exact_time_interpretation_with_ties(self) -> None:
        rollout = CensoredBranchingRollout(
            events=(
                TimedBranchingEvent(0, 0, 0.0, None, 0, 0),
                TimedBranchingEvent(1, 0, 0.0, 0, 0, 1),
            ),
            observation_end=1.0,
            requested_horizon=1.0,
        )
        audit = audit_branching_timestamps((rollout,), 0.05)
        self.assertEqual(audit.zero_lag_fraction, 1.0)
        self.assertFalse(audit.exact_time_likelihood_supported)

    def test_three_way_regime_classification_is_fail_closed(self) -> None:
        self.assertEqual(
            classify_branching_regime(0.7, 0.95), "certified_subcritical"
        )
        self.assertEqual(classify_branching_regime(0.9, 1.1), "indeterminate")
        self.assertEqual(
            classify_branching_regime(1.05, 1.3), "certified_supercritical"
        )


class FinitePopulationControlTests(unittest.TestCase):
    def test_genuine_depletion_reaches_the_population_boundary(self) -> None:
        rollouts = simulate_depleting_branching_experiment(
            np.array([[2.2]]),
            np.array([[1.0]]),
            rollout_count=80,
            roots_per_rollout=1,
            horizon=20.0,
            population_by_type=(25,),
            seed=2026091702,
        )
        capped = [item for item in rollouts if item.population_cap_reached]
        self.assertGreater(len(capped), 20)
        self.assertTrue(all(len(item.events) == 25 for item in capped))

    def test_extinction_only_poisson_trees_follow_the_dual_mean(self) -> None:
        mean = 1.3
        q = poisson_extinction_probability(mean)
        children, parents, retained = simulate_extinct_poisson_trees(
            mean,
            rollout_count=3_000,
            population_guard=2_000,
            seed=2026091703,
        )
        self.assertGreater(retained, 1_400)
        self.assertLess(abs(children / parents - mean * q), 0.055)


class LoadedNetworkTests(unittest.TestCase):
    @staticmethod
    def _run(map_name: str):
        base = next(
            item for item in standard_map_catalogue() if item.name == map_name
        )
        map_spec = extend_network_map(base, 55.0)
        limits = dict(standard_robot_catalogue())["standard"]
        config = NetworkConfig(map_spec, limits, fleet_size=30, dt=0.05)
        jobs = deterministic_network_jobs(
            {route.name: 0.40 for route in map_spec.routes}, 70.0
        )
        zone = map_spec.conflict_zones[0]
        route = map_spec.routes[0]
        location = float(
            route.s[int(np.argmin(np.hypot(route.x - zone.x, route.y - zone.y)))]
        )
        disturbances = (
            NetworkDisturbance(0, route.name, location, 15.0, 6.0),
        )
        return simulate_route_network(
            config,
            70.0,
            jobs,
            disturbances=disturbances,
            record_stride=20,
        )

    def test_merge_runs_multiple_robots_with_hard_invariants(self) -> None:
        result = self._run("two_to_one_merge")
        self.assertGreater(result.total_admissions, 20)
        self.assertGreater(result.total_completions, 0)
        self.assertGreater(sum(event.is_causal for event in result.events), 0)
        self.assertEqual(result.collision_count, 0)
        self.assertEqual(result.conflict_violation_count, 0)
        self.assertEqual(result.disturbance_violation_count, 0)
        self.assertEqual(result.mass_balance_residual, 0)
        self.assertEqual(result.fleet_balance_residual, 0)
        self.assertLessEqual(
            result.max_abs_yaw_rate,
            result.config.limits.max_yaw_rate * 1.02,
        )
        self.assertLessEqual(
            result.max_abs_wheel_speed,
            result.config.limits.max_wheel_speed * 1.02,
        )

    def test_intersection_enforces_single_owner_reservation(self) -> None:
        result = self._run("four_way_intersection")
        self.assertGreater(result.total_admissions, 20)
        self.assertEqual(result.collision_count, 0)
        self.assertEqual(result.conflict_violation_count, 0)
        self.assertLessEqual(
            int(np.max(result.trajectory.occupied_conflict_zones)), 1
        )

    def test_extended_map_preserves_central_conflict_geometry(self) -> None:
        base = next(
            item
            for item in standard_map_catalogue()
            if item.name == "two_to_one_merge"
        )
        extended = extend_network_map(base, 120.0)
        self.assertTrue(all(route.length >= 119.99 for route in extended.routes))
        self.assertEqual(extended.conflict_zones, base.conflict_zones)


if __name__ == "__main__":
    unittest.main()
