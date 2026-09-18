from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from amr_capacity.capacity_theory import d_stop
from amr_capacity.estimation import (
    diagnose_branching_forest,
    diagnose_chain_forest,
    diagnose_multitype_forest_identity,
    estimate_direct_branching,
    estimate_horizon_branching,
    pool_direct_branching,
    pool_direct_branching_events,
    pool_horizon_branching,
    retype_events_by_minimum_speed,
    retype_events_by_position,
    select_complete_root_cohorts,
    summarize_paired_rollout,
    validate_branching_out_of_sample,
    validate_finite_chain_event_sets,
)
from amr_capacity.geometry import RectangularLoop
from amr_capacity.simulation import (
    CorridorConfig,
    CrossingEvent,
    InterventionEvent,
    UnsafeCrossingScheduleError,
    run_paired_rollout,
    simulate_corridor,
    uniform_initial_state,
)


def make_config(
    *,
    speed: float = 1.2,
    crossing_s: float = 7.0,
    dt: float = 0.02,
) -> CorridorConfig:
    return CorridorConfig(
        loop=RectangularLoop(20.0, 10.0),
        desired_speed=speed,
        acceleration=1.0,
        braking=1.5,
        reaction_time=0.2,
        dt=dt,
        robot_length=0.8,
        safety_margin=0.2,
        sensor_range=8.0,
        crossing_s=crossing_s,
        crossing_half_width=0.4,
    )


class GeometryTests(unittest.TestCase):
    def test_rectangle_mapping_at_segment_starts(self) -> None:
        loop = RectangularLoop(20.0, 10.0)
        s = np.array([0.0, 20.0, 30.0, 50.0, 60.0])
        x, y, heading = loop.to_xy_heading(s)
        np.testing.assert_allclose(x, [0.0, 20.0, 20.0, 0.0, 0.0])
        np.testing.assert_allclose(y, [0.0, 0.0, 10.0, 10.0, 0.0])
        np.testing.assert_allclose(
            heading, [0.0, np.pi / 2.0, np.pi, -np.pi / 2.0, 0.0]
        )

    def test_scalar_mapping(self) -> None:
        x, y, heading = RectangularLoop(4.0, 2.0).to_xy_heading(5.0)
        self.assertEqual((x, y), (4.0, 1.0))
        self.assertAlmostEqual(heading, np.pi / 2.0)


class SimulatorValidationTests(unittest.TestCase):
    def test_sensor_must_cover_stopping_distance(self) -> None:
        with self.assertRaises(ValueError):
            CorridorConfig(
                loop=RectangularLoop(10.0, 5.0),
                desired_speed=2.0,
                acceleration=1.0,
                braking=1.0,
                reaction_time=0.2,
                dt=0.05,
                robot_length=0.8,
                safety_margin=0.2,
                sensor_range=1.0,
                crossing_s=5.0,
            )

    def test_uniform_state_rejects_unsafe_headway(self) -> None:
        config = make_config(speed=1.5)
        with self.assertRaises(ValueError):
            uniform_initial_state(config, 40)

    def test_one_robot_ring_is_supported(self) -> None:
        config = make_config()
        result = simulate_corridor(config, 2.0, robot_count=1)
        self.assertEqual(result.collision_count, 0)
        self.assertEqual(len(result.events), 0)
        self.assertAlmostEqual(result.total_traversals, 2.4 / 60.0, places=12)

    def test_duration_and_record_stride(self) -> None:
        config = make_config()
        result = simulate_corridor(
            config, 2.0, robot_count=5, record_stride=10
        )
        self.assertEqual(result.trajectory.time.size, 11)
        self.assertEqual(result.trajectory.s.shape, (11, 5))
        with self.assertRaises(ValueError):
            simulate_corridor(config, 2.001, robot_count=5)
        with self.assertRaises(ValueError):
            simulate_corridor(
                config,
                2.0,
                crossing_events=[CrossingEvent(1.001, 0.001, 5)],
                robot_count=5,
            )

    def test_initial_offset_must_be_finite(self) -> None:
        with self.assertRaises(ValueError):
            uniform_initial_state(make_config(), 5, offset=np.nan)

    def test_unsafe_crossing_activation_is_rejected(self) -> None:
        speed, start = 1.2, 4.0
        crossing_s = speed * start + 0.5
        config = make_config(speed=speed, crossing_s=crossing_s)
        required = d_stop(
            speed,
            config.crossing_clearance,
            config.reaction_time,
            config.braking,
        )
        self.assertGreater(required, 0.5)
        with self.assertRaises(UnsafeCrossingScheduleError):
            simulate_corridor(
                config,
                10.0,
                crossing_events=[CrossingEvent(start, 2.0, 1)],
                robot_count=10,
            )


class PairedCascadeTests(unittest.TestCase):
    @staticmethod
    def _synthetic_chain(
        size: int, *, fleet_size: int, censored_last: bool = False
    ) -> tuple[InterventionEvent, ...]:
        if not 1 <= size <= fleet_size:
            raise ValueError("invalid synthetic chain size")
        return tuple(
            InterventionEvent(
                event_id=index,
                robot_id=index,
                event_type=0,
                start_time=float(index),
                end_time=float(index + 1),
                cause="crossing" if index == 0 else "robot",
                blocker_robot_id=None if index == 0 else index - 1,
                parent_event_id=None if index == 0 else index - 1,
                root_primary_event_id=0,
                generation=index,
                crossing_source_id=0 if index == 0 else None,
                severity_loss=0.5,
                minimum_speed=0.0,
                censored=censored_last and index == size - 1,
            )
            for index in range(size)
        )

    def setUp(self) -> None:
        speed, start, target_gap = 1.2, 4.0, 2.2
        self.config = make_config(
            speed=speed, crossing_s=speed * start + target_gap
        )
        self.crossing = CrossingEvent(start, 6.0, 7)

    def test_control_is_intervention_free(self) -> None:
        control = simulate_corridor(self.config, 35.0, robot_count=20)
        self.assertEqual(len(control.events), 0)
        self.assertEqual(control.collision_count, 0)
        self.assertEqual(control.crossing_violation_count, 0)
        np.testing.assert_allclose(
            control.final_state.speed, self.config.desired_speed
        )

    def test_contact_attribution_builds_generation_chain(self) -> None:
        pair = run_paired_rollout(
            self.config,
            35.0,
            [self.crossing],
            robot_count=20,
            record_stride=5,
        )
        causal = [event for event in pair.treated.events if event.is_causal]
        self.assertEqual(len(causal), 6)
        self.assertEqual([event.generation for event in causal], list(range(6)))
        self.assertTrue(causal[0].is_primary)
        for parent, child in zip(causal[:-1], causal[1:], strict=True):
            self.assertEqual(child.parent_event_id, parent.event_id)
            self.assertEqual(child.blocker_robot_id, parent.robot_id)
        self.assertEqual(pair.treated.collision_count, 0)
        self.assertEqual(pair.treated.crossing_violation_count, 0)
        self.assertGreater(pair.attributable_severity, 0.0)
        self.assertGreater(pair.traversal_loss, 0.0)

        route = pair.treated.trajectory.s
        speed = pair.treated.trajectory.speed
        leader = np.roll(np.arange(pair.treated.robot_count), -1)
        gaps = np.mod(
            route[:, leader] - route, pair.treated.config.route_length
        )
        required = d_stop(
            speed,
            pair.treated.config.robot_clearance,
            pair.treated.config.reaction_time,
            pair.treated.config.braking,
        )
        self.assertGreaterEqual(np.min(gaps - required), -1e-9)

    def test_direct_estimator_does_not_mix_generations(self) -> None:
        pair = run_paired_rollout(
            self.config, 35.0, [self.crossing], robot_count=20
        )
        direct = estimate_direct_branching(pair.treated.events)
        self.assertAlmostEqual(direct.matrix[0, 0], 5.0 / 6.0, places=12)
        short_horizon = estimate_horizon_branching(pair.treated.events, 2.0)
        long_horizon = estimate_horizon_branching(pair.treated.events, 8.0)
        self.assertGreater(long_horizon.matrix[0, 0], direct.matrix[0, 0])
        self.assertGreater(long_horizon.matrix[0, 0], short_horizon.matrix[0, 0])
        identity = diagnose_chain_forest(pair.treated.events)
        self.assertTrue(identity.rooted_single_parent_forest)
        self.assertTrue(identity.algebraic_identity)
        self.assertAlmostEqual(
            identity.in_sample_resolvent,
            identity.measured_progeny_per_primary,
        )

    def test_concatenated_rollout_ids_are_rejected(self) -> None:
        first = self._synthetic_chain(2, fleet_size=4)
        second = self._synthetic_chain(3, fleet_size=4)
        with self.assertRaisesRegex(ValueError, "rollout-local"):
            estimate_direct_branching(first + second)
        pooled = pool_direct_branching_events((first, second))
        self.assertAlmostEqual(pooled.matrix[0, 0], 3.0 / 5.0)
        typed = tuple(
            tuple(replace(event, event_type=event.event_id % 2) for event in chain)
            for chain in (first, second)
        )
        identity = diagnose_multitype_forest_identity(
            typed, number_of_types=2
        )
        self.assertTrue(identity.algebraic_identity)
        self.assertTrue(identity.complete_forest_rho_bound_verified)
        self.assertLessEqual(identity.training_spectral_radius, 1.0)
        self.assertAlmostEqual(identity.training_mean_progeny, 2.5)
        self.assertAlmostEqual(identity.primary_mix_resolvent, 2.5)

    def test_complete_root_selection_and_structural_diagnostic(self) -> None:
        chain = self._synthetic_chain(3, fleet_size=4)
        selection = select_complete_root_cohorts(chain, 0.0, 0.5)
        self.assertEqual(selection.complete_primary_roots, 1)
        self.assertEqual(selection.unresolved_primary_roots, 0)
        self.assertEqual(selection.complete_root_sizes, (3,))
        self.assertEqual(selection.complete_root_max_generations, (2,))
        structure = diagnose_branching_forest(selection.events)
        self.assertEqual(structure.fanout_parent_events, 0)
        self.assertEqual(structure.maximum_direct_children, 1)
        self.assertEqual(structure.maximum_generation, 2)

    def test_severity_and_spatial_retyping(self) -> None:
        chain = self._synthetic_chain(3, fleet_size=4)
        positioned = tuple(
            replace(
                event,
                minimum_speed=0.0 if index == 0 else 0.4,
                start_position=float(index) + 0.25,
            )
            for index, event in enumerate(chain)
        )
        severity = retype_events_by_minimum_speed(positioned)
        spatial = retype_events_by_position(positioned, (0.0, 1.0, 2.0, 3.0))
        self.assertEqual([event.event_type for event in severity], [0, 1, 1])
        self.assertEqual([event.event_type for event in spatial], [0, 1, 2])

    def test_paired_rollout_is_bitwise_deterministic(self) -> None:
        first = run_paired_rollout(
            self.config, 35.0, [self.crossing], robot_count=20, record_stride=10
        )
        second = run_paired_rollout(
            self.config, 35.0, [self.crossing], robot_count=20, record_stride=10
        )
        np.testing.assert_array_equal(
            first.treated.trajectory.s, second.treated.trajectory.s
        )
        self.assertEqual(first.treated.events, second.treated.events)

    def test_summary_uses_control_exposure(self) -> None:
        pair = run_paired_rollout(
            self.config, 35.0, [self.crossing], robot_count=20
        )
        summary = summarize_paired_rollout(pair)
        self.assertEqual(summary.primary_events, 1)
        self.assertEqual(summary.causal_events, 6)
        self.assertAlmostEqual(
            summary.primary_events_per_traversal,
            1.0 / pair.control.total_traversals,
            places=12,
        )
        self.assertAlmostEqual(summary.measured_total_progeny_per_primary, 6.0)
        self.assertAlmostEqual(summary.predicted_total_progeny_per_primary, 6.0)

    def test_held_out_validation_and_pooled_estimators(self) -> None:
        durations = (3.5, 4.5, 5.5, 6.5, 4.0, 5.0, 6.0, 7.0)
        pairs = tuple(
            run_paired_rollout(
                self.config,
                35.0,
                [CrossingEvent(4.0, duration, source_id=index)],
                robot_count=20,
                record_stride=20,
            )
            for index, duration in enumerate(durations)
        )
        direct = pool_direct_branching(pairs[:4])
        horizon = pool_horizon_branching(pairs[:4], 8.0)
        self.assertGreater(horizon.matrix[0, 0], direct.matrix[0, 0])

        held_out = validate_branching_out_of_sample(
            pairs[:4], pairs[4:], max_generation=19
        )
        expected_events = tuple(
            event
            for pair in pairs[4:]
            for event in pair.treated.events
            if event.is_causal and not event.censored
        )
        self.assertEqual(
            held_out.validation_primary_events,
            sum(event.is_primary for event in expected_events),
        )
        self.assertEqual(
            held_out.validation_causal_events,
            len(expected_events),
        )
        self.assertIsNotNone(held_out.predicted_total_progeny_per_primary)
        self.assertIsNotNone(held_out.relative_error)
        self.assertIsNotNone(
            held_out.truncated_predicted_total_progeny_per_primary
        )
        self.assertIsNotNone(held_out.truncated_relative_error)
        self.assertGreaterEqual(held_out.relative_error, 0.0)

    def test_finite_chain_treats_fleet_boundary_as_right_censoring(self) -> None:
        fleet_size = 4
        saturated = self._synthetic_chain(
            fleet_size, fleet_size=fleet_size, censored_last=True
        )
        validation = validate_finite_chain_event_sets(
            (saturated, saturated),
            (saturated,),
            fleet_size,
        )
        self.assertEqual(validation.training.propagation_probability, 1.0)
        self.assertIsNone(validation.infinite_prediction)
        self.assertEqual(validation.finite_prediction, float(fleet_size))
        self.assertEqual(validation.measured_unique_robots_per_primary, fleet_size)
        self.assertEqual(validation.finite_relative_error, 0.0)
        self.assertTrue(validation.training.algebraic_boundary_identity)
        self.assertEqual(validation.validation_status, "descriptive_not_predictive")

    def test_finite_chain_rejects_temporal_censoring_below_boundary(self) -> None:
        censored = self._synthetic_chain(2, fleet_size=4, censored_last=True)
        resolved = self._synthetic_chain(2, fleet_size=4)
        with self.assertRaisesRegex(ValueError, "right-censored"):
            validate_finite_chain_event_sets((censored,), (resolved,), 4)

    def test_zero_collisions_across_speed_and_density_sweep(self) -> None:
        start = 4.0
        for speed in (0.6, 0.9, 1.2, 1.5):
            for robot_count in (8, 12, 16, 20):
                headway = 60.0 / robot_count
                provisional = make_config(speed=speed)
                required = d_stop(
                    speed,
                    provisional.crossing_clearance,
                    provisional.reaction_time,
                    provisional.braking,
                )
                target_gap = min(headway - 0.2, required + 0.35)
                config = make_config(
                    speed=speed, crossing_s=speed * start + target_gap
                )
                pair = run_paired_rollout(
                    config,
                    25.0,
                    [CrossingEvent(start, 4.0, 1)],
                    robot_count=robot_count,
                    record_stride=20,
                )
                self.assertEqual(pair.treated.collision_count, 0)
                self.assertEqual(pair.treated.crossing_violation_count, 0)

    def test_event_statistics_converge_with_time_step(self) -> None:
        results = []
        for dt in (0.04, 0.02, 0.01):
            config = CorridorConfig(
                loop=RectangularLoop(20.0, 10.0),
                desired_speed=1.2,
                acceleration=1.0,
                braking=1.5,
                reaction_time=0.24,
                dt=dt,
                robot_length=0.8,
                safety_margin=0.2,
                sensor_range=8.0,
                crossing_s=7.0,
                crossing_half_width=0.4,
            )
            pair = run_paired_rollout(
                config,
                36.0,
                [CrossingEvent(4.0, 6.0, 1)],
                robot_count=20,
                record_stride=max(1, round(0.2 / dt)),
            )
            causal = tuple(event for event in pair.treated.events if event.is_causal)
            results.append(
                (
                    len(causal),
                    pair.attributable_severity,
                    pair.traversal_loss,
                    np.array([event.start_time for event in causal]),
                )
            )
        self.assertEqual([item[0] for item in results], [6, 6, 6])
        severity = np.array([item[1] for item in results])
        traversal = np.array([item[2] for item in results])
        self.assertLess(
            abs(severity[2] - severity[1]),
            0.6 * abs(severity[1] - severity[0]),
        )
        self.assertLess(
            abs(traversal[2] - traversal[1]),
            0.6 * abs(traversal[1] - traversal[0]),
        )
        self.assertLess(abs(severity[2] - severity[1]) / severity[2], 0.01)
        self.assertLess(
            np.max(np.ptp(np.vstack([item[3] for item in results]), axis=0)),
            0.4,
        )


if __name__ == "__main__":
    unittest.main()
