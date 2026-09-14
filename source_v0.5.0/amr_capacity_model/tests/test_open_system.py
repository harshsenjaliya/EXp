from __future__ import annotations

import unittest

import numpy as np

from amr_capacity.estimation import estimate_direct_branching
from amr_capacity.open_metrics import (
    analyze_arrival_cohort,
    analyze_load_segments,
    analyze_operating_window,
    bootstrap_mean_interval,
    empirical_capacity_bracket,
    estimate_queue_drift,
    little_law_diagnostic,
    summarize_open_counterfactual,
    summarize_rate_replications,
)
from amr_capacity.open_system import (
    CrossingRequest,
    LoadSegment,
    OpenCorridorConfig,
    deterministic_arrival_times,
    piecewise_deterministic_arrival_times,
    piecewise_poisson_arrival_times,
    poisson_arrival_times,
    poisson_crossing_requests,
    poisson_crossing_requests_multi,
    run_open_paired_rollout,
    simulate_open_corridor,
)


def make_open_config(
    *,
    length: float = 30.0,
    speed: float = 1.2,
    dt: float = 0.05,
    crossing_x: float | tuple[float, ...] = 15.0,
) -> OpenCorridorConfig:
    return OpenCorridorConfig(
        corridor_length=length,
        desired_speed=speed,
        acceleration=1.0,
        braking=1.5,
        reaction_time=0.2,
        dt=dt,
        robot_length=0.8,
        safety_margin=0.2,
        sensor_range=8.0,
        crossing_x=crossing_x,
        crossing_half_width=0.4,
    )


class ProcessGenerationTests(unittest.TestCase):
    def test_deterministic_arrivals_exclude_horizon(self) -> None:
        values = deterministic_arrival_times(1.0, 5.0)
        np.testing.assert_array_equal(values, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(deterministic_arrival_times(0.0, 5.0).size, 0)

    def test_poisson_streams_are_local_and_reproducible(self) -> None:
        first = poisson_arrival_times(0.5, 100.0, seed=41)
        second = poisson_arrival_times(0.5, 100.0, seed=41)
        third = poisson_arrival_times(0.5, 100.0, seed=42)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, third))
        self.assertTrue(np.all(np.diff(first) > 0.0))

    def test_piecewise_stream_respects_segment_support(self) -> None:
        segments = (
            LoadSegment(0.0, 50.0, 0.2, "low"),
            LoadSegment(50.0, 100.0, 0.8, "high"),
        )
        values = piecewise_poisson_arrival_times(segments, seed=7)
        self.assertTrue(np.all((0.0 < values) & (values < 100.0)))
        self.assertGreater(np.count_nonzero(values >= 50.0), 0)
        regular = piecewise_deterministic_arrival_times(segments)
        self.assertGreater(regular.size, 0)
        self.assertTrue(np.all(np.diff(regular) > 0.0))
        with self.assertRaises(ValueError):
            piecewise_poisson_arrival_times(
                (LoadSegment(0.0, 60.0, 0.2), LoadSegment(50.0, 80.0, 0.2))
            )

    def test_crossing_request_stream_is_reproducible(self) -> None:
        first = poisson_crossing_requests(
            0.05, 100.0, (2.0, 4.0), seed=9, source_id_start=100
        )
        second = poisson_crossing_requests(
            0.05, 100.0, (2.0, 4.0), seed=9, source_id_start=100
        )
        self.assertEqual(first, second)
        self.assertEqual(len({item.source_id for item in first}), len(first))
        self.assertTrue(all(2.0 <= item.duration <= 4.0 for item in first))

    def test_multi_crossing_stream_is_reproducible_and_typed(self) -> None:
        first = poisson_crossing_requests_multi(
            0.05, 100.0, (2.0, 4.0), 3, seed=9
        )
        second = poisson_crossing_requests_multi(
            0.05, 100.0, (2.0, 4.0), 3, seed=9
        )
        self.assertEqual(first, second)
        self.assertEqual({item.crossing_id for item in first}, {0, 1, 2})
        self.assertEqual(len({item.source_id for item in first}), len(first))


class OpenCorridorInvariantTests(unittest.TestCase):
    def test_multiple_guarded_crossings_preserve_all_invariants(self) -> None:
        config = make_open_config(crossing_x=(5.0, 10.0, 15.0, 20.0, 25.0))
        requests = tuple(
            CrossingRequest(20.0 + 0.5 * index, 4.0, index, index)
            for index in range(config.crossing_count)
        )
        result = simulate_open_corridor(
            config,
            100.0,
            deterministic_arrival_times(0.7, 100.0),
            crossing_requests=requests,
            record_stride=5,
        )
        self.assertEqual(result.mass_balance_residual, 0)
        self.assertEqual(result.collision_count, 0)
        self.assertEqual(result.crossing_violation_count, 0)
        self.assertEqual(
            {item.crossing_id for item in result.crossing_services},
            set(range(config.crossing_count)),
        )
        self.assertGreaterEqual(
            int(np.max(result.trajectory.active_crossing_count)), 2
        )
        primary_events = [event for event in result.events if event.is_primary]
        self.assertTrue(primary_events)
        self.assertTrue(
            all(
                event.crossing_id in range(config.crossing_count)
                and event.start_position is not None
                and 0.0 <= event.start_position <= config.corridor_length
                for event in primary_events
            )
        )

    def test_overlapping_crossing_zones_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            make_open_config(crossing_x=(10.0, 11.0))

    def test_config_rejects_an_inadequate_sensor(self) -> None:
        with self.assertRaises(ValueError):
            OpenCorridorConfig(
                corridor_length=20.0,
                desired_speed=2.0,
                acceleration=1.0,
                braking=1.0,
                reaction_time=0.2,
                dt=0.05,
                robot_length=0.8,
                safety_margin=0.2,
                sensor_range=1.0,
                crossing_x=10.0,
            )

    def test_fifo_completion_and_exact_mass_balance(self) -> None:
        result = simulate_open_corridor(
            make_open_config(length=20.0, crossing_x=10.0),
            50.0,
            [0.0, 5.0, 10.0],
            record_stride=10,
        )
        self.assertEqual(result.mass_balance_residual, 0)
        self.assertEqual(result.total_completions, 3)
        self.assertEqual(
            [item.job_id for item in result.completed_traversals], [0, 1, 2]
        )
        self.assertEqual(result.collision_count, 0)
        self.assertEqual(result.crossing_violation_count, 0)
        self.assertEqual(len(result.events), 0)

    def test_last_substep_arrival_is_not_lost(self) -> None:
        result = simulate_open_corridor(
            make_open_config(length=20.0, crossing_x=10.0),
            10.0,
            [9.99],
            record_stride=20,
        )
        self.assertEqual(result.total_arrivals, 1)
        self.assertEqual(result.total_admissions, 0)
        self.assertEqual(result.final_queue_length, 1)
        self.assertEqual(result.mass_balance_residual, 0)

    def test_unsafe_request_is_delayed_until_the_robot_clears(self) -> None:
        config = make_open_config(length=20.0, crossing_x=6.0)
        result = simulate_open_corridor(
            config,
            30.0,
            [0.0],
            crossing_requests=[CrossingRequest(4.5, 2.0, 17)],
        )
        service = result.crossing_services[0]
        self.assertEqual(service.status, "completed")
        self.assertIsNotNone(service.wait_time)
        self.assertGreater(service.wait_time, 1.0)
        self.assertEqual(result.crossing_violation_count, 0)
        self.assertEqual(result.collision_count, 0)

    def test_crossing_service_queue_reports_horizon_censoring(self) -> None:
        config = make_open_config(length=20.0, crossing_x=10.0)
        result = simulate_open_corridor(
            config,
            10.0,
            [],
            crossing_requests=(
                CrossingRequest(1.0, 20.0, 4),
                CrossingRequest(2.0, 2.0, 5),
            ),
        )
        self.assertEqual(
            [item.status for item in result.crossing_services],
            ["active_at_end", "unserved"],
        )
        self.assertEqual(result.crossing_services[0].occupied_time, 9.0)
        self.assertIsNone(result.crossing_services[1].activation_time)

    def test_duplicate_crossing_source_ids_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            simulate_open_corridor(
                make_open_config(),
                20.0,
                [],
                crossing_requests=(
                    CrossingRequest(1.0, 2.0, 7),
                    CrossingRequest(5.0, 2.0, 7),
                ),
            )

    def test_finite_physical_fleet_is_reused_and_conserved(self) -> None:
        base = make_open_config()
        config = OpenCorridorConfig(
            corridor_length=base.corridor_length,
            desired_speed=base.desired_speed,
            acceleration=base.acceleration,
            braking=base.braking,
            reaction_time=base.reaction_time,
            dt=base.dt,
            robot_length=base.robot_length,
            safety_margin=base.safety_margin,
            sensor_range=base.sensor_range,
            crossing_x=base.crossing_x,
            fleet_size=3,
            empty_return_time=10.0,
        )
        result = simulate_open_corridor(
            config,
            200.0,
            deterministic_arrival_times(1.0, 200.0),
            record_stride=5,
        )
        self.assertEqual(result.fleet_balance_residual, 0)
        self.assertEqual(
            set(item.robot_id for item in result.completed_traversals),
            {0, 1, 2},
        )
        np.testing.assert_array_equal(
            result.trajectory.work_in_process
            + result.trajectory.available_robots
            + result.trajectory.returning_robots,
            np.full(result.trajectory.time.size, 3),
        )
        metrics = analyze_operating_window(result, 100.0, 200.0)
        self.assertIsNotNone(metrics.mean_fleet_utilization)
        self.assertLessEqual(metrics.mean_fleet_utilization, 1.0)
        self.assertGreater(result.final_queue_length, 0)

    def test_invalid_finite_fleet_configuration_is_rejected(self) -> None:
        base = make_open_config()
        with self.assertRaises(ValueError):
            OpenCorridorConfig(
                corridor_length=base.corridor_length,
                desired_speed=base.desired_speed,
                acceleration=base.acceleration,
                braking=base.braking,
                reaction_time=base.reaction_time,
                dt=base.dt,
                robot_length=base.robot_length,
                safety_margin=base.safety_margin,
                sensor_range=base.sensor_range,
                crossing_x=base.crossing_x,
                fleet_size=0,
            )
        with self.assertRaises(ValueError):
            OpenCorridorConfig(
                corridor_length=base.corridor_length,
                desired_speed=base.desired_speed,
                acceleration=base.acceleration,
                braking=base.braking,
                reaction_time=base.reaction_time,
                dt=base.dt,
                robot_length=base.robot_length,
                safety_margin=base.safety_margin,
                sensor_range=base.sensor_range,
                crossing_x=base.crossing_x,
                fleet_size=True,
            )

    def test_two_phase_gate_prevents_dense_stream_starvation(self) -> None:
        config = make_open_config()
        result = simulate_open_corridor(
            config,
            80.0,
            deterministic_arrival_times(0.7, 80.0),
            crossing_requests=[CrossingRequest(20.0, 5.0, 3)],
            record_stride=5,
        )
        service = result.crossing_services[0]
        self.assertEqual(service.status, "completed")
        self.assertGreater(service.activation_time, service.request_time)
        self.assertTrue(np.any(result.trajectory.crossing_reserved))
        causal = [event for event in result.events if event.is_causal]
        self.assertGreaterEqual(len(causal), 5)
        self.assertEqual(causal[0].generation, 0)
        estimate = estimate_direct_branching(result.events)
        self.assertGreater(estimate.matrix[0, 0], 0.0)
        self.assertEqual(result.collision_count, 0)
        self.assertEqual(result.crossing_violation_count, 0)

    def test_zero_collisions_over_load_and_disturbance_sweep(self) -> None:
        config = make_open_config()
        for index, rate in enumerate((0.2, 0.6, 1.0)):
            arrivals = poisson_arrival_times(rate, 80.0, seed=100 + index)
            requests = poisson_crossing_requests(
                0.04, 80.0, (2.0, 4.0), seed=200 + index, start_time=5.0
            )
            result = simulate_open_corridor(
                config,
                80.0,
                arrivals,
                crossing_requests=requests,
                record_stride=20,
            )
            self.assertEqual(result.mass_balance_residual, 0)
            self.assertEqual(result.collision_count, 0)
            self.assertEqual(result.crossing_violation_count, 0)

    def test_paired_rollout_uses_common_jobs_and_increases_latency(self) -> None:
        config = make_open_config()
        arrivals = deterministic_arrival_times(0.7, 100.0)
        requests = (CrossingRequest(20.0, 5.0, 1), CrossingRequest(55.0, 5.0, 2))
        pair = run_open_paired_rollout(
            config, 100.0, arrivals, requests, record_stride=5
        )
        self.assertEqual(pair.treated.mass_balance_residual, 0)
        self.assertEqual(pair.control.mass_balance_residual, 0)
        self.assertEqual(pair.completion_loss, pair.terminal_backlog_increase)
        self.assertGreater(pair.completion_loss, 0)
        self.assertGreater(pair.severity_increase, 0.0)
        self.assertGreater(pair.mean_paired_sojourn_increase, 0.0)
        summary = summarize_open_counterfactual(pair)
        self.assertGreater(summary.queue_time_increase, 0.0)
        self.assertEqual(summary.completion_loss, pair.completion_loss)

    def test_open_event_statistics_converge_with_time_step(self) -> None:
        severity = []
        latency = []
        event_counts = []
        for dt in (0.1, 0.05, 0.025):
            config = make_open_config(dt=dt)
            pair = run_open_paired_rollout(
                config,
                100.0,
                deterministic_arrival_times(0.5, 100.0),
                (CrossingRequest(20.0, 5.0, 1), CrossingRequest(55.0, 5.0, 2)),
                record_stride=max(1, round(0.25 / dt)),
            )
            severity.append(pair.treated.total_severity_loss)
            latency.append(pair.mean_paired_sojourn_increase)
            event_counts.append(len([event for event in pair.treated.events if event.is_causal]))
        self.assertEqual(event_counts, [12, 12, 12])
        self.assertLess(abs(severity[2] - severity[1]), abs(severity[1] - severity[0]))
        self.assertLess(abs(latency[2] - latency[1]), abs(latency[1] - latency[0]))


class OpenSystemStatisticalTests(unittest.TestCase):
    def test_hac_drift_recovers_an_exact_linear_trend(self) -> None:
        time = np.linspace(0.0, 100.0, 501)
        queue = 3.0 + 0.2 * time
        estimate = estimate_queue_drift(time, queue)
        self.assertAlmostEqual(estimate.slope, 0.2, places=12)
        self.assertLess(estimate.standard_error, 1e-12)
        self.assertGreater(estimate.lower_confidence, 0.19)

    def test_operating_window_separates_stable_and_overloaded_loads(self) -> None:
        config = make_open_config()
        low = simulate_open_corridor(
            config,
            200.0,
            deterministic_arrival_times(0.4, 200.0),
            record_stride=5,
        )
        high = simulate_open_corridor(
            config,
            200.0,
            deterministic_arrival_times(0.9, 200.0),
            record_stride=5,
        )
        low_metrics = analyze_operating_window(
            low, 100.0, 200.0, requested_arrival_rate=0.4
        )
        high_metrics = analyze_operating_window(
            high, 100.0, 200.0, requested_arrival_rate=0.9
        )
        self.assertEqual(low_metrics.stability, "no_growth_detected")
        self.assertEqual(high_metrics.stability, "growth_detected")
        self.assertLess(low_metrics.drift.upper_confidence, 0.002)
        self.assertGreater(high_metrics.drift.lower_confidence, 0.1)

    def test_fixed_followup_cohort_and_little_law_diagnostic(self) -> None:
        config = make_open_config()
        result = simulate_open_corridor(
            config,
            250.0,
            deterministic_arrival_times(0.4, 250.0),
            record_stride=5,
        )
        metrics = analyze_operating_window(result, 100.0, 240.0)
        cohort = analyze_arrival_cohort(result, 100.0, 150.0, 50.0)
        diagnostic = little_law_diagnostic(metrics)
        self.assertEqual(cohort.completion_probability, 1.0)
        self.assertGreater(cohort.restricted_mean_sojourn_time, 20.0)
        self.assertLess(abs(diagnostic.relative_residual), 0.01)

    def test_replication_summary_and_capacity_bracket(self) -> None:
        config = make_open_config()
        by_rate = {}
        for rate in (0.4, 0.9):
            records = []
            for seed in (1, 2):
                # Regular arrivals isolate the classifier test; stochastic
                # replications are exercised by the process-generator tests.
                result = simulate_open_corridor(
                    config,
                    160.0,
                    deterministic_arrival_times(rate, 160.0),
                    record_stride=5,
                )
                records.append(
                    analyze_operating_window(
                        result, 80.0, 160.0, requested_arrival_rate=rate
                    )
                )
            by_rate[rate] = summarize_rate_replications(
                rate, records, resamples=200, seed=seed
            )
        bracket = empirical_capacity_bracket(by_rate.values())
        self.assertTrue(bracket.resolved)
        self.assertEqual(bracket.largest_no_growth_rate, 0.4)
        self.assertEqual(bracket.smallest_growth_rate, 0.9)

    def test_bootstrap_and_segment_analysis_are_deterministic(self) -> None:
        first = bootstrap_mean_interval([1.0, 2.0, 3.0], resamples=500, seed=8)
        second = bootstrap_mean_interval([1.0, 2.0, 3.0], resamples=500, seed=8)
        self.assertEqual(first, second)

        config = make_open_config()
        segments = (
            LoadSegment(0.0, 80.0, 0.3, "up-low"),
            LoadSegment(80.0, 160.0, 0.8, "up-high"),
        )
        arrivals = piecewise_poisson_arrival_times(segments, seed=2)
        result = simulate_open_corridor(
            config, 160.0, arrivals, record_stride=5
        )
        summaries = analyze_load_segments(result, segments, discard_fraction=0.5)
        self.assertEqual(len(summaries), 2)
        self.assertEqual(summaries[0].retained_start_time, 40.0)
        self.assertEqual(summaries[1].retained_start_time, 120.0)


if __name__ == "__main__":
    unittest.main()
