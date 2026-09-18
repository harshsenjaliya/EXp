from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from amr_capacity.censored_branching import (
    bootstrap_exposure_aware_branching,
    estimate_exposure_aware_branching,
    pool_exposure_aware_branching_events,
    select_observed_root_cohorts,
)
from amr_capacity.estimation import (
    estimate_direct_branching,
    pool_direct_branching_events,
    select_complete_root_cohorts,
)
from amr_capacity.simulation import InterventionEvent
from amr_capacity.synthetic import simulate_censored_branching_forests


def _event(
    event_id: int,
    event_type: int,
    start: float,
    end: float,
    *,
    parent: int | None,
    generation: int,
    censored: bool,
) -> InterventionEvent:
    return InterventionEvent(
        event_id=event_id,
        robot_id=event_id,
        event_type=event_type,
        start_time=start,
        end_time=end,
        cause="crossing" if parent is None else "robot",
        blocker_robot_id=parent,
        parent_event_id=parent,
        root_primary_event_id=0,
        generation=generation,
        crossing_source_id=0 if parent is None else None,
        severity_loss=end - start,
        minimum_speed=0.0,
        censored=censored,
        crossing_id=0 if parent is None else None,
    )


def _partly_censored_forest() -> tuple[InterventionEvent, ...]:
    return (
        _event(0, 0, 0.0, 4.0, parent=None, generation=0, censored=True),
        _event(1, 1, 1.0, 2.0, parent=0, generation=1, censored=False),
        _event(2, 1, 3.0, 4.0, parent=0, generation=1, censored=True),
        _event(3, 0, 1.5, 2.5, parent=1, generation=2, censored=False),
    )


class ExposureAwareLikelihoodTests(unittest.TestCase):
    def test_censored_exposure_and_observed_births_are_retained(self) -> None:
        events = _partly_censored_forest()
        estimate = estimate_exposure_aware_branching(
            events, number_of_types=2
        )
        np.testing.assert_allclose(estimate.exposure_time, [5.0, 2.0])
        np.testing.assert_array_equal(estimate.completed_parent_counts, [1, 1])
        np.testing.assert_array_equal(estimate.censored_parent_counts, [1, 1])
        np.testing.assert_array_equal(
            estimate.direct_child_counts,
            [[0, 2], [1, 0]],
        )
        np.testing.assert_allclose(
            estimate.birth_rate_matrix,
            [[0.0, 0.4], [0.5, 0.0]],
        )
        np.testing.assert_allclose(estimate.recovery_rates, [0.2, 0.5])
        np.testing.assert_allclose(estimate.matrix, [[0.0, 2.0], [1.0, 0.0]])
        self.assertAlmostEqual(estimate.spectral_radius, np.sqrt(2.0))
        self.assertEqual(estimate.status, "identified")
        self.assertTrue(np.isfinite(estimate.log_likelihood_up_to_constant))

        complete_parent = estimate_direct_branching(events, number_of_types=2)
        self.assertEqual(complete_parent.spectral_radius, 0.0)

    def test_observed_selector_does_not_condition_on_complete_trees(self) -> None:
        events = _partly_censored_forest()
        observed = select_observed_root_cohorts(events, 0.0, 0.5)
        complete = select_complete_root_cohorts(events, 0.0, 0.5)
        self.assertEqual(observed.selected_primary_roots, 1)
        self.assertEqual(observed.complete_primary_roots, 0)
        self.assertEqual(observed.unresolved_primary_roots, 1)
        self.assertEqual(len(observed.events), 4)
        self.assertEqual(complete.events, ())

    def test_unresolved_active_row_fails_closed(self) -> None:
        root = _event(
            0, 0, 0.0, 2.0, parent=None, generation=0, censored=True
        )
        estimate = estimate_exposure_aware_branching((root,))
        self.assertIsNone(estimate.spectral_radius)
        self.assertTrue(np.isnan(estimate.matrix[0, 0]))
        self.assertEqual(
            estimate.status,
            "unidentified_active_type_without_resolution",
        )

    def test_rollout_local_identifiers_pool_but_concatenation_is_rejected(self) -> None:
        first = _partly_censored_forest()
        second = _partly_censored_forest()
        pooled = pool_exposure_aware_branching_events(
            (first, second), number_of_types=2
        )
        np.testing.assert_allclose(pooled.matrix, [[0.0, 2.0], [1.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "rollout-local"):
            estimate_exposure_aware_branching(
                first + second, number_of_types=2
            )

    def test_child_must_start_during_observed_parent_exposure(self) -> None:
        events = list(_partly_censored_forest())
        events[1] = replace(events[1], start_time=4.5, end_time=5.0)
        with self.assertRaisesRegex(ValueError, "parent's observed exposure"):
            estimate_exposure_aware_branching(events, number_of_types=2)


class SyntheticBranchingValidationTests(unittest.TestCase):
    def test_seeded_simulation_is_deterministic(self) -> None:
        first = simulate_censored_branching_forests(
            [[0.8]], [1.0], 2.0, 20, seed=77
        )
        second = simulate_censored_branching_forests(
            [[0.8]], [1.0], 2.0, 20, seed=77
        )
        self.assertEqual(first, second)

    def test_known_subcritical_and_supercritical_scalar_laws(self) -> None:
        estimates = {}
        for index, truth in enumerate((0.65, 1.35)):
            forests = simulate_censored_branching_forests(
                [[truth]],
                [1.0],
                4.0,
                1_200,
                seed=20260914 + index,
            )
            estimate = pool_exposure_aware_branching_events(forests)
            self.assertIsNotNone(estimate.spectral_radius)
            self.assertLess(abs(estimate.spectral_radius - truth), 0.08)
            estimates[truth] = estimate

            complete_forests = []
            for forest in forests:
                selection = select_complete_root_cohorts(
                    forest, 0.0, 0.5
                )
                if selection.events:
                    complete_forests.append(selection.events)
            complete_estimate = pool_direct_branching_events(complete_forests)
            self.assertLessEqual(complete_estimate.spectral_radius, 1.0 + 1e-12)
        self.assertLess(estimates[0.65].spectral_radius, 1.0)
        self.assertGreater(estimates[1.35].spectral_radius, 1.0)

    def test_multitype_matrix_is_recovered(self) -> None:
        truth = np.array([[0.70, 0.70], [0.20, 0.85]])
        forests = simulate_censored_branching_forests(
            truth,
            [1.0, 0.8],
            4.0,
            1_500,
            primary_distribution=[0.5, 0.5],
            seed=314159,
        )
        estimate = pool_exposure_aware_branching_events(
            forests, number_of_types=2
        )
        self.assertIsNotNone(estimate.spectral_radius)
        np.testing.assert_allclose(estimate.matrix, truth, atol=0.12, rtol=0.0)
        self.assertGreater(estimate.spectral_radius, 1.0)

    def test_forest_bootstrap_is_seeded_and_non_degenerate(self) -> None:
        forests = simulate_censored_branching_forests(
            [[1.20]], [1.0], 3.0, 350, seed=1234
        )
        first = bootstrap_exposure_aware_branching(
            forests, resamples=200, seed=99
        )
        second = bootstrap_exposure_aware_branching(
            forests, resamples=200, seed=99
        )
        self.assertEqual(first.valid_resamples, 200)
        self.assertEqual(first.valid_resamples, second.valid_resamples)
        np.testing.assert_array_equal(first.matrix_lower, second.matrix_lower)
        np.testing.assert_array_equal(first.matrix_upper, second.matrix_upper)
        self.assertLess(
            first.spectral_radius_lower,
            first.point_estimate.spectral_radius,
        )
        self.assertGreater(
            first.spectral_radius_upper,
            first.point_estimate.spectral_radius,
        )


if __name__ == "__main__":
    unittest.main()
