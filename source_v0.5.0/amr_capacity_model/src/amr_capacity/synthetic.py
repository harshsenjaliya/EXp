"""Synthetic continuous-time branching forests with administrative censoring."""

from __future__ import annotations

import heapq
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike

from .simulation import InterventionEvent


def _as_offspring_matrix(values: ArrayLike) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError("offspring_matrix must be a nonempty square matrix")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("offspring_matrix must be finite and nonnegative")
    return matrix.copy()


def simulate_censored_branching_forests(
    offspring_matrix: ArrayLike,
    recovery_rates: ArrayLike,
    observation_horizon: float,
    root_count: int,
    *,
    primary_distribution: ArrayLike | None = None,
    seed: int | None = None,
    max_events_per_root: int = 100_000,
) -> tuple[tuple[InterventionEvent, ...], ...]:
    """Simulate independent right-censored multitype branching forests.

    A type-a individual resolves after an exponential lifetime with rate
    delta[a]. During its active lifetime it produces type-b children as an
    independent Poisson process with rate beta[a, b] = B[a, b] * delta[a].
    Consequently, B is the true next-generation mean matrix. Observation ends
    at a fixed horizon; active individuals are retained with censored=True and
    all births observed before the horizon remain in the forest.
    """

    matrix = _as_offspring_matrix(offspring_matrix)
    type_count = matrix.shape[0]
    recovery = np.asarray(recovery_rates, dtype=float).reshape(-1)
    if recovery.size != type_count:
        raise ValueError("recovery_rates must have one entry per event type")
    if not np.all(np.isfinite(recovery)) or np.any(recovery <= 0.0):
        raise ValueError("recovery_rates must be finite and strictly positive")
    if not np.isfinite(observation_horizon) or observation_horizon <= 0.0:
        raise ValueError("observation_horizon must be finite and positive")
    if (
        isinstance(root_count, (bool, np.bool_))
        or not isinstance(root_count, (int, np.integer))
        or root_count < 1
    ):
        raise ValueError("root_count must be a positive integer")
    if (
        isinstance(max_events_per_root, (bool, np.bool_))
        or not isinstance(max_events_per_root, (int, np.integer))
        or max_events_per_root < 1
    ):
        raise ValueError("max_events_per_root must be a positive integer")

    if primary_distribution is None:
        primary = np.full(type_count, 1.0 / type_count, dtype=float)
    else:
        primary = np.asarray(primary_distribution, dtype=float).reshape(-1)
        if primary.size != type_count:
            raise ValueError(
                "primary_distribution must have one entry per event type"
            )
        if not np.all(np.isfinite(primary)) or np.any(primary < 0.0):
            raise ValueError("primary_distribution must be finite and nonnegative")
        total = float(np.sum(primary))
        if total <= 0.0:
            raise ValueError("primary_distribution must have positive mass")
        primary = primary / total

    rng = np.random.default_rng(seed)
    birth_rates = matrix * recovery[:, None]
    forests: list[tuple[InterventionEvent, ...]] = []

    for root_index in range(int(root_count)):
        root_type = int(rng.choice(type_count, p=primary))
        pending: list[tuple[float, int, int, int | None, int]] = [
            (0.0, 0, root_type, None, 0)
        ]
        records: dict[int, InterventionEvent] = {}
        next_event_id = 1

        while pending:
            start_time, event_id, event_type, parent_id, generation = heapq.heappop(
                pending
            )
            lifetime = float(rng.exponential(1.0 / recovery[event_type]))
            natural_end = start_time + lifetime
            censored = natural_end >= observation_horizon
            end_time = (
                float(observation_horizon) if censored else float(natural_end)
            )
            exposure = end_time - start_time
            if exposure <= 0.0:
                raise AssertionError("synthetic event has nonpositive exposure")

            parent = None if parent_id is None else records.get(parent_id)
            if parent_id is not None and parent is None:
                raise AssertionError("synthetic parent must be processed first")
            records[event_id] = InterventionEvent(
                event_id=event_id,
                robot_id=event_id,
                event_type=event_type,
                start_time=float(start_time),
                end_time=end_time,
                cause="crossing" if parent_id is None else "robot",
                blocker_robot_id=parent_id,
                parent_event_id=parent_id,
                root_primary_event_id=0,
                generation=generation,
                crossing_source_id=root_index if parent_id is None else None,
                severity_loss=exposure,
                minimum_speed=0.0,
                censored=censored,
                start_position=None,
                crossing_id=0 if parent_id is None else None,
            )

            means = birth_rates[event_type] * exposure
            offspring_counts = np.asarray(rng.poisson(means), dtype=int)
            offspring_total = int(np.sum(offspring_counts))
            if next_event_id + offspring_total > int(max_events_per_root):
                raise RuntimeError(
                    "synthetic branching tree exceeded max_events_per_root"
                )
            for child_type, child_count in enumerate(offspring_counts):
                if child_count == 0:
                    continue
                birth_times = np.sort(
                    rng.uniform(start_time, end_time, size=int(child_count))
                )
                for birth_time in birth_times:
                    child_id = next_event_id
                    next_event_id += 1
                    heapq.heappush(
                        pending,
                        (
                            float(birth_time),
                            child_id,
                            int(child_type),
                            event_id,
                            generation + 1,
                        ),
                    )

        forests.append(tuple(records[key] for key in sorted(records)))
    return tuple(forests)
