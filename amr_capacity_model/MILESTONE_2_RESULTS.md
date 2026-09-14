# Milestone 2: Preliminary Corridor Results

> **Superseded evidence notice.** The 40 s horizon right-censored active
> descendants near criticality. Consequently, the finite-depth error reductions
> reported below are partly censoring artifacts and must not be cited. Milestone
> 4 reruns the experiment for 120 s and introduces a boundary-censored chain
> estimator that distinguishes temporal censoring from reaching all (N) robots.
> See `MILESTONE_4_RESULTS.md` for the corrected results.

## Status

These results validate the software and expose modeling behavior. They are not
yet publication evidence: each condition currently has 12 paired rollouts,
split into six training and six held-out runs, with no confidence intervals.

All successful runs satisfy both the physical non-overlap checks and the full
post-step stopping-distance invariant. The test suite contains 44 tests,
including 64 speed-density safety conditions and integration-step convergence.

## Experimental protocol

- Rectangular one-way loop: 60 m perimeter.
- Fixed step: 0.02 s.
- Treated rollout: one guarded crossing occupation.
- Control rollout: identical initial state with no crossing occupation.
- Crossing duration: uniformly sampled from 3–7 s using seed 20260914.
- Initial phase: varied while the physical crossing remains fixed.
- Direct parent: proximate active blocking robot at child-event onset.
- Training/validation split: first six versus final six rollouts per condition.

## Attribution result

At 1.0 m/s and 20 robots, the direct estimate is \(B=0.714\), whereas an
8-second time-window estimate is \(1.286\). At 1.5 m/s the corresponding values
are \(0.940\) and \(3.160\). The horizon estimate counts later generations
under several parents and can falsely classify a finite contact chain as
supercritical.

This is the strongest result of the milestone because it directly justifies
generation-level attribution. The full horizon-sensitivity table is stored in
`outputs/corridor_horizon_sensitivity.csv`.

## Speed response

At 20 robots, direct offspring rises from \(0.400\) at 0.6 m/s to \(0.940\) at
1.5 m/s. Held-out mean progeny rises from 2.0 to 8.67 interventions per primary.

The infinite resolvent is reasonable away from the near-critical finite-fleet
regime but predicts 16.67 interventions per primary at 1.5 m/s. A depth-19
truncation predicts 11.83, reducing relative error from 92.3% to 36.5%. The
remaining error is retained and signals finite-size dependence, sampling
uncertainty, and non-identical cascade-duration distributions.

## Density response

At 1.2 m/s, measured direct offspring increases with robot count:

| Robots | Headway (m) | Direct \(B\) | \(B/N\) | Held-out progeny |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 7.500 | 0.250 | 0.03125 | 1.17 |
| 12 | 5.000 | 0.500 | 0.04167 | 1.83 |
| 16 | 3.750 | 0.667 | 0.04167 | 3.17 |
| 20 | 3.000 | 0.806 | 0.04032 | 5.17 |
| 24 | 2.500 | 0.893 | 0.03720 | 8.83 |
| 28 | 2.143 | 0.961 | 0.03430 | 12.83 |

The middle range supports an approximate \(B\approx N\phi\) scaling with
\(\phi\approx0.04\). The decline in \(B/N\) at high density is consistent with
the one-proximate-child ceiling \(B\le1\). This makes the homogeneous closure a
testable local approximation, not a global identity.

## Numerical correction discovered during implementation

An instantaneous safe-speed cap produced no collisions but violated the full
stopping-distance invariant by approximately 0.0235 m during a dense cascade.
The implementation now solves a discrete one-step condition that includes
movement during the integration interval. The minimum observed post-correction
margin in the diagnostic case is approximately \(4.3\times10^{-11}\) m.

Halving the integration step halves the severity and traversal error while
preserving cascade event count, consistent with first-order numerical
convergence.

## What can and cannot be claimed

Supported now:

- the simulator implements deterministic, collision-free contact cascades;
- direct and time-window estimators measure materially different objects;
- speed and density increase propagation in this controlled corridor;
- finite-depth accounting matters near the finite-fleet boundary.

Not supported yet:

- a real open-system saddle-node or hysteresis;
- universal exponents or metastability;
- multi-type non-normal cascade validation;
- confidence-calibrated capacity predictions;
- transfer to a specific industrial AMR fleet.

## Next scientific milestone

The next implementation should add an open corridor with queued task arrivals
and finite entry capacity. That experiment must measure \(n(t)\), throughput,
queue growth, and relaxation time under slow and fast arrival-rate ramps. A
separate fan-out or merge layout is required to create and estimate genuinely
multi-type, non-normal \(B\) matrices.
