# Milestone 4: Structural Audit, Censoring Repair, and Burden Coverage

> **Post-audit correction.** Three rows below have \(\widehat\phi=1\), zero
> resolved training cascades, and six fleet-saturated cascades. Because
> \(M_N(1)=N\), their zero held-out error is algebraic and is now labeled
> `descriptive_not_predictive`. Across the 14 non-vacuous speed and density
> rows, mean absolute error is 14.03% for the plain scalar estimator and 13.73%
> for the boundary-finite estimator. This milestone therefore does **not**
> establish a predictive finite-boundary advantage. See Milestone 5 for the
> corrected benchmark and multi-type audit.

## Verdict

The simulator is now a credible research instrument, but the saddle-node and
capacity-collapse hypothesis is **not yet validated**. This milestone repairs
three structural defects in the earlier evidence rather than optimizing a
headline:

1. the original open data occupied only \(x\in[0,0.00521]\), where the
   parameter-free fold curve is nearly flat;
2. scalar in-sample resolvent agreement in a single-parent event forest is an
   algebraic identity;
3. the former near-critical finite-depth result was partly caused by temporal
   right censoring at the 40 s simulation horizon.

The implementation now detects each condition explicitly and refuses the
corresponding claim.

## 1. Burden-domain repair

For one crossing, primary severity can accrue at no more than one robot-second
per second. At saturated control rate \(\Theta_0\), this gives the empirical
design ceiling

\[
x=\frac{g}{T_0}\le \frac{1}{\Theta_0T_0}.
\]

With \(\Theta_0\approx0.675\) jobs/s and \(T_0=25\) s, the ceiling is about
0.059. Increasing one request stream therefore cannot cover \(x\in[0.1,1]\);
it merely keeps the same exogenous gate continuously occupied.

The open simulator now supports up to 16 independently queued crossing
locations, each with its own two-phase reservation gate. Every active and
reserved location participates in the stopping-distance and protected-zone
assertions. The stress design keeps every individual service stream subloaded:
the largest per-gate offered load is 0.30.

The quick profile spans the following pooled burden values:

| Regime | Arrival rate | Crossings | Per-gate load | Pooled (x) | Treated/control rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original-like baseline | 0.80 | 1 | 0.075 | 0.00467 | 0.929 |
| Distributed | 0.80 | 12 | 0.150 | 0.114 | 0.417 |
| Distributed | 0.80 | 16 | 0.150 | 0.135 | 0.266 |
| Mid-load coverage | 0.35 | 16 | 0.150 | 0.230 | 0.456 |
| Long-occupancy coverage | 0.20 | 16 | 0.250 | 0.496 | 0.297 |
| Extreme coverage | 0.10 | 16 | 0.250 | 0.911 | 0.617 |

These last three rows are domain-coverage conditions, not saturated-capacity
tests. A large \(x\) at low arrival rate can coexist with weak coupling because
few robots are simultaneously in transit. The fold model depends on both
burden and endogenous density.

Increasing the 16-gate request rate from 0.05 to 0.10 requests/s per gate
reduced pooled \(x\) from 0.135 to 0.113. Upstream blocks starved downstream
crossings of robot exposure. Disturbance rate is therefore not a monotone
control for realized burden in a serial corridor. This is a useful topology
effect, but it is not a fold.

## 2. Chain-forest identity and held-out validation

For a completely observed single-parent forest with \(E\) causal events and
\(P\) primary roots, there are \(E-P\) edges. Hence

\[
\widehat B=\frac{E-P}{E}
\quad\Longrightarrow\quad
\frac{1}{1-\widehat B}=\frac{E}{P}.
\]

The new `ChainForestDiagnostic` records the parent count, edge count, in-sample
resolvent, measured progeny, residual, and an `algebraic_identity` flag. Such a
row is always labeled `descriptive_not_predictive`.

The new stress experiment fits \(\widehat B\) on three replications and scores
progeny on three different replications. Quick-profile held-out relative error
ranges from 2.1% to 37.1% across the ten regimes. This is a real prediction
test, although it tests transfer of finite chain length only. A serial lane can
fan out over time when one stopped robot successively affects several
followers, but the corrected data contain too few such parents for robust
supercritical inference. See Milestone 5.

The original quick open profile has too few completed fixed-follow-up root
cohorts: all six load conditions are now reported as `indeterminate` rather
than silently reusing an in-sample identity. The 30-replication profile is
required for an open-load held-out estimate.

## 3. Temporal censoring and the finite-boundary detector

At speeds of 1.6 m/s and above, several descendants were still active at the
old 40 s horizon. Omitting those descendants from the held-out numerator while
using their incoming edges in \(\widehat B\) made a depth truncation appear to
regularize the resolvent. The old 92% to 37% and 97% to 33% error reductions
must not be cited as finite-fleet validation.

Closed rollouts now run for 120 s. Every cascade in the generated sweep either
resolves or reaches every robot. For chain size \(L_r\) in a fleet of \(N\), the
boundary-censored propagation estimator is

\[
S=\sum_r(L_r-1),\qquad
F=\sum_r\mathbf 1\{L_r<N\text{ and resolved}\},\qquad
\widehat\phi=\frac{S}{S+F}.
\]

A cascade reaching \(N\) contributes \(N-1\) successes and no artificial
failure because another robot does not exist. A temporally censored cascade
with \(L_r<N\) is indeterminate and is never treated as a failure. Prediction
uses

\[
M_N(\widehat\phi)=\sum_{k=0}^{N-1}\widehat\phi^k.
\]

The corrected preliminary results are:

| Condition | Training boundary status | \(\widehat\phi\) | Infinite prediction | Finite prediction | Held-out unique robots | Finite error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| \(v=1.8\), \(N=20\) | 4/6 saturated | 0.981 | 52.5 | 16.76 | 17.33 | 3.29% |
| \(v=2.0\), \(N=20\) | 6/6 saturated | 1.000 | undefined | 20.00 | 20.00 | 0.00% |
| \(v=1.2\), \(N=32\) | 6/6 saturated | 1.000 | undefined | 32.00 | 32.00 | 0.00% |
| \(v=1.2\), \(N=33\) | 6/6 saturated | 1.000 | undefined | 33.00 | 33.00 | 0.00% |

The last three rows are not prediction tests. With no resolved training
cascade, \(\widehat\phi=1\) and \(M_N(\widehat\phi)=N\); the validation cascades
also contain every robot, so prediction and measurement equal \(N\) by
construction.

At \(v=1.8\), the fair deployable baseline is the plain scalar estimate:
18.17 predicted versus 17.33 measured, or 4.81% error. The boundary-finite
estimate has 3.29% error. The 52.5 boundary-infinite value has 202.88% error,
but it is not a meaningful operational baseline because no user would combine
the boundary-censored estimate with an untruncated infinite resolvent.

Across all 14 non-vacuous rows, the plain and boundary-finite estimators have
mean absolute errors of 14.03% and 13.73%, respectively. Using a one-percentage-
point practical tie band, the finite estimator wins four rows, loses three, and
ties seven. The corrected conclusion is that the present scalar sweep does not
distinguish them.

## 4. The missing faster-is-slower plot

Fleet circulation rate was added to every closed rollout. The treated rate is
monotone over the entire stopping-distance-feasible speed sweep:

| Speed (m/s) | Treated rate | Control rate |
| ---: | ---: | ---: |
| 0.6 | 0.1997 | 0.2000 |
| 1.5 | 0.4941 | 0.5000 |
| 1.8 | 0.5885 | 0.6000 |
| 2.0 | 0.6421 | 0.6667 |

The disturbance penalty grows, but throughput does not turn over. Therefore
faster-is-slower, collapse, and a critical speed are not observed in this
closed experiment. One isolated crossing event over 120 s is not a stationary
disturbance process; recurrent guarded disturbances are needed for that test.

The density sweep stops at \(N=33\). At 1.2 m/s the stopping-distance headway
is 1.768 m, while a 60 m loop gives only 1.765 m at \(N=34\). Running 36 or 40
robots would violate the model's own initial safety invariant and is rejected.

## 5. Claim boundary

Supported by the quick generated evidence:

- collision-free, mass-balanced single- and multi-crossing simulation;
- burden-domain coverage from approximately 0.005 to 0.91 without an
  individually overloaded crossing stream;
- automatic detection of the scalar in-sample chain identity;
- independent held-out chain-length prediction;
- automatic detection of the vacuous \(M_N(1)=N\) boundary identity;
- non-monotone realized burden caused by downstream starvation.

Not supported:

- a saddle-node fold or parameter-free fold-curve validation;
- capacity collapse or faster-is-slower turnover;
- a predictive advantage for the boundary-finite chain estimator;
- supercritical branching in the single-lane event forest;
- non-normal multi-type network evidence;
- industrial external validity.

## 6. Correct next milestone

The multi-crossing corridor can support a useful multi-type *subcritical*
experiment once events are typed by their start position. It must pool
rollout-level sufficient statistics because event IDs restart in every run.
Assigning every descendant its root gate is a diagonal negative control, not a
directed topology model. A genuine merge or fan-out mechanism remains necessary
if the corrected estimator cannot produce robust parent fan-out and a
supercritical regime. Fold testing remains separate: locate a capacity boundary
from queue stability first, then evaluate \(R\) versus \(g/T_0\) at that
independently located boundary.

## Generated artifacts

- `corridor_rollout_runs.csv`: raw closed rollout rates and cohort state;
- `corridor_speed_sweep.csv` and `corridor_density_sweep.csv`: naïve and
  boundary-censored estimates side by side;
- `corridor_speed_throughput.png`: the negative faster-is-slower result;
- `finite_depth_truncation.png`: non-vacuous estimator benchmark and identity
  exclusions;
- `burden_stress_runs.csv`, `burden_stress_summary.csv`, and
  `burden_stress_heldout.csv`;
- `burden_domain_coverage.png` and `burden_stress_manifest.json`;
- `open_heldout_branching.csv` and algebraic-identity fields in
  `open_attribution_diagnostics.csv`;
- `claim_audit.json`: machine-readable supported/unsupported claim ledger.
