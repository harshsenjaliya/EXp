# Milestone 3: Open-System Research Simulator

> **Superseded claim notice.** Later audits found that the single-parent
> resolvent comparison is algebraic in-sample, that several apparent
> finite-boundary successes are vacuous at \(\widehat\phi=1\), and that typed
> training-primary-mix agreement has the same identity. Use
> `MILESTONE_5_RESULTS.md` for the current claim boundary.

> **Post-audit correction.** The single-gate burden occupied only
> (x\in[0,0.00521]), the scalar open-corridor resolvent comparison is an
> in-sample chain-forest identity, and no collapse was observed. The former
> “parameter-free fold diagnostic” is therefore a domain-coverage failure, not
> fold evidence. Milestone 4 adds explicit identity detection, held-out
> root-cohort validation, distributed burden coverage, and a machine-readable
> claim audit. See `MILESTONE_4_RESULTS.md`.

## What changed

Version 0.3.0 extends the fixed-population demonstration with an independent
open-boundary simulator containing
exogenous jobs, FIFO admission, endogenous work in process, repeated guarded
crossing requests, reusable finite fleets, per-job timing, direct causal event
lineage, and matched no-crossing counterfactuals.

The statistical layer now separates three questions that were previously easy
to conflate:

1. Is completion flow tracking arrival flow?
2. Is upstream backlog showing sustained growth over the observed horizon?
3. Does the queue regenerate by returning to zero?

Queue slope uses a Newey--West HAC standard error. Across-seed intervals
resample realizations rather than correlated time steps. Arrival-cohort latency
uses fixed follow-up so unfinished jobs remain in the estimand.

## Verification status

- 65 unit/integration tests pass.
- Zero robot collisions and zero occupied-crossing violations occur in every
  automated load/disturbance sweep and every generated experiment row.
- Every open run has exact job mass balance.
- Every finite-fleet run satisfies active + returning + available = \(N\) at
  every recorded step and at the final horizon.
- Open-event count, crossing activation time, severity, and paired latency show
  first-order convergence as \(\Delta t\) decreases from 0.10 to 0.025 s.
- The executable example and all experiment scripts complete from a clean
  `PYTHONPATH=src` invocation.

## Quick-profile observations

These values are software-level preliminary evidence, not paper estimates. The
quick profile uses three stochastic replications per load, 240 s per run, and
only two replications in the finite-fleet sweep.

### Capacity evidence

For both the crossing and no-crossing arms, the conservative finite-run bracket
was 0.65--0.80 jobs/s: 0.65 was the largest tested load with no-growth evidence
in at least two of three runs, while 0.80 was the first higher load with growth
evidence in at least two of three. The intervening 0.70 point was indeterminate,
so the code correctly refused to interpolate through it.

At the saturated 0.80 jobs/s point, mean completion rate was 0.644 jobs/s with
crossing requests and 0.675 jobs/s without them. The coarse bracket therefore
does not yet resolve the smaller capacity shift visible in service rate. The
30-seed, fine-grid paper profile is required for that inference.

### Attribution ablation

Direct one-generation estimates remained below one at every load. At 0.60
jobs/s the mean direct estimate was 0.909, while the 10 s correlational horizon
estimate was 1.727. At 0.80 jobs/s the corresponding values were 0.917 and
1.875. This reproduces the expected double-counting failure when total
downstream activity is treated as direct offspring.

The direct estimate at 0.70 jobs/s fell to 0.611 because the retained quick
window contained very few primary parents. That point is a sampling warning,
not evidence of a non-monotone physical law.

### Parameter-free fold diagnostic

The simulator does not use the fold curve internally. The generated diagnostic
plots measured direct offspring against measured primary burden and overlays

\[
R_{sn}(x)=\frac{\sqrt{1+x}}{\sqrt{1+x}+\sqrt{x}}.
\]

Several higher-load points lie near the curve, but the quick profile is too
sparse to claim collapse. Equality should be tested only at an independently
located capacity boundary, with confidence intervals propagated through both
axes. The current plot is a falsification target, not a successful validation
claim.

### Matched state-memory experiment

At every rate, the up segment, down segment, and empty-start control replay the
same regular task releases and the same crossing-request realization. In the
slow ramp at 0.65 jobs/s:

- empty-start mean queue: 3.038 jobs;
- upward branch: 4.353 jobs;
- downward branch: 20.353 jobs.

Thus the downward excess over its matched empty-start control was 17.316 jobs.
At 0.50 jobs/s the downward excess was 8.713 jobs. After draining to 0.35
jobs/s, both branches returned to the matched 0.022-job baseline.

This is strong evidence of finite-time state/backlog memory. It is **not**
evidence of two stationary equilibria or static hysteresis. Proving the latter
requires dwell-time scaling, quasi-stationary distributions, and metastable
transition-time measurements.

### Finite physical fleet

With an 8 s return delay and a saturated task source, completion rate initially
tracks the circulation ceiling \(N/(T_0+T_{return})\). It then saturates at the
corridor bottleneck:

| Fleet \(N\) | Crossing rate | No-crossing rate |
|---:|---:|---:|
| 2 | 0.0667 | 0.0667 |
| 8 | 0.2500 | 0.2667 |
| 20 | 0.5667 | 0.5833 |
| 30 | 0.6125 | 0.6667 |

The \(N=30\) crossing realization reduced saturated completions by about 8.1%
relative to its paired control. The utilization curve falls at large \(N\)
because additional robots wait outside the active bottleneck rather than
increasing service capacity. This is the finite-fleet truncation the unbounded
mean-field model cannot represent on its own.

## Generated artifacts

- `open_capacity_runs.csv`: per-seed operating-window evidence;
- `open_capacity_summary.csv`: replication-level bootstrap summaries;
- `open_attribution_diagnostics.csv`: direct and horizon estimators plus fold
  coordinates;
- `open_ramp_memory.csv`: matched up/down/empty-start segment results;
- `open_finite_fleet_runs.csv` and `open_finite_fleet_summary.csv`;
- `open_experiment_manifest.json`: exact profile, seed, configuration, bracket,
  and interpretation constraints;
- four publication-oriented diagnostic figures.

## What is now defensible

The code is a serious, invariant-checked experimental foundation for a paper.
It can support claims about causal attribution bias, finite-fleet truncation,
queue-growth evidence, paired disturbance cost, and dynamic state memory in a
single-file bottleneck.

It is not yet sufficient for the full world-stage non-normal multi-type claim.
Milestone 5 shows that spatial typing already produces a non-normal matrix in
the serial corridor, because one parent can affect successive followers over
time. Genuine fan-out is extremely rare in the corrected quick data, however,
so a merge/fan-out reservation network remains the most credible route to a
well-powered supercritical test.
