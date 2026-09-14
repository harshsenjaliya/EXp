# Shield-Coupled AMR Capacity Model

This project is the tested mathematical and simulation foundation for a
shield-coupled AMR fleet study. It implements:

- conservative stopping distance and its safe-speed inverse;
- homogeneous closed-system intervention critical speed;
- the open-system saddle-node capacity boundary;
- the stable/unstable quasi-stationary equilibrium branches around that fold;
- the speed-dependent fold-capacity wrapper used for controller sweeps;
- a finite fleet/space correction that separates fold-limited from
  occupancy-limited throughput;
- the parameter-free scalar fold curve;
- exact cumulative-cascade budgets for non-normal branching matrices;
- severity-weighted mean cascade loss and its density derivative;
- scalar and topology-aware fold margins;
- metrics adapters that import the reference equations rather than copying them;
- vectorized fixed-step dynamics on a rectangular 2D route;
- a guarded pedestrian-crossing perturbation;
- causal intervention lineages based on the proximate blocking robot;
- treated/no-crossing paired counterfactual rollouts;
- direct-offspring, time-window-ablation, and held-out cascade estimators;
- automatic detection of the scalar single-parent chain identity;
- automatic detection of the corresponding multi-type primary-mix identity;
- strict rejection of duplicate rollout-local event IDs;
- boundary-censored finite-chain estimation using unique robots per primary;
- an open-boundary simulator with stochastic job arrivals and endogenous WIP;
- optional reusable finite fleets with explicit return delay and conservation;
- a starvation-free two-phase guarded crossing protocol;
- several independently queued and guarded crossing locations;
- per-job queue, travel, sojourn, and severity records;
- HAC queue-drift inference, regeneration diagnostics, and conservative
  finite-run evidence labels;
- fixed-follow-up cohort latency that includes unfinished jobs;
- replicated capacity bracketing and matched fast/slow ramp controls;
- fail-closed claim auditing that separates coverage, prediction, and fold
  identification;
- spatially typed multi-type estimation with exact and LP \(G^\star\)
  cross-checks and an inherited-root-gate negative control;
- exposure-aware censored multitype branching inference with rollout-cluster
  bootstrap and known sub/supercritical synthetic validation;
- differential-drive kinematic speed envelopes, 2D Bezier routes, static
  footprint clearance, and guarded route-obstacle benchmarks across six maps.

The equations describe a research model.  They do not constitute an industrial
safety certification or a protective-field design method.

See [`MODEL_SPEC.md`](MODEL_SPEC.md) for the equation-to-function mapping,
assumptions, and the finite-density correction discovered during implementation.
See [`SIMULATOR_SPEC.md`](SIMULATOR_SPEC.md) for state, event, attribution, and
safety conventions.
See [`MILESTONE_2_RESULTS.md`](MILESTONE_2_RESULTS.md) for the preliminary
deterministic experiment findings and their limitations.
See [`OPEN_SYSTEM_SPEC.md`](OPEN_SYSTEM_SPEC.md) for the open-system state,
finite-fleet semantics, guarded crossing automaton, and statistical estimands.
See [`MILESTONE_3_RESULTS.md`](MILESTONE_3_RESULTS.md) for the generated
open-system evidence and the claims it does and does not support.
See [`MILESTONE_4_RESULTS.md`](MILESTONE_4_RESULTS.md) for the structural audit,
corrected finite-boundary estimator, distributed burden sweep, and current
publication claim boundary.
See [`MILESTONE_5_RESULTS.md`](MILESTONE_5_RESULTS.md) for the finite-estimator
benchmark, duplicate-ID forensic audit, generalized multi-type identity, and
corrected non-normality experiment.
See [`MILESTONE_6_RESULTS.md`](MILESTONE_6_RESULTS.md) for the censored
likelihood, differential-drive formulation, map/obstacle catalogue, registered
experiments, and explicit scope boundary.

## Run the validation suite

From this directory:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Generate reference outputs

```bash
PYTHONPATH=src python scripts/generate_reference_outputs.py
```

This writes a parameter-free fold curve, non-normal topology table, and a PNG
reference figure to `outputs/`. It also writes the normalized stable and
unstable equilibrium branches as both CSV and PNG.

## Run the paired corridor experiment

```bash
PYTHONPATH=src python scripts/run_corridor_experiment.py
```

This performs separate training and held-out rollouts across speed and density,
writes CSV results, compares direct attribution against several time horizons,
records treated/control circulation rates, follows every cascade to resolution
or the finite fleet boundary, and generates the corridor validation figures.
The script uses a fixed seed.

## Run the open-system experiment

```bash
PYTHONPATH=src python scripts/run_open_system_experiment.py --profile quick
```

The quick profile runs replicated stochastic load points, a matched
treated/control capacity sweep, fast/slow ramps with exact empty-start replays,
a finite physical-fleet sweep, attribution ablations, a burden-domain audit,
and held-out branching rows. It writes CSV, JSON, and PNG artifacts to
`outputs/`. A single-lane in-sample resolvent is explicitly not treated as
validation.

The `paper` profile expands to 30 seeds, a 1,800 s horizon, finer load spacing,
and 10,000 bootstrap resamples. It is deliberately expensive:

```bash
PYTHONPATH=src python scripts/run_open_system_experiment.py --profile paper
```

## Run the distributed-burden stress experiment

```bash
PYTHONPATH=src python scripts/run_burden_stress_experiment.py --profile quick
```

This sweep uses 1--16 independent guarded crossings and separately chosen
coverage regimes to span approximately \(x=0.005\) to \(0.91\) without making
any individual crossing queue unstable. It is a domain-coverage and stress
test, not a fold test. Use `--profile paper` for the 30-replication version.

## Run the corrected multi-type experiment

```bash
PYTHONPATH=src python scripts/run_multitype_experiment.py --profile quick
```

This fits scalar, severity, and spatial direct-offspring matrices using
separate event-ID namespaces and complete primary-root cohorts. It scores on
independent rollouts, verifies the direct and LP forms of \(G^\star\), and keeps
root-gate inheritance only as a diagonal negative control. The generated quick
profile finds subcritical but non-normal amplification; it does not establish a
supercritical transition. Use `--profile paper` for the longer replicated
experiment.

## Run Milestone 6

```bash
PYTHONPATH=src python scripts/run_milestone6_experiment.py --profile smoke
PYTHONPATH=src python scripts/run_milestone6_experiment.py --profile quick
```

The first experiment recovers known subcritical and supercritical multitype
matrices under temporal/population censoring and compares against the
complete-tree ablation. The second computes curvature-aware nominal time and
runs stationary obstacle, pedestrian, forklift, closure, and occlusion
disturbances on straight, L-turn, S-curve, merge, intersection, and warehouse
routes. Use `--profile paper` only for paper-facing estimates.

After generating the experiments, run the fail-closed claim ledger:

```bash
PYTHONPATH=src python scripts/audit_research_claims.py
```

## Run the executable example

```bash
PYTHONPATH=src python examples/quick_start.py
```

The example performs a speed sweep under a finite coupled-density limit,
identifies the usable capacity-maximizing speed for the supplied burden model,
reports which boundary is active, and prints the two equilibrium branches at
95% of that capacity.

For the open system:

```bash
PYTHONPATH=src python examples/open_system_quick_start.py
```

## Simulator integration rule

The `simulation`, `open_system`, and metrics modules must import functions from
`amr_capacity.capacity_theory`.  Closed-form equations must never be copied into
robot or experiment classes.  This module remains the single source of truth.

Simulation and metrics code call the canonical analytical functions. New
layouts must continue to import these functions rather than duplicating the
quadratic, resolvent, or saddle-node formulas.
