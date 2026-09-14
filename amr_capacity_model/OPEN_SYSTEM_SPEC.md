# Open-System Simulator Specification

## Scientific role

`amr_capacity.open_system` is a microscopic falsification environment for the
endogenous-density theory. Jobs arrive from an exogenous point process, wait in
an upstream FIFO queue, enter only when the local safety contract permits, and
leave after one corridor traversal. Consequently, queue length and work in
process are outcomes rather than inputs.

The simulator does **not** call the saddle-node formula when advancing robots.
Analytical predictions can therefore fail against it. That separation is
intentional and required for a meaningful validation experiment.

## State and conservation laws

Active robot arrays are ordered front to rear. At step (k), the microscopic
state contains physical robot ID, job ID, longitudinal position, speed,
arrival/admission time, accumulated severity loss, and active intervention ID.

Every step asserts job conservation:

\[
A(t)=Q(t)+W(t)+D(t),
\]

where (A) is cumulative arrivals, (Q) is the upstream queue, (W) is work
in process, and (D) is completed traversals.

Two fleet modes are available:

- `fleet_size=None`: an unlimited traversal-token population for the open
  mean-field experiment;
- finite `fleet_size=N`: physical robot IDs are reused after an
  `empty_return_time`, with the additional invariant

\[
N=W(t)+R(t)+I(t),
\]

where (R) is returning robots and (I) is idle/available robots. The return
leg is an abstract delay outside the modeled bottleneck; it has no collision
geometry.

## Hybrid time model

Robot sensing and motion use fixed-step integration. Job arrivals and crossing
requests are event-time inputs that join their queues at the next integration
boundary. This deliberately avoids claiming a fully event-driven dynamics
engine.

For follower (i), the immediate leader is obtained in (O(N)) from the
front-to-rear order. The shield calls the canonical `v_safe_next_step` function
and enforces

\[
\frac{v_{i,k}+v_{i,k+1}}{2}\Delta t
+d^{\mathrm{stop}}(v_{i,k+1})\le d_{i,k}^{\mathrm{free}}.
\]

Acceleration is clipped to ([-b,a]). The code then independently checks the
next-step stopping invariant, physical non-overlap, and occupied-crossing
exclusion. A dynamically unreachable shield command is an assertion failure,
not an emergency repair.

## Admission semantics

Jobs are admitted FIFO. A finite fleet additionally requires an available
physical robot. The new robot is placed at the entry with `admission_speed`
only when the rear clearance satisfies the stopping-distance contract. Normal
launch acceleration is not labeled as a shield intervention. A shield event
starts only if an obstacle cap is below the unconstrained next speed.

## Independently guarded crossings

A configuration contains one or more ordered, non-overlapping crossing
positions. Each position owns an independent FIFO request queue, active service
state, and gate robot. A pedestrian request and physical path occupancy are
distinct:

1. If the crossing is already safely occupiable, the request activates.
2. Otherwise, closer robots that cannot stop are allowed to clear.
3. The closest farther robot that can stop safely becomes the gate robot and
   receives a reservation constraint.
4. Followers react only through their immediate leaders.
5. When the physical zone is clear and the nearest approaching robot satisfies
   the stopping contract, occupancy begins for the requested duration.

This protocol prevents both unsafe obstacle insertion and starvation under a
dense platoon. Request time, crossing ID, activation time, release time,
waiting time, and horizon censoring are all logged. Reservation and physical
occupancy are separate trajectory signals. Aggregate logs retain the old
`crossing_active` Boolean and additionally record active/reserved crossing
counts.

## Direct causal lineage

An intervention episode is keyed by its proximate limiting constraint. A cause
or parent transition closes the old episode and begins a new one. Processing
front to rear means a follower can reference a leader event created in the same
step without a causal cycle.

- a crossing reservation/occupancy constraint creates a primary event;
- a constrained follower inherits the active event of its immediate blocking
  leader as its one-generation parent;
- a constrained follower whose leader has no active event is background;
- recovery remains in the current episode until desired speed is restored.

This event table is compatible with `estimate_direct_branching`. Time-window
counting remains an explicitly correlational ablation and is not substituted
into the branching resolvent.

Events also store their longitudinal start position. Primary events store the
crossing index directly, avoiding a seed-dependent external
`source_id`-to-gate map.

In a complete scalar single-parent forest, the in-sample equality

\[
(1-\widehat B)^{-1}=E/P
\]

is algebraic. `ChainForestDiagnostic` detects that structure. The same
conservation law applies to a training primary-mix resolvent under any event
typing; `MultitypeForestIdentityDiagnostic` detects it. Paper-facing rows must
keep rollout-local ID namespaces separate, fit on fixed primary-root cohorts,
and score different replications.

The serial multi-crossing layout produces a non-normal spatial matrix and rare
temporal fan-out, but the corrected quick sweep remains subcritical. It is
therefore evidence for a topology penalty in \(G^\star\), not evidence for
supercritical branching.

## Counterfactual design

`run_open_paired_rollout` replays exactly the same job arrival times with and
without crossing requests. Job IDs make completion and sojourn comparisons
paired even when the two systems admit or complete different numbers of jobs.
Fleet-level effects include completion loss, terminal backlog increase,
severity increase, integrated population-time increase, and mean paired
sojourn increase among jobs completed in both arms.

## Finite-run statistical evidence

Finite simulation cannot prove positive recurrence. The analysis therefore
uses the labels `no_growth_detected`, `growth_detected`, and `indeterminate`
rather than declaring a run mathematically stable or unstable.

Queue trend is estimated by OLS with a Bartlett-kernel Newey--West covariance,
because adjacent backlog samples are strongly autocorrelated. Regeneration
diagnostics separately report empty-time fraction, number of returns to zero,
last empty time, terminal busy-period duration, and longest observed busy
period. Growth requires a positive lower confidence bound together with a long
terminal busy period and no empty observation. No-growth evidence requires
observed regeneration and approximate flow balance. Ambiguous cases remain
indeterminate.

Across independent seeds, bootstrap intervals resample realizations—not time
steps. The empirical capacity output is a bracket between the largest tested
load with repeated no-growth evidence and the smallest higher load with
repeated growth evidence. It does not interpolate across indeterminate loads.

## Censoring-safe latency

A completed-job mean is optimistically biased near overload because slow jobs
remain unfinished. `analyze_arrival_cohort` instead selects jobs with identical
follow-up and reports

\[
\frac{1}{m}\sum_{j=1}^m \min(T_j,\tau),
\]

the restricted mean sojourn time at horizon \(\tau\), plus the probability of
completion within \(\tau\).

## Ramp interpretation

The experiment runner includes fast and slow up/down ramps. At each load, the
up and down branches reuse the same task-release pattern and crossing-request
realization. A third run replays that exact segment from an empty system.

An up/down gap is reported as **dynamic backlog memory**. It is not called
static hysteresis: finite dwell time, queued work, returning robots, or an
unfinished crossing service can produce the loop without multiple stationary
distributions.

## Computational structure

Robot operations are vectorized NumPy calculations. Immediate-leader lookup is
\(O(N)\) per time step; there is no pairwise \(O(N^2)\) distance matrix. Job and
per-crossing queues are currently Python FIFO containers because their operations
are small relative to integration. Trajectory storage is aggregate and
strided, so dynamic robot populations do not require padded position tensors.

## Deliberate limitations

- One lane with one or more serial crossings; no merge, fan-out, grid routing,
  deadlock, or reservation scheduler is implemented yet.
- Collision geometry is longitudinal centerline separation, not swept 2D
  footprint geometry.
- The external empty-return leg is a delay, not a simulated route.
- The quick experiment profile is a software/estimand check with three seeds;
  it is not publication-grade statistical evidence.
- Queue evidence is finite-horizon. A proof of the microscopic process's
  recurrence properties is outside this implementation.
- The universal fold curve is not yet tested. The burden sweep is a domain
  coverage and stress audit. Fold validation requires an independently located
  loss of a stable branch before comparing measured \(R\) with \(g/T_0\).
