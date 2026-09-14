# Milestone 6: Censored Branching and Kinematic Map Generalization

## Status

Milestone 6 is the first version designed to test the two weakest links left by
the Milestone 5 audit:

1. complete-tree maximum likelihood cannot identify a supercritical process;
2. the original corridor treats commanded linear speed as nominal travel time
   and therefore cannot separate turning limits from shield-induced burden.

Version 0.6.0 adds a synthetic ground-truth validation layer for censored
multitype branching and a separate 2D route-level kinematic benchmark. The
implementation milestone is complete and has a registered quick-profile result.
Paper-facing numerical claims must still come from the preregistered paper
profile and its committed output manifest, not from CI smoke or quick data.

## 1. Exposure-aware branching likelihood

For a type-a parent, direct type-b children are modeled as an age-dependent
Poisson process

\[
\lambda_{ab}(u)
= B_{ab}\gamma_{ab}\exp(-\gamma_{ab}u),\qquad u\ge0.
\]

A parent observed for only \(C_i\) seconds contributes exposure

\[
E_{iab}
=\int_0^{C_i}\gamma_{ab}e^{-\gamma_{ab}u}\,du
=1-e^{-\gamma_{ab}C_i}.
\]

Let \(N_{ab}\) be the number of attributed direct \(a\to b\) children.
Ignoring terms independent of \(B\), the log likelihood is

\[
\ell(B)
=\sum_{a,b}\left[
N_{ab}\log B_{ab}
-B_{ab}\sum_{i:T_i=a}E_{iab}
\right].
\]

Its entrywise maximum is

\[
\boxed{
\widehat B_{ab}
=\frac{N_{ab}}
{\sum_{i:T_i=a}\left(1-e^{-\gamma_{ab}C_i}\right)}
}.
\]

This uses unresolved late parents instead of deleting their trees. The stopping
time may be the nominal horizon or a finite-population cap; exposure is measured
only up to the actual chronological stopping time.

### What this repairs

The former estimator divided edges by every observed parent. In a completely
observed single-parent forest, edge conservation forces its active-type
spectral radius to be no greater than one. It therefore cannot supply positive
evidence for supercriticality, regardless of how many simulations are run.

The exposure denominator is smaller than the complete-parent denominator when
follow-up is incomplete. It is not constrained by the forest identity and can
recover \(\rho(B)>1\) when the known data-generating process is
supercritical.

### Assumptions

The synthetic theorem-level validation makes the assumptions explicit:

- direct parentage and event type are observed correctly;
- independent rollout forests are the sampling clusters;
- recovery rates \(\gamma_{ab}\) are known for the first validation;
- conditional on parent type, direct offspring follow the exponential kernel;
- the population cap is an observed chronological stopping rule;
- roots are exogenous.

Recovery-rate misspecification is now an explicit paired sensitivity sweep on
nested realized forests. Uncertain parent attribution and interval censoring
remain separate ablations, not silently absorbed into this result.

### Uncertainty

Bootstrap samples resample whole rollouts with replacement. Events inside a
rollout are never treated as independent. Each bootstrap draw recomputes
\(\widehat B\), \(\rho(\widehat B)\), and the probability
\(\Pr[\rho(\widehat B)>1]\).

The validation sweep uses known matrices scaled to true spectral radii

\[
\rho(B)\in\{0.60,0.85,1.00,1.15,1.30\}.
\]

The old complete-tree estimator is retained as a named ablation.

## 2. Differential-drive kinematic formulation

Each route is a curvature-continuous natural cubic spline sampled in arc
length \(s\). Curvature and its spatial derivative are evaluated from
analytic spline derivatives. The
nominal unicycle kinematics are

\[
\dot x=v\cos\theta,\qquad
\dot y=v\sin\theta,\qquad
\dot\theta=\omega,\qquad
\omega=v\kappa(s).
\]

For wheel track \(L_w\), the differential-drive wheel speeds are

\[
v_L=v\left(1-\frac{L_w\kappa}{2}\right),\qquad
v_R=v\left(1+\frac{L_w\kappa}{2}\right).
\]

The pointwise curvature envelope is

\[
v_{\mathrm{env}}(s)=\min\left\{
v_{\max},
v_{\mathrm{map}},
\frac{\omega_{\max}}{|\kappa|},
\sqrt{\frac{a_{\mathrm{lat,max}}}{|\kappa|}},
\frac{v_{w,\max}}{1+L_w|\kappa|/2},
\sqrt{\frac{(1-q)\alpha_{\omega,\max}}{|\kappa'|}}
\right\}.
\]

The implementation uses \(q=1/2\). The remaining yaw-acceleration budget
limits longitudinal acceleration because

\[
\dot\omega=a\kappa+v^2\kappa'.
\]

For each arc segment,

\[
|a|\le
\min\left(
a_{\mathrm{long}},
\frac{q\alpha_{\omega,\max}}
{\max(|\kappa_i|,|\kappa_{i+1}|)}
\right).
\]

A forward/backward pass imposes acceleration, braking, start-speed, and
end-speed reachability. Nominal map time is then

\[
T_0(M,r)
\approx\sum_i\frac{2\Delta s_i}{v_i+v_{i+1}}.
\]

A same-length zero-curvature profile provides the straight reference. Their
difference is the turning penalty.

### Separation required by the theory

Turning, map speed limits, and wheel limits belong to \(T_0(M,r)\). A
protective slowdown or stop caused by a route obstruction contributes to the
severity-weighted intervention burden \(g\). Counting a planned turn as an
intervention would inflate \(g/T_0\) and invalidate the parameter-free fold
diagnostic.

## 3. Environments and obstacles

The deterministic catalogue contains:

| Map | Main mechanism | Routes | Conflict model |
| --- | --- | ---: | --- |
| Straight crossing | longitudinal obstruction | 1 | none |
| L-turn | curvature and yaw limits | 1 | none |
| S-curve | changing curvature | 1 | none |
| Two-to-one merge | converging flows | 2 | reservation zone |
| Four-way intersection | crossing flows | 2 | reservation zone |
| Warehouse grid | repeated aisle turns | 1 | static shelf field |

Every route is checked against the robot's circumscribed footprint and safety
margin for three registered platform classes:

| Platform | Role | Key effect |
| --- | --- | --- |
| Compact agile | small, high yaw/acceleration limits | geometry/map limit often binds |
| Standard | reference AMR | mixed map and kinematic limitation |
| Heavy payload | larger footprint, low lateral/yaw limits | turn envelope often binds |

Each of the three robot classes runs five independent obstruction classes at
light, medium, and heavy duration levels:

- unexpected stationary pallet;
- pedestrian crossing;
- moving forklift crossing;
- temporary aisle closure;
- occlusion/delayed detection through a larger reaction-time contract.

The obstruction layer uses a two-phase activation rule. Guarded requests become
active only when the robot can stop before the occupied interval or has already
cleared it. Unguarded unexpected obstacles fail closed if their requested
activation violates the same contract. Every rollout records collision count,
occupied-zone violation count, stopping margin, wheel speed, yaw rate, yaw
acceleration, lateral acceleration, traversal delay, and severity-weighted
loss. Every one of the 360 platform-route-obstacle-level cases is run on three nested
time grids, \(\Delta t\), \(\Delta t/2\), and \(\Delta t/4\). Guarded activation
makes individual hybrid trajectories nonsmooth at switching surfaces, so
monotone Richardson convergence is not assumed. Grid independence is instead
registered on the two finest grids for the observables used by the paper:
relative traversal-time error at most 3% and absolute burden error
\(|\Delta x|\le0.05\) across every row. The induced fold-curve tolerance
\(|\Delta R_{\rm sn}|\le0.02\) is enforced only on the informative domain
\(x\ge0.10\), because \(dR_{\rm sn}/dx\) is singular at zero. Low-burden fold
errors, three-grid means, 95th percentiles, maxima, and worst-case identifiers
remain reported rather than hidden. Traversal-time and burden tolerances are
hard executable invariants. The transformed fold tolerance is a separate
evidence check: smoke-data failure does not masquerade as a software failure,
but the paper profile must pass it before a fold-resolution claim. Severity
loss itself is integrated with the trapezoidal rule over the fixed-step
trajectory.


## 4. Verified quick-profile result

The registered quick profile completed on 2026-09-14 at commit
b45014c1c8599bfdf79cac6402c577c6a634726b. The immutable workflow record is
[Actions run 34879911615](https://github.com/harshsenjaliya/EXp/actions/runs/34879911615);
its 12-file artifact has ID 10362252881 and SHA-256
3f964dfe603e48f5b2596ac33310aa996c3643565ad55677d6db4142cb289200.

| True rho | Exposure-aware estimate (95% cluster bootstrap) | Complete-tree ablation |
| ---: | ---: | ---: |
| 0.60 | 0.5916 [0.5457, 0.6310] | 0.4283 |
| 0.85 | 0.9051 [0.8484, 0.9568] | 0.5957 |
| 1.00 | 1.0307 [0.9851, 1.0827] | 0.6593 |
| 1.15 | 1.1655 [1.1210, 1.2069] | 0.7068 |
| 1.30 | 1.2960 [1.2462, 1.3390] | 0.7535 |

The exposure-aware mean absolute error is 0.0227; the complete-tree ablation's
is 0.3513. All five tested truths lie inside their intervals. This is five-point
containment, not an interval-coverage-frequency estimate. The two subcritical
cases have upper endpoints below one, the critical case is correctly
indeterminate, and the two supercritical cases have lower endpoints above one.

On paired nested forests, the maximum correctly specified recovery-kernel error
is 0.0211 and the largest estimate range over 0.5--4.0 s follow-up horizons is
0.0603. In contrast, recovery-rate misspecification over the registered
0.5--1.5 multiplier sweep produces absolute error as large as 0.6002. Known or
independently estimated recovery rates are therefore a substantive assumption,
not a harmless implementation detail.

The route benchmark contains 360 cases: three robot classes, eight routes, five
obstacle mechanisms, and three levels. It spans
\(x\in[0.00293,1.41624]\). All cases have zero static collisions, zero occupied
route-zone violations, nonnegative stopping margin, and bounded kinematics.
Maximum utilization is 0.6342 for yaw rate, 0.2904 for yaw acceleration, 1.0000
for lateral acceleration, and 1.0000 for wheel speed.

The two finest quick grids pass the registered direct-observable gates: maximum
relative traversal-time change is 0.02285 and maximum absolute burden change is
0.03743. They do not yet pass the separate fold-resolution check. Across the
165 rows with \(x\ge0.1\), the maximum induced change in \(R_{\rm sn}\) is
0.02353 against the registered 0.02 target. The worst case is the heavy-payload
straight route with a medium unexpected stationary obstacle
(\(x:0.13495\to0.10151\)). Three-grid aggregate traversal errors are also
non-monotone. These facts are retained as limitations and must be resolved by
the paper profile or by a better hybrid event integrator before a numerical
fold-resolution claim.

These quick results validate the software layers and the censoring repair. They
remain synthetic branching evidence and single-robot route evidence; they are
not an observed multi-robot capacity collapse.

## 5. Scope boundary

The kinematic benchmark follows one robot along one route at a time. Merge and
intersection routes and their shared zones define geometry for the next
multi-robot scheduler, but this version does not claim that independent
single-route traversals validate merge throughput, non-normal cascade
propagation, or an open-system fold.

Likewise, synthetic branching recovery validates the estimator against a known
data-generating law. It does not prove that the physical AMR simulator or an
industrial fleet follows that law.

The paper claim becomes stronger only when the layers are connected:

1. log causally attributed, right-censored parent exposure from a multi-robot
   map;
2. estimate \(B\), uncertainty, and \(G^\star\) out of sample;
3. locate capacity from queue drift independently of the branching fit;
4. test whether the observed boundary obeys the fold curve using kinematic
   \(T_0(M,r)\) and severity-weighted \(g\).

## 6. Reproducible execution

From the project directory:

    python -m unittest discover -s tests -v
    python scripts/run_milestone6_experiment.py --profile smoke
    python scripts/run_milestone6_experiment.py --profile quick
    python scripts/run_milestone6_experiment.py --profile paper

The profiles differ only in replication count, bootstrap count, and integration
step. Seeds, matrices, map geometry, obstacle schedules, equations, and claim
boundaries are written to the manifest.

Generated files are:

- milestone6_branching_validation.csv;
- milestone6_branching_matrix_entries.csv;
- milestone6_branching_sensitivity.csv;
- milestone6_kinematic_profiles.csv;
- milestone6_obstacle_benchmark.csv;
- milestone6_step_convergence.csv;
- milestone6_branching_recovery.png;
- milestone6_branching_sensitivity.png;
- milestone6_map_catalogue.png;
- milestone6_obstacle_delay.png;
- milestone6_step_convergence.png;
- milestone6_manifest.json.

## 7. Acceptance criteria before a paper claim

The paper-profile run—not the lower-resolution CI smoke run—must satisfy all
of the following:

- exposure-aware bias and interval coverage are reported at every true radius;
- supercritical classification is not inferred from the naive estimator;
- results include recovery-rate and nested-horizon sensitivity;
- every kinematic rollout has zero collision and occupied-zone violations;
- every route case passes the registered finest-grid time/burden tolerances;
- every informative-burden row passes the fold-diagnostic tolerance in the
  paper profile;
- every limit is reported in physical units;
- turning time is included in \(T_0\), never in \(g\);
- the independent multi-robot capacity-boundary experiment remains a separate
  requirement.

Milestone 6 therefore removes two known structural blockers and supplies the
cross-map experiment scaffold. It does not relabel scaffold validation as
capacity-collapse evidence.
