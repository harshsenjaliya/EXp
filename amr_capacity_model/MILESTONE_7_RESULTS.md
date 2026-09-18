# Milestone 7: Grouped Cascade Inference and Loaded Route Networks

## Status

Version 0.7.0 closes the implementation gap between the censored branching
estimator and fixed-step robot logs, and it moves the merge/intersection maps
from static geometry into an interacting finite-fleet simulator. All software
and safety gates pass in the registered smoke and quick profiles. The quick
profile is diagnostic evidence only: it has 12 synthetic outer datasets per
truth and two independent network rollouts per condition. Paper-facing
confidence claims remain preregistered to the `paper` profile.

The current evidence supports four claims:

1. offspring means and exponential recovery rates can be estimated jointly
   across known subcritical and supercritical synthetic truths;
2. fixed-step simulator logs require a grouped-time observation model because
   many direct parent/child events share a timestamp;
3. the estimator now runs directly on independently generated simulator logs
   and predicts held-out cascade size without the chain-forest identity;
4. multi-robot merge and intersection experiments run at N=200 with exact job
   and fleet conservation, zero collisions, zero protected-zone violations,
   and differential-drive limits enforced.

The quick data do **not** yet support a confidence-certified physical phase
transition, a calibrated scheduling trigger at rho=1, or a parameter-free fold
validation. Those are explicit paper-profile gates.

## 1. Joint offspring and recovery likelihood

For parent type \(a\), child type \(b\), parent follow-up \(C_i\), and child
age \(u_j\), the continuous-time log likelihood is

\[
\ell_{ab}(B,\gamma)
=N_{ab}\log B_{ab}+N_{ab}\log\gamma_{ab}
-\gamma_{ab}\sum_j u_j
-B_{ab}\sum_i\left(1-e^{-\gamma_{ab}C_i}\right).
\]

For fixed \(\gamma_{ab}\),

\[
\widehat B_{ab}(\gamma)
=\frac{N_{ab}}
{\sum_i\left(1-e^{-\gamma_{ab}C_i}\right)},
\]

so each \(\gamma_{ab}\) is fitted by a bounded one-dimensional profile
likelihood rather than a coupled matrix optimisation.

### Fixed-step grouped likelihood

The simulator records event starts on an integration grid \(\Delta\). For lag
bin \(k\), the integrated exponential mass is

\[
q_{ab}(k)
=e^{-\gamma_{ab}k\Delta}
\left(1-e^{-\gamma_{ab}\Delta}\right),
\qquad k=0,1,\ldots
\]

and a parent observed for \(K_i\) right-inclusive bins contributes

\[
H_{iab}=1-e^{-\gamma_{ab}K_i\Delta}.
\]

The grouped log likelihood is

\[
\ell^{\Delta}_{ab}(B,\gamma)
=N_{ab}\log B_{ab}
+\sum_j\log q_{ab}(k_j)
-B_{ab}\sum_i H_{iab}.
\]

This assigns finite probability to \(k=0\). It therefore handles same-step
offspring without adding arbitrary jitter or pretending timestamps have more
resolution than the integrator.

### Exposure-policy registration

The primary model retains the Milestone 6 Hawkes convention: an event is a
point whose kernel is observed until the rollout boundary. A competing
`active_interval` model ends exposure when the robot's logged slowdown ends.
Both are reported. The active-interval result cannot silently replace the
registered model because the two quantities have different interpretations.

## 2. Repeated-dataset calibration

A bootstrap distribution conditional on one dataset is not an empirical
false-positive rate. Milestone 7 generates independent outer datasets and
bootstraps each dataset as a rollout cluster.

Quick-profile results:

| True rho | Mean joint estimate | Joint MAE | 95% interval coverage | Wrong-regime frequency |
| ---: | ---: | ---: | ---: | ---: |
| 0.85 | 0.8546 | 0.0312 | 12/12 | 0/12 |
| 1.00 | 0.9919 | 0.0375 | 10/12 | 2/12 |
| 1.15 | 1.1862 | 0.0486 | 11/12 | 0/12 |

The recovery-rate misspecification failure from Milestone 6 is removed: gamma
is no longer assumed known. However, the 2/12 one-sided decisions at exactly
rho=1 fail the registered 5% boundary-error gate. More outer datasets and a
calibrated decision margin are required before the interval can drive a
scheduler.

## 3. Estimator-to-simulator bridge

Six independent multi-crossing corridor runs were generated at each offered
load. Events were typed as full-stop versus reduced-speed interventions, kept
in rollout-local namespaces, and adapted without selecting only extinct trees.

| Arrival rate | Zero-lag edge fraction | Grouped rho | Active-interval ablation | \(G^\star\) | Held-out progeny error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.20 | 0.237 | 0.111 | 0.247 | 1.43 | 0.8% |
| 0.35 | 0.520 | 0.527 | 0.684 | 3.31 | 4.1% |
| 0.50 | 0.766 | 0.800 | 0.862 | 5.96 | 12.8% |
| 0.70 | 0.780 | 0.809 | 0.856 | 6.21 | 6.2% |

The held-out comparison uses the first half of the rollout namespaces for
training and the second half for measurement. It is not the algebraic
\(E/P=1/(1-\widehat B)\) identity. The full resolvent is evaluated only when
the training matrix is subcritical.

The table also demonstrates why \(G^\star\), rather than rho alone, remains in
the paper: rho changes little between the two highest loads while cumulative
amplification continues to grow.

## 4. Finite-population and extinction controls

The new depletion simulator assigns a finite susceptible population by type.
Every accepted intervention consumes one previously unaffected individual;
candidate births compete chronologically for those remaining individuals.
This is a physical depletion mechanism, not an arbitrary stop after N log
records.

At uncapped truth rho=1.30, the quick profile observed population-boundary
fractions 0.302, 0.108, 0.016, and 0.000 for total populations 20, 50, 100, and
200. The corresponding estimates were 1.012, 1.034, 1.077, and 1.163. The
finite population therefore produces a measurable downward bias that shrinks
with population size but is not negligible at N=200 over the registered
horizon.

For a Poisson Galton--Watson process with \(m=1.3\), the extinction probability
is \(q=0.577030\), so the extinction-conditioned dual mean is
\(mq=0.750139\). From 20,000 uncensored trees, the empirical extinct-tree mean
was 0.750178 (absolute error \(3.9\times10^{-5}\)). This isolates the duality
mechanism from temporal censoring.

## 5. Loaded merge and intersection simulator

The route-network simulator adds:

- finite physical fleets and per-route job queues;
- curvature-aware nominal speed profiles and differential-drive limits;
- shared conflict-zone ownership;
- aligned cross-route following on the common merge segment;
- guarded route disturbances with paired no-disturbance controls;
- contact-level causal attribution and severity/spatial event typing;
- exact job mass balance and physical-fleet conservation;
- global collision, reservation, obstruction, braking, yaw-rate, lateral
  acceleration, and wheel-speed assertions.

The quick grid contains both maps, fleet caps N in {50, 100, 200}, three loads,
and two independent disturbance realizations. Route length is increased with N
by adding straight feeder storage while preserving the central turn/conflict
geometry. Actual mean WIP, lane density, and headway are logged because a fleet
cap is not a density measurement.

The primary point estimates contain a narrow crossing: the N=200 intersection
moves from rho about 0.973 at load 0.40 to 1.008 at load 0.70. The N=100 merge
also straddles one (approximately 1.026, 0.978, and 0.996 across its load
sweep). These are **point-estimate observations only**. Two rollout clusters
are insufficient for a cluster-bootstrap regime certificate, so every quick
network verdict is `insufficient_clusters`.

All generated network runs have:

- zero robot collisions;
- zero conflict-zone ownership violations;
- zero active-disturbance intrusions;
- zero job or fleet conservation residual;
- yaw rate, lateral acceleration, and wheel speeds within registered limits.

Treated throughput turns downward in at least one fixed map/fleet sweep, while
the matched control remains on its geometric reservation ceiling. This is a
candidate faster-is-slower/cascade-capacity result, not yet a paper claim.

## 6. Registered paper-profile gates

The paper profile must satisfy all of the following before a manuscript may
state a physical critical transition or calibrated capacity certificate:

1. at least 200 independent outer datasets per synthetic truth;
2. empirical 95% interval coverage of at least 90% at every registered truth;
3. wrong one-sided regime decisions at true rho=1 no greater than 5%;
4. at least 20 independent network rollout clusters per condition;
5. N=100, 200, and 300 on both merge and intersection maps;
6. at least one fixed map/fleet sweep with confidence-certified subcritical and
   supercritical conditions;
7. held-out progeny error reported at every subcritical condition;
8. burden reported separately as `x_route`, `x_primary_fleet`, and
   `x_total_fleet`—never pooled;
9. zero collisions and zero protected-zone violations in every retained run;
10. an independent established robotics simulator before the final
    ICRA/IROS/RA-L submission.

## 7. Current claim boundary

Milestone 7 is a complete software milestone and a strong quick-profile
research probe. It makes the project substantially closer to a publishable
robotics paper because the estimator, event logs, kinematics, and loaded maps
now execute in one reproducible path. It is not yet the final paper evidence.
The remaining work is numerical replication/calibration, attribution ablation
on the loaded network, and external-simulator validation—not another rewrite
of the central mathematics.
