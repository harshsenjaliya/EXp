# Exposure-Aware Censored Branching Specification

## Purpose

Milestone 5 showed that selecting only completely resolved trees forces the
direct-offspring maximum-likelihood estimate to have spectral radius at or
below one. That estimator is valid for descriptive complete-tree summaries,
but it cannot identify a supercritical physical law because conditioning on
eventual extinction removes the surviving trees.

Milestone 6 adds a separate continuous-time likelihood that retains every
observed descendant of roots selected in a fixed start-time window. It is
designed for administrative right censoring at a simulation or logging
horizon. It does not alter the complete-tree estimator or retroactively turn
its in-sample identities into predictions.

## Stochastic model

A causal intervention of parent type a remains active for a random lifetime.
While active it generates direct type-b intervention starts as an independent
Poisson process with rate

\[
\beta_{ab}.
\]

It resolves with exponential hazard

\[
\delta_a.
\]

The next-generation mean matrix is

\[
B_{ab}=\frac{\beta_{ab}}{\delta_a}.
\]

Its spectral radius determines the subcritical or supercritical regime under
this fitted continuous-time branching abstraction.

For parent type a, define:

- \(E_a\): total observed active-event time, including partial time from
  events still active at the horizon;
- \(C_{ab}\): observed direct child starts of type b whose proximate parent
  has type a;
- \(D_a\): observed parent resolutions;
- \(Z_a\): parents active at the observation horizon.

Ignoring event-time constants, the log likelihood is

\[
\ell =
\sum_{a,b}\left(C_{ab}\log\beta_{ab}-\beta_{ab}E_a\right)
+
\sum_a\left(D_a\log\delta_a-\delta_aE_a\right).
\]

The maximum-likelihood estimates are

\[
\widehat\beta_{ab}=\frac{C_{ab}}{E_a},
\qquad
\widehat\delta_a=\frac{D_a}{E_a},
\qquad
\widehat B_{ab}=\frac{C_{ab}}{D_a}.
\]

Although exposure cancels algebraically in the final ratio, it does not cancel
from the joint likelihood or rate diagnostics. Most importantly, children
already observed from a horizon-censored parent remain in \(C\), while that
parent is not falsely added to \(D\).

## Cohort rule

Roots are selected only by primary start time in a fixed interval
[root_start, root_end). Every observed descendant of each selected root is
retained, even when that tree remains unresolved at the horizon. A non-root
event is accepted only if its proximate parent is present in the same
rollout-local event namespace.

Independent rollout or root forests are pooled through sufficient statistics.
Event identifiers are never concatenated into a global namespace.

## Fail-closed identifiability

An active parent type with positive exposure but zero observed resolutions has
an unidentified recovery rate and therefore an unidentified row of B. The API
returns NaN for that row, no spectral radius, and the status
unidentified_active_type_without_resolution. It never silently substitutes a
zero row.

Unobserved configured types do not block estimation. They remain zero rows and
must not be interpreted as empirically identified.

## Uncertainty

The bootstrap resamples whole independent forests rather than individual
events. This preserves parent-child dependence and the rollout-local identifier
contract. Resamples that fail to identify a parent type active in the point
estimate are rejected and counted; the returned valid-resample total makes
sparse-type instability visible.

The reported intervals are percentile bootstrap intervals. They quantify
finite-sample variation under the synthetic or simulator-generating process;
they do not correct model misspecification.

## Synthetic validation

The synthetic generator is the continuous-time analogue of a multitype
Galton-Watson process. It accepts a known next-generation matrix B and recovery
rates delta, then uses child-start rates beta = B times delta. Observation stops
at a fixed horizon, with active individuals and births before the horizon
retained.

The benchmark includes known scalar subcritical, near-critical, and
supercritical laws plus multitype non-normal and supercritical matrices. It
compares:

1. the exposure-aware likelihood using complete and censored trees;
2. the prior direct-offspring estimate after complete-tree selection;
3. the known generating matrix and spectral radius.

The decisive acceptance test is recovery of a known supercritical spectral
radius above one while the complete-tree estimate remains at or below one.

## Assumptions and claim boundary

The likelihood assumes type-specific constant child-start and recovery rates
over the observation window, independent administrative censoring, correct
proximate-parent attribution, and no missing events inside the retained
cohorts. Bootstrap validity also requires independent forests at the chosen
resampling level.

Passing the known-law benchmark validates estimator implementation and exposes
the complete-tree selection bias. It does not establish that the AMR corridor
itself is supercritical, identify an operational capacity fold, or provide
industrial safety evidence. Multi-layout, obstacle, visibility, footprint,
turn-rate, and acceleration studies must be performed separately after this
estimator gate.
