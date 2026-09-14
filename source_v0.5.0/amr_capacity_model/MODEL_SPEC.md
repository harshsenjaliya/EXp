# Mathematical Model and Software Contract

## Scope

This milestone implements and tests the analytical reference layer. It is not
yet the 2D fleet simulator. Geometry, route planning, task arrivals, paired
counterfactual rollouts, and direct-offspring attribution will be added on top
of this package and must call these functions rather than copy their equations.

## Local shield

The conservative stopping-distance abstraction is

\[
d^{\mathrm{stop}}(v)=a_0+v\tau^r+\frac{v^2}{2b}.
\]

`d_stop` evaluates this expression and `v_safe` implements its nonnegative
inverse. The simulator must use the same braking magnitude in its integrator
and call `assert_zero_collisions` as a hard invariant.

With spare headway \(A=h-a_0\),

\[
\tau^{\mathrm{buf}}(v)=
\left[\frac{A}{v}-\tau^r-\frac{v}{2b}\right]_+,
\qquad
\phi(v)=\chi e^{-\gamma\tau^{\mathrm{buf}}(v)}.
\]

These are implemented by `tau_buffer` and `offspring_probability`.

## Scalar open-system fold

Let \(T_0=\ell/v\), \(g=p\bar D\), and let \(n\) denote coupled
work-in-process. Under the quasi-stationary mean-field closure,

\[
n=\eta\lambda\left(T_0+\frac{g}{1-n\phi}\right).
\]

Equivalently, the sustainable arrival rate at a chosen density is

\[
\mu(n)=\frac{n(1-n\phi)}
{\eta\left[T_0(1-n\phi)+g\right]}.
\]

For \(g>0\), the two fixed points coalesce at

\[
\lambda_{\mathrm{sn}}=
\frac{1}{\eta\phi\left(\sqrt{T_0+g}+\sqrt g\right)^2},
\]

\[
n_{\mathrm{sn}}=
\frac{\sqrt{T_0+g}}
{\phi\left(\sqrt{T_0+g}+\sqrt g\right)},
\qquad
\mathcal R_{\mathrm{sn}}=n_{\mathrm{sn}}\phi.
\]

Writing \(x=g/T_0\) gives the parameter-free prediction

\[
\mathcal R_{\mathrm{sn}}(x)=
\frac{\sqrt{1+x}}{\sqrt{1+x}+\sqrt{x}},
\qquad \frac12<\mathcal R_{\mathrm{sn}}<1
\quad (x>0).
\]

The functions `lambda_saddle_node`, `n_saddle_node`,
`r_saddle_node_from_burden`, and `scalar_equilibria` are the canonical
implementations.

The identification \(\mathcal R=n\phi\) is a homogeneous mean-field closure.
Here \(n\) must measure the population of locally coupled opportunities, not be
silently substituted by arbitrary global fleet size. In a single-file route,
every event has at most one proximate child, so \(\mathcal R\le1\); density can
still increase the probability that this child is triggered. The included
density sweep tests whether an approximately linear \(B(N,v)\approx N\phi(v)\)
regime exists before contact saturation. General layouts must use measured
\(B(n,v)\), not assume this identification.

## Finite-density correction

The fold boundary alone is not the usable capacity of a finite facility. If
\(\phi(v)\) becomes tiny at low speed, the unbounded mean-field model can make
\(\lambda_{\mathrm{sn}}\) very large by allowing arbitrarily many robots to
accumulate. A facility or finite fleet imposes \(n\le n_{\max}\). Therefore

\[
\lambda_{\mathrm{cap}}(v)=
\mu\!\left(\min\{n_{\max},n_{\mathrm{sn}}(v)\}\right).
\]

`capacity_with_density_limit` implements this correction and reports whether
the active boundary is the cascade fold or finite occupancy. Any claim about
an optimal speed must use this corrected capacity unless an unbounded-density
assumption is explicitly intended.

## Multi-type non-normal cascades

The package uses the row-parent convention: \(B_{ab}\) is the expected number
of direct type-\(b\) children caused by one type-\(a\) parent. With severity
vector \(d\) and primary-type distribution \(a\),

\[
H=a^\top(I-B)^{-1}d,
\qquad
H_n=a^\top(I-B)^{-1}B_n(I-B)^{-1}d.
\]

`mean_cascade_loss` and `cascade_loss_derivative` implement these equations.
The exact worst-case additive progeny budget is

\[
G^\star=\min_{\zeta,G}\;G
\quad\text{s.t.}\quad
\mathbf 1+B\zeta\preceq\zeta,
\quad \zeta\preceq G\mathbf 1.
\]

`cascade_budget_lp` solves this LP. With the strictly positive right-hand side
\(\mathbf1\), it is feasible precisely for \(\rho(B)<1\), so a separate
spectral constraint is redundant. The included directed-chain and fan-out
examples show that \(\rho(B)=0\) can coexist with large cumulative cascades.

For a finite observable depth \(D\), the matching quantity is

\[
H_D=a^\top\left(\sum_{k=0}^{D}B^k\right)d.
\]

`truncated_cascade_loss` and `truncated_cascade_budget` implement this without
requiring \(\rho(B)<1\). Infinite and finite-depth predictions are reported
side by side rather than selecting the one that fits better after the fact.

For the scalar closed-chain experiment, reaching all \(N\) robots right-censors
the next propagation opportunity. If cascade \(r\) reaches \(L_r\) unique
robots, the boundary-aware estimate is

\[
S=\sum_r(L_r-1),\qquad
F=\sum_r\mathbf1\{L_r<N\text{ and resolved}\},\qquad
\widehat\phi=\frac{S}{S+F}.
\]

`estimate_finite_chain_propagation` implements this likelihood accounting.
Temporal censoring below the fleet boundary is reported as indeterminate and
is never converted into a propagation failure.

If every training cascade reaches the fleet boundary, then
\(\widehat\phi=1\) and \(M_N(1)=N\). A held-out sample in which every cascade
also reaches \(N\) has zero error by construction. The implementation labels
this case `descriptive_not_predictive`; it is a boundary identity, not evidence
that the finite model generalizes.

Event identifiers are local to one rollout. Concatenating independent event
tables before fitting silently aliases unrelated parents unless duplicate IDs
are rejected. `estimate_direct_branching` now rejects that input, while
`pool_direct_branching_events` combines rollout-level sufficient statistics.

There is a multi-type counterpart to the scalar forest identity. For complete
typed forests, let \(n\) count all events by type and \(r\) count roots. Direct
edge conservation gives

\[
n^\top B=n^\top-r^\top,
\qquad
\left(\frac{r}{P}\right)^\top(I-B)^{-1}\mathbf1=\frac{E}{P}.
\]

Thus a training primary-mix resolvent restates training mean progeny for *any*
typing. `diagnose_multitype_forest_identity` detects this. Type structure can
still change the worst-case budget \(G^\star\), and only independent trees can
test transfer.

## Assumptions not proven by this code

- Little's-law substitution is a quasi-stationary mean-field approximation.
- Direct offspring must be attributed by proximate blocking, not merely by a
  fixed time horizon.
- The 2D shield must be shown to compose across robots and integration steps.
- The scalar parameter-free curve is not automatically universal for a
  non-normal multi-type matrix. The general fold condition is the elasticity
  identity returned by `fold_elasticity_identity`.
- The open-system layer now implements HAC queue-trend intervals, regeneration
  diagnostics, and across-seed bootstrap summaries. Adaptive-policy confidence
  sequences, metastability proofs, and static hysteresis remain beyond this
  reference layer.
- Fitting \(B\) and checking total progeny on the same homogeneous cascade is
  algebraically circular. Paper-facing validation must fit on training
  rollouts and score on separate held-out rollouts.

The quasi-stationary Little's-law closure is now directly diagnosable using
`little_law_diagnostic`, but numerical agreement in one operating window is not
a proof of timescale separation. The load-ramp experiment reports matched
empty-start controls so finite backlog memory is not mislabeled as bistability.
