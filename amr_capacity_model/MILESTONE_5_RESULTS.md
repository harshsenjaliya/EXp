# Milestone 5: Estimator Identifiability and Multi-Type Audit

## Verdict

The two scalar corrections in the external audit are confirmed:

1. rows with \(\widehat\phi=1\), no resolved training cascade, and a
   fleet-saturated validation cohort are algebraic boundary identities, not
   predictions;
2. the fair aggregate comparison is the plain scalar resolvent against the
   boundary-finite estimator, and the present data do not distinguish them.

The claimed supercritical multi-type result is **not confirmed**. Reproducing
the supplied probe exposed two implementation defects: rollout-local event IDs
were concatenated into one namespace, and every descendant was assigned its
root gate. The first inflates offspring counts; the second forces the gate
matrix to be diagonal. After both defects are removed, all substantive
multi-type estimates in the generated quick profile are subcritical.

The useful positive result is different: event start-position typing produces
a directed, non-normal branching matrix whose exact cumulative cascade budget
is much larger than the spectral-radius proxy. This is promising evidence for
the \(G^\star\) contribution, but it is still a quick-profile simulation result,
not a finished paper claim.

## 1. Finite-boundary claim repair

For a finite chain of \(N\) robots,

\[
M_N(\phi)=\sum_{k=0}^{N-1}\phi^k.
\]

Therefore \(M_N(1)=N\). If all training cascades reach the fleet boundary,
\(\widehat\phi=1\) by construction. If the held-out cascades also contain all
\(N\) robots, prediction and measurement both equal \(N\) with no remaining
degree of freedom. The implementation now labels exactly these rows
`descriptive_not_predictive`.

Three rows meet that condition: \(v=2.0\), \(N=32\), and \(N=33\). They are
excluded from predictive scoring.

At \(v=1.8\), the fair comparison is:

| Estimator | Prediction | Held-out measurement | Relative error |
| --- | ---: | ---: | ---: |
| Plain scalar \(1/(1-\widehat B)\) | 18.17 | 17.33 | 4.81% |
| Boundary finite \(M_N(\widehat\phi)\) | 16.76 | 17.33 | 3.29% |
| Boundary estimate with infinite resolvent | 52.50 | 17.33 | 202.88% |

The last row is diagnostic only. It combines a boundary-censored propagation
estimate with an infinite-population prediction and is not a sensible
operational baseline.

Across all 14 non-vacuous speed and density rows:

| Aggregate statistic | Plain scalar | Boundary finite |
| --- | ---: | ---: |
| Mean absolute relative error | 14.03% | 13.73% |
| Wins with a one-percentage-point tie band | 3 | 4 |
| Losses with a one-percentage-point tie band | 4 | 3 |
| Practical ties | 7 | 7 |

The 0.30 percentage-point difference in mean error is not evidence that the
finite estimator predicts better. The corrected figure is
`outputs/finite_depth_truncation.png`.

## 2. Why the supplied multi-type probe reports \(\widehat\rho>1\)

### 2.1 Rollout-local ID collision

The supplied probe flattens all training rollouts before estimation:

```python
tr = tuple(event for rollout in train for event in rollout)
estimate_direct_branching(tr, number_of_types=k)
```

Each simulator rollout starts `event_id` at zero. In the old estimator, a
dictionary and set keyed only by `event_id` collapse unrelated parents from
different runs. Child records from every rollout remain in the numerator while
many eligible parents disappear from the denominator. This can inflate every
row of \(\widehat B\) and its spectral radius.

The estimator now rejects duplicate event IDs with an explicit error. The
correct interface is `pool_direct_branching_events`, which estimates sufficient
statistics inside each rollout namespace and then sums those statistics.

### 2.2 Root-gate inheritance is diagonal

The supplied gate retyping assigns one gate label to the root and copies that
same label to every descendant in the cascade. Every observed edge therefore
has the form \(k\to k\), so

\[
\widehat B_{kj}=0\qquad(k\ne j).
\]

This matrix cannot represent propagation between gates. It is retained in the
new experiment as `root_gate_control`, a negative control whose off-diagonal
mass is asserted to be exactly zero. The supplied implementation also stores
the source-to-gate map in one global dictionary that is cleared for each seed,
so earlier rollouts are retyped with the final seed's map.

### 2.3 Corrected reproduction

As a minimal apples-to-apples diagnostic, keeping the supplied warm-up filter,
seeds, and held-out calculation but replacing concatenation with per-rollout
pooling changes the reported S/R result as follows:

| Configuration | Supplied S/R \(\widehat\rho\) | Corrected S/R \(\widehat\rho\) | Held-out progeny |
| --- | ---: | ---: | ---: |
| 16 gates, 0.015 requests/s/gate | 2.457 | 0.855 | 7.00 |
| 12 gates, 0.05 requests/s/gate | 1.710 | 0.657 | 5.09 |
| 16 gates, 0.05 requests/s/gate | 1.710 | 0.657 | 3.86 |

This minimal repair can still orphan a parent at the warm-up boundary, so the
paper-facing experiment instead selects complete roots in a fixed window. The
precise values change with that cohort rule, but the regime conclusion does
not: correct pooling removes the reported supercritical result.

## 3. A stronger identifiability result

The scalar chain identity generalizes to every event typing. Let \(n_a\) be the
number of eligible causal events of type \(a\), let \(r_a\) be the number of
primary roots of type \(a\), and estimate direct offspring by

\[
\widehat B_{ab}=
\frac{\text{number of }a\to b\text{ direct edges}}{n_a}.
\]

For a completely observed single-parent forest, direct-edge conservation gives

\[
n^\top\widehat B=n^\top-r^\top.
\]

Whenever the resolvent exists,

\[
\frac{r^\top}{P}(I-\widehat B)^{-1}\mathbf 1
=\frac{n^\top\mathbf 1}{P}
=\frac{E}{P},
\]

where \(P=\sum_a r_a\) and \(E=\sum_a n_a\). Thus the training primary-mix
resolvent equals the observed training progeny for scalar, S/R, spatial, or any
other typing. Multi-type agreement on the training mix is still bookkeeping,
not validation.

All 42 typing-by-arrival rows in the new experiment satisfy this identity to
numerical precision. Independent held-out trees remain necessary. Moreover,
selecting only completed trees from a genuinely supercritical process
conditions on extinction and biases the fitted law toward a subcritical one.
A future supercritical estimator must include unresolved parents through an
exposure- or survival-aware likelihood instead of silently deleting them.

There is also a finite-sample upper bound. Since

\[
n^\top\widehat B=n^\top-r^\top\preceq n^\top,
\]

the Collatz--Wielandt inequality applied to \(\widehat B^\top\) gives
\(\rho(\widehat B)\le1\) on the active types. Under irreducibility, at least one
root makes the inequality strict and \(\rho(\widehat B)<1\). Consequently, no
amount of arrival-rate or fleet-size tuning can produce a defensible
\(\widehat\rho>1\) from this complete-tree maximum-likelihood estimator alone.
That is an estimator limitation, not evidence that the underlying physical
process must be subcritical.

## 4. Correct multi-type experiment

The new quick profile uses six independent replications at each of seven
arrival rates. The first three runs train the model and the final three score
it. Primary roots are selected in a fixed time window, descendants are retained
regardless of their own start time, and only complete trees enter the current
likelihood. All simulations assert zero collisions, zero protected-crossing
violations, and exact fleet mass balance.

Events are typed four ways:

- scalar, as the reference;
- protective stop versus reduced speed (S/R);
- 4, 8, or 16 fixed spatial regions based on each event's own start position;
- inherited root gate, as the diagonal negative control.

Selected results are:

| Typing | Largest \(\widehat\rho\) | Exact \(G^\star\) at 0.8 jobs/s | Largest \(G^\star/[1/(1-\rho)]\) | Minimum active-type count at 0.8 jobs/s |
| --- | ---: | ---: | ---: | ---: |
| Scalar | 0.874 | 7.96 | 1.00 | 836 |
| S/R | 0.848 | 8.24 | 1.26 | 305 |
| Spatial 4 | 0.819 | 9.93 | 1.82 | 48 |
| Spatial 8 | 0.744 | 11.18 | 2.89 | 7 |
| Spatial 16 | 0.633 | 11.34 | 4.22 | 7 |

Here \(G^\star=\lVert(I-B)^{-1}\mathbf1\rVert_\infty\), and its direct solve
matches the LP formulation to \(10^{-9}\) in every subcritical row. The
spectral radius alone understates worst-case cumulative amplification. The
4-region result is the more statistically credible quick-profile estimate
because every active parent type has at least 48 observations; the 16-region
\(4.22\times\) gap is exploratory because its sparsest active type has only
seven parent observations.

The serial corridor does permit temporal fan-out in principle, but it is rare
in this dataset: six fan-out parents occur across 1,144 complete roots, and no
parent has more than two direct children. This corrects the earlier categorical
claim that a single lane cannot fan out, while also explaining why this layout
does not provide robust supercritical evidence.

## 5. What is now supported

Supported by the generated quick-profile evidence:

- duplicate rollout-ID contamination is detected and rejected;
- complete root cohorts and per-rollout sufficient-statistic pooling work;
- inherited root-gate typing is demonstrably diagonal;
- independent held-out progeny scoring is operational;
- spatial typing produces directed non-normal amplification in a subcritical
  regime;
- exact and LP formulations of \(G^\star\) agree;
- the three \(M_N(1)=N\) identities are detected automatically;
- all generated rollouts satisfy the hard safety and accounting invariants.

Not supported:

- a predictive advantage for the finite-chain estimator;
- a supercritical transition in the corrected corridor data;
- a saddle-node fold, faster-is-slower turnover, or capacity collapse;
- the parameter-free fold curve at an independently detected fold;
- industrial external validity.

## 6. Consequence for the paper plan

The merge/fan-out layout is no longer needed merely to prove that a typed
single-lane matrix can be non-normal; the spatial corridor already does that.
It is still the highest-probability way to obtain enough genuine parent fan-out
to estimate a supercritical regime with dynamic range. Before claiming such a
regime, the estimator itself must be extended to use censored exposure rather
than completed-tree selection.

The next paper-grade sequence is therefore:

1. run the current corrected multi-type experiment with the paper profile and
   bootstrap uncertainty, emphasizing the 4-region model unless finer types
   obtain adequate parent counts;
2. implement an exposure-aware censored branching likelihood and verify it on
   synthetic Galton--Watson data with known subcritical and supercritical laws;
3. add a merge or reservation fan-out layout only for robust supercritical and
   finite-population testing;
4. locate an open-system capacity boundary independently from queue drift, then
   test the parameter-free \(R_{\mathrm{sn}}(g/T_0)\) curve at that boundary.

The codebase is ready for these experiments. The central collapse theory is
mathematically coherent, but the simulator has not yet supplied its decisive
empirical result.

## Generated artifacts

- `outputs/multitype_runs.csv`: per-rollout cohort, fan-out, WIP, and invariant
  diagnostics;
- `outputs/multitype_summary.csv`: held-out predictions and matrix diagnostics;
- `outputs/multitype_matrix_entries.csv`: every estimated \(B_{ab}\) entry;
- `outputs/multitype_validation.png`: corrected \(\rho\), \(G^\star\), held-out,
  and non-normality comparisons;
- `outputs/multitype_spatial_matrix.png`: the directed spatial matrix;
- `outputs/multitype_manifest.json`: settings and fail-closed regime findings;
- `outputs/claim_audit.json`: machine-readable publication-claim ledger.
