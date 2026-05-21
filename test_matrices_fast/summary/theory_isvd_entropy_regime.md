# iSVD Entropy-Regime Limitation

Date: 2026-04-30
Backlog: `summary/overview/score_family_workflow.txt` [TH-04]
Resolves: `summary/overview/score_design_overview.txt` Q18

## Verdict

iSVD is the right baseline for low-support, spike-like signal, but it is
not a sufficient objective for finite-window recovery when the desired
direction is diffuse and window-stable while competing nuisance directions
are high-leverage and transient.

The shortcoming is not that SVD is "wrong" in population. The shortcoming
is the streaming finite-window selection rule:

```text
choose directions with largest ||M_gain x||^2
```

This rule cannot tell a repeatable diffuse direction from a one-window
leverage spike if the spike has larger observed stacked norm in the current
gain matrix. A different scoring function is motivated exactly when it uses
evidence that iSVD discards: row-response entropy/support, current-future
balance, cross-window replication, robustness to top-row leverage, or
subspace/frame structure.

This is a complementarity claim, not a universal dominance claim. The
low-sample-entropy regime remains iSVD/SVD territory.

## Finite-Window Separation Model

Let `u, v in R^d` be orthonormal candidate directions. In one streaming step,
iSVD sees a stacked gain matrix

```text
M_gain = [B_prev; A_cur]
```

and selects `u` over `v` whenever

```text
||M_gain u||^2 > ||M_gain v||^2.
```

Consider a two-window model with `m` current rows and `m` future rows:

```text
A_cur u = L e_1,        A_fut u = 0
A_cur v = a 1_m,        A_fut v = a 1_m
B_prev u = beta_u z_u,  B_prev v = beta_v z_v
```

where `e_1` is a single-row spike, `1_m` is a diffuse response over all
rows, and `z_u,z_v` are unit carry responses. Then

```text
||M_gain u||^2 = beta_u^2 + L^2
||M_gain v||^2 = beta_v^2 + m a^2.
```

Thus iSVD selects the spike direction `u` whenever

```text
L^2 - m a^2 > beta_v^2 - beta_u^2.              (1)
```

But a forward-stability or two-window evidence objective selects `v`. For
example, the simple score

```text
S_stable(x) = min(||A_cur x||^2, ||A_fut x||^2)
```

gives

```text
S_stable(u) = 0,       S_stable(v) = m a^2.
```

So under (1), the same finite data make iSVD choose `u` even though the
stable two-window oracle chooses `v`.

The entropy gap is explicit. The row-response distribution of `u` in
`A_cur` has effective support `1` and top-row share `1`; the distribution of
`v` has effective support `m` and top-row share `1/m`. In high dimension,
the effect is amplified: if there are many nuisance spike directions, the
largest one-window spike has an extreme-value advantage, while the diffuse
direction's row average concentrates only at rate `sqrt(m)`. Finite windows
therefore create a real selection bias toward transient leverage.

Responsible assumptions:

- leverage: a nuisance direction can put most observed energy in one row;
- sample size: the window is small enough that one spike beats diffuse
  accumulated energy;
- spectral gap: iSVD follows the largest observed gain, even if that gap is
  a finite-window artifact;
- response entropy: low effective support is not penalized by stacked norm;
- window stability: iSVD's gain score does not require the current response
  to replicate in the future window.

## Empirical Audit

Sources:

- terminal iSVD cosines: `summary/bench_matrix_sweep_value_only_online/*_win64.json`;
- entropy/top-row/balance audit: `summary/infra_oracle_entropy_audit/audit.csv`;
- cross-window replication proxy: `summary/diag03_subsample_stability/raw.csv`.

Definitions:

- `terminal exact cos0^2/cos1^2`: final block iSVD alignment with
  `V_exact[:,0:2]`;
- `visible eff_frac`: median effective row support divided by visible rows
  on `[A_cur; A_fut]`, over audited blocks and slots;
- `top-row share`: median largest row-response energy share on visible rows;
- `cur/fut balance`: median `min(energy_cur, energy_fut) /
  max(energy_cur, energy_fut)`;
- `rep_fail_norm`: median normalized next-window replication error from
  DIAG-03.

| matrix | iSVD terminal exact cos0^2/cos1^2 | candidate | visible eff_frac | top-row share | cur/fut balance | rep_fail_norm |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| static-cex | 0.045/0.044 | oracle | 1.000 | 0.016 | 1.000 | 1.58e-09 |
|  |  | iSVD/mgain | 0.016 | 0.998 | 1.37e-06 | 2.66e-07 |
| diffuse-diffuse | 0.911/0.489 | oracle | 1.000 | 0.016 | 1.000 | 2.25e-09 |
|  |  | iSVD/mgain | 0.275 | 0.113 | 1.75e-04 | 1.38e-07 |
| mixed-tail-sharp | 0.078/0.013 | oracle | 1.000 | 0.016 | 1.000 | 2.13e-09 |
|  |  | iSVD/mgain | 0.018 | 0.982 | 5.04e-05 | 9.52e-08 |
| mixed-tail-balanced | 0.182/0.060 | oracle | 1.000 | 0.016 | 1.000 | 1.16e-09 |
|  |  | iSVD/mgain | 0.026 | 0.889 | 8.11e-05 | 9.34e-08 |
| mixed-tail-soft | 0.359/0.003 | oracle | 1.000 | 0.016 | 1.000 | 1.55e-09 |
|  |  | iSVD/mgain | 0.049 | 0.750 | 1.42e-04 | 6.26e-08 |
| etf-basket-basis | 1.000/0.205 | oracle | 0.654 | 0.059 | 0.879 | 1.38e-10 |
|  |  | iSVD/mgain | 0.637 | 0.066 | 0.768 | 4.26e-10 |
| residual-spiky-shocks | 0.930/0.637 | oracle | 0.456 | 0.122 | 0.778 | 2.11e-09 |
|  |  | iSVD/mgain | 0.216 | 0.231 | 0.002 | 1.92e-08 |
| risk-residual-panel | 0.611/0.471 | oracle | 0.220 | 0.322 | 0.522 | 2.25e-09 |
|  |  | iSVD/mgain | 0.025 | 0.879 | 0.003 | 7.57e-09 |

## Reading

The high-entropy counterexample pattern is strong on `static-cex` and the
mixed-tail matrices. The oracle directions are diffuse, balanced across
current and future windows, and nearly perfectly replicating. The iSVD/mgain
directions are low-support, top-row dominated, and current/future imbalanced.
Terminal iSVD alignment then collapses, especially on `static-cex`,
`mixed-tail-sharp`, `mixed-tail-balanced`, and slot 2 of `mixed-tail-soft`.

`diffuse-diffuse` is not a clean iSVD failure in slot 0: terminal cos0^2 is
0.911. It still shows the same mechanism in the audit, and slot 1 remains
only 0.489. Use it as evidence that iSVD can partially recover diffuse
structure when the observed spectral ordering is favorable, not as evidence
that gain-only scoring is reliable for both slots.

`etf-basket-basis` is a different high-entropy case. iSVD captures slot 0
perfectly but slot 1 only reaches 0.205. The audit does not show the extreme
top-row pathology there; this is more a rank/slot allocation problem than a
pure entropy-regime failure.

The low/boundary cases explain why iSVD remains the established baseline.
`residual-spiky-shocks` and `risk-residual-panel` have low or moderate oracle
support, and iSVD is the strongest policy in the operational table. A
high-entropy scoring function should not be sold as replacing iSVD there.

## Consequence For Score Design

Yes, there are convincing shortcomings to iSVD, but they are conditional:

1. iSVD is vulnerable to finite-window, high-leverage nuisance directions
   because it optimizes observed stacked norm only.
2. The failure is severe when the target direction is diffuse and stable
   across windows but has smaller one-window energy than a transient spike.
3. The right motivation for a different scoring function is not "beat iSVD
   everywhere"; it is "add reliability evidence that stacked norm ignores."

The next score families should therefore be judged by whether they improve
the high-sample-entropy regime without sacrificing the conceded
low-sample-entropy regime. The empirical audit supports using iSVD as the
baseline and as the low-entropy specialist, while motivating entropy,
balance, replication, robust row aggregation, or frame-level evidence for
the high-entropy finite-window regime.
