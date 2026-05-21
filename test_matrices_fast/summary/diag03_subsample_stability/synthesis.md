# DIAG-03 subsample-stability diagnostic — synthesis

Probe: `summary/diag03_subsample_stability/probe.py`  
Matrices: etf-basket-basis, residual-spiky-shocks, risk-residual-panel, diffuse-diffuse, static-cex, mixed-tail-sharp, mixed-tail-balanced, mixed-tail-soft  
Blocks (anchor): [1, 6, 12, 31]; half_win=32; slots: 1, 2  
Subsample draws per (frac, candidate): 30  
Subsample fractions: [0.5, 0.75]  
Total candidate-rows: 368

## Mean / variance / IQR by (matrix, block, slot, candidate)

Full per-row dump in `raw.csv`. Below: summary aggregates of
g1 (cur-window) sub-sample mean / CV at p=0.50 across all
matrix-block-slot rows, broken down by candidate.

| candidate | n | mean(g1_p50_mean) | median(g1_p50_cv) | median(g2_p50_cv) | median(diff_p50_cv) |
|---|---|---|---|---|---|
| combined | 64 | 0.005494 | 0.008149 | 0.2106 | 0.008503 |
| mgain | 64 | 0.02911 | 0.3564 | 0.2118 | 0.6387 |
| oracle | 64 | 0.01482 | 0.004065 | 0.004493 | 0.9635 |
| rowcheat | 64 | 2.418e-06 | 0.2026 | 0.9354 | 1 |
| s6_opt | 64 | 0.01112 | 0.003166 | 0.4383 | 6.772 |
| sketch | 48 | 1.003e-05 | 0.2091 | 0.2063 | 1.451 |

## Correlation: predictor -> outcome (Pearson r)

Predictors: raw u_g1, rel_h1, and subsample-instability metrics.
Outcomes: `rep_fail_norm` (next-window replication error,
|u_g1_next - u_g2| / u_g2) and `carry_decay` (1 -
max_j cos^2(v, V_state^{b+1}[:, j])).

### outcome = `rep_fail_norm`

| predictor | Pearson r | n |
|---|---|---|
| `u_g1` | -0.102 | 368 |
| `rel_h1` | -0.307 | 368 |
| `g1_p50_cv` | +0.338 | 368 |
| `g1_p75_cv` | +0.324 | 368 |
| `g2_p50_cv` | -0.161 | 368 |
| `g2_p75_cv` | -0.158 | 368 |
| `diff_p50_cv` | -0.106 | 368 |
| `diff_p75_cv` | -0.041 | 368 |
| `diff_p50_std` | -0.105 | 368 |
| `diff_p75_std` | -0.096 | 368 |

### outcome = `carry_decay`

| predictor | Pearson r | n |
|---|---|---|
| `u_g1` | -0.274 | 368 |
| `rel_h1` | -0.314 | 368 |
| `g1_p50_cv` | +0.311 | 368 |
| `g1_p75_cv` | +0.302 | 368 |
| `g2_p50_cv` | +0.539 | 368 |
| `g2_p75_cv` | +0.537 | 368 |
| `diff_p50_cv` | +0.090 | 368 |
| `diff_p75_cv` | +0.018 | 368 |
| `diff_p50_std` | +0.311 | 368 |
| `diff_p75_std` | +0.287 | 368 |

## Verdict-rule numbers

- `rep_fail_norm`: base_max(|r|) over u_g1/rel_h1 = 0.307; best instability |r| = 0.338 at g1_p50_cv=+0.338; gap = +0.031.
- `carry_decay`: base_max(|r|) over u_g1/rel_h1 = 0.314; best instability |r| = 0.539 at g2_p50_cv=+0.539; gap = +0.225.

## VERDICT: **SIGNAL**  (best gap = +0.225)

Rule: SIGNAL if gap > +0.10 on at least one outcome; WEAK-SIGNAL if -0.05 < gap <= +0.10; else NO-SIGNAL.

## FAM-09 unblock recommendation

UNBLOCK FAM-09. Subsample instability adds predictive
information beyond raw u_X / relH1 on at least one of
{next-window replication, carry-alignment decay}.
Suggested wiring: u_X -> u_X / (1 + lambda * CV(u_X|sub))
with lambda calibrated on the high-entropy §6 matrices.
Acceptance gate per the FAM-09 backlog still applies
(gradient check or derivative-free; improve a
high-entropy failure without §6 regression).

## Toolkit promotion note

Per workflow §6 handoff checklist: any new infra built in a
diagnostic should become a permanent toolkit entry if it is
reusable. The subsample-stability sampler in `probe.py`
(`_subsample_u` + `_summary_stats`) IS reusable: it can
be called from any candidate-vs-evidence audit. Promote
`diagnostic_toolkit.txt §8(p)` from NOT BUILT -> SHIPPED
with this probe as the canonical reference, and keep
the helpers available for FAM-09 prototyping.

