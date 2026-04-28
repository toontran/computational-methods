# Variant D0 spec

Date: 2026-04-28
Family: row-concentration guard (FAM-02).
Identifier in code: `D0` (in `r_sk_g_score.py` variant choices, in
`half_window_sliding_hmean_experiment.py --rsk-variant`).

## Score

```
score_D0(v) = S6(v) · relH1(A_cur v)
```

with

- `S6(v) = HM3(u_sk, u_g1, u_g2)` when sketch present,
  `HM2(u_g1, u_g2)` for block 1 fall-through.
- `u_sk = raw_sk / sk_F2_low`, `u_g1 = raw_g1 / cur_F2`, `u_g2 = raw_g2 / fut_F2`.
- `relH1` = normalized Shannon entropy of `(A_cur v)²` row distribution
  (`hmean_evidence_score.entropy_relH1_value_grad`; same as in S4/S5).

## Hyperparameters

None. D0 is parameter-free.

## Smoothness / domain

Smooth on the sphere wherever (a) all u_X > 0 and (b) `A_cur v` is nonzero
(so the relH1 distribution is well-defined). This is the same domain S6 is
defined on.

When any of u_sk, u_g1, u_g2 falls below `eps = 1e-30`, the score is set to
0 and the gradient to 0 (matches S6 / S4 behavior).

When `A_cur v ≈ 0`, relH1's `entropy_relH1_value_grad` returns 0 with grad ≈ 0
(safeguarded by its own `1e-30` floor on `S = Σ y²`).

## Prediction

- residual-spiky-shocks cos1² climbs from S6's 0.266 toward iSVD's 0.637.
  iSVD wins because it integrates across blocks; relH1 disqualifies the
  single-row-dominated v's S6 currently picks.
- risk-residual-panel similar lift (no S6 number on overview §6 but online
  is 0.064; iSVD 0.471).
- mixed-tail-sharp / -balanced / static-cex / etf-basket-basis: NEUTRAL or
  small effect (small loss possible if S6 optimum has notably lower relH1
  than competing candidates; small win possible if relH1 tightens the basin
  P4 around the oracle).
- diffuse-diffuse: NEUTRAL (uniform rowspace ⇒ relH1 ≈ 1 broadly).

## Acceptance criteria

T1 (FD grad check): rel_err < 1e-7 at float64 on random unit v across blocks
{1, 2, 12, 31} of mixed-tail-sharp.

T2 (per-block, with `--no-oracle-warmstart`): on the canonical 3-matrix probe
(static-cex, mixed-tail-sharp, diffuse-diffuse) at blocks {1, 2, 6, 12, 31},
the oracle's score under D0 must rank ABOVE `hm_triplet_raw_best`. This is
the no-row-cheat invariant — STOP and kill D0 if violated on any block.

T3 (streaming bench, win64 sliding): cos1² on residual-spiky-shocks ≥ iSVD's
0.637 AND no matrix in the §6 table regresses by >0.05 cos1² vs S6 (frozen
in `baseline/`).

## Decision

- **Ship** if T1, T2, T3 all pass.
- **Iterate** if T1, T2 pass and T3 partially passes (residual-spiky-shocks
  improves substantially but does not fully reach iSVD), with hyperparameter
  options being out-of-scope (D0 has none) — escalate to D1 / D2.
- **Kill** if T2 fails on any block, OR if T3's residual-spiky-shocks does
  not improve, OR if any §6 matrix regresses by >0.05 cos1² vs S6.
