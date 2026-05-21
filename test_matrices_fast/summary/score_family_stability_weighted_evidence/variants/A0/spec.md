# FAM-09 A0: diagnostic stability-weighted HM evidence

Date: 2026-04-29
Status: spec + expanded diagnostic prototype; no optimizer / no T3.

## Exact score

For candidate unit vector `v` and window `X in {sk, g1, g2}`:

```text
u_X(v) = ||A_X v||^2 / ||A_X||_F^2
cv_X(v; p) = std_t(u_X(v; S_t)) / max(mean_t(u_X(v; S_t)), eps)
u_X_stab(v) = u_X(v) / (1 + lambda * cv_X(v; p))
```

Then:

```text
Score_A0(v) = HM2(u_g1_stab, u_g2_stab)           at block 1
Score_A0(v) = HM3(u_sk_stab, u_g1_stab, u_g2_stab) at block >= 2
```

Prototype defaults: `p=0.50`, `n_subsamples=12`, `lambda=1.0`, deterministic
RNG seed. The expanded diagnostic run covers `static-cex`, `diffuse-diffuse`,
and `mixed-tail-sharp` at blocks `[1, 6, 12, 31]`, slots `[1, 2]`.
The candidate panel is inherited from DIAG-03: oracle, S6 opt, mgain,
combined, rowcheat, and sketch when present.

## Hypothesis addressed

FAM-09 addresses H1/H2 through evidence augmentation: high raw-HM non-oracle
candidates may be landscape winners because their row-energy evidence is
fragile under subsampling. Penalizing CV should lower those winners before the
HM aggregator chooses them.

## Expected oracle-vs-winner gap sign

Expected diagnostic sign at b31 after a real optimizer exists:

```text
Score_A0(oracle_v2_proj) - Score_A0(winner) >= 0
```

For this A0 prototype, there is no newly optimized winner. The reported
observable is rank/rerank among the existing candidate panel; oracle rank should
improve or remain stable as `lambda` increases on high-entropy failure blocks.

## Acceptance for A0

- Generate `diagnostics/prototype_A0.csv` and
  `diagnostics/prototype_A0_summary.md`.
- Report whether `lambda=1` improves oracle panel rank versus raw HM on the
  small block set.
- Do not run long T3 benches from A0.

## A0 result

The expanded run is a diagnostic signal, not a ship result:

- Oracle panel rank improves on 8 of 24 matrix/block/slot panels, is unchanged
  on 15, and worsens on 1.
- The strongest positive case is `mixed-tail-sharp`: 4 improved, 4 unchanged,
  0 worsened, mean rank movement `+0.875`.
- The visible regression is `static-cex` block 31 slot 2: raw winner `oracle`,
  stability-weighted winner `sketch`, oracle rank `1 -> 2`.

Verdict: A0 shows the DIAG-03 stability signal can be wired into the evidence
layer, but the all-window Monte-Carlo CV penalty is not yet safe enough for
optimizer integration or T3. Next candidate should test deterministic CV
estimation and window-specific penalties, especially `g2`-only or lower
`lambda` for sketch/current evidence.
