# Plateau-width quantifier (INFRA-05) — synthesis

Date: 2026-04-28
Probe: `plateau_width_probe.py`
Backlog: INFRA-05 in `summary/overview/score_family_workflow.txt` §5
Resolves: Q2 in `summary/overview/score_design_overview.txt` §7
Toolkit gap: `summary/overview/diagnostic_toolkit.txt` §8 (c)

## What this probe measures

For a fixed (matrix, block, score variant), draw N=200 random Stiefel frames
V on St(d, r) sampled uniformly inside the union rowspan
B_union = rowspan(A_sketch ∪ A_cur ∪ A_fut), keep those with
`score(V) ≥ τ · score(V_opt)` (default τ=0.95), and report the principal
angles of each accepted V to the projected oracle frame V_oracle. The
distribution of `min principal cos²` across accepted samples is a direct
quantification of the basin shape: tightly clustered near 1 → narrow basin
near oracle; spread (or clustered far from 1) → wide / multimodal basin
disconnected from oracle.

Default settings: variant=S6, rank=1, half_win=32, τ=0.95, sampling within
B_union, no-oracle-warmstart on the V_opt computation.

## Headline numbers (S6, r=1, blocks 6/12/31)

All cells: N=200 accepted on N=200 target via pure rejection (no MCMC needed).

| matrix              | block | accept_rate | opt min_cos² | sample mean | p50    | p90    | p99    | mean angle (rad) | p90 angle |
|---------------------|------:|------------:|-------------:|------------:|-------:|-------:|-------:|-----------------:|----------:|
| static-cex          |    6  |   4.24%     |   0.561      |   0.057     |  0.052 |  0.105 |  0.145 |   1.340          |  1.435    |
| static-cex          |   12  |   3.77%     |   0.402      |   0.065     |  0.061 |  0.100 |  0.146 |   1.320          |  1.389    |
| static-cex          |   31  |   2.84%     |   0.260      |   0.072     |  0.069 |  0.100 |  0.142 |   1.303          |  1.352    |
| mixed-tail-sharp    |    6  |   4.59%     |   0.346      |   0.032     |  0.023 |  0.073 |  0.153 |   1.415          |  1.531    |
| mixed-tail-sharp    |   12  |   4.44%     |   0.054      |   0.024     |  0.015 |  0.064 |  0.097 |   1.439          |  1.533    |
| mixed-tail-sharp    |   31  |   3.65%     |   0.000      |   0.014     |  0.007 |  0.034 |  0.074 |   1.474          |  1.554    |
| diffuse-diffuse     |    6  |   6.49%     |   0.000      |   0.013     |  0.007 |  0.030 |  0.083 |   1.479          |  1.557    |
| diffuse-diffuse     |   12  |   6.74%     |   0.076      |   0.019     |  0.010 |  0.049 |  0.112 |   1.459          |  1.551    |
| diffuse-diffuse     |   31  |   4.83%     |   0.103      |   0.033     |  0.025 |  0.078 |  0.124 |   1.411          |  1.536    |

`min_cos²` is the smallest principal cos² between span(V_sample) and the
projected oracle frame; for r=1 it equals `cos²(v_sample, v_oracle_proj)`.
`sample mean / p50 / p90 / p99` are over the 200 accepted samples.

## Verdict per matrix at b31 (terminal)

The basin is **wide and multimodal** on all three matrices, not a narrow
basin near the oracle.

- **static-cex**:
  - 200/200 accepted at 2.84% rejection rate. The distribution of
    `min_cos²` is tightly clustered around 0.07 (p50=0.069, p90=0.10,
    p99=0.14, max ≈ 0.15). Equivalently the principal angle to oracle
    sits near π/2 with a *narrow* spread (p50=1.305, p90=1.352 rad ≈
    77.5°). The optimizer's V_opt (cos² = 0.260) is a positive outlier
    relative to the random-sample distribution — it is closer to oracle
    than 99% of in-sublevel samples — but it is still not pinned to the
    oracle. The plateau shape: a thick high-score shell that touches
    oracle only at a thin sliver, and the bulk of the shell is
    near-orthogonal.
  - Operationally: many high-score, near-orthogonal-to-oracle frames
    exist; the optimizer drifts within them. Lifting the score off
    rank-1 (FAM-01) is the design lever.

- **mixed-tail-sharp**:
  - The widest plateau and the worst optimizer landing of the three.
    Sample distribution is even tighter against zero (mean=0.014,
    p50=0.007, p90=0.034) and the optimizer itself lands at *exactly*
    cos² = 0.000 — i.e. fully orthogonal to oracle. Mean angle 1.474 rad
    (≈ 84.4°). Acceptance rate 3.65%: the high-score level set is
    geometrically large.
  - This is the M1/M2 mechanism (combined_obj_sketch_bias_synthesis.txt
    references M1 phi de-rewarding oracle and M2 carry pinning slot-2 to
    span(V_state)) playing out in the S6 score: the score's wide
    high-score shell happens to live almost entirely OFF oracle.
    Verifies §3 of `score_design_overview.txt`: "the dominant reason S6
    underperforms on the streaming bench despite the score correctly
    ranking the oracle in T2" is the plateau drift, not a ranking bug.

- **diffuse-diffuse**:
  - At b31 the optimizer reaches cos² = 0.103 — closer than mixed-tail-
    sharp's 0.000 but still far from oracle. The sample distribution
    (mean=0.033, p50=0.025, p90=0.078) shows the basin sits at the same
    near-zero shell. This is the M3 regime (`overview §1bis`): the
    population top SV requires integration across A_cur and A_fut
    blocks, which a single-window score cannot recover at b31. The
    plateau is wide AND oracle is genuinely unreachable from a
    single-window scalar score.

## What the data says about the basin shape

Across all 9 (matrix, block) cells, accepted samples cluster in a thin shell
of `min_cos²` (interquartile widths ≤ 0.05) but that shell sits at small
cos² (≤ 0.1 in the worst cases). The basin is not narrow-around-oracle; it
is **wide-but-multimodal-and-displaced**:

- The score's high-score level set is a (d−1 = 62)-dimensional codim-1
  manifold (overview §3). When intersected with the union rowspan
  B_union, it produces a thick shell.
- Inside that shell, oracle is one of many points but is *not* the global
  basin center. The bulk of in-sublevel volume sits at small cos² to
  oracle.
- The optimizer's V_opt is sometimes (static-cex, b6/b12/b31) a bit
  closer to oracle than the random-sample bulk (and is a star outlier on
  the boxplot), but it is still firmly off-oracle on the matrices where
  the streaming bench reports failure (mixed-tail-sharp b31: 0.000,
  diffuse-diffuse b31: 0.103). This is consistent with the §3 picture:
  the optimizer finds a high-score point that is not the oracle.

## Block evolution

- static-cex: opt cos² 0.561 → 0.402 → 0.260 (b6 → b12 → b31). The
  optimizer drifts AWAY from oracle as the carry matures. Sample
  distribution is essentially flat across blocks (p50 0.052/0.061/0.069),
  i.e. the plateau shape is stable; only the optimizer's perch on it
  drifts.
- mixed-tail-sharp: opt cos² 0.346 → 0.054 → 0.000. The drift is
  monotone-and-catastrophic, consistent with the S6→online slot-2 gap
  on this matrix (overview §6).
- diffuse-diffuse: opt cos² 0.000 → 0.076 → 0.103. Mild improvement as
  the carry contributes more information, but the basin is wide
  throughout and oracle sits at the edge of the high-score shell. The
  optimizer never approaches oracle.

## Acceptance rates and method

All 9 cells were satisfied by pure rejection with acceptance rates between
2.84% and 6.74% (well above the 1% threshold for MCMC fallback). MCMC was
not exercised. Rates are higher on diffuse-diffuse (6–7%) than on the
counterexample matrices (3–4%), meaning the high-score level set occupies
*more* of B_union on diffuse — consistent with diffuse rowspace having
many candidate "balance" directions.

## Implications for FAM-01

This is the pre-requisite probe for FAM-01 (rank-r lift). The central
hypothesis of FAM-01 is that lifting to higher r shrinks the codim-1
plateau because a rank-r Grassmannian-invariant score has a lower-dim
high-score level set on St(d, r). With these b31 / r=1 numbers as the
baseline, the FAM-01 acceptance criterion ("plateau-width shrinks with
r") becomes operational:

- **Mixed-tail-sharp** is the cleanest single-matrix discriminator.
  Baseline at r=1: opt min_cos² = 0.000, sample p90 = 0.034, mean angle
  1.47 rad. A successful FAM-01 lift to r=3 should push the optimizer's
  min_cos² up well off zero AND tighten the in-sublevel sample
  distribution toward higher cos². If the lift produces no plateau
  improvement on mixed-tail-sharp by these numbers, FAM-01 is the wrong
  fix and FAM-03 (subspace-trace) is the next candidate per the §5
  sequencing recommendation.
- **Static-cex** baseline opt min_cos² = 0.260 with sample p99 = 0.142
  is a useful complementary signal — already partially-oracle-aligned,
  so the lift should both raise the optimizer's cos² and compress the
  sample distribution upward.
- **Diffuse-diffuse** baseline tells us the lift is necessary but may
  not be sufficient: the M3 mechanism (single-window scores blind to
  cross-block integration) is structural; the plateau probe will improve
  with r but cos² may stay capped below oracle without iSVD-style
  cross-block integration.

## Re-run instructions

Probe is at `plateau_width_probe.py` (test_matrices_fast/). Defaults match
the §6 spec:

```
python plateau_width_probe.py \
  --matrices static-cex mixed-tail-sharp diffuse-diffuse \
  --blocks 6 12 31 --num-samples 200 --variant S6 --rank 1 \
  --no-oracle-warmstart
```

To extend to the §6 table:
```
python plateau_width_probe.py \
  --matrices static-cex mixed-tail-sharp mixed-tail-balanced mixed-tail-soft \
             diffuse-diffuse etf-basket-basis residual-spiky-shocks \
  --blocks 31 --num-samples 200 --variant S6 --rank 1 --no-oracle-warmstart
```

To compare aggregator (AB-01 follow-up):
`--variant S6_GM`.

To check the rank-r lift (after FAM-01 lands):
`--rank 2` (or higher); the per-CSV columns generalize automatically and
`min_cos²` becomes the smallest principal cos² across r principal angles.

Outputs land in `summary/infra_plateau_width/`. Per-matrix CSVs are
`{matrix}_b{block}_{variant}_r{rank}.csv` with one row per accepted sample
(score, per-column cos², per-column angle_rad, sampling method). Cross-
matrix boxplot at the chosen block is `summary_plot.png`. JSON dump of all
per-(matrix, block) summaries is `summary.json`.
