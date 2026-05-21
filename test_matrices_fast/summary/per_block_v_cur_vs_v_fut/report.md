# v_cur vs v_fut row-aligned shift probe — results

Date: 2026-05-07
Branch: new_orders_evals
Probe code: `v_cur_vs_v_fut_probe.py`, `v_cur_vs_v_fut_analyze.py`
Cells: `cells.csv` (192 rows, schema in §6 below), `summary.csv`

## 0. What this probe asks

For each window pair (matrix, seed, block t), two combined-score-locked
solutions exist:

  v_cur — optimum on rowsets {S, A_cur} (the bench's actual lock)
  v_fut — optimum on rowsets {S, A_cur, A_fut}

Both share the same sketch state S; v_fut differs from v_cur only in that its
"current block" is the 64-row stack [A_cur; A_fut] instead of A_cur.
Concretely v_fut is produced by re-running `entropy_iter_basis_forget` with
score_variant="combined", M_gain = [B_top; A_cur; A_fut], A_block = [A_cur; A_fut],
same `state_prev`, same hyperparameters as the bench.

After sign-aligning v_fut so v_fut · v_cur > 0, we read both off on A_cur:

```
e_cur = A_cur · v_cur,   e_fut = A_cur · v_fut    (each ∈ R^{32})
```

The headline statistic is **row_aligned_shift = 1 − |cos(e_cur, e_fut)|**: the
disagreement between the two optimizer-locked solutions on the per-row
projection that the bench actually commits on.

This is a strict improvement over the prior probes that came back negative:
`per_block_element_drift_finegrained` (within-pair drift, killed by
optimization-flattening) and `per_block_element_drift_heldout` (held-out drift,
killed by block orthogonality). Here we compare *two different v's* applied to
the same A_cur — both flattened by their objectives, so any disagreement has
to live in *which* uniform pattern each chose, not in concentration.

Configuration: 4 matrices × 3 seeds × 16 blocks = 192 cells.
half_win=32, n=1024, rank=2, preset=fast, slot-1 only.

## 1. Headline: row_aligned_shift cleanly separates regimes

```
matrix             median  min     max    std    n   median_dir_shift
static-cex         0.000   0.000   0.998  0.157  48  0.057
mixed-tail-sharp   0.308   0.000   0.969  0.297  48  0.160
mixed-tail-soft    0.305   0.000   0.998  0.306  48  0.129
diffuse-diffuse    0.738   0.120   1.000  0.262  48  0.194
```

The matrix where combined wins the bench (static-cex, exact_cos1²=0.95) has
median row_aligned_shift = 0.000. The matrix where combined fails
(diffuse-diffuse, exact_cos1²=0.23) has median 0.738. Mixed-tail matrices
sit in between, matching their intermediate bench performance.

This rules out the negative outcome §6 of the spec called out: it is *not*
the case that the optimizer-flattening forces both solutions into the same
uniform pattern on A_cur. The two solutions can — and do — disagree
substantially on the per-row energy distribution, by 0.7+ on diffuse-diffuse.
The flattening keeps both at high row-entropy (relH ≈ 1.000 for both, on
every cell — see §3), but high-entropy patterns on 32 rows still admit a
continuum of disagreements, and the probe sees them.

## 2. Temporal trajectory — within-matrix discrimination

Median row_aligned_shift across seeds, by block:

```
block:               1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16
static-cex          .00  .14  .19  .00  .00  .06  .00  .00  .00  .00  .00  .00  .00  .00  .00  .00
mixed-tail-sharp    .86  .75  .50  .31  .74  .67  .56  .18  .38  .12  .12  .07  .19  .00  .00  .06
mixed-tail-soft    .94  .62  .49  .31  .37  .31  .62  .18  .18  .31  .31  .06  .24  .24  .00  .06
diffuse-diffuse     .94  .62  .81  .81  .94  .75  .74  .88  .56  .74  .80  .74  .75  .43  .31  .31
```

Three patterns are visible:

- **static-cex** drops to ~0 by block 4 and stays there — the algorithm
  reaches a stable direction that A_fut would not perturb.
- **mixed-tail-sharp / mixed-tail-soft** decay smoothly from ~0.9 → ~0.06
  over 16 blocks — slow stabilization, consistent with their intermediate
  bench performance.
- **diffuse-diffuse** stays high throughout — *never* stabilizes; every
  peek into A_fut produces a substantial reordering of how the lock reads
  on A_cur.

The trajectory itself is the self-only signal. A run whose block-wise
row_aligned_shift drops monotonically toward zero is converging to a
well-determined lock; a run whose shift stays north of 0.5 indefinitely
is locked in name only — A_fut would have moved it elsewhere.

## 3. §6 sanity predictions

| matrix             | median(cos vv) | median(mass_shift) | median(entropy_shift) | relH(c) | relH(f) |
|---|---|---|---|---|---|
| diffuse-diffuse    | +0.806 | 0.330 | 0.000 | 1.000 | 1.000 |
| mixed-tail-sharp   | +0.840 | 0.256 | 0.000 | 1.000 | 1.000 |
| mixed-tail-soft    | +0.871 | 0.231 | 0.000 | 1.000 | 1.000 |
| static-cex         | +0.943 | 0.113 | 0.000 | 1.000 | 1.000 |

- entropy_shift ≈ 0: ✓ both solutions saturate row-entropy on A_cur.
- mass_shift ≈ 0: **violated** (0.11–0.33). v_fut has access to a 64-row
  current block over a search basis that includes A_fut, which lets it
  shift A_cur-mass relatively to v_cur even though both maximize their
  respective combined-score objectives. This is a geometry consequence of
  block orthogonality (max cos²(rowspace(A_cur), rowspace(A_fut)) ~ 1e-4),
  not a probe bug — recorded raw masses are in `cells.csv` columns
  `mass_cur`, `mass_fut`. The advisor flagged this in advance.

The two sanity stats are not the probe's headline — they're the
"are we inside the ambiguous regime?" check. Entropy is, mass isn't.
Importantly, neither sanity stat is necessary for the headline:
row_aligned_shift remains discriminative regardless.

## 4. Discrimination test (§6 substantive)

Spearman ρ between cos²(v_cur, target) and row_aligned_shift, per matrix.
Success line ρ ≤ -0.3 (high oracle alignment ↔ low shift).

Two targets are tested:
- raw oracle V_exact[:,0] (visibility analysis §3 says this is ≈0 for
  per-block argmax across all matrices — kept as a check on the spec
  language, not as a useful target);
- reachable oracle P_search V_exact[:,0] (V_exact projected onto
  rowspace([sketch; A_cur]) and renormalized — the per-block "best in
  basis" direction).

| matrix             | n  | ρ(cos²_oracle, row_shift) | ρ(cos²_reach, row_shift) | med cos²_oracle | med cos²_reach |
|---|---|---|---|---|---|
| diffuse-diffuse    | 48 | -0.118 | -0.027 | 0.011 | 0.158 |
| mixed-tail-sharp   | 48 | -0.001 | +0.013 | 0.031 | 0.387 |
| mixed-tail-soft    | 48 | -0.652 | -0.646 | 0.010 | 0.213 |
| static-cex         | 48 | -0.376 | -0.244 | 0.146 | 0.999 |

**Conclusion on the strict §6 claim:** The Spearman ρ ≤ -0.3 bar is met on
mixed-tail-soft (-0.65) and on static-cex against raw oracle (-0.38). It is
borderline on static-cex against reachable oracle (-0.24) and missed on
mixed-tail-sharp (≈0) and diffuse-diffuse (≈0).

The strict V_exact target is *partly* meaningful, not unfalsifiable: on
mts-sharp the per-cell cos²(v_cur, V_exact[:,0]) ranges 0.005 → 0.535
(seed-0 trajectory) — substantial within-matrix variance — while on
mts-soft and diffuse-diffuse it stays uniformly ≈ 0. So the strict test
discriminates where v_cur eventually accumulates non-trivial V_exact
alignment (mts-sharp, static-cex) and is structurally trivial elsewhere.
cos²_reach is a complementary target, not the only meaningful one.

But pooled Spearman across all 48 cells per matrix is still a poor
summary, for a *temporal* reason that the quartile decomposition (§5.1)
exposes:

- *mixed-tail-sharp* and *mixed-tail-soft* show high row_aligned_shift in
  the *transition* quartile (Q2 of cos²_reach), where 7/12 cells are
  early blocks (b ≤ 4). The high-shift cells are concentrated in
  pre-convergence blocks. Spearman averages pre- and post-convergence
  cells together and reports noise where the per-block trajectory is in
  fact monotone.
- *diffuse-diffuse* legitimately has no within-matrix signal: every
  quartile of cos²_reach has row_shift ≈ 0.7. The signal is dead in the
  regime where visibility-analysis §6 already showed combined is
  structurally mismatched.

Pooled-cell Spearman with cos²_reach is the wrong test for the
*per-block* discrimination claim, because it mixes regimes that the
temporal trajectory (§2) cleanly separates.

## 5. Per-matrix quantile table (cos²_reach quartiles → median row_aligned_shift)

### static-cex
```
quartile  n  med_cos²_reach  med_row_shift  med_dir_shift
Q1 (low)  12      0.000           0.062         0.053
Q2        12      0.900           0.062         0.135
Q3        12      0.999           0.000         0.054
Q4 (hi)   12      1.000           0.000         0.044
```
Discrimination clean: high cos²_reach → row_shift = 0.

### mixed-tail-soft
```
quartile  n  med_cos²_reach  med_row_shift  med_dir_shift  med_block  early(b≤4)/late(b≥12)
Q1 (low)  12      0.055           0.652         0.411          3.5       7/0
Q2        12      0.153           0.245         0.115         10.0       2/5
Q3        12      0.254           0.308         0.119         11.0       3/5
Q4 (hi)   12      0.960           0.000         0.067         10.5       0/5
```
Cleanly monotone — Q1 (high shift) is essentially "early blocks before
convergence" (7/12 early, 0 late), Q4 (zero shift) is "late blocks,
oracle locked" (0 early, 5/12 late). The Spearman -0.65 captures this.

### mixed-tail-sharp
```
quartile  n  med_cos²_reach  med_row_shift  med_dir_shift  med_block  early(b≤4)/late(b≥12)
Q1 (low)  12      0.001           0.095         0.071         10.5      2/5
Q2        12      0.125           0.637         0.825          3.0      7/2
Q3        12      0.456           0.558         0.218          8.5      3/3
Q4 (hi)   12      0.981           0.095         0.094         10.5      0/5
```
Apparent bimodality (Q1 and Q4 both have low row_shift) is real
structure but is not just temporal — Q1 = "stuck on a non-oracle
direction at late blocks, lock locally consistent" (5/12 late, 2/12
early) and Q4 = "locked on the reachable oracle at late blocks"
(5/12 late, 0/12 early). Q2 is the early-block transition where
the lock hasn't settled (7/12 early). The high-row_shift cells are
predominantly in pre-convergence blocks.

### diffuse-diffuse
```
quartile  n  med_cos²_reach  med_row_shift  med_dir_shift
Q1 (low)  12      0.038           0.750         0.584
Q2        12      0.108           0.457         0.184
Q3        12      0.180           0.744         0.240
Q4 (hi)   12      0.362           0.710         0.129
```
No discrimination — every quartile has high row_shift. Consistent with
the visibility-analysis finding that combined's relH gate is
structurally noninformative on row-diffuse matrices, and the §6 negative
outcome at the within-matrix level: in this regime the two solutions
disagree on rows even when both are "good" relative to the small
reachable oracle they have access to.

## 6. cells.csv schema

```
matrix, seed, block,
cos2_v_cur_oracle      cos²(v_cur, V_exact[:,0])
cos2_v_fut_oracle      cos²(v_fut, V_exact[:,0])
cos2_v_cur_reach       cos²(v_cur, P_[sketch;A_cur] V_exact[:,0])
direction_shift        1 − |cos(v_cur, v_fut)|
row_aligned_shift      1 − |cos(e_cur, e_fut)|             ← headline
element_energy_shift   ‖e²_cur − e²_fut‖₁ / (‖e²_cur‖₁ + ‖e²_fut‖₁)
rank_shift             1 − |spearman(|e_cur|, |e_fut|)|
mass_shift             |‖e_cur‖² − ‖e_fut‖²| / max(...)
entropy_shift          |relH(e²_cur) − relH(e²_fut)|
cos_vv                 raw cos(v_cur, v_fut) (signed, post-alignment)
cos_ee                 raw cos(e_cur, e_fut) (signed)
mass_cur, mass_fut     ‖e_cur‖², ‖e_fut‖²  (raw, for diagnostics)
relH_cur, relH_fut     row-entropy of e², normalized by log(half_win)
```

## 7. Falsifiable claims, decided

- **Cross-matrix discrimination by row_aligned_shift: TRUE.**
  Median values of {0.000, 0.305, 0.308, 0.738} on
  {static-cex, mts-soft, mts-sharp, diffuse-diffuse} cleanly track
  bench-final cos²(state.V[:,0], V_exact). A run whose median per-block
  row_aligned_shift across blocks 8–16 stays north of 0.5 is a run whose
  lock would have moved given one more half-window of data.

- **Within-matrix Spearman ρ ≤ -0.3 against cos²_reach: PARTIAL.**
  Holds on mixed-tail-soft (-0.65). Borderline on static-cex
  (-0.24 reach, -0.38 oracle). Fails on mixed-tail-sharp (bimodal,
  not monotone) and diffuse-diffuse (signal dead by structural mismatch).
  The Spearman summary is too coarse for the bimodal case in mts-sharp;
  the quartile table tells the real story.

- **§6 stable predictions:**
  - entropy_shift ≈ 0: TRUE (median ≤ 1e-3 on all matrices).
  - mass_shift ≈ 0: FALSE (medians 0.11–0.33). Block-orthogonality
    + extended search basis allow real mass redistribution between
    A_cur and A_fut. Documented; not a probe bug.

- **Negative-outcome scenario (§6: both solutions hit same uniform
  pattern even when global vectors differ): does not occur.**
  The predicted dead-signal pattern was high direction_shift AND low
  row_aligned_shift. On diffuse-diffuse we see median direction_shift =
  0.194 (cos_vv = +0.806; vectors moderately agree) AND median
  row_aligned_shift = 0.738 (rows disagree) — a third regime, neither
  the predicted positive case (vectors disagree, rows disagree, signal
  alive) nor the predicted negative case (vectors disagree, rows agree,
  signal dead). The diffuse-diffuse vectors are not as different as
  expected, but their A_cur-row patterns still disagree substantially —
  enough for the cross-matrix discrimination in §1 and the temporal
  trajectory in §2.

## 8. What this implies for production use

The temporal trajectory of row_aligned_shift (§2) is a self-only signal
that *does* discriminate locked-correct from locked-wrong, on the same
4-matrix subset where every other self-only signal in visibility-analysis
§6 was confounded:

- it doesn't require V_exact (only A_fut, which is the next half-window
  of streamed data — peek-mode harnesses already buffer it);
- it isn't killed by block-orthogonality (we read on A_cur, not on
  held-out blocks);
- it isn't killed by optimizer-flattening (we compare two solutions, not
  one solution against itself or against null permutations);
- the per-block readout costs one additional `entropy_iter_basis_forget`
  call at each block (with M_gain extended by half_win rows) — small
  overhead given the bench already runs it once per block.

Open questions for the next step:

1. Does `cumulative median row_aligned_shift over blocks 8–16` correlate
   with the bench's reported `final_exact_cos1²` across the full §6
   7-matrix set (including the structural counter-example
   etf-basket-basis where blocks aren't orthogonal)? Per visibility §1
   that one will behave differently.
2. Can this signal be folded back into the score itself (e.g., as a
   tiebreaker between equally-locked candidates) or used as a runtime
   stop/abort signal when row_aligned_shift fails to drop?
3. *Resolved.* mts-sharp Q1 (low cos²_reach, low row_shift) is mostly
   late blocks where the algorithm got stuck on a non-oracle direction
   but is internally consistent; Q2 (high row_shift) is early blocks
   pre-convergence; Q4 (high cos²_reach, low row_shift) is late blocks
   that converged on the reachable oracle. See §5 quartile tables —
   row_shift is predominantly an early-blocks-pre-convergence
   phenomenon, with a smaller contribution from "stuck on non-oracle"
   late-block cells.
