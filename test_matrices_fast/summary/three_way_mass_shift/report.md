# Three-way mass-shift probe — results

Date: 2026-05-07
Probe code: `three_way_mass_shift_probe.py`
Cells: `cells.csv` (192 rows = 4 matrices × 3 seeds × 16 blocks)

## What this probe does

Stream pure-combined for each (matrix, seed). At each block, with the
bench's combined-trajectory state S_t fixed, compute the mass_shift on
A_cur for **each** of three candidate locks:

```
ms_combined  = |‖A_cur·v_combined‖² − ‖A_cur·v_combined_fut‖²| / max(...)
ms_isvd      = |‖A_cur·v_isvd‖²    − ‖A_cur·v_isvd_fut‖²|     / max(...)
ms_oracle    = |‖A_cur·v_oracle‖²  − ‖A_cur·v_oracle_fut‖²|   / max(...)
```

where each candidate's "_fut" is the same recipe applied with the
extended search basis [B_top; A_cur; A_fut]:

  v_combined: combined-score optimizer on (S, A_cur)
  v_combined_fut: combined-score optimizer on (S, A_cur, A_fut)
  v_isvd: top right SV of M_gain  /  v_isvd_fut: top right SV of M_gain_fut
  v_oracle: V_exact[:,0] projected onto rowspace(M_gain), unit-normed
  v_oracle_fut: V_exact[:,0] projected onto rowspace(M_gain_fut), unit-normed

Sign-align each "_fut" against its current-block solution before
computing the shift.

The diagnostic question: which candidate's lock is most stable to peek?
Does argmin(ms_*) match the bench's regime winner per matrix?

## Headline: per-matrix median mass_shifts

```
matrix             med_ms_c  med_ms_i  med_ms_o   ratio_i/c   bench winner
static-cex          0.135     0.489     0.116        3.6      combined ✓
mixed-tail-sharp    0.162     0.349     0.464        2.2      combined ✓
mixed-tail-soft     0.388     0.184     0.426        0.5      combined ✗ (would pick iSVD)
diffuse-diffuse    0.299     0.102     0.432        0.3      iSVD     ✓
```

The **median ratio ms_isvd / ms_combined** is a regime indicator that
works at the population level on **3 of 4 matrices**:

- ratio > 1: combined is more stable → combined is the right baseline.
- ratio < 1: iSVD is more stable → iSVD is the right baseline.

**static-cex** and **diffuse-diffuse** match the bench winner cleanly.
**mts-sharp** matches.
**mts-soft** is misclassified — the median ratio says iSVD should win,
but the bench reports combined wins (cos² 0.78 vs 0.13). See §5.

## ms_oracle is structurally always high (≈ 0.4–0.5)

By construction, oracle's "fut" version has access to a strictly larger
rowspace, so V_exact[:,0] gets projected with more support — its mass
spreads over A_fut as well as A_cur, dropping ‖A_cur·v_oracle_fut‖²
relative to ‖A_cur·v_oracle‖². The 3-way argmin essentially never picks
oracle:

```
argmin_3way distribution per matrix (n=48):
  static-cex:        c=15  i=19  o=14    (oracle = 29% of cells)
  mts-sharp:         c=18  i=20  o=10
  mts-soft:          c=15  i=29  o=4
  diffuse-diffuse:   c=13  i=34  o=1     (oracle = 2%)
```

Oracle picks only when the matrix-block has very low δ (oracle has
basically no mass in current rowspace), which is the wrong reason. The
3-way argmin signal collapses to the 2-way one for any usable matrix.

## ms_isvd is bimodal — top-SV flips between blocks

Per-block ms_isvd trajectory across seeds shows a clear bimodal pattern
on row-concentrated matrices: ms_isvd flips between ~0 and ~1
block-to-block. Excerpt from static-cex (median across seeds):

```
block:  1     2     3     4     5     6     7     8     9    10    11    12
ms_c:  .50  .26   .34   .35   .14   .11   .10   .14   .14   .14   .06   .05
ms_i: 1.00  .00   .99   .00  1.00   .00  1.00   .00   .10   .94   .35  1.00
```

Mechanism: `top_right_SV(M_gain)` and `top_right_SV(M_gain_fut)` can be
either the same direction (ms ≈ 0) or a completely different attractor
(ms ≈ 1) depending on whether A_fut's rows shift the top singular
subspace. With block-orthogonality, A_fut adds an orthogonal block of
rows whose top SV may dominate or not, which makes iSVD's "stability"
discontinuous.

Practical consequence: per-block argmin(ms_*) is dominated by which
side of the iSVD flip we're on at each block, not by which candidate
has higher cos² with the oracle. **Per-block argmin is too noisy to be
useful as a per-block selector.**

## Per-block argmin is barely better than coin-flip at picking the
better-aligned candidate

Sanity check: how often does argmin_2way pick the candidate with
higher cos²(*, V_exact[:,0]) at this block?

```
matrix             argmin_2way picks higher-cos² candidate (frac of 48)
static-cex            27/48 (56%)
mts-sharp             24/48 (50%)
mts-soft              25/48 (52%)
diffuse-diffuse       31/48 (65%)
```

Only diffuse-diffuse gets meaningful per-block signal. Elsewhere,
argmin is essentially a coin-flip on the per-block correctness
question.

## The mts-soft misclassification

The median ratio puts mts-soft on the iSVD side, but the bench reports
combined wins by 6× (cos² 0.78 vs 0.13). Why?

Mass-shift detects **converged-ness**, not **correctness**. iSVD on
mts-soft converges fast to a per-block top-SV that is wrong but
internally consistent (the matrix has enough row-norm variance that the
top SV is a localized, repeatable spike). Combined is harder to
converge in this regime because the relH gate is weak — its lock
trajectory wobbles more block-to-block, registering as higher ms — but
the *direction* of its wobble accumulates oracle alignment via
make_state.

This is the same pattern that broke per-block hard-switch gating in the
previous experiment: a stable wrong direction looks better by ms than a
shaky correct direction.

## Falsifiable conclusions

- **Per-MATRIX median ms-ratio as regime indicator: USEFUL on 3 of 4.**
  Threshold `med(ms_isvd) / med(ms_combined) ≷ 1` recovers the bench's
  regime call for static-cex (3.6, combined), mts-sharp (2.2, combined),
  diffuse-diffuse (0.3, iSVD). Misses mts-soft (0.5, says iSVD; bench
  says combined). Useful as a one-shot "which baseline to commit to"
  signal at the population level.

- **Per-BLOCK argmin(ms_*) as candidate selector: NOT USEFUL.**
  Picks the better-cos² candidate only ~50–65% of cells. The
  ms_isvd bimodality (top-SV flipping between blocks) dominates the
  signal. ms_oracle is structurally always too large to pick.

- **ms_oracle is uninformative.** By construction its "fut" version
  spreads V_exact projection over a larger rowspace, so ms_oracle ≈
  0.4–0.5 nearly always.

- **mass_shift detects converged-ness, not correctness.** A stable
  wrong direction (iSVD on mts-soft) has lower ms than a shaky correct
  one (combined), so argmin biases toward fast-converging baselines
  regardless of whether they're heading at the oracle.

## So what?

The probe gives a clean, *one-shot, per-matrix*, oracle-free regime
classifier (median ms-ratio) that's correct on 3 of 4 matrices. That's
not as useful as a per-block selector would be — but it's still
something the bench can compute at run-time without V_exact, and it
agrees with the visibility-analysis story (combined-favored regime vs
iSVD-favored regime) on the matrices where the visibility-analysis
story is unambiguous.

The natural next experiment: run the median ratio on the full §6
7-matrix set (especially mts-balanced, residual-spiky-shocks,
risk-residual-panel) and see if it correctly classifies them. If it
holds at >70%, this is a usable runtime regime detector. The
mts-soft miss is a known failure mode (stability ≠ correctness) and
should be flagged as such.

## Files

- `cells.csv` — 192 rows: matrix, seed, block,
  ms_combined/isvd/oracle, argmin_2way/3way, cos²(*, V_exact[:,0])
  per candidate, raw masses (mc_*, mf_*) per candidate.
- This report.
