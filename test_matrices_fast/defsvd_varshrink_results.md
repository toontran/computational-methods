# `defsvd-varshrink` (DefSVD-VarShrink) — implementation & benchmark results

Authoritative spec: `~/pj/meeting_focused/relaxation3_c7_recalibration_spec.md`,
Part 2. Variance-driven gap **reshaping** (NOT budget-matched to FD — this is the
distinction from the failed `defsvd-recal-carry`, which matched FD's deflation
budget and saturated).

Date: 2026-05-21. n=128, win=32, rank=2, seed=0, preset=fast (benchmark defaults).

---

## 1. Implementation

### Where (non-destructive edits to `benchmark_defsvd.py`)

1. New branch `elif mode == "DefSVD-VarShrink":` inserted before the
   `DefSVD-OrthDef` branch in `run_streaming()`. Nothing removed.
2. Method tuple `("defsvd-varshrink", "DefSVD-VarShrink", False)` appended to the
   `methods` list (after `defsvd-recal-carry`). Full pre-existing list intact.

### Per-window algorithm (spec Part 2 steps 1–4)

`M_gain = [prev_sketch; A_block]` with `prev_sketch = S_r @ V_r.T` (carry `B`),
exactly as the harness already builds at lines 96–97.

1. `V_hat` = top-r right singular vectors of `M_gain` (the **iSVD basis** —
   recalibration never changes which directions are retained); honest magnitudes
   `s_hat` via `projected_subspace_svd(M_gain, V_hat)`.
2. **Split variance (C7a)** on retained dirs `w_j = V_hat[:,j]`: `M=16`
   subsamples of `A_block` rows at `p=0.5`, drawn from a **local**
   `np.random.RandomState(1000003*(start0+1)+7)` (deterministic per window;
   the global seed the benchmark sets is never touched). For split `m`:
   `s_j^(m) = ||B w_j||^2 + ||A_{S_m} w_j||^2` (B-term from the carry; `=0` first
   block), relative `r_j^(m) = s_j^(m)/sum_k`. Then `var_j = Var_m(r_j^(m))`.
   No extra SVD — just norms.
3. **Shrinkage-to-mean (C7c):** `rho_j = min(1, lambda*var_j)`,
   `r_tilde_j = 1/r + (r_hat_j - 1/r)(1 - rho_j)`, renormalize to sum 1,
   `s_tilde_j^2 = r_tilde_j * sum_k s_hat_k^2` (preserve retained energy as
   sum-of-squares per the spec/Part-1 note), clip `>= 0`, `s_tilde = sqrt`.
4. Store `V_hat` with magnitudes `s_tilde` via `_new_state(...)` — same basis as
   iSVD, only magnitudes reshaped.
5. First block (`V_r/S_r is None`): behaves like iSVD honest magnitudes
   (`s_tilde = s_hat`, no shrink — no variance reference yet).

`lambda` is read per call from `VARSHRINK_LAMBDA` env var (default 1.0).

### CRITICAL design point (the shrink CENTER)

To honor the **hard, repeated invariant** "lambda=0 reduces to iSVD honest
magnitude EXACTLY" (spec C7c property (i); Part 2 note (b); task), the quantity
shrunk toward `1/r` is the **honest retained relative energy**
`r_hat_j = s_hat_j^2 / sum_k s_hat_k^2`, NOT the MC sample mean `r_bar_j`. The MC
splits supply `var_j` (which sets `rho`); `r_bar_j` is retained only as a
diagnostic in the state dict. At `lambda=0`: `rho=0` ⇒ `r_tilde = r_hat`
(already sums to 1, renorm is a no-op) ⇒ `s_tilde = s_hat`. This is the only
reading of step 3 consistent with all three statements of the invariant. (An
initial implementation that used `r_bar` as the center failed the sanity check
by O(0.1) in alignment and O(0.4) in sigma1 rel-err; switching the center to
`r_hat` fixed it to ~1e-14.)

### Commands

```
cd /home/ttran02/pj/computational-methods/test_matrices_fast
python run_varshrink_benchmark.py          # sweep + sanity + tables + timing
# or single point via the harness CLI:
VARSHRINK_LAMBDA=2.0 python benchmark_defsvd.py --matrices static-cex-noisy --skip-combined
```

`run_varshrink_benchmark.py` calls `run_streaming(..., l=...)` directly to hit
both `l=1` and `l=rank`, and loops `lambda in {0, 0.5, 1, 2, 5}` via the env var.
Raw output is saved to `varshrink_run_output.txt`.

---

## 2. Sanity: lambda=0 == iSVD — PASS (to machine precision)

Over all 6 matrices x {l=1, l=r}:

```
MAX |align diff|  = 5.529e-14
MAX |relerr diff| = 1.905e-15
```

Every individual cell agrees to <6e-14. lambda=0 reduces to iSVD exactly.

---

## 3. lambda-sweep tables

Alignment = `||V_r V_r^T V_exact[:,:l]||_F / sqrt(l)`. Higher is better.
relerr_sval = `|sigma1_est - sigma1| / sigma1`. Discriminators FIRST.

### l = 1

| matrix | method | align | relerr_sval |
|---|---|---|---|
| **static-cex-noisy** | isvd-ref | 0.226600 | 1.74e-03 |
| | fd-ref | 0.174672 | 7.62e-01 |
| | defsvd-carryonly | 0.174670 | 1.75e-03 |
| | varshrink lam=0 | 0.226600 | 1.74e-03 |
| | varshrink lam=0.5 | 0.228848 | 1.88e-03 |
| | varshrink lam=1 | 0.231554 | 2.03e-03 |
| | varshrink lam=2 | 0.239032 | 2.34e-03 |
| | **varshrink lam=5** | **0.325191** | 3.42e-03 |
| **static-cex-gauss** | isvd-ref | 0.181332 | 1.77e-03 |
| | fd-ref | 0.158808 | 7.60e-01 |
| | defsvd-carryonly | 0.158807 | 1.78e-03 |
| | varshrink lam=0 | 0.181332 | 1.77e-03 |
| | varshrink lam=2 | 0.183908 | 2.33e-03 |
| | **varshrink lam=5** | **0.197456** | 3.33e-03 |
| **static-cex-exptail** | isvd-ref | 0.061803 | 1.20e-03 |
| | fd-ref | 0.123290 | 8.92e-01 |
| | defsvd-carryonly | 0.123290 | 6.23e-03 |
| | varshrink lam=0 | 0.061803 | 1.20e-03 |
| | varshrink lam=5 | 0.062142 | 1.28e-03 |
| static-cex (hadamard) | isvd-ref | 0.149301 | 1.42e-03 |
| | fd-ref | 0.265409 | 8.74e-01 |
| | defsvd-carryonly | 0.265410 | 3.31e-02 |
| | varshrink lam=5 | 0.153514 | 3.07e-03 |
| mixed-tail-sharp | isvd-ref | 0.074122 | 1.22e-03 |
| | fd-ref | 0.468311 | 8.66e-01 |
| | defsvd-carryonly | 0.467634 | 4.12e-02 |
| | varshrink lam=5 | 0.074122 | 1.22e-03 |
| crowded-strategy | isvd-ref | 0.999791 | 9.08e-04 |
| | fd-ref | 0.999819 | 2.67e-01 |
| | defsvd-carryonly | 0.999647 | 3.05e-03 |
| | varshrink lam=5 | 0.999778 | 1.20e-03 |

### l = r (=2)

| matrix | method | align | relerr_sval |
|---|---|---|---|
| **static-cex-noisy** | isvd-ref | 0.173297 | 1.74e-03 |
| | fd-ref | 0.164629 | 7.62e-01 |
| | defsvd-carryonly | 0.164627 | 1.75e-03 |
| | varshrink lam=0 | 0.173297 | 1.74e-03 |
| | varshrink lam=2 | 0.181452 | 2.34e-03 |
| | **varshrink lam=5** | **0.239167** | 3.42e-03 |
| **static-cex-gauss** | isvd-ref | 0.234597 | 1.77e-03 |
| | fd-ref | 0.120265 | 7.60e-01 |
| | defsvd-carryonly | 0.120264 | 1.78e-03 |
| | varshrink lam=0 | 0.234597 | 1.77e-03 |
| | varshrink lam=2 | **0.234768** | 2.33e-03 |
| | varshrink lam=5 | 0.234324 | 3.33e-03 |
| **static-cex-exptail** | isvd-ref | **0.092170** | 1.20e-03 |
| | fd-ref | 0.129287 | 8.92e-01 |
| | defsvd-carryonly | 0.129287 | 6.23e-03 |
| | varshrink lam=0 | 0.092170 | 1.20e-03 |
| | varshrink lam=2 | 0.092011 | 1.23e-03 |
| | varshrink lam=5 | 0.091711 | 1.28e-03 |
| static-cex (hadamard) | isvd-ref | 0.148235 | 1.42e-03 |
| | fd-ref | 0.253963 | 8.74e-01 |
| | defsvd-carryonly | 0.253962 | 3.31e-02 |
| | varshrink lam=5 | 0.148794 | 3.07e-03 |
| mixed-tail-sharp | isvd-ref | 0.184273 | 1.22e-03 |
| | fd-ref | 0.490992 | 8.66e-01 |
| | defsvd-carryonly | 0.490424 | 4.12e-02 |
| | varshrink lam=5 | 0.184273 | 1.22e-03 |
| crowded-strategy | isvd-ref | 0.998843 | 9.08e-04 |
| | fd-ref | 0.994485 | 2.67e-01 |
| | defsvd-carryonly | 0.976047 | 3.05e-03 |
| | varshrink lam=5 | 0.998838 | 1.20e-03 |

(Full per-lambda rows for every matrix are in `varshrink_run_output.txt`.)

---

## 4. Runtime (mean over matrices, l=r, 5 reps)

| method | per-stream time |
|---|---|
| isvd-ref | 1.54 ms |
| fd-ref | 1.58 ms |
| defsvd-carryonly | 1.89 ms |
| **defsvd-varshrink (lam=1, M=16)** | **3.90 ms** |

The M-split cost roughly **2.5x iSVD / 2x carryonly**. The overhead is the 16
mask-draws + `||A_{S_m} w_j||^2` evals per window (NO extra SVD — the score
functional is C4-cheap), plus the one `projected_subspace_svd`. At this matrix
size that fixed cost dominates; the relative multiplier would shrink for larger A.

---

## 5. Verdict on the discriminator

### Does best-lambda varshrink beat iSVD on noisy / gauss / exptail?

| discriminator | l=1: isvd → best varshrink | l=r: isvd → best varshrink |
|---|---|---|
| static-cex-noisy | 0.2266 → **0.3252** (lam=5) WIN | 0.1733 → **0.2392** (lam=5) WIN |
| static-cex-gauss | 0.1813 → **0.1975** (lam=5) WIN | 0.2346 → 0.2348 (lam=2) ESSENTIALLY TIED, non-monotone |
| static-cex-exptail | 0.0618 → 0.0621 (lam=5) marginal WIN | 0.0922 → 0.0922 (lam=0) **TIE; loses for lam>0** |

**At l=1:** varshrink beats iSVD on all 3 discriminators (noisy clearly, gauss
clearly, exptail marginally). It is the FIRST method in this family to beat iSVD
on the discriminators rather than collapse to carryonly's hadamard-disease.

**At l=r:** wins clearly on **noisy** (+0.066), is **essentially tied on gauss**
(+0.0002 at lam=2 — noise-level, and NON-MONOTONE in lambda: best is lam=2 not
lam=5, the gain peaks then reverses, so gap-compression is not universally
helpful even on a discriminator), and on **exptail it does NOT win — best is
lam=0 (= iSVD), and any lam>0 slightly *degrades* it** (0.0922 → 0.0917 at
lam=5). So at l=r it beats iSVD on 1 clear of 3, ties gauss, and does not beat
iSVD on exptail.

**Fragility note on the noisy l=r win.** Per-window alignment trajectory
(noisy, l=2) is `[0.2525, 0.1741, 0.211, 0.1733]` at lam=0 vs
`[0.2525, 0.1741, 0.2088, 0.2392]` at lam=5 — identical for the first three
windows, diverging only at the FINAL window. The +0.066 headline gain therefore
comes from gap-compression on a single (last) window, not a sustained
per-window improvement. Read it as "reshaping helped the terminal state here",
not "consistently better every window".

**Honest call on the "≥2-of-3-loss" bar:** at l=r it does NOT lose on 2 of 3 —
it ties-or-loses on exactly 1 (exptail) and improves on the other 2. So it
clears the failure bar the spec set for `defsvd-recal-carry`. But the gauss l=r
win is within noise-level margin, and the exptail l=r result says the gap
reshaping is *neutral-to-mildly-harmful* exactly where iSVD's honest gap is
already correct. **Do NOT read this as a free win on exptail.**

### Does large lambda drive toward equal-weight / conservative behavior?

**Yes, empirically confirmed.** Per-window diagnostics (noisy, lam=5) show both
retained directions get the same `rho` (var_j is identical across the r=2
complementary pair, since `r_1^(m) + r_2^(m) = 1` forces equal variances), so
`r_tilde -> [0.501, 0.499]` — the equal-weight subspace. The carried sigma's
flatten toward uniform as lambda grows. This is the C7c shrinkage-to-mean
working as designed. **Caveat: this clean equal-weight collapse is partly an
r=2 artifact** — the identical-var_j mechanism does not hold for r>2, where
directions can shrink at different rates; the r>2 regime was not benchmarked.

**Important nuance (matches the Part-1 C7c CAVEAT):** "conservative" here means
**the alignment IMPROVES, while sigma1 rel-err GROWS**. As lambda grows the
relerr_sval climbs monotonically (noisy l=r: 1.7e-3 → 3.4e-3; gauss: 1.8e-3 →
3.3e-3) — flattening the gap costs sigma1 accuracy — yet the *subspace*
alignment improves because the dominant momentary direction stops truncating the
persistent weak one. So the per-pair gap is being COMPRESSED (weak monotonicity
toward uniform), NOT every singular value shrinking in a way that helps sigma1.
The relerr stays tiny throughout (max 3.4e-3 vs FD's 0.76), so the cost is cheap.

### Is the M-split cost justified?

**On the discriminators at l=1 and on noisy at both l: yes** — this is the only
member of the deflation family that genuinely improves subspace alignment over
iSVD on the noisy/gauss signal bases (where carryonly/recal-carry collapse to
FD-spirit and lose). The improvement is monotone and tunable in lambda, and the
mechanism (gap compression under split-variance uncertainty) is exactly the
intended antidote to the Appendix-A u-witness failure.

**At l=r overall: borderline.** The clear win is noisy; gauss is marginal;
exptail is neutral-to-slightly-negative. If the use case is l=r recovery on
exptail-type spectra, the 2.5x cost buys nothing. If it is l=1 / noisy-gauss
subspace tracking, the cost is justified.

### Bottom line vs `defsvd-recal-carry`

`defsvd-recal-carry` FAILED because its FD-matched deflation budget saturated and
collapsed it to `carryonly` on every discriminator (and was actively harmful on
options-vol-surface). `defsvd-varshrink` does NOT saturate — it reshapes the
carried gap by variance with energy preserved, never zeroing the budget. As a
result it is the first method here to BEAT iSVD on the discriminators at l=1 (all
3) and at l=r (noisy clearly; gauss only essentially tied and non-monotone). The honest
caveats are: exptail at l=r (no gain; mild loss for lam>0), gauss l=r being a
noise-level tie that peaks at lam=2 then reverses, and the noisy l=r win being a
single-window (terminal) effect rather than a sustained one. Do NOT soften
exptail: at l=r, lambda>0 makes exptail slightly worse, and the best
exptail-l=r choice is lambda=0 (plain iSVD).
