# defsvd-varshrink — FULL 21-matrix suite (supersedes the 6-matrix verdict)

Run: `run_varshrink_full_benchmark.py` → `varshrink_full_run_output.txt`.
Suite = all 18 `BENCHMARK_MATRICES` + the 3 static-cex stress variants
(noisy/gauss/exptail). n=128, win=32, seed=0, ranks ∈ {2, 8}, l ∈ {1, rank},
λ ∈ {0, 0.5, 1, 2, 5}. λ=0 ≡ iSVD honest-magnitude (verified).

## Headline correction

**The earlier "first method that beats iSVD" verdict was wrong** — it came from
a 6-matrix slice with **per-matrix cherry-picked λ** and was concentrated on
`static-cex-noisy` at the tiny `rank=2`. On the full suite, **varshrink ≈ iSVD**:
when the variance signal is inert (almost everywhere) it reduces to
iSVD-with-honest-magnitudes.

## Cross-suite tallies (W/T/L, best-λ varshrink vs baseline)

| rank | l | vs iSVD | vs FD | vs carryonly |
|---|---|---|---|---|
| 2 | 1 | 6/15/0 (mean Δ +0.006) | 6/3/12 | 7/3/11 |
| 2 | 2 | 4/17/0 (mean Δ +0.003) | 6/1/14 | 9/1/11 |
| 8 | 1 | 3/18/0 (mean Δ **+0.000**) | 9/11/1 | 9/11/1 |
| 8 | 8 | 2/19/0 (mean Δ **+0.000**) | **21/0/0** (mean Δ +0.30) | **21/0/0** (mean Δ +0.31) |

Two facts kill the optimistic read:

1. **The "vs iSVD wins" are cherry-picked and vanish at honest fixed λ.** With
   λ fixed at 1 (no per-matrix selection):
   - rank=2, l=2: **3 win / 12 tie / 6 LOSE** vs iSVD.
   - rank=8, l=8: **2 win / 6 tie / 13 LOSE** vs iSVD (mean Δ ≈ 0).
   At a single honest λ, varshrink is marginally **worse** than iSVD across the
   suite, especially at the meaningful rank=8.

2. **"Beats FD/carryonly 21/0/0" is iSVD's merit, not the recalibration's.**
   varshrink tracks iSVD, and iSVD crushes the FD-family here (e.g.
   `risk-residual-panel` rank=8 l=8: iSVD 0.986 vs FD 0.135). So beating FD just
   restates that λ=0 ≡ iSVD is good; the recalibration adds nothing to it.

## The recalibration is inert on ~19 of 21 matrices

In the per-matrix tables, `vs@0.0 … vs@5.0` are **identical** for almost every
matrix (e.g. `diffuse-diffuse` 0.9857 flat; all `mixed-tail-*` flat). λ only
moves the result on `static-cex-noisy` (and slightly `gauss`) at rank=2 —
exactly the small-rank, structured-noise corner. Elsewhere `var_j ≈ 0` or the
shrinkage doesn't change which subspace survives truncation, so varshrink = iSVD.

## Where λ does anything (rank=2)

- `static-cex-noisy` l=2: iSVD 0.173 → 0.239 at λ=5 (real, but a single-corner,
  small-rank effect; at rank=8 iSVD already gets 0.986 and λ does nothing).
- `static-cex-gauss` l=2: 0.2346 → 0.2348 (noise).
- `static-cex-exptail`: λ>0 slightly **degrades** at every rank/l.

## Verdict (honest)

On the full benchmark, **defsvd-varshrink does not beat iSVD.** It is
statistically iSVD (variance signal inert almost everywhere); at a fixed honest
λ it is slightly worse on the subspace metric at rank=8. Its only robust
property — beating the FD-family — is inherited from λ=0 ≡ iSVD and is not a
contribution of the recalibration. The ~2.5× runtime cost (M=16 split evals) is
**not justified**: you would simply run iSVD.

The C7 *theory* (C5′, C7a/b/c) is sound and confirmed; the **practical payoff of
variance-driven gap-shrinking on this benchmark is essentially nil**, because the
local-window split-variance is inert wherever iSVD is already good — and where
iSVD is bad (the hadamard/exptail corners), gap-compression doesn't recover the
lost direction. This is consistent with the prior central finding that the
carry/window energy *ratio* is the lever; per-direction variance reshaping is not.
