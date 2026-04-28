# FAM-02: Row-concentration guard — known vs. unknown

Date: 2026-04-28
Pointers: `summary/overview/score_design_overview.txt` §2bis(d), §6, §7;
          `summary/overview/diagnostic_toolkit.txt` §6 (relH1 invariant), §7
          (residual-spiky-shocks regime).

This is a 1-pager. Detail lives in the overview docs; this file only carries the
delta that motivates the family.

## Settled (don't re-test)

- **S6 design closes the M1/M3 mechanisms (overview §1bis), but explicitly
  removes relH1.** §2(d) lists "no row-cheat protection" as the known cost; this
  is parked as F1 / Q3 in §7. Verified pathology: S6 picks low-relH1 v's on
  mixed-tail-sharp.
- **On the spiky-residual regime, iSVD/hybrid beat both online and S6**:
  residual-spiky-shocks cos1² = iSVD 0.637, online 0.246, S6 0.266;
  risk-residual-panel cos1² = iSVD 0.471, online 0.064 (overview §6).
- **relH1 is already implemented** in `hmean_evidence_score.entropy_relH1_value_grad`
  and is the per-block diagnostic invariant in `r_sk_g_score.analyze_block`
  (column `relH1`). It is normalized Shannon entropy of the `A_cur v` row-energy
  distribution, with analytic gradient.
- **S6 has clean analytic value+grad** (FD rel<1e-7 at float64) and is smooth on
  the sphere wherever all u's > 0. The product S6·relH1 is smooth on the same
  region (relH1 is smooth wherever the row-energy distribution is positive).

## Open (this family resolves)

- **Q3 (overview §7)**: does multiplying S6 by relH1 tighten the basin enough
  to recover the iSVD regime on residual-spiky-shocks without giving back the
  P4-driven losses on mixed-tail-* / static-cex / etf-basket-basis?
- Sub-questions for D1 / D2 (out of scope here, spec only):
  - Does leverage-weighted u_X address the same regime structurally rather
    than as a multiplicative factor?
  - Is a soft sigmoid gate on u_sk a smoother-still alternative to relH1
    multiplication?

## Toolkit gaps (NOT in scope here, but flagged)

- The relH1 invariant is per-block-only. No streaming-bench-time tracker of
  `relH1` of the chosen v over blocks (would help diagnose D0 wins / losses).
- No explicit row-cheat baseline at slot-2 beyond `hm_triplet_raw_best`. For
  this family it suffices.

## What "ship D0" looks like

T3 (sliding, win64): cos1² on residual-spiky-shocks ≥ iSVD's 0.637 AND no
matrix in the §6 table regresses by >0.05 cos1² vs S6 baseline (frozen in
`baseline/`). T2 oracle ranking unchanged vs S6 on the canonical 3-matrix
probe set (oracle still above `hm_triplet_raw`).
