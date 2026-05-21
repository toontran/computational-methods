# FAM-01 B0 — T3 streaming bench synthesis

Date: 2026-04-29
Bench: half_win=32 (full window=64), seed=0
Driver: `bench_T3.py`
Per-matrix outputs: `bench/{matrix}_win64.{json,csv}`
Backlog item: [FAM-01] (score_family_workflow.txt)

## Verdict — **KILLED**

B0 (HM3-rank-r joint-frame Stiefel) fails the FAM-01 acceptance rule on the high-entropy §6 partial-pass set. 3 of 4 high-entropy matrices regress cos0² by ≥0.42 — far past the 0.02 regression bar.

## Acceptance scope

- Partial-pass set (per S-2 partition in [FAM-01]): residual-spiky-shocks (PASS-with-overshoot) + 4 high-entropy FAIL-slot-2-only matrices (etf-basket-basis, mixed-tail-sharp, mixed-tail-balanced, mixed-tail-soft).
- Acceptance: cos0² lift ≥0.05 over greedy S6 on ≥2 high-entropy matrices AND no high-entropy §6 cos0² regression > 0.02.
- Per §1ter framing, residual-spiky-shocks is conceded to iSVD; its numbers are informational only and do not gate ship/kill.

## Per-matrix results

| matrix | S6 cos0² | VO cos0² | B0 cos0² | Δcos0² (B0−S6) | S6 cos1² | B0 cos1² | Δcos1² (B0−S6) |
|---|---:|---:|---:|---:|---:|---:|---:|
| etf-basket-basis | 1.0000 | 1.0000 | 0.9954 | **−0.0046** ✓ | 0.6518 | 0.8205 | **+0.1688** ★ |
| mixed-tail-sharp | 0.7569 | 0.7598 | 0.0177 | **−0.7392** ✗ | 0.0125 | 0.0000 | −0.0125 |
| mixed-tail-balanced | 0.4435 | 0.6264 | 0.0243 | **−0.4192** ✗ | 0.0015 | 0.0000 | −0.0015 |
| mixed-tail-soft | 0.8595 | 0.7372 | 0.0320 | **−0.8275** ✗ | 0.0240 | 0.0036 | −0.0204 |
| residual-spiky-shocks (informational) | 0.4123 | 0.2500 | 0.2893 | −0.1230 | 0.2667 | 0.0083 | −0.2584 |

Acceptance check on high-entropy subset:
- cos0² lift ≥ 0.05 on ≥ 2 matrices: **0 of 4** (etf-basket-basis is within tolerance, not a lift; the other 3 collapse). FAIL.
- No high-entropy cos0² regression > 0.02: **3 of 4 regress by 0.42–0.83**. FAIL.

## Verdict — KILL B0

The collapse on mixed-tail-{sharp,balanced,soft} is a pattern, not an isolated optimizer failure. On those 3 matrices, B0's tail_mass is 0.85–0.99 and cos0² is 0.018–0.032 — the rank-r joint-frame optimizer is finding a degenerate frame that misses the dominant direction entirely.

## What the bench teaches

1. **Slot-1 collapse is the dominant failure mode** for HM3-rank-r joint optimization on mixed-tail matrices, not the slot-2 evidence-pinning that the S-2 reread anticipated. The two-part S-2 rule predicted slot-1 capture would carry through; T3 falsifies that on 3 of 4 high-entropy matrices.
2. **etf-basket-basis is the lone bright spot**: B0 lifts cos1² by +0.169 over S6 (0.65 → 0.82) while staying within tolerance on cos0². This contradicts the S-2 prediction that slot-2 evidence is structurally pinned, but it is one matrix out of four — not enough to motivate a B0 retry.
3. **Margin sensitivity** correlates with collapse. etf-basket-basis has S6 cos0² ≈ 1.0 (slot-1 fully captured); the 3 collapse matrices have S6 cos0² in [0.44, 0.86]. The rank-r joint search appears to fail specifically where slot-1 is near a margin, suggesting the failure is structural to the joint objective rather than to optimizer initialization.

## Failure-mode classification

- **Structural, not implementation**. The pattern across 3 matrices, the optimizer steps completing (31 of 31 on each), and the consistent tail_mass concentration argue against a bug in the joint Stiefel optimizer. The HM3-rank-r objective itself is mis-aligned with slot-1 capture in the mixed-tail regime.
- This adds support for the §1quinquies framing that current evidence (HM3 over u_X) is insufficient on the FAIL-slot-2-only set, not just the FAIL-subspace pair. The structural fix is evidence augmentation (FAM-07 / FAM-08 / FAM-09), not a different aggregator over the same evidence.

## Recommended follow-ups

1. **Close FAM-01 B0** as KILLED. Do not pursue B1 (carry-confidence multipliers) or B2 (trace-form rank-r) on the same evidence base — the structural argument generalizes.
2. **Promote evidence-augmentation track**. FAM-07 spec is frozen with T1 passing; the etf-basket-basis cos1² lift is a weak positive signal that frame-correlation evidence may help. Pull T2 next on FAM-07.
3. **DIAG-07** (M2 per-direction score-decomposition audit) on the 3 collapse matrices would confirm whether the slot-1 misranking is the same per-direction reweighting failure that DIAG-04b's E2 partially addressed but AB-03 phase 1 killed at the landscape level.
4. **Re-examine etf-basket-basis** as a probe: why does B0 lift cos1² there but collapse cos0² elsewhere? A targeted diagnostic on this single matrix could surface what evidence/structure is different.

## Cross-references

- score_family_workflow.txt [FAM-01]: status partial → KILLED for B0 (current-evidence aggregator).
- score_design_overview.txt §1quinquies: T3 falsifies the slot-1-survives prediction on the FAIL-slot-2-only set; structural redirect to evidence augmentation strengthened.
- summary/diag03_subsample_stability/synthesis.md: FAM-09 is now the highest-priority unblocked evidence-augmentation candidate (P2).
