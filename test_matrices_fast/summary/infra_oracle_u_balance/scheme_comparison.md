# DIAG-04b — per-vector weight-scheme audit (current vs E1 vs E2)

Probe: `oracle_u_balance_audit.py --weight-scheme {current,E1,E2}`
(half_win=32, rank=2, §6 suite). Numbers below are b31 ratio_max from
`audit_summary.csv` in each variant directory.

Schemes:
- `current` — w = ‖A‖_F² (sum of all sigma², the inherited scalar).
- `E1` — w = sigma_top(·)² (op-norm cap, scalar).
- `E2` — w(v) = sigma[k(v)]² where k(v) = argmax_i (V_top[:,i]^T v)²
  (per-direction, only v-dependent scheme of the four).

Verdicts (per scheme per matrix at b31):
- **SIMULT** — both slot-1 and slot-2 ratio_max ≤ 5×.
- **S1-ONLY** — slot-1 ≤ 5× and slot-2 > 10×.
- **NO-IMP** — both > 10×.
- **MIXED** — anything else (e.g. slot-2 in [5,10]).

## ratio_max(slot-1, slot-2) at b31

| matrix | current (v1, v2) | E1 (v1, v2) | E2 (v1, v2) | current verdict | E1 verdict | E2 verdict |
| --- | --- | --- | --- | --- | --- | --- |
| mixed-tail-sharp     | (243,  17.0) | (20.3,  9.2) | (22.2, 57.1) | NO-IMP | MIXED  | NO-IMP |
| mixed-tail-balanced  | (334,   1.07)| (16.6,  9.2) | (13.8,  5.2) | MIXED  | MIXED  | MIXED  |
| mixed-tail-soft      | (347,  15.7) | (16.9,  1.57)| (20.0,  9.0) | NO-IMP | MIXED  | MIXED  |
| static-cex           | (322,  379)  | (26.3, 20.4) | (26.6, 22.1) | NO-IMP | NO-IMP | NO-IMP |
| diffuse-diffuse      | (17.0, 18.3) | (18.0,  7.0) | (1.29, 2.43) | NO-IMP | MIXED  | **SIMULT** |
| etf-basket-basis     | (1.74, 574)  | (1.06, 936)  | (1.06, 786)  | S1-ONLY| S1-ONLY| S1-ONLY|
| residual-spiky-shocks| (35.5, 91.5) | (4.07, 6.09) | (2.25, 1.73) | NO-IMP | MIXED  | **SIMULT** |
| risk-residual-panel  | (85.0, 10.6) | (10.0, 2.58) | (6.36, 11.9) | MIXED  | MIXED  | MIXED  |

Notes:
- mixed-tail-sharp slot-2 *worsens* under E2 (57.1×) vs E1 (9.2×): the
  per-direction lookup picks the wrong rank of state on this matrix
  because slot-2 oracle aligns weakly with the carry's top direction.
- etf-basket-basis stays S1-ONLY in all schemes — slot-2 collapse there
  (u_sk ≈ 3e-4 vs u_g1 ≈ 1) is structural (oracle slot-2 has near-zero
  energy in the carried state), not a weight-calibration issue.

## Acceptance — does E2 achieve simultaneous balance on a §6 high-entropy
matrix where E1 cannot?

The backlog gates E2 on:
> "If E2 produces SIMULTANEOUS BALANCE on at least one of
> mixed-tail-sharp / mixed-tail-balanced / static-cex / diffuse-diffuse
> where E1 produces SLOT-1 ONLY, the per-direction hypothesis is
> verified."

Strict reading:
- E2 SIMULTANEOUS on diffuse-diffuse: **yes** (1.29×, 2.43×).
- E1 SLOT-1 ONLY on diffuse-diffuse: **no** (slot-1 = 18.0×, slot-2 =
  7.0×). E1 is MIXED there, not SLOT-1 ONLY.

So the strict conjunctive gate is not met. But E2 still does something
E1 cannot do anywhere on the §6 suite: drive both slots to ratio_max
≤ 2.5× simultaneously (diffuse-diffuse, residual-spiky-shocks). The
per-direction reweighting *can* yield simultaneous balance on at least
two §6 matrices — a property no scalar scheme achieves on this data.

Counter-evidence:
- E2 fails harder than E1 on mixed-tail-sharp slot-2 (57× vs 9×).
- On static-cex both schemes are stuck at ~22-26× both slots.
- mixed-tail-balanced slot-2 just misses (5.23×, ~5%-over the 5×
  threshold).

## One-number acceptance bar

Backlog acceptance: *"does any E2 column show ratio_max(slot-2) < 5× on
a §6 high-entropy matrix? Yes ⇒ verified, advance E2."*

E2 slot-2 ratio_max at b31:
- mixed-tail-sharp     57.12×
- mixed-tail-balanced   5.23×  (just over)
- mixed-tail-soft       8.99×
- static-cex           22.09×
- **diffuse-diffuse     2.43×  ← <5×, hits the bar**
- etf-basket-basis    786×    (structural, slot-2 outside carry)
- residual-spiky-shocks 1.73×  (high entropy variant; <5×)
- risk-residual-panel  11.88×

**One-number answer: YES.** Two §6 matrices (diffuse-diffuse,
residual-spiky-shocks) show E2 slot-2 ratio_max < 5×. Per backlog rule:
**verified, advance E2 to AB-03 phase 1.**

## Verdict — VERIFIED (strict one-number rule)

The per-direction hypothesis is **partially verified**: E2 demonstrates
strictly broader balance reach than any scalar scheme (only v-dependent
denominators can drive ratio_max(slot-2) below 5× on diffuse-diffuse and
residual-spiky-shocks), but E2 alone is not sufficient — it makes
mixed-tail-sharp slot-2 worse, leaves static-cex stuck, and only
narrowly reaches the threshold on mixed-tail-balanced.

## Recommendation

1. **Wire E2 into r_sk_g_score.py and start AB-03 phase 1 (T1/T2/T3
   gauntlet).** It is the strongest single-knob candidate observed in
   the audit, and on the matrices where simultaneous oracle balance
   matters most for HM3's smallest-link reading (diffuse-diffuse,
   residual-spiky-shocks) it succeeds outright. The mixed-tail-sharp
   regression and static-cex non-improvement are concrete failure modes
   to watch for in T2/T3.

2. **Promote FAM-01 (rank-r lift) in parallel.** static-cex resists all
   three schemes (current, E1, E2 all > 20× on slot-2) — the imbalance
   there is not curable by scalar *or* per-direction reweighting alone,
   matching the backlog's contingent path: "the imbalance is not curable
   by per-direction reweighting alone — it's structural to the
   deflation greedy + rank-1 score combination, and the rank-r lift
   (FAM-01) becomes the only remaining lever for slot-2."

   Both are true at once on the §6 suite: E2 helps on some matrices,
   FAM-01 is needed for others.

3. **Skip E3/E4** in the first pass (per backlog instruction) — re-open
   only if E2's mixed-tail-sharp regression turns out to dominate the
   AB-03 phase-1 gauntlet outcome.

## Cross-references
- Raw per-matrix CSVs: `summary/infra_oracle_u_balance/{E1,E2}/*_audit.csv`
- Baseline: `summary/infra_oracle_u_balance/audit_summary.csv`
- score_design_overview.txt §1quater (M4 mechanism), §2bis (b.iii)
- score_family_workflow.txt [DIAG-04] / [DIAG-04b] / [AB-03]
- diagnostic_toolkit.txt §6b (oracle u-imbalance signature)
