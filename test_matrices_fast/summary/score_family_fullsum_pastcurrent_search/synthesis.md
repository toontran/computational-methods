# Full-sum × stacked-entropy variants (S7, S8) — past+current search domain

Date: 2026-05-01

## Variants

Two new variants in `r_sk_g_score.py:r_sk_g_value_grad`. Streaming wiring in
`half_window_sliding_hmean_experiment.py` overrides the search basis to
`rowspace([A_sketch; A_cur])` only — A_fut is excluded from the search domain
but still consumed by the score. Slot-2 picker (deflated against
V_default[:,0] like S6).

- **S7** = `(raw_sk + raw_g1 + raw_g2) · relH1_full`,
  where `relH1_full` is normalized Shannon entropy of the energy
  distribution of `[A_sketch; A_cur; A_fut] v` stacked over all rows.
  Generalizes the original `combined` score from `[B; A_w]` to
  `[B; A_cur; A_fut]`. Block-1 fall-through stacks `[A_cur; A_fut]` only.
- **S8** = `raw_sk + raw_g1 + raw_g2`, same domain as S7, no entropy. The
  argmax in the restricted basis is the leading right SV of the stacked
  matrix that is reachable from past+current rows.

FD gradient check (b1, b2, b12, b31 on mixed-tail-sharp): rel_err
1.8e-10 to 4.5e-10 for both, consistent with S1–S6.

## 2-matrix sanity bench (half_win=32, seed=0)

| matrix             | method   | cos0²  | cos1²  | Δcos1² vs S6 |
|---|---|---|---|---|
| etf-basket-basis   | isvd     | 1.000  | 0.205  | — |
| etf-basket-basis   | combined | 1.000  | 0.017  | — |
| etf-basket-basis   | online   | 1.000  | 0.969  | — |
| etf-basket-basis   | S6 ref   | 1.000  | 0.807  | 0 |
| etf-basket-basis   | **S7**   | 1.000  | **0.917** | **+0.110** |
| etf-basket-basis   | **S8**   | 1.000  | 0.410  | -0.397 |
| mixed-tail-sharp   | isvd     | 0.078  | 0.013  | — |
| mixed-tail-sharp   | combined | 0.894  | 0.019  | — |
| mixed-tail-sharp   | online   | 0.863  | 0.050  | — |
| mixed-tail-sharp   | S6 ref   | 0.861  | 0.111  | 0 |
| mixed-tail-sharp   | **S7**   | 0.778  | 0.028  | -0.083 |
| mixed-tail-sharp   | **S8**   | 0.086  | 0.023  | -0.088 |

S6 reference numbers from `summary/bench_matrix_sweep_value_only_online/`
(2026-04-28); the new runs use the same harness. Only the policy column
above was rerun; isvd/combined/online numbers come from this run and match
the §6 table within rounding.

## Reading

- **etf-basket-basis** (Q0 live operational gap): S7 lifts cos1² by +0.110
  over S6 and narrows the gap to value-only online from -0.162 to -0.052.
  This is the §6 priority target named in the overview.
- **mixed-tail-sharp** (high-entropy tail-conspiracy regime): S7 regresses
  on both slots vs S6. S8 collapses to ~iSVD (slot-1 lost) — without the
  entropy bias the raw-sum argmax is dominated by tail rows in A_cur, so
  past+current restriction alone is not sufficient. Entropy is doing real
  work but the score itself loses to S6 on this matrix.
- **S8 vs S7**: entropy is load-bearing in the tail-mixed regime
  (S7 0.778 vs S8 0.086 cos0² on mts) and helpful but not transformative
  on the structured-spectrum regime (S7 0.917 vs S8 0.410 cos1² on etf).

## Open questions

1. Does S7's etf gain replicate across seeds (single-seed sanity here)?
2. How does S7 do on the rest of the §6 suite — particularly
   diffuse-diffuse and the FAIL-subspace matrices (static-cex)?
3. Does the cos0² regression on mixed-tail-sharp become a kill under
   the §6 high-entropy no-regression criterion?

Full §6 sweep (7 matrices × {S7, S8}, seed=0, half_win=32) is the next
gate before any acceptance claim.

## Files

- `summary/score_family_fullsum_pastcurrent_search/etf_S7.{json,csv,txt}`
- `summary/score_family_fullsum_pastcurrent_search/etf_S8.{json,csv,txt}`
- `summary/score_family_fullsum_pastcurrent_search/mts_S7.{json,csv,txt}`
- `summary/score_family_fullsum_pastcurrent_search/mts_S8.{json,csv,txt}`
- Code: `r_sk_g_score.py:r_sk_g_value_grad` variant branches "S7"/"S8";
  `half_window_sliding_hmean_experiment.py` lines ~640-660 for the
  past+current search-basis override.
