## Comprehension Loop 5

No source edits were made.

- **Major** — The M1 mechanism reverses the entropy intuition from DOC A. DOC A says when `row(N_w)<n`, high-entropy pooled vectors are boosted (DOC A lines 218--219), and DOC B's recap has `phi_w=(n/k)^{h_2/(2 log k)}` with `phi_w>=1` (DOC B lines 40--63). But M1 says "anomalously high `\ell_4/\ell_2` tail ... inflates `\phi_w`" (lines 118--120). Since high `\ell_4/\ell_2` means lower `h_2`, this makes the failure mechanism hard to parse. Say whether the offender has high entropy or a high tail ratio.

- **Moderate** — The report mixes general-r notation with a rank-2 frame test. The singular-goal paragraph defines `V_r`, rank failure as `<r`, and `Z_oracle` from `P_union V_r` (lines 288--307), but the optimisation domain is `St(d,2)` throughout. State that this report fixes `r=2`, or rewrite the definitions as `V_2`/rank `<2`.

- **Moderate** — The optimizer says restarts are warm-started from "the oracle `V_2`" (line 455), while the feasible domain is restricted to `B_union` via `Z=Q_union W` (line 447). A cold reader needs to know whether the restart is `V_2`, `P_union V_2`, or the QR/polar projected `Z_oracle`.

- **Moderate** — M2's scale story is still easy to misread. Main text says the energy term reduces to `||Bv||^2` once `rho_past >= 5` (lines 123--127), but Appendix A.2 emphasizes `rho_B` remains small (`0.063--0.071`, lines 560--564). Add one sentence distinguishing full-history Frobenius scale from directional dominance of the retained sketch on the deflation complement.
