## Comprehension Loop 2

No source edits were made.

- **Major** — $B_{\mathrm{union}}$ shifts type. It is defined as a subspace/range, `range((B; A_w; A_fut))` (DOC B, lines 280--284), but the optimizer later writes `Z=B_union W` (lines 419--420), which treats it as an orthonormal basis matrix. Name the basis separately, e.g. `Q_union`, and keep `B_union` for the subspace.

- **Major** — The V2 cross term needs a row-pairing convention. `\langle A_cur Z, A_fut Z\rangle_F` (lines 398--404) is only directly meaningful if the two half-windows have the same row count and rows are paired in order. Since DOC A only stacks row windows, a reader needs one sentence saying whether this is a paired-response diagnostic, a same-size-half-window convention, or just an implementation artifact.

- **Moderate** — “The current best is S6” (lines 198--200) becomes confusing once §3 says V1--V3 fail and “cheaper feature additions have been ruled out” (lines 437--453). Clarify “current best” means the best previous operational replacement before the sufficiency sweep, not a surviving recommendation.

- **Moderate** — The M3 block-1 alignment labels conflict. The main text says combined has `<0.05` against exact oracle columns and `0.096` against `oracle_proj` (lines 137--143), but Appendix A.3 labels the column `cos^2(hat v_1, v_1^*)` while reporting `0.096` (lines 579--587). Rename the column or split exact-oracle vs projected-oracle alignments.
