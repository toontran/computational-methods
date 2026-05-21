## Comprehension Loop 3

- **Major**: The definition of `B_union` still uses `range((B; A_w; A_fut))` for a vertically stacked row matrix (current_report lines 283-286). Since the search variables live in `R^d`, a reader expects the relevant object to be the row span/right search subspace, not the column range of the stacked matrix. The following sentence says this is "the search subspace" (lines 287-290), but the displayed formula points to the wrong ambient space.

- **Major**: The V1 headline says the future-direction term "lowers `S_1(Z_oracle)` relative to plain HM" (line 453), and Appendix B repeats "V1 and V2 lower the score at `Z_oracle`" (lines 780-781). The table does not support that uniformly: V1 oracle scores rise for `diffuse-diffuse` (`0.0066 -> 0.0099`) and `mixed-tail-soft` (`0.0215 -> 0.0223`), and only fall for `residual-spiky-shocks` (lines 760-765). This makes the V1 reading hard to trust.

- **Moderate**: Appendix A.4 says "The `S_6` argmax is the oracle (gap <= 0.007)" (line 671), but the table immediately above reports positive gaps for all three matrices (lines 663-667), and earlier text frames these as regression matrices where `S_6` fails (lines 241-246). If "effectively tied" is intended, say that; "is the oracle" overstates the claim.

- **Minor**: The phrase "oracle-warm `S_6`" in M3 (lines 141-143) appears before the later explanation that the block-1 isolation actually uses `HM_2(u_cur,u_fut)` with `u_sk` dropped (lines 620-625). A cold reader briefly expects full `S_6` with sketch slot handling at `B=0`; name it consistently at first mention.
