## Comprehension Loop 4

No source edits were made.

- **Moderate** — Editorial severity tags appear inside DOC B as if part of the report: "`[Major] Unlike ...`" near `S_6` (lines 181--183), "`[Major] In this report...`" (lines 273--275), and "`[Major] Let Q_union...`" (lines 288--289). A cold reader cannot tell whether these are intended labels or leftover review markup. Remove or convert them to prose.

- **Moderate** — The current/future window granularity is still slippery. Lines 21--23 say `A_fut` is "the next half-window after the current block `A_w`; `A_w` itself is not split," but §2 defines `A_cur:=A_w` (lines 369--371) and V2 says `A_cur` and `A_fut` are "same-size-half-window" matrices (lines 411--414). State whether `A_w` is itself a half-window in the sweep, or whether the sweep redefines block size.

- **Moderate** — The recap substitution is hard to trust on first read. Lines 29--33 replace `row(N_w)` by `k` in the base and call it the "same substitution" as the A2 exponent replacement; lines 46--50 say this "does not change the empirical score." Since DOC A keeps `row(N_w)` in the base, name this as a diagnostic normalisation and state which multiplier Appendix A used.

- **Minor** — V2's cross term needs sign/zero conventions. The formula is a raw normalised Frobenius inner product (lines 405--410), but the conclusion says `S_2(Z_oracle)≈0` and "near the bottom" (line 461). Say whether negative correlations are allowed, clipped, or absent, and how zero denominators are handled.
