## Comprehension Loop 1

No source edits were made.

- **Major** — DOC B drops DOC A's reservoir `R` without saying whether it is intentionally removed, absorbed, or irrelevant. DOC A's entropy term uses `C_w=(R;A_w)`, but §1's `S_6` and §2's sweep use only `B`, `A_w/A_cur`, and `A_fut`. Add near `S_6`: "`R` is not used by the replacement scores; row-sample evidence is being discarded/replaced by current-future split evidence."

- **Major** — The allowed evidence set is unclear when `A^\top A` appears. "The singular goal" allows "`A^\top A`-type quantities"; H2 treats actual `A^\top A` as non-window-local/oracle-like; the implication proposes "window-local approximations." Define deployable window-local quantities, diagnostic global quantities, and approximations requiring validation.

- **Moderate** — `A_w`, `A_{\mathrm{cur}}`, and `A_{\mathrm{fut}}` are hard to align with DOC A. §2 says `A_{\mathrm{cur}}:=A_w`, while §1 allows a future half-window. Clarify whether DOC A's `A_w` is split into current/future halves, or whether `A_fut` is an additional next window.

- **Moderate** — `Z_{\mathrm{oracle}}` is used before its construction is specified. In "singular goal," oracle recovery may mean projection of `V_r` into `B_union`, then the test immediately uses `Z_{\mathrm{oracle}}`. Add how the projected frame is formed, orthonormalised, and handled if rank-deficient.

- **Minor** — "unique maximiser" conflicts with later right-rotation-invariant frame scores. Say "unique subspace maximiser" for `\|A_XZ\|_F^2` scores, reserving oriented-frame uniqueness for scores that break right rotations.
