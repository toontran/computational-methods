## Writing-Style / Comprehension Advisor

No source edits were made.

### Fatal/Major flags

- **Major / Likely / Core** — DOC B changes the visible data model from DOC A without a clean bridge. DOC A has one current window `A_w` plus old sketch `B` and row reservoir `R`; DOC B immediately uses `A_cur` and peek `A_fut` (§1, §2), then says combined is "BLIND TO A_fut." A reader needs one sentence before §1bis: "In the bench, DOC A's current block `A_w` is split into two visible half-windows `A_cur` and `A_fut`; S1..S6 add this split-sample signal, whereas the original combined score only used `B` and `A_cur`." Also clarify whether `A_fut` is operational lookahead or diagnostic-only.

- **Major / Certain / Implementation** — `A_sketch`, `B`, `B_top`, and `state.s · state.V^T` are not mapped back to DOC A's carried sketch. DOC A is careful that `B = Sigma V_new^T` is the norm-preserving right-action form of the left-projected operator, not raw sampled rows. DOC B should define at first use: "`A_sketch` is DOC A's carried `B`; `state.s · state.V^T` stores its SVD/right-action form; `B_top` means [define or replace with `A_sketch`]."

### Other comprehension edits

- **Moderate / Likely / Core** — The "replacement" claim for S1..S6 blurs two changes: removing DOC A's entropy multiplier `phi_w` and adding future-half evidence. Split the sentence around §1bis: "S1..S6 replace the original within-window entropy correction with split-window balance scores; they also change the available evidence from `[B; A_w]` to `[A_sketch; A_cur; A_fut]` in the bench."

- **Moderate / Certain / Assumption** — The status of A2 is clear later (§1ter, "Theory status") but late for readers coming from DOC A, where A2 is load-bearing. Add a forward pointer near the first `phi` discussion: "A2 is treated here as the derivation scaffold for combined, not as an assumption retained by S6; §1ter/‘Theory status’ explains the downgrade."

- **Moderate / Likely / Presentation** — The entropy-bias language needs one sign bridge. DOC A defines `phi_w = ||C_w v||_4^{4c_w} ||C_w v||_2^{-4c_w}` with `c_w < 0`, so "phi rewards entropic spread" is true but not immediate. Add parenthetical: "Because `row(N_w) < n`, `c_w < 0`, so lower concentration / higher `h_2(C_wv)` increases `phi_w`."

- **Moderate / Likely / Method** — `value-only` is used while §1 says `A_fut` is a "peek half-window." That sounds non-causal unless the streaming block really has both halves available before committing. Add a definition: "value-only means no `V_exact`/oracle labels; it may still use the bench-visible split sample `A_fut`."

- **Minor / Certain / Presentation** — The section order creates backtracking: §1quinquies supersedes §1quater, but §1ter then appears after both and is also controlling for "why phi." Consider a short "Reading order" note after §1bis: "Current controlling sections are §1quinquies for identifiability and §1ter for entropy-bias interpretation; §1quater is historical."

- **Minor / Certain / Scope** — `B_union` is defined carefully in §1quinquies, but §3 later uses `dim(B_union) ~64` without reminding the reader it is a right-rowspace subspace, not DOC A's sketch matrix `B`. Add "`B_union` (right rowspace)" in §3's first sentence.
