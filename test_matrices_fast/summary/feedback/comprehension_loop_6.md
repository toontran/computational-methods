## Comprehension Loop 6

No source edits were made.

- **Moderate** — The $E_2$ “success” metric is underspecified. Lines 238--257 say $\mathrm{HM}_3$ needs balanced $(u_{\mathrm{sk}},u_{\mathrm{cur}},u_{\mathrm{fut}})$, then say $E_2$ “drives the oracle slot ratio inside $2.4\times$,” but the only nearby ratio is $u_{\mathrm{sk}}/u_{\mathrm{cur}}$. State whether $2.4\times$ is max/min over all three slots, only sketch/current, or another audit ratio.

- **Moderate** — V1’s scale is hard to compare to HM. $S_1$ multiplies dimensionless $S_{\mathrm{HM}}$ by $\|A_{\mathrm{fut}}^\top A_{\mathrm{fut}}Z\|_F^2/\|A_{\mathrm{fut}}Z\|_F^2$ (lines 416--419), a squared-energy scale, but the conclusion compares V1 and V3 score gaps directly (lines 474--478; table lines 789--794). Add the normalization or say only within-feature argmax gaps are meaningful.

- **Moderate** — “Row-cheat” appears as if it is both an existing screen and future work. Lines 367--371 include “row-cheat dominance” among canonical H3 screens; line 509 says to build a frame-level row-cheat diagnostic. Clarify whether a vector-level version already exists and the frame-level version is missing, or whether this is proposed only.

- **Minor** — The $\cos_1^2$ notation changes indexing convention late. The block-31 table uses “terminal $\cos_1^2$” for the leading direction (lines 632--642), while Appendix A.4 later explains zero-based $\cos_0^2+\cos_1^2$ and says this differs from the §1 shorthand (lines 725--731). Define the shorthand at first use.

- **Trivial** — The frame-lifting definitions include the tautology $A_{\mathrm{fut}}:=A_{\mathrm{fut}}$ (lines 383--387). Rename the source matrix or simply say “and $A_{\mathrm{fut}}$ is the following same-size half-window.”
