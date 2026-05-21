# Overall Triage — 2026-04-29

DOC A: `reports/approximation/new_approx_combined.txt` (lower-bound derivation; reference)
DOC B: `reports/approximation/current_report.txt` (target document)

Five issues a careful reader hits on one cold read of DOC B:

1. **M3 is structurally mis-shelved as a failure.** §1's heading says "Two failures and one repair signal," but M3 sits inside the same bullet list as M1/M2 and is filed in "Appendix A. Evidence for the combined-score failures…". Text already calls M3 a "repair signal" / "fix direction" (lines 17–19, 173–174, 191–192), so the visual grouping fights the prose. Promote M3 to its own paragraph after M1/M2 — frame it as the bridge to S₆. *Why it matters:* M3 is the only constructive result in §1; burying it in the failure list weakens the §1 → §2 narrative. → **Advisor #5 (Logic)**.

2. **§1ter "three faces of one reliability failure" reading is missing here.** The score-design overview unifies M1/M2/M3; DOC B does not cite it. Either import that framing or explicitly say DOC B uses the older split. → **#5**.

3. **"First repair signal" implies more follow.** Line 18 calls M3 "the first repair signal"; no second is named. Drop "first" or list the others. → **#1**.

4. **DOC A → DOC B notation drift on φ.** DOC A uses base `row(N_w)/n ≤ 1` with negative exponent; DOC B's recap form flips to `n/row(C_w) ≥ 1`. Reconciliation paragraph (lines 52–76) is correct but dense; one reader-facing sentence up front would prevent confusion. → **#1**.

5. **§3 "Status" item 1 flags an unreconciled numeric discrepancy** (0.297 vs 0.2164 on diffuse-diffuse). Load-bearing because it sits inside the closure argument. → **#3 (Evidence)**.
