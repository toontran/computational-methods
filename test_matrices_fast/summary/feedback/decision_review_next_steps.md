# Decision Review: Next Steps

## Fatal / Major Flags

- **Fatal [confidence: high; scope: Q0 / production-comparison decision]**: the decision criterion still depends on `future_hmean_online`, but the report says the cited online numbers are oracle-aware and must be replaced by the value-only rerun (`INFRA-10`). Until that rerun is propagated, the project cannot decide whether S6/rank-r must "match online" or merely beat a much lower operational baseline. The document acknowledges this, but the priority queue still leans on the old online gap as motivation.

- **Major [confidence: high; scope: Q1 / FAM-01 / rank-r work]**: the next action should be framed as a kill/continue diagnostic, not as implementation momentum. The load-bearing question is whether a value-only frame-level score makes the oracle frame the winner. If `FAM-01-DIAG` finds `Z_winner > Z_oracle`, the next step is feature redesign, not more Stiefel optimizer work.

- **Major [confidence: medium-high; scope: open-work management]**: the open-question list mixes resolved diagnostics, low-value documentation cleanup, and true decision gates. That makes the queue look broader than it is. The live decision-relevant queue is much shorter: value-only online rerun, frame-level oracle-vs-winner screen, then reliability-feature probes only if the frame screen fails.

## Actual Decision At Stake

The report informs this project decision:

**Should the score-design path continue as the preferred value-only, differentiable, rank-r-liftable alternative to online, or should the project pivot toward new evidence features / hybrid SVD coverage because magnitude-only HM-style scoring cannot identify the oracle frame?**

The current conclusion partly drives that decision. It does more than characterize a phenomenon: it says scalar rank-1 score variants have plateau/argmax-identifiability failures, and that the next structural test is rank-r frame scoring. But the decision is not fully actionable until the oracle-aware online ceiling is replaced by the value-only online baseline. Without that, the target gap and success bar are unstable.

The cheapest experiment whose result can change the next action is:

**Run `FAM-01-DIAG`: high-budget frame-level oracle-vs-winner screens for the rank-r HM3 lift on the §6 high-entropy matrices, after or alongside propagating the value-only online rerun.**

Decision rule:

- If rank-r HM3 makes `Z_oracle` competitive with or above `Z_winner`, proceed to T3 streaming validation.
- If rank-r HM3 still has `Z_winner > Z_oracle`, stop treating rank-r lifting as the fix and move to evidence features: cross-window consistency, robust aggregation, support/stability-aware `u_X`.
- If value-only online is much weaker than the oracle-aware ceiling, lower the operational match target and re-rank work by whether it beats actual value-only online, not the ceiling.

## Information Value Of Open Items

| Item | Information value | Cost | Decision relevance | Recommendation |
|---|---:|---:|---:|---|
| `INFRA-10` value-only online rerun / table propagation | Very high | Low-medium | Very high | Do first or in parallel with Q1. It defines the real target. |
| Q1 / `FAM-01`: rank-r frame lift | Very high | Medium-high | Very high | Top structural experiment. Treat as a sufficiency diagnostic before full production work. |
| `FAM-01-DIAG` frame-level oracle-vs-winner screen | Very high | Medium | Very high | Highest information per cost among score-design experiments. Required before believing rank-r helps. |
| Q2 plateau width quantification | Medium | Low-medium | Medium | Useful after Q1 failure to distinguish broad plateau from wrong feature peak. Not first. |
| Q3 relH1 | Low as D0; medium for new forms | Medium | Medium | D0 is refuted. Only revisit as reliability-aware `u_X`, not as `S6 * relH1(A_cur v)`. |
| Q4 leverage/per-row weighting | Medium | Medium | Medium | Defer until Q1 says magnitude-only frame HM3 is insufficient. |
| Q5 carry-confidence multipliers | Medium-high | Medium | Medium-high | More meaningful after rank-r is in place; likely a feature/weighting follow-up, not first. |
| Q6 rank-aware threshold | Medium | Medium | Medium | Defer behind Q1 and Q5. It is plausible but not the cheapest decision gate. |
| Q7 HM vs GM | Closed | None | Low | Resolved/refuted. Do not keep in the active queue. |
| Q8 HM3 theory | Closed for current decision | None | Low | Useful framing, but not action-changing now. |
| Q9 entropy proxy theory | Closed at model level | None | Low | Background only unless designing TH-02/TH-03. |
| Q10 regime-label audit | Closed | None | Medium historically | Its main decision impact already landed: residual-spiky is boundary, not primary ship criterion. |
| Q11 tail-conspiracy verification | Closed | None | Medium historically | Use its result to justify reliability features; do not keep as open work. |
| Q12 scalar-score impossibility / counterexample | Medium | High | Medium | Valuable if writing a paper or deciding to abandon scalar scores permanently; not the next experiment. |
| Q13 cross-window correlation | High if Q1 fails | Medium | High | Best next feature probe after rank-r HM3 fails the frame screen. |
| Q14 robust row aggregation | Medium-high if row-cheat remains active | Medium | Medium-high | Pair with row-cheat diagnostics; not first unless Q1 failure is row/outlier-driven. |
| Q15 subsampling stability | Medium | Medium-high | Medium | Good reliability signal, but likely slower than Q13/Q14. |
| Q16 oracle u-balance / AB-03 | Mostly closed for phase 1 | Low remaining | Medium | Do not continue balance-only schemes unless they pass oracle-vs-winner screens. |
| Q17 notation/domain cleanup | Low | Low | Low | Documentation hygiene. Do after decision gates unless ambiguity blocks implementation. |

## Recommended Re-Prioritisation

1. **Propagate `INFRA-10` value-only online results.** This fixes the target used by Q0 and prevents optimizing against an oracle-aware ceiling.
2. **Run `FAM-01-DIAG` before full rank-r streaming investment.** The key measurement is whether the frame-level score identifies the oracle frame, not whether the optimizer can be implemented.
3. **If `FAM-01-DIAG` passes, run T3 rank-r validation on the §6 high-entropy suite.** Evaluate against value-only online, not the oracle-aware table.
4. **If `FAM-01-DIAG` fails, pivot from aggregator tweaks to evidence features.** Start with Q13 cross-window response consistency, then Q14 robust row aggregation, then Q5/Q6 rank/carry-aware weighting.
5. **Keep Q2 as a diagnostic, not a blocker.** It explains failures; it is less likely than Q1/Q13 to change the next action.
6. **Move resolved items out of the active open-work list.** Q7-Q11 and Q16 phase 1 should be summarized as prior evidence, not competing next steps.

## Low-Value Vs Load-Bearing

Load-bearing:

- value-only online rerun (`INFRA-10`);
- frame-level oracle-vs-winner screen (`FAM-01-DIAG`);
- rank-r T3 only after the frame screen passes;
- reliability-aware evidence features if the frame screen fails.

Low-value or premature:

- further scalar rank-1 epicycles without oracle-vs-winner screens;
- balance-only audits that do not check the actual argmax;
- D0-style relH1 multiplication;
- notation cleanup as a decision blocker;
- theory polish before the empirical gate decides whether the current score family is alive.

Bottom line: the next-step queue should be one decision gate deep. First establish the real value-only online target and whether rank-r HM3 identifies the oracle frame. That result decides whether to build rank-r production machinery or pivot to new reliability-aware evidence.
