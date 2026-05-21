# Idea / Scientific-Critique Advisor: Scientific Critique

Source order: DOC A `../reports/approximation/new_approx_combined.txt`, then DOC B
`summary/overview/score_design_overview.txt`.

## Major / Fatal Flags

- **Major | Likely | Core**: The current decomposition is not genuinely
  orthogonal. M1/M2/M3/M4, plateau drift, tail conspiracy, carry pinning, and
  oracle u-imbalance are overlapping projections of one identifiability
  problem: the visible evidence does not uniquely identify the oracle frame.
  DOC B partly acknowledges this in DIAG-05 and §1quinquies, but several later
  priority statements still treat rank-r lift, relH1, carry confidence, and
  robust aggregation as separable fixes before proving which evidence signal
  makes `Z_oracle` the maximizer.

- **Major | Likely | Method**: The most load-bearing proposed fix, FAM-01
  rank-r lift, is under-justified as a remedy for the plateau. A scalar
  per-vector level set being codimension 1 does not imply a Frobenius frame HM3
  score will pin the oracle subspace. DOC B correctly adds S-2
  `Score(Z_winner) <= Score(Z_oracle)` as the definitive screen, but the
  narrative still leans toward "rank-r lift first" before demonstrating that
  the current evidence model ranks the oracle frame above row-cheat and
  non-oracle balanced frames.

- **Major | Likely | Assumption**: The high-sample-entropy complementarity
  story is plausible but not yet established as a decision principle. DOC B
  concedes that DOC A's normalized entropy invariance (A2) is not expected to
  hold on the suite, and that HM3 is only a max-min visible-stability surrogate.
  The replacement premise, "high visible entropy identifies high-entropy
  oracles," is still model- and matrix-dependent; DIAG-01 supports regime
  labels, but not that the score's high-entropy winner is the oracle.

## Verdict

**Major | Likely | Core**: The most principled part remains DOC A's combined
score derivation: if (A1), (A2), and `||N_w v||_4 <= ||A v||_4` hold, the
combined score follows from the lower-bound chain ending at `(*)`. But the
project has empirically moved away from those assumptions. DOC B's current
science is therefore not "derived approximation improved by fixes"; it is
"designing a value-only reliability surrogate whose argmax should identify the
oracle frame."

That reframing is directionally right. The strongest current principle is
§1quinquies: ask whether the score makes the oracle vector/frame the winner,
not whether audit features look balanced at the oracle. The weakest current
principle is the residual habit of treating local symptoms as semi-independent
design axes. The next decisive evidence is not another rank-1 scalar variant;
it is a high-budget frame-level oracle-vs-winner and row-cheat screen.

## Robust claims

- **Moderate | Certain | Core**: DOC A's algebra is internally coherent under
  its assumptions. The lower bound
  `||A v||_2^2 >= ||N_w v||_2^2 exp(0.5(h_2(A v)-h_2(N_w v)))` follows from
  the `l4` subset inequality, and (A2) converts it into
  `(row(N_w)/n)^(-h_2(N_w v)/(2 log row(N_w)))`. Substituting (A1) and the
  `C_w` half of (A2) gives the combined score.

- **Moderate | Likely | Method**: Carrying
  `B_next = Sigma V_new^T` rather than diagonal column norms is correct. The
  DOC A example `M_w = [[1,1],[0,0]]` with `V_hat=[e1,e2]` shows that column
  norms erase cross-correlations: the diagonal carry gives constant norm 1 on
  the sphere, while the true quadratic vanishes at `(1,-1)/sqrt(2)` and equals
  2 at `(1,1)/sqrt(2)`.

- **Major | Likely | Core**: The original combined score has verified failure
  modes on the suite. The reported M1 numbers are especially diagnostic:
  `gain_tot` is nearly flat (`0.819-0.833`), while `phi` separates
  `combined_v_2` from `oracle_v_2_proj_perp` (`5.66` vs `3.70`), so the score
  gap is genuinely the entropy multiplier rather than energy.

- **Major | Likely | Core**: Greedy deflation alone is not the explanation for
  original combined-score failure. DOC B's R1 says joint Stiefel-2 with the
  same score gives essentially the same state-pinned slot-2 at later blocks
  (`state_align ~= 0.989`). That supports the stronger claim that the score,
  not merely the greedy extraction procedure, is misaligned.

- **Moderate | Likely | Method**: E2 is correctly killed as a sufficient design
  principle. The decisive evidence is not just bench regression; it is the
  oracle-vs-winner gap moving the argmax farther from the oracle:
  `+0.041` on diffuse-diffuse and `+0.047` on residual-spiky-shocks versus
  S6's `+0.007`. That directly refutes "better oracle u-balance implies better
  score."

- **Moderate | Likely | Method**: The oracle-aware online caveat is essential.
  Since the old online pool includes `q2_vs_q1oracle`, the §6 online numbers
  are a ceiling, not an operational comparator. This materially changes the
  interpretation of gaps such as S6 `0.013` vs online `0.850` on
  mixed-tail-sharp.

## Weakly supported

- **Major | Likely | Core**: The decomposition is not orthogonal enough to
  support independent fix queues. M1's phi penalty, M2's carry pinning, M3's
  missing future evidence, M4's oracle u-imbalance, and plateau drift all
  change the same object: which point or frame maximizes the visible score.
  DOC B's DIAG-05 unification is the right move, but the open-question list
  still risks optimizing support, weights, aggregator, and rank-r geometry as
  if they were clean axes.

- **Major | Likely | Assumption**: The rank-r plateau argument is suggestive,
  not yet a scientific conclusion. "A scalar-per-v score has codim-1 level
  sets" is true but too generic; almost every continuous scalar objective has
  level sets. The relevant question is whether the near-optimal set intersects
  many non-oracle subspaces at comparable score. INFRA-05 and the T2 gaps are
  evidence, but not a proof that rank-r Frobenius HM3 will shrink the bad set.

- **Major | Likely | Core**: The complementarity claim has a missing bridge.
  DOC B says iSVD is the low-entropy specialist and the score family should be
  the high-entropy specialist. But the key empirical failures include
  diffuse-diffuse (`S6 cos1^2 = 0.005`) and mixed-tail-sharp (`0.013`), which
  are classified as high-entropy/home-turf. Until a score succeeds there, the
  complementarity story is an aspiration, not an explanation.

- **Moderate | Likely | Method**: HM3 as "balance enforcer" is locally
  justified but not uniquely principled. TH-01's
  `min_i u_i <= HM_k(u) <= k min_i u_i` only justifies HM as a smooth max-min
  proxy. It does not justify equal source weights, Frobenius denominators, or
  magnitude-only `u_X` as the right evidence variables.

- **Moderate | Likely | Method**: The row-concentration / relH1 path remains
  underspecified after D0. D0 refutes only `S6 * relH1(A_cur v)`, not
  reliability-aware evidence generally. The document says this, but the next
  experiments need to distinguish "support modifies evidence" from "support is
  another multiplier."

- **Moderate | Speculative | Assumption**: Cross-window correlation is a
  plausible missing signal, but its role is not yet isolated. HM3 balances
  magnitudes across windows; a correlation/CCA term would test whether the same
  structure recurs. However, for independent row draws from the same right
  direction, raw response-vector correlation across different rows may be the
  wrong observable unless rows are paired by latent component, leverage class,
  or feature response statistics.

## Probably wrong

- **Major | Likely | Core**: "Slot-1 is solved" is too strong. DOC B itself
  reports S6 `cos0^2` values such as `0.733` on mixed-tail-sharp, `0.687` on
  mixed-tail-balanced, `0.749` on diffuse-diffuse, and `0.333` on
  residual-spiky-shocks. The text later says the gap is "entirely in slot-2,"
  but force-oracle-v2 failing on 6/7 matrices because `V_default[:,0]` is bad
  contradicts that. A better claim is: slot-1 is less broken than slot-2 on
  several clean-rowspace matrices, but anchor error remains load-bearing.

- **Major | Likely | Method**: Treating rank-r lift as the "structural fix"
  before S-2 passes is probably premature. A frame-level objective can remove
  oriented-slot artifacts, but if `Score(Z_winner) > Score(Z_oracle)` under the
  same magnitude-only evidence, joint optimization will simply find a better
  non-oracle frame. DOC B says this in §1quinquies; the priority rationale
  should fully absorb it.

- **Moderate | Likely | Presentation**: The phrase "high-sample-entropy
  regime" is overloaded. DIAG-01 measuring oracle effective support supports
  a label about the oracle, not necessarily the score winner, row-cheat, or
  tail-conspiracy candidate. A matrix can have high-entropy oracle directions
  and still have higher-scoring non-oracle visible conspiracies.

- **Moderate | Speculative | Assumption**: The claim that Q4
  per-row/leverage-aware weighting is "probably equivalent up to scale" is not
  safe under heavy tails. If the central pathology is outlier leverage and
  unreliable row evidence, leverage-aware weighting may change the ordering of
  candidates, not merely rescale the score.

- **Minor | Likely | Method**: The block-1 HM2 fall-through is called natural,
  but its own numbers show it can prefer a local split-sample optimizer over
  the projected full-matrix oracle by a small energy gap (`0.01638` vs
  `0.01580` on mixed-tail-sharp b1). That may be acceptable, but it is not a
  derivation of oracle identification; it is a local evidence criterion.

## Reframings

- **Major | Likely | Core**: Reframe the whole line as an identifiability
  problem, not a sequence of fixes to combined. The key question is:
  "Which value-only evidence variables make the oracle subspace the unique or
  stable maximizer over the available row-space?" Under this framing, entropy,
  HM balance, relH1, robust row aggregation, carry confidence, and rank-r lift
  are candidate evidence/geometry choices, not independent mechanisms.

- **Major | Likely | Method**: Separate three concepts that are currently
  interleaved:
  1. **Evidence sufficiency**: does the score rank `Z_oracle` above
     non-oracle frames and row-cheat frames?
  2. **Optimization reachability**: can the implemented ascent find the
     high-scoring oracle basin if it exists?
  3. **Streaming state stability**: does repeatedly carrying the selected
     frame preserve the target across blocks?
  Current sweeps often mix these. Force-oracle-frame tests state ceiling;
  oracle-vs-winner tests evidence sufficiency; restart/landscape probes test
  optimization.

- **Moderate | Likely | Core**: Reframe DOC A's `phi` as a derived but brittle
  estimator, not simply a bad entropy gate. Under (A2), `phi` is principled.
  Under the benchmark matrices, A2 fails and `phi` becomes an unreliable
  high-entropy inductive bias. That distinction prevents overcorrecting away
  all entropy information just because the original estimator failed.

- **Moderate | Likely | Method**: Reframe relH1 and robust aggregation as
  changes to `u_X`, not post-hoc penalties. The reliability-aware evidence
  section says this correctly. Scientifically, support should alter the
  estimated reliability of `||A_X v||^2`, not merely multiply an already
  magnitude-only score after the fact.

- **Moderate | Speculative | Core**: Reframe "online baseline" into two
  baselines: oracle-aware ceiling and value-only candidate-pool policy. The
  former answers "what if the right vector is in the pool"; the latter answers
  "what can an operational method do." Score-design should compete with the
  second and use the first only as a reachability upper bound.

## Next experiments

- **Major | Certain | Method**: Run FAM-01-DIAG before treating rank-r lift as
  a fix. For each §6 matrix and selected blocks, compare high-budget
  `Z_winner` against `Z_oracle` and `Z_rowcheat` for the frame score
  `HM3(||A_sketch Z||_F^2/||A_sketch||_F^2,
       ||A_cur Z||_F^2/||A_cur||_F^2,
       ||A_fut Z||_F^2/||A_fut||_F^2)`.
  Acceptance: `Score(Z_oracle) >= Score(Z_winner)` within tolerance and
  `Score(Z_oracle) >= Score(Z_rowcheat)` on the high-entropy/home-turf
  matrices. If this fails, do not proceed as if Stiefel optimization is the
  missing ingredient.

- **Major | Likely | Method**: Add a three-way diagnostic table for every new
  score: evidence sufficiency, optimizer reachability, streaming stability.
  Minimal columns: oracle score, best-restart winner score, row-cheat score,
  principal angles to oracle, force-oracle-frame terminal cos sum, normal
  streaming terminal cos sum. This would prevent another E2-style result where
  an audit metric improves while the argmax moves away from the oracle.

- **Major | Likely | Core**: Test reliability-aware `u_X` definitions at the
  frame level, not only scalar multipliers. Examples:
  capped/Huberized row-energy sums, trimmed row-energy sums, or
  `u_X(Z) * support_X(Z)` where support is computed per source. Compare these
  against row-cheat and oracle-vs-winner screens before T3 streaming.

- **Moderate | Likely | Assumption**: Quantify A2 failure directly on the same
  matrices. For oracle, S6 winner, combined winner, and row-cheat directions,
  measure
  `h_2(C_w v)/log row(C_w)`,
  `h_2(N_w v)/log row(N_w)`, and
  `h_2(A v)/log n`.
  This would connect DOC A to DOC B and show when `phi` fails because entropy
  invariance fails versus when optimization/carry causes the problem.

- **Moderate | Likely | Method**: Replace "plateau width" sampling by
  near-optimal set geometry at vector and frame levels. Sample points with
  `score >= 0.95 score(opt)`, but report principal-angle distribution,
  row-support reliability, source-balance, and future-block persistence. A
  wide near-optimal set is only scientifically relevant if it contains many
  unreliable non-oracle candidates that the optimizer actually reaches.

- **Moderate | Speculative | Method**: Prototype cross-window consistency only
  after defining the observable. A useful first version may compare per-feature
  or per-component response statistics rather than raw rowwise correlation
  between unrelated rows. Screen it with oracle-vs-winner first; do not judge
  it by terminal streaming cos alone.

- **Minor | Likely | Implementation**: Once value-only online reruns land,
  rewrite all Q0 success thresholds against that baseline. Keep oracle-aware
  online in tables only as a ceiling, clearly separated from operational
  competitors.
