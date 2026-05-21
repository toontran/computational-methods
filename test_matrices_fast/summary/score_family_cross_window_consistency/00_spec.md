# FAM-07 — Cross-window consistency / correlation evidence (SPEC, frozen)

Date: 2026-04-29
Backlog: `summary/overview/score_family_workflow.txt` [FAM-07] (line ~1705)
Resolves: `summary/overview/score_design_overview.txt` Q13 (line ~1461);
toolkit gap §8(n) in `summary/overview/diagnostic_toolkit.txt`
Scope of this document: SPEC ONLY. T1 (gradient check) prototype is in
`01_t1_grad_check.py` and its rel-error log is in `02_t1_grad_check_log.txt`.
T2 and T3 are described but not implemented and not run.

------------------------------------------------------------------------

## 1. Motivation in one paragraph

HM3 (the in-pipeline score's evidence aggregator,
`hmean_evidence_score.py:hm_evi_value_grad`) enforces **magnitude balance**
across the three windows: the candidate must produce non-vanishing
`u_sk(v) = ||A_sk v||² / ||A_sk||_F²`, `u_g1(v) = ||A_cur v||² / ||A_cur||_F²`,
`u_g2(v) = ||A_fut v||² / ||A_fut||_F²`. It does NOT check whether
`A_cur v` and `A_fut v` (or their column-side response patterns) are
**directionally consistent**. The `static-cex` and `diffuse-diffuse`
FAIL-subspace matrices (§1quinquies of `score_design_overview.txt`,
canonical labels at line ~315) are the empirical motivation: `u_g1` and
`u_g2` are simultaneously high at non-oracle directions whose column-side
responses point to incompatible row-supports — exactly the regime HM3
cannot reject. Q13 calls for a cross-window correlation / CCA-style
statistic that closes this gap; FAM-07 produces the spec.

## 2. Constraint set & what "consistent" can mean

`A_cur ∈ R^{m_c × n}` and `A_fut ∈ R^{m_f × n}` have different row counts
in general (`m_c = m_f = half_win` for in-window streaming, but the spec
must not depend on equality). Therefore `A_cur v` and `A_fut v` live in
DIFFERENT row spaces and a direct cosine `<A_cur v, A_fut v>` is not
defined. We push the comparison to the column side, which is shared:

    g_cur(v) := A_cur^T A_cur v         in R^n
    g_fut(v) := A_fut^T A_fut v         in R^n

These are the column-side "Gram responses" — the same n-dimensional
linear functionals HM3 already uses internally as gradient pieces. They
agree iff the two windows assign coherent column-side energy to v.

## 3. Vector-level consistency term (rank-1)

### 3.1 Definition

Let `M_c = A_cur^T A_cur`, `M_f = A_fut^T A_fut`. Define:

    g_c(v) = M_c v ,         g_f(v) = M_f v

    rho_F(v) := < g_c(v), g_f(v) > / ( ||g_c(v)|| · ||g_f(v)|| )      (3.1)

This is a cosine in R^n between the two column-side responses. It is
scale-invariant in v. It is the natural rank-1 special case of (4.1).

### 3.2 Smoothness / domain

`rho_F` is smooth on `{v : g_c(v) ≠ 0  AND  g_f(v) ≠ 0}`. Both factors are
positive everywhere u_g1(v) > 0 and u_g2(v) > 0 respectively, so the
domain coincides with HM3's well-defined domain. Use a `eps = 1e-30`
floor on the denominator norms to avoid 0/0 on degenerate seeds.

### 3.3 Gradient (closed form)

Write `p = g_c(v)`, `q = g_f(v)`, `np = ||p||`, `nq = ||q||`,
`s = <p, q>`. Then

    rho_F(v) = s / (np · nq)

    d s  = q^T (M_c dv) + p^T (M_f dv)         = (M_c q + M_f p)^T dv
    d np = p^T (M_c dv) / np                  = (M_c p / np)^T dv
    d nq = q^T (M_f dv) / nq                  = (M_f q / nq)^T dv

So

    grad rho_F = (M_c q + M_f p) / (np · nq)
                 - rho_F · ( M_c p / np² + M_f q / nq² )            (3.2)

Equivalently in u-form (rho_F^2 in [0, 1]) the squared variant
`rho2(v) = rho_F(v)²` has gradient `2 rho_F · grad rho_F`.

### 3.4 Sign convention

We use `rho_F` (signed) inside diagnostics (T2) but `rho_F^2` inside any
multiplicative score factor — sign-flip degeneracy is irrelevant for
HM3-augmentation purposes, and the squared form is `O(1)`-invariant in
the sign of v.

## 4. Frame-level extension (rank-r)

The spec is explicit that the rank-r form must be a **frame
correlation**, NOT independent column correlations (FAM-07 backlog
Notes, line ~1721). We use a Frobenius / RV-coefficient style statistic.

### 4.1 Frame Gram responses

For `V ∈ St(n, r)`:

    G_c(V) = M_c V    in R^{n × r}
    G_f(V) = M_f V    in R^{n × r}

### 4.2 Frame consistency loss

Define

    K(V)   := G_c(V)^T G_f(V)   = V^T M_c M_f V          in R^{r × r}    (4.0)
    N(V)   := ||K(V)||_F²      = trace(K(V)^T K(V))                       (4.1a)
    D_c(V) := ||G_c(V)||_F²    = trace(V^T M_c² V)                        (4.1b)
    D_f(V) := ||G_f(V)||_F²    = trace(V^T M_f² V)                        (4.1c)

    rho_frame(V) := N(V) / ( D_c(V) · D_f(V) )                            (4.2)

Bounds: `0 ≤ rho_frame(V) ≤ 1` (Cauchy–Schwarz on the matrix inner
product `<X, Y>_F = trace(X^T Y)` with X = G_c, Y = G_f, treated as
elements of R^{n × r}).

Equality rho_frame = 1 iff the column-side responses span the same
subspace of R^n with the same r-frame energy distribution — i.e. the
two windows agree on the r-frame's full column-side Gram structure.

This IS frame-level: it does not factor into per-column correlations,
and it is invariant under right-rotations `V → V Q` (Q ∈ O(r)) — the
same O(r)-invariance HM3-frame and S6 already enjoy, which keeps it
drop-in compatible with the Stiefel-ascent infrastructure
(stiefel_grad_check.py).

### 4.3 Closed-form gradient

Both M_c and M_f are symmetric SPSD. Let `S := M_c M_f` (NOT symmetric).
Then `K = V^T S V` and `N = trace(K^T K)`. Differentiating:

    dK = (dV)^T S V + V^T S dV
    dN = 2 trace(K^T dK)
       = 2 [ trace(K^T (dV)^T S V) + trace(K^T V^T S dV) ]
       = 2 [ <S V K^T, dV>_F + <S^T V K, dV>_F ]

so

    grad N  = 2 ( S V K^T + S^T V K )                                    (4.3a)
            = 2 ( (M_c M_f V) K^T + (M_f M_c V) K )

    grad Dc = 2 M_c² V                                                    (4.3b)
    grad Df = 2 M_f² V                                                    (4.3c)

By the quotient rule on (4.2),

    grad rho_frame
        = (1 / (D_c D_f)) · grad N
          - (N / (D_c² D_f)) · grad Dc
          - (N / (D_c D_f²)) · grad Df                                   (4.4)

This is the EUCLIDEAN gradient w.r.t. the entries of V. The Stiefel
tangent projection is applied by the FD harness
(`stiefel_grad_check.stiefel_tangent_project`) — the score
implementation returns the unconstrained gradient, exactly as
`frame_S6_value_grad` does. (`stiefel_grad_check.py` lines ~334–379.)

### 4.4 Why rho_frame, not per-column cosines

The "independent column correlations" alternative `(1/r) Σ_j ρ_F(V[:,j])`
fails three things FAM-07 must respect:

  - It is NOT O(r)-invariant (right-rotates V change the value), so it
    is incompatible with the subspace-claim layer of `score_design_-
    overview.txt §1quinquies`: a rank-r frame score may only score
    span(V), not the chosen basis within span(V).
  - It cannot detect frame-level drift like a 90° in-span rotation, the
    exact pathology that makes the FAIL-subspace pair (`static-cex`,
    `diffuse-diffuse`) hard.
  - It is not aligned with FAM-01 / S6's frame Stiefel-ascent path; the
    optimizer would have to be modified to handle column-orientation,
    which is out of scope for FAM-07.

`rho_frame` is the smallest extension that (a) reduces to (3.1)² for r=1
(modulo a fixed factor) and (b) is O(r)-invariant.

## 5. Relationship to HM3 (additive, multiplicative, replacement?)

**Recommendation: MULTIPLICATIVE, opt-in via a hyperparameter `lambda_F`.**
HM3 stays the unmodified magnitude-balance backbone; FAM-07 adds a
consistency factor.

Vector level (matches the existing `score = HM_evi · relH1` pattern in
`hmean_evidence_score.py:hm_evi_value_grad`):

    score_FAM07_vec(v) = HM_evi(v) · relH1(v) · (rho_F(v)²)^lambda_F     (5.1)

Frame level (matches the S6/HM3 frame lift in `stiefel_grad_check.py`):

    score_FAM07_frame(V) = HM3_frame(V) · rho_frame(V)^lambda_F          (5.2)

Defaults:

    lambda_F = 1.0      (full-strength multiplier; T2/T3 can sweep 0/0.5/1/2)

Justifications for multiplicative over additive:

  - HM3 is already a (weighted) reciprocal-mean — it lives on a different
    scale than rho_F ∈ [0, 1], so additive mixing requires a rescale
    hyperparameter that defeats the "drop-in" requirement.
  - The product is bounded by HM3 (because rho_frame ∈ [0, 1]),
    preserving HM3's row-cheat dominance behavior (S-3 in
    score_design_overview.txt §1quinquies, line ~338): if HM3 already
    rejects a candidate, multiplying by ≤1 cannot resurrect it.
  - The product gradient is a clean superposition of HM3's gradient
    (already verified in INFRA-02) and rho_frame's gradient (verified
    in this spec's prototype) — no new cross-terms beyond chain rule,
    so the gradient check decomposes.

Replacement (rho_frame INSTEAD OF HM_evi) is **explicitly not
recommended** because rho_frame ignores magnitudes — the two windows
can have rho_frame = 1 while u_g1 ≈ 0 (collinearity at zero energy is
trivially satisfied). HM3 + rho_frame is strictly more informative than
either alone.

Additive (HM_evi + alpha · rho_frame) is left as a kill-criterion
fallback (see §8) — used only if the multiplicative form fails T2/T3.

## 6. Hyperparameters and defaults

| name        | type   | default | role                                            |
|-------------|--------|---------|-------------------------------------------------|
| `lambda_F`  | float  | 1.0     | exponent on the consistency multiplier          |
| `eps_norm`  | float  | 1e-30   | floor on ||g_c||, ||g_f||, D_c, D_f             |
| `mode`      | str    | "mult"  | "mult" (5.1)/(5.2), "add" (kill-fallback)       |
| `alpha`     | float  | 0.1     | additive-mode weight (only if `mode=add`)       |
| `signed`    | bool   | False   | True ⇒ use rho_F (signed) at vector level       |

In gradient context, `lambda_F = 1` and `mode = mult` are the **frozen
defaults**. T2 ablation is allowed to vary `lambda_F ∈ {0, 0.5, 1, 2}`;
T3 ships with `lambda_F = 1` unless T2 demonstrably wants otherwise.

## 7. Acceptance bars (T1 / T2 / T3) — verbatim from the backlog

T1 — Gradient check (THIS SPEC SHIPS).

  Pass criterion (per [FAM-07] Acceptance, workflow line ~1712 and §8(n)
  toolkit reference): `rel < 1e-7` against finite differences along
  Stiefel tangent directions, on synthetic data, at float64. Use the
  INFRA-02 harness (`stiefel_grad_check.stiefel_fd_check`) with
  `retraction = "polar"`, `direction_mode = "column"`, `eps = 1e-5`,
  `n_directions = 8`. Accepted iff `max_rel < 1e-7` on (a) rank 1 (b)
  rank 2 (c) rank 3 (d) rank-1 score = `rho_F²` (e) rank-r score =
  `rho_frame`. Trace-form sanity row alongside (rel < 1e-9), as INFRA-02
  prescribes. Numbers are recorded in `02_t1_grad_check_log.txt`.

T2 — Diagnostic identification (description, NOT RUN).

  At b1, b2, b12, b31 of `mixed-tail-sharp`, `static-cex`,
  `diffuse-diffuse`, `etf-basket-basis`, dump the per-block table

      label, u_g1, u_g2, HM_evi, rho_F (or rho_frame), score_HM3,
      score_FAM07, align_v1, align_v2

  for the candidates `combined_optimizer_v2`, `hm_triplet_evidence_best`,
  `oracle_v1_proj_S+G1+G2`, `oracle_v2_proj_S+G1+G2`,
  `frame_oracle_proj`, `frame_S6` (greedy), `frame_rowcheat`. Pass iff
  there is at least one (matrix, block) cell where:

      u_g1 ≥ 0.5 · u_g1(oracle_v_k_proj)  AND
      u_g2 ≥ 0.5 · u_g2(oracle_v_k_proj)  AND
      rho_F  < 0.5

  i.e. the "high HM-magnitude, low cross-window consistency" diagnostic
  signature actually fires on a real failure mode. If T2 cannot find
  this signature on any §6 high-entropy cell, FAM-07 is killed (see §8).

T3 — Streaming bench (description, NOT RUN).

  Wire `score_FAM07_frame` (5.2) as a `--score-variant` option in
  `r_sk_g_score.py`, run the §6 streaming bench at `half_win=64` (the
  primary handoff configuration per workflow §6) on the seven §6
  matrices plus `etf-basket-basis`. Pass iff:

      (a) at least ONE of the four high-entropy matrices
          (`mixed-tail-sharp`, `mixed-tail-balanced`, `mixed-tail-soft`,
          `risk-residual-panel`) improves cos0² by ≥ 0.05 vs the
          parallel S6/HM3 run,
      (b) NONE of `static-cex`, `mixed-tail-sharp`, `diffuse-diffuse`,
          `etf-basket-basis` regresses cos0² by > 0.02 (the §6 no-regress
          set called out in the FAM-07 backlog).

  Failures of (b) on `static-cex` or `diffuse-diffuse` are EXPECTED to
  motivate further work in FAM-07 (these are the FAIL-subspace pair the
  spec primarily targets) but are NOT ship-blockers iff (a) lands on
  high-entropy AND the regression is < 0.05 (graceful-degradation rule
  consistent with FAM-01 B0's two-part S-2 acceptance, line ~1788).

## 8. Kill criteria (mandatory; from the backlog Acceptance)

FAM-07 is killed and the directory is archived if any of the following
fires:

  K1. **Numerically unstable.** rho_frame's gradient produces rel-error
      ≥ 1e-7 on more than 10% of T1 synthetic cells at float64, OR the
      gradient check requires `eps_norm > 1e-12` to converge. (Both
      indicate ill-conditioning that won't survive streaming-state drift.)

  K2. **Duplicates HM3.** T2 finds < 1 cell across the (matrix, block)
      probe grid where the diagnostic signature
      "high HM-magnitudes, low rho_F" fires. This means rho_F adds no
      new information beyond what HM3 already encodes, regardless of
      its T1 numerical correctness.

  K3. **Regresses the no-regress set.** T3 regresses cos0² on
      `etf-basket-basis` by > 0.02 (the matrix where S6 already wins
      slot-1 on §6 — DIAG-04b note, line ~1495), OR regresses any §6
      matrix by > 0.05 even with a high-entropy gain elsewhere.

  K4. **Replacement of HM3 fails worse than multiplicative.** Optional
      kill: the replacement variant `score = rho_frame · relH1`
      (no HM3) shipped instead of (5.2) regresses every §6 matrix.
      This is documentation-only kill (we already DO NOT recommend
      replacement); records the negative result for posterity.

K1 and K2 are dispositive at SPEC time; K3 is dispositive at T3 time.

## 9. Compatibility / write scope

- This spec does NOT modify `r_sk_g_score.py` or `hmean_evidence_score.py`
  (per task constraint "do not modify r_sk_g_score.py").
- The T1 prototype is self-contained in `01_t1_grad_check.py`.
- When FAM-07 graduates to T2/T3, the implementation surface is:
    1. A new `frame_FAM07_value_grad(M_c, M_f, V)` returning
       `(rho_frame(V), grad)` — drop-in alongside
       `frame_S6_value_grad` in `stiefel_grad_check.py`.
    2. A composite `frame_S6_FAM07_value_grad(...)` that multiplies
       (5.2) and chain-rules the gradient.
    3. A new `--score-variant rho_F_mult` option in `r_sk_g_score.py`.
- Keep write scope SEPARATE from FAM-01 unless FAM-07 is explicitly
  wired into a FAM-01 rank-r variant — backlog Notes, line ~1723.

## 10. Open design questions (input requested before T2/T3)

Q-FAM07-1 (lambda_F default).
  Spec freezes `lambda_F = 1`; T2 ablation will sweep {0, 0.5, 1, 2}.
  If user prefers a softer default (e.g. 0.5) for the initial ship to
  reduce risk of false rejections on low-rho but otherwise-good
  candidates, please flag before T3.

Q-FAM07-2 (which HM3 backbone to multiply onto).
  Three options for (5.2): plain `HM_evi`, `HM_evi · relH1` (the
  current shipped score), or just `HM3_frame` (S6/S6_GM). Spec assumes
  "mirror whatever variant T3 is benched against": for the streaming
  pipeline that is the S6 frame score `HM3_frame(V)`. Confirm before
  wiring.

Q-FAM07-3 (RV vs alternative frame correlations).
  rho_frame in (4.2) is a Frobenius-normalized RV-style statistic. An
  alternative is the canonical-correlation product
    rho_CCA = prod_k cos²(theta_k)   (squared canonical correlations
                                      between range(G_c) and range(G_f))
  or `det(K^T K) / (det(G_c^T G_c) det(G_f^T G_f))` (a multi-linear
  generalization of cos²). RV is chosen for cleaner gradient and unit
  range, but if T2 shows weak signal on FAIL-subspace, Q-FAM07-3 is the
  natural escalation. Defer to T2 evidence; no decision needed at SPEC.

Q-FAM07-4 (sketch term).
  Should rho_frame extend to three-window consistency
  (`rho_3 := f(M_sk V, M_c V, M_f V)`)? At rank 1, possible via
  trace-correlation among three vectors; at rank-r, requires a
  three-Gram analogue of the RV coefficient. Not spec'd; flagged as a
  follow-up if FAM-07 ships.

------------------------------------------------------------------------

## Files

- This spec: `summary/score_family_cross_window_consistency/00_spec.md`
- T1 prototype: `summary/score_family_cross_window_consistency/01_t1_grad_check.py`
- T1 log: `summary/score_family_cross_window_consistency/02_t1_grad_check_log.txt`
- Stiefel FD harness used: `stiefel_grad_check.py` (INFRA-02)
- Reference HM3 backbone: `hmean_evidence_score.py:hm_evi_value_grad`
- Reference S6 frame backbone: `stiefel_grad_check.py:frame_S6_value_grad`
