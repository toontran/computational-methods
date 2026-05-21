HM3 theory note: stability surrogate, not recovery theorem
===========================================================

Date: 2026-04-28
Backlog item: TH-01
Verdict: iterate


1. Question
-----------

The current S6 score uses

  u_sk(v) = ||A_sketch v||^2 / ||A_sketch||_F^2
  u_g1(v) = ||A_cur v||^2    / ||A_cur||_F^2
  u_g2(v) = ||A_fut v||^2    / ||A_fut||_F^2

and, when the sketch is present,

  HM3(v) = 3 / (1/u_sk(v) + 1/u_g1(v) + 1/u_g2(v)).

At block 1 it falls through to HM2(u_g1, u_g2). The question is whether
this has a theorem-shaped story comparable to the original combined
score's A2-derived phi factor, or whether it must remain labeled as a
heuristic inductive bias.

The answer is in between: HM3 is a defensible smooth proxy for a visible
split-sample stability objective, but not a lower bound on population
energy, population entropy, or subspace recovery without additional
model assumptions. S6 should therefore be presented as a stability
surrogate / inductive bias, not as a derived estimator.


2. Evidence from local context
------------------------------

The local design docs already distinguish the two theory statuses:

  - summary/overview/score_design_overview.txt §1ter says combined has
    an A2-derived lower-bound / surrogate story for phi, while HM3 is
    currently a high-entropy stability bias.
  - summary/tail_conspiracy_insights_synthesis.txt §"The Combined
    Score" records the combined score as

      (||Bv||^2 + ||A_w v||^2) * phi_w(v),

    with phi_w written in terms of the pooled collision ratio of
    C_w v. Under the A2 story, normalized h_2 is assumed to transfer
    from the observed rows to the unseen/full matrix.
  - summary/f_hm3_score_implementation_context.txt defines S6 as the
    F-weighted HM3 above, records that HM3 makes combined_v1/v2 nearly
    zero when u_g2 is tiny, and records the block-1 HM2 equivalence to
    the c-weighted HM-evidence objective up to a per-block scale.
  - reports/approximation/new_approx_hmean_combinations_no_phi_online.txt
    gives the same harmonic-mean gradient and explicitly states that
    the no-phi objectives rank by normalized gain balance only.

The requested read-only file reports/approximation/new_approx_combined.txt
is not present in this checkout. I therefore rely on the local overview
summary of A2 and on summary/tail_conspiracy_insights_synthesis.txt for
the combined-score formula.


3. A stability objective for which HM3 is natural
-------------------------------------------------

Define the robust visible-window objective

  R(v) = min{u_sk(v), u_g1(v), u_g2(v)}

with the sketch term omitted at block 1. R asks for a direction whose
normalized captured energy is not starved in any visible source: carry,
current window, or future probe window.

HM3 is a smooth monotone proxy for R:

  min_i u_i <= HM_k(u_1,...,u_k) <= k * min_i u_i      for u_i > 0.

The left inequality follows because every u_i is at least the minimum,
so the reciprocal average is at most 1/min. The right inequality follows
because the reciprocal sum contains at least the reciprocal of the
minimum. Thus HM3 is not the minimum, and not a lower bound on the
minimum, but it is pinned to the minimum within a factor of k and becomes
small as soon as any term is small.

This gives a clean surrogate interpretation:

  maximize HM3(u_sk, u_g1, u_g2)

is the smooth reciprocal-barrier relaxation of

  maximize the worst normalized visible response.

The reciprocal form is also exactly the implemented gradient shape in
summary/f_hm3_score_implementation_context.txt §2 and
reports/approximation/new_approx_hmean_combinations_no_phi_online.txt:
small u_i terms receive large gradient weight through 1/u_i^2. That is
the intended behavior of a max-min stability proxy.


4. Assumptions needed for the surrogate to matter
--------------------------------------------------

The max-min statement above is algebraic and needs only positive u_i.
Interpreting it as useful for streaming SVD needs additional assumptions:

  H1. Comparable normalization. The denominators ||A_X||_F^2 make
      u_sk, u_g1, and u_g2 comparable fractions of available visible
      energy. This is the F-weighting assumption in S6.

  H2. Split-sample relevance. A direction that has stable normalized
      response on A_cur and A_fut is more likely to have stable response
      on later windows than a direction that only wins on one visible
      window. This is a sample-stability assumption, not an entropy
      invariance theorem.

  H3. Carry relevance. The rank-r carry response is a useful compressed
      summary of prior high-value directions. This is stronger than
      merely observing prior rows, because rank-r compression can erase
      information. It is partially motivated by the sk_F2_low scaling
      argument in summary/overview/score_design_overview.txt §2.

  H4. Positive-signal regime. The useful directions have nontrivial
      response in each visible source. If the true target is absent from
      A_fut, or has been lost by the rank-r carry, max-min stability can
      intentionally reject it.

Under H1-H4, HM3 is a natural objective for visible stability. These
assumptions do not make HM3 a proven lower bound for final
cos^2(V_carry, V_exact), and they do not prove that the optimizer will
pin the oracle direction. INFRA-05 already shows the opposite can happen
empirically: high HM3 level sets can be wide and displaced from oracle.


5. Comparison to the combined score's A2/phi story
--------------------------------------------------

Combined/phi theory:

  - Objective: observed gain times a phi multiplier derived from pooled
    response entropy / collision ratio.
  - Key theory assumption: A2, summarized in
    summary/overview/score_design_overview.txt §1ter as normalized h_2
    invariance from C_w to N_w to A.
  - Claim shape: if A2 holds, observed response entropy provides an
    algebraic correction / surrogate for unseen energy.

HM3/S6 theory from this note:

  - Objective: smooth proxy for maximizing the worst normalized response
    across carry/current/future visible samples.
  - Key assumptions: comparability of Frobenius-normalized responses,
    split-sample relevance, carry relevance, and nontrivial positive
    signal in all sources.
  - Claim shape: if those assumptions hold, HM3 is a reasonable smooth
    surrogate for visible stability. It does not extrapolate entropy to
    unseen rows and does not lower-bound population recovery.

Assumption comparison:

  - Weaker than A2:
      HM3 does not require normalized h_2(C_w v) to be invariant across
      C_w, N_w, and A. It does not require a row-count extrapolation
      formula for unseen energy. It also does not require phi-style
      within-window entropy to be the correct correction.

  - Stronger than A2:
      HM3 needs A_cur and A_fut to be relevant split samples for future
      behavior, and it needs the rank-r carry to remain a meaningful
      proxy after compression. A2 is an entropy-transfer assumption;
      HM3 adds explicit cross-window/carry comparability requirements.

  - Different from A2:
      A2/phi is a within-window entropy extrapolation story. HM3 is a
      between-window stability story. relH1, if added later, is the
      within-window stability check; it is not supplied by HM3 itself.


6. Negative boundary result
---------------------------

No A2-strength-only theorem can justify HM3 as a recovery objective.
Reason: A2 constrains an entropy statistic of observed responses. It
does not force u_sk, u_g1, and u_g2 to agree, does not require A_fut to
be representative of later windows, and says nothing about information
lost by the rank-r carry.

Concrete obstruction: take two candidate directions v and w with the
same normalized h_2 behavior on C_w, so A2 gives no entropy reason to
separate them. Let v have high observed/current response and low future
probe response, while w has slightly lower current response but balanced
responses across A_cur and A_fut. HM3 will prefer w because it optimizes
visible balance; a combined/phi argument can still prefer v if gain
dominates. Neither preference is theoremically "right" for final SVD
recovery unless we assume something about how A_fut and the carry
predict the future/full matrix.

Therefore HM3 cannot be promoted to a lower-bound or consistency story
using only assumptions comparable to A2. The strongest valid TH-01 claim
is the stability-surrogate claim above.


7. Verdict and propagation
--------------------------

Verdict for TH-01: iterate.

Closure status:

  - TH-01 is resolved as option (b): HM3 is the natural smooth proxy for
    a max-min visible stability objective.
  - It is not resolved as option (a): no lower-bound / population
    surrogate derivation is established.
  - The negative caveat from option (c) also applies to recovery claims:
    without assumptions stronger/different than A2, S6 must remain
    labeled as a heuristic inductive bias for subspace recovery.

Operational label to use elsewhere:

  S6/HM3 = visible split-sample stability surrogate, still heuristic for
  final subspace recovery.

Requested propagation outside TH-01 ownership:

  - TH-02 should decide when H2 is valid: when visible-window stability
    predicts population / future-window stability.
  - DIAG-01 should not be edited here, but its oracle entropy audit is
    still needed before regime labels become measured facts.
  - TH-03 should handle the separate scalar-score plateau limitation:
    even a valid stability surrogate can have high-score non-oracle
    ridges.

Diagnostic toolkit propagation:

  No toolkit propagation was needed. This note adds no diagnostic,
  invariant, measurement, or output folder, and it does not close
  diagnostic_toolkit.txt §8(m). It clarifies what diagnostics would be
  needed to go beyond the stability-surrogate claim.
