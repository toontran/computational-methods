"""r_sk-based slot-2 score: sketch-shortfall + raw-future HM.

For each candidate v in the union span at block k:

    raw_sk(v) = ||A_sketch v||^2          (A_sketch = state["s"] · state["V"]^T)
    raw_g1(v) = ||A_cur   v||^2
    raw_g2(v) = ||A_fut   v||^2
    r_sk(v)   = 1 - c_sk · raw_sk(v)      (c_sk = N_seen / ||A_sketch||_F^2)

Six variants (chosen via --variant):

    S1 (additive):
        score(v) = -max(r_sk, 0)^2 + alpha · HM(raw_g1, raw_g2)
    S2 (multiplicative gate):
        score(v) = (1 - max(r_sk, 0))^beta · HM(raw_g1, raw_g2)
    S3 (carry-aware):
        score(v) = -max(r_sk, 0)^2 · state_align(v)^gamma + alpha · HM(raw_g1, raw_g2)
    S4 (HM-of-three with relH1):
        score(v) = HM(r_sk, raw_g1, raw_g2) · relH1(A_cur v)
            with score := 0 (and grad := 0) whenever r_sk ≤ 0.
    S5 (sketch-strength gate, value-only — no analytic gradient yet):
        With T = 1/c_sk = ‖A_seen‖²_F / N_seen,
            r_sk_excess(v) = (raw_sk(v) − T) / T = c_sk · raw_sk − 1
        score(v) = HM3(r_sk_excess, raw_g1, raw_g2) · relH1(A_cur v)
                       when raw_sk ≥ T (i.e. v captures more sketch energy than a
                       typical past row),
        score(v) = 0  otherwise.
        Block 1 fall-through (no sketch): HM2(raw_g1, raw_g2) · relH1, same as S4.
    S6 (F-weighted HM3, no gate, no relH1):
        u_sk(v) = raw_sk(v) / sk_F2_low      sk_F2_low = ||A_sketch||_F^2 (rank-r carry)
        u_g1(v) = raw_g1(v) / cur_F2         cur_F2    = ||A_cur||_F^2
        u_g2(v) = raw_g2(v) / fut_F2         fut_F2    = ||A_fut||_F^2
        score(v) = HM3(u_sk, u_g1, u_g2)        when sketch present (block ≥ 2)
        score(v) = HM2(u_g1, u_g2)               when no sketch yet (block 1)
        Smooth on the sphere wherever all u's > 0; no clip, no relH1.
    S6_GM (F-weighted GM3, no gate, no relH1) — AGGREGATOR ABLATION (AB-01):
        Same u_sk/u_g1/u_g2 as S6.
        score(v) = (u_sk · u_g1 · u_g2)^(1/3)   when sketch present
        score(v) = (u_g1 · u_g2)^(1/2)          block-1 fall-through
        Like S6, GM3 zeros out when any u_X = 0 ("balance enforcer"), but
        penalizes imbalance more smoothly. Cleaner log-additive gradient:
            ∇log GM = (1/k) Σ ∇log u_X  ⇒  ∇GM = (GM/k) Σ (1/u_X)·∇u_X
        Hypothesis: smoother optimization landscape (narrower P4 plateau).
        See summary/score_family_aggregator_ablation/synthesis.md.
    D0 (S6 · relH1) — ROW-CONCENTRATION GUARD (FAM-02 D0):
        score(v) = S6(v) · relH1(A_cur v)
        Multiplicative row-concentration guard: penalizes v's whose A_cur v
        energy is dominated by a single row. Smooth on the sphere wherever
        S6 and relH1 are both positive (no S5-style hard gate). Product-rule
        gradient:  ∇D0 = relH1·∇S6 + S6·∇relH1.
        Hyperparameter: none.
        See summary/score_family_row_concentration_guard/variants/D0/spec.md.
    S6_OP (op-norm-weighted HM3, no gate, no relH1) — WEIGHTING ABLATION (AB-02):
        Same HM3 aggregator as S6, but the per-block UNIT-FIXER divides by
        sigma_max(A_X)^2 instead of ||A_X||_F^2:
            u_sk(v) = raw_sk / sk_op2_low      sk_op2_low = sigma_max(A_sketch)^2
            u_g1(v) = raw_g1 / cur_op2          cur_op2    = sigma_max(A_cur)^2
            u_g2(v) = raw_g2 / fut_op2          fut_op2    = sigma_max(A_fut)^2
        score(v) = HM3(u_sk, u_g1, u_g2)        when sketch present
        score(v) = HM2(u_g1, u_g2)              block-1 fall-through
        Now u_X(v) in [0, 1] EXACTLY (the leading singular direction
        achieves 1). Hypothesis: tighter [0, 1] cap may behave better on
        heavy-tailed spectra (e.g. static-cex) by tying the unit to the
        leading direction rather than aggregate energy.
        See summary/score_family_aggregator_ablation/S6_OP_synthesis.md.

where HM(a, b) = 2ab/(a + b), HM(a, b, c) = 3/(1/a + 1/b + 1/c),
state_align(v) = ||V_state^T v||^2 with V_state = state["V"], and
relH1(A_cur v) is the normalized Shannon entropy of the A_cur v energy
distribution (same as in `hmean_evidence_score`).

Drop-in policy for the existing `optimize_future_hmean_in_basis` scaffolding
in `future_hmean_optimizer_diagnostic.py`.

See summary/r_sk_score_implementation_context.txt for derivation.
"""

import argparse
import csv
import json
import os
import time

import numpy as np

import cex_restricted_space_probe as probe
import half_window_sliding_hmean_experiment as hm
from future_hmean_optimizer_diagnostic import (
    combined_score,
    optimize_future_hmean_in_basis,
    orth_basis_against,
    rowspace_basis,
)
from hmean_combinations_optimizer_diagnostic import (
    candidate_denoms,
    optimize_combination_in_basis,
)
from hmean_evidence_score import (
    entropy_relH1_value_grad,
    hm_evi_value_grad,
    per_block_constants,
    stream_to_block,
)
from row_cheat_baseline import (
    frame_score_S6 as _frame_score_S6,
    frame_score_S6_GM as _frame_score_S6_GM,
    oracle_frame_proj as _oracle_frame_proj,
    top_r_rows_frame as _top_r_rows_frame,
)
from second_slot_tail_bias_diagnostic import make_state, raw_oracle_columns
from subspace_metrics import principal_angles


# --------------------------------------------------------------------------
# Score / gradient
# --------------------------------------------------------------------------


def r_sk_g_value_grad(
    A_sketch, A_cur, A_fut, c_sk, v,
    variant="S1", alpha=1.0, beta=2.0, gamma=1.0, V_state=None,
    cur_F2=None, fut_F2=None, sk_F2_low=None,
    cur_op2=None, fut_op2=None, sk_op2_low=None,
):
    """Return (score, grad, r_sk, raw_g1, raw_g2, hm_g, sat_term, state_align).

    `sat_term` is the contribution of the sketch term to the score:
        S1: -max(r_sk, 0)^2
        S2: (1 - max(r_sk, 0))^beta   (the multiplicative gate)
        S3: -max(r_sk, 0)^2 · state_align(v)^gamma
        S6: u_sk = raw_sk / sk_F2_low (else 0 if no sketch)
        S6_OP: u_sk = raw_sk / sk_op2_low (else 0 if no sketch)

    For S6, cur_F2/fut_F2/sk_F2_low MUST be provided (else they are computed from
    the matrices, which is wasteful; pass them in from per_block_constants).
    For S6_OP, cur_op2/fut_op2/sk_op2_low MUST be provided (operator-norm-squared
    weightings; cache per block to avoid repeated SVDs).
    """
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    eps = 1e-30

    A_c = np.asarray(A_cur, dtype=np.float64)
    A_f = np.asarray(A_fut, dtype=np.float64)
    A_sk = None
    if A_sketch is not None:
        A_sk_arr = np.asarray(A_sketch, dtype=np.float64)
        if A_sk_arr.size:
            A_sk = A_sk_arr

    # Raw responses.
    if A_sk is not None and c_sk > 0.0:
        y_sk = A_sk @ v
        raw_sk = float(np.dot(y_sk, y_sk))
        r_sk = 1.0 - c_sk * raw_sk
    else:
        y_sk = None
        raw_sk = 0.0
        r_sk = 0.0  # No sketch yet → no shortfall, no penalty.

    y_c = A_c @ v
    raw_g1 = float(np.dot(y_c, y_c))
    y_f = A_f @ v
    raw_g2 = float(np.dot(y_f, y_f))

    # HM(raw_g1, raw_g2) = 2 a b / (a + b).
    denom_h = max(raw_g1 + raw_g2, eps)
    hm_g = 2.0 * raw_g1 * raw_g2 / denom_h
    grad_hm = np.zeros_like(v)
    if (raw_g1 + raw_g2) > eps:
        coeff_g1 = 4.0 * raw_g2 * raw_g2 / (denom_h * denom_h)  # 2 · 2b^2/(a+b)^2
        coeff_g2 = 4.0 * raw_g1 * raw_g1 / (denom_h * denom_h)  # 2 · 2a^2/(a+b)^2
        grad_hm = grad_hm + coeff_g1 * (A_c.T @ y_c)
        grad_hm = grad_hm + coeff_g2 * (A_f.T @ y_f)

    # state_align term (only used in S3).
    s_align = 1.0
    grad_s = np.zeros_like(v)
    if V_state is not None:
        Vs = np.asarray(V_state, dtype=np.float64)
        if Vs.size:
            if Vs.ndim == 1:
                Vs = Vs.reshape(-1, 1)
            # V_state is (n, r); columns are orthonormal in normal use.
            proj = Vs.T @ v
            s_align = float(np.dot(proj, proj))
            grad_s = 2.0 * (Vs @ proj)

    p = max(r_sk, 0.0)

    if variant == "S1":
        sat_term = -p * p
        score = sat_term + alpha * hm_g
        grad = alpha * grad_hm
        if r_sk > 0.0 and A_sk is not None:
            # d(-p^2)/dv = -2 r_sk · dr_sk/dv = -2 r_sk · (-2 c_sk A_sk^T y_sk)
            #            = +4 c_sk r_sk · A_sk^T y_sk
            grad = grad + 4.0 * c_sk * r_sk * (A_sk.T @ y_sk)
    elif variant == "S2":
        gate = (1.0 - p) ** beta  # for r_sk ≤ 0 this is 1.
        sat_term = gate
        score = gate * hm_g
        grad = gate * grad_hm
        if r_sk > 0.0 and A_sk is not None and beta > 0.0:
            # dgate/dv = -beta (1 - r_sk)^(beta - 1) · dr_sk/dv
            #          = -beta (1 - r_sk)^(beta - 1) · (-2 c_sk A_sk^T y_sk)
            #          = +2 beta c_sk (1 - r_sk)^(beta - 1) · A_sk^T y_sk
            base = max(1.0 - r_sk, 0.0)
            d_gate = 2.0 * beta * c_sk * (base ** (beta - 1.0)) * (A_sk.T @ y_sk)
            grad = grad + d_gate * hm_g
    elif variant == "S3":
        sa_pow = s_align ** gamma if s_align > eps or gamma >= 0 else 0.0
        sat_term = -p * p * sa_pow
        score = sat_term + alpha * hm_g
        grad = alpha * grad_hm
        if r_sk > 0.0:
            if A_sk is not None:
                # d(-p^2 · s^gamma)/dv |_{r_sk part} = -2 r_sk · dr_sk/dv · s^gamma
                #                                    = +4 c_sk r_sk · A_sk^T y_sk · s^gamma
                grad = grad + 4.0 * c_sk * r_sk * sa_pow * (A_sk.T @ y_sk)
            if gamma != 0.0 and s_align > eps:
                # d(-p^2 · s^gamma)/dv |_{s part} = -p^2 · gamma · s^{gamma - 1} · grad_s
                grad = grad - (r_sk * r_sk) * gamma * (s_align ** (gamma - 1.0)) * grad_s
    elif variant == "S4":
        # HM(r_sk, raw_g1, raw_g2) · relH1(A_cur v); score := 0 if r_sk ≤ 0.
        # Block 1 (no sketch yet): fall through to HM2(raw_g1, raw_g2) · relH1,
        # which is identical to the combined optimizer's slot-2 score there.
        relH1, grad_relH1 = entropy_relH1_value_grad(A_c, v)
        if A_sk is None:
            if raw_g1 <= eps or raw_g2 <= eps:
                sat_term = 0.0
                score = 0.0
                grad = np.zeros_like(v)
            else:
                D2 = raw_g1 + raw_g2
                HM2 = 2.0 * raw_g1 * raw_g2 / D2
                sat_term = HM2
                # d HM2 / dv = (HM2^2 / 2) · ((1/raw_g1^2)·draw_g1/dv + (1/raw_g2^2)·draw_g2/dv).
                coef2 = (HM2 * HM2) / 2.0
                grad_hm2 = (
                    coef2 * (1.0 / (raw_g1 * raw_g1)) * (2.0 * (A_c.T @ y_c))
                    + coef2 * (1.0 / (raw_g2 * raw_g2)) * (2.0 * (A_f.T @ y_f))
                )
                score = HM2 * relH1
                grad = relH1 * grad_hm2 + HM2 * grad_relH1
        elif r_sk <= 0.0 or raw_g1 <= eps or raw_g2 <= eps:
            sat_term = 0.0
            score = 0.0
            grad = np.zeros_like(v)
        else:
            D = 1.0 / r_sk + 1.0 / raw_g1 + 1.0 / raw_g2
            HM3 = 3.0 / D
            sat_term = HM3
            # d HM3 / dv = (HM3^2 / 3) · ( (1/r_sk^2)·dr_sk/dv
            #                              + (1/raw_g1^2)·draw_g1/dv
            #                              + (1/raw_g2^2)·draw_g2/dv )
            coef = (HM3 * HM3) / 3.0
            grad_hm3 = (
                coef * (1.0 / (r_sk * r_sk)) * (-2.0 * c_sk) * (A_sk.T @ y_sk)
                + coef * (1.0 / (raw_g1 * raw_g1)) * (2.0 * (A_c.T @ y_c))
                + coef * (1.0 / (raw_g2 * raw_g2)) * (2.0 * (A_f.T @ y_f))
            )
            score = HM3 * relH1
            grad = relH1 * grad_hm3 + HM3 * grad_relH1
    elif variant == "S5":
        # Sketch-strength gate. r_sk_excess = c_sk * raw_sk - 1 (positive when v
        # captures more sketch energy than a typical past row); score := 0 (and
        # grad := 0) below the gate.
        relH1, grad_relH1 = entropy_relH1_value_grad(A_c, v)
        if A_sk is None:
            # Block-1 fall-through: HM2 · relH1 (matches S4 block-1).
            if raw_g1 <= eps or raw_g2 <= eps:
                sat_term = 0.0
                score = 0.0
                grad = np.zeros_like(v)
            else:
                D2 = raw_g1 + raw_g2
                HM2 = 2.0 * raw_g1 * raw_g2 / D2
                sat_term = HM2
                coef2 = (HM2 * HM2) / 2.0
                grad_hm2 = (
                    coef2 * (1.0 / (raw_g1 * raw_g1)) * (2.0 * (A_c.T @ y_c))
                    + coef2 * (1.0 / (raw_g2 * raw_g2)) * (2.0 * (A_f.T @ y_f))
                )
                score = HM2 * relH1
                grad = relH1 * grad_hm2 + HM2 * grad_relH1
        else:
            r_sk_excess = c_sk * raw_sk - 1.0
            if r_sk_excess <= eps or raw_g1 <= eps or raw_g2 <= eps:
                sat_term = 0.0
                score = 0.0
                grad = np.zeros_like(v)
            else:
                D = 1.0 / r_sk_excess + 1.0 / raw_g1 + 1.0 / raw_g2
                HM3 = 3.0 / D
                sat_term = HM3
                # d r_sk_excess / dv = +2 c_sk · A_sk^T y_sk (sign-flipped vs S4).
                coef = (HM3 * HM3) / 3.0
                grad_hm3 = (
                    coef * (1.0 / (r_sk_excess * r_sk_excess)) * (+2.0 * c_sk) * (A_sk.T @ y_sk)
                    + coef * (1.0 / (raw_g1 * raw_g1)) * (2.0 * (A_c.T @ y_c))
                    + coef * (1.0 / (raw_g2 * raw_g2)) * (2.0 * (A_f.T @ y_f))
                )
                score = HM3 * relH1
                grad = relH1 * grad_hm3 + HM3 * grad_relH1
    elif variant == "S6":
        # F-weighted HM3 (no gate, no relH1).
        #   u_sk = raw_sk / sk_F2_low  (rank-r carry; tight cap)
        #   u_g1 = raw_g1 / cur_F2
        #   u_g2 = raw_g2 / fut_F2
        # Block-1 fall-through: HM2(u_g1, u_g2).
        if cur_F2 is None or fut_F2 is None:
            raise ValueError("S6 requires cur_F2 and fut_F2 kwargs")
        W_c = float(cur_F2)
        W_f = float(fut_F2)
        if W_c <= eps or W_f <= eps:
            sat_term = 0.0
            score = 0.0
            grad = np.zeros_like(v)
        else:
            u_g1 = raw_g1 / W_c
            u_g2 = raw_g2 / W_f
            have_sketch = (
                A_sk is not None
                and sk_F2_low is not None
                and float(sk_F2_low) > eps
            )
            if have_sketch:
                W_sk = float(sk_F2_low)
                u_sk = raw_sk / W_sk
                sat_term = float(u_sk)
                if u_sk <= eps or u_g1 <= eps or u_g2 <= eps:
                    score = 0.0
                    grad = np.zeros_like(v)
                else:
                    D = 1.0 / u_sk + 1.0 / u_g1 + 1.0 / u_g2
                    HM3 = 3.0 / D
                    score = float(HM3)
                    coef = (HM3 * HM3) / 3.0
                    # d HM3 / dv = coef * sum_X (1/u_X^2)·(1/W_X)·(2 A_X^T y_X)
                    grad = (
                        coef * (1.0 / (u_sk * u_sk)) * (1.0 / W_sk) * (2.0 * (A_sk.T @ y_sk))
                        + coef * (1.0 / (u_g1 * u_g1)) * (1.0 / W_c) * (2.0 * (A_c.T @ y_c))
                        + coef * (1.0 / (u_g2 * u_g2)) * (1.0 / W_f) * (2.0 * (A_f.T @ y_f))
                    )
            else:
                sat_term = 0.0  # no sketch contribution this block
                if u_g1 <= eps or u_g2 <= eps:
                    score = 0.0
                    grad = np.zeros_like(v)
                else:
                    D2 = 1.0 / u_g1 + 1.0 / u_g2
                    HM2 = 2.0 / D2
                    score = float(HM2)
                    coef2 = (HM2 * HM2) / 2.0
                    grad = (
                        coef2 * (1.0 / (u_g1 * u_g1)) * (1.0 / W_c) * (2.0 * (A_c.T @ y_c))
                        + coef2 * (1.0 / (u_g2 * u_g2)) * (1.0 / W_f) * (2.0 * (A_f.T @ y_f))
                    )
    elif variant == "S6_GM":
        # F-weighted GM3 (no gate, no relH1) — aggregator ablation (AB-01).
        #   u_sk = raw_sk / sk_F2_low, u_g1 = raw_g1 / cur_F2, u_g2 = raw_g2 / fut_F2
        # Block-1 fall-through: GM2(u_g1, u_g2) = sqrt(u_g1 * u_g2).
        if cur_F2 is None or fut_F2 is None:
            raise ValueError("S6_GM requires cur_F2 and fut_F2 kwargs")
        W_c = float(cur_F2)
        W_f = float(fut_F2)
        if W_c <= eps or W_f <= eps:
            sat_term = 0.0
            score = 0.0
            grad = np.zeros_like(v)
        else:
            u_g1 = raw_g1 / W_c
            u_g2 = raw_g2 / W_f
            have_sketch = (
                A_sk is not None
                and sk_F2_low is not None
                and float(sk_F2_low) > eps
            )
            if have_sketch:
                W_sk = float(sk_F2_low)
                u_sk = raw_sk / W_sk
                sat_term = float(u_sk)
                if u_sk <= eps or u_g1 <= eps or u_g2 <= eps:
                    score = 0.0
                    grad = np.zeros_like(v)
                else:
                    # GM3 = (u_sk * u_g1 * u_g2)^(1/3)
                    GM3 = float((u_sk * u_g1 * u_g2) ** (1.0 / 3.0))
                    score = GM3
                    # d GM3 / dv = (GM3 / 3) * sum_X (1/u_X) * du_X/dv
                    #            = (GM3 / 3) * sum_X (1/u_X) * (1/W_X) * (2 A_X^T y_X)
                    coef = GM3 / 3.0
                    grad = (
                        coef * (1.0 / u_sk) * (1.0 / W_sk) * (2.0 * (A_sk.T @ y_sk))
                        + coef * (1.0 / u_g1) * (1.0 / W_c) * (2.0 * (A_c.T @ y_c))
                        + coef * (1.0 / u_g2) * (1.0 / W_f) * (2.0 * (A_f.T @ y_f))
                    )
            else:
                sat_term = 0.0  # no sketch contribution this block
                if u_g1 <= eps or u_g2 <= eps:
                    score = 0.0
                    grad = np.zeros_like(v)
                else:
                    # GM2 = sqrt(u_g1 * u_g2)
                    GM2 = float((u_g1 * u_g2) ** 0.5)
                    score = GM2
                    # d GM2 / dv = (GM2 / 2) * sum_X (1/u_X) * (1/W_X) * (2 A_X^T y_X)
                    coef2 = GM2 / 2.0
                    grad = (
                        coef2 * (1.0 / u_g1) * (1.0 / W_c) * (2.0 * (A_c.T @ y_c))
                        + coef2 * (1.0 / u_g2) * (1.0 / W_f) * (2.0 * (A_f.T @ y_f))
                    )
    elif variant == "D0":
        # FAM-02 D0: S6(v) · relH1(A_cur v) — multiplicative row-concentration
        # guard. Smooth wherever the underlying S6 score is smooth (all u_X > 0)
        # and A_cur v has nonzero energy. Gradient by product rule:
        #     ∇D0 = relH1 · ∇S6 + S6 · ∇relH1.
        # Block-1 fall-through (no sketch): HM2(u_g1, u_g2) · relH1.
        if cur_F2 is None or fut_F2 is None:
            raise ValueError("D0 requires cur_F2 and fut_F2 kwargs")
        W_c = float(cur_F2)
        W_f = float(fut_F2)
        relH1_val, grad_relH1 = entropy_relH1_value_grad(A_c, v)
        if W_c <= eps or W_f <= eps:
            sat_term = 0.0
            score = 0.0
            grad = np.zeros_like(v)
        else:
            u_g1 = raw_g1 / W_c
            u_g2 = raw_g2 / W_f
            have_sketch = (
                A_sk is not None
                and sk_F2_low is not None
                and float(sk_F2_low) > eps
            )
            if have_sketch:
                W_sk = float(sk_F2_low)
                u_sk = raw_sk / W_sk
                sat_term = float(u_sk)
                if u_sk <= eps or u_g1 <= eps or u_g2 <= eps:
                    score = 0.0
                    grad = np.zeros_like(v)
                else:
                    D = 1.0 / u_sk + 1.0 / u_g1 + 1.0 / u_g2
                    HM3 = 3.0 / D
                    coef = (HM3 * HM3) / 3.0
                    grad_S6 = (
                        coef * (1.0 / (u_sk * u_sk)) * (1.0 / W_sk) * (2.0 * (A_sk.T @ y_sk))
                        + coef * (1.0 / (u_g1 * u_g1)) * (1.0 / W_c) * (2.0 * (A_c.T @ y_c))
                        + coef * (1.0 / (u_g2 * u_g2)) * (1.0 / W_f) * (2.0 * (A_f.T @ y_f))
                    )
                    score = float(HM3 * relH1_val)
                    grad = relH1_val * grad_S6 + HM3 * grad_relH1
            else:
                sat_term = 0.0  # no sketch contribution this block
                if u_g1 <= eps or u_g2 <= eps:
                    score = 0.0
                    grad = np.zeros_like(v)
                else:
                    D2 = 1.0 / u_g1 + 1.0 / u_g2
                    HM2 = 2.0 / D2
                    coef2 = (HM2 * HM2) / 2.0
                    grad_S6 = (
                        coef2 * (1.0 / (u_g1 * u_g1)) * (1.0 / W_c) * (2.0 * (A_c.T @ y_c))
                        + coef2 * (1.0 / (u_g2 * u_g2)) * (1.0 / W_f) * (2.0 * (A_f.T @ y_f))
                    )
                    score = float(HM2 * relH1_val)
                    grad = relH1_val * grad_S6 + HM2 * grad_relH1
    elif variant == "S6_OP":
        # Op-norm-weighted HM3 (no gate, no relH1) — weighting ablation (AB-02).
        #   u_sk = raw_sk / sk_op2_low,  u_g1 = raw_g1 / cur_op2,  u_g2 = raw_g2 / fut_op2
        # Block-1 fall-through: HM2(u_g1, u_g2). Same HM3 form as S6 but
        # the per-block constants are sigma_max(A_X)^2 instead of ||A_X||_F^2,
        # so u_X is bounded above by 1 exactly (achieved by the leading SV).
        if cur_op2 is None or fut_op2 is None:
            raise ValueError("S6_OP requires cur_op2 and fut_op2 kwargs")
        W_c = float(cur_op2)
        W_f = float(fut_op2)
        if W_c <= eps or W_f <= eps:
            sat_term = 0.0
            score = 0.0
            grad = np.zeros_like(v)
        else:
            u_g1 = raw_g1 / W_c
            u_g2 = raw_g2 / W_f
            have_sketch = (
                A_sk is not None
                and sk_op2_low is not None
                and float(sk_op2_low) > eps
            )
            if have_sketch:
                W_sk = float(sk_op2_low)
                u_sk = raw_sk / W_sk
                sat_term = float(u_sk)
                if u_sk <= eps or u_g1 <= eps or u_g2 <= eps:
                    score = 0.0
                    grad = np.zeros_like(v)
                else:
                    D = 1.0 / u_sk + 1.0 / u_g1 + 1.0 / u_g2
                    HM3 = 3.0 / D
                    score = float(HM3)
                    coef = (HM3 * HM3) / 3.0
                    # d HM3 / dv = coef * sum_X (1/u_X^2)·(1/W_X)·(2 A_X^T y_X)
                    grad = (
                        coef * (1.0 / (u_sk * u_sk)) * (1.0 / W_sk) * (2.0 * (A_sk.T @ y_sk))
                        + coef * (1.0 / (u_g1 * u_g1)) * (1.0 / W_c) * (2.0 * (A_c.T @ y_c))
                        + coef * (1.0 / (u_g2 * u_g2)) * (1.0 / W_f) * (2.0 * (A_f.T @ y_f))
                    )
            else:
                sat_term = 0.0  # no sketch contribution this block
                if u_g1 <= eps or u_g2 <= eps:
                    score = 0.0
                    grad = np.zeros_like(v)
                else:
                    D2 = 1.0 / u_g1 + 1.0 / u_g2
                    HM2 = 2.0 / D2
                    score = float(HM2)
                    coef2 = (HM2 * HM2) / 2.0
                    grad = (
                        coef2 * (1.0 / (u_g1 * u_g1)) * (1.0 / W_c) * (2.0 * (A_c.T @ y_c))
                        + coef2 * (1.0 / (u_g2 * u_g2)) * (1.0 / W_f) * (2.0 * (A_f.T @ y_f))
                    )
    else:
        raise ValueError(
            f"unknown variant {variant!r}; expected S1, S2, S3, S4, S5, S6, S6_GM, S6_OP, or D0"
        )

    return (
        float(score),
        np.ascontiguousarray(grad, dtype=np.float64),
        float(r_sk),
        float(raw_g1),
        float(raw_g2),
        float(hm_g),
        float(sat_term),
        float(s_align),
    )


# --------------------------------------------------------------------------
# Adapter into existing optimizer scaffold
# --------------------------------------------------------------------------


def make_r_sk_g_optimizer(
    A_cur, A_fut, A_sketch, c_sk,
    variant="S1", alpha=1.0, beta=2.0, gamma=1.0, V_state=None,
    cur_F2=None, fut_F2=None, sk_F2_low=None,
    cur_op2=None, fut_op2=None, sk_op2_low=None,
):
    def value_grad(_unused_cur, _unused_fut, v):
        del _unused_cur, _unused_fut
        score, grad, r_sk, raw_g1, raw_g2, hm_g, sat_term, s_align = r_sk_g_value_grad(
            A_sketch, A_cur, A_fut, c_sk, v,
            variant=variant, alpha=alpha, beta=beta, gamma=gamma, V_state=V_state,
            cur_F2=cur_F2, fut_F2=fut_F2, sk_F2_low=sk_F2_low,
            cur_op2=cur_op2, fut_op2=fut_op2, sk_op2_low=sk_op2_low,
        )
        # Optimizer's expected signature: (val, grad, gain1, gain2, relH1).
        # We repurpose the share slots to carry diagnostics: raw_g1, raw_g2, r_sk.
        return score, grad, raw_g1, raw_g2, r_sk

    return value_grad


def optimize_r_sk_g_in_basis(
    A_cur, A_fut, A_sketch, c_sk,
    B, starts, rng, maxit, tol, random_starts,
    variant="S1", alpha=1.0, beta=2.0, gamma=1.0, V_state=None,
    cur_F2=None, fut_F2=None, sk_F2_low=None,
    cur_op2=None, fut_op2=None, sk_op2_low=None,
):
    original = optimize_future_hmean_in_basis.__globals__["future_hmean_value_grad"]
    optimize_future_hmean_in_basis.__globals__["future_hmean_value_grad"] = make_r_sk_g_optimizer(
        A_cur, A_fut, A_sketch, c_sk,
        variant=variant, alpha=alpha, beta=beta, gamma=gamma, V_state=V_state,
        cur_F2=cur_F2, fut_F2=fut_F2, sk_F2_low=sk_F2_low,
        cur_op2=cur_op2, fut_op2=fut_op2, sk_op2_low=sk_op2_low,
    )
    try:
        return optimize_future_hmean_in_basis(
            A_cur, A_fut, B, starts, rng, maxit=maxit, tol=tol, random_starts=random_starts
        )
    finally:
        optimize_future_hmean_in_basis.__globals__["future_hmean_value_grad"] = original


# --------------------------------------------------------------------------
# Per-block diagnostic
# --------------------------------------------------------------------------


def _state_V(state):
    if state is None:
        return None
    V = state.get("V")
    if V is None:
        return None
    V = np.asarray(V, dtype=np.float64)
    return V if V.size else None


def _scores_for(v, A_sketch_for, A_cur, A_fut, c_sk, alpha, beta, gamma, V_state,
                c_g1, c_g2, w_evi_fixed, w_evi_c,
                cur_F2=None, fut_F2=None, sk_F2_low=None,
                cur_op2=None, fut_op2=None, sk_op2_low=None):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    nv = float(np.linalg.norm(v))
    if nv <= 1e-30:
        return None
    v = v / nv

    s1 = r_sk_g_value_grad(A_sketch_for, A_cur, A_fut, c_sk, v,
                           variant="S1", alpha=alpha)
    s2 = r_sk_g_value_grad(A_sketch_for, A_cur, A_fut, c_sk, v,
                           variant="S2", beta=beta)
    s3 = r_sk_g_value_grad(A_sketch_for, A_cur, A_fut, c_sk, v,
                           variant="S3", alpha=alpha, gamma=gamma, V_state=V_state)
    s4 = r_sk_g_value_grad(A_sketch_for, A_cur, A_fut, c_sk, v,
                           variant="S4")
    s5 = r_sk_g_value_grad(A_sketch_for, A_cur, A_fut, c_sk, v,
                           variant="S5")
    s6 = r_sk_g_value_grad(A_sketch_for, A_cur, A_fut, c_sk, v,
                           variant="S6",
                           cur_F2=cur_F2, fut_F2=fut_F2, sk_F2_low=sk_F2_low)
    s6gm = r_sk_g_value_grad(A_sketch_for, A_cur, A_fut, c_sk, v,
                             variant="S6_GM",
                             cur_F2=cur_F2, fut_F2=fut_F2, sk_F2_low=sk_F2_low)
    sd0 = r_sk_g_value_grad(A_sketch_for, A_cur, A_fut, c_sk, v,
                            variant="D0",
                            cur_F2=cur_F2, fut_F2=fut_F2, sk_F2_low=sk_F2_low)
    s6op = r_sk_g_value_grad(A_sketch_for, A_cur, A_fut, c_sk, v,
                             variant="S6_OP",
                             cur_op2=cur_op2, fut_op2=fut_op2, sk_op2_low=sk_op2_low)

    # Existing HM-evi (fixed) and HM-evi (c-weights).
    score_evi_fixed, *_ = hm_evi_value_grad(
        A_sketch_for, A_cur, A_fut, c_sk, c_g1, c_g2, *w_evi_fixed, v
    )
    score_evi_c, *_ = hm_evi_value_grad(
        A_sketch_for, A_cur, A_fut, c_sk, c_g1, c_g2, *w_evi_c, v
    )
    return {
        "v": v,
        "S1": s1, "S2": s2, "S3": s3, "S4": s4, "S5": s5, "S6": s6,
        "S6_GM": s6gm,
        "D0": sd0,
        "S6_OP": s6op,
        "score_evi_fixed": float(score_evi_fixed),
        "score_evi_c": float(score_evi_c),
    }


def analyze_block(args, matrix, A, V_exact, snap, block_id):
    rank = int(args.rank)
    half_win = int(args.half_win)
    A_cur = snap["A_cur"]
    A_fut = snap["A_fut"]
    A_sketch = snap["A_sketch"]
    M_gain = snap["M_gain"]
    state = snap["state"]
    old_row_memory = snap["old_row_memory"]
    V_default = snap["V_default"]
    diag = snap["diag"]

    consts = per_block_constants(A, block_id, half_win)
    c_sk = consts["c_sk"]
    c_g1 = consts["c_g1"]
    c_g2 = consts["c_g2"]
    cur_F2 = float(consts["cur_F2"])
    fut_F2 = float(consts["fut_F2"])
    A_sketch_for = A_sketch if A_sketch.size else None
    V_state = _state_V(state)

    # Rank-r CARRY Frobenius (state.s · state.V^T): differs from sk_F2 in
    # consts (which is the FULL prefix Frobenius). S6 uses sk_F2_low.
    if A_sketch_for is not None:
        sk_F2_low = float(np.sum(np.asarray(A_sketch_for, dtype=np.float64) ** 2))
    else:
        sk_F2_low = 0.0

    # Operator-norm-squared per-block constants for S6_OP (AB-02).
    # Cached once per (block, matrix) since each is a single SVD.
    # A_sketch is rank-r (= state.s · state.V^T) so its top SV is just state.s[0]
    # (state.s sorted descending by streaming SVD invariant).
    cur_op2 = float(np.linalg.svd(np.asarray(A_cur, dtype=np.float64), compute_uv=False)[0] ** 2) \
        if A_cur.size else 0.0
    fut_op2 = float(np.linalg.svd(np.asarray(A_fut, dtype=np.float64), compute_uv=False)[0] ** 2) \
        if A_fut.size else 0.0
    if A_sketch_for is not None:
        if state is not None and state.get("s") is not None and np.asarray(state["s"]).size:
            sk_op2_low = float(np.asarray(state["s"], dtype=np.float64)[0] ** 2)
        else:
            sk_op2_low = float(np.linalg.svd(np.asarray(A_sketch_for, dtype=np.float64),
                                             compute_uv=False)[0] ** 2)
    else:
        sk_op2_low = 0.0

    # Carried sketch right singular vectors (state.V) — the directions that
    # maximise raw_sk = ‖B_top v‖² on the sphere. Used both as candidates
    # below and as warm-starts for the S5 sketch-init optimizer variant.
    sketch_v1 = V_state[:, 0] if V_state is not None and V_state.shape[1] >= 1 else None
    sketch_v2 = V_state[:, 1] if V_state is not None and V_state.shape[1] >= 2 else None

    alpha = float(args.alpha)
    beta = float(args.beta)
    gamma = float(args.gamma)

    # Subspace bases for projected oracle candidates.
    if A_sketch_for is not None:
        union_stack = np.vstack([A_sketch, A_cur, A_fut])
    else:
        union_stack = np.vstack([A_cur, A_fut])
    B_union = rowspace_basis(union_stack)

    def project_unit(vec, B):
        if vec is None or B is None or B.size == 0:
            return None
        p = B @ (B.T @ vec)
        nv = float(np.linalg.norm(p))
        return None if nv <= 1e-30 else p / nv

    oracle_v1 = V_exact[:, 0] / max(np.linalg.norm(V_exact[:, 0]), 1e-30)
    oracle_v2 = V_exact[:, 1] / max(np.linalg.norm(V_exact[:, 1]), 1e-30)
    oracle_v1_proj = project_unit(oracle_v1, B_union)
    oracle_v2_proj = project_unit(oracle_v2, B_union)

    # Existing HM-triplet (raw) for comparison.
    Q_oracle, raw_oracle = raw_oracle_columns(M_gain, V_exact, rank, np.float64)
    pool = hm.build_candidates(V_default, Q_oracle, raw_oracle, M_gain, A_cur, A_fut)
    pool = {k: pool.get(k) for k in hm.ONLINE_POOL}
    weights_existing = (state["rows_seen"] if state is not None else 0,
                        A_cur.shape[0], A_fut.shape[0])
    denoms, _ = candidate_denoms(pool, A_cur, A_fut, A_sketch_for)
    if A_sketch_for is not None:
        union_for_search = np.vstack([A_sketch, A_cur, A_fut]).astype(np.float64, copy=False)
    else:
        union_for_search = np.vstack([A_cur, A_fut]).astype(np.float64, copy=False)
    B_search = orth_basis_against(rowspace_basis(union_for_search), V_default[:, 0])

    starts = [V_default[:, 1]]
    starts.extend([v for v in pool.values() if v is not None])
    Vbasis = diag.get("Vbasis_final")
    if Vbasis is not None:
        Vb = np.asarray(Vbasis, dtype=np.float64)
        for j in range(min(Vb.shape[1], 8)):
            starts.append(Vb[:, j])

    # hm_triplet_raw_best (the optimizer that wins under any HM).
    denoms_raw = {k: 1.0 for k in ("sketch", "gain1", "gain2", "sketch_gain1", "sketch_gain2", "sketch_raw_for_concat")}
    starts_raw = list(starts) + ([oracle_v1_proj, oracle_v2_proj] if oracle_v1_proj is not None else [])
    triplet_raw = optimize_combination_in_basis(
        "future_hmean_triplet_online",
        A_cur, A_fut, A_sketch_for, denoms_raw, weights_existing,
        B_search, starts_raw,
        np.random.default_rng(args.seed + 9001 + block_id),
        args.union_maxit, args.union_tol, args.union_random_starts,
    )

    # Warm-starts available in streaming (no oracle access).
    starts_streaming = list(starts)
    # Diagnostic-only: warm-starts that include the oracle projections (best-case).
    starts_diag = list(starts) + ([oracle_v1_proj, oracle_v2_proj] if oracle_v1_proj is not None else [])
    if getattr(args, "no_oracle_warmstart", False):
        starts_new = starts_streaming
    else:
        starts_new = starts_diag

    rsk_best = {}
    # Per-variant seed offsets keep the random restarts deterministic but
    # distinct across variants. ord(variant[1]) maps S1..S6→{49..54}; the
    # GM ablation uses an offset that does not collide with any of those.
    variant_seed_off = {
        "S1": ord("1") * 13, "S2": ord("2") * 13, "S3": ord("3") * 13,
        "S4": ord("4") * 13, "S5": ord("5") * 13, "S6": ord("6") * 13,
        "S6_GM": ord("7") * 13,  # 7 is the next free integer slot
        "D0": ord("8") * 13,     # FAM-02 D0
        "S6_OP": ord("9") * 13,  # AB-02 weighting ablation
    }
    for variant in ("S1", "S2", "S3", "S4", "S5", "S6", "S6_GM", "D0", "S6_OP"):
        rsk_best[variant] = optimize_r_sk_g_in_basis(
            A_cur, A_fut, A_sketch_for, c_sk,
            B_search, starts_new,
            np.random.default_rng(args.seed + 41000 + block_id + variant_seed_off[variant]),
            args.union_maxit, args.union_tol, args.union_random_starts,
            variant=variant, alpha=alpha, beta=beta, gamma=gamma, V_state=V_state,
            cur_F2=cur_F2, fut_F2=fut_F2, sk_F2_low=sk_F2_low,
            cur_op2=cur_op2, fut_op2=fut_op2, sk_op2_low=sk_op2_low,
        )

    # Rank-2 sequential HM3 (S4) optimization in the FULL union span.
    # First find v1 (unconstrained over union), then deflate and find v2.
    # Matches streaming-init constraints when --no-oracle-warmstart is set.
    starts_full = starts_new
    rsk_S4_v1 = optimize_r_sk_g_in_basis(
        A_cur, A_fut, A_sketch_for, c_sk,
        B_union, starts_full,
        np.random.default_rng(args.seed + 51000 + block_id),
        args.union_maxit, args.union_tol, args.union_random_starts,
        variant="S4", alpha=alpha, beta=beta, gamma=gamma, V_state=V_state,
    )
    v1_hm3 = None if rsk_S4_v1 is None else rsk_S4_v1["vec"]
    if v1_hm3 is not None:
        B_deflated = orth_basis_against(B_union, v1_hm3)
        rsk_S4_v2 = optimize_r_sk_g_in_basis(
            A_cur, A_fut, A_sketch_for, c_sk,
            B_deflated, starts_full,
            np.random.default_rng(args.seed + 52000 + block_id),
            args.union_maxit, args.union_tol, args.union_random_starts,
            variant="S4", alpha=alpha, beta=beta, gamma=gamma, V_state=V_state,
        )
        v2_hm3 = None if rsk_S4_v2 is None else rsk_S4_v2["vec"]
    else:
        v2_hm3 = None

    # Same flow for S5 (sketch-strength gate).
    rsk_S5_v1 = optimize_r_sk_g_in_basis(
        A_cur, A_fut, A_sketch_for, c_sk,
        B_union, starts_full,
        np.random.default_rng(args.seed + 53000 + block_id),
        args.union_maxit, args.union_tol, args.union_random_starts,
        variant="S5", alpha=alpha, beta=beta, gamma=gamma, V_state=V_state,
    )
    v1_S5 = None if rsk_S5_v1 is None else rsk_S5_v1["vec"]
    if v1_S5 is not None:
        B_def_S5 = orth_basis_against(B_union, v1_S5)
        rsk_S5_v2 = optimize_r_sk_g_in_basis(
            A_cur, A_fut, A_sketch_for, c_sk,
            B_def_S5, starts_full,
            np.random.default_rng(args.seed + 54000 + block_id),
            args.union_maxit, args.union_tol, args.union_random_starts,
            variant="S5", alpha=alpha, beta=beta, gamma=gamma, V_state=V_state,
        )
        v2_S5 = None if rsk_S5_v2 is None else rsk_S5_v2["vec"]
    else:
        v2_S5 = None

    # S6 sequential rank-2 over B_union (no V_default[:,0] deflation).
    rsk_S6_v1 = optimize_r_sk_g_in_basis(
        A_cur, A_fut, A_sketch_for, c_sk,
        B_union, starts_full,
        np.random.default_rng(args.seed + 57000 + block_id),
        args.union_maxit, args.union_tol, args.union_random_starts,
        variant="S6", alpha=alpha, beta=beta, gamma=gamma, V_state=V_state,
        cur_F2=cur_F2, fut_F2=fut_F2, sk_F2_low=sk_F2_low,
    )
    v1_S6 = None if rsk_S6_v1 is None else rsk_S6_v1["vec"]
    if v1_S6 is not None:
        B_def_S6 = orth_basis_against(B_union, v1_S6)
        rsk_S6_v2 = optimize_r_sk_g_in_basis(
            A_cur, A_fut, A_sketch_for, c_sk,
            B_def_S6, starts_full,
            np.random.default_rng(args.seed + 58000 + block_id),
            args.union_maxit, args.union_tol, args.union_random_starts,
            variant="S6", alpha=alpha, beta=beta, gamma=gamma, V_state=V_state,
            cur_F2=cur_F2, fut_F2=fut_F2, sk_F2_low=sk_F2_low,
        )
        v2_S6 = None if rsk_S6_v2 is None else rsk_S6_v2["vec"]
    else:
        v2_S6 = None

    # S6_GM sequential rank-2 over B_union (aggregator ablation, AB-01).
    rsk_S6GM_v1 = optimize_r_sk_g_in_basis(
        A_cur, A_fut, A_sketch_for, c_sk,
        B_union, starts_full,
        np.random.default_rng(args.seed + 59000 + block_id),
        args.union_maxit, args.union_tol, args.union_random_starts,
        variant="S6_GM", alpha=alpha, beta=beta, gamma=gamma, V_state=V_state,
        cur_F2=cur_F2, fut_F2=fut_F2, sk_F2_low=sk_F2_low,
    )
    v1_S6GM = None if rsk_S6GM_v1 is None else rsk_S6GM_v1["vec"]
    if v1_S6GM is not None:
        B_def_S6GM = orth_basis_against(B_union, v1_S6GM)
        rsk_S6GM_v2 = optimize_r_sk_g_in_basis(
            A_cur, A_fut, A_sketch_for, c_sk,
            B_def_S6GM, starts_full,
            np.random.default_rng(args.seed + 60000 + block_id),
            args.union_maxit, args.union_tol, args.union_random_starts,
            variant="S6_GM", alpha=alpha, beta=beta, gamma=gamma, V_state=V_state,
            cur_F2=cur_F2, fut_F2=fut_F2, sk_F2_low=sk_F2_low,
        )
        v2_S6GM = None if rsk_S6GM_v2 is None else rsk_S6GM_v2["vec"]
    else:
        v2_S6GM = None

    # D0 sequential rank-2 over B_union (FAM-02 row-concentration guard).
    rsk_D0_v1 = optimize_r_sk_g_in_basis(
        A_cur, A_fut, A_sketch_for, c_sk,
        B_union, starts_full,
        np.random.default_rng(args.seed + 61000 + block_id),
        args.union_maxit, args.union_tol, args.union_random_starts,
        variant="D0", alpha=alpha, beta=beta, gamma=gamma, V_state=V_state,
        cur_F2=cur_F2, fut_F2=fut_F2, sk_F2_low=sk_F2_low,
    )
    v1_D0 = None if rsk_D0_v1 is None else rsk_D0_v1["vec"]
    if v1_D0 is not None:
        B_def_D0 = orth_basis_against(B_union, v1_D0)
        rsk_D0_v2 = optimize_r_sk_g_in_basis(
            A_cur, A_fut, A_sketch_for, c_sk,
            B_def_D0, starts_full,
            np.random.default_rng(args.seed + 62000 + block_id),
            args.union_maxit, args.union_tol, args.union_random_starts,
            variant="D0", alpha=alpha, beta=beta, gamma=gamma, V_state=V_state,
            cur_F2=cur_F2, fut_F2=fut_F2, sk_F2_low=sk_F2_low,
        )
        v2_D0 = None if rsk_D0_v2 is None else rsk_D0_v2["vec"]
    else:
        v2_D0 = None

    # S6_OP sequential rank-2 over B_union (weighting ablation, AB-02).
    rsk_S6OP_v1 = optimize_r_sk_g_in_basis(
        A_cur, A_fut, A_sketch_for, c_sk,
        B_union, starts_full,
        np.random.default_rng(args.seed + 63000 + block_id),
        args.union_maxit, args.union_tol, args.union_random_starts,
        variant="S6_OP", alpha=alpha, beta=beta, gamma=gamma, V_state=V_state,
        cur_op2=cur_op2, fut_op2=fut_op2, sk_op2_low=sk_op2_low,
    )
    v1_S6OP = None if rsk_S6OP_v1 is None else rsk_S6OP_v1["vec"]
    if v1_S6OP is not None:
        B_def_S6OP = orth_basis_against(B_union, v1_S6OP)
        rsk_S6OP_v2 = optimize_r_sk_g_in_basis(
            A_cur, A_fut, A_sketch_for, c_sk,
            B_def_S6OP, starts_full,
            np.random.default_rng(args.seed + 64000 + block_id),
            args.union_maxit, args.union_tol, args.union_random_starts,
            variant="S6_OP", alpha=alpha, beta=beta, gamma=gamma, V_state=V_state,
            cur_op2=cur_op2, fut_op2=fut_op2, sk_op2_low=sk_op2_low,
        )
        v2_S6OP = None if rsk_S6OP_v2 is None else rsk_S6OP_v2["vec"]
    else:
        v2_S6OP = None

    # S5 sketch-init: warm-start ONLY from carried state.V columns (no random
    # restarts). Tests whether the carry direction's basin reaches a useful
    # local max under S5.
    if sketch_v1 is not None:
        sketch_starts = [sketch_v1]
        if sketch_v2 is not None:
            sketch_starts.append(sketch_v2)
        rsk_S5_sk1 = optimize_r_sk_g_in_basis(
            A_cur, A_fut, A_sketch_for, c_sk,
            B_union, sketch_starts,
            np.random.default_rng(args.seed + 55000 + block_id),
            args.union_maxit, args.union_tol, 0,
            variant="S5", alpha=alpha, beta=beta, gamma=gamma, V_state=V_state,
        )
        v1_S5_sk = None if rsk_S5_sk1 is None else rsk_S5_sk1["vec"]
        if v1_S5_sk is not None:
            B_def_S5_sk = orth_basis_against(B_union, v1_S5_sk)
            rsk_S5_sk2 = optimize_r_sk_g_in_basis(
                A_cur, A_fut, A_sketch_for, c_sk,
                B_def_S5_sk, sketch_starts,
                np.random.default_rng(args.seed + 56000 + block_id),
                args.union_maxit, args.union_tol, 0,
                variant="S5", alpha=alpha, beta=beta, gamma=gamma, V_state=V_state,
            )
            v2_S5_sk = None if rsk_S5_sk2 is None else rsk_S5_sk2["vec"]
        else:
            v2_S5_sk = None
    else:
        v1_S5_sk = None
        v2_S5_sk = None

    # Top-2 right singular vectors of M_gain = [B_top; A_cur] — the directions
    # the rank-2 SVD of (sketch + current block) would prefer.
    M_gain_arr = np.asarray(M_gain, dtype=np.float64)
    if M_gain_arr.size:
        _, _, Vt_mgain = np.linalg.svd(M_gain_arr, full_matrices=False)
        mgain_svd_v1 = Vt_mgain[0] if Vt_mgain.shape[0] >= 1 else None
        mgain_svd_v2 = Vt_mgain[1] if Vt_mgain.shape[0] >= 2 else None
    else:
        mgain_svd_v1 = None
        mgain_svd_v2 = None

    # Rank-r row-cheat frame (INFRA-03): top-r rows of A_fut by squared norm,
    # stacked as columns and orthonormalized via QR. Generalization of the
    # slot-2 hm_triplet_raw_best lower bound for value-only score
    # exploitability. Per-vector columns are also injected into the candidate
    # panel so the per-block table reflects the per-direction shares; the
    # frame-level score is computed below alongside the oracle-frame score.
    V_rowcheat_frame = _top_r_rows_frame(A_fut, rank)
    rowcheat_v1 = V_rowcheat_frame[:, 0] if V_rowcheat_frame is not None and V_rowcheat_frame.shape[1] >= 1 else None
    rowcheat_v2 = V_rowcheat_frame[:, 1] if V_rowcheat_frame is not None and V_rowcheat_frame.shape[1] >= 2 else None
    V_oracle_frame_proj = _oracle_frame_proj(V_exact, B_union, rank)

    candidates = {
        "combined_v1": V_default[:, 0],
        "combined_v2": V_default[:, 1],
        "sketch_v1": sketch_v1,
        "sketch_v2": sketch_v2,
        "mgain_svd_v1": mgain_svd_v1,
        "mgain_svd_v2": mgain_svd_v2,
        "hm_triplet_raw_best": None if triplet_raw is None else triplet_raw["vec"],
        "rowcheat_v1": rowcheat_v1,
        "rowcheat_v2": rowcheat_v2,
        "r_sk_g_S1_best": None if rsk_best["S1"] is None else rsk_best["S1"]["vec"],
        "r_sk_g_S2_best": None if rsk_best["S2"] is None else rsk_best["S2"]["vec"],
        "r_sk_g_S3_best": None if rsk_best["S3"] is None else rsk_best["S3"]["vec"],
        "r_sk_g_S4_best": None if rsk_best["S4"] is None else rsk_best["S4"]["vec"],
        "r_sk_g_S4_v1_full": v1_hm3,
        "r_sk_g_S4_v2_deflate": v2_hm3,
        "r_sk_g_S5_best": None if rsk_best["S5"] is None else rsk_best["S5"]["vec"],
        "r_sk_g_S5_v1_full": v1_S5,
        "r_sk_g_S5_v2_deflate": v2_S5,
        "r_sk_g_S5_sketch_init_v1": v1_S5_sk,
        "r_sk_g_S5_sketch_init_v2": v2_S5_sk,
        "r_sk_g_S6_best": None if rsk_best["S6"] is None else rsk_best["S6"]["vec"],
        "r_sk_g_S6_v1_full": v1_S6,
        "r_sk_g_S6_v2_deflate": v2_S6,
        "r_sk_g_S6_GM_best": None if rsk_best["S6_GM"] is None else rsk_best["S6_GM"]["vec"],
        "r_sk_g_S6_GM_v1_full": v1_S6GM,
        "r_sk_g_S6_GM_v2_deflate": v2_S6GM,
        "r_sk_g_D0_best": None if rsk_best["D0"] is None else rsk_best["D0"]["vec"],
        "r_sk_g_D0_v1_full": v1_D0,
        "r_sk_g_D0_v2_deflate": v2_D0,
        "r_sk_g_S6_OP_best": None if rsk_best["S6_OP"] is None else rsk_best["S6_OP"]["vec"],
        "r_sk_g_S6_OP_v1_full": v1_S6OP,
        "r_sk_g_S6_OP_v2_deflate": v2_S6OP,
        "oracle_v1_proj": oracle_v1_proj,
        "oracle_v2_proj": oracle_v2_proj,
    }

    # HM-evi weights for cross-comparison.
    w_evi_fixed = (float(rank), float(half_win), float(half_win))
    w_evi_c = (float(c_sk * c_sk), float(c_g1 * c_g1), float(c_g2 * c_g2))

    rows = []
    for label, v in candidates.items():
        if v is None:
            continue
        info = _scores_for(v, A_sketch_for, A_cur, A_fut, c_sk,
                           alpha, beta, gamma, V_state,
                           c_g1, c_g2, w_evi_fixed, w_evi_c,
                           cur_F2=cur_F2, fut_F2=fut_F2, sk_F2_low=sk_F2_low,
                           cur_op2=cur_op2, fut_op2=fut_op2, sk_op2_low=sk_op2_low)
        if info is None:
            continue
        v = info["v"]
        s1 = info["S1"]; s2 = info["S2"]; s3 = info["S3"]; s4 = info["S4"]; s5 = info["S5"]; s6 = info["S6"]
        s6gm = info["S6_GM"]
        sd0 = info["D0"]
        s6op = info["S6_OP"]
        from hmean_evidence_score import entropy_relH1_value_grad as _entropy_value_grad
        relH1, _ = _entropy_value_grad(A_cur, v)
        comb = combined_score(M_gain, A_cur, v, A.shape[0], state, old_row_memory)
        align_v1 = float(np.dot(v, oracle_v1) ** 2)
        align_v2 = float(np.dot(v, oracle_v2) ** 2)
        align_v1_proj = (
            float(np.dot(v, oracle_v1_proj) ** 2) if oracle_v1_proj is not None else float("nan")
        )
        align_v2_proj = (
            float(np.dot(v, oracle_v2_proj) ** 2) if oracle_v2_proj is not None else float("nan")
        )
        # Principal-angle view (INFRA-01, summary/overview/diagnostic_toolkit.txt §1).
        # pa1_o2 = cos² of the single principal angle between span(v) and the
        # 2-dim oracle plane span([oracle_v1_proj, oracle_v2_proj]); equals
        # ||V_oracle^T v||² = align_v1_proj + align_v2_proj for orthonormal
        # V_oracle. Dual to the pairwise align columns: how much of v lies in
        # the oracle plane regardless of slot assignment. Computed via SVD of
        # V_opt^T V_oracle in subspace_metrics.principal_angles.
        if oracle_v1_proj is not None and oracle_v2_proj is not None:
            V_oracle_2 = np.column_stack([oracle_v1_proj, oracle_v2_proj])
            pa_cos2, _ = principal_angles(v, V_oracle_2)
            pa1_o2 = float(pa_cos2[0]) if pa_cos2.size else float("nan")
        else:
            pa1_o2 = float("nan")
        rows.append({
            "matrix": matrix, "block": block_id, "label": label,
            "r_sk": s1[2],
            "raw_g1": s1[3],
            "raw_g2": s1[4],
            "hm_g": s1[5],
            "hm3": s4[6],
            "hm3_user": s5[6],
            "relH1": float(relH1),
            "sat_S1": s1[6],
            "sat_S2": s2[6],
            "sat_S3": s3[6],
            "state_align": s3[7],
            "score_S1": s1[0],
            "score_S2": s2[0],
            "score_S3": s3[0],
            "score_S4": s4[0],
            "score_S5": s5[0],
            "score_S6": s6[0],
            "sat_S6": s6[6],
            "score_S6_GM": s6gm[0],
            "sat_S6_GM": s6gm[6],
            "score_D0": sd0[0],
            "sat_D0": sd0[6],
            "score_S6_OP": s6op[0],
            "sat_S6_OP": s6op[6],
            "score_evi_fixed": info["score_evi_fixed"],
            "score_evi_c": info["score_evi_c"],
            "combined_score": comb,
            "align_v1": align_v1,
            "align_v2": align_v2,
            "align_v1_proj": align_v1_proj,
            "align_v2_proj": align_v2_proj,
            "pa1_o2": pa1_o2,
        })

    # Per-block frame-level subspace alignment (INFRA-01). For each rank-2
    # candidate frame [v1, v2], report the two principal-angle cos² versus
    # the oracle frame [oracle_v1_proj, oracle_v2_proj]. cos2[0] is the
    # most-aligned direction (smallest principal angle); cos2[1] the
    # least-aligned. (cos2[0]+cos2[1])/2 is the Grassmann-style mean
    # alignment used in the rank-r plateau detector (§5 score_design_overview).
    subspace_align = {}
    if oracle_v1_proj is not None and oracle_v2_proj is not None:
        V_oracle_2 = np.column_stack([oracle_v1_proj, oracle_v2_proj])
        frames = {
            "combined": (V_default[:, 0], V_default[:, 1]),
            "sketch":   (sketch_v1, sketch_v2),
            "mgain_svd": (mgain_svd_v1, mgain_svd_v2),
            "r_sk_g_S4": (v1_hm3, v2_hm3),
            "r_sk_g_S5": (v1_S5, v2_S5),
            "r_sk_g_S5_sketch_init": (v1_S5_sk, v2_S5_sk),
            "r_sk_g_S6": (v1_S6, v2_S6),
            "r_sk_g_S6_GM": (v1_S6GM, v2_S6GM),
            "r_sk_g_D0": (v1_D0, v2_D0),
            "r_sk_g_S6_OP": (v1_S6OP, v2_S6OP),
            "rowcheat":   (rowcheat_v1, rowcheat_v2),
        }
        for name, (a, b) in frames.items():
            if a is None or b is None:
                continue
            V_frame = np.column_stack([a, b])
            cos2, _ = principal_angles(V_frame, V_oracle_2)
            if cos2.size == 2:
                subspace_align[name] = (float(cos2[0]), float(cos2[1]))

    # Rank-r row-cheat exploitability check (INFRA-03). Score the row-cheat
    # frame and the projected oracle frame under the rank-r lifts of S6 and
    # S6_GM and record both. The acceptance criterion (printed by write_text)
    # is score(V_oracle_frame) >= score(V_rowcheat_frame). When this fails on
    # any block, the score is row-exploitable at rank r.
    rowcheat_summary = {}
    if V_rowcheat_frame is not None:
        rowcheat_summary["frame_S6"] = _frame_score_S6(
            A_sketch_for, A_cur, A_fut, sk_F2_low, cur_F2, fut_F2, V_rowcheat_frame
        )
        rowcheat_summary["frame_S6_GM"] = _frame_score_S6_GM(
            A_sketch_for, A_cur, A_fut, sk_F2_low, cur_F2, fut_F2, V_rowcheat_frame
        )
        rowcheat_summary["rowcheat_dim"] = int(V_rowcheat_frame.shape[1])
    if V_oracle_frame_proj is not None:
        rowcheat_summary["frame_S6_oracle"] = _frame_score_S6(
            A_sketch_for, A_cur, A_fut, sk_F2_low, cur_F2, fut_F2, V_oracle_frame_proj
        )
        rowcheat_summary["frame_S6_GM_oracle"] = _frame_score_S6_GM(
            A_sketch_for, A_cur, A_fut, sk_F2_low, cur_F2, fut_F2, V_oracle_frame_proj
        )
        rowcheat_summary["oracle_dim"] = int(V_oracle_frame_proj.shape[1])

    info = {
        "matrix": matrix, "block_id": block_id, "half_win": half_win, "rank": rank,
        "N_sk": consts["N_sk"], "sk_F2": consts["sk_F2"], "sk_F2_low": sk_F2_low,
        "cur_F2": cur_F2, "fut_F2": fut_F2,
        "cur_op2": cur_op2, "fut_op2": fut_op2, "sk_op2_low": sk_op2_low,
        "c_sk": c_sk, "c_g1": c_g1, "c_g2": c_g2,
        "alpha": alpha, "beta": beta, "gamma": gamma,
        "union_dim": int(B_union.shape[1]),
        "state_rank": int(V_state.shape[1]) if V_state is not None else 0,
        "subspace_align": subspace_align,
        "rowcheat_frame": rowcheat_summary,
    }
    return info, rows


# --------------------------------------------------------------------------
# Gradient check (finite differences)
# --------------------------------------------------------------------------


def gradient_check(A, V_exact, args, matrix, block_id):
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    rank = int(args.rank)
    half_win = int(args.half_win)
    blocks = {block_id}
    snaps = stream_to_block(args, A, V_exact, work_dtype, rank, block_id, blocks)
    snap = snaps[block_id]
    consts = per_block_constants(A, block_id, half_win)
    c_sk = consts["c_sk"]

    A_sketch = snap["A_sketch"] if snap["A_sketch"].size else None
    A_cur = snap["A_cur"]
    A_fut = snap["A_fut"]
    V_state = _state_V(snap["state"])

    cur_F2 = float(consts["cur_F2"])
    fut_F2 = float(consts["fut_F2"])
    sk_F2_low = (
        float(np.sum(np.asarray(A_sketch, dtype=np.float64) ** 2)) if A_sketch is not None else 0.0
    )
    # Op-norm-squared per-block constants for S6_OP (AB-02). Cached.
    cur_op2 = float(np.linalg.svd(np.asarray(A_cur, dtype=np.float64), compute_uv=False)[0] ** 2) \
        if A_cur.size else 0.0
    fut_op2 = float(np.linalg.svd(np.asarray(A_fut, dtype=np.float64), compute_uv=False)[0] ** 2) \
        if A_fut.size else 0.0
    if A_sketch is not None:
        st = snap.get("state")
        if st is not None and st.get("s") is not None and np.asarray(st["s"]).size:
            sk_op2_low = float(np.asarray(st["s"], dtype=np.float64)[0] ** 2)
        else:
            sk_op2_low = float(np.linalg.svd(np.asarray(A_sketch, dtype=np.float64),
                                             compute_uv=False)[0] ** 2)
    else:
        sk_op2_low = 0.0

    rng = np.random.default_rng(0)
    n = A.shape[1]
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)

    # For S5 the analytic-gradient region is raw_sk > T. A random v on the unit
    # sphere has raw_sk ≈ ‖B_top‖²_F/n, which is below T whenever the sketch
    # singular values are not yet large. Force a carry-aligned probe for S5 so
    # the FD comparison is meaningful (clipped-region grad is identically 0).
    v_S5 = v
    if A_sketch is not None and V_state is not None and V_state.size:
        v_S5 = V_state[:, 0] / max(np.linalg.norm(V_state[:, 0]), 1e-30)

    sample = rng.choice(n, size=20, replace=False)
    h = 1e-6
    rel_errs = {}
    for variant in ("S1", "S2", "S3", "S4", "S5", "S6", "S6_GM", "D0", "S6_OP"):
        kwargs = dict(variant=variant, alpha=float(args.alpha),
                      beta=float(args.beta), gamma=float(args.gamma),
                      V_state=V_state,
                      cur_F2=cur_F2, fut_F2=fut_F2, sk_F2_low=sk_F2_low,
                      cur_op2=cur_op2, fut_op2=fut_op2, sk_op2_low=sk_op2_low)
        v_use = v_S5 if variant == "S5" else v
        score0, grad, *_ = r_sk_g_value_grad(A_sketch, A_cur, A_fut, c_sk, v_use, **kwargs)
        fd = np.zeros(len(sample))
        for k, i in enumerate(sample):
            ei = np.zeros(n); ei[i] = 1.0
            s_p, *_ = r_sk_g_value_grad(A_sketch, A_cur, A_fut, c_sk, v_use + h*ei, **kwargs)
            s_m, *_ = r_sk_g_value_grad(A_sketch, A_cur, A_fut, c_sk, v_use - h*ei, **kwargs)
            fd[k] = (s_p - s_m) / (2 * h)
        ga = grad[sample]
        abs_err = np.abs(ga - fd)
        denom = max(float(np.max(np.abs(fd))), 1e-30)
        rel_err = float(np.max(abs_err) / denom)
        rel_errs[variant] = rel_err
        print(f"  block {block_id:>2} {variant}: score={score0: .6e}  "
              f"max|g-fd|={float(np.max(abs_err)):.3e}  rel={rel_err:.3e}")
    return rel_errs


# --------------------------------------------------------------------------
# Main entry: per-matrix runner
# --------------------------------------------------------------------------


def run_matrix(args, matrix, blocks_to_report):
    work_dtype = np.float32 if args.dtype == "float32" else np.float64
    A, V_exact, _, _ = probe.generate_matrix_input(
        matrix=matrix, n=args.n, preset=args.preset, seed=args.seed,
        r_sig=args.r_sig, alpha_sig=args.alpha_sig, alpha_tail=args.alpha_tail,
        tail_scale=args.tail_scale, sigma1=args.sigma1, v_type=args.v_type,
        shuffle_rows=args.shuffle_rows, row_shuffle_seed=args.row_shuffle_seed,
    )
    A = np.asarray(A, np.float64)
    V_exact = np.asarray(V_exact, np.float64)
    target = max(blocks_to_report)
    snapshots = stream_to_block(args, A, V_exact, work_dtype, int(args.rank), target, set(blocks_to_report))
    out_rows = []
    out_info = {}
    for b in sorted(blocks_to_report):
        if b not in snapshots:
            continue
        info, rows = analyze_block(args, matrix, A, V_exact, snapshots[b], b)
        out_info[b] = info
        out_rows.extend(rows)
    return out_info, out_rows


def write_text(path, infos, rows):
    by_block = {}
    for r in rows:
        by_block.setdefault(r["block"], []).append(r)
    with open(path, "w", encoding="utf-8") as f:
        for block_id in sorted(by_block.keys()):
            info = infos[block_id]
            f.write(f"== block {block_id}  matrix={info['matrix']}  N_sk={info['N_sk']}  "
                    f"union_dim={info['union_dim']}  state_rank={info['state_rank']} ==\n")
            f.write(f"  c_sk={info['c_sk']:.4e}  c_g1={info['c_g1']:.4e}  c_g2={info['c_g2']:.4e}  "
                    f"alpha={info['alpha']:.3f}  beta={info['beta']:.3f}  gamma={info['gamma']:.3f}\n")
            f.write(
                f"  {'label':<24} {'r_sk':>7} {'raw_g1':>8} {'raw_g2':>8} {'hm_g':>8} {'hm3':>8} {'hm3_usr':>8} "
                f"{'sat_S1':>8} {'sat_S2':>8} {'sat_S3':>8} {'sat_S6':>8} {'sat_GM':>8} {'sat_D0':>8} {'sat_OP':>8} "
                f"{'score_S1':>10} {'score_S2':>10} {'score_S3':>10} {'score_S4':>10} {'score_S5':>10} {'score_S6':>10} {'score_GM':>10} {'score_D0':>10} {'score_OP':>10} "
                f"{'sc_evi(F)':>10} {'sc_evi(c)':>10} "
                f"{'comb':>9} {'align_v1':>9} {'align_v2':>9} {'al_v1pr':>9} {'al_v2pr':>9} {'pa1_o2':>9} {'st_align':>9}\n"
            )
            for r in by_block[block_id]:
                f.write(
                    f"  {r['label']:<24} "
                    f"{r['r_sk']:>7.3f} {r['raw_g1']:>8.4f} {r['raw_g2']:>8.4f} {r['hm_g']:>8.4f} {r['hm3']:>8.4f} {r['hm3_user']:>8.4f} "
                    f"{r['sat_S1']:>8.4f} {r['sat_S2']:>8.4f} {r['sat_S3']:>8.4f} {r['sat_S6']:>8.4f} {r['sat_S6_GM']:>8.4f} {r['sat_D0']:>8.4f} {r['sat_S6_OP']:>8.4f} "
                    f"{r['score_S1']:>10.4e} {r['score_S2']:>10.4e} {r['score_S3']:>10.4e} {r['score_S4']:>10.4e} {r['score_S5']:>10.4e} {r['score_S6']:>10.4e} {r['score_S6_GM']:>10.4e} {r['score_D0']:>10.4e} {r['score_S6_OP']:>10.4e} "
                    f"{r['score_evi_fixed']:>10.4e} {r['score_evi_c']:>10.4e} "
                    f"{r['combined_score']:>9.4f} {r['align_v1']:>9.4f} {r['align_v2']:>9.4f} "
                    f"{r['align_v1_proj']:>9.4f} {r['align_v2_proj']:>9.4f} {r['pa1_o2']:>9.4f} "
                    f"{r['state_align']:>9.4f}\n"
                )
            # Acceptance check: score(oracle_v2_proj) ≥ score(hm_triplet_raw_best) - eps for any variant.
            picks = {r["label"]: r for r in by_block[block_id]}
            oracle = picks.get("oracle_v2_proj")
            hmraw = picks.get("hm_triplet_raw_best")
            if oracle is not None and hmraw is not None:
                _vlist = ("S1", "S2", "S3", "S4", "S5", "S6", "S6_GM", "D0", "S6_OP")
                deltas = {v: oracle[f"score_{v}"] - hmraw[f"score_{v}"] for v in _vlist}
                f.write(f"  Δ(oracle_v2 − hm_triplet_raw): "
                        f"S1={deltas['S1']:+.4e}  S2={deltas['S2']:+.4e}  "
                        f"S3={deltas['S3']:+.4e}  S4={deltas['S4']:+.4e}  S5={deltas['S5']:+.4e}  "
                        f"S6={deltas['S6']:+.4e}  S6_GM={deltas['S6_GM']:+.4e}  "
                        f"D0={deltas['D0']:+.4e}  S6_OP={deltas['S6_OP']:+.4e}\n")
            # Per-block subspace alignment (INFRA-01): principal-angle cos²
            # of each rank-2 candidate frame [v1, v2] vs the oracle frame
            # [oracle_v1_proj, oracle_v2_proj]. cos2[0] = closest principal
            # direction; cos2[1] = farthest. mean = (cos2[0]+cos2[1])/2 is
            # the Grassmann-style alignment summary used by the rank-r
            # plateau detector (§5 score_design_overview.txt).
            sub = info.get("subspace_align") or {}
            if sub:
                f.write(f"  subspace alignment (cos² principal angles vs oracle 2-frame):\n")
                f.write(f"    {'frame':<24} {'cos2_a':>8} {'cos2_b':>8} {'mean':>8}\n")
                for name, (c0, c1) in sub.items():
                    f.write(f"    {name:<24} {c0:>8.4f} {c1:>8.4f} {(c0 + c1) / 2.0:>8.4f}\n")
            # Rank-r exploitability check (INFRA-03): T2 STOP rule generalizes
            # to "score(V_oracle_frame) ≥ score(V_rowcheat_frame)" using the
            # rank-r lift of S6 / S6_GM (§5 of score_design_overview.txt).
            # rank_r = number of orthonormal columns surviving QR on the
            # top-r rows of A_fut.
            rcf = info.get("rowcheat_frame") or {}
            if rcf and "frame_S6_oracle" in rcf and "frame_S6" in rcf:
                rd = rcf.get("rowcheat_dim", "?")
                od = rcf.get("oracle_dim", "?")
                d_S6 = rcf["frame_S6_oracle"]["score"] - rcf["frame_S6"]["score"]
                d_GM = (
                    rcf["frame_S6_GM_oracle"]["score"]
                    - rcf["frame_S6_GM"]["score"]
                )
                f.write(
                    f"  Δ(oracle_frame − rowcheat_frame) [rank_r="
                    f"{rd},oracle_r={od}]: "
                    f"frame_S6={d_S6:+.4e}  frame_S6_GM={d_GM:+.4e}\n"
                )
                f.write(
                    f"    rowcheat_frame: S6={rcf['frame_S6']['score']:.4e}  "
                    f"S6_GM={rcf['frame_S6_GM']['score']:.4e}  "
                    f"u_sk={rcf['frame_S6']['u_sk']:.4e}  "
                    f"u_g1={rcf['frame_S6']['u_g1']:.4e}  "
                    f"u_g2={rcf['frame_S6']['u_g2']:.4e}\n"
                )
                f.write(
                    f"    oracle_frame:   S6={rcf['frame_S6_oracle']['score']:.4e}  "
                    f"S6_GM={rcf['frame_S6_GM_oracle']['score']:.4e}  "
                    f"u_sk={rcf['frame_S6_oracle']['u_sk']:.4e}  "
                    f"u_g1={rcf['frame_S6_oracle']['u_g1']:.4e}  "
                    f"u_g2={rcf['frame_S6_oracle']['u_g2']:.4e}\n"
                )
            f.write("\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default="mixed-tail-sharp")
    p.add_argument("--matrices", nargs="*", default=None,
                   help="If given, run multiple matrices.")
    p.add_argument("--out-prefix", default="summary/r_sk_g_score")
    p.add_argument("--blocks", nargs="+", type=int, default=[2, 6, 12, 31])
    p.add_argument("--variant", choices=("S1", "S2", "S3", "S4", "S5", "S6", "S6_GM", "D0", "S6_OP"), default="S6",
                   help="Variant to use when wiring streaming. Diagnostic always reports all variants.")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--half-win", type=int, default=32)
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--preset", default="fast")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle-rows", action="store_true", default=True)
    p.add_argument("--row-shuffle-seed", type=int, default=0)
    p.add_argument("--old-memory-size", type=int, default=32)
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--q0", type=int, default=8)
    p.add_argument("--qmax", type=int, default=48)
    p.add_argument("--krylov-depth", type=int, default=2)
    p.add_argument("--residual-tol", type=float, default=0.01)
    p.add_argument("--expansion-maxit", type=int, default=8)
    p.add_argument("--num-restarts", type=int, default=3)
    p.add_argument("--maxit", type=int, default=120)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--post-expansion-maxit", type=int, default=80)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--patience-rel-tol", type=float, default=1e-5)
    p.add_argument("--union-maxit", type=int, default=120)
    p.add_argument("--union-tol", type=float, default=1e-9)
    p.add_argument("--union-random-starts", type=int, default=24)
    p.add_argument("--r-sig", type=int, default=2)
    p.add_argument("--alpha-sig", type=float, default=0.003)
    p.add_argument("--alpha-tail", type=float, default=0.0145)
    p.add_argument("--tail-scale", type=float, default=0.99)
    p.add_argument("--sigma1", type=float, default=0.991)
    p.add_argument("--v-type", choices=("id", "U", "rand"), default="rand")
    p.add_argument("--gradient-check", action="store_true")
    p.add_argument("--no-oracle-warmstart", action="store_true",
                   help="Drop oracle_v1/v2_proj from warm-starts (matches streaming init).")
    return p.parse_args()


def main():
    args = parse_args()
    matrices = args.matrices if args.matrices else [args.matrix]

    if args.gradient_check:
        work_dtype = np.float32 if args.dtype == "float32" else np.float64
        for matrix in matrices:
            print(f"matrix={matrix}")
            A, V_exact, _, _ = probe.generate_matrix_input(
                matrix=matrix, n=args.n, preset=args.preset, seed=args.seed,
                r_sig=args.r_sig, alpha_sig=args.alpha_sig, alpha_tail=args.alpha_tail,
                tail_scale=args.tail_scale, sigma1=args.sigma1, v_type=args.v_type,
                shuffle_rows=args.shuffle_rows, row_shuffle_seed=args.row_shuffle_seed,
            )
            A = np.asarray(A, np.float64)
            V_exact = np.asarray(V_exact, np.float64)
            for b in sorted(args.blocks):
                gradient_check(A, V_exact, args, matrix, b)
        return

    overall_rows = []
    overall_infos = {}
    for matrix in matrices:
        t0 = time.time()
        infos, rows = run_matrix(args, matrix, args.blocks)
        for b, info in infos.items():
            overall_infos[(matrix, b)] = info
        overall_rows.extend(rows)
        print(f"done matrix={matrix} blocks={sorted(infos.keys())} elapsed={time.time()-t0:.2f}s")

    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    csv_path = args.out_prefix + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if overall_rows:
            w = csv.DictWriter(f, fieldnames=list(overall_rows[0].keys()))
            w.writeheader()
            w.writerows(overall_rows)
    json_path = args.out_prefix + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "infos": {f"{m}|{b}": info for (m, b), info in overall_infos.items()},
            "rows": overall_rows,
        }, f, indent=2, sort_keys=True, default=float)
    # Per-matrix txt files (handier than the global mash-up).
    for matrix in matrices:
        rows_m = [r for r in overall_rows if r["matrix"] == matrix]
        if not rows_m:
            continue
        infos_m = {b: overall_infos[(matrix, b)] for b in sorted({r["block"] for r in rows_m})}
        write_text(args.out_prefix + f"_{matrix}.txt", infos_m, rows_m)
    print(f"wrote {csv_path} {json_path} and per-matrix .txt files")


if __name__ == "__main__":
    main()
