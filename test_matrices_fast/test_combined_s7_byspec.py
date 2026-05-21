"""Validate the principled S7-by-spec extension of the combined score.

Three checks against `cex_restricted_space_probe`:
  T1. h2=0 (A_aux empty)  → with_aux output equals combined byte-for-byte.
  T2. h2>0                 → with_aux score/grad match the spec's analytical
                             formula score = (||Bv||^2+||A_w v||^2+||A_aux v||^2) * phi'
                             with phi' pooled over [R; A_w; A_aux].
  T3. Gradient FD check    → analytical gradient matches numerical gradient.

Run: python test_combined_s7_byspec.py
"""

from __future__ import annotations

import numpy as np

import cex_restricted_space_probe as probe


def make_fixture(seed, d=20, rows_B=4, rows_w=8, rows_aux=5, rows_R=6):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((rows_B, d))
    A_w = rng.standard_normal((rows_w, d))
    A_aux = rng.standard_normal((rows_aux, d))
    R = rng.standard_normal((rows_R, d))
    v = rng.standard_normal(d)
    v = v / np.linalg.norm(v)
    return B, A_w, A_aux, R, v


def spec_score(B, A_w, A_aux, R, v, n_total, n_old):
    # Direct evaluation of the spec formula.
    Bv = B @ v
    Awv = A_w @ v
    Auxv = A_aux @ v
    Rv = R @ v if R is not None and R.size else None

    gain = float(Bv @ Bv) + float(Awv @ Awv) + float(Auxv @ Auxv)

    pooled_y2 = float(Awv @ Awv) + float(Auxv @ Auxv)
    pooled_y4 = float(np.sum(Awv ** 4)) + float(np.sum(Auxv ** 4))
    rows_entropy = int(A_w.shape[0]) + int(A_aux.shape[0])
    if Rv is not None:
        pooled_y2 += float(Rv @ Rv)
        pooled_y4 += float(np.sum(Rv ** 4))
        rows_entropy += int(R.shape[0])

    rows_seen = n_old + int(A_w.shape[0]) + int(A_aux.shape[0])
    rows_ref_eff = max(int(n_total), rows_entropy)
    rows_seen = min(rows_seen, rows_ref_eff)

    c = np.log(rows_seen / rows_ref_eff) / (2.0 * np.log(rows_entropy))
    phi = float(np.exp(c * (np.log(pooled_y4) - 2.0 * np.log(pooled_y2))))
    return gain * phi, phi, c


def spec_grad(B, A_w, A_aux, R, v, n_total, n_old, score, phi, c):
    Bv = B @ v
    Awv = A_w @ v
    Auxv = A_aux @ v
    Rv = R @ v if R is not None and R.size else None

    grad_energy = 2.0 * (B.T @ Bv) + 2.0 * (A_w.T @ Awv) + 2.0 * (A_aux.T @ Auxv)
    gain = float(Bv @ Bv) + float(Awv @ Awv) + float(Auxv @ Auxv)

    pooled_y2 = float(Awv @ Awv) + float(Auxv @ Auxv)
    pooled_y4 = float(np.sum(Awv ** 4)) + float(np.sum(Auxv ** 4))
    cy = A_w.T @ Awv + A_aux.T @ Auxv
    cy3 = A_w.T @ (Awv ** 3) + A_aux.T @ (Auxv ** 3)
    if Rv is not None:
        pooled_y2 += float(Rv @ Rv)
        pooled_y4 += float(np.sum(Rv ** 4))
        cy = cy + R.T @ Rv
        cy3 = cy3 + R.T @ (Rv ** 3)

    grad_log_phi = 4.0 * c * (cy3 / pooled_y4 - cy / pooled_y2)
    return phi * grad_energy + score * grad_log_phi


def test_T1_aux_none_identity():
    B, A_w, A_aux, R, v = make_fixture(seed=1)
    n_total = 1024
    n_old = 200

    M_gain = np.vstack([B, A_w])
    rows_block = A_w.shape[0]

    s_ref = probe.combined_streaming_score_grad_reduced(
        M_gain, A_w, None, None, R, v, rows_block, n_total, n_old,
    )
    s_aux_none = probe.combined_streaming_score_grad_reduced_with_aux(
        M_gain, A_w, None, None, R, v, rows_block, n_total, n_old,
        A_aux=None,
    )
    s_aux_empty = probe.combined_streaming_score_grad_reduced_with_aux(
        M_gain, A_w, None, None, R, v, rows_block, n_total, n_old,
        A_aux=np.zeros((0, B.shape[1])),
    )

    for label, s in [("None", s_aux_none), ("empty", s_aux_empty)]:
        assert s[0] == s_ref[0], f"T1 score mismatch ({label}): {s[0]} vs {s_ref[0]}"
        np.testing.assert_array_equal(s[1], s_ref[1], err_msg=f"T1 grad mismatch ({label})")
        assert s[2] == s_ref[2], f"T1 s mismatch ({label})"
        assert s[3] == s_ref[3], f"T1 H mismatch ({label})"
    print(f"T1 PASS  combined == with_aux(A_aux=None)  score={s_ref[0]:.10g}  ||grad||={np.linalg.norm(s_ref[1]):.6g}")


def test_T2_spec_formula_match():
    B, A_w, A_aux, R, v = make_fixture(seed=2)
    n_total = 1024
    n_old = 200

    M_gain = np.vstack([B, A_w])
    rows_block = A_w.shape[0]

    score, grad, s_norm, H = probe.combined_streaming_score_grad_reduced_with_aux(
        M_gain, A_w, None, None, R, v, rows_block, n_total, n_old,
        A_aux=A_aux,
    )

    expected_score, phi, c = spec_score(B, A_w, A_aux, R, v, n_total, n_old)
    expected_grad = spec_grad(B, A_w, A_aux, R, v, n_total, n_old, expected_score, phi, c)

    rel = abs(score - expected_score) / max(abs(expected_score), 1e-30)
    grad_rel = np.linalg.norm(grad - expected_grad) / max(np.linalg.norm(expected_grad), 1e-30)
    assert rel < 1e-12, f"T2 score rel_err={rel:.3e}  impl={score}  spec={expected_score}"
    assert grad_rel < 1e-12, f"T2 grad rel_err={grad_rel:.3e}"
    print(f"T2 PASS  with_aux matches spec formula  score={score:.10g}  rel_err={rel:.2e}  grad_rel={grad_rel:.2e}")


def test_T3_finite_difference_gradient():
    B, A_w, A_aux, R, v = make_fixture(seed=3)
    n_total = 1024
    n_old = 200

    M_gain = np.vstack([B, A_w])
    rows_block = A_w.shape[0]

    def f(x):
        s, _, _, _ = probe.combined_streaming_score_grad_reduced_with_aux(
            M_gain, A_w, None, None, R, x, rows_block, n_total, n_old,
            A_aux=A_aux,
        )
        return s

    _, grad_anal, _, _ = probe.combined_streaming_score_grad_reduced_with_aux(
        M_gain, A_w, None, None, R, v, rows_block, n_total, n_old,
        A_aux=A_aux,
    )

    eps = 1e-6
    grad_fd = np.zeros_like(v)
    for i in range(len(v)):
        e = np.zeros_like(v)
        e[i] = eps
        grad_fd[i] = (f(v + e) - f(v - e)) / (2.0 * eps)

    rel = np.linalg.norm(grad_anal - grad_fd) / max(np.linalg.norm(grad_anal), 1e-30)
    assert rel < 1e-6, f"T3 FD gradient mismatch rel={rel:.3e}"
    print(f"T3 PASS  FD gradient matches analytical  rel_err={rel:.2e}")


def test_T4_h2_zero_equivalence_random_v():
    # End-to-end: with A_aux of shape (0, d), with_aux must agree with combined
    # across many random v's, simulating the optimizer probing the search space.
    B, A_w, _, R, _ = make_fixture(seed=4)
    n_total = 1024
    n_old = 200
    M_gain = np.vstack([B, A_w])
    rows_block = A_w.shape[0]

    rng = np.random.default_rng(99)
    A_aux_empty = np.zeros((0, B.shape[1]))
    max_diff_score = 0.0
    max_diff_grad = 0.0
    for _ in range(50):
        v = rng.standard_normal(B.shape[1])
        v = v / np.linalg.norm(v)
        s_ref = probe.combined_streaming_score_grad_reduced(
            M_gain, A_w, None, None, R, v, rows_block, n_total, n_old,
        )
        s_new = probe.combined_streaming_score_grad_reduced_with_aux(
            M_gain, A_w, None, None, R, v, rows_block, n_total, n_old,
            A_aux=A_aux_empty,
        )
        max_diff_score = max(max_diff_score, abs(s_ref[0] - s_new[0]))
        max_diff_grad = max(max_diff_grad, float(np.max(np.abs(s_ref[1] - s_new[1]))))
    assert max_diff_score == 0.0, f"T4 score not byte-equal: {max_diff_score}"
    assert max_diff_grad == 0.0, f"T4 grad not byte-equal: {max_diff_grad}"
    print(f"T4 PASS  50 random v: score and grad byte-equal between combined and with_aux(empty)")


if __name__ == "__main__":
    test_T1_aux_none_identity()
    test_T2_spec_formula_match()
    test_T3_finite_difference_gradient()
    test_T4_h2_zero_equivalence_random_v()
    print("\nAll 4 checks passed.")
