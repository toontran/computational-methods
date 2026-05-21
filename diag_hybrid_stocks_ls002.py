#!/usr/bin/env python3
"""Diagnostic runner: evaluate combined-score components on kernel_stocks_1000_0.02
with a SIMULATED streaming state (old_row_memory populated), so phi is non-trivial.

Question: does the entropy multiplier phi discriminate meaningfully between
candidate directions on this matrix, or does it collapse toward a constant?
If phi is nearly constant across candidates, then combined-score == gain^2 (up to const)
and hybrid cannot beat iSVD here regardless of optimizer effort.
"""
import numpy as np
import scipy as sp
import scipy.io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils

home_dir = os.path.expanduser("~")
points = np.asarray(sp.io.mmread(os.path.join(home_dir, "data/data_2m.mtx")))[:1000, :]


def build_rbf(points, ls):
    sq = np.sum(points**2, axis=1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2.0 * points @ points.T, 0.0)
    return np.exp(-d2 / (2.0 * ls**2)).astype(np.float32)


def diagnose(K, tag):
    N = K.shape[0]
    U_ex, S_ex, Vt_ex = np.linalg.svd(K)

    # Simulate streaming: pass 10 prior blocks of win=10 into R_old_memory
    win = 10
    k = 5
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    prior_rows = perm[:10 * win]       # 100 rows already seen
    current_rows = perm[10 * win:11 * win]  # next 10 rows
    A_block = K[current_rows, :]
    R_old_memory = K[prior_rows, :]     # pooled memory of prior blocks
    # M_gain for current step ~ A_block (no deflation simulation)
    M_gain = A_block.copy()

    # rows_ref = N (total matrix rows), matches utils.py:7606 wiring
    rows_ref = N
    state_prev = {"rows_seen": 10 * win}

    V_oracle = Vt_ex[:k, :].T
    _, _, Vt_A = np.linalg.svd(A_block, full_matrices=False)
    V_isvd = Vt_A[:k, :].T
    V_rand = rng.standard_normal((N, k)).astype(np.float32)
    V_rand, _ = np.linalg.qr(V_rand)
    # Also: top-k from the CONCATENATED prior+current rows (iSVD-ish after 10 blocks)
    V_from_seen = np.linalg.svd(np.vstack([R_old_memory, A_block]), full_matrices=False)[2][:k, :].T

    print(f"\n=== {tag}: σ₁/σ₂ = {S_ex[0]/S_ex[1]:.4f} ===")
    print(f"  streaming state: rows_seen_prior=100, current_block=10 rows, rows_ref={rows_ref}")

    def scores(label, V):
        rows_entropy_arr = []
        phis = []
        gain2s = []
        totals = []
        for j in range(V.shape[1]):
            comp = utils.combined_score_component_details(
                M_gain, A_block, V[:, j], rows_ref,
                state_prev=state_prev, old_row_memory=R_old_memory
            )
            phis.append(comp['phi'])
            gain2s.append(comp['gain2'])
            totals.append(comp['score_total'])
            rows_entropy_arr.append(comp['rows_entropy'])
        return np.asarray(phis), np.asarray(gain2s), np.asarray(totals), np.asarray(rows_entropy_arr)

    for label, V in [("oracle (top-k of full K)", V_oracle),
                     ("iSVD_pick(current block)", V_isvd),
                     ("iSVD_pick(all_seen)     ", V_from_seen),
                     ("random                 ", V_rand)]:
        phi, g2, tot, re_ = scores(label, V)
        # Also show the ratio of phi_max/phi_min among these 5 candidates — phi "discriminability"
        phi_ratio = phi.max() / phi.min() if phi.min() > 0 else float("inf")
        print(f"  {label:<28}: phi in [{phi.min():.4f}, {phi.max():.4f}]  "
              f"ratio={phi_ratio:.3f}  gain2 in [{g2.min():.3f}, {g2.max():.3f}]  "
              f"score_total in [{tot.min():.3e}, {tot.max():.3e}]")

    # Key summary metric: if we stack oracle + iSVD + random all together and score them,
    # does the "winner by gain2 alone" disagree with the "winner by gain2*phi"?
    V_all = np.column_stack([V_oracle[:, 0], V_isvd[:, 0], V_from_seen[:, 0], V_rand[:, 0]])
    labels = ["oracle", "isvd_current", "isvd_all_seen", "random"]
    g2s = []
    totals = []
    for j in range(V_all.shape[1]):
        comp = utils.combined_score_component_details(
            M_gain, A_block, V_all[:, j], rows_ref,
            state_prev=state_prev, old_row_memory=R_old_memory
        )
        g2s.append(comp['gain2'])
        totals.append(comp['score_total'])
    g2s = np.asarray(g2s)
    totals = np.asarray(totals)
    rank_by_g2 = np.argsort(-g2s)
    rank_by_tot = np.argsort(-totals)
    print(f"\n  Head-to-head (oracle vs iSVD vs random, column 0 of each):")
    for j in range(4):
        print(f"    {labels[j]:>14}: gain2={g2s[j]:.5f}  score_total={totals[j]:.5f}")
    print(f"  ranking by gain2  : {[labels[i] for i in rank_by_g2]}")
    print(f"  ranking by combined: {[labels[i] for i in rank_by_tot]}")
    if list(rank_by_g2) == list(rank_by_tot):
        print("  -> phi does NOT re-order candidates; combined-score reduces to gain^2 here.")
    else:
        print("  -> phi DOES re-order candidates; combined-score carries extra signal.")


# Run at two lengthscales for contrast
for ls in [0.02, 0.2236]:
    K = build_rbf(points, ls)
    diagnose(K, f"kernel_stocks_1000_{ls}")
