import numpy as np

import cex_restricted_space_probe as probe


def score_frame(A_block, Z, rows_ref):
    vals = [
        probe.score_full_vector_forget(
            A_block,
            A_block,
            Z[:, j],
            rows_ref,
            state_prev=None,
            score_variant="combined",
            old_row_memory=None,
            row_concentration_lambda=0.0,
        )
        for j in range(Z.shape[1])
    ]
    return float(np.sum(vals)), np.asarray(vals, dtype=float)


def component(A_block, v, rows_ref):
    return probe.combined_score_component_details(
        A_block,
        A_block,
        v,
        rows_ref,
        state_prev=None,
        old_row_memory=None,
    )


def response_report(label, A, v, block_size=32, top_k=12, row_perm=None):
    y = np.asarray(A @ v, dtype=float).reshape(-1)
    abs_y = np.abs(y)
    top = np.argsort(abs_y)[-top_k:][::-1]
    block_energy = []
    for start in range(0, A.shape[0], block_size):
        stop = min(start + block_size, A.shape[0])
        block_energy.append(float(np.dot(y[start:stop], y[start:stop])))
    block_energy = np.asarray(block_energy)
    if row_perm is None:
        top_rows = [(int(i + 1), float(y[i]), float(abs_y[i] ** 2)) for i in top]
    else:
        top_rows = [
            (int(i + 1), int(np.asarray(row_perm)[i] + 1), float(y[i]), float(abs_y[i] ** 2))
            for i in top
        ]
    print(f"{label}_response_top_rows_1based", top_rows)
    print(f"{label}_response_block_energy", block_energy, "frac", block_energy / max(float(np.sum(block_energy)), 1e-30))
    print(
        f"{label}_response_stats",
        "l2sq", float(np.dot(y, y)),
        "max_frac", float(np.max(abs_y * abs_y) / max(float(np.dot(y, y)), 1e-30)),
        "top4_frac", float(np.sum(np.sort(abs_y * abs_y)[-4:]) / max(float(np.dot(y, y)), 1e-30)),
    )


def singular_alignment_report(label, A_mat, directions, top_k=8):
    _, s, vh = np.linalg.svd(A_mat, full_matrices=False)
    V = vh.T
    print(f"{label}_top_svals", s[:top_k])
    for dlabel, v in directions:
        print(f"{label}_{dlabel}_absdot_top_right_svecs", np.abs(np.asarray(v).reshape(-1) @ V[:, :top_k]))


def outside_decomposition_report(label, A, A_block, V_score, Q_oracle, V_exact, row_perm, rows_ref):
    print(f"{label}_score_sum", score_frame(A_block, V_score[:, :2], rows_ref))
    print(f"{label}_principal_cosines", probe.subspace_principal_cosines(V_score[:, :2], Q_oracle[:, :2]))
    raw_cols = []
    _, Q_row = probe.projected_true_span_oracle(A_block, V_exact[:, :2], 2, dtype=np.float64)
    for j in range(2):
        v = probe.project_onto_span(V_exact[:, j], Q_row).reshape(-1)
        raw_cols.append(v / np.linalg.norm(v))
    raw = np.column_stack(raw_cols)
    print(f"{label}_vecnorm_raw_projected_oracle_into_V_score", np.linalg.norm(V_score[:, :2] @ (V_score[:, :2].T @ raw), axis=0))

    for j in range(2):
        v = np.asarray(V_score[:, j], dtype=float)
        oracle_part = Q_oracle[:, :2] @ (Q_oracle[:, :2].T @ v)
        outside = v - oracle_part
        outside_norm = float(np.linalg.norm(outside))
        print(
            f"{label}_v{j + 1}_oracle_part_norm",
            float(np.linalg.norm(oracle_part)),
            "outside_norm",
            outside_norm,
            "outside_energy",
            outside_norm * outside_norm,
        )
        if outside_norm <= 1e-10:
            continue
        outside /= outside_norm
        exact_abs = np.abs(outside @ V_exact)
        top_exact = np.argsort(exact_abs)[-10:][::-1]
        print(f"{label}_v{j + 1}_outside_top_global_exact_1based", [(int(i + 1), float(exact_abs[i])) for i in top_exact])
        response_report(f"{label}_v{j + 1}_outside_first_block", A_block, outside, row_perm=row_perm[:32])
        response_report(f"{label}_v{j + 1}_outside_all_rows", A, outside, row_perm=row_perm)


def signed(v, ref):
    return v if float(v @ ref) >= 0.0 else -v


def best_rotation_with_one_outside(A_block, q1, q2, w, rows_ref, ngrid=4001):
    best = None
    for theta in np.linspace(-0.5 * np.pi, 0.5 * np.pi, ngrid):
        z1 = q1
        z2 = np.cos(theta) * q2 + np.sin(theta) * w
        total, vals = score_frame(A_block, np.column_stack([z1, z2]), rows_ref)
        if best is None or total > best[0]:
            best = (total, vals, theta, z2)
    return best


def main():
    np.random.seed(0)
    A, V_exact, _, _ = probe.generate_matrix_input(
        matrix="static-cex",
        n=128,
        preset="small",
        seed=0,
        r_sig=2,
        alpha_sig=0.003,
        alpha_tail=0.0145,
        tail_scale=0.99,
        sigma1=0.991,
        v_type="rand",
    )
    A = np.asarray(A, dtype=np.float64)
    A_block = np.asarray(A[:32, :], dtype=np.float64)
    rows_ref = A.shape[0]
    rng_state = np.random.get_state()
    np.random.seed(0)
    _ = np.linalg.qr(np.random.randn(128, 128), mode="reduced")
    row_perm = np.random.permutation(128)
    np.random.set_state(rng_state)
    print("first_block_unpermuted_rows_1based", (row_perm[:32] + 1))

    Q_oracle, Q_row = probe.projected_true_span_oracle(
        A_block, V_exact[:, :2], 2, dtype=np.float64
    )

    V_fast, s_fast, H_fast, score_fast, diag_fast = probe.entropy_iter_basis_forget(
        M_gain=A_block.astype(np.float32),
        active_r=2,
        rows_ref=128,
        V_init=None,
        q0=5,
        qmax=200,
        krylov_depth=2,
        residual_tol=0.01,
        expansion_maxit=64,
        num_restarts=2,
        maxit=120,
        tol=1e-8,
        rng=np.random.default_rng(0),
        verbose=False,
        state_prev=None,
        A_block=A_block.astype(np.float32),
        rows_total=32,
        reduced_optimizer="cex",
        basis_selection="greedy",
        joint_warm_start_oracle=True,
        oracle_warm_start_target=V_exact,
        work_dtype=np.float32,
        expansion_direction="residual",
        reuse_line_search_grad=True,
        expansion_warm_start=True,
        post_expansion_maxit=60,
        row_concentration_lambda=0.0,
        score_variant="combined",
        old_row_memory=None,
        oracle_projection_row_samples=None,
    )
    print("fast_oracle_warm_first_block_s", s_fast)
    print("fast_oracle_warm_first_block_H", H_fast)
    print("fast_oracle_warm_first_block_scores", score_fast)
    print("fast_oracle_warm_first_block_diag", {
        "subspace_dims": diag_fast["subspace_dims"][:2].tolist(),
        "grad_perp_ratio": diag_fast["grad_perp_ratio"][:2].tolist(),
        "regularized_score_sum": diag_fast["regularized_score_sum"],
    })
    outside_decomposition_report(
        "fast_oracle_warm",
        A,
        A_block,
        np.asarray(V_fast, dtype=np.float64),
        Q_oracle,
        V_exact,
        row_perm,
        rows_ref,
    )
    raw_cols = []
    for j in range(2):
        v = probe.project_onto_span(V_exact[:, j], Q_row).reshape(-1)
        raw_cols.append(v / np.linalg.norm(v))
    raw = np.column_stack(raw_cols)

    total_q, vals_q = score_frame(A_block, Q_oracle[:, :2], rows_ref)
    total_raw1_q2, vals_raw1_q2 = score_frame(
        A_block, np.column_stack([raw[:, 0], Q_oracle[:, 1]]), rows_ref
    )
    print("oracle_qr", total_q, vals_q)
    print("raw_overlap", abs(float(raw[:, 0] @ raw[:, 1])))
    print("raw1_q2", total_raw1_q2, vals_raw1_q2)
    for name, v in [("q1", Q_oracle[:, 0]), ("q2", Q_oracle[:, 1]), ("raw1", raw[:, 0]), ("raw2", raw[:, 1])]:
        c = component(A_block, v, rows_ref)
        print(
            name,
            "score", c["score_total"],
            "gain2", c["gain2"],
            "phi", c["phi"],
            "H", c["pooled_H"],
            "relH", c["pooled_rel_H"],
        )

    total0, vals0, grad0, _, _ = probe.entropyscore_forget_joint_reduced_eval(
        A_block,
        Q_oracle[:, :2],
        A_block.shape[0],
        rows_ref,
        optimizer="cex",
        row_concentration_lambda=0.0,
        score_variant="combined",
    )
    tangent = probe.stiefel_tangent_gradient(Q_oracle[:, :2], grad0, None)
    outside_tangent = tangent - Q_oracle[:, :2] @ (Q_oracle[:, :2].T @ tangent)
    print(
        "oracle_tangent",
        "total", total0,
        "vals", vals0,
        "norm", np.linalg.norm(tangent, ord="fro"),
        "outside_norms", np.linalg.norm(outside_tangent, axis=0),
    )
    for j in range(2):
        g = outside_tangent[:, j]
        ng = np.linalg.norm(g)
        if ng <= 1e-12:
            continue
        g /= ng
        response_report(f"oracle_outside_gradient_col{j + 1}_first_block", A_block, g, row_perm=row_perm[:32])
        response_report(f"oracle_outside_gradient_col{j + 1}_all_rows", A, g, row_perm=row_perm)
        exact_abs_g = np.abs(g @ V_exact)
        top_exact_g = np.argsort(exact_abs_g)[-8:][::-1]
        print(
            f"oracle_outside_gradient_col{j + 1}_top_global_exact_1based",
            [(int(i + 1), float(exact_abs_g[i])) for i in top_exact_g],
        )

    rng = np.random.default_rng(123)
    starts = []
    starts.append(Q_oracle[:, :2])
    for _ in range(199):
        Z0 = rng.standard_normal((A.shape[1], 2))
        starts.append(probe.retract_stiefel_reduced(Z0, None))

    best = None
    for idx, Z0 in enumerate(starts):
        cand = probe.basic_projected_ascent_joint_reduced_forget(
            A_block,
            Z0,
            None,
            A_block.shape[0],
            rows_ref,
            maxit=400,
            tol=1e-10,
            optimizer="cex",
            reuse_line_search_grad=True,
            row_concentration_lambda=0.0,
            score_variant="combined",
        )
        if best is None or cand[1] > best[2]:
            best = (idx, *cand)

    idx, Z_best, total_best, vals_best, s_best, H_best, stop = best
    print("best_joint", "restart", idx, "total", total_best, "vals", vals_best, "s", s_best, "H", H_best, "stop", stop)
    print("best_vs_oracle_cosines", probe.subspace_principal_cosines(Z_best, Q_oracle[:, :2]))
    print("best_vs_raw_absdot")
    print(np.abs(Z_best.T @ raw))
    print("best_vs_q_absdot")
    print(np.abs(Z_best.T @ Q_oracle[:, :2]))

    # Identify the main outside-oracle direction in the best second vector.
    dots = Z_best.T @ Q_oracle[:, :2]
    q_energy = np.sum(dots * dots, axis=1)
    outside_idx = int(np.argmin(q_energy))
    z_out = Z_best[:, outside_idx]
    outside = z_out - Q_oracle[:, :2] @ (Q_oracle[:, :2].T @ z_out)
    outside /= np.linalg.norm(outside)
    outside = signed(outside, z_out)
    print("outside_idx", outside_idx, "oracle_energy", q_energy[outside_idx])
    response_report("best_outside_first_block", A_block, outside, row_perm=row_perm[:32])
    response_report("best_outside_all_rows", A, outside, row_perm=row_perm)

    rot = best_rotation_with_one_outside(
        A_block, Q_oracle[:, 0], Q_oracle[:, 1], outside, rows_ref
    )
    print("rotate_q2_toward_best_outside", "total", rot[0], "vals", rot[1], "theta", rot[2])
    c_rot = component(A_block, rot[3], rows_ref)
    print(
        "rotated_second_components",
        "gain2", c_rot["gain2"],
        "phi", c_rot["phi"],
        "H", c_rot["pooled_H"],
        "relH", c_rot["pooled_rel_H"],
    )

    _, _, vh = np.linalg.svd(A_block, full_matrices=False)
    Vsvd = vh.T
    print("outside_absdot_top_svd", np.abs(outside @ Vsvd[:, :8]))
    singular_alignment_report(
        "first_block",
        A_block,
        [
            ("q1", Q_oracle[:, 0]),
            ("q2", Q_oracle[:, 1]),
            ("outside", outside),
            ("outside_grad1", outside_tangent[:, 0] / np.linalg.norm(outside_tangent[:, 0])),
            ("outside_grad2", outside_tangent[:, 1] / np.linalg.norm(outside_tangent[:, 1])),
        ],
    )
    singular_alignment_report(
        "all_rows",
        A,
        [
            ("q1", Q_oracle[:, 0]),
            ("q2", Q_oracle[:, 1]),
            ("outside", outside),
            ("outside_grad1", outside_tangent[:, 0] / np.linalg.norm(outside_tangent[:, 0])),
            ("outside_grad2", outside_tangent[:, 1] / np.linalg.norm(outside_tangent[:, 1])),
        ],
    )
    exact_abs = np.abs(outside @ V_exact)
    top_exact = np.argsort(exact_abs)[-10:][::-1]
    print("outside_top_global_exact_1based", [(int(i + 1), float(exact_abs[i])) for i in top_exact])

    for exact_idx in range(2, 8):
        w = V_exact[:, exact_idx] - Q_oracle[:, :2] @ (Q_oracle[:, :2].T @ V_exact[:, exact_idx])
        nw = np.linalg.norm(w)
        if nw <= 1e-12:
            continue
        w /= nw
        rot_i = best_rotation_with_one_outside(
            A_block, Q_oracle[:, 0], Q_oracle[:, 1], w, rows_ref, ngrid=2001
        )
        print(
            f"rotate_q2_toward_exact_v{exact_idx + 1}",
            "total", rot_i[0],
            "vals", rot_i[1],
            "theta", rot_i[2],
        )


if __name__ == "__main__":
    main()
