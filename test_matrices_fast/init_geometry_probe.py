import argparse

import numpy as np

import cex_restricted_space_probe as probe


def normalized_rows(X, eps=1e-14):
    X_arr = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X_arr, axis=1)
    keep = norms > eps
    if not np.any(keep):
        return X_arr[:0], np.array([], dtype=int)
    return X_arr[keep] / norms[keep, None], np.flatnonzero(keep)


def orth(X):
    Q, R = np.linalg.qr(np.asarray(X, dtype=np.float64), mode="reduced")
    keep = np.abs(np.diag(R)) > 1e-12
    return Q[:, keep]


def subspace_cosines(A, B):
    QA = orth(A)
    QB = orth(B)
    if QA.size == 0 or QB.size == 0:
        return np.array([])
    return np.linalg.svd(QA.T @ QB, compute_uv=False)


def row_proximity(frame, X):
    Q = orth(frame)
    Xn, row_ids = normalized_rows(X)
    if Xn.size == 0 or Q.size == 0:
        return {"idx": None, "projection": np.nan, "col_cos": np.nan, "col": None}
    projections = np.linalg.norm(Xn @ Q, axis=1)
    best = int(np.argmax(projections))
    col_hits = np.abs(Xn @ Q)
    best_col_flat = int(np.argmax(col_hits[best]))
    return {
        "idx": int(row_ids[best]),
        "projection": float(projections[best]),
        "col_cos": float(col_hits[best, best_col_flat]),
        "col": best_col_flat + 1,
    }


def fmt_cos(values):
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return "n/a"
    return ",".join(f"{x:.6f}" for x in vals)


def row_space_projection(M_gain, X):
    X_arr = np.asarray(X, dtype=np.float64)
    row_basis = orth(np.asarray(M_gain, dtype=np.float64).T)
    if row_basis.size == 0 or X_arr.size == 0:
        return np.zeros_like(X_arr)
    return row_basis @ (row_basis.T @ X_arr)


def oracle_frame(M_gain, V_exact, rank):
    return orth(row_space_projection(M_gain, V_exact[:, :rank]))


def stiefel_align_to_target(frame, target):
    frame_arr = np.asarray(frame, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    if frame_arr.size == 0 or target_arr.size == 0:
        return np.nan
    I = np.eye(frame_arr.shape[0], dtype=np.float64)
    return float(np.linalg.norm((I - frame_arr @ frame_arr.T) @ target_arr, ord="fro"))


def frame_carry(frame, M_gain):
    _, s_new, Vt_new, _ = probe.left_projected_operator_svd_factors(frame.T, M_gain)
    return np.asarray(Vt_new.T, dtype=np.float64), np.asarray(s_new, dtype=np.float64)


def evaluate_joint_frame(frame, state, A_block_work, M_gain, rows_ref, optimizer="cex", row_concentration_lambda=0.0):
    frame = np.asarray(frame, dtype=np.float64)
    Z = np.eye(frame.shape[1], dtype=np.float64)
    if state is None:
        B = np.ascontiguousarray(A_block_work @ frame, dtype=frame.dtype)
        total, vals, _, s, H = probe.entropyscore_forget_joint_reduced_eval(
            B, Z, A_block_work.shape[0], rows_ref, optimizer=optimizer,
            row_concentration_lambda=row_concentration_lambda
        )
    else:
        B_gain = np.ascontiguousarray(M_gain @ frame, dtype=frame.dtype)
        B_block = np.ascontiguousarray(A_block_work @ frame, dtype=frame.dtype)
        C_prev = np.ascontiguousarray(state["V"].T @ frame, dtype=frame.dtype)
        total, vals, _, s, H = probe.entropyscore_forget_joint_streaming_reduced_eval(
            B_gain, B_block, C_prev, state["s2"], Z, A_block_work.shape[0], rows_ref, optimizer=optimizer,
            row_concentration_lambda=row_concentration_lambda
        )
    return float(total), np.asarray(vals, dtype=np.float64), np.asarray(s, dtype=np.float64), np.asarray(H, dtype=np.float64)


def run_joint_from_basis(
    basis, Z0, state, A_block_work, M_gain, rows_ref, optimizer="cex", maxit=1000, tol=1e-10,
    reuse_line_search_grad=True, row_concentration_lambda=0.0
):
    basis = np.asarray(basis, dtype=np.float64)
    Z0 = np.asarray(Z0, dtype=np.float64)
    if state is None:
        B = np.ascontiguousarray(A_block_work @ basis, dtype=basis.dtype)
        Z, total, vals, s, H, stop = probe.basic_projected_ascent_joint_reduced_forget(
            B, Z0, None, A_block_work.shape[0], rows_ref,
            maxit=maxit, tol=tol, optimizer=optimizer, reuse_line_search_grad=reuse_line_search_grad,
            row_concentration_lambda=row_concentration_lambda
        )
    else:
        B_gain = np.ascontiguousarray(M_gain @ basis, dtype=basis.dtype)
        B_block = np.ascontiguousarray(A_block_work @ basis, dtype=basis.dtype)
        C_prev = np.ascontiguousarray(state["V"].T @ basis, dtype=basis.dtype)
        Z, total, vals, s, H, stop = probe.basic_projected_ascent_joint_reduced_streaming_forget(
            B_gain, B_block, C_prev, state["s2"], Z0, None, A_block_work.shape[0], rows_ref,
            maxit=maxit, tol=tol, optimizer=optimizer, reuse_line_search_grad=reuse_line_search_grad,
            row_concentration_lambda=row_concentration_lambda
        )
    frame = np.ascontiguousarray(basis @ Z, dtype=np.float64)
    return frame, float(total), np.asarray(vals, dtype=np.float64), np.asarray(s, dtype=np.float64), np.asarray(H, dtype=np.float64), stop


def columnwise_joint_gradient(frame, state, A_block_work, M_gain, rows_ref):
    frame = np.asarray(frame, dtype=np.float64)
    grads = []
    scores = []
    for j in range(frame.shape[1]):
        v = frame[:, j]
        if state is None:
            logf, grad_log, _, _ = probe.entropyscore_forget_logscore_grad_rows(A_block_work, v, rows_ref)
        else:
            logf, grad_log, _, _ = probe.entropyscore_forget_streaming_logscore_grad(
                M_gain, A_block_work, state["V"], state["s2"], v, rows_ref
            )
        score = float(np.exp(logf))
        grads.append(np.asarray(score * grad_log, dtype=np.float64))
        scores.append(score)
    return np.column_stack(grads), np.asarray(scores, dtype=np.float64)


def dominant_outside_projection_summary(frame, state, A_block_work, M_gain, V_exact, rows_ref, row_concentration_lambda=0.0):
    Q = np.asarray(frame, dtype=np.float64)
    _, _, _, G_tan, _, _ = probe.entropyscore_forget_joint_full_score_tangent(
        M_gain, A_block_work, Q, rows_ref, state_prev=state, optimizer="cex",
        row_concentration_lambda=row_concentration_lambda
    )
    split = probe.stiefel_tangent_rotation_split(Q, G_tan)
    G_out = split["outside_complement"]
    tan_norm = split["tangent_norm"]
    out_norm = split["outside_complement_norm"]
    inside_norm = split["inside_rotation_norm"]
    if G_out.size == 0 or out_norm <= 1e-14:
        return {
            "tangent_norm": tan_norm,
            "inside_norm": inside_norm,
            "outside_norm": out_norm,
            "proj_exact_v3": np.nan,
            "proj_exact_v4": np.nan,
            "proj_svec_1_2": np.nan,
            "proj_svec_1_5": np.nan,
            "proj_svec_3_5": np.nan,
            "top_rows": "n/a",
        }

    u, _, _ = np.linalg.svd(G_out, full_matrices=False)
    dom = np.asarray(u[:, 0], dtype=np.float64)

    exact_proj = row_space_projection(M_gain, V_exact[:, 2:4])
    exact_resids = []
    for j in range(min(2, exact_proj.shape[1])):
        v = np.asarray(exact_proj[:, j], dtype=np.float64)
        v = v - Q @ (Q.T @ v)
        nv = float(np.linalg.norm(v))
        if nv > 1e-14:
            exact_resids.append(v / nv)
    exact_v3 = abs(float(dom @ exact_resids[0])) if len(exact_resids) > 0 else np.nan
    exact_v4 = abs(float(dom @ exact_resids[1])) if len(exact_resids) > 1 else np.nan

    _, _, M_vh = np.linalg.svd(np.asarray(M_gain, dtype=np.float64), full_matrices=False)
    block_v = M_vh.T
    spans = {
        "proj_svec_1_2": block_v[:, : min(2, block_v.shape[1])],
        "proj_svec_1_5": block_v[:, : min(5, block_v.shape[1])],
        "proj_svec_3_5": block_v[:, 2: min(5, block_v.shape[1])],
    }
    span_proj = {}
    for key, span in spans.items():
        if span.size == 0:
            span_proj[key] = np.nan
        else:
            span_proj[key] = float(np.linalg.norm(orth(span).T @ dom))

    Xn, row_ids = normalized_rows(M_gain)
    if Xn.size == 0:
        top_rows = "n/a"
    else:
        cos = np.abs(Xn @ dom)
        order = np.argsort(cos)[::-1][:5]
        top_rows = ",".join(f"{int(row_ids[i]) + 1}:{cos[i]:.6f}" for i in order)

    return {
        "tangent_norm": tan_norm,
        "inside_norm": inside_norm,
        "outside_norm": out_norm,
        "proj_exact_v3": exact_v3,
        "proj_exact_v4": exact_v4,
        "proj_svec_1_2": span_proj["proj_svec_1_2"],
        "proj_svec_1_5": span_proj["proj_svec_1_5"],
        "proj_svec_3_5": span_proj["proj_svec_3_5"],
        "top_rows": top_rows,
    }


def make_params(args):
    return {
        "q0": args.q0,
        "qmax": args.qmax,
        "krylov_depth": args.krylov_depth,
        "residual_tol": args.residual_tol,
        "expansion_maxit": args.expansion_maxit,
        "num_restarts": args.num_restarts,
        "maxit": args.maxit,
        "tol": args.tol,
        "post_expansion_maxit": args.post_expansion_maxit,
        "reduced_optimizer": "cex",
        "expansion_direction": "residual",
        "reuse_line_search_grad": True,
        "expansion_warm_start": True,
        "basis_selection": "joint",
        "joint_solver": "riemannian",
        "joint_warm_start_oracle": args.joint_warm_start_oracle,
        "row_concentration_lambda": args.row_concentration_lambda,
        "work_dtype": np.float64 if args.dtype == "float64" else np.float32,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect the geometry of the best joint-Stiefel restart seed on each streaming block."
    )
    parser.add_argument("--matrix", default="static-cex")
    parser.add_argument("--preset", default="cex-replicate")
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--win", type=int, default=32)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle-rows", action="store_true")
    parser.add_argument("--row-shuffle-seed", type=int)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--q0", type=int, default=4)
    parser.add_argument("--qmax", type=int, default=32)
    parser.add_argument("--krylov-depth", type=int, default=2)
    parser.add_argument("--residual-tol", type=float, default=1e-4)
    parser.add_argument("--expansion-maxit", type=int, default=8)
    parser.add_argument("--num-restarts", type=int, default=64)
    parser.add_argument("--maxit", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--post-expansion-maxit", type=int, default=500)
    parser.add_argument("--joint-warm-start-oracle", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--oracle-gradient-decomp", action="store_true")
    parser.add_argument("--oracle-tradeoff-table", action="store_true")
    parser.add_argument("--row-concentration-lambda", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    A, V_exact, _, sigma1 = probe.generate_matrix_input(
        matrix=args.matrix,
        n=args.n,
        preset=args.preset,
        seed=args.seed,
        shuffle_rows=args.shuffle_rows,
        row_shuffle_seed=args.row_shuffle_seed,
    )
    A = np.asarray(A, dtype=np.float64)
    params = make_params(args)

    state = None
    final_align = None
    final_relerr = None
    oracle_tradeoff_rows = []
    oracle_grad_rows = []
    print(
        "block | initial_best | final_best | seed_vs_solution | seed_vs_M_top | seed_vs_exact_top | "
        "found_vs_proj_exact_rowM | found_vs_M_top | exact_v1_proj | block_row | block_row_proj | "
        "gain_row | gain_row_proj | score_sum"
    )
    print(
        "----- | ------------ | ---------- | ---------------- | ------------- | ----------------- | "
        "------------------------ | -------------- | ------------- | --------- | -------------- | "
        "-------- | ------------- | ---------"
    )

    for block_idx, start0 in enumerate(range(0, A.shape[0], args.win)):
        end0 = min(start0 + args.win, A.shape[0])
        A_block = A[start0:end0, :]
        work_dtype = params["work_dtype"]
        A_block_work = A_block.astype(work_dtype, copy=False)
        if state is None:
            M_gain = A_block_work
            V_init = None
            rows_seen = A_block.shape[0]
        else:
            B_top = state["s"].astype(work_dtype)[:, None] * state["V"].astype(work_dtype).T
            M_gain = np.vstack([B_top, A_block_work]).astype(work_dtype, copy=False)
            V_init = state["V"].astype(work_dtype, copy=False)
            rows_seen = state["rows_seen"] + A_block.shape[0]

        V_score, H_score, score_score, diag = None, None, None, None
        V_score, _, H_score, score_score, diag = probe.entropy_iter_basis_forget(
            M_gain=M_gain,
            active_r=args.rank,
            rows_ref=A.shape[0],
            V_init=V_init,
            q0=params["q0"],
            qmax=params["qmax"],
            krylov_depth=params["krylov_depth"],
            residual_tol=params["residual_tol"],
            expansion_maxit=params["expansion_maxit"],
            num_restarts=params["num_restarts"],
            maxit=params["maxit"],
            tol=params["tol"],
            rng=np.random.default_rng(args.seed),
            verbose=False,
            state_prev=state,
            A_block=A_block_work,
            rows_total=rows_seen,
            reduced_optimizer=params["reduced_optimizer"],
            work_dtype=work_dtype,
            expansion_direction=params["expansion_direction"],
            reuse_line_search_grad=params["reuse_line_search_grad"],
            expansion_warm_start=params["expansion_warm_start"],
            post_expansion_maxit=params["post_expansion_maxit"],
            basis_selection=params["basis_selection"],
            joint_solver=params["joint_solver"],
            joint_warm_start_oracle=params["joint_warm_start_oracle"],
            oracle_warm_start_target=V_exact,
            row_concentration_lambda=params["row_concentration_lambda"],
        )

        history = diag["joint_seed_history"]
        initial = history[0]
        seed_frame = np.asarray(initial["best_seed_full"], dtype=np.float64)[:, : args.rank]
        solution_frame = np.asarray(initial["best_solution_full"], dtype=np.float64)[:, : args.rank]
        _, _, M_vh = np.linalg.svd(np.asarray(M_gain, dtype=np.float64), full_matrices=False)
        M_top = M_vh.T[:, : args.rank]
        exact_top = V_exact[:, : args.rank]
        seed_solution_cos = subspace_cosines(seed_frame, solution_frame)
        seed_m_cos = subspace_cosines(seed_frame, M_top)
        seed_exact_cos = subspace_cosines(seed_frame, exact_top)
        exact_v1_proj = float(np.linalg.norm(orth(seed_frame).T @ V_exact[:, 0]))
        block_row = row_proximity(seed_frame, A_block)
        gain_row = row_proximity(seed_frame, M_gain)

        _, s_new, Vt_new, _ = probe.left_projected_operator_svd_factors(V_score.T, M_gain)
        V_r = Vt_new.T
        _, _, M_vh_current = np.linalg.svd(np.asarray(M_gain, dtype=np.float64), full_matrices=False)
        row_basis_current = M_vh_current.T
        projected_exact = row_basis_current @ (row_basis_current.T @ V_exact[:, : args.rank])
        found_proj_exact_cos = subspace_cosines(V_r[:, : args.rank], projected_exact)
        found_m_top_cos = subspace_cosines(V_r[:, : args.rank], row_basis_current[:, : args.rank])
        align = np.linalg.norm((V_r @ V_r.T) @ V_exact[:, :1], "fro")
        relerr = abs(s_new[0] - sigma1) / sigma1
        final_align = float(align)
        final_relerr = float(relerr)

        if args.oracle_tradeoff_table or args.oracle_gradient_decomp:
            Q_oracle = oracle_frame(M_gain, V_exact, args.rank)
            exact_top = np.asarray(V_exact[:, : args.rank], dtype=np.float64)
            q_oracle_rank = Q_oracle.shape[1]

            if args.oracle_tradeoff_table:
                current_score = float(np.sum(score_score[: args.rank]))
                current_align = stiefel_align_to_target(V_r[:, : args.rank], exact_top)
                oracle_tradeoff_rows.append({
                    "block": f"{start0 + 1}:{end0}",
                    "candidate": "high-random-current",
                    "status": "ok",
                    "score_sum": current_score,
                    "align_exact": current_align,
                    "cos_qoracle": fmt_cos(subspace_cosines(V_r[:, : args.rank], Q_oracle)),
                })

                if q_oracle_rank >= args.rank:
                    oracle_score, _, _, _ = evaluate_joint_frame(
                        Q_oracle, state, A_block_work, M_gain, A.shape[0],
                        optimizer=params["reduced_optimizer"],
                        row_concentration_lambda=params["row_concentration_lambda"],
                    )
                    oracle_carry, _ = frame_carry(Q_oracle, M_gain)
                    oracle_tradeoff_rows.append({
                        "block": f"{start0 + 1}:{end0}",
                        "candidate": "direct-Q_oracle",
                        "status": "ok",
                        "score_sum": oracle_score,
                        "align_exact": stiefel_align_to_target(oracle_carry[:, : args.rank], exact_top),
                        "cos_qoracle": fmt_cos(subspace_cosines(oracle_carry[:, : args.rank], Q_oracle)),
                    })

                    if q_oracle_rank == args.rank:
                        if state is None:
                            basis = Q_oracle
                            Z0 = np.eye(args.rank, dtype=np.float64)
                        else:
                            basis = Q_oracle
                            Z0 = np.eye(args.rank, dtype=np.float64)
                        rot_frame, rot_score, _, _, _, _ = run_joint_from_basis(
                            basis, Z0, state, A_block_work, M_gain, A.shape[0],
                            optimizer=params["reduced_optimizer"], maxit=params["maxit"], tol=params["tol"],
                            reuse_line_search_grad=params["reuse_line_search_grad"],
                            row_concentration_lambda=params["row_concentration_lambda"],
                        )
                        rot_carry, _ = frame_carry(rot_frame, M_gain)
                        oracle_tradeoff_rows.append({
                            "block": f"{start0 + 1}:{end0}",
                            "candidate": "Q_oracle-rotate",
                            "status": "ok",
                            "score_sum": rot_score,
                            "align_exact": stiefel_align_to_target(rot_carry[:, : args.rank], exact_top),
                            "cos_qoracle": fmt_cos(subspace_cosines(rot_carry[:, : args.rank], Q_oracle)),
                        })

                    row_basis = orth(np.asarray(M_gain, dtype=np.float64).T)
                    if row_basis.shape[1] >= args.rank:
                        Z0 = row_basis.T @ Q_oracle
                        unrestricted_frame, unrestricted_score, _, _, _, _ = run_joint_from_basis(
                            row_basis, Z0, state, A_block_work, M_gain, A.shape[0],
                            optimizer=params["reduced_optimizer"], maxit=params["maxit"], tol=params["tol"],
                            reuse_line_search_grad=params["reuse_line_search_grad"],
                            row_concentration_lambda=params["row_concentration_lambda"],
                        )
                        unrestricted_carry, _ = frame_carry(unrestricted_frame, M_gain)
                        oracle_tradeoff_rows.append({
                            "block": f"{start0 + 1}:{end0}",
                            "candidate": "Q_oracle-unrestricted",
                            "status": "ok",
                            "score_sum": unrestricted_score,
                            "align_exact": stiefel_align_to_target(unrestricted_carry[:, : args.rank], exact_top),
                            "cos_qoracle": fmt_cos(subspace_cosines(unrestricted_carry[:, : args.rank], Q_oracle)),
                        })
                    else:
                        oracle_tradeoff_rows.append({
                            "block": f"{start0 + 1}:{end0}",
                            "candidate": "Q_oracle-unrestricted",
                            "status": "skip-rowspace",
                            "score_sum": np.nan,
                            "align_exact": np.nan,
                            "cos_qoracle": "n/a",
                        })
                else:
                    oracle_tradeoff_rows.extend([
                        {
                            "block": f"{start0 + 1}:{end0}",
                            "candidate": "direct-Q_oracle",
                            "status": "skip-rank",
                            "score_sum": np.nan,
                            "align_exact": np.nan,
                            "cos_qoracle": "n/a",
                        },
                        {
                            "block": f"{start0 + 1}:{end0}",
                            "candidate": "Q_oracle-rotate",
                            "status": "skip-rank",
                            "score_sum": np.nan,
                            "align_exact": np.nan,
                            "cos_qoracle": "n/a",
                        },
                        {
                            "block": f"{start0 + 1}:{end0}",
                            "candidate": "Q_oracle-unrestricted",
                            "status": "skip-rank",
                            "score_sum": np.nan,
                            "align_exact": np.nan,
                            "cos_qoracle": "n/a",
                        },
                    ])

                greedy_params = dict(params)
                greedy_params["basis_selection"] = "greedy"
                greedy_score_frame, _, _, greedy_score_vals, _ = probe.entropy_iter_basis_forget(
                    M_gain=M_gain,
                    active_r=args.rank,
                    rows_ref=A.shape[0],
                    V_init=V_init,
                    q0=greedy_params["q0"],
                    qmax=greedy_params["qmax"],
                    krylov_depth=greedy_params["krylov_depth"],
                    residual_tol=greedy_params["residual_tol"],
                    expansion_maxit=greedy_params["expansion_maxit"],
                    num_restarts=greedy_params["num_restarts"],
                    maxit=greedy_params["maxit"],
                    tol=greedy_params["tol"],
                    rng=np.random.default_rng(args.seed),
                    verbose=False,
                    state_prev=state,
                    A_block=A_block_work,
                    rows_total=rows_seen,
                    reduced_optimizer=greedy_params["reduced_optimizer"],
                    work_dtype=work_dtype,
                    expansion_direction=greedy_params["expansion_direction"],
                    reuse_line_search_grad=greedy_params["reuse_line_search_grad"],
                    expansion_warm_start=greedy_params["expansion_warm_start"],
                    post_expansion_maxit=greedy_params["post_expansion_maxit"],
                    basis_selection=greedy_params["basis_selection"],
                    row_concentration_lambda=greedy_params["row_concentration_lambda"],
                )
                greedy_carry, _ = frame_carry(greedy_score_frame, M_gain)
                oracle_tradeoff_rows.append({
                    "block": f"{start0 + 1}:{end0}",
                    "candidate": "old-greedy",
                    "status": "ok",
                    "score_sum": float(np.sum(greedy_score_vals[: args.rank])),
                    "align_exact": stiefel_align_to_target(greedy_carry[:, : args.rank], exact_top),
                    "cos_qoracle": fmt_cos(subspace_cosines(greedy_carry[:, : args.rank], Q_oracle)),
                })

            if args.oracle_gradient_decomp:
                if q_oracle_rank >= args.rank:
                    oracle_grad_rows.append({
                        "block": f"{start0 + 1}:{end0}",
                        **dominant_outside_projection_summary(
                            Q_oracle[:, : args.rank], state, A_block_work, M_gain, V_exact, A.shape[0],
                            row_concentration_lambda=params["row_concentration_lambda"],
                        ),
                    })
                else:
                    oracle_grad_rows.append({
                        "block": f"{start0 + 1}:{end0}",
                        "tangent_norm": np.nan,
                        "inside_norm": np.nan,
                        "outside_norm": np.nan,
                        "proj_exact_v3": np.nan,
                        "proj_exact_v4": np.nan,
                        "proj_svec_1_2": np.nan,
                        "proj_svec_1_5": np.nan,
                        "proj_svec_3_5": np.nan,
                        "top_rows": "n/a",
                    })

        state = {
            "V": V_r,
            "s": s_new,
            "s2": s_new ** 2,
            "H": np.asarray(H_score[: len(s_new)], dtype=np.float32),
            "score": np.asarray(score_score[: len(s_new)], dtype=np.float32),
            "rows_seen": rows_seen,
            "diag": diag,
        }

        print(
            f"{start0 + 1}:{end0} | "
            f"{initial['best_seed_label']}#{initial['best_restart']} | "
            f"{diag['joint_best_seed_label']}#{diag['joint_best_restart']} | "
            f"{fmt_cos(seed_solution_cos)} | "
            f"{fmt_cos(seed_m_cos)} | "
            f"{fmt_cos(seed_exact_cos)} | "
            f"{fmt_cos(found_proj_exact_cos)} | "
            f"{fmt_cos(found_m_top_cos)} | "
            f"{exact_v1_proj:.6f} | "
            f"{block_row['idx']} | {block_row['projection']:.6f} | "
            f"{gain_row['idx']} | {gain_row['projection']:.6f} | "
            f"{float(np.sum(score_score[: args.rank])):.6f}"
        )

    print(f"\nfinal_align={final_align:.6f} final_relerr={final_relerr:.8f}")

    if args.oracle_tradeoff_table and oracle_tradeoff_rows:
        print("\noracle_tradeoff_table")
        print("block | candidate | status | score_sum | align_exact | cos_qoracle")
        print("----- | --------- | ------ | --------- | ----------- | -----------")
        for row in oracle_tradeoff_rows:
            print(
                f"{row['block']} | {row['candidate']} | {row['status']} | "
                f"{row['score_sum']:.6f} | {row['align_exact']:.6f} | {row['cos_qoracle']}"
            )

    if args.oracle_gradient_decomp and oracle_grad_rows:
        print("\noracle_gradient_decomp")
        print(
            "block | tangent_norm | inside_norm | outside_norm | proj_exact_v3 | proj_exact_v4 | "
            "proj_svec_1_2 | proj_svec_1_5 | proj_svec_3_5 | top_rows"
        )
        print(
            "----- | ------------ | ----------- | ------------ | ------------- | ------------- | "
            "------------- | ------------- | ------------- | --------"
        )
        for row in oracle_grad_rows:
            print(
                f"{row['block']} | {row['tangent_norm']:.6f} | {row['inside_norm']:.6f} | "
                f"{row['outside_norm']:.6f} | "
                f"{row['proj_exact_v3']:.6f} | {row['proj_exact_v4']:.6f} | "
                f"{row['proj_svec_1_2']:.6f} | {row['proj_svec_1_5']:.6f} | "
                f"{row['proj_svec_3_5']:.6f} | {row['top_rows']}"
            )


if __name__ == "__main__":
    main()
