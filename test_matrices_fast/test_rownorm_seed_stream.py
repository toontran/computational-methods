"""End-to-end streaming test: row-norm-normalized block SVD as first-block seed.

Reports per-block:
  score_sum, cos_min vs oracle, opt_proj_norms (v1_proj, v2_proj onto V_score),
  outside_1, outside_2, orig_row1_frac (first block only).
"""
import argparse
import numpy as np

import cex_restricted_space_probe as probe
import init_geometry_probe as geom
import try_outside_direction_steering as tods


def row_norm_seed(A_block, rank):
    """Row-L2-normalized top-r right singular vectors of A_block."""
    row_norms = np.linalg.norm(A_block, axis=1, keepdims=True)
    safe = np.where(row_norms > 0, row_norms, 1.0)
    A_rn = A_block / safe
    _, _, Vt = np.linalg.svd(A_rn, full_matrices=False)
    return np.ascontiguousarray(Vt.T[:, :rank])


def run_stream(A, V_exact, args, variant):
    params = dict(tods.FAST_PARAMS)
    params.update(variant)
    work_dtype = np.float64 if params.get("dtype", args.dtype) == "float64" else np.float32
    state = None
    row_perm = tods.first_block_row_perm(args.n, args.seed)
    blocks = []

    for block_idx, start0 in enumerate(range(0, A.shape[0], args.win)):
        end0 = min(start0 + args.win, A.shape[0])
        A_block = np.asarray(A[start0:end0, :], dtype=work_dtype)

        if state is None:
            M_gain = A_block
            rows_seen = A_block.shape[0]
            if variant.get("rownorm_seed_first_block", False) or variant.get("rownorm_seed_all_blocks", False):
                V_init = np.asarray(row_norm_seed(A_block, args.rank), dtype=work_dtype)
            else:
                V_init = None
        else:
            B_top = state["s"].astype(work_dtype)[:, None] * state["V"].astype(work_dtype).T
            M_gain = np.vstack([B_top, A_block]).astype(work_dtype, copy=False)
            if variant.get("rownorm_seed_all_blocks", False):
                V_init = np.asarray(row_norm_seed(A_block, args.rank), dtype=work_dtype)
            else:
                V_init = state["V"].astype(work_dtype, copy=False)
            rows_seen = state["rows_seen"] + A_block.shape[0]

        V_score, s_score, H_score, score_score, diag = probe.entropy_iter_basis_forget(
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
            A_block=A_block,
            rows_total=rows_seen,
            reduced_optimizer="cex",
            work_dtype=work_dtype,
            expansion_direction=params.get("expansion_direction", "residual"),
            reuse_line_search_grad=True,
            expansion_warm_start=params.get("expansion_warm_start", True),
            post_expansion_maxit=params["post_expansion_maxit"],
            basis_selection=params["basis_selection"],
            joint_warm_start_greedy=params["joint_warm_start_greedy"],
            joint_warm_start_oracle=params["joint_warm_start_oracle"],
            oracle_warm_start_target=V_exact,
            joint_warm_start_rotations=params["joint_warm_start_rotations"],
            joint_warm_start_perturbations=params["joint_warm_start_perturbations"],
            joint_oversample=params["joint_oversample"],
            joint_solver=params["joint_solver"],
            row_concentration_lambda=params["row_concentration_lambda"],
            row_leverage_lambda=params["row_leverage_lambda"],
            row_leverage_mode=params["row_leverage_mode"],
            row_leverage_rank=params["row_leverage_rank"],
            score_variant="combined",
            old_row_memory=None,
            oracle_projection_row_samples=None,
        )

        # Post: projected oracle diagnostic on this block's M_gain
        diag_dtype = np.float64
        M_d = np.asarray(M_gain, dtype=diag_dtype)
        V_exact_d = np.asarray(V_exact, dtype=diag_dtype)
        Q_oracle, Q_row = probe.projected_true_span_oracle(M_d, V_exact_d, args.rank, dtype=diag_dtype)
        raw_cols = []
        for j in range(args.rank):
            vp = probe.project_onto_span(V_exact_d[:, j], Q_row).reshape(-1)
            vn = float(np.linalg.norm(vp))
            if vn > 1e-30:
                raw_cols.append(vp / vn)
        raw_proj = np.column_stack(raw_cols) if raw_cols else np.zeros((M_d.shape[1], 0))
        V_opt = probe.orthonormalize_columns(np.asarray(V_score[:, : args.rank], dtype=diag_dtype), dtype=diag_dtype)
        opt_proj_norms = np.linalg.norm(V_opt @ (V_opt.T @ raw_proj), axis=0) if raw_proj.size else np.zeros(0)
        pc = probe.subspace_principal_cosines(V_opt, Q_oracle)

        selected = np.asarray(V_score[:, : args.rank], dtype=diag_dtype)
        outside = []
        v3_abs = []
        row1_frac = []
        for j in range(args.rank):
            v = selected[:, j]
            out = v - Q_oracle @ (Q_oracle.T @ v)
            on = float(np.linalg.norm(out))
            outside.append(on)
            if on > 1e-12 and block_idx == 0:
                w = out / on
                if V_exact.shape[1] > 2:
                    v3_abs.append(float(abs(w @ V_exact_d[:, 2])))
                else:
                    v3_abs.append(np.nan)
                y = np.asarray(A[start0:end0, :] @ w)
                energy = float(np.dot(y, y))
                hits = np.where(row_perm[start0:end0] == 0)[0]
                row1_frac.append(float(y[int(hits[0])] ** 2 / max(energy, 1e-30)) if len(hits) else 0.0)
            else:
                v3_abs.append(np.nan)
                row1_frac.append(np.nan)

        # Update state from post-selection SVD
        _, s_new, Vt_new, _ = probe.left_projected_operator_svd_factors(V_score.T, M_gain)
        V_r = np.asarray(Vt_new.T, dtype=np.float64)

        final_align = float(np.linalg.norm((V_r[:, : args.rank] @ V_r[:, : args.rank].T) @ V_exact[:, :1], "fro"))
        final_relerr = float(abs(s_new[0] - args.sigma1) / args.sigma1)

        blocks.append({
            "block": block_idx + 1,
            "score_sum": float(np.sum(score_score[: args.rank])),
            "cos_min": float(np.min(pc)) if pc.size else np.nan,
            "opt_proj_v1": float(opt_proj_norms[0]) if opt_proj_norms.size > 0 else np.nan,
            "opt_proj_v2": float(opt_proj_norms[1]) if opt_proj_norms.size > 1 else np.nan,
            "outside_1": outside[0] if len(outside) > 0 else np.nan,
            "outside_2": outside[1] if len(outside) > 1 else np.nan,
            "v3_out_2": v3_abs[1] if len(v3_abs) > 1 else np.nan,
            "row1_frac_2": row1_frac[1] if len(row1_frac) > 1 else np.nan,
            "final_align": final_align,
            "final_relerr": final_relerr,
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

    return blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--win", type=int, default=32)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--matrix-preset", default="small")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()

    A, V_exact, sigma1 = tods.generate_static_cex(args)
    args.sigma1 = sigma1

    variants = [
        ("greedy oraclewarm-ON",                                  {}),
        ("greedy oraclewarm-OFF",                                 {"joint_warm_start_oracle": False}),
        ("greedy oraclewarm-OFF + rownorm",                       {"joint_warm_start_oracle": False, "rownorm_seed_first_block": True}),
        ("greedy oraclewarm-OFF + rownorm-all",                   {"joint_warm_start_oracle": False, "rownorm_seed_all_blocks": True}),
        ("joint(greedy-warm) oraclewarm-ON",                      {"basis_selection": "joint", "joint_warm_start_greedy": True, "num_restarts": 2}),
        ("joint(greedy-warm) oraclewarm-OFF",                     {"basis_selection": "joint", "joint_warm_start_greedy": True, "joint_warm_start_oracle": False, "num_restarts": 2}),
        ("joint(greedy-warm) oraclewarm-OFF + rownorm",           {"basis_selection": "joint", "joint_warm_start_greedy": True, "joint_warm_start_oracle": False, "num_restarts": 2, "rownorm_seed_first_block": True}),
        ("joint(greedy-warm+restarts8) oraclewarm-ON",            {"basis_selection": "joint", "joint_warm_start_greedy": True, "num_restarts": 8}),
        ("joint(greedy-warm+restarts8) oraclewarm-OFF + rownorm", {"basis_selection": "joint", "joint_warm_start_greedy": True, "joint_warm_start_oracle": False, "num_restarts": 8, "rownorm_seed_first_block": True}),
    ]

    for label, variant in variants:
        print(f"=== {label} ===")
        blocks = run_stream(A, V_exact, args, variant)
        hdr = f"{'blk':>3} | {'score_sum':>9} | {'cos_min':>7} | {'proj_v1':>7} | {'proj_v2':>7} | {'out_1':>7} | {'out_2':>7} | {'v3_out2':>7} | {'row1_fr':>7} | {'final_align':>11} | {'final_relerr':>12}"
        print(hdr)
        print("-" * len(hdr))
        for b in blocks:
            def f(x, w, p=4):
                if x is None or (isinstance(x, float) and not np.isfinite(x)):
                    return " " * w
                return f"{x:{w}.{p}f}"
            print(f"{b['block']:>3} | {f(b['score_sum'],9,5)} | {f(b['cos_min'],7,4)} | {f(b['opt_proj_v1'],7,4)} | {f(b['opt_proj_v2'],7,4)} | {f(b['outside_1'],7,4)} | {f(b['outside_2'],7,4)} | {f(b['v3_out_2'],7,4)} | {f(b['row1_frac_2'],7,4)} | {f(b['final_align'],11,6)} | {f(b['final_relerr'],12,7)}")
        print()


if __name__ == "__main__":
    main()
