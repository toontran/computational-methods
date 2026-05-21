import argparse
from types import SimpleNamespace

import numpy as np

import cex_restricted_space_probe as probe
import init_geometry_probe as geom


FAST_PARAMS = {
    "q0": 5,
    "qmax": 200,
    "krylov_depth": 2,
    "residual_tol": 1e-2,
    "expansion_maxit": 64,
    "num_restarts": 2,
    "maxit": 120,
    "tol": 1e-8,
    "post_expansion_maxit": 60,
    "basis_selection": "greedy",
    "joint_warm_start_greedy": False,
    "joint_warm_start_oracle": True,
    "joint_warm_start_rotations": 0,
    "joint_warm_start_perturbations": 0,
    "joint_oversample": 0,
    "joint_solver": "riemannian",
    "row_concentration_lambda": 0.0,
    "row_leverage_lambda": 0.0,
    "row_leverage_mode": "none",
    "row_leverage_rank": 2,
}


def fmt(x):
    if isinstance(x, str):
        return x
    if x is None:
        return ""
    x = float(x)
    if not np.isfinite(x):
        return "nan"
    return f"{x:.6g}"


def generate_static_cex(args):
    np.random.seed(args.seed)
    A, V_exact, _, sigma1 = probe.generate_matrix_input(
        matrix="static-cex",
        n=args.n,
        preset=args.matrix_preset,
        seed=args.seed,
        r_sig=2,
        alpha_sig=0.003,
        alpha_tail=0.0145,
        tail_scale=0.99,
        sigma1=0.991,
        v_type="rand",
    )
    return np.asarray(A, dtype=np.float64), np.asarray(V_exact, dtype=np.float64), float(sigma1)


def first_block_row_perm(n, seed):
    rng_state = np.random.get_state()
    np.random.seed(seed)
    _ = np.linalg.qr(np.random.randn(n, n), mode="reduced")
    row_perm = np.random.permutation(n)
    np.random.set_state(rng_state)
    return row_perm


def run_entropy(A, V_exact, args, variant, stop_after_first=False):
    work_dtype = np.float64 if variant.get("dtype", args.dtype) == "float64" else np.float32
    params = dict(FAST_PARAMS)
    params.update(variant)
    state = None
    first = None
    final = None
    row_perm = first_block_row_perm(args.n, args.seed)

    for block_idx, start0 in enumerate(range(0, A.shape[0], args.win)):
        end0 = min(start0 + args.win, A.shape[0])
        A_block = np.asarray(A[start0:end0, :], dtype=work_dtype)
        A_block_opt = A_block
        row_weights = None
        if state is None and variant.get("downweight_orig_rows"):
            row_weights = np.ones(A_block.shape[0], dtype=work_dtype)
            for orig_row in variant["downweight_orig_rows"]:
                hits = np.where(row_perm[start0:end0] == int(orig_row) - 1)[0]
                row_weights[hits] = float(variant.get("downweight_factor", 1.0))
            A_block_opt = A_block * row_weights[:, None]
        if state is None:
            M_gain = A_block_opt
            V_init = None
            rows_seen = A_block.shape[0]
            old_rows = None
        else:
            B_top = state["s"].astype(work_dtype)[:, None] * state["V"].astype(work_dtype).T
            M_gain = np.vstack([B_top, A_block_opt]).astype(work_dtype, copy=False)
            V_init = state["V"].astype(work_dtype, copy=False)
            rows_seen = state["rows_seen"] + A_block.shape[0]
            old_rows = None

        oracle_rows = None
        if params.get("oracle_sketch_all_seen_rows", False):
            oracle_rows = A[:end0, :].astype(work_dtype, copy=False)

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
            A_block=A_block_opt,
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
            old_row_memory=old_rows,
            oracle_projection_row_samples=oracle_rows,
        )

        _, s_new, Vt_new, _ = probe.left_projected_operator_svd_factors(V_score.T, M_gain)
        V_r = np.asarray(Vt_new.T, dtype=np.float64)
        Q_oracle = geom.oracle_frame(np.asarray(M_gain, dtype=np.float64), V_exact, args.rank)
        selected = np.asarray(V_score[:, : args.rank], dtype=np.float64)
        cos = probe.subspace_principal_cosines(selected, Q_oracle[:, : args.rank])
        outside = []
        v3_abs = []
        row1_frac = []
        if Q_oracle.shape[1] >= args.rank:
            for j in range(args.rank):
                v = selected[:, j]
                out = v - Q_oracle[:, : args.rank] @ (Q_oracle[:, : args.rank].T @ v)
                out_norm = float(np.linalg.norm(out))
                outside.append(out_norm)
                if out_norm > 1e-12:
                    w = out / out_norm
                    v3_abs.append(float(abs(w @ V_exact[:, 2])) if V_exact.shape[1] > 2 else np.nan)
                    y = np.asarray(A[start0:end0, :] @ w, dtype=float)
                    energy = float(np.dot(y, y))
                    hits = np.where(row_perm[start0:end0] == 0)[0]
                    if len(hits):
                        row1_frac.append(float(y[int(hits[0])] ** 2 / max(energy, 1e-30)))
                    else:
                        row1_frac.append(0.0)
                else:
                    v3_abs.append(0.0)
                    row1_frac.append(0.0)

        block_summary = {
            "label": params["label"],
            "block": block_idx + 1,
            "basis": params["basis_selection"],
            "lambda": params["row_concentration_lambda"],
            "q0": params["q0"],
            "qmax": params["qmax"],
            "score_sum": float(np.sum(score_score[: args.rank])),
            "reg_score_sum": float(diag["regularized_score_sum"]),
            "cos_min": float(np.min(cos)) if len(cos) else np.nan,
            "outside_1": outside[0] if len(outside) > 0 else np.nan,
            "outside_2": outside[1] if len(outside) > 1 else np.nan,
            "v3_out_2": v3_abs[1] if len(v3_abs) > 1 else np.nan,
            "orig_row1_frac_2": row1_frac[1] if len(row1_frac) > 1 else np.nan,
            "best_seed": diag.get("joint_best_seed_label", ""),
        }
        if first is None:
            first = block_summary
        final = block_summary
        final["final_align"] = float(
            np.linalg.norm((V_r[:, : args.rank] @ V_r[:, : args.rank].T) @ V_exact[:, :1], "fro")
        )
        final["final_relerr"] = float(abs(s_new[0] - args.sigma1) / args.sigma1)

        state = {
            "V": V_r,
            "s": s_new,
            "s2": s_new ** 2,
            "H": np.asarray(H_score[: len(s_new)], dtype=np.float32),
            "score": np.asarray(score_score[: len(s_new)], dtype=np.float32),
            "rows_seen": rows_seen,
            "diag": diag,
        }
        if stop_after_first:
            break

    return first, final


def print_table(title, rows, headers):
    print(title)
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(fmt(row.get(h))))
    print(" | ".join(h.ljust(widths[h]) for h in headers))
    print(" | ".join("-" * widths[h] for h in headers))
    for row in rows:
        print(" | ".join(fmt(row.get(h)).ljust(widths[h]) for h in headers))
    print()


def main():
    parser = argparse.ArgumentParser(description="Try optimizer steering variants for first-block outside directions.")
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--win", type=int, default=32)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--matrix-preset", default="small")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--full-stream", action="store_true")
    args = parser.parse_args()

    A, V_exact, sigma1 = generate_static_cex(args)
    args.sigma1 = sigma1

    variants = [
        {"label": "baseline-greedy-oraclewarm"},
        {"label": "rowconc-0.01", "row_concentration_lambda": 0.01},
        {"label": "rowconc-0.03", "row_concentration_lambda": 0.03},
        {"label": "rowconc-0.1", "row_concentration_lambda": 0.1},
        {"label": "rowconc-0.2", "row_concentration_lambda": 0.2},
        {"label": "rowconc-0.3", "row_concentration_lambda": 0.3},
        {"label": "rowconc-0.5", "row_concentration_lambda": 0.5},
        {"label": "rowconc-1.0", "row_concentration_lambda": 1.0},
        {"label": "rowlev-norm-0.001", "row_leverage_lambda": 0.001, "row_leverage_mode": "row-norm"},
        {"label": "rowlev-norm-0.003", "row_leverage_lambda": 0.003, "row_leverage_mode": "row-norm"},
        {"label": "rowlev-norm-0.01", "row_leverage_lambda": 0.01, "row_leverage_mode": "row-norm"},
        {"label": "rowlev-norm-0.03", "row_leverage_lambda": 0.03, "row_leverage_mode": "row-norm"},
        {"label": "rowlev-top2-0.001", "row_leverage_lambda": 0.001, "row_leverage_mode": "top-svd", "row_leverage_rank": 2},
        {"label": "rowlev-top2-0.003", "row_leverage_lambda": 0.003, "row_leverage_mode": "top-svd", "row_leverage_rank": 2},
        {"label": "rowlev-top2-0.01", "row_leverage_lambda": 0.01, "row_leverage_mode": "top-svd", "row_leverage_rank": 2},
        {"label": "rowlev-top2-0.03", "row_leverage_lambda": 0.03, "row_leverage_mode": "top-svd", "row_leverage_rank": 2},
        {"label": "joint-oraclewarm", "basis_selection": "joint", "num_restarts": 8, "joint_warm_start_greedy": True},
        {"label": "limited-q32", "qmax": 32, "expansion_maxit": 8, "post_expansion_maxit": None},
        {"label": "svd-seed-only", "joint_warm_start_oracle": False, "q0": 32, "qmax": 32, "expansion_maxit": 0, "num_restarts": 8},
        {"label": "downweight-orig1-0.5", "downweight_orig_rows": [1], "downweight_factor": 0.5},
        {"label": "downweight-orig1-0.25", "downweight_orig_rows": [1], "downweight_factor": 0.25},
        {"label": "downweight-orig1-0.0", "downweight_orig_rows": [1], "downweight_factor": 0.0},
    ]

    first_rows = []
    final_rows = []
    for variant in variants:
        first, final = run_entropy(A, V_exact, args, variant, stop_after_first=not args.full_stream)
        first_rows.append(first)
        final_rows.append(final)

    first_headers = [
        "label",
        "score_sum",
        "reg_score_sum",
        "cos_min",
        "outside_1",
        "outside_2",
        "v3_out_2",
        "orig_row1_frac_2",
        "best_seed",
    ]
    print_table("first-block steering metrics", first_rows, first_headers)

    if args.full_stream:
        final_headers = ["label", "final_align", "final_relerr", "cos_min", "outside_1", "outside_2", "score_sum"]
        print_table("final-block streaming metrics", final_rows, final_headers)


if __name__ == "__main__":
    main()
