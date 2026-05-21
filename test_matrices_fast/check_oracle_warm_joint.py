import argparse

import numpy as np

import cex_restricted_space_probe as probe


def fmt(x):
    if isinstance(x, str):
        return x
    return f"{float(x):.12g}"


def frame_metrics(label, A_block, Vbasis, Z, Q_oracle, V_exact, rows_ref, maxit):
    V = np.asarray(Vbasis @ Z, dtype=np.float64)
    start_total, start_vals, _, _, _ = probe.entropyscore_forget_joint_reduced_eval(
        A_block @ Vbasis,
        Z,
        A_block.shape[0],
        rows_ref,
        optimizer="cex",
        score_variant="combined",
    )
    Z_opt, opt_total, opt_vals, _, _, stop = probe.basic_projected_ascent_joint_reduced_forget(
        A_block @ Vbasis,
        Z,
        np.zeros((Vbasis.shape[1], 0), dtype=Vbasis.dtype),
        A_block.shape[0],
        rows_ref,
        maxit=maxit,
        tol=1e-8,
        optimizer="cex",
        reuse_line_search_grad=True,
        score_variant="combined",
    )
    V_opt = np.asarray(Vbasis @ Z_opt, dtype=np.float64)
    cos_start = probe.subspace_principal_cosines(V, Q_oracle[:, :2])
    cos_opt = probe.subspace_principal_cosines(V_opt, Q_oracle[:, :2])

    outside = []
    v3 = []
    for j in range(2):
        out = V_opt[:, j] - Q_oracle[:, :2] @ (Q_oracle[:, :2].T @ V_opt[:, j])
        nout = float(np.linalg.norm(out))
        outside.append(nout)
        v3.append(float(abs((out / nout) @ V_exact[:, 2])) if nout > 1e-12 else 0.0)

    return {
        "label": label,
        "start_score": start_total,
        "opt_score": opt_total,
        "score_gain": opt_total - start_total,
        "start_cos_min": float(np.min(cos_start)),
        "opt_cos_min": float(np.min(cos_opt)),
        "outside_1": outside[0],
        "outside_2": outside[1],
        "v3_out_2": v3[1],
        "stop": stop["reason"],
        "iters": stop["iters"],
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect first-block oracle-warm joint starts.")
    parser.add_argument("--q0", type=int, default=5)
    parser.add_argument("--maxit", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    A, V_exact, _, _ = probe.generate_matrix_input(
        matrix="static-cex",
        n=128,
        preset="small",
        seed=args.seed,
        r_sig=2,
        alpha_sig=0.003,
        alpha_tail=0.0145,
        tail_scale=0.99,
        sigma1=0.991,
        v_type="rand",
    )
    A = np.asarray(A, dtype=np.float64)
    V_exact = np.asarray(V_exact, dtype=np.float64)
    A_block = np.asarray(A[:32, :], dtype=np.float32)
    rows_ref = A.shape[0]

    if args.q0 >= min(A_block.shape):
        _, _, vh = np.linalg.svd(A_block, full_matrices=False)
        Vbasis = np.ascontiguousarray(vh[: args.q0, :].T, dtype=A_block.dtype)
    else:
        Vbasis, _, _ = probe.build_entropy_fast_subspace(
            A_block, active_r=2, q_subspace=args.q0, method="lanczos", dtype=A_block.dtype
        )

    Q_oracle, _ = probe.projected_true_span_oracle(A_block, V_exact[:, :2], 2, dtype=np.float64)
    oracle_score, oracle_vals = (
        lambda x: (x[0], x[1])
    )(
        probe.entropyscore_forget_joint_reduced_eval(
            A_block.astype(np.float64),
            Q_oracle[:, :2],
            A_block.shape[0],
            rows_ref,
            optimizer="cex",
            score_variant="combined",
        )
    )

    Z_oracle = probe.make_oracle_stiefel_warm_start(
        A_block, Vbasis, V_exact, joint_rank=2, active_r=2, Qz=np.zeros((Vbasis.shape[1], 0), dtype=Vbasis.dtype)
    )
    Z_eye = np.zeros((Vbasis.shape[1], 2), dtype=Vbasis.dtype)
    Z_eye[:2, :2] = np.eye(2, dtype=Vbasis.dtype)
    Z_eye = probe.retract_stiefel_reduced(Z_eye, np.zeros((Vbasis.shape[1], 0), dtype=Vbasis.dtype))

    rows = [
        frame_metrics("oracle-projected", A_block, Vbasis, Z_oracle, Q_oracle, V_exact, rows_ref, args.maxit),
        frame_metrics("subspace-svd-eye", A_block, Vbasis, Z_eye, Q_oracle, V_exact, rows_ref, args.maxit),
    ]

    print(f"q0={args.q0}")
    print(f"exact_Q_oracle_score={oracle_score:.12f} vals={oracle_vals}")
    print(f"Q_oracle_column_overlap={abs(float(Q_oracle[:, 0] @ Q_oracle[:, 1])):.12g}")
    headers = [
        "label",
        "start_score",
        "opt_score",
        "score_gain",
        "start_cos_min",
        "opt_cos_min",
        "outside_1",
        "outside_2",
        "v3_out_2",
        "stop",
        "iters",
    ]
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(fmt(row[h])))
    print(" | ".join(h.ljust(widths[h]) for h in headers))
    print(" | ".join("-" * widths[h] for h in headers))
    for row in rows:
        print(" | ".join(fmt(row[h]).ljust(widths[h]) for h in headers))


if __name__ == "__main__":
    main()
