import numpy as np

from cex_restricted_space_probe import (
    entropyscore_forget_score_grad_reduced,
    entropyscore_forget_streaming_score_grad_reduced,
    orthonormalize_columns,
)


def project_q(Y, Qz):
    Y = np.asarray(Y)
    if Qz is None or np.asarray(Qz).size == 0:
        return np.ascontiguousarray(Y)
    Q = np.asarray(Qz, dtype=Y.dtype)
    return np.ascontiguousarray(Y - Q @ (Q.T @ Y), dtype=Y.dtype)


def stiefel_retract(Y, Qz=None, eps=1e-12):
    Yp = project_q(Y, Qz)
    Q, R = np.linalg.qr(np.asarray(Yp, dtype=np.float64), mode="reduced")
    if Q.shape[1] == 0 or np.min(np.abs(np.diag(R))) <= eps:
        return None
    signs = np.sign(np.diag(R))
    signs[signs == 0.0] = 1.0
    Q = Q * signs[None, :]
    return np.ascontiguousarray(Q[:, : Yp.shape[1]], dtype=Yp.dtype)


def random_stiefel(q, r, rng, Qz=None, dtype=np.float64):
    for _ in range(100):
        Z = np.asarray(rng.standard_normal((q, r)), dtype=dtype)
        Z = stiefel_retract(Z, Qz)
        if Z is not None and Z.shape == (q, r):
            return Z
    raise RuntimeError("Could not generate a random feasible Stiefel frame.")


def complete_from_first(z1, r, rng, Qz=None):
    z1 = np.asarray(z1).reshape(-1, 1)
    q = z1.shape[0]
    Q_forbid = z1 if Qz is None or np.asarray(Qz).size == 0 else np.column_stack([Qz, z1])
    cols = [z1[:, 0]]
    for _ in range(max(0, r - 1)):
        z = random_stiefel(q, 1, rng, Q_forbid, dtype=z1.dtype)[:, 0]
        cols.append(z)
        Q_forbid = orthonormalize_columns(np.column_stack([Q_forbid, z]), dtype=z1.dtype)
    return stiefel_retract(np.column_stack(cols), Qz)


def reduced_score_grad(problem, Z):
    scores = []
    grads = []
    svals = []
    entropies = []
    for j in range(Z.shape[1]):
        z = np.ascontiguousarray(Z[:, j], dtype=problem["dtype"])
        if problem["state_prev"] is None:
            score, grad, s, H = entropyscore_forget_score_grad_reduced(
                problem["B_block"], z, problem["rows_block"], problem["rows_ref"]
            )
        else:
            score, grad, s, H = entropyscore_forget_streaming_score_grad_reduced(
                problem["B_gain"],
                problem["B_block"],
                problem["C_prev"],
                problem["s2_old"],
                z,
                problem["rows_block"],
                problem["rows_ref"],
            )
        scores.append(float(score))
        grads.append(np.asarray(grad, dtype=problem["dtype"]))
        svals.append(float(s))
        entropies.append(float(H))
    return (
        float(np.sum(scores)),
        np.asarray(scores, dtype=float),
        np.column_stack(grads),
        np.asarray(svals, dtype=float),
        np.asarray(entropies, dtype=float),
    )


def stiefel_gradient(Z, G, Qz=None):
    Gp = project_q(G, Qz)
    sym = 0.5 * (Z.T @ Gp + Gp.T @ Z)
    return np.ascontiguousarray(Gp - Z @ sym, dtype=Z.dtype)


def optimize_joint(problem, Z0, maxit=200, tol=1e-8, armijo=1e-4, max_ls=25):
    Z = stiefel_retract(Z0, problem.get("Qz"))
    if Z is None:
        raise RuntimeError("Initial joint frame is infeasible.")
    F, scores, G, svals, entropies = reduced_score_grad(problem, Z)
    history = []
    stop = {"reason": "maxit", "iters": int(maxit), "grad_norm": np.nan}

    for it in range(maxit):
        grad = stiefel_gradient(Z, G, problem.get("Qz"))
        grad_norm = float(np.linalg.norm(grad, "fro"))
        if grad_norm <= tol:
            stop = {"reason": "grad_tol", "iters": it, "grad_norm": grad_norm}
            break

        alpha = 1.0
        accepted = False
        rejected = 0
        F_old = F
        Z_old = Z
        eval_old = (scores, G, svals, entropies)
        for _ in range(max_ls):
            Zt = stiefel_retract(Z_old + alpha * grad, problem.get("Qz"))
            if Zt is not None:
                Ft, scores_t, G_t, svals_t, entropies_t = reduced_score_grad(problem, Zt)
                if Ft >= F_old + armijo * alpha * grad_norm * grad_norm:
                    Z = Zt
                    F = Ft
                    scores = scores_t
                    G = G_t
                    svals = svals_t
                    entropies = entropies_t
                    accepted = True
                    break
            rejected += 1
            alpha *= 0.5

        improvement = float(F - F_old) if accepted else 0.0
        history.append(
            {
                "iter": it + 1,
                "F": float(F),
                "grad_norm": grad_norm,
                "alpha": float(alpha),
                "improvement": improvement,
                "rejected": int(rejected),
            }
        )

        if not accepted:
            scores, G, svals, entropies = eval_old
            Z = Z_old
            stop = {
                "reason": "line_search_fail",
                "iters": it + 1,
                "grad_norm": grad_norm,
                "line_search_steps": int(max_ls),
            }
            break
        if abs(improvement) <= 1e-12 * max(1.0, abs(F_old)):
            stop = {"reason": "f_change_tol", "iters": it + 1, "grad_norm": grad_norm}
            break
    else:
        grad = stiefel_gradient(Z, G, problem.get("Qz"))
        stop = {"reason": "maxit", "iters": int(maxit), "grad_norm": float(np.linalg.norm(grad, "fro"))}

    return {
        "Z": Z,
        "F": float(F),
        "scores": scores,
        "s": svals,
        "H": entropies,
        "G": G,
        "grad": stiefel_gradient(Z, G, problem.get("Qz")),
        "grad_norm": float(np.linalg.norm(stiefel_gradient(Z, G, problem.get("Qz")), "fro")),
        "history": history,
        "stop": stop,
    }


def principal_angles(Za, Zb):
    Qa = orthonormalize_columns(Za, dtype=np.float64)
    Qb = orthonormalize_columns(Zb, dtype=np.float64)
    if Qa.size == 0 or Qb.size == 0:
        return np.zeros(0)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.degrees(np.arccos(s))


def coupling_matrix(problem, Z):
    _, _, G, _, _ = reduced_score_grad(problem, Z)
    return np.asarray(G.T @ G, dtype=float)


def perturbation_probe(problem, Z, epsilons, trials, rng):
    base, _, _, _, _ = reduced_score_grad(problem, Z)
    rows = []
    for eps in epsilons:
        deltas = []
        for _ in range(trials):
            W = rng.standard_normal(Z.shape)
            Zp = stiefel_retract(Z + float(eps) * W, problem.get("Qz"))
            Fp, _, _, _, _ = reduced_score_grad(problem, Zp)
            deltas.append(float(Fp - base))
        rows.append(
            {
                "eps": float(eps),
                "min": float(np.min(deltas)),
                "mean": float(np.mean(deltas)),
                "max": float(np.max(deltas)),
            }
        )
    return rows


def curvature_probe(problem, Z, epsilons, trials, rng):
    base, _, _, _, _ = reduced_score_grad(problem, Z)
    rows = []
    for eps in epsilons:
        vals = []
        for _ in range(trials):
            H = stiefel_gradient(Z, rng.standard_normal(Z.shape), problem.get("Qz"))
            H_norm = max(float(np.linalg.norm(H, "fro")), 1e-30)
            H = H / H_norm
            Zp = stiefel_retract(Z + float(eps) * H, problem.get("Qz"))
            Zm = stiefel_retract(Z - float(eps) * H, problem.get("Qz"))
            Fp, _, _, _, _ = reduced_score_grad(problem, Zp)
            Fm, _, _, _, _ = reduced_score_grad(problem, Zm)
            vals.append(float((Fp - 2.0 * base + Fm) / (float(eps) ** 2)))
        rows.append(
            {
                "eps": float(eps),
                "min": float(np.min(vals)),
                "mean": float(np.mean(vals)),
                "max": float(np.max(vals)),
            }
        )
    return rows
