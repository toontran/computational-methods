import contextlib
import io
import sys
from types import SimpleNamespace

import cex_restricted_space_probe as probe


def make(matrix, n, win):
    return SimpleNamespace(
        mat_input=None,
        matrix=matrix,
        n=n,
        r_sig=2,
        alpha_sig=0.003,
        alpha_tail=0.0145,
        tail_scale=0.99,
        sigma1=0.991,
        v_type="rand",
        mode="restricted",
        rank=2,
        win=win,
        preset="fast",
        shuffle_rows=True,
        row_shuffle_seed=0,
        cex_replicate=False,
        q0=8, qmax=48, krylov_depth=2, residual_tol=1e-2,
        expansion_maxit=8, num_restarts=8, maxit=120, tol=1e-8,
        seed=0, normalize_by_sigma=False, carry="left",
        reduced_optimizer="cex",
        basis_selection="greedy",
        joint_warm_start_greedy=False,
        joint_warm_start_oracle=False,
        rownorm_seed_first_block=True,
        rownorm_seed_all_blocks=True,
        joint_warm_start_rotations=0,
        joint_warm_start_rotation_angle=0.7853981633974483,
        joint_warm_start_perturbations=0,
        joint_warm_start_perturb_scale=1e-2,
        joint_default_svd_start=True,
        joint_oversample=0,
        joint_oversample_rotate="svd",
        joint_solver="riemannian",
        row_concentration_lambda=0.0,
        row_leverage_lambda=0.0,
        row_leverage_mode="none",
        row_leverage_rank=2,
        score_variant="combined",
        old_memory_size=win,
        debug_mode="off",
        oracle_candidate_check=True,
        oracle_sketch_all_seen_rows=False,
        dump_score_components=True,
        dump_oracle_old_row_responses=False,
        dump_oracle_old_row_response_block=3,
        dtype="float32",
        expansion_direction="residual",
        reuse_line_search_grad=True,
        expansion_warm_start=True,
        post_expansion_maxit=80,
        benchmark_output=None,
        benchmark_append=False,
        verbose=False,
    )


if __name__ == "__main__":
    matrix = sys.argv[1] if len(sys.argv) > 1 else "static-cex"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    win = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        probe.run(make(matrix, n, win))
    print(buf.getvalue())
