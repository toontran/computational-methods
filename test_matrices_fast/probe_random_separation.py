"""Random finite-sample separation probe for TH-05.

Backlog: summary/overview/score_family_workflow.txt [TH-05]

Purpose. Hardens TH-04 (summary/theory_isvd_entropy_regime.md) against the
"the separation only happens because the rows were placed adversarially"
objection. We sample row order from an iid model and check that the
transient-spike vs recurring-diffuse separation between
  - iSVD/streaming gain norm        (||A_cur x||^2),
  - the current-plus-validation norm (||[A_cur; A_fut] x||^2), and
  - the HM2 stability score          (HM(||A_cur x||^2, ||A_fut x||^2))
still appears at non-negligible probability under randomization.

Minimal iid model.
  Two unit candidate directions u and v in R^d. Every row independently:
    response in direction v: deterministic a (diffuse, every row same);
    response in direction u: L with probability p, else 0 (rare spike).
  m current rows, q validation rows, no prior carry (B_prev = 0).

Closed-form facts used.
  Let K_cur = # current rows with a u-spike  ~ Binomial(m, p),
      K_fut = # validation rows with a u-spike  ~ Binomial(q, p),
      independent. Then
    ||A_cur u||^2 = K_cur L^2,  ||A_fut u||^2 = K_fut L^2,
    ||A_cur v||^2 = m a^2,      ||A_fut v||^2 = q a^2.
  Single-spike-in-current event:
    P(K_cur = 1, K_fut = 0) = m p (1 - p)^(m + q - 1).
  Conditional on m a^2 < L^2 < (m+q) a^2, this event yields
    iSVD picks u  (L^2 > m a^2),
    stacked picks v ((m+q) a^2 > L^2 + 0),
    HM2 picks v   (HM2_u = 0 since K_fut = 0).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np


def hm2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized HM2 = 2 x y / (x + y); returns 0 when x + y = 0."""
    s = x + y
    out = np.zeros_like(s, dtype=np.float64)
    mask = s > 0.0
    out[mask] = 2.0 * x[mask] * y[mask] / s[mask]
    return out


def run_one_setting(
    m: int,
    q: int,
    a: float,
    L: float,
    p: float,
    n_trials: int,
    rng: np.random.Generator,
) -> dict:
    a2 = float(a) * float(a)
    L2 = float(L) * float(L)

    E_cur_v = m * a2
    E_fut_v = q * a2
    HM2_v = 2.0 * E_cur_v * E_fut_v / (E_cur_v + E_fut_v) if (E_cur_v + E_fut_v) > 0 else 0.0

    K_cur = rng.binomial(m, p, size=n_trials).astype(np.int64)
    K_fut = rng.binomial(q, p, size=n_trials).astype(np.int64)

    E_cur_u = K_cur.astype(np.float64) * L2
    E_fut_u = K_fut.astype(np.float64) * L2

    isvd_picks_u = (E_cur_u > E_cur_v).astype(np.int64)
    isvd_picks_v = (E_cur_u < E_cur_v).astype(np.int64)

    full_total_u = E_cur_u + E_fut_u
    full_total_v = E_cur_v + E_fut_v
    stacked_picks_v = (full_total_u < full_total_v).astype(np.int64)
    stacked_picks_u = (full_total_u > full_total_v).astype(np.int64)

    HM2_u = hm2(E_cur_u, E_fut_u)
    hm2_picks_v = (HM2_u < HM2_v).astype(np.int64)
    hm2_picks_u = (HM2_u > HM2_v).astype(np.int64)

    # joint TH-04 separation event: iSVD wrong (picks u), stacked right, HM2 right.
    joint = (isvd_picks_u & stacked_picks_v & hm2_picks_v).astype(np.int64)

    # exact "one current spike, no validation spike" event from the spec.
    single_spike = ((K_cur == 1) & (K_fut == 0)).astype(np.int64)

    p_single_theory = m * p * (1.0 - p) ** (m + q - 1)
    cond_param_holds = (m * a2 < L2) and (L2 < (m + q) * a2)

    n = int(n_trials)
    return {
        "m": int(m),
        "q": int(q),
        "a": float(a),
        "L": float(L),
        "p": float(p),
        "L_over_a": float(L / a) if a > 0 else float("inf"),
        "L2_over_m_a2": float(L2 / (m * a2)) if (m * a2) > 0 else float("inf"),
        "L2_over_full_a2": float(L2 / ((m + q) * a2)) if ((m + q) * a2) > 0 else float("inf"),
        "param_condition_m_a2_lt_L2_lt_full_a2": bool(cond_param_holds),
        "p_single_spike_theory": float(p_single_theory),
        "freq_single_spike": float(single_spike.mean()),
        "freq_isvd_picks_u": float(isvd_picks_u.mean()),
        "freq_isvd_picks_v": float(isvd_picks_v.mean()),
        "freq_stacked_picks_v": float(stacked_picks_v.mean()),
        "freq_stacked_picks_u": float(stacked_picks_u.mean()),
        "freq_hm2_picks_v": float(hm2_picks_v.mean()),
        "freq_hm2_picks_u": float(hm2_picks_u.mean()),
        "freq_joint_separation": float(joint.mean()),
        "n_trials": n,
    }


def make_grid() -> list[dict]:
    settings: list[dict] = []

    # 1) Canonical TH-05 setting, p chosen at the maximizer of m p (1-p)^(m+q-1).
    settings.append({
        "label": "canonical_m32_q96_L8_pOpt",
        "m": 32, "q": 96, "a": 1.0, "L": 8.0, "p": 1.0 / 128.0,
    })

    # 2) Sweep p around the canonical optimum 1/(m+q) = 1/128.
    for p in [1.0 / 1024.0, 1.0 / 512.0, 1.0 / 256.0, 1.0 / 128.0,
              1.0 / 64.0, 1.0 / 32.0, 1.0 / 16.0]:
        settings.append({
            "label": f"sweep_p_canonical_p{p:.5g}",
            "m": 32, "q": 96, "a": 1.0, "L": 8.0, "p": p,
        })

    # 3) Sweep L/a inside, on, and outside the parameter window (m=32, q=96, a=1).
    #    Window m a^2 < L^2 < (m+q) a^2 is sqrt(32) ~ 5.657 < L < sqrt(128) ~ 11.314.
    for L in [4.0, 5.0, 5.7, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 11.5, 13.0]:
        settings.append({
            "label": f"sweep_L_canonical_L{L:.3g}",
            "m": 32, "q": 96, "a": 1.0, "L": L, "p": 1.0 / 128.0,
        })

    # 4) Sweep window sizes (m, q) at L^2 placed in mid-window, p at optimum.
    for (m, q) in [(8, 24), (16, 48), (32, 96), (64, 192), (128, 384)]:
        L = float(np.sqrt(0.5 * (m + (m + q))))  # midway energy: (m + (m+q))/2
        settings.append({
            "label": f"sweep_window_m{m}_q{q}",
            "m": m, "q": q, "a": 1.0, "L": L, "p": 1.0 / (m + q),
        })

    # 5) Sweep validation-to-current ratio q/m (m fixed, q varies).
    for q in [16, 32, 64, 96, 192, 384]:
        m = 32
        L = float(np.sqrt(0.5 * (m + (m + q))))
        settings.append({
            "label": f"sweep_q_over_m_q{q}",
            "m": m, "q": q, "a": 1.0, "L": L, "p": 1.0 / (m + q),
        })

    return settings


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-trials", type=int, default=200_000,
                    help="Monte Carlo trials per setting")
    ap.add_argument("--seed", type=int, default=20260430)
    ap.add_argument("--out-dir", type=str,
                    default=os.path.join("summary",
                                         "theory_isvd_random_finite_sample"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    grid = make_grid()
    rows = []
    t0 = time.time()
    for i, s in enumerate(grid):
        r = run_one_setting(
            m=s["m"], q=s["q"], a=s["a"], L=s["L"], p=s["p"],
            n_trials=args.n_trials, rng=rng,
        )
        r["label"] = s["label"]
        rows.append(r)
        print(f"[{i+1:3d}/{len(grid)}] {s['label']:38s} "
              f"m={r['m']:>4d} q={r['q']:>4d} L={r['L']:.3g} p={r['p']:.4g}  "
              f"P_th={r['p_single_spike_theory']:.5g}  "
              f"single={r['freq_single_spike']:.5g}  "
              f"iSVD->u={r['freq_isvd_picks_u']:.4f}  "
              f"stack->v={r['freq_stacked_picks_v']:.4f}  "
              f"HM2->v={r['freq_hm2_picks_v']:.4f}  "
              f"joint={r['freq_joint_separation']:.5g}  "
              f"cond_ok={r['param_condition_m_a2_lt_L2_lt_full_a2']}",
              flush=True)
    dt = time.time() - t0
    print(f"\nTotal MC time: {dt:.1f}s   trials/setting: {args.n_trials}")

    # Write CSV.
    csv_path = os.path.join(args.out_dir, "random_separation_grid.csv")
    field_order = [
        "label", "m", "q", "a", "L", "p",
        "L_over_a", "L2_over_m_a2", "L2_over_full_a2",
        "param_condition_m_a2_lt_L2_lt_full_a2",
        "p_single_spike_theory", "freq_single_spike",
        "freq_isvd_picks_u", "freq_isvd_picks_v",
        "freq_stacked_picks_v", "freq_stacked_picks_u",
        "freq_hm2_picks_v", "freq_hm2_picks_u",
        "freq_joint_separation", "n_trials",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=field_order)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in field_order})
    print(f"Wrote {csv_path}")

    # Write JSON.
    json_path = os.path.join(args.out_dir, "random_separation_grid.json")
    with open(json_path, "w") as f:
        json.dump({
            "spec": "TH-05 random finite-sample iSVD vs stability separation",
            "model": (
                "Two orthonormal directions u, v. Every row has v-response a "
                "(diffuse). Each row independently has u-response L with "
                "probability p, else 0. m current rows, q validation rows, "
                "no prior carry."
            ),
            "selection_rules": {
                "isvd": "argmax_x ||A_cur x||^2",
                "stacked": "argmax_x ||[A_cur; A_fut] x||^2",
                "hm2": ("argmax_x HM2(||A_cur x||^2, ||A_fut x||^2) "
                        "with HM2(a,b)=2ab/(a+b), 0 if a+b=0"),
            },
            "n_trials_per_setting": int(args.n_trials),
            "seed": int(args.seed),
            "rows": rows,
        }, f, indent=2, sort_keys=False)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
