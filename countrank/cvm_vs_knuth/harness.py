
import argparse, json
from pathlib import Path
from estimators import Estimator
from datasets import perm_with_repeats, zipf_stream, adversarial_repeats, edge_m_eq_D_plus_1, true_F0
from claims import run_all
import numpy as np
import matplotlib.pyplot as plt

def make_stream(kind, D, m, seed, alpha=1.0):
    if kind == "perm":
        return perm_with_repeats(D, m, seed, repeat_factor=1.5)
    elif kind == "zipf":
        return zipf_stream(D, m, seed, alpha=alpha)
    elif kind == "adv":
        return adversarial_repeats(D, m, seed, block=10)
    elif kind == "edge":
        return edge_m_eq_D_plus_1(D, seed)
    else:
        raise ValueError("Unknown dataset kind")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=str, default="cvm", choices=["cvm","knuth_d","knuth_dprime"])
    ap.add_argument("--D", type=int, default=10000)
    ap.add_argument("--m", type=int, default=100000)
    ap.add_argument("--s", type=int, default=512)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--claims", type=str, default="all")
    ap.add_argument("--dataset", type=str, default="perm", choices=["perm","zipf","adv","edge"])
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--plotdir", type=str, default="plots")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    Path(args.plotdir).mkdir(parents=True, exist_ok=True)

    # Demo single run for plots
    stream, _ = make_stream(args.dataset, args.D, args.m, args.seed, args.alpha)
    est, logs = Estimator(stream, s=args.s, seed=args.seed, variant=args.variant)
    F0 = true_F0(stream)
    res = dict(variant=args.variant, D=args.D, m=args.m, s=args.s, epsilon=args.epsilon, delta=args.delta,
               estimate=est, true_F0=F0, time_per_million_updates=logs.time_per_million)
    with open(Path(args.outdir)/f"single_run_{args.variant}.json","w") as f:
        json.dump(res, f, indent=2)

    # Plot p trajectory
    plt.figure()
    xs = [e[0] for e in logs.p_events]
    ys = [e[1] for e in logs.p_events]
    plt.plot(xs, ys)
    plt.xlabel("t (event index)")
    plt.ylabel("p (cutoff)")
    plt.title(f"Cutoff trajectory: {args.variant}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(args.plotdir)/f"p_traj_{args.variant}.png")
    plt.close()

    # Run claims
    all_results = run_all(variant=args.variant)
    with open(Path(args.outdir)/f"claims_{args.variant}.json","w") as f:
        json.dump(all_results, f, indent=2)

    print(json.dumps({"single_run": res, "claims_count": len(all_results)}, indent=2))

if __name__ == "__main__":
    main()
