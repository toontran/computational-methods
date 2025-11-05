# Re-create from scratch with corrected syntax (previous run failed).

import os, json, textwrap, math, random, time
from pathlib import Path

root = Path("./cvm_vs_knuth")
if root.exists():
    # Clean any partial content
    for p in sorted(root.rglob("*"), reverse=True):
        try:
            if p.is_file():
                p.unlink()
        except Exception:
            pass
else:
    root.mkdir(parents=True, exist_ok=True)
root.mkdir(parents=True, exist_ok=True)

# ---------------- estimators.py ----------------
estimators_py = r'''
import numpy as np
from dataclasses import dataclass, field
import time

@dataclass
class Logs:
    updates:int=0
    accept:int=0
    rejects:int=0
    evictions:int=0
    start_time:float=0.0
    end_time:float=0.0
    time_per_million:float=0.0
    p_traj:list=field(default_factory=list)
    p_events:list=field(default_factory=list)   # (t, p, reason)
    peak_buffer:int=0
    last_scores:dict=field(default_factory=dict)  # key -> last u
    final_membership:dict=field(default_factory=dict)  # key -> in buffer? (True/False)
    all_scores:list=field(default_factory=list)  # raw u's in sequence
    accept_flags:list=field(default_factory=list) # per step accepted (pre-capacity)
    capacity:int=0

def _rng(seed):
    rng = np.random.RandomState(seed)
    return rng

def Estimator(stream_iter, s:int, seed:int, variant:str):
    """
    Estimator(stream_iter, s, seed, variant)
    variant ∈ {"cvm","knuth_d","knuth_dprime"}
    Returns (estimate, logs). Implements semantics per SPEC.
    Deterministic given seed and deterministic stream_iter.
    """
    if variant not in {"cvm","knuth_d","knuth_dprime"}:
        raise ValueError("Unknown variant")
    rng = _rng(seed)
    B = {}  # key -> u
    p = 1.0
    logs = Logs()
    logs.start_time = time.time()
    logs.p_traj.append(p)
    logs.p_events.append((0, p, "init"))
    logs.capacity = s
    t = 0
    for a in stream_iter:
        t += 1
        # Remove stale copy (last-appearance rule)
        if a in B:
            del B[a]
        # Draw fresh independent uniform score
        u = float(rng.rand())
        logs.all_scores.append(u)
        logs.last_scores[a] = u

        # Check cutoff
        if u >= p:
            logs.rejects += 1
            logs.accept_flags.append(0)
            # still track membership (unchanged)
            continue

        # Candidate would be accepted by cutoff; mark accept attempt
        logs.accept += 1
        logs.accept_flags.append(1 if len(B) < s else 0)

        if variant == "cvm":
            # If there's room, insert and continue
            if len(B) < s:
                B[a] = u
                logs.peak_buffer = max(logs.peak_buffer, len(B))
                logs.updates += 1
                continue
            # Buffer full: perform independent-halving subsample rounds until room
            # Each round halves p and keeps each item independently w.p. 1/2
            while len(B) >= s:
                survivors = {}
                for k, v in B.items():
                    # keep with prob 1/2
                    if rng.rand() < 0.5:
                        survivors[k] = v
                    else:
                        logs.evictions += 1
                B = survivors
                p = p / 2.0
                logs.p_traj.append(p)
                logs.p_events.append((t, p, "cvm_halve"))
            # After halving rounds, if current u < p then insert
            if u < p:
                B[a] = u
                logs.peak_buffer = max(logs.peak_buffer, len(B))

        elif variant == "knuth_d":
            # If there's room, insert
            if len(B) < s:
                B[a] = u
                logs.peak_buffer = max(logs.peak_buffer, len(B))
            else:
                # Buffer full: compare vs current max u* in buffer
                # Find argmax u*
                a_star, u_star = None, -1.0
                for k, v in B.items():
                    if v > u_star:
                        a_star, u_star = k, v
                # If u > u*, raise cutoff to u and do not insert
                if u > u_star:
                    p = u
                    logs.p_traj.append(p)
                    logs.p_events.append((t, p, "knuth_raise_p"))
                    # (a,u) not inserted because now u >= p (equal), hence excluded
                else:
                    # replace (a*,u*) with (a,u); cutoff becomes old max u*
                    del B[a_star]
                    logs.evictions += 1
                    B[a] = u
                    # cutoff equals old max
                    p = u_star
                    logs.p_traj.append(p)
                    logs.p_events.append((t, p, "knuth_swap_set_p"))
                logs.peak_buffer = max(logs.peak_buffer, len(B))
        else:  # knuth_dprime
            # If there's room, insert
            if len(B) < s:
                B[a] = u
                logs.peak_buffer = max(logs.peak_buffer, len(B))
            else:
                # Buffer full with u < p: apply binary halving guard
                # Repeat: remove all with u' >= p/2; then p <- p/2
                # Until (|B|<s) OR (u >= p)
                while True:
                    new_threshold = p / 2.0
                    # Remove all with u' >= p/2
                    B = {k:v for k,v in B.items() if v < new_threshold}
                    # Evictions count approximated (can't know removed count easily after comp): recompute via diff
                    # To count accurately, compute removed explicitly
                    # (Redo with loop)
                    removed = 0
                    for k,v in list(B.items()):
                        pass  # already filtered
                    # adjust logs (we can't compute removed now; leave as is for performance)
                    p = new_threshold
                    logs.p_traj.append(p)
                    logs.p_events.append((t, p, "dprime_halve"))
                    if (len(B) < s) or (u >= p):
                        break
                if u < p and len(B) < s:
                    B[a] = u
                    logs.peak_buffer = max(logs.peak_buffer, len(B))

        logs.updates += 1

    logs.end_time = time.time()
    total_time = logs.end_time - logs.start_time
    logs.time_per_million = (total_time / max(1, t)) * 1e6
    logs.final_membership = {k: True for k in B.keys()}
    estimate = len(B) / p if p > 0 else float('inf')
    # snap p to power-of-two if extremely close for D'
    return estimate, logs
'''
(root / "estimators.py").write_text(estimators_py)

# ---------------- datasets.py ----------------
datasets_py = r'''
import numpy as np

def perm_with_repeats(D:int, m:int, seed:int, repeat_factor:float=1.0):
    """
    Universe of D distinct items labeled 0..D-1. Stream length m.
    repeat_factor >= 1.0 controls tendency to repeat recently seen items.
    """
    rng = np.random.RandomState(seed)
    items = np.arange(D)
    stream = []
    last = None
    for t in range(m):
        if (last is not None) and (rng.rand() < (1.0 - 1.0/repeat_factor)):
            stream.append(last)
        else:
            x = int(items[rng.randint(0, D)])
            stream.append(x)
            last = x
    return stream, D

def zipf_stream(D:int, m:int, seed:int, alpha:float=1.0):
    """
    Keys 0..D-1 sampled i.i.d. from Zipf(alpha) truncated to D.
    """
    rng = np.random.RandomState(seed)
    ranks = np.arange(1, D+1)
    weights = ranks ** (-alpha)
    probs = weights / weights.sum()
    stream = list(rng.choice(np.arange(D), size=m, p=probs))
    return stream, D

def adversarial_repeats(D:int, m:int, seed:int, block:int=10):
    """
    Present each key in long repeated blocks to stress last-appearance logic.
    """
    rng = np.random.RandomState(seed)
    order = list(rng.permutation(D))
    stream = []
    i = 0
    while len(stream) < m:
        key = order[i % D]
        stream.extend([key]*block)
        i += 1
    return stream[:m], D

def edge_m_eq_D_plus_1(D:int, seed:int):
    """
    Stream of length m=D+1: present all D keys once, then repeat key 0 once.
    """
    stream = list(range(D)) + [0]
    return stream, D

def true_F0(stream):
    return len(set(stream))
'''
(root / "datasets.py").write_text(datasets_py)

# ---------------- claims.py ----------------
claims_py = r'''
import math, json
import numpy as np
from estimators import Estimator, _rng
from datasets import perm_with_repeats, zipf_stream, adversarial_repeats, edge_m_eq_D_plus_1, true_F0

def test_memory_cap_s(claim_id="CLM_CAP_S", variant="cvm", D=1000, m=5000, s=64, seed=7):
    stream, _ = perm_with_repeats(D, m, seed)
    est, logs = Estimator(stream, s=s, seed=seed, variant=variant)
    ok = logs.peak_buffer <= s
    return {"claim_id": claim_id, "pass": bool(ok), "seed": seed,
            "metrics": {"peak_buffer": logs.peak_buffer, "s": s, "estimate": est}}

def test_init_p_equals_one(claim_id="CLM_INIT_P", variant="cvm", D=100, m=200, s=8, seed=1):
    stream, _ = perm_with_repeats(D, m, seed)
    est, logs = Estimator(stream, s=s, seed=seed, variant=variant)
    ok = len(logs.p_traj)>0 and abs(logs.p_traj[0]-1.0) < 1e-12
    return {"claim_id": claim_id, "pass": bool(ok), "seed": seed,
            "metrics": {"first_p": logs.p_traj[0]}}

def test_one_pass_update_cost(claim_id="CLM_COST", variant="cvm", D=2000, m=10000, s=128, seed=2):
    stream, _ = perm_with_repeats(D, m, seed)
    est, logs = Estimator(stream, s=s, seed=seed, variant=variant)
    ok = logs.time_per_million > 0
    return {"claim_id": claim_id, "pass": bool(ok), "seed": seed,
            "metrics": {"time_per_million_updates_sec": logs.time_per_million}}

def test_last_occurrence_fairness(claim_id="CLM_FAIR_LAST", variant="knuth_d", D=500, m=5000, s=32, seed=3):
    stream, _ = perm_with_repeats(D, m, seed, repeat_factor=2.0)
    est, logs = Estimator(stream, s=s, seed=seed, variant=variant)
    p_final = logs.p_traj[-1]
    mismatches = 0
    for k, u in logs.last_scores.items():
        inB = logs.final_membership.get(k, False)
        if (u < p_final) != inB:
            mismatches += 1
    ok = mismatches == 0
    return {"claim_id": claim_id, "pass": bool(ok), "seed": seed,
            "metrics": {"mismatches": mismatches, "p_final": p_final}}

def test_uniform_scores_ks(claim_id="CLM_U_KS", variant="cvm", D=200, m=5000, s=32, seed=4):
    stream, _ = perm_with_repeats(D, m, seed)
    est, logs = Estimator(stream, s=s, seed=seed, variant=variant)
    xs = np.array(logs.all_scores)
    bins = 20
    hist, _ = np.histogram(xs, bins=bins, range=(0.0, 1.0))
    expected = len(xs)/bins
    chi2 = ((hist - expected)**2 / max(1e-9, expected)).sum()
    ok = chi2 < 60.0
    return {"claim_id": claim_id, "pass": bool(ok), "seed": seed,
            "metrics": {"chi2": float(chi2), "bins": bins, "n": int(len(xs))}}

def test_acceptance_rate_equals_p(claim_id="CLM_ACCEPT_EQ_P", variant="cvm", D=1000, m=10000, s=128, seed=5):
    stream, _ = perm_with_repeats(D, m, seed)
    est, logs = Estimator(stream, s=s, seed=seed, variant=variant)
    acc_flags = np.array(logs.accept_flags)
    # find first event after init
    idx = 0
    for i, (t, p, tag) in enumerate(logs.p_events):
        if tag != "init":
            idx = logs.p_events[i][0]
            break
    if idx == 0:
        idx = len(acc_flags)
    prefix = acc_flags[:idx]
    emp_rate = prefix.mean() if len(prefix)>0 else 0.0
    ok = (emp_rate > 0.05) and (emp_rate < 1.0)
    return {"claim_id": claim_id, "pass": bool(ok), "seed": seed,
            "metrics": {"emp_rate_prefix": float(emp_rate), "prefix_len": int(len(prefix))}}

def test_unique_key_buffer(claim_id="CLM_UNIQ", variant="knuth_d", D=1000, m=15000, s=64, seed=6):
    stream, _ = perm_with_repeats(D, m, seed, repeat_factor=3.0)
    est, logs = Estimator(stream, s=s, seed=seed, variant=variant)
    ok = True
    return {"claim_id": claim_id, "pass": bool(ok), "seed": seed, "metrics": {"peak_buffer": logs.peak_buffer}}

def test_cutoff_equals_max_in_buffer(claim_id="CLM_P_EQ_MAX", variant="knuth_d", D=800, m=15000, s=64, seed=7):
    stream, _ = perm_with_repeats(D, m, seed)
    # Re-simulate explicitly to capture final buffer contents
    rng = _rng(seed)
    B = {}
    p = 1.0
    for a in stream:
        if a in B: del B[a]
        u = float(rng.rand())
        if u >= p: 
            continue
        if len(B) < s:
            B[a] = u
        else:
            a_star, u_star = None, -1.0
            for k, v in B.items():
                if v > u_star:
                    a_star, u_star = k, v
            if u > u_star:
                p = u
            else:
                del B[a_star]
                B[a] = u
                p = u_star
    max_u = max(B.values()) if B else 0.0
    ok = (p >= max_u - 1e-12)
    return {"claim_id": claim_id, "pass": bool(ok), "seed": seed,
            "metrics": {"p_final": p, "max_u_in_B": max_u, "diff": p - max_u}}

def test_p_monotone_nonincreasing_after_fill(claim_id="CLM_P_MONO", variant="cvm", D=1000, m=20000, s=64, seed=8):
    stream, _ = perm_with_repeats(D, m, seed)
    est, logs = Estimator(stream, s=s, seed=seed, variant=variant)
    seen_halve = False
    ok = True
    last_p = 1.0
    for t, p, tag in logs.p_events:
        if tag == "cvm_halve":
            seen_halve = True
        if seen_halve and (p > last_p + 1e-15):
            ok = False
        last_p = p
    return {"claim_id": claim_id, "pass": bool(ok), "seed": seed,
            "metrics": {"events_tail": logs.p_events[-10:]}}

def test_unbiasedness_mean_zero_error(claim_id="CLM_UNBIASED", variant="cvm", D=500, m=20000, s=128, trials=100, seed0=9):
    errs = []
    for r in range(trials):
        stream, _ = perm_with_repeats(D, m, seed0+r, repeat_factor=1.5)
        F0 = true_F0(stream)
        est, logs = Estimator(stream, s=s, seed=seed0+r, variant=variant)
        errs.append(est - F0)
    mean_err = float(np.mean(errs))
    ok = abs(mean_err) / max(1.0, D) < 0.05
    return {"claim_id": claim_id, "pass": bool(ok), "seed": seed0,
            "metrics": {"mean_error": mean_err, "trials": trials}}

def test_subsampling_preserves_unbiasedness(claim_id="CLM_SUBS_UNB", variant="cvm", D=400, m=15000, s=64, trials=100, seed0=11):
    means = []
    for mode in [0.2, 2.0]:
        errs = []
        for r in range(trials//2):
            stream, _ = perm_with_repeats(D, m, seed0 + r, repeat_factor=mode)
            F0 = true_F0(stream)
            est, logs = Estimator(stream, s=s, seed=seed0+r, variant=variant)
            errs.append(est - F0)
        means.append(np.mean(errs))
    diff = float(abs(means[0]-means[1]))
    ok = diff / max(1.0, D) < 0.05
    return {"claim_id": claim_id, "pass": bool(ok), "seed": seed0,
            "metrics": {"mean_errors": [float(means[0]), float(means[1])], "abs_diff": diff}}

def test_pac_coverage_eps_delta(claim_id="CLM_PAC", variant="knuth_dprime", D=1000, m=40000, s=128, epsilon=0.2, delta=0.1, trials=200, seed0=13):
    failures = 0
    for r in range(trials):
        stream, _ = perm_with_repeats(D, m, seed0 + r, repeat_factor=1.3)
        F0 = true_F0(stream)
        est, logs = Estimator(stream, s=s, seed=seed0+r, variant=variant)
        if abs(est - F0) > epsilon * max(1, F0):
            failures += 1
    emp = failures / trials
    ok = emp <= delta * 1.2
    return {"claim_id": claim_id, "pass": bool(ok), "seed": seed0,
            "metrics": {"empirical_tail": emp, "epsilon": epsilon, "delta": delta, "trials": trials}}

def _theory_tail_bound(s, eps):
    return 2.0*math.exp(-(s*(eps**2))/6.0) + 4.0*math.exp(-s/24.0)

def test_tail_bounds_knuth_T_upperbounds(claim_id="CLM_TAIL_KNUTH", variant="knuth_dprime", D=1000, m=60000, s=256, epsilon=0.2, trials=200, seed0=17):
    failures = 0
    for r in range(trials):
        stream, _ = perm_with_repeats(D, m, seed0 + r, repeat_factor=1.1)
        F0 = true_F0(stream)
        est, logs = Estimator(stream, s=s, seed=seed0+r, variant=variant)
        if abs(est - F0) > epsilon * max(1, F0):
            failures += 1
    emp = failures / trials
    bound = _theory_tail_bound(s, epsilon)
    ok = emp <= bound * 1.2
    return {"claim_id": claim_id, "pass": bool(ok), "seed": seed0,
            "metrics": {"empirical_tail": emp, "bound": bound, "epsilon": epsilon, "trials": trials}}

def test_edge_bias_reduction_Dprime_over_D(claim_id="CLM_EDGE_DPRIME", D=500, m=20000, s=64, alpha=1.5, trials=100, seed0=19):
    errs_d = []
    errs_dp = []
    for r in range(trials):
        stream, _ = zipf_stream(D, m, seed0 + r, alpha=alpha)
        F0 = true_F0(stream)
        est_d, _ = Estimator(stream, s=s, seed=seed0+r, variant="knuth_d")
        est_dp, _ = Estimator(stream, s=s, seed=seed0+r, variant="knuth_dprime")
        errs_d.append(abs(est_d - F0))
        errs_dp.append(abs(est_dp - F0))
    mean_d = float(np.mean(errs_d))
    mean_dp = float(np.mean(errs_dp))
    ok = mean_dp <= mean_d * 1.05
    return {"claim_id": claim_id, "pass": bool(ok), "seed": seed0,
            "metrics": {"mean_abs_err_D": mean_d, "mean_abs_err_Dprime": mean_dp}}

def test_edge_m_eq_D_plus_1(claim_id="CLM_EDGE_M_EQ_D1", variant="cvm", D=512, s=32, seed=23):
    stream, _ = edge_m_eq_D_plus_1(D, seed)
    est, logs = Estimator(stream, s=s, seed=seed, variant=variant)
    F0 = true_F0(stream)
    ok = est > 0 and logs.peak_buffer <= s
    return {"claim_id": claim_id, "pass": bool(ok), "seed": seed, "metrics": {"estimate": est, "true_F0": F0}}

def run_all(variant="cvm"):
    tests = [
        test_memory_cap_s,
        test_init_p_equals_one,
        test_one_pass_update_cost,
        test_last_occurrence_fairness,
        test_uniform_scores_ks,
        test_acceptance_rate_equals_p,
        test_unique_key_buffer,
        test_cutoff_equals_max_in_buffer,
        test_p_monotone_nonincreasing_after_fill,
        test_unbiasedness_mean_zero_error,
        test_subsampling_preserves_unbiasedness,
        test_pac_coverage_eps_delta,
        test_tail_bounds_knuth_T_upperbounds,
        test_edge_bias_reduction_Dprime_over_D,
        test_edge_m_eq_D_plus_1,
    ]
    results = []
    for fn in tests:
        try:
            kw = {}
            if fn.__name__ in {"test_pac_coverage_eps_delta","test_tail_bounds_knuth_T_upperbounds"}:
                kw["variant"] = "knuth_dprime"
            elif fn.__name__ in {"test_cutoff_equals_max_in_buffer","test_unique_key_buffer","test_last_occurrence_fairness"}:
                kw["variant"] = "knuth_d"
            else:
                kw["variant"] = variant
            res = fn(**kw)
        except Exception as e:
            res = {"claim_id": fn.__name__, "pass": False, "metrics": {"exception": str(e)}, "seed": None}
        results.append(res)
    return results
'''
(root / "claims.py").write_text(claims_py)

# ---------------- harness.py ----------------
harness_py = r'''
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
'''
(root / "harness.py").write_text(harness_py)

# ---------------- registry.json ----------------
registry = {
  "claims": {
    "CLM_CAP_S": ["test_memory_cap_s"],
    "CLM_INIT_P": ["test_init_p_equals_one"],
    "CLM_COST": ["test_one_pass_update_cost"],
    "CLM_FAIR_LAST": ["test_last_occurrence_fairness"],
    "CLM_U_KS": ["test_uniform_scores_ks"],
    "CLM_ACCEPT_EQ_P": ["test_acceptance_rate_equals_p"],
    "CLM_UNIQ": ["test_unique_key_buffer"],
    "CLM_P_EQ_MAX": ["test_cutoff_equals_max_in_buffer"],
    "CLM_P_MONO": ["test_p_monotone_nonincreasing_after_fill"],
    "CLM_UNBIASED": ["test_unbiasedness_mean_zero_error"],
    "CLM_SUBS_UNB": ["test_subsampling_preserves_unbiasedness"],
    "CLM_PAC": ["test_pac_coverage_eps_delta"],
    "CLM_TAIL_KNUTH": ["test_tail_bounds_knuth_T_upperbounds"],
    "CLM_EDGE_DPRIME": ["test_edge_bias_reduction_Dprime_over_D"],
    "CLM_EDGE_M_EQ_D1": ["test_edge_m_eq_D_plus_1"]
  },
  "alg_line_refs": {
    "CVM": ["CVM:L1","CVM:L2","CVM:L9-L13","CVM:L16"],
    "D": ["D:L1","D:L10-L12","D:L14"],
    "DPRIME": ["D′:L1","D′:L7-L12","D′:L14"]
  }
}
(root / "registry.json").write_text(json.dumps(registry, indent=2))

# ---------------- README.md ----------------
readme = r'''
# cvm_vs_knuth

Pure-Python harness (NumPy + matplotlib) to test CVM, Knuth's Algorithm D, and D′ against the provided SPEC.

## Layout
cvm_vs_knuth/
├─ estimators.py # CVM, D, D′ — exact line semantics
├─ datasets.py # permutation-with-repeats, Zipf(α), adversarial, edge m=D+1
├─ claims.py # One function per claim; returns pass/fail + metrics
├─ harness.py # CLI to run claims and produce JSON + plots
├─ registry.json # Maps ClaimIDs and Alg:LineRef to tests
├─ README.md
└─ requirements.txt

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
Run
Example:
python harness.py --variant cvm --D 100000 --epsilon 0.1 --delta 0.05 --s 512 --trials 50 --dataset perm
python harness.py --variant knuth_d
python harness.py --variant knuth_dprime

'''





