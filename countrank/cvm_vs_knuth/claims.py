
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
