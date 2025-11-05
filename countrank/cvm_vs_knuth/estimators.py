
import random
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
    p_at_event:list=field(default_factory=list)  # cutoff prior to processing event
    buffer_full_before_event:list=field(default_factory=list)
    peak_buffer:int=0
    last_scores:dict=field(default_factory=dict)  # key -> last u
    final_membership:dict=field(default_factory=dict)  # key -> in buffer? (True/False)
    all_scores:list=field(default_factory=list)  # raw u's in sequence
    accept_flags:list=field(default_factory=list) # per step accepted (pre-capacity)
    capacity:int=0
    halving_records:list=field(default_factory=list)
    trace_key:int=None
    trace_membership:list=field(default_factory=list)
    trace_p:list=field(default_factory=list)
    trace_first_hit:int=None

def _rng(seed):
    return random.Random(seed)

def Estimator(stream_iter, s:int, seed:int, variant:str, trace_key=None):
    """
    Estimator(stream_iter, s, seed, variant)
    variant ∈ {"cvm","cvm_total","knuth_d","knuth_dprime"}
    Returns (estimate, logs). Implements semantics per SPEC.
    Deterministic given seed and deterministic stream_iter.
    """
    if variant not in {"cvm","cvm_total","knuth_d","knuth_dprime"}:
        raise ValueError("Unknown variant")
    rng = _rng(seed)
    B = {}  # key -> u
    p = 1.0
    logs = Logs()
    logs.start_time = time.time()
    logs.p_traj.append(p)
    logs.p_events.append((0, p, "init"))
    logs.capacity = s
    logs.trace_key = trace_key
    trace_seen = False
    t = 0
    for a in stream_iter:
        t += 1
        # Remove stale copy (last-appearance rule)
        if a in B:
            del B[a]
        if trace_key is not None and a == trace_key:
            trace_seen = True
        # Draw fresh independent uniform score
        u = rng.random()
        logs.all_scores.append(u)
        logs.last_scores[a] = u

        logs.buffer_full_before_event.append(len(B) >= s)
        logs.p_at_event.append(p)

        # Check cutoff
        if u >= p:
            logs.rejects += 1
            logs.accept_flags.append(0)
            # event finalization handled below
        else:
            # Candidate would be accepted by cutoff
            logs.accept += 1
            logs.accept_flags.append(1 if len(B) < s else 0)

            if variant in {"cvm", "cvm_total"}:
                if len(B) < s:
                    B[a] = u
                    logs.peak_buffer = max(logs.peak_buffer, len(B))
                    logs.updates += 1
                else:
                    # Buffer full: perform independent-halving subsample rounds until room
                    while len(B) >= s:
                        pre_size = len(B)
                        if variant == "cvm":
                            survivors = {}
                            kept = 0
                            for k, v in B.items():
                                if rng.random() < 0.5:
                                    survivors[k] = v
                                    kept += 1
                                else:
                                    logs.evictions += 1
                            B = survivors
                            p = p / 2.0
                        else:
                            new_threshold = p / 2.0
                            keys = list(B.keys())
                            if len(keys) <= 1:
                                keep_count = 0
                            else:
                                low = len(keys) // 2
                                high = (len(keys) + 1) // 2
                                if low == high:
                                    keep_count = low
                                else:
                                    keep_count = low if rng.random() < 0.5 else high
                            if keep_count > 0:
                                chosen_keys = rng.sample(keys, keep_count)
                                survivors = {}
                                for k in chosen_keys:
                                    survivors[k] = rng.random() * new_threshold
                            else:
                                survivors = {}
                            logs.evictions += pre_size - len(survivors)
                            B = survivors
                            p = new_threshold
                        logs.p_traj.append(p)
                        logs.p_events.append((t, p, "cvm_halve" if variant == "cvm" else "cvm_total_halve"))
                        logs.halving_records.append({
                            "variant": variant,
                            "before": pre_size,
                            "after": len(B),
                            "removed": pre_size - len(B),
                            "p": p
                        })
                    if u < p:
                        B[a] = u
                        logs.peak_buffer = max(logs.peak_buffer, len(B))
                        logs.updates += 1
            elif variant == "knuth_d":
                if len(B) < s:
                    B[a] = u
                    logs.peak_buffer = max(logs.peak_buffer, len(B))
                    logs.updates += 1
                else:
                    a_star, u_star = None, -1.0
                    for k, v in B.items():
                        if v > u_star:
                            a_star, u_star = k, v
                    if u > u_star:
                        p = u
                        logs.p_traj.append(p)
                        logs.p_events.append((t, p, "knuth_raise_p"))
                    else:
                        del B[a_star]
                        logs.evictions += 1
                        B[a] = u
                        p = u_star
                        logs.p_traj.append(p)
                        logs.p_events.append((t, p, "knuth_swap_set_p"))
                        logs.updates += 1
                    logs.peak_buffer = max(logs.peak_buffer, len(B))
            else:  # knuth_dprime
                if len(B) < s:
                    B[a] = u
                    logs.peak_buffer = max(logs.peak_buffer, len(B))
                    logs.updates += 1
                else:
                    while True:
                        pre_size = len(B)
                        new_threshold = p / 2.0
                        kept = {k: v for k, v in B.items() if v < new_threshold}
                        removed = pre_size - len(kept)
                        if removed > 0:
                            logs.evictions += removed
                        B = kept
                        p = new_threshold
                        logs.p_traj.append(p)
                        logs.p_events.append((t, p, "dprime_halve"))
                        repeat = (removed == 0 and len(B) >= s and u < p)
                        logs.halving_records.append({
                            "variant": "knuth_dprime",
                            "before": pre_size,
                            "after": len(B),
                            "removed": removed,
                            "repeat": repeat,
                            "p": p
                        })
                        if (len(B) < s) or (u >= p):
                            break
                    if u < p and len(B) < s:
                        B[a] = u
                        logs.peak_buffer = max(logs.peak_buffer, len(B))
                        logs.updates += 1

        if trace_key is not None:
            if trace_seen and logs.trace_first_hit is None:
                logs.trace_first_hit = t
            logs.trace_membership.append(1 if trace_key in B else 0)
            logs.trace_p.append(p if p > 0 else float('inf'))

    logs.end_time = time.time()
    total_time = logs.end_time - logs.start_time
    logs.time_per_million = (total_time / max(1, t)) * 1e6
    logs.final_membership = {k: True for k in B.keys()}
    estimate = len(B) / p if p > 0 else float('inf')
    # snap p to power-of-two if extremely close for D'
    return estimate, logs
