
import random


def perm_with_repeats(D:int, m:int, seed:int, repeat_factor:float=1.0):
    """
    Universe of D distinct items labeled 0..D-1. Stream length m.
    repeat_factor >= 1.0 controls tendency to repeat recently seen items.
    """
    rng = random.Random(seed)
    items = list(range(D))
    stream = []
    last = None
    for t in range(m):
        if (last is not None) and (rng.random() < (1.0 - 1.0/repeat_factor)):
            stream.append(last)
        else:
            x = int(items[rng.randrange(0, D)])
            stream.append(x)
            last = x
    return stream, D

def zipf_stream(D:int, m:int, seed:int, alpha:float=1.0):
    """
    Keys 0..D-1 sampled i.i.d. from Zipf(alpha) truncated to D.
    """
    rng = random.Random(seed)
    ranks = list(range(1, D+1))
    weights = [rank ** (-alpha) for rank in ranks]
    stream = rng.choices(range(D), weights=weights, k=m)
    return stream, D

def adversarial_repeats(D:int, m:int, seed:int, block:int=10):
    """
    Present each key in long repeated blocks to stress last-appearance logic.
    """
    rng = random.Random(seed)
    order = list(range(D))
    rng.shuffle(order)
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
