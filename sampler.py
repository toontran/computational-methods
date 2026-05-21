def deterministic_space_covering_pairs_unique(
    k_min: int,
    k_max: int,
    w_min: int,
    w_max: int,
    m: int,
):
    if not (k_min <= k_max and w_min <= w_max):
        raise ValueError("Invalid min/max bounds")
    if m <= 0:
        raise ValueError("m must be positive")

    total_possible = (k_max - k_min + 1) * (w_max - w_min + 1)
    if m > total_possible:
        raise ValueError("m exceeds number of distinct integer pairs")

    pairs = []
    seen = set()
    i = 0

    while len(pairs) < m:
        u = (i + 0.5) / m
        u = u % 1.0
        v = radical_inverse_base2(i)

        k = round(k_min + u * (k_max - k_min))
        w = round(w_min + v * (w_max - w_min))

        k = max(k_min, min(k, k_max))
        w = max(w_min, min(w, w_max))

        if (k, w) not in seen:
            seen.add((k, w))
            pairs.append((k, w))

        i += 1

    return pairs

def radical_inverse_base2(i):
    x, f = 0.0, 0.5
    while i > 0:
        x += f * (i & 1)
        i >>= 1
        f *= 0.5
    return x

print(deterministic_space_covering_pairs_unique(1, 99, 17, 199, 30))
