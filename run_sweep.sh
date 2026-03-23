#!/bin/bash
set -euo pipefail

# =========================
# Fixed experiment settings
# =========================
method_name="isvd"

# Matrix size N
N=1000

# Single matrix for now
matrix_name="kernel_stocks_${N}_1.0"
# matrix_name="GHS_psdef/bmw7st_1" # 141347
# matrix_name="ND/nd24k" # 72,000
# matrix_name="Rothberg/cfd2" # 123440
# matrix_name="Mulvey/finan512" # 74752

# Number of machines and which machine this script is for
NUM_MACHINES=30

# Total number of (k, win_size) pairs to generate across all machines.
# With 30 machines and 10 pairs each, use 300.  
TOTAL_PAIRS=300

usage() {
    echo "Usage: $0 <machine_id>"
    echo "  machine_id must be between 0 and $((NUM_MACHINES - 1))"
    echo ""
    echo "Current settings:"
    echo "  matrix_name   = $matrix_name"
    echo "  method_name   = $method_name"
    echo "  N             = $N"
    echo "  NUM_MACHINES  = $NUM_MACHINES"
    echo "  TOTAL_PAIRS   = $TOTAL_PAIRS"
    echo ""
    echo "Assignment rule:"
    echo "  1) deterministically generate TOTAL_PAIRS unique (k, win_size) pairs"
    echo "  2) sort them by win_size ascending"
    echo "  3) assign pair j to machine (j % NUM_MACHINES)"
    echo ""
    echo "This gives each machine a spread across small/medium/large win_size."
}

if [ $# -ne 1 ]; then
    usage
    exit 1
fi

machine_id="$1"

if ! [[ "$machine_id" =~ ^[0-9]+$ ]] || [ "$machine_id" -lt 0 ] || [ "$machine_id" -ge "$NUM_MACHINES" ]; then
    echo "Error: machine_id must be an integer between 0 and $((NUM_MACHINES - 1))"
    usage
    exit 1
fi

k_min=1
if [ "$N" -le 5000 ]; then
    k_max=$(( N < 500 ? N - 1 : 500 ))
elif [ "$N" -le 20000 ]; then
    k_max=300
elif [ "$N" -le 100000 ]; then
    k_max=200
else
    k_max=100
fi
# k_max=100 # remove later
win_min=$((N / 100)) # at least 1% of N
win_max=$((N / 5 - 1))

if [ "$k_max" -lt "$k_min" ] || [ "$win_max" -lt "$win_min" ]; then
    echo "Error: invalid parameter bounds derived from N=$N"
    exit 1
fi

mkdir -p logs

safe_matrix_name="${matrix_name//\//_}"

mapfile -t assigned_pairs < <(
python3 - <<PY
from math import floor

k_min = $k_min
k_max = $k_max
w_min = $win_min
w_max = $win_max
total_pairs = $TOTAL_PAIRS
num_machines = $NUM_MACHINES
machine_id = $machine_id

def radical_inverse(i: int, base: int) -> float:
    x = 0.0
    f = 1.0 / base
    while i > 0:
        x += f * (i % base)
        i //= base
        f /= base
    return x

def deterministic_pairs_unique(k_min, k_max, w_min, w_max, m):
    total_possible = (k_max - k_min + 1) * (w_max - w_min + 1)
    if m > total_possible:
        raise ValueError(f"Requested {m} pairs, but only {total_possible} distinct integer pairs exist")

    pairs = []
    seen = set()
    i = 1

    while len(pairs) < m:
        u = radical_inverse(i, 2)
        v = radical_inverse(i, 3)

        k = round(k_min + u * (k_max - k_min))
        w = round(w_min + v * (w_max - w_min))

        k = max(k_min, min(k, k_max))
        w = max(w_min, min(w, w_max))

        if (k, w) not in seen:
            seen.add((k, w))
            pairs.append((k, w))

        i += 1

    return pairs

pairs = deterministic_pairs_unique(k_min, k_max, w_min, w_max, total_pairs)

# Small win_size is slower, so sort by win_size and distribute round-robin.
pairs.sort(key=lambda p: (p[1], p[0]))

assigned = [p for idx, p in enumerate(pairs) if idx % num_machines == machine_id]

for k, w in assigned:
    print(f"{k} {w}")
PY
)

echo "Machine ID: $machine_id / $((NUM_MACHINES - 1))"
echo "Matrix: $matrix_name"
echo "Method: $method_name"
echo "N: $N"
echo "Bounds: k in [$k_min, $k_max], win_size in [$win_min, $win_max]"
echo "Assigned pairs: ${#assigned_pairs[@]}"
echo ""

for pair in "${assigned_pairs[@]}"; do
    read -r k win_size <<< "$pair"
    mem_size=$((k + win_size))
    log_filename="logs/${safe_matrix_name}_${method_name}_mem${mem_size}_win${win_size}_k${k}.txt"

    echo "========================================"
    echo "Machine:   $machine_id"
    echo "Matrix:    $matrix_name"
    echo "Method:    $method_name"
    echo "k:         $k"
    echo "win_size:  $win_size"
    echo "mem_size:  $mem_size"
    echo "Log file:  $log_filename"
    echo "Command:"
    echo "python3 -u main.py \"$matrix_name\" \"$method_name\" \"$win_size\" \"$k\" 2>&1 | tee -a \"$log_filename\""
    echo ""

    python3 -u main.py "$matrix_name" "$method_name" "$win_size" "$k" 2>&1 | tee -a "$log_filename"
done
