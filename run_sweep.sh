#!/bin/bash
set -euo pipefail

# =========================
# Fixed experiment settings
# =========================
# entropyscore_forget_deflated_svd_aux
# Switched from entropyscore_forget_svd_aux → entropyscore_hybrid to target the
# combined-score / deflated-SVD split evaluated in the session notes
# (utils_combined_hybrid_port_session_notes.txt §2–§4).
# Override via env: METHOD_NAME=isvd ./run_sweep.sh <machine_id>
method_name="${METHOD_NAME:-future_hmean_online}"   # isvd, entropyscore_expansion, entropyscore_forget, entropyscore_forget_svd_aux, entropyscore_combined, entropyscore_hybrid, future_hmean_online
#SCORE_RANK_VALUES=(16)
SCORE_RANK_VALUES=(0 1 2 4 16)

# Current fast/balanced entropyscore_forget configuration.
# These flags are especially relevant for entropyscore_forget and aux variants.
# num_restarts bumped 2→8 per session notes §3: num_restarts=2 is nondeterministic
# on close-σ matrices due to float32/BLAS path divergence.
if [[ "$method_name" == future_hmean_online* ]]; then
    FORGET_CONFIG_LABEL="future_hmean_online_f32_r3_pe80"
    FORGET_ARGS=(
        --q0 5
        --qmax 200
        --expansion-maxit 64
        --expansion-direction residual
        --num-restarts 3
        --maxit 120
        --reduced-optimizer cex
        --work-dtype float32
        --reuse-line-search-grad
        --expansion-warm-start
        --post-expansion-maxit 80
    )
else
    FORGET_CONFIG_LABEL="fast_f32_reswarm_r8_pe60"
    FORGET_ARGS=(
        --q0 5
        --qmax 200
        --expansion-maxit 64
        --expansion-direction residual
        --num-restarts 8
        --maxit 120
        --reduced-optimizer cex
        --work-dtype float32
        --reuse-line-search-grad
        --expansion-warm-start
        --post-expansion-maxit 60
    )
fi

# Matrix size N
N=1000
# N=1138

# Single matrix for now
# Lowered lengthscale to tighten σ₁/σ₂ for hybrid/combined evaluation
# (session notes: test_matrices_fast/summary/utils_combined_hybrid_port_session_notes.txt §5).
# Measured σ₁/σ₂ on N=1000 stocks:
#   ls=0.2236 → 1.490  ls=0.1 → 1.338  ls=0.05 → 1.093  ls=0.02 → 1.023  ls=0.01 → 1.066
# 0.02 gives the tightest near-tied gap, so hybrid/combined should have the most
# room to beat iSVD there.
# matrix_name="kernel_stocks_${N}_1.0"
matrix_name="kernel_stocks_${N}_0.2236"
# matrix_name="kernel_stocks_${N}_0.02"
# matrix_name="kernel_stocks_${N}_0.7071"
# matrix_name="kernel_stocks_${N}_1.0"
# matrix_name="GHS_psdef/bmw7st_1" # 141347
# matrix_name="ND/nd24k"           # 72000
# matrix_name="Rothberg/cfd2"      # 123440
# matrix_name="Mulvey/finan512"    # 74752
# matrix_name="HB/1138_bus"          # 1138

# Number of machines and which machine this script is for
NUM_MACHINES=20

# Grid dimensions.
# Total number of candidate runs = NUM_K_VALUES * NUM_WIN_VALUES * NUM_SCORE_RANK_VALUES
NUM_K_VALUES=8
NUM_WIN_VALUES=4
NUM_SCORE_RANK_VALUES=${#SCORE_RANK_VALUES[@]}
RAW_TOTAL_RUNS=$((NUM_K_VALUES * NUM_WIN_VALUES * NUM_SCORE_RANK_VALUES))

# CPU cap for each main.py invocation, expressed as a percentage of the CPUs
# visible to this process. systemd CPUQuota is measured relative to one core,
# so 50% of 16 visible CPUs becomes CPUQuota=800%.
CPU_LIMIT_OF_AVAILABLE_PERCENT=${CPU_LIMIT_OF_AVAILABLE_PERCENT:-50}
OUTPUT_NAME="${OUTPUT_NAME:-output}"
LOCAL_OUTPUT_ROOT="${LOCAL_OUTPUT_ROOT:-${OUTPUT_NAME}}"
SCRATCH_OUTPUT_ROOT="${SCRATCH_OUTPUT_ROOT:-/scratch/ttran02/${OUTPUT_NAME}}"
SKIP_COMPLETED_CONFIGS="${SKIP_COMPLETED_CONFIGS:-1}"
COMPLETION_MARKER="${COMPLETION_MARKER:-other_info.txt}"

usage() {
    echo "Usage: $0 <machine_id>"
    echo "  machine_id must be between 0 and $((NUM_MACHINES - 1))"
    echo ""
    echo "Current settings:"
    echo "  matrix_name     = $matrix_name"
    echo "  method_name     = $method_name"
    echo "  score_ranks     = ${SCORE_RANK_VALUES[*]}"
    echo "  forget_config   = $FORGET_CONFIG_LABEL"
    echo "  forget_args     = ${FORGET_ARGS[*]}"
    echo "  N               = $N"
    echo "  NUM_MACHINES    = $NUM_MACHINES"
    echo "  NUM_K_VALUES    = $NUM_K_VALUES"
    echo "  NUM_WIN_VALUES  = $NUM_WIN_VALUES"
    echo "  NUM_SCORE_RANK_VALUES = $NUM_SCORE_RANK_VALUES"
    echo "  RAW_TOTAL_RUNS  = $RAW_TOTAL_RUNS"
    echo "  CPU_LIMIT_OF_AVAILABLE_PERCENT = $CPU_LIMIT_OF_AVAILABLE_PERCENT"
    echo "  OUTPUT_NAME     = $OUTPUT_NAME"
    echo "  LOCAL_OUTPUT_ROOT = $LOCAL_OUTPUT_ROOT"
    echo "  SCRATCH_OUTPUT_ROOT = $SCRATCH_OUTPUT_ROOT"
    echo "  SKIP_COMPLETED_CONFIGS = $SKIP_COMPLETED_CONFIGS"
    echo ""
    echo "Assignment rule:"
    echo "  1) build an explicit Cartesian grid of (score_rank, k, win_size) triples"
    echo "  2) canonicalize each triple with effective_score_rank = min(score_rank, k)"
    echo "  3) remove duplicate effective runs"
    echo "  4) sort by win_size ascending, then score_rank ascending, then k ascending"
    echo "  5) assign run j to machine (j % NUM_MACHINES)"
    echo ""
    echo "This preserves exact same-win, same-k, and same-score-rank slices globally,"
    echo "while balancing work across machines via round-robin."
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

# =========================
# Parameter bounds
# =========================
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

win_min=$((N / 100))      # at least 1% of N
win_max=$((N / 5 - 1))

if [ "$k_max" -lt "$k_min" ] || [ "$win_max" -lt "$win_min" ]; then
    echo "Error: invalid parameter bounds derived from N=$N"
    exit 1
fi

if [ "$NUM_K_VALUES" -lt 1 ] || [ "$NUM_WIN_VALUES" -lt 1 ] || [ "$NUM_SCORE_RANK_VALUES" -lt 1 ]; then
    echo "Error: NUM_K_VALUES, NUM_WIN_VALUES, and SCORE_RANK_VALUES must all be non-empty"
    exit 1
fi

if ! [[ "$CPU_LIMIT_OF_AVAILABLE_PERCENT" =~ ^[0-9]+$ ]] || [ "$CPU_LIMIT_OF_AVAILABLE_PERCENT" -lt 1 ]; then
    echo "Error: CPU_LIMIT_OF_AVAILABLE_PERCENT must be a positive integer"
    exit 1
fi

if ! command -v systemd-run >/dev/null 2>&1; then
    echo "Error: systemd-run is required for CPUQuota limiting"
    exit 1
fi

available_cpus="$(nproc)"
cpu_quota_percent=$((available_cpus * CPU_LIMIT_OF_AVAILABLE_PERCENT))

mkdir -p logs
safe_matrix_name="${matrix_name//\//_}"
matrix_postfix="${matrix_name##*/}"
score_rank_values_py="$(printf '%s\n' "${SCORE_RANK_VALUES[@]}" | paste -sd, -)"

forget_label=""
for forget_arg in "${FORGET_ARGS[@]}"; do
    if [[ "$forget_arg" == "--cex-replicate" ]]; then
        forget_label="_cexrep"
        break
    fi
done

get_row_permutation_names() {
    case "$matrix_name" in
        cex1|cex_structured_new|cex1_mat|cex1_matlab)
            printf '%s\n' original
            ;;
        *)
            printf '%s\n' \
                random_uniform \
                random_uniform_2 \
                random_uniform_3 \
                random_uniform_4 \
                random_uniform_5
            ;;
    esac
}

run_folder_name() {
    local row_permutation="$1"
    local score_rank="$2"
    local k="$3"
    local win_size="$4"
    local mem_size

    mem_size=$((k + win_size))
    printf '%s_%s_%s_size_%s_ssize_%s_k_%s_sr_%s%s_reservoir_greedy\n' \
        "$matrix_postfix" \
        "$method_name" \
        "$row_permutation" \
        "$mem_size" \
        "$win_size" \
        "$k" \
        "$score_rank" \
        "$forget_label"
}

is_completed_folder() {
    local folder_name="$1"

    [[ -f "$LOCAL_OUTPUT_ROOT/$folder_name/$COMPLETION_MARKER" ]] \
        || [[ -f "$SCRATCH_OUTPUT_ROOT/$folder_name/$COMPLETION_MARKER" ]]
}

is_completed_config() {
    local score_rank="$1"
    local k="$2"
    local win_size="$3"
    local row_permutation
    local folder_name

    while IFS= read -r row_permutation; do
        folder_name="$(run_folder_name "$row_permutation" "$score_rank" "$k" "$win_size")"
        if ! is_completed_folder "$folder_name"; then
            return 1
        fi
    done < <(get_row_permutation_names)

    return 0
}

mapfile -t assigned_runs < <(
python3 - <<PY
import math

k_min = $k_min
k_max = $k_max
w_min = $win_min
w_max = $win_max
num_k_values = $NUM_K_VALUES
num_w_values = $NUM_WIN_VALUES
score_rank_values = [${score_rank_values_py}]
num_machines = $NUM_MACHINES
machine_id = $machine_id

def unique_sorted_ints(vals):
    return sorted(set(int(round(v)) for v in vals))

def geomspace_int(lo, hi, n):
    if n <= 1:
        return [lo]
    if lo <= 0 or hi <= 0:
        raise ValueError("geomspace_int requires positive bounds")
    vals = []
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    for i in range(n):
        t = i / (n - 1)
        vals.append(math.exp((1 - t) * log_lo + t * log_hi))
    vals = unique_sorted_ints(vals)
    vals[0] = lo
    vals[-1] = hi
    return vals

def linspace_int(lo, hi, n):
    if n <= 1:
        return [lo]
    vals = []
    for i in range(n):
        t = i / (n - 1)
        vals.append(lo + t * (hi - lo))
    vals = unique_sorted_ints(vals)
    vals[0] = lo
    vals[-1] = hi
    return vals

def fill_missing(vals, lo, hi, target):
    vals = sorted(set(vals))
    if len(vals) >= target:
        return vals[:target]

    existing = set(vals)
    candidates = [x for x in range(lo, hi + 1) if x not in existing]

    if not candidates:
        return vals[:target]

    need = target - len(vals)

    if need >= len(candidates):
        return sorted(vals + candidates)[:target]

    chosen = []
    if need == 1:
        chosen = [candidates[len(candidates) // 2]]
    else:
        for i in range(need):
            idx = round(i * (len(candidates) - 1) / (need - 1))
            chosen.append(candidates[idx])

    return sorted(set(vals + chosen))[:target]

# Log-like spacing is usually better for coverage over wide ranges.
k_values = geomspace_int(k_min, k_max, num_k_values)
w_values = geomspace_int(w_min, w_max, num_w_values)

# If rounding collapsed too many points, fill deterministically.
k_values = fill_missing(k_values, k_min, k_max, num_k_values)
w_values = fill_missing(w_values, w_min, w_max, num_w_values)

# If still short due to tiny ranges, fall back to linear spacing then refill.
if len(k_values) < num_k_values:
    k_values = fill_missing(linspace_int(k_min, k_max, num_k_values), k_min, k_max, num_k_values)
if len(w_values) < num_w_values:
    w_values = fill_missing(linspace_int(w_min, w_max, num_w_values), w_min, w_max, num_w_values)

runs = [(score_rank, k, w) for w in w_values for score_rank in score_rank_values for k in k_values]
raw_total_runs = len(runs)

unique_runs = []
seen = set()
for score_rank, k, w in runs:
    effective_score_rank = min(score_rank, k)
    run = (effective_score_rank, k, w)
    if run in seen:
        continue
    seen.add(run)
    unique_runs.append(run)

runs = unique_runs
runs.sort(key=lambda run: (run[2], run[0], run[1]))

assigned = [run for idx, run in enumerate(runs) if idx % num_machines == machine_id]

print(f"# RAW_TOTAL_RUNS {raw_total_runs}")
print(f"# DEDUPED_TOTAL_RUNS {len(runs)}")

for score_rank, k, w in assigned:
    print(f"{score_rank} {k} {w}")
PY
)

raw_total_runs_reported=""
deduped_total_runs=""
filtered_runs=()
for run in "${assigned_runs[@]}"; do
    if [[ "$run" == \#\ RAW_TOTAL_RUNS* ]]; then
        raw_total_runs_reported="${run##* }"
    elif [[ "$run" == \#\ DEDUPED_TOTAL_RUNS* ]]; then
        deduped_total_runs="${run##* }"
    elif [[ -n "$run" ]]; then
        filtered_runs+=("$run")
    fi
done
assigned_runs=("${filtered_runs[@]}")

echo "Machine ID: $machine_id / $((NUM_MACHINES - 1))"
echo "Matrix: $matrix_name"
echo "Method: $method_name"
echo "Score ranks: ${SCORE_RANK_VALUES[*]}"
echo "N: $N"
echo "Bounds: k in [$k_min, $k_max], win_size in [$win_min, $win_max]"
echo "Grid: ${NUM_K_VALUES} k-values x ${NUM_WIN_VALUES} win-values x ${NUM_SCORE_RANK_VALUES} score-ranks = $RAW_TOTAL_RUNS raw runs"
echo "Reported raw runs: ${raw_total_runs_reported:-unknown}"
echo "Deduped runs: ${deduped_total_runs:-unknown}"
echo "Assigned runs: ${#assigned_runs[@]}"
echo "Available CPUs: $available_cpus"
echo "CPU limit: ${CPU_LIMIT_OF_AVAILABLE_PERCENT}% of available CPUs = CPUQuota=${cpu_quota_percent}%"
echo "Local output root: $LOCAL_OUTPUT_ROOT"
echo "Scratch output root: $SCRATCH_OUTPUT_ROOT"
echo "Skip completed configs: $SKIP_COMPLETED_CONFIGS"
echo ""

for run in "${assigned_runs[@]}"; do
    read -r score_rank k win_size <<< "$run"
    #k=16
    #score_rank=16
    mem_size=$((k + win_size))
    log_filename="logs/${safe_matrix_name}_${method_name}_${FORGET_CONFIG_LABEL}_sr${score_rank}_mem${mem_size}_win${win_size}_k${k}.txt"

    if [[ "$SKIP_COMPLETED_CONFIGS" == "1" ]] && is_completed_config "$score_rank" "$k" "$win_size"; then
        echo "========================================"
        echo "Skipping completed config"
        echo "Machine:   $machine_id"
        echo "Matrix:    $matrix_name"
        echo "Method:    $method_name"
        echo "k:         $k"
        echo "win_size:  $win_size"
        echo "mem_size:  $mem_size"
        echo "score_rank:$score_rank"
        echo "Checked:   $LOCAL_OUTPUT_ROOT and $SCRATCH_OUTPUT_ROOT"
        echo ""
        continue
    fi

    echo "========================================"
    echo "Machine:   $machine_id"
    echo "Matrix:    $matrix_name"
    echo "Method:    $method_name"
    echo "k:         $k"
    echo "win_size:  $win_size"
    echo "mem_size:  $mem_size"
    echo "score_rank:$score_rank"
    echo "forget_cfg:$FORGET_CONFIG_LABEL"
    echo "Log file:  $log_filename"
    run_command=(
        systemd-run --user --scope -p "CPUQuota=${cpu_quota_percent}%"
        env
        "OUTPUT_NAME=$OUTPUT_NAME"
        "SCRATCH_OUTPUT_ROOT=$SCRATCH_OUTPUT_ROOT"
        "SKIP_COMPLETED_CONFIGS=$SKIP_COMPLETED_CONFIGS"
        "COMPLETION_MARKER=$COMPLETION_MARKER"
        python3 -u main.py "$matrix_name" "$method_name" "$win_size" "$k" "$score_rank" "${FORGET_ARGS[@]}"
    )
    echo "Command:"
    printf '  '
    printf '%q ' "${run_command[@]}"
    echo "2>&1 | tee -a \"$log_filename\""
    echo ""

    "${run_command[@]}" 2>&1 | tee -a "$log_filename"
    #break
done
