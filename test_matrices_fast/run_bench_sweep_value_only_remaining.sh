#!/usr/bin/env bash
# Run remaining 7 matrices (mixed-tail-sharp already done as smoke test).
set -u
cd "$(dirname "$0")"

export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

OUT=summary/bench_matrix_sweep_value_only_online
POLICIES=(isvd combined hybrid future_hmean_online future_hmean_r_sk_g)

MATRICES=(
  mixed-tail-balanced mixed-tail-soft static-cex
  diffuse-diffuse etf-basket-basis residual-spiky-shocks risk-residual-panel
)

run_one() {
  local M="$1"
  local OUTDIR="$2"
  local JSON="$OUTDIR/${M}_win64.json"
  local CSV="$OUTDIR/${M}_win64.csv"
  local TXT="$OUTDIR/${M}_win64.txt"
  local LOG="$OUTDIR/${M}_win64.log"
  python half_window_sliding_hmean_experiment.py \
    --matrix "$M" \
    --policies "${POLICIES[@]}" \
    --rsk-variant S6 \
    --half-win 32 \
    --json-out "$JSON" --csv-out "$CSV" --text-out "$TXT" \
    > "$LOG" 2>&1
  echo "done $M ($(date +%H:%M:%S))"
}

T0=$(date +%s)
echo "start $(date) — value-only online sweep, 7 matrices in 4-way batches"

# Batch 1: 4 matrices in parallel
for j in 0 1 2 3; do
  M="${MATRICES[$j]}"
  run_one "$M" "$OUT" &
done
wait
echo "batch 1 complete ($(date +%H:%M:%S))"

# Batch 2: remaining 3 matrices
for j in 4 5 6; do
  M="${MATRICES[$j]}"
  run_one "$M" "$OUT" &
done
wait
echo "batch 2 complete ($(date +%H:%M:%S))"

T1=$(date +%s)
echo "all done. total elapsed $((T1 - T0))s"
