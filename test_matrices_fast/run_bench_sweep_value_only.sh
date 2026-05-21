#!/usr/bin/env bash
# INFRA-10: value-only online baseline
# Re-run the 7-matrix §6 sweep at half_win=32 using the corrected
# (value-only) candidate pool for future_hmean_online.
# The previous run at summary/bench_matrix_sweep/ used an oracle-aware
# pool (containing q2_vs_q1oracle = V_exact[:,1] projected into the row
# span); those numbers are kept as a labeled "ceiling" reference.
set -u
cd "$(dirname "$0")"

export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

OUT=summary/bench_matrix_sweep_value_only_online
POLICIES=(isvd combined hybrid future_hmean_online future_hmean_r_sk_g)

# 7-matrix suite from score_design_overview.txt §6 table.
# (risk-residual-panel exists in the existing bench_matrix_sweep/, so include
# it as the optional 8th.)
MATRICES=(
  mixed-tail-sharp mixed-tail-balanced mixed-tail-soft static-cex
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
echo "start $(date) — value-only online sweep, 8 matrices in 4-way batches"

for i in 0 4; do
  for j in 0 1 2 3; do
    M="${MATRICES[$((i+j))]}"
    run_one "$M" "$OUT" &
  done
  wait
  echo "batch starting at idx=$i complete ($(date +%H:%M:%S))"
done

T1=$(date +%s)
echo "all done. total elapsed $((T1 - T0))s"
