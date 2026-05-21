#!/usr/bin/env bash
# Run the future_hmean_evidence policy on the same 17-matrix sweep as
# run_bench_sweep_v2.sh, leaving the existing 4-policy outputs untouched.
set -u
cd "$(dirname "$0")"

export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

OUT=summary/bench_matrix_sweep_evidence
mkdir -p "$OUT"

POLICIES=(future_hmean_evidence)

MATRICES=(
  crowded-strategy diffuse-diffuse etf-basket-basis execution-cost-slippage
  futures-term-structure intraday-liquidity-shape macro-factor-panel mixed-tail-balanced
  mixed-tail-soft options-vol-surface rates-cross-currency realized-vol-corr
  residual-spiky-shocks risk-residual-panel stat-arb-spreads static-cex
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
    --half-win 32 \
    --json-out "$JSON" --csv-out "$CSV" --text-out "$TXT" \
    > "$LOG" 2>&1
  echo "done $M ($(date +%H:%M:%S))"
}

T0=$(date +%s)
echo "start $(date) - 16 batched matrices in 4-way batches"

for i in 0 4 8 12; do
  for j in 0 1 2 3; do
    M="${MATRICES[$((i+j))]}"
    run_one "$M" "$OUT" &
  done
  wait
  echo "batch starting at idx=$i complete ($(date +%H:%M:%S))"
done

echo "start mixed-tail-sharp solo $(date)"
python half_window_sliding_hmean_experiment.py \
  --matrix mixed-tail-sharp \
  --policies "${POLICIES[@]}" \
  --half-win 32 \
  --json-out "$OUT/mixed-tail-sharp_win64.json" \
  --csv-out "$OUT/mixed-tail-sharp_win64.csv" \
  --text-out "$OUT/mixed-tail-sharp_win64.txt" \
  > "$OUT/mixed-tail-sharp_win64.log" 2>&1
echo "done mixed-tail-sharp ($(date +%H:%M:%S))"

T1=$(date +%s)
echo "all done. total elapsed $((T1 - T0))s"
