#!/usr/bin/env bash
# Run the future_hmean_r_sk_g policy with --rsk-variant D0 on the same
# 7-matrix suite used by summary/bench_matrix_sweep_r_sk_g_S6/.
# FAM-02 D0 (row-concentration guard): summary/score_family_row_concentration_guard/.
set -u
cd "$(dirname "$0")"

export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

OUT=summary/score_family_row_concentration_guard/variants/D0/bench
mkdir -p "$OUT"

POLICIES=(isvd combined future_hmean_r_sk_g)

# 7-matrix suite from summary/bench_matrix_sweep_r_sk_g_S6/.
MATRICES=(
  static-cex mixed-tail-sharp diffuse-diffuse mixed-tail-balanced
  mixed-tail-soft etf-basket-basis residual-spiky-shocks
)

# risk-residual-panel is in the §6 table but not in the S6 baseline folder;
# add it so we can size the spiky-residual lift across both spiky matrices.
EXTRA_MATRICES=(risk-residual-panel)

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
    --rsk-variant D0 \
    --half-win 32 \
    --json-out "$JSON" --csv-out "$CSV" --text-out "$TXT" \
    > "$LOG" 2>&1
  echo "done $M ($(date +%H:%M:%S))"
}

T0=$(date +%s)
echo "start $(date) — D0 streaming bench, 8 matrices"

# Batch: 4 in parallel, then the rest.
for M in "${MATRICES[@]}" "${EXTRA_MATRICES[@]}"; do
  run_one "$M" "$OUT" &
  while [[ $(jobs -p | wc -l) -ge 4 ]]; do
    sleep 1
  done
done
wait

T1=$(date +%s)
echo "all done. total elapsed $((T1 - T0))s"
