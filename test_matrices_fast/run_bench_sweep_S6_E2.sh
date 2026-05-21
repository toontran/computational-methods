#!/usr/bin/env bash
# Run the 7-matrix bench sweep for the S6_E2 weighting-ablation variant
# (AB-03 phase 1, T3). Mirrors run_bench_sweep_S6_GM.sh / S6_OP. Output:
# summary/score_family_aggregator_ablation/S6_E2_<matrix>_win64.{json,csv,txt,log}
set -u
cd "$(dirname "$0")"

export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

OUT=summary/score_family_aggregator_ablation
mkdir -p "$OUT"

MATRICES=(
  static-cex
  mixed-tail-sharp
  mixed-tail-balanced
  mixed-tail-soft
  diffuse-diffuse
  etf-basket-basis
  residual-spiky-shocks
)

POLICIES=(combined isvd future_hmean_r_sk_g)

run_one() {
  local M="$1"
  local OUTDIR="$2"
  local JSON="$OUTDIR/S6_E2_${M}_win64.json"
  local CSV="$OUTDIR/S6_E2_${M}_win64.csv"
  local TXT="$OUTDIR/S6_E2_${M}_win64.txt"
  local LOG="$OUTDIR/S6_E2_${M}_win64.log"
  python half_window_sliding_hmean_experiment.py \
    --matrix "$M" \
    --policies "${POLICIES[@]}" \
    --rsk-variant S6_E2 \
    --half-win 32 \
    --json-out "$JSON" --csv-out "$CSV" --text-out "$TXT" \
    > "$LOG" 2>&1
  echo "done $M ($(date +%H:%M:%S))"
}

T0=$(date +%s)
echo "start $(date) — 7 matrices in parallel"

for M in "${MATRICES[@]}"; do
  run_one "$M" "$OUT" &
done
wait

T1=$(date +%s)
echo "all done. total elapsed $((T1 - T0))s"
