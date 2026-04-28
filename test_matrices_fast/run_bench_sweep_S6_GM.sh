#!/usr/bin/env bash
# Run the 7-matrix bench sweep for the S6_GM aggregator-ablation variant
# (AB-01). Mirrors run_bench_sweep_v2.sh and the existing S6 sweep CLI.
# Output: summary/score_family_aggregator_ablation/<matrix>_win64.{json,csv,txt,log}
set -u
cd "$(dirname "$0")"

export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

OUT=summary/score_family_aggregator_ablation
mkdir -p "$OUT"

# 7-matrix S6 suite (matches summary/bench_matrix_sweep_r_sk_g_S6/).
MATRICES=(
  static-cex
  mixed-tail-sharp
  mixed-tail-balanced
  mixed-tail-soft
  diffuse-diffuse
  etf-basket-basis
  residual-spiky-shocks
)

# Same 3 baseline policies used in the S6 sweep, plus future_hmean_r_sk_g
# with --rsk-variant S6_GM. The S6 (HM3) numbers are already in
# summary/bench_matrix_sweep_r_sk_g_S6/; we reproduce the baselines so the
# aggregator-ablation folder is self-contained.
POLICIES=(combined isvd future_hmean_r_sk_g)

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
    --rsk-variant S6_GM \
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
