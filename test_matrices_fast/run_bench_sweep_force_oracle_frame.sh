#!/usr/bin/env bash
# AB-03 closure probe (frame variant): bench sweep with --force-oracle-frame
# (V_selected ← rank2_svd_frame(oracle_v1_proj, oracle_v2_proj, M_gain) at
# every block). Replaces BOTH V_default[:,0] and chosen_v2 with oracle
# projections. Establishes the cos² ceiling under the current
# rowspan-of-window constraint.
#
# Output: summary/score_family_aggregator_ablation/forceFrame_<matrix>_win64.*
set -u
cd "$(dirname "$0")"

export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

OUT=summary/score_family_aggregator_ablation
mkdir -p "$OUT"

MATRICES=(
  diffuse-diffuse
  residual-spiky-shocks
  mixed-tail-soft
  mixed-tail-sharp
  static-cex
  mixed-tail-balanced
  etf-basket-basis
)

POLICIES=(future_hmean_r_sk_g)

run_one() {
  local M="$1"
  local OUTDIR="$2"
  local JSON="$OUTDIR/forceFrame_${M}_win64.json"
  local CSV="$OUTDIR/forceFrame_${M}_win64.csv"
  local TXT="$OUTDIR/forceFrame_${M}_win64.txt"
  local LOG="$OUTDIR/forceFrame_${M}_win64.log"
  python half_window_sliding_hmean_experiment.py \
    --matrix "$M" \
    --policies "${POLICIES[@]}" \
    --rsk-variant S6_E2 \
    --force-oracle-frame \
    --half-win 32 \
    --json-out "$JSON" --csv-out "$CSV" --text-out "$TXT" \
    > "$LOG" 2>&1
  echo "done $M ($(date +%H:%M:%S))"
}

T0=$(date +%s)
echo "start $(date) — force-oracle-frame bench"

for M in "${MATRICES[@]}"; do
  run_one "$M" "$OUT" &
done
wait

T1=$(date +%s)
echo "all done. total elapsed $((T1 - T0))s"
