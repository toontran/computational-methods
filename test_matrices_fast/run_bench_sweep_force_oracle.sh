#!/usr/bin/env bash
# AB-03 closure probe: bench sweep with --force-oracle-v2 (chosen_v2 ←
# V_exact[:,1] projected into rowspace(A_sketch ∪ A_h1 ∪ A_h2)). Bypasses
# the rsk_g optimizer at every block. Tests whether the cos1² regression
# under S6_E2 is reachability (oracle is the right pick → forced bench
# recovers cos1²) or streaming-state divergence (still bad even when
# every pick is the oracle projection).
#
# Output: summary/score_family_aggregator_ablation/forceO2_<matrix>_win64.*
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
  local JSON="$OUTDIR/forceO2_${M}_win64.json"
  local CSV="$OUTDIR/forceO2_${M}_win64.csv"
  local TXT="$OUTDIR/forceO2_${M}_win64.txt"
  local LOG="$OUTDIR/forceO2_${M}_win64.log"
  python half_window_sliding_hmean_experiment.py \
    --matrix "$M" \
    --policies "${POLICIES[@]}" \
    --rsk-variant S6_E2 \
    --force-oracle-v2 \
    --half-win 32 \
    --json-out "$JSON" --csv-out "$CSV" --text-out "$TXT" \
    > "$LOG" 2>&1
  echo "done $M ($(date +%H:%M:%S))"
}

T0=$(date +%s)
echo "start $(date) — force-oracle-v2 bench"

for M in "${MATRICES[@]}"; do
  run_one "$M" "$OUT" &
done
wait

T1=$(date +%s)
echo "all done. total elapsed $((T1 - T0))s"
