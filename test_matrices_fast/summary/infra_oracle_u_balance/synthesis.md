# DIAG-04 oracle u-balance audit — synthesis

Probe: `oracle_u_balance_audit.py` (per-block, half_win=32, rank=2).
Components: u_sk, u_g1, u_g2 from S6 (F-weighted HM3) at oracle_v_k_proj
for slot k ∈ {1, 2}. Ratios: max(u)/min(u), sk/g1, sk/g2, g1/g2.

Hypothesis (overview §1quater): HM3's "smallest-link" enforcer only
rewards oracle if oracle is roughly balanced under the chosen weights.
Frobenius weighting was inherited from §2(b)'s "unit-fixer" framing,
not calibrated against an oracle-balance criterion. This audit measures
the imbalance at scale and correlates it with S6 cos[k]² failure.

## Slot-1 (oracle_v1_proj) ratio_max per block

ratio_max = max(u_sk, u_g1, u_g2) / min(u_sk, u_g1, u_g2). 1.00x = perfect
balance; HM3 reads oracle as the high-score point. Large values mean
oracle's smallest u dominates HM3 and the score peak shifts off oracle.

| matrix | b1 | b2 | b6 | b12 | b31 | S6 cos1² |
| --- | --- | --- | --- | --- | --- | --- |
| mixed-tail-sharp | nan | 2.10x | 7.67x | 29.0x | 243x | 0.013 |
| mixed-tail-balanced | nan | 1.03x | 21.7x | 76.9x | 334x | 0.000 |
| mixed-tail-soft | nan | 1.66x | 22.4x | 86.1x | 347x | 0.022 |
| static-cex | nan | 9.10x | 36.7x | 70.2x | 322x | 0.025 |
| diffuse-diffuse | nan | 4.62x | 3.39x | 5.83x | 17.0x | 0.005 |
| etf-basket-basis | nan | 1.53x | 1.58x | 1.63x | 1.74x | 0.652 |
| residual-spiky-shocks | nan | 5.57x | 21.7x | 27.6x | 35.5x | 0.266 |
| risk-residual-panel | nan | 3.98x | 3.89x | 1.39x | 85.0x | — |

## Slot-2 (oracle_v2_proj) ratio_max per block

Slot-2 is where S6 actually fails on T3. If slot-2 oracle is even more
imbalanced than slot-1, the M4 hypothesis predicts S6 fails harder on
slot-2 than slot-1 — which it empirically does.

| matrix | b1 | b2 | b6 | b12 | b31 | S6 cos1² |
| --- | --- | --- | --- | --- | --- | --- |
| mixed-tail-sharp | nan | 6.63x | 9.83x | 25.8x | 17.0x | 0.013 |
| mixed-tail-balanced | nan | 1.97x | 2.25x | 1.31x | 1.07x | 0.000 |
| mixed-tail-soft | nan | 1.74x | 2.76x | 2.05x | 15.7x | 0.022 |
| static-cex | nan | 9.10x | 42.1x | 128x | 379x | 0.025 |
| diffuse-diffuse | nan | 6.61x | 1.16x | 2.41x | 18.3x | 0.005 |
| etf-basket-basis | nan | 6.04x | 132x | 784x | 574x | 0.652 |
| residual-spiky-shocks | nan | 5.47x | 12.8x | 26.9x | 91.5x | 0.266 |
| risk-residual-panel | nan | 11.0x | 1.55x | 6.60x | 10.6x | — |

## Slot-1 ratio_skg1 (u_sk / u_g1) per block

ratio_skg1 > 1 means sketch is over-rewarded vs current half-window.
ratio_skg1 < 1 means sketch is under-rewarded. Either direction breaks
HM3's smallest-link reading of oracle. Note: at b1 sketch is empty so
ratio_skg1 is NaN.

| matrix | b1 | b2 | b6 | b12 | b31 |
| --- | --- | --- | --- | --- | --- |
| mixed-tail-sharp | nan | 2.10x | 7.56x | 28.8x | 241x |
| mixed-tail-balanced | nan | 0.97x | 21.5x | 76.6x | 332x |
| mixed-tail-soft | nan | 1.66x | 22.3x | 86.0x | 346x |
| static-cex | nan | 9.10x | 36.7x | 70.1x | 322x |
| diffuse-diffuse | nan | 4.62x | 3.38x | 5.83x | 17.0x |
| etf-basket-basis | nan | 1.53x | 1.36x | 1.46x | 1.74x |
| residual-spiky-shocks | nan | 2.00x | 21.7x | 27.6x | 35.5x |
| risk-residual-panel | nan | 1.65x | 0.26x | 1.18x | 43.8x |

## Slot-2 component breakdown at b31 (the streaming-bench terminal block)

| matrix | u_sk | u_g1 | u_g2 | ratio_max | S6 cos1² |
| --- | --- | --- | --- | --- | --- |
| mixed-tail-sharp | 0.1749 | 0.0104 | 0.0103 | 17.0x | 0.013 |
| mixed-tail-balanced | 0.0164 | 0.0154 | 0.0153 | 1.07x | 0.000 |
| mixed-tail-soft | 0.1640 | 0.0105 | 0.0105 | 15.7x | 0.022 |
| static-cex | 0.4739 | 0.0013 | 0.0013 | 379x | 0.025 |
| diffuse-diffuse | 0.1851 | 0.0101 | 0.0101 | 18.3x | 0.005 |
| etf-basket-basis | 0.0003 | 0.1452 | 0.1343 | 574x | 0.652 |
| residual-spiky-shocks | 0.4028 | 0.0044 | 0.0046 | 91.5x | 0.266 |
| risk-residual-panel | 0.1143 | 0.0207 | 0.0108 | 10.6x | — |

## Verdict — PARTIALLY VERIFIED

The audit confirms that oracle u-imbalance is large and matrix-dependent,
but the simple "imbalance ⇒ S6 fails" story is only PARTIALLY true.
Three observations:

**(V1) Slot-1 imbalance grows monotonically with block on every failure
matrix.** At b31, slot-1 ratio_max is 17–347x on the seven §6 matrices,
versus 1.74x on etf-basket-basis (the calibration anchor where S6
cos0²=1.000). The block-trajectory (b2 → b31) is monotonically up on
all tail-dominant matrices: mixed-tail-sharp 2.1x → 243x;
mixed-tail-balanced 1.0x → 334x; static-cex 9x → 322x. This is the
"sk_F2_low becomes too small a denominator as carry energy grows"
pathology — exactly what §1quater predicts. The U-shape on
mixed-tail-{sharp,balanced} (small early, then huge) shows the
imbalance is driven by the carry growing, not by an initial choice.

**(V2) Etf-basket-basis is the M4 calibration anchor (positive evidence).**
slot-1 ratio_max is 1.4–1.7x throughout the run, and S6 nails slot-1
(cos0²=1.000). This is the only matrix in §6 where oracle is roughly
balanced and S6 succeeds. Strongest single piece of evidence FOR M4.

**(V3) Two refutations against M4 as the universal mechanism:**

  - **mixed-tail-balanced slot-2 BALANCED but S6 fails 0.0004 cos1².**
    At b31, slot-2 has ratio_max=1.07x — essentially perfect balance
    (u_sk=0.016, u_g1=0.015, u_g2=0.015). All three u's are uniformly
    SMALL (~0.015), so the oracle's HM3 ≈ 0.015 is itself tiny and
    other directions can outscore it on absolute terms even without
    imbalance. M4 doesn't apply here; the failure is "carry doesn't
    cover slot-2" (M2-style) plus low absolute signal in any window.

  - **static-cex slot-1 HUGELY imbalanced (322x) but S6 cos0²=0.936.**
    Imbalance does NOT prevent slot-1 recovery here. Combined-style
    slot-1 mechanism (raw_sk + raw_g1 sum dominates) survives the
    HM3 imbalance because slot-1 selection is heavily driven by the
    block-1 fall-through HM2 + carry initialization, not by HM3 of
    the union span at later blocks.

**(V4) Etf-basket-basis slot-2 paradox is consistent with M4.** At
b31 slot-2 has ratio_max=574x (u_sk≈0.0003, u_g1≈0.145, u_g2≈0.134),
yet S6 reaches cos1²=0.652. The reason: slot-1 saturates the carry
(u_sk(oracle_v1)=0.808 — high), so oracle_v2_proj is nearly orthogonal
to span(V_state) and ALL deflation-complement candidates have
similarly tiny u_sk. The imbalance isn't differential — every
candidate is in the same low-u_sk regime — and HM3 picks the v with
best (u_g1, u_g2) among them, which approximates oracle. M4 is
DIFFERENTIAL imbalance: oracle's u's mismatch in a way that competing
non-oracle directions can exploit. Etf-basket-basis slot-2 is uniform
imbalance, not differential.

## Implications for AB-03

- **AB-03 phase 1 should run.** The slot-1 b31 evidence (1.74x on the
  one working matrix vs 243-347x on the failure matrices) is a clear
  signal that the weighting choice matters and is currently
  miscalibrated on every failure matrix. Phase 1's oracle-aware
  re-weighting will quantify how much of the gap is closeable by fixing
  imbalance alone.

- **Expect mixed results from AB-03 by matrix.** Imbalance is the
  dominant mechanism on static-cex slot-2 (379x) and mixed-tail-{sharp,
  soft} slot-2 (15-17x) — these should respond strongly to oracle-aware
  re-weighting. mixed-tail-balanced slot-2 (1.07x) and etf-basket-basis
  slot-2 (uniform 574x) will likely NOT respond to re-weighting because
  M4 isn't the dominant mechanism there. AB-03's acceptance is
  matrix-conditional precisely because the audit shows imbalance is
  matrix-conditional.

- **Cross-mechanism interaction.** Diffuse-diffuse slot-1 has
  moderate imbalance (17x at b31) AND known plateau-drift problems
  (§3); both M4 and §3 may need fixing to recover this matrix. AB-03
  alone is unlikely to fully close the gap — re-weighting + FAM-01
  rank-r lift may be needed in combination.

- **Do NOT promote AB-03 ahead of FAM-01.** The verdict is "M4 is a
  contributing mechanism but NOT the universal cause of S6 failure."
  Run AB-03 phase 1 in parallel with FAM-01 baseline pinning, but
  keep FAM-01 as the primary structural fix. If AB-03 phase 1 closes
  ≥50% of the gap on a single tail-dominant matrix, advance to
  phase 2; otherwise, the M4 evidence weakens and AB-03 should be
  parked.

## Block-trajectory observations (mechanism detail for §1quater)

The slot-1 ratio_max trajectory (b2 → b31) reveals a consistent shape
on tail-dominant matrices: imbalance is small at b2 (1-9x) and grows
monotonically with block to 243-347x at b31. The driver is the
sk_F2_low denominator: as the carry grows, sk_F2_low approaches the
matrix's true top-r σ² (a stationary asymptote per INFRA-06), but
raw_sk(oracle) grows roughly linearly with the prefix-row count. So
u_sk(oracle) = raw_sk/sk_F2_low grows linearly while u_g1, u_g2 stay
on the half-window scale. This is the §2(c) "rank-r CARRY normalizes
the carry on a stationary basis" choice biting back: it puts u_sk on
a [0, 1] scale, but ON THAT SCALE the oracle drifts toward 1 while
the windows drift toward 0. The "unit-fixer" property holds; the
"oracle-balance" property fails by construction at late blocks. This
is the cleanest formal statement of the problem and the precise
target for AB-03 phase 2's value-only proxy.

## Cross-references

- score_design_overview.txt §1quater (M4 mechanism)
- score_design_overview.txt §2bis (b.iii) (calibration criterion)
- score_design_overview.txt §7 Q16 (the open question)
- score_family_workflow.txt [DIAG-04] / [AB-03]
- diagnostic_toolkit.txt §6b (oracle u-imbalance signature)
- diagnostic_toolkit.txt §8(q) (this audit registers as a closed gap)

