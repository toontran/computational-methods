# FAM-09 A0 synthesis

Date: 2026-04-29

## Scope

A0 is a diagnostic-only prototype for stability-weighted evidence:

```text
u_X_stab(v) = u_X(v) / (1 + lambda * CV_X(v; p=0.50))
Score_A0(v) = HM2/HM3 over u_X_stab
```

It reuses the DIAG-03 row-subsample helpers and candidate panel. No gradient
support, optimizer integration, or T3 streaming bench was attempted.

Command run:

```bash
python summary/score_family_stability_weighted_evidence/prototype_A0.py \
  --matrices static-cex diffuse-diffuse mixed-tail-sharp \
  --blocks 1 6 12 31 \
  --n-subsamples 12 \
  --lam 1.0
```

Artifacts:

- `summary/score_family_stability_weighted_evidence/variants/A0/diagnostics/prototype_A0.csv`
- `summary/score_family_stability_weighted_evidence/variants/A0/diagnostics/prototype_A0_summary.md`

## Result

Across 24 matrix/block/slot panels:

| matrix | panels | improved | unchanged | worsened | mean raw_rank - stab_rank |
|---|---:|---:|---:|---:|---:|
| diffuse-diffuse | 8 | 1 | 7 | 0 | +0.125 |
| mixed-tail-sharp | 8 | 4 | 4 | 0 | +0.875 |
| static-cex | 8 | 3 | 4 | 1 | +0.500 |
| total | 24 | 8 | 15 | 1 | +0.500 |

Positive movement means the oracle candidate moved up after the stability
penalty. The best evidence is `mixed-tail-sharp`, where A0 moves the oracle
above `s6_opt` on b1 slot 1, b1 slot 2, and b6 slot 1, and moves the b31
slot-1 winner from `mgain` to `oracle`.

The main regression is `static-cex` b31 slot 2: raw HM ranks oracle first,
while A0 ranks `sketch` first and moves oracle from rank 1 to rank 2.
This is enough to block optimizer/T3 promotion for the current all-window
`lambda=1` penalty.

## Interpretation

DIAG-03's stability signal is wireable into score evidence without immediate
collapse in this fixed-panel diagnostic. The penalty preferentially shrinks
fragile high-HM candidates enough to improve several high-entropy panels, with
the clearest gain on `mixed-tail-sharp`.

However, the regression shows that "penalize every window equally by CV" is too
coarse. A shippable FAM-09 variant should not proceed by simply dropping this
Monte-Carlo penalty into the optimizer. The next useful variant is A1:

- deterministic row partitions or cached subsample masks, so the objective is
  reproducible during optimization;
- a lambda/window sweep, especially `g2`-only or weaker `sk/g1` penalties,
  because DIAG-03's strongest predictive metric was `g2_p50_cv`;
- Tier-A S-1/S-3 fixed-block checks before any T3 run.

## Verdict

ITERATE. A0 validates the diagnostic plumbing and gives a positive initial
signal, but it is not safe to ship or bench as a streaming policy yet.
