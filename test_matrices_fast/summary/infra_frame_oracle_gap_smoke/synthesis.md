# FAM-01-DIAG synthesis

Run date: 2026-04-28
Block: b31
Elapsed seconds: 60.474

Canonical screen: `hm` with `anchor=free`.
Optimizer: analytic Stiefel gradient ascent for HM; non-HM ablations use the slower finite-difference plug-in path.

Verdict by matrix:
- mixed-tail-sharp: FAIL evidence model insufficient; delta=+0.0121885; pa_cos2=[0.9920792289843017, 0.6010342414791667]
- static-cex: FAIL evidence model insufficient; delta=+0.0332697; pa_cos2=[0.19870830711178955, 0.008773971641728665]

Summary:
- delta > 0: mixed-tail-sharp, static-cex
- delta <= 0: none
