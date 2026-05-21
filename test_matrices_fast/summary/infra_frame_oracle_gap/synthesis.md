# FAM-01-DIAG synthesis

Run date: 2026-04-28
Block: b31
Elapsed seconds: 436.635

Canonical screen: `hm` with `anchor=free`.
Optimizer: analytic retraction-based Stiefel gradient ascent.

Verdict by matrix:
- diffuse-diffuse: FAIL evidence model insufficient; delta=+0.00942303; pa_cos2=[0.5068847404017041, 0.057235017558232716]
- residual-spiky-shocks: FAIL evidence model insufficient; delta=+0.00504075; pa_cos2=[0.966536297309988, 0.8667304706294771]
- mixed-tail-soft: FAIL evidence model insufficient; delta=+0.0157919; pa_cos2=[0.8962681435310151, 0.4738612166741599]
- mixed-tail-sharp: FAIL evidence model insufficient; delta=+0.0130147; pa_cos2=[0.970976886884803, 0.4842994309518743]
- static-cex: FAIL evidence model insufficient; delta=+0.0347974; pa_cos2=[0.3308217404577158, 0.30115642936484516]
- mixed-tail-balanced: FAIL evidence model insufficient; delta=+0.0120987; pa_cos2=[0.9879456376794037, 0.5295501897313308]
- etf-basket-basis: FAIL evidence model insufficient; delta=+0.0335412; pa_cos2=[0.9955647171557913, 0.1536517023651401]

Summary:
- delta > 0: diffuse-diffuse, residual-spiky-shocks, mixed-tail-soft, mixed-tail-sharp, static-cex, mixed-tail-balanced, etf-basket-basis
- delta <= 0: none

Conclusion:
- The canonical magnitude-only HM frame score fails S-2 on every probed
  matrix. This is evidence-model insufficiency at rank 2, not merely
  greedy anchoring.
- Anchor sensitivity narrows the weak slot on some matrices but does not
  make Z_oracle the maximizer uniformly. Static-cex fails even with either
  oracle column fixed.
- Required evidence-augmentation ablations are recorded separately in
  `summary/infra_frame_oracle_gap_ablations/`; both HM x total energy and
  HM x Gram/right-space agreement also have delta > 0 on all seven
  matrices in the free-frame screen.
