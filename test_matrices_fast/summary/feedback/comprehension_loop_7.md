## Comprehension Loop 7

No source edits were made.

- **Major** — Appendix A.2's joint-control sentence contradicts its table. Lines 577--585 list joint state alignment as $0.834$, $0.897$, $0.988$ for blocks $6,12,31$, but lines 600--604 say it gives $0.989$ at all three blocks. A reader cannot tell which is authoritative.

- **Moderate** — V3 is labelled an H1 baseline even though it tests optimisation, not an H1-style aggregator/weighting change. Lines 328--339 define H1 as smarter aggregation/weighting; line 412 says V3 has "no aggregator change" and only swaps greedy for joint optimisation. Rename it as an optimisation-control baseline, or explain why it belongs under H1.

- **Moderate** — The singular-goal quote still overstates uniqueness before the frame/subspace convention appears. Lines 281--283 ask for the "global oracle subspace" as the "unique maximiser," while lines 317--319 later say rotation-invariant scores only identify the two-plane. Move the caveat earlier or say "unique maximising subspace."

- **Minor** — The V1 description compresses a weighted Rayleigh quantity into "average squared singular value." Lines 417--420 define $\|A_{\mathrm{fut}}^\top A_{\mathrm{fut}}Z\|_F^2/\|A_{\mathrm{fut}}Z\|_F^2$; "average" can read as an unweighted mean. Say "Rayleigh-style future-direction strength" or specify the weighting.

- **Minor** — Appendix A.4 says "oracle slots are roughly balanced" from a table that only reports $u_{\mathrm{sk}}/u_{\mathrm{cur}}$ at $v_1^\star$ (lines 667--686). Add that this is a proxy table, or include $u_{\mathrm{fut}}$ / max-min if the claim is about all slots.
