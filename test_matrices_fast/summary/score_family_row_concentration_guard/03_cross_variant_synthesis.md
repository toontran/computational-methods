# Cross-variant synthesis

Date: 2026-04-28
Family: row-concentration guard.

## D0 conclusion

Only D0 was closed in this round. D0 = `S6 * relH1(A_cur v)` is **killed**.
It passes gradient sanity and does not materially change the T2 oracle-vs-
hm_triplet ranking pattern, but T3 fails the reframed high-entropy
acceptance: diffuse-diffuse improves, while mixed-tail-balanced,
mixed-tail-sharp, and static-cex regress in cos1^2 versus the frozen S6
baseline.

See `variants/D0/synthesis.md` for the evidence table and run notes.

## Scope

D1 and D2 remain spec-only. They were not implemented, benchmarked, or
accepted/rejected in this closure. No family-level acceptance language
should treat D1/D2 as evidence.

## Propagation

Backlog propagation: FAM-02 D0 is closed as killed. The current sequencing
should no longer list "FAM-02 D0 close" as an active item.

Overview propagation: Q3's parameter-free multiplicative form is refuted
with a pointer to this family. This does not edit DIAG-01-owned HIGH/LOW
regime labels.

Toolkit propagation: no new diagnostic was added and no diagnostic was
closed, so `diagnostic_toolkit.txt` does not need an update.
