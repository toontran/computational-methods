# FAM-03 Gradient Theory

Date: 2026-04-29

E0 uses a linear trace objective on Stiefel frames:

`Score_E0(Z) = sum_X alpha_X trace(Z^T C_X Z)`

where `C_X = A_X^T A_X / ||A_X||_F^2`, `X in {sk, cur, fut}` for available
windows, and `alpha_X = 1` for the E0 spec. At b31 all three windows are
available; at b1 the sketch term is omitted.

Equivalently:

`Score_E0(Z) = trace(Z^T C_E0 Z)`

with:

`C_E0 = sum_X alpha_X A_X^T A_X / ||A_X||_F^2`.

## Euclidean Gradient

Because each `C_X` is symmetric:

`grad_E Score_E0(Z) = 2 C_E0 Z`.

Implementation form:

`grad += 2 * alpha_X * A_X.T @ (A_X @ Z) / ||A_X||_F^2`

for every available window with positive Frobenius denominator.

## Stiefel Tangent Projection

Use the shared INFRA-02 projection:

`P_Z(G) = G - Z sym(Z^T G)`.

The optimizer should first project the Euclidean gradient into `B_union`, then
project to `T_Z St(d, r)`, matching `probe_frame_oracle_gap.py`.

## Retraction And Check

Use the existing polar retraction from `stiefel_grad_check.py` /
`probe_frame_oracle_gap.py`. T1 should run central finite differences along
projected tangent directions and require relative error below `1e-7`, matching
the existing Stiefel acceptance bar.

Because E0 is exactly a trace form, it should also agree with the known-correct
trace sanity harness. Any failure here is an implementation error, not a
mathematical ambiguity.

