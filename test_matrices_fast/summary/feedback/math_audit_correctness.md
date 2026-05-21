# Mathematical-Correctness Advisor: equation and claim audit

Sources: DOC A `../reports/approximation/new_approx_combined.txt`; DOC B
`summary/overview/score_design_overview.txt`.

## Fatal / Major flags

- **Major (Certain, Method)**: DOC B's rank-r bound
  `u_X(V) in [0, r/r_max(A_X)]` is not correct as a general bound for
  `u_X(V)=||A_X V||_F^2/||A_X||_F^2`. For orthonormal `V`,
  `0 <= u_X(V) <= 1`. A value like `r/rank(A_X)` is only an average-energy
  heuristic under equal singular values or random subspaces, not an upper
  bound. This affects the claimed scale interpretation of the rank-r lift.

- **Major (Likely, Method)**: the claim that increasing to `r=3` or `r=4`
  makes the plateau dimension shrink is not mathematically established.
  The Stiefel/Grassmann search dimension generally grows with `r` in the
  relevant range; rank-r may still help by changing the target to a subspace
  and removing greedy anchoring, but not by the stated dimension argument.

## Per-equation audit

| Equation / claim | Status | Notes |
|---|---|---|
| DOC A `h_2(y)=-log(||y||_4^4/||y||_2^4)=4 log||y||_2-4 log||y||_4` | Correct | Requires `y != 0`, which DOC A states. |
| DOC A inequality `(*)` under A2 and `||N_wv||_4 <= ||Av||_4` | Conditional | Algebra is coherent under the stated assumptions; the assumptions are empirical/modeling assumptions, not identities. |
| DOC A substitution from A1/A2 into combined score | Conditional | Correct as a surrogate derivation if A1/A2 are accepted. DOC B correctly treats this as brittle operationally. |
| DOC A pooled entropy formula for `C_w=[R;A_w]` | Correct | `||C_wv||_p^p` decomposes additively over stacked rows as written. |
| DOC A simplified multiplier `phi_w=(||y||_4^4/||y||_2^4)^{c_w}` | Correct | Since `c_w<0` when `row(N_w)<n`, lower concentration increases `phi_w`; DOC B should keep this sign bridge explicit. |
| DOC A gradient of `log phi_w` | Correct | The derivative of `log||C_wv||_4` gives `C_w^T(C_wv)^{odot 3}/||C_wv||_4^4`; denominator nonzero conditions are needed. |
| DOC A tangent projection `(I-QQ^T-vv^T) grad` | Conditional | Correct when `Q` is orthonormal and `Q^T v=0`. If not, the projector needs the projector onto `span(Q)` rather than literal `QQ^T`. |
| DOC B `B_union := rowspan([B;A_cur;A_fut]) = range([B;A_cur;A_fut]^T) subset R^d` | Correct | This fixes the older dimensional hazard where `range([B;A_w;A_fut])` would have meant column space in row-coordinate space. |
| DOC B HM3 example `HM3(0.48,0.0012,0.0012) ~= 0.0018` | Correct | Uses `HM_k=k/(sum_i 1/u_i)`. Computation is approximately right. |
| DOC B statement `HM3` is dominated by smallest argument | Correct | Mathematically true for positive arguments. "Hard constraint" is rhetorical; the score remains smooth and positive when all `u_i>0`. |
| DOC B `u_X(V)=||A_XV||_F^2/||A_X||_F^2` for `V in St(d,r)` | Correct | Dimension should be `St(d,r)`, not `St(n,r)`. The formula itself is right for a right subspace. |
| DOC B `u_X(V) in [0, r/r_max(A_X)]` | Wrong | General bound is `[0,1]`. `r/rank(A_X)` is not an upper bound unless additional equal-spectrum assumptions are imposed, and even then it is more like a baseline fraction. |
| DOC B right-orthogonal invariance `score(VO)=score(V)` | Correct | Frobenius norms are invariant under right multiplication by `O in O(r)`, so the frame score is a Grassmann/subspace function. |
| DOC B scalar level set: sphere in `dim(B_union)=64` gives a 62-dimensional regular level set | Conditional | Dimension arithmetic is right for a regular value of a smooth scalar score on `S^63`. It does not by itself prove a wide near-optimal plateau or optimizer drift. |
| DOC B "with `r=3` or `r=4` the plateau dimension shrinks" | Wrong / Unsupported | The search manifold dimension changes to `dr-r(r+1)/2` for Stiefel or `r(d-r)` for Grassmann, which increases from `r=2` to `r=3/4` when `d` is large. Any benefit must come from target/score geometry, not a generic dimension shrink. |
| DOC B block-1 equivalence HM2 vs weighted HM-evi | Conditional | Correct up to per-block constants if `cur_F2 ~= fut_F2` and the weights are equal enough. It is not an exact identity when denominators differ materially. |
| DOC B `argmax k(v)=argmax_i (state.V[:,i]^T v)^2` | Conditional | Well-defined only with a tie convention. This matters for differentiability of per-vector weights near ties. |

## Recommended fixes

1. Replace `u_X(V) in [0, r/r_max(A_X)]` with `0 <= u_X(V) <= 1`; if an
   average-energy baseline is useful, state it separately as a heuristic.
2. Rewrite the rank-r motivation: "rank-r scoring matches the operational
   subspace target and tests greedy-anchor artifacts" rather than "the plateau
   dimension shrinks."
3. Fix `V in St(n,r)` to `V in St(d,r)` in the rank-r lift section.
4. Add explicit nonzero/tie conventions for HM slots, projections, and
   per-vector `k(v)` weights where those are promoted from diagnostics to
   implementation.
