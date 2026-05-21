Documentation cleanup: B_union, targets, frame/subspace language
================================================================

Date: 2026-04-28

Scope
-----

DOC-01 cleaned the active overview/toolkit/workflow documentation so the
search domain and target labels are consistent before this terminology is
copied back into reports.

Files touched
-------------

  - summary/overview/score_design_overview.txt
  - summary/overview/diagnostic_toolkit.txt
  - summary/overview/score_family_workflow.txt
  - summary/feedback/doc_cleanup.md

Terminology now in force
------------------------

  - `B_union` is the right row-space audit domain:
      `rowspan([B; A_cur; A_fut]) = range([B; A_cur; A_fut]^T) subset R^d`.
    It is not the column-space/range of the raw vertically stacked matrix.
  - Vector tests use slotwise targets `v_i` / `oracle_v{i}_proj`.
  - Oriented-frame tests use `Z_oracle`, `Z_winner`, and `Z_rowcheat`.
  - Grassmann-invariant rank-r tests report `span(V_r)` recovery through
    principal angles. Residual oriented-frame gaps under a right-rotation
    invariant objective are not described as span-recovery failures.

Online-baseline note
--------------------

DOC-01 was terminology-focused. The active docs have since also been patched
by INFRA-10 / FB-001: bare `online` in acceptance criteria means the
value-only `future_hmean_online` rerun under
`summary/bench_matrix_sweep_value_only_online/`. Older
`summary/bench_matrix_sweep/` references are oracle-aware ceiling artifacts
because their pool included `q2_vs_q1oracle`; keep them only as labeled
upper-bound context, never as ship targets.

Remaining report-sync work
--------------------------

  - Propagate these definitions into `../reports/approximation/current_report.txt`.
  - Re-check any report-local score-family prose copied from older notes for
    bare `range([B; A_cur; A_fut])`, unlabeled "union span", or frame/subspace
    target slippage.
