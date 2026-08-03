# Tactical claim benchmark cold archive

This directory preserves the superseded closed-claim experiment and open-nomination support files for audit only. The historical `p0_open/` directory was not present when migration ran, so no P0-open output snapshot could be copied.

- `code/` is the final source snapshot before cleanup.
- `code/stage2b_legacy/` preserves the retired direct/hybrid/closed-catalog
  commentary pipeline and its CLI. It is not importable or runnable in place.
- `outputs/closed_claim/` contains runs, GT, reports, judge decisions, and judge frame bundles.
- `outputs/superseded/` contains older open-nomination inputs and reports.

Nothing here is imported or tested by the active pipeline. Direct execution is not supported.
