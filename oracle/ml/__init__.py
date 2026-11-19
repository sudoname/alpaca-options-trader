"""Oracle 4.0 ML research package (research-only; no production imports).

Modules:
    schema   -- typed records, outcome-status taxonomy, UTC helpers
    ingest   -- research-ledger (sqlite) readers with dedup + validation
    features -- point-in-time feature construction (as-available-at-decision)
    labels   -- multi-horizon forward labels with explicit censoring
    dataset  -- CLI: build a versioned feature/label store + manifest

Nothing in this package is imported by any production trading path.
"""
