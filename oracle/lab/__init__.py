"""
Oracle Lab — offline research / robustness harness (ANALYTICS ONLY).

The Lab is an *import-only, offline* research surface that reuses the exact
production components (``MarketView`` / ``HistoricalMarketView``,
``oracle_agents``, ``oracle_voting``, ``ev_engine``, ``cost_model``,
``expected_move_engine``) to build point-in-time feature/label datasets and to
measure strategy robustness (metrics, parameter sweeps, walk-forward).

It has NO live effect. Nothing in this package opens, sizes, prices, blocks, or
alters a real/paper trade, reads creds, or hits the network. It exists so a
researcher can ask "would this rule have worked?" without touching the live
decision path. The live path in ``smart_trader.py`` is byte-identical whether or
not this package is imported.

Modules:
  metrics              pure performance/robustness statistics over trade records
  dataset              point-in-time FeatureSnapshot rows via MarketView (no leak)
  experiment           ExperimentConfig + deterministic run_experiment
  parameter_sweep      deterministic cartesian sweep over a param grid
  walk_forward         TRAIN -> VALIDATE -> FREEZE -> TEST -> ROLL, IS vs OOS
  parameter_stability  plateau-vs-spike robustness of the swept parameters

Design rules (mirror oracle/signals/candlestick_patterns.py):
  * Pure where possible. No network, no creds, no live-trade effect.
  * Deterministic given (dataset, config, seed).
  * Prefer None/empty over a guess; never raise on malformed input in the
    research path (fail-open). Leakage checks are the one place we DO raise
    when temporal integrity is enabled.
"""

from oracle.lab import metrics  # noqa: F401

__all__ = ["metrics"]

# Heavier sub-modules (experiment/sweep/walk_forward/stability) are imported
# lazily by callers to keep ``import oracle.lab`` cheap and side-effect free.
