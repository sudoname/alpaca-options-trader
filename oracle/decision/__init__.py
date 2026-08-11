"""
Oracle Decision Kernel — one decision path for live, paper, and backtest.

The whole point of this package is *convergence*. Today the live/paper path in
``smart_trader.py`` and the various ``backtest_*.py`` scripts each decide
direction, EV, PoP and vetoes their own way, so a parameter sweep would tune a
system that is not the system that trades. This package extracts the LIVE
decision logic into a single pure function::

    decide(snapshot, portfolio_state, strategy_state, config) -> Decision

and nothing else. ``decide`` reads a frozen ``Snapshot`` (point-in-time inputs)
and returns an immutable ``Decision``. It performs NO I/O and submits NO orders
— execution is a separate concern handled by the ``ExecutionClient`` adapters.

Design rules (mirror oracle/signals/candlestick_patterns.py):
  * Pure compute. No network, no creds, no broker calls, no live effect.
  * Deterministic: identical (snapshot, portfolio, strategy, config) inputs
    yield an equal Decision, so ``decision_live == decision_backtest``.
  * Fail-open: a bad slice degrades to abstain (NO-TRADE), never raises in the
    decision path. Direction stays an OUTPUT, never an input.
  * Flag-gated at the call sites: when ``ENABLE_UNIFIED_DECISION_KERNEL`` is OFF
    the legacy ``smart_trader`` path is byte-identical.

Modules:
  schema     frozen Snapshot / PortfolioState / StrategyState / DecisionConfig
             + immutable Decision (fingerprint equality for parity tests)
  direction  faithful extraction of the determine_option_strategy tally
  kernel     the decide() orchestrator (direction -> head -> EM -> contract ->
             EV/PoP -> conviction -> vetoes -> size)
"""

from oracle.decision import schema  # noqa: F401
from oracle.decision.schema import (  # noqa: F401
    Decision,
    DecisionConfig,
    PortfolioState,
    Snapshot,
    StrategyState,
    make_no_trade,
)

__all__ = [
    "schema",
    "Snapshot",
    "PortfolioState",
    "StrategyState",
    "DecisionConfig",
    "Decision",
    "make_no_trade",
    "decide",
    "compute_direction",
]


def __getattr__(name):
    # Lazy re-export so importing the package does not eagerly pull in the
    # optional Oracle head / EV modules the kernel touches (keeps import cheap
    # and fail-open). ``from oracle.decision import decide`` still works.
    if name == "decide":
        from oracle.decision.kernel import decide as _decide
        return _decide
    if name == "compute_direction":
        from oracle.decision.direction import compute_direction as _cd
        return _cd
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
