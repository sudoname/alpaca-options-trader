"""
Oracle Execution — swappable ExecutionClient adapters.

The decision kernel (``oracle.decision.decide``) produces a ``Decision`` but
NEVER submits an order. Execution is a separate, narrow concern handled here so
that live, paper, backtest and shadow all run the SAME decision code and differ
only by which ExecutionClient they are handed:

    live      -> AlpacaExecutionClient(paper=False)
    paper     -> AlpacaExecutionClient(paper=True)     # same code, paper base_url
    backtest  -> SimExecutionClient(...)               # deterministic fills
    shadow    -> ShadowExecutionClient(inner=...)       # reads real, writes nothing

Interface (``client.ExecutionClient``): get_quote, get_positions,
get_buying_power, submit_order, cancel_order, replace_order, get_order,
get_fills. Value objects (Quote / OrderRequest / OrderResult / Fill / Position)
are frozen dataclasses so a sim run is reproducible and comparable to live.

Design rules (mirror the rest of Oracle):
  * Adapters WRAP existing infrastructure; they do not duplicate strategy logic.
  * SimExecutionClient is pure/deterministic (no network, no creds).
  * The Alpaca adapter takes an injectable transport so it is unit-testable
    offline; with no transport it uses ``requests`` against the same endpoints
    the live ``smart_trader`` already uses.
  * Fail-open reads (None/empty on error); writes surface a rejected result
    rather than raising.
"""

from oracle.execution import client  # noqa: F401
from oracle.execution.client import (  # noqa: F401
    ExecutionClient,
    Fill,
    OrderRequest,
    OrderResult,
    Position,
    Quote,
    ShadowExecutionClient,
    SimExecutionClient,
)

__all__ = [
    "client",
    "ExecutionClient",
    "Quote",
    "OrderRequest",
    "OrderResult",
    "Fill",
    "Position",
    "SimExecutionClient",
    "ShadowExecutionClient",
    "AlpacaExecutionClient",
]


def __getattr__(name):
    # Lazy re-export so importing the package never pulls in the alpaca-py SDK
    # (the Alpaca adapter imports the SDK lazily too, keeping import creds-free).
    if name == "AlpacaExecutionClient":
        from oracle.execution.alpaca import AlpacaExecutionClient as _AEC
        return _AEC
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
