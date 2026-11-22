"""Kalshi prediction-market research integration (Oracle 4.0.2 P1-6).

Research-path only: nothing in the production trading path imports this
package.  All broker/exchange access is read-only public market data.

Modules:
  * ``client``          -- read-only public Kalshi REST client;
  * ``market_mapping``  -- Kalshi series -> Oracle underlying mapping;
  * ``collector``       -- point-in-time snapshot collector (sqlite);
  * ``feature_builder`` -- PIT Kalshi-derived features for an underlying;
  * ``validation``      -- matched ablation experiments A-E.

Hard rule: settlement results are NEVER inputs.  The snapshot store has
no settlement/result column at all, so leakage is structurally impossible
downstream of collection.
"""
