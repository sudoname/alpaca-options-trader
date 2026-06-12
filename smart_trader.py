"""
Advanced Options Trading System with ML and Position Management
"""

import os
import sys
import json
import requests
from datetime import datetime, timedelta
import argparse
from typing import Dict, List, Optional, Tuple
import pickle
import math

class SmartOptionsTrader:
    def __init__(self, ticker: str = None, quantity: int = 1):
        self.load_credentials()
        self.base_url = "https://paper-api.alpaca.markets" if self.paper else "https://api.alpaca.markets"
        self.data_url = "https://data.alpaca.markets"
        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key
        }
        self.ticker = ticker
        self.quantity = quantity

        # Dynamic stop loss and take profit parameters are now loaded from .env in load_credentials()

        self.load_trading_history()
        self.load_ml_model()

        # RL advisory layer (shadow mode: observes & learns, never overrides)
        self.rl_advisor = None
        try:
            from rl_wrapper import RLAdvisor, rl_enabled
            if rl_enabled():
                self.rl_advisor = RLAdvisor(strat_name='smart_trader')
                print("[RL] Advisor active (shadow mode)")
        except Exception as e:
            print(f"[RL] Advisor unavailable: {e}")

        # Shadow recorder (additive, fail-open, behind SHADOW_RECORDER_ENABLED).
        # Records every decision+outcome under ONE decision_id with NET-of-cost
        # P/L; never changes what is traded. This is the fix for the dead RL
        # loop (gross P/L logged under mismatched keys that never fired).
        self.shadow_recorder = None
        self._episode_store = None
        try:
            from rl_wrapper import _env as _rl_env
            if _rl_env('SHADOW_RECORDER_ENABLED', 'false').lower() in ('1', 'true', 'yes', 'on'):
                from shadow_recorder import ShadowRecorder
                from episode_store import EpisodeStore
                from cost_model import CostModel, load_cost_config_from_env
                self._episode_store = EpisodeStore('episodes.db')
                self.shadow_recorder = ShadowRecorder(
                    self._episode_store,
                    CostModel(load_cost_config_from_env()),
                    advisor=self.rl_advisor,
                    strat_name='smart_trader',
                )
                print("[SHADOW] Recorder active -> episodes.db")
        except Exception as e:
            print(f"[SHADOW] Recorder unavailable: {e}")

        # Risk engine: hard caps that live OUTSIDE the learner (per-trade budget,
        # daily loss, max concurrent, PDT, kill-switch). It is the last line of
        # capital protection and runs BEFORE every order is sent. Fail-closed: if
        # it can't be built, _risk_check blocks. Always on; not feature-flagged.
        self.risk_engine = None
        try:
            from risk_engine import RiskEngine, load_risk_limits_from_env
            self.risk_engine = RiskEngine(load_risk_limits_from_env())
            print("[RISK] Engine active (fail-closed hard caps)")
        except Exception as e:
            print(f"[RISK] Engine unavailable (trades will be blocked): {e}")

        # Sentiment (Fear & Greed) risk filter. Fail-open. Uses an Alpaca-backed
        # market data provider so the custom score (PRIMARY) can be computed here
        # too; falls back to CNN if Alpaca data is unavailable. Used to scale
        # position size in place_order_with_stops; never blocks/crashes a trade.
        self.sentiment_service = None
        try:
            from sentiment import SentimentService, AlpacaMarketDataProvider, SentimentConfig
            if SentimentConfig.from_env().enabled:
                feed = os.getenv('SENTIMENT_ALPACA_FEED', 'iex')
                provider = AlpacaMarketDataProvider(self.data_url, self.headers, feed=feed)
                self.sentiment_service = SentimentService(provider)
                print("[SENTIMENT] Fear & Greed filter active")
        except Exception as e:
            print(f"[SENTIMENT] Filter unavailable: {e}")

        # Per-ticker News signal. Fail-open. Fetches recent headlines from
        # Alpaca's news endpoint and scores them; used to (1) tilt call/put
        # direction, (2) nudge option ranking, and (3) gate/size entries.
        # Never blocks or crashes a trade on failure.
        # Human-readable cause of the most recent blocked/failed entry, set by
        # place_order_with_stops so callers (Telegram) can report the real reason.
        self.last_block_reason = None

        self.news_service = None
        try:
            from news import NewsService, NewsConfig
            if NewsConfig.from_env().enabled:
                self.news_service = NewsService(
                    NewsConfig.from_env(), self.data_url, self.headers
                )
                print("[NEWS] Headline signal active")
        except Exception as e:
            print(f"[NEWS] signal unavailable: {e}")

    def load_credentials(self):
        """Load API credentials and trading parameters.

        Config resolves shell env first, then ``.env``, then code defaults via the
        shared ``config_loader.ConfigLoader`` (Phase 4.5). ``env_vars`` keeps a
        dict-compatible ``.get(name, default)`` so every downstream lookup below is
        unchanged in form but now honors a shell ``KEY=... python ...`` override.
        """
        from config_loader import ConfigLoader
        env_vars = ConfigLoader()

        self.api_key = env_vars.get('ALPACA_API_KEY', '')
        self.secret_key = env_vars.get('ALPACA_SECRET_KEY', '')
        self.paper = env_vars.get('ALPACA_PAPER', 'true').lower() == 'true'

        # Budget per trade (dollars) from .env with default
        self.max_budget_per_trade = float(env_vars.get('MAX_BUDGET_PER_TRADE', '500'))

        # Liquidity gate: skip contracts whose bid/ask spread (as a % of ask)
        # exceeds this. Tighter = only liquid contracts pass.
        self.max_spread_pct = float(env_vars.get('MAX_SPREAD_PCT', '15'))

        # Confidence-based position sizing. Quantity is keyed off the directional
        # signal strength from determine_option_strategy (winning side's signal
        # count): strength >= very_high -> 3 contracts, >= high -> 2, else 1.
        self.conf_high_signals = int(env_vars.get('CONF_HIGH_SIGNALS', '2'))
        self.conf_very_high_signals = int(env_vars.get('CONF_VERY_HIGH_SIGNALS', '4'))
        # Updated by determine_option_strategy; read when sizing a confidence-based order.
        self.last_signal_strength = 0

        # Real option Greeks/IV from Alpaca snapshots. OFF by default so scoring
        # behavior is unchanged until an operator opts in via USE_REAL_GREEKS.
        # When on, select_best_option overrides the hardcoded Greek/IV fallbacks
        # with live snapshot values whenever they're present, and falls back to
        # the heuristic values for any field the snapshot is missing.
        self.use_real_greeks = str(
            env_vars.get('USE_REAL_GREEKS', 'false')
        ).strip().lower() in ('1', 'true', 'yes', 'on')

        # --- Phase 2: contract-selection refinements ----------------------- #
        # All sub-gates below are OFF by default; with their flags unset the
        # selection behaves exactly as before. Each can reject a contract (with
        # a logged reason) and/or nudge its score toward a preferred target.
        def _flag(name):
            return str(env_vars.get(name, 'false')).strip().lower() in (
                '1', 'true', 'yes', 'on')

        def _f2(name, default):
            try:
                return float(env_vars.get(name, str(default)))
            except (TypeError, ValueError):
                return default

        def _i2(name, default):
            try:
                return int(float(env_vars.get(name, str(default))))
            except (TypeError, ValueError):
                return default

        # 1) DTE targeting: prefer contracts near OPTION_TARGET_DTE, gate to the
        #    [OPTION_MIN_DTE, OPTION_MAX_DTE] window. Gated by USE_DTE_TARGETING.
        self.option_min_dte = _i2('OPTION_MIN_DTE', 30)
        self.option_max_dte = _i2('OPTION_MAX_DTE', 90)
        self.option_target_dte = _i2('OPTION_TARGET_DTE', 45)
        self.use_dte_targeting = _flag('USE_DTE_TARGETING')

        # 2) Delta targeting (needs a real delta; otherwise falls back to the
        #    existing behavior). Gated by USE_DELTA_TARGETING.
        self.option_target_call_delta = _f2('OPTION_TARGET_CALL_DELTA', 0.40)
        self.option_target_put_delta = _f2('OPTION_TARGET_PUT_DELTA', -0.40)
        self.option_max_delta_distance = _f2('OPTION_MAX_DELTA_DISTANCE', 0.20)
        self.use_delta_targeting = _flag('USE_DELTA_TARGETING')

        # 3) Cost/EV gate: reject wide spreads and negative post-cost edge using
        #    cost_model.py. Gated by USE_COST_EV_GATE. MAX_OPTION_SPREAD_PCT is a
        #    FRACTION (0.15 = 15%); MIN_POST_COST_EDGE is in expectancy %-points.
        self.use_cost_ev_gate = _flag('USE_COST_EV_GATE')
        self.min_post_cost_edge = _f2('MIN_POST_COST_EDGE', 0.00)
        self.max_option_spread_pct = _f2('MAX_OPTION_SPREAD_PCT', 0.15)

        # 4) Option liquidity filter. Gated by USE_OPTION_LIQUIDITY_FILTER. When
        #    volume/OI are missing it FAILS OPEN unless REQUIRE_OPTION_LIQUIDITY_DATA.
        self.min_option_volume = _f2('MIN_OPTION_VOLUME', 0)
        self.min_option_open_interest = _f2('MIN_OPTION_OPEN_INTEREST', 0)
        self.use_option_liquidity_filter = _flag('USE_OPTION_LIQUIDITY_FILTER')
        self.require_option_liquidity_data = _flag('REQUIRE_OPTION_LIQUIDITY_DATA')

        # Lazily-built cost model for the EV gate (fail-open if unavailable).
        self._phase2_cm = None

        # --- Phase 3: direction quality + sizing safety -------------------- #
        # All OFF by default so default behavior is byte-for-byte unchanged.
        #  * USE_SKIP_ON_WEAK_SIGNAL: when on, determine_option_strategy may
        #    return 'skip' (NO_TRADE) instead of a low-conviction default CALL,
        #    either because the bull/bear tallies are tied (no edge) or because
        #    fewer than MIN_DIRECTION_SIGNALS total signals fired.
        #  * USE_NORMALIZED_CONFIDENCE: when on, the conviction used for sizing
        #    folds in signal margin + total + agreement so a pile of weak or
        #    duplicated signals can't inflate size; a collapsed (0) confidence
        #    sizes to 0 contracts (skip). It never *increases* size vs. before.
        self.use_skip_on_weak_signal = _flag('USE_SKIP_ON_WEAK_SIGNAL')
        self.min_direction_signals = _i2('MIN_DIRECTION_SIGNALS', 2)
        self.use_normalized_confidence = _flag('USE_NORMALIZED_CONFIDENCE')
        # Set by determine_option_strategy when it returns 'skip'; surfaced by
        # the scheduler and Telegram so the operator sees *why* nothing traded.
        self.last_skip_reason = None

        # --- Phase 4: portfolio-level options risk controls ---------------- #
        # All OFF by default so default behavior is byte-for-byte unchanged.
        #  * USE_PORTFOLIO_GREEK_LIMITS: when on, an aggregate-exposure gate runs
        #    before the per-trade risk engine and blocks an entry that would push
        #    the book past a delta/vega/theta/same-direction/per-underlying cap.
        #    Greeks come from real Alpaca snapshots when available, else from the
        #    same heuristic fallbacks select_best_option uses. Limits live in
        #    portfolio_risk.PortfolioLimits (loaded from .env).
        #  * USE_REALIZED_PNL_KILLSWITCH: when on, the kill-switch / daily-loss
        #    inputs use *realized* dollar P/L for today (RealizedPnLTracker) instead
        #    of equity-minus-last_equity, so a transient unrealized drawdown on open
        #    options can no longer trip the switch. Resets daily by construction.
        self.portfolio_limits = None
        try:
            from portfolio_risk import load_portfolio_limits_from_env
            self.portfolio_limits = load_portfolio_limits_from_env()
            if self.portfolio_limits.enabled:
                print("[PORTFOLIO] Aggregate greek limits active")
        except Exception as e:
            print(f"[PORTFOLIO] limits unavailable: {e}")
        self.use_portfolio_greek_limits = bool(
            self.portfolio_limits and self.portfolio_limits.enabled)
        self.use_realized_pnl_killswitch = _flag('USE_REALIZED_PNL_KILLSWITCH')

        # Load profit/loss thresholds from .env with defaults
        self.base_stop_loss = float(env_vars.get('BASE_STOP_LOSS', '0.10'))
        self.base_take_profit = float(env_vars.get('BASE_TAKE_PROFIT', '0.20'))
        self.max_stop_loss = float(env_vars.get('MAX_STOP_LOSS', '0.25'))
        self.max_take_profit = float(env_vars.get('MAX_TAKE_PROFIT', '0.50'))
        self.trailing_stop_distance = float(env_vars.get('TRAILING_STOP_DISTANCE', '0.05'))
        # Trailing stop arms only after this much profit (fraction of entry).
        # 0 = legacy: any tick above entry arms it, which converts long-barrier
        # trades into ~-trailing_distance scratches on the first wobble.
        self.trailing_arm_profit_pct = float(
            env_vars.get('TRAILING_ARM_PROFIT_PCT', '0'))
        # Signal-based exits (momentum reversal / vol spike / regime change /
        # profit giveback / pullback-from-high) in should_exit_dynamically.
        # Disable (=0) to let positions run their stop/target race.
        self.signal_exits_enabled = str(
            env_vars.get('SIGNAL_EXITS_ENABLED', 'true')
        ).strip().lower() in ('1', 'true', 'yes', 'on')

        # Operational safety controls (duplicate guard / stale-quote / fill readback).
        self.quote_max_age_sec = float(env_vars.get('QUOTE_MAX_AGE_SEC', '30'))
        self.fill_wait_sec = float(env_vars.get('FILL_WAIT_SEC', '5'))
        self.fill_poll_sec = float(env_vars.get('FILL_POLL_SEC', '0.5'))

        # Roll-on-profit: when a winner reaches ROLL_TRIGGER_PCT (default
        # MAX_TAKE_PROFIT), close it and re-enter a cheaper, further-OTM contract
        # on the same underlying/direction to lock gains while staying in the
        # trade. Disabled by default so existing behavior is unchanged.
        self.roll_enabled = str(
            env_vars.get('ROLL_ENABLED', 'false')
        ).strip().lower() in ('1', 'true', 'yes', 'on')
        # Trigger as a P/L percentage (e.g. 500 = +500%). Defaults to the
        # max_take_profit clamp expressed in percent.
        self.roll_trigger_pct = float(
            env_vars.get('ROLL_TRIGGER_PCT', self.max_take_profit * 100)
        )
        # How far out-of-the-money the re-entry strike must be (fraction of the
        # underlying price). 0.05 = at least 5% OTM.
        self.roll_otm_pct = float(env_vars.get('ROLL_OTM_PCT', '0.05'))
        # Upper bound on OTM distance so the roll doesn't buy a worthless lotto.
        self.roll_max_otm_pct = float(env_vars.get('ROLL_MAX_OTM_PCT', '0.20'))

        # --- Startup mode banner (Phase 4.5) ------------------------------ #
        # One line summarizing every Oracle mode flag so the operator can see at
        # a glance which optional gates are active for this process. Purely a log
        # line — it reads already-resolved flags and changes no behavior.
        try:
            _mode_flags = {
                'USE_REAL_GREEKS': self.use_real_greeks,
                'USE_DTE_TARGETING': self.use_dte_targeting,
                'USE_DELTA_TARGETING': self.use_delta_targeting,
                'USE_COST_EV_GATE': self.use_cost_ev_gate,
                'USE_OPTION_LIQUIDITY_FILTER': self.use_option_liquidity_filter,
                'USE_SKIP_ON_WEAK_SIGNAL': self.use_skip_on_weak_signal,
                'USE_NORMALIZED_CONFIDENCE': self.use_normalized_confidence,
                'USE_PORTFOLIO_GREEK_LIMITS': self.use_portfolio_greek_limits,
                'USE_REALIZED_PNL_KILLSWITCH': self.use_realized_pnl_killswitch,
            }
            print("[ORACLE] mode flags: " + " ".join(
                f"{k}={'on' if v else 'off'}" for k, v in _mode_flags.items()))
        except Exception:
            pass

    def load_trading_history(self):
        """Load historical trades for learning"""
        self.history_file = 'trading_history.json'
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                self.trading_history = json.load(f)
        else:
            self.trading_history = {
                'trades': [],
                'performance_metrics': {},
                'learned_patterns': {}
            }

    def save_trading_history(self):
        """Save trading history for future learning"""
        with open(self.history_file, 'w') as f:
            json.dump(self.trading_history, f, indent=2, default=str)

    def load_ml_model(self):
        """Load or initialize ML model for trade optimization"""
        self.model_file = 'trade_optimizer.pkl'
        if os.path.exists(self.model_file):
            with open(self.model_file, 'rb') as f:
                self.ml_model = pickle.load(f)
        else:
            # Initialize simple scoring model
            self.ml_model = {
                'weights': {
                    'delta': 0.30,
                    'gamma': 0.10,
                    'theta': 0.15,
                    'vega': 0.10,
                    'iv': 0.15,
                    'moneyness': 0.20
                },
                'success_patterns': [],
                'failure_patterns': []
            }

    def save_ml_model(self):
        """Save ML model after updates"""
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.ml_model, f)

    def get_account(self):
        """Get account information"""
        response = requests.get(f"{self.base_url}/v2/account", headers=self.headers)
        return response.json() if response.status_code == 200 else None

    def get_positions(self):
        """Get current positions"""
        response = requests.get(f"{self.base_url}/v2/positions", headers=self.headers)
        return response.json() if response.status_code == 200 else []

    def get_orders(self):
        """Get current orders"""
        response = requests.get(f"{self.base_url}/v2/orders", headers=self.headers)
        return response.json() if response.status_code == 200 else []

    def _humanize_risk_breaches(self, risk_verdict: Dict) -> str:
        """Map raw risk-engine breach codes to a human-readable explanation."""
        try:
            max_conc = self.risk_engine.limits.max_concurrent
        except Exception:
            max_conc = "?"
        labels = {
            "max_concurrent": (f"the maximum number of concurrent open positions "
                               f"is already reached (limit {max_conc})"),
            "over_budget": "the trade cost exceeds the per-trade budget",
            "daily_loss_limit": "today's realized loss limit has been hit",
            "kill_switch": "the daily loss kill-switch is tripped",
            "pdt_block": "pattern-day-trader headroom is exhausted",
            "pdt_unknown": "pattern-day-trader headroom could not be verified",
            "nonpositive_cost": "the computed trade cost was not positive",
            "risk_engine_unavailable": "the risk engine is unavailable",
            "exception": "the risk engine hit an unexpected error",
        }
        breaches = risk_verdict.get("breaches") or []
        parts = [labels.get(b, b) for b in breaches]
        return "; ".join(parts) if parts else (risk_verdict.get("reason") or "blocked")

    @staticmethod
    def _occ_underlying(symbol: str) -> str:
        """Extract the underlying root from an OCC option symbol.

        OCC format is ROOT + YYMMDD + C/P + 8-digit strike, i.e. the trailing 15
        characters encode date+type+strike. Returns the uppercased root, or the
        uppercased symbol unchanged if it doesn't look like an OCC option.
        """
        s = (symbol or "").upper()
        if len(s) > 15 and s[-15:-8] and s[-9] in ("C", "P") and s[-8:].isdigit():
            return s[:-15]
        return s

    @staticmethod
    def _occ_parse(symbol: str):
        """Parse an OCC option symbol -> (underlying, 'call'/'put', strike).

        Returns (underlying, None, None) when the symbol isn't a parseable OCC
        option. Pure; no network. Strike is the 8-digit field / 1000.
        """
        s = (symbol or "").upper()
        if len(s) > 15 and s[-15:-8] and s[-9] in ("C", "P") and s[-8:].isdigit():
            underlying = s[:-15]
            opt_type = "call" if s[-9] == "C" else "put"
            try:
                strike = int(s[-8:]) / 1000.0
            except ValueError:
                strike = None
            return underlying, opt_type, strike
        return s, None, None

    def _fallback_greeks(self, opt_type, strike, underlying_price):
        """Heuristic per-contract greeks when a real snapshot is unavailable.

        Mirrors the delta heuristic and static vega/theta fallbacks used by
        select_best_option so the portfolio aggregate degrades gracefully
        (Phase 4 requirement 3). Returns magnitudes; sign handling lives in
        portfolio_risk. Never raises.
        """
        delta = 0.5
        try:
            if strike and underlying_price:
                if opt_type == "call":
                    delta = min(0.95, max(0.05,
                        (underlying_price - strike) / underlying_price * 0.7 + 0.5))
                else:
                    delta = abs(min(-0.05, max(-0.95,
                        (strike - underlying_price) / underlying_price * 0.7 - 0.5)))
        except Exception:
            delta = 0.5
        return {"delta": abs(delta), "vega": 0.10, "theta": 0.05}

    def _position_greeks(self, positions):
        """Build portfolio_risk-shaped position dicts from live Alpaca positions.

        For each option position: parse the OCC symbol for underlying/type/strike,
        read |qty|, and attach per-contract greeks — REAL snapshot greeks when
        available (delta/vega/theta), otherwise heuristic fallbacks. Fail-open:
        any per-position error is skipped rather than aborting the whole gate.
        Returns a list of {underlying, direction, qty, delta, vega, theta}.
        """
        out = []
        for p in positions or []:
            try:
                sym = p.get("symbol", "")
                underlying, opt_type, strike = self._occ_parse(sym)
                if opt_type is None:
                    continue  # not an option position
                try:
                    qty = abs(int(float(p.get("qty", 0) or 0)))
                except (TypeError, ValueError):
                    qty = 0
                if qty <= 0:
                    continue

                greeks = {}
                if not p.get("mock", False):
                    try:
                        greeks = self.get_option_snapshot(sym) or {}
                    except Exception:
                        greeks = {}

                have = (isinstance(greeks.get("delta"), (int, float)) and
                        isinstance(greeks.get("vega"), (int, float)) and
                        isinstance(greeks.get("theta"), (int, float)))
                if have:
                    g = {"delta": greeks["delta"], "vega": greeks["vega"],
                         "theta": greeks["theta"]}
                else:
                    price = None
                    try:
                        price = self.get_current_price(underlying)
                    except Exception:
                        price = None
                    g = self._fallback_greeks(opt_type, strike, price)

                out.append({
                    "underlying": underlying,
                    "direction": opt_type,
                    "qty": qty,
                    "delta": g["delta"],
                    "vega": g["vega"],
                    "theta": g["theta"],
                })
            except Exception:
                continue
        return out

    def _portfolio_greek_check(self, option, order_quantity):
        """Phase 4 aggregate-exposure gate (opt-in via USE_PORTFOLIO_GREEK_LIMITS).

        Projects this candidate onto the current book and tests the portfolio
        delta/vega/theta and same-direction/per-underlying caps. Returns a
        portfolio_risk verdict dict ({allowed, reason, breaches, current_*,
        projected_*}). Logs the requested current/projected greeks plus the
        decision.

        Fail policy: a real cap *breach* blocks (allowed=False). Any computation
        error (missing greeks, snapshot/network hiccup) FAILS OPEN — consistent
        with greeks being advisory/fallback data — so the authoritative
        capital-protection block remains the fail-closed risk engine that runs
        immediately after. A no-op (allowed=True) is returned when disabled.
        """
        if not getattr(self, "use_portfolio_greek_limits", False) or not self.portfolio_limits:
            return {"allowed": True, "reason": "disabled", "breaches": []}
        try:
            from portfolio_risk import check_portfolio_limits, summarize_for_log
            positions = self.get_positions()
            current = self._position_greeks(positions if isinstance(positions, list) else [])
            new_trade = {
                "underlying": self._occ_underlying(option.get("symbol", ""))
                              or (self.ticker or "").upper(),
                "direction": option.get("type", "call"),
                "qty": int(order_quantity or 1),
                "delta": option.get("delta", 0.0),
                "vega": option.get("vega", 0.0),
                "theta": option.get("theta", 0.0),
            }
            verdict = check_portfolio_limits(current, new_trade, self.portfolio_limits)
            print(f"[PORTFOLIO] {summarize_for_log(verdict)}")
            print(f"[PORTFOLIO] same_direction={verdict['same_direction']}->"
                  f"{verdict['projected_same_direction']} "
                  f"per_underlying={verdict['per_underlying']}->"
                  f"{verdict['projected_per_underlying']}")
            if verdict["allowed"]:
                print("[PORTFOLIO] decision: OK")
            else:
                print(f"[PORTFOLIO] decision: BLOCKED reason={verdict['reason']}")
            return verdict
        except Exception as e:
            # Fail-open on computation error (the fail-closed risk engine still runs).
            print(f"[PORTFOLIO] check error (ignored, failing open): {e}")
            return {"allowed": True, "reason": f"error:{type(e).__name__}", "breaches": []}

    def _risk_check(self, trade_cost, qty: int = 1, may_day_trade: bool = True):
        """Run the fail-closed risk engine before an order is placed.

        Gathers live inputs:
          * open_positions     = len(get_positions())
          * realized_pnl_today = account equity - last_equity (today's P/L)
          * pdt_remaining      = PDTTracker day-trade headroom
        Fail-closed: if the engine is missing or any required input can't be
        fetched, the trade is BLOCKED (RiskEngine.check already returns
        allowed=False on a None input). Entries are treated as potential day
        trades (may_day_trade=True) so PDT headroom is enforced conservatively.
        Returns the engine verdict dict {allowed, reason, breaches}.
        """
        if self.risk_engine is None:
            return {"allowed": False, "reason": "risk_engine_unavailable",
                    "breaches": ["risk_engine_unavailable"]}
        try:
            positions = self.get_positions()
            open_positions = len(positions) if isinstance(positions, list) else None

            # Per-underlying concentration cap input (opt-in via
            # MAX_POSITIONS_PER_UNDERLYING). Count open option positions whose
            # OCC symbol resolves to the current underlying. Fail-open: if this
            # can't be computed, pass None so the cap is simply skipped.
            positions_for_underlying = None
            try:
                if isinstance(positions, list) and self.ticker:
                    positions_for_underlying = sum(
                        1 for p in positions
                        if self._occ_underlying(p.get('symbol', '')) == self.ticker.upper()
                    )
            except Exception:
                positions_for_underlying = None

            # Today's P/L for the kill-switch / daily-loss gate.
            #  * Default (USE_REALIZED_PNL_KILLSWITCH off): legacy behavior =
            #    account equity - last_equity. This includes the *unrealized*
            #    mark of every open option, so a transient intraday drawdown on
            #    open contracts can trip the switch.
            #  * Phase 4 fix (flag on): use *realized* dollar P/L for today only
            #    (RealizedPnLTracker, resets daily). Unrealized option marks no
            #    longer trip the switch; closed-trade losses still do.
            realized_pnl_today = None
            if getattr(self, 'use_realized_pnl_killswitch', False):
                try:
                    from realized_pnl_tracker import RealizedPnLTracker
                    realized_pnl_today = RealizedPnLTracker().get_today_realized()
                except Exception:
                    realized_pnl_today = None
            else:
                acct = self.get_account() or {}
                equity = acct.get('equity')
                last_equity = acct.get('last_equity')
                if equity is not None and last_equity is not None:
                    realized_pnl_today = float(equity) - float(last_equity)

            pdt_remaining = None
            try:
                from pdt_tracker import PDTTracker
                pdt_remaining = PDTTracker().get_remaining_day_trades()
            except Exception:
                pdt_remaining = None

            return self.risk_engine.check(
                trade_cost=trade_cost,
                realized_pnl_today=realized_pnl_today,
                open_positions=open_positions,
                pdt_remaining=pdt_remaining,
                may_day_trade=may_day_trade,
                positions_for_underlying=positions_for_underlying,
            )
        except Exception as e:
            return {"allowed": False,
                    "reason": f"risk_input_error:{type(e).__name__}",
                    "breaches": ["risk_input_error"]}

    def _record_blocked_decision(self, option, underlying_symbol, dynamic_levels,
                                 quote, qty, risk_verdict):
        """Record a risk-blocked entry as a SKIP decision (verdict in risk_json).

        Observational only; never raises into the trading path. No-op unless the
        shadow recorder is active. This ensures blocked trades are not silently
        dropped from the episode record.
        """
        if not getattr(self, 'shadow_recorder', None):
            return
        try:
            analysis_ctx = {
                'direction': (option.get('type') or 'call').upper(),
                'momentum': dynamic_levels.get('momentum', 0),
                'confidence': option.get('score', 0),
                'should_trade': False,
            }
            self.shadow_recorder.on_decision(
                symbol=option['symbol'],
                underlying=underlying_symbol,
                analysis=analysis_ctx,
                quote=quote,
                entry_premium=(quote or {}).get('ask'),
                qty=qty,
                mode='live-paper-blocked',
                as_of=datetime.now().isoformat(),
                day_of_week=datetime.now().weekday(),
                risk=risk_verdict,
            )
            print("[RISK] Blocked decision recorded to episodes.db")
        except Exception as e:
            print(f"[RISK] block-record failed: {e}")

    # ----------------------------------------------------------------------- #
    # Operational safety: duplicate guard / stale quote / fill readback /
    # restart reconciliation. All additive; the trading path calls these.
    # ----------------------------------------------------------------------- #
    @staticmethod
    def _parse_alpaca_ts(ts_str):
        """Parse an Alpaca RFC3339 timestamp to an aware UTC datetime.

        Handles a trailing 'Z' and nanosecond precision (truncated to micros).
        Returns None if it cannot be parsed.
        """
        if not ts_str:
            return None
        try:
            from datetime import timezone
            s = str(ts_str).strip()
            if s.endswith('Z'):
                s = s[:-1] + '+00:00'
            if '.' in s:
                head, frac = s.split('.', 1)
                tz = ''
                for sign in ('+', '-'):
                    idx = frac.find(sign)
                    if idx != -1:
                        tz, frac = frac[idx:], frac[:idx]
                        break
                s = f"{head}.{frac[:6]}{tz}"
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None

    def _quote_is_fresh(self, quote):
        """True if the quote timestamp is within quote_max_age_sec of now.

        Fail-closed: a missing/unparseable timestamp is treated as STALE so a
        trade is never placed on an unverifiable quote.
        """
        from datetime import timezone
        ts = self._parse_alpaca_ts((quote or {}).get('ts'))
        if ts is None:
            return False
        age = (datetime.now(timezone.utc) - ts).total_seconds()
        return abs(age) <= self.quote_max_age_sec

    def _has_open_or_pending(self, option_symbol):
        """Duplicate-order guard.

        True if there is already a live position, a pending/open order, or a
        tracked active trade for this exact option symbol. Fail-closed on the
        broker checks: if positions or orders cannot be fetched, return True
        (treat as duplicate) so an unguarded re-entry can never slip through.
        A corrupt local tracker is ignored (broker state is authoritative).
        """
        try:
            positions = self.get_positions()
            if not isinstance(positions, list):
                return True
            for p in positions:
                if p.get('symbol') == option_symbol and float(p.get('qty', 0) or 0) != 0:
                    return True
        except Exception:
            return True

        try:
            open_states = {'new', 'accepted', 'pending_new', 'partially_filled',
                           'accepted_for_bidding', 'held', 'pending_replace'}
            for o in (self.get_orders() or []):
                if o.get('symbol') == option_symbol and o.get('status') in open_states:
                    return True
        except Exception:
            return True

        try:
            if os.path.exists('active_trades.json'):
                with open('active_trades.json', 'r') as f:
                    for t in json.load(f):
                        if t.get('symbol') == option_symbol:
                            return True
        except Exception:
            pass

        return False

    def _await_fill(self, order_id):
        """Poll an order until it reaches a terminal state or the wait expires.

        Returns the latest order dict (or None). Lets the caller read the REAL
        filled_avg_price / filled_qty instead of assuming the quote ask, and
        detect rejected / canceled / partial fills.
        """
        import time
        terminal = {'filled', 'canceled', 'rejected', 'expired', 'done_for_day'}
        latest = None
        deadline = time.time() + self.fill_wait_sec
        while True:
            try:
                resp = requests.get(f"{self.base_url}/v2/orders/{order_id}",
                                    headers=self.headers)
                if resp.status_code == 200:
                    latest = resp.json()
                    if latest.get('status') in terminal:
                        return latest
            except Exception:
                pass
            if time.time() >= deadline:
                return latest
            time.sleep(self.fill_poll_sec)

    def reconcile_open_trades(self):
        """Re-sync tracked active trades against live Alpaca positions on startup.

        * Tracked trade with NO live position -> it closed while the bot was
          down: record the outcome (net via the close path) and drop it.
        * Live position with NO tracked trade -> surfaced as untracked (external
          / manual entry); left in place, only logged.
        Rewrites active_trades.json with the still-open tracked trades. Skips
        safely (no destructive change) if positions cannot be fetched.
        """
        active_file = 'active_trades.json'
        if not os.path.exists(active_file):
            return
        try:
            with open(active_file, 'r') as f:
                active_trades = json.load(f)
        except Exception as e:
            print(f"[RECONCILE] Could not read {active_file}: {e}")
            return

        try:
            positions = self.get_positions()
        except Exception as e:
            print(f"[RECONCILE] Could not fetch positions (skipping): {e}")
            return
        if not isinstance(positions, list):
            print("[RECONCILE] Positions unavailable (skipping).")
            return

        pos_by_symbol = {p.get('symbol'): p for p in positions}
        still_open = []
        closed = 0
        for trade in active_trades:
            sym = trade.get('symbol')
            if sym in pos_by_symbol:
                still_open.append(trade)
                continue
            closed_pnl = 0
            try:
                last = self.get_option_price(sym)
                entry_price = trade.get('entry_price')
                if last and entry_price:
                    exit_px = last.get('mid') or last.get('ask') or last.get('bid') or 0
                    if exit_px and entry_price:
                        closed_pnl = ((exit_px - entry_price) / entry_price) * 100
            except Exception:
                closed_pnl = 0
            print(f"[RECONCILE] {sym} no longer held; recording close "
                  f"(~{closed_pnl:+.1f}%).")
            try:
                self.record_trade_outcome(trade, 'reconciled_closed', closed_pnl)
            except Exception as e:
                print(f"[RECONCILE] record_trade_outcome failed for {sym}: {e}")
            closed += 1

        tracked = {t.get('symbol') for t in active_trades}
        for sym, p in pos_by_symbol.items():
            if sym not in tracked:
                print(f"[RECONCILE] Untracked live position: {sym} "
                      f"(qty {p.get('qty')}). Not monitored.")

        try:
            with open(active_file, 'w') as f:
                json.dump(still_open, f, indent=2, default=str)
        except Exception as e:
            print(f"[RECONCILE] Could not write {active_file}: {e}")

        print(f"[RECONCILE] {len(still_open)} open / {closed} closed-out reconciled.")

    def get_market_status(self):
        """Check if market is open"""
        response = requests.get(f"{self.base_url}/v2/clock", headers=self.headers)
        if response.status_code == 200:
            return response.json()
        else:
            return {'is_open': False}

    def get_current_price(self, ticker=None):
        """Get current stock price using last trade price"""
        symbol = ticker or self.ticker
        if not symbol:
            return None

        # Use last trade price for most accurate current price
        response = requests.get(
            f"{self.data_url}/v2/stocks/{symbol}/trades/latest",
            headers=self.headers,
            params={'feed': 'iex'}  # Use IEX data for free tier
        )
        if response.status_code == 200:
            data = response.json()
            return float(data['trade']['p'])  # 'p' = price
        return None

    def get_price_history(self, ticker: str = None, days: int = 10) -> List[float]:
        """Get historical prices for volatility calculation"""
        symbol = ticker or self.ticker
        if not symbol:
            return []

        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        response = requests.get(
            f"{self.data_url}/v2/stocks/{symbol}/bars",
            headers=self.headers,
            params={
                'timeframe': '1Day',
                'start': start_time.strftime('%Y-%m-%d'),
                'end': end_time.strftime('%Y-%m-%d'),
                'limit': days + 5,
                'feed': 'iex'  # Use IEX data for free tier
            }
        )

        if response.status_code == 200:
            data = response.json()
            bars = data.get('bars', [])
            # IEX returns bars as a list, not nested by symbol
            return [float(bar['c']) for bar in bars]
        return []

    def calculate_volatility(self, ticker: str = None) -> float:
        """Calculate historical volatility"""
        # ~30 calendar days (~20 trading bars / ~19 returns) for a stable
        # annualized vol estimate. The default 10-day window only gave ~6
        # returns, which is far too noisy.
        prices = self.get_price_history(ticker, days=30)
        if len(prices) < 2:
            return 0.20  # Default 20% volatility

        # Calculate daily returns
        returns = []
        for i in range(1, len(prices)):
            daily_return = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(daily_return)

        if not returns:
            return 0.20

        # Calculate standard deviation of returns (volatility)
        mean_return = sum(returns) / len(returns)
        variance = sum([(r - mean_return) ** 2 for r in returns]) / len(returns)
        volatility = math.sqrt(variance) * math.sqrt(252)  # Annualized volatility

        return min(max(volatility, 0.10), 0.80)  # Cap between 10% and 80%

    def calculate_momentum(self, ticker: str = None) -> float:
        """Calculate momentum indicator"""
        # ~15 calendar days (~10 trading bars) guarantees the 3- and 5-bar
        # momentum/smoothing windows always have data. A 5-day window could
        # drop to ~3 bars after a weekend, making momentum unreliable.
        prices = self.get_price_history(ticker, days=15)
        if len(prices) < 3:
            return 0

        # Calculate momentum as recent price change
        recent_momentum = (prices[-1] - prices[-3]) / prices[-3] if len(prices) >= 3 else 0

        # Smooth momentum with moving average
        if len(prices) >= 5:
            recent_avg = sum(prices[-3:]) / 3
            older_avg = sum(prices[-5:-2]) / 3
            trend_momentum = (recent_avg - older_avg) / older_avg
            momentum = (recent_momentum + trend_momentum) / 2
        else:
            momentum = recent_momentum

        return momentum

    def get_market_regime(self, ticker: str = None) -> str:
        """Determine market regime (trending, ranging, volatile)"""
        volatility = self.calculate_volatility(ticker)
        momentum = abs(self.calculate_momentum(ticker))

        if volatility > 0.30:
            return "volatile"
        elif momentum > 0.05:
            return "trending"
        else:
            return "ranging"

    def calculate_dynamic_levels(self, ticker: str = None, current_price: float = None) -> Dict:
        """Calculate dynamic stop loss and take profit levels"""
        volatility = self.calculate_volatility(ticker)
        momentum = self.calculate_momentum(ticker)
        market_regime = self.get_market_regime(ticker)

        # Base adjustments
        vol_multiplier = 1 + (volatility - 0.20) * 2  # Adjust based on volatility
        momentum_multiplier = 1 + abs(momentum) * 5  # Adjust based on momentum

        # Regime-based adjustments
        regime_adjustments = {
            "volatile": {"stop_multiplier": 1.5, "profit_multiplier": 1.8},
            "trending": {"stop_multiplier": 0.8, "profit_multiplier": 1.5},
            "ranging": {"stop_multiplier": 1.2, "profit_multiplier": 1.0}
        }

        regime_adj = regime_adjustments.get(market_regime, {"stop_multiplier": 1.0, "profit_multiplier": 1.0})

        # Calculate dynamic stop loss
        dynamic_stop_loss = self.base_stop_loss * vol_multiplier * regime_adj["stop_multiplier"]
        dynamic_stop_loss = min(max(dynamic_stop_loss, self.base_stop_loss), self.max_stop_loss)

        # Calculate dynamic take profit
        dynamic_take_profit = self.base_take_profit * momentum_multiplier * regime_adj["profit_multiplier"]
        dynamic_take_profit = min(max(dynamic_take_profit, self.base_take_profit), self.max_take_profit)

        # Trailing stop adjustments
        if momentum > 0.03:  # Strong upward momentum
            trailing_distance = self.trailing_stop_distance * 0.7  # Tighter trailing
        elif momentum < -0.03:  # Strong downward momentum
            trailing_distance = self.trailing_stop_distance * 1.5  # Looser trailing
        else:
            trailing_distance = self.trailing_stop_distance

        return {
            "stop_loss_percent": dynamic_stop_loss,
            "take_profit_percent": dynamic_take_profit,
            "trailing_stop_distance": trailing_distance,
            "volatility": volatility,
            "momentum": momentum,
            "market_regime": market_regime,
            "vol_multiplier": vol_multiplier,
            "momentum_multiplier": momentum_multiplier
        }

    def get_option_price(self, symbol):
        """Get current option price using latest quotes endpoint"""
        try:
            response = requests.get(
                f"{self.data_url}/v1beta1/options/quotes/latest",
                headers=self.headers,
                params={'symbols': symbol, 'feed': 'indicative'}
            )

            if response.status_code == 200:
                data = response.json()
                # Response format: {"quotes": {"SYMBOL": {...}}}
                if 'quotes' in data and symbol in data['quotes']:
                    quote = data['quotes'][symbol]
                    bid = float(quote.get('bp', 0))
                    ask = float(quote.get('ap', 0))

                    if bid > 0 or ask > 0:
                        return {
                            'bid': bid,
                            'ask': ask,
                            'mid': (bid + ask) / 2 if (bid > 0 and ask > 0) else (ask if ask > 0 else bid),
                            'ts': quote.get('t'),
                        }

            print(f"[OPTION PRICE] No quote data for {symbol}, status: {response.status_code}")
            return None
        except Exception as e:
            print(f"[OPTION PRICE ERROR] {symbol}: {e}")
            return None

    def get_option_snapshot(self, symbol):
        """Fetch real Greeks + implied volatility for one contract (fail-open).

        Hits Alpaca's options snapshots endpoint and returns a dict containing
        only the fields actually present, among {delta, gamma, theta, vega, iv}.
        Returns {} when the snapshot, greeks, or IV are unavailable (e.g. on the
        indicative feed, for some 0DTE contracts, or on any error) so callers can
        fall back to heuristic values. Never raises.
        """
        try:
            response = requests.get(
                f"{self.data_url}/v1beta1/options/snapshots",
                headers=self.headers,
                params={'symbols': symbol, 'feed': 'indicative'},
            )
            if response.status_code != 200:
                return {}
            data = response.json() or {}
            snap = (data.get('snapshots') or {}).get(symbol) or {}
            return self._parse_snapshot_greeks(snap)
        except Exception as e:
            print(f"[GREEKS ERROR] {symbol}: {e}")
            return {}

    @staticmethod
    def _parse_snapshot_greeks(snapshot: Dict) -> Dict:
        """Pure parser: extract {delta,gamma,theta,vega,iv} from a snapshot dict.

        Alpaca's snapshot carries a `greeks` object (delta/gamma/theta/vega/rho)
        and a top-level `impliedVolatility`. Only numeric, present values are
        returned; missing/non-numeric fields are omitted so the caller keeps its
        fallback. No network, fully unit-testable.
        """
        out: Dict = {}
        if not isinstance(snapshot, dict):
            return out
        greeks = snapshot.get('greeks') or {}
        if isinstance(greeks, dict):
            for src, dst in (('delta', 'delta'), ('gamma', 'gamma'),
                             ('theta', 'theta'), ('vega', 'vega')):
                val = greeks.get(src)
                if isinstance(val, (int, float)):
                    out[dst] = float(val)
        iv = snapshot.get('impliedVolatility')
        if isinstance(iv, (int, float)):
            out['iv'] = float(iv)
        return out

    @staticmethod
    def _apply_real_greeks(option_data: Dict, snapshot_greeks: Dict) -> str:
        """Override fallback Greeks/IV in-place with real snapshot values.

        For each of delta/gamma/theta/vega/iv, uses the real value when present
        in `snapshot_greeks`, otherwise leaves the existing fallback untouched.
        Returns a human-readable log string tagging each field real/fallback.
        Pure (no network); the heart of the Phase 1 wiring and unit-tested.
        """
        fields = ('delta', 'gamma', 'theta', 'vega', 'iv')
        parts = []
        snapshot_greeks = snapshot_greeks or {}
        for f in fields:
            if f in snapshot_greeks and isinstance(snapshot_greeks[f], (int, float)):
                option_data[f] = float(snapshot_greeks[f])
                src = 'real'
            else:
                src = 'fallback'
            val = option_data.get(f)
            parts.append(f"{f}={val:.4f}({src})" if isinstance(val, (int, float))
                         else f"{f}=None({src})")
        return " ".join(parts)

    def calculate_option_score(self, option: Dict) -> float:
        """Calculate option score using ML-enhanced model"""
        base_score = 0

        # Apply learned weights
        weights = self.ml_model['weights']

        # Delta score (prefer 0.5-0.7 for momentum)
        if 'delta' in option:
            delta_optimal = 0.6
            delta_score = (1 - abs(option['delta'] - delta_optimal)) * weights['delta']
            base_score += delta_score * 100

        # Gamma score (moderate gamma for flexibility)
        if 'gamma' in option:
            gamma_score = min(option['gamma'] * 10, 1) * weights['gamma']
            base_score += gamma_score * 100

        # Theta score (minimize time decay)
        if 'theta' in option:
            theta_penalty = abs(option['theta']) * weights['theta']
            base_score += (1 - min(theta_penalty, 1)) * 100

        # IV score (prefer reasonable IV)
        if 'iv' in option:
            iv_optimal = 0.25
            iv_score = (1 - abs(option['iv'] - iv_optimal)) * weights['iv']
            base_score += iv_score * 100

        # Moneyness score
        if 'moneyness' in option:
            moneyness_score = (1 - abs(option['moneyness'])) * weights['moneyness']
            base_score += moneyness_score * 100

        # Apply learned patterns boost/penalty
        pattern_adjustment = self.apply_learned_patterns(option)
        base_score *= (1 + pattern_adjustment)

        return min(max(base_score, 0), 100)

    def apply_learned_patterns(self, option: Dict) -> float:
        """Apply learned patterns from historical trades"""
        adjustment = 0

        # Check success patterns
        for pattern in self.ml_model.get('success_patterns', []):
            if self.matches_pattern(option, pattern):
                adjustment += 0.1

        # Check failure patterns
        for pattern in self.ml_model.get('failure_patterns', []):
            if self.matches_pattern(option, pattern):
                adjustment -= 0.15

        return adjustment

    def matches_pattern(self, option: Dict, pattern: Dict) -> bool:
        """Check if option matches a learned pattern"""
        threshold = 0.1
        matches = 0
        checks = 0

        for key in ['delta', 'gamma', 'theta', 'iv']:
            if key in option and key in pattern:
                checks += 1
                if abs(option[key] - pattern[key]) <= threshold:
                    matches += 1

        return matches / checks > 0.7 if checks > 0 else False

    def determine_option_strategy(self, ticker: str = None) -> str:
        """Determine whether to trade calls or puts based on market analysis"""
        symbol = ticker or self.ticker

        # Get market conditions
        momentum = self.calculate_momentum(symbol)
        volatility = self.calculate_volatility(symbol)
        market_regime = self.get_market_regime(symbol)

        # Get price trends. ~20 calendar days (~14 trading bars) ensures the
        # 3- and 5-bar trend slices always have data, even right after a
        # weekend; a 10-day window could thin to ~6 bars.
        prices = self.get_price_history(symbol, days=20)
        if len(prices) < 5:
            return 'call'  # Default to calls if insufficient data

        # Calculate short and medium term trends
        short_trend = (prices[-1] - prices[-3]) / prices[-3]  # 3-day trend
        medium_trend = (prices[-1] - prices[-5]) / prices[-5]  # 5-day trend

        # Decision logic
        bearish_signals = 0
        bullish_signals = 0

        # Momentum analysis
        if momentum < -0.03:  # Strong negative momentum
            bearish_signals += 2
        elif momentum < -0.01:  # Moderate negative momentum
            bearish_signals += 1
        elif momentum > 0.03:  # Strong positive momentum
            bullish_signals += 2
        elif momentum > 0.01:  # Moderate positive momentum
            bullish_signals += 1

        # Trend analysis
        if short_trend < -0.02:  # Short-term downtrend
            bearish_signals += 1
        elif short_trend > 0.02:  # Short-term uptrend
            bullish_signals += 1

        if medium_trend < -0.03:  # Medium-term downtrend
            bearish_signals += 1
        elif medium_trend > 0.03:  # Medium-term uptrend
            bullish_signals += 1

        # Volatility consideration (high vol favors direction plays)
        if volatility > 0.4 and bearish_signals > bullish_signals:
            bearish_signals += 1
        elif volatility > 0.4 and bullish_signals > bearish_signals:
            bullish_signals += 1

        # Market regime consideration
        if market_regime == 'volatile' and bearish_signals > 0:
            bearish_signals += 1  # Volatile markets often favor puts

        # News direction tilt (fail-open): add bull/bear votes from recent
        # headlines for this symbol. No effect when news is unavailable/neutral.
        if self.news_service:
            try:
                from news import news_direction_vote, NewsConfig
                news = self.news_service.get_news(symbol)
                bull, bear = news_direction_vote(news, NewsConfig.from_env())
                if bull or bear:
                    bullish_signals += bull
                    bearish_signals += bear
                    print(f"[NEWS] Direction votes for {symbol}: "
                          f"+{bull} bull / +{bear} bear "
                          f"(score {news.get('score')} {news.get('label')})")
            except Exception as e:
                print(f"[NEWS] direction vote error (ignored): {e}")

        # Phase 3 diagnostics (always computed; only acted on behind flags so
        # default behavior is preserved).
        total_signals = bullish_signals + bearish_signals
        signal_margin = abs(bullish_signals - bearish_signals)
        self.last_skip_reason = None

        # Make decision
        if bearish_signals > bullish_signals and bearish_signals >= 2:
            strategy = 'put'
        else:
            strategy = 'call'  # Default to calls unless strong bearish signals

        # Phase 3 (1): weak/flat-signal SKIP. OFF by default. When enabled, refuse
        # to trade rather than default to a low-conviction CALL when either too
        # few directional signals fired or the bull/bear tallies are tied.
        if self.use_skip_on_weak_signal:
            if total_signals < self.min_direction_signals:
                self.last_skip_reason = (
                    f"below_min_signals ({total_signals} < {self.min_direction_signals})")
                strategy = 'skip'
            elif bullish_signals == bearish_signals:
                self.last_skip_reason = (
                    f"flat_signal (bull {bullish_signals} == bear {bearish_signals})")
                strategy = 'skip'

        # Phase 3 (2): conviction/confidence used downstream for sizing. Default
        # (flag off) is the winning side's signal count — identical to before.
        # Normalized mode discounts the margin by the agreement ratio so weak or
        # duplicated signals can't inflate size; a tie/empty tally yields 0.
        if strategy == 'skip':
            confidence = 0
        elif self.use_normalized_confidence:
            confidence = self._normalized_confidence(
                bullish_signals, bearish_signals, strategy)
        else:
            confidence = bearish_signals if strategy == 'put' else bullish_signals
        self.last_signal_strength = confidence

        # Phase 3 (5): structured logging of the decision inputs.
        print(f"[STRATEGY] Analysis for {symbol}:")
        print(f"[STRATEGY] Momentum: {momentum:.3f}, Volatility: {volatility:.1%}")
        print(f"[STRATEGY] Short trend: {short_trend:.2%}, Medium trend: {medium_trend:.2%}")
        print(f"[STRATEGY] bullish_score={bullish_signals} bearish_score={bearish_signals} "
              f"total_signals={total_signals} signal_margin={signal_margin}")
        print(f"[STRATEGY] confidence={confidence} strategy={strategy.upper()}")
        if strategy == 'skip':
            print(f"[STRATEGY] skip_reason={self.last_skip_reason}")

        return strategy

    def _normalized_confidence(self, bullish_signals, bearish_signals, strategy) -> int:
        """Phase 3 normalized confidence (gated by USE_NORMALIZED_CONFIDENCE).

        Returns an integer strength on the SAME scale _confidence_to_quantity
        already consumes, so downstream sizing is unchanged in shape. It folds:
          * signal_margin   = abs(bull - bear)   — the directional lead
          * total_signals   = bull + bear        — how much evidence exists
          * agreement_ratio = winning / total    — how one-sided it is
        The score is the margin discounted by the agreement ratio (rounded), so
        e.g. 4-vs-3 (margin 1, agreement 0.57) collapses to 1 instead of looking
        like very-high conviction. It is capped at the raw winning-side count so
        normalization can only *lower* (never raise) size vs. current behavior.
        A tied/empty tally returns 0 (→ size 0 / skip).
        """
        total = bullish_signals + bearish_signals
        if total <= 0:
            return 0
        margin = abs(bullish_signals - bearish_signals)
        if margin <= 0:
            return 0
        agreement = max(bullish_signals, bearish_signals) / total  # 0.5 .. 1.0
        raw = bearish_signals if strategy == 'put' else bullish_signals
        eff = int(margin * agreement + 0.5)  # round half up
        return max(0, min(eff, raw))

    def _cost_model(self):
        """Lazily build (and cache) the cost model used by the EV gate."""
        if self._phase2_cm is None:
            from cost_model import CostModel, load_cost_config_from_env
            self._phase2_cm = CostModel(load_cost_config_from_env())
        return self._phase2_cm

    def _dte_distance_mult(self, dte) -> float:
        """Score multiplier in [0.8, 1.0]: 1.0 at OPTION_TARGET_DTE, decaying
        with distance across the configured DTE window."""
        target = self.option_target_dte
        span = max(self.option_max_dte - target, target - self.option_min_dte, 1)
        dist = abs(dte - target) / span
        return max(0.8, 1.0 - 0.2 * min(dist, 1.0))

    def _delta_distance_mult(self, delta, target) -> float:
        """Score multiplier in [0.8, 1.0]: 1.0 at the target delta, decaying with
        distance across OPTION_MAX_DELTA_DISTANCE."""
        md = self.option_max_delta_distance or 0.0
        if md <= 0:
            return 1.0
        dist = abs(delta - target) / md
        return max(0.8, 1.0 - 0.2 * min(dist, 1.0))

    def evaluate_contract_phase2(self, *, dte, delta, has_real_delta, strategy,
                                 bid, ask, volume, open_interest,
                                 volume_present=True, oi_present=True):
        """Phase 2 advisory contract gate (pure; no network).

        Returns (reject_reason, score_mult). reject_reason is None when the
        contract passes; otherwise one of: bad_dte, bad_delta, wide_spread,
        negative_ev, low_volume, low_open_interest, missing_liquidity_data.
        score_mult is a preference multiplier (1.0 = no nudge). Every sub-gate is
        OFF unless its env flag is set, so with defaults this returns (None, 1.0)
        and selection is unchanged.
        """
        score_mult = 1.0
        strategy = (strategy or 'call').lower()

        # 1) DTE window + target preference.
        if self.use_dte_targeting and dte is not None:
            if dte < self.option_min_dte or dte > self.option_max_dte:
                return ('bad_dte', 1.0)
            score_mult *= self._dte_distance_mult(dte)

        # 2) Delta targeting (only with a REAL delta; missing -> fall back).
        if self.use_delta_targeting and has_real_delta and delta is not None:
            target = (self.option_target_call_delta if strategy == 'call'
                      else self.option_target_put_delta)
            if abs(delta - target) > self.option_max_delta_distance:
                return ('bad_delta', 1.0)
            score_mult *= self._delta_distance_mult(delta, target)

        # 3) Cost / EV gate. Fail-open on any computation error.
        if self.use_cost_ev_gate:
            try:
                b = float(bid or 0.0)
                a = float(ask or 0.0)
                spread_frac = (a - b) / a if a > 0 else 1.0
                if spread_frac > self.max_option_spread_pct:
                    return ('wide_spread', 1.0)
                gross_target_pct = self.base_take_profit * 100.0
                edge = self._cost_model().adjusted_expectancy(
                    gross_target_pct, b, a, qty=1, hold_days=max(0, int(dte or 0)))
                if edge < self.min_post_cost_edge:
                    return ('negative_ev', 1.0)
            except Exception:
                pass  # never block a trade on a broken cost calc

        # 4) Liquidity thresholds.
        if self.use_option_liquidity_filter:
            if not volume_present or not oi_present:
                if self.require_option_liquidity_data:
                    return ('missing_liquidity_data', 1.0)
            else:
                if float(volume or 0) < self.min_option_volume:
                    return ('low_volume', 1.0)
                if float(open_interest or 0) < self.min_option_open_interest:
                    return ('low_open_interest', 1.0)

        return (None, score_mult)

    def select_best_option(self, contracts, current_price, strategy=None):
        """Select best option using enhanced ML scoring with call/put intelligence.

        Args:
            strategy: force 'call' or 'put'. If None, auto-detect via
                determine_option_strategy() (preserves existing behavior).
        """
        # Determine optimal strategy (call or put)
        if strategy:
            strategy = strategy.lower()
        else:
            strategy = self.determine_option_strategy()

        print(f"[STRATEGY] Selected strategy: {strategy.upper()}")
        print(f"[CONTRACTS] Analyzing {len(contracts)} contracts")

        best_option = None
        best_score = -1
        validated_count = 0

        # News ranking nudge (fail-open): fetch once for this underlying and
        # precompute a multiplier that boosts contracts aligned with bullish/
        # bearish coverage and trims those against it. 1.0 when unavailable.
        news_mult = 1.0
        if self.news_service:
            try:
                from news import news_score_multiplier, NewsConfig
                news = self.news_service.get_news(self.ticker)
                news_mult = news_score_multiplier(news, strategy, NewsConfig.from_env())
                if news_mult != 1.0:
                    print(f"[NEWS] Ranking multiplier x{news_mult:.3f} for "
                          f"{self.ticker} {strategy.upper()} "
                          f"(score {news.get('score')} {news.get('label')})")
            except Exception as e:
                print(f"[NEWS] ranking error (ignored): {e}")

        for contract in contracts:
            strike = float(contract['strike_price'])
            contract_type = contract.get('type', 'call').lower()
            expiration = contract['expiration_date']

            # Filter by strategy and moneyness
            if strategy == 'call':
                # For calls, prefer ITM or near-the-money
                if strike > current_price * 1.05:  # Skip far OTM calls
                    continue
                if contract_type != 'call':
                    continue

                # Calculate call-specific metrics
                delta = min(0.95, max(0.05, (current_price - strike) / current_price * 0.7 + 0.5))
                moneyness = (current_price - strike) / strike

            else:  # strategy == 'put'
                # For puts, prefer ITM or near-the-money
                if strike < current_price * 0.95:  # Skip far OTM puts
                    continue
                if contract_type != 'put':
                    continue

                # Calculate put-specific metrics
                delta = min(-0.05, max(-0.95, (strike - current_price) / current_price * 0.7 - 0.5))
                moneyness = (strike - current_price) / strike

            # Validate that this option actually exists by checking if we can get a quote
            option_symbol = contract['symbol']

            # Days-to-expiration for the Phase 2 DTE gate (None if unparseable).
            try:
                dte = (datetime.strptime(expiration, '%Y-%m-%d').date()
                       - datetime.now().date()).days
            except Exception:
                dte = None

            # Get pricing data - either from API or mock Black-Scholes calculation
            ask_price = 0
            bid_price = 0
            spread = 0
            # Track whether liquidity data was actually supplied (vs defaulted)
            # so the Phase 2 liquidity filter can fail-open on missing data.
            volume_present = contract.get('volume') is not None
            oi_present = contract.get('open_interest') is not None
            try:
                volume = float(contract.get('volume', 0) or 0)
            except (TypeError, ValueError):
                volume = 0
            try:
                open_interest = float(contract.get('open_interest', 0) or 0)
            except (TypeError, ValueError):
                open_interest = 0

            if contract.get('mock', False):
                # Use mock Black-Scholes prices
                bid_price = contract.get('mock_bid', 0)
                ask_price = contract.get('mock_ask', 0)
                spread = ask_price - bid_price
                validated_count += 1
                print(f"[MOCK] ${strike:.2f} {contract_type.upper()} exp {expiration} - Bid: ${bid_price:.2f}, Ask: ${ask_price:.2f} (Black-Scholes)")
            else:
                # Try to get real quote from API
                option_quote = self.get_option_price(option_symbol)
                if not option_quote or option_quote['ask'] <= 0:
                    print(f"[SKIP] No valid quote for {option_symbol} (Strike: ${strike:.2f}, Exp: {expiration})")
                    continue

                ask_price = option_quote['ask']
                bid_price = option_quote['bid']
                spread = ask_price - bid_price
                spread_pct = (spread / ask_price * 100) if ask_price > 0 else 100

                validated_count += 1
                print(f"[VALIDATED] ${strike:.2f} {contract_type.upper()} exp {expiration} - Bid: ${bid_price:.2f}, Ask: ${ask_price:.2f}, Vol: {volume}, OI: {open_interest}")

                # Skip options whose spread is wider than the configured max.
                if spread_pct > self.max_spread_pct:
                    print(f"[SKIP] Spread too wide: {spread_pct:.1f}% "
                          f"(max {self.max_spread_pct:.0f}%)")
                    continue

            # Budget filter: skip contracts whose per-contract cost exceeds the
            # max budget for a single trade. Keeps selection consistent with the
            # budget enforced later in place_order_with_stops (smart_trader.py).
            contract_cost = ask_price * 100
            if ask_price > 0 and contract_cost > self.max_budget_per_trade:
                print(f"[SKIP] ${strike:.2f} {contract_type.upper()} exp {expiration} - cost ${contract_cost:.2f} exceeds budget ${self.max_budget_per_trade:.2f}")
                continue

            # Calculate option metrics. delta/gamma/theta/vega/iv start as
            # heuristic fallbacks; when USE_REAL_GREEKS is on they're overridden
            # below with live Alpaca snapshot values where available.
            option_data = {
                'symbol': contract['symbol'],
                'underlying': self.ticker,
                'strike': strike,
                'expiration': contract['expiration_date'],
                'type': strategy,
                'delta': delta,
                'gamma': 0.01,
                'theta': -0.05,
                'vega': 0.10,
                'iv': 0.25,
                'moneyness': abs(moneyness),
                'mock': contract.get('mock', False),
                'ask': ask_price,
                'bid': bid_price,
                'spread': spread,
                'volume': volume,
                'open_interest': open_interest
            }

            # Wire real Greeks/IV from the Alpaca options snapshot (opt-in,
            # fail-open). Fetched when USE_REAL_GREEKS overrides scoring inputs OR
            # when USE_DELTA_TARGETING needs a real delta to gate on. Only the
            # fields the snapshot returns are used; missing ones keep fallbacks.
            snap_greeks = {}
            if (self.use_real_greeks or self.use_delta_targeting) and not contract.get('mock', False):
                snap_greeks = self.get_option_snapshot(option_symbol)
                if self.use_real_greeks:
                    log = self._apply_real_greeks(option_data, snap_greeks)
                    print(f"[GREEKS] {self.ticker} {option_symbol} {log}")
            has_real_delta = isinstance(snap_greeks.get('delta'), (int, float))
            gate_delta = snap_greeks['delta'] if has_real_delta else option_data['delta']

            # Phase 2 contract gate (DTE / delta / cost-EV / liquidity). All
            # sub-gates are off by default -> (None, 1.0). A rejection logs the
            # reason and skips the contract; otherwise score_mult nudges ranking.
            reject_reason, phase2_mult = self.evaluate_contract_phase2(
                dte=dte, delta=gate_delta, has_real_delta=has_real_delta,
                strategy=strategy, bid=bid_price, ask=ask_price,
                volume=volume, open_interest=open_interest,
                volume_present=volume_present, oi_present=oi_present)
            if reject_reason:
                print(f"[REJECT] {option_symbol} reason={reject_reason}")
                continue

            # Calculate base score
            score = self.calculate_option_score(option_data)

            # Boost score for strategy alignment
            if contract_type == strategy:
                score *= 1.1  # 10% boost for matching strategy

            # News ranking nudge (fail-open; 1.0 when news unavailable)
            score *= news_mult

            # Phase 2 target preference (1.0 unless DTE/delta targeting is on)
            score *= phase2_mult

            # Add liquidity scoring (critical for real trading)
            if not contract.get('mock', False):
                # Volume scoring (max 15 points)
                if volume > 100:
                    volume_score = min(15, volume / 100)
                else:
                    volume_score = volume / 10  # Penalize low volume

                # Open interest scoring (max 15 points)
                if open_interest > 100:
                    oi_score = min(15, open_interest / 100)
                else:
                    oi_score = open_interest / 10  # Penalize low OI

                # Spread scoring (tighter spread = higher score, max 10 points)
                spread_score = max(0, 10 - spread_pct)

                liquidity_score = volume_score + oi_score + spread_score
                score += liquidity_score

                print(f"[LIQUIDITY] Vol: {volume_score:.1f}, OI: {oi_score:.1f}, Spread: {spread_score:.1f} = Total: {liquidity_score:.1f}")

            if score > best_score:
                best_score = score
                best_option = option_data
                best_option['score'] = score
                best_option['strategy_type'] = strategy

        print(f"[VALIDATION] {validated_count} contracts validated with real quotes")

        if best_option:
            # Carry the directional conviction (signal strength) onto the chosen
            # option so place_order_with_stops can size the position by confidence.
            best_option['confidence'] = self.last_signal_strength

        if best_option and not best_option.get('mock', False):
            print(f"[BEST OPTION] Strike: ${best_option['strike']:.2f} {best_option['type'].upper()}")
            print(f"[BEST OPTION] Expiration: {best_option['expiration']}")
            print(f"[BEST OPTION] Score: {best_option['score']:.2f}")

        return best_option

    def _confidence_to_quantity(self, strength) -> int:
        """Map a directional signal strength to a contract count.

        very high (>= conf_very_high_signals) -> 3 contracts
        high      (>= conf_high_signals)      -> 2 contracts
        regular   (otherwise)                 -> 1 contract

        Phase 3 sizing safety: only when USE_NORMALIZED_CONFIDENCE is on can a
        collapsed confidence (<= 0) size to 0 contracts (skip). In default mode
        the floor stays at 1, so existing behavior is unchanged.
        """
        try:
            s = int(strength)
        except (TypeError, ValueError):
            return 1
        if self.use_normalized_confidence and s <= 0:
            return 0
        if s >= self.conf_very_high_signals:
            return 3
        if s >= self.conf_high_signals:
            return 2
        return 1

    def place_order_with_stops(self, option: Dict, quantity: int = None):
        """Place order with dynamic stop loss and take profit.

        On a no-op/blocked entry this returns None and records a human-readable
        cause in ``self.last_block_reason`` so callers (e.g. the Telegram bot)
        can surface the actual reason instead of a generic message.
        """
        # Sizing: an explicit quantity (e.g. a manual Telegram trade) is always
        # honored. When the caller passes no quantity (the automated screener),
        # size by directional conviction carried on the option.
        if quantity is not None:
            order_quantity = quantity
        else:
            strength = option.get('confidence', self.last_signal_strength)
            order_quantity = self._confidence_to_quantity(strength)
            tier = {3: "very high", 2: "high", 1: "regular", 0: "skip"}.get(order_quantity, "regular")
            print(f"[SIZE] conviction {strength} -> {order_quantity} contract(s) ({tier})")
        self.last_block_reason = None

        # Phase 3 sizing safety: a confidence that collapses to 0 contracts is a
        # NO_TRADE — skip rather than place a non-positive-size order. Only
        # reachable with USE_NORMALIZED_CONFIDENCE on (default sizing floors at 1).
        if order_quantity is not None and order_quantity <= 0:
            print(f"[SIZE] confidence too weak to size ({order_quantity}); skipping entry")
            self.last_block_reason = "signal confidence too weak to size a position"
            return None

        # Duplicate-order guard: never stack a second position/order on the same
        # contract. Fail-closed if broker positions/orders can't be verified.
        option_symbol = option['symbol']
        if self._has_open_or_pending(option_symbol):
            print(f"[DUP] Open position/order already exists for {option_symbol}; skipping.")
            self.last_block_reason = "an open position or pending order already exists for this contract"
            return None

        # Calculate dynamic levels based on market conditions
        underlying_symbol = option.get('underlying', self.ticker)
        dynamic_levels = self.calculate_dynamic_levels(underlying_symbol)

        print(f"[DYNAMIC LEVELS] Market Regime: {dynamic_levels['market_regime']}")
        print(f"[DYNAMIC LEVELS] Volatility: {dynamic_levels['volatility']:.2%}")
        print(f"[DYNAMIC LEVELS] Momentum: {dynamic_levels['momentum']:.2%}")
        print(f"[DYNAMIC LEVELS] Stop Loss: {dynamic_levels['stop_loss_percent']:.2%}")
        print(f"[DYNAMIC LEVELS] Take Profit: {dynamic_levels['take_profit_percent']:.2%}")

        # Get current option price before placing order
        option_symbol = option['symbol']
        current_option_price = self.get_option_price(option_symbol)

        if not current_option_price:
            print(f"[ERROR] Cannot get current price for option {option_symbol}")
            self.last_block_reason = "could not get a current quote for the contract (untradeable or no market data)"
            return None

        entry_price = current_option_price['ask']
        bid_price = current_option_price['bid']

        print(f"[OPTION PRICE] Bid: ${bid_price:.2f}, Ask: ${entry_price:.2f}")

        # Validate price is reasonable
        if entry_price <= 0:
            print(f"[ERROR] Invalid option price: ${entry_price}")
            self.last_block_reason = f"the contract returned an invalid ask price (${entry_price})"
            return None

        # Stale-quote guard: never trade on an out-of-date quote. Fail-closed:
        # a missing/unparseable timestamp is treated as stale.
        if not self._quote_is_fresh(current_option_price):
            print(f"[STALE] Entry quote for {option_symbol} is stale/unverifiable "
                  f"(ts={current_option_price.get('ts')}, "
                  f"max_age={self.quote_max_age_sec}s); skipping.")
            self.last_block_reason = "the entry quote was stale/unverifiable (skipped for safety)"
            return None

        # Calculate total cost
        total_cost = entry_price * 100 * order_quantity
        print(f"[COST] Total cost for {order_quantity} contract(s): ${total_cost:.2f}")

        # Check if within budget
        if total_cost > self.max_budget_per_trade:
            print(f"[WARNING] Cost ${total_cost:.2f} exceeds budget ${self.max_budget_per_trade:.2f}")
            # Adjust quantity to fit budget
            order_quantity = int(self.max_budget_per_trade / (entry_price * 100))
            if order_quantity < 1:
                print(f"[ERROR] Cannot afford even 1 contract at ${entry_price:.2f}")
                self.last_block_reason = (f"one contract costs ${entry_price * 100:.2f}, over the "
                                          f"${self.max_budget_per_trade:.2f} per-trade budget")
                return None
            total_cost = entry_price * 100 * order_quantity
            print(f"[ADJUSTED] New quantity: {order_quantity} contract(s), cost: ${total_cost:.2f}")

        # Sentiment risk filter (fail-open): scale size down in fearful/euphoric
        # markets and block aggressive entries during Extreme Fear. Never crashes.
        if self.sentiment_service:
            try:
                from sentiment import adjust_trade_risk_by_sentiment, summarize_for_log
                sentiment = self.sentiment_service.get_sentiment()
                print(f"[SENTIMENT] {summarize_for_log(sentiment)}")
                decision = adjust_trade_risk_by_sentiment(
                    {'size': order_quantity,
                     'confidence': dynamic_levels.get('confidence'),
                     'direction': option.get('type')},
                    sentiment,
                )
                print(f"[SENTIMENT] {decision['reason']}")
                if not decision['allowed']:
                    print("[SENTIMENT] Trade blocked by sentiment filter")
                    self.last_block_reason = f"sentiment filter: {decision['reason']}"
                    return None
                if decision['adjusted_size'] != order_quantity:
                    order_quantity = decision['adjusted_size']
                    total_cost = entry_price * 100 * order_quantity
                    print(f"[SENTIMENT] Adjusted quantity: {order_quantity} "
                          f"contract(s), cost: ${total_cost:.2f}")
            except Exception as e:
                print(f"[SENTIMENT] filter error (ignored): {e}")

        # News gate (fail-open): block or shrink an entry when recent headlines
        # strongly OPPOSE the trade direction. Mirrors the sentiment gate above.
        if self.news_service:
            try:
                from news import adjust_trade_by_news, summarize_for_log, NewsConfig
                news = self.news_service.get_news(self.ticker)
                print(f"[NEWS] {summarize_for_log(news)}")
                decision = adjust_trade_by_news(
                    {'size': order_quantity,
                     'confidence': dynamic_levels.get('confidence'),
                     'direction': option.get('type')},
                    news, NewsConfig.from_env(),
                )
                print(f"[NEWS] {decision['reason']}")
                if not decision['allowed']:
                    print("[NEWS] Trade blocked by news filter")
                    self.last_block_reason = f"news filter: {decision['reason']}"
                    return None
                if decision['adjusted_size'] != order_quantity:
                    order_quantity = decision['adjusted_size']
                    total_cost = entry_price * 100 * order_quantity
                    print(f"[NEWS] Adjusted quantity: {order_quantity} "
                          f"contract(s), cost: ${total_cost:.2f}")
            except Exception as e:
                print(f"[NEWS] gate error (ignored): {e}")

        # ---- Phase 4: portfolio-level greek gate (opt-in, fail-open) ----------
        # Runs AFTER sizing so the projection uses the final contract count, and
        # BEFORE the fail-closed risk engine. Blocks an entry that would push the
        # aggregate book past a delta/vega/theta/same-direction/per-underlying
        # cap. OFF by default -> no-op (allowed) and behavior is unchanged.
        if getattr(self, 'use_portfolio_greek_limits', False):
            pf_verdict = self._portfolio_greek_check(option, order_quantity)
            if not pf_verdict.get('allowed', True):
                print(f"[PORTFOLIO] BLOCKED: {pf_verdict.get('reason')}")
                self.last_block_reason = (
                    "portfolio greek limit: " + str(pf_verdict.get('reason'))
                )
                return None

        # ---- Hard risk gate (fail-closed; last line of capital protection) ----
        # Runs AFTER budget/sentiment sizing so trade_cost is final, and BEFORE
        # the order is sent. On any breach the trade is blocked and the verdict
        # is recorded; the verdict is also attached to the placed decision below.
        risk_verdict = self._risk_check(total_cost, qty=order_quantity)
        if not risk_verdict.get('allowed', False):
            print(f"[RISK] BLOCKED: {risk_verdict.get('reason')}")
            self._record_blocked_decision(
                option, underlying_symbol, dynamic_levels,
                current_option_price, order_quantity, risk_verdict,
            )
            self.last_block_reason = (
                "risk engine: " + self._humanize_risk_breaches(risk_verdict)
            )
            return None
        print(f"[RISK] OK ({risk_verdict.get('reason')})")

        # Place main order
        order_data = {
            'symbol': option['symbol'],
            'qty': order_quantity,
            'side': 'buy',
            'type': 'market',
            'time_in_force': 'day',
            'asset_class': 'us_option'
        }

        # Network/API failures must never crash the caller (fail-safe to None).
        try:
            response = requests.post(
                f"{self.base_url}/v2/orders",
                headers=self.headers,
                json=order_data,
                timeout=10,
            )
        except Exception as e:
            print(f"[ORDER ERROR] Network/API failure placing "
                  f"{option['symbol']}: {e}")
            self.last_block_reason = "a network/API error occurred submitting the order"
            return None

        if response.status_code in [200, 201]:
            try:
                order = response.json()
            except Exception as e:
                print(f"[ORDER ERROR] Order accepted (HTTP "
                      f"{response.status_code}) but response unparseable: {e}")
                self.last_block_reason = "the broker accepted the order but returned an unreadable response"
                return None
            order_id = order.get('id')

            # Confirm the fill instead of assuming the quote ask: read the REAL
            # filled_avg_price / filled_qty, and handle rejected/partial fills.
            filled = self._await_fill(order_id) or order
            status = filled.get('status')
            try:
                filled_qty = float(filled.get('filled_qty', 0) or 0)
            except (TypeError, ValueError):
                filled_qty = 0.0

            if status in ('rejected', 'canceled', 'expired') and filled_qty <= 0:
                print(f"[FILL] Order {order_id} {status} with no fill; no position opened.")
                self.last_block_reason = f"the broker {status} the order with no fill"
                return None

            entry_ask = current_option_price['ask']  # quote ask drives cost model
            fill_price = None
            try:
                fap = filled.get('filled_avg_price')
                if fap:
                    fill_price = float(fap)
            except (TypeError, ValueError):
                fill_price = None

            if fill_price and fill_price > 0:
                entry_price = fill_price  # realized buy fill -> stop/take/triggers
                print(f"[FILL] {order_id} {status}: {filled_qty:g} @ ${fill_price:.2f}")
            else:
                print(f"[FILL] {order_id} {status}: no fill price yet; "
                      f"using quote ask ${entry_price:.2f}")

            if filled_qty > 0 and int(filled_qty) != order_quantity:
                print(f"[FILL] Partial/adjusted fill: {filled_qty:g} of {order_quantity}")
                order_quantity = int(filled_qty)

            # Store trade info with dynamic levels
            trade_info = {
                'order_id': order_id,
                'symbol': option['symbol'],
                'underlying_symbol': underlying_symbol,
                'entry_price': entry_price,
                'entry_bid': bid_price,
                'entry_ask': entry_ask,
                'quantity': order_quantity,
                'dynamic_stop_loss_percent': dynamic_levels['stop_loss_percent'],
                'dynamic_take_profit_percent': dynamic_levels['take_profit_percent'],
                'trailing_stop_distance': dynamic_levels['trailing_stop_distance'],
                'stop_loss_trigger': entry_price * (1 - dynamic_levels['stop_loss_percent']),
                'take_profit_trigger': entry_price * (1 + dynamic_levels['take_profit_percent']),
                'partial_close_done': False,
                'trailing_stop_active': False,
                'highest_price': entry_price,
                'entry_time': datetime.now().isoformat(),
                'market_conditions': {
                    'volatility': dynamic_levels['volatility'],
                    'momentum': dynamic_levels['momentum'],
                    'market_regime': dynamic_levels['market_regime']
                }
            }

            # Phase 10H: freeze the entry-time belief (expected EV, POP,
            # max loss) so the calibration analytics can score this trade
            # when it closes. Analytics-only metadata — a failed stamp never
            # affects the order that was just placed.
            try:
                from entry_ev_stamp import compute_entry_stamp
                stamp = compute_entry_stamp(
                    option, dynamic_levels, entry_price, order_quantity,
                    bid=bid_price, ask=entry_ask)
                if stamp:
                    trade_info['metrics'] = stamp
                    print(f"[EV STAMP] EV ${stamp['expected_value']:+.2f} "
                          f"POP {stamp['probability_of_profit']:.0%} "
                          f"max loss ${stamp['max_loss']:.0f}")
            except Exception as e:
                print(f"[EV STAMP] skipped: {e}")

            action = (option.get('type') or 'call').upper()  # CALL/PUT
            analysis_ctx = {
                'direction': action,
                'momentum': dynamic_levels.get('momentum', 0),
                'confidence': option.get('score', 0),
            }

            # Shadow recorder owns the RL loop: log the decision under ONE
            # decision_id (episode row + pending RL experience). It never blocks
            # the trade. When it is disabled, fall back to the legacy RL hook.
            if getattr(self, 'shadow_recorder', None):
                try:
                    quote = self.get_option_price(option['symbol'])
                    decision_id = self.shadow_recorder.on_decision(
                        symbol=option['symbol'],
                        underlying=underlying_symbol,
                        analysis=analysis_ctx,
                        quote=quote,
                        entry_premium=entry_price,
                        qty=order_quantity,
                        mode='live-paper',
                        as_of=datetime.now().isoformat(),
                        day_of_week=datetime.now().weekday(),
                        risk=risk_verdict,
                    )
                    if decision_id:
                        trade_info['decision_id'] = decision_id
                except Exception as e:
                    print(f"[SHADOW] on_decision failed: {e}")
            elif self.rl_advisor:
                try:
                    advice = self.rl_advisor.observe_and_log(
                        analysis_ctx, order['id'], action,
                        day_of_week=datetime.now().weekday()
                    )
                    print(f"[RL] Recommended: {advice['recommended_action']} | "
                          f"Rule: {advice['rule_action']} | "
                          f"Agree: {advice['agreement']}")
                except Exception as e:
                    print(f"[RL] observe failed: {e}")

            # Save to active trades file (after decision_id is attached).
            self.save_active_trade(trade_info)

            return order

        # Non-2xx: surface the broker's rejection reason then fail.
        try:
            print(f"[ORDER ERROR] Order rejected for {option['symbol']} "
                  f"(HTTP {response.status_code}): {response.text[:300]}")
            self.last_block_reason = (f"the broker rejected the order "
                                      f"(HTTP {response.status_code}): {response.text[:200]}")
        except Exception:
            print(f"[ORDER ERROR] Order rejected (HTTP {response.status_code})")
            self.last_block_reason = f"the broker rejected the order (HTTP {response.status_code})"
        return None

    def save_active_trade(self, trade_info: Dict):
        """Save active trade for monitoring"""
        active_file = 'active_trades.json'

        if os.path.exists(active_file):
            with open(active_file, 'r') as f:
                active_trades = json.load(f)
        else:
            active_trades = []

        active_trades.append(trade_info)

        with open(active_file, 'w') as f:
            json.dump(active_trades, f, indent=2, default=str)

    def monitor_positions(self):
        """Monitor positions for stop loss and take profit"""
        active_file = 'active_trades.json'

        if not os.path.exists(active_file):
            return

        with open(active_file, 'r') as f:
            active_trades = json.load(f)

        positions = self.get_positions()
        updated_trades = []

        for trade in active_trades:
            position = next((p for p in positions if p['symbol'] == trade['symbol']), None)

            if not position:
                # Position closed externally. Estimate realized P/L from the last
                # known option quote so the learning loop sees a real number
                # instead of the old default 0 (which silently killed the RL
                # update). Fail-open: fall back to 0 if no quote is available.
                closed_pnl = 0
                try:
                    last = self.get_option_price(trade['symbol'])
                    entry_price = trade.get('entry_price')
                    if last and entry_price:
                        exit_px = last.get('mid') or last.get('ask') or last.get('bid') or 0
                        if exit_px and entry_price:
                            closed_pnl = ((exit_px - entry_price) / entry_price) * 100
                except Exception:
                    closed_pnl = 0
                self.record_trade_outcome(trade, 'closed', closed_pnl)
                continue

            current_price = float(position['current_price']) if position['current_price'] else 0
            entry_price = trade['entry_price']

            if current_price == 0:
                updated_trades.append(trade)
                continue

            pnl_percent = ((current_price - entry_price) / entry_price) * 100

            # Update highest price for trailing stop. Arming is gated by
            # TRAILING_ARM_PROFIT_PCT so the trade must earn protection first
            # (legacy 0 = any uptick arms).
            from exit_manager import should_arm_trailing
            if current_price > trade['highest_price']:
                trade['highest_price'] = current_price
            if not trade.get('trailing_stop_active') and should_arm_trailing(
                    entry_price, current_price, self.trailing_arm_profit_pct):
                trade['trailing_stop_active'] = True

            # Use dynamic levels or fallback to static
            stop_loss_percent = trade.get('dynamic_stop_loss_percent', 0.10) * 100
            take_profit_percent = trade.get('dynamic_take_profit_percent', 0.20) * 100
            trailing_distance = trade.get('trailing_stop_distance', 0.05)

            # Re-calculate dynamic levels for current market conditions
            underlying_symbol = trade.get('underlying_symbol', self.ticker)
            if underlying_symbol:
                current_dynamic = self.calculate_dynamic_levels(underlying_symbol)
                # Update levels if market conditions have changed significantly
                old_regime = trade.get('market_conditions', {}).get('market_regime', 'ranging')
                new_regime = current_dynamic['market_regime']

                if old_regime != new_regime:
                    print(f"[REGIME CHANGE] {old_regime} → {new_regime}")
                    stop_loss_percent = current_dynamic['stop_loss_percent'] * 100
                    take_profit_percent = current_dynamic['take_profit_percent'] * 100
                    trailing_distance = current_dynamic['trailing_stop_distance']

            # Check for partial close at dynamic take profit level
            partial_threshold = take_profit_percent * 0.6  # 60% of take profit target
            if pnl_percent >= partial_threshold and not trade['partial_close_done']:
                self.close_partial_position(trade, position, 0.4)  # Close 40%
                trade['partial_close_done'] = True
                print(f"[PARTIAL CLOSE] Closed 40% at {pnl_percent:.1f}% (target: {partial_threshold:.1f}%)")

            else:
                # Phase 5: the stop / take-profit / trailing-stop decision is now
                # made by the shared exit manager so the Telegram monitor enforces
                # the SAME logic. Roll-on-profit, partial close, and the richer
                # should_exit_dynamically checks stay scheduler-only and keep their
                # legacy reason codes, so recorded outcomes are byte-identical.
                # check_expiration=False: the scheduler keeps expiration handling
                # inside should_exit_dynamically (reason 'dynamic_exit').
                from exit_manager import evaluate_exit, enforce_exit
                _levels = {
                    'stop_loss_percent': stop_loss_percent,
                    'take_profit_percent': take_profit_percent,
                    'trailing_stop_distance': trailing_distance,
                }
                _decision = evaluate_exit(
                    trade, current_price, _levels,
                    roll_enabled=self.roll_enabled, check_expiration=False)

                # Stop loss.
                if _decision.action == 'stop_loss':
                    print(f"[STOP LOSS] Dynamic stop at {stop_loss_percent:.1f}%")
                    enforce_exit(self, trade, position, 'dynamic_stop_loss',
                                 pnl_percent, 'scheduler', current_price)
                    continue

                # Roll-on-profit: at ROLL_TRIGGER_PCT (default MAX_TAKE_PROFIT)
                # close the winner and re-enter a cheaper, further-OTM contract on
                # the same underlying/direction. Checked before the dynamic
                # full-exit so the winner can run to the roll trigger. Only active
                # when roll_enabled.
                elif self.roll_enabled and pnl_percent >= self.roll_trigger_pct:
                    print(f"[ROLL] Trigger at {pnl_percent:.1f}% "
                          f"(>= {self.roll_trigger_pct:.1f}%)")
                    if self.roll_position(trade, position, pnl_percent):
                        self.record_trade_outcome(trade, 'roll_take_profit', pnl_percent)
                        continue
                    # No re-entry target found -> fall back to a flat take-profit.
                    self.close_position(trade, position, 'roll_failed_take_profit')
                    self.record_trade_outcome(trade, 'roll_failed_take_profit', pnl_percent)
                    continue

                # Dynamic take profit (full exit). When rolling is enabled the
                # exit manager suppresses this so winners run to roll_trigger_pct.
                elif _decision.action == 'take_profit':
                    print(f"[TAKE PROFIT] Dynamic target at {take_profit_percent:.1f}%")
                    enforce_exit(self, trade, position, 'dynamic_take_profit',
                                 pnl_percent, 'scheduler', current_price)
                    continue

                # Dynamic trailing stop.
                elif _decision.action == 'trailing_stop':
                    print(f"[TRAILING STOP] Dynamic trailing at {trailing_distance:.1%}")
                    enforce_exit(self, trade, position, 'dynamic_trailing_stop',
                                 pnl_percent, 'scheduler', current_price)
                    continue

            # Dynamic exit based on market conditions
            if self.should_exit_dynamically(trade, position, current_price):
                self.close_position(trade, position, 'dynamic_exit')
                self.record_trade_outcome(trade, 'dynamic_exit', pnl_percent)
                continue

            updated_trades.append(trade)

        # Save updated active trades
        with open(active_file, 'w') as f:
            json.dump(updated_trades, f, indent=2, default=str)

    def close_partial_position(self, trade: Dict, position: Dict, percentage: float):
        """Close partial position"""
        qty_to_close = math.floor(float(position['qty']) * percentage)

        if qty_to_close > 0:
            order_data = {
                'symbol': trade['symbol'],
                'qty': qty_to_close,
                'side': 'sell',
                'type': 'market',
                'time_in_force': 'day',
                'asset_class': 'us_option'
            }

            requests.post(f"{self.base_url}/v2/orders", headers=self.headers, json=order_data)
            print(f"[PARTIAL CLOSE] Closed {qty_to_close} contracts at +20% profit")

    def close_position(self, trade: Dict, position: Dict, reason: str):
        """Close entire position"""
        order_data = {
            'symbol': trade['symbol'],
            'qty': position['qty'],
            'side': 'sell',
            'type': 'market',
            'time_in_force': 'day',
            'asset_class': 'us_option'
        }

        requests.post(f"{self.base_url}/v2/orders", headers=self.headers, json=order_data)
        print(f"[CLOSE] Position closed - Reason: {reason}")

    @staticmethod
    def direction_from_occ(occ_symbol: str) -> Optional[str]:
        """Infer 'call'/'put' from an OCC option symbol (…YYMMDD[C|P]NNNNNNNN).

        Active-trade rows store only the OCC ``symbol`` (no option type), so the
        roll path recovers direction from the standard layout: 8-digit strike,
        preceded by a single C/P, preceded by the 6-digit expiration. Returns
        None when the symbol doesn't match (caller then skips the roll).
        """
        if not occ_symbol or len(occ_symbol) < 9:
            return None
        cp = occ_symbol[-9].upper()
        if cp == 'C':
            return 'call'
        if cp == 'P':
            return 'put'
        return None

    @staticmethod
    def _roll_candidates(contracts, current_price, direction,
                         otm_pct, max_otm_pct):
        """Pure filter/sort for roll re-entry candidates (no network).

        Keeps same-direction contracts whose strike is between ``otm_pct`` and
        ``max_otm_pct`` out-of-the-money relative to ``current_price`` (higher
        strike for calls, lower for puts), sorted nearest-expiration first then
        nearest to the OTM floor. Quote/budget validation happens in
        ``select_roll_option``; this part is unit-testable offline.
        """
        if not current_price or current_price <= 0:
            return []
        direction = (direction or '').lower()
        out = []
        for c in contracts:
            if (c.get('type') or 'call').lower() != direction:
                continue
            try:
                strike = float(c['strike_price'])
            except (KeyError, TypeError, ValueError):
                continue
            if direction == 'call':
                low = current_price * (1 + otm_pct)
                high = current_price * (1 + max_otm_pct)
                if not (low <= strike <= high):
                    continue
                otm_dist = (strike - current_price) / current_price
            else:  # put
                low = current_price * (1 - max_otm_pct)
                high = current_price * (1 - otm_pct)
                if not (low <= strike <= high):
                    continue
                otm_dist = (current_price - strike) / current_price
            out.append((c, otm_dist))
        # Nearest expiration first, then closest to the OTM floor (least far OTM
        # that still clears the threshold) -> the cheapest reasonable re-entry.
        out.sort(key=lambda t: (t[0].get('expiration_date', '9999-99-99'), t[1]))
        return [c for c, _ in out]

    def select_roll_option(self, contracts, current_price, direction):
        """Pick a cheaper, further-OTM contract for a roll re-entry.

        Reuses the OCC quote/spread/budget validation logic but, unlike
        ``select_best_option``, deliberately targets OTM strikes (which
        ``select_best_option`` skips). Returns an option_data dict shaped for
        ``place_order_with_stops`` or None if nothing qualifies.
        """
        direction = (direction or '').lower()
        candidates = self._roll_candidates(
            contracts, current_price, direction,
            self.roll_otm_pct, self.roll_max_otm_pct,
        )
        print(f"[ROLL] {len(candidates)} OTM {direction} candidate(s) "
              f"({self.roll_otm_pct:.0%}-{self.roll_max_otm_pct:.0%} OTM)")

        for contract in candidates:
            option_symbol = contract['symbol']
            strike = float(contract['strike_price'])
            expiration = contract.get('expiration_date')

            if contract.get('mock', False):
                bid_price = contract.get('mock_bid', 0)
                ask_price = contract.get('mock_ask', 0)
            else:
                quote = self.get_option_price(option_symbol)
                if not quote or quote.get('ask', 0) <= 0:
                    continue
                ask_price = quote['ask']
                bid_price = quote['bid']
                spread_pct = ((ask_price - bid_price) / ask_price * 100) if ask_price > 0 else 100
                if spread_pct > self.max_spread_pct:
                    continue

            if ask_price <= 0:
                continue
            if ask_price * 100 > self.max_budget_per_trade:
                # Too rich for the roll; candidates are sorted cheapest-ish so a
                # later one may fit — keep scanning.
                continue

            print(f"[ROLL] Re-entry target ${strike:.2f} {direction.upper()} "
                  f"exp {expiration} ask ${ask_price:.2f}")
            return {
                'symbol': option_symbol,
                'underlying': self.ticker,
                'strike': strike,
                'expiration': expiration,
                'type': direction,
                'ask': ask_price,
                'bid': bid_price,
                'mock': contract.get('mock', False),
                'score': 0,
                'strategy_type': direction,
            }

        print("[ROLL] No OTM candidate passed quote/budget validation")
        return None

    def roll_position(self, trade: Dict, position: Dict, pnl_percent: float) -> bool:
        """Roll a winning option into a cheaper, further-OTM contract.

        Closes the current position (locking gains) only after a valid re-entry
        target is found, then opens it via ``place_order_with_stops`` so all
        gates (budget, sentiment, fail-closed risk, fill readback) still apply.
        Returns True if the old position was closed (rolled or, if re-entry
        failed after close, simply taken-profit); False if no target was found
        and the caller should flat-close instead.
        """
        underlying = trade.get('underlying_symbol', self.ticker)
        direction = self.direction_from_occ(trade.get('symbol', ''))
        if direction is None:
            print(f"[ROLL] Could not parse direction from {trade.get('symbol')}; "
                  "skipping roll")
            return False

        current_price = self.get_current_price(underlying)
        if not current_price:
            print(f"[ROLL] No underlying price for {underlying}; skipping roll")
            return False

        contracts = self.get_option_contracts(underlying)
        new_opt = self.select_roll_option(contracts, current_price, direction)
        if not new_opt:
            print("[ROLL] No suitable OTM contract; taking profit flat instead")
            return False

        # Lock gains: close the winner first, then re-enter.
        self.close_position(trade, position, 'roll_take_profit')

        qty = int(trade.get('quantity', self.quantity) or self.quantity)
        order = self.place_order_with_stops(new_opt, qty)
        if order:
            print(f"[ROLL] {trade['symbol']} -> {new_opt['symbol']} "
                  f"(+{pnl_percent:.0f}% locked, re-entered OTM)")
            self._stamp_rolled_from(new_opt['symbol'], trade)
        else:
            print("[ROLL] Re-entry order did not fill; profit already locked "
                  "by close")
        # Old position is closed either way -> handled; don't flat-close again.
        return True

    def _stamp_rolled_from(self, new_symbol: str, old_trade: Dict):
        """Tag the freshly-saved roll re-entry row for traceability.

        Adds ``rolled_from`` and carries over the originating ``source`` (e.g.
        'alpaca_scheduler') so EOD close targeting still recognizes the new row.
        Fail-open: any error is logged and ignored.
        """
        active_file = 'active_trades.json'
        try:
            if not os.path.exists(active_file):
                return
            with open(active_file, 'r') as f:
                trades = json.load(f)
            for t in trades:
                if t.get('symbol') == new_symbol and 'rolled_from' not in t:
                    t['rolled_from'] = old_trade.get('symbol')
                    if old_trade.get('source') is not None:
                        t['source'] = old_trade['source']
                    break
            with open(active_file, 'w') as f:
                json.dump(trades, f, indent=2, default=str)
        except Exception as e:
            print(f"[ROLL] stamp rolled_from failed (ignored): {e}")

    def should_exit_dynamically(self, trade: Dict, position: Dict, current_price: float) -> bool:
        """Determine if position should be exited based on dynamic conditions"""
        underlying_symbol = trade.get('underlying_symbol', self.ticker)

        # Time-based exit (close if near expiration)
        if 'expiration' in trade:
            days_to_expiry = (datetime.strptime(trade['expiration'], '%Y-%m-%d') - datetime.now()).days
            if days_to_expiry <= 2:
                print("[EXIT] Near expiration")
                return True

        # SIGNAL_EXITS_ENABLED=0: skip every condition-based exit below
        # (momentum reversal, vol spike, regime change, profit giveback,
        # pullback-from-high) so the position runs its stop/target race.
        # Near-expiration above still applies.
        if not getattr(self, 'signal_exits_enabled', True):
            return False

        # Get current market conditions
        current_dynamic = self.calculate_dynamic_levels(underlying_symbol)
        entry_momentum = trade.get('market_conditions', {}).get('momentum', 0)
        current_momentum = current_dynamic['momentum']

        # Momentum reversal detection (more sophisticated)
        momentum_change = abs(current_momentum - entry_momentum)
        if momentum_change > 0.08 and current_momentum * entry_momentum < 0:  # Sign reversal
            print(f"[EXIT] Momentum reversal: {entry_momentum:.2%} → {current_momentum:.2%}")
            return True

        # Volatility spike exit
        entry_volatility = trade.get('market_conditions', {}).get('volatility', 0.20)
        current_volatility = current_dynamic['volatility']
        vol_increase = (current_volatility - entry_volatility) / entry_volatility

        if vol_increase > 0.5:  # 50% volatility increase
            print(f"[EXIT] Volatility spike: {entry_volatility:.1%} → {current_volatility:.1%}")
            return True

        # Market regime change exit (if unfavorable)
        entry_regime = trade.get('market_conditions', {}).get('market_regime', 'ranging')
        current_regime = current_dynamic['market_regime']

        # Exit if regime becomes unfavorable for options
        if entry_regime in ['trending', 'ranging'] and current_regime == 'volatile':
            entry_price = trade['entry_price']
            pnl_percent = ((current_price - entry_price) / entry_price) * 100

            # Only exit if we're not in significant profit
            if pnl_percent < 15:
                print(f"[EXIT] Unfavorable regime change: {entry_regime} → {current_regime}")
                return True

        # Price action based exit
        if trade.get('highest_price'):
            entry_price = trade['entry_price']
            highest_price = trade['highest_price']

            # Calculate maximum adverse excursion (MAE) and maximum favorable excursion (MFE)
            mae = min(0, ((current_price - entry_price) / entry_price)) * 100
            mfe = ((highest_price - entry_price) / entry_price) * 100

            # Exit if we've given back too much profit after a good run
            if mfe > 20 and mae < -10:  # Had 20%+ profit but now down 10%+
                print(f"[EXIT] Profit giveback: MFE {mfe:.1f}%, current MAE {mae:.1f}%")
                return True

            # Pullback from high threshold (dynamic based on volatility)
            pullback_threshold = 0.15 + (current_volatility - 0.20) * 0.5  # Adjust for volatility
            pullback = (highest_price - current_price) / highest_price

            if pullback > pullback_threshold:
                print(f"[EXIT] Pullback {pullback:.1%} > threshold {pullback_threshold:.1%}")
                return True

        return False

    def record_trade_outcome(self, trade: Dict, outcome: str, pnl_percent: float = 0):
        """Record trade outcome for learning"""
        trade_record = {
            'symbol': trade['symbol'],
            'entry_time': trade['entry_time'],
            'exit_time': datetime.now().isoformat(),
            'outcome': outcome,
            'pnl_percent': pnl_percent,
            'metrics': trade.get('metrics', {})
        }

        # Phase 10H: when the entry carried an EV stamp, flatten the frozen
        # beliefs plus the realized DOLLAR P/L onto the record so the
        # ev_calibration / pop_calibration loaders can read scheduler trades
        # directly (they key on top-level expected_value / probability_of_
        # profit / pnl). Records without a stamp keep their legacy shape.
        try:
            stamp = trade.get('metrics') or {}
            if isinstance(stamp, dict) and stamp.get('expected_value') is not None:
                pnl_dollars = None
                entry_price = trade.get('entry_price')
                qty = trade.get('quantity', 1) or 1
                if entry_price:
                    pnl_dollars = (float(entry_price) * 100.0 * float(qty)
                                   * (float(pnl_percent) / 100.0))
                trade_record.update({
                    'pnl': pnl_dollars,
                    'expected_value': stamp.get('expected_value'),
                    'probability_of_profit': stamp.get('probability_of_profit'),
                    'max_loss': stamp.get('max_loss'),
                    'ev_per_dollar_risk': stamp.get('ev_per_dollar_risk'),
                    'order_id': trade.get('order_id'),
                })
        except Exception as e:
            print(f"[EV STAMP] close flatten skipped: {e}")

        self.trading_history['trades'].append(trade_record)

        # Update ML model based on outcome
        if pnl_percent > 10:
            self.ml_model['success_patterns'].append(trade.get('metrics', {}))
        elif pnl_percent < -5:
            self.ml_model['failure_patterns'].append(trade.get('metrics', {}))

        # Adjust weights based on performance
        self.update_model_weights(pnl_percent)

        self.save_trading_history()
        self.save_ml_model()

        # Phase 4: accumulate realized dollar P/L for today's kill-switch input.
        # Side-effect only (writes realized_pnl_log.json); it never blocks a
        # trade and only feeds the switch when USE_REALIZED_PNL_KILLSWITCH is on.
        # Dollar P/L = entry notional * pnl_percent. Never raises.
        try:
            entry_price = trade.get('entry_price')
            qty = trade.get('quantity', 1) or 1
            if entry_price:
                realized_dollars = float(entry_price) * 100.0 * float(qty) * (float(pnl_percent) / 100.0)
                from realized_pnl_tracker import RealizedPnLTracker
                RealizedPnLTracker().add_realized(realized_dollars, trade.get('symbol'))
        except Exception as e:
            print(f"[REALIZED] record skipped: {e}")

        # Shadow recorder owns the close when a decision_id is present: attach
        # the NET-of-cost outcome to the same id and update the agent. No-op for
        # legacy trades without a decision_id.
        if getattr(self, 'shadow_recorder', None) and trade.get('decision_id'):
            try:
                entry_price = trade.get('entry_price')
                entry_bid = trade.get('entry_bid')
                entry_ask = trade.get('entry_ask')

                # Pull the REAL exit quote so net P/L reflects the true
                # round-trip spread instead of a price synthesized from
                # pnl_percent. Only a two-sided quote is used as bid/ask; a
                # one-sided or missing quote degrades to a mid fallback.
                exit_bid = exit_ask = exit_mid = None
                exit_quote = self.get_option_price(trade['symbol'])
                if exit_quote:
                    eb_q = exit_quote.get('bid') or 0
                    ea_q = exit_quote.get('ask') or 0
                    if eb_q > 0 and ea_q > 0:
                        exit_bid, exit_ask = eb_q, ea_q
                    exit_mid = exit_quote.get('mid')

                # Last-resort mid if no quote at all (keeps the loop closing).
                if exit_mid is None and entry_price:
                    exit_mid = entry_price * (1 + pnl_percent / 100.0)

                self.shadow_recorder.on_close(
                    trade['decision_id'],
                    entry_bid=entry_bid,
                    entry_ask=entry_ask,
                    exit_bid=exit_bid,
                    exit_ask=exit_ask,
                    entry_price=entry_price,
                    exit_price=exit_mid,
                    qty=trade.get('quantity', 1),
                    gross_pnl_pct=pnl_percent,
                    outcome=outcome,
                    closed_at=datetime.now().isoformat(),
                )
                print(f"[SHADOW] Outcome recorded (gross {pnl_percent:+.1f}%)")
            except Exception as e:
                print(f"[SHADOW] on_close failed: {e}")
        # Legacy RL hook (only when the shadow recorder is not handling the loop).
        elif getattr(self, 'rl_advisor', None) and pnl_percent:
            try:
                self.rl_advisor.record_outcome(trade.get('order_id'), pnl_percent)
                print(f"[RL] Outcome recorded: {pnl_percent:+.1f}%")
            except Exception as e:
                print(f"[RL] record_outcome failed: {e}")

    def update_model_weights(self, pnl_percent: float):
        """Update model weights based on trade performance"""
        learning_rate = 0.01

        if pnl_percent > 0:
            # Successful trade - reinforce current weights slightly
            adjustment = learning_rate * (pnl_percent / 100)
        else:
            # Failed trade - adjust weights
            adjustment = -learning_rate * (abs(pnl_percent) / 100)

        # Apply adjustments with normalization
        total = 0
        for key in self.ml_model['weights']:
            self.ml_model['weights'][key] *= (1 + adjustment)
            total += self.ml_model['weights'][key]

        # Normalize weights to sum to 1
        for key in self.ml_model['weights']:
            self.ml_model['weights'][key] /= total

    def generate_performance_report(self) -> Dict:
        """Generate performance report from trading history"""
        if not self.trading_history['trades']:
            return {'message': 'No trading history available'}

        trades = self.trading_history['trades']

        winning_trades = [t for t in trades if t['pnl_percent'] > 0]
        losing_trades = [t for t in trades if t['pnl_percent'] < 0]

        total_pnl = sum(t['pnl_percent'] for t in trades)
        avg_win = sum(t['pnl_percent'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['pnl_percent'] for t in losing_trades) / len(losing_trades) if losing_trades else 0

        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades) * 100 if trades else 0,
            'avg_win_percent': avg_win,
            'avg_loss_percent': avg_loss,
            'total_pnl_percent': total_pnl,
            'current_weights': self.ml_model['weights'],
            'patterns_learned': {
                'success': len(self.ml_model['success_patterns']),
                'failure': len(self.ml_model['failure_patterns'])
            }
        }

    def trade_symbol(self, ticker: str = None, quantity: int = None):
        """Trade specified symbol with quantity"""
        symbol = ticker or self.ticker
        qty = quantity or self.quantity

        if not symbol:
            print("[ERROR] No ticker symbol specified")
            return False

        print(f"[TRADE] Looking for {symbol} options with quantity {qty}")

        # Check options access first
        has_options_access = self.check_options_access()
        if not has_options_access:
            print(f"[WARNING] Alpaca account lacks options trading access")
            print(f"[INFO] Using simulation mode with real market analysis")

        # Get current price
        current_price = self.get_current_price(symbol)
        if not current_price:
            print(f"[ERROR] Cannot get current price for {symbol}")
            return False

        print(f"[PRICE] {symbol} current price: ${current_price:.2f}")

        # Get option chain and select best option
        contracts = self.get_option_contracts(symbol)
        if not contracts:
            print(f"[ERROR] No option contracts found for {symbol}")
            return False

        best_option = self.select_best_option(contracts, current_price)
        if not best_option:
            print(f"[ERROR] No suitable options found for {symbol}")
            return False

        print(f"[SELECTED] {best_option['symbol']} - Score: {best_option.get('score', 0):.2f}")

        # Check if this is a mock contract
        if best_option.get('mock', False):
            print(f"[SIMULATION] Mock option selected - no real order will be placed")
            print(f"[SIMULATION] To enable real options trading, upgrade to Alpaca Pro")
            return True  # Return success for simulation

        # Place real order
        order = self.place_order_with_stops(best_option, qty)
        if order:
            print(f"[SUCCESS] Real order placed for {qty} contracts of {symbol}")
            return True
        else:
            print(f"[ERROR] Failed to place order for {symbol}")
            return False

    def check_options_access(self) -> bool:
        """Check if account has options trading access"""
        try:
            response = requests.get(
                f"{self.base_url}/v2/options/contracts",
                headers=self.headers,
                params={'underlying_symbols': 'AAPL', 'limit': 1}
            )
            return response.status_code == 200
        except:
            return False

    def get_option_contracts(self, ticker: str):
        """Get option contracts for ticker including both calls and puts"""
        # First check if options are available
        if not self.check_options_access():
            print(f"[OPTIONS] Alpaca account does not have options access")
            print(f"[OPTIONS] Generating mock option contracts for {ticker}")
            return self.generate_mock_option_contracts(ticker)

        expiration_start = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
        expiration_end = (datetime.now() + timedelta(days=90)).strftime('%Y-%m-%d')

        # Bound strikes to a window around the current price. Without this the
        # endpoint returns the lowest strikes first (deep ITM), so for a
        # high-priced underlying the whole page is far-ITM contracts whose
        # premium blows the per-trade budget -> nothing qualifies. A +/-20%
        # window keeps the near-the-money (affordable) strikes for any price.
        price = self.get_current_price(ticker)
        strike_params = {}
        if price:
            strike_params['strike_price_gte'] = round(price * 0.80, 2)
            strike_params['strike_price_lte'] = round(price * 1.20, 2)

        # Get both calls and puts
        all_contracts = []

        for option_type in ['call', 'put']:
            response = requests.get(
                f"{self.base_url}/v2/options/contracts",
                headers=self.headers,
                params={
                    'underlying_symbols': ticker,
                    'expiration_date_gte': expiration_start,
                    'expiration_date_lte': expiration_end,
                    'type': option_type,
                    'limit': 100,
                    **strike_params
                }
            )

            if response.status_code == 200:
                contracts = response.json().get('option_contracts', [])
                # Add type to each contract
                for contract in contracts:
                    contract['type'] = option_type
                all_contracts.extend(contracts)

        if not all_contracts:
            print(f"[OPTIONS] No real contracts found, using mock contracts for {ticker}")
            return self.generate_mock_option_contracts(ticker)

        return all_contracts

    # ----------------------------------------------------------------------- #
    # Phase 6A: defined-risk spread PROPOSALS (simulation only, never trades)
    # ----------------------------------------------------------------------- #
    def _spread_leg_from_contract(self, contract: Dict, action: str):
        """Build a spread_builder.SpreadLeg from a chain contract + a live quote.

        Pulls bid/ask from the options-quotes endpoint; a missing/zero quote is
        passed through as None so the builder's `missing_quote` gate fires. OI /
        volume are taken from the contract when present (else None -> liquidity
        gate fails open). No orders, ever.
        """
        from spread_builder import SpreadLeg

        def _f(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        sym = contract.get('symbol', '')
        quote = self.get_option_price(sym) or {}
        bid = quote.get('bid')
        ask = quote.get('ask')
        return SpreadLeg(
            action=action, option_type=contract.get('type', ''),
            strike=_f(contract.get('strike_price')) or 0.0,
            bid=bid if (bid and bid > 0) else None,
            ask=ask if (ask and ask > 0) else None,
            symbol=sym, expiration=contract.get('expiration_date', ''),
            open_interest=_f(contract.get('open_interest')),
            volume=_f(contract.get('volume')),
        )

    def propose_spread(self, ticker: str = None, config=None):
        """Build the best defined-risk spread PROPOSAL for ``ticker`` (no orders).

        Fully fail-open: any error/empty data -> a `no_trade` proposal whose
        reason names the cause. Steps:
          1. price + IV/HV -> volatility state; momentum -> trend.
          2. select_spread_strategy(vol_state, trend) -> strategy (or no_trade).
          3. snap target strikes to the nearest available, fetch per-leg quotes.
          4. spread_builder.build_spread applies all hard safety rejections.
          5. log the [SPREAD_PROPOSAL] block and return the proposal object.

        This method NEVER submits an order; it only returns a proposal object.
        """
        from spread_builder import (
            SpreadConfig, build_spread, classify_volatility, classify_trend,
            select_spread_strategy, format_proposal_log, no_trade_proposal,
            compute_oracle_score, quality_check, map_reason_to_quality,
            BULLISH_PUT_CREDIT_SPREAD, BEARISH_CALL_CREDIT_SPREAD,
            DEBIT_CALL_SPREAD, DEBIT_PUT_SPREAD, IRON_CONDOR, NO_TRADE,
            REASON_MISSING_CHAIN, REASON_WEAK_VOL_EDGE,
        )

        symbol = (ticker or self.ticker or '').upper()
        cfg = config or SpreadConfig.from_env()

        try:
            if not symbol:
                return no_trade_proposal("no_symbol", symbol)

            price = self.get_current_price(symbol)
            if not price or price <= 0:
                return no_trade_proposal("no_underlying_price", symbol)

            # 1) Signals: IV/HV -> vol state, momentum -> trend.
            hv = self.calculate_volatility(symbol)
            momentum = self.calculate_momentum(symbol)
            trend = classify_trend(momentum, cfg)

            contracts = self.get_option_contracts(symbol) or []
            if not contracts:
                return no_trade_proposal(REASON_MISSING_CHAIN, symbol)

            # Use the nearest available expiration as the structure's expiry.
            expirations = sorted({c.get('expiration_date') for c in contracts
                                  if c.get('expiration_date')})
            if not expirations:
                return no_trade_proposal(REASON_MISSING_CHAIN, symbol)
            target_exp = expirations[0]
            chain = [c for c in contracts if c.get('expiration_date') == target_exp]

            calls = {}
            puts = {}
            for c in chain:
                try:
                    k = float(c.get('strike_price'))
                except (TypeError, ValueError):
                    continue
                (calls if c.get('type') == 'call' else puts)[k] = c
            call_strikes = sorted(calls)
            put_strikes = sorted(puts)

            # IV from an ATM call snapshot (fail-open: None -> vol_state unknown).
            iv = None
            if call_strikes:
                atm = min(call_strikes, key=lambda s: abs(s - price))
                iv = (self.get_option_snapshot(calls[atm].get('symbol', '')) or {}).get('iv')
            vol_state = classify_volatility(iv, hv, cfg)

            # 2) Strategy selection (Requirement 4).
            strategy = select_spread_strategy(vol_state, trend)
            print(f"[SPREAD] {symbol} price={price:.2f} iv={iv} hv={hv:.3f} "
                  f"vol_state={vol_state} momentum={momentum:.4f} trend={trend} "
                  f"-> strategy={strategy}")
            if strategy == NO_TRADE:
                return no_trade_proposal(
                    f"{REASON_WEAK_VOL_EDGE} vol={vol_state} trend={trend}", symbol)

            # 3) Snap target strikes to nearest available, then build legs.
            def near(strikes, target):
                return min(strikes, key=lambda s: abs(s - target))

            W = cfg.wing_width
            legs = []
            if strategy == BULLISH_PUT_CREDIT_SPREAD:
                if len(put_strikes) < 2:
                    return no_trade_proposal(REASON_MISSING_CHAIN, symbol)
                short_k = near(put_strikes, price * 0.98)
                long_k = near(put_strikes, short_k - W)
                if long_k >= short_k:
                    return no_trade_proposal(REASON_MISSING_CHAIN, symbol)
                legs = [self._spread_leg_from_contract(puts[short_k], 'sell'),
                        self._spread_leg_from_contract(puts[long_k], 'buy')]
            elif strategy == BEARISH_CALL_CREDIT_SPREAD:
                if len(call_strikes) < 2:
                    return no_trade_proposal(REASON_MISSING_CHAIN, symbol)
                short_k = near(call_strikes, price * 1.02)
                long_k = near(call_strikes, short_k + W)
                if long_k <= short_k:
                    return no_trade_proposal(REASON_MISSING_CHAIN, symbol)
                legs = [self._spread_leg_from_contract(calls[short_k], 'sell'),
                        self._spread_leg_from_contract(calls[long_k], 'buy')]
            elif strategy == DEBIT_CALL_SPREAD:
                if len(call_strikes) < 2:
                    return no_trade_proposal(REASON_MISSING_CHAIN, symbol)
                long_k = near(call_strikes, price)
                short_k = near(call_strikes, long_k + W)
                if short_k <= long_k:
                    return no_trade_proposal(REASON_MISSING_CHAIN, symbol)
                legs = [self._spread_leg_from_contract(calls[long_k], 'buy'),
                        self._spread_leg_from_contract(calls[short_k], 'sell')]
            elif strategy == DEBIT_PUT_SPREAD:
                if len(put_strikes) < 2:
                    return no_trade_proposal(REASON_MISSING_CHAIN, symbol)
                long_k = near(put_strikes, price)
                short_k = near(put_strikes, long_k - W)
                if short_k >= long_k:
                    return no_trade_proposal(REASON_MISSING_CHAIN, symbol)
                legs = [self._spread_leg_from_contract(puts[long_k], 'buy'),
                        self._spread_leg_from_contract(puts[short_k], 'sell')]
            elif strategy == IRON_CONDOR:
                if len(put_strikes) < 2 or len(call_strikes) < 2:
                    return no_trade_proposal(REASON_MISSING_CHAIN, symbol)
                short_put_k = near(put_strikes, price * 0.97)
                long_put_k = near(put_strikes, short_put_k - W)
                short_call_k = near(call_strikes, price * 1.03)
                long_call_k = near(call_strikes, short_call_k + W)
                if not (long_put_k < short_put_k < short_call_k < long_call_k):
                    return no_trade_proposal(REASON_MISSING_CHAIN, symbol)
                legs = [self._spread_leg_from_contract(puts[long_put_k], 'buy'),
                        self._spread_leg_from_contract(puts[short_put_k], 'sell'),
                        self._spread_leg_from_contract(calls[short_call_k], 'sell'),
                        self._spread_leg_from_contract(calls[long_call_k], 'buy')]
            else:
                return no_trade_proposal("unknown_strategy", symbol)

            # 4) Build + validate (all hard safety gates live in the builder).
            proposal = build_spread(strategy, legs, cfg, symbol)

            if proposal.is_tradeable:
                # Recompute the oracle_score with the MEASURED vol_state/trend
                # (the builder's default score assumes only selection-alignment).
                proposal.oracle_score = compute_oracle_score(
                    proposal, cfg, vol_state=vol_state, trend=trend)
                # Phase 6B quality floors (no-ops unless configured).
                q = quality_check(proposal, cfg)
                if q:
                    proposal = no_trade_proposal(q, symbol, proposal.legs)
            else:
                # Surface a Phase-6B quality reason for the hard-safety rejection.
                proposal.reason = map_reason_to_quality(proposal.reason)

            # 5) Log the required [SPREAD_PROPOSAL] block.
            print(format_proposal_log(proposal))
            return proposal
        except Exception as e:
            print(f"[SPREAD] propose_spread error (ignored): {e}")
            return no_trade_proposal(f"error:{type(e).__name__}", symbol)

    def calculate_black_scholes_price(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
        """Calculate option price using Black-Scholes model"""
        from math import log, sqrt, exp
        from scipy.stats import norm

        # Black-Scholes formula
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        if option_type.lower() == 'call':
            price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return max(price, 0.01)  # Minimum price of $0.01

    def generate_mock_option_contracts(self, ticker: str):
        """Generate realistic mock option contracts with Black-Scholes pricing"""
        current_price = self.get_current_price(ticker)
        if not current_price:
            return []

        # Calculate volatility for more accurate pricing
        volatility = self.calculate_volatility(ticker)

        # Risk-free rate (approximate current rate)
        risk_free_rate = 0.045  # 4.5% annual rate

        # Generate realistic strikes around current price
        contracts = []
        base_date = datetime.now() + timedelta(days=45)  # ~6 weeks out
        exp_date = base_date.strftime('%Y-%m-%d')
        days_to_exp = 45
        time_to_exp = days_to_exp / 365.0  # Convert to years

        # Generate ITM and OTM strikes
        strikes = [
            current_price * 0.95,  # 5% ITM call
            current_price * 0.97,  # 3% ITM call
            current_price * 1.03,  # 3% OTM call
            current_price * 1.05,  # 5% OTM call
        ]

        for i, strike in enumerate(strikes):
            strike_rounded = round(strike, 2)

            # Calculate realistic prices using Black-Scholes
            call_price = self.calculate_black_scholes_price(
                current_price, strike_rounded, time_to_exp, risk_free_rate, volatility, 'call'
            )
            put_price = self.calculate_black_scholes_price(
                current_price, strike_rounded, time_to_exp, risk_free_rate, volatility, 'put'
            )

            contracts.extend([
                {
                    'symbol': f'{ticker}{base_date.strftime("%y%m%d")}C{int(strike_rounded * 1000):08d}',
                    'strike_price': str(strike_rounded),
                    'expiration_date': exp_date,
                    'type': 'call',
                    'mock': True,
                    'mock_bid': call_price * 0.98,  # Slightly below mid
                    'mock_ask': call_price * 1.02   # Slightly above mid
                },
                {
                    'symbol': f'{ticker}{base_date.strftime("%y%m%d")}P{int(strike_rounded * 1000):08d}',
                    'strike_price': str(strike_rounded),
                    'expiration_date': exp_date,
                    'type': 'put',
                    'mock': True,
                    'mock_bid': put_price * 0.98,
                    'mock_ask': put_price * 1.02
                }
            ])

        return contracts


def main():
    """Command line interface for options trading"""
    parser = argparse.ArgumentParser(description='Smart Options Trader - Trade any symbol with specified quantity')
    parser.add_argument('command', help='Command: SYMBOL QUANTITY (e.g., "IREN 5", "AAPL 3")')
    parser.add_argument('--monitor', action='store_true', help='Monitor existing positions')
    parser.add_argument('--status', action='store_true', help='Show performance status')
    parser.add_argument('--continuous', action='store_true', help='Run continuous monitoring')

    args = parser.parse_args()

    # Parse command for symbol and quantity
    parts = args.command.upper().split()

    if len(parts) == 2:
        symbol, quantity_str = parts
        try:
            quantity = int(quantity_str)
        except ValueError:
            print(f"[ERROR] Invalid quantity: {quantity_str}")
            return 1
    else:
        print(f"[ERROR] Invalid command format. Use: SYMBOL QUANTITY (e.g., 'IREN 5')")
        return 1

    # Initialize trader
    trader = SmartOptionsTrader(ticker=symbol, quantity=quantity)

    # Reconcile any open positions / pending trades left over from a prior run
    # before doing anything else (trade, monitor, or status).
    trader.reconcile_open_trades()

    if args.status:
        report = trader.generate_performance_report()
        print("\n=== PERFORMANCE REPORT ===")
        for key, value in report.items():
            print(f"{key}: {value}")
        return 0

    if args.monitor:
        print(f"[MONITOR] Checking existing positions...")
        trader.monitor_positions()
        return 0

    if args.continuous:
        print(f"[CONTINUOUS] Starting continuous trading for {symbol} with quantity {quantity}")
        import time
        while True:
            try:
                trader.monitor_positions()
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                print("\n[STOP] Continuous monitoring stopped")
                break
        return 0

    # Execute single trade
    print(f"[START] Smart Options Trader")
    print(f"[TARGET] {symbol} with quantity {quantity}")

    success = trader.trade_symbol()

    if success:
        print(f"[COMPLETE] Trade executed successfully")
        return 0
    else:
        print(f"[FAILED] Trade execution failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())