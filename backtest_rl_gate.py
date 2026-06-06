"""
Walk-forward backtest of the conservative RL veto-only gate.

Compares the BASELINE rule-based SPY 1DTE strategy against the GATED version
(rule strategy + RL veto) on real market data, with NO Schwab dependency:

  * SPY daily OHLC bars come from Alpaca market data (reuses .env creds).
  * VIX comes from stooq (free CSV) and/or a realized-volatility proxy.
  * vix_change and intraday_position are computed from REAL data (the older
    enhanced backtest faked vix_change with random.uniform and never computed
    intraday_position).

The gate is veto-only: it can turn a rule-chosen CALL/PUT into a SKIP, but it
never flips direction and never mutates the live Q-table (an isolated temp
table is used). It learns full-info from every tradeable day's realized P/L so
the regime memory fills in; gate decisions on day T use only what was learned
from days < T (honest walk-forward).

Usage:
    python backtest_rl_gate.py --selftest
    python backtest_rl_gate.py --start 2026-01-01 --vix-source both --report
    python backtest_rl_gate.py --start 2026-01-01 --epochs 25 \
        --min-visits 3 --min-confidence 0.5
"""

import os
import csv
import io
import math
import random
import argparse
import tempfile
from datetime import datetime, timedelta

import requests

from rl_agent import QLearningAgent
from rl_wrapper import RLAdvisor, _gate_config
from rl_env import extract_features, state_key, compute_reward
from features import compute_features
from market_view import HistoricalMarketView, make_bar
from cost_model import CostModel, load_cost_config_from_env

MIN_CONFIDENCE = 70.0          # rule confidence floor (matches enhanced backtest)
PROFIT_TARGET_PCT = 20.0
ANNUALIZER = math.sqrt(252.0)  # daily -> annualized vol
SPREAD_FRAC = 0.06             # synthesized bid/ask spread as a fraction of premium
HOLD_DAYS = 1                  # decide at close of T, exit on T+1


# --------------------------------------------------------------------------- #
# Config / credentials (manual .env parse, matching the rest of the project)
# --------------------------------------------------------------------------- #
def _load_env():
    env = {}
    if os.path.exists(".env"):
        try:
            with open(".env", "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        env[k.strip()] = v.strip()
        except OSError:
            pass
    return env


def _alpaca_ctx():
    env = _load_env()
    return {
        "data_url": "https://data.alpaca.markets",
        "headers": {
            "APCA-API-KEY-ID": env.get("ALPACA_API_KEY", ""),
            "APCA-API-SECRET-KEY": env.get("ALPACA_SECRET_KEY", ""),
        },
        "feed": env.get("SCREENER_ALPACA_FEED", "iex"),
    }


# --------------------------------------------------------------------------- #
# Data: SPY daily bars (Alpaca) + VIX (stooq / realized-vol proxy)
# --------------------------------------------------------------------------- #
def fetch_spy_bars(start, end):
    """Daily SPY OHLC bars as a list of {date, o, h, l, c} sorted ascending."""
    ctx = _alpaca_ctx()
    bars = []
    page_token = None
    while True:
        params = {
            "timeframe": "1Day",
            "start": start,
            "end": end,
            "limit": 10000,
            "feed": ctx["feed"],
            "adjustment": "raw",
        }
        if page_token:
            params["page_token"] = page_token
        try:
            resp = requests.get(
                f"{ctx['data_url']}/v2/stocks/SPY/bars",
                headers=ctx["headers"],
                params=params,
                timeout=30,
            )
        except requests.RequestException as e:
            print(f"[DATA] SPY bars request failed: {e}")
            break
        if resp.status_code != 200:
            print(f"[DATA] SPY bars status {resp.status_code}: {resp.text[:200]}")
            break
        payload = resp.json()
        for b in payload.get("bars", []) or []:
            bars.append(
                {
                    "date": b["t"][:10],
                    "o": float(b["o"]),
                    "h": float(b["h"]),
                    "l": float(b["l"]),
                    "c": float(b["c"]),
                }
            )
        page_token = payload.get("next_page_token")
        if not page_token:
            break
    bars.sort(key=lambda x: x["date"])
    return bars


def fetch_vix_stooq(start, end):
    """Daily VIX close keyed by date string from stooq's free CSV endpoint."""
    d1 = start.replace("-", "")
    d2 = end.replace("-", "")
    url = f"https://stooq.com/q/d/l/?s=^vix&d1={d1}&d2={d2}&i=d"
    out = {}
    try:
        resp = requests.get(url, timeout=30)
    except requests.RequestException as e:
        print(f"[DATA] stooq VIX request failed: {e}")
        return out
    if resp.status_code != 200 or not resp.text.strip():
        print(f"[DATA] stooq VIX status {resp.status_code}")
        return out
    reader = csv.DictReader(io.StringIO(resp.text))
    for row in reader:
        try:
            out[row["Date"]] = float(row["Close"])
        except (KeyError, ValueError, TypeError):
            continue
    return out


def proxy_vix_from_spy(bars, window=20):
    """Realized-volatility VIX proxy: annualized rolling std of SPY returns x100."""
    out = {}
    rets = []
    for i, b in enumerate(bars):
        if i == 0:
            rets.append(0.0)
        else:
            prev = bars[i - 1]["c"]
            rets.append((b["c"] - prev) / prev if prev else 0.0)
        win = rets[max(0, i - window + 1): i + 1]
        if len(win) >= 2:
            mean = sum(win) / len(win)
            var = sum((r - mean) ** 2 for r in win) / (len(win) - 1)
            out[b["date"]] = math.sqrt(var) * ANNUALIZER * 100.0
        else:
            out[b["date"]] = 15.0
    return out


def build_vix_series(bars, source):
    """Return {date: vix} for the given source ('stooq' or 'proxy')."""
    if not bars:
        return {}
    if source == "proxy":
        return proxy_vix_from_spy(bars)
    # stooq, with proxy fallback for any missing dates
    start, end = bars[0]["date"], bars[-1]["date"]
    stooq = fetch_vix_stooq(start, end)
    proxy = proxy_vix_from_spy(bars)
    series = {}
    for b in bars:
        d = b["date"]
        series[d] = stooq.get(d, proxy.get(d, 15.0))
    return series


# --------------------------------------------------------------------------- #
# Strategy logic (Schwab-free reimplementation of the enhanced 1DTE rules)
# --------------------------------------------------------------------------- #
def analyze_features(raw):
    """Rule scoring on the shared NO-LOOKAHEAD feature block (replaces the old
    analyze_day).

    The momentum signal is `spy_change` (the COMPLETED day's close-vs-open move,
    legitimately known at that session's close) instead of the old
    `first_30min_move`, which read the day's high/low and was therefore
    lookahead. The decision is made at the close of day T; the trade outcome
    belongs to the next session T+1. `raw` is the `compute_features(...)["raw"]`
    block, so backtest and live share one feature path (train-serve skew = 0).
    """
    spy_change = raw.get("spy_change", 0.0)
    gap = raw.get("gap", 0.0)
    vix_level = raw.get("vix_level", 15.0)
    vix_change = raw.get("vix_change", 0.0)
    intraday_position = raw.get("intraday_position", 0.5)

    bullish = 0
    bearish = 0
    skip_reasons = []

    if vix_level > 30:
        skip_reasons.append(f"VIX too high ({vix_level:.1f})")
    if abs(gap) > 1.0:
        skip_reasons.append(f"Large gap ({gap:+.2f}%)")

    # Signal 1: completed-day momentum (no lookahead)
    if spy_change > 0.3:
        bullish += 2
    elif spy_change > 0.1:
        bullish += 1
    elif spy_change < -0.3:
        bearish += 2
    elif spy_change < -0.1:
        bearish += 1

    # Signal 2: VIX level
    if vix_level > 25:
        bearish += 1
    elif vix_level < 15:
        bullish += 1

    # Signal 3: moderate gap
    if 0.3 < gap < 1.0:
        bullish += 1
    elif -1.0 < gap < -0.3:
        bearish += 1

    # Signal 4: VIX direction (REAL, not random)
    if vix_change < -5:
        bullish += 1
    elif vix_change > 5:
        bearish += 1

    total = bullish + bearish
    if total == 0:
        direction, confidence = None, 0.0
    elif bullish > bearish:
        direction, confidence = "CALL", (bullish / total) * 100
    elif bearish > bullish:
        direction, confidence = "PUT", (bearish / total) * 100
    else:
        direction = "CALL" if spy_change >= 0 else "PUT"
        confidence = 50.0

    should_trade = not skip_reasons and confidence >= MIN_CONFIDENCE

    return {
        "direction": direction,
        "confidence": confidence,
        "spy_change": spy_change,
        "gap": gap,
        "vix_level": vix_level,
        "vix_change": vix_change,
        "intraday_position": intraday_position,
        "should_trade": should_trade,
        "skip_reasons": skip_reasons,
    }


def simulate_trade(direction, bar, rng):
    """Seeded reimplementation of simulate_option_trade_enhanced.

    Returns (gross_profit_pct, entry_premium). The gross figure is the option's
    premium move before costs; the entry premium is returned so the shared cost
    model can convert it to a NET number against a synthesized bid/ask spread.
    """
    entry_premium = rng.uniform(0.60, 1.20)
    spy_open, spy_close = bar["o"], bar["c"]

    total_checks = 6 * 4  # 10:00-16:00, every 15 min
    intraday = []
    for i in range(total_checks):
        progress = (i + 1) / total_checks
        target = spy_open + (spy_close - spy_open) * progress
        noise = rng.uniform(-0.001, 0.001) * spy_open
        intraday.append(target + noise)

    delta_effect, gamma_effect = 0.375, 0.15
    max_profit_pct = 0.0
    current_premium = entry_premium

    for i, spy_price in enumerate(intraday):
        minutes = i * 15
        hour = 10 + (minutes // 60)
        spy_move_pct = ((spy_price - spy_open) / spy_open) * 100 if spy_open else 0.0
        if direction == "CALL":
            opt_move = spy_move_pct * (delta_effect + gamma_effect)
        else:
            opt_move = -spy_move_pct * (delta_effect + gamma_effect)
        opt_move += -0.02 * (minutes / 60)  # theta decay
        current_premium = max(entry_premium * (1 + opt_move / 100), 0.01)
        profit_pct = ((current_premium - entry_premium) / entry_premium) * 100
        max_profit_pct = max(max_profit_pct, profit_pct)

        if profit_pct >= 20:
            return 20.0, entry_premium
        if max_profit_pct >= 15 and profit_pct <= max_profit_pct - 10:
            return profit_pct, entry_premium
        if hour < 11 and profit_pct <= -20:
            return -20.0, entry_premium
        if profit_pct <= -30:
            return -30.0, entry_premium

    return ((current_premium - entry_premium) / entry_premium) * 100, entry_premium


def _seeded_rng(date_str):
    """Deterministic RNG per trading day so baseline & gated see identical P/L."""
    return random.Random(int.from_bytes(date_str.encode(), "big") % (2 ** 31))


def realized_net_detail(direction, outcome_bar, date_str, cost_model):
    """Gross simulation on the OUTCOME bar (T+1) converted to NET via a
    synthesized bid/ask spread (SPREAD_FRAC of premium) run through the shared
    cost model. Returns a dict carrying both gross and net plus the fills used
    so an episode row can be written."""
    gross, entry_premium = simulate_trade(direction, outcome_bar, _seeded_rng(date_str))
    half = max(entry_premium * SPREAD_FRAC / 2.0, 0.0)
    entry_bid, entry_ask = max(0.0, entry_premium - half), entry_premium + half
    exit_mid = max(entry_premium * (1 + gross / 100.0), 0.0)
    exit_bid, exit_ask = max(0.0, exit_mid - half), exit_mid + half
    res = cost_model.net_pnl(entry_bid, entry_ask, exit_bid, exit_ask,
                             qty=1, hold_days=HOLD_DAYS)
    return {
        "gross_pct": gross,
        "net_pct": res["net_pnl_pct"],
        "net_dollars": res["net_pnl_dollars"],
        "entry_bid": entry_bid,
        "entry_ask": entry_ask,
        "entry_price": res["entry_price"],
        "exit_price": res["exit_price"],
    }


def realized_net(direction, outcome_bar, date_str, cost_model):
    """NET P/L % for the trade (costs included). See realized_net_detail."""
    return realized_net_detail(direction, outcome_bar, date_str, cost_model)["net_pct"]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _max_drawdown(equity):
    peak = equity[0] if equity else 0.0
    mdd = 0.0
    for v in equity:
        peak = max(peak, v)
        mdd = max(mdd, peak - v)
    return mdd


def _summarize(label, pnls, start_equity=500.0):
    wins = [p for p in pnls if p > 0]
    equity = [start_equity]
    for p in pnls:
        # P/L% is on premium; scale to a notional $100 contract for an equity curve.
        equity.append(equity[-1] + p)
    total = sum(pnls)
    n = len(pnls)
    win_rate = (len(wins) / n * 100) if n else 0.0
    expectancy = (total / n) if n else 0.0
    return {
        "label": label,
        "trades": n,
        "win_rate": win_rate,
        "total_pnl_pct": total,
        "expectancy_pct": expectancy,
        "effective_sample_size": float(n),
        "final_equity": equity[-1],
        "max_drawdown": _max_drawdown(equity),
    }


# --------------------------------------------------------------------------- #
# Walk-forward engine
# --------------------------------------------------------------------------- #
def _build_view(bars, vix, idx):
    """HistoricalMarketView pinned to the CLOSE of bars[idx], carrying only data
    knowable by then (bars[:idx+1]). Used so features carry no lookahead."""
    decision_bar = bars[idx]
    as_of = datetime.strptime(decision_bar["date"], "%Y-%m-%d").replace(hour=16, minute=0)
    daily = {
        "SPY": [make_bar(b["date"], b["o"], b["h"], b["l"], b["c"]) for b in bars[: idx + 1]]
    }
    vbars = []
    for b in bars[: idx + 1]:
        v = vix.get(b["date"], 15.0)
        vbars.append(make_bar(b["date"], v, v, v, v))
    mv = HistoricalMarketView(as_of, daily=daily, vix_series={"^VIX": vbars})
    return mv, as_of


def _tradeable_days(bars, vix):
    """Yield (decision_bar, outcome_bar, analysis, features) with no lookahead.

    Decide at the CLOSE of day T (all of day T's OHLC is legitimately known by
    then); the trade outcome belongs to the NEXT session T+1. Features come from
    the shared `compute_features` path over a HistoricalMarketView pinned to
    close(T), so there is no high/low lookahead and no train-serve skew.
    """
    for i in range(1, len(bars) - 1):
        decision_bar = bars[i]
        outcome_bar = bars[i + 1]
        try:
            dow = datetime.strptime(decision_bar["date"], "%Y-%m-%d").weekday()
        except ValueError:
            continue
        if dow >= 5:
            continue
        mv, as_of = _build_view(bars, vix, i)
        feats = compute_features(as_of, mv, symbol="SPY", strat_name="spy_1dte")
        analysis = analyze_features(feats["raw"])
        yield decision_bar, outcome_bar, analysis, feats


def _write_episode(store, decision_bar, outcome_bar, analysis, feats, gate, detail):
    """Persist one decision+outcome under a single decision_id (NET P/L)."""
    did = store.log_decision(
        symbol="SPY",
        underlying="SPY",
        strat="spy_1dte",
        features=feats,
        quote={"bid": detail["entry_bid"], "ask": detail["entry_ask"],
               "ts": decision_bar["date"]},
        modeled_cost={"spread_frac": SPREAD_FRAC, "hold_days": HOLD_DAYS},
        rule_action=analysis["direction"] or "SKIP",
        rule_confidence=analysis["confidence"],
        gate=gate,
        chosen_action=analysis["direction"] or "SKIP",
        qty=1,
        mode="1DTE",
        as_of=feats.get("as_of"),
    )
    store.record_outcome(
        did,
        fill_price=detail["entry_price"],
        exit_price=detail["exit_price"],
        gross_pnl_pct=detail["gross_pct"],
        net_pnl_pct=detail["net_pct"],
        net_pnl_dollars=detail["net_dollars"],
        hold_days=HOLD_DAYS,
        outcome="closed",
        closed_at=outcome_bar["date"],
    )
    return did


def run_walkforward(bars, vix, overrides, learn=True, use_partial_table=True,
                    cost_model=None, episode_store=None):
    """One sequential pass.

    Returns (baseline_pnls, gated_pnls, vetoed_states) as NET-of-cost P/L. When
    use_partial_table is True (honest walk-forward) gate decisions use only what
    was learned from earlier days, because learning happens AFTER each day's
    decision. If `episode_store` is given, each decision+outcome is persisted
    under one decision_id.
    """
    cost_model = cost_model or CostModel(load_cost_config_from_env())
    os.environ["RL_MODE"] = "gate"  # backtest forces gate evaluation
    tmp = tempfile.NamedTemporaryFile(
        prefix="rl_qtable_bt_", suffix=".json", delete=False
    )
    tmp.close()
    advisor = RLAdvisor(
        strat_name="spy_1dte",
        experience_file=tmp.name + ".exp",
        qtable_file=tmp.name,
    )
    advisor.agent.reset()

    baseline_pnls, gated_pnls, vetoed = [], [], []

    for decision_bar, outcome_bar, analysis, feats in _tradeable_days(bars, vix):
        direction = analysis["direction"]
        wants_trade = analysis["should_trade"] and direction in ("CALL", "PUT")

        if not wants_trade:
            continue  # both baseline and gated skip; nothing to compare/learn

        detail = realized_net_detail(direction, outcome_bar, outcome_bar["date"], cost_model)
        net = detail["net_pct"]
        baseline_pnls.append(net)

        gate = advisor.gate_decision(analysis, overrides=overrides)
        if gate["veto"]:
            vetoed.append(
                {
                    "date": decision_bar["date"],
                    "state_key": gate["state_key"],
                    "direction": direction,
                    "q": gate["q"],
                    "visits": gate["visits"],
                    "baseline_pnl": net,
                }
            )
            # gated skips -> realizes 0
        else:
            gated_pnls.append(net)

        if episode_store is not None:
            _write_episode(episode_store, decision_bar, outcome_bar, analysis, feats, gate, detail)

        if learn:
            reward = compute_reward(net, direction)
            skey = state_key(extract_features(analysis, None, None, "spy_1dte"))
            advisor.agent.update(skey, direction, reward, done=True)

    try:
        os.remove(tmp.name)
        os.remove(tmp.name + ".exp")
    except OSError:
        pass

    return baseline_pnls, gated_pnls, vetoed


def train_then_eval(bars, vix, overrides, epochs, cost_model=None):
    """Thicken the Q-table over `epochs` passes, then evaluate the gate in a
    final no-learning pass (in-sample demonstration when data is thin). NET P/L.
    """
    cost_model = cost_model or CostModel(load_cost_config_from_env())
    os.environ["RL_MODE"] = "gate"
    tmp = tempfile.NamedTemporaryFile(
        prefix="rl_qtable_bt_", suffix=".json", delete=False
    )
    tmp.close()
    advisor = RLAdvisor(
        strat_name="spy_1dte",
        experience_file=tmp.name + ".exp",
        qtable_file=tmp.name,
    )
    advisor.agent.reset()

    days = list(_tradeable_days(bars, vix))
    trade_days = [
        (outcome_bar, a) for (_d, outcome_bar, a, _f) in days
        if a["should_trade"] and a["direction"] in ("CALL", "PUT")
    ]

    for _ in range(max(1, epochs)):
        for outcome_bar, analysis in trade_days:
            net = realized_net(analysis["direction"], outcome_bar, outcome_bar["date"], cost_model)
            reward = compute_reward(net, analysis["direction"])
            skey = state_key(extract_features(analysis, None, None, "spy_1dte"))
            advisor.agent.update(skey, analysis["direction"], reward, done=True)

    baseline_pnls, gated_pnls, vetoed = [], [], []
    for outcome_bar, analysis in trade_days:
        direction = analysis["direction"]
        net = realized_net(direction, outcome_bar, outcome_bar["date"], cost_model)
        baseline_pnls.append(net)
        gate = advisor.gate_decision(analysis, overrides=overrides)
        if gate["veto"]:
            vetoed.append(
                {
                    "date": outcome_bar["date"],
                    "state_key": gate["state_key"],
                    "direction": direction,
                    "q": gate["q"],
                    "visits": gate["visits"],
                    "baseline_pnl": net,
                }
            )
        else:
            gated_pnls.append(net)

    try:
        os.remove(tmp.name)
        os.remove(tmp.name + ".exp")
    except OSError:
        pass

    return baseline_pnls, gated_pnls, vetoed


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _print_comparison(title, baseline_pnls, gated_pnls, vetoed):
    base = _summarize("BASELINE", baseline_pnls)
    gated = _summarize("GATED", gated_pnls)
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)
    hdr = f"{'metric':<18}{'BASELINE':>14}{'GATED':>14}"
    print(hdr)
    print("-" * len(hdr))
    print(f"{'trades':<18}{base['trades']:>14d}{gated['trades']:>14d}")
    print(f"{'vetoed':<18}{'-':>14}{len(vetoed):>14d}")
    print(f"{'win_rate %':<18}{base['win_rate']:>14.1f}{gated['win_rate']:>14.1f}")
    print(f"{'net total P/L %':<18}{base['total_pnl_pct']:>14.1f}"
          f"{gated['total_pnl_pct']:>14.1f}")
    print(f"{'net expectancy %':<18}{base['expectancy_pct']:>14.2f}"
          f"{gated['expectancy_pct']:>14.2f}")
    print(f"{'eff sample size':<18}{base['effective_sample_size']:>14.1f}"
          f"{gated['effective_sample_size']:>14.1f}")
    print(f"{'final equity':<18}{base['final_equity']:>14.1f}"
          f"{gated['final_equity']:>14.1f}")
    print(f"{'max drawdown':<18}{base['max_drawdown']:>14.1f}"
          f"{gated['max_drawdown']:>14.1f}")
    if vetoed:
        print(f"\nVetoed setups ({len(vetoed)}):")
        for v in vetoed:
            print(f"  {v['date']} {v['direction']:<4} q={v['q']:+.4f} "
                  f"visits={v['visits']:<3} basePnL={v['baseline_pnl']:+6.1f}%  "
                  f"{v['state_key']}")
    else:
        print("\nNo vetoes fired (insufficient negative evidence under thresholds).")


def run_for_source(bars, source, overrides, epochs, cost_model=None, episode_store=None):
    vix = build_vix_series(bars, source)
    print(f"\n########## VIX SOURCE: {source.upper()} "
          f"({len([d for d in vix])} days) ##########")

    b, g, vetoed = run_walkforward(
        bars, vix, overrides, learn=True,
        cost_model=cost_model, episode_store=episode_store,
    )
    _print_comparison(
        f"HONEST WALK-FORWARD (decide-then-learn, NET of cost)  [vix={source}]",
        b, g, vetoed,
    )

    if epochs > 1:
        b2, g2, vetoed2 = train_then_eval(bars, vix, overrides, epochs, cost_model=cost_model)
        _print_comparison(
            f"IN-SAMPLE DEMO (table trained {epochs} epochs, NET)  [vix={source}]",
            b2, g2, vetoed2,
        )


def run_backtest(args):
    overrides = {
        "min_visits": args.min_visits,
        "max_q": args.max_q,
        "min_confidence": args.min_confidence,
    }
    print("=" * 64)
    print("RL VETO-GATE WALK-FORWARD BACKTEST")
    print("=" * 64)
    print(f"Period: {args.start} -> {args.end}")
    print(f"Gate thresholds: min_visits={overrides['min_visits']} "
          f"max_q={overrides['max_q']} min_confidence={overrides['min_confidence']}")

    bars = fetch_spy_bars(args.start, args.end)
    print(f"[DATA] Retrieved {len(bars)} SPY daily bars")
    if len(bars) < 2:
        print("[ERROR] Not enough SPY data to backtest.")
        return 1

    cost_model = CostModel(load_cost_config_from_env())

    episode_store = None
    if getattr(args, "write_episodes", False):
        from episode_store import EpisodeStore
        # Separate DB so backtest rows never collide with shadow/live episodes.
        episode_store = EpisodeStore("episodes_backtest.db")
        print("[EPISODES] Writing decision+outcome rows to episodes_backtest.db")

    try:
        sources = ["stooq", "proxy"] if args.vix_source == "both" else [args.vix_source]
        for src in sources:
            run_for_source(bars, src, overrides, args.epochs,
                           cost_model=cost_model, episode_store=episode_store)
    finally:
        if episode_store is not None:
            print(f"[EPISODES] stats: {episode_store.stats()}")
            episode_store.close()
    return 0


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network)
# --------------------------------------------------------------------------- #
def _self_test():
    print("=" * 50)
    print("RL GATE SELF-TEST")
    print("=" * 50)
    os.environ["RL_MODE"] = "gate"
    tmp = tempfile.NamedTemporaryFile(
        prefix="rl_qtable_gatetest_", suffix=".json", delete=False
    )
    tmp.close()
    advisor = RLAdvisor(
        strat_name="spy_1dte",
        experience_file=tmp.name + ".exp",
        qtable_file=tmp.name,
    )
    advisor.agent.reset()

    overrides = {"min_visits": 5, "max_q": -0.10, "min_confidence": 0.75}
    analysis = {
        "direction": "CALL",
        "confidence": 80.0,
        "spy_change": -0.4,
        "vix_level": 27.0,
        "vix_change": 8.0,
        "gap": 0.2,
        "intraday_position": 0.2,
        "should_trade": True,
    }

    # 1. Empty table -> never veto.
    g0 = advisor.gate_decision(analysis, overrides=overrides)
    test1 = g0["veto"] is False
    print(f"[1] empty table -> veto={g0['veto']} (expect False) "
          f"reason='{g0['reason']}'  {'PASS' if test1 else 'FAIL'}")

    # 2. Train a strongly-negative, well-visited state -> veto under gate.
    skey = state_key(extract_features(analysis, None, None, "spy_1dte"))
    for _ in range(10):
        advisor.agent.update(skey, "CALL", -0.30, done=True)
    g1 = advisor.gate_decision(analysis, overrides=overrides)
    test2 = g1["veto"] is True
    print(f"[2] negative+visited -> veto={g1['veto']} (expect True) "
          f"q={g1['q']:.3f} visits={g1['visits']} conf={g1['confidence']:.2f} "
          f"{'PASS' if test2 else 'FAIL'}")

    # 3. Same state, but mode != gate -> never veto.
    os.environ["RL_MODE"] = "shadow"
    g2 = advisor.gate_decision(analysis, overrides=overrides)
    test3 = g2["veto"] is False
    print(f"[3] shadow mode -> veto={g2['veto']} (expect False) "
          f"{'PASS' if test3 else 'FAIL'}")
    os.environ["RL_MODE"] = "gate"

    # 4. Positive state -> never veto even if well-visited.
    pos = dict(analysis)
    pos["vix_level"] = 13.0  # different regime/state
    pkey = state_key(extract_features(pos, None, None, "spy_1dte"))
    for _ in range(10):
        advisor.agent.update(pkey, "CALL", +0.25, done=True)
    g3 = advisor.gate_decision(pos, overrides=overrides)
    test4 = g3["veto"] is False
    print(f"[4] positive+visited -> veto={g3['veto']} (expect False) "
          f"q={g3['q']:.3f}  {'PASS' if test4 else 'FAIL'}")

    try:
        os.remove(tmp.name)
        os.remove(tmp.name + ".exp")
    except OSError:
        pass

    ok = test1 and test2 and test3 and test4
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _self_test_costs():
    """No-cred check that NET < gross and that the feature path is no-lookahead."""
    print("=" * 50)
    print("COST + POINT-IN-TIME SELF-TEST")
    print("=" * 50)
    cm = CostModel(load_cost_config_from_env())

    # NET must be strictly below gross (spread + slippage + fees always cost).
    outcome_bar = {"o": 470.0, "h": 474.0, "l": 469.0, "c": 473.0}
    detail = realized_net_detail("CALL", outcome_bar, "2026-01-07", cm)
    cost1 = detail["net_pct"] < detail["gross_pct"]
    print(f"[1] gross={detail['gross_pct']:+.2f}% net={detail['net_pct']:+.2f}% "
          f"(expect net<gross)  {'PASS' if cost1 else 'FAIL'}")

    # A flat move (gross 0) must still be net-negative after costs.
    flat_bar = {"o": 470.0, "h": 470.0, "l": 470.0, "c": 470.0}
    flat = realized_net_detail("CALL", flat_bar, "2026-01-08", cm)
    cost2 = flat["net_pct"] < 0
    print(f"[2] flat move -> net={flat['net_pct']:+.2f}% (expect <0)  "
          f"{'PASS' if cost2 else 'FAIL'}")

    # Point-in-time: a bar whose close is after as_of must never be returned.
    daily = {
        "SPY": [
            make_bar("2026-01-06", 470, 472, 469, 471),
            make_bar("2026-01-07", 471, 475, 470, 474),  # the future bar
        ]
    }
    mv = HistoricalMarketView(datetime(2026, 1, 6, 16, 0), daily=daily)
    seen = mv.daily_bars("SPY", 30)
    pit = all(b.date <= "2026-01-06" for b in seen) and len(seen) == 1
    print(f"[3] only bars<=as_of returned ({len(seen)})  {'PASS' if pit else 'FAIL'}")

    ok = cost1 and cost2 and pit
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    cfg = _gate_config()
    parser = argparse.ArgumentParser(description="RL veto-gate walk-forward backtest")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument(
        "--vix-source", choices=["stooq", "proxy", "both"], default="both"
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--min-visits", type=int, default=cfg["min_visits"])
    parser.add_argument("--max-q", type=float, default=cfg["max_q"])
    parser.add_argument("--min-confidence", type=float, default=cfg["min_confidence"])
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--selftest-costs", action="store_true",
                        help="no-cred cost/point-in-time checks (net<gross)")
    parser.add_argument("--write-episodes", action="store_true",
                        help="persist decision+outcome rows to episodes_backtest.db")
    args = parser.parse_args()

    if args.selftest:
        return _self_test()
    if args.selftest_costs:
        return _self_test_costs()
    return run_backtest(args)


if __name__ == "__main__":
    import sys

    sys.exit(main())
