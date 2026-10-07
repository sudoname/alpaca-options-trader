import json
import math
import os
from datetime import date, timedelta

from iv_rv_experiment import (
    IVRVConfig,
    NOT_RICH,
    RICH,
    UNKNOWN,
    annualized_realized_vol,
    build_paper_trader,
    classify_richness,
    conservative_entry_quotes,
    earliest_expiration,
    intrinsic_settlement_quotes,
    iv_rv_ratio,
    leg_intrinsic,
    scan_once,
    settle_expired,
    summarize_results,
)
from spread_builder import (
    BULLISH_PUT_CREDIT_SPREAD,
    DEBIT_CALL_SPREAD,
    SpreadLeg,
    SpreadProposal,
)

TODAY = date(2026, 10, 6)
EXP = (TODAY + timedelta(days=7)).isoformat()


# --------------------------------------------------------------------------- #
# Fixtures / stubs (no network)
# --------------------------------------------------------------------------- #
def make_cfg(tmp_path, enabled=True, universe=("SPY",), min_ratio=1.25):
    return IVRVConfig(
        enabled=enabled,
        min_iv_rv_ratio=min_ratio,
        rv_window=252,
        min_rv_obs=5,
        max_opens_per_run=3,
        min_oracle_score=0.0,
        universe=list(universe),
        scan_ledger=str(tmp_path / "ledger.jsonl"),
        positions_file=str(tmp_path / "pos.json"),
        trades_file=str(tmp_path / "trades.json"),
    )


def credit_proposal(symbol="SPY"):
    # Bull put credit spread: SELL 100p (1.95/2.05), BUY 95p (1.20/1.30).
    legs = [
        SpreadLeg("sell", "put", 100, bid=1.95, ask=2.05,
                  symbol="SPY_P100", expiration=EXP),
        SpreadLeg("buy", "put", 95, bid=1.20, ask=1.30,
                  symbol="SPY_P95", expiration=EXP),
    ]
    return SpreadProposal(
        strategy_name=BULLISH_PUT_CREDIT_SPREAD, symbol=symbol, legs=legs,
        net_credit_or_debit=0.75, max_profit=75.0, max_loss=425.0,
        breakeven=99.25, width=5.0, oracle_score=80.0)


def debit_proposal(symbol="SPY"):
    legs = [
        SpreadLeg("buy", "call", 100, bid=2.00, ask=2.10,
                  symbol="SPY_C100", expiration=EXP),
        SpreadLeg("sell", "call", 105, bid=0.90, ask=1.00,
                  symbol="SPY_C105", expiration=EXP),
    ]
    return SpreadProposal(
        strategy_name=DEBIT_CALL_SPREAD, symbol=symbol, legs=legs,
        net_credit_or_debit=-1.20, max_profit=380.0, max_loss=120.0,
        oracle_score=80.0)


class StubTrader:
    """Duck-typed stand-in for SmartOptionsTrader. No network, no creds."""

    def __init__(self, price=100.0, iv=0.30, proposal=None):
        self._price = price
        self._iv = iv
        self._proposal = proposal or credit_proposal()

    def get_current_price(self, symbol):
        return self._price

    def get_option_contracts(self, symbol):
        return [{"expiration_date": EXP, "type": "call",
                 "strike_price": "100", "symbol": f"{symbol}_C100"}]

    def get_option_snapshot(self, occ_symbol):
        return {"iv": self._iv}

    def propose_spread(self, symbol):
        return self._proposal


def rich_closes(n=20):
    """Alternating +1%/-1% log returns -> a known positive RV."""
    px = [100.0]
    for i in range(n):
        px.append(px[-1] * math.exp(0.01 if i % 2 == 0 else -0.01))
    return px


def expected_rv(closes):
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    n = len(rets)
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / (n - 1)
    return math.sqrt(var) * math.sqrt(252.0)


# --------------------------------------------------------------------------- #
# Pure signal functions
# --------------------------------------------------------------------------- #
class TestSignals:
    def test_rv_matches_known_value(self):
        closes = rich_closes(20)
        rv = annualized_realized_vol(closes, window=252, min_obs=5)
        assert abs(rv - expected_rv(closes)) < 1e-12

    def test_rv_insufficient_obs(self):
        assert annualized_realized_vol([100, 101, 102], min_obs=60) is None

    def test_rv_bad_inputs(self):
        assert annualized_realized_vol([], min_obs=2) is None
        assert annualized_realized_vol([100, None, -5], min_obs=2) is None

    def test_ratio_and_richness(self):
        assert abs(iv_rv_ratio(0.30, 0.20) - 1.5) < 1e-12
        assert iv_rv_ratio(None, 0.2) is None
        assert iv_rv_ratio(0.3, 0) is None
        assert classify_richness(1.5, 1.25) == RICH
        assert classify_richness(1.1, 1.25) == NOT_RICH
        assert classify_richness(None, 1.25) == UNKNOWN


class TestExecutionConventions:
    def test_conservative_quotes_cross_the_spread(self):
        legs = [l.as_dict() for l in credit_proposal().legs]
        q = conservative_entry_quotes(legs)
        assert q["SPY_P100"] == 1.95   # sell at bid
        assert q["SPY_P95"] == 1.30    # buy at ask

    def test_conservative_quotes_missing_side(self):
        legs = [l.as_dict() for l in credit_proposal().legs]
        legs[1]["ask"] = None          # buy leg without an ask
        assert conservative_entry_quotes(legs) is None

    def test_leg_intrinsic(self):
        assert leg_intrinsic({"type": "call", "strike": 100}, 105) == 5.0
        assert leg_intrinsic({"type": "call", "strike": 100}, 95) == 0.0
        assert leg_intrinsic({"type": "put", "strike": 100}, 90) == 10.0
        assert leg_intrinsic({"type": "put", "strike": 100}, 110) == 0.0
        assert leg_intrinsic({"type": "put", "strike": None}, 110) is None

    def test_intrinsic_settlement_quotes_including_worthless(self):
        legs = [l.as_dict() for l in credit_proposal().legs]
        q = intrinsic_settlement_quotes(legs, 90.0)
        assert q["SPY_P100"] == 10.0 and q["SPY_P95"] == 5.0
        q2 = intrinsic_settlement_quotes(legs, 105.0)
        assert q2["SPY_P100"] == 0.0 and q2["SPY_P95"] == 0.0

    def test_earliest_expiration(self):
        legs = [l.as_dict() for l in credit_proposal().legs]
        assert earliest_expiration(legs) == date.fromisoformat(EXP)
        legs[0]["expiration"] = "garbage"
        assert earliest_expiration(legs) is None


# --------------------------------------------------------------------------- #
# Scan
# --------------------------------------------------------------------------- #
class TestScanDefaultOff:
    def test_disabled_writes_nothing(self, tmp_path):
        cfg = make_cfg(tmp_path, enabled=False)
        summary = scan_once(cfg, trader_factory=lambda s: StubTrader(),
                            closes_by_symbol={"SPY": rich_closes()},
                            today=TODAY)
        assert summary["enabled"] is False
        assert summary["scanned"] == 0
        assert not os.path.exists(cfg.scan_ledger)
        assert not os.path.exists(cfg.positions_file)


class TestScanOpensCreditSpread:
    def test_rich_symbol_opens_with_crossed_entry(self, tmp_path):
        cfg = make_cfg(tmp_path)
        closes = rich_closes()           # rv ~ 0.159 -> iv 0.30 ratio ~ 1.9
        summary = scan_once(cfg, trader_factory=lambda s: StubTrader(),
                            closes_by_symbol={"SPY": closes}, today=TODAY)
        assert summary["opened"] == 1 and summary["rich"] == 1
        row = summary["rows"][0]
        assert row["action"] == "opened"
        assert row["strategy"] == BULLISH_PUT_CREDIT_SPREAD
        # Crossed entry: +1.30 (buy@ask) - 1.95 (sell@bid) = -0.65 credit mark
        assert abs(row["entry_mark"] - (-0.65)) < 1e-9

        pt = build_paper_trader(cfg)
        pos = pt.get_open_positions()[0]
        assert pos["iv_rv_ratio"] == row["ratio"]
        assert pos["atm_iv"] == 0.30
        assert abs(pos["realized_vol"] - expected_rv(closes)) < 1e-12
        assert pos["dte"] == 7
        assert pos["entry_underlying_price"] == 100.0

        # Ledger has exactly one JSONL row for the scanned symbol.
        with open(cfg.scan_ledger, encoding="utf-8") as fh:
            rows = [json.loads(l) for l in fh if l.strip()]
        assert len(rows) == 1 and rows[0]["position_id"] == pos["id"]

    def test_not_rich_symbol_skipped(self, tmp_path):
        cfg = make_cfg(tmp_path)
        summary = scan_once(cfg,
                            trader_factory=lambda s: StubTrader(iv=0.15),
                            closes_by_symbol={"SPY": rich_closes()},
                            today=TODAY)
        assert summary["opened"] == 0
        row = summary["rows"][0]
        assert row["verdict"] == NOT_RICH
        assert row["reason"].startswith("iv_not_rich")
        assert not os.path.exists(cfg.positions_file)

    def test_debit_proposal_skipped(self, tmp_path):
        cfg = make_cfg(tmp_path)
        summary = scan_once(
            cfg,
            trader_factory=lambda s: StubTrader(proposal=debit_proposal()),
            closes_by_symbol={"SPY": rich_closes()}, today=TODAY)
        assert summary["opened"] == 0
        assert summary["rows"][0]["reason"].startswith("not_credit_strategy")

    def test_trader_error_fails_open(self, tmp_path):
        cfg = make_cfg(tmp_path)

        def boom(_):
            raise RuntimeError("api down")

        summary = scan_once(cfg, trader_factory=boom,
                            closes_by_symbol={"SPY": rich_closes()},
                            today=TODAY)
        assert summary["scanned"] == 1 and summary["opened"] == 0
        assert summary["rows"][0]["reason"].startswith("error:")


# --------------------------------------------------------------------------- #
# Settlement at intrinsic
# --------------------------------------------------------------------------- #
def open_one(tmp_path):
    cfg = make_cfg(tmp_path)
    scan_once(cfg, trader_factory=lambda s: StubTrader(),
              closes_by_symbol={"SPY": rich_closes()}, today=TODAY)
    return cfg


class TestSettlement:
    def test_pinned_above_short_keeps_full_crossed_credit(self, tmp_path):
        cfg = open_one(tmp_path)
        summary = settle_expired(cfg, price_lookup=lambda s, d: 105.0,
                                 today=TODAY + timedelta(days=8))
        assert summary["settled"] == 1
        trade = summary["trades"][0]
        # Both puts worthless: mark 0; entry_mark -0.65 -> pnl +65.
        assert abs(trade["pnl"] - 65.0) < 1e-9
        assert trade["exit_reason"] == "expired_intrinsic"
        assert trade["actual_move"] == 5.0
        pt = build_paper_trader(cfg)
        assert not pt.get_open_positions()
        assert len(pt.load_trades()) == 1

    def test_below_long_strike_loses_width_minus_credit(self, tmp_path):
        cfg = open_one(tmp_path)
        summary = settle_expired(cfg, price_lookup=lambda s, d: 90.0,
                                 today=TODAY + timedelta(days=8))
        trade = summary["trades"][0]
        # Intrinsics: short put 10, long put 5 -> mark = 5 - 10 = -5.
        # pnl = (-5 - (-0.65)) * 100 = -435 (width 5 minus crossed credit).
        assert abs(trade["pnl"] - (-435.0)) < 1e-9

    def test_not_yet_expired_left_open(self, tmp_path):
        cfg = open_one(tmp_path)
        summary = settle_expired(cfg, price_lookup=lambda s, d: 105.0,
                                 today=TODAY + timedelta(days=1))
        assert summary["settled"] == 0 and summary["pending"] == 1
        assert build_paper_trader(cfg).get_open_positions()

    def test_missing_close_retried_later(self, tmp_path):
        cfg = open_one(tmp_path)
        summary = settle_expired(cfg, price_lookup=lambda s, d: None,
                                 today=TODAY + timedelta(days=8))
        assert summary["settled"] == 0 and summary["pending"] == 1
        assert build_paper_trader(cfg).get_open_positions()

    def test_disabled_settle_noop(self, tmp_path):
        cfg = open_one(tmp_path)
        cfg_off = make_cfg(tmp_path, enabled=False)
        summary = settle_expired(cfg_off, price_lookup=lambda s, d: 105.0,
                                 today=TODAY + timedelta(days=8))
        assert summary["enabled"] is False and summary["checked"] == 0
        assert build_paper_trader(cfg).get_open_positions()


# --------------------------------------------------------------------------- #
# Evidence-gate summary
# --------------------------------------------------------------------------- #
class TestSummarize:
    def test_small_sample_never_passes_gate(self, tmp_path):
        cfg = open_one(tmp_path)
        settle_expired(cfg, price_lookup=lambda s, d: 105.0,
                       today=TODAY + timedelta(days=8))
        s = summarize_results(cfg)
        assert s["n_settled"] == 1 and s["n_open"] == 0
        assert s["gate_passed"] is False   # n < 40 regardless of outcome

    def test_mean_and_t_math(self, tmp_path):
        cfg = make_cfg(tmp_path)
        pt = build_paper_trader(cfg)
        # Synthetic settled trades: returns per $ max-loss of +0.10 and -0.05.
        pt.save_trades([
            {"pnl": 42.5, "max_loss": 425.0},
            {"pnl": -21.25, "max_loss": 425.0},
        ])
        s = summarize_results(cfg)
        assert s["n_settled"] == 2
        assert abs(s["mean_ret_per_maxloss"] - 0.025) < 1e-9
        assert s["win_rate"] == 0.5
        assert s["gate_passed"] is False
