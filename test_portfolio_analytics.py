"""Unit tests for portfolio_analytics + sector_map (pure, no network)."""

import portfolio_analytics as pa
import sector_map


def test_self_tests_pass():
    assert pa._self_test() == 0
    assert sector_map._self_test() == 0


def test_parse_position_option_and_equity():
    call = pa.parse_position({"symbol": "AAPL260710C00270000", "qty": "2",
                              "side": "long"})
    assert call["underlying"] == "AAPL"
    assert call["kind"] == "call"
    assert call["strike"] == 270.0
    assert call["signed_qty"] == 2.0
    eq = pa.parse_position({"symbol": "NVDA", "qty": "10", "side": "short"})
    assert eq["kind"] == "equity"
    assert eq["signed_qty"] == -10.0


def test_net_greeks_live_and_fallback():
    positions = pa.parse_positions([
        {"symbol": "AAPL260710C00270000", "qty": "2", "side": "long"},
        {"symbol": "AAPL260710P00270000", "qty": "1", "side": "long"},
    ])
    ng = pa.net_greeks(positions, {"AAPL260710C00270000": {"delta": 0.6}})
    # call 0.6*2*100=120 ; put heuristic -0.5*1*100=-50 -> 70
    assert abs(ng["net_delta"] - 70.0) < 1e-6
    assert ng["greeks_live"] == 1 and ng["greeks_fallback"] == 1


def test_sector_weights_sum_to_one():
    positions = pa.parse_positions([
        {"symbol": "AAPL260710C00270000", "qty": "2", "side": "long",
         "market_value": "600"},
        {"symbol": "XOM260117C00110000", "qty": "1", "side": "long",
         "market_value": "100"},
    ])
    se = pa.sector_exposure(positions)
    assert abs(sum(se["weights"].values()) - 1.0) < 1e-6
    assert "Energy" in se["weights"]


def test_beta_and_correlation():
    spy = [100, 101, 102, 101, 103, 104, 103, 105]
    spy_ret = pa._returns(spy)
    lev = [100.0]
    for r in spy_ret:
        lev.append(lev[-1] * (1 + 2.0 * r))
    pb = pa.portfolio_beta({"LEV": 1.0}, {"LEV": lev}, spy)
    assert abs(pb["portfolio_beta"] - 2.0) < 0.05
    cs = pa.correlation_score({"A": 1.0, "B": 1.0}, {"A": spy, "B": list(spy)})
    assert abs(cs["correlation_score"] - 1.0) < 1e-6


def test_export_positions_parse_real_file_shape():
    # Loader fails open on a missing file.
    assert pa.load_export_positions("does_not_exist.csv") == []


def test_compute_and_format_never_raise_on_empty():
    rep = pa.compute_portfolio([])
    md = pa.format_markdown(rep)
    assert "Portfolio" in md
