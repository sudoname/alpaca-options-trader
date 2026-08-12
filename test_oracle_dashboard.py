"""
Offline tests for the Oracle Web Dashboard API (Phase 1).

No creds, no network, no broker, no WSGI server: every test drives the Flask app
through ``app.test_client()`` and monkeypatches providers so nothing leaves the
box. The contract pinned here:

  1. READ-ONLY BY CONSTRUCTION. The module imports no execution path — asserted
     by source grep (no place/execute/open/close/submit).
  2. FAIL-OPEN, NEVER 500. Every ``/api/*`` returns 200 + a JSON body carrying a
     ``verdict``; empty sources -> INSUFFICIENT_DATA; a raising provider ->
     ``verdict == "ERROR"`` (still 200).
  3. AUTH IS DEFENSE-IN-DEPTH. When user+pass are configured, protected routes
     401 without creds and 200 with; ``/api/health`` stays open. Unset -> open.
  4. TTL CACHE. A second hit within the TTL is served from cache (provider not
     re-invoked); ``?fresh=1`` bypasses.
  5. ``_self_test()`` returns 0.
"""

import re
import unittest

import oracle_dashboard as od
from oracle_dashboard import DashboardConfig, create_app, _safe_provider


def _cfg(user="", password="", ttl=0):
    return DashboardConfig(host="127.0.0.1", port=0, cache_ttl=ttl,
                           basic_auth_user=user, basic_auth_pass=password)


def _app(user="", password="", ttl=0, explain_ctx=None, regime_ctx=None):
    """Build an app with the context builders stubbed offline.

    The live builders issue read-only Alpaca GETs; tests must never touch the
    network, so default them to no-evidence stubs (yield INSUFFICIENT_DATA).
    """
    app = create_app(_cfg(user, password, ttl))
    app.config["EXPLAIN_CTX_BUILDER"] = explain_ctx or (lambda s: {})
    app.config["REGIME_CTX_BUILDER"] = regime_ctx or (lambda: {})
    # Keep sentiment offline: the live report scrapes CNN + GETs Alpaca bars.
    app.config["SENTIMENT_REPORT"] = lambda: {"verdict": "INSUFFICIENT_DATA"}
    return app


class TestHealthAndFailOpen(unittest.TestCase):
    def setUp(self):
        self.client = _app().test_client()

    def test_health_ok(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json().get("status"), "ok")

    ALL_ENDPOINTS = (
        "/api/daily", "/api/regime", "/api/sentiment", "/api/agents",
        "/api/probability",
        "/api/weights", "/api/feature-importance", "/api/ev-attribution",
        "/api/regime-performance", "/api/hypotheses", "/api/calibration/pop",
        "/api/calibration/ev", "/api/calibration/triple-gap",
        "/api/explain/SPY", "/api/positions",
        "/api/single-leg/kpis", "/api/single-leg/positions",
        "/api/single-leg/episodes", "/api/put-time-stop",
    )

    def test_every_endpoint_200_with_verdict(self):
        for path in self.ALL_ENDPOINTS:
            r = self.client.get(path)
            self.assertEqual(r.status_code, 200, msg=path)
            body = r.get_json()
            self.assertIsInstance(body, dict, msg=path)
            self.assertIn("verdict", body, msg=path)

    def test_raising_provider_yields_error_verdict(self):
        out = _safe_provider(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        self.assertEqual(out["verdict"], "ERROR")
        self.assertIn("boom", out["error"])

    def test_non_dict_provider_is_error(self):
        out = _safe_provider(lambda: ["not", "a", "dict"])
        self.assertEqual(out["verdict"], "ERROR")


class TestPutTimeStop(unittest.TestCase):
    """/api/put-time-stop joins boundaries to resolutions over the shadow ledger.

    Offline: the observer's load_records/summarize are monkeypatched with canned
    ledger data so no JSONL file / .env / disk read is needed.
    """

    def setUp(self):
        from oracle import put_time_stop_observer as ptss
        self.ptss = ptss
        self._orig = (ptss.load_records, ptss.summarize)

    def tearDown(self):
        self.ptss.load_records, self.ptss.summarize = self._orig

    def test_empty_ledger_is_insufficient(self):
        self.ptss.load_records = lambda path=None: []
        self.ptss.summarize = lambda path=None: {
            "n_boundaries": 0, "n_resolved": 0, "mean_shadow_delta_pct": None,
            "median_shadow_delta_pct": None, "count_helped": 0, "count_hurt": 0}
        body = _app().test_client().get("/api/put-time-stop").get_json()
        self.assertEqual(body.get("verdict"), "INSUFFICIENT_DATA")
        self.assertEqual(body.get("boundaries"), [])

    def test_boundary_join_surfaces_resolution(self):
        recs = [
            {"type": "boundary", "key": "k1", "symbol": "SOFI...P", "underlying":
             "SOFI", "entry_time": "2026-08-07T15:39:33", "entry_price": 1.0,
             "cap_days": 5, "hold_days_at_boundary": 5, "boundary_mark": 1.06,
             "boundary_pnl_pct": 6.0},
            {"type": "boundary", "key": "k2", "symbol": "QQQ...P", "underlying":
             "QQQ", "entry_time": "2026-08-09T10:00:00", "entry_price": 2.0,
             "cap_days": 5, "hold_days_at_boundary": 6, "boundary_mark": 1.6,
             "boundary_pnl_pct": -20.0},
            {"type": "resolution", "key": "k2", "boundary_pnl_pct": -20.0,
             "actual_exit_pnl_pct": -35.0, "held_extra_days": 4,
             "shadow_delta_pct": 15.0},
        ]
        self.ptss.load_records = lambda path=None: recs
        self.ptss.summarize = lambda path=None: {
            "n_boundaries": 2, "n_resolved": 1, "mean_shadow_delta_pct": 15.0,
            "median_shadow_delta_pct": 15.0, "count_helped": 1, "count_hurt": 0}
        body = _app().test_client().get("/api/put-time-stop").get_json()
        self.assertEqual(body.get("verdict"), "OK")
        self.assertEqual(body.get("n_boundaries"), 2)
        self.assertEqual(body.get("count_helped"), 1)
        self.assertEqual(body.get("cap_days"), 5)
        by_key = {b["symbol"]: b for b in body["boundaries"]}
        # newest entry_time first
        self.assertEqual(body["boundaries"][0]["symbol"], "QQQ...P")
        self.assertTrue(by_key["QQQ...P"]["resolved"])
        self.assertEqual(by_key["QQQ...P"]["shadow_delta_pct"], 15.0)
        self.assertFalse(by_key["SOFI...P"]["resolved"])
        self.assertIsNone(by_key["SOFI...P"]["shadow_delta_pct"])


class TestExplainSanitization(unittest.TestCase):
    def setUp(self):
        self.client = _app().test_client()

    def test_valid_tickers_pass(self):
        for t in ("SPY", "AAPL", "BRK.B"):
            r = self.client.get(f"/api/explain/{t}")
            self.assertEqual(r.status_code, 200, msg=t)
            self.assertIn("verdict", r.get_json(), msg=t)

    def test_lowercase_is_normalized(self):
        self.assertEqual(self.client.get("/api/explain/spy").status_code, 200)

    def test_junk_tickers_rejected_400(self):
        for junk in ("spy;rm", "TOOLONGTICKER", "1", "a%20b", "SPY1"):
            r = self.client.get(f"/api/explain/{junk}")
            self.assertEqual(r.status_code, 400, msg=junk)
            self.assertEqual(r.get_json().get("verdict"), "ERROR", msg=junk)


class TestExplainContext(unittest.TestCase):
    """The route feeds the injectable builder's ctx into compute_oracle_explain."""

    def test_populated_ctx_is_not_insufficient(self):
        ctx = {"trend": "up", "momentum": 0.05, "realized_vol": 0.2}
        client = _app(explain_ctx=lambda s: ctx).test_client()
        body = client.get("/api/explain/SPY").get_json()
        self.assertEqual(body.get("verdict"), "OK")

    def test_empty_ctx_stays_insufficient(self):
        client = _app(explain_ctx=lambda s: {}).test_client()
        body = client.get("/api/explain/SPY").get_json()
        self.assertEqual(body.get("verdict"), "INSUFFICIENT_DATA")

    def test_builder_receives_sanitized_symbol(self):
        seen = {}

        def builder(sym):
            seen["sym"] = sym
            return {}

        client = _app(explain_ctx=builder).test_client()
        client.get("/api/explain/spy")
        self.assertEqual(seen.get("sym"), "SPY")

    def test_raising_builder_fails_open(self):
        def boom(s):
            raise RuntimeError("network down")

        client = _app(explain_ctx=boom).test_client()
        r = client.get("/api/explain/SPY")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json().get("verdict"), "ERROR")


class TestRegimeContext(unittest.TestCase):
    """The route feeds the injectable builder's ctx into the regime report."""

    def test_populated_ctx_is_ok_with_label(self):
        ctx = {"trend": "up", "momentum": 0.05, "realized_vol": 0.012}
        client = _app(regime_ctx=lambda: ctx).test_client()
        body = client.get("/api/regime").get_json()
        self.assertEqual(body.get("verdict"), "OK")
        self.assertTrue(body.get("label"))

    def test_empty_ctx_stays_insufficient(self):
        client = _app(regime_ctx=lambda: {}).test_client()
        body = client.get("/api/regime").get_json()
        self.assertEqual(body.get("verdict"), "INSUFFICIENT_DATA")

    def test_raising_builder_fails_open(self):
        def boom():
            raise RuntimeError("network down")

        client = _app(regime_ctx=boom).test_client()
        r = client.get("/api/regime")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json().get("verdict"), "ERROR")


class TestSentiment(unittest.TestCase):
    """The /api/sentiment route surfaces the injectable SENTIMENT_REPORT."""

    def _client(self, report):
        app = create_app(_cfg())
        app.config["SENTIMENT_REPORT"] = lambda: report
        return app.test_client()

    def test_available_report_passes_through(self):
        report = {"verdict": "OK", "score": 55.0, "classification": "Neutral",
                  "source": "blend"}
        body = self._client(report).get("/api/sentiment").get_json()
        self.assertEqual(body.get("verdict"), "OK")
        self.assertEqual(body.get("score"), 55.0)
        self.assertEqual(body.get("classification"), "Neutral")

    def test_insufficient_report_is_not_500(self):
        r = self._client({"verdict": "INSUFFICIENT_DATA"}).get("/api/sentiment")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json().get("verdict"), "INSUFFICIENT_DATA")

    def test_raising_report_fails_open_to_error(self):
        def boom():
            raise RuntimeError("cnn down")
        app = create_app(_cfg())
        app.config["SENTIMENT_REPORT"] = boom
        r = app.test_client().get("/api/sentiment")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json().get("verdict"), "ERROR")


class TestBasicAuth(unittest.TestCase):
    def setUp(self):
        self.client = create_app(_cfg(user="u", password="p")).test_client()

    def test_401_without_creds(self):
        r = self.client.get("/api/daily")
        self.assertEqual(r.status_code, 401)
        self.assertIn("WWW-Authenticate", r.headers)

    def test_200_with_creds(self):
        import base64
        tok = base64.b64encode(b"u:p").decode()
        r = self.client.get("/api/daily",
                            headers={"Authorization": f"Basic {tok}"})
        self.assertEqual(r.status_code, 200)

    def test_wrong_creds_rejected(self):
        import base64
        tok = base64.b64encode(b"u:wrong").decode()
        r = self.client.get("/api/daily",
                            headers={"Authorization": f"Basic {tok}"})
        self.assertEqual(r.status_code, 401)

    def test_health_open_under_auth(self):
        self.assertEqual(self.client.get("/api/health").status_code, 200)

    def test_open_when_unconfigured(self):
        client = create_app(_cfg()).test_client()
        self.assertEqual(client.get("/api/daily").status_code, 200)


class TestTTLCache(unittest.TestCase):
    def test_second_hit_served_from_cache(self):
        calls = {"n": 0}

        def provider(*args, **kwargs):
            calls["n"] += 1
            return {"verdict": "OK", "n": calls["n"]}

        app = create_app(_cfg(ttl=60))
        app.config["REGIME_CTX_BUILDER"] = lambda: {}
        # Patch the provider by overriding the module function the route imports.
        import oracle_intelligence_reports as oir
        orig = oir.compute_oracle_regime_report
        oir.compute_oracle_regime_report = provider
        try:
            client = app.test_client()
            first = client.get("/api/regime").get_json()
            second = client.get("/api/regime").get_json()
            self.assertEqual(first, second)
            self.assertEqual(calls["n"], 1)  # provider invoked once
            fresh = client.get("/api/regime?fresh=1").get_json()
            self.assertEqual(calls["n"], 2)  # bypassed cache
            self.assertEqual(fresh["n"], 2)
        finally:
            oir.compute_oracle_regime_report = orig


class TestReadOnlyByConstruction(unittest.TestCase):
    def test_no_execution_symbols_in_source(self):
        import inspect
        src = inspect.getsource(od)
        # Call-form tokens: matches actual execution calls, not the read-only
        # ``open_positions`` variable / ``load_open_spread_positions`` loader.
        forbidden = ("place_option_order(", "submit_order(", "execute_trade(",
                     "open_position(", "close_position(", "record_outcome(")
        for token in forbidden:
            self.assertNotIn(token, src, msg=f"execution token {token!r} present")

    def test_only_get_routes(self):
        app = create_app(_cfg())
        for rule in app.url_map.iter_rules():
            self.assertNotIn("POST", rule.methods, msg=str(rule))
            self.assertNotIn("PUT", rule.methods, msg=str(rule))
            self.assertNotIn("DELETE", rule.methods, msg=str(rule))


class TestSelfTest(unittest.TestCase):
    def test_self_test_passes(self):
        self.assertEqual(od._self_test(), 0)


if __name__ == "__main__":
    unittest.main()
