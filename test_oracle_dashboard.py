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


class TestHealthAndFailOpen(unittest.TestCase):
    def setUp(self):
        self.client = create_app(_cfg()).test_client()

    def test_health_ok(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json().get("status"), "ok")

    ALL_ENDPOINTS = (
        "/api/daily", "/api/regime", "/api/agents", "/api/probability",
        "/api/weights", "/api/feature-importance", "/api/ev-attribution",
        "/api/regime-performance", "/api/hypotheses", "/api/calibration/pop",
        "/api/calibration/ev", "/api/calibration/triple-gap",
        "/api/explain/SPY", "/api/positions",
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


class TestExplainSanitization(unittest.TestCase):
    def setUp(self):
        self.client = create_app(_cfg()).test_client()

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

        def provider():
            calls["n"] += 1
            return {"verdict": "OK", "n": calls["n"]}

        app = create_app(_cfg(ttl=60))
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
