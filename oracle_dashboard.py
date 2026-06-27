"""
Oracle Web Dashboard — read-only analytics API (Phase 1 skeleton).

A SEPARATE Flask process from ``alps-bot``/``alps-scheduler``. It imports only
pure ``compute_*``/``build_*`` report functions and read-only loaders, exposes
only ``GET`` endpoints, and binds to ``127.0.0.1`` behind an nginx reverse proxy
(TLS via certbot + basic auth; see deploy/nginx-oracle-dashboard.conf). It
CANNOT open, size, price, gate, or close any real or paper position — there is
no execution path in this module by construction.

Project idioms mirrored here:
  * ``DashboardConfig.from_env(path=".env", loader=None)`` — shell > .env > default.
  * ``_self_test() -> int`` and ``if __name__ == "__main__": sys.exit(...) | serve``.
  * Fail-open everywhere: a missing data file or a raising provider yields a JSON
    body carrying a ``verdict`` ("INSUFFICIENT_DATA"/"ERROR"), never a 500.

Each handler lazy-imports its backing module inside a try/except so that an
import error or runtime fault in one analytics module degrades that one endpoint
to ``{"verdict": "ERROR", ...}`` without taking down the server.

Phase 1 endpoints: /api/health, /api/daily, /api/regime, /api/agents, plus the
static host for ``dashboard/`` (added in Phase 3).
"""

import hmac
import os
import re
import sys
import time

from flask import Flask, jsonify, request, Response, send_from_directory

from config_loader import ConfigLoader

# Buildless frontend lives in ``dashboard/`` next to this module.
_DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "dashboard")

VERDICT_ERROR = "ERROR"
VERDICT_INSUFFICIENT = "INSUFFICIENT_DATA"

# Tickers: 1-8 uppercase letters/dots only (e.g. SPY, BRK.B). Anything else is
# rejected before it reaches a provider — no path traversal, no injection.
_TICKER_RE = re.compile(r"^[A-Z.]{1,8}$")


class DashboardConfig:
    """Resolved dashboard settings (shell env > .env > code default)."""

    def __init__(self, host, port, cache_ttl, basic_auth_user, basic_auth_pass):
        self.host = host
        self.port = port
        self.cache_ttl = cache_ttl
        self.basic_auth_user = basic_auth_user
        self.basic_auth_pass = basic_auth_pass

    @property
    def auth_enabled(self) -> bool:
        """App-level basic auth is enforced only when BOTH creds are set.

        Off for local dev; in production nginx enforces auth regardless, so this
        is defense-in-depth, not the primary gate.
        """
        return bool(self.basic_auth_user) and bool(self.basic_auth_pass)

    @classmethod
    def from_env(cls, path: str = ".env", loader=None) -> "DashboardConfig":
        c = loader if loader is not None else ConfigLoader(path=path)
        return cls(
            host=c.get_str("DASHBOARD_HOST", "127.0.0.1"),
            port=c.get_int("DASHBOARD_PORT", 8787),
            cache_ttl=c.get_int("DASHBOARD_CACHE_TTL", 60),
            basic_auth_user=c.get_str("DASHBOARD_BASIC_AUTH_USER", ""),
            basic_auth_pass=c.get_str("DASHBOARD_BASIC_AUTH_PASS", ""),
        )


class _TTLCache:
    """Tiny in-process ``{key: (ts, payload)}`` cache.

    A browser refresh shouldn't re-read ``episodes.db``/JSONL on every hit.
    Entries live ``ttl`` seconds; ``ttl <= 0`` disables caching. ``?fresh=1``
    bypasses (handled by the caller).
    """

    def __init__(self, ttl: int):
        self.ttl = ttl
        self._store: dict = {}

    def get(self, key):
        if self.ttl <= 0:
            return None
        hit = self._store.get(key)
        if hit is None:
            return None
        ts, payload = hit
        if (time.time() - ts) > self.ttl:
            self._store.pop(key, None)
            return None
        return payload

    def put(self, key, payload):
        if self.ttl > 0:
            self._store[key] = (time.time(), payload)

    def clear(self):
        self._store.clear()


def _safe_provider(fn):
    """Call a zero-arg provider, never raising.

    Returns the provider's dict on success; on ANY exception returns a fail-open
    body carrying ``verdict: "ERROR"`` so the endpoint stays 200.
    """
    try:
        result = fn()
        if isinstance(result, dict):
            # Composite reports (e.g. the daily summary) are read field-by-field
            # and don't follow the single-verdict convention. A successful call
            # without a verdict is OK; per-report verdicts are preserved.
            result.setdefault("verdict", "OK")
            return result
        return {"verdict": VERDICT_ERROR,
                "error": f"provider returned {type(result).__name__}, expected dict"}
    except Exception as e:  # pragma: no cover - exercised via monkeypatch in tests
        return {"verdict": VERDICT_ERROR, "error": str(e)}


def create_app(config: "DashboardConfig" = None) -> Flask:
    """Build the Flask app. Pure constructor — no network, no serving."""
    cfg = config if config is not None else DashboardConfig.from_env()
    app = Flask(__name__, static_folder=None)
    app.config["DASHBOARD"] = cfg
    cache = _TTLCache(cfg.cache_ttl)
    app.config["DASHBOARD_CACHE"] = cache

    # Read-only evidence-context builder for /api/explain. Injectable so the
    # self-test / unit tests stay offline (they override it with a stub that
    # returns {}); the default assembles live market context (and itself fails
    # open to {} on missing creds / no network).
    def _default_explain_ctx(symbol):
        try:
            import explain_context
            return explain_context.build_explain_context(symbol)
        except Exception:
            return {}
    app.config.setdefault("EXPLAIN_CTX_BUILDER", _default_explain_ctx)

    # Read-only market context for /api/regime. Injectable so the self-test /
    # unit tests stay offline (override with a stub returning {}); the default
    # assembles live SPY context and itself fails open to {} on missing creds /
    # no network. Without a context the regime report stays INSUFFICIENT_DATA.
    def _default_regime_ctx():
        try:
            import explain_context
            return explain_context.build_explain_context("SPY")
        except Exception:
            return {}
    app.config.setdefault("REGIME_CTX_BUILDER", _default_regime_ctx)

    # Read-only Fear & Greed report for /api/sentiment. Injectable so the
    # self-test / unit tests stay offline (override with a stub); the default
    # blends CNN's index with a self-computed score over read-only market data
    # and itself fails open to INSUFFICIENT_DATA on missing creds / no network.
    def _default_sentiment():
        try:
            import sentiment_report
            return sentiment_report.compute_sentiment_report()
        except Exception:
            return {"verdict": VERDICT_INSUFFICIENT}
    app.config.setdefault("SENTIMENT_REPORT", _default_sentiment)

    # -- basic-auth (defense-in-depth) ----------------------------------- #
    def _check_auth(auth) -> bool:
        if auth is None:
            return False
        user_ok = hmac.compare_digest(str(auth.username or ""),
                                      cfg.basic_auth_user)
        pass_ok = hmac.compare_digest(str(auth.password or ""),
                                      cfg.basic_auth_pass)
        return user_ok and pass_ok

    @app.before_request
    def _enforce_auth():
        if not cfg.auth_enabled:
            return None
        if request.path == "/api/health":
            return None  # health stays open for liveness probes
        if _check_auth(request.authorization):
            return None
        return Response(
            "Authentication required.", 401,
            {"WWW-Authenticate": 'Basic realm="Oracle Dashboard"'})

    # -- cached JSON helper ---------------------------------------------- #
    def _cached_json(key, provider):
        if request.args.get("fresh") != "1":
            cached = cache.get(key)
            if cached is not None:
                return jsonify(cached)
        payload = _safe_provider(provider)
        cache.put(key, payload)
        return jsonify(payload)

    # -- endpoints ------------------------------------------------------- #
    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok", "verdict": "OK"})

    @app.route("/api/daily")
    def daily():
        def provider():
            import oracle_daily_report
            return oracle_daily_report.build_daily_report()
        return _cached_json("daily", provider)

    @app.route("/api/regime")
    def regime():
        def provider():
            import oracle_intelligence_reports as oir
            ctx = app.config["REGIME_CTX_BUILDER"]() or None
            return oir.compute_oracle_regime_report(regime_raw=ctx, symbol="SPY")
        return _cached_json("regime", provider)

    @app.route("/api/sentiment")
    def sentiment():
        def provider():
            fn = app.config.get("SENTIMENT_REPORT")
            return fn() if fn else {"verdict": VERDICT_INSUFFICIENT}
        return _cached_json("sentiment", provider)

    @app.route("/api/agents")
    def agents():
        def provider():
            import oracle_intelligence_reports as oir
            return oir.compute_oracle_agent_report()
        return _cached_json("agents", provider)

    @app.route("/api/probability")
    def probability():
        def provider():
            import oracle_intelligence_reports as oir
            return oir.compute_oracle_probability_report()
        return _cached_json("probability", provider)

    @app.route("/api/weights")
    def weights():
        def provider():
            import oracle_intelligence_reports as oir
            return oir.compute_oracle_weight_changes()
        return _cached_json("weights", provider)

    @app.route("/api/feature-importance")
    def feature_importance():
        def provider():
            import oracle_intelligence_reports as oir
            return oir.compute_oracle_feature_importance()
        return _cached_json("feature-importance", provider)

    @app.route("/api/ev-attribution")
    def ev_attribution():
        def provider():
            import ev_attribution as eva
            return eva.compute_ev_attribution()
        return _cached_json("ev-attribution", provider)

    @app.route("/api/regime-performance")
    def regime_performance():
        def provider():
            import oracle_intelligence_reports as oir
            return oir.compute_oracle_regime_performance()
        return _cached_json("regime-performance", provider)

    @app.route("/api/hypotheses")
    def hypotheses():
        def provider():
            import hypothesis_engine as he
            items = he.compute_all_hypotheses()
            ranked = sorted(
                items or [],
                key=lambda h: ((h.get("confidence") or 0),
                               abs(h.get("effect_size") or 0)),
                reverse=True)
            verdict = "OK" if ranked else VERDICT_INSUFFICIENT
            return {"verdict": verdict, "hypotheses": ranked}
        return _cached_json("hypotheses", provider)

    @app.route("/api/calibration/pop")
    def calibration_pop():
        def provider():
            import pop_calibration as pc
            return pc.compute_pop_calibration()
        return _cached_json("calibration-pop", provider)

    @app.route("/api/calibration/ev")
    def calibration_ev():
        def provider():
            import ev_calibration as ec
            return ec.compute_ev_calibration()
        return _cached_json("calibration-ev", provider)

    @app.route("/api/calibration/triple-gap")
    def calibration_triple_gap():
        def provider():
            import calibration_reports as cr
            return cr.compute_triple_gap_report()
        return _cached_json("calibration-triple-gap", provider)

    @app.route("/api/explain/<ticker>")
    def explain(ticker):
        clean = str(ticker or "").strip().upper()
        if not _TICKER_RE.match(clean):
            return jsonify({"verdict": VERDICT_ERROR,
                            "error": "invalid ticker"}), 400

        def provider():
            import oracle_intelligence_reports as oir
            ctx_builder = app.config.get("EXPLAIN_CTX_BUILDER")
            ctx = ctx_builder(clean) if ctx_builder else None
            return oir.compute_oracle_explain(clean, ctx=ctx)
        return _cached_json(f"explain:{clean}", provider)

    @app.route("/api/positions")
    def positions():
        def provider():
            import oracle_analytics as oa
            import oracle_daily_report as odr
            cfg_a = oa.AnalyticsConfig.from_env()
            open_positions = oa.load_open_spread_positions(cfg_a)
            account = odr.build_daily_report().get("account", {})
            return {"verdict": "OK", "positions": open_positions,
                    "account": account}
        return _cached_json("positions", provider)

    # -- single-leg deployment views ------------------------------------- #
    # This box runs the single-leg intraday bot; the Oracle/spread-paper
    # analytics above are dormant here. These endpoints read the single-leg
    # stores (active_trades.json / trading_history.json / realized_pnl_log.json
    # / episodes.db) so the dashboard reflects the activity that exists.
    @app.route("/api/single-leg/kpis")
    def single_leg_kpis():
        def provider():
            import single_leg_reports as slr
            return slr.compute_single_leg_kpis()
        return _cached_json("single-leg-kpis", provider)

    @app.route("/api/single-leg/positions")
    def single_leg_positions():
        def provider():
            import single_leg_reports as slr
            return slr.compute_single_leg_positions()
        return _cached_json("single-leg-positions", provider)

    @app.route("/api/single-leg/episodes")
    def single_leg_episodes():
        def provider():
            import single_leg_reports as slr
            return slr.compute_single_leg_episodes()
        return _cached_json("single-leg-episodes", provider)

    # -- evidence-EV leaderboard + consolidated daily report v2 ---------- #
    @app.route("/api/evidence-leaderboard")
    def evidence_leaderboard():
        def provider():
            import evidence_attribution as ea
            return ea.compute_all()
        return _cached_json("evidence-leaderboard", provider)

    @app.route("/api/daily-v2")
    def daily_v2():
        def provider():
            import daily_report_v2 as drv2
            report = drv2.build_consolidated_report()
            return drv2._json_safe(report)
        return _cached_json("daily-v2", provider)

    # -- static frontend ------------------------------------------------- #
    @app.route("/")
    def index():
        if not os.path.isdir(_DASHBOARD_DIR):
            return jsonify({"status": "api-only",
                            "hint": "dashboard/ not found"}), 200
        return send_from_directory(_DASHBOARD_DIR, "index.html")

    @app.route("/<path:filename>")
    def static_files(filename):
        # Serve the requested asset when it exists; otherwise fall back to the
        # SPA entry point so client-side (TanStack Router) deep links resolve
        # instead of 404ing. /api/* routes are matched earlier, so this only
        # affects static/front-end paths.
        full = os.path.join(_DASHBOARD_DIR, filename)
        if os.path.isfile(full):
            return send_from_directory(_DASHBOARD_DIR, filename)
        return send_from_directory(_DASHBOARD_DIR, "index.html")

    return app


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network; uses Flask's in-process test client)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    # 1. Health is 200 + JSON, open even when auth is configured.
    app = create_app(DashboardConfig(host="127.0.0.1", port=0, cache_ttl=0,
                                     basic_auth_user="", basic_auth_pass=""))
    # Keep explain offline+deterministic: stub the context builder (the live
    # builder would otherwise issue read-only Alpaca GETs during the gate).
    app.config["EXPLAIN_CTX_BUILDER"] = lambda s: {}
    # Keep regime offline+deterministic: an empty ctx keeps the regime report
    # at INSUFFICIENT_DATA without issuing read-only Alpaca GETs during the gate.
    app.config["REGIME_CTX_BUILDER"] = lambda: {}
    # Keep sentiment offline+deterministic: the live report would otherwise scrape
    # CNN and issue read-only Alpaca GETs during the gate.
    app.config["SENTIMENT_REPORT"] = lambda: {"verdict": VERDICT_INSUFFICIENT}
    client = app.test_client()
    r = client.get("/api/health")
    if r.status_code != 200 or r.get_json().get("status") != "ok":
        print("FAIL: /api/health not 200/ok:", r.status_code); ok = False

    # 2. Each analytics endpoint returns 200 + JSON carrying a verdict, even
    #    with empty data sources (fail-open, never a 500).
    endpoints = (
        "/api/daily", "/api/regime", "/api/sentiment", "/api/agents",
        "/api/probability",
        "/api/weights", "/api/feature-importance", "/api/ev-attribution",
        "/api/regime-performance", "/api/hypotheses", "/api/calibration/pop",
        "/api/calibration/ev", "/api/calibration/triple-gap",
        "/api/explain/SPY", "/api/positions",
        "/api/single-leg/kpis", "/api/single-leg/positions",
        "/api/single-leg/episodes",
    )
    for path in endpoints:
        r = client.get(path)
        if r.status_code != 200:
            print(f"FAIL: {path} status {r.status_code}"); ok = False
            continue
        body = r.get_json()
        if not isinstance(body, dict) or "verdict" not in body:
            print(f"FAIL: {path} body missing verdict: {body!r}"); ok = False

    # 2b. Junk tickers are rejected with 400 (sanitized before any provider).
    for junk in ("spy;rm", "TOOLONGTICKER", "1", "a%20b"):
        if client.get(f"/api/explain/{junk}").status_code != 400:
            print(f"FAIL: junk ticker {junk!r} should 400"); ok = False

    # 2c. With a populated evidence context, explain leaves INSUFFICIENT_DATA.
    app_ctx = create_app(DashboardConfig(host="127.0.0.1", port=0, cache_ttl=0,
                                         basic_auth_user="", basic_auth_pass=""))
    app_ctx.config["EXPLAIN_CTX_BUILDER"] = lambda s: {
        "trend": "up", "momentum": 0.05, "realized_vol": 0.2}
    rc = app_ctx.test_client().get("/api/explain/SPY")
    jb = rc.get_json()
    if rc.status_code != 200 or jb.get("verdict") == VERDICT_INSUFFICIENT:
        print("FAIL: explain with ctx should not be INSUFFICIENT_DATA:", jb); ok = False

    # 2d. /api/regime: empty ctx stays INSUFFICIENT_DATA; a populated ctx flips
    #     to OK with a regime label (read-only context wiring).
    app_reg = create_app(DashboardConfig(host="127.0.0.1", port=0, cache_ttl=0,
                                         basic_auth_user="", basic_auth_pass=""))
    app_reg.config["REGIME_CTX_BUILDER"] = lambda: {}
    rr0 = app_reg.test_client().get("/api/regime")
    if rr0.status_code != 200 or rr0.get_json().get("verdict") != VERDICT_INSUFFICIENT:
        print("FAIL: regime empty ctx should be INSUFFICIENT_DATA:",
              rr0.get_json()); ok = False
    app_reg2 = create_app(DashboardConfig(host="127.0.0.1", port=0, cache_ttl=0,
                                          basic_auth_user="", basic_auth_pass=""))
    app_reg2.config["REGIME_CTX_BUILDER"] = lambda: {
        "trend": "up", "momentum": 0.05, "realized_vol": 0.012}
    rr1 = app_reg2.test_client().get("/api/regime")
    jr = rr1.get_json()
    if rr1.status_code != 200 or jr.get("verdict") == VERDICT_INSUFFICIENT \
            or not jr.get("label"):
        print("FAIL: regime with ctx should be OK with a label:", jr); ok = False

    # 3. A raising provider degrades to verdict=ERROR, still 200.
    err = _safe_provider(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    if err.get("verdict") != VERDICT_ERROR or "boom" not in err.get("error", ""):
        print("FAIL: raising provider should yield verdict=ERROR:", err); ok = False

    # 4. Basic auth: 401 without creds, 200 with, when configured.
    authed = create_app(DashboardConfig(host="127.0.0.1", port=0, cache_ttl=0,
                                        basic_auth_user="u", basic_auth_pass="p"))
    ac = authed.test_client()
    if ac.get("/api/daily").status_code != 401:
        print("FAIL: protected endpoint should 401 without creds"); ok = False
    import base64
    tok = base64.b64encode(b"u:p").decode()
    if ac.get("/api/daily", headers={"Authorization": f"Basic {tok}"}
              ).status_code != 200:
        print("FAIL: valid creds should pass"); ok = False
    if ac.get("/api/health").status_code != 200:
        print("FAIL: health should stay open under auth"); ok = False

    print("oracle_dashboard self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        return _self_test()
    cfg = DashboardConfig.from_env()
    app = create_app(cfg)
    from waitress import serve
    print(f"Oracle dashboard serving on http://{cfg.host}:{cfg.port} "
          f"(auth={'on' if cfg.auth_enabled else 'off'})")
    serve(app, host=cfg.host, port=cfg.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
