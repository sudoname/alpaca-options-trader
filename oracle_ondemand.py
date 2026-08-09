#!/usr/bin/env python3
"""On-demand Oracle analysis for a single underlying symbol (Telegram).

Bridges a live, read-only market context into the SAME evidence + explain
producers the shadow path uses, so a user can ask Oracle to analyze any symbol
(e.g. right after ADD_SYMBOL, or via ORACLE_EXPLAIN).

STRICTLY ADVISORY / READ-ONLY. This module never places, sizes, blocks, or
closes a trade, never records to episodes.db, and never flips a flag. It only
reads market data (the same reads ANALYZE already does) and formats a report.
Every step fails open: a missing slice degrades to "n/a", never an exception.

Public surface:
  build_symbol_ctx(symbol, trader=None) -> dict
      Assemble a live ctx: the underlying market slice via explain_context,
      enriched with a representative option (direction from
      determine_option_strategy, contract from select_best_option) through the
      trader's own _build_evidence_ctx so the slate matches production.
  analyze_symbol_text(symbol, trader=None, ctx=None) -> str
      Compute the Oracle explain (P(call)/P(put) + agent contributions) and the
      full Oracle 2.1 evidence slate over that ctx, formatted for Telegram.
      When ctx is supplied it is used as-is (the offline path for the self-test).
"""

from typing import Optional


# --------------------------------------------------------------------------- #
# small, local formatting helpers (kept independent of the analytics module)
# --------------------------------------------------------------------------- #
def _pct(x) -> str:
    try:
        return f"{float(x) * 100:.0f}%"
    except (TypeError, ValueError):
        return "n/a"


def _num(x, fmt="{:.2f}") -> str:
    try:
        return fmt.format(float(x))
    except (TypeError, ValueError):
        return "n/a"


def _yn(x) -> str:
    if x is True:
        return "yes"
    if x is False:
        return "no"
    return "n/a"


def _d(evidence: dict, key: str) -> dict:
    """evidence[key] as a dict, or {} when absent/malformed."""
    v = evidence.get(key) if isinstance(evidence, dict) else None
    return v if isinstance(v, dict) else {}


# --------------------------------------------------------------------------- #
# live context assembly (read-only)
# --------------------------------------------------------------------------- #
def build_symbol_ctx(symbol: str, trader=None) -> dict:
    """Build a live evidence ctx for ``symbol``. Never raises; returns {} when
    no data/creds are available so the caller can degrade gracefully.

    Mirrors the production path: underlying market slice (trend / momentum /
    realized_vol / regime / volume_ratio / rel_strength / candlestick) from
    explain_context, then a representative option (direction + contract) folded
    in through the trader's own _build_evidence_ctx, so the resulting slate is
    identical in shape to what the shadow recorder logs on a real fill.
    """
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return {}

    # 1) representative option + a trader to enrich the ctx (best effort).
    option: dict = {}
    try:
        if trader is None:
            from smart_trader import SmartOptionsTrader
            trader = SmartOptionsTrader(ticker=symbol)
    except Exception:
        trader = None

    if trader is not None:
        try:
            strat = trader.determine_option_strategy(symbol)
        except Exception:
            strat = None
        side = str(strat).lower() if strat else None
        if side not in ("call", "put"):
            side = "call"  # neutral default; direction is evidence, not a trigger
        try:
            price = trader.get_current_price(symbol)
        except Exception:
            price = None
        try:
            contracts = trader.get_option_contracts(symbol)
            if contracts and price:
                option = trader.select_best_option(
                    contracts, price, strategy=side) or {}
        except Exception:
            option = {}
        # If contract lookup failed, still record the direction the model chose
        # so the direction-dependent slices (thesis/extension/repricing) engage.
        if not option:
            option = {"type": side}

    # 2) fold into the production ctx builder when we have a trader; otherwise
    #    fall back to the underlying-only explain context.
    ctx: dict = {}
    try:
        if trader is not None and hasattr(trader, "_build_evidence_ctx"):
            ctx = trader._build_evidence_ctx(symbol, option, {}) or {}
        else:
            import explain_context
            ctx = explain_context.build_explain_context(symbol) or {}
    except Exception:
        ctx = {}
    return ctx


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def analyze_symbol_text(symbol: str, trader=None,
                        ctx: Optional[dict] = None) -> str:
    """Return a Telegram-formatted live Oracle analysis for ``symbol``.

    ``ctx`` may be supplied to bypass the live build (used by the offline
    self-test). Never raises.
    """
    symbol = (symbol or "").strip().upper() or "?"
    try:
        if ctx is None:
            ctx = build_symbol_ctx(symbol, trader)
        ctx = ctx if isinstance(ctx, dict) else {}

        header = f"🔮 *Oracle Analysis — {symbol}* _(live · advisory)_"
        if not ctx:
            return "\n".join([
                header, "",
                "No live market context available for this symbol right now "
                "(market data/creds unavailable).",
                "*Verdict:* `INSUFFICIENT_DATA`",
                "", "_Advisory only — no order placed, nothing recorded._"])

        # --- explain: agent votes -> P(call)/P(put) + attribution ---------- #
        try:
            import oracle_intelligence_reports as oir
            report = oir.compute_oracle_explain(symbol, ctx=ctx)
        except Exception:
            report = {}
        prob = (report or {}).get("probability", {}) or {}
        expl = (report or {}).get("explanation", {}) or {}

        # --- full Oracle 2.1 evidence slate -------------------------------- #
        try:
            import evidence_context
            evidence = evidence_context.compute_evidence(ctx) or {}
        except Exception:
            evidence = {}

        o21 = _d(evidence, "oracle21")
        em = _d(evidence, "expected_move")
        thesis = _d(evidence, "thesis")
        conv = _d(evidence, "conviction")
        ext = _d(evidence, "extension")
        rep = _d(evidence, "repricing")
        cat = _d(evidence, "catalyst")

        side = "CALL" if str(ctx.get("direction")).lower() == "up" else (
            "PUT" if str(ctx.get("direction")).lower() == "down" else "—")

        lines = [header, ""]
        lines.append(f"*Direction model:* `{side}`")
        lines.append(
            f"*P(call)* `{_pct(prob.get('p_call'))}` · "
            f"*P(put)* `{_pct(prob.get('p_put'))}` · "
            f"*P(no-trade)* `{_pct(prob.get('p_no_trade'))}`")
        lines.append("")

        contrib = expl.get("agent_contributions", {}) or {}
        if contrib:
            lines.append("*Top agent contributions:*")
            for name, share in sorted(contrib.items(), key=lambda kv: kv[1],
                                      reverse=True)[:5]:
                lines.append(f"`{name}` {_pct(share)}")
            lines.append("")
        if expl.get("top_reasons"):
            lines.append("*Top reasons:*")
            for r in expl["top_reasons"][:4]:
                lines.append(f"• {r}")
            lines.append("")

        ver = evidence.get("evidence_version") or o21.get("version") or "?"
        lines.append(f"*Oracle 2.1 slate* _(v{ver})_:")
        mode = thesis.get("mode") or o21.get("mode")
        lines.append(f"• Mode: `{mode or 'n/a'}`")
        cval = conv.get("conviction")
        if cval is None:
            cval = o21.get("conviction")
        tier = conv.get("tier") or o21.get("conviction_tier")
        lines.append(
            f"• Conviction: `{_num(cval)}`"
            + (f" (tier `{tier}`)" if tier else ""))
        emp = em.get("sigma1_pct")
        if emp is None:
            emp = thesis.get("expected_move_pct")
        lines.append(f"• Expected move (1σ): `±{_num(emp, '{:.1f}')}%`"
                     if emp is not None else "• Expected move (1σ): `n/a`")
        lines.append(f"• Extension (chase): `{_yn(ext.get('extended'))}`")
        lines.append(f"• Repricing (pullback): `{_yn(rep.get('opportunity'))}`")
        sev = cat.get("severity")
        ctype = cat.get("catalyst_type")
        lines.append(
            f"• Catalyst: `{ctype}` (severity `{_num(sev)}`)" if ctype
            else "• Catalyst: `none`")
        inv = thesis.get("invalidation_pct")
        decay = thesis.get("decay_horizon_days")
        if inv is not None or decay is not None:
            lines.append(
                f"• Thesis: invalidation `{_num(inv, '{:.1f}')}%`, "
                f"decay `{_num(decay, '{:.0f}')}d`")

        lines += ["", "_Advisory only — no order placed, nothing recorded. "
                  "Reply_ `" + symbol + "` _for the tradeable contract + Greeks._"]
        return "\n".join(lines)
    except Exception as e:  # pragma: no cover - absolute fail-open
        return (f"🔮 *Oracle Analysis — {symbol}*\n\n"
                f"❌ Could not build analysis: {e}")


# --------------------------------------------------------------------------- #
# no-network self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Synthetic ctx identical in spirit to the live-fill ctx, so compute_evidence
    # produces the full 2.1 slate WITHOUT any network / trader / DB.
    demo = {
        "regime": "trending", "trend": "up", "momentum": 0.08,
        "realized_vol": 0.18, "vix": 16.0, "volume_ratio": 1.6,
        "news_score": 0.5, "news_count": 6, "iv_rank": 40.0,
        "signal_strength": 3, "dte": 30, "delta": 0.45, "direction": "up",
        "iv": 0.35, "hv": 0.20, "mode": "swing", "price": 100.0,
    }
    import sys

    def _safe(s):
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        return s.encode(enc, "replace").decode(enc)

    txt = analyze_symbol_text("TESTX", ctx=demo)
    print(_safe(txt))
    print("\n" + "=" * 60)
    ok = ("Oracle Analysis" in txt and "2.1 slate" in txt
          and "P(call)" in txt and "Mode:" in txt)
    # empty ctx must fail open to a clean INSUFFICIENT_DATA report
    empty = analyze_symbol_text("TESTX", ctx={})
    ok_empty = "INSUFFICIENT_DATA" in empty and "Advisory only" in empty
    print("SELF-TEST:", "PASS" if (ok and ok_empty) else "FAIL")
    sys.exit(0 if (ok and ok_empty) else 1)
