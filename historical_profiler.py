"""
P13C — Historical setup profiler ("Oracle's historical memory").

Aggregates every closed setup into a profile keyed by the shared setup key
(regime, volatility, direction, signal_strength, [pattern], dte_bucket,
delta_bucket). For each key it stores count, win_rate, avg_pnl, avg_pnl_pct,
avg_ev, avg_holding_time, profit_factor and max_loss_observed, plus a global
roll-up. The result is cached to a local JSON file with an ATOMIC write so a
later run can read it back without re-scanning every store.

Strictly ANALYTICS / READ-ONLY: it never opens, sizes, prices, blocks or alters
any real or paper trade and never reaches the network beyond the fail-open
loaders. The single cache write is itself fail-open — if it fails, the in-memory
profile is still returned. Records can be injected for deterministic testing.
"""

import os
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import ev_attribution as eva
import feature_buckets as fb
import learned_edge as le
import oracle_analytics as oa
from config_loader import ConfigLoader

SCHEMA_VERSION = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ProfilerConfig:
    profile_file: str = "oracle_setup_profile.json"
    spread_trades_file: str = "spread_paper_trades.json"
    trade_history_file: str = "trading_history.json"
    episode_db_file: str = "episodes.db"

    @staticmethod
    def from_env(path: str = ".env",
                 loader: Optional[ConfigLoader] = None) -> "ProfilerConfig":
        try:
            cfg = loader if loader is not None else ConfigLoader(path=path)
            return ProfilerConfig(
                profile_file=cfg.get_str("ORACLE_SETUP_PROFILE_FILE",
                                         "oracle_setup_profile.json"),
                spread_trades_file=cfg.get_str("SPREAD_PAPER_TRADES_FILE",
                                               "spread_paper_trades.json"),
                trade_history_file=cfg.get_str("TRADE_HISTORY_FILE",
                                               "trading_history.json"),
                episode_db_file=cfg.get_str("EPISODE_DB_FILE", "episodes.db"),
            )
        except Exception:  # pragma: no cover - fail-open
            return ProfilerConfig()


def _le_config(config: ProfilerConfig) -> le.LearnedEdgeConfig:
    return le.LearnedEdgeConfig(
        spread_trades_file=config.spread_trades_file,
        trade_history_file=config.trade_history_file,
        episode_db_file=config.episode_db_file,
    )


def _profile_block(rows: List[dict]) -> dict:
    """The per-setup stat block over a cohort of closed records."""
    n = len(rows)
    wins = sum(1 for r in rows if oa._is_win(r))
    bstats = eva.bucket_stats(rows)
    pnls = [oa._trade_pnl(r) for r in rows]
    pnl_pcts = [oa._trade_pnl_pct(r) for r in rows]
    evs = [eva._ev(r) for r in rows]
    holds = [le._hold_days(r) for r in rows]
    return {
        "count": n,
        "win_rate": round(wins / n, 4) if n else 0.0,
        "avg_pnl": round(bstats.get("average_pnl", 0.0), 2),
        "avg_pnl_pct": round(le._mean(pnl_pcts), 4) if le._mean(pnl_pcts) is not None else None,
        "avg_ev": round(le._mean(evs), 4) if le._mean(evs) is not None else None,
        "avg_holding_time": round(le._mean(holds), 4) if le._mean(holds) is not None else None,
        "profit_factor": bstats.get("profit_factor"),
        "max_loss_observed": bstats.get("max_loss_observed", 0.0),
    }


def compute_profile(records: Optional[List[dict]] = None,
                    config: Optional[ProfilerConfig] = None) -> dict:
    """Build the setup-keyed profile + global roll-up. Never raises.

    Pass ``records`` to stay pure/offline; otherwise they are loaded fail-open."""
    try:
        cfg = config or ProfilerConfig.from_env()
        if records is None:
            records = le.load_edge_records(_le_config(cfg))
        records = records or []

        groups: Dict[str, List[dict]] = {}
        for r in records:
            try:
                key = fb.make_setup_key(r)
                ks = fb.setup_key_str(key)
            except Exception:
                continue
            groups.setdefault(ks, []).append(r)

        profiles = {ks: _profile_block(rows) for ks, rows in groups.items()}
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": _now_iso(),
            "global": _profile_block(records),
            "profiles": profiles,
        }
    except Exception:  # pragma: no cover - fail-open
        return {
            "schema_version": SCHEMA_VERSION, "generated_at": _now_iso(),
            "global": _profile_block([]), "profiles": {},
        }


def _atomic_write_json(path: str, payload: dict) -> bool:
    """Write JSON via a temp file + os.replace. Fail-open (returns success)."""
    if not path:
        return False
    try:
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
        return True
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return False


def save_profile(profile: dict, config: Optional[ProfilerConfig] = None) -> bool:
    cfg = config or ProfilerConfig.from_env()
    return _atomic_write_json(cfg.profile_file, profile)


def load_or_build_profile(config: Optional[ProfilerConfig] = None,
                          records: Optional[List[dict]] = None,
                          rebuild: bool = True) -> dict:
    """Return a cached profile if present (and not rebuilding), else build,
    cache (fail-open) and return. Never raises."""
    cfg = config or ProfilerConfig.from_env()
    if not rebuild and records is None:
        cached = oa.read_json(cfg.profile_file)
        if isinstance(cached, dict) and cached.get("profiles") is not None:
            return cached
    profile = compute_profile(records=records, config=cfg)
    if records is None:               # only persist disk-sourced profiles
        save_profile(profile, cfg)
    return profile


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network — injected records + temp cache file)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    records = (
        [le._rec("trending", "up", 0.20, i % 4 != 0, rid=f"a{i}")
         for i in range(8)]
        + [le._rec("ranging", "down", 0.10, i % 2 == 0, rid=f"b{i}",
                   pattern="hammer") for i in range(6)]
    )
    prof = compute_profile(records=records)

    if prof["schema_version"] != SCHEMA_VERSION:
        print("FAIL: schema version"); ok = False
    if prof["global"]["count"] != 14:
        print("FAIL: global count", prof["global"]["count"]); ok = False
    if not prof["profiles"]:
        print("FAIL: profiles empty"); ok = False

    # The pattern-stamped cohort must carry a pattern dimension in its key.
    if not any("pattern=hammer" in ks for ks in prof["profiles"]):
        print("FAIL: pattern key missing", list(prof["profiles"])); ok = False
    # The non-pattern cohort omits the pattern dimension entirely.
    if not any("pattern=" not in ks and "regime=trending" in ks
               for ks in prof["profiles"]):
        print("FAIL: non-pattern key should omit pattern"); ok = False

    # Empty records -> clean, no profiles, never raises.
    empty = compute_profile(records=[])
    if empty["profiles"] or empty["global"]["count"] != 0:
        print("FAIL: empty profile not clean", empty); ok = False

    # Atomic cache write + readback round-trips.
    tmp_path = f"_profiler_selftest_{os.getpid()}.json"
    cfg = ProfilerConfig(profile_file=tmp_path)
    try:
        if not save_profile(prof, cfg):
            print("FAIL: save_profile reported failure"); ok = False
        back = oa.read_json(tmp_path)
        if not isinstance(back, dict) or back["global"]["count"] != 14:
            print("FAIL: cache round-trip", back); ok = False
        # No stray temp file left behind.
        if any(f.startswith(f"{tmp_path}.tmp") for f in os.listdir(".")):
            print("FAIL: temp file left behind"); ok = False
    finally:
        for f in (tmp_path,):
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass

    # Bad write target fails open (returns False, does not raise).
    if save_profile(prof, ProfilerConfig(profile_file="")) is not False:
        print("FAIL: empty path should report False"); ok = False

    print("historical_profiler self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
