"""Canonical read-only historical bars cache for Oracle 4.0.1 (Stage A).

Replaces the timestamp-less daily-close + synthetic-uniform-grid pathway
(``valuation._step_returns`` / ``shadow._history_obs``) with genuine
timestamped, UTC-normalized OHLCV bars.

Rules (per the Oracle 4.0.1 master prompt, Stage A):
  * every bar carries a real UTC timestamp (bar START) and an explicit
    resolution; a bar is *available* only once it has completed
    (``ts_utc + resolution <= as_of``) -- never mid-bar, never late-arriving;
  * daily and intraday series are separate resolutions; nothing here ever
    rescales a daily series onto an intraday grid;
  * when data is unavailable the loader raises ``BarsUnavailable`` with a
    structured, counted reason -- it never invents values;
  * within-session gaps are bridged only when validated and bounded
    (a single return across <= ``max_gap_bars`` missing bars); larger gaps
    break the return chain and are counted;
  * overnight (cross-session) returns are kept distinct from intraday
    returns -- they are never mixed into one pool;
  * sessions are defined on the America/New_York exchange calendar
    (zoneinfo handles DST); holidays are absent sessions; early closes are
    observed from the data and labeled, never padded;
  * every ingest writes a manifest with the source file SHA-256, timezone
    mapping, coverage and drop counts.

No production module imports this file; the store is opened read-only for
all consumers (only the ingest CLI writes).
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

__all__ = [
    "Bar",
    "BarsUnavailable",
    "BarStore",
    "ReturnSeries",
    "DEFAULT_DB",
    "ET",
    "RTH_OPEN",
    "RTH_CLOSE",
    "rth_window_utc",
    "session_date_of",
    "is_rth",
    "intraday_log_returns",
]

DEFAULT_DB = os.path.join("artifacts", "oracle4", "bars", "bars.sqlite")

ET = ZoneInfo("America/New_York")
UTC = timezone.utc

# Regular trading hours, America/New_York wall clock.
RTH_OPEN = (9, 30)
RTH_CLOSE = (16, 0)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    symbol        TEXT    NOT NULL,
    resolution_s  INTEGER NOT NULL,
    ts_utc        TEXT    NOT NULL,   -- bar START, ISO8601 Z
    open          REAL,
    high          REAL,
    low           REAL,
    close         REAL    NOT NULL,
    volume        REAL,
    session_date  TEXT    NOT NULL,   -- America/New_York trading date
    rth           INTEGER NOT NULL,   -- 1 = inside 09:30-16:00 ET
    source        TEXT    NOT NULL,
    PRIMARY KEY (symbol, resolution_s, ts_utc)
);
CREATE INDEX IF NOT EXISTS idx_bars_session
    ON bars (symbol, resolution_s, session_date);
CREATE TABLE IF NOT EXISTS ingest_manifests (
    source        TEXT NOT NULL,
    ingested_at   TEXT NOT NULL,
    manifest_json TEXT NOT NULL
);
"""


# --------------------------------------------------------------------------- #
# value objects
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Bar:
    symbol: str
    resolution_s: int
    ts_utc: datetime            # bar START, tz-aware UTC
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: float
    volume: Optional[float]
    session_date: str           # YYYY-MM-DD (America/New_York)
    rth: bool
    source: str

    @property
    def end_utc(self) -> datetime:
        return self.ts_utc + timedelta(seconds=self.resolution_s)

    def completed_by(self, as_of: datetime) -> bool:
        """A bar exists for PIT purposes only once it has fully closed."""
        return self.end_utc <= as_of


class BarsUnavailable(Exception):
    """Structured refusal: the loader will not guess.

    reason is one of:
      NO_COVERAGE            -- no bars for symbol/resolution in the window
      SYMBOL_UNKNOWN         -- symbol absent from the store entirely
      RESOLUTION_UNAVAILABLE -- symbol present, but not at this resolution
      STALE                  -- newest completed bar older than tolerance
      INSUFFICIENT_HISTORY   -- fewer bars than the caller's minimum
      STORE_MISSING          -- the sqlite store file does not exist
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)

    def to_dict(self) -> Dict[str, str]:
        return {"reason": self.reason, "detail": self.detail}


@dataclass
class ReturnSeries:
    """Session-aware log returns built from real bar timestamps."""
    intraday: List[float] = field(default_factory=list)
    overnight: List[float] = field(default_factory=list)
    n_bars: int = 0
    n_sessions: int = 0
    n_gap_bridged: int = 0      # validated small gaps bridged by one return
    n_chain_breaks: int = 0     # gaps too large: chain broken, counted
    resolution_s: int = 0


# --------------------------------------------------------------------------- #
# calendar / session helpers (zoneinfo handles DST)
# --------------------------------------------------------------------------- #

def session_date_of(ts_utc: datetime) -> str:
    """Exchange session date (ET calendar date) for a UTC instant."""
    return ts_utc.astimezone(ET).date().isoformat()


def rth_window_utc(session: date) -> Tuple[datetime, datetime]:
    """UTC [open, close) for a session's regular hours. DST-correct."""
    open_et = datetime(session.year, session.month, session.day,
                       RTH_OPEN[0], RTH_OPEN[1], tzinfo=ET)
    close_et = datetime(session.year, session.month, session.day,
                        RTH_CLOSE[0], RTH_CLOSE[1], tzinfo=ET)
    return open_et.astimezone(UTC), close_et.astimezone(UTC)


def is_rth(ts_utc: datetime) -> bool:
    """True when the bar START falls inside 09:30-16:00 ET."""
    et = ts_utc.astimezone(ET)
    o, c = rth_window_utc(et.date())
    return o <= ts_utc < c


def _parse_utc(text: str) -> datetime:
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# store
# --------------------------------------------------------------------------- #

class BarStore:
    """SQLite-backed bars cache. Consumers open read-only; ingest writes."""

    def __init__(self, path: str = DEFAULT_DB, readonly: bool = True) -> None:
        self.path = path
        self.readonly = readonly
        if readonly:
            if not os.path.exists(path):
                raise BarsUnavailable("STORE_MISSING", path)
            self.conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        else:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self.conn = sqlite3.connect(path)
            self.conn.executescript(_SCHEMA)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "BarStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- write path (ingest only) ----------------------------------------- #

    def insert_bars(self, bars: Iterable[Bar]) -> Dict[str, int]:
        """Keep-first dedupe on (symbol, resolution, ts). Counts, never merges."""
        if self.readonly:
            raise RuntimeError("store opened read-only")
        inserted = dup = 0
        cur = self.conn.cursor()
        for b in bars:
            res = cur.execute(
                "INSERT OR IGNORE INTO bars "
                "(symbol, resolution_s, ts_utc, open, high, low, close,"
                " volume, session_date, rth, source) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (b.symbol, b.resolution_s, _iso(b.ts_utc), b.open, b.high,
                 b.low, b.close, b.volume, b.session_date,
                 1 if b.rth else 0, b.source))
            if res.rowcount == 1:
                inserted += 1
            else:
                dup += 1
        self.conn.commit()
        return {"inserted": inserted, "dropped_duplicate_ts": dup}

    def write_manifest(self, source: str, manifest: Dict) -> None:
        if self.readonly:
            raise RuntimeError("store opened read-only")
        self.conn.execute(
            "INSERT INTO ingest_manifests (source, ingested_at, manifest_json)"
            " VALUES (?,?,?)",
            (source, _iso(datetime.now(UTC)), json.dumps(manifest, sort_keys=True)))
        self.conn.commit()

    # ---- read path (PIT-safe) --------------------------------------------- #

    def load_bars(
        self,
        symbol: str,
        resolution_s: int,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        as_of: Optional[datetime] = None,
        rth_only: bool = True,
        min_bars: int = 0,
        max_staleness: Optional[timedelta] = None,
    ) -> List[Bar]:
        """Completed bars only, ordered by timestamp.

        ``as_of`` enforces the completed-bar rule: a bar whose END is after
        ``as_of`` does not exist yet.  ``max_staleness`` (measured from
        ``as_of`` back to the newest completed bar END) turns silent gaps
        into a counted STALE refusal.
        """
        q = ("SELECT * FROM bars WHERE symbol=? AND resolution_s=?")
        args: List = [symbol, resolution_s]
        if start is not None:
            q += " AND ts_utc >= ?"
            args.append(_iso(start))
        if end is not None:
            q += " AND ts_utc < ?"
            args.append(_iso(end))
        if as_of is not None:
            # completed-bar rule: bar start <= as_of - resolution
            q += " AND ts_utc <= ?"
            args.append(_iso(as_of - timedelta(seconds=resolution_s)))
        if rth_only:
            q += " AND rth = 1"
        q += " ORDER BY ts_utc"
        rows = self.conn.execute(q, args).fetchall()
        if not rows:
            self._explain_empty(symbol, resolution_s)
        bars = [self._row_to_bar(r) for r in rows]
        if as_of is not None and max_staleness is not None:
            newest_end = bars[-1].end_utc
            if as_of - newest_end > max_staleness:
                raise BarsUnavailable(
                    "STALE",
                    f"newest completed bar ends {_iso(newest_end)}, "
                    f"as_of {_iso(as_of)} exceeds tolerance "
                    f"{max_staleness.total_seconds():.0f}s")
        if min_bars and len(bars) < min_bars:
            raise BarsUnavailable(
                "INSUFFICIENT_HISTORY",
                f"{len(bars)} bars < required {min_bars}")
        return bars

    def _explain_empty(self, symbol: str, resolution_s: int) -> None:
        row = self.conn.execute(
            "SELECT COUNT(*) c FROM bars WHERE symbol=?", (symbol,)).fetchone()
        if row["c"] == 0:
            raise BarsUnavailable("SYMBOL_UNKNOWN", symbol)
        row = self.conn.execute(
            "SELECT COUNT(*) c FROM bars WHERE symbol=? AND resolution_s=?",
            (symbol, resolution_s)).fetchone()
        if row["c"] == 0:
            raise BarsUnavailable(
                "RESOLUTION_UNAVAILABLE", f"{symbol}@{resolution_s}s")
        raise BarsUnavailable(
            "NO_COVERAGE", f"{symbol}@{resolution_s}s in requested window")

    @staticmethod
    def _row_to_bar(r: sqlite3.Row) -> Bar:
        return Bar(
            symbol=r["symbol"], resolution_s=r["resolution_s"],
            ts_utc=_parse_utc(r["ts_utc"]), open=r["open"], high=r["high"],
            low=r["low"], close=r["close"], volume=r["volume"],
            session_date=r["session_date"], rth=bool(r["rth"]),
            source=r["source"])

    def coverage(self, symbol: str, resolution_s: int) -> Dict:
        """Coverage summary incl. observed early closes (never padded)."""
        rows = self.conn.execute(
            "SELECT session_date, COUNT(*) n, MIN(ts_utc) lo, MAX(ts_utc) hi "
            "FROM bars WHERE symbol=? AND resolution_s=? AND rth=1 "
            "GROUP BY session_date ORDER BY session_date",
            (symbol, resolution_s)).fetchall()
        if not rows:
            return {"sessions": 0, "bars": 0}
        early = []
        for r in rows:
            last_start = _parse_utc(r["hi"])
            _, close_utc = rth_window_utc(date.fromisoformat(r["session_date"]))
            if close_utc - (last_start + timedelta(seconds=resolution_s)) \
                    > timedelta(minutes=30):
                early.append(r["session_date"])
        return {
            "sessions": len(rows),
            "bars": sum(r["n"] for r in rows),
            "first_session": rows[0]["session_date"],
            "last_session": rows[-1]["session_date"],
            "early_close_observed": early[:50],
            "n_early_close_observed": len(early),
        }


# --------------------------------------------------------------------------- #
# session-aware returns (no synthetic grid, ever)
# --------------------------------------------------------------------------- #

def intraday_log_returns(
    bars: Sequence[Bar],
    max_gap_bars: int = 5,
) -> ReturnSeries:
    """Log returns from real timestamps, split intraday vs overnight.

    * consecutive bars in the same session, gap == resolution -> intraday;
    * same session, gap of 2..max_gap_bars resolutions -> one bridged
      return across the validated gap (counted; no values invented);
    * same session, larger gap -> chain break (counted), no return emitted;
    * session change -> overnight return (close-to-first-close), kept in a
      separate pool.
    """
    import math

    out = ReturnSeries()
    if not bars:
        return out
    out.resolution_s = bars[0].resolution_s
    out.n_bars = len(bars)
    sessions = set()
    prev: Optional[Bar] = None
    for b in bars:
        sessions.add(b.session_date)
        if prev is not None and prev.close > 0 and b.close > 0:
            r = math.log(b.close / prev.close)
            if b.session_date != prev.session_date:
                out.overnight.append(r)
            else:
                gap = (b.ts_utc - prev.ts_utc).total_seconds()
                n_steps = round(gap / b.resolution_s)
                if n_steps == 1:
                    out.intraday.append(r)
                elif 2 <= n_steps <= max_gap_bars:
                    out.intraday.append(r)
                    out.n_gap_bridged += 1
                else:
                    out.n_chain_breaks += 1
        prev = b
    out.n_sessions = len(sessions)
    return out
