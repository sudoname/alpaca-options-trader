"""
Phase 8A — Oracle training-dataset builder (advisory, offline-pure).

Accumulates a flat, append-only training table that pairs, for each decision:
    * all FEATURES that were available at decision time,
    * all PREDICTIONS the engine produced (expected moves, vol edge, oracle
      score, the shadow recommendation, etc.), and
    * the ACTUAL OUTCOME once the trade/idea resolves.

Rows are written to ``oracle_training_dataset.csv``. The builder is purely a
recorder — it makes NO trade decisions and touches no broker. Persistence is
best-effort and never raises; a corrupt/missing file behaves as empty.

Two-phase usage:
    rid = builder.log(features=..., predictions=..., symbol="SPY")   # outcome blank
    ...                                                              # later, on close
    builder.update_outcome(rid, {"pnl_pct": 0.42, "outcome": "win"})

Dict inputs are flattened with stable prefixes: ``feat_*`` / ``pred_*`` /
``out_*``. The CSV header is the union of all columns seen so far and is rewritten
when it grows, so callers may evolve their feature/prediction sets freely.
"""

import csv
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config_loader import ConfigLoader

logger = logging.getLogger(__name__)

META_FIELDS = ("row_id", "timestamp", "symbol")


@dataclass
class OracleDatasetConfig:
    enabled: bool = True
    dataset_file: str = "oracle_training_dataset.csv"

    @staticmethod
    def from_env(path: str = ".env",
                 loader: Optional[ConfigLoader] = None) -> "OracleDatasetConfig":
        cfg = loader if loader is not None else ConfigLoader(path=path)
        return OracleDatasetConfig(
            enabled=cfg.get_bool("ORACLE_DATASET_ENABLED", True),
            dataset_file=cfg.get_str("ORACLE_DATASET_FILE",
                                     "oracle_training_dataset.csv"),
        )


def _flatten(prefix: str, d: Optional[Dict]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    if not d:
        return out
    for k, v in d.items():
        # One level of nesting is enough for our payloads; deeper -> str().
        if isinstance(v, dict):
            for k2, v2 in v.items():
                out[f"{prefix}{k}_{k2}"] = v2
        else:
            out[f"{prefix}{k}"] = v
    return out


class OracleDatasetBuilder:
    def __init__(self, config: Optional[OracleDatasetConfig] = None):
        self.config = config or OracleDatasetConfig.from_env()

    # -- IO (best-effort) ------------------------------------------------- #
    def _path(self) -> str:
        return self.config.dataset_file

    def load_rows(self) -> List[dict]:
        path = self._path()
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return []
        try:
            with open(path, "r", newline="", encoding="utf-8") as f:
                return list(csv.DictReader(f))
        except Exception as exc:
            logger.warning("oracle dataset read failed (%s): %s", path, exc)
            return []

    def _write_all(self, rows: List[dict], fieldnames: List[str]) -> None:
        path = self._path()
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
        except Exception as exc:
            logger.warning("oracle dataset write failed (%s): %s", path, exc)

    @staticmethod
    def _ordered_fieldnames(rows: List[dict]) -> List[str]:
        seen: List[str] = list(META_FIELDS)
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.append(k)
        return seen

    # -- record ----------------------------------------------------------- #
    def log(self, features: Optional[Dict] = None,
            predictions: Optional[Dict] = None,
            outcome: Optional[Dict] = None,
            symbol: str = "", row_id: Optional[str] = None) -> str:
        """Append a row of features + predictions (+ optional outcome).

        Returns the generated ``row_id`` (use it later with update_outcome).
        """
        rid = row_id or uuid.uuid4().hex[:12]
        row: Dict[str, object] = {
            "row_id": rid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
        }
        row.update(_flatten("feat_", features))
        row.update(_flatten("pred_", predictions))
        row.update(_flatten("out_", outcome))

        rows = self.load_rows()
        rows.append(row)
        self._write_all(rows, self._ordered_fieldnames(rows))
        print(f"[ORACLE_DATASET] logged row_id={rid} sym={symbol or '?'} "
              f"feats={len(_flatten('feat_', features))} "
              f"preds={len(_flatten('pred_', predictions))} "
              f"outcome={'yes' if outcome else 'pending'}")
        return rid

    def update_outcome(self, row_id: str, outcome: Dict) -> bool:
        """Fill in the ``out_*`` columns for an existing row. Returns success."""
        rows = self.load_rows()
        out_cols = _flatten("out_", outcome)
        updated = False
        for r in rows:
            if r.get("row_id") == row_id:
                r.update(out_cols)
                updated = True
                break
        if not updated:
            return False
        self._write_all(rows, self._ordered_fieldnames(rows))
        print(f"[ORACLE_DATASET] outcome row_id={row_id} "
              f"fields={list(out_cols.keys())}")
        return True

    # -- stats (for the Telegram command) -------------------------------- #
    def stats(self) -> dict:
        rows = self.load_rows()
        total = len(rows)
        # A row "has outcome" if any out_* column is non-empty.
        out_keys = [k for r in rows for k in r.keys() if k.startswith("out_")]
        out_keys = list(dict.fromkeys(out_keys))
        with_outcome = [
            r for r in rows
            if any(str(r.get(k, "")).strip() not in ("", "None") for k in out_keys)
        ]
        pnls: List[float] = []
        for r in with_outcome:
            for key in ("out_pnl_pct", "out_net_pnl_pct", "out_pnl"):
                v = r.get(key)
                if v not in (None, "", "None"):
                    try:
                        pnls.append(float(v))
                        break
                    except (TypeError, ValueError):
                        pass
        wins = sum(1 for p in pnls if p > 0)
        n_out = len(with_outcome)
        return {
            "total_rows": total,
            "with_outcome": n_out,
            "pending": total - n_out,
            "completion_rate": round(n_out / total, 4) if total else 0.0,
            "n_with_pnl": len(pnls),
            "win_rate": round(wins / len(pnls), 4) if pnls else 0.0,
            "mean_pnl": round(sum(pnls) / len(pnls), 4) if pnls else 0.0,
            "dataset_file": self._path(),
        }


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network; temp file)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    import tempfile

    ok = True
    d = tempfile.mkdtemp()
    cfg = OracleDatasetConfig(enabled=True,
                              dataset_file=os.path.join(d, "ds.csv"))
    b = OracleDatasetBuilder(cfg)

    rid = b.log(features={"hv20": 0.2, "trend": "bullish"},
                predictions={"oracle_score": 72.0, "volatility_edge": 0.3},
                symbol="SPY")
    if b.stats()["total_rows"] != 1 or b.stats()["pending"] != 1:
        print("FAIL: initial stats", b.stats()); ok = False

    if not b.update_outcome(rid, {"pnl_pct": 0.5, "outcome": "win"}):
        print("FAIL: update_outcome"); ok = False
    if b.update_outcome("missing", {"pnl_pct": 1.0}):
        print("FAIL: update_outcome should miss"); ok = False

    # Second row with a loss; evolving columns (new feature) triggers rewrite.
    b.log(features={"hv20": 0.4, "trend": "bearish", "vix_regime": "high"},
          predictions={"oracle_score": 30.0},
          outcome={"pnl_pct": -0.3, "outcome": "loss"}, symbol="QQQ")

    s = b.stats()
    if s["total_rows"] != 2 or s["with_outcome"] != 2:
        print("FAIL: stats totals", s); ok = False
    if s["n_with_pnl"] != 2 or abs(s["win_rate"] - 0.5) > 1e-9:
        print("FAIL: win_rate", s); ok = False
    if abs(s["mean_pnl"] - 0.1) > 1e-9:
        print("FAIL: mean_pnl", s); ok = False

    # Header should include feat_/pred_/out_ columns and survive the rewrite.
    with open(cfg.dataset_file, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    for col in ("row_id", "feat_hv20", "pred_oracle_score", "out_pnl_pct",
                "feat_vix_regime"):
        if col not in header:
            print("FAIL: header missing", col, header); ok = False

    print("oracle_dataset_builder self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv or len(sys.argv) == 1:
        sys.exit(_self_test())
