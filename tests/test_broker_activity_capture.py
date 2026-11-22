"""Read-only activity ingestion keeps identity, account scope, and late fees."""
import csv
import json
from datetime import datetime, timezone

import pytest

from broker_activity_capture import _open_store, collect_once, export_csv, fetch_page_http


AFTER = "2026-11-20T00:00:00+00:00"


def test_late_fee_and_repeated_fill_are_idempotent(tmp_path):
    path = tmp_path / "a.sqlite"
    fill = {"id": "fill-1", "activity_type": "FILL", "order_id": "ord-1",
            "symbol": "SPY261120C00700000", "qty": "1", "price": "2.5", "side": "buy"}
    fee = {"id": "fee-1", "activity_type": "FEE", "net_amount": "-0.02"}
    conn = _open_store(str(path), "paper")
    try:
        first = collect_once(conn, lambda after, token: [fill], after=AFTER)
        second = collect_once(conn, lambda after, token: [fill, fee], after=AFTER)
        assert first["inserted"] == 1
        assert second["inserted"] == 1
        assert second["fee_without_order_id"] == 1
        output = tmp_path / "activities.csv"
        export_csv(conn, str(output))
        with output.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 2
        assert {r["id"] for r in rows} == {"fill-1", "fee-1"}
        assert next(r for r in rows if r["id"] == "fee-1")["order_id"] == ""
    finally:
        conn.close()
    with pytest.raises(ValueError, match="scope mismatch"):
        _open_store(str(path), "live")


def test_pages_use_last_activity_id_as_token(tmp_path):
    conn = _open_store(str(tmp_path / "a.sqlite"), "paper")
    calls = []

    def fetch(after, token):
        calls.append((after, token))
        if token is None:
            return [{"id": f"a-{i}", "activity_type": "FILL"}
                    for i in range(100)]
        assert token == "a-99"
        return [{"id": "fee-later", "activity_type": "FEE",
                 "order_id": "ord-1", "net_amount": "-0.01"}]

    try:
        stats = collect_once(conn, fetch, after=AFTER)
        assert stats["pages"] == 2 and stats["inserted"] == 101
        assert calls[1][1] == "a-99"
    finally:
        conn.close()


def test_bad_pagination_does_not_report_completion(tmp_path):
    conn = _open_store(str(tmp_path / "a.sqlite"), "paper")
    try:
        with pytest.raises(RuntimeError, match="stalled"):
            collect_once(conn, lambda after, token: [
                {"id": "same", "activity_type": "FILL"}] * 100,
                after=AFTER)
    finally:
        conn.close()


def test_fetch_is_get_and_keeps_auth_in_headers_only(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key-not-in-url")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret-not-in-url")
    observed = {}

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self, size=-1):
            return b"[]"

    def opener(req, timeout):
        observed["method"] = req.get_method()
        observed["url"] = req.full_url
        observed["headers"] = req.headers
        return Response()

    assert fetch_page_http("paper", AFTER, opener=opener) == []
    assert observed["method"] == "GET"
    assert "key-not-in-url" not in observed["url"]
    assert "secret-not-in-url" not in observed["url"]
    assert observed["headers"]["Apca-api-key-id"] == "key-not-in-url"
