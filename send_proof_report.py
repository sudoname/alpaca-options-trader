"""
Daily ORACLE_PROOF_REPORT push.

Run by cron after the EOD close on trading days: generates the proof
report (best-EV performance verdict over closed, EV-stamped trades) and
pushes it straight to the owner's Telegram chat.

Usage (cron sources .env first for TELEGRAM_* and file paths):
    cd /var/www/alps && set -a && . ./.env && set +a && python3 send_proof_report.py

Exit codes: 0 sent, 1 not sent (missing creds / Telegram refused).
Generation itself fail-opens — an analytics error still sends a stub so
the absence of a report is never silent.
"""

import os
import sys

import requests


def build_text() -> str:
    try:
        from best_ev_performance import generate_oracle_proof_report_text
        return generate_oracle_proof_report_text()
    except Exception as e:  # never skip the send because analytics broke
        return f"ORACLE PROOF REPORT failed to generate: {e}"


def send(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in env")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    try:
        resp = requests.post(url, data=data, timeout=30)
        if resp.status_code == 200:
            return True
        # Mirror telegram_bot.send_message: unbalanced Markdown entities
        # get a 400 — retry once as plain text rather than drop the report.
        print(f"sendMessage HTTP {resp.status_code}: {resp.text[:200]}")
        data.pop("parse_mode", None)
        resp = requests.post(url, data=data, timeout=30)
        return resp.status_code == 200
    except Exception as e:
        print(f"sendMessage error: {e}")
        return False


def main() -> int:
    text = build_text()
    ok = send(text)
    print(f"proof report {'sent' if ok else 'NOT sent'} "
          f"({len(text)} chars)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
