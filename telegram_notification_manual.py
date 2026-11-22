"""
Test Telegram Notification
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
chat_id = os.getenv('TELEGRAM_CHAT_ID', '')

if not bot_token or not chat_id:
    print("ERROR: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not found in .env")
    exit(1)

print(f"Bot Token: {bot_token[:20]}...")
print(f"Chat ID: {chat_id}")

message = """
*SPY 1DTE Strategy - Test Notification*

This is a test message to confirm Telegram integration is working.

If you received this, your SPY 1DTE strategy will send you:
✅ Trade entry notifications
✅ Progress updates (+10%, +15%)
✅ Trade exit notifications

Configuration: *READY*
"""

url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
data = {
    'chat_id': chat_id,
    'text': message,
    'parse_mode': 'Markdown'
}

print("\nSending test message...")
response = requests.post(url, data=data, timeout=10)

if response.status_code == 200:
    print("[SUCCESS] Test message sent to Telegram!")
    print(f"Check your Telegram app (Chat ID: {chat_id})")
else:
    print(f"[FAILED] Error: {response.status_code}")
    print(response.text)
