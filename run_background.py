"""
Run Telegram bot in background with auto-restart
"""

import subprocess
import time
import sys
import os

def run_bot():
    """Run bot with auto-restart on crash"""
    print("[SERVICE] Starting Telegram trading bot service...")

    while True:
        try:
            # Start the bot
            print(f"[{time.strftime('%H:%M:%S')}] Starting bot...")

            process = subprocess.Popen([
                sys.executable, 'telegram_bot.py'
            ], cwd=os.path.dirname(__file__))

            # Wait for process to finish
            process.wait()

            print(f"[{time.strftime('%H:%M:%S')}] Bot stopped. Restarting in 5 seconds...")
            time.sleep(5)

        except KeyboardInterrupt:
            print("\n[SERVICE] Service stopped by user")
            if process:
                process.terminate()
            break
        except Exception as e:
            print(f"[ERROR] {e}. Restarting in 10 seconds...")
            time.sleep(10)

if __name__ == "__main__":
    run_bot()