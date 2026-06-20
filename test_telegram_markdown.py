"""
Offline regression lock for the Telegram send path's Markdown safety.

No creds, no network: ``requests`` is monkeypatched so nothing leaves the box.

Pins the fix for the production HTTP-400s ("can't parse entities" /
"message is too long") observed in /var/log/alps-bot.log:

  1. ``_markdown_balanced`` is a CONSERVATIVE legacy-Markdown validator. It
     flags not just odd delimiter counts but INTERLEAVING (``*NO_TRADE*``,
     ``flat_signal``) — the actual cause of the byte-offset-98 400 on the
     "NO_TRADE" notice. Clean, well-formed Markdown stays True.
  2. ``_post_message`` chooses ``parse_mode="Markdown"`` ONLY when the chunk is
     safe; otherwise it posts plain text UP-FRONT (one request, no 400, no log
     noise). A balanced chunk that is still rejected falls open to plain text
     on a single retry.
  3. ``send_message`` chunks over-long text under Telegram's 4096 cap, so the
     "message is too long" 400 cannot recur.
"""

import unittest

import telegram_bot
from telegram_bot import TelegramTradingBot


# The exact NO_TRADE notice analyze_ticker() builds (interleaved * and _).
def _no_trade_message(ticker="AAPL", reason="flat_signal (bull 2 == bear 2)"):
    return (f"\u23f8\ufe0f *{ticker}* — No trade.\n\n"
            f"The direction model returned *NO_TRADE* due to a "
            f"weak/flat signal ({reason}). Skipping contract "
            f"lookup and order.")


class _FakeResponse:
    def __init__(self, status_code, text="ok"):
        self.status_code = status_code
        self.text = text


class _FakeRequests:
    """Records every POST and replays a scripted sequence of responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []  # list of the `data` dict passed to each post

    def post(self, url, data=None, **kwargs):
        self.calls.append(dict(data or {}))
        return self._responses.pop(0) if self._responses else _FakeResponse(200)


class TestMarkdownBalanced(unittest.TestCase):
    mb = staticmethod(TelegramTradingBot._markdown_balanced)

    def test_clean_markdown_is_safe(self):
        for ok in ("", "hello world", "*SPY* is _up_ today",
                   "use `code` here", "pre ```block``` end",
                   "*a* _b_ `c`", "📋 *Daily Trading Summary*"):
            self.assertTrue(self.mb(ok), msg=repr(ok))

    def test_interleaving_is_unsafe(self):
        # The production offender: a `_` opening inside a `*` bold entity.
        self.assertFalse(self.mb("*NO_TRADE*"))
        self.assertFalse(self.mb(_no_trade_message()))

    def test_unbalanced_is_unsafe(self):
        for bad in ("start *SPY", "Closed bull_put_credit_spread +5%",
                    "trailing `code", "_lonely"):
            self.assertFalse(self.mb(bad), msg=repr(bad))

    def test_split_entity_per_chunk(self):
        # Chunking can split *bold* across a boundary; each half is unsafe.
        self.assertFalse(self.mb("foo *bar"))
        self.assertFalse(self.mb("baz* qux"))


class TestPostMessageParseMode(unittest.TestCase):
    def setUp(self):
        self.bot = TelegramTradingBot()
        self.bot.bot_token = "TEST"
        self._orig = telegram_bot.requests

    def tearDown(self):
        telegram_bot.requests = self._orig

    def test_safe_text_uses_markdown_single_post(self):
        fake = _FakeRequests([_FakeResponse(200)])
        telegram_bot.requests = fake
        self.assertTrue(self.bot._post_message("1", "*SPY* is _up_"))
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(fake.calls[0].get("parse_mode"), "Markdown")

    def test_unsafe_text_posts_plain_upfront(self):
        # No 400 round-trip: a single plain-text POST, no parse_mode.
        fake = _FakeRequests([_FakeResponse(200)])
        telegram_bot.requests = fake
        self.assertTrue(self.bot._post_message("1", _no_trade_message()))
        self.assertEqual(len(fake.calls), 1)
        self.assertNotIn("parse_mode", fake.calls[0])

    def test_balanced_but_rejected_falls_open_to_plain(self):
        # A safe-looking chunk Telegram still 400s (e.g. a bad link) -> retry.
        fake = _FakeRequests([_FakeResponse(400, "Bad Request"),
                              _FakeResponse(200)])
        telegram_bot.requests = fake
        self.assertTrue(self.bot._post_message("1", "see [x](y) ok"))
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(fake.calls[0].get("parse_mode"), "Markdown")
        self.assertNotIn("parse_mode", fake.calls[1])

    def test_unsafe_text_not_retried(self):
        # Plain from the start -> a single failing POST is not retried.
        fake = _FakeRequests([_FakeResponse(400, "Bad Request")])
        telegram_bot.requests = fake
        self.assertFalse(self.bot._post_message("1", "_lonely"))
        self.assertEqual(len(fake.calls), 1)


class TestSendMessageChunking(unittest.TestCase):
    def setUp(self):
        self.bot = TelegramTradingBot()
        self.bot.bot_token = "TEST"
        self.bot.chat_id = "1"
        self._orig = telegram_bot.requests

    def tearDown(self):
        telegram_bot.requests = self._orig

    def test_long_message_is_chunked_under_cap(self):
        fake = _FakeRequests([_FakeResponse(200)] * 10)
        telegram_bot.requests = fake
        long_text = "\n".join(f"line {i} " + "x" * 100 for i in range(200))
        self.assertTrue(self.bot.send_message(long_text))
        self.assertGreater(len(fake.calls), 1)
        for call in fake.calls:
            self.assertLessEqual(len(call["text"]),
                                 TelegramTradingBot.TELEGRAM_MAX_CHARS)


if __name__ == "__main__":
    unittest.main()
