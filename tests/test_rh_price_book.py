"""Login-flow tests for rh_price_book (fake robin_stocks, no network).

Covers the TOTP-optional / interactive-authentication rework:
  * headless (trade path) NEVER prompts or hits the network without a TOTP
    code or a stored session pickle;
  * any unexpected challenge prompt during a headless attempt dies fast
    (patched input) instead of hanging;
  * interactive --login mode allows SMS/device-approval prompts;
  * TOTP, when configured, is still passed through.
"""

import builtins
import sys
import types

import rh_price_book
from rh_price_book import (
    RH_SESSION_EXPIRES_SEC,
    RHPriceBookClient,
    _stored_session_files,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
def install_fake_rh(monkeypatch, login_fn):
    """Inject a fake robin_stocks.robinhood module exposing login_fn."""
    pkg = types.ModuleType("robin_stocks")
    mod = types.ModuleType("robin_stocks.robinhood")
    mod.login = login_fn
    pkg.robinhood = mod
    monkeypatch.setitem(sys.modules, "robin_stocks", pkg)
    monkeypatch.setitem(sys.modules, "robin_stocks.robinhood", mod)
    return mod


def install_fake_pyotp(monkeypatch, code="123456"):
    mod = types.ModuleType("pyotp")

    class TOTP:
        def __init__(self, secret):
            self.secret = secret

        def now(self):
            return code

    mod.TOTP = TOTP
    monkeypatch.setitem(sys.modules, "pyotp", mod)


def no_sessions(monkeypatch):
    monkeypatch.setattr(rh_price_book, "_stored_session_files",
                        lambda p=None: [])


def fake_sessions(monkeypatch):
    monkeypatch.setattr(rh_price_book, "_stored_session_files",
                        lambda p=None: ["/tmp/.tokens/robinhood.pickle"])


def make_client(**kw):
    return RHPriceBookClient("user@example.com", "hunter2", **kw)


# --------------------------------------------------------------------------- #
# Headless guard: never prompt, never block
# --------------------------------------------------------------------------- #
class TestHeadlessGuard:
    def test_no_totp_no_pickle_returns_false_without_login_call(self, monkeypatch):
        calls = []
        install_fake_rh(monkeypatch, lambda **kw: calls.append(kw))
        no_sessions(monkeypatch)
        client = make_client()
        assert client._ensure_login() is False
        assert calls == []          # network login never attempted
        assert client._logged_in is False

    def test_stored_pickle_allows_headless_attempt(self, monkeypatch):
        calls = []
        install_fake_rh(monkeypatch, lambda **kw: calls.append(kw))
        fake_sessions(monkeypatch)
        client = make_client(pickle_path="suffix")
        assert client._ensure_login() is True
        assert client._logged_in is True
        assert len(calls) == 1
        kw = calls[0]
        assert kw["username"] == "user@example.com"
        assert kw["store_session"] is True
        assert kw["expiresIn"] == RH_SESSION_EXPIRES_SEC
        assert kw["pickle_name"] == "suffix"
        assert "mfa_code" not in kw

    def test_headless_challenge_prompt_dies_fast(self, monkeypatch):
        # A stale pickle can still lead robin_stocks into an SMS input()
        # prompt; headless that must raise (EOFError) -> False, not hang.
        def login_fn(**kw):
            builtins.input("Enter SMS code: ")

        install_fake_rh(monkeypatch, login_fn)
        fake_sessions(monkeypatch)
        real_input = builtins.input
        client = make_client()
        assert client._ensure_login() is False
        assert client._logged_in is False
        assert builtins.input is real_input   # restored

    def test_input_restored_after_success(self, monkeypatch):
        install_fake_rh(monkeypatch, lambda **kw: None)
        fake_sessions(monkeypatch)
        real_input = builtins.input
        client = make_client()
        assert client._ensure_login() is True
        assert builtins.input is real_input

    def test_missing_robin_stocks_returns_false(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "robin_stocks", None)
        monkeypatch.setitem(sys.modules, "robin_stocks.robinhood", None)
        client = make_client()
        assert client._ensure_login() is False

    def test_already_logged_in_short_circuits(self):
        client = make_client()
        client._logged_in = True
        assert client._ensure_login() is True


# --------------------------------------------------------------------------- #
# Interactive mode: prompts allowed (SMS / device approval)
# --------------------------------------------------------------------------- #
class TestInteractive:
    def test_interactive_allows_prompt_and_logs_in(self, monkeypatch):
        answered = []

        def login_fn(**kw):
            answered.append(builtins.input("Enter SMS code: "))

        install_fake_rh(monkeypatch, login_fn)
        no_sessions(monkeypatch)          # no pickle, no TOTP: headless would bail
        monkeypatch.setattr(builtins, "input", lambda *_a: "424242")
        client = make_client()
        assert client._ensure_login(interactive=True) is True
        assert answered == ["424242"]
        assert client._logged_in is True

    def test_interactive_failure_returns_false(self, monkeypatch):
        def login_fn(**kw):
            raise RuntimeError("challenge rejected")

        install_fake_rh(monkeypatch, login_fn)
        no_sessions(monkeypatch)
        client = make_client()
        assert client._ensure_login(interactive=True) is False
        assert client._logged_in is False


# --------------------------------------------------------------------------- #
# TOTP stays supported (optional)
# --------------------------------------------------------------------------- #
class TestTotp:
    def test_totp_code_passed_headless_without_pickle(self, monkeypatch):
        calls = []
        install_fake_rh(monkeypatch, lambda **kw: calls.append(kw))
        install_fake_pyotp(monkeypatch, code="654321")
        no_sessions(monkeypatch)
        client = make_client(mfa_secret="BASE32SECRET")
        assert client._ensure_login() is True
        assert calls[0]["mfa_code"] == "654321"

    def test_broken_pyotp_falls_back_to_guard(self, monkeypatch):
        calls = []
        install_fake_rh(monkeypatch, lambda **kw: calls.append(kw))
        monkeypatch.setitem(sys.modules, "pyotp", None)   # import fails
        no_sessions(monkeypatch)
        client = make_client(mfa_secret="BASE32SECRET")
        assert client._ensure_login() is False
        assert calls == []


# --------------------------------------------------------------------------- #
# _stored_session_files helper
# --------------------------------------------------------------------------- #
class TestStoredSessionFiles:
    def test_literal_file(self, tmp_path):
        p = tmp_path / "robinhood.pickle"
        p.write_bytes(b"x")
        assert str(p) in _stored_session_files(str(p))

    def test_directory_of_pickles(self, tmp_path):
        p = tmp_path / "robinhoodABC.pickle"
        p.write_bytes(b"x")
        found = _stored_session_files(str(tmp_path))
        assert str(p) in found

    def test_nonexistent_path_never_raises(self, tmp_path):
        found = _stored_session_files(str(tmp_path / "nope"))
        assert isinstance(found, list)
        assert not any("nope" in f for f in found)

    def test_none_never_raises(self):
        assert isinstance(_stored_session_files(None), list)
