"""
Phase 4.5 — centralized configuration loading.

One resolver for the whole project so every module reads config the same way.
This removes the historical inconsistency where ``smart_trader.load_credentials``,
``risk_engine.load_risk_limits_from_env`` and ``portfolio_risk`` read ONLY the
``.env`` file (ignoring shell env), while ``run_alpaca_intraday.Config`` already
let a shell ``KEY=... python ...`` override ``.env``.

Resolution order for EVERY key (highest priority first):
  1. shell environment   (``os.environ``)        -- a `KEY=... python ...` wins
  2. ``.env`` file        (manual parse, no python-dotenv)
  3. code-supplied default

Parsing semantics are kept identical to the per-module helpers they replace, so
existing keys keep their current defaults and behavior:
  * bool : truthy set {1, true, yes, on} (case-insensitive); else False
  * float/int : tolerant parse — returns the default on TypeError/ValueError
                (int parses through float so "3.0" -> 3)

``ConfigLoader`` exposes a dict-compatible ``get(name, default)`` so it can be a
drop-in for the parsed-``.env`` dict the old loaders passed around, while also
offering typed getters (``get_bool``/``get_float``/``get_int``/``get_str``) for
new code and tests. No network and no side effects beyond reading the env file.
"""

import os
from typing import Mapping, Optional

_TRUTHY = ("1", "true", "yes", "on")


def parse_env_file(path: str = ".env") -> dict:
    """Manual ``.env`` parse (no python-dotenv), matching the project idiom.

    Skips blank lines and ``#`` comments; splits on the first ``=``. Returns an
    empty dict when the file is missing or unreadable (fail-open).
    """
    data: dict = {}
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    s = line.strip()
                    if "=" in s and not s.startswith("#"):
                        key, value = s.split("=", 1)
                        data[key.strip()] = value.strip()
    except OSError:
        pass
    return data


class ConfigLoader:
    """Resolve config with shell-env precedence over the ``.env`` file.

    Parameters
    ----------
    path:
        Path to the ``.env`` file to use as the fallback layer. Ignored when
        ``file_values`` is supplied.
    file_values:
        Pre-parsed ``.env``-style mapping to use as the fallback layer instead
        of reading ``path``. Lets callers (e.g. the scheduler's ``--selftest``)
        inject a deterministic fallback dict.
    environ:
        Mapping used as the highest-priority layer. Defaults to ``os.environ``;
        tests can pass a dict to simulate shell exports.
    """

    def __init__(
        self,
        path: str = ".env",
        file_values: Optional[Mapping[str, str]] = None,
        environ: Optional[Mapping[str, str]] = None,
    ):
        if file_values is not None:
            self._file = dict(file_values)
        else:
            self._file = parse_env_file(path)
        self._environ = environ if environ is not None else os.environ

    # -- core resolution -------------------------------------------------- #
    def get(self, name, default=None):
        """Return the resolved value (shell > .env > default), unparsed.

        Mirrors ``dict.get`` so this object is a drop-in for the parsed-``.env``
        dicts the old loaders used.
        """
        if name in self._environ:
            return self._environ[name]
        if name in self._file:
            return self._file[name]
        return default

    # -- typed getters ---------------------------------------------------- #
    def get_str(self, name, default: str = "") -> str:
        v = self.get(name, None)
        return default if v is None else str(v)

    def get_bool(self, name, default: bool = False) -> bool:
        v = self.get(name, None)
        if v is None:
            return default
        return str(v).strip().lower() in _TRUTHY

    def get_float(self, name, default: float = 0.0) -> float:
        v = self.get(name, None)
        if v is None:
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def get_int(self, name, default: int = 0) -> int:
        v = self.get(name, None)
        if v is None:
            return default
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return default

    # -- introspection ---------------------------------------------------- #
    def __contains__(self, name) -> bool:
        return name in self._environ or name in self._file


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network; uses a temp file + injected environ)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    import tempfile

    ok = True
    d = tempfile.mkdtemp()
    path = os.path.join(d, ".env")
    with open(path, "w") as f:
        f.write("# comment\n")
        f.write("FOO=from_file\n")
        f.write("FLAG=false\n")
        f.write("NUM=10\n")
        f.write("RATIO=0.25\n")

    # Shell env overrides .env.
    c = ConfigLoader(path=path, environ={"FOO": "from_shell"})
    if c.get("FOO") != "from_shell":
        print("FAIL: shell should override .env", c.get("FOO")); ok = False

    # .env used when shell missing.
    c = ConfigLoader(path=path, environ={})
    if c.get("FOO") != "from_file":
        print("FAIL: .env should be used when shell missing", c.get("FOO")); ok = False

    # Default used when neither exists.
    if c.get("MISSING", "fallback") != "fallback":
        print("FAIL: default should be used when key absent"); ok = False

    # Bool parsing (file vs shell vs default).
    if c.get_bool("FLAG", True) is not False:
        print("FAIL: FLAG=false should parse False"); ok = False
    c2 = ConfigLoader(path=path, environ={"FLAG": "on"})
    if c2.get_bool("FLAG", False) is not True:
        print("FAIL: shell FLAG=on should parse True"); ok = False
    if c.get_bool("MISSING", True) is not True:
        print("FAIL: bool default should apply when absent"); ok = False

    # Numeric parsing.
    if c.get_int("NUM", 0) != 10:
        print("FAIL: NUM should parse to 10"); ok = False
    if abs(c.get_float("RATIO", 0.0) - 0.25) > 1e-9:
        print("FAIL: RATIO should parse to 0.25"); ok = False
    if c.get_int("RATIO", 0) != 0:
        print("FAIL: int('0.25') should floor to 0"); ok = False
    if c.get_int("FOO", 7) != 7:
        print("FAIL: bad int should fall back to default"); ok = False
    if abs(c.get_float("MISSING", 1.5) - 1.5) > 1e-9:
        print("FAIL: float default should apply when absent"); ok = False

    print("config_loader self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv or len(sys.argv) == 1:
        sys.exit(_self_test())
