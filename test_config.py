"""
Phase 4.5 — offline tests for the centralized config loader.

Covers the required cases:
  * shell env overrides .env
  * .env is used when shell env missing
  * defaults used when neither exists
  * boolean parsing works
  * numeric parsing works
  * existing Phase 0–4 flags still parse correctly (through the real loaders)

Run with:
    python -X utf8 -m unittest test_config -v
    python -X utf8 -m pytest test_config.py -q

NO network and NO broker calls. ConfigLoader is exercised with an injected
``environ`` dict and a temp ``.env`` file, so nothing touches the real shell
environment or the project's ``.env``.
"""

import os
import tempfile
import unittest

from config_loader import ConfigLoader, parse_env_file
from risk_engine import load_risk_limits_from_env
from portfolio_risk import load_portfolio_limits_from_env


def _write_env(text: str) -> str:
    """Write `text` to a temp .env file and return its path."""
    d = tempfile.mkdtemp()
    path = os.path.join(d, ".env")
    with open(path, "w") as f:
        f.write(text)
    return path


# --------------------------------------------------------------------------- #
# Resolution order
# --------------------------------------------------------------------------- #
class TestResolutionOrder(unittest.TestCase):
    def setUp(self):
        self.path = _write_env(
            "FOO=from_file\n"
            "# a comment\n"
            "FLAG=false\n"
            "NUM=10\n"
            "RATIO=0.25\n"
        )

    def test_shell_overrides_dotenv(self):
        c = ConfigLoader(path=self.path, environ={"FOO": "from_shell"})
        self.assertEqual(c.get("FOO"), "from_shell")
        self.assertEqual(c.get_str("FOO"), "from_shell")

    def test_dotenv_used_when_shell_missing(self):
        c = ConfigLoader(path=self.path, environ={})
        self.assertEqual(c.get("FOO"), "from_file")

    def test_default_when_neither_exists(self):
        c = ConfigLoader(path=self.path, environ={})
        self.assertEqual(c.get("MISSING", "fallback"), "fallback")
        self.assertIsNone(c.get("MISSING"))
        self.assertEqual(c.get_str("MISSING", "d"), "d")

    def test_contains_reflects_both_layers(self):
        c = ConfigLoader(path=self.path, environ={"ONLY_SHELL": "1"})
        self.assertIn("FOO", c)          # from file
        self.assertIn("ONLY_SHELL", c)   # from shell
        self.assertNotIn("NOPE", c)


# --------------------------------------------------------------------------- #
# Typed parsing
# --------------------------------------------------------------------------- #
class TestTypedParsing(unittest.TestCase):
    def setUp(self):
        self.path = _write_env(
            "FLAG_FALSE=false\n"
            "FLAG_TRUE=true\n"
            "FLAG_ONE=1\n"
            "FLAG_ON=on\n"
            "FLAG_YES=YES\n"
            "FLAG_GARBAGE=banana\n"
            "INT_OK=42\n"
            "INT_FLOATY=3.0\n"
            "INT_FRAC=0.25\n"
            "INT_BAD=abc\n"
            "FLOAT_OK=1.75\n"
            "FLOAT_BAD=xyz\n"
        )
        self.c = ConfigLoader(path=self.path, environ={})

    def test_bool_parsing(self):
        self.assertFalse(self.c.get_bool("FLAG_FALSE", True))
        self.assertTrue(self.c.get_bool("FLAG_TRUE", False))
        self.assertTrue(self.c.get_bool("FLAG_ONE", False))
        self.assertTrue(self.c.get_bool("FLAG_ON", False))
        self.assertTrue(self.c.get_bool("FLAG_YES", False))
        # Anything outside the truthy set is False.
        self.assertFalse(self.c.get_bool("FLAG_GARBAGE", True))
        # Missing -> default.
        self.assertTrue(self.c.get_bool("FLAG_MISSING", True))
        self.assertFalse(self.c.get_bool("FLAG_MISSING", False))

    def test_bool_shell_override(self):
        c = ConfigLoader(path=self.path, environ={"FLAG_FALSE": "on"})
        self.assertTrue(c.get_bool("FLAG_FALSE", False))

    def test_int_parsing(self):
        self.assertEqual(self.c.get_int("INT_OK", 0), 42)
        self.assertEqual(self.c.get_int("INT_FLOATY", 0), 3)   # "3.0" -> 3
        self.assertEqual(self.c.get_int("INT_FRAC", 9), 0)     # "0.25" floors to 0
        self.assertEqual(self.c.get_int("INT_BAD", 7), 7)      # garbage -> default
        self.assertEqual(self.c.get_int("INT_MISSING", 5), 5)  # missing -> default

    def test_float_parsing(self):
        self.assertAlmostEqual(self.c.get_float("FLOAT_OK", 0.0), 1.75)
        self.assertAlmostEqual(self.c.get_float("FLOAT_BAD", 2.5), 2.5)   # garbage -> default
        self.assertAlmostEqual(self.c.get_float("FLOAT_MISSING", 9.9), 9.9)


# --------------------------------------------------------------------------- #
# parse_env_file
# --------------------------------------------------------------------------- #
class TestParseEnvFile(unittest.TestCase):
    def test_skips_comments_and_blanks_and_strips(self):
        path = _write_env(
            "\n"
            "# comment line\n"
            "KEY = value \n"
            "EMPTY=\n"
            "EQ=a=b=c\n"
        )
        d = parse_env_file(path)
        self.assertEqual(d["KEY"], "value")        # key/value both stripped
        self.assertEqual(d["EMPTY"], "")
        self.assertEqual(d["EQ"], "a=b=c")         # only first '=' splits
        self.assertNotIn("# comment line", d)

    def test_missing_file_is_empty(self):
        self.assertEqual(parse_env_file("does_not_exist_42.env"), {})


# --------------------------------------------------------------------------- #
# Existing Phase 0–4 flags still parse correctly (through the real loaders)
# --------------------------------------------------------------------------- #
class TestPhaseFlagsThroughLoaders(unittest.TestCase):
    """The risk_engine / portfolio_risk loaders now resolve via ConfigLoader.

    These assert the historical defaults are unchanged AND that a shell override
    wins over the .env file, using a temp .env + a patched os.environ.
    """

    def test_risk_limits_defaults_from_env_file(self):
        path = _write_env(
            "MAX_BUDGET_PER_TRADE=2500\n"
            "DAILY_LOSS_LIMIT=300\n"
            "MAX_CONCURRENT_POSITIONS=3\n"
            "MIN_PDT_REMAINING=1\n"
            "KILL_SWITCH_LOSS=500\n"
            "MAX_POSITIONS_PER_UNDERLYING=2\n"
        )
        # No shell overrides for these keys.
        saved = {k: os.environ.pop(k) for k in (
            "MAX_BUDGET_PER_TRADE", "DAILY_LOSS_LIMIT", "MAX_CONCURRENT_POSITIONS",
            "MIN_PDT_REMAINING", "KILL_SWITCH_LOSS", "MAX_POSITIONS_PER_UNDERLYING",
        ) if k in os.environ}
        try:
            lim = load_risk_limits_from_env(path)
            self.assertEqual(lim.max_budget_per_trade, 2500.0)
            self.assertEqual(lim.daily_loss_limit, 300.0)
            self.assertEqual(lim.max_concurrent, 3)
            self.assertEqual(lim.min_pdt_remaining, 1)
            self.assertEqual(lim.kill_switch_loss, 500.0)
            self.assertEqual(lim.max_per_underlying, 2)
        finally:
            os.environ.update(saved)

    def test_risk_limits_defaults_when_absent(self):
        # Clear any process-env copies (python-dotenv, imported by pdt_tracker,
        # may have populated os.environ from the real .env) so "absent" truly
        # means absent from BOTH layers and we exercise the code defaults.
        path = _write_env("# empty\n")
        keys = ("MAX_BUDGET_PER_TRADE", "DAILY_LOSS_LIMIT",
                "MAX_CONCURRENT_POSITIONS", "MIN_PDT_REMAINING",
                "KILL_SWITCH_LOSS", "MAX_POSITIONS_PER_UNDERLYING")
        saved = {k: os.environ.pop(k) for k in keys if k in os.environ}
        try:
            lim = load_risk_limits_from_env(path)
            self.assertEqual(lim.max_budget_per_trade, 500.0)
            self.assertEqual(lim.daily_loss_limit, 300.0)
            self.assertEqual(lim.max_concurrent, 3)
            self.assertEqual(lim.kill_switch_loss, 500.0)
            self.assertEqual(lim.max_per_underlying, 1000)   # historical no-op default
        finally:
            os.environ.update(saved)

    def test_risk_limits_shell_overrides_env_file(self):
        path = _write_env("MAX_BUDGET_PER_TRADE=2500\n")
        os.environ["MAX_BUDGET_PER_TRADE"] = "9999"
        try:
            lim = load_risk_limits_from_env(path)
            self.assertEqual(lim.max_budget_per_trade, 9999.0)
        finally:
            os.environ.pop("MAX_BUDGET_PER_TRADE", None)

    def test_portfolio_limits_defaults_from_env_file(self):
        path = _write_env(
            "USE_PORTFOLIO_GREEK_LIMITS=true\n"
            "MAX_PORTFOLIO_ABS_DELTA=5.0\n"
            "MAX_PORTFOLIO_ABS_VEGA=10.0\n"
            "MAX_PORTFOLIO_THETA_LOSS=5.0\n"
            "MAX_SAME_DIRECTION_POSITIONS=3\n"
            "MAX_POSITIONS_PER_UNDERLYING=2\n"
        )
        saved = {k: os.environ.pop(k) for k in (
            "USE_PORTFOLIO_GREEK_LIMITS", "MAX_PORTFOLIO_ABS_DELTA",
            "MAX_PORTFOLIO_ABS_VEGA", "MAX_PORTFOLIO_THETA_LOSS",
            "MAX_SAME_DIRECTION_POSITIONS", "MAX_POSITIONS_PER_UNDERLYING",
        ) if k in os.environ}
        try:
            lim = load_portfolio_limits_from_env(path)
            self.assertTrue(lim.enabled)
            self.assertEqual(lim.max_abs_delta, 5.0)
            self.assertEqual(lim.max_abs_vega, 10.0)
            self.assertEqual(lim.max_theta_loss, 5.0)
            self.assertEqual(lim.max_same_direction, 3)
            self.assertEqual(lim.max_per_underlying, 2)
        finally:
            os.environ.update(saved)

    def test_portfolio_limits_disabled_by_default(self):
        path = _write_env("# empty\n")
        keys = ("USE_PORTFOLIO_GREEK_LIMITS", "MAX_PORTFOLIO_ABS_DELTA",
                "MAX_PORTFOLIO_ABS_VEGA", "MAX_PORTFOLIO_THETA_LOSS",
                "MAX_SAME_DIRECTION_POSITIONS", "MAX_POSITIONS_PER_UNDERLYING")
        saved = {k: os.environ.pop(k) for k in keys if k in os.environ}
        try:
            lim = load_portfolio_limits_from_env(path)
            self.assertFalse(lim.enabled)             # OFF by default
            self.assertEqual(lim.max_abs_delta, 5.0)
            self.assertEqual(lim.max_same_direction, 3)
        finally:
            os.environ.update(saved)

    def test_portfolio_limits_shell_enables(self):
        path = _write_env("USE_PORTFOLIO_GREEK_LIMITS=false\n")
        os.environ["USE_PORTFOLIO_GREEK_LIMITS"] = "true"
        try:
            lim = load_portfolio_limits_from_env(path)
            self.assertTrue(lim.enabled)              # shell wins over .env=false
        finally:
            os.environ.pop("USE_PORTFOLIO_GREEK_LIMITS", None)

    def test_phase2_3_flag_strings_parse_truthy(self):
        """Spot-check the truthy set used by every USE_* flag across phases."""
        path = _write_env(
            "USE_DTE_TARGETING=1\n"
            "USE_DELTA_TARGETING=yes\n"
            "USE_COST_EV_GATE=on\n"
            "USE_OPTION_LIQUIDITY_FILTER=TRUE\n"
            "USE_SKIP_ON_WEAK_SIGNAL=false\n"
            "USE_NORMALIZED_CONFIDENCE=0\n"
        )
        c = ConfigLoader(path=path, environ={})
        self.assertTrue(c.get_bool("USE_DTE_TARGETING"))
        self.assertTrue(c.get_bool("USE_DELTA_TARGETING"))
        self.assertTrue(c.get_bool("USE_COST_EV_GATE"))
        self.assertTrue(c.get_bool("USE_OPTION_LIQUIDITY_FILTER"))
        self.assertFalse(c.get_bool("USE_SKIP_ON_WEAK_SIGNAL"))
        self.assertFalse(c.get_bool("USE_NORMALIZED_CONFIDENCE"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
