"""Oracle 3.0 offline test suite (no creds / no network / no order placement).

Written with stdlib ``unittest`` so it runs three ways:
  * ``python -m pytest tests/``        (pytest collects unittest TestCases)
  * ``python -m unittest discover tests``
  * ``python run_selftests.py``        (imports + runs each module's tests)
"""
