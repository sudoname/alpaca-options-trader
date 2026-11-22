"""Missing data cannot silently reuse the superseded GBM result."""

import json

from oracle.ml import profitability_research as pr


def test_missing_dataset_and_calibration_abstain(tmp_path):
    result = pr.run_corrected_gbm(str(tmp_path / "dataset.sqlite"),
                                  str(tmp_path / "cal.json"), str(tmp_path))
    assert result["status"] == "INSUFFICIENT_DATA"
    assert set(result["missing_inputs"]) == {"dataset", "execution_calibration"}
    assert not (tmp_path / "replay402_corrected.json").exists()


def test_preflight_valuation_is_not_silently_ignored(tmp_path):
    dataset = tmp_path / "d.sqlite"
    calibration = tmp_path / "c.json"
    dataset.touch()
    calibration.write_text(json.dumps({"one_way_frac": .02}))
    result = pr.run_corrected_gbm(str(dataset), str(calibration), str(tmp_path),
                                  str(tmp_path / "v.json"))
    assert result["missing_inputs"] == ["valuation"]


def test_incomplete_calibration_refused_before_replay(tmp_path):
    dataset = tmp_path / "d.sqlite"
    calibration = tmp_path / "c.json"
    dataset.touch()
    calibration.write_text(json.dumps({"slippage": {"entry_vs_mid": {
        "quantiles": {"mean": 0.02}}}}))
    result = pr.run_corrected_gbm(str(dataset), str(calibration), str(tmp_path))
    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["missing_inputs"] == ["measured_execution_calibration"]
