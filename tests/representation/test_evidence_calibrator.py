from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from representation.evidence.calibrator import PerChannelCalibrator

def _fitted() -> PerChannelCalibrator:
    calibrator = PerChannelCalibrator()
    calibrator.fit(
        {"appearance": np.asarray([0.1, 0.5, 0.9])},
        np.asarray([0, 1, 1]),
    )
    return calibrator

def test_per_channel_calibrator_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "calibrator.json"
    _fitted().save(path)
    loaded = PerChannelCalibrator.load(path)
    assert loaded.calibrate(0.9, "appearance") == pytest.approx(1.0, abs=0.1)
    assert loaded.calibrate(0.1, "appearance") == pytest.approx(0.0, abs=0.1)

def test_per_channel_calibrator_rejects_tampered_schema(tmp_path: Path) -> None:
    path = tmp_path / "calibrator.json"
    _fitted().save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        PerChannelCalibrator.load(path)

def test_constant_score_calibrator_round_trip(tmp_path: Path) -> None:
    calibrator = PerChannelCalibrator()
    calibrator.fit(
        {"appearance": np.asarray([0.5, 0.5])},
        np.asarray([0, 1]),
    )
    path = tmp_path / "calibrator.json"
    calibrator.save(path)
    loaded = PerChannelCalibrator.load(path)
    assert loaded.calibrate(0.5, "appearance") == pytest.approx(0.5)
