"""
Accuracy tests for the PPG analysis.

Run:  python -m pytest tests -q

Set PPG_SAMPLE_XLSX to a VinCense workbook to also compare against the
device heart rate stored in that file.
"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "streamlit_app"))
sys.path.insert(0, HERE)

import ppg_analysis as pa  # noqa: E402
from synthetic import make_ppg  # noqa: E402


def analyse(samples):
    """Run the real pipeline (flip, DC removal, filter, pulses, quality)."""
    result = pa.analyze_samples(samples)
    return result["quality"], result["peaks"]


@pytest.mark.parametrize("kind", ["notch", "strong_notch", "sawtooth"])
@pytest.mark.parametrize("hr", list(range(40, 181, 10)))
@pytest.mark.parametrize("seed", [0, 1])
def test_clean_and_noisy_heart_rate(kind, hr, seed):
    for noise, wander, hrv in [(0.01, 0.1, 0.01), (0.08, 0.4, 0.03)]:
        samples, truth = make_ppg(
            hr, kind, noise=noise, wander=wander, hrv=hrv, seed=seed
        )
        quality, _ = analyse(samples)
        assert quality["heart_rate"] is not None
        assert abs(quality["heart_rate"] - truth) <= 2.5


def test_flat_signal_reports_no_pulse():
    quality, peaks = analyse(np.full(7500, 250000.0))
    assert len(peaks) == 0
    assert quality["classification"] == "Poor"


def test_pure_noise_is_not_good():
    rng = np.random.default_rng(3)
    quality, _ = analyse(250000 + 500 * rng.standard_normal(7500))
    assert quality["classification"] != "Good"


def test_short_recording_does_not_crash():
    quality, _ = analyse(np.arange(50, dtype=float))
    assert quality["classification"] == "Poor"


@pytest.mark.skipif(
    not os.environ.get("PPG_SAMPLE_XLSX"),
    reason="set PPG_SAMPLE_XLSX to run against real readings",
)
def test_sample_workbook_matches_device_hr():
    import pandas as pd

    df = pd.read_excel(os.environ["PPG_SAMPLE_XLSX"], sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]

    good_errors = []
    for _, reading in df.iterrows():
        analysis = pa.analyze_reading(reading)
        quality = analysis["quality"]
        if quality["classification"] == "Good":
            good_errors.append(
                abs(quality["heart_rate"] - float(reading["HeartRate"]))
            )

    assert len(good_errors) > 50
    assert np.mean(np.array(good_errors) <= 5) >= 0.95
