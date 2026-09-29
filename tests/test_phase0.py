"""
Phase 0: signal-pipeline checks.

(a) raw IR is flipped for analysis, the Raw IR graph keeps the original
(b) beats below 50% of the median beat prominence are rejected
(c) noise score is measured on the DC-removed signal, before the band-pass
(d) rise = foot -> peak, fall = peak -> next foot, edge feet bounded by one
    median beat interval
(e) a reading is selected by row id, never by Timestamp
plus: median HR within 1 BPM on synthetic signals of known rate.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "streamlit_app"))
sys.path.insert(0, HERE)

import ppg_analysis as pa  # noqa: E402
from ppg_features import extract_all_features, extract_pulse_features  # noqa: E402
from selection import get_selected_reading  # noqa: E402
from synthetic import make_ppg  # noqa: E402

FS = 500


# ------------------------------------------------------------------ (a) flip

@pytest.mark.parametrize("kind", ["notch", "strong_notch"])
@pytest.mark.parametrize("hr", [60, 90])
def test_a_pulse_points_up_and_peaks_land_on_systolic(kind, hr):
    raw, _, info = make_ppg(
        hr, kind, noise=0.01, wander=0.1, hrv=0.01, seed=3, return_info=True
    )

    # Device polarity: the pulse points DOWN in the raw signal.
    result = pa.analyze_samples(raw)

    # The graph data still shows the original raw IR ...
    assert np.array_equal(result["waveform"]["Raw IR"].to_numpy(), raw)
    # ... while the analysis works on the flipped copy.
    assert np.array_equal(
        result["waveform"]["Raw IR (flipped)"].to_numpy(), -raw
    )

    # Detected peaks sit on the true systolic peaks (within 60 ms).
    truth = info["systolic"]
    truth = truth[(truth > 250) & (truth < len(raw) - 250)]
    peaks = result["peaks"]
    assert len(peaks) >= len(truth) - 1

    for t in truth:
        assert np.min(np.abs(peaks - t)) <= 0.06 * FS


# ---------------------------------------------------- (b) 50% prominence rule

def _bump(n, centre, width):
    x = np.arange(n)
    return np.exp(-0.5 * ((x - centre) / width) ** 2)


def test_b_small_peaks_below_half_median_prominence_are_rejected():
    n = 7500
    period = int(0.8 * FS)                     # 75 BPM
    signal = np.zeros(n)

    main = list(range(300, n - 300, period))
    for c in main:
        signal += _bump(n, c, 12)              # real beat, amplitude 1
        signal += 0.3 * _bump(n, c + 250, 12)  # small bump, far enough
        #                                        from the beat to survive
        #                                        the distance rule

    peaks = pa.detect_pulses(signal)

    assert len(peaks) == len(main)
    for c in main:
        assert np.min(np.abs(peaks - c)) <= 3


def test_b_comparable_peaks_are_kept():
    n = 7500
    period = int(0.8 * FS)
    signal = np.zeros(n)

    main = list(range(300, n - 300, period))
    for k, c in enumerate(main):
        signal += (1.0 if k % 2 else 0.7) * _bump(n, c, 12)  # 70% vs 100%

    assert len(pa.detect_pulses(signal)) == len(main)


# ------------------------------------------------------------------ (c) noise

def test_c_noise_score_is_measured_before_the_bandpass_filter():
    clean, _ = make_ppg(75, "notch", noise=0.005, wander=0.1, seed=1)
    noisy, _ = make_ppg(75, "notch", noise=0.35, wander=0.1, seed=1)

    q_clean = pa.analyze_samples(clean)["quality"]
    q_noisy = pa.analyze_samples(noisy)["quality"]

    assert q_clean["noise_score"] >= 90
    assert q_noisy["noise_score"] <= 55
    assert q_noisy["noise_score"] < q_clean["noise_score"]


def test_c_same_signal_scores_100_only_when_measured_after_the_filter():
    noisy, _ = make_ppg(75, "notch", noise=0.35, wander=0.1, seed=1)
    result = pa.analyze_samples(noisy)
    wf = result["waveform"]

    after_filter = pa.calculate_quality(
        wf["Filtered PPG"].to_numpy(), result["peaks"]
    )["noise_score"]
    before_filter = pa.calculate_quality(
        wf["Filtered PPG"].to_numpy(),
        result["peaks"],
        noise_signal=wf["DC Removed PPG"].to_numpy(),
    )["noise_score"]

    assert after_filter == 100          # the flaw being fixed
    assert before_filter < 100
    assert result["quality"]["noise_score"] == before_filter


# ------------------------------------------------------------- (d) foot timing

def _triangle_train(rise_s, fall_s, beats, lead=0, tail=0):
    """Asymmetric pulses: foot at 0, peak after rise_s, next foot fall_s later."""
    rise, fall = int(rise_s * FS), int(fall_s * FS)
    one = np.concatenate([np.linspace(0, 1, rise, endpoint=False),
                          np.linspace(1, 0, fall, endpoint=False)])
    sig = np.concatenate([np.zeros(lead)] + [one] * beats + [np.zeros(tail)])
    peaks = np.array([lead + k * (rise + fall) + rise for k in range(beats)])
    return sig, peaks


def test_d_rise_is_foot_to_peak_and_fall_is_peak_to_next_foot():
    sig, peaks = _triangle_train(0.2, 0.6, beats=8, tail=1)
    table = extract_pulse_features(sig, peaks, FS)

    assert len(table) == 8
    assert np.allclose(table["Rise Time (s)"], 0.2, atol=0.01)
    assert np.allclose(table["Fall Time (s)"], 0.6, atol=0.01)
    assert np.allclose(table["Pulse Width (s)"], 0.8, atol=0.01)


def test_d_first_and_last_beats_are_bounded_by_one_median_interval():
    # Beats every 0.8 s, but the signal keeps sloping down for 4 s after
    # the last peak and starts 4 s "uphill" before the first one. The feet
    # of the first and last beat must stay within one median interval.
    sig, peaks = _triangle_train(0.2, 0.6, beats=6)
    long_ramp_up = np.linspace(-30, 0, 4 * FS, endpoint=False)
    long_ramp_down = np.linspace(1, -30, 4 * FS)
    sig = np.concatenate([long_ramp_up, sig, long_ramp_down])
    peaks = peaks + len(long_ramp_up)

    table = extract_pulse_features(sig, peaks, FS)
    interval = 0.8

    assert table["Rise Time (s)"].iloc[0] <= interval + 1e-9
    assert table["Fall Time (s)"].iloc[-1] <= interval + 1e-9
    # Interior beats are untouched by the bound.
    assert np.allclose(table["Rise Time (s)"].iloc[1:-1], 0.2, atol=0.01)
    assert np.allclose(table["Fall Time (s)"].iloc[1:-1], 0.6, atol=0.01)


# --------------------------------------------------------------- (e) selection

def test_e_reading_is_selected_by_row_id_not_timestamp():
    df = pd.DataFrame({
        "Timestamp": ["2026-09-02 10:00:00"] * 3 + ["2026-09-02 10:05:00"],
        "PatientId": [1, 2, 3, 2],
        "HeartRate": [60, 70, 80, 90],
    })
    # A filter keeps patients 2 and 3 only: original row ids 1, 2, 3.
    filtered = df[df["PatientId"].isin([2, 3])]

    # Table row 1 is the second displayed row: original row id 2, which
    # shares its Timestamp with row ids 0 and 1.
    row_id, reading = get_selected_reading(df, filtered, 1)

    assert row_id == 2
    assert reading["PatientId"] == 3
    assert reading["HeartRate"] == 80


# ------------------------------------------------------------ median HR, known

@pytest.mark.parametrize("hr", [45, 60, 90, 120, 175])
@pytest.mark.parametrize("kind", ["notch", "strong_notch", "sawtooth"])
@pytest.mark.parametrize("seed", [0, 1])
def test_median_hr_within_1_bpm(hr, kind, seed):
    for noise, wander, hrv in [(0.01, 0.1, 0.01), (0.05, 0.3, 0.02)]:
        raw, truth = make_ppg(
            hr, kind, noise=noise, wander=wander, hrv=hrv, seed=seed
        )
        result = pa.analyze_samples(raw)
        summary = extract_all_features(
            result["waveform"]["Filtered PPG"].to_numpy(),
            result["peaks"],
            FS,
        )["summary"]

        assert abs(summary["Median HR (BPM)"] - truth) <= 1.0
