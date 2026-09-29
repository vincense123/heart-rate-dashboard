"""
PPG Feature Analysis
Research/engineering use only.
"""

import numpy as np
import pandas as pd


# np.trapezoid only exists in NumPy 2.x; older installs use np.trapz.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _clean(x):
    if x is None:
        return np.array([], dtype=float)
    a = np.asarray(x, dtype=float).reshape(-1)
    return a[np.isfinite(a)]


def _mean(x):
    x = _clean(x)
    return float(np.mean(x)) if len(x) else np.nan


def _std(x):
    x = _clean(x)
    return float(np.std(x, ddof=1)) if len(x) > 1 else np.nan


def _cv(x):
    x = _clean(x)
    if len(x) < 2 or abs(np.mean(x)) < 1e-12:
        return np.nan
    return float(np.std(x, ddof=1) / abs(np.mean(x)) * 100.0)


def extract_timing_features(peaks, sampling_rate=500):
    """Pulse interval, HR and beat-to-beat variability."""
    peaks = np.asarray(peaks, dtype=int)

    result = {
        "Detected Pulses": int(len(peaks)),
        "Mean IBI (s)": np.nan,
        "IBI SD (s)": np.nan,
        "IBI CV (%)": np.nan,
        "Mean HR (BPM)": np.nan,
        "Median HR (BPM)": np.nan,
        "HR SD (BPM)": np.nan,
        "HR CV (%)": np.nan,
    }

    if len(peaks) < 2:
        return result

    ibi = np.diff(peaks) / float(sampling_rate)
    ibi = ibi[ibi > 0]

    if not len(ibi):
        return result

    hr = 60.0 / ibi
    valid_hr = hr[(hr >= 30) & (hr <= 220)]

    result.update({
        "Mean IBI (s)": _mean(ibi),
        "IBI SD (s)": _std(ibi),
        "IBI CV (%)": _cv(ibi),
        "Mean HR (BPM)": _mean(valid_hr),
        "Median HR (BPM)": float(np.median(valid_hr)) if len(valid_hr) else np.nan,
        "HR SD (BPM)": _std(valid_hr),
        "HR CV (%)": _cv(valid_hr),
    })

    return result


def _foot_between(signal, lo, hi):
    """Index of the minimum of signal[lo:hi + 1] (the pulse foot)."""
    lo = max(0, int(lo))
    hi = min(len(signal) - 1, int(hi))

    if hi <= lo:
        return lo

    return lo + int(np.argmin(signal[lo:hi + 1]))


def extract_pulse_features(filtered_ppg, peaks, sampling_rate=500):
    """
    Extract morphology features from each detected pulse.

    The foot of a pulse is the minimum of the signal between two
    neighbouring peaks. For beat k:

        rise time = foot before the peak  -> peak
        fall time = peak -> foot after the peak
        width     = foot before -> foot after

    The foot before the FIRST peak and the foot after the LAST peak
    have no neighbouring peak, so they are searched within one median
    beat interval of that peak (and never outside the recording).
    """
    signal = _clean(filtered_ppg)
    peaks = np.asarray(peaks, dtype=int)

    if len(signal) == 0 or len(peaks) < 2:
        return pd.DataFrame()

    peaks = np.unique(peaks[(peaks >= 0) & (peaks < len(signal))])

    if len(peaks) < 2:
        return pd.DataFrame()

    median_interval = int(round(np.median(np.diff(peaks))))

    # feet[k] is the foot before beat k; feet[k + 1] the foot after it.
    feet = [
        _foot_between(signal, peaks[0] - median_interval, peaks[0])
    ]

    for previous, current in zip(peaks[:-1], peaks[1:]):
        feet.append(_foot_between(signal, previous, current))

    feet.append(
        _foot_between(signal, peaks[-1], peaks[-1] + median_interval)
    )

    rows = []

    for i, peak in enumerate(peaks):
        start = feet[i]
        end = feet[i + 1]

        if not (start < peak < end):
            continue

        baseline = float(signal[start])
        amplitude = float(signal[peak] - baseline)

        rise_s = (peak - start) / sampling_rate
        fall_s = (end - peak) / sampling_rate
        width_s = (end - start) / sampling_rate

        corrected = np.maximum(signal[start:end + 1] - baseline, 0)
        area = float(_trapz(corrected, dx=1.0 / sampling_rate))

        rows.append({
            "Pulse": i + 1,
            "Start Sample": int(start),
            "Peak Sample": int(peak),
            "End Sample": int(end),
            "Amplitude": amplitude,
            "Pulse Width (s)": width_s,
            "Rise Time (s)": rise_s,
            "Fall Time (s)": fall_s,
            "Rise Slope": amplitude / rise_s,
            "Fall Slope": amplitude / fall_s,
            "Pulse Area": area,
        })

    return pd.DataFrame(rows)


def summarize_pulses(df):
    """Recording-level statistics from pulse morphology."""
    if df is None or df.empty:
        return {
            "Pulses Analysed": 0,
            "Mean Amplitude": np.nan,
            "Amplitude SD": np.nan,
            "Amplitude CV (%)": np.nan,
            "Mean Pulse Width (s)": np.nan,
            "Pulse Width SD (s)": np.nan,
            "Mean Rise Time (s)": np.nan,
            "Mean Fall Time (s)": np.nan,
            "Mean Rise Slope": np.nan,
            "Mean Fall Slope": np.nan,
            "Mean Pulse Area": np.nan,
            "Pulse Area CV (%)": np.nan,
        }

    return {
        "Pulses Analysed": len(df),
        "Mean Amplitude": _mean(df["Amplitude"]),
        "Amplitude SD": _std(df["Amplitude"]),
        "Amplitude CV (%)": _cv(df["Amplitude"]),
        "Mean Pulse Width (s)": _mean(df["Pulse Width (s)"]),
        "Pulse Width SD (s)": _std(df["Pulse Width (s)"]),
        "Mean Rise Time (s)": _mean(df["Rise Time (s)"]),
        "Mean Fall Time (s)": _mean(df["Fall Time (s)"]),
        "Mean Rise Slope": _mean(df["Rise Slope"]),
        "Mean Fall Slope": _mean(df["Fall Slope"]),
        "Mean Pulse Area": _mean(df["Pulse Area"]),
        "Pulse Area CV (%)": _cv(df["Pulse Area"]),
    }


def extract_waveform_features(filtered_ppg, sampling_rate=500):
    """Basic statistics of the filtered waveform."""
    x = _clean(filtered_ppg)

    if not len(x):
        return {}

    return {
        "Samples": len(x),
        "Duration (s)": len(x) / sampling_rate,
        "Mean": float(np.mean(x)),
        "Std Dev": float(np.std(x)),
        "RMS": float(np.sqrt(np.mean(x ** 2))),
        "Peak-to-Peak": float(np.max(x) - np.min(x)),
        "Minimum": float(np.min(x)),
        "Maximum": float(np.max(x)),
    }


def estimate_respiratory_rate(filtered_ppg, peaks=None, sampling_rate=500):
    """
    Exploratory respiratory-rate estimate from slow PPG amplitude modulation.
    Not a clinical RR calculation.
    """
    signal = _clean(filtered_ppg)

    if len(signal) < sampling_rate * 8:
        return {
            "Respiratory Rate (BPM)": np.nan,
            "Respiratory Confidence (%)": 0.0,
        }

    peaks = np.asarray(peaks if peaks is not None else [], dtype=int)
    peaks = peaks[(peaks >= 0) & (peaks < len(signal))]

    if len(peaks) >= 5:
        amplitudes = []
        times = []
        half = max(1, int(0.5 * sampling_rate))

        for peak in peaks:
            lo = max(0, peak - half)
            hi = min(len(signal), peak + half + 1)
            amplitudes.append(float(signal[peak] - np.min(signal[lo:hi])))
            times.append(peak / sampling_rate)

        times = np.asarray(times)
        amplitudes = np.asarray(amplitudes)

        grid = np.arange(0, len(signal) / sampling_rate, 0.25)

        if len(grid) >= 20 and len(np.unique(times)) >= 2:
            envelope = np.interp(grid, times, amplitudes)
            return _rr_from_envelope(envelope, 4.0)

    centered = signal - np.mean(signal)
    envelope = np.abs(centered)

    block = max(1, int(sampling_rate / 4))
    usable = (len(envelope) // block) * block

    if usable < 32:
        return {
            "Respiratory Rate (BPM)": np.nan,
            "Respiratory Confidence (%)": 0.0,
        }

    envelope = envelope[:usable].reshape(-1, block).mean(axis=1)
    return _rr_from_envelope(envelope, 4.0)


def _rr_from_envelope(envelope, fs):
    envelope = _clean(envelope)

    if len(envelope) < 32:
        return {
            "Respiratory Rate (BPM)": np.nan,
            "Respiratory Confidence (%)": 0.0,
        }

    envelope = envelope - np.mean(envelope)

    frequencies = np.linspace(0.10, 0.50, 401)
    t = np.arange(len(envelope)) / fs
    powers = []

    for f in frequencies:
        sine = np.sin(2 * np.pi * f * t)
        cosine = np.cos(2 * np.pi * f * t)
        powers.append(
            np.dot(envelope, sine) ** 2
            + np.dot(envelope, cosine) ** 2
        )

    powers = np.asarray(powers)
    idx = int(np.argmax(powers))
    median_power = float(np.median(powers))

    if median_power <= 0:
        confidence = 0.0
    else:
        confidence = float(
            np.clip(
                (powers[idx] / median_power - 1.0) * 18.0,
                0.0,
                100.0,
            )
        )

    return {
        "Respiratory Rate (BPM)": float(frequencies[idx] * 60.0),
        "Respiratory Confidence (%)": confidence,
    }


def calculate_spo2_from_red_ir(red, ir):
    """
    Exploratory RED/IR ratio-of-ratios calculation.

    Requires both RED and IR channels. A device-specific calibration
    curve is required before this can be considered a validated SpO2 method.
    """
    red = _clean(red)
    ir = _clean(ir)
    n = min(len(red), len(ir))

    if n < 10:
        return {
            "SpO2 Available": False,
            "SpO2 (%)": np.nan,
            "SpO2 Ratio": np.nan,
            "SpO2 Status": "Insufficient RED + IR data",
        }

    red = red[:n]
    ir = ir[:n]

    red_dc = abs(np.mean(red))
    ir_dc = abs(np.mean(ir))

    if red_dc <= 1e-12 or ir_dc <= 1e-12:
        return {
            "SpO2 Available": False,
            "SpO2 (%)": np.nan,
            "SpO2 Ratio": np.nan,
            "SpO2 Status": "Invalid DC component",
        }

    ratio = (np.std(red) / red_dc) / (np.std(ir) / ir_dc)

    # Exploratory empirical approximation only.
    spo2 = float(np.clip(110.0 - 25.0 * ratio, 0.0, 100.0))

    return {
        "SpO2 Available": True,
        "SpO2 (%)": spo2,
        "SpO2 Ratio": float(ratio),
        "SpO2 Status": "Exploratory only - requires calibration",
    }


def extract_all_features(
    filtered_ppg,
    peaks,
    sampling_rate=500,
    red_ppg=None,
    ir_ppg=None,
):
    """Run all currently possible feature calculations."""
    pulse_df = extract_pulse_features(
        filtered_ppg,
        peaks,
        sampling_rate,
    )

    summary = {}
    summary.update(
        extract_waveform_features(
            filtered_ppg,
            sampling_rate,
        )
    )
    summary.update(
        extract_timing_features(
            peaks,
            sampling_rate,
        )
    )
    summary.update(
        summarize_pulses(
            pulse_df
        )
    )
    summary.update(
        estimate_respiratory_rate(
            filtered_ppg,
            peaks,
            sampling_rate,
        )
    )

    if red_ppg is not None and ir_ppg is not None:
        summary.update(
            calculate_spo2_from_red_ir(
                red_ppg,
                ir_ppg,
            )
        )
    else:
        summary.update({
            "SpO2 Available": False,
            "SpO2 (%)": np.nan,
            "SpO2 Ratio": np.nan,
            "SpO2 Status": "RED channel not available",
        })

    return {
        "summary": summary,
        "pulse_features": pulse_df,
    }
