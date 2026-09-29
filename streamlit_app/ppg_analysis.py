import pandas as pd
import numpy as np

from scipy.signal import (
    butter,
    sosfiltfilt,
    find_peaks,
)


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 500

LOW_CUTOFF = 0.5
HIGH_CUTOFF = 8.0
FILTER_ORDER = 4

MIN_HEART_RATE = 40
MAX_HEART_RATE = 180

MIN_PEAK_DISTANCE = int(
    SAMPLE_RATE * 60 / MAX_HEART_RATE
)

# Beats with a prominence below this fraction of the median beat
# prominence are rejected (dicrotic / diastolic bumps, small artifacts).
PROMINENCE_KEEP_FRACTION = 0.5

# The device IR signal falls when blood volume rises, so the pulse
# points down in the raw data. Analysis uses the flipped signal so the
# systolic peak points up. The Raw IR graph keeps the original.
POLARITY = -1.0


# ============================================================
# FILTER
# ============================================================

def bandpass_filter(
    signal,
    fs=SAMPLE_RATE,
    lowcut=LOW_CUTOFF,
    highcut=HIGH_CUTOFF,
    order=FILTER_ORDER,
):
    signal = np.asarray(
        signal,
        dtype=float,
    )

    if len(signal) < 20:
        return signal.copy()

    nyquist = fs / 2.0

    low = max(
        lowcut / nyquist,
        0.001,
    )

    high = min(
        highcut / nyquist,
        0.999,
    )

    sos = butter(
        order,
        [low, high],
        btype="bandpass",
        output="sos",
    )

    try:
        return sosfiltfilt(
            sos,
            signal,
        )
    except Exception:
        return signal.copy()


# ============================================================
# DC REMOVAL
# ============================================================

def remove_dc(signal):
    signal = np.asarray(
        signal,
        dtype=float,
    )

    window = min(
        201,
        len(signal)
        if len(signal) % 2
        else len(signal) - 1,
    )

    if window >= 3:
        baseline = (
            pd.Series(signal)
            .rolling(
                window=window,
                center=True,
                min_periods=1,
            )
            .median()
            .to_numpy()
        )
    else:
        baseline = np.full(
            len(signal),
            np.median(signal),
        )

    return (
        signal - baseline,
        baseline,
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_signal(signal):
    signal = np.asarray(
        signal,
        dtype=float,
    )

    median = np.median(signal)

    mad = np.median(
        np.abs(
            signal - median
        )
    )

    if mad < 1e-9:
        std = np.std(signal)

        if std < 1e-9:
            return np.zeros_like(signal)

        return (
            signal - median
        ) / std

    return (
        signal - median
    ) / (
        1.4826 * mad
    )


# ============================================================
# PULSE DETECTION
# ============================================================

def estimate_pulse_period(signal, fs=SAMPLE_RATE):
    """
    Dominant pulse period (seconds) from the autocorrelation of the
    filtered signal, or None when there is no clear periodicity.

    The smallest lag whose autocorrelation is close to the strongest
    one is used, so a pulse period is never mistaken for a multiple
    of itself.
    """
    x = np.asarray(signal, dtype=float)

    # Ignore the filter start/end transients (same 0.5 s as pulse
    # detection) and cap extreme spikes so a single artifact cannot
    # dominate the autocorrelation.
    edge = int(fs * 0.5)

    if len(x) > 4 * edge:
        x = x[edge:len(x) - edge]

    low, high = np.percentile(x, [2, 98])
    x = np.clip(x, low, high)
    x = x - np.mean(x)
    n = len(x)

    # Search slightly beyond the 40-180 BPM validity range so that
    # rates at the limits are not cut off at the edge of the window.
    lag_min = int(fs * 60 / (MAX_HEART_RATE + 20))
    lag_max = min(int(fs * 60 / (MIN_HEART_RATE - 5)), n // 2)

    if n < 4 * lag_min or lag_max <= lag_min + 2:
        return None

    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(x, size)
    acf = np.fft.irfft(spectrum * np.conj(spectrum))[:n]

    if acf[0] <= 0:
        return None

    acf = acf / acf[0]
    # Compensate for the shrinking overlap at larger lags.
    acf = acf * (n / (n - np.arange(n)))

    window = acf[lag_min:lag_max + 1]

    candidates, _ = find_peaks(window)

    if len(candidates) == 0:
        return None

    heights = window[candidates]
    strongest = float(np.max(heights))

    if strongest < 0.25:
        return None

    chosen = candidates[np.argmax(heights >= 0.85 * strongest)]

    return float(chosen + lag_min) / fs


def detect_pulses(
    signal,
    fs=SAMPLE_RATE,
):
    normalized = normalize_signal(
        signal
    )

    if len(normalized) < 100:
        return np.array(
            [],
            dtype=int,
        )

    amplitude = (
        np.percentile(
            normalized,
            90,
        )
        -
        np.percentile(
            normalized,
            10,
        )
    )

    if amplitude < 0.05:
        return np.array(
            [],
            dtype=int,
        )

    prominence = max(
        0.15,
        amplitude * 0.10,
    )

    peaks, properties = find_peaks(
        normalized,
        distance=MIN_PEAK_DISTANCE,
        prominence=prominence,
    )

    # --------------------------------------------------------
    # SECOND PASS: reject secondary (dicrotic / diastolic) bumps
    #
    # The first pass only knows the 180 BPM limit, so the small
    # second bump inside every heartbeat can be counted as a pulse
    # (reading roughly double the true rate). Once the real pulse
    # period is known, only one peak per period is allowed, and
    # peaks far smaller than the typical pulse are dropped.
    # --------------------------------------------------------

    period = estimate_pulse_period(signal, fs)

    if period is not None and len(peaks) >= 3:
        # Once the real period is known it replaces the generic
        # 180 BPM limit, so rates close to 180 BPM are not thinned out.
        distance = max(
            int(0.2 * fs),
            int(0.6 * period * fs),
        )

        peaks, properties = find_peaks(
            normalized,
            distance=distance,
            prominence=prominence,
        )

    # A real beat is not much smaller than the typical beat: drop any
    # peak whose prominence is below 50% of the median beat prominence.
    if len(peaks) >= 3:
        typical = np.median(
            properties["prominences"]
        )

        keep = (
            properties["prominences"]
            >= PROMINENCE_KEEP_FRACTION * typical
        )

        peaks = peaks[keep]

    edge = int(
        fs * 0.5
    )

    return peaks[
        (peaks >= edge)
        &
        (peaks < len(signal) - edge)
    ]


# ============================================================
# QUALITY
# ============================================================

def calculate_quality(
    signal,
    peaks,
    fs=SAMPLE_RATE,
    noise_signal=None,
):
    """
    signal        band-pass filtered PPG (interval / amplitude / coverage)
    noise_signal  DC-removed PPG BEFORE the band-pass filter. Noise must
                  be measured here: the 0.5-8 Hz band-pass removes almost
                  all high-frequency noise, so on the filtered signal the
                  noise score is always 100%. Defaults to `signal` only
                  for callers that have nothing else.
    """
    result = {
        "score": 0.0,
        "classification": "Poor",
        "peak_count": len(peaks),
        "interval_score": 0.0,
        "amplitude_score": 0.0,
        "noise_score": 0.0,
        "coverage_score": 0.0,
        "heart_rate": None,
        "median_interval": None,
        "interval_cv": None,
    }

    if len(signal) < 100:
        return result

    duration = len(signal) / fs

    expected_min = (
        duration
        * MIN_HEART_RATE
        / 60
    )

    expected_max = (
        duration
        * MAX_HEART_RATE
        / 60
    )

    # --------------------------------------------------------
    # COVERAGE
    # --------------------------------------------------------

    if len(peaks) >= 3:
        coverage_score = min(
            100,
            (
                len(peaks)
                / max(expected_min, 1)
            )
            * 100,
        )

        if (
            len(peaks)
            > expected_max * 1.25
        ):
            coverage_score *= 0.5

    else:
        coverage_score = 0

    # --------------------------------------------------------
    # INTERVAL CONSISTENCY
    # --------------------------------------------------------

    if len(peaks) >= 3:
        intervals = (
            np.diff(peaks)
            / fs
        )

        median_interval = (
            np.median(intervals)
        )

        interval_cv = (
            np.std(intervals)
            / median_interval
            if median_interval > 0
            else 1
        )

        heart_rate = (
            60 / median_interval
            if median_interval > 0
            else None
        )

        result["median_interval"] = (
            median_interval
        )

        result["interval_cv"] = (
            interval_cv
        )

        result["heart_rate"] = (
            heart_rate
        )

        if interval_cv <= 0.03:
            interval_score = 100
        elif interval_cv <= 0.05:
            interval_score = 90
        elif interval_cv <= 0.08:
            interval_score = 75
        elif interval_cv <= 0.12:
            interval_score = 55
        elif interval_cv <= 0.18:
            interval_score = 30
        else:
            interval_score = 10

        if (
            heart_rate is not None
            and not (
                MIN_HEART_RATE
                <= heart_rate
                <= MAX_HEART_RATE
            )
        ):
            interval_score *= 0.4

    else:
        interval_score = 0

    # --------------------------------------------------------
    # AMPLITUDE CONSISTENCY
    # --------------------------------------------------------

    if len(peaks) >= 3:
        amplitudes = []

        for i in range(
            len(peaks) - 1
        ):
            segment = signal[
                peaks[i]:peaks[i + 1]
            ]

            if len(segment) > 5:
                amplitudes.append(
                    np.max(segment)
                    - np.min(segment)
                )

        if len(amplitudes) >= 2:
            median_amp = np.median(
                amplitudes
            )

            amplitude_cv = (
                np.std(amplitudes)
                / median_amp
                if median_amp > 0
                else 1
            )

            if amplitude_cv <= 0.10:
                amplitude_score = 100
            elif amplitude_cv <= 0.20:
                amplitude_score = 85
            elif amplitude_cv <= 0.30:
                amplitude_score = 70
            elif amplitude_cv <= 0.45:
                amplitude_score = 50
            elif amplitude_cv <= 0.60:
                amplitude_score = 30
            else:
                amplitude_score = 10

        else:
            amplitude_score = 0

    else:
        amplitude_score = 0

    # --------------------------------------------------------
    # NOISE
    # --------------------------------------------------------

    noise_input = (
        signal
        if noise_signal is None
        else np.asarray(noise_signal, dtype=float)
    )

    if len(noise_input) >= 11:
        smooth = (
            pd.Series(noise_input)
            .rolling(
                window=11,
                center=True,
                min_periods=1,
            )
            .mean()
            .to_numpy()
        )

        residual = (
            noise_input - smooth
        )

        signal_std = np.std(
            noise_input
        )

        noise_ratio = (
            np.std(residual)
            / signal_std
            if signal_std > 1e-9
            else 1
        )

        if noise_ratio <= 0.03:
            noise_score = 100
        elif noise_ratio <= 0.05:
            noise_score = 90
        elif noise_ratio <= 0.08:
            noise_score = 75
        elif noise_ratio <= 0.12:
            noise_score = 55
        elif noise_ratio <= 0.20:
            noise_score = 30
        else:
            noise_score = 10

    else:
        noise_score = 0

    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    score = float(
        np.clip(
            interval_score * 0.35
            + amplitude_score * 0.25
            + noise_score * 0.25
            + coverage_score * 0.15,
            0,
            100,
        )
    )

    if score >= 80:
        classification = "Good"
    elif score >= 60:
        classification = "Acceptable"
    else:
        classification = "Poor"

    result.update({
        "score": score,
        "classification": classification,
        "interval_score": interval_score,
        "amplitude_score": amplitude_score,
        "noise_score": noise_score,
        "coverage_score": min(
            coverage_score,
            100,
        ),
    })

    return result


# ============================================================
# EXTRACT PPG
# ============================================================

def extract_ppg(reading):
    cycle_columns = [
        "PPGValue cycle 1 (0-1500)",
        "PPGValue cycle 2 (1501-3000)",
        "PPGValue cycle 3 (3001-4500)",
        "PPGValue cycle 4 (4501-6000)",
        "PPGValue cycle 5 (6001-7500)",
    ]

    ppg = []

    for column in cycle_columns:
        if (
            column not in reading.index
            or pd.isna(reading[column])
        ):
            continue

        for value in str(
            reading[column]
        ).split(","):
            try:
                number = float(
                    value.strip()
                )

                if np.isfinite(number):
                    ppg.append(number)

            except (
                ValueError,
                TypeError,
            ):
                pass

    return np.asarray(
        ppg[:7500],
        dtype=float,
    )


# ============================================================
# COMPLETE READING ANALYSIS
# ============================================================

def analyze_samples(raw):
    """Full pipeline on a raw sample array (used by the app and tests)."""
    raw = np.asarray(raw, dtype=float)

    if len(raw) == 0:
        return None

    samples = np.arange(len(raw))

    # Flip so the pulse points up; the original stays in "Raw IR".
    oriented = POLARITY * raw

    dc_removed, baseline = remove_dc(oriented)

    filtered = bandpass_filter(dc_removed)

    peaks = detect_pulses(filtered)

    quality = calculate_quality(
        filtered,
        peaks,
        noise_signal=dc_removed,
    )

    waveform = pd.DataFrame({
        "Sample Number": samples,
        "Raw IR": raw,
        "Raw IR (flipped)": oriented,
        "Baseline": baseline,
        "DC Removed PPG": dc_removed,
        "Filtered PPG": filtered,
        "Detected Pulse": np.isin(
            samples,
            peaks,
        ).astype(int),
    })

    return {
        "waveform": waveform,
        "peaks": peaks,
        "quality": quality,
    }


def analyze_reading(reading):
    return analyze_samples(
        extract_ppg(
            reading
        )
    )
