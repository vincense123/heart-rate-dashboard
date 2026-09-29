"""
Synthetic PPG generator with a KNOWN heart rate.

Used to check that the analysis returns the right number when the
right answer is known. Signals are 15 s at 500 Hz (7,500 samples), like
the VinCense device, with a large DC offset and integer quantisation.
"""

import numpy as np

FS = 500
N = 7500


def _beat_template(kind, n):
    """One pulse, n samples long, systolic peak first."""
    t = np.linspace(0, 1, n, endpoint=False)

    if kind == "notch":
        # Sharp systolic upstroke + smaller diastolic (dicrotic) bump.
        return (
            np.exp(-0.5 * ((t - 0.18) / 0.075) ** 2)
            + 0.45 * np.exp(-0.5 * ((t - 0.47) / 0.10) ** 2)
        )

    if kind == "strong_notch":
        # Diastolic bump almost as tall as the systolic peak - the
        # classic cause of double counting.
        return (
            np.exp(-0.5 * ((t - 0.16) / 0.07) ** 2)
            + 0.70 * np.exp(-0.5 * ((t - 0.46) / 0.09) ** 2)
        )

    if kind == "sawtooth":
        # Slow rise, fast fall, like the raw VinCense recordings.
        rise = np.clip(t / 0.62, 0, 1) ** 1.3
        fall = np.clip((t - 0.62) / 0.14, 0, 1)
        shape = rise * (1 - fall)
        shape += 0.10 * np.exp(-0.5 * ((t - 0.86) / 0.05) ** 2)
        return shape

    raise ValueError(kind)


def make_ppg(
    hr,
    kind="notch",
    noise=0.02,
    wander=0.15,
    hrv=0.02,
    dc=300000.0,
    scale=2000.0,
    seed=0,
    seconds=N / FS,
    device_polarity=True,
    return_info=False,
):
    """
    Return (samples, true_hr), or (samples, true_hr, info) when
    return_info is True (info["systolic"] = sample index of each true
    systolic peak).

    device_polarity  the real VinCense IR signal FALLS when blood volume
                     rises, so the pulse points down in the raw data.
                     True (default) reproduces that; the "sawtooth"
                     template is already in device polarity.

    hr      true mean heart rate in BPM
    noise   white-noise std relative to pulse amplitude
    wander  baseline-wander amplitude relative to pulse amplitude
    hrv     beat-to-beat interval variation (std / mean)
    """
    rng = np.random.default_rng(seed)
    n = int(round(seconds * FS))
    signal = np.zeros(n)

    # Fraction of a beat at which the systolic peak sits (after the
    # analysis flips the device signal so the pulse points up).
    peak_fraction = {"notch": 0.18, "strong_notch": 0.16, "sawtooth": 0.76}[kind]
    systolic = []

    period = 60.0 / hr
    t_beat = rng.uniform(0, period)
    beat_times = []

    while t_beat < seconds + period:
        beat_times.append(t_beat)
        t_beat += period * (1 + hrv * rng.standard_normal())

    for i, start in enumerate(beat_times):
        length = (
            beat_times[i + 1] - start
            if i + 1 < len(beat_times)
            else period
        )
        m = max(8, int(round(length * FS)))
        first = int(round(start * FS))
        beat = _beat_template(kind, m) * (1 + 0.05 * rng.standard_normal())

        peak_index = first + int(round(peak_fraction * m))
        if 0 <= peak_index < n:
            systolic.append(peak_index)

        lo, hi = max(first, 0), min(first + m, n)
        if hi > lo:
            signal[lo:hi] += beat[lo - first:hi - first]

    time = np.arange(n) / FS
    signal += wander * np.sin(2 * np.pi * 0.25 * time + rng.uniform(0, 6.28))
    signal += 0.5 * wander * np.sin(2 * np.pi * 0.07 * time + rng.uniform(0, 6.28))
    signal += noise * rng.standard_normal(n)

    if device_polarity and kind != "sawtooth":
        signal = -signal

    samples = np.round(dc + scale * signal)

    # True rate, from the beat times that fall inside the recording.
    inside = np.array([b for b in beat_times if 0 <= b < seconds])
    true_hr = 60.0 / np.mean(np.diff(inside)) if len(inside) > 2 else hr

    if return_info:
        return samples, true_hr, {"systolic": np.array(systolic, dtype=int)}

    return samples, true_hr
