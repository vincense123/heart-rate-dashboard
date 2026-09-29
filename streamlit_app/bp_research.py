"""
Cuffless BP research dataset builder.

Research/engineering use only.

Purpose:
Pair PPG-derived features from each selected recording with the
calibrated RMS cuff SBP/DBP values already present in the dataset.

Device BP is NOT used as the reference target.
"""

import re
import numpy as np
import pandas as pd

from ppg_analysis import analyze_reading
from ppg_features import extract_all_features


RMS_BP_CANDIDATES = [
    "BPCalibrationRef",
    "RMS BP",
    "RMSBP",
    "Calibrated RMS BP",
]


FEATURE_KEYS = [
    "Detected Pulses",
    "Mean IBI (s)",
    "IBI SD (s)",
    "IBI CV (%)",
    "Mean HR (BPM)",
    "Median HR (BPM)",
    "HR SD (BPM)",
    "HR CV (%)",
    "Mean Amplitude",
    "Amplitude SD",
    "Amplitude CV (%)",
    "Mean Pulse Width (s)",
    "Pulse Width SD (s)",
    "Mean Rise Time (s)",
    "Mean Fall Time (s)",
    "Mean Rise Slope",
    "Mean Fall Slope",
    "Mean Pulse Area",
    "Pulse Area CV (%)",
    "Samples",
    "Duration (s)",
    "Mean",
    "Std Dev",
    "RMS",
    "Peak-to-Peak",
    "Minimum",
    "Maximum",
]


def parse_bp(value):
    """Parse SBP/DBP from a value such as 130/80."""
    if pd.isna(value):
        return np.nan, np.nan

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)",
        str(value),
    )

    if not match:
        return np.nan, np.nan

    return float(match.group(1)), float(match.group(2))


def find_rms_bp_column(df):
    """Find the calibrated RMS BP reference column."""
    for col in RMS_BP_CANDIDATES:
        if col in df.columns:
            return col

    for col in df.columns:
        name = str(col).lower()
        if "rms" in name and "bp" in name:
            return col

    return None


def build_bp_feature_dataset(df, sampling_rate=500):
    """
    Create one row per valid recording:
    metadata + PPG features + RMS SBP/DBP target.

    No BP model is trained here.
    """
    rms_col = find_rms_bp_column(df)

    if rms_col is None:
        raise ValueError(
            "Calibrated RMS BP column not found. "
            "Expected BPCalibrationRef or an RMS BP column."
        )

    rows = []

    for _, reading in df.iterrows():

        rms_sbp, rms_dbp = parse_bp(
            reading.get(rms_col)
        )

        if pd.isna(rms_sbp) or pd.isna(rms_dbp):
            continue

        try:
            analysis = analyze_reading(reading)
        except Exception:
            continue

        if not analysis:
            continue

        # Only the 7,500-sample / 500 Hz VinCense format may enter
        # the BP dataset (never pad or resample shorter recordings).
        if len(analysis.get("waveform", [])) != 7500:
            continue

        waveform = analysis.get("waveform", {})

        if "Filtered PPG" not in waveform:
            continue

        filtered_ppg = waveform[
            "Filtered PPG"
        ].to_numpy(dtype=float)

        peaks = np.asarray(
            analysis.get("peaks", []),
            dtype=int,
        )

        if len(filtered_ppg) == 0 or len(peaks) < 2:
            continue

        try:
            result = extract_all_features(
                filtered_ppg=filtered_ppg,
                peaks=peaks,
                sampling_rate=sampling_rate,
            )
        except Exception:
            continue

        summary = result["summary"]

        row = {
            "Timestamp": reading.get("Timestamp"),
            "PatientId": reading.get("PatientId"),
            "PatientAge": reading.get("PatientAge"),
            "PatientGender": reading.get("PatientGender"),
            "Skin Tone": reading.get("PatientSkinTone"),
            "RMS Reference Flag": reading.get("RMS_Reference_Flag"),
            "PPG Samples": len(waveform),
            "RMS SBP": rms_sbp,
            "RMS DBP": rms_dbp,
        }

        for key in FEATURE_KEYS:
            row[key] = summary.get(key, np.nan)

        rows.append(row)

    return pd.DataFrame(rows), rms_col


def correlation_table(bp_df):
    """
    Exploratory Pearson correlation between PPG features
    and RMS SBP/DBP.
    """
    if bp_df is None or bp_df.empty:
        return pd.DataFrame()

    excluded = {
        "Timestamp",
        "PatientId",
        "PatientGender",
        "Skin Tone",
        "RMS Reference Flag",
        "PPG Samples",
        "RMS SBP",
        "RMS DBP",
    }

    numeric_features = [
        c for c in bp_df.columns
        if c not in excluded
        and pd.api.types.is_numeric_dtype(bp_df[c])
    ]

    rows = []

    for feature in numeric_features:

        for target in ["RMS SBP", "RMS DBP"]:

            pair = bp_df[
                [feature, target]
            ].dropna()

            if len(pair) < 3:
                corr = np.nan
            elif pair[feature].nunique() < 2:
                corr = np.nan
            else:
                corr = pair[feature].corr(
                    pair[target],
                    method="pearson",
                )

            rows.append({
                "Feature": feature,
                "Target": target,
                "N": len(pair),
                "Pearson r": corr,
                "Absolute |r|": (
                    abs(corr)
                    if pd.notna(corr)
                    else np.nan
                ),
            })

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            ["Target", "Absolute |r|"],
            ascending=[True, False],
        )

    return result
