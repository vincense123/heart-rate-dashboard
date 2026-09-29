import io
import re

import matplotlib.pyplot as plt
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ppg_analysis import (
    analyze_reading,
    SAMPLE_RATE,
    LOW_CUTOFF,
    HIGH_CUTOFF,
)


def safe_filename(text):
    return re.sub(
        r'[\\/*?:"<>|]',
        "_",
        str(text),
    ).replace(
        " ",
        "_",
    )[:100]


def row_value(
    row,
    names,
    default="",
):
    for name in names:
        if (
            name in row.index
            and not pd.isna(row[name])
        ):
            return row[name]

    return default


def make_plot_png(
    waveform,
    y_column,
    title,
    peaks=None,
    marker_label=None,
):
    figure, axis = plt.subplots(
        figsize=(12, 4.5)
    )

    axis.plot(
        waveform["Sample Number"],
        waveform[y_column],
        linewidth=1.0,
        label=title,
    )

    if (
        peaks is not None
        and len(peaks) > 0
        and y_column == "Filtered PPG"
    ):
        axis.scatter(
            peaks,
            waveform[
                y_column
            ].iloc[peaks],
            s=18,
            label=marker_label or "Detected Pulse",
        )

    if y_column != "Raw IR":
        axis.axhline(
            0,
            linewidth=0.7,
        )

    axis.set_xlabel(
        "Sample Number"
    )

    axis.set_ylabel(
        "IR Value"
        if y_column in [
            "Raw IR",
            "Baseline",
        ]
        else "PPG Value"
    )

    axis.set_title(title)

    axis.grid(
        True,
        alpha=0.25,
    )

    axis.legend(
        loc="best"
    )

    figure.tight_layout()

    buffer = io.BytesIO()

    figure.savefig(
        buffer,
        format="png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    buffer.seek(0)

    return buffer.getvalue()


def waveform_png(
    analysis,
    number,
):
    waveform = analysis["waveform"]
    quality = analysis["quality"]
    peaks = analysis["peaks"]

    return make_plot_png(
        waveform,
        "Filtered PPG",
        (
            f"VinCense PPG - Reading {number} | "
            f"Quality {quality['score']:.1f}% "
            f"({quality['classification']})"
        ),
        peaks=peaks,
        marker_label="Detected Pulse",
    )



def _parse_bp_pair(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)",
        str(value).strip(),
    )
    if not match:
        return None

    return float(match.group(1)), float(match.group(2))


def _format_difference(value):
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.1f}"


def calculate_bp_difference(reading):
    """
    Absolute difference between Device BP and calibrated RMS BP.

    Example:
    Device BP = 115/70
    RMS BP    = 130/45
    Result    = 15/25
    """

    systolic = row_value(
        reading,
        ["BPSystolic", "BP Systolic"],
        "",
    )
    diastolic = row_value(
        reading,
        ["BPDiastolic", "BP Diastolic"],
        "",
    )

    device = _parse_bp_pair(
        f"{systolic}/{diastolic}"
    )

    rms = row_value(
        reading,
        [
            "BPCalibrationRef",
            "RMS BP",
            "RMSBP",
            "BPRMS",
            "Calibrated RMS BP",
            "BP Calibration RMS",
        ],
        "",
    )

    reference = _parse_bp_pair(rms)

    if device is None or reference is None:
        return ""

    sys_diff = abs(device[0] - reference[0])
    dia_diff = abs(device[1] - reference[1])

    return (
        f"{_format_difference(sys_diff)}/"
        f"{_format_difference(dia_diff)}"
    )



def reading_info_table(
    reading,
    analysis,
):
    quality = analysis["quality"]

    timestamp = row_value(
        reading,
        ["Timestamp"],
    )

    patient_id = row_value(
        reading,
        ["PatientId", "PatientID"],
    )

    patient_name = row_value(
        reading,
        ["PatientName", "Patient Name"],
    )

    age = row_value(
        reading,
        ["PatientAge", "Age"],
    )

    gender = row_value(
        reading,
        ["PatientGender", "Gender"],
    )

    skin = row_value(
        reading,
        [
            "PatientSkinTone",
            "SkinTone",
            "Skin Colour",
            "Skin Color",
        ],
    )

    systolic = row_value(
        reading,
        ["BPSystolic", "BP Systolic"],
    )

    diastolic = row_value(
        reading,
        ["BPDiastolic", "BP Diastolic"],
    )

    device_bp = (
        f"{systolic}/{diastolic}"
        if str(systolic).strip()
        and str(diastolic).strip()
        else ""
    )

    rms_bp = row_value(
        reading,
        [
            "RMS BP",
            "RMSBP",
            "BPRMS",
            "Calibrated RMS BP",
            "BPCalibrationRef",
            "BP Calibration RMS",
        ],
    )

    bp_difference = row_value(
        reading,
        [
            "BP Difference",
            "bp difference",
            "BPDifference",
        ],
    )

    device_hr = row_value(
        reading,
        [
            "HeartRate",
            "HR",
            "PulseRate",
        ],
    )

    estimated_hr = (
        f"{quality['heart_rate']:.1f}"
        if quality["heart_rate"] is not None
        else "Not reliable"
    )

    info = [
        [
            "Timestamp",
            str(timestamp),
            "Patient ID",
            str(patient_id),
            "Patient Name",
            str(patient_name),
        ],
        [
            "Age",
            str(age),
            "Gender",
            str(gender),
            "Skin Tone",
            str(skin),
        ],
        [
            "Device BP",
            str(device_bp),
            "RMS BP",
            str(rms_bp),
            "BP Difference",
            str(bp_difference),
        ],
        [
            "Device HR",
            str(device_hr),
            "Estimated HR",
            estimated_hr,
            "Detected Pulses",
            str(quality["peak_count"]),
        ],
        [
            "PPG Quality",
            f"{quality['score']:.1f}%",
            "Classification",
            quality["classification"],
            "Samples",
            str(len(analysis["waveform"])),
        ],
    ]

    table = Table(
        info,
        colWidths=[
            27 * mm,
            45 * mm,
            27 * mm,
            40 * mm,
            32 * mm,
            45 * mm,
        ],
    )

    table.setStyle(
        TableStyle([
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.35,
                colors.grey,
            ),
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.whitesmoke,
            ),
            (
                "BACKGROUND",
                (2, 0),
                (2, -1),
                colors.whitesmoke,
            ),
            (
                "BACKGROUND",
                (4, 0),
                (4, -1),
                colors.whitesmoke,
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                8,
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE",
            ),
        ])
    )

    return table


def summary_row(
    number,
    reading,
    analysis,
):
    quality = analysis["quality"]

    timestamp = row_value(
        reading,
        ["Timestamp"],
    )

    patient_id = row_value(
        reading,
        ["PatientId", "PatientID"],
    )

    patient_name = row_value(
        reading,
        ["PatientName", "Patient Name"],
    )

    age = row_value(
        reading,
        ["PatientAge", "Age"],
    )

    gender = row_value(
        reading,
        ["PatientGender", "Gender"],
    )

    skin = row_value(
        reading,
        [
            "PatientSkinTone",
            "SkinTone",
            "Skin Colour",
            "Skin Color",
        ],
    )

    systolic = row_value(
        reading,
        ["BPSystolic", "BP Systolic"],
    )

    diastolic = row_value(
        reading,
        ["BPDiastolic", "BP Diastolic"],
    )

    device_bp = (
        f"{systolic}/{diastolic}"
        if str(systolic).strip()
        and str(diastolic).strip()
        else ""
    )

    rms_bp = row_value(
        reading,
        [
            "RMS BP",
            "RMSBP",
            "BPRMS",
            "Calibrated RMS BP",
            "BPCalibrationRef",
            "BP Calibration RMS",
        ],
    )

    bp_difference = row_value(
        reading,
        [
            "BP Difference",
            "bp difference",
            "BPDifference",
        ],
    )

    device_hr = row_value(
        reading,
        [
            "HeartRate",
            "HR",
            "PulseRate",
        ],
    )

    estimated_hr = (
        f"{quality['heart_rate']:.1f}"
        if quality["heart_rate"] is not None
        else "Not reliable"
    )

    return [
        number,
        str(timestamp),
        str(patient_id),
        str(patient_name),
        str(age),
        device_bp,
        str(rms_bp),
        str(bp_difference),
        str(device_hr),
        estimated_hr,
        quality["peak_count"],
        f"{quality['score']:.1f}%",
        quality["classification"],
        str(gender),
        str(skin),
    ]


def build_summary_table(
    summary,
):
    headers = [
        "Reading",
        "Timestamp",
        "Patient ID",
        "Name",
        "Age",
        "Device BP",
        "RMS BP",
        "BP Diff",
        "Device HR",
        "Est. HR",
        "Pulses",
        "PPG %",
        "Quality",
        "Gender",
        "Skin",
    ]

    if not summary:
        return None

    table = Table(
        [headers] + summary,
        repeatRows=1,
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey,
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.25,
                colors.grey,
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                5.5,
            ),
            (
                "ALIGN",
                (0, 0),
                (-1, -1),
                "CENTER",
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE",
            ),
        ])
    )

    return table


def add_final_page(
    story,
    summary,
    title="Final Vitals & PPG Statistics",
):
    styles = getSampleStyleSheet()

    story.append(
        Paragraph(
            title,
            styles["Title"],
        )
    )

    story.append(
        Spacer(
            1,
            5 * mm,
        )
    )

    table = build_summary_table(
        summary
    )

    if table is not None:
        story.append(table)
    else:
        story.append(
            Paragraph(
                "No valid readings were available.",
                styles["Normal"],
            )
        )

    story.append(
        Spacer(
            1,
            5 * mm,
        )
    )

    story.append(
        Paragraph(
            "PPG quality is an engineering signal-quality "
            "index based on pulse consistency, amplitude "
            "consistency, noise and waveform coverage. "
            "It is not a medical diagnosis or clinical validation.",
            styles["Normal"],
        )
    )


def create_pdf(filtered_data):
    """
    Existing complete PDF export.

    Each reading contains the patient/vitals information and
    the main filtered PPG waveform, followed by a final
    vitals/statistics page.
    """

    pdf_buffer = io.BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=landscape(A4),
        rightMargin=10 * mm,
        leftMargin=10 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
    )

    styles = getSampleStyleSheet()
    story = []
    summary = []

    story.append(
        Paragraph(
            "VinCense PPG Waveform Analysis",
            styles["Title"],
        )
    )

    story.append(
        Spacer(
            1,
            5 * mm,
        )
    )

    story.append(
        Paragraph(
            f"Sampling rate: {SAMPLE_RATE} Hz | "
            f"Band-pass: {LOW_CUTOFF}-{HIGH_CUTOFF} Hz",
            styles["Normal"],
        )
    )

    story.append(
        PageBreak()
    )

    for number, (_, reading) in enumerate(
        filtered_data.iterrows(),
        start=1,
    ):
        analysis = analyze_reading(
            reading
        )

        if analysis is None:
            continue

        story.append(
            Paragraph(
                f"Reading {number}",
                styles["Heading2"],
            )
        )

        story.append(
            reading_info_table(
                reading,
                analysis,
            )
        )

        story.append(
            Spacer(
                1,
                4 * mm,
            )
        )

        png = waveform_png(
            analysis,
            number,
        )

        story.append(
            Image(
                io.BytesIO(png),
                width=255 * mm,
                height=117 * mm,
            )
        )

        story.append(
            PageBreak()
        )

        summary.append(
            summary_row(
                number,
                reading,
                analysis,
            )
        )

    add_final_page(
        story,
        summary,
    )

    document.build(
        story
    )

    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()


def create_current_view_pdf(filtered_data):
    """
    New current-view export.

    Exports exactly the readings currently visible after
    Patient ID / Patient Name / Date filtering.

    For every reading the PDF contains:

    1. Raw IR
    2. DC-Removed PPG
    3. Filtered PPG

    It also includes the selected reading's vitals and PPG
    quality information, then a final summary table.
    """

    pdf_buffer = io.BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=landscape(A4),
        rightMargin=8 * mm,
        leftMargin=8 * mm,
        topMargin=8 * mm,
        bottomMargin=8 * mm,
    )

    styles = getSampleStyleSheet()
    story = []
    summary = []

    story.append(
        Paragraph(
            "VinCense Current View – Raw / DC / Filtered PPG",
            styles["Title"],
        )
    )

    story.append(
        Spacer(
            1,
            3 * mm,
        )
    )

    story.append(
        Paragraph(
            f"Current filtered readings: {len(filtered_data)} | "
            f"Sampling rate: {SAMPLE_RATE} Hz | "
            f"Band-pass: {LOW_CUTOFF}-{HIGH_CUTOFF} Hz",
            styles["Normal"],
        )
    )

    story.append(
        PageBreak()
    )

    for number, (_, reading) in enumerate(
        filtered_data.iterrows(),
        start=1,
    ):
        analysis = analyze_reading(
            reading
        )

        if analysis is None:
            continue

        waveform = analysis["waveform"]
        peaks = analysis["peaks"]
        quality = analysis["quality"]

        timestamp = row_value(
            reading,
            ["Timestamp"],
        )

        story.append(
            Paragraph(
                f"Reading {number} — {timestamp}",
                styles["Heading2"],
            )
        )

        story.append(
            reading_info_table(
                reading,
                analysis,
            )
        )

        story.append(
            Spacer(
                1,
                3 * mm,
            )
        )

        # 1. RAW
        raw_png = make_plot_png(
            waveform,
            "Raw IR",
            f"1. Raw IR — Reading {number}",
        )

        story.append(
            Image(
                io.BytesIO(raw_png),
                width=260 * mm,
                height=72 * mm,
            )
        )

        story.append(
            Spacer(
                1,
                2 * mm,
            )
        )

        # 2. DC REMOVED
        dc_png = make_plot_png(
            waveform,
            "DC Removed PPG",
            f"2. DC-Removed PPG — Reading {number}",
        )

        story.append(
            Image(
                io.BytesIO(dc_png),
                width=260 * mm,
                height=72 * mm,
            )
        )

        story.append(
            PageBreak()
        )

        # 3. FILTERED
        filtered_png = make_plot_png(
            waveform,
            "Filtered PPG",
            (
                f"3. Filtered PPG — Reading {number} | "
                f"Quality {quality['score']:.1f}% "
                f"({quality['classification']})"
            ),
            peaks=peaks,
            marker_label="Detected Pulse",
        )

        story.append(
            Image(
                io.BytesIO(filtered_png),
                width=260 * mm,
                height=85 * mm,
            )
        )

        story.append(
            Spacer(
                1,
                4 * mm,
            )
        )

        # Small analysis summary
        analysis_table = Table([
            [
                "PPG Quality",
                f"{quality['score']:.1f}%",
                "Classification",
                quality["classification"],
                "Detected Pulses",
                str(quality["peak_count"]),
            ],
            [
                "Pulse Interval",
                f"{quality['interval_score']:.0f}%",
                "Amplitude",
                f"{quality['amplitude_score']:.0f}%",
                "Noise",
                f"{quality['noise_score']:.0f}%",
            ],
            [
                "Coverage",
                f"{quality['coverage_score']:.0f}%",
                "Estimated HR",
                (
                    f"{quality['heart_rate']:.1f} BPM"
                    if quality["heart_rate"] is not None
                    else "Not reliable"
                ),
                "Samples",
                str(len(waveform)),
            ],
        ])

        analysis_table.setStyle(
            TableStyle([
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.35,
                    colors.grey,
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.whitesmoke,
                ),
                (
                    "BACKGROUND",
                    (2, 0),
                    (2, -1),
                    colors.whitesmoke,
                ),
                (
                    "BACKGROUND",
                    (4, 0),
                    (4, -1),
                    colors.whitesmoke,
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER",
                ),
            ])
        )

        story.append(
            analysis_table
        )

        summary.append(
            summary_row(
                number,
                reading,
                analysis,
            )
        )

        story.append(
            PageBreak()
        )

    add_final_page(
        story,
        summary,
        title="Current View – Final Vitals & Statistics",
    )

    document.build(
        story
    )

    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()


# ============================================================
# CURRENT UNIQUE PATIENT - FULL DETAILED PDF
# ============================================================

def create_current_patient_detailed_pdf(patient_data):
    """
    Detailed PDF for the current unique patient view.

    The caller should pass data that contains only one unique
    patient after the application's filters are applied.

    For every reading the PDF contains:
        1. Raw IR waveform
        2. DC-Removed PPG waveform
        3. Filtered PPG waveform with detected pulses
        4. Remaining vitals + BP/RMS comparison + PPG analysis

    A final page contains the complete patient summary table.
    """

    if patient_data is None or len(patient_data) == 0:
        return b""

    # --------------------------------------------------------
    # Safety: require one unique patient
    # --------------------------------------------------------

    patient_key_columns = [
        column
        for column in ["PatientId", "PatientName"]
        if column in patient_data.columns
    ]

    if patient_key_columns:
        unique_patients = (
            patient_data[patient_key_columns]
            .drop_duplicates()
        )

        if len(unique_patients) != 1:
            raise ValueError(
                "Current Patient PDF requires exactly one unique patient. "
                "Please select one Patient ID/Patient Name in the filters."
            )

    pdf_buffer = io.BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=landscape(A4),
        rightMargin=8 * mm,
        leftMargin=8 * mm,
        topMargin=8 * mm,
        bottomMargin=8 * mm,
    )

    styles = getSampleStyleSheet()
    story = []
    summary = []

    first_reading = patient_data.iloc[0]

    patient_id = row_value(
        first_reading,
        ["PatientId", "PatientID"],
        "Unknown",
    )

    patient_name = row_value(
        first_reading,
        ["PatientName", "Patient Name"],
        "Unknown",
    )

    # --------------------------------------------------------
    # COVER / PATIENT SUMMARY
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "VinCense - Current Patient Full PPG Analysis",
            styles["Title"],
        )
    )

    story.append(
        Spacer(
            1,
            4 * mm,
        )
    )

    story.append(
        Paragraph(
            f"Patient ID: {patient_id} | "
            f"Patient Name: {patient_name} | "
            f"Total readings: {len(patient_data)}",
            styles["Heading2"],
        )
    )

    story.append(
        Spacer(
            1,
            2 * mm,
        )
    )

    story.append(
        Paragraph(
            f"Sampling rate: {SAMPLE_RATE} Hz | "
            f"Band-pass filter: {LOW_CUTOFF}-{HIGH_CUTOFF} Hz | "
            "Filter order: 4",
            styles["Normal"],
        )
    )

    story.append(
        Spacer(
            1,
            4 * mm,
        )
    )

    story.append(
        Paragraph(
            "This report follows the current patient view in the app. "
            "Each reading is presented as Raw IR, DC-Removed PPG, "
            "Filtered PPG, followed by vitals and signal analysis.",
            styles["Normal"],
        )
    )

    story.append(
        PageBreak()
    )

    # --------------------------------------------------------
    # EVERY READING
    # --------------------------------------------------------

    for number, (_, reading) in enumerate(
        patient_data.iterrows(),
        start=1,
    ):
        analysis = analyze_reading(
            reading
        )

        if analysis is None:
            continue

        waveform = analysis["waveform"]
        peaks = analysis["peaks"]
        quality = analysis["quality"]

        timestamp = row_value(
            reading,
            ["Timestamp"],
            "",
        )

        # ====================================================
        # READING HEADER + RAW
        # ====================================================

        story.append(
            Paragraph(
                f"Reading {number} — {timestamp}",
                styles["Heading1"],
            )
        )

        story.append(
            reading_info_table(
                reading,
                analysis,
            )
        )

        story.append(
            Spacer(
                1,
                3 * mm,
            )
        )

        story.append(
            Paragraph(
                "1. Raw IR",
                styles["Heading2"],
            )
        )

        raw_png = make_plot_png(
            waveform,
            "Raw IR",
            f"Raw IR — Reading {number}",
        )

        story.append(
            Image(
                io.BytesIO(raw_png),
                width=260 * mm,
                height=72 * mm,
            )
        )

        story.append(
            Spacer(
                1,
                2 * mm,
            )
        )

        # ====================================================
        # DC REMOVED
        # ====================================================

        story.append(
            Paragraph(
                "2. DC-Removed PPG",
                styles["Heading2"],
            )
        )

        dc_png = make_plot_png(
            waveform,
            "DC Removed PPG",
            f"DC-Removed PPG — Reading {number}",
        )

        story.append(
            Image(
                io.BytesIO(dc_png),
                width=260 * mm,
                height=72 * mm,
            )
        )

        story.append(
            PageBreak()
        )

        # ====================================================
        # FILTERED
        # ====================================================

        story.append(
            Paragraph(
                "3. Filtered PPG",
                styles["Heading1"],
            )
        )

        filtered_title = (
            f"Filtered PPG — Reading {number} | "
            f"Quality {quality['score']:.1f}% "
            f"({quality['classification']})"
        )

        filtered_png = make_plot_png(
            waveform,
            "Filtered PPG",
            filtered_title,
            peaks=peaks,
            marker_label="Detected Pulse",
        )

        story.append(
            Image(
                io.BytesIO(filtered_png),
                width=260 * mm,
                height=92 * mm,
            )
        )

        story.append(
            Spacer(
                1,
                3 * mm,
            )
        )

        # ====================================================
        # REMAINING VITALS + ANALYSIS
        # ====================================================

        story.append(
            Paragraph(
                "4. Vitals & Signal Analysis",
                styles["Heading2"],
            )
        )

        systolic = row_value(reading, ["BPSystolic", "BP Systolic"], "")
        diastolic = row_value(reading, ["BPDiastolic", "BP Diastolic"], "")
        device_bp = (
            f"{systolic}/{diastolic}"
            if str(systolic).strip() and str(diastolic).strip()
            else ""
        )

        rms_bp = row_value(
            reading,
            [
                "RMS BP", "RMSBP", "BPRMS",
                "Calibrated RMS BP", "BPCalibrationRef",
                "BP Calibration RMS",
            ],
            "",
        )

        bp_difference = calculate_bp_difference(reading)

        if not bp_difference:
            bp_difference = row_value(
                reading,
                ["BP Difference", "bp difference", "BPDifference"],
                "",
            )

        device_hr = row_value(
            reading,
            [
                "HeartRate",
                "HR",
                "PulseRate",
            ],
            "",
        )

        estimated_hr = (
            f"{quality['heart_rate']:.1f} BPM"
            if quality["heart_rate"] is not None
            else "Not reliable"
        )

        spo2 = row_value(
            reading,
            [
                "SpO2",
                "SPO2",
                "OxygenSaturation",
            ],
            "",
        )

        temperature = row_value(
            reading,
            [
                "Temperature",
                "Temp",
                "SkinTemperature",
            ],
            "",
        )

        respiratory_rate = row_value(
            reading,
            [
                "RespiratoryRate",
                "RR",
                "RespRate",
            ],
            "",
        )

        vitals_analysis = [
            [
                "Device BP",
                str(device_bp),
                "Calibrated RMS BP",
                str(rms_bp),
                "BP Difference",
                str(bp_difference),
            ],
            [
                "Device HR",
                str(device_hr),
                "Estimated PPG HR",
                estimated_hr,
                "Detected Pulses",
                str(quality["peak_count"]),
            ],
            [
                "SpO2",
                str(spo2),
                "Temperature",
                str(temperature),
                "Respiratory Rate",
                str(respiratory_rate),
            ],
            [
                "PPG Quality",
                f"{quality['score']:.1f}%",
                "Classification",
                quality["classification"],
                "Samples",
                str(len(waveform)),
            ],
            [
                "Pulse Interval",
                f"{quality['interval_score']:.0f}%",
                "Amplitude",
                f"{quality['amplitude_score']:.0f}%",
                "Noise",
                f"{quality['noise_score']:.0f}%",
            ],
            [
                "Coverage",
                f"{quality['coverage_score']:.0f}%",
                "Sampling Rate",
                f"{SAMPLE_RATE} Hz",
                "Filter",
                f"{LOW_CUTOFF}-{HIGH_CUTOFF} Hz",
            ],
        ]

        vitals_table = Table(
            vitals_analysis,
            colWidths=[
                34 * mm,
                43 * mm,
                43 * mm,
                43 * mm,
                35 * mm,
                43 * mm,
            ],
        )

        vitals_table.setStyle(
            TableStyle([
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.35,
                    colors.grey,
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.whitesmoke,
                ),
                (
                    "BACKGROUND",
                    (2, 0),
                    (2, -1),
                    colors.whitesmoke,
                ),
                (
                    "BACKGROUND",
                    (4, 0),
                    (4, -1),
                    colors.whitesmoke,
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    7.5,
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER",
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
            ])
        )

        story.append(
            vitals_table
        )

        story.append(
            Spacer(
                1,
                3 * mm,
            )
        )

        story.append(
            Paragraph(
                "PPG Analysis: "
                f"pulse consistency {quality['interval_score']:.0f}%, "
                f"amplitude consistency {quality['amplitude_score']:.0f}%, "
                f"noise score {quality['noise_score']:.0f}%, "
                f"waveform coverage {quality['coverage_score']:.0f}%. "
                f"Overall engineering signal-quality score: "
                f"{quality['score']:.1f}% "
                f"({quality['classification']}).",
                styles["Normal"],
            )
        )

        story.append(
            Spacer(
                1,
                2 * mm,
            )
        )

        story.append(
            Paragraph(
                "BP comparison uses the Device BP and calibrated RMS "
                "BP values available in the uploaded sheet. No new "
                "medical calibration or clinical correction is applied "
                "by this export.",
                styles["Normal"],
            )
        )

        story.append(
            PageBreak()
        )

        summary.append(
            summary_row(
                number,
                reading,
                analysis,
            )
        )

    # --------------------------------------------------------
    # FINAL PATIENT SUMMARY
    # --------------------------------------------------------

    add_final_page(
        story,
        summary,
        title="Current Patient - Final Vitals & PPG Statistics",
    )

    document.build(
        story
    )

    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()
