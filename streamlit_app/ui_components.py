import streamlit as st
import plotly.graph_objects as go


def show_selected_details(reading):
    detail_columns = [
        "PatientId",
        "PatientName",
        "PatientAge",
        "BPCalibrationRef",
        "bp difference",
        "HeartRate",
        "Temperature",
        "SpO2",
        "RespiratoryRate",
        "BPSystolic",
        "BPDiastolic",
        "PatientGender",
        "PatientSkinTone",
    ]

    available = [
        column
        for column in detail_columns
        if column in reading.index
    ]

    with st.expander(
        "Selected Reading Details",
        expanded=True,
    ):
        st.dataframe(
            {
                "Parameter": available,
                "Value": [
                    reading[column]
                    for column in available
                ],
            },
            use_container_width=True,
            hide_index=True,
        )


def show_raw_graph(analysis):
    waveform = analysis["waveform"]

    st.subheader(
        "1. Raw IR Signal"
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=waveform["Sample Number"],
            y=waveform["Raw IR"],
            mode="lines",
            name="Raw IR",
        )
    )

    figure.update_layout(
        height=500,
        xaxis_title="Sample Number",
        yaxis_title="IR Value",
        hovermode="x unified",
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )


def show_dc_graph(analysis):
    waveform = analysis["waveform"]

    st.subheader(
        "2. DC-Removed PPG"
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=waveform["Sample Number"],
            y=waveform["DC Removed PPG"],
            mode="lines",
            name="DC Removed PPG",
        )
    )

    figure.add_hline(
        y=0,
        line_width=1,
    )

    figure.update_layout(
        height=500,
        xaxis_title="Sample Number",
        yaxis_title="DC-Removed IR",
        hovermode="x unified",
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )


def show_filtered_graph(analysis):
    waveform = analysis["waveform"]

    st.subheader(
        "3. Filtered PPG"
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=waveform["Sample Number"],
            y=waveform["Filtered PPG"],
            mode="lines",
            name="Filtered PPG",
        )
    )

    figure.add_hline(
        y=0,
        line_width=1,
    )

    figure.update_layout(
        height=500,
        xaxis_title="Sample Number",
        yaxis_title="Filtered PPG",
        hovermode="x unified",
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )


def show_pulse_graph(analysis):
    waveform = analysis["waveform"]
    peaks = analysis["peaks"]

    st.subheader(
        "4. Pulse Detection"
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=waveform["Sample Number"],
            y=waveform["Filtered PPG"],
            mode="lines",
            name="Filtered PPG",
        )
    )

    if len(peaks):
        figure.add_trace(
            go.Scatter(
                x=peaks,
                y=waveform[
                    "Filtered PPG"
                ].iloc[peaks],
                mode="markers",
                name="Detected Pulses",
                marker=dict(size=8),
            )
        )

    figure.add_hline(
        y=0,
        line_width=1,
    )

    figure.update_layout(
        height=500,
        xaxis_title="Sample Number",
        yaxis_title="Filtered PPG",
        hovermode="x unified",
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )


def show_quality(analysis):
    quality = analysis["quality"]

    st.subheader(
        "5. PPG Signal Quality"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Signal Quality",
            f"{quality['score']:.1f}%",
        )

    with col2:
        st.metric(
            "Detected Pulses",
            quality["peak_count"],
        )

    with col3:
        if quality["heart_rate"] is not None:
            st.metric(
                "Estimated Pulse Rate",
                f"{quality['heart_rate']:.1f} BPM",
            )
        else:
            st.metric(
                "Estimated Pulse Rate",
                "Not reliable",
            )

    if quality["classification"] == "Good":
        st.success(
            "PPG Signal Quality: GOOD"
        )
    elif quality["classification"] == "Acceptable":
        st.warning(
            "PPG Signal Quality: ACCEPTABLE"
        )
    else:
        st.error(
            "PPG Signal Quality: POOR"
        )

    st.subheader(
        "Quality Components"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Pulse Interval",
        f"{quality['interval_score']:.0f}%",
    )

    col2.metric(
        "Amplitude",
        f"{quality['amplitude_score']:.0f}%",
    )

    col3.metric(
        "Noise",
        f"{quality['noise_score']:.0f}%",
    )

    col4.metric(
        "Coverage",
        f"{quality['coverage_score']:.0f}%",
    )

    with st.expander(
        "Technical Analysis"
    ):
        if quality["median_interval"] is not None:
            st.write(
                "Median pulse interval: "
                f"{quality['median_interval']:.3f} seconds"
            )

        if quality["interval_cv"] is not None:
            st.write(
                "Pulse interval variation: "
                f"{quality['interval_cv'] * 100:.2f}%"
            )

        st.write(
            "Sampling rate: 500 Hz"
        )

        st.write(
            "Band-pass: 0.5-8.0 Hz"
        )

        st.write(
            "Filter order: 4"
        )

    st.info(
        "The PPG quality percentage is an engineering "
        "signal-quality index based on pulse consistency, "
        "amplitude consistency, noise and waveform coverage. "
        "It is not a medical diagnosis or clinical validation."
    )
