import streamlit as st
import pandas as pd
import numpy as np

from ppg_analysis import analyze_reading

from ppg_features import extract_all_features
from bp_research import build_bp_feature_dataset, correlation_table

from filters import apply_filters

from pdf_export import (
    create_pdf,
    create_current_view_pdf,
    create_current_patient_detailed_pdf,
)

from ui_components import (
    show_selected_details,
    show_raw_graph,
    show_dc_graph,
    show_filtered_graph,
    show_pulse_graph,
    show_quality,
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="VinCense PPG Waveform Analyzer",
    layout="wide",
)

st.title(
    "VinCense PPG Waveform Analyzer"
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload VinCense Excel File",
    type=["xlsx"],
)


if uploaded_file:

    excel = pd.ExcelFile(
        uploaded_file
    )

    sheet_name = st.selectbox(
        "Select Data Sheet",
        excel.sheet_names,
    )

    df = pd.read_excel(
        uploaded_file,
        sheet_name=sheet_name,
    )

    # Column headers in real exports can carry stray spaces
    # (e.g. "bp difference "), which silently hid that column.
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    st.success(
        "File Loaded"
    )


    # ========================================================
    # FILTERS
    # ========================================================

    filtered_df = apply_filters(
        df
    )

    st.subheader(
        f"Available Readings : {len(filtered_df)}"
    )


    # ========================================================
    # PDF EXPORT BUTTONS
    # ========================================================

    if len(filtered_df) > 0:

        col1, col2, col3 = st.columns(3)


        # ----------------------------------------------------
        # ALL PDF
        # ----------------------------------------------------

        with col1:

            if st.button(
                "Export ALL PPG Analysis PDF",
                use_container_width=True,
            ):

                with st.spinner(
                    "Creating complete PPG analysis PDF..."
                ):

                    st.session_state[
                        "ppg_pdf"
                    ] = create_pdf(
                        filtered_df
                    )


            if (
                "ppg_pdf"
                in st.session_state
            ):

                st.download_button(
                    "Download ALL PPG Analysis PDF",
                    st.session_state[
                        "ppg_pdf"
                    ],
                    file_name=(
                        "VinCense_PPG_Waveforms_and_Vitals_Stats.pdf"
                    ),
                    mime="application/pdf",
                    use_container_width=True,
                    key="download_all_pdf",
                )


        # ----------------------------------------------------
        # CURRENT VIEW PDF
        # ----------------------------------------------------

        with col2:

            if st.button(
                "Export CURRENT VIEW – Raw + DC + Filtered",
                use_container_width=True,
            ):

                with st.spinner(
                    "Creating current-view waveform PDF..."
                ):

                    st.session_state[
                        "current_view_pdf"
                    ] = create_current_view_pdf(
                        filtered_df
                    )


            if (
                "current_view_pdf"
                in st.session_state
            ):

                st.download_button(
                    "Download CURRENT VIEW PDF",
                    st.session_state[
                        "current_view_pdf"
                    ],
                    file_name=(
                        "VinCense_Current_View_Raw_DC_Filtered.pdf"
                    ),
                    mime="application/pdf",
                    use_container_width=True,
                    key="download_current_view_pdf",
                )


        # ----------------------------------------------------
        # INFORMATION
        # ----------------------------------------------------

        with col3:

            st.caption(
                "Export only the reading selected in the "
                "clickable table."
            )


        st.caption(
            "ALL PDF = all filtered readings. "
            "CURRENT VIEW PDF = all readings currently visible. "
            "For a single-reading report, select a row below "
            "and use the Selected Reading PDF button."
        )


    else:

        st.warning(
            "No readings available for export."
        )


    # ========================================================
    # CLICKABLE TABLE
    # ========================================================

    display_columns = [

        "Timestamp",
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

    ]


    display_columns = [
        column
        for column in display_columns
        if column in filtered_df.columns
    ]


    table = (
        filtered_df[
            display_columns
        ]
        .reset_index(drop=True)
    )


    event = st.dataframe(

        table,

        use_container_width=True,

        height=400,

        on_select="rerun",

        selection_mode="single-row",

    )


    # ========================================================
    # SELECTED READING
    # ========================================================

    if event.selection.rows:

        row_number = (
            event.selection.rows[0]
        )


        # Look the reading up by its position in the filtered table.
        # (Matching on Timestamp alone could open a different
        # patient's reading if two rows ever share a timestamp.)
        reading = filtered_df.iloc[
            row_number
        ]

        selected_timestamp = reading.get(
            "Timestamp"
        )


        st.success(
            f"Selected Reading : {selected_timestamp}"
        )


        # ====================================================
        # EXISTING SELECTED DETAILS
        # ====================================================

        show_selected_details(
            reading
        )


        # ====================================================
        # EXISTING PPG ANALYSIS
        # ====================================================

        analysis = analyze_reading(
            reading
        )


        if analysis is None:

            st.error(
                "No valid PPG samples found."
            )

            st.stop()


        # Existing graphs.
        show_raw_graph(
            analysis
        )

        show_dc_graph(
            analysis
        )

        show_filtered_graph(
            analysis
        )

        show_pulse_graph(
            analysis
        )

        show_quality(
            analysis
        )


        # ====================================================
        # NEW PPG FEATURE ANALYSIS
        # ====================================================

        st.markdown(
            "---"
        )

        st.header(
            "6. PPG Feature Analysis"
        )

        st.caption(
            "Engineering analysis calculated from the selected "
            "PPG waveform. These values are for research/testing "
            "and are not clinical validation."
        )


        # IMPORTANT:
        # The existing ppg_analysis.py stores the same filtered
        # waveform used by the graph inside:
        #
        # analysis["waveform"]["Filtered PPG"]
        #
        # Pulse locations are stored in:
        #
        # analysis["peaks"]

        waveform = analysis[
            "waveform"
        ]

        filtered_ppg = (
            waveform[
                "Filtered PPG"
            ]
            .to_numpy(
                dtype=float
            )
        )

        detected_peaks = np.asarray(
            analysis[
                "peaks"
            ],
            dtype=int,
        )


        # ----------------------------------------------------
        # RUN FEATURE ENGINE
        # ----------------------------------------------------

        try:

            feature_result = (
                extract_all_features(
                    filtered_ppg=filtered_ppg,
                    peaks=detected_peaks,
                    sampling_rate=500,
                )
            )


            feature_summary = (
                feature_result[
                    "summary"
                ]
            )


            pulse_features = (
                feature_result[
                    "pulse_features"
                ]
            )


            # =================================================
            # HEART RATE & TIMING
            # =================================================

            st.subheader(
                "Heart Rate & Pulse Timing"
            )


            col1, col2, col3, col4 = (
                st.columns(4)
            )


            with col1:

                value = (
                    feature_summary.get(
                        "Mean HR (BPM)",
                        np.nan,
                    )
                )

                st.metric(
                    "Mean HR",
                    "N/A"
                    if pd.isna(value)
                    else f"{value:.1f} BPM",
                )


            with col2:

                value = (
                    feature_summary.get(
                        "Median HR (BPM)",
                        np.nan,
                    )
                )

                st.metric(
                    "Median HR",
                    "N/A"
                    if pd.isna(value)
                    else f"{value:.1f} BPM",
                )


            with col3:

                value = (
                    feature_summary.get(
                        "Mean IBI (s)",
                        np.nan,
                    )
                )

                st.metric(
                    "Mean IBI",
                    "N/A"
                    if pd.isna(value)
                    else f"{value:.3f} s",
                )


            with col4:

                value = (
                    feature_summary.get(
                        "IBI CV (%)",
                        np.nan,
                    )
                )

                st.metric(
                    "IBI Variation",
                    "N/A"
                    if pd.isna(value)
                    else f"{value:.1f}%",
                )


            timing_table = pd.DataFrame(
                [
                    {
                        "Detected Pulses":
                            feature_summary.get(
                                "Detected Pulses"
                            ),
                        "HR SD (BPM)":
                            feature_summary.get(
                                "HR SD (BPM)"
                            ),
                        "HR CV (%)":
                            feature_summary.get(
                                "HR CV (%)"
                            ),
                        "IBI SD (s)":
                            feature_summary.get(
                                "IBI SD (s)"
                            ),
                    }
                ]
            )


            st.dataframe(
                timing_table,
                use_container_width=True,
                hide_index=True,
            )


            # =================================================
            # PULSE MORPHOLOGY
            # =================================================

            st.subheader(
                "Pulse Morphology"
            )


            col1, col2, col3, col4 = (
                st.columns(4)
            )


            with col1:

                value = (
                    feature_summary.get(
                        "Mean Amplitude",
                        np.nan,
                    )
                )

                st.metric(
                    "Mean Amplitude",
                    "N/A"
                    if pd.isna(value)
                    else f"{value:.2f}",
                )


            with col2:

                value = (
                    feature_summary.get(
                        "Amplitude CV (%)",
                        np.nan,
                    )
                )

                st.metric(
                    "Amplitude Variation",
                    "N/A"
                    if pd.isna(value)
                    else f"{value:.1f}%",
                )


            with col3:

                value = (
                    feature_summary.get(
                        "Mean Pulse Width (s)",
                        np.nan,
                    )
                )

                st.metric(
                    "Pulse Width",
                    "N/A"
                    if pd.isna(value)
                    else f"{value:.3f} s",
                )


            with col4:

                value = (
                    feature_summary.get(
                        "Pulse Area CV (%)",
                        np.nan,
                    )
                )

                st.metric(
                    "Pulse Area Variation",
                    "N/A"
                    if pd.isna(value)
                    else f"{value:.1f}%",
                )


            morphology_table = pd.DataFrame(
                [
                    {
                        "Mean Rise Time (s)":
                            feature_summary.get(
                                "Mean Rise Time (s)"
                            ),
                        "Mean Fall Time (s)":
                            feature_summary.get(
                                "Mean Fall Time (s)"
                            ),
                        "Mean Rise Slope":
                            feature_summary.get(
                                "Mean Rise Slope"
                            ),
                        "Mean Fall Slope":
                            feature_summary.get(
                                "Mean Fall Slope"
                            ),
                        "Mean Pulse Area":
                            feature_summary.get(
                                "Mean Pulse Area"
                            ),
                        "Pulse Width SD (s)":
                            feature_summary.get(
                                "Pulse Width SD (s)"
                            ),
                    }
                ]
            )


            st.dataframe(
                morphology_table,
                use_container_width=True,
                hide_index=True,
            )


            # =================================================
            # WAVEFORM STATISTICS
            # =================================================

            st.subheader(
                "Waveform Statistics"
            )


            col1, col2, col3, col4 = (
                st.columns(4)
            )


            with col1:

                st.metric(
                    "Samples",
                    str(
                        feature_summary.get(
                            "Samples",
                            len(filtered_ppg),
                        )
                    ),
                )


            with col2:

                value = (
                    feature_summary.get(
                        "Duration (s)",
                        np.nan,
                    )
                )

                st.metric(
                    "Duration",
                    "N/A"
                    if pd.isna(value)
                    else f"{value:.2f} s",
                )


            with col3:

                value = (
                    feature_summary.get(
                        "Peak-to-Peak",
                        np.nan,
                    )
                )

                st.metric(
                    "Peak-to-Peak",
                    "N/A"
                    if pd.isna(value)
                    else f"{value:.2f}",
                )


            with col4:

                value = (
                    feature_summary.get(
                        "RMS",
                        np.nan,
                    )
                )

                st.metric(
                    "RMS",
                    "N/A"
                    if pd.isna(value)
                    else f"{value:.2f}",
                )


            # =================================================
            # PULSE-BY-PULSE TABLE
            # =================================================

            if (
                pulse_features is not None
                and not pulse_features.empty
            ):

                with st.expander(
                    "Pulse-by-Pulse Feature Table"
                ):

                    st.dataframe(
                        pulse_features,
                        use_container_width=True,
                        hide_index=True,
                    )


        except Exception as error:

            st.error(
                "PPG feature analysis could not be calculated."
            )

            st.code(
                str(error)
            )


        # ====================================================
        # ====================================================
        # CUFFLESS BP RESEARCH
        # ====================================================

        st.markdown("---")

        st.header("7. Cuffless BP Research")

        st.caption(
            "PPG-derived features are paired with the calibrated RMS "
            "cuff SBP/DBP values in the uploaded sheet. RMS BP is the "
            "reference target for this research stage. No medical "
            "correction or clinical validation is claimed."
        )

        if st.button(
            "Build PPG → RMS BP Research Dataset",
            use_container_width=True,
            key="build_bp_research_dataset",
        ):

            with st.spinner(
                "Extracting PPG features and pairing them with RMS BP..."
            ):

                try:

                    bp_dataset, rms_column = build_bp_feature_dataset(
                        filtered_df,
                        sampling_rate=500,
                    )

                    st.session_state["bp_research_dataset"] = bp_dataset
                    st.session_state["bp_rms_column"] = rms_column
                    st.session_state["bp_correlation_table"] = (
                        correlation_table(bp_dataset)
                    )

                except Exception as error:

                    st.error(
                        "BP research dataset could not be created."
                    )
                    st.code(str(error))

        if "bp_research_dataset" in st.session_state:

            bp_dataset = st.session_state["bp_research_dataset"]

            rms_column = st.session_state.get(
                "bp_rms_column",
                "BPCalibrationRef",
            )

            st.success(
                f"Valid BP research recordings: {len(bp_dataset)} | "
                f"RMS reference column: {rms_column}"
            )

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Research Samples",
                    len(bp_dataset),
                )

            with c2:
                st.metric(
                    "RMS SBP Range",
                    (
                        "N/A"
                        if bp_dataset.empty
                        else f"{bp_dataset['RMS SBP'].min():.0f}–"
                             f"{bp_dataset['RMS SBP'].max():.0f}"
                    ),
                )

            with c3:
                st.metric(
                    "RMS DBP Range",
                    (
                        "N/A"
                        if bp_dataset.empty
                        else f"{bp_dataset['RMS DBP'].min():.0f}–"
                             f"{bp_dataset['RMS DBP'].max():.0f}"
                    ),
                )

            if len(bp_dataset) < 20:

                st.warning(
                    "This is currently a small research dataset. "
                    "Correlation/model results will not be reliable "
                    "for BP accuracy assessment until substantially "
                    "more paired readings are collected."
                )

            with st.expander(
                "PPG Features + RMS BP Target Dataset",
                expanded=True,
            ):

                st.dataframe(
                    bp_dataset,
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "Download BP Research Dataset CSV",
                    bp_dataset.to_csv(index=False),
                    file_name="PPG_to_RMS_BP_research_dataset.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_bp_research_dataset",
                )

            correlations = st.session_state.get(
                "bp_correlation_table"
            )

            if (
                correlations is not None
                and not correlations.empty
            ):

                with st.expander(
                    "Exploratory PPG Feature ↔ RMS BP Correlation",
                    expanded=True,
                ):

                    st.dataframe(
                        correlations,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.caption(
                        "Pearson correlation is exploratory only. "
                        "It does not establish BP accuracy or clinical "
                        "validity. Correlations can be unstable with "
                        "small datasets."
                    )

        # SELECTED READING PDF
        # ====================================================

        st.markdown(
            "---"
        )

        st.markdown(
            "### Selected Reading Export"
        )

        st.caption(
            "Exports ONLY the reading currently selected in the table. "
            "It does not export the patient's other readings."
        )


        export_key = str(
            selected_timestamp
        )


        if st.button(
            "Prepare Selected Reading PDF",
            use_container_width=True,
            key=(
                "prepare_selected_reading_pdf_"
                + export_key
            ),
        ):

            with st.spinner(
                "Creating PDF for the selected reading..."
            ):

                selected_reading_df = (
                    pd.DataFrame(
                        [reading]
                    )
                )

                st.session_state[
                    "selected_reading_pdf"
                ] = (
                    create_current_patient_detailed_pdf(
                        selected_reading_df
                    )
                )

                st.session_state[
                    "selected_reading_pdf_name"
                ] = (
                    "VinCense_Selected_Reading_"
                    + str(
                        selected_timestamp
                    )
                    .replace(
                        ":",
                        "-",
                    )
                    .replace(
                        "/",
                        "-",
                    )
                    .replace(
                        " ",
                        "_",
                    )
                    + ".pdf"
                )


        if (
            "selected_reading_pdf"
            in st.session_state
        ):

            st.download_button(
                "Download Selected Reading PDF",
                st.session_state[
                    "selected_reading_pdf"
                ],
                file_name=st.session_state.get(
                    "selected_reading_pdf_name",
                    "VinCense_Selected_Reading.pdf",
                ),
                mime="application/pdf",
                use_container_width=True,
                key=(
                    "download_selected_reading_pdf_"
                    + export_key
                ),
            )
