import streamlit as st


def apply_filters(df):
    """
    Applies the existing Patient ID, Patient Name and Date
    filters. Each filter remains a dropdown, while the resulting
    data is displayed in the clickable table in app.py.
    """

    st.subheader("Filters")

    col1, col2, col3 = st.columns(3)

    filtered_df = df.copy()

    # --------------------------------------------------------
    # PATIENT ID
    # --------------------------------------------------------

    with col1:
        if "PatientId" in df.columns:
            patient_ids = sorted(
                df["PatientId"]
                .dropna()
                .unique()
                .tolist(),
                key=lambda value: str(value),
            )

            selected_id = st.selectbox(
                "Select Patient ID",
                ["All"] + patient_ids,
            )

            if selected_id != "All":
                filtered_df = filtered_df[
                    filtered_df["PatientId"] == selected_id
                ]

    # --------------------------------------------------------
    # PATIENT NAME
    # --------------------------------------------------------

    with col2:
        if "PatientName" in filtered_df.columns:
            names = sorted(
                filtered_df["PatientName"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected_name = st.selectbox(
                "Select Patient Name",
                ["All"] + names,
            )

            if selected_name != "All":
                filtered_df = filtered_df[
                    filtered_df["PatientName"]
                    .astype(str)
                    == selected_name
                ]

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    with col3:
        if "Timestamp" in filtered_df.columns:
            dates = sorted(
                filtered_df["Timestamp"]
                .astype(str)
                .str[:10]
                .unique()
                .tolist()
            )

            selected_date = st.selectbox(
                "Select Date",
                ["All"] + dates,
            )

            if selected_date != "All":
                filtered_df = filtered_df[
                    filtered_df["Timestamp"]
                    .astype(str)
                    .str.startswith(selected_date)
                ]

    return filtered_df
