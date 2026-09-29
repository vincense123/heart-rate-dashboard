"""Selecting a reading from the clickable table."""


def get_selected_reading(df, filtered_df, table_row):
    """
    Return the reading behind row `table_row` of the displayed table.

    The table shows `filtered_df` in order (index reset for display),
    so table row N is `filtered_df.index[N]`: the ORIGINAL row id of
    the sheet. The reading is then fetched from `df` by that id. The
    Timestamp is never used to find a reading: two rows can share one.
    """
    row_id = filtered_df.index[table_row]

    return row_id, df.loc[row_id]
