from __future__ import annotations

import pandas as pd


def assert_no_event_window_rows(labeled_df: pd.DataFrame, failure_events: pd.DataFrame) -> None:
    merged = labeled_df.merge(
        failure_events,
        on="asset_id",
        how="left",
        suffixes=("", "_evt"),
    )
    in_window = (merged["ts"] >= merged["event_start"]) & (merged["ts"] <= merged["event_end"])
    if in_window.any():
        raise AssertionError("Found rows within failure event windows.")


def assert_time_sorted(df: pd.DataFrame) -> None:
    for asset_id, group in df.groupby("asset_id"):
        if not group["ts"].is_monotonic_increasing:
            raise AssertionError(f"Timestamps are not sorted for asset {asset_id}.")
