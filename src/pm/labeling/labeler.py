from __future__ import annotations

from datetime import timedelta
from typing import Tuple

import numpy as np
import pandas as pd


def _label_asset(
    asset_df: pd.DataFrame,
    events_df: pd.DataFrame,
    horizon_hours: int,
    drop_event_window: bool,
) -> pd.DataFrame:
    asset_df = asset_df.sort_values("ts").copy()
    events_df = events_df.sort_values("event_start").copy()

    starts = events_df["event_start"].to_numpy()
    ends = events_df["event_end"].to_numpy()
    types = events_df["failure_type"].to_numpy()

    labels = []
    keep_mask = []
    horizon = np.array(timedelta(hours=horizon_hours), dtype="timedelta64[ns]")

    for t in asset_df["ts"].to_numpy():
        if len(starts) == 0:
            labels.append("None")
            keep_mask.append(True)
            continue

        pos = np.searchsorted(starts, t, side="right") - 1
        in_event = pos >= 0 and t <= ends[pos]
        if in_event and drop_event_window:
            keep_mask.append(False)
            labels.append("None")
            continue

        next_pos = pos + 1 if in_event else np.searchsorted(starts, t, side="right")
        label = "None"
        if next_pos < len(starts):
            if starts[next_pos] <= t + horizon:
                label = str(types[next_pos])
        labels.append(label)
        keep_mask.append(True)

    asset_df["label"] = labels
    return asset_df.loc[keep_mask].reset_index(drop=True)


def create_labels(
    sensor_wide: pd.DataFrame,
    failure_events: pd.DataFrame,
    horizon_hours: int = 24,
    drop_event_window: bool = True,
) -> pd.DataFrame:
    rows = []
    for asset_id, group in sensor_wide.groupby("asset_id"):
        events = failure_events[failure_events["asset_id"] == asset_id]
        rows.append(_label_asset(group, events, horizon_hours, drop_event_window))
    return pd.concat(rows, ignore_index=True)


def get_label_stats(labeled_df: pd.DataFrame) -> Tuple[pd.Series, int]:
    counts = labeled_df["label"].value_counts()
    total = len(labeled_df)
    return counts, total
