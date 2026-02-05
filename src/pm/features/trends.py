from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def _slope(values: np.ndarray) -> float:
    if len(values) < 2 or np.all(np.isnan(values)):
        return np.nan
    x = np.arange(len(values))
    y = np.array(values, dtype=float)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return np.nan
    coef = np.polyfit(x[mask], y[mask], 1)
    return float(coef[0])


def add_trend_features(df: pd.DataFrame, sensors: List[str], window_steps: int) -> pd.DataFrame:
    out = df.copy()
    grouped = df.groupby("asset_id")
    for sensor in sensors:
        slope = grouped[sensor].rolling(window=window_steps, min_periods=2).apply(_slope, raw=True)
        out[f"{sensor}_trend_slope"] = slope.reset_index(level=0, drop=True).values
    return out
