from __future__ import annotations

from typing import Iterable, List

import pandas as pd


def add_lag_features(df: pd.DataFrame, sensors: List[str], lags: Iterable[int]) -> pd.DataFrame:
    out = df.copy()
    grouped = df.groupby("asset_id")
    for lag in lags:
        shifted = grouped[sensors].shift(lag)
        for sensor in sensors:
            out[f"{sensor}_lag{lag}"] = shifted[sensor].values
    return out


def add_diff_features(df: pd.DataFrame, sensors: List[str]) -> pd.DataFrame:
    out = df.copy()
    for sensor in sensors:
        lag1 = out.get(f"{sensor}_lag1")
        if lag1 is not None:
            out[f"{sensor}_diff_lag1"] = out[sensor] - lag1
    return out
