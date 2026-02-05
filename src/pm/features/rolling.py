from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd


def add_rolling_features(
    df: pd.DataFrame,
    sensors: List[str],
    window_steps: Iterable[int],
) -> pd.DataFrame:
    out = df.copy()
    for w in window_steps:
        rolled = df.groupby("asset_id")[sensors].rolling(window=w, min_periods=1)
        stats = {
            "mean": rolled.mean().reset_index(level=0, drop=True),
            "std": rolled.std().reset_index(level=0, drop=True),
            "min": rolled.min().reset_index(level=0, drop=True),
            "max": rolled.max().reset_index(level=0, drop=True),
            "median": rolled.median().reset_index(level=0, drop=True),
        }
        for stat, values in stats.items():
            for sensor in sensors:
                out[f"{sensor}_roll{w}_{stat}"] = values[sensor].values
    return out


def add_missingness_features(
    df: pd.DataFrame,
    sensors: List[str],
    window_steps: int,
) -> pd.DataFrame:
    out = df.copy()
    rolled = df.groupby("asset_id")[sensors].rolling(window=window_steps, min_periods=1)
    miss = rolled.apply(lambda x: x.isna().mean(), raw=False).reset_index(level=0, drop=True)
    for sensor in sensors:
        out[f"{sensor}_missing_rate"] = miss[sensor].values
    return out


def add_last_seen_features(df: pd.DataFrame, sensors: List[str]) -> pd.DataFrame:
    out = df.copy()
    for asset_id, group in df.groupby("asset_id"):
        idx = group.index.to_numpy()
        for sensor in sensors:
            values = group[sensor].to_numpy()
            is_valid = ~pd.isna(values)
            last_valid_idx = pd.Series(idx).where(is_valid).ffill().to_numpy()
            distance = idx - last_valid_idx
            out.loc[idx, f"{sensor}_last_seen"] = distance
    return out
