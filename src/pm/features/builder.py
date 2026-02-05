from __future__ import annotations

from typing import Dict, List

import pandas as pd

from pm.features.frequency import add_fft_band_power
from pm.features.lags import add_diff_features, add_lag_features
from pm.features.rolling import add_last_seen_features, add_missingness_features, add_rolling_features
from pm.features.selection import sensor_columns
from pm.features.trends import add_trend_features


def _window_steps(windows_minutes: List[int], freq_minutes: int) -> List[int]:
    steps = []
    for w in windows_minutes:
        steps.append(max(1, int(round(w / freq_minutes))))
    return steps


def build_features(df: pd.DataFrame, cfg: Dict, freq_minutes: int) -> pd.DataFrame:
    df = df.sort_values(["asset_id", "ts"]).reset_index(drop=True)
    sensors = sensor_columns(df)
    windows_minutes = cfg.get("windows_minutes", [60])
    lags = cfg.get("lags", [1])
    trend_points = cfg.get("trend_points", 6)
    missing_cfg = cfg.get("missingness", {})

    window_steps = _window_steps(windows_minutes, freq_minutes)
    df = add_rolling_features(df, sensors, window_steps)
    df = add_lag_features(df, sensors, lags)
    df = add_diff_features(df, sensors)
    df = add_trend_features(df, sensors, max(2, trend_points))

    if missing_cfg.get("enabled", True):
        df = add_missingness_features(df, sensors, max(window_steps))
        df = add_last_seen_features(df, sensors)

    fft_cfg = cfg.get("fft", {})
    if fft_cfg.get("enabled", False):
        df = add_fft_band_power(df, sensors, max(window_steps), fft_cfg.get("bands", [0.1]))

    # Diff vs rolling mean for smallest window
    smallest = min(window_steps)
    for sensor in sensors:
        col = f"{sensor}_roll{smallest}_mean"
        if col in df.columns:
            df[f"{sensor}_diff_rollmean"] = df[sensor] - df[col]

    return df
