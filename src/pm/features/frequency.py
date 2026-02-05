from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def _band_power(values: np.ndarray, band: float) -> float:
    if len(values) < 4 or np.all(np.isnan(values)):
        return np.nan
    y = np.nan_to_num(values, nan=0.0)
    fft = np.fft.rfft(y)
    power = np.abs(fft) ** 2
    idx = min(int(band * len(power)), len(power) - 1)
    return float(power[idx])


def add_fft_band_power(
    df: pd.DataFrame,
    sensors: List[str],
    window_steps: int,
    bands: List[float],
) -> pd.DataFrame:
    out = df.copy()
    grouped = df.groupby("asset_id")
    for sensor in sensors:
        for band in bands:
            series = grouped[sensor].rolling(window=window_steps, min_periods=4).apply(
                lambda x: _band_power(x, band), raw=True
            )
            out[f"{sensor}_fft_{band}"] = series.reset_index(level=0, drop=True).values
    return out
