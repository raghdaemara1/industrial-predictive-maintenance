from __future__ import annotations

from typing import Tuple

import pandas as pd


def time_split(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    val_end: str,
    test_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    train_mask = (df["ts"] >= pd.to_datetime(train_start, utc=True)) & (
        df["ts"] < pd.to_datetime(train_end, utc=True)
    )
    val_mask = (df["ts"] >= pd.to_datetime(train_end, utc=True)) & (
        df["ts"] < pd.to_datetime(val_end, utc=True)
    )
    test_mask = (df["ts"] >= pd.to_datetime(val_end, utc=True)) & (
        df["ts"] < pd.to_datetime(test_end, utc=True)
    )
    return df[train_mask], df[val_mask], df[test_mask]
