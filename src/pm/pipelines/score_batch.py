from __future__ import annotations

from pathlib import Path

import pandas as pd

from pm.common.config import load_config
from pm.common.io import read_parquet, write_parquet
from pm.data.loaders import long_to_wide
from pm.features.builder import build_features
from pm.modeling.predict import predict_with_meta
from pm.modeling.registry import load_run


def _load_input(path: str | Path) -> pd.DataFrame:
    df = read_parquet(path)
    if {"sensor_name", "value"}.issubset(df.columns):
        return long_to_wide(df)
    return df


def run(run_id: str, input_path: str, output_path: str) -> Path:
    run_obj = load_run("artifacts", run_id)
    base_cfg = run_obj["configs"]["base"]
    feat_cfg = load_config(base_cfg["features"]["config_path"])

    df = _load_input(input_path)
    features = build_features(df, feat_cfg, base_cfg["data"]["frequency_minutes"])

    feature_list = run_obj["features"]
    for col in feature_list:
        if col not in features.columns:
            features[col] = pd.NA
    features = features[["asset_id", "ts"] + feature_list]

    preds = predict_with_meta(run_obj["model"], features, run_obj["label_encoder"])
    write_parquet(preds, output_path)
    return Path(output_path)
