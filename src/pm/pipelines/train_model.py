from __future__ import annotations

from pathlib import Path

from pm.common.config import load_config
from pm.common.io import read_parquet
from pm.common.time_utils import utc_now_id
from pm.modeling.registry import save_run
from pm.modeling.split import time_split
from pm.modeling.train import train_xgb


def run(config_path: str, model_path: str) -> str:
    cfg = load_config(config_path)
    model_cfg = load_config(model_path)

    data_path = Path(cfg["data"]["processed_dir"]) / "features.parquet"
    df = read_parquet(data_path)

    train_df, val_df, _ = time_split(
        df,
        cfg["splits"]["train_start"],
        cfg["splits"]["train_end"],
        cfg["splits"]["val_end"],
        cfg["splits"]["test_end"],
    )

    model, le, meta = train_xgb(
        train_df,
        val_df,
        model_cfg["model"],
        model_cfg.get("imbalance", {}).get("use_class_weights", True),
    )

    run_id = utc_now_id()
    save_run(
        cfg["project"]["run_dir"],
        run_id,
        model,
        le,
        meta["feature_names"],
        {"base": cfg, "model": model_cfg},
        {"status": "trained"},
    )
    return run_id
