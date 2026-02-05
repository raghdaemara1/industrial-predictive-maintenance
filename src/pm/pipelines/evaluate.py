from __future__ import annotations

from pathlib import Path

import pandas as pd

from pm.common.io import read_parquet, save_json, write_parquet
from pm.modeling.metrics import compute_metrics, early_warning_rate
from pm.modeling.predict import predict_proba
from pm.modeling.registry import load_run, run_dir
from pm.modeling.split import time_split


def run(run_id: str) -> Path:
    cfg = load_run("artifacts", run_id)["configs"]["base"]
    data_path = Path(cfg["data"]["processed_dir"]) / "features.parquet"
    df = read_parquet(data_path)
    _, _, test_df = time_split(
        df,
        cfg["splits"]["train_start"],
        cfg["splits"]["train_end"],
        cfg["splits"]["val_end"],
        cfg["splits"]["test_end"],
    )

    run_obj = load_run("artifacts", run_id)
    model = run_obj["model"]
    le = run_obj["label_encoder"]

    pred_df, _ = predict_proba(model, test_df, le)
    y_true = le.transform(test_df["label"].astype(str))
    y_pred = le.transform(pred_df["pred_label"].astype(str))

    metrics = compute_metrics(y_true, y_pred, labels=le.transform(le.classes_))
    metrics["early_warning_rate"] = early_warning_rate(
        test_df["label"].astype(str), pred_df["pred_label"].astype(str)
    )

    rdir = run_dir(cfg["project"]["run_dir"], run_id)
    save_json(metrics, rdir / "metrics.json")

    cm = pd.DataFrame(metrics["confusion_matrix"], index=le.classes_, columns=le.classes_)
    write_parquet(cm.reset_index().rename(columns={"index": "true_label"}), rdir / "reports" / "confusion_matrix.parquet")
    return rdir
