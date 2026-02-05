from __future__ import annotations

from pathlib import Path
from typing import Dict

from pm.common.io import ensure_dir, load_joblib, load_json, save_joblib, save_json


def run_dir(base_dir: str | Path, run_id: str) -> Path:
    return ensure_dir(Path(base_dir) / run_id)


def save_run(
    base_dir: str | Path,
    run_id: str,
    model,
    label_encoder,
    feature_names,
    configs: Dict,
    metrics: Dict,
) -> Path:
    rdir = run_dir(base_dir, run_id)
    save_joblib(model, rdir / "model.joblib")
    save_joblib(label_encoder, rdir / "label_encoder.joblib")
    save_json({"features": feature_names}, rdir / "features.json")
    save_json(configs, rdir / "configs.json")
    save_json(metrics, rdir / "metrics.json")
    return rdir


def load_run(base_dir: str | Path, run_id: str) -> Dict:
    rdir = Path(base_dir) / run_id
    return {
        "model": load_joblib(rdir / "model.joblib"),
        "label_encoder": load_joblib(rdir / "label_encoder.joblib"),
        "features": load_json(rdir / "features.json")["features"],
        "configs": load_json(rdir / "configs.json"),
        "metrics": load_json(rdir / "metrics.json"),
    }
