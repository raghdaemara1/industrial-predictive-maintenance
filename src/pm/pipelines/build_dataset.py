from __future__ import annotations

from pathlib import Path

from pm.common.config import load_config
from pm.common.io import write_parquet
from pm.data.loaders import load_events, load_sensor_long, load_sensor_wide, long_to_wide
from pm.data.synthetic import SyntheticConfig, generate_synthetic
from pm.features.builder import build_features
from pm.labeling.labeler import create_labels


def _raw_missing(raw_dir: Path) -> bool:
    required = [
        raw_dir / "sensor_readings_long.parquet",
        raw_dir / "sensor_readings_wide.parquet",
        raw_dir / "failure_events.parquet",
        raw_dir / "downtime_events.parquet",
    ]
    return not all(p.exists() for p in required)


def run(config_path: str) -> Path:
    cfg = load_config(config_path)
    raw_dir = Path(cfg["data"]["raw_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])

    if _raw_missing(raw_dir):
        generate_synthetic(raw_dir, SyntheticConfig(freq_minutes=cfg["data"]["frequency_minutes"]))

    sensor_long = load_sensor_long(cfg["data"]["sensor_long_path"])
    sensor_wide_path = Path(cfg["data"]["sensor_wide_path"])
    if sensor_wide_path.exists():
        sensor_wide = load_sensor_wide(sensor_wide_path)
    else:
        sensor_wide = long_to_wide(sensor_long)

    failure_events = load_events(cfg["data"]["failure_events_path"])
    labeled = create_labels(
        sensor_wide,
        failure_events,
        horizon_hours=cfg["labeling"]["horizon_hours"],
        drop_event_window=cfg["labeling"]["drop_event_window"],
    )

    feat_cfg = load_config(cfg["features"]["config_path"])
    features = build_features(labeled, feat_cfg, cfg["data"]["frequency_minutes"])
    out_path = processed_dir / "features.parquet"
    write_parquet(features, out_path)
    return out_path
