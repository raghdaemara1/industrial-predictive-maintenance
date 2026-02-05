# Industrial Predictive Maintenance (XGBoost + Time-Series Features)

Production-style reference project for predicting failure type (or `None`) from multivariate time-series sensor data using leakage-free feature engineering and time-based splits. Dataset schema and structure are inspired by https://github.com/kokikwbt/predictive-maintenance, with a synthetic generator so the pipeline runs end-to-end from scratch.

## Quickstart

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Build dataset (uses synthetic data if raw/ is empty)
python -m pm.cli build-dataset --config configs/base.yaml

# Train model
python -m pm.cli train --config configs/base.yaml --model configs/model_xgb.yaml

# Evaluate a run
python -m pm.cli evaluate --run-id <run_id>

# Score batch
python -m pm.cli score-batch --run-id <run_id> --input data/raw/sensor_readings_long.parquet --output artifacts/preds.parquet
```

## Dataset Schema (Inspired by kokikwbt/predictive-maintenance)

Supported inputs:
- **Wide sensor table**: `asset_id`, `ts`, `sensor_1..sensor_n`
- **Long sensor table**: `asset_id`, `ts`, `sensor_name`, `value`

Additional tables:
- `failure_events`: `asset_id`, `event_start`, `event_end`, `failure_type`
- `downtime_events`: `asset_id`, `event_start`, `event_end`, `downtime_minutes`

Conversion from long to wide is provided for modeling.

## Labeling Rule (Leakage-Safe)

Prediction horizon `H` (default 24h). For each `(asset_id, ts)`:
- label is the **first** failure event starting in `(t, t+H]`
- if no event in horizon, label = `None`

Leakage prevention:
- Features only use sensor values at or before `t`
- Event-period values are excluded from feature generation

## Time Splits

All splits are **time-based** (no random splits), simulating real deployment.

## Artifacts

Run artifacts are stored in `artifacts/<run_id>/` and include:
- model, encoders, feature list
- configs used
- metrics and reports (confusion matrix, per-class metrics)

## Notes

- Designed for CPU execution.
- Minimal dependencies.
- Synthetic generator runs when `data/raw/` is empty.
