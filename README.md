# Industrial Predictive Maintenance System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready machine learning pipeline for predicting equipment failures from multivariate time-series sensor data. Built with a focus on **leakage-free feature engineering**, **time-based validation**, and **reproducible ML workflows**.

**Author:** Raghda Emara
**GitHub:** [raghdaemara1](https://github.com/raghdaemara1)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Pipeline Components](#pipeline-components)
- [Testing](#testing)
- [Results & Metrics](#results--metrics)
- [Technical Highlights](#technical-highlights)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

This project demonstrates a complete end-to-end machine learning pipeline for **predictive maintenance** in industrial settings. The system predicts equipment failure types before they occur, enabling proactive maintenance scheduling and reducing unplanned downtime.

### Problem Statement

Industrial equipment failures lead to:
- Costly unplanned downtime
- Safety hazards
- Reduced production efficiency

This solution uses sensor data to predict failures **24 hours in advance**, allowing maintenance teams to act proactively.

### Solution Approach

- **Multi-class classification** using XGBoost
- **Time-series feature engineering** with rolling statistics, lags, and trends
- **Strict leakage prevention** ensuring realistic model evaluation
- **Production-ready architecture** with artifact management and batch scoring

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Leakage-Free Pipeline** | Features only use data available at prediction time |
| **Time-Based Splits** | Train/validation/test splits respect temporal ordering |
| **Comprehensive Feature Engineering** | Rolling stats, lag features, trend analysis, missingness tracking |
| **Synthetic Data Generator** | Included for testing and demonstration |
| **Artifact Management** | Versioned model runs with full reproducibility |
| **Extensive Test Suite** | Unit tests specifically for leakage prevention |
| **Production CLI** | Command-line interface for all pipeline stages |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  Raw Sensor Data      │    Failure Events     │    Downtime Events          │
│  (Long/Wide Format)   │    (asset, type, ts)  │    (asset, duration)        │
└───────────┬───────────┴──────────┬────────────┴──────────┬──────────────────┘
            │                      │                       │
            ▼                      ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA PROCESSING LAYER                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │   Cleaning  │───▶│   Labeling  │───▶│   Feature   │───▶│    Split    │   │
│  │ & Validation│    │  (Horizon)  │    │  Engineering│    │ (Time-based)│   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
└───────────────────────────────────────────┬─────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODELING LAYER                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │   XGBoost   │───▶│   Metrics   │───▶│   Artifact  │───▶│   Reports   │   │
│  │  Training   │    │  Evaluation │    │   Storage   │    │  Generation │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
└───────────────────────────────────────────┬─────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INFERENCE LAYER                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │               Batch Scoring Pipeline                                 │    │
│  │   New Data ──▶ Feature Engineering ──▶ Model Inference ──▶ Predictions│   │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
industrial-predictive-maintenance/
│
├── configs/                    # Configuration files
│   ├── base.yaml              # Data paths, splits, labeling params
│   ├── model_xgb.yaml         # XGBoost hyperparameters
│   └── features.yaml          # Feature engineering settings
│
├── src/pm/                    # Main source code
│   ├── common/                # Shared utilities
│   │   ├── config.py          # YAML configuration loader
│   │   ├── io.py              # I/O operations (parquet, JSON)
│   │   ├── logging.py         # Logging setup
│   │   └── time_utils.py      # Timestamp utilities
│   │
│   ├── data/                  # Data handling
│   │   ├── schema.py          # Column name constants
│   │   ├── loaders.py         # Data loading functions
│   │   ├── cleaning.py        # Data validation & cleaning
│   │   └── synthetic.py       # Synthetic data generator
│   │
│   ├── labeling/              # Label generation
│   │   ├── labeler.py         # Horizon-based labeling
│   │   └── leakage_checks.py  # Leakage prevention assertions
│   │
│   ├── features/              # Feature engineering
│   │   ├── builder.py         # Main feature orchestrator
│   │   ├── rolling.py         # Rolling statistics
│   │   ├── lags.py            # Lag & diff features
│   │   ├── trends.py          # Trend slope calculation
│   │   ├── frequency.py       # FFT features (optional)
│   │   └── selection.py       # Sensor column selection
│   │
│   ├── modeling/              # Model training & evaluation
│   │   ├── train.py           # XGBoost training
│   │   ├── predict.py         # Inference
│   │   ├── metrics.py         # Evaluation metrics
│   │   ├── split.py           # Time-based splitting
│   │   └── registry.py        # Artifact management
│   │
│   ├── pipelines/             # End-to-end pipelines
│   │   ├── build_dataset.py   # Data preparation pipeline
│   │   ├── train_model.py     # Training pipeline
│   │   ├── evaluate.py        # Evaluation pipeline
│   │   └── score_batch.py     # Batch inference pipeline
│   │
│   ├── reports/               # Report generation
│   │   └── generate_report.py
│   │
│   └── cli.py                 # Command-line interface
│
├── tests/                     # Unit tests
│   ├── test_labeling.py       # Labeling logic tests
│   ├── test_splits.py         # Time split validation
│   ├── test_features_no_leakage.py  # Leakage prevention tests
│   └── test_inference_contract.py   # Output format tests
│
├── data/                      # Data directories (gitignored)
│   ├── raw/                   # Raw input data
│   ├── interim/               # Intermediate processing
│   └── processed/             # Final features
│
├── artifacts/                 # Model runs (gitignored)
│   └── <run_id>/              # Per-run artifacts
│
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
├── LICENSE                    # MIT License
└── README.md                  # This file
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/raghdaemara1/industrial-predictive-maintenance.git
cd industrial-predictive-maintenance

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.\.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1. Build Dataset

Generates synthetic data (if raw data doesn't exist), applies labeling, and engineers features:

```bash
python -m pm.cli build-dataset --config configs/base.yaml
```

### 2. Train Model

Trains XGBoost model with time-based cross-validation:

```bash
python -m pm.cli train --config configs/base.yaml --model configs/model_xgb.yaml
```

### 3. Evaluate Model

Evaluates a trained model on the test set:

```bash
python -m pm.cli evaluate --run-id <run_id>
```

### 4. Batch Scoring

Score new data using a trained model:

```bash
python -m pm.cli score-batch \
    --run-id <run_id> \
    --input data/raw/sensor_readings_long.parquet \
    --output artifacts/predictions.parquet
```

---

## Configuration

### Base Configuration (`configs/base.yaml`)

```yaml
project_name: industrial_pm
run_dir: artifacts

data:
  raw: data/raw
  interim: data/interim
  processed: data/processed
  sensor_freq_minutes: 60

labeling:
  horizon_hours: 24  # Predict failures 24h in advance

splits:
  train_end: "2020-02-01"
  val_end: "2020-02-15"
  test_end: "2020-03-20"
```

### Model Configuration (`configs/model_xgb.yaml`)

```yaml
model:
  n_estimators: 400
  max_depth: 6
  learning_rate: 0.05
  objective: multi:softprob
  eval_metric: mlogloss
  use_class_weights: true
  early_stopping_rounds: 25
```

### Feature Configuration (`configs/features.yaml`)

```yaml
windows_minutes: [15, 60, 360]  # Rolling window sizes
lags: [1, 3, 12]                # Lag periods
trend_points: 8                 # Points for trend calculation
track_missingness: true         # Track missing data patterns
use_fft: false                  # FFT features (optional)
```

---

## Pipeline Components

### Data Ingestion

Supports two sensor data formats:

| Format | Columns | Use Case |
|--------|---------|----------|
| **Wide** | `asset_id`, `ts`, `sensor_1`...`sensor_n` | Pre-pivoted data |
| **Long** | `asset_id`, `ts`, `sensor_name`, `value` | Flexible schema |

### Labeling Strategy

- **Prediction Horizon:** 24 hours (configurable)
- **Label Assignment:** First failure type occurring within horizon
- **No Failure:** Labeled as `None`
- **Leakage Prevention:** Event-period rows excluded from training

### Feature Engineering

| Feature Type | Description | Example |
|-------------|-------------|---------|
| **Rolling Stats** | Mean, std, min, max, median over windows | `sensor_1_roll_mean_60m` |
| **Lag Features** | Previous values at fixed intervals | `sensor_1_lag_3` |
| **Diff Features** | Change from previous values | `sensor_1_diff_1` |
| **Trend Slopes** | Linear trend over recent points | `sensor_1_trend_slope` |
| **Missingness** | Missing value indicators & distance | `sensor_1_missing_flag` |

### Model Training

- **Algorithm:** XGBoost (multi-class classification)
- **Class Balancing:** Automatic class weight computation
- **Early Stopping:** Prevents overfitting
- **Artifact Logging:** All training artifacts versioned

---

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_features_no_leakage.py -v

# Run with coverage
pytest tests/ --cov=src/pm --cov-report=html
```

### Test Categories

| Test File | Purpose |
|-----------|---------|
| `test_labeling.py` | Validates labeling logic and horizon |
| `test_splits.py` | Ensures time-based splits don't overlap |
| `test_features_no_leakage.py` | Verifies no future data leakage |
| `test_inference_contract.py` | Validates prediction output format |

---

## Results & Metrics

The pipeline evaluates models using:

- **Accuracy:** Overall prediction accuracy
- **Macro F1-Score:** Balanced metric for multi-class
- **Per-Class Metrics:** Precision, recall, F1 per failure type
- **Confusion Matrix:** Detailed classification breakdown
- **Early Warning Rate:** Percentage of failures predicted in advance

Example metrics output:

```
==================== Model Evaluation Report ====================
Run ID: 20240115_143022
Test Set Size: 1,234 samples

Overall Metrics:
  - Accuracy: 0.847
  - Macro F1: 0.812
  - Early Warning Rate: 0.923

Per-Class Performance:
  - None:        P=0.91  R=0.89  F1=0.90
  - Bearing:     P=0.78  R=0.82  F1=0.80
  - Electrical:  P=0.71  R=0.74  F1=0.72
  - Overheating: P=0.85  R=0.79  F1=0.82
================================================================
```

---

## Technical Highlights

### Leakage Prevention

This project implements rigorous leakage prevention:

1. **Temporal Feature Constraints:** Rolling windows only look backward
2. **Event Window Exclusion:** Data during failure events excluded
3. **Time-Based Splits:** No random shuffling that could leak future info
4. **Automated Testing:** Unit tests verify leakage-free properties

### Production-Ready Design

- **Configuration-Driven:** All parameters in YAML files
- **Artifact Versioning:** Every run creates timestamped artifacts
- **CLI Interface:** Easy integration with job schedulers
- **Modular Architecture:** Components can be used independently

### Reproducibility

- **Fixed Random Seeds:** Synthetic data is deterministic
- **Config Logging:** All configs saved with model artifacts
- **Dependency Pinning:** Exact versions in requirements.txt

---

## Future Improvements

- [ ] Add SHAP explainability for model interpretability
- [ ] Implement hyperparameter tuning with Optuna
- [ ] Add FastAPI endpoint for real-time inference
- [ ] Support additional models (LightGBM, CatBoost)
- [ ] Add data drift monitoring
- [ ] Docker containerization
- [ ] MLflow integration for experiment tracking

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

**Raghda Emara**

- GitHub: [@raghdaemara1](https://github.com/raghdaemara1)

---

*Built with passion for industrial AI and predictive analytics*
