from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
def _class_weights(y: np.ndarray) -> Dict[int, float]:
    unique, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    weights = {cls: total / (len(unique) * cnt) for cls, cnt in zip(unique, counts)}
    return weights
def prepare_xy(
    df: pd.DataFrame,
    le: Optional[LabelEncoder] = None,
) -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    """
    Split a labelled DataFrame into feature matrix X and integer label array y.
    Args:
        df: DataFrame containing asset_id, ts, label, and feature columns.
        le: An already-fitted LabelEncoder. When None (default), a new encoder
            is fitted on df["label"] — use this only for the training split.
            Pass the training encoder for val/test splits so that class-to-integer
            mappings are identical across all splits.
    Returns:
        (X, y, le) — feature DataFrame, integer labels, fitted encoder.
    """
    features = df.drop(columns=["asset_id", "ts", "label"])
    if le is None:
        # Training split: fit a new encoder on the training label set
        le = LabelEncoder()
        y = le.fit_transform(df["label"].astype(str))
    else:
        # Val / test split: reuse the training encoder — do NOT re-fit
        y = le.transform(df["label"].astype(str))
    return features, y, le
def train_xgb(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_cfg: Dict,
    use_class_weights: bool,
) -> Tuple[XGBClassifier, LabelEncoder, Dict]:
    # Fit encoder on train; reuse it for val so class integers are consistent
    X_train, y_train, le = prepare_xy(train_df)
    X_val, y_val, _ = prepare_xy(val_df, le=le)  # passes the train encoder
    sample_weight = None
    if use_class_weights:
        weights = _class_weights(y_train)
        sample_weight = np.array([weights[c] for c in y_train])
    model = XGBClassifier(
        objective=model_cfg.get("objective", "multi:softprob"),
        eval_metric=model_cfg.get("eval_metric", "mlogloss"),
        n_estimators=model_cfg.get("n_estimators", 200),
        max_depth=model_cfg.get("max_depth", 6),
        learning_rate=model_cfg.get("learning_rate", 0.1),
        subsample=model_cfg.get("subsample", 0.8),
        colsample_bytree=model_cfg.get("colsample_bytree", 0.8),
        min_child_weight=model_cfg.get("min_child_weight", 1),
        reg_alpha=model_cfg.get("reg_alpha", 0.0),
        reg_lambda=model_cfg.get("reg_lambda", 1.0),
        n_jobs=model_cfg.get("n_jobs", 4),
    )
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=model_cfg.get("early_stopping_rounds", 20),
        verbose=False,
    )
    return model, le, {"feature_names": list(X_train.columns)}


