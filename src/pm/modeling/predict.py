from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def predict_proba(
    model: XGBClassifier, df: pd.DataFrame, label_encoder: LabelEncoder
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = df.drop(columns=["asset_id", "ts", "label"], errors="ignore")
    proba = model.predict_proba(X)
    classes = label_encoder.classes_
    proba_df = pd.DataFrame(proba, columns=[f"proba_{c}" for c in classes])
    preds = model.predict(X)
    pred_labels = label_encoder.inverse_transform(preds)
    pred_df = pd.DataFrame({"pred_label": pred_labels})
    return pred_df, proba_df


def predict_with_meta(
    model: XGBClassifier, df: pd.DataFrame, label_encoder: LabelEncoder
) -> pd.DataFrame:
    pred_df, proba_df = predict_proba(model, df, label_encoder)
    out = df[["asset_id", "ts"]].reset_index(drop=True)
    out = pd.concat([out, pred_df, proba_df], axis=1)
    return out
