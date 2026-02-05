import numpy as np
import pandas as pd

from pm.modeling.predict import predict_with_meta
from pm.modeling.train import train_xgb


def test_inference_contract_columns_and_probs():
    ts = pd.date_range("2020-01-01", periods=12, freq="H", tz="UTC")
    df = pd.DataFrame(
        {
            "asset_id": ["A1"] * len(ts),
            "ts": ts,
            "sensor_1": np.random.randn(len(ts)),
            "label": ["None"] * 6 + ["Overheat"] * 6,
        }
    )
    train_df = df.iloc[:8].copy()
    val_df = df.iloc[8:].copy()
    model, le, meta = train_xgb(
        train_df,
        val_df,
        {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1},
        use_class_weights=False,
    )
    features = df.drop(columns=["label"])
    pred = predict_with_meta(model, features, le)
    assert "asset_id" in pred.columns
    assert "ts" in pred.columns
    assert "pred_label" in pred.columns
    proba_cols = [c for c in pred.columns if c.startswith("proba_")]
    assert len(proba_cols) == len(le.classes_)
    assert np.allclose(pred[proba_cols].sum(axis=1), 1.0, atol=1e-3)
