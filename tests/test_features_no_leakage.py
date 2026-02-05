import pandas as pd

from pm.features.builder import build_features
from pm.labeling.leakage_checks import assert_no_event_window_rows


def test_lag_features_no_future():
    ts = pd.date_range("2020-01-01", periods=5, freq="H", tz="UTC")
    df = pd.DataFrame(
        {
            "asset_id": ["A1"] * len(ts),
            "ts": ts,
            "sensor_1": [1, 2, 3, 4, 5],
            "label": ["None"] * len(ts),
        }
    )
    cfg = {"windows_minutes": [60], "lags": [1], "trend_points": 2, "missingness": {"enabled": False}}
    out = build_features(df, cfg, freq_minutes=60)
    # lag1 at index 2 should be previous value (2)
    assert out.loc[2, "sensor_1_lag1"] == 2


def test_no_event_window_rows():
    ts = pd.date_range("2020-01-01", periods=6, freq="H", tz="UTC")
    labeled = pd.DataFrame(
        {
            "asset_id": ["A1"] * len(ts),
            "ts": ts,
            "label": ["None"] * len(ts),
        }
    )
    failure_events = pd.DataFrame(
        {
            "asset_id": ["A1"],
            "event_start": [ts[3]],
            "event_end": [ts[4]],
            "failure_type": ["Power"],
        }
    )
    # remove event window
    labeled = labeled[labeled["ts"] < ts[3]]
    assert_no_event_window_rows(labeled, failure_events)
