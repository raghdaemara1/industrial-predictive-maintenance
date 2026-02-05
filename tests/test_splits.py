import pandas as pd

from pm.modeling.split import time_split


def test_time_split_order_and_non_overlap():
    ts = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
    df = pd.DataFrame({"asset_id": ["A1"] * len(ts), "ts": ts, "label": ["None"] * len(ts)})
    train, val, test = time_split(df, "2020-01-01", "2020-01-05", "2020-01-08", "2020-01-11")
    assert train["ts"].max() < val["ts"].min()
    assert val["ts"].max() < test["ts"].min()
