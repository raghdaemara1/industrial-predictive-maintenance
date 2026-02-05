import pandas as pd

from pm.labeling.labeler import create_labels


def test_labeling_horizon_and_drop_window():
    ts = pd.date_range("2020-01-01", periods=8, freq="H", tz="UTC")
    sensor_wide = pd.DataFrame(
        {
            "asset_id": ["A1"] * len(ts),
            "ts": ts,
            "sensor_1": range(len(ts)),
        }
    )
    failure_events = pd.DataFrame(
        {
            "asset_id": ["A1"],
            "event_start": [ts[5]],
            "event_end": [ts[6]],
            "failure_type": ["Overheat"],
        }
    )
    labeled = create_labels(sensor_wide, failure_events, horizon_hours=3, drop_event_window=True)
    # t=2 should see event at t=5 within horizon (2+3=5)
    t2_label = labeled[labeled["ts"] == ts[2]]["label"].iloc[0]
    assert t2_label == "Overheat"
    # t=1 should not
    t1_label = labeled[labeled["ts"] == ts[1]]["label"].iloc[0]
    assert t1_label == "None"
    # t=5 should be dropped
    assert (labeled["ts"] == ts[5]).sum() == 0
