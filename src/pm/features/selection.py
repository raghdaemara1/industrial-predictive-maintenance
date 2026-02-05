from __future__ import annotations

from typing import List


def sensor_columns(df) -> List[str]:
    return [c for c in df.columns if c not in {"asset_id", "ts", "label"}]
