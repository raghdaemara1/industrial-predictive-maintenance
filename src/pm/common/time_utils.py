from __future__ import annotations

from datetime import datetime


def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def utc_now_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")
