from __future__ import annotations

from pathlib import Path

from pm.common.io import load_json, save_json


def run(run_id: str) -> Path:
    metrics_path = Path("artifacts") / run_id / "metrics.json"
    metrics = load_json(metrics_path)
    report = {
        "summary": {
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "early_warning_rate": metrics.get("early_warning_rate"),
        }
    }
    out_path = Path("artifacts") / run_id / "reports" / "report.json"
    save_json(report, out_path)
    return out_path
