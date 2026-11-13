"""Append-only experiment tracking (JSON Lines).

Records are never overwritten or deleted by the application: each experiment is
appended as one line in ``outputs/experiments/experiments.jsonl``.
"""

from __future__ import annotations

import json
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import PATHS
from .hyperparameter_tuning import to_jsonable


def new_experiment_id(prefix: str = "EXP") -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:6]}"


def log_experiment(record: dict, path: Path | None = None) -> dict:
    path = path or PATHS.experiments_log
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "experiment_id": record.get("experiment_id") or new_experiment_id(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        **{k: v for k, v in record.items() if k != "experiment_id"},
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(to_jsonable(rec)) + "\n")
    return rec


def load_experiments(path: Path | None = None) -> list[dict]:
    path = path or PATHS.experiments_log
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def experiments_frame(path: Path | None = None) -> pd.DataFrame:
    rows = load_experiments(path)
    if not rows:
        return pd.DataFrame()
    flat = []
    for r in rows:
        v = r.get("validation_metrics", {}) or {}
        flat.append({
            "Experiment ID": r.get("experiment_id"),
            "Date/time (UTC)": r.get("timestamp"),
            "Source": r.get("source", ""),
            "Model": r.get("model_label", r.get("model")),
            "Parameters": json.dumps(r.get("params", {}), sort_keys=True),
            "Split": r.get("split", ""),
            "CV RMSLE": r.get("cv_rmsle_mean"),
            "CV RMSLE std": r.get("cv_rmsle_std"),
            "Val MAE": v.get("MAE"),
            "Val RMSE": v.get("RMSE"),
            "Val R²": v.get("R2"),
            "Val RMSLE": v.get("RMSLE"),
            "Train time (s)": r.get("train_seconds"),
            "Feature eng.": r.get("feature_engineering"),
        })
    return pd.DataFrame(flat)
