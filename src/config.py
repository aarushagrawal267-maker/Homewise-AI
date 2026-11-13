"""Project configuration and filesystem paths.

Everything is resolved relative to the project root so the code behaves the same
on Windows and Linux regardless of the working directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / "config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with open(CONFIG_FILE, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass(frozen=True)
class Paths:
    root: Path
    raw_data: Path
    processed_data: Path
    models: Path
    figures: Path
    reports: Path
    experiments: Path

    @property
    def train_csv(self) -> Path:
        return self.raw_data / load_config()["data"]["train_file"]

    @property
    def test_csv(self) -> Path:
        return self.raw_data / load_config()["data"]["test_file"]

    @property
    def description_txt(self) -> Path:
        return self.raw_data / load_config()["data"]["description_file"]

    @property
    def model_bundle(self) -> Path:
        return self.models / "homewise_model.joblib"

    @property
    def candidates_dir(self) -> Path:
        return self.models / "candidates"

    @property
    def lab_models_dir(self) -> Path:
        return self.models / "lab"

    @property
    def experiments_log(self) -> Path:
        return self.experiments / "experiments.jsonl"

    def ensure(self) -> None:
        for p in (self.raw_data, self.processed_data, self.models, self.figures,
                  self.reports, self.experiments, self.candidates_dir, self.lab_models_dir):
            p.mkdir(parents=True, exist_ok=True)


def get_paths() -> Paths:
    cfg = load_config()["paths"]
    return Paths(
        root=PROJECT_ROOT,
        raw_data=PROJECT_ROOT / cfg["raw_data"],
        processed_data=PROJECT_ROOT / cfg["processed_data"],
        models=PROJECT_ROOT / cfg["models"],
        figures=PROJECT_ROOT / cfg["figures"],
        reports=PROJECT_ROOT / cfg["reports"],
        experiments=PROJECT_ROOT / cfg["experiments"],
    )


CONFIG = load_config()
PATHS = get_paths()
TARGET: str = CONFIG["data"]["target"]
ID_COL: str = CONFIG["data"]["id_column"]
SEED: int = int(CONFIG["split"]["random_seed"])
KAGGLE_URL: str = CONFIG["data"]["kaggle_url"]
