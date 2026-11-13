"""Dataset acquisition and loading.

Acquisition order (see ``scripts/download_data.py``):

1. Files already present in ``data/raw`` (e.g. downloaded manually from Kaggle).
2. The Kaggle API (``kaggle competitions download``) when credentials exist.
3. A public mirror of the identical competition files hosted by the
   *Dive into Deep Learning* textbook (d2l.ai). Downloads are accepted only when
   their MD5 checksums match the official competition files.

The raw CSVs are never modified; all cleaning happens on in-memory copies.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .config import ID_COL, KAGGLE_URL, PATHS, TARGET

# Reference MD5 checksums of the competition files (sizes 460,676 and 451,405
# bytes, matching the files listed on the Kaggle competition data page).
EXPECTED_MD5 = {
    "train.csv": "80ccab65fb115cbad143dbbd2bcd5577",
    "test.csv": "dcec4b79bf9c7317bd9e17789bf888f0",
}

MIRROR_URLS = {
    "train.csv": "https://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_train.csv",
    "test.csv": "https://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_test.csv",
}


class DatasetMissingError(FileNotFoundError):
    """Raised when the Kaggle CSV files are not available."""


MANUAL_INSTRUCTIONS = f"""
The Kaggle House Prices dataset was not found in {PATHS.raw_data}.

Option A (automatic):  python scripts/download_data.py

Option B (manual):
  1. Open {KAGGLE_URL}
  2. Sign in, accept the competition rules and click "Download All".
  3. Unzip and copy train.csv, test.csv and data_description.txt into:
       {PATHS.raw_data}
""".strip()


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_status() -> dict[str, dict]:
    """Report which raw files exist and whether they match the official checksums."""
    status = {}
    for name in ("train.csv", "test.csv", "data_description.txt"):
        p = PATHS.raw_data / name
        entry = {"path": str(p), "exists": p.exists()}
        if p.exists():
            entry["size_bytes"] = p.stat().st_size
            if name in EXPECTED_MD5:
                entry["md5"] = md5sum(p)
                entry["checksum_ok"] = entry["md5"] == EXPECTED_MD5[name]
        status[name] = entry
    return status


def datasets_available() -> bool:
    return PATHS.train_csv.exists() and PATHS.test_csv.exists()


def _download_via_kaggle(dest: Path) -> bool:
    if shutil.which("kaggle") is None:
        return False
    try:
        subprocess.run(
            ["kaggle", "competitions", "download", "-c",
             "house-prices-advanced-regression-techniques", "-p", str(dest)],
            check=True, capture_output=True, timeout=300,
        )
    except Exception:
        return False
    archive = dest / "house-prices-advanced-regression-techniques.zip"
    if not archive.exists():
        return False
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    archive.unlink()
    return (dest / "train.csv").exists()


def _download_via_mirror(dest: Path) -> list[str]:
    fetched = []
    for name, url in MIRROR_URLS.items():
        target = dest / name
        if target.exists():
            continue
        tmp = dest / f".{name}.part"
        with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
        digest = md5sum(tmp)
        if digest != EXPECTED_MD5[name]:
            tmp.unlink()
            raise ValueError(
                f"Checksum mismatch for mirrored {name}: got {digest}, "
                f"expected {EXPECTED_MD5[name]}. File rejected."
            )
        tmp.replace(target)
        fetched.append(name)
    return fetched


def acquire_dataset(verbose: bool = True) -> dict[str, dict]:
    """Make sure train.csv and test.csv exist in data/raw, downloading if needed."""
    PATHS.ensure()
    log = print if verbose else (lambda *a, **k: None)
    if datasets_available():
        log("Dataset already present in", PATHS.raw_data)
    elif _download_via_kaggle(PATHS.raw_data):
        log("Downloaded via the Kaggle API.")
    else:
        log("Kaggle API unavailable; downloading the checksum-verified d2l.ai mirror...")
        fetched = _download_via_mirror(PATHS.raw_data)
        log("Fetched:", ", ".join(fetched) or "nothing (already present)")
    return dataset_status()


def _require(path: Path) -> None:
    if not path.exists():
        raise DatasetMissingError(MANUAL_INSTRUCTIONS)


def load_train(path: Path | None = None) -> pd.DataFrame:
    """Load the labelled competition training data (returns a fresh copy)."""
    path = path or PATHS.train_csv
    _require(path)
    return pd.read_csv(path, keep_default_na=True)


def load_test(path: Path | None = None) -> pd.DataFrame:
    """Load the unlabelled competition test data (no SalePrice)."""
    path = path or PATHS.test_csv
    _require(path)
    return pd.read_csv(path, keep_default_na=True)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate raw features from the SalePrice target (drops the Id column)."""
    y = df[TARGET].astype(float)
    X = df.drop(columns=[TARGET, ID_COL], errors="ignore")
    return X, y
