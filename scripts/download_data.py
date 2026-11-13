"""Acquire the Kaggle House Prices dataset into data/raw.

Usage:
    python scripts/download_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import MANUAL_INSTRUCTIONS, acquire_dataset  # noqa: E402


def main() -> int:
    try:
        status = acquire_dataset(verbose=True)
    except Exception as exc:  # network blocked, checksum mismatch, ...
        print(f"Automatic download failed: {exc}\n")
        print(MANUAL_INSTRUCTIONS)
        return 1
    print(json.dumps(status, indent=2))
    ok = all(status[f]["exists"] for f in ("train.csv", "test.csv"))
    if ok and not status["data_description.txt"]["exists"]:
        print("\nNote: data_description.txt is optional. The app uses a built-in "
              "copy of the feature descriptions when it is absent.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
