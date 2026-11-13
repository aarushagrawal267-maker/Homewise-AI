"""Train, tune, compare and persist all HomeWise AI models.

Usage:
    python scripts/train_all.py            # full run (about 10–20 min on a laptop)
    python scripts/train_all.py --quick    # smaller search budgets (about 5 min)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import MANUAL_INSTRUCTIONS, acquire_dataset, datasets_available  # noqa: E402
from src.train import run_training  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="reduce hyperparameter search budgets")
    ap.add_argument("--skip-figures", action="store_true", help="do not regenerate static EDA figures")
    args = ap.parse_args()
    if not datasets_available():
        try:
            acquire_dataset()
        except Exception as exc:
            print(f"Could not obtain the dataset automatically: {exc}\n\n{MANUAL_INSTRUCTIONS}")
            return 1
    run_training(quick=args.quick, skip_figures=args.skip_figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
