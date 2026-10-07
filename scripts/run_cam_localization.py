"""Run Stage 9 CAM localization on the fixed test split."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.localization.pipeline import LocalizationConfig, run_localization


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate weakly supervised CAM localization."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Inclusive threshold on normalized CAM values (default: 0.5).",
    )
    parser.add_argument(
        "--disable-two-sigma-filter",
        action="store_true",
        help="Keep every DFS region rather than applying the documented area filter.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    run_localization(
        PROJECT_ROOT,
        LocalizationConfig(
            threshold=args.threshold,
            apply_two_sigma_filter=not args.disable_two_sigma_filter,
            batch_size=args.batch_size,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
