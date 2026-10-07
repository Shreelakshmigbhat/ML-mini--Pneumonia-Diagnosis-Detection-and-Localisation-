"""Command-line entry point for the Stage 1 metadata pipeline."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.metadata import main


if __name__ == "__main__":
    raise SystemExit(main())
