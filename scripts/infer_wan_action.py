"""Command-line entry point for AffordanceWAM infer."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.infer import main

if __name__ == "__main__":
    raise SystemExit(main())
