#!/usr/bin/env python3
"""CLI auxiliar; la implementación está en lima3d."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lima3d.pipeline import main

if __name__ == "__main__":
    main()
