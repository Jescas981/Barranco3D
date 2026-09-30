#!/usr/bin/env python3
"""Create platform crops and masks from video frames."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lima3d.region_editor import main

if __name__ == '__main__':
    main()
