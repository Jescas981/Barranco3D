#!/usr/bin/env python3
"""Etapa 2: SfM disperso por coalición en paralelo (CPU), según config.yaml."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lima3d.scheduler import main

if __name__ == '__main__':
    main({'sfm'}, __doc__)
