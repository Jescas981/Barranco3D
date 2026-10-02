#!/usr/bin/env python3
"""Sample scene videos and export optical-flow distributions and block maps."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lima3d.flow_experiment import main

if __name__ == '__main__':
    main()
