"""Autonomous Power System tests: rate, selected points or reconstruction."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.testing import main

if __name__ == '__main__':
    main('power')
