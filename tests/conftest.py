"""Gemeinsame Fixtures: das Handbeispiel (4 Pakete, Kapazität 8) und kleine Zufallsinstanzen mit festem Zufalls-Seed."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def hb():
    """Handbeispiel: Gewichte 4, 3, 3, 5, Werte 6, 7, 9, 9, Kapazität 8. Handrechnung: Optimum 18 mit {3, 4}, eindeutig; Status (-1, -1, +1, +1)."""
    return np.array([6, 7, 9, 9]), np.array([4, 3, 3, 5]), 8
