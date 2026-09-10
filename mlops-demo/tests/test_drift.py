"""
Tests for drift_check.py

Same idea as test_train.py: prove the drift detector actually detects
drift when it should, and stays quiet when it shouldn't — using small,
controlled fake data instead of trusting it by eye.
"""

import os
import sys
import tempfile
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drift_check import check_drift


def _write_csv(data):
    df = pd.DataFrame(data)
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    df.to_csv(f.name, index=False)
    f.close()
    return f.name


@pytest.fixture
def identical_datasets():
    """Two datasets that are basically the same distribution -> no drift expected."""
    base = {
        "distance_km": list(range(1, 51)),
        "num_stops": [i % 5 for i in range(50)],
        "is_rush_hour": [i % 2 for i in range(50)],
        "driver_experience": [i % 15 for i in range(50)],
    }
    old_path = _write_csv(base)
    new_path = _write_csv(base)
    yield old_path, new_path
    os.remove(old_path)
    os.remove(new_path)


@pytest.fixture
def clearly_different_datasets():
    """Two datasets with obviously different ranges -> drift expected."""
    old = {
        "distance_km": list(range(1, 51)),
        "num_stops": [i % 5 for i in range(50)],
        "is_rush_hour": [0] * 50,
        "driver_experience": [i % 15 for i in range(50)],
    }
    new = {
        "distance_km": list(range(500, 550)),  # totally different range
        "num_stops": [i % 5 for i in range(50)],
        "is_rush_hour": [1] * 50,               # totally different range
        "driver_experience": [i % 15 for i in range(50)],
    }
    old_path = _write_csv(old)
    new_path = _write_csv(new)
    yield old_path, new_path
    os.remove(old_path)
    os.remove(new_path)


def test_no_drift_on_identical_data(identical_datasets):
    old_path, new_path = identical_datasets
    result = check_drift(old_path=old_path, new_path=new_path)
    assert result["any_drift"] is False


def test_drift_detected_on_different_data(clearly_different_datasets):
    old_path, new_path = clearly_different_datasets
    result = check_drift(old_path=old_path, new_path=new_path)
    assert result["any_drift"] is True
    assert result["distance_km"]["drifted"] is True
    assert result["is_rush_hour"]["drifted"] is True
