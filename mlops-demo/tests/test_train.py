"""
Tests for train.py

Why tests matter for a portfolio project: anyone reviewing your repo
(a recruiter, a hiring manager, another engineer) can run `pytest`
and immediately see that the pipeline behaves correctly — without
reading a single line of your code. This is one of the fastest ways
to signal "this person builds things properly."

Run all tests with:  pytest
Run just this file:  pytest tests/test_train.py -v
"""

import os
import sys
import tempfile
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import train_and_save, next_version_number
from config import CONFIG


@pytest.fixture
def sample_data_path():
    """Create a small, fake CSV so tests don't depend on the real dataset."""
    df = pd.DataFrame({
        "distance_km": [1, 5, 10, 15, 2, 8, 12, 3, 9, 6] * 5,
        "num_stops": [0, 1, 2, 3, 0, 1, 2, 3, 1, 2] * 5,
        "is_rush_hour": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 5,
        "driver_experience": [1, 3, 5, 7, 2, 4, 6, 8, 3, 5] * 5,
        "delivery_minutes": [10, 25, 35, 50, 12, 28, 38, 15, 30, 22] * 5,
    })
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        path = f.name
    yield path
    os.remove(path)


def test_train_and_save_returns_metadata(sample_data_path):
    """Training should return metadata describing what was trained."""
    metadata = train_and_save(data_path=sample_data_path)
    assert "version" in metadata
    assert "mae_minutes" in metadata
    assert metadata["mae_minutes"] >= 0


def test_train_and_save_creates_model_file(sample_data_path):
    """A .joblib model file should actually be written to disk."""
    metadata = train_and_save(data_path=sample_data_path)
    model_path = os.path.join(CONFIG["models_dir_path"], f"model_v{metadata['version']}.joblib")
    assert os.path.exists(model_path)


def test_next_version_number_increments():
    """Each new training run should get a higher version number than the last."""
    v1 = next_version_number()
    open(os.path.join(CONFIG["models_dir_path"], f"model_v{v1}.joblib"), "a").close()
    v2 = next_version_number()
    assert v2 == v1 + 1
    os.remove(os.path.join(CONFIG["models_dir_path"], f"model_v{v1}.joblib"))
