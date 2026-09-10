"""
STEP 2: Train a model and save it.

What this file does, in plain words:
  1. Load the delivery data
  2. Split it into a "practice" part and a "quiz" part (train/test split)
  3. Teach a model to predict delivery_minutes from the other columns
  4. Check how good the model's guesses are on the quiz part
  5. Save the model to disk with a version number, so we never lose
     an old model when we make a new one

Run it with:  python3 train.py
"""

import os
import json
import glob
import joblib
import pandas as pd
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from config import CONFIG

MODELS_DIR = CONFIG["models_dir_path"]
FEATURES = CONFIG["features"]
TARGET = CONFIG["target"]


def next_version_number():
    """Look in the models/ folder and figure out what version comes next."""
    existing = glob.glob(os.path.join(MODELS_DIR, "model_v*.joblib"))
    if not existing:
        return 1
    numbers = [int(f.split("_v")[-1].split(".")[0]) for f in existing]
    return max(numbers) + 1


def train_and_save(data_path=None):
    data_path = data_path or CONFIG["old_data_path"]
    df = pd.read_csv(data_path)
    X = df[FEATURES]
    y = df[TARGET]

    # Hold back 20% of the data to honestly test the model afterwards.
    # The model never sees this part while learning.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    # Mean Absolute Error = on average, how many minutes off were we?
    mae = mean_absolute_error(y_test, predictions)

    version = next_version_number()
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"model_v{version}.joblib")
    joblib.dump(model, model_path)

    metadata = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "mae_minutes": round(mae, 3),
        "training_rows": len(df),
        "features": FEATURES,
    }
    metadata_path = os.path.join(MODELS_DIR, f"model_v{version}.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Trained model v{version}")
    print(f"On average, predictions are off by {mae:.2f} minutes")
    print(f"Saved to {model_path}")
    return metadata


if __name__ == "__main__":
    train_and_save()
