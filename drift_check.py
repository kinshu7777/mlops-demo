"""
STEP 4: Check whether the world has changed since we trained.

Plain-English idea:
  When we trained the model, distances, rush-hour patterns, etc. looked
  a certain way. Months later, real life might look different (maybe
  the delivery zone expanded, or rush hour got worse). If the NEW data
  looks statistically different from the OLD data, the model's
  assumptions are stale. This is called "data drift".

How we check it:
  For each column, we run a Kolmogorov-Smirnov test (a standard
  statistics test that answers: "are these two piles of numbers likely
  drawn from the same underlying pattern, or not?"). A low p-value
  (below 0.05) means "probably not the same pattern anymore" -> drift.

  You don't need to memorize the statistics - just remember:
  low p-value = something changed = maybe retrain.

Run it with:  python3 drift_check.py
It prints a report AND exits with code 1 if drift is found (so a
CI/CD pipeline can automatically detect "drift happened" and react).
"""

import os
import sys
import json
import pandas as pd
from scipy.stats import ks_2samp

from config import CONFIG

FEATURES = CONFIG["features"]
REPORT_PATH = CONFIG["drift_report_path"]
P_VALUE_THRESHOLD = CONFIG["drift_p_value_threshold"]


def check_drift(old_path=None, new_path=None):
    old_path = old_path or CONFIG["old_data_path"]
    new_path = new_path or CONFIG["new_data_path"]
    old_df = pd.read_csv(old_path)
    new_df = pd.read_csv(new_path)

    report = {}
    drift_found = False

    for col in FEATURES:
        stat, p_value = ks_2samp(old_df[col], new_df[col])
        column_drifted = bool(p_value < P_VALUE_THRESHOLD)
        drift_found = drift_found or column_drifted
        report[col] = {
            "p_value": round(float(p_value), 5),
            "drifted": bool(column_drifted),
        }

    report["any_drift"] = drift_found

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    result = check_drift()

    print("Drift check results (p < 0.05 means the column has drifted):")
    for col in FEATURES:
        status = "DRIFTED" if result[col]["drifted"] else "stable"
        print(f"  {col:20s} p={result[col]['p_value']:.5f}  -> {status}")

    if result["any_drift"]:
        print("\nDrift detected - the model should be retrained.")
        sys.exit(1)
    else:
        print("\nNo significant drift - model is still trustworthy.")
        sys.exit(0)
