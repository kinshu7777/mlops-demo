"""
config.py — loads settings from config.json so nothing is hardcoded.

Why this file exists:
  Right now, "features", file paths, and thresholds were scattered
  across train.py, app.py, and drift_check.py as hardcoded values.
  If you wanted to add a new feature to the model (say,
  'weather_condition'), you'd have to edit three files and hope you
  didn't miss one.

  With this file, every script imports ONE shared config object.
  Change config.json, and every script picks up the change
  automatically. This is what "future-proofing" means in practice:
  when requirements change, you edit data, not code.
"""

import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")


def load_config():
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)

    # Turn relative paths into full paths, resolved from the project root,
    # so it doesn't matter which folder you run a script from.
    cfg["data_dir_path"] = os.path.join(HERE, cfg["data_dir"])
    cfg["models_dir_path"] = os.path.join(HERE, cfg["models_dir"])
    cfg["old_data_path"] = os.path.join(cfg["data_dir_path"], cfg["old_data_file"])
    cfg["new_data_path"] = os.path.join(cfg["data_dir_path"], cfg["new_data_file"])
    cfg["drift_report_path"] = os.path.join(cfg["models_dir_path"], "drift_report.json")

    return cfg


CONFIG = load_config()
