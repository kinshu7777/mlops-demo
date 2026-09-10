"""
control.py — the ONE file you use to run everything.

Why this file exists:
  Instead of remembering "run train.py, then drift_check.py, then
  uvicorn app:app...", you now type one command with a keyword after
  it. This is called a CLI (Command Line Interface) with
  "subcommands" — the same pattern real tools like `git` or `docker`
  use (e.g. `git commit`, `docker run`).

  It's also "future-proof" in a specific sense: if you later add a
  new step to this project (say, a data-validation check), you add
  ONE new function here and ONE new subcommand. Nothing else in the
  project has to change, and every command you already learned still
  works exactly the same.

Available commands:
  python3 control.py generate-data      Create practice data
  python3 control.py train              Train a new model version
  python3 control.py predict            Predict from the terminal
  python3 control.py check-drift        Check if data has drifted
  python3 control.py serve              Start the prediction API
  python3 control.py dashboard          Start the monitoring dashboard
  python3 control.py status             Show current model + drift state
  python3 control.py pipeline           Run check-drift, retrain only if needed
  python3 control.py config             Show current config.json settings

Run `python3 control.py --help` any time to see this list again.
Run `python3 control.py <command> --help` for options on one command.
"""

import argparse
import subprocess
import sys
import json
import glob
import os

from config import CONFIG
import train as train_module
import drift_check as drift_module


# ---------------------------------------------------------------------
# A tiny color helper - no extra library needed (colorama/rich would add
# a dependency just for this). Windows Terminal, VS Code, and modern
# PowerShell all support these ANSI codes natively.
# ---------------------------------------------------------------------
class Color:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


def ok(text):
    return f"{Color.GREEN}{text}{Color.END}"


def warn(text):
    return f"{Color.YELLOW}{text}{Color.END}"


def bad(text):
    return f"{Color.RED}{text}{Color.END}"


def heading(text):
    return f"{Color.BOLD}{Color.CYAN}{text}{Color.END}"


# ---------------------------------------------------------------------
# Each function below does ONE job. control.py's only responsibility
# is deciding, based on what the user typed, which function to call.
# This separation is what makes it easy to add features later without
# breaking existing ones.
# ---------------------------------------------------------------------

def cmd_generate_data(args):
    """Create fresh practice data."""
    subprocess.run([sys.executable, os.path.join("data", "make_data.py")], check=True)


def cmd_train(args):
    """Train a new model version and save it."""
    train_module.train_and_save()


def cmd_predict(args):
    """Make one prediction directly from the terminal, no API needed."""
    import joblib
    import pandas as pd

    model_files = glob.glob(os.path.join(CONFIG["models_dir_path"], "model_v*.joblib"))
    if not model_files:
        print("No trained model found. Run: python3 control.py train")
        sys.exit(1)

    latest = max(model_files, key=lambda f: int(f.split("_v")[-1].split(".")[0]))
    model = joblib.load(latest)

    row = {
        "distance_km": args.distance_km,
        "num_stops": args.num_stops,
        "is_rush_hour": args.is_rush_hour,
        "driver_experience": args.driver_experience,
    }
    features = pd.DataFrame([row], columns=CONFIG["features"])
    prediction = model.predict(features)[0]
    print(f"Predicted delivery time: {prediction:.1f} minutes")


def cmd_check_drift(args):
    """Run the drift check and print the results."""
    result = drift_module.check_drift()
    for col in CONFIG["features"]:
        drifted = result[col]["drifted"]
        status = bad("DRIFTED") if drifted else ok("stable")
        print(f"  {col:20s} p={result[col]['p_value']:.5f}  -> {status}")
    if result["any_drift"]:
        print(warn("\nDrift detected."))
    else:
        print(ok("\nNo drift detected."))
    return result["any_drift"]


def cmd_serve(args):
    """Start the prediction API (this takes over the terminal until you Ctrl+C)."""
    subprocess.run([
        sys.executable, "-m", "uvicorn", "app:app",
        "--host", CONFIG["api_host"],
        "--port", str(CONFIG["api_port"]),
        "--reload",
    ])


def cmd_dashboard(args):
    """Start the monitoring dashboard (this takes over the terminal until you Ctrl+C)."""
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "dashboard.py",
        "--server.port", str(CONFIG["dashboard_port"]),
    ])


def cmd_status(args):
    """Show a quick summary: which model version is active, and drift state."""
    print(heading("=== Project Status ===\n"))
    meta_files = sorted(glob.glob(os.path.join(CONFIG["models_dir_path"], "model_v*.json")))
    if not meta_files:
        print(bad("No trained models yet."))
    else:
        with open(meta_files[-1]) as f:
            latest = json.load(f)
        print(ok(f"Latest model: v{latest['version']}"))
        print(f"  Trained at:  {latest['trained_at']}")
        print(f"  Accuracy:    off by {latest['mae_minutes']} minutes on average")
        print(f"  Total versions trained: {len(meta_files)}")

    if os.path.exists(CONFIG["drift_report_path"]):
        with open(CONFIG["drift_report_path"]) as f:
            drift = json.load(f)
        state = bad("DRIFT DETECTED") if drift["any_drift"] else ok("stable")
        print(f"\nLast drift check: {state}")
    else:
        print(warn("\nNo drift check has been run yet."))


def cmd_pipeline(args):
    """The 'set it and forget it' command: check drift, retrain only if needed."""
    print(heading("Checking for drift..."))
    drifted = cmd_check_drift(args)
    if drifted:
        print(warn("\nDrift found -> training a new model version..."))
        cmd_train(args)
        print(ok("Pipeline complete: retrained."))
    else:
        print(ok("\nNo drift -> current model is still good. Nothing to do."))


def cmd_test(args):
    """Run the automated test suite."""
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])
    sys.exit(result.returncode)


def cmd_config(args):
    """Print the current settings from config.json."""
    printable = {k: v for k, v in CONFIG.items() if not k.endswith("_path")}
    print(json.dumps(printable, indent=2))


# ---------------------------------------------------------------------
# Argument parsing: this is what turns "python3 control.py train" into
# a call to cmd_train(). Adding a new command later means adding one
# subparser block here, following the same pattern.
# ---------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="control.py",
        description="Command center for the delivery-time-predictor project.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("generate-data", help="Create practice data").set_defaults(func=cmd_generate_data)
    subparsers.add_parser("train", help="Train a new model version").set_defaults(func=cmd_train)
    subparsers.add_parser("check-drift", help="Check if data has drifted").set_defaults(func=cmd_check_drift)
    subparsers.add_parser("serve", help="Start the prediction API").set_defaults(func=cmd_serve)
    subparsers.add_parser("dashboard", help="Start the monitoring dashboard").set_defaults(func=cmd_dashboard)
    subparsers.add_parser("status", help="Show model + drift status").set_defaults(func=cmd_status)
    subparsers.add_parser("pipeline", help="Check drift, retrain only if needed").set_defaults(func=cmd_pipeline)
    subparsers.add_parser("config", help="Show current config.json settings").set_defaults(func=cmd_config)
    subparsers.add_parser("test", help="Run the automated test suite").set_defaults(func=cmd_test)

    predict_parser = subparsers.add_parser("predict", help="Predict delivery time from the terminal")
    predict_parser.add_argument("--distance_km", type=float, required=True)
    predict_parser.add_argument("--num_stops", type=int, required=True)
    predict_parser.add_argument("--is_rush_hour", type=int, choices=[0, 1], required=True)
    predict_parser.add_argument("--driver_experience", type=float, required=True)
    predict_parser.set_defaults(func=cmd_predict)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
