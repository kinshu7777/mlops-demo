"""
STEP 3: Turn the model into something other programs can use.

Right now the model just sits on your hard drive. This file wraps it in
a tiny web server so any app (a website, a mobile app, another service)
can send delivery details and get a time prediction back.

Run it with:   python3 control.py serve
Then visit:    http://127.0.0.1:8000/docs
That last link gives you a free, auto-generated page where you can
try the API by clicking buttons - no coding needed to test it.
"""

import os
import glob
import json
import joblib
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from config import CONFIG

MODELS_DIR = CONFIG["models_dir_path"]

# Grouping endpoints under named "tags" makes the docs page organize
# them into labeled sections instead of one flat list - this is what
# gives the page a more polished, professional structure.
tags_metadata = [
    {
        "name": "Predictions",
        "description": "Get a predicted delivery time from the trained model.",
    },
    {
        "name": "Monitoring",
        "description": "Check whether the service is alive and which model version it's serving.",
    },
]

app = FastAPI(
    title="Delivery Time Predictor API",
    description=(
        "Predicts delivery time in minutes from distance, number of stops, "
        "rush-hour status, and driver experience. Built as part of an "
        "end-to-end MLOps pipeline that also monitors for data drift and "
        "retrains automatically. See the project README for the full "
        "architecture."
    ),
    version="1.0.0",
    contact={"name": "Project README", "url": "https://github.com"},
    license_info={"name": "MIT"},
    openapi_tags=tags_metadata,
)


def _read_current_status():
    """Gather the same info the dashboard shows, for display on the homepage."""
    accuracy_text = "—"
    meta_files = sorted(glob.glob(os.path.join(MODELS_DIR, "model_v*.json")))
    if meta_files:
        with open(meta_files[-1]) as f:
            latest = json.load(f)
        accuracy_text = f"±{latest['mae_minutes']:.1f} min"

    drift_text, drift_color = "Unknown", "#888780"
    if os.path.exists(CONFIG["drift_report_path"]):
        with open(CONFIG["drift_report_path"]) as f:
            drift = json.load(f)
        if drift.get("any_drift"):
            drift_text, drift_color = "Drift detected", "#D85A30"
        else:
            drift_text, drift_color = "Stable", "#0F6E56"

    return accuracy_text, drift_text, drift_color


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
def root():
    """A real homepage instead of a bare redirect - shows live pipeline status."""
    accuracy_text, drift_text, drift_color = _read_current_status()
    dashboard_url = f"http://127.0.0.1:{CONFIG['dashboard_port']}"

    return f"""
    <html>
    <head>
        <title>Delivery Time Predictor</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: #F1EFE8;
                display: flex;
                justify-content: center;
                padding: 3rem 1rem;
                margin: 0;
            }}
            .card {{
                background: #FFFFFF;
                border-radius: 16px;
                padding: 2rem;
                max-width: 480px;
                width: 100%;
                box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            }}
            .header {{ text-align: center; margin-bottom: 1.5rem; }}
            .icon {{
                width: 52px; height: 52px; border-radius: 12px;
                background: #E6F1FB; display: flex; align-items: center;
                justify-content: center; margin: 0 auto 1rem; font-size: 26px;
            }}
            h1 {{ margin: 0 0 6px; font-size: 22px; font-weight: 600; color: #1a1a1a; }}
            .subtitle {{ color: #666; margin: 0; font-size: 14px; }}
            .stats {{
                display: grid; grid-template-columns: repeat(3, 1fr);
                gap: 12px; margin-bottom: 1.5rem;
            }}
            .stat {{
                background: #F7F6F2; border-radius: 10px;
                padding: 1rem 0.5rem; text-align: center;
            }}
            .stat-label {{ font-size: 12px; color: #777; margin: 0 0 4px; }}
            .stat-value {{ font-size: 20px; font-weight: 600; margin: 0; color: #1a1a1a; }}
            .actions {{ display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; }}
            .btn {{
                padding: 10px 18px; border-radius: 8px; border: 1px solid #ddd;
                background: #fff; text-decoration: none; color: #1a1a1a;
                font-size: 14px; font-weight: 500;
            }}
            .btn:hover {{ background: #f5f5f5; }}
        </style>
    </head>
    <body>
        <div class="card">
            <div class="header">
                <div class="icon">🚚</div>
                <h1>Delivery time predictor</h1>
                <p class="subtitle">End-to-end MLOps pipeline for delivery time prediction</p>
            </div>
            <div class="stats">
                <div class="stat">
                    <p class="stat-label">Model version</p>
                    <p class="stat-value">v{model_version}</p>
                </div>
                <div class="stat">
                    <p class="stat-label">Accuracy</p>
                    <p class="stat-value">{accuracy_text}</p>
                </div>
                <div class="stat">
                    <p class="stat-label">Status</p>
                    <p class="stat-value" style="color:{drift_color}">{drift_text}</p>
                </div>
            </div>
            <div class="actions">
                <a class="btn" href="/docs">📘 API docs</a>
                <a class="btn" href="{dashboard_url}">📊 Dashboard</a>
                <a class="btn" href="/health">💓 Health</a>
            </div>
        </div>
    </body>
    </html>
    """


def load_latest_model():
    """Find the highest-numbered model file and load it."""
    model_files = glob.glob(os.path.join(MODELS_DIR, "model_v*.joblib"))
    if not model_files:
        raise FileNotFoundError(
            "No trained model found. Run: python3 control.py train"
        )
    latest = max(model_files, key=lambda f: int(f.split("_v")[-1].split(".")[0]))
    version = int(latest.split("_v")[-1].split(".")[0])
    return joblib.load(latest), version


model, model_version = load_latest_model()


# This describes exactly what data the API expects to receive.
# FastAPI uses it to reject bad requests automatically, with a helpful
# error message, before your code even has to think about it.
# The json_schema_extra "example" is what pre-fills the docs page's
# "Try it out" form, so testers don't start from a blank slate.
class DeliveryRequest(BaseModel):
    distance_km: float = Field(..., gt=0, description="Distance in kilometers")
    num_stops: int = Field(..., ge=0, description="Number of stops before this one")
    is_rush_hour: int = Field(..., ge=0, le=1, description="1 if rush hour, else 0")
    driver_experience: float = Field(..., ge=0, description="Years of driving experience")

    model_config = {
        "json_schema_extra": {
            "example": {
                "distance_km": 5.0,
                "num_stops": 2,
                "is_rush_hour": 1,
                "driver_experience": 3.0,
            }
        }
    }


class PredictionResponse(BaseModel):
    predicted_minutes: float = Field(..., description="Predicted delivery time in minutes")
    model_version: int = Field(..., description="Which trained model version produced this prediction")


class HealthResponse(BaseModel):
    status: str
    model_version: int


@app.get(
    "/health",
    tags=["Monitoring"],
    summary="Check service health",
)
def health(request: Request):
    """
    Returns the service's health status.

    Other PROGRAMS calling this (monitoring tools, load balancers,
    control.py) get plain JSON - that's the actual contract this
    endpoint promises, and changing it would break anything relying
    on it. A human visiting it in a BROWSER instead sees a small
    styled page, detected by the browser's Accept header.
    """
    payload = {"status": "ok", "model_version": model_version}

    wants_html = "text/html" in request.headers.get("accept", "")
    if not wants_html:
        return JSONResponse(payload)

    return HTMLResponse(f"""
    <html>
    <head>
        <title>Health - Delivery Time Predictor</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: #F1EFE8;
                display: flex;
                justify-content: center;
                padding: 3rem 1rem;
                margin: 0;
            }}
            .card {{
                background: #FFFFFF;
                border-radius: 16px;
                padding: 2rem;
                max-width: 380px;
                width: 100%;
                text-align: center;
                box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            }}
            .dot {{
                width: 12px; height: 12px; border-radius: 50%;
                background: #0F6E56; display: inline-block; margin-right: 8px;
            }}
            h1 {{ margin: 0 0 6px; font-size: 20px; font-weight: 600; color: #1a1a1a; }}
            .row {{
                display: flex; justify-content: space-between;
                padding: 10px 0; border-top: 1px solid #eee; font-size: 14px;
            }}
            .label {{ color: #777; }}
            .value {{ font-weight: 600; color: #1a1a1a; }}
            .back {{
                display: inline-block; margin-top: 1.25rem; padding: 8px 16px;
                border-radius: 8px; border: 1px solid #ddd; text-decoration: none;
                color: #1a1a1a; font-size: 13px;
            }}
            .back:hover {{ background: #f5f5f5; }}
            .raw {{ margin-top: 1rem; font-size: 12px; color: #999; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1><span class="dot"></span>Service is healthy</h1>
            <div class="row"><span class="label">Status</span><span class="value">{payload['status']}</span></div>
            <div class="row"><span class="label">Model version</span><span class="value">v{payload['model_version']}</span></div>
            <a class="back" href="/">← Back to home</a>
            <p class="raw">Raw JSON: {json.dumps(payload)}</p>
        </div>
    </body>
    </html>
    """)


@app.post(
    "/predict",
    tags=["Predictions"],
    summary="Predict delivery time",
    response_model=PredictionResponse,
)
def predict(request: DeliveryRequest):
    """Send delivery details, get back a predicted number of minutes."""
    features = [[
        request.distance_km,
        request.num_stops,
        request.is_rush_hour,
        request.driver_experience,
    ]]
    prediction = model.predict(features)[0]
    return {
        "predicted_minutes": round(float(prediction), 1),
        "model_version": model_version,
    }