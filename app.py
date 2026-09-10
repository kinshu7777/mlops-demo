"""
STEP 3: Turn the model into something other programs can use.

Right now the model just sits on your hard drive. This file wraps it in
a tiny web server so any app (a website, a mobile app, another service)
can send delivery details and get a time prediction back.

Run it with:   python3 control.py serve
Then visit:    http://127.0.0.1:8000
You'll land on a homepage with a real form you can fill in and submit -
no coding or API knowledge needed. The /docs page (linked from there)
is for developers who want the raw programmatic API instead.
"""

import os
import glob
import json
import joblib
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
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


# ---------------------------------------------------------------------
# Shared page styling, so the homepage, the form result, and error
# pages all look like one consistent product instead of three
# unrelated screens bolted together.
# ---------------------------------------------------------------------
PAGE_STYLE = """
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: #F1EFE8;
        display: flex;
        justify-content: center;
        padding: 3rem 1rem;
        margin: 0;
    }
    .card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 2rem;
        max-width: 480px;
        width: 100%;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .header { text-align: center; margin-bottom: 1.5rem; }
    .icon {
        width: 52px; height: 52px; border-radius: 12px;
        background: #E6F1FB; display: flex; align-items: center;
        justify-content: center; margin: 0 auto 1rem; font-size: 26px;
    }
    h1 { margin: 0 0 6px; font-size: 22px; font-weight: 600; color: #1a1a1a; }
    .subtitle { color: #666; margin: 0; font-size: 14px; }
    .stats {
        display: grid; grid-template-columns: repeat(3, 1fr);
        gap: 12px; margin-bottom: 1.5rem;
    }
    .stat {
        background: #F7F6F2; border-radius: 10px;
        padding: 1rem 0.5rem; text-align: center;
    }
    .stat-label { font-size: 12px; color: #777; margin: 0 0 4px; }
    .stat-value { font-size: 20px; font-weight: 600; margin: 0; color: #1a1a1a; }
    .actions { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; }
    .btn {
        padding: 10px 18px; border-radius: 8px; border: 1px solid #ddd;
        background: #fff; text-decoration: none; color: #1a1a1a;
        font-size: 14px; font-weight: 500; cursor: pointer;
        font-family: inherit;
    }
    .btn:hover { background: #f5f5f5; }
    .btn-primary { background: #1a1a1a; color: #fff; border-color: #1a1a1a; }
    .btn-primary:hover { background: #333; }
    .divider { border-top: 1px solid #eee; margin: 1.5rem 0; }
    .field { margin-bottom: 1rem; }
    .field label {
        display: block; font-size: 13px; color: #555;
        margin-bottom: 4px; font-weight: 500;
    }
    .field input, .field select {
        width: 100%; padding: 9px 10px; border-radius: 8px;
        border: 1px solid #ddd; font-size: 14px; box-sizing: border-box;
        font-family: inherit;
    }
    .field input:focus, .field select:focus { outline: 2px solid #cfe2ff; border-color: #1a1a1a; }
    .field .hint { font-size: 12px; color: #999; margin-top: 3px; }
    .error-box {
        background: #FAECE7; border: 1px solid #F0997B; border-radius: 8px;
        padding: 10px 12px; margin-bottom: 1rem; font-size: 13px; color: #712B13;
    }
    .result-box {
        background: #EAF3DE; border-radius: 10px; padding: 1.25rem;
        text-align: center; margin-bottom: 1.5rem;
    }
    .result-value { font-size: 32px; font-weight: 600; color: #27500A; margin: 4px 0 0; }
    .result-label { font-size: 13px; color: #3B6D11; margin: 0; }
"""


def _page(title, body_html):
    """Wrap any page's inner HTML in the shared layout and styling."""
    return f"""
    <html>
    <head>
        <title>{title}</title>
        <style>{PAGE_STYLE}</style>
    </head>
    <body>{body_html}</body>
    </html>
    """


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


def _prediction_form_html(values=None, error=None):
    """
    The actual friendly form a non-technical person fills in.

    `values` lets us re-fill the form with whatever the person already
    typed if something went wrong, instead of wiping it and making
    them start over - a small thing that makes a form feel a lot less
    frustrating to use.
    """
    values = values or {"distance_km": "5", "num_stops": "2", "driver_experience": "3"}
    is_rush_hour = values.get("is_rush_hour", "0")

    error_html = f'<div class="error-box">{error}</div>' if error else ""

    return f"""
    <form method="post" action="/predict-form">
        {error_html}
        <div class="field">
            <label for="distance_km">Distance (km)</label>
            <input type="number" step="0.1" min="0.1" name="distance_km" id="distance_km"
                   value="{values.get('distance_km', '')}" required>
        </div>
        <div class="field">
            <label for="num_stops">Number of stops before this delivery</label>
            <input type="number" step="1" min="0" name="num_stops" id="num_stops"
                   value="{values.get('num_stops', '')}" required>
        </div>
        <div class="field">
            <label for="driver_experience">Driver experience (years)</label>
            <input type="number" step="0.1" min="0" name="driver_experience" id="driver_experience"
                   value="{values.get('driver_experience', '')}" required>
        </div>
        <div class="field">
            <label for="is_rush_hour">Is it rush hour?</label>
            <select name="is_rush_hour" id="is_rush_hour">
                <option value="0" {"selected" if is_rush_hour == "0" else ""}>No</option>
                <option value="1" {"selected" if is_rush_hour == "1" else ""}>Yes</option>
            </select>
        </div>
        <button type="submit" class="btn btn-primary" style="width: 100%;">Predict delivery time</button>
    </form>
    """


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
def root():
    """The homepage: live pipeline status plus a real, fillable prediction form."""
    accuracy_text, drift_text, drift_color = _read_current_status()
    dashboard_url = f"http://127.0.0.1:{CONFIG['dashboard_port']}"

    body = f"""
    <div class="card">
        <div class="header">
            <div class="icon">🚚</div>
            <h1>Delivery time predictor</h1>
            <p class="subtitle"></p>
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
        <div class="divider"></div>
        {_prediction_form_html()}
        <div class="divider"></div>
        <div class="actions">
            <a class="btn" href="/docs">📘 API docs</a>
            <a class="btn" href="{dashboard_url}">📊 Dashboard</a>
            <a class="btn" href="/health">💓 Health</a>
        </div>
    </div>
    """
    return _page("Delivery Time Predictor", body)


def load_latest_model():
    """Find the highest-numbered model file and load it."""
    model_files = glob.glob(os.path.join(MODELS_DIR, "model_v*.joblib"))
    if not model_files:
        raise FileNotFoundError(
            "No trained model found.\n\n"
            "  This usually just means the model hasn't been trained yet "
            "in this copy of the project.\n"
            "  Fix it by running these two commands, then trying again:\n\n"
            "      python3 control.py generate-data\n"
            "      python3 control.py train\n"
        )
    latest = max(model_files, key=lambda f: int(f.split("_v")[-1].split(".")[0]))
    version = int(latest.split("_v")[-1].split(".")[0])
    return joblib.load(latest), version


try:
    model, model_version = load_latest_model()
except FileNotFoundError as e:
    # A friendly message in the terminal instead of a raw Python
    # traceback, since this is the single most common first-run error
    # anyone hits with this project.
    print("\n" + "=" * 60)
    print("  COULD NOT START: " + str(e))
    print("=" * 60 + "\n")
    raise SystemExit(1)


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


# ---------------------------------------------------------------------
# Friendly error handling: by default, sending bad data to a FastAPI
# app returns a technical-looking JSON error (a raw Pydantic error
# list). That's fine for developers using /docs, but confusing for
# anyone else. This handler translates it into a plain sentence for
# browsers, while still giving developers the detailed version via
# the API.
# ---------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def friendly_validation_error(request: Request, exc: RequestValidationError):
    wants_html = "text/html" in request.headers.get("accept", "")
    if not wants_html:
        # Developers calling the API directly still get the full detail.
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    # A human filling in the homepage form gets a plain-English sentence
    # instead. We map the most common cause (a missing/blank field or a
    # negative number) to something a non-programmer would understand.
    first_error = exc.errors()[0] if exc.errors() else {}
    field = first_error.get("loc", ["field"])[-1]
    message = f"Please enter a valid value for '{field}' (it can't be blank or negative)."

    return HTMLResponse(
        _page("Something's not right - Delivery Time Predictor", f"""
        <div class="card">
            <div class="header">
                <div class="icon">⚠️</div>
                <h1>One of the fields needs fixing</h1>
                <p class="subtitle">{message}</p>
            </div>
            <a class="btn btn-primary" href="/" style="display:block; text-align:center;">← Back to the form</a>
        </div>
        """),
        status_code=422,
    )


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

    body = f"""
    <div class="card" style="text-align:center;">
        <h1><span style="display:inline-block;width:12px;height:12px;border-radius:50%;
            background:#0F6E56;margin-right:8px;"></span>Service is healthy</h1>
        <div style="display:flex;justify-content:space-between;padding:10px 0;border-top:1px solid #eee;font-size:14px;">
            <span style="color:#777;">Status</span><span style="font-weight:600;">{payload['status']}</span>
        </div>
        <div style="display:flex;justify-content:space-between;padding:10px 0;border-top:1px solid #eee;font-size:14px;">
            <span style="color:#777;">Model version</span><span style="font-weight:600;">v{payload['model_version']}</span>
        </div>
        <a class="btn" href="/" style="display:inline-block;margin-top:1.25rem;">← Back to home</a>
        <p style="margin-top:1rem;font-size:12px;color:#999;">Raw JSON: {json.dumps(payload)}</p>
    </div>
    """
    return HTMLResponse(_page("Health - Delivery Time Predictor", body))


@app.post(
    "/predict",
    tags=["Predictions"],
    summary="Predict delivery time",
    response_model=PredictionResponse,
)
def predict(request: DeliveryRequest):
    """Send delivery details, get back a predicted number of minutes. For programs/developers."""
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


@app.post("/predict-form", include_in_schema=False, response_class=HTMLResponse)
def predict_form(
    distance_km: float = Form(..., gt=0),
    num_stops: int = Form(..., ge=0),
    is_rush_hour: int = Form(..., ge=0, le=1),
    driver_experience: float = Form(..., ge=0),
):
    """
    The human-friendly version of /predict: takes a normal HTML form
    submission (not JSON) and shows the result as a nice page instead
    of raw text. This is what non-technical users actually interact
    with - /predict above stays there for other programs to call.
    """
    values = {
        "distance_km": distance_km,
        "num_stops": num_stops,
        "is_rush_hour": str(is_rush_hour),
        "driver_experience": driver_experience,
    }

    features = [[distance_km, num_stops, is_rush_hour, driver_experience]]
    prediction = model.predict(features)[0]

    accuracy_text, drift_text, drift_color = _read_current_status()

    body = f"""
    <div class="card">
        <div class="header">
            <div class="icon">🚚</div>
            <h1>Delivery time predictor</h1>
            <p class="subtitle">End-to-end MLOps pipeline for delivery time prediction</p>
        </div>
        <div class="result-box">
            <p class="result-label">Predicted delivery time</p>
            <p class="result-value">{prediction:.1f} minutes</p>
        </div>
        {_prediction_form_html(values)}
        <div class="divider"></div>
        <div class="actions">
            <a class="btn" href="/docs">📘 API docs</a>
            <a class="btn" href="/health">💓 Health</a>
        </div>
    </div>
    """
    return _page("Prediction result - Delivery Time Predictor", body)
