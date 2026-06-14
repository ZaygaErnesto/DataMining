"""
app.py — FastAPI web application for predictive maintenance classification.
==========================================================================

Endpoints
---------
GET  /              Render the dashboard web UI.
GET  /api/metrics   Fetch training/evaluation metrics.
POST /api/predict   Run inference on a single sensor reading dict.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# ------------------------------------------------------------------ #
# Setup and Configuration
# ------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Predictive Maintenance API",
    description="API for classifying machine failure types based on sensor telemetry.",
    version="1.0.0",
)

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "src" / "static"
TEMPLATES_DIR = PROJECT_ROOT / "src" / "templates"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "metrics"

# Create directories if missing
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Load model, metadata and label encoder
try:
    PIPELINE = joblib.load(MODELS_DIR / "best_model.joblib")
    with (MODELS_DIR / "best_model_metadata.json").open("r") as f:
        METADATA = json.load(f)
    FEATURE_SET = METADATA.get("feature_set", "raw")
    CLASS_NAMES = METADATA.get("class_names", [])
    
    LE_PATH = MODELS_DIR / "label_encoder.joblib"
    LABEL_ENCODER = joblib.load(LE_PATH) if LE_PATH.exists() else None
    logger.info("Successfully loaded ML model artifacts.")
except Exception as e:
    logger.error("Failed to load model artifacts: %s", e)
    PIPELINE = None
    METADATA = {}
    FEATURE_SET = "raw"
    CLASS_NAMES = []
    LABEL_ENCODER = None


# ------------------------------------------------------------------ #
# Pydantic Request Schema
# ------------------------------------------------------------------ #
class TelemetryRequest(BaseModel):
    machine_type: str = Field(..., alias="Type", description="Machine product type (L, M, H)")
    air_temp: float = Field(..., alias="Air temperature [K]", description="Air Temperature in Kelvin")
    process_temp: float = Field(..., alias="Process temperature [K]", description="Process Temperature in Kelvin")
    rotational_speed: float = Field(..., alias="Rotational speed [rpm]", description="Rotational Speed in RPM")
    torque: float = Field(..., alias="Torque [Nm]", description="Torque in Nm")
    tool_wear: float = Field(..., alias="Tool wear [min]", description="Tool Wear in minutes")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Type": "M",
                "Air temperature [K]": 298.1,
                "Process temperature [K]": 308.6,
                "Rotational speed [rpm]": 1551,
                "Torque [Nm]": 42.8,
                "Tool wear [min]": 0,
            }
        }


# ------------------------------------------------------------------ #
# Route Handlers
# ------------------------------------------------------------------ #

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the dashboard HTML interface."""
    return templates.TemplateResponse(request, "index.html", {"class_names": CLASS_NAMES})


@app.get("/api/metrics")
async def get_metrics():
    """Fetch training and quality gate metrics."""
    metrics_file = METRICS_DIR / "metrics.json"
    if not metrics_file.exists():
        return JSONResponse(
            status_code=404,
            content={"detail": "Metrics report not generated yet."}
        )

    with open(metrics_file, "r") as f:
        metrics_data = json.load(f)
    
    return metrics_data


@app.post("/api/predict")
async def predict_telemetry(payload: TelemetryRequest):
    """Perform a prediction on a single telemetry reading."""
    if PIPELINE is None:
        raise HTTPException(
            status_code=503,
            detail="Machine learning model is not loaded/available on server."
        )

    # Convert request payload to DataFrame matching expected structure
    input_dict = {
        "Type": payload.machine_type,
        "Air temperature [K]": payload.air_temp,
        "Process temperature [K]": payload.process_temp,
        "Rotational speed [rpm]": payload.rotational_speed,
        "Torque [Nm]": payload.torque,
        "Tool wear [min]": payload.tool_wear,
    }
    input_df = pd.DataFrame([input_dict])

    try:
        # Conditionally apply feature engineering
        if FEATURE_SET == "engineered":
            from src.feature_engineering import engineer_features
            input_df = engineer_features(input_df)

        # Run prediction
        pred_code = PIPELINE.predict(input_df)[0]
        
        # Decode target integer class code back to target label string
        if LABEL_ENCODER is not None:
            pred_label = LABEL_ENCODER.inverse_transform([pred_code])[0]
        else:
            pred_label = CLASS_NAMES[pred_code] if pred_code < len(CLASS_NAMES) else str(pred_code)

        return {
            "prediction": pred_label,
            "feature_set_used": FEATURE_SET,
        }
    except Exception as e:
        logger.error("Error during prediction pipeline execution: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="127.0.0.1", port=8000, reload=True)
