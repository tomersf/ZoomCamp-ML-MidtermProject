"""FastAPI service that predicts whether a newly discovered wildfire will grow large (10+ acres).

Run:
    uv run uvicorn predict:app --port 8000
Docs:
    http://localhost:8000/docs
"""
import os
import pickle
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).resolve().parent / "models" / "wildfire_model.pkl"))

with open(MODEL_PATH, "rb") as f:
    artifact = pickle.load(f)
model, threshold, features = artifact["model"], artifact["threshold"], artifact["features"]

State = Literal[
    "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID", "IL", "IN", "KS",
    "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV",
    "NY", "OH", "OK", "OR", "PA", "PR", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY",
]
Cause = Literal[
    "Arson", "Campfire", "Children", "Debris Burning", "Equipment Use", "Fireworks", "Lightning",
    "Miscellaneous", "Missing/Undefined", "Powerline", "Railroad", "Smoking", "Structure",
]


class Fire(BaseModel):
    """What we know when a fire is first discovered."""
    discovery_date: date = Field(examples=["1993-02-06"])
    discovery_hour: int | None = Field(default=13, ge=0, le=23, description="Hour of day (0-23), if known")
    latitude: float = Field(ge=17, le=72, examples=[34.584500])
    longitude: float = Field(ge=-180, le=-60, examples=[-85.507100])
    state: State
    stat_cause_descr: Cause


class Prediction(BaseModel):
    probability_large: float
    is_large: bool
    threshold: float


STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Wildfire severity predictor")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    """Simple web UI: pick a state on the map, fill the form, call /predict."""
    return FileResponse(STATIC_DIR / "index.html")


def to_features(fire: Fire) -> pd.DataFrame:
    """Turn the request into the exact feature row the model was trained on."""
    d = pd.Timestamp(fire.discovery_date)
    row = {
        "fire_year": d.year,
        "discovery_doy": d.day_of_year,
        "latitude": fire.latitude,
        "longitude": fire.longitude,
        "state": fire.state,
        "stat_cause_descr": fire.stat_cause_descr,
        "discovery_hour": float("nan") if fire.discovery_hour is None else fire.discovery_hour,
        "month": d.month,
        "day_of_week": d.day_of_week,
    }
    return pd.DataFrame([row])[features]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(fire: Fire):
    proba = float(model.predict_proba(to_features(fire))[0, 1])
    return Prediction(probability_large=proba, is_large=proba >= threshold, threshold=threshold)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
