"""
LifeGuardian AI — Prediction Module
=====================================
Loads the saved pipeline and exposes a single predict() function.
The model is loaded once at import time; invalid inputs and missing
model files are handled gracefully.

Uses pure-Python/NumPy pipeline (no scipy dependency).

DISCLAIMER: Predictions are illustrative only — trained on synthetic data.
"""

import json
import os
import pickle
import sys
from typing import Optional

import numpy as np

# ── paths ──────────────────────────────────────────────────────────────
_ML_DIR        = os.path.dirname(os.path.abspath(__file__))
_ROOT          = os.path.join(_ML_DIR, "..", "..", "models")
_PIPELINE_FILE = os.path.join(_ROOT, "pipeline.pkl")
_METRICS_FILE  = os.path.join(_ROOT, "metrics.json")

# Ensure ml/ modules are importable when pickle loads the saved pipeline
if _ML_DIR not in sys.path:
    sys.path.insert(0, _ML_DIR)

# Pre-import pipeline classes so pickle can find them by module path
from pipeline import SimpleEncoder, ActivityPipeline  # noqa: F401 (needed for unpickling)
from pure_rf import RandomForestClassifier             # noqa: F401 (needed for unpickling)

VALID_STATES = {"walking", "sitting", "lying_down", "inactive"}
LABEL_ORDER  = ["low", "medium", "high"]

_pipeline    = None
_metrics     = None
_load_error: Optional[str] = None


def _load() -> None:
    global _pipeline, _metrics, _load_error
    if not os.path.exists(_PIPELINE_FILE):
        _load_error = (
            f"Model file not found: {_PIPELINE_FILE}. "
            "Run `python ml/train.py` from the backend directory first."
        )
        return
    try:
        with open(_PIPELINE_FILE, "rb") as f:
            _pipeline = pickle.load(f)
    except Exception as e:
        _load_error = f"Failed to load model: {e}"
        return

    if os.path.exists(_METRICS_FILE):
        try:
            with open(_METRICS_FILE, "r", encoding="utf-8") as f:
                _metrics = json.load(f)
        except Exception:
            _metrics = None

    _load_error = None


_load()


def is_model_ready() -> bool:
    return _pipeline is not None


def get_load_error() -> Optional[str]:
    return _load_error


def get_metrics() -> Optional[dict]:
    return _metrics


def predict(
    activity_state: str,
    inactivity_duration_minutes: float,
    movement_intensity: float,
    hour_of_day: int = 12,
) -> dict:
    """
    Run the saved ML pipeline on a single observation.

    Returns:
        prediction    : "low" | "medium" | "high"
        probabilities : {low: float, medium: float, high: float}
        model_ready   : bool
        error         : str | None
    """
    if not is_model_ready():
        return {
            "prediction":    "unknown",
            "probabilities": {},
            "model_ready":   False,
            "error":         _load_error or "Model not loaded.",
        }

    if activity_state not in VALID_STATES:
        return {
            "prediction":    "unknown",
            "probabilities": {},
            "model_ready":   True,
            "error": (
                f"Invalid activity_state '{activity_state}'. "
                f"Must be one of: {sorted(VALID_STATES)}"
            ),
        }

    inactivity_duration_minutes = float(max(0.0, min(120.0, inactivity_duration_minutes)))
    movement_intensity          = float(max(0.0, min(100.0, movement_intensity)))
    hour_of_day                 = int(max(0, min(23, hour_of_day)))

    X = np.array([[activity_state,
                   inactivity_duration_minutes,
                   movement_intensity,
                   hour_of_day]], dtype=object)

    try:
        label     = _pipeline.predict(X)[0]
        proba_raw = _pipeline.predict_proba(X)[0]
        classes   = _pipeline.model.classes_.tolist()
        proba_map = {cls: float(round(p, 4)) for cls, p in zip(classes, proba_raw)}
        for lbl in LABEL_ORDER:
            proba_map.setdefault(lbl, 0.0)
    except Exception as e:
        return {
            "prediction":    "unknown",
            "probabilities": {},
            "model_ready":   True,
            "error":         f"Prediction error: {e}",
        }

    return {
        "prediction":    label,
        "probabilities": proba_map,
        "model_ready":   True,
        "error":         None,
    }
