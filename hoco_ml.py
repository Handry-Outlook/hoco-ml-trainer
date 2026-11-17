"""
hoco_ml.py — shared machine-learning helpers for HOCO.

Used by:
  * cloud.py      (inference: logs per-model features, applies learned weights / calibration)
  * train_hoco.py (training: builds the dataset, fits weights + calibration, publishes them)

Design in one paragraph
-----------------------
Every lightning run produces one "field" per model (the same thunderstorm-risk
transform HOCO already uses, but for that model alone). Those fields are
smoothed and sampled on a coarse grid (one sample every CELL_PX pixels, roughly
9 km) and saved next to the run. Once real lightning has been observed, the
trainer learns P(lightning in cell) = sigmoid(b + sum_m w_m * field_m) with
non-negative weights — i.e. it learns the model weights from verification
instead of hand-tuning them. Until enough per-model data exists, a monotone
(isotonic) calibration maps the existing ensemble value to an honest
probability. Both are only switched on when they beat the current method on
held-out runs.
"""

from __future__ import annotations

import io
import json
import math
import re
from datetime import datetime, timezone

import cv2
import numpy as np

ML_CONFIG_COLLECTION = "system_stats"
ML_CONFIG_DOC = "hoco_ml"
FEATURE_PREFIX = "ml/features"
FEATURE_SCHEMA_VERSION = 1

# One sample every 7 px. This matches the hover-point grid get_geojson_from_matrix
# has always written, so legacy runs can be used for calibration too.
CELL_PX = 7
SMOOTH_KERNEL = (11, 11)

# Lightning-capable models, fixed column order for the training matrix.
LIGHTNING_MODELS = ["rpdid2", "swisseu", "ez", "swissnow", "ezswiss", "ukmo2km", "swissmrf", "hardmi"]

# The website historically used "harmi" for HARMONIE-DMI; the backend key is "hardmi".
MODEL_ALIASES = {"harmi": "hardmi", "harmonie": "hardmi"}

REGION_BOUNDS = {
    "England": {"top": 56.578, "bottom": 48.788, "left": -10.234, "right": 5.471},
    "Scotland": {"top": 60, "bottom": 55.9161, "left": -9.4482, "right": 0},
    "UK": {"top": 61.0, "bottom": 49.0, "left": -11.0, "right": 3.0},
    "Poland": {"top": 55.4243, "bottom": 47.9787, "left": 11.7142, "right": 27.1252},
    "Germany": {"top": 55.8567, "bottom": 46.9809, "left": 0.9558, "right": 16.9409},
    "Baltics": {"top": 63.4481, "bottom": 52.7695, "left": 11.6949, "right": 36.4801},
}

# Risk bands (percent). Polygons are drawn as non-overlapping bands so the
# map never stacks translucent layers on top of each other.
RISK_BANDS = [
    {"min": 5, "max": 10, "fill": "#9fd8c4", "label": "5-10%"},
    {"min": 10, "max": 15, "fill": "#66c2a4", "label": "10-15%"},
    {"min": 15, "max": 20, "fill": "#b3e05a", "label": "15-20%"},
    {"min": 20, "max": 30, "fill": "#fff200", "label": "20-30%"},
    {"min": 30, "max": 40, "fill": "#ffb327", "label": "30-40%"},
    {"min": 40, "max": 50, "fill": "#ff7f27", "label": "40-50%"},
    {"min": 50, "max": 60, "fill": "#ff4127", "label": "50-60%"},
    {"min": 60, "max": 70, "fill": "#e0182d", "label": "60-70%"},
    {"min": 70, "max": 80, "fill": "#b0186b", "label": "70-80%"},
    {"min": 80, "max": 90, "fill": "#8a2bbd", "label": "80-90%"},
    {"min": 90, "max": 101, "fill": "#4b1fd1", "label": "90%+"},
]


def canonical_model(name: str) -> str:
    key = str(name or "").strip().lower()
    return MODEL_ALIASES.get(key, key)


def region_bounds(region: str) -> dict:
    return REGION_BOUNDS.get(region, REGION_BOUNDS["England"])


# ---------------------------------------------------------------------------
# Lead-time bucket
# ---------------------------------------------------------------------------
_DAY_LABEL_RE = re.compile(r"Day\s*(\d)", re.IGNORECASE)


def run_bucket(job_id: str, start_dt: datetime | None = None, now: datetime | None = None) -> str:
    """day0 / day1 / day2, from the AUTO id label or from lead time."""
    match = _DAY_LABEL_RE.search(str(job_id or ""))
    if match:
        return f"day{min(2, int(match.group(1)))}"
    if start_dt is not None:
        now = now or datetime.utcnow()
        lead_h = (start_dt - now).total_seconds() / 3600.0
        if lead_h < 18:
            return "day0"
        if lead_h < 42:
            return "day1"
        return "day2"
    return "day0"


# ---------------------------------------------------------------------------
# Field helpers (shared by inference + feature logging, so they always match)
# ---------------------------------------------------------------------------
def smooth(field: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(np.asarray(field, dtype=np.float32), SMOOTH_KERNEL, 0)


def sample_grid(field: np.ndarray) -> np.ndarray:
    return np.asarray(field, dtype=np.float32)[::CELL_PX, ::CELL_PX]


def grid_shape(h: int, w: int) -> tuple[int, int]:
    return (math.ceil(h / CELL_PX), math.ceil(w / CELL_PX))


def lonlat_to_cell(lon, lat, region: str, h: int, w: int):
    """Vectorised lon/lat -> (row, col) on the sample grid, -1 when outside."""
    b = region_bounds(region)
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    x = (lon - b["left"]) / (b["right"] - b["left"]) * w
    y = (b["top"] - lat) / (b["top"] - b["bottom"]) * h
    gh, gw = grid_shape(h, w)
    col = np.rint(x / CELL_PX).astype(np.int64)
    row = np.rint(y / CELL_PX).astype(np.int64)
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h) & (row >= 0) & (row < gh) & (col >= 0) & (col < gw)
    row = np.where(inside, row, -1)
    col = np.where(inside, col, -1)
    return row, col


# ---------------------------------------------------------------------------
# Feature logging
# ---------------------------------------------------------------------------
def build_feature_npz(model_fields: dict, ensemble_raw_smoothed: np.ndarray, region: str, meta: dict) -> bytes:
    """
    model_fields: {model: raw thunderstorm-risk field (H, W)} — unweighted by user weights.
    ensemble_raw_smoothed: the HOCO matrix after the display smoothing, BEFORE any calibration.
    """
    h, w = ensemble_raw_smoothed.shape
    arrays = {
        "ensemble": sample_grid(ensemble_raw_smoothed).astype(np.float16),
    }
    present = []
    for model, field in model_fields.items():
        model = canonical_model(model)
        if model not in LIGHTNING_MODELS or field is None:
            continue
        arrays[f"m_{model}"] = sample_grid(smooth(field)).astype(np.float16)
        present.append(model)

    meta = dict(meta)
    meta.update({
        "schema": FEATURE_SCHEMA_VERSION,
        "region": region,
        "height": int(h),
        "width": int(w),
        "cell_px": CELL_PX,
        "models": present,
    })
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def read_feature_npz(data: bytes) -> tuple[dict, dict]:
    with np.load(io.BytesIO(data), allow_pickle=False) as npz:
        meta = json.loads(bytes(npz["meta"]).decode("utf-8"))
        arrays = {k: npz[k].astype(np.float32) for k in npz.files if k != "meta"}
    return meta, arrays


# ---------------------------------------------------------------------------
# Learned-model application
# ---------------------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def apply_isotonic(x_prob: np.ndarray, calibration: dict) -> np.ndarray:
    kx = np.asarray(calibration.get("x", []), dtype=np.float64)
    ky = np.asarray(calibration.get("y", []), dtype=np.float64)
    if kx.size < 2:
        return np.asarray(x_prob, dtype=np.float32)
    return np.interp(np.clip(x_prob, 0, 1), kx, ky).astype(np.float32)


def apply_logistic(smoothed_fields: dict, params: dict, shape) -> np.ndarray:
    z = np.full(shape, float(params.get("intercept", -4.0)), dtype=np.float32)
    for model, weight in (params.get("weights") or {}).items():
        field = smoothed_fields.get(model)
        if field is not None and weight:
            z += float(weight) * field
    return sigmoid(z).astype(np.float32)


def pick_bucket_params(ml_doc: dict | None, bucket: str, kind: str):
    """kind = 'weights' (stage 2) or 'calibration' (stage 1). Returns params or None."""
    if not ml_doc:
        return None
    section = ml_doc.get(kind) or {}
    for scope in (bucket, "pooled"):
        params = section.get(scope)
        if params and params.get("status") == "active":
            return params
    return None


def load_ml_doc(db) -> dict | None:
    try:
        snap = db.collection(ML_CONFIG_COLLECTION).document(ML_CONFIG_DOC).get()
        return snap.to_dict() if snap.exists else None
    except Exception as exc:  # never let ML break a forecast run
        print(f"[ML] Could not load learned model: {exc}")
        return None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# AI HOCO (published product)
# ---------------------------------------------------------------------------
# system_stats/hoco_ml  -> latest TRAINING output (candidate, readiness, history). Changes daily.
# system_stats/ai_hoco  -> the PUBLISHED AI HOCO model. Only changes when an admin publishes,
#                          or when the daily update finds a candidate that is no worse.
AI_DOC = "ai_hoco"
AI_RUN_PREFIX = "AI-"
AI_TRAIN_PREFIX = "AITRAIN-"


def load_ai_doc(db) -> dict | None:
    try:
        snap = db.collection(ML_CONFIG_COLLECTION).document(AI_DOC).get()
        return snap.to_dict() if snap.exists else None
    except Exception as exc:
        print(f"[AI] Could not load AI HOCO config: {exc}")
        return None


def ai_is_live(ai_doc: dict | None) -> bool:
    return bool(ai_doc) and ai_doc.get("status") == "published"


def pick_ai_method(ai_doc: dict | None, bucket: str):
    """Published method for a day bucket: {'kind': 'weights'|'calibration', 'params': {...}} or None."""
    if not ai_doc:
        return None
    model = ai_doc.get("model") or {}
    entry = model.get(bucket) or model.get("pooled")
    if not entry or entry.get("kind") not in ("weights", "calibration", "context"):
        return None
    return entry


def predict_method(entry: dict, x_ens: np.ndarray, model_fields: dict) -> np.ndarray:
    """Probability from a published/candidate method. model_fields are SMOOTHED per-model fields."""
    params = entry["params"]
    if entry["kind"] == "weights":
        shape = np.shape(x_ens)
        return apply_logistic(model_fields, params, shape)
    return apply_isotonic(x_ens, params)
