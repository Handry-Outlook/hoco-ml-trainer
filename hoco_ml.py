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
FEATURE_SCHEMA_VERSION = 2   # 2: adds raw per-model fields in 6-hour valid-time blocks, by run age

# One sample every 7 px. This matches the hover-point grid get_geojson_from_matrix
# has always written, so legacy runs can be used for calibration too.
CELL_PX = 7
SMOOTH_KERNEL = (11, 11)

# Lightning-capable models by CODE NAME, fixed column order for the training matrix.
# Real model names never leave the private backend: cloud.py translates them to these
# codes before anything is uploaded, so the trainer, Firestore and the website only see codes.
LIGHTNING_MODELS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]


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


# Regions inside the lightning observation coverage (Met Office feed + archives:
# roughly 18W-8E, 48.5N-63N). Only these are used for training, and only these get
# AI HOCO runs. Germany, Poland, the Baltics etc. have no verifying lightning data, so
# training on them would teach the model that storms there never happen.
AI_REGIONS = {"England", "Scotland", "UK", "Wales", "Ireland and Northern Ireland"}

# Where the lightning feed actually detects strikes (measured from 1.6 M archived strikes, Mar 2025 – Nov 2026):
# a lon/lat box with a straight south-east edge running from about (-1.9 E, 48.5 N) to (8 E, 51.9 N).
# Paris, Champagne, eastern France and southern Belgium are OUTSIDE it. Cells outside are never used for
# training or verification, because "no strike recorded" there does not mean "no lightning".
LIGHTNING_COVERAGE = {"west": -18.0, "east": 8.0, "south": 48.5, "north": 63.2,
                      "edge_slope": 0.3491, "edge_offset": 49.1739}
COVERAGE_MARGIN_DEG = 0.2    # stay ~20 km inside the edge, so a cell's 20 km neighbourhood is fully observed

# What an AI HOCO percentage means: the chance of lightning WITHIN THIS DISTANCE of the point during the
# window ("storms near here"). Traditional HOCO's model-radius spread and the accuracy checker's 24 km
# blocks express the same idea. Training labels, climatology maps and verification all use it.
NEIGHBOURHOOD_KM = 20.0


def in_lightning_coverage(lon, lat, margin: float = COVERAGE_MARGIN_DEG):
    c = LIGHTNING_COVERAGE
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    return ((lon >= c["west"] + margin) & (lon <= c["east"] - margin) & (lat >= c["south"] + margin)
            & (lat <= c["north"] - margin) & (lat >= c["edge_slope"] * lon + c["edge_offset"] + margin))


def canonical_model(name: str) -> str:
    return str(name or "").strip().lower()


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


def pool_grid(field: np.ndarray) -> np.ndarray:
    """Average over each CELL_PX x CELL_PX cell (keeps sparse raw signals that point-sampling would miss)."""
    f = np.asarray(field, dtype=np.float32)
    h, w = f.shape
    gh, gw = grid_shape(h, w)
    pad = np.zeros((gh * CELL_PX, gw * CELL_PX), np.float32)
    pad[:h, :w] = f
    return pad.reshape(gh, CELL_PX, gw, CELL_PX).mean(axis=(1, 3))


def grid_shape(h: int, w: int) -> tuple[int, int]:
    return (math.ceil(h / CELL_PX), math.ceil(w / CELL_PX))


def cell_km(region: str, h: int, w: int) -> tuple[float, float]:
    """Approximate (km per sample cell in x, in y) for a region chart of h x w pixels."""
    b = region_bounds(region)
    lat0 = math.radians((b["top"] + b["bottom"]) / 2)
    kx = (b["right"] - b["left"]) * 111.32 * math.cos(lat0) / w * CELL_PX
    ky = (b["top"] - b["bottom"]) * 110.57 / h * CELL_PX
    return kx, ky


def neighbourhood_kernel(region: str, h: int, w: int, km: float = NEIGHBOURHOOD_KM) -> np.ndarray:
    """Elliptical uint8 kernel covering every cell whose centre is within `km` of the middle cell."""
    kx, ky = cell_km(region, h, w)
    rx, ry = int(math.ceil(km / kx)), int(math.ceil(km / ky))
    yy, xx = np.mgrid[-ry:ry + 1, -rx:rx + 1]
    return (((xx * kx) ** 2 + (yy * ky) ** 2) <= km ** 2).astype(np.uint8)


# Learned methods see each model's STRONGEST value within NEIGHBOURHOOD_KM (matching the 20 km question)
# and, as input "ens", Traditional HOCO's own forecast, so they correct Traditional instead of rebuilding it.
ENS_INPUT = "ens"
ENS_FLOOR = 1e-3


def ens_feature(x_prob):
    """Traditional's probability as a non-negative log-odds input (0 at or below 0.1%)."""
    p = np.clip(np.asarray(x_prob, dtype=np.float64), ENS_FLOOR, 1 - ENS_FLOOR)
    return (np.log(p / (1 - p)) - math.log(ENS_FLOOR / (1 - ENS_FLOOR))).astype(np.float32)


def pixel_neighbourhood_kernel(region: str, h: int, w: int, km: float = NEIGHBOURHOOD_KM) -> np.ndarray:
    kx, ky = cell_km(region, h, w)
    px_x, px_y = kx / CELL_PX, ky / CELL_PX
    rx, ry = int(math.ceil(km / px_x)), int(math.ceil(km / px_y))
    yy, xx = np.mgrid[-ry:ry + 1, -rx:rx + 1]
    return (((xx * px_x) ** 2 + (yy * px_y) ** 2) <= km ** 2).astype(np.uint8)


def pool_fields(fields: dict, region: str, h: int, w: int, km: float, grid: bool) -> dict:
    """Each field -> its maximum within `km`, computed on the training sample grid so live runs see exactly
    what training saw. grid=True: fields are already on that grid; False: full-size fields (pooled on the
    grid, then scaled back up to full size)."""
    if not km:
        return dict(fields)
    k = neighbourhood_kernel(region, h, w, km)
    out = {}
    for m, f in fields.items():
        if f is None:
            out[m] = None
            continue
        f = np.asarray(f, dtype=np.float32)
        if grid:
            out[m] = cv2.dilate(f, k)
        else:
            g = cv2.dilate(np.ascontiguousarray(sample_grid(f)), k)
            # each grid value covers the pixels nearest to its sample point (pixel 7*i -> cell i exactly)
            idx_r = np.minimum((np.arange(f.shape[0]) + CELL_PX // 2) // CELL_PX, g.shape[0] - 1)
            idx_c = np.minimum((np.arange(f.shape[1]) + CELL_PX // 2) // CELL_PX, g.shape[1] - 1)
            out[m] = g[np.ix_(idx_r, idx_c)]
    return out


def prepare_inputs(fields: dict, x_ens, params: dict, region: str, h: int, w: int, grid: bool = False) -> dict:
    """The exact inputs a learned 'weights' or 'context' model was trained on."""
    out = pool_fields(fields, region, h, w, float(params.get("inputPoolKm") or 0), grid)
    if ENS_INPUT in (params.get("weights") or {}) and x_ens is not None:
        out[ENS_INPUT] = ens_feature(x_ens)
    return out


def coverage_grid(region: str, h: int, w: int) -> np.ndarray:
    """Boolean (gh, gw) mask of sample-grid cells inside the lightning coverage."""
    b = region_bounds(region)
    gh, gw = grid_shape(h, w)
    lon = b["left"] + (np.arange(gw) * CELL_PX) / w * (b["right"] - b["left"])
    lat = b["top"] - (np.arange(gh) * CELL_PX) / h * (b["top"] - b["bottom"])
    LON, LAT = np.meshgrid(lon, lat)
    return in_lightning_coverage(LON, LAT)


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
def build_feature_npz(model_fields: dict, ensemble_raw_smoothed: np.ndarray, region: str, meta: dict,
                      raw_blocks: list | None = None) -> bytes:
    """
    model_fields: {model: raw thunderstorm-risk field (H, W)} — unweighted by user weights.
    ensemble_raw_smoothed: the HOCO matrix after the display smoothing, BEFORE any calibration.
    raw_blocks: [{model, rank, run, block, frames, field}] — UNSMOOTHED per-model signal summed over the
        frames valid in one 6-hour block (block = start, UTC 'YYYYMMDDHH'); rank 0 = newest model run.
        Lets training learn lead-time weights, run-age decay and per-model smoothing later.
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

    blocks_meta = []
    for i, b in enumerate(raw_blocks or []):
        model = canonical_model(b.get("model"))
        if model not in LIGHTNING_MODELS or b.get("field") is None:
            continue
        key = f"r{i}"
        arrays[key] = pool_grid(b["field"]).astype(np.float16)
        blocks_meta.append({"key": key, "model": model, "rank": int(b.get("rank", 0)), "run": str(b.get("run")),
                            "block": str(b.get("block")), "frames": int(b.get("frames", 0))})

    meta = dict(meta)
    meta.update({
        "raw_blocks": blocks_meta,
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
# ---------------------------------------------------------------------------
# AI HOCO safeguards (applied to every AI method, in training scores, previews and live runs alike)
# ---------------------------------------------------------------------------
#  1. No storm signal -> no AI risk: where Traditional is below 1% AND no model shows storms within 20 km,
#     AI may not go above Traditional. Stops the faint 5-10% "wash" the context/calibration terms put down
#     over quiet areas.
#  2. Boosts need agreement: AI can raise a spot above Traditional only as far as the models agree there
#     (share of the available models showing storms within 20 km). One model on its own -> no boost;
#     about 60% of models or more -> the full learned boost. Lowering is never limited.
#  3. Outside the lightning coverage (N France and beyond) AI has nothing to learn from or be checked
#     against, so it hands back to Traditional, fading over ~40 km so there is no edge line.
GUARD_SIGNAL_X = 0.01          # same "signal" test as the training rows
GUARD_SIGNAL_MODEL = 0.05
GUARD_AGREE_NONE = 0.25        # share of models agreeing at/below which AI cannot boost
GUARD_AGREE_FULL = 0.60        # share at/above which the full boost is allowed
GUARD_COVER_FADE_DEG = 0.4     # fade from AI to Traditional over this distance inside the coverage edge


def agreement(pooled_fields: dict, shape):
    """(share of available models with a storm signal, number agreeing) per cell; inputs are the 20 km-pooled
    fields. A model counts as available in a cell when its value there is finite (NaN = model missing)."""
    n_agree = np.zeros(shape, np.float32)
    n_have = np.zeros(shape, np.float32)
    for m, f in (pooled_fields or {}).items():
        if m == ENS_INPUT or f is None:
            continue
        f = np.asarray(f, dtype=np.float32)
        if f.shape != tuple(shape):
            continue
        ok = np.isfinite(f)
        n_have += ok
        n_agree += ok & (np.nan_to_num(f, nan=0.0) > GUARD_SIGNAL_MODEL)
    share = np.where(n_have > 0, n_agree / np.maximum(n_have, 1), 0.0).astype(np.float32)
    return share, n_agree


def coverage_weight(region: str, shape, grid_hw=None) -> np.ndarray:
    """1 well inside the lightning coverage, 0 outside, smooth in between. shape = output (h, w);
    grid_hw = full chart size when the output is on the sample grid."""
    b = region_bounds(region)
    h, w = shape
    lon = b["left"] + (np.arange(w) + 0.5) / w * (b["right"] - b["left"])
    lat = b["top"] - (np.arange(h) + 0.5) / h * (b["top"] - b["bottom"])
    LON, LAT = np.meshgrid(lon, lat)
    c = LIGHTNING_COVERAGE
    coslat = np.cos(np.radians(LAT))
    d = np.minimum.reduce([(LON - c["west"]) * coslat, (c["east"] - LON) * coslat, LAT - c["south"], c["north"] - LAT,
                           (LAT - (c["edge_slope"] * LON + c["edge_offset"])) / math.sqrt(1 + c["edge_slope"] ** 2)])
    t = np.clip((d - COVERAGE_MARGIN_DEG) / GUARD_COVER_FADE_DEG, 0, 1)
    return (t * t * (3 - 2 * t)).astype(np.float32)


def guard_ai(p_ai, x_ens, pooled_fields: dict | None, region: str | None = None, cover=True, recent=None):
    """Apply the three safeguards. p_ai, x_ens: probabilities (same shape). pooled_fields: model fields already
    pooled to their 20 km maximum, on the same grid. region=None skips the coverage hand-back (training rows
    are all inside the coverage). recent: observed recent-lightning strength 0..1 on the same grid (or None);
    real strikes count as a storm signal and as agreement, so AI may raise risk where storms are already going."""
    p = np.asarray(p_ai, dtype=np.float64)
    x = np.clip(np.asarray(x_ens, dtype=np.float64), 0, 1)
    share, n_agree = agreement(pooled_fields, p.shape)
    r = np.zeros(p.shape) if recent is None else np.clip(np.nan_to_num(np.asarray(recent, dtype=np.float64)), 0, 1)
    signal = (x >= GUARD_SIGNAL_X) | (n_agree > 0) | (r > 0)
    p = np.where(signal, p, np.minimum(p, x))                                       # 1
    a = np.clip((share - GUARD_AGREE_NONE) / (GUARD_AGREE_FULL - GUARD_AGREE_NONE), 0, 1)
    a = np.where(n_agree >= 2, a, 0.0)                                               # a lone model never boosts
    a = np.maximum(a, r)                                                             # ... unless storms are observed
    p = np.where(p > x, x + (p - x) * a, p)                                          # 2
    if cover and region:
        cw = coverage_weight(region, p.shape)
        p = cw * p + (1 - cw) * x                                                    # 3
    return p.astype(np.float32)


_AUTO_ISSUE_RE = re.compile(r"^AUTO-(\d{12,13})")


def issue_ms_from_id(job_id: str, fallback_dt: datetime | None = None):
    """When a run was issued (ms). AUTO ids carry it; otherwise the run's start time."""
    m = _AUTO_ISSUE_RE.match(job_id or "")
    if m:
        return float(m.group(1))
    if fallback_dt is None:
        return None
    if fallback_dt.tzinfo is None:
        fallback_dt = fallback_dt.replace(tzinfo=timezone.utc)
    return fallback_dt.timestamp() * 1000.0


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


def predict_method(entry: dict, x_ens: np.ndarray, model_fields: dict, region: str = "England",
                   src_hw=None, grid: bool = False) -> np.ndarray:
    """Probability from a published/candidate method. model_fields are SMOOTHED per-model fields.
    src_hw = (height, width) of the full-size chart (needed for the 20 km input pooling)."""
    params = entry["params"]
    if entry["kind"] == "weights":
        shape = np.shape(x_ens)
        h, w = src_hw or shape
        return apply_logistic(prepare_inputs(model_fields, x_ens, params, region, h, w, grid), params, shape)
    return apply_isotonic(x_ens, params)
