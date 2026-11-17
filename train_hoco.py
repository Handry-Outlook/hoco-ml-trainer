"""
train_hoco.py — learns HOCO's model weights from verified lightning.

Run daily from GitHub Actions (see .github/workflows/train-hoco.yml).
Needs env FIREBASE_SERVICE_ACCOUNT (same secret cloud.py uses).

What it does
------------
1. Loads observed lightning strikes (same GitHub sources the accuracy verifier uses).
2. Turns every finished AUTO run into training rows on a ~9 km grid:
     label  = did lightning strike in that cell during the run's window?
     x_ens  = HOCO's existing ensemble probability for that cell
     m_*    = each model's own thunderstorm field (only for runs logged by the new cloud.py)
   Rows are cached in Storage (ml/training_rows.npz) so each run is processed once.
3. Stage 1 — isotonic calibration of x_ens -> honest probability (works on legacy runs too).
4. Stage 2 — non-negative logistic regression: P = sigmoid(b + sum w_m * m_m).
   These w_m ARE the learned model weights.
5. Each stage is only marked "active" if it beats the current method on the most
   recent 25% of runs (held out, never trained on). Results go to
   Firestore system_stats/hoco_ml, which cloud.py and the website read.

Usage:
    python train_hoco.py                 # normal daily run
    python train_hoco.py --dry-run       # train + report, do not publish
    python train_hoco.py --rebuild       # ignore cached rows and rebuild from scratch
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import requests
from scipy.optimize import minimize

import hoco_ml
import hoco_context

LIGHTNING_URLS = [
    "https://raw.githubusercontent.com/HandryOutlook/lightning_data_new/refs/heads/main/lightning_data.json",
    "https://raw.githubusercontent.com/HandryOutlook/lightning_data_new/refs/heads/main/lightning_data_2025_autumn.json",
    "https://raw.githubusercontent.com/HandryOutlook/lightning_data_new_2026/refs/heads/main/lightning_data_2026_summer_part1.json",
    "https://raw.githubusercontent.com/HandryOutlook/lightning_data_new_2026/refs/heads/main/lightning_data_2026_summer.json",
]
ROWS_BLOB = "ml/training_rows.npz"
ROLLING_DAYS = 365                # default look-back; --since overrides
MAX_NEW_RUNS_PER_PASS = 600       # --max-runs overrides (use ~4000 for the first backlog pass)
LEGACY_RUN_KEEP = 0.25            # old runs (no per-model data) only feed calibration; AUTO issues 8
                                  # near-identical runs per window per day, so a quarter is plenty
MIN_LIGHTNING_DAYS = 8            # separate days with lightning before AI HOCO may be published
BUCKETS = ["day0", "day1", "day2"]

# Negative sampling (weights restore the true balance, so probabilities stay calibrated)
KEEP_SIGNAL_NEG = 0.30   # cells with forecast signal but no lightning
KEEP_QUIET_NEG = 0.03    # cells with no signal and no lightning
SIGNAL_EPS = 0.01        # x_ens >= 1% or any model field > 0.05 counts as "signal"

# Gates
MIN_CAL_ROWS, MIN_CAL_POS, MIN_CAL_RUNS = 3000, 60, 15
MIN_FIT_POS, MIN_FIT_RUNS = 200, 20
MIN_BRIER_IMPROVEMENT = 0.01    # must beat baseline on held-out runs: Brier -1% ...
MIN_LOGLOSS_IMPROVEMENT = 0.05  # ... or log-loss -5%, and never be worse on either
L2 = 1e-3

CTX_COLS = ["region_i", "fh", "fw", "crow", "ccol", "ctx_ok", "doy", "setup", "season", "cape", "u500", "v500", "s", "s_down"]
COLS = ["run", "bucket", "t_end", "x_ens", "label", "weight"] + [f"m_{m}" for m in hoco_ml.LIGHTNING_MODELS] + CTX_COLS
REGION_LIST = list(hoco_ml.REGION_BOUNDS.keys())
MIN_CTX_POS, MIN_CTX_RUNS = 400, 30
L2_CTX = 3e-3
CTX = None  # ContextEnv, set in main()


# ============================================================================
# Firebase
# ============================================================================
def init_firebase():
    import firebase_admin
    from firebase_admin import credentials, firestore, storage
    if not firebase_admin._apps:
        info = json.loads(os.environ["FIREBASE_SERVICE_ACCOUNT"])
        firebase_admin.initialize_app(credentials.Certificate(info), {
            "storageBucket": "hoco-3b23e.firebasestorage.app"
        })
    return firestore.client(), storage.bucket()


# ============================================================================
# Lightning observations
# ============================================================================
_TIME_KEYS = ["strike_time", "time", "timestamp", "date", "datetime", "created_at", "t", "ts", "epoch", "utc", "dt"]


def _parse_time_ms(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) * 1000.0 if value < 1e10 else float(value)
    s = str(value).strip()
    if re.fullmatch(r"\d{9,13}", s):
        n = float(s)
        return n * 1000.0 if n < 1e10 else n
    m = re.match(r"^(\d{2})/(\d{2})/(\d{4})[\sT]*(\d{2}:\d{2}(?::\d{2})?)?", s)
    if m:
        dd, mm, yyyy, hms = m.groups()
        s = f"{yyyy}-{mm}-{dd}T{hms or '00:00:00'}"
        if len(s) == 16:
            s += ":00"
        s += "Z"
    elif re.match(r"^\d{8}T\d{6}", s):
        s = f"{s[0:4]}-{s[4:6]}-{s[6:8]}T{s[9:11]}:{s[11:13]}:{s[13:15]}Z"
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp() * 1000.0
    except ValueError:
        return None


def parse_strike(s):
    if s is None:
        return None
    props = s.get("properties", s) if isinstance(s, dict) else s
    try:
        if isinstance(s, dict) and isinstance(s.get("geometry"), dict) and len(s["geometry"].get("coordinates", [])) >= 2:
            lon, lat = map(float, s["geometry"]["coordinates"][:2])
        elif isinstance(props, dict) and isinstance(props.get("coordinates"), list) and len(props["coordinates"]) >= 2:
            lon, lat = map(float, props["coordinates"][:2])
        elif isinstance(s, list) and len(s) >= 2:
            lon, lat = float(s[0]), float(s[1])
        else:
            lon = float(props.get("lon", props.get("lng", props.get("longitude", props.get("x")))))
            lat = float(props.get("lat", props.get("latitude", props.get("y"))))
    except (TypeError, ValueError):
        return None
    if abs(lon) > 30 and abs(lat) <= 30:
        lon, lat = lat, lon
    tval = None
    if isinstance(props, dict):
        for k in _TIME_KEYS:
            if props.get(k) is not None:
                tval = props[k]
                break
    elif isinstance(s, list) and len(s) > 2:
        tval = s[2]
    t = _parse_time_ms(tval)
    if t is None or not np.isfinite(lon) or not np.isfinite(lat):
        return None
    return lon, lat, t


def load_strikes():
    raw = []
    for url in LIGHTNING_URLS:
        try:
            res = requests.get(url, timeout=60)
            res.raise_for_status()
            data = res.json()
            arr = data if isinstance(data, list) else (
                data.get("features") or data.get("strikes") or data.get("data")
                or next((v for v in data.values() if isinstance(v, list)), []))
            raw.extend(arr)
            print(f"[LIGHTNING] {len(arr)} from {url}")
        except Exception as e:
            print(f"[LIGHTNING] Failed {url}: {e}")
    parsed = [p for p in (parse_strike(s) for s in raw) if p]
    if not parsed:
        raise RuntimeError("No lightning strikes could be parsed — cannot label training data.")
    arr = np.array(parsed, dtype=np.float64)
    arr = np.unique(np.round(arr, 5), axis=0)  # archives can overlap; drop exact duplicate strikes
    order = np.argsort(arr[:, 2])
    arr = arr[order]
    print(f"[LIGHTNING] Parsed {len(arr)}/{len(raw)} strikes, "
          f"{datetime.fromtimestamp(arr[0, 2] / 1000, timezone.utc):%Y-%m-%d} → "
          f"{datetime.fromtimestamp(arr[-1, 2] / 1000, timezone.utc):%Y-%m-%d %H:%M}Z")
    return arr  # columns lon, lat, t_ms (sorted by time)


def strikes_in_window(strikes, start_ms, end_ms):
    lo = np.searchsorted(strikes[:, 2], start_ms, side="left")
    hi = np.searchsorted(strikes[:, 2], end_ms, side="right")
    return strikes[lo:hi]


# ============================================================================
# Run -> rows
# ============================================================================
def parse_hoco_time(value):
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%d/%m/%Y %H:%M").replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None


def label_grid(strikes, region, h, w):
    gh, gw = hoco_ml.grid_shape(h, w)
    label = np.zeros((gh, gw), dtype=np.int8)
    if len(strikes):
        r, c = hoco_ml.lonlat_to_cell(strikes[:, 0], strikes[:, 1], region, h, w)
        ok = r >= 0
        label[r[ok], c[ok]] = 1
    return label


def legacy_ensemble_grid(geojson, region):
    """Rebuild the 7 px point grid from an old geojson (risk% -> prob)."""
    pts = []
    for f in geojson.get("features", []):
        props = f.get("properties") or {}
        geom = f.get("geometry") or {}
        if props.get("feature_type") != "point" or geom.get("type") != "Point":
            continue
        lon, lat = geom["coordinates"][:2]
        pts.append((lon, lat, float(props.get("risk", 0)) / 100.0))
    if len(pts) < 4:
        return None
    pts = np.array(pts)
    b = hoco_ml.region_bounds(region)
    ulon = np.unique(np.round(pts[:, 0], 4))
    ulat = np.unique(np.round(pts[:, 1], 4))
    if len(ulon) < 2 or len(ulat) < 2:
        return None
    dlon = np.median(np.diff(ulon))
    dlat = np.median(np.diff(ulat))
    w = int(round(hoco_ml.CELL_PX * (b["right"] - b["left"]) / dlon))
    h = int(round(hoco_ml.CELL_PX * (b["top"] - b["bottom"]) / dlat))
    if not (100 < w < 4000 and 100 < h < 4000):
        return None
    gh, gw = hoco_ml.grid_shape(h, w)
    grid = np.zeros((gh, gw), dtype=np.float32)
    r, c = hoco_ml.lonlat_to_cell(pts[:, 0], pts[:, 1], region, h, w)
    ok = r >= 0
    grid[r[ok], c[ok]] = np.maximum(grid[r[ok], c[ok]], pts[ok, 2])
    return grid, h, w


def rows_from_grids(run_idx, bucket_idx, t_end_days, x_ens, label, model_grids, rng, region="England",
                    size=(0, 0), ctx=None, s_grid=None, s_down=None):
    n = x_ens.size
    x = x_ens.reshape(-1)
    y = label.reshape(-1).astype(np.int8)
    feats = {m: np.full(n, np.nan, dtype=np.float32) for m in hoco_ml.LIGHTNING_MODELS}
    any_model = np.zeros(n, dtype=bool)
    for m, g in (model_grids or {}).items():
        if m in feats:
            feats[m] = g.reshape(-1).astype(np.float32)
            any_model |= feats[m] > 0.05
    signal = (x >= SIGNAL_EPS) | any_model
    keep_p = np.where(y == 1, 1.0, np.where(signal, KEEP_SIGNAL_NEG, KEEP_QUIET_NEG))
    keep = rng.random(n) < keep_p
    out = {
        "run": np.full(keep.sum(), run_idx, dtype=np.int32),
        "bucket": np.full(keep.sum(), bucket_idx, dtype=np.int8),
        "t_end": np.full(keep.sum(), t_end_days, dtype=np.float64),
        "x_ens": x[keep].astype(np.float32),
        "label": y[keep],
        "weight": (1.0 / keep_p[keep]).astype(np.float32),
    }
    for m in hoco_ml.LIGHTNING_MODELS:
        out[f"m_{m}"] = feats[m][keep]
    k = int(keep.sum())
    crow, ccol = np.unravel_index(np.nonzero(keep)[0], x_ens.shape)
    out.update({
        "region_i": np.full(k, REGION_LIST.index(region) if region in REGION_LIST else 0, dtype=np.int16),
        "fh": np.full(k, size[0], dtype=np.int32), "fw": np.full(k, size[1], dtype=np.int32),
        "crow": crow.astype(np.int32), "ccol": ccol.astype(np.int32),
        "ctx_ok": np.full(k, 1 if ctx else 0, dtype=np.int8),
        "doy": np.full(k, ctx["doy"] if ctx else 0, dtype=np.int16),
        "setup": np.full(k, ctx["setup"] if ctx else 0, dtype=np.int8),
        "season": np.full(k, ctx["season"] if ctx else 0, dtype=np.int8),
        "cape": np.full(k, ctx["cape"] if ctx else 0.0, dtype=np.float32),
        "u500": np.full(k, ctx["u500"] if ctx else 0.0, dtype=np.float32),
        "v500": np.full(k, ctx["v500"] if ctx else 0.0, dtype=np.float32),
        "s": (s_grid.reshape(-1)[keep] if s_grid is not None else np.full(k, np.nan)).astype(np.float32),
        "s_down": (s_down.reshape(-1)[keep] if s_down is not None else np.full(k, np.nan)).astype(np.float32),
    })
    return out


def build_run_rows(run_id, run, strikes, gcs, run_idx, rng):
    region = run.get("region") or "England"
    if isinstance(region, list):
        region = region[0]
    start = parse_hoco_time(run.get("startTime"))
    end = parse_hoco_time(run.get("endTime") or run.get("startTime"))
    if not start or not end:
        return None, "bad times"
    bucket = hoco_ml.run_bucket(run_id, start.replace(tzinfo=None))
    bucket_idx = BUCKETS.index(bucket)
    win = strikes_in_window(strikes, start.timestamp() * 1000, end.timestamp() * 1000)

    # Preferred: per-model feature log from the new cloud.py
    blob = gcs.blob(run.get("mlFeaturePath") or f"{hoco_ml.FEATURE_PREFIX}/{run_id}.npz")
    if blob.exists():
        meta, arrays = hoco_ml.read_feature_npz(blob.download_as_bytes())
        h, w = meta["height"], meta["width"]
        x_ens = np.clip(arrays["ensemble"] * 0.2, 0, 1)  # raw ensemble, BEFORE calibration
        model_grids = {k[2:]: v for k, v in arrays.items() if k.startswith("m_")}
        label = label_grid(win, region, h, w)
        ctx = meta.get("context") or (CTX.run_ctx(region, start, end) if CTX else None)
        present = [m for m in hoco_ml.LIGHTNING_MODELS if m in model_grids]
        s_grid = hoco_context.model_mean(model_grids, present)
        s_down = hoco_context.shift_downstream(s_grid, ctx, region, h, w) if s_grid is not None else None
        return rows_from_grids(run_idx, bucket_idx, end.timestamp() / 86400, x_ens, label, model_grids, rng,
                               region=region, size=(h, w), ctx=ctx, s_grid=s_grid, s_down=s_down), "features"

    # Legacy: rebuild ensemble from the geojson point grid (only valid for uncalibrated runs)
    import zlib
    if (zlib.crc32(run_id.encode()) % 1000) / 1000.0 >= LEGACY_RUN_KEEP:
        return None, "legacy (sampled out)"
    mode = (run.get("ml") or {}).get("mode", "hand_weights")
    if mode != "hand_weights":
        return None, "calibrated run without feature log"
    path = run.get("geojsonPath") or f"forecasts/{run_id}.geojson"
    gblob = gcs.blob(path)
    if not gblob.exists():
        return None, "no geojson"
    try:
        geo = json.loads(gblob.download_as_bytes())
    except Exception as e:
        return None, f"bad geojson: {e}"
    if isinstance(geo, dict) and (geo.get("properties") or {}).get("hoco_layout") == "bands_v2":
        return None, "bands_v2 geojson without feature log"
    built = legacy_ensemble_grid(geo, region)
    if built is None:
        return None, "no point grid"
    x_ens, h, w = built
    label = label_grid(win, region, h, w)
    return rows_from_grids(run_idx, bucket_idx, end.timestamp() / 86400, x_ens, label, None, rng,
                           region=region, size=(h, w)), "legacy"


# ============================================================================
# Row cache
# ============================================================================
def load_rows(gcs, rebuild=False):
    empty = {c: np.zeros(0, dtype=np.float32) for c in COLS}
    if rebuild:
        return empty, []
    blob = gcs.blob(ROWS_BLOB)
    if not blob.exists():
        return empty, []
    with np.load(io.BytesIO(blob.download_as_bytes()), allow_pickle=False) as npz:
        rows = {c: npz[c] for c in COLS if c in npz.files}
        run_ids = json.loads(bytes(npz["run_ids"]).decode()) if "run_ids" in npz.files else []
    for c in COLS:
        rows.setdefault(c, np.full(len(rows["label"]), np.nan, dtype=np.float32))
    return rows, run_ids


def save_rows(gcs, rows, run_ids):
    buf = io.BytesIO()
    np.savez_compressed(buf, run_ids=np.frombuffer(json.dumps(run_ids).encode(), dtype=np.uint8),
                        **{c: rows[c] for c in COLS})
    gcs.blob(ROWS_BLOB).upload_from_string(buf.getvalue(), content_type="application/octet-stream")
    print(f"[CACHE] Saved {len(rows['label'])} rows from {len(run_ids)} runs ({len(buf.getvalue()) / 1e6:.1f} MB).")


def concat_rows(a, b):
    return {c: np.concatenate([a[c], b[c]]) for c in COLS}


def subset(rows, mask):
    return {c: v[mask] for c, v in rows.items()}



# ============================================================================
# Context environment: upper-air history (cached) + climatology maps
# ============================================================================
class ContextEnv:
    """Caches Open-Meteo history in Storage and builds leakage-free climatology maps."""

    def __init__(self, gcs, strikes, offline=False):
        self.gcs, self.strikes, self.offline = gcs, strikes, offline
        self.hourly, self.dirty, self.failed = {}, set(), set()
        self.maps_cache, self.saved_maps = {}, {}

    def _blob(self, region):
        return self.gcs.blob(f"ml/context/hourly_{region.replace(' ', '_')}.json")

    def series(self, region):
        if region not in self.hourly:
            blob = self._blob(region)
            try:
                self.hourly[region] = json.loads(blob.download_as_bytes()) if blob.exists() else {}
            except Exception:
                self.hourly[region] = {}
        return self.hourly[region]

    def ensure(self, region, start, end):
        ser = self.series(region)
        if self.offline or region in self.failed:
            return ser
        day = start.replace(hour=0, minute=0, second=0, microsecond=0)
        missing = []
        while day <= end:
            if day.strftime("%Y-%m-%dT12:00") not in ser:
                missing.append(day)
            day += timedelta(days=1)
        i = 0
        while i < len(missing):
            chunk = missing[i:i + 90]
            try:
                got = hoco_context.fetch_hourly(region, chunk[0], chunk[-1], historical=True)
                ser.update({k: list(v) for k, v in got.items()})
                self.dirty.add(region)
            except Exception as e:
                print(f"[CONTEXT] History fetch failed for {region} ({e}); context will be neutral where missing.")
                self.failed.add(region)
                break
            i += 90
        return ser

    def run_ctx(self, region, start, end):
        ser = self.ensure(region, start, end)
        return hoco_context.summarize_window({k: tuple(v) for k, v in ser.items()}, start, end)

    def daily_setups(self, region):
        if self.strikes is None or not len(self.strikes):
            return {}
        first = datetime.fromtimestamp(self.strikes[0, 2] / 1000, timezone.utc)
        last = datetime.fromtimestamp(self.strikes[-1, 2] / 1000, timezone.utc)
        ser = {k: tuple(v) for k, v in self.ensure(region, first, last).items()}
        out, day = {}, first.replace(hour=0, minute=0, second=0, microsecond=0)
        while day <= last:
            summ = hoco_context.summarize_window(ser, day, day + timedelta(hours=23))
            if summ:
                out[day.strftime("%Y-%m-%d")] = (summ["setup"], summ["season"])
            day += timedelta(days=1)
        return out

    def maps(self, keys, cutoff_ms=None):
        cut = None if cutoff_ms is None else int(cutoff_ms // 86400000)
        out = {}
        for region, h, w in keys:
            ck = (cut, region, h, w)
            if ck not in self.maps_cache:
                ds = self.daily_setups(region)
                self.maps_cache[ck] = hoco_context.build_climatology(
                    self.strikes, ds, region, h, w, None if cut is None else cut * 86400000) if ds else None
            out[hoco_context.map_key(region, h, w)] = self.maps_cache[ck]
        return out

    def load_saved(self, path):
        if path not in self.saved_maps:
            blob = self.gcs.blob(path)
            self.saved_maps[path] = hoco_context.load_climatology(blob.download_as_bytes()) if blob.exists() else {}
        return self.saved_maps[path]

    def save(self):
        for region in self.dirty:
            self._blob(region).upload_from_string(json.dumps(self.hourly[region]), content_type="application/json")
        self.dirty.clear()


def row_map_keys(rows):
    keys = set()
    for ri, fh, fw in set(zip(rows["region_i"].tolist(), rows["fh"].tolist(), rows["fw"].tolist())):
        if fh and fw:
            keys.add((REGION_LIST[int(ri)], int(fh), int(fw)))
    return sorted(keys)


def context_design(rows, models, maps):
    X = feature_matrix(rows, models)
    s = np.nan_to_num(rows["s"].astype(np.float64), nan=0.0)
    sd = np.nan_to_num(rows["s_down"].astype(np.float64), nan=0.0)
    n = len(s)
    f_clim, f_setup = np.zeros(n), np.zeros(n)
    ok = rows["ctx_ok"] == 1
    for region, h, w in row_map_keys(rows):
        m = (maps or {}).get(hoco_context.map_key(region, h, w))
        if m is None:
            continue
        sel = (rows["region_i"].astype(int) == REGION_LIST.index(region)) & (rows["fh"].astype(int) == h) & (rows["fw"].astype(int) == w)
        r, c = rows["crow"][sel].astype(np.int64), rows["ccol"][sel].astype(np.int64)
        f_clim[sel] = m["clim"][r, c]
        combo = rows["setup"][sel].astype(int) * len(hoco_context.SEASONS) + rows["season"][sel].astype(int)
        f_setup[sel] = np.where(ok[sel], m["setup"][combo, r, c], 0.0)
    ang = 2 * np.pi * (rows["doy"].astype(np.float64) - 15) / 365.25
    cos_d = np.where(ok, np.cos(ang), 0.0)
    sin_d = np.where(ok, np.sin(ang), 0.0)
    cape_n = np.where(ok, np.log1p(np.maximum(rows["cape"].astype(np.float64), 0)) / np.log1p(1000.0), 0.0)
    C = np.column_stack(hoco_context.context_columns(s, f_clim, f_setup, cos_d, sin_d, cape_n))
    return np.column_stack([X, sd]), C


def fit_glm(Xpos, C, y, w):
    """Logistic regression: non-negative weights on model signals, free-sign context terms."""
    wn = w / np.sum(w)
    kp, kc = Xpos.shape[1], C.shape[1]
    base = np.clip(np.sum(wn * y), 1e-4, 0.5)
    theta0 = np.concatenate([[np.log(base / (1 - base))], np.full(kp, 0.1), np.zeros(kc)])
    l2 = np.concatenate([[0.0], np.full(kp, L2), np.full(kc, L2_CTX)])

    def loss_grad(theta):
        z = theta[0] + Xpos @ theta[1:1 + kp] + C @ theta[1 + kp:]
        p = np.clip(hoco_ml.sigmoid(z), 1e-9, 1 - 1e-9)
        loss = -np.sum(wn * (y * np.log(p) + (1 - y) * np.log(1 - p))) + np.sum(l2 * theta ** 2)
        r = wn * (p - y)
        grad = np.concatenate([[np.sum(r)], Xpos.T @ r, C.T @ r]) + 2 * l2 * theta
        return loss, grad

    res = minimize(loss_grad, theta0, jac=True, method="L-BFGS-B",
                   bounds=[(None, None)] + [(0.0, None)] * kp + [(None, None)] * kc, options={"maxiter": 800})
    return res.x, bool(res.success)


def fit_context_params(rows, models, maps):
    Xpos, C = context_design(rows, models, maps)
    theta, ok = fit_glm(Xpos, C, rows["label"].astype(np.float64), rows["weight"].astype(np.float64))
    k = len(models)
    return {
        "intercept": round(float(theta[0]), 5),
        "weights": {m: round(float(v), 5) for m, v in zip(models, theta[1:1 + k])},
        "downstream": round(float(theta[1 + k]), 5),
        "downstreamHours": hoco_context.DOWNSTREAM_HOURS,
        "context": {c: round(float(v), 5) for c, v in zip(hoco_context.CONTEXT_COLS, theta[2 + k:])},
        "models": list(models),
    }, ok


def predict_context(params, rows, maps):
    models = params.get("models") or list(params["weights"].keys())
    Xpos, C = context_design(rows, models, maps)
    theta = np.concatenate([[params["intercept"]], [params["weights"].get(m, 0.0) for m in models],
                            [params.get("downstream", 0.0)],
                            [params["context"].get(c, 0.0) for c in hoco_context.CONTEXT_COLS]])
    kp = Xpos.shape[1]
    return hoco_ml.sigmoid(theta[0] + Xpos @ theta[1:1 + kp] + C @ theta[1 + kp:])


def context_rows(rows):
    return subset(rows, has_features_mask(rows) & np.isfinite(rows["s"]))


def val_cutoff_ms(val):
    return float((val["t_end"].min() - 1.5) * 86400000)


def train_context(rows, version, trained_at, models_hint=None):
    rows = context_rows(rows)
    n_runs = len(np.unique(rows["run"])) if len(rows["label"]) else 0
    n_pos = int(rows["label"].sum()) if len(rows["label"]) else 0
    info = {"status": "insufficient_data", "nRows": int(len(rows["label"])), "nPositives": n_pos, "nRuns": n_runs,
            "needs": {"positives": MIN_CTX_POS, "runs": MIN_CTX_RUNS}}
    if CTX is None or n_runs < MIN_CTX_RUNS or n_pos < MIN_CTX_POS:
        return info
    models = [m for m in hoco_ml.LIGHTNING_MODELS
              if np.mean(np.isfinite(rows[f"m_{m}"])) > 0.2 and np.nanmax(rows[f"m_{m}"]) > 0]
    train, val = split_by_time(rows)
    if train is None or val["label"].sum() < 20:
        info["status"] = "insufficient_validation"
        return info
    maps_tr = CTX.maps(row_map_keys(rows), val_cutoff_ms(val))
    params, ok = fit_context_params(train, models, maps_tr)
    vy, vw = val["label"].astype(np.float64), val["weight"].astype(np.float64)
    p_new = predict_context(params, val, maps_tr)
    p_base = np.clip(val["x_ens"].astype(np.float64), 0, 1)
    base, new = metric_block(p_base, vy, vw), metric_block(p_new, vy, vw)
    passes = ok and new["brier"] <= base["brier"] and new["logloss"] <= base["logloss"] and (
        new["brier"] <= base["brier"] * (1 - MIN_BRIER_IMPROVEMENT)
        or new["logloss"] <= base["logloss"] * (1 - MIN_LOGLOSS_IMPROVEMENT))

    maps_all = CTX.maps(row_map_keys(rows), None)
    final, ok_all = fit_context_params(rows, models, maps_all)
    info.update({
        "status": "active" if (passes and ok_all) else "shadow",
        "version": version, "trainedAt": trained_at, **final,
        "validation": {"baselineName": "hand_weights", "baseline": base, "model": new},
        "setupEffects": setup_effects(final, rows, maps_all),
        "_maps": maps_all,
    })
    X = feature_matrix(rows, models)
    signal = rows["x_ens"] >= 0.05
    contrib = np.array([final["weights"][m] for m in models]) * (X[signal].mean(axis=0) if signal.any() else X.mean(axis=0))
    info["importance"] = {m: round(float(v), 4) for m, v in zip(models, contrib / contrib.sum() if contrib.sum() > 0 else contrib)}
    return info


def setup_effects(params, rows, maps):
    """Per setup x season: what Traditional said, what AI says, what happened (where models show storms)."""
    out = []
    p_ai = predict_context(params, rows, maps)
    sig = (np.nan_to_num(rows["s"], nan=0.0) >= 0.25) & (rows["ctx_ok"] == 1)
    for si, sname in enumerate(hoco_context.SETUPS):
        for qi, qname in enumerate(hoco_context.SEASONS):
            m = sig & (rows["setup"] == si) & (rows["season"] == qi)
            wsum = float(np.sum(rows["weight"][m]))
            if wsum <= 0:
                continue
            w = rows["weight"][m].astype(np.float64)
            out.append({
                "setup": sname, "season": qname,
                "runs": int(len(np.unique(rows["run"][m]))),
                "traditional": round(float(np.sum(w * np.clip(rows["x_ens"][m], 0, 1)) / wsum), 4),
                "ai": round(float(np.sum(w * p_ai[m]) / wsum), 4),
                "observed": round(float(np.sum(w * rows["label"][m]) / wsum), 4),
            })
    return out

# ============================================================================
# Metrics
# ============================================================================
def brier(p, y, w):
    return float(np.sum(w * (p - y) ** 2) / np.sum(w))


def logloss(p, y, w):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(w))


def auc(p, y, w):
    pos_w, neg_w = np.sum(w[y == 1]), np.sum(w[y == 0])
    if pos_w == 0 or neg_w == 0:
        return None
    order = np.argsort(p, kind="mergesort")
    p, y, w = p[order], y[order], w[order]
    # tie-aware: group equal scores
    _, starts = np.unique(p, return_index=True)
    wp = np.add.reduceat(w * (y == 1), starts)
    wn = np.add.reduceat(w * (y == 0), starts)
    cum_neg_before = np.concatenate([[0.0], np.cumsum(wn)[:-1]])
    total = float(np.sum(wp * (cum_neg_before + 0.5 * wn)))
    return float(total / (pos_w * neg_w))


def metric_block(p, y, w):
    return {"brier": round(brier(p, y, w), 6), "logloss": round(logloss(p, y, w), 6),
            "auc": None if (a := auc(p, y, w)) is None else round(a, 4),
            "n": int(len(y)), "positives": int(np.sum(y))}


def reliability(p, y, w, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1)
        if np.sum(w[m]) > 0:
            out.append({"forecast": round(float(np.sum(w[m] * p[m]) / np.sum(w[m])), 4),
                        "observed": round(float(np.sum(w[m] * y[m]) / np.sum(w[m])), 4),
                        "weight": round(float(np.sum(w[m])), 1)})
    return out


# ============================================================================
# Stage 1 — isotonic calibration
# ============================================================================
def fit_isotonic(x, y, w, max_knots=60):
    # Pre-bin to keep PAV fast on millions of rows
    order = np.argsort(x)
    x, y, w = x[order], y[order], w[order]
    n_bins = min(400, max(10, len(x) // 50))
    split = np.array_split(np.arange(len(x)), n_bins)
    bx = np.array([np.sum(w[s] * x[s]) / np.sum(w[s]) for s in split])
    by = np.array([np.sum(w[s] * y[s]) / np.sum(w[s]) for s in split])
    bw = np.array([np.sum(w[s]) for s in split])

    # Pool-adjacent-violators
    vals, wts, xs = [], [], []
    for xi, yi, wi in zip(bx, by, bw):
        vals.append(yi); wts.append(wi); xs.append([xi])
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            wts[-2] += wts[-1]; vals[-2] = v; xs[-2] += xs[-1]
            vals.pop(); wts.pop(); xs.pop()
    kx = np.array([np.mean(g) for g in xs])
    ky = np.array(vals)

    # Anchor ends so interpolation covers [0, 1]
    kx = np.concatenate([[0.0], kx, [1.0]])
    ky = np.concatenate([[ky[0]], ky, [ky[-1]]])
    kx, keep = np.unique(kx, return_index=True)
    ky = np.maximum.accumulate(ky[keep])
    if len(kx) > max_knots:
        sel = np.unique(np.round(np.linspace(0, len(kx) - 1, max_knots)).astype(int))
        kx, ky = kx[sel], ky[sel]
    ky = np.clip(ky, 0.0005, 0.995)
    return {"x": [round(float(v), 5) for v in kx], "y": [round(float(v), 5) for v in ky]}


# ============================================================================
# Stage 2 — non-negative logistic regression (the learned weights)
# ============================================================================
def feature_matrix(rows, models):
    return np.column_stack([np.nan_to_num(np.maximum(rows[f"m_{m}"], 0), nan=0.0) for m in models]).astype(np.float64)


def fit_logistic(X, y, w):
    wn = w / np.sum(w)
    k = X.shape[1]
    base = np.clip(np.sum(wn * y), 1e-4, 0.5)
    theta0 = np.concatenate([[np.log(base / (1 - base))], np.full(k, 0.1)])

    def loss_grad(theta):
        z = theta[0] + X @ theta[1:]
        p = hoco_ml.sigmoid(z)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        loss = -np.sum(wn * (y * np.log(p) + (1 - y) * np.log(1 - p))) + L2 * np.sum(theta[1:] ** 2)
        r = wn * (p - y)
        grad = np.concatenate([[np.sum(r)], X.T @ r + 2 * L2 * theta[1:]])
        return loss, grad

    res = minimize(loss_grad, theta0, jac=True, method="L-BFGS-B",
                   bounds=[(None, None)] + [(0.0, None)] * k, options={"maxiter": 500})
    return float(res.x[0]), res.x[1:], bool(res.success)


# ============================================================================
# Training orchestration
# ============================================================================
def split_by_time(rows, frac_val=0.25):
    runs = np.unique(rows["run"])
    if len(runs) < 4:
        return None, None
    run_end = {r: rows["t_end"][rows["run"] == r].max() for r in runs}
    ordered = sorted(runs, key=lambda r: run_end[r])
    n_val = max(1, int(len(ordered) * frac_val))
    val_runs = set(ordered[-n_val:])
    is_val = np.isin(rows["run"], list(val_runs))
    return subset(rows, ~is_val), subset(rows, is_val)


def train_calibration(rows, version, trained_at):
    y, w, x = rows["label"].astype(np.float64), rows["weight"].astype(np.float64), rows["x_ens"].astype(np.float64)
    n_runs = len(np.unique(rows["run"]))
    info = {"status": "insufficient_data", "nRows": int(len(y)), "nPositives": int(y.sum()), "nRuns": n_runs}
    if len(y) < MIN_CAL_ROWS or y.sum() < MIN_CAL_POS or n_runs < MIN_CAL_RUNS:
        return info
    train, val = split_by_time(rows)
    if train is None or val["label"].sum() < 10:
        info["status"] = "insufficient_validation"
        return info

    cal = fit_isotonic(train["x_ens"].astype(np.float64), train["label"].astype(np.float64), train["weight"].astype(np.float64))
    vy, vw, vx = val["label"].astype(np.float64), val["weight"].astype(np.float64), val["x_ens"].astype(np.float64)
    p_new = hoco_ml.apply_isotonic(vx, cal).astype(np.float64)
    base = metric_block(np.clip(vx, 0, 1), vy, vw)
    new = metric_block(p_new, vy, vw)
    passes = new["brier"] <= base["brier"] and new["logloss"] <= base["logloss"] and (
        new["brier"] <= base["brier"] * (1 - MIN_BRIER_IMPROVEMENT)
        or new["logloss"] <= base["logloss"] * (1 - MIN_LOGLOSS_IMPROVEMENT))

    final = fit_isotonic(x, y, w)  # refit on everything once it has proved itself
    info.update(final)
    info.update({
        "status": "active" if passes else "shadow",
        "version": version, "trainedAt": trained_at,
        "validation": {"baseline": base, "model": new,
                       "reliabilityBaseline": reliability(np.clip(vx, 0, 1), vy, vw),
                       "reliabilityModel": reliability(p_new, vy, vw)},
    })
    return info


def train_weights(rows, version, trained_at, calibration_info):
    has_feat = np.zeros(len(rows["label"]), dtype=bool)
    for m in hoco_ml.LIGHTNING_MODELS:
        has_feat |= np.isfinite(rows[f"m_{m}"])
    rows = subset(rows, has_feat)
    n_runs = len(np.unique(rows["run"])) if len(rows["label"]) else 0
    info = {"status": "insufficient_data", "nRows": int(len(rows["label"])),
            "nPositives": int(rows["label"].sum()) if len(rows["label"]) else 0, "nRuns": n_runs,
            "needs": {"positives": MIN_FIT_POS, "runs": MIN_FIT_RUNS}}
    if n_runs < MIN_FIT_RUNS or info["nPositives"] < MIN_FIT_POS:
        return info

    models = [m for m in hoco_ml.LIGHTNING_MODELS
              if np.mean(np.isfinite(rows[f"m_{m}"])) > 0.2 and np.nanmax(rows[f"m_{m}"]) > 0]
    train, val = split_by_time(rows)
    if train is None or val["label"].sum() < 20:
        info["status"] = "insufficient_validation"
        return info

    b0, wts, ok = fit_logistic(feature_matrix(train, models), train["label"].astype(np.float64), train["weight"].astype(np.float64))
    vy, vw = val["label"].astype(np.float64), val["weight"].astype(np.float64)
    p_new = hoco_ml.sigmoid(b0 + feature_matrix(val, models) @ wts)

    # Baseline = whatever production would otherwise show: calibrated ensemble if active, else raw
    vx = np.clip(val["x_ens"].astype(np.float64), 0, 1)
    if calibration_info.get("status") == "active":
        p_base = hoco_ml.apply_isotonic(vx, calibration_info).astype(np.float64)
        base_name = "calibrated_ensemble"
    else:
        p_base, base_name = vx, "hand_weights"
    base = metric_block(p_base, vy, vw)
    new = metric_block(p_new, vy, vw)
    passes = ok and new["brier"] <= base["brier"] and new["logloss"] <= base["logloss"] and (
        new["brier"] <= base["brier"] * (1 - MIN_BRIER_IMPROVEMENT)
        or new["logloss"] <= base["logloss"] * (1 - MIN_LOGLOSS_IMPROVEMENT))

    # Refit on everything
    X_all = feature_matrix(rows, models)
    b0, wts, ok_all = fit_logistic(X_all, rows["label"].astype(np.float64), rows["weight"].astype(np.float64))
    signal = rows["x_ens"] >= 0.05
    mean_f = X_all[signal].mean(axis=0) if signal.any() else X_all.mean(axis=0)
    contrib = wts * mean_f
    importance = contrib / contrib.sum() if contrib.sum() > 0 else np.zeros_like(contrib)

    info.update({
        "status": "active" if (passes and ok_all) else "shadow",
        "version": version, "trainedAt": trained_at,
        "intercept": round(b0, 5),
        "weights": {m: round(float(v), 5) for m, v in zip(models, wts)},
        "importance": {m: round(float(v), 4) for m, v in zip(models, importance)},
        "baseRate": round(float(hoco_ml.sigmoid(b0)), 5),
        "validation": {"baselineName": base_name, "baseline": base, "model": new,
                       "reliabilityBaseline": reliability(p_base, vy, vw),
                       "reliabilityModel": reliability(p_new, vy, vw)},
    })
    return info


def train_all(rows, version, trained_at):
    calibration, weights, context = {}, {}, {}
    for scope in BUCKETS + ["pooled"]:
        sel = rows if scope == "pooled" else subset(rows, rows["bucket"] == BUCKETS.index(scope))
        calibration[scope] = train_calibration(sel, version, trained_at)
        weights[scope] = train_weights(sel, version, trained_at, calibration[scope])
        context[scope] = train_context(sel, version, trained_at)
        c, wv, cx = calibration[scope], weights[scope], context[scope]
        print(f"[TRAIN] {scope:6s} calibration={c['status']:<18} weights={wv['status']:<18} context={cx['status']:<18} "
              f"rows={c['nRows']} pos={c['nPositives']} runs={c['nRuns']}")
        if cx.get("context"):
            print(f"         context terms: {cx['context']}  downstream={cx.get('downstream')}")
        if wv.get("weights"):
            print(f"         learned weights: {wv['weights']}  (val Brier {wv['validation']['model']['brier']} "
                  f"vs {wv['validation']['baselineName']} {wv['validation']['baseline']['brier']})")
    return calibration, weights, context



# ============================================================================
# AI HOCO lifecycle: candidate -> readiness -> (admin publish) -> daily updates
# ============================================================================
READY_MAX_OBS_LAG_DAYS = 4
UPDATE_BRIER_TOLERANCE = 1.002    # a daily update may be at most 0.2% worse on Brier...
UPDATE_LOGLOSS_TOLERANCE = 1.005  # ...and 0.5% on log-loss than the published model, on fresh runs
HISTORY_KEEP = 180
UPDATE_LOG_KEEP = 90


def has_features_mask(rows):
    m = np.zeros(len(rows["label"]), dtype=bool)
    for name in hoco_ml.LIGHTNING_MODELS:
        m |= np.isfinite(rows[f"m_{name}"])
    return m


def fit_kind(kind, rows, models, cutoff_ms=None):
    y, w = rows["label"].astype(np.float64), rows["weight"].astype(np.float64)
    if kind == "context":
        maps = CTX.maps(row_map_keys(rows), cutoff_ms)
        params, _ = fit_context_params(rows, models, maps)
        params["_maps"] = maps
        return params
    if kind == "calibration":
        return fit_isotonic(rows["x_ens"].astype(np.float64), y, w)
    b0, wts, _ = fit_logistic(feature_matrix(rows, models), y, w)
    return {"intercept": round(b0, 5), "weights": {m: round(float(v), 5) for m, v in zip(models, wts)}}


def predict_rows(kind, params, rows):
    x = np.clip(rows["x_ens"].astype(np.float64), 0, 1)
    if kind == "context":
        maps = params.get("_maps")
        if maps is None and params.get("climatology") and CTX is not None:
            maps = CTX.load_saved(params["climatology"])
        return predict_context(params, rows, maps)
    if kind == "calibration":
        return hoco_ml.apply_isotonic(x, params).astype(np.float64)
    z = np.full(len(x), float(params.get("intercept", -4.0)))
    for m, wt in (params.get("weights") or {}).items():
        col = rows.get(f"m_{m}")
        if col is not None:
            z += float(wt) * np.nan_to_num(np.maximum(col, 0), nan=0.0)
    return hoco_ml.sigmoid(z)


CONTEXT_PARAM_KEYS = ("intercept", "weights", "downstream", "downstreamHours", "context", "models")


def build_candidate(calibration, weights, rows, context=None, climatology_path=None):
    """
    Best proven method per day. For each kind (context-aware, learned weights, calibration) take
    the first active scope (this day, then all days) that is no worse than Traditional HOCO on THIS
    day's newest unseen runs (Brier and log-loss). Then keep whichever kind scored best there.
    """
    sections = [("context", context or {}), ("weights", weights), ("calibration", calibration)]
    candidate = {}
    for b in BUCKETS:
        options = []
        for kind, section in sections:
            for scope in (b, "pooled"):
                info = section.get(scope) or {}
                if info.get("status") != "active":
                    continue
                if kind == "context":
                    params = {k: info[k] for k in CONTEXT_PARAM_KEYS if k in info}
                    params["climatology"] = climatology_path
                    params["_maps"] = info.get("_maps")
                elif kind == "weights":
                    params = {"intercept": info["intercept"], "weights": info["weights"]}
                else:
                    params = {"x": info["x"], "y": info["y"]}
                entry = {
                    "kind": kind, "scope": scope, "params": params,
                    "models": list(info.get("weights", {}).keys()) if kind != "calibration" else [],
                    "importance": info.get("importance"),
                    "version": info.get("version"), "trainedAt": info.get("trainedAt"),
                }
                if kind == "context":
                    entry["setupEffects"] = info.get("setupEffects")
                cmp = heldout_compare(entry, rows, b, reference=None)
                if cmp is None:
                    continue
                m, t = cmp["candidate"], cmp["reference"]
                if m["brier"] <= t["brier"] and m["logloss"] <= t["logloss"]:
                    entry["vsTraditional"] = {"model": m, "traditional": t,
                                              "reliability": cmp["reliability"],
                                              "reliabilityTraditional": cmp["reliabilityReference"]}
                    options.append(entry)
                    break
        if options:
            candidate[b] = min(options, key=lambda e: e["vsTraditional"]["model"]["brier"])
    return candidate


def strip_private(candidate):
    """Remove in-memory objects before writing to Firestore."""
    clean = {}
    for b, e in candidate.items():
        e = dict(e)
        e["params"] = {k: v for k, v in e["params"].items() if not k.startswith("_")}
        clean[b] = e
    return clean


def heldout_compare(cand, rows, bucket, reference=None):
    """
    Refit the candidate WITHOUT the newest 25% of this day's runs, score it on them,
    and score `reference` (a published entry, or Traditional HOCO when None) on the same rows.
    """
    feat = has_features_mask(rows) & (np.isfinite(rows["s"]) if "s" in rows else True)
    need_feat = cand["kind"] in ("weights", "context") or (reference or {}).get("kind") in ("weights", "context")
    usable = feat if need_feat else np.ones(len(feat), dtype=bool)
    rows_b = subset(rows, usable & (rows["bucket"] == BUCKETS.index(bucket)))
    if len(rows_b["label"]) == 0:
        return None
    _, val = split_by_time(rows_b)
    if val is None or val["label"].sum() < 10:
        return None
    val_runs = np.unique(val["run"])
    scope_rows = subset(rows, usable) if cand["scope"] == "pooled" else rows_b
    train = subset(scope_rows, ~np.isin(scope_rows["run"], val_runs))
    if len(train["label"]) == 0 or train["label"].sum() < 5:
        return None
    held = fit_kind(cand["kind"], train, cand["models"], cutoff_ms=val_cutoff_ms(val))
    y, w = val["label"].astype(np.float64), val["weight"].astype(np.float64)
    p_c = predict_rows(cand["kind"], held, val)
    if reference:
        p_r = predict_rows(reference["kind"], reference["params"], val)
    else:
        p_r = np.clip(val["x_ens"].astype(np.float64), 0, 1)
    return {"candidate": metric_block(p_c, y, w), "reference": metric_block(p_r, y, w),
            "reliability": reliability(p_c, y, w), "reliabilityReference": reliability(p_r, y, w)}


def assess_readiness(rows, candidate, coverage_end, now):
    feat = has_features_mask(rows)
    feat_runs = int(len(np.unique(rows["run"][feat]))) if feat.any() else 0
    feat_pos = int(rows["label"][feat].sum()) if feat.any() else 0
    pos_feat = feat & (rows["label"] == 1)
    light_days = int(len(np.unique(np.floor(rows["t_end"][pos_feat])))) if pos_feat.any() else 0
    lag_days = (now - coverage_end).total_seconds() / 86400
    checks = [
        {"id": "observations", "label": "Lightning observations are up to date",
         "ok": lag_days <= READY_MAX_OBS_LAG_DAYS, "detail": f"latest strike {lag_days:.1f} days ago"},
        {"id": "runs", "label": f"At least {MIN_FIT_RUNS} verified runs with per-model data",
         "ok": feat_runs >= MIN_FIT_RUNS, "detail": f"{feat_runs} / {MIN_FIT_RUNS}",
         "progress": min(1.0, feat_runs / MIN_FIT_RUNS)},
        {"id": "lightning", "label": f"At least {MIN_FIT_POS} lightning grid cells to learn from",
         "ok": feat_pos >= MIN_FIT_POS, "detail": f"{feat_pos} / {MIN_FIT_POS}",
         "progress": min(1.0, feat_pos / MIN_FIT_POS)},
        {"id": "lightning_days", "label": f"Lightning on at least {MIN_LIGHTNING_DAYS} separate days (independent events)",
         "ok": light_days >= MIN_LIGHTNING_DAYS, "detail": f"{light_days} / {MIN_LIGHTNING_DAYS}",
         "progress": min(1.0, light_days / MIN_LIGHTNING_DAYS)},
    ]
    for b in BUCKETS:
        c = candidate.get(b)
        checks.append({"id": f"{b}_beats", "label": f"{b.replace('day', 'Day ')} beats Traditional HOCO on unseen runs",
                       "ok": c is not None,
                       "detail": (f"{ {'context': 'weights + context', 'weights': 'learned weights', 'calibration': 'calibration'}[c['kind']]} ({c['scope']})" if c else "not yet")})
    c0 = candidate.get("day0")
    checks.append({"id": "day0_weights", "label": "Day 0 uses learned model weights (not just calibration)",
                   "ok": bool(c0 and c0["kind"] in ("weights", "context")),
                   "detail": {"context": "learned weights + weather context", "weights": "learned weights",
                              "calibration": "calibration only"}.get(c0["kind"], "") if c0 else "not yet"})
    return {"ready": all(c["ok"] for c in checks), "checks": checks,
            "featureRuns": feat_runs, "featurePositives": feat_pos, "lightningDays": light_days}


def daily_update(ai_doc, rows, candidate, version, trained_at):
    """After publish: swap in a day's new candidate only if it is no worse than what is live."""
    model = dict(ai_doc.get("model") or {})
    entries, changed = [], False
    for b in BUCKETS:
        cand, live = candidate.get(b), model.get(b)
        if not cand:
            entries.append({"bucket": b, "action": "kept", "reason": "no proven candidate today"})
            continue
        cmp = heldout_compare(cand, rows, b, reference=live)
        if cmp is None:
            entries.append({"bucket": b, "action": "kept", "reason": "not enough fresh verified runs to compare"})
            continue
        c, r = cmp["candidate"], cmp["reference"]
        ok = c["brier"] <= r["brier"] * UPDATE_BRIER_TOLERANCE and c["logloss"] <= r["logloss"] * UPDATE_LOGLOSS_TOLERANCE
        if ok:
            model[b] = {**cand, "comparison": {"vsLive": c, "live": r}}
            changed = True
        entries.append({"bucket": b, "action": "updated" if ok else "kept",
                        "reason": "no worse than live model on fresh runs" if ok else "would have been worse than live model",
                        "candidateBrier": c["brier"], "liveBrier": r["brier"]})
    log = (ai_doc.get("updateLog") or []) + [{"at": trained_at, "version": version, "changed": changed, "entries": entries}]
    return model, changed, log[-UPDATE_LOG_KEEP:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--since", type=str, default="", help="YYYY-MM-DD: use runs ending on/after this date")
    ap.add_argument("--max-runs", type=int, default=MAX_NEW_RUNS_PER_PASS, help="new runs to ingest this pass")
    args = ap.parse_args()

    db, gcs = init_firebase()
    strikes = load_strikes()
    coverage_end = datetime.fromtimestamp(strikes[-1, 2] / 1000, timezone.utc)
    now = datetime.now(timezone.utc)
    oldest = now - timedelta(days=ROLLING_DAYS)
    if args.since:
        oldest = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    print(f"[RUNS] Using runs that ended after {oldest:%Y-%m-%d}.")
    rng = np.random.default_rng(20261117)

    global CTX
    CTX = ContextEnv(gcs, strikes)
    rows, run_ids = load_rows(gcs, args.rebuild)
    done = set(run_ids)

    # Firestore cost: after the first pass only read runs created recently (single-field
    # createdAt index, no composite index needed); the first pass / --rebuild / --since reads all.
    col = db.collection("hoco_requests")
    if run_ids and not args.rebuild and not args.since:
        lag_days = max(0.0, (now - coverage_end).total_seconds() / 86400)
        recent = now - timedelta(days=7 + lag_days)
        snap = col.where("createdAt", ">=", recent).stream()
        print(f"[RUNS] Incremental read: runs created since {recent:%Y-%m-%d}.")
    else:
        snap = col.where("type", "in", ["automated", "ai_training"]).stream()
    todo = []
    for doc in snap:
        if doc.id in done:
            continue
        run = doc.to_dict()
        if run.get("type") not in ("automated", "ai_training"):
            continue
        if run.get("status") != "completed":
            continue
        end = parse_hoco_time(run.get("endTime") or run.get("startTime"))
        if not end or end < oldest or end > coverage_end:
            continue  # too old, or lightning observations don't cover it yet
        todo.append((end, doc.id, run))
    todo.sort(key=lambda t: t[0])
    backlog = len(todo)
    todo = todo[:max(1, args.max_runs)]
    print(f"[RUNS] {len(todo)} of {backlog} waiting runs will be added this pass "
          f"(observations cover up to {coverage_end:%Y-%m-%d %H:%M}Z).")

    stats = {}
    for end, run_id, run in todo:
        try:
            new_rows, how = build_run_rows(run_id, run, strikes, gcs, len(run_ids), rng)
        except Exception as e:
            new_rows, how = None, f"error: {e}"
        stats[how] = stats.get(how, 0) + 1
        run_ids.append(run_id)  # never retry a run; unusable runs are simply skipped
        if new_rows is not None:
            rows = concat_rows(rows, new_rows)
    print(f"[RUNS] Outcome: {stats}")

    # Rolling window
    if len(rows["label"]):
        keep = rows["t_end"] >= oldest.timestamp() / 86400
        rows = subset(rows, keep)

    if not args.dry_run:
        save_rows(gcs, rows, run_ids)

    if len(rows["label"]) == 0:
        print("[TRAIN] No rows yet — nothing to train.")
        return

    version = int(now.strftime("%Y%m%d%H"))
    trained_at = now.isoformat()
    calibration, weights, context = train_all(rows, version, trained_at)

    clim_path = None
    if any((context.get(sc) or {}).get("status") == "active" for sc in context):
        maps_all = next(c["_maps"] for c in context.values() if c.get("_maps"))
        clim_path = f"ml/context/clim_v{version}.npz"
        if not args.dry_run:
            gcs.blob(clim_path).upload_from_string(hoco_context.save_climatology(maps_all), content_type="application/octet-stream")
    candidate = build_candidate(calibration, weights, rows, context, clim_path)
    readiness = assess_readiness(rows, candidate, coverage_end, now)

    prev = {}
    try:
        snap = db.collection(hoco_ml.ML_CONFIG_COLLECTION).document(hoco_ml.ML_CONFIG_DOC).get()
        prev = snap.to_dict() if snap.exists else {}
    except Exception:
        pass

    def gain(b):
        vt = (candidate.get(b) or {}).get("vsTraditional")
        if not vt or not vt["traditional"]["brier"]:
            return None
        return round(1 - vt["model"]["brier"] / vt["traditional"]["brier"], 4)

    history = (prev.get("history") or []) + [{
        "at": trained_at, "rows": int(len(rows["label"])), "positives": int(rows["label"].sum()),
        "runs": int(len(np.unique(rows["run"]))), "featureRuns": readiness["featureRuns"],
        "featurePositives": readiness["featurePositives"], "ready": readiness["ready"],
        "gain": {b: gain(b) for b in BUCKETS},
    }]

    doc = {
        "version": version,
        "trainedAt": trained_at,
        "observationsCoverUntil": coverage_end.isoformat(),
        "data": {
            "rows": int(len(rows["label"])),
            "positives": int(rows["label"].sum()),
            "runs": int(len(np.unique(rows["run"]))),
            "featureRuns": readiness["featureRuns"],
            "windowDays": ROLLING_DAYS,
        },
        "calibration": calibration,
        "weights": weights,
        "context": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in context.items()},
        "candidate": strip_private(candidate),
        "readiness": readiness,
        "history": history[-HISTORY_KEEP:],
        "method": {
            "stage1": "isotonic calibration of the hand-weighted ensemble",
            "stage2": "non-negative L2 logistic regression on per-model thunderstorm fields",
            "gate": "held-out latest 25% of runs; never worse on Brier or log-loss, and Brier -1% or log-loss -5%",
            "dailyUpdate": "after publish, a day's model is replaced only if no worse than the live model on fresh runs",
            "gridKm": 9,
        },
    }

    # Daily update of the published AI HOCO model
    ai_ref = db.collection(hoco_ml.ML_CONFIG_COLLECTION).document(hoco_ml.AI_DOC)
    ai_snap = ai_ref.get()
    ai_doc = ai_snap.to_dict() if ai_snap.exists else None
    ai_update = None
    if hoco_ml.ai_is_live(ai_doc):
        model, changed, log = daily_update(ai_doc, rows, candidate, version, trained_at)
        ai_update = {"model": strip_private(model), "updateLog": log, "lastCheckedAt": trained_at}
        if changed:
            ai_update.update({"version": version, "lastUpdatedAt": trained_at})
        print(f"[AI] Daily update: {'model updated' if changed else 'live model kept'} -> "
              f"{[(e['bucket'], e['action']) for e in log[-1]['entries']]}")
    else:
        print(f"[AI] AI HOCO not published (status: {(ai_doc or {}).get('status', 'training')}). "
              f"Ready to publish: {readiness['ready']}")

    if args.dry_run:
        print(json.dumps({"readiness": readiness, "candidate": {b: {k: v for k, v in c.items() if k != 'params'} for b, c in strip_private(candidate).items()}}, indent=2, default=str)[:4000])
        return
    from firebase_admin import firestore
    db.collection(hoco_ml.ML_CONFIG_COLLECTION).document(hoco_ml.ML_CONFIG_DOC).set(doc)
    if ai_update:
        ai_ref.update(ai_update)  # top-level replace, so dropped models do not linger
    CTX.save()
    print(f"[PUBLISH] system_stats/hoco_ml updated (version {version}).")


if __name__ == "__main__":
    sys.exit(main())

