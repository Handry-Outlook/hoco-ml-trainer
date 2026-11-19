"""
hoco_context.py — weather "context" for AI HOCO.

The basic AI HOCO only learns how much to trust each model. This module adds the
things a forecaster would weigh up from experience:

  * WHERE storms tend / tend not to happen      -> lightning climatology map per grid cell
  * the SETUP (low-level flow sector x season)  -> climatology map for that setup, e.g.
        "cold-season westerly": storms favour exposed west coasts / windward high ground and
        are rare inland. This is how orographic and coastal effects get in without needing
        a terrain model: they show up in where lightning actually happened in each setup.
  * SEASON and instability (CAPE)               -> lets trust in the models change with the
        time of year, e.g. learning that winter westerly signals are over-forecast
  * STORM MOTION (500 hPa steering flow)        -> a copy of the model signal moved
        downstream, so AI HOCO can learn if the models tend to be slow with storms

Upper-air data comes from Open-Meteo (free, no key; non-commercial licence):
  live runs   -> api.open-meteo.com/v1/forecast                (forecast at issue time)
  training    -> historical-forecast-api.open-meteo.com/v1/forecast (archived forecasts, 2022+)
Every lookup is best-effort: if it fails, context features fall back to neutral values
and AI HOCO behaves like the weights-only model.
"""

from __future__ import annotations

import io
import json
import math
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import requests

import hoco_ml

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
HOURLY_VARS = ["wind_speed_850hPa", "wind_direction_850hPa", "wind_speed_500hPa", "wind_direction_500hPa", "cape"]

SETUPS = ["northerly", "easterly", "southerly", "westerly", "slack"]
SEASONS = ["cold", "warm"]  # cold = Oct-Mar, warm = Apr-Sep
SLACK_MS = 5.0
DOWNSTREAM_HOURS = 1.0
SETUP_SHRINK_DAYS = 30.0    # setup maps lean on the overall climatology until a setup has ~30 days
CLIM_SMOOTH_CELLS = 1.5

# Anchor points used to describe the region's flow (averaged)
ANCHORS = {
    "England": [(50.7, -3.5), (52.5, -1.5), (54.3, -2.0), (51.5, 0.5)],
    "Scotland": [(55.9, -4.2), (57.3, -4.3), (56.5, -2.8)],
    "UK": [(50.7, -3.5), (52.5, -1.5), (54.6, -6.0), (56.5, -4.0)],
    "Wales": [(52.3, -3.7)],
    "Ireland and Northern Ireland": [(53.4, -7.7), (54.6, -6.5)],
}

CONTEXT_COLS = ["f_clim", "f_setup", "cos_doy", "sin_doy", "cape_n", "s_cos", "s_setup", "s_cape",
                "r_near", "r_down", "r_dom", "s_rdom", "k_cons", "k_trend", "sc", "s_sc", "sc_ok"]
CONTEXT_LABELS = {
    "f_clim": "Where lightning usually happens",
    "f_setup": "Where lightning happens in this setup",
    "cos_doy": "Time of year (winter vs summer)",
    "sin_doy": "Time of year (spring vs autumn)",
    "cape_n": "Instability (CAPE)",
    "s_cos": "Trust in models by season",
    "s_setup": "Trust in models by setup & place",
    "s_cape": "Trust in models by instability",
    "r_near": "Lightning nearby just before the run",
    "r_down": "Recent lightning moving this way",
    "r_dom": "Storms already active in the region",
    "s_rdom": "Trust in models when storms are active",
    "k_cons": "Model runs agreeing run after run",
    "k_trend": "Newest model runs trending up (vs older runs)",
    "sc": "Severe-storm ingredients (supercell composite)",
    "s_sc": "Trust in models where severe ingredients are present",
    "sc_ok": "Supercell charts available for this run",
}

# Recent lightning: what the strike feed showed just before the run was issued. Traditional HOCO never sees this.
RECENT_HOURS = 3.0           # strikes in the last 3 h before issue ...
RECENT_KM = 30.0             # ... within 30 km of the cell
RECENT_DOWN_HOURS = 3.0      # the same strikes moved along the 500 hPa steering flow for 3 h
RECENT_DOMAIN_HOURS = 6.0    # any lightning in the region in the last 6 h
RECENT_NEAR_SAT = 50.0       # strikes for a full-strength local signal
RECENT_DOMAIN_SAT = 2000.0   # strikes for a full-strength "storms are active" signal
METOFFICE_LIGHTNING_URL = "https://data.consumer-digital.api.metoffice.gov.uk/v1/lightning"


# ---------------------------------------------------------------------------
# Upper-air context
# ---------------------------------------------------------------------------
def anchors_for(region: str):
    if region in ANCHORS:
        return ANCHORS[region]
    b = hoco_ml.region_bounds(region)
    return [((b["top"] + b["bottom"]) / 2, (b["left"] + b["right"]) / 2)]


def wind_to_uv(speed, direction_from_deg):
    rad = np.deg2rad(direction_from_deg)
    return -speed * np.sin(rad), -speed * np.cos(rad)


def fetch_hourly(region: str, start: datetime, end: datetime, historical: bool = False, timeout: int = 60):
    """Region-averaged hourly series {iso_hour: (u850, v850, u500, v500, cape)}. Raises on failure."""
    pts = anchors_for(region)
    params = {
        "latitude": ",".join(f"{p[0]:.3f}" for p in pts),
        "longitude": ",".join(f"{p[1]:.3f}" for p in pts),
        "hourly": ",".join(HOURLY_VARS),
        "wind_speed_unit": "ms",
        "timezone": "GMT",
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
    }
    res = requests.get(HISTORICAL_URL if historical else FORECAST_URL, params=params, timeout=timeout)
    res.raise_for_status()
    data = res.json()
    blocks = data if isinstance(data, list) else [data]
    acc = {}
    for blk in blocks:
        hr = blk.get("hourly") or {}
        times = hr.get("time") or []
        cols = [np.array([np.nan if v is None else v for v in hr.get(k, [None] * len(times))], dtype=float) for k in HOURLY_VARS]
        u850, v850 = wind_to_uv(cols[0], cols[1])
        u500, v500 = wind_to_uv(cols[2], cols[3])
        for i, t in enumerate(times):
            acc.setdefault(t, []).append((u850[i], v850[i], u500[i], v500[i], cols[4][i]))
    return {t: tuple(np.nanmean(np.array(v), axis=0).tolist()) for t, v in acc.items()}


def setup_index(u850, v850):
    speed = math.hypot(u850, v850)
    if not np.isfinite(speed) or speed < SLACK_MS:
        return SETUPS.index("slack")
    direction_from = (math.degrees(math.atan2(-u850, -v850)) + 360) % 360
    if direction_from >= 315 or direction_from < 45:
        return SETUPS.index("northerly")
    if direction_from < 135:
        return SETUPS.index("easterly")
    if direction_from < 225:
        return SETUPS.index("southerly")
    return SETUPS.index("westerly")


def season_index(dt: datetime):
    return 0 if dt.month in (10, 11, 12, 1, 2, 3) else 1


def summarize_window(series: dict, start: datetime, end: datetime) -> dict | None:
    vals = []
    t = start.replace(minute=0, second=0, microsecond=0)
    while t <= end:
        v = series.get(t.strftime("%Y-%m-%dT%H:%M"))
        if v is not None:
            vals.append(v)
        t += timedelta(hours=1)
    if not vals:
        return None
    arr = np.array(vals, dtype=float)
    mean = np.nanmean(arr, axis=0)
    cape = float(np.nanpercentile(arr[:, 4], 75)) if np.isfinite(arr[:, 4]).any() else 0.0
    mid = start + (end - start) / 2
    doy = mid.timetuple().tm_yday
    return {
        "u850": float(mean[0]), "v850": float(mean[1]), "u500": float(mean[2]), "v500": float(mean[3]),
        "cape": cape if np.isfinite(cape) else 0.0,
        "setup": setup_index(mean[0], mean[1]), "season": season_index(mid), "doy": doy,
    }


def run_context(region: str, start: datetime, end: datetime, historical: bool = False) -> dict | None:
    try:
        series = fetch_hourly(region, start - timedelta(hours=1), end + timedelta(hours=1), historical=historical)
        return summarize_window(series, start, end)
    except Exception as e:
        print(f"[CONTEXT] Upper-air lookup failed ({e}); using neutral context.")
        return None


def describe(ctx: dict | None) -> str:
    if not ctx:
        return "unknown setup"
    return f"{SEASONS[ctx['season']]}-season {SETUPS[ctx['setup']]}"


# ---------------------------------------------------------------------------
# Climatology maps
# ---------------------------------------------------------------------------
def map_key(region, h, w):
    return f"{region}|{int(h)}|{int(w)}"


def build_climatology(strikes: np.ndarray, daily_setup: dict, region: str, h: int, w: int, cutoff_ms: float | None = None):
    """
    strikes: (N,3) lon, lat, t_ms. daily_setup: {'YYYY-MM-DD': (setup_idx, season_idx)} for every day
    in the archive (lightning or not). Uses only days strictly before cutoff_ms.
    Returns {'clim': (gh,gw), 'setup': (len(SETUPS)*len(SEASONS), gh, gw), 'days': [...]} in logit units.
    """
    gh, gw = hoco_ml.grid_shape(h, w)
    days = sorted(d for d in daily_setup
                  if cutoff_ms is None or datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp() * 1000 < cutoff_ms - 86400000)
    n_combo = len(SETUPS) * len(SEASONS)
    counts = np.zeros((gh, gw), np.float32)
    counts_s = np.zeros((n_combo, gh, gw), np.float32)
    ndays_s = np.zeros(n_combo, np.float32)
    if not days:
        return None
    day_index = {d: i for i, d in enumerate(days)}
    st = strikes if cutoff_ms is None else strikes[strikes[:, 2] < cutoff_ms]
    r, c = hoco_ml.lonlat_to_cell(st[:, 0], st[:, 1], region, h, w)
    ok = r >= 0
    dates = np.array([datetime.fromtimestamp(t / 1000, timezone.utc).strftime("%Y-%m-%d") for t in st[ok, 2]])
    rr, cc = r[ok], c[ok]
    for d in days:
        combo = daily_setup[d][0] * len(SEASONS) + daily_setup[d][1]
        ndays_s[combo] += 1
    kernel = hoco_ml.neighbourhood_kernel(region, h, w)   # same "within 20 km" as the training labels
    for d in np.unique(dates):
        if d not in day_index:
            continue
        m = dates == d
        grid = np.zeros((gh, gw), np.uint8)
        grid[rr[m], cc[m]] = 1
        grid = cv2.dilate(grid, kernel).astype(np.float32)  # "within 20 km"
        counts += grid
        combo = daily_setup[d][0] * len(SEASONS) + daily_setup[d][1]
        counts_s[combo] += grid
    n = float(len(days))
    # Outside the lightning coverage "no strikes" means "not observed": blur only over covered cells
    # (normalised convolution, so the edge doesn't drag nearby values down) and keep the maps neutral there.
    cov = hoco_ml.coverage_grid(region, h, w).astype(np.float32)
    cov_blur = np.maximum(cv2.GaussianBlur(cov, (0, 0), CLIM_SMOOTH_CELLS), 1e-3)
    blur = lambda a: cv2.GaussianBlur(a * cov, (0, 0), CLIM_SMOOTH_CELLS) / cov_blur
    inside = cov > 0
    freq = np.clip(blur(counts) / n, 1e-4, 0.6)
    if inside.any():
        freq[~inside] = float(freq[inside].mean())
    logit = lambda p: np.log(p / (1 - p))
    f_all = logit(freq)
    clim = (f_all - (f_all[inside].mean() if inside.any() else f_all.mean())).astype(np.float32)
    setup = np.zeros((n_combo, gh, gw), np.float32)
    for k in range(n_combo):
        shrunk = (blur(counts_s[k]) + SETUP_SHRINK_DAYS * freq) / (ndays_s[k] + SETUP_SHRINK_DAYS)
        setup[k] = logit(np.clip(shrunk, 1e-4, 0.6)) - f_all
        setup[k][~inside] = 0.0
    return {"clim": clim, "setup": setup, "days": int(n), "setupDays": ndays_s.tolist()}


def save_climatology(maps: dict) -> bytes:
    arrays, meta = {}, {}
    for key, m in maps.items():
        if m is None:
            continue
        arrays[f"clim::{key}"] = m["clim"].astype(np.float16)
        arrays[f"setup::{key}"] = m["setup"].astype(np.float16)
        meta[key] = {"days": m["days"], "setupDays": m["setupDays"]}
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def load_climatology(data: bytes) -> dict:
    out = {}
    with np.load(io.BytesIO(data), allow_pickle=False) as npz:
        meta = json.loads(bytes(npz["meta"]).decode())
        for key in meta:
            out[key] = {"clim": npz[f"clim::{key}"].astype(np.float32), "setup": npz[f"setup::{key}"].astype(np.float32), **meta[key]}
    return out


# ---------------------------------------------------------------------------
# Feature construction (shared by training and inference so they always match)
# ---------------------------------------------------------------------------
def model_mean(fields: dict, models: list):
    present = [fields[m] for m in models if m in fields and fields[m] is not None]
    if not present:
        return None
    return sum(np.nan_to_num(np.maximum(f, 0), nan=0.0) for f in present) / len(present)


def shift_downstream(field: np.ndarray, ctx: dict | None, region: str, h: int, w: int, hours: float = DOWNSTREAM_HOURS):
    """Move a (H', W') field along the 500 hPa steering flow. Works on pixel or sample grids."""
    if ctx is None or field is None:
        return field
    b = hoco_ml.region_bounds(region)
    fh, fw = field.shape
    lat0 = math.radians((b["top"] + b["bottom"]) / 2)
    km_per_px_x = (b["right"] - b["left"]) * 111.32 * math.cos(lat0) / fw
    km_per_px_y = (b["top"] - b["bottom"]) * 110.57 / fh
    dist_x = ctx["u500"] * 3.6 * hours / km_per_px_x
    dist_y = -ctx["v500"] * 3.6 * hours / km_per_px_y
    m = np.float32([[1, 0, dist_x], [0, 1, dist_y]])
    return cv2.warpAffine(field.astype(np.float32), m, (fw, fh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def run_scalars(ctx: dict | None):
    if not ctx:
        return 0.0, 0.0, 0.0
    ang = 2 * math.pi * (ctx["doy"] - 15) / 365.25  # cos = +1 in mid-January
    return math.cos(ang), math.sin(ang), math.log1p(max(ctx.get("cape", 0.0), 0.0)) / math.log1p(1000.0)


def clim_lookup(maps: dict | None, region: str, h: int, w: int, ctx: dict | None):
    """(f_clim, f_setup) grids for this region/size and setup, or zeros if unavailable."""
    gh, gw = hoco_ml.grid_shape(h, w)
    m = (maps or {}).get(map_key(region, h, w))
    if m is None:
        return np.zeros((gh, gw), np.float32), np.zeros((gh, gw), np.float32)
    f_setup = np.zeros((gh, gw), np.float32)
    if ctx:
        f_setup = m["setup"][ctx["setup"] * len(SEASONS) + ctx["season"]]
    return m["clim"], f_setup


def context_columns(S, f_clim, f_setup, cos_d, sin_d, cape_n, r_near=0.0, r_down=0.0, r_dom=0.0,
                    k_cons=0.0, k_trend=0.0, sc=0.0, sc_ok=0.0):
    """Columns in CONTEXT_COLS order; inputs are same-shape arrays or scalars."""
    S = np.asarray(S, dtype=np.float64)
    ones = np.ones_like(S)
    return [np.asarray(f_clim) * ones, np.asarray(f_setup) * ones, cos_d * ones, sin_d * ones, cape_n * ones,
            S * cos_d, S * np.asarray(f_setup), S * cape_n,
            np.asarray(r_near) * ones, np.asarray(r_down) * ones, np.asarray(r_dom) * ones, S * np.asarray(r_dom),
            np.asarray(k_cons) * ones, np.asarray(k_trend) * ones,
            np.asarray(sc) * ones, S * np.asarray(sc), np.asarray(sc_ok) * ones]


# ---------------------------------------------------------------------------
# Supercell composite (severe-storm ingredients) from the models that publish it
# ---------------------------------------------------------------------------
SC_PREFIX = "ml/features_sc"     # Storage: one small file per run, {sc_<model code>: peak value on the sample grid}
SC_SAT = 8.0                     # composite value treated as "full strength"
SC_STEP_HOURS = 3                # charts are read every 3 valid hours (the ingredients change slowly)
# Chart colour -> composite value (RGB). Filled from the probe charts; pixels far from every colour = 0.
SC_PALETTE = [                   # [((r, g, b), value)] from the chart legend; each band counts as its lower edge
    ((210, 233, 255), 0.2), ((117, 186, 255), 0.5), ((0, 105, 210), 1.0), ((20, 143, 27), 2.0),
    ((99, 237, 7), 3.0), ((255, 244, 43), 4.0), ((232, 220, 0), 6.0), ((255, 127, 39), 8.0),
    ((247, 30, 84), 10.0), ((136, 0, 0), 15.0), ((100, 0, 127), 20.0), ((194, 0, 251), 25.0),
    ((221, 102, 255), 30.0), ((235, 166, 255), 40.0), ((185, 122, 87), 50.0),
]
SC_TOLERANCE = 18


def decode_supercell(img_bgr):
    """Composite value per pixel from one chart image (BGR, already cropped to the map)."""
    img = np.asarray(img_bgr, dtype=np.int16)
    out = np.zeros(img.shape[:2], np.float32)
    best = np.full(img.shape[:2], SC_TOLERANCE + 1, np.int16)
    for (r, g, b), val in SC_PALETTE:
        d = np.max(np.abs(img - np.array([b, g, r], np.int16)), axis=2)
        hit = d < best
        out[hit] = val
        best[hit] = d[hit]
    out[best > SC_TOLERANCE] = 0.0
    return out


def pool_max(field: np.ndarray) -> np.ndarray:
    """Maximum over each sample-grid cell (CELL_PX x CELL_PX)."""
    f = np.asarray(field, dtype=np.float32)
    h, w = f.shape
    gh, gw = hoco_ml.grid_shape(h, w)
    c = hoco_ml.CELL_PX
    pad = np.zeros((gh * c, gw * c), np.float32)
    pad[:h, :w] = f
    return pad.reshape(gh, c, gw, c).max(axis=(1, 3))


def build_sc_npz(sc_grids: dict, meta: dict) -> bytes:
    arrays = {f"sc_{m}": np.asarray(g, np.float32).astype(np.float16) for m, g in sc_grids.items() if g is not None}
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def read_sc_npz(data: bytes):
    with np.load(io.BytesIO(data), allow_pickle=False) as npz:
        meta = json.loads(bytes(npz["meta"]).decode("utf-8"))
        grids = {k[3:]: npz[k].astype(np.float32) for k in npz.files if k.startswith("sc_")}
    return meta, grids


def supercell_input(sc_grids: dict | None, region: str, h: int, w: int):
    """{'sc': (gh, gw) 0..1 strongest composite within 20 km over the available models, 'ok': 1} or None."""
    grids = [np.nan_to_num(np.asarray(g, np.float32)) for g in (sc_grids or {}).values() if g is not None]
    if not grids:
        return None
    gh, gw = hoco_ml.grid_shape(h, w)
    m = np.zeros((gh, gw), np.float32)
    for g in grids:
        g = g[:gh, :gw]
        m[:g.shape[0], :g.shape[1]] = np.maximum(m[:g.shape[0], :g.shape[1]], g)
    m = cv2.dilate(m, hoco_ml.neighbourhood_kernel(region, h, w, hoco_ml.NEIGHBOURHOOD_KM))
    return {"sc": np.minimum(np.log1p(np.maximum(m, 0)) / math.log1p(SC_SAT), 1.0).astype(np.float32), "ok": 1.0}


# ---------------------------------------------------------------------------
# Run-to-run behaviour of each model (from the per-run 'raw blocks' saved with every run since schema 2)
# ---------------------------------------------------------------------------
TREND_SIGNAL = 1e-3   # a model run "shows storms" in a cell when its pooled raw signal is above this


def run_trend(blocks, region: str, h: int, w: int):
    """blocks: [(model, rank, grid)] with grid = hoco_ml.pool_grid(raw field) summed over one 6-h block
    (rank 0 = the model's newest run). Returns {'cons', 'trend'}: (gh, gw) grids, or None if nothing usable.
      cons  = share of all (model, run) pairs showing storms within 20 km  -> agreement run after run
      trend = (share of models whose NEWEST run shows storms) - (share whose OLDER runs did), over models
              with both, within 20 km -> +1 newly appeared, -1 dropped by the latest runs."""
    if not blocks:
        return None
    gh, gw = hoco_ml.grid_shape(h, w)
    per = {}   # (model, rank) -> max over blocks
    for model, rank, g in blocks:
        if g is None:
            continue
        g = np.nan_to_num(np.asarray(g, dtype=np.float32)[:gh, :gw])
        if g.shape != (gh, gw):
            pad = np.zeros((gh, gw), np.float32)
            pad[:g.shape[0], :g.shape[1]] = g
            g = pad
        key = (model, int(rank))
        per[key] = np.maximum(per[key], g) if key in per else g
    if not per:
        return None
    k = hoco_ml.neighbourhood_kernel(region, h, w, hoco_ml.NEIGHBOURHOOD_KM)
    shows = {key: cv2.dilate((g > TREND_SIGNAL).astype(np.uint8), k).astype(np.float32) for key, g in per.items()}
    cons = sum(shows.values()) / len(shows)
    models = {m for m, _ in shows}
    up = np.zeros((gh, gw), np.float32)
    n_both = 0
    for m in models:
        ranks = sorted(r for mm, r in shows if mm == m)
        if len(ranks) < 2 or ranks[0] != 0:
            continue
        older = np.mean([shows[(m, r)] for r in ranks[1:]], axis=0)
        up += shows[(m, 0)] - older
        n_both += 1
    trend = up / n_both if n_both else np.zeros((gh, gw), np.float32)
    return {"cons": cons.astype(np.float32), "trend": trend.astype(np.float32)}


def blocks_from_npz(meta: dict, arrays: dict):
    """[(model, rank, grid)] from a feature file (empty for files saved before schema 2)."""
    out = []
    for b in meta.get("raw_blocks") or []:
        g = arrays.get(b.get("key"))
        if g is not None and b.get("model") in hoco_ml.LIGHTNING_MODELS:
            out.append((b["model"], int(b.get("rank", 0)), g))
    return out


def blocks_from_live(raw_blocks):
    """Same, from cloud.py's in-memory raw blocks (full-size fields, real model names)."""
    out = []
    for b in raw_blocks or []:
        m = hoco_ml.canonical_model(b.get("model"))
        if m in hoco_ml.LIGHTNING_MODELS and b.get("field") is not None:
            out.append((m, int(b.get("rank", 0)), hoco_ml.pool_grid(b["field"])))
    return out


# ---------------------------------------------------------------------------
# Recent lightning (observed just before the run was issued)
# ---------------------------------------------------------------------------
def recent_lightning(strikes, issue_ms, region: str, h: int, w: int, ctx: dict | None = None):
    """Strikes just before issue, on the (gh, gw) sample grid of an h x w chart.
    strikes: array (lon, lat, t_ms) sorted by time. Returns {'near', 'down': grids 0..1, 'dom': 0..1} or None."""
    if strikes is None or issue_ms is None:
        return None
    strikes = np.asarray(strikes, dtype=np.float64).reshape(-1, 3)
    gh, gw = hoco_ml.grid_shape(h, w)
    t = strikes[:, 2]

    def window(hours):
        lo = np.searchsorted(t, issue_ms - hours * 3600e3, side="left")
        hi = np.searchsorted(t, issue_ms, side="left")       # strictly before issue: no peeking
        return strikes[lo:hi]

    def cells(s):
        if not len(s):
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        r, c = hoco_ml.lonlat_to_cell(s[:, 0], s[:, 1], region, h, w)
        ok = r >= 0
        return r[ok], c[ok]

    counts = np.zeros((gh, gw), np.float32)
    r, c = cells(window(RECENT_HOURS))
    np.add.at(counts, (r, c), 1.0)
    k = hoco_ml.neighbourhood_kernel(region, h, w, RECENT_KM).astype(np.float32)

    def squash(n, sat):
        return np.minimum(np.log1p(np.maximum(n, 0)) / math.log1p(sat), 1.0).astype(np.float32)

    near = squash(cv2.filter2D(counts, -1, k, borderType=cv2.BORDER_CONSTANT), RECENT_NEAR_SAT)
    moved = shift_downstream(counts, ctx, region, h, w, RECENT_DOWN_HOURS)
    down = squash(cv2.filter2D(np.ascontiguousarray(moved), -1, k, borderType=cv2.BORDER_CONSTANT), RECENT_NEAR_SAT)
    n_dom = len(cells(window(RECENT_DOMAIN_HOURS))[0])
    return {"near": near, "down": down, "dom": float(squash(n_dom, RECENT_DOMAIN_SAT))}


def recent_signal(recent, shape):
    """max(near, down) resized to `shape` (for the safeguards), or None."""
    if not recent:
        return None
    g = np.maximum(recent["near"], recent["down"])
    if g.shape != tuple(shape):
        g = cv2.resize(g, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return g


def _strike_ms(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) * 1000.0 if value < 1e10 else float(value)
    s = str(value).strip()
    if s.isdigit() and 9 <= len(s) <= 13:
        n = float(s)
        return n * 1000.0 if n < 1e10 else n
    if len(s) >= 10 and s[2] == "/" and s[5] == "/":        # dd/mm/yyyy[ hh:mm[:ss]]
        hms = s[10:].strip(" T") or "00:00:00"
        s = f"{s[6:10]}-{s[3:5]}-{s[0:2]}T{hms if len(hms) > 5 else hms + ':00'}Z"
    elif len(s) >= 15 and s[8] == "T" and s[:8].isdigit():  # yyyymmddThhmmss
        s = f"{s[0:4]}-{s[4:6]}-{s[6:8]}T{s[9:11]}:{s[11:13]}:{s[13:15]}Z"
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp() * 1000.0


def fetch_recent_strikes(timeout: int = 60):
    """The Met Office live feed (about the last 48 h) as an array (lon, lat, t_ms) sorted by time; None on failure."""
    def get(url):
        res = requests.get(url, timeout=timeout, headers={"Accept": "application/json"})
        res.raise_for_status()
        return res.json()

    def items(d):
        if isinstance(d, list):
            return d
        return d.get("lightning_strikes") or [] if isinstance(d, dict) else []

    try:
        base = get(METOFFICE_LIGHTNING_URL)
    except Exception as e:
        print(f"[RECENT] Met Office lightning feed failed: {e}")
        return None
    raw = list(items(base))
    for c in ((base.get("chunks") or []) if isinstance(base, dict) else []):
        n = c.get("chunk") if isinstance(c, dict) else c
        if n is None:
            continue
        try:
            raw += items(get(f"{METOFFICE_LIGHTNING_URL}?last-minutes=1&chunk={n}"))
        except Exception as e:
            print(f"[RECENT] chunk {n} failed: {e}")
    out = []
    for s in raw:
        if not isinstance(s, dict):
            continue
        c = s.get("coordinates") or (s.get("geometry") or {}).get("coordinates")
        t = _strike_ms(s.get("strike_time"))
        if t is None or not isinstance(c, list) or len(c) < 2:
            continue
        try:
            lon, lat = float(c[0]), float(c[1])
        except (TypeError, ValueError):
            continue
        if abs(lon) > 30 and abs(lat) <= 30:
            lon, lat = lat, lon
        out.append((lon, lat, t))
    arr = np.array(out, dtype=np.float64).reshape(-1, 3)
    arr = arr[np.argsort(arr[:, 2])]
    print(f"[RECENT] {len(arr)} strikes from the Met Office live feed")
    return arr


def apply_context_model(smoothed_fields: dict, params: dict, ctx: dict | None, maps: dict | None, region: str, shape,
                        x_ens=None, grid: bool = False, src_hw=None, recent: dict | None = None,
                        trend: dict | None = None, supercell: dict | None = None):
    """Probability from a 'context' model (pixel level; grid=True for sample-grid previews).
    recent: recent_lightning(...) for this run (None = no strikes seen, i.e. all zero)."""
    h, w = shape
    sh, sw = src_hw or shape
    smoothed_fields = hoco_ml.prepare_inputs(smoothed_fields, x_ens, params, region, sh, sw, grid)
    models = params.get("models") or [m for m in (params.get("weights") or {}) if m != hoco_ml.ENS_INPUT]
    z = np.full(shape, float(params.get("intercept", -4.0)), dtype=np.float64)
    for m, wt in (params.get("weights") or {}).items():
        f = smoothed_fields.get(m)
        if f is not None and wt:
            z += float(wt) * np.maximum(f, 0)
    S = model_mean(smoothed_fields, models)
    if S is None:
        S = np.zeros(shape, np.float32)
    z += float(params.get("downstream", 0.0)) * shift_downstream(S, ctx, region, h, w)
    f_clim, f_setup = clim_lookup(maps, region, h, w, ctx)
    f_clim = cv2.resize(f_clim, (w, h), interpolation=cv2.INTER_LINEAR)
    f_setup = cv2.resize(f_setup, (w, h), interpolation=cv2.INTER_LINEAR)
    cos_d, sin_d, cape_n = run_scalars(ctx)
    r_near = r_down = 0.0
    r_dom = 0.0
    if recent:
        r_near = cv2.resize(recent["near"], (w, h), interpolation=cv2.INTER_LINEAR)
        r_down = cv2.resize(recent["down"], (w, h), interpolation=cv2.INTER_LINEAR)
        r_dom = float(recent["dom"])
    k_cons = k_trend = 0.0
    if trend:
        k_cons = cv2.resize(trend["cons"], (w, h), interpolation=cv2.INTER_LINEAR)
        k_trend = cv2.resize(trend["trend"], (w, h), interpolation=cv2.INTER_LINEAR)
    sc, sc_ok = 0.0, 0.0
    if supercell:
        sc = cv2.resize(supercell["sc"], (w, h), interpolation=cv2.INTER_LINEAR)
        sc_ok = 1.0
    cols = context_columns(S, f_clim, f_setup, cos_d, sin_d, cape_n, r_near, r_down, r_dom, k_cons, k_trend, sc, sc_ok)
    coefs = params.get("context") or {}
    for name, col in zip(CONTEXT_COLS, cols):
        z += float(coefs.get(name, 0.0)) * col
    return hoco_ml.sigmoid(z).astype(np.float32)
