"""
Sunbears Sports Analytics Dashboard (Dash)

- Loads hockey tracking from ONE CSV (players_tracking.csv) with team_id
  • Required columns: timestamp (or timeframe), player_id, x, y, team_id
  • Optional columns: vx, vy
  • team_id convention: 0 = Defense, nonzero = Offense
- Top row: Digital Tracking (Plotly) + Video in proportional, rounded cards
- Unified bottom "suite" (single rounded container):
  • Centered toolbar: [Mode selector] | Prev / Play / Next / Speed / Loop + frame readout (right)
  • Single scrubber slider
  • Editor controls (team filter + overlay chips + Reset)
- Editor Mode (ephemeral edits):
  • Playback auto-pauses, Play button disabled
  • Drag players by moving the circle (move-only; fixed radius)
  • Voronoi & Pitch Control recompute instantly from edited positions (current frame only)
  • Coverage Control (beta) recomputes instantly from edited positions (current frame only)
  • Changing frame OR switching modes clears edits; returning to a frame shows original data

Sync rules in this build:
- Play/Pause controls BOTH tracking & video
- Speed (0.5×, 1×, 2×) applied to BOTH
- Loop ON: both restart
- Loop OFF: shorter stops first; longer continues; Play button shows "Pause" until BOTH stopped
- Soft sync (no seeking)
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import math
import numpy as np
import dash
from dash import html, dcc, Dash
from dash.dependencies import Input, Output, State
import dash.exceptions
import pandas as pd
import plotly.graph_objs as go

from utils import compute_voronoi  # Voronoi + clipping


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

# Files
# TRACKING_CSV = "assets/players_tracking.csv"   # merged input with team_id (0=Defense, else Offense)
TRACKING_CSV = "assets/clip_scene/scene_5.csv"   # merged input with team_id (0=Defense, else Offense)
# VIDEO_FILENAME = "sample_video.mp4"            # place under ./assets/ (optional)
VIDEO_FILENAME = "clip_scene/scene_5.mp4"            # place under ./assets/ (optional)
FIELD_IMAGE = "field_hockey.png"               # rink image under ./assets/
ICON_IMAGE = "sunbears_icon.webp"              # header icon under ./assets/

# Rink bounds (match your data)
RINK_BOUNDS: Dict[str, float] = {"x_min": 0.0, "x_max": 61.0, "y_min": 0.0, "y_max": 30.0}

# Default trail length used if "Trails" overlay is enabled
TRAIL_DEFAULT = 40

# Player circle radius for Editor Mode (rink-units)
PLAYER_RADIUS = 0.7

# -------- Velocity arrows (display units are rink units) --------
VELOCITY_MIN_MAG = 0.01
VELOCITY_SCALE   = 2.0
VELOCITY_MAX_LEN = 4.0
VELOCITY_LINE_W  = 2
# Quiver arrowhead geometry
VELOCITY_HEAD_FRAC = 0.35           # fraction of shaft length
VELOCITY_HEAD_MAX  = 1.2            # cap absolute head length
VELOCITY_HEAD_DEG  = 28.0           # head opening half-angle (deg)

# -------- Pitch Control (beta) parameters --------
PC_GRID_W = 240
PC_GRID_H = 120
PC_TAU_REACT = 0.40   # s
PC_TAU_ACCEL = 0.70   # s of "speed credit"
# PC_LAMBDA = 1.6       # time decay -> sharper vs softer fields
PC_LAMBDA = 3.2       # time decay -> sharper vs softer fields
PC_OPACITY = 0.50     # heatmap opacity
PC_VMAX_FALLBACK = 5.0
PC_VMAX: float = PC_VMAX_FALLBACK
# Cached per timestamp (for playback). Editor recomputes live from edited positions.
_PC_CACHE: Dict[int, np.ndarray] = {}

# -------- Coverage Control (beta) styling/params --------
# Simple “mark the nearest attacker, stand between target & goal with a small standoff”
COV_STANDOFF = 3.0            # meters toward our goal from anticipated attacker point
COV_VEL_GAIN = 0.8            # anticipation = pos + gain * vel
COV_EMA_ALPHA = 0.35          # smoothing for playback cache
COV_MAX_STEP = 1.0            # step cap per frame in the EMA path
COV_GHOST_SIZE = 16
COV_GHOST_COLOR = "rgba(46,134,222,0.42)"  # semi-transparent defense blue
COV_CONNECTOR_COLOR = "rgba(46,134,222,0.7)"
COV_CONNECTOR_WIDTH = 2
# Cached suggested positions per timestamp for smooth playback
_COV_CACHE: Dict[int, pd.DataFrame] = {}

STYLES: Dict[str, Any] = {
    "page": {"background": "#f6f7fb", "fontFamily": "Inter, Segoe UI, Arial, sans-serif"},
}

COLOR_MAP = {"Offense": "#e74c3c", "Defense": "#2e86de"}

# Voronoi fill colors (light, slightly transparent)
VORONOI_FILL = {
    "Defense": "rgba(223,233,249,0.55)",  # #dfe9f9 @ 55%
    "Offense": "rgba(248,223,220,0.55)",  # #f8dfdc @ 55%
}

# Heatmap colorscale (Defense→Offense)
PC_COLORSCALE = [
    [0.00, "rgb(46,134,222)"],   # Defense blue
    [0.50, "rgb(245,246,250)"],  # near white
    [1.00, "rgb(231,76,60)"],    # Offense red
]

# Shared timer base at 1× speed (ms)
TARGET_FPS = 28
BASE_INTERVAL_MS = 1000.0 / TARGET_FPS


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------

def load_tracking_data_single(path: str) -> pd.DataFrame:
    """
    Read merged CSV, normalize columns, derive 'team' from team_id (0=Defense, else Offense).
    Accepts:
      - 'timestamp' or 'timeframe' (cast to int)
      - 'player_id', 'x', 'y' (required)
      - 'vx', 'vy' (optional -> default to 0.0)
      - 'team_id' (required -> 0=Defense, else Offense)
    Returns canonical columns:
      ['timestamp','player_id','team','x','y','vx','vy']
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Rename timeframe -> timestamp
    col_map: Dict[str, str] = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "timeframe":
            col_map[c] = "timestamp"
        elif lc in {"timestamp", "player_id", "x", "y", "vx", "vy", "team_id", "team"}:
            col_map[c] = lc
    if col_map:
        df.rename(columns=col_map, inplace=True)

    required = {"timestamp", "player_id", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in tracking data: {missing}. Found: {list(df.columns)}")

    # Velocity optional
    if "vx" not in df.columns:
        df["vx"] = 0.0
    if "vy" not in df.columns:
        df["vy"] = 0.0

    # team_id required (primary) or fallback to 'team' textual
    if "team_id" not in df.columns:
        if "team" not in df.columns:
            raise ValueError("Merged tracking file must include 'team_id' (0=Defense, else Offense) or a 'team' label.")
        # Fallback: textual 'team' -> 0/1 best-effort
        t_lower = df["team"].astype(str).str.strip().str.lower()
        df["team_id"] = np.where(t_lower.eq("defense") | t_lower.eq("defensive") | t_lower.eq("def"), 0, 1)

    # Types
    df["timestamp"] = df["timestamp"].astype(int)
    df["player_id"] = df["player_id"].astype(str)
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)
    df["vx"] = df["vx"].astype(float)
    df["vy"] = df["vy"].astype(float)

    # Normalize team from team_id
    def _to_int_safe(v):
        try:
            return int(float(v))
        except Exception:
            return 1  # treat unknown as Offense
    team_id_int = df["team_id"].apply(_to_int_safe)
    df["team"] = np.where(team_id_int == 0, "Defense", "Offense")

    out = df[["timestamp", "player_id", "team", "x", "y", "vx", "vy"]].copy()
    out.sort_values(["timestamp", "team", "player_id"], kind="mergesort", inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def _clamp_df(df_: Optional[pd.DataFrame], bounds: Dict[str, float]) -> Optional[pd.DataFrame]:
    if df_ is None or df_.empty:
        return df_
    df_ = df_.copy()
    df_["x"] = df_["x"].clip(bounds["x_min"], bounds["x_max"])
    df_["y"] = df_["y"].clip(bounds["y_min"], bounds["y_max"])
    return df_


def make_trails(df: pd.DataFrame, current_ts: int, trail_len: int) -> pd.DataFrame:
    start_ts = max(df["timestamp"].min(), current_ts - trail_len)
    return df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= current_ts)].copy()


def _aspect_padding_from_bounds(bounds: Dict[str, float]) -> str:
    w = bounds["x_max"] - bounds["x_min"]
    h = bounds["y_max"] - bounds["y_min"]
    pct = (h / w) * 100.0 if w > 0 else 56.25
    return f"{pct:.3f}%"


def _apply_edits_to_frame(df_frame: pd.DataFrame, edits_for_ts: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Apply per-frame edited positions to a single-frame dataframe."""
    if not edits_for_ts:
        return df_frame
    df_frame = df_frame.copy()
    df_frame["__key"] = df_frame["team"].astype(str) + "|" + df_frame["player_id"].astype(str)
    keys = set(df_frame["__key"])
    for key, info in edits_for_ts.items():
        if key in keys and "x" in info and "y" in info:
            df_frame.loc[df_frame["__key"] == key, ["x", "y"]] = [info["x"], info["y"]]
    df_frame.drop(columns="__key", inplace=True)
    return df_frame


def _apply_edits_to_trails(trails_df: pd.DataFrame, edits_store: Dict[str, Any]) -> pd.DataFrame:
    """Optionally reflect edited positions on the recent trail segment for visual continuity."""
    if not isinstance(edits_store, dict) or trails_df.empty:
        return trails_df
    trails_df = trails_df.copy()
    trails_df["__key"] = trails_df["team"].astype(str) + "|" + trails_df["player_id"].astype(str)
    edited_ts = set(edits_store.keys())
    if not edited_ts:
        trails_df.drop(columns="__key", inplace=True)
        return trails_df
    mask = trails_df["timestamp"].astype(str).isin(edited_ts)
    idx = trails_df.index[mask]
    for i in idx:
        ts_key = str(trails_df.at[i, "timestamp"])
        pkey = trails_df.at[i, "__key"]
        if pkey in edits_store.get(ts_key, {}):
            info = edits_store[ts_key][pkey]
            trails_df.at[i, "x"] = float(info["x"])
            trails_df.at[i, "y"] = float(info["y"])
    trails_df.drop(columns="__key", inplace=True)
    return trails_df


# ---------- Velocity as NON-DRAGGABLE quiver line traces ----------
def _velocity_quiver_traces(df_frame: pd.DataFrame) -> List[go.Scatter]:
    """Return two line traces (Offense/Defense) that draw arrows as polylines."""
    traces: List[go.Scatter] = []
    head_ang = math.radians(VELOCITY_HEAD_DEG)

    for team in ["Offense", "Defense"]:
        sub = df_frame[df_frame["team"] == team]
        if sub.empty:
            continue

        xs: List[float] = []
        ys: List[float] = []
        for _, row in sub.iterrows():
            vx = float(row.get("vx", 0.0))
            vy = float(row.get("vy", 0.0))
            mag = math.hypot(vx, vy)
            if mag < VELOCITY_MIN_MAG:
                continue

            L = min(VELOCITY_MAX_LEN, mag * VELOCITY_SCALE)
            if L <= 0:
                continue

            dx = (vx / (mag + 1e-9)) * L
            dy = (vy / (mag + 1e-9)) * L
            x0 = float(row["x"])
            y0 = float(row["y"])
            x1 = x0 + dx
            y1 = y0 + dy

            # Shaft
            xs += [x0, x1, None]
            ys += [y0, y1, None]

            # Arrowhead (two short segments)
            head_len = min(L * VELOCITY_HEAD_FRAC, VELOCITY_HEAD_MAX)
            theta = math.atan2(dy, dx)
            left = theta + head_ang
            right = theta - head_ang

            xs += [x1, x1 - head_len * math.cos(left), None]
            ys += [y1, y1 - head_len * math.sin(left), None]
            xs += [x1, x1 - head_len * math.cos(right), None]
            ys += [y1, y1 - head_len * math.sin(right), None]

        if xs:
            traces.append(
                go.Scattergl(
                    x=xs, y=ys, mode="lines",
                    line=dict(width=VELOCITY_LINE_W, color=COLOR_MAP.get(team, "#333")),
                    hoverinfo="skip", showlegend=False,
                )
            )
    return traces


# --------------------------------------------------------------------------------------
# Pitch Control (beta): fast symmetric model
# --------------------------------------------------------------------------------------

def _auto_calibrate_vmax(df_all: pd.DataFrame) -> float:
    try:
        sp = np.sqrt(np.square(df_all["vx"].to_numpy()) + np.square(df_all["vy"].to_numpy()))
        vmax = float(np.nanpercentile(sp, 95))
        if not np.isfinite(vmax) or vmax <= 0:
            return PC_VMAX_FALLBACK
        return max(PC_VMAX_FALLBACK * 0.5, min(PC_VMAX_FALLBACK * 4.0, vmax))
    except Exception:
        return PC_VMAX_FALLBACK


def _pitch_control_probs(
    df_frame: pd.DataFrame,
    bounds: Dict[str, float],
    grid_w: int = PC_GRID_W,
    grid_h: int = PC_GRID_H,
    vmax: float = PC_VMAX
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (x_grid, y_grid, Z_offense) where Z_offense in [0,1] with shape (grid_h, grid_w)."""
    if df_frame is None or len(df_frame) < 2:
        gx = np.linspace(bounds["x_min"], bounds["x_max"], grid_w)
        gy = np.linspace(bounds["y_min"], bounds["y_max"], grid_h)
        return gx, gy, np.zeros((grid_h, grid_w), dtype=float)

    gx = np.linspace(bounds["x_min"], bounds["x_max"], grid_w)
    gy = np.linspace(bounds["y_min"], bounds["y_max"], grid_h)
    Gx, Gy = np.meshgrid(gx, gy)
    pts = np.stack([Gx.ravel(), Gy.ravel()], axis=1)

    xs = df_frame["x"].to_numpy(dtype=float)
    ys = df_frame["y"].to_numpy(dtype=float)
    vxs = df_frame["vx"].to_numpy(dtype=float)
    vys = df_frame["vy"].to_numpy(dtype=float)
    teams = df_frame["team"].astype(str).to_numpy()
    is_off = (teams == "Offense")
    is_def = (teams == "Defense")

    if is_off.sum() == 0 or is_def.sum() == 0:
        Z = np.ones((grid_h, grid_w), dtype=float) if is_off.sum() > 0 else np.zeros((grid_h, grid_w), dtype=float)
        return gx, gy, Z

    speeds = np.sqrt(vxs * vxs + vys * vys)

    dx = xs[:, None] - pts[None, :, 0]
    dy = ys[:, None] - pts[None, :, 1]
    d = np.sqrt(dx * dx + dy * dy) + 1e-9

    t = PC_TAU_REACT + np.maximum(0.0, d - speeds[:, None] * PC_TAU_ACCEL) / max(vmax, 1e-3)
    r = np.exp(-PC_LAMBDA * t)
    R_off = r[is_off].sum(axis=0)
    R_def = r[is_def].sum(axis=0)

    Z_off = (R_off / (R_off + R_def + 1e-12)).reshape((grid_h, grid_w))
    return gx, gy, Z_off


def _pc_cached_for_timestamp(ts: int, df_all: pd.DataFrame, bounds: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cache pitch control per timestamp for speed (used in playback mode)."""
    if ts in _PC_CACHE:
        Z = _PC_CACHE[ts]
        gx = np.linspace(bounds["x_min"], bounds["x_max"], PC_GRID_W)
        gy = np.linspace(bounds["y_min"], bounds["y_max"], PC_GRID_H)
        return gx, gy, Z

    df_frame = df_all[df_all["timestamp"] == ts]
    gx, gy, Z = _pitch_control_probs(df_frame, bounds, PC_GRID_W, PC_GRID_H, PC_VMAX)
    _PC_CACHE[ts] = Z
    return gx, gy, Z


# --------------------------------------------------------------------------------------
# Coverage Control (beta): simple nearest-mark + goal-line standoff
# --------------------------------------------------------------------------------------

def _goal_point(bounds: Dict[str, float]) -> Tuple[float, float]:
    """Assume own goal for Defense on the left side."""
    gx = bounds["x_min"]
    gy = (bounds["y_min"] + bounds["y_max"]) / 2.0
    return gx, gy


def _instant_coverage_for_frame(df_frame: pd.DataFrame, bounds: Dict[str, float]) -> pd.DataFrame:
    """
    Compute instantaneous suggested positions for Defense only, based on:
      - Assign each defender to nearest attacker
      - Anticipate attacker (pos + gain*vel)
      - Place defender on line (goal -> anticipated attacker), with a standoff toward goal
    Returns DataFrame [player_id, team='Defense', sug_x, sug_y].
    """
    if df_frame is None or df_frame.empty:
        return pd.DataFrame(columns=["player_id", "team", "sug_x", "sug_y"])

    gx, gy = _goal_point(bounds)
    defs = df_frame[df_frame["team"] == "Defense"].copy()
    offs = df_frame[df_frame["team"] == "Offense"].copy()

    if defs.empty:
        return pd.DataFrame(columns=["player_id", "team", "sug_x", "sug_y"])

    # No attackers: slight nudge toward goal
    if offs.empty:
        vecx = gx - defs["x"].to_numpy(float)
        vecy = gy - defs["y"].to_numpy(float)
        d = np.sqrt(vecx * vecx + vecy * vecy) + 1e-9
        nudx = vecx / d * 0.5
        nudy = vecy / d * 0.5
        return pd.DataFrame(
            {"player_id": defs["player_id"].astype(str).values, "team": "Defense",
             "sug_x": defs["x"].to_numpy(float) + nudx, "sug_y": defs["y"].to_numpy(float) + nudy}
        )

    D = defs[["x", "y"]].to_numpy(float)
    O = offs[["x", "y"]].to_numpy(float)
    OV = (offs[["vx", "vy"]].to_numpy(float)
          if {"vx", "vy"}.issubset(offs.columns) else np.zeros_like(O))

    # Nearest-offender assignment (fast; not one-to-one optimal)
    def_to_off = []
    for di in range(D.shape[0]):
        d = D[di]
        j = int(np.argmin(np.sqrt(((O - d) ** 2).sum(axis=1))))
        def_to_off.append(j)

    sug_x = np.zeros(D.shape[0], dtype=float)
    sug_y = np.zeros(D.shape[0], dtype=float)
    g = np.array([gx, gy], dtype=float)

    for di in range(D.shape[0]):
        j = def_to_off[di]
        target = O[j] + COV_VEL_GAIN * OV[j]            # anticipate
        g2t = target - g
        n = np.linalg.norm(g2t) + 1e-9
        dir_g2t = g2t / n
        sug = target - dir_g2t * COV_STANDOFF           # toward goal from target
        sug[0] = np.clip(sug[0], bounds["x_min"], bounds["x_max"])
        sug[1] = np.clip(sug[1], bounds["y_min"], bounds["y_max"])
        sug_x[di], sug_y[di] = float(sug[0]), float(sug[1])

    return pd.DataFrame(
        {"player_id": defs["player_id"].astype(str).values, "team": "Defense",
         "sug_x": sug_x, "sug_y": sug_y}
    )


def _precompute_coverage_cache(df_all: pd.DataFrame, timestamps: List[int], bounds: Dict[str, float]) -> Dict[int, pd.DataFrame]:
    """Precompute smoothed suggestions (EMA + capped step) for smooth playback."""
    cache: Dict[int, pd.DataFrame] = {}
    prev: Dict[str, Tuple[float, float]] = {}  # player_id -> (x, y)

    for ts in timestamps:
        frame = df_all[df_all["timestamp"] == ts]
        inst = _instant_coverage_for_frame(frame, bounds)
        if inst.empty:
            cache[ts] = inst
            continue

        sx = inst["sug_x"].to_numpy(float)
        sy = inst["sug_y"].to_numpy(float)
        pids = inst["player_id"].astype(str).tolist()

        for i, pid in enumerate(pids):
            cur = np.array([sx[i], sy[i]], dtype=float)
            if pid in prev:
                pr = np.array(prev[pid], dtype=float)
                blended = pr + COV_EMA_ALPHA * (cur - pr)
                step = blended - pr
                sl = float(np.linalg.norm(step))
                if sl > COV_MAX_STEP:
                    step *= (COV_MAX_STEP / (sl + 1e-9))
                    blended = pr + step
                sx[i], sy[i] = float(blended[0]), float(blended[1])
                prev[pid] = (sx[i], sy[i])
            else:
                prev[pid] = (float(cur[0]), float(cur[1]))

        smoothed = inst.copy()
        smoothed["sug_x"] = sx
        smoothed["sug_y"] = sy
        cache[ts] = smoothed

    return cache


# --------------------------------------------------------------------------------------
# Figure builder
# --------------------------------------------------------------------------------------

def build_tracking_figure(
    df_frame_for_display: pd.DataFrame,           # may be team-filtered
    df_frame_for_pc_full: pd.DataFrame,           # both teams, edited if in editor
    trails_df: Optional[pd.DataFrame],
    bounds: Dict[str, float],
    team_filter: str,
    show_players: bool,
    show_voronoi: bool,
    show_trails: bool,
    show_velocity: bool,
    show_pc: bool,
    show_coverage: bool,
    coverage_df: Optional[pd.DataFrame],
    current_ts: int,
    mode: str,
    edits_enabled: bool,
) -> Tuple[go.Figure, List[str]]:
    """
    Returns (figure, shape_index_map).
    - In Playback mode: players are scatter markers (not draggable), shape_index_map = []
    - In Editor mode with players visible: players are draggable circle shapes; shape_index_map maps index -> "Team|player_id"
    """
    data: List[go.Scatter] = []
    shape_index_map: List[str] = []

    df_frame = _clamp_df(df_frame_for_display, bounds)
    df_pc_full = _clamp_df(df_frame_for_pc_full, bounds)
    trails_df = _clamp_df(trails_df, bounds)

    # team filter (affects players/trails/voronoi/coverage; PC uses df_pc_full with both teams)
    if team_filter != "both":
        keep = "Offense" if team_filter == "offense" else "Defense"
        df_frame = df_frame[df_frame["team"] == keep]
        if trails_df is not None:
            trails_df = trails_df[trails_df["team"] == keep]
        if coverage_df is not None:
            shown_def_pids = set(df_frame[df_frame["team"] == "Defense"]["player_id"].astype(str).tolist())
            coverage_df = coverage_df[coverage_df["player_id"].astype(str).isin(shown_def_pids)]

    # trails
    if show_trails and trails_df is not None and not trails_df.empty:
        for (_, _pid), seg in trails_df.groupby(["team", "player_id"]):
            data.append(
                go.Scatter(
                    x=seg["x"], y=seg["y"], mode="lines",
                    line=dict(width=2), opacity=0.35,
                    hoverinfo="skip", showlegend=False,
                )
            )

    # pitch control heatmap (below everything)
    if show_pc:
        if mode == "editor":
            gx, gy, Z = _pitch_control_probs(df_pc_full, bounds, PC_GRID_W, PC_GRID_H, PC_VMAX)
        else:
            gx, gy, Z = _pc_cached_for_timestamp(current_ts, df, bounds)

        data.append(
            go.Heatmap(
                x=gx, y=gy, z=Z,
                colorscale=PC_COLORSCALE,
                zmin=0.0, zmax=1.0,
                showscale=False,
                hovertemplate="Offense control: %{z:.2f}<extra></extra>",
                opacity=PC_OPACITY,
            )
        )

    # voronoi (team-colored, slightly transparent)
    if show_voronoi and len(df_frame) >= 2:
        positions = df_frame[["x", "y"]].values.tolist()
        pos_teams = df_frame["team"].tolist()
        vor = compute_voronoi(positions, bounds)
        for idx, poly in vor.items():
            if not poly:
                continue
            xs, ys = zip(*poly)
            xs, ys = list(xs) + [xs[0]], list(ys) + [ys[0]]
            team = pos_teams[idx] if idx < len(pos_teams) else "Offense"
            fill = VORONOI_FILL.get(team, VORONOI_FILL["Offense"])
            data.append(
                go.Scatter(
                    x=xs, y=ys, fill="toself", fillcolor=fill,
                    line=dict(color="rgba(0,0,0,0.12)", width=1), hoverinfo="skip", showlegend=False,
                )
            )

    # velocity (as non-draggable line traces, under shapes)
    if show_velocity and not df_frame.empty:
        data.extend(_velocity_quiver_traces(df_frame))

    # coverage control (connectors + ghost defenders)
    if show_coverage and coverage_df is not None and not coverage_df.empty:
        cur_defs = df_frame[df_frame["team"] == "Defense"][["player_id", "x", "y"]].copy()
        cur_defs["player_id"] = cur_defs["player_id"].astype(str)
        cov = coverage_df.merge(cur_defs, on="player_id", how="inner", suffixes=("_sug", "_act"))
        if not cov.empty:
            xs, ys = [], []
            for _, r in cov.iterrows():
                xs += [float(r["x"]), float(r["sug_x"]), None]
                ys += [float(r["y"]), float(r["sug_y"]), None]
            data.append(
                go.Scatter(
                    x=xs, y=ys, mode="lines",
                    line=dict(width=COV_CONNECTOR_WIDTH, color=COV_CONNECTOR_COLOR, dash="dot"),
                    opacity=0.9, hoverinfo="skip", showlegend=False
                )
            )
            data.append(
                go.Scatter(
                    x=cov["sug_x"], y=cov["sug_y"],
                    mode="markers+text",
                    marker=dict(size=COV_GHOST_SIZE, color=COV_GHOST_COLOR, line=dict(width=1, color="white")),
                    text=[f"{pid}′" for pid in cov["player_id"]],
                    textposition="middle center",
                    hoverinfo="skip", showlegend=False,
                )
            )

    # base figure + rink image
    fig = go.Figure(data=data)
    fig.add_layout_image(
        dict(
            source=f"/assets/{FIELD_IMAGE}", xref="x", yref="y",
            x=bounds["x_min"], y=bounds["y_max"],
            sizex=bounds["x_max"] - bounds["x_min"],
            sizey=bounds["y_max"] - bounds["y_min"],
            sizing="stretch", layer="below", opacity=1.0,
        )
    )
    fig.update_layout(
        autosize=True,
        uirevision="static",
        showlegend=False,
        title=None,
        xaxis=dict(range=[bounds["x_min"], bounds["x_max"]], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[bounds["y_min"], bounds["y_max"]], showgrid=False, zeroline=False,
                   visible=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=12, r=12, t=12, b=12),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        hovermode="closest",
    )

    # players + jersey numbers (annotations) + draggable circles only in editor
    if show_players and not df_frame.empty:
        if mode == "editor" and edits_enabled:
            shapes = []
            annotations = []
            for team in ["Offense", "Defense"]:
                sub = df_frame[df_frame["team"] == team].copy()
                if sub.empty:
                    continue
                sub = sub.sort_values(["player_id"], kind="mergesort")
                for _, row in sub.iterrows():
                    x, y = float(row["x"]), float(row["y"])
                    pid = str(row["player_id"])
                    key = f"{team}|{pid}"
                    shape_index_map.append(key)
                    shapes.append(
                        dict(
                            type="circle",
                            xref="x", yref="y",
                            x0=x - PLAYER_RADIUS, x1=x + PLAYER_RADIUS,
                            y0=y - PLAYER_RADIUS, y1=y + PLAYER_RADIUS,
                            fillcolor=COLOR_MAP[team],
                            opacity=0.95,
                            line=dict(color="white", width=1),
                        )
                    )
                    annotations.append(
                        dict(
                            x=x, y=y, xref="x", yref="y",
                            text=pid, showarrow=False,
                            font=dict(color="white", size=12, family="Inter, Arial"),
                            captureevents=False,
                        )
                    )
            fig.update_layout(shapes=shapes, annotations=annotations)
        else:
            for team in ["Offense", "Defense"]:
                sub = df_frame[df_frame["team"] == team]
                if sub.empty:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=sub["x"], y=sub["y"], mode="markers+text",
                        marker=dict(size=12, color=COLOR_MAP[team], line=dict(width=1, color="white")),
                        text=sub["player_id"], textposition="middle center",
                        showlegend=False,
                    )
                )

    return fig, shape_index_map

def build_header() -> html.Div:
    # Header row: icon + texts | spacer | Import button (top-right)
    return html.Div(
        [
            html.Div(
                [
                    html.Img(src=f"/assets/{ICON_IMAGE}", className="sb-header__icon"),
                    html.Div(
                        [
                            html.H2("Sunbears Dashboard", className="sb-header__title"),
                            html.Div("Digital Tracking • Analytics • Playback", className="sb-header__subtitle"),
                        ],
                        className="sb-header__texts",
                    ),
                    html.Div(className="sb-header__spacer"),  # pushes the import button to the right
                    html.Button(
                        "Import Data",
                        id="btn-import",
                        className="sb-btn sb-btn--primary sb-btn--import",
                        title="(Coming soon) Import tracking CSV & video for analysis",
                        n_clicks=0,
                    ),
                ],
                className="sb-header__row",
            )
        ],
        className="sb-header",
    )


def build_top_row(bounds: Dict[str, float]) -> html.Div:
    ar_padding = _aspect_padding_from_bounds(bounds)
    video_path = Path("assets") / VIDEO_FILENAME
    right_panel_child = (
        html.Video(id="video-player", src=f"/assets/{VIDEO_FILENAME}", controls=True, className="sb-video")
        if video_path.exists()
        else html.Div(f"Place a video at ./assets/{VIDEO_FILENAME}", className="sb-placeholder")
    )

    left = html.Div(
        [html.Div([dcc.Graph(id="tracking-graph", className="sb-graph", config={"responsive": True})],
                  className="sb-media__content")],
        className="sb-media sb-media--graph", style={"--ar": ar_padding},
    )
    right = html.Div(
        [html.Div([right_panel_child], className="sb-media__content")],
        className="sb-media sb-media--video", style={"--ar": ar_padding},
    )
    return html.Div([left, right], className="sb-grid-2col")


def build_bottom_panel(timestamps: List[int]) -> html.Div:
    n = len(timestamps)
    start, end = 0, n - 1

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(className="sb-controls-spacer"),
                            html.Div(
                                [
                                    dcc.RadioItems(
                                        id="mode-selector",
                                        options=[
                                            {"label": "Playback Mode", "value": "playback"},
                                            {"label": "Editor Mode", "value": "editor"},
                                        ],
                                        value="playback",
                                        inline=True,
                                        className="sb-segment sb-mode",
                                    ),
                                    html.Span(className="sb-ctlbar__sep"),
                                    html.Button("⏮ Prev", id="btn-prev", n_clicks=0, className="sb-btn"),
                                    html.Button("▶ Play", id="btn-play", n_clicks=0, className="sb-btn", disabled=False),
                                    html.Button("Next ⏭", id="btn-next", n_clicks=0, className="sb-btn"),
                                    dcc.Dropdown(
                                        id="speed-dropdown",
                                        options=[
                                            {"label": "0.5×", "value": 0.5},
                                            {"label": "1.0×", "value": 1.0},
                                            {"label": "2.0×", "value": 2.0},
                                        ],
                                        value=1.0,
                                        clearable=False,
                                        className="sb-speed",
                                        style={"width": "120px"},
                                    ),
                                    dcc.Checklist(
                                        id="loop-toggle",
                                        options=[{"label": "Loop", "value": "loop"}],
                                        value=[],
                                        className="sb-chip-toggle",
                                    ),
                                ],
                                className="sb-ctlbar",
                            ),
                            html.Div(id="frame-readout", className="sb-readout"),
                        ],
                        className="sb-controls-grid",
                    ),

                    html.Div(className="sb-divider"),

                    html.Div(
                        dcc.Slider(
                            id="time-slider-main",
                            min=start, max=end, value=start, step=1,
                            tooltip={"always_visible": False, "placement": "bottom"},
                            updatemode="drag",
                            className="sb-timeline",
                        ),
                        className="sb-card sb-card--padded",
                    ),

                    html.Div(className="sb-divider"),

                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div("Show team:", className="sb-label"),
                                    dcc.RadioItems(
                                        id="team-filter",
                                        options=[
                                            {"label": "Both", "value": "both"},
                                            {"label": "Offense", "value": "offense"},
                                            {"label": "Defense", "value": "defense"},
                                        ],
                                        value="both",
                                        inline=True,
                                        className="sb-segment",
                                    ),
                                ],
                                className="sb-row",
                            ),
                            html.Div(
                                [
                                    html.Div("Overlays:", className="sb-label"),
                                    dcc.Checklist(
                                        id="overlay-options",
                                        options=[
                                            {"label": "Show Players", "value": "players"},
                                            {"label": "Show Trails", "value": "trails"},
                                            {"label": "Show Voronoi", "value": "voronoi"},
                                            {"label": "Show Velocity", "value": "velocity"},
                                            {"label": "Pitch Control", "value": "pc"},
                                            {"label": "Coverage Control (beta)", "value": "coverage"},
                                            {"label": "EPV / xT (soon)", "value": "epvxt", "disabled": True},
                                        ],
                                        value=["players", "voronoi"],
                                        inline=True,
                                        className="sb-chips",
                                    ),
                                    html.Button("Reset", id="btn-reset-editor", n_clicks=0, className="sb-link"),
                                ],
                                className="sb-row sb-row--wrap",
                            ),
                        ],
                        className="sb-editor",
                    ),
                ],
                className="sb-suite sb-panel",
            ),
        ],
        className="sb-bottom",
    )


# --------------------------------------------------------------------------------------
# Build data & app at import time (Render needs server at module scope)
# --------------------------------------------------------------------------------------

Path("assets").mkdir(exist_ok=True)

if not Path(TRACKING_CSV).exists():
    raise FileNotFoundError(f"CSV file not found: {TRACKING_CSV}")

df = load_tracking_data_single(TRACKING_CSV)
timestamps = sorted(df["timestamp"].unique())
if not timestamps:
    raise RuntimeError("No timestamps found in tracking data.")

# Calibrate v_max from data (95th percentile of speed)
PC_VMAX = _auto_calibrate_vmax(df)

# Precompute coverage suggestions for smooth playback
_COV_CACHE = _precompute_coverage_cache(df, timestamps, RINK_BOUNDS)

app: Dash = dash.Dash(__name__)
server = app.server

app.title = "Sunbears Dashboard"
app.layout = html.Div(
    [
        build_header(),
        html.Div([build_top_row(RINK_BOUNDS), build_bottom_panel(timestamps)], className="sb-container"),
        dcc.Interval(id="play-interval", interval=int(BASE_INTERVAL_MS), disabled=True),
        dcc.Interval(id="video-poll", interval=500, n_intervals=0),
        dcc.Store(id="is-playing", data=False),
        dcc.Store(id="timestamps", data=timestamps),
        dcc.Store(id="edits-store", data={}),
        dcc.Store(id="shape-index-map", data=[]),
        dcc.Store(id="video-ctrl", data=None),
        dcc.Store(id="video-state", data={"playing": False, "ended": False}),
    ],
    className="sb-page",
    style=STYLES["page"],
)


# --------------------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------------------

@app.callback(Output("play-interval", "interval"), Input("speed-dropdown", "value"))
def set_speed(rate):
    try:
        rate = float(rate or 1.0)
        rate = max(0.1, min(2.0, rate))
    except Exception:
        rate = 1.0
    return int(max(20, BASE_INTERVAL_MS / rate))

@app.callback(Output("btn-play", "disabled"), Input("mode-selector", "value"))
def toggle_play_disabled(mode):
    return mode == "editor"

@app.callback(
    Output("time-slider-main", "value"),
    Output("is-playing", "data"),
    Output("play-interval", "disabled"),
    Output("video-ctrl", "data"),
    Input("play-interval", "n_intervals"),
    Input("btn-prev", "n_clicks"),
    Input("btn-next", "n_clicks"),
    Input("btn-play", "n_clicks"),
    Input("mode-selector", "value"),
    State("loop-toggle", "value"),
    State("time-slider-main", "value"),
    State("timestamps", "data"),
    State("is-playing", "data"),
    prevent_initial_call=True,
)
def playback_driver(_tick, _prev, _next, _play_clicks, mode,
                    loop_vals, cur_idx, ts_list, is_playing):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    total = len(ts_list)
    last = total - 1
    loop_on = "loop" in (loop_vals or [])

    if trigger == "mode-selector":
        return cur_idx, False, True, {"action": "pause", "token": f"mode:{mode}:{cur_idx}"}

    if trigger == "btn-play":
        new_playing = not bool(is_playing)
        new_idx = cur_idx
        if new_playing and cur_idx >= last:
            new_idx = 0
        vid_cmd = {"action": "pause", "token": f"btn:{_play_clicks}:pause"}
        if new_playing:
            vid_cmd = {"action": "play", "restart": True, "token": f"btn:{_play_clicks}:play"}
        return new_idx, new_playing, (not new_playing), vid_cmd

    if trigger == "btn-prev":
        if loop_on and cur_idx == 0:
            new_idx = last
        else:
            new_idx = max(0, cur_idx - 1)
        return new_idx, is_playing, (not is_playing), dash.no_update

    if trigger == "btn-next":
        if loop_on and cur_idx == last:
            new_idx = 0
        else:
            new_idx = min(last, cur_idx + 1)
        return new_idx, is_playing, (not is_playing), dash.no_update

    if trigger == "play-interval":
        if not is_playing or mode == "editor":
            raise dash.exceptions.PreventUpdate
        nxt = cur_idx + 1
        if nxt > last:
            if loop_on:
                return 0, True, False, dash.no_update
            else:
                return last, False, True, dash.no_update
        return nxt, True, False, dash.no_update

    raise dash.exceptions.PreventUpdate


@app.callback(Output("frame-readout", "children"), Input("time-slider-main", "value"), State("timestamps", "data"))
def update_readout(idx, ts_list):
    return f"Frame {idx} / {len(ts_list) - 1}"


@app.callback(
    Output("tracking-graph", "figure"),
    Output("shape-index-map", "data"),
    Output("tracking-graph", "config"),
    Input("time-slider-main", "value"),
    Input("overlay-options", "value"),
    Input("team-filter", "value"),
    Input("mode-selector", "value"),
    Input("edits-store", "data"),
    State("timestamps", "data"),
)
def update_figure(time_index: int, overlay_values, team_filter, mode, edits_store, ts_list):
    current_timestamp = ts_list[time_index]
    ts_key = str(current_timestamp)

    # Current frame data (full, both teams)
    df_frame_full = df[df["timestamp"] == current_timestamp].copy()

    # Apply per-frame edits to BOTH: (a) full frame for PC/Coverage; (b) display frame (team filter applied later)
    edits_for_ts = (edits_store or {}).get(ts_key, {}) if mode == "editor" else {}
    df_frame_full = _apply_edits_to_frame(df_frame_full, edits_for_ts)
    df_frame_for_display = df_frame_full.copy()

    # Flags
    show_trails = "trails" in (overlay_values or [])
    show_velocity = "velocity" in (overlay_values or [])
    show_pc = "pc" in (overlay_values or [])
    show_coverage = "coverage" in (overlay_values or [])

    # Trails (apply edits to trails only in editor)
    trails_df = make_trails(df, current_timestamp, TRAIL_DEFAULT) if show_trails else None
    if trails_df is not None and mode == "editor":
        trails_df = _apply_edits_to_trails(trails_df, edits_store or {})

    # Coverage control source
    coverage_df = None
    if show_coverage:
        if mode == "editor" and edits_for_ts:
            coverage_df = _instant_coverage_for_frame(df_frame_full, RINK_BOUNDS)
        else:
            coverage_df = _COV_CACHE.get(current_timestamp, None)

    fig, shape_map = build_tracking_figure(
        df_frame_for_display=df_frame_for_display,
        df_frame_for_pc_full=df_frame_full,
        trails_df=trails_df,
        bounds=RINK_BOUNDS,
        team_filter=team_filter,
        show_players=("players" in (overlay_values or [])),
        show_voronoi=("voronoi" in (overlay_values or [])),
        show_trails=show_trails,
        show_velocity=show_velocity,
        show_pc=show_pc,
        show_coverage=show_coverage,
        coverage_df=coverage_df,
        current_ts=current_timestamp,
        mode=mode,
        edits_enabled=True,
    )

    # Graph config: enable only SHAPE drag; block annotation drag
    if mode == "editor" and ("players" in (overlay_values or [])):
        cfg = {
            "responsive": True,
            "displayModeBar": True,
            "editable": True,
            "edits": {
                "shapePosition": True,
                "annotationPosition": False,
                "annotationText": False,
                "titleText": False,
                "axisTitleText": False,
                "legendPosition": False
            }
        }
    else:
        cfg = {"responsive": True, "displayModeBar": True, "editable": False}

    return fig, shape_map, cfg


@app.callback(
    Output("edits-store", "data"),
    Input("tracking-graph", "relayoutData"),
    Input("time-slider-main", "value"),
    Input("mode-selector", "value"),
    State("overlay-options", "value"),
    State("shape-index-map", "data"),
    State("timestamps", "data"),
    State("edits-store", "data"),
    prevent_initial_call=True,
)
def edits_manager(relayout, time_idx, mode, overlays, shape_map, ts_list, edits_store):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger == "time-slider-main" or trigger == "mode-selector":
        return {}

    if mode != "editor" or "players" not in (overlays or []):
        raise dash.exceptions.PreventUpdate

    if not relayout or not isinstance(relayout, dict) or not shape_map:
        raise dash.exceptions.PreventUpdate

    per_idx: Dict[int, Dict[str, float]] = {}
    for k, v in relayout.items():
        if not (k.startswith("shapes[") and "]" in k):
            continue
        try:
            idx = int(k.split("[", 1)[1].split("]", 1)[0])
        except Exception:
            continue
        per_idx.setdefault(idx, {})
        if k.endswith(".x0"):
            per_idx[idx]["x0"] = float(v)
        elif k.endswith(".x1"):
            per_idx[idx]["x1"] = float(v)
        elif k.endswith(".y0"):
            per_idx[idx]["y0"] = float(v)
        elif k.endswith(".y1"):
            per_idx[idx]["y1"] = float(v)

    if not per_idx:
        raise dash.exceptions.PreventUpdate

    edits_store = edits_store or {}
    ts_key = str(ts_list[time_idx])
    edits_store.setdefault(ts_key, {})

    for idx, bbox in per_idx.items():
        if idx < 0 or idx >= len(shape_map):
            continue
        if not all(k in bbox for k in ("x0", "x1", "y0", "y1")):
            continue
        cx = (bbox["x0"] + bbox["x1"]) / 2.0
        cy = (bbox["y0"] + bbox["y1"]) / 2.0
        key = shape_map[idx]             # "Team|player_id"
        team = key.split("|", 1)[0]
        edits_store[ts_key][key] = {"x": float(cx), "y": float(cy), "team": team}

    return edits_store


# ---------------------- Clientside small helpers ----------------------

# FIX: also pause video when mode switches to "editor" (even if loop is ON)
app.clientside_callback(
    """
    function(cmd, speedVal, loopVals, modeVal, _poll) {
        const video = document.getElementById("video-player");
        const state = {playing: false, ended: false};

        if (!video) { return state; }

        // Always pause when entering editor mode (prevents looped videos from continuing)
        try {
            if (modeVal === "editor") {
                video.pause();
            }
        } catch (e) {}

        // Speed / Loop - must NOT start playback
        if (speedVal !== undefined && speedVal !== null) {
            try { video.playbackRate = Number(speedVal) || 1.0; } catch (e) {}
        }
        try { video.loop = Array.isArray(loopVals) && loopVals.indexOf("loop") !== -1; } catch (e) {}

        // Command de-dup
        try {
            const lastTok = window.__sb_last_cmd_token || null;
            const tok = cmd && cmd.token ? String(cmd.token) : null;

            if (cmd && cmd.action && tok && tok !== lastTok) {
                window.__sb_last_cmd_token = tok;

                if (cmd.action === "play") {
                    if (cmd.restart && (video.ended || (video.duration && video.currentTime >= video.duration - 0.01))) {
                        video.currentTime = 0;
                    }
                    video.play();
                } else if (cmd.action === "pause") {
                    video.pause();
                }
            }
        } catch (e) {}

        // Report state
        try {
            state.ended = !!video.ended;
            state.playing = !(video.paused || video.ended);
        } catch (e) {}

        return state;
    }
    """,
    Output("video-state", "data"),
    [Input("video-ctrl", "data"),
     Input("speed-dropdown", "value"),
     Input("loop-toggle", "value"),
     Input("mode-selector", "value"),
     Input("video-poll", "n_intervals")],
)

app.clientside_callback(
    """
    function(isPlaying, vstate) {
        const vp = vstate && vstate.playing;
        return (isPlaying || vp) ? "⏸ Pause" : "▶ Play";
    }
    """,
    Output("btn-play", "children"),
    [Input("is-playing", "data"), Input("video-state", "data")],
)


if __name__ == "__main__":
    app.run(debug=True)
