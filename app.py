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
  • Voronoi, Pitch Control and Scoring Threat recompute instantly from edited positions
  • Reference Defense shows where pitch-control coverage would move the defenders of the frame
  • Changing frame OR switching modes clears edits; returning to a frame shows original data

Defensive analytics (./analytics):
- Pitch Control: physics-based control model (Spearman), who gets to a loose puck first
- Scoring Threat: off-puck scoring opportunity of the attack, as a surface and as a number
- Reference Defense: a unit driven by pitch-control coverage that replays the clip against
  the tracked attack; the panel compares the threat it concedes with the tracked defense

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
from dash.dependencies import Input, Output, State, ALL
import dash.exceptions
import pandas as pd
import plotly.graph_objs as go

from utils import compute_voronoi  # Voronoi + clipping
from analytics import EngineConfig, SceneAnalytics

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

# Files (defaults at startup)
TRACKING_CSV = "assets/clip_scene/scene_5.csv"   # merged input with team_id (0=Defense, else Offense)
VIDEO_FILENAME = "clip_scene/scene_5.mp4"        # path under ./assets/
FIELD_IMAGE = "field_hockey.png"                 # rink image under ./assets/
ICON_IMAGE = "sunbears_icon.webp"                # header icon under ./assets/

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

# -------- Defensive analytics (./analytics) --------
# Frame rate of the tracking [Hz]. None = infer it from the stored velocities.
TRACKING_FPS: Optional[float] = None
# Dropdown value for "the puck carrier is not known": results are averaged over the attackers.
CARRIER_AUTO = "auto"

# Pitch control heatmap
PC_OPACITY = 0.50

# Scoring-threat heatmap, in % chance of a goal from the next puck event per 100 m² of ice.
# The scale is fixed so that colours can be compared between frames and clips.
THREAT_ZMAX = 2.0
THREAT_OPACITY = 0.90
THREAT_COLORSCALE = [
    [0.00, "rgba(255,255,255,0)"],
    [0.05, "rgba(250,215,160,0.55)"],
    [0.25, "rgba(245,176,65,0.85)"],
    [0.60, "rgba(211,84,0,0.90)"],
    [1.00, "rgba(120,40,31,0.95)"],
]

# Reference defense (pitch-control coverage): ghosts on the rink, line in the timeline
REF_COLOR = "#1e8449"
REF_GHOST_SIZE = 16
REF_GHOST_FILL = "rgba(30,132,73,0.45)"
REF_CONNECTOR_COLOR = "rgba(30,132,73,0.75)"
REF_CONNECTOR_WIDTH = 2
TRACKED_COLOR = "#2e86de"   # tracked defense in the timeline (same blue as the defenders)

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

            # Arrowhead
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
# Defensive analytics: pitch control, scoring threat, reference defense
# --------------------------------------------------------------------------------------

ANALYTICS: Optional[SceneAnalytics] = None
# Last editor frame that was analysed (the figure and the tiles ask for the same one).
_EDIT_MEMO: Dict[str, Any] = {}


def _new_analytics(df_all: pd.DataFrame) -> SceneAnalytics:
    """Engine for a clip; the replay of the reference defense starts in the background."""
    engine = SceneAnalytics(df_all, RINK_BOUNDS, EngineConfig(fps=TRACKING_FPS))
    engine.ensure_timeline(None)
    return engine


def _carrier_arg(value: Any) -> Optional[str]:
    return None if value in (None, "", CARRIER_AUTO) else str(value)


def _carrier_options() -> List[Dict[str, str]]:
    opts = [{"label": "Unknown (average over attackers)", "value": CARRIER_AUTO}]
    opts += [{"label": f"Attacker {pid}", "value": pid} for pid in ANALYTICS.carrier_choices()]
    return opts


def _editor_analysis(df_frame_full: pd.DataFrame, carrier: Optional[str]) -> Dict[str, Any]:
    """Threat and suggested positions of an editor frame, memoised on the positions."""
    key = (
        id(ANALYTICS), carrier,
        tuple(df_frame_full["team"].astype(str) + "|" + df_frame_full["player_id"].astype(str)),
        tuple(np.round(df_frame_full[["x", "y"]].to_numpy(float), 3).ravel()),
    )
    memo = _EDIT_MEMO
    if memo.get("key") != key:
        state = ANALYTICS.state_from_frame(df_frame_full)
        memo = {
            "key": key,
            "state": state,
            "result": ANALYTICS.evaluate(state, carrier),
            "suggestion": ANALYTICS.suggest(state, carrier),
        }
        _EDIT_MEMO.clear()
        _EDIT_MEMO.update(memo)
    return memo


def _pct(value: float, digits: int = 2) -> str:
    return "–" if value is None or not np.isfinite(value) else f"{100.0 * value:.{digits}f}%"


def _delta(new: float, base: float) -> str:
    """Relative change of ``new`` with respect to ``base`` as signed text."""
    if not (np.isfinite(new) and np.isfinite(base)) or base <= 0:
        return "–"
    change = 100.0 * (new / base - 1.0)
    sign = "−" if change < 0 else "+"
    return f"{sign}{abs(change):.0f}%"


def _timeline_figure(carrier: Optional[str]) -> Dict[str, Any]:
    """Threat over the clip against the tracked and the reference defense."""
    tl = ANALYTICS.timeline(carrier)
    fig = go.Figure()
    axis = dict(showgrid=True, gridcolor="#eef1f5", zeroline=False, linecolor="#d1d5db",
                tickfont=dict(color="#6b7280", size=11), title_font=dict(color="#6b7280", size=12))
    annotations = []
    if len(tl["frame_idx"]) == 0:
        annotations.append(dict(text="The threat timeline needs attackers and defenders in the clip.",
                                showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5,
                                font=dict(size=13, color="#6b7280")))
    else:
        x = tl["frame_idx"] / ANALYTICS.fps          # seconds from the start of the clip
        series = [("Tracked defense", tl["obso_tracked"], TRACKED_COLOR),
                  ("Reference defense", tl["obso_reference"], REF_COLOR)]
        for name, y, color in series:
            fig.add_trace(go.Scatter(
                x=x, y=100.0 * y, name=name, mode="lines",
                line=dict(color=color, width=2),
                hovertemplate="%{y:.2f}%<extra>" + name + "</extra>",
            ))
        # Direct labels at the right end, pushed apart so they never overlap.
        ends = [100.0 * float(y[-1]) for _, y, _ in series]
        for k, (name, _, color) in enumerate(series):
            above = ends[k] >= ends[1 - k]
            annotations.append(dict(
                x=float(x[-1]), y=ends[k], xref="x", yref="y", text=name.split()[0],
                showarrow=False, xanchor="left", xshift=6, yshift=8 if above else -8,
                font=dict(size=11, color="#374151"),
            ))
    fig.update_layout(
        autosize=True,
        margin=dict(l=56, r=76, t=34, b=44),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#ffffff", font=dict(color="#111827", size=12)),
        legend=dict(orientation="h", x=0, y=1.14, xanchor="left", yanchor="top",
                    font=dict(color="#374151", size=12)),
        xaxis=dict(title="Time in the clip", ticksuffix=" s", hoverformat=".1f", **axis),
        yaxis=dict(title="Scoring threat (%)", rangemode="tozero", ticksuffix="%", **axis),
        annotations=annotations,
        # The first shape is the cursor of the current frame; it is moved in the browser.
        shapes=[dict(type="line", xref="x", yref="paper", x0=0, x1=0, y0=0, y1=1,
                     line=dict(color="#111827", width=1.5, dash="dot"))],
        uirevision="timeline",
        meta=dict(fps=ANALYTICS.fps),
    )
    return fig.to_plotly_json()


def _tile(label: str, value: str, note: str, accent: str, wide: bool = False) -> html.Div:
    return html.Div(
        [
            html.Div(label, className="sb-tile__label"),
            html.Div(value, className="sb-tile__value"),
            html.Div(note, className="sb-tile__note"),
        ],
        className="sb-tile sb-tile--wide" if wide else "sb-tile", style={"--tile-accent": accent},
    )


def _receiver_bars(result: Dict[str, Any]) -> List[Any]:
    """Share of the threat carried by each attacker, largest first."""
    ids, vals = result.get("att_ids", []), np.asarray(result.get("per_attacker", []), dtype=float)
    if len(ids) == 0:
        return []
    top = float(vals.max()) if vals.max() > 0 else 1.0
    rows = [html.Div("Threat by attacker", className="sb-bars__title")]
    for k in np.argsort(-vals):
        rows.append(html.Div(
            [
                html.Div(f"#{ids[k]}", className="sb-bars__id"),
                html.Div(html.Div(className="sb-bars__fill", style={"width": f"{100.0 * vals[k] / top:.1f}%"}),
                         className="sb-bars__track"),
                html.Div(_pct(vals[k]), className="sb-bars__value"),
            ],
            className="sb-bars__row",
        ))
    return rows


# --------------------------------------------------------------------------------------
# Figure builder
# --------------------------------------------------------------------------------------

def build_tracking_figure(
    df_frame_for_display: pd.DataFrame,
    trails_df: Optional[pd.DataFrame],
    bounds: Dict[str, float],
    team_filter: str,
    show_players: bool,
    show_voronoi: bool,
    show_trails: bool,
    show_velocity: bool,
    pc_field: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    threat_field: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    reference_df: Optional[pd.DataFrame],
    mode: str,
    edits_enabled: bool,
) -> Tuple[go.Figure, List[str]]:
    data: List[go.Scatter] = []
    shape_index_map: List[str] = []

    df_frame = _clamp_df(df_frame_for_display, bounds)
    trails_df = _clamp_df(trails_df, bounds)

    # team filter (affects players/trails/voronoi/reference; the heatmaps always use both teams)
    if team_filter != "both":
        keep = "Offense" if team_filter == "offense" else "Defense"
        df_frame = df_frame[df_frame["team"] == keep]
        if trails_df is not None:
            trails_df = trails_df[trails_df["team"] == keep]
        if reference_df is not None:
            shown_def_pids = set(df_frame[df_frame["team"] == "Defense"]["player_id"].astype(str).tolist())
            reference_df = reference_df[reference_df["player_id"].astype(str).isin(shown_def_pids)]

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
    if pc_field is not None:
        gx, gy, Z = pc_field
        data.append(
            go.Heatmap(
                x=gx, y=gy, z=Z,
                colorscale=PC_COLORSCALE,
                zmin=0.0, zmax=1.0, zsmooth="best",
                showscale=False,
                hovertemplate="Offense control: %{z:.2f}<extra></extra>",
                opacity=PC_OPACITY,
            )
        )

    # scoring-threat heatmap: where the attack's chance of a goal comes from
    if threat_field is not None:
        gx, gy, Z = threat_field
        data.append(
            go.Heatmap(
                x=gx, y=gy, z=1.0e4 * Z,
                colorscale=THREAT_COLORSCALE,
                zmin=0.0, zmax=THREAT_ZMAX, zsmooth="best",
                showscale=False,
                hovertemplate="Scoring threat: %{z:.2f}% per 100 m²<extra></extra>",
                opacity=THREAT_OPACITY,
            )
        )

    # voronoi
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

    # velocity
    if show_velocity and not df_frame.empty:
        data.extend(_velocity_quiver_traces(df_frame))

    # reference defense (connectors + ghost defenders)
    if reference_df is not None and not reference_df.empty:
        cur_defs = df_frame[df_frame["team"] == "Defense"][["player_id", "x", "y"]].copy()
        cur_defs["player_id"] = cur_defs["player_id"].astype(str)
        ref = reference_df.merge(cur_defs, on="player_id", how="inner")
        if not ref.empty:
            xs, ys = [], []
            for _, r in ref.iterrows():
                xs += [float(r["x"]), float(r["sug_x"]), None]
                ys += [float(r["y"]), float(r["sug_y"]), None]
            data.append(
                go.Scatter(
                    x=xs, y=ys, mode="lines",
                    line=dict(width=REF_CONNECTOR_WIDTH, color=REF_CONNECTOR_COLOR, dash="dot"),
                    opacity=0.9, hoverinfo="skip", showlegend=False
                )
            )
            data.append(
                go.Scatter(
                    x=ref["sug_x"], y=ref["sug_y"],
                    mode="markers+text",
                    marker=dict(size=REF_GHOST_SIZE, color=REF_GHOST_FILL, line=dict(width=1.5, color=REF_COLOR)),
                    text=[f"{pid}′" for pid in ref["player_id"]],
                    textposition="middle center",
                    textfont=dict(color="#0b3d20", size=11),
                    hovertemplate="Reference defender %{text}<extra></extra>", showlegend=False,
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

    # players + labels / draggable in editor
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


# --------------------------------------------------------------------------------------
# UI builders
# --------------------------------------------------------------------------------------

def build_header() -> html.Div:
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
                    html.Div(className="sb-header__spacer"),
                    html.Button(
                        "Import Data",
                        id={"role": "import", "action": "open"},
                        className="sb-btn sb-btn--primary sb-btn--import",
                        title="Import tracking CSV & matching MP4 (same scene name)",
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
                                            {"label": "Scoring Threat", "value": "threat"},
                                            {"label": "Reference Defense", "value": "reference"},
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


def build_analytics_panel() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Defensive analytics", className="sb-analytics__title"),
                            html.Div(
                                "Scoring threat is the chance that the next puck event ends in a goal. "
                                "The reference defense starts where the tracked defenders start and "
                                "then follows pitch-control coverage against the tracked attack.",
                                className="sb-analytics__lede",
                            ),
                        ],
                        className="sb-analytics__head",
                    ),
                    html.Div(
                        [
                            html.Div("Puck carrier", className="sb-analytics__field"),
                            dcc.Dropdown(
                                id="carrier-select",
                                options=_carrier_options(),
                                value=CARRIER_AUTO,
                                clearable=False,
                                className="sb-carrier",
                            ),
                        ],
                        className="sb-analytics__carrier",
                    ),
                ],
                className="sb-analytics__top",
            ),
            html.Div(
                [
                    html.Div(
                        dcc.Graph(
                            id="threat-timeline", className="sb-timeline-graph",
                            style={"height": "310px"},
                            config={"responsive": True, "displayModeBar": False},
                        ),
                        className="sb-analytics__chart",
                    ),
                    html.Div(
                        [
                            html.Div(id="threat-tiles", className="sb-tiles"),
                            html.Div(id="threat-receivers", className="sb-bars"),
                        ],
                        className="sb-analytics__side",
                    ),
                ],
                className="sb-analytics__grid",
            ),
            html.Div(id="analytics-note", className="sb-analytics__note"),
            # Figure of the timeline once the replay is ready, and the poll that waits for it
            dcc.Store(id="timeline-figure"),
            dcc.Interval(id="timeline-poll", interval=700, disabled=True),
        ],
        className="sb-panel sb-analytics",
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

# Pitch control, scoring threat and the reference defense of the clip
ANALYTICS = _new_analytics(df)

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.title = "Sunbears Dashboard"
app.layout = html.Div(
    [
        build_header(),
        html.Div(
            [build_top_row(RINK_BOUNDS), build_bottom_panel(timestamps), build_analytics_panel()],
            className="sb-container",
        ),
        
        # Modal placeholder
        html.Div(id="import-modal-container"),
        
        # Stores
        dcc.Store(id="is-playing", data=False),
        dcc.Store(id="timestamps", data=timestamps),
        dcc.Store(id="edits-store", data={}),
        dcc.Store(id="shape-index-map", data=[]),
        dcc.Store(id="video-ctrl", data=None),
        dcc.Store(id="video-state", data={"playing": False, "ended": False}),
        dcc.Store(id="dataset-info", data={"csv": TRACKING_CSV, "video": f"assets/{VIDEO_FILENAME}"}),

        # Intervals
        dcc.Interval(id="play-interval", interval=int(BASE_INTERVAL_MS), disabled=True),
        dcc.Interval(id="video-poll", interval=500, n_intervals=0),
    ],
    className="sb-page",
    style=STYLES["page"],
)


# --------------------------------------------------------------------------------------
# Import Modal UI helper
# --------------------------------------------------------------------------------------

def build_import_modal():
    return html.Div(
        [
            html.Div(
                [
                    html.H3("Import New Dataset", className="sb-modal__title"),
                    html.Label("Folder Path:"),
                    dcc.Input(
                        id={"role": "import", "field": "folder"},
                        type="text",
                        placeholder="e.g., assets/clip_scene",
                        value="assets/clip_scene",
                        className="sb-input",
                    ),
                    html.Label("Scene Name:"),
                    dcc.Input(
                        id={"role": "import", "field": "scene"},
                        type="text",
                        placeholder="e.g., scene_12",
                        className="sb-input",
                    ),
                    html.Div(id="import-error", className="sb-error"),
                    html.Div(
                        [
                            html.Button("Cancel", id={"role": "import", "action": "cancel"}, className="sb-btn"),
                            html.Button("Confirm", id={"role": "import", "action": "confirm"}, className="sb-btn sb-btn--primary"),
                        ],
                        className="sb-modal__actions",
                    ),
                ],
                className="sb-modal__content",
            )
        ],
        className="sb-modal",
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


# Adjust slider bounds when timestamps change (after import)
@app.callback(
    Output("time-slider-main", "max"),
    Output("time-slider-main", "min"),
    Input("timestamps", "data"),
)
def adjust_slider_bounds(ts_list):
    if not ts_list:
        raise dash.exceptions.PreventUpdate
    return len(ts_list) - 1, 0


@app.callback(
    Output("tracking-graph", "figure"),
    Output("shape-index-map", "data"),
    Output("tracking-graph", "config"),
    Input("time-slider-main", "value"),
    Input("overlay-options", "value"),
    Input("team-filter", "value"),
    Input("mode-selector", "value"),
    Input("edits-store", "data"),
    Input("carrier-select", "value"),
    Input("timeline-figure", "data"),   # redraw when the replay of the reference defense is ready
    State("timestamps", "data"),
)
def update_figure(time_index: int, overlay_values, team_filter, mode, edits_store, carrier_value,
                  _timeline_ready, ts_list):
    current_timestamp = ts_list[time_index]
    ts_key = str(current_timestamp)
    overlays = overlay_values or []

    # Current frame data (full, both teams)
    df_frame_full = df[df["timestamp"] == current_timestamp].copy()

    edits_for_ts = (edits_store or {}).get(ts_key, {}) if mode == "editor" else {}
    df_frame_full = _apply_edits_to_frame(df_frame_full, edits_for_ts)
    df_frame_for_display = df_frame_full.copy()

    # Flags
    show_trails = "trails" in overlays
    show_velocity = "velocity" in overlays
    show_pc = "pc" in overlays
    show_threat = "threat" in overlays
    show_reference = "reference" in overlays

    # Trails
    trails_df = make_trails(df, current_timestamp, TRAIL_DEFAULT) if show_trails else None
    if trails_df is not None and mode == "editor":
        trails_df = _apply_edits_to_trails(trails_df, edits_store or {})

    # Defensive analytics. Playback reads the tracked frame and the replayed reference
    # defense; the editor analyses the frame as it is drawn and suggests positions for it.
    pc_field = threat_field = reference_df = None
    carrier = _carrier_arg(carrier_value)
    if (show_pc or show_threat or show_reference) and ANALYTICS.has_frame(current_timestamp):
        if mode == "editor":
            info = _editor_analysis(df_frame_full, carrier)
            if show_pc:
                pc_field = ANALYTICS.pitch_control(info["state"], carrier)
            if show_threat:
                threat_field = ANALYTICS.threat_surface(info["result"])
            if show_reference:
                reference_df = info["suggestion"]["positions"]
        else:
            if show_pc:
                pc_field = ANALYTICS.pitch_control_at(current_timestamp, carrier)
            if show_threat:
                threat_field = ANALYTICS.threat_surface(ANALYTICS.evaluate_at(current_timestamp, carrier))
            if show_reference and ANALYTICS.timeline_ready(carrier):
                reference_df = ANALYTICS.reference_at(current_timestamp, carrier)

    fig, shape_map = build_tracking_figure(
        df_frame_for_display=df_frame_for_display,
        trails_df=trails_df,
        bounds=RINK_BOUNDS,
        team_filter=team_filter,
        show_players=("players" in overlays),
        show_voronoi=("voronoi" in overlays),
        show_trails=show_trails,
        show_velocity=show_velocity,
        pc_field=pc_field,
        threat_field=threat_field,
        reference_df=reference_df,
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
        key = shape_map[idx]  # "Team|player_id"
        team = key.split("|", 1)[0]
        edits_store[ts_key][key] = {"x": float(cx), "y": float(cy), "team": team}

    return edits_store


# ---------------------- Defensive analytics panel ----------------------

@app.callback(
    Output("carrier-select", "options"),
    Output("carrier-select", "value"),
    Output("analytics-note", "children"),
    Input("timestamps", "data"),
)
def refresh_analytics_meta(_ts_list):
    """A new clip has its own attackers, frame rate and attack direction."""
    note = (
        f"Read from the clip: {ANALYTICS.describe()}. "
        "The tracked attack does not react to the reference defense, so read the gap as "
        "how much threat was defendable, not as what would have happened."
    )
    return _carrier_options(), CARRIER_AUTO, note


@app.callback(
    Output("timeline-figure", "data"),
    Output("timeline-poll", "disabled"),
    Input("carrier-select", "value"),
    Input("timestamps", "data"),
    Input("timeline-poll", "n_intervals"),
    State("timeline-figure", "data"),
)
def update_timeline(carrier_value, _ts_list, _tick, current):
    """The replay runs in the background; poll until it is ready, then draw it."""
    carrier = _carrier_arg(carrier_value)
    if ANALYTICS.ensure_timeline(carrier):
        return _timeline_figure(carrier), True
    return (dash.no_update if current is None else None), False


# Move the cursor of the timeline in the browser: no server round trip per frame.
app.clientside_callback(
    """
    function(fig, idx) {
        if (!fig) {
            return {data: [], layout: {
                xaxis: {visible: false}, yaxis: {visible: false},
                margin: {l: 10, r: 10, t: 10, b: 10},
                annotations: [{text: "Replaying the clip with the reference defense…",
                               showarrow: false, xref: "paper", yref: "paper", x: 0.5, y: 0.5,
                               font: {size: 13, color: "#6b7280"}}]}};
        }
        const layout = Object.assign({}, fig.layout);
        if (layout.shapes && layout.shapes.length) {
            const t = idx / ((layout.meta && layout.meta.fps) || 1);
            const cursor = Object.assign({}, layout.shapes[0], {x0: t, x1: t});
            layout.shapes = [cursor].concat(layout.shapes.slice(1));
        }
        return Object.assign({}, fig, {layout: layout});
    }
    """,
    Output("threat-timeline", "figure"),
    Input("timeline-figure", "data"),
    Input("time-slider-main", "value"),
)


@app.callback(
    Output("time-slider-main", "value", allow_duplicate=True),
    Input("threat-timeline", "clickData"),
    State("timestamps", "data"),
    prevent_initial_call=True,
)
def seek_from_timeline(click, ts_list):
    """Clicking the timeline jumps to that moment of the clip."""
    try:
        idx = int(round(float(click["points"][0]["x"]) * ANALYTICS.fps))
    except (TypeError, KeyError, IndexError, ValueError):
        raise dash.exceptions.PreventUpdate
    return max(0, min(len(ts_list) - 1, idx))


@app.callback(
    Output("threat-tiles", "children"),
    Output("threat-receivers", "children"),
    Input("time-slider-main", "value"),
    Input("carrier-select", "value"),
    Input("mode-selector", "value"),
    Input("edits-store", "data"),
    Input("timeline-figure", "data"),
    State("timestamps", "data"),
)
def update_threat_tiles(time_index, carrier_value, mode, edits_store, _timeline_ready, ts_list):
    if time_index is None or time_index >= len(ts_list):
        raise dash.exceptions.PreventUpdate
    ts = ts_list[time_index]
    if not ANALYTICS.has_frame(ts):
        raise dash.exceptions.PreventUpdate
    carrier = _carrier_arg(carrier_value)

    if mode == "editor":
        frame = df[df["timestamp"] == ts].copy()
        frame = _apply_edits_to_frame(frame, (edits_store or {}).get(str(ts), {}))
        info = _editor_analysis(frame, carrier)
        result = info["result"]
        now, other = result["obso"], info["suggestion"]["obso"]
        first = _tile("Scoring threat, this frame", _pct(now),
                      "Against the defenders as placed. Drag a player to see it change.", TRACKED_COLOR)
        second = _tile("With suggested positions", _pct(other),
                       f"{_delta(other, now)} with the defenders on the green markers", REF_COLOR)
    else:
        result = ANALYTICS.evaluate_at(ts, carrier)
        now = result["obso"]
        ready = ANALYTICS.timeline_ready(carrier)
        other = ANALYTICS.reference_threat_at(ts, carrier) if ready else float("nan")
        first = _tile("Scoring threat now", _pct(now), "Against the tracked defense", TRACKED_COLOR)
        second = _tile("Reference defense now", _pct(other),
                       f"{_delta(other, now)} vs tracked" if ready else "Replaying the clip…", REF_COLOR)

    if ANALYTICS.timeline_ready(carrier):
        tl = ANALYTICS.timeline(carrier)
        mt, mr = tl["mean_tracked"], tl["mean_reference"]
        if np.isfinite(mt) and np.isfinite(mr) and mt > 0:
            verb = "less" if mr <= mt else "more"
            note = f"Reference concedes {abs(100.0 * (mr / mt - 1.0)):.0f}% {verb} over the clip"
        else:
            note = "Needs attackers and defenders in the clip"
        third = _tile("Clip average, tracked → reference", f"{_pct(mt)} → {_pct(mr)}", note, "#6b7280", wide=True)
    else:
        third = _tile("Clip average, tracked → reference", "–", "Replaying the clip…", "#6b7280", wide=True)

    return [first, second, third], _receiver_bars(result)


# ---------------------- Import Modal & Dataset Switching ----------------------

# One callback handles open/cancel via pattern matching (no missing-ID errors)
@app.callback(
    Output("import-modal-container", "children"),
    Input({"role": "import", "action": ALL}, "n_clicks"),
    State("import-modal-container", "children"),
    prevent_initial_call=True
)
def toggle_import_modal(btn_clicks, modal_children):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    trig = ctx.triggered[0]["prop_id"]  # e.g. {"role":"import","action":"open"}.n_clicks

    if '"open"' in trig:
        return build_import_modal()
    if '"cancel"' in trig:
        return None
    return modal_children


# Perform import on confirm
@app.callback(
    Output("dataset-info", "data", allow_duplicate=True),
    Output("timestamps", "data", allow_duplicate=True),
    Output("time-slider-main", "value", allow_duplicate=True),
    Output("import-error", "children", allow_duplicate=True),
    Output("import-modal-container", "children", allow_duplicate=True),
    Input({"role": "import", "action": "confirm"}, "n_clicks"),
    State({"role": "import", "field": "folder"}, "value"),
    State({"role": "import", "field": "scene"}, "value"),
    prevent_initial_call=True,
)
def process_import(confirm_click, folder, scene_name):
    if not confirm_click:
        raise dash.exceptions.PreventUpdate

    if not scene_name or not folder:
        return dash.no_update, dash.no_update, dash.no_update, "Please enter both folder and scene name.", dash.no_update

    folder_path = Path(folder)
    csv_path = folder_path / f"{scene_name}.csv"
    video_path = folder_path / f"{scene_name}.mp4"

    if not csv_path.exists():
        return dash.no_update, dash.no_update, dash.no_update, f"CSV not found: {csv_path}", dash.no_update
    if not video_path.exists():
        return dash.no_update, dash.no_update, dash.no_update, f"Video not found: {video_path}", dash.no_update

    try:
        new_df = load_tracking_data_single(str(csv_path))
    except Exception as e:
        return dash.no_update, dash.no_update, dash.no_update, f"Error loading CSV: {e}", dash.no_update

    new_timestamps = sorted(new_df["timestamp"].unique())
    if not new_timestamps:
        return dash.no_update, dash.no_update, dash.no_update, "No timestamps found in the CSV.", dash.no_update

    # Update global variables & caches
    global df, ANALYTICS
    df = new_df
    ANALYTICS = _new_analytics(df)
    _EDIT_MEMO.clear()

    return (
        {"csv": str(csv_path), "video": f"assets/{video_path.as_posix()}".replace("assets/assets/", "assets/")},
        new_timestamps,
        0,     # reset slider to start
        "",    # clear any error
        None   # close modal
    )


# Update video source when dataset changes
@app.callback(
    Output("video-player", "src"),
    Input("dataset-info", "data"),
)
def update_video_src(dataset_info):
    if not dataset_info or "video" not in dataset_info:
        raise dash.exceptions.PreventUpdate
    path = dataset_info["video"]
    if not path.startswith("assets/"):
        path = f"assets/{path}"
    return f"/{path}"


# ---------------------- Clientside small helpers ----------------------

# Pause video when entering editor, manage loop/speed, and de-dupe commands
app.clientside_callback(
    """
    function(cmd, speedVal, loopVals, modeVal, _poll) {
        const video = document.getElementById("video-player");
        const state = {playing: false, ended: false};

        if (!video) { return state; }

        try { if (modeVal === "editor") { video.pause(); } } catch (e) {}

        if (speedVal !== undefined && speedVal !== null) {
            try { video.playbackRate = Number(speedVal) || 1.0; } catch (e) {}
        }
        try { video.loop = Array.isArray(loopVals) && loopVals.indexOf("loop") !== -1; } catch (e) {}

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
