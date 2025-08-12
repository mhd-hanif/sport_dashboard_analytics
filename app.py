"""
Sunbears Sports Analytics Dashboard (Dash)

- Loads hockey tracking from two CSVs (Defense/Offense)
- Top row: Digital Tracking (Plotly) + Video in proportional, rounded cards
- Bottom tabs:
  • Analysis Playback: centered control bar + single scrubber + frame readout
  • Editor / Overlays: segmented team filter + chip overlays (incl. Coverage Control - soon)
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import dash
from dash import html, dcc, Dash
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

from utils import compute_voronoi  # Voronoi + clipping


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

# Files
DEFENSIVE_CSV = "assets/defensive_players_hockey.csv"
OFFENSIVE_CSV = "assets/offensive_players_hockey.csv"
VIDEO_FILENAME = "sample_video.mp4"   # place under ./assets/ (optional)
FIELD_IMAGE = "field_hockey.png"      # rink image under ./assets/
ICON_IMAGE = "sunbears_icon.webp"     # header icon under ./assets/

# Rink bounds (match your data)
RINK_BOUNDS: Dict[str, float] = {"x_min": 0.0, "x_max": 61.0, "y_min": 0.0, "y_max": 30.0}

# Default trail length used if "Trails" overlay is enabled (no UI control now)
TRAIL_DEFAULT = 40

STYLES: Dict[str, Any] = {
    "page": {"background": "#f6f7fb", "fontFamily": "Inter, Segoe UI, Arial, sans-serif"},
}


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------

def load_tracking_data(def_path: str, off_path: str) -> pd.DataFrame:
    """Read both CSVs, normalize columns, add team labels, sort by time."""
    def _read_and_normalize(p: str, label: str) -> pd.DataFrame:
        df = pd.read_csv(p)
        df.columns = [c.strip() for c in df.columns]

        # Map timeframe->timestamp; keep player_id, x, y
        col_map = {}
        for c in df.columns:
            lc = c.lower()
            if lc == "timeframe":
                col_map[c] = "timestamp"
            elif lc in {"timestamp", "player_id", "x", "y"}:
                col_map[c] = lc
        if col_map:
            df.rename(columns=col_map, inplace=True)

        required = {"timestamp", "player_id", "x", "y"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {label} data: {missing}. Found: {list(df.columns)}")

        df["timestamp"] = df["timestamp"].astype(int)
        df["player_id"] = df["player_id"].astype(str)
        df["x"] = df["x"].astype(float)
        df["y"] = df["y"].astype(float)
        df["team"] = label
        return df[["timestamp", "player_id", "team", "x", "y"]]

    def_df = _read_and_normalize(def_path, "Defense")
    off_df = _read_and_normalize(off_path, "Offense")

    df = pd.concat([def_df, off_df], ignore_index=True).sort_values(
        ["timestamp", "team", "player_id"], kind="mergesort"
    )
    df.reset_index(drop=True, inplace=True)
    return df


# --------------------------------------------------------------------------------------
# Figure builders
# --------------------------------------------------------------------------------------

def _clamp_df(df_: Optional[pd.DataFrame], bounds: Dict[str, float]) -> Optional[pd.DataFrame]:
    if df_ is None or df_.empty:
        return df_
    df_ = df_.copy()
    df_["x"] = df_["x"].clip(bounds["x_min"], bounds["x_max"])
    df_["y"] = df_["y"].clip(bounds["y_min"], bounds["y_max"])
    return df_


def build_tracking_figure(
    df_frame: pd.DataFrame,
    bounds: Dict[str, float],
    team_filter: str,
    show_players: bool,
    show_voronoi: bool,
    show_trails: bool,
    trails_df: Optional[pd.DataFrame],
) -> go.Figure:
    data: List[go.Scatter] = []

    df_frame = _clamp_df(df_frame, bounds)
    trails_df = _clamp_df(trails_df, bounds)

    # team filter
    if team_filter != "both":
        keep = "Offense" if team_filter == "offense" else "Defense"
        df_frame = df_frame[df_frame["team"] == keep]
        if trails_df is not None:
            trails_df = trails_df[trails_df["team"] == keep]

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

    # players
    if show_players and not df_frame.empty:
        color_map = {"Offense": "#e74c3c", "Defense": "#2e86de"}
        for team in ["Offense", "Defense"]:
            sub = df_frame[df_frame["team"] == team]
            if sub.empty:
                continue
            data.append(
                go.Scatter(
                    x=sub["x"], y=sub["y"], mode="markers+text",
                    marker=dict(size=12, color=color_map[team], line=dict(width=1, color="white")),
                    text=sub["player_id"], textposition="middle center", name=team,
                )
            )

    # voronoi
    if show_voronoi and len(df_frame) >= 2:
        positions = df_frame[["x", "y"]].values.tolist()
        vor = compute_voronoi(positions, bounds)
        palette = [
            "rgba(231, 76, 60, 0.18)","rgba(46, 134, 222, 0.18)","rgba(39, 174, 96, 0.18)",
            "rgba(241, 196, 15, 0.18)","rgba(155, 89, 182, 0.18)","rgba(26, 188, 156, 0.18)",
        ]
        for idx, poly in vor.items():
            if not poly:
                continue
            xs, ys = zip(*poly)
            xs, ys = list(xs) + [xs[0]], list(ys) + [ys[0]]
            data.append(
                go.Scatter(
                    x=xs, y=ys, fill="toself", fillcolor=palette[idx % len(palette)],
                    line=dict(color="rgba(0,0,0,0.12)", width=1), hoverinfo="skip", showlegend=False,
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
        xaxis=dict(range=[bounds["x_min"], bounds["x_max"]], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[bounds["y_min"], bounds["y_max"]], showgrid=False, zeroline=False,
                   visible=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=12, r=12, t=12, b=12),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def make_trails(df: pd.DataFrame, current_ts: int, trail_len: int) -> pd.DataFrame:
    start_ts = max(df["timestamp"].min(), current_ts - trail_len)
    return df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= current_ts)]


# --------------------------------------------------------------------------------------
# UI builders
# --------------------------------------------------------------------------------------

def _aspect_padding_from_bounds(bounds: Dict[str, float]) -> str:
    w = bounds["x_max"] - bounds["x_min"]
    h = bounds["y_max"] - bounds["y_min"]
    pct = (h / w) * 100.0 if w > 0 else 56.25
    return f"{pct:.3f}%"


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
            dcc.Tabs(
                id="bottom-tabs", value="playback",
                children=[
                    dcc.Tab(
                        label="Analysis Playback", value="playback",
                        selected_style={"borderTop": "2px solid #2563eb", "fontWeight": 600},
                        children=[
                            html.Div(
                                [
                                    # Centered control bar with right-aligned readout
                                    html.Div(
                                        [
                                            html.Div(className="sb-controls-spacer"),  # left empty cell
                                            html.Div(
                                                [
                                                    html.Button("⏮ Prev", id="btn-prev", n_clicks=0, className="sb-btn"),
                                                    html.Button("▶ Play", id="btn-play", n_clicks=0, className="sb-btn"),
                                                    html.Button("Next ⏭", id="btn-next", n_clicks=0, className="sb-btn"),
                                                    dcc.Dropdown(
                                                        id="speed-dropdown",
                                                        options=[
                                                            {"label": "0.5×", "value": 2},
                                                            {"label": "1.0×", "value": 1},
                                                            {"label": "2.0×", "value": 0.5},
                                                            {"label": "4.0×", "value": 0.25},
                                                        ],
                                                        value=1,
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
                                                className="sb-ctlbar",  # the pill container
                                            ),
                                            html.Div(id="frame-readout", className="sb-readout"),
                                        ],
                                        className="sb-controls-grid",
                                    ),

                                    # Single scrubber
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
                                ],
                                className="sb-panel",
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Editor / Overlays", value="editor",
                        selected_style={"borderTop": "2px solid #2563eb", "fontWeight": 600},
                        children=[
                            html.Div(
                                [
                                    # Team filter segmented
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

                                    # Overlay chips (with extra "Coverage Control (soon)")
                                    html.Div(
                                        [
                                            html.Div("Overlays:", className="sb-label"),
                                            dcc.Checklist(
                                                id="overlay-options",
                                                options=[
                                                    {"label": "Show Players", "value": "players"},
                                                    {"label": "Show Trails", "value": "trails"},
                                                    {"label": "Show Voronoi", "value": "voronoi"},
                                                    {"label": "Coverage Control (soon)", "value": "coverage", "disabled": True},
                                                    {"label": "Pitch Control (soon)", "value": "pc", "disabled": True},
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
                                className="sb-panel",
                            )
                        ],
                    ),
                ],
            )
        ],
        className="sb-bottom",
    )


# --------------------------------------------------------------------------------------
# App & callbacks
# --------------------------------------------------------------------------------------

def main():
    Path("assets").mkdir(exist_ok=True)
    if not Path(DEFENSIVE_CSV).exists() or not Path(OFFENSIVE_CSV).exists():
        raise FileNotFoundError(f"CSV files not found:\n - {DEFENSIVE_CSV}\n - {OFFENSIVE_CSV}")

    df = load_tracking_data(DEFENSIVE_CSV, OFFENSIVE_CSV)
    timestamps = sorted(df["timestamp"].unique())
    if not timestamps:
        raise RuntimeError("No timestamps found in tracking data.")

    app: Dash = dash.Dash(__name__)
    app.title = "Sunbears Dashboard"

    app.layout = html.Div(
        [
            build_header(),
            html.Div([build_top_row(RINK_BOUNDS), build_bottom_panel(timestamps)], className="sb-container"),
            # timers & state
            dcc.Interval(id="play-interval", interval=100, disabled=True),
            dcc.Store(id="is-playing", data=False),
            dcc.Store(id="timestamps", data=timestamps),
        ],
        className="sb-page",
        style=STYLES["page"],
    )

    # Speed -> interval
    @app.callback(Output("play-interval", "interval"), Input("speed-dropdown", "value"))
    def set_speed(mult):
        return int(100 * mult)  # base 100 ms

    # Playback driver: single source of truth for play/pause/prev/next/loop/end-stop
    @app.callback(
        Output("time-slider-main", "value"),
        Output("is-playing", "data"),
        Output("play-interval", "disabled"),
        Output("btn-play", "children"),
        Input("play-interval", "n_intervals"),
        Input("btn-prev", "n_clicks"),
        Input("btn-next", "n_clicks"),
        Input("btn-play", "n_clicks"),
        State("loop-toggle", "value"),
        State("time-slider-main", "value"),
        State("timestamps", "data"),
        State("is-playing", "data"),
        prevent_initial_call=True,
    )
    def playback_driver(_tick, _prev, _next, _play_clicks,
                        loop_vals, cur_idx, ts_list, is_playing):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        trigger = ctx.triggered[0]["prop_id"].split(".")[0]

        total = len(ts_list)
        last = total - 1
        loop = "loop" in (loop_vals or [])

        def label(paused: bool) -> str:
            return "⏸ Pause" if paused else "▶ Play"

        # Play/Pause button toggled
        if trigger == "btn-play":
            new_playing = not bool(is_playing)
            new_idx = cur_idx
            # if starting from end, restart at beginning
            if new_playing and cur_idx >= last:
                new_idx = 0
            return new_idx, new_playing, (not new_playing), label(new_playing)

        # Previous frame
        if trigger == "btn-prev":
            if loop and cur_idx == 0:
                new_idx = last
            else:
                new_idx = max(0, cur_idx - 1)
            return new_idx, is_playing, (not is_playing), label(is_playing)

        # Next frame
        if trigger == "btn-next":
            if loop and cur_idx == last:
                new_idx = 0
            else:
                new_idx = min(last, cur_idx + 1)
            return new_idx, is_playing, (not is_playing), label(is_playing)

        # Timer tick during playback
        if trigger == "play-interval":
            if not is_playing:
                raise dash.exceptions.PreventUpdate
            nxt = cur_idx + 1
            if nxt > last:
                if loop:
                    return 0, True, False, "⏸ Pause"  # wrap & keep playing
                else:
                    return last, False, True, "▶ Play"  # stop at end, show Play
            return nxt, True, False, "⏸ Pause"

        raise dash.exceptions.PreventUpdate

    # Frame readout (top-right)
    @app.callback(Output("frame-readout", "children"), Input("time-slider-main", "value"), State("timestamps", "data"))
    def update_readout(idx, ts_list):
        return f"Frame {idx} / {len(ts_list) - 1}"

    # Graph update
    @app.callback(
        Output("tracking-graph", "figure"),
        Input("time-slider-main", "value"),
        Input("overlay-options", "value"),
        Input("team-filter", "value"),
    )
    def update_figure(time_index: int, overlay_values, team_filter):
        current_timestamp = timestamps[time_index]
        df_frame = df[df["timestamp"] == current_timestamp]
        show_players = "players" in overlay_values
        show_voronoi = "voronoi" in overlay_values
        show_trails = "trails" in overlay_values
        trails_df = make_trails(df, current_timestamp, TRAIL_DEFAULT) if show_trails else None

        return build_tracking_figure(
            df_frame=df_frame,
            bounds=RINK_BOUNDS,
            team_filter=team_filter,
            show_players=show_players,
            show_voronoi=show_voronoi,
            show_trails=show_trails,
            trails_df=trails_df,
        )

    # Reset editor to defaults
    @app.callback(
        Output("team-filter", "value"),
        Output("overlay-options", "value"),
        Input("btn-reset-editor", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_editor(_n):
        return "both", ["players", "voronoi"]

    app.run(debug=True)


if __name__ == "__main__":
    main()
