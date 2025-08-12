"""
Sunbears Sports Analytics Dashboard (Dash)

- Loads hockey tracking from two CSVs (Defense/Offense):
  columns: timeframe, player_id, x, y  (timeframe -> timestamp)
- Side-by-side: Digital Tracking (Plotly) + Video
- Bottom tabs: Playback + Editor/Overlays
- Animation (Play/Pause/Prev/Next/Speed), team filter, trails, Voronoi
- Rink drawn via background image stretched to tracking bounds
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

# Small shared styles that don't need media queries
STYLES: Dict[str, Any] = {
    "page": {"background": "#f6f7fb", "fontFamily": "Inter, Segoe UI, Arial, sans-serif"},
    "card": {"background": "#ffffff", "borderRadius": "14px", "boxShadow": "0 8px 20px rgba(0,0,0,0.06)", "padding": "10px"},
    "card_msg": {"display": "flex", "alignItems": "center", "justifyContent": "center", "color": "#6b7280"},
    "controls_row": {"display": "flex", "gap": "10px", "alignItems": "center"},
    "controls_button": {"padding": "8px 12px", "borderRadius": "10px"},
}


# --------------------------------------------------------------------------------------
# Data Loading
# --------------------------------------------------------------------------------------

def load_tracking_data(def_path: str, off_path: str) -> pd.DataFrame:
    """
    Read both CSVs, normalize columns, add team labels, sort by time.
    Required columns (after normalization): timestamp, player_id, x, y
    """
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
# Figure Builders
# --------------------------------------------------------------------------------------

def _clamp_df(df_: Optional[pd.DataFrame], bounds: Dict[str, float]) -> Optional[pd.DataFrame]:
    """Display-only clamp to keep markers inside bounds if slightly out."""
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
    """Build the Plotly figure for the current frame and overlay settings."""
    data: List[go.Scatter] = []

    df_frame = _clamp_df(df_frame, bounds)
    trails_df = _clamp_df(trails_df, bounds)

    # Team filter
    if team_filter != "both":
        keep = "Offense" if team_filter == "offense" else "Defense"
        df_frame = df_frame[df_frame["team"] == keep]
        if trails_df is not None:
            trails_df = trails_df[trails_df["team"] == keep]

    # Trails
    if show_trails and trails_df is not None and not trails_df.empty:
        for (_, _pid), seg in trails_df.groupby(["team", "player_id"]):
            data.append(
                go.Scatter(
                    x=seg["x"],
                    y=seg["y"],
                    mode="lines",
                    line=dict(width=2),
                    opacity=0.35,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # Players
    if show_players and not df_frame.empty:
        color_map = {"Offense": "#e74c3c", "Defense": "#2e86de"}
        for team in ["Offense", "Defense"]:
            sub = df_frame[df_frame["team"] == team]
            if sub.empty:
                continue
            data.append(
                go.Scatter(
                    x=sub["x"],
                    y=sub["y"],
                    mode="markers+text",
                    marker=dict(size=12, color=color_map[team], line=dict(width=1, color="white")),
                    text=sub["player_id"],
                    textposition="middle center",
                    name=team,
                )
            )

    # Voronoi
    if show_voronoi and len(df_frame) >= 2:
        positions = df_frame[["x", "y"]].values.tolist()
        vor = compute_voronoi(positions, bounds)
        palette = [
            "rgba(231, 76, 60, 0.18)",   # red
            "rgba(46, 134, 222, 0.18)",  # blue
            "rgba(39, 174, 96, 0.18)",   # green
            "rgba(241, 196, 15, 0.18)",  # yellow
            "rgba(155, 89, 182, 0.18)",  # purple
            "rgba(26, 188, 156, 0.18)",  # teal
        ]
        for idx, poly in vor.items():
            if not poly:
                continue
            xs, ys = zip(*poly)
            xs, ys = list(xs) + [xs[0]], list(ys) + [ys[0]]
            data.append(
                go.Scatter(
                    x=xs,
                    y=ys,
                    fill="toself",
                    fillcolor=palette[idx % len(palette)],
                    line=dict(color="rgba(0,0,0,0.12)", width=1),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # Base figure + rink image
    fig = go.Figure(data=data)
    fig.add_layout_image(
        dict(
            source=f"/assets/{FIELD_IMAGE}",
            xref="x",
            yref="y",
            x=bounds["x_min"],
            y=bounds["y_max"],  # top-left corner in data coords
            sizex=bounds["x_max"] - bounds["x_min"],
            sizey=bounds["y_max"] - bounds["y_min"],
            sizing="stretch",
            layer="below",
            opacity=1.0,
        )
    )

    fig.update_layout(
        autosize=True,  # allow Plotly to resize with container
        xaxis=dict(range=[bounds["x_min"], bounds["x_max"]], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(
            range=[bounds["y_min"], bounds["y_max"]],
            showgrid=False,
            zeroline=False,
            visible=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        margin=dict(l=12, r=12, t=12, b=12),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def make_trails(df: pd.DataFrame, current_ts: int, trail_len: int) -> pd.DataFrame:
    """Return a slice of df within [current_ts - trail_len, current_ts]."""
    start_ts = max(df["timestamp"].min(), current_ts - trail_len)
    return df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= current_ts)]


# --------------------------------------------------------------------------------------
# UI Builders
# --------------------------------------------------------------------------------------

def _aspect_padding_from_bounds(bounds: Dict[str, float]) -> str:
    """Return percentage string for padding-top that matches rink aspect (height/width * 100%)."""
    w = bounds["x_max"] - bounds["x_min"]
    h = bounds["y_max"] - bounds["y_min"]
    pct = (h / w) * 100.0 if w > 0 else 56.25  # default ~16:9 if something odd
    return f"{pct:.3f}%"


def build_header() -> html.Div:
    """Header with bigger, centered icon and left-aligned texts, inside a centered container."""
    return html.Div(
        [
            html.Div(
                [
                    html.Img(src=f"/assets/{ICON_IMAGE}", className="sb-header__icon"),
                    html.Div(
                        [
                            html.H2("Sunbears Dashboard", className="sb-header__title"),
                            html.Div("Digital Tracking • Analysis • Playback", className="sb-header__subtitle"),
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
    """
    Top row with tracking graph and (optional) video.
    Both sides sit inside identical aspect-ratio boxes so they resize proportionally together.
    """
    # shared aspect ratio (height/width) taken from rink bounds
    ar_padding = _aspect_padding_from_bounds(bounds)

    video_path = Path("assets") / VIDEO_FILENAME
    right_panel_child: html.Component = (
        html.Video(id="video-player", src=f"/assets/{VIDEO_FILENAME}", controls=True, className="sb-video")
        if video_path.exists()
        else html.Div(f"Place a video at ./assets/{VIDEO_FILENAME}", className="sb-placeholder")
    )

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [dcc.Graph(id="tracking-graph", className="sb-graph", config={"responsive": True})],
                        className="sb-media__content",
                    )
                ],
                className="sb-media",
                style={"--ar": ar_padding},  # same aspect for both sides
            ),
            html.Div(
                [
                    html.Div([right_panel_child], className="sb-media__content")
                ],
                className="sb-media",
                style={"--ar": ar_padding},  # same aspect for both sides
            ),
        ],
        className="sb-grid-2col",
    )


def build_bottom_panel(timestamps: list[int]) -> html.Div:
    """Bottom tabs with playback controls and overlay editor, centered with container padding."""
    return html.Div(
        [
            dcc.Tabs(
                id="bottom-tabs",
                value="playback",
                children=[
                    dcc.Tab(
                        label="Analysis Playback",
                        value="playback",
                        selected_style={"borderTop": "2px solid #2563eb", "fontWeight": 600},
                        children=[
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Button("⏮ Prev", id="btn-prev", n_clicks=0, style=STYLES["controls_button"]),
                                            html.Button("▶ Play", id="btn-play", n_clicks=0, style=STYLES["controls_button"]),
                                            html.Button("Next ⏭", id="btn-next", n_clicks=0, style=STYLES["controls_button"]),
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
                                                style={"width": "120px", "marginLeft": "12px"},
                                            ),
                                        ],
                                        style=STYLES["controls_row"],
                                    ),
                                    html.Div(
                                        dcc.Slider(
                                            id="time-slider",
                                            min=0,
                                            max=len(timestamps) - 1,
                                            value=0,
                                            step=1,
                                            updatemode="drag",
                                        ),
                                        style={"marginTop": "14px"},
                                    ),
                                ],
                                className="sb-card sb-card--padded",
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Editor / Overlays",
                        value="editor",
                        selected_style={"borderTop": "2px solid #2563eb", "fontWeight": 600},
                        children=[
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("Show team:", style={"fontWeight": 600}),
                                            dcc.RadioItems(
                                                id="team-filter",
                                                options=[
                                                    {"label": "Both", "value": "both"},
                                                    {"label": "Offense", "value": "offense"},
                                                    {"label": "Defense", "value": "defense"},
                                                ],
                                                value="both",
                                                inline=True,
                                            ),
                                        ],
                                        style={"marginBottom": "10px"},
                                    ),
                                    dcc.Checklist(
                                        id="overlay-options",
                                        options=[
                                            {"label": "Show Players", "value": "players"},
                                            {"label": "Show Trails", "value": "trails"},
                                            {"label": "Show Voronoi", "value": "voronoi"},
                                            {"label": "Pitch Control (coming soon)", "value": "pc"},
                                            {"label": "EPV / xT (coming soon)", "value": "epvxt"},
                                        ],
                                        value=["players"],
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Trail length (frames):"),
                                            dcc.Slider(id="trail-len", min=5, max=200, step=5, value=40),
                                        ],
                                        style={"marginTop": "12px"},
                                    ),
                                ],
                                className="sb-card sb-card--padded",
                            )
                        ],
                    ),
                ],
            )
        ],
        className="sb-bottom",
    )


# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------

def main():
    # Ensure required files/folders exist
    Path("assets").mkdir(exist_ok=True)
    if not Path(DEFENSIVE_CSV).exists() or not Path(OFFENSIVE_CSV).exists():
        raise FileNotFoundError(f"CSV files not found:\n - {DEFENSIVE_CSV}\n - {OFFENSIVE_CSV}")

    # Load data
    df = load_tracking_data(DEFENSIVE_CSV, OFFENSIVE_CSV)
    timestamps = sorted(df["timestamp"].unique())
    if not timestamps:
        raise RuntimeError("No timestamps found in tracking data.")

    # App
    app: Dash = dash.Dash(__name__)
    app.title = "Sunbears Dashboard"

    # Layout
    app.layout = html.Div(
        [
            # Header (full-width) with centered inner container
            build_header(),

            # Main centered container
            html.Div(
                [
                    build_top_row(RINK_BOUNDS),
                    build_bottom_panel(timestamps),
                ],
                className="sb-container",
            ),

            # State + timer
            dcc.Interval(id="play-interval", interval=100, disabled=True),
            dcc.Store(id="is-playing", data=False),
            dcc.Store(id="timestamps", data=timestamps),
        ],
        className="sb-page",
        style=STYLES["page"],
    )

    # ----------------------------------------------------------------------------------
    # Callbacks
    # ----------------------------------------------------------------------------------

    @app.callback(
        Output("is-playing", "data"),
        Output("play-interval", "disabled"),
        Output("btn-play", "children"),
        Input("btn-play", "n_clicks"),
        State("is-playing", "data"),
        prevent_initial_call=True,
    )
    def toggle_play(n_clicks, is_playing):
        new_state = not bool(is_playing)
        return new_state, (not new_state), ("⏸ Pause" if new_state else "▶ Play")

    @app.callback(
        Output("time-slider", "value"),
        Input("play-interval", "n_intervals"),
        Input("btn-prev", "n_clicks"),
        Input("btn-next", "n_clicks"),
        Input("speed-dropdown", "value"),
        State("time-slider", "value"),
        State("timestamps", "data"),
        prevent_initial_call=True,
    )
    def advance_frame(n_intervals, n_prev, n_next, speed_val, current_idx, ts_list):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        trigger = ctx.triggered[0]["prop_id"].split(".")[0]
        max_idx = len(ts_list) - 1

        if trigger == "btn-prev":
            return max(0, current_idx - 1)
        if trigger == "btn-next":
            return min(max_idx, current_idx + 1)
        if trigger == "speed-dropdown":
            return current_idx  # speed changes do not move the slider

        # interval tick
        nxt = current_idx + 1
        return 0 if nxt > max_idx else nxt

    @app.callback(Output("play-interval", "interval"), Input("speed-dropdown", "value"))
    def set_speed(mult):
        # Base 100 ms, divided/multiplied by selected factor
        return int(100 * mult)

    @app.callback(
        Output("tracking-graph", "figure"),
        Input("time-slider", "value"),
        Input("overlay-options", "value"),
        Input("team-filter", "value"),
        State("trail-len", "value"),
    )
    def update_figure(time_index: int, overlay_values, team_filter, trail_len):
        current_timestamp = timestamps[time_index]
        df_frame = df[df["timestamp"] == current_timestamp]
        show_players = "players" in overlay_values
        show_voronoi = "voronoi" in overlay_values
        show_trails = "trails" in overlay_values
        trails_df = make_trails(df, current_timestamp, trail_len) if show_trails else None

        return build_tracking_figure(
            df_frame=df_frame,
            bounds=RINK_BOUNDS,
            team_filter=team_filter,
            show_players=show_players,
            show_voronoi=show_voronoi,
            show_trails=show_trails,
            trails_df=trails_df,
        )

    app.run(debug=True)


if __name__ == "__main__":
    main()
