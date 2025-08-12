"""
Sunbears Sports Analytics Dashboard (Dash)
- Loads hockey tracking from two CSVs (Defense/Offense) with columns:
  timeframe, player_id, x, y
- Side-by-side: Digital Tracking (Plotly) + Video
- Bottom tabs for Playback and Editor/Overlays
- Frame-by-frame animation with Play/Pause/Prev/Next/Speed
- Team filter, trails, Voronoi overlay
"""

from pathlib import Path
import pandas as pd
import dash
from dash import html, dcc, Dash
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

from utils import compute_voronoi, create_rink_shapes

# ---------- Files ----------
# Put your CSVs in the project root (or change these paths)
DEFENSIVE_CSV = "assets/defensive_players_hockey.csv"
OFFENSIVE_CSV = "assets/offensive_players_hockey.csv"

# Place any mp4 in ./assets and set filename here (optional)
VIDEO_FILENAME = "sample_video.mp4"

# ---------- Rink bounds (meters, NHL 200x85 ft -> 60.96 x 25.91 m) ----------
RINK_BOUNDS = {"x_min": 0.0, "x_max": 60.96, "y_min": 0.0, "y_max": 25.91}


# -------------------------- Data loading --------------------------
def load_tracking_data(def_path: str, off_path: str) -> pd.DataFrame:
    """
    Load Sunbears hockey tracking data from two CSVs.

    Expected columns in each CSV (case-insensitive):
      - timeframe  -> will be renamed to 'timestamp' (int frame index)
      - player_id  -> kept as string
      - x, y       -> floats (meters)
    """
    def _read_and_normalize(p: str, label: str) -> pd.DataFrame:
        df = pd.read_csv(p)
        # Make column matching case-insensitive and whitespace-safe
        df.columns = [c.strip() for c in df.columns]
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
            raise ValueError(f"Missing columns in {label} data: {missing}. "
                             f"Found columns: {list(df.columns)}")

        # Clean dtypes
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


# -------------------------- Figure builder --------------------------
def build_tracking_figure(
    df_frame: pd.DataFrame,
    bounds: dict,
    team_filter: str,
    show_players: bool,
    show_voronoi: bool,
    show_trails: bool,
    trails_df: pd.DataFrame | None,
) -> go.Figure:
    data = []

    # Apply team filter for markers/voronoi
    if team_filter != "both":
        keep = "Offense" if team_filter == "offense" else "Defense"
        df_frame = df_frame[df_frame["team"] == keep]

    # Trails
    if show_trails and trails_df is not None and not trails_df.empty:
        if team_filter != "both":
            keep = "Offense" if team_filter == "offense" else "Defense"
            trails_df = trails_df[trails_df["team"] == keep]
        for (_, pid), seg in trails_df.groupby(["team", "player_id"]):
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

    # Player markers
    if show_players and not df_frame.empty:
        color_map = {"Offense": "red", "Defense": "blue"}
        for team in ["Offense", "Defense"]:
            sub = df_frame[df_frame["team"] == team]
            if sub.empty:
                continue
            data.append(
                go.Scatter(
                    x=sub["x"],
                    y=sub["y"],
                    mode="markers+text",
                    marker=dict(size=12, color=color_map[team], opacity=0.9),
                    text=sub["player_id"],
                    textposition="middle center",
                    name=team,
                    hoverinfo="text",
                )
            )

    # Voronoi overlay
    if show_voronoi and len(df_frame) >= 2:
        positions = df_frame[["x", "y"]].values.tolist()
        vor = compute_voronoi(positions, bounds)
        palette = [
            "rgba(255, 0, 0, 0.2)",
            "rgba(0, 0, 255, 0.2)",
            "rgba(0, 255, 0, 0.2)",
            "rgba(255, 255, 0, 0.2)",
            "rgba(255, 0, 255, 0.2)",
            "rgba(0, 255, 255, 0.2)",
        ]
        for idx, poly in vor.items():
            if not poly:
                continue
            xs, ys = zip(*poly)
            xs = list(xs) + [xs[0]]
            ys = list(ys) + [ys[0]]
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

    # Layout (rink + equal aspect)
    fig = go.Figure(data=data)
    fig.update_layout(
        shapes=create_rink_shapes(RINK_BOUNDS),
        xaxis=dict(range=[bounds["x_min"], bounds["x_max"]], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[bounds["y_min"], bounds["y_max"]], showgrid=False, zeroline=False, visible=False,
                   scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        hovermode="closest",
    )
    return fig


def make_trails(df: pd.DataFrame, current_ts: int, trail_len: int) -> pd.DataFrame:
    """Return subset with frames [current_ts - trail_len, current_ts] per player."""
    start_ts = max(df["timestamp"].min(), current_ts - trail_len)
    return df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= current_ts)]


# -------------------------- App --------------------------
def main():
    # Load data (robust error if files missing)
    if not Path(DEFENSIVE_CSV).exists() or not Path(OFFENSIVE_CSV).exists():
        raise FileNotFoundError(
            f"CSV files not found. Check paths:\n - {DEFENSIVE_CSV}\n - {OFFENSIVE_CSV}"
        )
    df = load_tracking_data(DEFENSIVE_CSV, OFFENSIVE_CSV)
    timestamps = sorted(df["timestamp"].unique())
    if not timestamps:
        raise RuntimeError("No timestamps were found in the tracking data.")

    app: Dash = dash.Dash(__name__)
    app.title = "Sunbears Dashboard"

    # ---------------- Layout ----------------
    app.layout = html.Div(
        [
            html.H2("Sunbears Dashboard", style={"margin": "10px 12px"}),

            # Top row: tracking (left) and video (right)
            html.Div(
                [
                    dcc.Graph(id="tracking-graph", style={"height": "480px", "width": "100%"}),
                    html.Video(
                        id="video-player",
                        src=f"/assets/{VIDEO_FILENAME}",
                        controls=True,
                        style={"height": "480px", "width": "100%", "background": "#000"},
                    ),
                ],
                id="top-row",
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "12px",
                    "alignItems": "start",
                    "padding": "0 12px",
                },
            ),

            # Bottom tabs: Playback / Editor-Overlays
            html.Div(
                [
                    dcc.Tabs(
                        id="bottom-tabs",
                        value="playback",
                        children=[
                            dcc.Tab(
                                label="Analysis Playback",
                                value="playback",
                                children=[
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Button("⏮︎ Prev", id="btn-prev", n_clicks=0, className="sb-btn"),
                                                    html.Button("▶︎ Play", id="btn-play", n_clicks=0, className="sb-btn"),
                                                    html.Button("⏭︎ Next", id="btn-next", n_clicks=0, className="sb-btn"),
                                                    dcc.Dropdown(
                                                        id="speed-dropdown",
                                                        options=[
                                                            {"label": "0.5×", "value": 2},   # slower
                                                            {"label": "1.0×", "value": 1},
                                                            {"label": "2.0×", "value": 0.5}, # faster
                                                            {"label": "4.0×", "value": 0.25},
                                                        ],
                                                        value=1,
                                                        clearable=False,
                                                        style={"width": "110px", "marginLeft": "12px"},
                                                    ),
                                                ],
                                                style={"display": "flex", "gap": "8px", "alignItems": "center"},
                                            ),
                                            html.Div(
                                                [
                                                    dcc.Slider(
                                                        id="time-slider",
                                                        min=0,
                                                        max=len(timestamps) - 1,
                                                        value=0,
                                                        step=1,
                                                        updatemode="drag",
                                                    )
                                                ],
                                                style={"marginTop": "12px"},
                                            ),
                                        ],
                                        style={"padding": "12px"},
                                    )
                                ],
                            ),
                            dcc.Tab(
                                label="Editor / Overlays",
                                value="editor",
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
                                        style={"padding": "12px"},
                                    )
                                ],
                            ),
                        ],
                    )
                ],
                style={"margin": "12px"},
            ),

            # Hidden state / timers
            dcc.Interval(id="play-interval", interval=100, disabled=True),
            dcc.Store(id="is-playing", data=False),
            dcc.Store(id="timestamps", data=timestamps),
        ],
        style={"fontFamily": "Inter, Arial, sans-serif"},
    )

    # ---------------- Callbacks ----------------

    # Toggle play/pause
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
        return new_state, (not new_state), ("⏸︎ Pause" if new_state else "▶︎ Play")

    # Step frames on interval / prev / next / speed change
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
            return current_idx
        # interval tick -> advance (loop)
        nxt = current_idx + 1
        if nxt > max_idx:
            nxt = 0
        return nxt

    # Adjust speed by changing the interval (base 100 ms)
    @app.callback(Output("play-interval", "interval"), Input("speed-dropdown", "value"))
    def set_speed(mult):
        return int(100 * mult)

    # Update figure on any relevant change
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
    # Ensure ./assets exists so the video can be served (even if missing)
    Path("assets").mkdir(exist_ok=True)
    main()
