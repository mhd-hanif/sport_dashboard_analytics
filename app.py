"""
Dash application for a sports analytics proof‑of‑concept.

This script launches a simple web dashboard using the Dash framework.  It loads
tracking data from a CSV file and displays the positions of players on a
soccer pitch.  A time slider allows the user to move between frames.  A set
of checkboxes toggles the display of player markers and the Voronoi
partition associated with the current positions.  The Voronoi computation is
performed by functions in ``utils.py``, adapted from the original
``ice_hockey_simulator`` project【124951572244772†L0-L40】.

Running this script will start a local development server.  Open the printed
URL in your web browser to interact with the dashboard.
"""

import os
import pandas as pd
import numpy as np
import dash
from dash import html, dcc, Dash
from dash.dependencies import Input, Output
import plotly.graph_objs as go

from utils import compute_voronoi

# Constants defining the field dimensions (in metres)
FIELD_BOUNDS = {
    "x_min": 0.0,
    "x_max": 105.0,
    "y_min": 0.0,
    "y_max": 68.0,
}

# Name of the MP4 file placed in the assets folder.  Replace this with your
# own video file if desired.  If the file does not exist, the video
# component will simply not render anything.
VIDEO_FILENAME = "sample_video.mp4"

DATA_FILE = "sample_tracking.csv"


def load_tracking_data(path: str) -> pd.DataFrame:
    """Load tracking data from a CSV file.

    The expected columns are: timestamp (int), player_id (str), team (str),
    x (float) and y (float).  Additional columns are ignored.

    Args:
        path: Path to the CSV file.

    Returns:
        A pandas DataFrame.
    """
    df = pd.read_csv(path)
    required_cols = {"timestamp", "player_id", "team", "x", "y"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in tracking data: {missing}")
    return df


def create_pitch_shapes() -> list:
    """Return a list of Plotly shape definitions representing a soccer pitch.

    The pitch is drawn with a 105 × 68 metre dimension.  Shapes include
    the outer boundaries and the penalty boxes.  Colours and line widths
    are chosen for clarity.
    """
    shapes = []
    # Outer rectangle
    shapes.append(
        dict(
            type="rect",
            x0=FIELD_BOUNDS["x_min"],
            x1=FIELD_BOUNDS["x_max"],
            y0=FIELD_BOUNDS["y_min"],
            y1=FIELD_BOUNDS["y_max"],
            line=dict(color="black", width=2),
            fillcolor="rgba(0, 128, 0, 0.05)",
        )
    )
    # Midfield line
    shapes.append(
        dict(
            type="line",
            x0=FIELD_BOUNDS["x_min"],
            x1=FIELD_BOUNDS["x_max"],
            y0=FIELD_BOUNDS["y_max"] / 2,
            y1=FIELD_BOUNDS["y_max"] / 2,
            line=dict(color="black", width=1),
        )
    )
    # Penalty areas (assuming standard soccer dimensions: 16.5 m from each end, 40.3 m wide)
    penalty_width = 40.3
    penalty_depth = 16.5
    shapes.append(
        dict(
            type="rect",
            x0=FIELD_BOUNDS["x_min"],
            x1=FIELD_BOUNDS["x_min"] + penalty_depth,
            y0=(FIELD_BOUNDS["y_max"] - penalty_width) / 2,
            y1=(FIELD_BOUNDS["y_max"] + penalty_width) / 2,
            line=dict(color="black", width=1),
        )
    )
    shapes.append(
        dict(
            type="rect",
            x0=FIELD_BOUNDS["x_max"] - penalty_depth,
            x1=FIELD_BOUNDS["x_max"],
            y0=(FIELD_BOUNDS["y_max"] - penalty_width) / 2,
            y1=(FIELD_BOUNDS["y_max"] + penalty_width) / 2,
            line=dict(color="black", width=1),
        )
    )
    return shapes


def build_figure(df_frame: pd.DataFrame, show_players: bool, show_voronoi: bool) -> go.Figure:
    """Construct a Plotly figure for the given frame.

    Args:
        df_frame: DataFrame containing tracking data for a single timestamp.
        show_players: Whether to draw player markers and labels.
        show_voronoi: Whether to overlay Voronoi polygons around players.

    Returns:
        A Plotly ``Figure`` object.
    """
    # Base scatter traces for players
    data = []
    if show_players:
        teams = df_frame["team"].unique()
        colors = {team: "blue" if i == 0 else "red" for i, team in enumerate(teams)}
        for team in teams:
            team_df = df_frame[df_frame["team"] == team]
            data.append(
                go.Scatter(
                    x=team_df["x"],
                    y=team_df["y"],
                    mode="markers+text",
                    marker=dict(size=12, color=colors[team], opacity=0.9),
                    text=team_df["player_id"],
                    textposition="middle center",
                    name=f"Team {team}",
                    hoverinfo="text",
                )
            )
    # Voronoi overlay
    if show_voronoi and len(df_frame) >= 2:
        positions = df_frame[["x", "y"]].values.tolist()
        voronoi_regions = compute_voronoi(positions, FIELD_BOUNDS)
        # Use a colormap to distinguish cells
        palette = [
            "rgba(255, 0, 0, 0.2)",  # light red
            "rgba(0, 0, 255, 0.2)",  # light blue
            "rgba(0, 255, 0, 0.2)",  # light green
            "rgba(255, 255, 0, 0.2)",  # light yellow
            "rgba(255, 0, 255, 0.2)",  # magenta
            "rgba(0, 255, 255, 0.2)",  # cyan
        ]
        for idx, polygon in voronoi_regions.items():
            if not polygon:
                continue
            xs, ys = zip(*polygon)
            # Close the polygon by repeating the first point
            xs = list(xs) + [xs[0]]
            ys = list(ys) + [ys[0]]
            data.append(
                go.Scatter(
                    x=xs,
                    y=ys,
                    fill="toself",
                    fillcolor=palette[idx % len(palette)],
                    line=dict(color="rgba(0,0,0,0.1)", width=1),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    # Assemble the figure
    fig = go.Figure(data=data)
    # Add pitch shapes
    fig.update_layout(
        shapes=create_pitch_shapes(),
        xaxis=dict(
            range=[FIELD_BOUNDS["x_min"], FIELD_BOUNDS["x_max"]],
            showgrid=False,
            zeroline=False,
            visible=False,
        ),
        yaxis=dict(
            range=[FIELD_BOUNDS["y_min"], FIELD_BOUNDS["y_max"]],
            showgrid=False,
            zeroline=False,
            visible=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="white",
        hovermode="closest",
    )
    return fig


def main():
    # Load tracking data
    df = load_tracking_data(DATA_FILE)
    timestamps = sorted(df["timestamp"].unique())

    # Initialise Dash app
    app: Dash = dash.Dash(__name__)
    app.title = "Sports Analytics POC (Dash)"

    # Build the layout
    app.layout = html.Div(
        [
            html.H2("Sports Analytics Dashboard (Python POC)"),
            # Video player (optional)
            html.Div(
                [
                    html.Video(
                        id="video-player",
                        src=f"/assets/{VIDEO_FILENAME}",
                        controls=True,
                        style={"width": "100%", "maxWidth": "640px"},
                    )
                ],
                style={"textAlign": "center", "marginBottom": "1em"},
            ),
            # Controls panel
            html.Div(
                [
                    html.Label("Frame"),
                    dcc.Slider(
                        id="time-slider",
                        min=0,
                        max=len(timestamps) - 1,
                        value=0,
                        marks={i: str(ts) for i, ts in enumerate(timestamps)},
                        step=1,
                        updatemode="drag",
                    ),
                    html.Br(),
                    dcc.Checklist(
                        id="overlay-options",
                        options=[
                            {"label": "Show Players", "value": "players"},
                            {"label": "Show Voronoi", "value": "voronoi"},
                        ],
                        value=["players"],
                        labelStyle={"display": "block"},
                    ),
                ],
                style={"width": "100%", "maxWidth": "400px", "margin": "0 auto"},
            ),
            # Graph
            dcc.Graph(id="tracking-graph"),
        ],
        style={"fontFamily": "Arial, sans-serif", "padding": "1em"},
    )

    # Callback to update the graph
    @app.callback(
        Output("tracking-graph", "figure"),
        [Input("time-slider", "value"), Input("overlay-options", "value")],
    )
    def update_figure(time_index: int, overlay_values):
        current_timestamp = timestamps[time_index]
        df_frame = df[df["timestamp"] == current_timestamp]
        show_players = "players" in overlay_values
        show_voronoi = "voronoi" in overlay_values
        return build_figure(df_frame, show_players, show_voronoi)

    # Run server
    app.run(debug=True)


if __name__ == "__main__":
    main()