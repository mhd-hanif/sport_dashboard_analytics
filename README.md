# Sunbears Sports Analytics Dashboard (Dash)

Interactive hockey tracking dashboard built with **Python + Dash**.

Displays player positions on a rink background, allows frame-by-frame playback, and supports overlays such as trails and Voronoi regions. Can also display a short synchronized video alongside the tracking plot.

## Features

- **Interactive Playback**
  - Single timeline slider for entire sequence
  - Centered playback controls: Prev / Play / Next / Speed / Loop
  - Frame indicator: `Frame i / N`  
    - Stops at last frame; Play restarts from beginning
- **Overlays**
  - Player markers
  - Fixed-length trails (default length from file)
  - Voronoi regions clipped to rink boundaries
- **Layout**
  - Left: Tracking visualization
  - Right: Video player (optional)
- **Header**
  - Sunbears icon and title
  - Subtitle for description
- **Responsive design** with custom CSS styling

## Repository Structure

- `app.py` — Main Dash application (layout, callbacks, figure rendering)
- `utils.py` — Utility functions (Voronoi computation, polygon clipping)
- `assets/`
  - `app.css` — Custom styling
  - `defensive_players_hockey.csv` — Example defense player tracking
  - `offensive_players_hockey.csv` — Example offense player tracking
  - `field_hockey.png` — Rink background
  - `sunbears_icon.webp` — Dashboard icon
  - `sample_video.mp4` — Example synchronized video (optional)

## Data Format

Two CSV files are expected:

- **Defensive players** — `assets/defensive_players_hockey.csv`
- **Offensive players** — `assets/offensive_players_hockey.csv`

**Required columns:**
- `timeframe` (or `timestamp`)
- `player_id`
- `x`
- `y`

The app will:
- Map `timeframe` → `timestamp` if necessary
- Coerce `timestamp` to `int`
- Coerce `player_id` to `str`
- Coerce `x` and `y` to `float`
- Assign team labels based on the file source (Defense/Offense)

## Running the App

1. Install dependencies:
   ```bash
   pip install dash plotly pandas shapely scipy
   ```
2. Place your CSV files and optional video in the assets/ folder.
3. Run:
   ```bash
   python app.py
   ```
4. Open the local server in your web browser

## Notes

- The Voronoi overlay uses `scipy.spatial.Voronoi` and `shapely` for clipping to rink boundaries.
- Styling is handled automatically by Dash loading `assets/app.css`.
- To change team colors, update the color map in `app.py` or CSS classes in `app.css`.

### Future Development
We are actively developing advanced **spatial analysis** features for player movement, including:

- **Coverage Control** — to evaluate and optimize spatial occupation.
- **Pitch Control** — to assess space influence dynamically.
- **xT (Expected Threat)** and **EPV (Expected Possession Value)** — to quantify scoring potential and decision-making impact.

_Stay tuned — coming soon!_
