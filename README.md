# Sports Analytics Dashboard (Python Dash Prototype)

This repository contains a very simple proof‑of‑concept for a sports analytics dashboard built
entirely in Python using [Dash](https://dash.plotly.com/).  It is intended for quick prototyping and
does not require any knowledge of JavaScript, HTML or CSS.  The goal of this app is to
demonstrate how player tracking data can be visualised and interactively explored in a browser.

## Overview

The prototype loads a short tracking dataset and plots the positions of players on a soccer
pitch.  It provides a time slider to move through frames and checkboxes to toggle the
display of player markers and their Voronoi partitions.  The Voronoi computation is
implemented in pure Python based on the algorithm used in the original
`ice_hockey_simulator` project【124951572244772†L0-L40】.  Although there is an element for video
playback in the layout, no sample video is provided; you can supply your own MP4 file by
placing it in the `assets/` folder and adjusting the filename in `app.py`.

This code is provided as a starting point.  Feel free to modify it to suit your needs –
add more overlays, compute density‑weighted pitch control, or integrate with more
comprehensive back‑end analytics.  For a more scalable and polished solution, refer to the
plan using React and FastAPI in the accompanying report.

## Getting Started

1. **Install Dependencies**

   The app depends on a few Python libraries.  You can install them by running:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

   The main dependencies are:

   - `dash` – a Python framework for building interactive web applications【991642812866752†L106-L117】.
   - `plotly` – provides the graphing backend used by Dash.
   - `pandas` and `numpy` for data manipulation.
   - `scipy` and `shapely` for Voronoi computation and polygon clipping.

2. **Run the App**

   To start the development server, execute:

   ```bash
   python app.py
   ```

   Dash will print a local URL (typically `http://127.0.0.1:8050`).  Open it in your
   browser to view the dashboard.  Use the slider to scrub through time and the checkboxes
   to toggle overlays.

3. **Data**

   A small synthetic tracking dataset is included as `sample_tracking.csv`.  Each row
   corresponds to a single player at a given timestamp.  The columns are:

   - `timestamp`: integer frame index
   - `player_id`: string identifier for a player
   - `team`: team name (e.g. "A" or "B")
   - `x`, `y`: position coordinates in metres on a 105 × 68 m pitch

   You can replace this file with your own tracking data.  Ensure that the format is
   preserved and that all coordinates fall within the field bounds defined in `app.py`.

4. **Adding a Video**

   If you have a video clip that corresponds to your tracking data, copy the MP4 file
   into the `assets/` folder and update the `VIDEO_FILENAME` constant in `app.py` to the
   correct filename.  Dash automatically serves files located in the `assets/` directory.

## Repository Structure

```
sport_dashboard_analytics/
├── app.py                 # Main Dash application
├── requirements.txt        # Python dependencies
├── sample_tracking.csv      # Example tracking data
├── utils.py                # Voronoi computation functions
├── assets/
│   └── (optional video)    # Place your video here (e.g. sample.mp4)
└── README.md
```

## Limitations

This is a bare‑bones prototype and has several limitations:

1. **No real video overlay** – The player markers and Voronoi partitions are drawn on a
   separate Plotly graph, not over the video.  Combining a video element with a canvas for
   real‑time overlays is possible (see the MDN guide on drawing video to a canvas【140129824977829†L108-L113】), but it is beyond the scope of this simplified POC.
2. **No density weighting or pitch control** – Only geometric Voronoi partitions are
   computed.  In the full project, pitch control is calculated by weighting regions with a
   density field and computing density‑weighted centroids【124951572244772†L42-L79】.  You can
   extend `utils.py` to include those calculations.
3. **Synthetic data only** – The included tracking data is randomly generated.  Replace it
   with real tracking data to obtain meaningful visualisations.

Despite these limitations, this prototype demonstrates how you can quickly build
interactive dashboards in Python and lays the groundwork for more advanced features.

## License

This project is provided under the MIT License.  See the LICENSE file in the
`ice_hockey_simulator` repository for more details.