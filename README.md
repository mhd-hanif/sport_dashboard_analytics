# Sports Analytics Dashboard (Python Dash Prototype)

This repository contains a very simple proof‑of‑concept for a sports analytics dashboard built
entirely in Python using [Dash](https://dash.plotly.com/).  It is intended for quick prototyping and
does not require any knowledge of JavaScript, HTML or CSS.  The goal of this app is to
demonstrate how player tracking data can be visualised and interactively explored in a browser.

## Overview

The dashboard loads one or more CSV files containing player tracking
data and plots the positions of players on a hockey rink.  A slider
allows the user to step through frames, and checkboxes toggle the
display of player markers, trail lines and Voronoi partitions.  The
Voronoi computation is implemented in pure Python using NumPy,
SciPy and Shapely (see `src/utils/geometry.py`).  All static assets
(images, videos) are served automatically from the `assets/` folder by
Dash【37895621177989†L68-L71】.

The structure of this repository is organised into a `src/` package.  Each module has a single
responsibility:

* `config.py` stores constants such as file names and rink bounds.
* `data_loader.py` defines functions for reading and normalising
  tracking CSV files.
* `figure.py` builds the Plotly figure given a frame of data.
* `layout.py` builds the Dash layout using reusable UI components.
* `callbacks.py` contains all Dash callbacks, separated from layout
  definitions for clarity.
* `utils/geometry.py` implements the Voronoi computation and polygon
  clipping.
* `main.py` acts as the entry point: it loads data, creates the Dash
  app, registers the layout and callbacks, and runs the server.


## Getting Started

### 1. Install dependencies

It is recommended to use a virtual environment when running the
application.  From the project root run:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The required packages include Dash, Plotly, Pandas, NumPy, SciPy and
Shapely.

### 2. Prepare your data

A small synthetic tracking dataset is provided in
`data/sample_tracking.csv`.  The CSV must contain at least the columns
`timestamp`, `player_id`, `team`, `x` and `y`【37895621177989†L55-L64】.  If you have
your own tracking files you can replace the sample file or add
additional CSVs – simply update the `DATA_FILES` variable in
`src/config.py` to point to the appropriate paths.

### 3. Run the app

To start the development server run:

```bash
python -m src.main
```

Dash will print a local URL (typically `http://127.0.0.1:8050`) which
you can open in a browser.  Use the slider to move through time and
the checkboxes and radio buttons to toggle overlays and filter by team.

## Project Structure

```
sport_dashboard_analytics/
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── data/
│   └── sample_tracking.csv # Example tracking data
├── assets/
│   └── (optional files)    # Place your images/videos here (e.g. rink image)
└── src/
    ├── __init__.py         # Makes src a package
    ├── config.py           # Constants and configuration
    ├── data_loader.py      # Data loading utilities
    ├── figure.py           # Plotly figure builder
    ├── layout.py           # Dash layout components
    ├── callbacks.py        # Dash callback functions
    └── utils/
        ├── __init__.py
        └── geometry.py     # Voronoi computation and clipping
```