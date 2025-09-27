# trajectory_sketcher.py
# Draw player trajectories with smooth motion (Catmull–Rom spline) and stops, export to CSV.
# Controls:
#   1–9: select player_id 1..9
#   0  : select player_id 10
#   n/p: next/previous player (wraps 1..MAX_PLAYERS)
#   Click: add point
#   Backspace: undo last point
#   [: decrease speed factor   ]: increase speed factor
#   H: add hold (stop) at last point (+HOLD_FRAMES)
#   T: toggle team (0/1)
#   S: save CSV
#   Q: quit

from typing import Dict, List, Tuple
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ------------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------------
ASSET_IMAGE_PATH: str = "assets/field_hockey.png"

# Rink bounds (meters)
RINK_BOUNDS: Dict[str, float] = {"x_min": 0.0, "x_max": 61.0, "y_min": 0.0, "y_max": 30.0}

# Video timing
VIDEO_FPS: float = 30.0
VIDEO_FRAMES: int = 600  # e.g., 15 s @ 30 fps

# Output CSV
OUT_CSV: str = "scene_5.csv"

# Export orientation fix (UI uses Y-down; your player expects Y-up)
FLIP_Y_FOR_EXPORT: bool = True

# Speed / Hold settings
SPEED_MIN = 0.1
SPEED_MAX = 5.0
SPEED_STEP = 0.1
DEFAULT_SPEED = 1.0
HOLD_FRAMES = 30  # frames to hold each time you press 'H'

# Players
MAX_PLAYERS = 12
# ------------------------------------------------------------------------------------

# Data structure:
# paths[player_id] = {
#   "pts": [(x,y), ...],
#   "team": 0/1,
#   "segment_speeds": [speed_factor, ...],  # per segment (len = len(pts)-1)
#   "holds": {vertex_index: total_hold_frames}
# }
paths: Dict[int, Dict[str, object]] = {}
current_pid: int = 1
current_team: int = 0
current_speed: float = DEFAULT_SPEED

xmin, xmax = RINK_BOUNDS["x_min"], RINK_BOUNDS["x_max"]
ymin, ymax = RINK_BOUNDS["y_min"], RINK_BOUNDS["y_max"]

# ------------------------- spline helpers (Catmull–Rom / Hermite) -------------------
def _catmull_rom_tangents(pts: np.ndarray, holds: Dict[int, int]) -> np.ndarray:
    """
    Compute per-vertex tangents. If a vertex has a hold, force tangent = 0 (stop).
    Endpoints use clamped tangents (zero) for natural ease in/out.
    """
    n = len(pts)
    m = np.zeros_like(pts, dtype=float)
    for i in range(n):
        if i in holds and holds[i] > 0:
            m[i] = 0.0
        elif i == 0 or i == n - 1:
            m[i] = 0.0  # clamp ends
        else:
            m[i] = 0.5 * (pts[i + 1] - pts[i - 1])
    return m

def _hermite(p0, p1, m0, m1, t):
    """Cubic Hermite basis (t in [0,1])."""
    t2 = t * t
    t3 = t2 * t
    h00 = 2*t3 - 3*t2 + 1
    h10 = t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 = t3 - t2
    return h00*p0 + h10*m0 + h01*p1 + h11*m1

def sample_spline_segmented(points: List[Tuple[float, float]],
                            seg_frames: List[int],
                            holds: Dict[int, int]) -> np.ndarray:
    """
    Sample smooth positions along a Catmull–Rom (Hermite) spline with per-segment frame counts.
    Holds are inserted later; here we just generate the moving samples with smooth velocities.
    """
    pts = np.array(points, dtype=float)
    n = len(pts)
    if n == 1:
        return np.repeat(pts, sum(seg_frames), axis=0)

    # per-vertex tangents (force zero where holds exist to create real stops there)
    m = _catmull_rom_tangents(pts, holds)

    samples = [pts[0]]
    for i, nfr in enumerate(seg_frames):
        p0, p1 = pts[i], pts[i + 1]
        m0, m1 = m[i], m[i + 1]
        if nfr <= 1:
            seg_pts = [p1]
        else:
            ts = np.linspace(0.0, 1.0, nfr, endpoint=True)
            seg_pts = [_hermite(p0, p1, m0, m1, t) for t in ts[1:]]  # skip first to avoid dup
        samples.extend(seg_pts)
    return np.array(samples, dtype=float)

# ------------------------- time allocation helpers ----------------------------------
def generate_time_distribution(points: List[Tuple[float, float]],
                               speeds: List[float],
                               holds: Dict[int, int],
                               n_total: int) -> List[int]:
    """Distribute frames across segments by (length / speed), subtracting hold frames."""
    if len(points) < 2:
        return [n_total]
    pts = np.array(points, dtype=float)
    seg_lens = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(axis=1))
    weighted = seg_lens / np.maximum(np.array(speeds, dtype=float), 1e-6)
    total_hold = int(sum(holds.values()))
    travel_frames = max(1, n_total - total_hold)
    total_weight = float(np.sum(weighted)) if np.sum(weighted) > 0 else 1.0
    seg_frames = [max(1, int(round(travel_frames * (w / total_weight)))) for w in weighted]
    return seg_frames

# ----------------------------------- export -----------------------------------------
def export_csv(paths_dict: Dict[int, Dict[str, object]], out_path: str) -> None:
    """
    Export trajectories with schema:
      timeframe,player_id,x,y,team_id,vx,vy
    Smooth motion via spline; true stops at holds; optional Y flip for field coords.
    """
    fps = VIDEO_FPS
    dt = 1.0 / fps
    y_max = RINK_BOUNDS["y_max"]
    rows: List[List[str]] = []

    for pid, info in paths_dict.items():
        pts = info["pts"]            # type: ignore
        team = info["team"]          # type: ignore
        speeds = info["segment_speeds"]  # type: ignore
        holds = info["holds"]        # type: ignore

        if len(pts) < 1:
            continue

        if len(pts) == 1:
            S = np.repeat(np.array(pts, dtype=float), VIDEO_FRAMES, axis=0)
        else:
            seg_frames = generate_time_distribution(pts, speeds, holds, VIDEO_FRAMES)
            S_move = sample_spline_segmented(pts, seg_frames, holds)

            # Insert exact holds at vertex indices (repeat the exact vertex position)
            full_traj = []
            # Map: end of segment i corresponds to vertex i+1
            move_idx = 0
            for i in range(len(seg_frames)):
                # segment i contributes seg_frames[i] - 1 new samples after the initial sample
                nseg = seg_frames[i]
                if i == 0:
                    # include first sample (already in list)
                    segment_chunk = S_move[0 : nseg]
                    move_idx = nseg
                else:
                    segment_chunk = S_move[move_idx : move_idx + nseg - 1]
                    move_idx += (nseg - 1)
                full_traj.extend(segment_chunk)
                vertex_idx = i + 1
                if vertex_idx in holds and holds[vertex_idx] > 0:
                    # repeat the exact vertex position
                    vpos = pts[vertex_idx]
                    full_traj.extend([vpos] * int(holds[vertex_idx]))

            S = np.array(full_traj, dtype=float)

            # Trim / pad to VIDEO_FRAMES
            if len(S) > VIDEO_FRAMES:
                S = S[:VIDEO_FRAMES]
            elif len(S) < VIDEO_FRAMES:
                S = np.vstack([S, np.repeat(S[-1][None, :], VIDEO_FRAMES - len(S), axis=0)])

        # Velocities (then optional Y flip)
        vx = np.zeros(VIDEO_FRAMES)
        vy = np.zeros(VIDEO_FRAMES)
        vx[1:] = (S[1:, 0] - S[:-1, 0]) / dt
        vy[1:] = (S[1:, 1] - S[:-1, 1]) / dt
        vx[0], vy[0] = vx[1], vy[1]

        if FLIP_Y_FOR_EXPORT:
            y_out = y_max - S[:, 1]
            vy_out = -vy
        else:
            y_out = S[:, 1]
            vy_out = vy

        for k in range(VIDEO_FRAMES):
            rows.append([
                k,                 # timeframe
                pid,               # player_id
                f"{S[k, 0]:.3f}",  # x
                f"{y_out[k]:.3f}", # y (flipped if enabled)
                int(team),         # team_id
                f"{vx[k]:.3f}",
                f"{vy_out[k]:.3f}",
            ])

    rows.sort(key=lambda r: (int(r[1]), int(r[0])))

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timeframe", "player_id", "x", "y", "team_id", "vx", "vy"])
        w.writerows(rows)

    print(f"Saved {out_path} (players: {len(paths_dict)})")

# --------------------------------- UI ---------------------------------
img = mpimg.imread(ASSET_IMAGE_PATH)
fig, ax = plt.subplots(figsize=(11, 5.5))

# Y-down UI (image draws naturally)
ax.imshow(img, extent=[xmin, xmax, ymax, ymin])
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymax, ymin)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")

status = ax.text(0.01, 0.02, "", transform=ax.transAxes, color="k",
                 bbox=dict(facecolor="w", alpha=0.75))

def ensure_player(pid: int):
    if pid not in paths:
        paths[pid] = {"pts": [], "team": 0, "segment_speeds": [], "holds": {}}

def refresh_status():
    npts = len(paths.get(current_pid, {"pts": []})["pts"]) if current_pid in paths else 0
    status.set_text(
        f"Player: {current_pid} | Team: {current_team} | Points: {npts} | "
        f"Speed: {current_speed:.1f}x | FPS: {VIDEO_FPS:.1f} | Frames: {VIDEO_FRAMES}"
    )

def redraw():
    ax.clear()
    ax.imshow(img, extent=[xmin, xmax, ymax, ymin])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    for pid, info in paths.items():
        pts = info["pts"]  # type: ignore
        if not pts:
            continue
        xs, ys = zip(*pts)
        color = "C0" if info["team"] == 0 else "C3"  # type: ignore
        ax.plot(xs, ys, "-o", lw=2, ms=4, color=color, alpha=0.95,
                label=f"pid {pid} (team {info['team']})")
    if len(paths) > 0:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.75, ncols=2)
    refresh_status()
    fig.canvas.draw_idle()

def on_click(event):
    global current_pid
    if event.inaxes != ax:
        return
    x, y = event.xdata, event.ydata
    if x is None or y is None:
        return
    ensure_player(current_pid)
    info = paths[current_pid]
    info["pts"].append((float(x), float(y)))
    if len(info["pts"]) > 1:
        info["segment_speeds"].append(current_speed)
    redraw()

def next_player():
    global current_pid, current_team
    current_pid = 1 if current_pid >= MAX_PLAYERS else current_pid + 1
    ensure_player(current_pid)
    current_team = paths[current_pid]["team"]  # type: ignore

def prev_player():
    global current_pid, current_team
    current_pid = MAX_PLAYERS if current_pid <= 1 else current_pid - 1
    ensure_player(current_pid)
    current_team = paths[current_pid]["team"]  # type: ignore

def on_key(event):
    global current_pid, current_team, current_speed
    # Direct number keys: 1..9 -> 1..9, 0 -> 10
    if event.key in list("1234567890"):
        pid = 10 if event.key == "0" else int(event.key)
        if pid <= MAX_PLAYERS:
            current_pid = pid
            ensure_player(current_pid)
            current_team = paths[current_pid]["team"]  # type: ignore
            redraw()
        return

    if event.key == "n":
        next_player(); redraw(); return
    if event.key == "p":
        prev_player(); redraw(); return

    if event.key == "backspace":
        if current_pid in paths and paths[current_pid]["pts"]:  # type: ignore
            paths[current_pid]["pts"].pop()                     # type: ignore
            if paths[current_pid]["segment_speeds"]:            # type: ignore
                paths[current_pid]["segment_speeds"].pop()      # type: ignore
            redraw()
    elif event.key == "t":
        ensure_player(current_pid)
        current_team = 1 - int(paths[current_pid]["team"])      # type: ignore
        paths[current_pid]["team"] = current_team               # type: ignore
        redraw()
    elif event.key == "[":
        current_speed = max(SPEED_MIN, current_speed - SPEED_STEP)
        redraw()
    elif event.key == "]":
        current_speed = min(SPEED_MAX, current_speed + SPEED_STEP)
        redraw()
    elif event.key == "h":
        # Add hold at the last point (true stop)
        if current_pid in paths and paths[current_pid]["pts"]:  # type: ignore
            idx = len(paths[current_pid]["pts"]) - 1            # type: ignore
            holds = paths[current_pid]["holds"]                 # type: ignore
            holds[idx] = holds.get(idx, 0) + HOLD_FRAMES       # type: ignore
            print(f"Added hold at point {idx} of player {current_pid} (+{HOLD_FRAMES} frames)")
            redraw()
    elif event.key == "s":
        export_csv(paths, OUT_CSV)
    elif event.key == "q":
        plt.close(fig)

fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("key_press_event", on_key)

fig.suptitle(
    f"Draw: 1-9 players 1..9, 0 is 10, n/p next/prev (1..{MAX_PLAYERS}) | "
    "Click add point | Backspace undo | [/]: speed +/- | H: hold | T: team | S: save | Q: quit",
    fontsize=10
)
ensure_player(current_pid)
redraw()
plt.show()
