"""Scene-level analytics for the dashboard.

``SceneAnalytics`` turns the tracking table of one clip into the quantities the
dashboard shows:

* the pitch-control field of the frame (who gets to a loose puck first),
* the scoring threat of the attack, as a surface and as a number (OBSO),
* a reference defense: a defending unit driven by pitch-control coverage that
  starts where the tracked defenders start and then plays the whole clip
  against the tracked attack,
* suggested positions for the defenders of a single (possibly edited) frame.

The models work in SI units in a canonical rink frame with the defended goal
on the left. The engine infers from the data what the clip does not say:

* Frame rate and velocity units, by comparing the stored velocities with the
  displacement between frames.
* Attack direction, from the side of the rink the defending team stays on.
* Goalies, as players who stay in front of a goal for the whole clip. The
  scoring model already stands for the goalie, so goalies are left out.
* Puck carrier. Tracking clips often have no puck. Unless a carrier is chosen,
  every result is the average over the attackers who may have it.
"""

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analytics.coverage import PitchControlCoverage
from analytics.pitch_control import PitchControlParams, control_from_times, time_to_intercept
from analytics.rink import Grid, Rink
from analytics.threat import ScoringModel, ThreatModel, evaluate_obso

DEFENSE = "Defense"
OFFENSE = "Offense"


@dataclass(frozen=True)
class EngineConfig:
    """Settings of the scene engine.

    Attributes:
        fps (float|None): Frame rate of the tracking [Hz]. ``None`` infers it
            from the velocities.
        fallback_fps (float): Frame rate assumed when it cannot be inferred
            (velocities stored per frame, or no velocities at all) [Hz].
        control_rate (float): Rate at which the reference defense is commanded [Hz].
        timeline_rate (float): Rate at which threat is sampled over the clip [Hz].
        eval_cell (float): Cell of the grid on which threat is integrated [m].
        zone_depth (float): Depth of that grid from the defended end [m].
        display_cell (float): Cell of the pitch-control heatmap [m].
        skater_v_max (float): Speed limit of the reference defenders [m/s].
        skater_a_max (float): Acceleration limit of the reference defenders [m/s^2].
        goalie_extent (float): A goalie moves less than this over the clip [m].
        goalie_distance (float): ... and stays this close to a goal [m].
        suggestion_iterations (int): Centroid iterations for the suggested
            positions of a single frame.
    """

    fps: Optional[float] = None
    fallback_fps: float = 10.0
    control_rate: float = 10.0
    timeline_rate: float = 5.0
    eval_cell: float = 0.75
    zone_depth: float = 30.0
    display_cell: float = 1.0
    skater_v_max: float = 8.0
    skater_a_max: float = 6.0
    goalie_extent: float = 1.5
    goalie_distance: float = 5.0
    suggestion_iterations: int = 15


@dataclass
class FrameState:
    """Skaters of one frame in the canonical frame (goalies removed)."""

    att_ids: List[str]
    att_pos: np.ndarray
    att_vel: np.ndarray
    def_ids: List[str]
    def_pos: np.ndarray
    def_vel: np.ndarray


class SceneAnalytics:
    """Analytics of one tracking clip.

    Args:
        df (DataFrame): Tracking in the dashboard's canonical columns
            ``timestamp, player_id, team, x, y, vx, vy`` with ``team`` equal
            to ``"Defense"`` or ``"Offense"``.
        bounds (dict): Rink bounds ``x_min, x_max, y_min, y_max`` [m].
        config (EngineConfig): Settings.
    """

    def __init__(self, df: pd.DataFrame, bounds: Dict[str, float],
                 config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.bounds = dict(bounds)
        length = float(bounds["x_max"] - bounds["x_min"])
        width = float(bounds["y_max"] - bounds["y_min"])
        self.rink = Rink(length=length, width=width, goal=(Rink.goal[0], width / 2.0))
        self.params = PitchControlParams()
        self.threat = ThreatModel(scoring=ScoringModel(rink=self.rink))
        self.eval_grid = Grid(self.rink, cell=self.config.eval_cell,
                              x_limit=min(self.config.zone_depth, length))
        self.display_grid = Grid(self.rink, cell=self.config.display_cell)
        self.coverage = PitchControlCoverage(grid=self.eval_grid, threat=self.threat,
                                             v_max=self.config.skater_v_max)

        self.timestamps: List[int] = sorted(int(t) for t in df["timestamp"].unique())
        self._index = {ts: i for i, ts in enumerate(self.timestamps)}

        self._infer_units(df)
        self._find_goalies(df)
        self._infer_direction(df)
        self._build_arrays(df)

        self._eval_cache: Dict[Tuple[int, Optional[str]], dict] = {}
        self._pc_cache: Dict[Tuple[int, Optional[str]], np.ndarray] = {}
        self._timeline_cache: Dict[Optional[str], dict] = {}
        self._timeline_lock = threading.Lock()
        self._timeline_started: set = set()

    # ------------------------------------------------------------ inference
    def _infer_units(self, df: pd.DataFrame) -> None:
        """Frame rate [Hz] and the factor that turns stored velocities into m/s."""
        cfg = self.config
        d = df.sort_values(["team", "player_id", "timestamp"])
        grp = d.groupby(["team", "player_id"], sort=False)
        dts = grp["timestamp"].diff().to_numpy(float)
        step = np.hypot(grp["x"].diff().to_numpy(float), grp["y"].diff().to_numpy(float))
        with np.errstate(divide="ignore", invalid="ignore"):
            step = step / dts
        speed = np.hypot(d["vx"].to_numpy(float), d["vy"].to_numpy(float))
        ok = np.isfinite(step) & (step > 1e-3) & (speed > 1e-3)

        self.has_velocity = bool(ok.sum() >= 5)
        ratio = float(np.median(speed[ok] / step[ok])) if self.has_velocity else float("nan")
        if self.has_velocity and ratio > 2.0:
            # Velocities are per second: their ratio to the step is the frame rate.
            self.fps = float(cfg.fps or round(ratio, 1))
            self.vel_scale = 1.0
            self.fps_source = "set" if cfg.fps else "from velocities"
        else:
            # Velocities are per frame (or missing): the frame rate is not in the data.
            self.fps = float(cfg.fps or cfg.fallback_fps)
            self.vel_scale = self.fps
            self.fps_source = "set" if cfg.fps else "assumed"

    def _goal_mouths(self) -> np.ndarray:
        """Centres of the two goal mouths in dashboard coordinates."""
        gx, gy = self.rink.goal
        b = self.bounds
        return np.array([[b["x_min"] + gx, b["y_min"] + gy],
                         [b["x_max"] - gx, b["y_min"] + gy]])

    def _find_goalies(self, df: pd.DataFrame) -> None:
        cfg = self.config
        self.goalies: Dict[str, List[str]] = {DEFENSE: [], OFFENSE: []}
        self._goalie_side: Dict[str, str] = {}
        if len(self.timestamps) < 10:
            return
        mouths = self._goal_mouths()
        for (team, pid), g in df.groupby(["team", "player_id"]):
            extent = max(g["x"].max() - g["x"].min(), g["y"].max() - g["y"].min())
            centre = np.array([g["x"].mean(), g["y"].mean()])
            dist = np.linalg.norm(mouths - centre, axis=1)
            if extent < cfg.goalie_extent and dist.min() < cfg.goalie_distance:
                self.goalies.setdefault(team, []).append(str(pid))
                self._goalie_side.setdefault(team, "left" if dist[0] <= dist[1] else "right")

    def _infer_direction(self, df: pd.DataFrame) -> None:
        """Side of the goal the team labelled Defense defends."""
        if DEFENSE in self._goalie_side:
            side = self._goalie_side[DEFENSE]
        elif OFFENSE in self._goalie_side:
            side = "right" if self._goalie_side[OFFENSE] == "left" else "left"
        else:
            mean_x = df.groupby("team")["x"].mean()
            mid = 0.5 * (self.bounds["x_min"] + self.bounds["x_max"])
            if DEFENSE in mean_x and OFFENSE in mean_x:
                side = "left" if mean_x[DEFENSE] <= mean_x[OFFENSE] else "right"
            else:
                side = "left" if float(mean_x.iloc[0]) <= mid else "right"
        self.defended_side = side
        self.mirrored = side == "right"

    # ------------------------------------------------------------ transforms
    def _to_canonical(self, x, y, vx, vy):
        b = self.bounds
        xc = np.asarray(x, dtype=float) - b["x_min"]
        yc = np.asarray(y, dtype=float) - b["y_min"]
        vxc = np.asarray(vx, dtype=float) * self.vel_scale
        vyc = np.asarray(vy, dtype=float) * self.vel_scale
        if self.mirrored:
            xc = self.rink.length - xc
            vxc = -vxc
        return np.column_stack([xc, yc]), np.column_stack([vxc, vyc])

    def _to_dashboard(self, pos: np.ndarray) -> np.ndarray:
        pos = np.asarray(pos, dtype=float)
        out = pos.copy()
        if self.mirrored:
            out[..., 0] = self.rink.length - out[..., 0]
        out[..., 0] += self.bounds["x_min"]
        out[..., 1] += self.bounds["y_min"]
        return out

    def _grid_to_dashboard(self, grid: Grid, values: np.ndarray):
        """Axes and image of a field on ``grid`` with x increasing in dashboard coordinates."""
        img = grid.as_image(values)
        xs = grid.xs
        if self.mirrored:
            xs = (self.rink.length - xs)[::-1]
            img = img[:, ::-1]
        return xs + self.bounds["x_min"], grid.ys + self.bounds["y_min"], img

    # ------------------------------------------------------------ arrays
    def _build_arrays(self, df: pd.DataFrame) -> None:
        """Positions and velocities as (T, n, 2) arrays, NaN where a player is absent."""
        d = df.copy()
        d["player_id"] = d["player_id"].astype(str)
        if not self.has_velocity:
            # No usable velocities: finite differences of the positions, per frame.
            d = d.sort_values(["team", "player_id", "timestamp"])
            g = d.groupby(["team", "player_id"], sort=False)
            dts = g["timestamp"].diff()
            d["vx"] = (g["x"].diff() / dts).fillna(0.0)
            d["vy"] = (g["y"].diff() / dts).fillna(0.0)
        pos, vel = self._to_canonical(d["x"], d["y"], d["vx"], d["vy"])
        d["cx"], d["cy"], d["cvx"], d["cvy"] = pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1]

        T = len(self.timestamps)
        self.ids: Dict[str, List[str]] = {}
        self._pos: Dict[str, np.ndarray] = {}
        self._vel: Dict[str, np.ndarray] = {}
        row = d["timestamp"].map(self._index).to_numpy()
        for team in (OFFENSE, DEFENSE):
            sub = d[d["team"] == team]
            ids = [p for p in sorted(sub["player_id"].unique(), key=_natural)
                   if p not in self.goalies.get(team, [])]
            col = {p: j for j, p in enumerate(ids)}
            P = np.full((T, len(ids), 2), np.nan)
            V = np.zeros((T, len(ids), 2))
            keep = sub["player_id"].isin(ids).to_numpy()
            r = row[(d["team"] == team).to_numpy()][keep]
            c = sub["player_id"][keep].map(col).to_numpy()
            P[r, c, 0], P[r, c, 1] = sub["cx"][keep], sub["cy"][keep]
            V[r, c, 0], V[r, c, 1] = sub["cvx"][keep], sub["cvy"][keep]
            self.ids[team], self._pos[team], self._vel[team] = ids, P, V

    # ------------------------------------------------------------ frames
    def has_frame(self, ts: int) -> bool:
        """Whether ``ts`` is a frame of this clip."""
        return int(ts) in self._index

    def state_at(self, ts: int) -> FrameState:
        """Skaters of a tracked frame."""
        i = self._index[int(ts)]
        out = {}
        for team in (OFFENSE, DEFENSE):
            P, V = self._pos[team][i], self._vel[team][i]
            ok = np.isfinite(P[:, 0])
            out[team] = ([p for p, k in zip(self.ids[team], ok) if k], P[ok], V[ok])
        return FrameState(*out[OFFENSE], *out[DEFENSE])

    def state_from_frame(self, df_frame: pd.DataFrame) -> FrameState:
        """Skaters of a frame given as a table (e.g. after dragging players)."""
        out = {}
        for team in (OFFENSE, DEFENSE):
            sub = df_frame[df_frame["team"] == team].copy()
            sub["player_id"] = sub["player_id"].astype(str)
            sub = sub[~sub["player_id"].isin(self.goalies.get(team, []))]
            sub = sub.sort_values("player_id", key=lambda s: s.map(_natural))
            vx = sub["vx"] if "vx" in sub and self.has_velocity else np.zeros(len(sub))
            vy = sub["vy"] if "vy" in sub and self.has_velocity else np.zeros(len(sub))
            pos, vel = self._to_canonical(sub["x"], sub["y"], vx, vy)
            out[team] = (sub["player_id"].tolist(), pos.reshape(-1, 2), vel.reshape(-1, 2))
        return FrameState(*out[OFFENSE], *out[DEFENSE])

    @staticmethod
    def _carriers(att_ids: List[str], carrier: Optional[str]) -> List[int]:
        if carrier is not None and str(carrier) in att_ids:
            return [att_ids.index(str(carrier))]
        return list(range(len(att_ids)))

    def carrier_choices(self) -> List[str]:
        """Attackers who can be chosen as the puck carrier."""
        return list(self.ids[OFFENSE])

    # ------------------------------------------------------------ per frame
    def evaluate(self, state: FrameState, carrier: Optional[str] = None,
                 def_pos: Optional[np.ndarray] = None,
                 def_vel: Optional[np.ndarray] = None) -> dict:
        """Scoring threat of a frame, optionally with other defender positions."""
        dp = state.def_pos if def_pos is None else def_pos
        dv = state.def_vel if def_vel is None else def_vel
        res = evaluate_obso(state.att_pos, state.att_vel, dp, dv,
                            self._carriers(state.att_ids, carrier),
                            self.eval_grid, self.threat, self.params)
        res["att_ids"] = list(state.att_ids)
        return res

    def evaluate_at(self, ts: int, carrier: Optional[str] = None) -> dict:
        key = (int(ts), carrier)
        if key not in self._eval_cache:
            self._eval_cache[key] = self.evaluate(self.state_at(ts), carrier)
        return self._eval_cache[key]

    def threat_surface(self, result: dict):
        """Threat surface of ``evaluate`` as ``(x, y, z)`` for a heatmap."""
        return self._grid_to_dashboard(self.eval_grid, result["surface"])

    def pitch_control(self, state: FrameState, carrier: Optional[str] = None):
        """Probability that the attack controls each location of the rink.

        Returns:
            tuple: ``(x, y, z)`` for a heatmap, ``z`` of shape (ny, nx).
        """
        grid = self.display_grid
        pts = grid.points
        carriers = self._carriers(state.att_ids, carrier)
        if len(state.att_ids) == 0 or len(state.def_ids) == 0:
            fill = 1.0 if len(state.att_ids) else 0.0
            return self._grid_to_dashboard(grid, np.full(len(pts), fill))
        tau_a = time_to_intercept(state.att_pos, state.att_vel, pts, self.params)
        tau_d = time_to_intercept(state.def_pos, state.def_vel, pts, self.params)
        c_att = np.zeros(len(pts))
        for c in carriers:
            flight = np.linalg.norm(pts - state.att_pos[c], axis=1) / self.params.puck_speed
            ppcf_att, _ = control_from_times(tau_a, tau_d, flight, self.params)
            c_att += ppcf_att.sum(axis=0)
        return self._grid_to_dashboard(grid, c_att / len(carriers))

    def pitch_control_at(self, ts: int, carrier: Optional[str] = None):
        key = (int(ts), carrier)
        if key not in self._pc_cache:
            self._pc_cache[key] = self.pitch_control(self.state_at(ts), carrier)
        return self._pc_cache[key]

    def suggest(self, state: FrameState, carrier: Optional[str] = None) -> dict:
        """Positions the coverage law moves the defenders of this frame to.

        Returns:
            dict: ``positions`` (table ``player_id, sug_x, sug_y`` in dashboard
            coordinates) and ``obso`` (threat with the defenders there, same
            velocities as in the frame).
        """
        carriers = self._carriers(state.att_ids, carrier)
        target = self.coverage.targets(state.def_pos, state.att_pos, carriers,
                                       iterations=self.config.suggestion_iterations)
        target = self._clip(target)
        obso = self.evaluate(state, carrier, def_pos=target)["obso"] if len(target) else float("nan")
        return {"positions": self._ghost_table(state.def_ids, target), "obso": obso}

    # ------------------------------------------------------------ whole clip
    def timeline(self, carrier: Optional[str] = None) -> dict:
        """Threat of the tracked defense and of the reference defense over the clip.

        The reference defenders start at the positions and velocities of the
        tracked defenders in the first frame, follow the pitch-control
        coverage law under the skater limits and face the tracked attack,
        which does not react to them.

        Returns:
            dict: ``frame_idx`` (sampled frame indices), ``obso_tracked`` and
            ``obso_reference`` at these frames, their clip means
            ``mean_tracked`` and ``mean_reference``, ``def_ids`` and
            ``ref_pos`` (reference positions for every frame, (T, nd, 2), in
            dashboard coordinates). ``frame_idx`` is empty if a team is absent.
        """
        # One replay at a time: callers that arrive during a replay wait for it.
        with self._timeline_lock:
            if carrier not in self._timeline_cache:
                self._timeline_cache[carrier] = self._run_timeline(carrier)
            return self._timeline_cache[carrier]

    def timeline_ready(self, carrier: Optional[str] = None) -> bool:
        """Whether ``timeline(carrier)`` returns without computing."""
        return carrier in self._timeline_cache

    def ensure_timeline(self, carrier: Optional[str] = None) -> bool:
        """Start the replay in the background if needed; ``True`` once it is ready.

        The replay takes a few seconds, so interactive callers poll this
        instead of waiting inside ``timeline``.
        """
        if self.timeline_ready(carrier):
            return True
        if carrier not in self._timeline_started:
            self._timeline_started.add(carrier)
            threading.Thread(target=self.timeline, args=(carrier,), daemon=True).start()
        return False

    def _run_timeline(self, carrier: Optional[str]) -> dict:
        cfg = self.config
        T = len(self.timestamps)
        empty = {"frame_idx": np.array([], dtype=int), "obso_tracked": np.array([]),
                 "obso_reference": np.array([]), "mean_tracked": float("nan"),
                 "mean_reference": float("nan"), "def_ids": [], "ref_pos": np.zeros((T, 0, 2))}
        first = self.state_at(self.timestamps[0])
        if not first.def_ids or not self.ids[OFFENSE]:
            return empty

        stride = max(1, int(round(self.fps / cfg.control_rate)))
        idx = list(range(0, T, stride))
        if idx[-1] != T - 1:
            idx.append(T - 1)

        every = max(1, int(round(cfg.control_rate / cfg.timeline_rate)))
        pos = first.def_pos.copy()
        vel = _limit(first.def_vel.copy(), cfg.skater_v_max)
        sampled, tracked, reference, ref_pos = [], [], [], []
        for j, i in enumerate(idx):
            ts = self.timestamps[i]
            state = self.state_at(ts)
            ref_pos.append(pos.copy())
            if j % every == 0 or j == len(idx) - 1:
                sampled.append(i)
                if state.att_ids:
                    tracked.append(self.evaluate_at(ts, carrier)["obso"] if state.def_ids else np.nan)
                    reference.append(self.evaluate(state, carrier, def_pos=pos, def_vel=vel)["obso"])
                else:
                    tracked.append(0.0)
                    reference.append(0.0)
            if j + 1 < len(idx):
                dt = (self.timestamps[idx[j + 1]] - ts) / self.fps
                if state.att_ids:
                    u, _ = self.coverage.nominal(pos, vel, state.att_pos, state.att_vel,
                                                 self._carriers(state.att_ids, carrier))
                else:
                    u = np.zeros_like(pos)
                pos, vel = self._skate(pos, vel, u, dt)

        ref_pos = np.array(ref_pos)                                   # (n, nd, 2)
        full = np.empty((T, ref_pos.shape[1], 2))
        frames = np.arange(T)
        for d in range(ref_pos.shape[1]):
            for k in range(2):
                full[:, d, k] = np.interp(frames, idx, ref_pos[:, d, k])
        tracked = np.array(tracked, dtype=float)
        reference = np.array(reference, dtype=float)
        return {
            "frame_idx": np.array(sampled, dtype=int),
            "obso_tracked": tracked,
            "obso_reference": reference,
            "mean_tracked": float(np.nanmean(tracked)),
            "mean_reference": float(np.nanmean(reference)),
            "def_ids": list(first.def_ids),
            "ref_pos": self._to_dashboard(full),
        }

    def reference_at(self, ts: int, carrier: Optional[str] = None) -> pd.DataFrame:
        """Reference defenders of a tracked frame as ``player_id, sug_x, sug_y``."""
        tl = self.timeline(carrier)
        i = self._index[int(ts)]
        xy = tl["ref_pos"][i]
        return pd.DataFrame({"player_id": tl["def_ids"], "team": DEFENSE,
                             "sug_x": xy[:, 0], "sug_y": xy[:, 1]})

    def reference_threat_at(self, ts: int, carrier: Optional[str] = None) -> float:
        """Threat against the reference defense at a tracked frame (interpolated)."""
        tl = self.timeline(carrier)
        if len(tl["frame_idx"]) == 0:
            return float("nan")
        return float(np.interp(self._index[int(ts)], tl["frame_idx"], tl["obso_reference"]))

    # ------------------------------------------------------------ helpers
    def _clip(self, pos: np.ndarray) -> np.ndarray:
        pos = np.array(pos, dtype=float)
        if len(pos):
            pos[:, 0] = np.clip(pos[:, 0], 0.5, self.rink.length - 0.5)
            pos[:, 1] = np.clip(pos[:, 1], 0.5, self.rink.width - 0.5)
        return pos

    def _skate(self, pos, vel, u, dt):
        """Velocity tracks the command under acceleration and speed limits."""
        cfg = self.config
        vel = vel + _limit(u - vel, cfg.skater_a_max * dt)
        vel = _limit(vel, cfg.skater_v_max)
        return self._clip(pos + vel * dt), vel

    def _ghost_table(self, def_ids: List[str], pos: np.ndarray) -> pd.DataFrame:
        xy = self._to_dashboard(pos) if len(pos) else np.zeros((0, 2))
        return pd.DataFrame({"player_id": list(def_ids), "team": DEFENSE,
                             "sug_x": xy[:, 0], "sug_y": xy[:, 1]})

    def describe(self) -> str:
        """One line on what was inferred from the clip."""
        parts = [f"{self.fps:g} fps ({self.fps_source})",
                 f"defended goal on the {self.defended_side}"]
        goalies = [f"{team} #{p}" for team in (DEFENSE, OFFENSE) for p in self.goalies.get(team, [])]
        if goalies:
            parts.append("goalies left out: " + ", ".join(goalies))
        return " · ".join(parts)


def _limit(v: np.ndarray, bound: float) -> np.ndarray:
    """Scale the rows of ``v`` so that none is longer than ``bound``."""
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(norm > bound, v * bound / np.maximum(norm, 1e-12), v)


def _natural(pid) -> tuple:
    """Sort key that orders "2" before "10" and keeps non-numeric ids last."""
    s = str(pid)
    try:
        return (0, float(s), s)
    except ValueError:
        return (1, 0.0, s)
