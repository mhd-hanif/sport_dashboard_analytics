"""Tests of the defensive analytics package (run with ``pytest`` from the repository root)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analytics import (  # noqa: E402
    Grid, PitchControlCoverage, PitchControlParams, SceneAnalytics, evaluate_obso, spearman_control,
)

BOUNDS = {"x_min": 0.0, "x_max": 61.0, "y_min": 0.0, "y_max": 30.0}
SCENES = Path(__file__).resolve().parents[1] / "assets" / "clip_scene"


def _load(name):
    df = pd.read_csv(SCENES / name).rename(columns={"timeframe": "timestamp"})
    df["team"] = np.where(df["team_id"] == 0, "Defense", "Offense")
    df["player_id"] = df["player_id"].astype(str)
    return df[["timestamp", "player_id", "team", "x", "y", "vx", "vy"]]


def _mirror(df):
    out = df.copy()
    out["x"] = BOUNDS["x_max"] - out["x"]
    out["vx"] = -out["vx"]
    return out


@pytest.fixture(scope="module")
def players():
    rng = np.random.default_rng(1)
    return {
        "att_pos": rng.uniform([6, 3], [28, 27], (4, 2)), "att_vel": rng.normal(0, 2, (4, 2)),
        "def_pos": rng.uniform([6, 3], [22, 27], (4, 2)), "def_vel": rng.normal(0, 2, (4, 2)),
    }


def test_control_is_shared_between_the_teams(players):
    pts = Grid(cell=2.0, x_limit=30.0).points
    att, dfn = spearman_control(players["att_pos"], players["att_vel"], players["def_pos"],
                                players["def_vel"], players["att_pos"][0], pts, PitchControlParams())
    assert att.min() >= 0.0 and dfn.min() >= 0.0
    assert np.allclose(att.sum(axis=0) + dfn.sum(axis=0), 1.0)


def test_threat_splits_over_attackers_and_carriers(players):
    grid = Grid(cell=1.0, x_limit=30.0)
    args = (players["att_pos"], players["att_vel"], players["def_pos"], players["def_vel"])
    single = [evaluate_obso(*args, [c], grid) for c in range(4)]
    both = evaluate_obso(*args, range(4), grid)
    assert both["obso"] == pytest.approx(np.mean([r["obso"] for r in single]))
    assert both["per_attacker"].sum() == pytest.approx(both["obso"])
    assert both["surface"].sum() * grid.cell_area == pytest.approx(both["obso"])


def test_a_defender_in_the_slot_lowers_the_threat(players):
    grid = Grid(cell=1.0, x_limit=30.0)
    far = np.array([[28.0, 2.0]])
    slot = np.array([[8.0, 15.0]])
    zero = np.zeros((1, 2))
    kw = dict(att_pos=players["att_pos"], att_vel=players["att_vel"], carriers=[0], grid=grid)
    assert evaluate_obso(def_pos=slot, def_vel=zero, **kw)["obso"] < evaluate_obso(def_pos=far, def_vel=zero, **kw)["obso"]


def test_centroid_iteration_descends_the_design_objective(players):
    cov = PitchControlCoverage()
    before = cov.objective(players["def_pos"], players["att_pos"], [0, 1, 2, 3])
    target = cov.targets(players["def_pos"], players["att_pos"], [0, 1, 2, 3], iterations=10)
    assert cov.objective(target, players["att_pos"], [0, 1, 2, 3]) < before


def test_units_direction_and_goalies_are_read_from_the_clip():
    per_second = SceneAnalytics(_load("scene_5.csv"), BOUNDS)
    assert per_second.fps == pytest.approx(30.0) and per_second.vel_scale == 1.0
    assert per_second.defended_side == "right"
    assert per_second.goalies["Defense"] == ["1"]

    per_frame = SceneAnalytics(_load("scene_0.csv"), BOUNDS)
    assert per_frame.fps_source == "assumed" and per_frame.vel_scale == per_frame.fps
    assert per_frame.defended_side == "left"
    assert per_frame.goalies == {"Defense": ["6"], "Offense": ["6"]}


def test_results_do_not_depend_on_the_attack_direction():
    df = _load("scene_5.csv")
    df = df[df["timestamp"] <= 60]
    a, b = SceneAnalytics(df, BOUNDS), SceneAnalytics(_mirror(df), BOUNDS)
    assert a.defended_side != b.defended_side
    ts = a.timestamps[30]
    assert a.evaluate_at(ts)["obso"] == pytest.approx(b.evaluate_at(ts)["obso"])
    xa, _, za = a.pitch_control_at(ts)
    xb, _, zb = b.pitch_control_at(ts)
    assert np.all(np.diff(xa) > 0) and np.all(np.diff(xb) > 0)
    assert np.allclose(za, zb[:, ::-1])
    ga, gb = a.reference_at(ts), b.reference_at(ts)
    assert np.allclose(ga["sug_x"], BOUNDS["x_max"] - gb["sug_x"]) and np.allclose(ga["sug_y"], gb["sug_y"])


def test_reference_defense_replays_the_clip():
    df = _load("scene_5.csv")
    df = df[df["timestamp"] <= 150]
    eng = SceneAnalytics(df, BOUNDS)
    tl = eng.timeline()
    assert tl["ref_pos"].shape == (len(eng.timestamps), 3, 2)
    # It starts where the tracked defenders start ...
    start = df[(df["timestamp"] == eng.timestamps[0]) & (df["team"] == "Defense") & (df["player_id"] != "1")]
    start = start.set_index("player_id").loc[tl["def_ids"]]
    assert np.allclose(tl["ref_pos"][0], start[["x", "y"]].to_numpy())
    # ... and respects the speed limit of a skater.
    step = np.linalg.norm(np.diff(tl["ref_pos"], axis=0), axis=2) * eng.fps
    assert step.max() <= eng.config.skater_v_max + 1e-6
    assert np.isfinite(tl["mean_tracked"]) and np.isfinite(tl["mean_reference"])
    assert eng.timeline_ready() and not eng.timeline_ready("7")


def test_edited_frame_and_suggestion():
    df = _load("scene_5.csv")
    eng = SceneAnalytics(df[df["timestamp"] <= 30], BOUNDS)
    ts = eng.timestamps[10]
    frame = df[df["timestamp"] == ts]
    state = eng.state_from_frame(frame)
    assert state.def_ids == ["2", "3", "4"] and state.att_ids == ["5", "6", "7", "8"]
    assert eng.evaluate(state)["obso"] == pytest.approx(eng.evaluate_at(ts)["obso"])
    sug = eng.suggest(state, carrier="7")
    assert list(sug["positions"]["player_id"]) == ["2", "3", "4"]
    assert sug["obso"] <= eng.evaluate(state, "7")["obso"] * 1.02
