"""Export reference numbers from the Python analytics package for the JavaScript tests.

Run from the repository root:  python web/tests/make_golden.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analytics import SceneAnalytics  # noqa: E402

BOUNDS = {"x_min": 0.0, "x_max": 61.0, "y_min": 0.0, "y_max": 30.0}
CASES = [("scene_5.csv", 150, [0, 60, 150], "7"), ("scene_0.csv", 10**9, [0, 40, 118], "5")]


def load(name, t_max):
    df = pd.read_csv(ROOT / "assets" / "clip_scene" / name).rename(columns={"timeframe": "timestamp"})
    df["team"] = np.where(df["team_id"] == 0, "Defense", "Offense")
    df["player_id"] = df["player_id"].astype(str)
    return df[df["timestamp"] <= t_max][["timestamp", "player_id", "team", "x", "y", "vx", "vy"]]


out = []
for name, t_max, frames, carrier in CASES:
    eng = SceneAnalytics(load(name, t_max), BOUNDS)
    case = {
        "file": name, "tMax": t_max, "carrier": carrier, "fps": eng.fps, "fpsSource": eng.fps_source,
        "side": eng.defended_side, "goalies": {"D": eng.goalies["Defense"], "O": eng.goalies["Offense"]},
        "ids": {"O": eng.ids["Offense"], "D": eng.ids["Defense"]}, "frames": [], "timelines": [],
    }
    for i in frames:
        ts = eng.timestamps[i]
        state = eng.state_at(ts)
        for c in (None, carrier):
            res = eng.evaluate(state, c)
            _, _, pc = eng.pitch_control(state, c)
            sug = eng.suggest(state, c)
            if eng.mirrored:
                pc = pc[:, ::-1]                      # back to the canonical frame
            case["frames"].append({
                "i": i, "carrier": c, "obso": res["obso"], "perAttacker": res["per_attacker"].tolist(),
                "surfaceSum": float(res["surface"].sum()), "surfaceMax": float(res["surface"].max()),
                "pcMean": float(pc.mean()), "pcSample": pc.ravel()[::97].tolist(),
                "suggestion": sug["positions"][["sug_x", "sug_y"]].to_numpy().tolist(), "suggestionObso": sug["obso"],
            })
    for c in (None, carrier):
        tl = eng.timeline(c)
        case["timelines"].append({
            "carrier": c, "idx": tl["frame_idx"].tolist(), "tracked": tl["obso_tracked"].tolist(),
            "reference": tl["obso_reference"].tolist(), "meanTracked": tl["mean_tracked"],
            "meanReference": tl["mean_reference"], "defIds": tl["def_ids"],
            "refPosLast": tl["ref_pos"][-1].tolist(), "refPosMid": tl["ref_pos"][len(eng.timestamps) // 2].tolist(),
        })
    out.append(case)

(Path(__file__).parent / "golden.json").write_text(json.dumps(out))
print("wrote golden.json:", [c["file"] for c in out])
