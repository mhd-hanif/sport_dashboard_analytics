// Scene engine: turns the tracking table of one clip into what the dashboard shows.
// JavaScript port of ../../analytics/engine.py (same inference rules, same replay).
//
// The engine reads from the clip what the clip does not say: the frame rate and velocity
// units, which goal is defended, and the goalies (left out: the scoring model already stands
// for the goalie). The puck is not tracked: the carrier is either unknown (results are
// averaged over the attackers) or given by puck marks, "attacker X has it from frame i on".

import { ThreatModel, makeRink, PARAMS } from "./models.js";

export const CONFIG = {
  fps: null,              // frame rate [Hz]; null = infer from the velocities
  fallbackFps: 10.0,      // used when the velocities are per frame or missing
  controlRate: 10.0,      // rate of the reference defense [Hz]
  timelineRate: 5.0,      // rate at which threat is sampled over the clip [Hz]
  aMax: 6.0,              // acceleration limit of the reference defenders [m/s²]
  goalieExtent: 1.5,      // a goalie moves less than this over the clip [m]
  goalieDistance: 5.0,    // ... and stays this close to a goal [m]
  suggestionIterations: 15,
};

const natural = (a, b) => {
  const fa = Number(a), fb = Number(b), na = Number.isFinite(fa), nb = Number.isFinite(fb);
  if (na && nb) return fa - fb || (a < b ? -1 : a > b ? 1 : 0);
  if (na !== nb) return na ? -1 : 1;
  return a < b ? -1 : a > b ? 1 : 0;
};
const roundHalfEven = (v) => { const r = Math.round(v); return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r; };
const median = (a) => { const s = [...a].sort((x, y) => x - y), h = s.length >> 1; return s.length % 2 ? s[h] : 0.5 * (s[h - 1] + s[h]); };
function limit(v, bound) { const n = Math.hypot(v[0], v[1]); return n > bound ? [(v[0] * bound) / n, (v[1] * bound) / n] : v; }

// Parse a tracking CSV into rows { t, team: "D" | "O", id, x, y, vx, vy }.
// Columns: timeframe or timestamp, player_id, x, y, team_id (0 = defense) or team; vx, vy optional.
export function parseCsv(text) {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length);
  if (lines.length < 2) throw new Error("The file has no data rows.");
  const head = lines[0].split(",").map((h) => h.trim().toLowerCase());
  const col = (name) => head.indexOf(name);
  const ct = col("timeframe") >= 0 ? col("timeframe") : col("timestamp");
  const need = { "timeframe or timestamp": ct, player_id: col("player_id"), x: col("x"), y: col("y") };
  const missing = Object.keys(need).filter((k) => need[k] < 0);
  const cTeamId = col("team_id"), cTeam = col("team");
  if (cTeamId < 0 && cTeam < 0) missing.push("team_id");
  if (missing.length) throw new Error("Missing columns: " + missing.join(", ") + ".");
  const cvx = col("vx"), cvy = col("vy"), rows = [];
  for (let i = 1; i < lines.length; i++) {
    const f = lines[i].split(",");
    let defense;
    if (cTeamId >= 0) defense = Math.trunc(Number(f[cTeamId])) === 0;
    else defense = ["defense", "defensive", "def"].includes(String(f[cTeam]).trim().toLowerCase());
    const row = {
      t: Math.trunc(Number(f[ct])), team: defense ? "D" : "O", id: String(f[need.player_id]).trim(),
      x: Number(f[need.x]), y: Number(f[need.y]),
      vx: cvx >= 0 ? Number(f[cvx]) || 0 : 0, vy: cvy >= 0 ? Number(f[cvy]) || 0 : 0,
    };
    if (Number.isFinite(row.t) && Number.isFinite(row.x) && Number.isFinite(row.y)) rows.push(row);
  }
  if (!rows.length) throw new Error("No valid rows in the file.");
  return rows;
}

export class Scene {
  constructor(rows, bounds = { xMin: 0, xMax: 61, yMin: 0, yMax: 30 }, config = {}) {
    this.cfg = { ...CONFIG, ...config };
    this.bounds = bounds;
    this.rink = makeRink(bounds.xMax - bounds.xMin, bounds.yMax - bounds.yMin);
    this.model = new ThreatModel(this.rink);
    this.timestamps = [...new Set(rows.map((r) => r.t))].sort((a, b) => a - b);
    this.T = this.timestamps.length;
    const index = new Map(this.timestamps.map((t, i) => [t, i]));

    // One series per player, in time order.
    const series = new Map();
    for (const r of rows) {
      const key = r.team + "|" + r.id;
      if (!series.has(key)) series.set(key, { team: r.team, id: r.id, rows: [] });
      series.get(key).rows.push(r);
    }
    const all = [...series.values()].sort((a, b) => (a.team === b.team ? (a.id < b.id ? -1 : 1) : a.team < b.team ? -1 : 1));
    all.forEach((s) => s.rows.sort((a, b) => a.t - b.t));

    this._inferUnits(all);
    this._findGoalies(all);
    this._inferDirection(rows);

    // Skaters per frame in the canonical frame, and every player per frame for drawing.
    this.ids = { O: [], D: [] };
    for (const team of ["O", "D"]) {
      this.ids[team] = all.filter((s) => s.team === team && !this.goalies[team].includes(s.id)).map((s) => s.id).sort(natural);
    }
    this.frames = Array.from({ length: this.T }, () => ({ O: new Map(), D: new Map(), all: [] }));
    for (const s of all) {
      const goalie = this.goalies[s.team].includes(s.id);
      s.rows.forEach((r, j) => {
        let vx = r.vx, vy = r.vy;
        if (!this.hasVelocity) {
          const prev = s.rows[j - 1];
          vx = prev ? (r.x - prev.x) / (r.t - prev.t) : 0; vy = prev ? (r.y - prev.y) / (r.t - prev.t) : 0;
        }
        const f = this.frames[index.get(r.t)];
        f.all.push({ team: s.team, id: s.id, x: r.x, y: r.y, vx: vx * this.velScale, vy: vy * this.velScale, goalie });
        if (!goalie) f[s.team].set(s.id, this._canonical(r.x, r.y, vx, vy));
      });
    }
  }

  _inferUnits(all) {
    const ratios = [];
    for (const s of all) for (let j = 1; j < s.rows.length; j++) {
      const a = s.rows[j - 1], b = s.rows[j];
      const step = Math.hypot(b.x - a.x, b.y - a.y) / (b.t - a.t), speed = Math.hypot(b.vx, b.vy);
      if (Number.isFinite(step) && step > 1e-3 && speed > 1e-3) ratios.push(speed / step);
    }
    this.hasVelocity = ratios.length >= 5;
    const ratio = this.hasVelocity ? median(ratios) : NaN;
    if (this.hasVelocity && ratio > 2.0) {
      // Velocities are per second: their ratio to the step between frames is the frame rate.
      this.fps = this.cfg.fps || Math.round(ratio * 10) / 10;
      this.velScale = 1.0;
      this.fpsSource = this.cfg.fps ? "set" : "from velocities";
    } else {
      this.fps = this.cfg.fps || this.cfg.fallbackFps;
      this.velScale = this.fps;
      this.fpsSource = this.cfg.fps ? "set" : "assumed";
    }
  }

  _findGoalies(all) {
    this.goalies = { D: [], O: [] };
    this._goalieSide = {};
    if (this.T < 10) return;
    const b = this.bounds, gy = b.yMin + this.rink.goalY;
    const mouths = [[b.xMin + this.rink.goalX, gy], [b.xMax - this.rink.goalX, gy]];
    for (const s of all) {
      const xs = s.rows.map((r) => r.x), ys = s.rows.map((r) => r.y);
      const extent = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
      const cx = xs.reduce((a, v) => a + v, 0) / xs.length, cy = ys.reduce((a, v) => a + v, 0) / ys.length;
      const d = mouths.map((m) => Math.hypot(m[0] - cx, m[1] - cy));
      if (extent < this.cfg.goalieExtent && Math.min(...d) < this.cfg.goalieDistance) {
        this.goalies[s.team].push(s.id);
        if (!(s.team in this._goalieSide)) this._goalieSide[s.team] = d[0] <= d[1] ? "left" : "right";
      }
    }
  }

  _inferDirection(rows) {
    let side;
    if ("D" in this._goalieSide) side = this._goalieSide.D;
    else if ("O" in this._goalieSide) side = this._goalieSide.O === "left" ? "right" : "left";
    else {
      const mean = (team) => { const v = rows.filter((r) => r.team === team); return v.length ? v.reduce((a, r) => a + r.x, 0) / v.length : NaN; };
      const md = mean("D"), mo = mean("O"), mid = 0.5 * (this.bounds.xMin + this.bounds.xMax);
      if (Number.isFinite(md) && Number.isFinite(mo)) side = md <= mo ? "left" : "right";
      else side = (Number.isFinite(md) ? md : mo) <= mid ? "left" : "right";
    }
    this.defendedSide = side;
    this.mirrored = side === "right";
  }

  // ---- transforms -------------------------------------------------------------------
  _canonical(x, y, vx, vy) {
    let xc = x - this.bounds.xMin, vxc = vx * this.velScale;
    if (this.mirrored) { xc = this.rink.length - xc; vxc = -vxc; }
    return [xc, y - this.bounds.yMin, vxc, vy * this.velScale];
  }
  toDashboard(x, y) {
    return [(this.mirrored ? this.rink.length - x : x) + this.bounds.xMin, y + this.bounds.yMin];
  }

  // ---- frames -----------------------------------------------------------------------
  // Skaters of frame ``i``. ``edits`` maps "D|id" / "O|id" to { x, y } in dashboard coordinates.
  state(i, edits = null) {
    const f = this.frames[i], out = { attIds: [], att: [], defIds: [], def: [] };
    for (const [team, ids, list] of [["O", out.attIds, out.att], ["D", out.defIds, out.def]]) {
      for (const id of this.ids[team]) {
        const p = f[team].get(id);
        if (!p) continue;
        const e = edits && edits[team + "|" + id];
        ids.push(id);
        list.push(e ? [...this._canonical(e.x, e.y, 0, 0).slice(0, 2), p[2], p[3]] : p.slice());
      }
    }
    return out;
  }

  // Carrier of frame ``i`` given puck marks [{ i, id }] (id null = unknown).
  static carrierAt(i, marks) {
    let id = null;
    for (const m of marks || []) if (m.i <= i) id = m.id; else break;
    return id;
  }
  _carriers(state, carrierId) {
    const k = carrierId === null || carrierId === undefined ? -1 : state.attIds.indexOf(String(carrierId));
    return k >= 0 ? [k] : state.att.map((_, j) => j);
  }

  evaluate(state, carrierId = null, def = null) {
    return this.model.evaluate(state.att, def || state.def, this._carriers(state, carrierId));
  }
  pitchControl(state, carrierId = null) {
    return this.model.pitchControl(state.att, state.def, this._carriers(state, carrierId));
  }

  // Positions the coverage law moves the defenders of this frame to, and the threat there.
  suggest(state, carrierId = null) {
    const carriers = this._carriers(state, carrierId);
    const target = this.model.targets(state.def, state.att, carriers, this.cfg.suggestionIterations).map((t) => this._clip(t));
    const moved = target.map((t, i) => [t[0], t[1], state.def[i][2], state.def[i][3]]);
    return {
      ids: state.defIds, pos: target.map((t) => this.toDashboard(t[0], t[1])),
      obso: target.length ? this.model.evaluate(state.att, moved, carriers).obso : NaN,
    };
  }

  // How much the threat rises if a defender is taken off the ice: the threat he removes.
  defenderValues(state, carrierId, base) {
    const carriers = this._carriers(state, carrierId);
    return state.def.map((_, i) => this.model.evaluate(state.att, state.def.filter((__, j) => j !== i), carriers).obso - base);
  }

  // ---- whole clip -------------------------------------------------------------------
  // Threat over the clip against the tracked and the reference defense. The reference
  // defenders start where the tracked defenders start, follow pitch-control coverage under
  // skater limits and face the tracked attack, which does not react to them.
  // Generator: yields the fraction done, returns the result.
  *timelineSteps(marks = []) {
    const cfg = this.cfg, T = this.T, first = this.state(0);
    const empty = { idx: [], tracked: [], reference: [], meanTracked: NaN, meanReference: NaN, defIds: [], refPos: new Float32Array(0) };
    if (!first.defIds.length || !this.ids.O.length) return empty;

    const stride = Math.max(1, roundHalfEven(this.fps / cfg.controlRate));
    const every = Math.max(1, roundHalfEven(cfg.controlRate / cfg.timelineRate));
    const idx = [];
    for (let i = 0; i < T; i += stride) idx.push(i);
    if (idx[idx.length - 1] !== T - 1) idx.push(T - 1);

    let ref = first.def.map((p) => { const v = limit([p[2], p[3]], PARAMS.vMax); return [p[0], p[1], v[0], v[1]]; });
    const sampled = [], tracked = [], reference = [], path = [];
    for (let j = 0; j < idx.length; j++) {
      const i = idx[j], st = this.state(i), carrier = Scene.carrierAt(i, marks);
      path.push(ref.map((p) => [p[0], p[1]]));
      if (j % every === 0 || j === idx.length - 1) {
        sampled.push(i);
        if (st.att.length) {
          tracked.push(st.def.length ? this.evaluate(st, carrier).obso : NaN);
          reference.push(this.evaluate(st, carrier, ref).obso);
        } else { tracked.push(0); reference.push(0); }
      }
      if (j + 1 < idx.length) {
        const dt = (this.timestamps[idx[j + 1]] - this.timestamps[i]) / this.fps;
        const u = st.att.length ? this.model.nominal(ref, st.att, this._carriers(st, carrier)) : ref.map(() => [0, 0]);
        ref = ref.map((p, d) => {
          const dv = limit([u[d][0] - p[2], u[d][1] - p[3]], cfg.aMax * dt);
          const v = limit([p[2] + dv[0], p[3] + dv[1]], PARAMS.vMax);
          const q = this._clip([p[0] + v[0] * dt, p[1] + v[1] * dt]);
          return [q[0], q[1], v[0], v[1]];
        });
      }
      yield (j + 1) / idx.length;
    }

    // Reference positions for every frame (linear in between), in dashboard coordinates.
    const nd = first.defIds.length, refPos = new Float32Array(T * nd * 2);
    for (let j = 0; j + 1 < idx.length || j === 0; j++) {
      const a = idx[j], b = idx[Math.min(j + 1, idx.length - 1)];
      for (let i = a; i <= b; i++) {
        const w = b > a ? (i - a) / (b - a) : 0;
        for (let d = 0; d < nd; d++) {
          const pa = path[j][d], pb = path[Math.min(j + 1, idx.length - 1)][d];
          const q = this.toDashboard(pa[0] + w * (pb[0] - pa[0]), pa[1] + w * (pb[1] - pa[1]));
          refPos[(i * nd + d) * 2] = q[0]; refPos[(i * nd + d) * 2 + 1] = q[1];
        }
      }
      if (idx.length === 1) break;
    }
    const nanmean = (a) => { const v = a.filter(Number.isFinite); return v.length ? v.reduce((s, x) => s + x, 0) / v.length : NaN; };
    return { idx: sampled, tracked, reference, meanTracked: nanmean(tracked), meanReference: nanmean(reference), defIds: first.defIds, refPos };
  }
  timeline(marks = []) {
    const it = this.timelineSteps(marks);
    for (;;) { const s = it.next(); if (s.done) return s.value; }
  }

  _clip(p) {
    return [Math.min(Math.max(p[0], 0.5), this.rink.length - 0.5), Math.min(Math.max(p[1], 0.5), this.rink.width - 0.5)];
  }

  describe() {
    const parts = [`${this.fps} fps (${this.fpsSource})`, `defended goal on the ${this.defendedSide}`];
    const g = [...this.goalies.D.map((p) => "Defense #" + p), ...this.goalies.O.map((p) => "Offense #" + p)];
    if (g.length) parts.push("goalies left out: " + g.join(", "));
    return parts.join(" · ");
  }
}
