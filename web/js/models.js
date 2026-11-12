// Pitch control, scoring threat and the pitch-control coverage law.
//
// JavaScript port of the Python package ../../analytics (pitch_control.py, threat.py,
// coverage.py). Same equations, same constants, same order of operations; the tests in
// ../tests compare the two on numbers exported from Python. SI units, canonical rink frame:
// x along the rink (0..61 m), y across (0..30 m), defended goal on the left.
//
// Players are plain arrays [x, y, vx, vy].

// Euclidean length. Math.hypot is several times slower and this matches numpy's norm.
const len = (a, b) => Math.sqrt(a * a + b * b);

export const PARAMS = {
  vMax: 8.0,        // maximum skating speed [m/s]
  tReact: 0.5,      // reaction time [s]
  reach: 1.5,       // stick reach [m]
  sigma: 0.45,      // arrival-time uncertainty [s]
  lam: 4.3,         // control rate [1/s]
  kappaDef: 1.0,    // advantage of the defending team in the control rate
  puckSpeed: 20.0,  // speed of a pass [m/s]
  dt: 0.04,         // integration step [s]
  tMax: 6.0,        // integration horizon [s]
  tol: 0.01,        // stop once this little is left uncontrolled everywhere
};

export const SCORING = { sRef: 0.25, dRef: 3.0, gamma: 1.2, sMax: 0.35 };
export const TRANSITION_SIGMA = 8.0;

export const COVERAGE = {
  beta: 4.0, minGap: 0.5, lookahead: 0.5, carrierWeight: 0.8, iterations: 5, gain: 4.0,
};

export function makeRink(length = 61.0, width = 30.0) {
  return { length, width, goalX: 4.3, goalY: width / 2, goalWidth: 1.83 };
}

// Regular grid of cell centres, row-major: index = row * nx + col.
export function makeGrid(rink, cell, xLimit = null) {
  const len = xLimit === null ? rink.length : xLimit;
  const nx = Math.round(len / cell), ny = Math.round(rink.width / cell);
  const xs = new Float64Array(nx), ys = new Float64Array(ny);
  for (let q = 0; q < nx; q++) xs[q] = ((q + 0.5) * len) / nx;
  for (let r = 0; r < ny; r++) ys[r] = ((r + 0.5) * rink.width) / ny;
  const n = nx * ny, px = new Float64Array(n), py = new Float64Array(n);
  for (let r = 0; r < ny; r++) for (let q = 0; q < nx; q++) { px[r * nx + q] = xs[q]; py[r * nx + q] = ys[r]; }
  return { nx, ny, n, xs, ys, px, py, area: (xs[1] - xs[0]) * (ys[1] - ys[0]) };
}

// Scoring probability from a point: grows with the angle under which the goal mouth is seen.
export function scoringAt(rink, x, y) {
  if (x <= rink.goalX) return 0.0;
  const half = 0.5 * rink.goalWidth;
  const ax = rink.goalX - x, ay = rink.goalY + half - y;
  const bx = rink.goalX - x, by = rink.goalY - half - y;
  const psi = Math.abs(Math.atan2(ax * by - ay * bx, ax * bx + ay * by));
  const psiRef = 2.0 * Math.atan((0.5 * rink.goalWidth) / SCORING.dRef);
  return Math.min(SCORING.sMax, SCORING.sRef * Math.pow(psi / psiRef, SCORING.gamma));
}

function transitionAt(x, y, x0, y0) {
  const s2 = TRANSITION_SIGMA * TRANSITION_SIGMA;
  const dx = x - x0, dy = y - y0;
  return Math.exp(-(dx * dx + dy * dy) / (2.0 * s2)) / (2.0 * Math.PI * s2);
}

// Times for every player to reach every cell, flat [player * m + cell].
function timesToIntercept(players, px, py, m) {
  const out = new Float64Array(players.length * m);
  players.forEach((p, k) => {
    const sx = p[0] + p[2] * PARAMS.tReact, sy = p[1] + p[3] * PARAMS.tReact;
    for (let c = 0; c < m; c++) {
      const d = Math.max(len(px[c] - sx, py[c] - sy) - PARAMS.reach, 0.0);
      out[k * m + c] = PARAMS.tReact + d / PARAMS.vMax;
    }
  });
  return out;
}

// Integrate the control rate from the arrival of the puck (Spearman). ``tau`` holds the
// attackers first, then the defenders. Returns the normalised control of every player.
//
// A player has arrived at time t with probability 1 / (1 + exp(-scale (t - tau))). With
// t = puckTime + step dt the exponential factors into exp(scale tau) exp(-scale puckTime) r^step,
// r = exp(-scale dt), so the inner loop needs no exponential.
function controlFromTimes(tau, nPlayers, nAtt, puckTime, m) {
  const ppcf = new Float64Array(nPlayers * m), sum = new Float64Array(m), d = new Float64Array(nPlayers);
  const scale = Math.PI / (Math.sqrt(3.0) * PARAMS.sigma);
  const late = new Float64Array(nPlayers * m), early = new Float64Array(m);
  for (let i = 0; i < late.length; i++) late[i] = Math.exp(scale * tau[i]);
  for (let c = 0; c < m; c++) early[c] = Math.exp(-scale * puckTime[c]);
  const rate = new Float64Array(nPlayers);
  for (let k = 0; k < nPlayers; k++) rate[k] = (k < nAtt ? PARAMS.lam : PARAMS.lam * PARAMS.kappaDef) * PARAMS.dt;
  const r = Math.exp(-scale * PARAMS.dt), steps = Math.ceil(PARAMS.tMax / PARAMS.dt);
  let decay = 1.0;                                   // r^step
  for (let step = 0; step < steps; step++, decay *= r) {
    let maxRem = 0.0;
    for (let c = 0; c < m; c++) {
      const rem = 1.0 - sum[c];
      if (rem <= 0.0) continue;                      // nothing left to hand out at this cell
      const q = early[c] * decay;
      let total = 0.0;
      for (let k = 0; k < nPlayers; k++) {
        d[k] = (rem * rate[k]) / (1.0 + late[k * m + c] * q);
        total += d[k];
      }
      // Never hand out more probability than what is left.
      const f = total > rem ? rem / total : 1.0;
      let s = 0.0;
      for (let k = 0; k < nPlayers; k++) { ppcf[k * m + c] += d[k] * f; s += ppcf[k * m + c]; }
      sum[c] = s;
      if (1.0 - s > maxRem) maxRem = 1.0 - s;
    }
    if (maxRem < PARAMS.tol) break;
  }
  for (let c = 0; c < m; c++) {
    const s = Math.max(sum[c], 1e-12);
    for (let k = 0; k < nPlayers; k++) ppcf[k * m + c] /= s;
  }
  return ppcf;
}

// Everything that depends on the rink only: grids and the scoring surface.
export class ThreatModel {
  constructor(rink, evalCell = 0.75, zoneDepth = 30.0, displayCell = 1.0) {
    this.rink = rink;
    this.grid = makeGrid(rink, evalCell, Math.min(zoneDepth, rink.length));
    this.display = makeGrid(rink, displayCell);
    // Only the cells with a chance of scoring contribute to any integral.
    const live = [];
    for (let c = 0; c < this.grid.n; c++) if (scoringAt(rink, this.grid.px[c], this.grid.py[c]) > 0.0) live.push(c);
    this.live = Int32Array.from(live);
    this.m = live.length;
    this.lx = new Float64Array(this.m); this.ly = new Float64Array(this.m); this.ls = new Float64Array(this.m);
    live.forEach((c, j) => {
      this.lx[j] = this.grid.px[c]; this.ly[j] = this.grid.py[c];
      this.ls[j] = scoringAt(rink, this.grid.px[c], this.grid.py[c]);
    });
  }

  // Off-puck scoring opportunity of the attack, averaged over the possible carriers.
  // Returns { obso, perAttacker, surface } with ``surface`` (threat per m²) on the full grid.
  evaluate(att, def, carriers) {
    const m = this.m, na = att.length, n = na + def.length;
    const out = { obso: 0.0, perAttacker: new Array(na).fill(0.0), surface: new Float32Array(this.grid.n) };
    if (na === 0 || carriers.length === 0) return out;
    const tau = timesToIntercept(att.concat(def), this.lx, this.ly, m);
    const puckTime = new Float64Array(m), surf = new Float64Array(m);
    for (const c of carriers) {
      const x0 = att[c][0], y0 = att[c][1];
      for (let j = 0; j < m; j++) puckTime[j] = len(this.lx[j] - x0, this.ly[j] - y0) / PARAMS.puckSpeed;
      const ppcf = controlFromTimes(tau, n, na, puckTime, m);
      for (let j = 0; j < m; j++) {
        const dens = this.ls[j] * transitionAt(this.lx[j], this.ly[j], x0, y0);
        let cAtt = 0.0;
        for (let k = 0; k < na; k++) { const v = ppcf[k * m + j]; cAtt += v; out.perAttacker[k] += v * dens * this.grid.area; }
        surf[j] += dens * cAtt;
      }
    }
    const nc = carriers.length;
    for (let k = 0; k < na; k++) { out.perAttacker[k] /= nc; out.obso += out.perAttacker[k]; }
    for (let j = 0; j < m; j++) out.surface[this.live[j]] = surf[j] / nc;
    return out;
  }

  // Probability that the attack controls each cell of the display grid (whole rink).
  pitchControl(att, def, carriers) {
    const g = this.display, out = new Float32Array(g.n);
    if (att.length === 0 || def.length === 0 || carriers.length === 0) return out.fill(att.length ? 1.0 : 0.0);
    const na = att.length, n = na + def.length;
    const tau = timesToIntercept(att.concat(def), g.px, g.py, g.n);
    const puckTime = new Float64Array(g.n);
    for (const c of carriers) {
      for (let j = 0; j < g.n; j++) puckTime[j] = len(g.px[j] - att[c][0], g.py[j] - att[c][1]) / PARAMS.puckSpeed;
      const ppcf = controlFromTimes(tau, n, na, puckTime, g.n);
      for (let j = 0; j < g.n; j++) { let s = 0.0; for (let k = 0; k < na; k++) s += ppcf[k * g.n + j]; out[j] += s / carriers.length; }
    }
    return out;
  }

  // ---- pitch-control coverage -------------------------------------------------------
  // One centroid step of the defenders ``xd`` ([x, y] each) against attackers ``xa``.
  _step(xd, xa, carriers) {
    const m = this.m, nd = xd.length, na = xa.length, n = nd + na;
    const { beta, minGap, carrierWeight } = COVERAGE, vMax = PARAMS.vMax;
    const numX = new Float64Array(nd), numY = new Float64Array(nd), den = new Float64Array(nd);
    const gap = new Float64Array(n), dist = new Float64Array(n), p = new Float64Array(n);
    const pos = xd.concat(xa);
    for (const c of carriers) {
      const x0 = xa[c][0], y0 = xa[c][1];
      for (let j = 0; j < m; j++) {
        const qx = this.lx[j], qy = this.ly[j];
        let trans = transitionAt(qx, qy, x0, y0);
        if (carrierWeight < 1.0 && na > 1) {
          let others = 0.0;
          for (let a = 0; a < na; a++) if (a !== c) others += transitionAt(qx, qy, xa[a][0], xa[a][1]);
          trans = carrierWeight * trans + ((1.0 - carrierWeight) * others) / (na - 1);
        }
        const phi = this.ls[j] * trans;
        const flight = len(qx - x0, qy - y0) / PARAMS.puckSpeed;
        const free = PARAMS.reach + vMax * Math.max(flight - PARAMS.tReact, 0.0);
        let best = -Infinity;
        for (let k = 0; k < n; k++) {
          dist[k] = len(qx - pos[k][0], qy - pos[k][1]);
          gap[k] = Math.max(dist[k] - free, 0.0);
          p[k] = (-beta * gap[k]) / vMax;
          if (p[k] > best) best = p[k];
        }
        let total = 0.0;
        for (let k = 0; k < n; k++) { p[k] = Math.exp(p[k] - best); total += p[k]; }
        let cAtt = 0.0;
        for (let k = nd; k < n; k++) cAtt += p[k] / total;
        for (let i = 0; i < nd; i++) {
          if (gap[i] <= 0.0) continue;
          const w = (phi * (p[i] / total) * cAtt * beta) / vMax, dd = Math.max(dist[i], 1e-9);
          numX[i] += (w * (qx - pos[i][0])) / dd;
          numY[i] += (w * (qy - pos[i][1])) / dd;
          den[i] += w / Math.max(gap[i], minGap);
        }
      }
    }
    return xd.map((q, i) => (den[i] > 1e-15 ? [q[0] + numX[i] / den[i], q[1] + numY[i] / den[i]] : [q[0], q[1]]));
  }

  // Positions the centroid iteration moves the defenders to.
  targets(xd, xa, carriers, iterations = COVERAGE.iterations) {
    let t = xd.map((q) => [q[0], q[1]]);
    if (!t.length || !xa.length || !carriers.length) return t;
    for (let it = 0; it < Math.max(1, iterations); it++) t = this._step(t, xa, carriers);
    return t;
  }

  // Move-to-centroid command [m/s] of every defender (players are [x, y, vx, vy]).
  nominal(def, att, carriers) {
    const la = COVERAGE.lookahead;
    const xd = def.map((p) => [p[0] + la * p[2], p[1] + la * p[3]]);
    const xa = att.map((p) => [p[0] + la * p[2], p[1] + la * p[3]]);
    const target = this.targets(xd, xa, carriers);
    return target.map((t, i) => {
      let ux = COVERAGE.gain * (t[0] - xd[i][0]), uy = COVERAGE.gain * (t[1] - xd[i][1]);
      const s = len(ux, uy);
      if (s > PARAMS.vMax) { ux *= PARAMS.vMax / s; uy *= PARAMS.vMax / s; }
      return [ux, uy];
    });
  }
}
