// Sunbears Defense Lab: the page. Drawing and interaction live here; the models run in
// two workers (worker.js), one for the frame on screen and one for the replay of the clip.

import { Scene, parseCsv } from "./engine.js";

const $ = (id) => document.getElementById(id);
const C = { offense: "#e5484d", defense: "#2f7fe0", ref: "#1e9e5a", tracked: "#2f7fe0", ink: "#0d1b2a", muted: "#7a8aa0" };
const THREAT_TOP = 2.0;       // % chance of a goal per 100 m² at which the threat colour saturates
const TRAIL = 40;             // frames
const PICK_RADIUS = 1.6;      // m

const root = $("app");
const BASE = root.dataset.clipBase || "";
const CLIPS = (root.dataset.clips || "5,0,1,2,3,4").split(",").map((n) => ({
  name: "Scene " + n.trim(), csv: `${BASE}scene_${n.trim()}.csv`, video: `${BASE}scene_${n.trim()}.mp4`,
}));

const S = {
  scene: null, gen: 0, i: 0, playing: false, speed: 1, loop: false, t0: 0, i0: 0,
  ov: { threat: true, reference: true, pc: false, voronoi: false, trails: false, velocity: false },
  edits: null, drag: null, marks: [], marksVer: 0,
  frame: null, cache: new Map(), busy: false, pending: null,
  timeline: null, progress: 0, tlReq: 0, moments: [], hover: null, videoOk: false,
};

const frameWorker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
const clipWorker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
const vid = $("video");

// ------------------------------------------------------------------ helpers
const pct = (v, d = 2) => (Number.isFinite(v) ? (100 * v).toFixed(d) + "%" : "–");
function delta(a, b) {
  if (!Number.isFinite(a) || !Number.isFinite(b) || b <= 0) return "–";
  const ch = 100 * (a / b - 1);
  return (ch < 0 ? "−" : "+") + Math.abs(ch).toFixed(0) + "%";
}
function interp(idx, vals, i) {
  if (!idx.length) return NaN;
  if (i <= idx[0]) return vals[0];
  for (let k = 1; k < idx.length; k++) if (i <= idx[k]) return vals[k - 1] + ((i - idx[k - 1]) / (idx[k] - idx[k - 1])) * (vals[k] - vals[k - 1]);
  return vals[vals.length - 1];
}
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }
function banner(text) { const b = $("banner"); b.hidden = !text; b.textContent = text || ""; }
const fps = () => S.scene.fps;
const carrierNow = () => Scene.carrierAt(S.i, S.marks);

// Players of the frame on screen, with the dragged positions applied.
function playersNow() {
  return S.scene.frames[S.i].all.map((p) => {
    const e = S.edits && S.edits[p.team + "|" + p.id];
    return e ? { ...p, x: e.x, y: e.y } : p;
  });
}
// Green markers: the replayed reference defense, or the suggestion for an edited frame.
function ghostsNow() {
  if (S.edits) {
    const f = S.frame;
    return f && f.edited && f.i === S.i && f.suggestion ? f.suggestion.ids.map((id, d) => ({ id, x: f.suggestion.pos[d][0], y: f.suggestion.pos[d][1] })) : [];
  }
  const tl = S.timeline;
  if (!tl || !tl.defIds.length) return [];
  const nd = tl.defIds.length;
  return tl.defIds.map((id, d) => ({ id, x: tl.refPos[(S.i * nd + d) * 2], y: tl.refPos[(S.i * nd + d) * 2 + 1] }));
}

// ------------------------------------------------------------------ loading
function loadRows(rows, videoUrl) {
  pause();
  S.gen++;
  S.scene = new Scene(rows);
  Object.assign(S, { i: 0, edits: null, marks: [], marksVer: 0, frame: null, timeline: null, moments: [], busy: false, pending: null });
  S.cache.clear();
  frameWorker.postMessage({ type: "load", rows });
  clipWorker.postMessage({ type: "load", rows });
  setVideo(videoUrl);
  $("note").textContent = `Read from the clip: ${S.scene.describe()}. The tracked attack does not react to the reference ` +
    "defense, so read the gap as how much threat was defendable, not as what would have happened.";
  banner("");
  requestTimeline();
  setFrame(0);
}
async function loadClip(clip) {
  try {
    const res = await fetch(clip.csv);
    if (!res.ok) throw new Error("file not found");
    loadRows(parseCsv(await res.text()), clip.video);
  } catch (err) {
    banner(`Could not open ${clip.name} (${err.message}). Serve the repository with a web server, or use “Open CSV / MP4”.`);
  }
}
async function openFiles(files) {
  const csv = [...files].find((f) => /\.csv$/i.test(f.name)), mp4 = [...files].find((f) => /\.mp4$/i.test(f.name));
  const url = mp4 ? URL.createObjectURL(mp4) : null;
  if (csv) {
    try { loadRows(parseCsv(await csv.text()), url); $("clip-select").value = ""; }
    catch (err) { banner(`Could not read ${csv.name}: ${err.message}`); }
  } else if (url && S.scene) setVideo(url);
}
function setVideo(url) {
  S.videoOk = false;
  vid.pause();
  $("video-empty").hidden = !!url;
  vid.hidden = !url;
  if (url) vid.src = url; else vid.removeAttribute("src");
}
vid.addEventListener("loadedmetadata", () => {
  // The video is the clock only if it lasts as long as the clip.
  const clip = S.scene.T / fps();
  S.videoOk = Math.abs(vid.duration - clip) < Math.max(1.0, 0.15 * clip);
  vid.currentTime = S.i / fps();
});
vid.addEventListener("error", () => { S.videoOk = false; vid.hidden = true; $("video-empty").hidden = false; });

// ------------------------------------------------------------------ model requests
function requestFrame() {
  if (!S.scene) return;
  const want = { threat: S.ov.threat, pc: S.ov.pc, suggest: !!S.edits, values: !S.playing && !S.drag };
  const key = S.edits ? null : `${S.i}|${S.marksVer}|${+want.threat}${+want.pc}${+want.values}`;
  if (key && S.cache.has(key)) { S.frame = S.cache.get(key); S.pending = null; return; }
  S.pending = { type: "frame", gen: S.gen, key, i: S.i, edits: S.edits, marks: S.marks, want };
  pump();
}
function pump() {
  if (S.busy || !S.pending) return;
  S.busy = true;
  frameWorker.postMessage(S.pending);
  S.pending = null;
}
frameWorker.onmessage = (e) => {
  S.busy = false;
  const m = e.data;
  if (m.gen === S.gen) {
    if (m.key) { if (S.cache.size > 900) S.cache.clear(); S.cache.set(m.key, m); }
    if (!!S.edits === m.edited && (S.playing || m.i === S.i)) { S.frame = m; drawRink(); drawPanels(); }
  }
  pump();
};
function requestTimeline() {
  S.timeline = null; S.progress = 0; S.moments = [];
  clipWorker.postMessage({ type: "timeline", gen: S.gen, req: ++S.tlReq, marks: S.marks });
}
clipWorker.onmessage = (e) => {
  const m = e.data;
  if (m.gen !== S.gen || m.req !== S.tlReq) return;
  if (m.type === "progress") { S.progress = m.done; drawChart(); return; }
  S.timeline = m.result;
  findMoments();
  drawAll();
};

// Moments where the reference defense concedes much less than the tracked one.
function findMoments() {
  const tl = S.timeline, out = [];
  const order = tl.idx.map((_, k) => k).filter((k) => tl.tracked[k] > 0 && tl.tracked[k] - tl.reference[k] > 0.1 * tl.tracked[k])
    .sort((a, b) => (tl.tracked[b] - tl.reference[b]) - (tl.tracked[a] - tl.reference[a]));
  for (const k of order) {
    if (out.length >= 4) break;
    if (out.every((o) => Math.abs(tl.idx[o] - tl.idx[k]) >= 2 * fps())) out.push(k);
  }
  S.moments = out.sort((a, b) => a - b).map((k) => ({ i: tl.idx[k], change: delta(tl.reference[k], tl.tracked[k]) }));
}

// ------------------------------------------------------------------ playback
function setFrame(i, fromClock = false) {
  S.i = Math.max(0, Math.min(S.scene.T - 1, i));
  S.edits = null;
  if (!fromClock) {
    S.t0 = performance.now(); S.i0 = S.i;
    if (!vid.hidden && vid.readyState > 0) vid.currentTime = S.i / fps();
  }
  requestFrame();
  drawAll();
}
function play() {
  if (!S.scene) return;
  if (S.i >= S.scene.T - 1) setFrame(0);
  S.playing = true; S.edits = null; S.t0 = performance.now(); S.i0 = S.i;
  $("btn-play").textContent = "Pause";
  if (S.videoOk) {
    S.vTarget = S.i / fps(); S.vCheck = true;
    vid.playbackRate = S.speed; vid.currentTime = S.vTarget; vid.play().catch(() => { S.videoOk = false; });
  }
  requestAnimationFrame(tick);
}
function pause() {
  if (!S.playing) return;
  S.playing = false;
  $("btn-play").textContent = "Play";
  vid.pause();
  requestFrame();          // now with the per-defender values
  drawAll();
}
function tick(now) {
  if (!S.playing) return;
  const last = S.scene.T - 1, wall = S.i0 + Math.floor(((now - S.t0) / 1000) * fps() * S.speed);
  if (S.videoOk && S.vCheck && !vid.seeking && vid.readyState >= 2) {
    // Some servers cannot seek inside a video (no byte ranges): the video then restarts from
    // zero. If it did not land where it was sent, the timer keeps the clock.
    S.vCheck = false;
    if (Math.abs(vid.currentTime - S.vTarget) > 0.75 + ((now - S.t0) / 1000) * S.speed) S.videoOk = false;
  }
  let i = S.videoOk && !S.vCheck ? (vid.ended ? last : Math.round(vid.currentTime * fps())) : wall;
  if (i >= last) {
    if (!S.loop) { setFrame(last, true); pause(); return; }
    i = 0; S.t0 = now; S.i0 = 0;
    if (S.videoOk) { S.vTarget = 0; S.vCheck = true; vid.currentTime = 0; vid.play().catch(() => { S.videoOk = false; }); }
  }
  if (i !== S.i) setFrame(i, true);
  requestAnimationFrame(tick);
}

// ------------------------------------------------------------------ rink
const rinkCanvas = $("rink"), rctx = rinkCanvas.getContext("2d");
const heat = document.createElement("canvas"), hctx = heat.getContext("2d");
let view = { k: 1, pad: 0, w: 0, h: 0 };
const B = () => S.scene.bounds;
const X = (x) => view.pad + (x - B().xMin) * view.k;
const Y = (y) => view.pad + (B().yMax - y) * view.k;

function fit(canvas, cssH) {
  const dpr = window.devicePixelRatio || 1, w = canvas.clientWidth;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(cssH * dpr)) {
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(cssH * dpr);
    canvas.style.height = cssH + "px";
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return w;
}
function rinkPath(ctx) {
  const b = B(), r = 8.5 * view.k;
  ctx.beginPath();
  ctx.roundRect(X(b.xMin), Y(b.yMax), (b.xMax - b.xMin) * view.k, (b.yMax - b.yMin) * view.k, r);
}
function drawIce(ctx) {
  const b = B(), L = b.xMax - b.xMin, mid = (b.yMin + b.yMax) / 2, k = view.k;
  rinkPath(ctx); ctx.fillStyle = "#fbfdff"; ctx.fill();
  ctx.save(); rinkPath(ctx); ctx.clip();
  const vline = (x, color, w) => { ctx.strokeStyle = color; ctx.lineWidth = w; ctx.beginPath(); ctx.moveTo(X(x), Y(b.yMax)); ctx.lineTo(X(x), Y(b.yMin)); ctx.stroke(); };
  vline(b.xMin + 21.8, "#7fa8e6", 0.3 * k); vline(b.xMax - 21.8, "#7fa8e6", 0.3 * k);
  vline(b.xMin + L / 2, "#ef9a9a", 0.3 * k);
  vline(b.xMin + 4.3, "#ef9a9a", 1.2); vline(b.xMax - 4.3, "#ef9a9a", 1.2);
  ctx.strokeStyle = "#f2b6b6"; ctx.lineWidth = 1.2;
  const circle = (x, y, r) => { ctx.beginPath(); ctx.arc(X(x), Y(y), r * k, 0, 2 * Math.PI); ctx.stroke(); };
  circle(b.xMin + L / 2, mid, 4.5);
  for (const x of [b.xMin + 10.3, b.xMax - 10.3]) for (const y of [mid - 6.75, mid + 6.75]) circle(x, y, 4.5);
  for (const [x, dir] of [[b.xMin + 4.3, 1], [b.xMax - 4.3, -1]]) {      // crease and net
    ctx.fillStyle = "rgba(127,168,230,0.35)";
    ctx.beginPath(); ctx.arc(X(x), Y(mid), 1.8 * k, -Math.PI / 2, Math.PI / 2, dir < 0); ctx.fill();
    ctx.strokeStyle = "#c0392b"; ctx.lineWidth = 2;
    ctx.strokeRect(X(x) - (dir > 0 ? 1.1 * k : 0), Y(mid + 0.915), 1.1 * k, 1.83 * k);
  }
  ctx.restore();
}
// Paint a field given on a grid of the canonical frame (defended goal on the left).
function drawField(ctx, grid, values, x0, x1, color) {
  const { nx, ny } = grid, img = hctx.createImageData(nx, ny), flip = S.scene.mirrored;
  heat.width = nx; heat.height = ny;
  for (let r = 0; r < ny; r++) for (let q = 0; q < nx; q++) {
    const c = color(values[r * nx + q]), o = ((ny - 1 - r) * nx + (flip ? nx - 1 - q : q)) * 4;
    img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = c[3];
  }
  hctx.putImageData(img, 0, 0);
  const a = S.scene.toDashboard(x0, 0)[0], b = S.scene.toDashboard(x1, 0)[0], left = Math.min(a, b);
  ctx.save(); rinkPath(ctx); ctx.clip();
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
  ctx.drawImage(heat, X(left), Y(B().yMax), Math.abs(b - a) * view.k, (B().yMax - B().yMin) * view.k);
  ctx.restore();
}
const THREAT_STOPS = [[0, [255, 255, 255, 0]], [0.05, [250, 215, 160, 140]], [0.25, [245, 176, 65, 215]], [0.6, [211, 84, 0, 230]], [1, [120, 40, 31, 242]]];
function threatColor(v) {
  const z = Math.min(1, (1e4 * v) / THREAT_TOP);
  for (let s = 1; s < THREAT_STOPS.length; s++) if (z <= THREAT_STOPS[s][0]) {
    const [z0, c0] = THREAT_STOPS[s - 1], [z1, c1] = THREAT_STOPS[s], w = (z - z0) / (z1 - z0);
    return c0.map((c, j) => c + w * (c1[j] - c));
  }
  return THREAT_STOPS[THREAT_STOPS.length - 1][1];
}
function controlColor(v) {      // defense blue - white - offense red
  const w = Math.abs(v - 0.5) * 2, c = v < 0.5 ? [47, 127, 224] : [229, 72, 77];
  return [255 + w * (c[0] - 255), 255 + w * (c[1] - 255), 255 + w * (c[2] - 255), 135];
}
// Voronoi cell of ``p`` among ``pts``: the rink rectangle cut by every bisector.
function voronoiCell(p, pts) {
  const b = B();
  let poly = [[b.xMin, b.yMin], [b.xMax, b.yMin], [b.xMax, b.yMax], [b.xMin, b.yMax]];
  for (const q of pts) {
    if (q === p || !poly.length) continue;
    const nx = q.x - p.x, ny = q.y - p.y, c = nx * (p.x + q.x) / 2 + ny * (p.y + q.y) / 2, out = [];
    for (let j = 0; j < poly.length; j++) {
      const a = poly[j], d = poly[(j + 1) % poly.length], da = nx * a[0] + ny * a[1] - c, dd = nx * d[0] + ny * d[1] - c;
      if (da <= 0) out.push(a);
      if (da * dd < 0) { const t = da / (da - dd); out.push([a[0] + t * (d[0] - a[0]), a[1] + t * (d[1] - a[1])]); }
    }
    poly = out;
  }
  return poly;
}

function drawRink() {
  if (!S.scene) return;
  const b = B(), w = rinkCanvas.clientWidth, pad = 10;
  view = { pad, k: (w - 2 * pad) / (b.xMax - b.xMin), w };
  view.h = (b.yMax - b.yMin) * view.k + 2 * pad;
  fit(rinkCanvas, view.h);
  const ctx = rctx, k = view.k, players = playersNow(), f = S.frame, model = S.scene.model;
  ctx.clearRect(0, 0, w, view.h);
  drawIce(ctx);
  if (S.ov.pc && f && f.pc) drawField(ctx, model.display, f.pc, 0, S.scene.rink.length, controlColor);
  if (S.ov.threat && f && f.surface) drawField(ctx, model.grid, f.surface, 0, model.grid.xs.length * (model.grid.xs[1] - model.grid.xs[0]), threatColor);
  if (S.ov.voronoi) {
    ctx.save(); rinkPath(ctx); ctx.clip();
    for (const p of players) {
      const poly = voronoiCell(p, players);
      if (poly.length < 3) continue;
      ctx.beginPath(); poly.forEach((q, j) => (j ? ctx.lineTo(X(q[0]), Y(q[1])) : ctx.moveTo(X(q[0]), Y(q[1])))); ctx.closePath();
      ctx.fillStyle = p.team === "D" ? "rgba(47,127,224,0.10)" : "rgba(229,72,77,0.10)"; ctx.fill();
      ctx.strokeStyle = "rgba(13,27,42,0.18)"; ctx.lineWidth = 1; ctx.stroke();
    }
    ctx.restore();
  }
  rinkPath(ctx); ctx.strokeStyle = "#8d9db3"; ctx.lineWidth = 2.5; ctx.stroke();

  if (S.ov.trails) {
    const paths = new Map();
    for (let j = Math.max(0, S.i - TRAIL); j <= S.i; j++) for (const p of S.scene.frames[j].all) {
      const key = p.team + "|" + p.id;
      if (!paths.has(key)) paths.set(key, { team: p.team, pts: [] });
      paths.get(key).pts.push(p);
    }
    ctx.lineWidth = 2;
    for (const { team, pts } of paths.values()) {
      ctx.strokeStyle = team === "D" ? "rgba(47,127,224,0.45)" : "rgba(229,72,77,0.45)";
      ctx.beginPath(); pts.forEach((p, j) => (j ? ctx.lineTo(X(p.x), Y(p.y)) : ctx.moveTo(X(p.x), Y(p.y)))); ctx.stroke();
    }
  }
  if (S.ov.velocity) {
    ctx.lineWidth = 2;
    for (const p of players) {
      const sp = Math.hypot(p.vx, p.vy);
      if (sp < 0.2) continue;
      const x1 = p.x + 0.5 * p.vx, y1 = p.y + 0.5 * p.vy, a = Math.atan2(Y(y1) - Y(p.y), X(x1) - X(p.x));    // where he is in 0.5 s
      ctx.strokeStyle = ctx.fillStyle = p.team === "D" ? C.defense : C.offense;
      ctx.beginPath(); ctx.moveTo(X(p.x), Y(p.y)); ctx.lineTo(X(x1), Y(y1)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(X(x1), Y(y1));
      ctx.lineTo(X(x1) - 7 * Math.cos(a - 0.45), Y(y1) - 7 * Math.sin(a - 0.45));
      ctx.lineTo(X(x1) - 7 * Math.cos(a + 0.45), Y(y1) - 7 * Math.sin(a + 0.45)); ctx.fill();
    }
  }
  const r = Math.max(8, 0.75 * k);
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  if (S.ov.reference) {
    const here = new Map(players.filter((p) => p.team === "D").map((p) => [p.id, p]));
    ctx.font = `600 ${Math.round(r * 0.95)}px system-ui, sans-serif`;
    for (const g of ghostsNow()) {
      const p = here.get(g.id);
      if (!p) continue;
      ctx.setLineDash([3, 4]); ctx.strokeStyle = "rgba(30,158,90,0.85)"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(X(p.x), Y(p.y)); ctx.lineTo(X(g.x), Y(g.y)); ctx.stroke(); ctx.setLineDash([]);
      ctx.beginPath(); ctx.arc(X(g.x), Y(g.y), r * 1.05, 0, 2 * Math.PI);
      ctx.fillStyle = "rgba(30,158,90,0.30)"; ctx.fill(); ctx.strokeStyle = C.ref; ctx.lineWidth = 2; ctx.stroke();
      ctx.fillStyle = "#0b3d20"; ctx.fillText(g.id + "′", X(g.x), Y(g.y) + 0.5);
    }
  }
  const carrier = carrierNow();
  ctx.font = `700 ${Math.round(r * 1.0)}px system-ui, sans-serif`;
  for (const p of players) {
    const color = p.team === "D" ? C.defense : C.offense, edited = S.edits && S.edits[p.team + "|" + p.id];
    ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), r, 0, 2 * Math.PI);
    ctx.fillStyle = p.goalie ? "#ffffff" : color; ctx.fill();
    ctx.strokeStyle = edited ? C.ink : p.goalie ? color : "#ffffff"; ctx.lineWidth = edited ? 2.5 : 2; ctx.stroke();
    ctx.fillStyle = p.goalie ? color : "#ffffff"; ctx.fillText(p.goalie ? "G" : p.id, X(p.x), Y(p.y) + 0.5);
    if (p.team === "O" && p.id === carrier) {       // the puck
      ctx.beginPath(); ctx.arc(X(p.x) + r * 0.95, Y(p.y) + r * 0.95, r * 0.42, 0, 2 * Math.PI);
      ctx.fillStyle = C.ink; ctx.fill(); ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.5; ctx.stroke();
    }
  }
  $("edit-badge").hidden = $("reset-frame").hidden = !S.edits;
}

// Dragging players and marking the puck.
function rinkPoint(e) {
  const box = rinkCanvas.getBoundingClientRect(), b = B();
  return { x: (e.clientX - box.left - view.pad) / view.k + b.xMin, y: b.yMax - (e.clientY - box.top - view.pad) / view.k };
}
rinkCanvas.addEventListener("pointerdown", (e) => {
  if (!S.scene) return;
  const q = rinkPoint(e);
  let best = null, bd = Math.max(PICK_RADIUS, 16 / view.k);
  for (const p of playersNow()) { const d = Math.hypot(p.x - q.x, p.y - q.y); if (!p.goalie && d < bd) { bd = d; best = p; } }
  if (!best) return;
  pause();
  S.drag = { key: best.team + "|" + best.id, team: best.team, id: best.id, sx: e.clientX, sy: e.clientY, moved: false };
  rinkCanvas.setPointerCapture(e.pointerId);
});
rinkCanvas.addEventListener("pointermove", (e) => {
  const d = S.drag;
  if (!d) return;
  if (!d.moved && Math.hypot(e.clientX - d.sx, e.clientY - d.sy) < 4) return;
  d.moved = true;
  const q = rinkPoint(e), b = B();
  S.edits = { ...(S.edits || {}), [d.key]: { x: Math.min(b.xMax - 0.5, Math.max(b.xMin + 0.5, q.x)), y: Math.min(b.yMax - 0.5, Math.max(b.yMin + 0.5, q.y)) } };
  requestFrame();
  drawRink();
});
function endDrag() {
  const d = S.drag;
  if (!d) return;
  S.drag = null;
  if (!d.moved && d.team === "O") toggleMark(d.id); else if (d.moved) { requestFrame(); drawAll(); }
}
rinkCanvas.addEventListener("pointerup", endDrag);
rinkCanvas.addEventListener("pointercancel", endDrag);

// "Attacker ``id`` has the puck from this frame on" (or nobody known, if he already has it).
function toggleMark(id) {
  const next = carrierNow() === id ? null : id;
  const marks = S.marks.filter((m) => m.i !== S.i).concat([{ i: S.i, id: next }]).sort((a, b) => a.i - b.i);
  S.marks = marks.filter((m, j) => m.id !== (j ? marks[j - 1].id : null));      // drop marks that change nothing
  marksChanged();
}
function marksChanged() {
  S.marksVer++; S.cache.clear();
  requestTimeline(); requestFrame(); drawAll();
}

// ------------------------------------------------------------------ panels
function tile(label, value, note, color) {
  const t = el("div", "tile");
  t.style.setProperty("--accent", color);
  t.append(el("div", "tile__label", label), el("div", "tile__value", value), el("div", "tile__note", note));
  return t;
}
function row(id, frac, value, extra, color, puck) {
  const r = el("div", "row"), track = el("div", "row__track"), fill = el("div", "row__fill");
  fill.style.width = (100 * Math.max(0, Math.min(1, frac))).toFixed(1) + "%"; fill.style.background = color;
  track.append(fill);
  r.append(el("div", "row__id" + (puck ? " row__id--puck" : ""), "#" + id), track, el("div", "row__value", value), el("div", "row__extra", extra));
  return r;
}
function drawPanels() {
  if (!S.scene) return;
  const f = S.frame && !!S.edits === S.frame.edited ? S.frame : null;      // may lag the screen by a frame or two
  const tl = S.timeline, now = f ? f.obso : NaN, tiles = [];
  if (S.edits) {
    const sug = f && f.suggestion ? f.suggestion.obso : NaN;
    tiles.push(tile("Scoring threat, edited frame", pct(now), "With the players where you put them", C.tracked));
    tiles.push(tile("With suggested positions", pct(sug), `${delta(sug, now)} with the defenders on the green markers`, C.ref));
  } else {
    const ref = tl ? interp(tl.idx, tl.reference, S.i) : NaN;
    tiles.push(tile("Scoring threat now", pct(now), "Against the tracked defense", C.tracked));
    tiles.push(tile("Reference defense now", pct(ref), tl ? `${delta(ref, now)} vs tracked` : "Replaying the clip…", C.ref));
  }
  if (tl && Number.isFinite(tl.meanTracked) && tl.meanTracked > 0) {
    const ch = 100 * (tl.meanReference / tl.meanTracked - 1);
    tiles.push(tile("Clip average, tracked → reference", `${pct(tl.meanTracked)} → ${pct(tl.meanReference)}`,
      `Reference concedes ${Math.abs(ch).toFixed(0)}% ${ch <= 0 ? "less" : "more"} over the clip`, C.muted));
  } else tiles.push(tile("Clip average, tracked → reference", "–", tl ? "Needs attackers and defenders" : "Replaying the clip…", C.muted));
  $("tiles").replaceChildren(...tiles);

  const defs = $("defenders"), atts = $("attackers");
  if (!f) { defs.replaceChildren(); atts.replaceChildren(); return; }
  const players = playersNow(), ghosts = new Map(ghostsNow().map((g) => [g.id, g]));
  const top = f.values ? Math.max(...f.values, 1e-9) : 1;
  defs.replaceChildren(...f.defIds.map((id, d) => {
    const p = players.find((q) => q.team === "D" && q.id === id), g = ghosts.get(id);
    const off = p && g ? Math.hypot(p.x - g.x, p.y - g.y).toFixed(1) + " m" : "–";
    return row(id, f.values ? f.values[d] / top : 0, f.values ? pct(f.values[d]) : "·", off, C.defense, false);
  }));
  const carrier = carrierNow(), topA = Math.max(...f.perAttacker, 1e-9);
  const order = f.attIds.map((_, k) => k).sort((a, b) => f.perAttacker[b] - f.perAttacker[a]);
  atts.replaceChildren(...order.map((k) => row(f.attIds[k], f.perAttacker[k] / topA, pct(f.perAttacker[k]), f.attIds[k] === carrier ? "puck" : "", C.offense, f.attIds[k] === carrier)));
}

// ------------------------------------------------------------------ timeline chart
const chart = $("chart"), cctx = chart.getContext("2d");
let cg = null;        // chart geometry
function drawChart() {
  if (!S.scene) return;
  const H = 230, w = fit(chart, H), ctx = cctx, m = { l: 46, r: 14, t: 16, b: 26 };
  const dur = (S.scene.T - 1) / fps(), tl = S.timeline;
  ctx.clearRect(0, 0, w, H);
  ctx.font = "12px system-ui, sans-serif";
  const px = (t) => m.l + (t / Math.max(dur, 1e-9)) * (w - m.l - m.r);
  cg = { m, w, dur };
  if (!tl) {
    ctx.fillStyle = "#9fb0c7"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(`Replaying the clip with the reference defense… ${Math.round(100 * S.progress)}%`, w / 2, H / 2 - 12);
    ctx.fillStyle = "#22344d"; ctx.fillRect(w / 2 - 120, H / 2 + 4, 240, 6);
    ctx.fillStyle = C.ref; ctx.fillRect(w / 2 - 120, H / 2 + 4, 240 * S.progress, 6);
    return;
  }
  if (!tl.idx.length) {
    ctx.fillStyle = "#9fb0c7"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText("The timeline needs attackers and defenders in the clip.", w / 2, H / 2);
    return;
  }
  const top = Math.max(...tl.tracked.filter(Number.isFinite), ...tl.reference) * 1.12 || 0.01;
  const py = (v) => H - m.b - (v / top) * (H - m.t - m.b), t = tl.idx.map((i) => i / fps());
  // grid and axes
  ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.lineWidth = 1;
  const stepY = [0.0025, 0.005, 0.01, 0.02, 0.05].find((s) => top / s <= 5) || 0.1;
  for (let v = 0; v <= top; v += stepY) {
    ctx.strokeStyle = "#1e2f47"; ctx.beginPath(); ctx.moveTo(m.l, py(v)); ctx.lineTo(w - m.r, py(v)); ctx.stroke();
    ctx.fillStyle = "#8ea0b8"; ctx.fillText((100 * v).toFixed(stepY < 0.005 ? 2 : 1) + "%", m.l - 8, py(v));
  }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  const stepX = [1, 2, 5, 10, 20, 30, 60].find((s) => dur / s <= 10) || 120;
  for (let s = 0; s <= dur + 1e-9; s += stepX) ctx.fillText(s + " s", px(s), H - m.b + 8);
  // gap between the two defenses
  for (let k = 1; k < t.length; k++) {
    if (![tl.tracked[k - 1], tl.tracked[k]].every(Number.isFinite)) continue;
    const better = tl.reference[k - 1] + tl.reference[k] <= tl.tracked[k - 1] + tl.tracked[k];
    ctx.fillStyle = better ? "rgba(30,158,90,0.28)" : "rgba(229,160,60,0.28)";
    ctx.beginPath(); ctx.moveTo(px(t[k - 1]), py(tl.tracked[k - 1])); ctx.lineTo(px(t[k]), py(tl.tracked[k]));
    ctx.lineTo(px(t[k]), py(tl.reference[k])); ctx.lineTo(px(t[k - 1]), py(tl.reference[k - 1])); ctx.fill();
  }
  for (const [vals, color] of [[tl.reference, "#35c47a"], [tl.tracked, "#5aa2f5"]]) {
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.beginPath();
    let pen = false;
    vals.forEach((v, k) => { if (!Number.isFinite(v)) { pen = false; return; } pen ? ctx.lineTo(px(t[k]), py(v)) : ctx.moveTo(px(t[k]), py(v)); pen = true; });
    ctx.stroke();
  }
  // key moments, puck marks, hover and the cursor of the frame on screen
  ctx.fillStyle = "#f5c451";
  for (const mo of S.moments) { const x = px(mo.i / fps()); ctx.beginPath(); ctx.moveTo(x - 5, m.t - 10); ctx.lineTo(x + 5, m.t - 10); ctx.lineTo(x, m.t - 2); ctx.fill(); }
  ctx.textBaseline = "bottom"; ctx.font = "600 11px system-ui, sans-serif";
  for (const mk of S.marks) {
    const x = px(mk.i / fps());
    ctx.fillStyle = "#e8eef7"; ctx.beginPath(); ctx.arc(x, H - m.b, 4, 0, 2 * Math.PI); ctx.fill();
    ctx.fillText(mk.id === null ? "?" : "#" + mk.id, x, H - m.b - 6);
  }
  if (S.hover !== null) { ctx.strokeStyle = "rgba(232,238,247,0.35)"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(px(S.hover), m.t); ctx.lineTo(px(S.hover), H - m.b); ctx.stroke(); }
  const xc = px(S.i / fps());
  ctx.strokeStyle = "#e8eef7"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(xc, m.t - 4); ctx.lineTo(xc, H - m.b); ctx.stroke();

  const at = S.hover !== null ? S.hover : S.i / fps(), i = at * fps();
  const a = interp(tl.idx, tl.tracked, i), b = interp(tl.idx, tl.reference, i);
  $("chart-readout").textContent = `${at.toFixed(1)} s · tracked ${pct(a)} · reference ${pct(b)} (${delta(b, a)})`;
}
function chartTime(e) {
  const box = chart.getBoundingClientRect();
  return Math.max(0, Math.min(cg.dur, ((e.clientX - box.left - cg.m.l) / (cg.w - cg.m.l - cg.m.r)) * cg.dur));
}
let scrubbing = false;
chart.addEventListener("pointerdown", (e) => { if (!cg) return; scrubbing = true; chart.setPointerCapture(e.pointerId); pause(); setFrame(Math.round(chartTime(e) * fps())); });
chart.addEventListener("pointermove", (e) => { if (!cg) return; S.hover = chartTime(e); if (scrubbing) setFrame(Math.round(S.hover * fps())); else drawChart(); });
chart.addEventListener("pointerup", () => { scrubbing = false; });
chart.addEventListener("pointerleave", () => { S.hover = null; drawChart(); });

function drawFoot() {
  const last = S.scene.T - 1;
  $("readout").textContent = `${(S.i / fps()).toFixed(1)} s · frame ${S.i} / ${last}`;
  const mo = $("moments");
  if (!S.timeline) mo.replaceChildren(el("span", "moments__none", "after the replay"));
  else if (!S.moments.length) mo.replaceChildren(el("span", "moments__none", "none: the two defenses stay close"));
  else mo.replaceChildren(...S.moments.map((m) => {
    const b = el("button", "moment", `${(m.i / fps()).toFixed(1)} s  ${m.change}`);
    b.type = "button"; b.addEventListener("click", () => { pause(); setFrame(m.i); });
    return b;
  }));
  const known = S.marks.length > 0;
  $("marks-text").textContent = known
    ? "Puck: " + S.marks.map((m) => `${(m.i / fps()).toFixed(1)} s ${m.id === null ? "unknown" : "#" + m.id}`).join(" → ")
    : "Puck carrier unknown: results are averaged over the attackers.";
  $("clear-marks").hidden = !known;
}
function drawAll() { drawRink(); drawPanels(); drawChart(); drawFoot(); }

// ------------------------------------------------------------------ controls
$("btn-play").addEventListener("click", () => (S.playing ? pause() : play()));
$("btn-prev").addEventListener("click", () => { pause(); setFrame(S.i - 1); });
$("btn-next").addEventListener("click", () => { pause(); setFrame(S.i + 1); });
$("speed").addEventListener("change", (e) => { S.speed = Number(e.target.value) || 1; vid.playbackRate = S.speed; S.t0 = performance.now(); S.i0 = S.i; });
$("loop").addEventListener("change", (e) => { S.loop = e.target.checked; });
$("reset-frame").addEventListener("click", () => { S.edits = null; requestFrame(); drawAll(); });
$("clear-marks").addEventListener("click", () => { S.marks = []; marksChanged(); });
document.querySelectorAll("#overlays input").forEach((box) => {
  box.checked = S.ov[box.dataset.ov];
  box.addEventListener("change", () => { S.ov[box.dataset.ov] = box.checked; requestFrame(); drawRink(); });
});
const sel = $("clip-select");
CLIPS.forEach((c, k) => { const o = el("option", "", c.name); o.value = String(k); sel.append(o); });
const own = el("option", "", "Your file"); own.value = ""; own.hidden = true; sel.append(own);
sel.addEventListener("change", () => { if (sel.value !== "") loadClip(CLIPS[Number(sel.value)]); });
$("file-input").addEventListener("change", (e) => { openFiles(e.target.files); e.target.value = ""; });
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => { e.preventDefault(); if (e.dataTransfer && e.dataTransfer.files.length) openFiles(e.dataTransfer.files); });
window.addEventListener("keydown", (e) => {
  if (!S.scene || /^(INPUT|SELECT|TEXTAREA|BUTTON)$/.test(e.target.tagName)) return;
  if (e.code === "Space") { e.preventDefault(); S.playing ? pause() : play(); }
  if (e.code === "ArrowLeft") { pause(); setFrame(S.i - 1); }
  if (e.code === "ArrowRight") { pause(); setFrame(S.i + 1); }
});
new ResizeObserver(() => { if (S.scene) { drawRink(); drawChart(); } }).observe(document.body);

loadClip(CLIPS[0]);
