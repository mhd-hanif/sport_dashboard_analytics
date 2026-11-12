// Runs the models off the main thread. Two instances are used: one answers per-frame
// requests while the other replays the clip with the reference defense.

import { Scene } from "./engine.js";

let scene = null;
let job = 0;

self.onmessage = async (e) => {
  const m = e.data;
  if (m.type === "load") {
    scene = new Scene(m.rows);
    job++;
    return;
  }
  if (!scene) return;

  if (m.type === "frame") {
    const st = scene.state(m.i, m.edits);
    const carrier = Scene.carrierAt(m.i, m.marks);
    const res = scene.evaluate(st, carrier);
    const out = {
      type: "frame", gen: m.gen, key: m.key, i: m.i, edited: !!m.edits, carrier,
      obso: res.obso, perAttacker: res.perAttacker, attIds: st.attIds, defIds: st.defIds,
    };
    if (m.want.threat) out.surface = res.surface;
    if (m.want.pc) out.pc = scene.pitchControl(st, carrier);
    if (m.want.suggest) out.suggestion = scene.suggest(st, carrier);
    if (m.want.values) out.values = scene.defenderValues(st, carrier, res.obso);
    self.postMessage(out);
    return;
  }

  if (m.type === "timeline") {
    const mine = ++job;
    const it = scene.timelineSteps(m.marks);
    let last = performance.now();
    for (;;) {
      const s = it.next();
      if (s.done) { self.postMessage({ type: "timeline", gen: m.gen, req: m.req, result: s.value }); return; }
      if (performance.now() - last > 80) {
        self.postMessage({ type: "progress", gen: m.gen, req: m.req, done: s.value });
        await new Promise((r) => setTimeout(r));          // let a newer request in
        if (mine !== job) return;
        last = performance.now();
      }
    }
  }
};
