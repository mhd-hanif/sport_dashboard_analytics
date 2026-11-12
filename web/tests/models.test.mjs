// The JavaScript models must give the numbers of the Python package (golden.json is written
// by make_golden.py). Run from the repository root:  cd web && npm test
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { Scene, parseCsv } from "../js/engine.js";

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const golden = JSON.parse(readFileSync(here("./golden.json"), "utf8"));
const close = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), `${what}: ${a} vs ${b}`);

for (const g of golden) {
  const rows = parseCsv(readFileSync(here("../../assets/clip_scene/" + g.file), "utf8")).filter((r) => r.t <= g.tMax);
  const scene = new Scene(rows);

  test(`${g.file}: what is read from the clip`, () => {
    assert.equal(scene.fps, g.fps);
    assert.equal(scene.fpsSource, g.fpsSource);
    assert.equal(scene.defendedSide, g.side);
    assert.deepEqual(scene.goalies, g.goalies);
    assert.deepEqual(scene.ids, g.ids);
  });

  test(`${g.file}: threat, pitch control and suggested positions of a frame`, () => {
    for (const f of g.frames) {
      const st = scene.state(f.i), res = scene.evaluate(st, f.carrier);
      close(res.obso, f.obso, 1e-9, "obso");
      res.perAttacker.forEach((v, k) => close(v, f.perAttacker[k], 1e-9, "per attacker"));
      close(res.surface.reduce((a, v) => a + v, 0), f.surfaceSum, 1e-5, "surface sum");
      const pc = scene.pitchControl(st, f.carrier);
      close(pc.reduce((a, v) => a + v, 0) / pc.length, f.pcMean, 1e-6, "pitch control mean");
      f.pcSample.forEach((v, k) => close(pc[k * 97], v, 1e-6, "pitch control cell"));
      const sug = scene.suggest(st, f.carrier);
      sug.pos.forEach((p, d) => { close(p[0], f.suggestion[d][0], 1e-7, "suggested x"); close(p[1], f.suggestion[d][1], 1e-7, "suggested y"); });
      close(sug.obso, f.suggestionObso, 1e-8, "suggested obso");
    }
  });

  test(`${g.file}: replay of the reference defense`, () => {
    for (const t of g.timelines) {
      const tl = scene.timeline(t.carrier === null ? [] : [{ i: 0, id: t.carrier }]);
      assert.deepEqual(tl.idx, t.idx);
      assert.deepEqual(tl.defIds, t.defIds);
      tl.tracked.forEach((v, k) => close(v, t.tracked[k], 1e-8, "tracked"));
      // The replay is a closed loop that amplifies last-digit differences between the two
      // languages: exact at the start of the clip, close in threat afterwards.
      tl.reference.forEach((v, k) => close(v, t.reference[k], k < 8 ? 1e-9 : 1e-2, "reference"));
      close(tl.meanTracked, t.meanTracked, 1e-8, "mean tracked");
      close(tl.meanReference, t.meanReference, 1e-3, "mean reference");
      const nd = tl.defIds.length, at = (i, d, k) => tl.refPos[(i * nd + d) * 2 + k];
      for (const [i, want] of [[scene.T >> 1, t.refPosMid], [scene.T - 1, t.refPosLast]]) {
        want.forEach((p, d) => assert.ok(Math.hypot(at(i, d, 0) - p[0], at(i, d, 1) - p[1]) < 0.5, "reference defender within 0.5 m"));
      }
    }
  });
}

test("puck marks give the carrier of a frame", () => {
  const marks = [{ i: 10, id: "7" }, { i: 40, id: null }, { i: 60, id: "5" }];
  assert.equal(Scene.carrierAt(0, marks), null);
  assert.equal(Scene.carrierAt(10, marks), "7");
  assert.equal(Scene.carrierAt(45, marks), null);
  assert.equal(Scene.carrierAt(500, marks), "5");
});
