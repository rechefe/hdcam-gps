// Acceptance tests for Figure 1, from learning/specs/figure-1-correlation.md.
// Run: node --test learning/explainer/tests/figure1.test.mjs
// Needs Playwright with a Chromium build (preinstalled in Claude Code cloud sessions).
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(process.env.PLAYWRIGHT_REQUIRE_FROM || "/opt/node22/lib/node_modules/");
const { chromium } = require("playwright");

const here = dirname(fileURLToPath(import.meta.url));
const page = readFileSync(join(here, "..", "index.html"), "utf8");
const dir = mkdtempSync(join(tmpdir(), "fig1-"));
const file = join(dir, "page.html");
// The artifact host wraps the page in a skeleton; do the same, and drop the CDN script the sandbox cannot reach.
writeFileSync(file, `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"></head><body>${page.replace(/<script src="https:\/\/cdn[^"]*"[^>]*><\/script>/g, "")}</body></html>`);

let browser;
before(async () => { browser = await chromium.launch(); });
after(async () => { await browser.close(); });

async function open(width, deviceScaleFactor = 1) {
  const p = await browser.newPage({ viewport: { width, height: 900 }, deviceScaleFactor });
  const errors = [];
  p.on("pageerror", (e) => errors.push(e.message));
  p.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  await p.goto("file://" + file);
  await p.waitForFunction(() => document.querySelector("#fig1")?.dataset.ymax);
  return { p, errors };
}
const settle = (p) => p.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
async function set(p, id, value) {
  await p.evaluate(([id, value]) => { const el = document.querySelector(id); el.value = value; el.dispatchEvent(new Event("input")); el.dispatchEvent(new Event("change")); }, [id, String(value)]);
  await settle(p);
}
const state = (p) => p.$eval("#fig1", (f) => ({ ...f.dataset }));
const boxes = (p) => p.evaluate(() => ["#fig1", "#f1-c", "#f1-p"].map((id) => {
  const el = document.querySelector(id); if (!el) return `${id} missing`;
  const b = el.getBoundingClientRect(); return `${id} ${Math.round(b.top + scrollY)} ${Math.round(b.height)}`;
}).join(" | "));
const BIN = (hz) => (hz + 5000) / 500;

test("the figure does not move or resize when any control changes", async () => {
  for (const width of [1280, 1000, 820, 400]) {
    const { p } = await open(width);
    const ref = await boxes(p);
    assert.ok(!ref.includes("missing"), ref);
    const seen = [];
    for (let b = 0; b <= 20; b++) { await set(p, "#f1-dop", b); seen.push(["dop=" + b, await boxes(p)]); }
    for (let n = 1; n <= 10; n++) { await set(p, "#f1-nc", n); seen.push(["nc=" + n, await boxes(p)]); }
    for (const c of [30, 37, 45]) { await set(p, "#f1-cn0", c); seen.push(["cn0=" + c, await boxes(p)]); }
    for (const [what, got] of seen) assert.equal(got, ref, `width ${width}, ${what}`);
    await p.close();
  }
});

test("on a high-density screen every canvas in the page keeps its height through repeated redraws", async () => {
  for (const dpr of [2, 3]) {
    const { p } = await open(1000, dpr);
    const heights = () => p.evaluate(() => [...document.querySelectorAll("canvas")].map((c) => c.id + ":" + Math.round(c.getBoundingClientRect().height)).join(" "));
    const ref = await heights();
    for (let n = 1; n <= 10; n++) await set(p, "#f1-nc", n);
    for (let b = 0; b <= 20; b += 4) await set(p, "#f1-dop", b);
    await set(p, "#f2-cn0", 40); await set(p, "#f4-thr", 950); await p.setViewportSize({ width: 900, height: 900 }); await settle(p);
    assert.equal(await heights(), ref, `dpr ${dpr}`);
    await p.close();
  }
});

test("the y range stays 0 to 35 after every interaction", async () => {
  const { p } = await open(1000);
  for (const [id, v] of [["#f1-nc", 10], ["#f1-dop", 0], ["#f1-cn0", 30], ["#f1-prn", 12], ["#f1-cn0", 45]]) {
    await set(p, id, v);
    assert.equal((await state(p)).ymax, "35", `${id}=${v}`);
  }
  await p.close();
});

test("redraw controls answer within 30 ms and recompute controls within 300 ms", async () => {
  const { p } = await open(1000);
  const timed = (id, v) => p.evaluate(([id, v]) => new Promise((r) => {
    const t = performance.now(), el = document.querySelector(id); el.value = v; el.dispatchEvent(new Event("input"));
    requestAnimationFrame(() => requestAnimationFrame(() => r(performance.now() - t)));
  }), [id, String(v)]);
  for (const v of [3, 9, 0, 20]) assert.ok((await timed("#f1-dop", v)) < 30 + 34, "doppler");
  for (const v of [2, 10]) assert.ok((await timed("#f1-nc", v)) < 30 + 34, "nc");
  for (const v of [33, 44]) assert.ok((await timed("#f1-cn0", v)) < 300 + 34, "cn0");
  await p.close();
});

test("the true hypothesis at 45 dB-Hz with 10 periods peaks at code phase 400 and is detected", async () => {
  const { p } = await open(1000);
  await set(p, "#f1-prn", 7); await set(p, "#f1-cn0", 45); await set(p, "#f1-dop", BIN(1500)); await set(p, "#f1-nc", 10);
  const s = await state(p);
  assert.equal(s.peakPhase, "400");
  assert.equal(s.detected, "right");
  await p.close();
});

test("an absent PRN is never detected", async () => {
  const { p } = await open(1000);
  await set(p, "#f1-prn", 12);
  for (let n = 1; n <= 10; n++) { await set(p, "#f1-nc", n); assert.equal((await state(p)).detected, "no", `nc=${n}`); }
  await p.close();
});

test("a 900 Hz Doppler error at 42 dB-Hz is not detected", async () => {
  const { p } = await open(1000);
  await set(p, "#f1-dop", BIN(2500));
  for (const n of [1, 5, 10]) { await set(p, "#f1-nc", n); assert.equal((await state(p)).detected, "no", `nc=${n}`); }
  await p.close();
});

test("no errors and no horizontal overflow", async () => {
  for (const width of [1280, 820, 400]) {
    const { p, errors } = await open(width);
    await set(p, "#f1-dop", 0); await set(p, "#f1-nc", 10); await set(p, "#f1-prn", 3);
    assert.deepEqual(errors, []);
    assert.equal(await p.evaluate(() => document.documentElement.scrollWidth - innerWidth), 0, `width ${width}`);
    await p.close();
  }
});
