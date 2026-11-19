// The emd-head plate: a head drawn as an engraved atlas plate of the 10-20 system, with the three
// measured electrodes (C3, Cz, C4) as cups colored by the right-minus-left difference in log power at
// the frequency on the slider, and a seismograph strip of the four difference spectra under it. The
// numbers and sentences come from emd-head-model.js; this file only draws them. Drawn on a plain 2D
// canvas with the model's own projection (hidden lines removed exactly for the sphere), so the plate
// needs no WebGL and keeps the engraving's fine line weights.
const VQ = new URL(import.meta.url).search; // the page's version, passed on so a browser never mixes versions
const M = await import(new URL(`./emd-head-model.js${VQ}`, import.meta.url).href);

const KIND = "emd-head";
const reduced = matchMedia("(prefers-reduced-motion: reduce)");
const darkQuery = matchMedia("(prefers-color-scheme: dark)");

const link = document.createElement("link");
link.rel = "stylesheet";
link.href = new URL(`./${KIND}.css${VQ}`, import.meta.url).href;
document.head.append(link);

for (const fig of document.querySelectorAll(`figure.plate-live[data-demo="${KIND}"]`)) mount(fig).catch((e) => {
  const root = fig.querySelector(".demo-root");
  if (root) root.innerHTML = `<p class="eh-fail">The live plate could not load its data (${String(e.message || e).replace(/[<>&]/g, "")}).</p>`;
});

// ------------------------------------------------------------------------------------------ geometry
const RAD = Math.PI / 180;
const sph = (lat, lon) => [Math.cos(lat * RAD) * Math.sin(lon * RAD), Math.sin(lat * RAD), Math.cos(lat * RAD) * Math.cos(lon * RAD)];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const add = (a, b, k = 1) => [a[0] + k * b[0], a[1] + k * b[1], a[2] + k * b[2]];
const scl = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = (a) => scl(a, 1 / Math.hypot(a[0], a[1], a[2]));
const GHOST = 0.42; // below the ear line only the outline goes on, fading out by y = -0.42 (25 degrees)

/** The scalp, engraved: meridians and parallels every 18 degrees (10% of the 10-20 arcs) with finer
 *  ones between, down to the ear line (the plane of the nasion, the inion, and the preauricular points).
 *  The arcs through the vertex at azimuth 0, 90, 180, and 270 are the 10-20 reference arcs, and the
 *  parallel 18 degrees up is the 10-20 circumference. */
function lattice() {
  const lines = [];
  for (let lon = 0; lon < 360; lon += 9) {
    const major = lon % 18 === 0, ref = lon % 90 === 0;
    const pts = [];
    for (let lat = major ? 90 : 63; lat >= 0; lat -= 3) pts.push(sph(lat, lon));
    lines.push({ pts, kind: ref ? "ref" : major ? "major" : "minor" });
  }
  for (let lat = 81; lat > 0; lat -= 9) {
    if (lat === 18) continue;
    const pts = [];
    for (let lon = 0; lon <= 360; lon += 3) pts.push(sph(lat, lon));
    lines.push({ pts, kind: lat % 18 === 0 ? "major" : "minor" });
  }
  const ring = [], rim = [];
  for (let lon = 0; lon <= 360; lon += 3) { ring.push(sph(18, lon)); rim.push(sph(0, lon)); }
  lines.push({ pts: ring, kind: "ref" });
  lines.push({ pts: rim, kind: "rim" });
  const hatch = [];
  for (let lat = 85.5; lat > 0; lat -= 4.5) {
    if (Math.abs(lat % 9) < 1e-9) continue;
    const pts = [];
    for (let lon = 0; lon <= 360; lon += 3) pts.push(sph(lat, lon));
    hatch.push(pts);
  }
  return { lines, hatch };
}
const LATTICE = lattice();

/** Small engraved outlines of the nose and ears, in the head frame (nose toward -z). */
function catmull(pts, n = 6) {
  const out = [];
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[Math.max(0, i - 1)], p1 = pts[i], p2 = pts[i + 1], p3 = pts[Math.min(pts.length - 1, i + 2)];
    for (let k = 0; k < n; k++) {
      const t = k / n, t2 = t * t, t3 = t2 * t;
      out.push([0, 1].map((j) => 0.5 * (2 * p1[j] + (-p0[j] + p2[j]) * t + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t2 + (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t3)));
    }
  }
  out.push(pts[pts.length - 1]);
  return out;
}
function features() {
  const strokes = [], ears = [];
  const seg = (a, b, n = 10) => { const s = []; for (let k = 0; k <= n; k++) s.push(add(a, add(b, a, -1), k / n)); return s; };
  // the nose: the ridge from the root to the tip, the two wings, and the nostrils' base
  const root = sph(6, 180), tip = [0, -0.13, -1.22], col = [0, -0.21, -1.11];
  strokes.push(catmull([[0, 0], [0.45, 0.42], [1, 1]], 6).map(([a, b]) => add(add(root, add(tip, root, -1), a * 0.5 + b * 0.5), [0, 0.012 * Math.sin(Math.PI * a), -0.02 * Math.sin(Math.PI * a)])));
  strokes.push(catmull([[0, 0], [0.5, 0.65], [1, 1]], 4).map(([a, b]) => add(tip, add(col, tip, -1), b)));
  for (const s of [-1, 1]) {
    const wing = catmull([[0, 0], [0.35, 0.2], [0.75, 0.55], [1, 0.95]], 5).map(([u, v]) => [s * (0.035 + 0.1 * u), -0.15 - 0.065 * v, -1.18 + 0.15 * u]);
    strokes.push(wing);
    strokes.push(seg(col, [s * 0.075, -0.205, -1.05], 4));
  }
  // the ears: helix, lobe, inner fold, and tragus, in a plane turned back and out from the side
  const helix = catmull([[0, 0.15], [0.06, 0.21], [0.14, 0.205], [0.195, 0.13], [0.2, 0.03], [0.18, -0.06], [0.14, -0.13], [0.1, -0.19], [0.065, -0.235], [0.025, -0.225], [0.005, -0.16]]);
  const fold = catmull([[0.04, 0.12], [0.1, 0.15], [0.145, 0.09], [0.14, -0.01], [0.105, -0.08], [0.06, -0.11]]);
  const tragus = catmull([[0.0, 0.01], [0.035, 0.0], [0.04, -0.06], [0.012, -0.08]]);
  for (const s of [-1, 1]) {
    const beta = 26 * RAD, gamma = 16 * RAD;
    const h = [s * Math.sin(beta), 0, Math.cos(beta)], v = [s * Math.sin(gamma), Math.cos(gamma), 0], base = [s * 0.992, -0.11, 0.04];
    const to3 = (curve) => curve.map(([u, w]) => add(add(base, h, u), v, w));
    ears.push({ shape: to3(helix), inner: [to3(fold), to3(tragus)], center: to3([[0.1, 0]])[0] });
  }
  return { nose: strokes, ears };
}
const FEATURES = features();

// ------------------------------------------------------------------------------------------ the plate
async function mount(fig) {
  const root = fig.querySelector(".demo-root");
  const data = await (await fetch(fig.dataset.src)).json();
  const freqs = data.freqs, uid = Math.random().toString(36).slice(2, 8);
  const i12 = Math.max(0, freqs.indexOf(12));
  const st = {
    variant: data.variants[0].id, i: i12,
    view: reduced.matches ? { pol: 0.62, az: 0, zoom: 1 } : { ...M.VIEW.top, zoom: 1 },
    target: null, auto: !reduced.matches, clock: 0, sweepClock: 0, sweepStart: i12,
    visible: false, touched: false,
  };
  st.target = { ...st.view };

  root.innerHTML = `
<div class="eh">
  <div class="eh-sets" role="radiogroup" aria-label="Data set">
    ${data.variants.map((v) => `<button type="button" role="radio" data-v="${v.id}" aria-checked="${v.id === st.variant}" tabindex="${v.id === st.variant ? 0 : -1}"><span class="eh-set-name">${v.label.replace(/ window$/, '<span class="eh-set-w"> window</span>')}</span><span class="eh-set-n">${v.n_subjects} subjects<span class="eh-set-w">, ${v.window.replace(" after the cue", "").replace(" to ", "–")}</span></span></button>`).join("")}
  </div>
  <div class="eh-freq">
    <label for="eh-f-${uid}">Frequency</label>
    <div class="eh-range">
      <input id="eh-f-${uid}" type="range" min="0" max="${freqs.length - 1}" step="1" value="${st.i}">
      <div class="eh-ticks" aria-hidden="true"></div>
    </div>
    <output for="eh-f-${uid}" class="eh-fout"></output>
  </div>
  <div class="eh-body">
    <div class="eh-plate">
      <canvas class="eh-head" tabindex="0" role="img"></canvas>
      <div class="eh-tools">
        <button type="button" class="eh-play" aria-pressed="false"></button>
        <span class="eh-zoom"><button type="button" data-act="out" aria-label="Zoom out">−</button><button type="button" data-act="in" aria-label="Zoom in">+</button></span>
        <button type="button" data-act="reset">Reset view</button>
      </div>
      <p class="eh-hint">Drag to turn the head; pinch, Ctrl-scroll, or the buttons to zoom. From the keyboard, with the head focused: the arrow keys turn it, plus and minus zoom, 0 resets, and Space plays or pauses the slow turn and sweep.</p>
    </div>
    <div class="eh-side">
      <table class="eh-read">
        <caption>Right minus left, at <span class="eh-cap-f"></span></caption>
        <thead><tr><th scope="col"><span class="eh-sr">Channel</span></th><th scope="col" class="eh-unit">dB</th><th scope="col">95% interval</th></tr></thead>
        <tbody>${M.ROWS.map((k) => `<tr data-k="${k}"><th scope="row"><i class="eh-sw"></i><b>${M.ROW_LABEL[k]}</b><small>${k === "average" ? "C3, Cz, and C4 as one signal" : M.SIDE[k]}</small></th><td class="eh-m"></td><td class="eh-ci"></td></tr>`).join("")}</tbody>
      </table>
      <div class="eh-scale"></div>
      <p class="eh-say"></p>
    </div>
  </div>
  <div class="eh-stripwrap">
    <p class="eh-strip-title">The four difference spectra, 1 to 45 Hz, each with its 95% band</p>
    <canvas class="eh-strip" role="img"></canvas>
  </div>
  <ul class="eh-notes">
    <li><b>Average.</b> The three channels averaged into one signal before its spectrum is taken (the file’s “average” channel). The original pipeline averaged them the same way but also z-scored each window, which this spectrum does not. It isn’t the mean of the three rows above it.</li>
    <li><b>Windows.</b> One two-second window per imagined movement: ${data.variants.map((v) => `${v.label}, ${v.window}`).join("; ")}. The later window is a check the study ran.</li>
    <li><b>Intervals.</b> <span class="eh-int"></span></li>
    <li><b>Head.</b> An idealized sphere with C3, Cz, and C4 at their 10-20 places: Cz at the vertex, C3 and C4 20% of the ear-to-ear arc to each side. The faint dashed arcs are the 10-20 system’s reference lines. Nothing is drawn between the electrodes, because nothing was measured there.</li>
    <li><b>Colors.</b> <span class="eh-scale-note"></span> A cup or a cursor mark with a dashed rim has a 95% interval that includes zero; the strip’s trace is heavy where neighboring intervals leave out zero on the same side.</li>
  </ul>
  <p class="eh-checked" hidden></p>
</div>`;

  const $ = (s) => root.querySelector(s);
  const head = $(".eh-head"), strip = $(".eh-strip"), slider = $(`#eh-f-${uid}`), out = $(".eh-fout");
  const play = $(".eh-play"), say = $(".eh-say"), ticks = $(".eh-ticks");
  const hctx = head.getContext("2d"), sctx = strip.getContext("2d");

  // the slider's tick marks: one per measured frequency, the idle sweep's range shaded
  const n1 = freqs.length - 1;
  ticks.innerHTML = freqs.map((f, i) => `<i style="left:${(100 * i) / n1}%" class="${f % 5 === 0 || i === 0 ? "eh-tk-l" : ""}">${f % 10 === 0 || i === 0 || i === n1 ? `<span>${f}</span>` : ""}</i>`).join("")
    + `<em style="left:${(100 * freqs.indexOf(data.sweep[0])) / n1}%;width:${(100 * (freqs.indexOf(data.sweep[1]) - freqs.indexOf(data.sweep[0]))) / n1}%"></em>`;

  // ---------------------------------------------------------------------------------------- colors
  let C = {};
  function readColors() {
    const cs = getComputedStyle(fig);
    const v = (k) => cs.getPropertyValue(`--${k}`).trim();
    C = { ink: v("ink"), ink2: v("ink-2"), paper: v("paper"), paper2: v("paper-2"), mount: v("mount"), rule: v("rule"), rule2: v("rule-2"), accent: v("accent-ink"),
          neg: v("eh-neg"), pos: v("eh-pos"), zero: v("eh-zero"), type: v("type") || "monospace", serif: v("serif") || "serif" };
    C.dark = M.relLum(toHex(C.paper)) < 0.3;
    C.plate = C.dark ? C.paper : C.mount;
    C.inks = { neg: toHex(C.neg), pos: toHex(C.pos), zero: toHex(C.zero) };
  }
  function toHex(c) {
    if (/^#/.test(c)) return c.length === 4 ? "#" + [...c.slice(1)].map((x) => x + x).join("") : c.slice(0, 7);
    const m = c.match(/[\d.]+/g);
    return m ? "#" + m.slice(0, 3).map((x) => Math.round(+x).toString(16).padStart(2, "0")).join("") : "#888888";
  }
  const mix = (a, b, t) => { const p = toHex(a), q = toHex(b); return "#" + [1, 3, 5].map((i) => Math.round(parseInt(p.slice(i, i + 2), 16) * (1 - t) + parseInt(q.slice(i, i + 2), 16) * t).toString(16).padStart(2, "0")).join(""); };
  const alpha = (c, a) => { const h = toHex(c); return `rgba(${parseInt(h.slice(1, 3), 16)},${parseInt(h.slice(3, 5), 16)},${parseInt(h.slice(5, 7), 16)},${a})`; };

  // ---------------------------------------------------------------------------------------- state
  const variant = () => M.variantOf(data, st.variant);
  const state = () => M.stateAt(data, st.variant, st.i);
  let lastWords = "";
  function updateWords() {
    const s = state(), v = variant(), lim = M.scaleLimit(v);
    out.textContent = `${M.fmtFreq(s.freq)} Hz`;
    slider.value = st.i;
    slider.setAttribute("aria-valuetext", `${M.fmtFreq(s.freq)} Hz`);
    $(".eh-cap-f").textContent = `${M.fmtFreq(s.freq)} Hz`;
    for (const r of M.readouts(s)) {
      const tr = root.querySelector(`tr[data-k="${r.key}"]`);
      tr.querySelector(".eh-m").textContent = r.mean;
      tr.querySelector(".eh-ci").textContent = r.ci;
      tr.querySelector(".eh-sw").style.background = M.inkAt(s.rows[r.key].mean, lim, C.inks);
      tr.dataset.dir = r.dir;
    }
    const words = M.say(s);
    if (words !== lastWords) { say.textContent = words; lastWords = words; }
    $(".eh-int").textContent = M.intervalsNote(v, freqs.length);
    $(".eh-scale-note").textContent = `A symmetric scale from ${M.MINUS}${M.fmt(lim, 1)} to ${M.fmt(lim, 1)} dB, the largest difference in this data set rounded up. Blue is less power when people imagine the right hand than the left; orange is more. The cups, the swatches, and the strip’s fills share it.`;
    head.setAttribute("aria-label", `A head drawn on the 10-20 system, with three electrodes colored by the difference. ${["C3", "Cz", "C4"].map((k) => `${k}: ${M.fmt(s.rows[k].mean)} dB`).join("; ")}. Average: ${M.fmt(s.rows.average.mean)} dB. ${words}`);
    strip.setAttribute("aria-label", `Difference spectra of C3, Cz, C4, and their average, 1 to 45 Hz, with the cursor at ${M.fmtFreq(s.freq)} Hz.`);
    drawScale();
  }
  let scaleKey = "";
  function drawScale() {
    const v = variant(), lim = M.scaleLimit(v), N = 24, key = `${v.id}|${C.inks.neg}|${C.inks.pos}|${C.inks.zero}`;
    if (key === scaleKey) return;
    scaleKey = key;
    const cells = Array.from({ length: N }, (_, k) => { const x = -lim + ((k + 0.5) / N) * 2 * lim; return `<rect x="${(k * 100) / N}" y="0" width="${100 / N + 0.2}" height="10" fill="${M.inkAt(x, lim, C.inks)}"/>`; }).join("");
    $(".eh-scale").innerHTML = `<svg viewBox="0 0 100 10" preserveAspectRatio="none" class="eh-bar" aria-hidden="true">${cells}<line class="eh-bar-zero" x1="50" x2="50" y1="0" y2="10"/></svg>
      <div class="eh-bar-ticks" aria-hidden="true"><span>${M.MINUS}${M.fmt(lim, 1)}</span><span>0</span><span>${M.fmt(lim, 1)} dB</span></div>
      <div class="eh-bar-words"><span>less power with the right hand</span><span>more power with the right hand</span></div>`;
  }

  // ---------------------------------------------------------------------------------------- sizing
  let HW = 0, HH = 0, SW = 0, SH = 0, dpr = 1, DPR = 0;
  function size() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    const hw = head.parentElement.clientWidth, sw = strip.parentElement.clientWidth;
    const hh = Math.round(Math.min(420, Math.max(340, hw * 0.98)));
    const sh = Math.round(sw < 520 ? 252 : 330);
    if (dpr !== DPR) { DPR = dpr; HW = SW = 0; }
    if (hw !== HW || hh !== HH) { HW = hw; HH = hh; head.style.height = `${hh}px`; head.width = Math.round(hw * dpr); head.height = Math.round(hh * dpr); }
    if (sw !== SW || sh !== SH) { SW = sw; SH = sh; strip.style.height = `${sh}px`; strip.width = Math.round(sw * dpr); strip.height = Math.round(sh * dpr); }
  }

  // ---------------------------------------------------------------------------------------- the head
  function drawHead() {
    if (!HW) return;
    const w = HW, h = HH, ctx = hctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const base = Math.max(60, Math.min(w / 2 - 58, h / 2 - 46, 176));
    const cam = M.camera(st.view, w, h - 18, base);
    const R = M.outlineRadius(cam), P = (q) => M.project(cam, q);
    const s = state(), v = variant(), lim = M.scaleLimit(v);
    ctx.lineCap = "round"; ctx.lineJoin = "round";

    // the plate under the scalp: the part of the sphere above the ear line that faces the camera
    const ol = M.outlineCircle(cam, 180);
    const above = ol.map((q) => q[1] >= 0);
    ctx.beginPath();
    if (above.every(Boolean)) { ctx.arc(cam.cx, cam.cy, R, 0, 2 * Math.PI); }
    else if (above.some(Boolean)) {
      const k0 = above.findIndex((a, k) => a && !above[(k - 1 + above.length) % above.length]);
      const arc = [];
      for (let k = 0; k < ol.length; k++) { const j = (k0 + k) % ol.length; if (!above[j]) break; arc.push(ol[j]); }
      const rim = [];
      for (let a = 0; a < 360; a += 2) { const q = sph(0, a); rim.push({ q, f: M.facing(cam, q) }); }
      const r0 = rim.findIndex((p, k) => p.f && !rim[(k - 1 + rim.length) % rim.length].f);
      const rarc = [];
      if (r0 >= 0) for (let k = 0; k < rim.length; k++) { const j = (r0 + k) % rim.length; if (!rim[j].f) break; rarc.push(rim[j].q); }
      const last = P(arc[arc.length - 1]);
      if (rarc.length && Math.hypot(P(rarc[0]).x - last.x, P(rarc[0]).y - last.y) > Math.hypot(P(rarc[rarc.length - 1]).x - last.x, P(rarc[rarc.length - 1]).y - last.y)) rarc.reverse();
      [...arc, ...rarc].forEach((q, k) => { const p = P(q); k ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y); });
      ctx.closePath();
    }
    ctx.fillStyle = alpha(C.plate, C.dark ? 0.84 : 0.92);
    ctx.fill();

    // engraved shading: fine hatching along the parallels on the side away from a light at the upper left
    const light = unit(add(add(scl(cam.r, -0.55), cam.u, 0.62), cam.f, -0.55));
    strokeRuns(ctx, LATTICE.hatch.map((pts) => ({ pts })), cam, (mid) => Math.min(1, Math.max(0, (0.38 - Math.max(0, dot(mid, light))) / 0.3)), C.ink, C.dark ? 0.22 : 0.28, 0.55);
    // the lattice, then the 10-20 reference arcs, dashed, and the ear line
    const one = () => 1;
    strokeRuns(ctx, LATTICE.lines.filter((l) => l.kind === "minor"), cam, one, C.ink, C.dark ? 0.17 : 0.2, 0.55);
    strokeRuns(ctx, LATTICE.lines.filter((l) => l.kind === "major"), cam, one, C.ink, C.dark ? 0.32 : 0.38, 0.65);
    ctx.setLineDash([3.2, 2.6]);
    strokeRuns(ctx, LATTICE.lines.filter((l) => l.kind === "ref"), cam, one, C.ink2, C.dark ? 0.7 : 0.72, 0.95);
    ctx.setLineDash([]);
    strokeRuns(ctx, LATTICE.lines.filter((l) => l.kind === "rim"), cam, one, C.ink, 0.8, 1.15);
    // the outline: engraved down to the ear line, then a fading pencil line for the rest of the head
    const outlinePaths = new Map();
    for (let k = 0; k < ol.length - 1; k++) {
      const y = (ol[k][1] + ol[k + 1][1]) / 2, a = y >= 0 ? 0.88 : Math.round(10 * 0.42 * Math.max(0, 1 + y / GHOST)) / 10;
      if (a < 0.05) continue;
      if (!outlinePaths.has(a)) outlinePaths.set(a, new Path2D());
      const p = P(ol[k]), q = P(ol[k + 1]), path = outlinePaths.get(a);
      path.moveTo(p.x, p.y); path.lineTo(q.x, q.y);
    }
    for (const [a, path] of outlinePaths) { ctx.strokeStyle = alpha(C.ink, a); ctx.lineWidth = a === 0.88 ? 1.4 : 0.85; ctx.stroke(path); }
    // the nose, then the ears: filled with the plate's paper so they read as solid, clipped to outside
    // the head's outline when they sit behind it
    const strokeVisible = (pts) => {
      let pen = false;
      for (let k = 0; k < pts.length - 1; k++) {
        const mid = scl(add(pts[k], pts[k + 1]), 0.5);
        if (M.hidden(cam, mid)) { pen = false; continue; }
        const p = P(pts[k]), q = P(pts[k + 1]);
        if (!pen) ctx.moveTo(p.x, p.y);
        ctx.lineTo(q.x, q.y); pen = true;
      }
    };
    ctx.strokeStyle = alpha(C.ink, 0.82); ctx.lineWidth = 0.95;
    ctx.beginPath(); for (const pts of FEATURES.nose) strokeVisible(pts); ctx.stroke();
    for (const ear of FEATURES.ears) {
      ctx.save();
      if (M.hidden(cam, ear.center)) {
        ctx.beginPath(); ctx.rect(0, 0, w, h); ctx.arc(cam.cx, cam.cy, R + 0.6, 0, 2 * Math.PI, true); ctx.clip("evenodd");
      }
      const shape = ear.shape.map(P);
      ctx.beginPath(); shape.forEach((p, k) => (k ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y))); ctx.closePath();
      ctx.fillStyle = C.plate; ctx.fill();
      ctx.strokeStyle = alpha(C.ink, 0.85); ctx.lineWidth = 1; ctx.beginPath(); shape.forEach((p, k) => (k ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y))); ctx.stroke();
      ctx.strokeStyle = alpha(C.ink, 0.55); ctx.lineWidth = 0.75;
      ctx.beginPath(); for (const c of ear.inner) c.map(P).forEach((p, k) => (k ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y))); ctx.stroke();
      ctx.restore();
    }

    // the three cups, farthest first, then their leaders and labels
    const cups = M.ELECTRODES.map((e) => cupGeometry(e, cam)).sort((a, b) => b.depth - a.depth);
    for (const cup of cups) drawCup(ctx, cup, M.inkAt(s.rows[cup.e].mean, lim, C.inks), cam, M.direction(s.rows[cup.e]) === "unclear");
    drawLabels(ctx, cups, cam, R, w, h, s);
    drawMedallion(ctx, w, h, s, lim);
    // the key to the dashed arcs, in the plate's corner
    ctx.strokeStyle = alpha(C.ink2, 0.8); ctx.lineWidth = 0.95; ctx.setLineDash([3.2, 2.6]);
    ctx.beginPath(); ctx.moveTo(4, 12); ctx.lineTo(26, 12); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = C.ink2; ctx.font = `italic 12px ${C.serif}`; ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.fillText("10-20 arcs", 31, 12);
  }

  /** Strokes polylines on the sphere, keeping only segments that face the camera, in runs so dashes flow. */
  function strokeRuns(ctx, lines, cam, alphaOf, color, a0, width) {
    const LEVELS = 6, paths = Array.from({ length: LEVELS }, () => new Path2D()), used = new Array(LEVELS).fill(false);
    for (const { pts } of lines) {
      let prev = -1;
      let pa = M.project(cam, pts[0]);
      for (let k = 0; k < pts.length - 1; k++) {
        const b = pts[k + 1], pb = M.project(cam, b);
        const mid = unit(add(pts[k], b));
        let lvl = -1;
        if (M.facing(cam, mid)) { const a = alphaOf(mid); if (a > 0.04) lvl = Math.min(LEVELS - 1, Math.round(a * (LEVELS - 1))); }
        if (lvl >= 0) {
          if (lvl !== prev) paths[lvl].moveTo(pa.x, pa.y);
          paths[lvl].lineTo(pb.x, pb.y); used[lvl] = true;
        }
        prev = lvl; pa = pb;
      }
    }
    ctx.lineWidth = width;
    paths.forEach((p, l) => { if (!used[l] || l === 0) return; ctx.strokeStyle = alpha(color, (a0 * l) / (LEVELS - 1)); ctx.stroke(p); });
  }

  function cupGeometry(e, cam) {
    const p = M.electrode(e), n = p;
    const t1 = unit(cross(n, Math.abs(n[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0])), t2 = cross(n, t1);
    const rho = 0.1, hgt = 0.05, N = 36;
    const ring = (center, r) => Array.from({ length: N }, (_, k) => { const a = (2 * Math.PI * k) / N; return add(center, add(scl(t1, Math.cos(a) * r), scl(t2, Math.sin(a) * r))); });
    const top = scl(p, 1 + hgt);
    const visible = dot(add(cam.C, top, -1), n) > 0 && !M.hidden(cam, top);
    return { e, p, n, top, visible, depth: M.project(cam, top).z, base: ring(scl(p, 1.004), rho), rim: ring(top, rho), lip: ring(top, rho * 0.62) };
  }
  function drawCup(ctx, cup, color, cam, unclear) {
    if (!cup.visible) return;
    const P = (q) => M.project(cam, q);
    const pts = [...cup.base, ...cup.rim].map(P);
    const hull = convexHull(pts);
    ctx.beginPath(); hull.forEach((p, k) => (k ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y))); ctx.closePath();
    ctx.fillStyle = mix(color, C.dark ? C.paper : C.ink, C.dark ? 0.4 : 0.3); ctx.fill();
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1; ctx.stroke();
    const rim = cup.rim.map(P);
    ctx.beginPath(); rim.forEach((p, k) => (k ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y))); ctx.closePath();
    ctx.fillStyle = color; ctx.fill(); ctx.strokeStyle = C.ink; ctx.lineWidth = 1.1;
    // a dashed rim: the 95% interval includes zero
    if (unclear) ctx.setLineDash([2.4, 2]);
    ctx.stroke(); ctx.setLineDash([]);
    const lip = cup.lip.map(P);
    ctx.beginPath(); lip.forEach((p, k) => (k ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y))); ctx.closePath();
    ctx.strokeStyle = alpha(C.ink, 0.45); ctx.lineWidth = 0.8; ctx.stroke();
    const c = P(cup.top);
    ctx.fillStyle = alpha(C.ink, 0.7); ctx.beginPath(); ctx.arc(c.x, c.y, 1.3, 0, 2 * Math.PI); ctx.fill();
  }

  /** Leader lines from each cup to its label, outside the head where there is room. */
  function drawLabels(ctx, cups, cam, R, w, h, s) {
    const DEF = { C3: [-1, -0.62], C4: [1, -0.62], Cz: [0.5, -0.9] };
    const boxes = [{ x: 0, y: 0, w: 104, h: 22, key: true }]; // the key to the dashed arcs
    const fName = `13px ${C.type}`, fVal = `11.5px ${C.type}`;
    for (const cup of [...cups].sort((a, b) => M.ELECTRODES.indexOf(a.e) - M.ELECTRODES.indexOf(b.e))) {
      const q = M.project(cam, cup.top);
      let dx = q.x - cam.cx + DEF[cup.e][0] * R * 0.45, dy = q.y - cam.cy + DEF[cup.e][1] * R * 0.45;
      const l = Math.hypot(dx, dy) || 1; dx /= l; dy /= l;
      const ax = cam.cx + dx * (R + 16), ay = cam.cy + dy * (R + 16);
      ctx.font = fName; const w1 = ctx.measureText(cup.e).width;
      ctx.font = fVal; const val = `${M.fmt(s.rows[cup.e].mean)} dB`, w2 = ctx.measureText(val).width;
      const bw = Math.max(w1, w2) + 10, bh = 33;
      let bx = dx < -0.35 ? ax - bw : dx > 0.35 ? ax : ax - bw / 2;
      let by = dx < -0.35 || dx > 0.35 ? ay - bh / 2 : dy < 0 ? ay - bh : ay;
      bx = Math.max(3, Math.min(w - bw - 3, bx)); by = Math.max(3, Math.min(h - bh - 3, by));
      for (const o of boxes) if (bx < o.x + o.w && bx + bw > o.x && by < o.y + o.h && by + bh > o.y) by = by < o.y ? o.y - bh - 4 : o.y + o.h + 4;
      boxes.push({ x: bx, y: by, w: bw, h: bh, cup, q, val });
    }
    for (const b of boxes) {
      if (b.key) continue;
      // the leader: from the cup's center to the nearest side of the label, with a short shoulder
      const ex = b.q.x < b.x ? b.x : b.q.x > b.x + b.w ? b.x + b.w : b.x + b.w / 2;
      const ey = b.q.x < b.x || b.q.x > b.x + b.w ? b.y + b.h / 2 : b.q.y < b.y ? b.y : b.y + b.h;
      ctx.strokeStyle = alpha(C.ink, b.cup.visible ? 0.8 : 0.4); ctx.lineWidth = 0.8;
      if (!b.cup.visible) ctx.setLineDash([2, 2.5]);
      ctx.beginPath(); ctx.moveTo(b.q.x, b.q.y); ctx.lineTo(ex, ey); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = alpha(C.ink, 0.85); ctx.beginPath(); ctx.arc(ex, ey, 1.6, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = alpha(C.plate, 0.88); ctx.fillRect(b.x, b.y, b.w, b.h);
      ctx.textBaseline = "alphabetic";
      const right = ex === b.x + b.w && b.q.x > b.x + b.w;
      ctx.textAlign = right ? "right" : ex === b.x ? "left" : "center";
      const tx = ctx.textAlign === "right" ? b.x + b.w - 5 : ctx.textAlign === "left" ? b.x + 5 : b.x + b.w / 2;
      ctx.fillStyle = C.ink; ctx.font = `13px ${C.type}`; ctx.fillText(b.cup.e, tx, b.y + 14);
      ctx.fillStyle = C.ink2; ctx.font = `11.5px ${C.type}`; ctx.fillText(b.val, tx, b.y + 28);
    }
  }

  /** The channel average, which has no place on the head: a medallion in the corner of the plate. */
  function drawMedallion(ctx, w, h, s, lim) {
    const r = 12, x = 6 + r + 2, y = h - 10 - r;
    ctx.fillStyle = M.inkAt(s.rows.average.mean, lim, C.inks);
    ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill();
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.1; ctx.stroke();
    ctx.strokeStyle = alpha(C.ink, 0.45); ctx.lineWidth = 0.8; ctx.beginPath(); ctx.arc(x, y, r * 0.62, 0, 2 * Math.PI); ctx.stroke();
    ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
    ctx.fillStyle = C.ink; ctx.font = `12px ${C.type}`; ctx.fillText(`Average ${M.fmt(s.rows.average.mean)} dB`, x + r + 8, y - 1);
    ctx.fillStyle = C.ink2; ctx.font = `italic 12px ${C.serif}`; ctx.fillText("C3, Cz, and C4 as one signal", x + r + 8, y + 13);
  }

  // ---------------------------------------------------------------------------------------- the strip
  function drawStrip() {
    if (!SW) return;
    const w = SW, h = SH, ctx = sctx, v = variant(), s = state();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const L = 62, Rm = 12, T = 26, B = 30, rowH = (h - T - B) / 4;
    const lim = M.stripLimit(v), clim = M.scaleLimit(v), f0 = freqs[0], f1 = freqs[freqs.length - 1];
    const fx = (f) => L + ((f - f0) / (f1 - f0)) * (w - L - Rm);
    // the idle sweep's range, 8 to 30 Hz (mu and beta)
    ctx.fillStyle = alpha(C.ink, C.dark ? 0.06 : 0.045);
    ctx.fillRect(fx(data.sweep[0]), T - 8, fx(data.sweep[1]) - fx(data.sweep[0]), h - B - T + 8);
    ctx.fillStyle = C.ink2; ctx.font = `italic 11.5px ${C.serif}`; ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
    ctx.textAlign = "right";
    ctx.fillText(w < 420 ? "shaded: 8–30 Hz" : "shaded: mu and beta, 8–30 Hz", w - Rm, T - 12);
    M.ROWS.forEach((k, j) => {
      const ch = v.channels[k], y0 = T + rowH * (j + 0.5), half = rowH / 2 - 3;
      const fy = (d) => y0 - (d / lim) * half;
      // the 95% band, hatched like an engraving
      const band = new Path2D();
      freqs.forEach((f, i) => (i ? band.lineTo(fx(f), fy(ch.hi[i])) : band.moveTo(fx(f), fy(ch.hi[i]))));
      for (let i = freqs.length - 1; i >= 0; i--) band.lineTo(fx(freqs[i]), fy(ch.lo[i]));
      band.closePath();
      ctx.save(); ctx.clip(band);
      ctx.strokeStyle = alpha(C.ink, C.dark ? 0.32 : 0.3); ctx.lineWidth = 0.6; ctx.beginPath();
      for (let x = L - rowH; x < w; x += 3.5) { ctx.moveTo(x, y0 + rowH / 2); ctx.lineTo(x + rowH, y0 - rowH / 2); }
      ctx.stroke(); ctx.restore();
      // the fill from zero to the mean, in the scale's inks
      for (let i = 0; i < freqs.length - 1; i++) {
        const sub = 4;
        for (let q = 0; q < sub; q++) {
          const ta = q / sub, tb = (q + 1) / sub;
          const fa = freqs[i] + (freqs[i + 1] - freqs[i]) * ta, fb = freqs[i] + (freqs[i + 1] - freqs[i]) * tb;
          const ma = ch.mean[i] + (ch.mean[i + 1] - ch.mean[i]) * ta, mb = ch.mean[i] + (ch.mean[i + 1] - ch.mean[i]) * tb;
          const parts = ma * mb < 0 ? [[fa, ma, fa + (fb - fa) * (ma / (ma - mb)), 0], [fa + (fb - fa) * (ma / (ma - mb)), 0, fb, mb]] : [[fa, ma, fb, mb]];
          for (const [xa, ya, xb, yb] of parts) {
            ctx.fillStyle = M.inkAt((ya + yb) / 2, clim, C.inks);
            ctx.beginPath(); ctx.moveTo(fx(xa), fy(0)); ctx.lineTo(fx(xa), fy(ya)); ctx.lineTo(fx(xb), fy(yb)); ctx.lineTo(fx(xb), fy(0)); ctx.closePath(); ctx.fill();
          }
        }
      }
      // the band's edges, the zero line, and the mean
      ctx.strokeStyle = alpha(C.ink2, 0.55); ctx.lineWidth = 0.6;
      for (const key of ["lo", "hi"]) { ctx.beginPath(); freqs.forEach((f, i) => (i ? ctx.lineTo(fx(f), fy(ch[key][i])) : ctx.moveTo(fx(f), fy(ch[key][i])))); ctx.stroke(); }
      ctx.strokeStyle = alpha(C.ink, 0.5); ctx.lineWidth = 0.7; ctx.beginPath(); ctx.moveTo(L, y0); ctx.lineTo(w - Rm, y0); ctx.stroke();
      // the mean: a fine line, drawn heavy between neighboring frequencies whose intervals both leave out zero on the same side
      const dirs = freqs.map((_, i) => M.direction({ lo: ch.lo[i], hi: ch.hi[i] }));
      ctx.strokeStyle = C.ink;
      for (const heavy of [false, true]) {
        ctx.lineWidth = heavy ? 2.4 : 1; ctx.beginPath();
        for (let i = 0; i < freqs.length - 1; i++) {
          const h2 = dirs[i] !== "unclear" && dirs[i] === dirs[i + 1];
          if (h2 !== heavy) continue;
          ctx.moveTo(fx(freqs[i]), fy(ch.mean[i])); ctx.lineTo(fx(freqs[i + 1]), fy(ch.mean[i + 1]));
        }
        ctx.stroke();
      }
      // the row's name
      ctx.fillStyle = C.ink; ctx.font = `12px ${C.type}`; ctx.textAlign = "left"; ctx.textBaseline = "middle";
      ctx.fillText(M.ROW_LABEL[k], 2, y0 - (k === "average" ? 6 : 0));
      if (k === "average") { ctx.fillStyle = C.ink2; ctx.font = `italic 10.5px ${C.serif}`; ctx.fillText("one signal", 2, y0 + 8); }
    });
    // the frequency axis: a tick at every measured frequency, labels every 5 Hz
    const ya = h - B + 4;
    ctx.strokeStyle = alpha(C.ink, 0.7); ctx.lineWidth = 0.8;
    ctx.beginPath(); ctx.moveTo(L, ya); ctx.lineTo(w - Rm, ya); ctx.stroke();
    ctx.beginPath();
    for (const f of freqs) { const x = fx(f), long = f % 5 === 0; ctx.moveTo(x, ya); ctx.lineTo(x, ya + (long ? 5 : 2.5)); }
    ctx.stroke();
    ctx.fillStyle = C.ink2; ctx.font = `11px ${C.type}`; ctx.textAlign = "center"; ctx.textBaseline = "top";
    const every = w < 420 ? 10 : 5;
    for (const f of freqs) if (f % every === 0 || f === f0 || f === f1) ctx.fillText(String(f), fx(f), ya + 7);
    // the calibration mark, as on a seismograph: this much difference in every row
    const cal = lim >= 2 ? 1 : lim >= 0.5 ? 0.5 : 0.25;
    const calPx = (cal / lim) * (rowH / 2 - 3), cy0 = ya + 4, cx0 = 8;
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.2; ctx.beginPath();
    ctx.moveTo(cx0, cy0); ctx.lineTo(cx0, cy0 + calPx); ctx.moveTo(cx0 - 3, cy0); ctx.lineTo(cx0 + 3, cy0); ctx.moveTo(cx0 - 3, cy0 + calPx); ctx.lineTo(cx0 + 3, cy0 + calPx); ctx.stroke();
    ctx.fillStyle = C.ink2; ctx.font = `10.5px ${C.type}`; ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.fillText(`${cal} dB`, cx0 + 7, cy0 + calPx / 2);
    // the cursor at the slider's frequency, with each row's value as a small cup
    const x = fx(s.freq);
    ctx.strokeStyle = C.accent; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(x, T - 4); ctx.lineTo(x, h - B + 4); ctx.stroke();
    ctx.font = `12px ${C.type}`; const tag = `${M.fmtFreq(s.freq)} Hz`, tw = ctx.measureText(tag).width + 10;
    const tx = Math.max(2, Math.min(w - tw - 2, x - tw / 2));
    ctx.fillStyle = C.accent; ctx.fillRect(tx, 2, tw, 16);
    ctx.beginPath(); ctx.moveTo(x - 4, 18); ctx.lineTo(x + 4, 18); ctx.lineTo(x, 23); ctx.closePath(); ctx.fill();
    ctx.fillStyle = C.dark ? C.paper : C.mount; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(tag, tx + tw / 2, 10.5);
    M.ROWS.forEach((k, j) => {
      const y0 = T + rowH * (j + 0.5), y = y0 - (s.rows[k].mean / lim) * (rowH / 2 - 3);
      ctx.fillStyle = M.inkAt(s.rows[k].mean, clim, C.inks); ctx.beginPath(); ctx.arc(x, y, 4.8, 0, 2 * Math.PI); ctx.fill();
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1.1;
      if (M.direction(s.rows[k]) === "unclear") ctx.setLineDash([2, 1.8]);
      ctx.stroke(); ctx.setLineDash([]);
    });
  }

  // ---------------------------------------------------------------------------------------- motion
  let loop = 0, last = 0;
  const wake = () => { if (!loop && st.visible && !document.hidden) loop = requestAnimationFrame(frame); };
  function frame(now) {
    loop = 0;
    if (!st.visible || document.hidden) { last = 0; return; }
    const dt = last ? Math.min((now - last) / 1000, 0.05) : 1 / 60;
    last = now;
    if (st.auto) {
      st.clock += dt; st.sweepClock += dt;
      const a = M.autoView(st.clock);
      st.target = { ...st.target, pol: a.pol, az: a.az };
      const i = M.sweepIndex(st.sweepClock, freqs, data.sweep[0], data.sweep[1], st.sweepStart, 0.55, st.sweepHold ?? 3);
      if (i !== st.i) { st.i = i; updateWords(); drawStrip(); }
    }
    const k = 1 - Math.exp(-dt * (st.auto ? 3 : 9));
    let moving = false;
    for (const key of ["pol", "az", "zoom"]) {
      const d = st.target[key] - st.view[key];
      if (Math.abs(d) > 1e-4) { st.view[key] += d * k; moving = true; } else st.view[key] = st.target[key];
    }
    drawHead();
    if (st.auto || moving) loop = requestAnimationFrame(frame); else last = 0;
  }
  function setAuto(on, hold = 1) {
    st.auto = on;
    play.textContent = on ? "Pause" : "Play";
    play.setAttribute("aria-pressed", String(on));
    play.setAttribute("aria-label", on ? "Pause the slow turn and sweep" : "Play the slow turn and sweep");
    if (on) {
      st.clock = 0;
      const inSweep = freqs[st.i] >= data.sweep[0] && freqs[st.i] <= data.sweep[1];
      st.sweepStart = inSweep ? st.i : i12; st.sweepClock = 0; st.sweepHold = hold;
      wake();
    } else updateWords();
  }
  function touch() {
    if (!st.touched) { st.touched = true; say.setAttribute("aria-live", "polite"); }
    if (st.auto) setAuto(false);
  }
  function setFreq(i) { st.i = Math.max(0, Math.min(freqs.length - 1, i)); updateWords(); drawStrip(); drawHead(); }

  // ---------------------------------------------------------------------------------------- input
  root.querySelectorAll(".eh-sets button").forEach((b, k, all) => {
    b.addEventListener("click", () => {
      touch(); st.variant = b.dataset.v;
      all.forEach((x) => { x.setAttribute("aria-checked", String(x === b)); x.tabIndex = x === b ? 0 : -1; });
      updateWords(); drawStrip(); drawHead();
    });
    b.addEventListener("keydown", (e) => {
      const d = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0;
      if (!d) return;
      e.preventDefault();
      const nb = all[(k + d + all.length) % all.length]; nb.focus(); nb.click();
    });
  });
  slider.addEventListener("input", () => { touch(); setFreq(+slider.value); });
  play.addEventListener("click", () => { if (!st.touched) { st.touched = true; say.setAttribute("aria-live", "polite"); } setAuto(!st.auto); });
  const zoomBy = (f) => { st.target.zoom = Math.max(M.VIEW.zoomMin, Math.min(M.VIEW.zoomMax, st.target.zoom * f)); wake(); };
  root.querySelector('[data-act="in"]').addEventListener("click", () => { touch(); zoomBy(1.2); });
  root.querySelector('[data-act="out"]').addEventListener("click", () => { touch(); zoomBy(1 / 1.2); });
  const reset = () => { touch(); st.target = { ...M.VIEW.top, zoom: 1 }; if (reduced.matches) st.view = { ...st.target }; drawHead(); wake(); };
  root.querySelector('[data-act="reset"]').addEventListener("click", reset);

  // drag to turn, two fingers to pinch, Ctrl-wheel (and trackpad pinch) to zoom
  const ptrs = new Map();
  let pinch = null;
  head.addEventListener("pointerdown", (e) => {
    touch(); head.setPointerCapture(e.pointerId); ptrs.set(e.pointerId, [e.clientX, e.clientY]); pinch = null;
  });
  head.addEventListener("pointermove", (e) => {
    if (!ptrs.has(e.pointerId)) return;
    const prev = ptrs.get(e.pointerId); ptrs.set(e.pointerId, [e.clientX, e.clientY]);
    if (ptrs.size === 1) {
      const dx = e.clientX - prev[0], dy = e.clientY - prev[1], k = 2.6 / Math.max(240, HH);
      st.view.az -= dx * k; st.view.pol = Math.max(0, Math.min(M.VIEW.polMax, st.view.pol - dy * k));
      st.target.az = st.view.az; st.target.pol = st.view.pol; drawHead();
    } else if (ptrs.size === 2) {
      const [a, b] = [...ptrs.values()], d = Math.hypot(a[0] - b[0], a[1] - b[1]);
      if (pinch) { st.view.zoom = st.target.zoom = Math.max(M.VIEW.zoomMin, Math.min(M.VIEW.zoomMax, st.view.zoom * (d / pinch))); drawHead(); }
      pinch = d;
    }
  });
  const up = (e) => { ptrs.delete(e.pointerId); pinch = null; };
  head.addEventListener("pointerup", up); head.addEventListener("pointercancel", up);
  head.addEventListener("wheel", (e) => {
    if (!e.ctrlKey && !e.metaKey) return;
    e.preventDefault(); touch(); zoomBy(Math.exp(-e.deltaY * 0.01));
  }, { passive: false });
  head.addEventListener("keydown", (e) => {
    const k = e.key;
    if (k === " ") { e.preventDefault(); setAuto(!st.auto); return; }
    const turn = { ArrowLeft: [0, 0.12], ArrowRight: [0, -0.12], ArrowUp: [-0.1, 0], ArrowDown: [0.1, 0] }[k];
    if (turn) {
      e.preventDefault(); touch();
      st.target.pol = Math.max(0, Math.min(M.VIEW.polMax, st.target.pol + turn[0])); st.target.az += turn[1]; wake(); return;
    }
    if (k === "+" || k === "=") { e.preventDefault(); touch(); zoomBy(1.2); }
    else if (k === "-" || k === "_") { e.preventDefault(); touch(); zoomBy(1 / 1.2); }
    else if (k === "0" || k === "Home") { e.preventDefault(); reset(); }
  });
  // the strip: press or drag along it to move the frequency (it snaps to the measured ones)
  const stripFreq = (e) => {
    const r = strip.getBoundingClientRect(), L = 62, Rm = 12;
    const f = freqs[0] + ((e.clientX - r.left - L) / (r.width - L - Rm)) * (freqs[freqs.length - 1] - freqs[0]);
    setFreq(M.snapIndex(freqs, f));
  };
  strip.addEventListener("pointerdown", (e) => { touch(); strip.setPointerCapture(e.pointerId); stripFreq(e); });
  strip.addEventListener("pointermove", (e) => { if (strip.hasPointerCapture(e.pointerId)) stripFreq(e); });

  // ---------------------------------------------------------------------------------------- lifecycle
  function redrawAll() { size(); updateWords(); drawStrip(); drawHead(); }
  readColors();
  setAuto(st.auto, 3);
  redrawAll();
  new ResizeObserver(() => { size(); drawStrip(); drawHead(); }).observe(root);
  new IntersectionObserver((es) => { st.visible = es[0].isIntersecting; if (st.visible) wake(); }, { threshold: 0.02 }).observe(fig);
  document.addEventListener("visibilitychange", wake);
  const recolor = () => { readColors(); updateWords(); drawStrip(); drawHead(); };
  new MutationObserver(recolor).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  darkQuery.addEventListener("change", recolor);
  reduced.addEventListener("change", () => { if (reduced.matches) setAuto(false); });
  link.addEventListener("load", redrawAll);
  if (document.fonts) document.fonts.ready.then(redrawAll);
  if (document.fonts) document.fonts.load(`12px ${C.type}`).then(() => { drawStrip(); drawHead(); }).catch(() => {});

  // the latest check run, quietly under the plate
  fetch(fig.dataset.src.replace(/\.json$/, ".checks.json")).then((r) => (r.ok ? r.json() : null)).then((c) => {
    if (!c || !c.line) return;
    const p = $(".eh-checked"); p.textContent = c.line; p.hidden = false;
  }).catch(() => {});
}

/** Andrew's monotone chain. */
function convexHull(pts) {
  const p = pts.map((q) => [q.x, q.y]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], hi = [];
  for (const q of p) { while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
  for (let i = p.length - 1; i >= 0; i--) { const q = p[i]; while (hi.length >= 2 && cr(hi[hi.length - 2], hi[hi.length - 1], q) <= 0) hi.pop(); hi.push(q); }
  return [...lo.slice(0, -1), ...hi.slice(0, -1)].map(([x, y]) => ({ x, y }));
}
