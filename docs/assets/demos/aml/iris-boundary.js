// "Where the rule draws the line" (chapter I, live plate): the study's classifier refit in the page
// on the two petal measurements, its decision score drawn over the petal plane as a stipple
// engraving. Fine dots fill the plane, more of them where the score is larger, in the ink of the
// class the rule picks there; the boundary, where the score crosses zero, is a firm engraved line;
// dotted level lines mark fixed scores; and the flowers sit on top as seeds, filled for the
// problem's first class and open for the other, with a short ray for each flower when several share
// a spot. Left alone, the plate pulls a fresh print for each kernel in turn, so the boundary bends
// from a straight line into curves.
//
// Numbers and sentences come from iris-boundary-model.js, which the check script holds to the
// study's own code; colors come from the page's inks, read again when the printing changes.
const KIND = "iris-boundary";
// The page loads this file with a version query; the model and the stylesheet get the same one.
const V = new URL(import.meta.url).search;
let Mo = null;
const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

const ORDER = ["dot", "d2", "d3", "gauss"];
const DWELL = 4.6, WIPE = 1.7, WIPE_TAP = 0.55; // seconds: each print's stay, and the roller's pass
const LEVELS = [-3, -2, -1, 1, 2, 3];

function ensureCss() {
  if (document.querySelector(`link[data-css="${KIND}"]`)) return;
  const l = document.createElement("link");
  l.rel = "stylesheet"; l.href = new URL(`./${KIND}.css${V}`, import.meta.url).href; l.dataset.css = KIND;
  document.head.append(l);
}

const SVG = "http://www.w3.org/2000/svg";
function el(tag, attrs = {}, kids = []) {
  const n = tag.startsWith("svg:") ? document.createElementNS(SVG, tag.slice(4)) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") n.textContent = v;
    else if (k === "html") n.innerHTML = v;
    else n.setAttribute(k, v);
  }
  for (const k of kids) n.append(k);
  return n;
}

// Arrow keys move the choice within a radio group, with one tab stop for the group.
function radioKeys(group) {
  group.addEventListener("keydown", (e) => {
    const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
    if (!step) return;
    const bs = [...group.querySelectorAll("button")];
    const i = bs.indexOf(document.activeElement);
    if (i < 0) return;
    e.preventDefault();
    const next = bs[(i + step + bs.length) % bs.length];
    next.click(); next.focus();
  });
}

/** A small glyph for each kernel's button: the kind of line it draws. */
function glyph(id) {
  const s = el("svg:svg", { viewBox: "0 0 22 16", class: "ib-glyph", "aria-hidden": "true" });
  const d = {
    dot: "M3 14 L19 2",
    d2: "M3 2 C 8 16, 14 16, 19 2",
    d3: "M2 13 C 7 -4, 12 20, 20 3",
    gauss: "M11 2.5 C 18 2.5, 19.5 8, 16 11 C 13 14, 9 13.5, 6.5 12.5 C 2.5 11, 2.5 6, 5 4.2 C 6.8 3, 8.5 2.5, 11 2.5 Z",
  }[id];
  s.append(el("svg:path", { d, class: "ib-glyph-line" }));
  return s;
}

// Interleaved gradient noise: a cheap, even ordering for the stipple sites, so the dots thin out
// evenly rather than in clumps.
const ign = (i, j) => { const t = 0.06711056 * i + 0.00583715 * j; const u = 52.9829189 * (t - Math.floor(t)); return u - Math.floor(u); };
const hash = (i) => { let x = Math.imul(i ^ 0x9e3779b9, 0x85ebca6b); x ^= x >>> 13; x = Math.imul(x, 0xc2b2ae35); x ^= x >>> 16; return (x >>> 0) / 4294967296; };

export async function mount(fig) {
  ensureCss();
  const root = fig.querySelector(".demo-root");
  if (!root || root.dataset.mounted) return;
  root.dataset.mounted = "1";
  Mo = Mo || (await import(`./${KIND}-model.js${V}`));
  const data = await (await fetch(fig.dataset.src)).json();
  const M = Mo.shape(data);
  const reduceQ = matchMedia("(prefers-reduced-motion: reduce)");
  const coarse = matchMedia("(pointer: coarse)").matches;
  const still = new URLSearchParams(location.search).has("still");
  const st = {
    n: 3, id: reduceQ.matches ? "gauss" : "dot", k: 0,
    cycling: !reduceQ.matches && !still, touched: false, probe: null,
  };
  // the probe starts on the one spot that versicolor and virginica flowers share
  st.probe = { x: [4.8, 1.8], at: Mo.flowersAt(M, st.n, [4.8, 1.8]) };

  // ---------------------------------------------------------------- controls
  const probs = el("div", { class: "ib-problems", role: "radiogroup", "aria-label": "Problem" });
  for (const P of M.problems) {
    const b = el("button", { type: "button", role: "radio", "data-n": P.n }, [el("span", { class: "ib-pn", text: String(P.n) }), el("span", { text: Mo.buttonLabel(M, P.n) })]);
    b.addEventListener("click", () => { touch(); setState({ n: P.n }); });
    probs.append(b);
  }
  radioKeys(probs);
  const KNAME = { dot: "Dot product", d2: "Degree 2", d3: "Degree 3", gauss: "Gaussian" };
  const KLONG = { dot: "Dot product", d2: "Polynomial, degree 2", d3: "Polynomial, degree 3", gauss: "Gaussian" };
  const kers = el("div", { class: "ib-kernels", role: "radiogroup", "aria-label": "Kernel" });
  for (const id of ORDER) {
    const b = el("button", { type: "button", role: "radio", "data-id": id, "aria-label": KLONG[id] }, [glyph(id), el("span", { text: KNAME[id] })]);
    b.addEventListener("click", () => { touch(); setState({ id }); });
    kers.append(b);
  }
  radioKeys(kers);
  const slider = el("input", { type: "range", min: String(Mo.K_MIN), max: String(Mo.K_MAX), step: "1", value: "0", class: "ib-range", "aria-label": "Gaussian width" });
  const studyMark = el("span", { class: "ib-mark", "aria-hidden": "true" }, [el("span", { class: "ib-mark-tick" }), el("span", { class: "ib-mark-lab", text: "study" })]);
  studyMark.style.left = `calc(${((0 - Mo.K_MIN) / (Mo.K_MAX - Mo.K_MIN)) * 100}% + ${(0.5 - (0 - Mo.K_MIN) / (Mo.K_MAX - Mo.K_MIN)) * 22}px)`;
  const wout = el("output", { class: "ib-wout", "aria-live": "off" });
  const studyB = el("button", { type: "button", class: "ib-tool ib-studyw", text: "Study’s width" });
  const widthRow = el("div", { class: "ib-width" }, [
    el("span", { class: "ib-k", text: "Gaussian width" }),
    el("div", { class: "ib-wline" }, [el("div", { class: "ib-track" }, [slider, studyMark]), studyB]),
    wout,
  ]);
  let slideRaf = 0;
  slider.addEventListener("input", () => {
    touch();
    const k = Number(slider.value);
    if (slideRaf) cancelAnimationFrame(slideRaf);
    slideRaf = requestAnimationFrame(() => { slideRaf = 0; setState({ id: "gauss", k }, { wipe: false }); });
  });
  studyB.addEventListener("click", () => { touch(); setState({ id: "gauss", k: 0 }); });
  const controls = el("div", { class: "ib-controls" }, [
    el("div", { class: "ib-field" }, [el("span", { class: "ib-k", text: "Problem", "aria-hidden": "true" }), probs]),
    el("div", { class: "ib-field" }, [el("span", { class: "ib-k", text: "Kernel", "aria-hidden": "true" }), kers]),
    widthRow,
  ]);

  // ---------------------------------------------------------------- the plane
  const cv = el("canvas", { class: "ib-cv", tabindex: "0", role: "img" });
  const plane = el("div", { class: "ib-plane" }, [cv]);
  const cycleB = el("button", { type: "button", class: "ib-tool ib-cycle" });
  const clearB = el("button", { type: "button", class: "ib-tool", text: "Clear the probe" });
  const hint = el("p", { class: "ib-hint", text: coarse ? "Tap the plane to read the score there; tap a seed to read its flowers." : "Click or drag on the plane to read the score there. With the plane focused, arrow keys move the probe, comma and period step through the flowers, and Escape clears it." });
  const tools = el("div", { class: "ib-tools" }, [cycleB, clearB, hint]);

  // ---------------------------------------------------------------- readouts
  const big = el("span", { class: "ib-num" });
  const pick = el("p", { class: "ib-pick" });
  const gauge = el("svg:svg", { class: "ib-gauge", viewBox: "0 0 320 46", "aria-hidden": "true" });
  const probeBody = el("p", { class: "ib-card-body" });
  // the probe card starts silent: until the reader touches the plate, it cycles the kernels on its own,
  // and the score at the probe changes with each one (update() sets aria-live)
  const probeCard = el("section", { class: "ib-card ib-probe", "aria-live": "off" }, [el("p", { class: "ib-k", text: "Score at the probe" }), el("p", { class: "ib-big" }, [big]), pick, gauge, probeBody]);
  const fml = el("p", { class: "ib-fml" });
  const ruleB = el("p", { class: "ib-rule-b" });
  const ruleN = el("p", { class: "ib-rule-n" });
  const ruleCard = el("section", { class: "ib-card ib-rule" }, [el("p", { class: "ib-k", text: "The rule, refit here" }), fml, ruleB, ruleN]);
  const caption = el("p", { class: "ib-caption", "aria-live": "off" });
  const legend = el("div", { class: "ib-legend" });
  const checked = el("p", { class: "ib-checked" });
  fetch(fig.dataset.src.replace(/\.json$/, ".checks.json")).then((r) => (r.ok ? r.json() : null)).then((c) => { if (c && c.line) checked.textContent = c.line; }).catch(() => {});
  root.append(el("div", { class: "ib" }, [controls, plane, tools, el("div", { class: "ib-read" }, [probeCard, ruleCard]), caption, legend, checked]));

  // ---------------------------------------------------------------- colors, read from the page
  let C = null;
  function readColors() {
    const cs = getComputedStyle(fig), g = (n) => cs.getPropertyValue(n).trim();
    const probe = document.createElement("canvas").getContext("2d");
    probe.fillStyle = g("--paper");
    const h = probe.fillStyle;
    const rgb = h.startsWith("#") ? [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)) : [255, 255, 255];
    C = {
      ink: g("--ink"), ink2: g("--ink-2"), paper: g("--paper"), paper2: g("--paper-2"), rule: g("--rule"), rule2: g("--rule-2"),
      accent: g("--accent-ink"), sp: [g("--ib-setosa"), g("--ib-versicolor"), g("--ib-virginica")], rest: g("--ib-rest"),
      type: g("--type") || "monospace", serif: g("--serif") || "serif",
      dark: 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] < 128,
    };
  }
  /** The two inks of problem n: the first class's species ink, and the other class's (plain ink for two species pooled). */
  function inks(n) {
    const c = Mo.classes(M, n);
    return { a: C.sp[c.one[0]], b: c.other.length === 2 ? C.rest : C.sp[c.other[0]] };
  }

  // ---------------------------------------------------------------- size and geometry
  let W = 0, H = 0, dpr = 1, geo = null, sites = null;
  const ctx = cv.getContext("2d");
  function resize() {
    const w = plane.clientWidth;
    if (!w || w < 160) return;
    const small = w < 560;
    const mL = small ? 34 : 46, mR = small ? 8 : 14, mT = small ? 26 : 28, mB = small ? 38 : 42;
    const sx = Mo.WINDOW.x[1] - Mo.WINDOW.x[0], sy = Mo.WINDOW.y[1] - Mo.WINDOW.y[0];
    const pw = w - mL - mR, kx = pw / sx;
    // one centimeter is the same length both ways where there is room; on a narrow screen petal
    // width is drawn larger than petal length, so the plane stays tall enough to read and tap
    const ph = Math.round(Math.max(small ? 250 : 262, Math.min(420, kx * sy)));
    const h = ph + mT + mB;
    const d = Math.min(window.devicePixelRatio || 1, 2);
    if (w === W && h === H && d === dpr && geo) return;
    W = w; H = h; dpr = d;
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    cv.style.height = `${H}px`;
    geo = { small, L: mL, T: mT, R: W - mR, B: mT + ph, pw, ph, kx, ky: ph / sy };
    sites = makeSites();
    layers.clear();
    // a new size cancels any wipe and paints the current plot whole, at once if the plate is on screen
    wipe = null; cur = null;
    if (C && visible && !document.hidden) draw(performance.now());
    else describe(null);
    need = true; wake();
  }
  const X = (x) => geo.L + (x - Mo.WINDOW.x[0]) * geo.kx;
  const Y = (y) => geo.B - (y - Mo.WINDOW.y[0]) * geo.ky;
  const toCm = (px, py) => [Mo.WINDOW.x[0] + (px - geo.L) / geo.kx, Mo.WINDOW.y[0] + (geo.B - py) / geo.ky];

  /** The stipple sites: a jittered grid over the plane, each with its own place in the thinning order. */
  function makeSites() {
    const sp = geo.small ? 2.05 : 2.3;
    const nx = Math.floor(geo.pw / sp), ny = Math.floor(geo.ph / sp);
    const n = nx * ny;
    const px = new Float32Array(n), py = new Float32Array(n), cx = new Float64Array(n), cy = new Float64Array(n), rank = new Float32Array(n);
    let k = 0;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const x = geo.L + (i + 0.5 + (hash(k * 2 + 1) - 0.5) * 0.7) * (geo.pw / nx);
      const y = geo.T + (j + 0.5 + (hash(k * 2 + 2) - 0.5) * 0.7) * (geo.ph / ny);
      px[k] = x; py[k] = y;
      const c = toCm(x, y); cx[k] = c[0]; cy[k] = c[1];
      rank[k] = ign(i, j);
      k++;
    }
    return { n, px, py, cx, cy, rank };
  }

  // ---------------------------------------------------------------- fits, cached
  const fits = new Map();
  function fitOf(n, id, k) {
    const key = `${n}|${id}|${id === "gauss" ? k : 0}`;
    if (!fits.has(key)) {
      const F = Mo.fit(M, n, id, Mo.sigmaAt(M, id === "gauss" ? k : 0));
      fits.set(key, { F, S: Mo.levelStep(F) });
      if (fits.size > 48) fits.delete(fits.keys().next().value);
    }
    return fits.get(key);
  }

  // ---------------------------------------------------------------- one print: stipple, level lines, boundary
  const layers = new Map(); // key -> {canvas, ...}, the most recent few
  const keyOf = (s) => `${s.n}|${s.id}|${s.id === "gauss" ? s.k : 0}`;
  function layerOf(s) {
    const key = keyOf(s);
    if (layers.has(key)) { const L = layers.get(key); layers.delete(key); layers.set(key, L); return L; }
    const L = renderLayer(s);
    layers.set(key, L);
    while (layers.size > 7) layers.delete(layers.keys().next().value);
    return L;
  }
  function renderLayer(s) {
    const { F, S } = fitOf(s.n, s.id, s.k);
    const c = document.createElement("canvas");
    c.width = cv.width; c.height = cv.height;
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const ink = inks(s.n);
    g.save();
    g.beginPath(); g.rect(geo.L, geo.T, geo.pw, geo.ph); g.clip();

    // the stipple: a site is inked when its place in the thinning order is under the density there
    const DMAX = C.dark ? 0.5 : 0.54, R0 = geo.small ? 0.4 : 0.42, R1 = geo.small ? 0.46 : 0.52;
    const pa = new Path2D(), pb = new Path2D();
    const L3 = 3 * S.L;
    for (let i = 0; i < sites.n; i++) {
      const h = F.fast(sites.cx[i], sites.cy[i]);
      const d = Math.min(1, Math.abs(h) / L3);
      if (sites.rank[i] >= DMAX * d) continue;
      const r = R0 + R1 * d;
      const p = h >= 0 ? pa : pb;
      p.moveTo(sites.px[i] + r, sites.py[i]);
      p.arc(sites.px[i], sites.py[i], r, 0, 2 * Math.PI);
    }
    g.fillStyle = ink.a; g.fill(pa);
    g.fillStyle = ink.b; g.fill(pb);

    // level lines and the boundary, found on a grid of about 4 px
    const G = Mo.gridSpec(Math.max(24, Math.round(geo.pw / 4)), Math.max(12, Math.round(geo.ph / 4)));
    const vals = Mo.gridValues(F, G);
    const path = (line) => { const p = new Path2D(); line.forEach((q, i) => (i ? p.lineTo(X(q.x[0]), Y(q.x[1])) : p.moveTo(X(q.x[0]), Y(q.x[1])))); return p; };
    const labels = [];
    const levelLines = {};
    for (const j of LEVELS) {
      const lines = Mo.contour(F, G, vals, j * S.L, 7);
      levelLines[j] = lines;
      g.lineJoin = "round"; g.lineCap = "round";
      for (const line of lines) {
        const p = path(line);
        g.setLineDash([]); g.strokeStyle = C.paper2; g.lineWidth = 5; g.stroke(p);
        g.setLineDash([0.01, 4.4]); g.strokeStyle = j > 0 ? ink.a : ink.b; g.lineWidth = geo.small ? 1.8 : 2; g.stroke(p);
      }
    }
    g.setLineDash([]);
    const boundary = Mo.contour(F, G, vals, 0);
    for (const line of boundary) {
      const p = path(line);
      g.strokeStyle = C.paper2; g.lineWidth = geo.small ? 5.5 : 6.5; g.stroke(p);
      g.strokeStyle = C.ink; g.lineWidth = geo.small ? 1.9 : 2.2; g.stroke(p);
    }

    g.restore();

    // labels on the level lines and the boundary, where they fit
    const spots = flowerSpots(s.n);
    const clear = (x, y, w, h) => x - w / 2 > geo.L + 2 && x + w / 2 < geo.R - 2 && y - h / 2 > geo.T + 2 && y + h / 2 < geo.B - 2
      && labels.every((r) => Math.abs(r.x - x) > (r.w + w) / 2 + 3 || Math.abs(r.y - y) > (r.h + h) / 2 + 2)
      && spots.every((q) => Math.abs(X(q.x[0]) - x) > w / 2 + 6 || Math.abs(Y(q.x[1]) - y) > h / 2 + 6);
    const label = (lines, text, color, weight) => {
      g.font = `${weight} ${geo.small ? 10.5 : 11.5}px ${C.type}`;
      const w = g.measureText(text).width + 2, h = 12;
      const cands = [];
      for (const line of lines) {
        const pts = line.map((q) => [X(q.x[0]), Y(q.x[1])]);
        let len = 0; const cum = [0];
        for (let i = 1; i < pts.length; i++) { len += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]); cum.push(len); }
        if (len < 60) continue;
        for (const f of [0.5, 0.35, 0.65, 0.2, 0.8, 0.1, 0.9]) {
          const at = f * len; let i = 1; while (i < cum.length - 1 && cum[i] < at) i++;
          const t = (at - cum[i - 1]) / Math.max(1e-9, cum[i] - cum[i - 1]);
          cands.push({ x: pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * t, y: pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * t, len });
        }
      }
      cands.sort((a, b) => b.len - a.len);
      const c0 = cands.find((q) => clear(q.x, q.y, w, h));
      if (!c0) return;
      labels.push({ x: c0.x, y: c0.y, w, h });
      g.textAlign = "center"; g.textBaseline = "middle";
      g.fillStyle = C.paper2; g.fillRect(c0.x - w / 2 - 2, c0.y - h / 2, w + 4, h);
      g.fillStyle = color; g.fillText(text, c0.x, c0.y + 0.5);
    };
    label(boundary, "0", C.ink, 700);
    for (const j of [1, -1, 2, -2, 3, -3]) label(levelLines[j], Mo.fmtLevel(j, S), j > 0 ? ink.a : ink.b, 400);

    // the Gaussian's width, as a scale bar in a corner no flower uses
    if (s.id === "gauss") {
      const x0 = X(0.75), x1 = X(0.75 + F.sigma), y = Y(2.48);
      g.strokeStyle = C.paper2; g.lineWidth = 5; g.lineCap = "butt";
      g.beginPath(); g.moveTo(x0, y); g.lineTo(x1, y); g.stroke();
      g.strokeStyle = C.ink; g.lineWidth = 1.2;
      g.beginPath(); g.moveTo(x0, y); g.lineTo(x1, y); g.moveTo(x0, y - 4); g.lineTo(x0, y + 4); g.moveTo(x1, y - 4); g.lineTo(x1, y + 4); g.stroke();
      const t = `σ = ${Mo.fmtSigma(F.sigma)} cm${s.k === 0 ? " (the study’s)" : ""}`;
      g.font = `italic ${geo.small ? 11 : 12}px ${C.serif}`; g.textAlign = "left"; g.textBaseline = "bottom";
      const tw = g.measureText(t).width;
      g.fillStyle = C.paper2; g.fillRect(x0 - 2, y - 20, tw + 6, 14);
      g.fillStyle = C.ink; g.fillText(t, x0, y - 6);
    }
    return { canvas: c, F, S, boundary };
  }
  function star(g, p, color) {
    g.save();
    g.fillStyle = C.paper2; g.beginPath(); g.arc(p[0], p[1], 9.5, 0, 2 * Math.PI); g.fill();
    g.strokeStyle = color; g.lineWidth = 1.6;
    g.beginPath(); g.arc(p[0], p[1], 5.6, 0, 2 * Math.PI); g.stroke();
    g.beginPath();
    for (const [dx, dy] of [[0, -1], [0, 1], [-1, 0], [1, 0]]) { g.moveTo(p[0] + dx * 3, p[1] + dy * 3); g.lineTo(p[0] + dx * 9, p[1] + dy * 9); }
    g.lineWidth = 1.3; g.stroke();
    g.fillStyle = color; g.beginPath(); g.arc(p[0], p[1], 1.6, 0, 2 * Math.PI); g.fill();
    g.restore();
  }

  // ---------------------------------------------------------------- the flowers, as seeds
  const spotCache = new Map();
  /** The problem's flowers grouped by spot, in order of their first flower. */
  function flowerSpots(n) {
    if (spotCache.has(n)) return spotCache.get(n);
    const C0 = Mo.classes(M, n), by = new Map();
    for (const f of C0.rows) {
      const key = `${f.x[0]},${f.x[1]}`;
      if (!by.has(key)) by.set(key, { x: f.x, at: [] });
      by.get(key).at.push(f.i);
    }
    const out = [...by.values()];
    spotCache.set(n, out);
    return out;
  }
  function drawSeeds() {
    const ink = inks(st.n), one = Mo.problem(M, st.n).one;
    const r = geo.small ? 2.3 : 3.2, halo = geo.small ? 1.2 : 1.5;
    for (const q of flowerSpots(st.n)) {
      const x = X(q.x[0]), y = Y(q.x[1]);
      const cls = q.at.map((i) => (M.flowers[i].s === one ? "a" : "b"));
      if (q.at.length > 1) {
        // a ray for each flower at this spot, in its class's ink
        ctx.lineCap = "round";
        const n = q.at.length;
        for (let k = 0; k < n; k++) {
          const t = -Math.PI / 2 + (2 * Math.PI * k) / n;
          const c = Math.cos(t), s = Math.sin(t);
          ctx.strokeStyle = C.paper2; ctx.lineWidth = geo.small ? 2.6 : 3;
          const r0 = r + halo + 0.2, r1 = r + (geo.small ? 4.6 : 5.4);
          ctx.beginPath(); ctx.moveTo(x + c * r0, y + s * r0); ctx.lineTo(x + c * r1, y + s * r1); ctx.stroke();
          ctx.strokeStyle = cls[k] === "a" ? ink.a : ink.b; ctx.lineWidth = geo.small ? 1 : 1.15;
          ctx.beginPath(); ctx.moveTo(x + c * r0, y + s * r0); ctx.lineTo(x + c * r1, y + s * r1); ctx.stroke();
        }
      }
      // the seed: filled for the first class, open for the other, split when both sit here
      ctx.fillStyle = C.paper2; ctx.beginPath(); ctx.arc(x, y, r + halo, 0, 2 * Math.PI); ctx.fill();
      const hasA = cls.includes("a"), hasB = cls.includes("b");
      if (hasA && hasB) {
        ctx.fillStyle = ink.a; ctx.beginPath(); ctx.arc(x, y, r, Math.PI / 2, (3 * Math.PI) / 2); ctx.closePath(); ctx.fill();
        ctx.strokeStyle = ink.b; ctx.lineWidth = 1.3; ctx.beginPath(); ctx.arc(x, y, r - 0.35, -Math.PI / 2, Math.PI / 2); ctx.stroke();
      } else if (hasA) {
        ctx.fillStyle = ink.a; ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill();
      } else {
        ctx.strokeStyle = ink.b; ctx.lineWidth = 1.3; ctx.beginPath(); ctx.arc(x, y, r - 0.35, 0, 2 * Math.PI); ctx.stroke();
      }
    }
  }

  // ---------------------------------------------------------------- the frame: plate mark, rulers, labels
  function drawFrame() {
    const { L, T, R, B } = geo;
    ctx.strokeStyle = C.rule2; ctx.lineWidth = 1;
    ctx.strokeRect(L - 0.5, T - 0.5, R - L + 1, B - T + 1);
    ctx.strokeStyle = C.rule; ctx.strokeRect(L - 4.5, T - 4.5, R - L + 9, B - T + 9);
    ctx.font = `${geo.small ? 10.5 : 11.5}px ${C.type}`;
    ctx.fillStyle = C.ink2; ctx.strokeStyle = C.ink2; ctx.lineWidth = 1;
    // petal length: a tick every half centimeter, numbered every centimeter
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (let v = Math.ceil(Mo.WINDOW.x[0] * 2) / 2; v <= Mo.WINDOW.x[1] + 1e-9; v += 0.5) {
      const x = Math.round(X(v)) + 0.5, whole = Math.abs(v - Math.round(v)) < 1e-9;
      ctx.beginPath(); ctx.moveTo(x, B + 4.5); ctx.lineTo(x, B + 4.5 + (whole ? 6 : 3)); ctx.stroke();
      if (whole) ctx.fillText(String(v), x, B + 13);
    }
    // petal width: a tick every tenth, numbered every half centimeter
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (let t = Math.ceil(Mo.WINDOW.y[0] * 10); t <= Math.floor(Mo.WINDOW.y[1] * 10 + 1e-9); t++) {
      const y = Math.round(Y(t / 10)) + 0.5, labeled = t % 5 === 0;
      ctx.beginPath(); ctx.moveTo(L - 4.5, y); ctx.lineTo(L - 4.5 - (labeled ? 6 : 2.5), y); ctx.stroke();
      if (labeled) ctx.fillText((t / 10).toFixed(1), L - 13, y);
    }
    ctx.font = `italic ${geo.small ? 11.5 : 12.5}px ${C.serif}`;
    ctx.textAlign = "right"; ctx.textBaseline = "top";
    ctx.fillText("petal length, cm", R, B + (geo.small ? 25 : 28));
    ctx.textAlign = "left"; ctx.textBaseline = "bottom";
    ctx.fillText("petal width, cm", L - (geo.small ? 30 : 40), T - 8);
  }

  // ---------------------------------------------------------------- the probe
  function drawProbe(layer) {
    const p = st.probe;
    if (!p) return;
    const x = X(p.x[0]), y = Y(p.x[1]);
    const h = Mo.score(layer.F, p.x);
    // leaders to the rulers
    ctx.save();
    ctx.setLineDash([1.5, 3]); ctx.strokeStyle = C.ink; ctx.globalAlpha = 0.65; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, y + 12); ctx.lineTo(x, geo.B + 4); ctx.moveTo(x - 12, y); ctx.lineTo(geo.L - 4, y); ctx.stroke();
    ctx.restore();
    // the reticle
    ctx.strokeStyle = C.paper2; ctx.lineWidth = 4.5;
    ctx.beginPath(); ctx.arc(x, y, 8.5, 0, 2 * Math.PI); ctx.stroke();
    ctx.strokeStyle = C.accent; ctx.lineWidth = 1.8;
    ctx.beginPath(); ctx.arc(x, y, 8.5, 0, 2 * Math.PI); ctx.stroke();
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    for (const [dx, dy] of [[0, -1], [0, 1], [-1, 0], [1, 0]]) { ctx.moveTo(x + dx * 10.5, y + dy * 10.5); ctx.lineTo(x + dx * 14.5, y + dy * 14.5); }
    ctx.stroke();
    // the reading beside it
    const t = Mo.fmtScore(h, layer.S);
    ctx.font = `${geo.small ? 12 : 13}px ${C.type}`;
    const tw = ctx.measureText(t).width;
    let tx = x + 18, ty = y - 16;
    if (tx + tw + 6 > geo.R) tx = x - 18 - tw;
    if (ty - 9 < geo.T) ty = y + 18;
    ctx.fillStyle = C.paper2; ctx.globalAlpha = 0.92; ctx.fillRect(tx - 4, ty - 9, tw + 8, 18); ctx.globalAlpha = 1;
    ctx.strokeStyle = C.accent; ctx.lineWidth = 1; ctx.strokeRect(tx - 3.5, ty - 8.5, tw + 7, 17);
    ctx.fillStyle = C.ink; ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.fillText(t, tx, ty + 0.5);
    // the coordinates on the rulers
    ctx.font = `${geo.small ? 10.5 : 11}px ${C.type}`;
    const dp = p.at.length ? 1 : 2;
    const cx = Mo.halfUp(p.x[0], dp), cy = Mo.halfUp(p.x[1], dp);
    const wx = ctx.measureText(cx).width + 8, wy = ctx.measureText(cy).width + 8;
    const bx = Math.max(geo.L - 4, Math.min(geo.R + 4 - wx, x - wx / 2));
    ctx.fillStyle = C.accent; ctx.fillRect(bx, geo.B + 8, wx, 15);
    ctx.fillRect(geo.L - 6 - wy, Math.max(geo.T - 4, Math.min(geo.B - 11, y - 7.5)), wy, 15);
    ctx.fillStyle = C.paper2; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(cx, bx + wx / 2, geo.B + 16);
    ctx.fillText(cy, geo.L - 6 - wy / 2, Math.max(geo.T - 4, Math.min(geo.B - 11, y - 7.5)) + 8);
  }

  // ---------------------------------------------------------------- the loop: prints and the roller
  let cur = null;            // the print shown: {s, layer}
  let wipe = null;           // {from, to, t0, dur}
  let need = true, raf = 0, visible = false, nextAt = 0, timer = 0;
  const jag = Array.from({ length: 200 }, (_, i) => (hash(i + 77) - 0.5) * 7 + Math.sin(i * 0.9) * 2.5);
  /** The region the roller has passed (left of the front at fx), or the region it has not (right). */
  function sidePath(fx, left) {
    const p = new Path2D(), step = 6, edge = left ? -10 : W + 10;
    p.moveTo(edge, -step);
    for (let y = -step, i = 0; y <= H + step; y += step, i++) p.lineTo(fx + jag[i % jag.length], y);
    p.lineTo(edge, H + step); p.closePath();
    return p;
  }
  function draw(now) {
    if (!geo || !C) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const s = { n: st.n, id: st.id, k: st.k };
    if (!cur || keyOf(cur.s) !== keyOf(s)) cur = { s, layer: layerOf(s) };
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    if (wipe && now - wipe.t0 >= wipe.dur * 1000) wipe = null;
    if (wipe) {
      const t = Math.max(0, (now - wipe.t0) / (wipe.dur * 1000));
      const e = t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2;
      const fx = geo.L - 14 + e * (geo.pw + 28);
      for (const [layer, left] of [[wipe.from, false], [cur.layer, true]]) {
        ctx.save(); ctx.scale(dpr, dpr); ctx.clip(sidePath(fx, left)); ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.drawImage(layer.canvas, 0, 0);
        ctx.restore();
      }
      ctx.save(); ctx.scale(dpr, dpr);
      ctx.beginPath(); ctx.rect(geo.L, geo.T, geo.pw, geo.ph); ctx.clip();
      ctx.beginPath();
      for (let y = -6, i = 0; y <= H + 6; y += 6, i++) (i ? ctx.lineTo(fx + jag[i % jag.length], y) : ctx.moveTo(fx + jag[i % jag.length], y));
      ctx.strokeStyle = C.accent; ctx.globalAlpha = 0.55 * Math.sin(Math.PI * Math.min(1, t)); ctx.lineWidth = 1.2; ctx.lineJoin = "round"; ctx.stroke();
      ctx.restore();
    } else ctx.drawImage(cur.layer.canvas, 0, 0);
    root.dataset.drawn = wipe ? "" : keyOf(cur.s);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawFrame();
    drawSeeds();
    // the class averages, when the rule compares distances to them (the dot product), above the seeds
    if (st.id === "dot" && !wipe) {
      const A = Mo.averages(cur.layer.F), ink = inks(st.n);
      const pa = [X(A.a[0]), Y(A.a[1])], pb = [X(A.b[0]), Y(A.b[1])];
      ctx.save(); ctx.setLineDash([2, 3]); ctx.strokeStyle = C.ink2; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(...pa); ctx.lineTo(...pb); ctx.stroke(); ctx.restore();
      star(ctx, pa, ink.a); star(ctx, pb, ink.b);
    }
    drawProbe(cur.layer);
    // with no wipe running, this paint has drawn the current plot whole, at the current size
    if (!wipe) describe(cur.s);
  }
  function frame(now) {
    raf = 0;
    draw(now);
    need = false;
    if (wipe && visible && !document.hidden) raf = requestAnimationFrame(frame);
    else if (wipe) { wipe = null; need = true; wake(); }
    schedule();
  }
  function wake() { if (!raf && visible && !document.hidden) raf = requestAnimationFrame(frame); }

  // The cycle: after each print has stood for DWELL seconds, the next kernel is pulled across.
  function schedule() {
    clearTimeout(timer);
    if (!st.cycling || !visible || document.hidden || reduceQ.matches || wipe) return;
    const wait = Math.max(200, nextAt - performance.now());
    timer = setTimeout(() => {
      if (!st.cycling || !visible || document.hidden) return;
      const id = ORDER[(ORDER.indexOf(st.id) + 1) % ORDER.length];
      setState({ id, k: id === "gauss" ? 0 : st.k }, { auto: true });
    }, wait);
    // print the next one ahead of time, so the roller never waits on it
    const id = ORDER[(ORDER.indexOf(st.id) + 1) % ORDER.length];
    const s = { n: st.n, id, k: id === "gauss" ? 0 : st.k };
    if (!layers.has(keyOf(s))) setTimeout(() => { if (geo && C) layerOf(s); }, 60);
  }

  function setState(ch, { auto = false, wipe: doWipe = true } = {}) {
    const before = { n: st.n, id: st.id, k: st.k };
    Object.assign(st, ch);
    const after = { n: st.n, id: st.id, k: st.k };
    if (keyOf(before) === keyOf(after)) { update(); return; }
    if (st.probe) st.probe.at = Mo.flowersAt(M, st.n, st.probe.x).length && st.probe.at.length ? Mo.flowersAt(M, st.n, st.probe.x) : [];
    if (geo && C && cur && doWipe && !reduceQ.matches && before.n === after.n) {
      wipe = { from: cur.layer, t0: performance.now(), dur: auto ? WIPE : WIPE_TAP };
    } else wipe = null;
    // update() first, so that after a manual change the live regions are on before the new words are
    // written (while the kernels cycle on their own, they stay off)
    update();
    // with no wipe, the new plot is painted now, and that paint describes it; otherwise the words say
    // "Redrawing the plot." until a paint draws it whole
    if (!wipe && geo && C && visible && !document.hidden) draw(performance.now());
    else describe(null);
    nextAt = performance.now() + (DWELL + (auto ? WIPE : 0)) * 1000;
    need = true; wake();
  }
  // touch() records a deliberate action and stops the cycling; update() makes the caption and the probe
  // card live only while the plate has been touched and is not cycling. Only the flags change here; every
  // caller then calls update() once, after its own change, so the live regions are written with the new
  // state and never with the old one first. Slider inputs within one frame coalesce into one update.
  function touch() {
    st.touched = true;
    if (st.cycling) { st.cycling = false; clearTimeout(timer); }
  }
  cycleB.addEventListener("click", () => {
    st.touched = true;
    st.cycling = !st.cycling && !reduceQ.matches;
    nextAt = performance.now() + 900;
    update(); schedule();
  });
  clearB.addEventListener("click", () => { touch(); st.probe = null; update(); need = true; wake(); });

  // ---------------------------------------------------------------- words and readouts
  // The caption and the plane's accessible name are written only here. A paint that draws a plot whole
  // passes that plot (draw(), with no wipe running), and both then describe it; a change of plot or of
  // size passes null (setState(), resize()), and both say "Redrawing the plot." until that paint.
  let named = null;
  function describe(s) {
    const t = s ? Mo.stateText(M, s.n, s.id, s.k, fitOf(s.n, s.id, s.k).F).join(" ") : "Redrawing the plot.";
    const a = s ? Mo.summary(M, s.n, s.id, s.k) : "Redrawing the plot.";
    if (caption.textContent !== t) caption.textContent = t;
    if (named !== a) { named = a; cv.setAttribute("aria-label", a); }
  }
  function update() {
    probs.querySelectorAll("button").forEach((b) => { const on = Number(b.dataset.n) === st.n; b.setAttribute("aria-checked", String(on)); b.tabIndex = on ? 0 : -1; });
    kers.querySelectorAll("button").forEach((b) => { const on = b.dataset.id === st.id; b.setAttribute("aria-checked", String(on)); b.tabIndex = on ? 0 : -1; });
    slider.value = String(st.k);
    slider.setAttribute("aria-valuetext", Mo.widthText(M, st.k));
    wout.textContent = Mo.widthText(M, st.k);
    widthRow.classList.toggle("is-off", st.id !== "gauss");
    widthRow.classList.toggle("is-beyond", st.k !== 0);
    studyB.disabled = st.k === 0;
    // the caption and the probe card are read out only after the reader has touched the plate, and not
    // while the kernels cycle on their own
    const live = st.touched && !st.cycling ? "polite" : "off";
    caption.setAttribute("aria-live", live); probeCard.setAttribute("aria-live", live);
    cycleB.textContent = st.cycling ? "Pause the kernels" : "Cycle the kernels";
    cycleB.setAttribute("aria-pressed", String(st.cycling));
    cycleB.disabled = reduceQ.matches;
    clearB.disabled = !st.probe;
    const { F, S } = fitOf(st.n, st.id, st.k);
    fml.replaceChildren(el("span", { text: Mo.formula(M, st.id) }), ...(st.id === "gauss" ? [el("br"), el("span", { text: `σ = ${Mo.fmtSigma(F.sigma)} cm` })] : []));
    ruleB.textContent = `b = ${Mo.fmtPlain(F.b, S)}`;
    ruleN.textContent = Mo.ruleText(M, st.n);
    const p = st.probe;
    if (!p) {
      big.textContent = "—";
      pick.replaceChildren();
      probeBody.textContent = coarse ? "Tap a point of the plane to read the score there." : "Click a point of the plane to read the score there.";
      probeCard.classList.add("is-empty");
    } else {
      const h = Mo.score(F, p.x);
      big.textContent = Mo.fmtScore(h, S);
      const sd = Mo.side(h), ink = inks(st.n);
      const sw = el("span", { class: "ib-swatch" });
      sw.style.background = sd === "a" ? ink.a : "transparent";
      sw.style.boxShadow = `inset 0 0 0 1.4px ${sd === "a" ? ink.a : ink.b}`;
      pick.replaceChildren(sw, el("span", { text: `the rule picks ${Mo.className(M, st.n, sd)}` }));
      probeBody.textContent = Mo.probeText(M, st.n, F, p.x, p.at).join(" ");
      probeCard.classList.remove("is-empty");
    }
    drawGauge(F, S);
    drawLegend();
  }

  // the gauge: the score's scale, three level steps each way, stippled as the plane is
  function drawGauge(F, S) {
    gauge.replaceChildren();
    if (!C) return;
    const ink = inks(st.n), x0 = 12, x1 = 308, mid = (x0 + x1) / 2, span = (x1 - x0) / 2 / 3.4;
    const gx = (v) => mid + Math.max(-3.4, Math.min(3.4, v / S.L)) * span;
    const g = el("svg:g");
    for (let i = 0; i < 260; i++) {
      const u = (i + 0.5) / 260, v = (u * 2 - 1) * 3.4;
      const d = Math.min(1, Math.abs(v) / 3);
      for (let r = 0; r < 3; r++) {
        if (ign(i, r) >= 0.78 * d ** 0.8) continue;
        g.append(el("svg:circle", { cx: (x0 + u * (x1 - x0)).toFixed(1), cy: (13 + r * 3.4 + (hash(i * 3 + r) - 0.5) * 1.6).toFixed(1), r: (0.55 + 0.5 * d).toFixed(2), fill: v >= 0 ? ink.a : ink.b }));
      }
    }
    gauge.append(g);
    gauge.append(el("svg:line", { x1: x0, x2: x1, y1: 24.5, y2: 24.5, stroke: C.rule2, "stroke-width": 1 }));
    for (let j = -3; j <= 3; j++) {
      const x = mid + j * span;
      gauge.append(el("svg:line", { x1: x, x2: x, y1: j === 0 ? 7 : 22, y2: 28, stroke: j === 0 ? C.ink : C.ink2, "stroke-width": j === 0 ? 2 : 1 }));
      gauge.append(el("svg:text", { x, y: 40, "text-anchor": "middle", class: "ib-g-lab", fill: C.ink2, text: Mo.fmtLevel(j, S) }));
    }
    const p = st.probe;
    if (p) {
      const h = Mo.score(F, p.x), x = gx(h), beyond = Math.abs(h) > 3.4 * S.L;
      gauge.append(el("svg:path", { d: `M${x} 22 l-5.5 -9 h11 z`, fill: C.accent, stroke: C.paper2, "stroke-width": 1 }));
      if (beyond) gauge.append(el("svg:path", { d: h > 0 ? `M${x + 7} 13 l6 4.5 l-6 4.5` : `M${x - 7} 13 l-6 4.5 l6 4.5`, fill: "none", stroke: C.accent, "stroke-width": 1.5 }));
    }
  }

  function drawLegend() {
    legend.replaceChildren();
    if (!C) return;
    const ink = inks(st.n), c = Mo.classes(M, st.n);
    const seed = (fill, stroke) => { const s = el("svg:svg", { viewBox: "-8 -8 16 16", class: "ib-ico", "aria-hidden": "true" }); s.append(el("svg:circle", { r: 3.6, fill: fill || "none", stroke: stroke || "none", "stroke-width": stroke ? 1.3 : 0 })); return s; };
    const key = (ico, html) => el("span", { class: "ib-key" }, [ico, el("span", { html })]);
    legend.append(key(seed(ink.a), `<i>${M.species[c.one[0]]}</i> (${c.na})`));
    legend.append(key(seed(null, ink.b), `<i>${Mo.className(M, st.n, "b", "list")}</i> (${c.nb})`));
    const rays = el("svg:svg", { viewBox: "-10 -10 20 20", class: "ib-ico", "aria-hidden": "true" });
    rays.append(el("svg:circle", { r: 3, fill: C.ink2 }));
    for (let k = 0; k < 3; k++) { const t = -Math.PI / 2 + (2 * Math.PI * k) / 3; rays.append(el("svg:line", { x1: Math.cos(t) * 4.6, y1: Math.sin(t) * 4.6, x2: Math.cos(t) * 8.4, y2: Math.sin(t) * 8.4, stroke: C.ink2, "stroke-width": 1.2, "stroke-linecap": "round" })); }
    legend.append(key(rays, "a ray for each flower at one spot"));
    const ln = (dash, w, color) => { const s = el("svg:svg", { viewBox: "0 0 26 10", class: "ib-ico ib-ico-wide", "aria-hidden": "true" }); s.append(el("svg:line", { x1: 2, x2: 24, y1: 5, y2: 5, stroke: color, "stroke-width": w, "stroke-dasharray": dash || "none", "stroke-linecap": "round" })); return s; };
    legend.append(key(ln(null, 2.2, C.ink), "boundary: score 0"));
    legend.append(key(ln("0.01 3.7", 1.7, C.ink2), "level lines"));
    if (st.id === "dot") {
      const s = el("svg:svg", { viewBox: "-11 -11 22 22", class: "ib-ico", "aria-hidden": "true" });
      s.append(el("svg:circle", { r: 5.2, fill: "none", stroke: C.ink2, "stroke-width": 1.5 }), el("svg:path", { d: "M0-10V-3M0 3V10M-10 0H-3M3 0H10", stroke: C.ink2, "stroke-width": 1.3 }));
      legend.append(key(s, "class average"));
    }
  }

  // ---------------------------------------------------------------- touch, mouse, and keys
  function probeAt(e, snapRadius) {
    const r = cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
    if (px < geo.L - 8 || px > geo.R + 8 || py < geo.T - 8 || py > geo.B + 8) return false;
    let best = null, bd = Infinity;
    for (const q of flowerSpots(st.n)) {
      const d = Math.hypot(X(q.x[0]) - px, Y(q.x[1]) - py);
      if (d < bd) { bd = d; best = q; }
    }
    if (best && bd <= snapRadius) st.probe = { x: [...best.x], at: [...best.at] };
    else {
      const c = toCm(Math.max(geo.L, Math.min(geo.R, px)), Math.max(geo.T, Math.min(geo.B, py)));
      const clamp = (v, [lo, hi]) => Math.max(lo, Math.min(hi, v));
      st.probe = { x: [Number(Mo.halfUp(clamp(c[0], Mo.WINDOW.x), 2)), Number(Mo.halfUp(clamp(c[1], Mo.WINDOW.y), 2))], at: [] };
    }
    return true;
  }
  let down = null;
  cv.addEventListener("pointerdown", (e) => {
    down = { x: e.clientX, y: e.clientY, t: performance.now(), type: e.pointerType, id: e.pointerId, moved: 0 };
    if (e.pointerType === "mouse") { cv.setPointerCapture(e.pointerId); touch(); probeAt(e, 9); update(); need = true; wake(); }
  });
  cv.addEventListener("pointermove", (e) => {
    if (!down || down.id !== e.pointerId) return;
    down.moved += Math.abs(e.clientX - down.x) + Math.abs(e.clientY - down.y);
    down.x = e.clientX; down.y = e.clientY;
    if (down.type === "mouse" && probeAt(e, 9)) { update(); need = true; wake(); }
  });
  const up = (e) => {
    if (!down || down.id !== e.pointerId) return;
    if (down.type !== "mouse" && e.type === "pointerup" && down.moved < 10 && performance.now() - down.t < 700) {
      touch(); probeAt(e, 18); update(); need = true; wake();
    }
    down = null;
  };
  cv.addEventListener("pointerup", up); cv.addEventListener("pointercancel", up);

  cv.addEventListener("keydown", (e) => {
    const k = e.key, step = e.shiftKey ? 0.25 : 0.05;
    const mv = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, step], ArrowDown: [0, -step] }[k];
    if (mv) {
      e.preventDefault(); touch();
      const p = st.probe ? st.probe.x : [4, 1.3];
      const clamp = (v, [lo, hi]) => Math.max(lo, Math.min(hi, v));
      const x = [Number(Mo.halfUp(clamp(p[0] + (st.probe ? mv[0] : 0), Mo.WINDOW.x), 2)), Number(Mo.halfUp(clamp(p[1] + (st.probe ? mv[1] : 0), Mo.WINDOW.y), 2))];
      st.probe = { x, at: Mo.flowersAt(M, st.n, x) };
    } else if (k === "," || k === ".") {
      e.preventDefault(); touch();
      const spots = flowerSpots(st.n);
      let i = st.probe ? spots.findIndex((q) => q.x[0] === st.probe.x[0] && q.x[1] === st.probe.x[1]) : -1;
      i = i < 0 ? (k === "." ? 0 : spots.length - 1) : (i + (k === "." ? 1 : -1) + spots.length) % spots.length;
      st.probe = { x: [...spots[i].x], at: [...spots[i].at] };
    } else if (k === "Escape") { st.probe = null; }
    else if (k === " ") { e.preventDefault(); cycleB.click(); return; }
    else return;
    update(); need = true; wake();
  });

  // ---------------------------------------------------------------- printings, motion, visibility
  const repaint = () => { readColors(); layers.clear(); cur = null; wipe = null; update(); need = true; wake(); };
  new MutationObserver(repaint).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", repaint);
  reduceQ.addEventListener("change", () => { if (reduceQ.matches) { st.cycling = false; wipe = null; } update(); need = true; wake(); });
  if (document.fonts) document.fonts.ready.then(() => { layers.clear(); cur = null; need = true; wake(); });
  new IntersectionObserver((es) => {
    visible = es[0].isIntersecting;
    if (visible) { resize(); if (!nextAt || nextAt < performance.now()) nextAt = performance.now() + 2600; need = true; wake(); schedule(); }
    else { clearTimeout(timer); }
  }, { rootMargin: "60px" }).observe(plane);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) { nextAt = performance.now() + 1500; wake(); } else clearTimeout(timer); });
  new ResizeObserver(() => resize()).observe(plane);

  readColors();
  describe(null);
  update();
  resize();
}

document.querySelectorAll(`figure.plate-live[data-demo="${KIND}"]`).forEach((f) => { mount(f).catch((err) => console.error(err)); });
