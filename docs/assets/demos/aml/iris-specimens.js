// The Iris specimen case: the 150 flowers in a glass case on three of their four measurements, in
// centimeters, each a pressed-seed disc in its species' ink. The case has centimeter rulers along
// three edges, the cloud prints a faint shadow of itself on the floor (the flowers on the two floor
// measurements alone), and the chosen problem's two class averages are engraved stars joined by a
// fine line labeled with their distance in all four measurements. A picked flower drops a thin line
// to its shadow.
//
// Everything here is line work and flat discs, so the case is drawn on a 2D canvas with its own
// perspective projection: crisp engraved lines and type at any pixel ratio, and no WebGL to fall
// back from. Values and sentences come from iris-specimens-model.js; colors from the page's inks,
// read again when the printing changes.
const KIND = "iris-specimens";
// The page loads this file with a version query; the model and the stylesheet get the same one, so
// a new build never runs a new view on an old model.
const V = new URL(import.meta.url).search;
let Mo = null;
const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
const usDate = (iso) => { const [y, m, d] = iso.split("-").map(Number); return `${MONTHS[m - 1]} ${d}, ${y}`; };

const YAW0 = 0.55, PITCH0 = 0.42, TURN = 0.11; // radians; radians per second when turning on its own
const PITCH_MIN = 0.06, PITCH_MAX = 1.5, ZOOM_MIN = 0.6, ZOOM_MAX = 4;
const TAU = 0.14; // seconds: how quickly things glide to a new place

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

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const lerp = (a, b, t) => a + (b - a) * t;

/** The small star icon used in the legend and the readout. */
function starIcon(cls) {
  const s = el("svg:svg", { viewBox: "-12 -12 24 24", class: `is-ico ${cls}`, "aria-hidden": "true" });
  s.append(el("svg:circle", { r: 5.2, class: "is-ico-fill" }), el("svg:circle", { r: 7.4, class: "is-ico-line" }),
    el("svg:path", { d: "M0-11V-4M0 4V11M-11 0H-4M4 0H11", class: "is-ico-line" }));
  return s;
}

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
  const st = {
    n: 3, axes: [...Mo.DEFAULT_AXES], yaw: YAW0, pitch: PITCH0, zoom: 1,
    turning: !reduceQ.matches, touched: false, pick: null, hover: null, above: null,
  };

  // ---------------------------------------------------------------- controls
  const probs = el("div", { class: "is-problems", role: "radiogroup", "aria-label": "Problem" });
  for (const P of M.problems) {
    const b = el("button", { type: "button", role: "radio", "data-n": P.n }, [el("span", { class: "is-pn", text: String(P.n) }), el("span", { text: Mo.buttonLabel(M, P.n) })]);
    b.addEventListener("click", () => { touch(); setProblem(P.n); });
    probs.append(b);
  }
  radioKeys(probs);
  const selects = [];
  const axisBox = el("div", { class: "is-axes" });
  ["Across", "Depth", "Up"].forEach((word, k) => {
    const s = el("select", { "aria-label": `${["First floor", "Second floor", "Upright"][k]} measurement` });
    M.features.forEach((f, j) => s.append(el("option", { value: String(j), text: f.name })));
    s.addEventListener("change", () => { touch(); setAxis(k, Number(s.value)); });
    selects.push(s);
    axisBox.append(el("label", { class: "is-axis" }, [el("span", { class: "is-k", text: word }), s]));
  });
  const controls = el("div", { class: "is-controls" }, [
    el("div", { class: "is-field" }, [el("span", { class: "is-k", text: "Problem", "aria-hidden": "true" }), probs]),
    el("div", { class: "is-field" }, [el("span", { class: "is-k", text: "Measurements drawn", "aria-hidden": "true" }), axisBox]),
  ]);

  // ---------------------------------------------------------------- the case
  const cv = el("canvas", { class: "is-cv", tabindex: "0", role: "img" });
  const turnB = el("button", { type: "button", class: "is-tool" });
  const aboveB = el("button", { type: "button", class: "is-tool", "aria-pressed": "false", text: "From above" });
  const resetB = el("button", { type: "button", class: "is-tool", "aria-label": "Reset the view", text: "Reset" });
  const zoomOut = el("button", { type: "button", class: "is-tool is-zoom", "aria-label": "Zoom out", text: "−" });
  const zoomIn = el("button", { type: "button", class: "is-tool is-zoom", "aria-label": "Zoom in", text: "+" });
  const mac = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
  const hint = el("p", { class: "is-hint", text: coarse ? "Drag sideways to turn, pinch to zoom, and tap a flower or a star to read it." : `Drag to turn, ${mac ? "⌘" : "Ctrl"}-scroll or pinch to zoom, and click a flower or a star to read it. With the case focused, arrow keys turn it, comma and period step through the flowers, and A and Shift+A step through the averages.` });
  const tools = el("div", { class: "is-tools" }, [turnB, aboveB, resetB, el("span", { class: "is-zooms" }, [zoomOut, zoomIn]), hint]);
  const wheelHint = el("p", { class: "is-wheel", "aria-hidden": "true", text: `Hold ${mac ? "⌘" : "Ctrl"} and scroll to zoom the case` });
  const caseBox = el("div", { class: "is-case" }, [cv, wheelHint]);
  for (const [b, f] of [[zoomOut, 1 / 1.25], [zoomIn, 1.25]]) b.addEventListener("click", () => { touch(); st.zoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, st.zoom * f)); wake(); });

  // ---------------------------------------------------------------- readouts
  const big = el("span", { class: "is-num" });
  const small = el("p", { class: "is-small" });
  const scale = el("svg:svg", { class: "is-scale", viewBox: "0 0 320 50", role: "img" });
  const avgHead = el("p", { class: "is-k" });
  const avgCard = el("section", { class: "is-avg" }, [
    avgHead,
    el("p", { class: "is-big" }, [big, el("span", { class: "is-unit", text: " cm apart in all four measurements" })]),
    scale, small,
  ]);
  const cardBody = el("p", { class: "is-card-body" });
  const card = el("section", { class: "is-card", "aria-live": "polite" }, [el("p", { class: "is-k", text: "Specimen" }), cardBody]);
  const caption = el("p", { class: "is-caption", "aria-live": "off" });
  const legend = el("div", { class: "is-legend" });
  M.species.forEach((name, s) => legend.append(el("span", { class: "is-key" }, [el("i", { class: `is-seed is-s${s}`, "aria-hidden": "true" }), el("span", { html: `<i>Iris ${name}</i> (${M.speciesN[s]})` })])));
  legend.append(el("span", { class: "is-key" }, [starIcon("is-ico-avg"), el("span", { text: "class average" })]));
  legend.append(el("span", { class: "is-key" }, [(() => { const s = el("svg:svg", { viewBox: "-12 -12 24 24", class: "is-ico", "aria-hidden": "true" }); s.append(el("svg:circle", { r: 5.5, class: "is-ico-ring" })); return s; })(), el("span", { text: "a pooled species’ own average" })]));
  const note = el("p", { class: "is-note", text: "Each average is over all of that class’s flowers among the 150; the classifier itself takes its averages from each split’s training flowers. Distances are straight-line distances, computed here from the measurements. One centimeter is the same length along every edge of the case." });
  const checked = el("p", { class: "is-checked" });
  fetch(fig.dataset.src.replace(/\.json$/, ".checks.json")).then((r) => (r.ok ? r.json() : null)).then((c) => { if (c && c.line) checked.textContent = c.line; }).catch(() => {});
  root.append(controls, caseBox, tools, el("div", { class: "is-read" }, [avgCard, card]), caption, legend, note, checked);

  // the distance scale: 0 to 4.5 cm, a pin for each problem
  const X0 = 14, X1 = 306, SMAX = 4.5;
  const sx = (d) => X0 + (d / SMAX) * (X1 - X0);
  scale.append(el("svg:line", { x1: X0, x2: X1, y1: 30, y2: 30, class: "is-sc-base" }));
  for (let t = 0; t <= SMAX * 10 + 1e-9; t++) {
    const h = t % 10 === 0 ? 8 : t % 5 === 0 ? 5.5 : 3;
    scale.append(el("svg:line", { x1: sx(t / 10), x2: sx(t / 10), y1: 30, y2: 30 + h, class: t % 10 === 0 ? "is-sc-tick is-sc-cm" : "is-sc-tick" }));
    if (t % 10 === 0) scale.append(el("svg:text", { x: sx(t / 10), y: 48, class: "is-sc-lab", text: String(t / 10) }));
  }
  const pins = {};
  for (const P of M.problems) {
    const g = el("svg:g", { class: "is-pin", transform: `translate(${sx(P.d4)},0)` });
    g.append(el("svg:line", { x1: 0, x2: 0, y1: 19, y2: 30, class: "is-pin-stem" }), el("svg:circle", { cx: 0, cy: 11, r: 8, class: "is-pin-head" }),
      el("svg:text", { x: 0, y: 14.6, class: "is-pin-n", text: String(P.n) }));
    scale.append(g);
    pins[P.n] = g;
  }
  scale.setAttribute("aria-label", Mo.scaleLabel(M));

  // ---------------------------------------------------------------- the bodies that move
  // Each flower, each problem's two averages, and each species' own average have a place in the
  // case (cm from its center); when the problem or the measurements change, they glide to the new one.
  const body = (v) => ({ cur: Mo.toBox(v, st.axes), to: Mo.toBox(v, st.axes) });
  const flowers = M.flowers.map((f) => body(f.v));
  const P0 = Mo.problem(M, st.n);
  const starA = body(P0.a), starB = body(P0.b);
  const own = M.speciesAvg.map((v) => ({ ...body(v), alpha: 0 }));
  const half = { cur: Mo.halfSizes(st.axes), to: Mo.halfSizes(st.axes) };
  const view = { yaw: st.yaw, pitch: st.pitch, zoom: st.zoom }; // the drawn view, gliding to st's
  let rulerAlpha = 1, excludedAlpha = new Array(M.species.length).fill(1);

  function targets() {
    const P = Mo.problem(M, st.n);
    flowers.forEach((b, i) => { b.to = Mo.toBox(M.flowers[i].v, st.axes); });
    starA.to = Mo.toBox(P.a, st.axes); starB.to = Mo.toBox(P.b, st.axes);
    own.forEach((b, s) => { b.to = Mo.toBox(M.speciesAvg[s], st.axes); });
    half.to = Mo.halfSizes(st.axes);
  }

  // ---------------------------------------------------------------- state changes
  function setProblem(n) {
    st.n = n;
    if (st.pick && st.pick.type !== "flower") st.pick = null;
    targets(); update(); wake();
  }
  function setAxis(k, j) {
    const a = [...st.axes], was = a.indexOf(j);
    if (was === k) return;
    if (was >= 0) a[was] = a[k]; // the measurement already drawn on another axis trades places
    a[k] = j;
    st.axes = a;
    // which flowers share the picked one's spot depends on the three measurements drawn
    if (st.pick && st.pick.type === "flower") st.pick.stack = stackOf(st.pick.i);
    targets(); update(); wake();
  }
  // With reduced motion, the case starts still and never turns on its own; the reader may still turn it.
  function setTurning(on) { st.turning = on; update(); wake(); }
  function touch() { if (!st.touched) { st.touched = true; caption.setAttribute("aria-live", "polite"); } if (st.turning) setTurning(false); }

  turnB.addEventListener("click", () => { st.touched = true; setTurning(!st.turning); });
  aboveB.addEventListener("click", () => {
    touch();
    if (st.above) { Object.assign(st, st.above); st.above = null; }
    else { st.above = { yaw: st.yaw, pitch: st.pitch }; st.yaw = Math.round(st.yaw / (2 * Math.PI)) * 2 * Math.PI; st.pitch = PITCH_MAX; }
    update(); wake();
  });
  resetB.addEventListener("click", () => {
    touch();
    Object.assign(st, { yaw: YAW0 + Math.round((st.yaw - YAW0) / (2 * Math.PI)) * 2 * Math.PI, pitch: PITCH0, zoom: 1, above: null });
    update(); wake();
  });

  // ---------------------------------------------------------------- words and readouts
  function update() {
    probs.querySelectorAll("button").forEach((b) => { const on = Number(b.dataset.n) === st.n; b.setAttribute("aria-checked", String(on)); b.tabIndex = on ? 0 : -1; });
    selects.forEach((s, k) => { s.value = String(st.axes[k]); });
    turnB.textContent = st.turning ? "Pause turning" : "Turn";
    turnB.setAttribute("aria-pressed", String(st.turning));
    aboveB.setAttribute("aria-pressed", String(Boolean(st.above)));
    const R = Mo.readout(M, st.n, st.axes);
    avgHead.textContent = Mo.avgHeading(M, st.n);
    big.textContent = R.s4;
    small.textContent = `${R.s3} cm apart on the three measurements drawn`;
    for (const [n, g] of Object.entries(pins)) g.classList.toggle("is-on", Number(n) === st.n);
    caption.textContent = Mo.problemText(M, st.n, st.axes).join(" ");
    cv.setAttribute("aria-label", Mo.summary(M, st.n, st.axes));
    const p = st.pick;
    let text;
    if (!p) text = [coarse ? "Tap a flower or a star to read it." : "Click a flower or a star to read it."];
    else if (p.type === "flower") text = Mo.flowerText(M, st.n, p.i, st.axes, p.stack || []);
    else if (p.type === "avg") text = Mo.averageText(M, st.n, p.side);
    else text = Mo.pooledText(M, st.n, p.s);
    cardBody.textContent = text.join(" ");
    card.classList.toggle("is-empty", !p);
  }

  // ---------------------------------------------------------------- colors, read from the page
  let C = null;
  function readColors() {
    const cs = getComputedStyle(fig), g = (n) => cs.getPropertyValue(n).trim();
    const paper = g("--paper");
    const rgb = (() => { const c = document.createElement("canvas").getContext("2d"); c.fillStyle = paper; const h = c.fillStyle; return h.startsWith("#") ? [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)) : [255, 255, 255]; })();
    C = {
      ink: g("--ink"), ink2: g("--ink-2"), paper, paper2: g("--paper-2"), rule: g("--rule"), rule2: g("--rule-2"),
      sp: [g("--iris-setosa"), g("--iris-versicolor"), g("--iris-virginica")], type: g("--type") || "monospace", serif: g("--serif") || "serif",
      dark: (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) < 128,
    };
  }
  readColors();
  const repaint = () => { readColors(); wake(); };
  new MutationObserver(repaint).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", repaint);
  reduceQ.addEventListener("change", () => { if (reduceQ.matches) st.turning = false; update(); wake(); });
  if (document.fonts) document.fonts.ready.then(() => wake());

  // ---------------------------------------------------------------- size
  let W = 0, H = 0, dpr = 1;
  const ctx = cv.getContext("2d");
  function resize() {
    const w = caseBox.clientWidth;
    if (!w) return;
    const h = w < 560 ? Math.round(Math.min(400, Math.max(340, w * 1.12))) : Math.round(Math.min(560, Math.max(460, w * 0.66)));
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (w !== W || h !== H || cv.width !== Math.round(w * dpr)) {
      W = w; H = h;
      cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
      cv.style.height = `${h}px`;
      wake();
    }
  }
  new ResizeObserver(resize).observe(caseBox);

  // ---------------------------------------------------------------- projection
  let cam = null;
  function camera() {
    const [a, b, c] = half.cur;
    const R = Math.hypot(a, b, c), D = 5.2 * R;
    const cp = Math.cos(view.pitch), sp = Math.sin(view.pitch), cy = Math.cos(view.yaw), sy = Math.sin(view.yaw);
    const pos = [D * cp * sy, D * sp, D * cp * cy];
    const f = [-cp * sy, -sp, -cp * cy], r = [cy, 0, -sy], u = [-sy * sp, cp, -cy * sp];
    const small = W < 560;
    const mx = small ? 20 : 64, myTop = small ? 26 : 34, myBot = small ? 40 : 48;
    const ac = Math.hypot(a, c);
    // px per cm at the center: the box fits at nearly any turn, at this tilt, with room for the rulers
    let k = Math.min((W / 2 - mx) / (0.92 * ac), (H - myTop - myBot) / 2 / (b * cp + 0.9 * ac * sp));
    // seen from nearly above, fit the box as it actually lies, so the floor's print fills the case
    const wAbove = Math.max(0, Math.min(1, (view.pitch - 1.0) / 0.45));
    if (wAbove > 0) {
      const ay = Math.abs(cy), as = Math.abs(sy);
      const kPlan = Math.min((W / 2 - mx) / (a * ay + c * as), (H - myTop - myBot) / 2 / (b * cp + (a * as + c * ay) * sp));
      k = lerp(k, Math.min(kPlan, 2.4 * k), wAbove);
    }
    k *= (D - 0.3 * R) / D;
    k *= view.zoom;
    const F = k * D;
    const cx = W / 2, cyS = myTop + (H - myTop - myBot) / 2;
    return { pos, f, r, u, F, k, cx, cy: cyS, D, R };
  }
  function project(p) {
    const v = [p[0] - cam.pos[0], p[1] - cam.pos[1], p[2] - cam.pos[2]];
    const z = dot(v, cam.f);
    return { x: cam.cx + (cam.F * dot(v, cam.r)) / z, y: cam.cy - (cam.F * dot(v, cam.u)) / z, z };
  }

  // ---------------------------------------------------------------- drawing helpers
  function line(p, q, color, w, alpha = 1, dash = null) {
    ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = w;
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
    if (dash) ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }
  function text(s, x, y, { font, color, align = "center", base = "middle", halo = true, alpha = 1 }) {
    ctx.font = font; ctx.textAlign = align; ctx.textBaseline = base; ctx.globalAlpha = alpha;
    if (halo) { ctx.lineJoin = "round"; ctx.lineWidth = 3.5; ctx.strokeStyle = C.paper2; ctx.strokeText(s, x, y); }
    ctx.fillStyle = color; ctx.fillText(s, x, y);
    ctx.globalAlpha = 1;
  }
  const clampX = (x, w) => Math.max(4 + w / 2, Math.min(W - 4 - w / 2, x));
  const clampY = (y) => Math.max(9, Math.min(H - 9, y));

  // A centimeter ruler along one edge of the case: a strip just outside the edge, a tick every
  // millimeter where there is room, every half centimeter, and a numbered tick every centimeter.
  function ruler(j, at, out, title, alpha) {
    const [lo, hi] = Mo.SPANS[j];
    const P = (v, o) => project([at(v)[0] + out[0] * o, at(v)[1] + out[1] * o, at(v)[2] + out[2] * o]);
    const a = P(lo, 0), b = P(hi, 0);
    const len = Math.hypot(b.x - a.x, b.y - a.y);
    if (len < 24) return;
    const perMm = len / ((hi - lo) * 10);
    ctx.globalAlpha = alpha;
    // the strip
    ctx.fillStyle = C.paper; ctx.globalAlpha = alpha * (C.dark ? 0.5 : 0.75);
    const s0 = P(lo, 0.26), s1 = P(hi, 0.26);
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.lineTo(s1.x, s1.y); ctx.lineTo(s0.x, s0.y); ctx.closePath(); ctx.fill();
    line(s0, s1, C.ink2, 0.8, alpha * 0.8);
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = C.ink; ctx.lineWidth = 0.8;
    ctx.beginPath();
    const n = Math.round((hi - lo) * 10);
    for (let t = 0; t <= n; t++) {
      const cm = t % 10 === 0, halfCm = t % 5 === 0;
      if (!cm && !halfCm && perMm < 2.6) continue;
      if (!cm && perMm * 5 < 3) continue;
      const v = lo + t / 10, p = P(v, 0), q = P(v, cm ? 0.24 : halfCm ? 0.16 : 0.09);
      ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
    const every = perMm * 10 < 15 ? 2 : 1;
    const font = `${W < 560 ? 10 : 11}px ${C.type}`;
    // numbers and the name sit a fixed number of pixels beyond the strip, outward on screen
    const mid = (lo + hi) / 2, m0 = P(mid, 0.26), m1 = P(mid, 0.6);
    let ox = m1.x - m0.x, oy = m1.y - m0.y;
    const ol = Math.hypot(ox, oy) || 1; ox /= ol; oy /= ol;
    const reach = (w, h, gap) => gap + Math.abs(ox) * w / 2 + Math.abs(oy) * h / 2;
    ctx.font = font;
    for (let v = Math.ceil(lo); v <= hi + 1e-9; v += every) {
      const q = P(v, 0.26), w = ctx.measureText(String(v)).width, r = reach(w, 10, 5);
      text(String(v), clampX(q.x + ox * r, w), clampY(q.y + oy * r), { font, color: C.ink, alpha });
    }
    ctx.font = `italic ${W < 560 ? 11.5 : 12.5}px ${C.serif}`;
    const tw = ctx.measureText(title.s).width, r = reach(14, 10, 5) + reach(tw, 13, 6) - 5;
    text(title.s, clampX(m0.x + ox * r, tw), clampY(m0.y + oy * r), { font: ctx.font, color: C.ink2, alpha });
  }

  // An engraved crosshair star at a class average, its center filled with the class's inks.
  function star(p, inks, scaleK, picked) {
    const R = 7.2 * scaleK, r0 = 4.2 * scaleK, r1 = 13.5 * scaleK;
    ctx.lineCap = "round";
    const strokeAll = (color, w) => {
      ctx.strokeStyle = color; ctx.lineWidth = w;
      ctx.beginPath(); ctx.arc(p.x, p.y, R, 0, 2 * Math.PI); ctx.stroke();
      ctx.beginPath();
      for (const [dx, dy] of [[0, -1], [0, 1], [-1, 0], [1, 0]]) { ctx.moveTo(p.x + dx * r0, p.y + dy * r0); ctx.lineTo(p.x + dx * r1, p.y + dy * r1); }
      for (const [dx, dy] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) { const s = Math.SQRT1_2; ctx.moveTo(p.x + dx * s * (R + 1.5), p.y + dy * s * (R + 1.5)); ctx.lineTo(p.x + dx * s * (R + 4.5 * scaleK), p.y + dy * s * (R + 4.5 * scaleK)); }
      ctx.stroke();
    };
    strokeAll(C.paper2, 4.2);
    // the seal: one ink, or two halves for a class that pools two species
    const rr = R - 1.6;
    if (inks.length === 1) { ctx.fillStyle = inks[0]; ctx.beginPath(); ctx.arc(p.x, p.y, rr, 0, 2 * Math.PI); ctx.fill(); }
    else {
      ctx.fillStyle = inks[0]; ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.arc(p.x, p.y, rr, Math.PI / 2, (3 * Math.PI) / 2); ctx.fill();
      ctx.fillStyle = inks[1]; ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.arc(p.x, p.y, rr, -Math.PI / 2, Math.PI / 2); ctx.fill();
    }
    strokeAll(C.ink, picked ? 1.9 : 1.3);
    ctx.fillStyle = C.ink; ctx.beginPath(); ctx.arc(p.x, p.y, 1.3, 0, 2 * Math.PI); ctx.fill();
    ctx.lineCap = "butt";
  }

  // A specimen tag pinned to an average: a leader up and to one side, then a small ruled card that
  // reads AVERAGE over the class's name.
  function specimenTag(p, name, side, small) {
    const L = small ? 26 : 36, lean = small ? 12 : 18;
    let ex = p.x + side * lean, ey = p.y - L;
    const nameFont = `italic ${small ? 11.5 : 13}px ${C.serif}`, capFont = `${small ? 8 : 9}px ${C.type}`;
    ctx.font = nameFont; const wn = ctx.measureText(name).width;
    ctx.font = capFont; const wc = ctx.measureText("AVERAGE").width;
    const w = Math.max(wn, wc) + (small ? 10 : 12), h = small ? 28 : 32;
    if (ey - h < 2) ey = p.y + L + h; // no room above: hang it below
    let x0 = side > 0 ? ex : ex - w;
    x0 = Math.max(2, Math.min(W - w - 2, x0));
    const y0 = ey - h;
    line({ x: p.x + side * 5, y: p.y + (ey < p.y ? -9 : 9) }, { x: side > 0 ? x0 + 6 : x0 + w - 6, y: ey < p.y ? ey : y0 }, C.ink, 0.9, 0.85);
    ctx.fillStyle = C.paper; ctx.globalAlpha = 0.94; ctx.fillRect(x0, y0, w, h); ctx.globalAlpha = 1;
    ctx.strokeStyle = C.ink; ctx.lineWidth = 0.9; ctx.strokeRect(x0 + 0.5, y0 + 0.5, w - 1, h - 1);
    const pad = small ? 5 : 6;
    text("AVERAGE", x0 + pad, y0 + (small ? 8 : 9), { font: capFont, color: C.ink2, align: "left", halo: false });
    text(name, x0 + pad, y0 + h - (small ? 8 : 9), { font: nameFont, color: C.ink, align: "left", halo: false });
  }

  // ---------------------------------------------------------------- one frame
  let shot = { flowers: [], stars: [] }; // where things were drawn, for picking
  function draw() {
    if (!W || !C) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    cam = camera();
    const [a, b, c] = half.cur;
    const small = W < 560;
    const P = Mo.problem(M, st.n);
    const corner = (sx, sy, sz) => [sx * a, sy * b, sz * c];
    const corners = [];
    for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) corners.push({ s: [sx, sy, sz], p: project(corner(sx, sy, sz)) });
    const nearest = corners.reduce((m, q) => (q.p.z < m.p.z ? q : m));
    const floorC = corners.filter((q) => q.s[1] === -1);
    const nearFloor = floorC.reduce((m, q) => (q.p.z < m.p.z ? q : m));
    const [nx, , nz] = nearFloor.s;

    // the floor: a mounting card, with a faint centimeter grid
    ctx.fillStyle = C.paper; ctx.globalAlpha = C.dark ? 0.55 : 0.7;
    const fc = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([sx, sz]) => project(corner(sx, -1, sz)));
    ctx.beginPath(); fc.forEach((q, i) => (i ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y))); ctx.closePath(); ctx.fill();
    ctx.globalAlpha = 1;
    const [jx, jz] = st.axes;
    ctx.strokeStyle = C.rule; ctx.lineWidth = 0.7; ctx.beginPath();
    const midOf = (j) => (Mo.SPANS[j][0] + Mo.SPANS[j][1]) / 2;
    if (Math.abs(a - half.to[0]) < 0.02 && Math.abs(c - half.to[2]) < 0.02) {
      for (let v = Math.ceil(Mo.SPANS[jx][0]); v <= Mo.SPANS[jx][1]; v++) { const x = v - midOf(jx), p = project([x, -b, -c]), q = project([x, -b, c]); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); }
      for (let v = Math.ceil(Mo.SPANS[jz][0]); v <= Mo.SPANS[jz][1]; v++) { const z = -(v - midOf(jz)), p = project([-a, -b, z]), q = project([a, -b, z]); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); }
    }
    ctx.stroke();

    // the shadow print: each flower on the floor, as a flat disc lying in the floor's plane
    const ex = project([1, -b, 0]), ez = project([0, -b, 1]), e0 = project([0, -b, 0]);
    const rs = ((small ? 3.1 : 3.6) * Math.sqrt(view.zoom)) / cam.k; // the discs' own size, in cm on the floor
    for (let s = 0; s < M.species.length; s++) {
      ctx.beginPath();
      flowers.forEach((f, i) => {
        if (M.flowers[i].s !== s) return;
        const q = project([f.cur[0], -b, f.cur[2]]);
        // the floor's local axes at this point, from the projected unit vectors (perspective varies slowly)
        const sc = e0.z / q.z;
        ctx.setTransform(dpr * (ex.x - e0.x) * sc, dpr * (ex.y - e0.y) * sc, dpr * (ez.x - e0.x) * sc, dpr * (ez.y - e0.y) * sc, dpr * q.x, dpr * q.y);
        ctx.moveTo(rs, 0); ctx.arc(0, 0, rs, 0, 2 * Math.PI);
      });
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = C.sp[s]; ctx.globalAlpha = (C.dark ? 0.22 : 0.2) * excludedAlpha[s];
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    // the two averages' shadows: small crosses on the floor
    for (const sb of [starA, starB]) {
      const q = project([sb.cur[0], -b, sb.cur[2]]);
      line({ x: q.x - 4, y: q.y }, { x: q.x + 4, y: q.y }, C.ink, 1, 0.45);
      line({ x: q.x, y: q.y - 4 }, { x: q.x, y: q.y + 4 }, C.ink, 1, 0.45);
    }

    // the case's edges: the three at the corner nearest the eye are drawn over the flowers
    const edges = [];
    corners.forEach((q, i) => corners.forEach((r2, j) => { if (j > i && q.s.filter((v, k) => v !== r2.s[k]).length === 1) edges.push([q, r2]); }));
    const isFront = ([q, r2]) => q === nearest || r2 === nearest;
    for (const e of edges) if (!isFront(e)) {
      const floor = e[0].s[1] === -1 && e[1].s[1] === -1;
      line(e[0].p, e[1].p, C.ink, floor ? 1.2 : 0.9, floor ? 0.8 : 0.5);
    }

    // the rulers: across and into the floor along the floor edges nearest the eye, and up at a side corner
    const settled = Math.abs(a - half.to[0]) + Math.abs(b - half.to[1]) + Math.abs(c - half.to[2]) < 0.03;
    const ra = rulerAlpha;
    if (ra > 0.02) {
      const jy = st.axes[2], name = (j) => `${M.features[j].name}, cm`;
      ruler(jx, (v) => [v - midOf(jx), -b, nz * c], [0, 0, nz], { s: name(jx) }, ra);
      ruler(jz, (v) => [nx * a, -b, -(v - midOf(jz))], [nx, 0, 0], { s: name(jz) }, ra);
      // the upright ruler stands at whichever neighbor of the nearest floor corner sits farther left
      const cA = project(corner(-nx, -1, nz)), cB = project(corner(nx, -1, -nz));
      const useA = cA.x < cB.x;
      const [ux, uz] = useA ? [-nx, nz] : [nx, -nz];
      const out = useA ? [-nx, 0, 0] : [0, 0, -nz];
      // seen from nearly above, the upright ruler has no length left to read: it fades out
      uprightRuler(jy, (v) => [ux * a, v - midOf(jy), uz * c], out, ra * Math.max(0, Math.min(1, (1.3 - view.pitch) / 0.2)));
    }

    // the flowers and the averages, far to near
    const items = [];
    const R0 = (small ? 3.1 : 3.6) * Math.sqrt(view.zoom);
    const pick = st.pick;
    shot = { flowers: [], stars: [] };
    flowers.forEach((f, i) => {
      const q = project(f.cur);
      const r = Math.max(2.2, Math.min(9, (R0 * cam.D) / q.z * (cam.D - cam.R) / cam.D * 1.08));
      shot.flowers[i] = { x: q.x, y: q.y, r };
      items.push({ z: q.z, draw: () => seed(q, r, M.flowers[i].s, pick && pick.type === "flower" && pick.i === i, st.hover === i) });
    });
    const pA = project(starA.cur), pB = project(starB.cur);
    const pooled = P.other.length === 2 ? P.other : [];
    own.forEach((o, s) => {
      const target = pooled.includes(s) ? 1 : 0;
      o.alpha = reduceQ.matches ? target : o.alpha + (target - o.alpha) * 0.18;
      if (o.alpha < 0.02) return;
      const q = project(o.cur);
      shot.stars.push({ x: q.x, y: q.y, type: "pooled", s, on: pooled.includes(s) });
      items.push({ z: q.z, draw: () => ring(q, s, o.alpha, pick && pick.type === "pooled" && pick.s === s) });
    });
    items.sort((p, q) => q.z - p.z);

    // the faint line through the pooled species' own averages: the other class's average sits on it
    if (pooled.length === 2 && own[pooled[0]].alpha > 0.02) {
      const q0 = project(own[pooled[0]].cur), q1 = project(own[pooled[1]].cur);
      line(q0, q1, C.ink2, 1, 0.55 * own[pooled[0]].alpha, [2, 4]);
    }
    for (const it of items) it.draw();

    // the drop line from a picked flower to its shadow
    if (pick && pick.type === "flower") {
      const f = flowers[pick.i], q = project(f.cur), g = project([f.cur[0], -b, f.cur[2]]);
      line(q, g, C.ink, 1.1, 0.9, [3, 3]);
      ctx.save();
      const sc = e0.z / g.z;
      ctx.setTransform(dpr * (ex.x - e0.x) * sc, dpr * (ex.y - e0.y) * sc, dpr * (ez.x - e0.x) * sc, dpr * (ez.y - e0.y) * sc, dpr * g.x, dpr * g.y);
      ctx.beginPath(); ctx.arc(0, 0, rs * 1.9, 0, 2 * Math.PI); ctx.restore();
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1.1; ctx.stroke();
      const sh = shot.flowers[pick.i];
      text(`No. ${pick.i + 1}`, sh.x + sh.r + 6, sh.y - sh.r - 6, { font: `${small ? 10.5 : 11.5}px ${C.type}`, color: C.ink, align: "left" });
    }

    // the line joining the two averages, with its distance in all four measurements
    line(pA, pB, C.paper2, 4, 0.8);
    line(pA, pB, C.ink, 1.25, 1);
    star(pB, P.other.map((s) => C.sp[s]), small ? 0.9 : 1, pick && pick.type === "avg" && pick.side === "b");
    star(pA, P.one.map((s) => C.sp[s]), small ? 0.9 : 1, pick && pick.type === "avg" && pick.side === "a");
    shot.stars.push({ x: pA.x, y: pA.y, type: "avg", side: "a" }, { x: pB.x, y: pB.y, type: "avg", side: "b" });
    // tags: each average gets a pinned label above it, leaning away from the other one
    const leftFirst = pA.x <= pB.x;
    specimenTag(pA, Mo.starLabel(M, P.one), leftFirst ? -1 : 1, small);
    specimenTag(pB, Mo.starLabel(M, P.other), leftFirst ? 1 : -1, small);
    // the distance, under the middle of the line
    let dx = pB.x - pA.x, dy = pB.y - pA.y;
    const dl = Math.hypot(dx, dy) || 1; dx /= dl; dy /= dl;
    const lab = Mo.lineLabel(M, st.n), lfont = `${small ? 11 : 12}px ${C.type}`;
    ctx.font = lfont;
    const lw = ctx.measureText(lab).width;
    const mxp = (pA.x + pB.x) / 2, myp = (pA.y + pB.y) / 2;
    let nxp = -dy, nyp = dx;
    if (nyp < 0) { nxp = -nxp; nyp = -nyp; } // below the line
    const push = 14 + Math.abs(nxp) * (lw / 2) + (dl < 70 ? 6 : 0);
    const lx = clampX(mxp + nxp * push, lw + 10), ly = Math.min(H - 10, myp + nyp * push);
    ctx.fillStyle = C.paper2; ctx.globalAlpha = 0.9; ctx.fillRect(lx - lw / 2 - 5, ly - 8.5, lw + 10, 17); ctx.globalAlpha = 1;
    ctx.strokeStyle = C.ink; ctx.lineWidth = 0.8; ctx.strokeRect(lx - lw / 2 - 5 + 0.5, ly - 8.5 + 0.5, lw + 9, 16);
    text(lab, lx, ly + 0.5, { font: lfont, color: C.ink, halo: false });

    // the front edges, over everything
    for (const e of edges) if (isFront(e)) line(e[0].p, e[1].p, C.ink, 0.9, 0.5);
    rulerTarget = settled ? 1 : 0;
  }
  let rulerTarget = 1;

  // The upright ruler: the same strip and ticks, its numbers beside it, its name above its top end.
  function uprightRuler(j, at, out, alpha) {
    if (alpha < 0.02) return;
    const [lo, hi] = Mo.SPANS[j];
    const P = (v, o) => project([at(v)[0] + out[0] * o, at(v)[1] + out[1] * o, at(v)[2] + out[2] * o]);
    const a = P(lo, 0), b = P(hi, 0);
    const len = Math.hypot(b.x - a.x, b.y - a.y);
    if (len < 24) return;
    const perMm = len / ((hi - lo) * 10);
    // the strip and its ticks
    ctx.fillStyle = C.paper; ctx.globalAlpha = alpha * (C.dark ? 0.5 : 0.75);
    const s0 = P(lo, 0.26), s1 = P(hi, 0.26);
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.lineTo(s1.x, s1.y); ctx.lineTo(s0.x, s0.y); ctx.closePath(); ctx.fill();
    line(s0, s1, C.ink2, 0.8, alpha * 0.8);
    line(a, b, C.ink, 1.1, alpha * 0.8);
    ctx.globalAlpha = alpha; ctx.strokeStyle = C.ink; ctx.lineWidth = 0.8; ctx.beginPath();
    const n = Math.round((hi - lo) * 10);
    for (let t = 0; t <= n; t++) {
      const cm = t % 10 === 0, halfCm = t % 5 === 0;
      if (!cm && !halfCm && perMm < 2.6) continue;
      const v = lo + t / 10, p = P(v, 0), q = P(v, cm ? 0.24 : halfCm ? 0.16 : 0.09);
      ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y);
    }
    ctx.stroke(); ctx.globalAlpha = 1;
    const every = perMm * 10 < 15 ? 2 : 1;
    const font = `${W < 560 ? 10 : 11}px ${C.type}`;
    // numbers on the outer side of the strip, a fixed number of pixels out on screen
    const o1 = P(lo, 0.6), o0 = P(lo, 0.26);
    let ox = o1.x - o0.x, oy = o1.y - o0.y;
    const ol = Math.hypot(ox, oy) || 1; ox /= ol; oy /= ol;
    ctx.font = font;
    for (let v = Math.ceil(lo); v <= hi + 1e-9; v += every) {
      const q = P(v, 0.26), w = ctx.measureText(String(v)).width, r = 5 + Math.abs(ox) * w / 2 + Math.abs(oy) * 5;
      text(String(v), clampX(q.x + ox * r, w), clampY(q.y + oy * r), { font, color: C.ink, alpha, align: "center" });
    }
    const tq = P(hi, 0.13);
    ctx.font = `italic ${W < 560 ? 11.5 : 12.5}px ${C.serif}`;
    const s = `${M.features[j].name}, cm`, tw = ctx.measureText(s).width;
    text(s, clampX(tq.x, tw), tq.y - 16, { font: ctx.font, color: C.ink2, alpha });
  }

  function seed(q, r, s, picked, hovered) {
    const fade = excludedAlpha[s];
    ctx.globalAlpha = 0.25 + 0.75 * fade;
    if (picked) { ctx.fillStyle = C.paper2; ctx.beginPath(); ctx.arc(q.x, q.y, r + 4, 0, 2 * Math.PI); ctx.fill(); }
    ctx.fillStyle = C.sp[s];
    ctx.beginPath(); ctx.arc(q.x, q.y, r, 0, 2 * Math.PI); ctx.fill();
    if (fade > 0.3) {
      ctx.lineWidth = r > 3 ? 0.9 : 0.7; ctx.strokeStyle = C.dark ? C.paper : C.ink; ctx.globalAlpha = (C.dark ? 0.85 : 0.7) * fade;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    if (picked || hovered) {
      ctx.lineWidth = picked ? 1.6 : 1; ctx.strokeStyle = C.ink;
      ctx.beginPath(); ctx.arc(q.x, q.y, r + (picked ? 3.2 : 2.4), 0, 2 * Math.PI); ctx.stroke();
    }
  }
  function ring(q, s, alpha, picked) {
    ctx.globalAlpha = alpha;
    ctx.lineWidth = 4; ctx.strokeStyle = C.paper2; ctx.beginPath(); ctx.arc(q.x, q.y, 6, 0, 2 * Math.PI); ctx.stroke();
    ctx.lineWidth = picked ? 2.6 : 2; ctx.strokeStyle = C.sp[s]; ctx.stroke();
    ctx.lineWidth = 0.8; ctx.strokeStyle = C.ink; ctx.beginPath(); ctx.arc(q.x, q.y, 7.4, 0, 2 * Math.PI); ctx.stroke();
    ctx.globalAlpha = 1;
  }

  // ---------------------------------------------------------------- the loop
  let visible = false, raf = 0, last = 0;
  function moving() {
    const d = (p, q) => Math.abs(p[0] - q[0]) + Math.abs(p[1] - q[1]) + Math.abs(p[2] - q[2]);
    if (d(half.cur, half.to) > 1e-4) return true;
    if (d(starA.cur, starA.to) > 1e-4 || d(starB.cur, starB.to) > 1e-4) return true;
    if (flowers.some((f) => d(f.cur, f.to) > 1e-4)) return true;
    if (Math.abs(view.yaw - st.yaw) + Math.abs(view.pitch - st.pitch) + Math.abs(view.zoom - st.zoom) > 1e-4) return true;
    if (Math.abs(rulerAlpha - rulerTarget) > 0.01) return true;
    if (own.some((o, s) => Math.abs(o.alpha - (Mo.problem(M, st.n).other.length === 2 && Mo.problem(M, st.n).other.includes(s) ? 1 : 0)) > 0.01)) return true;
    return excludedAlpha.some((v, s) => Math.abs(v - (s === Mo.problem(M, st.n).excluded ? 0 : 1)) > 0.01);
  }
  function step(dt) {
    const snap = reduceQ.matches;
    const t = snap ? 1 : 1 - Math.exp(-dt / TAU);
    const go = (b) => { for (let k = 0; k < 3; k++) b.cur[k] = lerp(b.cur[k], b.to[k], t); };
    flowers.forEach(go); go(starA); go(starB); own.forEach(go); go(half);
    if (st.turning && visible && !document.hidden) st.yaw += TURN * Math.min(dt, 0.05);
    const tv = snap ? 1 : 1 - Math.exp(-dt / 0.12);
    view.yaw = lerp(view.yaw, st.yaw, st.turning ? 1 : tv); view.pitch = lerp(view.pitch, st.pitch, tv); view.zoom = lerp(view.zoom, st.zoom, tv);
    const settled = Math.abs(half.cur[0] - half.to[0]) + Math.abs(half.cur[1] - half.to[1]) + Math.abs(half.cur[2] - half.to[2]) < 0.03;
    rulerAlpha = snap ? (settled ? 1 : 0) : lerp(rulerAlpha, settled ? 1 : 0, settled ? 1 - Math.exp(-dt / 0.18) : 1);
    const ex = Mo.problem(M, st.n).excluded;
    excludedAlpha = excludedAlpha.map((v, s) => lerp(v, s === ex ? 0 : 1, snap ? 1 : 1 - Math.exp(-dt / TAU)));
  }
  function frame(now) {
    raf = 0;
    // easing takes the real time since the last frame, so a slow frame lands nearer the target
    // instead of lagging; only the turn is held to small steps
    const dt = Math.min(1, last ? (now - last) / 1000 : 0.016);
    last = now;
    step(dt);
    draw();
    if (visible && !document.hidden && (st.turning || drag || moving())) raf = requestAnimationFrame(frame);
    else last = 0;
  }
  function wake() { if (!raf && visible) raf = requestAnimationFrame(frame); }
  new IntersectionObserver((es) => { visible = es[0].isIntersecting; if (visible) { resize(); wake(); } }, { rootMargin: "120px" }).observe(caseBox);
  document.addEventListener("visibilitychange", wake);

  // ---------------------------------------------------------------- touch, mouse, and keys
  const pts = new Map();
  let drag = null, pinch = null;
  cv.addEventListener("pointerdown", (e) => {
    cv.setPointerCapture(e.pointerId);
    pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pts.size === 1) drag = { x0: e.clientX, y0: e.clientY, t0: performance.now(), moved: 0, type: e.pointerType };
    if (pts.size === 2) { const [p, q] = [...pts.values()]; pinch = { d: Math.hypot(p.x - q.x, p.y - q.y), z: st.zoom }; drag = null; }
  });
  cv.addEventListener("pointermove", (e) => {
    if (!pts.has(e.pointerId)) { hoverAt(e); return; }
    const prev = pts.get(e.pointerId);
    pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pinch && pts.size === 2) {
      const [p, q] = [...pts.values()];
      st.zoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, (pinch.z * Math.hypot(p.x - q.x, p.y - q.y)) / pinch.d));
      view.zoom = st.zoom; touch(); wake();
      return;
    }
    if (!drag) return;
    const dx = e.clientX - prev.x, dy = e.clientY - prev.y;
    drag.moved += Math.abs(dx) + Math.abs(dy);
    if (drag.moved > 5) {
      touch();
      st.above = null;
      st.yaw -= dx * 0.0085; st.pitch = Math.max(PITCH_MIN, Math.min(PITCH_MAX, st.pitch + dy * 0.0065));
      view.yaw = st.yaw; view.pitch = st.pitch;
      update(); wake();
    }
  });
  const up = (e) => {
    if (!pts.has(e.pointerId)) return;
    pts.delete(e.pointerId);
    if (drag && e.type === "pointerup" && drag.moved <= 6 && performance.now() - drag.t0 < 600) pickAt(e);
    if (pts.size < 2) pinch = null;
    if (!pts.size) drag = null;
    wake();
  };
  cv.addEventListener("pointerup", up); cv.addEventListener("pointercancel", up);
  cv.addEventListener("pointerleave", () => { if (st.hover !== null) { st.hover = null; cv.style.cursor = ""; wake(); } });
  // The wheel zooms with Ctrl or Command held (a trackpad pinch arrives that way); a plain wheel
  // scrolls the page as usual, and says once how to zoom.
  let wheelTimer = 0;
  cv.addEventListener("wheel", (e) => {
    if (!(e.ctrlKey || e.metaKey)) {
      wheelHint.classList.add("is-on");
      clearTimeout(wheelTimer); wheelTimer = setTimeout(() => wheelHint.classList.remove("is-on"), 1400);
      return;
    }
    e.preventDefault(); touch();
    st.zoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, st.zoom * Math.exp(-e.deltaY * (e.deltaMode ? 0.05 : 0.008))));
    wake();
  }, { passive: false });

  function local(e) { const r = cv.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }
  function hit(e, radius) {
    const p = local(e);
    let best = null, bd = Infinity;
    for (const s of shot.stars) {
      if (s.type === "pooled" && !s.on) continue;
      const d = Math.hypot(s.x - p.x, s.y - p.y);
      if (d < 13 && d < bd) { bd = d; best = s; }
    }
    if (best) return { star: best };
    const near = [];
    shot.flowers.forEach((f, i) => { const d = Math.hypot(f.x - p.x, f.y - p.y); if (d < Math.max(radius, f.r + 3)) near.push({ i, d }); });
    if (!near.length) return null;
    near.sort((u, v) => u.d - v.d);
    // flowers drawn at the same spot on the three measurements shown
    const top = M.flowers[near[0].i].v, key = (v) => st.axes.map((j) => v[j]).join();
    const stack = near.filter((q) => key(M.flowers[q.i].v) === key(top)).map((q) => q.i).sort((u, v) => u - v);
    return { i: near[0].i, stack };
  }
  function pickAt(e) {
    touch();
    const h = hit(e, e.pointerType === "touch" ? 16 : 9);
    if (!h) st.pick = null;
    else if (h.star) st.pick = h.star.type === "avg" ? { type: "avg", side: h.star.side } : { type: "pooled", s: h.star.s };
    else {
      let i = h.i;
      // a second tap on a spot two flowers share reads the next one
      if (st.pick && st.pick.type === "flower" && h.stack.length > 1 && h.stack.includes(st.pick.i)) i = h.stack[(h.stack.indexOf(st.pick.i) + 1) % h.stack.length];
      st.pick = { type: "flower", i, stack: h.stack };
    }
    update(); wake();
  }
  function hoverAt(e) {
    if (e.pointerType !== "mouse") return;
    const h = hit(e, 8);
    const i = h && !h.star ? h.i : null;
    cv.style.cursor = h ? "pointer" : "";
    if (i !== st.hover) { st.hover = i; wake(); }
  }
  function stackOf(i) {
    const key = (v) => st.axes.map((j) => v[j]).join(), k = key(M.flowers[i].v);
    return M.flowers.filter((f) => key(f.v) === k).map((f) => f.i);
  }
  cv.addEventListener("keydown", (e) => {
    const k = e.key;
    let used = true;
    if (k === "ArrowLeft") st.yaw += 0.14;
    else if (k === "ArrowRight") st.yaw -= 0.14;
    else if (k === "ArrowUp") st.pitch = Math.min(PITCH_MAX, st.pitch + 0.08);
    else if (k === "ArrowDown") st.pitch = Math.max(PITCH_MIN, st.pitch - 0.08);
    else if (k === "+" || k === "=") st.zoom = Math.min(ZOOM_MAX, st.zoom * 1.15);
    else if (k === "-" || k === "_") st.zoom = Math.max(ZOOM_MIN, st.zoom / 1.15);
    else if (k === "0") { resetB.click(); return; }
    else if (k === "." || k === ",") {
      const n = M.flowers.length, cur = st.pick && st.pick.type === "flower" ? st.pick.i : (k === "." ? -1 : 0);
      const i = (cur + (k === "." ? 1 : -1) + n) % n;
      st.pick = { type: "flower", i, stack: stackOf(i) };
    } else if ((k === "a" || k === "A") && !e.metaKey && !e.ctrlKey && !e.altKey) {
      // the averages, which a pointer picks on their stars and rings: the two class averages, then each
      // pooled species' own average; Shift+A steps back
      const P = Mo.problem(M, st.n);
      const stops = [{ type: "avg", side: "a" }, { type: "avg", side: "b" }, ...(P.other.length === 2 ? P.other.map((s) => ({ type: "pooled", s })) : [])];
      const at = st.pick ? stops.findIndex((q) => q.type === st.pick.type && q.side === st.pick.side && q.s === st.pick.s) : -1;
      const back = e.shiftKey;
      st.pick = stops[at < 0 ? (back ? stops.length - 1 : 0) : (at + (back ? -1 : 1) + stops.length) % stops.length];
    } else if (k === "Escape") st.pick = null;
    else if (k === " ") { e.preventDefault(); st.touched = true; setTurning(!st.turning); return; }
    else used = false;
    if (!used) return;
    e.preventDefault();
    if (k !== "Escape") touch();
    update(); wake();
  });

  update();
  resize();
}

document.querySelectorAll(`figure.plate-live[data-demo="${KIND}"]`).forEach((f) => { mount(f).catch((err) => console.error(err)); });
