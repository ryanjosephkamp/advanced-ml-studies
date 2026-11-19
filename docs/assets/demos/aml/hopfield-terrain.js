// The recall terrain (chapter II, live plate). Exact recall from the study's committed summary,
// drawn as an engraved survey: a pale sheet lifted over its own map. The sheet's corners are the
// measured cells (overlap rho by load P, the load on a log axis), its faces are linear
// interpolation between them, and its contours fall every 0.1 of recall, with the 50% contour in the
// chapter's ink and each overlap's capacity P50 marked on it. Ruled lines drop from every measured
// point to the floor, where the same contours are printed flat. A tap snaps to the nearest measured
// point. Switching the rule or the noise moves the sheet for 0.6 s, and the plate says that nothing
// on it is measured until it settles. The map view draws the same data from above on a plain canvas,
// and it is what shows without WebGL.
//
// All numbers and sentences come from hopfield-terrain-model.js, which the check script tests.
const VQ = new URL(import.meta.url).search; // the page's version, passed on so a browser never mixes versions
const M = await import(new URL(`./hopfield-terrain-model.js${VQ}`, import.meta.url).href);

const KIND = "hopfield-terrain";
const W = 10, D = 6.4, H = 3.0, FLOOR = -1.45; // world units: load across, overlap in depth, recall up
const FOV = 30;
const HOME = { az: 0.42, el: 0.52, zoom: 1 };
const SWAY = { amp: 0.3, period: 40 }; // radians, seconds: from about 7 to 41 degrees round from the front
const reduced = matchMedia("(prefers-reduced-motion: reduce)");
const darkQuery = matchMedia("(prefers-color-scheme: dark)");

{
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = new URL(`./${KIND}.css${VQ}`, import.meta.url).href;
  document.head.append(link);
}

const el = (tag, attrs = {}, kids = []) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") n.textContent = v;
    else if (k === "html") n.innerHTML = v;
    else n.setAttribute(k, v);
  }
  for (const k of kids) n.append(k);
  return n;
};

// ------------------------------------------------------------------ colors from the page's inks
const probe = document.createElement("canvas").getContext("2d");
function rgba(s) {
  probe.fillStyle = "#000";
  probe.fillStyle = s || "#000";
  const v = probe.fillStyle;
  if (v[0] === "#") return [1, 3, 5].map((i) => parseInt(v.slice(i, i + 2), 16)).concat(1);
  const m = v.match(/[\d.]+/g).map(Number);
  return [m[0], m[1], m[2], m[3] ?? 1];
}
const mixc = (a, b, t) => [0, 1, 2].map((i) => a[i] + (b[i] - a[i]) * t).concat(1);
const over = (c, bg) => mixc(bg, c, c[3]); // a translucent ink laid on a background
const css = (c, a = 1) => `rgba(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])},${a})`;
const lum = (c) => (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255;

function readColors(node) {
  const cs = getComputedStyle(node);
  const v = (k) => rgba(cs.getPropertyValue(`--${k}`).trim());
  const c = { ink: v("ink"), ink2: v("ink-2"), paper: v("paper"), paper2: v("paper-2"), rule: v("rule"), rule2: v("rule-2"), mount: v("mount"), accent: v("accent-ink") };
  c.dark = lum(c.paper) < 0.4;
  c.stage = c.paper2;
  // day: a pale paper sheet over its printed map; night: chalk on a lifted slate
  c.sheet = c.dark ? mixc(c.paper2, c.ink, 0.1) : c.mount;
  c.floor = c.dark ? mixc(c.paper2, c.ink, 0.045) : mixc(c.mount, c.paper2, 0.45);
  c.thin = mixc(c.sheet, c.ink, c.dark ? 0.62 : 0.58);
  c.hach = mixc(c.sheet, c.ink, c.dark ? 0.42 : 0.4);
  c.floorHach = mixc(c.floor, c.ink, c.dark ? 0.26 : 0.25);
  c.floorThin = mixc(c.floor, c.ink, c.dark ? 0.5 : 0.46);
  c.net = mixc(c.sheet, c.ink, c.dark ? 0.15 : 0.12);
  c.graticule = mixc(c.floor, c.ink, 0.12);
  c.drop = mixc(c.stage, c.ink, c.dark ? 0.3 : 0.32);
  c.tint = mixc(c.floor, c.accent, c.dark ? 0.16 : 0.13);
  c.wash = c.dark ? 0.06 : 0.08;
  c.edge = mixc(c.stage, c.ink, 0.55);
  return c;
}

// ------------------------------------------------------------------ the plate
async function mount(fig) {
  const root = fig.querySelector(".demo-root");
  const data = await (await fetch(fig.dataset.src)).json();
  const S = M.shape(data);
  const name = (id) => S.names[id];

  const st = {
    rule: data.rules[0].id, noise: data.noises[0], view: "terrain",
    sel: { i: 0, j: data.loads.indexOf(18) >= 0 ? data.loads.indexOf(18) : 0 },
    z: Float64Array.from(S.grid(data.rules[0].id, data.noises[0]).z), morph: null,
    // ?still holds the view at home, for screenshots and bug reports
    az: HOME.az, el: HOME.el, zoom: HOME.zoom, drift: !reduced.matches && !new URLSearchParams(location.search).has("still"),
    t0: performance.now(), tween: null,
    visible: false, three: null, flat: false, colors: null,
  };

  // ---------------------------------------------------------------- the controls
  const radios = (label, items, cls) => {
    const g = el("div", { class: `ht-radios ${cls}`, role: "radiogroup", "aria-label": label });
    g.append(el("span", { class: "ht-lab", "aria-hidden": "true", text: label }));
    for (const [value, text] of items) g.append(el("button", { type: "button", role: "radio", "data-v": String(value), text }));
    g.addEventListener("keydown", (e) => {
      const d = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
      if (!d) return;
      const bs = [...g.querySelectorAll("button:not([disabled])")], i = bs.indexOf(document.activeElement);
      if (i < 0) return;
      e.preventDefault();
      const nx = bs[(i + d + bs.length) % bs.length];
      nx.click(); nx.focus();
    });
    return g;
  };
  const setRadios = (g, value) => g.querySelectorAll("button").forEach((b) => {
    const on = b.dataset.v === String(value);
    b.setAttribute("aria-checked", String(on));
    b.tabIndex = on ? 0 : -1;
  });

  const rules = radios("Rule", data.rules.map((r) => [r.id, r.name]), "ht-rules");
  const noise = radios("Noise", data.noises.map((n) => [n, `${n}%`]), "ht-noise");
  const view = radios("View", [["terrain", "Terrain"], ["map", "Map"]], "ht-view");

  const glCv = el("canvas", { class: "ht-gl", "aria-hidden": "true" });
  const hud = el("canvas", { class: "ht-hud", role: "img" });
  const zoomOut = el("button", { type: "button", class: "ht-tool", "aria-label": "Zoom out", text: "−" });
  const zoomIn = el("button", { type: "button", class: "ht-tool", "aria-label": "Zoom in", text: "+" });
  const reset = el("button", { type: "button", class: "ht-tool ht-reset", text: "Reset" });
  const note = el("p", { class: "ht-note", text: M.FACES });
  const stage = el("div", {
    class: "ht-stage", tabindex: "0",
    "aria-label": "The recall terrain. Arrow keys step to the next measured point; plus and minus zoom; R resets the view.",
  }, [glCv, hud, el("div", { class: "ht-tools" }, [zoomOut, zoomIn, reset])]);

  const big = el("span", { class: "ht-big" }), ci = el("span", { class: "ht-ci" });
  const coords = el("dl", { class: "ht-coords" });
  const say = el("p", { class: "ht-say" }), sayRowP = el("p", { class: "ht-rowsay" });
  const read = el("div", { class: "ht-read", "aria-live": "polite" }, [
    el("div", { class: "ht-fig" }, [el("span", { class: "ht-figlab", text: "Exact recall" }), big, ci]), coords, say, sayRowP]);
  const legend = el("p", {
    class: "ht-legend", html:
      '<span><svg viewBox="0 0 12 12" aria-hidden="true"><path d="M6 2.5 9.5 6 6 9.5 2.5 6z" class="k-dot"/></svg><span class="ht-dotname">measured point</span></span>' +
      '<span><svg viewBox="0 0 24 12" aria-hidden="true"><path d="M1 6h22" class="k-thin"/></svg>contour every 0.1 of recall</span>' +
      '<span><svg viewBox="0 0 24 12" aria-hidden="true"><path d="M1 6h22" class="k-half"/></svg>50% recall</span>' +
      '<span><svg viewBox="0 0 12 12" aria-hidden="true"><path d="M6 1.5 10.5 6 6 10.5 1.5 6z" class="k-p50"/></svg>P50, as the paper writes it</span>' +
      '<span><svg viewBox="0 0 12 12" aria-hidden="true"><path d="M6 1.5 10.5 6 6 10.5 1.5 6z" class="k-open"/></svg>2–3</span>' +
      '<span><svg viewBox="0 0 24 12" aria-hidden="true"><path d="M2 6h13" class="k-half"/><path d="M15 2.5 21 6 15 9.5z" class="k-p50"/></svg>&gt;40</span>' +
      '<span><svg viewBox="0 0 24 12" aria-hidden="true"><path d="M2 2v8M8 2v8M14 2v8M20 2v8" class="k-drop"/></svg>ruled down to the map on the floor</span>',
  });
  const hint = el("p", { class: "ht-hint", text: "Drag to turn, pinch or press + and − to zoom, and tap to read the nearest measured point." });
  const checked = el("p", { class: "ht-checked" });
  root.append(el("div", { class: "ht" }, [
    el("div", { class: "ht-bar" }, [noise, view]), rules, stage, note, read, legend, hint, checked]));

  fetch(fig.dataset.src.replace(/\.json$/, ".checks.json")).then((r) => (r.ok ? r.json() : null)).then((c) => {
    if (c && c.line) checked.textContent = c.line;
  }).catch(() => {});

  // ---------------------------------------------------------------- state changes
  const touch = () => { st.drift = false; };
  rules.addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b || b.dataset.v === st.rule) return;
    morphTo(b.dataset.v, st.noise);
  });
  noise.addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b || Number(b.dataset.v) === st.noise) return;
    morphTo(st.rule, Number(b.dataset.v));
  });
  /** Switch rule or noise; the words name what changes. */
  function morphTo(rule, nz) {
    const moving = Boolean(st.morph);
    const from = rule !== st.rule ? name(st.rule) : `${st.noise}% noise`;
    const to = rule !== st.rule ? name(rule) : `${nz}% noise`;
    st.rule = rule; st.noise = nz;
    const target = S.grid(rule, nz).z;
    if (reduced.matches) { st.z = Float64Array.from(target); st.morph = null; }
    else st.morph = { from: Float64Array.from(st.z), to: target, t0: performance.now(), words: M.sayMorph(from, to, moving) };
    setRadios(rules, rule); setRadios(noise, nz);
    readout(); dirty(); wake();
  }
  view.addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b || b.disabled) return;
    setView(b.dataset.v);
  });
  function setView(v) {
    st.view = v;
    setRadios(view, v);
    stage.classList.toggle("is-map", v === "map");
    readout(); dirty(); wake();
  }
  zoomIn.addEventListener("click", () => { touch(); st.zoom = clampZoom(st.zoom / 1.25); dirty(); });
  zoomOut.addEventListener("click", () => { touch(); st.zoom = clampZoom(st.zoom * 1.25); dirty(); });
  /** The home view: looking down more steeply on a tall stage, so the plan fills more of it. */
  const home = () => {
    const r = stage.clientWidth / Math.max(stage.clientHeight, 1);
    return { az: HOME.az, el: r < 1 ? 0.7 : r < 1.3 ? 0.6 : HOME.el, zoom: HOME.zoom };
  };
  reset.addEventListener("click", () => {
    st.tween = reduced.matches ? null : { from: { az: st.az, el: st.el, zoom: st.zoom }, to: home(), t0: performance.now() };
    if (reduced.matches) Object.assign(st, home());
    st.drift = !reduced.matches; st.t0 = performance.now() + 500;
    dirty(); wake();
  });
  const clampZoom = (z) => Math.min(Math.max(z, 0.5), 1.7);

  stage.addEventListener("keydown", (e) => {
    if (e.target !== stage) return;
    // up the screen is toward overlap 0 in both views (the terrain's low overlaps lie at the back)
    const mv = { ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [-1, 0], ArrowDown: [1, 0] }[e.key];
    if (mv) { e.preventDefault(); touch(); st.sel = M.step(S, st.sel.i, st.sel.j, mv[0], mv[1]); readout(); dirty(); return; }
    if (e.key === "+" || e.key === "=") { e.preventDefault(); zoomIn.click(); }
    else if (e.key === "-" || e.key === "_") { e.preventDefault(); zoomOut.click(); }
    else if (e.key === "r" || e.key === "R") { e.preventDefault(); reset.click(); }
    else if (e.key === " ") { e.preventDefault(); st.drift = !st.drift && !reduced.matches; st.t0 = performance.now() - driftPhaseTime(); wake(); }
  });
  // keep the sway continuous when it resumes from where the reader left it
  const driftPhaseTime = () => (Math.asin(Math.max(-1, Math.min(1, (st.az - HOME.az) / SWAY.amp))) / (2 * Math.PI)) * SWAY.period * 1000;

  // ---------------------------------------------------------------- the readout
  function readout() {
    const g = S.grid(st.rule, st.noise), c = M.cell(S, g, st.sel.i, st.sel.j);
    const args = { rule: name(st.rule), noise: st.noise, level: c.level, P: c.P, recall: c.recall, lo: c.lo, hi: c.hi, sets: S.sets };
    const r = M.readout(args);
    // while a switch runs, nothing measured is shown: the surface on screen is not the target's
    const mv = Boolean(st.morph), to = name(st.rule);
    big.textContent = mv ? "…" : r.recall; ci.textContent = mv ? "" : r.ci;
    coords.replaceChildren(...[["P", r.P], ["ρ", r.rho], ["Rule", r.rule], ["Noise", r.noise]].map(([k, v]) => el("div", {}, [el("dt", { text: k }), el("dd", { text: v })])));
    const ms = M.markers(S, st.rule, st.noise);
    say.textContent = mv ? st.morph.words : M.sayCell(args);
    sayRowP.textContent = mv ? M.sayRowMoving(ms[st.sel.i].level, to, st.noise) : M.sayRow(ms[st.sel.i]);
    read.classList.toggle("is-moving", mv);
    note.textContent = mv ? M.sayMoving(to, st.noise) : st.view === "map" ? M.MAP_NOTE : M.FACES;
    hud.setAttribute("aria-label", mv ? M.sayMoving(to, st.noise) : M.sayCapacities(to, st.noise, ms));
    legend.querySelector(".ht-dotname").textContent = M.dotName(mv);
  }

  // ---------------------------------------------------------------- picking and turning
  const pts = new Map();
  let drag = null;
  stage.addEventListener("pointerdown", (e) => {
    if (e.target.closest(".ht-tools")) return;
    pts.set(e.pointerId, [e.clientX, e.clientY]);
    if (pts.size === 1) drag = { x: e.clientX, y: e.clientY, moved: false, az: st.az, el: st.el };
    if (pts.size === 2) {
      const p = [...pts.values()];
      drag = { pinch: Math.hypot(p[0][0] - p[1][0], p[0][1] - p[1][1]), zoom: st.zoom, my: (p[0][1] + p[1][1]) / 2, el: st.el, moved: true };
    }
    try { stage.setPointerCapture(e.pointerId); } catch {}
  });
  stage.addEventListener("pointermove", (e) => {
    if (!pts.has(e.pointerId) || !drag) return;
    pts.set(e.pointerId, [e.clientX, e.clientY]);
    if (st.view !== "terrain" || !st.three) return;
    if (pts.size >= 2 && drag.pinch) {
      const p = [...pts.values()];
      const d = Math.hypot(p[0][0] - p[1][0], p[0][1] - p[1][1]), my = (p[0][1] + p[1][1]) / 2;
      st.zoom = clampZoom(drag.zoom * drag.pinch / Math.max(d, 1));
      st.el = clampEl(drag.el + (my - drag.my) * 0.006);
      touch(); dirty();
      return;
    }
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (!drag.moved && Math.hypot(dx, dy) > 5) drag.moved = true;
    if (drag.moved) {
      touch(); st.tween = null;
      st.az = drag.az - dx * 0.009;
      if (e.pointerType === "mouse" || e.pointerType === "pen") st.el = clampEl(drag.el + dy * 0.006);
      dirty();
    }
  });
  const up = (e) => {
    if (!pts.has(e.pointerId)) return;
    pts.delete(e.pointerId);
    if (e.type === "pointerup" && drag && !drag.moved && pts.size === 0) {
      const r = stage.getBoundingClientRect();
      pick(e.clientX - r.left, e.clientY - r.top);
    }
    if (pts.size === 0) drag = null;
    // one finger left after a pinch: carry on as a turn from where it is (never as a tap)
    else if (pts.size === 1) { const [x, y] = [...pts.values()][0]; drag = { x, y, moved: true, az: st.az, el: st.el }; }
  };
  stage.addEventListener("pointerup", up);
  stage.addEventListener("pointercancel", up);
  stage.addEventListener("wheel", (e) => {
    // a trackpad pinch (or a held Control or Command key) zooms; a plain wheel scrolls the page
    if (!(e.ctrlKey || e.metaKey) || st.view !== "terrain") return;
    e.preventDefault(); touch();
    st.zoom = clampZoom(st.zoom * Math.exp(e.deltaY * 0.01));
    dirty();
  }, { passive: false });
  const clampEl = (v) => Math.min(Math.max(v, 0.14), 1.42);

  function pick(x, y) {
    let cellAt = null;
    if (st.view === "terrain" && st.three) cellAt = st.three.pick(x, y);
    else if (mapGeo) {
      const hit = M.nearestPoint(mapGeo.pts, x, y);
      if (hit.k >= 0 && hit.d < 48) cellAt = { i: Math.floor(hit.k / S.nc), j: hit.k % S.nc };
    }
    if (!cellAt) return;
    touch();
    st.sel = cellAt;
    readout(); dirty();
  }

  // ---------------------------------------------------------------- drawing
  let loop = 0, need = true, mapGeo = null;
  const dirty = () => { need = true; wake(); };
  const animating = () => Boolean(st.morph || st.tween || (st.drift && st.view === "terrain" && st.three && !reduced.matches));
  function wake() { if (!loop && st.visible && !document.hidden) loop = requestAnimationFrame(frame); }
  function frame(now) {
    loop = 0;
    if (!st.visible || document.hidden) return;
    if (st.morph) {
      const s = M.ease((now - st.morph.t0) / (M.MORPH_S * 1000));
      M.mix(st.morph.from, st.morph.to, s, st.z);
      if (s >= 1) { st.z = Float64Array.from(st.morph.to); st.morph = null; readout(); }
      need = true;
    }
    if (st.tween) {
      const s = M.ease((now - st.tween.t0) / 550), f = st.tween.from;
      // the shorter way round to the home azimuth
      const to = st.tween.to, daz = Math.atan2(Math.sin(to.az - f.az), Math.cos(to.az - f.az));
      st.az = f.az + daz * s; st.el = f.el + (to.el - f.el) * s; st.zoom = f.zoom + (to.zoom - f.zoom) * s;
      if (s >= 1) st.tween = null;
      need = true;
    } else if (st.drift && st.view === "terrain" && st.three && !reduced.matches && now > st.t0) {
      st.az = HOME.az + SWAY.amp * Math.sin((2 * Math.PI * (now - st.t0)) / (SWAY.period * 1000));
      need = true;
    }
    if (need) draw();
    if (animating()) loop = requestAnimationFrame(frame);
  }
  function draw() {
    need = false;
    const w = stage.clientWidth, h = stage.clientHeight;
    if (!w || !h) return;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    if (hud.width !== Math.round(w * dpr) || hud.height !== Math.round(h * dpr)) { hud.width = Math.round(w * dpr); hud.height = Math.round(h * dpr); }
    const ctx = hud.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const c = st.colors;
    const font = getComputedStyle(root);
    const fonts = { type: font.getPropertyValue("--type").trim() || "monospace", serif: font.getPropertyValue("--serif").trim() || "serif", fat: font.getPropertyValue("--fat").trim() || "serif" };
    const g = { ctx, w, h, c, fonts, small: w < 520 };
    g.carto = cartouche(g);
    if (st.view === "terrain" && st.three) {
      st.three.render(w, h, dpr);
      drawHud3(g);
    } else if (st.view === "map" || st.flat) {
      drawMap(g);
    }
  }

  // ---------------------------------------------------------------- shared marks
  function halo(ctx, text, x, y, c, color, width = 3) {
    ctx.lineJoin = "round"; ctx.lineWidth = width; ctx.strokeStyle = css(c.stage, 0.92);
    ctx.strokeText(text, x, y);
    ctx.fillStyle = color; ctx.fillText(text, x, y);
  }
  /** A label on a small paper ground, so the lines under it don't break its figures. */
  function tag(ctx, text, x, y, c, color) {
    const tw = ctx.measureText(text).width;
    ctx.fillStyle = css(c.stage, 0.9);
    ctx.fillRect(x - 3, y - 7.5, tw + 6, 15);
    ctx.fillStyle = color; ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.fillText(text, x, y + 0.5);
  }
  function diamond(ctx, x, y, r, c, open, alpha = 1) {
    ctx.save(); ctx.globalAlpha = alpha;
    ctx.beginPath(); ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y); ctx.closePath();
    ctx.fillStyle = open ? css(c.sheet) : css(c.accent);
    ctx.strokeStyle = open ? css(c.accent) : css(c.sheet);
    ctx.lineWidth = open ? 1.8 : 1.4;
    ctx.fill(); ctx.stroke();
    ctx.restore();
  }
  function arrow(ctx, x, y, dx, dy, c, alpha = 1) {
    const n = Math.hypot(dx, dy) || 1, ux = dx / n, uy = dy / n, L = 14;
    ctx.save(); ctx.globalAlpha = alpha; ctx.strokeStyle = css(c.accent); ctx.fillStyle = css(c.accent); ctx.lineWidth = 1.8;
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + ux * L, y + uy * L); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x + ux * (L + 5), y + uy * (L + 5)); ctx.lineTo(x + ux * L - uy * 4, y + uy * L + ux * 4); ctx.lineTo(x + ux * L + uy * 4, y + uy * L - ux * 4); ctx.closePath(); ctx.fill();
    ctx.restore();
    return [x + ux * (L + 7), y + uy * (L + 7)];
  }
  /** Labels give way to each other: a label that would cover one already placed is left out. */
  function placer(w, h) {
    const boxes = [];
    return (x, y, wd, ht) => {
      const b = [x - 2, y - 2, x + wd + 2, y + ht + 2];
      if (b[0] < 2 || b[2] > w - 2 || b[1] < 2 || b[3] > h - 2) return false;
      if (boxes.some((o) => b[0] < o[2] && b[2] > o[0] && b[1] < o[3] && b[3] > o[1])) return false;
      boxes.push(b);
      return true;
    };
  }
  function cartouche({ ctx, c, fonts, small }) {
    const title = name(st.rule), sub = M.cartoSub(st.noise, Boolean(st.morph));
    ctx.save();
    ctx.font = `700 ${small ? 13 : 15}px ${fonts.fat}`;
    const tw = ctx.measureText(title).width;
    ctx.font = `400 ${small ? 10.5 : 11.5}px ${fonts.type}`;
    const sw = ctx.measureText(sub).width;
    const bw = Math.max(tw, sw) + 20, bh = small ? 42 : 48, x = 10, y = 10;
    ctx.fillStyle = css(c.stage, 0.9); ctx.fillRect(x, y, bw, bh);
    ctx.strokeStyle = css(c.ink, 0.75); ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, bw - 1, bh - 1);
    ctx.strokeStyle = css(c.ink, 0.35); ctx.strokeRect(x + 3.5, y + 3.5, bw - 7, bh - 7);
    ctx.fillStyle = css(c.ink); ctx.textBaseline = "alphabetic"; ctx.textAlign = "left";
    ctx.font = `700 ${small ? 13 : 15}px ${fonts.fat}`; ctx.fillText(title, x + 10, y + (small ? 20 : 23));
    ctx.fillStyle = css(c.ink2); ctx.font = `400 ${small ? 10.5 : 11.5}px ${fonts.type}`; ctx.fillText(sub, x + 10, y + (small ? 34 : 39));
    ctx.restore();
    return { x: 10, y: 10, w: bw, h: bh };
  }

  // ---------------------------------------------------------------- the 3D view's labels
  function drawHud3(g) {
    const { ctx, w, h, c, fonts, small } = g, T3 = st.three;
    const place = placer(w, h);
    // the cartouche's corner and the tools' corner are taken
    place(g.carto.x, g.carto.y, g.carto.w, g.carto.h);
    const tools = stage.querySelector(".ht-tools").getBoundingClientRect(), sr = stage.getBoundingClientRect();
    place(tools.left - sr.left, tools.top - sr.top, tools.width, tools.height);
    const ax = T3.axes();
    ctx.save();
    ctx.textBaseline = "middle";
    // the load axis, along the near edge of the floor
    ctx.font = `400 ${small ? 10 : 11}px ${fonts.type}`;
    const order = [2, 40, 12, 4, 6, 24, 9, 3, 18, 5, 30, 8, 15, 7];
    for (const P of order) {
      const p = T3.project(...ax.pTick(P));
      if (!p.ok) continue;
      const s = String(P), tw = ctx.measureText(s).width;
      if (place(p.x - tw / 2, p.y - 6, tw, 12)) { ctx.textAlign = "center"; halo(ctx, s, p.x, p.y, c, css(c.ink2)); }
    }
    // the overlap axis, along the near side
    for (let i = 0; i < S.nr; i++) {
      if (small && i % 3 !== 0 && i !== S.nr - 1) continue;
      const p = T3.project(...ax.rTick(i));
      if (!p.ok) continue;
      const s = data.levels[i], tw = ctx.measureText(s).width;
      if (place(p.x - tw / 2, p.y - 6, tw, 12)) { ctx.textAlign = "center"; halo(ctx, s, p.x, p.y, c, css(c.ink2)); }
    }
    // the recall scale on its post
    for (const r of [0, 0.5, 1]) {
      const p = T3.project(...ax.zTick(r));
      if (!p.ok) continue;
      const s = r === 0.5 ? "0.5" : String(r), tw = ctx.measureText(s).width;
      // outside the post, or inside it when the stage's edge is too near
      const x = [p.x + (ax.postLeft ? -12 - tw : 12), p.x + (ax.postLeft ? 8 : -8 - tw)].find((x0) => place(x0, p.y - 6, tw, 12));
      if (x !== undefined) { ctx.textAlign = "left"; halo(ctx, s, x, p.y, c, r === 0.5 ? css(c.accent) : css(c.ink2)); }
    }
    // axis titles, set along their edges a fixed step outside the scale's figures, and moved in
    // along the edge's normal when the stage's edge is near (left out if even that won't fit)
    ctx.font = `italic 400 ${small ? 11.5 : 12.5}px ${fonts.serif}`;
    const title = (text, a, b) => {
      const p = T3.project(...a), q = T3.project(...b);
      if (!p.ok || !q.ok) return;
      let ang = Math.atan2(q.y - p.y, q.x - p.x);
      if (ang > Math.PI / 2) ang -= Math.PI; else if (ang < -Math.PI / 2) ang += Math.PI;
      const tw = ctx.measureText(text).width;
      const ex = Math.abs(Math.cos(ang)) * tw / 2 + 7, ey = Math.abs(Math.sin(ang)) * tw / 2 + 7;
      const x = Math.min(Math.max((p.x + q.x) / 2, ex + 2), w - ex - 2), y = Math.min(Math.max((p.y + q.y) / 2, ey + 2), h - ey - 2);
      if (Math.hypot(x - (p.x + q.x) / 2, y - (p.y + q.y) / 2) > 18) return;
      ctx.save(); ctx.translate(x, y); ctx.rotate(ang); ctx.textAlign = "center";
      halo(ctx, text, 0, 0, c, css(c.ink2));
      ctx.restore();
    };
    title("patterns stored, P (log scale)", ...ax.pTitle);
    title("expected overlap ρ", ...ax.rTitle);
    ctx.restore();

    // the selected point, its tag, and the capacities on the 50% contour
    const ms = M.markers(S, st.rule, st.noise);
    const settled = !st.morph;
    const k = st.sel.i * S.nc + st.sel.j;
    const sp = T3.project(...T3.vertex(k));
    if (sp.ok) {
      ctx.save();
      ctx.strokeStyle = css(c.accent); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(sp.x, sp.y, 7.5, 0, 2 * Math.PI); ctx.stroke();
      if (settled) {
        const g2 = S.grid(st.rule, st.noise);
        const text = M.f3(g2.z[k]);
        ctx.font = `700 ${small ? 11 : 12}px ${fonts.type}`; ctx.textBaseline = "middle";
        const tw = ctx.measureText(text).width;
        const spot = [[sp.x + 11, sp.y - 13], [sp.x - 11 - tw, sp.y - 13], [sp.x + 11, sp.y + 13], [sp.x - 11 - tw, sp.y + 13]]
          .find(([x, y]) => place(x - 3, y - 7.5, tw + 6, 15));
        if (spot) tag(ctx, text, spot[0], spot[1], c, css(c.ink));
      }
      ctx.restore();
    }
    if (!settled) return;
    // the markers first, then their labels (the selected row's first), each set clear of the line:
    // above it, else below it, else left out
    const sorted = [ms[st.sel.i], ...ms.filter((m) => m.i !== st.sel.i)];
    ctx.save();
    ctx.font = `700 ${small ? 10.5 : 11.5}px ${fonts.type}`; ctx.textBaseline = "middle";
    const labels = [];
    for (const m of [...sorted].reverse()) {
      const pos = T3.markerAt(m), p = T3.project(...pos);
      if (!p.ok) continue;
      const hidden = T3.occluded(pos), a = hidden ? 0.3 : 1;
      let tip = null;
      if (m.kind === "above") {
        const q = T3.project(pos[0] + 1, pos[1], pos[2]);
        tip = arrow(ctx, p.x, p.y, q.x - p.x, q.y - p.y, c, a);
        tip.push(q.x < p.x);
      } else diamond(ctx, p.x, p.y, m.i === st.sel.i ? 6 : 5, c, m.kind !== "value", a);
      if (!hidden || m.i === st.sel.i) labels.unshift({ m, p, tip });
    }
    for (const { m, p, tip } of labels) {
      const tw = ctx.measureText(m.label).width;
      const spots = tip
        ? [[tip[2] ? tip[0] - tw - 2 : tip[0] + 2, tip[1]], [p.x - tw / 2, p.y - 13]]
        : [[p.x + 6, p.y - 12], [p.x - 6 - tw, p.y - 12], [p.x + 6, p.y + 12], [p.x - 6 - tw, p.y + 12]];
      const spot = spots.find(([x, y]) => place(x - 3, y - 7.5, tw + 6, 15));
      if (spot) tag(ctx, m.label, spot[0], spot[1], c, css(c.accent));
    }
    ctx.restore();
  }

  // ---------------------------------------------------------------- the map (and the view without WebGL)
  function drawMap(g) {
    const { ctx, w, h, c, fonts, small } = g;
    const L = small ? 40 : 52, R = small ? 46 : 64, T = small ? 64 : 72, B = small ? 46 : 52;
    const pw = w - L - R, ph = h - T - B;
    const X = (u) => L + u * pw, Y = (v) => T + v * ph;
    const z = st.z, settled = !st.morph;
    // the sheet
    ctx.fillStyle = css(c.floor); ctx.fillRect(L, T, pw, ph);
    // where recall is at least 50%
    ctx.fillStyle = css(c.tint);
    for (const Tr of S.tris) {
      const poly = M.above(S, z, Tr, M.HALF);
      if (poly.length < 3) continue;
      ctx.beginPath(); poly.forEach(([u, v], n) => (n ? ctx.lineTo(X(u), Y(v)) : ctx.moveTo(X(u), Y(v)))); ctx.closePath(); ctx.fill();
    }
    // the survey net through the measured points
    ctx.strokeStyle = css(c.graticule); ctx.lineWidth = 1;
    ctx.beginPath();
    for (const u of S.u) { ctx.moveTo(Math.round(X(u)) + 0.5, T); ctx.lineTo(Math.round(X(u)) + 0.5, T + ph); }
    for (const v of S.v) { ctx.moveTo(L, Math.round(Y(v)) + 0.5); ctx.lineTo(L + pw, Math.round(Y(v)) + 0.5); }
    ctx.stroke();
    // hachures: strokes straight downhill from each contour to the next, heavier where steeper
    ctx.strokeStyle = css(c.floorHach); ctx.lineCap = "butt";
    const kx = pw / W, ky = ph / D;
    for (const hs of M.hachureStrokes(S, z, W, D, { gap: Math.max(0.12, 6.5 / kx) })) {
      for (let s = 0; s < hs.tris.length; s++) {
        const steep = Math.min(Math.atan(M.downhill(S, z, S.tris[hs.tris[s]], W, D).slope * H) / (Math.PI / 2), 1);
        ctx.lineWidth = 0.4 + 0.9 * steep;
        ctx.beginPath(); ctx.moveTo(L + hs.pts[s][0] * kx, T + hs.pts[s][1] * ky); ctx.lineTo(L + hs.pts[s + 1][0] * kx, T + hs.pts[s + 1][1] * ky); ctx.stroke();
      }
    }
    // contours every 0.1, the 50% contour in the accent, each labeled once on its longest line
    const place = placer(w, h);
    place(g.carto.x, g.carto.y, g.carto.w, g.carto.h);
    const labels = [];
    for (const lv of M.LEVELS) {
      const half = lv === M.HALF;
      const lines = M.chain(M.contour(z, S.tris, lv));
      ctx.strokeStyle = half ? css(c.accent) : css(c.floorThin);
      ctx.lineWidth = half ? 2.6 : 1;
      ctx.lineJoin = "round"; ctx.lineCap = "round";
      let longest = null;
      for (const ln of lines) {
        const xy = ln.pts.map((p) => { const [u, v] = M.at(S, z, p); return [X(u), Y(v)]; });
        ctx.beginPath();
        xy.forEach(([x, y], n) => (n ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        ctx.stroke();
        let len = 0;
        for (let n = 1; n < xy.length; n++) len += Math.hypot(xy[n][0] - xy[n - 1][0], xy[n][1] - xy[n - 1][1]);
        if (!longest || len > longest.len) longest = { len, xy };
      }
      if (longest && longest.len > 40) labels.push({ lv, half, xy: longest.xy, len: longest.len });
    }
    // the measured points, which the labels below keep clear of
    const pts = [];
    ctx.strokeStyle = css(c.ink); ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < S.nr; i++) for (let j = 0; j < S.nc; j++) {
      const x = X(S.u[j]), y = Y(S.v[i]);
      pts.push([x, y]);
      ctx.moveTo(x - 2.5, y); ctx.lineTo(x + 2.5, y); ctx.moveTo(x, y - 2.5); ctx.lineTo(x, y + 2.5);
      place(x - 3, y - 3, 6, 6);
    }
    ctx.stroke();
    mapGeo = { pts };
    ctx.strokeStyle = css(c.ink, 0.6); ctx.lineWidth = 1; ctx.strokeRect(L - 0.5, T - 0.5, pw + 1, ph + 1);
    // the selected row and point
    const si = st.sel.i, sj = st.sel.j;
    ctx.strokeStyle = css(c.ink, 0.55); ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(L, Y(S.v[si])); ctx.lineTo(L + pw, Y(S.v[si])); ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = css(c.accent); ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(X(S.u[sj]), Y(S.v[si]), 7, 0, 2 * Math.PI); ctx.stroke();
    place(X(S.u[sj]) - 9, Y(S.v[si]) - 9, 18, 18);
    // P50 on the 50% contour
    const ms = settled ? M.markers(S, st.rule, st.noise) : [];
    for (const m of ms) {
      const [mu, mv] = M.markerUV(m), x = X(mu), y = Y(mv);
      if (m.kind === "above") { arrow(ctx, x - 18, y, 1, 0, c); place(x - 20, y - 6, 22, 12); }
      else { diamond(ctx, x, y, m.i === si ? 6 : 5, c, m.kind !== "value"); place(x - 7, y - 7, 14, 14); }
    }
    if (settled) {
      ctx.font = `400 ${small ? 9 : 10}px ${fonts.type}`;
      for (const { lv, half, xy, len } of labels) {
        // try the middle of the line first, then points either side of it
        const text = lv.toFixed(1), tw = ctx.measureText(text).width;
        for (const f of [0.5, 0.35, 0.65, 0.2, 0.8]) {
          let acc = 0, at = null;
          for (let n = 1; n < xy.length && !at; n++) {
            const d = Math.hypot(xy[n][0] - xy[n - 1][0], xy[n][1] - xy[n - 1][1]);
            if (acc + d >= f * len) { const t = (f * len - acc) / d; at = [xy[n - 1][0] + (xy[n][0] - xy[n - 1][0]) * t, xy[n - 1][1] + (xy[n][1] - xy[n - 1][1]) * t]; }
            acc += d;
          }
          if (at && place(at[0] - tw / 2 - 3, at[1] - 7, tw + 6, 14)) {
            ctx.fillStyle = css(c.floor); ctx.fillRect(at[0] - tw / 2 - 2, at[1] - 6, tw + 4, 12);
            ctx.fillStyle = half ? css(c.accent) : css(c.ink2); ctx.textAlign = "center"; ctx.textBaseline = "middle";
            ctx.fillText(text, at[0], at[1] + 0.5);
            break;
          }
        }
      }
    }
    // axes
    ctx.font = `400 ${small ? 10 : 11}px ${fonts.type}`; ctx.textBaseline = "middle"; ctx.fillStyle = css(c.ink2);
    for (const P of [2, 40, 12, 4, 6, 24, 9, 3, 18, 5, 30, 8, 15, 7]) {
      const x = X(S.uOfP(P)), s = String(P), tw = ctx.measureText(s).width;
      if (place(x - tw / 2, T + ph + 9, tw, 12)) { ctx.textAlign = "center"; ctx.fillText(s, x, T + ph + 15); }
    }
    ctx.textAlign = "right";
    for (let i = 0; i < S.nr; i++) ctx.fillText(data.levels[i], L - 7, Y(S.v[i]));
    ctx.font = `italic 400 ${small ? 11.5 : 12.5}px ${fonts.serif}`; ctx.fillStyle = css(c.ink2); ctx.textAlign = "center";
    ctx.fillText("patterns stored, P (log scale)", L + pw / 2, T + ph + 35);
    ctx.save(); ctx.translate(small ? 10 : 14, T + ph / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("expected overlap ρ", 0, 0); ctx.restore();
    // P50 as a column at the right, like the paper's Table II
    // the column's head, set as the paper sets it: an italic P with 50 below the line
    ctx.fillStyle = css(c.ink2); ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
    ctx.font = `italic 400 ${small ? 12 : 13}px ${fonts.serif}`;
    ctx.fillText("P", L + pw + 8, T - 9);
    const pW = ctx.measureText("P").width;
    ctx.font = `400 ${small ? 8.5 : 9}px ${fonts.serif}`;
    ctx.fillText("50", L + pw + 8 + pW + 0.5, T - 6);
    ctx.textBaseline = "middle";
    ctx.font = `700 ${small ? 10.5 : 11.5}px ${fonts.type}`;
    for (const m of ms) {
      ctx.fillStyle = m.i === si ? css(c.accent) : css(c.ink);
      ctx.fillText(m.label, L + pw + 8, Y(M.markerUV(m)[1]));
    }
  }

  // ---------------------------------------------------------------- the 3D scene, once it is near
  async function build() {
    let T;
    try {
      // ?flat shows the plate as it is without WebGL
      if (new URLSearchParams(location.search).has("flat")) throw new Error("flat");
      const probeCv = document.createElement("canvas");
      if (!(probeCv.getContext("webgl2") || probeCv.getContext("webgl"))) throw new Error("no WebGL");
      T = await import("./three.min.js");
      st.three = scene3(T);
    } catch {
      st.flat = true;
      glCv.hidden = true;
      const tb = view.querySelector('[data-v="terrain"]');
      tb.disabled = true; tb.title = "The terrain view needs WebGL; the map shows the same data.";
      setView("map");
      return;
    }
    st.el = home().el;
    stage.classList.add("is-ready");
    dirty();
  }

  function scene3(T) {
    const renderer = new T.WebGLRenderer({ canvas: glCv, antialias: true, alpha: true, powerPreference: "low-power" });
    renderer.setClearColor(0x000000, 0);
    const scene = new T.Scene();
    const camera = new T.PerspectiveCamera(FOV, 1, 0.1, 300);
    const col = (a) => new T.Color().setRGB(a[0] / 255, a[1] / 255, a[2] / 255, T.SRGBColorSpace);
    const X = (u) => (u - 0.5) * W, Z = (v) => (v - 0.5) * D, Y = (r) => r * H;
    const nV = S.nr * S.nc, nT = S.tris.length;

    const dynamic = (maxVerts, color = false) => {
      const g = new T.BufferGeometry();
      g.setAttribute("position", new T.Float32BufferAttribute(new Float32Array(maxVerts * 3), 3));
      if (color) g.setAttribute("color", new T.Float32BufferAttribute(new Float32Array(maxVerts * 3), 3));
      g.setDrawRange(0, 0);
      return g;
    };
    const writer = (g) => {
      const a = g.attributes.position.array;
      let n = 0;
      return {
        v(x, y, z) { if (n + 3 <= a.length) { a[n++] = x; a[n++] = y; a[n++] = z; } },
        done() { g.attributes.position.needsUpdate = true; g.setDrawRange(0, n / 3); },
      };
    };
    const add = (o, order = 0) => { o.frustumCulled = false; o.renderOrder = order; scene.add(o); return o; };
    const basic = (o = {}) => new T.MeshBasicMaterial({ side: T.DoubleSide, ...o });
    const lineMat = (o = {}) => new T.LineBasicMaterial(o);

    // materials: polygons lying under lines are pushed back, so lines on them always show
    const mat = {
      floor: basic({ polygonOffset: true, polygonOffsetFactor: 2, polygonOffsetUnits: 4 }),
      floorThin: basic(), floorHach: basic(), floorHalf: basic({ polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1 }),
      // the sheet is tracing paper: the map on the floor shows through it
      surf: basic({ vertexColors: true, transparent: true, polygonOffset: true, polygonOffsetFactor: 1.5, polygonOffsetUnits: 3 }),
      thin: basic({ polygonOffset: true, polygonOffsetFactor: -0.5, polygonOffsetUnits: -1 }),
      hach: basic({ polygonOffset: true, polygonOffsetFactor: -0.5, polygonOffsetUnits: -1 }),
      half: basic({ polygonOffset: true, polygonOffsetFactor: -1.5, polygonOffsetUnits: -3 }),
      row: basic({ polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -2 }),
      rim: basic({ polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -2 }),
      dots: basic(), sel: basic(),
      net: lineMat(), grat: lineMat(), drop: lineMat(), axis: lineMat(), selDrop: lineMat(),
    };

    // the floor sheet, its graticule, and its frame
    const floorGeo = new T.PlaneGeometry(W, D); floorGeo.rotateX(-Math.PI / 2);
    const floor = add(new T.Mesh(floorGeo, mat.floor), 0); floor.position.y = FLOOR;
    const gratGeo = new T.BufferGeometry(), gv = [];
    for (const u of S.u) gv.push(X(u), FLOOR, Z(0), X(u), FLOOR, Z(1));
    for (const v of S.v) gv.push(X(0), FLOOR, Z(v), X(1), FLOOR, Z(v));
    gratGeo.setAttribute("position", new T.Float32BufferAttribute(gv, 3));
    add(new T.LineSegments(gratGeo, mat.grat), 1);
    const floorThinGeo = dynamic(nT * 8 * 6 + 6000); add(new T.Mesh(floorThinGeo, mat.floorThin), 2);
    const floorHalfGeo = dynamic(nT * 6 + 4000); add(new T.Mesh(floorHalfGeo, mat.floorHalf), 3);
    const floorHachGeo = dynamic(60000); add(new T.Mesh(floorHachGeo, mat.floorHach), 2);
    // the frame, the scale ticks, and the recall post (placed toward the camera; see axes())
    const axisGeo = dynamic(400); add(new T.LineSegments(axisGeo, mat.axis), 2);

    // the ruled lines from each measured point to the floor
    const dropGeo = dynamic(nV * 2); add(new T.LineSegments(dropGeo, mat.drop), 2);
    const selDropGeo = dynamic(2); add(new T.LineSegments(selDropGeo, mat.selDrop), 6);

    // the sheet: flat triangles through the measured points, washed darker where steeper
    const surfGeo = dynamic(nT * 3, true); add(new T.Mesh(surfGeo, mat.surf), 3);
    const netGeo = dynamic((S.nr * (S.nc - 1) + S.nc * (S.nr - 1)) * 2); add(new T.LineSegments(netGeo, mat.net), 4);
    const thinGeo = dynamic(nT * 8 * 6); add(new T.Mesh(thinGeo, mat.thin), 4);
    const hachGeo = dynamic(60000); add(new T.Mesh(hachGeo, mat.hach), 4);
    const halfGeo = dynamic(nT * 6); add(new T.Mesh(halfGeo, mat.half), 5);
    const rowGeo = dynamic(S.nc * 6); add(new T.Mesh(rowGeo, mat.row), 5);
    // the sheet's outline: its four edge profiles, through the measured points on its border
    const rimGeo = dynamic((S.nc + S.nr) * 2 * 12); add(new T.Mesh(rimGeo, mat.rim), 5);
    const dotsGeo = dynamic(nV * 24); add(new T.Mesh(dotsGeo, mat.dots), 6);
    const selGeo = dynamic(24); add(new T.Mesh(selGeo, mat.sel), 7);

    const vx = (k) => [X(S.u[k % S.nc]), Y(st.z[k]), Z(S.v[Math.floor(k / S.nc)])];
    const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const norm = (a) => { const n = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / n, a[1] / n, a[2] / n]; };
    /** A flat strip of width w from p to q, in the plane with normal n, its ends carried past by ext. */
    function strip(wr, p, q, n, w, ext = 0) {
      const d = norm(sub(q, p)), s = norm(cross(n, d)), hw = w / 2;
      const a = [p[0] - d[0] * ext, p[1] - d[1] * ext, p[2] - d[2] * ext], b = [q[0] + d[0] * ext, q[1] + d[1] * ext, q[2] + d[2] * ext];
      const P = (o, k) => [o[0] + s[0] * hw * k, o[1] + s[1] * hw * k, o[2] + s[2] * hw * k];
      for (const v of [P(a, -1), P(a, 1), P(b, 1), P(a, -1), P(b, 1), P(b, -1)]) wr.v(...v);
    }
    function octa(wr, p, r) {
      const [x, y, z] = p, V = [[r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0], [0, 0, r], [0, 0, -r]];
      for (const [a, b, c2] of [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
        for (const k of [a, b, c2]) wr.v(x + V[k][0], y + V[k][1], z + V[k][2]);
    }
    const triNormal = (T3) => { const [a, b, c2] = T3.map(vx); const n = norm(cross(sub(b, a), sub(c2, a))); return n[1] < 0 ? n.map((x) => -x) : n; };

    let washTone = null;
    /** Everything that follows the heights: rebuilt each frame of a switch, once otherwise. */
    function rebuild() {
      const c = st.colors, z = st.z;
      // the sheet and its wash
      {
        const wr = writer(surfGeo), ca = surfGeo.attributes.color.array;
        let n = 0;
        const sheet = c.sheet, ink = c.ink;
        for (const T3 of S.tris) {
          const d = M.downhill(S, z, T3, W, D);
          const tone = Math.pow(Math.min(Math.atan(d.slope * H) / (Math.PI / 2), 1), 0.85) * c.wash;
          const k = col(mixc(sheet, ink, tone));
          for (const v of T3) { wr.v(...vx(v)); ca[n++] = k.r; ca[n++] = k.g; ca[n++] = k.b; }
        }
        wr.done(); surfGeo.attributes.color.needsUpdate = true;
        washTone = c.wash;
      }
      // the measured net on the sheet, the dots, and the ruled drops
      {
        const wn = writer(netGeo), wd = writer(dotsGeo), wl = writer(dropGeo);
        for (let i = 0; i < S.nr; i++) for (let j = 0; j < S.nc; j++) {
          const k = i * S.nc + j, p = vx(k);
          if (j < S.nc - 1) { wn.v(...p); wn.v(...vx(k + 1)); }
          if (i < S.nr - 1) { wn.v(...p); wn.v(...vx(k + S.nc)); }
          octa(wd, p, 0.036);
          wl.v(p[0], p[1], p[2]); wl.v(p[0], FLOOR, p[2]);
        }
        wn.done(); wd.done(); wl.done();
        // the outline, drawn as two crossed strips per edge so it keeps its weight from any side
        const wr = writer(rimGeo), rim = (k1, k2) => { const a = vx(k1), b = vx(k2); strip(wr, a, b, [0, 1, 0], 0.03, 0.015); strip(wr, a, b, norm(cross(sub(b, a), [0, 1, 0])), 0.03, 0.015); };
        for (let j = 0; j < S.nc - 1; j++) { rim(j, j + 1); rim((S.nr - 1) * S.nc + j, (S.nr - 1) * S.nc + j + 1); }
        for (let i = 0; i < S.nr - 1; i++) { rim(i * S.nc, (i + 1) * S.nc); rim(i * S.nc + S.nc - 1, (i + 1) * S.nc + S.nc - 1); }
        wr.done();
      }
      // contours on the sheet and on the floor, and the hachures between them on both
      {
        const wt = writer(thinGeo), wh = writer(halfGeo), ft = writer(floorThinGeo), fh = writer(floorHalfGeo);
        const wk = writer(hachGeo), fk = writer(floorHachGeo);
        const up = [0, 1, 0], fy = FLOOR + 0.002;
        for (const lv of M.LEVELS) {
          const half = lv === M.HALF, con = M.contour(z, S.tris, lv);
          con.segs.forEach(([p, q], s) => {
            const n = triNormal(S.tris[con.tri[s]]);
            const a = M.at(S, z, p), b = M.at(S, z, q);
            const A = [X(a[0]), Y(a[2]), Z(a[1])], B = [X(b[0]), Y(b[2]), Z(b[1])];
            if (Math.hypot(...sub(B, A)) < 1e-6) return;
            strip(half ? wh : wt, A, B, n, half ? 0.07 : 0.022, half ? 0.035 : 0.011);
            strip(half ? fh : ft, [A[0], fy, A[2]], [B[0], fy, B[2]], up, half ? 0.055 : 0.018, half ? 0.027 : 0.009);
          });
        }
        // a stroke's weight follows the steepness of the face it crosses, as in the old surveys
        for (const hs of M.hachureStrokes(S, z, W, D, { gap: 0.17 })) {
          for (let s = 0; s < hs.tris.length; s++) {
            const a = hs.pts[s], b = hs.pts[s + 1], T3 = S.tris[hs.tris[s]];
            const steep = Math.min(Math.atan(M.downhill(S, z, T3, W, D).slope * H) / (Math.PI / 2), 1);
            const A = [a[0] - W / 2, Y(a[2]), a[1] - D / 2], B = [b[0] - W / 2, Y(b[2]), b[1] - D / 2];
            if (Math.hypot(...sub(B, A)) < 1e-6) continue;
            const wgt = 0.006 + 0.02 * steep;
            strip(wk, A, B, triNormal(T3), wgt);
            strip(fk, [A[0], fy, A[2]], [B[0], fy, B[2]], up, wgt * 0.8);
          }
        }
        wt.done(); wh.done(); ft.done(); fh.done(); wk.done(); fk.done();
      }
      selection();
    }
    /** The selected point: its row picked out on the sheet, its dot, and its drop in the accent. */
    function selection() {
      const { i, j } = st.sel, k = i * S.nc + j, p = vx(k);
      const wr = writer(rowGeo);
      // the row's edges run across the load axis, so a strip laid with the vertical as its normal
      // spreads in depth, flat on the sheet
      for (let jj = 0; jj < S.nc - 1; jj++) strip(wr, vx(i * S.nc + jj), vx(i * S.nc + jj + 1), [0, 1, 0], 0.04, 0.02);
      wr.done();
      const ws = writer(selGeo); octa(ws, p, 0.075); ws.done();
      const wd = writer(selDropGeo); wd.v(p[0], p[1], p[2]); wd.v(p[0], FLOOR, p[2]); wd.done();
    }

    function paint() {
      const c = st.colors;
      mat.floor.color = col(c.floor); mat.floorHach.color = col(c.floorHach);
      mat.floorThin.color = col(c.floorThin); mat.floorHalf.color = col(c.accent);
      mat.thin.color = col(c.thin); mat.hach.color = col(c.hach); mat.half.color = col(c.accent); mat.row.color = col(mixc(c.sheet, c.ink, 0.85));
      mat.surf.opacity = c.dark ? 0.86 : 0.9;
      mat.rim.color = col(mixc(c.sheet, c.ink, c.dark ? 0.75 : 0.8));
      mat.dots.color = col(mixc(c.sheet, c.ink, 0.8)); mat.sel.color = col(c.accent);
      mat.net.color = col(c.net); mat.grat.color = col(c.graticule); mat.drop.color = col(c.drop);
      mat.axis.color = col(c.edge); mat.selDrop.color = col(c.accent);
    }

    // which floor edges face the camera: the load scale runs along the nearer of the two long edges,
    // the overlap scale along the nearer short edge, and the recall post stands just outside the far
    // end of the overlap scale, beside the sheet rather than in front of it
    let side = null;
    function axes() {
      const sx = Math.sin(st.az) >= 0 ? 1 : -1, sz = Math.cos(st.az) >= 0 ? 1 : -1;
      const ez = Z(sz > 0 ? 1 : 0), ex = X(sx > 0 ? 1 : 0), out = 0.42;
      const postX = ex + sx * 0.5, postZ = Z(sz > 0 ? 0 : 1) - sz * 0.3;
      return {
        sx, sz, postLeft: sx < 0,
        pTick: (P) => [X(S.uOfP(P)), FLOOR, ez + sz * out],
        rTick: (i) => [ex + sx * out, FLOOR, Z(S.v[i])],
        zTick: (r) => [postX, Y(r), postZ],
        pTitle: [[X(0), FLOOR, ez + sz * 1.25], [X(1), FLOOR, ez + sz * 1.25]],
        rTitle: [[ex + sx * 1.3, FLOOR, Z(0)], [ex + sx * 1.3, FLOOR, Z(1)]],
        post: [postX, postZ],
      };
    }
    function axisLines() {
      const a = axes(), key = `${a.sx},${a.sz}`;
      if (key === side) return;
      side = key;
      const wr = writer(axisGeo);
      const f = FLOOR;
      // the frame of the floor
      for (const [p, q] of [[[X(0), f, Z(0)], [X(1), f, Z(0)]], [[X(1), f, Z(0)], [X(1), f, Z(1)]], [[X(1), f, Z(1)], [X(0), f, Z(1)]], [[X(0), f, Z(1)], [X(0), f, Z(0)]]]) { wr.v(...p); wr.v(...q); }
      // ticks at every measured load and level
      const ez = Z(a.sz > 0 ? 1 : 0), ex = X(a.sx > 0 ? 1 : 0);
      for (const u of S.u) { wr.v(X(u), f, ez); wr.v(X(u), f, ez + a.sz * 0.16); }
      for (const v of S.v) { wr.v(ex, f, Z(v)); wr.v(ex + a.sx * 0.16, f, Z(v)); }
      // the recall post, from the floor's level of zero recall to one, ticked every 0.1
      const [px, pz] = a.post;
      wr.v(px, f, pz); wr.v(px, Y(1), pz);
      for (let k = 0; k <= 10; k++) { const y = Y(k / 10), L = k % 5 === 0 ? 0.2 : 0.1; wr.v(px, y, pz); wr.v(px + a.sx * L, y, pz); }
      wr.done();
    }

    const v3 = new T.Vector3();
    const target = new T.Vector3(0, (H + FLOOR) / 2 - 0.35, 0.35);
    function placeCamera(w, h) {
      const aspect = w / h;
      camera.aspect = aspect;
      const vf = (FOV * Math.PI) / 180, hf = 2 * Math.atan(Math.tan(vf / 2) * aspect);
      // the footprint's diagonal across, and its depth and height up the screen, at this elevation
      const across = Math.hypot(W, D) + 1.6, upward = (H - FLOOR) * Math.cos(st.el) + (Math.hypot(W, D) + 2.2) * Math.sin(st.el) + 0.9;
      const dist = Math.max(across / (2 * Math.tan(hf / 2)), upward / (2 * Math.tan(vf / 2))) * 1.02 * st.zoom;
      camera.position.set(target.x + dist * Math.cos(st.el) * Math.sin(st.az), target.y + dist * Math.sin(st.el), target.z + dist * Math.cos(st.el) * Math.cos(st.az));
      camera.lookAt(target);
      camera.near = Math.max(0.1, dist - 20); camera.far = dist + 20;
      camera.updateProjectionMatrix();
      camera.updateMatrixWorld();
    }

    let builtFor = null;
    function render(w, h, dpr) {
      if (renderer.getPixelRatio() !== dpr) renderer.setPixelRatio(dpr);
      const size = renderer.getSize(new T.Vector2());
      if (size.x !== w || size.y !== h) renderer.setSize(w, h, false);
      placeCamera(w, h);
      // rebuild what follows the heights only when they, the selection, or the inks changed
      const key = st.morph ? null : `${st.rule}:${st.noise}:${st.sel.i}:${st.sel.j}:${st.colors.dark}:${washTone}`;
      if (!key || key !== builtFor) { rebuild(); builtFor = key && `${st.rule}:${st.noise}:${st.sel.i}:${st.sel.j}:${st.colors.dark}:${washTone}`; }
      axisLines();
      renderer.render(scene, camera);
    }
    function project(x, y, z) {
      v3.set(x, y, z).project(camera);
      return { x: (v3.x * 0.5 + 0.5) * stage.clientWidth, y: (-v3.y * 0.5 + 0.5) * stage.clientHeight, ok: v3.z > -1 && v3.z < 1 };
    }

    // a ray from the camera through a point of the stage, against the sheet's triangles
    function hitSurface(o, d) {
      let best = Infinity;
      for (const T3 of S.tris) {
        const [a, b, c2] = T3.map(vx);
        const e1 = sub(b, a), e2 = sub(c2, a), p = cross(d, e2), det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
        if (Math.abs(det) < 1e-12) continue;
        const inv = 1 / det, s = sub(o, a), u = (s[0] * p[0] + s[1] * p[1] + s[2] * p[2]) * inv;
        if (u < 0 || u > 1) continue;
        const q = cross(s, e1), v = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) * inv;
        if (v < 0 || u + v > 1) continue;
        const t = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * inv;
        if (t > 1e-6 && t < best) best = t;
      }
      return best;
    }
    function rayAt(x, y) {
      const w = stage.clientWidth, h = stage.clientHeight;
      v3.set((x / w) * 2 - 1, -(y / h) * 2 + 1, 0.5).unproject(camera);
      const o = [camera.position.x, camera.position.y, camera.position.z];
      return { o, d: norm(sub([v3.x, v3.y, v3.z], o)) };
    }
    function pick(x, y) {
      const { o, d } = rayAt(x, y);
      let t = hitSurface(o, d);
      // no hit on the sheet: the floor's map picks too
      if (!Number.isFinite(t) && Math.abs(d[1]) > 1e-9) {
        const tf = (FLOOR - o[1]) / d[1];
        const fx = o[0] + d[0] * tf, fz = o[2] + d[2] * tf;
        if (tf > 0 && Math.abs(fx) <= W / 2 + 0.3 && Math.abs(fz) <= D / 2 + 0.3) t = tf;
      }
      if (Number.isFinite(t)) {
        const px = o[0] + d[0] * t, pz = o[2] + d[2] * t;
        return M.nearestCell(S, px / W + 0.5, pz / D + 0.5, W, D);
      }
      // otherwise the nearest dot on screen, if one is close
      const ptsS = [];
      for (let k = 0; k < nV; k++) { const p = project(...vx(k)); ptsS.push(p.ok ? [p.x, p.y] : null); }
      const n = M.nearestPoint(ptsS, x, y);
      return n.k >= 0 && n.d < 40 ? { i: Math.floor(n.k / S.nc), j: n.k % S.nc } : null;
    }
    function occluded(p) {
      const o = [camera.position.x, camera.position.y, camera.position.z], v = sub(p, o), dist = Math.hypot(...v);
      return hitSurface(o, norm(v)) < dist - 0.05;
    }
    function markerAt(m) {
      const [mu, mv] = M.markerUV(m);
      return [X(mu) + (m.kind === "above" ? 0.15 : 0), Y(M.HALF), Z(mv)];
    }
    paint();
    return { render, project, pick, occluded, markerAt, axes, vertex: vx, paint: () => { paint(); builtFor = null; } };
  }

  // ---------------------------------------------------------------- start
  st.colors = readColors(fig);
  setRadios(rules, st.rule); setRadios(noise, st.noise); setRadios(view, st.view);
  readout();
  const repaint = () => { st.colors = readColors(fig); st.three?.paint(); dirty(); };
  new MutationObserver(repaint).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "class", "style"] });
  darkQuery.addEventListener("change", repaint);
  reduced.addEventListener("change", () => { st.drift = !reduced.matches; dirty(); });
  document.fonts?.ready.then(dirty);
  let started = false;
  new IntersectionObserver((es) => {
    for (const e of es) if (e.isIntersecting && !started) { started = true; build(); }
  }, { rootMargin: "600px 0px" }).observe(fig);
  new IntersectionObserver((es) => { st.visible = es[0].isIntersecting; if (st.visible) { need = true; wake(); } }, { threshold: 0.02 }).observe(stage);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) { need = true; wake(); } });
  new ResizeObserver(() => dirty()).observe(stage);
}

document.querySelectorAll(`figure.plate-live[data-demo="${KIND}"]`).forEach((f) => mount(f).catch((err) => {
  const r = f.querySelector(".demo-root");
  if (r) r.textContent = "The live plate could not load its data.";
  console.error(err);
}));
