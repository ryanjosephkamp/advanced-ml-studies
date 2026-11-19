// Chapter III's live plate, "Draw a matrix": a punch-card press. The reader picks one of the study's
// five random-matrix families and a target size k, and presses "Draw a matrix". A punch head sweeps
// down a printer's card and punches the new matrix into it, row by row: a filled disc for each nonzero
// entry, sized by its magnitude, in one ink for plus and another for minus, and an open ring for each
// zero. Then the 150 Iris flowers slide along straight lines from their starting view (three of the
// four measurements, scaled to [0, 1] as the study scales them) to the place the matrix sends them
// (E = X R / sqrt(k), computed in the page exactly as study/common.py computes it), inside a plain
// ruled frame drawn in parallel projection, which flattens into a plane for k = 2. A running tally of
// the reader's draws stands against its theory tick. Everything shown comes from
// projection-lottery-model.js, which the check script holds to the study's code.
const KIND = "projection-lottery";
const VER = new URL(import.meta.url).search;
const M = await import(`./${KIND}-model.js${VER}`);
const reduced = matchMedia("(prefers-reduced-motion: reduce)");
const darkMq = matchMedia("(prefers-color-scheme: dark)");

{
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = new URL(`./${KIND}.css${VER}`, import.meta.url).href;
  document.head.append(link);
}

let uid = 0;
for (const fig of document.querySelectorAll(`figure.plate-live[data-demo="${KIND}"]`)) {
  mount(fig).catch((err) => {
    const root = fig.querySelector(".demo-root");
    if (root) root.innerHTML = `<p class="pl-error">The live plate could not load (${String(err.message || err)}).</p>`;
  });
}

// timings, in milliseconds
const T = { ret: 520, hold: 140, slide: 1550, punch: 620, rest: 4600 };
const PITCH = { solid: 0.44, plane: 0.98 };
const TURN = 0.075; // radians per second while untouched
const ease = (t) => (t <= 0 ? 0 : t >= 1 ? 1 : t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
const clamp01 = (t) => Math.max(0, Math.min(1, t));
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

async function mount(fig) {
  const id = ++uid;
  const root = fig.querySelector(".demo-root");
  const src = fig.dataset.src;
  const data = await (await fetch(src)).json();
  const fams = data.families, famById = Object.fromEntries(fams.map((f) => [f.id, f]));
  const names = data.measurements, species = data.species, N = data.X.length, D = data.d, KMAX = Math.max(...data.ks);
  const Xs = M.minmax(data.X);
  // the starting view: petal length across, petal width in depth, sepal length up; sepal width is not drawn
  const SHOW = ["petal length", "petal width", "sepal length"].map((n) => names.indexOf(n));
  const HIDDEN = names.find((_, i) => !SHOW.includes(i));
  const seedParam = M.seedFromQuery(location.search);
  const fixedSeed = seedParam !== null;
  const seed = fixedSeed ? seedParam : M.freshSeed();

  root.innerHTML = `
<div class="pl">
  <div class="pl-controls">
    <fieldset class="pl-fams"><legend>Matrix family</legend>${fams.map((f) => `
      <label class="pl-chip"><input type="radio" name="pl-fam-${id}" value="${f.id}"><span>${esc(f.label)}</span></label>`).join("")}
    </fieldset>
    <div class="pl-row2">
      <fieldset class="pl-ks"><legend>Target size</legend>${data.ks.map((k) => `
        <label class="pl-chip"><input type="radio" name="pl-k-${id}" value="${k}"><span><i>k</i> = ${k}</span></label>`).join("")}
      </fieldset>
      <div class="pl-actions">
        <button class="pl-draw" type="button">Draw a matrix</button>
        <button class="pl-more" type="button">Draw 100</button>
      </div>
    </div>
    <p class="pl-def"></p>
  </div>
  <div class="pl-body">
    <div class="pl-viewcol">
      <div class="pl-view">
        <canvas class="pl-canvas" tabindex="0" role="img"></canvas>
        <p class="pl-state" aria-hidden="true"></p>
        <button class="pl-replay" type="button" aria-label="Replay the move from the starting view">↺ Replay</button>
        <p class="pl-hint" aria-hidden="true">drag to turn · tap a flower</p>
      </div>
      <p class="pl-legend"><span class="pl-key"><svg viewBox="-6 -6 12 12" aria-hidden="true"><circle r="3.6" class="pl-m0"/></svg>setosa</span><span class="pl-key"><svg viewBox="-6 -6 12 12" aria-hidden="true"><circle r="4" class="pl-m1"/></svg>versicolor</span><span class="pl-key"><svg viewBox="-6 -6 12 12" aria-hidden="true"><path d="M0 -4.9 L4.4 3.2 L-4.4 3.2 Z" class="pl-m2"/></svg>virginica</span></p>
    </div>
    <div class="pl-cardcol">
      <svg class="pl-card" role="img"></svg>
      <p class="pl-legend pl-cardkey"><span class="pl-key"><svg viewBox="-7 -7 14 14" aria-hidden="true"><circle r="5.5" class="pl-pos"/></svg>plus</span><span class="pl-key"><svg viewBox="-7 -7 14 14" aria-hidden="true"><circle r="5.5" class="pl-neg"/></svg>minus</span><span class="pl-key"><svg viewBox="-7 -7 14 14" aria-hidden="true"><circle r="4.5" class="pl-zero"/></svg>zero</span><span class="pl-keynote">area shows size</span></p>
    </div>
  </div>
  <div class="pl-say" aria-live="off"><p class="pl-say-rows"></p><p class="pl-say-cols"></p></div>
  <p class="pl-flower"></p>
  <section class="pl-tally" aria-label="Tally of draws">
    <div class="pl-tally-head"><h4 class="pl-tally-title"></h4><button class="pl-clear" type="button">Clear this tally</button></div>
    ${["zero", "both"].map((ev) => `
    <div class="pl-bar" data-ev="${ev}">
      <span class="pl-bar-name">${ev === "zero" ? "A column all zero" : "Both petal rows zero"}</span>
      <span class="pl-track" aria-hidden="true"><span class="pl-band"></span><span class="pl-fill"></span><span class="pl-theory"></span>${ev === "zero" ? '<span class="pl-study"></span>' : ""}</span>
      <span class="pl-bar-val"></span>
      <span class="pl-bar-note"></span>
    </div>`).join("")}
    <p class="pl-scale" aria-hidden="true"><span class="pl-scale-in">${[0, 25, 50, 75, 100].map((p) => `<span style="left:${p}%">${p}%</span>`).join("")}</span></p>
    <p class="pl-tally-say"></p>
    <p class="pl-formula"></p>
  </section>
  <div class="pl-notes">
    <p>Only the two ends of each move are computed. At the start, each measurement is scaled to [0, 1] over the 150 flowers (the study fits the same scaling on each draw's 105 training flowers), and ${esc(HIDDEN)} is left out because only three axes fit. At the end, the flowers are at E = X R / √<i>k</i>, as the study's code computes them. No support vector machine is trained here.</p>
    <p>As in the study, all five families come from the same two random matrices, a Gaussian G and a uniform U: the Gaussian family is G itself, and each other family takes G's signs and keeps an entry where U &lt; 1/<i>s</i>. Once a card is drawn, changing the family or <i>k</i> punches it again from the same G and U, with the chosen family's rule and the first <i>k</i> columns. Only your draws count in the tally: until you touch the plate, it draws the first card on its own, and another now and then unless motion is reduced or the address has a seed; those cards are not counted.</p>
    <p class="pl-seed"></p>
    <p class="pl-checked"></p>
  </div>
</div>`;

  const $ = (s) => root.querySelector(s);
  const view = $(".pl-view"), cv = $(".pl-canvas"), ctx = cv.getContext("2d");
  const card = $(".pl-card"), cardCol = $(".pl-cardcol");
  const stateEl = $(".pl-state"), hintEl = $(".pl-hint");

  // ---------------------------------------------------------------- state
  const st = {
    fam: famById.achlioptas3 || fams[0], k: data.ks[0], card: 0, own: 0, G: null, U: null, R: null, E: null, an: null,
    tallies: {}, yaw: 0.62, pitch: PITCH.solid, zoom: 1, userPitch: false,
    pose: null, anim: null, punchAt: 0, lastDone: 0, touched: false, visible: false, started: false, selected: -1,
  };
  const tallyOf = () => (st.tallies[`${st.fam.id}|${st.k}`] ??= { n: 0, zero: 0, both: 0 });

  // ---------------------------------------------------------------- poses: where every flower is, and the frame's labels
  const startFrame = (() => {
    const f = M.frameFor(SHOW.map((c) => Xs.map((x) => x[c])));
    return { ...f, titles: SHOW.map((c) => names[c]), dims: 3 };
  })();
  // a frame's first axis runs across the floor (world x), its second into the floor (world z), and its
  // third upright (world y), so a plane of two axes lies flat on the floor
  const WORLD = [0, 2, 1];
  const startPose = (() => {
    const pos = new Float64Array(3 * N);
    for (let i = 0; i < N; i++) for (let a = 0; a < 3; a++) pos[3 * i + WORLD[a]] = (Xs[i][SHOW[a]] - startFrame.center[a]) / startFrame.h;
    return { pos, hy: 1, pitch: PITCH.solid, labels: [{ frame: startFrame, alpha: 1 }], kind: "start" };
  })();
  function endPose(E, k) {
    const cols = Array.from({ length: k }, (_, j) => E.map((e) => e[j]));
    const f = { ...M.frameFor(cols), titles: cols.map((_, j) => `axis ${j + 1}`), dims: k };
    const pos = new Float64Array(3 * N);
    for (let i = 0; i < N; i++) for (let a = 0; a < 3; a++) pos[3 * i + WORLD[a]] = a < k ? (E[i][a] - f.center[a]) / f.h : 0;
    return { pos, hy: k === 3 ? 1 : 0, pitch: k === 3 ? PITCH.solid : PITCH.plane, labels: [{ frame: f, alpha: 1 }], kind: "end" };
  }
  function mix(A, B, e, raw) {
    const pos = new Float64Array(3 * N);
    for (let i = 0; i < pos.length; i++) pos[i] = A.pos[i] + (B.pos[i] - A.pos[i]) * e;
    const out = clamp01(1 - raw / 0.35), inn = clamp01((raw - 0.65) / 0.35);
    return { pos, hy: A.hy + (B.hy - A.hy) * e, pitch: A.pitch + (B.pitch - A.pitch) * e, kind: "moving",
      labels: [...A.labels.map((l) => ({ ...l, alpha: l.alpha * out })), ...B.labels.map((l) => ({ ...l, alpha: l.alpha * inn }))].filter((l) => l.alpha > 0.01) };
  }
  function poseAt(now) {
    const a = st.anim;
    if (!a) return st.pose;
    let t = now - a.t0;
    if (t < a.ret) return mix(a.from, startPose, ease(t / a.ret), t / a.ret);
    t -= a.ret;
    if (t < a.hold) return startPose;
    t -= a.hold;
    if (t < a.slide) return mix(startPose, a.end, ease(t / a.slide), t / a.slide);
    st.anim = null;
    st.pose = a.end;
    st.lastDone = now;
    return st.pose;
  }
  st.pose = startPose;

  // ---------------------------------------------------------------- drawing a card, applying it, and the tally
  function setCard(j) {
    st.card = j;
    ({ G: st.G, U: st.U } = M.drawCard(seed, j, D, KMAX));
  }
  function apply(animate) {
    st.R = M.projectionMatrix(st.fam, st.G, st.U, st.k);
    st.E = M.project(Xs, st.R);
    st.an = M.anatomy(st.R, data.petal_rows);
    const now = performance.now();
    const end = endPose(st.E, st.k);
    if (animate && !reduced.matches) {
      const from = poseAt(now);
      const atStart = from === startPose;
      st.anim = { t0: now, from, end, ret: atStart ? 0 : T.ret, hold: atStart ? 80 : T.hold, slide: T.slide };
      st.punchAt = now;
    } else {
      st.anim = null;
      st.pose = end;
      st.lastDone = now;
    }
    renderCard(animate && !reduced.matches);
    renderWords();
    renderTally();
    wake();
  }
  // the plate's own draws (before the reader touches it, so always cards 1 to st.own) turn over a card
  // but leave the tally alone
  function drawNew(times = 1, byReader = true) {
    if (!byReader) st.own += times;
    for (let t = 0; t < times; t++) {
      setCard(st.card + 1);
      M.addDraw(tallyOf(), M.projectionMatrix(st.fam, st.G, st.U, st.k), data.petal_rows, byReader);
    }
    apply(true);
  }

  // ---------------------------------------------------------------- colors, read from the page at draw time
  let C = {};
  function readColors() {
    const cs = getComputedStyle(fig);
    const v = (n, d) => cs.getPropertyValue(n).trim() || d;
    C = { ink: v("--ink", "#111"), ink2: v("--ink-2", "#555"), rule: v("--rule", "rgba(0,0,0,.15)"), rule2: v("--rule-2", "rgba(0,0,0,.4)"),
      accent: v("--accent-ink", "#12756a"), rubric: v("--rubric", "#2645a8"), paper: v("--paper", "#fff"),
      type: v("--type", "monospace"), serif: v("--serif", "serif") };
  }
  readColors();

  // ---------------------------------------------------------------- the frame and the cloud, on a canvas
  let screen = new Float32Array(2 * N), lastSize = "";
  function geometry(w, h, pitch) {
    const small = w < 520;
    const mx = small ? 30 : 46, my = small ? 34 : 40;
    const sx = (w - 2 * mx) / (2 * 1.42), sy = (h - 2 * my) / (2 * (Math.cos(pitch) * 0.98 + 1.42 * Math.sin(pitch)));
    const S = Math.max(40, Math.min(sx, sy)) * st.zoom;
    const cy = Math.cos(st.yaw), sy_ = Math.sin(st.yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const cx0 = w / 2, cy0 = h / 2 + (small ? 4 : 6);
    const P = (x, y, z) => {
      const xr = x * cy + z * sy_, zr = -x * sy_ + z * cy;
      return [cx0 + S * xr, cy0 - S * (y * cp - zr * sp), y * sp + zr * cp];
    };
    // the depth of a direction: how much it points toward the reader
    const toward = (dx, dy, dz) => dy * sp + (-dx * sy_ + dz * cy) * cp;
    return { P, toward, S, small };
  }
  function draw(now = performance.now()) {
    const w = view.clientWidth, h = view.clientHeight;
    if (!w || !h) return;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const size = `${w}x${h}x${dpr}`;
    if (size !== lastSize) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); lastSize = size; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const pose = poseAt(now);
    const pitch = st.userPitch ? st.pitch : pose.pitch;
    if (!st.userPitch) st.pitch = pitch;
    const g = geometry(w, h, pitch);
    drawFrame(g, pose);
    drawCloud(g, pose);
    const moving = pose.kind === "moving" || (st.anim && pose === startPose);
    stateEl.textContent = !st.card ? "Before · 3 of 4 measurements, scaled to [0, 1]" : moving ? "Moving · animation only"
      : pose.kind === "start" ? "Before · 3 of 4 measurements, scaled to [0, 1]" : `After · E = X R / √${st.k} · card ${st.card}`;
  }
  function line(P, a, b) { const p = P(...a), q = P(...b); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); }
  function drawFrame({ P, toward, small }, pose) {
    const hy = pose.hy;
    const xw = toward(1, 0, 0) < 0 ? 1 : -1, zw = toward(0, 0, 1) < 0 ? 1 : -1; // the two back walls
    const tall = hy > 0.02;
    // the box's edges
    ctx.lineWidth = 1;
    ctx.strokeStyle = C.rule2;
    ctx.beginPath();
    for (const y of tall ? [-hy, hy] : [0]) {
      line(P, [-1, y, -1], [1, y, -1]); line(P, [1, y, -1], [1, y, 1]); line(P, [1, y, 1], [-1, y, 1]); line(P, [-1, y, 1], [-1, y, -1]);
    }
    if (tall) for (const [x, z] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) line(P, [x, -hy, z], [x, hy, z]);
    ctx.stroke();
    // the rules on the floor and the back walls, one set per frame, fading as the frames change
    for (const { frame: f, alpha } of pose.labels) {
      const n = (a) => f.ticks[a].map((t) => (t - f.center[a]) / f.h).filter((v) => Math.abs(v) <= 1.0001);
      const nA = n(0), nB = n(1), nC = f.dims === 3 ? n(2) : [];
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = C.rule;
      ctx.beginPath();
      const fy = tall ? -hy : 0;
      for (const a of nA) line(P, [a, fy, -1], [a, fy, 1]);
      for (const b of nB) line(P, [-1, fy, b], [1, fy, b]);
      if (tall) {
        for (const b of nB) line(P, [xw, -hy, b], [xw, hy, b]);
        for (const a of nA) line(P, [a, -hy, zw], [a, hy, zw]);
        for (const c of nC) { line(P, [xw, c * hy, -1], [xw, c * hy, 1]); line(P, [-1, c * hy, zw], [1, c * hy, zw]); }
      }
      ctx.stroke();
      labels({ P, toward, small }, f, alpha, hy, xw, zw);
    }
    ctx.globalAlpha = 1;
  }
  function labels({ P, small }, f, alpha, hy, xw, zw) {
    const fy = hy > 0.02 ? -hy : 0;
    const ctr = P(0, fy, 0);
    const tickFont = `${small ? 9.5 : 10.5}px ${C.type}`, titleFont = `italic ${small ? 12 : 13}px ${C.serif}`;
    // the floor edge nearest the reader for each floor axis
    const edgeA = [[-1, fy, -zw], [1, fy, -zw]], edgeB = [[-xw, fy, -1], [-xw, fy, 1]];
    for (const [axis, edge] of [[0, edgeA], [1, edgeB]]) {
      const m = P(...edge[0].map((v, i) => (v + edge[1][i]) / 2));
      let ox = m[0] - ctr[0], oy = m[1] - ctr[1];
      const len = Math.hypot(ox, oy) || 1;
      ox /= len; oy /= len;
      ctx.globalAlpha = alpha;
      ctx.fillStyle = C.ink2; ctx.font = tickFont; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      for (const t of f.ticks[axis]) {
        const v = (t - f.center[axis]) / f.h;
        if (Math.abs(v) > 1.0001) continue;
        const pt = axis === 0 ? P(v, fy, -zw) : P(-xw, fy, v);
        ctx.fillText(M.tickText(t, f.step), pt[0] + ox * 13, pt[1] + oy * 12);
      }
      ctx.fillStyle = C.ink; ctx.font = titleFont;
      const along = Math.abs(ox) > Math.abs(oy) ? 44 : 29;
      ctx.fillText(f.titles[axis], m[0] + ox * along, m[1] + oy * along);
    }
    // the vertical axis, on the leftmost upright edge
    if (f.dims === 3 && hy > 0.02) {
      let best = null;
      for (const [x, z] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) { const p = P(x, 0, z); if (!best || p[0] < best.p[0]) best = { x, z, p }; }
      ctx.globalAlpha = alpha * clamp01((hy - 0.2) / 0.5);
      ctx.fillStyle = C.ink2; ctx.font = tickFont; ctx.textAlign = "right"; ctx.textBaseline = "middle";
      for (const t of f.ticks[2]) {
        const v = (t - f.center[2]) / f.h;
        if (Math.abs(v) > 1.0001) continue;
        const pt = P(best.x, v * hy, best.z);
        ctx.fillText(M.tickText(t, f.step), pt[0] - 7, pt[1]);
      }
      const top = P(best.x, hy, best.z);
      ctx.fillStyle = C.ink; ctx.font = titleFont; ctx.textAlign = "left"; ctx.textBaseline = "bottom";
      ctx.fillText(f.titles[2], Math.max(4, top[0] - 22), top[1] - 9);
    }
    ctx.globalAlpha = 1;
  }
  function drawCloud({ P, small }, pose) {
    const pts = [];
    for (let i = 0; i < N; i++) {
      const p = P(pose.pos[3 * i], pose.pos[3 * i + 1], pose.pos[3 * i + 2]);
      screen[2 * i] = p[0]; screen[2 * i + 1] = p[1];
      pts.push([p[2], i, p[0], p[1]]);
    }
    pts.sort((a, b) => a[0] - b[0]);
    const r0 = small ? 3.3 : 3.9;
    for (const [z, i, x, y] of pts) {
      const dn = clamp01((z + 1.6) / 3.2), r = r0 * (0.88 + 0.24 * dn);
      ctx.globalAlpha = 0.5 + 0.5 * dn;
      const sp = data.y[i];
      ctx.beginPath();
      if (sp === 0) { ctx.arc(x, y, r * 0.92, 0, 2 * Math.PI); ctx.strokeStyle = C.ink; ctx.lineWidth = 1.3; ctx.stroke(); }
      else if (sp === 1) { ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fillStyle = C.ink; ctx.fill(); }
      else { const t = r * 1.25; ctx.moveTo(x, y - t); ctx.lineTo(x + t * 0.9, y + t * 0.66); ctx.lineTo(x - t * 0.9, y + t * 0.66); ctx.closePath(); ctx.fillStyle = C.ink2; ctx.fill(); }
    }
    ctx.globalAlpha = 1;
    if (st.selected >= 0) {
      const x = screen[2 * st.selected], y = screen[2 * st.selected + 1];
      ctx.beginPath(); ctx.arc(x, y, r0 + 5, 0, 2 * Math.PI);
      ctx.strokeStyle = C.accent; ctx.lineWidth = 2; ctx.stroke();
    }
  }

  // ---------------------------------------------------------------- the card, as an SVG punch card
  const fmtV = (v) => (v === 0 ? "0" : (v > 0 ? "+" : "−") + Math.abs(v).toFixed(2));
  function renderCard(animate) {
    const R = st.R, an = st.an, k = st.k;
    const W = Math.round(Math.max(250, Math.min(cardCol.clientWidth || 280, 330)));
    const LW = 100, colW = Math.min(64, Math.floor((W - LW - 14) / k)), gridW = colW * k, x0 = W - 12 - gridW;
    const rail = 10, top = 26, headH = 30, RH = 50, foot = 28, cardH = headH + D * RH + foot, H = top + cardH + 4;
    const rows0 = top + headH;
    const rMax = Math.min(colW, RH) / 2 - 7;
    const hatch = `pl-hatch-${id}`;
    let s = `<defs><pattern id="${hatch}" class="pl-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6"/></pattern></defs>`;
    // the press: a rail with two guides, and the punch head that rests under it
    s += `<rect class="pl-rail" x="4" y="0" width="${W - 8}" height="${rail - 3}" rx="1.5"/>`;
    s += `<line class="pl-guide" x1="10" y1="${rail}" x2="10" y2="${top + cardH}"/><line class="pl-guide" x1="${W - 10}" y1="${rail}" x2="${W - 10}" y2="${top + cardH}"/>`;
    // the card
    s += `<path class="pl-stock" d="M${6 + 14} ${top} H${W - 6} V${top + cardH} H6 V${top + 14} Z"/>`;
    s += `<text class="pl-serial" x="22" y="${top + 19}">CARD ${String(st.card).padStart(4, "0")}</text>`;
    for (let j = 0; j < k; j++) s += `<text class="pl-head" x="${x0 + colW * (j + 0.5)}" y="${top + 19}">AXIS ${j + 1}</text>`;
    s += `<line class="pl-hair" x1="12" y1="${rows0}" x2="${W - 12}" y2="${rows0}"/>`;
    for (const j of an.zeroCols) {
      s += `<rect class="pl-zcol" x="${x0 + colW * j + 4}" y="${rows0 + 3}" width="${colW - 8}" height="${D * RH - 6}" fill="url(#${hatch})"/>`;
      s += `<text class="pl-tag" x="${x0 + colW * (j + 0.5)}" y="${rows0 + D * RH + 17}">ALL ZERO</text>`;
    }
    s += `<text class="pl-foot" x="14" y="${rows0 + D * RH + 17}">${esc(st.fam.label)}${st.fam.s ? `, s = ${st.fam.s}` : ""}</text>`;
    s += `<g class="pl-rows">`;
    for (let i = 0; i < D; i++) {
      const y = rows0 + RH * i, dropped = an.zeroRows.includes(i);
      s += `<g class="pl-r${dropped ? " is-dropped" : ""}" data-r="${i}">`;
      if (i) s += `<line class="pl-rowline" x1="12" y1="${y}" x2="${W - 12}" y2="${y}"/>`;
      if (dropped) s += `<rect class="pl-zrow" x="12" y="${y + 3}" width="${W - 24}" height="${RH - 6}" fill="url(#${hatch})"/>`;
      s += `<text class="pl-name" x="16" y="${y + 23}">${esc(names[i])}</text>`;
      if (dropped) s += `<line class="pl-strike" x1="15" y1="${y + 19}" x2="${16 + names[i].length * 6.1}" y2="${y + 19}"/><text class="pl-tag pl-tag-l" x="16" y="${y + 39}">DROPPED</text>`;
      for (let j = 0; j < k; j++) {
        const v = R[i][j], cx = x0 + colW * (j + 0.5), cy = y + 20;
        if (v === 0) s += `<g transform="translate(${cx} ${cy})"><g class="pl-hole"><circle class="pl-zero" r="5.5"/></g></g>`;
        else {
          const r = Math.max(2.2, rMax * Math.sqrt(Math.min(Math.abs(v), 3) / 3)), gl = r * 0.42;
          const glyph = r >= 7 ? `<path class="pl-glyph" d="M${-gl} 0 H${gl}${v > 0 ? ` M0 ${-gl} V${gl}` : ""}"/>` : "";
          s += `<g transform="translate(${cx} ${cy})"><g class="pl-hole"><circle class="${v > 0 ? "pl-pos" : "pl-neg"}" r="${r.toFixed(2)}"/>${glyph}</g></g>`;
        }
        s += `<text class="pl-val" x="${cx}" y="${y + 43}">${fmtV(v)}</text>`;
      }
      s += `</g>`;
    }
    // the punch head: a bar with one pin over each column
    s += `</g><g class="pl-punch"><rect x="8" y="${rail - 1}" width="${W - 16}" height="8" rx="1.5"/>`;
    for (let j = 0; j < k; j++) s += `<rect x="${x0 + colW * (j + 0.5) - 2}" y="${rail + 6}" width="4" height="6" rx="1"/>`;
    s += `</g>`;
    card.setAttribute("viewBox", `0 0 ${W} ${H}`);
    card.setAttribute("width", W); card.setAttribute("height", H);
    card.innerHTML = s;
    card.setAttribute("aria-label", `Card ${st.card}, ${st.fam.label} at k = ${k}, one row per measurement and one column per new axis. ` +
      names.map((n, i) => `${n}: ${R[i].map(fmtV).join(", ")}`).join("; ") + ".");
    if (!animate) return;
    // the punch head sweeps down the card and punches each row as it passes
    const head = card.querySelector(".pl-punch"), travel = rows0 + D * RH - rail;
    const sweep = T.punch * 0.78;
    head.animate([{ transform: "translateY(0)", opacity: 1 }, { transform: `translateY(${travel}px)`, opacity: 1, offset: 0.78 },
      { transform: `translateY(${travel}px)`, opacity: 0, offset: 0.86 }, { transform: "translateY(0)", opacity: 0, offset: 0.87 },
      { transform: "translateY(0)", opacity: 1 }], { duration: T.punch * 1.5, easing: "linear" });
    card.querySelectorAll(".pl-r").forEach((g) => {
      const i = Number(g.dataset.r), at = ((rows0 + RH * i + RH / 2 - rail) / travel) * sweep;
      g.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 70, delay: at, fill: "backwards" });
      g.querySelectorAll(".pl-hole").forEach((hole) => hole.animate([{ transform: "scale(1.45)" }, { transform: "scale(1)" }],
        { duration: 220, delay: at, easing: "cubic-bezier(.2,.8,.2,1)", fill: "backwards" }));
    });
    card.querySelectorAll(".pl-zcol, .pl-tag:not(.pl-tag-l)").forEach((el) => el.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 160, delay: sweep, fill: "backwards" }));
  }
  function blankCard() {
    st.R = Array.from({ length: D }, () => Array(st.k).fill(0));
    st.an = { zeroRows: [], zeroCols: [], rank: 0 };
    renderCard(false);
    card.querySelectorAll(".pl-val, .pl-tag").forEach((el) => el.remove());
    card.setAttribute("aria-label", "A blank card, before the first draw.");
  }

  // ---------------------------------------------------------------- words
  function renderWords() {
    $(".pl-def").textContent = M.sayFamily(st.fam);
    if (!st.card) return;
    $(".pl-say-rows").textContent = M.sayDropped(names, st.an.zeroRows);
    $(".pl-say-cols").textContent = M.sayColumns(st.k, st.an.zeroCols, st.an.rank);
    cv.setAttribute("aria-label", `The 150 flowers after card ${st.card} (${st.fam.label}, k = ${st.k}). ${M.sayColumns(st.k, st.an.zeroCols, st.an.rank)} Drag or use the arrow keys to turn the frame.`);
    renderFlower();
    // a link to the address already open would only move to the plate, so then the line says to reload
    const go = `Open the page with ${M.seedQuery(seed)}`;
    $(".pl-seed").innerHTML = esc(M.saySeed(seed, st.card, st.own, location.search === M.seedQuery(seed))).replace(go, `<a href="${M.seedQuery(seed)}#${esc(fig.id)}">${go}</a>`);
  }
  function renderFlower() {
    const el = $(".pl-flower"), i = st.selected;
    if (i < 0 || !st.E) { el.textContent = "Tap a flower to read its measurements and where the matrix puts it."; el.classList.remove("is-on"); return; }
    el.textContent = M.sayFlower(i, species[data.y[i]], data.X[i], Xs[i], st.E[i]);
    el.classList.add("is-on");
  }
  function renderTally() {
    const tl = tallyOf(), th = M.theory(st.fam.s, D, st.k), n = tl.n;
    $(".pl-tally-title").innerHTML = `Your tally · ${esc(st.fam.label)} at <i>k</i> = ${st.k} · ${n} ${n === 1 ? "draw" : "draws"}`;
    const printed = st.fam.id === "achlioptas3" && st.k === 2;
    const study = data.study_draws[st.fam.id]?.[String(st.k)];
    for (const ev of ["zero", "both"]) {
      const bar = root.querySelector(`.pl-bar[data-ev="${ev}"]`), c = tl[ev], p = th[ev];
      const band = M.binomBand(n, p);
      bar.querySelector(".pl-fill").style.width = `${n ? (100 * c) / n : 0}%`;
      bar.querySelector(".pl-theory").style.left = `${100 * p}%`;
      const b = bar.querySelector(".pl-band");
      if (band && n) { b.style.left = `${(100 * band[0]) / n}%`; b.style.width = `${(100 * (band[1] - band[0])) / n}%`; b.hidden = false; } else b.hidden = true;
      bar.querySelector(".pl-bar-val").textContent = M.sayCount(c, n);
      let note = `theory ${M.pct1(p)}% (the tick)${printed ? ", as the paper prints" : ""}`;
      if (ev === "zero" && study) {
        const s = bar.querySelector(".pl-study");
        s.style.left = `${(100 * study.zero_col) / study.n}%`;
        note += ` · the study's own draws: ${study.zero_col} of ${study.n} (◇)`;
      }
      if (n) note += `\n${M.sayBand(band, n)}`;
      bar.querySelector(".pl-bar-note").textContent = note;
    }
    $(".pl-tally-say").textContent = M.sayTally({ label: st.fam.label, k: st.k, n, zero: tl.zero, both: tl.both, pz: th.zero, pb: th.both });
    const fm = M.theoryFormulas(st.fam.s, D, st.k);
    $(".pl-formula").innerHTML = fm
      ? `Theory: each entry is zero with probability ${st.fam.s - 1}/${st.fam.s}, independently, so a zero column has probability <span class="pl-nb">${esc(fm.zero)}</span> and two zero petal rows <span class="pl-nb">${esc(fm.both)}</span>. The hatched band is the binomial law's central 95% for this many draws.`
      : `Theory: ${esc(st.fam.label)} has no zero entries, so both shares are 0%.`;
  }

  // ---------------------------------------------------------------- controls
  const famInputs = root.querySelectorAll(`input[name="pl-fam-${id}"]`), kInputs = root.querySelectorAll(`input[name="pl-k-${id}"]`);
  famInputs.forEach((inp) => { inp.checked = inp.value === st.fam.id; });
  kInputs.forEach((inp) => { inp.checked = Number(inp.value) === st.k; });
  function touch() {
    // the plate stops drawing on its own, and from now on each new card is announced
    if (!st.touched) $(".pl-say").setAttribute("aria-live", "polite");
    st.touched = true;
    hintEl.classList.add("is-quiet");
  }
  famInputs.forEach((inp) => inp.addEventListener("change", () => {
    touch(); st.fam = famById[inp.value];
    if (st.card) apply(true); else { renderWords(); renderTally(); blankCard(); }
  }));
  kInputs.forEach((inp) => inp.addEventListener("change", () => {
    touch(); st.k = Number(inp.value);
    if (st.card) apply(true); else { renderTally(); blankCard(); }
  }));
  $(".pl-draw").addEventListener("click", () => { touch(); drawNew(1); });
  $(".pl-more").addEventListener("click", () => { touch(); drawNew(100); });
  $(".pl-replay").addEventListener("click", () => { touch(); if (st.card) apply(true); });
  $(".pl-clear").addEventListener("click", () => { touch(); st.tallies[`${st.fam.id}|${st.k}`] = { n: 0, zero: 0, both: 0 }; renderTally(); });
  root.addEventListener("pointerdown", (e) => { if (!e.target.closest(".pl-view")) touch(); });
  root.addEventListener("keydown", (e) => { if (e.target !== cv) touch(); });

  // turning, zooming, and picking a flower on the canvas
  const ptrs = new Map();
  let drag = null, pinch = null;
  cv.addEventListener("pointerdown", (e) => {
    cv.setPointerCapture(e.pointerId);
    ptrs.set(e.pointerId, [e.clientX, e.clientY]);
    if (ptrs.size === 1) drag = { x: e.clientX, y: e.clientY, moved: 0, mouse: e.pointerType === "mouse" };
    if (ptrs.size === 2) { const [a, b] = [...ptrs.values()]; pinch = { d: Math.hypot(a[0] - b[0], a[1] - b[1]), zoom: st.zoom }; drag = null; }
  });
  cv.addEventListener("pointermove", (e) => {
    if (!ptrs.has(e.pointerId)) { cv.style.cursor = nearest(e) >= 0 ? "pointer" : "grab"; return; }
    ptrs.set(e.pointerId, [e.clientX, e.clientY]);
    if (pinch && ptrs.size === 2) {
      const [a, b] = [...ptrs.values()];
      st.zoom = Math.min(2.6, Math.max(0.6, (pinch.zoom * Math.hypot(a[0] - b[0], a[1] - b[1])) / pinch.d));
      touch(); draw();
      return;
    }
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.moved += Math.abs(dx) + Math.abs(dy);
    drag.x = e.clientX; drag.y = e.clientY;
    if (drag.moved > 4) {
      touch();
      st.yaw += dx * 0.009;
      if (drag.mouse && dy) { st.userPitch = true; st.pitch = Math.min(1.45, Math.max(0.06, st.pitch + dy * 0.007)); }
      draw();
    }
  });
  const up = (e) => {
    if (!ptrs.has(e.pointerId)) return;
    ptrs.delete(e.pointerId);
    if (drag && drag.moved <= 4 && e.type === "pointerup") {
      touch();
      const i = nearest(e, e.pointerType === "mouse" ? 12 : 20);
      st.selected = i === st.selected ? -1 : i;
      renderFlower(); draw();
    }
    if (ptrs.size < 2) pinch = null;
    if (!ptrs.size) drag = null;
  };
  cv.addEventListener("pointerup", up);
  cv.addEventListener("pointercancel", up);
  cv.addEventListener("wheel", (e) => {
    if (!e.ctrlKey) return; // a plain wheel scrolls the page; a pinch on a trackpad zooms
    e.preventDefault(); touch();
    st.zoom = Math.min(2.6, Math.max(0.6, st.zoom * Math.exp(-e.deltaY * 0.01)));
    draw();
  }, { passive: false });
  cv.addEventListener("keydown", (e) => {
    const k = e.key;
    if (k === "ArrowLeft" || k === "ArrowRight") st.yaw += (k === "ArrowLeft" ? -1 : 1) * 0.14;
    else if (k === "ArrowUp" || k === "ArrowDown") { st.userPitch = true; st.pitch = Math.min(1.45, Math.max(0.06, st.pitch + (k === "ArrowUp" ? 0.1 : -0.1))); }
    else if (k === "+" || k === "=") st.zoom = Math.min(2.6, st.zoom * 1.15);
    else if (k === "-" || k === "_") st.zoom = Math.max(0.6, st.zoom / 1.15);
    else if (k === "0") { st.zoom = 1; st.userPitch = false; }
    else if (k === "Escape") { st.selected = -1; renderFlower(); }
    else return;
    e.preventDefault(); touch(); draw();
  });
  function nearest(e, radius = 14) {
    const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    let best = -1, bd = radius * radius;
    for (let i = 0; i < N; i++) { const d = (screen[2 * i] - x) ** 2 + (screen[2 * i + 1] - y) ** 2; if (d < bd) { bd = d; best = i; } }
    return best;
  }

  // ---------------------------------------------------------------- alive: a slow turn and a draw now and then, until touched
  let loop = 0, last = 0;
  const auto = () => !st.touched && !reduced.matches;
  function frame(now) {
    loop = 0;
    const dt = last ? Math.min(0.05, (now - last) / 1000) : 0;
    last = now;
    if (auto()) st.yaw += TURN * dt;
    if (auto() && !fixedSeed && st.card && !st.anim && st.lastDone && now - st.lastDone > T.rest) drawNew(1, false);
    draw(now);
    if (st.visible && (st.anim || auto())) loop = requestAnimationFrame(frame);
    else last = 0;
  }
  function wake() { if (!loop && st.visible) loop = requestAnimationFrame(frame); else if (!st.visible) draw(); }
  new IntersectionObserver((es) => {
    st.visible = es[es.length - 1].isIntersecting;
    // the plate's first card, only if the reader has not touched the plate or drawn one first
    if (st.visible && !st.started) { st.started = true; if (!st.touched && !st.card) drawNew(1, false); }
    if (st.visible) wake();
  }, { threshold: 0.15 }).observe(view);
  document.addEventListener("visibilitychange", () => { last = 0; wake(); });
  let lastW = 0;
  new ResizeObserver(() => {
    const w = cardCol.clientWidth;
    if (w !== lastW) { lastW = w; if (st.card) renderCard(false); else blankCard(); }
    draw();
  }).observe(root);
  const repaint = () => { readColors(); draw(); };
  new MutationObserver(repaint).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "class", "style"] });
  darkMq.addEventListener("change", repaint);
  reduced.addEventListener("change", () => { if (reduced.matches) { st.anim = null; st.pose = st.card ? endPose(st.E, st.k) : startPose; } wake(); });
  if (document.fonts) document.fonts.load(`10px ${C.type}`).then(() => draw(), () => {});

  // first paint: the blank card and the flowers in their starting view
  blankCard();
  renderWords();
  renderTally();
  cv.setAttribute("aria-label", `The 150 flowers before any draw, on ${M.list(SHOW.map((c) => names[c]))}, each scaled to [0, 1]. Drag or use the arrow keys to turn the frame.`);
  if (reduced.matches) { st.started = true; drawNew(1, false); }
  draw();

  // the latest check run, quietly, when it exists
  fetch(src.replace(/\.json$/, ".checks.json")).then((r) => (r.ok ? r.json() : null)).then((c) => {
    if (c && c.line) $(".pl-checked").textContent = c.line;
  }).catch(() => {});
}
