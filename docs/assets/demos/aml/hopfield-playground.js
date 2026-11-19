// Watch it settle (chapter II, live plate): the study's Hopfield network on its 18 characters, run in
// the page one neuron at a time and drawn as a split-flap departures board. Each of the 144 neurons
// is a flap that falls over when its neuron changes; a strip-chart recorder beside the board draws
// the energy, its pen stepping down at every change while the paper runs; the stored characters sit
// in a rack below, each with how many bits the board differs from it. The status line is a row of
// letter flaps.
//
// Every number and sentence comes from hopfield-playground-model.js, which the check script tests
// against the study's code and committed results.
const VQ = new URL(import.meta.url).search; // the page's version, passed on so a browser never mixes versions
const M = await import(new URL(`./hopfield-playground-model.js${VQ}`, import.meta.url).href);

const KIND = "hopfield-playground";
const reduced = matchMedia("(prefers-reduced-motion: reduce)");
const darkQuery = matchMedia("(prefers-color-scheme: dark)");
const params = new URLSearchParams(location.search);
const T_SAME = 12, T_CHANGE = 46;     // ms per visit: a visit that changes nothing, one that flips a neuron
const T_LOAD = 750, T_HOLD = 2800;    // the query flipping in; the pause after a run before the next query
const FLIP_MS = 190, CELL_MS = 62;     // one neuron's flap; one letter of the status board

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
const over = (c, bg) => mixc(bg, c, c[3]);
const css = (c, a = 1) => `rgba(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])},${a})`;
const lum = (c) => (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255;

function readColors(node) {
  const cs = getComputedStyle(node);
  const v = (k) => rgba(cs.getPropertyValue(`--${k}`).trim());
  const c = { ink: v("ink"), ink2: v("ink-2"), paper: v("paper"), paper2: v("paper-2"), rule: v("rule"), rule2: v("rule-2"), mount: v("mount"), accent: v("accent-ink") };
  c.dark = lum(c.paper) < 0.4;
  c.stage = c.paper2;
  c.rule = over(c.rule, c.stage); c.rule2 = over(c.rule2, c.stage);
  // a flap's two faces: the blank (a neuron on, +1) and the ink (off, -1), as the characters print
  c.blank = c.dark ? mixc(c.stage, c.ink, 0.1) : c.mount;
  c.mark = c.dark ? mixc(c.stage, c.ink, 0.9) : c.ink;
  c.blankEdge = c.dark ? mixc(c.stage, c.ink, 0.2) : mixc(c.mount, c.ink, 0.26);
  c.gap = c.dark ? mixc(c.stage, [0, 0, 0], 0.3) : mixc(c.stage, c.ink, 0.12);
  c.housing = c.dark ? mixc(c.stage, [0, 0, 0], 0.22) : mixc(c.stage, c.ink, 0.06);
  c.frame = mixc(c.stage, c.ink, c.dark ? 0.55 : 0.7);
  // the status board: dark flaps, pale letters, in both printings
  c.cell = c.dark ? mixc(c.stage, [0, 0, 0], 0.45) : mixc(c.ink, [0, 0, 0], 0.15);
  c.letter = c.dark ? c.ink : c.mount;
  // the recorder's paper strip
  c.strip = c.dark ? mixc(c.stage, c.ink, 0.07) : c.mount;
  c.ruleOnStrip = mixc(c.strip, c.ink, c.dark ? 0.13 : 0.1);
  c.trace = c.ink;
  c.font = cs.getPropertyValue("--type").trim() || "monospace";
  c.serif = cs.getPropertyValue("--serif").trim() || "serif";
  return c;
}

const DPR = () => Math.min(2, window.devicePixelRatio || 1);
function fit(cv, w, h) {
  const d = DPR(), W = Math.round(w * d), H = Math.round(h * d);
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  cv.style.height = `${h}px`;
  const g = cv.getContext("2d");
  g.setTransform(d, 0, 0, d, 0, 0);
  return g;
}
function rrect(g, x, y, w, h, r) {
  g.beginPath();
  g.moveTo(x + r, y); g.lineTo(x + w - r, y); g.quadraticCurveTo(x + w, y, x + w, y + r);
  g.lineTo(x + w, y + h - r); g.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  g.lineTo(x + r, y + h); g.quadraticCurveTo(x, y + h, x, y + h - r);
  g.lineTo(x, y + r); g.quadraticCurveTo(x, y, x + r, y);
  g.closePath();
}
const ease = (t) => (t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t));

// ------------------------------------------------------------------ the plate
async function mount(fig) {
  const root = fig.querySelector(".demo-root");
  const data = await (await fetch(fig.dataset.src)).json();
  const S = M.shape(data);
  const ALL = data.chars.map((_, k) => k);
  const seedParam = Number.parseInt(params.get("seed") ?? "", 10);

  const st = {
    rule: "hebb", update: "01", stored: [...ALL], setU: null,
    target: 0, source: "test", flips: M.FLIPS[10], draw: Number.isFinite(seedParam) && seedParam >= 0 ? Math.min(seedParam, 99999) : 0,
    mode: "query",
    auto: !reduced.matches && !params.has("still"), playing: false,
    run: null, phase: "idle", pos: 0, clock: 0, phaseT: 0, hold: 0,
    board: Int8Array.from(S.chars[0]), anim: new Map(),
    paper: { speed: 0.05, at: 0, feed: 0, old: null }, stamps: new Map(), tally: null,
    visible: false, colors: null, cells: [],
  };
  let net = null, netKey = "", fixed = [];
  const commonState = (() => { // the state the assignment's run reached for every one of its 54 queries
    const n = M.network("hebb", "01", S.chars);
    return M.assignmentRecall(n, S.chars[0], 0).s;
  })();

  // ---------------------------------------------------------------- controls
  const keys = (label, items, cls) => {
    const g = el("div", { class: `hp-keys ${cls}`, role: "radiogroup", "aria-label": label });
    g.append(el("span", { class: "hp-lab", "aria-hidden": "true", text: label }));
    const bank = el("div", { class: "hp-bank" });
    for (const [value, text, sub] of items) {
      const b = el("button", { type: "button", role: "radio", "data-v": String(value) }, [el("i", { class: "hp-lamp", "aria-hidden": "true" }), el("span", { text })]);
      if (sub) b.append(el("small", { text: sub }));
      bank.append(b);
    }
    g.append(bank);
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
  const setKeys = (g, value) => g.querySelectorAll("button").forEach((b) => {
    const on = b.dataset.v === String(value);
    b.setAttribute("aria-checked", String(on));
    b.tabIndex = on ? 0 : -1;
  });
  const button = (cls, text, label) => el("button", { type: "button", class: cls, ...(label ? { "aria-label": label } : {}) }, [el("span", { text })]);
  const put = (node, text) => { if (node.textContent !== text) node.textContent = text; };

  const ruleKeys = keys("Rule", [["hebb", "Hebbian"], ["storkey", "Storkey"], ["pinv", "Pseudoinverse"]], "hp-rule");
  const updKeys = keys("Update", [["01", "0/1", "the assignment's"], ["pm", "±1", "standard"]], "hp-update");

  // the machine: status board, flap board, recorder
  const statusCv = el("canvas", { class: "hp-status", "aria-hidden": "true" });
  const boardCv = el("canvas", { class: "hp-board", role: "img", "aria-label": "The network's 144 neurons" });
  const chartCv = el("canvas", { class: "hp-chart", role: "img", "aria-label": "The energy over the run" });
  const chartHead = el("p", { class: "hp-chart-head" });
  const gauges = el("dl", { class: "hp-gauges" });
  const gauge = (k, label) => {
    const dd = el("dd", { text: "–" });
    gauges.append(el("div", { class: `hp-g hp-g-${k}` }, [el("dt", { text: label }), dd]));
    return dd;
  };
  const gSweep = gauge("sweep", "Sweep"), gChanges = gauge("changes", "Changes"), gEnergy = gauge("energy", "Energy"),
    gTarget = gauge("target", "Bits from query's character"), gFixed = gauge("fixed", "Fixed points");
  const visitLine = el("p", { class: "hp-visit", "aria-hidden": "true" });
  const machine = el("div", { class: "hp-machine", tabindex: "0", "aria-label": "The network. Space plays or pauses, the right arrow steps one neuron, End settles." }, [
    statusCv,
    el("div", { class: "hp-deck" }, [
      el("div", { class: "hp-boardwrap" }, [boardCv]),
      el("div", { class: "hp-recorder" }, [chartHead, chartCv]),
    ]),
    gauges, visitLine,
  ]);

  const playB = button("hp-key hp-play", "Play"), stepB = button("hp-key", "Step", "Step one neuron"),
    settleB = button("hp-key", "Settle", "Settle at once"), againB = button("hp-key", "Again", "Run this query again");
  const drawDown = button("hp-key hp-mini", "−", "Previous draw"), drawUp = button("hp-key hp-mini", "+", "Next draw");
  const drawOut = el("output", { class: "hp-count", "aria-live": "polite" });
  const transport = el("div", { class: "hp-transport" }, [
    el("div", { class: "hp-group" }, [playB, stepB, settleB, againB]),
    el("div", { class: "hp-group hp-drawgroup", role: "group", "aria-label": "Update order and noise draw" }, [
      el("span", { class: "hp-lab", text: "Draw" }), drawDown, drawOut, drawUp]),
  ]);

  const srcKeys = keys("Query", [["test", "Test file"], ["valid", "20% file"], ["noise", "Noise"]], "hp-src");
  const range = el("input", { type: "range", min: "0", max: "72", step: "1", value: String(st.flips), "aria-label": "Pixels flipped" });
  const rangeOut = el("output", { class: "hp-rangeout" });
  const ticks = el("div", { class: "hp-ticks", "aria-hidden": "true" }, [M.FLIPS[10], M.FLIPS[20]].map((f, i) => {
    const t = el("span", { text: `${i ? 20 : 10}%` });
    t.style.left = `calc(${(f / 72) * 100}% + ${(0.5 - f / 72) * 20}px)`;
    return t;
  }));
  const noiseRow = el("div", { class: "hp-noise" }, [el("span", { class: "hp-lab", text: "Flip" }), el("div", { class: "hp-rangebox" }, [range, ticks]), rangeOut]);
  const querySay = el("p", { class: "hp-query" });

  const modeKeys = keys("Tap a character to", [["query", "Query it"], ["store", "Store or remove it"]], "hp-mode");
  const pDown = button("hp-key hp-mini", "−", "Store one fewer"), pUp = button("hp-key hp-mini", "+", "Store one more");
  const pOut = el("output", { class: "hp-count" });
  const allB = button("hp-key hp-small", "All 18"), anotherB = button("hp-key hp-small", "Another set");
  const setNote = el("p", { class: "hp-setnote" });
  const rackCv = el("canvas", { class: "hp-rackcv", "aria-hidden": "true" });
  const rackBtns = el("div", { class: "hp-rackbtns" });
  const rack = el("div", { class: "hp-rack" }, [rackCv, rackBtns]);
  const runAllB = button("hp-key hp-runall", "Run every stored query");
  const tallySay = el("p", { class: "hp-tally", "aria-live": "polite" });
  const studySay = el("p", { class: "hp-studysay" });
  const rackPanel = el("div", { class: "hp-rackpanel" }, [
    el("div", { class: "hp-rackhead" }, [
      modeKeys,
      el("div", { class: "hp-group", role: "group", "aria-label": "How many characters are stored" }, [
        el("span", { class: "hp-lab", text: "Stored" }), pDown, pOut, pUp, allB, anotherB]),
    ]),
    rack, setNote,
    el("div", { class: "hp-tallyrow" }, [runAllB, el("div", { class: "hp-tallywords" }, [tallySay, studySay])]),
  ]);

  const runSay = el("p", { class: "hp-say", "aria-live": "polite" });
  // the run's sentence is printed on a ticket; its stub carries the query's character and the draw
  const stubNo = el("b", { class: "hp-stub-no" }), stubDraw = el("span", { class: "hp-stub-draw" });
  const ticket = el("div", { class: "hp-ticket" }, [
    el("div", { class: "hp-stub", "aria-hidden": "true" }, [el("span", { class: "hp-stub-lab", text: "Query" }), stubNo, stubDraw]), runSay]);
  const notes = el("ul", { class: "hp-notes" });
  const legend = el("p", { class: "hp-legend", html:
    '<span><svg viewBox="0 0 14 14" aria-hidden="true"><rect class="k-off" x="1" y="1" width="12" height="12" rx="1.5"/></svg>off (−1), the characters\' ink</span>' +
    '<span><svg viewBox="0 0 14 14" aria-hidden="true"><rect class="k-on" x="1" y="1" width="12" height="12" rx="1.5"/></svg>on (+1)</span>' +
    '<span><svg viewBox="0 0 14 14" aria-hidden="true"><path class="k-cursor" d="M1 5V1h4M9 1h4v4M13 9v4H9M5 13H1V9"/></svg>the neuron being visited</span>' +
    '<span><svg viewBox="0 0 14 14" aria-hidden="true"><rect class="k-target" x="1.5" y="1.5" width="11" height="11"/></svg>the query\'s character</span>' +
    '<span><svg viewBox="0 0 14 14" aria-hidden="true"><circle class="k-pin" cx="7" cy="7" r="3.5"/></svg>a fixed point of these weights</span>' +
    '<span><svg viewBox="0 0 22 14" aria-hidden="true"><path class="k-dash" d="M1 7h20"/></svg>its energy, on the strip</span>' +
    '<span><svg viewBox="0 0 22 14" aria-hidden="true"><path class="k-tick" d="M12 1v12M5 4h7M5 8h7M5 11h7"/></svg>every stored character\'s energy, on the pen\'s rail</span>' +
    '<span><svg viewBox="0 0 22 14" aria-hidden="true"><text class="k-num" x="11" y="10">43</text><path class="k-under" d="M6 13h10"/></svg>under a character: how many pixels the board differs from it, underlined for the nearest stored; ✓ or × once its own query has run</span>' });
  const checked = el("p", { class: "hp-checked" });
  root.append(el("div", { class: "hp" }, [
    el("div", { class: "hp-bar" }, [ruleKeys, updKeys]),
    machine, transport,
    el("div", { class: "hp-querypanel" }, [srcKeys, noiseRow, querySay]),
    rackPanel, ticket, notes, legend, checked,
  ]));

  fetch(fig.dataset.src.replace(/\.json$/, ".checks.json")).then((r) => (r.ok ? r.json() : null)).then((c) => {
    if (c && c.line) checked.textContent = c.line;
  }).catch(() => {});

  // ---------------------------------------------------------------- the network for the current setting
  function ensureNet() {
    const key = `${st.rule}|${st.update}|${[...st.stored].sort((a, b) => a - b).join(",")}`;
    if (key !== netKey) {
      net = M.netFor(S, st.rule, st.update, st.stored);
      netKey = key;
      fixed = M.fixedPoints(S, net, st.stored);
    }
    return net;
  }
  const isStored = (k) => st.stored.includes(k);

  // ---------------------------------------------------------------- a run, one neuron at a time
  function startRun(k, { play = true } = {}) {
    ensureNet();
    st.target = k;
    const r = M.run(S, net, st, k);
    r.rule = st.rule; r.isStored = isStored(k); r.maxSweeps = data.maxSweeps;
    // the paper's speed: constant through a run, chosen so the whole run fits on the strip
    let T = 0;
    r.at = new Float64Array(r.visits.length + 1);
    r.visits.forEach((v, i) => { r.at[i] = T; T += v.changed ? T_CHANGE : T_SAME; });
    r.at[r.visits.length] = T; r.T = T;
    // the energy after each visit, and the distance to the query's character after each visit
    const tgt = S.chars[k];
    let e = r.e0, s = Int8Array.from(r.q), d = M.hamming(s, tgt);
    r.eAt = new Float64Array(r.visits.length + 1); r.dAt = new Int16Array(r.visits.length + 1); r.cAt = new Int16Array(r.visits.length + 1);
    r.eAt[0] = e; r.dAt[0] = d;
    let c = 0;
    r.visits.forEach((v, i) => {
      if (v.changed) { e = v.energy; s[v.j] = -s[v.j]; d += s[v.j] === tgt[v.j] ? -1 : 1; c++; }
      r.eAt[i + 1] = e; r.dAt[i + 1] = d; r.cAt[i + 1] = c;
    });
    // the paper feeds the last run off before this one starts
    st.paper.old = st.run && st.paper.trace ? { trace: st.paper.trace, speed: st.paper.speed } : null;
    st.paper.feed = st.paper.old && !reduced.matches ? 1 : 0;
    st.run = r; st.pos = 0; st.clock = 0;
    st.paper.trace = buildTrace(r);
    // the board flips to the query
    setBoard(r.q, reduced.matches ? 0 : 1);
    st.phase = "load"; st.phaseT = 0; st.playing = play;
    setStatus(M.status("ready", { target: k }));
    readout(); words(); drawRack();
    wake();
  }

  /** The trace in run time: one point before the run, then a step at each change. */
  function buildTrace(r) {
    const lo = Math.min(r.e1, ...r.eStored), hi = Math.max(r.e0, ...r.eStored);
    const pad = Math.max((hi - lo) * 0.14, Math.abs(hi) * 1e-6, 1e-6);
    return { r, lo: lo - pad, hi: hi + pad };
  }

  function setBoard(s, animate) {
    const now = performance.now();
    for (let i = 0; i < M.N; i++) {
      if (st.board[i] !== s[i]) {
        if (animate) st.anim.set(i, { from: st.board[i], to: s[i], t0: now + (animate === 1 ? i * 2.2 : 0) });
        st.board[i] = s[i];
      }
    }
  }

  function applyVisit() {
    const r = st.run, v = r.visits[st.pos];
    if (v.changed) {
      const from = st.board[v.j];
      st.board[v.j] = v.on ? 1 : -1;
      if (!reduced.matches) st.anim.set(v.j, { from, to: st.board[v.j], t0: performance.now() });
    }
    st.pos++;
    if (st.pos === 1 || (st.pos > 1 && r.visits[st.pos - 1].sweep !== r.visits[st.pos - 2].sweep)) setStatus(M.status("run", { sweep: v.sweep }));
    if (st.pos >= r.visits.length) finish();
  }

  function finish() {
    const r = st.run;
    st.phase = "done"; st.playing = false; st.hold = 0;
    st.stamps.set(r.target, r.kind === "recalled");
    setStatus(M.status("done", r));
    readout(); words(); drawRack(); controls();
  }

  function settleNow() {
    const r = st.run;
    if (!r) return;
    if (st.phase === "load") st.phase = "visits";
    st.paper.feed = 0;
    const now = performance.now();
    let n = 0;
    while (st.pos < r.visits.length) {
      const v = r.visits[st.pos];
      if (v.changed) {
        const from = st.board[v.j];
        st.board[v.j] = v.on ? 1 : -1;
        if (!reduced.matches) st.anim.set(v.j, { from, to: st.board[v.j], t0: now + n++ * 7 });
      }
      st.pos++;
    }
    st.clock = 0;
    finish();
    wake();
  }

  // ---------------------------------------------------------------- the status board (letter flaps)
  const DRUM = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,.·:−+/";
  // The row reads like a departures board: the query's character, where its query comes from, the
  // draw, and the status. A phone shows the first and the last.
  const GROUPS_WIDE = [["Query", 2], ["From", 4], ["Draw", 4], ["Status", 18]], GROUPS_NARROW = [["Query", 2], ["Status", 18]];
  let statusText = "", groups = GROUPS_WIDE;
  function setStatus(text) {
    if (text !== undefined) statusText = text;
    layoutCells();
    const now = performance.now();
    const t = statusLine();
    st.cells.forEach((c, i) => {
      const want = t[i] ?? " ";
      if (want === (c.queue.at(-1) ?? c.ch)) return;
      if (reduced.matches) { c.ch = want; c.queue = []; return; }
      if (c.queue.length > 2) c.queue.length = 1; // a newer line overtakes letters still on their way
      // walk the drum forward from the letter showing, at most seven letters before the one wanted
      const a = Math.max(0, DRUM.indexOf(c.queue.at(-1) ?? c.ch)), b = Math.max(0, DRUM.indexOf(want));
      let steps = (b - a + DRUM.length) % DRUM.length;
      const q = [];
      let k = steps > 7 ? (b - 7 + DRUM.length) % DRUM.length : (a + 1) % DRUM.length;
      if (steps > 7) steps = 7;
      for (let n = 0; n < steps; n++) { q.push(DRUM[k]); k = (k + 1) % DRUM.length; }
      if (!q.length || q.at(-1) !== want) q.push(want);
      if (!c.queue.length) c.t0 = now + i * 22;
      c.queue.push(...q);
    });
    wake();
  }
  function statusLine() {
    const src = st.source === "test" ? "TEST" : st.source === "valid" ? "20%" : `${st.flips}PX`;
    const text = { Query: M.pad2(st.target), From: src, Draw: String(st.draw).padStart(4, " ").slice(-4), Status: statusText };
    return groups.map(([name, n]) => text[name].padEnd(n, " ").slice(0, n)).join("");
  }
  function layoutCells() {
    const w = statusCv.clientWidth || 300;
    groups = w >= 520 ? GROUPS_WIDE : GROUPS_NARROW;
    const n = groups.reduce((a, g) => a + g[1], 0);
    if (st.cells.length !== n) {
      const old = st.cells;
      st.cells = Array.from({ length: n }, (_, i) => old[i] ?? { ch: " ", queue: [], t0: 0 });
    }
  }

  // ---------------------------------------------------------------- drawing
  let loop = 0, last = 0;
  function wake() { if (!loop && st.visible && !document.hidden) { last = performance.now(); loop = requestAnimationFrame(frame); } }
  function frame(now) {
    loop = 0;
    if (!st.visible || document.hidden) return;
    // reduced motion turned on mid-tour: reading reduced.matches here can take the change before its
    // change event fires (Chromium then skips the event), so the loop hands the plate over itself
    if (st.auto && reduced.matches) { touch(); st.anim.clear(); }
    const dt = Math.min(64, now - last); last = now;
    let busy = false;
    const r = st.run;
    if (st.paper.feed > 0) { st.paper.feed = Math.max(0, st.paper.feed - dt / 420); busy = true; }
    if (r && st.phase === "load") {
      st.phaseT += dt;
      if (st.phaseT >= (reduced.matches ? 0 : T_LOAD) && st.playing) { st.phase = "visits"; st.clock = 0; }
      busy = true;
    } else if (r && st.phase === "visits" && st.playing) {
      st.clock += dt;
      while (st.phase === "visits" && st.pos < r.visits.length && st.clock >= (r.visits[st.pos].changed ? T_CHANGE : T_SAME)) {
        st.clock -= r.visits[st.pos].changed ? T_CHANGE : T_SAME;
        applyVisit();
      }
      if (st.phase === "visits") readout();
      busy = true;
    } else if (r && st.phase === "done" && st.auto) {
      st.hold += dt;
      if (st.hold >= T_HOLD) nextAuto();
      busy = true;
    }
    drawBoard(now); drawChart(now); drawStatus(now);
    if (st.anim.size || st.cells.some((c) => c.queue.length)) busy = true;
    if (busy) loop = requestAnimationFrame(frame);
  }

  function nextAuto() {
    const s = [...st.stored].sort((a, b) => a - b);
    const i = s.indexOf(st.target);
    startRun(s[(i + 1) % s.length]);
  }

  // the board
  let boardGeo = null;
  function drawBoard(now) {
    const c = st.colors, w = boardCv.clientWidth;
    if (!w) return;
    const g = fit(boardCv, w, w);
    g.clearRect(0, 0, w, w);
    const lab = Math.max(12, Math.round(w * 0.045));
    const pad = Math.round(w * 0.022);
    const x0 = lab + pad, y0 = lab + pad, inner = w - x0 - pad;
    const pitch = inner / M.SIDE, gap = Math.max(1.5, pitch * 0.1), t = pitch - gap, rad = Math.min(3, t * 0.12);
    boardGeo = { x0, y0, pitch };
    // the housing
    rrect(g, lab, lab, w - lab, w - lab, 5);
    g.fillStyle = css(c.housing); g.fill();
    g.lineWidth = 1.5; g.strokeStyle = css(c.frame); g.stroke();
    const vis = st.run && st.phase !== "load" && st.pos > 0 && st.pos <= st.run.visits.length ? st.run.visits[Math.min(st.pos, st.run.visits.length) - 1] : null;
    const cur = vis && st.phase !== "done" ? vis.j : -1;
    // the row and column numbers, the visited neuron's in ink
    g.font = `${Math.max(8, Math.round(lab * 0.62))}px ${c.font}`;
    g.textAlign = "center"; g.textBaseline = "middle";
    for (let i = 0; i < M.SIDE; i++) {
      const on = cur >= 0 && (cur % M.SIDE === i);
      g.fillStyle = css(on ? c.ink : c.ink2, on ? 1 : 0.75);
      g.fillText(String(i), x0 + (i + 0.5) * pitch - gap / 2, lab * 0.5);
      const onr = cur >= 0 && Math.floor(cur / M.SIDE) === i;
      g.fillStyle = css(onr ? c.ink : c.ink2, onr ? 1 : 0.75);
      g.fillText(String(i), lab * 0.5, y0 + (i + 0.5) * pitch - gap / 2);
    }
    for (let i = 0; i < M.N; i++) {
      const x = x0 + (i % M.SIDE) * pitch, y = y0 + Math.floor(i / M.SIDE) * pitch;
      const a = st.anim.get(i);
      if (a && now - a.t0 >= FLIP_MS) st.anim.delete(i);
      drawFlap(g, x, y, t, rad, st.board[i], a && now >= a.t0 ? (now - a.t0) / FLIP_MS : a ? 0 : 1, a ? a.from : st.board[i]);
    }
    if (cur >= 0) {
      const x = x0 + (cur % M.SIDE) * pitch, y = y0 + Math.floor(cur / M.SIDE) * pitch, o = gap * 0.5 + 0.5, L = t * 0.32;
      g.strokeStyle = css(c.ink); g.lineWidth = Math.max(1.5, gap * 0.55); g.lineCap = "square";
      g.beginPath();
      for (const [cx, cy, sx, sy] of [[x - o, y - o, 1, 1], [x + t + o, y - o, -1, 1], [x - o, y + t + o, 1, -1], [x + t + o, y + t + o, -1, -1]]) {
        g.moveTo(cx, cy + sy * L); g.lineTo(cx, cy); g.lineTo(cx + sx * L, cy);
      }
      g.stroke();
    }
  }
  /** One flap: `to` showing, turning over from `from` (u from 0 to 1, the split-flap's fall). */
  function drawFlap(g, x, y, t, rad, to, u, from) {
    const c = st.colors, face = (v) => (v > 0 ? c.blank : c.mark), mid = y + t / 2;
    const half = (v, top, scale, shade) => {
      g.save();
      g.beginPath();
      if (top) g.rect(x - 1, y - 1, t + 2, t / 2 + 1); else g.rect(x - 1, mid, t + 2, t / 2 + 1);
      g.clip();
      g.translate(0, mid); g.scale(1, scale); g.translate(0, -mid);
      rrect(g, x, y, t, t, rad);
      g.fillStyle = css(shade ? mixc(face(v), v > 0 ? c.ink : [0, 0, 0], shade) : face(v)); g.fill();
      if (v > 0) { g.lineWidth = 1; g.strokeStyle = css(c.blankEdge); g.stroke(); }
      g.restore();
    };
    if (u >= 1 || from === to) {
      rrect(g, x, y, t, t, rad);
      g.fillStyle = css(face(to)); g.fill();
      if (to > 0) { g.lineWidth = 1; g.strokeStyle = css(c.blankEdge); g.stroke(); }
    } else {
      const k = ease(u);
      half(to, true, 1, 0);           // the new top, uncovered as the old flap falls
      half(from, false, 1, 0);        // the old bottom, until the new flap lands on it
      if (k < 0.5) half(from, true, Math.max(0.02, Math.cos(Math.PI * k)), 0.28 * k * 2);
      else half(to, false, Math.max(0.02, -Math.cos(Math.PI * k)), 0.28 * (1 - k) * 2);
    }
    // the split between the two halves
    g.fillStyle = css(c.gap);
    g.fillRect(x, mid - 0.5, t, 1);
  }

  // the status board
  function drawStatus(now) {
    const c = st.colors, w = statusCv.clientWidth;
    if (!w) return;
    layoutCells();
    const n = st.cells.length, gapU = 0.9, units = n + gapU * (groups.length - 1);
    const cw = w / units, lab = 13, fh = Math.round(Math.min(38, Math.max(26, cw * 1.42))), h = lab + fh;
    const g = fit(statusCv, w, h);
    g.clearRect(0, 0, w, h);
    const gap = Math.max(1.5, cw * 0.1), fw = cw - gap;
    // the group labels
    g.font = `9px ${c.font}`; g.textAlign = "left"; g.textBaseline = "top"; g.fillStyle = css(c.ink2);
    let x = 0, i = 0;
    const xs = [];
    groups.forEach(([name, len], gi) => {
      g.fillText(name.toUpperCase(), x + gap / 2, 1);
      for (let k = 0; k < len; k++) { xs.push(x + gap / 2); x += cw; }
      x += gapU * cw;
    });
    g.font = `${Math.round(fh * 0.58)}px ${c.font}`;
    g.textAlign = "center"; g.textBaseline = "middle";
    const y0 = lab, mid = y0 + fh / 2;
    st.cells.forEach((cell, i) => {
      const x = xs[i];
      let next = null, u = 1;
      if (cell.queue.length && now >= cell.t0) {
        u = (now - cell.t0) / CELL_MS;
        while (u >= 1 && cell.queue.length) { cell.ch = cell.queue.shift(); cell.t0 += CELL_MS; u -= 1; }
        if (cell.queue.length) next = cell.queue[0];
      }
      const face = (ch, top, scale, shade) => {
        g.save();
        g.beginPath();
        if (top) g.rect(x - 1, y0, fw + 2, fh / 2); else g.rect(x - 1, mid, fw + 2, fh / 2);
        g.clip();
        g.translate(0, mid); g.scale(1, scale); g.translate(0, -mid);
        rrect(g, x, y0 + 1, fw, fh - 2, 2.5);
        g.fillStyle = css(shade ? mixc(c.cell, [0, 0, 0], shade) : c.cell); g.fill();
        g.fillStyle = css(c.letter, shade ? 1 - shade : 1);
        g.fillText(ch, x + fw / 2, mid + 1);
        g.restore();
      };
      if (next === null) { face(cell.ch, true, 1, 0); face(cell.ch, false, 1, 0); }
      else {
        const k = Math.max(0, Math.min(1, u));
        face(next, true, 1, 0); face(cell.ch, false, 1, 0);
        if (k < 0.5) face(cell.ch, true, Math.max(0.02, Math.cos(Math.PI * k)), 0.32 * k * 2);
        else face(next, false, Math.max(0.02, -Math.cos(Math.PI * k)), 0.32 * (1 - k) * 2);
      }
      g.fillStyle = css(c.stage);
      g.fillRect(x, mid - 0.6, fw, 1.2);
    });
  }

  // the recorder
  function drawChart(now) {
    const c = st.colors, w = chartCv.clientWidth;
    // beside the board, the strip is as tall as the board; under it (on a phone), a short strip
    const side = chartCv.getBoundingClientRect().left >= boardCv.getBoundingClientRect().right - 1;
    const h = side ? Math.max(160, boardCv.clientWidth - chartHead.offsetHeight - 4) : 136;
    if (!w) return;
    const g = fit(chartCv, w, h);
    g.clearRect(0, 0, w, h);
    const r = st.run, tr = st.paper.trace;
    g.font = `11px ${c.font}`;
    const reserve = r ? Math.max(g.measureText(M.fmtE(r.e0, st.rule)).width, g.measureText(M.fmtE(r.e1, st.rule)).width) + 20 : 54;
    const top = 14, bot = h - 18, xPen = w - Math.max(reserve, Math.min(w * 0.2, 96)), left = 0;
    // the paper's run time now
    const tNow = !r ? 0 : st.phase === "load" ? 0 : st.phase === "done" ? r.T : r.at[st.pos] + Math.min(st.clock, st.pos < r.visits.length ? (r.visits[st.pos].changed ? T_CHANGE : T_SAME) : 0);
    const speed = r ? Math.min(0.11, (xPen - 64) / Math.max(r.T, 1)) : 0.05;
    const feedPx = st.paper.feed > 0 ? (1 - ease(st.paper.feed)) * w : 0;
    const paperPos = tNow * speed;
    if (st.paper.feed === 0) st.paper.speed = speed;
    // the strip
    g.fillStyle = css(c.strip);
    g.fillRect(left, top - 10, w, bot - top + 20);
    g.strokeStyle = css(c.rule2); g.lineWidth = 1;
    g.beginPath(); g.moveTo(left, top - 9.5); g.lineTo(w, top - 9.5); g.moveTo(left, bot + 9.5); g.lineTo(w, bot + 9.5); g.stroke();
    // the paper's own ruling, moving with it
    const shift = (paperPos + feedPx) % 24;
    g.strokeStyle = css(c.ruleOnStrip); g.lineWidth = 1;
    g.beginPath();
    for (let x = xPen - shift; x > left; x -= 24) { g.moveTo(Math.round(x) + 0.5, top); g.lineTo(Math.round(x) + 0.5, bot); }
    for (let x = xPen - shift + 24; x < w; x += 24) { g.moveTo(Math.round(x) + 0.5, top); g.lineTo(Math.round(x) + 0.5, bot); }
    for (let k = 0; k <= 4; k++) { const y = Math.round(top + (k / 4) * (bot - top)) + 0.5; g.moveTo(left, y); g.lineTo(w, y); }
    g.stroke();
    // the sprocket holes along both edges
    g.fillStyle = css(c.stage);
    const hs = (paperPos + feedPx) % 16;
    for (let x = w + 16 - hs; x > left - 4; x -= 16) {
      g.beginPath(); g.arc(x, top - 5, 1.8, 0, Math.PI * 2); g.fill();
      g.beginPath(); g.arc(x, bot + 5, 1.8, 0, Math.PI * 2); g.fill();
    }
    const yOf = (e, T) => bot - ((e - T.lo) / (T.hi - T.lo)) * (bot - top);
    // the last run, feeding off to the left
    if (st.paper.old && st.paper.feed > 0) drawTrace(g, st.paper.old.trace, st.paper.old.trace.r.T, xPen - feedPx, st.paper.old.speed, yOf, 0.5);
    if (r && tr && st.paper.feed === 0) {
      // the query's character's own energy, when it is stored
      if (r.eTarget !== null) {
        const y = yOf(r.eTarget, tr);
        g.save();
        g.setLineDash([5, 4]); g.strokeStyle = css(c.accent); g.lineWidth = 1.4;
        g.beginPath(); g.moveTo(left, y); g.lineTo(w, y); g.stroke();
        g.restore();
        g.font = `10px ${c.font}`; g.fillStyle = css(c.accent); g.textAlign = "left"; g.textBaseline = "bottom";
        g.fillText(`${M.pad2(r.target)} ${M.fmtE(r.eTarget, st.rule)}`, left + 4, y - 2);
      }
      drawTrace(g, tr, tNow, xPen, speed, yOf, 1);
      // sweep marks along the bottom edge
      g.font = `9px ${c.font}`; g.fillStyle = css(c.ink2); g.textAlign = "left"; g.textBaseline = "middle";
      g.strokeStyle = css(c.ink2); g.lineWidth = 1;
      for (let i = 0; i < r.visits.length; i += M.N) {
        if (r.at[i] > tNow) break;
        const x = xPen - (tNow - r.at[i]) * speed;
        if (x < left - 30) continue;
        g.beginPath(); g.moveTo(Math.round(x) + 0.5, bot + 2); g.lineTo(Math.round(x) + 0.5, bot + 9); g.stroke();
        g.fillText(`S${r.visits[i].sweep}`, x + 3, bot + 6);
      }
      // the energy now, at the pen
      const e = r.eAt[Math.min(st.pos, r.visits.length)];
      const y = yOf(st.phase === "load" ? r.e0 : e, tr);
      g.font = `11px ${c.font}`; g.fillStyle = css(c.ink); g.textAlign = "left"; g.textBaseline = "middle";
      const label = M.fmtE(st.phase === "load" ? r.e0 : e, st.rule);
      g.fillText(label, xPen + 12, Math.min(bot - 6, Math.max(top + 6, y)));
    }
    // the pen's rail, with a tick at every stored character's energy, and the pen
    g.strokeStyle = css(c.ink2); g.lineWidth = 1;
    g.beginPath(); g.moveTo(Math.round(xPen) + 0.5, top - 8); g.lineTo(Math.round(xPen) + 0.5, bot + 8); g.stroke();
    if (r && tr && st.paper.feed === 0) {
      g.beginPath();
      for (const e of r.eStored) { const y = Math.round(yOf(e, tr)) + 0.5; g.moveTo(xPen - 7, y); g.lineTo(xPen, y); }
      g.stroke();
    }
    if (r && tr) {
      const e = st.phase === "load" ? r.e0 : r.eAt[Math.min(st.pos, r.visits.length)];
      const y = st.paper.feed > 0 ? (top + bot) / 2 : yOf(e, tr);
      g.fillStyle = css(c.ink);
      g.beginPath(); g.moveTo(xPen, y); g.lineTo(xPen + 9, y - 4.5); g.lineTo(xPen + 9, y + 4.5); g.closePath(); g.fill();
    }
  }
  function drawTrace(g, tr, tNow, xPen, speed, yOf, alpha) {
    const r = tr.r, c = st.colors;
    g.save();
    g.strokeStyle = css(c.trace, alpha); g.lineWidth = 1.6; g.lineJoin = "miter";
    g.beginPath();
    let x = xPen - tNow * speed, y = yOf(r.e0, tr);
    g.moveTo(x, y);
    for (let i = 0; i < r.visits.length; i++) {
      if (r.at[i + 1] > tNow + 1e-9) break;
      const v = r.visits[i];
      if (!v.changed) continue;
      const xx = xPen - (tNow - r.at[i + 1]) * speed;
      g.lineTo(xx, y);
      y = yOf(v.energy, tr);
      g.lineTo(xx, y);
    }
    g.lineTo(xPen, y);
    g.stroke();
    // the start of the run
    g.fillStyle = css(c.trace, alpha);
    g.beginPath(); g.arc(xPen - tNow * speed, yOf(r.e0, tr), 2.2, 0, Math.PI * 2); g.fill();
    g.restore();
  }

  // the rack
  let rackGeo = null;
  function layoutRack() {
    const w = rack.clientWidth || 300;
    const cols = w >= 640 ? 18 : w >= 400 ? 9 : 6;
    const cw = w / cols, px = Math.max(2, Math.min(3, Math.floor((cw - 8) / 12 * 2) / 2)), bs = px * 12;
    const ch = bs + 38, rows = Math.ceil(ALL.length / cols);
    rackGeo = { w, cols, cw, px, bs, ch, h: rows * ch };
    rackBtns.style.height = `${rackGeo.h}px`;
    if (rackBtns.children.length !== ALL.length) {
      rackBtns.replaceChildren(...ALL.map((k) => {
        const b = el("button", { type: "button", class: "hp-rackbtn", "data-k": String(k) });
        b.addEventListener("click", () => rackTap(k));
        return b;
      }));
    }
    [...rackBtns.children].forEach((b, k) => {
      b.style.left = `${(k % cols) * cw}px`; b.style.top = `${Math.floor(k / cols) * ch}px`;
      b.style.width = `${cw}px`; b.style.height = `${ch}px`;
    });
  }
  function drawRack() {
    if (!st.colors) return;
    layoutRack();
    const c = st.colors, G = rackGeo;
    const g = fit(rackCv, G.w, G.h);
    g.clearRect(0, 0, G.w, G.h);
    const sNow = st.board, dists = ALL.map((k) => M.hamming(sNow, S.chars[k]));
    const nearest = Math.min(...st.stored.map((k) => dists[k]));
    ALL.forEach((k) => {
      const cx = (k % G.cols) * G.cw + G.cw / 2, y = Math.floor(k / G.cols) * G.ch + 8;
      const x = cx - G.bs / 2, stored = isStored(k), tgt = k === st.target;
      g.save();
      g.globalAlpha = stored ? 1 : 0.38;
      // the tile
      g.fillStyle = css(c.blank); g.fillRect(x, y, G.bs, G.bs);
      g.fillStyle = css(c.mark);
      const ch = S.chars[k];
      for (let i = 0; i < M.N; i++) if (ch[i] < 0) g.fillRect(x + (i % 12) * G.px, y + Math.floor(i / 12) * G.px, G.px, G.px);
      g.restore();
      g.lineWidth = 1;
      if (stored) { g.strokeStyle = css(c.frame); g.strokeRect(x - 0.5, y - 0.5, G.bs + 1, G.bs + 1); }
      else { g.save(); g.setLineDash([3, 3]); g.strokeStyle = css(c.rule2); g.strokeRect(x - 0.5, y - 0.5, G.bs + 1, G.bs + 1); g.restore(); }
      if (tgt) {
        g.strokeStyle = css(c.accent); g.lineWidth = 2.2;
        g.strokeRect(x - 3, y - 3, G.bs + 6, G.bs + 6);
        g.fillStyle = css(c.accent);
        g.beginPath(); g.moveTo(cx - 5, y - 8); g.lineTo(cx + 5, y - 8); g.lineTo(cx, y - 3.5); g.closePath(); g.fill();
      }
      if (stored && fixed.includes(k)) {
        g.fillStyle = css(c.ink);
        g.beginPath(); g.arc(x + G.bs + 1, y - 1, 3.2, 0, Math.PI * 2); g.fill();
        g.strokeStyle = css(c.stage); g.lineWidth = 1.2; g.stroke();
      }
      // the number, the stamp, and the distance from the board
      g.textAlign = "center"; g.textBaseline = "alphabetic";
      g.font = `11px ${c.font}`;
      const stamp = st.stamps.has(k) ? (st.stamps.get(k) ? "✓" : "×") : "";
      g.fillStyle = css(stored ? c.ink : c.ink2);
      g.fillText(`${stamp ? stamp + " " : ""}${M.pad2(k)}`, cx, y + G.bs + 14);
      const d = dists[k];
      g.font = `10px ${c.font}`;
      g.fillStyle = css(stored && d === nearest ? c.ink : c.ink2);
      g.fillText(String(d), cx, y + G.bs + 27);
      if (stored && d === nearest) { g.fillStyle = css(c.ink); g.fillRect(cx - 7, y + G.bs + 30, 14, 1.5); }
    });
  }

  // ---------------------------------------------------------------- words and readouts
  function readout() {
    const r = st.run;
    if (!r) return;
    const p = st.phase === "load" ? 0 : Math.min(st.pos, r.visits.length);
    const v = p > 0 ? r.visits[p - 1] : null;
    gSweep.textContent = v ? String(v.sweep) : "–";
    gChanges.textContent = String(r.cAt[p]);
    gEnergy.textContent = M.fmtE(r.eAt[p], st.rule);
    gTarget.textContent = String(r.dAt[p]);
    gFixed.textContent = M.sayFixed(fixed.length, st.stored.length);
    visitLine.textContent = v ? M.sayVisit(v, (p - 1) % M.N + 1, st.rule, st.update) : M.sayVisitIdle(st.phase);
    drawRackSoon();
  }
  let rackQueued = false;
  function drawRackSoon() { if (!rackQueued) { rackQueued = true; requestAnimationFrame(() => { rackQueued = false; drawRack(); }); } }

  function words() {
    const r = st.run;
    if (!r) return;
    stubNo.textContent = M.pad2(r.target);
    stubDraw.textContent = `draw ${st.draw}`;
    // the plate runs one query after another on its own: a screen reader hears the run only once the reader takes over
    runSay.setAttribute("aria-live", st.auto ? "off" : "polite");
    querySay.textContent = M.sayQuery({ target: r.target, source: st.source, flips: r.flipsIn, isStored: r.isStored, draw: st.draw });
    chartHead.textContent = st.update === "01" ? "Energy on 0/1 states, −½xᵀWx" : "Energy, −½sᵀWs";
    if (st.phase !== "done") {
      runSay.textContent = "Settling: every neuron in turn takes the sign of its field, in a fresh random order each sweep.";
      ticket.classList.add("is-quiet");
      runSay.classList.add("is-quiet");
      notes.replaceChildren();
    } else {
      runSay.classList.remove("is-quiet");
      ticket.classList.remove("is-quiet");
      runSay.textContent = M.sayRun({ ...r, changes: r.flips, e1: r.e1 });
      const sr = M.studyRun(st, ALL.length, data.draws);
      const so = data.storkeyOrders["10"];
      const ns = M.sayNotes({
        update: st.update, rule: st.rule, changes: r.flips, pmRises: r.pmRises, ties: r.ties, nearTies: r.nearTies,
        fixedFile: data.fixed.storkey, orders: so.orders, ordMin: so.fixed[0], ordMax: so.fixed[1], ordMean: M.meanStr(so.fixed[2], so.orders, 2),
        pairing: M.studyRule(st.rule, st.update) !== null,
        study: sr, target: r.target, draw: st.draw, draws: data.draws, flips: r.flipsIn,
        successes: sr ? data.byQuery[sr.qset][sr.rule][r.target] : null,
        common: st.rule === "hebb" && st.update === "01" && st.stored.length === ALL.length && M.equal(r.s, commonState), baseQueries: data.baseline.queries,
      });
      notes.replaceChildren(...ns.map((t) => el("li", { text: t })));
    }
    boardCv.setAttribute("aria-label", st.phase === "done" ? `The network's 144 neurons after the run. ${runSay.textContent}` : "The network's 144 neurons, settling.");
    chartCv.setAttribute("aria-label", `${chartHead.textContent}, from ${M.fmtE(r.e0, st.rule)}${st.phase === "done" ? ` to ${M.fmtE(r.e1, st.rule)}` : ""}.`);
  }

  function controls() {
    setKeys(ruleKeys, st.rule); setKeys(updKeys, st.update); setKeys(srcKeys, st.source); setKeys(modeKeys, st.mode);
    noiseRow.classList.toggle("is-off", st.source !== "noise");
    range.disabled = st.source !== "noise";
    range.value = String(st.flips);
    // the three outputs are live regions (an <output> is a status): each is written only when its text
    // changes, so the tour's repeated redraws stay silent
    put(rangeOut, `${st.flips} px · ${M.fmt((100 * st.flips) / M.N, 1)}%`);
    put(drawOut, String(st.draw));
    put(pOut, String(st.stored.length));
    pDown.disabled = st.stored.length <= 1; pUp.disabled = st.stored.length >= ALL.length;
    anotherB.disabled = st.stored.length >= ALL.length || st.stored.length < 2;
    allB.disabled = st.stored.length === ALL.length;
    setNote.textContent = M.saySet({ P: st.stored.length, n: ALL.length, u: st.setU, sets: data.subsetsPerSize, mode: st.mode });
    playB.querySelector("span").textContent = st.playing && st.phase !== "done" ? "Pause" : "Play";
    [...rackBtns.children].forEach((b, k) => {
      const stored = isStored(k);
      b.setAttribute("aria-label", st.mode === "query" ? `Character ${k}${stored ? "" : ", not stored"}: query it` : `Character ${k}: ${stored ? "stored; remove it" : "not stored; store it"}`);
      if (st.mode === "store") { b.setAttribute("aria-pressed", String(stored)); b.removeAttribute("aria-current"); }
      else { b.removeAttribute("aria-pressed"); if (k === st.target) b.setAttribute("aria-current", "true"); else b.removeAttribute("aria-current"); }
      b.disabled = st.mode === "store" && stored && st.stored.length <= 1;
    });
  }

  // ---------------------------------------------------------------- changes the reader makes
  // the reader takes over: the run's sentence is heard from now on (set before any text changes)
  const touch = () => { st.auto = false; runSay.setAttribute("aria-live", "polite"); };
  function changed({ keepStamps = false } = {}) {
    if (!keepStamps) { st.stamps.clear(); st.tally = null; tallySay.textContent = ""; studySay.textContent = ""; }
    ensureNet();
    controls();
    // with reduced motion, a change shows its settled state at once; Play and Again still step through it
    startRun(st.target, { play: !reduced.matches });
    if (reduced.matches) settleNow();
  }
  ruleKeys.addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b || b.dataset.v === st.rule) return; touch(); st.rule = b.dataset.v; changed(); });
  updKeys.addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b || b.dataset.v === st.update) return; touch(); st.update = b.dataset.v; changed(); });
  srcKeys.addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b || b.dataset.v === st.source) return; touch(); st.source = b.dataset.v; changed(); });
  modeKeys.addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; st.mode = b.dataset.v; controls(); });
  range.addEventListener("input", () => { touch(); st.flips = Number(range.value); changed(); });
  drawDown.addEventListener("click", () => { touch(); st.draw = Math.max(0, st.draw - 1); changed(); });
  drawUp.addEventListener("click", () => { touch(); st.draw = Math.min(99999, st.draw + 1); changed(); });
  function rackTap(k) {
    touch();
    if (st.mode === "query") { st.target = k; controls(); startRun(k, { play: !reduced.matches }); if (reduced.matches) settleNow(); return; }
    const has = isStored(k);
    if (has && st.stored.length <= 1) return;
    st.stored = has ? st.stored.filter((x) => x !== k) : [...st.stored, k].sort((a, b) => a - b);
    st.setU = null;
    changed();
  }
  const studySet = (P, u) => M.studySubset(P, u).sort((a, b) => a - b);
  pDown.addEventListener("click", () => { touch(); const P = st.stored.length - 1; if (P < 1) return; st.setU = P >= 2 ? (st.setU ?? 0) : null; st.stored = P >= 2 ? studySet(P, st.setU) : [st.target]; pickTarget(); changed(); });
  pUp.addEventListener("click", () => { touch(); const P = st.stored.length + 1; if (P > ALL.length) return; st.setU = P < ALL.length ? (st.setU ?? 0) : null; st.stored = P < ALL.length ? studySet(P, st.setU) : [...ALL]; pickTarget(); changed(); });
  anotherB.addEventListener("click", () => { touch(); const P = st.stored.length; st.setU = ((st.setU ?? -1) + 1) % data.subsetsPerSize; st.stored = studySet(P, st.setU); pickTarget(); changed(); });
  allB.addEventListener("click", () => { touch(); st.stored = [...ALL]; st.setU = null; changed(); });
  /** After the set changes by count, query a stored character. */
  function pickTarget() { if (!isStored(st.target)) st.target = st.stored[0]; }

  playB.addEventListener("click", () => {
    touch();
    if (!st.run || st.phase === "done") { startRun(st.target, { play: true }); controls(); return; }
    st.playing = !st.playing;
    if (st.playing && st.phase === "load" && st.phaseT >= T_LOAD) st.phase = "visits";
    controls(); wake();
  });
  stepB.addEventListener("click", step);
  function step() {
    touch();
    if (!st.run || st.phase === "done") return;
    st.playing = false;
    if (st.phase === "load") st.phase = "visits";
    st.clock = 0;
    applyVisit(); readout(); controls(); wake();
  }
  settleB.addEventListener("click", () => { touch(); settleNow(); controls(); });
  againB.addEventListener("click", () => { touch(); startRun(st.target, { play: true }); controls(); });
  runAllB.addEventListener("click", () => {
    touch();
    ensureNet();
    const t = M.tally(S, net, st);
    t.per.forEach((o) => st.stamps.set(o.k, o.kind === "recalled"));
    tallySay.textContent = M.sayTally({ ...t, draw: st.draw, source: st.source, flips: st.flips });
    studySay.textContent = M.sayStudy(M.studyCompare(data, st));
    drawRack();
  });
  machine.addEventListener("keydown", (e) => {
    if (e.target !== machine) return;
    if (e.key === " ") { e.preventDefault(); playB.click(); }
    else if (e.key === "ArrowRight") { e.preventDefault(); step(); }
    else if (e.key === "End") { e.preventDefault(); settleB.click(); }
  });
  // a finger on the machine itself also hands it over to the reader
  machine.addEventListener("pointerdown", touch);

  // ---------------------------------------------------------------- start
  st.colors = readColors(fig);
  ensureNet();
  controls();
  layoutCells();
  startRun(0, { play: false });
  if (reduced.matches || params.has("still")) settleNow(); // ?still holds the settled first run, for screenshots
  const swatches = () => {
    const c = st.colors;
    legend.querySelector(".k-off").style.fill = css(c.mark);
    const on = legend.querySelector(".k-on");
    on.style.fill = css(c.blank); on.style.stroke = css(c.blankEdge);
  };
  swatches();
  const repaint = () => { st.colors = readColors(fig); swatches(); drawBoard(performance.now()); drawChart(performance.now()); drawStatus(performance.now()); drawRack(); };
  new MutationObserver(repaint).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "class", "style"] });
  darkQuery.addEventListener("change", repaint);
  reduced.addEventListener("change", () => { if (reduced.matches) { touch(); st.anim.clear(); } repaint(); });
  document.fonts?.ready.then(repaint);
  new IntersectionObserver((es) => {
    st.visible = es[0].isIntersecting;
    if (st.visible) {
      // the plate starts itself the first time it is seen, unless motion is reduced
      if (st.auto && st.phase === "load" && !st.playing) { st.playing = true; controls(); }
      wake();
    }
  }, { threshold: 0.15 }).observe(machine);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) wake(); });
  new ResizeObserver(() => { repaint(); setStatus(); }).observe(root);
}

document.querySelectorAll(`figure.plate-live[data-demo="${KIND}"]`).forEach((f) => mount(f).catch((err) => {
  const r = f.querySelector(".demo-root");
  if (r) r.textContent = "The live plate could not load its data.";
  console.error(err);
}));
