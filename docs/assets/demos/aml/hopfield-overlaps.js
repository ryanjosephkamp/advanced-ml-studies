// How alike the characters are (chapter II, live plate): the 18 by 18 matrix of overlaps as a
// quilt. Each square is woven, and the denser its cloth, the more alike its two characters: thread
// width is set so the threads cover a share of the square equal to the overlap. The diagonal (each
// character with itself) is left blank, and the closest pairs carry a running stitch. Below the quilt
// runs its selvage: every pair as one knot on a line from overlap 0 to 1, with the mean and the
// overlap that density alone would give. Touch or hover a square to lay its two characters side by
// side, with the pixels where they differ cross-stitched. On its own, the quilt walks the pairs from
// most alike to least. The rows and columns can be resewn in likeness order.
//
// Drawn on 2D canvases with colors read from the page's inks at draw time, and redrawn when the
// printing changes. Every value and sentence comes from hopfield-overlaps-model.js, which the check
// script tests.
const VQ = new URL(import.meta.url).search; // the page's version, passed on so a browser never mixes versions
const M = await import(new URL(`./hopfield-overlaps-model.js${VQ}`, import.meta.url).href);

const KIND = "hopfield-overlaps";
const NS = "http://www.w3.org/2000/svg";
const reduced = matchMedia("(prefers-reduced-motion: reduce)");
const darkQuery = matchMedia("(prefers-color-scheme: dark)");
const params = new URLSearchParams(location.search);

{
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = new URL(`./${KIND}.css${VQ}`, import.meta.url).href;
  document.head.append(link);
}

const el = (tag, attrs = {}, kids = []) => {
  const svg = tag.startsWith("svg:");
  const n = svg ? document.createElementNS(NS, tag.slice(4)) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") n.textContent = v;
    else n.setAttribute(k, v);
  }
  for (const k of kids) n.append(k);
  return n;
};
const ease = (t) => (t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t));

function inks(node) {
  const cs = getComputedStyle(node);
  const v = (k) => cs.getPropertyValue(`--${k}`).trim();
  return {
    ink: v("ink"), ink2: v("ink-2"), paper: v("paper"), paper2: v("paper-2"), rule: v("rule"), rule2: v("rule-2"),
    accent: v("accent-ink"), rubric: v("rubric"),
    type: v("type") || "monospace", serif: v("serif") || "serif",
  };
}
/** The weft in the ink, the warp a lighter tone of it toward the paper. */
function threadsOf(C) {
  const ink = rgba(C.ink), paper = rgba(C.paper2), dark = 0.2126 * paper[0] + 0.7152 * paper[1] + 0.0722 * paper[2] < 110;
  return { weft: mix(ink, ink, 0), warp: mix(ink, paper, dark ? 0.42 : 0.4) };
}

/** Parse any CSS color the canvas accepts into [r, g, b, a]. */
const probe = document.createElement("canvas").getContext("2d");
function rgba(c) {
  probe.fillStyle = "#000"; probe.fillStyle = c || "#000";
  const v = probe.fillStyle;
  if (v[0] === "#") return [1, 3, 5].map((i) => parseInt(v.slice(i, i + 2), 16)).concat(1);
  const m = v.match(/[\d.]+/g).map(Number);
  return [m[0], m[1], m[2], m[3] ?? 1];
}
const mix = (a, b, t) => `rgb(${[0, 1, 2].map((i) => Math.round(a[i] + (b[i] - a[i]) * t)).join(",")})`;

/**
 * A woven square: plain weave, `n` threads each way, each `w` of the spacing wide. The weft (across)
 * is printed in the ink, the warp (down) in a lighter tone of it, and at every other crossing the weft
 * passes over, so the cloth reads as cloth. Both tones carry one meaning: the overlap.
 */
function weaveSquare(ctx, x, y, s, v, n, weft, warp) {
  const p = s / n, t = M.weave(v) * p;
  if (t <= 0.02) return;
  ctx.fillStyle = weft;
  for (let j = 0; j < n; j++) ctx.fillRect(x, y + (j + 0.5) * p - t / 2, s, t);
  ctx.fillStyle = warp;
  for (let i = 0; i < n; i++) ctx.fillRect(x + (i + 0.5) * p - t / 2, y, t, s);
  ctx.fillStyle = weft;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    if ((i + j) % 2) continue; // here the weft goes over the warp
    ctx.fillRect(x + (i + 0.5) * p - t / 2 - 0.01, y + (j + 0.5) * p - t / 2, t + 0.02, t);
  }
}

// ------------------------------------------------------------------ the plate
async function mount(fig) {
  const root = fig.querySelector(".demo-root");
  const data = await (await fetch(fig.dataset.src)).json();
  const S = M.shape(data);
  const ST = M.stats(S), STI = M.stitched(S), TOUR = M.tour(S);
  const still = reduced.matches || params.has("still");

  const st = {
    a: ST.closest.a, b: ST.closest.b, kind: "file", ord: M.order(S, "file"),
    pos: Float64Array.from({ length: S.n }, (_, i) => i), move: null, glide: null,
    auto: !still, tourAt: 0, last: performance.now(), visible: false, colors: null, geo: null,
  };

  // ---------------------------------------------------------------- the controls
  const radios = (label, items, cls) => {
    const g = el("div", { class: `ho-radios ${cls}`, role: "radiogroup", "aria-label": label });
    g.append(el("span", { class: "ho-lab", "aria-hidden": "true", text: label }));
    for (const [value, text] of items) g.append(el("button", { type: "button", role: "radio", "data-v": String(value), text }));
    g.addEventListener("keydown", (e) => {
      const d = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
      if (!d) return;
      const bs = [...g.querySelectorAll("button")], i = bs.indexOf(document.activeElement);
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
  const orderR = radios("Order", [["file", "File order"], ["likeness", "Likeness order"]], "ho-order");
  const play = el("button", { type: "button", class: "ho-play" }); // its label says what a press does, so it carries no aria-pressed

  const cv = el("canvas", { class: "ho-cv", "aria-hidden": "true" });
  const stage = el("div", {
    class: "ho-stage", tabindex: "0",
    "aria-label": "The quilt of overlaps, 18 rows by 18 columns. Arrow keys move to the next square; its pair is read out beside the quilt.",
  }, [cv]);
  const orderNote = el("p", { class: "ho-ordernote" });

  // the pair, side by side
  const U = 10, GAP = 46, CW = 24 * U + GAP;
  const cmp = el("svg:svg", { class: "ho-cmp", viewBox: `-2 -18 ${CW + 4} ${12 * U + 22}`, "aria-hidden": "true" });
  const cmpParts = [0, 1].map((side) => {
    const x0 = side ? 12 * U + GAP : 0;
    const g = el("svg:g", { transform: `translate(${x0} 0)` });
    const ink = el("svg:path", { class: "ho-cink" }), marks = el("svg:g", { class: "ho-cmarks" });
    const num = el("svg:text", { class: "ho-cnum", x: 0, y: -6 });
    let grid = "";
    for (let i = 1; i < 12; i++) grid += `M${i * U} 0V${12 * U}M0 ${i * U}H${12 * U}`;
    g.append(el("svg:rect", { class: "ho-cbody", x: 0, y: 0, width: 12 * U, height: 12 * U }),
      el("svg:path", { class: "ho-cgrid", d: grid }), ink, el("svg:rect", { class: "ho-cframe", x: 0, y: 0, width: 12 * U, height: 12 * U }), marks, num);
    cmp.append(g);
    return { ink, marks, num };
  });
  const gapCount = el("svg:text", { class: "ho-gapn", x: 12 * U + GAP / 2, y: 6 * U + 4, "text-anchor": "middle" });
  const gapLab = el("svg:text", { class: "ho-gapl", x: 12 * U + GAP / 2, y: 6 * U + 17, "text-anchor": "middle", text: "differ" });
  cmp.append(gapCount, gapLab);
  const pairHead = el("p", { class: "ho-pairhead" });
  const big = el("span", { class: "ho-big" }), frac = el("span", { class: "ho-frac" });
  const say = el("p", { class: "ho-say" });
  const pairEl = el("div", { class: "ho-pair", "aria-live": "polite" }, [
    pairHead, cmp,
    el("div", { class: "ho-figs" }, [el("span", { class: "ho-figlab", text: "Overlap" }), big, frac]),
    say,
  ]);

  const selv = el("canvas", { class: "ho-selv", role: "img", "aria-label": `${M.sayStats(ST)} ${M.sayIndep(ST)}` });
  const legend = el("div", { class: "ho-legend" });
  const stats = el("p", { class: "ho-stats", text: `${M.sayStats(ST)} ${M.sayIndep(ST)}` });
  const checked = el("p", { class: "ho-checked" });
  root.append(el("div", { class: "ho" }, [
    el("div", { class: "ho-bar" }, [orderR, play]),
    el("div", { class: "ho-body" }, [el("div", { class: "ho-quilt" }, [stage, orderNote]), pairEl]),
    selv, legend, stats, checked,
  ]));

  // the legend: the weave, the stitch, and the cross-stitch, each with its swatch
  {
    const sw = (draw) => {
      const c = el("canvas", { class: "ho-sw", width: "40", height: "40", "aria-hidden": "true" });
      c._draw = draw;
      return c;
    };
    legend.append(
      el("span", {}, [sw((ctx, C) => { const th = threadsOf(C); weaveSquare(ctx, 1, 1, 18, 0.465, 3, th.weft, th.warp); }), document.createTextNode("denser weave: more alike")]),
      el("span", {}, [sw((ctx, C) => {
        const th = threadsOf(C); weaveSquare(ctx, 1, 1, 18, 0.931, 3, th.weft, th.warp);
        ctx.strokeStyle = C.paper; ctx.lineWidth = 1.3; ctx.setLineDash([2.4, 1.8]); ctx.strokeRect(3.5, 3.5, 13, 13);
      }), document.createTextNode(`stitched: differ in ${M.STITCH} pixels or fewer`)]),
      el("span", {}, [sw((ctx, C) => {
        ctx.strokeStyle = C.rubric; ctx.lineWidth = 1.6; ctx.lineCap = "round";
        ctx.beginPath(); ctx.moveTo(5, 5); ctx.lineTo(15, 15); ctx.moveTo(15, 5); ctx.lineTo(5, 15); ctx.stroke();
      }), document.createTextNode("cross-stitch: a pixel where the two differ")]),
      el("p", { class: "ho-legnote", text: `${M.WEAVE} ${M.sayStitch(STI)}` }),
    );
  }
  const drawSwatches = () => legend.querySelectorAll(".ho-sw").forEach((c) => {
    const ctx = c.getContext("2d");
    ctx.setTransform(2, 0, 0, 2, 0, 0); ctx.clearRect(0, 0, 20, 20);
    c._draw(ctx, st.colors);
  });

  fetch(fig.dataset.src.replace(/\.json$/, ".checks.json")).then((r) => (r.ok ? r.json() : null)).then((c) => {
    if (c && c.line) checked.textContent = c.line;
  }).catch(() => {});

  // ---------------------------------------------------------------- the pair
  function readPair() {
    const p = M.pair(S, st.a, st.b), rk = M.rank(S, st.a, st.b);
    pairHead.replaceChildren(el("span", { text: `Row ${st.a} · column ${st.b}` }), el("span", { text: M.isStitched(S, st.a, st.b) ? "stitched" : "" }));
    [st.a, st.b].forEach((k, side) => {
      const part = cmpParts[side];
      let d = "";
      const bits = S.chars[k];
      for (let r = 0; r < 12; r++) {
        let c = 0;
        while (c < 12) {
          if (bits[r * 12 + c] !== 0) { c++; continue; }
          const c0 = c;
          while (c < 12 && bits[r * 12 + c] === 0) c++;
          d += `M${c0 * U} ${r * U}h${(c - c0) * U}v${U}h${-(c - c0) * U}z`;
        }
      }
      part.ink.setAttribute("d", d || "M0 0");
      part.num.textContent = `Character ${k}`;
      part.marks.replaceChildren();
      for (const i of M.diffPixels(S, st.a, st.b)) {
        const x = (i % 12) * U, y = Math.floor(i / 12) * U;
        const dd = `M${x + 2.2} ${y + 2.2}L${x + 7.8} ${y + 7.8}M${x + 7.8} ${y + 2.2}L${x + 2.2} ${y + 7.8}`;
        part.marks.append(el("svg:path", { class: "ho-xhalo", d: dd }), el("svg:path", { class: "ho-x", d: dd }));
      }
    });
    gapCount.textContent = String(p.d);
    big.textContent = p.text;
    frac.textContent = `(${p.agree} agree − ${p.d} differ) / 144`;
    say.textContent = M.sayPair(p, rk, ST);
  }

  // ---------------------------------------------------------------- the quilt's geometry
  function geometry() {
    const w = stage.clientWidth;
    const small = w < 380, n = S.n;
    // wide: each row and column labeled with its number and a small print of its character
    const cell = small ? Math.floor(((w - 22) / n) * 4) / 4 : Math.floor(((w - 20) / (n + 1)) * 4) / 4;
    const thumb = small ? 0 : Math.max(cell - 5, 0);
    const L = small ? 20 : 16 + thumb + 4, T = small ? 18 : 14 + thumb + 4;
    const seam = cell >= 20 ? 2 : 1.5;
    return { w, h: Math.ceil(T + n * cell + 3), cell, thumb, L, T, seam, small };
  }

  // ---------------------------------------------------------------- drawing
  let need = true, loop = 0;
  const dirty = () => { need = true; wake(); };
  const animating = () => Boolean(st.move || st.glide);
  function wake() { if (!loop && st.visible && !document.hidden) loop = requestAnimationFrame(frame); }
  function frame(now) {
    loop = 0;
    if (!st.visible || document.hidden) return;
    if (st.move) {
      const s = ease((now - st.move.t0) / M.MOVE_MS);
      for (let j = 0; j < S.n; j++) st.pos[j] = st.move.from[j] + (st.move.to[j] - st.move.from[j]) * s;
      if (s >= 1) st.move = null;
      need = true;
    }
    if (st.glide && now - st.glide.t0 >= 260) { st.glide = null; need = true; }
    if (st.glide) need = true;
    if (need) draw(now);
    if (animating()) loop = requestAnimationFrame(frame);
  }

  function draw(now = performance.now()) {
    need = false;
    const G = geometry();
    st.geo = G;
    if (!G.w) return;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    if (stage.style.height !== `${G.h}px`) stage.style.height = `${G.h}px`;
    const W = Math.round(G.w * dpr), H = Math.round(G.h * dpr);
    if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, G.w, G.h);
    const C = st.colors, n = S.n, { cell, L, T, seam } = G;
    const X = (j) => L + st.pos[j] * cell, Y = (a) => T + st.pos[a] * cell;

    // the chosen row and column, washed faintly in the chapter's ink
    ctx.save(); ctx.globalAlpha = 0.12; ctx.fillStyle = C.accent;
    ctx.fillRect(L, Y(st.a), n * cell, cell); ctx.fillRect(X(st.b), T, cell, n * cell);
    ctx.restore();
    // the squares
    const inner = cell - seam, nt = M.threads(inner), th = threadsOf(C);
    for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) {
      if (a === b) continue;
      weaveSquare(ctx, X(b) + seam / 2, Y(a) + seam / 2, inner, M.pair(S, a, b).value, nt, th.weft, th.warp);
    }
    // the diagonal, blank: only a hairline so the quilt's order still reads
    ctx.strokeStyle = C.rule; ctx.lineWidth = 1;
    for (let a = 0; a < n; a++) {
      const x = X(a) + seam / 2, y = Y(a) + seam / 2;
      ctx.beginPath(); ctx.moveTo(x + 0.5, y + inner - 0.5); ctx.lineTo(x + inner - 0.5, y + 0.5); ctx.stroke();
    }
    // the running stitch round the closest pairs
    ctx.save();
    ctx.strokeStyle = C.paper; ctx.lineWidth = cell >= 20 ? 1.5 : 1.2;
    const dash = cell >= 20 ? [3, 2.2] : [2.2, 1.7];
    ctx.setLineDash(dash);
    for (const p of STI.pairs) for (const [a, b] of [[p.a, p.b], [p.b, p.a]]) {
      const ins = cell >= 20 ? 3 : 2.4;
      ctx.strokeRect(X(b) + seam / 2 + ins, Y(a) + seam / 2 + ins, inner - 2 * ins, inner - 2 * ins);
    }
    ctx.restore();

    // the selection: the chosen square in the chapter's ink, its mirror dashed
    let sx = X(st.b), sy = Y(st.a);
    if (st.glide) {
      const s = ease((now - st.glide.t0) / 260);
      sx = st.glide.x + (sx - st.glide.x) * s; sy = st.glide.y + (sy - st.glide.y) * s;
    }
    st.selAt = { x: sx, y: sy };
    ctx.strokeStyle = C.accent; ctx.lineWidth = 2.2;
    ctx.strokeRect(sx - 0.6, sy - 0.6, cell + 1.2, cell + 1.2);
    ctx.save(); ctx.setLineDash([3, 2.5]); ctx.lineWidth = 1.2;
    ctx.strokeRect(X(st.a) + 0.2, Y(st.b) + 0.2, cell - 0.4, cell - 0.4);
    ctx.restore();

    // the labels: numbers, and on a wide quilt a small print of each character
    ctx.font = `400 ${G.small ? 9 : 10}px ${C.type}`;
    ctx.textBaseline = "middle";
    for (let j = 0; j < n; j++) {
      const on = j === st.a || j === st.b;
      ctx.fillStyle = on ? C.accent : C.ink2;
      ctx.textAlign = "right";
      const ty = Y(j) + cell / 2;
      const rowOn = j === st.a, colOn = j === st.b;
      ctx.fillStyle = rowOn ? C.accent : C.ink2;
      ctx.fillText(String(j), G.small ? L - 4 : 14, ty);
      ctx.textAlign = "center";
      ctx.fillStyle = colOn ? C.accent : C.ink2;
      ctx.fillText(String(j), X(j) + cell / 2, G.small ? T - 8 : 7);
      if (G.thumb) {
        thumbAt(ctx, S.chars[j], L - 4 - G.thumb, Y(j) + (cell - G.thumb) / 2, G.thumb, rowOn ? C.accent : C.ink, C);
        thumbAt(ctx, S.chars[j], X(j) + (cell - G.thumb) / 2, T - 4 - G.thumb, G.thumb, colOn ? C.accent : C.ink, C);
      }
    }
    drawSelvage();
  }

  function thumbAt(ctx, bits, x, y, s, ink, C) {
    const q = s / 12;
    ctx.fillStyle = C.rule; ctx.fillRect(x - 0.5, y - 0.5, s + 1, s + 1);
    ctx.clearRect(x, y, s, s);
    ctx.fillStyle = ink;
    for (let r = 0; r < 12; r++) for (let c = 0; c < 12; c++) {
      if (bits[r * 12 + c] === 0) ctx.fillRect(x + c * q, y + r * q, q + 0.02, q + 0.02);
    }
  }

  // ---------------------------------------------------------------- the selvage: every pair as a knot
  function drawSelvage() {
    const w = selv.clientWidth;
    if (!w) return;
    const C = st.colors, small = w < 420;
    const h = small ? 96 : 104;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    if (selv.style.height !== `${h}px`) selv.style.height = `${h}px`;
    if (selv.width !== Math.round(w * dpr) || selv.height !== Math.round(h * dpr)) { selv.width = Math.round(w * dpr); selv.height = Math.round(h * dpr); }
    const ctx = selv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const x0 = 10, x1 = w - 12, base = h - 34;
    const X = (v) => x0 + v * (x1 - x0);
    // the knots: one per pair, stacked where pairs share an overlap
    const byNum = new Map();
    for (const p of M.pairs(S)) { const k = p.num; byNum.set(k, (byNum.get(k) || []).concat([p])); }
    const k = Math.max(2.2, Math.min(3.4, (x1 - x0) / 72 * 0.95));
    for (const [num, ps] of byNum) {
      ps.forEach((p, i) => {
        const x = X(num / M.N), y = base - 3 - i * (k + 0.8);
        const sel = (p.a === st.a && p.b === st.b) || (p.a === st.b && p.b === st.a);
        ctx.fillStyle = sel ? C.accent : C.ink;
        ctx.fillRect(x - k / 2, y - k, k, k);
        if (M.isStitched(S, p.a, p.b) && !sel) { ctx.strokeStyle = C.ink; ctx.lineWidth = 0.8; ctx.strokeRect(x - k / 2 - 1.5, y - k - 1.5, k + 3, k + 3); }
      });
    }
    // the line, its scale, and its two reference marks
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x0, base + 0.5); ctx.lineTo(x1, base + 0.5); ctx.stroke();
    ctx.font = `400 ${small ? 9 : 10}px ${C.type}`; ctx.fillStyle = C.ink2; ctx.textBaseline = "top"; ctx.textAlign = "center";
    for (let t = 0; t <= 10; t += 2) {
      const x = X(t / 10);
      ctx.beginPath(); ctx.moveTo(x + 0.5, base); ctx.lineTo(x + 0.5, base + 4); ctx.stroke();
      ctx.fillText(t === 0 ? "0" : t === 10 ? "1" : `0.${t}`, x, base + 6);
    }
    const mark = (v, text, row, color) => {
      const x = X(v);
      ctx.strokeStyle = color; ctx.lineWidth = 1.2; ctx.setLineDash([2, 2]);
      ctx.beginPath(); ctx.moveTo(x + 0.5, 4); ctx.lineTo(x + 0.5, base); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = color; ctx.textBaseline = "top"; ctx.textAlign = row ? "right" : "left";
      ctx.fillText(text, row ? x - 4 : x + 4, 2);
    };
    mark(ST.indepValue, `density alone ${ST.indep}`, 1, C.ink2);
    mark(ST.mean, `mean ${ST.meanText}`, 0, C.ink);
    ctx.font = `italic 400 ${small ? 11 : 12}px ${C.serif}`; ctx.fillStyle = C.ink2; ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText(`overlap of each of the ${ST.pairs} pairs`, (x0 + x1) / 2, base + 19);
  }

  // ---------------------------------------------------------------- choosing
  function choose(a, b, by) {
    if (a === b || (a === st.a && b === st.b)) return;
    if (by !== "tour") st.auto = false;
    // the tour moves every few seconds: a screen reader hears the pair only once the reader takes over
    pairEl.setAttribute("aria-live", st.auto ? "off" : "polite");
    if (!reduced.matches && st.selAt) st.glide = { x: st.selAt.x, y: st.selAt.y, t0: performance.now() };
    st.a = a; st.b = b;
    readPair(); render(); dirty();
  }
  function setOrder(kind) {
    st.kind = kind;
    st.ord = M.order(S, kind);
    const to = new Float64Array(S.n);
    st.ord.forEach((ch, i) => { to[ch] = i; });
    if (reduced.matches) { st.pos = to; st.move = null; }
    else st.move = { from: Float64Array.from(st.pos), to, t0: performance.now() };
    render(); dirty();
  }
  function cellAt(px, py) {
    const G = st.geo;
    if (!G) return null;
    const c = Math.floor((px - G.L) / G.cell), r = Math.floor((py - G.T) / G.cell);
    if (r < 0 || c < 0 || r >= S.n || c >= S.n || r === c) return null;
    return { a: st.ord[r], b: st.ord[c] };
  }
  stage.addEventListener("pointermove", (e) => {
    if (e.pointerType !== "mouse" || st.move) return;
    const r = stage.getBoundingClientRect(), hit = cellAt(e.clientX - r.left, e.clientY - r.top);
    if (hit) choose(hit.a, hit.b, "hand");
  });
  stage.addEventListener("pointerdown", (e) => {
    if (st.move) return;
    const r = stage.getBoundingClientRect(), hit = cellAt(e.clientX - r.left, e.clientY - r.top);
    if (hit) choose(hit.a, hit.b, "hand"); else st.auto = false;
    render();
  });
  stage.addEventListener("keydown", (e) => {
    const mv = { ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [-1, 0], ArrowDown: [1, 0] }[e.key];
    if (!mv) return;
    e.preventDefault();
    const s = M.step(st.ord, st.a, st.b, mv[0], mv[1]);
    if (s.moved) choose(s.a, s.b, "keys");
    else { st.auto = false; render(); }
  });
  orderR.addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b || b.dataset.v === st.kind) return;
    st.auto = false; setOrder(b.dataset.v);
  });
  play.addEventListener("click", () => {
    st.auto = !st.auto && !reduced.matches;
    st.last = performance.now();
    render();
  });

  function render() {
    setRadios(orderR, st.kind);
    orderNote.textContent = M.sayOrder(st.kind);
    play.textContent = st.auto ? "Pause" : "Play";
    play.classList.toggle("is-on", st.auto);
    pairEl.setAttribute("aria-live", st.auto ? "off" : "polite");
    play.disabled = reduced.matches;
  }

  // ---------------------------------------------------------------- alive: the pairs from most alike to least
  setInterval(() => {
    const now = performance.now();
    if (!st.auto || !st.visible || document.hidden) { st.last = now; return; }
    if (now - st.last < M.TOUR_MS) return;
    st.last = now;
    st.tourAt = (TOUR.findIndex(([a, b]) => (a === st.a && b === st.b) || (a === st.b && b === st.a)) + 1) % TOUR.length;
    const [a, b] = TOUR[st.tourAt];
    choose(a, b, "tour");
  }, 250);

  // ---------------------------------------------------------------- start
  const repaint = () => { st.colors = inks(fig); drawSwatches(); dirty(); };
  st.colors = inks(fig);
  render(); readPair(); drawSwatches();
  new MutationObserver(repaint).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "class", "style"] });
  darkQuery.addEventListener("change", repaint);
  reduced.addEventListener("change", () => { if (reduced.matches) { st.auto = false; st.move = null; st.glide = null; } render(); dirty(); });
  document.fonts?.ready.then(dirty);
  new IntersectionObserver((es) => { st.visible = es[0].isIntersecting; if (st.visible) dirty(); }, { threshold: 0.05 }).observe(root);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) dirty(); });
  new ResizeObserver(() => dirty()).observe(stage);
}

document.querySelectorAll(`figure.plate-live[data-demo="${KIND}"]`).forEach((f) => mount(f).catch((err) => {
  const r = f.querySelector(".demo-root");
  if (r) r.textContent = "The live plate could not load its data.";
  console.error(err);
}));
