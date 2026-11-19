// A chapter's opening miniature: a quiet picture from the chapter's own data, running on its own. Each one reads the
// data file of one of the chapter's plates; the Hopfield and projection miniatures also run that plate's model code,
// and the Iris and EMD ones draw the saved data directly. It starts once it is on screen, pauses off screen and in a
// hidden tab, has a Pause button, starts paused on a still frame when the reader prefers reduced motion, and makes
// no sound.
//   iris-turn           the 150 flowers in three measurements, turning (data/iris-specimens.json)
//   hopfield-settle     the assignment's network settling each character of the 20% file (hopfield-playground)
//   projection-shuffle  Iris multiplied by a new Eq. 4 matrix every few seconds (projection-lottery)
//   emd-peel            one EEG window and the pieces EMD splits it into, drawn in turn (emd-imf)
const VQ = new URL(import.meta.url).search;
const fig = document.querySelector(".hero-visual[data-hero]");
const reduced = matchMedia("(prefers-reduced-motion: reduce)");
const darkScheme = matchMedia("(prefers-color-scheme: dark)");

if (fig) start(fig).catch((e) => { console.warn("hero:", e); });

async function start(fig) {
  const stage = fig.querySelector(".hero-stage");
  const data = await (await fetch(new URL(fig.dataset.src, location.href))).json();
  const canvas = document.createElement("canvas");
  canvas.setAttribute("aria-hidden", "true");
  const note = document.createElement("p");
  note.className = "hero-note";
  note.setAttribute("aria-hidden", "true");
  const button = document.createElement("button");
  button.type = "button";
  button.className = "hero-pause";
  stage.append(canvas, note);
  (fig.querySelector("figcaption") || fig).prepend(button);  // beside the caption, clear of the picture
  const ctx = canvas.getContext("2d");
  const make = { "iris-turn": irisTurn, "hopfield-settle": hopfieldSettle, "projection-shuffle": projectionShuffle, "emd-peel": emdPeel }[fig.dataset.hero];
  if (!make) return;
  const scene = await make(data);
  let W = 0, H = 0, dpr = 1, t0 = null, visible = false, raf = 0;
  let paused = reduced.matches, last = paused ? scene.still ?? 0 : 0;
  const ink = () => {
    const cs = getComputedStyle(stage);
    const v = (n) => cs.getPropertyValue(n).trim();
    const dark = document.documentElement.dataset.theme === "dark" || (document.documentElement.dataset.theme !== "light" && darkScheme.matches);
    return { ink: v("--ink"), ink2: v("--ink-2"), accent: v("--accent-ink"), paper: v("--paper"), paper2: v("--paper-2"), rule: v("--rule-2"), dark };
  };
  const size = () => {
    dpr = Math.min(2, devicePixelRatio || 1);
    W = stage.clientWidth; H = stage.clientHeight;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
  };
  const draw = (t) => {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const said = scene.draw(ctx, W, H, t, ink());
    if (said !== undefined) { note.hidden = !said; if (said && note.textContent !== said) note.textContent = said; }
  };
  const label = () => {
    button.textContent = paused ? "Play" : "Pause";
    button.setAttribute("aria-label", `${paused ? "Play" : "Pause"} the opening picture`);
  };
  const frame = (now) => {
    raf = 0;
    if (t0 === null) t0 = now - last * 1000;
    last = (now - t0) / 1000;
    draw(last);
    if (visible && !document.hidden && !paused) raf = requestAnimationFrame(frame);
  };
  const stop = () => { if (raf) cancelAnimationFrame(raf); raf = 0; };
  const wake = () => {
    if (paused || !visible || document.hidden) { stop(); return; }
    if (!raf) { t0 = null; raf = requestAnimationFrame(frame); }
  };
  const redraw = () => { if (!raf) draw(last); };
  button.addEventListener("click", () => { paused = !paused; label(); if (paused) { stop(); redraw(); } else wake(); });
  label();
  size();
  draw(last);
  new ResizeObserver(() => { size(); redraw(); }).observe(stage);
  new IntersectionObserver((es) => { visible = es[0].isIntersecting; wake(); }).observe(stage);
  document.addEventListener("visibilitychange", wake);
  reduced.addEventListener("change", () => { paused = reduced.matches; label(); if (paused) { stop(); redraw(); } else wake(); });
  darkScheme.addEventListener("change", redraw);
  new MutationObserver(redraw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
}

const NOTE = 42;  // px kept clear at the bottom of the stage for the note (two lines on a phone)
const ease = (x) => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x));

// ---- Chapter I: the flowers, turning -----------------------------------------------------------------------------
function irisTurn(data) {
  // petal length across, sepal length up, petal width in depth; each scaled to its own range
  const axes = [2, 0, 3];
  const lo = axes.map((a) => Math.min(...data.flowers.map((f) => f[a])));
  const hi = axes.map((a) => Math.max(...data.flowers.map((f) => f[a])));
  const pts = data.flowers.map((f) => ({ p: axes.map((a, i) => (2 * (f[a] - lo[i])) / (hi[i] - lo[i]) - 1), s: f[4] }));
  const SPECIES = { light: ["#17766b", null, "#b8541c"], dark: ["#4fb39c", null, "#ffb27d"] };
  const box = [];
  for (const a of [-1, 1]) for (const b of [-1, 1]) {
    box.push([[-1, a, b], [1, a, b]], [[a, -1, b], [a, 1, b]], [[a, b, -1], [a, b, 1]]);
  }
  const TILT = 0.32, PERSP = 0.18, SPEED = 0.22, LEGEND = 92, PAD = 10;
  const turn = (th, [x, y, z]) => {
    const cs = Math.cos(th), sn = Math.sin(th), ct = Math.cos(TILT), st = Math.sin(TILT);
    const x1 = x * cs + z * sn, z1 = -x * sn + z * cs;
    const y1 = y * ct - z1 * st, z2 = y * st + z1 * ct;
    const k = 1 / (1 + z2 * PERSP);
    return [x1 * k, y1 * k, z2];
  };
  // the farthest any flower or box corner reaches over a whole turn, at unit size: the scale keeps all of it inside
  let reachX = 0, reachY = 0;
  for (let th = 0; th < 2 * Math.PI; th += 0.02) {
    for (const q of [...pts.map((q) => q.p), ...box.flat()]) {
      const [x, y] = turn(th, q);
      reachX = Math.max(reachX, Math.abs(x)); reachY = Math.max(reachY, Math.abs(y));
    }
  }
  return {
    still: 2.2,
    draw(ctx, W, H, t, c) {
      const th = 0.5 + t * SPEED, r = 4.6;
      const R = Math.min((W - LEGEND - 2 * PAD - 2 * r) / (2 * reachX), (H - 2 * PAD - 2 * r - NOTE) / (2 * reachY));
      const cx = LEGEND + (W - LEGEND) / 2, cy = PAD + r + (H - 2 * PAD - 2 * r - NOTE) / 2;
      const proj = (q) => { const [x, y, z] = turn(th, q); return [cx + x * R, cy - y * R, z]; };
      ctx.lineWidth = 1;
      ctx.strokeStyle = c.rule;
      for (const [a, b] of box) {
        const p = proj(a), q = proj(b);
        ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke();
      }
      const inks = (c.dark ? SPECIES.dark : SPECIES.light).map((v) => v ?? c.accent);
      const order = pts.map((q) => ({ ...q, at: proj(q.p) })).sort((a, b) => b.at[2] - a.at[2]);
      for (const q of order) {
        const depth = Math.max(0, Math.min(1, (1 - q.at[2] / Math.sqrt(3)) / 2));
        ctx.globalAlpha = 0.6 + 0.4 * depth;
        ctx.fillStyle = inks[q.s];
        ctx.beginPath(); ctx.arc(q.at[0], q.at[1], 2.4 + 2.2 * depth, 0, 2 * Math.PI); ctx.fill();
      }
      ctx.globalAlpha = 1;
      ctx.font = "italic 13px 'Monograph Serif', Georgia, serif";
      data.species.forEach((name, i) => {
        ctx.fillStyle = inks[i];
        ctx.beginPath(); ctx.arc(14, 18 + i * 18, 4, 0, 2 * Math.PI); ctx.fill();
        ctx.fillStyle = c.ink2; ctx.fillText(name, 24, 22 + i * 18);
      });
      return "petal length, sepal length, and petal width, each scaled to its own range";
    },
  };
}

// ---- Chapter II: each damaged character settles -----------------------------------------------------------------
async function hopfieldSettle(data) {
  const M = await import(new URL(`./hopfield-playground-model.js${VQ}`, import.meta.url).href);
  const P = M.shape(data);
  const net = M.network("hebb", "01", P.chars);
  const runs = new Map();  // each character's run, computed when its turn comes
  const run = (k) => {
    if (!runs.has(k)) {
      const r = M.recall(net, P.valid[k], M.orderRng(M.QSET_ID.valid42, 0, k), { record: true });
      runs.set(k, { k, q: P.valid[k], visits: r.visits, sweeps: r.sweeps, on: M.onCount(r.s), recalled: M.equal(r.s, P.chars[k]) });
    }
    return runs.get(k);
  };
  const RATE = 160, SHOW = 0.9, HOLD = 1.7;  // visits a second; seconds on the query, and on the settled state
  const len = (r) => SHOW + r.visits.length / RATE + HOLD;
  let at = 0, from = 0;  // the character on show, and when its turn began
  const grid = (ctx, s, x0, y0, cell, c, cur = -1) => {
    for (let i = 0; i < M.N; i++) {
      const x = x0 + (i % M.SIDE) * cell, y = y0 + Math.floor(i / M.SIDE) * cell;
      ctx.fillStyle = s[i] > 0 ? c.paper : c.ink;
      ctx.fillRect(x, y, cell - 1, cell - 1);
    }
    ctx.strokeStyle = c.rule; ctx.lineWidth = 1;
    ctx.strokeRect(x0 - 0.5, y0 - 0.5, M.SIDE * cell, M.SIDE * cell);
    if (cur >= 0) {
      ctx.strokeStyle = c.accent; ctx.lineWidth = 2;
      ctx.strokeRect(x0 + (cur % M.SIDE) * cell - 1, y0 + Math.floor(cur / M.SIDE) * cell - 1, cell + 1, cell + 1);
    }
  };
  return {
    still: SHOW + run(0).visits.length / RATE + 0.5,
    draw(ctx, W, H, t, c) {
      if (t < from) { at = 0; from = 0; }
      let r = run(at);
      while (t - from >= len(r)) { from += len(r); at = (at + 1) % P.n; r = run(at); }
      const u = t - from;
      const s = Int8Array.from(r.q);
      const n = Math.max(0, Math.min(r.visits.length, Math.floor((u - SHOW) * RATE)));
      for (let i = 0; i < n; i++) { const v = r.visits[i]; s[v.j] = v.on ? 1 : -1; }
      const room = H - NOTE - 8;  // the grids sit above the note, which can wrap to two lines on a phone
      const big = Math.floor(Math.min(room / M.SIDE, (W * 0.5) / M.SIDE));
      const small = Math.max(4, Math.floor(big * 0.5));
      const bx = Math.round(W * 0.58 - (M.SIDE * big) / 2), by = Math.round(8 + (room - M.SIDE * big) / 2);
      const sx = Math.round(Math.max(12, bx - M.SIDE * small - 36)), sy = by;
      grid(ctx, P.chars[r.k], sx, sy, small, c);
      ctx.font = "11px 'Monograph Type', ui-monospace, monospace";
      ctx.fillStyle = c.ink2;
      ctx.fillText(`stored ${r.k + 1}`, sx, sy + M.SIDE * small + 14);
      grid(ctx, s, bx, by, big, c, n > 0 && n < r.visits.length ? r.visits[n - 1].j : -1);
      if (u < SHOW) return `character ${r.k + 1} of 18 from the 20% file, before any update`;
      if (n < r.visits.length) return `character ${r.k + 1} of 18: sweep ${r.visits[Math.max(0, n - 1)].sweep}, one neuron at a time`;
      return `character ${r.k + 1} of 18: stopped after ${r.sweeps} sweeps with ${r.on} pixels on; ${r.recalled ? "recalled" : "not recalled"}`;
    },
  };
}

// ---- Chapter III: a new matrix, a new picture --------------------------------------------------------------------
async function projectionShuffle(data) {
  const M = await import(new URL(`./projection-lottery-model.js${VQ}`, import.meta.url).href);
  const fam = data.families.find((f) => f.id === "achlioptas3");
  const Xs = M.minmax(data.X);
  const SEED = 7, EVERY = 3.4, MOVE = 1.1;
  const cache = new Map();
  const card = (j) => {
    if (!cache.has(j)) {
      const { G, U } = M.drawCard(SEED, j, data.d, 3);
      const R = M.projectionMatrix(fam, G, U, 2);
      const E = M.project(Xs, R);
      const xs = E.map((e) => e[0]), ys = E.map((e) => e[1]);
      const span = (v) => { const a = Math.min(...v), b = Math.max(...v); return a === b ? [a - 0.5, b + 0.5] : [a - (b - a) * 0.08, b + (b - a) * 0.08]; };
      cache.set(j, { E, fx: span(xs), fy: span(ys), zero: M.zeroRows(R), zc: M.zeroCols(R) });
      if (cache.size > 6) cache.delete(cache.keys().next().value);
    }
    return cache.get(j);
  };
  const names = data.measurements;
  const list = (xs) => (xs.length < 3 ? xs.join(" and ") : `${xs.slice(0, -1).join(", ")}, and ${xs[xs.length - 1]}`);
  return {
    still: MOVE + 0.2,
    draw(ctx, W, H, t, c) {
      const j = Math.floor(t / EVERY), u = ease((t - j * EVERY) / MOVE);
      const a = card(Math.max(0, j - 1)), b = card(j);
      const pad = 26, w = W - 2 * pad, h = H - 14 - NOTE;  // the frame sits above the note
      const top = 14;
      const lerp = (p, q) => p + (q - p) * (j === 0 ? 1 : u);
      const fx = [lerp(a.fx[0], b.fx[0]), lerp(a.fx[1], b.fx[1])], fy = [lerp(a.fy[0], b.fy[0]), lerp(a.fy[1], b.fy[1])];
      ctx.strokeStyle = c.rule; ctx.lineWidth = 1;
      ctx.strokeRect(pad - 0.5, top - 0.5, w, h);
      for (let i = 0; i < Xs.length; i++) {
        const ex = lerp(a.E[i][0], b.E[i][0]), ey = lerp(a.E[i][1], b.E[i][1]);
        const x = pad + ((ex - fx[0]) / (fx[1] - fx[0])) * w, y = top + h - ((ey - fy[0]) / (fy[1] - fy[0])) * h;
        const sp = data.y[i];
        ctx.beginPath();
        if (sp === 2) { ctx.moveTo(x, y - 4.4); ctx.lineTo(x + 3.9, y + 2.8); ctx.lineTo(x - 3.9, y + 2.8); ctx.closePath(); ctx.fillStyle = c.ink2; ctx.fill(); }
        else if (sp === 1) { ctx.arc(x, y, 3.4, 0, 2 * Math.PI); ctx.fillStyle = c.ink; ctx.fill(); }
        else { ctx.arc(x, y, 3.2, 0, 2 * Math.PI); ctx.strokeStyle = c.ink; ctx.lineWidth = 1.2; ctx.stroke(); }
      }
      // the note and the frame speak of a draw once its points have settled; while they move, the one before
      const settled = j === 0 || t - j * EVERY >= MOVE;
      const d = settled ? b : a, k = settled ? j : j - 1;
      const petals = data.petal_rows.every((r) => d.zero.includes(r));
      if (petals) {
        ctx.strokeStyle = c.accent; ctx.lineWidth = 3;
        ctx.strokeRect(pad - 2, top - 2, w + 3, h + 3);
      }
      const kept = names.filter((_, r) => !d.zero.includes(r));
      const what = !kept.length ? "every row is zero, so every flower lands on one point"
        : kept.length === names.length ? "keeps all four measurements"
        : `keeps only ${list(kept)}`;
      return `Eq. 4, draw ${k + 1}: ${what}${petals && kept.length ? " (both petal rows are zero)" : ""}`;
    },
  };
}

// ---- Chapter IV: a window and its pieces --------------------------------------------------------------------------
function emdPeel(data) {
  const lanes = [{ label: "signal", y: data.signal, main: true }, ...data.pieces.map((p) => ({ label: p.label, y: p.y }))];
  const n = data.t.length, STEP = 0.9, HOLD = 3.2, cycle = STEP * lanes.length + HOLD + 0.8;
  const scale = lanes.map((l) => Math.max(1e-9, ...l.y.map(Math.abs)));
  return {
    still: cycle - 1.2,
    draw(ctx, W, H, t, c) {
      const u = t % cycle;
      const left = 66, right = 12, top = 10, bottom = 22;
      const lh = (H - top - bottom) / lanes.length;
      const fade = u > cycle - 0.8 ? 1 - (u - (cycle - 0.8)) / 0.8 : 1;
      ctx.font = "11px 'Monograph Type', ui-monospace, monospace";
      lanes.forEach((l, k) => {
        const p = Math.max(0, Math.min(1, (u - k * STEP) / STEP));
        const yc = top + lh * (k + 0.5), amp = lh * 0.42 / scale[k];
        ctx.globalAlpha = fade;
        ctx.fillStyle = c.ink2;
        ctx.fillText(l.label, 6, yc + 4);
        ctx.strokeStyle = c.rule; ctx.lineWidth = 0.5;
        ctx.beginPath(); ctx.moveTo(left, yc); ctx.lineTo(W - right, yc); ctx.stroke();
        if (p <= 0) return;
        ctx.strokeStyle = l.main ? c.ink : k === lanes.length - 1 ? c.ink2 : c.accent;
        ctx.lineWidth = l.main ? 1.5 : 1.3;
        ctx.beginPath();
        const m = Math.max(2, Math.floor(n * p));
        for (let i = 0; i < m; i++) {
          const x = left + (i / (n - 1)) * (W - left - right), y = yc - l.y[i] * amp;
          if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
        }
        ctx.stroke();
      });
      ctx.globalAlpha = 1;
      ctx.fillStyle = c.ink2;
      ctx.fillText("0 s", left, H - 6);
      const end = `${(data.t[n - 1] - data.t[0] + 1 / data.sfreq).toFixed(0)} s`;
      ctx.fillText(end, W - right - ctx.measureText(end).width, H - 6);
      return "";
    },
  };
}
