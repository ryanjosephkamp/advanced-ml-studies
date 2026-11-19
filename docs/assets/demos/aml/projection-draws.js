// Chapter III's live plate "One hundred draws": a raffle. The study's 100 committed draws of the
// February pipeline (k = 2, one split, default SVMs) are 100 numbered tickets in a wire drum. "Draw
// once" spins the drum and pulls one ticket at random; it flies out and unfolds into that draw's
// table, read the way the report read its own (the best of four kernels for Eq. 3 and Eq. 4 on five
// data sets), and is stamped with what it would have concluded. On the board below, every draw is a
// ticket stub at its exact best-kernel accuracy (Eq. 3 stacked above each line, Eq. 4 below), the
// report's draw is a diamond, and the ticket in hand lights up. A running count of the reader's pulls
// sits beside the true shares in all 100. Every number comes from projection-draws-model.js, which
// reads and counts the committed file and nothing else; tools/aml/projection-draws.check.mjs holds it
// to the file and to the paper's printed numbers.
const KIND = "projection-draws";
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
    if (root) root.innerHTML = `<p class="pd-error">The live plate could not load (${String(err.message || err)}).</p>`;
  });
}

// timings, in milliseconds
const T = { up: 260, spin: 700, down: 620, rise: 330, fly: 540, fold: 200, unfold: 480, stamp: 260, first: 2600, rest: 5400 };
const SPIN = 4.4, IDLE = 0.32; // drum speeds, radians per second
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

async function mount(fig) {
  const id = ++uid;
  const root = fig.querySelector(".demo-root");
  const src = fig.dataset.src;
  const data = await (await fetch(src)).json();
  const { tickets, bySeed } = M.prepare(data);
  const DS = data.datasets, REPORT = data.report_seed, N = tickets.length;
  const shortOf = Object.fromEntries(data.kernels.map((k) => [k.id, k.short]));
  const ALL = M.tally(tickets), IRIS = M.zeroColumnCount(tickets, "iris", "eq4");
  const seedParam = M.seedFromQuery(location.search);
  const fixedSeed = seedParam !== null;
  const pullSeed = fixedSeed ? seedParam : M.freshSeed();
  // the reader's pulls, the drum's own pulls, and where a returned ticket drops back in, each its own stream
  const { reader: rng, own: ownRng, drop: dropRng } = M.streams(pullSeed);
  const R42 = M.ticketNo(REPORT);

  root.innerHTML = `
<div class="pd">
  <div class="pd-top">
    <div class="pd-drum">
      <canvas class="pd-canvas" role="img" aria-label="A wire raffle drum holding ${N} numbered tickets, one for each committed draw."></canvas>
      <p class="pd-drumcap" aria-hidden="true">${N} tickets, 00 to ${N - 1}</p>
    </div>
    <div class="pd-controls">
      <button class="pd-draw" type="button"><span>Draw once</span></button>
      <button class="pd-more" type="button">Draw 20</button>
      <button class="pd-report" type="button">Ticket ${R42}, the report’s</button>
      <p class="pd-hint">or tap any stub on the board</p>
      <p class="pd-mini" aria-live="off"></p>
    </div>
    <div class="pd-ticketcol">
      <div class="pd-ticket">
        <div class="pd-stub" aria-hidden="true"><span class="pd-no-l">No.</span><span class="pd-no">${R42}</span><span class="pd-stubk"><i>k</i> = ${data.k}</span></div>
        <div class="pd-body">
          <p class="pd-thead"></p>
          <table class="pd-table">
            <thead><tr><th scope="col">Data set</th><th scope="col">Eq. 3</th><th scope="col">Eq. 4</th><th scope="col">Eq. 4 − Eq. 3</th></tr></thead>
            <tbody>${DS.map((d) => `<tr data-ds="${d.id}"><th scope="row">${esc(d.label)}</th><td data-eq="eq3"></td><td data-eq="eq4"></td><td class="pd-diff"></td></tr>`).join("")}</tbody>
          </table>
          <p class="pd-unit">test accuracy, %, best of four kernels</p>
          <div class="pd-stamp"><b></b><span></span></div>
        </div>
      </div>
      <div class="pd-say" aria-live="off"><p class="pd-leads"></p><p class="pd-verdict"></p></div>
      <p class="pd-zeros"></p>
    </div>
  </div>
  <section class="pd-board" aria-label="The board: all ${N} draws">
    <div class="pd-boardhead">
      <h4 class="pd-h">The board · all ${N} tickets</h4>
      <p class="pd-legend">
        <span class="pd-key"><svg viewBox="0 0 12 6" aria-hidden="true"><rect class="pd-s3" x="1" y="1.5" width="10" height="3"/></svg>Eq. 3, above each line</span>
        <span class="pd-key"><svg viewBox="0 0 12 6" aria-hidden="true"><rect class="pd-s4" x="1" y="1.5" width="10" height="3"/></svg>Eq. 4, below</span>
        <span class="pd-key"><svg viewBox="0 0 12 6" aria-hidden="true"><rect class="pd-sz" x="1" y="1.5" width="10" height="3"/></svg>Eq. 4 with a whole column of zeros</span>
        <span class="pd-key"><svg viewBox="-6 -6 12 12" aria-hidden="true"><path class="pd-dia" d="M0 -4.6 L4.6 0 L0 4.6 L-4.6 0 Z"/></svg>the report’s draw, ticket ${R42}</span>
        <span class="pd-key"><svg viewBox="0 0 12 12" aria-hidden="true"><line class="pd-med" x1="6" y1="0" x2="6" y2="12"/></svg>median of the ${N}</span>
        <span class="pd-key"><svg viewBox="0 0 12 6" aria-hidden="true"><rect class="pd-sa" x="1" y="1" width="10" height="4"/></svg>the ticket in hand</span>
      </p>
    </div>
    <svg class="pd-svg" tabindex="0" role="group" aria-label="The board. Each draw is a stub at its best-kernel test accuracy for each data set, Eq. 3 above each line and Eq. 4 below. Tap a stub to take that ticket; with the keyboard, the left and right arrows step through the tickets and the up and down arrows through the rows."></svg>
    <p class="pd-rank" aria-live="polite"></p>
  </section>
  <section class="pd-tally" aria-label="The pulls so far and all ${N} draws">
    <div class="pd-tally-head"><h4 class="pd-h pd-tally-title"></h4><button class="pd-clear" type="button">Clear the pulls</button></div>
    <div class="pd-vrow" data-row="all"><span class="pd-vname">All ${N} draws</span><span class="pd-vbar"></span></div>
    <div class="pd-vrow" data-row="mine"><span class="pd-vname">These pulls</span><span class="pd-vbar"></span></div>
    <p class="pd-vkey">
      <span class="pd-key"><span class="pd-sw pd-sw-h"></span>the report’s headline: Eq. 4 ahead on three or more data sets</span>
      <span class="pd-key"><span class="pd-sw pd-sw-o"></span>the opposite: Eq. 3 ahead on three or more</span>
      <span class="pd-key"><span class="pd-sw pd-sw-n"></span>neither</span>
      <span class="pd-key"><span class="pd-sw pd-sw-t"></span>the headline’s share in all ${N}</span>
    </p>
    <p class="pd-tally-say"></p>
    <div class="pd-iris">
      <span class="pd-vname">Iris, Eq. 4: a whole column of zeros</span>
      <span class="pd-itrack" aria-hidden="true"><span class="pd-ifill"></span><span class="pd-itick pd-ifile"></span><span class="pd-itick pd-itheory"></span></span>
      <span class="pd-ival"></span>
      <span class="pd-inote">solid tick: ${IRIS.any} of the ${N} committed draws · dashed tick: ${esc(data.paper.theory_zero_iris)}% in theory, as the paper prints</span>
    </div>
    <p class="pd-iris-say"></p>
  </section>
  <div class="pd-notes">
    <p>Each ticket is one of the study’s ${N} committed draws of the February pipeline (study/replicate.py, seeds 0 to ${N - 1}): the same scaling, one split of the data (random_state = ${data.split_seed}), default SVMs, and <i>k</i> = ${data.k}, so only the two matrices change from ticket to ticket. The best kernel is the highest of the four test accuracies (linear, polynomial of degree 2 and 3, and radial basis function, or RBF), chosen on the test set as the report did; small type under a value names the kernels that reach it.</p>
    <p>Ticket ${R42} is the report’s draw: it is the only one whose ${4 * 2 * DS.length} accuracies equal the February run’s committed results. Stubs at the same accuracy stack in ticket order; nothing is jittered. Eq. 3’s entries are never zero. The file records how many rows and columns of each Eq. 4 matrix were zero, not which ones, so the share of draws with both petal rows zero (${esc(data.paper.theory_both_iris)}% in theory) can’t be read from it.</p>
    <p class="pd-seed"></p>
    <p class="pd-checked"></p>
  </div>
  <div class="pd-fly" aria-hidden="true" hidden><span></span></div>
</div>`;

  const $ = (s) => root.querySelector(s);
  const pd = $(".pd"), drumEl = $(".pd-drum"), cv = $(".pd-canvas"), ctx = cv.getContext("2d");
  const ticketEl = $(".pd-ticket"), bodyEl = $(".pd-body"), stampEl = $(".pd-stamp"), noEl = $(".pd-no");
  const svg = $(".pd-svg"), flyEl = $(".pd-fly");

  // ---------------------------------------------------------------- state
  const st = {
    seed: REPORT, mine: [], touched: false, visible: false, started: false, busy: null,
    focusRow: DS.findIndex((d) => d.id === "seeds") * 2 + 1, lastAuto: 0,
  };

  // ---------------------------------------------------------------- colors, read from the page at draw time
  let C = {};
  function readColors() {
    const cs = getComputedStyle(fig);
    const v = (n, d) => cs.getPropertyValue(n).trim() || d;
    C = { ink: v("--ink", "#111"), ink2: v("--ink-2", "#555"), rule: v("--rule", "rgba(0,0,0,.15)"), rule2: v("--rule-2", "rgba(0,0,0,.4)"),
      accent: v("--accent-ink", "#12756a"), paper: v("--paper", "#fff"), paper2: v("--paper-2", "#eee"), type: v("--type", "monospace") };
  }
  readColors();

  // ---------------------------------------------------------------- the drum: 100 tickets in a turning hexagonal cage
  // Positions are in units of the cage's corner radius; the cage turns at drum.omega.
  const D = { theta: 0.3, omega: 0, target: 0, R: 100, cx: 0, cy: 0, parts: [], held: null, still: 0 };
  // a ticket's half width and half height, its radius against other tickets, and its margin from the walls
  const TW = 0.08, TH = 0.045, RC = 0.056, WM = 0.084, APO = Math.cos(Math.PI / 6), G = 9.5;
  {
    const r0 = new M.Rng(4242);
    for (const t of tickets) {
      const a = r0.next() * 2 * Math.PI, rad = 0.72 * Math.sqrt(r0.next());
      const x = rad * Math.cos(a), y = rad * Math.sin(a) * 0.6 + 0.25;
      D.parts.push({ seed: t.seed, x, y, px: x, py: y, a: r0.next() * Math.PI, va: 0 });
    }
    for (let i = 0; i < 520; i++) step(1 / 180);
  }
  function step(h) {
    const P = D.parts;
    D.theta += D.omega * h;
    for (const p of P) {
      const vx = (p.x - p.px) * 0.996, vy = (p.y - p.py) * 0.996;
      p.px = p.x; p.py = p.y;
      p.x += vx; p.y += vy + G * h * h;
      p.a += p.va * h; p.va *= 0.97;
    }
    const d0 = 2 * RC, d02 = d0 * d0;
    for (let it = 0; it < 2; it++) {
      for (let i = 0; i < P.length; i++) {
        const a = P[i];
        for (let j = i + 1; j < P.length; j++) {
          const b = P[j];
          const dx = b.x - a.x, dy = b.y - a.y, q = dx * dx + dy * dy;
          if (q >= d02 || q === 0) continue;
          const d = Math.sqrt(q), push = ((d0 - d) / d) * 0.45;
          a.x -= dx * push; a.y -= dy * push; b.x += dx * push; b.y += dy * push;
        }
      }
    }
    for (const p of P) {
      for (let k = 0; k < 6; k++) {
        const ang = D.theta + Math.PI / 6 + (k * Math.PI) / 3, nx = Math.cos(ang), ny = Math.sin(ang);
        const over = p.x * nx + p.y * ny - (APO - WM);
        if (over <= 0) continue;
        p.x -= over * nx; p.y -= over * ny;
        const vx = p.x - p.px, vy = p.y - p.py, tx = -ny, ty = nx;
        let vn = vx * nx + vy * ny, vt = vx * tx + vy * ty;
        const wt = (-D.omega * p.y * tx + D.omega * p.x * ty) * h;
        if (vn > 0) vn = -vn * 0.15;
        const dv = (wt - vt) * 0.32;
        vt += dv;
        p.px = p.x - (vn * nx + vt * tx); p.py = p.y - (vn * ny + vt * ty);
        p.va += (dv / h) * 7;
      }
    }
  }
  function energy() {
    let e = 0;
    for (const p of D.parts) e = Math.max(e, Math.abs(p.x - p.px) + Math.abs(p.y - p.py));
    return e;
  }
  function drumGeometry() {
    const w = drumEl.clientWidth, h = cv.clientHeight;
    const R = Math.max(40, Math.min(w * 0.4, h * 0.39));
    return { w, h, R, cx: w / 2 - R * 0.05, cy: R * 1.08 + 4 };
  }
  let lastSize = "";
  function drawDrum() {
    const { w, h, R, cx, cy } = drumGeometry();
    if (!w || !h) return;
    Object.assign(D, { R, cx, cy });
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const size = `${w}x${h}x${dpr}`;
    if (size !== lastSize) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); lastSize = size; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const ox = R * 0.1, oy = -R * 0.075; // the back face, in parallel projection
    const corner = (k, dx = 0, dy = 0) => {
      const a = D.theta + (k * Math.PI) / 3;
      return [cx + dx + R * Math.cos(a), cy + dy + R * Math.sin(a)];
    };
    // the stand
    const base = cy + R * 1.17;
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(cx - R * 0.95, base); ctx.lineTo(cx + R * 0.95 + ox, base);
    ctx.moveTo(cx, cy); ctx.lineTo(cx - R * 0.62, base); ctx.moveTo(cx, cy); ctx.lineTo(cx + R * 0.62, base);
    ctx.moveTo(cx + ox, cy + oy); ctx.lineTo(cx + ox + R * 0.62, base);
    ctx.stroke();
    // the back face and the rods
    ctx.strokeStyle = C.rule2; ctx.lineWidth = 1;
    ctx.beginPath();
    for (let k = 0; k <= 6; k++) { const [x, y] = corner(k, ox, oy); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
    for (let k = 0; k < 6; k++) { const [x, y] = corner(k), [x2, y2] = corner(k, ox, oy); ctx.moveTo(x, y); ctx.lineTo(x2, y2); }
    ctx.stroke();
    // the tickets
    const tw = TW * R, th = TH * R, big = tw >= 13;
    ctx.font = `${Math.max(6, th * 0.86).toFixed(1)}px ${C.type}`;
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (const p of D.parts) {
      ctx.save();
      ctx.translate(cx + p.x * R, cy + p.y * R);
      ctx.rotate(p.a);
      ctx.fillStyle = C.paper; ctx.strokeStyle = C.ink2; ctx.lineWidth = 0.8;
      ctx.beginPath(); ctx.rect(-tw, -th, 2 * tw, 2 * th); ctx.fill(); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(-tw * 0.45, -th); ctx.lineTo(-tw * 0.45, th); ctx.strokeStyle = C.rule2; ctx.stroke();
      if (big) { ctx.fillStyle = C.ink2; ctx.fillText(M.ticketNo(p.seed), tw * 0.25, 0.4); }
      ctx.restore();
    }
    // the front face, the axle, and the crank
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.8;
    ctx.beginPath();
    for (let k = 0; k <= 6; k++) { const [x, y] = corner(k); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
    ctx.stroke();
    ctx.lineWidth = 0.8; ctx.strokeStyle = C.ink2;
    ctx.beginPath();
    for (let k = 0; k <= 6; k++) {
      const a = D.theta + (k * Math.PI) / 3, x = cx + R * 0.93 * Math.cos(a), y = cy + R * 0.93 * Math.sin(a);
      k ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();
    const ca = D.theta * 1 + 0.9, kx = cx + R * 0.42 * Math.cos(ca), ky = cy + R * 0.42 * Math.sin(ca);
    ctx.strokeStyle = C.ink; ctx.lineWidth = 2.2;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(kx, ky); ctx.stroke();
    ctx.fillStyle = C.ink;
    ctx.beginPath(); ctx.arc(cx, cy, Math.max(3, R * 0.05), 0, 2 * Math.PI); ctx.fill();
    ctx.beginPath(); ctx.arc(kx, ky, Math.max(3.4, R * 0.055), 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = C.paper; ctx.beginPath(); ctx.arc(cx, cy, Math.max(1.2, R * 0.018), 0, 2 * Math.PI); ctx.fill();
  }

  // ---------------------------------------------------------------- the ticket in hand
  function fillTicket(seed) {
    const t = bySeed.get(seed);
    noEl.textContent = M.ticketNo(seed);
    ticketEl.classList.toggle("is-report", seed === REPORT);
    $(".pd-thead").textContent = M.sayTicket(seed, REPORT);
    for (const d of DS) {
      const row = root.querySelector(`tr[data-ds="${d.id}"]`), cell = t.ds[d.id];
      for (const eq of M.EQS) {
        const td = row.querySelector(`td[data-eq="${eq}"]`), c = cell[eq];
        td.innerHTML = `<span class="pd-v">${M.pct(c.best, d.n_test)}</span><small>${esc(M.sayKernels(c.kernels, shortOf))}</small>`;
        td.classList.toggle("is-lead", cell.lead === eq);
        td.classList.toggle("is-zero", eq === "eq4" && c.zc > 0);
      }
      const df = row.querySelector(".pd-diff");
      df.innerHTML = `<span class="pd-v">${M.pts(cell.diff, d.n_test)}</span><small>${cell.lead === "tie" ? "tie" : `${M.EQ_LABEL[cell.lead]} ahead`}</small>`;
      df.dataset.lead = cell.lead;
    }
    const sp = M.stampOf(t);
    stampEl.querySelector("b").textContent = sp.title;
    stampEl.querySelector("span").textContent = sp.sub;
    stampEl.dataset.v = t.verdict;
    $(".pd-leads").textContent = M.sayLeads(t, DS);
    $(".pd-verdict").textContent = M.sayVerdict(t, REPORT);
    $(".pd-zeros").textContent = M.sayZeros(t, DS);
    $(".pd-unit").textContent = DS.some((d) => t.ds[d.id].eq4.zc > 0)
      ? "test accuracy, %, best of four kernels · underlined: a whole column of zeros" : "test accuracy, %, best of four kernels";
  }
  function showTicket(seed, how) {
    st.seed = seed;
    pd.classList.remove("is-drawing");
    noEl.style.visibility = "";
    fillTicket(seed);
    renderSelection(how !== "still" && !reduced.matches);
    renderRank();
    if (how === "still" || reduced.matches) return;
    bodyEl.getAnimations().forEach((a) => a.cancel());
    stampEl.getAnimations().forEach((a) => a.cancel());
    bodyEl.animate([{ transform: "perspective(700px) rotateX(-84deg)", opacity: 0.35 }, { transform: "perspective(700px) rotateX(8deg)", opacity: 1, offset: 0.72 },
      { transform: "perspective(700px) rotateX(0deg)", opacity: 1 }], { duration: T.unfold, easing: "cubic-bezier(.2,.7,.3,1)" });
    root.querySelectorAll(".pd-table tbody tr").forEach((tr, i) => tr.animate([{ opacity: 0 }, { opacity: 1 }],
      { duration: 180, delay: T.unfold * 0.45 + i * 55, fill: "backwards" }));
    stampEl.animate([{ transform: "rotate(-7deg) scale(1.7)", opacity: 0 }, { transform: "rotate(-5deg) scale(0.96)", opacity: 1, offset: 0.7 },
      { transform: "rotate(-5deg) scale(1)", opacity: 1 }], { duration: T.stamp, delay: T.unfold + 60, easing: "cubic-bezier(.3,.6,.4,1)", fill: "backwards" });
  }

  // ---------------------------------------------------------------- drawing from the drum
  function finishPending() { if (st.busy) st.busy.finish(); }
  function returnHeld() {
    if (D.held == null) return;
    const s = D.held;
    D.held = null;
    if (D.parts.some((p) => p.seed === s)) return;
    const x = (dropRng.next() - 0.5) * 0.3, y = -0.55;
    D.parts.push({ seed: s, x, y, px: x, py: y + 0.01, a: dropRng.next() * Math.PI, va: 0 });
  }
  function draw(times, auto = false) {
    finishPending();
    if (!auto) touch();
    const seeds = Array.from({ length: times }, () => M.pull(auto ? ownRng : rng, tickets));
    const last = seeds[seeds.length - 1];
    const commit = () => { st.mine = M.countPulls(st.mine, seeds, !auto); renderTally(); };
    if (reduced.matches || !st.visible) {
      returnHeld();
      commit();
      showTicket(last, "still");
      drawDrum();
      return;
    }
    // fold the old ticket away, spin, pull, fly, and unfold
    returnHeld();
    pd.classList.add("is-drawing");
    noEl.style.visibility = "hidden";
    bodyEl.animate([{ transform: "perspective(700px) rotateX(0deg)", opacity: 1 }, { transform: "perspective(700px) rotateX(-84deg)", opacity: 0.3 }],
      { duration: T.fold, easing: "ease-in", fill: "forwards" });
    D.target = SPIN;
    wake();
    const timers = [];
    let fly = null, done = false;
    const land = () => {
      if (done) return;
      done = true;
      timers.forEach(clearTimeout);
      if (fly) fly.cancel();
      flyEl.hidden = true;
      bodyEl.getAnimations().forEach((a) => a.cancel());
      D.target = 0;
      // the pulled ticket stays out of the drum until the next pull
      const i = D.parts.findIndex((p) => p.seed === last);
      if (i >= 0) D.parts.splice(i, 1);
      D.held = last;
      commit();
      showTicket(last, "animate");
      st.busy = null;
      st.lastAuto = performance.now();
      wake();
    };
    timers.push(setTimeout(() => {
      D.target = 0;
      const i = D.parts.findIndex((p) => p.seed === last);
      const p = i >= 0 ? D.parts[i] : { x: 0, y: 0.4, a: 0 };
      if (i >= 0) D.parts.splice(i, 1);
      D.held = last;
      fly = flyTicket(p, last, land);
    }, T.up + T.spin));
    st.busy = { finish: land };
  }
  function flyTicket(p, seed, onEnd) {
    const host = pd.getBoundingClientRect(), c = cv.getBoundingClientRect(), stub = noEl.getBoundingClientRect();
    const fw = 58, fh = 32;
    flyEl.querySelector("span").textContent = M.ticketNo(seed);
    flyEl.hidden = false;
    const sx = c.left - host.left + D.cx + p.x * D.R, sy = c.top - host.top + D.cy + p.y * D.R;
    const rx = c.left - host.left + D.cx + p.x * D.R * 0.3, ry = c.top - host.top + D.cy - D.R * 1.18;
    const ex = stub.left - host.left + stub.width / 2, ey = stub.top - host.top + stub.height / 2;
    const s0 = (2 * TW * D.R) / fw, s2 = clamp(stub.width / fw, 0.8, 1.5);
    const at = (x, y, r, s) => `translate(${(x - fw / 2).toFixed(1)}px, ${(y - fh / 2).toFixed(1)}px) rotate(${r.toFixed(3)}rad) scale(${s.toFixed(3)})`;
    const tot = T.rise + T.fly;
    const anim = flyEl.animate([
      { transform: at(sx, sy, p.a, s0), opacity: 1 },
      { transform: at(rx, ry, -0.25, 1.25), opacity: 1, offset: T.rise / tot, easing: "cubic-bezier(.3,.1,.3,1)" },
      { transform: at(ex, ey, 0, s2), opacity: 1 },
    ], { duration: tot, easing: "cubic-bezier(.4,0,.2,1)", fill: "forwards" });
    anim.onfinish = onEnd;
    return anim;
  }
  function take(seed) {
    // a ticket chosen by number (the report's, or a stub on the board): read, not counted as a pull
    finishPending();
    touch();
    returnHeld();
    showTicket(seed, "animate");
    wake();
  }

  // ---------------------------------------------------------------- the board
  let B = null; // layout: rows, stubs, and the x scale
  const [AX0, AX1] = M.axisRange(tickets, DS);
  const STACK = Object.fromEntries(DS.map((d) => [d.id, Object.fromEntries(M.EQS.map((eq) => [eq, M.stacks(tickets, d.id, eq)]))]));
  function layoutBoard() {
    const W = Math.max(280, Math.round(svg.parentElement.clientWidth));
    const small = W < 560;
    const gutter = small ? 36 : 46, mr = 10, pitch = small ? 2.6 : 3, gap = small ? 0.7 : 0.85;
    const X = (v) => gutter + ((v - AX0) / (AX1 - AX0)) * (W - gutter - mr);
    const pxPerPt = (W - gutter - mr) / (AX1 - AX0);
    let y = 20;
    const rows = [], stubs = [];
    for (const d of DS) {
      const s3 = STACK[d.id].eq3, s4 = STACK[d.id].eq4;
      const labelY = y + 12;
      const up = s3.tallest * pitch, down = s4.tallest * pitch;
      const rule = labelY + 8 + up;
      const sw = clamp((100 / d.n_test) * pxPerPt * 0.78, 1.6, small ? 6 : 7);
      rows.push({ d, labelY, rule, up, down, sw });
      for (const eq of M.EQS) {
        const S = STACK[d.id][eq];
        for (const t of tickets) {
          const c = t.ds[d.id][eq];
          const lv = S.level.get(t.seed);
          const x = X((100 * c.best) / d.n_test);
          const yTop = eq === "eq3" ? rule - (lv + 1) * pitch + gap / 2 : rule + lv * pitch + gap / 2;
          stubs.push({ seed: t.seed, ds: d.id, eq, x, y: yTop, w: sw, h: pitch - gap, zero: eq === "eq4" && c.zc > 0 });
        }
      }
      y = rule + down + 16;
    }
    const axisY = y + 2;
    const H = axisY + 22;
    B = { W, H, X, rows, stubs, axisY, small, pitch, gutter, mr };
  }
  function renderBoard() {
    layoutBoard();
    const { W, H, X, rows, stubs, axisY, small, gutter } = B;
    let s = "";
    // grid and axis
    s += `<g class="pd-grid">`;
    for (let v = Math.ceil(AX0 / 10) * 10; v <= AX1; v += 10) s += `<line x1="${X(v)}" y1="15" x2="${X(v)}" y2="${axisY}"/>`;
    s += `</g><g class="pd-axis">`;
    for (let v = Math.ceil(AX0 / 10) * 10; v <= AX1; v += 10) s += `<text x="${X(v)}" y="${axisY + 14}">${v}${v === AX1 ? "%" : ""}</text><text x="${X(v)}" y="10">${v}${v === AX1 ? "%" : ""}</text>`;
    s += `<line class="pd-axisline" x1="${gutter}" y1="${axisY}" x2="${W - B.mr}" y2="${axisY}"/></g>`;
    // rows
    for (const r of rows) {
      const { d } = r;
      s += `<text class="pd-rowname" x="${small ? 2 : gutter}" y="${r.labelY}">${esc(d.label)}<tspan class="pd-rownote" dx="7">${d.n_test} test rows, each ${M.pct(1, d.n_test)} points</tspan></text>`;
      s += `<text class="pd-side" x="${gutter - 6}" y="${r.rule - 3}">Eq. 3</text><text class="pd-side" x="${gutter - 6}" y="${r.rule + 9}">Eq. 4</text>`;
      for (const eq of M.EQS) {
        const m = X(100 * d.median[eq]), y0 = eq === "eq3" ? r.rule - r.up - 3 : r.rule, y1 = eq === "eq3" ? r.rule : r.rule + r.down + 3;
        s += `<line class="pd-med" x1="${m}" y1="${y0}" x2="${m}" y2="${y1}"/>`;
      }
    }
    s += `<g class="pd-stubs">`;
    for (const b of stubs) s += `<rect class="${b.eq === "eq3" ? "pd-s3" : b.zero ? "pd-sz" : "pd-s4"}" x="${(b.x - b.w / 2).toFixed(2)}" y="${b.y.toFixed(2)}" width="${b.w.toFixed(2)}" height="${b.h.toFixed(2)}"/>`;
    s += `</g>`;
    for (const r of rows) s += `<line class="pd-rule" x1="${gutter}" y1="${r.rule}" x2="${W - B.mr}" y2="${r.rule}"/>`;
    s += `<g class="pd-sel"></g><g class="pd-dias">`;
    for (const b of stubs) if (b.seed === REPORT) s += `<path class="pd-dia" data-eq="${b.eq}" d="${dia(b.x, b.y + b.h / 2, small ? 4.4 : 5)}"/>`;
    s += `</g><g class="pd-focus"></g>`;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("width", W); svg.setAttribute("height", H);
    svg.innerHTML = s;
    renderSelection();
  }
  const dia = (x, y, r) => `M${x.toFixed(2)} ${(y - r).toFixed(2)} L${(x + r).toFixed(2)} ${y.toFixed(2)} L${x.toFixed(2)} ${(y + r).toFixed(2)} L${(x - r).toFixed(2)} ${y.toFixed(2)} Z`;
  function renderSelection(pulse = false) {
    if (!B) return;
    const g = svg.querySelector(".pd-sel");
    if (!g) return;
    const t = bySeed.get(st.seed);
    let s = "";
    for (const r of B.rows) {
      const mine = B.stubs.filter((b) => b.seed === st.seed && b.ds === r.d.id);
      const b3 = mine.find((b) => b.eq === "eq3"), b4 = mine.find((b) => b.eq === "eq4");
      for (const b of mine) {
        s += `<rect class="pd-sa" x="${(b.x - b.w / 2 - 1.2).toFixed(2)}" y="${(b.y - 0.6).toFixed(2)}" width="${(b.w + 2.4).toFixed(2)}" height="${(b.h + 1.2).toFixed(2)}"/>`;
        if (pulse) s += `<circle class="pd-pulse" cx="${b.x.toFixed(2)}" cy="${(b.y + b.h / 2).toFixed(2)}" r="9"/>`;
      }
      const lead = t.ds[r.d.id].lead;
      if (lead === "tie") s += `<circle class="pd-arrow" cx="${b3.x.toFixed(2)}" cy="${r.rule}" r="2.6"/>`;
      else {
        const dir = Math.sign(b4.x - b3.x), hx = b4.x - dir * 1.5;
        s += `<line class="pd-arrow" x1="${b3.x.toFixed(2)}" y1="${r.rule}" x2="${(hx - dir * 4).toFixed(2)}" y2="${r.rule}"/>`;
        s += `<path class="pd-head" d="M${hx.toFixed(2)} ${r.rule} l${(-dir * 6).toFixed(2)} -3.6 v7.2 Z"/>`;
      }
    }
    g.innerHTML = s;
    svg.querySelectorAll(".pd-dia").forEach((el) => el.classList.toggle("is-on", st.seed === REPORT));
    renderFocus();
  }
  function renderFocus() {
    const g = svg.querySelector(".pd-focus");
    if (!g || !B) return;
    const r = B.rows[Math.floor(st.focusRow / 2)], eq = st.focusRow % 2 ? "eq4" : "eq3";
    const b = B.stubs.find((x) => x.seed === st.seed && x.ds === r.d.id && x.eq === eq);
    g.innerHTML = st.showFocus && b ? `<circle class="pd-ring" cx="${b.x.toFixed(2)}" cy="${(b.y + b.h / 2).toFixed(2)}" r="7"/>` : "";
  }
  function renderRank() {
    const el = $(".pd-rank");
    if (st.rankRow == null) { el.textContent = "Tap a stub to take that ticket and see where it stands among the other 99."; el.classList.remove("is-on"); return; }
    const d = DS[Math.floor(st.rankRow / 2)], eq = st.rankRow % 2 ? "eq4" : "eq3";
    const t = bySeed.get(st.seed);
    el.textContent = M.sayRank(d.label, eq, st.seed, M.pct(t.ds[d.id][eq].best, d.n_test), M.rankOf(tickets, d.id, eq, st.seed));
    el.classList.add("is-on");
  }
  function nearestStub(e) {
    const r = svg.getBoundingClientRect(), k = B.W / r.width;
    const x = (e.clientX - r.left) * k, y = (e.clientY - r.top) * k;
    const lim = (e.pointerType === "mouse" ? 9 : 16) ** 2;
    let best = null, bd = lim, marked = null, md = (B.small ? 8 : 6.5) ** 2;
    for (const b of B.stubs) {
      const dx = x - b.x, dy = y - (b.y + b.h / 2), q = dx * dx + dy * dy * 1.6;
      if (q < bd) { bd = q; best = b; }
      // the report's diamond is a bigger target than one stub in its stack
      if (b.seed === REPORT && dx * dx + dy * dy < md) { md = dx * dx + dy * dy; marked = b; }
    }
    return marked || best;
  }
  svg.addEventListener("pointerdown", (e) => {
    touch();
    const b = nearestStub(e);
    if (!b) return;
    st.rankRow = DS.findIndex((d) => d.id === b.ds) * 2 + (b.eq === "eq4" ? 1 : 0);
    st.focusRow = st.rankRow;
    if (b.seed !== st.seed) take(b.seed); else { renderRank(); renderFocus(); }
  });
  svg.addEventListener("pointermove", (e) => { if (e.pointerType === "mouse") svg.style.cursor = nearestStub(e) ? "pointer" : "default"; });
  svg.addEventListener("keydown", (e) => {
    const k = e.key;
    if (k === "ArrowLeft" || k === "ArrowRight") { st.rankRow = st.focusRow; st.showFocus = true; take((st.seed + (k === "ArrowLeft" ? N - 1 : 1)) % N); }
    else if (k === "ArrowUp" || k === "ArrowDown") { st.focusRow = (st.focusRow + (k === "ArrowUp" ? 2 * DS.length - 1 : 1)) % (2 * DS.length); st.rankRow = st.focusRow; st.showFocus = true; renderRank(); renderFocus(); }
    else if (k === "Home" || k === "End") { st.rankRow = st.focusRow; st.showFocus = true; take(k === "Home" ? 0 : N - 1); }
    else return;
    e.preventDefault();
  });
  svg.addEventListener("blur", () => { st.showFocus = false; renderFocus(); });

  // ---------------------------------------------------------------- the tally
  function bar(el, c, n, tick) {
    const seg = (cls, v, label) => {
      const w = n ? (100 * v) / n : 0;
      return `<span class="pd-seg ${cls}" style="width:${w}%">${w >= 9 ? `<b>${label}</b>` : ""}</span>`;
    };
    el.innerHTML = n
      ? seg("pd-sw-h", c.headline, c.headline) + seg("pd-sw-o", c.opposite, c.opposite) + seg("pd-sw-n", c.neither, c.neither) + (tick != null ? `<span class="pd-truetick" style="left:${tick}%"></span>` : "")
      : `<span class="pd-empty">${M.NO_PULLS}</span>${tick != null ? `<span class="pd-truetick" style="left:${tick}%"></span>` : ""}`;
  }
  function renderTally() {
    const mine = M.tally(st.mine.map((s) => bySeed.get(s)));
    const n = mine.n;
    $(".pd-tally-title").textContent = `Your pulls so far · ${n}`;
    bar(root.querySelector('[data-row="all"] .pd-vbar'), ALL, ALL.n, (100 * ALL.headline) / ALL.n);
    bar(root.querySelector('[data-row="mine"] .pd-vbar'), mine, n, (100 * ALL.headline) / ALL.n);
    root.querySelector('[data-row="mine"] .pd-vname').textContent = n === 1 ? "This 1 pull" : `These ${n} pulls`;
    $(".pd-tally-say").textContent = M.sayTally(n, mine, ALL);
    $(".pd-mini").textContent = M.sayMini(n, mine, ALL);
    const zi = M.zeroColumnCount(st.mine.map((s) => bySeed.get(s)), "iris", "eq4");
    $(".pd-ifill").style.width = `${n ? (100 * zi.any) / n : 0}%`;
    $(".pd-ifile").style.left = `${(100 * IRIS.any) / IRIS.n}%`;
    $(".pd-itheory").style.left = `${Number(data.paper.theory_zero_iris)}%`;
    $(".pd-ival").textContent = M.sayCount(zi.any, n);
    $(".pd-iris-say").textContent = M.sayIris(n, zi.any, IRIS, data.paper.theory_zero_iris);
  }

  // ---------------------------------------------------------------- controls
  function touch() {
    // the drum stops pulling on its own, and from now on each new ticket is announced
    if (!st.touched) $(".pd-say").setAttribute("aria-live", "polite");
    st.touched = true;
  }
  $(".pd-draw").addEventListener("click", () => draw(1));
  $(".pd-more").addEventListener("click", () => draw(20));
  $(".pd-report").addEventListener("click", () => take(REPORT));
  // a pull still in the air lands first, so the clear holds
  $(".pd-clear").addEventListener("click", () => { finishPending(); touch(); st.mine = []; renderTally(); });
  root.addEventListener("pointerdown", () => touch());
  root.addEventListener("keydown", () => touch());

  // ---------------------------------------------------------------- alive: the drum turns, and pulls a ticket now and then, until touched
  let loop = 0, last = 0;
  const auto = () => !st.touched && !reduced.matches && !fixedSeed;
  function frame(now) {
    loop = 0;
    const dt = last ? Math.min(0.05, (now - last) / 1000) : 1 / 60;
    last = now;
    const idle = auto() || (!st.touched && !reduced.matches) ? IDLE : 0;
    const target = Math.max(D.target, idle);
    const rate = D.target > D.omega ? SPIN / (T.up / 1000) : SPIN / (T.down / 1000);
    D.omega = D.omega < target ? Math.min(target, D.omega + rate * dt) : Math.max(target, D.omega - rate * dt);
    const n = Math.max(1, Math.round(dt * 180));
    for (let i = 0; i < n; i++) step(dt / n);
    drawDrum();
    if (auto() && !st.busy && st.visible && now - st.lastAuto > (st.started === "auto" ? T.rest : T.first)) { st.started = "auto"; draw(1, true); }
    // the heap settles: after a second and a half of near stillness, the loop sleeps until the next pull
    const calm = D.omega < 0.001 && D.target === 0 && energy() < 1e-3;
    st.calm = calm ? (st.calm || 0) + 1 : 0;
    const moving = !calm || st.calm < 90;
    if (st.visible && (moving || auto() || st.busy)) loop = requestAnimationFrame(frame);
    else last = 0;
  }
  function wake() {
    if (reduced.matches) { drawDrum(); return; }
    if (!loop && st.visible) loop = requestAnimationFrame(frame);
    else if (!st.visible) drawDrum();
  }
  new IntersectionObserver((es) => {
    st.visible = es[es.length - 1].isIntersecting;
    if (st.visible && !st.started) { st.started = true; st.lastAuto = performance.now(); }
    if (st.visible) wake();
  }, { threshold: 0.2 }).observe(drumEl);
  document.addEventListener("visibilitychange", () => { last = 0; if (document.hidden) finishPending(); else wake(); });
  let lastW = 0;
  new ResizeObserver(() => {
    const w = svg.parentElement.clientWidth;
    if (w !== lastW) { lastW = w; renderBoard(); }
    drawDrum();
  }).observe(root);
  const repaint = () => { readColors(); drawDrum(); };
  new MutationObserver(repaint).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "class", "style"] });
  darkMq.addEventListener("change", repaint);
  reduced.addEventListener("change", () => { if (reduced.matches) { finishPending(); D.omega = 0; D.target = 0; } wake(); });
  if (document.fonts) document.fonts.load(`10px ${C.type}`).then(() => drawDrum(), () => {});

  // first paint: the report's own ticket, then the drum's pulls
  showTicket(REPORT, "still");
  renderBoard();
  renderTally();
  drawDrum();
  const ownPulls = fixedSeed ? "With a seed in the address, the drum pulls only when you do."
    : "Until you touch the plate, the drum pulls a ticket now and then on its own (not when motion is reduced), and those pulls are not counted.";
  $(".pd-seed").innerHTML = `Your pulls use seed ${pullSeed}. <a href="${M.seedQuery(pullSeed)}#${esc(fig.id)}">Open the page with ${M.seedQuery(pullSeed)}</a> to pull the same tickets in the same order. Each pull takes one of the ${N} tickets at random and puts it back. ${ownPulls} Nothing is trained or estimated in the page: it reads the committed file and counts.`;

  // the latest check run, quietly, when it exists
  fetch(src.replace(/\.json$/, ".checks.json")).then((r) => (r.ok ? r.json() : null)).then((c) => {
    if (c && c.line) $(".pd-checked").textContent = c.line;
  }).catch(() => {});
}
