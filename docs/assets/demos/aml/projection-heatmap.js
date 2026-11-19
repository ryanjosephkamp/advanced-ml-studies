// Chapter III's live plate "Every cell the study measured": a table set in type. Each measured cell
// of the study is a type block, inked to a density that shows its value, with the value set in
// figures on it; PCA and all features sit below a rule as the table's reference rows. Switching the
// data set resets the forme with a short slide; the ink can show the score, or each row's paired
// difference from the Gaussian matrix in two inks. Tapping a block reads it on the slip below, with
// its interval and its comparisons. Until the reader touches it, the plate reads down k = 2 on its
// own, one block at a time, and moves on to the next data set. Every number and sentence comes from
// projection-heatmap-model.js, which tools/aml/projection-heatmap.check.mjs holds to the study.
const KIND = "projection-heatmap";
const VER = new URL(import.meta.url).search;
const M = await import(`./${KIND}-model.js${VER}`);
const reduced = matchMedia("(prefers-reduced-motion: reduce)");

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
    if (root) root.innerHTML = `<p class="ph-error">The live plate could not load (${String(err.message || err)}).</p>`;
  });
}

// timings, in milliseconds
const T = { out: 150, in: 320, stagger: 26, roll: 640, read: 2600, firstRead: 1100 };
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
// a block's tint in each printing, and which printings reverse its figures; the stylesheet picks one
const tints = (p) => `--td:${p.day.tint};--tn:${p.night.tint}`;
const printed = (p) => (p.day.reversed ? " is-rev-d" : "") + (p.night.reversed ? " is-rev-n" : "");
// an animation's end, or its planned end by the clock, whichever comes first (a page that is not
// being drawn, such as a hidden tab, may never play it)
const settle = (anims, ms) => Promise.race([Promise.all(anims.map((a) => a.finished.catch(() => {}))), new Promise((r) => setTimeout(r, ms))]);

async function mount(fig) {
  const id = ++uid;
  const root = fig.querySelector(".demo-root");
  const src = fig.dataset.src;
  const data = await (await fetch(src)).json();
  const sets = data.datasets;

  root.innerHTML = `
<div class="ph">
  <div class="ph-controls">
    <fieldset class="ph-sets"><legend>Data set</legend>${sets.map((d) => `
      <label class="ph-sort"><input type="radio" name="ph-set-${id}" value="${d.id}"><span><b>${esc(d.label)}</b><i>d = ${d.d}</i></span></label>`).join("")}
    </fieldset>
    <fieldset class="ph-modes"><legend>Ink shows</legend>
      <label class="ph-sort"><input type="radio" name="ph-mode-${id}" value="score"><span><b>The score</b></span></label>
      <label class="ph-sort"><input type="radio" name="ph-mode-${id}" value="diff"><span><b>Minus the Gaussian</b></span></label>
    </fieldset>
  </div>
  <p class="ph-what"></p>
  <div class="ph-chase">
    <div class="ph-forme" role="group"></div>
    <div class="ph-roller" aria-hidden="true" hidden></div>
  </div>
  <div class="ph-foot">
    <div class="ph-key" aria-hidden="true"></div>
    <p class="ph-keysay"></p>
    <p class="ph-marks"><span><b>*</b> differs from the Gaussian matrix after Holm's correction (on a line of the slip, that line's comparison does)</span><span><b>†</b> its interval against the Gaussian lies inside ±2 points (on a line of the slip, that line's interval does)</span></p>
  </div>
  <p class="ph-status" aria-hidden="true"></p>
  <section class="ph-slip" aria-live="off" aria-label="The block being read"></section>
  <div class="ph-notes">
    <p>When the ink shows the score, each block is one row of the study's summary: the held-out score of the support vector machine whose kernel and settings were chosen by cross-validation on the training rows, averaged over ${data.n_draws} draws, each with its own 70/30 split (and, for a random matrix, its own matrix), with the study's 95% bootstrap interval. When it shows minus the Gaussian, each block is its row minus the Gaussian matrix at the same k, from the study's paired differences, and the Gaussian row keeps its own score. Below the families are two reference rows: PCA, fit to each draw's training rows, and all features, with no projection. The columns are the target sizes the study ran (k &lt; d).</p>
    <p>The ink is the only thing computed here. For the score, its density is the score's place between the floor (the majority-class share; for Glass, whose score is balanced accuracy, chance: one over the number of classes) and 100%. For a difference, it is the difference's size up to ${Math.round(100 * M.DIFF_FULL)} points. Differences are paired: within a draw, every row sees the same split. The study records PCA and all features as the Gaussian minus the reference, so their blocks reverse that sign, and their intervals with it.</p>
    <p>Holm's correction runs over the study's ${data.n_primary} primary comparisons (every other family against the Gaussian matrix, and Eq. 4 against Eq. 3, at every data set and k), by paired t-tests.</p>
    <p class="ph-checked"></p>
  </div>
</div>`;

  const $ = (s) => root.querySelector(s);
  const chase = $(".ph-chase"), forme = $(".ph-forme"), roller = $(".ph-roller"), slipEl = $(".ph-slip"), statusEl = $(".ph-status");
  const setInputs = [...root.querySelectorAll(`input[name="ph-set-${id}"]`)];
  const modeInputs = [...root.querySelectorAll(`input[name="ph-mode-${id}"]`)];

  // ---------------------------------------------------------------- state
  const st = { ds: sets[0], mode: "score", bs: [], pos: { r: 2, c: 1 }, touched: false, visible: false, busy: false, sweepAt: 0, printed: false, rolls: 0, want: null };

  // ---------------------------------------------------------------- setting the forme
  function setForme() {
    const ds = st.ds, rs = M.rows(data, ds), n = ds.ks.length;
    st.bs = M.blocks(data, ds, st.mode);
    chase.style.setProperty("--n", n);
    forme.dataset.mode = st.mode;
    let h = `<div class="ph-row ph-spanrow" aria-hidden="true"><span class="ph-stub"></span><span class="ph-spanner">Target size <i>k</i>, as the study ran it</span></div>`;
    h += `<div class="ph-row ph-headrow" aria-hidden="true"><span class="ph-stub"><i>k</i></span>${ds.ks.map((k) => `<span class="ph-k">${k}</span>`).join("")}</div>`;
    rs.forEach((row, r) => {
      const ref = row.kind === "reference";
      h += `<div class="ph-row${ref ? " is-ref" : ""}${row.id === "pca" ? " is-first-ref" : ""}${row.run ? "" : " is-none"}" data-row="${row.id}">`;
      h += `<span class="ph-stub"><b>${esc(row.label)}</b><i>${esc(row.sub)}</i></span>`;
      if (!row.run) {
        h += `<span class="ph-none">${esc(M.notRunNote(ds, row.id))}</span>`;
      } else {
        for (const b of st.bs.filter((x) => x.r === r)) {
          const span = b.c1 - b.c0 + 1;
          const wide = span > 1 ? `<span class="ph-wide">the same at every k</span>` : "";
          h += `<button type="button" class="ph-b${span > 1 ? " is-wide" : ""}${printed(b.print)}" tabindex="-1" data-r="${b.r}" data-c="${b.c0}" data-side="${b.side}" data-level="${b.level}" style="${tints(b.print)};--span:${span}" aria-label="${esc(M.blockLabel(data, ds, b))}"><span class="ph-fig">${esc(b.text)}</span>${b.marks ? `<span class="ph-mk" aria-hidden="true">${b.marks}</span>` : ""}${wide}</button>`;
        }
      }
      h += `</div>`;
    });
    forme.innerHTML = h;
    forme.setAttribute("aria-label", `${M.sayForme(data, ds, st.mode)} Use the arrow keys to read the blocks.`);
    $(".ph-what").textContent = M.sayForme(data, ds, st.mode);
    const key = M.keySteps(ds, st.mode);
    $(".ph-key").innerHTML = `<span class="ph-wedge">${key.patches.map((p) => `<span class="ph-step" data-side="${p.side}" style="${tints(p.print)}"></span>`).join("")}</span>`
      + `<span class="ph-ticks">${key.labels.map((l) => `<span style="left:${(100 * l.at).toFixed(3)}%">${esc(l.text)}</span>`).join("")}</span>`;
    $(".ph-keysay").textContent = M.sayKey(ds, st.mode);
    fixPos();
    select(false);
  }
  const blockEls = () => [...forme.querySelectorAll(".ph-b")];
  function fixPos() {
    // keep the reading point on a block: the same row and column if they exist here, else the nearest
    const rowsWith = [...new Set(st.bs.map((b) => b.r))];
    let r = st.pos.r;
    if (!rowsWith.includes(r)) r = rowsWith.reduce((a, x) => (Math.abs(x - r) < Math.abs(a - r) ? x : a), rowsWith[0]);
    const c = Math.min(st.pos.c, st.ds.ks.length - 1);
    st.pos = { r, c };
  }
  function current() { return M.blockAt(st.bs, st.pos.r, st.pos.c); }
  function select(focus) {
    const b = current();
    for (const el of blockEls()) {
      const on = b && Number(el.dataset.r) === b.r && Number(el.dataset.c) === b.c0;
      el.classList.toggle("is-sel", on);
      el.tabIndex = on ? 0 : -1;
      if (on && focus) el.focus({ preventScroll: true });
    }
    // the reader's finger: the row head and the column heads of the block being read
    forme.querySelectorAll(".ph-row[data-row]").forEach((row, r) => row.classList.toggle("is-on", !!b && b.r === r));
    forme.querySelectorAll(".ph-k").forEach((k, j) => k.classList.toggle("is-on", !!b && b.c0 <= j && j <= b.c1));
    renderSlip(b);
  }

  // ---------------------------------------------------------------- the slip
  function renderSlip(b) {
    if (!b) { slipEl.innerHTML = ""; return; }
    const s = M.slip(data, st.ds, b);
    const lines = s.lines.length ? `<div class="ph-lines" role="table" aria-label="Paired differences">
      <div class="ph-line ph-line-head" role="row"><span role="columnheader">Paired over the ${data.n_draws} draws</span><span role="columnheader">points</span><span role="columnheader">95% interval</span><span role="columnheader">Holm-adjusted <i>p</i></span></div>
      ${s.lines.map((l) => `<div class="ph-line${l.survives ? " is-holm" : ""}" role="row"><span class="ph-ln" role="cell">${esc(l.name)}${l.note ? `<small>${esc(l.note)}</small>` : ""}</span><span class="ph-lv" role="cell">${esc(l.value)}</span><span class="ph-li" role="cell">${esc(l.lo)} to ${esc(l.hi)}</span><span class="ph-lp" role="cell">${l.holm === null ? `<em title="Not one of the study's primary comparisons, so not in Holm's correction">not primary</em>` : `${esc(l.holm)}${l.survives ? " *" : ""}`}${l.equiv ? ` <span class="ph-eq">†</span>` : ""}</span></div>`).join("")}
    </div>` : "";
    slipEl.innerHTML = `<p class="ph-slip-head">${s.head.split(" · ").map((x) => `<span>${esc(x)}</span>`).join(" · ")}</p>
      <p class="ph-slip-value"><span class="ph-big">${esc(s.value)}%</span><span class="ph-slip-ci">${esc(s.metric)}, mean of ${data.n_draws} draws<br>95% interval ${esc(s.lo)} to ${esc(s.hi)}</span></p>
      ${lines}
      <p class="ph-say">${esc(s.say)}</p>`;
  }

  // ---------------------------------------------------------------- resetting the forme
  async function slideOut() {
    if (reduced.matches) return;
    const anims = blockEls().map((el) => el.animate([{ transform: "translateX(0)", opacity: 1 }, { transform: "translateX(-18px)", opacity: 0 }],
      { duration: T.out, easing: "cubic-bezier(.4,0,1,1)", fill: "forwards" }));
    await settle(anims, T.out + 60);
  }
  function slideIn() {
    if (reduced.matches) return;
    for (const el of blockEls()) {
      const c = Number(el.dataset.c), r = Number(el.dataset.r);
      el.animate([{ transform: "translateX(26px)", opacity: 0 }, { transform: "translateX(0)", opacity: 1 }],
        { duration: T.in, delay: c * T.stagger + r * 8, easing: "cubic-bezier(.15,.85,.25,1)", fill: "backwards" });
    }
  }
  async function setDataset(dsId, how = {}) {
    // a choice made while the forme is being reset is set right after it
    if (st.busy) { st.want = dsId; return; }
    const next = M.datasetOf(data, dsId);
    if (next === st.ds && !how.force) return;
    st.busy = true;
    if (!how.instant) await slideOut();
    st.ds = next;
    if (how.pos) st.pos = how.pos;
    setForme();
    if (!how.instant) slideIn();
    st.busy = false;
    const want = st.want;
    st.want = null;
    if (want && want !== st.ds.id) await setDataset(want);
    setInputs.forEach((inp) => { inp.checked = inp.value === st.ds.id; });
  }
  // changing the ink: a roller crosses the forme, and the new ink shows behind it
  function setMode(mode) {
    if (mode === st.mode) return;
    const ghost = reduced.matches ? null : ghostOf(false);
    st.mode = mode;
    modeInputs.forEach((inp) => { inp.checked = inp.value === mode; });
    const focused = forme.contains(document.activeElement);
    setForme();
    if (focused) select(true);
    if (ghost) printOver(ghost);
  }
  /** A copy of the forme as it is now (or uninked, when dry), laid over it until the roller passes. */
  function ghostOf(dry) {
    const g = forme.cloneNode(true);
    g.classList.add("ph-ghost");
    g.removeAttribute("role"); g.removeAttribute("aria-label");
    g.setAttribute("aria-hidden", "true");
    g.inert = true;
    g.querySelectorAll(".ph-b").forEach((b) => { b.classList.remove("is-sel"); if (dry) b.classList.add("is-dry"); });
    return g;
  }
  function printOver(ghost) {
    chase.querySelectorAll(".ph-ghost").forEach((g) => g.remove());
    const top = forme.offsetTop, left = forme.offsetLeft, w = forme.offsetWidth, h = forme.offsetHeight;
    Object.assign(ghost.style, { top: `${top}px`, left: `${left}px`, width: `${w}px` });
    chase.append(ghost);
    Object.assign(roller.style, { top: `${top - 7}px`, height: `${h + 14}px` });
    roller.hidden = false;
    const opts = { duration: T.roll, easing: "cubic-bezier(.45,.05,.35,1)", fill: "forwards" };
    const a = ghost.animate([{ clipPath: "inset(0 0 0 0)" }, { clipPath: "inset(0 0 0 100%)" }], opts);
    const b = roller.animate([{ transform: `translateX(${left - 5}px)` }, { transform: `translateX(${left + w - 5}px)` }], opts);
    const turn = ++st.rolls;
    settle([a, b], T.roll + 120).then(() => { ghost.remove(); if (turn === st.rolls) roller.hidden = true; });
  }

  // ---------------------------------------------------------------- the reader's hand
  function touch() {
    if (st.touched) return;
    st.touched = true;
    slipEl.setAttribute("aria-live", "polite");
    statusEl.textContent = "Tap a block, or use the arrow keys, to read another.";
    clearTimeout(st.timer);
  }
  setInputs.forEach((inp) => inp.addEventListener("change", () => { touch(); setDataset(inp.value); }));
  modeInputs.forEach((inp) => inp.addEventListener("change", () => { touch(); setMode(inp.value); }));
  forme.addEventListener("click", (e) => {
    const el = e.target.closest(".ph-b");
    if (!el) return;
    touch();
    st.pos = { r: Number(el.dataset.r), c: Number(el.dataset.c) };
    select(false);
  });
  forme.addEventListener("keydown", (e) => {
    if (!e.target.closest(".ph-b")) return;
    const next = M.move(st.bs, st.pos, e.key);
    if (next === st.pos && !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(e.key)) return;
    e.preventDefault();
    touch();
    st.pos = next;
    select(true);
  });
  root.addEventListener("pointerdown", touch);
  root.addEventListener("keydown", (e) => { if (e.key !== "Tab" && e.key !== "Shift") touch(); });

  // ---------------------------------------------------------------- alive: reading down k = 2, then the next data set
  function readNext() {
    st.timer = 0;
    if (st.touched || reduced.matches || !st.visible || document.hidden) return;
    const order = M.sweep(st.bs, st.ds);
    if (st.sweepAt < order.length) {
      st.pos = order[st.sweepAt++];
      select(false);
      schedule(T.read);
      return;
    }
    // the column is read: reset the forme with the next data set; after all five, change the ink
    st.sweepAt = 0;
    const i = sets.indexOf(st.ds), last = i === sets.length - 1;
    const go = async () => {
      const next = sets[(i + 1) % sets.length];
      await setDataset(next.id, { force: true, pos: { r: 0, c: Math.max(0, next.ks.indexOf(M.SWEEP_K)) } });
      if (last && !st.touched) setMode(st.mode === "score" ? "diff" : "score");
      st.sweepAt = 1;
      schedule(T.read);
    };
    go();
  }
  function schedule(ms) {
    clearTimeout(st.timer);
    if (st.touched || reduced.matches) return;
    st.timer = setTimeout(readNext, ms);
  }
  function wake() {
    if (st.touched || reduced.matches) return;
    if (st.visible && !document.hidden) { if (!st.timer) schedule(T.firstRead); }
    else { clearTimeout(st.timer); st.timer = 0; }
  }
  new IntersectionObserver((es) => {
    st.visible = es[es.length - 1].isIntersecting;
    // the first impression: the forme is set dry, and the roller inks it as it comes into view
    if (st.visible && !st.printed) { st.printed = true; if (!reduced.matches) printOver(ghostOf(true)); }
    wake();
  }, { threshold: 0.2 }).observe(forme);
  document.addEventListener("visibilitychange", wake);
  reduced.addEventListener("change", () => { if (reduced.matches) { clearTimeout(st.timer); st.timer = 0; statusEl.textContent = ""; } else wake(); });

  // ---------------------------------------------------------------- first print: Iris, the score, Eq. 4 at k = 2
  setInputs.forEach((inp) => { inp.checked = inp.value === st.ds.id; });
  modeInputs.forEach((inp) => { inp.checked = inp.value === st.mode; });
  setForme();
  if (!reduced.matches) {
    statusEl.textContent = `Reading down k = ${M.SWEEP_K} on its own · tap any block to take over`;
    st.sweepAt = 0;
  }

  // the latest check run, quietly, when it exists
  fetch(src.replace(/\.json$/, ".checks.json")).then((r) => (r.ok ? r.json() : null)).then((c) => {
    if (c && c.line) $(".ph-checked").textContent = c.line;
  }).catch(() => {});
}
