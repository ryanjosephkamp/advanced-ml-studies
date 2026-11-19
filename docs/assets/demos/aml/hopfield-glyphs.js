// The eighteen characters (chapter II, live plate): the study's 18 bitmaps set as a type specimen
// sheet, the way a foundry prints a specimen of a bitmap face: six across and three down in file
// order, each on its own body with its number and its count of on pixels set small beneath it.
// Beside the sheet is a proof of one character, pulled from either query file: the query is the
// galley, characters.csv is the copy, and every pixel the query flips is marked as a proofreader
// marks a galley, in blue pencil: ink the query added is struck through, ink it took away gets a
// caret, and each mark is repeated in the nearer margin on its row. On its own, the sheet proofs one
// character after another; a touch stops it.
//
// Drawn in SVG with the page's inks as CSS variables, so both printings come for free. Every count
// and sentence comes from hopfield-glyphs-model.js, which the check script tests.
const VQ = new URL(import.meta.url).search; // the page's version, passed on so a browser never mixes versions
const M = await import(new URL(`./hopfield-glyphs-model.js${VQ}`, import.meta.url).href);

const KIND = "hopfield-glyphs";
const NS = "http://www.w3.org/2000/svg";
const reduced = matchMedia("(prefers-reduced-motion: reduce)");
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

/** One path for all inked runs, so adjacent pixels print as one solid shape with no seams. */
const runPath = (bits, print, s = 1, x0 = 0, y0 = 0) =>
  M.runs(bits, print).map(([r, c, n]) => `M${x0 + c * s} ${y0 + r * s}h${n * s}v${s}h${-n * s}z`).join("") || "M0 0";

// ------------------------------------------------------------------ the proofreader's marks (proof units: one pixel = 10)
const U = 10;
const MARGIN = 48; // each margin holds four marks and their separators (the most any half-row of either file has)
const strikePath = (x, y) => `M${x - 1.6} ${y + U + 1.6}L${x + U + 1.6} ${y - 1.6}`;
const caretPath = (x, y) => `M${x + 2.2} ${y + 8.4}L${x + 5} ${y + 2.6}L${x + 7.8} ${y + 8.4}`;
// the dele: a looped stroke, as a proofreader writes "take out" in the margin (drawn in an 8 by 9 box)
const delePath = (x, y) => `M${x + 6.4} ${y}C${x + 3.6} ${y + 0.6} ${x + 1.6} ${y + 3.2} ${x + 2.6} ${y + 5.4}C${x + 3.4} ${y + 7.2} ${x + 6.4} ${y + 6.8} ${x + 6.2} ${y + 5}C${x + 6} ${y + 3.4} ${x + 3.4} ${y + 3.8} ${x + 2.4} ${y + 5.8}C${x + 1.6} ${y + 7.4} ${x + 1.2} ${y + 8.4} ${x + 0.4} ${y + 9}`;
const slashPath = (x, y) => `M${x} ${y + 9}L${x + 3.4} ${y}`;

// ------------------------------------------------------------------ the plate
async function mount(fig) {
  const root = fig.querySelector(".demo-root");
  const data = await (await fetch(fig.dataset.src)).json();
  const S = M.shape(data);
  const sheet = M.sheetStats(S);
  const files = Object.fromEntries(S.ids.map((id) => [id, M.fileStats(S, id)]));

  const st = {
    k: 0, noise: S.ids[0], print: "strokes",
    auto: !reduced.matches && !params.has("still"), visible: false, last: performance.now(),
  };
  if (!st.auto) st.k = 8; // the still state: the closest pair's first character, under the test file

  // ---------------------------------------------------------------- the controls
  const radios = (label, items, cls) => {
    const g = el("div", { class: `hg-radios ${cls}`, role: "radiogroup", "aria-label": label });
    g.append(el("span", { class: "hg-lab", "aria-hidden": "true", text: label }));
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
  const noiseR = radios("Query", [["none", "None"], ...S.ids.map((id) => [id, S.queries[id].name])], "hg-noise");
  const printR = radios("Ink", [["strokes", "Letterforms"], ["on", "On pixels"]], "hg-print");
  const play = el("button", { type: "button", class: "hg-play" }); // its label says what a press does, so it carries no aria-pressed
  const fileNote = el("p", { class: "hg-filenote" });

  // ---------------------------------------------------------------- the specimen sheet
  const grid = el("div", { class: "hg-grid", role: "radiogroup", "aria-label": "The 18 characters, six across, in file order" });
  const sorts = [];
  for (let k = 0; k < S.n; k++) {
    const svg = el("svg:svg", { viewBox: "-0.5 -0.5 13 13", "aria-hidden": "true", class: "hg-glyph" });
    svg.append(el("svg:rect", { class: "hg-em", x: -0.5, y: -0.5, width: 13, height: 13 }));
    const ink = el("svg:path", { class: "hg-ink" }), marks = el("svg:g", { class: "hg-tmarks" });
    svg.append(ink, marks);
    const b = el("button", { type: "button", role: "radio", class: "hg-sort", "data-k": String(k) }, [
      svg, el("span", { class: "hg-cap" }, [el("span", { class: "hg-no", text: String(k) }), el("span", { class: "hg-count", text: String(S.on[k]) })]),
    ]);
    sorts.push({ b, ink, marks });
    grid.append(b);
  }
  const sheetEl = el("div", { class: "hg-sheet" }, [
    el("p", { class: "hg-sheethead" }, [el("span", { text: "data/characters.csv" }), el("span", { text: "18 characters · 12 × 12" })]),
    grid,
    el("p", { class: "hg-key", text: "Under each: its number, counting from 0, and how many of its 144 pixels are on." }),
  ]);

  // ---------------------------------------------------------------- the proof
  const W = MARGIN * 2 + 12 * U, H = 12 * U;
  const proofSvg = el("svg:svg", { class: "hg-proofsvg", viewBox: `0 -6 ${W} ${H + 12}`, role: "img" });
  const gridLines = el("svg:path", { class: "hg-pgrid" });
  {
    let d = "";
    for (let i = 0; i <= 12; i++) d += `M${MARGIN + i * U} 0V${H}M${MARGIN} ${i * U}H${MARGIN + 12 * U}`;
    gridLines.setAttribute("d", d);
  }
  const proofInk = el("svg:path", { class: "hg-pink" });
  const proofMarks = el("svg:g", { class: "hg-pmarks" });
  const marginMarks = el("svg:g", { class: "hg-mmarks" });
  proofSvg.append(
    el("svg:rect", { class: "hg-pbody", x: MARGIN, y: 0, width: 12 * U, height: H }),
    gridLines, proofInk,
    el("svg:rect", { class: "hg-pframe", x: MARGIN, y: 0, width: 12 * U, height: H }),
    el("svg:path", { class: "hg-pmargin", d: `M${MARGIN - 4} -4V${H + 4}M${MARGIN + 12 * U + 4} -4V${H + 4}` }),
    proofMarks, marginMarks);
  const proofHead = el("p", { class: "hg-proofhead" });
  const tally = el("p", { class: "hg-tally" });
  const proofEl = el("div", { class: "hg-proof" }, [proofHead, proofSvg, tally]);

  const read = el("p", { class: "hg-read", "aria-live": "polite" });
  const legend = el("div", { class: "hg-legend" });
  const checked = el("p", { class: "hg-checked" });
  root.append(el("div", { class: "hg" }, [
    el("div", { class: "hg-bar" }, [noiseR, printR, play]),
    fileNote,
    el("div", { class: "hg-body" }, [sheetEl, proofEl]),
    read, legend, checked,
  ]));

  fetch(fig.dataset.src.replace(/\.json$/, ".checks.json")).then((r) => (r.ok ? r.json() : null)).then((c) => {
    if (c && c.line) checked.textContent = c.line;
  }).catch(() => {});

  // ---------------------------------------------------------------- drawing
  function drawSheet() {
    const id = st.noise === "none" ? null : st.noise;
    for (let k = 0; k < S.n; k++) {
      const { b, ink, marks } = sorts[k];
      const sel = k === st.k;
      const bits = sel && id ? S.queries[id].rows[k] : S.chars[k];
      ink.setAttribute("d", runPath(bits, st.print));
      marks.replaceChildren();
      if (sel && id) {
        // the proof in small: struck pixels in blue pencil, missing ones outlined
        let strike = "", caret = "";
        for (const f of S.queries[id].flips[k]) {
          if (M.markOf(f, st.print) === "strike") strike += `M${f.c} ${f.r}h1v1h-1z`;
          else caret += `M${f.c + 0.16} ${f.r + 0.16}h0.68v0.68h-0.68z`;
        }
        marks.append(el("svg:path", { class: "hg-tstrike", d: strike || "M0 0" }), el("svg:path", { class: "hg-tcaret", d: caret || "M0 0" }));
      }
      b.setAttribute("aria-checked", String(sel));
      b.tabIndex = sel ? 0 : -1;
      b.setAttribute("aria-label", M.sayGlyph(k, S.on[k], sel && id ? S.queries[id].flips[k].length : undefined));
    }
  }

  let drawn = 0;
  function drawProof() {
    const id = st.noise === "none" ? null : st.noise;
    const p = M.proof(S, st.k, id, st.print);
    const bits = id ? S.queries[id].rows[st.k] : S.chars[st.k];
    proofInk.setAttribute("d", runPath(bits, st.print, U, MARGIN, 0));
    drawn++;
    const animate = !reduced.matches;
    // the marks in the galley, in reading order, each drawn in as a pencil stroke
    proofMarks.replaceChildren();
    p.flips.forEach((f, n) => {
      const x = MARGIN + f.c * U, y = f.r * U, kind = M.markOf(f, st.print);
      const d = kind === "strike" ? strikePath(x, y) : caretPath(x, y);
      const g = el("svg:g", { class: `hg-mk hg-${kind}` });
      if (kind === "strike") g.append(el("svg:path", { class: "hg-halo", d }));
      else g.append(el("svg:rect", { class: "hg-hole", x: x + 0.9, y: y + 0.9, width: U - 1.8, height: U - 1.8 }));
      g.append(el("svg:path", { class: "hg-pen", d, pathLength: "1" }));
      // the pencil works through the galley in reading order, in at most 1.2 s however many marks there are
      if (animate) { const t = `${(n * Math.min(0.06, 1.2 / p.flips.length)).toFixed(3)}s`; for (const q of g.children) q.style.animationDelay = t; }
      proofMarks.append(g);
    });
    // the same marks in the margins, on their rows
    marginMarks.replaceChildren();
    const put = (list, r, side) => {
      if (!list.length) return;
      const per = Math.min(11.6, (MARGIN - 7) / list.length), w = list.length * per;
      let x = side === "left" ? MARGIN - 6 - w : MARGIN + 12 * U + 7;
      const y = r * U + 0.5;
      list.forEach((kind) => {
        const g = el("svg:g", { class: `hg-mm hg-m-${kind}` });
        if (kind === "strike") g.append(el("svg:path", { class: "hg-pen", d: delePath(x, y), pathLength: "1" }));
        else g.append(el("svg:rect", { class: "hg-ins", x: x + 1, y: y + 2, width: 5.6, height: 5.6 }));
        g.append(el("svg:path", { class: "hg-pen hg-slash", d: slashPath(x + 7.6, y), pathLength: "1" }));
        if (animate) g.querySelectorAll(".hg-pen").forEach((q) => { q.style.animationDelay = `${(0.25 + r * 0.05).toFixed(2)}s`; });
        marginMarks.append(g);
        x += per;
      });
    };
    for (let r = 0; r < M.SIDE; r++) { put(p.margins.left[r], r, "left"); put(p.margins.right[r], r, "right"); }

    const file = id ? files[id] : null;
    proofHead.replaceChildren(el("span", { class: "hg-ph-k", text: `Proof · character ${st.k}` }), el("span", { text: id ? S.queries[id].name.toLowerCase() : "clean copy" }));
    proofSvg.setAttribute("aria-label", `${M.sayProof(p, file)} ${id ? M.sayTally(p) : ""}`.trim());
    tally.textContent = M.sayTally(p);
    // the tour turns a character every few seconds: a screen reader hears the reading only once the reader takes over
    read.setAttribute("aria-live", st.auto ? "off" : "polite");
    read.textContent = M.sayProof(p, file);
    fileNote.textContent = id ? M.sayFile(file) : M.saySheet(sheet);
  }

  function drawLegend() {
    const sw = (cls, d, extra = []) => {
      const s = el("svg:svg", { viewBox: "-2 -2 14 14", "aria-hidden": "true" });
      s.append(...extra, el("svg:path", { class: cls, d }));
      return s;
    };
    const ink = st.print === "on" ? "On pixels (bit 1)" : "Off pixels (bit 0), the letterforms";
    legend.replaceChildren(
      el("span", {}, [sw("hg-k-ink", "M0 0h10v10h-10z"), document.createTextNode(`Ink: ${ink.charAt(0).toLowerCase()}${ink.slice(1)}`)]),
      el("span", {}, [sw("hg-k-pen", strikePath(0, 0), [el("svg:path", { class: "hg-k-ink", d: "M0 0h10v10h-10z" }), el("svg:path", { class: "hg-k-halo", d: strikePath(0, 0) })]), document.createTextNode("struck: ink the query added")]),
      el("span", {}, [sw("hg-k-pen", caretPath(0, 0), [el("svg:path", { class: "hg-k-hole", d: "M0.9 0.9h8.2v8.2h-8.2z" })]), document.createTextNode("caret: ink it took away")]),
      el("p", { class: "hg-legnote", text: `${M.sayInk(st.print)} ${M.sayMarks(st.print)}` }),
    );
  }

  function render() {
    setRadios(noiseR, st.noise); setRadios(printR, st.print);
    play.textContent = st.auto ? "Pause" : "Play";
    play.classList.toggle("is-on", st.auto);
    play.disabled = reduced.matches;
    drawSheet(); drawProof(); drawLegend();
  }

  // ---------------------------------------------------------------- the reader's hand
  const touch = () => { st.auto = false; };
  noiseR.addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    touch(); st.noise = b.dataset.v; render();
  });
  printR.addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    touch(); st.print = b.dataset.v; render();
  });
  grid.addEventListener("click", (e) => {
    const b = e.target.closest(".hg-sort"); if (!b) return;
    touch(); st.k = Number(b.dataset.k);
    // a tap on a character with no query laid over shows it clean; the proof still enlarges it
    render();
  });
  grid.addEventListener("keydown", (e) => {
    const mv = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
    let k = null;
    if (mv) k = M.step(st.k, mv[0], mv[1], S.n);
    else if (e.key === "Home") k = 0;
    else if (e.key === "End") k = S.n - 1;
    if (k === null) return;
    e.preventDefault(); touch(); st.k = k; render();
    sorts[k].b.focus();
  });
  play.addEventListener("click", () => {
    st.auto = !st.auto && !reduced.matches;
    st.last = performance.now();
    render();
  });
  reduced.addEventListener("change", () => { if (reduced.matches) st.auto = false; render(); });

  // ---------------------------------------------------------------- alive: one character after another, while on screen
  const tick = () => {
    const now = performance.now();
    // off screen or in a hidden tab, the clock waits, so the reader never comes back to a skipped page
    if (!st.auto || !st.visible || document.hidden) { st.last = now; return; }
    if (now - st.last >= M.TOUR_MS) {
      st.last = now;
      st.k = (st.k + 1) % S.n;
      drawSheet(); drawProof();
    }
  };
  setInterval(tick, 250);
  new IntersectionObserver((es) => { st.visible = es[0].isIntersecting; }, { threshold: 0.15 }).observe(root);

  render();
}

document.querySelectorAll(`figure.plate-live[data-demo="${KIND}"]`).forEach((f) => mount(f).catch((err) => {
  const r = f.querySelector(".demo-root");
  if (r) r.textContent = "The live plate could not load its data.";
  console.error(err);
}));
