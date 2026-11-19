// The emd-imf plate: one EEG window taken apart on a mixing desk. The five intrinsic mode functions
// (IMFs) and the residue are channel strips, each with its own small trace, a meter of its share of the
// window's variance, a fader that scales it from 0 to 1, and a mute key. The master trace is an
// oscilloscope screen: the signal against the sum of the pieces as mixed (channel 1) and the residual
// between them (channel 2), at one scale. Left alone, it plays a slow loop that takes every piece out
// and brings them back one at a time, from the residue up to IMF 1. The numbers and sentences come
// from emd-imf-model.js; this file only draws them, on plain 2D canvases.
const VQ = new URL(import.meta.url).search; // the page's version, passed on so a browser never mixes versions
const M = await import(new URL(`./emd-imf-model.js${VQ}`, import.meta.url).href);

const KIND = "emd-imf";
const reduced = matchMedia("(prefers-reduced-motion: reduce)");
const darkQuery = matchMedia("(prefers-color-scheme: dark)");

const link = document.createElement("link");
link.rel = "stylesheet";
link.href = new URL(`./${KIND}.css${VQ}`, import.meta.url).href;
document.head.append(link);

for (const fig of document.querySelectorAll(`figure.plate-live[data-demo="${KIND}"]`)) mount(fig).catch((e) => {
  const root = fig.querySelector(".demo-root");
  if (root) root.innerHTML = `<p class="ei-fail">The live plate could not load its data (${String(e.message || e).replace(/[<>&]/g, "")}).</p>`;
});

const esc = (s) => String(s).replace(/[<>&"]/g, (c) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;", '"': "&quot;" })[c]);

async function mount(fig) {
  const root = fig.querySelector(".demo-root");
  const data = await (await fetch(fig.dataset.src)).json();
  const prep = M.prepare(data);
  const P = prep.pieces, N = P.length, uid = Math.random().toString(36).slice(2, 8);
  const st = {
    faders: new Array(N).fill(1), vel: new Array(N).fill(0), target: new Array(N).fill(1), muted: new Array(N).fill(false),
    auto: !reduced.matches, returning: false, clock: 0, beamClock: 0, visible: false, touched: false, cursor: null, drag: -1,
  };
  let gains = M.effective(st.faders, st.muted), mx = M.mix(prep, gains), phase = null;

  const counts = Object.entries(data.imf_counts).map(([k, v]) => [+k, v]);
  const five = data.imf_counts[String(prep.nImfs)];
  const others = counts.filter(([k]) => k !== prep.nImfs);
  const n0 = (x) => x.toLocaleString("en-US");

  root.innerHTML = `
<div class="ei">
  <div class="ei-screen">
    <canvas class="ei-scope" tabindex="0" role="img"></canvas>
    <p class="ei-read" aria-live="off"></p>
    <p class="ei-announce" aria-live="polite"></p>
  </div>
  <div class="ei-desk" role="group" aria-label="The desk: one strip for each piece of the window">
    <p class="ei-key">Each strip: the piece, its frequency in hertz and its share of the window’s variance (both computed in the page from the file), its level, and a mute key.</p>
    <div class="ei-strips">
      ${P.map((p, k) => `
      <div class="ei-strip" data-k="${k}">
        <span class="ei-name">${p.residue ? `<span class="ei-long">Residue</span><abbr class="ei-short" title="Residue">Res.</abbr>` : `IMF&nbsp;${p.label.split(" ")[1]}`}</span>
        <span class="ei-freq" title="${p.residue ? "The residue is not an IMF and has no frequency of its own." : "Mean absolute instantaneous frequency, computed in the page"}">${p.residue ? "trend" : M.hz(p.freq).replace(" Hz", `<span class="ei-w">&nbsp;Hz</span>`)}</span>
        <canvas class="ei-mini" aria-hidden="true"></canvas>
        <span class="ei-meter" aria-hidden="true"><i class="ei-fill"></i><i class="ei-mark"></i></span>
        <span class="ei-share">${M.pct(p.share)}</span>
        <div class="ei-fader" role="slider" tabindex="0" aria-orientation="vertical" aria-valuemin="0" aria-valuemax="1" aria-label="${esc(p.label)}, level">
          <span class="ei-ticks" aria-hidden="true">${[1, 0.75, 0.5, 0.25, 0].map((v) => `<i style="--v:${v}"${v === 1 || v === 0 || v === 0.5 ? ' class="ei-tl"' : ""}>${v === 1 ? "<b>1</b>" : v === 0 ? "<b>0</b>" : ""}</i>`).join("")}</span>
          <span class="ei-slot" aria-hidden="true"></span>
          <span class="ei-cap" aria-hidden="true"></span>
        </div>
        <output class="ei-level"></output>
        <button type="button" class="ei-mute" aria-pressed="false" aria-label="Mute ${esc(p.label)}">mute</button>
      </div>`).join("")}
      <div class="ei-master">
        <span class="ei-name">Sum</span>
        <dl class="ei-sum">
          <div><dt>Pieces in</dt><dd class="ei-in"></dd></div>
          <div><dt>Residual</dt><dd class="ei-lost"></dd></div>
          <div><dt>Largest gap</dt><dd class="ei-gap"></dd></div>
        </dl>
        <div class="ei-keys">
          <button type="button" class="ei-btn ei-play" aria-pressed="false"></button>
          <button type="button" class="ei-btn" data-act="in">All in</button>
          <button type="button" class="ei-btn" data-act="out">All out</button>
        </div>
      </div>
      <span class="ei-g ei-g-f2" aria-hidden="true" style="grid-column: 1 / span ${data.f2_imfs}">the study’s IMF features use these ${M.count(data.f2_imfs)}</span>
      <span class="ei-g ei-g-res" aria-hidden="true" style="grid-column: ${N}">kept apart</span>
    </div>
  </div>
  <p class="ei-say"></p>
  <ul class="ei-notes">
    <li><b>The window.</b> The first of the study’s ${n0(data.n_windows)} ${esc(data.dataset)} windows (${esc(data.dataset_long)}, ${data.n_subjects} subjects): two seconds from a cue to imagine moving the left or right fist, at ${data.sfreq} Hz, so ${prep.n} samples. It is the window of the paper’s Figure ${data.figure.number}: “${esc(data.figure.caption)}”</li>
    <li><b>The channel.</b> Not one electrode: ${M.list(data.channels)} averaged into one signal and then z-scored, as the original pipeline did. The values have no unit; 1 on the screen is one standard deviation (SD) of this window, whose mean is 0 and SD 1 within the file’s rounding.</li>
    <li><b>The sum.</b> The file keeps ${M.count(data.digits)} significant digits. At every one of the ${prep.n} samples, the ${M.count(prep.nImfs)} IMFs and the residue add up to the signal within that rounding; the largest gap is ${M.gap(prep.maxGap)}.</li>
    <li><b>Computed in the page.</b> A piece’s share is its variance over the ${prep.n} samples divided by the signal’s; the residual’s share is computed the same way. The ${M.count(N)} shares add to ${M.pct(prep.sumShares)}, not 100%, because the pieces are not exactly uncorrelated. An IMF’s frequency is its mean absolute instantaneous frequency from the Hilbert transform, as the study’s IMF features compute it; the residue is not an IMF and has none.</li>
    <li><b>In the study.</b> EMD returned ${M.count(prep.nImfs)} IMFs for ${n0(five)} of the ${n0(data.n_windows)} ${esc(data.dataset)} windows (${M.list(others.map(([k, v]) => `${M.count(k)} for ${n0(v)}`))}). The IMF features use the first ${M.count(data.f2_imfs)} IMFs of each window. The original code counted the residue as an IMF and padded every window to eight rows; the corrected pipeline keeps the residue apart.</li>
    <li><b>Traces.</b> Lines join the ${prep.n} samples; the cursor stops only at samples. A fader multiplies its piece by the level under it, in steps of 0.01.</li>
  </ul>
  <p class="ei-checked" hidden></p>
</div>`;

  const $ = (s) => root.querySelector(s);
  const scope = $(".ei-scope"), sctx = scope.getContext("2d"), read = $(".ei-read"), say = $(".ei-say"), play = $(".ei-play");
  const strips = [...root.querySelectorAll(".ei-strip")].map((el) => ({
    el, mini: el.querySelector(".ei-mini"), fader: el.querySelector(".ei-fader"), cap: el.querySelector(".ei-cap"),
    level: el.querySelector(".ei-level"), mute: el.querySelector(".ei-mute"), fill: el.querySelector(".ei-fill"), mark: el.querySelector(".ei-mark"),
  }));
  const meterMax = Math.ceil(Math.max(...P.map((p) => p.share)) * 10 - 1e-9) / 10;
  strips.forEach((s, k) => { s.mark.style.left = `${(100 * P[k].share) / meterMax}%`; });

  // ---------------------------------------------------------------------------------------- colors
  let C = {};
  function readColors() {
    const cs = getComputedStyle(fig);
    const v = (k) => cs.getPropertyValue(`--${k}`).trim();
    C = { ink: v("ink"), ink2: v("ink-2"), paper: v("paper"), paper2: v("paper-2"), rule: v("rule"), rule2: v("rule-2"), accent: v("accent-ink"),
          type: v("type") || "monospace", serif: v("serif") || "serif" };
    C.dark = M.relLum(toHex(C.paper)) < 0.3;
  }
  function toHex(c) {
    if (/^#/.test(c)) return c.length === 4 ? "#" + [...c.slice(1)].map((x) => x + x).join("") : c.slice(0, 7);
    const m = c.match(/[\d.]+/g);
    return m ? "#" + m.slice(0, 3).map((x) => Math.round(+x).toString(16).padStart(2, "0")).join("") : "#888888";
  }
  const alpha = (c, a) => { const h = toHex(c); return `rgba(${parseInt(h.slice(1, 3), 16)},${parseInt(h.slice(3, 5), 16)},${parseInt(h.slice(5, 7), 16)},${a})`; };

  // ---------------------------------------------------------------------------------------- sizing
  let SW = 0, SH = 0, MW = 0, MH = 0, dpr = 1, DPR = 0;
  function size() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (dpr !== DPR) { DPR = dpr; SW = MW = 0; }
    const w = scope.parentElement.clientWidth;
    const h = Math.round(w < 480 ? 340 : Math.min(400, Math.max(340, w * 0.47)));
    if (w !== SW || h !== SH) { SW = w; SH = h; scope.style.height = `${h}px`; scope.width = Math.round(w * dpr); scope.height = Math.round(h * dpr); }
    const mw = strips[0].mini.clientWidth, mh = strips[0].mini.clientHeight;
    if (mw !== MW || mh !== MH) {
      MW = mw; MH = mh;
      for (const s of strips) { s.mini.width = Math.round(mw * dpr); s.mini.height = Math.round(mh * dpr); }
    }
  }

  // ---------------------------------------------------------------------------------------- the screen
  // Two lanes of six divisions each at 1 SD per division: channel 1 (signal and rebuilt) above, channel 2
  // (the residual) below; eight divisions of 0.25 s across.
  const DIV_X = 8, DIV_Y = 12, SEC = 2, LANE = DIV_Y / 2;
  function screenBox() {
    const L = 6, R = 6, T = 6, B = 24;
    return { x0: L, x1: SW - R, y0: T, y1: SH - B, w: SW - L - R, h: SH - T - B };
  }
  function drawScope() {
    if (!SW) return;
    const ctx = sctx, b = screenBox(), n = prep.n;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, SW, SH);
    const X = (i) => b.x0 + (prep.t[i] / SEC) * b.w;
    const YA = (v) => b.y0 + ((LANE / 2 - v) / DIV_Y) * b.h, YB = (v) => b.y0 + ((LANE + LANE / 2 - v) / DIV_Y) * b.h;
    // the screen, with its bezel
    const r = 9;
    const round = (x, y, w, h, rr) => { ctx.beginPath(); ctx.moveTo(x + rr, y); ctx.arcTo(x + w, y, x + w, y + h, rr); ctx.arcTo(x + w, y + h, x, y + h, rr); ctx.arcTo(x, y + h, x, y, rr); ctx.arcTo(x, y, x + w, y, rr); ctx.closePath(); };
    round(b.x0, b.y0, b.w, b.h, r);
    ctx.fillStyle = alpha(C.paper, C.dark ? 0.86 : 0.78); ctx.fill();
    ctx.save(); ctx.clip();
    // the graticule: dotted divisions, the two zero lines with fine ticks, the lanes' divider
    ctx.strokeStyle = alpha(C.ink, C.dark ? 0.16 : 0.2); ctx.lineWidth = 0.8; ctx.setLineDash([1, 3]);
    ctx.beginPath();
    for (let i = 1; i < DIV_X; i++) { const x = b.x0 + (i / DIV_X) * b.w; ctx.moveTo(x, b.y0); ctx.lineTo(x, b.y1); }
    for (let j = 1; j < DIV_Y; j++) { if (j === LANE) continue; const y = b.y0 + (j / DIV_Y) * b.h; ctx.moveTo(b.x0, y); ctx.lineTo(b.x1, y); }
    ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = alpha(C.ink, C.dark ? 0.32 : 0.36); ctx.lineWidth = 0.8; ctx.beginPath();
    for (const y0 of [YA(0), YB(0)]) {
      ctx.moveTo(b.x0, y0); ctx.lineTo(b.x1, y0);
      for (let i = 1; i < DIV_X * 5; i++) { const x = b.x0 + (i / (DIV_X * 5)) * b.w, l = i % 5 ? 2.5 : 4.5; ctx.moveTo(x, y0 - l); ctx.lineTo(x, y0 + l); }
    }
    ctx.stroke();
    ctx.strokeStyle = alpha(C.ink, C.dark ? 0.5 : 0.55); ctx.lineWidth = 1.1;
    ctx.beginPath(); ctx.moveTo(b.x0, b.y0 + b.h / 2); ctx.lineTo(b.x1, b.y0 + b.h / 2); ctx.stroke();

    // what is lost: hatched between the signal and the rebuilt trace, and between the residual and zero
    const step = b.w / n;
    const lost = (ya, yb) => {
      if (step >= 2) {
        ctx.strokeStyle = alpha(C.ink2, C.dark ? 0.5 : 0.42); ctx.lineWidth = 0.9; ctx.beginPath();
        for (let i = 0; i < n; i++) { const x = X(i), a = ya(i), c = yb(i); if (Math.abs(a - c) < 0.4) continue; ctx.moveTo(x, a); ctx.lineTo(x, c); }
        ctx.stroke();
      } else {
        ctx.fillStyle = alpha(C.ink2, C.dark ? 0.32 : 0.26); ctx.beginPath();
        for (let i = 0; i < n; i++) (i ? ctx.lineTo(X(i), ya(i)) : ctx.moveTo(X(i), ya(i)));
        for (let i = n - 1; i >= 0; i--) ctx.lineTo(X(i), yb(i));
        ctx.closePath(); ctx.fill();
      }
    };
    lost((i) => YA(prep.x[i]), (i) => YA(mx.rebuilt[i]));
    lost((i) => YB(mx.residual[i]), () => YB(0));

    const trace = (arr, Y, color, width, glow = 0) => {
      ctx.beginPath();
      for (let i = 0; i < n; i++) (i ? ctx.lineTo(X(i), Y(arr[i])) : ctx.moveTo(X(i), Y(arr[i])));
      ctx.lineJoin = "round"; ctx.lineCap = "round";
      if (glow) { ctx.save(); ctx.shadowColor = alpha(color, 0.9); ctx.shadowBlur = glow; ctx.strokeStyle = alpha(color, 0.55); ctx.lineWidth = width + 1.2; ctx.stroke(); ctx.restore(); }
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke();
    };
    // channel 1: the signal as a fine reference, the rebuilt trace in the chapter's ink (phosphor at night)
    trace(prep.x, YA, alpha(C.ink, C.dark ? 0.62 : 0.6), 1);
    trace(mx.rebuilt, YA, C.accent, 1.6, C.dark ? 7 : 0);
    // channel 2: the residual
    trace(mx.residual, YB, C.dark ? alpha(C.ink, 0.85) : C.ink, 1.15);

    // the beam, at night while the plate plays: the window drawn again at its own pace
    if (C.dark && st.auto && !reduced.matches) {
      const bi = M.beamAt(st.beamClock, n, prep.fs), tail = Math.round(0.25 * prep.fs);
      ctx.save(); ctx.lineCap = "round";
      for (let q = Math.max(1, bi - tail); q <= bi; q++) {
        const a = (q - (bi - tail)) / tail;
        ctx.strokeStyle = alpha(C.accent, 0.15 + 0.7 * a); ctx.lineWidth = 1.6 + 1.6 * a; ctx.shadowColor = C.accent; ctx.shadowBlur = 10 * a;
        ctx.beginPath(); ctx.moveTo(X(q - 1), YA(mx.rebuilt[q - 1])); ctx.lineTo(X(q), YA(mx.rebuilt[q])); ctx.stroke();
      }
      ctx.fillStyle = C.accent; ctx.shadowColor = C.accent; ctx.shadowBlur = 12;
      ctx.beginPath(); ctx.arc(X(bi), YA(mx.rebuilt[bi]), 2.6, 0, 2 * Math.PI); ctx.fill();
      ctx.restore();
    }

    // the cursor: one sample, read in all three traces
    if (st.cursor != null) {
      const i = st.cursor, x = X(i);
      ctx.strokeStyle = alpha(C.ink, 0.75); ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(x, b.y0); ctx.lineTo(x, b.y1); ctx.stroke(); ctx.setLineDash([]);
      const dot = (y, fill) => { ctx.beginPath(); ctx.arc(x, y, 3.4, 0, 2 * Math.PI); ctx.fillStyle = fill; ctx.fill(); ctx.strokeStyle = C.dark ? C.paper : "#ffffff"; ctx.lineWidth = 1.2; ctx.stroke(); };
      dot(YA(prep.x[i]), C.ink); dot(YA(mx.rebuilt[i]), C.accent); dot(YB(mx.residual[i]), C.ink);
    }
    ctx.restore();
    round(b.x0, b.y0, b.w, b.h, r);
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.4; ctx.stroke();

    // the screen's own lettering: the channels, the key, and the scale
    const narrow = b.w < 420;
    ctx.textBaseline = "middle"; ctx.textAlign = "left";
    let dry = false;
    const tag = (x, y, s) => {
      ctx.font = `10.5px ${C.type}`; const w = ctx.measureText(s).width + 8;
      if (!dry) { ctx.fillStyle = C.ink; ctx.fillRect(x, y - 8, w, 16); ctx.fillStyle = C.dark ? C.paper : "#fcfcf9"; ctx.fillText(s, x + 4, y + 0.5); }
      return x + w + 8;
    };
    const key = (x, y, s, color, width) => {
      ctx.font = `11.5px ${C.type}`;
      if (!dry) {
        if (color) { ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + 16, y); ctx.stroke(); }
        ctx.fillStyle = C.ink; ctx.fillText(s, x + (color ? 21 : 0), y + 0.5);
      }
      return x + (color ? 21 : 0) + ctx.measureText(s).width + 10;
    };
    const ya = b.y0 + 13, yb = b.y0 + b.h / 2 + 14;
    const rowA = () => key(key(tag(b.x0 + 8, ya, "CH1"), ya, "signal", alpha(C.ink, 0.75), 1), ya, "rebuilt", C.accent, 2);
    const rowB = () => key(tag(b.x0 + 8, yb, "CH2"), yb, narrow ? "residual" : `residual = signal ${M.MINUS} rebuilt`);
    // the lettering sits on a strip of the screen's paper, so a trace never runs through it
    for (const [row, y] of [[rowA, ya], [rowB, yb]]) {
      dry = true; const end = row(); dry = false;
      ctx.fillStyle = alpha(C.paper, C.dark ? 0.9 : 0.88); ctx.fillRect(b.x0 + 4, y - 10, end - b.x0 - 6, 20);
      row();
    }
    ctx.textAlign = "right"; ctx.fillStyle = C.ink2; ctx.font = `10.5px ${C.type}`;
    ctx.fillText(`0.25 s/div  1 SD/div`, b.x1 - 8, b.y1 - 10);
    // the time axis under the screen
    ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillStyle = C.ink2; ctx.font = `10.5px ${C.type}`;
    for (let s = 0; s <= SEC + 1e-9; s += 0.5) {
      const xx = b.x0 + (s / SEC) * b.w;
      ctx.textAlign = s === 0 ? "left" : s >= SEC ? "right" : "center";
      ctx.fillText(s === SEC ? `${s} s` : String(s), xx, b.y1 + 6);
    }
  }

  // ---------------------------------------------------------------------------------------- the strips
  function drawMini(k) {
    const s = strips[k], ctx = s.mini.getContext("2d"), w = MW, h = MH, y = P[k].y, n = prep.n, g = gains[k];
    if (!w) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const lim = Math.ceil(prep.ymax * 10) / 10, X = (i) => 1 + (i / (n - 1)) * (w - 2), Y = (v) => h / 2 - (v / lim) * (h / 2 - 2);
    ctx.strokeStyle = alpha(C.ink, 0.22); ctx.lineWidth = 0.7; ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
    // the piece at full level, faint, and as it is mixed now, in the chapter's ink
    ctx.lineJoin = "round";
    ctx.strokeStyle = alpha(C.ink2, C.dark ? 0.45 : 0.42); ctx.lineWidth = 0.8; ctx.beginPath();
    for (let i = 0; i < n; i++) (i ? ctx.lineTo(X(i), Y(y[i])) : ctx.moveTo(X(i), Y(y[i])));
    ctx.stroke();
    if (g > 0) {
      ctx.strokeStyle = C.accent; ctx.lineWidth = 1.15; ctx.beginPath();
      for (let i = 0; i < n; i++) (i ? ctx.lineTo(X(i), Y(g * y[i])) : ctx.moveTo(X(i), Y(g * y[i])));
      ctx.stroke();
    }
    if (st.cursor != null) {
      const x = X(st.cursor);
      ctx.strokeStyle = alpha(C.ink, 0.7); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
  }
  function placeCaps() {
    strips.forEach((s, k) => {
      const H = s.fader.clientHeight, ch = s.cap.offsetHeight || 22;
      s.cap.style.transform = `translateY(${((1 - st.faders[k]) * (H - ch)).toFixed(2)}px)`;
    });
  }

  // ---------------------------------------------------------------------------------------- words
  let lastWords = "", lastRead = "";
  function words() {
    const R = M.readouts(prep, mx);
    strips.forEach((s, k) => {
      const r = R.strips[k], fg = M.quantize(st.faders[k]);
      s.level.textContent = st.muted[k] ? "muted" : r.level;
      s.fader.setAttribute("aria-valuenow", fg.toFixed(2));
      s.fader.setAttribute("aria-valuetext", `${M.level(fg)}${st.muted[k] ? ", muted" : ""}`);
      s.mute.setAttribute("aria-pressed", String(st.muted[k]));
      s.el.classList.toggle("is-out", gains[k] === 0);
      s.fill.style.width = `${(100 * r.meter) / meterMax}%`;
    });
    $(".ei-in").textContent = R.inCount;
    $(".ei-lost").textContent = R.lost;
    $(".ei-gap").textContent = R.gap;
    const loopWords = st.auto && !st.returning && phase ? M.sayLoop(prep, phase) : null;
    const s = loopWords || M.say(prep, mx);
    if (s !== lastWords) { say.textContent = s; lastWords = s; }
    scope.setAttribute("aria-label", `An oscilloscope screen. Channel 1: the signal and the rebuilt trace, the sum of the pieces as mixed. Channel 2: the residual, signal minus rebuilt, at the same scale. ${s}`);
    const rd = st.cursor != null ? M.sayCursor(M.cursor(prep, mx, st.cursor), prep.n) : "Point at the screen, or drag along it, to read one sample.";
    if (rd !== lastRead) { read.textContent = rd; lastRead = rd; read.classList.toggle("is-on", st.cursor != null); }
  }
  function setPlayLabel() {
    play.textContent = st.auto ? "Pause" : "Play";
    play.setAttribute("aria-pressed", String(st.auto));
    play.setAttribute("aria-label", st.auto ? "Pause the slow loop" : "Play the slow loop: every piece out, then back in from the residue up");
  }

  // ---------------------------------------------------------------------------------------- state
  function refresh(force = false) {
    const g = M.effective(st.faders, st.muted);
    const changed = force || g.some((v, k) => v !== gains[k]);
    if (changed) { gains = g; mx = M.mix(prep, gains); for (let k = 0; k < N; k++) drawMini(k); }
    words();
    return changed;
  }
  function redrawAll() { size(); placeCaps(); refresh(true); drawScope(); }

  // ---------------------------------------------------------------------------------------- motion
  let loop = 0, last = 0;
  const settled = () => st.faders.every((x, k) => x === st.target[k] && st.vel[k] === 0);
  // on screen: the observer wakes the plate, and each frame checks the plate's box itself, so a missed
  // observer callback can neither stall it on screen nor keep it running off screen
  const onScreen = () => { const r = fig.getBoundingClientRect(); return r.bottom > 0 && r.top < innerHeight && r.width > 0; };
  const wake = () => { if (!st.visible && onScreen()) st.visible = true; if (!loop && st.visible && !document.hidden) loop = requestAnimationFrame(frame); };
  function frame(now) {
    loop = 0;
    st.visible = onScreen();
    if (!st.visible || document.hidden) { last = 0; return; }
    const dt = last ? Math.min((now - last) / 1000, 0.05) : 1 / 60;
    last = now;
    let moved = false;
    if (st.auto) {
      st.clock += dt; st.beamClock += dt;
      phase = M.loopAt(st.clock, N);
      st.target = phase.gains.slice();
      if (!st.returning) { st.faders = st.target.slice(); st.vel.fill(0); }
    }
    if (!st.auto || st.returning) {
      for (let k = 0; k < N; k++) {
        if (reduced.matches) { st.faders[k] = st.target[k]; st.vel[k] = 0; continue; }
        [st.faders[k], st.vel[k]] = M.follow(st.faders[k], st.vel[k], st.target[k], dt);
      }
      if (st.returning && settled()) st.returning = false;
    }
    placeCaps();
    moved = refresh();
    if (moved || (st.auto && C.dark)) drawScope();
    if (st.auto || !settled()) loop = requestAnimationFrame(frame); else last = 0;
  }
  function setAuto(on) {
    st.auto = on;
    if (on) {
      st.clock = 0; st.returning = true; st.muted.fill(false); phase = M.loopAt(0, N);
      st.target = phase.gains.slice();
    } else {
      phase = null; st.returning = false;
      st.target = st.faders.map(M.quantize);
    }
    setPlayLabel(); refresh(true); drawScope(); wake();
  }
  function touch() {
    if (!st.touched) { st.touched = true; say.setAttribute("aria-live", "polite"); }
    if (st.auto) setAuto(false);
  }

  // ---------------------------------------------------------------------------------------- input
  strips.forEach((s, k) => {
    const valueAt = (e) => {
      const r = s.fader.getBoundingClientRect(), ch = s.cap.offsetHeight || 22;
      return M.quantize(1 - (e.clientY - r.top - ch / 2) / Math.max(1, r.height - ch));
    };
    s.fader.addEventListener("pointerdown", (e) => {
      touch(); s.fader.setPointerCapture(e.pointerId); st.drag = k; s.fader.focus({ preventScroll: true });
      st.target[k] = valueAt(e); s.el.classList.add("is-drag"); wake(); e.preventDefault();
    });
    s.fader.addEventListener("pointermove", (e) => { if (s.fader.hasPointerCapture(e.pointerId)) { st.target[k] = valueAt(e); wake(); } });
    const up = (e) => { if (s.fader.hasPointerCapture?.(e.pointerId)) s.fader.releasePointerCapture(e.pointerId); st.drag = -1; s.el.classList.remove("is-drag"); };
    s.fader.addEventListener("pointerup", up); s.fader.addEventListener("pointercancel", up);
    s.fader.addEventListener("keydown", (e) => {
      const d = { ArrowUp: 0.05, ArrowRight: 0.05, ArrowDown: -0.05, ArrowLeft: -0.05, PageUp: 0.25, PageDown: -0.25 }[e.key];
      let v = null;
      if (d != null) v = M.quantize(M.quantize(st.target[k]) + d);
      else if (e.key === "Home") v = 0;
      else if (e.key === "End") v = 1;
      else if (e.key === "m" || e.key === "M") { e.preventDefault(); touch(); st.muted[k] = !st.muted[k]; refresh(); drawScope(); return; }
      if (v == null) return;
      e.preventDefault(); touch(); st.target[k] = v; wake();
    });
    s.mute.addEventListener("click", () => { touch(); st.muted[k] = !st.muted[k]; refresh(); drawScope(); });
  });
  play.addEventListener("click", () => { if (!st.touched) { st.touched = true; say.setAttribute("aria-live", "polite"); } setAuto(!st.auto); });
  $('[data-act="in"]').addEventListener("click", () => { touch(); st.muted.fill(false); st.target.fill(1); refresh(); drawScope(); wake(); });
  $('[data-act="out"]').addEventListener("click", () => { touch(); st.muted.fill(true); refresh(); drawScope(); });

  // the screen: point or drag to read one sample; arrow keys step through them
  const sampleFromEvent = (e) => {
    const r = scope.getBoundingClientRect(), b = screenBox();
    return M.sampleAt(((e.clientX - r.left - b.x0) / b.w) * SEC, prep.fs, prep.n);
  };
  const setCursor = (i) => { st.cursor = i; for (let k = 0; k < N; k++) drawMini(k); words(); drawScope(); };
  // The visible readout is never live: it follows the pointer and the loop's mix. A cursor step from the
  // keyboard is announced once, in its own polite region, with the reading at that moment.
  const announcer = $(".ei-announce");
  const announceCursor = () => { announcer.textContent = M.sayCursor(M.cursor(prep, mx, st.cursor), prep.n); };
  scope.addEventListener("pointermove", (e) => { if (e.pointerType === "mouse" || scope.hasPointerCapture(e.pointerId)) setCursor(sampleFromEvent(e)); });
  scope.addEventListener("pointerdown", (e) => { if (e.pointerType !== "mouse") scope.setPointerCapture(e.pointerId); setCursor(sampleFromEvent(e)); });
  scope.addEventListener("pointerleave", (e) => { if (e.pointerType === "mouse") setCursor(null); });
  scope.addEventListener("keydown", (e) => {
    const n = prep.n, c = st.cursor ?? -1;
    if (e.key === " ") { e.preventDefault(); if (!st.touched) { st.touched = true; say.setAttribute("aria-live", "polite"); } setAuto(!st.auto); return; }
    const d = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
    if (d) { e.preventDefault(); setCursor(Math.max(0, Math.min(n - 1, (c < 0 ? (d > 0 ? -1 : n) : c) + d * (e.shiftKey ? 16 : 1)))); announceCursor(); return; }
    if (e.key === "Home") { e.preventDefault(); setCursor(0); announceCursor(); }
    else if (e.key === "End") { e.preventDefault(); setCursor(n - 1); announceCursor(); }
    else if (e.key === "Escape") setCursor(null);
  });
  scope.addEventListener("blur", () => { if (st.cursor != null && !scope.matches(":hover")) setCursor(null); });

  // ---------------------------------------------------------------------------------------- lifecycle
  readColors();
  setPlayLabel();
  redrawAll();
  new ResizeObserver(() => { size(); placeCaps(); for (let k = 0; k < N; k++) drawMini(k); drawScope(); }).observe(root);
  new IntersectionObserver((es) => { st.visible = es[0].isIntersecting; if (st.visible) wake(); }, { threshold: 0.02 }).observe(fig);
  document.addEventListener("visibilitychange", wake);
  const recolor = () => { readColors(); for (let k = 0; k < N; k++) drawMini(k); drawScope(); };
  new MutationObserver(recolor).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  darkQuery.addEventListener("change", recolor);
  reduced.addEventListener("change", () => { if (reduced.matches && st.auto) setAuto(false); });
  link.addEventListener("load", () => { readColors(); redrawAll(); });
  if (document.fonts) document.fonts.ready.then(() => { readColors(); redrawAll(); });

  // the latest check run, quietly under the plate
  fetch(fig.dataset.src.replace(/\.json$/, ".checks.json")).then((r) => (r.ok ? r.json() : null)).then((c) => {
    if (!c || !c.line) return;
    const p = $(".ei-checked"); p.textContent = c.line; p.hidden = false;
  }).catch(() => {});
}
