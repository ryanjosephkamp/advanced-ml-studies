// The emd-switchboard plate (chapter IV): the EEGMAT subsets the study ran, as an old telephone
// switchboard. The five channels are jacks on a faceplate engraved with the 10-20 head (the other 14
// sites EEGMAT records are capped), each with a cord that rises from the shelf to plug it in; the five
// bands are lever keys. When the plugged channels and the keys that are on make one of the seven
// combinations in results/study/eegmat_subsets.csv, the lamp lights and the meter's needle swings to its
// committed balanced accuracy, with the 95% interval shaded on the arc. Any other combination leaves the
// lamp dark under a "not run" plate: the page computes no accuracy. When idle, the board steps through
// the seven on its own.
//
// Drawn in SVG with the page's inks as CSS variables, so it follows the printing without a redraw.
// Every number and sentence comes from emd-switchboard-model.js, which the check script tests.
const VQ = new URL(import.meta.url).search; // the page's version, passed on so a browser never mixes versions
const M = await import(new URL(`./emd-switchboard-model.js${VQ}`, import.meta.url).href);

const KIND = "emd-switchboard";
const NS = "http://www.w3.org/2000/svg";
const reduced = matchMedia("(prefers-reduced-motion: reduce)");

const link = document.createElement("link");
link.rel = "stylesheet";
link.href = new URL(`./${KIND}.css${VQ}`, import.meta.url).href;
document.head.append(link);

for (const fig of document.querySelectorAll(`figure.plate-live[data-demo="${KIND}"]`)) mount(fig).catch((e) => {
  const root = fig.querySelector(".demo-root");
  if (root) root.innerHTML = `<p class="sb-fail">The live plate could not load its data (${String(e.message || e).replace(/[<>&]/g, "")}).</p>`;
});

const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const f1 = (x) => (Math.round(x * 10) / 10).toString();
/** Text as HTML with intervals and ranges kept on one line ("[0.532, 0.576]", "1–4 Hz"). */
const keep = (s) => esc(s).replace(/\[[^\]]*\]|\d+–\d+(?: Hz)?/g, (m) => `<span class="sb-nw">${m}</span>`);

// ------------------------------------------------------------------------------------------ the faceplate
function faceplate(data) {
  const P = M.PANEL, { cx, cy, R } = P, live = new Set(data.channels);
  const s = [];
  // the plate, its screws, and its legend
  s.push(`<rect class="sb-plate" x="3" y="3" width="${P.w - 6}" height="${P.h - 6}" rx="9"/>`);
  for (const [x, y] of [[17, 17], [P.w - 17, 17], [17, P.shelfTop - 16], [P.w - 17, P.shelfTop - 16]]) s.push(screw(x, y, 4.6));
  s.push(`<text class="sb-legend" x="30" y="40">EEGMAT</text>`);
  s.push(`<text class="sb-legend-2" x="30" y="56">${M.legend(data)}</text>`);
  s.push(`<text class="sb-legend-2 sb-end" x="${P.w - 30}" y="40">nose up</text>`);
  // the head: the ear line, nose, and ears, with the 10-20 guide lines dashed
  s.push(`<circle class="sb-guide" cx="${cx}" cy="${cy}" r="${f1(0.8 * R)}"/>`);
  s.push(`<circle class="sb-guide sb-guide-2" cx="${cx}" cy="${cy}" r="${f1(0.4 * R)}"/>`);
  s.push(`<path class="sb-guide" d="M${cx} ${cy - R}V${cy + R}M${cx - R} ${cy}H${cx + R}"/>`);
  s.push(`<circle class="sb-head" cx="${cx}" cy="${cy}" r="${R}"/>`);
  s.push(`<path class="sb-head" d="M${cx - 13} ${f1(cy - R + 0.5)}L${cx} ${cy - R - 19}L${cx + 13} ${f1(cy - R + 0.5)}"/>`);
  for (const k of [-1, 1]) s.push(`<path class="sb-head" d="M${cx + k * R} ${cy - 22}c${k * 9} -3 ${k * 15} 4 ${k * 14} 14c0 9 -4 14 -6 20c-2 6 -5 9 -8 10"/>`);
  // the capped sites: blanking plugs, never wired
  for (const n of M.SITES) {
    if (live.has(n)) continue;
    const [x, y] = M.jackXY(n);
    s.push(`<g class="sb-cap"><circle cx="${f1(x)}" cy="${f1(y)}" r="${P.capR}"/><path d="M${f1(x - 3.2)} ${f1(y)}h6.4"/></g>`);
    const [dx, dy, anchor] = M.capLabel(n);
    s.push(`<text class="sb-cap-label" x="${f1(x + dx)}" y="${f1(y + dy)}" text-anchor="${anchor}">${n}</text>`);
  }
  // the five jacks
  for (const n of data.channels) {
    const [x, y] = M.jackXY(n);
    s.push(`<g class="sb-jack" data-ch="${n}"><circle class="sb-hit" cx="${f1(x)}" cy="${f1(y)}" r="27"/><circle class="sb-ring" cx="${f1(x)}" cy="${f1(y)}" r="${P.jackR}"/><circle class="sb-ring-2" cx="${f1(x)}" cy="${f1(y)}" r="${P.jackR - 4.5}"/><circle class="sb-hole" cx="${f1(x)}" cy="${f1(y)}" r="${P.holeR}"/></g>`);
  }
  // the cord shelf
  s.push(`<rect class="sb-shelf" x="10" y="${P.shelfTop}" width="${P.w - 20}" height="${P.h - P.shelfTop - 10}" rx="4"/>`);
  s.push(`<path class="sb-shelf-lip" d="M14 ${P.shelfTop + 8}H${P.w - 14}"/>`);
  for (const n of M.SHELF) { const [x, y] = M.holeXY(n); s.push(`<ellipse class="sb-shelf-hole" cx="${x}" cy="${y + 3}" rx="10" ry="4.5"/>`); }
  // the cords and plugs (moved every frame), then the jack labels on top with a halo
  s.push(`<g class="sb-cords"></g><g class="sb-plugs"></g>`);
  for (const n of data.channels) {
    const [x, y] = M.jackXY(n), [dx, dy, anchor] = M.JACK_LABEL[n] || [0, -22, "middle"];
    s.push(`<text class="sb-jack-label" data-ch="${n}" x="${f1(x + dx)}" y="${f1(y + dy)}" text-anchor="${anchor}">${n}</text>`);
  }
  return s.join("");
}
function screw(x, y, r) {
  return `<g class="sb-screw"><circle cx="${x}" cy="${y}" r="${r}"/><path d="M${f1(x - r * 0.7)} ${f1(y + r * 0.35)}L${f1(x + r * 0.7)} ${f1(y - r * 0.35)}"/></g>`;
}

// ------------------------------------------------------------------------------------------ the meter
function meterFace() {
  const G = M.METER, r = G.r, s = [];
  s.push(`<rect class="sb-plate" x="3" y="3" width="${G.w - 6}" height="${G.h - 6}" rx="9"/>`);
  for (const [x, y] of [[17, G.h - 17], [G.w - 17, G.h - 17]]) s.push(screw(x, y, 4.6));
  // the lamp and its legend
  s.push(`<g class="sb-lamp"><g class="sb-rays">${Array.from({ length: 8 }, (_, k) => { const a = (k * Math.PI) / 4; return `<path d="M${f1(34 + 17 * Math.cos(a))} ${f1(34 + 17 * Math.sin(a))}L${f1(34 + 23 * Math.cos(a))} ${f1(34 + 23 * Math.sin(a))}"/>`; }).join("")}</g><circle class="sb-lamp-bezel" cx="34" cy="34" r="13.5"/><circle class="sb-lamp-glass" cx="34" cy="34" r="9.5"/></g>`);
  s.push(`<text class="sb-legend sb-lamp-word" x="62" y="39">RUN</text>`);
  s.push(`<text class="sb-legend sb-end sb-norm-word" x="${G.w - 22}" y="39"></text>`);
  // the arc, the ticks, the labels, and chance
  const arc = (rad, a0, a1) => { const p = M.polar(rad, a0), q = M.polar(rad, a1); return `M${f1(p[0])} ${f1(p[1])}A${rad} ${rad} 0 0 1 ${f1(q[0])} ${f1(q[1])}`; };
  s.push(`<g class="sb-band"></g>`);
  s.push(`<path class="sb-arc" d="${arc(r.tickOut, G.from, G.to)}"/>`);
  for (const t of M.ticks()) {
    const a = M.angleOf(t.v), p = M.polar(r.tickOut, a), q = M.polar(t.major ? r.tickMajor : r.tickMinor, a);
    s.push(`<path class="${t.major ? "sb-tick-major" : "sb-tick"}" d="M${f1(p[0])} ${f1(p[1])}L${f1(q[0])} ${f1(q[1])}"/>`);
    if (t.label) { const l = M.polar(r.label, a); s.push(`<text class="sb-tick-label" x="${f1(l[0])}" y="${f1(l[1] + 4)}">${t.label}</text>`); }
  }
  const ca = M.angleOf(M.CHANCE), c0 = M.polar(r.bandIn - 4, ca), c1 = M.polar(r.tickOut, ca), cl = M.polar(r.bandIn - 14, ca);
  s.push(`<path class="sb-chance" d="M${f1(c0[0])} ${f1(c0[1])}L${f1(c1[0])} ${f1(c1[1])}"/>`);
  s.push(`<text class="sb-chance-label" x="${f1(cl[0] + 3)}" y="${f1(cl[1] + 4)}">chance</text>`);
  // the rest stop, the reference mark, the dial's words
  const st = M.polar(r.needle - 22, G.rest - 2.5);
  s.push(`<circle class="sb-stop" cx="${f1(st[0])}" cy="${f1(st[1])}" r="3.4"/>`);
  s.push(`<g class="sb-ref" style="display:none"><path d=""/><text class="sb-ref-label"></text></g>`);
  s.push(`<text class="sb-dial" x="${G.cx}" y="${G.cy - 74}">BALANCED ACCURACY</text>`);
  s.push(`<text class="sb-dial-2" x="${G.cx}" y="${G.cy - 58}">held-out subjects</text>`);
  // the needle, then the "not run" plate mounted over it, then the pivot
  s.push(`<g class="sb-needle"><path class="sb-needle-halo" d="M${G.cx} ${G.cy + 16}V${G.cy - r.needle}"/><path class="sb-needle-line" d="M${G.cx} ${G.cy + 16}V${G.cy - r.needle}"/><path class="sb-needle-tail" d="M${G.cx - 4} ${G.cy + 12}h8v8h-8z"/></g>`);
  s.push(`<g class="sb-notrun"><rect x="${G.cx - 74}" y="${G.cy - 49}" width="148" height="34" rx="3"/><rect class="sb-notrun-in" x="${G.cx - 69}" y="${G.cy - 44.5}" width="138" height="25" rx="2"/><circle cx="${G.cx - 64}" cy="${G.cy - 32}" r="1.8"/><circle cx="${G.cx + 64}" cy="${G.cy - 32}" r="1.8"/><text x="${G.cx}" y="${G.cy - 27}">NOT RUN</text></g>`);
  s.push(`<circle class="sb-pivot" cx="${G.cx}" cy="${G.cy}" r="9"/><circle class="sb-pivot-2" cx="${G.cx}" cy="${G.cy}" r="3"/>`);
  return s.join("");
}
function sector(rIn, rOut, a0, a1) {
  const p = M.polar(rOut, a0), q = M.polar(rOut, a1), u = M.polar(rIn, a1), v = M.polar(rIn, a0), big = a1 - a0 > 180 ? 1 : 0;
  return `M${f1(p[0])} ${f1(p[1])}A${rOut} ${rOut} 0 ${big} 1 ${f1(q[0])} ${f1(q[1])}L${f1(u[0])} ${f1(u[1])}A${rIn} ${rIn} 0 ${big} 0 ${f1(v[0])} ${f1(v[1])}Z`;
}

/** A small picture of a combination: the five jacks on the front of the head, then the five band keys. */
function glyph(data, sub) {
  const k = 18, cx = 15, cy = 26, a = 56 * Math.PI / 180;
  const dots = data.channels.map((n) => { const [x, y] = M.flat(n); const on = sub.channels.includes(n); return `<circle class="${on ? "on" : "off"}" cx="${f1(cx + k * x)}" cy="${f1(cy - k * y)}" r="2.2"/>`; }).join("");
  const keys = data.bands.map((b, i) => { const on = sub.bands.includes(b.id); return `<rect class="${on ? "on" : "off"}" x="${37 + i * 7}" y="${on ? 5 : 15}" width="4.4" height="${on ? 18 : 8}" rx="1"/>`; }).join("");
  const arc = `M${f1(cx - k * Math.sin(a))} ${f1(cy - k * Math.cos(a))}A${k} ${k} 0 0 1 ${f1(cx + k * Math.sin(a))} ${f1(cy - k * Math.cos(a))}M${cx - 3} ${cy - k + 0.4}L${cx} ${cy - k - 3.4}L${cx + 3} ${cy - k + 0.4}`;
  return `<svg class="sb-glyph" viewBox="0 0 72 27" aria-hidden="true"><path class="head" d="${arc}"/>${dots}<path class="base" d="M35 23.5H72"/>${keys}</svg>`;
}

// ------------------------------------------------------------------------------------------ the plate
async function mount(fig) {
  const root = fig.querySelector(".demo-root");
  const data = await (await fetch(fig.dataset.src)).json();
  const subs = data.subsets, uid = Math.random().toString(36).slice(2, 8);
  const full = subs.find((s) => s.full);
  const st = { ...M.stateOf(full, "subject"), auto: !reduced.matches, clock: 0, tourStart: subs.indexOf(full), tourAt: subs.indexOf(full), touched: false, visible: false };
  const plug = Object.fromEntries(data.channels.map((c) => [c, { p: 1, from: 1, to: 1, t: 1 }]));
  const needle = { a: M.needleAngle(data, st), v: 0 };

  root.innerHTML = `
<div class="sb">
  <div class="sb-bar">
    <p class="sb-tour"></p>
    <button type="button" class="sb-play" aria-pressed="false"></button>
  </div>
  <div class="sb-body">
    <div class="sb-board">
      <div class="sb-panel">
        <svg class="sb-face" viewBox="0 0 ${M.PANEL.w} ${M.PANEL.h}" role="img">${faceplate(data)}</svg>
        <div class="sb-cordrow" role="group" aria-label="Channels: plug a cord in or out" style="top:${((100 * M.PANEL.shelfTop) / M.PANEL.h).toFixed(3)}%;bottom:${((100 * 15) / M.PANEL.h).toFixed(3)}%">
          ${M.SHELF.map((n) => `<button type="button" class="sb-cordkey" role="switch" data-ch="${n}" aria-checked="true" aria-label="Channel ${n}"><span>${n}</span></button>`).join("")}
        </div>
      </div>
      <div class="sb-keys" role="group" aria-label="Bands: turn a key on or off">
        ${data.bands.map((b) => `<button type="button" class="sb-key" role="switch" data-b="${b.id}" aria-checked="true" aria-label="${M.BAND_WORD[b.id]} band, ${M.bandRange(b).replace("–", " to ")}"><span class="sb-lever" aria-hidden="true"><i></i></span><b>${M.BAND_WORD[b.id]}</b><small>${M.bandRange(b)}</small></button>`).join("")}
      </div>
      <p class="sb-hint">Tap a jack or its cord to plug a channel in or out, and a key to turn a band on or off.</p>
    </div>
    <div class="sb-meterbox">
      <svg class="sb-meter" viewBox="0 0 ${M.METER.w} ${M.METER.h}" role="img">${meterFace()}</svg>
      <div class="sb-norm" role="radiogroup" aria-label="Which values the meter reads">
        ${data.norms.map((n) => `<button type="button" role="radio" data-n="${n}" aria-checked="${n === st.norm}" tabindex="${n === st.norm ? 0 : -1}">${n === "subject" ? "Standardized <span>within each subject</span>" : "Unstandardized"}</button>`).join("")}
      </div>
    </div>
    <div class="sb-read">
      <p class="sb-title"><b></b><span class="sb-prov"></span></p>
      <p class="sb-combo"></p>
      <p class="sb-big"><span class="sb-val"></span> <span class="sb-ci"></span></p>
      <p class="sb-what"></p>
      <p class="sb-other"></p>
      <p class="sb-change"></p>
      <p class="sb-say" id="sb-say-${uid}"></p>
    </div>
  </div>
  <section class="sb-dir" aria-labelledby="sb-dir-${uid}">
    <h4 id="sb-dir-${uid}">The ${M.word(subs.length)} combinations the study ran</h4>
    <div class="sb-dir-head" aria-hidden="true"><span></span><span>Standardized</span><span>Unstandardized</span></div>
    <ol>${subs.map((s, i) => `<li><button type="button" data-i="${i}"><i class="sb-dot" aria-hidden="true"></i><span class="sb-dir-name">${esc(M.titleOf(s))}${s.paper_label ? "" : `<em> not in the paper</em>`}</span>${glyph(data, s)}<span class="sb-dir-v" data-n="subject"><span class="sb-sr">Standardized: </span>${M.fmt(M.shown(data, s, "subject").mean)}</span><span class="sb-dir-v" data-n="none"><span class="sb-sr">, unstandardized: </span>${M.fmt(M.shown(data, s, "none").mean)}</span></button></li>`).join("")}</ol>
  </section>
  <ul class="sb-notes">${M.notes(data).map((n) => `<li><b>${esc(n.head)}.</b> ${esc(n.text)}</li>`).join("")}</ul>
  <p class="sb-checked" hidden></p>
</div>`;

  const $ = (s) => root.querySelector(s), $$ = (s) => [...root.querySelectorAll(s)];
  const face = $(".sb-face"), meter = $(".sb-meter"), cordsG = $(".sb-cords"), plugsG = $(".sb-plugs");
  const needleG = $(".sb-needle"), bandG = $(".sb-band"), ref = $(".sb-ref"), play = $(".sb-play"), tour = $(".sb-tour");
  const say = $(".sb-say");

  // one cord path (with its halo) and one plug per channel
  const cordEls = {}, plugEls = {};
  for (const n of data.channels) {
    const g = document.createElementNS(NS, "g");
    g.setAttribute("class", "sb-cord");
    g.innerHTML = `<path class="sb-cord-halo"/><path class="sb-cord-line"/>`;
    cordsG.append(g); cordEls[n] = g;
    const p = document.createElementNS(NS, "g");
    p.setAttribute("class", "sb-plug"); p.dataset.ch = n;
    p.innerHTML = `<circle class="sb-hit" r="22"/><circle class="sb-plug-body" r="${M.PANEL.plugR}"/><circle class="sb-plug-ring" r="${M.PANEL.plugR - 4.6}"/>`;
    plugsG.append(p); plugEls[n] = p;
  }

  // ---------------------------------------------------------------------------------------- drawing
  function drawPlugs() {
    for (const n of data.channels) {
      const k = plug[n], P = M.plugAt(n, k.p), on = k.to === 1;
      plugEls[n].setAttribute("transform", `translate(${f1(P[0])} ${f1(P[1])})`);
      plugEls[n].classList.toggle("is-on", on);
      const B = M.cord(n, P), d = k.p > 0.002 ? `M${B.map((q, i) => `${i === 1 ? "C" : ""}${f1(q[0])} ${f1(q[1])}`).join(" ")}` : "";
      for (const el of cordEls[n].children) el.setAttribute("d", d);
      cordEls[n].classList.toggle("is-on", on);
    }
  }
  function drawNeedle() { needleG.setAttribute("transform", `rotate(${needle.a.toFixed(2)} ${M.METER.cx} ${M.METER.cy})`); }

  function render() {
    const r = M.readout(data, st), sub = M.match(data, st);
    // the switches
    for (const b of $$(".sb-cordkey")) b.setAttribute("aria-checked", String(st.chans.includes(b.dataset.ch)));
    for (const b of $$(".sb-key")) b.setAttribute("aria-checked", String(st.bands.includes(b.dataset.b)));
    for (const b of $$(".sb-norm button")) { const on = b.dataset.n === st.norm; b.setAttribute("aria-checked", String(on)); b.tabIndex = on ? 0 : -1; }
    for (const g of $$(".sb-jack")) g.classList.toggle("is-on", st.chans.includes(g.dataset.ch));
    face.setAttribute("aria-label", `The jack panel, a head seen from above with the nose up. ${st.chans.length ? `Plugged in: ${M.list(data.channels.filter((c) => st.chans.includes(c)))}.` : "No channel is plugged in."} Bands on: ${st.bands.length ? M.list(data.bands.filter((b) => st.bands.includes(b.id)).map((b) => M.BAND_WORD[b.id])) : "none"}.`);
    // the meter
    meter.classList.toggle("is-lit", r.lit);
    meter.classList.toggle("is-dark", !r.lit);
    $(".sb-norm-word").textContent = M.NORM_SHORT[st.norm].toUpperCase();
    if (sub) {
      const v = M.shown(data, sub, st.norm);
      bandG.innerHTML = `<path class="sb-ci-band" d="${sector(M.METER.r.bandIn, M.METER.r.bandOut, M.angleOf(v.lo), M.angleOf(v.hi))}"/>`;
      if (!sub.full) {
        const fv = M.shown(data, full, st.norm), a = M.angleOf(fv.mean), tip = M.polar(M.METER.r.bandIn - 1, a);
        const l = M.polar(M.METER.r.bandIn - 9, a - 4.5), rr = M.polar(M.METER.r.bandIn - 9, a + 4.5);
        ref.querySelector("path").setAttribute("d", `M${f1(tip[0])} ${f1(tip[1])}L${f1(l[0])} ${f1(l[1])}L${f1(rr[0])} ${f1(rr[1])}Z`);
        const lp = M.polar(M.METER.r.bandIn - 22, a), t = ref.querySelector("text");
        t.setAttribute("x", f1(lp[0])); t.setAttribute("y", f1(lp[1] + 4)); t.textContent = "all";
        ref.style.display = "";
      } else ref.style.display = "none";
      meter.setAttribute("aria-label", `Meter: ${r.value}, 95% interval ${r.ci.slice(1, -1).replace(", ", " to ")}, ${M.NORM_LABEL[st.norm]}. The lamp is lit.`);
    } else {
      bandG.innerHTML = ""; ref.style.display = "none";
      meter.setAttribute("aria-label", "Meter at rest on its stop, the lamp dark: not run.");
    }
    // the readout
    const read = $(".sb-read");
    read.dataset.status = r.status;
    $(".sb-title b").textContent = r.title;
    $(".sb-prov").textContent = r.provenance;
    $(".sb-combo").textContent = r.combo;
    $(".sb-val").textContent = r.value;
    $(".sb-ci").textContent = r.ci;
    $(".sb-what").textContent = r.what;
    $(".sb-other").innerHTML = keep(r.other);
    $(".sb-change").innerHTML = keep(r.change);
    if (say.textContent !== r.say) say.innerHTML = keep(r.say);
    // the directory
    const at = sub ? subs.indexOf(sub) : -1;
    $$(".sb-dir li button").forEach((b, i) => { if (i === at) b.setAttribute("aria-current", "true"); else b.removeAttribute("aria-current"); });
    root.querySelector(".sb-dir").dataset.norm = st.norm;
    // the tour's line
    tour.textContent = M.tourLine(st.auto, at, subs.length);
    // where things go
    for (const n of data.channels) setPlug(n, st.chans.includes(n) ? 1 : 0);
    needle.target = M.needleAngle(data, st);
    if (reduced.matches) { needle.a = needle.target; needle.v = 0; }
    wake();
  }
  function setPlug(n, to) {
    const k = plug[n];
    if (k.to === to) return;
    k.from = k.p; k.to = to; k.t = 0;
    if (reduced.matches) { k.p = to; k.t = 1; }
  }

  // ---------------------------------------------------------------------------------------- motion
  let loop = 0, last = 0;
  const wake = () => { if (!loop) loop = requestAnimationFrame(frame); };
  function frame(now) {
    loop = 0;
    const dt = last ? Math.min((now - last) / 1000, 0.05) : 1 / 60;
    last = now;
    if (!st.visible || document.hidden) { last = 0; drawPlugs(); drawNeedle(); return; }
    if (st.auto) {
      st.clock += dt;
      const i = M.tourIndex(st.clock, subs.length, st.tourStart);
      if (i !== st.tourAt) { st.tourAt = i; Object.assign(st, M.stateOf(subs[i], st.norm)); render(); }
    }
    let moving = false;
    for (const n of data.channels) {
      const k = plug[n];
      if (k.t < 1) { k.t = Math.min(1, k.t + dt / 0.55); k.p = k.from + (k.to - k.from) * k.t; moving = true; }
    }
    // the needle waits for the plugs to seat, then swings on its spring
    const seated = data.channels.every((n) => plug[n].t >= 0.7);
    if (seated) {
      for (let s = 0; s < 4; s++) [needle.a, needle.v] = M.springStep(needle.a, needle.v, needle.target, dt / 4);
      if (Math.abs(needle.a - needle.target) > 0.02 || Math.abs(needle.v) > 0.05) moving = true; else { needle.a = needle.target; needle.v = 0; }
    } else moving = true;
    drawPlugs(); drawNeedle();
    // one loop only: a tour step's render() may already have asked for the next frame
    if (!(st.auto || moving)) last = 0;
    else if (!loop) loop = requestAnimationFrame(frame);
  }
  function setAuto(on) {
    st.auto = on;
    play.textContent = on ? "Pause" : "Play the tour";
    play.setAttribute("aria-pressed", String(on));
    play.setAttribute("aria-label", on ? "Pause the tour of the seven runs" : "Play the tour of the seven runs");
    if (on) {
      const sub = M.match(data, st);
      st.clock = 0;
      if (sub) st.tourStart = st.tourAt = subs.indexOf(sub);
      else { st.tourStart = st.tourAt = 0; Object.assign(st, M.stateOf(subs[0], st.norm)); }
    }
    render();
  }
  function touch() {
    if (!st.touched) { st.touched = true; say.setAttribute("aria-live", "polite"); }
    if (st.auto) setAuto(false);
  }

  // ---------------------------------------------------------------------------------------- input
  const toggleChannel = (n) => { touch(); st.chans = st.chans.includes(n) ? st.chans.filter((c) => c !== n) : [...st.chans, n]; render(); };
  for (const b of $$(".sb-cordkey")) b.addEventListener("click", () => toggleChannel(b.dataset.ch));
  for (const g of [...$$(".sb-jack"), ...Object.values(plugEls)]) g.addEventListener("click", () => toggleChannel(g.dataset.ch));
  for (const b of $$(".sb-key")) b.addEventListener("click", () => { touch(); const id = b.dataset.b; st.bands = st.bands.includes(id) ? st.bands.filter((x) => x !== id) : [...st.bands, id]; render(); });
  const norms = $$(".sb-norm button");
  norms.forEach((b, k) => {
    b.addEventListener("click", () => { touch(); st.norm = b.dataset.n; render(); });
    b.addEventListener("keydown", (e) => {
      const d = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0;
      if (!d) return;
      e.preventDefault();
      const nb = norms[(k + d + norms.length) % norms.length]; nb.focus(); nb.click();
    });
  });
  for (const b of $$(".sb-dir li button")) b.addEventListener("click", () => { touch(); Object.assign(st, M.stateOf(subs[+b.dataset.i], st.norm)); render(); });
  play.addEventListener("click", () => { if (!st.touched) { st.touched = true; say.setAttribute("aria-live", "polite"); } setAuto(!st.auto); });

  // ---------------------------------------------------------------------------------------- lifecycle
  drawPlugs(); drawNeedle();
  setAuto(st.auto);
  new IntersectionObserver((es) => { st.visible = es[0].isIntersecting; if (st.visible) wake(); }, { threshold: 0.02 }).observe(fig);
  document.addEventListener("visibilitychange", wake);
  reduced.addEventListener("change", () => { if (reduced.matches) { setAuto(false); for (const n of data.channels) { plug[n].p = plug[n].to; plug[n].t = 1; } needle.a = needle.target; needle.v = 0; drawPlugs(); drawNeedle(); } });

  // the latest check run, quietly under the plate
  fetch(fig.dataset.src.replace(/\.json$/, ".checks.json")).then((r) => (r.ok ? r.json() : null)).then((c) => {
    if (!c || !c.line) return;
    const p = $(".sb-checked"); p.textContent = c.line; p.hidden = false;
  }).catch(() => {});
}
