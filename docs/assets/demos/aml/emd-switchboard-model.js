// The emd-switchboard plate's model: pure functions, no DOM, importable by Node. The view
// (emd-switchboard.js) draws what these return, and tools/aml/emd-switchboard.check.mjs holds every one
// of them to the study's committed files (results/study/eegmat_subsets.csv, results/study/summary.csv,
// results/study/diagnostics.json) and to the paper's printed numbers (paper/numbers.tex and its table).
//
// What is shown: the switches pick a set of channels and a set of bands. When that pair is one of the
// subsets the study ran, the plate shows the committed mean per-subject balanced accuracy on held-out
// subjects with its 95% interval, as it is in the file. Any other pair shows no number: the page computes
// no accuracy.

export const MINUS = "−";
export const CHANCE = 0.5;
export const NORM_LABEL = { subject: "standardized within each subject", none: "unstandardized" };
export const NORM_SHORT = { subject: "Standardized", none: "Unstandardized" };
/** The study's band names (study/features.py BAND_NAMES) as the plate writes them. */
export const BAND_WORD = { delta: "delta", theta: "theta", alpha_mu: "alpha", beta: "beta", gamma: "gamma" };
const WORD = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];
export const word = (n) => WORD[n] ?? String(n);
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);

// ---------------------------------------------------------------------------------------- numbers
/** Rounded as the paper's make_numbers.py does: to 9 places, then half up (away from zero). A value that
 *  rounds to zero is written without a sign; negative numbers take a true minus sign. */
export function fmt(x, places = 3) {
  const s = Math.abs(x).toFixed(9), [ip, fp] = s.split(".");
  let q = BigInt(ip + fp.slice(0, places));
  if (fp[places] >= "5") q += 1n;
  let t = q.toString().padStart(places + 1, "0");
  t = `${t.slice(0, t.length - places)}.${t.slice(t.length - places)}`;
  return q === 0n || x >= 0 ? t : MINUS + t;
}
export const fmtAbs = (x, places = 3) => fmt(Math.abs(x), places);
export const fmtCI = (lo, hi) => `[${fmt(lo)}, ${fmt(hi)}]`;
/** A whole count with thousands separators, as the paper prints them. */
export const fmtCount = (n) => n.toLocaleString("en-US");
const hz = (x) => String(Number(x));
export const bandRange = (b) => `${hz(b.lo)}–${hz(b.hi)} Hz`;
/** A list with the Oxford comma. */
export function list(items) {
  if (items.length <= 1) return items.join("");
  if (items.length === 2) return `${items[0]} and ${items[1]}`;
  return `${items.slice(0, -1).join(", ")}, and ${items[items.length - 1]}`;
}

// ---------------------------------------------------------------------------------------- the head
// An idealized spherical head. Positions are (polar angle from the vertex Cz, azimuth from the nasion
// toward the right ear), in degrees, by the 10-20 rule: the nasion-inion and ear-to-ear arcs are each
// 180 degrees; the circumference through Fpz, T3, Oz, and T4 lies 10% of those arcs above the nasion,
// the ears, and the inion (72 degrees from Cz), divided in 10% steps of its own length (36 degrees of
// azimuth) with Fp1 and Fp2 half a step (18 degrees) either side of the midline; Fz, Pz, C3, and C4 sit
// 20% of the arcs (36 degrees) from Cz. F3, F4, P3, and P4 sit halfway along the great-circle arcs from
// Fz to F7, Fz to F8, Pz to T5, and Pz to T6. Temporal names are the older ones EEGMAT uses (T3 to T6).
export const POLAR = {
  Cz: [0, 0], Fz: [36, 0], Pz: [36, 180], C3: [36, -90], C4: [36, 90],
  Fp1: [72, -18], Fp2: [72, 18], F7: [72, -54], F8: [72, 54], T3: [72, -90], T4: [72, 90],
  T5: [72, -126], T6: [72, 126], O1: [72, -162], O2: [72, 162],
};
export const MIDWAY = { F3: ["Fz", "F7"], F4: ["Fz", "F8"], P3: ["Pz", "T5"], P4: ["Pz", "T6"] };
/** The 19 sites of the 10-20 system. */
export const SITES = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"];
const RAD = Math.PI / 180;
const unit = (a) => { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; };
/** A site as a unit vector: x toward the right ear, y toward the nasion, z up through Cz. */
export function onSphere(name) {
  if (MIDWAY[name]) { const [a, b] = MIDWAY[name].map(onSphere); return unit([a[0] + b[0], a[1] + b[1], a[2] + b[2]]); }
  const [th, ph] = POLAR[name];
  return [Math.sin(th * RAD) * Math.sin(ph * RAD), Math.sin(th * RAD) * Math.cos(ph * RAD), Math.cos(th * RAD)];
}
/** A site on the flat panel: the azimuthal equidistant map about Cz, with the nasion-ear-inion line at
 *  radius 1 (x to the subject's right, y toward the nose). */
export function flat(name) {
  const [x, y, z] = onSphere(name);
  const r = Math.acos(Math.max(-1, Math.min(1, z))) / (Math.PI / 2), a = Math.atan2(x, y);
  return [r * Math.sin(a), r * Math.cos(a)];
}

// ---------------------------------------------------------------------------------------- the panel
/** The jack panel in SVG units: the head's ear line has radius R about (cx, cy); the cord shelf runs
 *  along the bottom, one hole per channel, in the order the cords hang. */
export const PANEL = { w: 400, h: 492, cx: 200, cy: 214, R: 162, jackR: 15, holeR: 7, plugR: 11.5, capR: 6, shelfTop: 414, holeY: 444 };
export const SHELF = ["Fp1", "F3", "Fz", "F4", "Fp2"];
export const jackXY = (name) => { const [x, y] = flat(name); return [PANEL.cx + PANEL.R * x, PANEL.cy - PANEL.R * y]; };
export const holeXY = (name) => [((SHELF.indexOf(name) + 0.5) * PANEL.w) / SHELF.length, PANEL.holeY];
/** Where each jack's label sits from the jack's center (SVG units), and its anchor: clear of the cords. */
export const JACK_LABEL = { Fp1: [-4, -21, "end"], Fp2: [4, -21, "start"], Fz: [0, -22, "middle"], F3: [-10, 32, "end"], F4: [10, 32, "start"] };
/** The two outer cords swing out past F3 and F4 on their way down. */
const SWING = { Fp1: -1, Fp2: 1 };
export const ease = (x) => (x <= 0 ? 0 : x >= 1 ? 1 : x < 0.5 ? 2 * x * x : 1 - 2 * (1 - x) * (1 - x));
/** Where a plug is at progress p (0 resting in its shelf hole, 1 in its jack): it rises from the shelf
 *  and comes in to the jack along one quadratic arc. */
export function plugAt(name, p) {
  const [hx, hy] = holeXY(name), [jx, jy] = jackXY(name), t = ease(p);
  const c = [hx + (jx - hx) * 0.15, jy + (hy - jy) * 0.12];
  const u = 1 - t;
  return [u * u * hx + 2 * u * t * c[0] + t * t * jx, u * u * hy + 2 * u * t * c[1] + t * t * jy];
}
/** The cord from a plug at P down to its shelf hole: a cubic that falls from the plug and settles into
 *  the hole, as four control points. */
export function cord(name, P) {
  const H = holeXY(name), drop = Math.max(0, H[1] - P[1]), sw = SWING[name] || 0;
  const b1 = sw ? [P[0] + sw * 0.18 * drop, P[1] + 0.1 * drop] : [P[0], P[1] + 0.5 * drop];
  return [P, b1, [H[0], H[1] - 0.2 * drop], H];
}
/** Where each capped site's small label sits from the site (SVG units), and its anchor: below by
 *  default, beside where a cord hangs past. */
const CAP_LABEL_AT = { Cz: [9, 4, "start"], Pz: [9, 4, "start"], C3: [-9, 4, "end"], C4: [9, 4, "start"] };
export const capLabel = (name) => CAP_LABEL_AT[name] || [0, PANEL.capR + 11, "middle"];
export function bezier(B, t) {
  const u = 1 - t, a = u * u * u, b = 3 * u * u * t, c = 3 * u * t * t, d = t * t * t;
  return [a * B[0][0] + b * B[1][0] + c * B[2][0] + d * B[3][0], a * B[0][1] + b * B[1][1] + c * B[2][1] + d * B[3][1]];
}

// ---------------------------------------------------------------------------------------- the meter
/** The meter in SVG units: a 100-degree arc from 0.45 to 0.75 balanced accuracy, read linearly; the
 *  needle rests on a stop below the scale when nothing is lit. */
export const METER = { w: 360, h: 256, cx: 180, cy: 234, lo: 0.45, hi: 0.75, from: -50, to: 50, rest: -59,
  r: { label: 194, tickOut: 178, tickMajor: 166, tickMinor: 171, bandOut: 160, bandIn: 150, ref: 146, needle: 176 } };
export const angleOf = (v) => METER.from + ((v - METER.lo) / (METER.hi - METER.lo)) * (METER.to - METER.from);
export const polar = (r, deg) => [METER.cx + r * Math.sin(deg * RAD), METER.cy - r * Math.cos(deg * RAD)];
/** The scale's ticks: every 0.01, labeled every 0.05. */
export function ticks() {
  const out = [];
  for (let k = 0; k <= Math.round((METER.hi - METER.lo) / 0.01); k++) {
    const v = Math.round((METER.lo + k * 0.01) * 100) / 100;
    out.push({ v, major: k % 5 === 0, label: k % 5 === 0 ? v.toFixed(2) : null });
  }
  return out;
}

// ---------------------------------------------------------------------------------------- the data
const cache = new WeakMap();
const keyOf = (chans, bands) => `${[...chans].sort().join(" ")}|${[...bands].sort().join(" ")}`;
function lookup(data) {
  if (!cache.has(data)) cache.set(data, new Map(data.subsets.map((s) => [keyOf(s.channels, s.bands), s])));
  return cache.get(data);
}
/** The committed subset whose channels and bands are exactly the ones switched on, or null. */
export function match(data, st) {
  if (!st.chans.length || !st.bands.length) return null;
  return lookup(data).get(keyOf(st.chans, st.bands)) || null;
}
/** A state with the switches of a committed subset. */
export const stateOf = (sub, norm) => ({ chans: [...sub.channels], bands: [...sub.bands], norm });
/** The values shown for a subset and a normalization. The full feature set shows the main summary's row,
 *  as the paper prints it (same mean; its interval comes from that run's own bootstrap draws). */
export function shown(data, sub, norm) {
  const r = sub[norm];
  if (sub.full) { const f = data.full_summary[norm]; return { mean: f.mean, lo: f.lo, hi: f.hi, d: 0, dlo: 0, dhi: 0, from: "summary" }; }
  return { mean: r.mean, lo: r.lo, hi: r.hi, d: r.d, dlo: r.dlo, dhi: r.dhi, from: "subsets" };
}
/** The file's subset name as a title: a capital first letter and en dashes in ranges. */
export const phraseOf = (sub) => sub.name.replace(/(\d)-(\d)/g, "$1–$2");
export const titleOf = (sub) => cap(phraseOf(sub));
const bandOf = (data, id) => data.bands.find((b) => b.id === id);
const inOrder = (all, some) => all.filter((x) => some.includes(x));

/** The switched-on channels and bands in words. */
export function describe(data, st) {
  const ch = inOrder(data.channels, st.chans), bd = inOrder(data.bands.map((b) => b.id), st.bands);
  const c = ch.length === data.channels.length ? `all ${word(ch.length)} channels` : list(ch);
  const b = bd.length === data.bands.length ? `all ${word(bd.length)} bands`
    : bd.length === 1 ? `the ${BAND_WORD[bd[0]]} band` : `the ${list(bd.map((x) => BAND_WORD[x]))} bands`;
  return `${c} with ${b}`;
}
/** Channels times bands: how many features a combination has (the study's F1_ch columns). */
export function features(st) {
  const c = st.chans.length, b = st.bands.length;
  return { c, b, n: c * b, text: `${c} channel${c === 1 ? "" : "s"} × ${b} band${b === 1 ? "" : "s"} = ${c * b} feature${c * b === 1 ? "" : "s"}` };
}

/** The committed subsets one switch away from a state, with the switch that reaches each. */
export function neighbors(data, st) {
  const out = [];
  for (const sub of data.subsets) {
    const flips = [];
    for (const c of data.channels) if (st.chans.includes(c) !== sub.channels.includes(c)) flips.push({ kind: "channel", id: c, on: sub.channels.includes(c) });
    for (const b of data.bands) if (st.bands.includes(b.id) !== sub.bands.includes(b.id)) flips.push({ kind: "band", id: b.id, on: sub.bands.includes(b.id) });
    if (flips.length === 1) out.push({ sub, flip: flips[0] });
  }
  return out;
}
const flipWords = (f) => (f.kind === "channel" ? `${f.on ? "plug in" : "unplug"} ${f.id}` : `turn ${f.on ? "on" : "off"} ${BAND_WORD[f.id]}`);

// ---------------------------------------------------------------------------------------- words
/** The comparison the paper draws for each subset, as a function of the subset's numbers. Keyed by the
 *  file's subset names; values standardized within each subject first, as the paper writes them. */
const SAY = {
  "all channels, all bands": (n) => `Held out, all ${word(n.c)} channels and all ${word(n.b)} bands (${n.n} features) score ${n.s}, with features standardized within each subject (${n.u} unstandardized). Chance is 0.50.`,
  "without Fp1 and Fp2": (n) => `Without Fp1 and Fp2, the two electrodes on the forehead, held-out accuracy falls to ${n.s} (${n.u} unstandardized), a loss of ${fmtAbs(n.vs.d)} ${fmtCI(-n.vs.dhi, -n.vs.dlo)} (${fmtAbs(n.vu.d)} unstandardized).`,
  "Fp1 and Fp2 only": (n) => `Fp1 and Fp2 alone give ${n.s} (${n.u} unstandardized): most of the held-out accuracy above chance comes from the two electrodes on the forehead.`,
  "alpha band only": (n) => `The alpha band alone (${n.ranges}), where mental arithmetic should show up, gives ${n.s} (${n.u} unstandardized).`,
  "alpha band only, without Fp1 and Fp2": (n) => `The alpha band alone at ${list(n.chans)} gives ${n.s} (${n.u} unstandardized).`,
  "without 1-4 and 30-45 Hz": (n) => `Without the ${n.dropped} bands, all ${word(n.c)} channels give ${n.s} (${n.u} unstandardized).`,
  "without 1-4 and 30-45 Hz, without Fp1 and Fp2": (n) => `Without the ${n.dropped} bands and without Fp1 and Fp2, ${list(n.chans)} give ${n.s} (${n.u} unstandardized).`,
};
export const SAY_KEYS = Object.keys(SAY);
function numbersFor(data, sub) {
  const vs = shown(data, sub, "subject"), vu = shown(data, sub, "none");
  const dropped = data.bands.filter((b) => !sub.bands.includes(b.id)).map(bandRange);
  return { vs, vu, s: fmt(vs.mean), u: fmt(vu.mean), c: sub.channels.length, b: sub.bands.length, n: sub.n_features,
    chans: sub.channels, ranges: list(sub.bands.map((id) => bandRange(bandOf(data, id)))), dropped: list(dropped) };
}

/** The sentence under the readout: a function of the state's numbers only. */
export function say(data, st) {
  if (!st.chans.length || !st.bands.length) {
    if (!st.chans.length && !st.bands.length) return "No channel is plugged in and no band is on, so there are no features.";
    return !st.chans.length ? "No channel is plugged in, so there are no features." : "No band is on, so there are no features.";
  }
  const sub = match(data, st);
  if (sub) {
    const n = numbersFor(data, sub);
    return (SAY[sub.name] || ((m) => `${titleOf(sub)} gives ${m.s} (${m.u} unstandardized).`))(n);
  }
  const near = neighbors(data, st);
  const hint = near.length ? ` One switch away: ${list(near.map((x) => `“${phraseOf(x.sub)}” (${flipWords(x.flip)})`))}.` : "";
  return `The study did not run ${describe(data, st)}, so the plate shows no accuracy for it.${hint}`;
}

/** Everything the readout prints for a state, as strings. */
export function readout(data, st) {
  const sub = match(data, st), f = features(st), other = st.norm === "subject" ? "none" : "subject";
  if (!sub) {
    const empty = !st.chans.length || !st.bands.length;
    return { status: empty ? "empty" : "not-run", lit: false, title: empty ? "Nothing patched in" : "Not run", provenance: "",
      combo: f.text, value: "", ci: "", what: "", other: "", change: "", say: say(data, st) };
  }
  const v = shown(data, sub, st.norm), o = shown(data, sub, other);
  return {
    status: "run", lit: true, title: titleOf(sub), name: sub.name,
    provenance: sub.paper_label ? "A row of the paper’s table" : "In the file; the paper doesn’t print this row",
    combo: f.text, value: fmt(v.mean), ci: fmtCI(v.lo, v.hi),
    what: `${cap(NORM_LABEL[st.norm])}: the mean over ${sub.n_subjects} held-out subjects, with its 95% interval`,
    other: `${NORM_SHORT[other]}: ${fmt(o.mean)} ${fmtCI(o.lo, o.hi)}`,
    change: sub.full ? "The full feature set, which the other six are compared with."
      : `Change from all channels and bands, paired over subjects: ${fmt(v.d)} ${fmtCI(v.dlo, v.dhi)}`,
    say: say(data, st),
  };
}
/** Where the needle goes: the lit subset's value for the chosen normalization, or the rest stop. */
export function needleAngle(data, st) {
  const sub = match(data, st);
  return sub ? angleOf(shown(data, sub, st.norm).mean) : METER.rest;
}

/** The notes under the plate, with the caveats the chapter attaches, as the chapter states them. */
export function notes(data) {
  const full = data.subsets.find((s) => s.full), fs = data.full_summary, n = full.n_subjects;
  const extra = data.subsets.filter((s) => !s.paper_label).map((s) => `“${titleOf(s)}”`);
  const w = data.windows, unused = data.recorded_channels - data.channels.length;
  return [
    { head: "What lights", text: `A row of ${data.source} (made by ${data.made_by}): band power per channel with linear discriminant analysis, each of the ${n} subjects held out in turn, and the mean of their balanced accuracies with a 95% bootstrap interval over subjects. The change is the mean paired difference from all channels and bands, with its own interval.` },
    { head: "The first row", text: `The paper prints all channels and bands from the main summary (${fs.source}), and so does the plate: ${fmt(fs.subject.mean)} ${fmtCI(fs.subject.lo, fs.subject.hi)} standardized, ${fmt(fs.none.mean)} ${fmtCI(fs.none.lo, fs.none.hi)} unstandardized. The subsets run repeats it with the same mean and its own bootstrap draws, ${fmtCI(full.subject.lo, full.subject.hi)} and ${fmtCI(full.none.lo, full.none.hi)}, and measures the changes against that run.` },
    { head: "Not in the paper", text: `${cap(word(extra.length))} of the ${word(data.subsets.length)} rows, ${list(extra)}, are in the file but not printed in the paper.` },
    { head: "The sockets", text: `EEGMAT records ${data.recorded_channels} channels, and the study uses ${word(data.channels.length)} of them, over the front of the head. The other ${unused} are capped here, at their 10-20 places on an idealized head; no run used them.` },
    { head: "Recording order", text: "Rest is recorded first, in its own file, so within a person a classifier can tell the two recordings apart." },
    { head: "The forehead", text: "Most of the held-out accuracy above chance comes from the two electrodes on the forehead. Slow and fast power at the forehead is what eye movements and muscle activity look like (the data set’s authors report removing such artifacts, so any left would be residual). So the held-out number may partly reflect recording order and non-brain activity." },
    { head: "Rest", text: `About three-quarters of the EEGMAT windows are rest: ${fmtCount(w.rest)} of ${fmtCount(w.total)}.` },
    { head: "Standardizing", text: "Standardizing each subject’s features over their own recordings, without labels, helps, but it needs the new person’s whole unlabeled session." },
  ];
}

/** The faceplate's legend: how many sites EEGMAT records and how many the study wired. */
export const legend = (data) => `${data.recorded_channels} sites, ${data.channels.length} wired`;
/** The line above the plate: whether it is stepping through the runs, and which run is lit. */
export function tourLine(auto, at, n) {
  if (auto) return `Stepping through the ${word(n)} runs on its own${at >= 0 ? `: ${at + 1} of ${n}` : ""}.`;
  return at >= 0 ? `Run ${at + 1} of ${n} lit.` : "No run lit.";
}

// ---------------------------------------------------------------------------------------- motion
/** The idle tour: hold the first combination, then step to the next committed one every `step` seconds,
 *  in the file's order, round and round. Returns an index into data.subsets. */
export const TOUR = { hold: 4, step: 6.5 };
export function tourIndex(t, n, start = 0) {
  if (t < TOUR.hold) return start;
  return (start + 1 + Math.floor((t - TOUR.hold) / TOUR.step)) % n;
}
/** The needle: a damped spring, as a meter movement swings and settles. One step of dt seconds. */
export const SPRING = { omega: 9, zeta: 0.4 };
export function springStep(a, v, target, dt) {
  const { omega, zeta } = SPRING;
  const acc = omega * omega * (target - a) - 2 * zeta * omega * v;
  const v2 = v + acc * dt;
  return [a + v2 * dt, v2];
}

// ---------------------------------------------------------------------------------------- color
const hexRgb = (h) => { h = h.trim().replace("#", ""); if (h.length === 3) h = [...h].map((c) => c + c).join(""); return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16) / 255); };
const toLin = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
export const relLum = (hex) => { const [r, g, b] = hexRgb(hex).map(toLin); return 0.2126 * r + 0.7152 * g + 0.0722 * b; };
export const contrast = (a, b) => { const [x, y] = [relLum(a), relLum(b)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05); };
