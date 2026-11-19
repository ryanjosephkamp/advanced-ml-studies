// The emd-head plate's model: pure functions, no DOM, importable by Node. The view (emd-head.js) draws
// what these return, and tools/aml/emd-head.check.mjs holds every one of them to the study's committed
// spectra (results/study/spectra.csv) and to the paper's printed numbers (paper/numbers.tex).
//
// What is shown: at each measured frequency, the right-minus-left difference in log power (dB) at C3,
// Cz, and C4, and in the "average" channel (the three channels averaged into one signal before its
// spectrum; the original pipeline also z-scored each window, which this spectrum does not), each with
// its 95% bootstrap interval over subjects.

export const ELECTRODES = ["C3", "Cz", "C4"];
export const ROWS = ["C3", "Cz", "C4", "average"];
export const ROW_LABEL = { C3: "C3", Cz: "Cz", C4: "C4", average: "Average" };
export const SIDE = { C3: "left", Cz: "midline", C4: "right" };
export const MINUS = "−";

// ---------------------------------------------------------------------------------------- the head
// A unit sphere in a right-handed frame: x toward the subject's right ear, y up through the vertex, z
// toward the back of the head. The nasion, inion, and the two preauricular points lie on the equator.
export const LANDMARKS = {
  nasion: [0, 0, -1], inion: [0, 0, 1], lpa: [-1, 0, 0], rpa: [1, 0, 0], vertex: [0, 1, 0],
};

/** The point a fraction f of the way along the arc from landmark `from` over the vertex to the
 *  opposite landmark: f = 0 is `from`, 0.5 the vertex, 1 the opposite point (the arc is 180 degrees). */
export function alongArc(from, f) {
  const a = LANDMARKS[from], v = LANDMARKS.vertex, t = Math.PI * f;
  return [Math.cos(t) * a[0] + Math.sin(t) * v[0], Math.cos(t) * a[1] + Math.sin(t) * v[1], Math.cos(t) * a[2] + Math.sin(t) * v[2]];
}

/** The 10-20 rule for the three electrodes: Cz halfway along the ear-to-ear arc (and the nasion-inion
 *  arc), C3 and C4 at 30% and 70% of the ear-to-ear arc from the left preauricular point. */
export const TEN_TWENTY = { C3: 0.3, Cz: 0.5, C4: 0.7 };
export const electrode = (name) => alongArc("lpa", TEN_TWENTY[name]);

// ---------------------------------------------------------------------------------------- the camera
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; };

export const VIEW = {
  dist: 5.4,                 // camera distance in head radii
  top: { pol: 0, az: 0 },    // straight down, the nose at the top of the picture
  tilt: { pol: 0.93, az: 0 }, // about 53 degrees down from the vertex, from behind
  polMax: 1.4, zoomMin: 0.6, zoomMax: 2.6, // the camera stays above the ear line
};

/** The camera for a view: `pol` is the angle from the vertex, `az` turns the camera around the vertical
 *  axis (0 = from behind, so the subject's left is on the left of the picture). `scale` is the picture
 *  radius in pixels of a sphere of radius 1 at the center of the view, before zoom. */
export function camera(view, w, h, scale) {
  const { pol, az, zoom = 1 } = view, D = VIEW.dist;
  const d = [Math.sin(az), 0, Math.cos(az)];
  const C = [D * (Math.sin(pol) * d[0]), D * Math.cos(pol), D * (Math.sin(pol) * d[2])];
  const f = norm([-C[0], -C[1], -C[2]]);
  const u = norm([-Math.cos(pol) * d[0], Math.sin(pol), -Math.cos(pol) * d[2]]);
  const r = cross(f, u);
  return { C, f, u, r, focal: scale * zoom * Math.sqrt(D * D - 1), cx: w / 2, cy: h / 2, D };
}

/** A point of the head frame in picture coordinates (y down), and its depth along the view. */
export function project(cam, P) {
  const v = sub(P, cam.C), z = dot(v, cam.f);
  return { x: cam.cx + (dot(v, cam.r) / z) * cam.focal, y: cam.cy - (dot(v, cam.u) / z) * cam.focal, z };
}

/** The picture radius of the head's outline (the sphere's tangent cone). */
export const outlineRadius = (cam) => cam.focal / Math.sqrt(cam.D * cam.D - 1);

/** A point on the sphere faces the camera. */
export const facing = (cam, P) => dot(P, cam.C) > dot(P, P);

/** A point outside the sphere is hidden behind it. */
export function hidden(cam, Q) {
  const d = sub(Q, cam.C), a = dot(d, d), b = 2 * dot(cam.C, d), c = dot(cam.C, cam.C) - 1;
  const disc = b * b - 4 * a * c;
  if (disc <= 0) return false;
  const t = (-b - Math.sqrt(disc)) / (2 * a);
  return t > 0 && t < 1 - 1e-9;
}

/** The outline: the circle where the camera's tangent cone touches the sphere, as head-frame points. */
export function outlineCircle(cam, n = 144) {
  const D = cam.D, c = cam.C.map((x) => x / (D * D)), rho = Math.sqrt(1 - 1 / (D * D));
  const out = [];
  for (let k = 0; k <= n; k++) {
    const t = (2 * Math.PI * k) / n, a = Math.cos(t) * rho, b = Math.sin(t) * rho;
    out.push([c[0] + a * cam.r[0] + b * cam.u[0], c[1] + a * cam.r[1] + b * cam.u[1], c[2] + a * cam.r[2] + b * cam.u[2]]);
  }
  return out;
}

// ---------------------------------------------------------------------------------------- the data
export const variantOf = (data, id) => data.variants.find((v) => v.id === id) || data.variants[0];

/** The measured frequency nearest to f (the slider and the strip snap to these), as an index. */
export function snapIndex(freqs, f) {
  let best = 0;
  for (let i = 1; i < freqs.length; i++) if (Math.abs(freqs[i] - f) < Math.abs(freqs[best] - f)) best = i;
  return best;
}

/** One state of the plate: a variant at a measured frequency, every row read straight from the data. */
export function stateAt(data, id, i) {
  const v = variantOf(data, id);
  const rows = {};
  for (const k of ROWS) { const c = v.channels[k]; rows[k] = { mean: c.mean[i], lo: c.lo[i], hi: c.hi[i] }; }
  return { variant: v.id, label: v.label, n: v.n_subjects, i, freq: data.freqs[i], rows };
}

// ---------------------------------------------------------------------------------------- numbers
/** Rounded as the paper's make_numbers.py does: to 9 places, then half up (away from zero). A value
 *  that rounds to zero is written without a sign. Negative numbers take a true minus sign. */
export function fmt(x, places = 2) {
  const s = Math.abs(x).toFixed(9), [ip, fp] = s.split(".");
  let q = BigInt(ip + fp.slice(0, places));
  if (fp[places] >= "5") q += 1n;
  let t = q.toString().padStart(places + 1, "0");
  t = `${t.slice(0, t.length - places)}.${t.slice(t.length - places)}`;
  return q === 0n || x >= 0 ? t : MINUS + t;
}
export const fmtAbs = (x, places = 2) => fmt(Math.abs(x), places);
export const fmtFreq = (f) => (Number.isInteger(f) ? String(f) : String(f));

/** Whether the 95% interval leaves out zero, and on which side. */
export const direction = (r) => (r.lo > 0 ? "higher" : r.hi < 0 ? "lower" : "unclear");

/** One end of an interval, rounded as the paper does. When the interval leaves out zero but an end
 *  would print as zero at two places, that end takes more places (up to five) until it shows its sign,
 *  so a readout never says "0.00" for an interval that leaves out zero. */
export function endText(x, clear) {
  let p = 2;
  while (clear && p < 5 && /^0\.0+$/.test(fmt(x, p))) p++;
  return fmt(x, p);
}
export const interval = (r) => { const clear = direction(r) !== "unclear"; return `${endText(r.lo, clear)} to ${endText(r.hi, clear)}`; };

/** Below this many subjects, the Intervals note adds the paper's caveat that percentile intervals over
 *  so few subjects may run too narrow (the paper says so for the 9 subjects of BCI IV 2a). */
export const FEW_SUBJECTS = 20;
/** The Intervals note for a variant, from its subject count and the number of measured frequencies. */
export function intervalsNote(v, nFreqs) {
  const few = v.n_subjects < FEW_SUBJECTS ? ` With only ${v.n_subjects} subjects, these percentile intervals may run too narrow.` : "";
  return `95% bootstrap intervals over the ${v.n_subjects} subjects, one frequency at a time, so some of the ${nFreqs} can leave out zero by chance.${few} The strip’s lines join the ${nFreqs} measured frequencies; the slider stops only at those.`;
}

/** The color scale's half range: the largest |mean| in the variant (all four rows, all frequencies),
 *  rounded up to the next 0.1 dB, so no value is clipped. */
export function scaleLimit(v) {
  let m = 0;
  for (const k of ROWS) for (const x of v.channels[k].mean) m = Math.max(m, Math.abs(x));
  return Math.ceil(m * 10 - 1e-9) / 10;
}
/** The strip's half range: the largest |interval end| in the variant, rounded up to the next 0.1 dB. */
export function stripLimit(v) {
  let m = 0;
  for (const k of ROWS) for (const key of ["lo", "hi"]) for (const x of v.channels[k][key]) m = Math.max(m, Math.abs(x));
  return Math.ceil(m * 10 - 1e-9) / 10;
}

// ---------------------------------------------------------------------------------------- color
const hexRgb = (h) => { h = h.trim().replace("#", ""); if (h.length === 3) h = [...h].map((c) => c + c).join(""); return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16) / 255); };
const toLin = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const toSrgb = (c) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);
export function oklab(hex) {
  const [r, g, b] = hexRgb(hex).map(toLin);
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s, 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s, 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s];
}
function fromOklab([L, A, B]) {
  const l = (L + 0.3963377774 * A + 0.2158037573 * B) ** 3, m = (L - 0.1055613458 * A - 0.0638541728 * B) ** 3, s = (L - 0.0894841775 * A - 1.291485548 * B) ** 3;
  const rgb = [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s, -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s, -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s];
  return "#" + rgb.map((c) => Math.round(Math.min(1, Math.max(0, toSrgb(c))) * 255).toString(16).padStart(2, "0")).join("");
}
export const distance = (a, b) => { const p = oklab(a), q = oklab(b); return Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]); };

/** The symmetric diverging scale: zero takes the `zero` ink, -limit the `neg` ink, +limit the `pos` ink,
 *  mixed in OKLab in proportion to |value| / limit. */
export function inkAt(value, limit, inks) {
  const t = Math.max(-1, Math.min(1, value / limit)), end = oklab(t < 0 ? inks.neg : inks.pos), z = oklab(inks.zero), a = Math.abs(t);
  return fromOklab([z[0] + (end[0] - z[0]) * a, z[1] + (end[1] - z[1]) * a, z[2] + (end[2] - z[2]) * a]);
}
export const relLum = (hex) => { const [r, g, b] = hexRgb(hex).map(toLin); return 0.2126 * r + 0.7152 * g + 0.0722 * b; };
export const contrast = (a, b) => { const [x, y] = [relLum(a), relLum(b)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05); };

// ---------------------------------------------------------------------------------------- words
/** The readout rows: label, side, mean, and interval, as shown. */
export function readouts(st) {
  return ROWS.map((k) => ({ key: k, label: ROW_LABEL[k], side: SIDE[k] || "one signal", mean: fmt(st.rows[k].mean), ci: interval(st.rows[k]), dir: direction(st.rows[k]) }));
}

const WITH = "with the right hand imagined instead of the left";

/** C3 and C4 differ clearly in opposite directions, and the channel average's interval includes zero
 *  with a mean at most half the smaller of the two: the plate then says the two sides nearly cancel. */
export function cancels(st) {
  const c3 = st.rows.C3, c4 = st.rows.C4, avg = st.rows.average, d3 = direction(c3), d4 = direction(c4);
  return d3 !== "unclear" && d4 !== "unclear" && d3 !== d4 && direction(avg) === "unclear"
    && Math.abs(avg.mean) <= 0.5 * Math.min(Math.abs(c3.mean), Math.abs(c4.mean));
}

/** The sentence under the readouts, a function of the state's numbers only. It describes the mean
 *  differences and whether each pointwise 95% interval leaves out zero; it claims no effect. */
export function say(st) {
  const c3 = st.rows.C3, c4 = st.rows.C4, avg = st.rows.average;
  const d3 = direction(c3), d4 = direction(c4), da = direction(avg), f = fmtFreq(st.freq);
  let one;
  if (d3 !== "unclear" && d4 !== "unclear" && d3 !== d4) {
    one = `At ${f} Hz, ${WITH}, power is ${fmtAbs(c3.mean)} dB ${d3} at C3 and ${fmtAbs(c4.mean)} dB ${d4} at C4, and both 95% intervals leave out zero: the two sides go opposite ways.`;
  } else if (d3 !== "unclear" && d3 === d4) {
    one = `At ${f} Hz, ${WITH}, power is ${d3} at both C3 and C4, by ${fmtAbs(c3.mean)} and ${fmtAbs(c4.mean)} dB, and both 95% intervals leave out zero.`;
  } else if (d3 !== "unclear") {
    one = `At ${f} Hz, ${WITH}, power is ${fmtAbs(c3.mean)} dB ${d3} at C3, and its 95% interval leaves out zero; at C4, the 95% interval includes zero (${fmt(c4.mean)} dB).`;
  } else if (d4 !== "unclear") {
    one = `At ${f} Hz, ${WITH}, power is ${fmtAbs(c4.mean)} dB ${d4} at C4, and its 95% interval leaves out zero; at C3, the 95% interval includes zero (${fmt(c3.mean)} dB).`;
  } else {
    one = `At ${f} Hz, neither C3 nor C4 shows a clear difference between right- and left-hand imagery: both 95% intervals include zero (${fmt(c3.mean)} and ${fmt(c4.mean)} dB).`;
  }
  let two;
  if (cancels(st)) two = `Averaged into one channel, they nearly cancel: ${fmt(avg.mean)} dB, with an interval that includes zero.`;
  else if (da === "unclear") two = `The channel average shows no clear difference: ${fmt(avg.mean)} dB, with an interval that includes zero.`;
  else two = `The channel average is ${fmtAbs(avg.mean)} dB ${da}, with an interval that leaves out zero.`;
  return `${one} ${two}`;
}
/** Which branch a state's sentence takes (for the checks' coverage). */
export function branch(st) {
  const d3 = direction(st.rows.C3), d4 = direction(st.rows.C4), da = direction(st.rows.average);
  const a = d3 !== "unclear" && d4 !== "unclear" ? (d3 !== d4 ? "opposite" : "same") : d3 !== "unclear" ? "c3-only" : d4 !== "unclear" ? "c4-only" : "neither";
  return `${a}/${cancels(st) ? "cancel" : da === "unclear" ? "avg-unclear" : "avg-clear"}`;
}

// ---------------------------------------------------------------------------------------- motion
const ease = (x) => 0.5 - 0.5 * Math.cos(Math.PI * Math.min(1, Math.max(0, x)));
/** The slow turn when nobody is touching the plate, as a function of seconds: hold the top view, tip
 *  over to the tilted view from behind, swing gently to each side, and come back. One cycle is 40 s. */
export const TURN = { hold: 4, tip: 7, swing: 22, back: 7, swingAz: 0.6 };
export function autoView(t) {
  const T = TURN, cyc = T.hold + T.tip + T.swing + T.back, s = ((t % cyc) + cyc) % cyc;
  if (s < T.hold) return { pol: VIEW.top.pol, az: 0 };
  if (s < T.hold + T.tip) return { pol: VIEW.tilt.pol * ease((s - T.hold) / T.tip), az: 0 };
  if (s < T.hold + T.tip + T.swing) return { pol: VIEW.tilt.pol, az: T.swingAz * Math.sin((2 * Math.PI * (s - T.hold - T.tip)) / T.swing) };
  return { pol: VIEW.tilt.pol * (1 - ease((s - T.hold - T.tip - T.swing) / T.back)), az: 0 };
}
/** The slow sweep when idle: up and down the measured frequencies from `lo` to `hi` Hz, one step every
 *  `step` seconds, after holding the start frequency for `hold` seconds. Returns an index into freqs. */
export function sweepIndex(t, freqs, lo, hi, start, step = 0.55, hold = 3) {
  const idx = freqs.map((f, i) => [f, i]).filter(([f]) => f >= lo && f <= hi).map(([, i]) => i);
  const n = idx.length, s0 = Math.max(0, idx.indexOf(start));
  if (t < hold || n < 2) return idx[s0] ?? start;
  const k = Math.floor((t - hold) / step), period = 2 * (n - 1), p = (((s0 + k) % period) + period) % period;
  return idx[p < n ? p : period - p];
}
