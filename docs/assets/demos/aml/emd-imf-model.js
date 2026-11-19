// The emd-imf plate's model: pure functions, no DOM, importable by Node. The view (emd-imf.js) draws
// what these return, and tools/aml/emd-imf.check.mjs holds every one of them to the study's committed
// window (results/study/imf_example.csv), to a recomputation in Python, and to the paper.
//
// What is shown: one EEG MMI window (C3, Cz, and C4 averaged, then z-scored, so it has no unit), its
// five intrinsic mode functions (IMFs), and the residue. Each piece has a level from 0 to 1 and a mute
// key; the rebuilt trace is the sum of the pieces at their levels, and the residual is the signal minus
// the rebuilt trace. Computed here from the file: each piece's share of the window’s variance, each
// IMF's mean absolute instantaneous frequency (as the study's IMF features compute it), the residual's
// share, and the largest gap between the signal and the sum of its pieces.

export const MINUS = "−";
export const TIMES = "×";

// ---------------------------------------------------------------------------------------- numbers
/** Rounded as the paper rounds: to 10 places, then half up (away from zero). A value that rounds to
 *  zero is written without a sign; negative numbers take a true minus sign. */
export function fmt(x, places = 2) {
  const s = Math.abs(x).toFixed(10), [ip, fp] = s.split(".");
  let q = BigInt(ip + fp.slice(0, places));
  if (fp[places] >= "5") q += 1n;
  let t = q.toString().padStart(places + 1, "0");
  t = places ? `${t.slice(0, t.length - places)}.${t.slice(t.length - places)}` : t;
  return q === 0n || x >= 0 ? t : MINUS + t;
}
/** A share in [0, 1] as a percentage with one decimal. */
export const pct = (share) => `${fmt(100 * share, 1)}%`;
/** A level (a multiple of 0.01) as a multiplier. */
export const level = (g) => `${TIMES}${fmt(g, 2)}`;
export const hz = (f) => `${fmt(f, 1)} Hz`;
/** The largest gap: six decimals when it is below rounding size, three otherwise. */
export const gap = (v) => (v < 0.0005 ? fmt(v, 6) : fmt(v, 3));
/** A sample's time in seconds, exactly (i / 160 has at most five decimals). */
export const secs = (t) => String(Number(t.toFixed(6)));

const WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];
export const count = (n) => WORDS[n] ?? String(n);
/** Names joined with the Oxford comma. */
export function list(items) {
  if (items.length <= 1) return items.join("");
  if (items.length === 2) return `${items[0]} and ${items[1]}`;
  return `${items.slice(0, -1).join(", ")}, and ${items[items.length - 1]}`;
}
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
/** Pieces by name, with a run of three or more consecutive IMFs written as a range ("IMF 1 to IMF 4"). */
export function nameList(pieces, ks) {
  const items = [];
  for (let j = 0; j < ks.length; j++) {
    let e = j;
    while (e + 1 < ks.length && ks[e + 1] === ks[e] + 1 && !pieces[ks[e + 1]].residue) e++;
    if (e - j >= 2 && !pieces[ks[j]].residue) { items.push(`${pieces[ks[j]].name} to ${pieces[ks[e]].name}`); j = e; }
    else items.push(pieces[ks[j]].name);
  }
  return list(items);
}

// ---------------------------------------------------------------------------------------- statistics
export function mean(y) { let s = 0; for (const v of y) s += v; return s / y.length; }
/** The population variance (the mean square about the mean), as numpy's var with ddof 0. */
export function variance(y) {
  const m = mean(y);
  let s = 0;
  for (const v of y) s += (v - m) * (v - m);
  return s / y.length;
}

/** Half a unit in the last kept digit of a value stored with `digits` significant digits (%.6g). */
export function halfUnit(v, digits) {
  if (v === 0) return 0;
  const e = parseInt(v.toExponential().split("e")[1], 10);
  return 0.5 * 10 ** (e - digits + 1);
}

// ---------------------------------------------------------------------------------------- frequency
/** The analytic signal of a real series, as scipy.signal.hilbert computes it: the discrete Fourier
 *  transform, the negative frequencies zeroed and the positive ones doubled, and the inverse transform.
 *  A plain DFT with an exact twiddle table (320 points is small). Returns [re, im]. */
export function analytic(y) {
  const n = y.length, cw = new Float64Array(n), sw = new Float64Array(n);
  for (let m = 0; m < n; m++) { cw[m] = Math.cos((2 * Math.PI * m) / n); sw[m] = Math.sin((2 * Math.PI * m) / n); }
  const Xr = new Float64Array(n), Xi = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    let r = 0, i = 0;
    for (let j = 0; j < n; j++) { const m = (j * k) % n; r += y[j] * cw[m]; i -= y[j] * sw[m]; }
    Xr[k] = r; Xi[k] = i;
  }
  const h = new Float64Array(n);
  h[0] = 1;
  if (n % 2 === 0) { h[n / 2] = 1; for (let k = 1; k < n / 2; k++) h[k] = 2; }
  else for (let k = 1; k < (n + 1) / 2; k++) h[k] = 2;
  const re = new Float64Array(n), im = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    let r = 0, i = 0;
    for (let k = 0; k < n; k++) {
      if (!h[k]) continue;
      const m = (j * k) % n, a = Xr[k] * h[k], b = Xi[k] * h[k];
      r += a * cw[m] - b * sw[m]; i += a * sw[m] + b * cw[m];
    }
    re[j] = r / n; im[j] = i / n;
  }
  return [re, im];
}

/** The mean absolute instantaneous frequency of one IMF in Hz, as the study's IMF features compute it
 *  (scripts/feature_extraction.py, _compute_inst_freq): the phase of the analytic signal, unwrapped as
 *  numpy.unwrap does, differenced, scaled by fs / 2 pi, and the mean of its absolute value. A constant
 *  series gets 0. */
export function meanAbsInstFreq(y, fs) {
  const m = mean(y);
  let s = 0;
  for (const v of y) s += (v - m) * (v - m);
  if (Math.sqrt(s / y.length) < 1e-12) return 0;
  const [re, im] = analytic(y);
  let sum = 0;
  for (let j = 1; j < y.length; j++) {
    const d = Math.atan2(im[j], re[j]) - Math.atan2(im[j - 1], re[j - 1]);
    let dm = d;
    if (Math.abs(d) >= Math.PI) {
      dm = (((d + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
      if (dm === -Math.PI && d > 0) dm = Math.PI;
    }
    sum += Math.abs((dm / (2 * Math.PI)) * fs);
  }
  return sum / (y.length - 1);
}

// ---------------------------------------------------------------------------------------- the window
/** Everything the plate computes once from the file: each piece's share of the window’s variance and
 *  frequency, and the gap between the signal and the sum of all six pieces at every sample, with the
 *  bound that the file's rounding allows there. */
export function prepare(data) {
  const x = data.signal, n = x.length, fs = data.sfreq, vx = variance(x);
  const nImfs = data.pieces.filter((p) => p.id !== "residue").length;
  const pieces = data.pieces.map((p, k) => {
    const residue = p.id === "residue";
    return { k, id: p.id, label: p.label, name: residue ? "the residue" : p.label, residue, y: p.y,
             share: variance(p.y) / vx, freq: residue ? null : meanAbsInstFreq(p.y, fs) };
  });
  let maxGap = 0, maxGapAt = 0, within = true, maxBound = 0;
  for (let i = 0; i < n; i++) {
    let s = 0, b = halfUnit(x[i], data.digits);
    for (const p of pieces) { s += p.y[i]; b += halfUnit(p.y[i], data.digits); }
    const g = Math.abs(s - x[i]);
    if (g > maxGap) { maxGap = g; maxGapAt = i; }
    if (g > b + 1e-12) within = false;
    maxBound = Math.max(maxBound, b);
  }
  let sumShares = 0;
  for (const p of pieces) sumShares += p.share;
  return { n, fs, t: data.t, x, vx, mean: mean(x), sd: Math.sqrt(vx), pieces, nImfs, digits: data.digits,
           maxGap, maxGapAt, maxBound, within, sumShares, ymax: Math.max(...pieces.map((p) => Math.max(...p.y.map(Math.abs)))) };
}

// ---------------------------------------------------------------------------------------- the mix
/** A level as the plate uses it: clamped to [0, 1] and snapped to hundredths, so the readout is exact. */
export const quantize = (g) => Math.round(Math.max(0, Math.min(1, g)) * 100) / 100;
/** The effective level of each piece: 0 when muted, else its fader, snapped. */
export const effective = (faders, muted) => faders.map((g, k) => (muted[k] ? 0 : quantize(g)));

/** The rebuilt trace (each piece times its level, summed), the residual (signal minus rebuilt), the
 *  residual's share of the window’s variance, its largest absolute value, and what the parts taken out
 *  would hold if the pieces were uncorrelated (the sum over pieces of (1 - level)^2 times the share). */
export function mix(prep, gains) {
  const { n, x, pieces } = prep;
  const rebuilt = new Float64Array(n), residual = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let k = 0; k < pieces.length; k++) if (gains[k]) s += gains[k] * pieces[k].y[i];
    rebuilt[i] = s; residual[i] = x[i] - s;
  }
  let maxRes = 0, maxResAt = 0;
  for (let i = 0; i < n; i++) if (Math.abs(residual[i]) > maxRes) { maxRes = Math.abs(residual[i]); maxResAt = i; }
  let apart = 0;
  for (let k = 0; k < pieces.length; k++) apart += (1 - gains[k]) ** 2 * pieces[k].share;
  return { gains: [...gains], rebuilt, residual, lost: variance(residual) / prep.vx, maxRes, maxResAt, apart };
}

/** Which pieces are in at full level, turned down part way, and out. */
export function classify(gains) {
  const full = [], part = [], out = [];
  gains.forEach((g, k) => (g === 1 ? full : g === 0 ? out : part).push(k));
  return { full, part, out };
}

/** The sentence's branch for a set of levels. "build" is the idle loop's settled states: the residue and
 *  the slowest IMFs in at full level, the fastest out, with two to four pieces in. */
export function branch(prep, gains) {
  const { full, part, out } = classify(gains), N = gains.length;
  if (!part.length && !out.length) return "whole";
  if (!part.length && !full.length) return "silent";
  if (!part.length && out.length === 1) return "one-out";
  if (!part.length && full.length === 1) return "solo";
  if (!part.length && full.length >= 2 && full.every((k, j) => k === N - full.length + j)) return "build";
  return "mix";
}

// ---------------------------------------------------------------------------------------- words
const freqWords = (p, nImfs) => (p.residue ? `the slow trend left after the ${count(nImfs)} IMFs` : `a wave with a mean absolute instantaneous frequency of ${hz(p.freq)}`);
/** Tenths of a percent, read back from a shown percentage, so sums agree with what the reader sees. */
const tenths = (s) => parseInt(s.replace(/[.%]/g, ""), 10);
const sumShown = (a, b) => { const t = tenths(a) + tenths(b); return `${Math.floor(t / 10)}.${t % 10}%`; };

/** The sentence under the desk, a function of the window's numbers and the levels only. */
export function say(prep, m) {
  const g = m.gains, P = prep.pieces, b = branch(prep, g), { full, part, out } = classify(g);
  const names = (ks) => nameList(P, ks);
  const are = (ks) => (ks.length === 1 ? "is" : "are");
  const lost = pct(m.lost);
  const corr = (shown, alone, what) => (shown === alone ? "" : ` ${what} ${alone}; the difference comes from correlation between the pieces.`);
  if (b === "whole") {
    return prep.within
      ? `All ${count(P.length)} pieces are in at full level, and they add back to the signal: at all ${prep.n} samples, the gap is at most ${gap(m.maxRes)}, within the rounding of a file that keeps ${count(prep.digits)} significant digits.`
      : `All ${count(P.length)} pieces are in at full level; the largest gap between their sum and the signal is ${gap(m.maxRes)}.`;
  }
  if (b === "silent") return `Every piece is out, so the rebuilt trace is flat at zero and the residual is the whole signal: ${lost} of the window’s variance.`;
  if (b === "one-out") {
    const p = P[out[0]], own = pct(p.share) === lost ? `, ${p.residue ? "the residue" : p.label}’s own share` : "";
    return `Only ${p.name} is out, so the residual is ${p.name} itself (${freqWords(p, prep.nImfs)}), and it holds ${lost} of the window’s variance${own}.`;
  }
  if (b === "solo") {
    const p = P[full[0]], alone = pct(p.share), both = sumShown(alone, lost);
    const tail = both === "100.0%" ? "" : ` The two add to ${both}, not 100%, because the pieces are not exactly uncorrelated.`;
    return `Only ${p.name} is in, ${freqWords(p, prep.nImfs)}. On its own, it holds ${alone} of the window’s variance, and the residual, the other ${count(P.length - 1)} pieces added together, holds ${lost}.${tail}`;
  }
  if (b === "build") {
    return `Built up from the slowest piece: ${names(full)} are in, and ${names(out)} ${are(out)} out. The residual, ${out.length === 1 ? P[out[0]].name : `${names(out)} added together`}, holds ${lost} of the window’s variance.${corr(lost, pct(m.apart), "Their shares add to")}`;
  }
  // mix: out, then turned down, then in at full level
  const clauses = [];
  if (out.length) clauses.push(`${names(out)} ${are(out)} out`);
  if (part.length) {
    const lv = part.map((k) => level(g[k])), same = lv.every((x) => x === lv[0]);
    clauses.push(`${names(part)} ${are(part)} turned down to ${same ? lv[0] : list(lv)}`);
  }
  if (full.length) clauses.push(full.length >= 3 ? `the other ${count(full.length)} pieces are in at full level` : `${names(full)} ${are(full)} in at full level`);
  return `${cap(clauses.length === 2 ? clauses.join(", and ") : list(clauses))}. The residual holds ${lost} of the window’s variance.${corr(lost, pct(m.apart), "Counted piece by piece, what was taken out holds")}`;
}

/** The plate's readouts for one mix: per strip, and for the sum. */
export function readouts(prep, m) {
  return {
    strips: prep.pieces.map((p, k) => ({ id: p.id, label: p.label, freq: p.residue ? "trend" : hz(p.freq), share: pct(p.share), level: level(m.gains[k]),
                                        meter: m.gains[k] ** 2 * p.share })),
    inCount: `${m.gains.filter((x) => x > 0).length} of ${prep.pieces.length}`,
    lost: pct(m.lost),
    gap: gap(m.maxRes),
  };
}

/** What the cursor reads at sample i (no interpolation: it stops only at samples). */
export function cursor(prep, m, i) {
  return { i, t: secs(prep.t[i]), signal: fmt(prep.x[i], 3), rebuilt: fmt(m.rebuilt[i], 3), residual: fmt(m.residual[i], 3) };
}
export const sayCursor = (c, n) => `At ${c.t} s (sample ${c.i + 1} of ${n}): signal ${c.signal}, rebuilt ${c.rebuilt}, residual ${c.residual}.`;
/** The sample nearest to a time in seconds. */
export const sampleAt = (sec, fs, n) => Math.max(0, Math.min(n - 1, Math.round(sec * fs)));

// ---------------------------------------------------------------------------------------- the idle loop
/** The slow loop when nobody is touching the plate, in seconds: hold the whole signal, take every piece
 *  out, then bring the pieces in one at a time from the residue up to IMF 1, each rising over `rise` and
 *  held for `hold`, and hold the whole signal again. The first cycle starts on the whole signal. */
export const LOOP = { first: 3, down: 1.6, silent: 1.6, rise: 1.4, hold: 3.4, whole: 4 };
const ease = (u) => 0.5 - 0.5 * Math.cos(Math.PI * Math.min(1, Math.max(0, u)));
export const loopCycle = (N = 6) => LOOP.down + LOOP.silent + N * (LOOP.rise + LOOP.hold) + LOOP.whole;
export function loopAt(t, N = 6) {
  const all = (v) => new Array(N).fill(v);
  if (t < LOOP.first) return { phase: "whole", piece: null, since: t, gains: all(1) };
  let s = (t - LOOP.first) % loopCycle(N);
  if (s < LOOP.down) return { phase: "down", piece: null, since: s, gains: all(1 - ease(s / LOOP.down)) };
  s -= LOOP.down;
  if (s < LOOP.silent) return { phase: "silent", piece: null, since: s, gains: all(0) };
  s -= LOOP.silent;
  for (let j = 0; j < N; j++) {
    const k = N - 1 - j, g = all(0);
    for (let q = k + 1; q < N; q++) g[q] = 1;
    if (s < LOOP.rise) { g[k] = ease(s / LOOP.rise); return { phase: "rise", piece: k, since: s, gains: g }; }
    s -= LOOP.rise;
    if (s < LOOP.hold) { g[k] = 1; return { phase: "hold", piece: k, since: s, gains: g }; }
    s -= LOOP.hold;
  }
  return { phase: "whole", piece: null, since: s, gains: all(1) };
}
/** While the loop moves a fader, the sentence says what it is doing, with no numbers. */
export function sayLoop(prep, phase) {
  if (phase.phase === "down") return "Taking every piece out.";
  if (phase.phase === "rise") return `Bringing in ${prep.pieces[phase.piece].name}.`;
  return null;
}

/** The night printing's beam: the sample it has reached, at the window's own pace (two seconds per sweep). */
export const beamAt = (t, n, fs) => Math.floor(((t % (n / fs)) + n / fs) % (n / fs) * fs) % n;

// ---------------------------------------------------------------------------------------- the fader's weight
/** One step of a critically damped spring toward the target: the fader cap's slight weight. Returns
 *  [position, velocity]; it settles exactly on the target once close. */
export function follow(x, v, target, dt, omega = 20) {
  const a = omega * omega * (target - x) - 2 * omega * v;
  v += a * dt; x += v * dt;
  if (Math.abs(target - x) < 0.002 && Math.abs(v) < 0.05) return [target, 0];
  return [x, v];
}

// ---------------------------------------------------------------------------------------- color
const hexRgb = (h) => { h = h.trim().replace("#", ""); if (h.length === 3) h = [...h].map((c) => c + c).join(""); return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16) / 255); };
const toLin = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
export const relLum = (hex) => { const [r, g, b] = hexRgb(hex).map(toLin); return 0.2126 * r + 0.7152 * g + 0.0722 * b; };
export const contrast = (a, b) => { const [x, y] = [relLum(a), relLum(b)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05); };
