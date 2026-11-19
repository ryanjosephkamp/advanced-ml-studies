// The model behind Chapter III's live plate "Every cell the study measured": a table set in type,
// one data set at a time, with one row per matrix family and one column per target size k the
// study ran, and PCA and all features as reference rows. Every block is one row of the study's
// committed summary (study/results/accuracy_summary.csv) or of its paired differences
// (study/results/paired_differences.csv); this module lays them out, rounds them as the paper
// does, sets the ink density each block is printed at, and writes every sentence the plate shows,
// as functions of numbers. Pure functions, no DOM: projection-heatmap.js draws them, and
// tools/aml/projection-heatmap.check.mjs holds them to the study's files and the paper.

export const MINUS = "−";
/** Differences ink fully at 10 points and beyond. */
export const DIFF_FULL = 0.1;
/** The study's level for Holm-adjusted p-values (paper/make_numbers.py counts p < 0.05). */
export const ALPHA = 0.05;
/** The column the plate reads down on its own, before it is touched. */
export const SWEEP_K = 2;

// ---- numbers as the paper prints them -------------------------------------------------------------

/**
 * x rounded half up (ties away from zero) to a number of decimals, as paper/make_numbers.py's rnd
 * does after rounding to 8 places. A negative value that rounds to zero keeps its sign ("−0.0"), as
 * Python's Decimal does.
 */
export function halfUp(x, places) {
  const neg = x < 0;
  const [ip, fp] = Math.abs(x).toFixed(8).split(".");
  const unit = 10n ** BigInt(8 - places);
  const all = BigInt(ip + fp);
  let q = all / unit;
  if ((all % unit) * 2n >= unit) q += 1n;
  const str = q.toString().padStart(places + 1, "0");
  const out = places ? `${str.slice(0, -places)}.${str.slice(-places)}` : str;
  return (neg ? MINUS : "") + out;
}
/** A score in [0, 1] as a percentage with one decimal ("88.6"). */
export const pct1 = (x) => halfUp(100 * x, 1);
/** A difference of scores in points, signed as the paper signs it ("−9.0", "1.3"). */
export const pts1 = (x) => halfUp(100 * x, 1);
/**
 * The same with a plus sign on a positive difference, for the blocks and the slip ("+1.3"). A value
 * that rounds to zero gets no plus: Breast Cancer's Eq. 3 interval at k = 4 ends at 1.1e-18, which
 * is zero up to floating-point error, and reads "0.0".
 */
export const signed1 = (x) => {
  const s = pts1(x);
  return x > 0 && s !== "0.0" ? `+${s}` : s;
};
/** The size of a difference in points, without its sign ("9.0"). */
export const abs1 = (x) => pts1(x).replace(MINUS, "");
/** A p-value to three decimals, as the paper prints it, or "< 0.001" when that would read 0.000. */
export const p3 = (p) => (halfUp(p, 3) === "0.000" ? "< 0.001" : halfUp(p, 3));
/** "p = 0.021" or "p < 0.001". */
export const pEq = (p) => (p3(p).startsWith("<") ? `p ${p3(p)}` : `p = ${p3(p)}`);
export const metricWord = (ds) => (ds.metric === "bal_acc" ? "balanced accuracy" : "accuracy");
export const count = (share, n) => Math.round(share * n);
export const list = (xs) => (xs.length < 3 ? xs.join(" and ") : `${xs.slice(0, -1).join(", ")}, and ${xs.at(-1)}`);

// ---- the forme: rows, columns, and blocks ---------------------------------------------------------

export const datasetOf = (data, id) => data.datasets.find((d) => d.id === id) || data.datasets[0];
const famLabel = (data, id) => (data.families.find((f) => f.id === id) || data.references.find((f) => f.id === id)).label;

/** The sparsity s of a family, as the row head writes it. */
export function sparsityText(ds, fam) {
  if (fam === "gaussian") return "dense";
  if (fam === "verysparse") {
    const exact = Number.isInteger(Math.sqrt(ds.d));
    return `s = √${ds.d} ${exact ? "=" : "≈"} ${ds.s_text.verysparse}`;
  }
  if (fam === "onepercol") return `s = d = ${ds.s_text.onepercol}`;
  return `s = ${ds.s_text[fam]}`;
}

/** The rows of one data set's table, top to bottom: the five families, then the two references. */
export function rows(data, ds) {
  const out = data.families.map((f) => ({
    id: f.id, kind: "family", label: f.label, run: ds.families.includes(f.id),
    sub: ds.families.includes(f.id) ? sparsityText(ds, f.id) : "not run",
  }));
  out.push({ id: "pca", kind: "reference", label: famLabel(data, "pca"), sub: "fit to the training rows", run: true });
  out.push({ id: "full", kind: "reference", label: famLabel(data, "full"), sub: `d = ${ds.d}, no projection`, run: true });
  return out;
}

/** Why a family has no row of blocks on a data set (Very sparse on Glass). */
export function notRunNote(ds, fam) {
  if (fam === "verysparse") return `Not run: on ${ds.label}, √d = √${ds.d} = ${Math.sqrt(ds.d)}, so very sparse would be the same matrix as Eq. 4.`;
  return `Not run on ${ds.label}.`;
}

/**
 * The score's tints, as a printer's tint ladder: ten steps of the ink mixed into the paper, one for
 * each tenth of the way from the floor (the majority-class share, or chance balanced accuracy) to
 * 100%. From the sixth step on, the figures are reversed out of the ink. The steps skip the band
 * around the middle where neither solid nor reversed figures would read at 4.5:1 in both printings.
 */
export const SCORE_TINTS = [0, 0.1, 0.2, 0.3, 0.4, 0.58, 0.685, 0.79, 0.895, 1];
export const REVERSED_FROM = 5;
/**
 * A difference's tints, one step per whole point, in vermilion below the Gaussian matrix and blue
 * above it. The two inks are lighter than the book's ink, so the band they skip lies higher, and on
 * the night board lower: by day the figures reverse from 8 points, at night from 6.
 */
export const DIFF_TINTS = {
  day: [0, 0.095, 0.19, 0.285, 0.38, 0.475, 0.57, 0.665, 0.92, 0.96, 1],
  night: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.85, 0.9, 0.95, 1],
};
export const DIFF_REVERSED_FROM = { day: 8, night: 6 };
export const PRINTINGS = ["day", "night"];
/** How a block of one side and step is printed in one printing: its tint, and whether its figures are reversed. */
export function inkOf(side, level, printing) {
  if (side === "ref" || side === "zero") return { tint: 0, reversed: false };
  if (side === "ink") return { tint: SCORE_TINTS[level], reversed: level >= REVERSED_FROM };
  return { tint: DIFF_TINTS[printing][level], reversed: level >= DIFF_REVERSED_FROM[printing] };
}
const inks = (side, level) => Object.fromEntries(PRINTINGS.map((p) => [p, inkOf(side, level, p)]));
/** The step of a score: which tenth of the way from the floor to 100% it falls in. */
export const scoreLevel = (v, floor) => Math.min(9, Math.max(0, Math.floor((10 * (v - floor)) / (1 - floor))));
/** The step of a difference: its whole points, as printed (−2.6 is step 2), up to 10 or more. */
export const diffLevel = (d) => Math.min(Math.round(100 * DIFF_FULL), Math.floor(Number(abs1(d))));
export const sideOf = (d) => (d < 0 ? "neg" : d > 0 ? "pos" : "zero");

/** The difference of a row from the Gaussian matrix at column j, in the direction "row minus Gaussian". */
export function vsGauss(ds, row, j) {
  if (row === "pca" || row === "full") {
    // the study records Gaussian minus the reference; reversing its sign reverses its interval
    const r = ds.diffs[`gaussian|${row}`][j];
    return { mean: -r.mean, lo: -r.hi, hi: -r.lo, holm: null, equiv: r.equiv, primary: false, higher: r.lower, equal: r.equal, lower: r.higher, flipped: true };
  }
  const r = ds.diffs[`${row}|gaussian`]?.[j];
  return r ? { ...r, flipped: false } : null;
}

/** Footnote marks for a family's block: * survives Holm's correction against the Gaussian, † interval inside ±2 points. */
export function marks(ds, row, j) {
  if (row === "gaussian" || row === "pca" || row === "full") return "";
  const r = ds.diffs[`${row}|gaussian`]?.[j];
  if (!r) return "";
  return (r.holm !== null && r.holm < ALPHA ? "*" : "") + (r.equiv ? "†" : "");
}

function scoreInk(v, floor) {
  const level = scoreLevel(v, floor);
  return { level, print: inks("ink", level) };
}

/**
 * Every block drawn for one data set in one mode ("score" or "diff"). Each has its row index r, its
 * first and last column c0..c1, what it shows, and how much ink. Only measured cells are blocks.
 */
export function blocks(data, ds, mode) {
  const rs = rows(data, ds), out = [], n = ds.ks.length, floor = ds.floor.value;
  rs.forEach((row, r) => {
    if (!row.run) return;
    if (row.id === "full" && mode === "score") {
      const [v, lo, hi] = ds.full;
      out.push({ r, c0: 0, c1: n - 1, row: row.id, k: null, kind: "score", value: v, lo, hi, text: pct1(v), ...scoreInk(v, floor), side: "ink", marks: "" });
      return;
    }
    ds.ks.forEach((k, j) => {
      const base = { r, c0: j, c1: j, row: row.id, k, marks: marks(ds, row.id, j) };
      const [v, lo, hi] = row.id === "full" ? ds.full : ds.scores[row.id][j];
      if (mode === "score") {
        out.push({ ...base, kind: "score", value: v, lo, hi, text: pct1(v), ...scoreInk(v, floor), side: "ink" });
      } else if (row.id === "gaussian") {
        out.push({ ...base, kind: "reference", value: v, lo, hi, text: pct1(v), level: 0, print: inks("ref", 0), side: "ref" });
      } else {
        const d = vsGauss(ds, row.id, j), level = diffLevel(d.mean);
        const side = sideOf(d.mean);
        out.push({ ...base, kind: "diff", value: d.mean, lo: d.lo, hi: d.hi, text: signed1(d.mean), level, print: inks(side, level), side });
      }
    });
  });
  return out;
}

/** The block covering row r at column c, if any. */
export const blockAt = (bs, r, c) => bs.find((b) => b.r === r && b.c0 <= c && c <= b.c1) || null;

/**
 * Moving the reading point with the arrow keys: left and right step along a row, up and down step to
 * the nearest row with a block under the same column (skipping a row that was not run), Home and End
 * go to the row's ends. The column is kept, so passing through the one-block All features row comes
 * back to the same k. Returns {r, c}; it stops at the edges.
 */
export function move(bs, pos, key) {
  const here = blockAt(bs, pos.r, pos.c);
  if (!here) return pos;
  const rowsWith = [...new Set(bs.map((b) => b.r))].sort((a, b) => a - b);
  const inRow = (r) => bs.filter((b) => b.r === r).sort((a, b) => a.c0 - b.c0);
  const ncol = Math.max(...bs.map((b) => b.c1)) + 1;
  if (key === "ArrowLeft" || key === "ArrowRight") {
    const row = inRow(pos.r), i = row.indexOf(here), next = row[i + (key === "ArrowLeft" ? -1 : 1)];
    return next ? { r: pos.r, c: key === "ArrowLeft" ? next.c1 : next.c0 } : pos;
  }
  if (key === "Home") return { r: pos.r, c: inRow(pos.r)[0].c0 };
  if (key === "End") return { r: pos.r, c: inRow(pos.r).at(-1).c1 };
  if (key === "ArrowUp" || key === "ArrowDown") {
    const i = rowsWith.indexOf(pos.r), r = rowsWith[i + (key === "ArrowUp" ? -1 : 1)];
    return r === undefined ? pos : { r, c: Math.min(pos.c, ncol - 1) };
  }
  return pos;
}

/** The plate's own reading, before it is touched: down the k = 2 column, top to bottom. */
export function sweep(bs, ds) {
  const c = Math.max(0, ds.ks.indexOf(SWEEP_K));
  return [...new Set(bs.map((b) => b.r))].sort((a, b) => a - b).map((r) => ({ r, c }));
}

// ---- the slip: what a tapped block reads ----------------------------------------------------------

/** One comparison line on the slip. */
function line(name, d, n, note) {
  const primary = d.holm !== null && d.holm !== undefined;
  return {
    name, value: signed1(d.mean), lo: signed1(d.lo), hi: signed1(d.hi),
    holm: primary ? p3(d.holm) : null, survives: primary && d.holm < ALPHA,
    equiv: !!d.equiv, counts: { below: count(d.lower, n), tied: count(d.equal, n), above: count(d.higher, n) }, note: note || "",
  };
}

/**
 * Everything the slip shows for one block: its heading, its figure and interval, the comparisons
 * the study measured for it (with their intervals and Holm results where it has them), and the
 * sentence that reads it.
 */
export function slip(data, ds, b) {
  const N = data.n_draws, label = famLabel(data, b.row), metric = metricWord(ds);
  const kText = b.k === null ? "one value for every k" : `k = ${b.k}`;
  const j = b.k === null ? -1 : ds.ks.indexOf(b.k);
  const v = b.row === "full" ? ds.full : ds.scores[b.row][j];
  const out = {
    head: `${ds.label} · ${label}${b.row !== "pca" && b.row !== "full" ? ` · ${sparsityText(ds, b.row)}` : ""} · ${kText}`,
    value: pct1(v[0]), lo: pct1(v[1]), hi: pct1(v[2]), metric, lines: [], say: "",
  };
  if (b.row === "full" && b.k === null) {
    out.say = sayFull(ds, v);
    return out;
  }
  if (b.row === "gaussian") {
    const p = ds.diffs["gaussian|pca"][j], f = ds.diffs["gaussian|full"][j];
    out.lines.push(line("Minus PCA", p, N), line("Minus all features", f, N));
    out.say = sayGaussian(ds, b.k, v, p, f);
    return out;
  }
  if (b.row === "pca" || b.row === "full") {
    const g = vsGauss(ds, b.row, j);
    out.lines.push(line("Minus the Gaussian", g, N, `the study's Gaussian minus ${b.row === "pca" ? "PCA" : "all features"}, sign reversed`));
    if (b.row === "pca") out.lines.push(line("Minus all features", ds.diffs["pca|full"][j], N));
    out.say = b.row === "pca" ? sayPca(ds, b.k, v, g) : sayFullAt(ds, b.k, v, g);
    return out;
  }
  const g = ds.diffs[`${b.row}|gaussian`][j];
  out.lines.push(line("Minus the Gaussian", g, N));
  const e3 = b.row === "achlioptas3" ? ds.diffs["achlioptas3|rademacher"]?.[j] : null;
  if (e3) out.lines.push(line("Minus Eq. 3", e3, N));
  out.lines.push(line("Minus all features", ds.diffs[`${b.row}|full`][j], N));
  out.say = sayFamily(data, ds, b.row, b.k, v, g, e3);
  return out;
}

// ---- sentences ------------------------------------------------------------------------------------

const ci = (d) => `95% interval ${pts1(d.lo)} to ${pts1(d.hi)}`;
const holmSentence = (data, d) => (d.holm < ALPHA
  ? `The difference survives Holm's correction over the study's ${data.n_primary} primary comparisons (adjusted ${pEq(d.holm)}).`
  : `After Holm's correction over the study's ${data.n_primary} primary comparisons, the difference is not detectable (adjusted ${pEq(d.holm)}).`);

/** A family's name inside a sentence. */
export const NAME = { gaussian: "the Gaussian matrix", rademacher: "Eq. 3", achlioptas3: "Eq. 4", verysparse: "the very sparse matrix", onepercol: "the one-per-column matrix" };
const cap = (s) => s[0].toUpperCase() + s.slice(1);

/** How often a row was below, tied with, and above the Gaussian matrix, over the draws. */
export function sayCounts(g, N) {
  const b = count(g.lower, N), t = count(g.equal, N), a = count(g.higher, N);
  return t === 0 ? `It was below the Gaussian matrix in ${b} of the ${N} draws and above in ${a}, with no ties.`
    : `It was below the Gaussian matrix in ${b} of the ${N} draws, tied with it in ${t}, and above in ${a}.`;
}

export function sayFamily(data, ds, row, k, v, g, e3) {
  const N = data.n_draws, name = NAME[row];
  let s = `On ${ds.label}, at k = ${k}, ${name} scores ${pct1(v[0])}% ${metricWord(ds)}. ${cap(name)} minus the Gaussian matrix is ${pts1(g.mean)} points on the same draws (${ci(g)}). `;
  s += holmSentence(data, g);
  if (g.equiv) s += " Its interval lies inside ±2 points, the study's margin for calling the two equivalent.";
  s += " " + sayCounts(g, N);
  if (e3) {
    s += ` Eq. 4 minus Eq. 3 is ${pts1(e3.mean)} points (${ci(e3)}; adjusted ${pEq(e3.holm)}, ${e3.holm < ALPHA ? "surviving the correction" : "not detectable after the correction"}).`;
  }
  return s;
}

export function sayGaussian(ds, k, v, p, f) {
  return `On ${ds.label}, at k = ${k}, the Gaussian matrix scores ${pct1(v[0])}% ${metricWord(ds)} (95% interval ${pct1(v[1])} to ${pct1(v[2])}). It is the dense reference that every other row is compared with, draw by draw. Gaussian minus PCA is ${pts1(p.mean)} points (${ci(p)}), and Gaussian minus all features is ${pts1(f.mean)} (${ci(f)}).`;
}

export function sayPca(ds, k, v, g) {
  return `On ${ds.label}, at k = ${k}, PCA scores ${pct1(v[0])}% ${metricWord(ds)}. PCA minus the Gaussian matrix is ${pts1(g.mean)} points on the same draws (${ci(g)}); the study records it as Gaussian minus PCA, ${pts1(-g.mean)}. PCA chooses its ${k === 1 ? "direction" : `${k} directions`} from each draw's training rows, and a random matrix never looks at the data.`;
}

export function sayFull(ds, v) {
  return `On ${ds.label}, the SVM on all ${ds.d} features scores ${pct1(v[0])}% ${metricWord(ds)} (95% interval ${pct1(v[1])} to ${pct1(v[2])}). Nothing is projected, so the study ran it once per draw, and it has one value for every k. To compare the Gaussian matrix with it at each k, ink the table by difference.`;
}

export function sayFullAt(ds, k, v, g) {
  return `On ${ds.label}, all ${ds.d} features score ${pct1(v[0])}% ${metricWord(ds)}. All features minus the Gaussian matrix at k = ${k} is ${pts1(g.mean)} points on the same draws (${ci(g)}); the study records it as Gaussian minus all features, ${pts1(-g.mean)}.`;
}

/** The block's name for a screen reader: what it is and what it shows. */
export function blockLabel(data, ds, b) {
  const label = famLabel(data, b.row);
  const where = b.k === null ? "one value for every k" : `k = ${b.k}`;
  if (b.kind === "diff") return `${label}, ${where}: ${b.text} points against the Gaussian matrix${b.marks.includes("*") ? ", surviving Holm's correction" : ""}${b.marks.includes("†") ? ", inside ±2 points" : ""}.`;
  return `${label}, ${where}: ${b.text}% ${metricWord(ds)}${b.marks.includes("*") ? "; differs from the Gaussian matrix after Holm's correction" : ""}${b.marks.includes("†") ? "; inside ±2 points of it" : ""}.`;
}

/**
 * What the table is, for the line above it and for a screen reader. In the difference ink, the
 * Gaussian row is the reference and keeps its own score (blocks() gives it kind "reference").
 */
export function sayForme(data, ds, mode) {
  const ks = list(ds.ks.map(String));
  const what = mode === "score"
    ? `Mean test ${metricWord(ds)} over the study's ${data.n_draws} draws`
    : `Each row minus the Gaussian matrix, in points, paired over the study's ${data.n_draws} draws (the Gaussian row keeps its own score)`;
  return `${ds.label}, d = ${ds.d}. ${what}, at the target sizes the study ran: k = ${ks}.`;
}

/** The ink key's words. */
export function sayKey(ds, mode) {
  if (mode === "score") {
    const floor = ds.floor.kind === "chance" ? `chance balanced accuracy, 1 in ${ds.floor.classes}` : "the share of the majority class";
    return `Each tint is a tenth of the way from ${pct1(ds.floor.value)}% (${floor}) to 100%.`;
  }
  return `Each tint is one more whole point of difference, up to ${Math.round(100 * DIFF_FULL)} or more: vermilion below the Gaussian matrix, blue above. The Gaussian row, uninked, is the reference.`;
}

/**
 * The ink key: a strip of the tints in order, and labels at fractions of its width. For the score,
 * the labels are the edges between tints; for a difference, the middles of the tints they name.
 */
export function keySteps(ds, mode) {
  if (mode === "score") {
    const f = ds.floor.value;
    const patches = SCORE_TINTS.map((_, i) => ({ side: "ink", level: i, print: inks("ink", i) }));
    const labels = [0, 2, 4, 6, 8, 10].map((i) => ({ at: i / 10, text: pct1(f + (i / 10) * (1 - f)) }));
    return { patches, labels };
  }
  const full = Math.round(100 * DIFF_FULL), patches = [];
  for (let L = full; L >= 1; L--) patches.push({ side: "neg", level: L, print: inks("neg", L) });
  patches.push({ side: "zero", level: 0, print: inks("zero", 0) });
  for (let L = 1; L <= full; L++) patches.push({ side: "pos", level: L, print: inks("pos", L) });
  const n = patches.length, mid = (i) => (i + 0.5) / n;
  const labels = [[0, `${MINUS}${full}`], [full / 2, `${MINUS}${full / 2}`], [full, "0"], [full * 1.5, `+${full / 2}`], [2 * full, `+${full}`]]
    .map(([i, text]) => ({ at: mid(i), text }));
  return { patches, labels };
}
