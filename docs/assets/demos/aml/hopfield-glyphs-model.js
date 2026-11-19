// The eighteen characters (chapter II): the pure model behind the type specimen sheet. No DOM; the
// view and the check script import the same functions, so what the page shows is what was checked.
//
// The data are the study's own files: data/characters.csv (18 rows of 144 bits, row-major 12 by 12)
// and the two query files, whose row k is a noisy copy of character k. A bit of 1 is "on" (the
// study's density counts the 1s); the strokes of the characters are the 0s. Every count, share, and
// sentence below is computed here from those bits.

export const SIDE = 12;
export const N = SIDE * SIDE;
export const COLS = 6; // the sheet: 6 across, 3 down, in file order
export const TOUR_MS = 3400; // the sheet proofreads itself, one character every 3.4 s, until touched
export const HALF = SIDE / 2; // a proofreader puts each mark in the nearer margin

const bitsOf = (s) => Uint8Array.from(s, (ch) => (ch === "1" ? 1 : 0));

/** The plate's data, shaped for drawing: bits, counts, and every flip of every query. */
export function shape(data) {
  const chars = data.characters.map(bitsOf);
  const on = chars.map(onCount);
  const queries = {};
  for (const q of data.queries) {
    const rows = q.rows.map(bitsOf);
    const flips = rows.map((r, k) => flipsOf(chars[k], r));
    queries[q.id] = { id: q.id, name: q.name, file: q.file, seed: q.seed, percent: q.percent, rows, flips };
  }
  return { n: chars.length, chars, on, queries, ids: data.queries.map((q) => q.id) };
}

export const onCount = (b) => b.reduce((s, x) => s + x, 0);

/** The pixels where query q differs from character c, in reading order; `to` is the query's bit. */
export function flipsOf(c, q) {
  const out = [];
  for (let i = 0; i < c.length; i++) if (c[i] !== q[i]) out.push({ i, r: Math.floor(i / SIDE), c: i % SIDE, to: q[i] });
  return out;
}

/** Whether a pixel prints in ink: "strokes" inks the off pixels (the letterforms), "on" inks the on pixels. */
export const inked = (bit, print) => (print === "on" ? bit === 1 : bit === 0);

/** A proofreader's mark for one flip: strike out ink the query added, caret where it took ink away. */
export const markOf = (flip, print) => (inked(flip.to, print) ? "strike" : "caret");

/** The marks repeated in the margins, row by row, each in the nearer margin, in reading order. */
export function margins(flips, print) {
  const left = Array.from({ length: SIDE }, () => []), right = Array.from({ length: SIDE }, () => []);
  for (const f of flips) (f.c < HALF ? left : right)[f.r].push(markOf(f, print));
  return { left, right };
}

/** Horizontal runs of inked pixels, [row, first column, length], for drawing without seams. */
export function runs(bits, print) {
  const out = [];
  for (let r = 0; r < SIDE; r++) {
    let c = 0;
    while (c < SIDE) {
      if (!inked(bits[r * SIDE + c], print)) { c++; continue; }
      const c0 = c;
      while (c < SIDE && inked(bits[r * SIDE + c], print)) c++;
      out.push([r, c0, c - c0]);
    }
  }
  return out;
}

// ------------------------------------------------------------------ numbers, as the paper writes them

/** num / den rounded half up to `places` decimals, from integers, so no float can tip a tie. */
export function halfUp(num, den, places) {
  if (den < 0) { num = -num; den = -den; }
  const neg = num < 0, a = Math.abs(num), k = 10 ** places;
  const q = Math.floor((2 * a * k + den) / (2 * den)); // half up on the magnitude
  const s = String(q).padStart(places + 1, "0");
  const out = places ? `${s.slice(0, -places)}.${s.slice(-places)}` : s;
  return neg && q ? `−${out}` : out;
}
/** A share of `den` in percent, half up. */
export const pct = (num, den, places) => halfUp(100 * num, den, places);
export const commas = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ",");

/** A query file's flips: per row, and over the file. */
export function fileStats(S, id) {
  const q = S.queries[id];
  const per = q.flips.map((f) => f.length);
  const sum = per.reduce((s, x) => s + x, 0), rows = per.length;
  const off = q.flips.reduce((s, f) => s + f.filter((x) => x.to === 0).length, 0);
  return {
    id, name: q.name, seed: q.seed, percent: q.percent, per, sum, rows, off, onFlips: sum - off,
    min: Math.min(...per), max: Math.max(...per), same: per.every((x) => x === per[0]),
    mean: halfUp(sum, rows, 2), share: pct(sum, rows * N, 2),
  };
}

/** The whole sheet's pixels: how many are on, and the range per character. */
export function sheetStats(S) {
  const total = S.n * N, on = S.on.reduce((s, x) => s + x, 0);
  return {
    total, on, share: pct(on, total, 2), min: Math.min(...S.on), max: Math.max(...S.on),
    minShare: pct(Math.min(...S.on), N, 1), maxShare: pct(Math.max(...S.on), N, 1),
    allOn: Array.from({ length: N }, (_, i) => S.chars.every((c) => c[i] === 1)).filter(Boolean).length,
  };
}

/** How far query k of a file is from its own character and from the nearest others (every one at that distance, lowest index first). */
export function nearest(S, id, k) {
  const q = S.queries[id].rows[k];
  const d = S.chars.map((c) => c.reduce((s, x, i) => s + (x !== q[i] ? 1 : 0), 0));
  let other = -1;
  for (let j = 0; j < S.n; j++) if (j !== k && (other < 0 || d[j] < d[other])) other = j;
  const all = d.map((x, j) => j).filter((j) => j !== k && d[j] === d[other]);
  return { own: d[k], other, otherD: d[other], ties: all.length - 1, all, d };
}

/** Everything the proof shows for character k under one query file (or none) and one print. */
export function proof(S, k, id, print) {
  const flips = id ? S.queries[id].flips[k] : [];
  const off = flips.filter((f) => f.to === 0).length;
  const strikes = flips.filter((f) => markOf(f, print) === "strike").length;
  return {
    k, id, print, on: S.on[k], flips, n: flips.length, off, onFlips: flips.length - off,
    strikes, carets: flips.length - strikes, margins: margins(flips, print),
    near: id ? nearest(S, id, k) : null,
  };
}

/** The sheet's arrow keys: one step across or down, stopping at the edges. */
export function step(k, dx, dy, n = 18) {
  const rows = Math.ceil(n / COLS), r = Math.floor(k / COLS), c = k % COLS;
  const r2 = Math.min(Math.max(r + dy, 0), rows - 1), c2 = Math.min(Math.max(c + dx, 0), COLS - 1);
  return Math.min(r2 * COLS + c2, n - 1);
}

// ------------------------------------------------------------------ the words (drafts, as functions of numbers)

const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;
const fileName = (st) => (st.id === "test" ? "the test file" : `the ${st.name}`);

/** The key to the ink. */
export function sayInk(print) {
  return print === "on"
    ? "Ink marks the on pixels (bit 1), as the network stores them; the letterforms show as paper."
    : "Ink marks the off pixels (bit 0), which draw the letterforms; the paper around them is the on pixels (bit 1).";
}

/** The sheet as a whole. */
export function saySheet(sh) {
  return `${commas(sh.on)} of the ${commas(sh.total)} pixels are on (${sh.share}%). A character has ${sh.min} to ${sh.max} of its 144 on (${sh.minShare}% to ${sh.maxShare}%), and ${sh.allOn} pixels are on in all 18.`;
}

/** One query file, over its 18 rows. */
export function sayFile(st) {
  const share = `${st.share}% of its ${commas(st.rows * N)} pixels`;
  const who = st.id === "test" ? "The test file" : `The ${st.name}${st.seed !== undefined ? ` (made with seed ${st.seed})` : ""}`;
  if (st.same) return `${who} flips exactly ${st.per[0]} pixels in every row (${share}).`;
  return `${who} flips ${st.min} to ${st.max} pixels per row, ${st.mean} on average (${share}).`;
}

/** The proof's reading for one character, clean or under a query file. */
export function sayProof(p, st) {
  const head = `Character ${p.k} (counting from 0) has ${p.on} of its 144 pixels on (${pct(p.on, N, 1)}%).`;
  if (!p.id) return `${head} This is the clean copy, row ${p.k} of characters.csv.`;
  const what = `Row ${p.k} of ${fileName(st)} is character ${p.k} with ${plural(p.n, "pixel", "pixels")} flipped (${pct(p.n, N, 2)}%): ${p.off} turned off and ${p.onFlips} turned on.`;
  const nr = p.near, many = nr.all.length > 1;
  const who = `character${many ? "s" : ""} ${nr.all.length > 2 ? `${nr.all.slice(0, -1).join(", ")}, and ${nr.all.at(-1)}` : nr.all.join(" and ")}`;
  let near;
  if (nr.otherD > nr.own) near = `It is still nearer its own character (${nr.own} pixels away) than any other; the nearest ${many ? `others, ${who}, are` : `other, ${who}, is`} ${nr.otherD} away.`;
  else if (nr.otherD === nr.own) near = `It is as near ${who} as its own (${nr.own} pixels away).`;
  else near = `It is nearer ${who} (${nr.otherD} pixels away) than its own (${nr.own}).`;
  return `${what} ${near}`;
}

/** What the marks mean, for the print in use. */
export function sayMarks(print) {
  const added = print === "on" ? "turned on" : "turned off";
  const taken = print === "on" ? "turned off" : "turned on";
  return `Struck through: ink the query added (a pixel ${added}). Caret: ink it took away (a pixel ${taken}). Each mark is repeated in the nearer margin, on its row, as a proofreader does.`;
}

/** The tally under the proof. */
export function sayTally(p) {
  if (!p.id) return "No query laid over it.";
  return `${plural(p.strikes, "strike", "strikes")}, ${plural(p.carets, "caret", "carets")}: ${plural(p.n, "flip", "flips")} in all.`;
}

/** A glyph's name for screen readers. */
export const sayGlyph = (k, on, flips) =>
  `Character ${k}: ${on} of 144 pixels on${flips === undefined ? "" : `; the query flips ${plural(flips, "pixel", "pixels")}`}.`;
