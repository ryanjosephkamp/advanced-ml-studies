// How alike the characters are (chapter II): the pure model behind the quilt of overlaps. No DOM;
// the view and the check script import the same functions, so what the page shows is what was
// checked.
//
// The data are the study's 18 bitmaps (data/characters.csv). In the paper's ±1 coding, with
// xi = 2 * bit - 1, the overlap of characters a and b is (1/144) * xi_a . xi_b: each pixel where they
// agree adds 1 and each where they differ takes 1 away, so the overlap is (agree - differ) / 144.
// Every value here is computed from the bits, as an exact fraction where the paper's rounding needs
// one.

export const SIDE = 12;
export const N = SIDE * SIDE;
export const STITCH = 8; // pairs that differ in this many pixels or fewer get a stitched outline
export const TOUR_MS = 2600; // on its own, the quilt walks the pairs from most alike to least, one every 2.6 s
export const MOVE_MS = 650; // rows and columns slide to a new order in 0.65 s

const bitsOf = (s) => Uint8Array.from(s, (ch) => (ch === "1" ? 1 : 0));
export const onCount = (b) => b.reduce((s, x) => s + x, 0);
export const hamming = (a, b) => a.reduce((s, x, i) => s + (x !== b[i] ? 1 : 0), 0);

/** The plate's data, shaped: the bitmaps and the pixels each pair differs in, computed here. */
export function shape(data) {
  const chars = data.characters.map(bitsOf);
  const differ = chars.map((a) => chars.map((b) => hamming(a, b)));
  return { n: chars.length, chars, differ, on: chars.map(onCount) };
}

// ------------------------------------------------------------------ rounding as the paper rounds (half up)

/** num / den rounded half up to `places` decimals, on integers, so no float can tip a tie. */
export function halfUp(num, den, places) {
  if (den < 0) { num = -num; den = -den; }
  const neg = num < 0, a = Math.abs(num), k = 10 ** places;
  const q = Math.floor((2 * a * k + den) / (2 * den));
  const s = String(q).padStart(places + 1, "0");
  const out = places ? `${s.slice(0, -places)}.${s.slice(-places)}` : s;
  return neg && q ? `−${out}` : out;
}
/** A float rounded half up, the way paper/make_numbers.py does it (first to six places, then half up). */
export function halfUpFloat(x, places) {
  const s = Math.abs(x).toFixed(6), [i, f] = s.split(".");
  const digits = BigInt(i + f), drop = 10n ** BigInt(6 - places);
  const q = (digits * 2n + drop) / (2n * drop);
  const t = q.toString().padStart(places + 1, "0");
  const out = places ? `${t.slice(0, -places)}.${t.slice(-places)}` : t;
  return x < 0 && q ? `−${out}` : out;
}
export const commas = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ",");

// ------------------------------------------------------------------ pairs

/** One pair: the pixels they differ and agree in, and the overlap (agree - differ) / 144. */
export function pair(S, a, b) {
  const d = S.differ[a][b], agree = N - d, num = agree - d;
  return { a, b, d, agree, num, value: num / N, text: halfUp(num, N, 3) };
}
/** All pairs a < b, in the order NumPy's upper triangle lists them. */
export function pairs(S) {
  const out = [];
  for (let a = 0; a < S.n; a++) for (let b = a + 1; b < S.n; b++) out.push(pair(S, a, b));
  return out;
}
/** The pixels where two characters differ, in reading order. */
export function diffPixels(S, a, b) {
  const out = [];
  for (let i = 0; i < N; i++) if (S.chars[a][i] !== S.chars[b][i]) out.push(i);
  return out;
}

/** Over all pairs: what the paper prints about them (paper/sections/data.tex). */
export function stats(S) {
  const ps = pairs(S), P = ps.length;
  const sumNum = ps.reduce((s, p) => s + p.num, 0), sumD = ps.reduce((s, p) => s + p.d, 0);
  const mean = sumNum / (N * P);
  const sd = Math.sqrt(ps.reduce((s, p) => s + (p.value - mean) ** 2, 0) / P); // over the pairs themselves (ddof 0)
  const dmin = Math.min(...ps.map((p) => p.d)), dmax = Math.max(...ps.map((p) => p.d));
  const closest = ps.find((p) => p.d === dmin), farthest = ps.find((p) => p.d === dmax);
  const on = S.on.reduce((s, x) => s + x, 0), tot = S.n * N;
  return {
    pairs: P, sumNum, mean, meanText: halfUp(sumNum, N * P, 3), sd, sdText: halfUpFloat(sd, 3),
    min: farthest, max: closest, minText: farthest.text, maxText: closest.text,
    hamMean: halfUp(sumD, P, 1), hamMin: dmin, hamMax: dmax, closest, farthest,
    on, total: tot, density: halfUp(on, tot, 4), densityPct: halfUp(100 * on, tot, 2),
    // independent bits at the characters' density: expected overlap (2d - 1)^2, from the exact density
    indep: halfUp((2 * on - tot) ** 2, tot * tot, 3), indepValue: ((2 * on - tot) / tot) ** 2,
  };
}

/** The stitched pairs (differ in STITCH pixels or fewer), closest first, and the next closest after them. */
export function stitched(S) {
  const ps = pairs(S).sort((p, q) => p.d - q.d || p.a - q.a || p.b - q.b);
  const inside = ps.filter((p) => p.d <= STITCH), next = ps.find((p) => p.d > STITCH);
  return { pairs: inside, next };
}
export const isStitched = (S, a, b) => a !== b && S.differ[a][b] <= STITCH;

/** Where a pair stands among all pairs: how many differ in fewer pixels, and how many in as few. */
export function rank(S, a, b) {
  const d = S.differ[a][b], ps = pairs(S);
  return { closer: ps.filter((p) => p.d < d).length, same: ps.filter((p) => p.d === d).length - 1, of: ps.length };
}

/** The tour: every pair from most alike to least (ties in file order). */
export function tour(S) {
  return pairs(S).sort((p, q) => p.d - q.d || p.a - q.a || p.b - q.b).map((p) => [p.a, p.b]);
}

// ------------------------------------------------------------------ the weave

/**
 * Thread width, as a share of the thread spacing, for an overlap v: a plain weave of width w covers
 * 1 - (1 - w)^2 of its square (each direction leaves (1 - w) open), so w = 1 - sqrt(1 - v) covers
 * exactly v. No overlap here is negative; one at or below 0 would weave no thread.
 */
export const weave = (v) => (v <= 0 ? 0 : v >= 1 ? 1 : 1 - Math.sqrt(1 - v));
export const coverage = (w) => 1 - (1 - w) ** 2;
/** Threads each way in a square of `px` CSS pixels: wider squares get finer cloth. */
export const threads = (px) => (px >= 22 ? 4 : 3);

// ------------------------------------------------------------------ orders

/**
 * Rows and columns in file order, or in likeness order: a chain that starts with the closest pair and,
 * one character at a time, adds whichever unplaced character is most like either end of the chain
 * (fewest differing pixels; on a tie, the lower number, and the end the chain grows from first).
 */
export function order(S, kind) {
  const ids = Array.from({ length: S.n }, (_, i) => i);
  if (kind !== "likeness") return ids;
  const c = stats(S).closest;
  const chain = [c.a, c.b], left = new Set(ids.filter((i) => i !== c.a && i !== c.b));
  while (left.size) {
    let best = null;
    for (const j of [...left].sort((x, y) => x - y)) {
      for (const end of ["tail", "head"]) {
        const d = S.differ[j][end === "tail" ? chain[chain.length - 1] : chain[0]];
        if (!best || d < best.d) best = { j, end, d };
      }
    }
    if (best.end === "tail") chain.push(best.j); else chain.unshift(best.j);
    left.delete(best.j);
  }
  return chain;
}

/** Arrow keys on the quilt: one cell across or down in the current order, hopping over the diagonal. */
export function step(ord, a, b, dRow, dCol) {
  const n = ord.length;
  let r = ord.indexOf(a), c = ord.indexOf(b);
  const r0 = r, c0 = c;
  r += dRow; c += dCol;
  if (r === c) { r += dRow; c += dCol; }
  if (r < 0 || c < 0 || r >= n || c >= n) return { a, b, moved: false, r: r0, c: c0 };
  return { a: ord[r], b: ord[c], moved: true };
}

// ------------------------------------------------------------------ the words (drafts, as functions of numbers)

const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;

/** One pair's reading. */
export function sayPair(p, rk, st) {
  const head = `Characters ${p.a} and ${p.b} differ in ${plural(p.d, "pixel", "pixels")} of 144 and agree in the other ${p.agree}, so their overlap is (${p.agree} − ${p.d}) / 144 = ${p.text}.`;
  let where;
  if (rk.closer === 0 && rk.same === 0) where = `No other pair of the ${rk.of} is as close.`;
  else if (rk.closer === 0) where = `No pair of the ${rk.of} is closer, and ${plural(rk.same, "other is", "others are")} as close.`;
  else if (rk.same === 0) where = `Of the ${rk.of} pairs, ${rk.closer} ${rk.closer === 1 ? "is" : "are"} closer.`;
  else where = `Of the ${rk.of} pairs, ${rk.closer} ${rk.closer === 1 ? "is" : "are"} closer and ${plural(rk.same, "other is", "others are")} as close.`;
  // compared exactly: p.num / 144 against the mean, sumNum / (144 * pairs)
  const cmp = p.num * st.pairs - st.sumNum;
  const side = cmp > 0 ? "above" : cmp < 0 ? "below" : "at";
  return `${head} ${where} That is ${side} the mean overlap, ${st.meanText}.`;
}

/** The quilt as a whole, as the paper states it. */
export function sayStats(st) {
  return `Over all ${st.pairs} pairs, the overlap averages ${st.meanText} (standard deviation ${st.sdText}) and runs from ${st.minText} to ${st.maxText}. Two characters differ in ${st.hamMean} pixels on average, from ${st.hamMin} to ${st.hamMax}.`;
}

/** What density alone would give. */
export function sayIndep(st) {
  return `Independent pixels, each on with probability ${st.density} (the characters' density, ${st.densityPct}%), would have an expected overlap of (2 × ${st.density} − 1)² = ${st.indep}, so density accounts for much of the ${st.meanText}.`;
}

/** The stitched outlines. */
export function sayStitch(sti) {
  const list = sti.pairs.map((p, i) => `${p.a} and ${p.b} (${p.d}${i ? "" : " pixels"})`);
  const joined = list.length > 1 ? `${list.slice(0, -1).join(", ")}, and ${list.at(-1)}` : list[0];
  return `The ${sti.pairs.length} stitched pairs differ in ${STITCH} pixels or fewer: ${joined}. The next closest pair differs in ${sti.next.d}.`;
}

/** The order in use. */
export function sayOrder(kind) {
  return kind === "likeness"
    ? "Likeness order: starting from the closest pair, each next character is the one most like either end of the row so far. Every cell keeps its value; only the places change."
    : "File order: rows and columns run 0 to 17 down and across, as in characters.csv.";
}

/** How to read the cloth. */
export const WEAVE = "The denser the weave, the more alike the pair: thread width is set so the threads cover a share of each square equal to the overlap. The diagonal, each character with itself, is left blank.";
