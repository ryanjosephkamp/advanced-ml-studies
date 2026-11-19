// The model behind Chapter III's live plate, "Draw a matrix": the study's five random-matrix families
// exactly as study/common.py builds them, the projection E = X R / √k of the 150 Iris flowers after
// the study's min-max scaling, what a drawn matrix drops, the theory of the two zero events, the exact
// binomial band a run of draws should land in, and every sentence the plate shows, as functions of
// numbers. Pure functions, no DOM: projection-lottery.js draws them, and
// tools/aml/projection-lottery.check.mjs checks them against the study's own code and files.

// ---- random numbers -------------------------------------------------------------------------------
// sfc32, seeded through splitmix32, for uniforms on [0, 1) with 53 random bits; Box-Muller for
// standard normals. Card j of seed S has its own stream, so any card can be drawn again on its own.

function splitmix32(a) {
  return () => {
    a = (a + 0x9e3779b9) | 0;
    let t = a ^ (a >>> 16);
    t = Math.imul(t, 0x21f0aaad);
    t ^= t >>> 15;
    t = Math.imul(t, 0x735a2d97);
    return (t ^ (t >>> 15)) >>> 0;
  };
}

export class Rng {
  constructor(seed) {
    const sm = splitmix32(seed >>> 0);
    this.a = sm(); this.b = sm(); this.c = sm(); this.d = sm();
    for (let i = 0; i < 12; i++) this.u32();
    this.spare = null;
  }
  u32() {
    const t = (((this.a + this.b) | 0) + this.d) | 0;
    this.d = (this.d + 1) | 0;
    this.a = this.b ^ (this.b >>> 9);
    this.b = (this.c + (this.c << 3)) | 0;
    this.c = (this.c << 21) | (this.c >>> 11);
    this.c = (this.c + t) | 0;
    return t >>> 0;
  }
  /** Uniform on [0, 1). */
  next() {
    return ((this.u32() >>> 5) * 67108864 + (this.u32() >>> 6)) / 9007199254740992;
  }
  /** Standard normal. */
  gauss() {
    if (this.spare !== null) { const s = this.spare; this.spare = null; return s; }
    let u = 0;
    while (u === 0) u = this.next();
    const v = this.next();
    const r = Math.sqrt(-2 * Math.log(u));
    this.spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  }
}

/** The stream for card j of seed S. */
export function cardSeed(seed, j) {
  let h = (Math.imul(seed >>> 0, 0x9e3779b1) ^ Math.imul((j >>> 0) + 0x632be5ab, 0x85ebca77)) >>> 0;
  h ^= h >>> 16; h = Math.imul(h, 0x7feb352d);
  h ^= h >>> 15; h = Math.imul(h, 0x846ca68b);
  return (h ^ (h >>> 16)) >>> 0;
}

/** A fresh seed for a visit, shown on the page so its draws can be repeated with ?seed=. */
export const freshSeed = () => 1 + Math.floor(Math.random() * 999999);
/**
 * The seed in the address (?seed=), as the page uses it: decimal digits only, read exactly at any
 * length and taken modulo 2^32 (?seed=9007199254740993 is seed 1), or null for anything else, such as
 * a sign, a point, or letters (?seed=7abc). The link the page offers, seedQuery(seed), reads back as
 * the same seed (?seed=4294967296 is seed 0, and so is its link, ?seed=0).
 */
export function seedFromQuery(search) {
  const v = new URLSearchParams(search).get("seed");
  return v !== null && /^[0-9]+$/.test(v) ? Number(BigInt(v) % 4294967296n) : null;
}
export const seedQuery = (seed) => `?seed=${seed >>> 0}`;
/**
 * The seed line, once a card is on the plate. Card j of a seed is the same G and U on every visit, so
 * the link repeats the cards by number. The matrix shown also depends on the family and k, and a new
 * visit starts at Eq. 4 and the first k, so the line says that the same family and k punch the same
 * matrix. The cards the plate draws on its own also take numbers, and how many it draws depends on the
 * visit (with a seed in the address, card 1 at most), so the line promises numbers, not the order of the
 * reader's draws, and names this visit's own cards: always cards 1 to own. When the address already is
 * seedQuery(seed) (atLink), following a link to it would only move to the plate, so the line says to
 * reload the page instead.
 */
export function saySeed(seed, card, own, atLink = false) {
  const mine = own === 0 ? "none" : own === 1 ? "card 1" : own === 2 ? "cards 1 and 2" : `cards 1 to ${own}`;
  const go = atLink ? `Reload the page (its address has ${seedQuery(seed)})` : `Open the page with ${seedQuery(seed)}`;
  return `Seed ${seed}, card ${card}. ${go} to draw the same cards by number from card 1 again: card ${card} there has the same G and U as card ${card} here, so the same family and k punch the same matrix. Cards the plate draws on its own also take numbers (here, ${mine}), so your draws can get other numbers there.`;
}

/**
 * One card: the Gaussian matrix G and the uniform matrix U, both d x kmax, as study/common.py's
 * base_matrices draws them (G first, then U). Target size k uses their first k columns, so the
 * projections are nested across k, as in the study.
 */
export function drawCard(seed, j, d = 4, kmax = 3) {
  const rng = new Rng(cardSeed(seed, j));
  const G = Array.from({ length: d }, () => Array.from({ length: kmax }, () => rng.gauss()));
  const U = Array.from({ length: d }, () => Array.from({ length: kmax }, () => rng.next()));
  return { G, U };
}

// ---- the study's matrices and projection ---------------------------------------------------------

/**
 * The d x k matrix R for one family (study/common.py, projection_matrix): the Gaussian matrix itself,
 * or sqrt(s) * sign(G) * [U < 1/s] for the sparse families (s = 1 is Eq. 3, s = 3 Eq. 4).
 */
export function projectionMatrix(fam, G, U, k) {
  if (fam.id === "gaussian") return G.map((r) => r.slice(0, k));
  const c = Math.sqrt(fam.s), t = 1.0 / fam.s;
  return G.map((r, i) => r.slice(0, k).map((g, j) => c * Math.sign(g) * (U[i][j] < t ? 1 : 0)));
}

/** Each column scaled to [0, 1] as scikit-learn's MinMaxScaler does it: X * scale + min. */
export function minmax(X) {
  const d = X[0].length, scale = [], add = [];
  for (let j = 0; j < d; j++) {
    let lo = Infinity, hi = -Infinity;
    for (const r of X) { if (r[j] < lo) lo = r[j]; if (r[j] > hi) hi = r[j]; }
    const range = hi - lo, sc = 1 / (range === 0 ? 1 : range);
    scale.push(sc); add.push(0 - lo * sc);
  }
  return X.map((r) => r.map((v, j) => v * scale[j] + add[j]));
}

/** E = X R / sqrt(k) (study/common.py, project). */
export function project(Xs, R) {
  const d = R.length, k = R[0].length, rk = Math.sqrt(k);
  return Xs.map((x) => {
    const e = new Array(k);
    for (let j = 0; j < k; j++) {
      let s = 0;
      for (let i = 0; i < d; i++) s += x[i] * R[i][j];
      e[j] = s / rk;
    }
    return e;
  });
}

// ---- what a matrix drops -------------------------------------------------------------------------

export const zeroRows = (R) => R.map((r, i) => (r.every((v) => v === 0) ? i : -1)).filter((i) => i >= 0);
export const zeroCols = (R) => R[0].map((_, j) => (R.every((r) => r[j] === 0) ? j : -1)).filter((j) => j >= 0);

function combos(n, m) {
  const out = [];
  const go = (start, acc) => {
    if (acc.length === m) { out.push(acc.slice()); return; }
    for (let i = start; i < n; i++) { acc.push(i); go(i + 1, acc); acc.pop(); }
  };
  go(0, []);
  return out;
}
function det(M) {
  if (M.length === 1) return M[0][0];
  if (M.length === 2) return M[0][0] * M[1][1] - M[0][1] * M[1][0];
  return M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
    + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
}

/** The rank of R (k <= 3), by its minors. The cloud of projected flowers spans exactly this many dimensions. */
export function rank(R, tol = 1e-9) {
  const d = R.length, k = R[0].length;
  let r = 0;
  for (let m = 1; m <= Math.min(d, k); m++) {
    let found = false;
    for (const rows of combos(d, m)) {
      for (const cols of combos(k, m)) {
        if (Math.abs(det(rows.map((i) => cols.map((j) => R[i][j])))) > tol) { found = true; break; }
      }
      if (found) break;
    }
    if (!found) break;
    r = m;
  }
  return r;
}

/** Everything the sentences say about one matrix. */
export function anatomy(R, petalRows = [2, 3]) {
  const zr = zeroRows(R), zc = zeroCols(R);
  return { k: R[0].length, zeroRows: zr, zeroCols: zc, rank: rank(R), bothPetal: petalRows.every((i) => zr.includes(i)) };
}

// ---- theory and the binomial band ---------------------------------------------------------------

/**
 * The chance that a draw has an all-zero column, and that both petal rows are all zero. An entry is
 * zero with probability q = (s - 1)/s, independently, so a column of d entries is all zero with
 * probability q^d and a row of k entries with probability q^k. The Gaussian (s = null) and Eq. 3
 * (s = 1) have no zero entries.
 */
export function theory(s, d, k) {
  if (s == null || s <= 1) return { zero: 0, both: 0, q: 0 };
  const q = (s - 1) / s;
  return { zero: 1 - (1 - q ** d) ** k, both: q ** (2 * k), q };
}

const SUP = { 1: "", 2: "²", 3: "³", 4: "⁴", 6: "⁶", 8: "⁸" };
/** The two formulas, written out for the family's s: "1 − (1 − (2/3)⁴)²" and "(2/3)⁴". */
export function theoryFormulas(s, d, k) {
  if (s == null || s <= 1) return null;
  const q = Number.isInteger(s) ? `(${s - 1}/${s})` : `(1 − 1/${s})`;
  return { zero: `1 − (1 − ${q}${SUP[d]})${SUP[k]}`, both: `${q}${SUP[2 * k]}` };
}

/**
 * The central 95% range of the number of hits in n draws with chance p, from the binomial law
 * itself: the smallest x with P(X <= x) >= 0.025 and the smallest with P(X <= x) >= 0.975.
 */
export function binomBand(n, p, lo = 0.025, hi = 0.975) {
  if (!(n > 0)) return null;
  if (p <= 0) return [0, 0];
  if (p >= 1) return [n, n];
  let lp = n * Math.log1p(-p);
  const lr = Math.log(p) - Math.log1p(-p);
  let cdf = 0, a = null;
  for (let x = 0; x <= n; x++) {
    if (x > 0) lp += Math.log((n - x + 1) / x) + lr;
    cdf += Math.exp(lp);
    if (a === null && cdf >= lo) a = x;
    if (cdf >= hi) return [a, x];
  }
  return [a ?? n, n];
}

// ---- numbers as the paper prints them -------------------------------------------------------------

/** x rounded half up to a number of decimals, as paper/make_numbers.py does (after rounding to 8 places). */
export function halfUp(x, places) {
  const neg = x < 0;
  const [ip, fp] = Math.abs(x).toFixed(8).split(".");
  const unit = 10n ** BigInt(8 - places);
  const all = BigInt(ip + fp);
  let q = all / unit;
  if ((all % unit) * 2n >= unit) q += 1n;
  const str = q.toString().padStart(places + 1, "0");
  const out = places ? `${str.slice(0, -places)}.${str.slice(-places)}` : str;
  return (neg && q !== 0n ? "−" : "") + out;
}
/** A share in [0, 1] as a percentage with one decimal, half up. */
export const pct1 = (share) => halfUp(100 * share, 1);

// ---- the frame's ticks ----------------------------------------------------------------------------

export function niceStep(raw) {
  const p = 10 ** Math.floor(Math.log10(raw));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= raw * (1 - 1e-9)) return m * p;
  return 10 * p;
}
/**
 * One shared scale for every axis, so the cloud keeps its shape: each axis is centered on its own
 * middle, and all have the same half width h (the largest half range, padded).
 */
export function frameFor(cols, pad = 1.1) {
  const lo = cols.map((c) => Math.min(...c)), hi = cols.map((c) => Math.max(...c));
  const half = Math.max(...lo.map((l, j) => (hi[j] - l) / 2));
  const h = half > 1e-9 ? half * pad : 0.5;
  const step = niceStep((2 * h) / 3.2);
  const center = lo.map((l, j) => (l + hi[j]) / 2);
  const ticks = center.map((c) => {
    const out = [];
    for (let t = Math.ceil((c - h) / step - 1e-9) * step; t <= c + h + 1e-9; t += step) out.push(Math.abs(t) < step * 1e-6 ? 0 : t);
    return out;
  });
  return { center, h, step, ticks };
}
export function tickText(t, step) {
  const dec = step >= 1 ? 0 : step >= 0.5 ? 1 : 2;
  const s = Math.abs(t).toFixed(dec);
  return (t < 0 && Number(s) !== 0 ? "−" : "") + s;
}

// ---- the sentences --------------------------------------------------------------------------------

const WORD = { 1: "one", 2: "two", 3: "three", 4: "four" };
const SHAPE = { 1: "fall on a line", 2: "lie in a plane", 3: "fill all three new axes" };

/** "a", "a and b", "a, b, and c". */
export function list(xs) {
  if (xs.length <= 1) return xs.join("");
  if (xs.length === 2) return `${xs[0]} and ${xs[1]}`;
  return `${xs.slice(0, -1).join(", ")}, and ${xs[xs.length - 1]}`;
}

/** Which measurements the matrix drops (its all-zero rows). */
export function sayDropped(names, zr) {
  if (!zr.length) return "No measurement is dropped: every row of the matrix has a nonzero entry.";
  if (zr.length === names.length) return `Every row is all zero, so all ${WORD[names.length]} measurements are dropped.`;
  const dropped = list(zr.map((i) => names[i]));
  const kept = list(names.filter((_, i) => !zr.includes(i)));
  return `The matrix drops ${dropped}: ${zr.length === 1 ? "its row is" : "their rows are"} all zero, so the projection uses only ${kept}.`;
}

/** Whether a whole column came out zero, and how many dimensions the flowers span. */
export function sayColumns(k, zc, rk) {
  if (!zc.length) {
    if (rk === k) return `No column came out all zero, so the flowers spread over ${k === 2 ? "both" : "all three"} new axes.`;
    if (k === 2) return "No column came out all zero, but the two columns are equal up to sign, so the flowers fall on a line.";
    return `No column came out all zero, but the three columns are not independent, so the flowers ${SHAPE[rk]}.`;
  }
  if (zc.length === k) return `${k === 2 ? "Both columns" : "All three columns"} came out all zero, so every flower sits at the origin.`;
  const nums = list(zc.map((j) => String(j + 1)));
  const one = zc.length === 1;
  return `${one ? "Column" : "Columns"} ${nums} came out all zero, so every flower sits at 0 on new ${one ? "axis" : "axes"} ${nums}, and the flowers ${SHAPE[rk]}.`;
}

/** What one family is, in its own numbers. */
export function sayFamily(fam) {
  if (fam.id === "gaussian") return "Gaussian: every entry is a standard normal number, so no entry is zero.";
  const s = fam.s;
  const root = Number.isInteger(Math.sqrt(s)) ? String(Math.sqrt(s)) : `√${s}`;
  const named = /\(s = ([^)]*)\)/.exec(fam.long || "");
  const sname = named && named[1] !== String(s) ? `${named[1]} = ${s}` : String(s);
  if (s === 1) return `${fam.label}: every entry is +1 or −1, with probability 1/2 each (s = 1), so no entry is zero.`;
  return `${fam.label}: each entry is +${root} or −${root} with probability 1/${2 * s} each, and 0 with probability ${s - 1}/${s} (s = ${sname}).`;
}

/**
 * One drawn matrix added to a tally ({n, zero, both}) when the reader drew it: one more draw, one
 * more with an all-zero column if it has one, and one more with both petal rows zero if it has them.
 * The cards the plate draws on its own, before the reader touches it, leave the tally as it was.
 */
export function addDraw(tally, R, petalRows, byReader) {
  if (!byReader) return tally;
  tally.n++;
  if (zeroCols(R).length) tally.zero++;
  if (petalRows.every((i) => R[i].every((v) => v === 0))) tally.both++;
  return tally;
}

/** The running tally for one family and k. */
export function sayTally({ label, k, n, zero, both, pz, pb }) {
  if (n === 0) return `No draws of yours in this tally with ${label} at k = ${k}. Each draw you make adds one to it; the cards the plate draws on its own, before you touch it, do not.`;
  const draws = `${n} ${n === 1 ? "draw" : "draws"}`;
  if (pz === 0 && pb === 0 && zero === 0 && both === 0) {
    return `In ${draws} with ${label} at k = ${k}, none had a column of zeros or a zero petal row: this family has no zero entries.`;
  }
  return `In ${draws} with ${label} at k = ${k}, ${zero} had a column of zeros (${pct1(zero / n)}%, against ${pct1(pz)}% in theory) and ${both} had both petal rows zero (${pct1(both / n)}%, against ${pct1(pb)}%).`;
}
/** The short count beside a bar: the reader's draws in this tally, none after a reset or the plate's own draws. */
export const sayCount = (c, n) => (n ? `${c} of ${n} · ${pct1(c / n)}%` : "no draws of yours");
/**
 * The band in whole counts, both ends included, with the shares rounded: it holds at least 95% of
 * runs, not exactly 95%, since counts are whole numbers. After three draws of Eq. 4 at k = 3, the band
 * for both petal rows zero is 0 to 1 of 3 (97.8% of runs), and 1 of 3 prints as 33.3%; after one draw
 * of Eq. 4 at k = 2, it is both outcomes and holds every run. A band of one count has no width, so
 * only its dotted edge shows; on this plate that count is 0 (a family with no zero entries, or a rare
 * event in one draw), so its share is exact.
 */
export function sayBand(band, n) {
  const runs = `At least 95% of runs of ${n} ${n === 1 ? "draw" : "draws"}`;
  if (band[0] === band[1]) return `${runs} land at exactly ${band[0]} of ${n} (the dotted line, at ${pct1(band[0] / n)}%)`;
  return `${runs} land from ${band[0]} to ${band[1]} of ${n}, both ends included (hatched; ${pct1(band[0] / n)}% to ${pct1(band[1] / n)}% rounded)`;
}

/** One flower, read off the plate. */
export function sayFlower(i, species, raw, scaled, e) {
  const f1 = (v) => v.toFixed(1), f2 = (v) => halfUp(v, 2);
  return `Flower ${i + 1}, ${species}: sepal ${f1(raw[0])} by ${f1(raw[1])} cm, petal ${f1(raw[2])} by ${f1(raw[3])} cm. ` +
    `Scaled: ${scaled.map(f2).join(", ")}. Projected: (${e.map(f2).join(", ")}).`;
}
