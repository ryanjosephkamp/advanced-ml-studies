// The model behind Chapter III's live plate "One hundred draws": the February report’s pipeline (one
// split, default SVMs, k = 2) over the study's 100 committed draws of the matrices, read from
// study/results/raw_replication.csv.gz. Nothing is trained or estimated here: each ticket is one
// committed draw, its accuracies are read from the file, and everything else is counting. Accuracies
// are held as whole numbers of correct test rows (the file stores each as a share of the test set, to
// six decimals), so every comparison and every rounding is exact. Pure functions, no DOM:
// projection-draws.js draws them, and tools/aml/projection-draws.check.mjs checks them against the
// committed files and the paper's printed numbers.

export const EQS = ["eq3", "eq4"];
export const EQ_LABEL = { eq3: "Eq. 3", eq4: "Eq. 4" };
const WORD = ["no", "one", "two", "three", "four", "five"];

// ---- reading the tickets --------------------------------------------------------------------------

/** Correct test rows behind a share stored to six decimals (n is at most 171, so the nearest whole number is exact). */
export const correct = (acc, n) => Math.round(acc * n);

/**
 * Every ticket, ready to read: per data set and matrix, the four kernels' correct test rows, the best
 * of them (the report’s rule: the highest test accuracy), which kernels reach it, and the matrix's
 * zero columns and zero rows; then which matrix leads on each data set and the ticket's verdict.
 */
export function prepare(data) {
  const ks = data.kernels.map((k) => k.id);
  const tickets = data.draws.map((dr) => {
    const ds = {};
    let ahead4 = 0, ahead3 = 0, ties = 0;
    for (const d of data.datasets) {
      const cell = {};
      for (const eq of EQS) {
        const c = dr.acc[d.id][eq].map((a) => correct(a, d.n_test));
        const best = Math.max(...c);
        const [zc, zr] = dr.zero[d.id][eq];
        cell[eq] = { c, best, kernels: ks.filter((_, i) => c[i] === best), zc, zr };
      }
      const diff = cell.eq4.best - cell.eq3.best;
      cell.diff = diff;
      cell.lead = diff > 0 ? "eq4" : diff < 0 ? "eq3" : "tie";
      if (diff > 0) ahead4++; else if (diff < 0) ahead3++; else ties++;
      ds[d.id] = cell;
    }
    return { seed: dr.seed, ds, ahead4, ahead3, ties, verdict: verdictOf(ahead4, ahead3) };
  });
  return { tickets, bySeed: new Map(tickets.map((t) => [t.seed, t])) };
}

/**
 * What a draw's table would have concluded. The report wrote that Eq. 4 outperforms Eq. 3 on three
 * of five data sets; a draw gives that headline when Eq. 4 is strictly ahead on three or more (as
 * the paper counts it), the opposite when Eq. 3 is, and neither when ties keep both below three.
 */
export function verdictOf(ahead4, ahead3) {
  if (ahead4 >= 3) return "headline";
  if (ahead3 >= 3) return "opposite";
  return "neither";
}

/** Counts over a list of tickets: the three verdicts, and how many data sets Eq. 4 led on (0 to 5). */
export function tally(tickets) {
  const out = { n: tickets.length, headline: 0, opposite: 0, neither: 0, byAhead: [0, 0, 0, 0, 0, 0] };
  for (const t of tickets) { out[t.verdict]++; out.byAhead[t.ahead4]++; }
  return out;
}

/** Where one ticket's best accuracy stands among the other 99 for one data set and matrix. */
export function rankOf(tickets, dsId, eq, seed) {
  const v = tickets.find((t) => t.seed === seed).ds[dsId][eq].best;
  let lower = 0, equal = 0, higher = 0;
  for (const t of tickets) {
    if (t.seed === seed) continue;
    const w = t.ds[dsId][eq].best;
    if (w < v) lower++; else if (w > v) higher++; else equal++;
  }
  return { lower, equal, higher, others: tickets.length - 1 };
}

/** How many tickets had a whole column of zeros in one data set's matrix (one or both columns). */
export function zeroColumnCount(tickets, dsId = "iris", eq = "eq4") {
  let one = 0, both = 0;
  for (const t of tickets) {
    const z = t.ds[dsId][eq].zc;
    if (z === 1) one++; else if (z >= 2) both++;
  }
  return { n: tickets.length, any: one + both, one, both };
}

/**
 * The board's stacks: tickets with the same best accuracy stack on one another in ticket order, and
 * nothing is moved sideways. Returns each ticket's level (0 next to the line) and the tallest stack.
 */
export function stacks(tickets, dsId, eq) {
  const groups = new Map();
  for (const t of [...tickets].sort((a, b) => a.seed - b.seed)) {
    const c = t.ds[dsId][eq].best;
    if (!groups.has(c)) groups.set(c, []);
    groups.get(c).push(t.seed);
  }
  const level = new Map();
  let tallest = 0;
  for (const seeds of groups.values()) {
    seeds.forEach((s, i) => level.set(s, i));
    tallest = Math.max(tallest, seeds.length);
  }
  return { level, tallest };
}

/** The board's axis, in percent: from a multiple of 5 at least 2 points below the lowest best accuracy, to 100. */
export function axisRange(tickets, datasets) {
  let lo = 100;
  for (const t of tickets) for (const d of datasets) for (const eq of EQS) lo = Math.min(lo, (100 * t.ds[d.id][eq].best) / d.n_test);
  return [Math.max(0, Math.floor((lo - 2) / 5) * 5), 100];
}

// ---- numbers as the paper prints them ---------------------------------------------------------------

/** 100 c / n as a percentage with one decimal, rounded half up, in exact integer arithmetic. */
export function pct(c, n) {
  const q = Math.floor((2000 * c + n) / (2 * n));
  return `${Math.floor(q / 10)}.${q % 10}`;
}
/** A difference of dc correct rows out of n, in percentage points with one decimal and its sign. */
export function pts(dc, n) {
  if (dc === 0) return "0.0";
  return (dc > 0 ? "+" : "−") + pct(Math.abs(dc), n);
}
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
/** A ticket's number as printed on it: two digits. */
export const ticketNo = (seed) => String(seed).padStart(2, "0");

// ---- random pulls -----------------------------------------------------------------------------------
// sfc32, seeded through splitmix32. Each pull takes one of the 100 tickets with equal chance and puts
// it back, so every pull is one of the study's draws at random.

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
  /** Uniform on [0, 1), 53 random bits. */
  next() { return ((this.u32() >>> 5) * 67108864 + (this.u32() >>> 6)) / 9007199254740992; }
  /** A whole number in [0, n), each with equal chance. */
  below(n) { return Math.floor(this.next() * n); }
}
export const freshSeed = () => 1 + Math.floor(Math.random() * 999999);
/**
 * The seed in the address (?seed=), as the page uses it: decimal digits only, read exactly at any
 * length and taken modulo 2^32 (?seed=9007199254740993 is seed 1), or null for anything else, such as
 * a sign, a point, or letters (?seed=7abc). The link the page offers, seedQuery(seed), reads back as
 * the same seed, so following it pulls the same tickets (?seed=4294967296 is seed 0, and so is its
 * link, ?seed=0).
 */
export function seedFromQuery(search) {
  const v = new URLSearchParams(search).get("seed");
  return v !== null && /^[0-9]+$/.test(v) ? Number(BigInt(v) % 4294967296n) : null;
}
export const seedQuery = (seed) => `?seed=${seed >>> 0}`;
/** One pull from the drum: a ticket's seed. */
export const pull = (rng, tickets) => tickets[rng.below(tickets.length)].seed;
/**
 * A visit's three streams: the reader's pulls, the drum's own pulls, and where a returned ticket drops
 * back into the drum. Only the first chooses the reader's tickets, so the reader's pulls at a seed
 * come in the same order whatever the drum does on its own, with or without motion.
 */
export const streams = (seed) => ({ reader: new Rng(seed), own: new Rng(seed ^ 0x6a09e667), drop: new Rng(seed ^ 0xbb67ae85) });

// ---- the sentences ------------------------------------------------------------------------------------

/** "a", "a and b", "a, b, and c". */
export function list(xs) {
  if (xs.length <= 1) return xs.join("");
  if (xs.length === 2) return `${xs[0]} and ${xs[1]}`;
  return `${xs.slice(0, -1).join(", ")}, and ${xs[xs.length - 1]}`;
}

/** The kernels that reach a matrix's best accuracy on one data set, as the ticket prints them. */
export function sayKernels(ids, shortOf) {
  if (ids.length === 4) return "all 4";
  return ids.map((k) => shortOf[k]).join(" ");
}

/** Which matrix leads on which data sets. */
export function sayLeads(t, datasets) {
  const by = (lead) => datasets.filter((d) => t.ds[d.id].lead === lead).map((d) => d.label);
  const a4 = by("eq4"), a3 = by("eq3"), tie = by("tie");
  if (a4.length === datasets.length) return "Eq. 4 comes out ahead on all five data sets.";
  if (a3.length === datasets.length) return "Eq. 3 comes out ahead on all five data sets.";
  const segs = [];
  if (a4.length) segs.push(`Eq. 4 comes out ahead on ${list(a4)}`);
  if (a3.length) segs.push(`${a4.length ? "Eq. 3 on" : "Eq. 3 comes out ahead on"} ${list(a3)}`);
  if (tie.length) segs.push(`the two tie on ${list(tie)}`);
  return `${segs.join("; ")}.`;
}

/** What the draw's table would have concluded. */
export function sayVerdict(t, reportSeed) {
  const report = t.seed === reportSeed;
  const of5 = (n) => (n === 5 ? "all five" : `${WORD[n]} of five`);
  if (t.verdict === "headline") {
    return report
      ? `Eq. 4 is ahead on ${of5(t.ahead4)} data sets: this is the report’s draw, and its headline.`
      : `Eq. 4 is ahead on ${of5(t.ahead4)} data sets, so this draw would also have given the report’s headline.`;
  }
  if (t.verdict === "opposite") return `Eq. 3 is ahead on ${of5(t.ahead3)} data sets, so this draw would have given the opposite headline.`;
  return `Neither matrix is ahead on three data sets (Eq. 4 on ${WORD[t.ahead4]}, Eq. 3 on ${WORD[t.ahead3]}, with ${WORD[t.ties]} tied), so this draw gives no headline either way.`;
}

/** The stamp on a ticket. */
export function stampOf(t) {
  if (t.verdict === "headline") return { title: "The report’s headline", sub: `Eq. 4 ahead on ${t.ahead4} of 5` };
  if (t.verdict === "opposite") return { title: "The opposite headline", sub: `Eq. 3 ahead on ${t.ahead3} of 5` };
  return { title: "No headline", sub: `${t.ahead4} for Eq. 4, ${t.ahead3} for Eq. 3, ${t.ties} tied` };
}

/** The zeros each Eq. 4 matrix had, as the file records them: how many rows and columns, not which. */
export function sayZeros(t, datasets) {
  const rows = datasets.map((d) => `${d.label} ${t.ds[d.id].eq4.zr} of ${d.d}`);
  const cols = datasets.filter((d) => t.ds[d.id].eq4.zc > 0).map((d) => (t.ds[d.id].eq4.zc >= 2 ? `both on ${d.label}` : `one on ${d.label}`));
  const c = cols.length ? `Whole columns of zeros: ${list(cols)}.` : "No column came out all zero.";
  return `Rows of zeros in Eq. 4’s matrices, each a dropped feature: ${list(rows)}. ${c}`;
}

/** Where one value on the board stands among the other 99 draws. */
export function sayRank(label, eq, seed, value, r) {
  return `${label}, ${EQ_LABEL[eq]}, ticket ${ticketNo(seed)}: ${value}%, higher than ${r.lower} of the other ${r.others} draws, equal to ${r.equal}, and lower than ${r.higher}.`;
}

/** The ticket's heading line. */
export function sayTicket(seed, reportSeed) {
  return seed === reportSeed ? `Seed ${seed}, the report’s own draw` : `Seed ${seed}`;
}

/**
 * The reader's pulls after a draw from the drum: the pulled tickets are added when the reader drew
 * them, and the drum's own pulls, before the reader touches the plate, leave the pulls as they were.
 */
export const countPulls = (mine, seeds, byReader) => (byReader ? [...mine, ...seeds] : mine);

/** The running count of the reader's pulls, beside the 100 committed draws. */
export function sayTally(n, mine, all) {
  if (n === 0) return `No pulls of yours in this count; the drum’s own pulls, before you touch the plate, are not counted. Across all ${all.n} committed draws, ${all.headline} give the report’s headline, ${all.opposite} the opposite, and ${all.neither} neither.`;
  const p = n === 1 ? "pull" : "pulls";
  return `In these ${n} ${p}, ${mine.headline} gave the report’s headline (${pct(mine.headline, n)}%), ${mine.opposite} the opposite (${pct(mine.opposite, n)}%), and ${mine.neither} neither (${pct(mine.neither, n)}%). Across all ${all.n} committed draws, the shares are ${pct(all.headline, all.n)}%, ${pct(all.opposite, all.n)}%, and ${pct(all.neither, all.n)}%.`;
}

/** The empty bar of the reader's pulls: none in the current count, after a reset or the drum's own pulls. */
export const NO_PULLS = "no pulls of yours";
/** The short count beside a bar: c of the reader's n pulls, or none of the reader's. */
export const sayCount = (c, n) => (n ? `${c} of ${n} · ${pct(c, n)}%` : NO_PULLS);

/** The running count in one line, beside the drum. */
export function sayMini(n, mine, all) {
  if (n === 0) return `No pulls of yours in this count. In all ${all.n} draws, ${all.headline} give the report’s headline.`;
  return `${n} ${n === 1 ? "pull" : "pulls"}: ${mine.headline} with the report’s headline (${pct(mine.headline, n)}%), against ${pct(all.headline, all.n)}% of all ${all.n}.`;
}

/** The Iris lottery: a whole column of zeros in the Eq. 4 matrix, in the reader's pulls, the file, and theory. */
export function sayIris(n, z, all, theory) {
  if (n === 0) return `Iris’s Eq. 4 matrix had a whole column of zeros in ${all.any} of the ${all.n} committed draws (${all.one} with one, ${all.both} with both); in theory, ${theory}% of draws do.`;
  const p = n === 1 ? "pull" : "pulls";
  return `Iris’s Eq. 4 matrix had a whole column of zeros in ${z} of these ${n} ${p} (${pct(z, n)}%), in ${all.any} of the ${all.n} committed draws (${pct(all.any, all.n)}%), and in ${theory}% of draws in theory.`;
}
