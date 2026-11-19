// The Hopfield playground's model (chapter II): the study's network, run in the page. Pure
// functions, no DOM, importable by Node, so the plate and its check run the same code.
//
// Everything follows the study's own code (study/hopfield_rules.py and study/run_study.py):
// - patterns are ±1 vectors of length 144 (a character's on pixel is +1);
// - Hebbian weights W = S^T S with a zero diagonal; Storkey's rule, one pattern at a time in the
//   order given, with the local field h_ij that leaves out neurons i and j; the pseudoinverse rule
//   W = Q (Q^T Q)^+ Q^T with a zero diagonal;
// - the standard update sets neuron j to +1 when (W s)_j + b_j >= 0 and to -1 otherwise, so a tie
//   (a field of exactly 0) turns the neuron on; the assignment's 0/1 update is the same update with
//   the bias b = W.sum(axis=0), since W[:, j] . x with x = (s + 1) / 2 is half of that field;
// - recall visits all 144 neurons in a fresh random order each sweep, recomputes the field at the
//   start of each sweep and updates it after every change, and stops after a sweep that changes
//   nothing, or after 50 sweeps;
// - every random order and noise draw comes from NumPy's own generator, PCG64 seeded through
//   SeedSequence(20260202, spawn_key=...), keyed as the study keys it, and Generator.permutation
//   and Generator.choice(replace=False) are reproduced draw for draw.

export const N = 144;
export const NEAR_TIE = 1e-9; // a nonzero field this small is a tie that rounding decides
export const SIDE = 12;
export const MAX_SWEEPS = 50;
export const SEED = 20260202;
export const DRAWS = 200; // the study's draws of update order (and noise) per query set
export const FLIPS = { 10: 14, 20: 29 }; // the study's fresh noise: bits flipped at 10% and 20%

// ================================================================== NumPy's random numbers
// SeedSequence (numpy/random/bit_generator.pyx) on 32-bit words.
const INIT_A = 0x43b0d7e5, MULT_A = 0x931e8875, INIT_B = 0x8b51f9dd, MULT_B = 0x58f38ded;
const MIX_L = 0xca01f9dd, MIX_R = 0x4973f715;

function words(n) {
  // _coerce_to_uint32_array for a non-negative integer: little-endian 32-bit words, [0] for 0
  if (!Number.isInteger(n) || n < 0) throw new Error(`seed words must be non-negative integers: ${n}`);
  const out = [];
  let v = BigInt(n);
  do { out.push(Number(v & 0xffffffffn)); v >>= 32n; } while (v > 0n);
  return out;
}

/** The pool of SeedSequence(entropy, spawn_key=key). */
export function seedPool(entropy, key = []) {
  let run = words(entropy);
  const spawn = key.flatMap(words);
  if (spawn.length > 0 && run.length < 4) run = run.concat(new Array(4 - run.length).fill(0));
  const ent = run.concat(spawn);
  let hc = INIT_A;
  const hashmix = (v) => {
    v = (v ^ hc) >>> 0;
    hc = Math.imul(hc, MULT_A) >>> 0;
    v = Math.imul(v, hc) >>> 0;
    return (v ^ (v >>> 16)) >>> 0;
  };
  const mix = (x, y) => {
    const r = (Math.imul(MIX_L, x) - Math.imul(MIX_R, y)) >>> 0;
    return (r ^ (r >>> 16)) >>> 0;
  };
  const pool = new Array(4);
  for (let i = 0; i < 4; i++) pool[i] = hashmix(i < ent.length ? ent[i] : 0);
  for (let s = 0; s < 4; s++) for (let d = 0; d < 4; d++) if (s !== d) pool[d] = mix(pool[d], hashmix(pool[s]));
  for (let s = 4; s < ent.length; s++) for (let d = 0; d < 4; d++) pool[d] = mix(pool[d], hashmix(ent[s]));
  return pool;
}

/** SeedSequence.generate_state(n, np.uint64), as BigInts. */
export function generateState64(pool, n) {
  let hc = INIT_B;
  const w = [];
  for (let i = 0; i < 2 * n; i++) {
    let v = pool[i % 4];
    v = (v ^ hc) >>> 0;
    hc = Math.imul(hc, MULT_B) >>> 0;
    v = Math.imul(v, hc) >>> 0;
    w.push((v ^ (v >>> 16)) >>> 0);
  }
  const out = [];
  for (let i = 0; i < n; i++) out.push((BigInt(w[2 * i + 1]) << 32n) | BigInt(w[2 * i]));
  return out;
}

const M64 = (1n << 64n) - 1n, M128 = (1n << 128n) - 1n;
const PCG_MULT = 0x2360ed051fc65da44385df649fccf645n; // (2549297995355413924 << 64) + 4865540595714422341

/** PCG64 (XSL-RR 128/64) as NumPy seeds and steps it, with NumPy's buffered 32-bit draws. */
export class PCG64 {
  constructor(seed128, inc128) {
    this.inc = ((inc128 << 1n) | 1n) & M128;
    this.state = 0n;
    this.state = (this.state * PCG_MULT + this.inc) & M128;
    this.state = (this.state + seed128) & M128;
    this.state = (this.state * PCG_MULT + this.inc) & M128;
    this.has32 = false;
    this.u32 = 0;
  }
  next64() {
    this.state = (this.state * PCG_MULT + this.inc) & M128;
    const s = this.state, x = ((s >> 64n) ^ s) & M64, r = s >> 122n;
    return r === 0n ? x : ((x >> r) | (x << (64n - r))) & M64;
  }
  next32() {
    if (this.has32) { this.has32 = false; return this.u32; }
    const n = this.next64();
    this.has32 = true;
    this.u32 = Number(n >> 32n);
    return Number(n & 0xffffffffn);
  }
}

/** np.random.default_rng(SeedSequence(entropy, spawn_key=key)): permutation and choice. */
export class Generator {
  constructor(entropy, key = []) {
    const [s0, s1, i0, i1] = generateState64(seedPool(entropy, key), 4);
    this.bits = new PCG64((s0 << 64n) | s1, (i0 << 64n) | i1);
  }
  /** random_interval: a masked draw in [0, max], retried until it fits. */
  interval(max) {
    if (max === 0) return 0;
    let mask = max;
    mask |= mask >>> 1; mask |= mask >>> 2; mask |= mask >>> 4; mask |= mask >>> 8; mask |= mask >>> 16;
    mask >>>= 0;
    let v;
    while ((v = (this.bits.next32() & mask) >>> 0) > max);
    return v;
  }
  /** random_bounded_uint64(0, rng, 0, use_masked=False): Lemire's method on 32 bits. */
  bounded(rng) {
    if (rng === 0) return 0;
    const excl = rng + 1;
    let m = this.bits.next32() * excl, left = m % 4294967296;
    if (left < excl) {
      const threshold = (4294967295 - rng) % excl;
      while (left < threshold) { m = this.bits.next32() * excl; left = m % 4294967296; }
    }
    return Math.floor(m / 4294967296);
  }
  /** Generator.permutation(n): arange(n) shuffled in place from the top down. */
  permutation(n) {
    const a = new Int32Array(n);
    for (let i = 0; i < n; i++) a[i] = i;
    for (let i = n - 1; i >= 1; i--) {
      const j = this.interval(i);
      const t = a[i]; a[i] = a[j]; a[j] = t;
    }
    return a;
  }
  /** Generator.choice(pop, size, replace=False): Floyd's algorithm, then a shuffle (pop <= 10000). */
  choice(pop, size) {
    if (size > pop || pop > 10000) throw new Error("choice: only size <= pop <= 10000");
    const idx = [], seen = new Set();
    for (let j = pop - size; j < pop; j++) {
      const v = this.bounded(j);
      if (!seen.has(v)) { seen.add(v); idx.push(v); } else { seen.add(j); idx.push(j); }
    }
    for (let i = size - 1; i >= 1; i--) {
      const j = this.bounded(i);
      const t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }
    return idx;
  }
}

/** The study's rng_for(*key). */
export const rngFor = (...key) => new Generator(SEED, key);

// ================================================================== the study's seed keys
// run_study.py: character draws are keyed (3, query set, draw, query) for the update order and
// (4, query set, draw, query) for fresh noise; the query sets are 0 test, 1 valid42, 2 fresh10,
// 3 fresh20. Subsets are drawn with (5, P, subset); their noise with (6, P, subset, query, noise
// index) and their order with (7, ...same).
export const QSETS = ["test", "valid42", "fresh10", "fresh20"];
export const QSET_ID = { test: 0, valid42: 1, fresh10: 2, fresh20: 3 };

/** The study's subset u of P characters, in the order it drew them. */
export const studySubset = (P, u) => rngFor(5, P, u).choice(18, P);

/**
 * The query for character k at draw d: a query file's row, or the character with `flips` bits
 * flipped. At 14 and 29 flips the draw is the study's own fresh 10% and 20% noise for the whole set;
 * at any other count the page keys its draw (8, flips, d, k), a key the study never used.
 */
export function noisyQuery(xi, flips, d, k) {
  const q = Int8Array.from(xi);
  if (flips <= 0) return q;
  const qid = flips === FLIPS[10] ? 2 : flips === FLIPS[20] ? 3 : null;
  const g = qid !== null ? rngFor(4, qid, d, k) : rngFor(8, flips, d, k);
  for (const i of g.choice(N, flips)) q[i] = -q[i];
  return q;
}
export const flipKey = (flips) => (flips === FLIPS[10] ? 2 : flips === FLIPS[20] ? 3 : null);

/** The update-order generator for query k at draw d of query set qid (the page's free noise uses 8). */
export const orderRng = (qid, d, k) => rngFor(3, qid, d, k);

// ================================================================== patterns
/** 0/1 rows (as in the CSVs) to ±1 patterns. */
export const toPM = (rows) => rows.map((r) => Int8Array.from(r, (v) => 2 * v - 1));
export const onCount = (s) => { let n = 0; for (let i = 0; i < N; i++) n += s[i] > 0; return n; };
export function hamming(a, b) { let d = 0; for (let i = 0; i < N; i++) d += a[i] !== b[i]; return d; }
export function equal(a, b) { for (let i = 0; i < N; i++) if (a[i] !== b[i]) return false; return true; }

// ================================================================== learning rules
/** hebb: W = S^T S, zero diagonal (integer-valued). */
export function hebb(S) {
  const W = new Float64Array(N * N);
  for (const xi of S) for (let i = 0; i < N; i++) {
    const a = xi[i], row = i * N;
    for (let j = 0; j < N; j++) W[row + j] += a * xi[j];
  }
  for (let i = 0; i < N; i++) W[i * N + i] = 0;
  return W;
}

/** storkey: one pattern at a time, in the order given (hopfield_rules.storkey). */
export function storkey(S) {
  let W = new Float64Array(N * N);
  const h = new Float64Array(N);
  for (const xi of S) {
    for (let i = 0; i < N; i++) {
      let s = 0;
      for (let k = 0; k < N; k++) s += W[i * N + k] * xi[k];
      h[i] = s;
    }
    const next = new Float64Array(N * N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const Hij = h[i] - W[i * N + j] * xi[j];     // H[i, j] = h_i - w_ij xi_j
      const Hji = h[j] - W[j * N + i] * xi[i];     // H.T[i, j] = H[j, i]
      const dW = (xi[i] * xi[j] - xi[i] * Hji - Hij * xi[j]) / N;
      next[i * N + j] = W[i * N + j] + dW;
    }
    for (let i = 0; i < N; i++) next[i * N + i] = 0;
    W = next;
  }
  return W;
}

/** Eigen-decomposition of a small symmetric matrix by cyclic Jacobi rotations. */
function jacobi(A, n) {
  const a = Float64Array.from(A), V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0, tot = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const v = a[i * n + j] ** 2; tot += v; if (i !== j) off += v; }
    if (off <= 1e-30 * tot) break;
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
      const apq = a[p * n + q];
      if (Math.abs(apq) < 1e-300) continue;
      const app = a[p * n + p], aqq = a[q * n + q];
      const th = (aqq - app) / (2 * apq);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) {
        const akp = a[k * n + p], akq = a[k * n + q];
        a[k * n + p] = c * akp - s * akq; a[k * n + q] = s * akp + c * akq;
      }
      for (let k = 0; k < n; k++) {
        const apk = a[p * n + k], aqk = a[q * n + k];
        a[p * n + k] = c * apk - s * aqk; a[q * n + k] = s * apk + c * aqk;
      }
      for (let k = 0; k < n; k++) {
        const vkp = V[k * n + p], vkq = V[k * n + q];
        V[k * n + p] = c * vkp - s * vkq; V[k * n + q] = s * vkp + c * vkq;
      }
    }
  }
  const vals = Array.from({ length: n }, (_, i) => a[i * n + i]);
  return { vals, V };
}

/** pinv: W = Q (Q^T Q)^+ Q^T with Q = S^T, zero diagonal. The cutoff is NumPy's: 1e-15 of the largest. */
export function pinv(S) {
  const P = S.length;
  const G = new Float64Array(P * P);
  for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) {
    let s = 0;
    for (let i = 0; i < N; i++) s += S[a][i] * S[b][i];
    G[a * P + b] = s;
  }
  const { vals, V } = jacobi(G, P);
  const top = Math.max(...vals.map(Math.abs));
  const Gp = new Float64Array(P * P);
  for (let k = 0; k < P; k++) {
    if (!(Math.abs(vals[k]) > 1e-15 * top)) continue;
    const inv = 1 / vals[k];
    for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) Gp[a * P + b] += V[a * P + k] * inv * V[b * P + k];
  }
  // A = Q G^+ (N x P), then W = A Q^T
  const A = new Float64Array(N * P);
  for (let i = 0; i < N; i++) for (let b = 0; b < P; b++) {
    let s = 0;
    for (let a = 0; a < P; a++) s += S[a][i] * Gp[a * P + b];
    A[i * P + b] = s;
  }
  const W = new Float64Array(N * N);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    let s = 0;
    for (let b = 0; b < P; b++) s += A[i * P + b] * S[b][j];
    W[i * N + j] = s;
  }
  for (let i = 0; i < N; i++) W[i * N + i] = 0;
  return W;
}

export const RULES = ["hebb", "storkey", "pinv"];
export const UPDATES = ["01", "pm"];
const LEARN = { hebb, storkey, pinv };

/** The bias of the assignment's 0/1 update in ±1 terms: b = W.sum(axis=0), the column sums. */
export function colSums(W) {
  const b = new Float64Array(N);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) b[j] += W[i * N + j];
  return b;
}

/** A network: weights for the stored patterns (in storage order) and the bias the update implies. */
export function network(rule, update, S) {
  if (!LEARN[rule]) throw new Error(`unknown rule ${rule}`);
  const W = S.length ? LEARN[rule](S) : new Float64Array(N * N);
  const b = update === "01" ? colSums(W) : new Float64Array(N);
  return { rule, update, W, b, P: S.length };
}

/** The study's name for a rule and update, or null when the study did not run that pair. */
export function studyRule(rule, update) {
  if (rule === "hebb") return update === "01" ? "hebb01" : "hebb";
  return update === "pm" ? rule : null;
}

// ================================================================== dynamics
/** h = W s + b. */
export function field(net, s, h = new Float64Array(N)) {
  const { W, b } = net;
  for (let i = 0; i < N; i++) {
    let t = 0;
    const row = i * N;
    for (let j = 0; j < N; j++) t += W[row + j] * s[j];
    h[i] = t + b[i];
  }
  return h;
}

/** True when one sweep from xi changes nothing, in any order (hopfield_rules.is_fixed_point). */
export function isFixedPoint(net, xi) {
  const h = field(net, xi);
  for (let i = 0; i < N; i++) if ((h[i] >= 0 ? 1 : -1) !== xi[i]) return false;
  return true;
}

/** The energy the chosen update never raises: -1/2 x^T W x on 0/1 states for the 0/1 update,
 *  -1/2 s^T W s - b^T s on ±1 states for the standard one (b = 0 there). */
export function energy(net, s) {
  const { W } = net;
  let e = 0;
  if (net.update === "01") {
    for (let i = 0; i < N; i++) {
      if (s[i] < 0) continue;
      const row = i * N;
      for (let j = 0; j < N; j++) if (s[j] > 0) e += W[row + j];
    }
    return -0.5 * e;
  }
  for (let i = 0; i < N; i++) {
    const row = i * N;
    let t = 0;
    for (let j = 0; j < N; j++) t += W[row + j] * s[j];
    e += s[i] * t + 2 * net.b[i] * s[i];
  }
  return -0.5 * e;
}

/** The standard energy on ±1 states, -1/2 s^T W s, which the 0/1 update can raise. */
export function energyPM(W, s) {
  let e = 0;
  for (let i = 0; i < N; i++) {
    const row = i * N;
    let t = 0;
    for (let j = 0; j < N; j++) t += W[row + j] * s[j];
    e += s[i] * t;
  }
  return -0.5 * e;
}

/**
 * Recall from s0 (hopfield_rules.recall): a fresh permutation each sweep from `rng`, the field
 * recomputed at each sweep's start and updated after each change; stops after a sweep with no
 * change, or after maxSweeps. With record, it also keeps every visit (neuron, field, change) and the
 * energy after each change, for the plate to replay one neuron at a time.
 */
export function recall(net, s0, rng, { maxSweeps = MAX_SWEEPS, record = false } = {}) {
  const s = Float64Array.from(s0);
  const h = new Float64Array(N);
  const { W } = net;
  const visits = record ? [] : null;
  let sweeps = maxSweeps, converged = false, ties = 0, nearTies = 0, flips = 0;
  const e0 = record ? energy(net, s) : 0;
  const ePM0 = record && net.update === "01" ? energyPM(W, s) : 0;
  let ePM = ePM0, pmRises = 0;
  for (let sweep = 1; sweep <= maxSweeps; sweep++) {
    field(net, s, h);
    let changed = 0;
    const order = rng.permutation(N);
    for (let t = 0; t < N; t++) {
      const j = order[t];
      const f = h[j];
      if (f === 0) ties++;
      else if (Math.abs(f) < NEAR_TIE) nearTies++;
      const nw = f >= 0 ? 1 : -1;
      const ch = nw !== s[j];
      if (ch) {
        const d = nw - s[j];
        for (let i = 0; i < N; i++) h[i] += W[i * N + j] * d;
        s[j] = nw;
        changed++;
        flips++;
      }
      if (record) {
        const v = { sweep, j, field: f, on: nw > 0, changed: ch };
        if (ch) {
          v.energy = energy(net, s);
          if (net.update === "01") { const e = energyPM(W, s); if (e > ePM + 1e-9 * Math.max(1, Math.abs(ePM))) pmRises++; ePM = e; }
        }
        visits.push(v);
      }
    }
    if (changed === 0) { sweeps = sweep; converged = true; break; }
  }
  const out = { s: Int8Array.from(s), sweeps, converged, ties, nearTies, flips };
  if (record) Object.assign(out, { visits, e0, pmRises });
  return out;
}

/**
 * What a final state is, against the stored patterns (in storage order) and their negatives. `tx`, the query's
 * own character, lets an unstored one be named when the state equals it exactly ("unstored"): the plate counts
 * recall only for stored characters, but the network can still settle on one it does not store.
 */
export function classify(s, stored, S, target, tx = null) {
  const dist = S.map((xi) => hamming(s, xi));
  let near = 0;
  for (let a = 1; a < S.length; a++) if (dist[a] < dist[near]) near = a;
  const ties = S.length ? stored.filter((c, a) => dist[a] === dist[near]) : [];
  const ti = stored.indexOf(target);
  let kind;
  if (ti >= 0 && dist[ti] === 0) kind = "recalled";
  else if (S.length && dist[near] === 0) kind = "other";
  else if (ti < 0 && tx && hamming(s, tx) === 0) kind = "unstored";
  else if (S.some((xi) => hamming(s, xi) === N)) kind = "inverse";
  else kind = "spurious";
  const inv = S.findIndex((xi) => hamming(s, xi) === N);
  return {
    kind, dist, near: S.length ? stored[near] : null, nearDist: S.length ? dist[near] : null, nearAll: ties,
    other: kind === "other" ? stored[near] : null, inverseOf: inv >= 0 ? stored[inv] : null,
    targetDist: ti >= 0 ? dist[ti] : null, on: onCount(s),
  };
}

// ================================================================== the plate's data and runs
const fromBits = (s) => Int8Array.from(s, (c) => (c === "1" ? 1 : -1));

/** The data file, shaped: ±1 characters and query files, and how many pixels each query flips. */
export function shape(data) {
  const chars = data.chars.map(fromBits), test = data.test.map(fromBits), valid = data.valid.map(fromBits);
  return {
    data, chars, test, valid, n: chars.length,
    testFlips: chars.map((c, k) => hamming(c, test[k])),
    validFlips: chars.map((c, k) => hamming(c, valid[k])),
  };
}

export const SOURCES = ["test", "valid", "noise"];

/** The study's query set for a source, or null for noise it never drew (any count but 14 and 29). */
export function qsetOf(source, flips) {
  if (source === "test") return "test";
  if (source === "valid") return "valid42";
  return flips === FLIPS[10] ? "fresh10" : flips === FLIPS[20] ? "fresh20" : null;
}

/** The query for character k at draw d. */
export function query(S, source, k, flips, d) {
  if (source === "test") return S.test[k];
  if (source === "valid") return S.valid[k];
  return noisyQuery(S.chars[k], flips, d, k);
}
export const queryFlips = (S, source, k, flips) => (source === "test" ? S.testFlips[k] : source === "valid" ? S.validFlips[k] : flips);

/** The update order for character k at draw d: the study's key for its query sets, else (9, flips, d, k). */
export function order(source, flips, d, k) {
  const q = qsetOf(source, flips);
  return q ? orderRng(QSET_ID[q], d, k) : rngFor(9, flips, d, k);
}

/** A network over the stored characters, stored in file order (Storkey's result depends on it). */
export function netFor(S, rule, update, stored) {
  const idx = [...stored].sort((a, b) => a - b);
  return network(rule, update, idx.map((c) => S.chars[c]));
}

/** One recall, recorded for the plate, with what it reached. */
export function run(S, net, st, k) {
  const stored = [...st.stored].sort((a, b) => a - b);
  const Sx = stored.map((c) => S.chars[c]);
  const q = query(S, st.source, k, st.flips, st.draw);
  const r = recall(net, q, order(st.source, st.flips, st.draw, k), { record: true });
  const c = classify(r.s, stored, Sx, k, S.chars[k]);
  const ti = stored.indexOf(k);
  return {
    ...r, ...c, q, target: k, flipsIn: queryFlips(S, st.source, k, st.flips),
    e1: r.visits.length ? lastEnergy(r) : r.e0,
    eTarget: ti >= 0 ? energy(net, Sx[ti]) : null,
    eStored: Sx.map((x) => energy(net, x)),
  };
}
function lastEnergy(r) {
  for (let i = r.visits.length - 1; i >= 0; i--) if (r.visits[i].energy !== undefined) return r.visits[i].energy;
  return r.e0;
}

/** Every stored character's query at the plate's draw: what each reached. */
export function tally(S, net, st) {
  const stored = [...st.stored].sort((a, b) => a - b);
  const Sx = stored.map((c) => S.chars[c]);
  const states = new Set(), out = [];
  for (const k of stored) {
    const r = recall(net, query(S, st.source, k, st.flips, st.draw), order(st.source, st.flips, st.draw, k));
    const c = classify(r.s, stored, Sx, k);
    states.add(r.s.join(","));
    out.push({ k, kind: c.kind, sweeps: r.sweeps, converged: r.converged, near: c.near, nearDist: c.nearDist, s: r.s });
  }
  return { per: out, recalled: out.filter((o) => o.kind === "recalled").length, distinct: states.size, P: stored.length };
}

/** Which stored characters are fixed points of the network. */
export const fixedPoints = (S, net, stored) => [...stored].sort((a, b) => a - b).filter((c) => isFixedPoint(net, S.chars[c]));

/** The assignment's own procedure (scripts/hopfield.py): seed 42 + query index, up to 100 sweeps. */
export const assignmentRecall = (net, q, idx) => recall(net, q, new Generator(42 + idx), { maxSweeps: 100 });

/** When a run is one the study made: all 18 stored, a pair of rule and update it ran, its query set, a draw under 200. */
export function studyRun(st, nChars = 18, draws = DRAWS) {
  const rule = studyRule(st.rule, st.update), qset = qsetOf(st.source, st.flips);
  if (!rule || !qset || st.stored.length !== nChars || st.draw >= draws) return null;
  return { rule, qset };
}

// ================================================================== numbers as the page writes them
const MINUS = "−";
const group = (digits) => digits.replace(/\B(?=(\d{3})+(?!\d))/g, ",");

/** Round half up (away from zero at .5), as the paper does, then write with grouped digits and a true minus. */
export function fmt(x, places = 0) {
  const m = 10 ** places;
  const v = Math.floor(Math.abs(x) * m + 0.5 + 1e-7);
  let s = String(Math.floor(v / m));
  s = group(s);
  if (places > 0) s += "." + String(v % m).padStart(places, "0");
  return (x < 0 && v !== 0 ? MINUS : "") + s;
}
/** The energy's places: Hebbian weights are whole numbers, so are its energies; the others take two. */
export const ePlaces = (rule) => (rule === "hebb" ? 0 : 2);
export const fmtE = (e, rule) => fmt(e, ePlaces(rule));
/** A field, signed: the assignment's W[:, j] . x for the 0/1 update (half the ±1 field), (W s)_j otherwise. */
export function fmtField(f, rule, update) {
  const v = update === "01" ? f / 2 : f;
  const s = fmt(v, ePlaces(rule));
  return v > 0 && s !== "0" && s !== "0.00" ? "+" + s : s;
}

/** Half up on a decimal string, done on its digits (for values the study wrote). */
export function halfUpStr(str, places) {
  const [i, f = ""] = String(str).split(".");
  const digits = (i + f.padEnd(places + 1, "0")).replace(/^-/, "");
  const keep = BigInt(digits.slice(0, i.replace(/^-/, "").length + places)), next = Number(digits[i.replace(/^-/, "").length + places] ?? 0);
  const v = keep + (next >= 5 ? 1n : 0n);
  const t = v.toString().padStart(places + 1, "0");
  return places ? `${t.slice(0, -places)}.${t.slice(-places)}` : t;
}
/** A mean from integer counts, half up to `places`, computed exactly. */
export function meanStr(sum, n, places) {
  const m = 10n ** BigInt(places);
  const v = (2n * BigInt(sum) * m + BigInt(n)) / (2n * BigInt(n));
  const t = v.toString().padStart(places + 1, "0");
  return places ? `${t.slice(0, -places)}.${t.slice(-places)}` : t;
}
export const pad2 = (k) => String(k).padStart(2, "0");
export const rc = (j) => `r${pad2(Math.floor(j / SIDE))} c${pad2(j % SIDE)}`;

// ================================================================== words (drafts; the numbers come in as arguments)
export const RULE_NAME = { hebb: "Hebbian", storkey: "Storkey", pinv: "pseudoinverse" };
export const UPDATE_NAME = { "01": "the assignment's 0/1 update", pm: "the standard ±1 update" };
const plural = (n, one, many = one + "s") => `${n} ${n === 1 ? one : many}`;
/** "11", "11 and 16", "3, 7, and 9" */
export function list(xs) {
  const s = xs.map(String);
  return s.length < 3 ? s.join(" and ") : `${s.slice(0, -1).join(", ")}, and ${s.at(-1)}`;
}
const chars = (xs) => (xs.length === 1 ? `character ${xs[0]}` : `characters ${list(xs)}`);

/** The status board's line (upper case, at most 18 characters). */
export function status(phase, o = {}) {
  if (phase === "ready") return `QUERY ${pad2(o.target)}`;
  if (phase === "run") return `SWEEP ${o.sweep}`;
  if (o.kind === "recalled") return `RECALLED ${pad2(o.target)}`;
  if (!o.converged) return "STILL MOVING";
  if (o.kind === "other") return `LANDED ON ${pad2(o.other)}`;
  if (o.kind === "unstored") return `EXACT ${pad2(o.target)}, UNSTORED`;
  if (o.kind === "inverse") return `MIRROR OF ${pad2(o.inverseOf)}`;
  return o.near === null ? "SPURIOUS" : `SPURIOUS, NEAR ${pad2(o.near)}`;
}

/** What the query is. */
export function sayQuery({ target, source, flips, isStored, draw }) {
  const what = source === "test" ? `Character ${target} from the test file, with ${plural(flips, "pixel")} flipped`
    : source === "valid" ? `Character ${target} from the 20% file, with ${plural(flips, "pixel")} flipped`
      : flips === 0 ? `Character ${target} itself, with no pixel flipped`
        : `Character ${target} with ${plural(flips, "pixel")} flipped at random (${fmt((100 * flips) / N, 1)}%, draw ${draw})`;
  return `${what}. ${isStored ? "It is stored." : "It is not stored, so the plate will not count it as recalled: watch where it goes."}`;
}

/** How the run went and where it ended. */
export function sayRun(r) {
  const E = (e) => fmtE(e, r.rule);
  const out = [];
  if (!r.converged) out.push(`After ${r.maxSweeps} sweeps, the study's limit, neurons were still changing (${plural(r.changes, "change")} so far); the energy went from ${E(r.e0)} to ${E(r.e1)}.`);
  else if (r.changes === 0) out.push(`The first sweep changed nothing: the query is already a fixed point, at energy ${E(r.e0)}.`);
  else {
    // r.changes counts changes, not neurons: a neuron can change in one sweep and change back in a later one.
    // A change at a field of exactly 0 leaves the energy where it was, and a fall smaller than the last place
    // shown can leave both ends reading alike, so the sentence says which of these this run did. Two tests
    // decide it: whether the ends differ by more than the tolerance, 1e-9 × |E| (at least 1e-9), and whether
    // they read alike at the last place shown. "Stayed" needs both: within the tolerance and reading alike.
    // A fall past the tolerance that reads apart says "fell from A to B", even when it is smaller than the
    // last place shown; ends within the tolerance that read apart give both readings.
    const tol = (e) => 1e-9 * Math.max(1, Math.abs(e)), places = ePlaces(r.rule);
    const a = E(r.e0), b = E(r.e1), unit = fmt(10 ** -places, places);
    let energy;
    if (r.e1 < r.e0 - tol(r.e0)) {
      energy = a !== b ? `the energy fell from ${a} to ${b}` : `the energy fell by less than ${unit}, so it reads ${b} at both ends`;
    } else if (r.e1 <= r.e0 + tol(r.e0) && a === b) {
      const atZero = r.visits?.length > 0 && r.visits.every((v) => !v.changed || v.field === 0);
      energy = `the energy stayed at ${a}${atZero ? `, since ${r.changes === 1 ? "the change" : "each change"} came at a field of exactly 0` : ""}`;
    } else if (r.e1 <= r.e0 + tol(r.e0)) {
      energy = `the energy changed by less than ${unit}, though it reads ${a} at the start and ${b} at the end`;
    } else energy = `the energy went from ${a} to ${b}`;
    out.push(`Settled in ${plural(r.sweeps, "sweep")}, the last of which changed nothing: ${plural(r.changes, "change")} in all, and ${energy}.`);
  }
  if (r.kind === "recalled") out.push(`The state is character ${r.target} exactly: recalled.`);
  else if (r.kind === "other") out.push(r.isStored ? `The state is character ${r.other}, a stored character, but not character ${r.target}.`
    : `The state is stored character ${r.other}; character ${r.target} is not stored.`);
  else if (r.kind === "unstored") out.push(`The state is character ${r.target} exactly, though it is not stored; the plate counts recall only for stored characters.`);
  else if (r.kind === "inverse") out.push(`The state is character ${r.inverseOf} with every pixel flipped, its negative.`);
  else {
    const own = r.isStored && r.nearAll.includes(r.target);
    const head = `The state is no stored character: ${plural(r.on, "pixel")} on, ${plural(r.nearDist, "bit")} from ${chars(r.nearAll)}`;
    if (own && r.nearAll.length === 1) out.push(`${head}, the query's own and the nearest stored.`);
    else if (own) out.push(`${head}, the nearest stored, one of them the query's own.`);
    else out.push(`${head}, the nearest stored. ` + (r.isStored ? `Character ${r.target} is ${plural(r.targetDist, "bit")} away.` : `Character ${r.target} is not stored.`));
  }
  if (r.kind !== "recalled" && r.converged) {
    const tol = (e) => 1e-9 * Math.max(1, Math.abs(e));
    // Two energies one of these sentences calls lower and higher are written at the last place shown or, where
    // they read alike there, at the fewest more places that set them apart (−55.962 and −55.961, not −55.96 twice);
    // then "lower still" gives this state's energy too, which the first sentence gave only at the last place shown.
    const apart = (e, f) => { let p = ePlaces(r.rule); while (p < 12 && fmt(e, p) === fmt(f, p)) p++; return [fmt(e, p), fmt(f, p)]; };
    const low = r.eStored.length ? Math.min(...r.eStored) : null;
    if (r.eStored.length >= 2 && r.e1 < low - tol(low)) {
      const [a, b] = apart(r.e1, low);
      out.push(`Its energy, ${a}, is below that of every stored character, the lowest of which is at ${b}.`);
    } else if (r.isStored) {
      const d = r.e1 - r.eTarget, [a, b] = apart(r.e1, r.eTarget);
      if (d < -tol(r.eTarget)) out.push(`Its energy, ${a}, is below character ${r.target}'s own, ${b}.`);
      else if (d > tol(r.eTarget)) out.push(`Character ${r.target}'s own energy is lower still, ${b}${a === E(r.e1) ? "" : ` (this state's is ${a})`}, but no single neuron's change leads down from here.`);
      else out.push(`Its energy equals character ${r.target}'s own, ${E(r.eTarget)}.`);
    }
  }
  return out.join(" ");
}

/** The notes under a run: which energy is drawn, ties, the storage order, and the study's own runs. */
export function sayNotes(n) {
  const out = [];
  if (n.update === "01") {
    if (n.changes === 0) out.push("The chart draws the energy on 0/1 states, −½xᵀWx, which the 0/1 update never raises.");
    else out.push(`Under the 0/1 update, the standard energy −½sᵀWs ${n.pmRises > 0 ? `rose at ${n.pmRises} of the ${plural(n.changes, "change")}` : "can rise (in this run it did not)"}, so the chart draws the energy on 0/1 states, −½xᵀWx, which this update never raises.`);
  } else out.push("The chart draws the energy −½sᵀWs, which the standard update never raises.");
  if (n.ties > 0) out.push(`${n.ties === 1 ? "Once" : `${n.ties} times`}, a neuron's field was exactly 0, and the tie rule set it on (a field of 0 counts as on, as in the study's code).`);
  if (n.nearTies > 0) out.push(`${n.nearTies === 1 ? "Once" : `${n.nearTies} times`}, a field was within 10⁻⁹ of 0, a tie up to rounding, which this page and the study's NumPy code can round differently.`);
  if (n.rule === "storkey") out.push(`Storkey's rule stores the characters one at a time, here in file order, and its weights depend on that order: under the standard ±1 update, with all 18 in file order, ${n.fixedFile} of them are fixed points; over the study's ${n.orders} random orders, ${n.ordMin} to ${n.ordMax} (${n.ordMean} on average).`);
  if (n.pairing === false) out.push(`The study ran ${RULE_NAME[n.rule]} weights only with the standard update; this pairing is the page's own.`);
  if (n.study) {
    const q = n.study.qset;
    const what = q === "test" ? `the test file's query for character ${n.target}` : q === "valid42" ? `the 20% file's query for character ${n.target}`
      : `character ${n.target} with fresh ${q === "fresh10" ? 10 : 20}% noise (${plural(n.flips, "pixel")})`;
    out.push(`This is one of the study's own runs: ${what} at draw ${n.draw} of ${n.draws}. Over all ${n.draws} draws, this query ${n.successes === 0 ? "never came back" : n.successes === n.draws ? "came back every time" : `came back ${plural(n.successes, "time")}`}.`);
  }
  if (n.common) out.push(`This run ended in the state the assignment's own run sent all ${n.baseQueries} of its queries to, with all 18 stored under Hebbian weights and the 0/1 update.`);
  return out;
}

/** The fixed-point readout. */
export const sayFixed = (f, P) => `${f} of ${P}`;

/** Every stored query at once. */
export function sayTally(t) {
  const src = t.source === "test" ? "from the test file" : t.source === "valid" ? "from the 20% file"
    : t.flips === 0 ? "from themselves, with no noise" : `with ${plural(t.flips, "pixel")} flipped`;
  if (t.P === 1) return `At draw ${t.draw}, the one stored character ${t.recalled ? "came back" : "did not come back"} ${src === "from themselves, with no noise" ? "from itself" : src}.`;
  return `At draw ${t.draw}, ${t.recalled} of the ${t.P} stored characters came back ${src}, and the ${t.P} queries ended in ${t.distinct === 1 ? "one state" : `${t.distinct} different states`}.`;
}

/** The study's measured number beside a tally, or why there is none. */
export function sayStudy(c) {
  if (c.kind === "full") {
    if (c.min === c.max) return `In the study, all ${c.draws} draws gave ${c.min} of 18.`;
    return `In the study, over ${c.draws} draws: ${c.mean} of 18 on average, from ${c.min} to ${c.max}.`;
  }
  // Storkey's weights depend on the storage order: the study stored each set in the order drawn, the page in file order
  if (c.kind === "subset") return `In the study's ${c.sets} random sets of ${c.P} characters, ${c.drawn ? "each stored in the order drawn, " : ""}with fresh ${c.noise}% noise, ${c.frac} of the queries came back.`;
  if (c.kind === "pairing") return "The study did not run this pairing of rule and update.";
  if (c.kind === "one") return "The study stored at least 2 characters.";
  if (c.kind === "files") return "The study ran smaller sets only with fresh noise of 14 and 29 pixels (10% and 20%).";
  return "The study drew fresh noise of 14 and 29 pixels only (10% and 20%).";
}

/** What the study measured for the plate's setting, as numbers for sayStudy. */
export function studyCompare(D, st) {
  const rule = studyRule(st.rule, st.update), P = st.stored.length, qset = qsetOf(st.source, st.flips);
  if (!rule) return { kind: "pairing" };
  if (P === D.chars.length) {
    if (!qset) return { kind: "noise" };
    const f = D.full[qset][rule];
    return { kind: "full", draws: f.draws, min: f.min, max: f.max, mean: halfUpStr(f.mean, 2) };
  }
  if (P < 2) return { kind: "one" };
  if (st.source !== "noise") return { kind: "files" };
  if (!qset) return { kind: "noise" };
  const noise = qset === "fresh10" ? 10 : 20;
  return { kind: "subset", P, noise, sets: D.subsetsPerSize, drawn: rule === "storkey", frac: halfUpStr(D.subsets[rule][String(noise)][String(P)], 3) };
}

/** One visit, for the line under the board. */
export function sayVisit(v, t, rule, update) {
  return `Visit ${t} of ${N} in sweep ${v.sweep}: neuron ${rc(v.j)}, field ${fmtField(v.field, rule, update)}, ${v.on ? "on" : "off"} (${v.changed ? "changed" : "no change"}).`;
}
export const sayVisitIdle = (phase) => (phase === "load" ? "The query flips onto the board." : "");

/** Which characters are stored. */
export function saySet({ P, n, u, sets }) {
  if (P === n) return `All ${n} characters are stored, in file order.`;
  if (u !== null && u !== undefined) return `${P} stored: the study's random set ${u} of ${P} characters (it drew ${sets} of each size), stored here in file order.`;
  return `${P} stored, chosen by hand, in file order.`;
}
