// "Where the rule draws the line", as numbers and sentences. No DOM: iris-boundary.js draws what this
// module computes, and tools/aml/iris-boundary.check.mjs holds every value and sentence to the
// study's own code, run in Python.
//
// The study's classifier (kamp_hw1.py, ParzenWindowClassifier) is refit here on the two petal
// measurements of every flower in the chosen problem. Each flower of the problem's first class gets
// weight 1/n+, each flower of the other class weight 1/n-, the bias is
//   b = (a+ . K a+ - a- . K a-) / 2
// from the kernel matrix K of those flowers, and the decision score before the sign is
//   h(x) = sum_i (a+_i - a-_i) K(x_i, x) - b,
// with h(x) >= 0 picking the first class. The kernels are the study's four, written as kamp_hw1.py
// writes them, with the Gaussian's width sigma = 1/sqrt(2) unless the reader moves the width slider.
// Measurements are in centimeters.

/** The part of the petal plane drawn, in cm: petal length across, petal width up. */
export const WINDOW = { x: [0.5, 7.5], y: [0, 2.75] };

/** The width slider: position k gives sigma = (study's sigma) * 2^(k / STEPS); k = 0 is the study's own. */
export const STEPS = 8, K_MIN = -16, K_MAX = 12;

/** Round half up to dp decimals, from the value first rounded to ten places, as the paper rounds. */
export function halfUp(x, dp) {
  const n = Math.round(x * 1e10) / 1e10;
  const k = 10 ** dp;
  const v = Math.floor(Number((Math.abs(n) * k).toFixed(6)) + 0.5) / k;
  return (n < 0 && v !== 0 ? "-" : "") + v.toFixed(dp);
}

/** Thousands separators on the whole part of a decimal string ("12345.6" -> "12,345.6"). */
function group(s) {
  const [w, f] = s.split(".");
  const g = w.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return f === undefined ? g : `${g}.${f}`;
}

const MINUS = "−";
// No line break inside a short formula: a no-break space around its signs, a word joiner after a slash.
const NB = " ", WJ = "⁠";

// ---------------------------------------------------------------- the data, shaped once

export function shape(data) {
  const flowers = data.flowers.map((r, i) => ({ i, x: [r[0], r[1]], s: r[2] }));
  return {
    data, flowers, species: data.species, features: data.features,
    problems: data.problems, kernels: data.kernels, sigmaStudy: data.sigma.value,
  };
}

export const problem = (M, n) => M.problems.find((p) => p.n === n);
export const kernelOf = (M, id) => M.kernels.find((k) => k.id === id);

/** The width at slider position k, in cm; position 0 is exactly the study's width. */
export const sigmaAt = (M, k) => (k === 0 ? M.sigmaStudy : M.sigmaStudy * 2 ** (k / STEPS));

/** The two classes of a problem: the species in each, and which flowers are in the problem. */
export function classes(M, n) {
  const P = problem(M, n);
  const other = M.species.map((_, s) => s).filter((s) => s !== P.one && s !== P.excluded);
  const rows = M.flowers.filter((f) => f.s !== P.excluded);
  return { P, one: [P.one], other, rows, na: rows.filter((f) => f.s === P.one).length, nb: rows.filter((f) => f.s !== P.one).length };
}

/** What a class is called in the text: one species, or "the other two species". */
export function className(M, n, side, form = "plain") {
  const C = classes(M, n);
  const set = side === "a" ? C.one : C.other;
  if (set.length === 1) return M.species[set[0]];
  if (form === "list") return `${M.species[set[0]]} and ${M.species[set[1]]}`;
  return "the other two species";
}

/** The problem's short name for its button. */
export function buttonLabel(M, n) {
  const C = classes(M, n);
  const one = M.species[C.one[0]];
  const cap = one[0].toUpperCase() + one.slice(1);
  return C.other.length === 2 ? `${cap} vs. rest` : `${cap} vs. ${M.species[C.other[0]]}`;
}

// ---------------------------------------------------------------- the study's kernels

/**
 * The kernel as kamp_hw1.py writes it, for two points of two measurements:
 *   dot_product_kernel: np.dot(x, y)
 *   polynomial_kernel:  (np.dot(x, y) + c) ** degree, with c = 1.0
 *   gaussian_kernel:    np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))
 * np.linalg.norm of a vector is the square root of its dot product with itself, and the study
 * squares that root again; so does this.
 */
export function kernelFn(M, id, sigma) {
  const K = kernelOf(M, id);
  if (id === "dot") return (x, y) => x[0] * y[0] + x[1] * y[1];
  if (id === "d2" || id === "d3") {
    const c = K.c, d = K.degree;
    return (x, y) => (x[0] * y[0] + x[1] * y[1] + c) ** d;
  }
  if (id === "gauss") {
    const den = 2 * sigma ** 2;
    return (x, y) => {
      const a = x[0] - y[0], b = x[1] - y[1];
      return Math.exp(-(Math.sqrt(a * a + b * b) ** 2) / den);
    };
  }
  throw new Error(`unknown kernel ${id}`);
}

/**
 * The rule fit to every flower of problem n on the two petal measurements, as
 * ParzenWindowClassifier.fit does: weights, the kernel matrix, and the bias b.
 */
export function fit(M, n, id, sigma = M.sigmaStudy) {
  const C = classes(M, n);
  const X = C.rows.map((f) => f.x), y = C.rows.map((f) => (f.s === C.P.one ? 1 : -1));
  const nPos = y.filter((v) => v === 1).length, nNeg = y.filter((v) => v === -1).length;
  const aPos = y.map((v) => (v === 1 ? 1 / nPos : 0)), aNeg = y.map((v) => (v === -1 ? 1 / nNeg : 0));
  const k = kernelFn(M, id, sigma);
  // b = (a+ . (K a+) - a- . (K a-)) / 2
  let qp = 0, qn = 0;
  for (let i = 0; i < X.length; i++) {
    let kp = 0, kn = 0;
    for (let j = 0; j < X.length; j++) { const v = k(X[i], X[j]); kp += v * aPos[j]; kn += v * aNeg[j]; }
    qp += aPos[i] * kp; qn += aNeg[i] * kn;
  }
  const b = (qp - qn) / 2;
  const ap = aPos.map((v, i) => v - aNeg[i]);
  const F = { M, n, id, sigma, X, y, idx: C.rows.map((f) => f.i), nPos, nNeg, ap, b, k };
  F.fast = fastScorer(F);
  return F;
}

/** The decision score before the sign, summed in the study's order: sum_i a'_i K(x_i, x) - b. */
export function score(F, x) {
  let s = 0;
  for (let i = 0; i < F.X.length; i++) s += F.ap[i] * F.k(F.X[i], x);
  return s - F.b;
}

/** Which class the rule picks at score h: ties go to the first class, as in the study. */
export const side = (h) => (h >= 0 ? "a" : "b");

/**
 * The same score, faster, for drawing: flowers at the same spot are summed into one weight, the
 * dot product is folded into one vector, and the Gaussian skips the root. It agrees with score()
 * up to rounding (the check holds it to the study's code at many points).
 */
function fastScorer(F) {
  const spots = new Map();
  F.X.forEach((x, i) => { const key = `${x[0]},${x[1]}`; const s = spots.get(key); if (s) s.w += F.ap[i]; else spots.set(key, { x, w: F.ap[i] }); });
  const U = [...spots.values()].filter((s) => s.w !== 0);
  const ux = Float64Array.from(U, (s) => s.x[0]), uy = Float64Array.from(U, (s) => s.x[1]), uw = Float64Array.from(U, (s) => s.w);
  const m = U.length, b = F.b;
  if (F.id === "dot") {
    let a0 = 0, a1 = 0;
    for (let i = 0; i < m; i++) { a0 += uw[i] * ux[i]; a1 += uw[i] * uy[i]; }
    return (x0, x1) => a0 * x0 + a1 * x1 - b;
  }
  if (F.id === "d2" || F.id === "d3") {
    const K = kernelOf(F.M, F.id), c = K.c, cube = K.degree === 3;
    return (x0, x1) => {
      let s = 0;
      for (let i = 0; i < m; i++) { const t = ux[i] * x0 + uy[i] * x1 + c; s += uw[i] * (cube ? t * t * t : t * t); }
      return s - b;
    };
  }
  const g = -1 / (2 * F.sigma ** 2);
  return (x0, x1) => {
    let s = 0;
    for (let i = 0; i < m; i++) { const a = ux[i] - x0, d = uy[i] - x1; s += uw[i] * Math.exp(g * (a * a + d * d)); }
    return s - b;
  };
}

/** The two class averages on the petal measurements (with the dot product, the rule compares distances to them). */
export function averages(F) {
  const a = [0, 0], c = [0, 0];
  F.X.forEach((x, i) => { const t = F.y[i] === 1 ? a : c; t[0] += x[0]; t[1] += x[1]; });
  return { a: [a[0] / F.nPos, a[1] / F.nPos], b: [c[0] / F.nNeg, c[1] / F.nNeg] };
}

/**
 * With the degree-2 kernel the score is a quadratic in the two measurements, x'Qx + (linear) +
 * (constant), with Q = sum_i a'_i x_i x_i'. Its boundary is a conic: a hyperbola when det Q < 0,
 * an ellipse when det Q > 0. Returns the conic's name, or null when it is degenerate.
 */
export function conic(F) {
  let q00 = 0, q01 = 0, q11 = 0, l0 = 0, l1 = 0, c0 = 0;
  const c = kernelOf(F.M, "d2").c;
  F.X.forEach((x, i) => { const w = F.ap[i]; q00 += w * x[0] * x[0]; q01 += w * x[0] * x[1]; q11 += w * x[1] * x[1]; l0 += w * x[0]; l1 += w * x[1]; c0 += w; });
  // h = q00 x^2 + 2 q01 x y + q11 y^2 + 2c (l0 x + l1 y) + c^2 c0 - b
  const det2 = q00 * q11 - q01 * q01;
  const A = [[q00, q01, c * l0], [q01, q11, c * l1], [c * l0, c * l1, c * c * c0 - F.b]];
  const det3 = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
  const scale = Math.max(Math.abs(q00), Math.abs(q11), 1e-300);
  if (Math.abs(det3) < 1e-9 * scale ** 3 || Math.abs(det2) < 1e-12 * scale * scale) return { det2, det3, name: null };
  return { det2, det3, name: det2 < 0 ? "hyperbola" : "ellipse" };
}

// ---------------------------------------------------------------- the score's scale, and its words

/**
 * The spacing of the level lines: a 1, 2, or 5 times a power of ten, chosen so that three steps
 * reach about the 75th percentile of the score's size at the problem's flowers. The stipple is
 * densest from three steps out. e is the power of ten.
 */
export function levelStep(F) {
  const a = F.X.map((x) => Math.abs(score(F, x))).sort((p, q) => p - q);
  const q = a[Math.floor(0.75 * (a.length - 1))];
  const raw = q / 3;
  let e = Math.floor(Math.log10(raw));
  let m = raw / 10 ** e;
  if (m >= 10 - 1e-9) { e += 1; m /= 10; }
  const mm = m >= 5 - 1e-9 ? 5 : m >= 2 - 1e-9 ? 2 : 1;
  return { L: Number(`${mm}e${e}`), m: mm, e, q };
}

/** How dense the stipple is at score h, from 0 at the boundary to 1 at three level steps and beyond. */
export const density = (h, L) => Math.min(1, Math.abs(h) / (3 * L));

/** The decimals a score is shown to: one more than the level step's. */
export const scoreDp = (S) => Math.max(0, 1 - S.e);

/** A score as shown: signed, half up, with more decimals if it would otherwise read as zero. */
export function fmtScore(h, S) {
  if (h === 0) return "0";
  let dp = scoreDp(S), s = halfUp(Math.abs(h), dp);
  while (Number(s) === 0 && dp < 14) { dp++; s = halfUp(Math.abs(h), dp); }
  return (h > 0 ? "+" : MINUS) + group(s);
}

/** A value at the score's precision, with a minus sign but no plus (for b). */
export function fmtPlain(v, S) {
  const s = group(halfUp(Math.abs(v), scoreDp(S)));
  return (v < 0 && Number(s.replace(/,/g, "")) !== 0 ? MINUS : "") + s;
}

/** A level line's label: the level, at the level step's precision. */
export function fmtLevel(j, S) {
  const dp = Math.max(0, -S.e);
  const s = group(halfUp(Math.abs(j) * S.L, dp));
  return (j > 0 ? "+" : j < 0 ? MINUS : "") + s;
}

/** The width as shown: two decimals, in cm. */
export const fmtSigma = (s) => halfUp(s, 2);

/** The words beside the width slider. */
export function widthText(M, k) {
  const s = sigmaAt(M, k);
  if (k === 0) return `σ = 1/√2 ≈ ${fmtSigma(s)} cm, the study’s fixed width`;
  return `σ = ${fmtSigma(s)} cm, beyond the study`;
}

const KWORD = { dot: "dot product", d2: "degree-2 polynomial kernel", d3: "degree-3 polynomial kernel", gauss: "Gaussian kernel" };

/** The sentences under the plate, for a problem, kernel, and width. */
export function stateText(M, n, id, k, F = null) {
  F = F || fit(M, n, id, sigmaAt(M, k));
  const C = classes(M, n), S = levelStep(F);
  const out = [];
  const one = M.species[C.one[0]];
  if (C.other.length === 2) out.push(`Problem ${n} sets ${one} (${C.na} flowers, filled) against ${className(M, n, "b", "list")} (${C.nb}, open).`);
  else out.push(`Problem ${n} sets ${one} (${C.na} flowers, filled) against ${M.species[C.other[0]]} (${C.nb}, open), with ${M.species[C.P.excluded]} removed.`);
  if (id === "dot") {
    out.push("With the dot product, the rule compares distances to the two class averages (the stars), so the boundary is the straight line of points equally far from both.");
  } else if (id === "d2") {
    const c = conic(F);
    out.push(`With the degree-2 polynomial kernel, the score is a polynomial of degree 2 in the two measurements, so the boundary is a conic${c.name ? `, here a ${c.name}` : ""}.`);
  } else if (id === "d3") {
    out.push("With the degree-3 polynomial kernel, the score is a polynomial of degree 3 in the two measurements, so the boundary is a cubic curve.");
  } else {
    const study = `σ${NB}=${NB}1/${WJ}√2${NB}≈${NB}${fmtSigma(M.sigmaStudy)}${NB}cm`;
    const sig = k === 0 ? `${study}, the study’s fixed width` : `σ${NB}=${NB}${fmtSigma(F.sigma)}${NB}cm, a width beyond the study, which fixed ${study}`;
    out.push(`With the Gaussian kernel (${sig}), each flower adds a bump of that width to the score, up for ${one} and down for ${className(M, n, "b")}, so the boundary can curve around groups of flowers.`);
    out.push(`Far from every flower, the score settles at −b${NB}=${NB}${fmtScore(-F.b, S)}, so the rule picks ${className(M, n, side(-F.b))} there.`);
  }
  out.push(`Dotted lines, where drawn, show roughly where the score is ${fmtLevel(1, S).replace("+", "±")}, ${fmtLevel(2, S).replace("+", "±")}, and ${fmtLevel(3, S).replace("+", "±")}; the stipple is densest beyond ${fmtLevel(3, S).replace("+", "±")}.`);
  if (n === 1 && id !== "dot") out.push("The study ran Problem 1 with the dot product only.");
  return out;
}

/** The flowers of the problem at a spot, in index order. */
export function flowersAt(M, n, x) {
  const C = classes(M, n);
  return C.rows.filter((f) => f.x[0] === x[0] && f.x[1] === x[1]).map((f) => f.i);
}

/**
 * The words for a probe at x (cm), with the flowers there if it sits on some. A free probe's
 * coordinates are already rounded to hundredths of a centimeter; the score is computed at them.
 */
export function probeText(M, n, F, x, at = []) {
  const S = levelStep(F), h = score(F, x);
  const out = [];
  if (at.length) {
    const sp = at.map((i) => M.flowers[i].s);
    const same = sp.every((s) => s === sp[0]);
    const where = `petal ${halfUp(x[0], 1)} cm long and ${halfUp(x[1], 1)} cm wide`;
    if (at.length === 1) out.push(`Flower ${at[0] + 1} (Iris ${M.species[sp[0]]}) sits here: ${where}.`);
    else if (same) out.push(`Flowers ${list(at.map((i) => String(i + 1)))} (all Iris ${M.species[sp[0]]}) sit here: ${where}.`);
    else out.push(`Flowers ${list(at.map((i, j) => `${i + 1} (Iris ${M.species[sp[j]]})`))} sit here: ${where}.`);
    out.push(h === 0 ? `The score here is exactly 0, a tie, which the rule gives to ${className(M, n, "a")}.` : `The score here is ${fmtScore(h, S)}, so the rule picks ${className(M, n, side(h))}.`);
  } else {
    const head = `At petal length ${halfUp(x[0], 2)} cm and width ${halfUp(x[1], 2)} cm`;
    out.push(h === 0 ? `${head}, the score is exactly 0, a tie, which the rule gives to ${className(M, n, "a")}.` : `${head}, the score is ${fmtScore(h, S)}, so the rule picks ${className(M, n, side(h))}.`);
  }
  return out;
}

function list(xs) {
  if (xs.length === 1) return xs[0];
  if (xs.length === 2) return `${xs[0]} and ${xs[1]}`;
  return `${xs.slice(0, -1).join(", ")}, and ${xs[xs.length - 1]}`;
}

/** The rule card's last line: what the rule is fit to here, and each class's weight. */
export function ruleText(M, n) {
  const C = classes(M, n);
  return `Fit to all ${C.na + C.nb} flowers of Problem ${n}: ${C.na} weighted 1/${C.na}, ${C.nb} weighted ${MINUS}1/${C.nb}.`;
}

/** A one-line summary for the plane's accessible name. */
export function summary(M, n, id, k) {
  const C = classes(M, n);
  const kw = id === "gauss" ? `the Gaussian kernel at σ = ${fmtSigma(sigmaAt(M, k))} cm` : `the ${KWORD[id]}`;
  return `The decision score of the rule refit on petal length and width, Problem ${n} (${M.species[C.one[0]]} against ${className(M, n, "b", "list")}), with ${kw}: stipple in each class’s ink where the rule picks it, and a firm line where the score is zero.`;
}

/** The kernel's formula, as the rule card shows it. */
export function formula(M, id) {
  if (id === "dot") return "K(x, y) = x · y";
  if (id === "d2") return "K(x, y) = (x · y + 1)²";
  if (id === "d3") return "K(x, y) = (x · y + 1)³";
  return "K(x, y) = exp(−‖x − y‖² / 2σ²)";
}

// ---------------------------------------------------------------- level lines and the boundary

/** A grid over the window: nx by ny cells. */
export function gridSpec(nx, ny, win = WINDOW) {
  return { x0: win.x[0], x1: win.x[1], y0: win.y[0], y1: win.y[1], nx, ny, dx: (win.x[1] - win.x[0]) / nx, dy: (win.y[1] - win.y[0]) / ny };
}

/** The fast score at every node of the grid, row by row (j up, i across). */
export function gridValues(F, G) {
  const v = new Float64Array((G.nx + 1) * (G.ny + 1));
  for (let j = 0; j <= G.ny; j++) {
    const y = G.y0 + j * G.dy;
    for (let i = 0; i <= G.nx; i++) v[j * (G.nx + 1) + i] = F.fast(G.x0 + i * G.dx, y);
  }
  return v;
}

/**
 * The contour of the score at a level, by marching squares on the grid, joined into polylines of
 * [x, y] points in cm. On each grid edge whose two ends lie on opposite sides of the level, the
 * crossing is found by bisection on the score (to 2^-36 of the edge, about 1e-12 cm), so the line is
 * where the score actually crosses, not an estimate between nodes (the dotted level lines, which
 * are drawn but not read, take fewer steps). A cell whose four corners
 * alternate is split by the sign at its center. Each point also carries the grid edge it lies on.
 */
export function contour(F, G, values, level = 0, iters = 36) {
  const W = G.nx + 1;
  const at = (i, j) => values[j * W + i] - level;
  const up = (v) => v >= 0;
  const pts = new Map();
  const node = (i, j) => [G.x0 + i * G.dx, G.y0 + j * G.dy];
  function cross(id) {
    if (pts.has(id)) return id;
    const horiz = id % 2 === 0, base = id >> 1, i = base % W, j = Math.floor(base / W);
    const p = node(i, j), q = horiz ? node(i + 1, j) : node(i, j + 1);
    let va = at(i, j), lo = 0, hi = 1;
    const f = (t) => F.fast(p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t) - level;
    for (let it = 0; it < iters; it++) {
      const mid = (lo + hi) / 2, vm = f(mid);
      if (up(vm) === up(va)) lo = mid; else hi = mid;
    }
    const t = (lo + hi) / 2;
    pts.set(id, { x: [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t], edge: id });
    return id;
  }
  const H = (i, j) => 2 * (j * W + i), V = (i, j) => 2 * (j * W + i) + 1;
  const segs = [];
  for (let j = 0; j < G.ny; j++) {
    for (let i = 0; i < G.nx; i++) {
      const a = up(at(i, j)), b = up(at(i + 1, j)), c = up(at(i + 1, j + 1)), d = up(at(i, j + 1));
      const e = [];
      if (a !== b) e.push(H(i, j));
      if (b !== c) e.push(V(i + 1, j));
      if (c !== d) e.push(H(i, j + 1));
      if (d !== a) e.push(V(i, j));
      if (e.length === 2) segs.push([cross(e[0]), cross(e[1])]);
      else if (e.length === 4) {
        const center = up(F.fast(G.x0 + (i + 0.5) * G.dx, G.y0 + (j + 0.5) * G.dy) - level);
        const [bot, right, top, left] = e.map(cross);
        if (center === a) { segs.push([bot, right]); segs.push([top, left]); } // a and c joined: cut off b and d
        else { segs.push([left, bot]); segs.push([right, top]); } // b and d joined: cut off a and c
      }
    }
  }
  // join segments that share a crossing into polylines
  const by = new Map();
  segs.forEach((s, k) => { for (const id of s) { if (!by.has(id)) by.set(id, []); by.get(id).push(k); } });
  const used = new Uint8Array(segs.length), lines = [];
  const walk = (k, from) => {
    const line = [from];
    let cur = k, at0 = from;
    while (cur !== -1 && !used[cur]) {
      used[cur] = 1;
      const s = segs[cur], next = s[0] === at0 ? s[1] : s[0];
      line.push(next);
      at0 = next;
      cur = (by.get(next) || []).find((q) => !used[q]) ?? -1;
    }
    return line;
  };
  // open lines start at a crossing that only one segment touches
  for (const [id, ks] of by) if (ks.length === 1 && !used[ks[0]]) lines.push(walk(ks[0], id));
  segs.forEach((s, k) => { if (!used[k]) lines.push(walk(k, s[0])); });
  return lines.map((ids) => ids.map((id) => pts.get(id)));
}
