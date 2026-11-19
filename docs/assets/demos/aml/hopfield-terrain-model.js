// The recall terrain's model, with no DOM: the measured grid, the mesh through it, its contours, the
// capacity markers, and every sentence the plate shows. hopfield-terrain.js draws these and
// tools/aml/hopfield-terrain.check.mjs holds them to the study's files.
//
// The grid: rows are the overlap levels rho (0.0 to 0.9), columns the measured loads P (2 to 9, then
// 12, 15, 18, 24, 30, and 40). A vertex is one measured cell (summary_synthetic_curves.csv). The
// load axis is logarithmic, as in the paper's figures: u = (ln P - ln 2) / (ln 40 - ln 2). Each
// square between four vertices is two flat triangles split along the same diagonal, so every point
// on a face is linear interpolation between measured vertices, and a contour crosses an edge where
// linear interpolation along that edge reaches its level. Along a row, that is exactly how the
// study reads its capacity P50 (linear in log P between the two loads around the crossing).

/** Contour levels: every 10% of recall. The 50% level is the capacity's. */
export const LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
export const HALF = 0.5;
/** How long a switch of rule or noise takes to settle, in seconds. */
export const MORPH_S = 0.6;

// ------------------------------------------------------------------ numbers as the paper prints them

/** A recall share written with four decimals in the summary, printed with three, rounded half up. */
export function f3(x) {
  const n = Math.round(x * 10000); // exact for a value written with four decimals
  return (Math.floor((n + 5) / 10) / 1000).toFixed(3);
}

/** How a P50 from summary_capacity.csv is classed: a value, between 2 and 3, above 40, or below 2. */
export function p50Kind(s) {
  if (s.startsWith(">")) return "above";
  if (s.startsWith("<")) return "below";
  return Number(s) < 3 ? "between" : "value";
}

/** A P50 as the paper writes it: "2–3" below 3 (Table II), ">40" and "<2" as written, else the value. */
export function p50Label(s) {
  return p50Kind(s) === "between" ? "2–3" : s;
}

// ------------------------------------------------------------------ the data, shaped

/** The plate's data file, with fast access to one rule's grid at one noise level. */
export function shape(data) {
  const nr = data.levels.length, nc = data.loads.length;
  const l0 = Math.log(data.loads[0]), l1 = Math.log(data.loads[nc - 1]);
  const v0 = Number(data.levels[0]), v1 = Number(data.levels[nr - 1]);
  const u = data.loads.map((P) => (Math.log(P) - l0) / (l1 - l0));
  const v = data.levels.map((lv) => (Number(lv) - v0) / (v1 - v0));
  const names = Object.fromEntries(data.rules.map((r) => [r.id, r.name]));
  const cache = new Map();
  function grid(rule, noise) {
    const key = `${rule}:${noise}`;
    if (cache.has(key)) return cache.get(key);
    const rows = data.recall[rule][String(noise)];
    const z = new Float64Array(nr * nc), lo = new Float64Array(nr * nc), hi = new Float64Array(nr * nc);
    for (let i = 0; i < nr; i++) for (let j = 0; j < nc; j++) {
      const [r, a, b] = rows[i][j];
      z[i * nc + j] = r; lo[i * nc + j] = a; hi[i * nc + j] = b;
    }
    const g = { rule, noise, z, lo, hi };
    cache.set(key, g);
    return g;
  }
  const uOfP = (P) => (Math.log(P) - l0) / (l1 - l0);
  return { data, nr, nc, u, v, names, grid, uOfP, tris: triangles(nr, nc), sets: data.sets };
}

/** The mesh's triangles: each square (i, j)-(i+1, j+1) split along the diagonal from (i, j) to (i+1, j+1). */
export function triangles(nr, nc) {
  const t = [];
  for (let i = 0; i < nr - 1; i++) for (let j = 0; j < nc - 1; j++) {
    const a = i * nc + j, b = a + 1, c = a + nc, d = c + 1;
    t.push([a, b, d], [a, d, c]);
  }
  return t;
}

/** One measured cell: its recall, interval, load, and level. */
export function cell(S, g, i, j) {
  const k = i * S.nc + j;
  return { i, j, P: S.data.loads[j], level: S.data.levels[i], recall: g.z[k], lo: g.lo[k], hi: g.hi[k] };
}

// ------------------------------------------------------------------ contours

/**
 * The contour of the heights z at level L through the mesh: one segment per triangle that has
 * vertices on both sides (a vertex at or above L counts as above). Each end of a segment lies on a
 * mesh edge from vertex a to vertex b (a < b), at the fraction t = (L - z_a) / (z_b - z_a) of the
 * way: linear interpolation. Returns { segs: [[p, q], ...], tri: [triangle index per segment] }.
 */
export function contour(z, tris, L) {
  const segs = [], tri = [];
  tris.forEach((T, n) => {
    const pts = [];
    for (const [x, y] of [[T[0], T[1]], [T[1], T[2]], [T[2], T[0]]]) {
      const a = Math.min(x, y), b = Math.max(x, y), za = z[a], zb = z[b];
      if ((za >= L) !== (zb >= L)) pts.push({ a, b, t: (L - za) / (zb - za) });
    }
    if (pts.length === 2) { segs.push(pts); tri.push(n); }
  });
  return { segs, tri };
}

/** A contour point's place in the plane (u, v) and its height. */
export function at(S, z, p) {
  const ua = S.u[p.a % S.nc], va = S.v[Math.floor(p.a / S.nc)];
  const ub = S.u[p.b % S.nc], vb = S.v[Math.floor(p.b / S.nc)];
  return [ua + p.t * (ub - ua), va + p.t * (vb - va), z[p.a] + p.t * (z[p.b] - z[p.a])];
}

/**
 * The part of triangle T where the interpolated recall is at least L, as a polygon in (u, v): the
 * triangle's corners at or above L and the contour's crossings on its edges, in order.
 */
export function above(S, z, T, L) {
  const P = T.map((k) => [S.u[k % S.nc], S.v[Math.floor(k / S.nc)], z[k]]);
  const out = [];
  for (let n = 0; n < 3; n++) {
    const a = P[n], b = P[(n + 1) % 3], ina = a[2] >= L, inb = b[2] >= L;
    if (ina) out.push([a[0], a[1]]);
    if (ina !== inb) {
      const t = (L - a[2]) / (b[2] - a[2]);
      out.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
    }
  }
  return out;
}

/** The segments joined into lines, end to end where they share an edge crossing. */
export function chain(c) {
  const key = (p) => `${p.a},${p.b},${p.t}`;
  const ends = new Map();
  c.segs.forEach((s, n) => s.forEach((p, e) => {
    const k = key(p);
    if (!ends.has(k)) ends.set(k, []);
    ends.get(k).push([n, e]);
  }));
  const used = new Uint8Array(c.segs.length), lines = [];
  const walk = (n, e) => {
    // from segment n, entering at its end e, keep going through the other end
    const pts = [c.segs[n][e]], tri = [];
    for (;;) {
      used[n] = 1;
      const out = c.segs[n][1 - e];
      pts.push(out); tri.push(c.tri[n]);
      const next = (ends.get(key(out)) || []).find(([m]) => !used[m]);
      if (!next) break;
      [n, e] = next;
    }
    return { pts, tri };
  };
  // open lines start at an end used once; the rest are loops
  c.segs.forEach((s, n) => {
    if (used[n]) return;
    const e = s.findIndex((p) => ends.get(key(p)).length === 1);
    if (e >= 0) lines.push(walk(n, e));
  });
  c.segs.forEach((s, n) => { if (!used[n]) lines.push(walk(n, 0)); });
  return lines;
}

/** A triangle's downhill direction in the plane, in world units (x = u * W, y = v * D), and its slope. */
export function downhill(S, z, T, W, D) {
  const P = T.map((k) => [S.u[k % S.nc] * W, S.v[Math.floor(k / S.nc)] * D, z[k]]);
  const e1 = [P[1][0] - P[0][0], P[1][1] - P[0][1], P[1][2] - P[0][2]];
  const e2 = [P[2][0] - P[0][0], P[2][1] - P[0][1], P[2][2] - P[0][2]];
  // the plane z = gx * x + gy * y + c through the three points
  const det = e1[0] * e2[1] - e1[1] * e2[0];
  const gx = (e1[2] * e2[1] - e2[2] * e1[1]) / det, gy = (e2[2] * e1[0] - e1[2] * e2[0]) / det;
  const s = Math.hypot(gx, gy);
  return s > 1e-12 ? { dx: -gx / s, dy: -gy / s, slope: s } : { dx: 0, dy: 0, slope: 0 };
}

// ------------------------------------------------------------------ hachures

/** For each triangle, its neighbor across each edge (edge n runs from corner n to corner n + 1), or -1. */
export function neighbors(tris) {
  const byEdge = new Map();
  tris.forEach((T, t) => T.forEach((a, n) => {
    const b = T[(n + 1) % 3], k = a < b ? `${a},${b}` : `${b},${a}`;
    if (!byEdge.has(k)) byEdge.set(k, []);
    byEdge.get(k).push(t);
  }));
  return tris.map((T, t) => T.map((a, n) => {
    const b = T[(n + 1) % 3], k = a < b ? `${a},${b}` : `${b},${a}`;
    return byEdge.get(k).find((o) => o !== t) ?? -1;
  }));
}

/**
 * Hachures in the old surveyors' manner: short strokes that start on a contour and run straight
 * downhill across the flat triangles until they reach the next contour down (or a flat, a valley,
 * the border, or maxLen). They start every `gap` world units along each contour line. Returns
 * strokes as lists of points [x, y, z] in world plan units (x = u * W, y = v * D) and recall z, with
 * the triangle each piece lies in: { pts, tris }. Drawing only: the strokes are read from the faces.
 */
export function hachureStrokes(S, z, W, D, { gap = 0.18, maxLen = 1.2, step = 0.1, levels = LEVELS } = {}) {
  const nb = S.nb || (S.nb = neighbors(S.tris));
  const P = (k) => [S.u[k % S.nc] * W, S.v[Math.floor(k / S.nc)] * D];
  const out = [];
  for (const L of levels) {
    const target = Math.max(L - step, 0);
    for (const ln of chain(contour(z, S.tris, L))) {
      let next = gap / 2;
      for (let s = 0; s < ln.tri.length; s++) {
        const a = at(S, z, ln.pts[s]), b = at(S, z, ln.pts[s + 1]);
        const A = [a[0] * W, a[1] * D], B = [b[0] * W, b[1] * D], seg = Math.hypot(B[0] - A[0], B[1] - A[1]);
        let t = next;
        for (; t < seg; t += gap) {
          const st = trace(S, z, nb, P, W, D, ln.tri[s], [A[0] + ((B[0] - A[0]) * t) / seg, A[1] + ((B[1] - A[1]) * t) / seg], L, target, maxLen);
          if (st) out.push(st);
        }
        next = t - seg;
      }
    }
  }
  return out;
}

function trace(S, z, nb, P, W, D, t, p, zStart, target, maxLen) {
  const pts = [[p[0], p[1], zStart]], tris = [];
  let zc = zStart, len = 0, entry = -1;
  for (let guard = 0; guard < 16; guard++) {
    const T = S.tris[t], d = downhill(S, z, T, W, D);
    if (!d.slope) break;
    // where the ray leaves this triangle (not back through the edge it came in by)
    const V = T.map(P);
    let exit = Infinity, edge = -1;
    for (let n = 0; n < 3; n++) {
      if (n === entry) continue;
      const [qx, qy] = V[n], [rx, ry] = V[(n + 1) % 3], ex = rx - qx, ey = ry - qy;
      const den = d.dx * ey - d.dy * ex;
      if (Math.abs(den) < 1e-12) continue;
      const wx = qx - p[0], wy = qy - p[1];
      const s = (wx * ey - wy * ex) / den, r = (wx * d.dy - wy * d.dx) / den;
      if (s > 1e-9 && r >= -1e-9 && r <= 1 + 1e-9 && s < exit) { exit = s; edge = n; }
    }
    if (edge < 0) break;
    const toTarget = (zc - target) / d.slope, room = maxLen - len;
    const s = Math.min(exit, toTarget, room);
    p = [p[0] + d.dx * s, p[1] + d.dy * s];
    zc -= d.slope * s; len += s;
    pts.push([p[0], p[1], zc]); tris.push(t);
    if (s < exit || nb[t][edge] < 0) break;
    // into the next triangle, entering by the shared edge
    const nt = nb[t][edge], shared = [T[edge], T[(edge + 1) % 3]];
    entry = S.tris[nt].findIndex((a, n) => shared.includes(a) && shared.includes(S.tris[nt][(n + 1) % 3]));
    t = nt;
  }
  return tris.length ? { pts, tris } : null;
}

// ------------------------------------------------------------------ the capacity on the 50% contour

/** Where a row of recall first falls below L, read as the study reads P50: linear in log P. */
export function firstCrossing(row, loads, L = HALF) {
  const j = row.findIndex((r) => r < L);
  if (j < 0) return { kind: "above" };
  if (j === 0) return { kind: "below" };
  const r0 = row[j - 1], r1 = row[j];
  const frac = (r0 - L) / (r0 - r1);
  const lp = Math.log(loads[j - 1]) + frac * (Math.log(loads[j]) - Math.log(loads[j - 1]));
  return { kind: "cross", P: Math.exp(lp), from: loads[j - 1], to: loads[j], j };
}

/** Each overlap's P50 marker for one rule and noise: as written, as the paper writes it, and where it sits. */
export function markers(S, rule, noise) {
  const g = S.grid(rule, noise);
  return S.data.capacity[rule][String(noise)].map(([p50, lo, hi], i) => {
    const kind = p50Kind(p50);
    const row = Array.from(g.z.subarray(i * S.nc, (i + 1) * S.nc));
    const x = firstCrossing(row, S.data.loads);
    const m = { i, level: S.data.levels[i], p50, lo, hi, kind, label: p50Label(p50), v: S.v[i] };
    if (kind === "value" || kind === "between") { m.u = S.uOfP(Number(p50)); m.from = x.from; m.to = x.to; }
    if (kind === "above") m.u = 1;
    if (kind === "below") m.u = 0;
    return m;
  });
}

/**
 * Where both views draw a marker, in the plane (u from the smallest load to the largest, v from the lowest overlap
 * to the highest): a finite one at its P50 on its row, a >40 one's arrow at its row's right edge. The map takes its
 * diamonds, arrows, and row labels, and the 3D view its markers, from here and nowhere else.
 */
export const markerUV = (m) => [m.u, m.v];

// ------------------------------------------------------------------ picking

/** The measured vertex nearest a point of the plane, in world units (x = u * W, y = v * D). */
export function nearestCell(S, u, v, W, D) {
  let best = 0, bd = Infinity;
  for (let i = 0; i < S.nr; i++) for (let j = 0; j < S.nc; j++) {
    const d = ((u - S.u[j]) * W) ** 2 + ((v - S.v[i]) * D) ** 2;
    if (d < bd) { bd = d; best = i * S.nc + j; }
  }
  return { i: Math.floor(best / S.nc), j: best % S.nc };
}

/** The measured point nearest a screen point, among projected points [[x, y], ...]. */
export function nearestPoint(pts, x, y) {
  let best = -1, bd = Infinity;
  pts.forEach((p, k) => { if (!p) return; const d = (p[0] - x) ** 2 + (p[1] - y) ** 2; if (d < bd) { bd = d; best = k; } });
  return { k: best, d: Math.sqrt(bd) };
}

/** One step across the grid from (i, j), held at its edges. */
export function step(S, i, j, di, dj) {
  return { i: Math.min(Math.max(i + di, 0), S.nr - 1), j: Math.min(Math.max(j + dj, 0), S.nc - 1) };
}

// ------------------------------------------------------------------ the switch between rules

/** Smooth start and stop over [0, 1]. */
export const ease = (s) => (s <= 0 ? 0 : s >= 1 ? 1 : s * s * (3 - 2 * s));

/** Heights part of the way from a to b (only ever drawn, never read as a measurement); exactly a at
 * s = 0 and exactly b at s = 1. */
export function mix(a, b, s, out = new Float64Array(a.length)) {
  for (let k = 0; k < a.length; k++) out[k] = (1 - s) * a[k] + s * b[k];
  return out;
}

// ------------------------------------------------------------------ sentences (functions of numbers)

/** The readout's fields at one measured point, as printed. */
export function readout({ rule, noise, level, P, recall, lo, hi }) {
  return { recall: f3(recall), ci: `[${f3(lo)}, ${f3(hi)}]`, P: String(P), rho: level, rule, noise: `${noise}%` };
}

/** The reading at one measured point. */
export function sayCell({ rule, noise, level, P, recall, lo, hi, sets }) {
  const side = recall >= HALF ? "at or above" : "below";
  return `${rule}, ${noise}% noise: with ${P} patterns stored at expected overlap ρ = ${level}, exact recall is ${f3(recall)} ` +
    `[${f3(lo)}, ${f3(hi)}] (a 95% interval over ${sets} pattern sets), ${side} 50% recall.`;
}

/** The capacity along the selected row, as the paper writes it. */
export function sayRow({ level, p50, lo, hi, kind, from, to }) {
  if (kind === "value") {
    return `At ρ = ${level}, recall first falls below 50% at P50 = ${p50} [${lo}, ${hi}], where the bright contour ` +
      `crosses this row: interpolated in log P between the measured loads ${from} and ${to}.`;
  }
  if (kind === "between") {
    return `At ρ = ${level}, recall falls below 50% between 2 and 3 patterns. The paper writes this as 2–3 and gives ` +
      "no number, since an interpolation between those two loads says nothing more.";
  }
  if (kind === "above") return `At ρ = ${level}, recall stays at or above 50% up to 40 patterns, the largest load measured, so P50 is written >40.`;
  return `At ρ = ${level}, recall is already below 50% with 2 patterns stored, so P50 is written <2.`;
}

/** The note while the surface moves between two rules (or noise levels). A switch made while the
 * surface still moves starts from a blend, not from a measurement, so it names only its target. */
export function sayMorph(from, to, interrupted = false) {
  if (interrupted) return `Moving to ${to}. The surface is interpolated until it settles, and nothing on it is measured.`;
  return `Moving from ${from} to ${to}. Until the surface settles, it is between two measurements, and nothing on it is measured.`;
}

/** The plate's whole reading for a screen reader: each overlap's P50 as the paper writes it. */
export function sayCapacities(rule, noise, ms) {
  return `Recall terrain for ${rule} at ${noise}% noise. P50, the load where recall first falls below 50%, by expected overlap ρ: ` +
    ms.map((m) => `${m.label} at ρ = ${m.level}`).join("; ") + ".";
}

/** What the faces are, said on screen. */
export const FACES = "Dots are measured points. The faces between them are linear interpolation, not measurements.";
export const MAP_NOTE = "Crosses are measured points; between them, linear interpolation. Hachures run downhill from each contour.";

/** While a switch runs, in place of the note and the plate's name (rule and noise are the target's). */
export const sayMoving = (rule, noise) =>
  `The surface is moving to ${rule} at ${noise}% noise; nothing on it is measured until it settles.`;
/** While a switch runs, in place of the selected row's capacity. */
export const sayRowMoving = (level, rule, noise) =>
  `At ρ = ${level}, the capacity for ${rule} at ${noise}% noise is read once the surface settles.`;
/** The cartouche's second line, and the legend's name for the grid marks: measured only once settled. */
export const cartoSub = (noise, moving) => `${noise}% noise · ${moving ? "moving, not measured" : "exact recall"}`;
export const dotName = (moving) => (moving ? "grid point on the interpolated surface" : "measured point");
