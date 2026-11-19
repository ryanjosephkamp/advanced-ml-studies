// The Iris specimen case, as numbers and sentences. No DOM: iris-specimens.js draws what this module
// computes, and tools/aml/iris-specimens.check.mjs holds every value and sentence to the data.
//
// The flowers are scikit-learn's 150 (data/iris-specimens.json). A problem splits them into two
// classes, as the study's code does (repeated_splits.py): one species against the other two pooled,
// or, on Problem 4, versicolor against virginica with setosa removed. A class average here is the
// mean of all of that class's flowers among the 150, in all four measurements; the classifier itself
// takes its averages from each split's training flowers. Distances are straight-line distances in
// centimeters, over all four measurements or over the three drawn.

/** The ruler span drawn for each measurement, in cm (whole or half centimeters, holding every flower). */
export const SPANS = [[4, 8], [1.5, 4.5], [0, 7], [0, 3]];

/** The measurements drawn by default: across the floor, into the floor, and up (indices into features). */
export const DEFAULT_AXES = [2, 3, 0];

/** Round half up to dp decimals, from the value first rounded to ten places, as the paper rounds. */
export function halfUp(x, dp) {
  const n = Math.round(x * 1e10) / 1e10;
  const k = 10 ** dp;
  const v = Math.floor(Number((Math.abs(n) * k).toFixed(6)) + 0.5) / k;
  return (n < 0 && v !== 0 ? "-" : "") + v.toFixed(dp);
}

const cm1 = (x) => halfUp(x, 1);
const cm2 = (x) => halfUp(x, 2);

export function mean(rows) {
  const m = [0, 0, 0, 0];
  for (const r of rows) for (let j = 0; j < 4; j++) m[j] += r[j];
  return m.map((s) => s / rows.length);
}

export function dist(a, b, dims = [0, 1, 2, 3]) {
  let s = 0;
  for (const j of dims) s += (a[j] - b[j]) ** 2;
  return Math.sqrt(s);
}

/** The plate's data, shaped once: flowers as {i, v, s}, each species' average, and the four problems. */
export function shape(data) {
  const flowers = data.flowers.map((r, i) => ({ i, v: r.slice(0, 4), s: r[4] }));
  const species = data.species;
  const bySpecies = species.map((_, s) => flowers.filter((f) => f.s === s).map((f) => f.v));
  return {
    data, flowers, species, features: data.features,
    speciesAvg: bySpecies.map(mean),
    speciesN: bySpecies.map((rows) => rows.length),
    problems: data.problems.map((p) => sides({ flowers, species }, p)),
  };
}

/** One problem's two classes: who is in each, how many flowers, and each class's average. */
function sides({ flowers, species }, p) {
  const one = [p.one];
  const other = species.map((_, s) => s).filter((s) => s !== p.one && s !== p.excluded);
  const rowsOf = (set) => flowers.filter((f) => set.includes(f.s)).map((f) => f.v);
  const a = mean(rowsOf(one)), b = mean(rowsOf(other));
  return { n: p.n, one, other, excluded: p.excluded, a, b, na: rowsOf(one).length, nb: rowsOf(other).length, d4: dist(a, b) };
}

export const problem = (M, n) => M.problems.find((p) => p.n === n);

/** The names of the three measurements drawn, as a list: "petal length, petal width, and sepal length". */
export function axisList(M, axes) {
  const names = axes.map((j) => M.features[j].name);
  return `${names[0]}, ${names[1]}, and ${names[2]}`;
}

/** What a class is called: one species, or two pooled. */
export function className(M, set, form = "plain") {
  const names = set.map((s) => M.species[s]);
  if (names.length === 1) return form === "avg" ? `the ${names[0]} average` : names[0];
  return form === "avg" ? `the average of ${names[0]} and ${names[1]}` : `${names[0]} and ${names[1]}`;
}

/** The short label drawn beside an average's star. */
export const starLabel = (M, set) => set.map((s) => M.species[s]).join(" + ");

/**
 * The numbers the readout shows for a problem and the three measurements drawn: the distance between
 * the two averages in all four measurements and in the three drawn, and, when the other class pools
 * two species, how far its average lies from each pooled species' own average.
 */
export function readout(M, n, axes) {
  const P = problem(M, n);
  const out = { n, d4: P.d4, d3: dist(P.a, P.b, axes), s4: cm1(P.d4), s3: cm1(dist(P.a, P.b, axes)) };
  if (P.other.length === 2) {
    out.pooled = P.other.map((s) => ({ s, d: dist(P.b, M.speciesAvg[s]), str: cm1(dist(P.b, M.speciesAvg[s])) }));
    // the other class's average lies nearer the one class's average than to either species it pools
    out.between = out.pooled.every((q) => P.d4 < q.d);
    out.halfway = Math.abs(out.pooled[0].d - out.pooled[1].d) < 1e-9;
  }
  return out;
}

/** The sentences under the case for a problem and the three measurements drawn. */
export function problemText(M, n, axes) {
  const P = problem(M, n), R = readout(M, n, axes);
  const one = className(M, P.one);
  const out = [];
  if (P.other.length === 2) out.push(`Problem ${n} sets ${one} against ${className(M, P.other)} together.`);
  else out.push(`Problem ${n} sets ${one} against ${className(M, P.other)}, with ${M.species[P.excluded]} removed (${P.na + P.nb} flowers).`);
  if (R.between) {
    const [p, q] = R.pooled;
    const where = R.halfway ? `halfway between the ${M.species[p.s]} and ${M.species[q.s]} averages, ${p.str} cm from each`
      : `between the ${M.species[p.s]} and ${M.species[q.s]} averages (${p.str} and ${q.str} cm from them)`;
    out.push(`The other class’s average falls ${where}, and only ${R.s4} cm from ${className(M, P.one, "avg")}, in all four measurements.`);
    const pp = M.data.paper;
    out.push(`So a rule that compares averages of the raw measurements has little to work with here: in the study, the dot-product rule’s mean test accuracy on Problem 3 was ${pp.dot_p3_mean} over ${pp.n_splits_all} random splits, below the majority-class rate of ${pp.maj_p3}.`);
  } else {
    out.push(`In all four measurements, the two averages are ${R.s4} cm apart.`);
  }
  out.push(`On the three measurements drawn (${axisList(M, axes)}), the two averages are ${R.s3} cm apart.`);
  return out;
}

/** The text for one flower: its measurements, and where it lies from the chosen problem's averages. */
export function flowerText(M, n, i, axes, stack = []) {
  const f = M.flowers[i], P = problem(M, n);
  const [sl, sw, pl, pw] = f.v.map(cm1);
  const out = [`Flower ${i + 1} of ${M.flowers.length}, Iris ${M.species[f.s]}: sepal ${sl} cm long and ${sw} cm wide, petal ${pl} cm long and ${pw} cm wide.`];
  if (f.s === P.excluded) {
    out.push(`Problem ${n} leaves ${M.species[f.s]} out.`);
  } else {
    const da = dist(f.v, P.a), db = dist(f.v, P.b);
    let dp = 1;
    while (dp < 4 && halfUp(da, dp) === halfUp(db, dp)) dp++;
    const nameA = className(M, P.one, "avg"), nameB = className(M, P.other, "avg");
    const own = P.one.includes(f.s) ? "a" : "b";
    let tail;
    if (da === db) tail = "so it is equally far from both";
    else {
      const near = da < db ? "a" : "b";
      tail = `so it is nearer ${near === "a" ? nameA : nameB}${near === own ? "" : " than its own class’s"}`;
    }
    out.push(`In all four measurements, it is ${halfUp(da, dp)} cm from ${nameA} and ${halfUp(db, dp)} cm from ${nameB}, ${tail}.`);
  }
  if (stack.length > 1) {
    const others = stack.filter((j) => j !== i).map((j) => j + 1);
    const list = others.length === 1 ? `flower ${others[0]}` : `flowers ${others.slice(0, -1).join(", ")} and ${others[others.length - 1]}`;
    out.push(`On the three measurements drawn (${axisList(M, axes)}), ${list} sits at the same spot; tap or click again to read it.`);
  }
  return out;
}

/** The text for one class average, picked on its star. */
export function averageText(M, n, side) {
  const P = problem(M, n);
  const set = side === "a" ? P.one : P.other, v = side === "a" ? P.a : P.b, count = side === "a" ? P.na : P.nb;
  const [sl, sw, pl, pw] = v.map(cm2);
  const who = set.length === 1 ? `The ${M.species[set[0]]} average` : `The average of ${className(M, set)}`;
  return [`${who}, over ${count} flowers: sepal ${sl} cm long and ${sw} cm wide, petal ${pl} cm long and ${pw} cm wide.`];
}

/** The text for a pooled species' own average (the small rings on Problems 1 to 3). */
export function pooledText(M, n, s) {
  const P = problem(M, n);
  const [sl, sw, pl, pw] = M.speciesAvg[s].map(cm2);
  const d = dist(M.speciesAvg[s], P.b);
  return [`The ${M.species[s]} average, over ${M.speciesN[s]} flowers: sepal ${sl} cm long and ${sw} cm wide, petal ${pl} cm long and ${pw} cm wide. It is ${cm1(d)} cm from the average of ${className(M, P.other)} together, in all four measurements.`];
}

/** The label on the line joining the two averages. */
export const lineLabel = (M, n) => `${cm1(problem(M, n).d4)} cm in 4D`;

/** The heading over the distance readout: how many flowers the chosen problem's averages are taken over. */
export function avgHeading(M, n) {
  const P = problem(M, n);
  return `Averages of the problem’s ${P.na + P.nb} flowers`;
}

/** The distance scale's accessible name: the distance between the two class averages on each problem. */
export function scaleLabel(M) {
  return "Distance between the two class averages in all four measurements, by problem: " + M.problems.map((P) => `Problem ${P.n}, ${cm1(P.d4)} cm`).join("; ") + ".";
}

/** A one-line summary for the canvas's accessible name. */
export function summary(M, n, axes) {
  const R = readout(M, n, axes);
  return `A box of the 150 Iris flowers on ${axisList(M, axes)}, in centimeters, colored by species. Problem ${n}: the two class averages are ${R.s4} cm apart in all four measurements.`;
}

/** The problem's short name for its button. */
export function buttonLabel(M, n) {
  const P = problem(M, n);
  const one = M.species[P.one[0]];
  const cap = one[0].toUpperCase() + one.slice(1);
  return P.other.length === 2 ? `${cap} vs. rest` : `${cap} vs. ${M.species[P.other[0]]}`;
}

// ---- the case's geometry (pure): where a flower sits in the box, in cm from the box's center ----

/** The box's half-sizes along across, up, and into the floor, for the three measurements drawn. */
export function halfSizes(axes) {
  const [x, z, y] = axes;
  return [x, y, z].map((j) => (SPANS[j][1] - SPANS[j][0]) / 2);
}

/** A point of four measurements in the box: x across, y up, z toward the viewer (larger depth farther). */
export function toBox(v, axes) {
  const [x, z, y] = axes;
  const mid = (j) => (SPANS[j][0] + SPANS[j][1]) / 2;
  return [v[x] - mid(x), v[y] - mid(y), -(v[z] - mid(z))];
}
