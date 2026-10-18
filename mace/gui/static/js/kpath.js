/* A band k-path as a list of segments: the browser half of Crystal_d3/d3_band_path.py.
 *
 *   {from: "GAMMA", to: "Y"}                                   a point of the computed path
 *   {from: "Y", to: "P", k_to: [0.25, 0.25, 0]}                 a point you added (fractional k)
 *   {from: "K", to: "GAMMA", hidden: true}                      kept in the editor, not drawn/computed
 *
 * Two segments that do not meet are a jump ("A|L"); a run without one is a piece. Every
 * operation returns a new list. tests/test_kpath_parity.py runs the same edits through both
 * halves and compares the JSON. No DOM here: the chip strip is kpath_editor.js. */

export const MAX_SEGMENTS = 20;            // BAND's NLINE
const GREEK = { "Γ": "GAMMA", "Δ": "DELTA", "Σ": "SIGMA", "Λ": "LAMBDA", "Π": "PI" };
const LABEL = /^[A-Za-z][A-Za-z0-9_']{0,15}$/;
const SUBS = "₀₁₂₃₄₅₆₇₈₉";

export function canonical(label) {
  let s = String(label ?? "").trim();
  s = s.replace(/<sub>(.*?)<\/sub>/g, "_$1").replace(/[${}]/g, "");
  s = s.replace(/[₀-₉]/g, (c) => String(SUBS.indexOf(c)));
  for (const [g, n] of Object.entries(GREEK)) s = s.split(g).join(n);
  return s.toUpperCase() === "G" || s.toUpperCase() === "GAMMA" ? "GAMMA" : s;
}

/** "GAMMA" -> "Γ", "F_0" -> "F<sub>0</sub>", "DELTA_0" -> "Δ<sub>0</sub>" (HTML; the input is escaped). */
export function display(label) {
  const esc = (t) => t.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  let s = canonical(label);
  const m = s.match(/^([^_]*)(?:_(.*))?$/);
  let base = m ? m[1] : s;
  const sub = m && m[2] !== undefined ? m[2] : null;
  const G = { GAMMA: "Γ", DELTA: "Δ", SIGMA: "Σ", LAMBDA: "Λ", PI: "Π" };
  base = base.replace(/^(GAMMA|DELTA|SIGMA|LAMBDA|PI)/, (w) => G[w]);
  return esc(base) + (sub !== null ? `<sub>${esc(sub)}</sub>` : "");
}

/** Plain-text form for tooltips and aria labels: "Γ", "F₀". */
export function plain(label) {
  return display(label).replace(/<sub>(.*?)<\/sub>/g, (_, d) => d.replace(/[0-9]/g, (c) => SUBS[+c]))
    .replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"');
}

export function labelProblem(label) {
  const s = canonical(label);
  if (!s) return "a point needs a label";
  if (!LABEL.test(s)) return `${JSON.stringify(String(label))} is not a usable label (letters, digits, _ and ', up to 16 characters)`;
  return null;
}

const clone = (x) => JSON.parse(JSON.stringify(x));
const sameK = (a, b) => (a == null || b == null) ? true : a.every((v, i) => Math.abs(v - b[i]) < 1e-6);

export function joined(a, b) {
  return canonical(a.to) === canonical(b.from) && sameK(a.k_to, b.k_from);
}

export function pieces(segs) {
  const out = [];
  segs.forEach((s, i) => { if (out.length && joined(segs[out[out.length - 1].slice(-1)[0]], s)) out[out.length - 1].push(i); else out.push([i]); });
  return out;
}

export const visible = (segs) => segs.filter((s) => !s.hidden);

export function pathTokens(segs) {
  const v = visible(segs), out = [];
  v.forEach((s, i) => {
    if (i === 0) out.push(canonical(s.from), canonical(s.to));
    else if (joined(v[i - 1], s)) out.push(canonical(s.to));
    else out.push("|", canonical(s.from), canonical(s.to));
  });
  return out;
}

export function fromTokens(tokens, ks = null) {
  const segs = [];
  let run = [];
  for (const t of [...tokens, "|"]) {
    if (t === "|") { for (let i = 0; i + 1 < run.length; i++) segs.push({ from: canonical(run[i]), to: canonical(run[i + 1]) }); run = []; }
    else run.push(t);
  }
  if (ks && ks.length === segs.length) segs.forEach((s, i) => { s.k_from = ks[i].slice(0, 3).map(Number); s.k_to = ks[i].slice(3, 6).map(Number); });
  return segs;
}

/* ---------------------------------------------------------------- operations */
export function move(segs, i, j) {
  const out = clone(segs);
  if (i >= 0 && i < out.length && j >= 0 && j < out.length) out.splice(j, 0, out.splice(i, 1)[0]);
  return out;
}

export function reverse(segs, i) {
  const out = clone(segs), s = out[i], r = { from: s.to, to: s.from };
  if ("k_to" in s) r.k_from = s.k_to;
  if ("k_from" in s) r.k_to = s.k_from;
  if (s.hidden) r.hidden = true;
  out[i] = r;
  return out;
}

export function remove(segs, i) { const out = clone(segs); out.splice(i, 1); return out; }

export function toggleHidden(segs, i) {
  const out = clone(segs);
  if (out[i].hidden) delete out[i].hidden; else out[i].hidden = true;
  return out;
}

/** Remove the point at `end` ("from"/"to") of segment i. Inside a piece, `join` replaces its two
 * segments by one between their outer points ("Remove H: joins L→A", a path to compute); without it
 * both are hidden, leaving a jump ("Remove H: hides L→H and H→A", a computed path on display, where
 * no data runs between those points). A piece's end takes its one segment (removed, or hidden). */
export function removePoint(segs, i, end, join) {
  const out = clone(segs), aff = removal(out, i, end);
  if (!join) { aff.forEach((k) => { out[k].hidden = true; }); return out; }
  if (aff.length === 2) {
    const [a, b] = aff, m = { from: out[a].from, to: out[b].to };
    if ("k_from" in out[a]) m.k_from = out[a].k_from;
    if ("k_to" in out[b]) m.k_to = out[b].k_to;
    out.splice(a, 2, m);
  } else out.splice(aff[0], 1);
  return out;
}

/** The segments removing the point at `end` of segment i touches: the two meeting there, or i alone. */
export function removal(segs, i, end) {
  const j = end === "to" ? i + 1 : i - 1;
  const [a, b] = end === "to" ? [i, j] : [j, i];
  return a >= 0 && b >= 0 && a < segs.length && b < segs.length && joined(segs[a], segs[b]) ? [a, b] : [i];
}

export function mainPiece(segs) {
  const ps = pieces(segs);
  let best = 0;
  ps.forEach((p, n) => { if (p.length > ps[best].length) best = n; });
  return best;
}

const regroup = (segs, order) => order.flat().map((i) => clone(segs[i]));

export function jumpsFirst(segs) {
  const ps = pieces(segs), m = mainPiece(segs);
  return ps.length ? regroup(segs, [...ps.filter((_, n) => n !== m), ps[m]]) : [];
}

export function jumpsLast(segs) {
  const ps = pieces(segs), m = mainPiece(segs);
  return ps.length ? regroup(segs, [ps[m], ...ps.filter((_, n) => n !== m)]) : [];
}

export function continuousOnly(segs) {
  const ps = pieces(segs);
  return ps.length ? regroup(segs, [ps[mainPiece(segs)]]) : [];
}

export function movePiece(segs, n, m) {
  const ps = pieces(segs);
  if (!(n >= 0 && n < ps.length && m >= 0 && m < ps.length)) return clone(segs);
  ps.splice(m, 0, ps.splice(n, 1)[0]);
  return regroup(segs, ps);
}

export function addPoint(segs, label, k, fromLabel = null, fromK = null) {
  const out = clone(segs);
  const seg = {};
  if (fromLabel !== null) { seg.from = canonical(fromLabel); if (fromK) seg.k_from = fromK.map(Number); }
  else if (out.length) { const last = out[out.length - 1]; seg.from = last.to; if ("k_to" in last) seg.k_from = [...last.k_to]; }
  else { seg.from = "GAMMA"; seg.k_from = [0, 0, 0]; }
  seg.to = canonical(label);
  seg.k_to = k.map(Number);
  out.push(seg);
  return out;
}

export const OPS = { move, reverse, remove, toggle_hidden: toggleHidden, remove_point: removePoint, jumps_first: jumpsFirst,
  jumps_last: jumpsLast, continuous_only: continuousOnly, move_piece: movePiece, add_point: addPoint };

export function applyOps(segs, ops) { return ops.reduce((s, [name, ...args]) => OPS[name](s, ...args), clone(segs)); }

/* ---------------------------------------------------------------- checks */
export function points(segs) {
  const out = {};
  for (const s of segs) for (const e of ["from", "to"]) { const k = s[`k_${e}`], l = canonical(s[e]); if (k && !(l in out)) out[l] = k; }
  return out;
}

/** Why point `label` at fractional `k` cannot be written for a slab (dim 2) or a polymer (dim 1), or null:
 * the non-periodic components are formally zero (CRYSTAL23 manual p. 309, BAND note 3). As
 * d3_band_path.out_of_plane_problem. */
export function outOfPlaneProblem(label, k, dim) {
  if (!k || (dim !== 1 && dim !== 2)) return null;
  const bad = [0, 1, 2].filter((i) => i >= dim && Math.abs(k[i]) > 1e-9);
  if (!bad.length) return null;
  const kind = dim === 2 ? "a slab (2D)" : "a polymer (1D)";
  const comps = bad.map((i) => `k_${"xyz"[i]} = ${Number(k[i])}`).join(", ");
  return `${label} has ${comps}: in ${kind} ${dim === 2 ? "k_z" : "k_y and k_z"} must be 0 (CRYSTAL23 manual p. 309, `
    + "BAND note 3: the non-periodic components are formally input as zero)";
}

/** {errors, warnings}, the same rules as d3_band_path.validate (needK: the deck needs coordinates;
 * dim: 3, or 2 for a slab and 1 for a polymer). */
export function validate(segs, computed = null, needK = false, dim = 3) {
  const errors = [], warnings = [];
  const shown = visible(segs);
  if (!shown.length) return { errors: ["At least one segment must stay visible"], warnings };
  if (shown.length > MAX_SEGMENTS) errors.push(`BAND takes at most ${MAX_SEGMENTS} segments; this path has ${shown.length}`);
  const known = computed ? points(computed) : {};
  shown.forEach((s, n) => {
    for (const e of ["from", "to"]) {
      const p = labelProblem(s[e]);
      if (p) errors.push(`segment ${n + 1}: ${p}`);
      const k = s[`k_${e}`];
      if (k && !k.every((v) => Number.isFinite(v) && Math.abs(v) <= 10)) errors.push(`segment ${n + 1}: the coordinates of ${s[e]} must be numbers between -10 and 10`);
      else if (outOfPlaneProblem(s[e], k, dim)) errors.push(`segment ${n + 1}: ${outOfPlaneProblem(s[e], k, dim)}`);
      if (needK && !k && !(canonical(s[e]) in known)) errors.push(`segment ${n + 1}: ${s[e]} is not a point of the computed path; give its coordinates`);
    }
    const ka = s.k_from || known[canonical(s.from)], kb = s.k_to || known[canonical(s.to)];
    if ((canonical(s.from) === canonical(s.to) && (!ka || !kb || sameK(ka, kb))) || (ka && kb && sameK(ka, kb)))
      errors.push(`segment ${n + 1} (${plain(s.from)} → ${plain(s.to)}) has no length`);
  });
  if (computed && computed.length) {
    const used = new Set(shown.flatMap((s) => [canonical(s.from), canonical(s.to)]));
    const order = [];
    for (const s of computed) for (const e of ["from", "to"]) { const l = canonical(s[e]); if (!order.includes(l)) order.push(l); }
    const skipped = order.filter((l) => !used.has(l));
    if (skipped.length) warnings.push(`The path no longer visits ${skipped.map(plain).join(", ")} (high-symmetry points of the computed path); band extrema there would be missed`);
  }
  return { errors, warnings };
}

export function samePath(segs, computed) {
  if (!segs || segs.some((s) => s.hidden) || segs.length !== computed.length) return false;
  return segs.every((s, i) => canonical(s.from) === canonical(computed[i].from) && canonical(s.to) === canonical(computed[i].to)
    && ["from", "to"].every((e) => !s[`k_${e}`] || sameK(s[`k_${e}`], computed[i][`k_${e}`])));
}
