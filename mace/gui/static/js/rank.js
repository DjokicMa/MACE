/* Palette ranking and the small geometry the palette's calculator needs.
 * Pure, no DOM: haystacks are lowercased once in prepare(), so a keystroke
 * only scores, it never builds strings. Tests run it in the page. */

/** Subsequence match with a score (higher is better), or -1. Both strings
 * must already be lowercase (fuzzyScore lowercases for callers that have not). */
export function scoreLower(q, text) {
  if (!q) return 0;
  let qi = 0, score = 0, prev = -2, start = -1;
  const n = text.length, m = q.length;
  for (let i = 0; i < n && qi < m; i++) {
    if (text.charCodeAt(i) !== q.charCodeAt(qi)) continue;
    if (start < 0) start = i;
    const before = i === 0 ? 32 : text.charCodeAt(i - 1);
    // a word start: after a space, ":", "/", ".", "_", "-" or "("
    const boundary = i === 0 || before === 32 || before === 58 || before === 47 || before === 46
      || before === 95 || before === 45 || before === 40;
    // a contiguous run counts a little more than a new word start, so "sup"
    // prefers Supercell over Set Up
    score += 1 + (i === prev + 1 ? 5 : 0) + (boundary ? 4 : 0);
    prev = i; qi++;
  }
  if (qi < m) return -1;
  return score - start * 0.05 - n * 0.01;
}
export const fuzzyScore = (query, text) => scoreLower(String(query).toLowerCase(), String(text).toLowerCase());

const isBoundary = (text, i) => {
  if (i === 0) return true;
  const c = text.charCodeAt(i - 1);
  return c === 32 || c === 58 || c === 47 || c === 46 || c === 95 || c === 45 || c === 40;
};
/** Does `part` occur in `text` starting at a word boundary at or after `from`? The first such index, or -1. */
function atBoundary(text, part, from = 0) {
  for (let i = text.indexOf(part, from); i >= 0; i = text.indexOf(part, i + 1)) if (isBoundary(text, i)) return i;
  return -1;
}
/** Match quality (plan T9b), both strings lowercase: 0 strong (the query
 * starts a word as one contiguous run, or every character starts a word: an
 * acronym), 1 good (two contiguous runs, each starting a word: "sup" in
 * "set up"), 2 weak (any other subsequence), -1 no match. */
export function quality(q, text) {
  if (!q) return 0;
  if (atBoundary(text, q) >= 0) return 0;
  // an acronym: each character the start of a later word
  let at = 0, ok = true;
  for (let k = 0; k < q.length && ok; k++) {
    let found = -1;
    for (let i = at; i < text.length; i++) if (text.charCodeAt(i) === q.charCodeAt(k) && isBoundary(text, i)) { found = i; break; }
    if (found < 0) ok = false; else at = found + 1;
  }
  if (ok && q.length > 1) return 0;
  for (let cut = 1; cut < q.length; cut++) {
    const a = q.slice(0, cut), b = q.slice(cut).trimStart();
    if (!b) continue;
    for (let i = atBoundary(text, a); i >= 0; i = atBoundary(text, a, i + 1)) {
      if (atBoundary(text, b, i + a.length) >= 0) return 1;
    }
  }
  return scoreLower(q, text) >= 0 ? 2 : -1;
}
/** The best quality over an item's prepared haystacks (label, title, aliases). */
export function qualityOf(q, p) {
  let best = -1;
  for (const h of [p.title, p.label, ...p.aliases]) {
    if (!h) continue;
    const t = quality(q, h);
    if (t >= 0 && (best < 0 || t < best)) best = t;
    if (best === 0) break;
  }
  return best;
}

/** items: [{label, title?, aliases?: [], ...}] -> prepared haystacks (lowercase, once). */
export function prepare(items) {
  return items.map((it) => ({ it, label: it.label.toLowerCase(), title: (it.title || "").toLowerCase(),
    aliases: (it.aliases || []).map((a) => a.toLowerCase()) }));
}

/** Best-first hits: [{it, score}]. A title match outranks a category match;
 * an alias match counts like the label. */
export function rank(query, prepared, limit = Infinity) {
  const q = query.trim().toLowerCase();
  const out = [];
  for (const p of prepared) {
    let s = scoreLower(q, p.label);
    // a title match outranks a category match, and a title that starts with the query most
    if (p.title) { const t = scoreLower(q, p.title); const b = t >= 0 ? t + 1 + (p.title.startsWith(q) ? 3 : 0) : -1; if (b > s) s = b; }
    for (const a of p.aliases) { const x = scoreLower(q, a); if (x > s) s = x; }
    if (s >= 0) out.push({ it: p.it, score: s, p });
  }
  out.sort((a, b) => b.score - a.score || (a.it.label < b.it.label ? -1 : a.it.label > b.it.label ? 1 : 0));
  return out.length > limit ? out.slice(0, limit) : out;
}

/* ---------------------------------------------------------------- frac <-> Cart */
/** cell: rows a, b, c (Å). Cartesian = f0·a + f1·b + f2·c. */
export function fracToCart(f, cell) {
  return [0, 1, 2].map((k) => f[0] * cell[0][k] + f[1] * cell[1][k] + f[2] * cell[2][k]);
}
/** The inverse, by solving with the cell's inverse (null for a singular cell). */
export function cartToFrac(x, cell) {
  const [a, b, c] = cell;
  const det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
  if (Math.abs(det) < 1e-12) return null;
  // f = x · inv(M), M with rows a, b, c
  const inv = [
    [(b[1] * c[2] - b[2] * c[1]) / det, (a[2] * c[1] - a[1] * c[2]) / det, (a[1] * b[2] - a[2] * b[1]) / det],
    [(b[2] * c[0] - b[0] * c[2]) / det, (a[0] * c[2] - a[2] * c[0]) / det, (a[2] * b[0] - a[0] * b[2]) / det],
    [(b[0] * c[1] - b[1] * c[0]) / det, (a[1] * c[0] - a[0] * c[1]) / det, (a[0] * b[1] - a[1] * b[0]) / det],
  ];
  return [0, 1, 2].map((k) => x[0] * inv[0][k] + x[1] * inv[1][k] + x[2] * inv[2][k]);
}
/** "0.5 0 0.5", "1/3, 2/3, 0.25" -> three numbers, or null. */
export function parseTriple(text) {
  const parts = text.trim().split(/[\s,;]+/).filter(Boolean);
  if (parts.length !== 3) return null;
  const nums = parts.map((p) => {
    const m = p.match(/^([-+]?\d*\.?\d+(?:e[-+]?\d+)?)(?:\/([-+]?\d*\.?\d+))?$/i);
    if (!m) return NaN;
    return m[2] !== undefined ? Number(m[1]) / Number(m[2]) : Number(m[1]);
  });
  return nums.every(Number.isFinite) ? nums : null;
}
