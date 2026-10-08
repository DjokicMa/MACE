/* Long folder paths, shortened to fit where they show (the welcome screen,
 * the Explorer's folder chip, the window title): the home folder as "~",
 * then the first segment and the last two with "/…/" between, then the last
 * one alone, and only then that last segment cut in the middle. The element
 * keeps the full path in its tooltip. Pure, plus one width helper. */

/** A named folder's prefix: "~" for HOME, "$SCRATCH" for the others. */
const aliasPrefix = (name) => (name === "HOME" ? "~" : "$" + name);

/** "~/…/runs/project" for a path longer than `max` characters. `aliases`
 * ([{name, path}] from the server: HOME and $SCRATCH, $WORK, $PSCRATCH when
 * set) shorten the start, the longest matching path first; `home` alone is
 * the older form. */
export function abbrev(full, { home = "", aliases = null, max = 60 } = {}) {
  let p = String(full || "");
  const list = (aliases && aliases.length ? aliases : home ? [{ name: "HOME", path: home }] : [])
    .map((a) => ({ name: a.name, path: String(a.path || "").replace(/\/+$/, "") }))
    .filter((a) => a.path)
    .sort((a, b) => b.path.length - a.path.length);
  const hit = list.find((a) => p === a.path || p.startsWith(a.path + "/"));
  if (hit) p = aliasPrefix(hit.name) + p.slice(hit.path.length);
  max = Math.max(4, Math.floor(max));
  if (p.length <= max) return p;
  const abs = p.startsWith("/");
  const segs = p.split("/").filter((s, k) => s || k === 0);
  const head = abs ? "/" + (segs[1] || "") : segs[0];
  const rest = abs ? segs.slice(2) : segs.slice(1);
  const last = rest.length ? rest[rest.length - 1] : head;
  if (rest.length >= 3) {
    const two = `${head}/…/${rest.slice(-2).join("/")}`;
    if (two.length <= max) return two;
  }
  if (rest.length >= 2) {
    const one = `${head}/…/${last}`;
    if (one.length <= max) return one;
  }
  if (rest.length === 1 && `${head}/${last}`.length <= max) return `${head}/${last}`;
  return middle(last, max);
}

/** "very_long_na…me_of_it": the middle replaced by an ellipsis. */
export function middle(text, max) {
  const s = String(text);
  if (s.length <= max) return s;
  const keep = Math.max(1, max - 1), a = Math.ceil(keep / 2), b = keep - a;
  return s.slice(0, a) + "…" + (b ? s.slice(-b) : "");
}

/* The width of one character ("0", the CSS ch unit) in an element's font,
 * measured once per font. */
const chCache = new Map();
let measureCv = null;
export function chars(el, pad = 0) {
  if (!el) return 60;
  const cs = getComputedStyle(el), font = `${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
  let w = chCache.get(font);
  if (!w) {
    measureCv = measureCv || document.createElement("canvas");
    const g = measureCv.getContext("2d");
    g.font = font;
    w = g.measureText("0").width || 7;
    chCache.set(font, w);
  }
  const box = el.clientWidth || (el.parentElement && el.parentElement.clientWidth) || 0;
  return box > 0 ? Math.max(8, Math.floor((box - pad) / w)) : 60;
}
