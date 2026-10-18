/* The k-path editor: one component for the band plot's options (a computed path on display) and
 * Calculation Setup (a path to compute). The path reads like the plot's k axis, a strip of chips:
 *
 *   (Γ) → (M) → (K) → (Γ) → (A) → (L) → (H) → (A)  [| A|L ×]  (L) → (M)  [| M|H ×]  (H) → (K)
 *
 * A point is a chip (× removes it); the arrow between two points is the segment: click it to
 * reverse it, the eye hides it, × removes it, drag it to reorder. A jump "|" is a chip too: drag
 * it to move the part after it, × drops that part. A selected segment or jump moves with ←/→,
 * Delete removes it, R reverses, H hides. Undo is the host's (Mod+Z in the plot and in Setup).
 *
 * render(host, opts) draws from opts every time; the host calls it again after each change.
 * opts = { id, mode: "display" | "calc", segs, computed, dimensionality (2 slab, 1 polymer), onChange(segs | null, label), icon(name),
 *          presets?: { list() -> [{name, ok}], apply(name), save(name), remove(name) },
 *          serverErrors?: [], serverWarnings?: [] }
 * onChange(null) is "back to the computed path". */
import * as K from "./kpath.js";
import { loadCss } from "./lazy/lazy_util.js";

export const ready = loadCss(import.meta.url, "../css/kpath.css");
const sel = new Map();          // editor id -> {kind: "seg" | "gap", i}
const refocus = new Set();      // editors whose selection takes the focus back after the host draws them again
const esc = (t) => String(t ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/** The next draw of this editor puts the focus back on its selected chip (an undo from the keyboard). */
export function keepFocus(id) { refocus.add(id); }

export function render(host, opts) {
  host.__kp = opts;
  hosts.set(opts.id, host);
  if (!host.__kpBound) { bind(host); host.__kpBound = true; }
  const { mode, computed, icon } = opts;
  const segs = opts.segs || computed;
  const edited = !!opts.segs && !K.samePath(opts.segs, computed);
  const calc = mode === "calc";
  const v = K.validate(segs, computed, false, opts.dimensionality || 3);
  const errors = [...(opts.serverErrors || []), ...v.errors.filter((e) => !(opts.serverErrors || []).includes(e))];
  const warnings = calc ? [...new Set([...v.warnings, ...(opts.serverWarnings || [])])] : [];
  const ps = K.pieces(segs);
  const s = sel.get(opts.id);
  const pt = (seg, i, end, custom) => {
    const lab = seg[end];
    const what = removalText(segs, i, end, calc);
    return `<span class="kp-pt${custom ? " custom" : ""}" role="listitem" aria-label="Point ${esc(K.plain(lab))}">${K.display(lab)}` +
      `<button type="button" class="kp-btn kp-xpt" data-i="${i}" data-end="${end}" data-tip="${esc(what)}" aria-label="${esc(what)}">${icon("x")}</button></span>`;
  };
  let strip = "";
  ps.forEach((p, n) => {
    if (n) {
      const a = segs[ps[n - 1][ps[n - 1].length - 1]], b = segs[p[0]];
      const on = s && s.kind === "gap" && s.i === n;
      strip += `<span class="kp-gap${on ? " sel" : ""}" role="listitem" tabindex="0" draggable="true" data-piece="${n}"
        aria-label="Jump ${esc(K.plain(a.to))} | ${esc(K.plain(b.from))}: arrow keys move the part after it, Delete drops it"
        data-tip="Jump ${esc(K.plain(a.to))}|${esc(K.plain(b.from))}: drag to move the part after it, × drops it">${K.display(a.to)}<b>|</b>${K.display(b.from)}` +
        `<button type="button" class="kp-btn kp-xgap" data-piece="${n}" data-tip="Drop the part after this jump (${esc(p.map((i) => `${K.plain(segs[i].from)}→${K.plain(segs[i].to)}`).join(", "))})" aria-label="Drop the part after the jump">${icon("x")}</button></span>`;
    }
    strip += pt(segs[p[0]], p[0], "from", "k_from" in segs[p[0]] && !(K.canonical(segs[p[0]].from) in K.points(computed)));
    for (const i of p) {
      const g = segs[i], on = s && s.kind === "seg" && s.i === i;
      const name = `${K.plain(g.from)} → ${K.plain(g.to)}`;
      strip += `<span class="kp-seg${g.hidden ? " kp-off" : ""}${on ? " sel" : ""}" role="listitem" tabindex="0" draggable="true" data-i="${i}"
        aria-label="Segment ${i + 1} of ${segs.length}: ${esc(name)}${g.hidden ? ", hidden" : ""}. Arrow keys move it, R reverses, H hides, Delete removes">` +
        `<button type="button" class="kp-btn kp-rev" data-i="${i}" tabindex="-1" data-tip="Reverse ${esc(name)} (R)" aria-label="Reverse ${esc(name)}">→</button>` +
        (g.hidden ? `<span class="kp-offmark" aria-hidden="true">${icon("eye-off")}</span>` : "") +
        // the eye and × float under the arrow (under the pointer, or selected): the strip never reflows
        `<span class="kp-tools"><button type="button" class="kp-btn kp-eye" data-i="${i}" tabindex="-1" data-tip="${g.hidden ? "Show" : "Hide"} ${esc(name)} (H)" aria-label="${g.hidden ? "Show" : "Hide"} ${esc(name)}" aria-pressed="${!!g.hidden}">${icon(g.hidden ? "eye-off" : "eye")}</button>` +
        `<button type="button" class="kp-btn kp-x" data-i="${i}" tabindex="-1" data-tip="Remove ${esc(name)} (Delete)" aria-label="Remove ${esc(name)}">${icon("x")}</button></span></span>`;
      strip += pt(g, i, "to", "k_to" in g && !(K.canonical(g.to) in K.points(computed)));
    }
  });
  const help = calc ? "Drag an arrow or a jump to reorder. Click an arrow to reverse that segment; point at it for the eye (leave out) and ×. The deck preview follows."
    : "Drag an arrow or a jump to reorder. Click an arrow to reverse that segment; point at it for the eye (hide) and ×. The plot follows.";
  const msgs = errors.map((e) => `<p class="kp-msg err" role="alert">${icon("triangle-alert")}<span>${esc(e)}</span></p>`).join("")
    + warnings.map((w) => `<p class="kp-msg warn" role="status">${icon("triangle-alert")}<span>${esc(w)}</span></p>`).join("");
  const many = ps.length > 1;
  let h = `<div class="kp" data-kp="${esc(opts.id)}">
    <div class="kp-head"><span class="kp-title${opts.heading === false ? " sr-only" : ""}" id="kp-t-${esc(opts.id)}">k-path</span><span class="kp-state${edited ? " edited" : ""}">${edited ? "edited" : "as computed"} · ${K.visible(segs).length} of ${segs.length} segment${segs.length === 1 ? "" : "s"} shown</span></div>
    <p class="kp-help">${help}</p>
    <div class="kp-strip" role="list" aria-labelledby="kp-t-${esc(opts.id)}">${strip}</div>${msgs}
    <div class="kp-acts">
      <button type="button" data-kp-act="jumps_first"${many ? "" : " disabled"} data-tip="Put the parts after a jump (|) before the main path">Jumps first</button>
      <button type="button" data-kp-act="jumps_last"${many ? "" : " disabled"} data-tip="Put the parts after a jump (|) after the main path">Jumps last</button>
      <button type="button" data-kp-act="continuous_only"${many ? "" : " disabled"} data-tip="${esc(keepText(segs))}">Continuous only</button>
      <button type="button" class="${edited ? "primary" : ""}" data-kp-act="reset"${edited || opts.segs ? "" : " disabled"} data-tip="Back to the path as computed">Reset to computed path</button></div>`;
  if (calc) {
    h += `<form class="kp-add" data-kp-add novalidate aria-label="Add a point">
      <span class="kp-sub">Add point</span>
      <input class="lab" name="label" placeholder="label" aria-label="Label of the new point" spellcheck="false" autocomplete="off">
      <input class="k" name="kx" placeholder="k1" aria-label="First fractional coordinate" inputmode="decimal">
      <input class="k" name="ky" placeholder="k2" aria-label="Second fractional coordinate" inputmode="decimal">
      <input class="k" name="kz" placeholder="k3" aria-label="Third fractional coordinate" inputmode="decimal">
      <button type="submit" data-tip="Add a segment from the last point to this one (fractional coordinates of the deck's cell)">Add point</button>
      <span class="kp-msg err kp-addmsg" role="alert"></span></form>`;
    const used = new Set(K.visible(segs).flatMap((g) => [K.canonical(g.from), K.canonical(g.to)]));
    const pts = Object.entries({ ...K.points(computed), ...K.points(segs) }).filter(([l]) => used.has(l));
    if (pts.length) h += `<details class="kp-ptlist"><summary class="kp-sub">Points and their coordinates</summary><ul class="kp-pts">${
      pts.map(([l, k]) => `<li><span>${K.display(l)}</span><span>${k.map((x) => fmt(x)).join("  ")}</span></li>`).join("")}</ul></details>`;
  }
  if (opts.presets) {
    const list = opts.presets.list();
    h += `<div class="kp-presets"><span class="kp-sub">Presets</span>
      <select data-kp-preset aria-label="k-path preset">${list.length ? list.map((p) => `<option value="${esc(p.name)}"${p.ok ? "" : " disabled"}>${esc(p.name)}${p.ok ? "" : " (other points)"}</option>`).join("") : '<option value="">none saved</option>'}</select>
      <button type="button" data-kp-act="preset_apply"${list.some((p) => p.ok) ? "" : " disabled"} data-tip="Use this preset on this plot">Apply</button>
      <button type="button" data-kp-act="preset_remove"${list.length ? "" : " disabled"} data-tip="Delete this preset">Delete</button>
      <input data-kp-preset-name placeholder="name" aria-label="Preset name" spellcheck="false">
      <button type="button" data-kp-act="preset_save" data-tip="Save this path as a preset for any plot with the same points">Save as preset</button></div>`;
  }
  if (opts.bz) h += '<div class="kp-bzslot"></div>';
  h += "</div>";
  host.innerHTML = h;
  if (opts.bz) { opts.bz.attach(host.querySelector(".kp-bzslot")); opts.bz.update(segs); }
  if (refocus.has(opts.id)) {
    const t = (s && (s.kind === "seg" ? host.querySelector(`.kp-seg[data-i="${s.i}"]`) : host.querySelector(`.kp-gap[data-piece="${s.i}"]`)))
      || host.querySelector(".kp-seg");
    if (t) t.focus({ preventScroll: true });
    refocus.delete(opts.id);
  }
}

/** "Remove H: hides L → H and H → A" (a plot) / "Remove H: joins L → A" (a path to compute). */
export function removalText(segs, i, end, calc) {
  const lab = K.plain(segs[i][end]), aff = K.removal(segs, i, end);
  const nm = (g) => `${K.plain(g.from)} → ${K.plain(g.to)}`;
  if (calc) return aff.length === 2 ? `Remove ${lab}: joins ${K.plain(segs[aff[0]].from)} → ${K.plain(segs[aff[1]].to)}` : `Remove ${lab}: removes ${nm(segs[aff[0]])}`;
  return `Remove ${lab}: hides ${aff.map((k) => nm(segs[k])).join(" and ")}`;
}

/** "Keeps the longest continuous part: Γ → M → K → Γ → A → L → H → A". */
export function keepText(segs) {
  const ps = K.pieces(segs);
  if (!ps.length) return "";
  const p = ps[K.mainPiece(segs)];
  return `Keeps the longest continuous part: ${[segs[p[0]].from, ...p.map((k) => segs[k].to)].map(K.plain).join(" → ")}; every jump (|) and what follows it goes`;
}

/* ---------------------------------------------------------------- linked highlights
 * The plot and the Brillouin zone view show what a control would do before it is used: the
 * segments a point's × hides or joins, the part Continuous only keeps; and the segment under the
 * pointer, in the strip or in the zone view, is marked in both. */
const hosts = new Map();        // editor id -> host element (the one drawn last)
export function highlight(id, indices, kind = "hover") {
  const host = hosts.get(id);
  if (!host || !host.isConnected) return;
  host.querySelectorAll(".kp-seg.kp-hl, .kp-seg.kp-prev, .kp-seg.kp-dim").forEach((x) => x.classList.remove("kp-hl", "kp-prev", "kp-dim"));
  if (!indices) return;
  const set = new Set(indices);
  host.querySelectorAll(".kp-seg").forEach((x) => {
    const on = set.has(+x.dataset.i);
    if (kind === "keep") x.classList.toggle("kp-dim", !on);
    else if (on) x.classList.add(kind === "hover" ? "kp-hl" : "kp-prev");
  });
}
function preview(host, indices, kind) {
  const o = host.__kp;
  highlight(o.id, indices, kind);
  if (o.bz) o.bz.highlight(indices, kind);
  if (o.onPreview) o.onPreview(indices ? { indices, kind } : null);
}

/* the Brillouin zone view's side of the link */
/** The segment under the pointer in the zone view (null: none): marked in the strip and on the plot. */
export function hoverFromBZ(id, i) {
  const host = hosts.get(id);
  if (host && host.isConnected) preview(host, i === null || i === undefined ? null : [i], "hover");
}
/** A click on a segment in the zone view selects its chip. */
export function selectFromBZ(id, i) {
  const host = hosts.get(id);
  if (!host || !host.isConnected) return;
  sel.set(id, { kind: "seg", i });
  markSel(host);
}
/** A click on a point in the zone view (Calculation Setup): a segment from the path's last point to it. */
export function pickFromBZ(id, label, point) {
  const host = hosts.get(id);
  if (!host || !host.isConnected) return;
  const o = host.__kp, segs = o.segs || o.computed;
  const last = segs.length ? segs[segs.length - 1] : null;
  if (last && K.canonical(last.to) === K.canonical(label)) { flash(host, `The path already ends at ${K.plain(label)}`); return; }
  const labs = new Set(o.computed.flatMap((g) => [K.canonical(g.from), K.canonical(g.to)]));
  const known = labs.has(K.canonical(label)) || !point;
  const seg = { from: last ? last.to : "GAMMA", to: K.canonical(label) };
  if (last && "k_to" in last) seg.k_from = last.k_to;
  if (!known) seg.k_to = point.k.slice();
  change(host, [...segs, seg], `Add point ${K.plain(label)}`);
}

const fmt = (x) => { const r = Math.round(x * 1e6) / 1e6; return String(r); };

function isInner(segs, i, end) {
  const j = end === "to" ? i + 1 : i - 1;
  return j >= 0 && j < segs.length && (end === "to" ? K.joined(segs[i], segs[j]) : K.joined(segs[j], segs[i]));
}

function change(host, segs, label, keepSel = null) {
  const o = host.__kp;
  if (segs && !K.visible(segs).length) { flash(host, "At least one segment must stay visible"); return; }
  if (keepSel) { sel.set(o.id, keepSel); refocus.add(o.id); }
  o.onChange(segs && K.samePath(segs, o.computed) ? null : segs, label);
}

function flash(host, text) {
  const strip = host.querySelector(".kp-strip");
  if (!strip) return;
  let m = host.querySelector(".kp-flash");
  if (!m) { m = document.createElement("p"); m.className = "kp-msg err kp-flash"; m.setAttribute("role", "alert"); strip.after(m); }
  m.innerHTML = `${host.__kp.icon("triangle-alert")}<span>${esc(text)}</span>`;
}

function bind(host) {
  const cur = () => host.__kp.segs || host.__kp.computed;
  const name = (g) => `${K.plain(g.from)} → ${K.plain(g.to)}`;
  host.addEventListener("click", (e) => {
    const o = host.__kp, segs = cur();
    const b = e.target.closest("button");
    const segEl = e.target.closest(".kp-seg"), gapEl = e.target.closest(".kp-gap");
    if (!b) {
      if (segEl) { sel.set(o.id, { kind: "seg", i: +segEl.dataset.i }); markSel(host); }
      else if (gapEl) { sel.set(o.id, { kind: "gap", i: +gapEl.dataset.piece }); markSel(host); }
      return;
    }
    const i = b.dataset.i !== undefined ? +b.dataset.i : null;
    if (b.classList.contains("kp-rev")) return change(host, K.reverse(segs, i), `Reverse ${name(segs[i])}`, { kind: "seg", i });
    if (b.classList.contains("kp-eye")) return change(host, K.toggleHidden(segs, i), `${segs[i].hidden ? "Show" : "Hide"} ${name(segs[i])}`, { kind: "seg", i });
    if (b.classList.contains("kp-x")) return change(host, K.remove(segs, i), `Remove ${name(segs[i])}`);
    if (b.classList.contains("kp-xpt")) {
      const lab = K.plain(segs[i][b.dataset.end]);
      return change(host, K.removePoint(segs, i, b.dataset.end, o.mode === "calc"), `Remove point ${lab}`);
    }
    if (b.classList.contains("kp-xgap")) {
      const n = +b.dataset.piece, ps = K.pieces(segs);
      return change(host, segs.filter((_, k) => !ps[n].includes(k)), `Drop the part after a jump`);
    }
    const act = b.dataset.kpAct;
    if (!act) return;
    if (act === "reset") { sel.delete(o.id); return o.onChange(null, "Reset the k-path"); }
    if (act === "jumps_first") return change(host, K.jumpsFirst(segs), "Jumps first");
    if (act === "jumps_last") return change(host, K.jumpsLast(segs), "Jumps last");
    if (act === "continuous_only") return change(host, K.continuousOnly(segs), "Continuous path only");
    const p = o.presets;
    if (p && act === "preset_apply") { const v = host.querySelector("[data-kp-preset]").value; if (v) p.apply(v); return; }
    if (p && act === "preset_remove") { const v = host.querySelector("[data-kp-preset]").value; if (v) p.remove(v); return; }
    if (p && act === "preset_save") {
      const inp = host.querySelector("[data-kp-preset-name]"), v = inp.value.trim();
      if (!v) { inp.setAttribute("aria-invalid", "true"); inp.focus(); flash(host, "Give the preset a name"); return; }
      p.save(v);
    }
  });
  const previewOf = (t) => {
    const o = host.__kp, segs = cur();
    const x = t.closest && t.closest(".kp-xpt");
    if (x) return [K.removal(segs, +x.dataset.i, x.dataset.end), "affected"];
    const c = t.closest && t.closest('[data-kp-act="continuous_only"]');
    if (c && !c.disabled) { const ps = K.pieces(segs); return ps.length ? [ps[K.mainPiece(segs)], "keep"] : null; }
    const g = t.closest && t.closest(".kp-seg");
    if (g) return [[+g.dataset.i], "hover"];
    return o ? null : null;
  };
  const enter = (e) => { const p = previewOf(e.target); if (p) preview(host, p[0], p[1]); };
  const leave = (e) => { if (previewOf(e.target) && !(e.relatedTarget && host.contains(e.relatedTarget) && previewOf(e.relatedTarget))) preview(host, null); };
  host.addEventListener("pointerover", enter);
  host.addEventListener("pointerout", leave);
  host.addEventListener("focusin", enter);
  host.addEventListener("focusout", leave);
  host.addEventListener("submit", (e) => {
    const f = e.target.closest("[data-kp-add]");
    if (!f) return;
    e.preventDefault();
    const o = host.__kp, segs = cur();
    const msg = f.querySelector(".kp-addmsg");
    f.querySelectorAll("input").forEach((x) => x.removeAttribute("aria-invalid"));
    const lab = f.label.value.trim();
    const bad = K.labelProblem(lab);
    const ks = [f.kx, f.ky, f.kz].map((x) => ({ x, v: x.value.trim() === "" ? NaN : Number(eval1(x.value)) }));
    const wrong = ks.filter((k) => !Number.isFinite(k.v) || Math.abs(k.v) > 10);
    if (bad) { f.label.setAttribute("aria-invalid", "true"); msg.textContent = bad; f.label.focus(); return; }
    if (wrong.length) { wrong.forEach((k) => k.x.setAttribute("aria-invalid", "true")); msg.textContent = "Coordinates are three numbers between -10 and 10 (fractions such as 1/3 work)"; wrong[0].x.focus(); return; }
    const k = ks.map((x) => x.v);
    const known = K.points(o.computed)[K.canonical(lab)];
    if (known && !known.every((v, n) => Math.abs(v - k[n]) < 1e-6)) {
      f.label.setAttribute("aria-invalid", "true");
      msg.textContent = `${K.plain(lab)} is already a point of this path, at ${known.map(fmt).join(" ")}: use another label`;
      return;
    }
    change(host, K.addPoint(segs, lab, k), `Add point ${K.plain(lab)}`);
  });
  host.addEventListener("keydown", (e) => {
    const o = host.__kp, segs = cur();
    const segEl = e.target.closest(".kp-seg"), gapEl = e.target.closest(".kp-gap");
    if (!segEl && !gapEl) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;           // Mod+Z stays the host's undo
    const key = e.key;
    if (segEl) {
      const i = +segEl.dataset.i;
      sel.set(o.id, { kind: "seg", i });
      if (key === "ArrowLeft" || key === "ArrowRight") {
        const j = i + (key === "ArrowLeft" ? -1 : 1);
        if (j >= 0 && j < segs.length) change(host, K.move(segs, i, j), `Move ${name(segs[i])}`, { kind: "seg", i: j });
      } else if (key === "Delete" || key === "Backspace") change(host, K.remove(segs, i), `Remove ${name(segs[i])}`, { kind: "seg", i: Math.max(0, Math.min(i, segs.length - 2)) });
      else if (key === "r" || key === "R") change(host, K.reverse(segs, i), `Reverse ${name(segs[i])}`, { kind: "seg", i });
      else if (key === "h" || key === "H") change(host, K.toggleHidden(segs, i), `${segs[i].hidden ? "Show" : "Hide"} ${name(segs[i])}`, { kind: "seg", i });
      else return;
    } else {
      const n = +gapEl.dataset.piece, ps = K.pieces(segs);
      sel.set(o.id, { kind: "gap", i: n });
      if (key === "ArrowLeft" || key === "ArrowRight") {
        const m = n + (key === "ArrowLeft" ? -1 : 1);
        if (m >= 0 && m < ps.length) change(host, K.movePiece(segs, n, m), "Move the part after a jump", { kind: "gap", i: Math.max(1, m) });
      } else if (key === "Delete" || key === "Backspace") change(host, segs.filter((_, k) => !ps[n].includes(k)), "Drop the part after a jump");
      else return;
    }
    e.preventDefault();
    e.stopPropagation();
  });
  // drag and drop: a segment onto a segment (it takes that place); a jump (the part after it) onto
  // any chip of another part (it takes that part's place)
  let drag = null;
  host.addEventListener("dragstart", (e) => {
    const segEl = e.target.closest(".kp-seg"), gapEl = e.target.closest(".kp-gap");
    if (!segEl && !gapEl) return;
    drag = segEl ? { kind: "seg", i: +segEl.dataset.i } : { kind: "gap", i: +gapEl.dataset.piece };
    (segEl || gapEl).classList.add("dragging");
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData("text/plain", "k-path");
  });
  const target = (e) => e.target.closest(".kp-seg, .kp-gap");
  host.addEventListener("dragover", (e) => {
    if (!drag) return;
    const t = target(e);
    host.querySelectorAll(".drop-before").forEach((x) => x.classList.remove("drop-before"));
    if (!t) return;
    e.preventDefault();
    t.classList.add("drop-before");
  });
  host.addEventListener("dragend", () => { drag = null; host.querySelectorAll(".dragging, .drop-before").forEach((x) => x.classList.remove("dragging", "drop-before")); });
  host.addEventListener("drop", (e) => {
    const t = target(e);
    if (!drag || !t) return;
    e.preventDefault();
    const segs = cur(), ps = K.pieces(segs), d = drag;
    drag = null;
    const pieceOf = (el) => (el.classList.contains("kp-gap") ? +el.dataset.piece : ps.findIndex((p) => p.includes(+el.dataset.i)));
    if (d.kind === "seg") {
      const j = t.classList.contains("kp-seg") ? +t.dataset.i : ps[+t.dataset.piece][0];
      if (j !== d.i) change(host, K.move(segs, d.i, j), `Move ${name(segs[d.i])}`, { kind: "seg", i: j });
    } else {
      const m = pieceOf(t);
      if (m >= 0 && m !== d.i) change(host, K.movePiece(segs, d.i, m), "Move the part after a jump", m > 0 ? { kind: "gap", i: m } : null);
    }
  });
}

function markSel(host) {
  const s = sel.get(host.__kp.id);
  host.querySelectorAll(".kp-seg.sel, .kp-gap.sel").forEach((x) => x.classList.remove("sel"));
  if (!s) return;
  const t = s.kind === "seg" ? host.querySelector(`.kp-seg[data-i="${s.i}"]`) : host.querySelector(`.kp-gap[data-piece="${s.i}"]`);
  if (t) { t.classList.add("sel"); t.focus({ preventScroll: true }); }
}

/** "1/3" -> 0.333…, "0.5" -> 0.5; anything else NaN (no eval). */
function eval1(t) {
  const s = String(t).trim();
  const m = s.match(/^(-?\d+(?:\.\d+)?)\s*\/\s*(\d+(?:\.\d+)?)$/);
  if (m) return Number(m[1]) / Number(m[2]);
  return /^-?(\d+\.?\d*|\.\d+)(e-?\d+)?$/i.test(s) ? Number(s) : NaN;
}
