/* Set Coordinates… (spec §2.3): for the selection, absolute (one atom goes
 * there; several move so their centroid does), relative (a Δ for all) or
 * per axis (only the filled axes are set, for each atom); Frac or Cart.
 * In Sym mode the change applies to the symmetry equivalents (locked on).
 * A ghost shows the result; Apply is one set_positions or move step. */
import { loadCss, panel } from "./lazy_util.js";

export async function open(H) {
  await loadCss(import.meta.url, "../../css/dialogs.css");
  const G = await import("../geom/core.js");
  const { ed, E } = H.editor();
  if (!ed) return null;
  const idx = [...ed.selection].sort((a, b) => a - b);
  if (!idx.length) { H.toast("Select atoms first", "info"); return null; }
  const s = ed.structure, sym = ed.symMode === "sym";
  let mode = "absolute", frame = ed.coordMode;
  const body = document.createElement("div");
  body.innerHTML = `<p class="note">${idx.length === 1 ? `${H.esc(s.symbols[idx[0]])}${idx[0] + 1}` : `${idx.length} atoms`}</p>
    <div class="tabs" role="radiogroup" aria-label="Mode">${[["absolute", "Absolute"], ["relative", "Relative"], ["axis", "Per axis"]].map(([v, t]) =>
      `<button class="tb-btn-text" role="radio" data-mode="${v}" aria-checked="${v === mode}" aria-selected="${v === mode}">${t}</button>`).join("")}</div>
    <div class="tabs" role="radiogroup" aria-label="Coordinates">${[["frac", "Frac"], ["cart", "Cart"]].map(([v, t]) =>
      `<button class="tb-btn-text" role="radio" data-frame="${v}" aria-checked="${v === frame}" aria-selected="${v === frame}">${t}</button>`).join("")}</div>
    <div class="grid3">${["x", "y", "z"].map((k) => `<label class="f">${k}<input class="field" data-ax="${k}" inputmode="decimal" aria-label="Set ${k}"></label>`).join("")}</div>
    <div class="row"><label class="tog"><input type="checkbox" class="sc-eq"${sym ? " checked" : ""} disabled><span>Apply to symmetry equivalents</span></label></div>
    <p class="note">${sym ? "Sym mode: the equivalents follow, always." : "P1 mode: only the selected atoms change."}</p>
    <div class="err" role="alert"></div>`;
  const inputs = ["x", "y", "z"].map((k) => body.querySelector(`[data-ax="${k}"]`));
  const err = body.querySelector(".err");
  const cur = (i) => (frame === "cart" ? s.cart[i] : s.frac[i]);
  const centroid = () => [0, 1, 2].map((k) => idx.reduce((t, i) => t + cur(i)[k], 0) / idx.length);
  const d0 = Number(H.settings.get("decimals.coord")) || 5;
  const fill = () => {
    const v = mode === "relative" ? [0, 0, 0] : mode === "axis" ? null : idx.length === 1 ? cur(idx[0]) : centroid();
    inputs.forEach((el, k) => { el.value = v ? G.fmtNum(v[k], d0) : ""; el.placeholder = mode === "axis" ? "keep" : ""; });
    preview();
  };
  /** {indices, frac: [[...]]} for the result, or {error}. */
  const result = () => {
    const vals = inputs.map((el) => (el.value.trim() === "" ? null : G.evalExpr(el.value)));
    const bad = vals.findIndex((r) => r && !r.ok);
    if (bad >= 0) return { error: `${"xyz"[bad]}: ${vals[bad].error}` };
    const v = vals.map((r) => (r ? r.value : null));
    if (mode !== "axis" && v.some((x) => x === null)) return { error: "Give x, y and z" };
    const c = centroid();
    const target = idx.map((i) => {
      const p = cur(i).slice();
      if (mode === "absolute") return p.map((x, k) => x + v[k] - c[k]);
      if (mode === "relative") return p.map((x, k) => x + v[k]);
      return p.map((x, k) => (v[k] === null ? x : v[k]));
    });
    return { frac: frame === "cart" ? target.map((x) => G.fracOf(s.cell, x)) : target };
  };
  const preview = () => {
    const r = result();
    err.textContent = r.error || "";
    inputs.forEach((el) => el.removeAttribute("aria-invalid"));
    if (r.error) { E.scene().setDrag(null); return; }
    E.scene().setDrag(idx.map((i, n) => [i, G.cartOf(s.cell, r.frac[n])]));
  };
  body.addEventListener("input", preview);
  body.querySelectorAll("[data-mode]").forEach((b) => b.addEventListener("click", () => {
    mode = b.dataset.mode;
    body.querySelectorAll("[data-mode]").forEach((x) => { x.setAttribute("aria-checked", String(x === b)); x.setAttribute("aria-selected", String(x === b)); });
    fill();
  }));
  body.querySelectorAll("[data-frame]").forEach((b) => b.addEventListener("click", () => {
    frame = b.dataset.frame;
    body.querySelectorAll("[data-frame]").forEach((x) => { x.setAttribute("aria-checked", String(x === b)); x.setAttribute("aria-selected", String(x === b)); });
    fill();
  }));
  const d = panel(H, { id: "set-coords", title: "Set Coordinates", body,
    foot: `<button class="tb-btn-text" data-act="cancel">Cancel</button><button class="tb-btn-text primary" data-act="apply">Apply</button>`,
    onClose: () => E.scene().setDrag(null) });
  d.querySelector('[data-act="cancel"]').addEventListener("click", () => d.close());
  d.querySelector('[data-act="apply"]').addEventListener("click", () => {
    const r = result();
    if (r.error) { err.textContent = r.error; return; }
    if (mode === "relative" || (mode === "absolute" && idx.length > 1)) {
      const df = G.sub3(r.frac[0], s.frac[idx[0]]);
      E.api.edit(ed, "move", { indices: idx, frac_delta: df });
    } else E.api.edit(ed, "set_positions", { indices: idx, frac: r.frac });
    d.close();
  });
  fill();
  return d;
}
