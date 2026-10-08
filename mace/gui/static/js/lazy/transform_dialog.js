/* Build › Transform Cell… (plan T7): spglib's primitive or conventional
 * standard cell, or the Niggli-reduced cell. The preview draws the new cell's
 * outline over the atoms, its atoms as ghosts and its a′ b′ c′ axes; the
 * readout says what changes ("8 → 2 atoms · a 4.212 → 2.978 Å · Fm-3m
 * kept"). In place by default (one undo step "Transform to Primitive"). */
import { openBuilder, fmtN, stemOf } from "./builder_util.js";

const KINDS = [["primitive", "Primitive", "prim"], ["conventional", "Conventional", "conv"], ["niggli", "Niggli reduced", "niggli"]];

export function open(H, opts = {}) {
  let kind = opts.kind || "primitive", last = null;
  return openBuilder(H, {
    id: "transform", title: "Transform Cell", route: "build/standardize", defaultResult: "inplace", primary: "Create",
    render(host, api) {
      host.innerHTML = `<div class="bld-row"><span>Cell</span><span class="segment" role="radiogroup" aria-label="Cell">${KINDS.map(([k, t]) =>
        `<button type="button" role="radio" data-k="${k}" aria-checked="${k === kind}">${t}</button>`).join("")}</span></div>
        <p class="bld-note" data-k="note"></p>`;
      host.querySelectorAll("[data-k]").forEach((b) => b.tagName === "BUTTON" && b.addEventListener("click", () => {
        kind = b.dataset.k;
        host.querySelectorAll("button[data-k]").forEach((x) => x.setAttribute("aria-checked", String(x.dataset.k === kind)));
        last = null; api.changed();
      }));
    },
    params(api) {
      const d = api.d, s = api.structure(), t = KINDS.find((k) => k[0] === kind);
      api.frame.params.querySelector('[data-k="note"]').textContent = kind === "niggli"
        ? "The shortest, most orthogonal cell of the same lattice (Niggli); the atoms are mapped into it"
        : "spglib's standard setting of the space group";
      const L = s.lattice, y = s.symmetry || {};
      const ro = last ? `<p><b>${fmtN(last.before)}</b> → <b>${fmtN(last.n)}</b> atoms · a <b>${L.a.toFixed(3)}</b> → <b>${last.a.toFixed(3)}</b> Å · ${
        y.available ? `${H.esc(y.international)} kept` : "group unknown"}</p>` : "<p>…</p>";
      const name = `${stemOf(d.title)}_${t[2]}`;
      return { params: { kind, symprec: api.ed ? api.ed.symprec : 1e-3 }, n: last ? last.n : s.numbers.length, name,
        label: `Transform to ${t[1]} of ${d.title}`, readout: ro,
        inPlace: { op: "transform", args: { kind }, label: `Transform to ${t[1]}` } };
    },
    drawPreview(r, api, core) {
      if (r.preview || r.cell) {
        const c = r.cell, a = Math.hypot(...c[0]);
        const n0 = api.structure().numbers.length;
        const changed = !last || last.n !== r.n || Math.abs(last.a - a) > 1e-9;
        last = { n: r.n, a, before: n0 };
        if (changed) setTimeout(() => api.changed(), 0);
      }
      return { kind: "transform", atoms: r.preview ? core.decodePacked(r.preview) : null, cell: r.cell, triad: true };
    },
  });
}
