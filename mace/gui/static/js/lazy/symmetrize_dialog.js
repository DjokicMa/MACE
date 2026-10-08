/* Build › Symmetrize… (spec §2.5, plan T7): the space group spglib finds at
 * 13 tolerances from 1e-5 to 0.1 Å (the stops where it changes in bold);
 * picking one previews the refined structure as ghosts, with arrows from
 * each atom to where it goes and the largest and RMS shift. Accept (Enter)
 * is one undo step "Symmetrize at 1e-3 Å"; Cancel (Esc) leaves none. */
import { openBuilder, fmtN, stemOf } from "./builder_util.js";

const STOPS = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1];
const fmtTol = (t) => (t >= 1e-3 ? String(+t.toPrecision(2)) : t.toExponential(0).replace("e-", "e-"));

export function open(H, opts = {}) {
  let k = STOPS.indexOf(1e-3), sweep = null, last = null;
  const S = H.shell();
  return openBuilder(H, {
    id: "symmetrize", title: "Symmetrize", route: "build/symmetrize", defaultResult: "inplace", primary: "Create",
    primaryInPlace: "Accept",
    render(host, api) {
      const cur = api.ed ? api.ed.symprec : 1e-3;
      k = STOPS.reduce((best, t, i) => (Math.abs(Math.log(t / cur)) < Math.abs(Math.log(STOPS[best] / cur)) ? i : best), 0);
      host.innerHTML = `<label class="bld-row"><span>Tolerance</span><input type="range" class="bld-slider" min="0" max="${STOPS.length - 1}" step="1" value="${k}" data-k="tol" aria-label="Tolerance"></label>
        <div class="bld-row"><span></span><b class="num" data-k="tolv"></b></div>
        <div class="bld-row" style="align-items:start"><span>Group at each stop</span><ul class="bld-list" role="listbox" aria-label="Group at each tolerance" data-k="sweep"><li class="bld-empty">Asking spglib…</li></ul></div>`;
      host.querySelector('[data-k="tol"]').addEventListener("input", (e) => { k = Number(e.target.value); paint(host); });
      host.querySelector('[data-k="sweep"]').addEventListener("click", (e) => {
        const li = e.target.closest("[data-i]"); if (!li) return;
        k = Number(li.dataset.i); host.querySelector('[data-k="tol"]').value = String(k); paint(host); api.changed();
      });
      paint(host);
      S.sourceOf(api.d).then((source) => S.apiTask("symmetrize/sweep", { source })).then((r) => {
        sweep = r.sweep; paint(host); api.changed();
      }).catch((e) => api.setError(e.message));
    },
    params(api) {
      const d = api.d, tol = STOPS[k], s = api.structure();
      const at = sweep ? sweep[k] : null;
      if (at && at.number === null) return { error: `spglib found no symmetry at ${fmtTol(tol)} Å; try a larger tolerance`, name: `${stemOf(d.title)}_sym` };
      const ro = last && last.symprec === tol
        ? `<p>${at ? `<b>${H.esc(at.international)} (${at.number})</b> · ` : ""}max shift <b>${last.max.toFixed(4)}</b> Å · RMS <b>${last.rms.toFixed(4)}</b> Å · <b>${fmtN(last.n)}</b> atoms${last.n !== s.numbers.length ? ` (from ${fmtN(s.numbers.length)})` : ""}</p>` +
          (last.n < s.numbers.length ? `<p class="warn">The result is the group's conventional cell, smaller than this one: a supercell loses its repeats</p>` : "")
        : "<p>…</p>";
      return { params: { symprec: tol }, n: last ? last.n : s.numbers.length, name: `${stemOf(d.title)}_sym`,
        label: `Symmetrize at ${fmtTol(tol)} Å of ${d.title}`, readout: ro,
        inPlace: { op: "symmetrize", args: { symprec: tol }, label: `Symmetrize at ${fmtTol(tol)} Å` } };
    },
    afterInPlace(ed) {
      // the refined group is the status bar's reference from now on (as the Phase 3 Symmetrize did)
      const y = ed.srv && ed.srv.structure && ed.srv.structure.symmetry;
      if (y && y.available) ed.referenceGroup = { number: y.number, international: y.international, n_operations: y.n_operations_conventional || y.n_operations };
      if (opts.afterAccept) opts.afterAccept();
    },
    drawPreview(r, api, core) {
      const info = r.info || {}, s = api.structure();
      const was = last;
      last = { symprec: info.symprec, max: info.max_shift || 0, rms: info.rms_shift || 0, n: r.n };
      if (!was || was.symprec !== last.symprec || was.max !== last.max) setTimeout(() => api.changed(), 0);
      const vec = info.vectors || [];
      const scale = last.max > 0 ? Math.max(1, Math.min(200, Math.round(1.2 / last.max))) : 1;
      const p = { kind: "symmetrize", arrows: [], note: scale > 1 ? `arrows ×${scale}` : "" };
      if (vec.length === s.numbers.length) {
        const cart = s.cart.map((c, i) => [c[0] + vec[i][0], c[1] + vec[i][1], c[2] + vec[i][2]]);
        p.atoms = { numbers: s.numbers, symbols: s.symbols, cart };
        p.arrows = s.cart.map((c, i) => (Math.hypot(...vec[i]) > 1e-5 ? [c, [c[0] + vec[i][0] * scale, c[1] + vec[i][1] * scale, c[2] + vec[i][2] * scale]] : null)).filter(Boolean);
        p.notePos = s.cart[0];
      }
      return p;
    },
  });

  function paint(host) {
    host.querySelector('[data-k="tolv"]').textContent = `${fmtTol(STOPS[k])} Å`;
    const ul = host.querySelector('[data-k="sweep"]');
    if (!sweep) return;
    ul.innerHTML = sweep.map((x, i) => `<li role="option" data-i="${i}" aria-selected="${i === k}" class="${x.change ? "change" : ""}">
      <span class="num">${fmtTol(x.symprec)} Å</span><span class="sub">${x.change ? "<b>" : ""}${x.number ? `${H.esc(x.international)} (${x.number})` : "no group"}${x.change ? "</b>" : ""}</span></li>`).join("");
  }
}
