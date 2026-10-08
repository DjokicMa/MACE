/* Build › Point Defect (spec §2.5, plan T6): a vacancy, a substitution, an
 * interstitial or an antisite, in a supercell sized so the defect is at
 * least Settings › Defect image distance (10 Å) from its periodic images.
 * The symmetry-inequivalent sites are listed with their Wyckoff letter and
 * multiplicity; picking an atom in the viewport (the dialog is non-modal)
 * selects its row, and the row's atom gets a halo. A charge goes with the
 * document as a hint for Setup. One structure per inequivalent site opens
 * one tab each. */
import { openBuilder, fmtN, stemOf } from "./builder_util.js";

const TABS = [["vacancy", "Vacancy"], ["substitution", "Substitution"], ["interstitial", "Interstitial"], ["antisite", "Antisite"]];
const det3 = (m) => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
  + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

/** opts: {tab, site (atom index)} from the atom menu. */
export function open(H, opts = {}) {
  let tab = opts.tab || "vacancy", info = null, infoKey = "", site = null, cand = 0, element = null, perSite = false;
  let sizeMode = "auto", loading = null;
  const nearest = new Map();          // matrix -> the distance between defect images (from the server)
  let curKey = "";
  const S = H.shell();
  const minD = () => Number(S.setting("defects.minImageDistance")) || 10;

  return openBuilder(H, {
    id: "defect", title: "Point Defect", route: "build/defect", defaultResult: "new", primary: "Create",
    render(host, api) {
      host.innerHTML = `<div class="bld-tabs" role="tablist" aria-label="Defect type">${TABS.map(([k, t]) =>
          `<button role="tab" data-tab="${k}" aria-selected="${k === tab}">${t}</button>`).join("")}</div>
        <div class="bld-row" style="align-items:start" data-pane="sites"><span>Site</span><ul class="bld-list" role="listbox" aria-label="Site" data-k="sites"><li class="bld-empty">Finding the sites…</li></ul></div>
        <div class="bld-row" style="align-items:start" data-pane="cands" hidden><span>Empty site</span><ul class="bld-list" role="listbox" aria-label="Empty site" data-k="cands"><li class="bld-empty">Searching…</li></ul></div>
        <div class="bld-row" data-pane="pos" hidden><span>Position (frac)</span><span class="bld-inline">
          ${[0, 1, 2].map((k) => `<input type="number" step="any" value="0.25" data-p="${k}" aria-label="${"xyz"[k]}">`).join("")}</span></div>
        <div class="bld-row" data-pane="el" hidden><span>Element</span><span class="bld-inline"><button type="button" class="tb-btn-text" data-k="el">Pick…</button>
          <select data-k="anti" aria-label="Element on the site" hidden></select></span></div>
        <div class="bld-row" style="align-items:start"><span>Supercell</span><span style="display:grid;gap:var(--s-1)">
          <label class="bld-check"><input type="radio" name="dsize" value="auto" checked> Smallest with images ≥ <b data-k="mind"></b> Å apart</label>
          <label class="bld-check" style="margin-left:22px"><input type="checkbox" data-k="mat"> Search matrices too</label>
          <label class="bld-check"><input type="radio" name="dsize" value="hand"> Set by hand
            <span class="bld-inline">${[0, 1, 2].map((k) => `<input type="number" min="1" max="10" step="1" value="${k < 3 ? 2 : 1}" data-n="${k}" aria-label="n${k + 1}" disabled>`).join("")}</span></label>
        </span></div>
        <label class="bld-check" data-pane="per"><input type="checkbox" data-k="per"> Generate one structure per inequivalent site</label>
        <label class="bld-row"><span>Charge</span><input type="number" step="1" value="0" data-k="charge" aria-label="Charge" style="width:80px"></label>`;
      host.querySelector('[data-k="mind"]').textContent = minD();
      host.querySelectorAll("[data-tab]").forEach((b) => b.addEventListener("click", () => {
        tab = b.dataset.tab;
        host.querySelectorAll("[data-tab]").forEach((x) => x.setAttribute("aria-selected", String(x.dataset.tab === tab)));
        if (tab === "antisite" && site !== null && !antiChoices(api).length) site = null;
        layout(host, api); load(api); api.changed();
      }));
      host.querySelector('[data-k="sites"]').addEventListener("click", (e) => {
        const li = e.target.closest("[data-i]"); if (!li) return;
        site = Number(li.dataset.i); selectAtom(api, site); renderSites(host, api); api.changed();
      });
      host.querySelector('[data-k="cands"]').addEventListener("click", (e) => {
        const li = e.target.closest("[data-c]"); if (!li) return;
        cand = Number(li.dataset.c);
        const c = info.candidates[cand];
        host.querySelectorAll("[data-p]").forEach((i, k) => { i.value = String(+c.frac[k].toFixed(6)); });
        renderCands(host, api); api.changed();
      });
      host.querySelector('[data-k="el"]').addEventListener("click", (e) => S.pickElement(e.currentTarget, (el) => {
        element = el; host.querySelector('[data-k="el"]').textContent = el; api.changed();
      }));
      host.querySelector('[data-k="anti"]').addEventListener("change", (e) => { element = e.target.value; api.changed(); });
      host.querySelectorAll('[name="dsize"]').forEach((r) => r.addEventListener("change", () => {
        sizeMode = host.querySelector('[name="dsize"]:checked').value;
        host.querySelectorAll("[data-n]").forEach((i) => { i.disabled = sizeMode !== "hand"; });
        api.changed();
      }));
      host.querySelector('[data-k="mat"]').addEventListener("change", () => { infoKey = ""; load(api); });
      host.querySelector('[data-k="per"]').addEventListener("change", (e) => { perSite = e.target.checked; api.changed(); });
      // a pick in the viewport selects the row of the atom's site (the dialog is non-modal)
      api.onSel = () => {
        const ed = api.ed;
        if (!ed || ed.selection.size !== 1 || !info) return;
        const i = [...ed.selection][0];
        if (!siteOf(i)) return;
        site = i; renderSites(host, api); api.changed();
      };
      document.addEventListener("mace-selection", api.onSel);
      if (opts.site !== undefined && opts.site !== null) site = opts.site;
      layout(host, api);
      load(api);
    },
    onClose(api) { document.removeEventListener("mace-selection", api.onSel); },
    params(api) {
      const host = api.frame.params, d = api.d, s = api.structure(), stem = stemOf(d.title);
      const charge = Number(host.querySelector('[data-k="charge"]').value);
      if (!info) return { error: "", blocked: true, name: `${stem}_defect`, readout: "<p>Finding the sites…</p>" };
      const m = matrix(host), det = det3(m);
      curKey = JSON.stringify(m);
      if (!(det > 0)) return { error: "n1, n2 and n3 are whole numbers from 1", name: `${stem}_defect` };
      const sc = det === 1 ? "" : isDiag(m) ? `${m[0][0]}x${m[1][1]}x${m[2][2]}_` : `sc${det}_`;
      if (!Number.isInteger(charge)) return { error: "The charge is a whole number", name: `${stem}_defect` };
      let p, label, name, inPlace = null, n = det * s.numbers.length;
      const st = site !== null ? siteOf(site) : null;
      const tag = st ? `${st.multiplicity}${st.wyckoff}` : "";
      if (tab === "interstitial") {
        const pos = [...host.querySelectorAll("[data-p]")].map((i) => Number(i.value));
        if (!element) return { error: "Pick the element to add", name: `${stem}_${sc}i`, blocked: true, readout: readout(n + 1) };
        if (pos.some((x) => !Number.isFinite(x))) return { error: "The position is three numbers", name: `${stem}_${sc}i` };
        const c = info.candidates && info.candidates[cand], ctag = c && sameFrac(c.frac, pos) ? `_${c.multiplicity}${c.wyckoff}` : "";
        p = { kind: "interstitial", element, position: pos, matrix: m };
        name = `${stem}_${sc}${element}_i${ctag}`; n += 1;
        label = `Interstitial ${element}${ctag ? ` (${ctag.slice(1)})` : ""} in ${d.title}`;
        inPlace = { op: "defect", args: { kind: "interstitial", element, frac: pos }, label: `Interstitial ${element}${ctag ? ` (${ctag.slice(1)})` : ""}` };
      } else {
        if (!st) return { error: "Pick a site (a row, or an atom in the viewport)", name: `${stem}_${sc}defect`, blocked: true, readout: readout(n) };
        if (tab === "vacancy") {
          p = { kind: "vacancy", site, matrix: m };
          name = `${stem}_${sc}V_${st.element}_${tag}`; n -= 1;
          label = `Vacancy at ${st.element} (${tag}) in ${d.title}`;
          inPlace = { op: "defect", args: { kind: "vacancy", index: site }, label: `Vacancy at ${st.element} (${tag})` };
        } else {
          const el = tab === "antisite" ? (element && antiChoices(api).includes(element) ? element : antiChoices(api)[0]) : element;
          if (!el) return { error: tab === "antisite" ? "Antisites need another element in the structure" : "Pick the element to put on the site",
            name: `${stem}_${sc}${tab}`, blocked: true, readout: readout(n) };
          if (el === st.element) return { error: "Pick a different element", name: `${stem}_${sc}${tab}` };
          p = { kind: tab, site, element: el, matrix: m };
          name = `${stem}_${sc}${el}_${st.element}_${tag}`;
          label = `${tab === "antisite" ? "Antisite" : "Substitution"} ${st.element}→${el} (${tag}) in ${d.title}`;
          inPlace = { op: "defect", args: { kind: tab, index: site, element: el }, label: `${tab === "antisite" ? "Antisite" : "Replace"} ${st.element}→${el} (${tag})` };
        }
      }
      const groupDrops = api.ed && api.ed.symMode === "sym" ? '<p class="bld-note">This structure: one atom, not its orbit; the space group drops.</p>' : "";
      const hints = charge ? { charge } : undefined;
      const many = perSite && tab !== "interstitial" ? sitesOfKind(api).map((x) => {
        const t = `${x.multiplicity}${x.wyckoff}`, el = p.element;
        const nm = tab === "vacancy" ? `${stem}_${sc}V_${x.element}_${t}` : `${stem}_${sc}${el}_${x.element}_${t}`;
        const lb = tab === "vacancy" ? `Vacancy at ${x.element} (${t}) in ${d.title}` : `${tab === "antisite" ? "Antisite" : "Substitution"} ${x.element}→${el} (${t}) in ${d.title}`;
        return { params: { ...p, site: x.index }, name: nm, label: lb, hints };
      }).filter((x) => tab === "vacancy" || (x.params.element && x.params.element !== siteOf(x.params.site).element)) : null;
      const ro = readout(n, many ? many.length : 0);
      return { params: p, n, name, label, hints, many, readout: ro + groupDrops, inPlace: many ? null : inPlace };
    },
    afterPreview(r, api) {
      const m = matrix(api.frame.params), key = JSON.stringify(m);
      const dist = r.info ? r.info.nearest_image : r.estimate ? r.estimate.nearest_image : null;
      if (dist !== null && dist !== undefined && !nearest.has(key)) { nearest.set(key, dist); api.changed(); }
    },
    drawPreview(r, api, core) {
      const s = api.structure(), p = { kind: "defect" };
      const m = matrix(api.frame.params);
      p.cell = m.map((row) => [0, 1, 2].map((k) => row[0] * s.cell[0][k] + row[1] * s.cell[1][k] + row[2] * s.cell[2][k]));
      if (perSite && tab !== "interstitial") p.halo = sitesOfKind(api).map((x) => x.index);
      else if (site !== null && tab !== "interstitial") {
        const st = siteOf(site);
        p.halo = [site];
        if (tab === "vacancy") p.vacancies = [s.cart[site]];
        else if (element) p.atoms = ghost(element, s.cart[site]);
        void st;
      }
      if (tab === "interstitial" && element) {
        const pos = [...api.frame.params.querySelectorAll("[data-p]")].map((i) => Number(i.value));
        p.atoms = ghost(element, pos.length === 3 ? [0, 1, 2].map((k) => pos[0] * s.cell[0][k] + pos[1] * s.cell[1][k] + pos[2] * s.cell[2][k]) : [0, 0, 0]);
        p.opacity = 0.6;
      } else if (p.atoms) p.opacity = 0.6;
      return p;
    },
    rebuild: null,
  });

  /* ---------------------------------------------------------------- helpers */
  function siteOf(i) { return info && info.sites ? info.sites.find((x) => x.orbit.includes(i)) : null; }
  function sitesOfKind(api) {
    if (!info || !info.sites) return [];
    if (tab === "antisite") return info.sites.filter((x) => antiChoices(api, x.element).length);
    return info.sites;
  }
  function antiChoices(api, onEl = null) {
    const s = api.structure(), els = [...new Set(s.symbols)];
    const own = onEl || (site !== null && siteOf(site) ? siteOf(site).element : null);
    return els.filter((e) => e !== own);
  }
  function matrix(host) {
    if (sizeMode === "hand") {
      const n = [...host.querySelectorAll("[data-n]")].map((i) => Number(i.value));
      return [[n[0], 0, 0], [0, n[1], 0], [0, 0, n[2]]];
    }
    return info && info.supercell ? info.supercell.matrix : [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  }
  function isDiag(m) { return m.every((r, i) => r.every((x, j) => i === j || x === 0)); }
  function sameFrac(a, b) { return a.every((x, k) => Math.abs(((x - b[k] + 0.5) % 1 + 1) % 1 - 0.5) < 1e-4); }
  function readout(n, structures = 0) {
    const cost = n > 500 ? `<p class="warn">CRYSTAL cost grows about as N³ (${fmtN(n)} atoms)</p>` : "";
    const chip = structures ? ` <span class="bld-chip">${structures} structures</span>` : "";
    return `<p><b>${fmtN(n)}</b> atoms · ${nearestText(nearest.get(curKey))}${chip}</p>${cost}`;
  }
  function nearestText(dist) {
    if (dist === null || dist === undefined) return "nearest defect image …";
    const text = `nearest defect image <b>${dist.toFixed(1)}</b> Å`;
    if (dist >= 8) return text;
    return `<span class="warn">${text}</span></p><p class="warn">Defect images closer than 8 Å interact strongly; results describe a defect concentration, not an isolated defect`;
  }
  function ghost(el, cart) {
    return { numbers: [1], symbols: [el], cart: [cart] };
  }
  function selectAtom(api, i) {
    const ed = api.ed;
    if (ed && api.E && api.E.select) api.E.select(ed, [i], "replace");
  }
  function layout(host, api) {
    const inter = tab === "interstitial";
    host.querySelector('[data-pane="sites"]').hidden = inter;
    host.querySelector('[data-pane="cands"]').hidden = !inter;
    host.querySelector('[data-pane="pos"]').hidden = !inter;
    host.querySelector('[data-pane="el"]').hidden = tab === "vacancy";
    host.querySelector('[data-k="el"]').hidden = tab === "antisite";
    const anti = host.querySelector('[data-k="anti"]');
    anti.hidden = tab !== "antisite";
    if (tab === "antisite") {
      const ch = antiChoices(api);
      anti.innerHTML = ch.map((e) => `<option value="${e}">${e}</option>`).join("");
      element = ch.includes(element) ? element : ch[0] || null;
      if (element) anti.value = element;
    }
    host.querySelector('[data-pane="per"]').hidden = inter;
    renderSites(host, api);
    renderCands(host, api);
  }
  function renderSites(host, api) {
    const ul = host.querySelector('[data-k="sites"]');
    if (!info) return;
    const list = sitesOfKind(api);
    if (!list.length) { ul.innerHTML = `<li class="bld-empty">${tab === "antisite" ? "Antisites need two elements" : "No sites"}</li>`; return; }
    const cur = site !== null ? siteOf(site) : null;
    ul.innerHTML = list.map((x) => `<li role="option" data-i="${x.index}" aria-selected="${cur === x}">${H.esc(x.element)} · ${x.multiplicity}${H.esc(x.wyckoff)} <span class="sub">×${x.multiplicity}</span></li>`).join("");
  }
  function renderCands(host, api) {
    const ul = host.querySelector('[data-k="cands"]');
    if (!info) return;
    if (info.interstitial_error) { ul.innerHTML = `<li class="bld-empty">${H.esc(info.interstitial_error)}</li>`; return; }
    const c = info.candidates;
    if (!c) return;
    if (!c.length) { ul.innerHTML = `<li class="bld-empty">No empty site at least 1.0 Å from every atom. Type a position instead</li>`; return; }
    ul.innerHTML = c.map((x, k) => `<li role="option" data-c="${k}" aria-selected="${k === cand}">${x.multiplicity}${H.esc(x.wyckoff)}
      <span class="sub">${x.distance.toFixed(2)} Å from the nearest atom</span></li>`).join("");
  }
  /** The structure's sites (and empty sites, and the supercell), once per change of what they depend on. */
  async function load(api) {
    const host = api.frame.params;
    const key = `${minD()}:${host.querySelector('[data-k="mat"]').checked}:${tab === "interstitial"}:${api.ed ? api.ed.srv.rev : 0}`;
    if (key === infoKey && info) return;
    infoKey = key;
    const want = key;
    try {
      const source = await S.sourceOf(api.d);
      const body = { source, symprec: api.ed ? api.ed.symprec : 1e-3, min_dist: minD(), search_matrices: host.querySelector('[data-k="mat"]').checked,
        interstitial: tab === "interstitial" };
      loading = S.apiTask("defect/info", body);
      const r = await loading;
      if (want !== infoKey || api.frame.closed) return;
      info = { ...(info || {}), ...r };
      if (tab === "interstitial" && !r.candidates && !r.interstitial_error) info.candidates = [];
      if (site === null && info.sites && info.sites.length && tab !== "interstitial") site = (sitesOfKind(api)[0] || info.sites[0]).index;
      if (tab === "interstitial" && info.candidates && info.candidates.length) {
        cand = Math.min(cand, info.candidates.length - 1);
        host.querySelectorAll("[data-p]").forEach((i, k) => { i.value = String(+info.candidates[cand].frac[k].toFixed(6)); });
      }
      if (info.supercell) nearest.set(JSON.stringify(info.supercell.matrix), info.supercell.nearest);
      if (info.supercell && sizeMode === "auto") {
        const m = info.supercell.matrix;
        if (isDiag(m)) host.querySelectorAll("[data-n]").forEach((i, k) => { i.value = String(m[k][k]); });
      }
      layout(host, api);
      api.changed();
    } catch (e) { api.setError(e.message); }
  }
}
