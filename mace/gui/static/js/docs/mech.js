/* Mechanical results documents (1.1.6): what an ELASTIC, EOS, QHA or CPHF output opens as from
 * the Explorer, as a transport file opens as its plots. One document per output, in the plot area,
 * with the Run Analysis layout: a strip of views (Results, then the run's plots), the plot toolbar
 * (Export ▾: PNG, SVG, CSV, Copy as TSV, Copy CLI Command) and the run's settings in the Inspector.
 *
 * The numbers are the server's (GET /api/elastic/document: mace.elastic.views.document, i.e.
 * `mace analyze elastic --json` plus the run's settings, plots and Born criteria; GET
 * /api/elastic/figure: one plot's arrays, which `mace analyze elastic FILE --plot NAME` draws
 * with matplotlib), so the CLI line Copy CLI Command gives draws what the screen shows.
 *
 * ELASTIC: C and S, the Voigt-Reuss-Hill moduli, Born stability per criterion, sound velocities and
 * the Debye temperature, the piezoelectric and photoelastic tensors; the directional Young's
 * modulus and linear compressibility (a 3D surface or polar cuts in the principal planes; a slab:
 * its in-plane polar plot). EOS: the fits, E(V) and P(V). QHA: V(T), α(T), B(T), heat capacity.
 * CPHF: ε∞, α, χ(1), refractive indices, β and γ; ε(ω) for a DYNAMIC run. */
import * as PK from "../plotkit.js";
import { matrixTable, VOIGT } from "../lazy/mx_util.js";

const KIND_TITLE = { ELASTIC: "Elastic", EOS: "Equation of state", QHA: "Quasi-harmonic", CPHF: "Dielectric response", FREQ: "Dielectric (INTCPHF)" };
const AXES = ["x", "y", "z"];
const EOS_NAMES = { murnaghan: "Murnaghan 1944", birch_murnaghan: "Birch-Murnaghan 1947", poirier_tarantola: "Poirier-Tarantola 1998", vinet: "Vinet 1987" };
/* the directional plots a 3D run draws as a surface or as principal-plane cuts (views.SHAPED) */
const SHAPED = ["youngs", "compressibility", "shear", "poisson", "piezo", "shg"];
/* ELATE's colors for the envelopes of G(n, m) and ν(n, m): the minimum green, the maximum blue, negative red
 * (the theme's tokens, light and dark) */
const ENV = { min: "--success", max: "--info", neg: "--danger" };

export function createMechView(host, H) {
  const { api, q, esc, toast, loadPlotly, download } = H;
  host.innerHTML = `<div class="mv pv" tabindex="-1">
    <div class="pv-head"><div class="pv-title"></div><div class="pv-chips"></div><div class="pv-acts"></div></div>
    <div class="rp-tabs mv-tabs" role="tablist" aria-label="Results and plots"></div>
    <div class="pv-note muted small" aria-live="polite"></div>
    <div class="mv-body" tabindex="0" aria-label="Results"></div>
    <div class="pv-plot mv-plot" tabindex="0" aria-label="Plot" hidden></div></div>`;
  const $ = (s) => host.querySelector(s);
  const el = $(".mv-plot"), body = $(".mv-body");
  let doc = null, kit = null, building = false, again = false;
  const cssVar = (n) => (H.plotVar ? H.plotVar(n, "scf") : H.cssVar ? H.cssVar(n) : getComputedStyle(document.documentElement).getPropertyValue(n).trim());
  const num = (v, d = 3) => (v === null || v === undefined || !Number.isFinite(Number(v)) ? "–" : Number(v).toFixed(d));
  const sig = (v, p = 4) => (v === null || v === undefined || !Number.isFinite(Number(v)) ? "–" : Number(Number(v).toPrecision(p)).toString());
  const key = (d) => `mace.gui.mech.${H.dirHash ? H.dirHash() : ""}.${d.path}`;
  /* the server's plot text keeps only <sub> and <sup> (E<sub>min</sub>, TPa<sup>−1</sup>); everything else is escaped */
  const rich = (t) => esc(String(t === undefined || t === null ? "" : t)).replace(/&lt;(\/?)(sub|sup)&gt;/g, "<$1$2>");
  /* a plain unit of the JSON (TPa^-1, C/m^2, A^3) as HTML */
  const uhtml = (u) => rich(String(u || "").replace(/\bA\^3/g, "Å^3").replace(/\^(-?\d+)/g, (m, e) => `<sup>${e.replace("-", "−")}</sup>`));

  function load(d) {
    if (d.opts) return;
    let saved = null;
    try { saved = JSON.parse(localStorage.getItem(key(d)) || "null"); } catch (e) { saved = null; }   // ls-key: mace.gui.mech.
    d.opts = { shape: "surface", width: 1.5, xr: null, yr: null, phases: [], fit: "", ...((saved && saved.opts) || {}) };
    d.tab = d.tab || (saved && saved.tab) || "results";
    d.figs = d.figs || {};
  }
  function persist(d) { try { localStorage.setItem(key(d), JSON.stringify({ tab: d.tab, opts: d.opts })); } catch (e) { /* private window */ } }   // ls-key: mace.gui.mech.

  /* ------------------------------------------------ the strip of views */
  function tabs() {
    const m = doc.mech, list = [["results", "Results"], ...((m && m.views) || []).map((v) => [v.key, v.label])];
    if (!list.some(([k]) => k === doc.tab)) doc.tab = "results";
    $(".mv-tabs").innerHTML = list.map(([k, t]) => `<button role="tab" class="chip${k === doc.tab ? " active" : ""}" data-tab="${k}" aria-selected="${k === doc.tab}" tabindex="${k === doc.tab ? 0 : -1}"${
      k !== "results" ? ` title="${esc(((m.views || []).find((v) => v.key === k) || {}).what || "")}"` : ""}>${rich(t)}</button>`).join("")
      + `<span class="rp-tabopts">${shapeSeg()}${phaseSeg()}</span>`;
  }
  const FITS = [["birch_murnaghan", "Birch-Murnaghan"], ["murnaghan", "Murnaghan"], ["poirier_tarantola", "Poirier-Tarantola"], ["vinet", "Vinet"]];
  /* the enthalpy tab: the other phases compared (each removable), Add phase… and the fit the comparison uses */
  function phaseSeg() {
    if (!doc || doc.tab !== "enthalpy") return "";
    const ph = doc.opts.phases || [];
    return `<span class="mv-phases">${ph.map((p) => `<span class="mv-phase" title="${esc(p)}">${esc(p.split("/").pop())}<button class="mv-phase-x" data-unphase="${esc(p)}" aria-label="Remove ${esc(p)}">×</button></span>`).join("")}
      <button class="tb-btn-text" data-addphase title="Compare the enthalpy with another phase's EOS output (same composition): mace analyze elastic --compare">Add phase…</button>${
      ph.length ? `<select class="mv-fit" aria-label="Fit of the comparison">${FITS.map(([k, l]) => `<option value="${k}"${(doc.opts.fit || "birch_murnaghan") === k ? " selected" : ""}>${l}</option>`).join("")}</select>` : ""}</span>
      <span class="mv-addphase" hidden><input class="mv-phase-in" list="mv-eos-list" placeholder="another EOS output" spellcheck="false" aria-label="Another phase's EOS output">
      <datalist id="mv-eos-list">${(H.outputs ? H.outputs("EOS") : []).filter((p) => p !== doc.path && !ph.includes(p)).map((p) => `<option value="${esc(p)}">`).join("")}</datalist>
      <button class="tb-btn-text" data-phaseok>Add</button><button class="tb-btn-text" data-phasecancel>Cancel</button></span>`;
  }
  function setPhases(list) {
    doc.opts.phases = list;
    persist(doc);
    tabs();
    draw();
  }
  const directional = () => doc && SHAPED.includes(doc.tab) && doc.mech && doc.mech.data && (doc.mech.data.dimensionality || 3) === 3;
  function shapeSeg() {
    if (!directional()) return "";
    return `<span class="segment" role="radiogroup" aria-label="Directional plot">${[["surface", "3D Surface"], ["planes", "Planes"]].map(([v, l]) =>
      `<button role="radio" data-shape="${v}" aria-checked="${doc.opts.shape === v}" title="${v === "surface" ? "The value along every direction, as a closed surface" : "Polar cuts in the xy, xz and yz planes"}">${l}</button>`).join("")}</span>`;
  }
  $(".mv-tabs").addEventListener("click", (e) => {
    const t = e.target.closest("[data-tab]");
    if (t && doc) { showTab(t.dataset.tab); return; }
    const s = e.target.closest("[data-shape]");
    if (s && doc && doc.opts.shape !== s.dataset.shape) { doc.opts.shape = s.dataset.shape; persist(doc); tabs(); if (kit) kit.refresh(); draw(); }
    if (!doc) return;
    if (e.target.closest("[data-addphase]")) { const a = $(".mv-addphase"); a.hidden = false; a.querySelector("input").focus(); }
    if (e.target.closest("[data-phasecancel]")) $(".mv-addphase").hidden = true;
    if (e.target.closest("[data-phaseok]")) addPhase($(".mv-phase-in").value);
    const x = e.target.closest("[data-unphase]");
    if (x) setPhases((doc.opts.phases || []).filter((p) => p !== x.dataset.unphase));
  });
  $(".mv-tabs").addEventListener("change", (e) => {
    if (doc && e.target.closest(".mv-fit")) { doc.opts.fit = e.target.value; persist(doc); draw(); }
  });
  $(".mv-tabs").addEventListener("keydown", (e) => {
    if (!doc || !e.target.closest(".mv-phase-in")) return;
    if (e.key === "Enter") { e.preventDefault(); addPhase(e.target.value); }
    if (e.key === "Escape") { e.preventDefault(); $(".mv-addphase").hidden = true; }
  });
  function addPhase(path) {
    const p = String(path || "").trim();
    if (!p || p === doc.path || (doc.opts.phases || []).includes(p)) { $(".mv-addphase").hidden = true; return; }
    setPhases([...(doc.opts.phases || []), p]);
  }
  $(".mv-tabs").addEventListener("keydown", (e) => {
    if (!["ArrowLeft", "ArrowRight"].includes(e.key) || !e.target.closest("[data-tab]")) return;
    const bs = [...host.querySelectorAll(".mv-tabs [data-tab]")], i = bs.indexOf(e.target.closest("[data-tab]"));
    const n = bs[(i + (e.key === "ArrowRight" ? 1 : bs.length - 1)) % bs.length];
    e.preventDefault(); showTab(n.dataset.tab); const b = host.querySelector(`.mv-tabs [data-tab="${n.dataset.tab}"]`); if (b) b.focus();
  });
  function showTab(t) {
    doc.tab = t;
    persist(doc);
    tabs();
    if (kit) kit.refresh();
    draw();
  }

  /* ------------------------------------------------ the Results tables */
  const tile = (label, value, unit, title = "") => `<div class="mv-tile"${title ? ` title="${esc(title)}"` : ""}><div class="mv-tile-l">${label}</div><div class="mv-tile-v">${value}${unit ? ` <span class="mv-u">${unit}</span>` : ""}</div></div>`;
  // the title in one span: the h3 is a flex row, and its <sub>/<sup> must not become items of it
  const section = (title, html, cls = "") => `<section class="mv-sec ${cls}"><h3><span>${title}</span></h3>${html}</section>`;
  /* a signed number with a true minus sign, and no "−0.0000" */
  const snum = (v, d = 4) => { const t = num(v, d); return /^-0\.0*$/.test(t) ? t.slice(1) : t.replace("-", "−"); };
  const badge = (ok, yes = "stable", no = "UNSTABLE") => `<span class="mx-badge ${ok ? "ok" : "bad"}">${ok ? yes : no}</span>`;
  function dirText(ex, unit) {
    if (!ex || !ex.min) return "";
    const u = unit ? ` ${uhtml(unit)}` : "";
    if (ex.isotropic) return `isotropic: ${sig(ex.max.value, 5)}${u} along every direction`;
    return `${sig(ex.min.value, 5)}${u} along ${esc(ex.min.label)} to ${sig(ex.max.value, 5)}${u} along ${esc(ex.max.label)}`
      + (ex.min.value > 0 ? ` (max/min ${num(ex.max.value / ex.min.value, 3)})` : "");
  }
  /* one extreme of the directional table: the value, n and (G, ν) m */
  const extCell = (e) => (e ? `<td>${sig(e.value, 5)}</td><td class="l">${esc(e.label)}${e.label2 ? ` <span class="mv-u">m ${esc(e.label2)}</span>` : ""}</td>` : "<td>–</td><td></td>");
  function directionalHtml(m, dim) {
    const dir = m.directional || {}, rows = [];
    for (const [k, name, sym] of [["youngs", "Young's modulus", "E(n)"], ["compressibility", "Linear compressibility", "β(n)"], ["shear", "Shear modulus", "G(n, m)"], ["poisson", "Poisson's ratio", "ν(n, m)"]]) {
      const e = dir[k];
      if (!e || !e.min) continue;
      rows.push(`<tr><th class="l">${name} ${sym}${e.unit ? ` <span class="mv-u">(${uhtml(e.unit)})</span>` : ""}</th>${extCell(e.min)}${extCell(e.max)}<td>${e.isotropic ? "isotropic" : e.ratio !== null && e.ratio !== undefined ? num(e.ratio, 3) : "–"}</td></tr>`);
    }
    if (!rows.length) return "";
    const neg = (dir.compressibility && dir.compressibility.min && dir.compressibility.min.value < 0 ? ` <span class="mx-badge run">negative linear compressibility</span>` : "")
      + (dir.poisson && dir.poisson.auxetic ? ` <span class="mx-badge run">auxetic (ν &lt; 0)</span>` : "");
    return section(`Directional${neg}`, `<table class="mx-table"><tr><th class="l">Property</th><th>Min</th><th class="l">along</th><th>Max</th><th class="l">along</th><th>Max/min</th></tr>${rows.join("")}</table>
      <p class="note">G and ν depend on the direction n and on m perpendicular to it: the minimum and maximum over m (ELATE: Gaillac, Pullumbi &amp; Coudert, J. Phys.: Condens. Matter 28, 275201 (2016)). ${
        dim === 3 ? "The tabs draw them as a 3D surface (G and ν: the minimum and maximum envelopes) or as cuts in the xy, xz and yz planes." : "The tabs draw them in the slab's plane."}</p>`);
  }
  function indicesHtml(m) {
    const der = m.data.derived || {}, idx = der.indices, ch = der.character, th = der.thermal, out = [];
    if (idx) {
      const r = (label, v, note = "", d = 4) => (v === null || v === undefined ? "" : `<tr><th class="l">${label}</th><td>${num(v, d)}</td><td class="l mv-u">${note}</td></tr>`);
      out.push(section("Anisotropy", `<table class="mx-table mv-kv">${r("Universal A<sup>U</sup>", idx.universal, "5 G<sub>V</sub>/G<sub>R</sub> + K<sub>V</sub>/K<sub>R</sub> − 6; 0 = isotropic")}${
        r("Zener A", idx.zener, "2 C<sub>44</sub>/(C<sub>11</sub> − C<sub>12</sub>); 1 = isotropic")}${r("Chung–Buessem A<sub>C</sub>", idx.chung_buessem, "(G<sub>V</sub> − G<sub>R</sub>)/(G<sub>V</sub> + G<sub>R</sub>)")}${
        r("E<sub>max</sub>/E<sub>min</sub>", idx.e_ratio, "", 3)}${r("G<sub>max</sub>/G<sub>min</sub>", idx.g_ratio, "", 3)}${r("ν<sub>max</sub>/ν<sub>min</sub>", idx.nu_ratio, "", 3)}</table>
        <p class="note">Ranganathan &amp; Ostoja-Starzewski, PRL 101, 055504 (2008); Chung &amp; Buessem, J. Appl. Phys. 38, 2535 (1967).</p>`));
    }
    if (ch) {
      const hv = ch.hardness || {};
      out.push(section("Mechanical character", `<table class="mx-table mv-kv">${ch.pugh_ratio !== undefined ? `<tr><th class="l">Pugh's ratio B/G</th><td>${num(ch.pugh_ratio, 3)}</td><td class="l"><span class="mx-badge run">${ch.ductile ? "ductile" : "brittle"}</span> <span class="mv-u">ductile above ${ch.pugh_threshold}</span></td></tr>` : ""}${
        (ch.cauchy || []).map((c, i) => `<tr><th class="l">${i ? "" : ch.system === "cubic" ? "Cauchy pressure " : "Cauchy relation "}${esc(c.relation).replace(/C(\d\d)/g, "C<sub>$1</sub>")}</th><td>${num(c.value, 2)} GPa</td><td class="l mv-u">${
          ch.system === "cubic" ? (c.value < 0 ? "negative: directional (covalent) bonding" : "positive: metallic bonding") : "0 when the Cauchy relation holds"}</td></tr>`).join("")}${
        hv.chen_gpa !== undefined ? `<tr><th class="l">Vickers hardness H<sub>V</sub> (Chen)</th><td>${num(hv.chen_gpa, 1)} GPa</td><td class="l mv-u">2 (k<sup>2</sup>G)<sup>0.585</sup> − 3, k = G/B${
          hv.chen_clamped ? `; the formula gives ${snum(hv.chen_formula_gpa, 2)} GPa (a soft solid outside the fit): shown as 0, not stored` : ""}</td></tr>
        <tr><th class="l">Vickers hardness H<sub>V</sub> (Tian)</th><td>${num(hv.tian_gpa, 1)} GPa</td><td class="l mv-u">0.92 k<sup>1.137</sup> G<sup>0.708</sup></td></tr>` : ""}</table>
        <p class="note">Pugh, Philos. Mag. 45, 823 (1954); Pettifor, Mater. Sci. Technol. 8, 345 (1992)${hv.references ? `; hardness: ${esc(hv.references.chen)}; ${esc(hv.references.tian)}` : ""}. Hill moduli; empirical estimates.</p>`));
    }
    if (th && (th.clarke_w_mk !== undefined || th.debye_temperature_k !== undefined)) {
      out.push(section("Thermal", `<table class="mx-table mv-kv">${th.debye_temperature_k !== undefined ? `<tr><th class="l">Debye temperature Θ<sub>D</sub></th><td>${num(th.debye_temperature_k, 1)} K</td><td class="l mv-u">mean sound velocity (Anderson 1963)</td></tr>` : ""}${
        th.clarke_w_mk !== undefined ? `<tr><th class="l">Minimum thermal conductivity κ<sub>min</sub> (Clarke)</th><td>${num(th.clarke_w_mk, 3)} W/(m·K)</td><td class="l mv-u">0.87 k<sub>B</sub> M<sub>a</sub><sup>−2/3</sup> E<sup>1/2</sup> ρ<sup>1/6</sup></td></tr>
        <tr><th class="l">Minimum thermal conductivity κ<sub>min</sub> (Cahill)</th><td>${num(th.cahill_w_mk, 3)} W/(m·K)</td><td class="l mv-u">(π/6)<sup>1/3</sup> k<sub>B</sub> n<sup>2/3</sup> (v<sub>l</sub> + 2 v<sub>t</sub>)</td></tr>
        <tr><th class="l">Mean atomic mass M<sub>a</sub></th><td>${num(th.mean_atomic_mass_amu, 3)} u</td><td></td></tr>` : ""}</table>
        ${th.references ? `<p class="note">${esc(th.references.clarke)}; ${esc(th.references.cahill)}. Hill E and the Voigt-Reuss-Hill sound velocities.</p>` : ""}`));
    }
    return out.length ? `<div class="mv-pair">${out.join("")}</div>` : "";
  }
  function soundHtml(m) {
    const s = ((m.data.derived || {}).directional || {}).sound;
    if (!s) return "";
    return `<table class="mx-table"><tr><th class="l">Christoffel</th><th>Min</th><th class="l">along</th><th>Max</th><th class="l">along</th><th>Group min</th><th>Group max</th></tr>${
      [["qL", "Quasi-longitudinal qL"], ["qS1", "Fast quasi-shear qS1"], ["qS2", "Slow quasi-shear qS2"]].map(([k, l]) => s[k] ? `<tr><td class="l">${l}</td>${extCell(s[k].min)}${extCell(s[k].max)}<td>${sig(s[k].group.min.value, 5)}</td><td>${sig(s[k].group.max.value, 5)}</td></tr>` : "").join("")}</table>
      <p class="note">Phase and group velocities over every direction, km/s (Christoffel equation from C and the density; the Sound Velocities tab draws them).</p>`;
  }
  function piezoHtml(m) {
    const pz = (m.data.derived || {}).piezo;
    if (!pz || !pz.longitudinal) return "";
    const k = pz.kind, sym = `${k}<sub>L</sub>`, mx = pz.longitudinal.max || {}, cp = pz.coupling || {};
    if (pz.zero) return section("Longitudinal piezoelectric coefficient", `<p class="note">Every coefficient is zero.</p>`);
    const eps = (doc && doc.opts && doc.opts.epsilon) || "";
    const coupling = k !== "d" ? "" : cp.available
      ? `<tr><th class="l">Coupling k<sup>2</sup> along x, y, z</th><td class="l">${cp.k2_axes.map((x) => num(x, 4)).join(", ")}</td></tr>
         <tr><th class="l">Largest k<sup>2</sup> (k)</th><td class="l">${num(cp.k2_max.value, 4)} (${num(cp.k_max, 3)}) along ${esc(cp.k2_max.label)}</td></tr>`
      : `<tr><th class="l">Coupling k<sup>2</sup></th><td class="l">needs the static permittivity ε<sup>T</sup></td></tr>`;
    return section(`Longitudinal piezoelectric coefficient ${sym}(n)`, `<table class="mx-table mv-kv">
      <tr><th class="l">Largest |${sym}|</th><td class="l">${sig(Math.abs(mx.value), 5)} ${uhtml(pz.unit)} along ${esc(mx.label)} <span class="mv-u">(${sym} = ${sig(mx.signed, 5)} there)</span></td></tr>
      <tr><th class="l">${sym} along x, y, z</th><td class="l">${(pz.axes || []).map((x) => sig(x, 5)).join(", ")} ${uhtml(pz.unit)}</td></tr>${coupling}</table>
      ${k === "d" ? `<div class="mv-eps"><label><span>ε<sup>T</sup> (static, stress-free)</span> <input type="text" class="mv-eps-in" value="${esc(eps)}" placeholder="ε or εxx εyy εzz" spellcheck="false" aria-label="Static relative permittivity"></label>
        <button class="tb-btn-text" data-mv="epsilon">Apply</button></div>
      <p class="note">k<sup>2</sup>(n) = ${sym}(n)<sup>2</sup> / (s<sub>L</sub>(n) ε<sub>0</sub> ε<sub>L</sub>(n)), s<sub>L</sub> = 1/E(n), ε<sub>L</sub> = n·ε<sup>T</sup>·n: the IEEE k<sub>33</sub> along any direction, under a field and stress along n.
        ${cp.available ? (cp.epsilon_source === "given" ? "With the ε<sup>T</sup> given." : "With the run's electronic ε<sub>∞</sub>: an upper bound, since ε<sup>T</sup> ≥ ε<sub>∞</sub>.") : "ELAPIEZO prints no permittivity: give the static, stress-free one (a CPHF/FREQ run's ε<sub>0</sub>, or a measured value)."}</p>` : ""}`);
  }
  function elasticHtml(m) {
    const D = m.data, mod = D.moduli || {}, el = D.elastic, parts = [];
    const dim = D.dimensionality || 3;
    if (el && mod.bulk_modulus_hill_gpa !== undefined) {
      parts.push(`<div class="mv-tiles">${tile("Bulk modulus K<sub>H</sub>", num(mod.bulk_modulus_hill_gpa, 1), "GPa", "Voigt-Reuss-Hill average")}${tile("Shear modulus G<sub>H</sub>", num(mod.shear_modulus_hill_gpa, 1), "GPa", "Voigt-Reuss-Hill average")}${
        tile("Young's modulus E", num(mod.youngs_modulus_gpa, 1), "GPa", "From the Hill K and G")}${tile("Poisson's ratio ν", num(mod.poisson_ratio, 3), "")}${
        mod.debye_temperature_k ? tile("Debye temperature Θ<sub>D</sub>", num(mod.debye_temperature_k, 0), "K", "From the mean sound velocity (Anderson 1963)") : ""}${
        tile("Born stability", badge(mod.born_stable, "stable", "UNSTABLE"), "", "Every eigenvalue of C positive (Mouhat & Coudert 2014)")}</div>`);
    } else if (el && mod.layer_modulus !== undefined) {
      parts.push(`<div class="mv-tiles">${tile("E<sub>x</sub>", sig(mod.youngs_modulus_x), esc(mod.unit))}${tile("E<sub>y</sub>", sig(mod.youngs_modulus_y), esc(mod.unit))}${
        tile("ν<sub>xy</sub>", num(mod.poisson_ratio_xy, 3), "")}${tile("G<sub>xy</sub>", sig(mod.shear_modulus_xy), esc(mod.unit))}${tile("Layer modulus", sig(mod.layer_modulus), esc(mod.unit), "(C11 + C22 + 2 C12) / 4")}${
        tile("Born stability", badge(mod.born_stable), "")}</div>`);
    }
    if (D.moduli_error) parts.push(`<p class="err">Moduli not computed: ${esc(D.moduli_error)}</p>`);
    const tens = [];
    if (el) {
      const n = el.matrix.length, labels = n === 6 ? VOIGT : n === 3 ? ["xx", "yy", "xy"] : null;
      tens.push(section(`Stiffness C <span class="mv-u">(${esc(el.unit)}${el.system ? ", " + esc(el.system.toLowerCase()) : ""})</span>`,
        matrixTable(el.matrix, { cols: labels, rows: labels, digits: 3 })));
    }
    if (D.compliance) tens.push(section(`Compliance S <span class="mv-u">(${esc(D.compliance.unit)})</span>`,
      matrixTable(D.compliance.matrix, { cols: VOIGT, rows: VOIGT, digits: 4 })));
    if (dim === 2 && mod.c2d) tens.push(section(`In-plane stiffness <span class="mv-u">(${esc(mod.unit)})</span>`,
      matrixTable(mod.c2d, { cols: ["xx", "yy", "xy"], rows: ["xx", "yy", "xy"], digits: 3 })));
    if (tens.length) parts.push(`<div class="mv-pair">${tens.join("")}</div>`);
    const row2 = [];
    if (mod.bulk_modulus_hill_gpa !== undefined) {
      const vrh = D.crystal_vrh;
      row2.push(section("Moduli <span class=\"mv-u\">(GPa)</span>", `<table class="mx-table"><tr><th></th><th>Voigt</th><th>Reuss</th><th>Hill</th></tr>
        <tr><th class="l">Bulk K</th><td>${num(mod.bulk_modulus_voigt_gpa, 2)}</td><td>${num(mod.bulk_modulus_reuss_gpa, 2)}</td><td>${num(mod.bulk_modulus_hill_gpa, 2)}</td></tr>
        <tr><th class="l">Shear G</th><td>${num(mod.shear_modulus_voigt_gpa, 2)}</td><td>${num(mod.shear_modulus_reuss_gpa, 2)}</td><td>${num(mod.shear_modulus_hill_gpa, 2)}</td></tr></table>
        <table class="mx-table mv-kv"><tr><th class="l">Young's modulus E (Hill)</th><td>${num(mod.youngs_modulus_gpa, 2)} GPa</td></tr>
        <tr><th class="l">E along x, y, z (1/S<sub>ii</sub>)</th><td>${(mod.youngs_modulus_xyz_gpa || []).map((x) => num(x, 1)).join(", ")} GPa</td></tr>
        <tr><th class="l">Poisson's ratio ν</th><td>${num(mod.poisson_ratio, 4)}</td></tr><tr><th class="l">Pugh ratio K/G</th><td>${num(mod.pugh_ratio, 3)}</td></tr>
        <tr><th class="l">Universal anisotropy A<sup>U</sup></th><td>${num(mod.universal_anisotropy, 4)} <span class="mv-u">(0 = isotropic)</span></td></tr></table>
        ${vrh ? `<p class="note">CRYSTAL's own VRH table: K<sub>H</sub> ${num(vrh.K_H, 2)}, G<sub>H</sub> ${num(vrh.G_H, 2)}, E<sub>H</sub> ${num(vrh.E_H, 2)} GPa, ν<sub>H</sub> ${num(vrh.v_H, 3)}</p>` : ""}`));
    }
    if ((m.born || []).length) {
      row2.push(section(`Born stability ${badge(m.born.every((b) => b.ok))}`, `<table class="mx-table"><tr><th class="l">Criterion</th><th>Value</th><th></th></tr>${
        m.born.map((b) => `<tr><td class="l">${esc(b.criterion)}</td><td>${sig(b.value, 6)} <span class="mv-u">${esc(b.unit)}</span></td><td class="${b.ok ? "mv-ok" : "mv-bad"}">${b.ok ? "✓" : "✗"}</td></tr>`).join("")}</table>
        <p class="note">${el && el.system ? `Conditions of the ${esc(el.system.toLowerCase())} system and ` : ""}C positive definite (Mouhat &amp; Coudert, PRB 90, 224104 (2014)).</p>`));
    }
    if (mod.sound_velocity_mean_km_s !== undefined || D.seismic) {
      row2.push(section("Sound velocities", `${mod.sound_velocity_mean_km_s !== undefined ? `<table class="mx-table mv-kv">
        <tr><th class="l">Transverse v<sub>t</sub></th><td>${num(mod.sound_velocity_transverse_km_s, 3)} km/s</td></tr>
        <tr><th class="l">Longitudinal v<sub>l</sub></th><td>${num(mod.sound_velocity_longitudinal_km_s, 3)} km/s</td></tr>
        <tr><th class="l">Mean v<sub>m</sub></th><td>${num(mod.sound_velocity_mean_km_s, 3)} km/s</td></tr>
        ${mod.debye_temperature_k ? `<tr><th class="l">Debye temperature Θ<sub>D</sub></th><td>${num(mod.debye_temperature_k, 1)} K</td></tr>` : ""}</table>` : ""}
        ${D.seismic ? `<table class="mx-table"><tr><th class="l">Wave vector</th><th>V<sub>p</sub></th><th>V<sub>s1</sub></th><th>V<sub>s2</sub></th></tr>${
          D.seismic.map((v) => `<tr><td class="l">[${v.direction.map((x) => num(x, 2)).join(" ")}]</td><td>${num(v.vp)}</td><td>${num(v.vs1)}</td><td>${num(v.vs2)}</td></tr>`).join("")}</table>
          <p class="note">Christoffel equation (CRYSTAL), km/s</p>` : ""}${soundHtml(m)}`));
    }
    if (row2.length) parts.push(`<div class="mv-pair">${row2.join("")}</div>`);
    parts.push(directionalHtml(m, dim));
    if (dim === 3) parts.push(indicesHtml(m));
    const t2 = (D.derived || {}).two_d;
    if (t2) parts.push(section("In-plane anisotropy", `<table class="mx-table mv-kv"><tr><th class="l">A<sup>SU</sup></th><td>${num(t2.anisotropy_su, 4)} <span class="mv-u">(0 = isotropic)</span></td></tr>
      <tr><th class="l">Area modulus (Voigt, Reuss)</th><td>${sig(t2.area_modulus_voigt)}, ${sig(t2.area_modulus_reuss)} ${uhtml(t2.unit)}</td></tr>
      <tr><th class="l">Shear modulus (Voigt, Reuss)</th><td>${sig(t2.shear_modulus_voigt)}, ${sig(t2.shear_modulus_reuss)} ${uhtml(t2.unit)}</td></tr>
      <tr><th class="l">E<sub>max</sub>/E<sub>min</sub>, G<sub>max</sub>/G<sub>min</sub></th><td>${num(t2.e_ratio, 3)}, ${num(t2.g_ratio, 3)}</td></tr></table>
      <p class="note">A<sup>SU</sup> = √[(K<sub>V</sub>/K<sub>R</sub> − 1)<sup>2</sup> + 2 (G<sub>V</sub>/G<sub>R</sub> − 1)<sup>2</sup>] of the in-plane block (Li, Shao, Gao &amp; Liu, Extreme Mech. Lett. 34, 100615 (2020)).</p>`));
    const pzl = piezoHtml(m);
    const pz = [];
    for (const [k, title, rows] of [["piezo_e", "Direct piezoelectric tensor e", AXES], ["piezo_d", "Converse piezoelectric tensor d", AXES],
      ["pockels", "Elasto-optic (Pockels) tensor p", VOIGT], ["piezo_optic", "Piezo-optic tensor π", VOIGT]]) {
      if (D[k]) pz.push(section(`${title} <span class="mv-u">(${uhtml(D[k].unit)})</span>`, matrixTable(D[k].matrix, { cols: VOIGT, rows, digits: 4 })));
    }
    if (pzl) pz.unshift(pzl);
    if (pz.length) parts.push(`<div class="mv-pair">${pz.join("")}</div>`);
    return parts.join("");
  }
  function eosHtml(m) {
    const e = m.data.eos || {}, f = e.fits || {}, bm = f.birch_murnaghan || Object.values(f)[0], pts = e.points || [];
    const parts = [];
    const bmName = f.birch_murnaghan ? "Birch-Murnaghan" : "";
    const bml = (s) => `${s}${bmName ? ` <span class="mv-u">(${bmName})</span>` : ""}`;
    if (bm) parts.push(`<div class="mv-tiles">${tile(bml("V<sub>0</sub>"), num(bm.volume_ang3, 4), "Å<sup>3</sup>")}${tile(bml("B<sub>0</sub>"), num(bm.bulk_modulus_gpa, 2), "GPa")}${
      tile(bml("B′<sub>0</sub>"), num(bm.bulk_modulus_prime, 2), "")}${tile(bml("E<sub>0</sub>"), num(bm.energy_au, 8), "Ha")}${tile("Volumes", String(pts.length), "")}${
      m.data.formula ? tile("Formula", `${esc(m.data.formula).replace(/(\d+)/g, "<sub>$1</sub>")}`, `Z = ${m.data.formula_units}`, "Formula units in the primitive cell: the enthalpy is per formula unit") : ""}</div>`);
    if (Object.keys(f).length) parts.push(section("Fits <span class=\"mv-u\">(all points)</span>", `<table class="mx-table"><tr><th class="l">Equation of state</th><th>V<sub>0</sub> (Å<sup>3</sup>)</th><th>E<sub>0</sub> (Ha)</th><th>B<sub>0</sub> (GPa)</th><th>B′<sub>0</sub></th></tr>${
      Object.entries(f).map(([n, x]) => `<tr><td class="l">${esc(EOS_NAMES[n] || n)}</td><td>${num(x.volume_ang3, 4)}</td><td>${num(x.energy_au, 8)}</td><td>${num(x.bulk_modulus_gpa, 2)}</td><td>${num(x.bulk_modulus_prime, 2)}</td></tr>`).join("")}</table>`));
    const qual = e.quality || {};
    if (Object.keys(qual).length) parts.push(section("Fit quality", `<table class="mx-table"><tr><th class="l">Equation of state</th><th>RMS (meV)</th><th>Largest (meV)</th><th>RMS / spread</th><th></th></tr>${
      Object.entries(qual).map(([n, x]) => `<tr><td class="l">${esc(EOS_NAMES[n] || n)}</td><td>${sig(x.rms_mev, 3)}</td><td>${sig(x.max_abs_mev, 3)}</td><td>${Number(x.rms_relative).toExponential(1).replace("-", "−")}</td><td><span class="mx-badge ${x.quality === "good" ? "ok" : x.quality === "fair" ? "run" : "bad"}">${x.quality}</span></td></tr>`).join("")}</table>
      <p class="note">Residuals of the computed energies from each fit (the Residuals tab draws them); the RMS relative to the spread of the computed energies: good below 10<sup>−3</sup>, fair below 10<sup>−2</sup>.</p>`));
    if (pts.length) {
      const v0 = m.data.volume_ang3, emin = Math.min(...pts.map((p) => p[1]));
      parts.push(section("Computed points", `<table class="mx-table"><tr><th>V (Å<sup>3</sup>)</th>${v0 ? "<th>V / V<sub>in</sub></th>" : ""}<th>E (Ha)</th><th>E − E<sub>min</sub> (meV)</th></tr>${
        pts.map((p) => `<tr><td>${num(p[0], 6)}</td>${v0 ? `<td>${num(p[0] / v0, 4)}</td>` : ""}<td>${num(p[1], 10)}</td><td>${num((p[1] - emin) * 27211.386245988, 3)}</td></tr>`).join("")}</table>`));
    }
    return `${parts[0] || ""}<div class="mv-pair">${parts.slice(1).join("")}</div>`;
  }
  function qhaHtml(m) {
    const qd = m.data.qha, T = qd.temperature_k, n = T.length;
    const at = (t) => { let b = 0; T.forEach((x, i) => { if (Math.abs(x - t) < Math.abs(T[b] - t)) b = i; }); return b; };
    const i3 = at(300);
    const step = Math.max(1, Math.round(n / 12)), pick = [...new Set([...Array.from({ length: n }, (_, i) => i).filter((i) => i % step === 0), i3, n - 1])].sort((a, b) => a - b);
    return `<div class="mv-tiles">${tile(`V(${num(T[i3], 0)} K)`, num(qd.volume_ang3[i3], 4), "Å<sup>3</sup>")}${tile(`α<sub>V</sub>(${num(T[i3], 0)} K)`, Number(qd.alpha_v_per_k[i3]).toExponential(3).replace("-", "−"), "K<sup>−1</sup>")}${
      tile(`B<sub>T</sub>(${num(T[i3], 0)} K)`, num(qd.bulk_modulus_t_gpa[i3], 3), "GPa")}${tile(`B<sub>S</sub>(${num(T[i3], 0)} K)`, num(qd.bulk_modulus_s_gpa[i3], 3), "GPa")}${tile("Temperatures", `${num(T[0], 0)}–${num(T[n - 1], 0)}`, "K", `${n} temperatures`)}</div>`
      + section(`Thermal properties at P = 0 <span class="mv-u">(${pick.length} of ${n} temperatures; Export ▾ › CSV on a plot has every one)</span>`,
        `<table class="mx-table"><tr><th>T (K)</th><th>V (Å<sup>3</sup>)</th><th>α<sub>V</sub> (K<sup>−1</sup>)</th><th>B<sub>T</sub> (GPa)</th><th>B<sub>S</sub> (GPa)</th><th>C<sub>p</sub>−C<sub>v</sub> (J/mol·K)</th></tr>${
          pick.map((i) => `<tr${i === i3 ? ' class="mv-hl"' : ""}><td>${num(T[i], 1)}</td><td>${num(qd.volume_ang3[i], 4)}</td><td>${Number(qd.alpha_v_per_k[i]).toExponential(3)}</td><td>${num(qd.bulk_modulus_t_gpa[i], 3)}</td><td>${num(qd.bulk_modulus_s_gpa[i], 3)}</td><td>${num(qd.cp_minus_cv_j_mol_k[i], 3)}</td></tr>`).join("")}</table>`)
      + qhaThermoHtml(qd, m.gruneisen_300k);
  }
  /* CRYSTAL's mode γ: the row nearest 300 K the database stores (qha_gamma_mode_300k, mace/elastic/properties.py
   * nearest_300k: the server sends its index), else the highest temperature when no row is within 15 K */
  function gammaTile(g, i) {
    if (i != null && g.gamma[i] != null)
      return tile(`γ (mode, ${num(g.temperature_k[i], 0)} K)`, num(g.gamma[i], 3), "", "CRYSTAL's Grüneisen table, the row nearest 300 K (the database's value)");
    return tile("γ (mode, CRYSTAL)", num(g.gamma[g.gamma.length - 1], 3), "", "CRYSTAL's Grüneisen table, highest temperature");
  }
  /* F, S, C_v, C_p and γ along V(T) (each volume's harmonic thermodynamics) and CRYSTAL's Grüneisen γ */
  function qhaThermoHtml(qd, gi) {
    const th = qd.thermo, g = qd.gruneisen, parts = [];
    if (th) {
      const Tt = th.temperature_k, j3 = Tt.reduce((b, x, i) => (Math.abs(x - 300) < Math.abs(Tt[b] - 300) ? i : b), 0);
      parts.push(`<div class="mv-tiles">${tile(`S(${num(Tt[j3], 0)} K)`, num(th.entropy_j_mol_k[j3], 3), "J/(mol·K)")}${tile(`C<sub>v</sub>(${num(Tt[j3], 0)} K)`, num(th.cv_j_mol_k[j3], 3), "J/(mol·K)")}${
        tile(`C<sub>p</sub>(${num(Tt[j3], 0)} K)`, num(th.cp_j_mol_k[j3], 3), "J/(mol·K)")}${th.gamma_th[j3] !== null ? tile(`γ<sub>th</sub>(${num(Tt[j3], 0)} K)`, num(th.gamma_th[j3], 3), "") : ""}${
        g ? gammaTile(g, gi) : ""}</div>`);
      parts.push(section(`Thermodynamic functions along V(T) <span class="mv-u">(per mole of cells; ${Tt.length} temperatures every volume prints)</span>`,
        `<table class="mx-table"><tr><th>T (K)</th><th>F − F(T<sub>0</sub>) (kJ/mol)</th><th>S (J/(mol·K))</th><th>C<sub>v</sub> (J/(mol·K))</th><th>C<sub>p</sub> (J/(mol·K))</th><th>γ<sub>th</sub></th></tr>${
          Tt.map((t, i) => `<tr${i === j3 ? ' class="mv-hl"' : ""}><td>${num(t, 1)}</td><td>${num(th.f_kj_mol[i] - th.f_kj_mol[0], 4)}</td><td>${num(th.entropy_j_mol_k[i], 4)}</td><td>${num(th.cv_j_mol_k[i], 4)}</td><td>${num(th.cp_j_mol_k[i], 4)}</td><td>${th.gamma_th[i] === null ? "–" : num(th.gamma_th[i], 3)}</td></tr>`).join("")}</table>
        <p class="note">F = EL + E0 + ET − TS, S and C<sub>v</sub> of each computed volume, interpolated to V(T) by an order-${th.polynomial_order} polynomial through the ${th.volumes_ang3.length} volumes; C<sub>p</sub> = C<sub>v</sub> + (C<sub>p</sub> − C<sub>v</sub>); γ<sub>th</sub> = α<sub>V</sub> B<sub>T</sub> V<sub>m</sub>/C<sub>v</sub>.</p>`));
    } else if (g) {
      parts.push(`<div class="mv-tiles">${gammaTile(g, gi)}</div>`);
    }
    return parts.join("");
  }
  function cphfHtml(m) {
    const D = m.data, de = D.dielectric, c = m.cphf, parts = [];
    if (de && c) {
      const pe = de.principal_epsilon, ri = de.refractive_indices;
      // three principal values, or one when the crystal is optically isotropic (they are equal)
      const three = (v, label) => (Math.max(...v) - Math.min(...v) <= 1e-6 * Math.abs(v[0])
        ? tile(`${label} <span class="mv-u">(isotropic)</span>`, num(v[0], 4), "") : tile(`${label} <span class="mv-u">(AA, BB, CC)</span>`, `<span class="mv-small">${v.map((x) => num(x, 4)).join(" · ")}</span>`, ""));
      parts.push(`<div class="mv-tiles">${pe ? three(pe, "ε<sub>∞</sub>") : ""}${ri ? three(ri, "Refractive index n") : ""}${
        tile("α (trace / 3)", num((c.alpha[0][0] + c.alpha[1][1] + c.alpha[2][2]) / 3, 4), "bohr<sup>3</sup>")}${tile("Frequency", c.frequency_ev ? `${num(c.frequency_ev, 3)} eV` : "static", "", c.n_frequencies > 1 ? `${c.n_frequencies} frequencies: the ε(ω) tab draws them` : "")}</div>`);
      const lab = ["x", "y", "z"];
      parts.push(`<div class="mv-pair">${section(`Dielectric tensor ε<sub>∞</sub> <span class="mv-u">(${c.frequency_ev ? num(c.frequency_ev, 3) + " eV" : "static"})</span>`, matrixTable(c.epsilon, { cols: lab, rows: lab, digits: 4 }))}${
        section("Polarizability α <span class=\"mv-u\">(bohr<sup>3</sup>)</span>", matrixTable(c.alpha, { cols: lab, rows: lab, digits: 4 }))}${
        section("Susceptibility χ<sup>(1)</sup>", matrixTable(c.chi1, { cols: lab, rows: lab, digits: 4 }))}${
        (pe || ri) && !(D.optics && D.optics.principal) ? section("Principal axes", `<table class="mx-table"><tr><th></th><th>AA</th><th>BB</th><th>CC</th></tr>${pe ? `<tr><th class="l">ε</th>${pe.map((x) => `<td>${num(x, 4)}</td>`).join("")}</tr>` : ""}${
          ri ? `<tr><th class="l">n</th>${ri.map((x) => `<td>${num(x, 4)}</td>`).join("")}</tr>` : ""}</table><p class="note">ε = n² = 1 + χ<sup>(1)</sup> (CRYSTAL)</p>`) : ""}</div>`);
    }
    const opt = D.optics || {};
    const pr = opt.principal, sh = opt.shg, ox = [];
    if (pr) {
      const cls = pr.optical_class;
      const bad = pr.symmetry_mismatch ? ` <span class="mx-badge bad">symmetry mismatch: a ${esc(pr.expected_class)} crystal</span>` : "";
      ox.push(section(`Principal axes of ε<sub>∞</sub> <span class="mx-badge ${pr.unphysical ? "bad" : "run"}">${esc(cls)}</span>${bad}`, `<table class="mx-table"><tr><th class="l">Axis</th><th>ε</th><th>n</th></tr>${
        pr.axis_labels.map((l, i) => `<tr><td class="l">${esc(l)}</td><td>${num(pr.epsilon[i], 4)}</td><td>${num(pr.n[i], 4)}</td></tr>`).join("")}</table>
        <table class="mx-table mv-kv"><tr><th class="l">Birefringence Δn = n<sub>max</sub> − n<sub>min</sub></th><td>${num(pr.birefringence, 4)}</td></tr>${
        cls === "uniaxial" ? `<tr><th class="l">n<sub>o</sub>, n<sub>e</sub></th><td>${num(pr.n_o, 4)}, ${num(pr.n_e, 4)}</td></tr>
          <tr><th class="l">n<sub>e</sub> − n<sub>o</sub></th><td>${sig(pr.birefringence_signed, 4).replace("-", "−")} <span class="mv-u">(optically ${esc(pr.sign)}; optic axis ${esc(pr.optic_axis_label)})</span></td></tr>` : ""}${
        cls === "biaxial" ? `<tr><th class="l">Optic axial angle 2V</th><td>${num(pr.optic_axial_angle_deg, 2)}° <span class="mv-u">(biaxial ${esc(pr.sign)})</span></td></tr>` : ""}</table>
        <p class="note">Eigenvalues and eigenvectors of ε<sub>∞</sub>${pr.frequency_ev ? ` at ${num(pr.frequency_ev, 3)} eV` : ""}; n = √ε. The Index Ellipsoid tab draws the indicatrix.</p>`));
    }
    if (sh && sh.zero === false) {
      const at = (e) => `${sig(Math.abs(e.value), 4)} pm/V along ${esc(e.label)}`;
      ox.push(section("Second-harmonic generation <span class=\"mv-u\">(pm/V)</span>", `<table class="mx-table mv-kv">
        <tr><th class="l">d<sub>11</sub>, d<sub>22</sub>, d<sub>33</sub> <span class="mv-u">(pump and harmonic along x, y, z)</span></th><td>${sh.d_axes.map((x) => sig(x, 4).replace("-", "−")).join(", ")}</td></tr>
        <tr><th class="l">Largest longitudinal d<sub>L</sub>(n) <span class="mv-u">(one polarization)</span></th><td>${at(sh.longitudinal.max)}</td></tr>
        <tr><th class="l">Largest |d(n)| <span class="mv-u">(pump along n, any harmonic)</span></th><td>${at(sh.magnitude.max)}</td></tr>
        <tr><th class="l">Powder ⟨d<sub>L</sub><sup>2</sup>⟩<sup>1/2</sup>, ⟨|d|<sup>2</sup>⟩<sup>1/2</sup></th><td>${sig(sh.powder_longitudinal, 4)}, ${sig(sh.powder_magnitude, 4)}</td></tr></table>
        <p class="note">d = χ<sup>(2)</sup>/2 (CRYSTAL's d(MKS)), Kleinman-symmetric; d<sub>eff</sub> without phase matching or dispersion. The SHG tab draws |d<sub>ijk</sub> n<sub>j</sub> n<sub>k</sub>|.</p>`));
    }
    if (ox.length) parts.push(`<div class="mv-pair">${ox.join("")}</div>`);
    const hy = [];
    for (const [k, title] of [["beta", "First hyperpolarizability β"], ["gamma", "Second hyperpolarizability γ"]]) {
      const h = D[k];
      if (!h) continue;
      const big = h.components.filter((r) => r.value === null || Math.abs(r.value) > 1e-10);
      const cell = (v, d) => (v === null || v === undefined ? "overflow" : sig(v, d));
      // a conclusion only from a complete, converged table: never "zero" from a diverged or partial one
      const why = h.diverged ? `<p class="note mv-bad-note">Diverged (${h.overflow ? "CRYSTAL printed overflowed values" : `values up to ${sig(h.largest_au, 3)} a.u.`}): not analyzed, and no conclusion about inversion symmetry.</p>`
        : h.complete === false ? `<p class="note mv-bad-note">Incomplete table (${h.components.length} of ${h.n_independent} independent components read): no conclusion.</p>` : "";
      hy.push(section(`${title} <span class="mv-u">(a.u.)</span>${h.diverged ? ' <span class="mx-badge bad">diverged</span>' : ""}`, why + (big.length ? `<table class="mx-table"><tr><th class="l">Component</th><th>a.u.</th><th>χ (a.u.)</th><th>${esc(h.units.mks)}</th><th>${esc(h.units.cgs)}</th></tr>${
        big.map((r) => `<tr><td class="l">${esc(r.component.toLowerCase())}</td><td>${cell(r.value, 6)}</td><td>${cell(r.chi, 6)}</td><td>${cell(r.mks, 5)}</td><td>${cell(r.cgs, 5)}</td></tr>`).join("")}</table>`
        : why ? "" : `<p class="note">Every component is zero${k === "beta" ? " (a centrosymmetric crystal)" : ""}</p>`)));
    }
    if (hy.length) parts.push(`<div class="mv-pair">${hy.join("")}</div>`);
    return parts.join("");
  }
  /* a FREQ run with INTENS + INTCPHF: Born charges, ε∞ and ε0, the computed LO-TO splitting */
  function freqHtml(m) {
    const b = m.data.born || {}, parts = [], lab = ["x", "y", "z"];
    const zs = (b.born_charges || []).map((x) => x.isotropic);
    const asr = Math.max(...(b.charge_sum || [[0]]).flat().map(Math.abs));
    parts.push(`<div class="mv-tiles">${b.epsilon_inf_iso !== undefined ? tile("ε<sub>∞</sub> (trace/3)", num(b.epsilon_inf_iso, 4), "") : ""}${
      b.epsilon_0_iso !== undefined ? tile("ε<sub>0</sub> (trace/3)", num(b.epsilon_0_iso, 4), "") : ""}${
      zs.length ? tile("Born charges Z*", `${num(Math.min(...zs), 3).replace("-", "−")} … ${num(Math.max(...zs), 3)}`, "e", "Tr(Z*)/3 of every atom") : ""}${
      tile("Σ Z* (acoustic sum)", num(asr, 4), "e", "The largest component of the sum over atoms: 0 when the charge neutrality holds")}</div>`);
    const C = ["xx", "yy", "zz", "xy", "xz", "yz", "yx", "zx", "zy"], ij = { x: 0, y: 1, z: 2 };
    parts.push(section(`Born effective charges Z* <span class="mv-u">(e, ${zs.length} atoms)</span>`, `<table class="mx-table"><tr><th>Atom</th><th class="l">El</th><th>Tr(Z*)/3</th>${
      C.map((c) => `<th>Z*<sub>${c}</sub></th>`).join("")}</tr>${(b.born_charges || []).map((x) => `<tr><td>${x.atom}</td><td class="l">${esc(x.symbol)}</td><td><b>${snum(x.isotropic)}</b></td>${
        C.map((c) => `<td>${snum(x.tensor[ij[c[0]]][ij[c[1]]])}</td>`).join("")}</tr>`).join("")}</table>
      <p class="note">CRYSTAL's ATOMIC BORN CHARGE TENSOR (row: the polarization, column: the displacement); Tr(Z*)/3 is its dynamic charge.</p>`));
    const tens = [];
    if (b.epsilon_inf) tens.push(section(`ε<sub>∞</sub> <span class="mv-u">(principal ${b.epsilon_inf_principal.map((x) => num(x, 4)).join(", ")})</span>`, matrixTable(b.epsilon_inf, { cols: lab, rows: lab, digits: 4 })));
    if (b.epsilon_0) tens.push(section(`ε<sub>0</sub> = ε<sub>∞</sub> + ε<sub>vib</sub> <span class="mv-u">(principal ${b.epsilon_0_principal.map((x) => num(x, 4)).join(", ")})</span>`, matrixTable(b.epsilon_0, { cols: lab, rows: lab, digits: 4 })));
    if (b.epsilon_vib) tens.push(section("ε<sub>vib</sub> <span class=\"mv-u\">(sum of the modes' oscillator strengths)</span>", matrixTable(b.epsilon_vib, { cols: lab, rows: lab, digits: 4 })));
    if (tens.length) parts.push(`<div class="mv-pair">${tens.join("")}</div>`);
    if ((b.lo_to || []).length) {
      parts.push(section("LO-TO splitting at Γ <span class=\"mv-u\">(cm<sup>−1</sup>, computed)</span>", `<div class="mv-pair">${b.lo_to.map((r) => {
        const pairs = r.to_cm.map((t, i) => [t, r.lo_cm[i]]);
        return `<div><table class="mx-table"><tr><th class="l" colspan="3">q along ${esc(r.direction)}</th></tr><tr><th>TO</th><th>LO</th><th>Δ</th></tr>${
          pairs.map(([t, l]) => `<tr${l - t > 0.05 ? ' class="mv-hl"' : ""}><td>${num(t, 1)}</td><td>${num(l, 1)}</td><td>${l - t > 0.05 ? num(l - t, 1) : "–"}</td></tr>`).join("")}</table>${
          r.lst_ratio_freq !== undefined ? `<p class="note">Lyddane–Sachs–Teller: Π ω<sub>LO</sub><sup>2</sup>/ω<sub>TO</sub><sup>2</sup> = ${num(r.lst_ratio_freq, 4)}, q·ε<sub>0</sub>·q / q·ε<sub>∞</sub>·q = ${num(r.lst_ratio_eps, 4)}</p>` : ""}</div>`;
      }).join("")}</div>
      <p class="note">The output prints no LO frequencies: they are the eigenvalues of the TO modes' dynamical matrix with its non-analytic term, ω<sub>n</sub><sup>2</sup>δ<sub>nm</sub> + (4π/V)(Z<sub>n</sub>·q)(Z<sub>m</sub>·q)/(q·ε<sub>∞</sub>·q), from the mode Born charges, frequencies and ε<sub>∞</sub> the run prints (Gonze &amp; Lee, PRB 55, 10355 (1997)). Highlighted: the modes that split.</p>`));
    }
    return parts.join("");
  }
  /* what makes results unphysical (analysis.warnings: the report's WARNING lines) */
  const warningsHtml = (m) => ((m.data && m.data.warnings) || []).map((w) => `<p class="mv-warn" role="alert"><b>Warning:</b> ${esc(w)}</p>`).join("");
  function resultsHtml(m) {
    const w = warningsHtml(m);
    if (m.kind === "FREQ") return w + freqHtml(m);
    if (m.kind === "ELASTIC") return w + elasticHtml(m);
    if (m.kind === "EOS") return w + eosHtml(m);
    if (m.kind === "QHA") return w + qhaHtml(m);
    if (m.kind === "CPHF") return w + cphfHtml(m);
    return w;
  }

  /* ------------------------------------------------ plots */
  const figKey = (d) => `${d.tab}|${directional() ? d.opts.shape : ""}${d.tab === "enthalpy" && (d.opts.phases || []).length ? `|${d.opts.phases.join(",")}|${d.opts.fit || ""}` : ""}`;
  async function figData(d) {
    const k = figKey(d);
    if (!d.figs[k]) {
      const ph = d.tab === "enthalpy" ? d.opts.phases || [] : [];
      d.figs[k] = await api("elastic/figure?" + q({ path: d.path, view: d.tab, shape: directional() ? d.opts.shape : "surface", ...(ph.length && d.opts.fit ? { fit: d.opts.fit } : {}) })
        + ph.map((p) => `&compare=${encodeURIComponent(p)}`).join(""));
    }
    return d.figs[k];
  }
  function base() {
    const grid = cssVar("--line"), axis = { gridcolor: grid, showline: true, mirror: true, linecolor: grid, zeroline: false };
    return { margin: { l: 72, r: 18, t: 18, b: 48, autoexpand: false }, ...(H.plotPaper ? H.plotPaper() : { paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)" }),
      font: { color: cssVar("--text"), size: 12, family: cssVar("--font-sans") }, hovermode: "closest", showlegend: true, legend: PK.legendTop(),
      modebar: { bgcolor: "rgba(0,0,0,0)", color: cssVar("--text-2"), activecolor: cssVar("--accent") }, xaxis: { ...axis }, yaxis: { ...axis } };
  }
  const pal = () => Array.from({ length: 6 }, (_, k) => cssVar(`--plot-${k + 1}`));
  function linesFigure(f) {
    const lay = base(), P = pal(), o = doc.opts;
    let k = 0;
    const traces = f.series.map((s) => s.role === "points"
      ? { x: s.x, y: s.y, name: s.name, mode: s.line ? "lines+markers" : "markers", type: "scatter", meta: s.key || "points",
        marker: { color: s.hollow ? cssVar("--text-2") : cssVar("--text"), size: s.hollow ? 7 : 8, symbol: s.hollow ? "circle-open" : "circle" },
        ...(s.line ? { line: { color: cssVar("--text"), width: 1, dash: "dot" } } : {}),
        hovertemplate: f.view === "eos" ? `V %{x:.4f} Å<sup>3</sup><br>%{y:.3f} meV<extra>${s.name}</extra>` : `%{x:.5g}, %{y:.5g}<extra>${s.name}</extra>` }
      : { x: s.x, y: s.y, name: s.name, mode: s.marker ? "lines+markers" : "lines", type: "scatter", meta: s.key || s.name, ...(s.marker ? { marker: { size: 7 } } : {}),
        line: { color: s.key === "reference" ? cssVar("--text-3") : P[k++ % P.length], width: o.width || 1.5, dash: s.dash || "solid" } });
    lay.xaxis.title = { text: f.x_label };
    lay.yaxis.title = { text: f.y_label };
    if (f.exponent) lay.yaxis.exponentformat = "power";
    if (o.xr) lay.xaxis.range = o.xr;
    if (o.yr) lay.yaxis.range = o.yr;
    if (f.view === "pv" || f.zero_line) lay.shapes = [{ type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: 0, y1: 0, line: { color: cssVar("--text-3"), width: 1, dash: "dot" } }];
    if ((f.markers || []).length) {
      // the transition pressures where the enthalpies cross
      lay.shapes = [...(lay.shapes || []), ...f.markers.map((x) => ({ type: "line", xref: "x", x0: x, x1: x, yref: "paper", y0: 0, y1: 1, line: { color: cssVar("--text-2"), width: 1, dash: "dot" } }))];
      lay.annotations = f.markers.map((x) => ({ x, xref: "x", y: 1, yref: "paper", yanchor: "top", xanchor: "left", showarrow: false, text: `P<sub>t</sub> = ${x.toFixed(2)} GPa`, font: { size: 11, color: cssVar("--text-2") } }));
    }
    return { traces, layout: lay };
  }
  function surfaceFigure(f) {
    const lay = base(), grid = cssVar("--line"), ink = cssVar("--text");
    const ax = (t) => ({ title: { text: t }, gridcolor: grid, zerolinecolor: grid, color: ink, backgroundcolor: "rgba(0,0,0,0)", showbackground: false });
    lay.margin = { l: 0, r: 0, t: 8, b: 0 };
    lay.showlegend = false;
    lay.scene = { aspectmode: "data", xaxis: ax("x"), yaxis: ax("y"), zaxis: ax("z"), camera: { eye: { x: 1.25, y: 1.1, z: 0.85 } } };   // true proportions
    const u = f.unit ? ` ${f.unit}` : "";
    if (f.envelopes) {
      // ELATE's style: the minimum over m opaque green, the maximum a translucent blue shell, negative values red
      lay.showlegend = true;
      lay.legend = { x: 0.01, y: 0.99, bgcolor: "rgba(0,0,0,0)", font: { color: ink } };
      const tr = ["max", "min"].map((k) => {
        const s = f.envelopes[k], c = cssVar(ENV[k]), neg = cssVar(ENV.neg);
        return { type: "surface", x: s.x, y: s.y, z: s.z, surfacecolor: s.value.map((r) => r.map((v) => (v < 0 ? 0 : 1))), cmin: 0, cmax: 1,
          colorscale: [[0, neg], [0.5, neg], [0.5, c], [1, c]], showscale: false, opacity: k === "max" ? 0.28 : 1, customdata: s.value,
          name: `${f.symbol}<sub>${k}</sub>`, showlegend: true, meta: k,
          hovertemplate: `${f.symbol}<sub>${k}</sub> = %{customdata:.5g}${u}<extra></extra>`, lighting: { ambient: 0.8, diffuse: 0.45, specular: 0.05 } };
      });
      return { traces: tr, layout: lay };
    }
    const s = f.surface, neg = f.negative;
    const lim = neg ? Math.max(...s.value.flat().map((v) => Math.abs(v || 0))) : undefined;
    return { traces: [{ type: "surface", x: s.x, y: s.y, z: s.z, surfacecolor: s.value, colorscale: neg ? "RdBu" : "Viridis", cmid: neg ? 0 : undefined,
      cmin: neg ? -lim : undefined, cmax: neg ? lim : undefined,
      colorbar: { title: { text: `${f.symbol}${f.unit ? ` (${f.unit})` : ""}`, side: "right" }, thickness: 14, len: 0.75, tickfont: { color: ink }, outlinewidth: 0 },
      hovertemplate: `${f.symbol} = %{surfacecolor:.5g}${u}<extra></extra>`, lighting: { ambient: 0.75, diffuse: 0.5, specular: 0.05 }, name: f.label }], layout: lay };
  }
  function planesFigure(f) {
    const lay = base(), P = pal(), grid = cssVar("--line"), n = f.planes.length, gap = n > 1 ? 0.1 : 0, w = (1 - gap * (n - 1)) / n;
    const curves = (c) => c.curves || [{ name: f.symbol, key: c.plane, r: c.r }];
    const rmax = Math.max(...f.planes.flatMap((c) => curves(c).flatMap((k) => k.r.map((x) => Math.abs(x || 0))))) * 1.05 || 1;
    const multi = !!f.planes[0].curves, u = f.unit ? ` ${f.unit}` : "";
    lay.margin = { l: 30, r: 30, t: 40, b: multi ? 50 : 30 };
    lay.annotations = [];
    const traces = [];
    f.planes.forEach((c, i) => {
      const id = i ? `polar${i + 1}` : "polar", x0 = i * (w + gap) + (n === 1 ? 0.25 : 0), x1 = n === 1 ? 0.75 : x0 + w;
      lay[id] = { domain: { x: [x0, x1], y: [0, 0.94] }, bgcolor: "rgba(0,0,0,0)",
        radialaxis: { range: [0, rmax], gridcolor: grid, linecolor: grid, angle: 45, tickangle: 45, nticks: 4, tickfont: { size: 10 } },
        angularaxis: { gridcolor: grid, linecolor: grid, direction: "counterclockwise", rotation: 0, dtick: 90, tickfont: { size: 10 } } };
      const head = c.plane.length === 2 ? `${c.plane} plane` : `plane ${c.plane} (0° ${c.axes[0]}, 90° ${c.axes[1]})`;
      lay.annotations.push({ text: `${head} · ${f.symbol}${f.unit ? ` (${f.unit})` : ""}`, xref: "paper", yref: "paper", x: (x0 + x1) / 2, y: 1.0, xanchor: "center", yanchor: "bottom", showarrow: false, font: { size: 12 } });
      curves(c).forEach((k, j) => {
        const env = f.view === "shear" || f.view === "poisson";
        const color = f.view === "piezo" ? cssVar("--accent") : multi && env ? cssVar(ENV[k.key]) : multi ? P[(f.view === "sound" ? j % 3 : j) % P.length] : P[i % P.length];
        const dash = k.dash || (env && k.key === "max" ? "dash" : "solid");
        traces.push({ type: "scatterpolar", subplot: id, theta: c.theta_deg, r: k.r.map((x) => Math.abs(x)), mode: "lines", name: k.name, meta: `${c.plane}:${k.key}`,
          legendgroup: k.key, showlegend: multi && i === 0, line: { color, width: (doc.opts.width || 1.5) * (dash === "dot" ? 0.8 : 1), dash },
          hovertemplate: `%{theta:.0f}°: ${k.name} = %{customdata:.5g}${u}<extra>${c.plane}</extra>`, customdata: k.r });
        if (k.r.some((x) => x < 0)) {
          // a negative value (β, ν, d<sub>L</sub>) in red, at its |value|
          traces.push({ type: "scatterpolar", subplot: id, theta: c.theta_deg, r: k.r.map((x) => (x < 0 ? -x : null)), mode: "lines", name: `${k.name} < 0`,
            meta: `${c.plane}:${k.key}:neg`, legendgroup: "neg", showlegend: false, line: { color: cssVar(ENV.neg), width: doc.opts.width || 1.5, dash }, hoverinfo: "skip" });
        }
      });
    });
    lay.showlegend = multi;
    if (multi) lay.legend = { orientation: "h", x: 0.5, xanchor: "center", y: -0.04, yanchor: "top", bgcolor: "rgba(0,0,0,0)" };
    return { traces, layout: lay };
  }
  async function draw() {
    if (!doc || !doc.mech) return;
    const plot = doc.tab !== "results";
    body.hidden = plot; el.hidden = !plot;
    if (!plot) {
      el.dataset.drawn = "";
      body.innerHTML = resultsHtml(doc.mech);
      $(".pv-note").textContent = noteOf(doc.mech);
      body.dataset.drawn = doc.path;
      return;
    }
    if (building) { again = true; return; }
    building = true;
    el.dataset.drawn = "";
    const d = doc;
    try {
      const f = await figData(d);
      if (doc !== d || d.tab !== f.view) return;
      await loadPlotly();
      if (!window.Plotly) throw new Error("plotly is not installed in this Python; pip install plotly");
      const fig = f.shape === "surface" ? surfaceFigure(f) : f.shape === "planes" ? planesFigure(f) : linesFigure(f);
      $(".pv-note").innerHTML = figNote(f);
      $(".pv-note").title = $(".pv-note").textContent;
      if (H.styleFigure && !f.shape) H.styleFigure(fig.traces, fig.layout, "scf");
      const config = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"] };
      const s = `${d.id}|${d.tab}|${f.shape || ""}`;
      if (el.__sig !== s) { if (el.__sig) window.Plotly.purge(el); await window.Plotly.newPlot(el, fig.traces, fig.layout, config); el.__sig = s; }
      else await window.Plotly.react(el, fig.traces, fig.layout, config);
      if (!f.shape) await PK.fitLegends(el, { t: 18, r: 18 });
      el.dataset.drawn = `${d.path}|${d.tab}|${f.shape || ""}`;
      if (kit) kit.refresh();                 // the toolbar follows the plot drawn (no range for a surface)
    } catch (e) { $(".pv-note").textContent = e.message; } finally {
      building = false;
      if (again) { again = false; draw(); }
    }
  }
  function figNote(f) {
    if (f.glyph) return `${rich(f.note)}; drag to turn it`;
    const ex = f.extrema;
    if (ex && ex.min) {
      const u = f.unit ? ` ${rich(f.unit)}` : "", sym = rich(f.symbol);
      const v = (x) => sig(x, 5).replace("-", "−");
      const at = (e) => `${v(e.value)}${u} along ${esc(e.label)}${e.label2 ? ` (m ${esc(e.label2)})` : ""}`;
      const txt = f.view === "piezo" ? `largest |${sym}| ${v(Math.abs(ex.max.value))}${u} along ${esc(ex.max.label)}`
        : ex.isotropic ? `isotropic: ${v(ex.max.value)}${u} along every direction` : `from ${at(ex.min)} to ${at(ex.max)}`;
      const how = f.shape === "surface" ? (f.envelopes ? `. Green: the minimum over m ⊥ n; blue shell: the maximum${f.negative ? "; red: negative" : ""}; drag to turn it`
        : `. The distance from the center is ${sym.startsWith("|") ? sym : `|${sym}|`} along each direction; drag to turn it`) : "";
      return `${esc(f.title)}: ${txt}${how}${f.negative && !f.envelopes ? `; red where ${sym} < 0` : ""}`;
    }
    return rich(f.note || "");
  }
  function noteOf(m) {
    const d = m.data || {};
    const what = { 3: "3D crystal", 2: "slab", 1: "polymer", 0: "molecule" }[d.dimensionality] || "";
    return `${KIND_TITLE[m.kind] || ""} results of ${m.name}${what ? ` (${what})` : ""}, read as mace analyze elastic reads them`;
  }

  /* ------------------------------------------------ exports */
  function table() {
    if (!doc || !doc.mech) return { columns: [], rows: [] };
    if (doc.tab !== "results") {
      const f = doc.figs[figKey(doc)];
      return f ? f.table : { columns: [], rows: [] };
    }
    // the Results tab: every number of its tables, one row each
    const rows = [], D = doc.mech.data;
    const mat = (name, m, unit, rl, cl) => (m || []).forEach((r, i) => r.forEach((v, j) => rows.push([`${name}_${(rl || [])[i] || i + 1}${(cl || [])[j] ? "," + cl[j] : j + 1}`, v, unit])));
    if (D.elastic) mat("C", D.elastic.matrix, D.elastic.unit, D.elastic.matrix.length === 3 ? ["xx", "yy", "xy"] : VOIGT, D.elastic.matrix.length === 3 ? ["xx", "yy", "xy"] : VOIGT);
    if (D.compliance) mat("S", D.compliance.matrix, D.compliance.unit, VOIGT, VOIGT);
    for (const [k, v] of Object.entries(D.moduli || {})) if (typeof v === "number" || typeof v === "boolean") rows.push([k, v, ""]);
    for (const b of doc.mech.born || []) rows.push([`born: ${b.criterion}`, b.value, b.unit]);
    const der = D.derived || {};
    for (const [k, e] of Object.entries(der.directional || {})) {
      if (k === "sound") { for (const [m, x] of Object.entries(e)) for (const w of ["min", "max"]) { rows.push([`${m} phase ${w} ${x[w].label}`, x[w].value, "km/s"]); rows.push([`${m} group ${w} ${x.group[w].label}`, x.group[w].value, "km/s"]); } continue; }
      for (const w of ["min", "max"]) if (e[w]) rows.push([`${k} ${w} ${e[w].label}${e[w].label2 ? " m " + e[w].label2 : ""}`, e[w].value, e.unit || ""]);
    }
    for (const sec of ["indices", "two_d"]) for (const [k, v] of Object.entries(der[sec] || {})) if (typeof v === "number") rows.push([k, v, ""]);
    const ch = der.character || {};
    if (ch.pugh_ratio !== undefined) rows.push(["pugh_ratio", ch.pugh_ratio, ch.ductile ? "ductile" : "brittle"]);
    for (const c of ch.cauchy || []) rows.push([`cauchy ${c.relation.replace("−", "-")}`, c.value, "GPa"]);
    for (const k of ["chen_gpa", "tian_gpa"]) if ((ch.hardness || {})[k] !== undefined) rows.push([`vickers_hardness_${k.replace("_gpa", "")}`, ch.hardness[k], "GPa"]);
    for (const [k, v] of Object.entries(der.thermal || {})) if (typeof v === "number") rows.push([k, v, ""]);
    if (der.piezo && der.piezo.longitudinal && der.piezo.longitudinal.max) {
      rows.push([`${der.piezo.kind}_L max ${der.piezo.longitudinal.max.label}`, der.piezo.longitudinal.max.signed, der.piezo.unit]);
      const cp = der.piezo.coupling || {};
      if (cp.available) rows.push([`k2 max ${cp.k2_max.label}`, cp.k2_max.value, ""]);
    }
    for (const k of ["piezo_e", "piezo_d", "pockels", "piezo_optic"]) if (D[k]) mat(k, D[k].matrix, D[k].unit, k.startsWith("piezo_") && k !== "piezo_optic" ? AXES : VOIGT, VOIGT);
    for (const [n, f] of Object.entries((D.eos && D.eos.fits) || {})) for (const [k, v] of Object.entries(f)) rows.push([`${n} ${k}`, v, ""]);
    ((D.eos && D.eos.points) || []).forEach((p) => rows.push([`point V=${p[0]}`, p[1], "Ha"]));
    if (D.qha) D.qha.temperature_k.forEach((t, i) => QHA_COLS.forEach(([k, u]) => rows.push([`${k} T=${t}`, D.qha[k][i], u])));
    const th = D.qha && D.qha.thermo;
    if (th) th.temperature_k.forEach((t, i) => [["f_kj_mol", "kJ/mol"], ["entropy_j_mol_k", "J/(mol K)"], ["cv_j_mol_k", "J/(mol K)"], ["cp_j_mol_k", "J/(mol K)"], ["gamma_th", ""]]
      .forEach(([k, u]) => rows.push([`${k} T=${t}`, th[k][i], u])));
    if (doc.mech.cphf) for (const k of ["epsilon", "alpha", "chi1"]) mat(k, doc.mech.cphf[k], k === "alpha" ? "bohr^3" : "", AXES, AXES);
    for (const k of ["beta", "gamma"]) if (D[k]) D[k].components.forEach((r) => rows.push([`${k}_${r.component}`, r.value, "a.u."]));
    const op = D.optics || {};
    if (op.principal) op.principal.axis_labels.forEach((l, i) => { rows.push([`epsilon_principal ${l}`, op.principal.epsilon[i], ""]); rows.push([`n_principal ${l}`, op.principal.n[i], ""]); });
    if (op.principal) rows.push(["birefringence", op.principal.birefringence, ""]);
    if (op.shg && op.shg.zero === false) { rows.push([`shg d_L max ${op.shg.longitudinal.max.label}`, op.shg.longitudinal.max.value, "pm/V"]); rows.push(["shg powder <d_L^2>^1/2", op.shg.powder_longitudinal, "pm/V"]); }
    return { columns: ["quantity", "value", "unit"], rows };
  }
  const QHA_COLS = [["volume_ang3", "A^3"], ["alpha_v_per_k", "1/K"], ["bulk_modulus_t_gpa", "GPa"], ["bulk_modulus_s_gpa", "GPa"], ["cp_minus_cv_j_mol_k", "J/(mol K)"]];
  function cli() {
    if (!doc || !doc.mech) return "";
    if (doc.tab === "results") return doc.mech.cli;
    const f = doc.figs[figKey(doc)];
    return f ? f.cli : `${doc.mech.cli} --plot ${doc.tab}`;
  }
  const stem = () => (doc ? doc.path.split("/").pop().replace(/\.[^.]+$/, "") : "results");
  /** The permittivity of the piezoelectric coupling: the document again with it (--epsilon). */
  async function applyEpsilon() {
    const d = doc, i = body.querySelector(".mv-eps-in");
    if (!d || !i) return;
    const v = i.value.trim();
    const was = d.opts.epsilon || "";
    d.opts.epsilon = v;
    try { await fetchDoc(d); } catch (e) { d.opts.epsilon = was; toast(e.message, "error"); return; }
    persist(d);
    if (doc === d) { tabs(); draw(); if (H.loaded) H.loaded(d); }
  }

  /* ------------------------------------------------ open */
  async function fetchDoc(d) {
    d.mech = await api("elastic/document?" + q(d.opts && d.opts.epsilon ? { path: d.path, epsilon: d.opts.epsilon } : { path: d.path }));
    d.figs = {};
  }
  return {
    async open(d) {
      doc = d;
      load(d);
      $(".pv-title").textContent = d.title;
      $(".pv-title").title = d.title;
      if (!kit) kit = PK.attach($(".pv-acts"), {
        get name() { return doc ? `${stem()}_${doc.tab}` : "results"; },
        host: el,
        get features() {
          const t = doc ? doc.tab : "results", plot = t !== "results";
          const f = plot && doc ? doc.figs[figKey(doc)] : null, lines = plot && !(f && f.shape);
          return { logY: false, units: null, range: lines, width: plot && !(f && f.shape === "surface"), png: plot, svg: plot, csv: true, cli: true, pub: false };
        },
        get: () => (doc ? doc.opts : {}),
        set: (patch) => { if (!doc) return; Object.assign(doc.opts, patch); persist(doc); draw(); },
        axes: () => { const f = doc && doc.figs[figKey(doc)]; return { x: (f && f.x_label) || "x", y: (f && f.y_label) || "y" }; },
        buttons: () => (doc && doc.tab === "results" ? [{ key: "report", label: "Copy Report", title: "The text mace analyze elastic prints" }] : []),
        button: (k) => { if (k === "report" && doc && doc.mech) PK.copyText(doc.mech.report).then((ok) => toast(ok ? "Report copied" : "Copy failed (the browser refused the clipboard)", ok ? "success" : "error")); },
        table: () => table(),
        cli: () => cli(),
        png: async () => { const url = await window.Plotly.toImage(el, { format: "png", scale: Number(H.setting ? H.setting("plots.exportScale") : 2) || 2, width: el.clientWidth, height: el.clientHeight }); download(PK.dataUrlBlob(url), `${kit.name || "plot"}.png`); },
        svg: async () => { const url = await window.Plotly.toImage(el, { format: "svg", width: el.clientWidth, height: el.clientHeight }); download(PK.dataUrlBlob(url), `${kit.name || "plot"}.svg`); },
        resetZoom: () => { el.__sig = null; draw(); },
      }, { toast, download, icon: H.icon });
      tabs();
      if (!d.mech) {
        body.hidden = false; el.hidden = true;
        body.innerHTML = "";
        $(".pv-note").innerHTML = `<span class="spinner"></span> Reading ${esc(d.path.split("/").pop())}…`;
        try { await fetchDoc(d); } catch (e) { if (doc === d) { $(".pv-note").textContent = e.message; body.innerHTML = ""; } return; }
        if (doc !== d) return;
        // opened before its kind was known (Mechanical Properties Results…, a file not yet classified): name it now
        if (d.mech.kind && / · results$/.test(d.title)) { d.title = d.title.replace(/ · results$/, ` · ${d.mech.kind}`); d.calc = d.mech.kind; $(".pv-title").textContent = d.title; }
        if (H.loaded) H.loaded(d);
      }
      if (d.mech.empty) {
        $(".mv-tabs").innerHTML = "";
        body.hidden = false; el.hidden = true;
        $(".pv-note").textContent = "";
        body.innerHTML = `<div class="mv-empty"><p>No elastic, piezoelectric, photoelastic, equation-of-state, quasi-harmonic or dielectric results in ${esc(d.mech.name)} yet: the run is still going, stopped early, or is another kind of run.</p>
          <div class="mx-actions"><button class="tb-btn-text" data-mv="text">Open as Text</button><button class="tb-btn-text" data-mv="structure">Open as Structure</button></div></div>`;
        body.dataset.drawn = d.path;
        return;
      }
      tabs();
      if (kit) kit.refresh();
      draw();
    },
    async reload(d) { if (!d) return; d.mech = null; if (doc === d) await this.open(d); },
    /** Lets go of document ``d`` (any document without one). */
    close(d = null) { if (!d || doc === d) doc = null; },
    showTab: (t) => doc && showTab(t),
    setShape: (s) => { if (doc && doc.opts.shape !== s) { doc.opts.shape = s; persist(doc); tabs(); if (kit) kit.refresh(); draw(); } },
    state: () => doc && { tab: doc.tab, shape: doc.opts.shape, kind: doc.mech && doc.mech.kind, views: doc.mech ? (doc.mech.views || []).map((v) => v.key) : [],
      drawn: doc.tab === "results" ? (body.dataset.drawn || "") : (el.dataset.drawn || ""), cli: cli() },
    figure: () => (el.data ? { traces: el.data.map((t) => ({ type: t.type, name: t.name, x: t.x, y: t.y, r: t.r })), layout: el.layout } : null),
    table: () => table(),
    el: () => el,
    body: () => body,
    bind(onAct) {
      body.addEventListener("click", (e) => {
        const b = e.target.closest("[data-mv]");
        if (!b || !doc) return;
        if (b.dataset.mv === "epsilon") { applyEpsilon(); return; }
        onAct(b.dataset.mv, doc);
      });
      body.addEventListener("keydown", (e) => { if (e.key === "Enter" && e.target.closest(".mv-eps-in")) { e.preventDefault(); applyEpsilon(); } });
    },
    setEpsilon: (v) => { const i = body.querySelector(".mv-eps-in"); if (i) { i.value = v; return applyEpsilon(); } return null; },
  };
}
