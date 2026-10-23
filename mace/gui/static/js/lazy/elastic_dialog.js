/* Analysis › Mechanical Properties Results… (1.1.6): the tensors, the
 * Voigt-Reuss-Hill moduli, Born stability, sound velocities, EOS fits and QHA
 * properties of a finished ELASTCON, ELAPIEZO, PIEZOCON, PIEZOCP, PHOTOELA,
 * EOS or QHA run (`mace analyze elastic --json`). Their decks (CRYSTAL23's
 * ELASTCON, ELAPIEZO, PIEZOCON, PIEZOCP, PHOTOELA, EOS, QHA and CPHF, manual
 * ch. 9, 10, 12-13) are written in Calculation Setup › 4 ELASTIC ... 7 CPHF
 * (js/setup/modules.js: mace opt2d12 --calc-type). */
import { shell, esc, num, themed, accent, matrixTable, VOIGT, copy } from "./mx_util.js";

/** opts: {path}: a CRYSTAL output (a finished elastic/EOS/QHA run; another output fills the field). */
export async function open(H, opts = {}) {
  const p = String(opts.path || "");
  const ui = await shell(H, {
    id: "elastic-dlg", title: "Mechanical Properties Results",
    tabs: [
      ["results", "Results", `
        <p class="note mx-pointer"><span>The decks are written in Calculation Setup › 4 ELASTIC (5 EOS, 6 QHA, 7 CPHF), on the optimized output
          (mace opt2d12 choices 4-7).</span> <button class="tb-btn-text" data-act="setup">Open Setup › ELASTIC</button></p>
        <div class="mx-grid"><label for="el-run">Output</label><span class="mx-inline"><input id="el-run" class="field" data-k="run" spellcheck="false" style="flex:1" placeholder="a finished ELASTCON / ELAPIEZO / PIEZOCON / PHOTOELA / EOS / QHA output">
          <button class="tb-btn-text primary" data-act="analyze">Analyze</button><button class="tb-btn-text" data-act="copy">Copy Report</button>
          <button class="tb-btn-text" data-act="doc" title="The output's results document: these tables, its plots (Young's modulus, E(V), V(T), ...) and its settings in the Inspector, as the Explorer opens it">Open as Document</button></span></div>
        <p class="note" data-k="r-state" aria-live="polite"></p>
        <div data-k="r-body" class="mx-section"></div>`],
    ],
  });
  const { $, d } = ui;
  const k = (n) => $(`[data-k="${n}"]`);
  let report = "";
  const isRun = /_(elastcon|elapiezo|piezocon|piezocp|photoela|eos|qha|cphf)[_.]/i.test(p);
  if (/\.out$/i.test(p)) k("run").value = p;
  // the deck: Calculation › Regular Calculations › 4-7 on the parent output (a finished run is not one)
  $('[data-act="setup"]').addEventListener("click", () => {
    d.close();
    H.run("calc.mechanics", /\.out$/i.test(p) && !isRun ? { path: p } : {});
  });

  async function analyze() {
    const path = k("run").value.trim();
    if (!path) { k("r-state").textContent = "Give an output"; return; }
    k("r-state").textContent = "Reading…";
    let r;
    try { r = await H.api("elastic/analyze?" + new URLSearchParams({ path })); } catch (e) { k("r-state").textContent = e.message; return; }
    report = r.report;
    if (r.empty) { k("r-state").textContent = "No elastic, piezoelectric, photoelastic or EOS results in this output (still running, or another kind of run)."; k("r-body").innerHTML = ""; return; }
    k("r-state").textContent = "";
    const D = r.data, parts = [];
    if (D.elastic) parts.push(`<h3>Elastic constants C (${esc(D.elastic.unit)}${D.elastic.system ? ", " + esc(D.elastic.system.toLowerCase()) : ""})</h3>` +
      matrixTable(D.elastic.matrix, { cols: VOIGT, rows: VOIGT }));
    const m = D.moduli;
    if (m && m.bulk_modulus_hill_gpa !== undefined) {
      parts.push(`<h3>Moduli (GPa) <span class="mx-badge ${m.born_stable ? "ok" : "bad"}">${m.born_stable ? "mechanically stable" : "UNSTABLE"}</span></h3>
        <table class="mx-table"><tr><th></th><th>Voigt</th><th>Reuss</th><th>Hill</th></tr>
        <tr><th class="l">Bulk K</th><td>${num(m.bulk_modulus_voigt_gpa, 2)}</td><td>${num(m.bulk_modulus_reuss_gpa, 2)}</td><td>${num(m.bulk_modulus_hill_gpa, 2)}</td></tr>
        <tr><th class="l">Shear G</th><td>${num(m.shear_modulus_voigt_gpa, 2)}</td><td>${num(m.shear_modulus_reuss_gpa, 2)}</td><td>${num(m.shear_modulus_hill_gpa, 2)}</td></tr></table>
        <p class="note">Young's modulus ${num(m.youngs_modulus_gpa, 1)} GPa (x, y, z: ${(m.youngs_modulus_xyz_gpa || []).map((x) => num(x, 1)).join(", ")}),
        Poisson's ratio ${num(m.poisson_ratio, 3)}, Pugh K/G ${num(m.pugh_ratio, 2)}, universal anisotropy ${num(m.universal_anisotropy, 3)},
        smallest eigenvalue of C ${num(m.min_eigenvalue_gpa, 2)} GPa${m.debye_temperature_k ? `; sound velocities ${num(m.sound_velocity_transverse_km_s, 2)} / ${num(m.sound_velocity_longitudinal_km_s, 2)} km/s (transverse / longitudinal), Debye temperature ${num(m.debye_temperature_k, 0)} K` : ""}.</p>` +
        (D.crystal_vrh ? `<p class="note">CRYSTAL's own VRH: K_H ${num(D.crystal_vrh.K_H, 2)}, G_H ${num(D.crystal_vrh.G_H, 2)}, E_H ${num(D.crystal_vrh.E_H, 2)} GPa, ν_H ${num(D.crystal_vrh.v_H, 3)}</p>` : ""));
    } else if (m && m.layer_modulus !== undefined) {
      parts.push(`<h3>In-plane moduli (${esc(m.unit)}) <span class="mx-badge ${m.born_stable ? "ok" : "bad"}">${m.born_stable ? "stable" : "UNSTABLE"}</span></h3>
        <p class="note">E_x ${num(m.youngs_modulus_x)}, E_y ${num(m.youngs_modulus_y)}, ν_xy ${num(m.poisson_ratio_xy)}, G_xy ${num(m.shear_modulus_xy)}, layer modulus ${num(m.layer_modulus)}</p>`);
    }
    if (D.moduli_error) parts.push(`<p class="err">${esc(D.moduli_error)}</p>`);
    if (D.seismic) parts.push(`<h3>Seismic velocities (km/s)</h3><table class="mx-table"><tr><th class="l">Direction</th><th>V_p</th><th>V_s1</th><th>V_s2</th></tr>${
      D.seismic.map((v) => `<tr><td class="l">[${v.direction.map((x) => num(x, 2)).join(" ")}]</td><td>${num(v.vp)}</td><td>${num(v.vs1)}</td><td>${num(v.vs2)}</td></tr>`).join("")}</table>`);
    for (const [key, title, rows] of [["piezo_e", "Direct piezoelectric tensor e", ["x", "y", "z"]], ["piezo_d", "Converse piezoelectric tensor d", ["x", "y", "z"]],
      ["pockels", "Elasto-optic (Pockels) tensor p", VOIGT], ["piezo_optic", "Piezo-optic tensor π", VOIGT]]) {
      if (D[key]) parts.push(`<h3>${title} (${esc(D[key].unit)})</h3>` + matrixTable(D[key].matrix, { cols: VOIGT, rows }));
    }
    // a slab's or polymer's QHA (and its EOS fit) is per area or length: CRYSTAL's own units
    const units = (u) => {
      const v = u || { size: "V", size_unit: "A^3", modulus_unit: "GPa" };
      const s = (x) => x.replace("A^3", "Å³").replace("A^2", "Å²").replace(/\bA\b/, "Å").replace(" x ", "·");
      return { size: v.size, size_unit: s(v.size_unit), modulus_unit: s(v.modulus_unit) };
    };
    if (D.eos) {
      const f = D.eos.fits || {}, u = units(D.eos.units);
      parts.push(`<h3>Equation of state</h3><table class="mx-table"><tr><th class="l">EOS</th><th>${u.size}₀ (${u.size_unit})</th><th>E₀ (Ha)</th><th>B₀ (${u.modulus_unit})</th><th>B₀′</th></tr>${
        Object.entries(f).map(([n, x]) => `<tr><td class="l">${esc(n.replace(/_/g, "-"))}</td><td>${num(x.volume_ang3, 4)}</td><td>${num(x.energy_au, 8)}</td><td>${num(x.bulk_modulus_gpa, 2)}</td><td>${num(x.bulk_modulus_prime, 2)}</td></tr>`).join("")}</table>
        <div class="mx-plot" data-k="eos-plot" aria-label="E(V)"></div>`);
    }
    if (D.qha) {
      const q = D.qha, n = q.temperature_k.length, pick = [...new Set([0, n >> 2, n >> 1, (3 * n) >> 2, n - 1])], u = units(q.units);
      parts.push(`<h3>Quasi-harmonic thermal properties (P = 0)</h3><table class="mx-table"><tr><th>T (K)</th><th>${u.size} (${u.size_unit})</th><th>K_T (${u.modulus_unit})</th><th>α_${u.size} (1/K)</th><th>K_S (${u.modulus_unit})</th><th>C_p−C_v (J/mol K)</th></tr>${
        pick.map((i) => `<tr><td>${num(q.temperature_k[i], 1)}</td><td>${num(q.volume_ang3[i], 4)}</td><td>${num(q.bulk_modulus_t_gpa[i], 2)}</td><td>${Number(q.alpha_v_per_k[i]).toExponential(3)}</td><td>${num(q.bulk_modulus_s_gpa[i], 2)}</td><td>${num(q.cp_minus_cv_j_mol_k[i], 3)}</td></tr>`).join("")}</table>
        <div class="mx-plot" data-k="qha-plot" aria-label="V(T) and thermal expansion"></div>`);
    }
    k("r-body").innerHTML = parts.join("");
    if (D.qha) {
      await H.M().loadPlotly();
      if (window.Plotly) {
        const q = D.qha, u = units(q.units), sz = `${u.size} (${u.size_unit})`, al = `α_${u.size} (1/K)`;
        Plotly.react(k("qha-plot"), [{ x: q.temperature_k, y: q.volume_ang3, name: sz, mode: "lines", line: { color: accent() } },
          { x: q.temperature_k, y: q.alpha_v_per_k, name: al, mode: "lines", yaxis: "y2" }],
        themed({ showlegend: true, legend: { orientation: "h" }, xaxis: { title: { text: "T (K)" } }, yaxis: { title: { text: sz } },
          yaxis2: { overlaying: "y", side: "right", title: { text: al }, showgrid: false, exponentformat: "e" } }), { displaylogo: false, responsive: true });
      }
    }
    if (D.eos && D.eos.points) {
      await H.M().loadPlotly();
      if (window.Plotly) {
        const pts = D.eos.points;
        Plotly.react(k("eos-plot"), [{ x: pts.map((q) => q[0]), y: pts.map((q) => q[1]), mode: "markers+lines", marker: { color: accent(), size: 8 }, line: { color: accent(), width: 1 } }],
          themed({ xaxis: { title: { text: `${units(D.eos.units).size} (${units(D.eos.units).size_unit})` } }, yaxis: { title: { text: "E (Ha)" } } }), { displaylogo: false, responsive: true });
      }
    }
  }
  $('[data-act="analyze"]').addEventListener("click", analyze);
  // the same output as a document (js/docs/mech.js): the tables, the plots and the Inspector's settings
  $('[data-act="doc"]').addEventListener("click", () => {
    const path = k("run").value.trim();
    if (!path) { k("r-state").textContent = "Give an output"; return; }
    d.close();
    H.shell().openMech(path);
  });
  $('[data-act="copy"]').addEventListener("click", () => { if (report) copy(H, report, "Report copied"); });
  if (opts.tab === "results" || isRun) analyze();
}
