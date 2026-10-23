/* Analysis › Phonons with Phonopy… (1.1.6): finite-displacement phonons
 * from a displaced set, each step the CLI's own command run as a task in the
 * set's folder (mace/gui/phonopy_routes.py):
 *   1 Forces     the displaced decks go to Submit (the walltime generate
 *                suggests; never tracked); the table shows each job's state
 *   2 Phonons    mace analyze phonopy DIR ... then the band structure, the
 *                DOS and the thermal properties, plotted here
 * The set itself (mace opt2d12 --out-file PARENT.out --calc-type PHONOPY --dim N N N ...) is
 * written in Calculation Setup › Phonopy (js/setup/modules.js); it opens
 * this dialog on the set it wrote. */
import { shell, val, esc, num, themed, accent } from "./mx_util.js";
import * as L from "../labels.js";

const STATE_CLASS = { done: "ok", failed: "bad", running: "run" };

/** opts: {path}: a phonopy set folder, or a CRYSTAL output (the parent: its folder's sets are listed). */
export async function open(H, opts = {}) {
  const S = H.shell();
  const p = String(opts.path || "");
  const isOut = /\.out$/i.test(p);
  const dirOf = (x) => (x.includes("/") ? x.slice(0, x.lastIndexOf("/")) : "");
  const ui = await shell(H, {
    id: "phonopy-dlg", title: "Phonons with Phonopy",
    tabs: [
      ["jobs", "1 · Forces", `
        <p class="note" data-k="info"></p>
        <p class="note mx-pointer"><span>The displaced supercells are written in Calculation Setup › Phonopy, on the optimized output
          (mace opt2d12 choice 8).</span> <button class="tb-btn-text" data-act="setup">Open Setup › Phonopy</button></p>
        <div class="mx-grid">
          <label for="ph-set">Displacement set</label><span class="mx-inline"><input id="ph-set" class="field" data-k="set" spellcheck="false" list="ph-sets" style="flex:1" placeholder="<stem>_ph<dim>_<functional>_optimized">
            <datalist id="ph-sets"></datalist><button class="tb-btn-text" data-act="refresh">Refresh</button></span>
        </div>
        <p class="note" data-k="set-sum" aria-live="polite"></p>
        <div class="mx-actions"><button class="tb-btn-text primary" data-act="submit">Submit the Jobs…</button>
          <span class="note">Each job's .ext is staged as fort.34 by the job script. Do not track them (no workflow step reads them).</span></div>
        <div data-k="set-warn"></div>
        <div style="max-height:260px;overflow:auto"><table class="mx-table" data-k="jobs"></table></div>`],
      ["post", "2 · Phonons", `
        <div class="mx-grid">
          <label for="ph-band">Band path</label><span class="mx-inline"><select id="ph-band" class="field" data-k="band-mode">
            <option value="auto">SeeK-path (automatic)</option><option value="custom">Custom (phonopy BAND points)</option><option value="none">None</option></select>
            <input class="field" data-k="band" spellcheck="false" placeholder="0 0 0  1/2 0 1/2, 1/2 1/2 1/2  0 0 0" style="flex:1" hidden></span>
          <label for="ph-labels">Band labels</label><input id="ph-labels" class="field" data-k="band_labels" placeholder="custom path only: G X L G">
          <label for="ph-bp">Points per segment</label><input id="ph-bp" type="number" class="field" data-k="band_points" min="5" max="501" placeholder="51">
          <label for="ph-mesh">DOS mesh</label><input id="ph-mesh" class="field" data-k="mesh" placeholder="100.0 (length) or &quot;a b c&quot;">
          <label>Temperatures (K)</label><span class="mx-inline"><input type="number" class="field" data-k="tmin" placeholder="0"> to
            <input type="number" class="field" data-k="tmax" placeholder="1000"> step <input type="number" class="field" data-k="tstep" placeholder="10"></span>
          <label for="ph-nac">NAC from</label><input id="ph-nac" class="field" data-k="nac_from" spellcheck="false" placeholder="optional: Γ FREQCALC .out with INTENS + INTCPHF (polar crystals)">
        </div>
        <div class="mx-actions"><button class="tb-btn-text" data-act="check">Check Outputs</button><button class="tb-btn-text primary" data-act="collect">Collect &amp; Compute Phonons</button>
          <label class="mx-check">Units <select class="field" data-k="unit"><option value="thz">THz</option><option value="cm">cm⁻¹</option></select></label>
          <span class="note" data-k="post-state" aria-live="polite"></span></div>
        <div class="mx-cmd" data-k="post-cmd"></div>
        <pre class="mx-pre" data-k="post-log" hidden></pre>
        <div class="mx-plots"><div class="mx-plot" data-k="band-plot" aria-label="Phonon band structure"></div><div class="mx-plot" data-k="dos-plot" aria-label="Phonon density of states"></div></div>
        <div class="mx-plot" data-k="thermal-plot" aria-label="Thermal properties" style="height:260px"></div>
        <p class="note" data-k="thermal-note"></p>`],
    ],
  });
  const { $, $$, show, d } = ui;
  const k = (name) => $(`[data-k="${name}"]`);
  let current = null, plotData = null;

  try {
    const inf = await H.api("phonopy/info");
    k("info").textContent = inf.phonopy
      ? `phonopy ${inf.version || ""} found${inf.seekpath ? ", seekpath found (automatic band paths)" : "; seekpath missing: give a custom band path"}.`
      : `phonopy is not installed where the MACE server runs: ${inf.install}`;
    if (!inf.seekpath) k("band-mode").value = "custom";
  } catch (e) { k("info").textContent = e.message; }

  async function listSets() {
    const base = p && !isOut ? p : dirOf(p);
    try {
      const r = await H.api("phonopy/sets?" + new URLSearchParams({ dir: base || "." }));
      $("#ph-sets").innerHTML = r.sets.map((s) => `<option value="${esc(s.dir)}">${esc(s.parent)} · ${s.jobs} jobs${s.collected ? " · collected" : ""}</option>`).join("");
      if (!k("set").value && r.sets.length) k("set").value = r.sets[0].dir;
    } catch (e) { /* a folder without sets */ }
  }
  async function refresh() {
    const dir = k("set").value.trim();
    if (!dir) { k("set-sum").textContent = "Pick the folder generate wrote (or write one in Setup › Phonopy first)."; return; }
    try {
      current = await H.api("phonopy/status?" + new URLSearchParams({ dir }));
    } catch (e) { current = null; k("set-sum").textContent = e.message; k("jobs").innerHTML = ""; return; }
    const c = current.counts || {};
    k("set-sum").textContent = `${current.jobs.length} displaced job(s): ${["done", "running", "failed", "not run"].filter((s) => c[s]).map((s) => `${c[s]} ${s}`).join(", ")}` +
      ` · supercell ${current.n_atoms_supercell} atoms · suggested walltime ${current.walltime || "?"}` +
      (current.results.length ? ` · results: ${current.results.join(", ")}` : "");
    k("set-warn").innerHTML = (current.warnings || []).map((w) => `<p class="note">⚠ ${esc(w)}</p>`).join("");
    k("jobs").innerHTML = `<tr><th class="l">Job</th><th class="l">State</th><th class="l">Note</th></tr>` + current.jobs.map((j) =>
      `<tr><td class="l">${esc(j.name)}</td><td class="l"><span class="mx-badge ${STATE_CLASS[j.state] || ""}">${esc(j.state)}</span></td><td class="l">${esc(j.reason || "")}</td></tr>`).join("");
    if (current.results.includes("band.yaml") || current.results.includes("total_dos.dat")) loadPlots();
  }
  function body(keys) {
    const b = {};
    for (const n of keys) { const v = val(k(n)); if (v !== undefined && v !== false) b[n] = v; }
    return b;
  }
  // the set is written in Setup: Calculation › Regular Calculations › 8 Phonopy on the parent output
  $('[data-act="setup"]').addEventListener("click", () => {
    const parent = isOut ? p : (current && current.parent_out) || "";
    d.close();
    H.run("calc.phonopy", parent ? { path: parent } : {});
  });
  $('[data-act="refresh"]').addEventListener("click", () => { listSets(); refresh(); });
  k("set").addEventListener("change", refresh);
  $('[data-act="submit"]').addEventListener("click", () => {
    const dir = k("set").value.trim();
    if (!dir) { H.toast("Pick a displacement set first", "info"); return; }
    S.builder("submit_dialog", "open", { targets: [dir], preset: current && current.walltime ? { walltime: current.walltime } : null });
  });
  k("band-mode").addEventListener("change", () => { k("band").hidden = k("band-mode").value !== "custom"; });
  k("band").hidden = k("band-mode").value !== "custom";
  async function collect(check) {
    const dir = k("set").value.trim();
    if (!dir) { show("jobs"); k("set-sum").textContent = "Pick the displacement set first."; return; }
    const mode = k("band-mode").value;
    const b = { dir, check, band: mode === "custom" ? (k("band").value.trim() || "auto") : mode,
      ...body(["band_labels", "band_points", "mesh", "tmin", "tmax", "tstep", "nac_from"]) };
    k("post-state").textContent = check ? "Checking the outputs…" : "Reading the forces and computing the phonons…";
    try {
      const r = await S.apiTask("phonopy/collect", b);
      k("post-cmd").textContent = r.command || "";
      k("post-log").hidden = false;
      k("post-log").textContent = r.log || "";
      k("post-state").textContent = r.code === 0 ? (check ? "All outputs usable." : `Wrote ${r.results.join(", ")}`) : "Collect stopped: see the log";
      if (!check && r.code === 0) { await refresh(); await loadPlots(); }
    } catch (e) { k("post-state").textContent = e.message; }
  }
  $('[data-act="check"]').addEventListener("click", () => collect(true));
  $('[data-act="collect"]').addEventListener("click", () => collect(false));
  k("unit").addEventListener("change", () => { if (plotData) drawPlots(); });

  async function loadPlots() {
    const dir = k("set").value.trim();
    if (!dir) return;
    try { plotData = await H.api("phonopy/data?" + new URLSearchParams({ dir })); } catch (e) { plotData = null; return; }
    await H.M().loadPlotly();
    drawPlots();
  }
  function drawPlots() {
    if (!window.Plotly || !plotData) return;
    const f = k("unit").value === "cm" ? plotData.cm1_per_thz : 1;
    const unit = L.unit(k("unit").value === "cm" ? "cm-1" : "THz");
    const col = accent();
    if (plotData.band) {
      const b = plotData.band;
      const traces = b.bands.map((y) => ({ x: b.distances, y: y.map((v) => v * f), mode: "lines", line: { color: col, width: 1.5 }, hoverinfo: "y" }));
      Plotly.react(k("band-plot"), traces, themed({ yaxis: { title: { text: L.withUnit("Frequency", unit) }, zeroline: true },
        xaxis: { tickvals: b.ticks.map((t) => t.x), ticktext: b.ticks.map((t) => t.label.replace(/\$\\Gamma\$|\\Gamma/g, "Γ")), showgrid: true, range: [b.distances[0], b.distances[b.distances.length - 1]] } }),
        { displaylogo: false, responsive: true });
    }
    if (plotData.dos) {
      const dd = plotData.dos;
      Plotly.react(k("dos-plot"), [{ x: dd.dos, y: dd.frequency.map((v) => v * f), mode: "lines", fill: "tozerox", line: { color: col, width: 1.5 } }],
        themed({ xaxis: { title: { text: "DOS" }, showticklabels: false }, yaxis: { title: { text: L.withUnit("Frequency", unit) } } }), { displaylogo: false, responsive: true });
    }
    const t = plotData.thermal;
    if (t && t.temperature) {
      Plotly.react(k("thermal-plot"), [
        // phonopy's C_V and S per mole of cells (labels.js units)
        { x: t.temperature, y: t.heat_capacity, name: L.withUnit(L.SYMBOLS.C_V, "J/(K mol)"), mode: "lines" },
        { x: t.temperature, y: t.entropy, name: L.withUnit("S", "J/(K mol)"), mode: "lines" },
        { x: t.temperature, y: t.free_energy, name: L.withUnit("F", "kJ/mol"), mode: "lines", yaxis: "y2" }],
        themed({ showlegend: true, legend: { orientation: "h" }, xaxis: { title: { text: L.withUnit("T", "K") } }, yaxis: { title: { text: L.withUnit(`${L.SYMBOLS.C_V}, S`, "J/(K mol)") } },
          yaxis2: { overlaying: "y", side: "right", title: { text: L.withUnit("F", "kJ/mol") }, showgrid: false } }), { displaylogo: false, responsive: true });
      const i300 = t.temperature.findIndex((x) => Math.abs(x - 300) < 1e-6);
      k("thermal-note").textContent = `Zero-point energy ${num(t.zero_point_energy, 3)} kJ/mol (per primitive cell)` +
        (i300 >= 0 ? `; at 300 K: C_v ${num(t.heat_capacity[i300], 2)} J/(K mol), S ${num(t.entropy[i300], 2)} J/(K mol), F ${num(t.free_energy[i300], 3)} kJ/mol` : "");
    }
  }
  d.addEventListener("mx-tab", (e) => { if (e.detail === "jobs") { listSets().then(refresh); } if (e.detail === "post" && !plotData) loadPlots(); });
  if (!isOut && p) { k("set").value = p; await refresh(); show(current && current.results.length ? "post" : "jobs"); }
  else listSets().then(refresh);
}
