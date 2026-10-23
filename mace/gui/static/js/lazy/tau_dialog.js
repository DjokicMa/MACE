/* Analysis › Transport Relaxation Time… (1.1.6): a physically documented
 * tau for BOLTZTRA's RELAXTIM (mace/transport, `mace tau`).
 *   Estimate          deformation-potential theory from m*, E1 and C (or tau
 *                     from a mobility you have)
 *   Strained Runs     E1 and C fitted from the strained SP outputs, then tau;
 *                     m* empty: the masses from the BAND / DOSS runs beside
 *                     the series or given (mace tau fit --m-star auto)
 *   Strain Decks      the strained SP decks of an SP parent (ELASTIC)
 * Every result shows its method, formula, reference and where each input
 * came from. */
import { shell, val, esc, num, copy } from "./mx_util.js";

const C_UNIT = { 3: "GPa", 2: "N/m", 1: "J/m" };

/** A mass and where it came from, short: "from BAND: 0.297 mₑ (Γ→X)", "from DOSS: 0.535 mₑ",
 * "= m_d: 0.535 mₑ". ``source`` is the CLI's sentence (mace/transport/mass.py auto_masses). */
export function massHint(source, value) {
  const s = String(source || ""), m = `${num(value, 3)} mₑ`;
  if (/^band curvature of /.test(s)) {
    const along = /\balong ([^(]+?)(?: \(|$)/.exec(s.slice(s.indexOf(":") + 1));
    return `from BAND: ${m}` + (along ? ` (${along[1].trim().replace(/^the path directions through /, "mean through ")})` : "");
  }
  if (/^DOS fit of /.test(s)) return `from DOSS: ${m}`;
  if (/^taken equal to m_d/.test(s)) return `= m_d: ${m}`;
  if (/^taken equal to m_c/.test(s)) return `= m*: ${m}`;
  return `${m}: ${s}`;
}

/** opts: {path}: a TRANSPORT/SP output or deck, or a folder with a strain series. */
export async function open(H, opts = {}) {
  const S = H.shell();
  const p = String(opts.path || "");
  const dirOf = (x) => (x.includes("/") ? x.slice(0, x.lastIndexOf("/")) : "");
  const ui = await shell(H, {
    id: "tau-dlg", title: "Transport Relaxation Time",
    tabs: [
      ["est", "Estimate", `
        <div class="mx-grid">
          <label for="tau-dim">System</label><select id="tau-dim" class="field" data-k="dim"><option value="3">3D crystal</option><option value="2">2D slab</option><option value="1">1D polymer</option></select>
          <label for="tau-m">m* (mₑ)</label><input id="tau-m" type="number" step="any" class="field" data-k="m_star" placeholder="effective mass along transport">
          <label for="tau-mo" data-show="2">m* across (mₑ)</label><input id="tau-mo" type="number" step="any" class="field" data-k="m_other" data-show="2" placeholder="2D: the in-plane mass across transport (default m*)">
          <label for="tau-e1">E1 (eV)</label><input id="tau-e1" type="number" step="any" class="field" data-k="e1" placeholder="deformation potential">
          <label for="tau-c" data-k="c-label">C (GPa)</label><input id="tau-c" type="number" step="any" class="field" data-k="c" placeholder="elastic constant along transport">
          <label for="tau-t">T (K)</label><input id="tau-t" type="number" step="any" class="field" data-k="temperature" value="300">
          <label>Instead</label><span class="mx-inline"><label class="mx-check"><input type="checkbox" data-k="use-mob"> From a mobility</label>
            <input type="number" step="any" class="field" data-k="mobility" placeholder="cm²/(V s)" disabled></span>
        </div>
        <div class="mx-actions"><button class="tb-btn-text primary" data-act="estimate">Compute τ</button><button class="tb-btn-text" data-act="copy-est">Copy</button></div>
        <p class="note" data-k="est-key" aria-live="polite"></p><pre class="mx-pre" data-k="est-out" hidden></pre>`],
      ["fit", "Strained Runs", `
        <div class="mx-grid">
          <label for="tau-dir">Folder</label><span class="mx-inline"><input id="tau-dir" class="field" data-k="dir" spellcheck="false" style="flex:1"><button class="tb-btn-text" data-act="scan">Find Series</button></span>
          <label for="tau-ser">Strain series</label><select id="tau-ser" class="field" data-k="series"></select>
          <label for="tau-car">Carrier</label><select id="tau-car" class="field" data-k="carrier"><option value="electron">electrons (CBM)</option><option value="hole">holes (VBM)</option></select>
          <label for="tau-fm">m* (mₑ)</label><input id="tau-fm" type="number" step="any" class="field" data-k="f_m_star" placeholder="auto: from the BAND/DOSS beside the series">
          <label data-k="fm-hint-l" hidden></label><p class="note" data-k="fm-hint" aria-live="polite" hidden></p>
          <label for="tau-bd">BAND.DAT</label><input id="tau-bd" class="field" data-k="band_dat" spellcheck="false" placeholder="optional: m* from the band curvature">
          <label for="tau-dos">DOSS</label><input id="tau-dos" class="field" data-k="dos_file" spellcheck="false" placeholder="optional: m_d from the DOS">
          <label data-k="md-hint-l" hidden></label><p class="note" data-k="md-hint" hidden></p>
          <label for="tau-fmo">m* across (2D)</label><input id="tau-fmo" type="number" step="any" class="field" data-k="f_m_other">
          <label for="tau-cf">C from ELASTCON</label><input id="tau-cf" class="field" data-k="c_from" spellcheck="false" placeholder="optional 3D: relaxed-ion C_ii of an ELASTCON output">
          <label for="tau-fe1">E1 override (eV)</label><input id="tau-fe1" type="number" step="any" class="field" data-k="f_e1">
          <label for="tau-ft">T (K)</label><input id="tau-ft" type="number" step="any" class="field" data-k="f_temperature" value="300">
          <label></label><label class="mx-check"><input type="checkbox" data-k="save"> Save the result as tau_&lt;carrier&gt;_&lt;axis&gt;.json next to the outputs</label>
        </div>
        <div class="mx-actions"><button class="tb-btn-text primary" data-act="fit">Fit E1 and C, Compute τ</button><button class="tb-btn-text" data-act="copy-fit">Copy</button>
          <button class="tb-btn-text" data-act="save-db" disabled title="Store τ and the mobility with the unstrained run in its folder's materials.db (Properties tab)">Save to Database</button></div>
        <p class="note" data-k="fit-key" aria-live="polite"></p><p class="note" data-k="db-state" aria-live="polite"></p><div data-k="fit-warn"></div><pre class="mx-pre" data-k="fit-out" hidden></pre>`],
      ["decks", "Strain Decks", `
        <div class="mx-grid">
          <label for="tau-par">SP deck</label><input id="tau-par" class="field" data-k="parent" spellcheck="false" placeholder="mat_sp.d12 of the optimized structure (one-component)">
          <label for="tau-ax">Axis</label><select id="tau-ax" class="field" data-k="axis"><option>x</option><option>y</option><option>z</option></select>
          <label for="tau-st">Strains</label><input id="tau-st" class="field" data-k="strains" value="-0.01,-0.005,0.005,0.01">
          <label></label><label class="mx-check"><input type="checkbox" data-k="force"> Replace decks that exist</label>
        </div>
        <p class="note">Each deck is the parent with CRYSTAL's ELASTIC strain (manual p. 47) after the atoms, plus the unstrained one at the same TOLDEE.
          Run them all (Submit), then fit them on the Strained Runs tab.</p>
        <div class="mx-actions"><button class="tb-btn-text primary" data-act="write">Write Strain Decks</button><button class="tb-btn-text" data-act="submit">Submit…</button></div>
        <p class="note" data-k="w-state" aria-live="polite"></p><div class="mx-cmd" data-k="w-cmd"></div>`],
    ],
  });
  const { $, $$, show } = ui;
  const k = (n) => $(`[data-k="${n}"]`);
  let estText = "", fitText = "", written = [];

  const dimChanged = () => {
    const dim = k("dim").value;
    $$('[data-show="2"]').forEach((el) => { el.hidden = dim !== "2"; });
    k("c-label").textContent = `C (${C_UNIT[dim]})`;
  };
  k("dim").addEventListener("change", dimChanged);
  dimChanged();
  k("use-mob").addEventListener("change", () => {
    const on = k("use-mob").checked;
    k("mobility").disabled = !on;
    ["e1", "c", "dim", "m_other"].forEach((n) => { k(n).disabled = on; });
  });
  const keyLine = (r) => `τ = ${num(r.tau_fs, 2)} fs · mobility ${num(r.mobility_cm2_Vs, 1)} cm²/(V s) · for BOLTZTRA: ${r.relaxtim.replace("\n", " ")}`;

  $('[data-act="estimate"]').addEventListener("click", async () => {
    const b = k("use-mob").checked ? { mobility: val(k("mobility")), m_star: val(k("m_star")) }
      : { dim: Number(k("dim").value), m_star: val(k("m_star")), m_other: val(k("m_other")), e1: val(k("e1")), c: val(k("c")), temperature: val(k("temperature")) };
    try {
      const r = await H.api("tau/estimate", { body: b });
      estText = r.report;
      k("est-key").textContent = keyLine(r.result);
      k("est-out").hidden = false; k("est-out").textContent = r.report;
    } catch (e) { k("est-key").textContent = e.message; }
  });
  $('[data-act="copy-est"]').addEventListener("click", () => { if (estText) copy(H, estText, "Copied"); });

  async function scan() {
    const dir = k("dir").value.trim() || ".";
    try {
      const r = await H.api("tau/series?" + new URLSearchParams({ dir }));
      k("series").innerHTML = r.series.length ? r.series.map((s, i) => `<option value="${i}">${esc(s.name)} (${s.outputs.length} outputs)</option>`).join("")
        : `<option value="">no <stem>_dp_<axis>_*pct.out in ${esc(r.dir || ".")}</option>`;
      k("series")._data = r.series;
    } catch (e) { k("fit-key").textContent = e.message; }
  }
  $('[data-act="scan"]').addEventListener("click", scan);
  /* the fit's request, as shown; Save to Database sends it again with save_db (the server fits again, then stores) */
  let lastFit = null;
  $('[data-act="fit"]').addEventListener("click", async () => {
    const ser = (k("series")._data || [])[Number(k("series").value)];
    if (!ser) { k("fit-key").textContent = "Find a strain series first"; return; }
    // m* empty: automatic, from the BAND.DAT / DOSS given here or the BAND and DOSS runs beside the series
    const b = { outputs: ser.outputs, carrier: k("carrier").value, m_star: val(k("f_m_star")), band_dat: val(k("band_dat")),
      dos_file: val(k("dos_file")), m_other: val(k("f_m_other")), c_from: val(k("c_from")), e1: val(k("f_e1")),
      temperature: val(k("f_temperature")), save: k("save").checked };
    k("fit-key").textContent = "Fitting…";
    k("db-state").textContent = "";
    $('[data-act="save-db"]').disabled = true;
    lastFit = null;
    massHints(null);
    try {
      const r = await H.api("tau/fit", { body: b });
      lastFit = { ...b, save: false };
      $('[data-act="save-db"]').disabled = false;
      const ms = r.masses || null;
      fitText = (ms && ms.report ? ms.report + "\n\n" : "") + r.report;
      const ft = r.result.fit;
      k("fit-key").textContent = `E1 ${num(ft.E1_eV, 3)} eV (R² ${num(ft.E1_r2, 3)}) · C ${num(ft.C, 4)} ${ft.C_unit} (R² ${num(ft.C_r2, 4)}) · ` + keyLine(r.result) +
        (r.saved ? ` · saved ${r.saved}` : "");
      k("fit-warn").innerHTML = [...(ft.warnings || []), ...((ms && ms.warnings) || [])].map((w) => `<p class="err">⚠ ${esc(w)}</p>`).join("");
      massHints(ms);
      k("fit-out").hidden = false; k("fit-out").textContent = fitText;
    } catch (e) { k("fit-key").textContent = e.message; k("fit-warn").innerHTML = ""; }
  });
  /* Where the automatic masses came from, under their fields: "from BAND: 0.297 mₑ (Γ→X)"
   * (the CLI's whole sentence is the hint's tooltip and the first part of the report). */
  function massHints(ms) {
    const set = (key, text, title) => {
      k(key).textContent = text || ""; k(key).title = title || "";
      k(key).hidden = !text; k(key + "-l").hidden = !text;
    };
    if (!ms) { set("fm-hint", ""); set("md-hint", ""); return; }
    const src = ms.sources || {};
    set("fm-hint", massHint(src.m_star, ms.m_star), src.m_star);
    set("md-hint", src.m_dos ? `m_d ${massHint(src.m_dos, ms.m_dos)}` : "", src.m_dos);
  }
  $('[data-act="copy-fit"]').addEventListener("click", () => { if (fitText) copy(H, fitText, "Copied"); });
  $('[data-act="save-db"]').addEventListener("click", async () => {
    if (!lastFit) return;
    k("db-state").textContent = "Saving…";
    try {
      const r = await H.api("tau/fit", { body: { ...lastFit, save_db: true } });
      const sd = r.saved_db || {};
      if (sd.error) { k("db-state").textContent = `Not saved: ${sd.error}`; return; }
      k("db-state").textContent = `Saved to ${sd.db} with ${sd.run.split("/").pop()}` +
        `${sd.calc_id ? ` (${sd.calc_id})` : " (untracked: the run has no calculation record)"}: ${sd.keys.join(", ")}`;
      document.dispatchEvent(new CustomEvent("mace-db-changed", { detail: { db: sd.db } }));
    } catch (e) { k("db-state").textContent = `Not saved: ${e.message}`; }
  });

  $('[data-act="write"]').addEventListener("click", async () => {
    const b = { parent: k("parent").value.trim(), axis: k("axis").value, strains: k("strains").value, force: k("force").checked };
    try {
      const r = await H.api("tau/strain", { body: b });
      written = r.written;
      k("w-state").textContent = `Wrote ${r.written.length} decks: ${r.written.map((x) => x.split("/").pop()).join(", ")}`;
      k("w-cmd").textContent = r.command;
      H.M().loadFiles && H.M().loadFiles().catch(() => {});
      k("dir").value = dirOf(b.parent) || ".";
    } catch (e) { k("w-state").textContent = e.message; }
  });
  $('[data-act="submit"]').addEventListener("click", () => {
    if (!written.length) { H.toast("Write the strain decks first", "info"); return; }
    S.builder("submit_dialog", "open", { targets: written });
  });

  if (/\.d12$/i.test(p)) { k("parent").value = p; k("dir").value = dirOf(p) || "."; }
  else if (p) { k("dir").value = /\.\w+$/.test(p) ? (dirOf(p) || ".") : p; }
  if (/_dp_[xyz]_/.test(p)) { show("fit"); scan(); }
}
