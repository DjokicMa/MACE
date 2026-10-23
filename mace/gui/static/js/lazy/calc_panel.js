/* Calculation › Calculators… (mace calc in the GUI): a single point, an
 * optimization, frequencies, molecular dynamics or partial charges of the
 * active structure with UFF/UFF4MOF, EQeq, xTB (GFN2, GFN1, GFN-FF), DFTB+
 * or an ML potential (MACE-MP from ACEsuit's mace-torch, which is unrelated
 * to this MACE tool; ORB-v3; MatterSim; CHGNet).
 *
 * GET /api/calc/backends lists what is installed (a missing backend shows the
 * commands that set it up, and ML Potentials… opens Settings › ML potentials,
 * which installs the ML environment and downloads weights when asked);
 * POST /api/calc/run starts a job in the process pool, with progress and
 * Cancel. An optimized or MD structure opens as a new document; with "Save
 * files" the report and the CLI's files go next to the document's file.
 * Niche fields carry data-tier="advanced" (the inspector's Basic/Advanced
 * convention). */
import { panel } from "./lazy_util.js";
import { openResult } from "./builder_util.js";
import * as profile from "../profile.js";

const TASK_LABEL = { sp: "Single point", opt: "Optimize", freq: "Frequencies", md: "Molecular dynamics", charges: "Charges" };

export function open(H) {
  const S = H.shell();
  const esc = (t) => H.esc(String(t == null ? "" : t));
  const doc = S.activeDoc();
  if (!doc) { H.toast("Open a structure first", "info"); return null; }
  let data = null, running = null, seq = 0, last = null, timer = 0, closed = false;

  const body = document.createElement("div");
  body.className = "calc-panel";
  body.innerHTML = `
    <p class="bld-note" data-k="doc"></p>
    <p class="bld-note" data-k="checking">Checking what is installed…</p>
    <label class="bld-row"><span>Backend</span><select data-k="backend" aria-label="Backend"></select></label>
    <label class="bld-row"><span>Method</span><select data-k="method" aria-label="Method"></select></label>
    <p class="bld-note insp-hint" data-k="note" hidden></p>
    <label class="bld-row"><span>Task</span><select data-k="task" aria-label="Task"></select></label>
    <div data-k="charge-row">
      <label class="bld-row"><span>Charge</span><input type="number" step="1" value="0" data-k="charge" aria-label="Charge"></label>
      <label class="bld-row" data-k="uhf-row"><span>Unpaired electrons</span><input type="number" min="0" step="1" value="0" data-k="uhf" aria-label="Unpaired electrons"></label>
      <label class="bld-row" data-k="mult-row" hidden><span>Spin multiplicity</span><input type="number" min="1" step="1" data-k="multiplicity" aria-label="Spin multiplicity" placeholder="from the file, else the lowest"></label>
      <p class="bld-note insp-hint" data-k="cs-note" hidden>This model takes the total charge and spin multiplicity: blank fields read them from the deck or output, else charge 0 and the lowest multiplicity (said in the result)</p>
    </div>
    <label class="bld-row" data-k="head-row" hidden><span>Task head</span><select data-k="task_head" aria-label="Task head"></select></label>
    <label class="bld-row" data-k="device-row"><span>Device</span><select data-k="device" aria-label="Device"></select></label>
    <div data-k="opt" hidden>
      <label class="bld-row"><span>fmax (eV/Å)</span><input type="number" min="0.0001" step="0.005" value="0.02" data-k="fmax" aria-label="fmax (eV/Å)"></label>
      <label class="bld-row"><span>Steps</span><input type="number" min="1" step="1" value="500" data-k="steps" aria-label="Steps"></label>
      <label class="bld-check"><input type="checkbox" data-k="relax_cell"> Relax the cell</label>
      <label class="bld-check" data-tier="advanced"><input type="checkbox" data-k="keep_symmetry"> Keep the space group</label>
      <label class="bld-row" data-tier="advanced"><span>Optimizer</span><select data-k="optimizer" aria-label="Optimizer"><option value="fire">FIRE</option><option value="bfgs">BFGS</option><option value="lbfgs">L-BFGS</option></select></label>
    </div>
    <div data-k="freq" hidden>
      <label class="bld-row" data-tier="advanced"><span>Displacement (Å)</span><input type="number" min="0.001" step="0.005" value="0.01" data-k="displacement" aria-label="Displacement (Å)"></label>
      <label class="bld-row"><span>Temperature (K)</span><input type="number" min="0" step="10" value="298.15" data-k="ftemp" aria-label="Thermochemistry temperature (K)"></label>
    </div>
    <div data-k="md" hidden>
      <label class="bld-row"><span>Ensemble</span><select data-k="ensemble" aria-label="Ensemble"><option value="nvt">NVT (Langevin)</option><option value="nve">NVE (Velocity Verlet)</option><option value="npt">NPT (Berendsen)</option></select></label>
      <label class="bld-row"><span>Temperature (K)</span><input type="number" min="0" step="10" value="300" data-k="temperature" aria-label="Temperature (K)"></label>
      <label class="bld-row"><span>Time step (fs)</span><input type="number" min="0.1" step="0.1" value="1" data-k="timestep" aria-label="Time step (fs)"></label>
      <label class="bld-row"><span>Steps</span><input type="number" min="1" step="100" value="1000" data-k="md_steps" aria-label="MD steps"></label>
      <label class="bld-row" data-k="pressure-row" hidden><span>Pressure (bar)</span><input type="number" step="1" value="1.01325" data-k="pressure" aria-label="Pressure (bar)"></label>
      <label class="bld-row" data-tier="advanced"><span>Friction (1/fs)</span><input type="number" min="0.0001" step="0.001" value="0.01" data-k="friction" aria-label="Friction (1/fs)"></label>
      <label class="bld-row" data-tier="advanced"><span>Save every (steps)</span><input type="number" min="1" step="1" value="10" data-k="interval" aria-label="Save every (steps)"></label>
      <label class="bld-row" data-tier="advanced"><span>Random seed</span><input type="number" step="1" data-k="seed" aria-label="Random seed" placeholder="random"></label>
    </div>
    <details data-k="adv" data-tier="advanced"><summary>Backend options</summary>
      <label class="bld-row" data-b="xtb"><span>Engine</span><select data-k="engine" aria-label="xTB engine"><option value="xtb">xtb program</option><option value="tblite">tblite (in-process)</option></select></label>
      <label class="bld-row" data-b="dftb"><span>Slater-Koster folder</span><input type="text" data-k="skf_dir" aria-label="Slater-Koster folder" placeholder="MACE_DFTB_SKF"></label>
      <label class="bld-row" data-b="dftb"><span>k-points</span><input type="text" data-k="kpts" aria-label="k-points" placeholder="auto, e.g. 4,4,4"></label>
      <label class="bld-row" data-b="uff"><span>Point charges</span><select data-k="charges" aria-label="UFF point charges"><option value="none">None (UFF)</option><option value="eqeq">EQeq, fixed</option></select></label>
      <label class="bld-row" data-b="uff"><span>Cutoff (Å)</span><input type="number" min="6" step="0.5" value="12.5" data-k="cutoff" aria-label="Cutoff (Å)"></label>
      <label class="bld-row" data-b="eqeq"><span>Charge centers</span><input type="text" data-k="charge_centers" aria-label="Charge centers" placeholder="e.g. Zn=2,Cu=1"></label>
      <label class="bld-row" data-b="mlip"><span>GPU memory share</span><input type="number" min="0.05" max="1" step="0.05" value="0.5" data-k="gpu_mem_fraction" aria-label="GPU memory share"></label>
      <label class="bld-row"><span>Threads</span><input type="number" min="1" step="1" data-k="threads" aria-label="Threads" placeholder="auto"></label>
      <label class="bld-row"><span>Time limit (s)</span><input type="number" min="1" step="60" value="3600" data-k="time_limit" aria-label="Time limit (s)"></label>
    </details>
    <label class="bld-check"><input type="checkbox" data-k="save"> Save files next to the document</label>
    <label class="bld-check" data-tier="advanced"><input type="checkbox" data-k="overwrite"> Overwrite existing files</label>
    <details data-k="more"><summary data-k="more-summary">Not installed: how to set it up</summary>
      <ul class="clean-unavailable" data-k="unavailable"></ul>
      <button type="button" class="tb-btn-text" data-k="recheck">Recheck</button>
      <button type="button" class="tb-btn-text" data-k="mlip-setup" title="Settings › ML potentials: install the ML environment, download weights">ML Potentials…</button></details>
    <div data-k="status" aria-live="polite"></div>
    <div data-k="result"></div>`;
  const foot = `<button type="button" class="tb-btn-text" data-k="cancel" disabled>Cancel</button><button type="button" class="tb-btn-text primary" data-k="run">Run</button>`;
  const d = panel(H, { id: "calc-panel", title: "Calculators", body, foot, wide: true, onClose: () => { closed = true; clearTimeout(timer); cancel(); } });
  const q = (k) => d.querySelector(`[data-k="${k}"]`);
  q("doc").textContent = `Structure: ${doc.title || "untitled"}`;
  q("save").checked = !!doc.path && profile.memory("calcSave", false);
  q("save").disabled = !doc.path;
  if (!doc.path) q("save").parentElement.title = "This document has no file: open the result as a new document instead";

  const be = () => (data && data.backends || []).find((b) => b.name === q("backend").value) || null;
  const meth = () => { const b = be(); return b ? (b.methods || []).find((m) => m.id === q("method").value) || null : null; };

  async function load(recheck = false) {
    try { data = recheck ? await H.api("calc/backends/recheck", { body: {} }) : await H.api("calc/backends"); } catch (e) { q("status").innerHTML = `<p class="warn">${esc(e.message)}</p>`; return; }
    fill();
    if (data.checking && !closed) { clearTimeout(timer); timer = setTimeout(() => load(), 1200); }
  }

  function fill() {
    q("checking").hidden = !data.checking;
    const sel = q("backend"), cur = sel.value || profile.memory("calcBackend", "uff");
    sel.innerHTML = (data.backends || []).map((b) => `<option value="${esc(b.name)}"${b.name === cur ? " selected" : ""}>${esc(b.label)}${b.available ? "" : " (not installed)"}</option>`).join("");
    if (!sel.value && sel.options.length) sel.selectedIndex = 0;
    const dev = q("device"), dcur = dev.value || "auto";
    const gpus = data.gpus || [];
    dev.innerHTML = `<option value="auto">Auto: ${esc(gpus.length ? "GPU 0 (" + gpus[0].name + "), else CPU" : "CPU (no GPU found)")}</option><option value="cpu">CPU</option>` +
      gpus.map((g) => `<option value="cuda:${g.index}">GPU ${g.index}: ${esc(g.name)}</option>`).join("");
    dev.value = [...dev.options].some((o) => o.value === dcur) ? dcur : "auto";
    dev.title = data.device_default || "";
    backendChanged(true);
    const off = [];
    for (const b of data.backends || []) for (const m of b.methods || []) if (!m.available && m.status !== "checking" && (m.hint || m.reason)) off.push({ b, m });
    const seen = new Set();
    q("unavailable").innerHTML = off.filter(({ m }) => { const k = m.hint || m.reason; if (seen.has(k)) return false; seen.add(k); return true; }).map(({ b, m }) =>
      `<li class="clean-hint"><b>${esc(b.name)} ${esc(m.id)}</b>: ${esc(m.reason || m.status)}${m.hint ? `<br><span class="insp-hint">${esc(m.hint)}</span>` : ""}${(m.commands || []).length
        ? `<span class="clean-cmds">${m.commands.map((c) => `<code class="clean-cmd" title="Click to copy">${esc(c)}</code>`).join("")}</span>` : ""}</li>`).join("");
    q("more").hidden = !off.length;
  }

  function backendChanged(keepMethod) {
    const b = be();
    if (!b) return;
    const ms = q("method"), mcur = keepMethod ? (ms.value || profile.memory(`calcMethod.${b.name}`, b.default_method)) : profile.memory(`calcMethod.${b.name}`, b.default_method);
    ms.innerHTML = (b.methods || []).map((m) => `<option value="${esc(m.id)}"${m.id === mcur ? " selected" : ""}${m.available ? "" : " disabled"}>${esc(m.label)}${m.available ? "" : " (not available)"}</option>`).join("");
    if (!ms.value || (meth() && !meth().available)) {
      const first = (b.methods || []).find((m) => m.available);
      if (first) ms.value = first.id;
    }
    const ts = q("task"), tcur = ts.value || profile.memory("calcTask", "opt");
    ts.innerHTML = (b.tasks || []).map((t) => `<option value="${t}"${t === tcur ? " selected" : ""}>${TASK_LABEL[t] || t}</option>`).join("");
    if (!ts.value && ts.options.length) ts.selectedIndex = 0;
    q("device-row").hidden = b.name !== "mlip";
    q("charge-row").hidden = !(b.needs_charge_input);
    q("uhf-row").hidden = b.name !== "xtb";
    d.querySelectorAll("[data-b]").forEach((el) => { el.hidden = el.dataset.b !== b.name; });
    methodChanged();
    taskChanged();
  }

  function methodChanged() {
    const m = meth(), n = q("note");
    const bits = [];
    if (m && m.family_label) bits.push(m.family_label);
    if (m && m.note) bits.push(m.note);
    if (m && m.limits) bits.push(m.limits);
    if (m && !m.available && m.reason) bits.push(`Not available: ${m.reason}`);
    if (m && m.role) bits.push(m.role);
    n.textContent = bits.join(" · ");
    n.hidden = !bits.length;
    // ML models that take a charge and spin (OrbMol-v2, UMA's omol head) and UMA's task heads
    const b = be(), cs = !!(b && b.name === "mlip" && m && (m.charge_spin || []).length);
    if (b && b.name === "mlip") {
      q("charge-row").hidden = !cs;
      q("mult-row").hidden = !cs; q("cs-note").hidden = !cs;
      if (cs && q("charge").dataset.blanked !== "1") { q("charge").value = ""; q("charge").placeholder = "from the file, else 0"; q("charge").dataset.blanked = "1"; }
      const heads = (m && m.tasks) || null;
      q("head-row").hidden = !heads;
      if (heads) q("task_head").innerHTML = `<option value="">By dimensionality (${Object.entries(m.task_default || {}).map(([d, t]) => `${d}D ${t}`).join(", ")})</option>`
        + Object.entries(heads).map(([k, v]) => `<option value="${esc(k)}">${esc(k)}: ${esc(v)}</option>`).join("");
    } else {
      q("mult-row").hidden = true; q("cs-note").hidden = true; q("head-row").hidden = true;
      if (q("charge").dataset.blanked === "1") { q("charge").value = "0"; q("charge").placeholder = ""; delete q("charge").dataset.blanked; }
    }
  }

  function taskChanged() {
    const t = q("task").value;
    for (const k of ["opt", "freq", "md"]) q(k).hidden = t !== k;
    q("pressure-row").hidden = q("ensemble").value !== "npt";
  }

  const num = (k) => { const v = q(k).value.trim(); return v === "" ? undefined : Number(v); };
  function request() {
    const b = be(), m = meth(), task = q("task").value;
    if (!b || !m) throw new Error("Pick a backend and a method");
    const body = { backend: b.name, method: m.id, task, device: q("device").value, save: q("save").checked && !!doc.path,
      overwrite: q("overwrite").checked, time_limit: num("time_limit"), threads: num("threads") };
    if (b.needs_charge_input) { body.charge = num("charge") || 0; if (b.name === "xtb") body.uhf = num("uhf") || 0; }
    if (b.name === "mlip" && (m.charge_spin || []).length) {      // blank: the server reads the file, else the lowest
      if (num("charge") !== undefined) body.charge = num("charge");
      if (num("multiplicity") !== undefined) body.multiplicity = num("multiplicity");
    }
    if (task === "opt") Object.assign(body, { fmax: num("fmax"), steps: num("steps"), relax_cell: q("relax_cell").checked,
      keep_symmetry: q("keep_symmetry").checked, optimizer: q("optimizer").value });
    if (task === "freq") Object.assign(body, { displacement: num("displacement"), temperature: num("ftemp") });
    if (task === "md") Object.assign(body, { ensemble: q("ensemble").value, temperature: num("temperature"), timestep: num("timestep"),
      md_steps: num("md_steps"), friction: num("friction"), interval: num("interval"), pressure: num("pressure"), seed: num("seed") });
    const o = {};
    if (b.name === "xtb") o.engine = q("engine").value;
    if (b.name === "dftb") { if (q("skf_dir").value.trim()) o.skf_dir = q("skf_dir").value.trim(); if (q("kpts").value.trim()) o.kpts = q("kpts").value.trim(); }
    if (b.name === "uff") { o.charges = q("charges").value; o.cutoff = num("cutoff"); }
    if (b.name === "eqeq" && q("charge_centers").value.trim()) o.charge_centers = q("charge_centers").value.trim();
    if (b.name === "mlip") { o.gpu_mem_fraction = num("gpu_mem_fraction"); if (m.tasks && q("task_head").value) o.task_head = q("task_head").value; }
    body.options = o;
    for (const k of Object.keys(body)) if (body[k] === undefined) delete body[k];
    return body;
  }

  function cancel() {
    seq += 1;
    if (running) { const id = running; running = null; H.api(`tasks/${id}/cancel`, { body: {} }).catch(() => {}); }
    busy(false);
  }
  function busy(on) {
    q("run").disabled = on;
    q("cancel").disabled = !on;
  }

  async function run() {
    let body;
    try { body = request(); } catch (e) { q("status").innerHTML = `<p class="warn">${esc(e.message)}</p>`; return; }
    profile.remember("calcBackend", body.backend);
    profile.remember(`calcMethod.${body.backend}`, body.method);
    profile.remember("calcTask", body.task);
    profile.remember("calcSave", !!body.save);
    const mine = ++seq;
    busy(true);
    q("result").innerHTML = "";
    q("status").innerHTML = `<p>Starting…</p>`;
    let r;
    try {
      body.source = await S.sourceOf(doc);
      const resp = await H.api("calc/run", { body, withStatus: true });
      r = resp.data;
      if (resp.status === 202 && r && r.task) {
        if (closed || mine !== seq) { H.api(`tasks/${r.task.id}/cancel`, { body: {} }).catch(() => {}); return; }
        running = r.task.id;
        r = await S.awaitTask(r.task, (t) => {
          if (mine !== seq) return;
          if (t.state === "running" || t.state === "queued") q("status").innerHTML = `<p>${esc(t.state === "queued" ? (t.message || "queued") : "Running")}… <b>${Math.round((t.progress || 0) * 100)}%</b> <span class="insp-hint">${esc(t.message || "")}</span></p>`;
        });
      } else if (r && r.task) S.taskEvent(r.task);
    } catch (e) {
      if (mine !== seq) return;
      busy(false);
      running = null;
      const hint = e.data && e.data.hint ? `<br><span class="insp-hint">${esc(e.data.hint)}</span>` : "";
      q("status").innerHTML = e.state === "canceled" ? "<p>Canceled</p>" : `<p class="warn">${esc(e.message)}${hint}</p>`;
      return;
    }
    if (mine !== seq || closed) return;
    running = null;
    busy(false);
    last = r;
    q("status").innerHTML = "";
    show(r);
    if (r.open && !r.open_error) {
      try { await openResult(H, S, doc, r.open, r.open.name, null); } catch (e) { H.toast(`Open result: ${e.message}`, "warning"); }
    }
  }

  function show(r) {
    const out = [];
    out.push(`<p data-k="summary"><b>${esc(r.summary)}</b></p>`);
    const rows = [];
    if (r.energy_eV != null) rows.push(["Energy", `${r.energy_eV.toFixed(6)} eV` + (r.energy_per_atom_eV != null ? ` (${r.energy_per_atom_eV.toFixed(6)} eV/atom)` : "")]);
    if (r.fmax_eV_A != null) rows.push(["Max force", `${r.fmax_eV_A.toFixed(4)} eV/Å`]);
    if (r.pressure_GPa != null) rows.push(["Pressure", `${r.pressure_GPa.toFixed(3)} GPa`]);
    const rep = r.report || {};
    if (rep.volume_change_pct != null) rows.push(["Volume change", `${rep.volume_change_pct.toFixed(2)} %`]);
    if (rep.device_text) rows.push(["Device", rep.device_text]);
    if (rep.model_label) rows.push(["Model", rep.model_label + (rep.task_head ? `, task head ${rep.task_head}` : "")]);
    if (rep.charge != null) rows.push(["Charge, multiplicity", `${rep.charge >= 0 ? "+" : ""}${rep.charge}, ${rep.spin_multiplicity} (${rep.charge_source}; ${rep.spin_source})`]);
    if (rep.atom_types) rows.push(["UFF types", Object.entries(rep.atom_types).map(([k, v]) => `${k} ${v}`).join(", ")]);
    if (r.thermo && r.thermo.zpe_eV != null) {
      rows.push(["ZPE", `${r.thermo.zpe_eV.toFixed(4)} eV`]);
      if (r.thermo.vib_free_energy_eV != null) rows.push([`F vib (${r.thermo.temperature_K} K)`, `${r.thermo.vib_free_energy_eV.toFixed(4)} eV`]);
    }
    if (r.md) rows.push(["MD", `${r.md.steps_done} steps, ⟨T⟩ ${r.md.mean_temperature_K != null ? r.md.mean_temperature_K.toFixed(1) : "?"} K, drift ${r.md.etot_drift_eV != null ? r.md.etot_drift_eV.toFixed(4) : "?"} eV`]);
    if (rows.length) out.push(`<table class="calc-table"><tbody>${rows.map(([k, v]) => `<tr><th scope="row" style="text-align:left;padding-right:8px">${esc(k)}</th><td>${esc(v)}</td></tr>`).join("")}</tbody></table>`);
    if (r.frequencies_cm1) {
      const f = r.frequencies_cm1;
      out.push(`<details><summary>${f.length} frequencies (cm⁻¹)</summary><p class="num" style="white-space:normal">${f.map((x) => (x < 0 ? `${Math.abs(x).toFixed(1)}i` : x.toFixed(1))).join(", ")}</p></details>`);
    }
    if (r.charges) {
      const by = {};
      r.charges.forEach((q0, i) => { const s = (r.symbols || [])[i] || "?"; (by[s] = by[s] || []).push(q0); });
      out.push(`<p>Charges by element: ${Object.entries(by).map(([s, v]) => `${esc(s)} ${Math.min(...v).toFixed(3)}…${Math.max(...v).toFixed(3)}`).join(" · ")}</p>`);
      out.push(`<details><summary>Every atom</summary><p class="num" style="white-space:normal">${r.charges.map((x, i) => `${esc((r.symbols || [])[i] || "")}${i + 1} ${x >= 0 ? "+" : ""}${x.toFixed(4)}`).join(", ")}</p></details>`);
    }
    if (rep.license_note) out.push(`<p class="warn" data-k="license-note">${esc(rep.license_note)}</p>`);
    for (const w of (r.warnings || []).concat((rep.typing_warnings || []).map((x) => `UFF typing: ${x}`), rep.charge_spin_warnings || [])) out.push(`<p class="warn">${esc(w)}</p>`);
    if (r.save_error) out.push(`<p class="warn">Save: ${esc(r.save_error)}</p>`);
    if (r.open_error) out.push(`<p class="warn">Open result: ${esc(r.open_error)}</p>`);
    const files = r.files || {};
    if (Object.keys(files).length && r.saved_to !== undefined) {
      out.push(`<p>Saved to ${esc(r.saved_to)}: ${Object.values(files).map((p) => `<a href="#" data-open="${esc(p)}">${esc(p.split("/").pop())}</a>`).join(", ")}</p>`);
    }
    if (r.calc_task === "opt") out.push(`<p class="insp-hint">A force-field/ML geometry: run a CRYSTAL OPT from it before SP, FREQ, BAND or DOSS.</p>`);
    const prov = r.provenance || {};
    if ((prov.citation || []).length) out.push(`<p class="insp-hint">Cite: ${esc(prov.citation[0].text)}${prov.citation.length > 1 ? " …" : ""} <button type="button" class="tb-btn-text" data-k="copy-cite">Copy citation</button></p>`);
    q("result").innerHTML = out.join("");
  }

  q("backend").addEventListener("change", () => { profile.remember("calcBackend", q("backend").value); backendChanged(false); });
  q("method").addEventListener("change", methodChanged);
  q("task").addEventListener("change", taskChanged);
  q("ensemble").addEventListener("change", taskChanged);
  q("recheck").addEventListener("click", () => { q("checking").hidden = false; load(true); });
  q("mlip-setup").addEventListener("click", () => H.openSettings("mlip"));
  q("run").addEventListener("click", run);
  q("cancel").addEventListener("click", () => { cancel(); q("status").innerHTML = "<p>Canceled</p>"; });
  d.addEventListener("click", (e) => {
    const c = e.target.closest(".clean-cmd");
    if (c && navigator.clipboard) navigator.clipboard.writeText(c.textContent).then(() => H.toast("Copied", "success")).catch(() => {});
    const a = e.target.closest("[data-open]");
    if (a) { e.preventDefault(); H.openPath(a.dataset.open); }
    if (e.target.closest('[data-k="copy-cite"]') && last) {
      const p = last.provenance || {};
      const text = (p.citation || []).map((c0) => c0.text + (c0.doi ? `, doi:${c0.doi}` : "")).concat((p.data_citation || []).map((c0) => c0.text + (c0.doi ? `, doi:${c0.doi}` : ""))).join("\n");
      d.dataset.copied = text;
      if (navigator.clipboard) navigator.clipboard.writeText(text).then(() => H.toast("Citation copied", "success")).catch(() => {});
    }
  });
  load();
  return d;
}
