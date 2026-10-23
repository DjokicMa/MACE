/* Settings › ML potentials (1.1.6): the ML environment and the weights.
 * The server runs `mace mlip install`'s own code (mace/preopt/mlip_install.py)
 * as a task: GET api/mlip/status, POST api/mlip/plan (the checks, before
 * anything happens), api/mlip/install, api/mlip/record (Use an Existing
 * Python…), api/mlip/fetch (one model's weights, after its license is shown
 * and accepted here). Each task's log streams into the dialog; Cancel stops
 * pip and removes a folder the run created. Clean's and the Calculators'
 * tiers are detected again when a task succeeds. Loaded on first use. */
import { loadCss } from "./lazy_util.js";
import { dialog } from "../ui.js";

let H = null, host = null, state = null;

const esc = (t) => H.esc(String(t == null ? "" : t));
const STATUS = { ready: "ready", needs_setup: "no environment", needs_fetch: "weights not downloaded", needs_ack: "academic use not acknowledged", import_failed: "does not import",
  probe_timeout: "no answer", bad_hash: "weights damaged", unsupported_python: "Python too old", missing: "file missing" };

export async function mount(el, h) {
  H = h; host = el;
  await loadCss(import.meta.url, "../../css/mlip.css");
  host.innerHTML = `<p class="muted">Looking at the ML environment… (a first look can take a few seconds)</p>`;
  await refresh();
}

/** From Clean or the Calculators: Settings › ML potentials. */
export function open(h) { h.openSettings("mlip"); }

async function refresh() {
  try { state = await H.api("mlip/status"); } catch (e) { if (host) host.innerHTML = `<p class="mlip-note">${esc(e.message)}</p>`; return; }
  render();
}

function render() {
  if (!host || !host.isConnected) return;
  const s = state;
  const env = s.python ? `<code>${esc(s.python)}</code> <span class="muted">(${esc(s.source)})</span>`
    : `none yet <span class="muted">(MACE looks in <code>${esc(s.default_dir)}</code>)</span>`;
  const rows = s.models.map((m) => `<tr data-model="${esc(m.id)}"${m.academic ? ' data-academic="1"' : ""}><td>${esc(m.label)}${m.default ? ' <span class="muted">(default)</span>' : ""}
      ${m.academic ? '<div class="mlip-academic small"><b>Academic, non-commercial use only</b> (opt-in; results are tagged “academic license”)</div>' : ""}
      ${m.gated ? '<div class="muted small">Gated: downloaded with your own Hugging Face access</div>' : ""}
      <div class="muted small">code ${esc(m.code_license)}, weights ${esc(m.weights_license)}${m.size_mb ? `; ${Math.round(m.size_mb)} MB` : "; weights in the package"}${m.scopes.length === 1 && m.scopes[0] === "calc" ? "; Calculators only" : ""}${m.env !== "default" ? `; its own environment (${esc(m.env)})` : ""}</div></td>
      <td><span class="mlip-status ${m.status === "ready" ? "ready" : ""}" title="${esc(m.reason)}">${esc(STATUS[m.status] || m.status)}</span></td>
      <td>${m.fetchable && ["needs_fetch", "bad_hash", "needs_ack"].includes(m.status) && s.host_ok
        ? `<button type="button" class="tb-btn-text" data-fetch="${esc(m.id)}">${m.academic ? "Acknowledge and Download…" : "Download Weights…"}</button>` : ""}</td></tr>`).join("");
  const others = Object.entries(s.env_pythons || {}).filter(([e]) => e !== "default")
    .map(([e, p]) => `<p class="muted small" id="mlip-env-${esc(e)}">${esc(e)} environment: ${p ? `<code>${esc(p)}</code>` : "none yet"}</p>`).join("");
  host.innerHTML = `<div class="mlip-pane">
    <p class="muted">ML potentials are third-party models, unrelated to this MACE tool. They run in their own Python environment; MACE installs one only when you ask here (or with <code>mace mlip install</code>).</p>
    ${s.host_ok ? "" : `<p class="mlip-note" id="mlip-hostnote">${esc(s.host_note)}</p>`}
    <p id="mlip-env">Environment: ${env}</p>${others}
    <div class="mlip-actions">
      <button type="button" class="tb-btn-text primary" id="mlip-setup"${s.host_ok ? "" : " disabled"}>${s.python ? "Repair or Extend the ML Environment…" : "Set Up ML Environment…"}</button>
      <button type="button" class="tb-btn-text" id="mlip-use"${s.loopback ? "" : ' disabled title="MACE GUI is on the network: use mace mlip setup --python in a shell"'}>Use an Existing Python…</button>
      <button type="button" class="tb-btn-text" id="mlip-recheck">Recheck</button></div>
    <table id="mlip-models"><thead><tr><th>Model</th><th>Status</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>`;
  host.querySelector("#mlip-setup").addEventListener("click", () => setupDialog());
  host.querySelector("#mlip-use").addEventListener("click", () => useDialog());
  host.querySelector("#mlip-recheck").addEventListener("click", async () => {
    host.querySelector("#mlip-recheck").disabled = true;
    try { await H.api("clean/methods/recheck", { body: {} }); } catch (e) { /* the status below says it */ }
    await refresh();
  });
  host.querySelectorAll("[data-fetch]").forEach((b) => b.addEventListener("click", () => fetchDialog(b.dataset.fetch)));
}

/* ---------------------------------------------------------------- a task with a live log */
function runTask(dlg, path, body, { logSel, resultSel, onDone }) {
  const log = dlg.querySelector(logSel), out = dlg.querySelector(resultSel);
  let taskId = null;
  log.hidden = false; out.hidden = true;
  const p = H.apiTask(path, body, (t) => {
    taskId = t.id; dlg.dataset.task = t.id;
    if (t.log) { log.textContent = t.log; log.scrollTop = log.scrollHeight; }
  });
  p.then((r) => {
    out.hidden = false; out.className = "mlip-result";
    onDone(r, out);
    refresh();
  }).catch((e) => {
    out.hidden = false; out.className = "mlip-result bad";
    const t = e.task || {};
    if (t.log) log.textContent = t.log;
    out.textContent = e.state === "canceled" ? "Canceled. A folder this run created was removed." : (e.message || "It failed");
    refresh();
  }).finally(() => { delete dlg.dataset.task; dlg.dispatchEvent(new Event("mlip-idle")); });
  return { cancel: () => { if (taskId) H.api(`tasks/${taskId}/cancel`, { body: {} }).catch(() => {}); } };
}

/* ---------------------------------------------------------------- Set Up ML Environment… */
function setupDialog() {
  const s = state;
  const pk = s.packages;
  const item = (p) => `<li><label class="chk"><input type="checkbox" data-pkg="${esc(p.package)}" data-env="${esc(p.env)}"${s.default_backends.includes(p.package) ? " checked" : ""}>
      <b>${esc(p.pin)}</b> <span class="muted">code ${esc(p.license)}${p.min_python !== "3.10" ? `; needs Python ${esc(p.min_python)}+` : ""}</span></label>
      <span class="models">${p.models.map((m) => `${esc(m.label)}: weights ${esc(m.weights_license)}${m.academic ? " (academic use only)" : ""}${m.gated ? " (gated)" : ""}`).join("; ")}</span></li>`;
  const envs = [...new Set(pk.map((p) => p.env))];
  const items = envs.map((e) => (e === "default" ? "" : `<li class="muted small">In its own environment (${esc(e)}; its pins conflict with mace-torch's):</li>`)
    + pk.filter((p) => p.env === e).map(item).join("")).join("");
  const def = s.models.find((m) => m.default) || {};
  const body = `<p class="muted">The same steps as <code>mace mlip install</code>: a Python environment, the CPU build of torch, the packages below with their licenses, a check, then MACE records it. Nothing is installed until you press Start.</p>
    <label class="row"><span>Location</span><input type="text" class="field" id="mlip-dir" value="${esc(s.default_dir)}"${s.loopback ? "" : " disabled"} spellcheck="false"></label>
    <fieldset><legend>Packages</legend><ul class="mlip-pkgs">${items}</ul></fieldset>
    <label class="chk"><input type="checkbox" id="mlip-gpu"> torch with CUDA (only when an NVIDIA GPU answers here; several GB)</label>
    <label class="chk"><input type="checkbox" id="mlip-fetch" checked> Then download the ${esc(def.label || "default model")} weights (${Math.round(def.size_mb || 0)} MB)</label>
    <label class="chk" id="mlip-accept-row"><input type="checkbox" id="mlip-accept"> I accept the ${esc(def.weights_license || "")} license of those weights${def.weights_license_url ? ` (<a href="${esc(def.weights_license_url)}" target="_blank" rel="noopener noreferrer">read it</a>)` : ""}</label>
    <details><summary>Advanced</summary><label class="row"><span>Python to build from (blank: found automatically, 3.10 or newer)</span>
      <input type="text" class="field" id="mlip-python" placeholder="e.g. /usr/bin/python3.11"${s.loopback ? "" : " disabled"} spellcheck="false"></label></details>
    <p class="mlip-plan" id="mlip-plan" role="status">Checking…</p><ul class="mlip-warn" id="mlip-warn" hidden></ul>
    <pre class="mlip-log" id="mlip-log" hidden aria-label="Install log"></pre><p id="mlip-result" hidden></p>`;
  const foot = `<button type="button" class="tb-btn-text" id="mlip-cancel">Close</button><button type="button" class="tb-btn-text primary" id="mlip-start" disabled>Start</button>`;
  const dlg = dialog({ id: "mlip-setup-dlg", title: "Set Up ML Environment", body, foot });
  dlg.classList.add("mlip-dlg");
  const q = (sel) => dlg.querySelector(sel);
  let plan = null, timer = 0, running = null, seq = 0;
  const backends = () => [...dlg.querySelectorAll("[data-pkg]:checked")].map((x) => x.dataset.pkg);
  const req = () => ({ dir: s.loopback ? q("#mlip-dir").value.trim() : undefined, backends: backends(), gpu: q("#mlip-gpu").checked,
    python: s.loopback ? (q("#mlip-python").value.trim() || undefined) : undefined });
  const ready = () => { q("#mlip-start").disabled = !!running || !plan || plan.problems.length > 0 || !backends().length || (q("#mlip-fetch").checked && !q("#mlip-accept").checked); };
  const fetchRow = () => { const on = backends().includes("mace-torch"); q("#mlip-fetch").disabled = !on; if (!on) q("#mlip-fetch").checked = false; q("#mlip-accept-row").hidden = !q("#mlip-fetch").checked; ready(); };
  async function check() {
    const mine = ++seq;
    plan = null; ready();
    if (!backends().length) { q("#mlip-plan").textContent = "Choose at least one package."; return; }
    q("#mlip-plan").textContent = "Checking…";
    try {
      const p = await H.api("mlip/plan", { body: req() });
      if (mine !== seq) return;
      plan = p;
      const free = p.free_bytes == null ? "" : p.free_bytes >= 1e13 ? `; ${(p.free_bytes / 1e12).toFixed(0)} TB free` : `; ${(p.free_bytes / 1e9).toFixed(0)} GB free`;
      const qu = p.quota && p.quota.available ? `; quota ${p.quota.used_gb.toFixed(0)} of ${p.quota.quota_gb.toFixed(0)} GB, ${p.quota.files.toLocaleString("en-US")} of ${p.quota.files_quota.toLocaleString("en-US")} files` : "";
      q("#mlip-plan").className = "mlip-plan" + (p.problems.length ? " bad" : "");
      q("#mlip-plan").innerHTML = p.problems.length ? esc(p.problems.join("; "))
        : `${p.state === "existing" ? "Repairs the environment in" : "Into"} <code>${esc(p.dir)}</code>, from ${esc(p.python)} (Python ${esc(p.python_version)}); ${p.torch === "gpu" ? "CUDA" : "CPU"} torch; about ${(p.estimate_mb / 1000).toFixed(1)} GB and ${p.estimate_files.toLocaleString("en-US")} files${free}${qu}.`;
      const w = q("#mlip-warn");
      w.hidden = !p.warnings.length;
      w.innerHTML = p.warnings.map((x) => `<li>${esc(x)}</li>`).join("");
    } catch (e) {
      if (mine !== seq) return;
      q("#mlip-plan").className = "mlip-plan bad"; q("#mlip-plan").textContent = e.message;
    }
    ready();
  }
  const later = () => { clearTimeout(timer); timer = setTimeout(check, 400); };
  dlg.addEventListener("change", (e) => {
    if (e.target.matches("[data-pkg]")) {
      // one environment per install: a package of another environment unticks the others and moves the folder
      const env = e.target.dataset.env;
      if (e.target.checked) {
        dlg.querySelectorAll("[data-pkg]").forEach((x) => { if (x.dataset.env !== env) x.checked = false; });
        if (s.loopback && s.default_dirs && s.default_dirs[env]) {
          const dirs = Object.values(s.default_dirs);
          if (!q("#mlip-dir").value.trim() || dirs.includes(q("#mlip-dir").value.trim())) q("#mlip-dir").value = s.default_dirs[env];
        }
      }
      fetchRow(); check();
    }
    else if (e.target.id === "mlip-gpu") check();
    else if (e.target.id === "mlip-fetch") fetchRow();
    else if (e.target.id === "mlip-accept") ready();
  });
  dlg.addEventListener("input", (e) => { if (e.target.id === "mlip-dir" || e.target.id === "mlip-python") later(); });
  dlg.addEventListener("mlip-idle", () => { running = null; q("#mlip-cancel").textContent = "Close"; dlg.querySelectorAll("input").forEach((x) => { x.disabled = !s.loopback && (x.id === "mlip-dir" || x.id === "mlip-python"); }); fetchRow(); });
  q("#mlip-cancel").addEventListener("click", () => { if (running) running.cancel(); else dlg.close(); });
  q("#mlip-start").addEventListener("click", () => {
    const body = { ...req(), fetch: q("#mlip-fetch").checked ? def.id : null, accept_license: q("#mlip-fetch").checked && q("#mlip-accept").checked };
    dlg.querySelectorAll("input").forEach((x) => { x.disabled = true; });
    q("#mlip-cancel").textContent = "Cancel";
    running = runTask(dlg, "mlip/install", body, { logSel: "#mlip-log", resultSel: "#mlip-result", onDone: (r, out) => {
      out.textContent = `The ML environment is ready: ${r.python}.` + (r.fetch && r.fetch.rc !== 0 ? " The weights were not downloaded: Download Weights… in Settings › ML potentials tries again." : "");
      H.toast("The ML environment is ready", "success");
    } });
    ready();
  });
  fetchRow();
  check();
  return dlg;
}

/* ---------------------------------------------------------------- Use an Existing Python… */
function useDialog() {
  const body = `<p class="muted">An environment you made yourself (with torch, ase and mace-torch or CHGNet): MACE probes it as before every use, then records it (<code>mace mlip setup --python</code>).</p>
    <label class="row"><span>Its interpreter</span><input type="text" class="field" id="mlip-use-python" placeholder="DIR/bin/python" spellcheck="false"></label>
    <pre class="mlip-log" id="mlip-use-log" hidden></pre><p id="mlip-use-result" hidden></p>`;
  const dlg = dialog({ id: "mlip-use-dlg", title: "Use an Existing Python", body, narrow: true,
    foot: `<button type="button" class="tb-btn-text" data-x="close">Close</button><button type="button" class="tb-btn-text primary" id="mlip-use-go">Probe and Record</button>` });
  dlg.classList.add("mlip-dlg");
  dlg.querySelector('[data-x="close"]').addEventListener("click", () => dlg.close());
  const go = dlg.querySelector("#mlip-use-go");
  dlg.addEventListener("mlip-idle", () => { go.disabled = false; });
  go.addEventListener("click", () => {
    const py = dlg.querySelector("#mlip-use-python").value.trim();
    if (!py) return;
    go.disabled = true;
    runTask(dlg, "mlip/record", { python: py }, { logSel: "#mlip-use-log", resultSel: "#mlip-use-result",
      onDone: (r, out) => { out.textContent = `Recorded: ${r.python}` + (r.override ? ` (${r.override} still takes precedence)` : ""); } });
  });
  return dlg;
}

/* ---------------------------------------------------------------- Download Weights… */
function fetchDialog(id) {
  const m = state.models.find((x) => x.id === id);
  if (!m) return null;
  const cite = (m.citation || []).map((c) => `<li>${esc(c)}</li>`).join("");
  const data = (m.training_data || []).map((c) => `<li>${esc(c)}</li>`).join("");
  const body = `<p>${esc(m.label)}${m.unrelated_note ? ` <span class="muted">(${esc(m.unrelated_note)})</span>` : ""}</p>
    <table class="kv"><tr><td>Weights license</td><td><b>${esc(m.weights_license)}</b>${m.weights_license_url ? ` (<a href="${esc(m.weights_license_url)}" target="_blank" rel="noopener noreferrer">the license text</a>)` : ""}</td></tr>
      <tr><td>Code license</td><td>${esc(m.code_license)} (${esc(m.package)})</td></tr><tr><td>Size</td><td>${Math.round(m.size_mb)} MB</td></tr>
      ${m.role ? `<tr><td>For</td><td>${esc(m.role)}</td></tr>` : ""}</table>
    <div class="mlip-license">${data ? `<p class="muted">Training data</p><ul>${data}</ul>` : ""}${cite ? `<p class="muted">Cite</p><ul>${cite}</ul>` : ""}</div>
    ${m.academic ? `<p class="mlip-note" id="mlip-fetch-academic-note"><b>Academic, non-commercial use only.</b> ${esc(m.label)}'s weights are under the ${esc(m.weights_license)}: they may not be used commercially. MACE never offers it by default or with Auto; every result made with it is tagged “academic license”. Your acknowledgment is recorded with today's date beside the weights.</p>
      <label class="chk"><input type="checkbox" id="mlip-fetch-accept" data-academic="1"> I will use this model for academic, non-commercial work only, under its license</label>`
    : m.gated ? `<p class="mlip-note" id="mlip-fetch-gated-note">${esc(m.gated.terms || m.weights_license)}${m.gated.aup_url ? ` (<a href="${esc(m.gated.aup_url)}" target="_blank" rel="noopener noreferrer">Acceptable Use Policy</a>)` : ""}. The weights are gated: accept the terms at <a href="${esc(m.gated.page)}" target="_blank" rel="noopener noreferrer">${esc(m.gated.page)}</a> with your own Hugging Face account first, and log in where MACE runs (<code>hf auth login</code>, or HF_TOKEN). MACE never accepts them for you and never stores your token.</p>
      <label class="chk"><input type="checkbox" id="mlip-fetch-accept"> I accept the ${esc(m.weights_license)} and its Acceptable Use Policy</label>`
    : `<label class="chk"><input type="checkbox" id="mlip-fetch-accept"> I accept the ${esc(m.weights_license)} license of these weights</label>`}
    <pre class="mlip-log" id="mlip-fetch-log" hidden></pre><p id="mlip-fetch-result" hidden></p>`;
  const dlg = dialog({ id: "mlip-fetch-dlg", title: "Download Weights", body,
    foot: `<button type="button" class="tb-btn-text" data-x="close">Close</button><button type="button" class="tb-btn-text primary" id="mlip-fetch-go" disabled>Download</button>` });
  dlg.classList.add("mlip-dlg");
  const go = dlg.querySelector("#mlip-fetch-go"), acc = dlg.querySelector("#mlip-fetch-accept");
  dlg.querySelector('[data-x="close"]').addEventListener("click", () => dlg.close());
  acc.addEventListener("change", () => { go.disabled = !acc.checked; });
  dlg.addEventListener("mlip-idle", () => { go.disabled = !acc.checked; });
  go.addEventListener("click", () => {
    go.disabled = true;
    runTask(dlg, "mlip/fetch", m.academic ? { id, accept_academic: acc.checked } : { id, accept_license: acc.checked }, { logSel: "#mlip-fetch-log", resultSel: "#mlip-fetch-result",
      onDone: (r, out) => { out.textContent = `Downloaded and verified (${r.status}).`; } });
  });
  return dlg;
}
