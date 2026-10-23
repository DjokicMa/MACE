/* Calculation Setup's decks beyond opt2d12's questions (1.1.6): the
 * mechanical-property modules, numbered as opt2d12's "Select calculation
 * type" menu numbers them after FREQ (4 ELASTIC, 5 EOS, 6 QHA, 7 CPHF, 8
 * PHONOPY: phonopy's displaced supercells). Each writes its deck(s) with the
 * CLI's own command, run as a task in the parent's folder, which keeps the
 * parent's method, basis and SCF settings:
 *   4-7      mace opt2d12 --out-file PARENT.out --calc-type ELASTIC|EOS|QHA|CPHF
 *            --non-interactive --elastic-settings JSON (mace/gui/mechanics_routes.py)
 *   8        mace opt2d12 --out-file PARENT.out --calc-type PHONOPY --non-interactive
 *            --dim N N N ... (phonopy_routes.py)
 * What the runs give is read under Analysis: Mechanical Properties Results
 * (mace analyze elastic) and Phonons with Phonopy (forces, then mace analyze
 * phonopy). One form per document and entry, kept while the page is open. */
import { val } from "../lazy/mx_util.js";

/* in opt2d12's menu order (Crystal_d12/d12_interactive.py, get_calculation_type).
 * Their numbers, groups and what the file needs: setup/schema calc_groups (mace/workflow/common/constants.py) */
export const MODULES = {
  ELASTIC: { label: "ELASTIC", cmd: "mace opt2d12 --calc-type ELASTIC", title: "Elastic, piezoelectric or photoelastic constants (ELASTIC): mace opt2d12 --calc-type ELASTIC" },
  EOS: { label: "EOS", cmd: "mace opt2d12 --calc-type EOS", title: "Equation of state (EOS): mace opt2d12 --calc-type EOS" },
  QHA: { label: "QHA", cmd: "mace opt2d12 --calc-type QHA", title: "Quasi-harmonic approximation (QHA): mace opt2d12 --calc-type QHA" },
  CPHF: { label: "CPHF", cmd: "mace opt2d12 --calc-type CPHF", title: "Dielectric response, CPHF/KS (CPHF): mace opt2d12 --calc-type CPHF" },
  PHONOPY: { label: "Phonopy", cmd: "mace opt2d12 --calc-type PHONOPY", title: "Finite-displacement phonons with phonopy (PHONOPY): the displaced supercells of this output (mace opt2d12 --calc-type PHONOPY)" },
};
export const isModule = (calc) => Object.prototype.hasOwnProperty.call(MODULES, calc);

/* the module's sub-keywords: key, label, kind, placeholder/help (CRYSTAL23 manual ch. 9, 10, 12-13) */
const FIELDS = [
  ["numderiv", "Points per strain (NUMDERIV)", "int", "3 (incl. the central one)"],
  ["stepsize", "Strain step (STEPSIZE)", "float", ""],
  ["deform", "Deformations (DEFORM)", "deform", "default: CRYSTAL's symmetry analysis"],
  ["relax", "Nuclear relaxation", "relax", ""],
  ["hessstep", "Hessian step, Å (HESSSTEP)", "float", "with NUCHESS"],
  ["pressure", "Pressure, GPa (PRESSURE)", "float", "pre-optimizes at this pressure"],
  ["presseos", "Structure is at P, GPa (PRESSEOS)", "float", "from an EOS run"],
  ["tolalpha", "CPHF/KS tolerance (TOLALPHA)", "int", "4 (PHOTOELA)"],
  ["fmixing", "CPHF/KS mixing % (FMIXING)", "int", "60"],
  ["dynamic", "Wavelength, nm (DYNAMIC)", "int", "static (ω → ∞) when empty"],
  ["range", "Volume range (RANGE)", "triple", "0.92 1.08 6"],
  ["prange", "Printed pressures (PRANGE)", "triple", "Pmin Pmax N (GPa)"],
  ["vrange", "Printed volumes (VRANGE)", "triple", "Vmin Vmax N (Å³)"],
  ["seismdir", "Extra wave directions (SEISMDIR)", "dirs", "1 1 0; 1 1 1"],
  ["toldeg", "Opt. TOLDEG", "float", "0.0003"], ["toldex", "Opt. TOLDEX", "float", "0.00062"],
  ["toldee", "Opt. TOLDEE", "int", "8"], ["maxcycle", "Opt. MAXCYCLE", "int", ""],
  ["qstep", "Volume step, % (STEP)", "float", "3"],
  ["points", "Volumes (POINTS)", "int", "4, 7 or 13"],
  ["temperat", "Temperatures (TEMPERAT)", "triple", "N Tmin Tmax (N ≥ 50): 50 10 300"],
  ["scelphono", "Phonon supercell (SCELPHONO)", "triple", "2 2 2 (empty: Γ only)"],
  ["neglefre", "Leave out frequencies (NEGLEFRE)", "triple", "cm⁻¹, space separated"],
  ["cphf_dynamic", "Wavelengths (DYNAMIC)", "triple", "1 589, or NSTEPS FIRST LAST (nm); pure LDA/GGA only"],
  ["damping", "Damping, Ha (DAMPING)", "float", "0.001-0.003"],
  ["histdiis", "DIIS history (HISTDIIS)", "int", "all cycles"],
  ["seledir", "Iterations per direction (SELEDIR)", "triple", "NX NY NZ"],
  ["toludik", "Degeneracy threshold (TOLUDIK)", "int", "6"],
  ["broyden", "BROYDEN accelerator", "triple", "W0 IMIX ISTART"],
  ["restart", "Restart", "restart", ""],
];
const FLAGS = [["preoptgeom", "Pre-optimize first (PREOPTGEOM)"], ["hessnum2", "Two-point Hessians (HESSNUM2)"],
  ["modepart", "Split the nuclear term by modes (MODEPART)"], ["awesome", "AWESoMe wave-velocity files (AWESOME)"],
  ["anderson", "ANDERSON accelerator"], ["third", "First hyperpolarizability β (THIRD)"],
  ["fourth", "Second hyperpolarizability γ (FOURTH)"], ["nodiis", "No DIIS (NODIIS)"], ["diisallk", "DIIS at every k point (DIISALLK)"],
  ["print", "Extended printing (PRINT, recommended)"]];
const LISTS = ["range", "prange", "vrange", "broyden", "temperat", "scelphono", "neglefre", "cphf_dynamic", "seledir"];
const VOIGT = ["xx", "yy", "zz", "yz", "xz", "xy"];
// the same fields in the workflow builder's ELASTIC, EOS, QHA and CPHF steps (lazy/workflow_dialog.js)
export { FIELDS as MODULE_FIELDS, FLAGS as MODULE_FLAGS, LISTS as MODULE_LISTS, VOIGT as MODULE_VOIGT };

let meta = null;                       // elastic/modes, asked once
const forms = new Map();               // `${doc id}:${calc}` -> {settings, write}

/** The two parts of an entry's form (kept while the page is open, so a redraw of the panel keeps
 * what was typed): `settings` goes in step 2, `write` is step 3. S: the Setup shell API, U: its UI helpers. */
export function formFor(S, U, d, calc) {
  const key = `${d.id}:${calc}`;
  let f = forms.get(key);
  if (!f || f.path !== d.path) {
    f = calc === "PHONOPY" ? phonopyForm(S, U, d) : elasticForm(S, U, d, calc);
    f.path = d.path;
    forms.set(key, f);
  }
  return f;
}
export function forget(docId) { for (const k of [...forms.keys()]) if (k.startsWith(`${docId}:`)) forms.delete(k); }

const el = (html) => { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstElementChild; };
const row = (esc, id, label, ctl, hint = "") => `<div class="su-row su-mod-row"><label class="su-l" for="${id}">${esc(label)}</label>
  <div class="su-ctl">${ctl}</div><div class="su-meta"><span class="su-h">${esc(hint)}</span></div></div>`;

/** Step 3 of an entry: its button, the state line, the command, the log and what comes next. */
function writePart(U, verb, button) {
  const w = el(`<section class="su-step su-mod-write" aria-label="${U.esc(verb)}"><h3 id="su-s3"><span class="num">3</span>${U.esc(verb)}</h3>
    <div class="su-saveacts"><button type="button" class="tb-btn-text primary" data-mact="write">${U.icon("save")}<span>${U.esc(button)}</span></button></div>
    <p class="su-note su-mod-state" data-mk="state" role="status" aria-live="polite"></p>
    <div class="su-cmd" data-mk="cmdbox" hidden><code class="mono" data-mk="cmd"></code></div>
    <details class="su-mod-log" data-mk="logbox" hidden><summary>Log</summary><pre class="su-mod-pre" data-mk="log"></pre></details>
    <div class="su-saved-acts" data-mk="next" hidden></div></section>`);
  return w;
}
function showRun(w, U, r, ok, text) {
  const k = (n) => w.querySelector(`[data-mk="${n}"]`);
  k("state").innerHTML = `${U.icon(ok ? "circle-check" : "circle-alert")}<span>${U.esc(text)}</span>`;
  k("state").classList.toggle("bad", !ok);
  k("cmd").textContent = (r && r.command) || "";
  k("cmdbox").hidden = !(r && r.command);
  k("log").textContent = (r && r.log) || "";
  k("logbox").hidden = !(r && r.log);
  if (r && r.log && !ok) k("logbox").open = true;
}

/* ---------------------------------------------------------------- 4-7: mace opt2d12 --calc-type ELASTIC|EOS|QHA|CPHF */
function elasticForm(S, U, d, calc) {
  const esc = U.esc;
  const id = (k) => `su-m-${calc.toLowerCase()}-${k}`;
  const settings = el(`<div class="su-mod" data-mod="${calc}"><div class="su-loading"><span class="spinner"></span><span>Reading the modules…</span></div></div>`);
  const write = writePart(U, "Write the deck", "Write Deck");
  const f = { settings, write, command: "" };
  const k = (n) => write.querySelector(`[data-mk="${n}"]`);
  (async () => {
    try { if (!meta) meta = await S.api("elastic/modes"); } catch (e) { settings.innerHTML = `<p class="su-warnline">${U.icon("circle-alert")}<span>${esc(e.message)}</span></p>`; return; }
    const own = calc !== "ELASTIC";
    const subs = meta.modes.filter((m) => (own ? m.mode === calc : !["EOS", "QHA", "CPHF"].includes(m.mode)));
    // the ELASTIC modules as the CLI numbers them in its "Mechanical property module" menu
    settings.innerHTML = (own ? "" : row(esc, id("mode"), "Module", `<select id="${id("mode")}" data-mf="mode">${subs.map((m, i) =>
      `<option value="${esc(m.mode)}">${i + 1} - ${esc(m.title)}</option>`).join("")}</select>`)) +
      `<p class="su-note su-mod-what" data-mk="what"></p><div data-mk="fields"></div>
      ${row(esc, id("output_dir"), "Output folder", `<input id="${id("output_dir")}" data-mo="output_dir" spellcheck="false" placeholder="next to the parent">`)}
      <p class="su-note">${U.icon("info")}<span>The deck keeps the parent's functional, basis, spin and SCF settings; SCF TOLDEE is kept at 8 or
        tighter (the modules' own threshold). Two-component (SOC) parents are refused: these modules optimize strained cells.
        Read the finished run with Analysis › Mechanical Properties Results.</span></p>`;
    const sel = settings.querySelector('[data-mf="mode"]');
    const mode = () => (own ? calc : sel.value);
    const fieldHtml = ([key, label, kind, ph], m) => {
      const fid = id(`f-${key}`);
      if (m === "CPHF" && key === "maxcycle") { label = "SC-CP iterations (MAXCYCLE)"; ph = "100"; }
      if (m === "CPHF" && key === "tolalpha") ph = "3 (CPHF)";
      if (m === "CPHF" && key === "fmixing") ph = "no mixing";
      if (kind === "deform") return `<div class="su-row su-mod-row"><span class="su-l">${esc(label)}</span><div class="su-ctl su-mod-checks" data-f="deform" role="group" aria-label="${esc(label)}">${VOIGT.map((v, i) =>
        `<label class="su-check"><input type="checkbox" data-d="${i}"> ${v}</label>`).join("")}</div><div class="su-meta"><span class="su-h">${esc(ph)}</span></div></div>`;
      if (kind === "relax") return row(esc, fid, label, `<select id="${fid}" data-f="relax">
        <option value="opt">Optimize atoms in each strained cell (default)</option>
        <option value="nuchess">Internal-strain Hessian (NUCHESS, recommended)</option>
        <option value="clampion">Clamped ions (CLAMPION)</option></select>`);
      if (kind === "restart") return row(esc, fid, label, `<select id="${fid}" data-f="restart">
        <option value="">none</option><option value="RESTART">RESTART (continue an incomplete run)</option><option value="RESTART2">RESTART2 (re-analyze a complete run)</option></select>`);
      const step = kind === "float" ? ' step="any"' : kind === "int" ? ' step="1"' : "";
      const type = kind === "float" || kind === "int" ? "number" : "text";
      const placeholder = key === "stepsize" ? String((meta.modes.find((x) => x.mode === m) || {}).step ?? "") : ph;
      return row(esc, fid, label, `<input id="${fid}" type="${type}"${step} data-f="${key}" placeholder="${esc(placeholder)}" spellcheck="false">`);
    };
    const render = () => {
      const m = meta.modes.find((x) => x.mode === mode());
      const keys = new Set(m.keys);
      const relaxable = keys.has("nuchess") || keys.has("clampion");
      settings.querySelector('[data-mk="what"]').innerHTML = `${U.icon("info")}<span>${esc(m.what.charAt(0).toUpperCase() + m.what.slice(1))}. ${esc(m.manual)}.` +
        (m.pcrystal_only ? " Runs with Pcrystal only (not MPPcrystal)." : "") + (m.dims.length < 3 ? ` ${m.dims.map((x) => x + "D").join("/")} only.` : "") + "</span>";
      const flags = FLAGS.filter(([key]) => keys.has(key));
      settings.querySelector('[data-mk="fields"]').innerHTML = FIELDS.filter(([key]) => (key === "relax" ? relaxable : keys.has(key))).map((x) => fieldHtml(x, m.mode)).join("") +
        (flags.length ? `<div class="su-row su-mod-row"><span class="su-l">Options</span><div class="su-ctl su-mod-checks" role="group" aria-label="Options">${flags.map(([key, label]) =>
          `<label class="su-check"><input type="checkbox" data-f="${key}"${key === "print" ? " checked" : ""}> ${esc(label)}</label>`).join("")}</div><div class="su-meta"></div></div>` : "");
    };
    if (sel) sel.addEventListener("change", render);
    render();
    f.settingsOf = () => {
      const s = { mode: mode() };
      settings.querySelectorAll("[data-f]").forEach((x) => {
        const key = x.dataset.f;
        if (key === "deform") {
          const fl = [...x.querySelectorAll("[data-d]")].map((c) => (c.checked ? 1 : 0));
          if (fl.some(Boolean)) s.deform = fl;
        } else if (key === "relax") {
          if (x.value === "nuchess") s.nuchess = true; else if (x.value === "clampion") s.clampion = true;
        } else if (LISTS.includes(key)) {
          const v = val(x); if (v) s[key] = v.split(/[\s,]+/);
        } else if (key === "seismdir") {
          const v = val(x); if (v) s.seismdir = v.split(";").map((y) => y.trim().split(/[\s,]+/)).filter((y) => y.length === 3);
        } else {
          const v = val(x); if (v !== undefined && v !== false) s[key] = v;
        }
      });
      return s;
    };
  })();
  write.querySelector('[data-mact="write"]').addEventListener("click", async () => {
    if (!f.settingsOf) return;
    const b = { parent: d.path, settings: f.settingsOf() };
    const od = val(settings.querySelector('[data-mo="output_dir"]')); if (od) b.output_dir = od;
    showRun(write, U, null, true, "Writing the deck…");
    k("next").hidden = true;
    try {
      const r = await S.apiTask("elastic/generate", b);
      f.command = r.command || "";
      if (r.code === 0 && r.deck) {
        await S.refreshFiles();               // the Explorer knows the deck before Open Deck opens it
        showRun(write, U, r, true, `Wrote ${r.deck}`);
        k("next").innerHTML = `<button type="button" class="tb-btn-text" data-mact="open">${U.icon("file-text")}<span>Open Deck</span></button>` +
          `<button type="button" class="tb-btn-text" data-mact="submit">${U.icon("play")}<span>Submit…</span></button>` +
          `<button type="button" class="tb-btn-text" data-mact="reveal">${U.icon("files")}<span>Reveal in Explorer</span></button>`;
        k("next").hidden = false;
        k("next").onclick = (e) => {
          const a = e.target.closest("[data-mact]");
          if (!a) return;
          if (a.dataset.mact === "open") S.openPath(r.deck);
          else if (a.dataset.mact === "submit") S.run("calc.submit", { paths: [r.deck] });
          else S.reveal(r.deck);
        };
      } else showRun(write, U, r, false, "Not written: see the log");
    } catch (e) { showRun(write, U, null, false, e.message); }
  });
  return f;
}

/* ---------------------------------------------------------------- 8 Phonopy: mace opt2d12 --calc-type PHONOPY */
function phonopyForm(S, U, d) {
  const esc = U.esc;
  const id = (k) => `su-m-phonopy-${k}`;
  const settings = el(`<div class="su-mod" data-mod="PHONOPY">
    <p class="su-note" data-mk="info">${U.icon("info")}<span>Looking for phonopy…</span></p>
    <div class="su-row su-mod-row"><span class="su-l" id="${id("dim-l")}">Supercell</span><div class="su-ctl su-mod-dim" role="group" aria-labelledby="${id("dim-l")}">${[0, 1, 2].map((n) =>
      `<input type="number" min="1" max="8" step="1" value="2" data-dim="${n}" aria-label="N${n + 1}">`).join("<span aria-hidden=\"true\">×</span>")}</div>
      <div class="su-meta"><span class="su-h">diagonal; radius ≥ 10 Å recommended</span></div></div>
    ${row(esc, id("amp"), "Displacement (Å)", `<input id="${id("amp")}" type="number" data-pk="amplitude" step="0.001" min="0.001" max="0.1" placeholder="0.01 (phonopy's default)">`)}
    ${row(esc, id("pm"), "± displacements", `<select id="${id("pm")}" data-pk="pm"><option value="auto">auto</option><option value="yes">yes</option><option value="no">no</option></select>`)}
    ${row(esc, id("d12"), "Parent deck", `<input id="${id("d12")}" data-pk="d12" spellcheck="false" placeholder="default: same name, .d12">`)}
    ${row(esc, id("out"), "Output folder", `<input id="${id("out")}" data-pk="output_dir" spellcheck="false" placeholder="next to the parent (the set's own folder goes in it)">`)}
    <div class="su-row su-mod-row"><span class="su-l">Options</span><div class="su-ctl su-mod-checks" role="group" aria-label="Options">
      <label class="su-check"><input type="checkbox" data-pk="cry_sym"> Symmetry in each displaced cell (CRY_SYM)</label>
      <label class="su-check"><input type="checkbox" data-pk="keep_shrink"> Keep the parent's SHRINK</label>
      <label class="su-check"><input type="checkbox" data-pk="force"> Write into a non-empty folder</label></div><div class="su-meta"></div></div>
    <p class="su-note">${U.icon("info")}<span>A finished 3D OPT or SP output is the parent. Run the displaced decks (Submit; do not track them),
      then collect their forces and plot the phonons with Analysis › Phonons with Phonopy.</span></p></div>`);
  const write = writePart(U, "Write the displaced decks", "Generate Displaced Decks");
  const f = { settings, write, command: "" };
  const k = (n) => write.querySelector(`[data-mk="${n}"]`);
  S.api("phonopy/info").then((inf) => {
    settings.querySelector('[data-mk="info"]').innerHTML = `${U.icon(inf.phonopy ? "info" : "triangle-alert")}<span>${esc(inf.phonopy
      ? `phonopy ${inf.version || ""} found where the MACE server runs.` : `phonopy is not installed where the MACE server runs: ${inf.install}`)}</span>`;
  }).catch((e) => { settings.querySelector('[data-mk="info"]').innerHTML = `${U.icon("circle-alert")}<span>${esc(e.message)}</span>`; });
  write.querySelector('[data-mact="write"]').addEventListener("click", async () => {
    const b = { parent: d.path };
    settings.querySelectorAll("[data-pk]").forEach((x) => { const v = val(x); if (v !== undefined && v !== false) b[x.dataset.pk] = v; });
    b.dim = [...settings.querySelectorAll("[data-dim]")].map((i) => String(Math.max(1, Math.round(Number(i.value) || 1)))).join(" ");
    showRun(write, U, null, true, "Writing the displaced decks…");
    k("next").hidden = true;
    try {
      const r = await S.apiTask("phonopy/generate", b);
      f.command = r.command || "";
      if (r.code === 0 && r.dir) {
        await S.refreshFiles();
        showRun(write, U, r, true, `Wrote the set in ${r.dir}` + (r.walltime ? ` (walltime per job ~ ${r.walltime})` : ""));
        k("next").innerHTML = `<button type="button" class="tb-btn-text" data-mact="submit">${U.icon("play")}<span>Submit the Jobs…</span></button>` +
          `<button type="button" class="tb-btn-text" data-mact="phonons">${U.icon("audio-waveform")}<span>Phonons with Phonopy…</span></button>` +
          `<button type="button" class="tb-btn-text" data-mact="reveal">${U.icon("files")}<span>Reveal in Explorer</span></button>`;
        k("next").hidden = false;
        k("next").onclick = (e) => {
          const a = e.target.closest("[data-mact]");
          if (!a) return;
          if (a.dataset.mact === "submit") S.builder("submit_dialog", "open", { targets: [r.dir], preset: r.walltime ? { walltime: r.walltime } : null });
          else if (a.dataset.mact === "phonons") S.run("analysis.phonopy", { path: r.dir });
          else S.reveal(r.dir);
        };
      } else showRun(write, U, r, false, "mace opt2d12 --calc-type PHONOPY refused or failed: see the log");
    } catch (e) { showRun(write, U, null, false, e.message); }
  });
  return f;
}
