/* Settings (Mod+,), loaded the first time it opens (spec §8.5). One
 * declarative SCHEMA drives the eight sections, the search and the reset
 * links; changes apply at once (no OK button) and save after 500 ms.
 *
 * Folder settings (1.1.6): a row a project folder's mace-folder.toml can
 * hold shows where its value comes from (a badge: Built-in, Mine, This
 * folder, This session) once the folder has a file, and offers Save to this
 * folder, Save as my default and Reset to my default; "This folder" lists the
 * file, its warnings and its command-line values (settings.js has the layers). */
import { loadCss } from "./lazy_util.js";
import { SECTIONS, SCHEMA, P3 } from "./settings_schema.js";
import { LIGHTING, STYLES, LIGHTING_KEYS, STYLE_KEYS, byId } from "../viewport/presets.js";

export { SECTIONS, SCHEMA };

let H, dlg, pane, nav, search, section = "appearance", res = null, cli = null;

/** Open (or bring back) the dialog; H is the shell's handle. */
export async function open(handle, at) {
  H = handle;
  await loadCss(import.meta.url, "../../css/dialogs.css");
  if (at) section = at;
  if (!dlg) build();
  if (!dlg.open) { H.overlayOpening(); H.opener = document.activeElement; dlg.showModal(); }
  search.value = "";
  render();
  search.focus();
  H.api("info").then((r) => { res = r.resources || null; if (dlg.open) refreshDyn(); }).catch(() => {});
  return dlg;
}

function build() {
  const { esc, icon } = H;
  dlg = document.createElement("dialog");
  dlg.id = "settings";
  dlg.className = "dlg settings-dlg";
  dlg.setAttribute("aria-label", "Settings");
  dlg.innerHTML = `<div class="dlg-head"><h2>Settings</h2>
    <input type="search" class="field set-search" id="set-search" placeholder="Search settings" aria-label="Search settings" autocomplete="off">
    <button class="icon-btn sm" data-close aria-label="Close">${icon("x")}</button></div>
    <div class="set-main"><nav class="set-nav" aria-label="Sections">${SECTIONS.map(([k, t]) =>
      `<button class="set-nav-item" data-sec="${k}"><span>${esc(t)}</span><span class="n"></span></button>`).join("")}</nav>
    <div class="set-pane" id="set-pane" tabindex="-1"></div></div>
    <div class="dlg-foot"><span class="grow muted">Saved to this browser and <code>${esc(H.settingsPath() || "~/.config/mace/gui.json")}</code><span class="set-foot-folder"></span></span></div>`;
  document.body.appendChild(dlg);
  pane = dlg.querySelector("#set-pane"); nav = dlg.querySelector(".set-nav"); search = dlg.querySelector("#set-search");
  dlg.querySelector("[data-close]").addEventListener("click", () => dlg.close());
  dlg.addEventListener("close", () => { if (H.opener && H.opener.focus) H.opener.focus(); });
  nav.addEventListener("click", (e) => { const b = e.target.closest("[data-sec]"); if (b) { section = b.dataset.sec; search.value = ""; render(); } });
  search.addEventListener("input", render);
  pane.addEventListener("change", onChange);
  pane.addEventListener("input", onInput);
  pane.addEventListener("keydown", (e) => { if (e.key === "Enter" && e.target.matches("input[type=number], input[type=text]")) commit(e.target); });
  pane.addEventListener("focusout", (e) => { if (e.target.matches("input[type=number], input[type=text], textarea")) commit(e.target); });
  pane.addEventListener("click", onClick);
  // a theme card previews its theme while the pointer or the focus is on it (Esc or leaving puts it back)
  const card = (e) => e.target.closest && e.target.closest(".theme-card");
  pane.addEventListener("pointerover", (e) => { const c = card(e); if (c) H.previewTheme(c.dataset.val); });
  pane.addEventListener("pointerout", (e) => { const c = card(e); if (c && !c.contains(e.relatedTarget)) H.previewTheme(null); });
  pane.addEventListener("focusin", (e) => { const c = card(e); if (c && c.matches(":focus-visible")) H.previewTheme(c.dataset.val); });
  pane.addEventListener("focusout", (e) => { if (card(e)) H.previewTheme(null); });
  dlg.addEventListener("close", () => H.previewTheme(null));
  H.settings.onChange((k) => {
    if (!dlg.open) return;
    refreshRow(k);
    SCHEMA.filter((r) => r.needs && r.needs[0] === k).forEach((r) => refreshRow(r.key));   // its dependents
  });
}

/* ---------------------------------------------------------------- render */
const matchRow = (r, q) => !q || `${r.label} ${r.key} ${r.help || ""} ${SECTIONS.find((s) => s[0] === r.section)[1]}`.toLowerCase().includes(q);
function render() {
  const q = search.value.trim().toLowerCase();
  const counts = {};
  SCHEMA.forEach((r) => { if (matchRow(r, q)) counts[r.section] = (counts[r.section] || 0) + 1; });
  if (q && "keyboard shortcuts keymap keybindings".includes(q)) counts.keyboard = 1;
  nav.querySelectorAll("[data-sec]").forEach((b) => {
    const k = b.dataset.sec;
    b.classList.toggle("active", !q && k === section);
    b.setAttribute("aria-current", String(!q && k === section));
    b.classList.toggle("dim", !!q && !counts[k]);
    b.querySelector(".n").textContent = q && counts[k] ? counts[k] : "";
  });
  if (q) {
    const rows = SCHEMA.filter((r) => matchRow(r, q));
    pane.innerHTML = rows.length ? SECTIONS.filter(([k]) => rows.some((r) => r.section === k)).map(([k, t]) =>
      `<h3 class="set-h">${H.esc(t)}</h3>${rows.filter((r) => r.section === k).map(rowHtml).join("")}`).join("")
      : `<p class="set-empty">No settings match “${H.esc(search.value.trim())}”</p>`;
    return;
  }
  const title = SECTIONS.find((s) => s[0] === section)[1];
  footFolder();
  if (section === "folder") { pane.innerHTML = folderPane(title); return; }
  if (section === "clusterdefaults") {
    // the site settings, read and changed through `mace site` (cluster_defaults.js)
    pane.innerHTML = `<h3 class="set-h">${H.esc(title)}</h3><div id="cluster-defaults-host"><p class="muted">Loading…</p></div>`;
    import("./cluster_defaults.js").then((m) => m.mount(pane.querySelector("#cluster-defaults-host"), H)).catch((e) => {
      const host = pane.querySelector("#cluster-defaults-host"); if (host) host.textContent = `The cluster defaults did not load: ${e.message}`; });
    return;
  }
  if (section === "keyboard") {
    pane.innerHTML = `<h3 class="set-h">${H.esc(title)}</h3><div id="keymap-host"><p class="muted">Loading…</p></div>`;
    import("./keymap_editor.js").then((m) => m.mount(pane.querySelector("#keymap-host"), H)).catch((e) => {
      const host = pane.querySelector("#keymap-host"); if (host) host.textContent = `The keymap editor did not load: ${e.message}`; });
    return;
  }
  const secRows = SCHEMA.filter((r) => r.section === section);
  // a section whose rows all wait for the same later phase says so once, not on every row
  const laters = new Set(secRows.map((r) => (r.wired === false ? (r.later ?? P3) : null)));
  const note = laters.size === 1 && [...laters][0] ? [...laters][0] : "";
  pane.innerHTML = `<h3 class="set-h">${H.esc(title)}</h3>` + (note ? `<p class="set-note">${H.esc(note)}: the value is kept for a later version.</p>` : "") +
    secRows.map((r) => rowHtml(r, !!note)).join("") +
    (section === "advanced" ? `<div class="set-actions"><button class="tb-btn-text" data-act="export">${H.icon("download")}<span>Export settings…</span></button>
      <button class="tb-btn-text" data-act="import">${H.icon("upload")}<span>Import settings…</span></button>
      <button class="tb-btn-text danger" data-act="reset-all">${H.icon("rotate-ccw")}<span>Reset all…</span></button>
      <input type="file" accept="application/json,.json" id="set-import" hidden></div><div class="set-confirm" hidden></div>` : "");
  if (section === "mace") loadCli();
  const fi = pane.querySelector("#set-import");
  if (fi) fi.addEventListener("change", () => { importSettings(fi.files[0]); fi.value = ""; });
  refreshDyn();
}

function rowHtml(r, sectionNote = false) {
  const { esc } = H, v = H.settings.get(r.key), id = `set-${r.key.replace(/\./g, "-")}`, err = `${id}-err`;
  // a control that only matters with another setting's value (Custom background) is off until then
  const off = r.needs && H.settings.get(r.needs[0]) !== r.needs[1];
  let ctl;
  if (r.control === "themes") {
    // one card per theme: its swatch (tokens.css), name and what it is for; pointing at one previews it.
    // The cards come in labeled groups (js/themes.js THEME_GROUPS), one radio group over all of them.
    const card = ([o, t, d]) => `<button role="radio" class="theme-card" data-key="${r.key}" data-val="${esc(o)}" aria-checked="${String(v) === String(o)}" title="${esc(d)}">` +
      `<span class="theme-swatch" data-swatch="${esc(o)}" aria-hidden="true"><i></i><i></i><i></i></span><span class="theme-name">${esc(t)}</span></button>`;
    const groups = [];
    for (const opt of r.options) {
      const g = opt[3] || "";
      if (!groups.length || groups[groups.length - 1].label !== g) groups.push({ label: g, opts: [] });
      groups[groups.length - 1].opts.push(opt);
    }
    ctl = `<span class="segment theme-groups" role="radiogroup" aria-label="${esc(r.label)}">${groups.map((g, n) =>
      `<span class="theme-group" role="group"${g.label ? ` aria-labelledby="${id}-g${n}"` : ""}>` +
      (g.label ? `<span class="theme-group-h" id="${id}-g${n}">${esc(g.label)}</span>` : "") +
      `<span class="theme-grid">${g.opts.map(card).join("")}</span></span>`).join("")}</span>`;
  } else if (r.control === "segmented") {
    ctl = `<span class="segment" role="radiogroup" aria-label="${esc(r.label)}">${r.options.map(([o, t]) =>
      `<button role="radio" data-key="${r.key}" data-val="${esc(o)}" aria-checked="${String(v) === String(o)}">${esc(t)}</button>`).join("")}</span>`;
  } else if (r.control === "select" || r.control === "cli") {
    const os = r.control === "cli" ? cliOptions(r) : typeof r.options === "function" ? r.options(H) : r.options;
    ctl = `<select class="field" id="${id}" data-key="${r.key}">${os.map(([o, t]) => `<option value="${esc(o)}"${String(v) === String(o) ? " selected" : ""}${(r.disabled || []).includes(o) ? " disabled" : ""}>${esc(t)}</option>`).join("")}</select>`;
  } else if (r.control === "check") {
    ctl = `<input type="checkbox" class="set-check" id="${id}" data-key="${r.key}"${v ? " checked" : ""}>`;
  } else if (r.control === "range") {
    ctl = `<span class="set-range"><input type="range" id="${id}" data-key="${r.key}" min="${r.min}" max="${r.max}" step="${r.step}" value="${v}">` +
      `<output for="${id}">${fmtNum(v, r)}</output></span>`;
  } else if (r.control === "color") {
    ctl = `<span class="set-color"><input type="color" id="${id}" data-key="${r.key}"${off ? " disabled" : ""} value="${/^#[0-9a-f]{6}$/i.test(v) ? v : H.tokenColor(r.key)}">` +
      `<span class="muted small">${v ? esc(v) : "Theme"}</span></span>`;
  } else if (r.control === "triple" || r.control === "pair") {
    ctl = `<span class="set-num"><input type="text" class="field" id="${id}" data-key="${r.key}" value="${esc((v || []).join(", "))}"` +
      ` spellcheck="false" inputmode="decimal" aria-describedby="${err}">${r.unit ? `<span class="unit">${esc(r.unit)}</span>` : ""}</span>`;
  } else if (r.control === "saved") {
    // a value the app keeps for you (no field to type in): what is saved, and Reset forgets it
    ctl = `<output class="set-saved muted small" id="${id}">${esc(r.summary(v))}</output>`;
  } else if (r.control === "list") {
    ctl = `<textarea class="field set-list" id="${id}" data-key="${r.key}" rows="3" spellcheck="false" aria-describedby="${err}">${esc((v || []).join("\n"))}</textarea>`;
  } else {
    const type = r.control === "number" ? "number" : "text";
    ctl = `<span class="set-num"><input type="${type}" class="field" id="${id}" data-key="${r.key}" value="${esc(v)}"` +
      (type === "number" ? ` min="${r.min}" max="${r.max}" step="${r.step}" inputmode="decimal"` : ` spellcheck="false"`) +
      ` aria-describedby="${err}">${r.unit ? `<span class="unit">${esc(r.unit)}</span>` : ""}</span>`;
  }
  if (r.action) ctl += ` <button type="button" class="tb-btn-text sm" data-act="${esc(r.action[0])}">${esc(r.action[1])}</button>`;
  const hint = [r.help, r.wired === false && !sectionNote ? (r.later ?? P3) : ""].filter(Boolean).join(" · ");
  const lab = r.control === "segmented" || r.control === "themes" ? `<span class="lbl" id="${id}-l">${esc(r.label)}</span>` : `<label id="${id}-l" for="${id}">${esc(r.label)}</label>`;
  return `<div class="set-row${off ? " off" : ""}" data-row="${r.key}">${lab}<div class="set-ctl">${ctl}` +
    `${hint ? `<div class="set-help">${esc(hint)}</div>` : ""}<div class="set-err" id="${err}" role="alert"></div><div class="set-dyn" data-dyn="${r.key}"></div>${layerHtml(r)}</div>` +
    `<button class="set-reset" data-reset="${r.key}"${resetHidden(r.key) ? " hidden" : ""}>Reset</button></div>`;
}
/* The plain Reset link resets the user's own value: off while the folder or this session decides */
const resetHidden = (k) => H.settings.isDefault(k) || (H.settings.source && ["folder", "session"].includes(H.settings.source(k)));

/* ---------------------------------------------------------------- the folder layer */
const SRC_LABEL = { default: "Built-in", user: "Mine", folder: "This folder", session: "This session" };
function srcTitle(src) {
  const f = H.settings.folderInfo();
  return { default: "MACE's built-in value", user: `Your default (${H.settingsPath() || "~/.config/mace/gui.json"})`,
    folder: `Set by ${f.file || "this folder's mace-folder.toml"}`, session: "Changed in this session; not saved yet" }[src];
}
function layerHtml(r) {
  if (!H.settings.folderable || !H.settings.folderable(r.key)) return "";
  const { esc } = H, f = H.settings.folderInfo(), src = H.settings.source(r.key);
  const badge = f.file || src === "session"
    ? `<span class="set-src" data-src="${src}" title="${esc(srcTitle(src))}">${esc(SRC_LABEL[src])}</span>` : "";
  const b = (act, label, title, extra = "") => `<button class="set-layer-btn${extra}" data-layer="${act}" data-key="${r.key}" title="${esc(title)}">${esc(label)}</button>`;
  const acts = [];
  const accepts = !H.settings.folderAccepts || H.settings.folderAccepts(r.key);
  if (src !== "folder" && f.can_save && !accepts) {
    acts.push(`<span class="set-layer-why muted small" title="A folder file names a preset palette; a palette of your own stays in your ~/.config/mace/element_colors.json">Your own palette: not saved to a folder</span>`);
  } else if (src !== "folder") acts.push(f.can_save ? b("folder", "Save to this folder", `Write this value into ${f.target}`, src === "session" ? "" : " quiet")
    : (src === "session" ? `<span class="set-layer-why muted small" title="${esc(f.why_not)}">Cannot save to this folder</span>` : ""));
  if (src === "session" || src === "folder") acts.push(b("mine-save", "Save as my default", "Keep this value as your own default (gui.json)"));
  if (src === "session" || (src === "folder" && f.can_save)) acts.push(b("mine", "Reset to my default",
    src === "folder" ? `Remove it from ${f.file}; your default applies` : "Forget this session's change"));
  const html = acts.filter(Boolean).join("");
  return badge || html ? `<div class="set-layer" data-layer-row="${r.key}">${badge}${html}</div>` : "";
}
function footFolder() {
  const el = dlg && dlg.querySelector(".set-foot-folder");
  if (!el || !H.settings.folderInfo) return;
  const f = H.settings.folderInfo();
  el.innerHTML = f.file ? ` · this folder: <code>${H.esc(f.file)}</code>` : "";
}
const fmtVal = (v) => (Array.isArray(v) ? v.join(", ") : typeof v === "string" ? (v || "(blank)") : String(v));
function folderPane(title) {
  const { esc } = H, f = H.settings.folderInfo();
  const rows = SCHEMA.filter((r) => r.key in (f.gui || {}));
  const help = f.help || {};
  let html = `<h3 class="set-h">${esc(title)}</h3>`;
  html += f.file
    ? `<p class="set-folder-file">Folder settings: <code>${esc(f.file)}</code>. They apply to everyone who works in this folder, in mace gui and in the CLI commands run here; a change you make in this session, or a command-line option, still wins.</p>`
    : `<p class="set-folder-file">This folder has no <code>${esc(f.name || "mace-folder.toml")}</code>. Save to this folder, next to any setting below a badge, creates <code>${esc(f.target || "")}</code>: a commented file you can keep in git or share with your group.</p>`;
  if (!f.can_save && f.why_not) html += `<p class="set-note">${esc(f.why_not)}</p>`;
  if ((f.warnings || []).length) html += `<ul class="set-folder-warn">${f.warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>`;
  if (rows.length) {
    html += `<h4 class="set-sub">Set by this folder</h4>` + rows.map((r) => `<div class="set-row" data-row="${r.key}"><span class="lbl">${esc(r.label)}</span>` +
      `<div class="set-ctl"><span class="set-folder-val">${esc(fmtVal(f.gui[r.key]))}` +
      `${H.settings.source(r.key) === "session" ? `<span class="muted"> · this session: ${esc(fmtVal(H.settings.get(r.key)))}</span>` : ""}</span>${layerHtml(r)}</div></div>`).join("");
  }
  const cli = Object.entries(f.cli || {});
  if (cli.length) {
    html += `<h4 class="set-sub">For the command line and Submit</h4><dl class="set-folder-cli">` +
      cli.map(([k, v]) => `<dt><code>${esc(k)}</code></dt><dd>${esc(fmtVal(v))}${help[k] ? ` <span class="muted small">· ${esc(help[k])}</span>` : ""}</dd>`).join("") + "</dl>";
  }
  if (f.file && !rows.length && !cli.length) html += `<p class="muted">The file sets nothing yet.</p>`;
  return html;
}
async function layerAction(btn) {
  const k = btn.dataset.key, act = btn.dataset.layer, label = (SCHEMA.find((r) => r.key === k) || {}).label || k;
  btn.disabled = true;
  try {
    if (act === "folder") { await H.settings.saveToFolder(k); H.toast(`Saved ${label} to this folder`, "success"); }
    else if (act === "mine-save") {
      const r = H.settings.saveAsDefault(k);
      H.toast(r.stillFolder ? `Saved ${label} as your default; this folder's value still applies here` : `Saved ${label} as your default`, "success");
    } else if (act === "mine") { await H.settings.resetToMine(k); H.toast(`${label}: your default applies`, "success"); }
  } catch (e) {
    H.toast(`${label}: ${e.message}`, "error");
  } finally {
    btn.disabled = false;
  }
  if (section === "folder" && !search.value.trim()) render(); else refreshRow(k);
  footFolder();
}
const fmtNum = (v, r) => `${Number(v).toFixed(String(r.step).includes(".") ? String(r.step).split(".")[1].length : 0)}${r.unit ? " " + r.unit : ""}`;

/* "CLI default: X", read from the server (never a GUI-only default). */
function cliOptions(r) {
  const d = (cli && cli[r.cli]) || null, blank = `CLI default${d ? ` (${d})` : ""}`;
  let list = [];
  if (r.key === "exampleConfig") list = (H.templates() || []).map((t) => t.name);
  else if (r.key === "basisSet") list = (cli && cli.basis_sets) || [];
  else list = (cli && cli.functionals) || [];
  const v = H.settings.get(r.key);
  if (v && !list.includes(v)) list = [v, ...list];
  return [["", blank], ...list.map((x) => [x, x])];
}
function loadCli() {
  const b = H.boot();
  if (b && b.mace_defaults) { cli = b.mace_defaults; }
  SCHEMA.filter((r) => r.control === "cli").forEach((r) => {
    const el = pane.querySelector(`[data-dyn="${r.key}"]`);
    const d = cli && cli[r.cli];
    if (el) el.textContent = `CLI default: ${d || "unknown"}`;
    if (el && cli && cli.source) el.title = `From ${cli.source[r.cli] || "the CLI"}`;
  });
}
function refreshDyn() {
  const w = pane.querySelector('[data-dyn="max_workers"]'), c = pane.querySelector('[data-dyn="cache_mb"]');
  if (w && res) w.textContent = `Auto: ${res.workers_cap} of ${res.cores} cores (25%) · using ${res.workers_effective}` + (res.pinned_by_env.includes("max_workers") ? " (set by MACE_GUI_MAX_WORKERS)" : "");
  if (c && res) c.textContent = `In use: ${res.cache_used_mb} MB of ${res.cache_mb} MB` + (res.pinned_by_env.includes("cache_mb") ? " (set by MACE_GUI_CACHE_MB)" : "");
}

/* A setting changed (here, by a shortcut, or reverted by the server): redraw
 * its row, except a field being typed in, which only updates its Reset link. */
function refreshRow(k) {
  if (section === "folder" && !search.value.trim()) { render(); return; }
  const row = pane.querySelector(`[data-row="${CSS.escape(k)}"]`), r = SCHEMA.find((x) => x.key === k);
  if (!row || !r) return;
  const typing = row.contains(document.activeElement) && document.activeElement.matches("input[type=number], input[type=text], textarea");
  if (typing) { row.querySelector("[data-reset]").hidden = resetHidden(k); return; }
  const had = row.contains(document.activeElement) ? document.activeElement.dataset.val || "ctl" : null;
  row.outerHTML = rowHtml(r, !!pane.querySelector(".set-note"));
  if (had) {   // keep keyboard focus on the control that changed
    const nr = pane.querySelector(`[data-row="${CSS.escape(k)}"]`);
    const f = had === "ctl" ? nr.querySelector("[data-key]") : nr.querySelector(`[data-val="${CSS.escape(had)}"]`);
    if (f) f.focus();
  }
  if (r.control === "cli") loadCli();
  refreshDyn();
}

/* ---------------------------------------------------------------- edits */
function typed(r, raw) {
  const def = H.settings.DEFAULTS[r.key];
  if (typeof def === "number") return Number(raw);
  if (typeof def === "boolean") return raw === true || raw === "true";
  return raw;
}
function onChange(e) {
  const t = e.target, k = t.dataset.key;
  if (!k) return;
  const r = SCHEMA.find((x) => x.key === k);
  if (t.type === "checkbox") H.settings.set(k, t.checked);
  else if (t.tagName === "SELECT" || t.type === "range") {
    H.settings.set(k, typed(r, t.value));
    // a Lighting or Style preset chosen as the default: its values become the defaults too (Custom: the saved look)
    const which = k === "renderPreset" ? [LIGHTING, LIGHTING_KEYS, "renderCustom"] : k === "stylePreset" ? [STYLES, STYLE_KEYS, "styleCustom"] : null;
    if (which) {
      const p = byId(which[0], t.value), saved = H.settings.get(which[2]);
      const vals = p ? p.values : saved && typeof saved === "object" ? saved : {};
      for (const key of which[1]) if (key in vals) H.settings.set(key, vals[key]);
    }
  }
  else if (t.type === "color") H.settings.set(k, t.value);
}
let inputFrame = 0;
function onInput(e) {
  const t = e.target;
  if (t.type !== "range") return;
  const r = SCHEMA.find((x) => x.key === t.dataset.key);
  t.nextElementSibling.textContent = fmtNum(t.value, r);
  cancelAnimationFrame(inputFrame);   // live preview, one per frame
  inputFrame = requestAnimationFrame(() => H.settings.set(r.key, Number(t.value)));
}
function commit(t) {
  const k = t.dataset.key, r = SCHEMA.find((x) => x.key === k);
  if (!r) return;
  const err = pane.querySelector(`#${t.getAttribute("aria-describedby")}`);
  const say = (m) => { if (err) err.textContent = m; t.setAttribute("aria-invalid", m && m.startsWith("Enter") ? "true" : "false"); };
  if (r.control === "list") { H.settings.set(k, t.value.split("\n").map((x) => x.trim()).filter(Boolean)); return; }
  if (r.control === "pair") {
    const xs = t.value.split(/[\s,;]+/).filter(Boolean).map(Number);
    if (xs.length !== 2 || !xs.every((x) => Number.isFinite(x) && x >= r.min && x <= r.max) || xs[1] - xs[0] < 0.1) {
      say(`Enter 2 numbers from ${r.min} to ${r.max}, low then high, e.g. -6, 6`); return;
    }
    say(""); H.settings.set(k, xs); return;
  }
  if (r.control === "triple") {
    const xs = t.value.split(/[\s,;]+/).filter(Boolean).map(Number);
    if (xs.length !== 3 || !xs.every((x) => Number.isFinite(x) && x >= r.min && x <= r.max)) { say(`Enter 3 numbers from ${r.min} to ${r.max}, e.g. 0, 0, 0`); return; }
    const other = H.settings.get(k === "displayLo" ? "displayHi" : "displayLo");
    const lo = k === "displayLo" ? xs : other, hi = k === "displayLo" ? other : xs;
    if (lo.some((x, j) => hi[j] - x < 0.05)) { say("The range's end must be above its start by at least 0.05 on every axis"); return; }
    say(""); H.settings.set(k, xs); return;
  }
  if (r.control === "element") {
    const sym = t.value.trim();
    if (!/^[A-Z][a-z]?$/.test(sym) || !H.elements().includes(sym)) { say("Enter an element symbol, e.g. C or Fe"); return; }
    say(""); H.settings.set(k, sym); return;
  }
  if (r.control !== "number") { say(""); H.settings.set(k, t.value.trim()); return; }
  const v = Number(t.value);
  if (t.value.trim() === "" || !Number.isFinite(v)) { say(`Enter a number from ${r.min} to ${r.max}`); return; }
  let c = Math.min(r.max, Math.max(r.min, v));
  if (r.step === 1) c = Math.round(c);
  say(c !== v ? `Must be ${r.min}–${r.max}${r.unit ? " " + r.unit : ""}: set to ${c}` : "");
  if (c !== v) t.value = String(c);
  H.settings.set(k, c);
}
function onClick(e) {
  const seg = e.target.closest(".segment [data-key]");
  if (seg) { H.settings.set(seg.dataset.key, typed(SCHEMA.find((x) => x.key === seg.dataset.key), seg.dataset.val)); return; }
  const lay = e.target.closest("[data-layer]");
  if (lay) { layerAction(lay); return; }
  const rs = e.target.closest("[data-reset]");
  if (rs) { H.settings.reset(rs.dataset.reset); refreshRow(rs.dataset.reset); return; }
  const act = e.target.closest("[data-act]");
  if (!act) return;
  if (act.dataset.act === "export") chooser("export");
  else if (act.dataset.act === "export-go") exportProfile();
  else if (act.dataset.act === "palette-editor") H.elementColors();
  else if (act.dataset.act === "import") pane.querySelector("#set-import").click();
  else if (act.dataset.act === "import-go") applyImport();
  else if (act.dataset.act === "reset-all") chooser("reset");
  else if (act.dataset.act === "reset-yes") resetProfile();
  else if (act.dataset.act === "reset-no") { const box = pane.querySelector(".set-confirm"); box.hidden = true; pendingImport = null; }
}

/* ---------------------------------------------------------------- the whole profile (polish P4) */
/* What Export, Import and Reset cover: the GUI profile only (gui.json and
 * gui-docprefs.json, and this browser's caches of them); `mace gui connect`'s
 * connect.toml is never exported, imported or reset here. */
const PARTS = [
  ["settings", "Settings", true, ""], ["ui", "Layout and panels", true, ""],
  ["memory", "Remembered choices", true, "Export choices, New Structure form, Clean method"],
  ["recent", "Recent files and folders", false, "Paths on this computer; leave off when sharing the file"],
  ["docprefs", "Per-document view options", false, "Paths on this computer; leave off when sharing the file"],
];
const RESET_LABEL = { settings: "Settings and keybindings", ui: "Layout", memory: "Remembered choices",
  recent: "Recent files and folders", docprefs: "Per-document view options" };
function chooser(kind) {
  const { esc } = H, box = pane.querySelector(".set-confirm");
  box.hidden = false;
  const rows = PARTS.map(([k, t, on, help]) => `<label class="set-part"><input type="checkbox" data-part="${k}"${on ? " checked" : ""}>` +
    `<span>${esc(kind === "reset" ? RESET_LABEL[k] : t)}</span>${help && kind !== "reset" ? `<small class="muted"> · ${esc(help)}</small>` : ""}</label>`).join("");
  box.innerHTML = kind === "export"
    ? `<div class="set-parts">${rows}</div><button class="tb-btn-text primary" data-act="export-go">Download</button><button class="tb-btn-text" data-act="reset-no">Cancel</button>`
    : `<div class="set-parts">${rows}</div><p class="muted">A copy of your current profile is saved first (Settings › Advanced keeps the last 3).</p>` +
      `<button class="tb-btn-text danger" data-act="reset-yes">Reset</button><button class="tb-btn-text" data-act="reset-no">Cancel</button>`;
  box.querySelector("button").focus();
}
const picked = () => [...pane.querySelectorAll(".set-confirm [data-part]")].filter((x) => x.checked).map((x) => x.dataset.part);

async function exportProfile() {
  const parts = picked();
  await H.settings.flush();
  await H.profile.flush();
  await H.docprefs.flush();
  const all = await H.api("profile?all=1");
  const out = { mace_gui_profile: 2, exported: new Date().toISOString(), mace_version: (H.boot() && H.boot().info && H.boot().info.version) || "" };
  if (parts.includes("settings")) out.settings = H.settings.nonDefault();
  if (parts.includes("ui")) out.ui = all.ui || {};
  if (parts.includes("memory")) out.memory = all.memory || {};
  if (parts.includes("recent")) out.recent = all.recent || {};
  if (parts.includes("docprefs")) out.docprefs = all.docprefs || {};
  const d = new Date(), stamp = `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, "0")}${String(d.getDate()).padStart(2, "0")}`;
  const blob = new Blob([JSON.stringify(out, null, 2) + "\n"], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = `mace-gui-profile-${stamp}.json`;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  if (window.__maceExportHook) window.__maceExportHook(JSON.stringify(out), "profile");
  pane.querySelector(".set-confirm").hidden = true;
  H.toast(`Downloaded ${a.download}`, "success");
}

let pendingImport = null;
export async function importSettings(file) {
  if (!file) return;
  let data;
  try { data = JSON.parse(await file.text()); }
  catch (e) { H.toast(`Not a settings file: ${e.message}`, "error"); return; }
  return previewImport(data);
}
/** What an imported file would replace, in a confirm box (a profile, today's {version, settings}, or a bare settings object). */
export function previewImport(data) {
  if (!data || typeof data !== "object") { H.toast("Not a settings file", "error"); return null; }
  const prof = data.mace_gui_profile ? data : { settings: data && typeof data.settings === "object" ? data.settings : data };
  pendingImport = prof;
  const parts = [];
  if (prof.settings) parts.push(`Settings: ${Object.keys(prof.settings).length} value${Object.keys(prof.settings).length === 1 ? "" : "s"}`);
  if (prof.ui) parts.push("Layout");
  if (prof.memory) parts.push("Remembered choices");
  if (prof.recent && prof.recent.files) {
    const n = Object.values(prof.recent.files).reduce((t, e) => t + ((e && e.items) || []).length, 0);
    parts.push(`Recent files: ${n}`);
  }
  if (prof.docprefs) parts.push("Per-document view options");
  const box = pane.querySelector(".set-confirm");
  box.hidden = false;
  box.innerHTML = `<span class="set-import-what">Import replaces: ${H.esc(parts.join(" · ") || "nothing")}</span>` +
    `<button class="tb-btn-text primary" data-act="import-go">Import</button><button class="tb-btn-text" data-act="reset-no">Cancel</button>`;
  box.querySelector("button").focus();
  return parts;
}
async function applyImport() {
  const prof = pendingImport;
  pendingImport = null;
  if (!prof) return;
  const notes = [];
  const skipped = prof.settings ? H.settings.setMany(prof.settings) : [];
  const body = {};
  if (prof.ui && typeof prof.ui === "object") body.ui = { set: prof.ui };
  if (prof.memory && typeof prof.memory === "object") body.memory = { set: prof.memory };
  if (prof.recent && typeof prof.recent === "object") body.recent = { files: prof.recent.files || {}, ...(Array.isArray(prof.recent.folders) ? { folders: prof.recent.folders } : {}) };
  if (prof.connect) notes.push("the connect profiles were left out (they are not part of the GUI profile)");
  try {
    await H.settings.flush();
    if (Object.keys(body).length) { await H.api("profile", { body }); H.profile.reload && await H.profile.reload(); }
    if (prof.docprefs) await H.api("profile/docprefs", { body: { kinds: prof.docprefs } });
  } catch (e) { H.toast(`Import: ${e.message}`, "error"); return; }
  pane.querySelector(".set-confirm").hidden = true;
  if (skipped.length) notes.unshift(`skipped ${skipped.join(", ")}`);
  H.toast(notes.length ? `Imported settings; ${notes.join("; ")}` : "Imported settings", notes.length ? "warning" : "success");
  render();
}

async function resetProfile() {
  const parts = picked();
  const sections = parts.filter((p) => p !== "docprefs");
  try {
    await H.settings.flush();
    await H.profile.flush();
    await H.api("profile/reset", { body: { sections, docprefs: parts.includes("docprefs") } });
  } catch (e) { H.toast(`Reset: ${e.message}`, "error"); return; }
  if (sections.includes("settings")) H.settings.resetAll();
  if (H.profile.reload) await H.profile.reload();
  if (parts.includes("docprefs")) H.docprefs.forgetAll();
  pane.querySelector(".set-confirm").hidden = true;
  H.toast(`Reset: ${parts.map((p) => RESET_LABEL[p]).join(", ") || "nothing"}`, "success");
  render();
}

/* Export, Import and Reset are browser downloads/uploads: the server never writes into --dir. */
export const close = () => dlg && dlg.open && dlg.close();
