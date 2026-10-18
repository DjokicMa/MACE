/* Export Element Legend… (1.1.6): a legend of the elements of one structure,
 * or the union over files, a folder, a Batch Setup selection or the open
 * documents, in the element palette, for figures. The server makes it with
 * the same function as `mace plotting --legend` (mace/utils/element_legend.py,
 * /api/legend), so the file is the one the CLI writes for the same choices.
 * The preview is that SVG; Export downloads SVG, PNG (DPI, transparent or on
 * a background) or a TSV/CSV/JSON table (symbol, name, Z, hex, RGB). */
import { loadCss, panel } from "./lazy_util.js";
import { isDarkTheme } from "../themes.js";

let H, dlg = null, scope = null, timer = 0, seq = 0, last = null;
const M = () => H.M();
const KEEP = ["format", "layout", "columns", "shape", "swatch", "font_size", "label", "order", "outline", "dpi"];

/** Text and background that suit the active theme: a dark theme, light text; Plot style Publication
 * (Settings › Appearance), black on white, as a figure. */
export function themeDefaults(theme, plotStyle) {
  if (plotStyle === "publication") return { text: "dark", background: "white" };
  return { text: isDarkTheme(theme) ? "light" : "dark", background: "none" };
}

function defaults() {
  const kept = H.profile.memory("elementLegend", null) || {};
  const base = { format: "svg", layout: "vertical", columns: 4, shape: "circle", swatch: 18, font_size: 14, label: "symbol",
    order: "z", outline: true, dpi: 300 };
  for (const k of KEEP) if (kept[k] !== undefined) base[k] = kept[k];
  return { ...base, ...themeDefaults(H.settings.get("theme"), H.settings.get("plots.style")) };
}

/** Open the dialog for scope {title, paths} (files and folders, relative to the served folder) or {title, symbols}. */
export async function open(handle, sc) {
  H = handle;
  await loadCss(import.meta.url, "../../css/dialogs.css");
  scope = sc;
  const o = defaults(), { esc } = H;
  const pal = M().schemeName(), draft = !!M().state.paletteDraft;
  const opt = (k, pairs, v) => pairs.map(([a, b]) => `<option value="${esc(a)}"${String(a) === String(v) ? " selected" : ""}>${esc(b)}</option>`).join("");
  const body = `<p class="lg-scope muted small" id="lg-scope">${esc(sc.title)}</p>
    <div class="lg-preview" id="lg-preview" aria-live="polite"><img id="lg-img" alt="Legend preview"></div>
    <p class="lg-msg small" id="lg-msg" role="status"></p>
    <div class="grid3">
      <label class="f">Palette<select class="field" data-k="palette">${draft ? `<option value="__draft" selected>Being edited (Element Colors)</option>` : ""}${opt("palette", M().paletteIds().map((k) => [k, M().paletteLabel(k)]), draft ? "" : pal)}</select></label>
      <label class="f">Format<select class="field" data-k="format">${opt("format", [["svg", "SVG (editable)"], ["png", "PNG"], ["tsv", "TSV table"], ["csv", "CSV table"], ["json", "JSON table"]], o.format)}</select></label>
      <label class="f">Order<select class="field" data-k="order">${opt("order", [["z", "Atomic number"], ["first", "First appearance"], ["alpha", "Alphabetical"]], o.order)}</select></label>
      <label class="f">Layout<select class="field" data-k="layout">${opt("layout", [["vertical", "Vertical"], ["horizontal", "Horizontal"], ["grid", "Grid"]], o.layout)}</select></label>
      <label class="f lg-cols">Columns<input class="field" type="number" min="1" max="30" step="1" data-k="columns" value="${o.columns}"></label>
      <label class="f">Label<select class="field" data-k="label">${opt("label", [["symbol", "Symbol"], ["name", "Full name"], ["both", "Both"]], o.label)}</select></label>
      <label class="f">Swatch<select class="field" data-k="shape">${opt("shape", [["circle", "Circle"], ["square", "Square"]], o.shape)}</select></label>
      <label class="f">Swatch size (px)<input class="field" type="number" min="4" max="200" step="1" data-k="swatch" value="${o.swatch}"></label>
      <label class="f">Font size (px)<input class="field" type="number" min="4" max="120" step="1" data-k="font_size" value="${o.font_size}"></label>
      <label class="f">Text<select class="field" data-k="text">${opt("text", [["dark", "Dark text"], ["light", "Light text"]], o.text)}</select></label>
      <label class="f">Background<select class="field" data-k="background">${opt("background", [["none", "Transparent"], ["white", "White"], ["black", "Black"]], o.background)}</select></label>
      <label class="f lg-dpi">PNG DPI<input class="field" type="number" min="30" max="2400" step="1" data-k="dpi" value="${o.dpi}"></label>
    </div>
    <label class="chk"><input type="checkbox" data-k="outline"${o.outline ? " checked" : ""}> A thin ring around each swatch</label>
    <label class="f">File name<input class="field" data-k="name" spellcheck="false" autocomplete="off" value="${esc(stem(sc))}.${o.format}"></label>`;
  dlg = panel(H, { id: "legend-dialog", title: "Export Element Legend", body, wide: true,
    foot: `<button class="tb-btn-text" data-v="cancel">Cancel</button><button class="tb-btn-text primary" data-v="export">Export</button>` });
  dlg.classList.add("lg-panel");
  dlg.oninput = dlg.onchange = (e) => { if (e.target.dataset.k && e.target.dataset.k !== "name") changed(); };
  dlg.querySelector('[data-v="cancel"]').onclick = () => dlg.close();
  dlg.querySelector('[data-v="export"]').onclick = () => exportNow();
  changed();
  return dlg;
}
export const isOpen = () => !!(dlg && dlg.open);
/** For tests: the last preview the server sent. */
export const lastPreview = () => last;

function stem(sc) {
  const t = (sc.stem || sc.title || "elements").replace(/\.[^./]+$/, "").replace(/[^\w.-]+/g, "_").replace(/^_+|_+$/g, "");
  return `${t || "elements"}_legend`;
}

/** The request for the dialog's choices, in format fmt. */
export function request(fmt) {
  const q = (k) => dlg.querySelector(`[data-k="${k}"]`);
  const num = (k) => Number(q(k).value);
  const options = { layout: q("layout").value, columns: num("columns"), shape: q("shape").value, swatch: num("swatch"),
    font_size: num("font_size"), label: q("label").value, order: q("order").value, text: q("text").value,
    background: q("background").value, outline: q("outline").checked, dpi: num("dpi") };
  const body = { format: fmt, options };
  if (scope.symbols) body.symbols = scope.symbols; else body.paths = scope.paths;
  if (q("palette").value === "__draft") { body.palette = M().schemeName(); body.colors = M().state.paletteDraft; }
  else body.palette = q("palette").value;
  return body;
}

function changed() {
  const q = (k) => dlg.querySelector(`[data-k="${k}"]`);
  dlg.querySelector(".lg-cols").hidden = q("layout").value !== "grid";
  dlg.querySelector(".lg-dpi").hidden = q("format").value !== "png";
  const nm = q("name"), ext = q("format").value;
  nm.value = nm.value.replace(/\.(svg|png|tsv|csv|json)$/i, "") + "." + ext;
  const o = request("svg").options;
  H.profile.remember("elementLegend", Object.fromEntries(KEEP.map((k) => [k, k === "format" ? ext : o[k]])));
  clearTimeout(timer);
  timer = setTimeout(preview, 120);
}

async function preview() {
  const mine = ++seq, msg = dlg.querySelector("#lg-msg");
  try {
    const r = await H.api("legend", { body: request("svg") });
    if (mine !== seq || !dlg) return;
    last = r;
    const img = dlg.querySelector("#lg-img"), old = img.src;
    img.src = URL.createObjectURL(new Blob([r.content], { type: "image/svg+xml" }));
    img.width = Math.round(r.width); img.height = Math.round(r.height);
    if (old.startsWith("blob:")) URL.revokeObjectURL(old);
    const bg = dlg.querySelector('[data-k="background"]').value;
    dlg.querySelector("#lg-preview").dataset.bg = bg === "none" ? (dlg.querySelector('[data-k="text"]').value === "light" ? "dark" : "light") : bg;
    msg.textContent = `${r.symbols.length} element${r.symbols.length === 1 ? "" : "s"}: ${r.symbols.join(" ")} · ${M().paletteLabel(r.palette)}` +
      (r.files ? ` · from ${r.files} file${r.files === 1 ? "" : "s"}` : "") + (r.skipped.length ? ` · ${r.skipped.length} could not be read` : "");
    msg.classList.remove("err");
    dlg.dataset.ready = String(mine);
  } catch (e) {
    if (mine !== seq || !dlg) return;
    msg.textContent = e.message; msg.classList.add("err");
  }
}

async function exportNow() {
  const fmt = dlg.querySelector('[data-k="format"]').value;
  let name = dlg.querySelector('[data-k="name"]').value.trim() || `${stem(scope)}.${fmt}`;
  if (!name.toLowerCase().endsWith(`.${fmt}`)) name += `.${fmt}`;
  try {
    const r = await H.api("legend", { body: request(fmt) });
    const type = { svg: "image/svg+xml", png: "image/png", tsv: "text/tab-separated-values", csv: "text/csv", json: "application/json" }[fmt];
    const url = r.encoding === "base64" ? `data:${type};base64,${r.content}` : URL.createObjectURL(new Blob([r.content], { type }));
    if (window.__maceExportHook) window.__maceExportHook(r.content, `legend-${fmt}`);
    const a = document.createElement("a");
    a.href = url; a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    if (url.startsWith("blob:")) setTimeout(() => URL.revokeObjectURL(url), 2000);
    H.toast(`Exported ${name} (${r.symbols.length} elements)`, "success");
    dlg.close();
  } catch (e) {
    const msg = dlg.querySelector("#lg-msg");
    msg.textContent = e.message; msg.classList.add("err");
  }
}

/** The legend of symbols as SVG text and its size, for an exported structure image's corner
 * (export_image.js): the active palette (or the one being edited), text that suits the background. */
export async function forImage(handle, symbols, { background = null, scale = 1 } = {}) {
  H = handle;
  const dark = background ? lum(background) < 0.4 : isDarkTheme(H.settings.get("theme"));
  const body = { format: "svg", symbols, palette: M().schemeName(),
    options: { layout: "vertical", label: "symbol", text: dark ? "light" : "dark", background: "none",
      swatch: Math.round(16 * scale), font_size: Math.round(13 * scale) } };
  if (M().state.paletteDraft) body.colors = M().state.paletteDraft;
  const r = await H.api("legend", { body });
  return { svg: r.content, width: r.width, height: r.height };
}
function lum(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec(String(hex).trim());
  if (!m) return 1;
  const c = [0, 2, 4].map((k) => parseInt(m[1].slice(k, k + 2), 16) / 255);
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
}
