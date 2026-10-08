/* Save As (spec §1.3, §2.6), loaded on first use: a folder inside the
 * served directory (the server refuses anything outside it), a file name
 * and a format: CIF with every atom, a symmetrized CIF (spglib's standard
 * cell), POSCAR or XYZ (extended, with the cell). Resolves {path, format,
 * cifMode} or null. Also the small confirmations saving needs. */
import { loadCss } from "./lazy_util.js";

const FORMATS = [["cif:p1", "CIF (every atom)", ".cif"], ["cif:symmetrized", "CIF, symmetrized (spglib)", ".cif"],
  ["poscar", "POSCAR (VASP)", ".vasp"], ["xyz", "XYZ (extended, with the cell)", ".xyz"]];

function modal(H, id, title, body, foot) {
  let d = document.getElementById(id);
  if (!d) { d = document.createElement("dialog"); d.id = id; document.body.appendChild(d); }
  d.className = "dlg narrow";
  d.innerHTML = `<div class="dlg-head"><h2></h2><button class="icon-btn sm" data-close aria-label="Close">${H.icon("x")}</button></div>
    <div class="dlg-body"></div><div class="dlg-foot">${foot}</div>`;
  d.querySelector("h2").textContent = title;
  const b = d.querySelector(".dlg-body");
  if (typeof body === "string") b.innerHTML = body; else b.appendChild(body);
  d.querySelector("[data-close]").addEventListener("click", () => d.close());
  return d;
}
/** A choice dialog: resolves the value of the button pressed (or null). */
export async function ask(H, { id, title, body, buttons }) {
  await loadCss(import.meta.url, "../../css/dialogs.css");
  H.overlayOpening();
  const opener = document.activeElement;
  return new Promise((resolve) => {
    let value = null;
    const d = modal(H, id, title, body, buttons.map(([v, t, primary]) =>
      `<button class="tb-btn-text${primary ? " primary" : ""}" data-v="${v}">${H.esc(t)}</button>`).join(""));
    d.querySelectorAll("[data-v]").forEach((b) => b.addEventListener("click", () => { value = b.dataset.v; d.close(); }));
    d.addEventListener("close", () => { if (opener && opener.focus) opener.focus(); resolve(value); }, { once: true });
    d.showModal();
    const primary = d.querySelector(".dlg-foot .primary");
    if (primary) primary.focus();
  });
}

export async function open(H, ed, defaultStem, preset = {}) {
  await loadCss(import.meta.url, "../../css/dialogs.css");
  H.overlayOpening();
  const opener = document.activeElement;
  const files = H.files();
  const folders = ["", ...new Set(files.map((f) => (f.path.includes("/") ? f.path.slice(0, f.path.lastIndexOf("/")) : "")).filter(Boolean))].sort();
  const want = preset.format || H.settings.get("saveFormat") || "cif";
  const fmt0 = want === "poscar" ? "poscar" : want === "xyz" ? "xyz" : preset.cifMode === "symmetrized" ? "cif:symmetrized" : "cif:p1";
  const dir0 = defaultStem.includes("/") ? defaultStem.slice(0, defaultStem.lastIndexOf("/")) : "";
  const name0 = defaultStem.split("/").pop();
  const body = document.createElement("div");
  body.innerHTML = `<label class="f">Folder<select class="field" data-k="dir" aria-label="Folder">${folders.map((f) =>
      `<option value="${H.esc(f)}"${f === dir0 ? " selected" : ""}>${H.esc(f ? f + "/" : "(the served folder)")}</option>`).join("")}</select></label>
    <label class="f">File name<input class="field" data-k="name" spellcheck="false" autocomplete="off" aria-label="File name"></label>
    <label class="f">Format<select class="field" data-k="fmt" aria-label="Format">${FORMATS.map(([v, t]) => `<option value="${v}"${v === fmt0 ? " selected" : ""}>${H.esc(t)}</option>`).join("")}</select></label>
    <p class="note">Inside ${H.esc(H.root())} only. MACE never writes anywhere else.</p><div class="err" role="alert"></div>`;
  const ext = (v) => FORMATS.find((f) => f[0] === v)[2];
  const nameEl = body.querySelector('[data-k="name"]'), fmtEl = body.querySelector('[data-k="fmt"]');
  nameEl.value = preset.name || name0 + ext(fmt0);
  fmtEl.addEventListener("change", () => {
    const base = nameEl.value.replace(/\.(cif|vasp|poscar|xyz)$/i, "");
    nameEl.value = base + ext(fmtEl.value);
  });
  return new Promise((resolve) => {
    let out = null;
    const d = modal(H, "save-as", "Save As", body,
      `<button class="tb-btn-text" data-v="cancel">Cancel</button><button class="tb-btn-text primary" data-v="save">Save</button>`);
    d.classList.add("save-dlg");
    const go = async () => {
      const name = nameEl.value.trim();
      if (!name || /[\\/]/.test(name) || name.startsWith(".")) { body.querySelector(".err").textContent = "A plain file name, not a path"; return; }
      const dir = body.querySelector('[data-k="dir"]').value, [format, cifMode] = fmtEl.value.split(":");
      const path = (dir ? dir + "/" : "") + name;
      // outputs, data files, databases, job scripts and queued decks are never written: say so here
      if (H.writeCheck) {
        try {
          const c = await H.writeCheck(path);
          if (c && c.protected) { body.querySelector(".err").textContent = c.protected; return; }
        } catch (e) { /* the save itself checks again */ }
      }
      out = { path, format, cifMode: cifMode || "p1" };
      d.close();
    };
    d.querySelector('[data-v="save"]').addEventListener("click", go);
    d.querySelector('[data-v="cancel"]').addEventListener("click", () => d.close());
    nameEl.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); go(); } });
    d.addEventListener("close", () => { if (opener && opener.focus) opener.focus(); resolve(out); }, { once: true });
    d.showModal();
    nameEl.focus();
    nameEl.setSelectionRange(0, nameEl.value.lastIndexOf("."));
  });
}
