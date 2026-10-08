/* Import from Clipboard (plan T9, B12): the pasted text in an editable box,
 * what it looks like ("Looks like POSCAR, 8 atoms"), and for a plain XYZ
 * whether to put it in a box (10 Å of vacuum) or keep it a molecule. Import
 * parses it on the server (/api/parse, never written into the folder) and
 * opens it as an untitled, modified structure. The detection here mirrors
 * structures.sniff(); the server decides. */
import { loadCss, panel } from "./lazy_util.js";

const NAMES = { cif: "CIF", poscar: "POSCAR", xyz: "XYZ", extxyz: "extXYZ", d12: "CRYSTAL input (.d12)" };
const floats = (line, n) => {
  const p = String(line || "").trim().split(/\s+/);
  return p.length >= n && p.slice(0, n).every((x) => x !== "" && Number.isFinite(Number(x)));
};

/** {fmt, atoms} for the text, or {fmt: null} when it looks like nothing MACE reads. */
export function detect(text) {
  const t = String(text || "");
  const lines = t.split(/\r?\n/).map((l) => l.replace(/\s+$/, ""));
  if (/^\s*data_/m.test(t) || t.includes("_cell_length_a")) {
    const loop = t.match(/_atom_site_fract_[xyz]/g);
    let n = 0;
    if (loop) {
      // the atom rows: after the _atom_site_ headers, until a blank line, loop_ or another tag
      const all = t.split(/\r?\n/);
      let k = all.findIndex((l) => /^\s*_atom_site_fract_/.test(l));
      while (k >= 0 && k < all.length && /^\s*_/.test(all[k])) k++;
      for (; k >= 0 && k < all.length; k++) {
        const l = all[k].trim();
        if (!l || l.startsWith("_") || l.startsWith("loop_") || l.startsWith("#") || l.startsWith("data_")) break;
        n++;
      }
    }
    return { fmt: "cif", atoms: n || null, unique: true };
  }
  if (lines.length >= 5 && floats(lines[1], 1) && [2, 3, 4].every((k) => floats(lines[k], 3))) {
    // VASP 5: element names on line 6, counts on line 7; VASP 4: counts on line 6
    const counts = floats(lines[5], 1) ? lines[5] : lines[6];
    const n = String(counts || "").trim().split(/\s+/).map(Number).filter(Number.isFinite).reduce((a, b) => a + b, 0);
    return { fmt: "poscar", atoms: n || null };
  }
  if (lines.length && /^\d+$/.test(lines[0].trim())) {
    return { fmt: lines.length > 1 && lines[1].includes('Lattice="') ? "extxyz" : "xyz", atoms: Number(lines[0].trim()) };
  }
  if (/^\s*(CRYSTAL|SLAB|POLYMER|MOLECULE)\s*$/m.test(t) && /^\s*END/m.test(t)) return { fmt: "d12", atoms: null };
  return { fmt: null, atoms: null };
}

export function describe(r) {
  if (!r.fmt) return "Doesn't look like a structure MACE reads (CIF, POSCAR, XYZ, extXYZ, d12)";
  const n = r.atoms ? `, ${r.atoms} ${r.unique ? "site" : "atom"}${r.atoms === 1 ? "" : "s"}` : "";
  return `Looks like ${NAMES[r.fmt]}${n}`;
}

/** open(H, text, onImport(text, fmt, box) -> Promise<doc|null>) */
export async function open(H, text, onImport) {
  await loadCss(import.meta.url, "../../css/dialogs.css");
  const body = document.createElement("div");
  body.className = "import-clip";
  body.innerHTML = `<label class="f" for="import-text">Structure text</label>
    <textarea id="import-text" class="field mono" rows="14" spellcheck="false" autocomplete="off"
      placeholder="Paste a POSCAR, CIF, XYZ or extXYZ here"></textarea>
    <p class="note" id="import-detect" role="status" aria-live="polite"></p>
    <div class="row" id="import-box-row" hidden><label class="tog"><input type="checkbox" id="import-box" checked>
      <span>Put it in a box (10 Å of vacuum); unchecked, it opens as a molecule (view-only)</span></label></div>
    <div class="err" role="alert"></div>`;
  const ta = body.querySelector("#import-text"), note = body.querySelector("#import-detect");
  const boxRow = body.querySelector("#import-box-row"), box = body.querySelector("#import-box"), err = body.querySelector(".err");
  ta.value = text || "";
  let cur = detect(ta.value);
  const d = panel(H, { id: "import-clipboard", title: "Import from Clipboard", body, wide: true,
    foot: `<button class="tb-btn-text" data-act="cancel">Cancel</button><button class="tb-btn-text primary" data-act="import">Import</button>` });
  const go = d.querySelector('[data-act="import"]');
  const update = () => {
    cur = detect(ta.value);
    note.textContent = ta.value.trim() ? describe(cur) : "The clipboard was empty or could not be read: paste the text here";
    boxRow.hidden = cur.fmt !== "xyz";
    go.disabled = !ta.value.trim() || !cur.fmt;
    err.textContent = "";
  };
  ta.addEventListener("input", update);
  update();
  d.querySelector('[data-act="cancel"]').addEventListener("click", () => d.close());
  go.addEventListener("click", async () => {
    if (go.disabled) return;
    go.disabled = true;
    try {
      const doc = await onImport(ta.value, cur.fmt, cur.fmt === "xyz" ? box.checked : true);
      if (doc) d.close();
      else err.textContent = "That text did not parse: see Problems";
    } finally {
      go.disabled = !cur.fmt;
    }
  });
  ta.focus();
  return d;
}
