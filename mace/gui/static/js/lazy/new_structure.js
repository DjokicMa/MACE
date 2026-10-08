/* New Structure… (Alt+N, spec §2.6, plan T12), loaded on first use: the
 * lattice (a b c α β γ), a space group by number or Hermann-Mauguin symbol
 * with its setting, and the asymmetric unit, one "El x y z" line per site.
 * The server builds the cell (edit/new, ase.spacegroup.crystal) and it
 * opens as an untitled, modified structure document: Save goes to Save As. */
import { loadCss } from "./lazy_util.js";
import * as profile from "../profile.js";

const KEY = "mace.gui.newStructure.v1";
const DEFAULTS = { name: "new_cell", a: 4.212, b: 4.212, c: 4.212, alpha: 90, beta: 90, gamma: 90, sg: "225", setting: 1,
  sites: "Mg 0 0 0\nO 0.5 0.5 0.5" };

function last() {
  const mem = profile.memory("newStructure", null);     // the last form, in the profile
  if (mem && typeof mem === "object") return { ...DEFAULTS, ...mem };
  try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(KEY) || "{}") }; } catch (e) { return { ...DEFAULTS }; }
}

/** The sites text as [{element, frac}]; throws with the line that is wrong. */
export function parseSites(text) {
  return text.split("\n").map((l) => l.trim()).filter((l) => l && !l.startsWith("#")).map((l) => {
    const p = l.split(/[\s,]+/);
    const frac = p.slice(1, 4).map(Number);
    if (p.length < 4 || !/^[A-Z][a-z]?$/.test(p[0]) || frac.some((x) => !Number.isFinite(x))) throw new Error(`Site line "${l}": write El x y z`);
    return { element: p[0], frac };
  });
}

export async function open(H, S) {
  await loadCss(import.meta.url, "../../css/dialogs.css");
  H.overlayOpening();
  const v = last(), opener = document.activeElement;
  let d = document.getElementById("new-structure");
  if (!d) { d = document.createElement("dialog"); d.id = "new-structure"; document.body.appendChild(d); }
  d.className = "dlg narrow new-dlg";
  const num = (k, label, unit) => `<label class="f">${label}<span class="unit-in"><input class="field" type="number" step="any" data-k="${k}" value="${H.esc(v[k])}" aria-label="${label}"><span>${unit}</span></span></label>`;
  d.innerHTML = `<div class="dlg-head"><h2>New Structure</h2><button class="icon-btn sm" data-close aria-label="Close">${H.icon("x")}</button></div>
    <div class="dlg-body">
      <label class="f">Name<input class="field" data-k="name" value="${H.esc(v.name)}" spellcheck="false" autocomplete="off"></label>
      <div class="grid3">${num("a", "a", "Å")}${num("b", "b", "Å")}${num("c", "c", "Å")}${num("alpha", "α", "°")}${num("beta", "β", "°")}${num("gamma", "γ", "°")}</div>
      <div class="grid2"><label class="f">Space group<input class="field" data-k="sg" value="${H.esc(v.sg)}" placeholder="225 or F m -3 m (blank: P1)" spellcheck="false"></label>
        <label class="f">Setting<select class="field" data-k="setting"><option value="1"${Number(v.setting) === 1 ? " selected" : ""}>1</option><option value="2"${Number(v.setting) === 2 ? " selected" : ""}>2</option></select></label></div>
      <label class="f">Sites (asymmetric unit, fractional)<textarea class="field" rows="5" data-k="sites" spellcheck="false">${H.esc(v.sites)}</textarea></label>
      <p class="note">One “El x y z” line per site; the space group makes the rest.</p>
      <div class="err" role="alert"></div>
    </div>
    <div class="dlg-foot"><button class="tb-btn-text" data-v="cancel">Cancel</button><button class="tb-btn-text primary" data-v="create">Create</button></div>`;
  const q = (k) => d.querySelector(`[data-k="${k}"]`), err = d.querySelector(".err");
  const close = () => d.close();
  d.querySelector("[data-close]").addEventListener("click", close);
  d.querySelector('[data-v="cancel"]').addEventListener("click", close);
  const create = async () => {
    err.textContent = "";
    const vals = Object.fromEntries(["name", "a", "b", "c", "alpha", "beta", "gamma", "sg", "setting", "sites"].map((k) => [k, q(k).value]));
    let sites;
    try { sites = parseSites(vals.sites); } catch (e) { err.textContent = e.message; return; }
    if (!sites.length) { err.textContent = "Add at least one site"; return; }
    const name = vals.name.trim() || "new_cell";
    if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,60}$/.test(name)) { err.textContent = "Name: letters, digits, _ and - only"; return; }
    const cellpar = ["a", "b", "c", "alpha", "beta", "gamma"].map((k) => Number(vals[k]));
    if (cellpar.some((x) => !Number.isFinite(x)) || cellpar.slice(0, 3).some((x) => x <= 0)) { err.textContent = "a, b and c must be positive numbers"; return; }
    let r;
    try {
      r = await H.api("edit/new", { body: { name, cellpar, spacegroup: vals.sg.trim() || null, setting: Number(vals.setting), sites,
        locked: true } });
    } catch (e) { err.textContent = e.message; return; }
    profile.remember("newStructure", vals);
    d.close();
    await S.openUntitled(r, name);
    H.toast(`New structure ${name}: ${r.symbols.length} atoms`, "success");
  };
  d.querySelector('[data-v="create"]').addEventListener("click", create);
  d.addEventListener("keydown", (e) => { if (e.key === "Enter" && e.target.tagName !== "TEXTAREA" && e.target.tagName !== "SELECT") { e.preventDefault(); create(); } });
  d.addEventListener("close", () => { if (opener && opener.focus && opener.isConnected) opener.focus(); }, { once: true });
  d.showModal();
  q("name").focus();
  q("name").select();
  return d;
}
