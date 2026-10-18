/* Charged systems (1.1.6): the charged-system switches of `mace convert` (cif2d12) and
 * `mace opt2d12` (--ion-charges, --net-charge, --site-states, --chemod, --fragment-bond-scale,
 * --spinlock, --spinlock-cycles), shared by Calculation Setup and Batch Setup. The list, labels
 * and hints come from the server (setup/schema: charge_flags). The GUI sends them as switches only:
 * what they mean is the CLI's to check, and its refusal is shown in its own words. */
export const SECTION = "Charged system";

const NUM = /^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/;

/** One control per switch. layout "setup": the Setup panel's rows; "batch": Batch Setup's fields
 * (no site states: they name atoms of one structure). */
export function rowsHtml(defs, flags, { esc, prefix, layout = "setup" }) {
  const f = flags || {};
  const batch = layout === "batch";
  return (defs || []).filter((d) => !batch || d.batch).map((d) => {
    const id = `${prefix}-${d.id}`;
    const set = d.id in f;
    const off = d.id === "fragment_bond_scale" && !("ion_charges" in f);
    const hint = off ? `Needs Find ions. ${d.hint}` : d.hint;
    const attrs = `id="${id}" data-charge="${esc(d.id)}" aria-describedby="${id}-h"`;
    let ctl;
    if (d.kind === "choice" && d.choices.length === 1) {
      ctl = `<input type="checkbox" role="switch" ${attrs} aria-checked="${set}"${set ? " checked" : ""}>`;
    } else if (d.kind === "choice") {
      ctl = `<select ${attrs}${batch ? ' class="field"' : ""}><option value=""${set ? "" : " selected"}>Not given (the CLI's default)</option>` +
        d.choices.map((c) => `<option value="${esc(c)}"${f[d.id] === c ? " selected" : ""}>${esc(c)}</option>`).join("") + "</select>";
    } else if (d.kind === "json") {
      ctl = `<textarea ${attrs} class="mono${batch ? " field" : ""}" rows="3" spellcheck="false" placeholder='[{"atom": 1, "shell_charges": [2, 0, 0, 0, 0]}]'>${esc(set ? f[d.id] : "")}</textarea>`;
    } else {
      ctl = `<input ${attrs} class="mono${batch ? " field" : ""}" inputmode="${d.kind === "int" ? "numeric" : "decimal"}" value="${esc(set ? String(f[d.id]) : "")}" ` +
        `placeholder="not given" autocomplete="off" spellcheck="false"${off ? " disabled" : ""}>`;
    }
    const sw = `<code class="cf-switch">${esc(d.switch)}</code>`;
    if (batch) {
      return `<div class="cf-field${set ? " changed" : ""}" data-charge-row="${esc(d.id)}"><label for="${id}">${esc(d.label)}</label>${ctl}` +
        `<span class="note" id="${id}-h">${sw} ${esc(hint)}</span></div>`;
    }
    return `<div class="cf-row${set ? " changed" : ""}" data-charge-row="${esc(d.id)}">
      <label for="${id}" class="su-l">${esc(d.label)}</label><div class="su-ctl">${ctl}</div>
      <div class="su-meta">${sw}<span class="su-h" id="${id}-h">${esc(hint)}</span></div></div>`;
  }).join("");
}

/** The value a control holds (undefined: the switch is not given). A number field keeps
 * text that is not a number, so the server names what is wrong with it. */
export function valueOf(def, el) {
  if (def.kind === "choice" && def.choices.length === 1) return el.checked ? def.choices[0] : undefined;
  const raw = String(el.value || "").trim();
  if (!raw) return undefined;
  if (def.kind === "number" || def.kind === "int") return NUM.test(raw) ? Number(raw) : raw;
  return raw;
}

/** flags with one switch set (or removed); the fragment bond scale goes with Find ions. */
export function withValue(flags, key, value) {
  const out = { ...(flags || {}) };
  if (value === undefined) delete out[key]; else out[key] = value;
  if (!("ion_charges" in out)) delete out.fragment_bond_scale;
  return out;
}

/** "--chemod yes --net-charge -1": the switches as a command line shows them. */
export function argsText(defs, flags) {
  const q = (v) => (/[\s"'[\]{}]/.test(String(v)) ? `'${String(v).replace(/'/g, "'\\''")}'` : String(v));
  return (defs || []).filter((d) => d.id in (flags || {})).map((d) => `${d.switch} ${q(flags[d.id])}`).join(" ");
}

/** Only the switches a batch takes. */
export function forBatch(defs, flags) {
  const ok = new Set((defs || []).filter((d) => d.batch).map((d) => d.id));
  return Object.fromEntries(Object.entries(flags || {}).filter(([k]) => ok.has(k)));
}
