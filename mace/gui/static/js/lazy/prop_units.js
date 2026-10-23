/* The Properties tab's display units (1.1.6): the one place the tab converts a
 * value to another unit, per atom or per formula unit. The server says what
 * each row is (mace/properties/units.py: its dimension, stored unit, basis,
 * whether it is extensive, the run's atoms and formula units) and sends the
 * catalog of units with their scales (CODATA 2018, mace/constants.py), so the
 * GUI and `mace analyze --list-properties` agree; this module only does the
 * arithmetic: value × scale(from) / scale(to) ÷ (atoms | formula units).
 * Display only: the database keeps its keys, values and units.
 *
 * A choice is "<unit id>|<norm>", norm one of total, atom, f.u. The state
 * ({preset, norm, over: {key: choice}}) is the tab's: the preset and the
 * normalization at the top set every row; a row's own choice (its Unit cell)
 * wins until the preset changes again. Normalize rescales every row and keeps
 * a row's own unit (setTop). */

export const NORMS = { total: "Total (per cell)", atom: "Per atom", "f.u.": "Per formula unit" };

function dimOf(row, cat) {
  return row && row.convertible && cat && cat.dimensions ? cat.dimensions[row.dim] || null : null;
}
function unitOf(dim, id) { return dim ? dim.units.find((u) => u.id === id) || null : null; }
function normOk(row, norm) {
  if (norm === "total") return true;
  if (!row.extensive) return false;
  return norm === "atom" ? row.atoms > 0 : norm === "f.u." ? row.formula_units > 0 : false;
}

/** A control at the top changed (``which``: preset or norm). A new preset sets every row
 * again (the rows' own units go); a new normalization applies to every row, those with
 * their own unit too, which keep that unit. */
export function setTop(st, which, value) {
  if (which === "preset") {
    st.preset = value;
    st.over = {};
  } else if (which === "norm") {
    st.norm = value;
    for (const [k, v] of Object.entries(st.over)) st.over[k] = `${String(v).split("|")[0]}|${value}`;
  }
  return st;
}

/** The unit label of a choice: eV, eV/atom, kJ/mol per f.u., meV/(cell·K). */
export function labelOf(row, cat, unitId, norm) {
  const u = unitOf(dimOf(row, cat), unitId);
  if (!u) return row.unit_text || "";
  const basis = norm === "total" ? row.basis : norm;
  return basis ? (u.per[basis] || `${u.id}/${basis}`) : u.label;
}

/** Every choice a row offers: [{group, choices: [{key, unit, norm, label}]}]; empty: no picker. */
export function choices(row, cat) {
  const dim = dimOf(row, cat);
  if (!dim) return [];
  const out = [];
  for (const norm of Object.keys(NORMS)) {
    if (!normOk(row, norm)) continue;
    let group = NORMS[norm];
    if (norm === "total" && !row.extensive) group = "";
    if (norm === "f.u." && row.formula) group = `Per formula unit (${row.formula}, Z = ${row.formula_units})`;
    out.push({ group, choices: dim.units.map((u) => ({ key: `${u.id}|${norm}`, unit: u.id, norm, label: labelOf(row, cat, u.id, norm) })) });
  }
  const n = out.reduce((a, g) => a + g.choices.length, 0);
  return n > 1 ? out : [];
}

/** The choice a row shows under the tab's state ({preset, norm, over}). */
export function chosen(row, cat, st) {
  const dim = dimOf(row, cat);
  const stored = { unit: row.unit_id, norm: "total" };
  if (!dim) return stored;
  const over = st.over && st.over[row.name];
  if (over) {
    const [unit, norm] = over.split("|");
    if (unitOf(dim, unit)) return { unit, norm: normOk(row, norm) ? norm : "total" };
  }
  const preset = (cat.presets || []).find((p) => p.id === st.preset);
  const pu = preset && (preset.units[row.preset_dim || row.dim] || preset.units[row.dim]);
  const unit = pu && unitOf(dim, pu) ? pu : row.unit_id;
  const norm = row.extensive && normOk(row, st.norm) ? st.norm : "total";
  return { unit, norm };
}

/** The value of a row in a choice: {value (a number, or the stored value), text, unit, stored}. */
export function shown(row, cat, ch) {
  const isStored = !ch || (ch.unit === row.unit_id && ch.norm === "total");
  if (isStored || !dimOf(row, cat)) return { value: row.value, text: row.display, unit: row.unit_text || "", stored: true };
  const dim = dimOf(row, cat);
  const from = unitOf(dim, row.unit_id), to = unitOf(dim, ch.unit);
  let v = Number(row.value) * from.scale / to.scale;
  if (ch.norm === "atom") v /= row.atoms;
  else if (ch.norm === "f.u.") v /= row.formula_units;
  // an energy to CRYSTAL's printed precision in the unit shown (units.energy_decimals), else its significant digits
  const text = row.full_precision && to.decimals != null ? v.toFixed(to.decimals) : gfmt(v, row.precision || 6);
  return { value: v, text, unit: labelOf(row, cat, ch.unit, ch.norm), stored: false };
}

/** A number to ``p`` significant digits, as Python's "{:.pg}" (the server's display.format_value). */
export function gfmt(v, p = 6) {
  if (!Number.isFinite(v)) return String(v);
  if (v === 0) return "0";
  p = Math.max(1, Math.round(p));
  const e = Number(v.toExponential(p - 1).split("e")[1]);
  const strip = (s) => (s.includes(".") ? s.replace(/0+$/, "").replace(/\.$/, "") : s);
  if (e < -4 || e >= p) {
    const [m, x] = v.toExponential(p - 1).split("e");
    const n = Math.abs(Number(x));
    return `${strip(m)}e${x[0] === "-" ? "-" : "+"}${n < 10 ? "0" : ""}${n}`;
  }
  return strip(v.toFixed(Math.max(0, p - 1 - e)));
}
