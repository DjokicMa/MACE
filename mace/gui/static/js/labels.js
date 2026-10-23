/* The text of every figure MACE draws: one entry per symbol and unit, written in plotly's markup.
 * mace/utils/plot_labels.py is its twin (mace plotting labels a figure as the GUI does;
 * tests/test_plot_labels.py holds the two to the same tables). No DOM here: node runs it for that test.
 *
 * The markup: <sub>, <sup> and <br>, a true minus (U+2212), and μ, Å, ° as characters. plotly draws it as
 * it is; `plain` turns a label into text where nothing draws markup (CSV/TSV headers, file names, a
 * textContent readout, a menu's <option>), `html` into HTML for innerHTML (everything else escaped), and
 * `svgText` into the tspans of a hand-drawn SVG <text>. Units go in parentheses after the quantity
 * (`withUnit`); a compound unit with negative powers (W m⁻¹ K⁻¹); a DOS counts states per energy unit. */

export const MINUS = "−";

/** a symbol with its sub- or superscript */
export const SYMBOLS = {
  E_F: "E<sub>F</sub>",
  E_vac: "E<sub>vac</sub>",
  E_ref: "E<sub>ref</sub>",
  E_min: "E<sub>min</sub>",
  E_final: "E<sub>final</sub>",
  E_g: "E<sub>g</sub>",
  kappa_0: "κ<sub>0</sub>",
  kappa_e: "κ<sub>e</sub>",
  kappa_L: "κ<sub>L</sub>",
  ZT_e: "ZT<sub>e</sub>",
  L_0: "L<sub>0</sub>",
  S2: "S<sup>2</sup>",
  C_V: "C<sub>V</sub>",
  e_1: "e<sub>1</sub>",
  e_2: "e<sub>2</sub>",
  log10: "log<sub>10</sub>",
  psi2: "|ψ|<sup>2</sup>",
};

/** a unit by its plain name (the menus' and the files' spelling) -> the label drawn */
export const UNITS = {
  eV: "eV", meV: "meV", Ha: "Ha", "kJ/mol": "kJ/mol", "kcal/mol": "kcal/mol", THz: "THz",
  "cm-1": `cm<sup>${MINUS}1</sup>`,
  "cm-3": `cm<sup>${MINUS}3</sup>`,
  A: "Å",
  A3: "Å<sup>3</sup>",
  "A-1": `Å<sup>${MINUS}1</sup>`,
  deg: "°",
  K: "K",
  fs: "fs",
  "nm-2": `nm<sup>${MINUS}2</sup>`,
  "km/mol": `km mol<sup>${MINUS}1</sup>`,
  "J/(K mol)": `J K<sup>${MINUS}1</sup> mol<sup>${MINUS}1</sup>`,
  "e/bohr3": `e bohr<sup>${MINUS}3</sup>`,
  "bohr-3": `bohr<sup>${MINUS}3</sup>`,
  "bohr-3/2": `bohr<sup>${MINUS}3/2</sup>`,
  "Ha/e": `Ha e<sup>${MINUS}1</sup>`,
  "uV/K": `µV K<sup>${MINUS}1</sup>`,
  "S/m": `S m<sup>${MINUS}1</sup>`,
  "A/(m K)": `A m<sup>${MINUS}1</sup> K<sup>${MINUS}1</sup>`,
  "W/(m K)": `W m<sup>${MINUS}1</sup> K<sup>${MINUS}1</sup>`,
  "uW/(cm K2)": `µW cm<sup>${MINUS}1</sup> K<sup>${MINUS}2</sup>`,
  "1e-8 W Ohm/K2": `10<sup>${MINUS}8</sup> W Ω K<sup>${MINUS}2</sup>`,
  "1e18 1/(Ohm m s)": `10<sup>18</sup> Ω<sup>${MINUS}1</sup> m<sup>${MINUS}1</sup> s<sup>${MINUS}1</sup>`,
  "1e11 W/(m K2 s)": `10<sup>11</sup> W m<sup>${MINUS}1</sup> K<sup>${MINUS}2</sup> s<sup>${MINUS}1</sup>`,
  "1/hbar2 eV fs/A": `ħ<sup>${MINUS}2</sup> eV fs Å<sup>${MINUS}1</sup>`,
};

/** A unit's label from its plain name ("cm-1" -> cm<sup>−1</sup>); another text as it is. */
export const unit = (name) => UNITS[name || ""] ?? (name || "");
/** "Name (unit)", the unit by its plain name or as markup; the name alone without one. */
export const withUnit = (name, u) => (u ? `${name} (${unit(u)})` : name);

/** " + 0.50 eV" for a Fermi shift (toFixed(2)), "" without one */
export const shiftText = (shift) => (shift ? ` ${shift > 0 ? "+" : MINUS} ${Math.abs(shift).toFixed(2)} eV` : "");
/** The energy axis of a band or DOS plot: E − E_F, E − E_vac on the vacuum scale, and with a Fermi shift s
 *  (the data move by −s) E − (E_F + s). */
export const energyTitle = (ref, shift) => {
  const zero = ref === "vacuum" ? SYMBOLS.E_vac : SYMBOLS.E_F;
  return shift ? `E ${MINUS} (${zero}${shiftText(shift)})` : `E ${MINUS} ${zero}`;
};
/** The dotted Fermi line's legend entry: E_F, or E_F + s. */
export const fermiKey = (shift) => `${SYMBOLS.E_F}${shiftText(shift)}`;

/* ------------------------------------------------------------ the markup for other targets */
const TAG = /<(sub|sup)>([\s\S]*?)<\/\1>/g;
const SUB = Object.fromEntries([..."0123456789+-−=()e"].map((c, i) => [c, "₀₁₂₃₄₅₆₇₈₉₊₋₋₌₍₎ₑ"[i]]));
const SUP = Object.fromEntries([..."0123456789+-−=()"].map((c, i) => [c, "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁻⁼⁽⁾"[i]]));
const ENT = { "&lt;": "<", "&gt;": ">", "&amp;": "&", "&quot;": '"', "&#39;": "'" };
const strip = (s) => s.replace(/<[^>]*>/g, "").replace(/&(lt|gt|amp|quot|#39);/g, (e) => ENT[e]);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

/** A label as text: a sub- or superscript in Unicode where it has one (κₑ, L₀, cm⁻¹, Å³), else E_F, E_min,
 *  x^(−3/2); <br> a space, other tags dropped (plot_labels.plain). */
export function plain(s) {
  const t = String(s ?? "").replace(/<br\s*\/?>/g, " ").replace(TAG, (_, tag, b) => {
    const body = strip(b), table = tag === "sub" ? SUB : SUP;
    if (body && [...body].every((c) => c in table)) return [...body].map((c) => table[c]).join("");
    const mark = tag === "sub" ? "_" : "^";
    return body.length <= 1 || tag === "sub" ? `${mark}${body}` : `${mark}(${body})`;
  });
  return strip(t);
}

/** A label for innerHTML: its <sub>, <sup> and <br> kept, everything else escaped. */
export const html = (s) => esc(s ?? "").replace(/&lt;(\/?)(sub|sup|br)&gt;/g, "<$1$2>");

/** A label in a hand-drawn SVG <text>: each sub- or superscript a smaller tspan moved by dy (and moved back
 *  after it), everything else escaped. dy, not baseline-shift: every browser draws it. */
export function svgText(s) {
  return String(s ?? "").replace(/<br\s*\/?>/g, " ").split(/(<(?:sub|sup)>[\s\S]*?<\/(?:sub|sup)>)/).map((part) => {
    const m = /^<(sub|sup)>([\s\S]*?)<\/\1>$/.exec(part);
    if (!m) return esc(strip(part));
    // the tspan's em is its own (75 %) size: 0.4em down is 0.3em of the text's, which the next one moves back
    const [down, back] = m[1] === "sub" ? ["0.4em", "-0.3em"] : ["-0.6em", "0.45em"];
    return `<tspan dy="${down}" font-size="75%">${esc(strip(m[2]))}</tspan><tspan dy="${back}">​</tspan>`;
  }).join("");
}
