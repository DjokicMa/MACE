/* The DOM-free part of plotkit.js: the unit tables and the two legend placements. plotkit.js
 * re-exports them; docs/bandfig.js (the band/DOS figure, which node runs for the parity test with
 * mace/plotting/plotly_bands.py) imports them from here. */
export const HARTREE_EV = 27.211386245988;          // CODATA 2018
export const AVOGADRO_EV_KJMOL = 96.48533212331;    // 1 eV per particle in kJ/mol
export const EV_KCALMOL = 23.060547830619;          // 1 eV per particle in kcal/mol
export const CM_PER_MEV = 8.065543937;              // 1 meV = 8.0655 cm⁻¹
export const CM_PER_THZ = 33.35640952;

/** value in the unit = value in eV × factor */
export const ENERGY_UNITS = { eV: 1, meV: 1000, Ha: 1 / HARTREE_EV, "kJ/mol": AVOGADRO_EV_KJMOL, "kcal/mol": EV_KCALMOL };
/** value in the unit = value in cm⁻¹ × factor */
export const WAVENUMBER_UNITS = { "cm-1": 1, meV: 1 / CM_PER_MEV, THz: 1 / CM_PER_THZ };
export const UNIT_LABEL = { eV: "eV", meV: "meV", Ha: "Ha", "kJ/mol": "kJ/mol", "kcal/mol": "kcal/mol", "cm-1": "cm⁻¹", THz: "THz" };
export const PER_LABEL = { cell: "", atom: "/atom", fu: "/f.u." };

export function legendTop(more = {}) {
  return { orientation: "h", x: 0, xanchor: "left", y: 1, yanchor: "bottom", font: { size: 11 }, ...more };
}
export function legendSide(more = {}) {
  return { orientation: "v", x: 1, xanchor: "left", y: 1, yanchor: "top", font: { size: 11 }, tracegroupgap: 0, ...more };
}
