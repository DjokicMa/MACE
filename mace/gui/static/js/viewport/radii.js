/* Atom radii for Settings › Viewport › Radii (B7, 1.1.6). The spheres are drawn at Atom scale × the
 * element's radius from the chosen set; app.js installs the set as 3Dmol's radius table (GLModel.vdwRadii,
 * which every view and the clipping and picking code read), so one switch reaches them all.
 *
 * - atomic: VESTA's atomic radii (its elements.ini). Atom scale 0.4 then draws VESTA's ball-and-stick
 *   rule, 40 % of the atomic radius (T9).
 * - covalent: Cordero et al. 2008, as ase.data.covalent_radii (geom/core.js COVALENT, the bond table's).
 * - ionic: Shannon (1976) effective ionic radii, six-coordinate, of each element's commonest ion; an
 *   element without one takes its covalent radius.
 * - vdw: 3Dmol's own van der Waals table (Bondi-type), as the classic views drew before 1.1.6; an element
 *   it lacks (every transition metal from Sc to Co, Y to Rh, La to Lu, Hf to Ir, the early actinides) takes
 *   Alvarez's radius (VDW_FILL, 1.1.6) instead of 3Dmol's default sphere. */
import { SYMBOLS, COVALENT } from "../geom/core.js";

export const ATOMIC = {
  H: 0.46, He: 1.22, Li: 1.57, Be: 1.12, B: 0.81, C: 0.77, N: 0.74, O: 0.74, F: 0.72, Ne: 1.6,
  Na: 1.91, Mg: 1.6, Al: 1.43, Si: 1.18, P: 1.1, S: 1.04, Cl: 0.99, Ar: 1.92, K: 2.35, Ca: 1.97,
  Sc: 1.64, Ti: 1.47, V: 1.35, Cr: 1.29, Mn: 1.37, Fe: 1.26, Co: 1.25, Ni: 1.25, Cu: 1.28, Zn: 1.37,
  Ga: 1.53, Ge: 1.22, As: 1.21, Se: 1.04, Br: 1.14, Kr: 1.98, Rb: 2.5, Sr: 2.15, Y: 1.82, Zr: 1.6,
  Nb: 1.47, Mo: 1.4, Tc: 1.35, Ru: 1.34, Rh: 1.34, Pd: 1.37, Ag: 1.44, Cd: 1.52, In: 1.67, Sn: 1.58,
  Sb: 1.41, Te: 1.37, I: 1.33, Xe: 2.18, Cs: 2.72, Ba: 2.24, La: 1.88, Ce: 1.82, Pr: 1.82, Nd: 1.82,
  Pm: 1.81, Sm: 1.81, Eu: 2.06, Gd: 1.79, Tb: 1.77, Dy: 1.77, Ho: 1.76, Er: 1.75, Tm: 1.74, Yb: 1.94,
  Lu: 1.72, Hf: 1.59, Ta: 1.47, W: 1.41, Re: 1.37, Os: 1.35, Ir: 1.36, Pt: 1.39, Au: 1.44, Hg: 1.55,
  Tl: 1.71, Pb: 1.75, Bi: 1.82, Po: 1.77, At: 1.45, Rn: 1.34, Fr: 2.7, Ra: 2.35, Ac: 2.03, Th: 1.8,
  Pa: 1.63, U: 1.56, Np: 1.56, Pu: 1.64, Am: 1.73,
};

// Shannon, CN 6 (the commonest ion): Li+ Na+ ... O2- S2- F- Cl- ...
export const IONIC = {
  H: 0.2, Li: 0.76, Be: 0.45, B: 0.27, C: 0.16, N: 1.46, O: 1.4, F: 1.33, Na: 1.02, Mg: 0.72, Al: 0.535,
  Si: 0.4, P: 0.38, S: 1.84, Cl: 1.81, K: 1.38, Ca: 1.0, Sc: 0.745, Ti: 0.605, V: 0.54, Cr: 0.615,
  Mn: 0.83, Fe: 0.78, Co: 0.745, Ni: 0.69, Cu: 0.73, Zn: 0.74, Ga: 0.62, Ge: 0.53, As: 0.46, Se: 1.98,
  Br: 1.96, Rb: 1.52, Sr: 1.18, Y: 0.9, Zr: 0.72, Nb: 0.64, Mo: 0.59, Ru: 0.62, Rh: 0.665, Pd: 0.86,
  Ag: 1.15, Cd: 0.95, In: 0.8, Sn: 0.69, Sb: 0.6, Te: 2.21, I: 2.2, Cs: 1.67, Ba: 1.35, La: 1.032,
  Ce: 1.01, Pr: 0.99, Nd: 0.983, Sm: 0.958, Eu: 0.947, Gd: 0.938, Tb: 0.923, Dy: 0.912, Ho: 0.901,
  Er: 0.89, Tm: 0.88, Yb: 0.868, Lu: 0.861, Hf: 0.71, Ta: 0.64, W: 0.6, Re: 0.63, Os: 0.63, Ir: 0.625,
  Pt: 0.8, Au: 1.37, Hg: 1.02, Tl: 1.5, Pb: 1.19, Bi: 1.03, Th: 0.94, U: 0.89,
};

/* The elements 3Dmol's van der Waals table lacks (Sc-Co, Y-Rh, La-Lu, Hf-Ir, Ac-Pa, Np-Es): Alvarez's
 * radii ("A cartography of the van der Waals territories", Dalton Trans. 42, 8617, 2013; ase.data.vdw_alvarez),
 * so a transition metal or a lanthanide is no longer 3Dmol's default 1.5 Å sphere (1.1.6). Pm and Fm onward have
 * no radius anywhere and keep the default. */
export const VDW_FILL = { Sc: 2.58, Ti: 2.46, V: 2.42, Cr: 2.45, Mn: 2.45, Fe: 2.44, Co: 2.4, Y: 2.75, Zr: 2.52, Nb: 2.56, Mo: 2.45, Tc: 2.44, Ru: 2.46, Rh: 2.44, La: 2.98, Ce: 2.88, Pr: 2.92, Nd: 2.95, Sm: 2.9, Eu: 2.87, Gd: 2.83, Tb: 2.79, Dy: 2.87, Ho: 2.81, Er: 2.83, Tm: 2.79, Yb: 2.8, Lu: 2.74, Hf: 2.63, Ta: 2.53, W: 2.57, Re: 2.49, Os: 2.48, Ir: 2.41, Ac: 2.8, Th: 2.93, Pa: 2.88, Np: 2.82, Pu: 2.81, Am: 2.83, Cm: 3.05, Bk: 3.4, Cf: 3.05, Es: 2.7 };

export const SETS = ["atomic", "covalent", "ionic", "vdw"];

/** {symbol: radius in Å} of a set; `vdw0` is 3Dmol's own table (the vdw set, and the fallback of atomic). */
export function table(set, vdw0) {
  if (set === "vdw" || !SETS.includes(set)) return { ...VDW_FILL, ...vdw0 };
  const out = {};
  SYMBOLS.forEach((sym, z) => {
    if (!z || sym === "X") return;
    const cov = COVALENT[z] || 0.75;
    out[sym] = set === "atomic" ? ATOMIC[sym] || cov : set === "ionic" ? IONIC[sym] || cov : cov;
  });
  return out;
}

/** Atom scale's factor for a set: with bonds, VESTA's rule (scale × radius) for atomic, covalent and ionic;
 *  × 0.7 for van der Waals (the 0.28 × vdW the classic views always drew); without bonds × 1.357 more. */
export const factor = (set, bonds) => (set === "vdw" ? 0.7 : 1) * (bonds ? 1 : 0.95 / 0.7);
