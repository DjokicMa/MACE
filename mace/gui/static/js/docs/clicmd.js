/* The command line that draws a plot as the GUI shows it (Copy CLI Command): mace plotting with the
 * plot's file, energy window, Fermi shift, energy zero, unit, a fort.25's domain and run, the Reference and
 * Labels menus' switches and style flags (plotpresets.js styleFlags,
 * the same as presets.py style_flags), and an edited k-path as a band_path.json written first; a transport
 * plot's quantity, temperature, component, x axis, κ_L and τ (mace plotting --transport). With a
 * mace-folder.toml (explicit), the values equal to mace plotting's defaults are written too: the
 * folder file's [plots] and [units] values would replace them. No DOM:
 * tests/test_gui_cli_command.py runs it in node and parses its flags with mace plotting's own parser. */
import { styleFlags } from "../plotpresets.js";

const MODE = { band: "--band", dos: "--dos", bandos: "--band-dos", phbandos: "--phonon-band-dos", ir: "--ir", raman: "--raman", transport: "--transport" };
// a transport quantity's mace plotting --quantity name (mace.utils.transport_plot.CLI_NAMES)
export const TRANSPORT_CLI = { SEEBECK: "seebeck", SIGMA: "sigma", SIGMAS: "sigmas", KAPPA: "kappa", TDF: "tdf", pf: "pf", kappa_e: "kappa-e",
  zt: "zt", lorenz: "lorenz", sigma_tau: "sigma-tau", pf_tau: "pf-tau" };
/** A word as the shell reads it: bare when safe, else in single quotes. */
export const shq = (s) => (/^[A-Za-z0-9_./=:+,@%-]+$/.test(String(s)) ? String(s) : `'${String(s).replace(/'/g, "'\\''")}'`);
const num = (v) => String(Number(Number(v).toPrecision(12)));

/** {kind: band|dos|bandos|phbandos (a phonon dispersion + DOS)|ir|raman|transport, file, emin, emax, shift, reference (auto|fermi|vacuum: fermi and vacuum are
 *  named, as auto takes the vacuum scale when the run's POTC files are there), unit, look (STYLE_OPTIONS names),
 *  kpath, menus ({domain, run, fermi_line, k_labels, spin_labels}, as plot.js menuArgs), explicit} -> text. */
export function cliCommand({ kind, file, emin = null, emax = null, shift = 0, reference = "auto", unit = "eV", look = {}, kpath = null,
  menus = {}, explicit = false, transport = null, logDos = false, partner = null }) {
  const pair = kind === "bandos" || kind === "phbandos", energy = kind === "band" || kind === "dos" || pair;
  const words = ["mace", "plotting", MODE[kind] || "--band", shq(file)];
  // a band + DOS pair from two folders (Add DOS…, Add Band…; Add Phonon DOS…, Add Phonon Bands…): the other half,
  // relative to the file's folder
  if (pair && partner) words.push("--partner", shq(partner));
  if (kind === "transport" && transport) {
    // {quantity, component, temperature (a value, "all" or null for the TDF), x, kappa_l, tau_fs}: the quantity,
    // temperature and x axis always named, the rest when not the defaults
    const t = transport;
    words.push(`--quantity=${TRANSPORT_CLI[t.quantity] || t.quantity}`);
    if (t.component && t.component !== "avg") words.push(`--component=${t.component}`);
    if (t.temperature !== null && t.temperature !== undefined) words.push(`--temperature=${t.temperature === "all" ? "all" : num(t.temperature)}`);
    if (t.x) words.push(`--x-axis=${t.x}`);
    if (t.kappa_l) words.push(`--kappa-lattice=${shq(Number.isFinite(Number(t.kappa_l)) ? num(t.kappa_l) : t.kappa_l)}`);
    if (t.tau_fs !== null && t.tau_fs !== undefined && t.tau_fs !== "") words.push(`--tau-fs=${num(t.tau_fs)}`);
  }
  if (energy) {
    if (Number.isFinite(Number(emin)) && emin !== null) words.push(`--e-lower=${num(emin)}`);
    if (Number.isFinite(Number(emax)) && emax !== null) words.push(`--e-upper=${num(emax)}`);
    if (Number(shift) || explicit) words.push(`--fermi-shift=${num(shift || 0)}`);
    if (reference === "vacuum" || reference === "fermi" || explicit) words.push(`--energy-reference=${reference === "vacuum" ? "vacuum" : "fermi"}`);
    if ((unit && unit !== "eV") || explicit) words.push(`--units=${shq(unit || "eV")}`);
    if (menus.domain && menus.domain !== "auto") words.push(`--f25-domain=${shq(menus.domain)}`);
    if (Number(menus.run) >= 1) words.push(`--f25-run=${Math.floor(Number(menus.run))}`);
    for (const k of ["fermi_line", "k_labels", "spin_labels"]) if (menus[k] === 0 || menus[k] === false) words.push(`--no-${k.replace("_", "-")}`);
    // Log Y (a DOS plot) or Log DOS (band + DOS): the DOS axis only
    if (logDos && (kind === "dos" || pair)) words.push("--log-dos");
  }
  // a spectrum takes the plot style only; band and DOS every style option
  const flags = energy ? styleFlags(look, explicit) : styleFlags({ style: look.style }, explicit).slice(0, 1);
  words.push(...flags.map(shq));
  if (energy && kind !== "dos" && kpath) words.push("--band-path", "band_path.json");
  const cmd = words.join(" ");
  if (!(energy && kind !== "dos" && kpath)) return cmd;
  return `cat > band_path.json <<'EOF'\n${JSON.stringify({ band_path: kpath })}\nEOF\n${cmd}`;
}
