/* The command line that draws a plot as the GUI shows it (Copy CLI Command): mace plotting with the
 * plot's file, energy window, Fermi shift, energy zero, unit and style flags (plotpresets.js styleFlags,
 * the same as presets.py style_flags), and an edited k-path as a band_path.json written first. With a
 * mace-folder.toml (explicit), the values equal to mace plotting's defaults are written too: the
 * folder file's [plots] and [units] values would replace them. No DOM:
 * tests/test_gui_cli_command.py runs it in node and parses its flags with mace plotting's own parser. */
import { styleFlags } from "../plotpresets.js";

const MODE = { band: "--band", dos: "--dos", bandos: "--band-dos", ir: "--ir", raman: "--raman" };
/** A word as the shell reads it: bare when safe, else in single quotes. */
export const shq = (s) => (/^[A-Za-z0-9_./=:+,@%-]+$/.test(String(s)) ? String(s) : `'${String(s).replace(/'/g, "'\\''")}'`);
const num = (v) => String(Number(Number(v).toPrecision(12)));

/** {kind: band|dos|bandos|ir|raman, file, emin, emax, shift, reference, unit, look (STYLE_OPTIONS names), kpath,
 *  explicit} -> text. */
export function cliCommand({ kind, file, emin = null, emax = null, shift = 0, reference = "fermi", unit = "eV", look = {}, kpath = null,
  explicit = false }) {
  const energy = kind === "band" || kind === "dos" || kind === "bandos";
  const words = ["mace", "plotting", MODE[kind] || "--band", shq(file)];
  if (energy) {
    if (Number.isFinite(Number(emin)) && emin !== null) words.push(`--e-lower=${num(emin)}`);
    if (Number.isFinite(Number(emax)) && emax !== null) words.push(`--e-upper=${num(emax)}`);
    if (Number(shift) || explicit) words.push(`--fermi-shift=${num(shift || 0)}`);
    if (reference === "vacuum" || explicit) words.push(`--energy-reference=${reference === "vacuum" ? "vacuum" : "fermi"}`);
    if ((unit && unit !== "eV") || explicit) words.push(`--units=${shq(unit || "eV")}`);
  }
  // a spectrum takes the plot style only; band and DOS every style option
  const flags = energy ? styleFlags(look, explicit) : styleFlags({ style: look.style }, explicit).slice(0, 1);
  words.push(...flags.map(shq));
  if (energy && kind !== "dos" && kpath) words.push("--band-path", "band_path.json");
  const cmd = words.join(" ");
  if (!(energy && kind !== "dos" && kpath)) return cmd;
  return `cat > band_path.json <<'EOF'\n${JSON.stringify({ band_path: kpath })}\nEOF\n${cmd}`;
}
