"""The manual CLI routes Calculation Setup is held to (plan T3, spike S6).

Each row of the matrix names an input, the CLI mode and calculation type,
what a user types at the CLI's questions (the *manual route*, pinned here
with the question each answer goes to, so a changed prompt fails with
"prompt changed", never with a wrong deck) and the change made in the GUI
(field answers, or settings written into the JSON by hand).

Everything here is public data: the demo set (examples/gui), small
outputs shipped in tests/data, and a placeholder wavefunction the tests
write themselves. The corpus rows (C*) copy files from test/ at run time
and skip without it.
"""
import gzip
import json
import os
import shutil
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import guiharness as H

REPO = H.REPO_ROOT
DATA = REPO / "tests" / "data"
ENTER = []          # every question answered with Enter (the CLI's defaults)
YES = [("Use these exact settings for the new calculation", "y")]      # opt2d12: the parent's settings

# --- the manual answers, question by question (spike S6 transcripts) --------------
_N_HEAD = [("use these default settings? [Y/n]", "n"), ("default symmetry settings? [Y/n]", ""),
           ("Select system dimensionality", "")]
_N_OPT = [("Select calculation type", ""), ("Enter your choice", ""), ("Select convergence level", ""),
          ("MAXTRADIUS", "")]
_N_DFT = [("Select method type", ""), ("Select functional category", ""), ("Select Hybrid functional", ""),
          ("Add D3 dispersion correction", ""), ("Enter your choice", ""), ("Enter your choice", ""),
          ("Enter your choice", ""), ("Select tolerance level", "")]
_N_ADV_TAIL = [("Use level shifting", ""), ("MAXCYCLE [", ""), ("FMIXING [", ""), ("Select method (1-2)", ""),
               ("Enable PPAN", ""), ("Modify memory settings", ""), ("Use HISTDIIS", ""), ("HISTDIIS size", "")]


def _with(pairs, prompt, answer):
    """``pairs`` with the answer to the first question containing ``prompt`` replaced."""
    out, done = [], False
    for p, a in pairs:
        if not done and prompt in p:
            out.append((p, answer))
            done = True
        else:
            out.append((p, a))
    assert done, prompt
    return out


MANUAL = {
    "pbe0": _N_HEAD + _N_OPT + _with(_with(_N_DFT, "Select functional category", "3"), "Select Hybrid functional", "4")
    + [("default advanced settings", "")],
    "sp": _N_HEAD + [("Select calculation type", "1")] + _N_DFT + [("default advanced settings", "")],
    "freq": _N_HEAD + [("Select calculation type", "3"), ("Select type (1-2)", ""), ("Select template", ""),
                       ("Select method (0-2)", ""), ("Calculate IR intensities", ""),
                       ("Configure advanced frequency settings", "")] + _N_DFT + [("default advanced settings", "")],
    "dzvp": [("use these default settings? [Y/n]", ""), ("Enter your choice", ""), ("Enter your choice", "6")],
    "spin_off": _N_HEAD + _N_OPT + _N_DFT + [("default advanced settings", "n"), ("Use spin-polarized calculation", "n"),
                                             ("Use Fermi smearing", "")] + _N_ADV_TAIL,
    "smear": _N_HEAD + _N_OPT + _N_DFT + [("default advanced settings", "n"), ("Use spin-polarized calculation", ""),
                                          ("SPINLOCK value", ""), ("Use Fermi smearing", "y"), ("Smearing width options", "3"),
                                          ("Enter smearing width in Hartree", "0.005"), ("Use level shifting", ""),
                                          ("MAXCYCLE [", ""), ("FMIXING [", "30")] + _N_ADV_TAIL[3:],
    "hse06_follow": [("Use these exact settings for the new calculation", "n"), ("Enter your choice", ""),
                     ("Select convergence level", ""), ("MAXTRADIUS", ""), ("Select method type", ""),
                     ("Select functional category", "3"), ("Select Hybrid functional", "7"),
                     ("Add D3 dispersion correction", ""), ("Keep the basis set from the original D12", "")],
    "fmix40": [("Use these exact settings for the new calculation", "n"), ("Select method type", ""),
               ("Select functional category", ""), ("Select Hybrid functional", ""), ("Add D3 dispersion correction", ""),
               ("Enter your choice", ""), ("Enter your choice", ""), ("Enter your choice", ""),
               ("Select SCF convergence level", ""), ("Use spin-polarized calculation", ""), ("SPINLOCK value", ""),
               ("Use Fermi smearing", ""), ("Use level shifting", ""), ("MAXCYCLE [", ""), ("FMIXING [", "40")],
    "band_vectors": [("Band path definition", ""), ("Band path format", "2")],
    "doss_total": [("DOS projection type", "1")],
}


class Row:
    def __init__(self, id, src, mode, calc, manual, answers=None, patches=None, spin=None, keep_parent=False,
                 only=None, band=False, note=""):
        self.id, self.src, self.mode, self.calc, self.manual = id, src, mode, calc, manual
        self.answers, self.patches = answers or {}, patches or {}
        self.spin, self.keep_parent, self.only, self.band, self.note = spin, keep_parent, only, band, note

    def __repr__(self):
        return f"row{self.id}"


MATRIX = [
    Row(1, "structures/MgO.cif", "cif2d12", "OPT", ENTER, spin=True),
    Row(2, "structures/MgO.cif", "cif2d12", "OPT", MANUAL["pbe0"], {"functional": "PBE0"}, spin=True),
    Row(3, "structures/LiF.cif", "cif2d12", "SP", MANUAL["sp"], spin=True),
    Row(4, "structures/Si.cif", "cif2d12", "FREQ", MANUAL["freq"], spin=True),
    Row(5, "structures/hBN.cif", "cif2d12", "OPT", MANUAL["dzvp"], {"basis_set": "POB-DZVP-REV2"}, spin=True),
    Row(6, "structures/diamond.cif", "cif2d12", "OPT", MANUAL["spin_off"], {"spin": False}, spin=False),
    Row(8, "outputs/Si_sp.out", "opt2d12", "OPT", YES, keep_parent=True),
    Row(9, "outputs/Si_sp.out", "opt2d12", "SP", YES, keep_parent=True),
    Row(10, "outputs/Si_sp.out", "opt2d12", "FREQ", YES, keep_parent=True),
    Row(11, "outputs/Si_sp.out", "opt2d12", "OPT", MANUAL["hse06_follow"], {"functional": "HSE06"}, only="functional"),
    Row(12, "outputs/Ag2Br3_opt_unfinished.out", "opt2d12", "OPT", YES, keep_parent=True),
    Row(13, "outputs/graphene_slab_sp.out", "opt2d12", "SP", YES, keep_parent=True),
    Row(14, "data/opt_restart/tqb_pto.out", "opt2d12", "FREQ", YES, keep_parent=True),
    Row(15, "outputs/Si_sp.out", "opt2d3", "BAND", ENTER, band=True),
    Row(16, "outputs/Si_sp.out", "opt2d3", "BAND", MANUAL["band_vectors"], {"band_format": "K-point vectors"}, band=True),
    Row(17, "outputs/graphene_slab_sp.out", "opt2d3", "BAND", ENTER, band=True),
    Row(18, "data/label_band/lih_p1bar.out", "opt2d3", "BAND", ENTER, band=True),
    Row(19, "outputs/Si_sp.out", "opt2d3", "DOSS", ENTER),
    Row(20, "data/opt_restart/tqb_pto.out", "opt2d3", "DOSS", MANUAL["doss_total"], {"doss_projection": "Total DOS only"}),
    Row(21, "outputs/Ag2Br3_opt_unfinished.out", "opt2d3", "BAND", ENTER, band=True),
    Row(22, "structures/MgO.cif", "cif2d12", "OPT", MANUAL["smear"],
        {"smearing": True, "smear_width": "0.005", "fmixing": "30"}, spin=True),
    Row(23, "data/rg/rg_q0.out", "opt2d12", "SP", MANUAL["fmix40"], {"fmixing": "40"}, only="fmixing"),
    Row(26, "messy/Si_sp.out", "opt2d3", "BAND", ENTER, band=True),
]


def served_root(tmp: Path) -> Path:
    """The folder the server shows for the matrix: the demo set, the small
    outputs from tests/data, and a placeholder wavefunction beside each
    .out a properties run starts from (written here, never shipped)."""
    root = H.examples_copy(tmp)
    for sub, names in (("opt_restart", ("tqb_pto.out", "tqb_pto.d12")),
                       ("label_band", ("lih_p1bar.out", "lih_p1bar.d12"))):
        (root / "data" / sub).mkdir(parents=True, exist_ok=True)
        for n in names:
            shutil.copy2(DATA / sub / n, root / "data" / sub / n)
    (root / "data" / "rg").mkdir(parents=True)
    (root / "data" / "rg" / "rg_q0.out").write_bytes(gzip.decompress((DATA / "supercell_parents" / "rg_q0.out.gz").read_bytes()))
    shutil.copy2(DATA / "supercell_parents" / "rg_q0.d12", root / "data" / "rg" / "rg_q0.d12")
    for out in ("outputs/Si_sp", "outputs/graphene_slab_sp", "outputs/Ag2Br3_opt_unfinished", "data/opt_restart/tqb_pto",
                "data/label_band/lih_p1bar"):
        (root / (out + ".f9")).write_text("wavefunction placeholder\n")
    messy(root / "messy")
    return root


def messy(d: Path) -> Path:
    """A folder as users have them (plan B2): two .d12, a sibling _opt.out, fort.9 beside the _sp.f9."""
    d.mkdir(parents=True, exist_ok=True)
    src = H.EXAMPLES / "outputs"
    shutil.copy2(src / "Si_sp.out", d / "Si_sp.out")
    shutil.copy2(src / "Si_sp.out", d / "Si_opt.out")
    shutil.copy2(src / "Si_sp.d12", d / "Si_sp.d12")
    (d / "Si_extra.d12").write_text((src / "Si_sp.d12").read_text().replace("SHRINK\n8 8", "SHRINK\n6 6"))
    (d / "fort.9").write_text("fort.9 placeholder\n")
    (d / "Si_sp.f9").write_text("Si_sp.f9 placeholder\n")
    return d


def env():
    e = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "PYTHONPATH", "MACE_HOME")}
    e.update(NO_COLOR="1", MACE_NO_BANNER="1", PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    return e


def run_cli_manual(args, answers, cwd, timeout=300):
    """The manual route: the real CLI with the answers typed on stdin (then
    Enter for anything after them). Each answer's question must appear, in
    order, before it is used: a changed prompt fails here."""
    stdin = "".join(a + "\n" for _, a in answers) + "\n" * 60
    r = subprocess.run([sys.executable, str(H.MACE_CLI), "--no-banner", *args], cwd=str(cwd), input=stdin,
                       capture_output=True, text=True, timeout=timeout, env=env())
    out = r.stdout + r.stderr
    pos = 0
    for prompt, _ in answers:
        k = out.find(prompt, pos)
        assert k >= 0, f"prompt changed: {prompt!r} not found after position {pos} in the CLI's questions:\n{out[pos:pos + 3000]}"
        pos = k + len(prompt)
    return r


def run_printed(command_argv, cwd, timeout=300):
    """The command Setup printed, unchanged ("mace" is mace_cli)."""
    assert command_argv[0] == "mace"
    r = subprocess.run([sys.executable, str(H.MACE_CLI), "--no-banner", *command_argv[1:]], cwd=str(cwd),
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout, env=env())
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    return r


def printed_command_argv(text, root=None):
    import shlex
    return shlex.split(text)


def typed(obj):
    """A JSON value with its types kept apart: 0 and 0.0, 30 and 30.0 differ."""
    if isinstance(obj, dict):
        return {k: typed(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [typed(v) for v in obj]
    return (type(obj).__name__, obj)


def strict_load(text):
    return json.loads(text, parse_float=Decimal, parse_int=int)


def strict_diff(a, b, path=""):
    """The keys whose (type, value) differ between two JSON values."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}{k} (only in {'second' if k not in a else 'first'})")
            else:
                out += strict_diff(a[k], b[k], f"{path}{k}.")
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            out += strict_diff(x, y, f"{path}{i}.")
        return out
    if type(a) is not type(b) or a != b:
        out.append(f"{path.rstrip('.')}: {a!r} != {b!r}")
    return out


def keyword_lines(deck):
    return [x.strip() for x in deck.splitlines() if x.strip()]
