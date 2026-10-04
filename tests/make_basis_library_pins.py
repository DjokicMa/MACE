#!/usr/bin/env python3
"""Regenerate tests/data/basis_library_pins.json.

The pins fix what MACE reads and writes for every bundled external basis
record: the bytes of each record, what read_basis_file returns (and warns) for
every element of both external families, the external coverage check, and the
decks cif2d12 / d12_writer / opt2d12 write from those records. They were taken
from the per-element basis directories before those were folded into one
library file per family, so the library must reproduce them exactly.

    python tests/make_basis_library_pins.py           # rewrite the pins
    python tests/make_basis_library_pins.py --check   # compare, rewrite nothing

Deck pins need ase (NewCifToD12 imports it). Run with the interpreter that has
it; test_basis_library_pins.py skips those pins where ase is missing.
"""
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
D12_DIR = REPO_ROOT / "Crystal_d12"
PINS = Path(__file__).resolve().parent / "data" / "basis_library_pins.json"
SAMPLE = Path(__file__).resolve().parent / "data" / "opt_restart"

for p in (str(REPO_ROOT), str(D12_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

FAMILIES = ("full.basis.triplezeta", "full.basis.doublezeta", "stuttgart")
EXTERNAL_FAMILIES = ("full.basis.triplezeta", "full.basis.doublezeta")
# Elements of the multi-element deck: ECP (Ag, Pb, I, La, U) and
# all-electron (Br, C, O) records in one basis block.
MIXED = ("Ag", "Br", "Pb", "I", "C", "O", "La", "U")


def sha(data):
    if isinstance(data, str):
        data = data.encode("ascii")
    return hashlib.sha256(data).hexdigest()


def family_path(family):
    import mace_config

    return str(mace_config.BASIS_SETS_DIR / family) + "/"


def record_bytes(family):
    """{key: bytes} for every record of a bundled family: from the old
    per-element directory when it is there, else from the library."""
    directory = REPO_ROOT / "Crystal_d12" / "basis_sets" / family
    if directory.is_dir():
        return {p.name: p.read_bytes() for p in sorted(directory.iterdir())
                if p.is_file() and p.name != "__init__.py"}
    import basis_library

    return {key: body for key, body in basis_library.iter_records(family)}


def element_of_key(key):
    """Atomic number a record key stands for (200 + Z for ECP records)."""
    n = int(key)
    return n - 200 if n > 200 else n


def _normalize(text, family):
    """Replace the absolute basis path in a message with a placeholder so the
    pin does not depend on where the checkout lives."""
    return text.replace(family_path(family), "<BASIS>/").replace(
        family_path(family).rstrip("/"), "<BASIS>")


def pin_files():
    return {fam: {k: sha(b) for k, b in sorted(record_bytes(fam).items())}
            for fam in FAMILIES}


def pin_read_basis_file():
    from d12_constants import read_basis_file

    out = {}
    for fam in EXTERNAL_FAMILIES:
        path = family_path(fam)
        for z in range(1, 100):
            for kind in ("EXTERNAL", "INTERNAL"):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    text = read_basis_file(path, z, kind)
                out[f"{fam}/{z}/{kind}"] = {
                    "sha256": sha(text),
                    "empty": text == "",
                    "warning": _normalize(buf.getvalue(), fam),
                }
    return out


def pin_compatibility():
    from d12_constants import check_basis_set_compatibility

    return {f"{fam}/{z}": list(check_basis_set_compatibility(family_path(fam), [z], "EXTERNAL"))
            for fam in EXTERNAL_FAMILIES for z in range(1, 101)}


def _cif(symbols, a=12.0):
    """A P1 cubic cell with one atom per symbol on a diagonal."""
    rows = "\n".join(
        f"  {s}  {s}{i}  {0.1 * i:.6f}  {0.07 * i:.6f}  {0.05 * i:.6f}  1"
        for i, s in enumerate(symbols))
    return f"""data_pin
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   {a}
_cell_length_b   {a}
_cell_length_c   {a}
_cell_angle_alpha   90.0
_cell_angle_beta   90.0
_cell_angle_gamma   90.0
_symmetry_Int_Tables_number   1
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
{rows}
"""


def _cif_options(family):
    from d12_constants import DEFAULT_TOLERANCES

    return dict(
        dimensionality="CRYSTAL", calculation_type="SP", basis_set_type="EXTERNAL",
        basis_set=family_path(family), method="DFT", dft_functional="PBE",
        is_spin_polarized=False, tolerances=dict(DEFAULT_TOLERANCES),
        scf_method="DIIS", symmetry_handling="P1",
    )


def _assert_no_leak(name, text):
    """Pinned deck text must not carry the checkout path or the tmp dir."""
    for leak in (str(REPO_ROOT), tempfile.gettempdir()):
        assert leak not in text, f"{name}: deck text contains {leak!r}"


def pin_cif_decks(workdir):
    """cif2d12 decks: one single-atom cell per element of each external family
    plus one mixed-element cell. Keyed by output filename, so the filename
    (which carries the basis family name) is pinned as well."""
    import NewCifToD12
    from d12_constants import ATOMIC_NUMBER_TO_SYMBOL

    out = {}
    for fam in EXTERNAL_FAMILIES:
        cifs = Path(workdir) / f"cifs_{fam}"
        decks = Path(workdir) / f"decks_{fam}"
        cifs.mkdir()
        decks.mkdir()
        zs = sorted({element_of_key(k) for k in record_bytes(fam)})
        for z in zs:
            sym = ATOMIC_NUMBER_TO_SYMBOL[z]
            (cifs / f"Z{z:03d}_{sym}.cif").write_text(_cif([sym]))
        (cifs / "mixed.cif").write_text(_cif(MIXED))
        cwd = os.getcwd()
        os.chdir(workdir)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                NewCifToD12.process_cifs(str(cifs), _cif_options(fam), str(decks),
                                         interactive=False)
        finally:
            os.chdir(cwd)
        for deck in sorted(decks.iterdir()):
            text = deck.read_text()
            _assert_no_leak(deck.name, text)
            out[f"cif2d12/{deck.name}"] = sha(text)
    return out


def pin_writer_blocks():
    """d12_writer.write_basis_block for the mixed set and each family."""
    import d12_writer
    from d12_constants import SYMBOL_TO_NUMBER

    zs = [SYMBOL_TO_NUMBER[s] for s in MIXED]
    out = {}
    for fam in EXTERNAL_FAMILIES:
        buf = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()):
            d12_writer.write_basis_block(
                buf, {"basis_set_type": "EXTERNAL", "basis_set": family_path(fam)},
                {"elements": zs})
        _assert_no_leak("write_basis_block", buf.getvalue())
        out[f"d12_writer/{fam}"] = sha(buf.getvalue())
    return out


def pin_opt2d12_deck(workdir):
    """One opt2d12 SP deck from the in-repo Ag2Br3 OPT sample, with the
    bundled TZ family chosen as an external basis (not the parent's records)."""
    work = Path(workdir) / "opt2d12"
    work.mkdir()
    for suffix in (".out", ".d12"):
        (work / f"tqc_agbr{suffix}").write_bytes((SAMPLE / f"tqc_agbr{suffix}").read_bytes())
    tz = family_path("full.basis.triplezeta")
    (work / "opts.json").write_text(json.dumps({
        "basis_set_type": "EXTERNAL", "basis_set_path": tz, "basis_set": tz,
        "use_original_external_basis": False}))
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR")}
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "mace_cli"), "opt2d12", "--out-file", "tqc_agbr.out",
         "--d12-file", "tqc_agbr.d12", "--calc-type", "SP", "--non-interactive",
         "--config-file", "opts.json", "--output-dir", "out"],
        cwd=work, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300, env=env)
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
    decks = sorted((work / "out").glob("*.d12"))
    assert len(decks) == 1, decks
    text = decks[0].read_text()
    # The config's basis was used, not the parent's own records: the bundled
    # Ag record is written as stored, with its leading spaces, where the
    # parent's records are written back stripped.
    assert "\n  12.567714 255.054771 0\n" in text, "opt2d12 kept the parent's basis"
    _assert_no_leak(decks[0].name, text)
    return {f"opt2d12/{decks[0].name}": sha(text)}


def ase_available():
    try:
        import ase  # noqa: F401
        return True
    except ImportError:
        return False


def build():
    pins = {
        "files": pin_files(),
        "read_basis_file": pin_read_basis_file(),
        "compatibility": pin_compatibility(),
    }
    with tempfile.TemporaryDirectory() as tmp:
        decks = {}
        decks.update(pin_writer_blocks())
        decks.update(pin_opt2d12_deck(tmp))
        if not ase_available():
            raise SystemExit("ase is needed to pin the cif2d12 decks")
        decks.update(pin_cif_decks(tmp))
    pins["decks"] = decks
    return pins


def main():
    pins = build()
    text = json.dumps(pins, indent=1, sort_keys=True) + "\n"
    if "--check" in sys.argv:
        same = PINS.read_text() == text
        print("pins match" if same else "pins DIFFER")
        return 0 if same else 1
    PINS.write_text(text)
    print(f"wrote {PINS} ({len(pins['decks'])} decks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
