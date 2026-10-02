"""Charged decks through the real commands: `mace convert` (cif2d12) and `mace ions`.

The decks written with the charge options must be the default deck plus the
charge records and nothing else, so each test strips the records again and
compares with the deck written without the options.
"""
import json
import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT
from ion_charges import read_deck_geometry, strip_charge_records
from ion_fixtures import cluster, ec, fsi, li_near, molecule_cif, molecule_deck, no3, p1_cif, shuffled

MACE_CLI = REPO_ROOT / "mace_cli"


def _env():
    env = dict(os.environ)
    for k in ("COLORTERM", "TERM", "NO_COLOR"):
        env.pop(k, None)
    return env


def _run(args, cwd, stdin=subprocess.DEVNULL):
    return subprocess.run([sys.executable, str(MACE_CLI)] + args, cwd=cwd, stdin=stdin,
                          capture_output=True, text=True, timeout=300, env=_env())


def li_ec2():
    solvent1, solvent2 = ec(), [(s, (x + 8.0, y, z)) for s, (x, y, z) in ec()]
    return shuffled(solvent1 + solvent2 + [li_near(solvent1, 1)], seed=11)


def _convert(tmp_path, name, cif_text, options, *flags):
    cifs = tmp_path / f"cifs_{len(flags)}"
    cifs.mkdir()
    (cifs / f"{name}.cif").write_text(cif_text)
    opts = tmp_path / "opts.json"
    opts.write_text(json.dumps(options))
    out = tmp_path / f"out_{len(flags)}"
    run = _run(["convert", "--no-banner", "--batch", "--options_file", str(opts), "--cif_dir", str(cifs),
                "--output_dir", str(out)] + list(flags), tmp_path)
    decks = list(out.glob("*.d12")) if out.exists() else []
    return run, (decks[0].read_text() if decks else None)


MOLECULE = {"calculation_type": "SP", "method": "DFT", "dft_functional": "HSESOL3C",
            "basis_set": "SOLDEF2MSVP", "basis_set_type": "INTERNAL", "dimensionality": "MOLECULE",
            "symmetry_handling": "P1", "write_only_unique": False,
            "tolerances": {"TOLINTEG": "7 7 7 7 14", "TOLDEE": 9}}


def test_cif2d12_molecule_with_ion_charges(tmp_path):
    pytest.importorskip("ase")
    atoms = li_ec2()
    cif = molecule_cif(atoms, name="li_ec2")
    run0, plain = _convert(tmp_path, "li_ec2", cif, MOLECULE)
    run, deck = _convert(tmp_path, "li_ec2", cif, MOLECULE, "--ion-charges", "auto")
    assert run.returncode == 0, run.stdout + run.stderr
    lines = deck.splitlines()
    i = lines.index("SOLDEF2MSVP")
    li = [l.split()[-1] for l in lines[4:4 + len(atoms)]].index("Li") + 1
    assert lines[i + 1:i + 4] == ["CHEMOD", "1", f"{li} 2.0 0.0 0.0 0.0 0.0"]
    assert lines[-3:] == ["DOPING", "-1", "END"]
    assert strip_charge_records(deck) == plain
    assert "Li+ (+1 each)" in run.stdout


def test_cif2d12_odd_count_refused_without_spin_then_written_with_it(tmp_path):
    pytest.importorskip("ase")
    cif = molecule_cif(li_ec2(), name="li_ec2")
    run, deck = _convert(tmp_path, "li_ec2", cif, MOLECULE, "--ion-charges", "auto", "--net-charge", "0")
    assert run.returncode != 0 and deck is None
    assert "95 electrons is odd" in run.stdout + run.stderr
    run, deck = _convert(tmp_path, "li_ec2", cif, MOLECULE, "--ion-charges", "auto", "--net-charge", "0",
                         "--spinlock", "1", "--spinlock-cycles", "30")
    assert run.returncode == 0, run.stdout + run.stderr
    lines = deck.splitlines()
    assert lines[lines.index("DFT") + 1] == "SPIN"
    assert lines[lines.index("SPINLOCK") + 1] == "1 30"
    assert "DOPING" not in lines


def test_cif2d12_periodic_p1_cell(tmp_path):
    pytest.importorskip("ase")
    a = 12.0
    atoms = [(s, (x + 6.0, y + 6.0, z + 6.0)) for s, (x, y, z) in no3()] + [("Li", (8.9, 6.0, 6.0))]
    cif = p1_cif((a, a, a, 90, 90, 90), [(s, tuple(v / a for v in p)) for s, p in atoms], name="lino3")
    options = dict(MOLECULE, dimensionality="CRYSTAL", dft_functional="PBE0", basis_set="POB-TZVP-REV2",
                   dft_grid="XLGRID")
    run0, plain = _convert(tmp_path, "lino3", cif, options)
    run, deck = _convert(tmp_path, "lino3", cif, options, "--ion-charges", "auto")
    assert run.returncode == 0, run.stdout + run.stderr
    lines = deck.splitlines()
    i = lines.index("POB-TZVP-REV2")
    assert lines[i + 1:i + 3] == ["CHEMOD", "4"]
    ps = sorted(r.split()[5] for r in lines[i + 3:i + 7] if len(r.split()) == 9)
    assert ps == ["4.333333", "4.333333", "4.333334"]
    assert "DOPING" not in lines and "CHARGED" not in lines
    assert strip_charge_records(deck) == plain


def test_cif2d12_without_flags_is_unchanged_by_the_new_options(tmp_path):
    pytest.importorskip("ase")
    cif = molecule_cif(li_ec2(), name="li_ec2")
    run0, plain = _convert(tmp_path, "li_ec2", cif, MOLECULE)
    run1, same = _convert(tmp_path, "li_ec2", cif, MOLECULE, "--chemod", "auto")
    assert run1.returncode == 0 and same == plain


# ------------------------------------------------------------------ mace ions

def test_mace_ions_on_a_shuffled_molecule_deck(tmp_path):
    anion = fsi()
    atoms = shuffled(cluster((anion, (0.0, 0.0, 0.0)), (ec(), (8.0, 0.0, 0.0))) + [li_near(anion, 2)], seed=4)
    deck = tmp_path / "cluster.d12"
    deck.write_text(molecule_deck(atoms))
    run = _run(["ions", str(deck), "--ion-charges", "auto", "-o", "out"], tmp_path)
    assert run.returncode == 0, run.stdout + run.stderr
    new = (tmp_path / "out" / "cluster.d12").read_text().splitlines()
    i = new.index("SOLDEF2MSVP")
    assert new[i + 1:i + 3] == ["CHEMOD", "5"]
    rows = new[i + 3:i + 8]
    syms = [s for s, _ in atoms]
    assert sorted(syms[int(r.split()[0]) - 1] for r in rows) == ["Li", "O", "O", "O", "O"]
    assert "DOPING" not in new
    assert strip_charge_records("\n".join(new) + "\n") == deck.read_text()


def test_mace_ions_refuses_an_odd_count_without_a_terminal(tmp_path):
    deck = tmp_path / "li_ec2.d12"
    deck.write_text(molecule_deck(li_ec2()))
    run = _run(["ions", str(deck), "--ion-charges", "auto", "--net-charge", "0", "-o", "out"], tmp_path)
    assert run.returncode == 1
    assert "NOT written" in run.stderr and "odd" in run.stderr
    assert not (tmp_path / "out" / "li_ec2.d12").exists()


def test_mace_ions_in_place_and_replace(tmp_path):
    deck = tmp_path / "li_ec2.d12"
    deck.write_text(molecule_deck(li_ec2()))
    assert _run(["ions", str(deck), "--net-charge", "1", "--in-place"], tmp_path).returncode == 0
    assert deck.read_text().splitlines()[-3:] == ["DOPING", "-1", "END"]
    again = _run(["ions", str(deck), "--net-charge", "1", "--in-place"], tmp_path)
    assert again.returncode == 1 and "already has DOPING" in again.stderr
    assert _run(["ions", str(deck), "--ion-charges", "auto", "--replace", "--in-place"], tmp_path).returncode == 0
    assert deck.read_text().count("DOPING") == 1 and "CHEMOD" in deck.read_text()


# ------------------------------------------------------------- deck reading

def test_read_slab_polymer_crystal_and_ecp_decks():
    slab = read_deck_geometry("t\nSLAB\n1\n4.0 5.0 90.0\n2\n3 0.5 0.0 1.5 Biso 1.0 Li\n8 0.0 0.5 -1.0\nEND\n")
    assert slab.periodic_dirs == (0, 1) and not slab.symmetric
    assert slab.xyz[0] == pytest.approx((2.0, 0.0, 1.5))
    assert slab.xyz[1] == pytest.approx((0.0, 2.5, -1.0))
    poly = read_deck_geometry("t\nPOLYMER\n1\n3.0\n1\n6 0.5 1.0 2.0\nEND\n")
    assert poly.periodic_dirs == (0,) and poly.xyz[0] == pytest.approx((1.5, 1.0, 2.0))
    cryst = read_deck_geometry("t\nCRYSTAL\n0 0 0\n1\n4.0 4.0 4.0 90 90 90\n1\n282 0.5 0.5 0.5\nEND\n")
    assert cryst.symbols == ["Pb"] and cryst.z_labels == [282]
    assert cryst.xyz[0] == pytest.approx((2.0, 2.0, 2.0))
    sym = read_deck_geometry("t\nCRYSTAL\n0 0 0\n225\n4.21\n2\n12 0 0 0\n8 .5 .5 .5\nEND\n")
    assert sym.symmetric and sym.cell is None


def test_geometry_editing_after_the_atoms_is_refused():
    from ion_charges import IonChargeError
    with pytest.raises(IonChargeError, match="SUPERCEL"):
        read_deck_geometry("t\nCRYSTAL\n0 0 0\n1\n4 4 4 90 90 90\n1\n3 0 0 0\nSUPERCEL\n2 0 0\n0 2 0\n0 0 2\nEND\n")


def test_mace_ions_symmetric_deck_takes_doping_only(tmp_path):
    deck = tmp_path / "mgo.d12"
    deck.write_text("mgo\nCRYSTAL\n0 0 0\n225\n4.21\n2\n12 0 0 0\n8 .5 .5 .5\nEND\nBASISSET\nPOB-TZVP-REV2\n"
                    "DFT\nPBE0\nENDDFT\nTOLINTEG\n7 7 7 7 14\nSHRINK\n8 8\nEND\n")
    bad = _run(["ions", str(deck), "--ion-charges", "auto", "-o", "out"], tmp_path)
    assert bad.returncode == 1 and "P1" in bad.stderr
    ok = _run(["ions", str(deck), "--net-charge", "2", "-o", "out"], tmp_path)
    assert ok.returncode == 0, ok.stderr
    assert (tmp_path / "out" / "mgo.d12").read_text().splitlines()[-3:] == ["DOPING", "-2", "END"]
    assert "parity was not checked" in ok.stderr
