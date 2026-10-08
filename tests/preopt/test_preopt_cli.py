"""`mace preopt` through the real mace_cli subprocess (A14, A13a, A17, A8):
options, outputs, exit codes, the d12 guard, the CIF in the input's own
group, provenance, help, and the `mace convert` warning for a marked CIF."""
import json
import os
import re
import shutil
import socket
import subprocess
import sys

import pytest

import preopt_fixtures as F
from preopt_fixtures import preopt_sandbox  # noqa: E402,F401  (autouse: temp root, XDG dirs)

pytest.importorskip("numpy")
pytest.importorskip("ase")
pytest.importorskip("spglib")

MACE_CLI = F.REPO / "mace_cli"


def mace(*args, cwd, env=None, timeout=300):
    e = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    e.update(env or {})
    r = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", *args], env=e, cwd=str(cwd),
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


@pytest.fixture
def work(tmp_path):
    d = tmp_path / "work"
    d.mkdir()
    for n in ("NaCl.cif", "MgO.cif", "Si.cif"):
        shutil.copy(F.EXAMPLES / n, d / n)
    return d


def test_springs_default_writes_cif_json_and_follow_up(work):
    rc, out, err = mace("preopt", "Si.cif", cwd=work)
    assert rc == 0, err
    assert out.startswith("Ran: Springs (rough spring model), converged in")
    assert "Run a CRYSTAL OPT from it before any SP, FREQ, BAND or DOSS" in out and "calculation type OPT" in out
    cif = (work / "Si_preopt.cif").read_text()
    assert "# mace-preopt: method=springs model=- version=1" in cif and "Starting geometry only" in cif
    assert "_symmetry_Int_Tables_number 227" in cif
    prov = json.loads((work / "Si_preopt.json").read_text())
    assert prov["input"] == "Si.cif" and len(prov["input_sha256"]) == 64 and prov["output_cif"]["mode"] == "input_group"
    text = json.dumps(prov)
    assert str(work) not in text and socket.gethostname() not in text
    rc, out, err = mace("preopt", "Si.cif", cwd=work)
    assert rc == 1 and "pass --force" in err
    rc, out, err = mace("preopt", "Si.cif", "--force", "--json", "-o", "out", cwd=work)
    assert rc == 0 and json.loads(out)["method_ran"] == "springs" and (work / "out" / "Si_preopt.cif").exists()


@pytest.mark.parametrize("argv", [("Si.cif", "--method", "bogus"), ("Si.cif", "--method", "mlip:omat-24"),
                                  ("Si.cif", "--fmax", "-1"), ("Si.cif", "--device", "gpu0"), ()])
def test_usage_errors_exit_2(work, argv):
    rc, out, err = mace("preopt", *argv, cwd=work)
    assert rc == 2, (out, err)


def test_missing_xtb_writes_springs_and_exits_3(work, monkeypatch):
    rc, out, err = mace("preopt", "NaCl.cif", "--method", "gfnff", cwd=work, env={"PATH": "/usr/bin:/bin"})
    assert rc == 3 and "the spring result was written instead" in err
    assert "GFN-FF refused (Not found: xtb was not found on PATH); kept the spring result" in out
    rc, out, err = mace("preopt", "NaCl.cif", "--method", "gfnff", "--no-fallback", "--force", cwd=work,
                        env={"PATH": "/usr/bin:/bin"})
    assert rc == 1 and "nothing was written" in err
    rc, out, err = mace("preopt", "NaCl.cif", "--method", "auto", "--force", cwd=work, env={"PATH": "/usr/bin:/bin"})
    assert rc == 0 and out.startswith("Ran: Auto: ML acesuit-mace-mpa-0-medium refused (not installed")


def test_nothing_to_clean_is_exit_1(work):
    from ase.build import bulk
    bulk("Cu", "fcc", a=3.61, cubic=True).write(str(work / "Cu.cif"))
    rc, out, err = mace("preopt", "Cu.cif", cwd=work)
    assert rc == 1 and "nothing to clean" in err and not (work / "Cu_preopt.cif").exists()


MGO_DECK = """MgO test deck
CRYSTAL
0 0 0
225
4.21
2
12 0.0 0.0 0.0
8 0.5 0.5 0.5
{edits}END
12 3
0 0 1 2.0 1.0
 1.0 1.0
0 1 1 8.0 1.0
 0.5 1.0 1.0
0 1 1 0.0 1.0
 0.2 1.0 1.0
8 2
0 0 1 2.0 1.0
 1.0 1.0
0 1 1 6.0 1.0
 0.4 1.0 1.0
99 0
{basis_extra}END
DFT
PBE0
END
SHRINK
8 8
TOLDEE
9
{scf_extra}END
"""


def deck(path, edits="", basis_extra="", scf_extra=""):
    path.write_text(MGO_DECK.format(edits=edits, basis_extra=basis_extra, scf_extra=scf_extra))
    return path


@pytest.mark.parametrize("method", ["springs", "gfnff", "mlip:chgnet-0.3.0", "auto"])
def test_defect_deck_is_refused_for_every_method(work, method):
    deck(work / "def.d12", edits="SUPERCEL\n2 0 0\n0 2 0\n0 0 2\nATOMREMO\n1\n1\n", scf_extra="CHARGED\n")
    rc, out, err = mace("preopt", "def.d12", "--method", method, "--charge", "0", cwd=work)
    assert rc == 1, (out, err)
    assert "def.d12 builds its geometry with SUPERCEL, ATOMREMO: mace preopt reads only the asymmetric unit" in err
    assert "Pre-optimize the expanded structure instead: open the .out of a run of this deck" in err
    assert not (work / "def_preopt.cif").exists()


def test_every_geometry_editing_keyword_is_refused(work):
    from mace.preopt import deckscan
    for kw in deckscan.GEOMETRY_EDITS:
        if kw == "GHOSTS":
            deck(work / "k.d12", basis_extra="GHOSTS\n1\n1\n")
        else:
            deck(work / "k.d12", edits=f"{kw}\n1\n")
        assert deckscan.scan((work / "k.d12").read_text())["geometry_edits"] == [kw], kw
    rc, out, err = mace("preopt", "k.d12", cwd=work)          # the last one (GHOSTS) through the CLI
    assert rc == 1 and "builds its geometry with GHOSTS" in err
    for kind in ("SLAB", "POLYMER", "MOLECULE", "EXTERNAL"):
        text = MGO_DECK.format(edits="", basis_extra="", scf_extra="").replace("CRYSTAL\n", f"{kind}\n", 1)
        assert deckscan.scan(text)["geometry_edits"] == [kind]
    plain = deckscan.scan(MGO_DECK.format(edits="", basis_extra="", scf_extra="CHEMOD\n1\n1 2.0\n"))
    assert plain["geometry_edits"] == [] and plain["charge_records"] == ["CHEMOD"]


@pytest.mark.parametrize("method", ["gfnff", "mlip:chgnet-0.3.0", "auto"])
def test_charged_deck_needs_a_charge(work, method):
    deck(work / "q.d12", scf_extra="CHARGED\n")
    rc, out, err = mace("preopt", "q.d12", "--method", method, cwd=work, env={"PATH": "/usr/bin:/bin"})
    assert rc == 1 and "this deck is charged (CHARGED found): pass --charge Q" in err
    rc, out, err = mace("preopt", "q.d12", "--method", method, "--charge", "1", cwd=work, env={"PATH": "/usr/bin:/bin"})
    assert rc in (0, 3), err
    assert json.loads((work / "q_preopt.json").read_text())["charge"] == 1
    assert "Use the options you saved for q.d12" in out


def test_charged_deck_runs_with_springs(work):
    deck(work / "q.d12", scf_extra="CHARGED\n")
    rc, out, err = mace("preopt", "q.d12", cwd=work)
    assert rc == 0, err
    assert "_symmetry_Int_Tables_number 225" in (work / "q_preopt.cif").read_text()


def test_p1_input_stays_p1_with_a_warning(work):
    from ase.io import write
    a = F.rattled_mgo()
    write(str(work / "r.cif"), a, format="cif")
    rc, out, err = mace("preopt", "r.cif", cwd=work)
    assert rc == 0, err
    assert "wrote a P1 CIF (the input is P1): cif2d12 will build a P1 deck" in out
    assert "_symmetry_Int_Tables_number 1" in (work / "r_preopt.cif").read_text()
    assert "symmetry_handling SPGLIB" not in out


def test_p1_input_symmetrized_first_says_how_to_keep_the_group(work):
    """--symmetrize on a P1 input raises the reference group, but the CIF can
    only be P1 (the input had no group to keep): the warning must not leave
    the user with a P1 deck while the follow-up says CRYSTAL keeps Fm-3m."""
    from ase.io import write
    write(str(work / "r.cif"), F.rattled_mgo(), format="cif")
    rc, out, err = mace("preopt", "r.cif", "--symmetrize", "0.3", cwd=work)
    assert rc == 0, err
    assert "raised from P1 to Fm-3m by symmetrize-first" in out
    assert ("wrote a P1 CIF (the input is P1) of a structure in Fm-3m: cif2d12's default symmetry_handling CIF "
            "would build a P1 deck from it; set symmetry_handling SPGLIB in your options for a deck in Fm-3m") in out


def test_cifout_round_trip_and_p1_fallback(tmp_path):
    import ase.io
    import numpy as np
    from mace.preopt import cifout
    a = ase.io.read(str(F.EXAMPLES / "MgO.cif"))
    m = cifout.write(tmp_path / "x.cif", a, name="x", header=["# h"], spacegroup=a.info["spacegroup"],
                     kinds=a.arrays["spacegroup_kinds"])
    assert m["mode"] == "input_group" and m["group"].startswith("225")
    assert (tmp_path / "x.cif").read_text().count("_atom_site_fract_x") == 1
    b = a.copy()
    b.positions[1] += [0.05, 0, 0]                    # a moved atom that the group cannot reproduce
    m = cifout.write(tmp_path / "y.cif", b, name="y", header=[], spacegroup=a.info["spacegroup"],
                     kinds=a.arrays["spacegroup_kinds"])
    assert m["mode"] == "p1" and m["reason"].startswith("re-expanding")
    back = ase.io.read(str(tmp_path / "y.cif"))
    assert len(back) == len(b) and np.allclose(sorted(back.get_scaled_positions().ravel()),
                                               sorted(b.get_scaled_positions().ravel()), atol=1e-8)
    m = cifout.write(tmp_path / "z.cif", a, name="z", header=[], spacegroup=a.info["spacegroup"],
                     kinds=a.arrays["spacegroup_kinds"], keep_group=False)
    assert m == {"mode": "p1", "reason": "the space group changed (--no-symmetry)", "group": "1 P 1"}


@pytest.mark.parametrize("argv", [("preopt", "-h"), ("preopt", "x.cif", "-h"), ("mlip", "-h"),
                                  ("mlip", "fetch", "--help")])
def test_help(work, argv):
    rc, out, err = mace(*argv, cwd=work)
    assert rc == 0 and f"Usage: mace {argv[0]}" in out and "Command to run" not in out
    if argv[0] == "preopt":
        assert "the next CRYSTAL step" in " ".join(out.split()) and "docs/preopt.md" in out
        # the --threads default as proc.default_threads computes it, SLURM included
        assert "SLURM_CPUS_PER_TASK" in out and "a quarter of the CPUs, 1-4" in out


def test_general_help_lists_both(work):
    rc, out, err = mace("--help", cwd=work)
    assert "preopt" in out and "mlip" in out
    # one line each in the grouped command list (the usage line reads COMMAND)
    assert re.search(r"^  preopt      \S", out, re.M) and re.search(r"^  mlip        \S", out, re.M)


# --- fake ML through the CLI, mace convert ----------------------------------------------

@pytest.fixture
def fake_ml(tmp_path):
    return {"MACE_MLIP_PYTHON": str(F.make_fake_python(tmp_path / "venv")),
            "FAKE_HELLO_LOG": str(tmp_path / "hello.log")}


def test_fake_ml_run_marks_the_cif_and_records_provenance(work, fake_ml):
    rc, out, err = mace("preopt", "NaCl.cif", "--method", "mlip:chgnet-0.3.0", "--charge", "1", cwd=work, env=fake_ml)
    assert rc == 0, err
    assert out.startswith("Ran: ML potential: CHGNet 0.3.0, converged in")
    assert "warning: ML potentials take no charge input; they see a neutral cell" in out
    cif = (work / "NaCl_preopt.cif").read_text()
    assert "# mace-preopt: method=mlip:chgnet-0.3.0 model=chgnet-0.3.0 version=1" in cif
    assert "Pre-optimized with ML potential: CHGNet 0.3.0 (BSD-3-Clause-LBNL)" in cif
    p = json.loads((work / "NaCl_preopt.json").read_text())
    assert p["model"] == "chgnet-0.3.0" and p["code_license"] == "BSD-3-Clause-LBNL" and p["training_data"]
    assert p["citation"][0]["doi"] == "10.1038/s42256-023-00716-3" and p["charge"] == 1 and p["device"] == "cpu"
    rc, out, err = mace("preopt", "NaCl.cif", "--method", "mlip:chgnet-0.3.0", "--force", cwd=work,
                        env={**fake_ml, "FAKE_MODE": "crash"})
    assert rc == 3 and "crash:" in out


def _options(tmp_path, calc):
    src = json.loads((F.REPO / "Crystal_d12" / "example_configs" / "standard_dft_opt.json").read_text())
    src["configuration"]["calculation_type"] = calc
    if calc != "OPT":
        src["configuration"].pop("optimization_type", None)
        src["configuration"].pop("optimization_settings", None)
    p = tmp_path / f"opts_{calc}.json"
    p.write_text(json.dumps(src))
    return p


def _convert(tmp_path, cif, opts, tag):
    d = tmp_path / f"conv_{tag}"
    (d / "in").mkdir(parents=True)
    shutil.copy(cif, d / "in" / "s.cif")
    rc, out, err = mace("convert", "--batch", "--options_file", str(opts), "--cif_dir", str(d / "in"),
                        "--output_dir", str(d / "out"), cwd=d)
    assert rc == 0, (out + err)[-2000:]
    decks = list((d / "out").glob("*.d12"))
    assert len(decks) == 1
    return decks[0].read_text(), err


def _mask(text):
    return re.sub(r"-?\d+\.\d+(E[-+]?\d+)?", "#", text)


def test_preopt_cif_converts_like_the_original(work, tmp_path, fake_ml):
    """A13(a): same space group and asymmetric unit, the same deck apart from
    the coordinate and lattice numbers; OPTGEOM, functional, basis and
    tolerances untouched."""
    from ase.io import write
    rc, out, err = mace("preopt", "MgO.cif", "--method", "mlip:chgnet-0.3.0", cwd=work, env=fake_ml)
    assert rc == 0, err
    opt = _options(tmp_path, "OPT")
    d0, _ = _convert(tmp_path, work / "MgO.cif", opt, "orig")
    d1, e1 = _convert(tmp_path, work / "MgO_preopt.cif", opt, "pre")
    l0, l1 = d0.splitlines(), d1.splitlines()
    assert l0[3] == l1[3] and l0[5] == l1[5]                       # space group, atom count
    assert _mask(d0) == _mask(d1)
    assert "is an FF/ML pre-optimized starting geometry" not in e1      # OPT: no warning


def test_convert_warns_for_a_marked_cif_and_a_non_opt_type(work, tmp_path, fake_ml):
    rc, out, err = mace("preopt", "NaCl.cif", "--method", "mlip:chgnet-0.3.0", cwd=work, env=fake_ml)
    assert rc == 0, err
    sp = _options(tmp_path, "SP")
    d_marked, e_marked = _convert(tmp_path, work / "NaCl_preopt.cif", sp, "m")
    assert ("warning: s.cif is an FF/ML pre-optimized starting geometry (mlip:chgnet-0.3.0); a SP calculation on "
            "it is not meaningful. Run a CRYSTAL OPT first.") in e_marked
    unmarked = work / "NaCl_unmarked.cif"
    unmarked.write_text("".join(ln for ln in (work / "NaCl_preopt.cif").read_text().splitlines(True)
                                if not ln.startswith("# mace-preopt:")))
    d_plain, e_plain = _convert(tmp_path, unmarked, sp, "u")
    assert "pre-optimized starting geometry" not in e_plain
    assert d_marked == d_plain                                     # the warning changes nothing in the deck
