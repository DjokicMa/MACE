"""opt2d12's PHONOPY set (choice 8 / --calc-type PHONOPY) on phonopy's own Si-CRYSTAL
example (needs phonopy).

The fixture is a trimmed copy of phonopy v4.7.2 example/Si-CRYSTAL (see
tests/fixtures/phonopy_crystal/README.md). The oracle for the geometry is
phonopy's own CRYSTAL writer, run on the same parent in a scratch directory:
MACE must hand CRYSTAL exactly the fort.34 phonopy would. The decks are
opt2d12's, the same as any deck of the parent apart from the set's edits.
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

phonopy = pytest.importorskip("phonopy")

from conftest import REPO_ROOT, phonopy_set  # noqa: E402

from mace.phonopy_bridge import cli  # noqa: E402

FIX = REPO_ROOT / "tests" / "fixtures" / "phonopy_crystal"
MACE_CLI = REPO_ROOT / "mace_cli"

INTERNAL_SI = """si_internal
CRYSTAL
0 0 0
227
5.44897370
1
14  .125  .125   .125
BASISSET
POB-TZVP-REV2
DFT
PBE0
END
SHRINK
8 8
TOLINTEG
8 8 8 8 16
TOLDEE
9
END
"""


def _parent(tmp_path, kind):
    shutil.copy(FIX / "Si" / "parent.out", tmp_path / "parent.out")
    if kind == "explicit":
        shutil.copy(FIX / "Si" / "parent.d12", tmp_path / "parent.d12")
    else:
        (tmp_path / "parent.d12").write_text(INTERNAL_SI)
    return tmp_path / "parent.out"


def _generate(tmp_path, monkeypatch=None, kind="explicit", **flags):
    _parent(tmp_path, kind)
    ok, out = phonopy_set(tmp_path, **flags)
    assert ok
    return out


def _oracle(tmp_path, monkeypatch):
    """phonopy's write_supercells_with_displacements, in its own directory."""
    from phonopy import Phonopy
    from phonopy.interface.crystal import read_crystal, write_supercells_with_displacements

    odir = tmp_path / "oracle"
    odir.mkdir()
    monkeypatch.chdir(odir)
    cell, conv = read_crystal(str(FIX / "Si" / "parent.out"))
    ph = Phonopy(cell, supercell_matrix=np.diag([2, 2, 2]), primitive_matrix="P",
                 calculator="crystal")
    ph.generate_displacements()
    write_supercells_with_displacements(
        ph.supercell, ph.supercells_with_displacements,
        list(range(1, len(ph.supercells_with_displacements) + 1)), conv, 8,
        template_file="NO_TEMPLATE")
    return odir


def test_ext_is_byte_identical_to_phonopys_writer(tmp_path, monkeypatch, capsys):
    out = _generate(tmp_path, monkeypatch)
    odir = _oracle(tmp_path, monkeypatch)
    man = json.loads((out / "mace_phonopy.json").read_text())
    assert man["jobs"], "no displacement written"
    for i, job in enumerate(man["jobs"], start=1):
        assert (out / f"{job}.ext").read_bytes() == (odir / f"supercell-{i:03d}.ext").read_bytes()


def test_conventional_numbers_match_the_elements(tmp_path, monkeypatch):
    out = _generate(tmp_path, monkeypatch)
    man = json.loads((out / "mace_phonopy.json").read_text())
    ph = phonopy.load(str(out / "phonopy_disp.yaml"), produce_fc=False)
    assert [c % 100 for c in man["conv_numbers_supercell"]] == list(ph.supercell.numbers)
    assert len(man["conv_numbers_supercell"]) == man["n_atoms_supercell"] == 16


@pytest.mark.parametrize("kind", ["explicit", "internal"])
def test_deck_structure(tmp_path, monkeypatch, kind):
    out = _generate(tmp_path, kind=kind)
    man = json.loads((out / "mace_phonopy.json").read_text())
    for job in man["jobs"]:
        lines = (out / f"{job}.d12").read_text().splitlines()
        assert lines[1] == "EXTERNAL"
        if kind == "internal":
            assert lines[2:4] == ["BASISSET", "POB-TZVP-REV2"]
        else:
            assert lines[2] == "END" and lines[3] == "14 7"
        keys = [ln.strip() for ln in lines]
        assert keys.count("TOLDEE") == 1 and keys.count("GRADCAL") == 1
        assert keys[-1] == "END" and keys[-2] == "GRADCAL"
        for banned in ("OPTGEOM", "FREQCALC", "ELASTCON", "EOS"):
            assert banned not in keys
        assert keys[keys.index("TOLDEE") + 1] == "10"


def test_unique_job_names_and_reloadable_records(tmp_path, monkeypatch):
    out = _generate(tmp_path, pm="yes")
    assert out.name == "parent_ph222_PBE0_optimized"     # named as opt2d12 names a deck of the parent
    man = json.loads((out / "mace_phonopy.json").read_text())
    assert man["jobs"] == ["parent_ph222_PBE0_optimized_disp-001", "parent_ph222_PBE0_optimized_disp-002"]
    ph = phonopy.load(str(out / "phonopy_disp.yaml"), produce_fc=False)
    assert len(ph.supercells_with_displacements) == len(man["jobs"])
    assert man["phonopy_version"] == phonopy.__version__
    assert len(man["parent_out_sha256"]) == 64
    titles = [(out / f"{j}.d12").read_text().splitlines()[0] for j in man["jobs"]]
    assert titles == man["jobs"]                          # opt2d12's title: the deck's name


def test_existing_output_dir_is_not_overwritten(tmp_path, monkeypatch, capsys):
    _generate(tmp_path)
    capsys.readouterr()
    assert phonopy_set(tmp_path)[0] is False
    assert "not empty (add --force to write into it)" in "".join(capsys.readouterr())
    assert phonopy_set(tmp_path, force=True)[0] is True


def test_output_dir_inside_workflow_outputs_is_refused(tmp_path, monkeypatch, capsys):
    _parent(tmp_path, "explicit")
    (tmp_path / "workflow_outputs" / "step_002_PH").mkdir(parents=True)
    ok, _ = phonopy_set(tmp_path, output_dir="workflow_outputs/step_002_PH")
    assert ok is False and "workflow_outputs" in "".join(capsys.readouterr())
    assert not list((tmp_path / "workflow_outputs").rglob("*.d12"))


def test_refused_parent_writes_nothing(tmp_path, monkeypatch, capsys):
    _parent(tmp_path, "explicit")
    (tmp_path / "parent.d12").write_text(
        (tmp_path / "parent.d12").read_text().replace("CRYSTAL\n", "SLAB\n", 1))
    assert phonopy_set(tmp_path)[0] is False
    err = capsys.readouterr().err
    assert err.startswith("Refusing parent.out:\n") and "[NOT_3D]" in err
    assert not list(tmp_path.glob("parent_ph*"))


@pytest.mark.parametrize("extra", ["END\n", "TOLDEE\n9\nEND\n"], ids=["end", "toldee-end"])
def test_stray_end_in_block3_is_refused_before_anything_is_written(
        tmp_path, monkeypatch, capsys, extra):
    """CRYSTAL stops reading block 3 at its first END, so a parent may carry a
    second one. Copied into a displaced deck it would stop CRYSTAL before
    GRADCAL (rc 0, no forces): refuse cleanly, with no half-written set."""
    _parent(tmp_path, "explicit")
    d12 = tmp_path / "parent.d12"
    d12.write_text(d12.read_text().rstrip("\n") + "\n" + extra)
    assert phonopy_set(tmp_path)[0] is False
    assert "[DECK_STRUCTURE]" in capsys.readouterr().err
    assert not list(tmp_path.glob("parent_ph*"))


def test_runs_without_contextlib_chdir(tmp_path, monkeypatch):
    """contextlib.chdir is Python 3.11+; MACE documents 3.7+, so the set and
    its analysis must not need it."""
    import contextlib
    monkeypatch.delattr(contextlib, "chdir", raising=False)
    out = _generate(tmp_path)
    assert (out / "phonopy_disp.yaml").is_file()


# ---------------------------------------------------------------------------
# The decks: opt2d12's, with the set's edits
# ---------------------------------------------------------------------------

def _shrink(deck_lines):
    i = [ln.strip() for ln in deck_lines].index("SHRINK")
    return deck_lines[i + 1].strip()


def test_shrink_is_divided_by_the_supercell_by_default(tmp_path, monkeypatch, capsys):
    out = _generate(tmp_path)
    man = json.loads((out / "mace_phonopy.json").read_text())
    assert man["scale_shrink"] is True
    lines = (out / f"{man['jobs'][0]}.d12").read_text().splitlines()
    assert _shrink(lines) == "4 4"                     # parent 8 8, 2x2x2
    assert "SHRINK 8 8 8 (ISP 8) -> 4 4 4 (ISP 4)" in capsys.readouterr().out


def test_keep_shrink(tmp_path, monkeypatch):
    out = _generate(tmp_path, keep_shrink=True)
    man = json.loads((out / "mace_phonopy.json").read_text())
    assert man["scale_shrink"] is False
    assert _shrink((out / f"{man['jobs'][0]}.d12").read_text().splitlines()) == "8 8"


def test_non_diagonal_dim_keeps_shrink_with_a_warning(tmp_path, monkeypatch, capsys):
    _parent(tmp_path, "explicit")
    ok, d = phonopy_set(tmp_path, dim="-1 1 1  1 -1 1  1 1 -1")
    assert ok
    assert "SHRINK kept" in "".join(capsys.readouterr())
    assert d.name == "parent_phm-1111-1111-1_PBE0_optimized"
    man = json.loads((d / "mace_phonopy.json").read_text())
    assert man["scale_shrink"] is False
    assert _shrink((d / f"{man['jobs'][0]}.d12").read_text().splitlines()) == "8 8"


def _scf(deck_text):
    """The deck from its DFT block on (the basis and geometry before it are compared apart)."""
    lines = [ln.strip() for ln in deck_text.splitlines()]
    return lines[lines.index("DFT"):]


@pytest.mark.parametrize("kind", ["explicit", "internal"])
def test_the_decks_inherit_the_parents_settings(tmp_path, kind):
    """A displaced deck is the SP deck opt2d12 writes from the same parent, with the set's
    edits only: EXTERNAL for the geometry, SHRINK divided by the supercell, TOLDEE at least
    10, MAXCYCLE at least 200, the buffers of a supercell of more than 5 atoms and GRADCAL
    before the final END."""
    import CRYSTALOptToD12 as O
    out = _generate(tmp_path, kind=kind)
    man = json.loads((out / "mace_phonopy.json").read_text())
    ours = (out / f"{man['jobs'][0]}.d12").read_text()
    import io
    cwd, stdin = os.getcwd(), sys.stdin
    os.chdir(tmp_path)
    sys.stdin = io.StringIO("")
    try:
        assert O.process_files("parent.out", "parent.d12", non_interactive=True, calc_type="SP",
                               unattended=True)[0]
        sp = (tmp_path / O.LAST_RESULT["deck"]).read_text()
    finally:
        sys.stdin = stdin
        os.chdir(cwd)
    sp_lines, our_lines = sp.splitlines(), ours.splitlines()
    # the basis: the parent's records, as the SP deck writes them
    start = sp_lines.index("BASISSET") if kind == "internal" else sp_lines.index("END") + 1
    basis_sp = sp_lines[start:sp_lines.index("DFT")]
    assert our_lines[2 if kind == "internal" else 3:our_lines.index("DFT")] == basis_sp
    want = _scf(sp)
    for key, value in (("SHRINK", "4 4"), ("TOLDEE", "10")):
        want[want.index(key) + 1] = value
    i = want.index("MAXCYCLE") + 1
    want[i] = str(max(200, int(want[i])))
    if "BIPOSIZE" not in want:            # d12_writer's buffers for more than 5 atoms (16 here)
        i = want.index("SCFDIR") + 1
        want[i:i] = ["BIPOSIZE", "110000000", "EXCHSIZE", "110000000"]
    assert want[-1] == "END"
    want = want[:-1] + ["GRADCAL", "END"]
    assert _scf(ours) == want
    assert not (out / "TEMPLATE").exists()


# ---------------------------------------------------------------------------
# --cry-sym: phonopy's CRY_SYM (symmetry operators in each .ext), opt-in
# ---------------------------------------------------------------------------

def test_cry_sym_ext_is_byte_identical_to_phonopys_writer(tmp_path, monkeypatch, capsys):
    from phonopy.interface.crystal import write_supercells_with_displacements

    out = _generate(tmp_path, cry_sym=True)
    text = "".join(capsys.readouterr())
    assert "R-3m" in text
    man = json.loads((out / "mace_phonopy.json").read_text())
    assert man["cry_sym"] is True and (out / "CRY_SYM").is_file()
    ph = phonopy.load(str(out / "phonopy_disp.yaml"), produce_fc=False)
    odir = tmp_path / "viaphonopy"
    odir.mkdir()
    (odir / "CRY_SYM").write_text("")
    monkeypatch.chdir(odir)
    write_supercells_with_displacements(
        ph.supercell, ph.supercells_with_displacements,
        list(range(1, len(ph.supercells_with_displacements) + 1)), man["conv_numbers"], 8,
        template_file="NO_TEMPLATE")
    for i, job in enumerate(man["jobs"], start=1):
        ours = (out / f"{job}.ext").read_text()
        assert ours == (odir / f"supercell-{i:03d}.ext").read_text()
        assert int(ours.splitlines()[4]) > 1          # more than the identity


def test_no_cry_sym_by_default(tmp_path, monkeypatch, capsys):
    out = _generate(tmp_path)
    man = json.loads((out / "mace_phonopy.json").read_text())
    assert man["cry_sym"] is False and not (out / "CRY_SYM").exists()
    for job in man["jobs"]:
        assert (out / f"{job}.ext").read_text().splitlines()[4] == "1"
    assert "P1" in capsys.readouterr().out


def test_cry_sym_warns_harder_for_a_rhombohedral_parent():
    from phonopy.structure.atoms import PhonopyAtoms

    from mace.phonopy_bridge import structure as S

    a, alpha, u = 4.75, np.radians(57.2), 0.234      # Bi, R-3m
    c = np.cos(alpha)
    ty = (c - c * c) / np.sin(alpha)
    lat = a * np.array([[1, 0, 0], [c, np.sin(alpha), 0],
                        [c, ty, np.sqrt(1 - c * c - ty * ty)]])
    bi = PhonopyAtoms(symbols=["Bi", "Bi"], cell=lat,
                      scaled_positions=[[u, u, u], [-u, -u, -u]])
    msgs = S.cry_sym_warnings(bi)
    assert any("R-3m" in m and "parent" in m for m in msgs)
    si = PhonopyAtoms(symbols=["Si", "Si"], cell=5.43 / 2 * np.array([[0, 1, 1], [1, 0, 1],
                                                                       [1, 1, 0]]),
                      scaled_positions=[[0.125] * 3, [-0.125] * 3])
    assert not any("parent" in m for m in S.cry_sym_warnings(si))


# ---------------------------------------------------------------------------
# Walltime: scaled with the supercell, in mace submit's --walltime format
# ---------------------------------------------------------------------------

def test_walltime_estimate_scales_with_the_supercell():
    """Calibrated on the heaviest measured case (CRYSTAL23, diamond B3LYP-D3,
    spin, XLGRID, 2x2x2 P1, SHRINK 6: 3160 s on 64 tasks). 3x3x3 at SHRINK 4
    did not finish its SCF in 2 h on 64 tasks, so its estimate must be far
    above 2 h."""
    two, capped2 = cli.suggest_walltime(6 ** 3, 16)
    three, capped3 = cli.suggest_walltime(4 ** 3, 54)
    assert two == "3:00:00" and not capped2      # 1.76 h measured-scaled, x 1.5
    days, _, hours = three.split(":")[0].rpartition("-")
    h3 = 24 * int(days or 0) + int(hours)
    assert h3 >= 12 and not capped3
    big, capped = cli.suggest_walltime(12 ** 3, 250)
    assert big == "7-00:00:00" and capped


def test_walltime_estimate_has_a_margin_over_the_measured_run():
    """The calibration run (3160 s on 64 tasks, ~6300 s on 32) must not land
    within a few percent of the suggestion: a job killed at its walltime leaves
    no forces, so the estimate carries a 1.5x margin (the old 2 h left 14%)."""
    measured_on_32 = 2 * 3160.0
    walltime, _ = cli.suggest_walltime(6 ** 3, 16)
    hours = int(walltime.split(":")[0])
    assert hours * 3600 >= 1.5 * measured_on_32


def test_walltime_format_is_what_mace_submit_accepts():
    """mace submit --walltime goes into #SBATCH -t: SLURM's H:MM:SS or D-HH:MM:SS."""
    import re
    pat = re.compile(r"^(\d+-\d{2}|\d{1,2}):\d{2}:\d{2}$")
    for nk, n in ((1, 2), (216, 16), (64, 54), (512, 128), (1728, 250)):
        assert pat.match(cli.suggest_walltime(nk, n)[0])
    assert cli.format_walltime(2) == "2:00:00"
    assert cli.format_walltime(23) == "23:00:00"
    assert cli.format_walltime(30) == "1-06:00:00"


def test_generate_prints_and_records_the_scaled_walltime(tmp_path, monkeypatch, capsys):
    out = _generate(tmp_path)
    text = capsys.readouterr().out
    man = json.loads((out / "mace_phonopy.json").read_text())
    assert man["walltime"] == "2:00:00"
    assert f"mace submit {out.name} --walltime 2:00:00" in text
    assert "24:00:00" not in text


# ---------------------------------------------------------------------------
# The real invocation path: mace opt2d12, menu choice 8 and --calc-type PHONOPY
# ---------------------------------------------------------------------------

def _cli(*args, cwd, stdin=""):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["MACE_NO_BANNER"] = "1"
    return subprocess.run([sys.executable, str(MACE_CLI), *args], cwd=cwd, env=env, input=stdin,
                          capture_output=True, text=True, timeout=600)


def test_choice_8_and_calc_type_phonopy_write_the_same_set(tmp_path):
    """Menu choice 8 with Enter at every question (the parent's values, the suggested supercell)
    and --calc-type PHONOPY --non-interactive write the same set. Only the tolerance question
    differs: its Enter is the Standard level (7 7 7 7 14), where --non-interactive keeps the
    phonopy example parent's own TOLINTEG 8 8 8 8 16 (no preset)."""
    menu, flag = tmp_path / "menu", tmp_path / "flag"
    for d in (menu, flag):
        d.mkdir()
        _parent(d, "explicit")
    r = _cli("opt2d12", "--out-file", "parent.out", cwd=menu, stdin="n\n8\n" + "\n" * 60)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert "8. Finite-displacement phonons with phonopy (PHONOPY)" in r.stdout
    assert "Supercell, \"N N N\" or 9 integers" in r.stdout and "[3 3 3]" in r.stdout
    r = _cli("opt2d12", "--out-file", "parent.out", "--calc-type", "PHONOPY", "--non-interactive", cwd=flag)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert "Supercell 3 3 3 (suggested" in r.stdout
    a, b = menu / "parent_ph333_PBE0_optimized", flag / "parent_ph333_PBE0_optimized"
    ma, mb = (json.loads((d / "mace_phonopy.json").read_text()) for d in (a, b))
    assert ma["jobs"] == mb["jobs"] and ma["jobs"]
    assert (a / "phonopy_disp.yaml").read_bytes() == (b / "phonopy_disp.yaml").read_bytes()
    for job in ma["jobs"]:
        assert (a / f"{job}.ext").read_bytes() == (b / f"{job}.ext").read_bytes()
        da, db = ((d / f"{job}.d12").read_text().splitlines() for d in (a, b))
        i = da.index("TOLINTEG")
        assert da[i + 1] == "7 7 7 7 14" and db[i + 1] == "8 8 8 8 16"
        assert da[:i + 1] + da[i + 2:] == db[:i + 1] + db[i + 2:]
        assert da[1] == "EXTERNAL" and da[-2:] == ["GRADCAL", "END"]


def test_an_unfinished_parent_is_refused_with_the_same_message(tmp_path):
    """Before anything is asked: --calc-type PHONOPY, and menu choice 8 right after it is
    chosen, print phonopy_bridge.parent's refusal and write nothing."""
    from mace.phonopy_bridge import parent as P
    _parent(tmp_path, "explicit")
    text = (tmp_path / "parent.out").read_text()
    (tmp_path / "parent.out").write_text(text[:text.index("EEEEEEEEEE TERMINATION")])
    deck, out_text = P.read_parent(tmp_path / "parent.d12", tmp_path / "parent.out")
    want = ["Refusing parent.out:"] + [f"  [{c}] {m}" for c, m in P.refusal_reasons(deck, out_text)]
    assert "  [ABNORMAL_TERMINATION]" in want[1]
    r = _cli("opt2d12", "--out-file", "parent.out", "--calc-type", "PHONOPY", "--non-interactive",
             "--dim", "2", "2", "2", cwd=tmp_path)
    assert r.returncode == 1 and r.stderr.splitlines() == want, r.stderr
    r = _cli("opt2d12", "--out-file", "parent.out", cwd=tmp_path, stdin="n\n8\n" + "\n" * 60)
    assert r.returncode == 1 and r.stderr.splitlines() == want, r.stderr
    assert "Select method type" not in r.stdout
    assert not list(tmp_path.glob("parent_ph*"))


def test_phonopy_flags_need_calc_type_phonopy(tmp_path):
    _parent(tmp_path, "explicit")
    r = _cli("opt2d12", "--out-file", "parent.out", "--calc-type", "SP", "--non-interactive",
             "--dim", "2", "2", "2", cwd=tmp_path)
    assert r.returncode == 2 and "--dim goes with --calc-type PHONOPY" in r.stderr


# Measured on CRYSTAL23 with phonopy 4.7.2 (diamond, B3LYP-D3, parent
# test/SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized): decks with these block-3 records,
# after the parent's basis, ran and gave forces for 2x2x2 (SHRINK 6 12) and 3x3x3
# (SHRINK 4 8). The parent has TOLDEE 9 and SHRINK 12 24.
DIAMOND_BLOCK3 = ["DFT", "SPIN", "B3LYP-D3", "XLGRID", "ENDDFT", "TOLINTEG",
                  "8 8 8 9 24", "SHRINK", "{shrink}", "SCFDIR", "MAXCYCLE", "1600", "FMIXING",
                  "30", "DIIS", "HISTDIIS", "100", "PPAN"]


@pytest.mark.parametrize("n,shrink", [(2, "6 12"), (3, "4 8")])
def test_diamond_decks_carry_the_measured_records(tmp_path, n, shrink):
    """The set opt2d12 writes from that parent has those records (in opt2d12's order), plus
    TOLDEE 10, d12_writer's buffers and GRADCAL before the final END; the basis is the
    parent's."""
    from conftest import find_data
    parent = find_data("SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.d12")
    for ext in ("out", "d12"):
        shutil.copy(parent.with_suffix(f".{ext}"), tmp_path / f"parent.{ext}")
    ok, out = phonopy_set(tmp_path, dim=f"{n} {n} {n}")
    assert ok
    man = json.loads((out / "mace_phonopy.json").read_text())
    deck = [ln.strip() for ln in (out / f"{man['jobs'][0]}.d12").read_text().splitlines()]
    text = parent.read_text().splitlines()
    basis = text[text.index("END") + 1:]
    basis = [" ".join(ln.split()) for ln in basis[:basis.index("99 0") + 1]]
    assert deck[1:3] == ["EXTERNAL", "END"]
    assert [" ".join(ln.split()) for ln in deck[3:3 + len(basis)]] == basis
    assert deck[3 + len(basis)] == "END"
    block3 = deck[4 + len(basis):]
    assert block3[-2:] == ["GRADCAL", "END"]
    assert block3[block3.index("TOLDEE") + 1] == "10"
    rest = [ln for ln in block3[:-2] if ln not in ("BIPOSIZE", "EXCHSIZE", "110000000", "TOLDEE", "10")]
    assert sorted(rest) == sorted(ln.format(shrink=shrink) for ln in DIAMOND_BLOCK3)
