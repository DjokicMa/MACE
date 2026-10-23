"""Reference energies in a database build, on real CRYSTAL23 outputs of the
test/ corpus (gitignored; skipped without it), through mace_cli as a
subprocess.

* A folder's properties that read other runs (formation energy, ...) are
  computed in a second pass, after every run's own rows are stored: two
  builds that read the same runs in opposite orders store the same values,
  and ``--skip-unchanged`` still computes them for the unchanged runs when a
  new reference run arrives.
* A reference run that cannot be read (a locked database) is an error of
  that property alone, and its stored rows stay.
* ``--reference`` names a reference a property takes; an output without a
  calculation record cannot save the choice (a warning says so).
* ΔH and ΔG of a run without its own FREQ thermodynamics borrow only the
  thermal correction of a FREQ run of its material, scaled by their atoms:
  diamond's SP (C2) against 1/8 of its 2x2x2 supercell's FREQ run (C16)
  gives ΔH = ΔG = ΔE (it was -14515 eV)."""
import json
import os
import shutil
import sqlite3
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, TEST_DATA

pytestmark = pytest.mark.skipif(not TEST_DATA.is_dir(), reason="test/ data corpus not present (gitignored)")

OPT = TEST_DATA / "OPT"
T137 = "3.4^2T137_rev1"
T2 = "3.4^9T2_BULK_OPTGEOM_TZ_opt_B3LYP-D3-D3_optimized_rev1"
DIA = "1_dia_opt_rev1"
DIA_SP = TEST_DATA / "SP" / "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized"
DIA_FREQ = TEST_DATA / "FREQ" / "1_dia_opt_rev1_freq_B3LYP-D3-D3_optimized_supercel222"
REF_PREFIXES = ("formation_", "cohesive_", "surface_", "adsorption_", "reaction_")

# mace_cli with the folder read in a fixed order (sorted, or reversed)
_ORDERED = (
    "import os, pathlib, runpy, sys\n"
    "order = sys.argv.pop(1)\n"
    "sys.path.insert(0, os.path.dirname(sys.argv[1]))\n"
    "orig = pathlib.Path.rglob\n"
    "pathlib.Path.rglob = lambda self, *a, **k: iter(sorted(orig(self, *a, **k), reverse=order == 'reverse'))\n"
    "sys.argv = sys.argv[1:]\n"
    "runpy.run_path(sys.argv[0], run_name='__main__')\n")


def _env():
    e = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    e["OMP_NUM_THREADS"] = "1"
    return e


def _cli(cwd, *args, order=None, ok=True):
    pre = ["-c", _ORDERED, order] if order else []
    p = subprocess.run([sys.executable, *pre, str(REPO_ROOT / "mace_cli"), "--no-banner", *args],
                       capture_output=True, text=True, env=_env(), cwd=str(cwd), timeout=900)
    if ok:
        assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return p


def _copy(dst, *stems):
    dst.mkdir(parents=True, exist_ok=True)
    for s in stems:
        for ext in (".out", ".d12"):
            shutil.copy(s.parent / (s.name + ext), dst / (s.name + ext))


def _ref_rows(db):
    """{output name: {name: (value, values of the JSON)}} of the reference-energy rows."""
    conn = sqlite3.connect(db)
    out = {}
    try:
        for f, name, v, t in conn.execute(
                "SELECT c.output_file, p.property_name, p.property_value, p.property_value_text FROM properties p "
                "JOIN calculations c ON c.calc_id = p.calc_id"):
            if not name.startswith(REF_PREFIXES) or name == "reference_runs":
                continue
            if v is None and t and t.startswith("{"):
                d = json.loads(t)
                v = (d.get("values"), d.get("formula"), [r.get("run") for r in d.get("runs", [])])
            out.setdefault(os.path.basename(f), {})[name] = v
    finally:
        conn.close()
    return out


def _calc(db, stem):
    conn = sqlite3.connect(db)
    try:
        return conn.execute("SELECT calc_id FROM calculations WHERE output_file LIKE ?",
                            (f"%{stem}.out",)).fetchone()[0]
    finally:
        conn.close()


def _rows(db, calc_id):
    conn = sqlite3.connect(db)
    try:
        return {n: (v, t) for n, v, t in conn.execute(
            "SELECT property_name, property_value, property_value_text FROM properties WHERE calc_id = ?",
            (calc_id,))}
    finally:
        conn.close()


def test_a_folder_build_does_not_depend_on_the_order_it_reads_the_runs(tmp_path):
    runs = tmp_path / "runs"
    for sub, stem in (("a", T137), ("b", T2), ("c", DIA)):
        _copy(runs / sub, OPT / stem)
    built = {}
    for order in ("sorted", "reverse"):
        db = tmp_path / f"{order}.db"
        p = _cli(tmp_path, "analyze", "--extract-properties", str(runs), "--db-path", str(db), order=order)
        assert "Reference properties of 3 run(s)" in p.stdout
        built[order] = _ref_rows(db)
    assert built["sorted"] == built["reverse"]
    # diamond is the carbon reference of both carbon allotropes, whichever was read first
    for stem in (T137, T2):
        assert built["sorted"][stem + ".out"]["formation_energy_ev_per_atom"] > 0
    assert "formation_energy_ev_per_atom" not in built["sorted"].get(DIA + ".out", {})


def test_skip_unchanged_computes_the_reference_properties_when_a_reference_arrives(tmp_path):
    runs = tmp_path / "runs"
    _copy(runs / "a", OPT / T137)
    _copy(runs / "b", OPT / T2)
    db = tmp_path / "m.db"
    _cli(tmp_path, "analyze", "--extract-properties", str(runs), "--db-path", str(db))
    before = {s: _ref_rows(db).get(s + ".out", {}).get("formation_energy_ev_per_atom") for s in (T137, T2)}
    # two carbon allotropes: the higher one against the lower one, which is zero by definition
    assert sorted(v is None for v in before.values()) == [False, True]
    # diamond (lower still) arrives; the two allotropes are unchanged and not extracted again
    _copy(runs / "c", OPT / DIA)
    p = _cli(tmp_path, "analyze", "--extract-properties", str(runs), "--db-path", str(db), "--skip-unchanged")
    assert p.stdout.count("Processing:") == 1
    after = _ref_rows(db)
    for s in (T137, T2):
        assert after[s + ".out"]["formation_energy_ev_per_atom"] > 0
        assert after[s + ".out"]["formation_energy_ev_per_atom"] != before[s]     # the lookup's new answer
        assert DIA + ".out" in after[s + ".out"]["formation_energy"][2]
    # the same values as a build that had every run from the start
    fresh = tmp_path / "fresh.db"
    _cli(tmp_path, "analyze", "--extract-properties", str(runs), "--db-path", str(fresh))
    assert _ref_rows(fresh) == after


def test_a_reference_that_cannot_be_read_never_deletes_stored_rows(tmp_path, monkeypatch, capsys):
    runs = tmp_path / "runs"
    _copy(runs / "b", OPT / T2)
    _copy(runs / "c", OPT / DIA)
    db = tmp_path / "m.db"
    _cli(tmp_path, "analyze", "--extract-properties", str(runs), "--db-path", str(db))
    cid, dia = _calc(db, T2), _calc(db, DIA)
    before = _rows(db, cid)
    assert "formation_energy_ev_per_atom" in before

    from mace.properties import registry
    from mace.utils import property_extractor as pe

    def locked(conn, calc_id):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(registry, "stored_values", locked)
    monkeypatch.setattr(sys, "argv", ["x", "--output-file", str(runs / "b" / (T2 + ".out")), "--db-path", str(db),
                                      "--reference", f"elements={dia}"])
    pe.main()
    assert "formation_energy failed: OperationalError: database is locked" in capsys.readouterr().out
    after = _rows(db, cid)
    assert set(after) == set(before)
    assert after["formation_energy_ev_per_atom"] == before["formation_energy_ev_per_atom"]
    prov = json.loads(after["extraction_provenance"][1])["derived"]
    assert prov["formation_energy"]["status"] == "error"
    assert prov["packing_fraction"]["status"] == "stored"       # the others still ran


def test_reference_names_are_checked_and_an_untracked_choice_is_not_saved(tmp_path):
    _copy(tmp_path / "r", OPT / T2, OPT / DIA)
    out = str(tmp_path / "r" / (T2 + ".out"))
    p = _cli(tmp_path, "analyze", "--extract-properties", "--output-file", out, "--db-path", "m.db",
             "--reference", "element=x.out", ok=False)
    assert p.returncode == 2
    assert "no derived property takes a reference named 'element'" in p.stderr and "elements" in p.stderr
    p = _cli(tmp_path, "analyze", "--extract-properties", "--output-file", out, "--db-path", "m.db",
             "--reference", f"elements={tmp_path / 'r' / (DIA + '.out')}")
    assert "has no calculation record: its --reference choices are used now but not saved" in p.stdout
    conn = sqlite3.connect(tmp_path / "m.db")
    try:
        assert conn.execute("SELECT count(*) FROM properties WHERE property_name = 'reference_runs'").fetchone()[0] == 0
        assert conn.execute("SELECT property_value FROM properties WHERE property_name = "
                            "'formation_energy_ev_per_atom'").fetchone()[0] > 0
    finally:
        conn.close()


def test_borrowed_thermodynamics_scale_with_the_cell(tmp_path):
    """Diamond's SP (C2) borrows the thermal correction of the 2x2x2 supercell's FREQ run (C16) of the
    same material: against 1/8 of that FREQ run, ΔH and ΔG equal ΔE (the same thermal correction per
    atom on both sides), and ΔE is the two cells' small energy difference."""
    _copy(tmp_path / "dia", DIA_SP, DIA_FREQ)
    other = tmp_path / "other"
    _copy(other, DIA_FREQ)
    db = tmp_path / "m.db"
    _cli(tmp_path, "analyze", "--extract-properties", str(tmp_path / "dia"), "--db-path", str(db))
    sp = str(tmp_path / "dia" / (DIA_SP.name + ".out"))
    _cli(tmp_path, "analyze", "--extract-properties", "--output-file", sp, "--db-path", str(db),
         "--reference", f"reactants={other / (DIA_FREQ.name + '.out')}")
    rows = _rows(db, _calc(db, DIA_SP.name))
    de = rows["reaction_energy_ev"][0]
    assert abs(de) < 0.02
    assert rows["reaction_enthalpy_ev"][0] == pytest.approx(de, abs=1e-6)
    assert rows["reaction_gibbs_energy_ev"][0] == pytest.approx(de, abs=1e-6)
    detail = json.loads(rows["reaction_energy"][1])
    assert detail["equation"] == "1/8 C16 → 1 C2"
    assert any("×1/8" in s for s in detail["thermo"]["from"]), detail["thermo"]
