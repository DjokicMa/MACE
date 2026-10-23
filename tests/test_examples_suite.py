"""The bundled example suite (examples/suite): manifest, hydration, `mace examples`, the GUI hook.

Self-contained: it reads only what is in the repository (the gitignored output corpus is not needed).
Hydration always goes into a temporary copy, never into the checkout.
"""
import json
import re
import subprocess
import sys

import pytest

from conftest import REPO_ROOT
from mace import examples_suite as ES

SUITE = REPO_ROOT / "examples" / "suite"
MACE_CLI = REPO_ROOT / "mace_cli"
ARCHETYPES = {"porous", "dense-3d", "slab-2d", "polymer-1d", "molecule-0d"}


@pytest.fixture(scope="module")
def manifest():
    return ES.load_manifest(SUITE)


@pytest.fixture(scope="module")
def copy(tmp_path_factory):
    return ES.copy_suite(tmp_path_factory.mktemp("suite") / "s")


# one entry per dimensionality with every calculation type (Marcus: "full coverage of at least 1 3d, 1 2d,
# 1 1d, and 1 0d materials"); n/a only where CRYSTAL23 does not run it, with the manual page
FULL_COVERAGE = {3: "diamond", 2: "hBN", 1: "polyacetylene", 0: "EC"}


def test_five_archetypes_and_every_calc_type_has_a_status(manifest):
    assert {a["id"] for a in manifest["archetypes"]} == ARCHETYPES
    assert manifest["calc_types"] == list(ES.CALC_TYPES)
    entries = {}
    for a in manifest["archetypes"]:
        assert a["entries"], a["id"]
        for e in a["entries"]:
            entries[e["id"]] = e
            for ct, c in e["cells"].items():
                assert ct in ES.CALC_TYPES and c["status"] in ("real", "failed", "pending", "n/a")
                if c["status"] == "n/a":
                    assert c.get("reason")
                if c["status"] == "failed":
                    assert c.get("note"), (e["id"], ct)          # the CRYSTAL line that says why
    for dim, eid in FULL_COVERAGE.items():
        e = entries[eid]
        assert e["dimensionality"] == dim, eid
        assert set(e["cells"]) == set(ES.CALC_TYPES), (eid, set(ES.CALC_TYPES) - set(e["cells"]))
        for ct, c in e["cells"].items():
            if c["status"] == "n/a":
                assert re.search(r"CRYSTAL23 manual.*p\.\d+", c["reason"]), (eid, ct, c["reason"])


def test_real_cells_name_files_and_pending_cells_ship_their_decks(manifest):
    known = {f["path"] for f in manifest["files"]}
    for p in ES.pending(manifest):
        for deck in p["decks"]:
            assert deck in known
        assert p["decks"] or p["requires"], p      # a deck, or what to run first to write it
    for a in manifest["archetypes"]:
        for e in a["entries"]:
            for ct, c in e["cells"].items():
                if c["status"] in ("real", "failed"):
                    assert c["files"] and all(f in known for f in c["files"]), (e["id"], ct)
                    assert any(f.endswith((".out", ".dat", ".DAT")) for f in c["files"]), (e["id"], ct)


def test_stored_files_exist_and_the_repository_stays_lean(manifest):
    total = 0
    for f in manifest["files"]:
        stored = SUITE / (f.get("stored") or f["path"])
        assert stored.is_file(), f
        total += stored.stat().st_size
    assert total < 11_000_000, total     # 14 calculation types on four full-coverage entries
    # nothing a run leaves behind: wavefunctions, scratch, job scripts
    assert not [f["path"] for f in manifest["files"] if re.search(r"\.(f9|f98|sh|o\d*)$|fort\.", f["path"])]


def test_hydration_restores_every_checksum_and_is_idempotent(manifest, copy):
    for f in manifest["files"]:
        p = copy / f["path"]
        assert p.is_file() and ES._sha256(p) == f["sha256"], f["path"]
        assert not (copy / (f["path"] + ".xz")).exists()
    again = ES.hydrate(copy)
    assert not again["written"] and not again["errors"] and again["present"]


def test_hydration_catches_a_corrupt_file(tmp_path):
    c = ES.copy_suite(tmp_path / "c", hydrated=False)
    victim = next(f for f in ES.load_manifest(c)["files"] if f.get("stored"))
    (c / victim["stored"]).write_bytes(b"not xz")
    res = ES.hydrate(c)
    assert any(victim["path"] in e for e in res["errors"])


def test_gui_hook_unpacks_a_suite_folder_once_and_ignores_others(tmp_path):
    c = ES.copy_suite(tmp_path / "c", hydrated=False)
    ES._DONE.discard(str(c.resolve()))
    res = ES.hydrate_for_root(c / "3-slab-2d")           # a folder inside the suite counts
    assert res and res["written"] and not res["errors"]
    assert ES.hydrate_for_root(c) is None                # already done in this process
    other = tmp_path / "plain"
    other.mkdir()
    assert ES.hydrate_for_root(other) is None


def test_decimate_cube_keeps_header_and_every_second_point():
    sys.path.insert(0, str(REPO_ROOT / "examples" / "gui"))
    import build_suite
    n = (4, 4, 4)
    head = ["title", "comment", "    1   0.0 0.0 0.0"] + [f"{n[k]:5d} " + " ".join("0.5" if i == k else "0.0" for i in range(3)) for k in range(3)] \
        + ["    1    1.0   0.0   0.0   0.0"]
    vals = [str(i) for i in range(64)]
    text = "\n".join(head + [" ".join(vals[a:a + 6]) for a in range(0, 64, 6)]) + "\n"
    out = build_suite.decimate_cube(text).split("\n")
    assert out[3].split()[0] == "2" and abs(float(out[3].split()[1]) - 1.0) < 1e-9
    kept = " ".join(out[7:]).split()
    assert kept == [str(i * 16 + j * 4 + k) for i in (0, 2) for j in (0, 2) for k in (0, 2)]


def test_large_cube_grids_are_decimated_by_four_and_zero_spin_cubes_are_spotted():
    """A chain's RANGE grid (polyacetylene: CRYSTAL's 100 x 518 x 404 points, 275 MB a cube) is decimated
    by 4, not 2, and a closed-shell run's SPIN cube (zero everywhere) is recognized, so neither pushes the
    stored suite past its size limit."""
    sys.path.insert(0, str(REPO_ROOT / "examples" / "gui"))
    import build_suite

    def head(n):
        return "\n".join(["t", "c", "    1   0.0 0.0 0.0"] + [f"{n[k]:5d} 0.1 0.0 0.0" for k in range(3)]
                         + ["    1    1.0   0.0   0.0   0.0"])
    assert build_suite.cube_step(head((100, 100, 403)) + "\n") == 2
    assert build_suite.cube_step(head((100, 518, 404)) + "\n") == 4
    n = (8, 8, 8)
    assert build_suite.decimate_cube(head(n) + "\n" + " ".join(["0.0"] * 512) + "\n", 4).split("\n")[0].endswith(
        "(every 4th grid point of the original)")
    assert build_suite.cube_is_zero(head(n) + "\n" + " ".join(["1.3E-15", "-2.3E-14"] * 256) + "\n")
    assert not build_suite.cube_is_zero(head(n) + "\n" + " ".join(["0.0"] * 511 + ["1.0E-03"]) + "\n")


def test_thinned_cube_grids_keep_an_even_periodic_step(manifest, copy):
    """A cube is thinned only by strides that divide each axis's point count (N/stride points at stride x the
    step), so a periodic grid stays evenly periodic for Bader charges and density dipoles; an axis with a prime
    count (graphene's 403 z points) stays whole."""
    sys.path.insert(0, str(REPO_ROOT / "examples" / "gui"))
    import build_suite
    assert build_suite.cube_strides([100, 518, 404], 4) == [4, 7, 4]
    assert build_suite.cube_strides([100, 140, 105], 2) == [2, 2, 3]
    assert build_suite.cube_strides([100, 100, 403], 2) == [2, 2, 1]
    n = (6, 6, 5)
    head = ["t", "c", "    1   0.0 0.0 0.0"] + [f"{n[k]:5d} " + " ".join("0.5" if i == k else "0.0" for i in range(3))
                                                  for k in range(3)] + ["    1    1.0   0.0   0.0   0.0"]
    out = build_suite.decimate_cube("\n".join(head + [" ".join(["1.0"] * 180)]) + "\n", 2).split("\n")
    assert [int(out[3 + k].split()[0]) for k in range(3)] == [3, 3, 5]
    assert float(out[3].split()[1]) == 1.0 and float(out[5].split()[3]) == 0.5
    for f in manifest["files"]:
        if f["path"].endswith(".CUBE") and "decimated" in f.get("note", ""):
            lines = (copy / f["path"]).read_text().split("\n", 6)
            assert "decimated by" in f["note"] and all(int(lines[3 + k].split()[0]) > 1 for k in range(3)), f["path"]


def test_no_private_paths_in_text_files(copy):
    bad = re.compile(r"djokicma|/mnt/|/home/|@msu\.edu", re.I)
    for p in copy.rglob("*"):
        if p.is_file():
            assert not bad.search(p.read_text(errors="replace")), p


def test_cli_lists_the_table_and_copies(tmp_path):
    r = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "examples"], capture_output=True, text=True,
                       cwd=str(REPO_ROOT), stdin=subprocess.DEVNULL, timeout=120)
    assert r.returncode == 0, r.stderr
    for word in ("EC", "polyacetylene", "graphene", "diamond", "SOD", "deck only", "output", "n/a"):
        assert word in r.stdout
    r = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "examples", "copy", str(tmp_path / "x")],
                       capture_output=True, text=True, cwd=str(REPO_ROOT), stdin=subprocess.DEVNULL, timeout=120)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "x" / "5-molecule-0d" / "EC" / "OPT" / "EC_opt.out").is_file()


def test_hydrated_outputs_open_in_the_structure_reader(copy):
    pytest.importorskip("ase")
    from mace.gui import structures as S
    want = {"5-molecule-0d/EC/OPT/EC_opt.out": ("C3H4O3", "MOLECULE"),
            "3-slab-2d/hBN/OPT/hBN_opt.out": ("BN", "SLAB"),
            "2-dense-3d/diamond/SP/diamond_sp.out": ("C2", "CRYSTAL"),
            "3-slab-2d/graphene/SP/graphene_sp.out": ("C2", "SLAB"),
            "4-polymer-1d/polyacetylene/OPT/polyacetylene_opt.out": ("C2H2", "POLYMER"),
            "3-slab-2d/4LG_AA/OPT/4LG_AA_opt.out": ("C96", "SLAB"),
            "1-porous/SOD/OPT/SOD_opt.d12": ("O24Si12", "CRYSTAL")}
    for rel, (formula, dim) in want.items():
        s = S.read_structure(copy / rel)
        assert s["formula"] == formula and s["dimensionality"] == dim, rel


def test_hydrated_plot_data_loads(copy):
    pytest.importorskip("plotly")
    from mace.gui import results
    assert results.band_data(copy / "2-dense-3d/diamond/BAND/diamond_band.BAND.DAT")["x"]
    assert results.dos_data(copy / "2-dense-3d/diamond/DOSS/diamond_doss.DOSS.DAT")["energy"]
    assert results.transport_data(copy / "2-dense-3d/C1-RCSR-bcu-f/TRANSPORT/bcu-f_transport.SEEBECK.dat")["blocks"]
    # runs made for the suite on the HPCC: a chain, a slab, a framework
    assert results.band_data(copy / "4-polymer-1d/polyacetylene/BAND/polyacetylene_band.BAND.DAT")["x"]
    assert results.dos_data(copy / "3-slab-2d/graphene/DOSS/graphene_doss.DOSS.DAT")["energy"]
    assert results.band_data(copy / "1-porous/SOD/BAND/SOD_band.BAND.DAT")["x"]
    assert results.transport_data(copy / "4-polymer-1d/polyacetylene/TRANSPORT/polyacetylene_transport.SEEBECK.DAT")["blocks"]
    # phonon band structures and DOS of the full-coverage entries (fort.25 of FREQCALC DISPERSION / PDOS)
    for rel in ("2-dense-3d/diamond/PHONON_BAND/diamond_phonon_band.f25",
                "3-slab-2d/hBN/PHONON_BAND/hBN_phonon_band.f25",
                "4-polymer-1d/polyacetylene/PHONON_BAND/polyacetylene_phonon_band.f25"):
        assert results.band_data(copy / rel)["domain"] == "phonon", rel
    for rel in ("2-dense-3d/diamond/PHONON_DOS/diamond_phonon_dos.f25", "3-slab-2d/hBN/PHONON_DOS/hBN_phonon_dos.f25",
                "4-polymer-1d/polyacetylene/PHONON_DOS/polyacetylene_phonon_dos.f25",
                "5-molecule-0d/EC/PHONON_DOS/EC_phonon_dos.f25"):
        assert results.dos_data(copy / rel)["energy"], rel


def test_mechanics_results_of_every_dimensionality(copy):
    """The ELASTIC, EOS, QHA and CPHF runs read back with `mace elastic analyze` (the GUI's Mechanical Properties
    Results): 3D in GPa and A^3, a slab or polymer in CRYSTAL's per-area or per-length units, a molecule's
    polarizability without a dielectric tensor."""
    from mace.elastic import analysis as A
    d = A.analyze_file(copy / "2-dense-3d/diamond/ELASTIC/diamond_elastic.out")
    c = d["elastic"]["matrix"]
    assert d["elastic"]["unit"] == "GPa" and 1000 < c[0][0] < 1250 and 100 < c[0][1] < 150 and 550 < c[3][3] < 650
    bm = A.analyze_file(copy / "2-dense-3d/diamond/EOS/diamond_eos.out")["eos"]["fits"]["birch_murnaghan"]
    assert 440 < bm["bulk_modulus_gpa"] < 470
    eps = A.analyze_file(copy / "2-dense-3d/diamond/CPHF/diamond_cphf.out")["dielectric"]["principal_epsilon"]
    assert 5.0 < eps[0] < 6.0
    q = A.analyze_file(copy / "4-polymer-1d/polyacetylene/QHA/polyacetylene_qha.out")
    assert q["dimensionality"] == 1 and q["qha"]["units"]["size_unit"] == "A"
    assert "L (A)" in A.report(q) and "GPa x A^2" in A.report(q)
    q = A.analyze_file(copy / "3-slab-2d/hBN/QHA/hBN_qha.out")
    assert q["dimensionality"] == 2 and q["qha"]["units"]["size_unit"] == "A^2"
    for rel, dim in (("5-molecule-0d/EC/CPHF/EC_cphf.out", 0), ("4-polymer-1d/polyacetylene/CPHF/polyacetylene_cphf.out", 1),
                     ("3-slab-2d/hBN/CPHF/hBN_cphf.out", 2)):
        po = A.analyze_file(copy / rel)
        assert po["dimensionality"] == dim and "dielectric" not in po, rel
        assert po["polarizability"]["components"][0]["alpha"] > 0, rel


def test_orbitals_of_a_spin_polarized_closed_shell_run_draw_alpha(copy):
    """CRYSTAL23 writes zero BETA coefficients for EC (SPIN, alpha = beta): ALPHA is drawn, BETA refused."""
    from mace.orbitals.dataset import OrbitalRun, OrbitalsError
    run = OrbitalRun.from_path(copy / "5-molecule-0d/EC/ORBITALS/EC_orbitals_orb.molden")
    kd = run.load(1)
    assert kd.failed_spins == ["BETA"] and kd.status == "ok"
    _, data, _ = run.grid(1, "ALPHA", [kd.homo("ALPHA")], max_points=100_000)
    assert data.size and abs(data).max() > 0
    with pytest.raises(OrbitalsError, match="BETA orbitals"):
        run.grid(1, "BETA", [1], max_points=100_000)


def test_polyacetylene_start_geometry_has_the_literature_bonds():
    sys.path.insert(0, str(REPO_ROOT / "examples" / "gui"))
    import math
    import build_suite
    lines = build_suite.polyacetylene_cif().splitlines()
    a = float(next(x for x in lines if x.startswith("_cell_length_a")).split()[1])
    b = float(next(x for x in lines if x.startswith("_cell_length_b")).split()[1])
    at = {x.split()[0]: (float(x.split()[2]) * a, float(x.split()[3]) * b, float(x.split()[4]) * b)
          for x in lines if re.match(r"^[CH]\d ", x)}
    d = lambda p, q, s=0.0: math.dist(p, (q[0] + s, q[1], q[2]))  # noqa: E731
    assert a == pytest.approx(2.46) and len(at) == 4
    assert d(at["C1"], at["C2"]) == pytest.approx(1.36, abs=1e-6)            # C=C
    assert d(at["C2"], at["C1"], a) == pytest.approx(1.44, abs=1e-6)         # C-C to the next cell
    assert d(at["C1"], at["H1"]) == pytest.approx(1.09, abs=1e-6)
    assert d(at["C2"], at["H2"]) == pytest.approx(1.09, abs=1e-6)
    assert len({p[2] for p in at.values()}) == 1                              # planar


def test_run_outcome_names_the_crystal_line_that_stopped_a_run():
    sys.path.insert(0, str(REPO_ROOT / "examples" / "gui"))
    import build_suite
    ok = " CYC  10 ETOT(AU) -1.0\n EEEEEEEEEE TERMINATION  DATE 21 10 2026\n"
    assert build_suite.run_outcome(ok, "SP") is None
    scf = " == SCF ENDED - TOO MANY CYCLES            E(AU) -7.59E+01 CYCLES 100\n EEEEEEEEEE TERMINATION\n"
    assert "TOO MANY CYCLES" in build_suite.run_outcome(scf, "FREQ")
    grid = " ERROR **** GRID3D **** USE RANGE NOT SCALE - SYSTEM HAS ZERO EXTENT IN Z\n"
    assert "GRID3D" in build_suite.run_outcome(grid, "CHARGE+POTENTIAL")
    assert build_suite.run_outcome(" CYC 1\n", "SP")                           # cut off: no normal termination
    assert build_suite.run_outcome(ok, "OPT")                                 # OPT without OPT END - CONVERGED


def test_cube_problem_flags_the_degenerate_grid_scale_writes_for_a_planar_system():
    sys.path.insert(0, str(REPO_ROOT / "examples" / "gui"))
    import build_suite
    head = " Charge density\n title\n    4    0.000000    1.396725    9.448419\n"
    good = head + "  100    0.046944    0.000000    0.000000\n  344    0.0 0.046950 0.0\n   10    0.0 0.0 0.1\n"
    bad = head + "  100    0.046944    0.000000    0.000000\n  344    0.0 0.046950 0.0\n    1    0.000000    0.000000    Infinity\n"
    assert build_suite.cube_problem(good) is None
    assert "RANGE" in build_suite.cube_problem(bad)
