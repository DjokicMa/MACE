"""A generated deck keeps the parent's SHRINK mesh.

Two drifts, both on every opt2d12/engine path:

* The one-line ``SHRINK / IS ISP`` form (the most common in the corpus) was
  not understood, so the mesh was regenerated from the cell: a parent's
  ``5 10`` became ``7 14`` and ``8 16`` became ``9 18``.
* A non-P1 parent with an anisotropic ``0 ISP / ka kb kc`` mesh was made
  uniform: ``0 60 / 30 30 10`` became ``30 60``, i.e. 30 along c where the
  parent used 10.

The corpus tests run the real ``mace_cli opt2d12`` and skip when ``test/`` is
absent. The writer tests pin the forms no corpus parent reaches.
"""
import io
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, TEST_DATA
from d12_writer import write_scf_section

MACE_CLI = REPO_ROOT / "mace_cli"

AG1BR1 = "OPT/Ag1Br1_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized"
AG1CL3 = "OPT/Ag1Cl3_sym_CRYSTAL_OPT_symm_PBE-D3_POB-TZVP-REV2_opt_B3LYP-D3-D3_optimized"
T1CA = "OPT/3,4^2T1-CA_BULK_OPTGEOM_TZ"
TI9SE2 = "OPT/Ti9Se2_mp-619905_sg55_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized"


def _copy_parent(stem, tmp_path):
    src = TEST_DATA / f"{stem}.out"
    if not src.exists():
        pytest.skip("test/ corpus not present (gitignored, ~12GB)")
    shutil.copy(src, tmp_path)
    shutil.copy(src.with_suffix(".d12"), tmp_path)
    return src.stem


def _opt2d12(tmp_path, name, calc_type):
    (tmp_path / "cfg.json").write_text(json.dumps({"calculation_type": calc_type}))
    result = subprocess.run(
        [sys.executable, str(MACE_CLI), "opt2d12", "--out-file", f"{name}.out",
         "--d12-file", f"{name}.d12", "--config-file", "cfg.json"],
        cwd=tmp_path, input="y\n" + "\n" * 17, capture_output=True, text=True,
        timeout=300)
    assert result.returncode == 0, (result.stdout + result.stderr)[-1500:]
    decks = [p for p in tmp_path.glob("*.d12") if p.name != f"{name}.d12"]
    assert len(decks) == 1, sorted(p.name for p in tmp_path.iterdir())
    return decks[0].read_text().splitlines()


def _shrink(lines):
    i = lines.index("SHRINK")
    first = lines[i + 1].split()
    if first[0] == "0":
        return [first, lines[i + 2].split()]
    return [first]


@pytest.mark.parametrize("stem,expected", [
    (AG1BR1, [["5", "10"]]),
    (AG1CL3, [["8", "16"]]),
    (T1CA, [["0", "60"], ["30", "30", "10"]]),
])
@pytest.mark.parametrize("calc_type", ["SP", "FREQ", "OPT"])
def test_child_keeps_parent_shrink(tmp_path, stem, expected, calc_type):
    name = _copy_parent(stem, tmp_path)
    assert _shrink((tmp_path / f"{name}.d12").read_text().splitlines()) == expected
    assert _shrink(_opt2d12(tmp_path, name, calc_type)) == expected


def test_orthorhombic_directional_mesh_keeps_axis_order(tmp_path):
    """Directional counts refer to the child's own reciprocal vectors, so they
    are only preserved correctly if the child writes the parent's cell in the
    same setting and axis order. Pin that on a real sg-55 (Pbam) parent with an
    anisotropic mesh edited in."""
    name = _copy_parent(TI9SE2, tmp_path)
    d12 = tmp_path / f"{name}.d12"
    lines = d12.read_text().splitlines()
    i = lines.index("SHRINK")
    lines[i + 1:i + 2] = ["0 32", "10 4 16"]
    d12.write_text("\n".join(lines) + "\n")

    child = _opt2d12(tmp_path, name, "SP")
    assert _shrink(child) == [["0", "32"], ["10", "4", "16"]]

    parent_cell = [float(x) for x in lines[4].split()]
    child_cell = [float(x) for x in child[4].split()]
    assert child[3].strip() == lines[3].strip() == "55"
    # Same axis order: the longest (b) axis stays second, the shortest (c) third.
    order = lambda c: sorted(range(3), key=lambda k: c[k])
    assert order(child_cell) == order(parent_cell)
    for p, c in zip(parent_cell, child_cell):
        assert c == pytest.approx(p, rel=0.05)


# --------------------------------------------------------------- writer units

def _write(k_points, dimensionality="CRYSTAL", spacegroup=225, **kw):
    buf = io.StringIO()
    write_scf_section(buf, {"TOLINTEG": "7 7 7 7 14", "TOLDEE": 7}, k_points,
                      dimensionality, False, 0.01, "DIIS", 800, 30, 2,
                      spacegroup, **kw)
    text = buf.getvalue()
    m = re.search(r"^SHRINK\n(.*?)\n(?:(\d+ \d+ \d+)\n)?", text, re.M)
    return [m.group(1)] + ([m.group(2)] if m.group(2) else [])


def test_writer_default_still_makes_non_p1_mesh_uniform():
    # A mesh not taken from the parent (config from another material, or
    # generated) keeps the old behaviour.
    assert _write((30, 30, 10)) == ["30 60"]


def test_writer_preserves_directional_when_asked():
    assert _write((30, 30, 10), preserve_directional=True) == ["0 60", "30 30 10"]
    # A uniform mesh keeps the one-line form either way.
    assert _write((12, 12, 12), preserve_directional=True) == ["12 24"]


def test_writer_carries_non_default_isp():
    assert _write((8, 8, 8), shrink_isp=8) == ["8 8"]
    assert _write((8, 8, 8)) == ["8 16"]


@pytest.mark.parametrize("dim,sg,expected", [
    ("CRYSTAL", 1, ["0 10", "5 5 5"]),
    ("SLAB", 1, ["0 10", "5 5 1"]),
    ("SLAB", 11, ["0 10", "5 5 1"]),
    ("POLYMER", 1, ["0 10", "5 1 1"]),
])
def test_two_number_parent_on_p1_or_low_dim_writes_same_mesh(dim, sg, expected):
    """A two-number parent's (5,5,5) on P1/SLAB/POLYMER comes out in the
    directional form: same mesh, different bytes. No corpus parent does this."""
    assert _write((5, 5, 5), dimensionality=dim, spacegroup=sg,
                  preserve_directional=True) == expected


# ------------------------------------------------ write_d12_file k-point input

def _k_passed(monkeypatch, tmp_path, settings, parent_k_points):
    import CRYSTALOptToD12 as M
    from d12_parsers import CrystalOutputParser
    from conftest import find_data

    geo = CrystalOutputParser(str(find_data("OPT/1_dia_opt_rev1.out"))).parse()
    s = dict(geo)
    s.update(settings)
    captured = {}

    class _Stop(Exception):
        pass

    def _spy(f, tol, k, *args, **kwargs):
        captured.update(k=k, **kwargs)
        raise _Stop()

    monkeypatch.setattr(M, "write_scf_section", _spy)
    with pytest.raises(_Stop):
        M.write_d12_file(str(tmp_path / "child.d12"), geo, s, parent_k_points=parent_k_points)
    return captured


def test_two_number_string_is_understood(monkeypatch, tmp_path):
    got = _k_passed(monkeypatch, tmp_path, {"k_points": "5 10"}, "5 10")
    assert got["k"] == (5, 5, 5)
    assert got["shrink_isp"] is None          # 10 == 2*5: default form


def test_two_number_string_with_unusual_isp(monkeypatch, tmp_path):
    got = _k_passed(monkeypatch, tmp_path, {"k_points": "8 8"}, "8 8")
    assert got["k"] == (8, 8, 8)
    assert got["shrink_isp"] == 8


def test_anisotropic_parent_mesh_is_not_made_uniform(monkeypatch, tmp_path):
    got = _k_passed(monkeypatch, tmp_path, {"k_points": "30 30 10", "spacegroup": 115},
                    "30 30 10")
    assert got["k"] == (30, 30, 10)
    assert got["preserve_directional"] is True


def test_mesh_not_from_this_parent_is_still_made_uniform(monkeypatch, tmp_path):
    """k_points that arrive from a config saved for another material (or with no
    parent deck at all) do not switch on the directional form."""
    got = _k_passed(monkeypatch, tmp_path, {"k_points": "30 30 10", "spacegroup": 115},
                    "12 24")
    assert got["k"] == (30, 30, 30)
    assert got["preserve_directional"] is False
    got = _k_passed(monkeypatch, tmp_path, {"k_points": "30 30 10", "spacegroup": 115}, None)
    assert got["preserve_directional"] is False


def test_two_number_mesh_not_from_this_parent_is_regenerated(monkeypatch, tmp_path):
    """A two-number 'IS ISP' mesh from a saved config or --shared-settings was
    written onto whatever material it was applied to; it is regenerated from
    this material's cell, as for any mesh that is not the parent's own."""
    import CRYSTALOptToD12 as M
    from d12_parsers import CrystalOutputParser
    from conftest import find_data

    geo = CrystalOutputParser(str(find_data("OPT/1_dia_opt_rev1.out"))).parse()
    a, b, c = [float(x) for x in geo["conventional_cell"][:3]]
    from_cell = M.generate_k_points(a, b, c, "CRYSTAL", 227)
    assert from_cell != (5, 5, 5)
    for parent in ("12 24", None):
        got = _k_passed(monkeypatch, tmp_path, {"k_points": "5 10", "spacegroup": 227}, parent)
        assert got["k"] == from_cell and got["shrink_isp"] is None


def test_config_from_another_material_does_not_impose_its_mesh(tmp_path):
    """--save-options on Ag1Br1 (SHRINK 5 10), then --config-file on Ti9Se2
    (SHRINK 15 30): Ti9Se2 used to get 5 10."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    name_a = _copy_parent(AG1BR1, a)
    name_b = _copy_parent(TI9SE2, b)
    env = dict(os.environ, NO_COLOR="1")
    save = subprocess.run(
        [sys.executable, str(MACE_CLI), "opt2d12", "--out-file", f"{name_a}.out",
         "--d12-file", f"{name_a}.d12", "--non-interactive", "--calc-type", "SP",
         "--save-options", "--options-file", str(tmp_path / "optsA.json")],
        cwd=a, input="", capture_output=True, text=True, timeout=300, env=env)
    saved = json.loads((tmp_path / "optsA.json").read_text())
    assert saved.get("k_points") == "5 10", (save.stdout + save.stderr)[-1500:]
    result = subprocess.run(
        [sys.executable, str(MACE_CLI), "opt2d12", "--out-file", f"{name_b}.out",
         "--d12-file", f"{name_b}.d12", "--config-file", str(tmp_path / "optsA.json"),
         "--non-interactive"],
        cwd=b, input="", capture_output=True, text=True, timeout=300, env=env)
    assert result.returncode == 0, (result.stdout + result.stderr)[-1500:]
    decks = [p for p in b.glob("*.d12") if p.name != f"{name_b}.d12"]
    assert len(decks) == 1
    assert _shrink(decks[0].read_text().splitlines()) == [["15", "30"]]
