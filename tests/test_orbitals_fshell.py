"""Spherical f functions and the self-check, on the real bcc Fe ORBITALS fixture.

CRYSTAL writes an f contraction coefficient of -N_xyz/sqrt(40) (Fe f shell, exp
1.598: -2.588 = -16.37/sqrt(40)), so each spherical f function is normalized
against a polynomial whose norm^2 is 40 xyz units, with the opposite sign.
"""
from pathlib import Path

import numpy as np
import pytest

from mace.orbitals.basis import bloch_overlap, orthonormality_error
from mace.orbitals.dataset import CHECK_FAIL, OrbitalRun, OrbitalsError

FE = Path(__file__).parent / "data" / "orbitals" / "fe" / "fe_f_00001_K000_real.molden"


def _fe():
    run = OrbitalRun.from_path(FE)
    return run, run.load(1)


def test_fe_all_mos_are_orthonormal_with_f_functions():
    _, kd = _fe()
    ns, SR = kd.basis.overlap_cells()
    S = bloch_overlap(ns, SR, np.zeros(3))
    for sp in kd.spins:
        cols = kd.channel(sp)
        err = orthonormality_error(kd.C[:, cols], S)
        assert err < 1e-3, (sp, err)          # 2.2e-4 measured; 0.975 with the old f normalization


def test_every_basis_function_is_normalized_on_site():
    _, kd = _fe()
    ns, SR = kd.basis.overlap_cells()
    i0 = [i for i, n in enumerate(ns) if not np.any(n)][0]
    diag = np.diag(SR[i0])
    assert np.allclose(diag, 1.0, atol=1e-3), diag[np.abs(diag - 1) > 1e-3]
    assert any(sh["l"] == 3 for sh in kd.basis.shells)


def test_f_dominated_virtual_integrates_to_one_on_a_grid():
    run, kd = _fe()
    sp = "ALPHA"
    g, data, meta = run.grid(1, sp, [25], part="re", spacing=0.2)
    norm = float((data ** 2).sum() * g.voxel_volume)
    assert abs(norm - 1.0) < 0.02, norm      # 0.025 with the old f normalization


def test_drawn_orbital_is_checked_itself(monkeypatch):
    run, kd = _fe()
    # break the coefficients of one high virtual: the load-time check (occupied + 2) cannot see it
    col = kd.channel("ALPHA")[29]
    kd.C[:, col] *= 3.0
    with pytest.raises(OrbitalsError, match="self-check"):
        run.grid(1, "ALPHA", [30], part="re", spacing=0.4)
    g, data, meta = run.grid(1, "ALPHA", [30], part="re", spacing=0.4, allow_failed=True)
    assert meta["check_drawn"] >= CHECK_FAIL
    assert meta["status"] == "fail"


def test_cube_names_of_different_mo_sets_never_collide():
    from mace.orbitals.cube import cube_name
    a = cube_name("r", 1, "ALPHA", [1, 2, 3, 4, 5, 6, 8], "abs2")
    b = cube_name("r", 1, "ALPHA", [1, 2, 3, 4, 5, 7, 8], "abs2")
    assert a != b and "MO1to8" in a
    assert cube_name("r", 1, "ALPHA", [5], "re") == "r_k1_alpha_MO5_re.cube"


def test_stale_cached_cube_is_recomputed(tmp_path):
    """A cube written before the f fix (no conv field) or for another MO set is not reused."""
    import shutil
    from mace.orbitals import api
    from mace.orbitals.basis import CONVENTION
    from mace.orbitals.cube import cube_meta, read_cube
    for f in FE.parent.iterdir():
        shutil.copy(f, tmp_path / f.name)
    run = api.open_run(tmp_path / FE.name)
    path, meta = api.make_cube(run, 1, "ALPHA", [25], part="re", quality="preview")
    assert cube_meta(read_cube(path)[5])["conv"] == str(CONVENTION)
    # simulate a 1.1.6-preview cube: same name, no conv field, wrong data
    lines = path.read_text().splitlines(True)
    lines[1] = lines[1].replace(f" conv={CONVENTION}", "")
    path.write_text("".join(lines))
    again, meta2 = api.make_cube(run, 1, "ALPHA", [25], part="re", quality="preview")
    assert again == path and not meta2["cached"]
    assert cube_meta(read_cube(path)[5])["conv"] == str(CONVENTION)
    third, meta3 = api.make_cube(run, 1, "ALPHA", [25], part="re", quality="preview")
    assert meta3["cached"]


@pytest.mark.parametrize("name,ef", [("o2/o2_c.out", None), ("h2o/h2o_c.out", None),
                                     ("fe/fe_f.out", -1.1091161e-01)])
def test_efermi_only_from_the_newk_efermi_line(name, ef):
    """The parent-SCF summary 'N. OF SCF CYCLES ... FERMI ENERGY -0.100E+01' is not a Fermi level."""
    from mace.orbitals.outfile import parse_out
    out = parse_out(FE.parent.parent / name)
    assert out.efermi == ef


def test_cli_levels_of_a_metal_use_the_fermi_level_like_the_gui(capsys):
    from mace.plotting.handlers.orbitals import print_levels
    _, kd = _fe()
    print_levels(kd, "BETA")
    out = capsys.readouterr().out
    assert "relative to the Fermi level E_F" in out and "occupation" in out
    row13 = [l for l in out.splitlines() if l.strip().startswith("13 ")][0].split()
    # beta 13 (t2g, -0.1366 Ha) against E_F -0.11091 Ha: -0.70 eV, not 0 (the HOMO zero)
    assert abs(float(row13[2]) - (-0.1366 + 0.11091161) * 27.211386) < 0.01


def _rewrite(tmp_path, src, fn):
    text = fn(src.read_text())
    out = tmp_path / src.name
    out.write_text(text)
    return out


def test_molden_5d_alone_means_spherical_f_too(tmp_path):
    from mace.orbitals.molden import read_molden
    ref = read_molden(FE)
    assert ref.spherical_d and ref.spherical_f
    p = _rewrite(tmp_path, FE, lambda t: "\n".join(l for l in t.splitlines() if l.strip().upper() != "[7F]") + "\n")
    md = read_molden(p)
    assert md.spherical_f and md.nbf == ref.nbf
    assert np.allclose(md.coef, ref.coef)


def test_molden_index_beyond_nbf_is_an_error_not_wrapped(tmp_path):
    from mace.orbitals.molden import MoldenError, read_molden
    ref = read_molden(FE)
    lines = FE.read_text().splitlines()
    # first coefficient line of the first (ALPHA) MO gets an index in the BETA range
    i = next(j for j, l in enumerate(lines) if l.strip().lower().startswith("occup"))
    while "=" in lines[i]:
        i += 1
    v = lines[i].split()[1]
    lines[i] = f"{ref.nbf + 1} {v}"
    p = tmp_path / FE.name
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(MoldenError, match="outside"):
        read_molden(p)
