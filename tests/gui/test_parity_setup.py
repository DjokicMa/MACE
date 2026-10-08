"""A molecule output newly openable in the GUI (plan E2-g): Setup Next
Calculation on it takes the path-mode opt2d12 route the CLI takes on the
.out, never the session source's "not periodic in 3D" refusal, and the deck
is byte for byte the manual CLI's on the same copy with the same answers,
MOLECULE kept (a follow-up preserves its parent). A real corpus molecule OPT
(with its .d12) is copied under a neutral name; skipped without the corpus."""
import shutil

import pytest

import guiharness as H
import setup_routes as R
import test_setup_routes as T

pytest.importorskip("numpy")
pytest.importorskip("ase")
SRC = H.REPO_ROOT / "test" / "OPT" / "1LiFSI-1DEC-conf3_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized"


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    if not SRC.with_suffix(".out").is_file():
        pytest.skip("test/ corpus not present (gitignored)")
    tmp = tmp_path_factory.mktemp("parity_setup")
    root = H.examples_copy(tmp)
    (root / "mol").mkdir()
    for ext in (".out", ".d12"):
        shutil.copyfile(SRC.with_suffix(ext), root / "mol" / f"mol_opt{ext}")
    (root / "mol" / "mol_opt.f9").write_text("wavefunction placeholder\n")
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1")
    yield root, base
    proc.terminate()
    proc.wait(timeout=15)


@pytest.mark.parametrize("calc", ["SP", "FREQ"])
def test_molecule_output_setup_equals_the_manual_cli(served, calc, tmp_path):
    root, base = served
    row = R.Row(900, "mol/mol_opt.out", "opt2d12", calc, R.YES, keep_parent=True)
    r, s, jg, gname, g = T.gui_route(root, base, row, tmp_path)
    jm, mname, m = T.manual_route(root, row, tmp_path)
    assert (r["deck_name"], r["deck"]) == (gname, g)
    assert gname == mname and g == m
    lines = [x.strip() for x in g.splitlines()]
    assert "MOLECULE" in lines[:4], lines[:6]
    assert "SPIN" in lines, "SPIN is the CLI's default"
