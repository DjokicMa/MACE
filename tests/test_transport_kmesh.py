"""TRANSPORT's own k mesh (1.1.6): BOLTZTRA's NEWK is chosen by a k spacing, not the SCF's SHRINK.

The transport distribution sums band velocities over the NEWK k points (CRYSTAL23 manual, eq. 14.4,
p.314). The SCF's mesh converges total energies, not transport: with diamond's IS 12 the TDF was so
spiky that the 300 K Lorenz number swung from 0.02 to 24 x 1e-8 W Ohm/K^2 across the valence band.
Crystal_d3/d3_config.transport_mesh takes IS = the largest ceil(|b_i| / spacing) over the periodic
reciprocal vectors (2*pi included), never below the parent's IS, and ISP = 2 IS. Presets: Screening
(the parent's mesh), Standard (TRANSPORT_K_STANDARD, the default, converged for the example suite's
five materials on the HPCC), Fine; another spacing; an IS as given. The presets depend on the periodic dimensionality (3D/2D/1D). One resolver for opt2d3 (its
question, --transport-k, --transport-is, --non-interactive, a settings file), Setup and the planner.
"""
import builtins
import json
import lzma
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "Crystal_d3"))
import d3_config as C  # noqa: E402

pytest.importorskip("numpy")

SUITE = REPO_ROOT / "examples" / "suite"
DIAMOND_B = 3.070            # |b| of diamond's primitive cell (a = 3.5448 A), 1/A


def _put(src: Path, dst: Path):
    dst.write_bytes(lzma.decompress(src.with_name(src.name + ".xz").read_bytes()))


@pytest.fixture()
def diamond(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _put(SUITE / "2-dense-3d" / "diamond" / "SP" / "diamond_sp.out", run / "diamond_sp.out")
    (run / "diamond_sp.d12").write_text((SUITE / "2-dense-3d" / "diamond" / "SP" / "diamond_sp.d12").read_text())
    for n in ("diamond_band.out", "diamond_band.BAND.DAT"):         # E_F for MURANGE (a spin-locked SP)
        _put(SUITE / "2-dense-3d" / "diamond" / "BAND" / n, run / n)
    (run / "diamond_sp.f9").write_text("wavefunction placeholder\n")
    return run / "diamond_sp.out"


def _newk(parent, *args, answers=""):
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    env["MACE_NO_BANNER"] = "1"
    out = parent.parent / f"o{len(list(parent.parent.glob('o*')))}"
    r = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "opt2d3", "-i", str(parent), "-t", "TRANSPORT",
                        "-o", str(out), *args], cwd=str(parent.parent), input=answers, capture_output=True, text=True,
                       env=env, timeout=180)
    decks = list(out.glob("*.d3"))
    if not decks:
        return r, None
    lines = decks[0].read_text().splitlines()
    return r, lines[lines.index("NEWK") + 1]


def test_the_resolver():
    recip = [DIAMOND_B] * 3
    std = C.TRANSPORT_K_STANDARD[3]
    n = math.ceil(DIAMOND_B / std)
    assert C.transport_mesh({}, 12, recip)[:2] == (n, 2 * n)                       # Standard by default
    assert C.transport_mesh({"k_resolution": "screening"}, 12, recip)[:2] == (12, 24)
    assert C.transport_mesh({"k_resolution": "fine"}, 12, recip)[0] == math.ceil(DIAMOND_B / C.TRANSPORT_K_FINE[3])
    # 2D and 1D: their own presets (a slab's and a polymer's mesh costs seconds where a crystal's costs an hour)
    assert C.transport_mesh({}, 12, [2.908, 2.908])[0] == math.ceil(2.908 / C.TRANSPORT_K_STANDARD[2])
    assert C.transport_mesh({}, 18, [2.555])[0] == math.ceil(2.555 / C.TRANSPORT_K_STANDARD[1])
    assert C.transport_mesh({"k_resolution": "spacing", "k_spacing": 1.0}, 12, recip)[:2] == (12, 24)   # never coarser
    assert C.transport_mesh({"k_resolution": "is", "newk_is": 7}, 12, recip)[:2] == (7, 14)             # as given
    assert "below the parent's IS 12" in C.transport_mesh({"k_resolution": "is", "newk_is": 7}, 12, recip)[2]
    assert C.transport_mesh({}, 12, None)[:2] == (12, 24)                                             # no cell
    # a slab and a polymer: only the periodic directions count
    assert C.transport_mesh({"k_resolution": "spacing", "k_spacing": 0.1}, 12, [2.9, 2.9])[0] == 29
    assert C.transport_mesh({"k_resolution": "spacing", "k_spacing": 0.1}, 12, [2.555])[0] == 26
    assert C.TRANSPORT_K_PRESETS["screening"] is None
    assert C.TRANSPORT_K_STANDARD == {3: 0.064, 2: 0.015, 1: 0.009}
    assert all(C.TRANSPORT_K_FINE[d] < C.TRANSPORT_K_STANDARD[d] for d in (1, 2, 3))


def test_reciprocal_lengths_of_the_suites_outputs(tmp_path):
    for entry, dim, b in (("2-dense-3d/diamond", 3, 3.070), ("3-slab-2d/hBN", 2, 2.908),
                          ("4-polymer-1d/polyacetylene", 1, 2.555), ("1-porous/SOD", 3, 1.016)):
        name = entry.split("/")[1]
        out = tmp_path / f"{name}_sp.out"
        _put(SUITE / entry / "SP" / f"{name}_sp.out", out)
        got = C.reciprocal_lengths(str(out))
        assert len(got) == dim and got == pytest.approx([b] * dim, abs=2e-3), entry


def test_opt2d3_routes_write_the_same_mesh(diamond):
    n = math.ceil(DIAMOND_B / C.TRANSPORT_K_STANDARD[3])
    r, enter = _newk(diamond, answers="\n" * 40)
    assert enter == f"{n} {2 * n}", r.stdout[-1500:]
    assert "Select k-point mesh (1-5) [2]: " in r.stdout and f"2: Standard - {C.TRANSPORT_K_STANDARD[3]:g} 1/Å k spacing, IS {n}" in r.stdout
    assert f"TRANSPORT k mesh: NEWK IS {n} ISP {2 * n}" in r.stdout
    assert _newk(diamond, "--non-interactive")[1] == enter
    assert _newk(diamond, "--transport-k", "screening", "--non-interactive")[1] == "12 24"
    assert _newk(diamond, "--transport-k", "0.2", "--non-interactive")[1] == "16 32"
    assert _newk(diamond, "--transport-is", "20", "--non-interactive")[1] == "20 40"
    r, _ = _newk(diamond, "--transport-k", "0.1", "--transport-is", "20")
    assert r.returncode == 2 and "not both" in r.stderr
    # the menu's other answers: 3 temperatures, the μ reference and 3 μ, 3 TDF, then the mesh
    answers = "\n" * 10
    assert _newk(diamond, answers=answers + "1\n" + "\n" * 30)[1] == "12 24"
    assert _newk(diamond, answers=answers + "4\n0.2\n" + "\n" * 30)[1] == "16 32"
    assert _newk(diamond, answers=answers + "5\n30\n" + "\n" * 30)[1] == "30 60"


def test_settings_file_and_workflow_default(diamond, tmp_path):
    from mace.workflow.common.constants import opt2d3_default_transport_config
    n = math.ceil(DIAMOND_B / C.TRANSPORT_K_STANDARD[3])
    cfg = tmp_path / "t.json"
    for conf, want in ((opt2d3_default_transport_config(), f"{n} {2 * n}"),
                       ({**opt2d3_default_transport_config(), "k_resolution": "screening"}, "12 24"),
                       ({k: v for k, v in opt2d3_default_transport_config().items() if k != "k_resolution"},
                        f"{n} {2 * n}")):                           # a settings file from before: Standard
        cfg.write_text(json.dumps({"version": "1.0", "type": "d3_configuration", "calculation_type": "TRANSPORT",
                                   "configuration": conf}))
        assert _newk(diamond, "--config-file", str(cfg))[1] == want
    ok, errors = C.validate_d3_config({"calculation_type": "TRANSPORT", "temperature_range": [100, 800, 50],
                                       "k_resolution": "coarse"})
    assert not ok and "k_resolution" in errors[0]


def test_planner_asks_the_mesh(tmp_path, monkeypatch):
    from mace.workflow.planner import WorkflowPlanner
    for answers, want in (([], {"k_resolution": "standard"}), (["1"], {"k_resolution": "screening"}),
                          (["4", "0.1"], {"k_resolution": "spacing", "k_spacing": 0.1}),
                          (["5", "30"], {"k_resolution": "is", "newk_is": 30})):
        it = iter(answers)
        monkeypatch.setattr(builtins, "input", lambda prompt="", it=it: next(it, ""))
        assert WorkflowPlanner(work_dir=tmp_path)._ask_transport_k() == want
    it2 = iter([])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(it2, ""))
    assert WorkflowPlanner(work_dir=tmp_path)._get_advanced_d3_config("TRANSPORT")["k_resolution"] == "standard"


def _suite_run(tmp_path, name="diamond", entry="2-dense-3d/diamond"):
    run = tmp_path / "t"
    run.mkdir()
    for f in sorted((SUITE / entry / "TRANSPORT").iterdir()):
        out = f.name[:-3] if f.name.endswith(".xz") else f.name
        (run / out).write_bytes(lzma.decompress(f.read_bytes()) if f.name.endswith(".xz") else f.read_bytes())
    return run / f"{name}_transport.SEEBECK.DAT"


@pytest.mark.parametrize("entry", ["2-dense-3d/diamond", "1-porous/SOD", "3-slab-2d/hBN", "3-slab-2d/graphene",
                                   "4-polymer-1d/polyacetylene"])
def test_the_suites_runs_are_at_the_standard_mesh_and_not_flagged(tmp_path, entry):
    from mace.utils import transport_mesh as M
    seebeck = _suite_run(tmp_path, entry.split("/")[1], entry)
    used, std = M.run_mesh(M.run_out(seebeck))
    assert used == std and M.mesh_note(seebeck) is None


def test_a_coarse_mesh_is_flagged_in_the_plot_and_the_database(tmp_path):
    """The suite's diamond run with its NEWK line made the SCF's IS 12 (the mesh 1.1.6 wrote before):
    the transport plot (GUI and mace plotting) and the extractor say the mesh looks too coarse."""
    from mace.utils import transport_mesh as M
    from mace.gui import results
    seebeck = _suite_run(tmp_path)
    out = seebeck.with_name("diamond_transport.out")
    text = out.read_text()
    i = text.rindex("CAPPA:IS1 48;IS2 48;IS3 48")
    out.write_text(text[:i] + "CAPPA:IS1 12;IS2 12;IS3 12" + text[i + len("CAPPA:IS1 48;IS2 48;IS3 48"):])
    note = M.mesh_note(seebeck)
    assert note.startswith("the k mesh looks too coarse for transport: its NEWK mesh is IS 12, well below the "
                           "Standard mesh of this cell (IS 48)")
    d = results.transport_data(seebeck, "lorenz")
    assert d["mesh_note"] == note and note in d["notes"]
    from mace.utils.property_extractor import CrystalPropertyExtractor
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        props = CrystalPropertyExtractor(enable_tracking=False).extract_all_properties(out)
    assert props["transport_mesh_note"] == note and props["transport"]["mesh_note"] == note
    assert f"Note: diamond_transport.out: {note}" in buf.getvalue()
    # the CLI's transport plot prints it
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    r = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "--no-banner", "plotting", "--transport",
                        str(seebeck), "--quantity", "lorenz"], cwd=str(tmp_path), capture_output=True, text=True,
                       env={**env, "MPLBACKEND": "Agg"}, timeout=600)
    assert r.returncode == 0 and "the k mesh looks too coarse for transport" in r.stdout, r.stdout[-1500:]
