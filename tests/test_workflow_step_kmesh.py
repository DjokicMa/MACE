"""A later workflow step's k-point mesh (SHRINK) over its parent's (1.1.6).

Marcus: "Of course by defualt things should be inherited but we should always be allowed to customize the  (spelling-ok: quoted)
workflow."

The k-point mesh was the last SCF setting a later step could not change. A step's ``kpoint_modifications``
({"new_k_points": ...}, in the spec, the plan and opt2d12's config file) sets its SHRINK record: "IS ISP"
(two numbers, CRYSTAL's one-line form), "IS1 IS2 IS3" per axis (the IS = 0 form, manual p. 124: a slab's
third factor is 1, a polymer's second and third) or "auto", MACE's density rule from each structure's cell.
A step without it keeps its parent's mesh, so the step after a changed one inherits the change. A mesh set
here keeps the symmetry: directions the space or layer group makes equivalent get the same (larger) factor,
since CRYSTAL stops on SHRINK BREAKS SYMMETRY. The GUESSP restart stays: the manual asks for the same
symmetry, atoms, basis functions and shells and allows other computational conditions (p. 114), and the
matrix read back is the direct-lattice P0.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT, find_data
from mace.workflow import guessp_chain
from mace.workflow.spec import SpecError, build_plan, load_spec, resolve_defaults

MACE_CLI = REPO_ROOT / "mace_cli"
AGBR = REPO_ROOT / "tests" / "data" / "opt_restart" / "tqc_agbr"          # R-3c (167), SHRINK 5 10
LOW_DIM = REPO_ROOT / "tests" / "data" / "low_dim_groups"                  # real CRYSTAL slab/polymer runs
sys.path.insert(0, str(REPO_ROOT / "Crystal_d12"))
import d12_constants as C  # noqa: E402


@pytest.fixture(autouse=True)
def _plain_site(monkeypatch, tmp_path):
    monkeypatch.setenv("MACE_SITE_CONFIG", "none")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def _env():
    return {k: v for k, v in os.environ.items()
            if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}


def _opt2d12(tmp_path, parent, config, d12_text=None):
    d12, out = Path(str(parent) + ".d12"), Path(str(parent) + ".out")
    (tmp_path / "p.d12").write_text(d12_text if d12_text is not None else d12.read_text())
    shutil.copy(out, tmp_path / "p.out")
    (tmp_path / "c.json").write_text(json.dumps(config))
    shutil.rmtree(tmp_path / "o", ignore_errors=True)
    r = subprocess.run([sys.executable, str(MACE_CLI), "opt2d12", "--out-file", "p.out", "--d12-file", "p.d12",
                        "--config-file", "c.json", "--non-interactive", "--output-dir", "o"],
                       cwd=tmp_path, capture_output=True, text=True, timeout=300, env=_env(),
                       stdin=subprocess.DEVNULL)
    decks = list((tmp_path / "o").glob("*.d12")) if (tmp_path / "o").is_dir() else []
    return r.returncode, (decks[0].read_text() if decks else None), r.stdout + r.stderr


def _shrink(deck):
    """The SHRINK record: its one or two lines."""
    lines = [line.strip() for line in deck.splitlines()]
    i = lines.index("SHRINK")
    return lines[i + 1] if not lines[i + 1].startswith("0 ") else f"{lines[i + 1]} / {lines[i + 2]}"


def _mesh(steps_key, value):
    return {steps_key: {"overrides": {"kpoint_modifications": {"new_k_points": value}}}}


# ------------------------------------------------------------------ the key
def test_the_mesh_forms_are_checked_and_normalized():
    for given, want in (("8 16", "8 16"), (" 8   8  4 ", "8 8 4"), (6, "6 6 6"), ([4, 8], "4 8"), ("AUTO", "auto"),
                        ("8", "8 8 8")):
        assert C.check_kpoint_modifications({"new_k_points": given}) == {"new_k_points": want}, given
    for bad in ("8 x", "0 16", "1 2 3 4", "", True, -2, "auto 8"):
        with pytest.raises(ValueError, match="new_k_points is"):
            C.check_kpoint_modifications({"new_k_points": bad})
    with pytest.raises(ValueError, match="only"):
        C.check_kpoint_modifications({"new_k_points": "8 16", "isp": 16})
    assert C.describe_k_mesh("8 16") == "SHRINK 8 16"
    assert C.describe_k_mesh("8 8 4") == "SHRINK factors 8 8 4 (per axis)"


@pytest.mark.parametrize("mesh, dim, group, want", [
    ((8, 8, 4), "CRYSTAL", 115, (8, 8, 4)),      # P-4m2: a = b, c free
    ((8, 6, 4), "CRYSTAL", 115, (8, 8, 4)),
    ((8, 6, 4), "CRYSTAL", 62, (8, 6, 4)),       # Pnma: no two directions related
    ((8, 6, 4), "CRYSTAL", 225, (8, 8, 8)),      # cubic
    ((8, 6, 4), "CRYSTAL", 167, (8, 8, 8)),      # R-3c: rhombohedral primitive vectors
    ((8, 6, 4), "CRYSTAL", 63, (8, 8, 8)),       # Cmcm: a centered lattice gets one factor
    ((8, 6, 4), "CRYSTAL", 1, (8, 6, 4)),
    ((8, 6, 1), "SLAB", 80, (8, 8, 1)),          # hexagonal layer group
    ((8, 6, 1), "SLAB", 37, (8, 6, 1)),          # primitive rectangular
    ((8, 6, 1), "SLAB", 47, (8, 8, 1)),          # centered rectangular: not raised to a density
    ((8, 1, 1), "POLYMER", 51, (8, 1, 1)),
])
def test_a_set_mesh_keeps_the_symmetry(mesh, dim, group, want):
    assert C.symmetric_k_points(mesh, dim, group) == want


# ------------------------------------------------------------------ opt2d12's deck: 3D, slab, polymer
def test_3d_one_line_per_axis_and_auto(tmp_path):
    assert _shrink(Path(str(AGBR) + ".d12").read_text()) == "5 10"
    code, deck, log = _opt2d12(tmp_path, AGBR, {"calculation_type": "SP", "kpoint_modifications":
                                                {"new_k_points": "8 16"}})
    assert code == 0 and deck, log[-3000:]
    assert _shrink(deck) == "8 16" and "k-points: 8 16 (this step's kpoint_modifications)" in log
    # an insulator's ISP = IS, as given
    code, deck, _ = _opt2d12(tmp_path, AGBR, {"calculation_type": "SP", "kpoint_modifications": {"new_k_points": "6 6"}})
    assert _shrink(deck) == "6 6"
    # R-3c: per-axis factors the rhombohedral vectors cannot have differ are raised to agree, with a note
    code, deck, log = _opt2d12(tmp_path, AGBR, {"calculation_type": "SP", "kpoint_modifications":
                                                {"new_k_points": "6 6 4"}})
    assert code == 0 and _shrink(deck) == "6 12"
    assert "Using k-points 6 6 6 (was 6 6 4) so the mesh keeps space group 167's symmetry" in log
    # auto: MACE's own density rule from the cell (the parent's 5 10 here)
    code, deck, log = _opt2d12(tmp_path, AGBR, {"calculation_type": "SP", "kpoint_modifications":
                                                {"new_k_points": "auto"}})
    assert code == 0 and _shrink(deck) == "5 10" and "automatic mesh from this structure's cell" in log


@pytest.mark.parametrize("value, want", [("8 8 4", "0 16 / 8 8 4"), ("8 6 4", "0 16 / 8 8 4"), ("8 16", "8 16")])
def test_3d_per_axis_mesh_is_written_directional(tmp_path, value, want):
    """P-4m2 CA bulk: CRYSTAL23 ran its own anisotropic 0 60 / 30 30 10 with symmetry-adapted Bloch functions
    enabled, so an anisotropic mesh needs no NOSYMADA; a and b stay equal."""
    out = find_data("OPT/3,4^2T1-CA_BULK_OPTGEOM_TZ.out", "SYMMETRY ADAPTION OF THE BLOCH FUNCTIONS ENABLED")
    assert _shrink(out.with_suffix(".d12").read_text()) == "0 60 / 30 30 10"
    parent = out.with_suffix("")
    code, deck, log = _opt2d12(tmp_path, parent, {"calculation_type": "SP",
                                                  "kpoint_modifications": {"new_k_points": value}})
    assert code == 0 and deck, log[-3000:]
    assert _shrink(deck) == want


@pytest.mark.parametrize("parent, value, want", [
    ("graphene_lg80", None, "0 24 / 12 12 1"),            # inherited
    ("graphene_lg80", "6 12", "0 12 / 6 6 1"),            # IS ISP on a slab
    ("graphene_lg80", "8 6 1", "0 16 / 8 8 1"),           # hexagonal: a = b
    ("graphene_lg37", "8 6 1", "0 16 / 8 6 1"),           # primitive rectangular: kept
    ("graphene_lg37", "8 6 9", "0 16 / 8 6 1"),           # the non-periodic factor is 1
    ("polyyne_rg51", None, "0 24 / 12 1 1"),
    ("polyyne_rg51", "6 12", "0 12 / 6 1 1"),
    ("polyyne_rg51", "9 9 9", "0 18 / 9 1 1"),
    ("polyyne_rg51", "auto", "0 32 / 16 1 1"),            # 2.56 A: 16 * 2.56 within 40-80 A
])
def test_slab_and_polymer_meshes(tmp_path, parent, value, want):
    cfg = {"calculation_type": "SP"}
    if value:
        cfg["kpoint_modifications"] = {"new_k_points": value}
    code, deck, log = _opt2d12(tmp_path, LOW_DIM / parent, cfg)
    assert code == 0 and deck, log[-3000:]
    assert _shrink(deck) == want


def test_a_bad_mesh_writes_nothing_and_says_why(tmp_path):
    code, deck, log = _opt2d12(tmp_path, AGBR, {"calculation_type": "SP", "kpoint_modifications":
                                                {"new_k_points": "8 x"}})
    assert code != 0 and deck is None and "kpoint_modifications.new_k_points is" in log


def test_k_points_for_all_files_takes_auto_too(tmp_path):
    code, deck, log = _opt2d12(tmp_path, LOW_DIM / "polyyne_rg51", {"calculation_type": "SP",
                                                                     "k_points_for_all_files": "auto"})
    assert code == 0 and _shrink(deck) == "0 32 / 16 1 1"


# ------------------------------------------------------------------ GUESSP across a k-point change
def test_a_changed_mesh_keeps_the_guessp_restart(tmp_path):
    parent = find_data("OPT/1_dia_opt_rev1.out", "SPIN POLARIZED DFT").with_suffix("")
    text = Path(str(parent) + ".d12").read_text().replace("\nSCFDIR\n", "\nGUESSP\nSCFDIR\n")
    code, deck, log = _opt2d12(tmp_path, parent, {"calculation_type": "SP",
                                                  "kpoint_modifications": {"new_k_points": "9 18"}}, d12_text=text)
    assert code == 0 and deck, log[-3000:]
    recs = [line.strip() for line in deck.splitlines()]
    assert _shrink(deck) == "9 18" and "GUESSP" in recs and "fresh SCF guess" not in recs[0]
    # the engine's restart check (staging the parent's fort.9 as fort.20) does not compare the mesh
    assert _shrink(text) != "9 18"
    assert guessp_chain.refusal(text, deck) is None


# ------------------------------------------------------------------ spec -> plan -> engine config
def _cif_spec(folder, steps, seq=("OPT", "OPT2", "SP")):
    shutil.copy(find_data("CIFs/1_dia_opt_rev1.cif"), folder / "A.cif")
    p = folder / "spec.json"
    p.write_text(json.dumps({"version": 1, "inputs": {"type": "cif", "files": ["A.cif"]},
                             "sequence": list(seq), "steps": steps}))
    return p


def test_spec_round_trips_to_the_plan_and_the_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = load_spec(_cif_spec(tmp_path, _mesh("OPT2_2", " 9  18 ")))
    assert spec["steps"]["OPT2_2"]["overrides"]["kpoint_modifications"] == {"new_k_points": "9 18"}
    _, wid, plan = build_plan(spec, tmp_path)
    assert plan["step_configurations"]["OPT2_2"]["kpoint_modifications"] == {"new_k_points": "9 18"}
    # the SP after it keeps its parent's (OPT2's) mesh
    assert "kpoint_modifications" not in plan["step_configurations"]["SP_3"]
    out = resolve_defaults(spec, tmp_path)
    assert out["OPT2_2"]["settings"]["kpoint_modifications"] == {"new_k_points": "9 18"}
    assert out["OPT2_2"]["warnings"] == []
    for bad, msg in (("8 x", "new_k_points is"), ({"IS": 8}, "only")):
        ov = {"kpoint_modifications": bad if isinstance(bad, dict) else {"new_k_points": bad}}
        with pytest.raises(SpecError, match=msg):
            load_spec(_cif_spec(tmp_path, {"OPT2_2": {"overrides": ov}}))


def test_the_planners_advanced_setup_asks_for_the_mesh(tmp_path, monkeypatch):
    from mace.workflow.planner import WorkflowPlanner
    planner = WorkflowPlanner(work_dir=tmp_path)
    for replies, want in ((["8 16"], {"new_k_points": "8 16"}), (["auto"], {"new_k_points": "auto"}),
                          (["8 x", "6 6 2"], {"new_k_points": "6 6 2"}), ([""], None)):
        it = iter(replies)
        monkeypatch.setattr("builtins.input", lambda prompt="", it=it: next(it))
        cfg = {}
        planner._ask_step_kpoints(cfg)
        assert cfg.get("kpoint_modifications") == want, (replies, cfg)
    # the Advanced SP setup asks it, beside the spin question
    monkeypatch.setattr(planner, "_get_basic_sp_config", lambda: {})
    monkeypatch.setattr("mace.workflow.planner.yes_no_prompt", lambda *a, **k: False)
    monkeypatch.setattr(planner, "_ask_step_spin", lambda config, **k: None)
    monkeypatch.setattr("builtins.input", lambda prompt="": "4 8")
    assert planner._get_advanced_sp_config()["kpoint_modifications"] == {"new_k_points": "4 8"}


# ------------------------------------------------------------------ the real CLI path
def _ok(p):
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-4000:]
    return p


def test_mace_workflow_spec_changes_shrink_on_opt2_and_the_steps_after_inherit_it(tmp_path):
    """mace workflow --spec (OPT -> OPT2 with SHRINK 9 18 -> OPT3 -> SP) on a corpus CIF, launched with fake
    SLURM, then the engine writes the later steps from stand-in outputs (the GUI preview's projection). OPT3's
    work folder also holds the first OPT's deck (the engine copies it for the symmetry): it still inherits
    OPT2's mesh."""
    pytest.importorskip("ase", reason="the workflow starts from a CIF, which mace convert reads with ase")
    from fake_slurm import FakeSlurm
    from test_array_default_path_golden import _mace
    fake = FakeSlurm(tmp_path / "fs")
    env = dict(fake.env, HOME=str(tmp_path / "home"), USER="tester", XDG_CACHE_HOME=str(tmp_path / "cache"),
               MACE_SITE_CONFIG="none")
    env.pop("MACE_HOME", None)
    env.pop("PYTHONPATH", None)
    (tmp_path / "home").mkdir(exist_ok=True)
    root = tmp_path / "copy"
    folder = root / "cifs"
    folder.mkdir(parents=True)
    (root / ".mace_preview_mirror").write_text("preview\n")
    _cif_spec(folder, _mesh("OPT2_2", "9 18"), seq=("OPT", "OPT2", "OPT3", "SP"))
    check = json.loads(_ok(_mace(folder, env, "workflow", "--spec", "spec.json", "--json")).stdout)
    assert check["steps"]["OPT2_2"]["settings"]["kpoint_modifications"] == {"new_k_points": "9 18"}
    w = json.loads(_ok(_mace(folder, env, "workflow", "--spec", "spec.json", "--write-plan", "--json")).stdout)
    _ok(_mace(folder, env, *w["launch"][1:]))
    first = [p for p in folder.rglob("*.d12") if "opt" in p.name.lower()]
    assert first and _shrink(first[0].read_text()) != "9 18"
    p = _ok(subprocess.run([sys.executable, "-m", "mace.workflow.projection", "--folder", str(folder),
                            "--workflow-id", w["workflow_id"], "--json"], cwd=str(folder),
                           env=dict(env, PYTHONPATH=str(REPO_ROOT)), capture_output=True, text=True, timeout=900))
    steps = {s["step"]: s for s in json.loads(p.stdout)["steps"]}
    decks = {k: Path(steps[k]["input_file"]).read_text() for k in ("OPT2_2", "OPT3_3", "SP_4")}
    assert {k: _shrink(v) for k, v in decks.items()} == {"OPT2_2": "9 18", "OPT3_3": "9 18", "SP_4": "9 18"}
    assert "OPTGEOM" in decks["OPT2_2"] and "OPTGEOM" in decks["OPT3_3"]
