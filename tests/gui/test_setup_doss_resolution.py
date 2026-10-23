"""Setup's DOSS Resolution row is opt2d3's energy-resolution menu (Crystal_d3/d3_config).

Screening / Standard / Fine / another step / a number of points, each option naming the points and
step it gives for the output in front; the deck Setup previews is the one `mace opt2d3` writes
with the same choice as a flag (--doss-step / --doss-points).
"""
import json
import os
import re
import subprocess
import sys

import pytest

import setup_routes as R
from test_setup_api import _run

pytest.importorskip("numpy")


def _npt(deck):
    lines = deck.splitlines()
    return int(lines[lines.index("DOSS") + 1].split()[1])


def _cli(folder, *flags):
    from conftest import REPO_ROOT
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    env["MACE_NO_BANNER"] = "1"
    out = folder / f"cli{len(list(folder.glob('cli*')))}"
    r = subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "opt2d3", "-i", "Si_sp.out", "-t", "DOSS",
                        "-o", str(out), "--non-interactive", *flags], cwd=str(folder), capture_output=True, text=True,
                       env=env, timeout=180)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    return _npt(next(out.glob("*.d3")).read_text())


@pytest.mark.parametrize("answers, flags, resolution", [
    ({}, (), "standard"),
    ({"doss_resolution": "Screening"}, ("--doss-step", "screening"), "screening"),
    ({"doss_resolution": "Fine"}, ("--doss-step", "fine"), "fine"),
    ({"doss_resolution": "Step", "doss_step": "0.5"}, ("--doss-step", "0.5"), "step"),
    ({"doss_resolution": "Points", "doss_points": "1000"}, ("--doss-points", "1000"), "points"),
])
def test_resolution_row_writes_the_clis_deck(tmp_path, answers, flags, resolution):
    folder = R.messy(tmp_path / "messy")
    r = _run(tmp_path / "w", folder / "Si_sp.out", "Si_sp.out", "opt2d3", "DOSS", answers=answers)
    conf = json.loads(r["json_text"])
    conf = conf.get("configuration", conf)
    assert conf["energy_resolution"] == resolution
    assert _npt(r["deck"]) == _cli(folder, *flags)
    rows = {f["id"]: f for f in r["form"]}
    row = rows["doss_resolution"]
    assert row["kind"] == "menu" and row["default"] == "Standard"
    texts = [o["text"] for o in row["options"]]
    assert [t.split(" - ")[0] for t in texts] == ["Screening", "Standard", "Fine", "Step", "Points"]
    # each preset names the points and step it gives for this output
    assert all(re.search(r"\(\d+ points, ~[\d.]+ eV apart\)", t) for t in texts[:3]), texts



def test_window_row_is_the_default_the_cli_writes(tmp_path):
    """A spin-locked diamond SP with its BAND run beside it (E_F from there): Standard over every band needs
    39552 points, so the Resolution row says so and the Window around E_F row is Yes by default (E_F -20..+15
    eV, its lower bound moved into the gap below the valence band by the BAND run: 1959 points, the deck
    `mace opt2d3 --non-interactive` writes); No writes every band, as `--doss-window all`."""
    import lzma
    from conftest import REPO_ROOT
    suite = REPO_ROOT / "examples" / "suite" / "2-dense-3d" / "diamond"
    run = tmp_path / "run"
    run.mkdir()
    for sub, name in (("SP", "diamond_sp.out"), ("BAND", "diamond_band.out"), ("BAND", "diamond_band.BAND.DAT")):
        (run / name).write_bytes(lzma.decompress((suite / sub / (name + ".xz")).read_bytes()))
    (run / "diamond_sp.f9").write_text("wavefunction placeholder\n")
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    env["MACE_NO_BANNER"] = "1"

    def cli(out, *flags):
        subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "opt2d3", "-i", "diamond_sp.out", "-t", "DOSS",
                        "-o", out, "--non-interactive", *flags], cwd=str(run), capture_output=True, text=True,
                       env=env, timeout=180, check=True)
        return next((run / out).glob("*.d3")).read_text()

    r = _run(tmp_path / "w0", run / "diamond_sp.out", "diamond_sp.out", "opt2d3", "DOSS")
    rows = {f["id"]: f for f in r["form"]}
    assert rows["doss_resolution"]["note"].startswith("This DOS needs 39552 points for a 0.02 eV step over all bands")
    assert rows["doss_window"]["value"] == "Yes" and _npt(r["deck"]) == 1959
    assert any(p.get("field") == "doss_resolution" and p["severity"] == "warning" for p in r["problems"])
    assert r["deck"] == cli("cli0")
    r = _run(tmp_path / "w1", run / "diamond_sp.out", "diamond_sp.out", "opt2d3", "DOSS", answers={"doss_window": False})
    assert _npt(r["deck"]) == 39552 and r["deck"] == cli("cli1", "--doss-window", "all")
    assert {f["id"]: f for f in r["form"]}["doss_window"]["value"] == "No"


def test_1d_rows_are_the_bands_by_index_the_cli_writes(tmp_path):
    """A 1D polyacetylene SP: the Bands by index row is Yes by default (bands 3 to 12, the deck `mace opt2d3
    --non-interactive` writes); the Conduction bands row is --doss-conduction-bands; No keeps every band."""
    import lzma
    from conftest import REPO_ROOT
    suite = REPO_ROOT / "examples" / "suite" / "4-polymer-1d" / "polyacetylene"
    run = tmp_path / "run"
    run.mkdir()
    for sub, name in (("SP", "polyacetylene_sp.out"), ("BAND", "polyacetylene_band.out"),
                      ("BAND", "polyacetylene_band.BAND.DAT")):
        (run / name).write_bytes(lzma.decompress((suite / sub / (name + ".xz")).read_bytes()))
    (run / "polyacetylene_sp.f9").write_text("wavefunction placeholder\n")
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    env["MACE_NO_BANNER"] = "1"

    def cli(out, *flags):
        subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "opt2d3", "-i", "polyacetylene_sp.out", "-t",
                        "DOSS", "-o", out, "--non-interactive", *flags], cwd=str(run), capture_output=True, text=True,
                       env=env, timeout=180, check=True)
        return next((run / out).glob("*.d3")).read_text()

    def bands(deck):
        lines = deck.splitlines()
        return lines[lines.index("DOSS") + 1].split()[2:4]

    r = _run(tmp_path / "w0", run / "polyacetylene_sp.out", "polyacetylene_sp.out", "opt2d3", "DOSS")
    rows = {f["id"]: f for f in r["form"]}
    assert rows["doss_band_index"]["value"] == "Yes" and "doss_window" not in rows
    assert "bands 3 to 12 by index" in rows["doss_resolution"]["note"]
    assert bands(r["deck"]) == ["3", "12"] and r["deck"] == cli("cli0")
    r = _run(tmp_path / "w1", run / "polyacetylene_sp.out", "polyacetylene_sp.out", "opt2d3", "DOSS",
             answers={"doss_conduction_bands": "8"})
    assert bands(r["deck"]) == ["3", "15"] and r["deck"] == cli("cli1", "--doss-conduction-bands", "8")
    conf = json.loads(r["json_text"])
    assert conf.get("configuration", conf)["conduction_bands"] == 8
    r = _run(tmp_path / "w2", run / "polyacetylene_sp.out", "polyacetylene_sp.out", "opt2d3", "DOSS",
             answers={"doss_band_index": False})
    assert bands(r["deck"]) == ["1", "34"] and r["deck"] == cli("cli2", "--doss-window", "all")
    assert {f["id"]: f for f in r["form"]}["doss_band_index"]["value"] == "No"


@pytest.mark.parametrize("answers, flags", [({}, ()), ({"transport_k": "Screening"}, ("--transport-k", "screening")),
                                            ({"transport_k": "Spacing", "transport_k_spacing": "0.2"},
                                             ("--transport-k", "0.2")),
                                            ({"transport_k": "IS", "transport_is": "20"}, ("--transport-is", "20"))])
def test_transport_k_row_writes_the_clis_mesh(tmp_path, answers, flags):
    """TRANSPORT's k-point mesh row is opt2d3's question (Crystal_d3/d3_config.transport_mesh): its deck's NEWK is
    the one `mace opt2d3 --non-interactive` writes with the same choice as a flag."""
    import shutil
    from conftest import REPO_ROOT
    run = tmp_path / "run"
    shutil.copytree(REPO_ROOT / "examples" / "gui" / "outputs", run)
    (run / "graphene_slab_sp.f9").write_text("wavefunction placeholder\n")
    r = _run(tmp_path / "w", run / "graphene_slab_sp.out", "graphene_slab_sp.out", "opt2d3", "TRANSPORT",
             answers=answers)
    rows = {f["id"]: f for f in r["form"]}
    assert rows["transport_k"]["kind"] == "menu" and rows["transport_k"]["default"] == "Standard"
    assert [o["text"].split(" - ")[0] for o in rows["transport_k"]["options"]] == \
        ["Screening", "Standard", "Fine", "Spacing", "IS"]
    env = {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}
    env["MACE_NO_BANNER"] = "1"
    subprocess.run([sys.executable, str(REPO_ROOT / "mace_cli"), "opt2d3", "-i", "graphene_slab_sp.out", "-t",
                    "TRANSPORT", "-o", "cli", "--non-interactive", *flags], cwd=str(run), capture_output=True,
                   text=True, env=env, timeout=180, check=True)
    cli = next((run / "cli").glob("*.d3")).read_text().splitlines()
    deck = r["deck"].splitlines()
    assert deck[deck.index("NEWK") + 1] == cli[cli.index("NEWK") + 1]
