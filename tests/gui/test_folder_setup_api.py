"""Calculation Setup's functional and basis set from Settings / a folder's
mace-folder.toml (MACE 1.1.6): a deck without a parent (cif2d12 from a CIF)
starts with them, in the GUI (Setup's defaults) and on the command line
(`mace convert` run in the folder); a follow-up (opt2d12) keeps its
parent's method whatever the folder says; blank means the CLI's own."""
import json
import shutil

import pytest

import guiharness as H
import p4util as U
import setup_routes as R
from test_setup_api import S, _cache

from mace.utils import folder_settings as F

pytest.importorskip("numpy")
pytestmark = pytest.mark.skipif(not U.mem_ok(1), reason="needs 1 GB free")


def _run(work, path, rel, mode, calc, defaults):
    get, put, _ = _cache()
    body = {"answers": {}, "patches": {}, "preview": True}
    if defaults is not None:
        body["defaults"] = defaults
    return S.run_setup(S.Request(body), mode, calc, work, path, rel, S.source_key(mode, path), get, put, None,
                       schema_info=S.schema())


def test_cif2d12_starts_with_the_gui_defaults(tmp_path):
    pytest.importorskip("ase")
    cif = H.EXAMPLES / "structures" / "MgO.cif"
    r = _run(tmp_path / "a", cif, "structures/MgO.cif", "cif2d12", "OPT",
             {"functional": "PBE0", "basis_set": "POB-DZVP-REV2"})
    data = json.loads(r["json_text"])
    assert (data["functional"], data["basis_set"]) == ("PBE0", "POB-DZVP-REV2")
    assert "PBE0-D3" in r["deck"].splitlines() and "POB-DZVP-REV2" in r["deck"].splitlines()
    f = next(x for x in r["form"] if x["id"] == "functional")
    assert f["default"].startswith("PBE0") and not f["changed"]        # the CLI's default now, not a change
    # blank: the CLI's own default, and no folder file of the CLI's own is read
    r = _run(tmp_path / "b", cif, "structures/MgO.cif", "cif2d12", "OPT", {"functional": "", "basis_set": ""})
    data = json.loads(r["json_text"])
    assert (data["functional"], data["basis_set"]) == ("HSE06", "POB-TZVP-REV2")
    with pytest.raises(S.SetupError):
        S.Request({"defaults": {"functional": "x" * 100}})


def test_opt2d12_keeps_the_parents_method(tmp_path):
    src = tmp_path / "src"
    shutil.copytree(H.EXAMPLES / "outputs", src)
    r = _run(tmp_path / "w", src / "Si_sp.out", "Si_sp.out", "opt2d12", "OPT", {"functional": "B3LYP", "basis_set": "STO-3G"})
    data = json.loads(r["json_text"])
    assert data["functional"] == "PBE0" and data["basis_set"] != "STO-3G"


def test_mace_convert_in_the_folder_reads_its_file(tmp_path):
    pytest.importorskip("ase")
    d = tmp_path / "proj"
    (d / "cif").mkdir(parents=True)
    shutil.copy2(H.EXAMPLES / "structures" / "MgO.cif", d / "cif")
    (d / F.FILENAME).write_text('[setup]\nfunctional = "PBE0"\nbasis_set = "POB-DZVP-REV2"\n')
    R.run_cli_manual(["convert", "--cif_dir", "cif", "--output_dir", "out", "--save_options", "--options_file", "m.json"],
                     R.ENTER, d)
    data = json.loads((d / "m.json").read_text())
    assert (data["functional"], data["basis_set"]) == ("PBE0", "POB-DZVP-REV2")
    # an unusable value: said, and the built-in default stays
    (d / F.FILENAME).write_text('[setup]\nfunctional = "NOTAFUNCTIONAL"\n')
    shutil.rmtree(d / "out")
    out = R.run_cli_manual(["convert", "--cif_dir", "cif", "--output_dir", "out", "--save_options", "--options_file",
                            "n.json"], R.ENTER, d)
    assert json.loads((d / "n.json").read_text())["functional"] == "HSE06"
    assert "NOTAFUNCTIONAL" in str(out)
