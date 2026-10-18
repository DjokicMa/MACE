"""Calculation Setup's preview of an HF-3c child of a spin-polarized parent is
the CLI's deck: unrestricted (UHF), byte for byte what `mace opt2d12
--config-file ... --yes` writes from the same settings."""
import json
import shutil
import subprocess
import sys

import pytest

import guiharness as H

pytest.importorskip("numpy")

sys.path.insert(0, str(H.REPO_ROOT))
from mace.gui import setup as S  # noqa: E402

CORPUS = H.REPO_ROOT / "test"
PARENT = "1_dia_opt_rev1"                    # DFT / SPIN / B3LYP-D3
corpus = pytest.mark.skipif(not (CORPUS / "OPT" / f"{PARENT}.out").is_file(),
                            reason="needs the test/ corpus")


def _cache():
    store = {}
    return (lambda k: store.get(json.dumps(k, default=str)),
            lambda k, v, n: store.__setitem__(json.dumps(k, default=str), v))


def _run(work, out, json_text=None, cache=None):
    get, put = cache
    req = S.Request({"answers": {}, "patches": {}, "preview": json_text is not None,
                     "json_text": json_text})
    return S.run_setup(req, "opt2d12", "SP", work, out, out.name, S.source_key("opt2d12", out),
                       get, put, None, schema_info=S.schema())


@corpus
@pytest.mark.parametrize("functional, basis", [("HF3C", "MINIX"), ("HFSOL3C", "SOLMINIX")])
def test_preview_is_the_clis_unrestricted_hf3c_deck(tmp_path, functional, basis):
    src = tmp_path / "src"
    src.mkdir()
    for ext in ("out", "d12"):
        shutil.copy(CORPUS / "OPT" / f"{PARENT}.{ext}", src)
    out = src / f"{PARENT}.out"
    cache = _cache()
    settings = json.loads(_run(tmp_path / "w0", out, cache=cache)["json_text"])
    settings.update(functional=functional, method="HF", basis_set=basis, basis_set_type="INTERNAL")
    text = json.dumps(settings, indent=2)

    r = _run(tmp_path / "w1", out, json_text=text, cache=cache)
    assert f"BASISSET\n{basis}\nUHF\n{functional}\nEND\n" in r["deck"], r["deck"]

    cli = tmp_path / "cli"
    shutil.copytree(src, cli)
    (cli / "o.json").write_text(text)
    p = subprocess.run([sys.executable, str(H.MACE_CLI), "opt2d12", "--out-file", out.name,
                        "--config-file", "o.json", "--yes"],
                       cwd=cli, capture_output=True, text=True, timeout=300,
                       stdin=subprocess.DEVNULL)
    assert p.returncode == 0, (p.stdout + p.stderr)[-2000:]
    assert (cli / r["deck_name"]).read_text() == r["deck"]
