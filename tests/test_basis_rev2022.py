"""The opt-in full.basis.triplezeta.rev2022 family.

It is full.basis.triplezeta with X_POB_TZVP_rev2 for Rb-I (not Tc) and
X_pob_TZVP_rev2 for La-Lu, fetched from crystal.unito.it by
basis_sets/fetch_unito.py. It is reached only through menu option 4, the
config alias "external_basis_family": "tz-rev2022", or mace_config's
"tz-rev2022" key; the default (option 2) stays byte-identical, which
test_basis_library_pins.py proves.
"""
import io
import json
import re
import subprocess
import sys
import types
import urllib.error
from pathlib import Path

import pytest

import basis_library as BL
from conftest import REPO_ROOT

BASIS_SETS = REPO_ROOT / "Crystal_d12" / "basis_sets"
sys.path.insert(0, str(BASIS_SETS))
import fetch_unito as F  # noqa: E402

REV = "full.basis.triplezeta.rev2022"
REPLACED = [z for z in range(37, 54) if z != 43] + list(range(57, 72))


@pytest.fixture(scope="module")
def rev():
    return BL.load_family(REV)


@pytest.fixture(scope="module")
def tz():
    return BL.load_family("full.basis.triplezeta")


# --- the library -----------------------------------------------------------

def test_only_rb_i_and_la_lu_change(rev, tz):
    replaced = {str(200 + z) for z in REPLACED}
    assert set(rev.keys()) | set(rev.missing) == set(tz.keys())
    for key in tz.keys():
        if key in replaced or key in rev.missing:
            continue
        assert rev.record_bytes(key) == tz.record_bytes(key), key
        assert [m for m in rev.provenance(key) if m[0] != "note"] == tz.provenance(key)
    for key in replaced - set(rev.missing):
        assert rev.record_bytes(key) != tz.record_bytes(key), key


@pytest.mark.parametrize("z", REPLACED)
def test_each_fetched_record_was_checked_and_says_where_it_came_from(rev, z):
    key = str(200 + z)
    if key in rev.missing:
        pytest.skip(f"{z} refused: {rev.missing[key]}")
    sym = F.symbol(z)
    record = rev.record(key)
    assert BL.cut_record(record) == (record, "")
    assert record.split()[0] == key, "label must be 200 + Z: MACE writes it in the geometry"
    assert record.split("\n")[1].strip() == "INPUT"
    assert rev.field(key, "source")[0].lower() == f"{sym}_pob_tzvp_rev2".lower()
    assert rev.field(key, "url")[0].startswith(F.BASE_URL)
    assert rev.field(key, "retrieved") and rev.field(key, "page_sha256")
    assert rev.field(key, "core")[0].startswith("ZNUC=")
    want = "10.1002/jcc.26839" if z < 57 else "10.1002/jcc.27097"
    assert rev.field(key, "doi") == [want]


def test_core_treatment_is_stated(rev, tz):
    """La's rev2 ECP has ZNUC 11 (46-electron core) where ERD has 29; Ce-Lu
    keep ZNUC = Z - 28; Rb-I keep pob-TZVP 2018's ECP. The description says so."""
    def znuc(lib, key):
        return float(lib.record(key).split("\n")[2].split()[0])

    assert znuc(rev, "257") == 11 and znuc(tz, "257") == 29
    for z in range(58, 72):
        assert znuc(rev, str(200 + z)) == znuc(tz, str(200 + z)) == z - 28
    for z in REPLACED[:16]:
        assert znuc(rev, str(200 + z)) == znuc(tz, str(200 + z))
    assert "DIFFERENT CORE" in rev.field("257", "provenance")[0]
    assert "ZNUC = 11, a 46-electron core" in rev.description
    assert "4f in the valence" in rev.description


def test_tc_keeps_the_erd_record_and_says_why(rev, tz):
    assert rev.record_bytes("43") == tz.record_bytes("43")
    assert "no Tc_POB_TZVP_rev2" in rev.field("43", "note")[0]


def test_compatibility_is_the_tz_set_plus_refused_elements():
    import mace_config
    from d12_constants import check_basis_set_compatibility

    for z in range(1, 101):
        _, tz_missing = check_basis_set_compatibility(mace_config.DEFAULT_TZ_PATH, [z], "EXTERNAL")
        _, rev_missing = check_basis_set_compatibility(
            mace_config.DEFAULT_TZ_REV2022_PATH, [z], "EXTERNAL")
        refused = str(200 + z) in BL.load_family(REV).missing
        assert rev_missing == (tz_missing or ([z] if refused else [])), z


# --- how it is chosen --------------------------------------------------------

def test_mace_config_names_it():
    import mace_config

    assert mace_config.DEFAULT_TZ_REV2022_PATH.endswith("/full.basis.triplezeta.rev2022/")
    assert mace_config.get_basis_path("tz-rev2022") == mace_config.BASIS_TZ_REV2022_DIR
    assert mace_config.get_basis_path("rev2022") == mace_config.BASIS_TZ_REV2022_DIR
    assert BL.bundled_family_for(mace_config.DEFAULT_TZ_REV2022_PATH) == REV


def _choose(monkeypatch, answer):
    import d12_constants

    def fake(prompt, options, default=None):
        if "Select basis set type" in prompt:
            return "1"
        if "Select external basis set" in prompt:
            assert answer in options
            return answer
        raise AssertionError(prompt)

    monkeypatch.setattr(d12_constants, "get_user_input", fake)
    return d12_constants.select_basis_set([37, 57, 6])


def test_menu_option_4_writes_the_rev2_rb_and_la_records(monkeypatch, capsys):
    from d12_constants import read_basis_file

    config = _choose(monkeypatch, "4")
    assert "4: TZVP-REV2 (2022 update: Rb-I, La-Lu)" in capsys.readouterr().out
    assert config["basis_set_type"] == "EXTERNAL"
    assert config["basis_set"] == config["basis_set_path"]
    assert config["basis_set"].endswith("/full.basis.triplezeta.rev2022/")
    rb = read_basis_file(config["basis_set"], 37)
    la = read_basis_file(config["basis_set"], 57)
    assert rb.startswith("237 8\nINPUT\n9. 0 2 4 4 2 0\n") and "1.91033355000" in rb
    assert la.startswith("257 11\nINPUT\n11. 0 2 2 2 1 0\n")


def test_menu_option_2_is_still_the_old_set(monkeypatch):
    from d12_constants import read_basis_file

    config = _choose(monkeypatch, "2")
    assert config["basis_set"].endswith("/full.basis.triplezeta/")
    assert "1.55101289000" in read_basis_file(config["basis_set"], 37)


def test_a_rev2022_parent_defaults_to_option_4(monkeypatch):
    import d12_constants
    import d12_interactive

    seen = {}

    def fake(prompt, options, default=None):
        seen[prompt] = default
        return default or "1"

    monkeypatch.setattr(d12_constants, "get_user_input", fake)
    path = str(BASIS_SETS / REV) + "/"
    d12_interactive.select_basis_set_with_defaults(
        [37], "DFT", current_basis=path, current_basis_type="EXTERNAL")
    assert seen.get("Select external basis set") == "4"


def test_config_alias_resolves_to_the_family():
    from d12_config import apply_external_basis_family, unwrap_d12_config

    conf = unwrap_d12_config({"external_basis_family": "tz-rev2022", "basis_set": "x"})
    assert conf["basis_set_type"] == "EXTERNAL"
    assert conf["basis_set"] == conf["basis_set_path"]
    assert BL.bundled_family_for(conf["basis_set"]) == REV
    plain = {"basis_set": "POB-TZVP-REV2"}
    assert unwrap_d12_config(plain) is plain
    with pytest.raises(ValueError, match="must be one of"):
        apply_external_basis_family({"external_basis_family": "tz-2099"})
    with pytest.raises(ValueError, match="name different basis sets"):
        apply_external_basis_family({"external_basis_family": "tz-rev2022",
                                     "basis_set_path": str(BASIS_SETS / "full.basis.triplezeta")})


RBI_CIF = """data_RbI
_symmetry_space_group_name_H-M   'F m -3 m'
_cell_length_a   7.3426
_cell_length_b   7.3426
_cell_length_c   7.3426
_cell_angle_alpha   90.0
_cell_angle_beta   90.0
_cell_angle_gamma   90.0
_symmetry_Int_Tables_number   225
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Rb  Rb1  0.0  0.0  0.0  1
  I  I1  0.5  0.5  0.5  1
"""


def test_cif2d12_batch_with_the_alias_writes_a_rev2022_deck(tmp_path):
    """Real CLI: `mace convert --batch` with "external_basis_family"."""
    pytest.importorskip("ase", reason="NewCifToD12 imports ase.io")
    (tmp_path / "cifs").mkdir()
    (tmp_path / "cifs" / "RbI.cif").write_text(RBI_CIF)
    opts = tmp_path / "opts.json"
    opts.write_text(json.dumps({
        "calculation_type": "SP", "method": "DFT", "dft_functional": "PBE",
        "external_basis_family": "tz-rev2022", "symmetry_handling": "P1"}))
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "mace_cli"), "convert", "--no-banner", "--batch",
         "--options_file", str(opts), "--cif_dir", str(tmp_path / "cifs"),
         "--output_dir", str(tmp_path / "out")],
        cwd=tmp_path, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
    (deck,) = (tmp_path / "out").glob("*.d12")
    assert deck.name == "RbI_CRYSTAL_SP_P1_PBE_full.basis.triplezeta.rev2022.d12"
    text = deck.read_text()
    rev = BL.load_family(REV)
    assert rev.record("237") in text and rev.record("253") in text
    assert str(REPO_ROOT) not in text


# --- the workflow executor ---------------------------------------------------

def _resolve(name):
    from mace.workflow.executor import WorkflowExecutor

    return WorkflowExecutor._extract_basis_path_from_d12(types.SimpleNamespace(), name)


def test_executor_keeps_a_rev2022_child_on_rev2022(monkeypatch, tmp_path):
    """MACE_TZ_BASIS_PATH / MACE_BASIS_DIR name the site's older TZ set; a
    rev2022 deck's name contains "full.basis.triplezeta" too, and must not
    be handed that set."""
    (tmp_path / "full.basis.triplezeta").mkdir()
    monkeypatch.setenv("MACE_TZ_BASIS_PATH", str(tmp_path / "full.basis.triplezeta"))
    monkeypatch.setenv("MACE_BASIS_DIR", str(tmp_path))
    got = _resolve("x_CRYSTAL_SP_symm_PBE_full.basis.triplezeta.rev2022.d12")
    assert BL.bundled_family_for(got) == REV
    assert _resolve("x_CRYSTAL_SP_symm_PBE_full.basis.triplezeta.d12") == str(
        tmp_path / "full.basis.triplezeta")


def test_no_new_substring_basis_checks():
    """The rev2022 family name contains "full.basis.triplezeta" and "tz";
    every substring test on a basis name was checked against it. A new one
    must be checked too - then add it here."""
    pattern = re.compile(r"""(['"])(?:full\.basis\.)?(?:triplezeta|doublezeta|tz|dz|tzvp|dzvp|"""
                         r"""TZVP|DZVP)\1\s+in\b|_indicator in """)
    known = {
        ("Crystal_d12/d12_interactive.py", 'elif "DZVP" in current_basis or "doublezeta" in current_basis:'),
        ("Crystal_d12/d12_interactive.py", 'elif "TZVP" in current_basis or "triplezeta" in current_basis:'),
        ("mace/database/materials.py", "elif 'POB' in part.upper() or 'TZVP' in part.upper() or 'DZVP' in part.upper():"),
        ("mace/workflow/engine.py", "elif 'POB' in part.upper() or 'TZVP' in part.upper() or 'DZVP' in part.upper():"),
        ("mace/workflow/executor.py", 'if "full.basis.triplezeta" in filename:'),
        ("mace/workflow/executor.py", 'elif "full.basis.doublezeta" in filename:'),
        ("mace/workflow/executor.py", "if any(tz_indicator in filename.lower() for tz_indicator in ['full.basis.triplezeta', 'triplezeta', 'tz', 'tzvp']):"),
        ("mace/workflow/executor.py", "elif any(dz_indicator in filename.lower() for dz_indicator in ['full.basis.doublezeta', 'doublezeta', 'dz', 'dzvp']):"),
        ("mace/workflow/planner.py", 'or "TZVP" in part.upper()'),
        ("mace/workflow/planner.py", 'or "DZVP" in part.upper()'),
    }
    found = set()
    for path in REPO_ROOT.rglob("*.py"):
        rel = path.relative_to(REPO_ROOT)
        if rel.parts[0] in ("tests", "test") or "site-packages" in rel.parts:
            continue
        for line in path.read_text(errors="ignore").splitlines():
            if pattern.search(line):
                found.add((rel.as_posix(), line.strip()))
    assert found == known


# --- fetch_unito on a page in the site's layout (offline) -------------------

PAGE = """<html><body>
<ul><li><a href="#Xx_POB_TZVP_rev2">Xx_POB_TZVP_rev2</a></li></ul>
<p><a name="Xx_POB_TZVP_2018"></a><font size="+2">Xx_POB_TZVP_2018</font></p>
<pre>237 1
HAYWSC
0 1 1 0.0 1.0
  0.5  1.0  1.0
Old paper, DOI: 10.1000/old
</pre>
<p><a name="Xx_POB_TZVP_rev2"></a><font size="+2">Xx_POB_TZVP_rev2</font></p>
<pre>237 4
INPUT
9. 1 1 1 0 0 0
  5.0  -1.5  0
  4.0  89.5  0
  2.0  0.43  0
0 1 2 8.0 1.0
  7.0  0.2  0.3
  6.0  -0.4  0.5
1 0 3 2.0 0.0
0 3 1 0.0 1.0
  0.15 1.0
0 4 1 0 1.0
  0.3 1.0


J. Laun, T. Bredow J. Comput. Chem.  (2022), 43:839&#8211;846
"A title", DOI: 10.1002/jcc.26839
</pre>
<p><a name="Xx_pob_TZVP_rev2_SOC"></a><font size="+2">Xx_pob_TZVP_rev2_SOC</font></p>
<pre>237 1
STUTSC
0 0 1 1 1.0
  1.9  1.0
Laun, https://doi.org/10.1002/jcc.26839
Note: To be used with CRYSTAL23
</pre>
<p><a name="Xx_all_electron"></a><font size="+2">Xx_all_electron</font></p>
<pre>37 1
0 0 1 1.0 1.0
  1.0 1.0
</pre>
</body></html>"""


def _page(text=PAGE):
    return F.Page(37, F.BASE_URL + "rubidium.html", text, "2026-10-01")


def test_cutter_follows_the_grammar_and_keeps_the_reference_out():
    found = F.get_set(_page(), "xx_pob_tzvp_REV2")        # anchors match case-insensitively
    assert found["anchor"] == "Xx_POB_TZVP_rev2"
    record = found["record"]
    # INPUT: ZNUC M M0..M4 then M+M0+..+M4 = 3 lines; an SP shell's lines carry
    # three numbers; an ITYB=1 shell (STO-nG) has no primitive lines.
    assert record.splitlines()[-1] == "  0.3 1.0" and len(record.splitlines()) == 14
    assert "Laun" not in record and "DOI" not in record
    assert found["reference"].startswith("J. Laun, T. Bredow")
    assert found["dois"] == ["10.1002/jcc.26839"]
    assert (found["label"], found["nshl"], found["core"]) == (237, 4, "ZNUC=9. (INPUT ECP)")
    assert found["url"].endswith("rubidium.html#Xx_POB_TZVP_rev2")
    assert F.ascii_text(found["reference"]).endswith("43:839-846 \"A title\", DOI: 10.1002/jcc.26839")
    assert F.check_ecp_set(37, found) is None


def test_cutter_handles_keyword_ecps_and_doi_links():
    found = F.get_set(_page(), "Xx_pob_TZVP_rev2_SOC")
    assert found["record"] == "237 1\nSTUTSC\n0 0 1 1 1.0\n  1.9  1.0\n"
    assert found["core"] == "STUTSC (internal ECP library)"
    assert found["dois"] == ["10.1002/jcc.26839"]


def test_a_record_whose_label_is_not_200_plus_z_is_refused():
    found = F.get_set(_page(), "Xx_all_electron")
    assert "label 37 is not 237" in F.check_ecp_set(37, found)


@pytest.mark.parametrize("bad, why", [
    ("237 4\nINPUT\n9. 1 1 0 0 0 0\n  5.0  -1.5  0\n", "record ends early"),
    ("237 1\nNOTANECP\n", "unknown pseudo-potential keyword"),
    ("237 1\nINPUT\n9. 0 0 0 0 0\n", "expected 7 numbers"),
    ("37 1\n0 1 1 2.0 1.0\n 1.0 1.0\n", "expected 3 numbers"),
    ("Title only\n", "not 'NAT NSHL'"),
])
def test_the_cutter_names_what_is_wrong(bad, why):
    with pytest.raises(BL.GrammarError, match=why):
        BL.cut_record(bad)


def test_page_names_fall_back_and_record_which_url_answered(tmp_path):
    tried = []

    class Resp(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(url, timeout=None):
        tried.append(url)
        if url.endswith("caesium.html"):
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
        return Resp(PAGE.encode())

    page = F.fetch_page(55, str(tmp_path), opener=opener)
    assert tried == [F.BASE_URL + "caesium.html", F.BASE_URL + "cesium.html"]
    assert page.url == F.BASE_URL + "cesium.html"
    assert (tmp_path / "cesium.html").is_file()
    again = F.fetch_page(55, str(tmp_path), opener=lambda *a, **k: pytest.fail("not cached"))
    assert again.sha256 == page.sha256


def test_an_unreachable_page_is_reported_not_guessed(tmp_path):
    def opener(url, timeout=None):
        raise urllib.error.URLError("offline")

    with pytest.raises(LookupError, match="no page for Z=43: .*technecium.html.*technetium.html"):
        F.fetch_page(43, str(tmp_path), opener=opener)
