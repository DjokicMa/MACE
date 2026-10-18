"""Spin-orbit coupling in Calculation Setup (1.1.6): the GUI only calls the
CLI. The SOC choices are keys of the CLI's own settings file ("soc",
"soc_ecp", "soc_basis", "soc_guess", "smear"), written as patches into the
JSON the CLI saved, and the preview runs the printed command on it; so a
GUI deck is byte for byte the deck the CLI writes from the same file, which
these tests check against ``mace convert`` / ``mace opt2d12`` run on their
own. Refusals and warnings are the CLI's words."""
import json
import shutil
import sys
from pathlib import Path

import pytest

import guiharness as H

pytest.importorskip("numpy")

sys.path.insert(0, str(H.REPO_ROOT))
from mace.gui import clirun  # noqa: E402
from mace.gui import setup as S  # noqa: E402

CORPUS = H.REPO_ROOT / "test"
TIPBO3_SP = "SP/TiPbO3_mp-20459_sg99_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized_sp_B3LYP-D3-D3_optimized"
AG2CL1_OPT = "OPT/Ag2Cl1_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized"
corpus = pytest.mark.skipif(not (CORPUS / (TIPBO3_SP + ".out")).is_file(), reason="needs the test/ corpus")

PBTE_CIF = """data_PbTe
_symmetry_space_group_name_H-M   'F m -3 m'
_symmetry_Int_Tables_number      225
_cell_length_a   6.462
_cell_length_b   6.462
_cell_length_c   6.462
_cell_angle_alpha   90
_cell_angle_beta    90
_cell_angle_gamma   90
loop_
_symmetry_equiv_pos_as_xyz
  'x, y, z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Pb1 Pb 0.0 0.0 0.0 1.0
Te1 Te 0.5 0.5 0.5 1.0
"""
# Hf in a rock-salt cell: STUTSC is refused for Hf (soc_ecp.STUTSC_NOT_THE_PUBLISHED_ECP)
HFO_CIF = PBTE_CIF.replace("data_PbTe", "data_HfO").replace("6.462", "5.10").replace(
    "Pb1 Pb", "Hf1 Hf").replace("Te1 Te", "O1 O")
# an all-electron 4p atom (As) next to an ECP atom (In): the CLI warns that As loses its splitting
INAS_CIF = PBTE_CIF.replace("data_PbTe", "data_InAs").replace("6.462", "6.058").replace(
    "Pb1 Pb", "In1 In").replace("Te1 Te", "As1 As")
# an open-shell f element: the CLI warns (EuO did not converge on HPCC)
EUO_CIF = PBTE_CIF.replace("data_PbTe", "data_EuO").replace("6.462", "5.144").replace(
    "Pb1 Pb", "Eu1 Eu").replace("Te1 Te", "O1 O")
# what a SOC deck needs from cif2d12: an explicit (external) basis and a functional the 2c-SCF runs
SOC_READY = {"basis_type": "EXTERNAL", "functional": "PBE0", "dispersion": "No"}


def _cache():
    store = {}
    return (lambda k: store.get(json.dumps(k, default=str)),
            lambda k, v, n: store.__setitem__(json.dumps(k, default=str), v))


def _run(work, path, rel, mode, calc, answers=None, patches=None, cache=None, preview=True):
    get, put = cache or _cache()
    req = S.Request({"answers": answers or {}, "patches": patches or {}, "preview": preview})
    return S.run_setup(req, mode, calc, work, path, rel, S.source_key(mode, path), get, put, None,
                       schema_info=S.schema())


def _cif(tmp_path, text, name):
    d = tmp_path / "src"
    d.mkdir(exist_ok=True)
    p = d / f"{name}.cif"
    p.write_text(text)
    return p


def _cli(args, cwd):
    """``mace <args>`` as its own process (the CLI a user runs)."""
    res = clirun.run(clirun.mace_argv("--no-banner", *args), cwd, None, timeout=300)
    return res


def _cli_convert(tmp_path, cif: Path, options: str) -> str:
    d = tmp_path / "cli"
    (d / "cif").mkdir(parents=True)
    shutil.copy2(cif, d / "cif" / cif.name)
    (d / "o.json").write_text(options)
    res = _cli(["convert", "--batch", "--options_file", "o.json", "--cif_dir", "cif", "--output_dir", "out"], d)
    decks = sorted((d / "out").glob("*.d12"))
    assert decks, res.log[-3000:]
    return decks[0].read_text()


def _with(json_text: str, keys) -> str:
    """The user's options file: the CLI's saved answers with the SOC keys added (json.dump indent 2)."""
    data = json.loads(json_text)
    data.update(keys)
    return json.dumps(data, indent=2)


def _parent(tmp_path, stem, symmremo=False, f9=True):
    """A corpus parent (.out + .d12) copied into a folder of its own, with a
    100-byte stand-in .f9 (the CLI checks only that it is there and not empty)."""
    d = tmp_path / "parent"
    d.mkdir(exist_ok=True)
    name = Path(stem).name
    shutil.copy2(CORPUS / f"{stem}.out", d / f"{name}.out")
    deck = (CORPUS / f"{stem}.d12").read_text()
    if symmremo:
        # GUESSPNOSO needs a parent run without symmetry: SYMMREMO in its geometry block (manual p. 78)
        deck = deck.replace("\nEND\n", "\nSYMMREMO\nEND\n", 1)
    (d / f"{name}.d12").write_text(deck)
    if f9:
        (d / f"{name}.f9").write_bytes(b"\0" * 100)
    return d / f"{name}.out"


def _cli_opt2d12(tmp_path, out: Path, config: str):
    d = out.parent
    (d / "cli.json").write_text(config)
    res = _cli(["opt2d12", "--out-file", out.name, "--config-file", "cli.json", "--yes", "--output-dir", "cli_out"], d)
    decks = sorted((d / "cli_out").glob("*.d12"))
    return (decks[0].read_text() if decks else None), res.log


# ------------------------------------------------------------ the schema: the CLI's own tables
def test_schema_names_the_soc_keys_and_the_clis_reasons():
    import soc_ecp
    sc = S.schema()["soc"]
    assert sc["calcs"] == ["SP"]
    assert sc["keys"] == list(S.SOC_KEYS) == ["soc", "soc_ecp", "soc_basis", "soc_guess", "smear"]
    assert sc["ecp_forms"] == list(soc_ecp.ECP_FORMS)
    assert sc["soc_basis_families"] == list(soc_ecp.SOC_BASIS_FAMILIES)
    assert sc["stutsc_refused"]["72"] == soc_ecp.STUTSC_NOT_THE_PUBLISHED_ECP[72]
    assert sc["metal_smear"] == soc_ecp.SOC_METAL_SMEAR_EXAMPLE == 0.005
    # OPT and FREQ: the refusal soc_deck itself gives, word for word
    assert sc["not_for"]["OPT"] == soc_ecp.unsupported_message("OPTGEOM")
    assert sc["not_for"]["FREQ"] == soc_ecp.unsupported_message("FREQCALC")
    assert "p. 166" in sc["not_for"]["OPT"]
    # properties: no SOC key; what opt2d3 writes from a 2c run (CRYSTALOptToD3.TWO_COMPONENT_D3_TYPES)
    assert "BAND" in sc["not_for"]["BAND"] and "DOSS" in sc["not_for"]["BAND"]
    # SOC keys are hand-written settings (patches), not CLI questions: no form row of their own
    for k in S.SOC_KEYS:
        assert S.PATCH_FIELDS[k]["kind"] == "soc" and S.DERIVED_BY_CLI[k] == "patch"


def test_request_checks_soc_values_with_the_clis_checks():
    S.Request({"patches": {"soc": True, "soc_ecp": "stutsc", "soc_basis": "pob-tzvp-rev2-soc",
                           "soc_guess": "guesspnoso", "smear": 0.005}})
    S.Request({"patches": {"soc": False, "soc_ecp": None}})
    with pytest.raises(S.SetupError, match="soc_ecp must be"):
        S.Request({"patches": {"soc_ecp": "stuttgart"}})
    with pytest.raises(S.SetupError, match="smear must be"):
        S.Request({"patches": {"smear": -1}})
    with pytest.raises(S.SetupError, match="soc_guess must be"):
        S.Request({"patches": {"soc_guess": "guessp"}})
    with pytest.raises(S.SetupError, match="soc is"):
        S.Request({"patches": {"soc": "yes please"}})


# ------------------------------------------------------------ GUI deck == CLI deck
@pytest.mark.parametrize("keys,expect", [
    ({"soc": True}, ("TWOCOMPON", "INPSOC")),
    ({"soc": True, "soc_ecp": "stutsc"}, ("TWOCOMPON", "STUTSC")),
    ({"soc": True, "smear": 0.005}, ("TWOCOMPON", "INPSOC", "SMEAR")),
])
def test_cif_soc_deck_is_the_clis_deck(tmp_path, keys, expect):
    pytest.importorskip("ase")
    cif = _cif(tmp_path, PBTE_CIF, "PbTe")
    cache = _cache()
    r0 = _run(tmp_path / "w0", cif, "src/PbTe.cif", "cif2d12", "SP", SOC_READY, cache=cache, preview=False)
    r = _run(tmp_path / "w1", cif, "src/PbTe.cif", "cif2d12", "SP", SOC_READY, keys, cache=cache)
    assert r["deck"], r.get("refused")
    lines = r["deck"].splitlines()
    for word in expect:
        assert word in lines
    # the settings file: the CLI's saved answers plus the keys, as a user writes them
    assert r["json_text"] == _with(r0["json_text"], keys)
    assert r["deck"] == _cli_convert(tmp_path, cif, _with(r0["json_text"], keys))
    soc = r["soc"]
    assert soc["applies"] and soc["values"]["soc"] is True and soc["refusal"] is None
    # what the CLI changed, in its own lines ("SOC deck: FMIXING 30 replaced by 50 ...")
    assert any(c.startswith("SOC deck:") and "FMIXING" in c for c in soc["changes"])
    # cif2d12 cannot tell a metal from a CIF: the CLI's one warning, on the SOC section (unless SMEAR is set)
    metal = [p for p in r["problems"] if p.get("field") == "soc" and "SMEAR" in p["message"]]
    assert bool(metal) == ("smear" not in keys)


@corpus
@pytest.mark.parametrize("keys,expect", [
    ({"calculation_type": "SP", "soc": True}, "INPSOC"),
    ({"calculation_type": "SP", "soc": True, "soc_basis": "pob-tzvp-rev2-soc"}, "STUTSC"),
    ({"calculation_type": "SP", "soc": True, "smear": 0.005}, "SMEAR"),
])
def test_out_parent_soc_deck_is_the_clis_deck(tmp_path, keys, expect):
    out = _parent(tmp_path, TIPBO3_SP)
    r = _run(tmp_path / "w", out, f"parent/{out.name}", "opt2d12", "SP",
             patches={k: v for k, v in keys.items() if k != "calculation_type"})
    assert r["deck"], r.get("refused")
    assert "TWOCOMPON" in r["deck"].splitlines() and expect in r["deck"].splitlines()
    cli, log = _cli_opt2d12(tmp_path, out, r["json_text"])
    assert cli == r["deck"], log[-3000:]


@corpus
def test_inpsoc_from_an_opt_parent_is_the_clis_deck(tmp_path):
    out = _parent(tmp_path, AG2CL1_OPT)
    r = _run(tmp_path / "w", out, f"parent/{out.name}", "opt2d12", "SP", patches={"soc": True})
    assert r["deck"] and "INPSOC" in r["deck"].splitlines(), r.get("refused")
    # the template a user writes: the parent's saved settings with "soc": true
    r0 = _run(tmp_path / "w0", out, f"parent/{out.name}", "opt2d12", "SP", preview=False)
    cli, log = _cli_opt2d12(tmp_path, out, _with(r0["json_text"], {"soc": True}))
    assert cli == r["deck"], log[-3000:]


@corpus
def test_guesspnoso_from_an_out_parent_is_the_clis_deck(tmp_path):
    out = _parent(tmp_path, TIPBO3_SP, symmremo=True)
    keys = {"soc": True, "soc_guess": "guesspnoso"}
    r = _run(tmp_path / "w", out, f"parent/{out.name}", "opt2d12", "SP", patches=keys)
    assert r["deck"], r.get("refused")
    assert "GUESSPNOSO" in r["deck"].splitlines()
    g = {x["value"]: x for x in r["soc"]["guesses"]}
    assert g["guesspnoso"]["ok"] and not g["guesspso"]["ok"] and g[None]["ok"]
    assert "one-component" in g["guesspso"]["reason"]
    cli, log = _cli_opt2d12(tmp_path, out, r["json_text"])
    assert cli == r["deck"], log[-3000:]
    # the CLI staged the parent's matrix as <deck>.f20 for the job (its own run, not the preview's)
    assert list((out.parent / "cli_out").glob("*.f20"))


# ------------------------------------------------------------ what the parent allows
@corpus
def test_a_parent_with_symmetry_offers_no_guesspnoso_and_says_why(tmp_path):
    out = _parent(tmp_path, TIPBO3_SP)
    r = _run(tmp_path / "w", out, f"parent/{out.name}", "opt2d12", "SP", patches={"soc": True})
    g = {x["value"]: x for x in r["soc"]["guesses"]}
    assert g[None]["ok"] and not g["guesspnoso"]["ok"]
    assert "the parent ran with symmetry" in g["guesspnoso"]["reason"]
    # chosen anyway (a hand edit): the CLI refuses with these words, shown as they are
    r2 = _run(tmp_path / "w2", out, f"parent/{out.name}", "opt2d12", "SP",
              patches={"soc": True, "soc_guess": "guesspnoso"})
    assert r2["deck"] is None and "the parent ran with symmetry" in r2["soc"]["refusal"]


@corpus
def test_a_deck_parent_has_no_wavefunction(tmp_path):
    out = _parent(tmp_path, TIPBO3_SP)
    d12 = out.with_suffix(".d12")
    r = _run(tmp_path / "w", d12, f"parent/{d12.name}", "opt2d12", "SP", patches={"soc": True})
    g = {x["value"]: x for x in r["soc"]["guesses"]}
    assert g[None]["ok"] and not g["guesspnoso"]["ok"] and not g["guesspso"]["ok"]
    assert "a deck does not have" in g["guesspso"]["reason"]


@corpus
def test_no_wavefunction_beside_the_out_is_said(tmp_path):
    out = _parent(tmp_path, TIPBO3_SP, symmremo=True, f9=False)
    r = _run(tmp_path / "w", out, f"parent/{out.name}", "opt2d12", "SP", patches={"soc": True}, preview=False)
    g = {x["value"]: x for x in r["soc"]["guesses"]}
    assert not g["guesspnoso"]["ok"] and "missing" in g["guesspnoso"]["reason"]


# ------------------------------------------------------------ calculations SOC is not for
def test_opt_with_soc_is_the_clis_refusal(tmp_path):
    pytest.importorskip("ase")
    cif = _cif(tmp_path, PBTE_CIF, "PbTe")
    r = _run(tmp_path / "w", cif, "src/PbTe.cif", "cif2d12", "OPT", SOC_READY, {"soc": True})
    import soc_ecp
    assert r["deck"] is None
    assert not r["soc"]["applies"]
    assert soc_ecp.unsupported_message("OPTGEOM") in r["soc"]["refusal"]


# ------------------------------------------------------------ refusals and warnings: the CLI's words
def test_internal_basis_refusal_is_the_clis(tmp_path):
    pytest.importorskip("ase")
    cif = _cif(tmp_path, PBTE_CIF, "PbTe")
    r = _run(tmp_path / "w", cif, "src/PbTe.cif", "cif2d12", "SP", patches={"soc": True})
    assert r["deck"] is None
    # the defaults need two changes, and the one refusal names both (E-M6)
    assert r["soc"]["refusal"].startswith("a SOC deck needs 2 changes")
    assert "internal basis-set library (BASISSET)" in r["soc"]["refusal"]
    assert '"basis_set_type": "EXTERNAL"' in r["soc"]["refusal"] and "HSE06" in r["soc"]["refusal"]
    assert any(p["severity"] == "error" and p.get("field") == "soc" and "BASISSET" in p["message"]
               for p in r["problems"])


def test_range_separated_hybrid_refusal_is_the_clis(tmp_path):
    pytest.importorskip("ase")
    cif = _cif(tmp_path, PBTE_CIF, "PbTe")
    r = _run(tmp_path / "w", cif, "src/PbTe.cif", "cif2d12", "SP", {"basis_type": "EXTERNAL"}, {"soc": True})
    assert r["deck"] is None and "HSE06" in r["soc"]["refusal"]


def test_stutsc_for_hf_is_refused_in_the_clis_words(tmp_path):
    pytest.importorskip("ase")
    import soc_ecp
    cif = _cif(tmp_path, HFO_CIF, "HfO")
    r = _run(tmp_path / "w", cif, "src/HfO.cif", "cif2d12", "SP", SOC_READY, {"soc": True, "soc_ecp": "stutsc"})
    assert r["deck"] is None
    assert soc_ecp.STUTSC_NOT_THE_PUBLISHED_ECP[72][:60] in r["soc"]["refusal"]


def test_all_electron_4p_warning_is_the_clis(tmp_path):
    pytest.importorskip("ase")
    cif = _cif(tmp_path, INAS_CIF, "InAs")
    r = _run(tmp_path / "w", cif, "src/InAs.cif", "cif2d12", "SP", SOC_READY, {"soc": True})
    assert r["deck"], r.get("refused")
    msgs = [p["message"] for p in r["problems"] if p.get("field") == "soc"]
    assert any("As (atomic number 33) is all-electron" in m and "InAs" in m for m in msgs), msgs


def test_f_element_warning_is_the_clis(tmp_path):
    pytest.importorskip("ase")
    cif = _cif(tmp_path, EUO_CIF, "EuO")
    r = _run(tmp_path / "w", cif, "src/EuO.cif", "cif2d12", "SP", SOC_READY, {"soc": True})
    msgs = [p["message"] for p in r["problems"] if p.get("field") == "soc"]
    assert any("Eu is an f element" in m for m in msgs), (msgs, r.get("refused"))


def test_scalar_deck_has_no_soc_noise(tmp_path):
    pytest.importorskip("ase")
    cif = _cif(tmp_path, PBTE_CIF, "PbTe")
    r = _run(tmp_path / "w", cif, "src/PbTe.cif", "cif2d12", "SP", SOC_READY)
    assert r["deck"] and "TWOCOMPON" not in r["deck"]
    assert r["soc"]["values"] == {} and r["soc"]["changes"] == [] and r["soc"]["refusal"] is None
    assert not [p for p in r["problems"] if p.get("field") == "soc"]


@corpus
@pytest.mark.parametrize("record", ["ANDERSON", "BROYDEN"])
def test_a_mixing_2c_does_not_run_is_refused_as_the_cli_says(tmp_path, record):
    out = _parent(tmp_path, TIPBO3_SP)
    d12 = out.with_suffix(".d12")
    lines = d12.read_text().splitlines()
    i = lines.index("DIIS")
    # the parent's DIIS / HISTDIIS 100 become the mixing 2c does not take (BROYDEN takes its 3 numbers)
    lines[i:i + 3] = [record] + (["0.0001 50 0"] if record == "BROYDEN" else [])
    d12.write_text("\n".join(lines) + "\n")
    r = _run(tmp_path / "w", out, f"parent/{out.name}", "opt2d12", "SP", patches={"soc": True})
    cli, log = _cli_opt2d12(tmp_path, out, r["json_text"])
    assert r["deck"] is None and cli is None
    assert record in r["soc"]["refusal"] and "p. 166" in r["soc"]["refusal"]
    assert r["soc"]["refusal"] == S.soc_refusal(log)          # the CLI's sentence, whole
    assert f"The CLI wrote no deck: {r['soc']['refusal']}" in [p["message"] for p in r["problems"]]


def test_a_net_charge_on_a_soc_deck_shows_the_clis_refusal_on_the_soc_section(tmp_path):
    """Setup's charged-system switches and SOC together: the CLI refuses the
    DOPING a net charge writes into a 2c deck, and the SOC section says so in
    its words (the charge records go in after the SOC deck is written)."""
    pytest.importorskip("ase")
    p = _cif(tmp_path, PBTE_CIF, "PbTe")
    get, put = _cache()
    req = S.Request({"answers": dict(SOC_READY, calc_type="SP"), "patches": {"soc": True},
                     "flags": {"net_charge": 2}, "preview": True})
    r = S.run_setup(req, "cif2d12", "SP", tmp_path / "w", p, "src/PbTe.cif", S.source_key("cif2d12", p),
                    get, put, None, schema_info=S.schema())
    assert "--net-charge 2" in r["command"] and r["deck"] is None
    why = r["soc"]["refusal"]
    assert why.startswith("DOPING in the SCF input (the charged-system settings)")
    errors = [x for x in r["problems"] if x["severity"] == "error"]
    assert errors[0]["field"] == "soc" and errors[0]["message"] == f"The CLI wrote no deck: {why}"
