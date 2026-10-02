"""An opt2d12 config file with "method": "HF" writes Hartree-Fock decks.

Corpus-free: the batch fixture of test_opt2d12_config_template_logic replaces
the parsers by the dictionaries they return (both parents are B3LYP-D3).

quick_screen.json used to read {"method": "HF", "functional": null}. Applied
by opt2d12 that null reached the filename code and every file stopped with
"argument of type 'NoneType' is not iterable"; a config naming only the
method kept the parent's B3LYP-D3. cif2d12 reads the same file as RHF
(d12_config.config_to_cif_options), and so does opt2d12 now.
"""
import json

import pytest

from test_opt2d12_config_template_logic import batch  # noqa: F401


def _run(batch, tmp_path, capsys, config):
    (tmp_path / "t.json").write_text(json.dumps(
        {"version": "1.0", "type": "d12_configuration",
         "configuration": {"calculation_type": "SP", **config}}))
    status, _ = batch("--directory", ".", "--config-file", "t.json", "--output-dir", "sp")
    return status, capsys.readouterr()


@pytest.mark.parametrize("config,hamiltonian", [
    ({"method": "HF", "functional": None}, "RHF"),
    ({"method": "HF"}, "RHF"),
    ({"method": "hf", "hf_method": "UHF"}, "UHF"),
    ({"method": "HF", "functional": "RHF"}, "RHF"),
    ({"method": "HF", "functional": "UHF"}, "UHF"),
])
def test_hf_config_writes_hf_decks(batch, tmp_path, capsys, config, hamiltonian):
    status, out = _run(batch, tmp_path, capsys, config)
    assert status == 0, out.out + out.err
    assert "2 written, 0 failed" in out.out, out.out + out.err
    decks = sorted((tmp_path / "sp").glob("*.d12"))
    assert [d.name for d in decks] == [f"agbr_sp_{hamiltonian}_optimized.d12",
                                       f"nacl_sp_{hamiltonian}_optimized.d12"]
    for deck in decks:
        lines = deck.read_text().splitlines()
        assert "DFT" not in lines
        # RHF is CRYSTAL's default Hamiltonian and has no keyword; UHF does
        assert ("UHF" in lines) == (hamiltonian == "UHF")


def test_hf_config_with_a_dft_functional_is_refused(batch, tmp_path, capsys):
    status, out = _run(batch, tmp_path, capsys, {"method": "HF", "functional": "PBE0"})
    assert "0 written, 2 failed" in out.out + out.err
    assert not list((tmp_path / "sp").glob("*.d12"))


@pytest.mark.parametrize("functional,basis", [("HF3C", "MINIX"), ("HFSOL3C", "SOLMINIX")])
def test_hf_3c_config_writes_its_own_basis(batch, tmp_path, capsys, functional, basis):
    """HF3C / HFSOL3C go with a pure HF calculation in MINIX / SOLMINIX
    (manual 5.3.1 p.158, 5.4.1 p.162; deck BASISSET / MINIX / HF3C / END,
    p.159). The parent's basis was kept: the external-basis parent (agbr)
    came out as "BASISSET / EXTERNAL (from original D12)" and the internal
    one (nacl) as BASISSET / POB-TZVP-REV2 under HF3C."""
    status, out = _run(batch, tmp_path, capsys, {"method": "HF", "functional": functional})
    assert status == 0, out.out + out.err
    for stem, atoms in (("agbr", ["47", "35"]), ("nacl", ["11", "17"])):
        deck = (tmp_path / "sp" / f"{stem}_sp_{functional}_optimized.d12").read_text()
        lines = deck.splitlines()
        k = lines.index("BASISSET")
        assert lines[k:k + 4] == ["BASISSET", basis, functional, "END"], deck
        assert "99 0" not in lines and "EXTERNAL" not in deck
        # an internal basis takes plain atomic numbers, not the ECP's 247
        assert [ln.split()[0] for ln in lines[6:8]] == atoms, deck


# --- the basis the log names is the basis the deck has ----------------------
#
# opt2d12 printed "Using external basis set from original D12 file" as soon as
# it read an external-basis parent, before the config chose the method; an
# HF3C child then logged that line and "Basis set: MINIX", and a DFT-3c child
# (method_modifications new_functional) logged it although the deck says
# BASISSET <3c basis> ("Preserving external basis settings" likewise). The
# line is now printed once, when the deck's basis is known.
# The decks are pinned by goldens written by 1.1.4 (data/opt2d12_basis_log/).

from pathlib import Path  # noqa: E402

GOLDEN = Path(__file__).parent / "data" / "opt2d12_basis_log"
EXTERNAL_LINE = "Using external basis set from original D12 file"
EXTERNAL_BLOCK = "\n".join(["247 3", "INPUT", "19. 0 1 2 1 0 0", "0 0 1 2.0 1.0", "1.0 1.0",
                            "35 2", "0 0 1 2.0 1.0", "1.0 1.0", "99 0"])
BASIS_CASES = {
    "HF3C": {"method": "HF", "functional": "HF3C"},
    "HFSOL3C": {"method": "HF", "functional": "HFSOL3C"},
    "PBEH3C": {"method_modifications": {"new_functional": "PBEH3C"}},
    "HSESOL3C": {"method_modifications": {"new_functional": "HSESOL3C"}},
    "PBE0": {"functional": "PBE0", "dispersion": False},
    "RHF": {"method": "HF"},
}


def external_parent_child(batch, tmp_path, capsys, config):
    """The SP deck opt2d12 writes from the external-basis parent (agbr) and
    the log of that run."""
    (tmp_path / "t.json").write_text(json.dumps(
        {"version": "1.0", "type": "d12_configuration",
         "configuration": {"calculation_type": "SP", **config}}))
    status, _ = batch("--out-file", "agbr.out", "--d12-file", "agbr.d12",
                      "--config-file", "t.json", "--output-dir", "sp")
    out = capsys.readouterr()
    decks = list((tmp_path / "sp").glob("agbr_sp_*.d12"))
    assert status == 0 and len(decks) == 1, out.out + out.err
    return out.out + out.err, decks[0]


@pytest.mark.parametrize("name", sorted(BASIS_CASES))
def test_logged_basis_is_the_decks_basis(batch, tmp_path, capsys, name):
    log, deck = external_parent_child(batch, tmp_path, capsys, BASIS_CASES[name])
    text = deck.read_text()
    has_external = EXTERNAL_BLOCK in text
    assert (EXTERNAL_LINE in log) == has_external, log
    # no other line claims the parent's external basis is kept
    claims = [ln for ln in log.splitlines()
              if "external basis" in ln.lower() and "Found external basis set data" not in ln]
    assert len(claims) == has_external, claims
    lines = text.splitlines()
    if "BASISSET" in lines:
        basis = lines[lines.index("BASISSET") + 1]
        assert not has_external
        assert f"Basis set: {basis}" in log or f"Basis set updated to: {basis}" in log, log
    else:
        assert has_external
    # the decks themselves are what 1.1.4 wrote
    assert text == (GOLDEN / f"agbr_{name}.d12").read_text()


# --- the legacy executor's log (WorkflowExecutor.run_crystal_opt_conversion)

INTERNAL_PARENT = """nacl
CRYSTAL
0 0 0
225
5.64
2
11 0.0 0.0 0.0
17 0.5 0.5 0.5
BASISSET
POB-TZVP-REV2
DFT
PBE0
ENDDFT
SHRINK
8 8
END
"""


def _executor_step(tmp_path, monkeypatch, capsys, parent_deck, step):
    from mace.workflow import executor as ex_mod

    ex = ex_mod.WorkflowExecutor.__new__(ex_mod.WorkflowExecutor)
    ex.work_dir = tmp_path
    ex.temp_dir = tmp_path / "tmp"
    ex.temp_dir.mkdir()
    d12 = tmp_path / "parent.d12"
    d12.write_text(parent_deck)
    (tmp_path / "parent.out").write_text("")
    sent = {}

    class Done:
        returncode = 0
        stderr = ""

    def run(cmd, **kwargs):
        options = cmd[cmd.index("--options-file") + 1]
        sent.update(json.loads(Path(options).read_text()))
        return Done()

    monkeypatch.setattr(ex_mod.subprocess, "run", run)
    ex.run_crystal_opt_conversion(str(tmp_path / "parent.out"), str(d12),
                                  tmp_path / "next", "SP", step)
    out = capsys.readouterr()
    return out.out + out.err, sent


@pytest.mark.parametrize("parent", ["external", "internal"])
@pytest.mark.parametrize("step,basis", [
    ({"method": "HF", "hf_method": "HF3C"}, "MINIX"),
    ({"method_modifications": {"new_functional": "PBEH3C"}}, "def2-mSVP"),
    # new_functional wins over hf_method, as in opt2d12
    ({"method": "HF", "hf_method": "HF3C",
      "method_modifications": {"new_functional": "PBE0"}}, None),
    ({"functional": "PBE0"}, None),
])
def test_executor_names_the_basis_the_step_uses(tmp_path, monkeypatch, capsys,
                                                 parent, step, basis):
    deck = ((GOLDEN / "agbr_PBE0.d12").read_text() if parent == "external"
            else INTERNAL_PARENT)
    log, sent = _executor_step(tmp_path, monkeypatch, capsys, deck, step)
    parent_claims = [ln for ln in log.splitlines()
                     if "Using original external basis" in ln or "Using external basis from" in ln
                     or "Preserving internal basis set" in ln
                     or "External basis detected but path not found" in ln]
    if basis:
        assert parent_claims == [], log
        assert len([ln for ln in log.splitlines() if "Basis set: " in ln]) == 1, log
        assert "the parent's basis is not used" in log and f"Basis set: {basis} (required by" in log
    else:
        assert len(parent_claims) == 1 and "Basis set: " not in log, log
    # the options handed to opt2d12 are what they were
    expected = {"calculation_type": "SP", "keep_current_settings": True}
    if parent == "external":
        expected["basis_set_type"] = "EXTERNAL"
        assert "use_original_external_basis" in sent
        expected["use_original_external_basis"] = sent["use_original_external_basis"]
        if "basis_set_path" in sent:
            expected["basis_set_path"] = sent["basis_set_path"]
    else:
        expected.update(basis_set_type="INTERNAL", basis_set="POB-TZVP-REV2")
    if "method_modifications" in step:
        expected["method_modifications"] = step["method_modifications"]
    assert sent == expected
