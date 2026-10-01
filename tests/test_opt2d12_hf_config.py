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
