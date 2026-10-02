"""Menu option 3 (custom basis path) still accepts the bundled folders.

Up to 1.1.4 the bundled sets were directories of numbered files, so a user
could type .../basis_sets/full.basis.triplezeta/ at the custom-path prompt.
They are now one library file per family (basis_library.py) and that folder
no longer exists; the prompt said "Path does not exist" and asked again,
although every other code path still serves that path from the library.
"""
import builtins

import pytest

from conftest import REPO_ROOT

BASIS_SETS = REPO_ROOT / "Crystal_d12" / "basis_sets"


def _custom(monkeypatch, typed):
    import d12_constants

    def fake(prompt, options, default=None):
        if "Select basis set type" in prompt:
            return "1"
        if "Select external basis set" in prompt:
            return "3"
        raise AssertionError(prompt)

    answers = iter(typed)

    def fake_input(prompt=""):
        try:
            return next(answers)
        except StopIteration:
            raise AssertionError(f"prompt asked again: {prompt!r}")

    monkeypatch.setattr(d12_constants, "get_user_input", fake)
    monkeypatch.setattr(builtins, "input", fake_input)
    return d12_constants.select_basis_set([8, 12])


@pytest.mark.parametrize("family", ["full.basis.triplezeta", "full.basis.doublezeta"])
@pytest.mark.parametrize("slash", ["/", ""])
def test_the_old_bundled_folder_path_is_accepted(monkeypatch, capsys, family, slash):
    from d12_constants import read_basis_file

    typed = str(BASIS_SETS / family) + slash
    config = _custom(monkeypatch, [typed])
    assert config["basis_set"] == config["basis_set_path"] == str(BASIS_SETS / family) + "/"
    assert read_basis_file(config["basis_set"], 8).startswith("8 ")
    out = capsys.readouterr().out
    assert "does not exist" not in out and "not a directory" not in out
    assert family in out


def test_a_custom_directory_still_works(monkeypatch, tmp_path):
    (tmp_path / "8").write_text("8 1\n0 0 1 2.0 1.0\n 1.0 1.0\n")
    config = _custom(monkeypatch, [str(tmp_path)])
    assert config["basis_set"] == str(tmp_path) + "/"


def test_a_missing_path_is_still_refused(monkeypatch, tmp_path):
    config = _custom(monkeypatch, [str(tmp_path / "nope"), "n"])
    assert config["basis_set"].endswith("/full.basis.triplezeta/")
