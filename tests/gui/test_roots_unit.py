"""mace/gui/roots.py: the allowed roots of Open Folder and the scratch folder
(one function for the title's "Scratch" and the dialog's chip)."""
import os
from pathlib import Path

import pytest

from mace.gui import roots as R


def test_scratch_from_env_then_common_patterns(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    env = {"HOME": str(tmp_path), "USER": "zqx", "SCRATCH": "", "WORK": str(work), "PSCRATCH": str(tmp_path / "no")}
    assert R.scratch_roots(env, patterns=()) == [work.resolve()]
    # $SCRATCH empty (as on clusters that set none): a user-owned folder from the patterns
    mnt = tmp_path / "mnt" / "scratch" / "zqx"
    mnt.mkdir(parents=True)
    pats = (str(tmp_path / "scratch" / "{user}"), str(tmp_path / "mnt" / "scratch" / "{user}"),
            str(tmp_path / "global" / "{user}"))
    got = R.scratch_roots({"HOME": str(tmp_path), "USER": "zqx", "SCRATCH": ""}, patterns=pats)
    assert got == [mnt.resolve()]
    # only the first existing pattern
    (tmp_path / "global" / "zqx").mkdir(parents=True)
    assert R.scratch_roots({"USER": "zqx"}, patterns=pats) == [mnt.resolve()]


def test_home_aliases_dedupe(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    assert R.home_aliases({"HOME": str(link)}) == [str(link), str(real)]
    assert R.home_aliases({"HOME": str(real)}) == [str(real)]


def test_allowed_roots_rules(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    env = {"HOME": str(tmp_path), "USER": "zqx"}
    assert R.allowed_roots(None, env)[0] == tmp_path.resolve()
    assert R.allowed_roots([str(a), str(b)], env) == [a.resolve(), b.resolve()]
    assert R.allowed_roots(["none"], env) is None
    assert R.allowed_roots(None, {**env, "MACE_GUI_ROOTS": ""}) is None
    assert R.allowed_roots(None, {**env, "MACE_GUI_ROOTS": os.pathsep.join([str(b), str(tmp_path / "gone")])}) \
        == [b.resolve()]


def test_credential_and_hidden_components():
    assert R.credential_component(("x", ".ssh", "id_ed25519")) == ".ssh"
    assert R.credential_component((".config", "gh", "hosts.yml")) == ".config/gh"
    assert R.credential_component((".config", "mace")) is None
    assert R.credential_component((".netrc",)) == ".netrc"
    assert R.hidden_component(("proj", ".git")) == ".git"
    assert R.hidden_component(("proj", "x.out")) is None


@pytest.mark.parametrize("parts", [
    (".pypirc",), (".npmrc",), ("proj", ".npmrc"), (".netrc",), (".git-credentials",), (".env",),
    ("proj", ".env"), ("proj", ".env.local"), (".bash_history",), (".zsh_history",),
    (".python_history",), (".psql_history",), (".Rhistory",), (".histfile",), (".docker", "config.json"),
    (".kube", "config"), (".aws", "credentials"), (".azure", "accessTokens.json"),
    (".config", "gcloud", "credentials.db"), (".gnupg", "private-keys-v1.d"), (".ssh", "config"),
    (".vault-token",), ("id_rsa",), ("proj", "keys", "id_ed25519"), ("id_ed25519_work",),
    ("id_ecdsa_sk",)])
def test_secret_files_are_hidden(parts):
    assert R.credential_component(parts), parts


@pytest.mark.parametrize("parts", [
    ("proj", "MgO.cif"), ("proj", "x.out"), ("run_history", "x.out"), ("proj", "id_list.txt"),
    ("id_test",), ("proj", "id_rsa.pub"), (".config", "mace"), ("environment.yml",), ("envs", "x.d12"),
    ("proj", "history.csv")])
def test_ordinary_files_are_not_hidden(parts):
    assert R.credential_component(parts) is None, parts


def test_project_secrets_do_not_make_a_home_folder(tmp_path):
    for name in (".env", ".npmrc", ".bash_history", "id_rsa"):
        (tmp_path / name).write_text("x")
    assert not R.holds_credentials(tmp_path)


def test_secret_refusal_says_hidden_for_safety():
    assert "hidden for safety" in R.secret_refusal(".npmrc")


def test_holds_credentials(tmp_path):
    assert not R.holds_credentials(tmp_path)
    (tmp_path / ".config" / "gh").mkdir(parents=True)
    assert R.holds_credentials(tmp_path)
