"""copy_dependencies.py copies every file it lists; none is missing.

Its list predated the move into the mace package, so it named files that no
longer exist and looked for Crystal_d12/Crystal_d3 under mace/, and copied
almost nothing. The copied Crystal scripts must also find every sibling module
they import, or they fail when run from the target directory.
"""
import ast
from pathlib import Path

from mace.utils.copy_dependencies import copy_dependencies

REPO = Path(__file__).resolve().parent.parent


def test_every_listed_file_is_copied(tmp_path, capsys):
    copied, missing = copy_dependencies(str(tmp_path))
    out = capsys.readouterr().out
    assert missing == 0, out
    assert copied == len(list(tmp_path.iterdir())) > 0
    assert not (tmp_path / "recovery_config.yaml").exists()


def test_copied_crystal_scripts_have_their_sibling_imports(tmp_path, capsys):
    copy_dependencies(str(tmp_path))
    capsys.readouterr()
    siblings = {p.stem for d in ("Crystal_d12", "Crystal_d3")
                for p in (REPO / d).glob("*.py")}
    copied = {p.stem for p in tmp_path.glob("*.py")}
    for name in sorted(copied & siblings):
        tree = ast.parse((tmp_path / f"{name}.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module.split(".")[0]]
            else:
                continue
            for mod in mods:
                if mod in siblings:
                    assert mod in copied, f"{name}.py imports {mod}, which is not copied"
