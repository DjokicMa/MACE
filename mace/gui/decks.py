"""Deck preview and options-JSON saving: the GUI -> JSON -> CLI loop.

The preview runs the real CLI (``mace convert --batch --options_file`` or
``mace opt2d12 --config-file --yes``) in a scratch directory, with stdin
closed, and returns the deck it writes. The GUI therefore never has a deck
writer of its own: what it shows is byte-for-byte what the CLI writes from
the same JSON file, and "Save JSON" writes exactly that file.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parents[2]
MACE_CLI = _REPO / "mace_cli"
EXAMPLES_D12 = _REPO / "Crystal_d12" / "example_configs"

if str(_REPO / "Crystal_d12") not in sys.path:
    sys.path.insert(0, str(_REPO / "Crystal_d12"))

MODES = ("cif2d12", "opt2d12")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}\.json$")


class DeckError(ValueError):
    """The JSON or the structure cannot make a deck (shown to the user)."""


def templates(summary: bool = False) -> List[Dict[str, Any]]:
    """The shipped example configs (Crystal_d12/example_configs/*.json); with
    ``summary`` only their names and descriptions (the page's boot request,
    which has to stay small; the configs come from /api/templates when one is
    loaded)."""
    out = []
    for path in sorted(EXAMPLES_D12.glob("*.json")):
        try:
            text = path.read_text()
            data = json.loads(text)
        except (OSError, json.JSONDecodeError):
            continue
        conf = data.get("configuration", data) if isinstance(data, dict) else {}
        # "text": the file's own bytes, which Setup hands to the CLI unchanged
        row = {"name": path.stem, "description": conf.get("description", ""), "calculation_type": conf.get("calculation_type")}
        out.append(row if summary else {**row, "json": data, "text": text})
    return out


def validate(mode: str, data: Any) -> List[str]:
    """Problems that would stop the CLI from using this JSON (empty = fine)."""
    from d12_config import config_to_cif_options, unwrap_d12_config

    if mode not in MODES:
        return [f"unknown mode {mode!r}"]
    if not isinstance(data, dict):
        return ["the options file must be a JSON object"]
    if mode == "cif2d12":
        try:
            config_to_cif_options(data)
        except Exception as exc:
            return [str(exc)]
        return []
    conf = unwrap_d12_config(data)
    if not isinstance(conf, dict):
        return ["no settings object in the config file"]
    from d12_constants import normalize_config_functionals
    return list(normalize_config_functionals(dict(conf))[1])


def cli_env() -> Dict[str, str]:
    """The CLI children's environment (clirun.cli_env, kept here for older callers)."""
    from .clirun import cli_env as _env
    return _env()


def _run(cmd: List[str], cwd: Path, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=timeout,
                          env=cli_env())


def command_line(mode: str, json_name: str, structure: Path) -> str:
    """The CLI line that turns ``structure`` + the saved JSON into a deck."""
    if mode == "cif2d12":
        return (f"mace convert --batch --options_file {json_name} "
                f"--cif_dir {structure.parent} --output_dir decks/")
    return (f"mace opt2d12 --out-file {structure} --config-file {json_name} "
            f"--yes --output-dir decks/")


def preview(mode: str, data: Dict[str, Any], structure: Path) -> Dict[str, Any]:
    """Write the deck the CLI writes for ``structure`` with this JSON."""
    problems = validate(mode, data)
    if problems:
        raise DeckError("; ".join(problems))
    structure = Path(structure)
    if mode == "cif2d12" and structure.suffix.lower() != ".cif":
        raise DeckError("cif2d12 previews take a .cif structure")
    if mode == "opt2d12" and structure.suffix.lower() not in (".out", ".o"):
        raise DeckError("opt2d12 previews take a CRYSTAL .out")

    work = Path(tempfile.mkdtemp(prefix="mace_gui_deck_"))
    try:
        opts = work / "options.json"
        opts.write_text(json.dumps(data, indent=2))
        out_dir = work / "out"
        if mode == "cif2d12":
            cif_dir = work / "cif"
            cif_dir.mkdir()
            shutil.copy2(structure, cif_dir / structure.name)
            cmd = [sys.executable, str(MACE_CLI), "convert", "--no-banner", "--batch",
                   "--options_file", str(opts), "--cif_dir", str(cif_dir),
                   "--output_dir", str(out_dir)]
        else:
            in_dir = work / "in"
            in_dir.mkdir()
            shutil.copy2(structure, in_dir / structure.name)
            deck = structure.with_suffix(".d12")
            if deck.is_file():
                shutil.copy2(deck, in_dir / deck.name)
            cmd = [sys.executable, str(MACE_CLI), "opt2d12", "--no-banner",
                   "--out-file", str(in_dir / structure.name), "--config-file",
                   str(opts), "--yes", "--output-dir", str(out_dir)]
        result = _run(cmd, work)
        log = (result.stdout + result.stderr).replace(str(work) + os.sep, "")
        decks = sorted(out_dir.glob("*.d12")) if out_dir.is_dir() else []
        if result.returncode != 0 or not decks:
            raise DeckError("the CLI wrote no deck:\n" + log[-3000:])
        return {"mode": mode, "deck_name": decks[0].name,
                "deck": decks[0].read_text(), "log": log[-6000:],
                "command": command_line(mode, "<saved>.json", structure)}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def save_json(root: Path, name: str, mode: str, data: Dict[str, Any],
              overwrite: bool = False, expected: Optional[Dict[str, Any]] = None) -> Path:
    """Write ``data`` as ``root/name`` (a plain file name ending .json)."""
    if not _SAFE_NAME.match(name or ""):
        raise DeckError("file name must be a plain name ending in .json "
                        "(letters, digits, . _ -)")
    problems = validate(mode, data)
    if problems:
        raise DeckError("; ".join(problems))
    target = Path(root) / name
    if target.is_symlink():
        # saving.py's rule: a link is never written through (it may lead out of --dir)
        raise PermissionError(f"{name} is a symbolic link; not overwritten")
    if target.exists() and not overwrite:
        raise DeckError(f"{name} already exists (tick overwrite to replace it)")
    from .saving import write_atomic, _UNSET
    existed = target.exists()
    # the stamp the browser was shown (or "did not exist"), checked again right before the rename
    write_atomic(target, (json.dumps(data, indent=2) + "\n").encode(),
                 expected=expected if expected is not None else (None if not existed else _UNSET),
                 reason="Saved the settings JSON", route="deck/save")
    return target
