"""The served folder's own settings (``mace-folder.toml``) for the GUI.

The file and its checks are mace/utils/folder_settings.py, shared with the
CLI. Here: what the page needs to show each setting's layer (``view``) and
the Settings dialog's "Save to this folder" / "Reset to my default"
(``save``), which write the file of the served folder only:

* the file found by walking up from the served folder is used as it is;
  when it lies above the served folder it is read but never written here
  (the page says where it is), so the GUI's writes stay inside ``--dir``;
* a file MACE refuses (bad TOML, writable by everyone, a symbolic link) is
  not overwritten either: it is fixed by hand first;
* the previous version goes to File History before each write.

A save is always a click in Settings: ``mace gui`` never writes the file on
its own.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from mace.utils import folder_settings as F


def view(root: Path) -> Dict[str, Any]:
    """The folder layer for a page: the GUI values the file sets, its CLI-only
    values, its warnings, and whether (and why not) Save to this folder works
    (folder_settings.save_target: the refusals `mace site set --folder` has)."""
    root = Path(root)
    fs = F.load(root)
    target, why = F.save_target(root)
    can, why = why is None, why or ""
    return {"file": str(fs.path) if fs.path else None, "target": str(target), "name": F.FILENAME,
            "can_save": can, "why_not": why, "gui": fs.gui(root), "cli": fs.cli_only(),
            "keys": dict(F.GUI_KEYS), "warnings": list(fs.warnings),
            # the values a choice key can hold in the file (Save to this folder is offered for those only)
            "choices": {g: list(F.KEYS[k]["choices"]) for g, k in F.GUI_KEYS.items() if F.KEYS[k]["type"] == "choice"},
            "help": {k: F.KEYS[k]["help"] for k in fs.cli_only()}}


def cli_env(root: Path) -> Dict[str, str]:
    """The environment of a CLI child that runs in the server's cache folder
    for the served folder (mace plotting for a plot or the gallery): the
    served folder's file pinned (MACE_FOLDER_SETTINGS), or none, never one
    found from the cache folder."""
    from . import clirun
    env = clirun.cli_env()
    f = F.find(Path(root))
    env[F.ENV] = str(f) if f is not None else "none"
    return env


def energy_window(root: Path):
    """The served folder's ``[plots] energy_window`` (low, high), or None."""
    return F.load(Path(root)).get("plots.energy_window")


def save(root: Path, sets: Any, unsets: Any = ()) -> Dict[str, Any]:
    """Set (``sets``: {GUI settings key: value}) and remove (``unsets``: [GUI
    key]) values in the served folder's file; returns the new view. Raises
    ValueError for a key the folder file cannot hold, a bad value, or a file
    this server may not write (see ``view``)."""
    root = Path(root)
    if not isinstance(sets, dict) or not isinstance(unsets, (list, tuple)):
        raise ValueError("set must be {key: value} and unset a list of keys")
    v = view(root)
    if not v["can_save"]:
        raise ValueError(v["why_not"])
    target = Path(v["target"])
    file_sets: Dict[str, Any] = {}
    file_unsets = []
    for k, val in sets.items():
        key = F.GUI_KEYS.get(k)
        if key is None:
            raise ValueError(f"{k} cannot be saved to a folder")
        if F.KEYS[key]["type"] == "path" and val in ("", None):
            file_unsets.append(key)            # a blank path is "the built-in choice": nothing to keep
            continue
        file_sets[key] = val
    for k in unsets:
        key = F.GUI_KEYS.get(k) if isinstance(k, str) else None
        if key is None:
            raise ValueError(f"{k} cannot be saved to a folder")
        file_unsets.append(key)
    if not file_sets and not file_unsets:
        return v
    existed = target.exists()
    _keep_history(target, existed)
    F.write(target, file_sets, file_unsets)
    if not existed:
        _record_created(target)
    return view(root)


def _keep_history(target: Path, existed: bool) -> None:
    if not existed:
        return
    try:
        from . import history
        history.keep(target, "Folder settings saved from Settings", route="settings", kind="input")
    except Exception:              # File History is a convenience here, never a reason to refuse a save
        pass


def _record_created(target: Path) -> None:
    try:
        from . import history
        history.record_created(target, "Folder settings created from Settings", route="settings")
    except Exception:
        pass

