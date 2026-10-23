"""The bundled example suite: five material archetypes across the CRYSTAL calculation types.

The suite lives in ``examples/suite/`` (built by ``examples/gui/build_examples.py
--suite``). Large text files (outputs, wavefunction-derived grids, spectra) are
stored xz-compressed next to a ``MANIFEST.json`` that records every file's
SHA-256, so the repository stays small. This module restores ("hydrates") them
on first use: ``mace examples hydrate``, ``mace examples copy DEST``, and the
GUI does it by itself the first time its Explorer lists a suite folder.

Standard library only; nothing here imports an optional package.
"""
from __future__ import annotations

import hashlib
import json
import lzma
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

KIND = "mace-example-suite"
MANIFEST = "MANIFEST.json"
CALC_TYPES = ("OPT", "SP", "FREQ", "BAND", "DOSS", "TRANSPORT", "CHARGE+POTENTIAL", "ORBITALS",
              "PHONON_BAND", "PHONON_DOS", "ELASTIC", "EOS", "QHA", "CPHF")
STATUS_WORDS = {"real": "output", "failed": "failed run", "pending": "deck only", "n/a": "n/a"}

_DONE: set = set()          # suite folders already hydrated (or found writable-less) this process


def default_suite_dir() -> Path:
    """examples/suite of this MACE checkout."""
    return Path(__file__).resolve().parents[1] / "examples" / "suite"


def load_manifest(root: Optional[Path] = None) -> Dict[str, Any]:
    root = Path(root) if root else default_suite_dir()
    data = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    if data.get("kind") != KIND:
        raise ValueError(f"{root / MANIFEST} is not a MACE example-suite manifest")
    return data


def find_suite(path: Path) -> Optional[Path]:
    """The suite folder that is ``path``, holds it (up to 5 levels up) or sits directly inside it."""
    p = Path(path)
    for c in [p, p / "suite", *list(p.parents)[:5]]:
        try:
            m = c / MANIFEST
            if m.is_file() and json.loads(m.read_text(encoding="utf-8")).get("kind") == KIND:
                return c
        except (OSError, ValueError):
            continue
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def files(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(manifest.get("files", []))


def hydrate(root: Optional[Path] = None, *, force: bool = False,
            only: Optional[Iterable[str]] = None) -> Dict[str, List[str]]:
    """Decompress every stored ``.xz`` of the suite next to itself.

    Returns {"written": [...], "present": [...], "errors": [...]} (paths relative
    to the suite). A present file with the right SHA-256 is left alone; one with
    the wrong hash is rewritten. ``only`` limits it to paths starting with one of
    the given prefixes.
    """
    root = Path(root) if root else default_suite_dir()
    man = load_manifest(root)
    prefixes = tuple(only or ())
    out: Dict[str, List[str]] = {"written": [], "present": [], "errors": []}
    for f in files(man):
        rel, stored = f["path"], f.get("stored")
        if not stored or stored == rel:
            continue
        if prefixes and not rel.startswith(prefixes):
            continue
        dst, src = root / rel, root / stored
        try:
            if dst.is_file() and not force and _sha256(dst) == f["sha256"]:
                out["present"].append(rel)
                continue
            if not src.is_file():
                out["errors"].append(f"{rel}: {stored} is missing")
                continue
            tmp = dst.with_name(dst.name + ".part")
            with lzma.open(src, "rb") as zin, open(tmp, "wb") as fh:
                shutil.copyfileobj(zin, fh)
            if _sha256(tmp) != f["sha256"]:
                tmp.unlink(missing_ok=True)
                out["errors"].append(f"{rel}: checksum differs from the manifest")
                continue
            os.replace(tmp, dst)
            out["written"].append(rel)
        except (OSError, EOFError, lzma.LZMAError) as exc:
            dst.with_name(dst.name + ".part").unlink(missing_ok=True)
            out["errors"].append(f"{rel}: {exc}")
    return out


def hydrate_for_root(served_root: Path) -> Optional[Dict[str, List[str]]]:
    """GUI hook: hydrate the suite the first time a folder in or around one is listed.

    A read-only install (no write permission) is left as is; ``mace examples copy``
    makes a writable copy. Never raises.
    """
    try:
        suite = find_suite(Path(served_root))
        if suite is None:
            return None
        key = str(suite.resolve())
        if key in _DONE:
            return None
        _DONE.add(key)
        return hydrate(suite)
    except Exception:
        return None


def copy_suite(dest: Path, *, hydrated: bool = True) -> Path:
    """Copy the suite to DEST (which must not exist or be empty).

    hydrated=True unpacks the compressed files in DEST and leaves the .xz copies out;
    the suite in the install itself is not touched.
    """
    src = default_suite_dir()
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"{dest} exists and is not empty")
    man = load_manifest(src)
    dest.mkdir(parents=True, exist_ok=True)
    names = {MANIFEST, "README.md"}
    for f in files(man):
        names.add(f.get("stored") or f["path"])
    for rel in sorted(names):
        if (src / rel).is_file():
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / rel, dest / rel)
    if hydrated:
        res = hydrate(dest)
        if res["errors"]:
            raise OSError("; ".join(res["errors"]))
        for f in files(man):
            if f.get("stored") and f["stored"] != f["path"]:
                (dest / f["stored"]).unlink(missing_ok=True)
    return dest


def matrix(manifest: Dict[str, Any]) -> List[List[str]]:
    """Rows [archetype, entry, status per calc type...] for ``mace examples list``."""
    rows = []
    for arch in manifest["archetypes"]:
        for ent in arch["entries"]:
            rows.append([arch["id"], ent["id"]] + [ent["cells"].get(ct, {}).get("status", "-") for ct in CALC_TYPES])
    return rows


def pending(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for arch in manifest["archetypes"]:
        for ent in arch["entries"]:
            for ct in CALC_TYPES:
                c = ent["cells"].get(ct, {})
                if c.get("status") == "pending":
                    out.append({"archetype": arch["id"], "entry": ent["id"], "calc": ct, **c})
    return out


def main(argv: Optional[List[str]] = None) -> int:
    """``mace examples [list | hydrate | copy DEST | path]``."""
    import argparse
    ap = argparse.ArgumentParser(
        prog="mace examples",
        description="The bundled example suite: inputs and CRYSTAL23 outputs for five material "
                    "archetypes (porous framework, dense 3D, 2D slab, 1D polymer, 0D molecule) "
                    "across the calculation types, to try the GUI and the plots without running anything first.",
        epilog="Examples:\n  mace examples                   the coverage table\n"
               "  mace examples hydrate           unpack the compressed files in place\n"
               "  mace examples copy ~/mace_examples\n  mace gui --dir $(mace examples path) --open",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("list", help="the archetype x calculation-type table (default)")
    h = sub.add_parser("hydrate", help="unpack the compressed files next to themselves")
    h.add_argument("--force", action="store_true", help="rewrite files that are already there")
    c = sub.add_parser("copy", help="copy the unpacked suite to a new folder")
    c.add_argument("dest")
    sub.add_parser("path", help="print the suite folder (unpacked first)")
    args = ap.parse_args(argv)
    cmd = args.cmd or "list"
    root = default_suite_dir()
    try:
        man = load_manifest(root)
    except (OSError, ValueError) as exc:
        print(f"mace examples: the example suite is not available here ({exc}).")
        return 1
    if cmd == "list":
        # one column per entry, one row per calculation type: 14 types do not fit across a terminal
        rows = matrix(man)
        word = {"real": "output", "failed": "failed", "pending": "deck", "n/a": "n/a"}
        ws = [max(len(r[0]), len(r[1]), 6) + 2 for r in rows]
        print(f"MACE example suite ({root})\n")
        print(f"{'':<18}" + "".join(f"{r[0]:<{w}}" for r, w in zip(rows, ws)))
        print(f"{'':<18}" + "".join(f"{r[1]:<{w}}" for r, w in zip(rows, ws)))
        for i, ct in enumerate(CALC_TYPES):
            print(f"{ct:<18}" + "".join(f"{word.get(r[2 + i], r[2 + i]):<{w}}" for r, w in zip(rows, ws)))
        print("\noutput = a real CRYSTAL23 output of this system is included; failed = failed run: its real output is "
              "included but CRYSTAL stopped early (the manifest note says why); deck = deck only: the input deck is included "
              "and its output is added when it has been run; n/a = CRYSTAL does not run it for this system, or "
              "gives no usable result (the manifest says why); "
              "- = shipped by another entry of the archetype.")
        return 0
    if cmd == "hydrate":
        res = hydrate(root, force=args.force)
        print(f"unpacked {len(res['written'])} file(s), {len(res['present'])} already present")
        for e in res["errors"]:
            print(f"  error: {e}")
        return 1 if res["errors"] else 0
    if cmd == "copy":
        try:
            dest = copy_suite(Path(args.dest).expanduser())
        except (OSError, ValueError) as exc:
            print(f"mace examples: {exc}")
            return 1
        print(f"copied to {dest}")
        return 0
    if cmd == "path":
        res = hydrate(root)
        for e in res["errors"]:
            print(f"error: {e}")
        print(root)
        return 1 if res["errors"] else 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
