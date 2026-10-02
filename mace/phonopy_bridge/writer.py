"""Write the displaced-supercell decks, their .ext files and the manifest."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

from . import parent as P

MANIFEST = "mace_phonopy.json"
DISP_YAML = "phonopy_disp.yaml"
TEMPLATE = "TEMPLATE"
CRY_SYM = "CRY_SYM"


@contextlib.contextmanager
def working_directory(path) -> Iterator[None]:
    """Run the block inside ``path`` and always change back (contextlib.chdir
    needs Python 3.11; MACE supports older interpreters)."""
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def job_names(stem: str, tag: str, n: int) -> List[str]:
    width = max(3, len(str(n)))
    return [f"{stem}_ph{tag}_disp-{i:0{width}d}" for i in range(1, n + 1)]


def write_set(outdir: Path, stem: str, tag: str, deck: P.ParentDeck, template: P.Template,
              phonon, conv_super: Sequence[int], cry_sym: bool = False) -> List[str]:
    """One <job>.ext + <job>.d12 per displaced supercell; returns the job names.
    ``cry_sym`` writes each cell's symmetry operators into its .ext, as phonopy's
    writer does when a CRY_SYM file exists."""
    from phonopy.interface.crystal import get_crystal_structure

    cells = phonon.supercells_with_displacements
    names = job_names(stem, tag, len(cells))
    width = max(3, len(str(len(cells))))
    for i, (name, cell) in enumerate(zip(names, cells), start=1):
        (outdir / f"{name}.ext").write_text(get_crystal_structure(cell, list(conv_super), cry_sym))
        title = f"{deck.title.strip()} phonopy displacement {i:0{width}d}/{len(cells):0{width}d}"
        (outdir / f"{name}.d12").write_text(P.deck_text(title, deck, template))
    return names


def write_manifest(outdir: Path, data: Dict) -> Path:
    path = outdir / MANIFEST
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def read_manifest(outdir: Path) -> Dict:
    return json.loads((Path(outdir) / MANIFEST).read_text())
