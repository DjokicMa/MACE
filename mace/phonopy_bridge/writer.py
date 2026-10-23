"""The displaced supercells' .ext files, phonopy_disp.yaml and the manifest."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

MANIFEST = "mace_phonopy.json"
DISP_YAML = "phonopy_disp.yaml"
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


def job_names(stem: str, n: int) -> List[str]:
    """<stem>_disp-NNN for each displaced supercell (stem: the name opt2d12
    gives the set, e.g. mat_opt_ph222_PBE0_optimized)."""
    width = max(3, len(str(n)))
    return [f"{stem}_disp-{i:0{width}d}" for i in range(1, n + 1)]


def write_ext_files(outdir: Path, names: Sequence[str], phonon, conv_super: Sequence[int],
                    cry_sym: bool = False) -> None:
    """One <job>.ext per displaced supercell, by phonopy's own CRYSTAL writer
    (the job script stages it as fort.34; the deck itself is opt2d12's).
    ``cry_sym`` writes each cell's symmetry operators into its .ext, as
    phonopy's writer does when a CRY_SYM file exists."""
    from phonopy.interface.crystal import get_crystal_structure

    for name, cell in zip(names, phonon.supercells_with_displacements):
        (outdir / f"{name}.ext").write_text(get_crystal_structure(cell, list(conv_super), cry_sym))


def write_manifest(outdir: Path, data: Dict) -> Path:
    path = outdir / MANIFEST
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def read_manifest(outdir: Path) -> Dict:
    return json.loads((Path(outdir) / MANIFEST).read_text())
