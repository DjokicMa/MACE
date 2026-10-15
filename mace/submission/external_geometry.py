"""Where an EXTERNAL deck's fort.34 comes from.

A deck whose geometry record (the line after the title) is EXTERNAL or DLVINPUT
holds no geometry: "The complete geometry input data are read from file
fort.34" (CRYSTAL23 manual p.21; file format pp.450-451). MACE's job script
(submitcrystal23.sh) stages ``<stem>.ext`` next to the deck as fort.34. A job
script written by hand may copy another file there instead, and MACE submits
an existing ``<stem>.sh`` as it is, so that file is the one the job will use.

Shared by `mace gui` (to draw the geometry), `mace preflight` (to stage it) and
`mace submit` (to warn before a job goes to the queue without one). Standard
library only.
"""
import re
import shlex
from pathlib import Path
from typing import List, Optional

EXTERNAL_KEYWORDS = ("EXTERNAL", "DLVINPUT")

#: Files in fort.34 format that MACE's job script does NOT stage: worth naming
#: when the staged one is missing.
UNSTAGED_SUFFIXES = (".gui", ".f34")

_SEGMENT_SPLIT = re.compile(r";|&&|\|\|")


def geometry_keyword(text: str) -> Optional[str]:
    """EXTERNAL or DLVINPUT when the deck's geometry record is one, else None.

    The first record is the title (free text, never a keyword); the geometry
    record is the first non-blank line after it.
    """
    lines = text.splitlines()
    for line in lines[1:]:
        word = line.strip().upper()
        if word:
            return word if word in EXTERNAL_KEYWORDS else None
    return None


def deck_keyword(deck: Path) -> Optional[str]:
    try:
        return geometry_keyword(Path(deck).read_text(errors="replace"))
    except OSError:
        return None


def _script_sources(deck: Path) -> List[Path]:
    """The files ``<stem>.sh`` copies to fort.34, with $DIR / $JOB (and
    $SLURM_SUBMIT_DIR) replaced by the deck's folder and stem. A source that
    still holds a variable after that cannot be resolved and is left out."""
    script = deck.with_suffix(".sh")
    try:
        text = script.read_text(errors="replace")
    except OSError:
        return []
    names = {"DIR": str(deck.parent), "SLURM_SUBMIT_DIR": str(deck.parent), "JOB": deck.stem}
    found: List[Path] = []
    for line in text.splitlines():
        for segment in _SEGMENT_SPLIT.split(line):
            try:
                words = shlex.split(segment, comments=True)
            except ValueError:
                continue
            if "cp" not in words:
                continue
            args = [w for w in words[words.index("cp") + 1:] if not w.startswith("-")]
            if len(args) < 2 or Path(args[-1]).name != "fort.34":
                continue
            src = args[-2]
            for name, value in names.items():
                src = src.replace("${%s}" % name, value).replace("$" + name, value)
            if "$" in src or "`" in src:
                continue
            p = Path(src)
            found.append(p if p.is_absolute() else deck.parent / p)
    return found


def fort34_sources(deck) -> List[Path]:
    """Files the deck's job would stage as fort.34, most specific first: what
    its own ``<stem>.sh`` copies, then ``<stem>.ext`` (MACE's job script)."""
    deck = Path(deck)
    out: List[Path] = []
    for p in _script_sources(deck) + [deck.with_suffix(".ext")]:
        if p not in out:
            out.append(p)
    return out


def fort34_source(deck) -> Optional[Path]:
    """The first of fort34_sources that exists, else None."""
    for p in fort34_sources(deck):
        if p.is_file():
            return p
    return None


def unstaged_geometry_files(deck) -> List[Path]:
    """Same-format files next to the deck that its job would not stage."""
    deck = Path(deck)
    staged = set(fort34_sources(deck))
    cands = [deck.with_suffix(s) for s in UNSTAGED_SUFFIXES] + [deck.parent / "fort.34"]
    return [p for p in cands if p.is_file() and p not in staged]


def missing_fort34_message(deck, text: Optional[str] = None) -> Optional[str]:
    """None when the deck is not EXTERNAL/DLVINPUT or its fort.34 source is
    there; else a plain sentence saying what is missing and what to do."""
    deck = Path(deck)
    if text is None:
        keyword = deck_keyword(deck)
    else:
        keyword = geometry_keyword(text)
    if keyword is None or fort34_source(deck) is not None:
        return None
    script = _script_sources(deck)
    ext = deck.with_suffix(".ext")
    if script:
        src = script[0]
        shown = src.name if src.parent == deck.parent else str(src)
        where = f"its job script {deck.stem}.sh copies {shown} to fort.34, and {shown} is missing"
        todo = f"Put {shown} in place (the structure in fort.34 or .gui format)"
        target = shown
    else:
        where = (f"there is no {ext.name} next to it (MACE's job script copies "
                 "<deck name>.ext to fort.34)")
        todo = f"Save the structure next to the deck as {ext.name} (fort.34 or .gui format)"
        target = ext.name
    msg = (f"{deck.name} reads its geometry from an external file ({keyword}: CRYSTAL "
           f"reads fort.34), but {where}. {todo}; without it CRYSTAL stops with no geometry.")
    stray = unstaged_geometry_files(deck)
    if stray:
        msg += (f" {stray[0].name} is next to the deck, but the job script does not "
                f"stage it: copy it to {target}.")
    return msg
