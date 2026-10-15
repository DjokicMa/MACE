"""Element symbols from CIF atom-site symbols and labels, whatever their case.

ase.io.cif takes the element from ``_atom_site_type_symbol`` (or, when that
column is missing, ``_atom_site_label``) with ``re.search("[A-Z][a-z]?")``.
That reads an all-caps "BI" as boron, "CL" as carbon and "SN3+" as sulfur, and
it fails outright on a lower-case "bi". MACE reads CIFs through
:func:`read_cif` (and :func:`normalize_block` for code that walks
``ase.io.cif.parse_cif`` blocks itself), which hands ase a symbol it reads
correctly and otherwise leaves ase's reading untouched. POSCAR species
lines get the same treatment from :func:`normalize_poscar_species`.

The rule (:func:`element_symbol`):

* Take the first run of letters in the token: "BI1" -> "BI", "Bi3+" -> "Bi",
  "O2-" -> "O", "Bi1a" -> "Bi".
* If its first two letters, capitalized ("BI" -> "Bi", "bi" -> "Bi"), are an
  element, that is the element. Only when they are not does the first letter
  alone name it ("Ow" -> "O", "HA1" -> "H", "B2" -> "B").
* "D" (deuterium) is kept as "D", which ase reads as H with the deuterium mass.

``_atom_site_type_symbol`` holds only the element and an optional charge, so
the rule is unambiguous there. A label without a type symbol can be
genuinely ambiguous in capitals: "CO1" may be cobalt or a carbon named "CO1",
"CA1" calcium or an alpha carbon. The two-letter element wins, as it does for
"BI1". A CIF that means otherwise should carry ``_atom_site_type_symbol``,
which is always preferred over the label.
"""

import re
from typing import List, Optional

_ASE_PICK = re.compile(r"[A-Z][a-z]?")          # what ase.io.cif matches
_LETTERS = re.compile(r"[A-Za-z]+")


def _valid() -> frozenset:
    from ase.data import chemical_symbols
    return frozenset(chemical_symbols[1:]) | {"D"}


def element_symbol(token) -> Optional[str]:
    """The element a CIF type symbol or label names, properly capitalized, or
    None when no element can be read from it. See the module docstring."""
    m = _LETTERS.search(str(token))
    if not m:
        return None
    letters = m.group(0)
    valid = _valid()
    two = letters[:2].capitalize()
    if len(two) == 2 and two in valid:
        return two
    one = letters[0].upper()
    return one if one in valid else None


def _ase_pick(token: str) -> Optional[str]:
    m = _ASE_PICK.search(str(token))
    return m.group(0) if m else None


def _fixed_column(values) -> Optional[List[str]]:
    """``values`` with each entry replaced by its element wherever ase would
    read a different one (or none); None when ase already reads them all."""
    out, changed = [], False
    for v in values:
        el = element_symbol(v)
        if el is not None and el != _ase_pick(v):
            out.append(el)
            changed = True
        else:
            out.append(v)
    return out if changed else None


def normalize_block(block):
    """An ase CIFBlock whose atom-site symbols ase reads as the right elements.

    The same block is returned when nothing needs changing. Otherwise a copy
    with ``_atom_site_type_symbol`` corrected; when the CIF has only labels,
    the copy gets a ``_atom_site_type_symbol`` column made from them, so the
    labels themselves stay as written."""
    from ase.io.cif import CIFBlock
    try:
        types = block.get("_atom_site_type_symbol")
        labels = block.get("_atom_site_label")
    except Exception:
        return block
    if isinstance(types, list):
        fixed = _fixed_column(types)
    elif types is None and isinstance(labels, list):
        fixed = _fixed_column(labels)
    else:
        return block
    if fixed is None:
        return block
    tags = dict(block.items())
    tags["_atom_site_type_symbol"] = fixed
    return CIFBlock(block.name, tags)


def read_cif(path, index=-1, **kwargs):
    """``ase.io.read(path, format="cif")`` with the element symbols read as
    :func:`element_symbol` reads them. ``index`` picks the data block (the
    last by default, as ase does); ``kwargs`` go to ``CIFBlock.get_atoms``."""
    from ase.io.cif import parse_cif
    images = []
    for block in parse_cif(str(path)):
        block = normalize_block(block)
        if block.has_structure():
            images.append(block.get_atoms(**kwargs))
    if not images:
        # Nothing readable: let ase raise its own error for this file.
        import ase.io
        return ase.io.read(str(path), format="cif", index=index)
    return images[index]


def normalize_poscar_species(text: str) -> str:
    """POSCAR/CONTCAR ``text`` with a VASP 5 species line ase can read.

    ase.io.vasp looks each species up exactly, so "BI S" or "bi s" raises
    KeyError. A species that is not already an element symbol is replaced
    by :func:`element_symbol` of it ("BI" -> "Bi"); the text is returned
    unchanged when nothing needs it. A VASP 4 file (counts on line 6, the
    species only in the free-text comment line) is left as it is."""
    lines = text.splitlines(keepends=True)
    if len(lines) < 7:
        return text
    words = lines[5].split()
    if not words or not words[0][:1].isalpha():
        return text
    valid = _valid()
    fixed = []
    for w in words:
        if w.startswith(("!", "#")):
            break
        el = None if w in valid else element_symbol(w)
        fixed.append(el or w)
    if fixed == words[:len(fixed)]:
        return text
    end = "\n" if lines[5].endswith("\n") else ""
    lines[5] = "   " + " ".join(fixed + words[len(fixed):]) + end
    return "".join(lines)
