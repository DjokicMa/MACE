"""Point defects in periodic cells: deck sets, output parsing and analysis.

Modules:
    deck_edit  host classification and the text edits (standard library only)
"""


class DefectError(Exception):
    """A deck set cannot be built (or analysed) as asked; the message says why."""


class NeedsPristineOutput(DefectError):
    """The defect decks need something only the pristine supercell output can tell
    (CRYSTAL's atom labels, the electron parity, coordinates)."""
