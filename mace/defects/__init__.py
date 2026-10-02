"""Point defects in periodic cells: deck sets, output parsing and analysis.

``mace defects build`` turns a single-point host deck into a pristine supercell
deck plus one deck per charge state, by editing the host's text (the method,
basis set, grids and tolerances are never re-rendered). ``mace defects potc``
stages the electrostatic-potential runs used for potential alignment and
``mace defects analyze`` reads the outputs back.

Modules:
    deck_edit  host classification and the text edits (standard library only)
    structure  geometry tables of CRYSTAL outputs, site resolution, PBC matching
    outputs    energies, electron counts, Mulliken charges/spins, band edges, POTC.DAT,
               dielectric tensors
    analysis   checks, charged-cell offsets, formation energies, levels
    image_charge  Ewald Madelung potential and the Lany-Zunger-scaled Makov-Payne term
    cli        the ``mace defects`` command
"""


class DefectError(Exception):
    """A deck set cannot be built (or analysed) as asked; the message says why."""


class NeedsPristineOutput(DefectError):
    """The defect decks need something only the pristine supercell output can tell
    (CRYSTAL's atom labels, the electron parity, coordinates)."""
