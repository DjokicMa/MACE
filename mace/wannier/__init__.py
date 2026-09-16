"""MACE -> Wannier90 hand-off (Layer 2).

MACE generates the CRYSTAL matrix-dump deck (the MATDUMP calc type) and drives
the conversion. It does NOT implement the conversion.

    The LCAO->Wannier90 method, and the ``lcao2wannier`` package that
    implements it, are William Comaskey's work.

``lcao2wannier`` is an OPTIONAL dependency, imported defensively exactly like
``ase``/``seekpath``/``plotly`` elsewhere in MACE. Nothing in Layer 1 (deck
generation and submission) needs it, and its absence must never produce a
traceback.
"""

from mace.wannier.driver import (  # noqa: F401
    LCAO2WANNIER_CREDIT,
    ConversionResult,
    Lcao2WannierUnavailable,
    build_command,
    check_parent_dump,
    convert,
    describe_missing_dependency,
    find_lcao2wannier,
)
