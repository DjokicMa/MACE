"""MACE -> Wannier90 hand-off (Layer 2).

MACE generates the CRYSTAL matrix-dump deck (the MATDUMP calc type) and drives
the conversion. It does NOT implement the conversion.

    The LCAO->Wannier90 method, and the ``lcao2wannier`` package that
    implements it, are William Comaskey's work.

``lcao2wannier`` is bundled here as ``mace.wannier.lcao2wannier`` under its
own MIT license (see ``lcao2wannier/LICENSE`` and ``lcao2wannier/VENDORED.md``
for the source version and every local change). Its package ``__init__`` is
lazy, so importing this module does not pull in numpy or scipy; nothing in
Layer 1 (deck generation and submission) needs them.
"""

from mace.wannier.driver import (  # noqa: F401
    LCAO2WANNIER_CREDIT,
    LCAO2WANNIER_MODULE,
    LCAO2WANNIER_VERSION,
    ConversionResult,
    Lcao2WannierUnavailable,
    build_command,
    convert,
)
