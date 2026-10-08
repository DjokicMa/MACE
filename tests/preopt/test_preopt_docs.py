"""docs/preopt.md says what A23 needs it to say: the license table, every
citation and training-data attribution, the limits, the OPT-next rule, the
d12 refusal and the install hints; and the GUI guide and DOCUMENTATION.md
link to it."""
import pytest

from preopt_fixtures import REPO, preopt_sandbox  # noqa: F401

DOC = REPO / "docs" / "preopt.md"


@pytest.mark.parametrize("needle", [
    # license table and the models offered (and not)
    "| `mlip:acesuit-mace-mpa-0-medium`: MACE-MPA-0 (ACEsuit mace-torch) | mace-torch 0.3.16, MIT | MIT |",
    "| `mlip:chgnet-0.3.0`: CHGNet 0.3.0 | chgnet 0.4.2, BSD-3-Clause (LBNL) |",
    "ACEsuit's `mace-torch` package is unrelated to this MACE",
    "Academic Software License", "are not offered", "LGPL-3.0, run as a separate program, never bundled",
    # citations and data
    "Angew. Chem. Int. Ed. 59, 15665", "WIREs Comput. Mol. Sci. 11", "NeurIPS 2022", "arXiv:2401.00096",
    "Nat. Mach. Intell. 5, 1031", "MPtrj", "CC BY 4.0", "arXiv:2410.12771", "J. Phys.\n  Mater. 9, 025014 (2026)",
    # limits and rules
    "The next CRYSTAL step must be an OPT", "GFN-FF ignores it", "GFN-FF ignores `--uhf`",
    "GGA/GGA+U", "about 1 % too large", "ML is refused above 300 atoms on the CPU in the GUI",
    "accepts a wrong `--charge`\n  silently",
    # the d12 refusal and its route
    "`mace preopt` refuses such a deck (exit 1)", "opening the .out of a run of the\ndeck or exporting a CIF",
    # install hints
    'conda install -c conda-forge "xtb>=6.7"', "module load xtb/6.7.1-gfbf-2024a", "export MACE_XTB=~/bin/xtb-gfnff",
    "Lmod refuses to load both", "mace mlip setup --python DIR/bin/python",
    "mace mlip fetch acesuit-mace-mpa-0-medium", "https://download.pytorch.org/whl/cpu",
    "MACE_XTB", "MACE_MLIP_PYTHON", "MACE_MLIP_DIR", "MACE_MLIP_NO_DOWNLOAD",
    "where it is called `cuda:0`",
])
def test_docs_say_it(needle):
    assert needle in DOC.read_text(), needle


def test_docs_are_linked():
    assert "(preopt.md)" in (REPO / "docs" / "gui.md").read_text()
    assert "docs/preopt.md" in (REPO / "DOCUMENTATION.md").read_text()
