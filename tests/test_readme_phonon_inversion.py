"""Crystal_d12/README.md says where the phonon SeeK-path path gets inversion.

Since the parent output reaches the SeeK-path helpers as a file
(d12_calc_freq._output_as_file), inversion comes from that output
(d3_kpoints.detect_inversion_from_crystal_output) and the space-group
number is only the fallback. The README still said the phonon path used the
number alone and that the output-based detection was not reached.
"""
import io
from pathlib import Path

from d12_calc_freq import write_frequency_section

README = Path(__file__).resolve().parent.parent / "Crystal_d12" / "README.md"

# F-43m by number (non-centrosymmetric), but the output calls it
# centrosymmetric: which one the path follows shows who decides.
OUTPUT = """\
 SPACE GROUP (CENTROSYMMETRIC)        :  F -4 3 M

 LATTICE PARAMETERS  (ANGSTROMS AND DEGREES) - CONVENTIONAL CELL
        A           B           C        ALPHA        BETA       GAMMA
     5.42000     5.42000     5.42000    90.00000    90.00000    90.00000

"""


def _source(crystal_system, output):
    settings = {"dispersion": True,
                "bands": {"path_method": "coordinates", "path": "auto",
                          "seekpath_full": True, "format": "seekpath",
                          "shrink": 16, "npoints": 100}}
    write_frequency_section(io.StringIO(), settings, crystal_system, 216, output)
    return settings["bands"]["kpath_source"]


def test_phonon_path_takes_inversion_from_the_output():
    assert _source("cubic", OUTPUT) == "seekpath_inv"
    assert _source("cubic-F", None) == "seekpath_noinv"


def test_readme_says_so():
    text = README.read_text()
    assert "in the phonon path it only has the space-group number" not in text
    assert "is not reached from here" not in text
    for sentence in [line for line in text.splitlines() if "**Inversion Symmetry" in line]:
        assert "detect_inversion_from_crystal_output" in sentence
        assert "_output_as_file" in sentence
