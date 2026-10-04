"""An optimisation that ended unconverged: the k-path is for the cell fort.9 holds.

"OPT END - FAILED" prints no FINAL OPTIMIZED GEOMETRY; the run's fort.9 is
the last point's SCF, whose geometry is the last block printed (measured on
HPCC, MSUCOF-4 OPT: point 800 of 800, its energy the one OPT END reports).
The lattice and structure were read from the first block (the input
geometry), so the BAND path did not match the wavefunction's cell. A run
killed mid-way (no OPT END) leaves an empty fort.9 and keeps the first block.
The output's layout is copied from that HPCC output.
"""
import pytest

np = pytest.importorskip("numpy")
d3_kpoints = pytest.importorskip("d3_kpoints")
si = pytest.importorskip("seekpath_interface")

import crystal_out_text as cot  # noqa: E402

START = (4.0, 4.0, 3.0, 90.0, 90.0, 90.0)
STEP = (4.1, 4.1, 3.2, 90.0, 90.0, 90.0)
ATOMS = [(20, "CA", 0, 0, 0)]


def _out(tmp_path, ended):
    text = (cot.echo("P 4/M M M", START)
            + cot.block(START, ATOMS, centering="1/0")
            + cot.block(STEP, ATOMS, header="COORDINATE AND CELL OPTIMIZATION - POINT    2",
                        centering="1/0"))
    if ended:
        text += (" CONVERGENCE TESTS UNSATISFIED AFTER 2 ENERGY AND GRADIENT CALCULATIONS\n\n"
                 " ******************************************************************\n"
                 " * OPT END - FAILED    * E(AU):  -5.144672299553E+04  POINTS    2 *\n"
                 " ******************************************************************\n")
    out = tmp_path / ("ended.out" if ended else "killed.out")
    out.write_text(text)
    return out


def test_unconverged_opt_reads_the_last_point(tmp_path):
    out = _out(tmp_path, ended=True)
    params = d3_kpoints.extract_lattice_parameters_from_output(str(out))
    assert (params["a"], params["c"]) == (4.1, 3.2)
    structure = si.parse_crystal_structure(str(out))
    assert structure["geometry"] == "last"
    assert np.isclose(structure["cell"][0][0], 4.1)


def test_killed_opt_still_reads_the_first_block(tmp_path):
    out = _out(tmp_path, ended=False)
    assert si.parse_crystal_structure(str(out))["geometry"] == "first"
    params = d3_kpoints.extract_lattice_parameters_from_output(str(out))
    assert (params["a"], params["c"]) == (4.0, 3.0)
