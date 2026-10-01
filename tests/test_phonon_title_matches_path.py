"""An opt2d12 SeeK-path phonon deck's title names the path the deck writes.

The title is written before the FREQCALC block and took its labels from
get_seekpath_labels(space_group, lattice_type) - the space-group number only -
while write_frequency_section reads the parent's output, whose lattice
parameters choose the SeeK-path variant. For AgBr in R-3c
(tests/data/opt_restart/tqc_agbr.out, a real CRYSTAL23 output) the title
listed GAMMA-L-B1-|-L-B-Z-GAMMA-X-Q-F-P1-Z-|-L-Z (11 segments) over a
10-segment hR2 path, and called it "default".
"""
import io
from pathlib import Path

import pytest

import CRYSTALOptToD12
from d12_calc_freq import write_frequency_section

DATA = Path(__file__).parent / "data" / "opt_restart"


def _freq_settings():
    return {
        "dispersion": True,
        "bands": {
            "path_method": "coordinates",
            "path": "auto",
            "seekpath_full": True,
            "format": "seekpath",
            "shrink": 16,
            "npoints": 100,
        },
    }


def _segments_named(labels):
    runs, run = [], []
    for label in labels + ["|"]:
        if label == "|":
            runs.append(run)
            run = []
        else:
            run.append(label)
    return sum(max(len(r) - 1, 0) for r in runs)


@pytest.mark.parametrize("out, space_group, system", [
    ("tqc_agbr.out", 167, "trigonal"),
    ("tqb_pto.out", 221, "cubic"),
])
def test_title_names_the_written_path(out, space_group, system):
    text = (DATA / out).read_text()
    geometry = {"spacegroup": space_group, "optimization_content": text}

    title_settings = _freq_settings()
    title = CRYSTALOptToD12._get_phonon_band_path_title(
        title_settings["bands"], geometry)

    deck_settings = _freq_settings()
    buf = io.StringIO()
    write_frequency_section(buf, deck_settings, system, space_group, text)
    lines = buf.getvalue().splitlines()
    i = lines.index("BANDS")
    n_segments = int(lines[i + 2])

    source, _, path = title.rpartition(" - ")
    labels = path.split("-")
    assert labels == deck_settings["bands"]["path_labels"]
    assert _segments_named(labels) == n_segments
    assert source.endswith("SeeKPath (w.I)")
