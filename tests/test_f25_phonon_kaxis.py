"""The k axis of a phonon fort.25 (mace/plotting/f25.py ``phonon_path_k``), on a real CRYSTAL23 run.

FREQCALC BANDS writes a BAND block's DK as the segment's length (bohr^-1), not the step between two points the
manual gives for a PROPERTIES run: diamond's Γ-X, X-W, W-L and L-Γ are 2π/a, π/a, √2π/a and √3π/a. Each line is
50 points end to end, and every later block drops its start (the output lists it again, the fort.25 does not),
so a later block's first point is one step (DK/49) past the previous end. Checked against the run's own output
(tests/data/f25/dia_phonon_f25.dispersion.txt, a byte-exact excerpt of its PHONON BANDS print): every point of the
converted path sits at the distance of the output's k point with the same frequencies. The electronic DK (a step)
is untouched (test_f25_plotting.py, the corpus).
"""
import math
import re

import numpy as np
import pytest

from conftest import REPO_ROOT
from mace.plotting import f25 as F
from mace.properties.defs import vibrations as VB

PH = REPO_ROOT / "tests" / "data" / "f25"
PH_F25 = PH / "dia_phonon_f25.fort.25"
A_CONV = 3.54481120                       # the .d12's conventional a (Å)
BOHR_A = 0.529177210903
CM_PER_HARTREE = 219474.6313632


def _recip():
    """CRYSTAL's primitive fcc reciprocal vectors (rows), bohr^-1."""
    g = 2 * math.pi / (A_CONV / BOHR_A)
    return g * np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], float)


def _out_points():
    """[(path, k number, fractional k, 6 frequencies cm-1)] of the excerpt of the run's output."""
    pts, path, den = [], None, None
    lines = (PH / "dia_phonon_f25.dispersion.txt").read_text().splitlines()
    for ln in lines:
        m = re.match(r"\s+PHONONS ALONG PATH:\s+(\d+)", ln)
        if m:
            path = int(m.group(1))
        m = re.search(r"IN UNITS OF DENOMINATOR\s+(\d+)", ln)
        if m:
            den = int(m.group(1))
        m = re.match(r"\s+DISPERSION K POINT NUMBER\s+(\d+) COORD:\s+\w\(\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*\)", ln)
        if m:
            pts.append([path, int(m.group(1)), np.array([int(m.group(k)) for k in (2, 3, 4)], float) / den, []])
        m = re.match(r"\s+\d+-\s+\d+\s+\S+\s+(-?\d+\.\d+)", ln)
        if m and pts:
            pts[-1][3].append(float(m.group(1)))
    return pts


def _dat_rows():
    rows = [ln.split() for ln in F.band_dat_text(PH_F25).splitlines() if ln and ln[0] not in "#@"]
    arr = np.array(rows, float)
    arr[:, 1:] *= CM_PER_HARTREE
    return arr


def test_the_header_dk_of_a_phonon_block_is_the_segment_length():
    b = _recip()
    segs = [x for x in F.read_blocks(PH_F25, ("BAND",)) if x.kind == "BAND"]
    assert [s.ncol for s in segs] == [50, 49, 49, 49]
    for s in segs:
        k0, k1 = np.array(s.ints[:3]) / 8 @ b, np.array(s.ints[3:6]) / 8 @ b
        assert s.dk == pytest.approx(np.linalg.norm(k1 - k0), rel=2e-6)
    g = 2 * math.pi / (A_CONV / BOHR_A)
    assert [s.dk for s in segs] == pytest.approx([g, g / 2, g * math.sqrt(2) / 2, g * math.sqrt(3) / 2], rel=2e-6)


def test_the_phonon_k_axis_is_the_path_distance_of_the_output_k_points():
    b = _recip()
    pts = _out_points()
    assert [p[1] for p in pts] == [1, 2, 49, 50, 51, 52, 99, 100, 101, 102, 149, 150, 151, 152, 199, 200]
    starts = {p[0]: p[2] for p in pts if p[1] in (1, 51, 101, 151)}
    lengths = {k: np.linalg.norm((starts[k + 1] if k < 4 else np.zeros(3)) @ b - starts[k] @ b) for k in (1, 2, 3, 4)}
    ticks = np.cumsum([0.0] + [lengths[k] for k in (1, 2, 3, 4)])
    rows = _dat_rows()
    assert rows.shape == (197, 7)                            # 4 x 50 points, the three repeated starts dropped
    _, dat_ticks = F.phonon_path_k([x for x in F.read_blocks(PH_F25, ("BAND",)) if x.kind == "BAND"])
    assert np.allclose(dat_ticks, ticks, atol=2e-6)
    assert np.allclose(dat_ticks, [0, 0.937968, 1.406952, 2.070195, 2.882499], atol=2e-6)
    for path, num, k, freqs in pts:
        x = ticks[path - 1] + np.linalg.norm(k @ b - starts[path] @ b)
        if num in (51, 101, 151):
            # not in the fort.25: the previous path's end (50, 100, 150) is the one row at this distance
            assert x == pytest.approx(ticks[path - 1], abs=1e-9)
            continue
        hit = np.where(np.abs(rows[:, 0] - x) < 2e-6)[0]
        assert hit.size == 1, (num, x)
        assert np.allclose(rows[hit[0], 1:], freqs, atol=0.006), (num, rows[hit[0]], freqs)
    # the last row is Γ (three zero acoustic modes, the triply degenerate optical one), at the path's full length
    assert rows[-1, 0] == pytest.approx(ticks[-1], abs=2e-6)
    assert np.allclose(rows[-1, 1:4], 0, atol=0.05) and np.allclose(rows[-1, 4:], 1380.06, atol=0.01)
    # the first point after each tick is one step (the segment's DK / 49) past it
    for i, n in ((1, 50), (2, 99), (3, 148)):
        assert rows[n, 0] - ticks[i] == pytest.approx(lengths[i + 1] / 49, abs=1e-6)       # the header: 6 digits


def test_gamma_l_sound_velocity_uses_the_true_step():
    """L-Γ (a later block, walked backward from Γ): the singlet acoustic branch's point next to Γ, 31.2650 cm-1 at
    k = √3π/a / 49 (the output's k point 199, (4,4,4)/392), gives 2 π c ν / k = 18.80 km/s; the least-squares slope
    agrees (the old step, DK/48, gave 18.45)."""
    rec = next(r for r in VB.acoustic_velocities(PH_F25) if r["to"] == "(4,4,4)")
    k1 = rec["segment_length_bohr_inv"] / 49 / (BOHR_A * 1e-10)
    hand = 2 * math.pi * 2.99792458e10 * 31.2650 / k1 / 1e3
    assert rec["velocities_km_s"][0] == pytest.approx(hand, rel=5e-3)
    gx = next(r for r in VB.acoustic_velocities(PH_F25) if r["to"] == "(4,0,4)")
    assert gx["velocities_km_s"][0] == pytest.approx(2 * math.pi * 2.99792458e10 * 26.1566
                                                     / (gx["segment_length_bohr_inv"] / 49 / (BOHR_A * 1e-10)) / 1e3, rel=5e-3)


def test_each_later_segment_starts_one_step_in_and_ends_on_its_tick():
    segs = [x for x in F.read_blocks(PH_F25, ("BAND",)) if x.kind == "BAND"]
    pos, ticks = F.phonon_path_k(segs)
    assert pos[0][0] == 0.0 and pos[0][-1] == pytest.approx(ticks[1])
    for i in (1, 2, 3):
        assert pos[i][0] == pytest.approx(ticks[i] + segs[i].dk / 49) and pos[i][-1] == pytest.approx(ticks[i + 1])


def _block_text(ints, cols, dk):
    vals = [v for c in cols for v in c]
    head = f"-%-0BAND{len(cols[0]):5d}{len(cols):5d}" + "".join(f"{v:12.5E}" for v in (0.0, dk, 0.0))
    body = ["".join(f"{v:12.5E}" for v in vals[i:i + 6]) for i in range(0, len(vals), 6)]
    return [head, f"{0.0:12.5E}{2500.0:12.5E}", "".join(f"{v:5d}" for v in ints)] + body


def test_a_later_segment_from_gamma_takes_gamma_from_the_previous_end(tmp_path):
    """X-Γ-X from the real Γ-X block: the second block (Γ-X, 49 points) drops Γ, the first block's last point; its
    acoustic slopes are the first block's, walked backward, and the real file's Γ-X ones."""
    gx = next(x for x in F.read_blocks(PH_F25, ("BAND",)) if x.kind == "BAND")
    cols = [gx.values[j * 6:(j + 1) * 6] for j in range(gx.ncol)]
    f = tmp_path / "x_phonon.f25"
    f.write_text("\n".join(_block_text((4, 0, 4, 0, 0, 0), cols[::-1], gx.dk) + _block_text((0, 0, 0, 4, 0, 4), cols[1:], gx.dk)) + "\n")
    recs = VB.acoustic_velocities(f)
    ref = next(r for r in VB.acoustic_velocities(PH_F25) if r["to"] == "(4,0,4)")["velocities_km_s"]
    assert [r["to"] for r in recs] == ["(4,0,4)", "(4,0,4)"]
    for r in recs:
        assert r["velocities_km_s"] == pytest.approx(ref, rel=1e-6)
