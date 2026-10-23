#!/usr/bin/env python3
"""Build examples/suite: the bundled five-archetype example suite.

    python examples/gui/build_examples.py --suite [--corpus /path/to/test]

Every output file is a copy of a real CRYSTAL23 run: either from the group's
own output corpus (the gitignored ``test/`` folder, pass its location with
--corpus or MACE_TEST_CORPUS) or from files MACE's tests already carry
(tests/data). Nothing is invented. Where no real output exists for an
archetype x calculation-type cell, the input deck is written by MACE itself
(``mace convert``, ``mace opt2d12``, ``mace opt2d3`` on a real output) and
the manifest marks the cell "pending": its output is added when the deck has
been run (--results). A run CRYSTAL did not finish normally is still shipped,
as a "failed" cell whose note quotes the line CRYSTAL stopped with.

Storage: wavefunction (fort.9) and scratch files are never copied; text files
over 32 KiB are stored xz-compressed and unpacked by ``mace examples hydrate``
(mace/examples_suite.py), and cube grids are decimated by 2 along each axis,
by 4 for a grid of more than CUBE_FINE_POINTS points (each cube's manifest note
says which).
"""
from __future__ import annotations

import hashlib
import json
import lzma
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SUITE = REPO / "examples" / "suite"
sys.path.insert(0, str(REPO))

from mace.examples_suite import CALC_TYPES, KIND  # noqa: E402

COMPRESS_OVER = 32 * 1024
# A cube grid with more points than this is decimated by 4, not 2: a chain's RANGE grid
# (polyacetylene: 100 x 518 x 404 points, 275 MB per cube) would otherwise add 14 MB
CUBE_FINE_POINTS = 8_000_000
PY = sys.executable
MACE = str(REPO / "mace_cli")

ARCHETYPES = [
    ("porous", "1-porous", "Porous framework (zeolite/MOF/COF-like)"),
    ("dense-3d", "2-dense-3d", "Dense 3D crystal"),
    ("slab-2d", "3-slab-2d", "2D material / slab"),
    ("polymer-1d", "4-polymer-1d", "1D polymer"),
    ("molecule-0d", "5-molecule-0d", "0D molecule"),
]

# The cells CRYSTAL23 does not run for a system of that dimensionality, with the manual page that says so
# (and, where a probe run on the MSU HPCC stopped, the line CRYSTAL stopped with).
MOLECULE_NA = {
    "BAND": "a molecule has no band structure (CRYSTAL23 manual, BAND: \"does not run for molecules\", p.309)",
    "DOSS": "CRYSTAL computes no density of states for a molecule (CRYSTAL23 manual, DOSS: \"do not run for molecules\", p.320)",
    "TRANSPORT": "BOLTZTRA's transport distribution function sums band velocities over the k points of the "
                 "Brillouin zone (CRYSTAL23 manual, eq. 14.4, p.314), and a molecule has no bands (BAND: \"does not "
                 "run for molecules\", p.309)",
    "PHONON_BAND": "a phonon dispersion samples k points defined by a SCELPHONO supercell of a 1D, 2D or 3D periodic "
                   "system (CRYSTAL23 manual, pp.232-233; SCELPHONO takes a 3D, 2D or 1D expansion matrix, p.73); "
                   "CRYSTAL23 stops a MOLECULE at SCELPHONO (SUPERCELL OPTION NOT ALLOWED FOR MOLECULES)",
    "ELASTIC": "elastic constants are strain derivatives along the periodic directions, and the elements of a "
               "non-periodic direction are null by definition (CRYSTAL23 manual, ch. 13, p.288): a molecule has none; "
               "a probe run of ELASTCON on EC stopped with \"SYSTEM MUST BE PERIODIC FOR ELAPIEZO\"",
    "EOS": "EOS scans the cell volume with volume-constrained optimizations (CRYSTAL23 manual, ch. 12, p.282), "
           "which CRYSTAL23 runs for 3D systems only (p.12: \"Constant volume or pressure constrained geometry "
           "optimizations (3D only)\"); a molecule has no cell volume",
    "QHA": "QHA fits phonon frequencies against the cell volume, relaxing the structure at each volume "
           "(CRYSTAL23 manual, sec. 9.2, p.253); it is generalized to 1D and 2D systems (p.14), not to molecules: a "
           "probe run of QHA on EC stopped with \"ELASTIC OPTION NOT ALLOWED FOR MOLECULES\"",
}
SLAB_NA = {
    "EOS": "EOS scans the cell volume with volume-constrained optimizations (CRYSTAL23 manual, ch. 12, p.282), "
           "which CRYSTAL23 runs for 3D systems only (p.12: \"Constant volume or pressure constrained geometry "
           "optimizations (3D only)\"); probe runs of EOS on h-BN and polyacetylene stopped with \"FOR EOS "
           "CALCULATION THE SYSTEM MUST BE 3D\"",
}
POLYMER_NA = dict(SLAB_NA)
# the entries whose closed-shell (all-zero) SPIN cube is shipped, as an example of the file
KEEP_ZERO_SPIN = ("SOD", "diamond")
# what to say of a finished run, by entry and calculation type (else "at the optimized geometry" or
# "from the SP wavefunction")
ORB_BETA = ("the ALPHA orbitals pass the GUI's orthonormality check; CRYSTAL23 writes the BETA coefficients of this "
            "spin-polarized closed-shell run as {0}, so only ALPHA is drawn (alpha = beta here)")
RAN = {
    "diamond": {
        "SP": "at the optimized geometry (a = 3.5448 A); spin-polarized, closed shell (alpha = beta)",
        "BAND": "SeeK-path path; indirect gap 5.99 eV (B3LYP-D3), alpha = beta",
        "DOSS": "E_F -24.2 to +15 eV, opt2d3's default window with its lower bound moved below the valence band by "
                "the BAND run, at the Standard energy step (1959 points, 0.02 eV apart; E_F from the BAND run, the "
                "SP's spins being locked), projected on C s, p, d",
        "ELASTIC": "C11 1150.4, C12 122.6, C44 618.2 GPa; K_Hill 465.2 GPa",
        "EOS": "Birch-Murnaghan B0 464.5 GPa, V0 11.142 A^3, 7 volumes",
        "QHA": "CRYSTAL's default QHA of the 2-atom cell: Gamma-only phonons at 4 volumes (only the Raman mode "
               "enters, so the thermal expansion is near zero; a supercell, mace elastic --scelphono, gives a "
               "physical one); K_T(300 K) 458.9 GPa",
        "CPHF": "eps_inf 5.387 (exp. 5.7), n 2.321",
        "ORBITALS": ORB_BETA.format("numbers that are not orbitals"),
        "FREQ": "F2g (Raman active, IR inactive) at 1379.9 cm-1 with this method and basis (exp. 1332 cm-1); "
                "IR and Raman spectra; spin-polarized CPKS, about 7 h on 16 cores",
        "PHONON_BAND": "SCELPHONO 2x2x2; X-GAMMA-L-W-GAMMA as integer records; all real (0 to 1388 cm-1, the "
                       "optical branch overbending above its Gamma value of 1380 cm-1)",
        "PHONON_DOS": "SCELPHONO 2x2x2 (8 k points); 0-2000 cm-1, 200 bins, atom-projected; the Gamma optical (F2g, "
                      "Raman) mode at 1380 cm-1 with this method and basis (exp. 1332 cm-1), zone-boundary modes "
                      "583-1318 cm-1, all real",
    },
    "hBN": {
        "OPT": "a = 2.4949 A",
        "SP": "at the optimized geometry; spin-polarized, closed shell (alpha = beta)",
        "BAND": "M-Gamma-K label path; direct gap 5.99 eV at K (HSEsol-3c), alpha = beta",
        "FREQ": "ZO 835 cm-1, E' (LO/TO) 1415 cm-1, IR and Raman spectra; all real",
        "PHONON_BAND": "SCELPHONO 2x2; M-Gamma-K up to 1412 cm-1; acoustic branches -3.5 cm-1 at Gamma (numerical noise)",
        "ELASTIC": "C11 3.827, C12 0.873, C66 1.477 Ha (2D): E 293 N/m, Poisson ratio 0.228",
        "QHA": "the 2D QHA: area S(300 K) 5.419 A^2, K_T 1871 GPa A",
        "CPHF": "polarizability alpha_xx 30.6, alpha_zz 4.7 bohr^3 (no dielectric tensor for a slab)",
        "PHONON_DOS": "SCELPHONO 2x2; 0-2000 cm-1, 200 bins, atom-projected",
        "ORBITALS": ORB_BETA.format("numbers that are not orbitals"),
    },
    "polyacetylene": {
        "SP": "at the optimized geometry; spin-polarized, closed shell (alpha = beta)",
        "BAND": "Gamma-X; direct gap 0.92 eV at X (HSEsol-3c), alpha = beta",
        "TRANSPORT": "from the SP wavefunction; for a chain BOLTZTRA writes the components along the chain axis only "
                     "(S_xx, sigma_xx, kappa_xx)",
        "FREQ": "339.7 to 3150.3 cm-1, all real (four zero modes: three translations and the rotation about the chain); "
                "IR and Raman spectra",
        "PHONON_BAND": "SCELPHONO 2, Gamma-X; the optical branches are real (up to 3150 cm-1); the four zero modes of "
                       "Gamma come out at -58.6 to 0.1 cm-1, as the dispersion does not project out the rotation about "
                       "the chain the Gamma FREQ run does (Eckart), and are positive beyond 9% of Gamma-X",
        "PHONON_DOS": "SCELPHONO 2; 0-4000 cm-1, 200 bins, atom-projected",
        "ELASTIC": "C11 4.804 Ha (along the chain; 1D)",
        "QHA": "the 1D QHA: length L(300 K) 2.466 A, K_T 8462 GPa A^2",
        "CPHF": "polarizability alpha_xx 2264 (along the chain), alpha_yy 17.4, alpha_zz 5.6 bohr^3",
        "ORBITALS": ORB_BETA.format("numbers that are not orbitals"),
    },
    "EC": {
        "SP": "at the optimized geometry; spin-polarized, closed shell (alpha = beta)",
        "FREQ": "158 to 3123 cm-1, all real (C=O stretch 1987 cm-1); IR and Raman spectra",
        "PHONON_DOS": "the Gamma-only phonon DOS of the molecule (manual p.236), 0-4000 cm-1, 200 bins: 3N = 30 states",
        "CPHF": "polarizability tensor, mean alpha 35.5 bohr^3 (no dielectric tensor for a molecule)",
        "ORBITALS": ORB_BETA.format("zeros"),
    },
}


def hbn_cif(a=2.504, box=20.0) -> str:
    """A planar h-BN monolayer in P-6m2, B and N on the mirror plane z = 0 (layer group 78's origin)."""
    lines = ["data_hBN_monolayer", "_chemical_name_common 'hexagonal boron nitride monolayer'",
             f"_cell_length_a {a:.4f}", f"_cell_length_b {a:.4f}", f"_cell_length_c {box:.4f}",
             "_cell_angle_alpha 90.00", "_cell_angle_beta 90.00", "_cell_angle_gamma 120.00",
             "_symmetry_space_group_name_H-M 'P -6 m 2'", "_symmetry_Int_Tables_number 187", "", "loop_",
             "_atom_site_label", "_atom_site_type_symbol", "_atom_site_fract_x", "_atom_site_fract_y",
             "_atom_site_fract_z", "_atom_site_occupancy",
             "B1 B 0.3333333333 0.6666666667 0.0000000000 1.0", "N1 N 0.6666666667 0.3333333333 0.0000000000 1.0"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------- helpers
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def cube_strides(n, s: int):
    """The stride along each axis: the divisor of the axis's point count closest to s (by ratio; a tie takes the
    smaller), so the thinned grid keeps an even step and stays periodic: N/stride points at stride x the step.
    An axis whose nearest divisor is 1 (a prime count) stays whole."""
    import math
    out = []
    for nk in n:
        divs = [d for d in range(1, nk + 1) if nk % d == 0]
        out.append(min(divs, key=lambda d: (abs(math.log(d / s)), d)))
    return out


def decimate_cube(text: str, s: int = 2) -> str:
    """A Gaussian cube thinned along each axis by the divisor of its point count nearest s (cube_strides), so the
    step stays even and the grid periodic; header kept, voxel vectors x stride."""
    lines = text.split("\n")
    nat = int(lines[2].split()[0])
    n = [int(lines[3 + k].split()[0]) for k in range(3)]
    vox = [[float(x) for x in lines[3 + k].split()[1:4]] for k in range(3)]
    head = lines[:6 + abs(nat)]
    vals = " ".join(lines[6 + abs(nat):]).split()
    assert len(vals) == n[0] * n[1] * n[2], "cube value count differs from its header"
    st = cube_strides(n, s)
    keep = []
    for i in range(0, n[0], st[0]):
        for j in range(0, n[1], st[1]):
            base = (i * n[1] + j) * n[2]
            keep.extend(vals[base + k] for k in range(0, n[2], st[2]))
    m = [x // k for x, k in zip(n, st)]
    for k in range(3):
        v = vox[k]
        head[3 + k] = f"{m[k]:5d} {v[0] * st[k]:12.6f} {v[1] * st[k]:12.6f} {v[2] * st[k]:12.6f}"
    if len(set(st)) == 1:
        nth = {2: "2nd", 3: "3rd"}.get(st[0], f"{st[0]}th")
        tail = f" (every {nth} grid point of the original)" if st[0] > 1 else ""
    else:
        tail = f" (grid strides {st[0]}, {st[1]}, {st[2]} of the original)"
    head[0] = head[0].split("./")[0].rstrip() + tail
    body = ["".join(f"{x:>13s}" for x in keep[a:a + 6]) for a in range(0, len(keep), 6)]
    return "\n".join(head + body) + "\n"


def cube_note(text: str) -> str:
    """The manifest note of a thinned cube: its strides."""
    lines = text.split("\n", 6)
    st = cube_strides([int(lines[3 + k].split()[0]) for k in range(3)], cube_step(text))
    if len(set(st)) == 1:
        return f"grid decimated by {st[0]} along each axis" if st[0] > 1 else "full grid"
    return f"grid decimated by {st[0]}, {st[1]}, {st[2]} along its axes (each stride divides that axis's point count)"


def cube_step(text: str) -> int:
    """The decimation step of a cube grid: 2, or 4 above CUBE_FINE_POINTS points."""
    lines = text.split("\n", 6)
    points = 1
    for k in range(3):
        points *= int(lines[3 + k].split()[0])
    return 4 if points > CUBE_FINE_POINTS else 2


def run_mace(args, cwd, answers=""):
    env = dict(os.environ, MPLBACKEND="Agg", NO_COLOR="1", TERM="dumb")
    env.pop("COLORTERM", None)
    r = subprocess.run([PY, MACE, *args], cwd=cwd, input=answers, capture_output=True, text=True,
                       env=env, timeout=300)
    return r


# The calculation types the full-coverage entries run from their SP (the MACE way: OPT -> SP -> these).
# FREQ, PHONON_BAND and PHONON_DOS are opt2d12's frequency templates (4: IR + Raman, 6: phonon band
# structure, 7: phonon density of states), every other prompt answered with Enter (its default, the
# parent's value); ELASTIC, EOS, QHA and CPHF are `opt2d12 --calc-type ... --non-interactive`.
FOLLOW_D12 = ("FREQ", "PHONON_BAND", "PHONON_DOS", "ELASTIC", "EOS", "QHA", "CPHF")
FOLLOW_D3 = ("BAND", "DOSS", "TRANSPORT", "CHARGE+POTENTIAL", "ORBITALS")
FREQ_TEMPLATES = {"FREQ": "4", "PHONON_BAND": "6", "PHONON_DOS": "7"}
# template 7 with 4000 cm-1 for "Maximum frequency for DOS" (after NUMDERIV, the DOS type and the supercell
# question), for a system with C-H stretches near 3100 cm-1 above the 2000 cm-1 default
PDOS_4000 = {"PHONON_DOS": "7\n\n\n\n4000"}
DECK_NAME = {"FREQ": "freq", "PHONON_BAND": "phonon_band", "PHONON_DOS": "phonon_dos", "ELASTIC": "elastic",
             "EOS": "eos", "QHA": "qha", "CPHF": "cphf", "BAND": "band", "DOSS": "doss", "TRANSPORT": "transport",
             "CHARGE+POTENTIAL": "charge+potential", "ORBITALS": "orbitals"}
MADE_BY = {"FREQ": "`mace opt2d12`, FREQ template 4 (IR + Raman spectra)",
           "PHONON_BAND": "`mace opt2d12`, FREQ template 6 (phonon band structure)",
           "PHONON_DOS": "`mace opt2d12`, FREQ template 7 (phonon density of states)",
           **{ct: f"`mace opt2d12 --calc-type {ct}`" for ct in ("ELASTIC", "EOS", "QHA", "CPHF")},
           **{ct: f"`mace opt2d3 -t {ct}`" for ct in FOLLOW_D3}}


def follow_ups(work: Path, sp_out: Path, sp_d12: Path, stem: str, d12_types=(), d3_types=(), tag="sp", templates=None,
               band_dir: Path | None = None):
    """Decks MACE writes from a finished run (<stem>_<tag>.out), renamed <stem>_<calc>.<ext>. Returns {calc: Path}.

    A d12 type in ``templates`` is answered at opt2d12's prompts (FREQ, then that frequency template, Enter for
    the rest) instead of --non-interactive --calc-type. ``band_dir``: the entry's finished BAND run, put beside the
    SP output as it is in a run folder, so a spin-locked SP (no E_F of its own) gets its Fermi level from it for
    the E_F-relative ranges (the DOSS window, TRANSPORT's chemical potentials)."""
    work.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sp_out, work / f"{stem}_{tag}.out")
    shutil.copy2(sp_d12, work / f"{stem}_{tag}.d12")
    if band_dir is not None:
        for name in (f"{stem}_band.BAND.DAT", f"{stem}_band.out"):
            if (band_dir / name).is_file():
                shutil.copy2(band_dir / name, work / name)
    made = {}
    for ct in d12_types:
        od = work / f"d12_{ct}"
        files = ["--out-file", str(work / f"{stem}_{tag}.out"), "--d12-file", str(work / f"{stem}_{tag}.d12")]
        if templates and ct in templates:
            r = run_mace(["opt2d12", *files, "--output-dir", str(od)], work,
                         answers=f"n\n3\n\n{templates[ct]}\n" + "\n" * 120)
        else:
            r = run_mace(["opt2d12", *files, "--calc-type", ct, "--non-interactive", "--output-dir", str(od)], work)
        found = sorted(od.glob("*.d12"))
        if r.returncode or not found:
            raise RuntimeError(f"opt2d12 {ct} failed for {stem}:\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
        made[ct] = found[0]
    for ct in d3_types:
        od = work / f"d3_{ct}"
        src = work / f"{stem}_{tag}.out"
        if ct == "ORBITALS":
            # the deck names its molden files after the output it was written from: from <stem>.out the
            # deck <stem>_orbitals.d3 writes <stem>_orbitals_orb*.molden, the names MACE's job script and
            # the GUI pair with that deck
            (work / "orb").mkdir(exist_ok=True)
            src = work / "orb" / f"{stem}.out"
            shutil.copy2(sp_out, src)
        r = run_mace(["opt2d3", "-i", str(src), "-t", ct, "-o", str(od)], work,
                     answers="y\n" + "\n" * 40)
        found = sorted(od.glob("*.d3"))
        if not found:
            raise RuntimeError(f"opt2d3 {ct} failed for {stem}:\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
        made[ct] = found[0]
    return made


# Graphene is a semimetal: with DIIS and no smearing its SCF oscillated (the tests/data SP stopped at 100 cycles,
# its FREQ child at 200). The suite's SP deck smears the Fermi surface and mixes with Broyden, as the CRYSTAL23
# manual suggests for a metallic slab (BROYDEN: FMIXING 80, BROYDEN 0.0001 50 2); the OPT deck written from it keeps
# them, and the d3 decks (BAND, DOSS, TRANSPORT, CHARGE+POTENTIAL) read its wavefunction.
GRAPHENE_SCF = ("SCFDIR\nMAXCYCLE\n100\nFMIXING\n30\nDIIS\nHISTDIIS\n100\n",
                "SMEAR\n0.001000\nSCFDIR\nMAXCYCLE\n200\nFMIXING\n80\nBROYDEN\n0.0001 50 2\n")
# spin-polarized, as MACE writes a deck by default (the tests/data deck is not)
GRAPHENE_SPIN = ("DFT\nHSESOL3C\n", "DFT\nSPIN\nHSESOL3C\n")


# Why the suite has no graphene FREQ (runs on the MSU HPCC for the 1.1.6 suite, FREQ decks written by opt2d12)
GRAPHENE_FREQ_NA = (
    "no usable FREQ for this semimetal: without Fermi smearing its SCF oscillates (the FREQ run stopped with "
    "SCF ENDED - TOO MANY CYCLES at 200 cycles), and with smearing CRYSTAL23's analytic gradients, from which "
    "FREQCALC builds the Hessian, are not reliable for a conducting system (CRYSTAL23 manual, SMEAR, p.127). "
    "FREQ runs from smeared SPs (Broyden mixing) gave the E2g (G) mode at 235 cm-1 with SMEAR 0.005 and at "
    "-719 cm-1 with SMEAR 0.001, where graphene's is near 1580 cm-1, so no FREQ deck or output is shipped")


def graphene_sp_deck(text: str) -> str:
    for old, new in (GRAPHENE_SCF, GRAPHENE_SPIN):
        assert text.count(old) == 1, "the graphene SP deck's DFT or SCF records changed"
        text = text.replace(old, new)
    return text


def convert_from_cif(W: Path, name: str, calc_types, extra=None):
    """SP/OPT decks `mace convert --batch` writes from the one CIF in W/cif (HSEsol-3c, SPIN). Returns {calc: Path}."""
    decks = {}
    for ct in calc_types:
        cfg = {"version": "1.0", "type": "d12_configuration", "configuration": {
            "name": name, "calculation_type": ct, "optimization_type": "FULLOPTG", "method": "DFT",
            "functional": "HSESOL3C", "is_3c_method": True, "dispersion": False, "basis_set": "SOLDEF2MSVP",
            "basis_set_type": "INTERNAL", "dft_grid": None,
            "tolerances": {"TOLINTEG": "8 8 8 9 24", "TOLDEE": 8},
            "optimization_settings": {"toldeg": 0.0003, "toldex": 0.0012, "toldee": 7, "maxcycle": 800},
            "scf_settings": {"method": "DIIS", "maxcycle": 800, "fmixing": 30}, "spin_polarized": True,
            **(extra or {})}}
        cj = W / f"cfg_{ct}.json"
        cj.write_text(json.dumps(cfg))
        od = W / f"out_{ct}"
        r = run_mace(["convert", "--cif_dir", str(W / "cif"), "--output_dir", str(od), "--batch", "--options_file", str(cj)], W)
        found = sorted(od.glob("*.d12"))
        if not found:
            raise RuntimeError(f"mace convert failed for {name} {ct}:\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
        decks[ct] = found[0]
    return decks


def polyacetylene_cif(a=2.46, double=1.36, single=1.44, ch=1.09, box=10.0) -> str:
    """Planar zigzag trans-(CH)x along x, 4 atoms per repeat, from literature bond lengths (P1, chain mid-box)."""
    import math
    u = (a * a - (single ** 2 - double ** 2)) / (2 * a)       # x of the second carbon: |C1C2| = double, |C2C1'| = single
    h = math.sqrt(double ** 2 - u * u)                         # zigzag amplitude
    c1, c2 = (0.0, h / 2), (u, -h / 2)

    def unit(v):
        n = math.hypot(*v)
        return v[0] / n, v[1] / n

    def hydrogen(c, nbrs):                                     # on the outer bisector of the two C-C bonds
        us = [unit((n[0] - c[0], n[1] - c[1])) for n in nbrs]
        s = unit((-(us[0][0] + us[1][0]), -(us[0][1] + us[1][1])))
        return c[0] + ch * s[0], c[1] + ch * s[1]

    atoms = [("C1", "C", c1), ("C2", "C", c2), ("H1", "H", hydrogen(c1, [c2, (c2[0] - a, c2[1])])),
             ("H2", "H", hydrogen(c2, [c1, (c1[0] + a, c1[1])]))]
    lines = ["data_trans_polyacetylene", "_chemical_name_common 'trans-polyacetylene (CH)x, planar zigzag chain'",
             f"_cell_length_a {a:.4f}", f"_cell_length_b {box:.4f}", f"_cell_length_c {box:.4f}",
             "_cell_angle_alpha 90.00", "_cell_angle_beta 90.00", "_cell_angle_gamma 90.00",
             "_symmetry_space_group_name_H-M 'P 1'", "_symmetry_Int_Tables_number 1", "", "loop_", "_atom_site_label",
             "_atom_site_type_symbol", "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z", "_atom_site_occupancy"]
    lines += [f"{lab} {el} {x / a:.10f} {0.5 + y / box:.10f} 0.5000000000 1.0" for lab, el, (x, y) in atoms]
    return "\n".join(lines) + "\n"


_FAIL_PATTERNS = (
    (re.compile(r"SCF ENDED - TOO MANY CYCLES.*"), "CRYSTAL stopped: {0}"),
    (re.compile(r"^\s*ERROR \*\*\*\*.*", re.M), "CRYSTAL stopped: {0}"),
)


def cube_problem(text: str):
    """None for a usable cube grid, else why it is not (CRYSTAL writes 1 point and an infinite step
    along a non-periodic axis that SCALE gave no extent, e.g. the normal of a planar chain)."""
    lines = text.split("\n", 6)
    for k in range(3):
        f = lines[3 + k].split()
        if int(f[0]) < 2 or any(x.lower() in ("infinity", "inf", "nan") for x in f[1:4]):
            return (f"CRYSTAL ended normally but wrote a degenerate grid ({int(f[0])} point(s) along axis {k + 1}, "
                    f"step {' '.join(f[1:4])}): SCALE on a system with no extent along that axis; RANGE is needed. "
                    f"The cube files are not shipped")
    return None


def cube_is_zero(text: str) -> bool:
    """True for a cube whose every value is zero to 1e-10 (the SPIN cube of a closed-shell run)."""
    lines = text.split("\n")
    nat = abs(int(lines[2].split()[0]))
    return all(abs(float(x)) < 1e-10 for x in " ".join(lines[6 + nat:]).split())


def run_outcome(text: str, calc: str):
    """None for a run that ended normally, else the reason it did not (the CRYSTAL line itself)."""
    for rx, fmt in _FAIL_PATTERNS:
        m = rx.search(text)
        if m:
            return fmt.format(" ".join(m.group(0).split()))
    if "EEEEEEEEEE TERMINATION" not in text:
        return "the run ended before CRYSTAL's normal termination"
    if calc == "OPT" and "OPT END - CONVERGED" not in text:
        return "the geometry optimization did not converge"
    return None


class Builder:
    def __init__(self, corpus: Path, work: Path):
        self.corpus, self.work = corpus, work
        self.files = []          # manifest file records
        self.archetypes = []

    # -- file placement ----------------------------------------------------
    def put(self, rel: str, data: bytes, source: str, note: str = "") -> str:
        dst = SUITE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        rec = {"path": rel, "bytes": len(data), "sha256": sha256_bytes(data), "source": source}
        if note:
            rec["note"] = note
        if len(data) > COMPRESS_OVER:
            blob = lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)
            (SUITE / (rel + ".xz")).write_bytes(blob)
            rec["stored"] = rel + ".xz"
            rec["stored_bytes"] = len(blob)
        else:
            dst.write_bytes(data)
        self.files.append(rec)
        return rel

    def real(self, rel_dir: str, name: str, src: Path, source_label: str, transform=None, note=""):
        data = src.read_bytes()
        if transform:
            data = transform(data.decode("utf-8", "replace")).encode()
        return self.put(f"{rel_dir}/{name}", data, source_label, note)

    def gen(self, rel_dir: str, name: str, src: Path, how: str):
        return self.put(f"{rel_dir}/{name}", src.read_bytes(), how)


def corpus_label(p: Path, corpus: Path) -> str:
    try:
        return "group corpus: " + p.relative_to(corpus).as_posix()
    except ValueError:
        return "tests/" + p.relative_to(REPO / "tests").as_posix()


def build(corpus: Path, results: Path | None = None):
    if not corpus.is_dir():
        sys.exit(f"The output corpus {corpus} is not there; pass --corpus /path/to/test (or set MACE_TEST_CORPUS).")
    if SUITE.exists():
        shutil.rmtree(SUITE)
    SUITE.mkdir(parents=True)
    work = Path(tempfile.mkdtemp(prefix="suite_build_"))
    B = Builder(corpus, work)
    LDG = REPO / "tests" / "data" / "low_dim_groups"
    cp = lambda sub, name: corpus / sub / name  # noqa: E731
    archs = {}

    folder_of = {aid: folder for aid, folder, _ in ARCHETYPES}

    def adopt_results(arch, eid, cells):
        """A deck-only cell whose run is in RESULTS/<archetype folder>/<entry>/<calc>/ becomes a real cell,
        or a failed one (its output is still shipped) when CRYSTAL did not end normally."""
        if results is None:
            return
        for ct, c in cells.items():
            rd = results / folder_of[arch] / eid / ct
            if c["status"] != "pending" or not rd.is_dir() or not list(rd.glob("*.out")):
                continue
            fl = list(c["decks"])
            rel = f"{folder_of[arch]}/{eid}/{ct}"
            reasons = [r for r in (run_outcome(o.read_text(errors="replace"), ct) for o in sorted(rd.glob("*.out"))) if r]
            bad_cubes = {f.name: cube_problem(f.read_text(errors="replace")) for f in sorted(rd.glob("*.CUBE"))}
            reasons += sorted({r for r in bad_cubes.values() if r})
            # a closed-shell run's SPIN cube is zero everywhere: not worth its 0.3-0.8 MB in the repository
            # (the SOD and diamond ones stay, as the suite's SPIN cubes)
            zero_spin = sorted(f.name for f in rd.glob("*_SPIN.CUBE")
                               if not bad_cubes.get(f.name) and (eid not in KEEP_ZERO_SPIN
                                                                 or cube_step(f.read_text(errors="replace")) == 4)
                               and cube_is_zero(f.read_text(errors="replace")))
            # a DOSS run's fort.25 holds its DOSS.DAT's numbers in another format: only the DOSS.DAT is
            # shipped (4LG_AA's DOSS keeps one, the suite's example of the format)
            dup_f25 = ct == "DOSS" and any(rd.glob("*.DOSS.DAT"))
            for f in sorted(rd.iterdir()):
                if not f.is_file() or f.suffix.lower() in (".f9", ".sh", ".o") or f.name.startswith("fort."):
                    continue
                if dup_f25 and f.suffix.lower() == ".f25":
                    continue
                if bad_cubes.get(f.name) or f.name in zero_spin:
                    continue
                if f"{rel}/{f.name}" in fl:
                    # the deck is already there (written by mace here); the run must have used the same one
                    if (SUITE / rel / f.name).read_bytes() != f.read_bytes():
                        raise RuntimeError(f"{f} differs from the deck this build wrote ({rel}/{f.name})")
                    continue
                if f.suffix.lower() in (".d12", ".d3") and c.get("_made_by"):
                    label = c["_made_by"]
                else:
                    label = f"CRYSTAL23 run on the MSU HPCC from the deck(s) above ({f.name})"
                cube = f.suffix.upper() == ".CUBE"
                step = cube_step(f.read_text(errors="replace")) if cube else 0
                fl.append(B.real(rel, f.name, f, label,
                                 transform=(lambda t, s=step: decimate_cube(t, s)) if cube else None,
                                 note=cube_note(f.read_text(errors="replace")) if cube else ""))
            if reasons:
                cells[ct] = {"status": "failed", "files": fl, "note": "; ".join(reasons) + " (run on the MSU HPCC for this suite)"}
            else:
                zero = "the spin density is zero everywhere (a closed-shell result), so its SPIN cube is not shipped" \
                    if zero_spin else ""
                cells[ct] = real_cell(fl, "; ".join(x for x in ("run on the MSU HPCC for this suite", c.get("_ran"), zero)
                                                    if x))

    def entry(arch, eid, title, formula, atoms, dim, method, notes, cells):
        adopt_results(arch, eid, cells)
        for c in cells.values():
            c.pop("_made_by", None)
            c.pop("_ran", None)
        archs.setdefault(arch, []).append({"id": eid, "title": title, "formula": formula, "atoms": atoms,
                                           "dimensionality": dim, "method": method, "notes": notes, "cells": cells})

    def real_cell(files, note=""):
        c = {"status": "real", "files": files}
        if note:
            c["note"] = note
        return c

    def pending_cell(decks, requires, note="", made_by="", ran=""):
        c = {"status": "pending", "decks": decks, "requires": requires}
        if note:
            c["note"] = note
        if made_by:
            c["_made_by"] = made_by        # provenance of a deck that comes with the results; not written out
        if ran:
            c["_ran"] = ran                # what to say about the run once its results are in; not written out
        return c

    def chain(d, stem, opt_out, opt_d12, na, ran=None, only=None, templates=None):
        """The cells after OPT of a full-coverage entry, the MACE way: SP from the OPT output (`mace opt2d12
        --calc-type SP`), then every other type from the SP (its output for the d12 decks, its wavefunction
        for the d3 runs). ``na``: {calc: reason} for what CRYSTAL23 does not run for this system;
        ``ran``: {calc: note} said of a finished run; ``only``: the follow-up types to make (default: all);
        ``templates``: FREQ template answers that differ from FREQ_TEMPLATES."""
        ran = ran or {}
        cells = {}
        W = work / f"chain_{stem}"
        sp_deck = None
        if opt_out is not None and opt_out.is_file():
            fu = follow_ups(W / "sp", opt_out, opt_d12, stem, ("SP",), (), tag="opt")
            sp_deck = fu["SP"]
            cells["SP"] = pending_cell([B.gen(f"{d}/SP", f"{stem}_sp.d12", sp_deck,
                                              "written by `mace opt2d12 --calc-type SP` from the OPT output")], [],
                                       ran=ran.get("SP", "at the optimized geometry"))
        else:
            cells["SP"] = pending_cell([], ["the optimized geometry: run OPT first, then write this deck with "
                                            "`mace opt2d12 --calc-type SP` from its output"],
                                       made_by="written by `mace opt2d12 --calc-type SP` from the OPT output of the run above",
                                       ran=ran.get("SP", "at the optimized geometry"))
        sp_out = results / d / "SP" / f"{stem}_sp.out" if results is not None else None
        d12s = [ct for ct in FOLLOW_D12 if ct not in na and (only is None or ct in only)]
        d3s = [ct for ct in FOLLOW_D3 if ct not in na and (only is None or ct in only)]
        if sp_deck is not None and sp_out is not None and sp_out.is_file() and run_outcome(
                sp_out.read_text(errors="replace"), "SP") is None:
            fu = follow_ups(W / "follow", sp_out, sp_deck, stem, d12s, d3s, tag="sp",
                            templates={**FREQ_TEMPLATES, **(templates or {})}, band_dir=results / d / "BAND")
            for ct in d12s + d3s:
                ext = "d3" if ct in FOLLOW_D3 else "d12"
                answer = " (\"Maximum frequency for DOS\" answered 4000 cm-1)" if (templates or {}).get(ct) == PDOS_4000.get(ct) else ""
                deck = B.gen(f"{d}/{ct}", f"{stem}_{DECK_NAME[ct]}.{ext}", fu[ct],
                             f"written by {MADE_BY[ct]}{answer} from the SP output")
                req = [f"{stem} SP wavefunction (fort.9): run SP/{stem}_sp.d12 first and keep its fort.9"] if ext == "d3" else []
                cells[ct] = pending_cell([deck], req, ran=ran.get(ct, "from the SP wavefunction" if ext == "d3"
                                                                  else "at the optimized geometry"))
        else:
            for ct in d12s + d3s:
                cells[ct] = pending_cell([], [f"the SP run: run SP first, then write this deck with {MADE_BY[ct]} from its output"],
                                         made_by=f"written by {MADE_BY[ct]} from the SP output of the run above",
                                         ran=ran.get(ct, ""))
        for ct, why in na.items():
            cells[ct] = {"status": "n/a", "reason": why}
        return cells

    # =============================================================== 5 / molecule: ethylene carbonate
    E = "EC_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized"
    d = "5-molecule-0d/EC"
    lab = lambda p: corpus_label(p, corpus)  # noqa: E731
    p = cp("OPT", E + ".d12"); q = cp("OPT", E + ".out"); c = cp("OPT", E + ".cif")
    cells = {"OPT": real_cell([B.real(f"{d}/OPT", "EC_opt.d12", p, lab(p)), B.real(f"{d}/OPT", "EC_opt.out", q, lab(q)),
                               B.real(f"{d}/OPT", "EC_opt.cif", c, lab(c))])}
    cells.update(chain(d, "EC", q, p, MOLECULE_NA, RAN.get("EC"), templates=PDOS_4000))
    entry("molecule-0d", "EC", "Ethylene carbonate (isolated molecule)", "C3H4O3", 10, 0,
          "HSEsol-3c / SOLDEF2MSVP, MOLECULE, spin-polarized default",
          "A 10-atom organic carbonate (a Li-ion electrolyte solvent), the suite's full-coverage 0D entry: the group's "
          "OPT run, then SP from it and every other calculation from that SP, all decks written by MACE.", cells)

    # =============================================================== 4 / polymer: trans-polyacetylene
    d = "4-polymer-1d/polyacetylene"
    W = work / "polyacetylene"
    (W / "cif").mkdir(parents=True)
    cif = polyacetylene_cif()
    (W / "cif" / "polyacetylene.cif").write_text(cif)
    decks = convert_from_cif(W, "polyacetylene", ("OPT",), {"dimensionality": "POLYMER", "rod_group": 1})
    cells = {"OPT": pending_cell([B.gen(f"{d}/OPT", "polyacetylene_opt.d12", decks["OPT"],
                                        "written by `mace convert --batch` from polyacetylene.cif (POLYMER, rod group 1)")], [])}
    cells.update(chain(d, "polyacetylene", results / d / "OPT" / "polyacetylene_opt.out" if results is not None else None,
                       decks["OPT"], POLYMER_NA, RAN.get("polyacetylene"), templates=PDOS_4000))
    B.put(f"{d}/structure/polyacetylene.cif", cif.encode(),
          "built here: planar zigzag trans-(CH)x, C=C 1.36 A, C-C 1.44 A, C-H 1.09 A, repeat 2.46 A (literature values)")
    entry("polymer-1d", "polyacetylene", "trans-Polyacetylene (CH)x chain", "C2H2", 4, 1,
          "HSEsol-3c / SOLDEF2MSVP, POLYMER (rod group 1), spin-polarized default",
          "A conjugated polymer with alternating single and double bonds, 4 atoms per repeat, the suite's full-coverage "
          "1D entry. The start geometry uses literature bond lengths; the OPT run relaxes the chain and the repeat "
          "length, SP runs at its geometry and every other calculation starts from that SP.", cells)

    # =============================================================== 3 / slab: h-BN, graphene, 4-layer graphene
    d = "3-slab-2d/hBN"
    W = work / "hbn"
    (W / "cif").mkdir(parents=True)
    cif = hbn_cif()
    (W / "cif" / "hBN.cif").write_text(cif)
    decks = convert_from_cif(W, "hBN", ("OPT",), {"dimensionality": "SLAB", "layer_group": 78})
    cells = {"OPT": pending_cell([B.gen(f"{d}/OPT", "hBN_opt.d12", decks["OPT"],
                                        "written by `mace convert --batch` from hBN.cif (SLAB, layer group 78)")], [],
                                 ran=RAN["hBN"]["OPT"])}
    cells.update(chain(d, "hBN", results / d / "OPT" / "hBN_opt.out" if results is not None else None,
                       decks["OPT"], SLAB_NA, RAN.get("hBN")))
    B.put(f"{d}/structure/hBN.cif", cif.encode(),
          "built here: planar h-BN monolayer, a = 2.504 A (bulk h-BN in-plane lattice constant), B-N 1.446 A, P-6m2")
    entry("slab-2d", "hBN", "Hexagonal boron nitride monolayer", "BN", 2, 2,
          "HSEsol-3c / SOLDEF2MSVP, SLAB (layer group 78, p-6m2), spin-polarized default",
          "A wide-gap insulator, two atoms, the suite's full-coverage 2D entry: OPT from the textbook monolayer, "
          "SP at its geometry and every other calculation from that SP.", cells)

    d = "3-slab-2d/graphene"
    base = LDG / "graphene_lg80"
    W = work / "graphene"
    sp_d12 = work / "graphene_sp_deck" / "graphene_sp.d12"
    sp_d12.parent.mkdir(parents=True)
    sp_d12.write_text(graphene_sp_deck(base.with_suffix(".d12").read_text()))
    # the follow-up decks come from the suite's own SP run once it is in RESULTS (else the tests/data output,
    # same geometry; the SCF records come from the deck)
    sp_out = results / "3-slab-2d" / "graphene" / "SP" / "graphene_sp.out" if results is not None else None
    fu = follow_ups(W, sp_out if sp_out is not None and sp_out.is_file() else base.with_suffix(".out"), sp_d12,
                    "graphene", ("OPT",), ("BAND", "DOSS", "TRANSPORT", "CHARGE+POTENTIAL"),
                    band_dir=results / "3-slab-2d" / "graphene" / "BAND" if results is not None else None)
    how = "written by `mace %s` from the real SP output (graphene SP run, layer group 80)"
    cells = {}
    cells["SP"] = pending_cell([B.gen(f"{d}/SP", "graphene_sp.d12", sp_d12,
                                      corpus_label(base.with_suffix(".d12"), corpus) + " with SPIN (MACE's default), "
                                      "SMEAR 0.001, FMIXING 80, BROYDEN 0.0001 50 2 and MAXCYCLE 200 in place of DIIS "
                                      "(the semimetal's SCF)")], [],
                               ran="graphene a = 2.4612 A as a SLAB in layer group 80, spin-polarized; Fermi smearing "
                                   "0.001 Ha and Broyden mixing, without which graphene's SCF oscillated")
    req_sp = ["graphene SP wavefunction (fort.9): run SP/graphene_sp.d12 first and keep its fort.9"]
    cells["OPT"] = pending_cell([B.gen(f"{d}/OPT", "graphene_opt.d12", fu["OPT"], how % "opt2d12 --calc-type OPT")], [])
    cells["FREQ"] = {"status": "n/a", "reason": GRAPHENE_FREQ_NA}
    for ct, nm in (("BAND", "band"), ("DOSS", "doss"), ("TRANSPORT", "transport"), ("CHARGE+POTENTIAL", "charge+potential")):
        cells[ct] = pending_cell([B.gen(f"{d}/{ct}", f"graphene_{nm}.d3", fu[ct], how % f"opt2d3 -t {ct}")], req_sp,
                                 ran="from the SP wavefunction")
    entry("slab-2d", "graphene", "Graphene monolayer (compact cell)", "C2", 2, 2,
          "HSEsol-3c / SOLDEF2MSVP, SLAB (layer group 80), spin-polarized, SMEAR 0.001 Ha",
          "Two atoms: every run here takes minutes. A semimetal: Fermi smearing and Broyden mixing in the SP deck "
          "(and the OPT deck written from it) are what let its SCF converge; the BAND, DOSS, TRANSPORT and "
          "CHARGE+POTENTIAL decks read the SP wavefunction. The full-coverage 2D entry is h-BN.", cells)

    d = "3-slab-2d/4LG_AA"
    P = "4LG_2x2_AA_opt_HSESOL3C_optimized"
    cells = {}
    p = cp("OPT", P + ".d12"); q = cp("OPT", P + ".out"); c = cp("OPT", P + ".cif")
    cells["OPT"] = real_cell([B.real(f"{d}/OPT", "4LG_AA_opt.d12", p, lab(p)), B.real(f"{d}/OPT", "4LG_AA_opt.out", q, lab(q)),
                              B.real(f"{d}/OPT", "4LG_AA_opt.cif", c, lab(c))],
                             "four AA-stacked graphene layers in a 2x2 cell; 20 optimization points")
    q = cp("DOSS", P + "_doss.out"); dd3 = cp("DOSS", P + "_doss.d3"); dat = cp("DOSS", P + "_doss.DOSS.DAT"); f25 = cp("DOSS", P + "_doss.f25")
    cells["DOSS"] = real_cell([B.real(f"{d}/DOSS", "4LG_AA_doss.d3", dd3, lab(dd3)), B.real(f"{d}/DOSS", "4LG_AA_doss.out", q, lab(q)),
                               B.real(f"{d}/DOSS", "4LG_AA_doss.DOSS.DAT", dat, lab(dat)), B.real(f"{d}/DOSS", "4LG_AA_doss.f25", f25, lab(f25))])
    entry("slab-2d", "4LG_AA", "Four-layer graphene, 2x2 cell", "C96", 96, 2,
          "HSEsol-3c / SOLDEF2MSVP, SLAB", "A larger real slab: optimization trajectory and density of states.", cells)

    # =============================================================== 2 / dense 3D: diamond + carbon net
    d = "2-dense-3d/diamond"
    D1 = "1_dia_opt_rev1"
    p = cp("OPT", D1 + ".d12"); q = cp("OPT", D1 + ".out"); c = corpus / "CIFs" / (D1 + ".cif")
    cells = {"OPT": real_cell([B.real(f"{d}/OPT", "diamond_opt.d12", p, lab(p)), B.real(f"{d}/OPT", "diamond_opt.out", q, lab(q)),
                               B.real(f"{d}/OPT", "diamond_opt.cif", c, lab(c))])}
    cells.update(chain(d, "diamond", q, p, {}, RAN.get("diamond")))
    entry("dense-3d", "diamond", "Diamond (2-atom primitive cell)", "C2", 2, 3,
          "B3LYP-D3 / pob-TZVP-type carbon basis (external, in the deck), CRYSTAL, spin-polarized default",
          "The suite's full-coverage 3D entry: the group's OPT run, then SP from it and every other calculation from "
          "that SP, all decks written by MACE. The decks embed a published pob-TZVP-type carbon basis.", cells)

    d = "2-dense-3d/C1-RCSR-bcu-f"
    T = "C1-RCSR-bcu-f_optimized_TRANSPORT"
    fl = []
    for suf in (".d3", ".out", ".KAPPA.dat", ".SEEBECK.dat", ".SIGMA.dat", ".SIGMAS.dat", ".TDF.dat"):
        s = cp("TRANSPORT", T + suf)
        fl.append(B.real(f"{d}/TRANSPORT", f"bcu-f_transport{suf}", s, lab(s)))
    entry("dense-3d", "C1-RCSR-bcu-f", "Carbon net bcu-f (RCSR), 8 atoms", "C8", 8, 3, "CRYSTAL BOLTZTRA run",
          "A 3D carbon net from the group's RCSR screening; only its transport run is shipped.",
          {"TRANSPORT": real_cell(fl, "BOLTZTRA: Seebeck, conductivity, electronic thermal conductivity, TDF")})

    # =============================================================== 1 / porous: sodalite framework
    d = "1-porous/SOD"
    W = work / "sod"
    (W / "cif").mkdir(parents=True)
    a = 8.6827
    cif = f"""data_SOD
_chemical_name_common 'sodalite framework (IZA code SOD), pure silica'
_cell_length_a {a}
_cell_length_b {a}
_cell_length_c {a}
_cell_angle_alpha 90.00
_cell_angle_beta 90.00
_cell_angle_gamma 90.00
_symmetry_space_group_name_H-M 'I -4 3 m'
_symmetry_Int_Tables_number 217

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Si1 Si 0.2500000000 0.0000000000 0.5000000000 1.0
O1 O 0.1434651800 0.1434651800 0.4504947700 1.0
"""
    (W / "cif" / "SOD.cif").write_text(cif)
    decks = convert_from_cif(W, "sod", ("OPT",))
    cells = {"OPT": pending_cell([B.gen(f"{d}/OPT", "SOD_opt.d12", decks["OPT"], "written by `mace convert --batch` from SOD.cif")], [])}
    opt_out = results / d / "OPT" / "SOD_opt.out" if results is not None else None
    cells.update(chain(d, "SOD", opt_out, decks["OPT"], {}, only=("BAND", "DOSS", "TRANSPORT", "CHARGE+POTENTIAL")))
    cells["FREQ"] = pending_cell([], ["the optimized geometry: run OPT first, then write this deck with "
                                      "`mace opt2d12 --calc-type FREQ` from its output"],
                                 made_by="written by `mace opt2d12 --calc-type FREQ` from the OPT output of the run above",
                                 ran="at the optimized geometry")
    B.put(f"{d}/structure/SOD.cif", cif.encode(), "built here: sodalite topology, idealized Si-O 1.610 A, Si-O-Si 145 degrees")
    entry("porous", "SOD", "Sodalite framework, pure silica (idealized)", "Si12O24", 36, 3,
          "HSEsol-3c / SOLDEF2MSVP, CRYSTAL, I-43m",
          "The corpus holds no porous MOF/COF/zeolite output, so every run here was made on the MSU HPCC for this "
          "suite. The start geometry is the "
          "ideal SOD net (all Si-O 1.610 A, tetrahedral), optimized by the OPT run; cages of about 5.5 A diameter. "
          "SP runs at the OPT geometry and the BAND, DOSS, TRANSPORT and CHARGE+POTENTIAL runs read its wavefunction.",
          cells)

    # =============================================================== manifest, ignore list, README
    man = {
        "kind": KIND, "schema": 1, "title": "MACE example suite",
        "calc_types": list(CALC_TYPES),
        "provenance": ("Output files are unmodified copies of CRYSTAL23 runs by the Mendoza group (the group's own data, "
                       "shared for use with MACE), of files in MACE's tests/data, or of CRYSTAL23 runs made for this suite "
                       "on the MSU HPCC from the decks next to them; each file's origin is in `source`. "
                       "Wavefunction (fort.9) and scratch files are not included. Input decks marked "
                       "\"written by mace\" were generated from those outputs by MACE itself. Cube grids are decimated "
                       "by 2 along each axis (by 4 for the polyacetylene grids, 100 x 518 x 404 points in CRYSTAL's run). The diamond decks embed a published pob-TZVP-type carbon basis set."),
        "archetypes": [{"id": aid, "folder": folder, "title": title, "entries": archs.get(aid, [])} for aid, folder, title in ARCHETYPES],
        "files": sorted(B.files, key=lambda f: f["path"]),
    }
    (SUITE / "MANIFEST.json").write_text(json.dumps(man, indent=1) + "\n")
    ign = ["# unpacked from the .xz copies by `mace examples hydrate` (or by the GUI when it lists this folder)", "*.part"]
    ign += sorted(f["path"] for f in B.files if f.get("stored"))
    (SUITE / ".gitignore").write_text("\n".join(ign) + "\n")
    (SUITE / "README.md").write_text(readme(man))
    shutil.rmtree(work, ignore_errors=True)
    tot = sum(p.stat().st_size for p in SUITE.rglob("*") if p.is_file())
    print(f"example suite written to {SUITE} ({sum(1 for _ in SUITE.rglob('*') if _.is_file())} files, {tot / 1e6:.2f} MB)")


def readme(man) -> str:
    from mace.examples_suite import STATUS_WORDS
    L = ["# MACE example suite", "",
         "```", "mace examples                 # the table below", "mace examples hydrate         # unpack the compressed files",
         "mace gui --dir examples/suite --open", "```", "",
         "Inputs and CRYSTAL23 outputs for five material archetypes across the calculation types, so the GUI's viewer and "
         "plots can be tried without running anything first. Folders: `<archetype>/<system>/<calculation type>/`.", "",
         "Four entries cover every calculation type, one per dimensionality: diamond (3D), h-BN (2D), polyacetylene (1D) "
         "and EC (0D). Each starts from an OPT run; SP runs at the optimized geometry and every other calculation starts "
         "from that SP (its output for the d12 decks, its wavefunction for the properties runs), with the decks MACE "
         "writes (`mace opt2d12`, `mace opt2d3`). n/a marks only what CRYSTAL23 does not run for that dimensionality, "
         "with the manual page below. The other entries are extra examples.", "",
         "| Archetype | System | " + " | ".join(man["calc_types"]) + " |", "|---|---|" + "---|" * len(man["calc_types"])]
    failed = []
    for a in man["archetypes"]:
        for e in a["entries"]:
            L.append(f"| {a['title']} | {e['id']} | " + " | ".join(STATUS_WORDS.get(e['cells'].get(ct, {}).get('status', '-'), '-') for ct in man["calc_types"]) + " |")
            failed += [f"- {e['id']} {ct}: {c['note']}" for ct, c in e["cells"].items() if c["status"] == "failed"]
    L += ["", "output = a real CRYSTAL23 output is included. failed run = the run's real output is included, but CRYSTAL "
          "stopped before finishing (reason below). deck only = the input deck is included and its output is added when it "
          "has been run (`mace examples` lists them; the manifest names what each one needs). n/a = CRYSTAL does not run it "
          "for this system, or gives no usable result (reason below)."]
    if failed:
        L += ["", "## Failed runs", ""] + failed
    na = {}
    for a in man["archetypes"]:
        for e in a["entries"]:
            for ct in man["calc_types"]:
                c = e["cells"].get(ct, {})
                if c.get("status") == "n/a":
                    na.setdefault(c["reason"], []).append(f"{e['id']} {ct}")
    if na:
        L += ["", "## n/a", ""] + [f"- {', '.join(cells)}: {why}" for why, cells in na.items()]
    L += ["", "## Provenance", "", man["provenance"], "",
          "## Size", "", "Text files over 32 KiB are stored `.xz`-compressed and unpacked in place the first time the GUI lists "
          "this folder or `mace examples hydrate` runs; `MANIFEST.json` carries each file's SHA-256. On a read-only install "
          "use `mace examples copy DEST`. A DOSS run's fort.25 repeats its DOSS.DAT in another format and is left out "
          "(4LG_AA's DOSS keeps one).", "",
          "Rebuild with `python examples/gui/build_examples.py --suite --corpus /path/to/test --results /path/to/runs`: "
          "the runs folder holds the finished runs (`<archetype>/<system>/<calculation type>/`), and every deck the build "
          "writes must be the one that ran."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", default=os.environ.get("MACE_TEST_CORPUS") or str(REPO / "test"))
    ap.add_argument("--results", default=None, help="folder of finished runs: <archetype folder>/<entry>/<calc type>/*")
    a = ap.parse_args()
    build(Path(a.corpus), Path(a.results) if a.results else None)
