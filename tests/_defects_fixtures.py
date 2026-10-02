"""Shared helpers for the mace.defects tests (no test_ prefix: not collected).

The decks under tests/defects_data/ are the ones run on HPCC for the defect
facts in round 6 (MgO V_O, hBN V_N; PBE0/POB-TZVP-REV2). Outputs are never
copied in whole: ``fake_output`` writes just the lines the parsers read, with
numbers taken from those runs.
"""
import difflib
from pathlib import Path

DATA = Path(__file__).resolve().parent / "defects_data"


def deck(name: str) -> str:
    return (DATA / f"{name}.d12").read_text()


def strip_records(text: str, keyword: str, n_records: int) -> str:
    """Remove a keyword and the n records that follow it."""
    lines = text.splitlines()
    i = [l.strip() for l in lines].index(keyword)
    return "\n".join(lines[:i] + lines[i + 1 + n_records:]) + "\n"


def mgo_host() -> str:
    """The measured pristine MgO deck without its SUPERCON: a primitive host."""
    return strip_records(deck("mgo_pris"), "SUPERCON", 3)


def hbn_host() -> str:
    return strip_records(deck("hbn3_pris"), "SUPERCEL", 2)


def same_but_title(a: str, b: str) -> bool:
    return a.splitlines()[1:] == b.splitlines()[1:] and a.endswith("\n") == b.endswith("\n")


def edits(host: str, built: str):
    """(inserted lines, deleted lines, replaced pairs) going from host to built, title excluded."""
    h = host.splitlines()[1:]
    b = built.splitlines()[1:]
    ins, dele, rep = [], [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=h, b=b, autojunk=False).get_opcodes():
        if op == "insert":
            ins += b[j1:j2]
        elif op == "delete":
            dele += h[i1:i2]
        elif op == "replace":
            rep.append((h[i1:i2], b[j1:j2]))
    return ins, dele, rep


def _e(x):
    return f"{x: .13E}"


def fake_output(*, energy=None, electrons=None, header=None, efermi=None, vbm=None, cbm=None,
                spin=None, sph=None, lattice=None, atoms=None, converged=True, mulliken_atoms=None,
                spin_atoms=None, vbm_k=1, cycles=14, d3=None, gcp=None):
    """A minimal CRYSTAL23-like output holding only the lines mace.defects reads."""
    out = [" CRYSTAL23 (synthetic excerpt for tests)"]
    if lattice is not None and atoms:
        out += [" ATOMS IN THE ASYMMETRIC UNIT    1 - ATOMS IN THE UNIT CELL:   %d" % len(atoms),
                "     ATOM                 X/A                 Y/B                 Z/C    ",
                " *******************************************************************************"]
        for i, (z, sym, x, y, zz) in enumerate(atoms, 1):
            out.append(f"  {i:5d} {'T' if i == 1 else 'F'} {z:3d} {sym:<2s}   0.0 0.0 0.0")
        out += ["", " DIRECT LATTICE VECTORS CARTESIAN COMPONENTS (ANGSTROM)",
                "          X                    Y                    Z"]
        out += ["   " + "   ".join(f"{v: .12E}" for v in row) for row in lattice]
        out += ["", "", " CARTESIAN COORDINATES - PRIMITIVE CELL",
                " *******************************************************************************",
                " *      ATOM          X(ANGSTROM)         Y(ANGSTROM)         Z(ANGSTROM)",
                " *******************************************************************************"]
        for i, (z, sym, x, y, zz) in enumerate(atoms, 1):
            out.append(f"  {i:3d}    {z:2d} {sym:<2s}   {x: .12E} {y: .12E} {zz: .12E}")
        out.append("")
    if header is not None:
        out.append(f" N. OF ELECTRONS PER CELL   {header}  EXCHANGE PSEUDO OVP (F(G))  (T4) 10**   -7")
    if efermi is not None:
        out += [" TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT FDIK        TELAPSE        1.00 TCPU        1.00",
                f" POSSIBLY CONDUCTING STATE - EFERMI(AU) {efermi:.7E} (RES. CHARGE  7.12E-10;IT.  6)",
                " TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT PDIG        TELAPSE        1.00 TCPU        1.00"]
    if vbm is not None:
        out += [" TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT FDIK        TELAPSE        1.00 TCPU        1.00",
                " INSULATING STATE",
                f" TOP OF VALENCE BANDS -    BAND    320; K {vbm_k:4d}; EIG {vbm:.7E} AU",
                f" BOTTOM OF VIRTUAL BANDS - BAND    321; K    1; EIG {cbm:.7E} AU",
                " DIRECT ENERGY BAND GAP:   7.2486 eV",
                " TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT PDIG        TELAPSE        1.00 TCPU        1.00"]
    if sph is not None:
        out.append(f" ::: EXT EL-SPHEROPOLE                            {_e(sph)}")
    if energy is not None:
        if converged:
            out.append(f" == SCF ENDED - CONVERGENCE ON ENERGY      E(AU) {_e(energy)} CYCLES {cycles:3d}")
        else:
            out.append(f" == SCF ENDED - TOO MANY CYCLES            E(AU) {_e(energy)} CYCLES 150")
    if energy is not None and (d3 is not None or gcp is not None):
        # CRYSTAL23 prints these after the SCF (SP; format as in -D3 and 3c outputs)
        total = energy + (d3 or 0.0) + (gcp or 0.0)
        if gcp is not None:
            out += ["", " DFT-D3-GCP DISPERSION AND COUNTERPOISE CORRECTION", "",
                    f" D3 DISPERSION ENERGY (AU)      {_e(d3 or 0.0)}",
                    f" GCP ENERGY (AU)                {_e(gcp)}",
                    f" TOTAL ENERGY + DISP + GCP (AU) {_e(total)}"]
        else:
            out += ["", f" D3 DISPERSION ENERGY (AU)       {_e(d3)}",
                    f" TOTAL ENERGY + DISP (AU)        {_e(total)}"]
    if electrons is not None:
        out += [" MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM",
                "     ALPHA+BETA ELECTRONS",
                f" MULLIKEN POPULATION ANALYSIS - NO. OF ELECTRONS  {electrons:.6f}", "",
                "  ATOM    Z CHARGE  A.O. POPULATION", ""]
        for lab, sym, z, pop in (mulliken_atoms or [(1, "O", 8, 9.293)]):
            out.append(f"  {lab:3d} {sym:<2s} {z:4d} {pop:6.3f}  0.792  1.181  0.682  1.163  0.620  0.620  0.631  0.599")
            out.append("                    0.599  0.607  0.591  0.591  0.608  0.002  0.001  0.001")
        out += ["", " OVERLAP POPULATION CONDENSED TO ATOMS FOR FIRST NEIGHBORS (ELECTRONS)"]
    if spin is not None:
        out += ["     ALPHA-BETA ELECTRONS",
                f" MULLIKEN POPULATION ANALYSIS - NO. OF ELECTRONS  {spin:10.6f}", "",
                "  ATOM    Z CHARGE  A.O. POPULATION", ""]
        for lab, sym, z, pop in (spin_atoms or [(1, "O", 8, 0.0)]):
            out.append(f"  {lab:3d} {sym:<2s} {z:4d} {pop:6.3f}  0.000 -0.000  0.001")
        out += ["", " OVERLAP POPULATION CONDENSED TO ATOMS FOR FIRST NEIGHBORS (ELECTRONS)"]
    return "\n".join(out) + "\n"
