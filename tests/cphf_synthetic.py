"""A CPHF output in CRYSTAL23's format with a chosen dielectric tensor (one or several frequencies)
and first hyperpolarizability, built from the real diamond CPHF run of tests/data/hpcc_116/mechanics
by replacing those two tables (diamond is optically isotropic and centrosymmetric). A uniaxial or
biaxial tensor also gets a crystal family and space group that allow it (tetragonal P-42m, the -42m
of a d14-only SHG tensor, or orthorhombic P222), or the one given (``family``, ``space_group``)."""
from pathlib import Path

MECH = Path(__file__).resolve().parent / "data" / "hpcc_116" / "mechanics"
COMPS = ("XX", "XY", "XZ", "YY", "YZ", "ZZ")
BETA = ("XXX", "XXY", "XXZ", "XYY", "XYZ", "XZZ", "YYY", "YYZ", "YZZ", "ZZZ")


def cphf_output(eps, beta_d=None, freqs=(0.0,), eps_im=None, gamma_tail=True, family=None, space_group=None):
    """``eps``: {comp: value} or {freq: {comp: value}}; ``eps_im`` the same for Im eps; ``beta_d``:
    {comp: d in pm/V} (the d(MKS) column; the other columns scaled from it)."""
    text = (MECH / "dia_cphf.out").read_text()
    i = text.index(" POLARIZABILITY (ALPHA), DIELECTRIC")
    j = text.index(" DIELECTRIC TENSOR", i)
    rows = []
    for f in freqs:
        e = eps.get(f, eps) if isinstance(next(iter(eps.values())), dict) else eps
        im = (eps_im.get(f, eps_im) if eps_im and isinstance(next(iter(eps_im.values())), dict) else eps_im) or {}
        for c in COMPS:
            v, w = e.get(c, 0.0), im.get(c, 0.0)
            rows.append(f"  {f:5.3f}   {c}      {(v - (c[0] == c[1])) * 20:8.4f}       {w * 20:6.4f}       {v:6.4f}       {w:6.4f}       {v - 1 if c[0] == c[1] else v:6.4f}")
    import numpy as np
    first = eps.get(freqs[0], eps) if isinstance(next(iter(eps.values())), dict) else eps
    E = np.array([[first.get("".join(sorted(a + b, key="XYZ".index)), 0.0) for b in "XYZ"] for a in "XYZ"])
    w = np.linalg.eigvalsh(E)
    if family is None and w[2] - w[0] > 1e-6 * abs(w[2]):
        uni = w[1] - w[0] <= 1e-6 * abs(w[2]) or w[2] - w[1] <= 1e-6 * abs(w[2])
        family, space_group = ("TETRAGONAL", "P -4 2 M") if uni else ("ORTHORHOMBIC", "P 2 2 2")
    if family is not None:
        text = text.replace(" CRYSTAL FAMILY                       :  CUBIC       ", f" CRYSTAL FAMILY                       :  {family:12s}", 1)
        text = text.replace(" SPACE GROUP (CENTROSYMMETRIC)        :  F D 3 M", f" SPACE GROUP (NONCENTROSYMMETRIC)     :  {space_group}", 1)
    k = text.index(" TENSOR IN PRINCIPAL AXES SYSTEM", j)
    m = text.index(" REFRACTIVE INDICES", k)
    end = text.index("\n", text.index(" AA", m)) + 1
    text = (text[:k] + " TENSOR IN PRINCIPAL AXES SYSTEM\n AA {:.6E} BB {:.6E} CC {:.6E}\n\n".format(*w)
            + " REFRACTIVE INDICES \n AA {:.6E} BB {:.6E} CC {:.6E}\n".format(*np.sqrt(w)) + text[end:])
    text = (text[:i] + " POLARIZABILITY (ALPHA), DIELECTRIC (EPSILON) AND FIRST-ORDER ELECTRIC\n SUSCEPTIBILITY (CHI(1)) TENSORS "
            "(INDEPENDENT COMPONENTS)\n\n  FR.(eV) COMP.   ALPHA(Re,Im)           EPSILON(Re,Im)                CHI(1)\n"
            + "\n".join(rows) + "\n\n" + text[j:])
    if beta_d is not None:
        a = text.index(" FIRST HYPERPOLARIZABILITY (BETA)")
        b = text.index(" ****", text.index("COMPONENT", a))
        b = text.index("\n", b) + 1
        c = text.index(" ****", b)
        lines = []
        for comp in BETA:
            d = beta_d.get(comp, 0.0)
            au = d * 0.514220632 * 2                  # d(MKS) = d(a.u.)/0.514 pm/V, d(a.u.) = chi(2)/2
            lines.append(f"     {comp}   {au * 12.0:11.4E}  ( {au * 12.0:11.4E})  {au:11.4E}  {d:11.4E}  {d * 0.02387:11.4E}")
        text = text[:b] + "\n" + "\n".join(lines) + "\n" + text[c:]
    return text
