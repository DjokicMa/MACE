"""UFF and UFF4MOF: the Universal Force Field and its MOF extension, native.

No suitable licensed package covers periodic UFF4MOF as an ase calculator
(RDKit's UFF is molecules only; Open Babel is GPL; lammps-interface writes
LAMMPS input), so MACE carries its own: numpy only, energies, forces and the
stress tensor for molecules, slabs and triclinic crystals.

The functional forms are the published UFF ones (Rappé et al., JACS 114,
10024, 1992) in the conventions RDKit uses, which the tests check term by
term:

* bonds: harmonic, E = k/2 (r - r0)^2, r0 = ri + rj + rBO - rEN,
  k = 664.12 Zi Zj / r0^3;
* angles: a cosine Fourier series in theta; at linear (n = 1), trigonal
  (n = 3), square-planar and octahedral (n = 4) centers E = K/n^2
  (1 - cos n theta) (linear: K (1 + cos theta)), otherwise
  E = K (C0 + C1 cos theta + C2 cos 2 theta) with the C's from theta0; plus
  RDKit's small-angle penalty below 30 degrees;
* torsions about sp2/sp3 main-group bonds: E = V/2 (1 - cos(n phi0) cos(n phi)),
  V divided by the number of torsions about the bond (none about metals, as
  UFF4MOF does);
* inversions at sp2 C/N/O (K = 6, 50 for C=O) and at P/As/Sb/Bi;
* van der Waals: Lennard-Jones 12-6 with x_ij = sqrt(xi xj) and
  D_ij = sqrt(Di Dj), 1-2 and 1-3 pairs excluded, switched smoothly to zero
  between ``cutoff`` - 1.5 Å and ``cutoff`` (12.5 Å) so energies, forces and
  stresses stay continuous (RDKit uses no cutoff: molecules smaller than
  11 Å get exactly RDKit's energy);
* optional fixed point charges (``charges="eqeq"``: EQeq charges of the
  starting geometry), summed with the damped shifted-force Coulomb kernel
  (alpha 0.2 1/Å, same cutoff), 1-2 and 1-3 pairs excluded. UFF itself has
  no charges, so the default is none.

Energies are kcal/mol inside and eV outside (ase units).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from . import base, uff_params as P, uff_typing as T

KCAL = None        # eV per kcal/mol, set on first use (ase.units)
G = 332.06         # UFF's constant: k = 2 G Zi Zj / r^3  (664.12 = 2 x 332.06)
COULOMB = 332.0637  # kcal/mol Å / e^2
DSF_ALPHA = 0.2
SWITCH_WIDTH = 1.5   # Å: the van der Waals energy is switched off smoothly over the last 1.5 Å before the cutoff
ANGLE_CORRECTION = 0.8660
CITATIONS = [
    {"text": "A. K. Rappé, C. J. Casewit, K. S. Colwell, W. A. Goddard III and W. M. Skiff, UFF, a full periodic "
             "table force field for molecular mechanics and molecular dynamics simulations, J. Am. Chem. Soc. 114, "
             "10024 (1992)", "doi": "10.1021/ja00051a040"},
]
UFF4MOF_CITATIONS = [
    {"text": "M. A. Addicoat, N. Vankova, I. F. Akter and T. Heine, Extension of the Universal Force Field to "
             "metal-organic frameworks, J. Chem. Theory Comput. 10, 880 (2014)", "doi": "10.1021/ct400952t"},
    {"text": "D. E. Coupry, M. A. Addicoat and T. Heine, Extension of the Universal Force Field for metal-organic "
             "frameworks, J. Chem. Theory Comput. 12, 5215 (2016)", "doi": "10.1021/acs.jctc.6b00664"},
]


def _kcal() -> float:
    global KCAL
    if KCAL is None:
        from ase import units
        KCAL = units.kcal / units.mol
    return KCAL


def rest_length(ti: str, tj: str, order: float, tab) -> float:
    pi, pj = tab[ti], tab[tj]
    ri, rj, xi, xj = pi[0], pj[0], pi[8], pj[8]
    rbo = -0.1332 * (ri + rj) * np.log(order)
    ren = ri * rj * (np.sqrt(xi) - np.sqrt(xj)) ** 2 / (xi * ri + xj * rj)
    return float(ri + rj + rbo - ren)


def angle_order(t: str) -> int:
    """The Fourier order of an angle centered on type t: 1, 3, 4, 2 (a
    tetrahedral-type center with theta0 = 90) or 0 (the general form)."""
    d = t[2] if len(t) > 2 else ""
    if d in ("", "1"):
        return 1
    if d in ("2", "R"):
        return 3
    if d in ("4", "6"):
        return 4
    return 0


def _shifted(pos, cell, idx, shifts):
    return pos[idx] + shifts @ cell


def _unit(v):
    n = np.linalg.norm(v, axis=1)
    return v / np.maximum(n, 1e-12)[:, None], n


class Terms:
    """The topology's terms as arrays, with their parameters."""

    def __init__(self, topo: T.Topology, numbers):
        tab = T.table(topo.mode)
        types = topo.types
        self.n = len(types)
        self.mode = topo.mode
        b = topo.bonds
        self.b_i = np.array([x[0] for x in b], int)
        self.b_j = np.array([x[1] for x in b], int)
        self.b_s = np.array([x[2] for x in b], float).reshape(-1, 3)
        r0, kb, bo = [], [], {}
        for i, j, s, o in b:
            r = rest_length(types[i], types[j], o, tab)
            r0.append(r)
            kb.append(2.0 * G * tab[types[i]][5] * tab[types[j]][5] / r ** 3)
            bo[(i, j, s)] = o
            bo[(j, i, (-s[0], -s[1], -s[2]))] = o
        self.b_r0, self.b_k = np.array(r0), np.array(kb)

        a = topo.angles
        self.a_a = np.array([x[0] for x in a], int)
        self.a_b = np.array([x[1] for x in a], int)
        self.a_c = np.array([x[2] for x in a], int)
        self.a_sa = np.array([x[3] for x in a], float).reshape(-1, 3)
        self.a_sc = np.array([x[4] for x in a], float).reshape(-1, 3)
        K, C0, C1, C2, order, th0 = [], [], [], [], [], []
        for (ia, ib, ic, sa, sc) in a:
            tb = types[ib]
            t0 = np.radians(tab[tb][1])
            r12 = rest_length(types[ia], tb, bo[(ib, ia, sa)], tab)
            r23 = rest_length(tb, types[ic], bo[(ib, ic, sc)], tab)
            ct = np.cos(t0)
            r13 = np.sqrt(r12 * r12 + r23 * r23 - 2.0 * r12 * r23 * ct)
            beta = 2.0 * G / (r12 * r23)
            k = beta * tab[types[ia]][5] * tab[types[ic]][5] / r13 ** 5 * r12 * r23 * \
                (3.0 * r12 * r23 * (1.0 - ct * ct) - r13 * r13 * ct)
            o = angle_order(tb)
            if o == 0 and abs(tab[tb][1] - 90.0) < 1e-6:
                o = 2
            st2 = max(np.sin(t0) ** 2, 1e-8)
            c2 = 1.0 / (4.0 * st2)
            K.append(k)
            C2.append(c2)
            C1.append(-4.0 * c2 * ct)
            C0.append(c2 * (2.0 * ct * ct + 1.0))
            order.append(o)
            th0.append(t0)
        self.a_K, self.a_C0, self.a_C1, self.a_C2 = map(np.array, (K, C0, C1, C2))
        self.a_order, self.a_th0 = np.array(order, int), np.array(th0)

        t = topo.torsions
        self.t_i = np.array([x[0] for x in t], int)
        self.t_j = np.array([x[1] for x in t], int)
        self.t_k = np.array([x[2] for x in t], int)
        self.t_l = np.array([x[3] for x in t], int)
        self.t_si = np.array([x[4] for x in t], float).reshape(-1, 3)
        self.t_sk = np.array([x[5] for x in t], float).reshape(-1, 3)
        self.t_sl = np.array([x[6] for x in t], float).reshape(-1, 3)
        self.t_V = np.array([x[7] for x in t], float)
        self.t_n = np.array([x[8] for x in t], int)
        self.t_c = np.array([x[9] for x in t], float)

        v = topo.inversions
        self.v_i = np.array([x[0] for x in v], int)
        self.v_j = np.array([x[1] for x in v], int)
        self.v_k = np.array([x[2] for x in v], int)
        self.v_l = np.array([x[3] for x in v], int)
        self.v_si = np.array([x[4] for x in v], float).reshape(-1, 3)
        self.v_sk = np.array([x[5] for x in v], float).reshape(-1, 3)
        self.v_sl = np.array([x[6] for x in v], float).reshape(-1, 3)
        self.v_K = np.array([x[7] for x in v], float)
        self.v_C0 = np.array([x[8] for x in v], float)
        self.v_C1 = np.array([x[9] for x in v], float)
        self.v_C2 = np.array([x[10] for x in v], float)

        self.x = np.array([tab[ty][2] for ty in types])
        self.D = np.array([tab[ty][3] for ty in types])
        self.excl = _keys(np.array([e[0] for e in topo.excluded], int), np.array([e[1] for e in topo.excluded], int),
                          np.array([e[2] for e in topo.excluded], int).reshape(-1, 3), self.n)


_OFF, _BASE = 64, 129


def _keys(i, j, s, n):
    s = np.asarray(s, np.int64) + _OFF
    return ((((i.astype(np.int64) * n + j) * _BASE + s[:, 0]) * _BASE + s[:, 1]) * _BASE + s[:, 2])


def energy_forces(terms: Terms, atoms, cutoff: float = 12.5, charges=None, want_stress: bool = True,
                  parts: Optional[Dict[str, float]] = None):
    """(E kcal/mol, gradient (N,3) kcal/mol/Å, virial sum (3,3) of g x P)."""
    pos = atoms.positions
    cell = atoms.cell.array
    n = len(atoms)
    grad = np.zeros((n, 3))
    W = np.zeros((3, 3))
    E = {}

    def acc(idx, g, Pp):
        np.add.at(grad, idx, g)
        if want_stress:
            W[:] += np.einsum("ka,kb->ab", g, Pp)

    # bonds
    if len(terms.b_i):
        Pi = pos[terms.b_i]
        Pj = _shifted(pos, cell, terms.b_j, terms.b_s)
        d = Pj - Pi
        r = np.linalg.norm(d, axis=1)
        dr = r - terms.b_r0
        E["bond"] = float(0.5 * (terms.b_k * dr * dr).sum())
        g = (terms.b_k * dr / np.maximum(r, 1e-12))[:, None] * d
        acc(terms.b_j, g, Pj)
        acc(terms.b_i, -g, Pi)

    # angles
    if len(terms.a_a):
        Pb = pos[terms.a_b]
        Pa = _shifted(pos, cell, terms.a_a, terms.a_sa)
        Pc = _shifted(pos, cell, terms.a_c, terms.a_sc)
        u, la = _unit(Pa - Pb)
        w, lc = _unit(Pc - Pb)
        c = np.clip((u * w).sum(1), -1.0, 1.0)
        K, o = terms.a_K, terms.a_order
        e = np.zeros_like(c)
        de = np.zeros_like(c)
        m = o == 0
        e[m] = K[m] * (terms.a_C0[m] + terms.a_C1[m] * c[m] + terms.a_C2[m] * (2 * c[m] ** 2 - 1))
        de[m] = K[m] * (terms.a_C1[m] + 4 * terms.a_C2[m] * c[m])
        m = o == 1
        e[m] = K[m] * (1 + c[m])
        de[m] = K[m]
        m = o == 2
        e[m] = K[m] / 4 * (1 + (2 * c[m] ** 2 - 1))
        de[m] = K[m] * c[m]
        m = o == 3
        e[m] = K[m] / 9 * (1 - (4 * c[m] ** 3 - 3 * c[m]))
        de[m] = -K[m] / 9 * (12 * c[m] ** 2 - 3)
        m = o == 4
        e[m] = K[m] / 16 * (1 - (8 * c[m] ** 4 - 8 * c[m] ** 2 + 1))
        de[m] = -K[m] / 16 * (32 * c[m] ** 3 - 16 * c[m])
        m = (o > 0) & (c > ANGLE_CORRECTION)
        if m.any():
            th = np.arccos(c[m])
            ex = np.exp(-20.0 * (th - terms.a_th0[m] + 0.25))
            e[m] += ex
            de[m] += -20.0 * ex * (-1.0 / np.maximum(np.sin(th), 1e-8))
        E["angle"] = float(e.sum())
        ga = de[:, None] * (w - c[:, None] * u) / la[:, None]
        gc = de[:, None] * (u - c[:, None] * w) / lc[:, None]
        acc(terms.a_a, ga, Pa)
        acc(terms.a_c, gc, Pc)
        acc(terms.a_b, -(ga + gc), Pb)

    # torsions
    if len(terms.t_i):
        Pj = pos[terms.t_j]
        Pi = _shifted(pos, cell, terms.t_i, terms.t_si)
        Pk = _shifted(pos, cell, terms.t_k, terms.t_sk)
        Pl = _shifted(pos, cell, terms.t_l, terms.t_sl)
        b1, b2, b3 = Pj - Pi, Pk - Pj, Pl - Pk
        uu, vv = np.cross(b1, b2), np.cross(b2, b3)
        nu, nv = np.linalg.norm(uu, axis=1), np.linalg.norm(vv, axis=1)
        ok = (nu > 1e-8) & (nv > 1e-8)
        c = np.zeros(len(nu))
        c[ok] = np.clip((uu[ok] * vv[ok]).sum(1) / (nu[ok] * nv[ok]), -1.0, 1.0)
        n_ = terms.t_n
        cn = np.where(n_ == 2, 2 * c ** 2 - 1, np.where(n_ == 3, 4 * c ** 3 - 3 * c,
                                                         32 * c ** 6 - 48 * c ** 4 + 18 * c ** 2 - 1))
        dcn = np.where(n_ == 2, 4 * c, np.where(n_ == 3, 12 * c ** 2 - 3, 192 * c ** 5 - 192 * c ** 3 + 36 * c))
        E["torsion"] = float((terms.t_V / 2 * (1 - terms.t_c * cn)).sum())
        de = np.where(ok, -terms.t_V / 2 * terms.t_c * dcn, 0.0)
        nu_, nv_ = np.maximum(nu, 1e-12)[:, None], np.maximum(nv, 1e-12)[:, None]
        gu = de[:, None] * (vv / nv_ - c[:, None] * uu / nu_) / nu_
        gv = de[:, None] * (uu / nu_ - c[:, None] * vv / nv_) / nv_
        gb1 = np.cross(b2, gu)
        gb2 = np.cross(gu, b1) + np.cross(b3, gv)
        gb3 = np.cross(gv, b2)
        acc(terms.t_i, -gb1, Pi)
        acc(terms.t_j, gb1 - gb2, Pj)
        acc(terms.t_k, gb2 - gb3, Pk)
        acc(terms.t_l, gb3, Pl)

    # inversions
    if len(terms.v_i):
        Pj = pos[terms.v_j]
        Pi = _shifted(pos, cell, terms.v_i, terms.v_si)
        Pk = _shifted(pos, cell, terms.v_k, terms.v_sk)
        Pl = _shifted(pos, cell, terms.v_l, terms.v_sl)
        uI, lI = _unit(Pi - Pj)
        uK, lK = _unit(Pk - Pj)
        uL, lL = _unit(Pl - Pj)
        p = np.cross(uK, uI)
        npn = np.maximum(np.linalg.norm(p, axis=1), 1e-12)
        nn = p / npn[:, None]
        s = np.clip((nn * uL).sum(1), -1.0, 1.0)
        siny = np.maximum(np.sqrt(np.maximum(1 - s * s, 0.0)), 1e-8)
        K = terms.v_K
        E["inversion"] = float((K * (terms.v_C0 + terms.v_C1 * siny + terms.v_C2 * (1 - 2 * s * s))).sum())
        dEds = K * (terms.v_C1 * (-s / siny) - 4.0 * terms.v_C2 * s)
        gp = dEds[:, None] * (uL - s[:, None] * nn) / npn[:, None]
        huK = np.cross(uI, gp)
        huI = np.cross(gp, uK)
        huL = dEds[:, None] * nn

        def unit_grad(uv, h, ln):
            return (h - (uv * h).sum(1)[:, None] * uv) / ln[:, None]
        gI, gK, gL = unit_grad(uI, huI, lI), unit_grad(uK, huK, lK), unit_grad(uL, huL, lL)
        acc(terms.v_i, gI, Pi)
        acc(terms.v_k, gK, Pk)
        acc(terms.v_l, gL, Pl)
        acc(terms.v_j, -(gI + gK + gL), Pj)

    # van der Waals (+ fixed charges)
    from ase.neighborlist import neighbor_list
    if n > 1 or any(atoms.pbc):
        ii, jj, SS, DD = neighbor_list("ijSD", atoms, float(cutoff), self_interaction=False)   # a scalar: the pair cutoff
        half = (ii < jj) | ((ii == jj) & ((SS[:, 0] > 0) | ((SS[:, 0] == 0) & ((SS[:, 1] > 0) | ((SS[:, 1] == 0) & (SS[:, 2] > 0))))))
        ii, jj, SS, DD = ii[half], jj[half], SS[half], DD[half]
        if len(ii) and len(terms.excl):
            keep = ~np.isin(_keys(ii, jj, SS, n), terms.excl)
            ii, jj, SS, DD = ii[keep], jj[keep], SS[keep], DD[keep]
        r = np.linalg.norm(DD, axis=1)
        x = np.sqrt(terms.x[ii] * terms.x[jj])
        Dd = np.sqrt(terms.D[ii] * terms.D[jj])
        q6 = (x / r) ** 6
        elj = Dd * (q6 * q6 - 2.0 * q6)
        dEdr = Dd * (12.0 * q6 - 12.0 * q6 * q6) / r
        ron = max(0.0, cutoff - SWITCH_WIDTH)
        sw = r > ron
        if sw.any():                    # CHARMM switch: the energy and the forces go smoothly to 0 at the cutoff
            rc2, ro2, r2 = cutoff * cutoff, ron * ron, r[sw] ** 2
            a, b, den = rc2 - r2, rc2 + 2 * r2 - 3 * ro2, (rc2 - ro2) ** 3
            S = a * a * b / den
            dS = 4 * r[sw] * a * (a - b) / den
            dEdr[sw] = S * dEdr[sw] + elj[sw] * dS
            elj[sw] = S * elj[sw]
        E["vdw"] = float(elj.sum())
        if charges is not None:
            from scipy.special import erfc
            q = np.asarray(charges, float)
            qq = COULOMB * q[ii] * q[jj]
            a = DSF_ALPHA
            ec = erfc(a * cutoff) / cutoff
            fc = ec / cutoff + 2 * a / np.sqrt(np.pi) * np.exp(-a * a * cutoff * cutoff) / cutoff
            E["coulomb"] = float((qq * (erfc(a * r) / r - ec + fc * (r - cutoff))).sum()
                                 - COULOMB * (ec / 2 + a / np.sqrt(np.pi)) * (q * q).sum())
            dEdr = dEdr + qq * (-erfc(a * r) / r ** 2 - 2 * a / np.sqrt(np.pi) * np.exp(-a * a * r * r) / r + fc)
        g = (dEdr / r)[:, None] * DD
        np.add.at(grad, jj, g)
        np.add.at(grad, ii, -g)
        if want_stress:
            W += np.einsum("ka,kb->ab", g, DD)
    if parts is not None:
        parts.update(E)
    return float(sum(E.values())), grad, W


def _calc_base():
    try:
        from ase.calculators.calculator import Calculator, all_changes
    except ImportError:
        return object, ()
    return Calculator, all_changes


_Calculator, _ALL = _calc_base()


class UFFCalculator(_Calculator):
    """UFF / UFF4MOF as an ase calculator (energy, forces, stress, energy parts).

    The topology (types, bond orders, terms) is built from the first
    structure it sees and kept: bonds never form or break. ``topology``
    holds it, ``parts`` the last energy split (eV)."""
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, mode: str = "uff4mof", cutoff: float = 12.5, charges=None, total_charge: float = 0.0, **kw):
        super().__init__(**kw)
        self.mode = mode
        self.cutoff = float(cutoff)
        self.charge_model = charges            # None, "eqeq" or a sequence of charges
        self.total_charge = float(total_charge)  # what the EQeq charges sum to
        self.topology: Optional[T.Topology] = None
        self.terms: Optional[Terms] = None
        self.q = None
        self.parts: Dict[str, float] = {}

    def setup(self, atoms) -> T.Topology:
        self.topology = T.build(atoms, self.mode)
        self.terms = Terms(self.topology, atoms.numbers)
        if isinstance(self.charge_model, str):
            if self.charge_model == "eqeq":
                from . import eqeq
                self.q = eqeq.charges(atoms, total_charge=self.total_charge)["charges"]
            elif self.charge_model != "none":
                raise ValueError(f"charges: none, eqeq or a list, not {self.charge_model!r}")
        elif self.charge_model is not None:
            self.q = np.asarray(self.charge_model, float)
            if self.q.shape != (len(atoms),):
                raise ValueError("one charge per atom")
        return self.topology

    def calculate(self, atoms=None, properties=("energy",), system_changes=_ALL):
        super().calculate(atoms, properties, system_changes)
        a = self.atoms
        if self.terms is None or self.terms.n != len(a):
            self.setup(a)
        periodic = bool(a.pbc.any()) and a.cell.rank == 3
        parts: Dict[str, float] = {}
        e, g, W = energy_forces(self.terms, a, self.cutoff, self.q, want_stress=periodic, parts=parts)
        k = _kcal()
        self.parts = {key: v * k for key, v in parts.items()}
        self.results["energy"] = self.results["free_energy"] = e * k
        self.results["forces"] = -g * k
        if periodic:
            s = 0.5 * (W + W.T) / a.get_volume() * k
            self.results["stress"] = np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])
        elif "stress" in properties:
            from ase.calculators.calculator import PropertyNotImplementedError
            raise PropertyNotImplementedError("a molecule has no stress")

    def close(self) -> None:
        pass


class UFFBackend(base.Backend):
    name = "uff"
    label = "UFF / UFF4MOF"
    family = "forcefield"
    license = "built into MACE (parameters: Rappé 1992; Addicoat 2014, Coupry 2016)"
    capabilities = base.Capabilities(frozenset({"sp", "opt", "freq", "md"}), frozenset({0, 1, 2, 3}),
                                     frozenset({"energy", "forces", "stress"}))
    citation = CITATIONS

    def methods(self) -> List[base.Method]:
        return [base.Method("uff4mof", "UFF4MOF (UFF plus the MOF metal and oxygen types)", default=True,
                            license=self.license, citation=CITATIONS + UFF4MOF_CITATIONS,
                            note="Reduces to plain UFF wherever no MOF type applies"),
                base.Method("uff", "UFF (Rappé 1992, the 127 original types)", license=self.license,
                            citation=list(CITATIONS))]

    def calculator(self, atoms, spec: base.CalcSpec, canceled: Optional[Callable[[], bool]] = None):
        o = spec.options or {}
        calc = UFFCalculator(mode=self.method(spec.method).id, cutoff=float(o.get("cutoff", 12.5)),
                             charges=o.get("charges"), total_charge=float(spec.charge))
        try:
            calc.setup(atoms)
        except ValueError as exc:
            raise base.CalcError(f"UFF typing: {exc}", kind="typing")
        return calc

    @staticmethod
    def report(calc: UFFCalculator) -> Dict[str, Any]:
        t = calc.topology
        if t is None:
            return {}
        return {"atom_types": t.type_counts(), "terms": t.counts(), "typing_warnings": list(t.warnings),
                "charges": ("eqeq" if isinstance(calc.charge_model, str) and calc.charge_model == "eqeq"
                            else "user" if calc.q is not None else "none"),
                "energy_parts_eV": {k: round(v, 6) for k, v in calc.parts.items()}}
