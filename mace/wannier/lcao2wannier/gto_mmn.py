"""Exact analytic GTO overlaps and MMN for CRYSTAL LCAO output.

Computes the Wannier90 MMN overlaps M_mn(k,b) = <u_mk|u_n,k+b> exactly from the
Gaussian basis, with no midpoint or Löwdin approximation:

    M(k,b) = C(k)^dag . Stilde^{(b)}(k+b) . C(k+b)
    Stilde^{(b)}_uv(k+b) = sum_g e^{i(k+b).g} S^{(b)}_uv(g)
    S^{(b)}_uv(g) = integral phi_u(r-A_u) e^{-i b.r} phi_v(r-A_v-g) dr

The momentum-shifted Gaussian integral is analytic: the e^{-i b.r} factor shifts
the Gaussian product centre to the complex point P - i b/2p and adds the scalar
e^{-i b.P} e^{-|b|^2/4p}. Reduces exactly to the plain overlap at b=0.

CRYSTAL conventions (validated to printed precision against S(R), see
scripts/analytic_mmn_dev.py):
  * Convention II Bloch phase e^{ik.g}; AOs centred at the nucleus.
  * printed coefficients used with UNNORMALIZED primitives e^{-a r^2}; the
    contracted AO is normalized to unit self-overlap.
  * order of internal storage (manual basisset p.27): S=s; P=x,y,z;
    D=2z^2-x^2-y^2,xz,yz,x^2-y^2,xy; F=7, G=9 real solid harmonics (see _FCART,
    _GCART). The Hermite/overlap engine is angular-momentum general; only the
    real-solid-harmonic -> Cartesian tables are per-l.
"""
import re
import numpy as np
from math import pi, sqrt

ANG2BOHR = 1.8897259886

# real-solid-harmonic -> Cartesian (powers, coefficient)
_DCART = [
    [((0, 0, 2), 2.0), ((2, 0, 0), -1.0), ((0, 2, 0), -1.0)],  # 2z^2-x^2-y^2
    [((1, 0, 1), 1.0)],                                         # xz
    [((0, 1, 1), 1.0)],                                         # yz
    [((2, 0, 0), 1.0), ((0, 2, 0), -1.0)],                      # x^2-y^2
    [((1, 1, 0), 1.0)],                                         # xy
]
_PCART = [[((1, 0, 0), 1.0)], [((0, 1, 0), 1.0)], [((0, 0, 1), 1.0)]]
_SCART = [[((0, 0, 0), 1.0)]]
# F real solid harmonics, CRYSTAL "order of internal storage" (manual basisset
# p.27): (2z^2-3x^2-3y^2)z, (4z^2-x^2-y^2)x, (4z^2-x^2-y^2)y, (x^2-y^2)z, xyz,
# (x^2-3y^2)x, (3x^2-y^2)y.
_FCART = [
    [((0, 0, 3), 2.0), ((2, 0, 1), -3.0), ((0, 2, 1), -3.0)],   # (2z^2-3x^2-3y^2)z
    [((1, 0, 2), 4.0), ((3, 0, 0), -1.0), ((1, 2, 0), -1.0)],   # (4z^2-x^2-y^2)x
    [((0, 1, 2), 4.0), ((2, 1, 0), -1.0), ((0, 3, 0), -1.0)],   # (4z^2-x^2-y^2)y
    [((2, 0, 1), 1.0), ((0, 2, 1), -1.0)],                      # (x^2-y^2)z
    [((1, 1, 1), 1.0)],                                         # xyz
    [((3, 0, 0), 1.0), ((1, 2, 0), -3.0)],                      # (x^2-3y^2)x
    [((2, 1, 0), 3.0), ((0, 3, 0), -1.0)],                      # (3x^2-y^2)y
]
# G real solid harmonics, same source (manual basisset p.27). G shells are
# polarization-only in CRYSTAL but are included here for completeness.
_GCART = [
    [((4, 0, 0), 3.0), ((2, 2, 0), 6.0), ((2, 0, 2), -24.0),
     ((0, 4, 0), 3.0), ((0, 2, 2), -24.0), ((0, 0, 4), 8.0)],   # 3x^4+6x^2y^2-24x^2z^2+3y^4-24y^2z^2+8z^4
    [((1, 0, 3), 4.0), ((3, 0, 1), -3.0), ((1, 2, 1), -3.0)],   # (4z^2-3x^2-3y^2)xz
    [((0, 1, 3), 4.0), ((2, 1, 1), -3.0), ((0, 3, 1), -3.0)],   # (4z^2-3x^2-3y^2)yz
    [((2, 0, 2), 6.0), ((4, 0, 0), -1.0), ((0, 2, 2), -6.0),
     ((0, 4, 0), 1.0)],                                         # (x^2-y^2)(6z^2-x^2-y^2)
    [((1, 1, 2), 6.0), ((3, 1, 0), -1.0), ((1, 3, 0), -1.0)],   # (6z^2-x^2-y^2)xy
    [((3, 0, 1), 1.0), ((1, 2, 1), -3.0)],                      # (x^2-3y^2)xz
    [((2, 1, 1), 3.0), ((0, 3, 1), -1.0)],                      # (3x^2-y^2)yz
    [((4, 0, 0), 1.0), ((2, 2, 0), -6.0), ((0, 4, 0), 1.0)],    # x^4-6x^2y^2+y^4
    [((3, 1, 0), 1.0), ((1, 3, 0), -1.0)],                      # (x^2-y^2)xy
]
_LCART = {0: _SCART, 1: _PCART, 2: _DCART, 3: _FCART, 4: _GCART}


# ----------------------------------------------------------------------------
# Basis parsing
# ----------------------------------------------------------------------------
def parse_gto_basis(lines):
    """Parse the 'LOCAL ATOMIC FUNCTIONS BASIS SET' block.

    Returns the ordered AO list matching the H(R)/S(R) matrix ordering. Each AO
    is a dict(center (Bohr), terms=[((lx,ly,lz), ang_coef)], prims=[(exp,coef)],
    norm). Handles atom basis reuse (atoms of equal symbol share a template)."""
    LMAP = {'S': 0, 'P': 1, 'D': 2, 'F': 3, 'G': 4}
    COL = {0: 1, 1: 2, 2: 3, 3: 3, 4: 3}
    start = next(i for i, ln in enumerate(lines)
                 if 'LOCAL ATOMIC FUNCTIONS BASIS SET' in ln)
    atoms = []           # (idx, sym, center)
    sym_shells = {}      # sym -> [shell dict] from the first atom of that sym
    cur_sym = cur_center = cur_shell = None
    # CRYSTAL prints element symbols in UPPER case (C, H, SC, FE, MG, ...), so the
    # symbol is one or two capital letters -- not [A-Z][a-z]? (which silently
    # dropped the 2-letter metals like SC, and their shells with them).
    atom_re = re.compile(r'^\s*(\d+)\s+([A-Z][A-Z]?)\s+'
                         r'(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$')
    shell_re = re.compile(r'^\s*(\d+)\s*-?\s*(\d*)\s+([SPDFG])\s*$')
    # Fixed-width columns: a full field (large exponents on heavy atoms) glues
    # adjacent numbers ("2.113E+02-2.283E+00"), so allow \s* (not \s+) between.
    prim_re = re.compile(r'^\s*(-?\d\.\d+E[+-]\d+)\s*(-?\d\.\d+E[+-]\d+)\s*'
                         r'(-?\d\.\d+E[+-]\d+)\s*(-?\d\.\d+E[+-]\d+)\s*$')
    for ln in lines[start + 2:]:
        if 'OVERLAP MATRIX' in ln:
            break
        m = atom_re.match(ln)
        if m:
            cur_sym = m.group(2)
            cur_center = np.array([float(m.group(3)), float(m.group(4)),
                                   float(m.group(5))])
            atoms.append((int(m.group(1)), cur_sym, cur_center))
            cur_shell = None
            continue
        m = shell_re.match(ln)
        if m and cur_sym is not None:
            cur_shell = dict(l=LMAP[m.group(3)], prims=[])
            first = next(a[0] for a in atoms if a[1] == cur_sym)
            if atoms[-1][0] == first:
                sym_shells.setdefault(cur_sym, []).append(cur_shell)
            continue
        m = prim_re.match(ln)
        if m and cur_shell is not None:
            v = [float(m.group(i)) for i in range(1, 5)]
            cur_shell['prims'].append((v[0], v[COL[cur_shell['l']]]))

    aos = []
    for idx, sym, center in atoms:
        for sh in sym_shells[sym]:
            for comp in _LCART[sh['l']]:
                aos.append(dict(center=center, terms=comp, prims=sh['prims']))
    for ao in aos:
        ao['norm'] = 1.0
        ao['norm'] = 1.0 / sqrt(_pair_overlap(ao, ao, np.zeros(3)).real)
    return aos


# ----------------------------------------------------------------------------
# Gaussian integrals
# ----------------------------------------------------------------------------
def _herm_1d(p, PA, PB, la, lb):
    """1-D overlap polynomial via Hermite recursion; PA/PB may be complex.

    p, PA, PB may be scalars or broadcastable arrays (e.g. one entry per
    primitive pair); the recursion is elementwise and returns S[la, lb] with the
    same shape as the broadcast of the inputs."""
    p = np.asarray(p, dtype=complex)
    PA = np.asarray(PA, dtype=complex)
    PB = np.asarray(PB, dtype=complex)
    shp = np.broadcast_shapes(p.shape, PA.shape, PB.shape)
    half = 0.5 / p
    S = np.zeros((la + 2, lb + 2) + shp, dtype=complex)
    S[0, 0] = np.ones(shp, dtype=complex)
    for i in range(la + 1):
        for j in range(lb + 1):
            if i == 0 and j == 0:
                continue
            if i > 0:
                S[i, j] = PA * S[i - 1, j] + half * (
                    (i - 1) * S[i - 2, j] + j * S[i - 1, j - 1])
            else:
                S[i, j] = PB * S[i, j - 1] + half * (
                    i * S[i - 1, j - 1] + (j - 1) * S[i, j - 2])
    return S[la, lb]


def _prim_mom(a, A, lA, b, B, lB, bvec):
    """<g_a(r-A)| e^{-i bvec.r} | g_b(r-B)>, analytic (complex).

    a, b may be scalars or broadcastable arrays of exponents (one element per
    primitive pair); the return has the broadcast shape."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    p = a + b
    AB = A - B
    P = [(a * A[d] + b * B[d]) / p for d in range(3)]
    bdotP = bvec[0] * P[0] + bvec[1] * P[1] + bvec[2] * P[2]
    res = ((pi / p) ** 1.5
           * np.exp(-a * b / p * np.dot(AB, AB))
           * np.exp(-np.dot(bvec, bvec) / (4 * p))
           * np.exp(-1j * bdotP)).astype(complex)
    for d in range(3):
        Ppd = P[d] - 0.5j * bvec[d] / p
        res = res * _herm_1d(p, Ppd - A[d], Ppd - B[d], lA[d], lB[d])
    return res


def _pair_overlap(ao_i, ao_j, Rj, bvec=None):
    """<ao_i(0)| e^{-i bvec.r} | ao_j(+Rj)>. bvec=None -> plain overlap.

    Vectorized over the primitive-pair grid: the inner double loop over
    primitives is replaced by a single broadcast call to _prim_mom."""
    if bvec is None:
        bvec = np.zeros(3)
    Ci = ao_i['center']
    Cj = ao_j['center'] + Rj
    ea = np.array([pr[0] for pr in ao_i['prims']])[:, None]   # (na, 1)
    ca = np.array([pr[1] for pr in ao_i['prims']])[:, None]
    eb = np.array([pr[0] for pr in ao_j['prims']])[None, :]   # (1, nb)
    cb = np.array([pr[1] for pr in ao_j['prims']])[None, :]
    cab = ca * cb                                              # (na, nb)
    tot = 0.0 + 0.0j
    for (li, ci) in ao_i['terms']:
        for (lj, cj) in ao_j['terms']:
            block = _prim_mom(ea, Ci, li, eb, Cj, lj, bvec)   # (na, nb)
            tot += ci * cj * np.sum(cab * block)
    return tot * ao_i['norm'] * ao_j['norm']


# ----------------------------------------------------------------------------
# MMN assembly
# ----------------------------------------------------------------------------
def reciprocal_bohr(lattice_bohr):
    """Reciprocal lattice rows (1/Bohr), 2pi convention: b_i . a_j = 2pi d_ij."""
    return 2 * pi * np.linalg.inv(lattice_bohr).T


def build_Sb_one(aos, bvec, Rg, cutoff=None, centers=None):
    """S^{(b)}(g) for ONE cell whose Cartesian shift is Rg (Bohr); (n_ao, n_ao)
    complex. Cells are independent, so this is the unit of work distributed by
    the parallel analytic precompute (one (b, cell) task each).

    The n*n centre-distance test is vectorized and the analytic integral runs over
    the near-pair list only, so the cost scales with the number of overlapping
    pairs, not n^2 (the old per-element Python cutoff check dominated for large n).
    `centers` may be passed precomputed to avoid rebuilding it per call."""
    n = len(aos)
    if centers is None:
        centers = np.array([ao['center'] for ao in aos])
    M = np.zeros((n, n), dtype=complex)
    cj = centers + Rg
    if cutoff is not None:
        d2 = ((cj[None, :, :] - centers[:, None, :]) ** 2).sum(-1)
        pairs = np.argwhere(d2 <= cutoff * cutoff)
    else:
        pairs = ((i, j) for i in range(n) for j in range(n))
    for i, j in pairs:
        M[i, j] = _pair_overlap(aos[int(i)], aos[int(j)], Rg, bvec)
    return M


def build_Sb_cells(aos, bvec, cells_R, cutoff=None):
    """S^{(b)}(g) for every cell g. cells_R = [(g_tuple, R_cart_bohr)]. Returns
    dict g_tuple -> (n_ao,n_ao) complex (see build_Sb_one for the cutoff)."""
    centers = np.array([ao['center'] for ao in aos])
    return {g: build_Sb_one(aos, bvec, Rg, cutoff, centers) for g, Rg in cells_R}


def Sb_at_kpb(Sb_cells, kpb_frac):
    """Fourier sum sum_g e^{i 2pi (k+b).g} S^{(b)}(g)."""
    n = next(iter(Sb_cells.values())).shape[0]
    S = np.zeros((n, n), dtype=complex)
    for g, M in Sb_cells.items():
        S += np.exp(2j * pi * np.dot(kpb_frac, g)) * M
    return S


def mmn_block(C_k, C_kpb, Sb_cells, kpb_frac):
    """M_mn(k,b) = C(k)^dag . Stilde^{(b)}(k+b) . C(k+b)."""
    S = Sb_at_kpb(Sb_cells, kpb_frac)
    return C_k.conj().T @ S @ C_kpb


def precompute_Sb_parallel(aos, unique_b, cells_R, cutoff=None, n_proc=1,
                           verbose=False):
    """S^{(b)}(g) for every b in ``unique_b`` (dict key -> b_cart), distributed
    over worker processes one (b, cell) task each — the same decomposition as
    the parallel .mmn write path (build_Sb_cells is the dominant cost of the
    analytic method on coarse meshes; Sc_all_center: 9 b x 7 cells at 1424
    AOs). Identical results to the serial dict build.

    cells_R : list of (g_tuple, Rg_cart_bohr) as used by Sb_at_kpb.
    Returns {b_key: {g_tuple: (n_ao, n_ao) complex}}.
    """
    keys = list(unique_b)
    ntasks = len(keys) * len(cells_R)
    if n_proc > 1 and ntasks > 1:
        import multiprocessing as mp
        nw = min(n_proc, ntasks)
        bg = [(bi, gi) for bi in range(len(keys))
              for gi in range(len(cells_R))]
        args = [(aos, unique_b[keys[bi]], cells_R[gi][1], cutoff)
                for bi, gi in bg]
        with mp.get_context('spawn').Pool(nw) as pool:
            res = pool.starmap(build_Sb_one, args)
        Sb_by_b = {key: {} for key in keys}
        for (bi, gi), M in zip(bg, res):
            Sb_by_b[keys[bi]][cells_R[gi][0]] = M
        if verbose:
            print(f"  Analytic GTO: S^(b)(g) over {len(keys)} b x "
                  f"{len(cells_R)} cell(s) = {ntasks} tasks [{nw} workers]")
    else:
        Sb_by_b = {key: build_Sb_cells(aos, unique_b[key], cells_R,
                                       cutoff=cutoff) for key in keys}
        if verbose:
            print(f"  Analytic GTO: S^(b)(g) for {len(keys)} unique "
                  f"b-vector(s) x {len(cells_R)} cell(s)")
    return Sb_by_b


def build_b_table(kpoints, neighbors, recip_bohr):
    """Shared (k, i) -> b-key table for the analytic MMN paths.

    ``neighbors`` iterates (k_idx, ni, neighbor_id, G_shift). Returns
    (bkey_of, unique_b, b0_key) with the b=0 entry always present (needed
    for the GTO-metric orthonormalization). One implementation for BOTH the
    .mmn write path and the hybrid pool overlaps — the b-vector rounding
    convention must never drift between them.
    """
    kpoints = np.asarray(kpoints, float)
    bkey_of, unique_b = {}, {}
    for k_idx, ni, nb_id, G in neighbors:
        b_frac = kpoints[nb_id] + np.asarray(G, float) - kpoints[k_idx]
        b_cart = b_frac @ recip_bohr
        key = tuple(np.round(b_cart, 6))
        bkey_of[(k_idx, ni)] = key
        unique_b.setdefault(key, b_cart)
    b0_key = tuple(np.round(np.zeros(3), 6))
    unique_b.setdefault(b0_key, np.zeros(3))
    return bkey_of, unique_b, b0_key


def gto_block_overlap(Ca, S, Cb, n_spatial):
    """C_a^H S C_b with the 2-component spin-block sum when the coefficient
    rows stack alpha above beta (rows == 2 * n_spatial); the GTO overlap is
    spatial and spin-diagonal."""
    if Ca.shape[0] == 2 * n_spatial:
        return (Ca[:n_spatial].conj().T @ S @ Cb[:n_spatial]
                + Ca[n_spatial:].conj().T @ S @ Cb[n_spatial:])
    return Ca.conj().T @ S @ Cb


def orthonormalize_gto_metric(C_list, Sb0, kpoints, n_spatial):
    """Orthonormalize each C(k) against the exact GTO overlap at b=0
    (in place): O = C^H S_GTO(k) C -> C <- C O^{-1/2}. This is what forces
    M(k,0) = I and every M(k,b) singular value <= 1 — C is only orthonormal
    w.r.t. the parsed Fourier S(k), which differs from the GTO overlap at
    printed-coefficient precision."""
    for k in range(len(C_list)):
        S0 = Sb_at_kpb(Sb0, kpoints[k])
        Ck = C_list[k]
        O = gto_block_overlap(Ck, S0, Ck, n_spatial)
        w, V = np.linalg.eigh(0.5 * (O + O.conj().T))
        isq = (1.0 / np.sqrt(np.maximum(w.real, 1e-12)))[:, None]
        C_list[k] = Ck @ (V @ (isq * V.conj().T))
    return C_list
