# Copyright (c) 2025 Computational Materials Science Team (lcao2wannier, author William Comaskey).
# MIT License, see LICENSE in this directory.
# Vendored into MACE from lcao2wannier 1.0.0; local changes are listed in VENDORED.md.
"""
Generic real-space (H(R)) postprocessing.

Hermitization and time-reversal enforcement are properties every tight-binding
model built from a Hermitian H(k) should satisfy exactly; small numerical
asymmetry creeps in from finite k-mesh Fourier transforms and (for SOC models)
imperfect Kramers degeneracy. These functions clean that up directly on the
real-space matrices, with no crystal-symmetry information required.
"""

import numpy as np
from typing import Dict, List, Tuple
import warnings
from datetime import datetime


def enforce_hermiticity(
    real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]]
) -> Dict[Tuple[int, int, int], Dict[str, np.ndarray]]:
    """
    Enforce Hermiticity: M(R) = [M(R) + M(-R)^dag] / 2

    This ensures that the Hamiltonian and overlap matrices satisfy the
    required symmetry relation between R and -R. Operates on whatever
    matrix keys are present (e.g. just 'H', or 'H' and 'S').
    """
    result = {}
    all_R = set(real_space_matrices.keys())

    for R, matrices in real_space_matrices.items():
        R_neg = tuple(-x for x in R)

        if R_neg in all_R:
            neg_matrices = real_space_matrices[R_neg]
            out = {}
            for key, M in matrices.items():
                out[key] = (M + neg_matrices[key].conj().T) / 2.0
            result[R] = out
        else:
            # No -R partner: keep as-is (should not happen for proper crystal)
            result[R] = {key: M.copy() for key, M in matrices.items()}

    return result


def enforce_time_reversal(
    real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]],
    norbs_spatial: int
) -> Dict[Tuple[int, int, int], Dict[str, np.ndarray]]:
    """
    Enforce time-reversal symmetry for SOC systems.

    For spinful systems in block spin ordering [up|dn]:
        M(R) = [M(R) + sigma_y @ M*(-R) @ sigma_y] / 2

    where sigma_y operates in spin space as:
        sigma_y = [[0, -i], [i, 0]] -> in block form: [[0, -I], [I, 0]] (times i)

    Actually, the time-reversal operator is T = i*sigma_y * K (complex conjugation).
    So: M(R) -> T M(-R) T^{-1} = sigma_y_block @ M*(-R) @ sigma_y_block^{-1}

    In block ordering, sigma_y_block swaps up/dn blocks with a sign. Operates
    on whatever matrix keys are present (e.g. just 'H', or 'H' and 'S').
    """
    n = norbs_spatial
    n2 = 2 * n  # Total dimension with spin

    # Build sigma_y in block form:
    # In up/dn block ordering:
    # sigma_y_block: up->dn with factor +1, dn->up with factor -1
    # Acting on [psi_up, psi_dn]: sigma_y gives [-psi_dn*, psi_up*]
    # But for matrix transformation: M -> U_L @ M* @ U_R
    # with U_L = sigma_y^T (conjugate transpose in spin space)
    # Reference: symmhr_addrptblock.py uses syl and syr matrices

    # Left: sigma_y^T * i  =  [[0, 1], [-1, 0]] in interleaved
    # Right: sigma_y * i   =  [[0, -1], [1, 0]] in interleaved
    # In block ordering:
    umat_L = np.zeros((n2, n2), dtype=np.complex128)
    umat_R = np.zeros((n2, n2), dtype=np.complex128)

    # sigma_y^T in block form: up-up=0, up-dn=I, dn-up=-I, dn-dn=0
    umat_L[:n, n:] = np.eye(n)      # up-dn block = +I
    umat_L[n:, :n] = -np.eye(n)     # dn-up block = -I

    # sigma_y in block form: up-up=0, up-dn=-I, dn-up=I, dn-dn=0
    umat_R[:n, n:] = -np.eye(n)     # up-dn block = -I
    umat_R[n:, :n] = np.eye(n)      # dn-up block = +I

    result = {}
    all_R = set(real_space_matrices.keys())

    for R, matrices in real_space_matrices.items():
        R_neg = tuple(-x for x in R)

        if R_neg in all_R:
            neg_matrices = real_space_matrices[R_neg]
            out = {}
            for key, M in matrices.items():
                M_neg_conj = neg_matrices[key].conj()
                M_TR = umat_L @ M_neg_conj @ umat_R
                out[key] = (M + M_TR) / 2.0
            result[R] = out
        else:
            result[R] = {key: M.copy() for key, M in matrices.items()}

    return result


def read_hr_file(path: str) -> Tuple[int, List[Tuple[int, int, int]],
                                      List[int], Dict[Tuple[int, int, int], np.ndarray]]:
    """
    Read a wannier90 ``*_hr.dat`` file.

    Returns
    -------
    num_wann : int
    R_list : list of (int, int, int)
        R-vectors in file order (may repeat if the file is malformed, but
        wannier90 never emits duplicates).
    ndegen : list of int
        Degeneracy weight for each R-vector, same order as R_list.
    H : dict
        {R: ndarray (num_wann, num_wann)} the Hamiltonian block at each R.
    """
    with open(path, 'r') as f:
        f.readline()  # comment/timestamp
        num_wann = int(f.readline().split()[0])
        nrpt = int(f.readline().split()[0])

        ndegen: List[int] = []
        while len(ndegen) < nrpt:
            ndegen.extend(int(x) for x in f.readline().split())

        R_list = []
        H = {}
        for _ in range(nrpt):
            Hmat = np.zeros((num_wann, num_wann), dtype=np.complex128)
            R = None
            for _ in range(num_wann * num_wann):
                parts = f.readline().split()
                rx, ry, rz, m, n = (int(x) for x in parts[:5])
                re, im = float(parts[5]), float(parts[6])
                Hmat[m - 1, n - 1] = re + 1j * im
                R = (rx, ry, rz)
            R_list.append(R)
            H[R] = Hmat

    return num_wann, R_list, ndegen, H


def write_hr_file(
    path: str,
    num_wann: int,
    R_list: List[Tuple[int, int, int]],
    ndegen: List[int],
    H: Dict[Tuple[int, int, int], np.ndarray],
    threshold: float = 0.0,
    header: str = None,
) -> None:
    """
    Write a wannier90 ``*_hr.dat`` file.

    Matrix elements with |H_mn(R)| below ``threshold`` are written as exactly
    zero (the R-vector/degeneracy list is unchanged: wannier90's format fixes
    the number of blocks up front, so small hoppings are zeroed, not dropped).
    """
    if header is None:
        header = ("Written by lcao_to_wannier90 (Stage 3 postprocessing) on "
                   + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    with open(path, 'w') as f:
        f.write(f"{header}\n")
        f.write(f"{num_wann:12d}\n")
        f.write(f"{len(R_list):12d}\n")
        for i in range(0, len(ndegen), 15):
            f.write(''.join(f"{d:5d}" for d in ndegen[i:i + 15]) + "\n")

        for R in R_list:
            Hmat = H[R]
            for n in range(num_wann):
                for m in range(num_wann):
                    val = Hmat[m, n]
                    if abs(val) < threshold:
                        val = 0.0
                    f.write(f"{R[0]:5d}{R[1]:5d}{R[2]:5d}{m + 1:5d}{n + 1:5d}"
                            f"{val.real:12.6f}{val.imag:12.6f}\n")
