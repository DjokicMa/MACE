#!/usr/bin/env python3
# Copyright (c) 2025 William Comaskey. MIT License, see LICENSE in this directory.
# Vendored into MACE from lcao2wannier 1.0.0; local changes are listed in VENDORED.md.
"""
lcao2wannier Multi-Stage Workflow Script

This script implements the Wannier90 workflow with three stages:

Stage 1: Generate .win file from CRYSTAL/LCAO output
Stage 2: Generate .eig, .amn, .mmn files using .nnkp neighbor information
Stage 3: Postprocess wannier90_hr.dat (Hermitization + time-reversal)
Stage 4: Plot LCAO band structure with PDWF projectability coloring

Usage:
    # Stage 1: Create .win file
    python lcao_to_wannier90.py --stage 1 --input material.out --seedname material

    # Then run: wannier90.x -pp material

    # Stage 2: Create data files using .nnkp
    python lcao_to_wannier90.py --stage 2 --input material.out --seedname material

    # Then run: wannier90.x material

    # Stage 3: Postprocess the tight-binding Hamiltonian
    python lcao_to_wannier90.py --stage 3 --input material.out --seedname material

    # Stage 4: Plot band structure with projectability
    python lcao_to_wannier90.py --stage 4 --input material.out --seedname material

For help:
    python lcao_to_wannier90.py --help
"""

import argparse
import sys
import os
import copy
import numpy as np
from pathlib import Path

from . import (
    parse_overlap_and_fock_matrices,
    parse_calculation_parameters,
    parse_atomic_basis_info,
    create_spin_block_matrices,
    create_nonsoc_full_matrices,
    prepare_real_space_matrices,
    Wannier90Engine,
    suggest_optimal_window,
    analyze_band_window,
    parse_atoms_from_crystal_output,
    estimate_fermi_energy,
)
from .parser import parse_overlap_and_fock_matrices_cached
from .projectability import select_bands_by_projectability, smart_select_bands
from .parser import parse_orbital_types
from .parser import parse_overlap_and_fock_matrices_streaming
from .utils import prune_zero_rvectors
from .basis_parser import parse_basis_shells, get_atom_list
from .valence_config import (
    build_target_mask, compute_num_wann, summarize_config,
)
from .lcao_pdwf import (
    compute_lowdin_projectability, classify_bands, determine_windows,
    check_frozen_interlopers, check_band_count, print_pdwf_summary,
    ClassificationParams,
)

# Threshold above which --method direct warns the user
LARGE_BASIS_THRESHOLD = 200


def _load_matrices(args):
    """Load real-space matrices + header, honoring --memory.

    Returns (lines, params, H_R_dict, S_R_dict, lattice_vectors_list).

    * fast (default): legacy behaviour — ``f.readlines()`` the whole file, parse,
      and organize into dicts. ``lines`` is the full file.
    * low: single streaming pass (no full readlines, no intermediate matrix
      list, float64 for non-SOC). ``lines`` is only the pre-matrix header
      region, which is all the downstream basis/orbital/atom parsers need.

    The two paths produce identical H_R_dict/S_R_dict (validated by
    scripts/validate_streaming_parser.py); ``low`` just trades the file buffer
    and a representation copy for a single pass.
    """
    low_mem = getattr(args, 'memory', 'fast') == 'low'
    if low_mem:
        print("  [memory=low] streaming parse: single pass, "
              "header-only buffer, float64 for non-SOC")
        H_R_dict, S_R_dict, lattice_vectors_list, lines = (
            parse_overlap_and_fock_matrices_streaming(args.input,
                                                      promote_complex='auto')
        )
        params = parse_calculation_parameters(lines)
        return lines, params, H_R_dict, S_R_dict, lattice_vectors_list

    with open(args.input, 'r') as f:
        lines = f.readlines()
    params = parse_calculation_parameters(lines)
    raw_matrices, lattice_vectors_list = parse_overlap_and_fock_matrices_cached(
        args.input, lines, cache_dir=getattr(args, 'lcao_cache_dir', None))
    H_R_dict, S_R_dict = {}, {}
    for mat_info in raw_matrices:
        R = tuple(mat_info['lattice_vector'])
        if mat_info['type'] == 'overlap':
            S_R_dict[R] = mat_info['data']
        elif mat_info['type'] == 'fock':
            spin_channel = mat_info.get('spin_channel', 0)
            H_R_dict.setdefault(R, {})[spin_channel] = mat_info['data']
    return lines, params, H_R_dict, S_R_dict, lattice_vectors_list

# Maximum allowed discrepancy between CRYSTAL-reported Fermi and
# HOMO estimated from (eigenvalues, num_electrons). Beyond this, the
# reported Fermi is deemed inconsistent with the printed H(R) matrices
# (e.g. SPINLOCK+level-shifter quirk in CRYSTAL23 SOC outputs) and we
# fall back to the electron-count estimate with a prominent warning.
FERMI_FRAME_TOLERANCE_EV = 1.0


def _check_disentanglement(args, engine, stage):
    """Validate the just-written .win against Wannier90 disentanglement rules
    using the engine's per-k eigenvalues. Prints a PASS/FAIL report; never
    raises (a violation is reported, not fatal, so the user still gets files).
    """
    try:
        from .wannier_checks import check_seed_windows, format_report
        win = f"{args.seedname}.win"
        eigs = getattr(engine, 'eigenvalues_list', None)
        if not os.path.exists(win) or not eigs:
            return
        ef = getattr(engine, 'e_fermi', None) or 0.0
        band_idx = getattr(engine, 'selected_band_indices', None)
        if band_idx is not None:
            eig_per_k = [np.asarray(e)[band_idx] - ef for e in eigs]
        else:
            eig_per_k = [np.asarray(e) - ef for e in eigs]
        rep = check_seed_windows(win, eig_per_k)
        print(format_report(
            rep, title=f"Stage {stage} consistency check: {args.seedname}"))
        if not rep.ok:
            print("  ⚠ wannier90.x will reject this .win. Adjust the windows "
                  "(see suggestion) or band selection before running wannier90.")
    except Exception as exc:
        print(f"  ⚠ Could not run disentanglement consistency check: {exc}")


def _detect_spin_type(filepath, max_lines=20000):
    """Classify the CRYSTAL calculation from its header.

    Returns one of:
      'soc'        - two-component / spin-orbit (spinor, channels coupled)
      'collinear'  - unrestricted spin-polarized (separate ALPHA/BETA Fock)
      'restricted' - closed-shell or restricted open-shell (single channel)

    Only the header is scanned (the 'TYPE OF CALCULATION' line appears near the
    top), so this is cheap even on multi-GB files.
    """
    soc = unrestricted = restricted = False
    with open(filepath, 'r', errors='replace') as f:
        for i, line in enumerate(f):
            if 'TWO-COMPONENT' in line and 'SCF' in line:
                soc = True
            up = line.upper()
            if 'UNRESTRICTED' in up:
                unrestricted = True
            if 'RESTRICTED' in up and 'UNRESTRICTED' not in up:
                restricted = True
            if i >= max_lines:
                break
    if soc:
        return 'soc'
    if unrestricted:
        return 'collinear'
    return 'restricted'


# Map --spin choice to the CRYSTAL Fock spin-channel key.
_SPIN_CHANNEL_KEY = {'alpha': 'ALPHA_ALPHA', 'beta': 'BETA_BETA'}


def _resolve_spin_runs(args, parser):
    """Validate --spin against the detected calculation type and return the
    list of per-run (spin_label, spin_channel_key, seedname) tuples.

    --spin is only meaningful for collinear spin-polarized stage 1/2 runs;
    on restricted or SOC outputs an explicit --spin is an error.
    """
    spin = getattr(args, 'spin', None)
    if args.stage not in (1, 2, 'all'):
        if spin is not None:
            parser.error("--spin only applies to --stage 1, 2, and all.")
        return [(None, None, args.seedname)]

    spin_type = _detect_spin_type(args.input)
    if spin_type != 'collinear':
        if spin is not None:
            parser.error(
                f"--spin is only valid for collinear spin-polarized "
                f"(UNRESTRICTED) outputs; this file is '{spin_type}'. "
                f"Remove --spin.")
        return [(None, None, args.seedname)]

    # Collinear spin-polarized: default (None) -> both channels.
    choice = spin or 'both'
    if choice == 'both':
        return [('alpha', 'ALPHA_ALPHA', f"{args.seedname}_alpha"),
                ('beta', 'BETA_BETA', f"{args.seedname}_beta")]
    return [(choice, _SPIN_CHANNEL_KEY[choice], args.seedname)]


def _sanity_check_fermi_energy(engine, params, args, stage_name=""):
    """
    Compare the parsed Fermi energy against the HOMO estimated from
    eigenvalues + electron count. Detects CRYSTAL SPINLOCK/level-shifter
    frame mismatches where the Fermi and H(R) matrices end up in different
    energy references.

    If `--fermi-energy` was passed explicitly, trusts that and returns.
    If the parsed Fermi disagrees with the electron-count estimate by
    more than FERMI_FRAME_TOLERANCE_EV, replaces engine.e_fermi with the
    estimate and prints a loud warning.

    Must be called AFTER engine.solve_all_kpoints().
    """
    # User override always wins — they've made a conscious choice.
    if getattr(args, 'fermi_energy', None) is not None:
        return

    # Spin-polarized single-channel run: the electron-count HOMO estimate needs
    # the per-spin electron count, which CRYSTAL reports only as a total. Skip
    # the check and trust the parsed (global) Fermi or an explicit --fermi-energy.
    if getattr(args, '_spin_label', None) is not None:
        print(f"  (Skipping electron-count Fermi check for "
              f"{args._spin_label.upper()} channel; using parsed Fermi.)")
        return

    # Can't sanity-check without a parsed Fermi and electron count.
    if params.fermi_energy is None or params.num_electrons is None:
        return

    # Compute HOMO-based estimate from our eigenvalues.
    # Spin degeneracy: SOC systems have spinor bands (1 electron/band);
    # non-SOC systems have spatial bands (2 electrons/band in the
    # non-spin-polarized case treated by our solver).
    has_soc = getattr(params, 'has_soc', False) or getattr(engine, 'has_soc', False)
    spin_deg = 1 if has_soc else 2
    try:
        # CRYSTAL reports the Fermi energy at the VBM (top of the highest
        # occupied band) for insulators, NOT at mid-gap. Compare against the VBM
        # from band filling, so a genuine frame mismatch (SPINLOCK/level shifter,
        # off by the shift amount) is still caught, WITHOUT falsely flagging the
        # legitimate gap/2 offset that separates the VBM from mid-gap in every
        # insulator (e.g. h-BN: 6.6 eV gap -> 3.3 eV VBM-vs-midgap difference).
        eigs = np.asarray(engine.eigenvalues_list, dtype=float)  # (nk, nbands)
        n_occ = int(round(params.num_electrons / spin_deg))
        if n_occ < 1 or n_occ > eigs.shape[1]:
            return
        estimated = float(eigs[:, n_occ - 1].max())  # VBM (HOMO across the BZ)
    except Exception as exc:
        print(f"  ⚠ Could not run Fermi sanity check: {exc}")
        return

    discrepancy = abs(params.fermi_energy - estimated)
    if discrepancy <= FERMI_FRAME_TOLERANCE_EV:
        # Parsed and estimated agree — nothing to do.
        return

    # Discrepancy is large. Almost always means the CRYSTAL output had
    # SPINLOCK active or a level-shifter-altered Fermi (e.g. two-component
    # SOC runs with LOCKING - FERMI ENERGY ALTERED BY LEVEL SHIFTER).
    stage_tag = f" ({stage_name})" if stage_name else ""
    print()
    print("!" * 78)
    print(f"  ⚠  FERMI ENERGY FRAME MISMATCH DETECTED{stage_tag}")
    print("!" * 78)
    print(f"  Parsed from CRYSTAL output : {params.fermi_energy:+.4f} eV")
    print(f"  VBM from band filling      : {estimated:+.4f} eV "
          f"(N_electrons={params.num_electrons})")
    print(f"  Discrepancy                : {discrepancy:.4f} eV "
          f"(tolerance {FERMI_FRAME_TOLERANCE_EV} eV)")
    print()
    print("  The parsed Fermi and the printed H(R) matrices appear to live")
    print("  in different energy reference frames. Common causes in CRYSTAL23:")
    print("    - SPINLOCK active ('LOCKING - FERMI ENERGY ALTERED BY LEVEL")
    print("       SHIFTER' in the output)")
    print("    - 'EIGENVALUE LEVEL SHIFTING OF X HARTREE' applied but not")
    print("       removed before the Fermi-energy header was written")
    print("    - Two-component SOC SCF with an internal Fermi-bias field")
    print()
    print(f"  → Falling back to the VBM from band filling "
          f"({estimated:+.4f} eV).")
    print(f"  → To override this, re-run with: "
          f"--fermi-energy <your_value_in_eV>")
    print("!" * 78)
    print()

    # Apply the fallback to both params (used later in the stage) and engine.
    params.fermi_energy = estimated
    engine.e_fermi = estimated


def _apply_method_projectability(engine, args, has_soc=False):
    """Apply Method 1: Smart projectability-based band selection."""
    print("\nProjectability-based band selection...")
    print("-" * 80)

    # Check for user override of num_wann
    user_num_wann = getattr(args, 'num_wann', None)

    # Use smart selector if eigenvalues and Fermi energy are available
    if engine.eigenvalues_list and engine.e_fermi is not None:
        result = smart_select_bands(
            engine.eigenvectors_list,
            engine.S_k_list,
            engine.eigenvalues_list,
            e_fermi=engine.e_fermi,
            has_soc=has_soc,
            proj_threshold=args.proj_threshold,
            verbose=True,
        )

        if result.num_wann == 0:
            print("ERROR: Smart selector found no suitable bands!")
            print(f"  Projectability threshold: {args.proj_threshold}")
            print(f"  Try lowering --proj-threshold (e.g., 0.8)")
            sys.exit(1)

        # Use frontier detection for disentanglement
        if user_num_wann is not None:
            # User override: force specific num_wann
            engine.num_wann = user_num_wann
            print(f"\n  User override: num_wann = {user_num_wann}")
        else:
            engine.num_wann = result.recommended_num_wann

        engine.selected_band_indices = result.selected_band_indices

        # Store disentanglement info on engine for win_file generation
        engine._dis_win = result.recommended_dis_win
        engine._dis_froz = result.recommended_dis_froz

        # If num_wann == num selected bands but dis_win suggests more bands,
        # expand selected_band_indices to include the extra bands in the
        # disentanglement window. This ensures num_bands > num_wann.
        num_selected = len(result.selected_band_indices)
        if (engine.num_wann >= num_selected
                and result.recommended_dis_win is not None):
            # Count bands in the recommended dis_win at each k-point
            ef = engine.e_fermi if engine.e_fermi is not None else 0.0
            win_min = ef + result.recommended_dis_win[0]
            win_max = ef + result.recommended_dis_win[1]

            # Find minimum consistent band count across all k-points
            min_bands = None
            for evals in engine.eigenvalues_list:
                count = np.sum((evals >= win_min) & (evals <= win_max))
                if min_bands is None or count < min_bands:
                    min_bands = count

            if min_bands > engine.num_wann:
                # Expand band selection to include extra bands for disentanglement
                # Find the band indices that are in the dis_win at Gamma
                evals_gamma = engine.eigenvalues_list[0]
                in_window = np.where(
                    (evals_gamma >= win_min) & (evals_gamma <= win_max)
                )[0]
                engine.selected_band_indices = in_window[:min_bands]
                engine._num_bands_for_win = min_bands
                print(f"\n  Expanded band selection for disentanglement:")
                print(f"    dis_win captures {min_bands} bands (min across k-points)")
                print(f"    Selected bands: {engine.selected_band_indices}")
            else:
                engine._num_bands_for_win = num_selected
        else:
            engine._num_bands_for_win = num_selected

        if result.recommended_dis_win is not None and engine._num_bands_for_win > engine.num_wann:
            print(f"\nDisentanglement setup:")
            print(f"  num_wann (target):   {engine.num_wann}")
            print(f"  num_bands (total):   {engine._num_bands_for_win}")
            print(f"  dis_win  (rel E_F):  [{result.recommended_dis_win[0]:.4f}, {result.recommended_dis_win[1]:.4f}] eV")
            if result.recommended_dis_froz is not None:
                print(f"  dis_froz (rel E_F):  [{result.recommended_dis_froz[0]:.4f}, {result.recommended_dis_froz[1]:.4f}] eV")
        elif result.recommended_dis_win is not None:
            print(f"\nDisentanglement setup:")
            print(f"  num_wann (frontier): {engine.num_wann}")
            print(f"  num_bands (total):   {engine._num_bands_for_win}")
            print(f"  dis_win  (rel E_F):  [{result.recommended_dis_win[0]:.4f}, {result.recommended_dis_win[1]:.4f}] eV")
            if result.recommended_dis_froz is not None:
                print(f"  dis_froz (rel E_F):  [{result.recommended_dis_froz[0]:.4f}, {result.recommended_dis_froz[1]:.4f}] eV")
        else:
            print(f"\nSelected {engine.num_wann} bands, no disentanglement needed")
        print(f"Quality score: {result.quality_score:.4f}")

    else:
        # Fallback to simple projectability if no eigenvalues/Fermi available
        result = select_bands_by_projectability(
            engine.eigenvectors_list,
            engine.S_k_list,
            threshold=args.proj_threshold,
            verbose=True,
        )

        if result.num_wann == 0:
            print("ERROR: No bands have projectability above threshold!")
            print(f"  Threshold: {args.proj_threshold}")
            print(f"  Try lowering --proj-threshold (e.g., 0.8)")
            sys.exit(1)

        engine.num_wann = result.num_wann
        engine.selected_band_indices = result.selected_band_indices
        print(f"\nSelected {result.num_wann} bands for Wannier functions")

    _apply_spread_window_assist(engine, args)

    # Select projection orbitals
    proj_method = getattr(args, 'projection_method', 'weight')
    print(f"\nSelecting projection orbitals (method={proj_method})...")
    print("-" * 80)
    engine.select_projections(verbose=True, method=proj_method)


def _apply_spread_window_assist(engine, args):
    """Refine dis_froz / dis_win / num_wann to minimize the Wannier spread, using
    projectability as the localizability proxy and honoring the user minimum
    frozen window. No-op unless --spread-window-assist is set. Must run AFTER band
    selection sets engine._dis_froz/_dis_win/num_wann and BEFORE select_projections
    (so the projection count matches the possibly-updated num_wann).
    """
    if not getattr(args, 'spread_window_assist', False):
        return
    if not engine.eigenvalues_list:
        return
    if getattr(args, 'window_mode', 'manifold') == 'spread':
        return _apply_spread_window_assist_legacy(engine, args)
    try:
        from .window_assist import manifold_windows
    except Exception as exc:
        print(f"  ⚠ spread-window-assist unavailable: {exc}")
        return

    ef = engine.e_fermi if engine.e_fermi is not None else 0.0
    eig_all_rel = np.array([np.asarray(e) - ef for e in engine.eigenvalues_list])
    user_window = tuple(getattr(args, 'min_froz_window', (-6.0, 3.0)))

    res = manifold_windows(eig_all_rel, user_window)
    if res is None or len(res['band_indices']) == 0:
        print(f"  [spread-window-assist] no bands in window "
              f"[{user_window[0]:.2f}, {user_window[1]:.2f}] eV; selection unchanged")
        return

    nw = res['num_wann']
    froz = res['dis_froz']
    win = res['dis_win']
    idx = res['band_indices']
    # The user window is a REGION OF INTEREST. num_wann = the bands passing
    # through it; dis_froz grows to the full ISOLATED extent of those bands;
    # dis_win spans the bands that connect to them; num_bands = the pool.
    print(f"\n  [spread-window-assist] band-structure target from window "
          f"[{user_window[0]:.2f}, {user_window[1]:.2f}] eV:")
    print(f"    {nw} bands pass through the window -> num_wann = {nw}")
    print(f"    dis_froz: [{froz[0]:.3f}, {froz[1]:.3f}] eV (full isolated extent "
          f"of the near-E_F bands)")
    print(f"    dis_win:  [{win[0]:.3f}, {win[1]:.3f}] eV (connecting bands; "
          f"{len(idx)} bands in pool)")

    engine.selected_band_indices = np.asarray(idx, dtype=int)
    engine.num_wann = int(nw)
    engine._num_bands_for_win = int(len(idx))
    engine._dis_froz = (float(froz[0]), float(froz[1]))
    engine._dis_win = None if len(idx) == nw else (float(win[0]), float(win[1]))


def _apply_spread_window_assist_legacy(engine, args):
    """'spread' window mode: keep the method's num_wann, grow the outer window to
    >= num_wann bands/k (preferring the valence side) using projectability as the
    localizability proxy. See --window-mode."""
    if engine.selected_band_indices is None:
        return
    try:
        from .projectability import compute_band_projectability
        from .window_assist import spread_minimizing_windows
    except Exception as exc:
        print(f"  ⚠ spread-window-assist unavailable: {exc}")
        return

    ef = engine.e_fermi if engine.e_fermi is not None else 0.0
    bands = np.asarray(engine.selected_band_indices)
    eig_sel_rel = np.array([np.asarray(e)[bands] - ef for e in engine.eigenvalues_list])
    proj_pk = getattr(engine, '_band_projectability', None)
    if proj_pk is None:
        proj_pk = compute_band_projectability(engine.eigenvectors_list,
                                              engine.S_k_list)
    proj_pk = np.asarray(proj_pk)
    proj_avg = np.mean(proj_pk, axis=0) if proj_pk.ndim == 2 else proj_pk
    proj_sel = proj_avg[bands]

    min_froz = tuple(getattr(args, 'min_froz_window', (-6.0, 3.0)))
    floor = getattr(args, 'assist_proj_floor', None)
    if floor is None:
        floor = getattr(args, 'proj_threshold', 0.9)

    res = spread_minimizing_windows(eig_sel_rel, proj_sel, engine.num_wann,
                                    min_froz, proj_floor=floor)
    print("\n  [spread-window-assist:spread] minimizing spread (projectability "
          f"proxy; min frozen [{min_froz[0]:.2f}, {min_froz[1]:.2f}] eV):")
    if engine._dis_froz is not None:
        print(f"    dis_froz: [{engine._dis_froz[0]:.3f}, {engine._dis_froz[1]:.3f}]"
              f" -> [{res.dis_froz[0]:.3f}, {res.dis_froz[1]:.3f}] eV")
    if engine._dis_win is not None:
        print(f"    dis_win:  [{engine._dis_win[0]:.3f}, {engine._dis_win[1]:.3f}]"
              f" -> [{res.dis_win[0]:.3f}, {res.dis_win[1]:.3f}] eV")
    if res.num_wann != engine.num_wann:
        print(f"    num_wann: {engine.num_wann} -> {res.num_wann}")
    for note in res.notes:
        print(f"    note: {note}")
    engine.num_wann = res.num_wann
    engine._dis_froz = res.dis_froz
    engine._dis_win = res.dis_win


def _insulator_frozen_max(
    eigenvalues, projectability, selected_indices, e_fermi, num_wann,
    num_electrons, has_soc, cur_froz_max_abs, cond_p_min=0.70, margin=0.3,
    verbose=True,
):
    """For an insulator, return a raised dis_froz_max (absolute eV) that also
    FREEZES the well-represented low conduction bands, or None to leave it as is.

    A valence-only frozen window gives an exact valence band model but leaves the
    conduction bands to disentanglement over the (necessarily wide) outer window,
    which reproduces them poorly. The low conduction bands right above the gap are
    the antibonding partners of the target orbitals, so they project well and CAN
    be frozen — but higher conduction bands acquire nearly-free-electron / diffuse
    character that the atom-centred target basis cannot represent (low
    projectability); freezing those would force an unrepresentable subspace and
    wreck localization. So we freeze the contiguous low conduction bands whose
    projectability stays >= cond_p_min, and stop where it drops:

        dis_froz_max = max_k(top of the last well-projected conduction band) + margin

    capped so the per-k frozen count never exceeds num_wann (the wannier90
    disentanglement requirement). Conduction bands overlap across the BZ, so there
    is no clean global gap to land in — the per-k count is the real constraint.

    Parameters
    ----------
    eigenvalues   : (nk, nbands) absolute eV (full solved set, incl. core).
    projectability: (nk, nbands) per-band projectability onto the target orbitals.
    selected_indices : indices of the bands written to the .eig.
    cur_froz_max_abs : current (valence-only) dis_froz_max, absolute eV.
    cond_p_min    : minimum projectability for a conduction band to be frozen.
    """
    if num_electrons is None or cur_froz_max_abs is None:
        return None
    spin_deg = 1 if has_soc else 2
    nb = eigenvalues.shape[1]
    n_occ = int(round(num_electrons / spin_deg))
    if n_occ < 1 or n_occ >= nb:
        return None
    # Insulator check: clean gap between highest occupied and lowest empty band.
    vbm = float(eigenvalues[:, n_occ - 1].max())
    cbm = float(eigenvalues[:, n_occ].min())
    if cbm - vbm < 0.2:                      # metal / semimetal — leave as is
        return None

    sel = np.asarray(selected_indices, dtype=int)
    if sel.size <= num_wann:                 # no disentanglement freedom anyway
        return None
    # Per-k count ceiling: highest cutoff keeping <= num_wann selected bands/k.
    sel_sorted = np.sort(eigenvalues[:, sel], axis=1)
    cap = float(np.min(sel_sorted[:, num_wann]))   # global min of (num_wann+1)-th

    avg_p = projectability.mean(axis=0)
    bot = eigenvalues.min(axis=0)
    top = eigenvalues.max(axis=0)
    # Conduction bands among the selected set, ordered by their lower edge.
    cond = sorted((b for b in sel if bot[b] > cbm - 0.5), key=lambda b: bot[b])
    froz_top = None
    for b in cond:
        if avg_p[b] < cond_p_min:            # NFE / diffuse — stop here
            break
        cand = top[b] + margin
        if cand - margin >= cap:             # would exceed num_wann frozen/k
            break
        froz_top = min(cand, cap - 0.05)
    if froz_top is None or froz_top <= cur_froz_max_abs + 0.1:
        return None
    if verbose:
        print(f"    [insulator] gap = {cbm - vbm:.2f} eV at E_F; raising "
              f"dis_froz_max {cur_froz_max_abs - e_fermi:+.2f} -> "
              f"{froz_top - e_fermi:+.2f} eV (rel E_F) to freeze the "
              f"well-projected low conduction (P >= {cond_p_min:.2f})")
    return froz_top


def gap_aware_windows(proj, eigenvalues, num_wann, e_fermi, cond_p_min=0.70):
    """Gap-aware two-window band selection for LCAO-PDWF (env-gated, opt-in).

    Selects the E_F-connected manifold of well-projected bands, then builds a
    frozen inner window that never encloses more than ``num_wann`` bands at any
    k-point and an outer (active) window with disentanglement freedom.

    Parameters
    ----------
    proj : (nk, nb) array
        Lowdin projectability onto the target AOs, values in [0, 1].
    eigenvalues : (nk, nb) array
        Band energies in eV (absolute, i.e. NOT shifted by E_F).
    num_wann : int
        Number of target Wannier functions.
    e_fermi : float
        Fermi energy in eV (absolute).
    cond_p_min : float
        Minimum average projectability for a band entirely above E_F to be
        eligible for freezing (matches ``_insulator_frozen_max``).

    Returns
    -------
    (frozen_idx, active_idx, (froz_min, froz_max), (win_min, win_max))
        Band-index arrays and window bounds in ABSOLUTE eV.
    """
    proj = np.asarray(proj, dtype=float)
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    nb = eigenvalues.shape[1]

    avgp = proj.mean(axis=0)                       # (nb,)
    avgE = eigenvalues.mean(axis=0) - e_fermi      # (nb,) relative to E_F

    # --- 1. Energy window: kill high-energy ghost states that project ~1.0 ---
    in_win = np.where((avgE >= -30.0) & (avgE <= 20.0))[0]
    if in_win.size == 0:
        in_win = np.arange(nb)

    # --- 2. Rank: top-num_wann bands by avg projectability within in_win ------
    n_seed = int(min(num_wann, in_win.size))
    order = in_win[np.argsort(avgp[in_win])[::-1]]   # descending avgp
    seed = np.sort(order[:n_seed])

    # --- 3. Manifold = the rank top-num_wann seed ----------------------------
    # The energy window (step 1) already rejects the high-energy ghost states, so
    # the seed is exactly the num_wann most-projectable *physical* bands. Keeping
    # the whole seed preserves every band the valence config asks for -- including
    # deep sigma-bonding (e.g. h-BN's -17 eV sigma-valence) that a hard-coded
    # [E_F-12, E_F+8] filter would clip. A gap-separated DEEP segment (e.g. an Sc
    # semicore) is NOT dropped here on purpose: excluding it lowers the Wannier
    # count, which is a deliberate num_wann decision, not a window heuristic. We
    # still compute the gap structure to report the deepest gap-separated segment
    # as a diagnostic (so a semicore is visible for that num_wann call).
    manifold = np.sort(seed)
    kmin = eigenvalues[:, seed].min(axis=0)
    kmax = eigenvalues[:, seed].max(axis=0)
    order_km = np.argsort(kmin)
    kmin_s, kmax_s = kmin[order_km], kmax[order_km]
    _deep_gap = 0.0
    for i in range(1, len(order_km)):
        _deep_gap = max(_deep_gap, kmin_s[i] - kmax_s[i - 1])

    man_min = float(eigenvalues[:, manifold].min())
    man_max = float(eigenvalues[:, manifold].max())

    # helper: max-over-k count of bands fully enclosed in [lo, hi]
    def _max_count(lo, hi):
        inside = (eigenvalues >= lo) & (eigenvalues <= hi)   # (nk, nb)
        return int(inside.sum(axis=1).max())

    # --- 4. Frozen window -----------------------------------------------------
    froz_min = man_min - 0.5
    # Projectability cap: never freeze poorly-projected conduction. The count
    # rule alone raises froz_max as high as "<= num_wann bands enclosed at any
    # k" permits, which freezes high-energy antibonding bands with weak AO
    # character and inflates Omega_OD (h-BN: Omega_OD ~45, wannierise stalls).
    # Match the standard path (_insulator_frozen_max): a band lying entirely
    # above E_F may only be frozen if it projects well onto the target AOs.
    # Bands touching or crossing E_F are never capped -- the Fermi surface
    # must stay frozen.
    bot = eigenvalues.min(axis=0)                  # (nb,) lower band edges
    poor = [b for b in range(nb)
            if avgp[b] < cond_p_min and bot[b] > e_fermi]
    proj_cap = min((float(bot[b]) for b in poor), default=np.inf) - 0.05
    proj_cap = max(proj_cap, e_fermi)

    # candidate upper energies: band energies inside [man_min, man_max]
    cand_e = np.unique(eigenvalues[(eigenvalues >= man_min) &
                                   (eigenvalues <= man_max)])
    cand_e = np.sort(cand_e)
    e_cap = None
    for e in cand_e:
        if e > proj_cap:
            break
        if _max_count(man_min, e) <= num_wann:
            e_cap = e
        else:
            break
    if e_cap is None:
        e_cap = man_min

    # froz_max = midpoint between e_cap and the next band energy above it, so
    # padding never pushes the count over num_wann (clamped to proj_cap so the
    # padding never reaches into a poorly-projected band either).
    all_e_sorted = np.unique(eigenvalues)
    above = all_e_sorted[all_e_sorted > e_cap]
    if above.size > 0:
        froz_max = min(0.5 * (e_cap + float(above[0])), proj_cap)
    else:
        froz_max = min(e_cap + 0.05, proj_cap)
    # guard: if the midpoint padding pushed the count over num_wann, retreat.
    if _max_count(froz_min, froz_max) > num_wann:
        froz_max = e_cap

    # frozen_idx = bands with per-band-avg energy inside [froz_min, froz_max]
    avgE_abs = eigenvalues.mean(axis=0)
    frozen_idx = np.where((avgE_abs >= froz_min) & (avgE_abs <= froz_max))[0]

    # --- 5. Active (outer) window: ALWAYS leave disentanglement freedom -------
    # Long-standing PDWF invariant (see the non-gap-aware path): num_bands >=
    # 1.5*num_wann and STRICTLY > num_wann, so wannier90 always has room to
    # disentangle. The seed manifold can fully exhaust the [-30,+20] eV seed
    # window (e.g. h-BN's sp2 manifold), so the extra active bands are drawn
    # from ALL real bands nearest the manifold in energy -- not just the seed
    # window. Ranking nearest-first with a ghost-gap guard keeps the high-energy
    # diffuse-GTO ghost states (which sit >GHOST_GAP eV above any real manifold)
    # out of the disentanglement space.
    min_active = min(max(int(np.ceil(1.5 * num_wann)), num_wann + 1), nb)
    GHOST_GAP = 40.0
    active = set(int(b) for b in manifold)
    man_ctr = 0.5 * (man_min + man_max)
    others = sorted((b for b in range(nb) if int(b) not in active),
                    key=lambda b: abs(float(eigenvalues[:, b].mean()) - man_ctr))
    for b in others:
        if len(active) >= min_active:
            break
        e = float(eigenvalues[:, b].mean())
        if e > man_max + GHOST_GAP or e < man_min - GHOST_GAP:
            continue                       # skip high-energy ghost states
        active.add(int(b))
    # Fallback: if the ghost guard starved us below the strict invariant (all
    # spare bands are ghosts -- pathological), take nearest real bands anyway so
    # disentanglement is never disabled when the spectrum could support it.
    if len(active) <= num_wann and nb > num_wann:
        for b in others:
            if len(active) > num_wann:
                break
            active.add(int(b))
    active_idx = np.array(sorted(active))

    win_min = float(eigenvalues[:, active_idx].min()) - 2.0
    win_max = float(eigenvalues[:, active_idx].max()) + 2.0

    # --- Verify invariants ----------------------------------------------------
    fc = _max_count(froz_min, froz_max)
    assert fc <= num_wann, (
        f"gap_aware_windows: frozen window encloses {fc} > num_wann="
        f"{num_wann} bands (froz=[{froz_min:.3f},{froz_max:.3f}])")
    assert nb <= num_wann or len(active_idx) > num_wann, (
        f"gap_aware_windows: active set {len(active_idx)} !> num_wann={num_wann} "
        f"-- disentanglement requires num_bands > num_wann")

    return frozen_idx, active_idx, (froz_min, froz_max), (win_min, win_max)


def _apply_hybrid_overrides(engine):
    """Post-PDWF overrides for --method hybrid (in-pipeline disentanglement).

    The disentanglement happens in our pipeline (lcao2wannier.hybrid_pipeline),
    so wannier90 gets an isolated manifold: num_bands == num_wann and NO dis_*
    keywords. The .win therefore must not carry windows; band selection is a
    placeholder (Stage 2 replaces the matrices with the rotated subspace).
    """
    import numpy as _np
    engine._dis_froz = None
    engine._dis_win = None
    engine._num_bands_for_win = engine.num_wann
    engine.selected_band_indices = _np.arange(engine.num_wann)
    engine._guiding_centres = False
    print(f"\n  [hybrid] wannierise-only hand-off: num_bands = num_wann = "
          f"{engine.num_wann}, no disentanglement windows in .win")


def _apply_baseline_overrides(engine, args):
    """Post-PDWF overrides for --baseline dis-froz-proj (stock-W90 baseline).

    Baseline for comparing stock Wannier90 projectability disentanglement
    (dis_froz_proj / dis_proj_min / dis_proj_max, introduced with PDWF;
    Qiao et al. 2023) against the hybrid method on identical inputs
    (Referee_Risk_Assessment A1). The .win gets the FULL hybrid band pool
    (num_bands = pool size, same contiguous pool and cap the hybrid method
    uses), the dis_froz_proj keywords, and NO dis_froz_min/max energy
    windows unless --window is explicit; stage 2 then emits the
    uncompressed pool .eig/.amn/.mmn (hybrid_pipeline.
    run_baseline_dis_froz_proj) instead of the pre-disentangled hand-off.
    """
    import numpy as _np
    from .hybrid_pipeline import select_hybrid_parent_pool
    num_wann = engine.num_wann
    n_target = int(_np.sum(engine.pdwf_target_mask))
    variant = getattr(args, 'baseline', 'dis-froz-proj')
    if variant == 'dis-froz-proj' and n_target != num_wann:
        print(f"\nERROR (--baseline dis-froz-proj): wannier90 requires "
              f"num_proj == num_wann, but the target mask has {n_target} "
              f"AOs vs num_wann = {num_wann}.")
        print("  Multi-zeta targets (the default: all radials per valence "
              "l-channel) and augmented channels cannot be written as a "
              "stock-wannier90 .amn.")
        print("  Re-run with --pdwf-first-radial-only (one radial per "
              "valence l-channel -> num_proj == num_wann exactly), or use "
              "--baseline dis-froz-proj-svd (SVD-compressed multi-zeta amn).")
        sys.exit(1)
    if variant == 'dis-froz-proj-svd' and n_target < num_wann:
        print(f"\nERROR (--baseline dis-froz-proj-svd): num_target "
              f"({n_target}) < num_wann ({num_wann}).")
        sys.exit(1)
    pool = select_hybrid_parent_pool(
        engine, pool_factor=getattr(args, 'hybrid_pool_cap', 1.5))
    engine.selected_band_indices = pool
    engine._num_bands_for_win = len(pool)
    engine._dis_froz = None
    engine._dis_win = tuple(args.window) if args.window else None
    engine._additional_keywords = {
        'dis_froz_proj': True,
        'dis_proj_min': getattr(args, 'dis_proj_min', 0.01),
        'dis_proj_max': getattr(args, 'dis_proj_max', 0.95),
    }
    print(f"\n  [baseline {variant}] stock-wannier90 projectability "
          f"disentanglement: num_bands = {len(pool)} (pool "
          f"[{pool[0]}..{pool[-1]}]), num_wann = {num_wann}, "
          f"dis_proj_min/max = "
          f"{engine._additional_keywords['dis_proj_min']:g}/"
          f"{engine._additional_keywords['dis_proj_max']:g}"
          + ("" if args.window else ", no energy windows"))


def _apply_method_pdwf(engine, args, has_soc, lines):
    """Apply LCAO-PDWF method: chemistry-grounded projectability band selection."""
    print("\nLCAO-PDWF band selection...")
    print("-" * 80)

    extended = getattr(args, 'extended', False)
    include_tm_p = getattr(args, 'include_tm_p', False)
    p_high = getattr(args, 'pdwf_p_high', 0.90)
    p_low = getattr(args, 'pdwf_p_low', 0.10)

    # Phase 1: Parse basis shells
    print("  Phase 1: Parsing basis set shells...")
    try:
        from .parser import parse_calculation_parameters
        params_calc = parse_calculation_parameters(lines)
        num_atoms_hint = params_calc.num_atoms
    except Exception:
        num_atoms_hint = None

    shells, num_ao_spatial = parse_basis_shells(lines, num_atoms=num_atoms_hint)
    atoms = get_atom_list(shells)
    print(f"    Found {len(shells)} shells, {num_ao_spatial} spatial AOs")
    print(f"    Atoms: {atoms}")

    # Phase 2: Build target mask
    print("  Phase 2: Building target orbital mask...")
    # CRYSTAL emits multi-zeta LCAO bases where valence character is spread
    # across several radial functions, and the first valence radial can be a
    # semicore function. Target ALL radials of the valence l-channels so the
    # near-E_F bands project correctly (the energy cutoff drops deep semicore).
    all_radials = not getattr(args, 'pdwf_first_radial_only', False)
    if getattr(args, 'baseline', None) == 'dis-froz-proj' and all_radials:
        # stock wannier90 requires num_proj == num_wann, so only the
        # first-radial target (one AO per valence orbital) is emittable —
        # the baseline mode implies it.
        print("  [baseline dis-froz-proj] multi-zeta targets cannot be "
              "written as a stock-wannier90 .amn (num_proj must equal "
              "num_wann) — using the first-radial target "
              "(--pdwf-first-radial-only implied)")
        all_radials = False
    target_mask = build_target_mask(
        shells, extended=extended, include_tm_p=include_tm_p,
        has_soc=has_soc, verbose=True, all_radials=all_radials,
    )
    num_wann = compute_num_wann(
        atoms, extended=extended, include_tm_p=include_tm_p,
        has_soc=has_soc,
    )
    print(f"\n    Target AOs: {int(np.sum(target_mask))} / {len(target_mask)}")
    print(f"    num_wann: {num_wann}")
    print(summarize_config(atoms, extended=extended,
                           include_tm_p=include_tm_p, has_soc=has_soc))

    # Align target_mask to engine orbital dimension.
    # parse_basis_shells may count more AOs than the H/S matrices contain
    # (e.g. 336 vs 296 spatial AOs in CrI3), causing a mismatch after SOC
    # doubling (672 vs 592). Truncate the spatial part to engine.num_orbitals//2.
    eng_dim = engine.num_orbitals
    mask_len_orig = len(target_mask)
    if mask_len_orig != eng_dim:
        if has_soc:
            spatial_engine = eng_dim // 2
            half_mask = mask_len_orig // 2
            mask_spatial = target_mask[:half_mask][:spatial_engine]
            target_mask = np.concatenate([mask_spatial, mask_spatial])
        else:
            target_mask = target_mask[:eng_dim]
        print(f"    [Note] target_mask trimmed {mask_len_orig}->{eng_dim} "
              f"(basis parser: {num_ao_spatial} spatial AOs, "
              f"engine: {eng_dim // (2 if has_soc else 1)})")

    # Phase 3: Compute Lowdin projectability
    print("\n  Phase 3: Computing Lowdin projectability...")
    eigenvalues = np.array(engine.eigenvalues_list)
    proj = compute_lowdin_projectability(
        engine.eigenvectors_list, engine.S_k_list, target_mask,
    )
    print(f"    Projectability range: [{np.min(proj):.4f}, {np.max(proj):.4f}]")
    # Expose Lowdin (target) projectability for the spread-window-assist so it
    # excludes diffuse bands by their REAL atomic character, not by projection
    # onto the full basis (which over-counts high-energy states).
    engine._band_projectability = np.asarray(proj)

    # Conduction-driven target augmentation (hybrid method, v2): the target
    # must include the orbital types the LOW CONDUCTION is made of — SnTe's
    # sp valence config left the gap-edge conduction at min_k P = 0 (d
    # character at the inversion pockets), an unfixable 1.5 eV variational
    # wall (channel study: calculations/SnTe/HYBRID_TARGET_STUDY.md).
    # Channel SETS are scored (superadditive: Sn-d alone useless, Te-d 5x,
    # both 33x) and the WF count is DECOUPLED from the target set: only
    # physical semicore bands raise num_wann; polarization channels enlarge
    # the mask only (h-BN + polarization-d at num_wann=36 collapsed,
    # Omega_I 269 — the diffuse continuum projects P~1.0 onto diffuse d).
    # the conduction-driven augmentation serves both automatic routes: the
    # window route classifies bands by the same projectability and needs
    # the conduction's orbital character in its target just as much
    if (args.method == 'hybrid' or getattr(args, 'window_route', False)) and not extended:
        _ef = engine.e_fermi if engine.e_fermi is not None else 0.0
        _avge = eigenvalues.mean(axis=0) - _ef
        _avgp = np.asarray(proj).mean(axis=0)
        _cond = np.where((_avge > 0.0) & (_avge < 8.0))[0]
        _cov = None
        if _cond.size:
            _low = _cond[np.argsort(_avge[_cond])[:4]]
            _cov = float(_avgp[_low].mean())
        _cov_min = getattr(args, 'augment_coverage_min', 0.70)
        # CAPACITY DEFICIT is a second, independent trigger. The coverage
        # probe above is semiconductor-shaped -- it asks whether the lowest
        # CONDUCTION bands are represented -- and in a metal there is no
        # conduction/valence split, so it passes even when the target is far
        # too small. Measured: every elemental 3d metal (Fe, V, Cr, Ni) took
        # the s+d = 6 target, coverage never fired, and the run died later on
        # a rank-deficient gauge (cond ~1e15) with the capacity gate warning
        # 'frozen demand 8 vs num_wann 6'. That demand IS the signal: if the
        # mask rules want more frozen states than the target can hold, the
        # channel set is too small, whatever the coverage says. Uses the SAME
        # select_pool_and_frozen/auto_trust_thresholds the real run uses, so
        # the pre-check cannot drift from the mask rule it is predicting.
        _cap_deficit = False
        try:
            from .hybrid import (auto_trust_thresholds,
                                             select_pool_and_frozen)
            _pf = getattr(args, 'hybrid_p_froz', None)
            _pb = getattr(args, 'hybrid_p_froz_band', None)
            if _pf is None or _pb is None:
                _auto = auto_trust_thresholds(
                    np.asarray(proj), eigenvalues, num_wann, e_fermi=_ef,
                    fid_emax=getattr(args, 'hybrid_fid_emax', 8.0),
                    margin=getattr(args, 'hybrid_trust_margin', 0.015),
                    thr_floor=getattr(args, 'hybrid_thr_floor', 0.55))
                _pf = _pf if _pf is not None else _auto['p_froz']
                _pb = _pb if _pb is not None else _auto['p_froz_band']
            _cap = {}
            select_pool_and_frozen(
                np.asarray(proj), eigenvalues, num_wann, p_froz=_pf,
                p_froz_band=_pb,
                p_floor=getattr(args, 'hybrid_p_floor', 0.10),
                pool_factor=getattr(args, 'hybrid_pool_cap', 1.5),
                e_fermi=_ef, capacity=_cap)
            _cap_deficit = _cap.get('headroom', 0) < 0
            if _cap_deficit:
                print(f"\n  [hybrid] capacity deficit BEFORE the run: the "
                      f"mask rules demand up to {_cap['need']} frozen states "
                      f"but the target holds only {num_wann} — the channel "
                      f"set is too small, so augmenting it.")
        except Exception as _e:            # never let the pre-check break a run
            print(f"  [hybrid] capacity pre-check skipped ({_e})")
        if ((_cov is not None and _cov < _cov_min) or _cap_deficit
                or os.environ.get('PDWF_AUG_FORCE')):
            if _cov is not None:
                print(f"\n  [hybrid] low-conduction coverage <P> = {_cov:.3f} "
                      f"< {_cov_min:.2f} with the standard valence config — "
                      f"the target "
                      f"is missing the conduction's orbital character.")
            from .channel_augment import augment_target_channels
            aug = augment_target_channels(
                shells, target_mask, proj, eigenvalues, _ef,
                engine.eigenvectors_list, engine.S_k_list, has_soc,
                retention_floor=getattr(args, 'augment_retention_floor', 0.40),
            )
            if aug is not None:
                target_mask = aug.mask
                num_wann = num_wann + aug.n_semicore
                proj = aug.proj
                engine._band_projectability = np.asarray(proj)
                engine._augmented_channels = [
                    (c.atom_index, c.l, c.n_semicore) for c in aug.selected]
                print(f"  [hybrid] augmented target: "
                      f"{int(np.sum(target_mask))} AOs, "
                      f"num_wann -> {num_wann}")

    # --method hybrid (and its --baseline emission modes): band selection is
    # per-state — the in-pipeline masks (hybrid) or wannier90's
    # dis_froz_proj (baseline) — so the PDWF window classification below is
    # meaningless for it: its windows are discarded by
    # _apply_hybrid_overrides/_apply_baseline_overrides, and the classifier
    # can abort outright on low-projectability targets (first-radial SnTe:
    # nothing classifies, sys.exit) that the per-state machinery handles.
    # Skip straight to the engine fields those overrides need. (The
    # env-gated PDWF_DUMP window-era diagnostic does not fire on this path;
    # --dump-masks / --hybrid-report-json are its hybrid-era replacements.)
    if args.method == 'hybrid':
        engine.num_wann = num_wann
        engine.pdwf_target_mask = target_mask
        engine.selected_orbital_indices = None
        if engine._override_num_iter is None:
            engine._override_num_iter = 10000
        if engine._override_dis_num_iter is None:
            engine._override_dis_num_iter = 5000
        engine._guiding_centres = True
        if getattr(args, 'baseline', None):
            print(f"\n  [baseline {args.baseline}] skipping PDWF window "
                  "classification — wannier90 disentangles by projectability")
        else:
            print("\n  [hybrid] skipping PDWF window classification — "
                  "selection is per-state (windows are not used)")
        return

    # Phase 4: Classify bands
    print("\n  Phase 4: Classifying bands...")
    e_fermi = engine.e_fermi if engine.e_fermi is not None else 0.0
    if p_high == 'auto' or p_low == 'auto':
        from .lcao_pdwf import auto_classification_params
        _params, _info = auto_classification_params(
            proj, eigenvalues, num_wann, e_fermi=e_fermi,
            fid_emax=getattr(args, 'hybrid_fid_emax', 8.0),
            margin=getattr(args, 'hybrid_trust_margin', 0.015),
            thr_floor=getattr(args, 'hybrid_thr_floor', 0.55))
        if _params is None:
            print("  [pdwf auto] the projectability distribution has no gray "
                  "plateau to read (no band entirely above the fidelity "
                  "ceiling, or nothing above the gate); using the fixed "
                  "thresholds 0.95/0.10")
            p_high = 0.95 if p_high == 'auto' else p_high
            p_low = 0.10 if p_low == 'auto' else p_low
        else:
            p_high = _params.p_high if p_high == 'auto' else p_high
            p_low = _params.p_low if p_low == 'auto' else p_low
            print(f"  [pdwf auto] derived thresholds: p_high = {p_high:.3f} "
                  f"(gray plateau {_info['plateau']:.3f} + margin), "
                  f"p_low = {p_low:.3f} "
                  f"({'junk cliff' if _info['junk_cliff'] else 'half the gate'}); "
                  f"{_info['n_trusted']} trusted of {_info['n_pool']} pool bands")
    classification = classify_bands(
        proj, eigenvalues, num_wann,
        ClassificationParams(p_high=p_high, p_low=p_low, e_fermi=e_fermi),
    )

    # Phase 5: Determine windows
    print("\n  Phase 5: Determining disentanglement windows...")
    windows = determine_windows(classification, eigenvalues, e_fermi)

    # [diagnostic, env-gated] dump the projectability profile + DFT k-path bands for
    # window-algorithm analysis. No effect unless PDWF_DUMP is set.
    import os as _os
    if _os.environ.get('PDWF_DUMP'):
        _band_kw = {}
        try:
            from .band_plot import (detect_lattice_type,
                get_kpath_for_lattice, generate_kpath, compute_band_structure,
                compute_path_projectability)
            _lat = engine.lattice_vectors
            _kp = generate_kpath(
                get_kpath_for_lattice(detect_lattice_type(_lat), npts=50), _lat)
            _bnd, _ev, _sk = compute_band_structure(
                engine.real_space_matrices, _lat, _kp)
            _band_kw = dict(band=_bnd, band_dist=_kp.distances,
                            tick_pos=np.asarray(_kp.tick_positions, float),
                            tick_lab=np.asarray(_kp.tick_labels),
                            band_proj=compute_path_projectability(_ev, _sk, target_mask))
        except Exception as _e:
            print(f"  [PDWF_DUMP] band structure skipped: {_e}")
        _dump = _os.environ['PDWF_DUMP']
        # spin-polarized runs process both channels in one invocation --
        # suffix the dump per channel so beta doesn't overwrite alpha
        _sl = getattr(args, '_spin_label', None)
        if _sl:
            _root, _ext = _os.path.splitext(_dump)
            _dump = f"{_root}_{_sl}{_ext or '.npz'}"
        np.savez(_dump, proj=proj, eig=eigenvalues,
                 e_fermi=float(e_fermi), num_wann=int(num_wann),
                 category=classification.category, **_band_kw)
        print(f"  [PDWF_DUMP] wrote profile + bands -> {_dump}")

    # Validate
    all_warnings = []
    if windows.dis_froz_min is not None:
        all_warnings += check_frozen_interlopers(
            eigenvalues, classification.frozen_indices,
            classification.excluded_indices,
            windows.dis_froz_min, windows.dis_froz_max,
        )
    if windows.dis_win_min is not None:
        all_warnings += check_band_count(
            eigenvalues, windows.dis_win_min, windows.dis_win_max, num_wann,
        )

    # Print summary
    print_pdwf_summary(classification, windows, eigenvalues,
                       e_fermi, all_warnings)

    # Handle poor projectability gap gracefully
    if classification.gap_quality < 2.0:
        print(f"\n  WARNING: Poor projectability gap (quality = "
              f"{classification.gap_quality:.2f}).")
        if len(classification.frozen_indices) == 0:
            # No frozen bands found — use energy-range-based freezing.
            # Freeze all bands with reasonable projectability below E_F + margin
            froz_margin_above = 3.0  # eV above E_F
            print(f"  No frozen bands from projectability. Using energy-range "
                  f"freezing up to E_F + {froz_margin_above:.1f} eV.")
            # Promote bands with avg_p >= p_low that are mostly below threshold
            avg_e = classification.band_energies
            avg_p = classification.avg_projectability
            for b in range(len(avg_p)):
                if (avg_p[b] >= p_low and
                    avg_e[b] <= e_fermi + froz_margin_above and
                    classification.category[b] != 'excluded'):
                    classification.category[b] = 'frozen'
            classification.frozen_indices = np.where(
                classification.category == 'frozen')[0]
            classification.disent_indices = np.where(
                classification.category == 'disent')[0]
            # Recompute windows with updated classification
            windows = determine_windows(classification, eigenvalues, e_fermi)
            print(f"  Energy-range frozen bands: {len(classification.frozen_indices)}")

    # Apply results to engine
    engine.num_wann = num_wann

    # Selected bands = frozen + disentangle
    all_active = np.union1d(classification.frozen_indices,
                            classification.disent_indices)
    if len(all_active) == 0:
        print("ERROR: No bands classified as frozen or disentangle!")
        sys.exit(1)

    # Ensure band ratio >= 1.5 for adequate disentanglement freedom
    min_ratio = 1.5
    min_bands = max(int(np.ceil(num_wann * min_ratio)), num_wann + 1)
    if len(all_active) < min_bands:
        nb = eigenvalues.shape[1]
        # Find outer window bounds from current active set
        if windows.dis_win_min is not None:
            win_lo = windows.dis_win_min
            win_hi = windows.dis_win_max
        else:
            active_eigs = eigenvalues[:, all_active]
            win_lo = float(np.min(active_eigs)) - 2.0
            win_hi = float(np.max(active_eigs)) + 2.0

        # Expand outer window to include more bands above
        candidates = []
        for b in range(nb):
            if b in set(all_active):
                continue
            band_eigs = eigenvalues[:, b]
            # Include bands that overlap with expanded outer window
            if np.any((band_eigs >= win_lo) & (band_eigs <= win_hi + 10.0)):
                avg_e_b = np.mean(band_eigs)
                candidates.append((avg_e_b, b))

        # Sort candidates by energy (prefer bands near the active set)
        candidates.sort(key=lambda x: x[0])
        for _, b in candidates:
            all_active = np.union1d(all_active, [b])
            if len(all_active) >= min_bands:
                break

        # Update outer window to encompass all active bands
        active_eigs = eigenvalues[:, all_active]
        windows.dis_win_min = float(np.min(active_eigs)) - 2.0
        windows.dis_win_max = float(np.max(active_eigs)) + 2.0

        ratio = len(all_active) / num_wann
        print(f"\n  Expanded band set to {len(all_active)} bands "
              f"(ratio {ratio:.2f}) for adequate disentanglement")

    engine.selected_band_indices = all_active

    # Store disentanglement windows (relative to E_F for win_file)
    if windows.dis_win_min is not None and len(all_active) > num_wann:
        engine._dis_win = (windows.dis_win_min - e_fermi,
                           windows.dis_win_max - e_fermi)
    else:
        engine._dis_win = None

    if windows.dis_froz_min is not None:
        engine._dis_froz = (windows.dis_froz_min - e_fermi,
                            windows.dis_froz_max - e_fermi)
    else:
        engine._dis_froz = None

    # For insulators, raise the frozen-window top through the gap so the low
    # conduction manifold is FROZEN too (a valence-only window leaves conduction
    # to disentanglement, which reproduces it poorly). Opt out: --frozen-conduction off.
    if (getattr(args, 'frozen_conduction', 'auto') != 'off'
            and engine._dis_froz is not None and windows.dis_froz_min is not None):
        new_froz_max_abs = _insulator_frozen_max(
            eigenvalues, np.asarray(proj), all_active, e_fermi, num_wann,
            getattr(params_calc, 'num_electrons', None), has_soc,
            windows.dis_froz_max,
            cond_p_min=getattr(args, 'conduction_pmin', 0.70),
        )
        if new_froz_max_abs is not None:
            windows.dis_froz_max = new_froz_max_abs
            engine._dis_froz = (windows.dis_froz_min - e_fermi,
                                new_froz_max_abs - e_fermi)

    # WINDOW ROUTE FROZEN CEILING. The automatic window route freezes the same
    # set the hybrid route freezes: the trust measurement (auto-trust
    # thresholds, gate, shell) and the frozen-ceiling walk of run_hybrid,
    # expressed as dis_froz_max. Every band with a state at or below the
    # ceiling is added to the exported band set so the frozen window is
    # complete; the Omega_I auto-window step then chooses only the outer
    # window (froz_max fixed). --hybrid-froz-emax none keeps the legacy
    # classification hull.
    if (getattr(args, 'window_route', False) and windows.dis_froz_min is not None
            and str(getattr(args, 'hybrid_froz_emax', 'auto')).lower() != 'none'):
        from .hybrid import (auto_trust_thresholds,
                                         select_pool_and_frozen,
                                         degeneracy_regime,
                                         symmetrize_degenerate)
        from .hybrid_pipeline import resolve_frozen_ceiling
        _fid = getattr(args, 'hybrid_fid_emax', 8.0)
        _pw = np.asarray(proj, float)
        _reg = degeneracy_regime(eigenvalues)
        if _reg['band_level']:
            _pw = symmetrize_degenerate(_pw, eigenvalues, dtol=_reg['dtol'])
        _auto = auto_trust_thresholds(
            _pw, eigenvalues, num_wann, e_fermi=e_fermi, fid_emax=_fid,
            margin=getattr(args, 'hybrid_trust_margin', 0.015),
            thr_floor=getattr(args, 'hybrid_thr_floor', 0.55))
        if _auto is None:
            print("  [window route] auto-trust undecidable: frozen ceiling not "
                  "derived; keeping the classification window")
        else:
            _pool, _adm, _trust = select_pool_and_frozen(
                _pw, eigenvalues, num_wann, p_froz=_auto['p_froz'],
                p_floor=0.10, pool_factor=1.5, p_froz_band=_auto['p_froz_band'],
                e_fermi=e_fermi, fermi_shell=(2.0, 3.0),
                shell_p_floor=getattr(args, 'hybrid_shell_p_floor', 'auto'))
            _ci = resolve_frozen_ceiling(
                np.asarray(eigenvalues, float)[:, _pool] - e_fermi, _trust,
                num_wann, mode=getattr(args, 'hybrid_froz_emax', 'auto'),
                fid_emax=_fid)
            _E_abs = e_fermi + _ci['E_ceil']
            _need = np.where((np.asarray(eigenvalues) <= _E_abs + 1e-9).any(axis=0)
                             & (np.asarray(eigenvalues) >= windows.dis_froz_min - 1e-9).any(axis=0))[0]
            _added = np.setdiff1d(_need, all_active)
            if _added.size:
                all_active = np.union1d(all_active, _added)
                engine.selected_band_indices = all_active
            windows.dis_froz_max = _E_abs
            if windows.dis_win_max is None or windows.dis_win_max < _E_abs + 2.0:
                windows.dis_win_max = _E_abs + 2.0
            engine._dis_froz = (windows.dis_froz_min - e_fermi, _ci['E_ceil'])
            engine._dis_win = (windows.dis_win_min - e_fermi,
                               windows.dis_win_max - e_fermi)
            engine._window_frozen_ceiling = _ci
            print(f"  [window route] frozen ceiling (same walk as the hybrid "
                  f"route, mode {_ci['mode']}): trusted-set top "
                  f"E_F{_ci['E_top']:+.3f} eV -> dis_froz_max = "
                  f"E_F{_ci['E_ceil']:+.3f} eV"
                  + (f"; {_added.size} band(s) added to the export"
                     if _added.size else ""))

    engine._num_bands_for_win = len(all_active)

    # Store full target mask for SVD-based PDWF Amn generation
    # (projects onto ALL target AOs, then SVD selects optimal num_wann subspace)
    engine.pdwf_target_mask = target_mask
    engine.selected_orbital_indices = None  # Not used with PDWF Amn

    # Optional spread-minimization window assist (refines _dis_froz/_dis_win/
    # num_wann; the SVD-based PDWF Amn adapts to the updated num_wann).
    _apply_spread_window_assist(engine, args)

    if windows.dis_win_min is not None and len(all_active) > num_wann:
        ratio = len(all_active) / num_wann
        print(f"\n  Disentanglement enabled:")
        print(f"    num_wann:  {num_wann}")
        print(f"    num_bands: {len(all_active)} (ratio {ratio:.2f})")
        if windows.dis_froz_min is not None:
            print(f"    frozen:    [{windows.dis_froz_min - e_fermi:+.1f}, "
                  f"{windows.dis_froz_max - e_fermi:+.1f}] eV (rel E_F)")
        print(f"    outer:     [{windows.dis_win_min - e_fermi:+.1f}, "
              f"{windows.dis_win_max - e_fermi:+.1f}] eV (rel E_F)")
    else:
        print(f"\n  No disentanglement needed: {num_wann} bands selected")

    # Set iteration counts for proper convergence.
    # Previous values (2000 / 200) regressed MgB2 PDWF from Ω_total = 8.09 Å²
    # to 44.28 Å² — the assumption that "SVD-based PDWF Amn converges fast"
    # is false in practice. Match projectability/symmetry defaults.
    if engine._override_num_iter is None:
        engine._override_num_iter = 10000
    if engine._override_dis_num_iter is None:
        engine._override_dis_num_iter = 5000
    # Guiding centres keeps the initial Wannier centres anchored, critical for
    # PDWF where the SVD-chosen subspace has weak initial localization.
    engine._guiding_centres = True

    # --- Gap-aware two-window override (opt-in, env-gated) --------------------
    # Runs after every other engine field is set so it takes precedence, and is
    # a strict no-op when GAP_AWARE_WINDOW is unset -> cannot affect normal runs.
    if os.environ.get('GAP_AWARE_WINDOW'):
        fidx, aidx, dfroz, dwin = gap_aware_windows(
            proj, eigenvalues, engine.num_wann, e_fermi,
            cond_p_min=getattr(args, 'conduction_pmin', 0.70))
        engine.selected_band_indices = np.asarray(sorted(aidx))
        engine._dis_froz = (dfroz[0] - e_fermi, dfroz[1] - e_fermi)
        engine._dis_win = ((dwin[0] - e_fermi, dwin[1] - e_fermi)
                           if len(aidx) > engine.num_wann else None)
        engine._num_bands_for_win = len(aidx)
        print(f"  [GAP_AWARE_WINDOW] froz rel E_F="
              f"[{dfroz[0] - e_fermi:+.1f},{dfroz[1] - e_fermi:+.1f}] "
              f"({len(fidx)} froz) | win rel E_F="
              f"[{dwin[0] - e_fermi:+.1f},{dwin[1] - e_fermi:+.1f}] "
              f"({len(aidx)} active)")


def _apply_method_direct(engine, args, has_soc, params):
    """Apply Method 2: Direct LCAO orbital mapping."""
    num_basis = engine.num_orbitals  # Already doubled for SOC

    print("\nDirect LCAO orbital mapping...")
    print("-" * 80)
    print(f"  num_basis (total orbitals): {num_basis}")
    if has_soc:
        print(f"  (includes SOC doubling: {params.num_ao} AOs x 2 = {num_basis} spinors)")

    # Safety check for large basis sets
    if num_basis > LARGE_BASIS_THRESHOLD:
        print(f"\n{'!'*70}")
        print(f"  CRITICAL WARNING: Large basis set detected!")
        print(f"  num_basis = {num_basis} exceeds threshold ({LARGE_BASIS_THRESHOLD})")
        print(f"  Wannier90 with {num_basis} Wannier functions will be very slow.")
        print(f"  Recommendation: Try --method projectability first.")
        print(f"{'!'*70}")

        if not args.force:
            try:
                response = input("\n  Continue anyway? [y/N]: ").strip().lower()
                if response != 'y':
                    print("  Aborted. Use --method projectability or --force to override.")
                    sys.exit(0)
            except EOFError:
                print("  Non-interactive mode: use --force to override. Aborting.")
                sys.exit(1)
        else:
            print("  --force flag set, proceeding...")

    # Set num_wann = num_basis (all orbitals)
    engine.num_wann = num_basis
    engine.selected_band_indices = np.arange(num_basis)

    # Select ALL orbitals as projections
    engine.selected_orbital_indices = np.arange(num_basis)

    # Skip spread minimization
    engine._override_num_iter = 0

    print(f"\n  num_wann = {num_basis} (all LCAO orbitals)")
    print(f"  num_iter will be set to 0 (skip spread minimization)")




def _infer_projections(atoms, num_wann, has_soc):
    """Infer Wannier90 projection strings from atoms and num_wann.

    Tries to match num_wann to standard orbital sets (per atom × spinor_factor):
      - s:     1 orbital per atom
      - p:     3 orbitals per atom
      - s+p:   4 orbitals per atom (l=0;l=1, NOT sp3 hybrids)
      - d:     5 orbitals per atom
      - s+p+d: 9 orbitals per atom

    Returns None if no match found (falls back to random projections).
    """
    if atoms is None or len(atoms) == 0:
        return None

    elements = sorted(set(sym for sym, _ in atoms))
    n_atoms = len(atoms)
    sf = 2 if has_soc else 1

    # Orbitals per atom (before spinor doubling)
    orb_per_atom = num_wann // (n_atoms * sf)
    remainder = num_wann % (n_atoms * sf)

    if remainder != 0:
        return None

    # Try s-orbitals: 1 per atom
    if orb_per_atom == 1:
        projections = [f"{e}:s" for e in elements]
        print(f"  Auto-inferred projections: {projections}")
        return projections

    # Try p-orbitals: 3 per atom
    if orb_per_atom == 3:
        projections = [f"{e}:p" for e in elements]
        print(f"  Auto-inferred projections: {projections}")
        return projections

    # Try s+p: 4 per atom (use l=0;l=1 not sp3 to avoid imposing
    # tetrahedral hybridization geometry on the initial guess)
    if orb_per_atom == 4:
        projections = [f"{e}:l=0;l=1" for e in elements]
        print(f"  Auto-inferred projections: {projections}")
        return projections

    # Try d-orbitals: 5 per atom
    if orb_per_atom == 5:
        projections = [f"{e}:d" for e in elements]
        print(f"  Auto-inferred projections: {projections}")
        return projections

    # Try s+p+d: 9 per atom (l= form: wannier90 parses everything after the
    # first ';' as an orbital name, so 'Sn:s;Sn:p' dies with "Problem
    # reading l state sn")
    if orb_per_atom == 9:
        projections = [f"{e}:l=0;l=1;l=2" for e in elements]
        print(f"  Auto-inferred projections: {projections}")
        return projections

    return None


def _projections_from_selection(engine, lines, lattice_vectors, has_soc):
    """Wannier90 projection strings from the engine's ACTUAL selected LCAO orbitals
    (engine.selected_orbital_indices), replacing the `random` placeholder. Emits
    'f=fx,fy,fz:l=L' for complete (site, l) shells and per-orbital
    'f=fx,fy,fz:l=L,mr=M' for partial shells (so a frontier selection that doesn't
    fill whole shells still yields REAL projection centres rather than `random`).
    Returns the list whose orbital count equals num_wann, or None (caller falls back
    to _infer_projections, then random) if the selection is unavailable/unparseable
    or the orbital count doesn't match num_wann. Robust for heterogeneous bases where
    _infer_projections (uniform orbitals/atom) gives up.
    """
    sel = getattr(engine, 'selected_orbital_indices', None)
    if sel is None or len(sel) == 0:
        return None
    try:
        from .parser import parse_atomic_basis_info, parse_orbital_types
        info = parse_atomic_basis_info(lines)
        otypes = parse_orbital_types(lines, has_soc=False, num_atoms=info.num_atoms)
    except Exception:
        return None
    L = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4}
    from collections import defaultdict
    # 2-component SOC: eigenvectors are 2N (alpha block 0:N, beta block N:2N), so map
    # each selected spinor index to its spatial AO (i mod N). Each spatial shell then
    # carries both spins -> 2*(2l+1) spinor WFs (wannier90 spinors=.true. doubles it).
    N = info.num_basis
    spin_mult = 2 if has_soc else 1
    sel_aos = (sorted(set(int(a) % N for a in sel)) if has_soc
               else sorted(int(a) for a in sel))
    shells = defaultdict(list)               # (atom_idx, l) -> [spatial_ao, ...]
    for ao in sel_aos:
        try:
            atom = int(info.basis_atom_map[ao])
            lch = otypes.get(ao + 1, '?').lower()
        except Exception:
            return None
        if lch not in L:
            return None
        shells[(atom, L[lch])].append(ao)
    try:
        inv_T = np.linalg.inv(np.asarray(lattice_vectors, float).T)
    except Exception:
        return None
    projections, n = [], 0
    for (atom, l), aos in sorted(shells.items()):
        deg = 2 * l + 1
        fr = inv_T @ np.asarray(info.atom_positions[atom], float)
        ctr = f"f={fr[0]:.6f},{fr[1]:.6f},{fr[2]:.6f}"
        if len(aos) % deg == 0:
            # complete radial shell(s): emit the whole l-shell (compact form)
            for _ in range(len(aos) // deg):
                projections.append(f"{ctr}:l={l}")
        else:
            # PARTIAL shell: emit one projection per selected AO, each a distinct
            # real harmonic (mr=1..2l+1, cycling for multi-zeta). wannier90 takes the
            # actual projection from the provided .amn, so mr only needs to be a valid
            # orbital — what matters is the WF count and the guiding CENTRE (the atom
            # position). This keeps REAL projections (hence useful guiding centres)
            # instead of falling back to `random`.
            for i in range(len(aos)):
                projections.append(f"{ctr}:l={l},mr={(i % deg) + 1}")
        n += spin_mult * len(aos)            # SOC: each spatial shell carries both spins
    if n != engine.num_wann:
        return None
    return projections


def _projections_from_pdwf_config(engine, lines, lattice_vectors, args, has_soc):
    """PDWF real projections from the per-atom VALENCE CONFIG — the same config that
    sets num_wann (compute_num_wann). PDWF nulls selected_orbital_indices because its
    .amn is an SVD over all target radials (no 1:1 AO->WF map), so
    _projections_from_selection bails. But num_wann = Σ_atom Σ_{l∈valence(atom)} (2l+1),
    so emit one `f=ctr:l=L` per valence l-channel of each atom: real atom centres that
    sum to num_wann. NOTE the SVD .amn columns are not atom-ordered, so the per-WF centre
    assignment is APPROXIMATE (physical anchors, far better than `random`, but not exact);
    validate guiding end-to-end. Returns None if structure/config unavailable or
    count != num_wann.
    """
    if getattr(engine, 'pdwf_target_mask', None) is None:
        return None
    try:
        from .parser import parse_atomic_basis_info
        from .valence_config import get_valence_l
        info = parse_atomic_basis_info(lines)
        inv_T = np.linalg.inv(np.asarray(lattice_vectors, float).T)
    except Exception:
        return None
    extended = getattr(args, 'extended', False)
    include_tm_p = getattr(args, 'include_tm_p', False)
    # v2 channel augmentation: semicore channels carry their own WFs
    # (num_wann += n_semicore), so they get a projection card at their atom;
    # polarization channels enlarge the target mask only — no card.
    aug_l = {}
    for (a_idx, l, n_semi) in getattr(engine, '_augmented_channels', []):
        if n_semi > 0:
            aug_l.setdefault(a_idx, set()).add(l)
    projections, n = [], 0
    for a in range(info.num_atoms):
        try:
            l_set = get_valence_l(info.atom_symbols[a], extended=extended,
                                  include_tm_p=include_tm_p)
        except KeyError:
            return None
        l_set = set(l_set) | aug_l.get(a, set())
        fr = inv_T @ np.asarray(info.atom_positions[a], float)
        ctr = f"f={fr[0]:.6f},{fr[1]:.6f},{fr[2]:.6f}"
        for l in sorted(l_set):
            projections.append(f"{ctr}:l={l}")
            n += (2 * l + 1) * (2 if has_soc else 1)
    if n != engine.num_wann:
        return None
    return projections


def _apply_method_window(engine, args):
    """Apply window-based band selection (fallback when --window is explicit).

    If --num-wann is also specified and is less than the number of bands
    in the window, sets up disentanglement: num_bands = bands in window,
    num_wann = user override, with appropriate energy windows.
    """
    e_min, e_max = args.window

    print("\nWindow-based band selection...")
    print("-" * 80)

    result = analyze_band_window(
        engine.eigenvalues_list,
        outer_window=(e_min, e_max),
        e_fermi=engine.e_fermi,
        window_is_relative=True
    )

    num_bands_in_window = result.num_wann
    if num_bands_in_window == 0:
        print("ERROR: No bands found in the energy window!")
        print("Please adjust the energy window and try again.")
        sys.exit(1)

    engine.selected_band_indices = result.frozen_indices

    # Check for --num-wann override (disentanglement mode)
    user_num_wann = getattr(args, 'num_wann', None)
    if user_num_wann is not None and user_num_wann < num_bands_in_window:
        # Disentanglement: num_bands > num_wann
        engine.num_wann = user_num_wann
        engine._num_bands_for_win = num_bands_in_window

        # Set disentanglement windows (absolute energies)
        ef = engine.e_fermi if engine.e_fermi is not None else 0.0
        dis_win_min = ef + e_min
        dis_win_max = ef + e_max

        # Frozen window: the inner energy range containing the target bands
        # Use a narrower window around the Fermi level for the frozen states
        # Find the energy range of the num_wann bands closest to E_F
        all_evals = []
        for evals in engine.eigenvalues_list:
            selected = evals[result.frozen_indices]
            all_evals.extend(selected)
        all_evals = np.sort(all_evals)
        # The frozen window should cover the main bands we want
        # Use: [min of selected bands, E_F + small margin]
        froz_min = result.frozen_energy_range[0]
        froz_max = result.frozen_energy_range[1]
        # Tighten to roughly cover num_wann bands (heuristic: center on E_F)
        dis_froz_min = froz_min
        dis_froz_max = froz_max

        engine._dis_win = (e_min, e_max)  # Relative to E_F
        engine._dis_froz = (dis_froz_min - ef, dis_froz_max - ef)  # Relative to E_F

        print(f"Disentanglement mode:")
        print(f"  num_bands (in window):  {num_bands_in_window}")
        print(f"  num_wann  (override):   {user_num_wann}")
        print(f"  dis_win  (rel E_F):     [{e_min:.4f}, {e_max:.4f}] eV")
        print(f"  dis_froz (rel E_F):     [{dis_froz_min - ef:.4f}, {dis_froz_max - ef:.4f}] eV")
        print(f"  Energy range: [{result.frozen_energy_range[0]:.2f}, {result.frozen_energy_range[1]:.2f}] eV")
    else:
        # No disentanglement: num_bands = num_wann
        engine.num_wann = num_bands_in_window
        print(f"Selected {num_bands_in_window} bands for Wannier functions")
        print(f"  Energy range: [{result.frozen_energy_range[0]:.2f}, {result.frozen_energy_range[1]:.2f}] eV")

    _apply_spread_window_assist(engine, args)

    # Select projection orbitals
    proj_method = getattr(args, 'projection_method', 'weight')
    print(f"\nSelecting projection orbitals (method={proj_method})...")
    print("-" * 80)
    engine.select_projections(verbose=True, method=proj_method)




def stage1_create_win(args, _return_state=False):
    """
    Stage 1: Parse LCAO output and create .win file only.

    This prepares the Wannier90 input file so the user can run
    'wannier90.x -pp seedname' to generate the .nnkp file.
    """
    print("=" * 80)
    print("STAGE 1: Creating Wannier90 Parameter File (.win)")
    print("=" * 80)
    print(f"Input file: {args.input}")
    print(f"Seedname: {args.seedname}")
    print()

    # Check input file exists
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    # Parse CRYSTAL output
    print("Step 1: Parsing CRYSTAL/LCAO output file...")
    print("-" * 80)

    lines, params, H_R_dict, S_R_dict, lattice_vectors_list = _load_matrices(args)
    # Apply user --k-grid override (for memory-limited systems or 2D slabs)
    if getattr(args, 'k_grid', None) is not None:
        original_kgrid = params.k_grid
        params.k_grid = tuple(args.k_grid)
        print(f"  Overriding k-grid: {original_kgrid} -> {params.k_grid} "
              f"(user --k-grid)")
    _validate_k_grid(params.k_grid, H_R_dict, " (stage 1)")
    # Apply user --fermi-energy override (bypass SPINLOCK/level-shift-corrupted value)
    if getattr(args, 'fermi_energy', None) is not None:
        original_fermi = params.fermi_energy
        params.fermi_energy = float(args.fermi_energy)
        print(f"  Overriding Fermi energy: {original_fermi} eV -> "
              f"{params.fermi_energy} eV (user --fermi-energy)")
    lattice_vectors = np.array(lattice_vectors_list)

    print(f"✓ Parsed calculation parameters:")
    if params.fermi_energy is not None:
        print(f"  Fermi energy: {params.fermi_energy:.6f} eV")
    else:
        print(f"  Fermi energy: Not found (will estimate later)")
    print(f"  K-grid: {params.k_grid}")
    print(f"  Number of AOs: {params.num_ao}")

    # Use SOC detection from parser (TWO-COMPONENT SCF marker)
    has_soc = params.has_soc

    print(f"  Spin-orbit coupling: {'Yes' if has_soc else 'No'}")

    # If SOC is detected, we need to double the number of orbitals for spinors
    if has_soc:
        print(f"  → Doubling orbitals for spinors: {params.num_ao} → {params.num_ao * 2}")

    # Organize matrices by R-vector
    print("\nStep 2: Organizing matrices...")
    print("-" * 80)

    # H_R_dict / S_R_dict were built in _load_matrices() (honoring --memory)

    num_basis = params.num_ao if params.num_ao else list(S_R_dict.values())[0].shape[0]
    print(f"✓ Organized matrices")
    print(f"  Basis size: {num_basis}")
    print(f"  Unique R-vectors for H: {len(H_R_dict)}")
    print(f"  Unique R-vectors for S: {len(S_R_dict)}")

    # Create spin-block matrices if SOC
    print("\nStep 3: Creating matrices...")
    print("-" * 80)

    if has_soc:
        H_full_list, S_full_list = create_spin_block_matrices(
            H_R_dict, S_R_dict, num_basis, lattice_vectors_list
        )
        matrix_size = H_full_list[0][1].shape[0]
        print(f"✓ Created {matrix_size}×{matrix_size} SOC matrices")
    else:
        # For non-SOC, symmetrize lower-triangular raw matrices using R/-R pairs
        H_full_list, S_full_list = create_nonsoc_full_matrices(
            H_R_dict, S_R_dict, lattice_vectors_list,
            spin_channel=getattr(args, '_spin_channel', None)
        )
        matrix_size = num_basis
        print(f"✓ Created {matrix_size}×{matrix_size} symmetrized matrices (no SOC)")

    # Prepare real-space matrices in the format engine expects
    print("\nStep 4: Preparing real-space matrices...")
    print("-" * 80)

    real_space_matrices = prepare_real_space_matrices(
        H_full_list, S_full_list, lattice_vectors,
        prune_tol=getattr(args, 'prune_tol', 0.0)
    )
    if not getattr(args, 'no_prune', False):
        real_space_matrices, _ = prune_zero_rvectors(
            real_space_matrices,
            threshold=getattr(args, 'prune_threshold', 0.0),
        )
    print(f"✓ Prepared {len(real_space_matrices)} R-vectors")

    # Initialize engine (window only needed for window-based fallback)
    print("\nStep 5: Initializing Wannier90 engine...")
    print("-" * 80)

    # Determine energy window (used as fallback or with explicit --window)
    if args.window:
        e_min, e_max = args.window
    else:
        e_min, e_max = -5.0, 3.0  # Default window

    engine = Wannier90Engine(
        real_space_matrices=real_space_matrices,
        k_grid=params.k_grid,
        lattice_vectors=lattice_vectors,
        seedname=args.seedname,
        num_wann=None,
        outer_window=(e_min, e_max),
        e_fermi=params.fermi_energy,
        window_is_relative=True
    )

    print("✓ Engine initialized")

    # Parse atomic basis information for phase-corrected MMN
    print("\nParsing atomic basis information...")
    try:
        if getattr(args, 'memory', 'fast') != 'low':
            with open(args.input, 'r') as f:
                lines = f.readlines()
        atomic_info = parse_atomic_basis_info(lines)
        engine.atom_positions = atomic_info.atom_positions

        # For SOC systems, double the basis_atom_map (spin up + spin down)
        num_orbitals = engine.num_orbitals
        if num_orbitals == 2 * atomic_info.num_basis:
            engine.basis_atom_map = np.concatenate([
                atomic_info.basis_atom_map,
                atomic_info.basis_atom_map
            ])
            print(f"✓ Parsed {atomic_info.num_atoms} atoms, {atomic_info.num_basis} basis functions (SOC: doubled basis map)")
        else:
            engine.basis_atom_map = atomic_info.basis_atom_map
            print(f"✓ Parsed {atomic_info.num_atoms} atoms, {atomic_info.num_basis} basis functions")
    except Exception as e:
        print(f"⚠ Warning: Could not parse atomic basis info: {e}")
        print("  MMN file will not have phase correction")

    # Solve eigenvalue problems
    print("\nStep 6: Solving eigenvalue problems...")
    print("-" * 80)

    engine.solve_all_kpoints(parallel=not args.no_parallel,
                             num_processes=getattr(args, 'lcao_workers', None),
                             num_bands=getattr(args, 'solve_nbands', None),
                             store_S=getattr(args, 'memory', 'fast') != 'low')
    print("✓ Eigenvalue problems solved")

    # Sanity-check parsed Fermi vs HOMO from electron count. Auto-falls
    # back to the electron-count estimate with a loud warning if the two
    # disagree by > FERMI_FRAME_TOLERANCE_EV (CRYSTAL SPINLOCK/shift
    # frame-mismatch detection).
    _sanity_check_fermi_energy(engine, params, args, stage_name="Stage 1")

    # Estimate Fermi energy if needed
    if params.fermi_energy is None:
        print("\nEstimating Fermi energy...")
        if params.num_electrons is not None:
            fermi_energy = estimate_fermi_energy(
                engine.eigenvalues_list,
                num_electrons=params.num_electrons,
                method='auto'
            )
            engine.e_fermi = fermi_energy
            print(f"✓ Estimated Fermi energy: {fermi_energy:.6f} eV")
        else:
            print("⚠ Cannot estimate Fermi energy (num_electrons not found)")
            print("  Using E_F = 0.0 eV")
            engine.e_fermi = 0.0

    # Method dispatch for band selection
    print(f"\nStep 7: Band selection (method={args.method})...")
    print("-" * 80)

    if args.method == 'direct':
        _apply_method_direct(engine, args, has_soc, params)
    elif args.method in ('pdwf', 'auto', 'hybrid'):
        try:
            _apply_method_pdwf(engine, args, has_soc, lines)
        except Exception as exc:
            # NOTE: SystemExit is deliberately NOT caught. It used to be, so
            # a deliberate abort inside the PDWF path (e.g. an infeasible
            # basis) was converted into a fallback run that completed green.
            if args.method == 'auto':
                print(f"\n  ⚠ PDWF method failed ({type(exc).__name__}: {exc}); "
                      f"falling back to projectability method.")
                print("  ⚠ The fallback selects bands on the LEGACY unbounded "
                      "compactness measure (C^dag S^2 C, not bounded by 1 and "
                      "carrying no target-orbital information), whose "
                      "thresholds were calibrated separately. Treat this "
                      "run's band selection as unvalidated: prefer fixing the "
                      "PDWF failure above, or set --method explicitly.")
                _apply_method_projectability(engine, args, has_soc=has_soc)
            else:
                raise
        if args.method == 'hybrid':
            if getattr(args, 'baseline', None):
                _apply_baseline_overrides(engine, args)
            else:
                _apply_hybrid_overrides(engine)
    elif args.method == 'projectability':
        if args.window:
            print(f"Using explicit energy window: [{e_min:.2f}, {e_max:.2f}] eV")
            _apply_method_window(engine, args)
        else:
            _apply_method_projectability(engine, args, has_soc=has_soc)

    # Parse atoms from CRYSTAL output if available
    print("\nStep 8: Extracting atomic positions...")
    print("-" * 80)

    try:
        with open(args.input, 'r') as f:
            crystal_lines = f.readlines()
        atoms_result = parse_atoms_from_crystal_output(crystal_lines)
        atoms = atoms_result[0]  # Returns (atoms_list, lattice_vectors)
        print(f"✓ Found {len(atoms)} atoms")
        for symbol, pos in atoms:
            print(f"  {symbol}: {pos}")
    except Exception as e:
        print(f"⚠ Could not parse atoms: {e}")
        print("  Will create .win without atoms block")
        atoms = None

    # Write .win file only
    print(f"\nStep 9: Writing {args.seedname}.win file...")
    print("-" * 80)

    # Prepare projections — prefer the ACTUAL selected LCAO orbitals (robust for
    # heterogeneous bases); fall back to the uniform-per-atom heuristic, then random.
    projections = args.projections if args.projections else None
    if projections is None:
        projections = _projections_from_selection(
            engine, lines, engine.lattice_vectors, has_soc)
        if projections is not None:
            print(f"  Real projections from {len(projections)} selected (site,l) "
                  f"shells (sum to num_wann={engine.num_wann})")
    # PDWF nulls selected_orbital_indices (its .amn is an SVD over target radials), so
    # derive real centres from the per-atom valence config instead of `random`.
    if projections is None and getattr(engine, 'pdwf_target_mask', None) is not None:
        projections = _projections_from_pdwf_config(
            engine, lines, engine.lattice_vectors, args, has_soc)
        if projections is not None:
            print(f"  Real PDWF projections from the valence config: "
                  f"{len(projections)} (atom,l) shells (sum to num_wann={engine.num_wann})")
    if projections is None and atoms is not None:
        projections = _infer_projections(atoms, engine.num_wann, has_soc)

    # Couple guiding_centres to projection success. Guiding centres anchor each
    # Wannier function to its projection centre: BENEFICIAL with real projections
    # (pulls the WFs onto their atoms), but SERIOUSLY detrimental with the `random`
    # fallback, where the centres are random points that drag the WFs apart
    # (Sc 6x6: diverged to ~5.7e4 Ang^2/WF with guiding+random, vs localizing to
    # ~11 Ang^2 with guiding off). So: guiding centres ON iff projections exist.
    _wanted_guiding = getattr(engine, '_guiding_centres', False)
    engine._guiding_centres = projections is not None
    if projections is None and _wanted_guiding:
        print("  ⚠ Projections fell back to `random` — guiding_centres DISABLED "
              "(random guiding centres would drag the Wannier functions apart)")
    elif projections is not None and not _wanted_guiding:
        print("  guiding_centres ENABLED (real projection centres available)")

    # Set num_iter for projectability method (5000 iterations for proper convergence)
    if args.method == 'projectability' and engine._override_num_iter is None:
        engine._override_num_iter = 5000

    # Auto-detect kpoint path for band structure plots
    kpoint_path = None
    if args.bands_plot:
        from .win_file import (
            KPATH_HEXAGONAL_2D, KPATH_HEXAGONAL_3D, KPATH_SIMPLE_CUBIC,
            KPATH_FCC, KPATH_BCC
        )
        custom = getattr(args, 'custom_kpath', None)
        lv = engine.lattice_vectors
        a1_len = np.linalg.norm(lv[0])
        a2_len = np.linalg.norm(lv[1])
        a3_len = np.linalg.norm(lv[2])

        if custom:
            # User-specified path: "G:0,0,0;M:0.5,0,0;K:0.33,0.33,0;G:0,0,0"
            pts = []
            for tok in custom.split(';'):
                label, coords = tok.split(':')
                pts.append((label, np.array([float(x) for x in coords.split(',')])))
            kpoint_path = []
            for i in range(len(pts) - 1):
                kpoint_path += [pts[i], pts[i + 1]]
            print(f"  Using user --custom-kpath ({len(pts)} high-symmetry points)")
        elif a3_len > 10 * a1_len:
            # 2D system: large vacuum along a3
            kpoint_path = KPATH_HEXAGONAL_2D
            print(f"  Auto-detected 2D hexagonal lattice -> Gamma-M-K band path")
        else:
            # 3D system: detect lattice type from angles and lengths
            angles = []
            for i, j in [(0, 1), (0, 2), (1, 2)]:
                cos_a = np.dot(lv[i], lv[j]) / (np.linalg.norm(lv[i]) * np.linalg.norm(lv[j]))
                angles.append(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))
            alpha, beta, gamma = angles
            lengths = [a1_len, a2_len, a3_len]
            equal_lengths = (abs(lengths[0] - lengths[1]) < 0.01 * lengths[0] and
                             abs(lengths[1] - lengths[2]) < 0.01 * lengths[1])
            all_90 = all(abs(a - 90.0) < 1.0 for a in [alpha, beta, gamma])

            if equal_lengths and all_90:
                kpoint_path = KPATH_SIMPLE_CUBIC
                print(f"  Auto-detected simple cubic lattice -> Gamma-X-M-R band path")
            elif equal_lengths and not all_90:
                # Could be FCC or BCC (rhombohedral primitive cell)
                avg_angle = np.mean([alpha, beta, gamma])
                if avg_angle < 80:
                    kpoint_path = KPATH_FCC
                    print(f"  Auto-detected FCC-like lattice -> Gamma-X-W-K-L band path")
                else:
                    kpoint_path = KPATH_BCC
                    print(f"  Auto-detected BCC-like lattice -> Gamma-H-N-P band path")

            # Hexagonal: two 90-deg angles + one 120-deg (or 60), two equal axes.
            # (which vector pair carries the 120 varies, so check order-independently)
            n_120 = sum(1 for a in (alpha, beta, gamma)
                        if abs(a - 120.0) < 2.0 or abs(a - 60.0) < 2.0)
            n_90 = sum(1 for a in (alpha, beta, gamma) if abs(a - 90.0) < 2.0)
            two_equal = (abs(a1_len - a2_len) < 0.02 * a1_len
                         or abs(a1_len - a3_len) < 0.02 * a1_len
                         or abs(a2_len - a3_len) < 0.02 * a2_len)
            hex_like = (n_120 == 1 and n_90 == 2 and two_equal)
            if kpoint_path is None and hex_like:
                kpoint_path = KPATH_HEXAGONAL_3D
                print(f"  Auto-detected 3D hexagonal lattice -> "
                      f"Gamma-M-K-Gamma-A-L-H-A band path")

            if kpoint_path is None:
                print(f"  Could not auto-detect lattice type for band path")
                print(f"  Lattice lengths: {lengths}, angles: {angles}")

    # Write only the .win file
    engine.write_files(
        verbose=True,
        write_win=True,
        use_nnkp=False,  # Don't use .nnkp in stage 1
        atoms=atoms,
        projections=projections,
        spinors=has_soc,
        bands_plot=args.bands_plot,
        kpoint_path=kpoint_path,
    )

    # Delete the .eig, .amn, .mmn files if they were created
    # (write_files creates them by default, but we only want .win in stage 1)
    for ext in ['.eig', '.amn', '.mmn']:
        filepath = f"{args.seedname}{ext}"
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"  (Removed premature {filepath})")

    # Record an explicit --fermi-energy in the .win so stage 2 re-applies it
    _persist_fermi_override(args)

    # Consistency check: does the generated .win satisfy Wannier90's
    # disentanglement rules at every k-point? (Catches steep bands entering the
    # frozen window before the user runs wannier90.x -- see wannier_checks.)
    _check_disentanglement(args, engine, stage=1)

    # Internal wannier90-exact .nnkp: kmesh.py reproduces wannier90's own
    # k-mesh construction (validated bit-for-bit against `wannier90.x -pp`
    # output on 82 .nnkp files across every material/grid in calculations/),
    # so the -pp round trip is no longer required between stages. Precision
    # note inside kmesh.write_nnkp: the same full-precision lattice/kpoints
    # written to the .win are used here.
    _nnkp_ok = False
    if not getattr(args, 'no_internal_nnkp', False):
        try:
            from .kmesh import kmesh_get, write_nnkp
            _kinfo = kmesh_get(np.asarray(engine.lattice_vectors, float),
                               np.asarray(engine.kpoints, float))
            write_nnkp(f"{args.seedname}.nnkp", engine.lattice_vectors,
                       engine.kpoints, _kinfo)
            _nnkp_ok = True
            print(f"\n✓ Created: {args.seedname}.nnkp (internal k-mesh: "
                  f"nntot={_kinfo.nntot}, shells {_kinfo.shell_list} — "
                  f"wannier90 -pp not required)")
        except Exception as _e:
            print(f"\n⚠ Internal .nnkp generation failed ({_e}); "
                  f"fall back to: wannier90.x -pp {args.seedname}")

    print()
    print("=" * 80)
    print("STAGE 1 COMPLETE!")
    print("=" * 80)
    print(f"✓ Created: {args.seedname}.win")
    print()
    if _nnkp_ok:
        print("NEXT STEP (wannier90 -pp NOT needed; .nnkp already written):")
        print(f"  python {sys.argv[0]} --stage 2 --input {args.input} --seedname {args.seedname}")
    else:
        print("NEXT STEP:")
        print(f"  Run Wannier90 preprocessing to generate neighbor information:")
        print(f"  → wannier90.x -pp {args.seedname}")
        print()
        print(f"  This will create {args.seedname}.nnkp")
        print()
        print("Then proceed to Stage 2:")
        print(f"  python {sys.argv[0]} --stage 2 --input {args.input} --seedname {args.seedname}")
    print("=" * 80)

    if _return_state:
        # --stage all continues on this engine (no re-parse / re-solve)
        return engine, lines, lattice_vectors, has_soc, _nnkp_ok


def _validate_k_grid(k_grid, H_R_dict, context=""):
    """Real k-grid sanity checks (the old help text claimed a divisibility
    rule that has no mathematical basis — H(k) is exact from the parsed
    H(R) on ANY grid). What can actually go wrong:
      * grid entries < 1 (hard error);
      * a >1 grid along a direction where every parsed R-vector has zero
        extent (2D/1D system): H(k) is constant along that axis, so the
        extra planes replicate identical k-points — pure waste (Sc slab at
        6x6x6 would solve 6x the work of 6x6x1 for the same physics).
    """
    kg = tuple(int(n) for n in k_grid)
    if any(n < 1 for n in kg):
        print(f"ERROR: invalid k-grid {kg}{context}: entries must be >= 1")
        sys.exit(1)
    if H_R_dict:
        rmax = [0, 0, 0]
        for R in H_R_dict:
            for i in range(3):
                rmax[i] = max(rmax[i], abs(int(R[i])))
        for i, ax in enumerate("xyz"):
            if kg[i] > 1 and rmax[i] == 0:
                print(f"  ⚠ k-grid warning{context}: {kg[i]} points along "
                      f"{ax} but every parsed R-vector has zero {ax}-extent "
                      f"(2D/1D system) — H(k) is constant along {ax}; use 1 "
                      f"to avoid solving {kg[i]}x redundant k-points.")


_FERMI_OVERRIDE_TAG = "! lcao2wannier: fermi_energy_override ="


def _persist_fermi_override(args):
    """Record an explicit --fermi-energy in the .win (as a comment) so a
    later stage-2 run cannot silently disagree with the stage-1 band
    selection frame."""
    if getattr(args, 'fermi_energy', None) is None:
        return
    win = f"{args.seedname}.win"
    try:
        with open(win) as f:
            text = f.read()
        if _FERMI_OVERRIDE_TAG not in text:
            with open(win, 'a') as f:
                f.write(f"\n{_FERMI_OVERRIDE_TAG} "
                        f"{float(args.fermi_energy):.10f} eV\n")
    except OSError:
        pass


def _recall_fermi_override(args):
    """Stage 2: re-apply a stage-1 --fermi-energy override recorded in the
    .win when the flag was not repeated (the .eig frame must match the
    band-selection frame)."""
    if getattr(args, 'fermi_energy', None) is not None:
        return
    try:
        with open(f"{args.seedname}.win") as f:
            for line in f:
                if line.startswith(_FERMI_OVERRIDE_TAG):
                    val = float(line.split("=")[1].split()[0])
                    args.fermi_energy = val
                    print(f"  Re-applying stage-1 --fermi-energy override "
                          f"from .win: {val} eV")
                    return
    except OSError:
        pass


def _spin_channel_index(args):
    """Spin channel index for collinear spin-polarized input: 0 = up/alpha
    (default), 1 = down/beta.  Selected by ``--spin {up,down}``."""
    return 1 if getattr(args, 'spin', 'up') == 'down' else 0


def _froz_emax_arg(value):
    """--hybrid-froz-emax: 'auto', 'none', or a float (eV above E_F)."""
    v = str(value).strip().lower()
    if v in ('auto', 'none'):
        return v
    try:
        return float(v)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--hybrid-froz-emax: want auto, none, or an energy in eV, got {value!r}")


def _float_or_auto(text):
    """argparse type: a float, or the word auto."""
    if str(text).strip().lower() == 'auto':
        return 'auto'
    return float(text)


def _read_win_value(text, key):
    """Read a scalar keyword (e.g. dis_froz_max) from .win text; None if absent."""
    import re
    m = re.search(rf'^[ \t]*{key}[ \t]*=[ \t]*([-\d.eE+]+)', text, re.MULTILINE)
    return float(m.group(1)) if m else None


def _patch_win_value(text, key, value):
    """Replace the value of a scalar keyword in .win text (in place)."""
    import re
    return re.subn(rf'(^[ \t]*{key}[ \t]*=[ \t]*)[-\d.eE+]+',
                   rf'\g<1>{value:.8f}', text, flags=re.MULTILINE)


def _apply_auto_window(seedname, froz_max_fixed=False):
    """Stage-2 post-step: pick the Omega_I-minimal disentanglement window from the
    just-written .amn/.mmn/.eig and patch the .win in place.

    Port of wien2wannier's pdwf_optwin: Omega_I is gauge-invariant, so it is
    evaluated directly from the overlaps without a full Wannierisation.
    """
    from .spread import auto_window_from_seedname
    win_path = f"{seedname}.win"
    with open(win_path) as f:
        text = f.read()
    nw = _read_win_value(text, 'num_wann')
    froz_min = _read_win_value(text, 'dis_froz_min')
    froz_max = _read_win_value(text, 'dis_froz_max')
    dis_min = _read_win_value(text, 'dis_win_min')
    dis_max = _read_win_value(text, 'dis_win_max')

    print("=" * 80)
    print("AUTO-WINDOW: Omega_I-optimal disentanglement window selection")
    print("=" * 80)
    if None in (nw, froz_min, froz_max, dis_min, dis_max):
        print("  No complete disentanglement window in .win "
              "(need dis_win + dis_froz); skipping auto-window.")
        return

    res = auto_window_from_seedname(seedname, int(nw), froz_min, froz_max,
                                    dis_min, dis_max,
                                    froz_max_fixed=(froz_max if froz_max_fixed
                                                    else None))
    if not np.isfinite(res['omega_I']):
        print("  No valid Omega_I-minimising window found; keeping Stage-1 window.")
        return

    ob = res['omega_I_before']
    # Do no harm: keep the Stage-1 window if it is already valid and at least as
    # good.  The coverage constraint guarantees no under-freezing but can RAISE
    # Omega_I when the original window was not under-freezing in the first place.
    if res.get('info_before', 1) == 0 and np.isfinite(ob) and ob <= res['omega_I'] + 1e-6:
        print(f"  Stage-1 window already Omega_I-optimal "
              f"(before {ob:.4f} <= candidate {res['omega_I']:.4f} Ang^2); .win unchanged.")
        return

    for key, val in (('dis_froz_min', res['froz_min']),
                     ('dis_froz_max', res['froz_max']),
                     ('dis_win_min', res['dis_min']),
                     ('dis_win_max', res['dis_max'])):
        text, _ = _patch_win_value(text, key, val)
    with open(win_path, 'w') as f:
        f.write(text)

    ob = res['omega_I_before']
    delta = f"  ({(ob - res['omega_I']) / ob * 100:+.1f}%)" if np.isfinite(ob) and ob > 0 else ""
    print(f"  Omega_I:  {ob:.4f}  ->  {res['omega_I']:.4f} Ang^2{delta}")
    print(f"  frozen window: [{res['froz_min']:.3f}, {res['froz_max']:.3f}] eV")
    print(f"  outer  window: [{res['dis_min']:.3f}, {res['dis_max']:.3f}] eV")
    print(f"  Patched {win_path}.  Re-run:  wannier90.x {seedname}")


def _attach_gto_basis(engine, args, lines):
    """Parse the GTO basis for the exact analytic MMN method, if requested."""
    if getattr(args, 'mmn_method', 'analytic') == 'analytic' or args.method == 'hybrid':
        print("\nParsing GTO basis for analytic MMN...")
        try:
            from .gto_mmn import parse_gto_basis
            engine.gto_aos = parse_gto_basis(lines)
            _cut = getattr(args, 'gto_cutoff', None)
            if _cut is not None:
                engine.gto_cutoff = float(_cut)
            print(f"✓ Parsed {len(engine.gto_aos)} contracted AOs "
                  f"(exact analytic MMN enabled, cutoff "
                  f"{getattr(engine, 'gto_cutoff', 16.0):g} Bohr)")
        except Exception as e:
            print(f"⚠ Warning: Could not parse GTO basis: {e}")
            print("  Falling back to midpoint MMN method.")
            args.mmn_method = 'midpoint'




def _spin_suffixed_path(path, args):
    """Suffix an output path with the collinear spin label (alpha/beta) so a
    --spin both run doesn't overwrite one channel with the other (mirrors the
    PDWF_DUMP convention)."""
    sl = getattr(args, '_spin_label', None)
    if not path or not sl:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}_{sl}{ext}"


def _apply_hybrid_guiding_centres(engine, seedname, anchors):
    """--hybrid-guiding-centres: anchor the wannierise at the SCDM anchor sites.

    Rewrites the .win projections block with one centre card per SCDM anchor
    AO (at its host atom's position, via the parser's orbital-to-atom map)
    and forces guiding_centres = .true. — wannier90's guiding centres ARE
    the projection centres, so the projections block is the mechanism. The
    .amn (written by run_hybrid) supplies the actual projection amplitudes;
    the cards only carry the guiding CENTRES.

    Safety lesson respected: guiding centres are only safe with real,
    non-random anchors (random centres dragged the Sc WFs apart to ~5.7e4
    Ang^2/WF). The SCDM anchors are AO columns of the disentangled subspace,
    so the centres are physical atomic sites by construction.

    Grammar: one `f=..:l=0` card = one orbital; with spinors = .true.
    wannier90 doubles each card, so num_wann/2 cards are written for SOC
    (consecutive sorted anchor pairs — normally the two spinor copies of
    one spatial AO, hence the same atom) and num_wann cards otherwise.
    """
    import re
    win_path = f"{seedname}.win"
    atom_map = getattr(engine, 'basis_atom_map', None)
    positions = getattr(engine, 'atom_positions', None)
    if atom_map is None or positions is None or not os.path.exists(win_path):
        print("  ⚠ --hybrid-guiding-centres: orbital-to-atom map or .win "
              "unavailable — guiding centres NOT applied")
        return
    with open(win_path) as f:
        text = f.read()
    spinors = re.search(r'^[ \t]*spinors[ \t]*=[ \t]*\.?t', text,
                        re.MULTILINE | re.IGNORECASE) is not None

    anchors = sorted(int(a) for a in anchors)
    if spinors:
        if len(anchors) % 2:
            print("  ⚠ --hybrid-guiding-centres: odd anchor count with "
                  "spinors — guiding centres NOT applied")
            return
        # Pair anchors by HOST ATOM, not by raw AO index: the spinor blocks
        # are stacked (beta partner of AO mu is mu+N), so raw-index-adjacent
        # pairing can straddle atoms whenever a (atom, spin) block hosts an
        # odd anchor count. Grouping by atom pairs each atom's alpha/beta
        # anchors with each other — exact (every card sits on a genuine
        # anchor atom) whenever each atom hosts an even total, which Kramers
        # symmetry guarantees for time-reversal-symmetric anchor sets.
        by_atom = sorted(anchors, key=lambda a: (int(atom_map[a]), a))
        pairs = [(by_atom[i], by_atom[i + 1])
                 for i in range(0, len(by_atom), 2)]
        mixed = sum(1 for a, b in pairs
                    if int(atom_map[a]) != int(atom_map[b]))
        if mixed:
            print(f"  ⚠ {mixed} anchor pair(s) span different atoms (odd "
                  f"per-atom anchor counts) — their centre uses the first "
                  f"anchor's atom (approximate)")
        reps = [a for a, _b in pairs]
    else:
        reps = anchors

    inv_T = np.linalg.inv(np.asarray(engine.lattice_vectors, float).T)
    cards = []
    for a in reps:
        fr = inv_T @ np.asarray(positions[int(atom_map[a])], float)
        cards.append(f"  f={fr[0]:.6f},{fr[1]:.6f},{fr[2]:.6f}:l=0")
    block = ("begin projections\n"
             "  ! SCDM anchor guiding centres (--hybrid-guiding-centres)\n"
             + "\n".join(cards) + "\nend projections")
    text, nsub = re.subn(r'begin projections.*?end projections', block,
                         text, count=1, flags=re.DOTALL)
    if nsub != 1:
        print("  ⚠ --hybrid-guiding-centres: no projections block found in "
              f"{win_path} — guiding centres NOT applied")
        return
    if re.search(r'^[ \t]*guiding_centres[ \t]*=', text, re.MULTILINE):
        text = re.sub(r'^[ \t]*guiding_centres[ \t]*=.*$',
                      'guiding_centres = .true.', text, count=1,
                      flags=re.MULTILINE)
    else:
        text = re.sub(r'^(num_print_cycles[ \t]*=.*)$',
                      r'\1\nguiding_centres = .true.', text, count=1,
                      flags=re.MULTILINE)
    with open(win_path, 'w') as f:
        f.write(text)
    print(f"  [guiding-centres] {len(cards)} SCDM anchor centre(s) written "
          f"into {win_path} (guiding_centres = .true.)")


def _stage2_finalize(engine, args, lines):
    """Stage-2 tail shared by --stage 2 and --stage all: write the
    .eig/.amn/.mmn (or run the hybrid pipeline) against the .nnkp neighbor
    order, then re-check the disentanglement rules."""
    nnkp_file = f"{args.seedname}.nnkp"
    print(f"\nStep 9: Writing data files using {nnkp_file} neighbors...")
    print("-" * 80)

    # Write only the data files, not .win (already exists)
    if args.method == 'hybrid' and getattr(args, 'raw_overlaps', False):
        print("  Canonical overlaps stage: emitting the selected raw parent "
              "pool without the hybrid disentanglement/gauge solver")
        if getattr(args, 'mmn_method', 'analytic') != 'analytic':
            raise SystemExit(
                "named --stage overlaps for --method hybrid requires "
                "--mmn-method analytic, matching the hybrid parent metric")
        from .hybrid_pipeline import run_raw_hybrid_parent
        run_raw_hybrid_parent(
            engine, args.seedname,
            pool_factor=getattr(args, 'hybrid_pool_cap', 1.5),
            verbose=True,
        )
    elif args.method == 'hybrid' and getattr(args, 'baseline', None):
        from .hybrid_pipeline import run_baseline_dis_froz_proj
        for flag, name in ((getattr(args, 'dump_masks', None), '--dump-masks'),
                           (getattr(args, 'hybrid_report_json', None),
                            '--hybrid-report-json'),
                           (getattr(args, 'hybrid_guiding_centres', False),
                            '--hybrid-guiding-centres')):
            if flag:
                print(f"  ⚠ {name} is ignored with --baseline "
                      f"{args.baseline} (masks/anchors are wannier90's "
                      f"job in this mode)")
        run_baseline_dis_froz_proj(
            engine, args.seedname,
            pool_factor=getattr(args, 'hybrid_pool_cap', 1.5),
            amn_mode=('svd' if args.baseline == 'dis-froz-proj-svd'
                      else 'lowdin'),
            verbose=True,
        )
    elif args.method == 'hybrid':
        from .hybrid_pipeline import run_hybrid
        # The hybrid target is CHANNEL-derived: num_wann is however many AOs
        # the selected channel set contains, so an explicit --num-wann cannot
        # be honoured without inventing which AOs to add or drop. Silently
        # ignoring it is the dangerous outcome -- it produced a 6-WF Fe model
        # while the user asked for 9, and the mismatch only surfaced later as
        # a rank-deficient gauge (cond 1.7e15). Refuse with the fix instead.
        _req_nw = getattr(args, 'num_wann', None)
        if _req_nw is not None and int(_req_nw) != int(engine.num_wann):
            raise SystemExit(
                f"--num-wann {int(_req_nw)} conflicts with the hybrid "
                f"method's channel-derived target of {int(engine.num_wann)} "
                f"Wannier functions. The hybrid target is a SET OF AO "
                f"CHANNELS, not a count, so the count cannot be changed "
                f"directly. Change the channel set instead:\n"
                f"  --include-tm-p   add the transition-metal p channel "
                f"(elemental 3d metals need s+p+d = 9; without it the "
                f"s+d = 6 target is rank-deficient)\n"
                f"  --extended       add the extended radial set\n"
                f"  --projections    specify the channels explicitly\n"
                f"Then re-run without --num-wann.")
        hyb = run_hybrid(
            engine, args.seedname,
            p_froz=getattr(args, 'hybrid_p_froz', None),
            p_floor=getattr(args, 'hybrid_p_floor', 0.10),
            pool_factor=getattr(args, 'hybrid_pool_cap', 1.5),
            p_froz_band=getattr(args, 'hybrid_p_froz_band', None),
            trust=getattr(args, 'hybrid_trust', 'auto'),
            fid_emax=getattr(args, 'hybrid_fid_emax', 8.0),
            froz_emax=getattr(args, 'hybrid_froz_emax', 'auto'),
            thr_floor=getattr(args, 'hybrid_thr_floor', 0.55),
            trust_margin=getattr(args, 'hybrid_trust_margin', 0.015),
            dis_niter=getattr(args, 'hybrid_dis_niter', 2000),
            dis_tol=getattr(args, 'hybrid_dis_tol', 1e-9),
            dis_rel_tol=getattr(args, 'hybrid_dis_rel_tol', 1e-10),
            report=getattr(args, 'hybrid_report', False),
            dump_masks=_spin_suffixed_path(
                getattr(args, 'dump_masks', None), args),
            report_json=_spin_suffixed_path(
                getattr(args, 'hybrid_report_json', None), args),
            dump_handoff=_spin_suffixed_path(
                getattr(args, 'dump_handoff', None), args),
            shell_p_floor=getattr(args, 'hybrid_shell_p_floor', 'auto'),
            shell_floor_max_frac=getattr(args, 'hybrid_shell_floor_budget', 0.01),
            gauge_rank_tol=getattr(args, 'hybrid_gauge_rank_tol', 1e-10),
            gauge_min_sv=getattr(args, 'hybrid_gauge_min_sv', None),
            require_capacity=getattr(args, 'hybrid_require_capacity', False),
            dis_check_every=getattr(args, 'hybrid_dis_check_every', 1),
            dis_mix=getattr(args, 'hybrid_dis_mix', 1.0),
            dis_mix_schedule=getattr(args, 'hybrid_dis_mix_schedule', None),
            dis_taper=getattr(args, 'hybrid_dis_taper', 0.5),
            dis_taper_on=getattr(args, 'hybrid_dis_taper_on', 'check'),
            dis_taper_reset_history=getattr(
                args, 'hybrid_dis_taper_reset_history', False),
            dis_log=getattr(args, 'hybrid_dis_log', None),
            dis_relax_late=getattr(args, 'hybrid_dis_relax_late', None),
            verbose=True,
        )
        if getattr(args, 'hybrid_guiding_centres', False):
            _apply_hybrid_guiding_centres(engine, args.seedname,
                                          hyb['anchors'])
    else:
        engine.write_files(
            verbose=True,
            write_win=False,  # Don't overwrite .win
            use_nnkp=True,    # Use .nnkp neighbors (CRITICAL!)
            mmn_method=getattr(args, 'mmn_method', 'analytic'),
        )

    # Consistency check: confirm the existing .win still satisfies the
    # disentanglement rules against the .eig bands we just wrote.
    _check_disentanglement(args, engine, stage=2)

    print()
    print("=" * 80)
    print("STAGE 2 COMPLETE!")
    print("=" * 80)
    print(f"✓ Created: {args.seedname}.eig")
    print(f"✓ Created: {args.seedname}.amn")
    print(f"✓ Created: {args.seedname}.mmn")
    print()
    print("All files generated with correct neighbor structure from .nnkp!")

    if getattr(args, 'auto_window', False):
        print()
        _apply_auto_window(
            args.seedname,
            froz_max_fixed=(getattr(args, 'window_route', False)
                            and str(getattr(args, 'hybrid_froz_emax', 'auto')).lower() != 'none'))

    print()
    print("NEXT STEP:")
    print(f"  Run Wannier90 to generate maximally localized Wannier functions:")
    print(f"  → wannier90.x {args.seedname}")
    print()
    print("=" * 80)


def stage_all_run(args):
    """--stage all: the full pipeline in ONE invocation and ONE solve.

    Stage 1 body (parse -> solve -> band selection -> .win + internal
    .nnkp) followed directly by the stage-2 tail on the SAME engine — no
    re-parse, no re-solve, no re-derivation of the band selection, and no
    wannier90 -pp round trip (the .nnkp comes from the wannier90-exact
    internal k-mesh). Optionally launches the wannierise itself
    (--wannier90 PATH).
    """
    state = stage1_create_win(args, _return_state=True)
    if state is None:
        print("ERROR: stage 1 did not produce an engine; cannot continue.")
        sys.exit(1)
    engine, lines, lattice_vectors, has_soc, nnkp_ok = state
    if not nnkp_ok:
        print("\nERROR (--stage all): the internal .nnkp generation failed, "
              "so the neighbor order cannot be guaranteed.")
        print(f"  Run: wannier90.x -pp {args.seedname}")
        print(f"  Then: --stage 2 --input {args.input} "
              f"--seedname {args.seedname}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("STAGE ALL: continuing to data files on the same engine "
          "(no re-parse / re-solve)")
    print("=" * 80)

    _attach_gto_basis(engine, args, lines)
    _stage2_finalize(engine, args, lines)

    w90 = getattr(args, 'wannier90', None)
    if w90:
        import subprocess
        seed_dir = os.path.dirname(args.seedname) or '.'
        seed_base = os.path.basename(args.seedname)
        cmd = os.environ.get('LCAO_W90_MPI', '').split() + [w90, seed_base]
        print("\n" + "=" * 80)
        print(f"Running wannier90: {' '.join(cmd)}  (cwd: {seed_dir})")
        print("=" * 80)
        ret = subprocess.run(cmd, cwd=seed_dir)
        if ret.returncode == 0:
            print(f"✓ wannier90 finished — see {args.seedname}.wout")
        else:
            print(f"⚠ wannier90 exited with code {ret.returncode} — "
                  f"see {args.seedname}.wout / .werr")


def stage2_create_data_files(args):
    """
    Stage 2: Read .nnkp and create .eig, .amn, .mmn files.

    This uses the neighbor information from Wannier90's preprocessing
    to generate data files with the exact neighbor structure expected.
    """
    print("=" * 80)
    print("STAGE 2: Creating Wannier90 Data Files (.eig, .amn, .mmn)")
    print("=" * 80)
    print(f"Input file: {args.input}")
    print(f"Seedname: {args.seedname}")
    print()

    # Check that .nnkp file exists
    nnkp_file = f"{args.seedname}.nnkp"
    if not os.path.exists(nnkp_file):
        print(f"ERROR: {nnkp_file} not found!")
        print()
        print("Run Stage 1 first (it writes the .nnkp via the internal "
              "wannier90-exact k-mesh):")
        print(f"  1. python {sys.argv[0]} --stage 1 --input {args.input} --seedname {args.seedname}")
        print(f"  2. python {sys.argv[0]} --stage 2 --input {args.input} --seedname {args.seedname}")
        print()
        print("(or do both in one go: --stage all; or generate the .nnkp "
              f"with `wannier90.x -pp {args.seedname}`)")
        print()
        sys.exit(1)

    # Check that .win file exists
    win_file = f"{args.seedname}.win"
    if not os.path.exists(win_file):
        print(f"ERROR: {win_file} not found!")
        print("Please run Stage 1 first to create the .win file.")
        sys.exit(1)

    print(f"✓ Found {nnkp_file}")
    print(f"✓ Found {win_file}")
    print()

    # Parse CRYSTAL output (same as stage 1)
    print("Step 1: Parsing CRYSTAL/LCAO output file...")
    print("-" * 80)

    lines, params, H_R_dict, S_R_dict, lattice_vectors_list = _load_matrices(args)
    # k-grid resolution: explicit --k-grid wins but must agree with the .win
    # mp_grid; without --k-grid, INFER mp_grid from the .win rather than
    # falling back to the native SHRINK mesh — a stage-1/stage-2 grid
    # mismatch silently mis-sizes the k-set and dies with an IndexError deep
    # in the neighbor lookup (hit on Sc: 6x6x1 stage 1, forgotten at stage 2).
    _win_mp = None
    try:
        import re as _re
        with open(win_file) as _f:
            _m = _re.search(r'^[ \t]*mp_grid[ \t]*[=:][ \t]*(\d+)[ \t]+(\d+)'
                            r'[ \t]+(\d+)', _f.read(), _re.MULTILINE)
        if _m:
            _win_mp = tuple(int(g) for g in _m.groups())
    except OSError:
        pass
    if getattr(args, 'k_grid', None) is not None:
        if _win_mp is not None and tuple(args.k_grid) != _win_mp:
            print(f"ERROR: --k-grid {tuple(args.k_grid)} contradicts "
                  f"mp_grid {_win_mp} in {win_file}.")
            print("  The .mmn neighbor order is defined on the .win grid; "
                  "re-run stage 1 or drop --k-grid.")
            sys.exit(1)
        original_kgrid = params.k_grid
        params.k_grid = tuple(args.k_grid)
        print(f"  Overriding k-grid: {original_kgrid} -> {params.k_grid} "
              f"(user --k-grid)")
    elif _win_mp is not None and _win_mp != tuple(params.k_grid or ()):
        print(f"  Inferring k-grid from {win_file}: mp_grid {_win_mp} "
              f"(native grid was {params.k_grid})")
        params.k_grid = _win_mp
    _validate_k_grid(params.k_grid, H_R_dict, " (stage 2)")
    # Re-apply a stage-1 Fermi override recorded in the .win (if any)
    _recall_fermi_override(args)
    # Apply user --fermi-energy override (bypass SPINLOCK/level-shift-corrupted value)
    if getattr(args, 'fermi_energy', None) is not None:
        original_fermi = params.fermi_energy
        params.fermi_energy = float(args.fermi_energy)
        print(f"  Overriding Fermi energy: {original_fermi} eV -> "
              f"{params.fermi_energy} eV (user --fermi-energy)")
    lattice_vectors = np.array(lattice_vectors_list)

    print(f"✓ Parsed calculation parameters:")
    if params.fermi_energy is not None:
        print(f"  Fermi energy: {params.fermi_energy:.6f} eV")
    else:
        print(f"  Fermi energy: Not found (will estimate later)")
    print(f"  K-grid: {params.k_grid}")
    print(f"  Number of AOs: {params.num_ao}")

    # Use SOC detection from parser (TWO-COMPONENT SCF marker)
    has_soc = params.has_soc

    print(f"  Spin-orbit coupling: {'Yes' if has_soc else 'No'}")

    # If SOC is detected, we need to double the number of orbitals for spinors
    if has_soc:
        print(f"  → Doubling orbitals for spinors: {params.num_ao} → {params.num_ao * 2}")

    # Organize matrices by R-vector
    print("\nStep 2: Organizing matrices...")
    print("-" * 80)

    # H_R_dict / S_R_dict were built in _load_matrices() (honoring --memory)

    num_basis = params.num_ao if params.num_ao else list(S_R_dict.values())[0].shape[0]
    print(f"✓ Organized matrices")
    print(f"  Basis size: {num_basis}")
    print(f"  Unique R-vectors for H: {len(H_R_dict)}")
    print(f"  Unique R-vectors for S: {len(S_R_dict)}")

    # Create spin-block matrices if SOC
    print("\nStep 3: Creating matrices...")
    print("-" * 80)

    if has_soc:
        H_full_list, S_full_list = create_spin_block_matrices(
            H_R_dict, S_R_dict, num_basis, lattice_vectors_list
        )
        matrix_size = H_full_list[0][1].shape[0]
        print(f"✓ Created {matrix_size}×{matrix_size} SOC matrices")
    else:
        # For non-SOC, symmetrize lower-triangular raw matrices using R/-R pairs
        H_full_list, S_full_list = create_nonsoc_full_matrices(
            H_R_dict, S_R_dict, lattice_vectors_list,
            spin_channel=getattr(args, '_spin_channel', None)
        )
        matrix_size = num_basis
        print(f"✓ Created {matrix_size}×{matrix_size} symmetrized matrices (no SOC)")

    # Prepare real-space matrices
    print("\nStep 4: Preparing real-space matrices...")
    print("-" * 80)

    real_space_matrices = prepare_real_space_matrices(
        H_full_list, S_full_list, lattice_vectors,
        prune_tol=getattr(args, 'prune_tol', 0.0)
    )
    if not getattr(args, 'no_prune', False):
        real_space_matrices, _ = prune_zero_rvectors(
            real_space_matrices,
            threshold=getattr(args, 'prune_threshold', 0.0),
        )
    print(f"✓ Prepared {len(real_space_matrices)} R-vectors")

    # Read num_wann and num_bands from .win file if it exists
    print("\nStep 5: Reading parameters from .win file...")
    print("-" * 80)

    num_wann_from_win = None
    num_bands_from_win = None
    dis_froz_min_from_win = None
    dis_froz_max_from_win = None
    win_file = f"{args.seedname}.win"

    if os.path.exists(win_file):
        with open(win_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('num_wann'):
                    try:
                        num_wann_from_win = int(line.split('=')[1].strip())
                        print(f"  Found num_wann = {num_wann_from_win}")
                    except:
                        pass
                elif line.startswith('num_bands'):
                    try:
                        num_bands_from_win = int(line.split('=')[1].strip())
                        print(f"  Found num_bands = {num_bands_from_win}")
                    except:
                        pass
                elif line.startswith('dis_froz_min'):
                    try:
                        dis_froz_min_from_win = float(line.split('=')[1].strip())
                        print(f"  Found dis_froz_min = {dis_froz_min_from_win}")
                    except:
                        pass
                elif line.startswith('dis_froz_max'):
                    try:
                        dis_froz_max_from_win = float(line.split('=')[1].strip())
                        print(f"  Found dis_froz_max = {dis_froz_max_from_win}")
                    except:
                        pass
        print(f"✓ Read parameters from {win_file}")
    else:
        print(f"⚠ Warning: {win_file} not found, using automatic band selection")

    # Determine energy window (same as stage 1)
    print("\nStep 6: Determining energy window...")
    print("-" * 80)

    if args.window:
        e_min, e_max = args.window
        print(f"Using user-specified window: [{e_min:.2f}, {e_max:.2f}] eV")
    else:
        e_min, e_max = -5.0, 3.0
        print(f"Using default window: [{e_min:.2f}, {e_max:.2f}] eV (relative to E_F)")

    # Initialize engine
    print("\nStep 7: Initializing Wannier90 engine...")
    print("-" * 80)

    engine = Wannier90Engine(
        real_space_matrices=real_space_matrices,
        k_grid=params.k_grid,
        lattice_vectors=lattice_vectors,
        seedname=args.seedname,
        num_wann=num_wann_from_win,  # Use value from .win file
        outer_window=(e_min, e_max),
        e_fermi=params.fermi_energy,
        window_is_relative=True
    )

    print("✓ Engine initialized")

    # Parse atomic basis information for phase-corrected MMN
    print("\nParsing atomic basis information...")
    try:
        if getattr(args, 'memory', 'fast') != 'low':
            with open(args.input, 'r') as f:
                lines = f.readlines()
        atomic_info = parse_atomic_basis_info(lines)
        engine.atom_positions = atomic_info.atom_positions

        # For SOC systems, double the basis_atom_map (spin up + spin down)
        num_orbitals = engine.num_orbitals
        if num_orbitals == 2 * atomic_info.num_basis:
            engine.basis_atom_map = np.concatenate([
                atomic_info.basis_atom_map,
                atomic_info.basis_atom_map
            ])
            print(f"✓ Parsed {atomic_info.num_atoms} atoms, {atomic_info.num_basis} basis functions (SOC: doubled basis map)")
        else:
            engine.basis_atom_map = atomic_info.basis_atom_map
            print(f"✓ Parsed {atomic_info.num_atoms} atoms, {atomic_info.num_basis} basis functions")
    except Exception as e:
        print(f"⚠ Warning: Could not parse atomic basis info: {e}")
        print("  MMN file will not have phase correction (may cause negative spreads!)")

    _attach_gto_basis(engine, args, lines)

    # Solve eigenvalue problems
    print("\nStep 7: Solving eigenvalue problems...")
    print("-" * 80)

    engine.solve_all_kpoints(parallel=not args.no_parallel,
                             num_processes=getattr(args, 'lcao_workers', None),
                             num_bands=getattr(args, 'solve_nbands', None),
                             store_S=getattr(args, 'memory', 'fast') != 'low')
    print("✓ Eigenvalue problems solved")

    # Sanity-check parsed Fermi vs HOMO from electron count. Auto-falls
    # back to the electron-count estimate with a loud warning if the two
    # disagree by > FERMI_FRAME_TOLERANCE_EV (CRYSTAL SPINLOCK/shift
    # frame-mismatch detection).
    _sanity_check_fermi_energy(engine, params, args, stage_name="Stage 2")

    # Estimate Fermi energy if needed
    if params.fermi_energy is None:
        print("\nEstimating Fermi energy...")
        if params.num_electrons is not None:
            fermi_energy = estimate_fermi_energy(
                engine.eigenvalues_list,
                num_electrons=params.num_electrons,
                method='auto'
            )
            engine.e_fermi = fermi_energy
            print(f"✓ Estimated Fermi energy: {fermi_energy:.6f} eV")
        else:
            print("⚠ Cannot estimate Fermi energy (num_electrons not found)")
            print("  Using E_F = 0.0 eV")
            engine.e_fermi = 0.0

    # Method dispatch for band selection
    print(f"\nStep 8: Band selection (method={args.method})...")
    print("-" * 80)

    if args.method in ('pdwf', 'auto', 'hybrid'):
        try:
            _apply_method_pdwf(engine, args, has_soc, lines)
        except Exception as exc:
            # NOTE: SystemExit is deliberately NOT caught. It used to be, so
            # a deliberate abort inside the PDWF path (e.g. an infeasible
            # basis) was converted into a fallback run that completed green.
            if args.method == 'auto':
                print(f"\n  ⚠ PDWF method failed ({type(exc).__name__}: {exc}); "
                      f"falling back to projectability method.")
                print("  ⚠ The fallback selects bands on the LEGACY unbounded "
                      "compactness measure (C^dag S^2 C, not bounded by 1 and "
                      "carrying no target-orbital information), whose "
                      "thresholds were calibrated separately. Treat this "
                      "run's band selection as unvalidated: prefer fixing the "
                      "PDWF failure above, or set --method explicitly.")
                _apply_method_projectability(engine, args, has_soc=has_soc)
            else:
                raise
        if args.method == 'hybrid':
            if getattr(args, 'baseline', None):
                _apply_baseline_overrides(engine, args)
            else:
                _apply_hybrid_overrides(engine)
    elif args.method == 'direct':
        _apply_method_direct(engine, args, has_soc, params)
    elif args.method == 'projectability':
        if args.window:
            print(f"Using explicit energy window: [{e_min:.2f}, {e_max:.2f}] eV")
            _apply_method_window(engine, args)
        elif num_wann_from_win is not None:
            # .win file exists with num_wann — use projectability but
            # verify consistency with .win parameters
            _apply_method_projectability(engine, args, has_soc=has_soc)
            if engine.num_wann != num_wann_from_win:
                print(f"\n⚠ Note: Projectability selected {engine.num_wann} bands, "
                      f"but .win file has num_wann = {num_wann_from_win}")
                print(f"  Using projectability result ({engine.num_wann} bands)")
        else:
            _apply_method_projectability(engine, args, has_soc=has_soc)

    # Write data files using .nnkp neighbors (shared tail with --stage all)
    _stage2_finalize(engine, args, lines)


def stage3_symmetrize_hr(args):
    """
    Stage 3: postprocess wannier90_hr.dat (Hermitization + time-reversal).

    Enforces H(R) = [H(R) + H(-R)^dag] / 2 and, for SOC models, time-reversal
    symmetry directly on the real-space Hamiltonian produced by Wannier90.
    Generic H(R) cleanup: no crystal symmetry is detected or imposed.

    Requires:
      - CRYSTAL output file (to determine spin-orbit coupling)
      - wannier90_hr.dat (from Wannier90 run after Stage 2)
    """
    from .postprocess import (
        enforce_hermiticity, enforce_time_reversal, read_hr_file, write_hr_file
    )

    print("=" * 80)
    print("STAGE 3: Postprocess Wannier90 Hamiltonian (wannier90_hr.dat)")
    print("=" * 80)
    print(f"Input file: {args.input}")
    print(f"Seedname: {args.seedname}")
    print()

    # Check input files
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    hr_file = args.hr_file or f"{args.seedname}_hr.dat"
    if not os.path.exists(hr_file):
        print(f"ERROR: HR file not found: {hr_file}")
        print()
        print("You must run Wannier90 first (after Stage 2) to produce the HR file.")
        print(f"  Expected: {hr_file}")
        print()
        print("If the file has a different name, use --hr-file to specify it.")
        sys.exit(1)

    print(f"  HR file: {hr_file}")
    print()

    # --- Step 1: Check for spin-orbit coupling ---
    print("Step 1: Checking for spin-orbit coupling...")
    print("-" * 80)

    with open(args.input, 'r') as f:
        lines = f.readlines()

    params = parse_calculation_parameters(lines)
    has_soc = params.has_soc
    print(f"  Spin-orbit coupling: {'Yes' if has_soc else 'No'}")

    # --- Step 2: Load the Wannier90 Hamiltonian ---
    print("\nStep 2: Loading Wannier90 Hamiltonian...")
    print("-" * 80)

    num_wann, R_list, ndegen, H = read_hr_file(hr_file)
    print(f"  Loaded: {num_wann} orbitals, {len(R_list)} R-points")

    # --- Step 3: Postprocess ---
    print("\nStep 3: Postprocessing...")
    print("-" * 80)

    matrices = {R: {'H': H[R]} for R in R_list}

    if not args.no_hermitize:
        matrices = enforce_hermiticity(matrices)
        print("  ✓ Hermiticity enforced: H(R) = [H(R) + H(-R)^dag] / 2")

    if not args.no_time_reversal and has_soc:
        norbs_spatial = num_wann // 2
        matrices = enforce_time_reversal(matrices, norbs_spatial)
        print("  ✓ Time-reversal symmetry enforced")

    H_out = {R: matrices[R]['H'] for R in R_list}

    # --- Step 4: Write output ---
    output_file = args.output or f"{hr_file}_postprocessed"
    write_hr_file(output_file, num_wann, R_list, ndegen, H_out,
                  threshold=args.symm_threshold)

    print()
    print("=" * 80)
    print("STAGE 3 COMPLETE!")
    print("=" * 80)
    print(f"  Output: {output_file}")
    print(f"  R-points: {len(R_list)}")
    print()
    print("The postprocessed HR file can be used with wannier_tools or other")
    print("tight-binding post-processing codes.")
    print("=" * 80)


def stage4_plot_bands(args):
    """
    Stage 4: Plot LCAO band structure with PDWF projectability coloring.

    Computes eigenvalues along a high-symmetry k-path and generates a
    two-panel plot with projectability coloring and projected DOS.
    """
    from .band_plot import (
        run_band_structure, parse_custom_kpath, get_kpath_for_lattice,
        PlotConfig,
    )
    from .basis_parser import parse_basis_shells, get_atom_list
    from .valence_config import (
        build_target_mask, compute_num_wann, summarize_config,
    )
    from .lcao_pdwf import (
        compute_lowdin_projectability, classify_bands, determine_windows,
        ClassificationParams, print_pdwf_summary,
    )
    from .band_selection import estimate_fermi_energy

    print("=" * 80)
    print("STAGE 4: LCAO Band Structure Plot")
    print("=" * 80)
    print(f"Input file: {args.input}")
    print(f"Seedname: {args.seedname}")
    print()

    # Check input file exists
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    # ---- Parse Crystal23 output ----
    print("Step 1: Parsing CRYSTAL/LCAO output file...")
    print("-" * 80)

    lines, params, H_R_dict, S_R_dict, lattice_vectors_list = _load_matrices(args)
    # Apply user --k-grid override (for memory-limited systems or 2D slabs)
    if getattr(args, 'k_grid', None) is not None:
        original_kgrid = params.k_grid
        params.k_grid = tuple(args.k_grid)
        print(f"  Overriding k-grid: {original_kgrid} -> {params.k_grid} "
              f"(user --k-grid)")
    # Apply user --fermi-energy override (bypass SPINLOCK/level-shift-corrupted value)
    if getattr(args, 'fermi_energy', None) is not None:
        original_fermi = params.fermi_energy
        params.fermi_energy = float(args.fermi_energy)
        print(f"  Overriding Fermi energy: {original_fermi} eV -> "
              f"{params.fermi_energy} eV (user --fermi-energy)")
    lattice_vectors = np.array(lattice_vectors_list)

    has_soc = params.has_soc
    print(f"  Fermi energy: {params.fermi_energy if params.fermi_energy else 'Not found'}")
    print(f"  K-grid: {params.k_grid}")
    print(f"  SOC: {'Yes' if has_soc else 'No'}")

    # Organize matrices
    # H_R_dict / S_R_dict were built in _load_matrices() (honoring --memory)

    # Create full matrices
    if has_soc:
        H_full_list, S_full_list = create_spin_block_matrices(
            H_R_dict, S_R_dict, params.num_ao, lattice_vectors_list
        )
    else:
        H_full_list, S_full_list = create_nonsoc_full_matrices(
            H_R_dict, S_R_dict, lattice_vectors_list,
            spin_channel=getattr(args, '_spin_channel', None)
        )

    real_space_matrices = prepare_real_space_matrices(
        H_full_list, S_full_list, lattice_vectors,
        prune_tol=getattr(args, 'prune_tol', 0.0)
    )
    if not getattr(args, 'no_prune', False):
        real_space_matrices, _ = prune_zero_rvectors(
            real_space_matrices,
            threshold=getattr(args, 'prune_threshold', 0.0),
        )
    print(f"  {len(real_space_matrices)} R-vectors")

    # ---- PDWF analysis on uniform grid (unless --no-pdwf) ----
    target_mask = None
    classification = None
    windows = None
    e_fermi = None

    no_pdwf = getattr(args, 'no_pdwf', False)

    if not no_pdwf:
        print("\nStep 2: PDWF analysis on uniform grid...")
        print("-" * 80)

        shells, num_ao = parse_basis_shells(lines, num_atoms=params.num_atoms)
        atoms = get_atom_list(shells)

        kgrid = params.k_grid if params.k_grid else [6, 6, 6]
        engine = Wannier90Engine(
            real_space_matrices=real_space_matrices,
            k_grid=kgrid,
            lattice_vectors=lattice_vectors,
        )
        engine.solve_all_kpoints(parallel=not getattr(args, 'no_parallel', False),
                                 num_processes=getattr(args, 'lcao_workers', None),
                                 validate_overlap=False)

        nb = engine.num_orbitals
        nk = engine.num_kpoints
        print(f"  {nk} k-points, {nb} bands")

        # Build target mask
        target_mask = build_target_mask(shells, has_soc=has_soc, verbose=False)
        matrix_size = engine.S_k_list[0].shape[0]
        if len(target_mask) != matrix_size:
            if len(target_mask) > matrix_size:
                target_mask = target_mask[:matrix_size]
            else:
                extended = np.zeros(matrix_size, dtype=bool)
                extended[:len(target_mask)] = target_mask
                target_mask = extended

        num_wann = compute_num_wann(atoms, has_soc=has_soc)
        print(f"  num_wann = {num_wann}")
        print(summarize_config(atoms, has_soc=has_soc))

        # Fermi energy
        eigenvalues = np.array(engine.eigenvalues_list)
        if params.fermi_energy is not None:
            e_fermi = params.fermi_energy
        else:
            e_fermi = estimate_fermi_energy(engine.eigenvalues_list, method='midgap')
        print(f"  E_Fermi = {e_fermi:.4f} eV")

        # PDWF classification
        proj_grid = compute_lowdin_projectability(
            engine.eigenvectors_list, engine.S_k_list, target_mask,
        )
        classification = classify_bands(
            proj_grid, eigenvalues, num_wann,
            ClassificationParams(
                p_high=getattr(args, 'pdwf_p_high', 0.95),
                p_low=getattr(args, 'pdwf_p_low', 0.10),
                e_fermi=e_fermi,
            ),
        )
        windows = determine_windows(classification, eigenvalues, e_fermi)
        print_pdwf_summary(classification, windows, eigenvalues, e_fermi, [])
    else:
        print("\nStep 2: Skipping PDWF analysis (--no-pdwf)")
        # Still need Fermi energy
        if params.fermi_energy is not None:
            e_fermi = params.fermi_energy
        else:
            # Quick solve to estimate Fermi energy
            kgrid = params.k_grid if params.k_grid else [6, 6, 6]
            engine = Wannier90Engine(
                real_space_matrices=real_space_matrices,
                k_grid=kgrid,
                lattice_vectors=lattice_vectors,
            )
            engine.solve_all_kpoints(parallel=not getattr(args, 'no_parallel', False),
                                     num_processes=getattr(args, 'lcao_workers', None),
                                     validate_overlap=False)
            e_fermi = estimate_fermi_energy(engine.eigenvalues_list, method='midgap')
        print(f"  E_Fermi = {e_fermi:.4f} eV")

    # ---- Determine k-path ----
    print("\nStep 3: Band structure computation...")
    print("-" * 80)

    kpath_spec = None
    kpath_type = getattr(args, 'kpath', 'auto')

    if kpath_type == 'custom':
        custom_str = getattr(args, 'custom_kpath', None)
        if custom_str is None:
            print("ERROR: --custom-kpath required with --kpath custom")
            sys.exit(1)
        kpath_spec = parse_custom_kpath(custom_str, npts=getattr(args, 'npts', 60))
    elif kpath_type != 'auto':
        kpath_spec = get_kpath_for_lattice(kpath_type, npts=getattr(args, 'npts', 60))

    # Plot configuration
    energy_range = getattr(args, 'energy_range', None)
    if energy_range is None:
        energy_range = (-20.0, 25.0)
    else:
        energy_range = tuple(energy_range)

    plot_config = PlotConfig(energy_range=energy_range)

    # Output path
    output_plot = getattr(args, 'output_plot', None)
    if output_plot is None:
        output_plot = f"{args.seedname}_bands.png"

    # Run band structure
    band_data = run_band_structure(
        real_space_matrices=real_space_matrices,
        lattice_vectors=lattice_vectors,
        e_fermi=e_fermi,
        output_path=output_plot,
        kpath_spec=kpath_spec,
        npts=getattr(args, 'npts', 60),
        target_mask=target_mask,
        classification=classification,
        windows=windows,
        config=plot_config,
        seedname=args.seedname,
        verbose=True,
    )

    print()
    print("=" * 80)
    print("STAGE 4 COMPLETE")
    print("=" * 80)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="lcao2wannier Multi-Stage Workflow Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WORKFLOW (one command; no wannier90 -pp needed — the .nnkp is generated
internally, validated exact against wannier90's own k-mesh):
    python %(prog)s --stage all --input material.out --seedname material \\
        --method hybrid --mmn-method analytic --k-grid 8 8 8
    wannier90.x material        (or add --wannier90 /path/to/wannier90.x)

  Staged flow (equivalent; also -pp-free):
    Stage 1: .win + .nnkp     python %(prog)s --stage 1 ...
    Stage 2: .eig/.amn/.mmn   python %(prog)s --stage 2 ...
                              (mp_grid inferred from the .win if --k-grid
                               is omitted)
    Then: wannier90.x material

  Stage 3: Postprocess the tight-binding Hamiltonian (Hermitization + TR)
    python %(prog)s --stage 3 --input material.out --seedname material

METHODS (Stages 1-2):
  --method hybrid (RECOMMENDED for production)
    Per-state projectability disentanglement performed IN-PIPELINE:
    per-(k,band) freezing masks, frozen-constrained Omega_I minimization,
    LCAO-native SCDM initial gauge, and a pre-disentangled
    num_bands = num_wann hand-off (no dis_* keywords -- wannier90 only
    rotates). Joint valence+conduction models where energy windows fail.
    Thresholds derived automatically from the projectability distribution
    (--hybrid-trust auto, the default); the target orbital set is
    auto-augmented when the low conduction is under-represented.
    Pair with --mmn-method analytic. See METHODOLOGY.md Sec. 12.

  --method pdwf
    Chemistry-grounded projectability band selection (Lowdin projectability
    onto the valence configuration) with energy-window disentanglement
    delegated to wannier90. --method auto = pdwf with fallback to
    projectability.

MMN METHODS (--mmn-method, Stage 2):
  analytic  (default) Exact momentum-shifted Gaussian integrals; M(k,b)
            contractive on ANY mesh (use for coarse grids and always with
            hybrid).
  midpoint  S(k+b/2) finite difference; best on dense well-conditioned
            meshes, can diverge on coarse ones.
  lowdin    Orthonormalized + atom-centered Berry phase; contractive,
            diagonal-position approximation.

  --method projectability (DEFAULT)
    Select bands by projectability onto LCAO basis.
    Bands with p_avg >= threshold are kept. No energy window needed.

TUNING / DIAGNOSTICS (hybrid):
  --hybrid-report                 print the projectability anatomy table
  --dump-masks / --hybrid-report-json
                                  the same anatomy, machine-readable
                                  (.npz per-state masks / fig3-schema JSON)
  --hybrid-thr-floor / --hybrid-trust-margin / --hybrid-pool-cap /
  --augment-coverage-min / --augment-retention-floor
                                  move the calibrated auto-rule constants
  --hybrid-dis-niter/-tol/-rel-tol
                                  disentangle sweep budget and tolerances
  --hybrid-guiding-centres        SCDM-anchor guiding centres in the .win
  --baseline dis-froz-proj        emit the stock-wannier90 dis_froz_proj
                                  baseline instead of the hybrid hand-off
LARGE SYSTEMS:
  --solve-nbands N                subset eigensolve (~2.5x at N ~ 35%%)
  --memory low                    streaming parser + on-demand S(k)
    Tune with --proj-threshold (default: 0.9).

  --method direct
    Use ALL LCAO orbitals as Wannier functions (num_wann = num_basis).
    Skips Wannier90 spread minimization (num_iter = 0).
    Warning issued for num_basis > 200; use --force to override.

STAGE 3 OPTIONS:
  --hr-file PATH      Path to wannier90_hr.dat (default: {seedname}_hr.dat)
  --no-hermitize      Skip Hermitization step
  --no-time-reversal  Skip time-reversal symmetry enforcement
  --symm-threshold    Threshold for dropping small hoppings (default: 1e-9)
  --output PATH       Output filename (default: {hr_file}_postprocessed)

EXAMPLES:
  # Bismuth with default projectability method
  python %(prog)s --stage 1 --input Bi.out --seedname bismuth
  wannier90.x -pp bismuth
  python %(prog)s --stage 2 --input Bi.out --seedname bismuth
  wannier90.x bismuth

  # Post-Wannierization H(R) postprocessing (Stage 3)
  python %(prog)s --stage 3 --input Bi.out --seedname bismuth

  # Direct LCAO mapping (all orbitals)
  python %(prog)s --stage 1 --input Bi.out --seedname bismuth --method direct

  # Explicit energy window (overrides projectability)
  python %(prog)s --stage 1 --input Bi.out --seedname bismuth --window -6 2
"""
    )

    io_group = parser.add_argument_group(
        'Input / output', 'Required input file and Wannier90 output naming')
    stages_group = parser.add_argument_group(
        'Stages', 'Run control: which stage to run and how to solve it')
    method_group = parser.add_argument_group(
        'Band selection & method', 'Which bands become Wannier functions')
    mmn_group = parser.add_argument_group(
        'MMN', 'Overlap-matrix (.mmn) algorithm')
    hybrid_group = parser.add_argument_group(
        'Hybrid pipeline (advanced)',
        'Tuning knobs for --method hybrid (safe defaults; most runs need none '
        'of these)')
    dev_group = parser.add_argument_group(
        'Developer / debug',
        'Instrumentation and A/B-comparison flags for developing or '
        'validating the hybrid pipeline (--dump-masks, --dump-handoff, '
        '--hybrid-report-json, --hybrid-dis-log, --baseline, and related '
        'knobs). The canonical lcao2wannier --help lists their contracts.')

    # Required arguments
    stages_group.add_argument('--stage', type=lambda s: s if s == 'all' else int(s),
                        choices=[1, 2, 3, 4, 'all'], required=True,
                        help='all: full pipeline in one run (parse/solve once, '
                             'write .win + .nnkp + .eig/.amn/.mmn; no '
                             'wannier90 -pp needed; add --wannier90 PATH to '
                             'also run the wannierise) | '
                             'Stage 1: Create .win (+ internal .nnkp) | '
                             'Stage 2: Create .eig/.amn/.mmn | '
                             'Stage 3: Postprocess wannier90_hr.dat | '
                             'Stage 4: Plot band structure')
    io_group.add_argument('--wannier90', type=str, default=None, metavar='PATH',
                        help='--stage all: run this wannier90 executable on '
                             'the finished seedname (prefix with mpirun via '
                             'env LCAO_W90_MPI, e.g. "mpirun -np 6")')
    io_group.add_argument('--input', '-i', type=str, required=True,
                        help='Input CRYSTAL/LCAO output file')
    io_group.add_argument('--seedname', '-s', type=str, required=True,
                        help='Seedname for Wannier90 files (e.g., "material")')

    # Optional arguments
    method_group.add_argument('--window', type=float, nargs=2, metavar=('E_MIN', 'E_MAX'),
                        help='Energy window in eV relative to Fermi level (default: -5.0 3.0)')
    stages_group.add_argument('--k-grid', type=int, nargs=3, metavar=('NX', 'NY', 'NZ'),
                        default=None,
                        help='Override Monkhorst-Pack k-grid (default: read from '
                             'CRYSTAL output SHRINK FACT). Use to downsample a '
                             'dense CRYSTAL k-grid for memory-limited systems '
                             '(e.g., --k-grid 6 6 6 for CrI3 on 8-32 GB RAM). '
                             'Any grid is mathematically valid (H(k) is exact '
                             'from the parsed H(R)); a warning is issued for '
                             'a >1 grid along a direction with no R-vector '
                             'extent (2D/1D systems: use 1 there).')
    stages_group.add_argument('--fermi-energy', type=float, default=None, metavar='EV',
                        help='Override Fermi energy in eV (bypass CRYSTAL-reported '
                             'value). Required when CRYSTAL SPINLOCK + LEVEL SHIFTER '
                             'corrupts the reported FERMI ENERGY (see "LOCKING - '
                             'FERMI ENERGY ALTERED BY LEVEL SHIFTER" in the LCAO '
                             'output). Estimate from band count: for an insulator, '
                             'set halfway between VBM and CBM; for a metal, use the '
                             'middle of the occupied/empty transition.')
    io_group.add_argument('--projections', type=str, nargs='+',
                        help='Wannier90 projection strings (default: random)')
    io_group.add_argument('--bands-plot', action='store_true',
                        help='Enable band structure plotting in Wannier90')
    stages_group.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel computation')
    stages_group.add_argument('--lcao-workers', type=int, default=None,
                        help=argparse.SUPPRESS)
    stages_group.add_argument('--raw-overlaps', action='store_true',
                        help=argparse.SUPPRESS)
    stages_group.add_argument('--lcao-cache-dir', type=str, default=None,
                        help=argparse.SUPPRESS)
    stages_group.add_argument('--lcao-parse-only', action='store_true',
                        help=argparse.SUPPRESS)
    stages_group.add_argument('--memory', type=str, choices=['fast', 'low'],
                        default='fast',
                        help='Memory strategy. fast (default): legacy in-RAM '
                             'parse. low: single streaming pass — no full-file '
                             'buffer, no intermediate matrix list, float64 for '
                             'non-SOC. Cuts peak RSS ~2-3x for large inputs at '
                             'no speed cost (parsing dominates). Use for files '
                             'that approach available RAM (see '
                             'scripts/estimate_memory.py).')
    stages_group.add_argument('--spin', choices=['alpha', 'beta', 'both'], default=None,
                        help='Collinear spin-polarized (UNRESTRICTED) outputs '
                             'only (stages 1-2). alpha/beta: Wannierize that '
                             'channel using the given seedname. both: two '
                             'independent runs -> <seed>_alpha and <seed>_beta '
                             '(sharing the overlap S(R)). Default on a '
                             'spin-polarized file is both. Error on restricted '
                             'or two-component SOC outputs.')
    stages_group.add_argument('--no-prune', action='store_true',
                        help='Do not drop all-zero R-vectors. By default, cells '
                             'whose H(R) and S(R) are exactly zero (CRYSTAL emits '
                             'more real-space cells than needed) are removed — '
                             'this is bit-identical and shrinks the stacked '
                             'arrays / Fourier cost.')
    stages_group.add_argument('--prune-threshold', type=float, default=0.0,
                        help='Prune R-vectors with max|H(R)|,max|S(R)| <= this '
                             '(default: 0.0 = exact zeros only, result-preserving). '
                             'A small value (e.g. 1e-10) also drops negligible '
                             'cells but may change results slightly.')
    method_group.add_argument('--spread-window-assist', action='store_true',
                        help='Refine the disentanglement windows to minimize the '
                             'Wannier spread: keep the subspace projectable '
                             '(exclude diffuse, low-projectability bands that '
                             'inflate Omega_I) while honoring --min-froz-window. '
                             'Stages 1-2, projectability/pdwf methods.')
    method_group.add_argument('--min-froz-window', type=float, nargs=2,
                        metavar=('LO', 'HI'), default=[-6.0, 3.0],
                        help='Minimum frozen window (eV relative to E_F) that the '
                             'spread-window-assist must always keep frozen -- the '
                             'band-structure target reproduced exactly by the '
                             'interpolation (default: -6.0 3.0, deeper into the '
                             'valence than the conduction side). If it holds more '
                             'than num_wann bands/k, num_wann is raised to honor it.')
    method_group.add_argument('--assist-proj-floor', type=float, default=None,
                        help='Projectability below which a band is treated as '
                             'diffuse by --spread-window-assist (default: '
                             '--proj-threshold). Raise it to exclude more bands '
                             'and tighten localization.')
    mmn_group.add_argument('--mmn-method', type=str, default='analytic',
                        choices=['midpoint', 'lowdin', 'lowdin_no_berry',
                                 'analytic'],
                        help="Algorithm for the .mmn overlaps. 'analytic' "
                             "(default): exact analytic Gaussian-overlap MMN "
                             "(unitary on any k-mesh; parses the CRYSTAL GTO "
                             "basis from the .out). 'midpoint': S evaluated "
                             "at k+b/2 (accurate on dense grids, non-unitary "
                             "on coarse ones). 'lowdin': Lowdin-orthonormalized "
                             "overlaps with explicit Berry phase (singular "
                             "values bounded [0,1]). 'lowdin_no_berry': same "
                             "without the explicit phase.")
    mmn_group.add_argument('--gto-cutoff', type=float, default=None,
                        help='Real-space cutoff in Bohr for the analytic GTO '
                             'overlaps used by --mmn-method analytic and '
                             '--method hybrid (default: 16). The truncation '
                             'is not isotropic, so raise this if S(k) needs '
                             'to be more accurate at long range.')
    method_group.add_argument('--frozen-conduction', type=str, default='auto',
                        choices=['auto', 'off'],
                        help="For insulators (PDWF method), whether to extend the "
                             "frozen window through the gap to pin the low "
                             "conduction manifold. 'auto' (default): freeze the "
                             "well-projected low conduction bands (projectability "
                             ">= --conduction-pmin), capped so the per-k frozen "
                             "count stays <= num_wann, so both valence and low "
                             "conduction are reproduced. 'off': valence-only "
                             "frozen window (exact valence, loose conduction).")
    method_group.add_argument('--conduction-pmin', type=float, default=0.70,
                        help="Minimum projectability for a conduction band to be "
                             "frozen by --frozen-conduction auto (default: 0.70). "
                             "LOWER it (e.g. 0.5) to freeze MORE conduction bands "
                             "above E_F; RAISE it to freeze fewer. Bands below the "
                             "threshold have nearly-free-electron/diffuse character "
                             "the atom-centred basis cannot represent, so freezing "
                             "them degrades localization.")
    method_group.add_argument('--window-mode', type=str, default='manifold',
                        choices=['manifold', 'spread'],
                        help="How --spread-window-assist positions the windows. "
                             "'manifold' (default): the --min-froz-window is a "
                             "region of interest; num_wann = the bands passing "
                             "through it, and the disentanglement window follows "
                             "those bands to their full extent (deep valence, "
                             "shallow conduction). 'spread': keep the method's "
                             "num_wann and grow the outer window to >= num_wann "
                             "bands/k preferring the valence side.")

    # Method selection
    method_group.add_argument('--method', type=str,
                        choices=['projectability', 'direct', 'pdwf', 'auto', 'hybrid', 'window'],
                        default='auto',
                        help='Wannierization method (default: auto). '
                             'hybrid: per-state masks, isolated Nb=Nw hand-off. '
                             'window: the automatic window route (PDWF band '
                             'classification at the derived thresholds, '
                             'Omega_I-optimal disentanglement windows, stock '
                             'Wannier90 disentanglement) = pdwf + --auto-window. '
                             'pdwf: LCAO-PDWF chemistry-grounded band selection. '
                             'auto: try pdwf, fall back to projectability. '
                             'projectability: energy-window + projectability.')
    method_group.add_argument('--proj-threshold', type=float, default=0.9,
                        help='Projectability threshold for band selection '
                             '(default: 0.9, used with --method projectability)')
    method_group.add_argument('--num-wann', type=int, default=None,
                        help='Override number of Wannier functions '
                             '(default: auto from frontier detection)')
    stages_group.add_argument('--auto-window', action='store_true',
                        help='Stage 2: after writing .amn/.mmn, choose the '
                             'disentanglement window that minimises the '
                             'gauge-invariant spread Omega_I and patch the .win '
                             'in place (used with --method pdwf/auto). '
                             'Re-run wannier90.x afterwards.')
    method_group.add_argument('--force', action='store_true',
                        help='Skip interactive confirmation for large basis sets '
                             '(used with --method direct)')
    method_group.add_argument('--projection-method', type=str,
                        choices=['weight', 'scdm'],
                        default='weight',
                        help='Projection orbital selection method: '
                             'weight (default, simple ranking) or '
                             'scdm (SCDM-L with QR column pivoting)')

    # PDWF-specific options
    method_group.add_argument('--extended', action='store_true',
                        help='Use extended valence config (include semi-core states). '
                             'Used with --method pdwf or --method auto.')
    method_group.add_argument('--include-tm-p', action='store_true',
                        help='Include p-channel for transition metals in standard mode. '
                             'Used with --method pdwf or --method auto.')
    hybrid_group.add_argument('--hybrid-p-froz', type=float, default=None,
                        help='Hybrid method: freeze states with P_nk >= this '
                             '(per-state PDWF semantics). Default: derived '
                             'from the projectability distribution '
                             '(--hybrid-trust auto); 0.95 in manual mode.')
    hybrid_group.add_argument('--hybrid-p-floor', type=float, default=0.10,
                        help='Hybrid method: admit only states with P_nk >= '
                             'this into the disentanglement pool (default 0.10)')
    hybrid_group.add_argument('--hybrid-p-froz-band', type=float, default=None,
                        help='Hybrid method: a band must average P >= this '
                             'over k for its states to be freezable (spike '
                             'filter / gray-band gate). Default: derived '
                             'from the projectability distribution '
                             '(--hybrid-trust auto); 0.85 in manual mode.')
    hybrid_group.add_argument('--hybrid-trust', default='auto',
                        choices=['auto', 'localization', 'manual'],
                        help='Hybrid method: how to choose the freezing '
                             'thresholds. auto (default): gate at the '
                             'gray-band plateau of the band-avg P '
                             'distribution + margin, state threshold below '
                             'the trusted bands\' min_k P (full-band freeze '
                             '= best fidelity). localization: auto gate but '
                             'threshold pinned at 0.95 (smallest spread, '
                             'weakest conduction fidelity). manual: fixed '
                             'defaults 0.95/0.85 unless given explicitly. '
                             'Explicit --hybrid-p-froz/-p-froz-band always '
                             'override the derived value for that knob.')
    stages_group.add_argument('--solve-nbands', type=int, default=None,
                        metavar='N',
                        help='Solve only the lowest N bands per k-point '
                             '(LAPACK subset driver; ~2.5x faster eigensolve '
                             'at N ~ 35%% of the basis and proportionally '
                             'smaller eigenvector storage). N must cover '
                             'every band the selection needs -- the hybrid '
                             'pool guard errors out if it hits the ceiling. '
                             'Rule of thumb: N >= 2 x num_wann + 50. Output '
                             'is gauge-equivalent, not byte-identical: the '
                             'subset LAPACK driver fixes eigenvector phases '
                             'differently (.eig and all wannier90 results '
                             'agree; verified Omega_T to 7 digits on h-BN).')
    stages_group.add_argument('--no-internal-nnkp', action='store_true',
                        help='Stage 1: do NOT write the .nnkp via the '
                             'internal wannier90-exact k-mesh; defer to '
                             'wannier90.x -pp (use for exotic .win inputs '
                             'with explicit nnkpts/shell overrides, or to '
                             'cross-check against a different wannier90 '
                             'build)')
    hybrid_group.add_argument('--hybrid-thr-floor', type=float, default=0.55,
                        help='Hybrid --hybrid-trust auto: lower clip on the '
                             'derived state threshold (default 0.55, '
                             'calibrated on SnTe: chasing trusted-band '
                             'min_k P below ~0.55 over-freezes and degrades '
                             'the occupied manifold)')
    hybrid_group.add_argument('--hybrid-trust-margin', type=float, default=0.015,
                        help='Hybrid --hybrid-trust auto: margin added to '
                             'the gray-plateau top to place the band gate '
                             '(default 0.015)')
    hybrid_group.add_argument('--hybrid-dis-niter', type=int, default=2000,
                        help='Hybrid: max Gauss-Seidel sweeps for the '
                             'frozen-constrained Omega_I minimization '
                             '(default 2000; a cap-hit warning reports the '
                             'tail slope)')
    hybrid_group.add_argument('--hybrid-dis-tol', type=float, default=1e-9,
                        help='Hybrid: absolute Omega_I sweep-to-sweep '
                             'convergence tolerance in Ang^2 (default 1e-9)')
    hybrid_group.add_argument('--hybrid-dis-rel-tol', type=float, default=1e-10,
                        help='Hybrid: RELATIVE Omega_I convergence tolerance '
                             '(default 1e-10; the effective stop for '
                             'production-scale Omega_I where the absolute '
                             'tol never fires)')
    hybrid_group.add_argument('--hybrid-report', action='store_true',
                        help='Hybrid: print the sorted band-averaged '
                             'projectability table with the gray plateau, '
                             'gate, and trusted set annotated (the anatomy '
                             'the auto-trust rule reads), then continue')
    hybrid_group.add_argument('--hybrid-pool-cap', type=float, default=1.5,
                        help='Hybrid method: band-pool size FLOOR as a '
                             'multiple of num_wann (pool size = '
                             'max(ceil(factor * num_wann), seed span); the '
                             'contiguous pool covering the seed manifold; '
                             'the flag name is historical). Default 1.5: '
                             'the Omega_I optimum saturates there and '
                             'oversized pools trap the Gauss-Seidel '
                             'iteration.')
    hybrid_group.add_argument('--hybrid-shell-floor-budget', type=float,
                        default=0.01, metavar='F',
                        help='Hybrid method: the automatic shell floor is '
                             'applied only if at most this FRACTION of '
                             'Fermi-shell states sit below p_froz. Measured: '
                             'h-BN 0.00, SnTe 0.0004 (floor is right), MgB2 '
                             '0.174 (floor would strip the sigma/pi Fermi '
                             'surface, eta 18.6 -> 281.8 meV). Default 0.01 '
                             'sits 25x above SnTe and 17x below MgB2.')
    hybrid_group.add_argument('--hybrid-shell-p-floor', default='auto',
                        metavar='P',
                        help='Hybrid method: representability guard on the '
                             'metal/semimetal Fermi shell. Freeze a near-E_F '
                             'state only where its projectability is at '
                             'least P; genuinely NFE/interstitial states in '
                             'the shell (P ~ 0) are released back to the '
                             'variational subspace instead of pinning a '
                             'direction the atomic anchors cannot represent '
                             '(band-inverted semimetals otherwise freeze '
                             'every state at every k and the disentangler '
                             'goes inert). Default: unset = freeze the whole '
                             'shell unconditionally (previous behaviour).')
    hybrid_group.add_argument('--hybrid-dis-check-every', type=int, default=1,
                        metavar='N',
                        help='Hybrid method: evaluate the Omega_I '
                             'convergence monitor every N sweeps instead of '
                             'every sweep. At small nb the monitor costs as '
                             'much as the sweep itself (MgB2: 7.1 of 16 s), '
                             'so the gain is bounded by ~1.5x and shrinks '
                             'with band count; the strided test is strictly '
                             'stronger (it spans N sweeps) but costs a few '
                             'extra sweeps to trip. Default 1.')
    hybrid_group.add_argument('--hybrid-dis-mix', type=float, default=1.0,
                        metavar='M',
                        help='Hybrid method: fixed over-relaxation of the '
                             'k-local operator, Zd_eff = M*Zd_new + '
                             '(1-M)*Zd_eff_prev (mixing against the previous '
                             'EFFECTIVE Zd). M > 1 over-relaxes, which is '
                             'the useful direction; damping (M < 1) measured '
                             'strictly worse upstream. Default 1.0 = the '
                             'unmixed iteration, bit-identical to before.')
    hybrid_group.add_argument('--hybrid-dis-mix-schedule', type=float, default=None,
                        metavar='PEAK',
                        help='Hybrid method: SCHEDULED over-relaxation -- '
                             'start at PEAK and taper geometrically toward '
                             '1.0 (see --hybrid-dis-taper/-taper-on). '
                             'Overrides --hybrid-dis-mix. PEAK is POSITIVE: '
                             'upstream writes the same setting as a negative '
                             '"relax" because there the minus sign is a '
                             '"schedule" sentinel, but here a negative value '
                             'is consumed literally and produces garbage. '
                             'WARNING: the relaxation selects WHICH Omega_I '
                             'minimum the run lands in on materials with '
                             'several (measured on h-BN: peak 2.5 under the '
                             'legacy trigger converges to an Omega_I 7%% '
                             'worse, reporting success). Compare at least '
                             'two peaks before trusting a number. Mutually '
                             'exclusive with Anderson acceleration (not '
                             'ported: the two anti-compose).')
    hybrid_group.add_argument('--hybrid-dis-relax-late', type=float, default=None,
                        metavar='A',
                        help='Hybrid method: RESTART the scheduled relaxation '
                             'at A once the geometric taper has spent itself '
                             '(alpha-1 < 1%%), instead of finishing the run '
                             'unrelaxed. Requires --hybrid-dis-mix-schedule. '
                             'Must be < 2 (at 2 the iteration reports a '
                             'DEGRADED minimum as converged; above it it '
                             'diverges). Upstream default 1.99 on LAPW; the '
                             'best value for LCAO is being measured and may '
                             'differ.')
    # Live convergence trace for the disentangler -- one flushed line per
    # Omega_I check with sweep, alpha, Omega_I, dOmega, the observed
    # contraction rho and the extrapolated remaining gain. Default:
    # <seedname>_disentangle.log. Pass an empty string to disable. Written
    # and flushed as the run proceeds, so a multi-hour disentanglement can be
    # followed live (stdout is block-buffered when piped and will NOT show
    # progress).
    dev_group.add_argument('--hybrid-dis-log', type=str, default=None,
                        metavar='PATH', help=argparse.SUPPRESS)
    hybrid_group.add_argument('--hybrid-dis-taper', type=float, default=0.5,
                        metavar='F',
                        help='Hybrid method: geometric taper factor for '
                             '--hybrid-dis-mix-schedule, cur = 1 + F*(cur-1). '
                             'Default 0.5 (upstream). F and the peak are '
                             'COUPLED: a peak above 2 survives only if the '
                             'taper drops it under 2 in one step, so '
                             'peak > 2 with F > 0.6 is refused at entry '
                             '(measured upstream: peak 2.5 with F=0.8 '
                             'diverges to Omega_I 181.6 in 200 sweeps).')
    hybrid_group.add_argument('--hybrid-dis-taper-on', choices=['check', 'stall'],
                        default='check',
                        help='Hybrid method: when the schedule tapers. '
                             '"check" (default, upstream semantics) = every '
                             'Omega_I check that did not converge; the '
                             'schedule therefore spends itself in ~10 checks '
                             'and acts as an OPENING accelerator. "stall" = '
                             'only on checks that failed to improve (this '
                             'code before v1.6), which holds the peak far '
                             'longer and is faster, but was measured to land '
                             'h-BN in a 7%%-worse minimum at peak 2.5. '
                             'Numbers measured under one trigger are NOT '
                             'comparable to the other.')
    # Drop the per-k mixing history at each taper event (this code before
    # v1.6; upstream never does). With the default "check" trigger this
    # makes over-relaxation a silent NO-OP -- the history is cleared before
    # it can ever be used -- so it is only meaningful with
    # --hybrid-dis-taper-on stall, for reproducing pre-v1.6 runs.
    dev_group.add_argument('--hybrid-dis-taper-reset-history',
                        action='store_true', help=argparse.SUPPRESS)
    hybrid_group.add_argument('--hybrid-require-capacity', action='store_true',
                        help='Hybrid method: abort instead of warning when '
                             'num_wann is below the frozen-set demand (the '
                             'per-k cap would silently discard frozen states '
                             'the mask rules asked for, giving a converged '
                             'model with quietly wrong bands). Off by '
                             'default: the truncation is reported either '
                             'way.')
    # Refuse to write the .amn when the SCDM gauge is rank-deficient, i.e.
    # when sigma_max/sigma_min exceeds 1/TOL at some k. Scale-free, so it is
    # safe across systems. Default 1e-10 (catches only genuine singularity);
    # raise to tighten.
    dev_group.add_argument('--hybrid-gauge-rank-tol', type=float,
                        default=1e-10, help=argparse.SUPPRESS)
    # Additionally refuse to write the .amn when the gauge worst-k minimum
    # singular value falls below SIGMA. OFF by default on purpose: absolute
    # sigma_min is not comparable across systems (measured 0.39 on Bi, 0.036
    # on h-BN, <5e-4 on Sc/294 WFs, all giving good models), so calibrate per
    # material before using this.
    dev_group.add_argument('--hybrid-gauge-min-sv', type=float, default=None,
                        metavar='SIGMA', help=argparse.SUPPRESS)
    hybrid_group.add_argument('--augment-coverage-min', type=float, default=0.70,
                        help='Hybrid method: low-conduction coverage <P> '
                             'below which conduction-driven target '
                             'augmentation is attempted (mean Lowdin P of '
                             'the 4 lowest conduction bands). Default 0.70.')
    hybrid_group.add_argument('--augment-retention-floor', type=float, default=0.40,
                        help='Hybrid target augmentation: drop candidate '
                             'channels whose mean on-site Lowdin retention '
                             '|S^(1/2)_mumu|^2 is below this floor (near-'
                             'linear-dependence ghosts, e.g. diffuse f). '
                             'Default 0.40.')
    # EMISSION BASELINE (hybrid method only, mutually exclusive with the
    # normal pre-disentangled hand-off): baseline for comparing stock
    # Wannier90 projectability disentanglement
    # (dis_froz_proj/dis_proj_min/max, PDWF; Qiao et al. 2023) against the
    # hybrid method. Writes the FULL hybrid band pool (.eig/.mmn
    # uncompressed, num_bands = pool size), a .win carrying the
    # dis_froz_proj keywords and no energy windows, and a Lowdin
    # target-projection .amn. dis-froz-proj: per-state projectability equals
    # ours EXACTLY (wannier90 requires num_proj == num_wann, so this implies
    # --pdwf-first-radial-only; augmented targets abort). dis-froz-proj-svd:
    # the full multi-zeta target, SVD-compressed to num_wann columns per k
    # (PDWF A construction on Lowdin rows) -- the stock machinery's
    # best-case configuration.
    dev_group.add_argument('--baseline', type=str, default=None,
                        choices=['dis-froz-proj', 'dis-froz-proj-svd'],
                        help=argparse.SUPPRESS)
    # --baseline dis-froz-proj: dis_proj_min written to the .win (states
    # with projectability below it are excluded). Default 0.01 (PDWF paper).
    dev_group.add_argument('--dis-proj-min', type=float, default=0.01,
                        help=argparse.SUPPRESS)
    # --baseline dis-froz-proj: dis_proj_max written to the .win (states
    # with projectability above it are frozen). Default 0.95 (PDWF paper).
    dev_group.add_argument('--dis-proj-max', type=float, default=0.95,
                        help=argparse.SUPPRESS)
    # Save the per-state anatomy (pool P_nk, eigenvalues, frozen/admitted
    # masks, pool indices, k-points, thresholds) as .npz -- the data behind
    # the auto-trust figure and the non-representability certificate.
    dev_group.add_argument('--dump-masks', type=str, default=None,
                        metavar='PATH.npz', help=argparse.SUPPRESS)
    # Save the hand-off Bloch basis B(k) (AO x N_w coefficients in the
    # GTO-orthonormalized frame, the states the emitted .mmn/.amn/.eig
    # describe) plus k-points and hand-off eigenvalues -- enables real-space
    # Wannier-function evaluation (scripts/wf_isosurface.py).
    dev_group.add_argument('--dump-handoff', type=str, default=None,
                        metavar='PATH.npz', help=argparse.SUPPRESS)
    # Write the --hybrid-report table as JSON (per pool band: Pbar, minP,
    # minE_rel_EF, relevant, cls; plus the derived thresholds). Schema
    # consumed by paper/figures_src/fig3_anatomy.py.
    dev_group.add_argument('--hybrid-report-json', type=str, default=None,
                        metavar='PATH.json', help=argparse.SUPPRESS)
    hybrid_group.add_argument('--hybrid-guiding-centres', action='store_true',
                        help='Hybrid method: after the SCDM gauge is built, '
                             'rewrite the .win projections block with the '
                             'anchor host-atom centres and set '
                             'guiding_centres = .true. (anchors are real '
                             'AO sites by construction, so the random-'
                             'centre hazard does not apply). Default off.')
    # pdwf/hybrid: target only the FIRST radial of each valence l-channel
    # (num_target == num_wann exactly) instead of all radials. Required for
    # --baseline dis-froz-proj.
    dev_group.add_argument('--pdwf-first-radial-only', action='store_true',
                        help=argparse.SUPPRESS)
    hybrid_group.add_argument('--hybrid-fid-emax', type=float, default=8.0,
                        help='Hybrid method, --hybrid-trust auto: fidelity '
                             'energy ceiling in eV above E_F. Bands dipping '
                             'below it are trust candidates; bands entirely '
                             'above it define the gray plateau the gate must '
                             'clear (default 8.0)')
    hybrid_group.add_argument('--hybrid-froz-emax', type=_froz_emax_arg, default='auto',
                              metavar='auto|none|EV',
                              help='frozen ceiling above E_F: the frozen set is every '
                                   'pool state below it (cap + fill, as wien2wannier '
                                   '--froz-emax). auto (default) walks down from the fidelity ceiling until the '
                                   'frozen demand fits num_wann; EV sets it; none keeps '
                                   'the per-state trust mask as the frozen set')
    method_group.add_argument('--pdwf-p-high', type=_float_or_auto, default='auto',
                        help='PDWF frozen threshold on the band-averaged '
                             'projectability; auto (default) = the band gate '
                             'the hybrid rule derives (gray plateau + margin)')
    method_group.add_argument('--pdwf-p-low', type=_float_or_auto, default='auto',
                        help='PDWF excluded threshold; auto (default) = the '
                             'junk cliff below the gate (largest jump in the '
                             'sorted band-averaged projectabilities)')

    # Stage 3 arguments (post-Wannierization H(R) postprocessing)
    stage3_group = parser.add_argument_group('Stage 3 options',
                                              'Post-Wannierization H(R) postprocessing')
    stage3_group.add_argument('--hr-file', type=str, default=None,
                              help='Path to wannier90_hr.dat for stage 3 '
                                   '(default: {seedname}_hr.dat)')
    stage3_group.add_argument('--output', '-o', type=str, default=None,
                              help='Output filename for the postprocessed HR file '
                                   '(default: {hr_file}_postprocessed)')
    stage3_group.add_argument('--no-hermitize', action='store_true',
                              help='Skip Hermitization step (stage 3)')
    stage3_group.add_argument('--no-time-reversal', action='store_true',
                              help='Skip time-reversal symmetry enforcement (stage 3)')
    stage3_group.add_argument('--symm-threshold', type=float, default=1e-9,
                              help='Threshold for dropping small hoppings in '
                                   'the postprocessed output (default: 1e-9)')

    # Stage 4 arguments (band structure plot)
    stage4_group = parser.add_argument_group('Stage 4 options',
                                              'Band structure plotting')
    stage4_group.add_argument('--kpath', type=str,
                              choices=['auto', 'hexagonal_2d', 'hexagonal_3d',
                                       'fcc', 'bcc', 'sc', 'custom'],
                              default='auto',
                              help='K-path type (default: auto-detect from lattice)')
    stage4_group.add_argument('--npts', type=int, default=60,
                              help='Number of k-points per segment (default: 60)')
    stage4_group.add_argument('--energy-range', type=float, nargs=2,
                              metavar=('E_MIN', 'E_MAX'), default=None,
                              help='Energy range relative to E_F in eV '
                                   '(default: -20 25)')
    stage4_group.add_argument('--output-plot', type=str, default=None,
                              help='Output plot filename '
                                   '(default: {seedname}_bands.png)')
    stage4_group.add_argument('--no-pdwf', action='store_true',
                              help='Skip PDWF projectability coloring '
                                   '(uniform color bands)')
    stage4_group.add_argument('--custom-kpath', type=str, default=None,
                              help='Custom k-path string: '
                                   '"G:0,0,0;M:0.5,0,0;K:0.333,0.333,0;G:0,0,0"')

    args = parser.parse_args(argv)

    if args.method == 'window':
        # the automatic window route: PDWF classification at the thresholds
        # the hybrid rule derives, Omega_I-optimal windows, stock Wannier90
        # disentanglement
        args.method = 'pdwf'
        args.auto_window = True
        args.window_route = True

    if getattr(args, 'baseline', None) and args.method != 'hybrid':
        parser.error(f"--baseline {args.baseline} is only valid with "
                     "--method hybrid (it is a baseline FOR the hybrid "
                     "method's emission)")

    # Used by the canonical front end to validate every advanced option before
    # a dry run/check returns or an output directory is created.  Argument
    # parsing and cross-option checks above are deliberately the only work.
    if getattr(args, 'lcao_parse_only', False):
        return args

    stage_fns = {
        1: stage1_create_win,
        2: stage2_create_data_files,
        3: stage3_symmetrize_hr,
        4: stage4_plot_bands,
        'all': stage_all_run,
    }
    stage_fn = stage_fns[args.stage]

    # Resolve spin handling: collinear spin-polarized stage 1/2 runs may expand
    # into two independent (alpha, beta) runs; everything else runs once.
    spin_runs = _resolve_spin_runs(args, parser)

    actual_seednames = []
    for spin_label, spin_channel, seedname in spin_runs:
        run_args = args
        if spin_label is not None:
            run_args = copy.copy(args)
            run_args.seedname = seedname
            run_args._spin_channel = spin_channel
            run_args._spin_label = spin_label
            if len(spin_runs) > 1:
                print("\n" + "#" * 72)
                print(f"# SPIN-POLARIZED RUN: {spin_label.upper()} channel "
                      f"-> seedname '{seedname}'")
                print("#" * 72)
        stage_fn(run_args)
        actual_seednames.append(seedname)
    return actual_seednames


if __name__ == '__main__':
    main()
