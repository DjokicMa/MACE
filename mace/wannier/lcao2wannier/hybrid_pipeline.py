# Copyright (c) 2025 William Comaskey. MIT License, see LICENSE in this directory.
# Vendored into MACE from lcao2wannier 1.0.0; local changes are listed in VENDORED.md.
"""Stage-2 driver for the hybrid PDWF-subspace + SCDM-gauge method.

Runs the in-pipeline per-state disentanglement (lcao2wannier.hybrid) on the
engine's solved LCAO data and writes a pre-disentangled, isolated-manifold
problem (num_bands == num_wann, no dis_* keywords) for stock wannier90 3.1:
.eig = projected-H eigenvalues, .mmn = W^H M W (analytic GTO overlaps),
.amn = AO-SCDM gauge.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import gto_mmn as _gto
from .hybrid import (select_pool_and_frozen, auto_trust_thresholds,
                     degeneracy_regime, symmetrize_degenerate,
                     disentangle_frozen, scdm_anchors, rotate_to_subspace,
                     mv_weights, omega_invariant, _select_pool,
                     audit_pool_coverage)
from .kpoints import read_nnkp_neighbors
from .wannier90 import _format_complex_block


def select_hybrid_parent_pool(engine, pool_factor=1.5, *, hybrid_preprocess=False):
    """Return the exact pre-disentanglement band pool used by hybrid.

    This is the shared selection boundary for raw-parent and baseline
    emission.  It intentionally delegates to :func:`hybrid._select_pool` so
    the named ``overlaps`` stage cannot acquire a separate selection policy.
    """
    proj = getattr(engine, '_band_projectability', None)
    if proj is None:
        raise RuntimeError("hybrid parent-pool emission needs the PDWF "
                           "projectability (engine._band_projectability)")
    proj = np.asarray(proj, float)
    eig_full = np.array([np.asarray(e) for e in engine.eigenvalues_list])
    e_fermi = engine.e_fermi if engine.e_fermi is not None else 0.0
    if hybrid_preprocess:
        reg = degeneracy_regime(eig_full)
        if reg['band_level']:
            proj = symmetrize_degenerate(proj, eig_full, dtol=reg['dtol'])
    pool = _select_pool(proj, eig_full, engine.num_wann,
                        pool_factor=pool_factor, e_fermi=e_fermi)
    nb_solved = proj.shape[1]
    if pool[-1] >= nb_solved - 1 and nb_solved < engine.num_orbitals:
        raise RuntimeError(
            f"hybrid pool [{pool[0]}..{pool[-1]}] hits the solve ceiling "
            f"({nb_solved} of {engine.num_orbitals} bands solved). "
            f"Raise --solve-nbands or drop it.")
    return pool


def _neighbors_from_nnkp(engine, seedname):
    """(neigh (nk,nntot), gshift (nk,nntot,3), bcart0 (nntot,3) 1/Ang)."""
    nl = read_nnkp_neighbors(f"{seedname}.nnkp", engine.recip_lattice,
                             engine.kpoints)
    nk = engine.num_kpoints
    nntot = len(nl[0])
    neigh = np.empty((nk, nntot), int)
    gshift = np.empty((nk, nntot, 3), int)
    for k in range(nk):
        for i, nb in enumerate(nl[k]):
            neigh[k, i] = nb['id']
            gshift[k, i] = np.asarray(nb['G_shift'], int)
    bcart0 = np.array([nl[0][i]['b_vec_cart'] for i in range(nntot)])
    return neigh, gshift, bcart0


def _analytic_pool_overlaps(engine, pool, neigh, gshift, verbose=True):
    """In-memory analytic-GTO M(k,b) for the pool bands + GTO-orthonormal C.

    Mirrors the analytic branch of write_mmn_file_lcao: precompute S^(b)(g)
    per unique b (b=0 included), orthonormalize C_pool against the b=0 GTO
    metric (M(k,0)=I so all singular values <= 1), then
    M(k,b) = C^H(k) Stilde^(b)(k+b) C(k+b).
    """
    gto_aos = getattr(engine, 'gto_aos', None)
    if gto_aos is None:
        raise RuntimeError(
            "hybrid method requires the analytic GTO basis on the engine "
            "(engine.gto_aos); run with --mmn-method analytic")
    cutoff = getattr(engine, 'gto_cutoff', 16.0)
    kpoints = engine.kpoints
    nk, nntot = neigh.shape

    lattice_bohr = np.asarray(engine.lattice_vectors, float) * _gto.ANG2BOHR
    recip_bohr = _gto.reciprocal_bohr(lattice_bohr)
    cells_R = [(g, np.asarray(g, float) @ lattice_bohr)
               for g in engine.real_space_matrices]

    # SHARED analytic-MMN helpers (gto_mmn): b-table, S^(b) precompute, and
    # GTO-metric orthonormalization are the same implementation the .mmn
    # write path uses — the conventions cannot drift between the two.
    bkey_of, unique_b, b0_key = _gto.build_b_table(
        kpoints,
        ((k, i, int(neigh[k, i]), gshift[k, i])
         for k in range(nk) for i in range(nntot)),
        recip_bohr)
    from .wannier90 import _mmn_choose_n_proc
    Sb_by_b = _gto.precompute_Sb_parallel(
        gto_aos, unique_b, cells_R, cutoff=cutoff,
        n_proc=_mmn_choose_n_proc(nk, nntot, len(gto_aos)), verbose=verbose)

    C_pool = [np.asarray(C)[:, pool].copy() for C in engine.eigenvectors_list]
    N = len(gto_aos)
    if verbose and C_pool[0].shape[0] == 2 * N:
        print(f"  [hybrid] 2-component SOC basis: {N} spatial AOs x 2 spinors")
    C_pool = _gto.orthonormalize_gto_metric(C_pool, Sb_by_b[b0_key],
                                            kpoints, N)

    npool = len(pool)
    mmn = np.empty((nk, nntot, npool, npool), complex)
    for k in range(nk):
        for i in range(nntot):
            kpb = kpoints[neigh[k, i]] + gshift[k, i]
            St = _gto.Sb_at_kpb(Sb_by_b[bkey_of[(k, i)]], kpb)
            mmn[k, i] = _gto.gto_block_overlap(C_pool[k], St,
                                               C_pool[neigh[k, i]], N)
    # The b=0 GTO metric is ALSO returned: C_pool is orthonormal against THIS
    # metric, not against engine.S_k_list (they differ at ~8e-4 for h-BN).
    # Any operator built on these states must use the same metric.
    return mmn, C_pool, (Sb_by_b[b0_key], N)


def _write_eig(path, eig):
    nk, nw = eig.shape
    with open(path, 'w') as f:
        for k in range(nk):
            for n in range(nw):
                f.write(f"{n + 1:5d}{k + 1:5d}{eig[k, n]:18.12f}\n")


def _write_amn(path, A, header="hybrid PDWF+SCDM gauge"):
    nk, nb, nw = A.shape
    with open(path, 'w') as f:
        f.write(header + "\n")
        f.write(f"{nb:12d}{nk:12d}{nw:12d}\n")
        for k in range(nk):
            for n in range(nw):
                for m in range(nb):
                    a = A[k, m, n]
                    f.write(f"{m + 1:5d}{n + 1:5d}{k + 1:5d}"
                            f"{a.real:18.12f}{a.imag:18.12f}\n")


def _write_mmn(path, mmn, neigh, gshift, header="hybrid pre-disentangled MMN"):
    nk, nntot, nw, _ = mmn.shape
    with open(path, 'w') as f:
        f.write(header + "\n")
        f.write(f"{nw:12d}{nk:12d}{nntot:12d}\n")
        for k in range(nk):
            for i in range(nntot):
                g = gshift[k, i]
                f.write(f"{k + 1:5d}{neigh[k, i] + 1:5d}"
                        f"{g[0]:5d}{g[1]:5d}{g[2]:5d}\n")
                f.write(_format_complex_block(mmn[k, i], order='F'))


def _parent_projection_amn(engine, pool, amn_mode):
    """Build Lowdin parent projections without changing the band-row space.

    ``raw-full`` retains every target-AO column because these files are an
    inspectable pre-hybrid intermediate.  Baseline modes retain their existing
    Wannier90-consumer constraints and roundoff clipping.
    """
    from .lcao_pdwf import compute_matrix_sqrt
    tmask = np.asarray(engine.pdwf_target_mask, bool)
    num_wann = int(engine.num_wann)
    n_target = int(tmask.sum())
    nproj = num_wann if amn_mode == 'svd' else n_target
    nk = len(engine.eigenvalues_list)
    amn = np.empty((nk, len(pool), nproj), complex)
    for k in range(nk):
        S_half = compute_matrix_sqrt(np.asarray(engine.S_k_list[k]))
        C_tilde = S_half @ np.asarray(engine.eigenvectors_list[k])
        A_full = C_tilde[tmask][:, pool].T
        if amn_mode == 'svd':
            U, s, _Vh = np.linalg.svd(A_full, full_matrices=False)
            amn[k] = U[:, :num_wann] * s[:num_wann][None, :]
        elif amn_mode == 'raw-full':
            # AMN stores bra amplitudes <psi_m|g_n>.  Conjugation is
            # load-bearing: under a pool gauge C -> C U these rows transform
            # as U^H A, matching M -> U^H M U and the bridge consumer's
            # C_pool = amn.conj().T reconstruction.
            amn[k] = A_full.conj()
        else:
            amn[k] = A_full
        if amn_mode != 'raw-full':
            # Stock Wannier90 rejects p_mk > 1 exactly. Keep this baseline-only
            # quantization guard out of raw data so its row norms remain the
            # exact hybrid projectability input.
            norms2 = (np.abs(amn[k]) ** 2).sum(axis=1)
            over = norms2 > 1.0 - 1e-10
            if over.any():
                amn[k][over] *= np.sqrt((1.0 - 1e-10) / norms2[over, None])
    return amn


def _print_trust_report(proj, eig_full, num_wann, e_fermi, fid_emax,
                        p_froz, p_froz_band, auto, pool_factor=1.5):
    """The projectability anatomy the auto-trust rule reads: pool bands
    sorted by band-averaged P, with the fidelity split, gray plateau, gate,
    and trusted set annotated. Diagnostic only — no effect on the run."""
    from .hybrid import _select_pool
    pool = _select_pool(proj, eig_full, num_wann, pool_factor=pool_factor,
                        e_fermi=e_fermi)
    avg_p = proj.mean(axis=0)
    min_p = proj.min(axis=0)
    avg_e = eig_full.mean(axis=0) - e_fermi
    bot_e = eig_full.min(axis=0) - e_fermi
    top_e = eig_full.max(axis=0) - e_fermi
    trusted = set(int(b) for b in auto['trusted']) if auto else set()
    plateau_top = auto['plateau_top'] if auto else None

    print("\n  [trust report] pool bands sorted by band-averaged P "
          f"(fidelity ceiling E_F{fid_emax:+g} eV; gate {p_froz_band:.3f}, "
          f"threshold {p_froz:.3f}):")
    print(f"    {'band':>5} {'<E>':>8} {'bot':>8} {'top':>8} "
          f"{'<P>':>7} {'min_k P':>8}   tags")
    order = pool[np.argsort(avg_p[pool])[::-1]]
    for b in order:
        tags = []
        if int(b) in trusted:
            tags.append("TRUSTED" + (" (full-band freeze)"
                                     if min_p[b] >= p_froz else ""))
        elif bot_e[b] > fid_emax:
            tags.append("continuum")
            if plateau_top is not None and abs(avg_p[b] - plateau_top) < 1e-12:
                tags.append("<-- gray-plateau top (sets the gate)")
        print(f"    {b:>5} {avg_e[b]:>8.2f} {bot_e[b]:>8.2f} "
              f"{top_e[b]:>8.2f} {avg_p[b]:>7.3f} {min_p[b]:>8.3f}   "
              f"{' '.join(tags)}")
    print()


#: frozen-ceiling walk (wien2wannier apply_froz_emax ladder): rungs 1 eV apart
#: down to FROZ_WALK_LO; below the fidelity ceiling the rungs are the integer
#: ladder E_fid, E_fid-1, ... so both programs visit the same ceilings.
FROZ_WALK_STEP = 1.0
FROZ_WALK_LO = 1.0
#: where 'auto' starts: 'fid' = E_fid (default; wien2wannier's rule), 'top' =
#: max(E_fid, top of the trusted frozen set). The ceiling study
#: (calculations/ceiling_study) tested 'top' on every cell where the two
#: differ: better h-BN conduction near and above E_F+8, worse bcc Fe (both
#: spins, both windows) and MgB2 Fermi window on LAPW. An extension beyond
#: E_fid is an explicit --hybrid-froz-emax <eV>; the log prints the trusted top.
FROZ_WALK_START_DEFAULT = 'fid'


def resolve_frozen_ceiling(eps_pool, trust_frozen, num_wann, mode='auto',
                           fid_emax=8.0, start=None):
    """The frozen ceiling E_ceil (eV rel. E_F) for the cap + fill.

    mode = <float>  explicit ceiling (the user's; not walked).
    mode = 'auto'   walk down a ladder until the frozen DEMAND fits,
                    demand(fe) = max_k #{pool states with eps <= fe} <= num_wann.
                    Rungs: start, start-1, ... while above fid_emax, then
                    fid_emax, fid_emax-1, ..., FROZ_WALK_LO. start = fid_emax
                    ('fid') or max(fid_emax, top of the trusted set) ('top').
    The walk refuses (RuntimeError) when no rung fits, like wien2wannier:
    freezing num_wann states at every k by truncation leaves no free
    direction and a zero gradient that reports success.
    No saturation test: an isolated target manifold legitimately freezes all
    num_wann states at every k (Bi bilayer, 16 of 16).
    """
    eps_pool = np.asarray(eps_pool, float)
    trust = np.asarray(trust_frozen, bool)
    E_top = float(eps_pool[trust].max()) if trust.any() else float(fid_emax)
    srt = np.sort(eps_pool, axis=1)
    sup = (float(srt[:, num_wann].min()) if eps_pool.shape[1] > num_wann
           else float('inf'))

    def demand(fe):
        return int((eps_pool <= fe + 1e-9).sum(axis=1).max())

    if str(mode).lower() != 'auto':
        fe = float(mode)
        return dict(mode=f'{fe:g} eV', E_ceil=fe, E_top=E_top, sup=sup,
                    start=None, start_value=fe, trail=[(fe, demand(fe))])
    start = start or FROZ_WALK_START_DEFAULT
    if start not in ('fid', 'top'):
        raise ValueError(f"froz_walk_start {start!r}: use 'fid' or 'top'")
    s0 = float(fid_emax) if start == 'fid' else max(float(fid_emax), E_top)
    rungs = []
    fe = s0
    while fe > float(fid_emax) + 1e-9:
        rungs.append(fe)
        fe -= FROZ_WALK_STEP
    fe = float(fid_emax)
    while fe >= FROZ_WALK_LO - 1e-9:
        rungs.append(fe)
        fe -= FROZ_WALK_STEP
    trail = []
    for fe in rungs:
        d = demand(fe)
        trail.append((fe, d))
        if d <= num_wann:
            return dict(mode='auto', E_ceil=fe, E_top=E_top, sup=sup,
                        start=start, start_value=s0, trail=trail)
    raise RuntimeError(
        f"frozen ceiling auto: no rung from E_F{s0:+.2f} down to "
        f"E_F{FROZ_WALK_LO:+.2f} eV fits num_wann={num_wann} (demand "
        f"{trail[-1][1]} at the floor). Any ceiling below the exact supremum "
        f"E_F{sup:+.4f} eV fits (= min_k eps_(num_wann+1)); pass it with "
        f"--hybrid-froz-emax, raise num_wann, or use --hybrid-froz-emax none.")


def apply_frozen_ceiling(frozen, admit, eps_pool, E_ceil, num_wann):
    """Cap + fill in place: frozen(k) := pool states with eps <= E_ceil (the
    num_wann deepest where more sit below), admitted as well. Returns counts
    relative to the incoming (trust) mask."""
    eps_pool = np.asarray(eps_pool, float)
    trust = np.asarray(frozen, bool).copy()
    new = np.zeros_like(trust)
    n_over = 0
    for k in range(eps_pool.shape[0]):
        idx = np.where(eps_pool[k] <= E_ceil + 1e-9)[0]
        if len(idx) > num_wann:
            idx = idx[np.argsort(eps_pool[k, idx], kind='stable')[:num_wann]]
            n_over += 1
        new[k, idx] = True
    frozen[:, :] = new
    admit[:, :] = np.asarray(admit, bool) | new
    return dict(added=int((new & ~trust).sum()),
                capped=int((trust & ~new).sum()), n_over=n_over)


def write_masks_npz(path, pool, proj_pool, eig_pool, admit, frozen,
                    kpoints_frac, e_fermi, p_froz, p_gate, p_floor,
                    fid_emax, fallback_used, frozen_trust=None, E_ceil=None,
                    froz_emax_mode='none'):
    """--dump-masks payload: the per-state anatomy of a hybrid run (.npz).

    Feeds the auto-trust anatomy figure and the non-representability
    certificate (per-state data the pipeline otherwise only prints). Arrays
    are in the POOL band basis (column j = pool_band_indices[j]):

      P_nk            (nk, npool)  Lowdin projectability of the pool states
      eps_nk          (nk, npool)  eigenvalues, eV relative to E_F
      eps_nk_raw      (nk, npool)  eigenvalues, absolute eV
      frozen_mask     (nk, npool)  bool, the frozen set the disentangler used
                      (after the frozen ceiling: pool states below E_ceil)
      frozen_trust_mask (nk, npool) bool, the per-state trust mask before the
                      ceiling (what the certificate tests); equals frozen_mask
                      when froz_emax = none
      E_ceil, froz_emax_mode   the applied ceiling (eV rel. E_F; NaN if none)
      admitted_mask   (nk, npool)  bool, frozen implies admitted
      pool_band_indices (npool,)   0-based band indices into the solved set
      kpoints_frac    (nk, 3)      fractional k-points
      e_fermi, p_froz, p_gate, p_floor, E_fid   scalars (p_gate is the
                      band-average gate, a.k.a. p_froz_band)
      fallback_used   bool — True when the auto-trust rule was attempted but
                      undecidable, so the fixed defaults (0.95/0.85) were used
    """
    pool = np.asarray(pool, int)
    eig_pool = np.asarray(eig_pool, float)
    np.savez(
        path,
        P_nk=np.asarray(proj_pool, float),
        eps_nk=eig_pool - float(e_fermi),
        eps_nk_raw=eig_pool,
        frozen_mask=np.asarray(frozen, bool),
        admitted_mask=np.asarray(admit, bool),
        pool_band_indices=pool,
        kpoints_frac=np.asarray(kpoints_frac, float),
        e_fermi=float(e_fermi),
        p_froz=float(p_froz),
        p_gate=float(p_gate),
        p_floor=float(p_floor),
        E_fid=float(fid_emax),
        fallback_used=bool(fallback_used),
        frozen_trust_mask=np.asarray(frozen if frozen_trust is None
                                     else frozen_trust, bool),
        E_ceil=float('nan') if E_ceil is None else float(E_ceil),
        froz_emax_mode=str(froz_emax_mode),
    )
    print(f"  [dump-masks] wrote per-state masks -> {path}")


def write_report_json(path, proj, eig_full, num_wann, e_fermi, fid_emax,
                      p_froz, p_froz_band, auto, pool_factor=1.5,
                      label=None):
    """--hybrid-report-json: the --hybrid-report table, machine-readable.

    Consumer: paper/figures_src/fig3_anatomy.py (the auto-trust anatomy
    figure). Schema — one row per POOL band, sorted by band index:

      bands: [{band, Pbar, minP, minE_rel_EF, relevant (0/1),
               cls ("trusted"|"gray"|"junk")}, ...]
      thresholds: {p_gate, p_froz, plateau, label, champion}

    relevant = 1 iff min_k eps <= E_F + fid_emax (fidelity-relevant).
    cls: "trusted" = relevant and Pbar >= p_gate (the trusted set the rule
    freezes); "gray" = untrusted but Pbar >= p_froz (projectable enough to
    fool the per-state threshold — exactly what the band gate blocks);
    "junk" = Pbar < p_froz. plateau is the gray-plateau top the auto rule
    read (null when the rule fell back or was overridden). champion is a
    hand-annotation (the per-material hand-tuned reference thresholds) that
    is not derivable from a run — always null here; fill it in by hand for
    the figure if wanted. label defaults to the seedname basename.
    """
    proj = np.asarray(proj, float)
    eig_full = np.asarray(eig_full, float)
    pool = _select_pool(proj, eig_full, num_wann, pool_factor=pool_factor,
                        e_fermi=e_fermi)
    avg_p = proj.mean(axis=0)
    min_p = proj.min(axis=0)
    bot_e = eig_full.min(axis=0) - e_fermi

    bands = []
    for b in pool:
        b = int(b)
        relevant = int(bot_e[b] <= fid_emax)
        if relevant and avg_p[b] >= p_froz_band:
            cls = "trusted"
        elif avg_p[b] >= p_froz:
            cls = "gray"
        else:
            cls = "junk"
        bands.append({
            "band": b,
            "Pbar": float(avg_p[b]),
            "minP": float(min_p[b]),
            "minE_rel_EF": float(bot_e[b]),
            "relevant": relevant,
            "cls": cls,
        })
    payload = {
        "bands": bands,
        "thresholds": {
            "p_gate": float(p_froz_band),
            "p_froz": float(p_froz),
            "plateau": (float(auto["plateau_top"]) if auto else None),
            "label": label,
            "champion": None,
        },
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
        f.write("\n")
    print(f"  [report-json] wrote projectability anatomy -> {path}")


def run_baseline_dis_froz_proj(engine, seedname, pool_factor=1.5,
                               amn_mode='lowdin', verbose=True,
                               _purpose='baseline', _hybrid_preprocess=False):
    """Stock-Wannier90 dis_froz_proj baseline emission (--baseline dis-froz-proj).

    The controlled experiment referees will demand (Referee_Risk_Assessment
    A1): stock wannier90's own projectability disentanglement (dis_froz_proj
    / dis_proj_min / dis_proj_max, introduced with PDWF) on EXACTLY the
    hybrid method's inputs. Builds the hybrid band pool (same contiguous
    pool, same cap, same augmented target set) but SKIPS the in-pipeline
    disentanglement, the num_bands = num_wann hand-off compression, and the
    SCDM gauge; wannier90 receives the full pool and disentangles it itself:

      .eig  pool-band eigenvalues, eV rel E_F (num_bands = pool size)
      .mmn  the UNCOMPRESSED analytic-GTO pool overlaps — the same M(k,b)
            the hybrid path computes before the W^H M W compression
            (shared implementation: _analytic_pool_overlaps)
      .amn  Lowdin target projection A[k, m, n] = [S(k)^{1/2} C(k)]_{T_n, pool_m}.
            This choice is load-bearing: wannier90 derives the per-state
            projectability as p_mk = sum_n |A_mn|^2, which with Lowdin rows
            equals compute_lowdin_projectability's P_mk exactly — baseline
            and hybrid see identical projectability data.

    The matching .win (num_bands = pool, dis_froz_proj = .true.,
    dis_proj_min/max, no energy windows) is written at stage 1 by
    _apply_baseline_overrides.

    Limitation (amn_mode='lowdin'): wannier90 requires num_proj ==
    num_wann, so the target mask must contain exactly num_wann AOs. The
    CLI therefore implies --pdwf-first-radial-only for this mode
    (multi-zeta rows are not emittable); if channel augmentation still
    grows the target past num_wann, the run aborts with a clear message
    rather than silently changing the experiment.

    amn_mode='svd' (--baseline dis-froz-proj-svd) lifts the first-radial
    restriction: the FULL (multi-zeta) Lowdin target projection
    [S^{1/2}C]_{T, pool} is SVD-compressed to its best num_wann-dimensional
    subspace per k (the PDWF-paper A construction, on Lowdin rows), so
    stock wannier90 sees the complete target chemistry through num_wann
    columns. The implied per-state projectability sum_n |A_mn|^2 is then
    the norm of the state's projection onto the top-num_wann right-singular
    subspace: <= the full-target P_mk (hence still within wannier90's hard
    [0, 1] check), and ~equal to it for states the target represents well.
    This is the stock machinery's best-case configuration — use it for the
    method-vs-method comparison; use 'lowdin' for the identical-
    projectability-data comparison.
    """
    proj = getattr(engine, '_band_projectability', None)
    if proj is None:
        raise RuntimeError("baseline dis-froz-proj needs the PDWF "
                           "projectability (engine._band_projectability); "
                           "run --method hybrid")
    proj = np.asarray(proj, float)
    num_wann = engine.num_wann
    eig_full = np.array([np.asarray(e) for e in engine.eigenvalues_list])
    e_fermi = engine.e_fermi if engine.e_fermi is not None else 0.0
    tmask = np.asarray(engine.pdwf_target_mask, bool)
    n_target = int(tmask.sum())

    raw_parent = _purpose == 'raw-parent'
    print("\n" + "=" * 70)
    print(("HYBRID: raw selected parent-pool emission "
           "(before disentanglement/gauge)" if raw_parent else
           "BASELINE: stock-wannier90 dis_froz_proj emission "
           "(no in-pipeline disentanglement)"))
    print("=" * 70)

    if amn_mode == 'lowdin' and n_target != num_wann:
        raise RuntimeError(
            f"--baseline dis-froz-proj needs num_proj == num_wann, but the "
            f"target mask has {n_target} AOs vs num_wann = {num_wann}. "
            f"Multi-zeta targets and augmented channels are not "
            f"representable in stock wannier90's .amn; re-run with "
            f"--pdwf-first-radial-only (and no augmentation), or use "
            f"--baseline dis-froz-proj-svd (SVD-compressed multi-zeta amn).")
    if amn_mode == 'svd' and n_target < num_wann:
        raise RuntimeError(
            f"--baseline dis-froz-proj-svd needs num_target >= num_wann "
            f"({n_target} < {num_wann}).")

    pool = select_hybrid_parent_pool(
        engine, pool_factor=pool_factor,
        hybrid_preprocess=_hybrid_preprocess)
    print(f"  pool: {len(pool)} bands [{pool[0]}..{pool[-1]}], "
          f"num_bands = {len(pool)}, num_wann = {num_wann}")
    if raw_parent:
        print("  intermediate parent data only; run --stage hybrid before "
              "Wannier90 localization")
    else:
        print(f"  disentanglement is delegated to wannier90 "
              f"(dis_froz_proj in {seedname}.win)")

    neigh, gshift, bcart0 = _neighbors_from_nnkp(engine, seedname)
    w, resid = mv_weights(bcart0)
    print(f"  MV b-weights: nntot={len(w)}, completeness residual {resid:.2e}")

    mmn_pool, _C_pool, _ = _analytic_pool_overlaps(engine, pool, neigh, gshift,
                                                verbose=verbose)

    # Lowdin target projection: the SAME S^{1/2} construction as
    # compute_lowdin_projectability, so wannier90's p_mk = sum_n |A_mn|^2
    # reproduces our P_mk to machine precision ('lowdin' mode; num_proj ==
    # num_wann == n_target). 'svd' mode compresses the multi-zeta rows to
    # the leading num_wann right-singular directions per k, retaining the
    # singular values (the PDWF A construction on Lowdin rows): p_mk
    # becomes the projection norm onto that subspace, <= the full P_mk.
    amn = _parent_projection_amn(engine, pool, amn_mode)

    _write_eig(f"{seedname}.eig", eig_full[:, pool] - e_fermi)
    _write_amn(
        f"{seedname}.amn", amn,
        header=(f"hybrid raw parent: full Lowdin target projection "
                f"(nproj={n_target}, desired_num_wann={num_wann})"
                if raw_parent else
                "dis_froz_proj baseline: SVD-compressed Lowdin target projection"
                if amn_mode == 'svd' else
                "dis_froz_proj baseline: Lowdin target projection"))
    _write_mmn(f"{seedname}.mmn", mmn_pool, neigh, gshift,
               header=("hybrid raw parent: analytic GTO pool overlaps"
                       if raw_parent else
                       "dis_froz_proj baseline: analytic GTO pool overlaps"))
    print(f"  wrote {seedname}.eig/.amn/.mmn "
          f"(num_bands = {len(pool)}, num_wann = {num_wann}, "
          f"uncompressed pool)")
    return {'pool': pool, 'n_target': n_target, 'num_wann': num_wann}


def run_raw_hybrid_parent(engine, seedname, pool_factor=1.5, verbose=True):
    """Emit the hybrid-selected raw parent pool without running its solver.

    The overlap and projection construction is the same shared implementation
    used by the stock-Wannier90 baseline. Every target-AO projection column is
    retained: ``nproj`` may exceed the desired ``num_wann`` because this is an
    inspectable upstream intermediate, not a Wannier90-consumable hand-off.
    The matching ``.win`` is marked as an intermediate and its ``num_bands``
    is updated to the pool dimension.
    """
    result = run_baseline_dis_froz_proj(
        engine, seedname, pool_factor=pool_factor, amn_mode='raw-full',
        verbose=verbose, _purpose='raw-parent', _hybrid_preprocess=True)
    pool = np.asarray(result['pool'], int)
    engine.selected_band_indices = pool
    engine._num_bands_for_win = len(pool)
    engine._dis_froz = None
    engine._dis_win = None
    engine._additional_keywords = {}
    from .win_file import update_win_parameter
    win_path = f"{seedname}.win"
    update_win_parameter(win_path, 'num_bands', len(pool))
    marker = "! lcao2wannier stage: raw hybrid parent pool; run --stage hybrid before localization\n"
    with open(win_path, 'r') as stream:
        text = stream.read()
    if marker not in text:
        with open(win_path, 'w') as stream:
            stream.write(marker + text)
    return result


def run_hybrid(engine, seedname, p_froz=None, p_floor=0.10, pool_factor=1.5,
               p_froz_band=None, dis_niter=2000, trust='auto', fid_emax=8.0,
               thr_floor=0.55, trust_margin=0.015, dis_tol=1e-9,
               dis_rel_tol=1e-10, report=False, dump_masks=None,
               report_json=None, dump_handoff=None, shell_p_floor=None,
               fermi_shell=(2.0, 3.0), gauge_rank_tol=1e-10,
               gauge_min_sv=None, require_capacity=False,
               dis_check_every=1, dis_mix=1.0, dis_mix_schedule=None,
               dis_taper=0.5, dis_taper_on='check',
               dis_taper_reset_history=False, dis_relax_late=None, dis_log=None,
               shell_floor_max_frac=0.01, regime='auto', froz_emax='auto',
               froz_walk_start=None, verbose=True):
    """Full hybrid Stage 2: masks -> disentangle -> SCDM -> write files.

    p_froz / p_froz_band default to None = derive from the projectability
    distribution (auto_trust_thresholds, `trust` mode). Explicit values
    always win per knob; trust='manual' uses the historical fixed defaults
    (0.95/0.85) for whichever knob was not given. ``report`` prints the
    sorted band-averaged projectability table with the plateau / gate /
    trusted structure annotated (the anatomy the auto rule reads).
    ``dump_masks`` / ``report_json`` write the same anatomy machine-readably
    (write_masks_npz / write_report_json) before the disentanglement runs.

    ``shell_p_floor`` / ``fermi_shell`` are forwarded to
    select_pool_and_frozen: the representability guard releases near-E_F
    states that are NFE/interstitial (P below the floor) from the metal
    Fermi shell, instead of pinning a direction the atomic anchors cannot
    represent. Default None keeps the unconditional shell.

    ``gauge_rank_tol`` / ``gauge_min_sv`` gate the SCDM gauge before the
    .amn is written (see the emission block for why only the first has a
    default).
    """
    proj = getattr(engine, '_band_projectability', None)
    if proj is None:
        raise RuntimeError("hybrid method needs the PDWF projectability "
                           "(engine._band_projectability); run --method hybrid")
    proj = np.asarray(proj, float)
    num_wann = engine.num_wann
    eig_full = np.array([np.asarray(e) for e in engine.eigenvalues_list])
    e_fermi = engine.e_fermi if engine.e_fermi is not None else 0.0

    print("\n" + "=" * 70)
    print("HYBRID PDWF-subspace + SCDM-gauge Wannierization")
    print("=" * 70)

    auto = None
    auto_attempted = (p_froz is None or p_froz_band is None) and trust != 'manual'
    if auto_attempted:
        auto = auto_trust_thresholds(
            proj, eig_full, num_wann, e_fermi=e_fermi, fid_emax=fid_emax,
            mode=('localization' if trust == 'localization' else 'fidelity'),
            thr_floor=thr_floor, margin=trust_margin,
            pool_factor=pool_factor)
        if auto is None:
            print("  [trust auto] distribution rule undecidable (no gray "
                  "plateau or nothing trusted) -- falling back to fixed "
                  "defaults 0.95/0.85")
        else:
            tr = auto['trusted']
            if p_froz is None:
                p_froz = auto['p_froz']
            if p_froz_band is None:
                p_froz_band = auto['p_froz_band']
            print(f"  [trust {trust}] gray-plateau top <P> = "
                  f"{auto['plateau_top']:.3f} -> gate {auto['p_froz_band']:.3f}"
                  f"; trusted {len(tr)} bands [{tr[0]}..{tr[-1]}] "
                  f"(min_k P = {auto['trusted_min_p']:.3f}) -> "
                  f"threshold {auto['p_froz']:.3f}")
            print(f"  [trust {trust}] using p_froz = {p_froz:.3f}, "
                  f"p_froz_band = {p_froz_band:.3f} "
                  f"(fidelity range E_F .. E_F + {fid_emax:g} eV)")
    if p_froz is None:
        p_froz = 0.95
    if p_froz_band is None:
        p_froz_band = 0.85

    # DEGENERACY REGIME (upstream auto_regime). Exactly-degenerate states are
    # defined only up to a rotation inside their multiplet, so an individual
    # projectability is not physically meaningful there -- only the group
    # mean is. Above the degeneracy threshold, average P over each group so
    # every mask decision acts on the whole multiplet and the masks become
    # rotation-invariant. Applied BEFORE auto_trust, which reads P.
    _reg = degeneracy_regime(eig_full)
    _band_level = (_reg['band_level'] if regime == 'auto'
                   else regime == 'band-level')
    if _band_level:
        _p_before = np.asarray(proj).copy()
        proj = symmetrize_degenerate(proj, eig_full, dtol=_reg['dtol'])
        if verbose:
            print(f"  [regime] degeneracy fraction {_reg['degen_frac']:.3f} "
                  f"(dtol {_reg['dtol']:g} eV) -> BAND-LEVEL: projectability "
                  f"symmetrized over degenerate groups "
                  f"(max |dP| = {np.abs(_p_before - proj).max():.2e})")
    elif verbose:
        print(f"  [regime] degeneracy fraction {_reg['degen_frac']:.3f} "
              f"-> per-state masks")

    if report:
        _print_trust_report(proj, eig_full, num_wann, e_fermi, fid_emax,
                            p_froz, p_froz_band, auto,
                            pool_factor=pool_factor)

    cap_info = {}
    pool, admit, frozen = select_pool_and_frozen(
        proj, eig_full, num_wann, p_froz=p_froz, p_floor=p_floor,
        pool_factor=pool_factor, p_froz_band=p_froz_band, e_fermi=e_fermi,
        fermi_shell=fermi_shell, shell_p_floor=shell_p_floor,
        shell_floor_max_frac=shell_floor_max_frac, capacity=cap_info)



    # T1.7 capacity gate. num_wann is the biggest surviving human judgement
    # call and nothing proved a request structurally achievable: the per-k
    # cap in select_pool_and_frozen silently discards frozen states the mask
    # rules demanded, so an impossible num_wann surfaced only as a converged
    # model with quietly wrong bands after the full overlap + disentangle.
    # Report before that money is spent.
    print(f"  [capacity] frozen demand max_k = {cap_info['need']} vs "
          f"num_wann = {num_wann} (headroom {cap_info['headroom']})")
    if cap_info['n_capped']:
        msg = (f"num_wann = {num_wann} is below the frozen-set demand: the "
               f"mask rules asked for up to {cap_info['need']} frozen states, "
               f"so the per-k cap discarded {cap_info['n_dropped']} state(s) "
               f"at {cap_info['n_capped']} of {len(cap_info['demand_per_k'])} "
               f"k-points. The model cannot represent the manifold the "
               f"fidelity target demands; raise num_wann (channel-aligned) "
               f"or relax the freezing thresholds.")
        if require_capacity:
            raise RuntimeError(msg + " [--hybrid-require-capacity]")
        print(f"  [capacity] WARNING: {msg}")
    elif cap_info['n_zero_freedom']:
        print(f"  [capacity] WARNING: {cap_info['n_zero_freedom']} of "
              f"{len(cap_info['demand_per_k'])} k-points have every one of "
              f"the {num_wann} directions frozen, so the disentangler has no "
              f"variational freedom there (Omega_I cannot improve at those "
              f"k). Consider a larger num_wann or --hybrid-shell-p-floor.")
    _fl, _fr = cap_info.get('shell_floor'), cap_info.get('shell_floor_frac', 0.0)
    _fm = cap_info.get('shell_floor_max_frac', 0.01)
    if _fl is not None:
        print(f"  [shell guard] floor P >= {_fl:.3f} applied to the Fermi "
              f"shell ({100*_fr:.2f}% of shell states are below it, within "
              f"the {100*_fm:.1f}% budget): the shell may not override the "
              f"projectability threshold")
    elif shell_p_floor == 'auto' and _fr > 0:
        print(f"  [shell guard] floor DECLINED: {100*_fr:.2f}% of Fermi-shell "
              f"states sit below p_froz (budget {100*_fm:.1f}%). Applying it "
              f"would strip a large part of the Fermi surface, which the "
              f"shell exists to pin — leaving the shell unconditional.")

    # T1.8 pool-coverage audit. The ceiling that builds the masks is computed
    # on the pool slice, so it can never reveal a sub-ceiling state the pool
    # excluded. Recompute it on the full solved set and say what was missed.
    # DETECTION ONLY: widening the pool is measurably harmful (h-BN 48-band
    # pool -> Omega_I 19 vs 8.58), so this reports and continues.
    cov = audit_pool_coverage(eig_full, pool, e_fermi=e_fermi,
                              fermi_shell=fermi_shell)
    if cov['n_below'] or cov['n_above']:
        parts = []
        if cov['n_below']:
            parts.append(f"{cov['n_below']} below (bands "
                         f"{cov['bands_below'].min()}.."
                         f"{cov['bands_below'].max()})")
        if cov['n_above']:
            parts.append(f"{cov['n_above']} above (bands "
                         f"{cov['bands_above'].min()}.."
                         f"{cov['bands_above'].max()})")
        print(f"  [pool audit] WARNING: {' and '.join(parts)} sub-ceiling "
              f"state(s) lie OUTSIDE the pool [{pool[0]}..{pool[-1]}]. The "
              f"true ceiling (full solved set) is E_F{cov['ceiling']:+.3f} "
              f"eV; the frozen manifold the fidelity target demands is not "
              f"fully representable by this pool.")
    elif verbose:
        print(f"  [pool audit] pool [{pool[0]}..{pool[-1]}] covers every "
              f"sub-ceiling state (ceiling E_F{cov['ceiling']:+.3f} eV)"
              if cov['ceiling'] is not None else
              "  [pool audit] no Fermi shell — coverage audit skipped")
    if cov['ceiling_at_solve_top']:
        print("  [pool audit] NOTE: the Fermi shell reaches the highest "
              "SOLVED band, so the ceiling is only a lower bound — "
              "re-run with more bands to audit it properly.")
    # band-subset solve guard: if the pool runs into the last SOLVED band
    # while the basis has more, the pool (and the auto-trust gray plateau)
    # were silently clipped — results would be wrong, not just suboptimal.
    nb_solved = proj.shape[1]
    if pool[-1] >= nb_solved - 1 and nb_solved < engine.num_orbitals:
        raise RuntimeError(
            f"hybrid pool [{pool[0]}..{pool[-1]}] hits the solve ceiling "
            f"({nb_solved} of {engine.num_orbitals} bands solved). "
            f"Raise --solve-nbands to at least "
            f"{int(pool[-1]) + max(10, num_wann // 4)} or drop it.")
    # FROZEN CEILING (cap + fill). The per-state rule above decides which
    # states the projectability TRUSTS; the ceiling decides which states are
    # FROZEN: every pool state at or below E_ceil, nothing above it, so the
    # frozen set is the energy window [pool bottom, E_ceil] at every k -- the
    # same semantics as wien2wannier's --froz-emax (apply_froz_emax + per-k
    # floor). Matched-extent tests: releasing trusted-set holes inside the
    # span loses bands (h-BN conduction 52.0 -> 4.9 meV when filled; MgB2
    # 24.1 -> 7.0). The trust mask is kept for the certificate/dump.
    _eps_pool = np.asarray(eig_full, float)[:, pool] - e_fermi
    frozen_trust = np.asarray(frozen, bool).copy()
    ceil_info = None
    if froz_emax is not None and str(froz_emax).lower() != 'none':
        ceil_info = resolve_frozen_ceiling(
            _eps_pool, frozen_trust, num_wann, mode=froz_emax,
            fid_emax=fid_emax, start=froz_walk_start)
        _app = apply_frozen_ceiling(frozen, admit, _eps_pool,
                                    ceil_info['E_ceil'], num_wann)
        ceil_info.update(_app)
        if verbose:
            _trail = ", ".join(f"{fe:+.2f}:{d}" for fe, d in ceil_info['trail'])
            print(f"  [frozen ceiling] mode={ceil_info['mode']}: trusted-set top "
                  f"E_F{ceil_info['E_top']:+.3f} eV, walk start "
                  f"E_F{ceil_info['start_value']:+.3f} eV; rungs (E-E_F:demand) "
                  f"{_trail} -> ceiling E_F{ceil_info['E_ceil']:+.3f} eV "
                  f"(capacity supremum E_F{ceil_info['sup']:+.3f})")
            print(f"  [frozen ceiling] frozen set = pool states below the "
                  f"ceiling: +{_app['added']} filled, -{_app['capped']} capped "
                  f"relative to the trust mask"
                  + (f"; {_app['n_over']} k over num_wann (deepest kept)"
                     if _app['n_over'] else ""))
    elif verbose:
        print("  [frozen ceiling] none: per-state trust mask used as the "
              "frozen set (legacy)")
    nfk = frozen.sum(axis=1)
    print(f"  pool: {len(pool)} bands [{pool[0]}..{pool[-1]}], num_wann={num_wann}")
    _fdesc = (f"pool states below E_F{ceil_info['E_ceil']:+.3f} eV"
              if ceil_info is not None else f"trust mask, P>={p_froz:.3f}")
    print(f"  frozen/k: min {nfk.min()} max {nfk.max()} "
          f"({_fdesc}); admitted/k: min {admit.sum(axis=1).min()} "
          f"max {admit.sum(axis=1).max()} (P>={p_floor})")

    if dump_masks:
        write_masks_npz(dump_masks, pool, proj[:, pool], eig_full[:, pool],
                        admit, frozen, engine.kpoints, e_fermi,
                        p_froz=p_froz, p_gate=p_froz_band, p_floor=p_floor,
                        fid_emax=fid_emax,
                        fallback_used=(auto_attempted and auto is None),
                        frozen_trust=frozen_trust,
                        E_ceil=(None if ceil_info is None
                                else ceil_info['E_ceil']),
                        froz_emax_mode=('none' if ceil_info is None
                                        else ceil_info['mode']))
    if report_json:
        write_report_json(report_json, proj, eig_full, num_wann, e_fermi,
                          fid_emax, p_froz, p_froz_band, auto,
                          pool_factor=pool_factor,
                          label=os.path.basename(seedname))

    neigh, gshift, bcart0 = _neighbors_from_nnkp(engine, seedname)
    w, resid = mv_weights(bcart0)
    print(f"  MV b-weights: nntot={len(w)}, completeness residual {resid:.2e}")

    mmn_pool, C_pool, _ = _analytic_pool_overlaps(engine, pool, neigh, gshift,
                                               verbose=verbose)


    # PDWF projections seed the free directions (target AOs, pool rows)
    tmask = engine.pdwf_target_mask
    amn_seed = np.stack([
        (np.asarray(S) @ np.asarray(C))[tmask][:, pool].T
        for S, C in zip(engine.S_k_list, engine.eigenvectors_list)])

    dis_info = {}
    V, om_i, hist = disentangle_frozen(
        mmn_pool, amn_seed, neigh, w, num_wann, admit, frozen,
        niter=dis_niter, tol=dis_tol, rel_tol=dis_rel_tol, verbose=verbose,
        check_every=dis_check_every, mix=dis_mix,
        mix_schedule=dis_mix_schedule, taper=dis_taper,
        taper_on=dis_taper_on,
        taper_reset_history=dis_taper_reset_history,
        relax_late=dis_relax_late,
        log_path=(f"{seedname}_disentangle.log" if dis_log is None
                  else (dis_log or None)),
        info=dis_info)
    # sweeps, not checks: with check_every > 1 the history is strided, and
    # len(hist) would under-report the real iteration count by that factor
    extra = "" if dis_mix == 1.0 else f", mix {dis_mix:g}"
    if dis_mix_schedule is not None:
        extra = (f", scheduled mix peak {dis_mix_schedule:g} -> "
                 f"{dis_info['final_mix']:.3g} (taper {dis_taper:g} on "
                 f"{dis_taper_on})")
    print(f"  disentangled: Omega_I = {om_i:.4f} Ang^2 "
          f"({dis_info['n_sweeps']} sweeps, {dis_info['n_checks']} checks "
          f"@ every {dis_info['check_every']}{extra})")


    eig_pool = eig_full[:, pool]
    eig_t, W, mmn_rot, _ = rotate_to_subspace(eig_pool, V, mmn_pool, neigh)

    B = [C_pool[k] @ W[k] for k in range(len(C_pool))]
    if dump_handoff:
        np.savez_compressed(
            dump_handoff, B=np.stack(B),
            kpoints=np.asarray(engine.kpoints), eig_handoff=eig_t,
            e_fermi=e_fermi, pool=np.asarray(pool))
        print(f"  hand-off Bloch basis dumped -> {dump_handoff}")
    anchors, A_gauge, gauge_info = scdm_anchors(B, num_wann)
    cond = gauge_info['sigma_min']
    print(f"  SCDM anchors (AO indices): {sorted(int(a) for a in anchors)}")
    print(f"  gauge conditioning [{gauge_info['candidate']}]: "
          f"min singular value over k = {cond:.3e} "
          f"(worst k = {gauge_info['argmin_k']}), "
          f"cond = {gauge_info['cond']:.3g}")

    # Gate the gauge BEFORE emitting it: wannier90 localizes from whatever
    # .amn it is handed and reports converged spreads either way, so a
    # rank-deficient gauge is a silent wrong answer. Two thresholds, because
    # sigma_min is scale-dependent:
    #   * rank deficiency (cond > 1/gauge_rank_tol) is scale-FREE and always
    #     fatal — this is the exactly-singular point-row failure the
    #     density-column trials were introduced to fix.
    #   * an absolute sigma_min floor is opt-in only. Do NOT default it:
    #     measured sigma_min spans 0.39 (Bi) to <5e-4 (Sc, 294 WFs), and
    #     the h-BN/SnTe/Sc champions (10.0/92.2, 10.4/46.6, 1.23 meV) all
    #     sit at 0.000-0.002, so any ported absolute floor (upstream uses
    #     2e-3 on differently-normalized LAPW gauges) would reject them.
    _scdm_bad = (not np.isfinite(gauge_info['cond'])
                 or gauge_info['cond'] > 1.0 / gauge_rank_tol)
    if _scdm_bad:
        raise RuntimeError(
            f"SCDM gauge is rank-deficient at k = {gauge_info['argmin_k']}: "
            f"sigma_min = {cond:.3e}, sigma_max = "
            f"{gauge_info['sigma_max']:.3e}, cond = {gauge_info['cond']:.3g} "
            f"(> 1/{gauge_rank_tol:g}). The .amn was NOT written: "
            f"wannier90 would localize from a singular initial gauge and "
            f"report converged spreads for a meaningless model. Try a "
            f"different target/channel set, or relax --hybrid-gauge-rank-tol "
            f"if this is genuinely acceptable.")
    if gauge_min_sv is not None and cond < gauge_min_sv:
        raise RuntimeError(
            f"SCDM gauge sigma_min = {cond:.3e} at k = "
            f"{gauge_info['argmin_k']} is below the requested floor "
            f"--hybrid-gauge-min-sv = {gauge_min_sv:g}. The .amn was NOT "
            f"written. NOTE: absolute sigma_min is not comparable across "
            f"systems (Bi 0.39, h-BN 0.036, Sc <5e-4 all give good models); "
            f"prefer the scale-free cond gate unless you have calibrated "
            f"this floor on this material.")

    # STAGED PUBLICATION. Every hand-off file is written to a .tmp path,
    # validated as a COMPLETE set (full-coverage contractivity + finiteness
    # on the serialized values), and only then renamed into place together.
    # A failure leaves any previous good hand-off untouched and never
    # publishes a partial one (the .tmp files stay behind for inspection).
    _tmp = lambda ext: f"{seedname}.{ext}.tmp"

    _write_eig(_tmp("eig"), eig_t - e_fermi)
    _write_amn(_tmp("amn"), A_gauge)
    _write_mmn(_tmp("mmn"), mmn_rot, neigh, gshift)

    # FINAL ACCEPTANCE, then atomic publication. Full coverage over every
    # (k,b) block of the serialized files -- a sampled or in-memory check
    # can pass a hand-off whose written form is broken. Nothing is
    # repaired, clipped, or unitarized to make this pass.
    from .verification import validate_handoff, HandoffValidationError
    try:
        _acc = validate_handoff(_tmp("eig"), _tmp("amn"), _tmp("mmn"),
                                num_wann=num_wann, tau=1e-6)
    except HandoffValidationError as _e:
        raise HandoffValidationError(
            f"hand-off REFUSED at final acceptance -- nothing was "
            f"published (the .tmp files remain for inspection): {_e}")
    sv_max = _acc['max_sv']

    for _ext in ('eig', 'amn', 'mmn'):
        os.replace(_tmp(_ext), f"{seedname}.{_ext}")
    print(f"  wrote {seedname}.eig/.amn/.mmn "
          f"(num_bands = num_wann = {num_wann}, pre-disentangled; "
          f"published after full-coverage acceptance)")
    return {'pool': pool, 'omega_i': om_i, 'anchors': anchors,
            'max_sv': sv_max, 'eig': eig_t,
            'p_froz': p_froz, 'p_froz_band': p_froz_band}
