# HPCC CRYSTAL23 runs of the 1.1.6 round (2026-10-21)

Real outputs of decks MACE wrote, run on MSU HPCC with `CRYSTAL/23-intel-2023a`
(Pcrystal, 8-16 MPI tasks, `mpirun` as `submitcrystal23.sh` runs it). Read by
`tests/test_elastic_decks.py`, `test_elastic_analysis.py`, `test_transport_tau.py`,
`test_soc_heavy_hpcc.py` and `test_phonopy_gui_routes.py`.

## soc/ - heavy-element two-component SCF

`mace convert --batch` from `cif/*.cif` (PBE0, spin-polarized, SP, TOLINTEG
7 7 7 7 14, the library basis `full.basis.triplezeta`, `"soc": true`):

| Deck | Basis / ECP | Extra | Result |
|---|---|---|---|
| `CsPbI3_soc.d12` | official pob-TZVP-rev2-SOC (`soc_basis`), STUTSC | - | 10 cycles, indirect gap 1.928 eV |
| `HgTe_soc.d12` | library basis, INPSOC | - | 11 cycles, direct gap 0.790 eV |
| `Bi_soc.d12` | library basis, INPSOC | SMEAR 0.01 | 17 cycles, metal |
| `W_soc.d12` | library basis, INPSOC | SMEAR 0.01 | 19 cycles, metal (output not kept) |

The outputs are complete and unedited. The decks' title line is the name the
converter gave; the runs used the decks as they are here.

## mechanics/ - ELASTCON, ELAPIEZO, PHOTOELA, EOS, CPHF, QHA of diamond; ELAPIEZO of BN

`mace opt2d12 --calc-type ELASTIC|EOS --elastic-settings ...` from the corpus
diamond optimization `test/OPT/1_dia_opt_rev1.out` (B3LYP-D3, SPIN, SPINLOCK,
external triple-zeta basis): `dia_elastcon` (NUCHESS, HESSNUM2, PRINT),
`dia_elapiezo` (PRINT), `dia_photoela` (PRINT; CPHF/KS dielectric tensors of
spin-polarized strained cells, 42 min on 16 tasks), `dia_eos` (PRINT),
`dia_cphf` (`--calc-type CPHF`, FOURTH: polarizability, β, γ; trimmed to the
header, the dielectric and β blocks, and the γ block to the end), `dia_qha`
(`--calc-type QHA`, TEMPERAT 60 10 600, Γ only: the header, the zero-pressure
HELMHOLTZ and lattice-expansion tables, and the termination line).

`bn_elapiezo`: zincblende BN, from MACE's own OPT deck of `cif/`-style input
(`mace convert`, PBE0, POB-TZVP-REV2, TOLINTEG 8 8 8 9 24; the 7 7 7 7 14
deck's SCF diverged to +430 Ha in the first OPT step) and the ELAPIEZO deck of
`mace opt2d12 --calc-type ELASTIC` (then written by the 1.1.6 elastic generate command). The ELAPIEZO run took three jobs: two hit their
walltime (1:15, 0:40) and the third, with `--restart RESTART` and the same job
name (ELASINFO.DAT in the scratch folder), finished in 60 min on 16 tasks.
This is that third output, trimmed like the others. The outputs are trimmed: the run
header through the first `N. OF ATOMS PER CELL` line, every `INFORMATION`
line after it, and the results from 4 lines before `FINAL RESULTS START`
(EOS: from `SORTING VOLUMES/ENERGIES`) to the end. Kept lines are byte for
byte; no number was edited.

`dia_eos_strained.out` and `dia_elapiezo_strained.out` are the same trimmed
outputs with, before the results, the lines of the full runs that hold a
strained cell: the central point (EOS: `VOLUME OF PRIMITIVE CELL` and
`ENERGY OF CENTRAL POINT`; ELAPIEZO: `CELL FORCES OF CENTRAL POINT` through
strain 1's reference cell), then the first strained cell's optimization (EOS:
from `REFERENCE GEOMETRY DEFINED`; ELAPIEZO: from 6 lines before its `OPT
END`) through its `FINAL OPTIMIZED GEOMETRY` atoms: a = 3.44764376 at 92 %
of the volume, a = 3.49163903 under strain 1. `dia_elastcon_nuchess.out` is
`dia_elastcon.out` with, before the results, the full run's Γ-point
frequencies and thermodynamic functions of the NUCHESS Hessian (from `+++
SYMMETRY ADAPTION OF VIBRATIONAL MODES +++` to the end of `OTHER
THERMODYNAMIC FUNCTIONS`). Kept lines are byte for byte.
`dia_qha_full.out` is the full QHA run trimmed differently, for each
volume's harmonic thermodynamics and the PVT relation (`tests/test_qha_thermo.py`): the header
through the first `N. OF ATOMS PER CELL` line; for each of the four volumes, the last
`PRIMITIVE CELL ... VOLUME=` line before its `HARMONIC VIBRATIONAL CONTRIBUTIONS` block, the
block's header (EL, E0) and its temperature blocks at 10 K and every 50 K (`THERMODYNAMIC
FUNCTIONS WITH VIBRATIONAL CONTRIBUTIONS` through `HEAT CAPACITY`); the QHA module from its
banner through the zero-pressure lattice-expansion table; the `PRESSURE-VOLUME-TEMPERATURE
RELATION` header and its rows at the same temperatures; the termination line. Kept lines are
byte for byte.
`dia_cphf_running.out` is the full CPHF run's first 657 lines, unedited: the header, the
SCF, the CPKS module banner and its first `CP_DRIVER` lines, as the output reads while the
CPHF block runs.
`dia_elapiezo.ELASINFO.DAT` and `dia_eos.EOSINFO.DAT` are the restart files
the diamond ELAPIEZO and EOS runs wrote in their scratch folders, unedited.

## strain/ - deformation-potential series of diamond

`mace tau strain` on the SP deck of the same structure (axis x; -1, -0.5, 0,
+0.5, +1 %), with SPINLOCK held 3 cycles (`dia_sp_dp_x_*.out`). Trimmed like
the mechanics outputs, with the tail starting 60 lines before the last `SCF
ENDED` (the last cycle's band edges and the Mulliken blocks are kept).
