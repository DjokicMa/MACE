# GUI parity fixtures (plan E16)

Small, generic CRYSTAL23 runs made for the GUI's CI tests (`tests/gui/test_parity_fixtures.py`):
nothing here comes from research data. Each output is the complete, untrimmed file CRYSTAL wrote;
the larger ones are gzipped (the tests decompress them into a temporary folder).

Made on 2026-10-07 with the site's CRYSTAL/23-intel-2023a module (serial `crystal` and `properties`,
one core, `OMP_NUM_THREADS=1`), from the decks kept beside them:

| Deck | Run | Kept |
|---|---|---|
| `water_opt.d12` | MOLECULE (C2v, point group 15) optimization from a distorted geometry, B3LYP/POB-TZVP-REV2: 6 optimization points | `water_opt.out` |
| `water_freq.d12` | FREQCALC with PREOPTGEOM, INTENS, INTRAMAN/INTCPHF, IRSPEC and RAMSPEC (default DAMPFAC) | `water_freq.out.gz`, `water_freq_IRSPEC.DAT.gz`, `water_freq_RAMSPEC.DAT.gz` |
| `mgo_sp.d12` | MgO rock salt SP, B3LYP/POB-TZVP-REV2, SHRINK 8 8 (its wave function feeds the properties runs) | (the output is not kept) |
| `mgo_doss.d3` | DOSS written by `mace opt2d3 -t DOSS` with every default, 1000 energy points instead of 10000 | `mgo_doss.out.gz`, `mgo_doss.DOSS.DAT.gz`, `mgo_doss.f25.gz` (CRYSTAL's DOSS.DAT and fort.25 under MACE's names) |
| `mgo_dens.d3` | ECH3, 20 points: CRYSTAL's own `DENS_CUBE.DAT` name | `mgo_dens/DENS_CUBE.DAT.gz`, `mgo_dens/mgo_dens.out` |
| `mgo_pato.d3` | PATO then ECH3 on the same grid (superposed atomic densities): SCF − PATO is the deformation density | `mgo_pato/DENS_CUBE.DAT.gz`, `mgo_pato/mgo_pato.out` |
| `mgo_echg.d3` | ECHG on MgO's (001) plane, 50 points (MAPNET with COORDINA A, B, C) | `mgo_echg.f25.gz` (CRYSTAL's fort.25, a MAPN block), `mgo_echg.out` |
| `mgo_wan.d3` | LOCALWF (VALENCE, SYMMWF) with PRINTPLO IPLOT −8, ICUBE 1, 30 points: the first Wannier function on a 3D grid | `mgo_wan/1_WAN.CUBE.gz`, `mgo_wan/mgo_wan.out.gz` |

What they showed:

- A molecule optimization prints only its initial and its final geometry table, however many points it
  takes (6 here); a symmetric molecule's table lists every atom (ATOMS IN THE ASYMMETRIC UNIT 2, IN THE
  UNIT CELL 3).
- CRYSTAL23's LOCALWF/PRINTPLO writes one `<k>_WAN.CUBE` per Wannier function, not the
  `<input>_WAN_CUBE.DAT` the manual names; the GUI reads both names as a signed (orbital) volume.
- MPPcrystal could not start on a development node (MPI_INIT aborted outside a batch job), so no
  MPPcrystal output is here.
