# `mace gui` example folder

```
mace gui --dir examples/gui --open
```

The folder the GUI's welcome screen and `docs/gui.md` point to. Every file is a
textbook structure or a copy of a CRYSTAL output that MACE's own tests already
use; nothing here is research data, and no number is made up for the GUI.
Rebuild everything here with `python examples/gui/build_examples.py`.

| File | Where it comes from |
|---|---|
| `structures/*.cif` | Textbook crystals (MgO, LiF, NaCl, Si, diamond, hBN): space group plus asymmetric unit at experimental lattice constants, written by `build_examples.py` |
| `outputs/Si_sp.out`, `.d12` | `tests/fixtures/phonopy_crystal/Si/`: the trimmed phonopy CRYSTAL example (BSD-3-Clause, A. J. Karttunen; see that folder's README for the license) |
| `outputs/graphene_slab_sp.out`, `.d12` | `tests/data/low_dim_groups/graphene_lg80.*`: a complete CRYSTAL23 HSEsol-3c single point of textbook graphene (a = 2.4612 Å) as a SLAB in layer group 80, run for MACE's layer-group tests; it shows a 2D slab |
| `outputs/Ag2Br3_opt_unfinished.out`, `.d12` | `tests/data/opt_restart/tqc_agbr.*`: the R-3c Ag2Br3 cell (mp-862982; the tests call it `tqc_agbr`): a FULLOPTG run that stopped after 4 OPT points (6 geometries), used for the before/after view |
| `materials.db` | Written by `MaterialDatabase` from the files above. It holds 5 materials and their OPT/SP calculations; every number comes from those files (a slab stores only its periodic lattice lengths, a and b, and no volume). It is kept in git, even though `*.db` is ignored. |
