# FREQCALC runs with INTENS + INTCPHF

Three B3LYP-D3 Γ-point frequency runs of the corpus `test/FREQ` (CRYSTAL23,
`mace opt2d12 --calc-type FREQ` with IR intensities from CPHF), complete and
unedited, read by `tests/test_freq_born.py` and the GUI's FREQ view (its
Dielectric button):

- `agbr_freq.out`: zinc-blende AgBr (F-43m, cubic), a converged CPHF.
- `ag2br2_freq.out`: Ag2Br2 (P6_3mc, hexagonal), a converged CPHF: uniaxial
  ε∞, E1 modes polar along x and y, the A mode along z.
- `ag2secl_freq.out`: Ag2SeCl (P4mm, tetragonal) with a CPHF that does NOT
  give the crystal's response: its DIIS test reaches 9.5e7, ε∞ comes out
  biaxial (ε_xx 2.060, ε_yy 1.368, though a tetragonal crystal needs
  ε_xx = ε_yy) and the β table overflows (`-8.9595+158`, no E). MACE flags the
  symmetry mismatch and the diverged β instead of reporting a biaxial crystal
  or a zero (centrosymmetric) β.
