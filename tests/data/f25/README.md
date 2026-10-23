# A real phonon fort.25 (CRYSTAL23 diamond)

`dia_phonon_f25.{d12,fort.25,PHONBANDS.DAT}`: one CRYSTAL23 B3LYP-D3 run of diamond
(SCELPHONO 2x2x2, FREQCALC DISPERSI + BANDS Γ-X-W-L-Γ, ISS 8, 50 points per line,
+ PDOS 1600 160), run on MSU HPCC in the 1.1.6 round; the files are the run's own,
unedited. `dia_phonon_f25.dispersion.txt` is a byte-exact excerpt of that run's
`.out` (its PHONON BANDS print: the four path headers and the k points at each
path's ends, with their frequencies). Read by `tests/test_f25_plotting.py`,
`tests/test_f25_phonon_kaxis.py`, `tests/test_props_freq.py`, `tests/test_kpath_info.py`
and the GUI phonon tests.
