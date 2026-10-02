# phonopy CRYSTAL-interface fixtures

Trimmed copies of the CRYSTAL example outputs that ship with phonopy, used by
`tests/test_phonopy_*.py`. Nothing here comes from the group's research data.

- Upstream: https://github.com/phonopy/phonopy, tag `v4.7.2`,
  commit `10bff75b22ad9218f560ba22014cdc25070b4573`,
  directories `example/Si-CRYSTAL` and `example/NaCl-CRYSTAL`
  (CRYSTAL17, PBE0/SVP, computed by A. J. Karttunen).
- License: BSD-3-Clause, reproduced below.

## Files and how they were trimmed

The trimming removes the long middle of each output (basis-set print, SCF
cycles, timing lines) and keeps every block the code under test reads:
the run header, the `PRIMITIVE CELL - CENTRING CODE` / `ATOMS IN THE ASYMMETRIC
UNIT` block, `DIRECT LATTICE VECTORS CARTESIAN COMPONENTS (ANGSTROM)`,
`CARTESIAN COORDINATES - PRIMITIVE CELL`, the `SCF ENDED` and `TOTAL ENERGY`
lines, the `CARTESIAN FORCES IN HARTREE/BOHR` block, the dielectric-tensor and
`ATOMIC BORN CHARGE TENSOR` blocks, and the `EEEEEEEEEE TERMINATION` line.
Kept lines are byte-for-byte upstream; no number was edited.

| File | From | What was kept |
|---|---|---|
| `NaCl/supercell-001.o`, `NaCl/supercell-002.o` | `supercell-00N.o` (4x4x4 displacements) | header and geometry up to the basis-set print; `== SCF ENDED` through `RESULTANT FORCE`; `TOTGRA` to the end |
| `NaCl/crystal.o` | `crystal.o` (Gamma FREQCALC, INTENS + INTCPHF) | geometry; SCF end; the 10-line dielectric block; the Born-tensor block; the mode table; termination |
| `NaCl/crystal.d12` | the deck `crystal.o` echoes in its header | verbatim |
| `NaCl/parent.out`, `Si/parent.out` | `crystal.o` | the single-point part every FREQCALC run starts with: geometry (without the FREQCALC banner and its three INPFREQ notes), SCF end, termination line |
| `NaCl/parent.d12`, `Si/parent.d12` | the echoed deck | verbatim, minus the `FREQCALC ... ENDFREQ` records |

`parent.out`/`parent.d12` are the SCF single point at the same geometry,
basis and settings as the FREQCALC run - the kind of parent `mace phonopy
generate` accepts (it refuses a FREQCALC output as the geometry source). The
Si run's echoed deck had no newline after its final `END` (the runner printed
`ENDCreating temporary directory ...`); the deck ends at that `END`.

## Full vs trimmed check (run once, before trimming was accepted)

With phonopy 4.7.2 on the upstream and trimmed files:

```
read_crystal crystal.o -> NaCl/parent.out identical
read_crystal crystal.o -> NaCl/crystal.o identical
read_crystal crystal.o -> Si/parent.out identical
read_crystal supercell-001.o -> NaCl/supercell-001.o identical
read_crystal supercell-002.o -> NaCl/supercell-002.o identical
forces 001 identical
printed_geometry 001 identical force_rows 128 128
forces 002 identical
printed_geometry 002 identical force_rows 128 128
BORN identical
```

(`read_crystal` = lattice, species, positions and conventional atomic numbers;
`forces` = `parse_set_of_forces`; `printed_geometry`/`force_rows` = MACE's
collect checks; `BORN` = `phonopy-crystal-born` output.)

## phonopy license

```
Copyright (c) 2014-2024, Phonopy. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of the phonopy project nor the
  names of its contributors may be used to endorse or promote products
  derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT
HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
