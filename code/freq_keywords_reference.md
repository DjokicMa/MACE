# CRYSTAL Frequency Calculation Keywords Reference

## Structure Overview

CRYSTAL frequency calculations can have different structures:

### 1. Standard FREQCALC (inside geometry block)
```
TITLE
EXTERNAL/CRYSTAL/SLAB/etc
[geometry specification]
FREQCALC
  [frequency keywords]
END
END
[basis set]
```

### 2. ANHARM (outside FREQCALC, top-level like OPTGEOM)
```
TITLE
EXTERNAL/CRYSTAL/SLAB/etc
[geometry specification]
ANHARM
  LB  # Atom label (must be H or D)
END
END
[basis set]
```

### 3. PBAND (inside geometry block, before FREQCALC)
```
TITLE
EXTERNAL/CRYSTAL/SLAB/etc
[geometry specification]
PBAND
  ISS NK FLAG1 FLAG2
  [k-point specifications]
FREQCALC
  DISPERSION
  [other keywords]
END
END
[basis set]
```

## Keyword Details

### ANHARM - Anharmonic X-H Stretching
- **Location**: Top-level, outside FREQCALC
- **Purpose**: Calculate anharmonic frequencies for X-H/X-D bonds
- **Format**:
  ```
  ANHARM
  LB  # Integer: Label of H/D atom to displace
  [optional keywords]
  END
  ```
- **Optional Keywords**:
  - `ISOTOPES NL` followed by NL lines of `LB AMASS`
  - `KEEPSYMM` - Maintain symmetry by moving all equivalent H atoms
  - `POINTS26` - Use 26 points instead of default 7
  - `NOGUESS` - Fresh SCF at each point
  - `PRINT` - Extended printing
  - `TEST` - Test mode, no calculations
- **Sensible Defaults**:
  - 7 points (sufficient accuracy)
  - No symmetry preservation (single H moves)
  
### ANHAPES - Anharmonic PES
- **Location**: Inside FREQCALC block
- **Purpose**: Calculate cubic and quartic force constants
- **Format**:
  ```
  ANHAPES
  NMODES
  MODE1 MODE2 MODE3 ... MODENM
  SCHEME STEP
  ```
- **Parameters**:
  - `NMODES`: Number of modes to treat anharmonically (exclude translations/rotations)
  - `MODE1...`: Mode numbers (start from 4 for 3D, 7 for molecules)
  - `SCHEME`: Numerical scheme (1-4, recommend 3)
  - `STEP`: Grid step size (recommend 0.9)
- **Optional**: `RESTPES` to restart from SCANPES.DAT

### VSCF - Vibrational Self-Consistent Field
- **Location**: Inside FREQCALC block (after ANHAPES)
- **Purpose**: Mean-field treatment of anharmonic vibrations
- **Format**: Single keyword `VSCF`
- **Optional Keywords**:
  - `VSCFTOL N` - Convergence tolerance 10^-N cm^-1 (default N=3)
  - `VSCFMIX N` - Mixing percentage (default 25)
- **Sensible Defaults**:
  - Tolerance: 1e-3 cm^-1
  - Mixing: 25%

### VCI - Vibrational Configuration Interaction
- **Location**: Inside FREQCALC block (after ANHAPES)
- **Purpose**: Beyond mean-field anharmonic treatment
- **Format**:
  ```
  VCI
  NQUANTA NMODES
  GUESS
  ```
- **Parameters**:
  - `NQUANTA`: Max excitation quanta (recommend 6)
  - `NMODES`: Max simultaneously excited modes (recommend 3)
  - `GUESS`: 0=harmonic, 1=VSCF (recommend 1)

### INS - Inelastic Neutron Scattering
- **Location**: Inside FREQCALC block (with DISPERSION)
- **Purpose**: Neutron-weighted phonon DOS
- **Format**:
  ```
  INS
  NUMA NBIN NWTYPE
  ```
- **Parameters**:
  - `NUMA`: Max frequency in cm^-1 (default 3000)
  - `NBIN`: Number of bins (default 300)
  - `NWTYPE`: 0=coherent, 1=incoherent, 2=both (default 2)

### PBAND - Phonon Bands in Geometry
- **Location**: Geometry block (before FREQCALC)
- **Purpose**: Construct supercell for specific k-point path
- **Format**:
  ```
  PBAND
  ISS NK FLAG1 FLAG2
  [if ISS>0: I1 I2 I3 J1 J2 J3 per line]
  [if ISS=0: LABELA LABELB per line]
  ```
- **Parameters**:
  - `ISS`: Shrink factor (0 for labels)
  - `NK`: Total k-points along line
  - `FLAG1`: >1 activates interpolation (default 1)
  - `FLAG2`: 0=primitive, 1=conventional cell

### TEMPERAT - Temperature Range
- **Location**: Inside FREQCALC block
- **Purpose**: Thermodynamics at multiple temperatures
- **Format**:
  ```
  TEMPERAT
  NT T1 T2
  ```
- **Parameters**:
  - `NT`: Number of temperature points
  - `T1`: Starting temperature (K)
  - `T2`: Ending temperature (K)
- **Sensible Values**: 5 100.0 500.0

### PRESSURE - Pressure Range
- **Location**: Inside FREQCALC block
- **Purpose**: Thermodynamics at multiple pressures
- **Format**:
  ```
  PRESSURE
  NP P1 P2
  ```
- **Parameters**:
  - `NP`: Number of pressure points
  - `P1`: Starting pressure (MPa)
  - `P2`: Ending pressure (MPa)
- **Default**: 1 0.101325 0.101325 (atmospheric)

### SCANMODE - Mode Scanning
- **Location**: Inside FREQCALC block
- **Purpose**: Scan geometry along normal modes
- **Format**:
  ```
  SCANMODE
  NMO INI IFI STEP
  N1 N2 ... NMO
  ```
- **Parameters**:
  - `NMO`: Number of modes to scan (>0 for SCF, <0 for geometry only)
  - `INI`: Initial point
  - `IFI`: Final point  
  - `STEP`: Step as fraction of max classical displacement
  - `N1...`: Mode numbers

### COMBMODE - Combination Modes
- **Location**: Inside FREQCALC block
- **Purpose**: Combination modes and overtones
- **Format**:
  ```
  COMBMODE
  [optional subkeywords]
  END
  ```
- **Subkeywords**:
  - `ALL` - All combinations
  - `IR` - IR active only
  - `RAMAN` - Raman active only
  - `IRRAMAN` - Both (default)
  - `FREQ` - Sort by frequency (default)
  - `IRREP` - Sort by irrep
  - `FREQRANGE FMIN FMAX` - Frequency range

### BETAVIB - Vibrational SHG/Pockels
- **Location**: Inside FREQCALC block
- **Purpose**: Vibrational contribution to nonlinear optics
- **Requires**: INTENS and INTRAMAN active
- **Format**:
  ```
  BETAVIB
  WAVELENGTH
  ```
- **Parameter**: Wavelength in nm (e.g., 1064.0 for Nd:YAG)

### ADP - Anisotropic Displacement Parameters
- **Location**: Inside FREQCALC block
- **Purpose**: Thermal displacement tensors
- **Format**:
  ```
  ADP
  NTYP NNEGL
  ```
- **Parameters**:
  - `NTYP`: Algorithm type (default 0)
  - `NNEGL`: Additional modes to neglect at Γ

## Implementation Priority

1. **High Priority** (enables many calculations):
   - INS (already in d12creation.py)
   - TEMPERAT/PRESSURE (partially exists)
   - ANHARM (new structure needed)

2. **Medium Priority** (specialized but important):
   - ANHAPES + VSCF/VCI
   - SCANMODE
   - PBAND

3. **Low Priority** (very specialized):
   - COMBMODE
   - BETAVIB
   - ADP

## UI Prompting Guidelines

### For ANHARM:
1. "Select H/D atom for anharmonic calculation"
2. "Maintain symmetry for equivalent H atoms? [no]"
3. "Use extended 26-point grid? [no - 7 points sufficient]"

### For ANHAPES:
1. "Number of modes for anharmonic treatment"
2. "Enter mode numbers (exclude first 3/6)"
3. "Numerical scheme (1-4) [3]"
4. "Step size [0.9]"
5. "Continue with VSCF? [yes]"
6. "Continue with VCI? [no]"

### For INS:
1. "Calculate INS spectrum? [no]"
2. "Maximum frequency (cm^-1) [3000]"
3. "Number of bins [300]"
4. "Neutron type (0=coh, 1=incoh, 2=both) [2]"

### For Multi-temperature:
1. "Calculate at multiple temperatures? [no]"
2. "Number of temperatures [5]"
3. "Temperature range (K) [100 500]"