# SeeKPath Integration Plan

## Overview

This document outlines the plan to integrate the official `seekpath` Python library into MACE for accurate band structure k-path generation. The integration will be implemented as a **standalone module** first, allowing testing before replacing the existing `seekpath_data` dictionary approach.

## Problem Statement

The current `seekpath_data` dictionary in `d3_kpoints.py` has two issues:

1. **Data Entry Errors**: Many paths have incorrect segment orders, wrong discontinuity markers, or label/segment mismatches
2. **Fundamental Limitation**: Non-cubic Bravais lattices have **parametric k-points** whose coordinates depend on the actual crystal's lattice parameters

### Parametric K-Points by Lattice Type

| Bravais Type | Has Parameters | Example Formula |
|--------------|----------------|-----------------|
| cP, cF, cI (cubic) | No | All fixed fractions |
| tI (tetragonal I) | Yes | H = (1 + c²/a²)/4 |
| oF (orthorhombic F) | Yes | J = (1+a²/b²-a²/c²)/4 |
| hR (rhombohedral) | Yes | D = a²/(4c²), N = 1/3+D |
| mC (monoclinic C) | Yes | Z = (2+a·cosβ/c)/(4sin²β) |

**A static dictionary cannot handle parametric k-points correctly.**

## Solution Architecture

### New Module: `seekpath_interface.py`

```
Crystal_d3/
├── d3_kpoints.py           # Existing (will be updated to use new interface)
├── seekpath_interface.py   # NEW: Standalone seekpath integration
├── CRYSTALOptToD3.py       # Existing D3 generator
└── ...
```

### Module Responsibilities

**`seekpath_interface.py`**:
- Convert CRYSTAL output data → seekpath input format
- Call seekpath library
- Convert seekpath output → MACE format (segments + labels)
- Handle edge cases and fallbacks

## Implementation Plan

### Phase 1: Create Standalone Module

#### 1.1 Core Conversion Functions

```python
# seekpath_interface.py

def cell_params_to_vectors(a, b, c, alpha, beta, gamma):
    """Convert (a,b,c,α,β,γ) to 3x3 cell matrix."""
    # Standard crystallographic convention
    pass

def parse_crystal_structure(output_file):
    """Extract full structure from CRYSTAL output.

    Returns:
        dict with 'cell', 'positions', 'numbers' for seekpath
    """
    pass

def get_seekpath_bandpath(structure_dict):
    """Call seekpath and get band path.

    Returns:
        dict with 'segments', 'labels', 'point_coords', 'bravais_lattice'
    """
    pass

def convert_to_mace_format(seekpath_result, shrink_factor):
    """Convert seekpath output to MACE's expected format.

    Returns:
        (segments_list, labels_list, kpath_info)
    """
    pass
```

#### 1.2 Main Interface Function

```python
def get_accurate_bandpath(
    output_file: str,
    shrink_factor: int = 16,
    with_time_reversal: bool = True
) -> Tuple[List[List[int]], List[str], Dict[str, Any]]:
    """
    Get accurate band path using seekpath library.

    Args:
        output_file: Path to CRYSTAL .out file
        shrink_factor: SHRINK factor for integer coordinates
        with_time_reversal: Whether to use time reversal symmetry

    Returns:
        Tuple of:
        - segments: List of [x1,y1,z1,x2,y2,z2] integer coordinates
        - labels: List of k-point labels with '|' discontinuity markers
        - kpath_info: Dict with metadata (bravais_lattice, has_inversion, etc.)
    """
    pass
```

### Phase 2: Structure Extraction

#### 2.1 From CRYSTAL Output Files

Need to extract:
- **Cell vectors** (3x3 matrix) - convert from a,b,c,α,β,γ
- **Atomic positions** (Nx3 fractional coordinates)
- **Atomic numbers** (list of Z values)

The existing `CrystalOutputParser` in `d12_parsers.py` already extracts:
- `primitive_cell`: [a, b, c, alpha, beta, gamma]
- `coordinates`: List of {atom_number, x, y, z}

#### 2.2 Cell Parameter → Vector Conversion

```python
def cell_params_to_vectors(a, b, c, alpha_deg, beta_deg, gamma_deg):
    """
    Convert cell parameters to 3x3 lattice vector matrix.
    Uses standard crystallographic convention (a along x).
    """
    import numpy as np

    alpha = np.radians(alpha_deg)
    beta = np.radians(beta_deg)
    gamma = np.radians(gamma_deg)

    # Standard crystallographic convention
    va = [a, 0, 0]
    vb = [b * np.cos(gamma), b * np.sin(gamma), 0]

    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cz = np.sqrt(c**2 - cx**2 - cy**2)
    vc = [cx, cy, cz]

    return np.array([va, vb, vc])
```

### Phase 3: SeeKPath Integration

#### 3.1 Calling SeeKPath

```python
import seekpath

def get_seekpath_bandpath(cell, positions, numbers, with_time_reversal=True):
    """
    Get band path from seekpath library.

    Args:
        cell: 3x3 numpy array of lattice vectors
        positions: Nx3 array of fractional coordinates
        numbers: List of atomic numbers
        with_time_reversal: Use time reversal symmetry

    Returns:
        seekpath result dictionary
    """
    structure = (cell, positions, numbers)

    result = seekpath.get_path(
        structure,
        with_time_reversal=with_time_reversal,
        recipe='hpkot',
        threshold=1e-7,
        symprec=1e-5
    )

    return result
```

#### 3.2 SeeKPath Output Format

```python
{
    'point_coords': {
        'GAMMA': [0.0, 0.0, 0.0],
        'X': [0.5, 0.0, 0.5],
        'K': [0.375, 0.375, 0.75],  # Parametric for non-cubic!
        ...
    },
    'path': [
        ('GAMMA', 'X'),
        ('X', 'U'),
        ('K', 'GAMMA'),  # Discontinuity: U ≠ K
        ('GAMMA', 'L'),
        ...
    ],
    'has_inversion_symmetry': True,
    'augmented_path': False,
    'bravais_lattice': 'cF',
    'bravais_lattice_extended': 'cF1',
    'primitive_lattice': [[...], [...], [...]],
    'reciprocal_primitive_lattice': [[...], [...], [...]]
}
```

### Phase 4: Format Conversion

#### 4.1 Convert to MACE Format

```python
def convert_to_mace_format(seekpath_result, shrink_factor=16):
    """
    Convert seekpath output to MACE's expected format.

    MACE expects:
    - segments: List of [x1,y1,z1, x2,y2,z2] as integers (scaled by shrink)
    - labels: List with '|' markers for discontinuities
    """
    point_coords = seekpath_result['point_coords']
    path = seekpath_result['path']

    segments = []
    labels = []

    prev_end_label = None

    for i, (start_label, end_label) in enumerate(path):
        # Check for discontinuity
        if prev_end_label is not None and start_label != prev_end_label:
            labels.append('|')

        # Add start label (only if first segment or after discontinuity)
        if i == 0 or labels[-1] == '|':
            labels.append(convert_label(start_label))

        # Add end label
        labels.append(convert_label(end_label))

        # Get coordinates and scale
        start_coord = point_coords[start_label]
        end_coord = point_coords[end_label]

        scaled_segment = [
            int(round(start_coord[0] * shrink_factor)),
            int(round(start_coord[1] * shrink_factor)),
            int(round(start_coord[2] * shrink_factor)),
            int(round(end_coord[0] * shrink_factor)),
            int(round(end_coord[1] * shrink_factor)),
            int(round(end_coord[2] * shrink_factor)),
        ]
        segments.append(scaled_segment)

        prev_end_label = end_label

    return segments, labels

def convert_label(seekpath_label):
    """Convert seekpath label to CRYSTAL-compatible ASCII."""
    conversions = {
        'GAMMA': 'G',
        'SIGMA': 'SIGMA',
        'SIGMA_0': 'SIGMA_0',
        'LAMBDA': 'LAMBDA',
        'DELTA': 'DELTA',
    }

    label = seekpath_label

    # Handle subscripts (seekpath uses _0, _2, etc.)
    # Already ASCII compatible

    return conversions.get(label, label)
```

### Phase 5: Shrink Factor Handling

#### 5.1 Minimum Shrink Calculation

Parametric k-points may have irrational coordinates. Need to find appropriate shrink factor.

```python
def calculate_minimum_shrink(point_coords, tolerance=1e-6):
    """
    Calculate minimum shrink factor to represent all k-point coordinates.

    For parametric points, we need a shrink that makes coordinates
    close to integers when multiplied.
    """
    from fractions import Fraction

    max_denom = 1

    for label, coords in point_coords.items():
        for coord in coords:
            if abs(coord) > tolerance:
                # Try to find a reasonable fraction
                frac = Fraction(coord).limit_denominator(100)
                max_denom = max(max_denom, frac.denominator)

    # Round up to even number
    shrink = max_denom
    if shrink % 2 == 1:
        shrink += 1

    return max(8, shrink)  # Minimum of 8 for robustness
```

### Phase 6: Error Handling & Fallbacks

#### 6.1 Graceful Degradation

```python
def get_accurate_bandpath(output_file, shrink_factor=16, with_time_reversal=True):
    """Main interface with error handling."""

    try:
        # Try seekpath first
        structure = parse_crystal_structure(output_file)
        result = get_seekpath_bandpath(**structure, with_time_reversal=with_time_reversal)
        segments, labels = convert_to_mace_format(result, shrink_factor)

        return segments, labels, {
            'source': 'seekpath',
            'bravais_lattice': result['bravais_lattice_extended'],
            'has_inversion': result['has_inversion_symmetry']
        }

    except ImportError:
        print("WARNING: seekpath not installed. Using fallback.")
        # Fall back to existing seekpath_data dictionary
        return get_seekpath_full_kpath_legacy(...)

    except Exception as e:
        print(f"WARNING: seekpath failed ({e}). Using fallback.")
        return get_seekpath_full_kpath_legacy(...)
```

### Phase 7: Testing

#### 7.1 Test Cases

Create test script `test_seekpath_interface.py`:

```python
def test_cubic_fcc():
    """Test cF1 - should have fixed coordinates."""
    # Diamond structure
    pass

def test_hexagonal():
    """Test hP1 - should have fixed coordinates."""
    pass

def test_rhombohedral():
    """Test hR1 - has parametric coordinates."""
    # Test with different c/a ratios
    pass

def test_monoclinic():
    """Test mC1 - has parametric coordinates."""
    # Test with different beta angles
    pass

def test_discontinuity_detection():
    """Verify '|' markers are placed correctly."""
    pass

def test_label_conversion():
    """Verify label ASCII conversion."""
    pass

def compare_with_old_implementation():
    """Compare new vs old for cubic (should match)."""
    pass
```

#### 7.2 Validation Against SeeKPath Web Tool

- Upload test structures to https://seekpath.materialscloud.io/
- Compare returned paths with our implementation
- Document any discrepancies

### Phase 8: Integration with Existing Code

#### 8.1 Update `d3_kpoints.py`

```python
# At top of file
try:
    from .seekpath_interface import get_accurate_bandpath, SEEKPATH_AVAILABLE
except ImportError:
    SEEKPATH_AVAILABLE = False

# In get_seekpath_full_kpath():
def get_seekpath_full_kpath(space_group, lattice_type, out_file=None):
    """Get SeeK-path k-paths."""

    if SEEKPATH_AVAILABLE and out_file:
        # Use accurate seekpath library
        segments, labels, kpath_info = get_accurate_bandpath(out_file)
        return segments, kpath_info
    else:
        # Fall back to static dictionary (cubic only accurate)
        return get_seekpath_full_kpath_legacy(space_group, lattice_type, out_file)
```

## File Structure

```
Crystal_d3/
├── seekpath_interface.py      # NEW: Main seekpath integration
├── test_seekpath_interface.py # NEW: Test suite
├── d3_kpoints.py              # MODIFIED: Use new interface
├── CRYSTALOptToD3.py          # MODIFIED: Pass output file path
└── ...
```

## Dependencies

Add to requirements:
```
seekpath>=2.0.0
```

## Timeline

1. **Phase 1-2**: Core module + structure extraction (Day 1)
2. **Phase 3-4**: SeeKPath integration + format conversion (Day 1)
3. **Phase 5-6**: Shrink handling + error handling (Day 2)
4. **Phase 7**: Testing (Day 2)
5. **Phase 8**: Integration with existing code (Day 3)

## Success Criteria

1. All cubic structures produce identical paths to current (fixed) implementation
2. Non-cubic structures produce paths matching seekpath web tool
3. Parametric k-points vary correctly with lattice parameters
4. Discontinuities are correctly detected and marked
5. Graceful fallback when seekpath unavailable
6. All existing tests pass

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| seekpath not installed | Graceful fallback to legacy dictionary |
| Structure parsing fails | Clear error messages, fallback |
| Shrink factor too small | Auto-calculate minimum required |
| Label incompatibility | Comprehensive label mapping |

## Notes

- The legacy `seekpath_data` dictionary can be retained for fallback
- Cubic lattices work fine with static data (no parameters)
- Primary benefit is for lower-symmetry systems
