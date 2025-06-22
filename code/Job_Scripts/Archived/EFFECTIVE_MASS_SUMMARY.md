# Effective Mass Calculation Summary

## Current Implementation Status

### ✅ Real Effective Mass Calculation is Already Implemented
The codebase has sophisticated effective mass calculations in `advanced_electronic_analyzer.py`:
- **Method**: Calculates from band structure curvature using finite differences
- **Formula**: m* = ℏ² / |d²E/dk²|
- **Features**:
  - Separate electron and hole effective masses
  - Sanity checks (0.01 to 10 m_e range)
  - Transport property calculations (mobility)

### ✅ Integration Complete
- `crystal_property_extractor.py` calls the advanced analyzer when BAND/DOSS data is available
- `update_effective_masses.py` can process existing calculations and update the database
- Successfully calculated for diamond: electron m* = 1.19 m_e, hole m* = 0.215 m_e

## Semimetal Cutoff Recommendations

### Current Implementation
The code uses a **two-level approach** for metal/semimetal classification:

1. **Band Gap Threshold**: 0.001 Ha ≈ 0.027 eV
   - If gap < 0.027 eV → Check DOS at Fermi level
   - If gap > 0.027 eV → Semiconductor or insulator

2. **DOS at Fermi Level**: g(E_F) > 0.05 × g_mean
   - If DOS(E_F) is significant → Metal
   - If DOS(E_F) is small → Semimetal

### Recommended Cutoffs

**Keep the current values** - they are well-chosen:

1. **Band Gap Cutoff: 0.027 eV (0.001 Ha)**
   - This is ~kT at room temperature
   - Appropriate for distinguishing thermal activation
   - Standard in computational materials science

2. **DOS Criterion: g(E_F) > 0.05 × g_mean**
   - Adaptive to material's DOS scale
   - Distinguishes between true metals and semimetals
   - Avoids false positives from numerical noise

3. **Classification Boundaries**:
   - **Metal**: gap < 0.027 eV AND high DOS(E_F)
   - **Semimetal**: gap < 0.027 eV AND low DOS(E_F)
   - **Semiconductor**: 0.027 eV < gap < 3.0 eV
   - **Insulator**: gap > 3.0 eV

## Usage in Workflow

To enable real effective mass calculations:

1. **Ensure BAND and DOSS calculations are included in workflow**
   ```bash
   python run_workflow.py --interactive
   # Select workflow with BAND and DOSS steps
   ```

2. **Properties are automatically extracted when calculations complete**
   - The enhanced queue manager triggers property extraction
   - Real effective masses calculated if band curvature data available

3. **Update existing calculations**
   ```bash
   python update_effective_masses.py
   ```

## Accuracy Considerations

### Good Accuracy Expected For:
- Direct gap semiconductors
- Materials with parabolic bands near extrema
- Well-converged band structure calculations

### Limited Accuracy For:
- Highly anisotropic materials
- Materials with flat bands
- Systems with band crossings near Fermi level

### Required for Good Results:
- Sufficient k-point sampling in BAND calculation
- At least 5 k-points around band extrema
- Converged SCF calculation

## Database Properties Added

When BAND/DOSS analysis is performed, these properties are stored:
- `electron_effective_mass_real` (m_e units)
- `hole_effective_mass_real` (m_e units)
- `electron_mobility_calculated` (cm²/(V·s))
- `hole_mobility_calculated` (cm²/(V·s))
- `band_gap_advanced_eV` (eV)
- `electronic_classification_advanced`
- `fermi_level_eV` (eV)
- `dos_at_fermi_level` (states/eV)