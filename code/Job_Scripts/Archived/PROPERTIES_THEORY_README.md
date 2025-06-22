# Advanced Properties Theory Documentation

## Electronic Properties Calculations

### Effective Mass (m*)

**Theory**: The effective mass is related to the curvature of the energy bands near critical points (band edges). It's defined as:

```
1/m* = (1/ℏ²) × (d²E/dk²)
```

Where:
- `E` is the energy of the band
- `k` is the wave vector  
- `ℏ` is the reduced Planck constant

**Implementation**: 
- **Estimation Method**: We estimate effective mass from band gap and energy scale
- **Hole Effective Mass**: `m_h* ≈ 0.5 × (E_gap/1.0 eV) × m_e` 
- **Electron Effective Mass**: `m_e* ≈ 0.3 × (E_gap/1.0 eV) × m_e`
- **Average Effective Mass**: `m_avg* = (m_h* × m_e*)^0.5`

**Units**: `m_e` (electron mass units)

**Note**: This is an approximation. Real effective mass calculation requires second derivatives of the band structure at specific k-points.

### Electronic Classification

**Categories**:
1. **Metal**: Finite density of states at Fermi level above threshold
2. **Semimetal**: Near-zero band gap with low DOS at Fermi level
3. **Semiconductor**: Clear band gap (0.1 - 4.0 eV)
4. **Insulator**: Large band gap (> 4.0 eV)

**Classification Logic**:
```python
if band_gap > 4.0:
    return "insulator"
elif band_gap > 0.1:
    return "semiconductor"  
elif band_gap > 0.0 and effective_mass < 0.1:
    return "semimetal"
else:
    return "metal"
```

### Transport Properties

#### Mobility (μ)

**Theory**: Charge carrier mobility relates to how quickly charge carriers move in response to an electric field:

```
μ = qτ/m*
```

Where:
- `q` is the charge
- `τ` is the scattering time
- `m*` is the effective mass

**Implementation**: Estimated using simplified model:
```python
mobility_electrons = 1000 / effective_mass_electrons  # cm²/(V·s)
mobility_holes = 800 / effective_mass_holes          # cm²/(V·s)
```

**Units**: `cm²/(V·s)`

#### Conductivity (σ)

**Theory**: Electronic conductivity depends on carrier concentration and mobility:

```
σ = q × n × μ
```

**Implementation**: Classified based on band gap and effective mass:
- **High**: Metals and small-gap semiconductors
- **Medium**: Regular semiconductors  
- **Low**: Large-gap semiconductors and insulators

**Units**: `S/m` (Siemens per meter)

#### Seebeck Coefficient (S)

**Theory**: Thermoelectric coefficient relating voltage to temperature difference:

```
S = (k_B/q) × ln(N_c/n)
```

**Implementation**: Estimated from band gap:
```python
seebeck_coeff = band_gap_ev * 80  # μV/K
```

**Units**: `μV/K` (microvolts per Kelvin)

## Crystallographic Properties

### Space Group Analysis

**Theory**: Space groups describe the symmetry operations of a crystal structure. CRYSTAL outputs the space group number according to the International Tables for Crystallography.

**Extraction**: Parsed from CRYSTAL output lines containing "SPACE GROUP".

### Lattice Parameters

**Theory**: Fundamental geometric parameters describing the unit cell:

- **Primitive Cell**: The smallest repeating unit
- **Crystallographic Cell**: Conventional unit cell (may be larger than primitive)
- **Lattice Constants**: a, b, c (lengths) and α, β, γ (angles)
- **Cell Volume**: V = abc√(1 + 2cosαcosβcosγ - cos²α - cos²β - cos²γ)

**Units**:
- Lengths: `Å` (Angstroms)
- Angles: `degrees`
- Volumes: `Å³` (cubic Angstroms)

### Atomic Positions

**Theory**: Fractional coordinates of atoms within the unit cell. CRYSTAL optimizes these positions during geometry optimization.

**Format**: Stored as JSON arrays with x, y, z coordinates for each atom.

## Electronic Structure Properties

### Band Gap Types

1. **Direct Band Gap**: Valence band maximum and conduction band minimum at same k-point
2. **Indirect Band Gap**: VBM and CBM at different k-points
3. **Alpha/Beta Gaps**: For spin-polarized calculations

**Extraction**: Parsed from CRYSTAL output patterns:
- `INDIRECT ENERGY BAND GAP:`
- `DIRECT ENERGY BAND GAP:`

### Density of States (DOS)

**Theory**: The density of states g(E) represents the number of electronic states per unit energy at energy E.

**Properties Extracted**:
- **DOS at Fermi Level**: `g(E_F)` - crucial for metallic behavior
- **Total DOS**: Integrated over all energies
- **Energy Range**: Minimum and maximum energies in DOS calculation

**Units**: `states/eV` or `states/Hartree`

### Band Structure

**Theory**: E(k) relationship showing electronic energy as a function of wave vector k.

**Properties Extracted**:
- **Fermi Energy**: Energy level where probability of occupation = 0.5
- **Valence Band Maximum (VBM)**: Highest occupied energy level
- **Conduction Band Minimum (CBM)**: Lowest unoccupied energy level
- **K-path Labels**: High-symmetry points (Γ, X, L, W, etc.)

## Population Analysis

### Mulliken Population Analysis

**Theory**: Partitions electron density among atoms and bonds:

```
Q_A = Z_A - Σ_μ∈A P_μμ - Σ_μ∈A Σ_ν∉A P_μν S_μν
```

Where:
- `Q_A` is the charge on atom A
- `Z_A` is the nuclear charge
- `P_μν` is the density matrix element
- `S_μν` is the overlap matrix element

**Properties**:
- **Atomic Charges**: Net charge on each atom
- **Bond Orders**: Electron density in bonds between atoms

**Units**: `electrons` (for charges), `dimensionless` (for overlap populations)

### Overlap Population

**Theory**: Measures covalent bonding character between atoms:

```
OP_AB = Σ_μ∈A Σ_ν∈B P_μν S_μν
```

- **Positive values**: Bonding character
- **Negative values**: Antibonding character
- **Zero**: No bonding interaction

## Computational Properties

### Optimization Convergence

**Theory**: Geometry optimization seeks the minimum energy configuration by adjusting atomic positions and lattice parameters.

**Convergence Criteria**:
- **TOLDEE**: Energy convergence threshold
- **TOLDEG**: Gradient convergence threshold  
- **TOLDEX**: Displacement convergence threshold

### SCF Convergence

**Theory**: Self-consistent field calculation iteratively solves the Kohn-Sham equations until electronic density converges.

**Properties Tracked**:
- **SCF Cycles**: Number of iterations required
- **Final Energy**: Converged total energy
- **Energy Change**: Final energy difference between cycles

## Advanced Material Classification

### Metal vs. Semimetal Classification

**Based on DOS Analysis**:

```python
def classify_from_dos(E, g, gcrit_factor=0.05):
    Ef_index = np.argmin(abs(E))     # closest to E_F = 0
    g_Ef = g[Ef_index]               # DOS at Fermi level
    g_mean = g[g > 0].mean()         # Average DOS
    g_crit = gcrit_factor * g_mean   # Threshold

    if gap > 0:
        return "semiconductor/insulator"
    elif g_Ef > g_crit:
        return "metal"
    else:
        return "semimetal"
```

**Physical Interpretation**:
- **Metals**: High DOS at Fermi level → good electrical conductivity
- **Semimetals**: Low DOS at Fermi level → poor electrical conductivity despite zero gap
- **Semiconductors**: Finite gap → activated conductivity

### Effective Mass Threshold for Semimetals

**Criterion**: Materials with `m* < 0.1 m_e` and small/zero gaps are classified as semimetals.

**Examples**:
- **Graphene**: Zero gap, very low effective mass → semimetal
- **Bismuth**: Small gap, low effective mass → semimetal
- **Silicon**: Moderate gap, moderate effective mass → semiconductor

## Implementation Notes

### Property Extraction Pipeline

1. **CRYSTAL Output Parsing**: Regular expressions extract values from text output
2. **Unit Assignment**: Intelligent unit detection based on property names
3. **Data Validation**: Check for reasonable values and physical consistency
4. **Database Storage**: Structured storage with metadata and provenance

### Approximations and Limitations

1. **Effective Mass**: Current implementation uses empirical scaling laws rather than band structure derivatives
2. **Transport Properties**: Simplified models without full Boltzmann transport calculation  
3. **Classification**: Based on band gaps and DOS, doesn't include temperature effects
4. **Scattering**: Transport calculations assume simplified scattering mechanisms

### Future Enhancements

1. **Real Effective Mass**: Calculate from actual band structure curvature
2. **Full BoltzTraP**: Integrate Boltzmann transport calculations
3. **Temperature Dependence**: Include thermal effects in transport properties
4. **Anisotropy**: Account for directional dependence in transport properties

## References

1. Ashcroft, N. W. & Mermin, N. D. "Solid State Physics" (1976)
2. Kittel, C. "Introduction to Solid State Physics" (2004)  
3. Martin, R. M. "Electronic Structure: Basic Theory and Practical Methods" (2004)
4. Madsen, G. K. H. & Singh, D. J. "BoltzTraP. A code for calculating band-structure dependent quantities" Comput. Phys. Commun. 175, 67 (2006)