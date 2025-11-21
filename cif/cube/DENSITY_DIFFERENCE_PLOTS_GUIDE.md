# Density Difference Plots - Scientific Guide

## What Are Density Difference Plots?

**Density difference plots** (also called **deformation density**, **charge density difference**, or **Δρ plots**) show how electron density redistributes when systems interact.

### Scientific Terminology:
- **Density Difference (Δρ)**: ρ(AB) - ρ(A) - ρ(B)
- **Deformation Density**: Changes from isolated atom reference
- **Charge Transfer**: Electron flow between systems
- **Charge Redistribution**: Polarization and reorganization effects

---

## Formula

For molecule on surface:

```
Δρ = ρ(hybrid) - ρ(surface) - ρ(molecule)
```

Where:
- ρ(hybrid) = density of combined system
- ρ(surface) = density of surface alone
- ρ(molecule) = density of molecule alone (at same position)

### Interpretation:
- **Positive Δρ** (red regions): Electron **accumulation** / gain
- **Negative Δρ** (blue regions): Electron **depletion** / loss
- **Zero Δρ** (white): No change from superposition

---

## Scientific Insights from Δρ Plots

### 1. **Bonding Character**
- **Covalent bonding**: Density accumulation **between** atoms
  - Example: Shared electron cloud in C-C bonds
- **Ionic bonding**: Depletion on one atom, accumulation on another
  - Example: Na⁺ (blue) → Cl⁻ (red)
- **van der Waals/physisorption**: Minimal redistribution (<0.01 e/Å³)
  - Example: Graphene on SiO₂

### 2. **Charge Transfer Magnitude**
Integrate Δρ over space:
```
ΔQ = ∫∫∫ Δρ dV
```

- **ΔQ > 0.1 e⁻**: Significant chemical bonding
- **ΔQ = 0.01-0.1 e⁻**: Moderate charge transfer (chemisorption)
- **ΔQ < 0.01 e⁻**: Weak interaction (physisorption)

### 3. **Interfacial Dipole Formation**
Charge redistribution creates:
- **Dipole moment** at interface
- **Work function changes** (ΔΦ)
- **Band alignment shifts**

### 4. **Orbital Hybridization**
Δρ reveals:
- **Hybridization** (sp², sp³, etc.)
- **π-π stacking** effects
- **Back-donation** in metal-organic systems

### 5. **Polarization Effects**
- **Molecular polarization**: Induced dipoles
- **Surface polarization**: Image charge effects
- **Push-back effect**: Electron cloud compression

---

## Typical Values (Δρ)

| System Type | Magnitude | Interpretation |
|-------------|-----------|----------------|
| Strong covalent bond | 0.1-0.5 e/Å³ | Shared electrons between atoms |
| Ionic bond | 0.05-0.2 e/Å³ | Charge transfer |
| Hydrogen bond | 0.01-0.05 e/Å³ | Partial charge transfer |
| Chemisorption | 0.01-0.1 e/Å³ | Chemical adsorption |
| Physisorption | <0.01 e/Å³ | van der Waals only |
| Vacuum/far field | ~10⁻⁶ e/Å³ | Numerical noise |

---

## Visualization Best Practices

### Isosurface Values:
For **density difference** (Δρ):
- **High sensitivity**: ±0.001 e/Å³ (0.001 e/Bohr³)
- **Medium**: ±0.01 e/Å³
- **Low (strong bonds)**: ±0.05 e/Å³

### Color Scheme:
- **Blue**: Electron depletion (Δρ < 0)
- **White**: No change (Δρ ≈ 0)
- **Red**: Electron accumulation (Δρ > 0)

This uses **diverging colormap** (RdBu_r in Plotly)

---

## Example Applications

### 1. **Molecule on Metal Surface**
```
FSI⁻ on graphene
```
**What to look for:**
- Blue below molecule → electron pushed away from surface
- Red above molecule → electron pulled from surface
- Net charge transfer → calculate ΔQ

**Physical interpretation:**
- Physisorption: Minimal redistribution
- Chemisorption: Significant redistribution with covalent character

### 2. **Catalysis Studies**
```
CO on Pt(111)
```
**What to look for:**
- Red between C-O → back-donation strengthens bond
- Blue on Pt → metal donates electrons
- Δρ at interface → active site character

### 3. **2D Material Heterostructures**
```
MoS₂/WSe₂ bilayer
```
**What to look for:**
- Interlayer charge transfer
- Band alignment from interfacial dipole
- Weak vs strong coupling regime

---

## Computational Details

### Grid Alignment Critical!
Your subtraction issue was caused by:
- **Origin mismatch**: 40.7 Bohr offset
- **Voxel size mismatch**: 0.162 vs 0.163 Bohr/voxel
- **Solution**: Trilinear interpolation to align grids

### Requirements for Accurate Δρ:
1. **Same basis set** for all calculations
2. **Same functional** (DFT method)
3. **Same grid resolution** (or interpolation)
4. **Same computational parameters** (tolerances, thresholds)
5. **Frozen geometry** (atoms at same positions)

### Integration Test:
Total Δρ should integrate to **zero** (charge conservation):
```python
ΔQ_total = ∫∫∫ Δρ dV ≈ 0
```
If not zero within ~0.01 e⁻, check:
- Grid alignment
- Calculation settings
- Convergence

---

## Advanced Analysis

### 1. **Bader Charge Analysis**
Partition Δρ by atom:
```
ΔQ_atom = ∫(Bader volume) Δρ dV
```

### 2. **Planar-Averaged Δρ**
For 2D materials:
```
Δρ_avg(z) = (1/A) ∫∫ Δρ(x,y,z) dx dy
```
Shows charge transfer profile across interface

### 3. **Electrostatic Potential Difference (ΔV)**
Related via Poisson equation:
```
∇²(ΔV) = -4πΔρ
```
Gives work function changes:
```
ΔΦ = -eΔV_interface
```

---

## Common Pitfalls

### 1. **Grid Mismatch** ⚠️
**Problem**: Your original issue - grids don't align

**Solution**: Use `subtract_cubes.py` with interpolation

### 2. **Different DFT Functionals**
**Problem**: ρ(hybrid, PBE) - ρ(surface, HSE) is meaningless

**Solution**: Use same functional for all calculations

### 3. **Geometry Relaxation**
**Problem**: Atoms move → mixing structural and electronic effects

**Solution**: Use frozen geometry or separate:
- Electronic: Frozen geometry Δρ
- Structural: Δρ from relaxation

### 4. **SCF Convergence**
**Problem**: Unconverged densities → artificial Δρ

**Solution**: Tight SCF thresholds (TOLDEE 8-10 in CRYSTAL)

---

## Practical Example: Your FSI⁻/Graphene System

### Setup:
1. **Hybrid**: FSI⁻ on 2x2 graphene supercell
2. **Isolated**: FSI⁻ molecule alone
3. **Difference**: Δρ = hybrid - graphene - FSI⁻

### Expected Results:

#### **Physisorption** (weak vdW):
- Small Δρ magnitude (<0.01 e/Å³)
- Symmetric depletion/accumulation
- ΔQ ≈ 0.01-0.05 e⁻ total

#### **Chemisorption** (chemical bond):
- Large Δρ (>0.05 e/Å³)
- Asymmetric charge transfer
- ΔQ > 0.1 e⁻
- Visible covalent bonding character

### Your Results:
From subtraction log:
```
Coverage: 15% of grid (molecule region)
Mean Δρ in region: 0.042 → 0.010 e/Bohr³
```

**Interpretation**:
- 0.010 e/Bohr³ = 0.0006 e/Å³ (convert: 1 Bohr = 0.529 Å)
- **This is physisorption regime!**
- Weak charge redistribution
- van der Waals dominated interaction

---

## Related Techniques

### 1. **HOMO-LUMO Difference**
Shows frontier orbital changes

### 2. **Spin Difference (Δσ)**
For magnetic systems:
```
Δσ = ρ_up - ρ_down
```

### 3. **ELF (Electron Localization Function)**
Complements Δρ for bonding analysis

### 4. **QTAIM (Quantum Theory of Atoms in Molecules)**
Bond critical points in Δρ

---

## References

### Key Papers:
1. **Density functional theory** - Hohenberg & Kohn (1964)
2. **Charge density analysis** - Bader (1990)
3. **Deformation density** - Hirshfeld (1977)
4. **Molecular adsorption** - Hammer & Nørskov (2000)

### Software:
- **CRYSTAL**: Your current tool
- **VASP**: Alternative DFT code
- **VESTA**: Visualization (also does Δρ)
- **Critic2**: Advanced charge analysis

---

## Quick Reference

### File Types:
- `*_DENS.CUBE`: Electron density (ρ)
- `*_SPIN.CUBE`: Spin density (ρ_up - ρ_down)
- `*_POT.CUBE`: Electrostatic potential (V)
- `*_subtracted.CUBE`: Density difference (Δρ)

### Commands:
```bash
# Create density difference
python subtract_cubes.py hybrid.CUBE isolated.CUBE difference.CUBE

# Visualize with automatic labels
python crystal_cubeviz_plotly.py difference.CUBE --iso-browse

# Slice view for interfacial profile
python crystal_cubeviz_plotly.py difference.CUBE --slice-browse z
```

---

## Summary

**Density difference plots** reveal:
1. ✅ Bonding character (covalent/ionic/vdW)
2. ✅ Charge transfer magnitude and direction
3. ✅ Orbital hybridization
4. ✅ Polarization effects
5. ✅ Work function changes
6. ✅ Chemical reactivity

**Critical requirements**:
- Same computational settings
- Proper grid alignment (use `subtract_cubes.py`)
- Appropriate isovalues for regime
- Diverging colormap (blue/red)

**Your corrected subtraction** now properly shows charge redistribution!
