# Automated CUBE File Analysis

This guide covers automated generation of:
1. **Electrostatic Potential (ESP) plots** - DENS shape colored by POT
2. **Density Difference (Δρ) plots** - Charge redistribution analysis

---

## 1. Electrostatic Potential (ESP) Plots

### What it does:
- Automatically finds matching DENS and POT file pairs
- Creates dual-CUBE visualization: density isosurface colored by potential
- Shows how electrostatic potential varies across the electron density surface

### Usage:
```bash
./make_esp_gifs.sh
```

### File Matching:
Pairs are matched by removing `_DENS` or `_POT` suffix:
```
4LG_FSI_2x2_AA_opt_charge+potential_DENS.CUBE  ─┐
                                                 ├─ Match!
4LG_FSI_2x2_AA_opt_charge+potential_POT.CUBE   ─┘
```

### Output:
```
gif_esp/
├── *_ESP_rotation.gif          # 360° rotation (4-6 MB)
```

**Note:** Only rotation GIFs are created for ESP. Iso-browse and slice-browse don't work well with dual-CUBE mode.

### Settings:
- Density isovalue: 0.01 e/Bohr³ (good default for molecular/surface systems)
- Rotation: 60 frames, 12 fps, 1200×1200 px
- Automatically handles large grid files

---

## 2. Density Difference (Δρ) Plots

### What it does:
- Finds triplets: `_opt`, `_FSI_opt`, and `_graphene_opt` files
- Calculates: **Δρ = ρ(hybrid) - ρ(surface) - ρ(molecule)**
- Aligns grids automatically (handles different origins/spacings)
- Generates all 3 GIF types for the difference

### Usage:
```bash
./make_difference_gifs.sh
```

### File Matching:
The script looks for matching triplets by pattern:

```
Base pattern: 4LG_FSI_TopMiddleBottom_2x2_ABAB

Required files:
  ✓ 4LG_FSI_TopMiddleBottom_2x2_ABAB_opt_charge+potential_DENS.CUBE           (hybrid)
  ✓ 4LG_FSI_TopMiddleBottom_2x2_ABAB_FSI_opt_charge+potential_DENS.CUBE      (molecule)
  ✓ 4LG_FSI_TopMiddleBottom_2x2_ABAB_graphene_opt_charge+potential_DENS.CUBE (surface)
                                       ^^^^^^^^^^^
                                       Keywords detected: _opt, _FSI_opt, _graphene_opt
```

**All three must exist** for processing.

### Output:
```
difference_cubes/
└── *_delta_rho.CUBE            # Subtracted CUBE file

gif_difference/
├── *_delta_rho_rotation.gif      # 360° rotation (2-5 MB)
├── *_delta_rho_iso_browse.gif    # Isovalue slider (0.5-1 MB)
└── *_delta_rho_slice_browse.gif  # Z-slice animation (5-10 MB)
```

### Scientific Interpretation:

**Positive Δρ (red):** Electron accumulation
- Regions where electrons move TO during adsorption
- Indicates attractive interactions, bonding regions
- Common in bonding regions between molecule and surface

**Negative Δρ (blue):** Electron depletion
- Regions where electrons move FROM during adsorption
- Indicates charge transfer or polarization
- Common in regions of charge rearrangement

**Magnitude:**
- ~0.001 e/Bohr³: Weak physisorption (van der Waals)
- ~0.01 e/Bohr³: Strong physisorption or weak chemisorption
- ~0.1 e/Bohr³: Strong chemisorption (covalent bonding)

---

## Complete Workflow Example

### Step 1: Generate individual CUBE GIFs
```bash
# All density files
./make_gifs.sh *_DENS.CUBE

# All potential files
./make_gifs.sh *_POT.CUBE

# All spin files
./make_gifs.sh *_SPIN.CUBE
```

**Output:** `gif_rotation/`, `gif_iso_browse/`, `gif_slice_browse/`

---

### Step 2: Generate ESP plots
```bash
./make_esp_gifs.sh
```

**Output:** `gif_esp/*_ESP_rotation.gif`

**Shows:** How electrostatic potential varies across density surface

---

### Step 3: Generate density differences
```bash
./make_difference_gifs.sh
```

**Output:**
- `difference_cubes/*_delta_rho.CUBE`
- `gif_difference/*_delta_rho_*.gif` (all 3 types)

**Shows:** Charge redistribution upon adsorption

---

## File Organization

After running all scripts:

```
your_directory/
├── *.CUBE                          # Original CUBE files
│
├── gif_rotation/                   # Individual file rotations
│   ├── *_DENS_rotation.gif
│   ├── *_POT_rotation.gif
│   └── *_SPIN_rotation.gif
│
├── gif_iso_browse/                 # Individual file iso-browse
│   ├── *_DENS_iso_browse.gif
│   └── ...
│
├── gif_slice_browse/               # Individual file slice-browse
│   ├── *_DENS_slice_browse.gif
│   └── ...
│
├── gif_esp/                        # Electrostatic potential plots
│   └── *_ESP_rotation.gif          # DENS shape + POT color
│
├── difference_cubes/               # Calculated differences
│   └── *_delta_rho.CUBE
│
└── gif_difference/                 # Density difference animations
    ├── *_delta_rho_rotation.gif
    ├── *_delta_rho_iso_browse.gif
    └── *_delta_rho_slice_browse.gif
```

---

## Performance

### ESP plots:
- **Time:** ~30-60 seconds per pair (rotation only)
- **Size:** 4-6 MB per GIF
- **Grid:** Uses density file grid (usually smaller)

### Density differences:
- **Time:** ~2-5 minutes per triplet (subtraction + 3 GIFs)
- **Subtraction:** ~30-60 seconds (with grid alignment)
- **GIFs:** ~1-2 minutes each
- **Size:**
  - CUBE file: ~10-50 MB
  - Rotation: 2-5 MB
  - Iso-browse: 0.5-1 MB
  - Slice-browse: 5-10 MB

---

## Troubleshooting

### ESP: "No matching DENS/POT pairs found"
- Check that files have exactly matching base names
- Example: `base_DENS.CUBE` and `base_POT.CUBE`
- Both must end with `_DENS.CUBE` and `_POT.CUBE`

### Difference: "Skipping: missing *_FSI_opt or *_graphene_opt"
- All three files must exist: `_opt`, `_FSI_opt`, and `_graphene_opt`
- Check file naming carefully
- The script looks for these exact keywords in the filename

### Difference: "Subtraction failed"
- Grid alignment issue (script uses `--align-grids` automatically)
- Check that all three CUBE files are valid
- Verify files are from compatible calculations

### GIF generation failed
- Check that kaleido and Pillow are installed
- Verify CUBE file is valid (try manually with `python crystal_cubeviz_plotly.py file.CUBE`)
- Check disk space

---

## Advanced: Manual Commands

### Manual ESP plot:
```bash
/home/marcus/anaconda3/bin/python crystal_cubeviz_plotly.py \
    --shape-cube density.CUBE \
    --color-cube potential.CUBE \
    --shape-iso 0.01 \
    --save-gif esp_output.gif \
    --rotation-frames 60 \
    --gif-fps 12
```

### Manual density difference:
```bash
# Step 1: Calculate difference
/home/marcus/anaconda3/bin/python crystal_cubeviz_plotly.py \
    --subtract hybrid.CUBE surface.CUBE molecule.CUBE \
    --align-grids \
    --output-cube delta_rho.CUBE

# Step 2: Generate GIFs
./make_gifs.sh delta_rho.CUBE
```

---

## Tips

### For publications:
- ESP plots are great for showing potential distribution
- Δρ plots are essential for understanding bonding mechanisms
- Include both rotation and iso-browse GIFs in supplementary materials

### For presentations:
- ESP rotation GIFs show charge distribution elegantly
- Δρ iso-browse is especially informative (shows magnitude evolution)

### For analysis:
- Slice-browse GIFs of Δρ show layer-by-layer charge transfer
- Very useful for 2D materials and interfaces

---

**Scripts:**
- `make_esp_gifs.sh` - Electrostatic potential plots
- `make_difference_gifs.sh` - Density difference analysis
- `make_gifs.sh` - Individual CUBE file GIFs

**Dependencies:** kaleido, Pillow, numpy, plotly (all pre-installed)
