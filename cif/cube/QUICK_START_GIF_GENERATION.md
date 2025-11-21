# Quick Start: GIF Generation for CUBE Files

## ✅ What's Working

All three GIF generation modes are fully functional:

1. ✓ **Rotation GIFs** - 360° spinning isosurface views
2. ✓ **Iso-Browse GIFs** - Animating through isovalue levels
3. ✓ **Slice-Browse GIFs** - Z-axis cross-section animations

---

## 🚀 Fastest Way to Start

### Generate ALL three types for a single file:
```bash
./generate_all_gifs.sh your_file.CUBE
```

**Example output:**
- `rotation_gifs/your_file_rotation.gif` (2.8 MB, 72 frames)
- `iso_browse_gifs/your_file_iso_browse.gif` (384 KB, 14-30 frames)
- `slice_browse_gifs/your_file_slice_browse.gif` (8 MB, ~76 frames)

---

## 📦 Batch Processing

### Process all density files:
```bash
./generate_all_gifs.sh *DENS*.CUBE
```

### Process specific files:
```bash
./generate_all_gifs.sh file1.CUBE file2.CUBE file3.CUBE
```

### Generate only one type for all files:
```bash
./generate_rotation_gifs.sh      # Only rotation (fastest)
./generate_iso_browse_gifs.sh    # Only iso-browse (small files)
./generate_slice_browse_gifs.sh  # Only slice-browse (detailed)
```

---

## 📊 What Each GIF Shows

### 1. **Rotation GIF** (Best for presentations)
- 360° rotation around Z-axis
- Shows 3D structure from all angles
- 72 frames = 5° per frame (very smooth)
- 15 FPS playback
- ~2-5 MB file size
- **Perfect for**: Talks, posters, quick 3D overview

### 2. **Iso-Browse GIF** (Best for understanding density distribution)
- Animates through different electron density thresholds
- Shows how structure changes with isovalue
- 14-30 frames (logarithmic spacing)
- 8 FPS playback
- ~0.3-1 MB file size
- **Perfect for**: Understanding density gradients, comparing core vs valence

### 3. **Slice-Browse GIF** (Best for 2D materials)
- 2D cross-sections sliding through Z-axis
- Shows layer-by-layer structure
- ~76 frames (one per Z-slice)
- 10 FPS playback
- ~8-15 MB file size
- **Perfect for**: Interfacial systems, 2D materials, charge redistribution

---

## 🎨 Automatic File Type Detection

The scripts automatically recognize and handle:

| File Pattern | What It Does |
|--------------|--------------|
| `*DENS*.CUBE` | Electron density - single isosurface |
| `*POT*.CUBE` | Potential - shows both +/− surfaces |
| `*SPIN*.CUBE` | Spin density - shows both α/β |
| `*SUBTRACTED*.CUBE` | Density difference - symmetric ±Δρ |

---

## 💡 Example Workflow

### For a density difference analysis:

```bash
# 1. Create the subtracted CUBE file
/home/marcus/anaconda3/bin/python crystal_cubeviz_plotly.py \
    --subtract hybrid.CUBE surface.CUBE molecule.CUBE \
    --align-grids \
    --output-cube SUBTRACTED_delta_rho.CUBE

# 2. Generate all three GIF types
./generate_all_gifs.sh SUBTRACTED_delta_rho.CUBE

# Output:
#   rotation_gifs/SUBTRACTED_delta_rho_rotation.gif
#   iso_browse_gifs/SUBTRACTED_delta_rho_iso_browse.gif
#   slice_browse_gifs/SUBTRACTED_delta_rho_slice_browse.gif
```

---

## ⚙️ Customization

### Edit quality settings in the scripts:

**For presentation quality (default):**
- Rotation: 72 frames, 15 FPS, 1200×1200 px
- Iso-browse: 30 isovalues, 8 FPS, 1200×900 px
- Slice-browse: All slices, 10 FPS, 1200×900 px

**For faster/smaller files:**
Edit the script variables:
```bash
ROTATION_FRAMES=36    # Fewer frames
FPS=8                 # Slower playback
WIDTH=800             # Lower resolution
HEIGHT=800
```

**For ultra-high quality:**
```bash
ROTATION_FRAMES=120   # More frames
FPS=20                # Smoother playback
WIDTH=1920            # Full HD
HEIGHT=1920
```

---

## 📂 Output Structure

After running the scripts:
```
your_directory/
├── rotation_gifs/
│   └── file1_rotation.gif           (2.8 MB)
├── iso_browse_gifs/
│   └── file1_iso_browse.gif         (0.4 MB)
└── slice_browse_gifs/
    └── file1_slice_browse.gif       (8.0 MB)
```

---

## ⏱️ Performance

**Single file (all three modes):**
- Rotation: ~20 seconds
- Iso-browse: ~10 seconds
- Slice-browse: ~20 seconds
- **Total: ~50 seconds**

**For 10 files:**
- ~8 minutes total

---

## 🐛 Troubleshooting

### Missing dependencies:
```bash
/home/marcus/anaconda3/bin/pip install kaleido Pillow
```

### Script not executable:
```bash
chmod +x generate_*.sh
```

### GIF too large:
Reduce frames or resolution in the script settings.

### Animation too fast/slow:
Change `FPS` variable in the script.

---

## 📚 Full Documentation

For complete details, see:
- **`GIF_GENERATION_GUIDE.md`** - Comprehensive documentation
- **`DENSITY_DIFFERENCE_PLOTS_GUIDE.md`** - Scientific background
- **`QUICK_START_CUBE_SUBTRACTION.md`** - Creating Δρ files

---

## ✨ Tips

### For publications:
```bash
# Use high quality for main figures
./generate_all_gifs.sh important_structure.CUBE
# Edit to 1920×1920, 120 frames for best quality
```

### For supplementary materials:
```bash
# Generate all three types for all structures
./generate_all_gifs.sh *.CUBE
```

### For presentations:
```bash
# Rotation GIFs are usually enough
./generate_rotation_gifs.sh
```

### For density differences (Δρ):
```bash
# Iso-browse is especially informative
./generate_iso_browse_gifs.sh *SUBTRACTED*.CUBE
```

---

**Generated:** 2025-10-27
**Status:** ✅ All features working
**Tested with:** `4LG_FSI_2x2_AA_FSI_opt_charge+potential_DENS.CUBE`
