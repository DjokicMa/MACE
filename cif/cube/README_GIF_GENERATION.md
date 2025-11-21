# GIF Generation for CUBE Files

## Quick Start

### Generate GIFs for a single file:
```bash
./make_gifs.sh your_file.CUBE
```

This creates 3 GIF animations in **~1-2 minutes**:
- `gif_rotation/your_file_rotation.gif` - 360° rotation (2-4 MB)
- `gif_iso_browse/your_file_iso_browse.gif` - Isovalue slider (0.5-1 MB)
- `gif_slice_browse/your_file_slice_browse.gif` - Z-axis slices (5-10 MB)

---

## Batch Processing

### Process all density files:
```bash
./make_gifs.sh *DENS*.CUBE
```

### Process specific files:
```bash
./make_gifs.sh file1.CUBE file2.CUBE file3.CUBE
```

### Process ALL CUBE files:
```bash
./make_gifs.sh *.CUBE
```

---

## What Each GIF Shows

### 1. **Rotation GIF** (`gif_rotation/`)
- 360° rotation around Z-axis
- 60 frames (6° per frame)
- 12 fps (~5 second loop)
- 1200×1200 px
- **Best for:** Presentations, posters, general 3D view

### 2. **Iso-Browse GIF** (`gif_iso_browse/`)
- Slides through 25 different isovalue levels
- 8 fps
- 1200×900 px (includes slider)
- **Best for:** Understanding density gradients, comparing regions

### 3. **Slice-Browse GIF** (`gif_slice_browse/`)
- 2D slices sliding through Z-axis
- 10 fps
- 1200×900 px (includes slider)
- **Best for:** 2D materials, layer-by-layer analysis

---

## Customization

Edit these variables at the top of `make_gifs.sh`:

```bash
ROTATION_FRAMES=60      # More frames = smoother (try 36 for faster, 90 for ultra-smooth)
ROTATION_FPS=12         # Playback speed
ISO_COUNT=25            # Number of isovalue levels
ISO_FPS=8               # Iso-browse playback speed
SLICE_FPS=10            # Slice-browse playback speed
WIDTH=1200              # Image width (try 800 for smaller, 1920 for HD)
HEIGHT=1200             # Image height (rotation)
HEIGHT_BROWSER=900      # Image height (browsers)
```

### For faster/smaller files:
```bash
ROTATION_FRAMES=36
WIDTH=800
HEIGHT=800
```

### For ultra-high quality:
```bash
ROTATION_FRAMES=90
WIDTH=1920
HEIGHT=1920
```

---

## File Type Auto-Detection

The script automatically detects and handles:
- **`*DENS*.CUBE`** - Electron density (single isosurface)
- **`*POT*.CUBE`** - Potential (shows +/− surfaces with `--show-both`)
- **`*SPIN*.CUBE`** - Spin density (shows α/β with `--show-both`)
- **`*SUBTRACTED*.CUBE`** - Density difference (symmetric ±Δρ with `--show-both`)

---

## Example: Density Difference Workflow

```bash
# 1. Create subtracted CUBE file
python crystal_cubeviz_plotly.py \
    --subtract hybrid.CUBE surface.CUBE molecule.CUBE \
    --align-grids \
    --output-cube SUBTRACTED_delta_rho.CUBE

# 2. Generate all 3 GIF types
./make_gifs.sh SUBTRACTED_delta_rho.CUBE
```

---

## Performance

**Per file (all 3 modes):**
- Rotation: ~30-40 seconds
- Iso-browse: ~20-30 seconds
- Slice-browse: ~30-40 seconds
- **Total: ~1-2 minutes**

**For 10 files:** ~10-20 minutes

---

## Output Structure

```
your_directory/
├── make_gifs.sh
├── gif_rotation/
│   ├── file1_rotation.gif
│   └── file2_rotation.gif
├── gif_iso_browse/
│   ├── file1_iso_browse.gif
│   └── file2_iso_browse.gif
└── gif_slice_browse/
    ├── file1_slice_browse.gif
    └── file2_slice_browse.gif
```

---

## Manual Commands (if you want more control)

### Rotation GIF:
```bash
python crystal_cubeviz_plotly.py file.CUBE \
    --save-gif output.gif \
    --rotation-frames 60 \
    --gif-fps 12 \
    --gif-width 1200 \
    --gif-height 1200
```

### Iso-Browse GIF:
```bash
python crystal_cubeviz_plotly.py file.CUBE \
    --iso-browse \
    --iso-count 25 \
    --save-gif output.gif \
    --gif-fps 8 \
    --gif-width 1200 \
    --gif-height 900
```

### Slice-Browse GIF:
```bash
python crystal_cubeviz_plotly.py file.CUBE \
    --slice-browse z \
    --save-gif output.gif \
    --gif-fps 10 \
    --gif-width 1200 \
    --gif-height 900
```

### For potential/spin/difference files (show both surfaces):
Add `--show-both` to any command above.

---

## Troubleshooting

### Atoms don't align with isosurface:
The script uses the default visualization (no modifications) which should work perfectly. If you see misalignment, check if you're manually adding flags like `--z-shift`.

### GIFs are too large:
Reduce `WIDTH`, `HEIGHT`, or `ROTATION_FRAMES` in the script.

### Animation is too fast/slow:
Adjust `ROTATION_FPS`, `ISO_FPS`, or `SLICE_FPS`.

### Script fails:
Make sure you have kaleido and Pillow installed:
```bash
pip install kaleido Pillow
```

---

## Tips

**For publications:**
- Rotation GIFs work great for main figures
- Iso-browse GIFs are excellent for supplementary materials

**For presentations:**
- Rotation GIFs at 1200×1200 are perfect for slides
- 60 frames gives smooth, professional-looking rotation

**For density differences:**
- Iso-browse mode is especially informative
- Script automatically uses `--show-both` for SUBTRACTED files

**For 2D materials:**
- Slice-browse shows layer-by-layer structure beautifully
- Great for interfacial systems

---

**Script:** `make_gifs.sh`
**Tested with:** CRYSTAL CUBE files (DENS, POT, SPIN, SUBTRACTED)
**Dependencies:** kaleido, Pillow (already installed)
