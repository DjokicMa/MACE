# MACE logo set

The MACE logo is the CLI banner's ANSI-Shadow wordmark (`WORDMARK` in
`mace/utils/ui.py`), redrawn cell for cell. Each `█` is a solid face block. The
banner's thin double-line shadow (`╗ ║ ╝ ═`) becomes a solid extrusion to the
lower right, so one geometry works at every size. The exact double-line version
is still available as a large "hero" mark.

Every file is plain SVG made of paths and rects. None of them uses `<text>`,
fonts or external assets. Each one has `role="img"`, `aria-label`, a `<title>`
and an explicit `width`/`height`. Gradient ids are namespaced (`mace-<file>-<role>`),
so you can inline several files in one page without id collisions.

## Files

| File | Use |
|---|---|
| `wordmark.svg` | Main wordmark for **light** backgrounds (deep gradient). 374 x 108. |
| `wordmark-on-dark.svg` | Wordmark for **dark** backgrounds (the exact CLI `crystal` palette). |
| `wordmark-mono.svg` | One color via `currentColor`. Use it for print, single-color contexts and themed UI. Extrusion faces use fill-opacity .62 (right) and .4 (bottom). |
| `lockup.svg` / `lockup-on-dark.svg` / `lockup-mono.svg` | Wordmark plus the subtitle "Mendoza Automated CRYSTAL Engine", justified to the wordmark's face width. The subtitle letters are hand-drawn monoline strokes (no font, no license issue). 374 x 146. |
| `hero.svg` / `hero-on-dark.svg` | The exact banner with the double-line outline. Only for large use (256 px wide and up): README header, splash, About dialog, slides. 420 x 144. |
| `icon.svg` | App icon and favicon on a 32 x 32 grid. A bright gradient squircle with a dark knocked-out stepped-block M (the banner's own M, 2 x 4 px cells, with a 1 x 2 px extrusion). It reads on both light and dark chrome. |
| `icon-16.svg` | Separately hinted 16 x 16 icon: integer-aligned 1 x 2 px cells, no extrusion. Use it whenever the icon renders at exactly 16 px. |
| `icon-512.png` | 512 px raster of `icon.svg` (app stores, PWA manifest, social preview). |
| `icon-192.png` | 192 px raster of `icon.svg` (the MACE GUI's web app manifest and app window). |
| `apple-touch-icon.png` | 180 px raster of `icon.svg` without its rounded corners and with no transparency (iOS and macOS round it themselves). |
| `favicon-32.png`, `favicon-16.png` | Raster favicons (16 comes from `icon-16.svg`). |

Suggested favicon markup:

```html
<link rel="icon" href="icon.svg" type="image/svg+xml">
<link rel="icon" href="favicon-32.png" sizes="32x32" type="image/png">
<link rel="icon" href="favicon-16.png" sizes="16x16" type="image/png">
```

Suggested README header (GitHub switches the image with the theme):

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/logo/hero-on-dark.svg">
  <img src="docs/logo/hero.svg" alt="MACE" width="420">
</picture>
```

## Colors

The CLI's `crystal` palette paints one color per banner row, top to bottom. The
logo turns those six row colors into one smooth vertical gradient (stops at
0, .2, .4, .6, .8 and 1 of the letter height).

| Banner row | CLI `crystal` (on-dark files, icon tile) | Deep (on-light files) |
|---|---|---|
| 1 | `#2dd4bf` teal-400 | `#0d9488` teal-600 |
| 2 | `#22d3ee` cyan-400 | `#0891b2` cyan-600 |
| 3 | `#38bdf8` sky-400 | `#0284c7` sky-600 |
| 4 | `#3b82f6` blue-500 | `#2563eb` blue-600 (the GUI `--accent`) |
| 5 | `#6366f1` indigo-500 | `#4f46e5` indigo-600 |
| 6 | `#818cf8` indigo-400 | `#6366f1` indigo-500 |

Both columns use the same hues. The deep set is one step darker, so the top
teal does not wash out on white. Other colors:

- Extrusion: each stop mixed toward `#0f172a`. On light, 34% for the right wall and 56% for the bottom. On dark, 30% and 50%.
- Icon glyph: `#0b1220`. Icon extrusion: `#0f172a` at 38% opacity over the tile.
- Subtitle: `#475569` on light, `#94a3b8` on dark.
- Hero outline: the same gradient as the faces, at 72% opacity.

## Minimum sizes

- Wordmark: 14 px tall. Below 20 px on light backgrounds, use `wordmark.svg`, never the on-dark file.
- Lockup: 240 px wide. Below that, use the wordmark alone (the subtitle gets too small to read).
- Hero: 256 px wide. Below that, the double lines turn into hairlines, so use the wordmark.
- Icon: `icon-16.svg` at exactly 16 px, and `icon.svg` from 20 px up. It is pixel-exact at 32, 64, 128 and so on.
- GUI menu bar (32 px bar): `icon.svg` at 20 px plus the one-color wordmark at 14 px tall, so the menu bar also shows the name MACE.

## Clear space

- Wordmark and lockup: keep at least one block row (the face height divided by 5, which is 20 units in the 108-unit file) clear on all sides.
- Icon: the tile is its own boundary. Keep at least 1/8 of the tile width clear around it.
- Do not recolor the gradients, add outlines or shadows, stretch the mark, or set the wordmark in a font.

## Casing rule

- **MACE** (all caps) is the product: prose, UI text, window titles, About dialog, docs headings, citation, and this logo. The wordmark always reads MACE.
- **`mace`** (lowercase) is only the literal command, Python package, paths and identifiers. Put it in code formatting in Markdown and HTML (`mace submit`). In plain terminal output and `--help` text it stays unformatted lowercase.
- Never write "Mace".
- The subtitle is written "Mendoza Automated CRYSTAL Engine". CRYSTAL is the only all-caps word because it is the program's name.

## Where the files are used

- The README header uses `hero.svg` and `hero-on-dark.svg` from this folder.
- The MACE GUI serves its own copies from `mace/gui/static/brand/` (the
  favicon, the menu bar, the About dialog, the welcome screen, and the web
  app manifest's icons, which name and draw its app window); a test keeps
  them byte-identical to the files here.

## Regenerating

The SVGs were drawn from the banner rows in `mace/utils/ui.py` as they are
in MACE 1.1.5. The generator lives outside the repository; edit the files
here, then copy the changed ones to `mace/gui/static/brand/`.
