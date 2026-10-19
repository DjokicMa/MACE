/* The 3D view's two preset tables (one definition: the Inspector's Lighting
 * and Style dropdowns, Settings › Viewport, the docs test and the server's
 * accepted values all read these). A preset is a set of values for its own
 * keys only, so any lighting combines with any style. "custom" is not a row:
 * it is the user's saved look (Settings renderCustom / styleCustom).
 *
 * Lighting keys: light, ambient, depth cueing, outline, ambient occlusion,
 * background, shininess (a specular highlight on the atoms).
 * Style keys: how atoms are drawn (atomDraw: auto follows the Display
 * section's atom style), colored (atomColor: scheme is Settings' color
 * scheme), their radii (atomRadii: auto is Settings › Radii), sizes,
 * polyhedra and a forced outline (styleOutline, for print). */

export const LIGHTING = [
  { id: "default", label: "Default", help: "One front light, subtle depth cueing, the theme's background",
    values: { lightStrength: 1, lightDirection: "front", ambient: 0, depthCue: 0.15, outline: false, ambientOcclusion: 0, background: "theme", shininess: 0 } },
  { id: "flat", label: "Flat", help: "Soft, even light with no depth cueing: colors read true",
    values: { lightStrength: 0.8, lightDirection: "front", ambient: 0.35, depthCue: 0, outline: false, ambientOcclusion: 0, background: "theme", shininess: 0 } },
  { id: "publication", label: "Publication", help: "White background, thin outlines and ambient occlusion, for papers",
    values: { lightStrength: 1, lightDirection: "front", ambient: 0.2, depthCue: 0, outline: true, outlineWidth: 0.1, ambientOcclusion: 0.5, background: "white", shininess: 0 } },
  { id: "presentation", label: "Presentation", help: "Black background, stronger light and depth cueing, for projectors",
    values: { lightStrength: 1.2, lightDirection: "front", ambient: 0.15, depthCue: 0.25, outline: false, ambientOcclusion: 0, background: "black", shininess: 0 } },
  { id: "glossy", label: "Glossy dark", help: "Black background, shiny atoms lit from the upper left, strong depth cueing, for slides",
    values: { lightStrength: 1.1, lightDirection: "upper-left", ambient: 0.18, depthCue: 0.45, outline: false, ambientOcclusion: 0, background: "black", shininess: 0.7 } },
];

/* needs: "polyhedra" (the structure view draws them), "spins" (Mulliken spin moments of the matching run) */
export const STYLES = [
  { id: "default", label: "Default", help: "Ball and stick with the Display section's atom style, Settings' colors and radii",
    values: { atomDraw: "auto", atomColor: "scheme", atomRadii: "auto", styleOutline: false, atomScale: 0.4, bondRadius: 0.12, showPolyhedra: false } },
  { id: "vesta", label: "VESTA-like", help: "VESTA's ball and stick: its atomic radii at 40 %, its colors, 0.12 Å bonds, whatever Settings choose",
    values: { atomDraw: "ballstick", atomColor: "vesta", atomRadii: "atomic", styleOutline: false, atomScale: 0.4, bondRadius: 0.12, showPolyhedra: false } },
  { id: "spacefill", label: "Space-filling", help: "Van der Waals spheres, no bonds",
    values: { atomDraw: "spacefill", atomColor: "scheme", atomRadii: "vdw", styleOutline: false, atomScale: 0.4, bondRadius: 0.12, showPolyhedra: false } },
  { id: "wireframe", label: "Wireframe / sticks", help: "Thin sticks only, for big cells, frameworks and supercells",
    values: { atomDraw: "sticks", atomColor: "scheme", atomRadii: "auto", styleOutline: false, atomScale: 0.4, bondRadius: 0.06, showPolyhedra: false } },
  { id: "grayscale", label: "Grayscale print", help: "Element colors as gray shades of the same lightness, with outlines",
    values: { atomDraw: "ballstick", atomColor: "gray", atomRadii: "auto", styleOutline: true, atomScale: 0.4, bondRadius: 0.12, showPolyhedra: false } },
  { id: "polyhedral", label: "Polyhedral", needs: "polyhedra",
    help: "Coordination polyhedra around cations (4 to 12 bonded neighbors, all more electronegative), in the cation's color",
    values: { atomDraw: "ballstick", atomColor: "scheme", atomRadii: "auto", styleOutline: false, atomScale: 0.3, bondRadius: 0.1, showPolyhedra: true } },
  { id: "magnetic", label: "Magnetic (spin-colored)", needs: "spins",
    help: "Atoms colored by their Mulliken spin moment: blue spin down, white none, red spin up",
    values: { atomDraw: "ballstick", atomColor: "spin", atomRadii: "auto", styleOutline: false, atomScale: 0.4, bondRadius: 0.12, showPolyhedra: false } },
];

export const LIGHTING_KEYS = ["depthCue", "lightStrength", "lightDirection", "ambient", "shininess", "outline", "outlineWidth",
  "outlineColor", "ambientOcclusion", "background", "backgroundColor"];
export const STYLE_KEYS = ["atomDraw", "atomColor", "atomRadii", "styleOutline", "atomScale", "bondRadius", "showPolyhedra"];

export const byId = (table, id) => table.find((p) => p.id === id) || null;
/** [[id, label], ...] with Custom last: the dropdowns' and Settings' options. */
export const options = (table) => table.map((p) => [p.id, p.label]).concat([["custom", "Custom"]]);
