/* The Rendering group's look on every 3Dmol viewer (polish A6): light
 * strength and direction, ambient light, outline, ambient occlusion and the
 * background, applied before every paint from the drawn document's values.
 *
 * Ambient light: 3Dmol 2.5.5's shaders start the light sum at zero
 * (vLightFront = vec3(0.0)) and know one directional light, so there is no
 * ambient term to set. installShaderPatch(v) wraps the viewer's renderer:
 * buildProgram gets each lit shader with a `maceAmbient` uniform added to
 * the light sum (vertex-lit meshes and the sphere and stick imposters), and
 * loadMaterialUniforms uploads the viewer's value. Every replacement checks
 * its exact source text first (GUARDS); a vendored 3Dmol that changed fails
 * the unit test loudly instead of silently losing the control.
 *
 * Default: exactly the look before 1.1.5: the light 3Dmol sets up, at
 * (0.2, 0.2, 1) normalized with intensity 1, no ambient, no outline. */

export const FRONT = [0.2, 0.2, 1];
export const DIRECTIONS = { front: FRONT, "upper-left": [-0.35, 0.35, 1], top: [0, 0.5, 1] };

/* [the text in the vendored shader, its replacement] */
export const GUARDS = [
  ["vLightFront = vec3( 0.0 );", "vLightFront = vec3( maceAmbient );"],
  ["vec3 vLight = directionalLightColor[ 0 ] * directionalLightWeighting;",
    "vec3 vLight = directionalLightColor[ 0 ] * directionalLightWeighting + vec3( maceAmbient );"],
  ["vec3 vLightResult = directionalLightWeighting;", "vec3 vLightResult = directionalLightWeighting + vec3( maceAmbient );"],
  ["vec3 light = vec3( max( dotProduct, 0.0 ) );", "vec3 light = vec3( max( dotProduct, 0.0 ) ) + vec3( maceAmbient );"],
];
const DECL = "uniform float maceAmbient;\n";

/** A shader source with the ambient term (unchanged when it has no light sum). */
export function patchSource(src) {
  if (typeof src !== "string" || src.includes("maceAmbient")) return src;
  let out = src, hit = false;
  for (const [a, b] of GUARDS) {
    if (out.includes(a)) { out = out.split(a).join(b); hit = true; }
  }
  return hit ? DECL + out : src;
}

const norm = (p) => { const n = Math.hypot(...p) || 1; return p.map((x) => x / n); };

/** Patch this viewer's renderer once: the ambient uniform in every lit program. */
export function installShaderPatch(v) {
  const r = v && v.renderer;
  if (!r || r.maceAmbientPatch || typeof r.buildProgram !== "function") return false;
  const build = r.buildProgram.bind(r), load = r.loadMaterialUniforms.bind(r);
  r.maceAmbient = 0;
  r.buildProgram = function (frag, vert, uniforms, params) {
    const f = patchSource(frag), vs = patchSource(vert);
    if ((f !== frag || vs !== vert) && uniforms && !uniforms.maceAmbient) uniforms.maceAmbient = { type: "f", value: r.maceAmbient };
    return build(f, vs, uniforms, params);
  };
  r.loadMaterialUniforms = function (locs, uniforms) {
    if (uniforms && uniforms.maceAmbient) uniforms.maceAmbient.value = r.maceAmbient;
    return load(locs, uniforms);
  };
  r.maceAmbientPatch = true;
  return true;
}

function lightOf(v) {
  const kids = (v.scene && v.scene.children) || [];
  return kids.find((c) => c && c.position && "intensity" in c && c.constructor && /Light/.test(c.constructor.name)) || null;
}

/** The outline color when none is chosen ("Theme"): the theme's text color
 * on the theme's background, else black or white, whichever stands out on
 * the drawn background (white for Publication, black for Presentation,
 * a custom color), so the outline never vanishes into it. `background` is
 * the setting ("theme", "white", "black", "custom"). */
const WHITE = 0xffffff;
const asHex = (n) => "#" + n.toString(16).padStart(6, "0");
export function autoOutline(background, backgroundColor, themeText) {
  const n = background === "white" ? WHITE : background === "black" ? 0
    : background === "custom" && /^#[0-9a-f]{6}$/i.test(backgroundColor || "") ? parseInt(backgroundColor.slice(1), 16) : null;
  if (n === null) return themeText;
  const luma = (0.299 * (n >> 16) + 0.587 * ((n >> 8) & 255) + 0.114 * (n & 255)) / 255;
  return asHex(luma > 0.5 ? 0 : WHITE);
}

/** Apply the values ({lightStrength, lightDirection, ambient, outline, outlineWidth,
 * outlineColor, ambientOcclusion, background}) to the viewer; cheap when nothing changed. */
export function apply(v, val) {
  if (!v || !v.scene) return;
  const key = JSON.stringify([val.lightStrength, val.lightDirection, val.ambient, val.outline, val.outlineWidth,
    val.outlineColor, val.ambientOcclusion, val.lod >= 3]);
  if (v.maceLook === key) return;
  v.maceLook = key;
  const L = lightOf(v), r = v.renderer;
  if (L) {
    const d = norm(DIRECTIONS[val.lightDirection] || FRONT);
    L.position.set ? L.position.set(d[0], d[1], d[2]) : Object.assign(L.position, { x: d[0], y: d[1], z: d[2] });
    L.intensity = Number.isFinite(+val.lightStrength) ? +val.lightStrength : 1;
    if (r) r._lightsNeedUpdate = true;
  }
  if (r && r.maceAmbientPatch) {
    r.maceAmbient = Math.max(0, Math.min(1, Number(val.ambient) || 0));
    r._currentMaterialId = -1;        // upload the uniform again with the next draw
    r._currentProgram = null;
  }
  const styles = [];
  const opt = {};
  if (val.outline) { styles.push("outline"); if (val.outlineColor) opt.color = val.outlineColor; opt.width = Number(val.outlineWidth) || 0.1; }
  // ambient occlusion turns itself off at LOD 3 and above (its cost grows with what is drawn)
  if (Number(val.ambientOcclusion) > 0 && !(val.lod >= 3)) { styles.push("ambientOcclusion"); opt.strength = Number(val.ambientOcclusion); }
  if (v.setViewStyle) {
    try { v.setViewStyle({ style: styles.join(" "), ...opt }); } catch (e) { /* WebGL1: no outline or AO */ }
  }
}
