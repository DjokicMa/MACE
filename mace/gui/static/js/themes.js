/* The UI themes (1.1.6): one list for Settings › Appearance, the View ›
 * Theme menu and the command search, in groups (THEME_GROUPS) that the
 * pickers show as labeled sections. Each id other than "system" is a
 * :root[data-theme="<id>"] token set in css/tokens.css (System uses no
 * attribute: light, or dark with the OS setting). boot.js, server.py
 * (_prefs_attrs) and diagnostics.py keep the same ids; a test checks that
 * all of them agree with tokens.css. An id never changes once shipped (it
 * is the saved preference); the label can. The Publication theme of the 1.1.6
 * preview is now Settings › Appearance › Plot style (a saved "publication" loads
 * as Light with Plot style Publication: settings.js, boot.js, settings.py). */
export const THEME_GROUPS = Object.freeze([
  { id: "base", label: "Light & Dark" },
  { id: "access", label: "Accessibility" },
  { id: "editor", label: "Editor themes" },
]);
export const THEMES = Object.freeze([
  { id: "system", group: "base", label: "System", icon: "monitor", desc: "Light or dark, as your computer is set" },
  { id: "light", group: "base", label: "Light", icon: "sun", desc: "The default light theme" },
  { id: "dark", group: "base", label: "Dark", icon: "moon", desc: "The default dark theme", dark: true },
  { id: "sepia", group: "base", label: "Sepia", icon: "book-open", desc: "Brown ink on warm tan paper, like an e-reader: for long sessions", aliases: ["paper"] },
  { id: "black", group: "base", label: "True Black", icon: "moon", desc: "Pure black surfaces for OLED screens", dark: true },
  { id: "hc-light", group: "access", label: "High Contrast Light", icon: "eye", desc: "Black on white, 7:1 text (WCAG AAA), strong borders" },
  { id: "hc-dark", group: "access", label: "High Contrast Dark", icon: "eye", desc: "White on black, 7:1 text (WCAG AAA), a yellow focus ring", dark: true },
  { id: "cvd", group: "access", label: "Colorblind Safe", icon: "circle-dot", desc: "Okabe–Ito hues for accents, status, plots and axes" },
  // popular editor palettes (sources and licenses in tokens.css), by name; a family's dark and light flavors side by side
  { id: "catppuccin-mocha", group: "editor", label: "Catppuccin Mocha", icon: "file-code", desc: "Catppuccin's dark flavor: soft pastels on deep blue-gray", dark: true },
  { id: "catppuccin-latte", group: "editor", label: "Catppuccin Latte", icon: "file-code", desc: "Catppuccin's light flavor: pastels on a cool off-white" },
  { id: "dracula", group: "editor", label: "Dracula", icon: "file-code", desc: "Purple and pink accents on dark gray", dark: true },
  { id: "gruvbox-dark", group: "editor", label: "Gruvbox Dark", icon: "file-code", desc: "Warm retro yellows and oranges on brown-gray", dark: true },
  { id: "nord", group: "editor", label: "Nord", icon: "file-code", desc: "Arctic frost blues on polar night", dark: true },
  { id: "one-dark", group: "editor", label: "One Dark", icon: "file-code", desc: "Atom's One Dark: blue accents on slate", dark: true },
  { id: "solarized-dark", group: "editor", label: "Solarized Dark", icon: "file-code", desc: "Solarized on its dark teal base", dark: true },
  { id: "solarized-light", group: "editor", label: "Solarized Light", icon: "file-code", desc: "Solarized on its light cream base" },
  { id: "tokyo-night", group: "editor", label: "Tokyo Night", icon: "file-code", desc: "Neon blues and purples on night navy", dark: true },
  { id: "tokyo-night-storm", group: "editor", label: "Tokyo Night Storm", icon: "file-code", desc: "Tokyo Night's lighter storm-blue variant", dark: true },
]);
export const THEME_IDS = Object.freeze(THEMES.map((t) => t.id));
export const themeOf = (id) => THEMES.find((t) => t.id === id) || THEMES[0];
export const groupLabel = (g) => (THEME_GROUPS.find((x) => x.id === g) || { label: "" }).label;
/** Whether a theme draws light text on dark surfaces (System: as the OS is set). */
export function isDarkTheme(id) {
  if (id === "system" || !THEME_IDS.includes(id)) {
    try { return matchMedia("(prefers-color-scheme: dark)").matches; } catch (e) { return false; }
  }
  return !!themeOf(id).dark;
}
