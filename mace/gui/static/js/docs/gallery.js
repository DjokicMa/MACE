/* The Plot Gallery (MACE 1.1.6): many outputs' plots of one kind at once, as a grid of panes or one at
 * a time (cycle mode, ← → keys), with a caption table under it.
 *
 * The server's gallery routes are mace.plotting.gallery, the module `mace plotting --gallery` runs:
 * gallery/collect gives the set's files (capped at 200, with the message) and the captions (material id,
 * formula, band gap and energy from the materials.db beside the files), gallery/pane one pane's lines
 * (drawn here by plotly, structure thumbnails as SVG), gallery/export the grid as one figure (the CLI
 * itself, in the server's cache) and gallery/csv the caption table.
 *
 * Lazy rendering: a pane is read and drawn when it comes near the view (IntersectionObserver) and purged
 * when it leaves, so a 200-pane set keeps only the panes on screen in memory. The grid's panes are static
 * plots; the cycle view's one pane is interactive. */
const KINDS = [["band", "Band structure"], ["dos", "Density of states"], ["bandos", "Band structure + DOS"],
  ["ir", "IR spectrum"], ["raman", "Raman spectrum"], ["opt", "Optimization energy"], ["scf", "SCF convergence"],
  ["structure", "Structure"]];
const ENERGY_KINDS = new Set(["band", "dos", "bandos"]);
const MAX_COLS = 8, LOADS = 3, MARGIN = "300px";

/** A DOS panel's DOS-axis range (mace/plotting/gallery.py dos_axis, plotly_bands dos_range's rule): the
 *  largest |DOS| drawn (`shared`: the grid's, with shared axes) plus 8%, symmetric about 0 when the spins
 *  are mirrored, from 0 otherwise. */
export function dosAxis(pl, shared) {
  let m = shared ? Number(shared[1]) : Number(pl.dos_max) || 0;
  m = m > 0 ? m * 1.08 : 1;
  return pl.dos_side === "mirrored" ? [-m, m] : [0, m];
}

export function createGallery(host, H) {
  // Settings › Appearance › Plot style (app.js styleFigure): each pane in its family's preset
  const FAMILY = { band: "band", dos: "band", bandos: "band", ir: "spectra", raman: "spectra", opt: "scf", scf: "scf", structure: "band" };
  const famOf = (kind) => FAMILY[kind] || "band";
  const styled = (f, kind) => { if (H.styleFigure) H.styleFigure(f.traces, f.layout, famOf(kind)); return f; };
  const setting = (k, d) => { const v = H.setting ? H.setting(k) : undefined; return v === undefined || v === null || v === "" ? d : v; };
  const dosColors = () => setting("plots.dosColors", "auto");
  // Settings › Plots › DOS smoothing (mace plotting --dos-smooth): the Gaussian sigma in eV, 0 off
  // Settings › Plots › Fermi shift (the folder file's fermi_shift shows there): the grid, Export and Copy CLI
  // Command all draw with it
  const fermiShift = () => { const v = Number(setting("plots.fermiShift", 0)); return Number.isFinite(v) && Math.abs(v) <= 50 ? v : 0; };
  const dosSmooth = () => { const v = Number(setting("plots.dosSmooth", 0)); return Number.isFinite(v) && v > 0 && v <= 1 ? v : 0; };
  // Settings › Plots spin look, by mace plotting's names (--spin-colors, --spin-up/down-color, --spin-down-style,
  // --band-color: a closed-shell run's bands and DOS total)
  const hex = (v) => (/^#[0-9a-fA-F]{6}$/.test(v || "") ? v.toLowerCase() : null);
  const spinLook = () => ({ spin_colors: setting("plots.spinColors", "auto"), spin_up_color: hex(setting("plots.spinUp", "")),
    spin_down_color: hex(setting("plots.spinDown", "")), spin_down_style: setting("plots.spinDownDash", "dash"),
    band_color: hex(setting("plots.bandColor", "")) });
  const { api, q, esc, icon, toast, cssVar, loadPlotly, apiTask, download } = H;
  host.innerHTML = `<div class="gal" tabindex="-1">
    <div class="gal-bar" role="toolbar" aria-label="Gallery">
      <label class="gal-f"><span>Set</span><select class="gal-set" aria-label="Set"></select></label>
      <label class="gal-f"><span>Plot</span><select class="gal-kind" aria-label="Plot kind">${KINDS.map(([k, l]) => `<option value="${k}">${l}</option>`).join("")}</select></label>
      <label class="gal-f"><span>Columns</span><input class="gal-cols" type="number" min="1" max="${MAX_COLS}" step="1" aria-label="Columns"></label>
      <label class="gal-f"><span>Axes</span><select class="gal-axes" aria-label="Axes"><option value="shared">Shared</option><option value="independent">Independent</option></select></label>
      <span class="gal-f gal-ewin"><span>Energy</span><input class="gal-emin" type="number" step="0.5" aria-label="Energy window from (eV)">
        <span class="muted">to</span><input class="gal-emax" type="number" step="0.5" aria-label="Energy window to (eV)"><span class="muted">eV</span></span>
      <span class="gal-seg" role="group" aria-label="View">
        <button class="tb-btn-text" data-act="grid" aria-pressed="true">${icon("grid-3x3")}<span>Grid</span></button>
        <button class="tb-btn-text" data-act="cycle" aria-pressed="false">${icon("play")}<span>One at a Time</span></button></span>
      <span class="gal-acts">
        <button class="tb-btn-text" data-act="export-svg" title="The grid as one SVG figure (mace plotting --gallery)">${icon("image-down")}<span>SVG</span></button>
        <button class="tb-btn-text" data-act="export-png" title="The grid as one PNG figure (mace plotting --gallery)">${icon("image-down")}<span>PNG</span></button>
        <button class="tb-btn-text" data-act="export-csv" title="The caption table as CSV">${icon("table")}<span>Captions CSV</span></button>
        <button class="tb-btn-text" data-act="copy-cli" title="Copy the mace plotting --gallery command">${icon("copy")}<span>CLI</span></button></span>
    </div>
    <div class="gal-note small" role="status" aria-live="polite"></div>
    <div class="gal-scroll">
      <div class="gal-grid"></div>
      <div class="gal-cycle" hidden>
        <button class="tb-btn gal-prev" data-act="prev" aria-label="Previous pane (Left arrow)">${icon("chevron-left")}</button>
        <figure class="gal-pane gal-big"><figcaption></figcaption><div class="gal-plot"></div></figure>
        <button class="tb-btn gal-next" data-act="next" aria-label="Next pane (Right arrow)">${icon("chevron-right")}</button>
        <div class="gal-count mono small" aria-live="polite"></div>
      </div>
      <div class="gal-caps"><table class="data gal-table"><thead><tr><th>#</th><th>Material</th><th>Formula</th><th>Gap (eV)</th><th>Energy (eV)</th><th>File</th></tr></thead><tbody></tbody></table></div>
    </div>
  </div>`;
  const $ = (s) => host.querySelector(s);
  const root = $(".gal"), grid = $(".gal-grid"), scroller = $(".gal-scroll"), cycle = $(".gal-cycle");
  let doc = null, set = null, gen = 0, relayoutTimer = 0, refetchTimer = 0;
  const cache = new Map();          // pane key -> data (the drawn panes' data and recent ones)
  const queue = [], loading = new Set();
  let active = 0;                   // fetches in flight
  const stats = { fetched: 0, maxDrawn: 0 };

  const g = () => doc.g;
  const keyOf = (path) => `${g().kind}|${path}|${ENERGY_KINDS.has(g().kind) ? `${g().emin}|${g().emax}|${dosColors()}|${dosSmooth()}|${fermiShift()}` : ""}`;

  /* ------------------------------------------------------------ the set */
  async function collect() {
    const my = ++gen;
    unloadAll();
    grid.innerHTML = "";
    set = null;
    note(`Collecting the ${label(g().kind).toLowerCase()} files…`);
    renderBar();
    try {
      const r = await api("gallery/collect", { body: { kind: g().kind, source: g().source, db: g().db || undefined } });
      if (my !== gen) return;
      set = r;
      g().index = Math.min(g().index || 0, Math.max(0, r.items.length - 1));
      renderAll();
    } catch (e) {
      if (my !== gen) return;
      note(e.message, "error");
    }
  }
  function label(kind) { return (KINDS.find((k) => k[0] === kind) || [kind, kind])[1]; }
  function note(text, cls = "") {
    const el = $(".gal-note");
    el.className = `gal-note small ${cls}`;
    el.textContent = text || "";
  }
  function renderNote() {
    if (!set) return;
    const parts = [];
    parts.push(`${set.items.length} ${set.items.length === 1 ? "pane" : "panes"} · ${set.source_label}.`);
    if (set.message) parts.push(set.message);
    const dbs = set.captions.databases || [];
    parts.push(dbs.length ? `Captions from ${dbs.join(", ")}.` : "No materials.db with these materials: the captions name the material only.");
    note(parts.join(" "), set.capped ? "warn" : "");
  }

  /* ------------------------------------------------------------ the toolbar */
  function renderBar() {
    const s = g();
    $(".gal-kind").value = s.kind;
    $(".gal-cols").value = s.cols;
    $(".gal-axes").value = s.shared ? "shared" : "independent";
    $(".gal-emin").value = s.emin; $(".gal-emax").value = s.emax;
    $(".gal-ewin").hidden = !ENERGY_KINDS.has(s.kind);
    host.querySelectorAll(".gal-seg [data-act]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.act === s.mode)));
    const sets = (H.sets ? H.sets(doc) : []).slice();
    if (!sets.some((x) => same(x.source, s.source))) sets.unshift({ label: s.sourceLabel || "This set", source: s.source });
    const sel = $(".gal-set");
    sel.innerHTML = sets.map((x, i) => `<option value="${i}">${esc(x.label)}</option>`).join("");
    sel.value = String(Math.max(0, sets.findIndex((x) => same(x.source, s.source))));
    sel._sets = sets;
    const none = !set || !set.items.length;
    host.querySelectorAll(".gal-acts [data-act^=export], .gal-seg [data-act=cycle]").forEach((b) => { b.disabled = none; });
  }
  const same = (a, b) => JSON.stringify(a || {}) === JSON.stringify(b || {});

  $(".gal-kind").addEventListener("change", (e) => { g().kind = e.target.value; g().index = 0; collect(); });
  $(".gal-set").addEventListener("change", (e) => {
    const x = e.target._sets[Number(e.target.value)];
    if (!x) return;
    g().source = x.source; g().sourceLabel = x.label; g().db = x.db || g().db; g().index = 0;
    doc.title = `Gallery: ${x.label}`;
    if (H.retitle) H.retitle(doc);
    collect();
  });
  $(".gal-cols").addEventListener("change", (e) => {
    const n = Math.max(1, Math.min(MAX_COLS, Math.round(Number(e.target.value) || 3)));
    e.target.value = n;
    g().cols = n;
    grid.style.setProperty("--cols", n);
    requestAnimationFrame(resizeDrawn);
  });
  $(".gal-axes").addEventListener("change", (e) => { g().shared = e.target.value === "shared"; relayout(); });
  for (const sel of [".gal-emin", ".gal-emax"]) {
    $(sel).addEventListener("change", () => {
      const lo = Number($(".gal-emin").value), hi = Number($(".gal-emax").value);
      if (!(Number.isFinite(lo) && Number.isFinite(hi) && lo < hi)) {
        toast("The energy window runs from a lower to a higher energy", "error");
        $(".gal-emin").value = g().emin; $(".gal-emax").value = g().emax;
        return;
      }
      g().emin = lo; g().emax = hi;
      clearTimeout(refetchTimer);
      refetchTimer = setTimeout(redrawAll, 150);
    });
  }
  host.addEventListener("click", (e) => {
    const b = e.target.closest("[data-act]");
    if (!b || b.disabled || !host.contains(b)) return;
    const a = b.dataset.act;
    if (a === "grid" || a === "cycle") setMode(a);
    else if (a === "prev") go(g().index - 1);
    else if (a === "next") go(g().index + 1);
    else if (a === "export-svg" || a === "export-png") exportFigure(a.slice(7));
    else if (a === "export-csv") exportCsv();
    else if (a === "copy-cli") H.copyText(cliCommand(), "Copied the mace plotting command");
  });
  root.addEventListener("keydown", (e) => {
    if (!doc || g().mode !== "cycle" || e.target.closest("input, select, textarea")) return;
    const n = set ? set.items.length : 0;
    const to = { ArrowLeft: g().index - 1, ArrowRight: g().index + 1, PageUp: g().index - 1, PageDown: g().index + 1, Home: 0, End: n - 1 }[e.key];
    if (to === undefined || e.ctrlKey || e.metaKey || e.altKey) return;
    e.preventDefault();
    go(to);
  });

  function cliCommand() {
    const s = g(), inputs = (set && set.cli_inputs) || [];
    const quote = (x) => (/^[A-Za-z0-9_./=+-]+$/.test(x) ? x : `'${String(x).replace(/'/g, "'\\''")}'`);
    const a = ["mace", "plotting", "--gallery", s.kind, ...inputs.map(quote), "--cols", String(s.cols), "-o", `gallery_${s.kind}.svg`];
    if (!s.shared) a.push("--independent-axes");
    // always the window: the line draws the same figure in any folder, whatever its mace-folder.toml says
    if (ENERGY_KINDS.has(s.kind)) a.push(`--e-lower=${s.emin}`, `--e-upper=${s.emax}`);
    // and the energy zero the grid draws with (with a mace-folder.toml always: its fermi_shift and
    // energy_reference would replace them)
    const explicitRef = !!(H.folderFile && H.folderFile());
    if (ENERGY_KINDS.has(s.kind) && (fermiShift() || explicitRef)) a.push(`--fermi-shift=${fermiShift()}`);
    if (ENERGY_KINDS.has(s.kind) && explicitRef) a.push("--energy-reference=auto");
    // the plot style, DOS colors and spin look of Settings, so the line draws what the grid shows; with a
    // mace-folder.toml the defaults too (its [plots] values would replace any left out)
    const explicit = !!(H.folderFile && H.folderFile()), sp = spinLook();
    const style = H.plotPreset ? H.plotPreset(famOf(s.kind)) : "default";
    if (style !== "default" || explicit) a.push("--style", style);
    if ((s.kind === "dos" || s.kind === "bandos") && (dosColors() !== "auto" || explicit)) a.push("--dos-colors", dosColors());
    if ((s.kind === "dos" || s.kind === "bandos") && (dosSmooth() || explicit)) a.push(`--dos-smooth=${dosSmooth()}`);
    if (ENERGY_KINDS.has(s.kind)) {
      if (sp.spin_colors !== "auto" || explicit) a.push("--spin-colors", sp.spin_colors);
      if (sp.spin_colors === "auto" || sp.spin_colors === "custom") {
        if (sp.spin_up_color) a.push("--spin-up-color", sp.spin_up_color);
        if (sp.spin_down_color) a.push("--spin-down-color", sp.spin_down_color);
      }
      if (sp.spin_down_style !== "dash" || explicit) a.push("--spin-down-style", sp.spin_down_style);
      if (sp.band_color) a.push("--band-color", sp.band_color);
    }
    return a.join(" ");
  }

  /* ------------------------------------------------------------ panes */
  let io = null;
  function observer() {
    if (io) io.disconnect();
    io = new IntersectionObserver((entries) => {
      for (const en of entries) {
        const i = Number(en.target.dataset.i);
        if (en.isIntersecting) want(i);
        else unload(en.target);
      }
    }, { root: scroller, rootMargin: MARGIN });
    return io;
  }
  function renderAll() {
    renderBar();
    renderNote();
    grid.style.setProperty("--cols", g().cols);
    const items = set.items;
    grid.innerHTML = items.length ? items.map((it, i) => `<figure class="gal-pane" data-i="${i}">
      <figcaption title="${esc(it.path)}"><span class="gal-n">${i + 1}.</span> ${esc(it.title)}</figcaption>
      <div class="gal-plot" data-state="idle"></div></figure>`).join("")
      : `<p class="muted gal-empty">${esc(set.message || "Nothing to show.")}</p>`;
    const o = observer();
    grid.querySelectorAll(".gal-pane").forEach((el) => o.observe(el));
    renderTable();
    setMode(g().mode, true);
  }
  function want(i) {
    const el = grid.querySelector(`.gal-pane[data-i="${i}"] .gal-plot`);
    if (!el || el.dataset.state === "drawn" || el.dataset.state === "loading" || queue.includes(i)) return;
    queue.push(i);
    pump();
  }
  function pump() {
    queue.sort((a, b) => a - b);
    while (active < LOADS && queue.length) {
      const i = queue.shift();
      const el = grid.querySelector(`.gal-pane[data-i="${i}"] .gal-plot`);
      if (!el) continue;
      el.dataset.state = "loading";
      active++;
      const my = gen;
      paneData(i).then((d) => { if (my === gen && el.dataset.state === "loading") return draw(el, d, false); })
        .catch((e) => { if (my === gen) { el.dataset.state = "error"; el.innerHTML = `<p class="gal-err">${esc(e.message)}</p>`; } })
        .finally(() => { active--; loading.delete(i); pump(); });
    }
  }
  async function paneData(i) {
    const it = set.items[i], k = keyOf(it.path);
    if (cache.has(k)) { const d = cache.get(k); cache.delete(k); cache.set(k, d); return d; }
    const o = { kind: g().kind, path: it.path };
    if (ENERGY_KINDS.has(g().kind)) { o.emin = g().emin; o.emax = g().emax; o.dos_colors = dosColors(); o.dos_smooth = dosSmooth(); o.shift = fermiShift(); }
    const d = await api("gallery/pane?" + q(o));
    stats.fetched++;
    cache.set(k, d);
    while (cache.size > 80) cache.delete(cache.keys().next().value);
    if (g().shared) { clearTimeout(relayoutTimer); relayoutTimer = setTimeout(relayout, 120); }
    return d;
  }
  function unload(fig) {
    const el = fig.querySelector(".gal-plot");
    if (!el || fig.classList.contains("gal-big")) return;
    if (el.dataset.state === "drawn" && window.Plotly) window.Plotly.purge(el);
    const qi = queue.indexOf(Number(fig.dataset.i));
    if (qi >= 0) queue.splice(qi, 1);
    if (el.dataset.state !== "loading") { el.innerHTML = ""; el.dataset.state = "idle"; }
  }
  function unloadAll() {
    queue.length = 0;
    if (io) io.disconnect();
    grid.querySelectorAll(".gal-pane").forEach(unload);
    const big = $(".gal-big .gal-plot");
    if (big.dataset.state === "drawn" && window.Plotly) window.Plotly.purge(big);
    big.innerHTML = ""; big.dataset.state = "idle";
  }
  function redrawAll() {
    if (!set) return;
    unloadAll();
    renderBar();
    const o = observer();
    grid.querySelectorAll(".gal-pane").forEach((el) => o.observe(el));
    if (g().mode === "cycle") showCycle();
  }

  /* the union of the drawn sets' ranges on the axes the energy window does not fix (shared axes) */
  function sharedLimits() {
    const out = [];
    for (const it of set ? set.items : []) {
      const d = cache.get(keyOf(it.path));
      if (!d) continue;
      (d.panels || []).forEach((pl, k) => {
        out[k] = out[k] || { x: null, y: null };
        for (const ax of ["x", "y"]) {
          if (pl.energy === ax || pl.equal) continue;
          if (pl.dos_side) {                    // a DOS axis: its largest |DOS| (dosAxis makes the range)
            const c = out[k][ax], top = Number(pl.dos_max) || 0;
            out[k][ax] = c ? [Math.min(c[0], 0), Math.max(c[1], top)] : [0, top];
            continue;
          }
          let lo = Infinity, hi = -Infinity;
          for (const s of pl.series || []) for (const v of s[ax] || []) {
            if (v === null || (pl[ax + "log"] && v <= 0)) continue;
            if (v < lo) lo = v;
            if (v > hi) hi = v;
          }
          if (lo > hi) continue;
          const c = out[k][ax];
          out[k][ax] = c ? [Math.min(c[0], lo), Math.max(c[1], hi)] : [lo, hi];
        }
      });
    }
    return out;
  }
  function figure(d, big) {
    const s = g(), panels = d.panels || [], lims = s.shared ? sharedLimits() : [];
    const fam = famOf(d.kind || g().kind);
    const pv = (n) => (H.plotVar ? H.plotVar(n, fam) : cssVar(n));      // the family preset's tokens, on the theme's or the forced plot surface
    const text = pv("--text"), grid2 = pv("--border") || "#ddd", bg = pv("--plot-bg") || pv("--surface") || "#fff";
    const pal = Array.from({ length: 10 }, (_, k) => pv(`--plot-${k + 1}`) || pv("--plot-1"));
    // the band plot's spin pair (plot.js, Settings › Plots): up solid, down in spin down's line style; the total in ink
    const spin = H.spinStyle ? H.spinStyle("band") : { up: pv("--accent"), down: pal[0], dash: "dash" };
    const role = { up: spin.up, down: spin.down, total: pv("--plot-ink") || text };
    // DOS projections in their DOS colors (plotpresets.js dosStyles, as mace plotting --gallery)
    const PP = H.presets ? H.presets() : null, looks = {};
    const preset = H.plotPreset ? H.plotPreset("band") : "default", bgHex = /^#/.test(bg) ? bg : "#ffffff";
    for (const pl of panels) {
      const projs = (pl.series || []).filter((sr) => sr.proj && sr.spin === "up").map((sr) => sr.proj);
      // the DOS plot's rule (bandfig.js dosLook): with both spins every projection solid, an element's orbitals shaded by weight
      if (PP && projs.length) PP.dosStyles(projs, { mode: pl.dos_mode || "auto", elementColors: H.elementColors ? H.elementColors() : {},
        preset, bg: bgHex, palette: pal.slice(1).concat(pal[0]),
        spin: !!pl.dos_spin, weights: pl.dos_weights || null }).lines.forEach((l) => { looks[l.label] = l; });
      // DOS colors Auto: the preset's projection colors by weight, clear of the spin pair when both spins are drawn
      const ppal = PP && pl.dos_side && (pl.dos_colors || "auto") === "auto"
        ? PP.projectionPalette(preset, pl.dos_side === "mirrored" ? [spin.up, spin.down] : null, bgHex) : null;
      if (ppal) (pl.dos_order || []).forEach((lab, i) => { if (looks[lab]) looks[lab] = { ...looks[lab], color: ppal[i % ppal.length], dash: "solid" }; });
    }
    // the DOS totals (the 1.1.6 DOS layout, bandfig.js): filled in the spin pair, or a closed-shell run's neutral fill under its line
    const ink = role.total, tone = PP && PP.luminance(bgHex) < 0.4 ? "dark" : "light";
    const bands = PP ? PP.bandColor(preset, "band", tone, spinLook().band_color, ink) : ink;
    const fills = PP ? PP.spinFills(spin, preset) : { up: "rgba(0,0,0,0.1)", down: "rgba(0,0,0,0.1)" };
    const nfill = PP ? PP.neutralFill(bands, preset) : "rgba(0,0,0,0.08)";
    const fs = big ? 11 : 8, annotations = [];
    const layout = { margin: big ? { l: 60, r: 16, t: 8, b: 44 } : { l: 40, r: 6, t: 4, b: 26 }, showlegend: false,
      paper_bgcolor: bg, plot_bgcolor: bg, font: { size: big ? 12 : 9, color: text }, shapes: [] };
    const total = panels.reduce((t, p) => t + (p.width || 1), 0) || 1;
    const traces = [];
    let x0 = 0;
    panels.forEach((pl, k) => {
      const ax = k ? String(k + 1) : "", w = (pl.width || 1) / total;
      const xa = { domain: [x0, Math.min(1, x0 + w - (k < panels.length - 1 ? 0.015 : 0))], anchor: "y" + ax, gridcolor: grid2, zeroline: false,
        title: { text: pl.xlabel || "", standoff: 2 }, type: pl.xlog ? "log" : "linear", automargin: true };
      const ya = { anchor: "x" + ax, gridcolor: grid2, zeroline: false, title: { text: k ? "" : (pl.ylabel || ""), standoff: 2 },
        type: pl.ylog ? "log" : "linear", automargin: true, exponentformat: "power" };
      x0 += w;
      const dosAx = pl.dos_side ? (pl.energy === "y" ? "x" : "y") : null, both = pl.dos_side === "mirrored";
      if (k > 0 && pl.energy === "y" && panels[0].energy === "y") { ya.matches = "y"; ya.showticklabels = false; }
      for (const [name, axo] of [["x", xa], ["y", ya]]) {
        let r = null;
        if (pl.energy === name) r = [Number(pl.fermi || 0) + s.emin, Number(pl.fermi || 0) + s.emax];
        else if (name === dosAx) {              // the DOS axis: symmetric when mirrored, |DOS| ticks, the DOS = 0 baseline in ink
          r = dosAxis(pl, lims[k] && lims[k][name]);
          if (both && PP) { const tk = PP.mirroredTicks(r[0], r[1]); axo.tickvals = tk.vals; axo.ticktext = tk.text; }
          if (pl.baseline !== undefined && pl.baseline !== null) Object.assign(axo, { zeroline: true, zerolinecolor: ink, zerolinewidth: 1 });
        } else if (lims[k] && lims[k][name]) r = lims[k][name];
        else if (pl[name + "lim"]) r = pl[name + "lim"];
        if (r && r[0] !== r[1]) axo.range = axo.type === "log" ? r.map((v) => Math.log10(Math.max(v, 1e-300))) : r;
        else axo.autorange = true;
      }
      if (pl.ticks && pl.ticks.length) {
        xa.tickvals = pl.ticks;
        xa.ticktext = (pl.ticklabels || []).length && pl.ticklabels.length <= 12 ? pl.ticklabels : pl.ticks.map(() => "");
      }
      for (const v of pl.hlines || []) layout.shapes.push({ type: "line", xref: `x${ax} domain`, x0: 0, x1: 1, yref: "y" + ax, y0: v, y1: v,
        line: { color: pv("--text-2"), width: 1, dash: "dash" }, layer: "below" });       // a zero line (the optimization pane)
      // E_F dotted (dashed reads as spin down), across the panel at E_F
      for (const v of pl.eflines || []) layout.shapes.push(pl.energy === "y"
        ? { type: "line", xref: `x${ax} domain`, x0: 0, x1: 1, yref: "y" + ax, y0: v, y1: v, line: { color: pv("--text-2"), width: 1, dash: "dot" } }
        : { type: "line", yref: `y${ax} domain`, y0: 0, y1: 1, xref: "x" + ax, x0: v, x1: v, line: { color: pv("--text-2"), width: 1, dash: "dot" } });
      // the energies the DOS run did not compute: shaded (no data, not zero states), named in the cycle view's DOS
      for (const [g0, g1] of pl.nodata || []) {
        const fill = PP && /^#/.test(ink) ? PP.rgba(ink, 0.05) : "rgba(0,0,0,0.05)";
        layout.shapes.push(pl.energy === "y"
          ? { type: "rect", xref: `x${ax} domain`, x0: 0, x1: 1, yref: "y" + ax, y0: g0, y1: g1, line: { width: 0 }, fillcolor: fill, layer: "below" }
          : { type: "rect", yref: `y${ax} domain`, y0: 0, y1: 1, xref: "x" + ax, x0: g0, x1: g1, line: { width: 0 }, fillcolor: fill, layer: "below" });
        if (big && pl.energy === "x" && g1 - g0 >= 0.15 * (s.emax - s.emin) && PP)
          annotations.push({ xref: "x" + ax, yref: `y${ax} domain`, x: (g0 + g1) / 2, y: 0.5, textangle: -90, showarrow: false,
            text: PP.DOS_LAYOUT.no_data, font: { size: 9, color: pv("--text-3") || pv("--text-2") } });
      }
      // each spin's half named over it, in its color
      if (both && pl.energy === "y") {
        for (const [t, c, xa2, sh] of [["<b>↑ up</b>", spin.up, "left", 3], ["<b>down ↓</b>", spin.down, "right", -3]])
          annotations.push({ xref: "x" + ax, yref: `y${ax} domain`, x: 0, y: 1, xanchor: xa2, xshift: sh, yanchor: "bottom", showarrow: false,
            text: t, font: { size: fs, color: c } });
        layout.margin.t = Math.max(layout.margin.t, fs + 6);
      } else if (both) {
        for (const [t, c, y, ya2] of [["<b>↑ up</b>", spin.up, 0.97, "top"], ["<b>down ↓</b>", spin.down, 0.03, "bottom"]])
          annotations.push({ xref: `x${ax} domain`, yref: `y${ax} domain`, x: 0.01, y, xanchor: "left", yanchor: ya2, showarrow: false,
            text: t, font: { size: fs, color: c } });
      }
      if (pl.dos_side && pl.dos_smooth && PP)
        annotations.push({ xref: `x${ax} domain`, yref: `y${ax} domain`, x: 0.99, y: 0.01, xanchor: "right", yanchor: "bottom", showarrow: false,
          text: `smoothed, σ = ${pl.dos_smooth} eV`, font: { size: fs - 1, color: pv("--text-2") } });
      for (const v of pl.vlines || []) layout.shapes.push({ type: "line", yref: `y${ax} domain`, y0: 0, y1: 1, xref: "x" + ax, x0: v, x1: v,
        line: { color: grid2, width: 0.6 }, layer: "below" });
      layout["xaxis" + ax] = xa; layout["yaxis" + ax] = ya;
      let p = 0;
      for (const sr of pl.series || []) {
        const lk = sr.role === "projection" && looks[sr.proj];
        const down = sr.role === "down" || sr.spin === "down", dosTotal = pl.dos_side && sr.role === "total";
        const color = lk ? lk.color : sr.role === "projection" ? pal[(1 + p++) % pal.length]
          : dosTotal ? (both ? (down ? spin.down : spin.up) : bands)
          : sr.role === "up" && pl.closed ? bands : (role[sr.role] || text);      // one set of bands: the neutral band color
        let dash = lk ? lk.dash : "solid";
        if (down && dash === "solid") dash = spin.dash;
        const fill = dosTotal ? { fill: pl.energy === "y" ? "tozerox" : "tozeroy", fillcolor: both ? (down ? fills.down : fills.up) : nfill } : {};
        traces.push({ type: "scatter", x: sr.x, y: sr.y, name: sr.name, xaxis: "x" + ax, yaxis: "y" + ax, connectgaps: false, ...fill,
          mode: sr.markers ? "lines+markers" : "lines", line: { color, dash, width: sr.role === "up" || sr.role === "down" ? 1 : 1.3 },
          marker: { size: big ? 6 : 4, color }, hoverinfo: big ? "x+y+name" : "skip" });
      }
    });
    if (annotations.length) layout.annotations = annotations;
    return { traces, layout };
  }
  function atomsSvg(pl) {
    const a = pl.atoms;
    if (!a || !a.x.length) return `<p class="gal-err">No atoms</p>`;
    const xs = a.x.concat(pl.cell ? pl.cell.x : []), ys = a.y.concat(pl.cell ? pl.cell.y : []);
    const pad = 1.6, x0 = Math.min(...xs) - pad, x1 = Math.max(...xs) + pad, y0 = Math.min(...ys) - pad, y1 = Math.max(...ys) + pad;
    const H2 = y1 - y0, fy = (y) => (y1 - y).toFixed(3);   // y up: the top of the view box is y1
    const cell = pl.cell ? `<polyline points="${pl.cell.x.map((x, i) => `${x.toFixed(3)},${fy(pl.cell.y[i])}`).join(" ")}" fill="none" stroke="var(--text-3)" stroke-width="${(H2 / 200).toFixed(3)}"/>` : "";
    const atoms = a.x.map((x, i) => `<circle cx="${x.toFixed(3)}" cy="${fy(a.y[i])}" r="${Math.max(0.2, a.r[i] * 0.4).toFixed(3)}" fill="${esc(a.color[i])}" stroke="var(--text-2)" stroke-width="${(H2 / 400).toFixed(3)}"><title>${esc(a.symbol[i])}</title></circle>`).join("");
    return `<svg class="gal-atoms" viewBox="${x0.toFixed(3)} 0 ${(x1 - x0).toFixed(3)} ${H2.toFixed(3)}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="${esc(pl.formula || "structure")}">${cell}${atoms}</svg>`;
  }
  async function draw(el, d, big) {
    if (d.error || !(d.panels || []).length) {
      el.dataset.state = "error";
      el.innerHTML = `<p class="gal-err">${esc(d.error || "Nothing to draw")}</p>`;
      return;
    }
    if (d.panels[0].atoms) {
      el.innerHTML = atomsSvg(d.panels[0]);
      el.dataset.state = "drawn";
      el.dataset.svg = "1";
      count();
      return;
    }
    await loadPlotly();
    if (!window.Plotly) { el.dataset.state = "error"; el.innerHTML = `<p class="gal-err">plotly is not installed here</p>`; return; }
    if (el.dataset.state !== "loading" && !big) return;            // scrolled away meanwhile
    el.innerHTML = "";
    delete el.dataset.svg;
    const f = styled(figure(d, big), d.kind || g().kind);
    await window.Plotly.newPlot(el, f.traces, f.layout, { displayModeBar: big, staticPlot: !big, responsive: false, displaylogo: false });
    el.dataset.state = "drawn";
    count();
  }
  function count() { stats.maxDrawn = Math.max(stats.maxDrawn, drawnCount()); }
  function drawnCount() { return grid.querySelectorAll('.gal-plot[data-state="drawn"]').length; }
  function relayout() {
    if (!set || !window.Plotly) return;
    grid.querySelectorAll('.gal-pane .gal-plot[data-state="drawn"]').forEach((el) => {
      if (el.dataset.svg) return;
      const i = Number(el.closest(".gal-pane").dataset.i), d = cache.get(keyOf(set.items[i].path));
      if (d) { const f = styled(figure(d, false), d.kind || g().kind); window.Plotly.react(el, f.traces, f.layout, { displayModeBar: false, staticPlot: true }); }
    });
    if (g().mode === "cycle") showCycle();
  }
  function resizeDrawn() {
    if (!window.Plotly) return;
    grid.querySelectorAll('.gal-plot[data-state="drawn"]').forEach((el) => { if (!el.dataset.svg) window.Plotly.Plots.resize(el); });
  }

  /* ------------------------------------------------------------ cycle mode */
  function setMode(mode, quiet = false) {
    if (!doc) return;
    if (mode === "cycle" && (!set || !set.items.length)) mode = "grid";
    g().mode = mode;
    grid.hidden = mode === "cycle";
    cycle.hidden = mode !== "cycle";
    host.querySelectorAll(".gal-seg [data-act]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.act === mode)));
    if (mode === "cycle") { showCycle(); if (!quiet) root.focus(); }
    else {
      const big = $(".gal-big .gal-plot");
      if (window.Plotly && big.dataset.state === "drawn") window.Plotly.purge(big);
      big.innerHTML = ""; big.dataset.state = "idle";
      if (!quiet) requestAnimationFrame(() => focusPane(g().index, false));
    }
    renderTable();
  }
  function go(i) {
    if (!set || !set.items.length) return;
    const n = set.items.length;
    g().index = ((i % n) + n) % n;           // wraps around both ends
    showCycle();
    renderTable();
  }
  async function showCycle() {
    if (!set || !set.items.length) return;
    const i = g().index, it = set.items[i], my = gen;
    $(".gal-big figcaption").innerHTML = `<span class="gal-n">${i + 1}.</span> ${esc(it.title)} <span class="muted small">${esc(it.path)}</span>`;
    $(".gal-count").textContent = `${i + 1} / ${set.items.length}`;
    const el = $(".gal-big .gal-plot");
    el.dataset.state = "loading";
    el.dataset.i = String(i);
    try {
      const d = await paneData(i);
      if (my !== gen || g().index !== i || g().mode !== "cycle") return;
      if (window.Plotly && !el.dataset.svg && el.querySelector(".plot-container")) window.Plotly.purge(el);
      await draw(el, d, true);
    } catch (e) {
      if (my === gen) { el.dataset.state = "error"; el.innerHTML = `<p class="gal-err">${esc(e.message)}</p>`; }
    }
  }
  function focusPane(i, flash = true) {
    const fig = grid.querySelector(`.gal-pane[data-i="${i}"]`);
    if (!fig) return;
    fig.scrollIntoView({ block: "nearest" });
    if (flash) { fig.classList.add("flash"); setTimeout(() => fig.classList.remove("flash"), 1000); }
  }

  /* ------------------------------------------------------------ captions */
  const num = (v, d) => (v === null || v === undefined ? "—" : Number(v).toFixed(d));
  function renderTable() {
    const tb = $(".gal-table tbody");
    if (!set) { tb.innerHTML = ""; return; }
    tb.innerHTML = set.captions.rows.map((r, i) => `<tr data-i="${i}" tabindex="0"${g().mode === "cycle" && i === g().index ? ' aria-current="true"' : ""}>
      <td class="num">${r.n}</td><td>${esc(r.material_id)}</td><td>${esc(r.formula || "—")}</td><td class="num">${num(r.band_gap_eV, 3)}</td>
      <td class="num">${num(r.energy_eV, 4)}</td><td class="mono small gal-file" title="${esc(r.file)}">${esc(r.file)}</td></tr>`).join("");
  }
  $(".gal-table tbody").addEventListener("click", (e) => {
    const tr = e.target.closest("tr[data-i]");
    if (!tr) return;
    const i = Number(tr.dataset.i);
    if (g().mode === "cycle") go(i);
    else { g().index = i; focusPane(i); }
  });
  $(".gal-table tbody").addEventListener("keydown", (e) => { if (e.key === "Enter") { const tr = e.target.closest("tr[data-i]"); if (tr) tr.click(); } });
  grid.addEventListener("dblclick", (e) => {
    const fig = e.target.closest(".gal-pane");
    if (fig && set && H.openPath) H.openPath(set.items[Number(fig.dataset.i)].path);
  });

  /* ------------------------------------------------------------ export */
  async function exportFigure(fmt) {
    if (!set || !set.items.length) return;
    const s = g();
    const body = { kind: s.kind, paths: set.items.map((x) => x.path), cols: s.cols, shared: s.shared, emin: s.emin, emax: s.emax, format: fmt, detach: true,
      style: H.plotPreset ? H.plotPreset(famOf(s.kind)) : "default", dos_colors: dosColors(), dos_smooth: dosSmooth(), ...spinLook(),
      ...(ENERGY_KINDS.has(s.kind) ? { shift: fermiShift() } : {}) };
    try {
      const r = await apiTask("gallery/export", body);
      if (!r || r.canceled || !r.key) return;
      const img = await api("plot-image?" + q({ key: r.key, name: r.name }), { blob: true });
      download(img.blob, r.name);
    } catch (e) { toast(`Export: ${e.message}`, "error"); }
  }
  async function exportCsv() {
    if (!set || !set.items.length) return;
    try {
      const r = await api("gallery/csv", { body: { paths: set.items.map((x) => x.path), db: g().db || undefined }, blob: true });
      download(r.blob, `gallery_${g().kind}_captions.csv`);
    } catch (e) { toast(`Captions CSV: ${e.message}`, "error"); }
  }

  return {
    open(d) {
      const fresh = doc !== d;
      if (fresh && doc) unloadAll();
      doc = d;
      if (fresh) { cache.clear(); return collect(); }
      renderBar();
      return null;
    },
    hide() { if (doc) unloadAll(); },
    reload: () => collect(),
    go, setMode,
    /* for the tests: what is on screen */
    state: () => ({ kind: doc && g().kind, mode: doc && g().mode, index: doc && g().index, cols: doc && g().cols, shared: doc && g().shared,
      items: set ? set.items.length : 0, total: set ? set.total : 0, capped: !!(set && set.capped), message: set ? set.message : null,
      drawn: drawnCount(), maxDrawn: stats.maxDrawn, fetched: stats.fetched, loading: active + queue.length,
      cycleDrawn: $(".gal-big .gal-plot").dataset.state === "drawn", note: $(".gal-note").textContent }),
    figure: (i) => { const el = i === "big" ? $(".gal-big .gal-plot") : grid.querySelector(`.gal-pane[data-i="${i}"] .gal-plot`);
      return el && el.data ? { traces: el.data.length, layout: { x: el.layout.xaxis && el.layout.xaxis.range, y: el.layout.yaxis && el.layout.yaxis.range } } : null; },
    command: () => cliCommand(),
  };
}
