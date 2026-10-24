/* Plot documents (spec §1.2, plan B5): a band structure, a DOS, both on one
 * energy axis, an IR or a Raman spectrum, drawn by plotly from the package
 * (loaded on the first plot only, §8.5) from the arrays mace plotting's own
 * readers return. Each document has its Plot options in the inspector, with
 * their own undo (Mod+Z on the plot) and Reset; the Fermi line can be dragged
 * to shift the energy reference; hovering a band reads out k, E − E_F and the
 * band. Export: PNG/SVG from plotly, the publication figure from mace
 * plotting itself (a task), and the plotted data as CSV. */
import * as PK from "../plotkit.js";
import * as K from "../kpath.js";
import * as KE from "../kpath_editor.js";
import { createBZ } from "../bz_view.js";
import * as BF from "./bandfig.js";
import { cliCommand } from "./clicmd.js";
import { robustRange } from "../maprange.js";
import * as L from "../labels.js";

const MERGE_MS = 800;

export function createPlotView(host, optsHost, H) {
  const { api, q, esc, icon, toast, loadPlotly, apiTask, download } = H;
  // a token as the plots draw it: the theme's, or the forced plot surface's (Settings › Appearance › Plot background)
  // ... in the plot family's preset (Settings › Appearance › Plot style): band & DOS, spectra, maps, SCF & run charts
  const familyOf = (d) => (!d ? "band" : d.kind === "map" ? "maps" : d.kind === "convergence" ? "scf" : d.kind === "ir" || d.kind === "raman" ? "spectra" : "band");
  // a plot's own Plot style (Plot options › Style) over its family's (Settings › Appearance › Plot style)
  const ownPreset = (fam) => { const s = doc && doc.opts && doc.opts.style, pp = H.presets ? H.presets() : null; return s && pp ? pp.resolveId(s, fam) : null; };
  const presetOf = (fam) => ownPreset(fam) || (H.plotPreset ? H.plotPreset(fam) : "default");
  /* a plot's background (Plot options › Style › Background, else Settings › Plots › Background), as the 3D
     view's: theme, white, black or a custom color; its lightness gives the plot the light or dark tokens */
  const HEX = /^#[0-9a-f]{6}$/i;
  const bgOf = (d) => { const o = (d && d.opts) || {}; return { mode: o.background || setting("plots.background", "theme"), color: o.background_color || setting("plots.backgroundColor", "") }; };
  const bgHex = (d) => {
    const b = bgOf(d);
    if (b.mode === "white") return H.cssVar("--viewport-white");
    if (b.mode === "black") return H.cssVar("--viewport-black");
    if (b.mode === "custom") return HEX.test(b.color) ? b.color : H.cssVar("--viewport-white");
    return "";
  };
  const surfaceOf = (d) => { const h = bgHex(d); if (!HEX.test(h)) return null; const n = parseInt(h.slice(1), 16);
    return (0.299 * (n >> 16) + 0.587 * ((n >> 8) & 255) + 0.114 * (n & 255)) / 255 > 0.5 ? "light" : "dark"; };
  const tokenOf = (fam) => (n) => { const pid = ownPreset(fam), sf = surfaceOf(doc); return pid && H.plotVarIn ? H.plotVarIn(n, pid, sf) : H.plotVar ? H.plotVar(n, fam, sf) : H.cssVar(n); };
  const cssVar = (n) => tokenOf(familyOf(doc))(n);
  // the light- or dark-plot data colors (spin pair, band ink): the plot's own background's lightness, else the theme's
  const toneOf = (d) => surfaceOf(d) || (H.plotTone ? H.plotTone() : "light");
  const paper = () => {
    const h = bgHex(doc), sf = surfaceOf(doc);
    if (sf) el.setAttribute("data-plot-surface", sf); else el.removeAttribute("data-plot-surface");
    return h ? { paper_bgcolor: h, plot_bgcolor: h } : H.plotPaper ? H.plotPaper() : { paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)" };
  };
  // the plot's own options that Make This the Default, Follow Settings and the style marks cover: the presets' and the background
  const styleNames = () => (H.presets ? H.presets().STYLE_OPTIONS.map((x) => x.name) : []).concat(["background", "background_color"]);
  host.innerHTML = `<div class="pv" tabindex="-1">
    <div class="pv-head"><div class="pv-title"></div><div class="pv-chips"></div><div class="pv-acts"></div></div>
    <div class="pv-note muted small"></div>
    <div class="pv-plot" tabindex="0" aria-label="Plot"></div>
    <div class="pv-readout mono" aria-live="off"></div>
    <img class="pv-img" alt="" hidden>
    <iframe class="pv-frame" sandbox="allow-scripts" title="Plot" hidden></iframe></div>`;
  const $ = (s) => host.querySelector(s);
  /* the title is cut to one line when the pane is narrow: its tooltip has the whole of it */
  const setTitle = (t) => { const e = $(".pv-title"); e.textContent = t; e.title = t; };
  const el = $(".pv-plot");
  let doc = null, building = false, kit = null;

  /* plotly's "responsive" follows window resizes only. A pane that changed size while the plot was
     hidden (another tab showing, the inspector or bottom panel toggled, a splitter dragged) refits
     when it shows again, at most once per frame. */
  let fitQueued = false;
  function fit() {
    const fl = el._fullLayout;
    if (!window.Plotly || !fl || building || !el.clientWidth || !el.clientHeight) return;
    if (Math.abs(fl.width - el.clientWidth) > 4 || Math.abs(fl.height - el.clientHeight) > 4) window.Plotly.Plots.resize(el);
  }
  if (typeof ResizeObserver === "function") {
    new ResizeObserver(() => {
      if (fitQueued) return;
      fitQueued = true;
      requestAnimationFrame(() => { fitQueued = false; fit(); });
    }).observe(el);
  }

  /* ------------------------------------------------------------ options and their history */
  const DEFAULTS = {
    bands: { emin: -6, emax: 6, shift: 0, spin: "both", width: 1.5, hidden: null, kpath: null,
      // the plot's own style (Plot options › Style; null: Settings), by the presets.py STYLE_OPTIONS names
      style: null, spin_colors: null, spin_up_color: null, spin_down_color: null, band_color: null, band_style: null, spin_down_style: null, dos_colors: null, dos_spin: null, dos_total: null, dos_smooth: null,
      // Reference and Labels (the toolbar's menus): the energy zero (null: auto), the dotted E_F line, the k-point labels and the spin words;
      // a fort.25: electronic or phonon (null: from the run) and which of its runs (null: the last)
      reference: null, fermi_line: true, k_labels: true, spin_labels: true, f25_domain: null, f25_run: null },
    spectrum: { xmin: null, xmax: null, normalize: false, component: "total", average: false, width: 1.5, style: null },
    // BOLTZTRA transport data (plan E8): the quantity (null: the opened file's own; another file of the run
    // beside it, or one derived from them: pf, kappa_e, zt, lorenz, sigma_tau, pf_tau), the temperature (a
    // value, or "all"), the tensor component, the x axis (mu or carriers), and for PF, κₑ and ZT a relaxation
    // time tau (fs) and for ZT a lattice κ_L (kl: W/(m·K) or a T,kappa_L CSV in the folder): the numbers and
    // curves of mace plotting --transport (mace.utils.transport_plot); ZT's peaks are mace analyze's
    transport: { quantity: null, T: null, component: "avg", x: "mu", width: 1.5, kl: "", tau: null, style: null },
    // Compare Convergence (plan E12 fallback): energy per atom above the lowest final energy of the set
    convergence: { width: 1.6, log: false, style: null },
    // a fort.25 2D map (plan E6): which map (density, spin), log scale, contour count, contour style (null: the
    // plot style's), color range (robust: the 1st to 99th percentile, symmetric about 0 for a signed map, so a
    // nuclear spike does not take the whole scale; full: min to max; custom: zmin to zmax)
    map: { which: 0, log: true, contours: 20, contour_style: null, crange: "robust", zmin: null, zmax: null, style: null },
    // a spectrum broadened from a FREQ output's modes (plan E4): width and line shape; fwhm null = the run's DAMPFAC or 8
    modes: { xmin: null, xmax: null, normalize: false, component: "total", average: false, width: 1.5, fwhm: null, shape: "lorentzian", eta: 1, style: null },
  };
  // the shared options (plotkit): log axes, units, ranges; new documents take the Settings defaults
  const setting = (k, dflt) => { try { const v = H.setting ? H.setting(k) : undefined; return v === undefined || v === null || v === "" ? dflt : v; } catch (e) { return dflt; } };
  const isSpec = (d) => d.kind === "ir" || d.kind === "raman";
  // the plots of plan E (a fort.25 map, Compare Convergence, BOLTZTRA data) keep their own options only
  const isOwn = (d) => d.kind === "map" || d.kind === "convergence" || d.kind === "transport";
  const defaultsFor = (d) => {
    if (isOwn(d)) return { background: null, background_color: null, ...(d.kind === "map" ? DEFAULTS.map : d.kind === "convergence" ? DEFAULTS.convergence : DEFAULTS.transport) };
    const own = d.source === "modes" ? { ...DEFAULTS.modes, shape: H.lineShape ? H.lineShape(d.kind) : "lorentzian" } : isSpec(d) ? DEFAULTS.spectrum : DEFAULTS.bands;
    // band/DOS: Settings' energy window and Fermi shift (a folder's mace-folder.toml [plots] may set them)
    const win = own === DEFAULTS.bands ? setting("plots.energyWindow", null) : null;
    const band = Array.isArray(win) && win.length === 2 ? { emin: Number(win[0]), emax: Number(win[1]), shift: Number(setting("plots.fermiShift", 0)) || 0 } : {};
    return { ...own, ...band, background: null, background_color: null, ylog: false, xlog: false, xr: null, yr: null,
      unit: isSpec(d) ? "cm-1" : setting("energyUnit", "eV"), width: Number(setting("plots.lineWidth", 1.5)) || 1.5 };   // Settings › Plots › Line width (1.5)
  };
  const key = (d) => `mace.gui.plot.${H.dirHash()}.${d.path}${d.source ? `|${d.source}|${d.kind}` : ""}`;
  function load(d) {
    if (d.opts) return;
    let saved = null;
    try { saved = JSON.parse(localStorage.getItem(key(d)) || "null"); } catch (e) { saved = null; }
    d.opts = { ...defaultsFor(d), ...((saved && saved.opts) || {}) };
    // the 1.1.6 preview's map option "style" was the contour style (now contour_style; "style" is the Plot style)
    if (d.kind === "map" && ["filled", "lines", "both"].includes(d.opts.style)) { d.opts.contour_style = d.opts.style; d.opts.style = null; }
    if ("dosColors" in d.opts) { if (d.opts.dosColors && !d.opts.dos_colors) d.opts.dos_colors = d.opts.dosColors; delete d.opts.dosColors; }   // the 1.1.6 preview's name
    d.hist = (saved && saved.hist) || [{ label: "Opened", opts: { ...d.opts }, t: Date.now() }];
    d.pos = saved && Number.isInteger(saved.pos) ? saved.pos : d.hist.length - 1;
  }
  function persist(d) {
    try { localStorage.setItem(key(d), JSON.stringify({ opts: d.opts, hist: d.hist.slice(-60), pos: Math.min(d.pos, 59) })); } catch (e) { /* private */ }
  }
  function setOpt(d, patch, label, mergeKey = null) {
    Object.assign(d.opts, patch);
    d.hist = d.hist.slice(0, d.pos + 1);
    const last = d.hist[d.hist.length - 1];
    if (mergeKey && last && last.mergeKey === mergeKey && Date.now() - last.t < MERGE_MS) d.hist[d.hist.length - 1] = { label, opts: { ...d.opts }, t: Date.now(), mergeKey };
    else d.hist.push({ label, opts: { ...d.opts }, t: Date.now(), mergeKey });
    d.pos = d.hist.length - 1;
    persist(d);
    redraw();
    renderOptions();
    if (kit) kit.refresh();
  }
  function undo(d) { if (d.pos <= 0) return null; d.pos -= 1; d.opts = { ...d.hist[d.pos].opts }; persist(d); redraw(); renderOptions(); return d.hist[d.pos + 1].label; }
  function redo(d) { if (d.pos >= d.hist.length - 1) return null; d.pos += 1; d.opts = { ...d.hist[d.pos].opts }; persist(d); redraw(); renderOptions(); return d.hist[d.pos].label; }
  function reset(d) { setOpt(d, defaultsFor(d), "Reset plot options"); }

  /* ------------------------------------------------------------ data */
  // the options the data are read with (the energy zero, a fort.25's domain and run): a change reads them again
  const dataKey = (d) => { const o = d.opts || {}; return JSON.stringify([o.reference || null, o.f25_domain || null, o.f25_run || null, o.pair || null]); };
  const staleData = (d) => !!(d.data && d.dataKey && d.dataKey !== dataKey(d) && (d.kind === "band" || d.kind === "dos"));
  async function fetchData(d) {
    if (d.data) return d.data;
    if (d.kind === "transport") {
      await fetchTransport(d);
      return d.data;
    }
    if (d.kind === "map") {
      d.data = { map: await api("map?" + q({ path: d.path })) };
      return d.data;
    }
    if (d.kind === "convergence") {
      d.data = { convergence: await api("convergence?" + d.paths.map((p) => "path=" + encodeURIComponent(p)).join("&")) };
      return d.data;
    }
    if (d.source === "modes") {
      const o = d.opts || {};
      const s0 = await api("spectrum?" + q({ path: d.path, source: "modes", kind: d.kind, fwhm: o.fwhm ?? "", shape: o.shape || "lorentzian", eta: o.eta ?? 1 }));
      d.data = { spectra: [s0], peaks: s0.peaks || [], out: s0.out };
      return d.data;
    }
    if (d.kind === "ir" || d.kind === "raman") {
      const s0 = await api("spectrum?" + q({ path: d.path }));
      d.data = { spectra: [s0], peaks: s0.peaks || [], out: s0.out || null };
      if (s0.group && s0.group.length > 1) d.data.group = s0.group;
      if (d.opts && d.opts.average && d.data.group) d.data.average = await api("spectrum?" + q({ path: d.path, average: 1 })).catch(() => null);
      return d.data;
    }
    const o = d.opts || {}, extra = { ...(o.reference ? { reference: o.reference } : {}), ...(o.f25_domain ? { domain: o.f25_domain } : {}), ...(o.f25_run ? { run: o.f25_run } : {}) };
    d.dataKey = dataKey(d);
    const first = await api((d.kind === "dos" ? "dos" : "bands") + "?" + q({ path: d.path, ...extra }));
    let band = d.kind === "dos" ? null : first, dos = d.kind === "dos" ? first : null;
    // the other half: Add DOS… / Add Band… (o.pair, from any folder), else the same run's beside the file
    const other = o.pair || first.partner;
    let pair = null;
    if (other) {
      try {
        const half = await api((d.kind === "dos" ? "bands" : "dos") + "?" + q({ path: other, ...extra }));
        if (d.kind === "dos") band = half; else dos = half;
      } catch (e) { if (o.pair) pair = { path: o.pair, error: e.message }; }
    }
    if (o.pair && band && dos) {
      try { pair = await pairData(d, band, dos, extra); } catch (e) { pair = { path: o.pair, error: e.message }; }
      if (pair.error) { if (d.kind === "dos") band = null; else dos = null; }
      else ({ band, dos } = pair.halves);
    }
    d.data = { band, dos, pair };
    return d.data;
  }

  /* ------------------------------------------------------------ band + DOS from two folders (1.1.6) */
  /* Add DOS… on a band plot, Add Band… on a DOS plot: the other half of the same system from any folder of the
     served one, checked by the server (bandos-pair: the same formula, else refused; the cell and E_F, else
     warned), on one energy zero: the vacuum scale only when both halves have it, and the DOS moved onto the
     band run's E_F when the two differ (mace plotting --band-dos --partner draws the same). The pairing is a
     plot option (pair: the other file's path), so it is kept per document, undone and reset as the others. */
  const isF25 = (p) => /\.f25$|(^|\/)fort\.25$/i.test(p || "");
  const halfKind = (f) => (f.kind === "band" || f.kind === "dos" ? f.kind : f.kind === "f25" ? (f.sub === "BAND" ? "band" : f.sub === "DOSS" ? "dos" : null) : null);
  const dirOf = (p) => (p.includes("/") ? p.slice(0, p.lastIndexOf("/")) : "");
  /** ``to`` relative to the folder ``from`` (both served paths): what --partner names in the folder of the file. */
  function relPath(from, to) {
    const a = from ? from.split("/") : [], b = to.split("/");
    let k = 0;
    while (k < a.length && k < b.length - 1 && a[k] === b[k]) k++;
    return [...a.slice(k).map(() => ".."), ...b.slice(k)].join("/");
  }
  async function pairData(d, band, dos, extra) {
    const bandPath = d.kind === "dos" ? d.opts.pair : d.path, dosPath = d.kind === "dos" ? d.path : d.opts.pair;
    const check = await api("bandos-pair?" + q({ band: bandPath, dos: dosPath }));
    if (check.error) return { path: d.opts.pair, error: check.error, check };
    // one energy zero: a POTC run beside one half only puts that half on the vacuum scale, so both are read on E_F
    if (band.reference !== dos.reference) {
      const fx = { ...extra, reference: "fermi" };
      [band, dos] = await Promise.all([api("bands?" + q({ path: bandPath, ...fx })), api("dos?" + q({ path: dosPath, ...fx }))]);
    }
    // on E_F the DOS moves onto the band run's E_F when the two runs' differ (the vacuum scale is one already)
    const s = band.reference === "fermi" && dos.reference === "fermi" ? Number(check.dos_shift) || 0 : 0;
    if (s) dos = { ...dos, energy: dos.energy.map((e) => e + s) };
    return { path: d.opts.pair, check, shift: s, halves: { band, dos } };
  }
  /* A phonon dispersion and a phonon DOS pair the same way (Add Phonon DOS…, Add Phonon Bands…): only with each other
     (the server refuses an electronic half and another supercell), on zero frequency and the plot's one unit. */
  const pairWords = (d) => isPhonon(d)
    ? { band: "Phonon Bands", dos: "Phonon DOS", bandText: "phonon dispersion", dosText: "phonon DOS", fig: "dispersion + DOS" }
    : { band: "Band", dos: "DOS", bandText: "band structure", dosText: "DOS", fig: "band + DOS" };
  /** The toolbar's Add DOS… / Add Band… (or Remove DOS / Remove Band once added), next to Log Y; a phonon plot's
   *  Add Phonon DOS… / Add Phonon Bands…. */
  function pairButtons() {
    if (!doc || !isBandDos(doc) || !doc.data) return [];
    const band = doc.data.band, dos = doc.data.dos, w = pairWords(doc), other = doc.kind === "dos" ? w.band : w.dos;
    if (doc.opts.pair) return [{ key: "unpair", label: `Remove ${other}`, title: `Draw ${doc.title} alone, without ${doc.opts.pair.split("/").pop()}` }];
    if (doc.kind === "band" && band && !dos) return [{ key: "pair", label: `Add ${w.dos}…`, title: `Add the ${w.dosText} of this system from another folder: one ${w.fig} figure` }];
    if (doc.kind === "dos" && dos && !band) return [{ key: "pair", label: `Add ${w.band}…`, title: `Add the ${w.bandText} of this system from another folder: one ${w.fig} figure` }];
    return [];
  }
  /** The other half's candidates: every DOS (or band) file of the served folder, the same material first, then
   * the nearest folders. */
  function pairCandidates(d) {
    const want = d.kind === "dos" ? "band" : "dos", ph = isPhonon(d);
    const mat = (p) => p.split("/").pop().replace(/\.(BAND|DOSS)\.DAT$|\.f25$|_band\.dat$|_doss\.dat$/i, "")
      .replace(/[._]phonon[._](band|dos|doss)$/i, "").replace(/[._](band|doss)$/i, "").toLowerCase();
    const me = mat(d.path), home = dirOf(d.path).split("/");
    const near = (p) => { const s = dirOf(p).split("/"); let k = 0; while (k < s.length && k < home.length && s[k] === home[k]) k++; return k; };
    // a phonon plot lists the phonon runs' files (the listing's domain), an electronic one the others
    return (H.files ? H.files() || [] : []).filter((f) => f.path !== d.path && halfKind(f) === want && (f.domain === "phonon") === ph)
      .map((f) => ({ path: f.path, name: f.name || f.path.split("/").pop(), dir: dirOf(f.path), same: mat(f.path) === me, near: near(f.path) }))
      .sort((a, b) => (b.same - a.same) || (b.near - a.near) || a.path.localeCompare(b.path));
  }
  function pickPair() {
    const d = doc, w = pairWords(d), want = d.kind === "dos" ? w.bandText : w.dosText;
    if (!H.dialog) return;
    const list = pairCandidates(d);
    const body = document.createElement("div");
    body.className = "pv-pair";
    body.innerHTML = `<p class="muted small">The ${want} of the same system from another folder, drawn with ${esc(d.title)} as one ${w.fig} figure
      ${isPhonon(d) ? "on one frequency axis (checked: the same formula and supercell; the cell, else a warning)"
        : "(checked: the same formula; the cell and the Fermi level, else a warning)"}.</p>
      <input type="search" class="pv-pair-q" placeholder="Filter files" aria-label="Filter files">
      <div class="pv-pair-list" role="listbox" aria-label="${esc(want)} files"></div><p class="pv-pair-msg" role="alert"></p>`;
    const dlg = H.dialog({ id: "plot-pair", title: `Add ${d.kind === "dos" ? w.band : w.dos}`, body });
    const box = body.querySelector(".pv-pair-list"), msg = body.querySelector(".pv-pair-msg");
    const show = (f) => {
      const shown = list.filter((x) => !f || x.path.toLowerCase().includes(f.toLowerCase()));
      box.innerHTML = shown.length ? shown.map((x) => `<button type="button" role="option" class="pv-pair-item" data-pair="${esc(x.path)}" title="${esc(x.path)}">` +
        `<span class="pv-pair-name">${esc(x.name)}</span><span class="pv-pair-dir muted">${esc(x.dir || ".")}</span></button>`).join("")
        : `<p class="muted small">No ${want} file${f ? " matches" : " in this folder"}.</p>`;
    };
    show("");
    body.querySelector(".pv-pair-q").addEventListener("input", (e) => show(e.target.value.trim()));
    box.addEventListener("click", async (e) => {
      const b = e.target.closest("[data-pair]");
      if (!b || doc !== d) return;
      msg.textContent = "Checking…";
      try {
        const bandPath = d.kind === "dos" ? b.dataset.pair : d.path, dosPath = d.kind === "dos" ? d.path : b.dataset.pair;
        const chk = await api("bandos-pair?" + q({ band: bandPath, dos: dosPath }));
        if (chk.error) { msg.textContent = chk.error; return; }
        dlg.close();
        setOpt(d, { pair: b.dataset.pair }, `Add ${want} ${b.dataset.pair.split("/").pop()}`);
        for (const w of chk.warnings || []) toast(w, "warning");
      } catch (err) { msg.textContent = err.message; }
    });
  }

  /* ------------------------------------------------------------ figures */
  const PAL = (css = cssVar) => BF.PAL(css);
  // what the energy axis reads: E - E_F, or E - E_vac with a POTC run (the API's reference)
  // the readout's energy: E − E_F, or with a Fermi shift E − (E_F + s) as the axis title (bandfig.js energyTitle)
  // (a textContent: the markup as text, L.plain)
  const eLabel = (d) => isPhonon(d) ? "ω" : L.plain(BF.energyTitle(((d.data && (d.data.band || d.data.dos)) || {}).reference, (d.opts && d.opts.shift) || 0));
  const kLabel = BF.kLabel;

  /* the user's defaults for a plot's style (Settings › Appearance › Plot style, Settings › Plots), by the
     presets.py STYLE_OPTIONS names; a plot's own choices (its options of the same names) go over them */
  function settingsLook() {
    return { style: H.plotPreset ? H.plotPreset("band") : "default", spin_colors: setting("plots.spinColors", "auto"),
      spin_up_color: setting("plots.spinUp", ""), spin_down_color: setting("plots.spinDown", ""),
      band_color: setting("plots.bandColor", ""), band_style: setting("plots.bandStyle", "auto"),
      spin_down_style: setting("plots.spinDownDash", "dash"), dos_colors: setting("plots.dosColors", "auto"),
      dos_spin: setting("plots.spinDos", "auto"), dos_total: setting("plots.dosTotal", true) !== false,
      dos_smooth: Number(setting("plots.dosSmooth", 0)) || 0 };
  }
  /* the figure's page-side inputs (bandfig.js env): the plot's look, its preset's tokens, the pane's width */
  function envOf(o, fermiKey = false) {
    const PP = H.presets ? H.presets() : null;
    const look = PP ? BF.lookOf(o, settingsLook(), PP) : settingsLook();
    const pid = PP ? PP.resolveId(look.style, "band") : "default";
    const css = o && o.style && H.plotVarIn ? (n) => H.plotVarIn(n, pid, surfaceOf(doc)) : tokenOf("band");
    return { css, look, tone: toneOf(doc), paper: paper(), presets: PP,
      elementColors: H.elementColors ? H.elementColors() : {},
      width: el.clientWidth || 900, height: el.clientHeight || 420, marginR: el._fullLayout && el._fullLayout.margin ? el._fullLayout.margin.r : 14, fermiKey };
  }
  // How a DOS is drawn (bandfig.js dosLook): the plot's look, its DOS colors
  const dosLook = (dos, css, o = {}) => BF.dosLook(dos, envOf(o), o);
  const dosRange = BF.dosRange;
  function bandTickLayout(band, range, dos, r = el._fullLayout && el._fullLayout.margin ? el._fullLayout.margin.r : 14) {
    return BF.bandTickLayout(band, range, dos, el.clientWidth || 900, r);
  }
  /* ------------------------------------------------------------ the k-path as edited (kpath_editor.js)
     The computed path comes from the tick labels ("A|L" is a jump); each segment is the data
     between its two ticks. An edited path (opts.kpath) lays those slices out again in its order,
     reversed where it says so, with a gap at each jump; hidden segments are left out. */
  const pathSegments = BF.pathSegments;
  function bandView(d) {
    const band = d.data && d.data.band, edit = d.opts && d.opts.kpath;
    if (!band || !edit) return band;
    const sig = JSON.stringify(edit);
    if (d.view && d.view.sig === sig && d.view.band === band) return d.view.out;
    const out = BF.editedBand(band, edit);
    d.view = { sig, band, out };
    return out;
  }
  /** Where each segment of the editor's list is drawn: {index: [x0, x1]} (hidden ones are not). */
  function segRanges(d) {
    const v = bandView(d);
    if (v && v.segX) return v.segX;
    const out = {};
    if (v) v.ticks.slice(0, -1).forEach((t, n) => { out[n] = [t, v.ticks[n + 1]]; });
    return out;
  }
  /* what an editor control would do, on the plot before it is used: the segments a point's ×
     hides (or the one under the pointer, in the strip or the Brillouin zone view) shaded; the part
     Continuous only keeps, with the rest grayed */
  function plotPreview(p) {
    if (!doc || !el.layout || !el.layout.shapes) return;
    const base = el.layout.shapes.filter((x) => !String(x.name || "").startsWith("kp-"));
    const add = [];
    if (p) {
      const R = segRanges(doc), set = new Set(p.indices);
      const rect = (x0, x1, fill, op) => ({ type: "rect", xref: "x", yref: "y domain", x0, x1, y0: 0, y1: 1, layer: "below", line: { width: 0 },
        fillcolor: fill, opacity: op, name: `kp-${p.kind}` });
      for (const [i, r] of Object.entries(R)) {
        const on = set.has(+i);
        if (p.kind === "keep" ? !on : on) add.push(rect(r[0], r[1], p.kind === "keep" ? cssVar("--text-3") : cssVar(p.kind === "affected" ? "--warning" : "--accent"), p.kind === "hover" ? 0.12 : 0.2));
      }
    }
    window.Plotly.relayout(el, { shapes: [...base, ...add] });
  }
  function bandFigure(d) {
    const fig = BF.bandFigure(bandView(d), d.data.dos, d.opts, envOf(d.opts));
    d.absShown = fig.absShown;
    if (fig.dosMode) d.dosMode = fig.dosMode;
    return { traces: fig.traces, layout: fig.layout, legendBase: fig.legendBase };
  }
  /* The band and DOS legends sit in the top and right margins (bandFigure). Plotly's own margin
     push leaves a legend where the previous size put it when a resize changes its height (it does
     not redraw while replotting), so the margins are set here from the legends as drawn: a legend
     never covers the data or the tick labels at any pane size or UI scale, and the exports, drawn
     from this layout, are the same. */
  let legendBase = null;
  function fitLegends() {
    if (!legendBase || building) return Promise.resolve();
    // the k labels' rows depend on the band panel's width, which the right margin changes
    return PK.fitLegends(el, legendBase, (r) => (doc && doc.data && doc.data.band && el.layout.xaxis
      ? bandTickLayout(doc.data.band, el.layout.xaxis.range, doc.data.dos, r) : {}));
  }
  function irLabel(shown, fallback) {
    const r = [...shown].filter((c) => c.startsWith("reflectance")).length;
    return r === 0 ? fallback || "Absorbance" : r === shown.size ? "Reflectance" : "Absorbance / reflectance";
  }
  function spectrumFigure(d) {
    const o = d.opts, css = tokenOf("spectra"), pal = PAL(css), grid = css("--line");
    const traces = [];
    const list = o.average && d.data.average ? [d.data.average] : d.data.spectra;
    const fx = PK.WAVENUMBER_UNITS[o.unit] || 1, ul = L.unit(PK.WAVENUMBER_UNITS[o.unit] ? o.unit : "cm-1");   // cm<sup>−1</sup> (labels.js)
    d.absShown = false;
    const shown = new Set();
    list.forEach((s, k) => {
      // Raman's components, or a crystal's IR absorbances and reflectances (IRSPEC.DAT columns 3-9)
      const keys = Object.keys(s.columns);
      const comps = o.component === "all" && (d.kind === "raman" || keys.length > 1) ? keys : [s.columns[o.component] ? o.component : keys[0]];
      comps.forEach((c) => shown.add(c));
      comps.forEach((c, j) => {
        let y = s.columns[c] || s.columns[Object.keys(s.columns)[0]];
        if (o.normalize) { const m = Math.max(...y.map(Math.abs)) || 1; y = y.map((v) => v / m); }
        if (o.ylog) { const a = PK.absIfNeeded(y); y = a.ys; if (a.abs) d.absShown = true; }
        traces.push({ x: s.x.map((v) => v * fx), y, mode: "lines", type: "scatter", name: list.length > 1 || (o.average && d.data.average) ? `${s.name} · ${c}` : c,
          line: { color: pal[(k + j) % pal.length], width: o.width }, hovertemplate: `%{x:.2f} ${ul}: %{y:.4g}<extra>${esc(c)}</extra>` });
      });
    });
    const xs = d.data.spectra[0].x.map((v) => v * fx);
    // the legend in a column right of the plot, the peaks' wavenumbers on a top axis (fitLegends sizes both margins)
    const lay = { margin: { l: 64, r: 14, t: 10, b: 44, autoexpand: false }, showlegend: traces.length > 1 || !!(o.average && d.data.average), hovermode: "x unified",
      legend: PK.legendSide(), ...paper(),
      font: { color: css("--text"), size: 12, family: css("--font-sans") },
      modebar: { bgcolor: "rgba(0,0,0,0)", color: css("--text-2"), activecolor: css("--accent") },
      xaxis: { title: { text: L.withUnit(o.unit === "cm-1" || !o.unit ? "Wavenumber" : "Energy", ul) },
        type: o.xlog ? "log" : "linear",
        range: o.xr || (o.xlog ? undefined : [(o.xmin ?? Math.min(...d.data.spectra[0].x)) * fx, (o.xmax ?? Math.max(...d.data.spectra[0].x)) * fx]),
        autorange: o.xlog && !o.xr ? true : undefined, gridcolor: grid, showline: true, mirror: true, linecolor: grid, zeroline: false },
      yaxis: { title: { text: o.normalize ? "Normalized" : d.kind === "ir" ? irLabel(shown, d.data.spectra[0].ylabel) : "Raman activity" },
        type: o.ylog ? "log" : "linear", range: o.yr || undefined,
        gridcolor: grid, showline: true, mirror: true, linecolor: grid, zeroline: false } };
    if (d.data.peaks && d.data.peaks.length) {
      // the strongest peaks' positions as labels on a top axis, above the data; a weaker peak too close to
      // a stronger one's label (13 px at the plot's width) is left unlabeled
      const xr = lay.xaxis.range || [Math.min(...xs), Math.max(...xs)], span = Math.abs(xr[1] - xr[0]) || 1;
      const gap = span * 13 / Math.max(200, (el.clientWidth || 900) - 64 - 14 - 110);
      const strongest = [];
      for (const p of [...d.data.peaks].sort((a, b) => b.intens - a.intens)) {
        if (strongest.length >= 12) break;
        if (strongest.every((q) => Math.abs(q.freq - p.freq) * fx >= gap)) strongest.push(p);
      }
      lay.xaxis2 = { overlaying: "x", matches: "x", side: "top", type: lay.xaxis.type, showgrid: false, zeroline: false, showline: false, ticks: "",
        tickmode: "array", tickvals: strongest.map((p) => p.freq * fx), ticktext: strongest.map((p) => `${(p.freq * fx).toFixed(o.unit === "cm-1" || !o.unit ? 0 : 2)}`),
        tickangle: -90, tickfont: { size: 9, color: css("--text-3") } };
      traces.push({ x: [null], y: [null], xaxis: "x2", mode: "markers", type: "scatter", meta: "peak-axis", showlegend: false, hoverinfo: "skip" });
      // one marker per active mode, on the curve: hover names the mode, a click opens its animation (plan E4)
      if (traces.length && d.data.out) {
        const t0 = traces[0], at = (f) => {
          let lo = 0, hi = t0.x.length - 1;
          while (hi - lo > 1) { const m = (lo + hi) >> 1; if (t0.x[m] < f) lo = m; else hi = m; }
          const w = (f - t0.x[lo]) / ((t0.x[hi] - t0.x[lo]) || 1);
          return t0.y[lo] + w * (t0.y[hi] - t0.y[lo]);
        };
        const unit = d.kind === "ir" ? ` ${L.unit("km/mol")}` : "";
        traces.push({ x: d.data.peaks.map((p) => p.freq * fx), y: d.data.peaks.map((p) => at(p.freq * fx)), mode: "markers", type: "scatter",
          name: "modes", meta: "peaks", showlegend: false, customdata: d.data.peaks.map((p) => p.mode),
          marker: { size: 7, color: css("--accent"), line: { width: 1, color: css("--surface") } },
          text: d.data.peaks.map((p) => `Mode ${p.mode} · ${p.freq.toFixed(1)} ${L.unit("cm-1")} · ${p.irrep || ""} · ${d.kind === "ir" ? "IR" : "Raman"} ${p.intens.toFixed(1)}${unit}${p.degeneracy > 1 ? ` (degenerate ×${p.degeneracy})` : ""}`),
          hovertemplate: "%{text}<extra>click: animate</extra>" });
        lay.hovermode = "closest";
      }
    }
    if (d.source === "modes") lay.yaxis.title.text = o.normalize ? "Normalized" : d.data.spectra[0].ylabel;
    return { traces, layout: lay, legendBase: { t: 10, r: 14 } };
  }

  /* a 2D map: contours in the plane of A, B, C (Å), the corners labeled */
  function mapFigure(d) {
    const m = d.data.map.maps[Math.min(d.opts.which || 0, d.data.map.maps.length - 1)], o = d.opts, css = tokenOf("maps");
    const positive = m.min > 0, log = o.log && positive;
    const z = m.values.map((col) => col.map((v) => (log ? Math.log10(v) : v)));
    const signedZ = !log && m.min < 0 && m.max > 0;
    const zr = o.crange === "custom" && Number.isFinite(o.zmin) && Number.isFinite(o.zmax) && o.zmax > o.zmin ? [o.zmin, o.zmax]
      : robustRange(z, { signed: signedZ, full: o.crange === "full" });
    const dist = (p, r) => Math.hypot(p[0] - r[0], p[1] - r[1], p[2] - r[2]);
    const ab = dist(m.A, m.B), bc = dist(m.B, m.C), sin = Math.sqrt(Math.max(0, 1 - m.cosxy * m.cosxy));
    // each corner's label sits inside the plane, on the side of its corner away from the edges and tick labels
    const cx = (ab + ab + bc * m.cosxy) / 3, cy = bc * sin / 3;
    const corner = (t, x, y) => ({ x, y, text: t, showarrow: false, font: { size: 13, color: css("--text") },
      xanchor: x < cx - 1e-6 ? "left" : x > cx + 1e-6 ? "right" : "center", yanchor: y < cy ? "bottom" : "top",
      bgcolor: css("--surface"), opacity: 0.85 });
    const signed = m.min < 0 && m.max > 0;
    // the Maps & volumes preset's colormap (signed: + blue, − red; VESTA-like: + yellow, − cyan) and contour
    // style (filled, lines or both: this map's own choice over the preset's)
    const pp = H.presets ? H.presets() : null, pid = presetOf("maps");
    const sc = pp ? pp.mapScale(pid, signed) : { colorscale: signed ? "RdBu" : "Viridis", reversescale: signed, contours: "both" };
    // a POTM/CLAS potential in physics colors (Settings › Viewport › Potential colors): red = +; chemistry keeps + blue, − red
    if (signed && m.potential && setting("volume.mepColors", "chemistry") === "physics") sc.reversescale = !sc.reversescale;
    const cstyle = o.contour_style || sc.contours, darkLines = pid === "vesta" || pid === "grayscale" || pid === "publication";
    const nc = Math.max(2, o.contours || 20), step = (zr[1] - zr[0]) / nc || 1;
    const trace = { type: "contour", x: m.x, y: m.y, z, colorscale: sc.colorscale, reversescale: sc.reversescale,
      zauto: false, zmin: zr[0], zmax: zr[1], autocontour: false,
      contours: { coloring: cstyle === "lines" ? "lines" : "heatmap", showlines: cstyle !== "filled", start: zr[0], end: zr[1], size: step },
      line: { width: cstyle === "lines" ? 1.2 : 0.6, ...(darkLines && cstyle === "both" ? { color: "rgba(0,0,0,0.55)" } : {}) },
      // the colorbar names the quantity and its unit, as mace plotting's cube slices (plot markup: labels.js)
      colorbar: { title: { text: `${log ? `${L.SYMBOLS.log10} ` : ""}${L.html(m.label)}${m.unit ? `<br>(${m.unit})` : ""}` }, thickness: 12 },
      hovertemplate: `%{x:.3f}, %{y:.3f} Å: ${log ? "10<sup>%{z:.3f}</sup>" : "%{z:.4g}"}${m.unit ? ` ${m.unit}` : ""}<extra>${L.html(m.label)}</extra>` };
    return { traces: [trace], layout: { margin: { l: 56, r: 14, t: 10, b: 44 }, ...paper(),
      font: { color: css("--text"), size: 12, family: css("--font-sans") },
      xaxis: { title: { text: "along AB (Å)" }, scaleanchor: "y", constrain: "domain", showgrid: false },
      yaxis: { title: { text: "toward C (Å)" }, showgrid: false },
      annotations: [corner("A", 0, 0), corner("B", ab, 0), corner("C", ab + bc * m.cosxy, bc * sin)] } };
  }

  /* Compare Convergence: each run's SCF energy per atom at each geometry, above the lowest final energy of the set */
  const HARTREE_MEV = 27211.386245988;
  function convergenceFigure(d) {
    const runs = d.data.convergence.runs, o = d.opts, css = tokenOf("scf"), pal = PAL(css), grid = css("--line");
    const finals = runs.map((r) => { const e = r.energies.filter((x) => x !== null); return e.length ? e[e.length - 1] / r.n_atoms : null; })
      .filter((x) => x !== null);
    const ref = finals.length ? Math.min(...finals) : 0;
    const traces = runs.map((r, k) => {
      const pts = r.energies.map((e, i) => [i + 1, e === null ? null : (e / r.n_atoms - ref) * HARTREE_MEV]).filter(([, y]) => y !== null);
      // "* OPT END - CONVERGED * E(AU): -7.62E+01 POINTS 3 *" -> "OPT END - CONVERGED, 3 points" (the energy is plotted)
      const stop = r.status ? r.status.replace(/\*/g, "").replace(/E\(AU\):\s*\S+/, "").replace(/POINTS\s+(\d+)/, ", $1 points")
        .replace(/\s+,/, ",").replace(/\s+/g, " ").trim() : "no OPT END line";
      return { x: pts.map((p) => p[0]), y: pts.map((p) => (o.log ? Math.max(p[1], 1e-6) : p[1])), mode: "lines+markers", type: "scatter",
        name: `${r.name} · ${stop}`, line: { color: pal[k % pal.length], width: o.width }, marker: { size: 5 },
        hovertemplate: `%{y:.3f} meV/atom at geometry %{x}<extra>${esc(r.name)}</extra>` };
    });
    return { legendBase: { t: 10, r: 14 }, traces, layout: { margin: { l: 70, r: 14, t: 10, b: 50, autoexpand: false }, showlegend: true, legend: PK.legendTop(),
      hovermode: "closest", ...paper(), font: { color: css("--text"), size: 12, family: css("--font-sans") },
      xaxis: { title: { text: "Geometry (initial, optimization points, final)" }, gridcolor: grid, showline: true, mirror: true, linecolor: grid, dtick: 1 },
      yaxis: { title: { text: "ΔE per atom (meV)" }, type: o.log ? "log" : "linear", gridcolor: grid, showline: true, mirror: true, linecolor: grid } } };
  }

  /* BOLTZTRA data: any quantity the run's files give (the file's own, its siblings', or derived from them),
   * read again when the quantity, κ_L or τ changed. A refusal (a κ_L or τ MACE refuses, files on different
   * grids) is shown in the options with what can be drawn: ZTₑ for a refused κ_L, else the file's own quantity */
  const trQuantity = (o) => (o && (o.quantity || (o.view === "zt" ? "zt" : null))) || null;
  const trKey = (o) => JSON.stringify([trQuantity(o), trQuantity(o) === "zt" ? String(o.kl || "").trim() : "", o.tau ?? null]);
  const staleTr = (d) => d && d.kind === "transport" && d.data && d.data.transport && d.data.trKey !== trKey(d.opts);
  async function fetchTransport(d) {
    const o = d.opts || {}, key = trKey(o), qn = trQuantity(o);
    const get = (quantity, kl, tau) => api("transport?" + q({ path: d.path, quantity: quantity || "", kappa_l: kl || "", tau_fs: tau ?? "" }));
    let t = null, err = null;
    try { t = await get(qn, qn === "zt" ? String(o.kl || "").trim() : "", o.tau); }
    catch (e) {
      err = e.message;
      // ZTₑ alone for a κ_L or τ refused; else the file's own quantity, else the first of the run that draws
      const tries = (qn === "zt" ? [["zt"]] : []).concat([[null], ["SEEBECK"], ["SIGMA"], ["KAPPA"], ["SIGMAS"]]);
      for (const [qq] of tries) { try { t = await get(qq); break; } catch (e2) { /* the next */ } }
      if (!t) throw e;
    }
    d.data = { ...(d.data || {}), transport: t, trError: err, trKey: key };
  }
  const fmtZt = (m) => (m ? `${m.value.toPrecision(4)} (${m.temperature_k} K, μ ${m.mu_ev.toFixed(3)} eV)` : "");
  // the menu's quantities: the run's files, then the ones derived from them (mace.utils.transport_plot.QUANTITIES)
  const TQ_RAW = ["SEEBECK", "SIGMA", "SIGMAS", "KAPPA", "TDF"], TQ_DERIVED = ["pf", "kappa_e", "zt", "lorenz", "sigma_tau", "pf_tau"];
  // the Quantity buttons (their markup drawn by L.html, as the plot's labels: labels.js)
  const TQ_SHORT = { SEEBECK: "S", SIGMA: "σ", SIGMAS: "σS", KAPPA: L.SYMBOLS.kappa_0, TDF: "TDF", pf: "PF", kappa_e: L.SYMBOLS.kappa_e, zt: "ZT", lorenz: "L", sigma_tau: "σ/τ", pf_tau: "PF/τ" };
  const TQ_TITLE = { SEEBECK: "Seebeck coefficient (SEEBECK.dat)", SIGMA: "Electrical conductivity (SIGMA.dat)", SIGMAS: "σS as CRYSTAL writes it, −σ·S (SIGMAS.dat)",
    KAPPA: "Zero-field thermal coefficient κ₀ (KAPPA.dat)", TDF: "Transport distribution function against energy (TDF.dat)", pf: "Power factor S²σ",
    kappa_e: "Electronic thermal conductivity κₑ = κ₀ − TσS²", zt: "Figure of merit ZT = S²σT/(κₑ + κ_L); without κ_L, ZTₑ (κ_L = 0), an upper bound",
    lorenz: "Lorenz number L = κₑ/(σT), with Sommerfeld's L₀", sigma_tau: "σ/τ at the run's relaxation time", pf_tau: "PF/τ = S²σ/τ at the run's relaxation time" };
  const compLabel = (t, c) => (c === "avg" ? t.avg_label : c);
  /* the curves of mace.utils.transport_plot.traces (mace plotting --transport draws the same): one temperature or
     all, one component or all three, against μ or the carrier concentration (p- and n-type apart), ZT's ZTₑ bound
     with a κ_L, the Lorenz number's L₀ line */
  const ZT_E_KEY = `${L.SYMBOLS.ZT_e} (${L.SYMBOLS.kappa_L} = 0)`;     // transport_plot.ZT_E_KEY
  function transportTraces(t, o) {
    const all = String(o.T) === "all" && t.blocks[0].T !== null;
    const blocks = t.blocks[0].T === null ? t.blocks.slice(0, 1) : all ? t.blocks : [t.blocks.find((b) => b.T === o.T) || t.blocks[0]];
    let comps = o.component === "all" ? t.components.slice() : [o.component === "avg" || t.components.includes(o.component) ? o.component || "avg" : "avg"];
    if (all && comps.length > 1) comps = ["avg"];
    const carriers = o.x === "carriers" && t.x !== "energy", ztTotal = t.quantity === "zt" && !t.upper_bound;
    const out = [];
    let k = 0;
    for (const b of blocks) {
      for (const c of comps) {
        const base = blocks.length > 1 ? `${b.T} K` : comps.length > 1 ? compLabel(t, c) : null;
        const series = [[b[c], "data", ztTotal ? `ZT (${t.kappa_l_label}, τ = ${t.tau_fs} fs)` : null]];
        if (ztTotal && b.bound && b.bound[c]) series.push([b.bound[c], "bound", ZT_E_KEY]);
        for (const [y, role, what] of series) {
          if (ztTotal && role === "data" && b.refused) continue;
          const name = what ? (base ? `${base} · ${what}` : what) : base || compLabel(t, c);
          const dash = role === "bound" ? "dash" : "solid", meta = t.quantity === "zt" ? (role === "bound" || !ztTotal ? "zt_e" : "zt_total") : role;
          if (carriers) {
            const vol = t.dim === 3 ? b.volume_cm3 : null;
            for (const [sign, kind, d2] of [[1, "p-type", dash], [-1, "n-type", role === "bound" ? "dot" : "dash"]]) {
              const xs = [], ys = [];
              b.carriers.forEach((n, i) => { if (n * sign > 0) { xs.push(vol ? Math.abs(n) / vol : Math.abs(n)); ys.push(y[i]); } });
              if (xs.length) out.push({ name: `${name}, ${kind}`, x: xs, y: ys, color: k, dash: d2, role, meta });
            }
          } else out.push({ name, x: b.mu, y, color: k, dash, role, meta });
        }
        k += 1;
      }
    }
    if (t.reference && out.length) {
      const xs = out.flatMap((c) => c.x);
      out.push({ name: t.reference.label, x: [Math.min(...xs), Math.max(...xs)], y: [t.reference.value, t.reference.value], color: null, dash: "dot", role: "reference", meta: "reference" });
    }
    return out;
  }
  const trNote = (t) => `${L.html(t.label)} from BOLTZTRA (${t.blocks[0].T === null ? "against energy" : `${t.blocks.length} temperature${t.blocks.length === 1 ? "" : "s"}`}), read with the database extractor's own reader`;
  /* ZT with κ_L: the y axis spans the total, so the dashed ZTₑ bound (no lattice term: it runs up where κₑ is
     tiny, in the gap) does not flatten it (transport_plot.y_range) */
  const trYRange = (t, curves) => {
    if (t.quantity !== "zt" || t.upper_bound) return null;
    const ys = curves.filter((c) => c.role === "data").flatMap((c) => c.y).filter((v) => v !== null && Number.isFinite(v));
    return ys.length && Math.max(...ys) > 0 ? [0, 1.08 * Math.max(...ys)] : null;
  };
  // the axis titles: transport_plot.x_title, y_title (plot markup: labels.js)
  const trXTitle = (t, o) => (t.x === "energy" ? L.withUnit("Energy", "eV") : o.x === "carriers" ? (t.dim === 3 ? L.withUnit("Carrier concentration |n|", "cm-3") : "Carriers per cell |N|")
    : L.withUnit("Chemical potential μ", "eV"));
  const trYTitle = (t) => (t.quantity === "zt" ? (t.upper_bound ? `${L.SYMBOLS.ZT_e} (${L.SYMBOLS.kappa_L} = 0, an upper bound)` : "Figure of merit ZT") : L.withUnit(t.label, t.unit));
  function transportFigure(d) {
    const t = d.data.transport, o = d.opts, css = tokenOf("band"), pal = PAL(css), grid = css("--line");
    const unit = t.unit ? ` ${L.html(t.unit)}` : "";
    const curves = transportTraces(t, o), yr = trYRange(t, curves);
    const traces = curves.map((c) => ({ x: c.x, y: c.y, mode: "lines", type: "scatter", name: c.name, meta: c.meta, connectgaps: false,
      line: c.role === "reference" ? { color: css("--text-3") || grid, width: 1, dash: "dot" } : { color: pal[c.color % pal.length], width: o.width, dash: c.dash },
      hovertemplate: c.role === "reference" ? `${L.html(c.name)}<extra></extra>` : `%{x:.4g}: %{y:.4g}${unit}<extra>${L.html(c.name)}</extra>` }));
    const carriers = o.x === "carriers" && t.x !== "energy";
    const lay = { margin: { l: 70, r: 14, t: 10, b: 44, autoexpand: false }, showlegend: traces.length > 1, hovermode: carriers ? "closest" : "x unified", legend: PK.legendTop(),
      ...paper(), font: { color: css("--text"), size: 12, family: css("--font-sans") },
      xaxis: { title: { text: trXTitle(t, o) }, type: carriers ? "log" : "linear", exponentformat: "power", gridcolor: grid, showline: true, mirror: true, linecolor: grid, zeroline: !carriers, zerolinecolor: grid },
      yaxis: { title: { text: trYTitle(t) }, ...(yr ? { range: yr } : t.quantity === "zt" ? { rangemode: "tozero" } : {}), exponentformat: "power", gridcolor: grid, showline: true, mirror: true, linecolor: grid, zeroline: true, zerolinecolor: grid } };
    return { traces, layout: lay, legendBase: { t: 10, r: 14 } };
  }

  let again = false;
  /* a broadened spectrum whose width or shape changed (undo, redo, reset): read again */
  const staleModes = (d) => d.source === "modes" && d.data && d.data.spectra && (
    (d.opts.fwhm ?? d.data.spectra[0].default_fwhm) !== d.data.spectra[0].fwhm || (d.opts.shape || "lorentzian") !== d.data.spectra[0].shape ||
    (d.kind === "raman" && (d.opts.eta ?? 1) !== d.data.spectra[0].eta));
  async function redraw() {
    if (!doc || !doc.data) return;
    if (staleModes(doc)) { const d = doc; d.data = null; try { await fetchData(d); } catch (e) { toast(e.message, "error"); return; } if (doc !== d) return; note(d); }
    if (staleData(doc)) { const d = doc; d.data = null; try { await fetchData(d); } catch (e) { toast(e.message, "error"); return; } if (doc !== d) return; note(d); renderOptions(); if (kit) kit.refresh(); }
    if (staleTr(doc)) {
      const d = doc;
      try { await fetchTransport(d); } catch (e) { toast(e.message, "error"); return; }
      if (doc !== d) return;
      // the subtitle names the quantity drawn (its one note: the label's markup kept, L.html)
      const line = $(".pv-note-line");
      if (line) line.innerHTML = `${trNote(d.data.transport)}<span class="pv-beyond"></span>`;
      renderOptions();
    }
    if (building) { again = true; return; }        // one figure at a time: the latest options win
    building = true;
    el.dataset.drawn = "";
    try {
      await loadPlotly();
      if (!window.Plotly) throw new Error("plotly is not installed in this Python; pip install plotly");
      const fig = doc.kind === "map" ? mapFigure(doc) : doc.kind === "convergence" ? convergenceFigure(doc) : doc.kind === "transport" ? transportFigure(doc) : doc.kind === "ir" || doc.kind === "raman" ? spectrumFigure(doc) : bandFigure(doc);
      // another document, or other subplots (a band plot after a band+DOS one): a new figure, not an update;
      // plotly's react cannot add a second x axis to an existing figure
      const sig = `${doc.path}${doc.kind === "transport" ? `|${doc.data.transport.quantity}|${doc.opts.x}` : ""}|${Object.keys(fig.layout).filter((k) => /^[xy]axis/.test(k)).sort().join(",")}`;
      const config = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"], edits: { shapePosition: true } };
      // Settings › Appearance › Plot style (app.js styleFigure): the screen and the PNG/SVG exports alike
      if (H.styleFigure) H.styleFigure(fig.traces, fig.layout, familyOf(doc), (doc.opts && doc.opts.style) || null, surfaceOf(doc));
      const bw = $(".pv-bword");
      if (bw && isBandDos(doc)) bw.textContent = BF.dashWord(envOf(doc.opts).look.spin_down_style || "dash");
      // the styled top margin is the base fitLegends adds the legends to (never the fitted one: that grows on every redraw)
      if (fig.legendBase && fig.layout.margin) fig.legendBase.t = Math.max(fig.legendBase.t, fig.layout.margin.t);
      // the same figure again (an option changed): keep the margins fitted to its legends (fitLegends)
      if (fig.legendBase && el.__sig === sig && el.layout && el.layout.margin) Object.assign(fig.layout.margin, { t: el.layout.margin.t, r: el.layout.margin.r });
      // no transitions: zoom and pan are instant (and so under reduced motion too)
      if (el.__sig !== sig) {
        if (el.__sig) window.Plotly.purge(el);
        el.__maceBound = false;
        await window.Plotly.newPlot(el, fig.traces, fig.layout, config);
        el.__sig = sig;
      } else {
        await window.Plotly.react(el, fig.traces, fig.layout, config);
      }
      building = false;
      fit();          // react keeps the size the figure had; the pane may have changed since
      legendBase = fig.legendBase || null;
      await fitLegends();
      bind();
      chips();
      el.dataset.drawn = doc.path;
      beyondNote();
      performance.mark("plot:drawn");
    } catch (e) { $(".pv-note").classList.remove("one"); $(".pv-note").textContent = e.message; } finally {
      building = false;
      if (again) { again = false; redraw(); }
    }
  }

  function chips() {
    const o = doc.opts, c = $(".pv-chips");
    const parts = [];
    if (doc.kind !== "ir" && doc.kind !== "raman" && o.shift) parts.push(`<span class="pv-chip" title="Fermi shift (drag the dotted line; Plot options)">E<sub>F</sub> ${o.shift > 0 ? "+" : "−"} ${Math.abs(o.shift).toFixed(2)} eV</span>`);
    if (doc.absShown) parts.push(`<span class="pv-chip" title="A log axis cannot show zero or negative values">|value| shown</span>`);
    // a band + DOS pair from two folders: the other file, and why it is not drawn or how it was aligned
    const pr = doc.data && doc.data.pair;
    if (pr && pr.error) parts.push(`<span class="pv-chip warn" title="${esc(pr.error)}">${esc(pr.path.split("/").pop())} not drawn</span>`);
    else if (pr) parts.push(`<span class="pv-chip" title="${esc([pr.path, ...((pr.check && pr.check.warnings) || []), ...((pr.check && pr.check.notes) || [])].join("\n"))}">with ${esc(relPath(dirOf(doc.path), pr.path))}${pr.shift ? ` · DOS ${pr.shift > 0 ? "+" : "−"}${Math.abs(pr.shift).toFixed(2)} eV` : ""}</span>`);
    c.innerHTML = parts.join("");
  }

  /* hover readout, the Fermi-line drag, the zoom that keeps the DOS axis fitted */
  function bind() {
    if (el.__maceBound) return;
    el.__maceBound = true;
    // the DOS spin key only explains the line styles: a click on it hides nothing
    // a projection's swatch: a click hides that projection (Plot options › Projections shows it again)
    el.on("plotly_legendclick", (ev) => {
      const t = ev && ev.data && ev.data[ev.curveNumber];
      if (t && t.meta === "key-proj" && Array.isArray(t.customdata) && doc && doc.data && doc.data.dos) {
        const lab = t.customdata[0], look = dosLook(doc.data.dos, tokenOf("band"), doc.opts);
        if (look.byLabel[lab] && !look.byLabel[lab].total) setOpt(doc, { hidden: [...new Set([...look.hidden, lab])] }, `Hide ${lab}`);
        return false;
      }
      return PK.keyClick(ev);
    });
    el.on("plotly_legenddoubleclick", PK.keyClick);
    // a resize (the window, a splitter, the UI scale) redraws the legends: fit the margins to them again
    el.on("plotly_afterplot", () => { if (legendBase && !el.__fitting && !building) requestAnimationFrame(() => fitLegends()); });
    el.on("plotly_hover", (ev) => {
      if (!doc || !doc.data || !ev.points || !ev.points.length) return;
      const p = ev.points[0], t = p.data;
      if (doc.kind === "ir" || doc.kind === "raman" || doc.kind === "transport" || doc.kind === "convergence" || doc.kind === "map") return;
      if (t.meta === "dos") {
        const e = p.y;
        const vals = el.data.filter((x) => x.meta === "dos").map((x) => {
          let best = 0, bd = Infinity;
          for (let i = 0; i < x.y.length; i++) { const dd = Math.abs(x.y[i] - e); if (dd < bd) { bd = dd; best = i; } }
          return `${x.name}: ${Number(x.x[best]).toFixed(3)}`;
        });
        $(".pv-readout").textContent = `${eLabel(doc)} = ${Number(e).toFixed(3)} ${PK.UNIT_LABEL[doc.opts.unit] || "eV"} · ${vals.join(" · ")}`;
        return;
      }
      const band = bandView(doc);
      const x = p.x, e = p.y, k = t.customdata ? t.customdata[p.pointIndex ?? p.pointNumber] : null;
      let kl = "";
      const at = band.ticks.findIndex((tk) => Math.abs(tk - x) < 1e-6);
      if (at >= 0) kl = kLabel(band.labels[at]);
      else {
        const i = band.ticks.findIndex((tk) => tk > x);
        if (i > 0) kl = `${kLabel(band.labels[i - 1])}→${kLabel(band.labels[i])} ${Math.round(((x - band.ticks[i - 1]) / (band.ticks[i] - band.ticks[i - 1])) * 100)}%`;
      }
      $(".pv-readout").textContent = `k: ${kl || "—"} · path ${Number(x).toFixed(4)} · ${eLabel(doc)} = ${Number(e).toFixed(3)} ${PK.UNIT_LABEL[doc.opts.unit] || "eV"}` +
        (k !== null && k >= 0 ? ` · band ${k + 1}${band.spin_polarized ? (t.meta === "down" ? " (β)" : " (α)") : ""}` : "");
    });
    el.on("plotly_unhover", () => { $(".pv-readout").textContent = ""; });
    // a click on a spectrum opens the mode of the nearest active peak (within half the line width)
    el.on("plotly_click", (ev) => {
      if (!doc || !(doc.kind === "ir" || doc.kind === "raman") || !doc.data || !doc.data.out || !ev.points || !ev.points.length) return;
      const pt = ev.points[0];
      // The marker under the click itself, from the click's own position: plotly throttles its
      // hover to one per 50 ms and a click re-uses the last one, so a click right after the pointer
      // moved came with a point it passed on the way (the strong peak's neighbor 17 px below it).
      let mode = null;
      const pk = (el.data || []).find((t) => t.meta === "peaks"), fl = el._fullLayout, e = ev.event;
      if (pk && fl && fl.xaxis && fl.yaxis && e && typeof e.clientX === "number") {
        // the click in the plot's own px: its screen offset is the UI scale's zoom x the plot's px
        const [px, py] = window.maceZoom.point(el, e.clientX, e.clientY), mx = px - fl._size.l, my = py - fl._size.t;
        let bi = -1, bd = 10;                      // px: a marker within reach of the pointer
        pk.x.forEach((x, i) => {
          const dd = Math.hypot(fl.xaxis.c2p(x) - mx, fl.yaxis.c2p(pk.y[i]) - my);
          if (dd <= bd) { bd = dd; bi = i; }
        });
        if (bi >= 0) mode = pk.customdata[bi];
      }
      if ((mode === null || mode === undefined) && pt.data && pt.data.meta === "peaks") mode = pt.customdata;
      if (mode === null || mode === undefined) {
        const half = Math.max(4, ((doc.data.spectra[0] && doc.data.spectra[0].fwhm) || 8) / 2);
        let best = null, bd = Infinity;
        const cm = pt.x / (PK.WAVENUMBER_UNITS[doc.opts.unit] || 1);        // the click in cm⁻¹
        for (const p of doc.data.peaks) { const dd = Math.abs(p.freq - cm); if (dd < bd) { bd = dd; best = p; } }
        if (!best || bd > half) return;
        mode = best.mode;
      }
      if (H.openModes) H.openModes(doc.data.out, { mode: Number(mode) });
    });
    el.on("plotly_relayout", (ev) => {
      if (!doc || !ev || !doc.data) return;        // a relayout of the previous figure while the next document loads
      const sy = Object.keys(ev).find((k) => /^shapes\[0\]\.y0$/.test(k));
      if (sy !== undefined && (doc.data.band || doc.data.dos)) {
        // the Fermi line was dragged (the band panel's, or a DOS-only plot's): the shift is where it went,
        // relative to the reader's E_F. The line is drawn at E_F (0, or the vacuum shift) in the plotted
        // unit: the energy now under it becomes the reference, s' = s + y - E_F (in eV)
        const ef = (doc.data.band || doc.data.dos).fermi || 0;
        const ns = Math.round(((doc.opts.shift || 0) + ev[sy] / (PK.ENERGY_UNITS[doc.opts.unit] || 1) - ef) * 1000) / 1000 || 0;
        setOpt(doc, { shift: ns }, `Fermi shift ${ns.toFixed(2)} eV`, "drag");
        return;
      }
      // a zoom along k: lay the tick labels out again for the new range (the rows depend on it)
      if (doc.data.band && el.layout.xaxis && Object.keys(ev).some((k) => k === "xaxis.autorange" || k.startsWith("xaxis.range"))) {
        const bv = bandView(doc), r = el.layout.xaxis.range, tk = bv.ticks;
        window.Plotly.relayout(el, bandTickLayout(bv, r || [tk[0], tk[tk.length - 1]], doc.data.dos));
      }
      if (doc.data.dos && Object.keys(ev).some((k) => k.startsWith("yaxis."))) {
        if (doc.opts.ylog || !el.layout.yaxis.range) return;
        // the DOS is drawn at E - s: the states on screen are those at the visible energy + s
        const fu = PK.ENERGY_UNITS[doc.opts.unit] || 1, s = doc.opts.shift || 0;
        const [lo, hi] = el.layout.yaxis.range.map((v) => s + v / fu);
        const ax = el.layout.xaxis2 ? "xaxis2" : "xaxis";
        window.Plotly.relayout(el, { [`${ax}.range`]: dosRange(doc.data.dos, Math.min(lo, hi), Math.max(lo, hi), doc.opts, dosLook(doc.data.dos, tokenOf("band"), doc.opts)) });
      }
    });
  }
  el.addEventListener("keydown", (e) => {
    if (!doc || !(e.ctrlKey || e.metaKey) || e.key.toLowerCase() !== "z") return;
    e.preventDefault(); e.stopPropagation();
    const l = e.shiftKey ? redo(doc) : undo(doc);
    if (l) toast(`${e.shiftKey ? "Redo" : "Undo"}: ${l}`, "info");
  });

  /* the k-path editor in Plot options; presets are kept in this browser for any plot with the same segments */
  const PRESETS = "mace.gui.kpath.presets";
  const readPresets = () => { try { return JSON.parse(localStorage.getItem(PRESETS) || "{}") || {}; } catch (e) { return {}; } };
  const writePresets = (p) => { try { localStorage.setItem(PRESETS, JSON.stringify(p)); } catch (e) { /* private */ } };
  const fits = (preset, computed) => K.visible(preset).length > 0 && preset.every((s) => computed.some((c) =>
    (K.canonical(c.from) === K.canonical(s.from) && K.canonical(c.to) === K.canonical(s.to)) || (K.canonical(c.from) === K.canonical(s.to) && K.canonical(c.to) === K.canonical(s.from))));
  const bzOf = (d) => {
    const band = d.data && d.data.band;
    // another band file (Add Band… picked another one, or removed it): its own zone and path facts
    if (d.bz && (!band || d.bzPath !== band.path)) { d.bz.destroy(); d.bz = null; d.kinfo = null; }
    if (!d.bz && band && band.path) {
      d.bz = createBZ({ api, path: band.path, onHover: (i) => KE.hoverFromBZ("plot", i), onSelect: (i) => KE.selectFromBZ("plot", i) });
      d.bzPath = band.path;
    }
    return d.bz || null;
  };
  function kpathEditor(kh) {
    const d = doc, band = d.data.band, computed = pathSegments(band);
    // the zone view sits in the Inspector's Brillouin zone section (every tier); the strip drives it
    KE.render(kh, { id: "plot", mode: "display", computed, segs: d.opts.kpath, icon, bz: bzOf(d), bzSlot: false, onPreview: plotPreview,
      // a zoom along k belongs to the old path: the edited one starts unzoomed
      onChange: (segs, label) => { if (doc === d) setOpt(d, { kpath: segs, xr: null }, label); },
      presets: {
        list: () => Object.entries(readPresets()).map(([name, segs]) => ({ name, ok: fits(segs, computed) })),
        apply: (name) => { const p = readPresets()[name]; if (p && fits(p, computed) && doc === d) setOpt(d, { kpath: K.samePath(p, computed) ? null : p, xr: null }, `k-path preset ${name}`); },
        save: (name) => { const p = readPresets(); p[name] = d.opts.kpath || computed; writePresets(p); toast(`Saved the k-path preset ${name}`, "info"); renderOptions(); },
        remove: (name) => { const p = readPresets(); delete p[name]; writePresets(p); renderOptions(); },
      } });
  }
  /* ------------------------------------------------------------ Inspector › Brillouin zone (1.1.6)
     A band plot (a BAND.DAT, a band fort.25, a phonon run's fort.25) shows the zone of the run's cell (its .out:
     a slab's 2D, a polymer's 1D zone) with the path it was computed along, its high-symmetry points labeled,
     and the path's facts: the segments with their k, the ISS (shrinking factor) and the k points, read from the
     run's .d3, else the fort.25's or the BAND.DAT's own records (GET api/kpath); a phonon run's ISS from the
     FREQCALC BANDS of its .d12. The zone view is the k-path editor's (bz_view.js); an edited path is drawn. */
  const frac = (x) => { const v = Math.round(x * 1e4) / 1e4; return String(v === 0 ? 0 : v); };
  const kText = (k) => (k ? k.map(frac).join(" ") : "");
  function bzSegs(d) {
    const band = d.data.band, info = d.kinfo && d.kinfo.found ? d.kinfo : null, named = pathSegments(band);
    if (d.opts.kpath) return d.opts.kpath;
    const withK = info && info.segments.every((s) => s.k_from && s.k_to) ? info.segments : null;
    if (withK && named && named.length === withK.length) return withK.map((s, i) => ({ ...named[i], k_from: s.k_from, k_to: s.k_to }));
    return withK || named || [];
  }
  function bzSection(slot, facts) {
    const d = doc, band = d.data.band, bz = bzOf(d);
    if (bz && slot) { bz.attach(slot); bz.update(bzSegs(d)); }
    if (!d.kinfo && band.path) {
      d.kinfo = { loading: true };
      api("kpath?" + q({ path: band.path })).then((r) => { d.kinfo = r; }, (e) => { d.kinfo = { error: e.message }; })
        .then(() => { if (doc === d) { const f = optsHost.querySelector(".po-kinfo"); if (f) kFacts(f); if (d.bz && !d.opts.kpath) d.bz.update(bzSegs(d)); } });
    }
    if (facts) kFacts(facts);
  }
  function kFacts(el) {
    const d = doc, band = d.data.band, info = d.kinfo || {};
    const segs = bzSegs(d), named = pathSegments(band) || [];
    const lab = (l) => (l ? K.display(l) : "?");
    // the path as the plot's ticks read (U|K: a jump)
    const path = (band.labels || []).map((l) => String(l).split("|").map((x) => esc(x)).join(" | ")).join(" → ");
    const rows = [["Path", path || "—"]];
    if (info.found) {
      rows.push(["Segments", String(info.lines ?? segs.length)], ["k points", info.points != null ? String(info.points) : "—"],
        ["ISS (shrinking factor)", info.iss != null ? String(info.iss) : "—"], ["From", esc(info.source)]);
    } else if (info.loading) rows.push(["From", "reading…"]);
    else rows.push(["From", esc(info.error || "the plot's tick labels (no .d3, fort.25 records or BAND.DAT header)")]);
    if (d.opts.kpath) rows.push(["Drawn", "the path as edited (k-path editor, Advanced)"]);
    const list = (info.found ? info.segments : []).map((s, i) => {
      const n = named[i] || s, a = s.k_from ? ` <span class="muted">(${kText(s.k_from)})</span>` : s.ints ? ` <span class="muted">(${s.ints[0].join(" ")})</span>` : "";
      const b = s.k_to ? ` <span class="muted">(${kText(s.k_to)})</span>` : s.ints ? ` <span class="muted">(${s.ints[1].join(" ")})</span>` : "";
      return `<li>${lab(n.from)}${a} → ${lab(n.to)}${b}</li>`;
    }).join("");
    el.innerHTML = `<dl class="po-kfacts">${rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("")}</dl>` +
      (list ? `<ol class="po-ksegs" aria-label="Path segments">${list}</ol>` +
        (info.iss == null && info.segments.some((s) => s.ints) ? `<p class="po-note">The segment ends are integers of the run's ISS, which its fort.25 does not record (the .d3 or .d12 beside it does).</p>` : "") : "");
  }

  optsHost.addEventListener("keydown", (e) => {
    // Mod+Z in the k-path editor is the plot's undo (the other controls keep their own, a text field's)
    if (!doc || !(e.ctrlKey || e.metaKey) || e.key.toLowerCase() !== "z" || !e.target.closest(".po-kpath") || e.target.matches("input")) return;
    e.preventDefault(); e.stopPropagation();
    KE.keepFocus("plot");
    const l = e.shiftKey ? redo(doc) : undo(doc);
    if (l) toast(`${e.shiftKey ? "Redo" : "Undo"}: ${l}`, "info");
  });

  /* ------------------------------------------------------------ the Plot options inspector */
  function num(id, label, v, unit, step = "0.5") {
    return `<label class="po-row" for="${id}"><span>${esc(label)}</span><input id="${id}" type="number" step="${step}" value="${v ?? ""}" placeholder="auto"><span class="po-u">${esc(unit)}</span></label>`;
  }
  /* the options are drawn again after every change: the control that had the focus keeps it, so the keyboard
   * (arrow keys on a slider, a number field) can go on changing it */
  function renderOptions() {
    const ae = document.activeElement, id = ae && ae.id && optsHost.contains(ae) ? ae.id : null;
    renderOptionsNow();
    const again = id && optsHost.querySelector(`#${CSS.escape(id)}`);
    if (again) again.focus({ preventScroll: true });
  }
  const resetButton = () => `<div class="po-acts"><button class="tb-btn-text" data-po="reset"${doc.pos > 0 || JSON.stringify(doc.opts) !== JSON.stringify(defaultsFor(doc)) ? "" : " disabled"}>${icon("rotate-ccw")}<span>Reset Plot Options</span></button></div>`;
  /* Plot options › Style and the toolbar's Style ▾ menu draw ONE list of controls (styleControls), each an
     option of the plot (presets.py STYLE_OPTIONS names; undo, Reset; "Settings" follows Settings, shown in
     brackets); Make This the Default writes them to Settings. A band + DOS plot has a Colors group (the color
     pair, both halves'), a Bands group (band style, band color, β line) and a DOS group (DOS projections, spin
     layout, total, smoothing), whichever of the two files was opened: the groups follow the figure's data, not
     the opened file's kind. A BAND.DAT alone has no DOS group, a DOSS.DAT alone no Bands group (its total's
     color and β line in the DOS group). */
  const PRESET_LABEL = (id) => { const p = H.presets && H.presets() && H.presets().SHARED.presets[id]; return p ? p.label : id; };
  const SCHEME_LABEL = (id) => (id === "auto" ? "Auto (the plot style's)" : id === "custom" ? "Custom" : ((H.presets && H.presets() && H.presets().SHARED.spin_schemes[id]) || {}).label || id);
  const DOS_LABEL = { auto: "Auto", total: "Total only", element: "By element", orbital: "By orbital", "element-orbital": "Element × orbital" };
  const DSPIN_LABEL = { auto: "Auto", mirrored: "Mirrored", overlaid: "Overlaid" };
  // DOS smoothing turned on: the shared width (presets.py DOS_SMOOTH_SIGMA, 0.05 eV); the Style menu's σ choices
  const SMOOTH_SIGMA = () => ((H.presets && H.presets() && H.presets().SHARED.dos_layout) || {}).smooth_sigma || 0.05;
  const SIGMA_CHOICES = [0.02, 0.05, 0.1, 0.2];
  const isBandDos = (d) => !!d && !isOwn(d) && !isSpec(d);
  function lookNow(d = doc) {
    if (isBandDos(d)) return envOf(d.opts).look;
    return { style: presetOf(familyOf(d)) };
  }
  /* the plot's style controls in groups: [{ key, head, ctrls }]; a control is { id (the Inspector's input),
     mv (the Style menu's value prefix), type (select, color, toggle, number), label, current (the value drawn),
     value (a select's own option, null: Settings), settings (Settings' value, as text), choices, apply(v) →
     [patch, undo label, merge key] } */
  function styleControls(d = doc) {
    const pp = H.presets ? H.presets() : null;
    if (!pp || !d || !d.opts) return [];
    const o = d.opts, fam = familyOf(d), look = lookNow(d);
    const base = isBandDos(d) ? settingsLook() : { style: H.plotPreset ? H.plotPreset(fam) : "default" };
    const choose = (id, mv, name, label, value, current, choices, settings, word) => ({ id, mv, type: "select", label, value, current, choices, settings,
      apply: (v) => [{ [name]: v || null }, `${label} ${v ? word(v) : "as Settings"}`] });
    const groups = [{ key: "plot", head: "Plot style", ctrls: [choose("po-style", "style", "style", "Plot style", o.style, look.style,
      pp.SHARED.order.filter((id) => pp.SHARED.presets[id].families.includes(fam)).map((id) => [id, PRESET_LABEL(id)]), PRESET_LABEL(base.style), PRESET_LABEL)] }];
    const BG = { theme: "Theme", white: "White", black: "Black", custom: "Custom" }, bg = bgOf(d);
    groups[0].ctrls.push(choose("po-bg", "bg", "background", "Background", o.background, bg.mode, Object.entries(BG), BG[setting("plots.background", "theme")] || "Theme", (v) => BG[v] || v));
    if (bg.mode === "custom") groups[0].ctrls.push({ id: "po-bgc", mv: "bgc", type: "color", label: "Background color", current: bgHex(d),
      apply: (v) => [{ background: "custom", background_color: v }, `Background ${v}`, "opt:po-bgc"] });
    if (!isBandDos(d)) return groups;
    const data = d.data || {}, band = data.band, dos = data.dos;
    if (!band && !dos) return groups;
    const color = (id, mv, name, label, current, title) => ({ id, mv, type: "color", label, current, title,
      apply: (v) => [{ [name]: v }, `${label} ${v}`, `opt:${id}`] });
    /* Colors: the Color pair (--spin-colors) drives the bands and the DOS at once: spin up / down of a
       spin-polarized run, a closed-shell run's Single band color (the first) and Overlay pair, the DOS total.
       Bands: Band style (Auto: Single with a gap, Overlay for a metal; Ink the neutral look), Single's
       Band color override, the β line. A DOSS.DAT alone: its total's Band color and β line in the DOS group */
    const tone = toneOf(doc), pid = pp.resolveId(look.style, "band");
    const parts = BF.lookParts(envOf(o), band || null).spin || {};
    const colors = [choose("po-spinc", "spin", "spin_colors", "Color pair", o.spin_colors, look.spin_colors || "auto",
      pp.SHARED.spin_color_choices.map((v) => [v, SCHEME_LABEL(v)]), SCHEME_LABEL(base.spin_colors), SCHEME_LABEL)];
    if (look.spin_colors === "custom") {
      const pair = pp.spinColors(pid, "band", tone, look.spin_colors, look.spin_up_color, look.spin_down_color);
      colors.push(color("po-spinup", "spinup", "spin_up_color", "First color", pair.up), color("po-spindn", "spindn", "spin_down_color", "Second color", pair.down));
    }
    groups.push({ key: "colors", head: "Colors", ctrls: colors });
    const BSL = { auto: "Auto", single: "Single", overlay: "Overlay", ink: "Ink" };
    const bsWord = (v) => (v === "auto" ? `Auto (${BSL[pp.bandStyle("auto", band)].toLowerCase()})` : BSL[v] || v);
    const spin = [];
    if (band) spin.push({ ...choose("po-bstyle", "bstyle", "band_style", "Band style", o.band_style, look.band_style || "auto",
      Object.keys(BSL).map((v) => [v, bsWord(v)]), bsWord(base.band_style || "auto"), bsWord),
      title: "A closed-shell run's bands: Single (one line, the pair's first color), Overlay (the second color dashed on top), Ink (the neutral color); Auto: Single with a gap, Overlay for a metal. A spin-polarized run draws up solid, down dashed (mace plotting --band-style)" });
    // the closed-shell half's Single color: the pair's first, or a custom Band color
    const closedHalf = (band && !band.spin_polarized) || (dos && !dos.spin_polarized);
    if (closedHalf && parts.style === "single") {
      const bcLabel = band ? "Band color" : "Total color", custom = /^#[0-9a-f]{6}$/i.test(look.band_color || "");
      const bc = parts.bands || "";
      const BCM = { auto: "Auto (first color)", custom: "Custom" };
      const tip = "Single bands and the one-sided DOS total of a run that is not spin-polarized (mace plotting --band-color)";
      spin.push({ id: "po-bandm", mv: "bandm", type: "select", label: bcLabel, value: o.band_color === null || o.band_color === undefined ? null : /^#[0-9a-f]{6}$/i.test(o.band_color) ? "custom" : "auto",
        current: custom ? "custom" : "auto", choices: Object.entries(BCM), settings: setting("plots.bandColor", "") ? "Custom" : BCM.auto, title: tip,
        apply: (v) => [{ band_color: v === "custom" ? bc : v === "auto" ? "" : null }, `${bcLabel} ${v ? BCM[v] : "as Settings"}`] });
      spin.push(color("po-bandc", "bandc", "band_color", band ? "Custom band color" : "Custom total color", bc, tip));
    }
    const dash = look.spin_down_style !== "solid";
    spin.push({ id: "po-bdash", mv: "bdash", type: "toggle", label: "β dashed", current: dash,
      title: "Spin down (β) dashed, spin up solid: the spins read apart without color (mace plotting --spin-down-solid turns it off)",
      apply: (v) => [{ spin_down_style: v ? "dash" : "solid" }, v ? "β dashed" : "β solid"] });
    if (band) groups.push({ key: "bands", head: "Bands", ctrls: spin });
    if (dos) {
      const sm = Number(look.dos_smooth) || 0;
      const ctrls = [
        choose("po-dosc", "dos", "dos_colors", "DOS projections", o.dos_colors, look.dos_colors || "auto", Object.entries(DOS_LABEL), DOS_LABEL[base.dos_colors] || base.dos_colors, (v) => DOS_LABEL[v] || v),
        choose("po-dspin", "dspin", "dos_spin", "DOS spin layout", o.dos_spin, look.dos_spin || "auto", Object.entries(DSPIN_LABEL), DSPIN_LABEL[base.dos_spin] || base.dos_spin, (v) => DSPIN_LABEL[v] || v),
        { id: "po-dtotal", mv: "dtotal", type: "toggle", label: "DOS total", current: look.dos_total !== false,
          apply: (v) => [{ dos_total: v }, v ? "DOS total" : "No DOS total"] },
        // DOS smoothing (display only, labeled in the plot): mace plotting --dos-smooth, [plots] dos_smooth
        { id: "po-dsmooth", mv: "dsmooth", type: "toggle", label: "DOS smoothing", current: sm > 0, title: "Gaussian smoothing of the drawn DOS, display only (mace plotting --dos-smooth)",
          apply: (v) => [{ dos_smooth: v ? SMOOTH_SIGMA() : 0 }, v ? "DOS smoothing" : "No DOS smoothing"] }];
      if (sm > 0) ctrls.push({ id: "po-dsigma", mv: "dsigma", type: "number", label: "Smoothing σ", unit: "eV", step: "0.01", current: sm,
        choices: [...new Set([...SIGMA_CHOICES, sm])].sort((a, b) => a - b).map((v) => [String(v), `σ ${v} eV`]),
        apply: (v) => { const s = Math.min(1, Math.max(0, Number(v) || 0)); return [{ dos_smooth: s }, `DOS smoothing σ ${s} eV`, "opt:dsigma"]; } });
      groups.push({ key: "dos", head: "DOS", ctrls: band ? ctrls : spin.concat(ctrls) });
    }
    return groups;
  }
  const styleControl = (pick) => { for (const g of styleControls()) for (const c of g.ctrls) if (pick(c)) return c; return null; };
  /* the Plot options panel's sections: one <details> each (the toolbar's menus of the same names), open or shut as the
     user left them; the niche ones carry data-tier="advanced" (the Inspector's Basic/Advanced tabs) */
  const catOpen = {};
  const sec = (key, title, body, opts = {}) => (body ? `<details class="po-cat" data-cat="${key}"${(catOpen[key] ?? opts.open ?? false) ? " open" : ""}${opts.adv ? ' data-tier="advanced"' : ""}><summary>${esc(title)}</summary>${body}</details>` : "");
  optsHost.addEventListener("toggle", (e) => { const d = e.target.closest && e.target.closest("details.po-cat"); if (d) catOpen[d.dataset.cat] = d.open; }, true);
  // style controls that only matter once the basics are set
  const ADV_CTRL = new Set(["po-bg", "po-bgc", "po-spinup", "po-spindn", "po-bandm", "po-bandc", "po-bdash", "po-dtotal", "po-dsmooth", "po-dsigma"]);
  function styleSection(which = "") {
    const pp = H.presets ? H.presets() : null;
    if (!pp || !doc) return "";
    const sel = (c) => `<label class="po-row" for="${c.id}"><span>${esc(c.label)}</span><select id="${c.id}">` +
      `<option value=""${c.value === null || c.value === undefined ? " selected" : ""}>Settings (${esc(c.settings)})</option>` +
      c.choices.map(([v, t]) => `<option value="${esc(v)}"${c.value === v ? " selected" : ""}>${esc(t)}</option>`).join("") + "</select></label>";
    const tip = (c) => (c.title ? ` title="${esc(c.title)}"` : "");
    const adv = (c, html) => (ADV_CTRL.has(c.id) ? `<div data-tier="advanced">${html}</div>` : html);
    const ctrl0 = (c) => c.type === "select" ? sel(c)
      : c.type === "color" ? `<label class="po-row" for="${c.id}"${tip(c)}><span>${esc(c.label)}</span><input id="${c.id}" type="color" value="${esc(c.current)}"></label>`
      : c.type === "toggle" ? `<label class="chk po-row"${tip(c)}><input type="checkbox" id="${c.id}"${c.current ? " checked" : ""}> ${esc(c.label)}</label>`
      : num(c.id, c.label, c.current, c.unit, c.step);
    const ctrl = (c) => adv(c, ctrl0(c));
    const want = STYLE_GROUPS[which] || null;
    let h = `<div class="po-style" role="group" aria-label="${which === "traces" ? "Traces" : "Style"}">${which ? "" : '<div class="po-h">Style</div>'}`;
    for (const g of styleControls().filter((x) => !want || want.includes(x.key))) {
      h += `<div class="po-sgroup" role="group" aria-label="${esc(g.head)}" data-group="${g.key}">` +
        (g.key === "plot" ? "" : `<div class="po-h po-sub">${esc(g.head)}</div>`) + g.ctrls.map(ctrl).join("") + "</div>";
    }
    if (which === "traces" && !want.some((k) => styleControls().some((x) => x.key === k))) return "";
    const own = styleNames().some((n) => doc.opts[n] !== null && doc.opts[n] !== undefined);
    if (which !== "traces") h += `<div class="po-acts"><button class="tb-btn-text" data-po="default"${own ? "" : " disabled"} title="Save this plot's style as the default of every plot (Settings)">Make This the Default</button></div>`;
    return h + "</div>";
  }
  /* one style control's new value (the Inspector's input or the Style menu's pick): one undoable option change */
  function styleApply(c, v) {
    if (!doc || !c) return;
    const [patch, label, mergeKey] = c.apply(v);
    setOpt(doc, patch, label, mergeKey || null);
  }
  function makeDefault() {
    if (!doc || !H.setSetting) return;
    const look = lookNow(), fam = familyOf(doc), set = (k, v) => { try { H.setSetting(k, v); } catch (e) { /* an unknown key */ } };
    if (doc.opts.style) {
      // the family's own preset when one is set in Settings, else every family's
      const own = setting(`plots.style.${fam}`, "same");
      set(own && own !== "same" ? `plots.style.${fam}` : "plots.style", look.style);
    }
    if (isBandDos(doc)) {
      // both halves of a band + DOS plot
      set("plots.spinColors", look.spin_colors); set("plots.spinUp", look.spin_up_color || ""); set("plots.spinDown", look.spin_down_color || "");
      set("plots.bandColor", look.band_color || ""); set("plots.bandStyle", look.band_style || "auto");
      set("plots.spinDownDash", look.spin_down_style); set("plots.dosColors", look.dos_colors); set("plots.spinDos", look.dos_spin);
      set("plots.dosTotal", look.dos_total !== false);
      set("plots.dosSmooth", Number(look.dos_smooth) || 0);
    }
    if (doc.opts.background) { const b = bgOf(doc); set("plots.background", b.mode); if (b.mode === "custom") set("plots.backgroundColor", bgHex(doc)); }
    const clear = Object.fromEntries(styleNames().map((n) => [n, null]).filter(([k]) => k in doc.opts));
    setOpt(doc, clear, "Style saved as the default");
    toast("This plot's style is the default now (Settings)", "success");
  }
  /* the toolbar's style menus: the same controls (styleControls), one pick at a time; a color opens the
     system color picker, a number its choices. A band or DOS plot splits them by what they change: Style (the
     plot style, background and color pair) and Traces (the band and DOS lines); Reference and Labels are popovers
     (catPop), Axes & Ranges the shared Plot Options popover with the energy window added (extraAxes). Another
     plot has the one Style menu. */
  const STYLE_GROUPS = { style: ["plot", "colors"], traces: ["bands", "dos"] };
  function styleMenu(cat = "") {
    if (!doc) return [];
    const want = STYLE_GROUPS[cat] || null;
    const items = [];
    styleControls().filter((g) => !want || want.includes(g.key)).forEach((g, k) => {
      if (k) items.push("-", { head: g.head, group: g.key });
      else if (g.key !== "plot") items.push({ head: g.head, group: g.key });
      for (const c of g.ctrls) {
        if (c.type === "select" || c.type === "number") {
          if (c.type === "number" || g.key !== "plot" || c !== g.ctrls[0]) items.push({ head: c.label });
          else items.push({ head: g.head, group: g.key });
          c.choices.forEach(([v, t]) => items.push({ v: `${c.mv}:${v}`, label: t, checked: String(c.current) === String(v) }));
        } else if (c.type === "toggle") items.push({ v: c.mv, label: c.label, checked: !!c.current, toggle: true });
        else items.push({ v: c.mv, label: `${c.label}…` });
      }
    });
    if (!items.length) return items;
    if (!cat || cat === "style") {
      items.push("-", { v: "settings", label: "Follow Settings" });
      if (cliKind(doc)) items.push({ v: "cli", label: "Copy CLI Command" });
    }
    return items;
  }
  /* the toolbar's categories of a band or DOS plot (plotkit: one button each) */
  function categories() {
    if (!doc || !isBandDos(doc) || !doc.data || (!doc.data.band && !doc.data.dos)) return [];
    const keys = styleControls().map((g) => g.key), phonon = isPhonon(doc);
    const out = [{ key: "style", label: "Style", type: "menu", title: "Plot style, background and color pair" }];
    if (keys.includes("bands") || keys.includes("dos")) out.push({ key: "traces", label: "Traces", type: "menu", title: "The band and DOS lines: band style, β line, DOS projections, spin layout, total, smoothing" });
    out.push({ key: "ref", label: "Reference", type: "pop", title: phonon ? "Frequency units and the fort.25 options" : "Fermi shift, energy zero and the Fermi level line" });
    out.push({ key: "labels", label: "Labels", type: "pop", title: "k-point labels and the spin words over the DOS" });
    return out;
  }
  const isPhonon = (d) => !!(d && d.data && ((d.data.band && d.data.band.domain === "phonon") || (d.data.dos && d.data.dos.domain === "phonon")));
  const ERW = { auto: "Auto (vacuum with the POTC files, else the Fermi level)", fermi: "Fermi level", vacuum: "Vacuum (needs the POTC files)" };
  /* Reference ▾: the energy zero. A Fermi shift moves it (also by dragging the dotted line); Fermi level or
     vacuum overrides Auto (mace plotting --fermi-shift, --energy-reference); the dotted line can be left out */
  function refPop() {
    const o = doc.opts, er = (doc.data && (doc.data.band || doc.data.dos)) || {}, f25 = /\.f25$|(^|\/)fort\.25$/i.test(doc.path || "");
    const opt = (v, t, dis) => `<option value="${v}"${(o.reference || "auto") === v ? " selected" : ""}${dis ? " disabled" : ""}>${esc(t)}</option>`;
    let h = "";
    if (isPhonon(doc)) {
      h += `<p class="po-note">Phonon frequencies: the axis unit is in Units ▾ (cm⁻¹, THz, meV).</p>`;
    } else {
      h += `<label class="pk-row"><span>Fermi shift</span><input type="number" step="0.01" id="pr-shift" value="${o.shift ?? 0}" aria-label="Fermi shift (eV)"><span class="po-u">eV</span></label>` +
        `<label class="pk-row"><span>Energy zero</span><select id="pr-ref" aria-label="Energy zero">${opt("auto", ERW.auto)}${opt("fermi", ERW.fermi)}${opt("vacuum", ERW.vacuum, !er.has_potc)}</select></label>` +
        `<label class="pk-row"><input type="checkbox" id="pr-fl"${o.fermi_line !== false ? " checked" : ""}> Dotted Fermi level line</label>`;
    }
    if (f25) {
      h += `<div class="pk-row" data-tier="advanced"><span>fort.25 holds</span><select id="pr-dom" aria-label="fort.25 data">${[["", "Read from the run"], ["electronic", "Electronic (energies)"], ["phonon", "Phonon (frequencies)"]]
        .map(([v, t]) => `<option value="${v}"${(o.f25_domain || "") === v ? " selected" : ""}>${t}</option>`).join("")}</select></div>`;
      const n = er.n_runs || 1;
      if (n > 1) h += `<label class="pk-row" data-tier="advanced"><span>Run</span><select id="pr-run" aria-label="Run of the fort.25">${Array.from({ length: n }, (_, k) =>
        `<option value="${k + 1}"${(o.f25_run || n) === k + 1 ? " selected" : ""}>${k + 1} of ${n}${k + 1 === n ? " (last)" : ""}</option>`).join("")}</select></label>`;
    }
    return h;
  }
  function refChange(t) {
    if (!doc) return false;
    const d = doc;
    if (t.id === "pr-shift") { const v = Number(t.value); setOpt(d, { shift: Number.isFinite(v) ? v : 0 }, `Fermi shift ${t.value || 0}`, "opt:shift"); return true; }
    if (t.id === "pr-ref") { setOpt(d, { reference: t.value === "auto" ? null : t.value }, `Energy zero ${ERW[t.value] ? ERW[t.value].split(" (")[0] : t.value}`); return true; }
    if (t.id === "pr-fl") { setOpt(d, { fermi_line: t.checked }, t.checked ? "Fermi level line" : "No Fermi level line"); return true; }
    if (t.id === "pr-dom") { setOpt(d, { f25_domain: t.value || null }, `fort.25 as ${t.value || "the run says"}`); return true; }
    if (t.id === "pr-run") { const n = Number(t.value); const last = ((d.data && (d.data.band || d.data.dos)) || {}).n_runs || 1; setOpt(d, { f25_run: n === last ? null : n }, `Run ${n}`); return true; }
    return false;
  }
  function labelsPop() {
    const o = doc.opts, band = doc.data && doc.data.band, dos = doc.data && doc.data.dos;
    let h = "";
    if (band) h += `<label class="pk-row"><input type="checkbox" id="pl-k"${o.k_labels !== false ? " checked" : ""}> k-point labels under the bands</label>`;
    if (dos) h += `<label class="pk-row"><input type="checkbox" id="pl-s"${o.spin_labels !== false ? " checked" : ""}> “up” and “down” over a mirrored DOS</label>`;
    return h || `<p class="po-note">Nothing to label.</p>`;
  }
  function labelsChange(t) {
    if (!doc) return false;
    if (t.id === "pl-k") { setOpt(doc, { k_labels: t.checked }, t.checked ? "k-point labels" : "No k-point labels"); return true; }
    if (t.id === "pl-s") { setOpt(doc, { spin_labels: t.checked }, t.checked ? "Spin labels" : "No spin labels"); return true; }
    return false;
  }
  const catPop = (key) => key === "ref" ? { html: doc ? refPop() : "", change: refChange } : { html: doc ? labelsPop() : "", change: labelsChange };
  /* Axes & Ranges ▾ (the Plot Options popover): the energy window leads it */
  function extraAxes() {
    if (!doc || !isBandDos(doc)) return null;
    const o = doc.opts, ph = isPhonon(doc);
    const f = (id, lab, v) => `<label class="pk-row"><span>${lab}</span><input type="number" step="0.5" data-e="${id}" value="${v ?? ""}" aria-label="${lab}"><span class="po-u">${ph ? "eV" : "eV"}</span></label>`;
    return { html: ph ? "" : f("emin", "Energy from", o.emin) + f("emax", "Energy to", o.emax),
      change: (t) => {
        if (!t.dataset.e) return false;
        const v = t.value === "" ? defaultsFor(doc)[t.dataset.e] : Number(t.value);
        setOpt(doc, { [t.dataset.e]: v }, `Energy ${t.dataset.e === "emin" ? "from" : "to"} ${t.value || "default"}`, `opt:${t.dataset.e}`);
        return true;
      } };
  }
  function pickColor(c) {
    document.getElementById("pk-colorpick")?.remove();
    const i = document.createElement("input");
    i.type = "color"; i.id = "pk-colorpick"; i.value = c.current; i.setAttribute("aria-label", c.label);
    i.style.cssText = "position:fixed;left:50%;top:40%;width:1px;height:1px;opacity:0;border:0;padding:0";
    const d = doc;
    i.addEventListener("change", () => { if (doc === d) styleApply(c, i.value); i.remove(); });
    i.addEventListener("blur", () => setTimeout(() => i.isConnected && i.remove(), 60000));
    document.body.appendChild(i);
    try { if (i.showPicker) i.showPicker(); else i.click(); } catch (e) { i.click(); }
  }
  function styleSet(v) {
    if (!doc) return;
    const at = v.indexOf(":"), k = at < 0 ? v : v.slice(0, at), val = at < 0 ? null : v.slice(at + 1);
    if (k === "settings") {
      const clear = Object.fromEntries(styleNames().map((n) => [n, null]).filter(([k]) => k in doc.opts));
      setOpt(doc, clear, "Style as Settings");
      return;
    }
    const c = styleControl((x) => x.mv === k);
    if (!c) return;
    if (c.type === "toggle") styleApply(c, !c.current);
    else if (c.type === "color") pickColor(c);
    else styleApply(c, val);
  }
  /* Copy CLI Command (clicmd.js): mace plotting with this plot's file, window, energy zero, unit and look */
  const cliKind = (d) => (!d || !d.data ? null : d.data.band && d.data.dos ? (isPhonon(d) ? "phbandos" : "bandos") : d.data.band ? "band" : d.data.dos ? "dos"
    : d.kind === "ir" || d.kind === "raman" ? (d.source === "modes" ? null : d.kind) : d.kind === "transport" && d.data.transport ? "transport" : null);
  /* a transport plot's options as mace plotting --transport takes them (the quantity, temperature and x axis always named) */
  function transportOpts(d) {
    const t = d.data.transport, o = d.opts, qn = t.quantity;
    return { quantity: qn, component: o.component || "avg", temperature: t.blocks[0].T === null ? null : String(o.T) === "all" ? "all" : (o.T ?? t.blocks[0].T),
      x: t.x === "energy" ? null : o.x || "mu", kappa_l: qn === "zt" && !t.upper_bound ? String(o.kl || "").trim() : "",
      tau_fs: ["pf", "kappa_e", "zt"].includes(qn) && o.tau !== null && o.tau !== undefined ? o.tau : null };
  }
  function cliText(d = doc) {
    const kind = cliKind(d);
    if (!kind) throw new Error("no mace plotting command draws this plot");
    if (kind === "transport") return cliCommand({ kind, file: d.data.transport.name, look: lookNow(d), transport: transportOpts(d) });
    const er = d.data.band || d.data.dos;
    let file = kind === "dos" ? d.data.dos.name : kind === "band" || kind === "bandos" || kind === "phbandos" ? d.data.band.name : d.title, partner = null;
    // a band + DOS pair from two folders: run in the opened file's folder, the other half relative to it
    if ((kind === "bandos" || kind === "phbandos") && d.opts.pair && d.data.pair && !d.data.pair.error) { file = d.path.split("/").pop(); partner = relPath(dirOf(d.path), d.opts.pair); }
    return cliCommand({ kind, file, partner, emin: d.opts.emin, emax: d.opts.emax, shift: d.opts.shift || 0,
      reference: isPhonon(d) ? "auto" : er && er.reference === "vacuum" ? "vacuum" : "fermi", unit: d.opts.unit || "eV", look: lookNow(d),
      menus: menuArgs(d.opts),
      kpath: kind !== "dos" && d.opts.kpath ? d.opts.kpath : null, logDos: !!d.opts.ylog,
      // with a mace-folder.toml its [plots] and [units] values would replace any default left out
      explicit: !!(H.folderFile && H.folderFile()) });
  }
  function renderOptionsNow() {
    if (!doc) { optsHost.innerHTML = ""; return; }
    const o = doc.opts, sp = doc.kind === "ir" || doc.kind === "raman";
    let h = "";
    if (doc.kind === "map") {
      const M = doc.data && doc.data.map;
      if (M && M.maps.length > 1) h += `<label class="po-row" for="po-mwhich"><span>Map</span><select id="po-mwhich">${M.maps.map((m, k) =>
        `<option value="${k}"${(o.which || 0) === k ? " selected" : ""}>${esc(m.label)}</option>`).join("")}</select></label>`;
      h += `<label class="chk po-row"><input type="checkbox" id="po-mlog"${o.log ? " checked" : ""}> Logarithmic scale (positive maps)</label>` +
        num("po-mcont", "Contours", o.contours, "", "1") +
        `<label class="po-row" for="po-mstyle"><span>Contour style</span><select id="po-mstyle">${[["", "Plot style's"], ["filled", "Filled"], ["lines", "Lines"], ["both", "Filled and lines"]].map(([v, t]) =>
          `<option value="${v}"${(o.contour_style || "") === v ? " selected" : ""}>${t}</option>`).join("")}</select></label>` +
        `<label class="po-row" for="po-crange"><span>Color range</span><select id="po-crange">${[["robust", "Auto (1st–99th percentile)"], ["full", "Full (min to max)"], ["custom", "Custom"]].map(([v, t]) =>
          `<option value="${v}"${(o.crange || "robust") === v ? " selected" : ""}>${t}</option>`).join("")}</select></label>` +
        (o.crange === "custom" ? num("po-zmin", "Color from", o.zmin, o.log ? "log₁₀" : "", "any") + num("po-zmax", "Color to", o.zmax, o.log ? "log₁₀" : "", "any") : "");
      if (M) { const m = M.maps[o.which || 0]; h += `<p class="po-h">${esc(m.label)}: min ${m.min.toPrecision(4)} · max ${m.max.toPrecision(4)} (${L.html(m.unit || M.unit)})</p>`; }
      h += styleSection() + resetButton();
      optsHost.innerHTML = h;
      return;
    }
    if (doc.kind === "convergence") {
      h += num("po-width", "Line width", o.width, "px", "0.2") +
        `<label class="chk po-row"><input type="checkbox" id="po-clog"${o.log ? " checked" : ""}> Logarithmic energy axis</label>` +
        `<p class="po-h">Each run's energy per atom at each printed geometry; molecules print only the initial and final ones.</p>`;
      h += styleSection() + resetButton();
      optsHost.innerHTML = h;
      return;
    }
    if (doc.kind === "transport") {
      const t = doc.data && doc.data.transport;
      if (t) {
        // the Quantity choice, as a density volume offers its run's potential: every quantity of the run, those its
        // files beside it do not allow grayed with the reason (on rows of their own: eleven do not fit beside the label)
        const btn = (k) => { const why = t.available[k]; return `<button role="radio" aria-checked="${k === t.quantity}" data-tquantity="${k}"${why ? ` aria-disabled="true"` : ""} title="${esc(why ? `${TQ_TITLE[k]}: ${why}` : TQ_TITLE[k])}">${TQ_SHORT[k]}</button>`; };
        h += `<div class="po-row po-stack"><span>Quantity</span><span class="segment po-tq" role="radiogroup" aria-label="Quantity from the files">${TQ_RAW.map(btn).join("")}</span>`
          + `<span class="segment po-tq" role="radiogroup" aria-label="Derived quantity">${TQ_DERIVED.map(btn).join("")}</span></div>`;
        if (t.blocks[0].T !== null) h += `<label class="po-row" for="po-temp"><span>Temperature</span><select id="po-temp">${t.blocks.map((b) =>
          `<option value="${b.T}"${String(o.T ?? t.blocks[0].T) === String(b.T) ? " selected" : ""}>${b.T} K</option>`).join("")}<option value="all"${String(o.T) === "all" ? " selected" : ""}>All temperatures</option></select></label>`;
        const comps = [["avg", `Average ${t.avg_label.replace(/ /g, "")}`], ...t.components.map((c) => [c, c])];
        if (t.components.length > 1 && String(o.T) !== "all") comps.push(["all", t.components.join(", ")]);
        h += `<label class="po-row" for="po-tcomp"><span>Component</span><select id="po-tcomp">${comps.map(([v, l]) =>
          `<option value="${v}"${(o.component || "avg") === v ? " selected" : ""}>${esc(l)}</option>`).join("")}</select></label>`;
        if (t.x !== "energy") h += `<label class="po-row" for="po-tx"><span>Against</span><select id="po-tx">${[["mu", "μ (eV)"], ["carriers", t.dim === 3 ? "Carriers (cm⁻³)" : "Carriers per cell"]].map(([v, l]) =>
          `<option value="${v}"${(o.x || "mu") === v ? " selected" : ""}>${l}</option>`).join("")}</select></label>`;
        if (t.quantity === "zt") h += `<label class="po-row" for="po-kl"><span>Lattice κ<sub>L</sub></span><input id="po-kl" type="text" spellcheck="false" autocomplete="off" value="${esc(o.kl || "")}" placeholder="value or .csv beside the run" title="The lattice thermal conductivity in W/(m·K): one value, or a CSV file with T,kappa_L rows (interpolated linearly, never extrapolated), named as it sits beside the run (else from the served folder)"><span class="po-u">W/(m·K)</span></label>`;
        if (["pf", "kappa_e", "zt"].includes(t.quantity)) h += num("po-tau", "Relaxation time τ", o.tau, "fs", "1").replace('placeholder="auto"', `placeholder="${t.tau_run_fs ? `${t.tau_run_fs} (the run's)` : "the run's"}"`);
        h += num("po-width", "Line width", o.width, "px", "0.2");
        if (doc.data.trError) h += `<p class="po-note po-err" role="alert">${esc(doc.data.trError)}</p>`;
        if (t.quantity === "zt") {
          h += `<p class="po-note">ZT<sub>e</sub> peak (electrons only, κ<sub>L</sub> = 0): ${esc(fmtZt(t.zt_e_max))}</p>`;
          if (t.zt_total_max) h += `<p class="po-note">ZT total peak: ${esc(fmtZt(t.zt_total_max))}, κ<sub>L</sub> ${Number(t.zt_total_max.kappa_l).toPrecision(4)} W/(m·K), τ ${t.tau_fs} fs (${esc(t.tau_fs !== t.tau_run_fs ? `given; the run used ${t.tau_run_fs} fs` : t.tau_source)})</p>`;
          else if (t.refused) h += `<p class="po-note po-err" role="alert">${esc(t.refused)}</p>`;
          else if (t.upper_bound && !doc.data.trError) h += `<p class="po-note">ZT<sub>e</sub> has no lattice term, so it is an upper bound: enter κ<sub>L</sub> for ZT = S<sup>2</sup>σT/(κ<sub>e</sub> + κ<sub>L</sub>). σ and κ<sub>0</sub> are at the run's relaxation time (RELAXTIM); the total depends on it.</p>`;
          const blk = t.blocks.find((b) => b.T === o.T) || t.blocks[0];
          if (blk.refused && String(o.T) !== "all") h += `<p class="po-note po-err" role="alert">${esc(blk.refused)}</p>`;
        } else if (t.tau_fs !== null && t.tau_fs !== undefined) h += `<p class="po-note">τ = ${t.tau_fs} fs (${esc(t.tau_source)})</p>`;
        for (const n of t.notes || []) h += `<p class="po-note">${esc(n)}</p>`;
      }
      h += styleSection() + `<div class="po-acts"><button class="tb-btn-text" data-po="reset">${icon("rotate-ccw")}<span>Reset Plot Options</span></button></div>`;
      optsHost.innerHTML = h;
      return;
    }
    if (!sp) {
      const band = doc.data && doc.data.band, dos = doc.data && doc.data.dos;
      const spin = (band && band.spin_polarized) || (dos && dos.spin_polarized);
      const ph = isPhonon(doc);
      let axes = ph ? `<p class="po-note">Phonon frequencies; pick the unit in Units ▾.</p>` : num("po-emin", "Energy from", o.emin, "eV") + num("po-emax", "Energy to", o.emax, "eV");
      axes += `<div data-tier="advanced">${num("po-width", "Line width", o.width, "px", "0.2")}</div>`;
      if (spin) axes += `<div class="po-row"><span>Spin</span><span class="segment po-spin" role="radiogroup" aria-label="Spin">${["both", "up", "down"].map((s) =>
        `<button role="radio" aria-checked="${o.spin === s}" data-spin="${s}">${s === "both" ? "Both" : s === "up" ? "α" : "β"}</button>`).join("")}</span></div>`;
      h += sec("axes", "Axes & ranges", axes, { open: true });
      const er = (doc.data && (band || dos)) || {};
      if (!ph) {
        const rf = num("po-shift", "Fermi shift", o.shift, "eV", "0.01") +
          `<div data-tier="advanced"><label class="po-row" for="po-ref"><span>Energy zero</span><select id="po-ref">${[["auto", "Auto"], ["fermi", "Fermi level"], ["vacuum", "Vacuum"]]
            .map(([v, t]) => `<option value="${v}"${(o.reference || "auto") === v ? " selected" : ""}${v === "vacuum" && !er.has_potc ? " disabled" : ""}>${t}</option>`).join("")}</select></label></div>` +
          `<label class="chk po-row"><input type="checkbox" id="po-fl"${o.fermi_line !== false ? " checked" : ""}> Dotted Fermi level line</label>`;
        h += sec("ref", "Fermi & reference", rf, { open: true });
      }
      let lb = "";
      if (band) lb += `<label class="chk po-row"><input type="checkbox" id="po-klabels"${o.k_labels !== false ? " checked" : ""}> k-point labels</label>`;
      if (dos) lb += `<label class="chk po-row"><input type="checkbox" id="po-sl"${o.spin_labels !== false ? " checked" : ""}> “up” and “down” over a mirrored DOS</label>`;
      h += sec("labels", "Labels", lb, { adv: true });
      if (band && pathSegments(band)) h += sec("path", "k-path", `<div class="po-kpath"></div>`, { adv: true, open: true });
      // the zone and the path it was computed along (every tier), the k-path editor's view: see bzSection
      if (band) h += sec("bz", "Brillouin zone", `<div class="po-bz"></div><div class="po-kinfo"></div>`, { open: true });
      // DOS colors (Settings › Plots, this plot's own choice over it) and the projections that mode draws, by their legend names
      if (dos) {
        // hidden: this plot's own list, or by default all but the 2-4 projections carrying the DOS (bandfig.js dosLook)
        const look = dosLook(dos, tokenOf("band"), o);
        h += sec("proj", "Projections", `<div class="po-proj">${look.lines.filter((l) => l.show).map((l) =>
          `<label class="chk"><input type="checkbox" data-proj="${esc(l.label)}"${look.hidden.includes(l.label) ? "" : " checked"}> ${esc(l.name)}</label>`).join("")}</div>`, { open: true });
      }
      h += sec("style", "Style", styleSection("style"), { open: true }) + sec("traces", "Traces", styleSection("traces"), { open: true });
      optsHost.innerHTML = h + resetButton();
      if (band) bzSection(optsHost.querySelector(".po-bz"), optsHost.querySelector(".po-kinfo"));
      const khb = optsHost.querySelector(".po-kpath");
      if (khb) kpathEditor(khb);
      return;
    } else {
      h += num("po-xmin", "Wavenumber from", o.xmin, "cm⁻¹", "10") + num("po-xmax", "Wavenumber to", o.xmax, "cm⁻¹", "10") +
        num("po-width", "Line width", o.width, "px", "0.2") +
        `<label class="chk po-row"><input type="checkbox" id="po-norm"${o.normalize ? " checked" : ""}> Normalize to the highest peak</label>`;
      const comps = doc.data ? Object.keys(doc.data.spectra[0].columns) : [];
      if (doc.data && (doc.kind === "raman" || comps.length > 1)) {
        const cur = o.component === "all" || comps.includes(o.component) ? o.component : comps[0];
        h += `<label class="po-row" for="po-comp"><span>Component</span><select id="po-comp">${["all", ...comps].map((c) => `<option value="${esc(c)}"${cur === c ? " selected" : ""}>${esc(c === "all" ? "All components" : c)}</option>`).join("")}</select></label>`;
      }
      if (doc.data && doc.data.group) h += `<label class="chk po-row"><input type="checkbox" id="po-avg"${o.average ? " checked" : ""}> Average the ${doc.data.group.length} conformers</label>`;
      if (doc.data && doc.data.spectra[0].broadening) h += `<p class="po-h">Broadening as CRYSTAL wrote it: ${esc(doc.data.spectra[0].broadening)}</p>`;
      if (doc.source === "modes") {
        const s0 = doc.data && doc.data.spectra[0];
        h += num("po-fwhm", "Peak width (FWHM)", o.fwhm ?? (s0 ? s0.default_fwhm : 8), "cm⁻¹", "1");
        if (doc.kind === "ir") h += `<label class="po-row" for="po-shape"><span>Line shape</span><select id="po-shape">${[["lorentzian", "Lorentzian"], ["gaussian", "Gaussian"]].map(([v, l]) =>
          `<option value="${v}"${o.shape === v ? " selected" : ""}>${l}</option>`).join("")}</select></label>`;
        else h += `<label class="po-row" for="po-eta"><span>Lorentzian share (η)</span><input id="po-eta" type="range" min="0" max="1" step="0.1" value="${o.eta ?? 1}"><span class="po-u">${(o.eta ?? 1).toFixed(1)}</span></label>`;
      }
    }
    h += styleSection() + resetButton();
    optsHost.innerHTML = h;
    const kh = optsHost.querySelector(".po-kpath");
    if (kh) kpathEditor(kh);
  }
  optsHost.addEventListener("change", (e) => {
    if (!doc) return;
    const t = e.target, o = doc.opts;
    const v = (x) => (x === "" ? null : Number(x));
    if (t.id === "po-ref") { setOpt(doc, { reference: t.value === "auto" ? null : t.value }, `Energy zero ${t.value}`); return; }
    if (t.id === "po-fl") { setOpt(doc, { fermi_line: t.checked }, t.checked ? "Fermi level line" : "No Fermi level line"); return; }
    // (po-kl is the transport view's κ_L field: the k-point labels box has its own id)
    if (t.id === "po-klabels") { setOpt(doc, { k_labels: t.checked }, t.checked ? "k-point labels" : "No k-point labels"); return; }
    if (t.id === "po-sl") { setOpt(doc, { spin_labels: t.checked }, t.checked ? "Spin labels" : "No spin labels"); return; }
    const map = { "po-emin": ["emin", "Energy from"], "po-emax": ["emax", "Energy to"], "po-shift": ["shift", "Fermi shift"],
      "po-width": ["width", "Line width"], "po-xmin": ["xmin", "Wavenumber from"], "po-xmax": ["xmax", "Wavenumber to"] };
    if (map[t.id]) { const [k, l] = map[t.id]; const val = v(t.value); setOpt(doc, { [k]: k === "shift" || k === "width" ? (val ?? defaultsFor(doc)[k]) : val }, `${l} ${t.value || "auto"}`, `opt:${k}`); }
    else if (t.id === "po-norm") setOpt(doc, { normalize: t.checked }, t.checked ? "Normalize" : "No normalization");
    else if (t.id === "po-avg") {
      const d = doc;
      const go = () => setOpt(d, { average: t.checked }, t.checked ? "Average the conformers" : "Each conformer");
      if (t.checked && !d.data.average) api("spectrum?" + q({ path: d.path, average: 1 })).then((a) => { d.data.average = a; go(); }).catch((err) => toast(err.message, "error"));
      else go();
    }
    else if (t.id === "po-comp") setOpt(doc, { component: t.value }, `Component ${t.value}`);
    else if (t.id && t.id.startsWith("po-") && styleControl((c) => c.id === t.id)) {
      // Plot options › Style (styleControls): the same controls as the Style menu
      const c = styleControl((x) => x.id === t.id);
      styleApply(c, c.type === "toggle" ? t.checked : t.value);
    }
    else if (t.id === "po-crange") setOpt(doc, { crange: t.value }, `Color range ${t.value}`);
    else if (t.id === "po-zmin" || t.id === "po-zmax") setOpt(doc, { [t.id === "po-zmin" ? "zmin" : "zmax"]: v(t.value) }, `Color ${t.id === "po-zmin" ? "from" : "to"} ${t.value || "auto"}`, `opt:${t.id}`);
    else if (t.id === "po-temp") setOpt(doc, { T: t.value === "all" ? "all" : Number(t.value) }, t.value === "all" ? "All temperatures" : `Temperature ${t.value} K`);
    else if (t.id === "po-tx") setOpt(doc, { x: t.value }, t.value === "carriers" ? "Against the carrier concentration" : "Against μ");
    else if (t.id === "po-mlog") setOpt(doc, { log: t.checked }, t.checked ? "Logarithmic scale" : "Linear scale");
    else if (t.id === "po-mcont") setOpt(doc, { contours: Math.max(2, Math.min(60, Number(t.value) || 20)) }, `Contours ${t.value}`);
    else if (t.id === "po-mstyle") setOpt(doc, { contour_style: t.value || null }, `Contour style ${t.value || "of the plot style"}`);
    else if (t.id === "po-mwhich") setOpt(doc, { which: Number(t.value) }, "Map");
    else if (t.id === "po-clog") setOpt(doc, { log: t.checked }, t.checked ? "Logarithmic axis" : "Linear axis");
    else if (t.id === "po-tcomp") setOpt(doc, { component: t.value }, `Component ${t.value}`);
    else if (t.id === "po-kl") setOpt(doc, { kl: t.value.trim() }, t.value.trim() ? `κ_L ${t.value.trim()}` : "No κ_L");
    else if (t.id === "po-tau") setOpt(doc, { tau: v(t.value) }, t.value ? `τ ${t.value} fs` : "τ of the run");
    else if (t.id === "po-fwhm" || t.id === "po-shape" || t.id === "po-eta") {
      const d = doc, patch = t.id === "po-fwhm" ? { fwhm: Math.min(50, Math.max(2, Number(t.value) || 8)) }
        : t.id === "po-shape" ? { shape: t.value } : { eta: Number(t.value), shape: Number(t.value) < 1 ? "pseudo-voigt" : "lorentzian" };
      Object.assign(d.opts, patch);
      d.data = null;
      fetchData(d).then(() => { if (doc === d) { setOpt(d, patch, t.id === "po-fwhm" ? `Peak width ${patch.fwhm} cm⁻¹` : `Line shape ${d.opts.shape}`); note(d); } })
        .catch((err) => toast(err.message, "error"));
    }
    else if (t.dataset.proj) {
      const hidden = new Set(doc.data && doc.data.dos ? dosLook(doc.data.dos, tokenOf("band"), o).hidden : o.hidden || []);
      if (t.checked) hidden.delete(t.dataset.proj); else hidden.add(t.dataset.proj);
      setOpt(doc, { hidden: [...hidden] }, `${t.checked ? "Show" : "Hide"} ${t.dataset.proj}`);
    }
  });
  optsHost.addEventListener("click", (e) => {
    // a transport quantity: in this document (its sibling files and the derived ones), as a volume colors by potential;
    // one the run's files do not allow says why
    const tq = e.target.closest("[data-tquantity]");
    if (tq && doc && tq.getAttribute("aria-disabled") === "true") { toast(tq.title, "info"); return; }
    if (tq && doc && tq.getAttribute("aria-checked") !== "true") { const k = tq.dataset.tquantity; setOpt(doc, { quantity: k, view: undefined }, TQ_TITLE[k] || k); return; }
    const s = e.target.closest("[data-spin]");
    if (s && doc) { setOpt(doc, { spin: s.dataset.spin }, `Spin ${s.dataset.spin}`); return; }
    const b = e.target.closest("[data-po]");
    if (b && doc && b.dataset.po === "reset") { reset(doc); toast("Plot options reset", "info", { label: "Undo", run: () => undo(doc) }); }
    if (b && doc && b.dataset.po === "default") makeDefault();
  });

  /* ------------------------------------------------------------ export and the shared options (plotkit) */
  const stemOf = (d) => d.title.replace(/\.[^.]+(\.[^.]+)?$/, "");
  async function image(fmt) {
    const scale = fmt === "png" ? Number(setting("plots.exportScale", 2)) || 2 : 1;
    const url = await window.Plotly.toImage(el, { format: fmt, scale, width: el.clientWidth, height: el.clientHeight });
    download(PK.dataUrlBlob(url), `${stemOf(doc)}.${fmt}`);
  }
  kit = PK.attach($(".pv-acts"), {
    get name() { return doc ? stemOf(doc) : "plot"; },
    host: el,
    get features() {
      // a transport plot: mace plotting --transport draws it (Publication Figure, Copy CLI Command)
      if (doc && isOwn(doc)) { const tr = doc.kind === "transport"; return { logY: false, logX: false, units: null, range: false, width: false, png: true, svg: true, image: true, csv: true, pub: tr, style: true, cli: tr }; }
      const bd = !!doc && isBandDos(doc) && !!doc.data && !!(doc.data.band || doc.data.dos);
      // Log Y is a DOS axis: a band plot alone has none (its energies are signed), a band + DOS plot's is Log DOS
      const both = bd && !!doc.data.band && !!doc.data.dos, bandOnly = bd && !!doc.data.band && !doc.data.dos;
      return { logY: !bandOnly, logX: !!doc && isSpec(doc),
        ...(both ? { logYLabel: "Log DOS", logYTitle: "Logarithmic DOS axis (the band energies stay linear)" } : {}),
        units: doc && isSpec(doc) ? "wavenumber" : isPhonon(doc) ? "phonon" : "energy", range: true, width: true,
        png: true, svg: true, image: true, csv: true, pub: !(doc && doc.source === "modes"), style: true, cli: !!cliKind(doc),
        ...(bd ? { moreLabel: "Axes & Ranges", moreTitle: "Energy window, axis ranges, line width and Reset Zoom" } : {}) };
    },
    get: () => (doc ? doc.opts : {}),
    set: (patch, label) => doc && setOpt(doc, patch, label),
    axes: () => (doc && isSpec(doc) ? { x: "Wavenumber", y: "Intensity" } : { x: "k-path / DOS", y: "Energy" }),
    table: () => tableOf(doc),
    png: () => image("png"), svg: () => image("svg"), image: () => exportImage(),
    pub: () => publication(),
    styleMenu: (cat) => styleMenu(cat), styleSet: (v) => styleSet(v), cli: () => cliText(),
    categories: () => categories(), catPop: (key) => catPop(key), extra: () => extraAxes(),
    buttons: () => pairButtons(),
    button: (key) => { if (!doc) return; if (key === "pair") pickPair(); else if (key === "unpair") { const w = pairWords(doc); setOpt(doc, { pair: null }, `Remove ${doc.kind === "dos" ? w.bandText : w.dosText}`); } },
    resetZoom: () => { if (window.Plotly && el.layout) { el.__sig = null; redraw(); } },
  }, { toast, download, icon });
  /* the plot's look as the plot task's parameters (plots.py look_args: the presets.py STYLE_OPTIONS names,
     only those that are not the defaults) */
  function lookOf(d) {
    const look = lookNow(d), out = {};
    for (const [k, v] of Object.entries(look)) {
      const dflt = H.presets ? H.presets().STYLE_DEFAULTS[k] : undefined;
      if (v === null || v === undefined || v === dflt) continue;
      out[k] = typeof v === "boolean" ? (v ? 1 : 0) : v;
    }
    return out;
  }
  /** {domain, run, fermi_line, k_labels, spin_labels} as the plot task reads them (plots.look_args): only what is
   * not the default (auto, the last run, on). */
  const menuArgs = (o) => ({ ...(o.f25_domain ? { domain: o.f25_domain } : {}), ...(o.f25_run ? { run: o.f25_run } : {}),
    ...(o.fermi_line === false ? { fermi_line: 0 } : {}), ...(o.k_labels === false ? { k_labels: 0 } : {}),
    ...(o.spin_labels === false ? { spin_labels: 0 } : {}) });
  async function publication() {
    // the edited k-path goes with it: mace plotting --band-path draws the same path (formatting is all it adds)
    // what is on screen: the band structure and its DOS when both are shown (mace plotting --band-dos)
    const both = !!(doc.data && doc.data.band && doc.data.dos), kind = both ? (isPhonon(doc) ? "phbandos" : "bandos") : null;
    const edit = (doc.kind === "band" || both) && doc.opts.kpath ? { band_path: JSON.stringify(doc.opts.kpath) } : {};
    // the window on screen (Settings' or the folder's, or the one typed in Plot options): --e-lower/--e-upper
    const win = (doc.kind === "band" || doc.kind === "dos") && Number.isFinite(Number(doc.opts.emin)) && Number.isFinite(Number(doc.opts.emax))
      ? { emin: Number(doc.opts.emin), emax: Number(doc.opts.emax) } : {};
    // and its energy zero: the Fermi shift (dragged or typed; --fermi-shift) and the reference drawn, always
    // named (--energy-reference fermi|vacuum; the server copies the POTC files along): auto would take the
    // vacuum scale whenever the POTC files are there, and a mace-folder.toml's energy_reference. With a
    // mace-folder.toml the server passes every value, the defaults too (plots.py explicit), as Copy CLI
    // Command writes them: the file's [plots] values must not replace what the screen shows
    const er = doc.data && (doc.data.band || doc.data.dos);
    const ref = (doc.kind === "band" || doc.kind === "dos") && er
      ? { ...(Number(doc.opts.shift) ? { shift: Number(doc.opts.shift) } : {}),
          ...(isPhonon(doc) ? {} : { reference: er.reference === "vacuum" ? "vacuum" : "fermi" }) } : {};
    // a fort.25's domain and run (--f25-domain, --f25-run) and the Reference and Labels menus' switches
    const menus = doc.kind === "band" || doc.kind === "dos" ? menuArgs(doc.opts) : {};
    // and its look (Settings › Appearance › Plot style, Plots): the same preset, DOS colors and spin styling
    const look = lookOf(doc);
    const unit = (doc.kind === "band" || doc.kind === "dos") && doc.opts.unit && doc.opts.unit !== "eV" ? { unit: doc.opts.unit } : {};
    // Log Y on a DOS, Log DOS on a band + DOS plot (--log-dos: the DOS axis only)
    if (doc.opts.ylog && doc.data && doc.data.dos) look.log_dos = 1;
    // a band + DOS pair from two folders (Add DOS…, Add Band…): its other half (mace plotting --partner)
    const partner = both && doc.opts.pair && doc.data.pair && !doc.data.pair.error ? { partner: doc.opts.pair } : {};
    // a transport plot: its quantity, temperature, component, x axis, κ_L and τ (mace plotting --transport)
    let tr = {};
    if (doc.kind === "transport" && doc.data && doc.data.transport) {
      const t = transportOpts(doc);
      tr = { kind: "transport", quantity: t.quantity, component: t.component, ...(t.temperature !== null ? { temperature: t.temperature } : {}),
        ...(t.x ? { x: t.x } : {}), ...(t.kappa_l ? { kappa_l: t.kappa_l } : {}), ...(t.tau_fs !== null ? { tau_fs: t.tau_fs } : {}) };
    }
    const r = await apiTask("plot?" + q({ path: doc.path, format: "svg", detach: 1, ...(kind ? { kind } : {}), ...partner, ...edit, ...win, ...ref, ...menus, ...unit, ...look, ...tr }), undefined, null, { get: true });
    const svg = r.images.find((n) => n.endsWith(".svg"));
    if (!svg) throw new Error("mace plotting made no SVG: " + (r.log || "").slice(-300));
    const res = await api("plot-image?" + q({ key: r.key, name: svg }), { blob: true });
    download(res.blob, svg);
    return r;
  }
  /* Export Image's dialog (plan E9): a pixel or physical size at a DPI, the same as the 3D views */
  function exportImage() {
    if (!doc || !H.exportImage) return;
    const w = el.clientWidth, h = el.clientHeight;
    H.exportImage({ stem: stemOf(doc), format: "png", plot: { size: [w, h],
      renderPng: (W) => window.Plotly.toImage(el, { format: "png", width: w, height: h, scale: W / w }),
      renderSvg: async () => decodeURIComponent((await window.Plotly.toImage(el, { format: "svg", width: w, height: h })).replace(/^data:image\/svg\+xml,/, "")) } });
  }
  /** The plotted data as {columns, rows}, in the plotted units (CSV, Copy as TSV). */
  function tableOf(d) {
    const rows = [];
    if (d.kind === "map") {
      const m = d.data.map.maps[d.opts.which || 0];
      m.values.forEach((col, j) => col.forEach((v, i) => rows.push([m.x[i], m.y[j], v])));
      return { columns: ["x_A", "y_A", "value"], rows };
    }
    if (d.kind === "convergence") {
      for (const r of d.data.convergence.runs) r.energies.forEach((e, i) => rows.push([r.name, i + 1, e ?? "", r.n_atoms, r.status || ""]));
      return { columns: ["run", "geometry", "energy_au", "n_atoms", "status"], rows };
    }
    if (d.kind === "transport") {
      // one block per temperature, all with the same columns (the first block's), in the plot's units; the carriers
      // per cell (and per cm³ for a 3D run); ZT with κ_L its ZTₑ bound too
      const t = d.data.transport, cols = ["avg", ...t.components], name = t.quantity.toLowerCase();
      const bound = t.blocks.some((b) => b.bound), vol = t.dim === 3 && t.blocks[0].volume_cm3, tdf = t.x === "energy";
      for (const b of t.blocks) b.mu.forEach((m, i) => rows.push([m, ...(tdf ? [] : [b.T, b.carriers[i], ...(vol ? [b.carriers[i] / b.volume_cm3] : [])]),
        ...cols.map((c) => (b[c] ? b[c][i] ?? "" : "")), ...(bound ? cols.map((c) => (b.bound && b.bound[c] ? b.bound[c][i] ?? "" : "")) : [])]));
      // the unit in its plain name (plot_labels.UNITS: uV/K, W/(m K)), a dot for a space: the header stays ASCII
      const u = t.unit_name ? `_${t.unit_name.replace(/ /g, ".")}` : "";
      return { columns: [tdf ? "energy_eV" : "mu_eV", ...(tdf ? [] : ["T_K", "carriers_per_cell", ...(vol ? ["carriers_cm-3"] : [])]),
        ...cols.map((c) => `${name}_${c}${u}`), ...(bound ? cols.map((c) => `zt_electronic_${c}`) : [])], rows };
    }
    if (isSpec(d)) {
      const s = d.data.spectra[0], cols = Object.keys(s.columns), fx = PK.WAVENUMBER_UNITS[d.opts.unit] || 1;
      const u = (PK.UNIT_LABEL[d.opts.unit] || "cm⁻¹").replace("⁻¹", "-1");
      const wl = s.wavelength_nm;          // IRSPEC.DAT's second column
      s.x.forEach((x, i) => rows.push([x * fx, ...(wl ? [wl[i]] : []), ...cols.map((c) => s.columns[c][i])]));
      return { columns: [`${d.opts.unit === "cm-1" ? "wavenumber" : "energy"}_${u}`, ...(wl ? ["wavelength_nm"] : []), ...cols], rows };
    }
    const { dos } = d.data, band = bandView(d), ph = isPhonon(d), shift = ph ? 0 : d.opts.shift || 0;
    const pu = ph ? (PK.PHONON_UNITS[d.opts.unit] ? d.opts.unit : "cm-1") : PK.ENERGY_UNITS[d.opts.unit] ? d.opts.unit : "eV";
    const f = (ph ? PK.PHONON_UNITS : PK.ENERGY_UNITS)[pu] || 1, u = pu || "eV";
    if (band) {
      const cols = ["path", ...band.up.map((_, k) => `${ph ? "mode" : "band"}${k + 1}${band.spin_polarized ? "_up" : ""}_${u}`), ...(band.down || []).map((_, k) => `band${k + 1}_down_${u}`)];
      band.x.forEach((x, i) => { if (x !== null) rows.push([x, ...band.up.map((b) => (b[i] - shift) * f), ...(band.down || []).map((b) => (b[i] - shift) * f)]); });
      if (!dos) return { columns: cols, rows };
      return { columns: cols, rows };
    }
    const cols = [`${ph ? "frequency" : "energy"}_${u}`, ...dos.projections.flatMap((p) => (dos.spin_polarized ? [`${p.label}_up`, `${p.label}_down`] : [p.label]))];
    dos.energy.forEach((e, i) => rows.push([(e - shift) * f, ...dos.projections.flatMap((p) => (dos.spin_polarized ? [p.up[i], p.down[i]] : [p.up[i]]))]));
    return { columns: cols, rows };
  }

  /** "Broadened from 54 IR-active modes (Lorentzian, FWHM 8 cm⁻¹, CRYSTAL's default)" */
  function sourceNote(d) {
    const s = d.data.spectra[0];
    const why = s.dampfac ? "the run's DAMPFAC" : s.fwhm === 8 ? "CRYSTAL's default" : "set here";
    return `Broadened from ${s.n_active} ${d.kind === "ir" ? "IR" : "Raman"}-active modes (${esc(s.shape_label)}, FWHM ${s.fwhm} cm⁻¹, ${why})`;
  }
  /* the bands more than ±25 eV from E_F are not loaded: said when the window (or the Fermi shift) reaches past
     them, and again whenever it changes (mace plotting and Copy CLI Command draw them) */
  function beyondNote() {
    const d = doc, span = $(".pv-note .pv-beyond");
    if (!d || !span) return;
    const band = d.data && d.data.band;
    let text = "";
    if (band && band.n_beyond > 0 && !isPhonon(d)) {
      const W = Number(band.window_ev) || 25, sh = Number(d.opts.shift) || 0;
      if (Number(d.opts.emin) + sh < -W || Number(d.opts.emax) + sh > W)
        text = ` · ${band.n_beyond} band${band.n_beyond === 1 ? "" : "s"} more than ${W} eV from E<sub>F</sub> not loaded (mace plotting draws them)`;
    }
    // CRYSTAL wrote no Fermi level (a spin-locked SCF): where the plot's E_F is from (mace.utils.fermi_level)
    if (band && band.fermi_note && !isPhonon(d)) text = ` · <span class="pv-fermi">${esc(band.fermi_note)}</span>` + text;
    if (span.innerHTML !== text) span.innerHTML = text;
  }
  function note(d) {
    if (d.source !== "modes" || !d.data) return;
    const line = sourceNote(d) + " · click a peak to animate its mode", tip = esc(`${line.replace(/<[^>]+>/g, "")}. Drag to zoom, double-click to reset; Mod+Z undoes an option.`);
    $(".pv-note").classList.add("one");
    $(".pv-note").innerHTML = `<span class="pv-note-line">${line}</span><span class="pv-help" data-help tabindex="0" role="img" title="${tip}" aria-label="${tip}">${icon("circle-help")}</span>`;
  }
  function rebuild() { if (doc && doc.data && el.offsetWidth > 0) { el.__sig = null; redraw(); } }
  document.addEventListener("mace-theme", rebuild);

  return {
    async open(d) {
      doc = d;
      $(".pv-img").hidden = true; $(".pv-frame").hidden = true; el.hidden = false; $(".pv-acts").hidden = false;
      setTitle(d.title);
      $(".pv-readout").textContent = "";
      load(d);
      renderOptions();
      if (kit) kit.refresh();
      $(".pv-note").classList.remove("one");
      $(".pv-note").innerHTML = `<span class="spinner"></span> Reading ${esc(d.title)}…`;
      try {
        await fetchData(d);
        if (doc !== d) return;
        // a phonon fort.25 reads in cm⁻¹ unless the plot's own unit is one of the frequency units; electronic
        // data (another run picked, or the domain switched back) leaves a frequency unit for eV
        if (isPhonon(d) && !PK.PHONON_UNITS[d.opts.unit]) d.opts.unit = "cm-1";
        else if (!isPhonon(d) && d.data && (d.data.band || d.data.dos) && !PK.ENERGY_UNITS[d.opts.unit]) d.opts.unit = "eV";
        const { band, dos } = d.data || {};
        const notes = [], help = [];
        if (d.data && d.data.map) {
          notes.push(`${d.data.map.maps.length} map${d.data.map.maps.length === 1 ? "" : "s"} from CRYSTAL's fort.25 (MAPN), the plane through A, B and C`);
        } else if (d.data && d.data.convergence) {
          notes.push(`${d.data.convergence.runs.length} runs: energy per atom at each geometry, from the outputs' own energies, above the lowest final energy of the set`);
        } else if (d.data && d.data.transport) {
          const t = d.data.transport;
          notes.push(trNote(t));
          help.push(...notes, "Plot options › Quantity: the run's files beside this one and what they give (ZT, power factor, Lorenz number, ...)");
        } else if (band || dos) {
          const files = [band && band.name, dos && dos.name].filter(Boolean);
          setTitle(files.join("  +  "));
          // one short line; the longer guidance is in the help icon's tooltip
          const meV = band && Math.max(1, Math.round(band.spin_max_diff * 1000));
          const sw = BF.dashWord(envOf(d.opts).look.spin_down_style || "dash");
          // phonons: frequencies, no spin and no Fermi level
          if (isPhonon(d)) notes.push(band ? `${band.up.length} phonon branches` : "phonon DOS");
          else notes.push(band ? (band.spin_polarized ? `spin-polarized (α and β up to ${meV} meV apart): α solid, β <span class="pv-bword">${sw}</span>`
            : band.spin_identical ? `spin-polarized · α = β (non-magnetic, ≤${meV} meV)` : "closed shell")
            : dos.spin_polarized ? "spin-polarized" : dos.spin_summed ? "spin-polarized · α = β (non-magnetic): total α + β drawn" : "closed shell");
          if (band && band.truncated) notes.push(`the ${band.up.length} bands nearest E<sub>F</sub> of ${band.n_bands}`);
          // the bands more than ±25 eV from E_F are not loaded: say so when the window reaches past them
          // (mace plotting and Copy CLI Command draw them)
          // (kept up to date when the window or the Fermi shift changes: beyondNote, in redraw)
          if (!isPhonon(d)) notes.push((band || dos).reference === "vacuum" ? "vacuum scale · dotted = E<sub>F</sub>, drag to shift" : "dotted = Fermi level, drag to shift");
          if (band && band.spin_identical) help.push(`α and β agree within ${meV} meV, so one set of bands is drawn`);
          if (dos && dos.spin_polarized) help.push("DOS: ↑ to the right, ↓ mirrored to the left (total hatched)");
          if (dos && dos.window) help.push(`DOS computed from ${dos.window[0].toFixed(2)} to ${dos.window[1].toFixed(2)} eV${dos.edge_pileup ? " (its edge bin, the states beyond it, is left out)" : ""}`);
          // a band-index DOS (a 1D material's default) that stops below E_F + 15 eV (mace.utils.doss_reach)
          if (dos && dos.reach_note) {
            notes.push(`DOS ends at E<sub>F</sub>${dos.window ? `+${dos.window[1].toFixed(1)}` : ""} eV (bands by index; the help says how to reach higher)`);
            help.unshift(dos.reach_note);
          }
          if ((band || dos).reference === "vacuum") help.push("energies on the vacuum scale (POTC run found); the dotted line is E_F");
          if (!isPhonon(d)) help.push("dragging the dotted line shifts the energy reference");
        } else if (d.data && d.data.spectra) {
          const s = d.data.spectra[0];
          if (d.source === "modes") notes.push(sourceNote(d));
          else notes.push(`CRYSTAL ${d.kind === "ir" ? "IRSPEC" : "RAMSPEC"}.DAT · ${s.n_points.toLocaleString("en-US")} points, read with mace plotting's reader${s.broadening ? `; broadening ${esc(s.broadening)}` : ""}`);
          if (d.data.out && d.data.peaks.length) notes.push("click a peak to animate its mode");
        }
        const drag = "Drag to zoom, double-click to reset; Mod+Z undoes an option.";
        // one line for every plot's subtitle: what does not fit is in the help icon's tooltip, with the guidance
        if (!help.length) help.push(...notes);
        const tip = esc(L.plain(`${help.join(" · ")}. ${drag}`));        // a title attribute: the markup as text
        $(".pv-note").classList.add("one");
        $(".pv-note").innerHTML = `<span class="pv-note-line">${notes.join(" · ")}<span class="pv-beyond"></span></span>`
          + `<span class="pv-help" data-help tabindex="0" role="img" title="${tip}" aria-label="${tip}">${icon("circle-help")}</span>`;
        renderOptions();
        if (kit) kit.refresh();           // a band or DOS plot's categories (Style, Traces, Reference, Labels) follow its data
        await redraw();
      } catch (e) {
        if (e.data && e.data.still_writing) {
          // a job is still writing the file: not an error; Retry reads it again
          $(".pv-note").innerHTML = `${esc(e.message)} <button class="tb-btn-text" data-act="retry-read">Retry</button>`;
          const b = $(".pv-note [data-act=retry-read]");
          b.addEventListener("click", () => { if (doc === d) { d.data = null; this.open(d); } }, { once: true });
          return;
        }
        $(".pv-note").textContent = e.message;
      }
    },
    /** An image or HTML plot already in the folder. */
    openImage(d, kind) {
      doc = null;
      el.hidden = true; $(".pv-acts").hidden = true; $(".pv-readout").textContent = ""; $(".pv-chips").innerHTML = "";
      setTitle(d.title);
      $(".pv-note").classList.remove("one");
      $(".pv-note").textContent = kind === "html" ? "An HTML plot in this folder, shown in a sandbox (its scripts cannot reach MACE)" : "An image in this folder";
      const src = "api/raw?" + q({ path: d.path });
      if (kind === "html") { const f = $(".pv-frame"); f.hidden = false; $(".pv-img").hidden = true; if (f.getAttribute("src") !== src) f.src = src; }
      else { const i = $(".pv-img"); i.hidden = false; $(".pv-frame").hidden = true; i.src = src; i.alt = d.title; }
      optsHost.innerHTML = `<p class="insp-hint">No options for a picture.</p>`;
    },
    close() { doc = null; },
    reset: () => doc && reset(doc), undo: () => doc && undo(doc), redo: () => doc && redo(doc),
    bz: () => doc && doc.bz ? doc.bz.state() : null,
    state: () => doc && { kind: doc.kind, source: doc.source || "file", opts: { ...doc.opts }, pos: doc.pos, hist: doc.hist.length, data: !!doc.data,
      drawn: el.dataset.drawn === doc.path && !building, peaks: doc.data && doc.data.peaks ? doc.data.peaks.length : 0,
      out: doc.data ? doc.data.out || null : null, fwhm: doc.data && doc.data.spectra ? doc.data.spectra[0].fwhm || null : null,
      transport: doc.kind === "transport" && doc.data && doc.data.transport ? { quantity: doc.data.transport.quantity, available: doc.data.transport.available,
        error: doc.data.trError, upper_bound: doc.data.transport.upper_bound, tau_fs: doc.data.transport.tau_fs, tau_source: doc.data.transport.tau_source } : null,
      zt: doc.kind === "transport" && doc.data && doc.data.transport && doc.data.transport.quantity === "zt" ? { key: doc.data.trKey, error: doc.data.trError,
        refused: doc.data.transport.refused, zt_e_max: doc.data.transport.zt_e_max, zt_total_max: doc.data.transport.zt_total_max,
        tau_fs: doc.data.transport.tau_fs, kappa_l_source: doc.data.transport.kappa_l_source } : null },
    figure: () => (el.data ? { traces: el.data.map((t) => ({ name: t.name, meta: t.meta, n: (t.x || []).length })), layout: el.layout } : null),
    setOpt: (patch, label) => doc && setOpt(doc, patch, label),
    /** Publication Figure… (mace plotting's SVG, downloaded); the task's result. */
    publication: () => (doc ? publication() : Promise.resolve(null)),
    /** Copy CLI Command's text (clicmd.js), or null when no mace plotting command draws this plot. */
    cliText: () => { try { return doc ? cliText() : null; } catch (e) { return null; } },
    el: () => el,
  };
}
