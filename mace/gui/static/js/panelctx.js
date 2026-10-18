/* The panel context (1.1.6): what the side panels refer to.
 *
 * Every document-scoped panel (the Inspector; the Atoms, Measurements, History,
 * Timeline, Modes and Coordination tabs) shows the document in focus in the
 * center (in a split, each editor group's page has its own panels and its own
 * context) and never what an earlier document left there. A panel with nothing
 * for that document shows one neutral placeholder that says why; its tab is
 * dimmed in the strip, never hidden. A plot, spectrum, volume or properties file
 * with no atoms of its own borrows the structure of the run it came from
 * (/api/related): read-only, and labeled with where it came from. Tasks, Jobs
 * and Problems are global and say so.
 *
 * The shell registers each panel once (createPanelContext().register) and calls
 * apply() after it renders them; the per-document scroll positions are kept
 * here and put back when a document is focused again. */

const base = (p) => String(p || "").split("/").pop();

/* What a document is, in a sentence: "NaCl.cif is a structure". */
const PLOT_NOUN = { band: "a band structure plot", dos: "a DOS plot", ir: "an IR spectrum", raman: "a Raman spectrum",
  f25: "a 2D map", transport: "a transport plot", convergence: "a convergence plot", image: "an image" };
export function nounOf(d) {
  if (!d) return "";
  if (d.kind === "plot") return PLOT_NOUN[d.fileKind] || (d.plot && PLOT_NOUN[d.plot.kind]) || "a plot";
  return ({ structure: "a structure", trajectory: "an optimization run", runplot: "a plot of a run", volume: "a volume",
    freq: "a normal-mode view", db: "the database", workflows: "the workflow list", gallery: "the plot gallery",
    text: "a text file", cluster: "the cluster view", job: "a job's details", batch: "Batch Setup",
    editor: "the classic editor" })[d.kind] || "a document";
}

/* A read-only structure borrowed from the run a document came from: the Atoms
 * table, Coordination and the Inspector read it as they read a run's frame. */
class BorrowedFrame {
  constructor(d, rel, coordMode) {
    this.doc = d; this.readOnly = true; this.borrowed = true;
    this.selection = new Set(); this.measurements = []; this.coordMode = coordMode;
    this.structure = rel.structure; this.source = rel.source; this.via = rel.via; this.projections = rel.projections || null;
  }
  get n() { return this.structure && this.structure.numbers ? this.structure.numbers.length : 0; }
  editBlock() { return `Read-only: these atoms are from ${base(this.source)}. Open ${base(this.source)} to edit them`; }
  get label() { return `From ${base(this.source)}`; }
}

/* A dimmed tab's reason for a screen reader: visually hidden text named by aria-describedby
   (aria-description is an ARIA 1.3 draft that the a11y gate refuses and older readers ignore). */
let descN = 0;
function describe(tab, text) {
  let box = document.getElementById("pc-why");
  if (!box) { box = document.createElement("div"); box.id = "pc-why"; box.className = "sr-only"; document.body.appendChild(box); }
  if (!tab.dataset.pcWhy) tab.dataset.pcWhy = `pc-why-${++descN}`;
  const id = tab.dataset.pcWhy;
  let span = document.getElementById(id);
  if (!text) {
    if (span) span.remove();
    const rest = (tab.getAttribute("aria-describedby") || "").split(/\s+/).filter((x) => x && x !== id);
    if (rest.length) tab.setAttribute("aria-describedby", rest.join(" ")); else tab.removeAttribute("aria-describedby");
    return;
  }
  if (!span) { span = document.createElement("span"); span.id = id; box.appendChild(span); }
  span.textContent = text;
  const ids = (tab.getAttribute("aria-describedby") || "").split(/\s+/).filter(Boolean);
  if (!ids.includes(id)) tab.setAttribute("aria-describedby", [...ids, id].join(" "));
}

export function createPanelContext(H) {
  // H: { esc, icon, api(route) -> Promise, own(d) -> view|null, wantsRelated(d) -> path|null, changed(d),
  //      openPath(path), activeTab() -> key, coordMode() -> "frac"|"cart" }
  const { esc, icon } = H;
  const panels = new Map();
  const rels = new WeakMap();            // doc -> { state, data, pick, frame }
  const scrolls = new Map();             // doc id -> Map(container key -> scrollTop)
  let cur = null, settleUntil = 0, lastApplied = null;

  /** One panel: { title, scope: "doc"|"global", note (global: what it covers), host(), tab(), applies(s, d), thing, what, borrow }. */
  function register(key, spec) { panels.set(key, { key, scope: "doc", borrow: false, ...spec }); }

  /* ---------------------------------------------------------------- the related run */
  function relOf(d) {
    if (!d) return null;
    const p = H.wantsRelated(d);
    if (!p) return null;
    let r = rels.get(d);
    if (!r || r.path !== p) {
      r = { path: p, state: "loading", data: null, pick: null, frame: null };
      rels.set(d, r);
      fetchRel(d, r);
    }
    return r;
  }
  function fetchRel(d, r) {
    const q = `related?path=${encodeURIComponent(r.path)}` + (r.pick ? `&pick=${encodeURIComponent(r.pick)}` : "");
    r.state = "loading"; r.frame = null;
    H.api(q).then((data) => {
      if (rels.get(d) !== r) return;
      r.data = data; r.state = data && data.state === "found" && data.structure ? "found" : (data && data.state) || "none";
      H.changed(d);
    }).catch((e) => {
      if (rels.get(d) !== r) return;
      r.state = "none"; r.data = { error: e.message };
      H.changed(d);
    });
  }
  function pick(d, path) {
    const r = rels.get(d);
    if (!r) return;
    r.pick = path;
    fetchRel(d, r);
    apply(d, true);
  }
  /** The borrowed frame of a document (null when it has its own structure, or none was found). */
  function borrowed(d) {
    if (!d || H.own(d)) return null;
    const r = relOf(d);
    if (!r || r.state !== "found") return null;
    if (!r.frame) r.frame = new BorrowedFrame(d, r.data, H.coordMode());
    return r.frame;
  }

  /* ---------------------------------------------------------------- what each panel shows */
  /** The panel context of a document: { doc, own, frame (own or borrowed view), rel }. */
  function subject(d) {
    const own = d ? H.own(d) : null;
    const rel = d && !own ? relOf(d) : null;
    return { doc: d || null, own, frame: own || borrowed(d), rel };
  }
  function applies(p, s) {
    if (p.scope !== "doc") return true;
    return !!p.applies(s);
  }
  /** The neutral placeholder's content for a panel with nothing for this document. */
  function emptyContent(p, s) {
    const d = s.doc, r = s.rel;
    const what = `${p.title === "Inspector" ? "The Inspector" : `The ${p.title} tab`} applies to ${p.what}.`;
    const acts = [];
    if (!d) return { head: `No ${p.thing}: no document is open.`, sub: what, acts };
    const src = r && r.state === "found" && r.data ? r.data.source : null;
    if (p.borrow && r && r.state === "loading") return { head: `Finding the run ${d.title} came from…`, sub: what, acts, busy: true };
    if (p.borrow && r && r.state === "ambiguous") {
      const c = (r.data && r.data.candidates) || [];
      c.forEach((x) => acts.push({ label: `Show ${base(x.path)}`, run: () => pick(d, x.path), tip: x.path }));
      return { head: `${d.title} could come from ${c.length} runs.`, sub: "Pick the one to show here, read-only.", acts };
    }
    if (src) acts.push({ label: `Open ${base(src)}`, run: () => H.openPath(src), tip: `Open the structure of the run ${d.title} came from` });
    let head = `No ${p.thing}: ${d.title} is ${nounOf(d)}.`;
    if (p.borrow && r && r.state === "none") head = `No ${p.thing}: no run found for ${d.title} (${nounOf(d)}).`;
    return { head, sub: what, acts };
  }
  function placeholderEl(p, s) {
    const c = emptyContent(p, s);
    const el = document.createElement("div");
    el.className = "pc-ph";
    el.setAttribute("role", "status");
    el.setAttribute("aria-live", "polite");
    el.tabIndex = 0;
    el.dataset.panel = p.key;
    el.innerHTML = `${icon(c.busy ? "scan-search" : "info", "i pc-ph-i")}<div class="pc-ph-b"><p class="pc-ph-t"></p><p class="pc-ph-s"></p>` +
      (c.acts.length ? `<div class="pc-ph-acts"></div>` : "") + `</div>`;
    el.querySelector(".pc-ph-t").textContent = c.head;
    el.querySelector(".pc-ph-s").textContent = c.sub;
    const box = el.querySelector(".pc-ph-acts");
    c.acts.forEach((a) => {
      const b = document.createElement("button");
      b.type = "button"; b.className = "tb-btn-text"; b.textContent = a.label;
      if (a.tip) b.title = a.tip;
      b.addEventListener("click", a.run);
      box.appendChild(b);
    });
    return el;
  }
  /** The Inspector's placeholder (the shell puts it where the sections go). */
  function inspectorPlaceholder(d) {
    const p = panels.get("inspector");
    return p ? placeholderEl(p, subject(d)) : document.createElement("div");
  }

  /* The label a panel's header carries: "Atoms · NaCl.cif", "Atoms · from mgo_doss.out (a DOS plot mgo_doss.DOSS.DAT)", "Tasks · all documents". */
  function headerText(p, s) {
    if (p.scope !== "doc") return `${p.title} · ${p.note}`;
    const d = s.doc;
    if (!d) return `${p.title} · no document`;
    if (p.borrow && !s.own && s.frame && s.frame.borrowed) return `${p.title} · from ${base(s.frame.source)} (${nounOf(d)} ${d.title})`;
    return `${p.title} · ${d.title}`;
  }

  /* ---------------------------------------------------------------- apply */
  function ensureLabel(strip) {
    let el = strip.querySelector(".pc-doc");
    if (!el) {
      el = document.createElement("span");
      el.className = "pc-doc";
      strip.appendChild(el);
    }
    return el;
  }
  /** Bring every registered panel in line with document d (the shell calls it after rendering them). */
  function apply(d, force = false) {
    const s = subject(d);
    const switched = cur !== d;
    if (switched) { cur = d; settleUntil = performance.now() + 400; }
    for (const p of panels.values()) {
      if (p.key === "inspector") continue;        // the shell renders the Inspector's placeholder in place
      const host = p.host && p.host();
      if (!host) continue;
      const on = applies(p, s);
      host.classList.toggle("pc-empty", !on);
      let ph = host.querySelector(":scope > .pc-ph");
      if (!on) {
        const fresh = placeholderEl(p, s);
        if (!ph) host.prepend(fresh);
        else if (force || switched || ph.textContent !== fresh.textContent) ph.replaceWith(fresh);
      } else if (ph) ph.remove();
      const tab = p.tab && p.tab();
      if (tab) {
        tab.classList.toggle("pc-off", !on);
        describe(tab, on ? "" : d ? `Nothing here for ${d.title}` : "No document is open");
      }
    }
    header(s);
    labelInspector(d);
    if (switched || force) restore(d);
    lastApplied = s;
    return s;
  }
  /** "Inspector · NaCl.cif": the Inspector's header names the document it shows. */
  function labelInspector(d = cur) {
    const insp = document.querySelector("#inspector .insp-bar");
    if (!insp) return;
    let lab = insp.querySelector(".pc-doc");
    if (!lab) {
      lab = document.createElement("span");
      lab.className = "pc-doc";
      const t = insp.querySelector(".insp-bar-title");
      if (t) t.after(lab); else insp.prepend(lab);
    }
    lab.textContent = d ? `· ${d.title}` : "· no document";
    lab.title = d ? `The Inspector shows ${d.title}` : "";
  }
  /** The bottom panel's label for its active tab. */
  function header(s = subject(cur)) {
    const strip = document.getElementById("bottom-tabs");
    if (!strip) return;
    const p = panels.get(H.activeTab());
    const el = ensureLabel(strip);
    el.textContent = p ? headerText(p, s) : "";
    el.title = el.textContent;
  }

  /* ---------------------------------------------------------------- per-document scroll */
  const KEYS = [["inspector", () => document.getElementById("inspector")],
    ["bottom", () => document.getElementById("bottom-body")],
    ["atoms-table", () => document.querySelector(".bottom-pane.atoms .at-scroll")]];
  function keyOf(el) {
    if (!el || !el.id && !el.classList) return null;
    if (el.id === "inspector") return "inspector";
    if (el.id === "bottom-body") return `bottom:${H.activeTab()}`;
    if (el.classList && el.classList.contains("at-scroll")) return "atoms-table";
    return null;
  }
  document.addEventListener("scroll", (e) => {
    if (!cur || performance.now() < settleUntil) return;
    const k = keyOf(e.target);
    if (!k) return;
    if (!scrolls.has(cur.id)) scrolls.set(cur.id, new Map());
    scrolls.get(cur.id).set(k, e.target.scrollTop);
  }, true);
  function restore(d) {
    if (!d) return;
    const saved = scrolls.get(d.id);
    const put = () => {
      for (const [k, get] of KEYS) {
        const el = get();
        if (!el) continue;
        const key = k === "bottom" ? `bottom:${H.activeTab()}` : k;
        const v = saved ? saved.get(key) : undefined;
        if (v !== undefined && el.scrollTop !== v) { el.scrollTop = v; el.dispatchEvent(new Event("scroll")); }
      }
    };
    requestAnimationFrame(() => requestAnimationFrame(() => { if (cur === d) put(); }));
    setTimeout(() => { if (cur === d) put(); }, 250);       // the Atoms table mounts once the editor module is in
  }

  /* ---------------------------------------------------------------- borrowed structure, shown */
  /** The read-only banner over the Atoms table of a borrowed structure (null removes it). */
  function banner(host, f) {
    let b = host.querySelector(":scope > .pc-from");
    if (!f) { if (b) b.remove(); return; }
    if (!b) {
      b = document.createElement("div");
      b.className = "pc-from";
      b.setAttribute("role", "note");
      host.prepend(b);
    }
    const key = `${f.source}|${f.doc.id}`;
    if (b.dataset.key === key) return;
    b.dataset.key = key;
    const proj = (f.projections || []).filter((x) => x.atoms && x.atoms.length);
    const byEl = new Map();
    proj.forEach((x) => { if (!byEl.has(x.element)) byEl.set(x.element, { orb: [], atoms: x.atoms }); byEl.get(x.element).orb.push(x.orbital); });
    const atomsText = (a) => (a.length > 6 ? `${a.length} atoms` : `atom${a.length === 1 ? "" : "s"} ${a.map((i) => i + 1).join(", ")}`);
    b.innerHTML = `<span class="pc-from-t">${icon("link-2", "i")}<span></span></span>` +
      `<button type="button" class="tb-btn-text pc-from-open"></button>` +
      (byEl.size ? `<span class="pc-from-proj"></span>` : "");
    b.querySelector(".pc-from-t > span").textContent = `From ${base(f.source)}, ${f.via || "the run it came from"}: read-only.`;
    b.querySelector(".pc-from-t").title = f.source;
    const open = b.querySelector(".pc-from-open");
    open.textContent = `Open ${base(f.source)}`;
    open.addEventListener("click", () => H.openPath(f.source));
    const pj = b.querySelector(".pc-from-proj");
    if (pj) pj.textContent = "Projections: " + [...byEl].map(([el, v]) => `${el} (${v.orb.join(", ")}) on ${atomsText(v.atoms)}`).join(" · ");
  }
  /** The Inspector's read-only Structure and Cell sections of a borrowed structure: [[key, title, el, caption]]. */
  function inspectorSections(d) {
    const f = borrowed(d);
    if (!f) return [];
    const s = f.structure, L = s.lattice || {}, y = s.symmetry || {};
    const num = (v, k = 4) => (Number.isFinite(v) ? Number(v).toFixed(k) : "–");
    const st = document.createElement("div");
    st.className = "pc-ro";
    st.innerHTML = `<table class="kv"><tr><td>Formula</td><td></td></tr><tr><td>Atoms</td><td>${f.n}</td></tr>` +
      `<tr><td>Periodicity</td><td>${esc(s.dimensionality || "")}</td></tr>` +
      (y.available ? `<tr><td>Space group</td><td>${esc(y.international || "")} (No. ${esc(String(y.number))})</td></tr>` : "") +
      `<tr><td>From</td><td class="pc-ro-src"></td></tr></table>` +
      `<p class="insp-hint">Read-only: the structure of the run ${esc(d.title)} came from (${esc(f.via || "")}).</p>` +
      `<div class="ins-row end"><button type="button" class="tb-btn-text pc-ro-open"></button></div>`;
    st.querySelector("tr td + td").textContent = s.formula || s.reduced_formula || "";
    st.querySelector(".pc-ro-src").textContent = f.source;
    const ob = st.querySelector(".pc-ro-open");
    ob.textContent = `Open ${base(f.source)}`;
    ob.addEventListener("click", () => H.openPath(f.source));
    const cell = document.createElement("div");
    cell.className = "pc-ro";
    cell.innerHTML = `<table class="kv">` + [["a", "a", "Å", 4], ["b", "b", "Å", 4], ["c", "c", "Å", 4], ["alpha", "α", "°", 3], ["beta", "β", "°", 3], ["gamma", "γ", "°", 3]]
      .map(([k, t, u, n]) => `<tr><td>${t}</td><td class="num">${num(L[k], n)} ${u}</td></tr>`).join("") +
      `<tr><td>Volume</td><td class="num">${num(L.volume, 3)} Å³</td></tr></table>`;
    const cap = `from ${base(f.source)}`;
    return [["structure", "Structure", st, cap], ["cell", "Cell", cell, cap]];
  }

  return { register, apply, header, labelInspector, subject, borrowed, banner, inspectorSections, inspectorPlaceholder, pick, nounOf,
    state: () => {
      const s = lastApplied || subject(cur);
      const out = { doc: cur ? cur.title : null, own: !!s.own, borrowed: s.frame && s.frame.borrowed ? s.frame.source : null,
        related: s.rel ? s.rel.state : null, id: cur ? cur.id : null,
        scroll: cur && scrolls.get(cur.id) ? Object.fromEntries(scrolls.get(cur.id)) : {}, candidates: s.rel && s.rel.data ? (s.rel.data.candidates || []).map((c) => c.path) : [], panels: {} };
      for (const p of panels.values()) out.panels[p.key] = { scope: p.scope, applies: applies(p, s), header: headerText(p, s) };
      return out;
    } };
}
