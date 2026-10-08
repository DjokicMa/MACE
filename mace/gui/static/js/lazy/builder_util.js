/* The builders' shared frame (plan T3b, B1, B10, B11): every builder dialog
 * (Supercell, Slab, Point Defect, Transform Cell, Symmetrize) is this frame
 * plus its own parameters.
 *  - a non-modal panel (§2.5: the viewport stays usable), one per document;
 *  - Result: New tab | This structure, remembered per builder; This structure
 *    stops at 5000 atoms (the editor's limit) and never applies to a slab;
 *  - the name of the new document, checked against the server's rule;
 *  - a live preview in the viewport: debounced 150 ms, the previous request
 *    aborted, outline only above 2000 atoms;
 *  - the preview lifecycle: editing is off ("Accept or discard the preview
 *    first"), Mod+Z discards, Enter applies, Esc closes, the preview stays with
 *    its document across tab switches, closing discards;
 *  - a build that takes longer than 300 ms: "Building… 45%" with Cancel in the
 *    dialog; closing the dialog hands it to the Tasks panel. */
import { panel, loadCss } from "./lazy_util.js";

const css = Promise.all([loadCss(import.meta.url, "../../css/dialogs.css"), loadCss(import.meta.url, "../../css/builders.css")]);
export const EDIT_MAX = 5000, PREVIEW_MAX = 2000;
const NAME_RE = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,60}$/;
const resultKey = (id) => `mace.gui.builder.${id}.result`;
const load = (k) => { try { return localStorage.getItem(k); } catch (e) { return null; } };
const store = (k, v) => { try { localStorage.setItem(k, v); } catch (e) { /* private window */ } };
export const fmtN = (n) => Number(n).toLocaleString("en-US");
export const stemOf = (t) => String(t || "structure").split("/").pop().replace(/\.[^.]+$/, "").replace(/[^A-Za-z0-9_.-]+/g, "_") || "structure";

/** Open a builder on the active structure document. spec: {id, title, route,
 * results: ["new", "inplace"] | ["new"], defaultResult, primary, render(host, api),
 * params(api) -> {params, error?, n?, cell?, name, label, readout?, inPlace?: {op, args, label}},
 * drawPreview?(resp, api) -> preview object for the scene, onOpen?(api), onClose?(api)} */
export async function openBuilder(H, spec) {
  const S = H.shell(), d = S.activeDoc();
  if (!d || d.kind !== "structure") { H.toast(`${spec.title}: Needs a structure`, "info"); return null; }
  if (d.builderFrame) {
    if (d.builderFrame.spec.id === spec.id) { d.builderFrame.focus(); return d.builderFrame; }
    d.builderFrame.discard("replaced");
  }
  await css;
  const { E, ed } = H.editor();
  return new Frame(H, S, E, d, ed, spec);
}

class Frame {
  constructor(H, S, E, d, ed, spec) {
    Object.assign(this, { H, S, E, d, ed, spec });
    this.results = spec.results || ["new", "inplace"];
    const saved = load(resultKey(spec.id));
    this.result = this.results.includes(saved) ? saved : (spec.defaultResult || this.results[0]);
    this.nameTouched = false;
    this.seq = 0;
    this.abort = null;
    this.timer = 0;
    this.busy = null;              // {task, pct} while a detached build runs
    this.lastPreview = null;
    this.closed = false;
    this.hidden = false;
    this.api = {
      H, S, E, d, ed, spec,
      structure: () => (ed ? ed.structure : null),
      changed: () => this.changed(),
      setReadout: (html) => { this.readout.innerHTML = html; },
      setError: (msg) => this.setError(msg),
      setPreview: (p) => this.drawPreview(p),
      run: () => this.go(),
      frame: this,
    };
    this.build();
    d.builderFrame = this;
    if (ed) ed.previewing = { kind: spec.id };
    this.keysFn = (e) => this.onKey(e);
    S.setPreviewKeys(this.keysFn);
    this.onSwitch = (ev) => this.switched(ev.detail);
    document.addEventListener("mace-doc-switch", this.onSwitch);
    S.inspectorChanged();
    H.refresh();
    if (spec.onOpen) spec.onOpen(this.api);
    this.changed(true);
  }

  build() {
    const { H, spec } = this, esc = H.esc;
    const body = document.createElement("div");
    body.className = "bld-body";
    const seg = this.results.length === 0 ? "" : this.results.length > 1
      ? `<span class="segment bld-result" role="radiogroup" aria-label="Result">
          <button role="radio" data-r="new" aria-checked="false">New tab</button><button role="radio" data-r="inplace" aria-checked="false">This structure</button></span>`
      : `<span class="segment bld-result" role="radiogroup" aria-label="Result" data-tip="Result" data-desc="${esc(spec.onlyNewReason || "Opens as a new document")}">
          <button role="radio" data-r="new" aria-checked="true" aria-disabled="true" disabled>New tab</button></span>`;
    // what the build makes (Result, Name) sits in the body under the parameters; the footer holds only Cancel and the primary
    body.innerHTML = `<fieldset class="bld-params"></fieldset><div class="bld-readout" aria-live="polite"></div>
      <div class="bld-error" role="alert" hidden></div>
      <div class="bld-out">${seg ? `<div class="bld-row"><span>Result</span>${seg}</div>` : ""}
      <label class="bld-row bld-name"><span>Name</span><input type="text" spellcheck="false" autocomplete="off" aria-label="Name of the new document"></label></div>`;
    const foot = `<div class="bld-foot">
      <span class="bld-spacer"></span>
      <button class="tb-btn-text" data-act="cancel">Cancel</button>
      <button class="tb-btn-text primary bld-go" data-act="go"><span class="bld-fill"></span><span class="bld-go-label">${esc(spec.primary || "Create")}</span></button></div>`;
    // never reuse a closed one: its close event may still be queued, and would close this frame
    // (one left open on another document's hidden frame is that frame's: discard it first)
    const stale = document.getElementById(`builder-${spec.id}`);
    if (stale) {
      if (stale.__frame && !stale.__frame.closed) stale.__frame.discard("replaced");
      stale.remove();
    }
    this.dlg = panel(H, { id: `builder-${spec.id}`, title: spec.title, body, foot, onClose: () => this.closedByPanel() });
    this.dlg.classList.add("builder-dialog");
    this.dlg.__frame = this;
    // a closed builder leaves no stale dialog behind (a later one with this id makes its own)
    const el = this.dlg;
    el.addEventListener("close", () => setTimeout(() => { if (!el.open) el.remove(); }, 0), { once: true });
    this.dlg.dataset.builder = spec.id;
    this.params = this.dlg.querySelector(".bld-params");
    this.readout = this.dlg.querySelector(".bld-readout");
    this.error = this.dlg.querySelector(".bld-error");
    this.nameEl = this.dlg.querySelector(".bld-name input");
    this.goBtn = this.dlg.querySelector(".bld-go");
    spec.render(this.params, this.api);
    this.dlg.querySelectorAll(".bld-result [data-r]").forEach((b) => b.addEventListener("click", () => {
      if (b.getAttribute("aria-disabled") === "true") { H.toast(b.dataset.reason || "Not available", "info"); return; }
      this.result = b.dataset.r;
      store(resultKey(spec.id), this.result);
      this.changed();
    }));
    this.nameEl.addEventListener("input", () => { this.nameTouched = true; this.validateName(); });
    this.dlg.querySelector('[data-act="cancel"]').addEventListener("click", () => (this.busy ? this.cancelTask() : this.discard("cancel")));
    this.goBtn.addEventListener("click", () => this.go());
    this.params.addEventListener("input", () => this.changed());
    this.params.addEventListener("change", () => this.changed());
    // the panel's own Esc handler closes it; a field that uses Esc (an open list) keeps it
    this.dlg.onkeydown = (e) => {
      if (e.key === "Escape") {
        e.preventDefault(); e.stopPropagation();
        if (this.busy) { this.handOff(); return; }
        this.discard("esc");
      }
    };
    // the selected tab when the dialog has tabs (Point Defect opened on Interstitial), else its first field
    const first = this.params.querySelector('[role="tab"][aria-selected="true"]') || this.params.querySelector("input, select, button");
    if (first) first.focus({ preventScroll: true });
  }

  info() {
    return { id: this.spec.id, result: this.result, name: this.nameEl.value, busy: !!this.busy, hidden: this.hidden,
      preview: this.lastPreview ? { n: this.lastPreview.n, outline: !!this.lastPreview.outline_only } : null,
      error: this.error.hidden ? "" : this.error.textContent, readout: this.readout.textContent,
      goDisabled: this.goBtn.disabled, goLabel: this.goBtn.textContent.trim(), docId: this.d.id };
  }
  focus() { if (this.hidden) this.show(); const f = this.dlg.querySelector(".bld-params input, .bld-params select"); (f || this.goBtn).focus(); }

  /* ---------------------------------------------------------------- state */
  current() {
    const p = this.spec.params(this.api) || {};
    this.cur = p;
    return p;
  }
  /** kind "hint": what the user still has to choose (a site, an element), in
   * the secondary text color, not as an error */
  setError(msg, kind = "error") {
    this.error.hidden = !msg;
    this.error.textContent = msg || "";
    this.error.classList.toggle("hint", !!msg && kind === "hint");
  }
  validateName() {
    const ok = NAME_RE.test(this.nameEl.value);
    this.nameEl.setAttribute("aria-invalid", String(!ok));
    if (!ok && this.result === "new") this.setError("Name: letters, digits, _, . and - only, starting with a letter or digit");
    else if (this.error.dataset.kind === "name" || /^Name:/.test(this.error.textContent)) this.setError("");
    this.updateGo();
    return ok;
  }
  updateGo() {
    const p = this.cur || {};
    const nameOk = this.result !== "new" || this.results.length === 0 || NAME_RE.test(this.nameEl.value);
    this.goBtn.disabled = !!this.busy || !!p.error || !nameOk || p.blocked === true;
  }
  updateResult(p) {
    const n = p.n || 0, block = this.ed ? this.ed.editBlock && this.inPlaceBlock() : "No editable structure";
    const over = n > EDIT_MAX ? `In place is limited to ${fmtN(EDIT_MAX)} atoms; this makes ${fmtN(n)}` : "";
    const noOp = !p.inPlace ? "This builder makes a new document only" : "";
    const why = block || over || noOp;
    const ip = this.dlg.querySelector('.bld-result [data-r="inplace"]');
    if (ip) {
      ip.setAttribute("aria-disabled", String(!!why));
      if (why) { ip.dataset.reason = why; ip.title = why; } else { delete ip.dataset.reason; ip.removeAttribute("title"); }
    }
    let r = this.result;
    if (r === "inplace" && why) r = "new";
    this.dlg.querySelectorAll(".bld-result [data-r]").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.r === r)));
    this.effective = r;
    this.dlg.querySelector(".bld-name").hidden = r !== "new" || this.results.length === 0;
    this.goBtn.querySelector(".bld-go-label").textContent = this.busy ? this.goBtn.querySelector(".bld-go-label").textContent
      : r === "inplace" ? (this.spec.primaryInPlace || "Apply") : (this.spec.primary || "Create");
  }
  /** Why this document cannot be changed in place (other than the preview itself). */
  inPlaceBlock() {
    const ed = this.ed;
    if (!ed) return "No editable structure";
    const prev = ed.previewing;
    ed.previewing = null;
    const why = ed.editBlock();
    ed.previewing = prev;
    return why;
  }

  changed(first = false) {
    if (this.closed) return;
    const p = this.current();
    if (!this.nameTouched && p.name) this.nameEl.value = p.name;
    this.setError(p.error || "", p.blocked ? "hint" : "error");
    if (p.readout !== undefined) this.readout.innerHTML = p.readout;
    this.updateResult(p);
    if (!p.error) this.validateName(); else this.updateGo();
    this.schedulePreview(first);
  }

  /* ---------------------------------------------------------------- preview */
  schedulePreview(now = false) {
    clearTimeout(this.timer);
    if (this.spec.preview === false) return;
    // the same parameters as the last preview: nothing to ask (a readout change is not a new preview)
    const key = JSON.stringify(this.cur && this.cur.params);
    if (!now && key === this.previewKey) {
      // the drawing may still differ (one site or every site): redraw the last answer
      if (this.lastPreview && !this.busy) this.drawPreview(this.lastPreview);
      return;
    }
    this.timer = setTimeout(() => this.preview(), now ? 0 : (this.spec.debounce || 150));
  }
  async preview() {
    const p = this.cur;
    if (this.closed || this.busy || !p || p.error) { if (p && p.error) this.drawPreview(null); return; }
    this.previewKey = JSON.stringify(p.params);
    if (this.abort) this.abort.abort();
    const ctl = new AbortController(), seq = ++this.seq;
    this.abort = ctl;
    try {
      let r;
      if (this.spec.preview) r = await this.spec.preview(this.api, p, ctl.signal);       // Clean: a proposal from the pool
      else {
        const source = await this.S.sourceOf(this.d);
        r = await this.H.api(this.spec.route, { body: { source, params: p.params, preview: true }, signal: ctl.signal });
      }
      if (!r) return;
      if (seq !== this.seq || this.closed) return;
      this.lastPreview = r;
      performance.mark(`ui:preview:${this.spec.id}`);
      this.drawPreview(r);
      if (this.spec.afterPreview) this.spec.afterPreview(r, this.api);
    } catch (e) {
      if (e.name === "AbortError" || seq !== this.seq) return;
      this.previewKey = null;                       // a failed preview is asked again on the next change
      this.setError(e.message);
      this.drawPreview(null);
    }
  }
  async drawPreview(r) {
    const scene = this.E && this.E.scene && this.E.scene();
    if (!scene) return;
    if (!r || this.hidden) { scene.setPreview(null); return; }
    const core = await import("../geom/core.js");
    let p = this.spec.drawPreview ? this.spec.drawPreview(r, this.api, core) : null;
    if (!p) {
      p = { kind: this.spec.id };
      if (r.preview) p.atoms = core.decodePacked(r.preview);
      if (r.cell) p.cell = r.cell;
      if (r.origin) p.origin = r.origin;
    }
    p.doc = this.ed;
    if (this.closed || this.hidden) return;
    scene.setPreview(p);
    this.frameView(p);
    this.dlg.dataset.previewN = String(r.n || 0);
    this.dlg.querySelector(".bld-readout").dataset.outline = r.outline_only ? "1" : "";
    if (r.outline_only) {
      const note = this.readout.querySelector(".bld-outline-note") || document.createElement("p");
      note.className = "bld-outline-note insp-hint";
      note.textContent = `Preview shows the cell only above ${fmtN(PREVIEW_MAX)} atoms`;
      if (!note.isConnected) this.readout.appendChild(note);
    } else {
      const note = this.readout.querySelector(".bld-outline-note");
      if (note) note.remove();
    }
  }

  /** Keep the whole preview in sight: zoom out (never in) when the ghosts or
   * the new cell reach past the free part of the viewport; the view goes back
   * when the dialog closes, unless the user moved the camera meanwhile. */
  frameView(p) {
    if (!this.E || !this.E.fitInclude || p.dimStructure) return;
    const pts = [];
    if (p.atoms && p.atoms.cart) for (const c of p.atoms.cart) pts.push(c);
    if (p.cell) {
      const C = p.cell, o = p.origin || [0, 0, 0];
      for (const i of [0, 1]) for (const j of [0, 1]) for (const k of [0, 1])
        pts.push([0, 1, 2].map((x) => o[x] + i * C[0][x] + j * C[1][x] + k * C[2][x]));
    }
    if (p.vacancies) for (const c of p.vacancies) pts.push(c);
    if (!pts.length) return;
    const before = this.E.fitInclude(pts, this.dlg.getBoundingClientRect());
    if (before && !this.viewBefore) this.viewBefore = before;
    if (before) this.viewAfter = this.E.viewNow();
  }
  restoreView() {
    if (!this.viewBefore || !this.E || !this.E.viewNow) return;
    const now = this.E.viewNow(), after = this.viewAfter;
    const untouched = now && after && now.every((x, k) => Math.abs(x - after[k]) < 1e-7);
    if (untouched && this.S.activeDoc() === this.d) this.E.setViewNow(this.viewBefore);
    this.viewBefore = this.viewAfter = null;
  }

  /* ---------------------------------------------------------------- keys and documents (B11) */
  onKey(e) {
    if (this.closed || this.hidden || this.S.activeDoc() !== this.d) return false;
    const mod = e.ctrlKey || e.metaKey, ae = document.activeElement;
    const inPanel = this.dlg.contains(ae), inViewport = !!(ae && ae.closest && ae.closest(".viewer-wrap"));
    if (mod && (e.key === "z" || e.key === "Z") && !e.shiftKey && !e.altKey) {
      if (ae && (ae.tagName === "INPUT" || ae.tagName === "TEXTAREA") && inPanel) return false;   // the field's own undo
      this.discard("mod-z");
      return true;
    }
    if (mod && ((e.key === "z" || e.key === "Z") && e.shiftKey || e.key === "y" || e.key === "Y")) return !(inPanel && ae && ae.tagName === "INPUT");
    if (e.key === "Escape" && !inPanel) {
      if (this.busy) this.handOff(); else this.discard("esc");
      return true;
    }
    if (e.key === "Enter" && !e.shiftKey && !mod && (inPanel || inViewport)) {
      if (ae && (ae.tagName === "TEXTAREA" || ae.tagName === "BUTTON" || ae.tagName === "SELECT" || ae.isContentEditable)) return false;
      if (ae && ae.tagName === "INPUT") ae.dispatchEvent(new Event("change", { bubbles: true }));
      this.go();
      return true;
    }
    return false;
  }
  switched({ from, to }) {
    if (this.closed) return;
    if (to !== this.d && !this.hidden) this.hide();
    else if (to === this.d && this.hidden) this.show();
  }
  hide() {
    this.hidden = true;
    this.dlg.hidden = true;
    if (this.S.previewKeysOwner() === this.keysFn) this.S.setPreviewKeys(null);
  }
  show() {
    this.hidden = false;
    this.dlg.hidden = false;
    this.keysFn = (e) => this.onKey(e);
    this.S.setPreviewKeys(this.keysFn);
    if (this.lastPreview) requestAnimationFrame(() => this.drawPreview(this.lastPreview));
    else this.schedulePreview(true);
  }

  /* ---------------------------------------------------------------- go */
  async go() {
    if (this.busy || this.closed) return;
    const p = this.current();
    this.updateResult(p);
    if (p.error || this.goBtn.disabled) { if (p.error) this.setError(p.error); return; }
    if (this.spec.go) return this.spec.go(this.api, p);
    if (this.effective === "inplace") return this.applyInPlace(p);
    if (!this.validateName()) return;
    return this.create(p);
  }
  async applyInPlace(p) {
    const ed = this.ed, ip = p.inPlace;
    this.viewBefore = null;                        // the framed view shows the result: keep it
    this.teardown();                               // editing is on again for this one edit
    try {
      await this.E.api.edit(ed, ip.op, ip.args, { label: ip.label, wait: true });
      if (this.spec.afterInPlace) this.spec.afterInPlace(ed, this.api);
    } catch (e) { this.H.toast(`${this.spec.title}: ${e.message}`, "error"); }
    this.close();
  }
  async create(p) {
    const name = this.nameEl.value, H = this.H, S = this.S;
    const locked = S.setting("symmetryModeDefault") !== "p1";
    // one document, or several (Point Defect: one per inequivalent site)
    const items = p.many ? p.many : [{ params: p.params, name, label: p.label, hints: p.hints }];
    this.setError("");
    const opened = [];
    try {
      const source = await S.sourceOf(this.d);
      if (this.abort) this.abort.abort();
      this.busy = { pct: 0 };
      this.lock(true);
      for (const it of items) {
        const openAs = { name: it.name, label: it.label || `${this.spec.title} of ${this.d.title}`, locked,
          symprec: this.ed ? this.ed.symprec : 1e-3, wrap: S.setting("wrapIntoCell") !== false, hints: it.hints || undefined };
        const body = { source, params: it.params, open_as: openAs, doc: this.d.title };
        const r = await H.api(this.spec.route, { body, withStatus: true });
        let res = r.data;
        if (r.status === 202 && r.data.task) {
          this.busy.task = r.data.task;
          this.progress(0);
          res = await S.awaitTask(r.data.task, (t) => this.progress(t.progress || 0));
          this.busy.task = null;
        }
        if (this.handedOff) {
          // the dialog was closed while it built: offer the result, never steal the focus
          H.toast(`${this.spec.title} ready`, "success", { label: "Open", run: () => openResult(H, S, this.d, res, it.name, this.spec) });
          continue;
        }
        opened.push(await openResult(H, S, this.d, res, it.name, this.spec));
      }
      this.busy = null;
      if (this.handedOff) return null;
      this.close();
      return opened[opened.length - 1] || null;
    } catch (e) {
      this.busy = null;
      if (this.closed) {
        if (e.state !== "canceled") { H.toast(`${this.spec.title} failed: ${e.message}`, "error"); S.problem({ source: this.spec.title, message: `${this.spec.title} failed: ${e.message}` }); }
        return null;
      }
      this.lock(false);
      if (e.state === "canceled" || e.message === "Canceled") { this.setError(""); this.changed(); return null; }
      this.setError(e.message);
      return null;
    }
  }
  progress(f) {
    if (!this.busy) return;
    this.busy.pct = Math.round(f * 100);
    this.goBtn.querySelector(".bld-go-label").textContent = `Building… ${this.busy.pct}%`;
    this.goBtn.querySelector(".bld-fill").style.width = `${this.busy.pct}%`;
    this.goBtn.setAttribute("aria-busy", "true");
  }
  lock(on) {
    this.params.disabled = on;
    this.nameEl.disabled = on;
    this.dlg.querySelectorAll(".bld-result button").forEach((b) => { b.disabled = on || this.results.length === 1; });
    this.dlg.querySelector('[data-act="cancel"]').textContent = on ? "Cancel Build" : "Cancel";
    if (!on) {
      this.goBtn.removeAttribute("aria-busy");
      this.goBtn.querySelector(".bld-fill").style.width = "0";
      this.updateResult(this.cur || {});
    }
    this.updateGo();
  }
  cancelTask() {
    if (this.busy && this.busy.task) this.S.cancelTask(this.busy.task.id);
  }
  /** Close the dialog while it builds: the task goes on in the Tasks panel. */
  handOff() {
    this.handedOff = true;
    this.teardown();
    this.closed = true;
    try { this.dlg.close(); } catch (e) { /* closed */ }
    this.H.toast(`${this.spec.title} goes on in the background: see Tasks`, "info");
  }

  /* ---------------------------------------------------------------- closing */
  teardown() {
    clearTimeout(this.timer);
    if (this.abort) this.abort.abort();
    const scene = this.E && this.E.scene && this.E.scene();
    if (scene && scene.preview && scene.preview.doc === this.ed) scene.setPreview(null);
    this.restoreView();
    if (this.ed) this.ed.previewing = null;
    if (this.d.builderFrame === this) this.d.builderFrame = null;
    if (this.S.previewKeysOwner() === this.keysFn) this.S.setPreviewKeys(null);
    document.removeEventListener("mace-doc-switch", this.onSwitch);
    if (this.spec.onClose) this.spec.onClose(this.api);
    this.S.inspectorChanged();
    this.H.refresh();
  }
  discard(why = "cancel") {
    if (this.closed) return;
    if (this.busy) { this.cancelTask(); }
    if (this.spec.onDiscard && why !== "done") this.spec.onDiscard(this.api, why);
    this.teardown();
    this.closed = true;
    try { this.dlg.close(); } catch (e) { /* closed */ }
  }
  close() { this.discard("done"); }
  closedByPanel() {
    if (this.closed) return;
    if (this.busy) { this.handOff(); return; }
    this.teardown();
    this.closed = true;
  }
}

/** A built result as a document: a session at once (untitled, modified, with
 * a draft and its provenance), or a held view-only structure. */
export async function openResult(H, S, parent, res, name, spec) {
  const meta = (res.session && res.session.doc_meta) || null;
  let nd;
  if (res.session) {
    nd = await S.openUntitled(res.session, name);
    const { E } = H.editor(), e = E.edOf(nd);
    nd.saveName = name;
    nd.meta = meta;
    if (e) { e.savedRev = -1; e.docMeta = meta; if (E.drafts) E.drafts.write(e); E.stateChanged(e); }
  } else if (res.held) {
    const core = await import("../geom/core.js");
    nd = await S.openHeld(core.decodePacked(res.structure), name, res.held, { ...(res.structure.doc_meta || {}), provenance: (res.provenance || null) });
    nd.saveName = name;
    if (res.reason) S.toast(`${name} opens view-only: ${res.reason}`, "info");
  }
  if (nd && spec && spec.rebuild) nd.rebuild = () => spec.rebuild(parent);
  return nd;
}

/* ---------------------------------------------------------------- the test builder */
/** Identity (MACE_GUI_TEST_HOOKS only): the frame on its own, before any real builder. */
export function openIdentity(H, opts = {}) {
  return openBuilder(H, {
    id: "identity", title: opts.title || "Identity", route: "build/identity", defaultResult: opts.result || "new",
    results: opts.onlyNew ? ["new"] : ["new", "inplace"], primary: "Create",
    render(host, api) {
      host.innerHTML = `<label class="bld-row"><span>Delay (ms)</span><input type="number" min="0" max="30000" step="100" value="${Number(opts.delay || 0)}" data-k="delay"></label>`;
    },
    params(api) {
      const s = api.structure(), delay = Number(api.frame.params.querySelector('[data-k="delay"]').value) || 0;
      const n = opts.n || (s ? s.numbers.length : 0);
      return { params: { delay_ms: delay, fake_n: opts.n || undefined }, n, name: `${stemOf(api.d.title)}_copy`, label: `Identity of ${api.d.title}`,
        readout: `${fmtN(n)} atoms`, inPlace: { op: "set_symprec", args: { symprec: api.ed ? api.ed.symprec : 1e-3 }, label: "Identity" } };
    },
  });
}
