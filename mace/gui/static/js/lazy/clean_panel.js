/* Build › Clean Geometry… (spec §2.9, plan T8; Clean tiers plan §8): a
 * pre-relaxation run in the process pool as a proposal that is not committed.
 * The viewport shows the proposal with the original as a 30% ghost and arrows
 * from each atom to where it goes; the summary reads "max shift 0.012 Å ·
 * RMS 0.004 Å · 6 bonds changed · group Fm-3m → Fm-3m · converged in 41
 * steps". Accept (Enter) is one undo step "Clean Geometry"; Discard (Esc,
 * Cancel, Mod+Z, closing) leaves none.
 *
 * Methods (GET /api/clean/methods): Springs (the default: the spring model,
 * with today's live preview), GFN-FF (the external xtb program), the ML
 * potentials (third-party models, unrelated to this MACE, run in their own
 * environment) and Auto. GFN-FF, ML and Auto take a minute or more, so they
 * never preview on a field change: Run starts a job, and a second Run cancels
 * the first (the loaded model stays). The GUI never installs or downloads
 * anything; a missing method shows the commands that set it up. */
import { openBuilder } from "./builder_util.js";
import * as profile from "../profile.js";

const RELAX_NOTE = {
  springs: "The cell follows the bond springs only; ionic crystals shrink a few percent",
  gfnff: "Cell relaxation of ionic solids often fails with GFN-FF",
  mlip: "ML potentials give lattices about 1% too large",
};

export function open(H) {
  const S = H.shell();
  let prop = null, msg = "", progress = null, methods = null, checking = true, runRequested = false;
  let taskId = null, lastResult = null, statusTimer = 0, runSeq = 0, closedFlag = false;
  const kindOf = (m) => (m === "springs" || m === "gfnff" || m === "auto" ? m : "mlip");
  const docMeta = (api) => (api.ed && (api.ed.docMeta || (api.ed.structure && api.ed.structure.doc_meta))) || api.d.meta || {};
  const discardProposal = (api) => {
    const ed = api.ed;
    if (!prop || !ed || !ed.srv.session) return;
    H.api("edit/clean_discard", { body: { session: ed.srv.session, proposal_id: prop.proposal_id } }).catch(() => {});
    prop = null;
  };
  const cancelTask = (id) => H.api(`tasks/${id}/cancel`, { body: {} }).catch(() => {});
  // a running job, or one whose task id has not come back yet (runSeq moves on: it is canceled on arrival)
  const cancelRunning = () => {
    runSeq += 1;
    if (taskId) { const id = taskId; taskId = null; cancelTask(id); }
  };
  const esc = (t) => H.esc(String(t == null ? "" : t));
  const methodOf = (id) => (methods || []).find((m) => m.id === id) || null;

  async function loadMethods(api, recheck = false) {
    try {
      const r = recheck ? await H.api("clean/methods/recheck", { body: {} }) : await H.api("clean/methods");
      checking = !!r.checking;
      methods = r.methods || [];
    } catch (e) { checking = false; methods = methods || [{ id: "springs", label: "Springs: rough, always available", available: true }]; }
    fillSelect(api);
    if (checking) { clearTimeout(statusTimer); statusTimer = setTimeout(() => loadMethods(api), 1000); }
    api.changed();
  }

  function fillSelect(api) {
    const host = api.frame.params, sel = host.querySelector('[data-k="method"]');
    // the method used last time (memory.cleanMethod), when it is still available
    const cur = sel.dataset.picked ? sel.value : profile.memory("cleanMethod", sel.value || "springs");
    sel.innerHTML = (methods || []).map((m) => `<option value="${esc(m.id)}"${m.available ? "" : " disabled"}${m.id === cur ? " selected" : ""}
      data-tip="${esc(m.unrelated_note || m.reason || "")}">${esc(m.label)}${m.available ? "" : " (not available)"}</option>`).join("");
    if (!methodOf(cur) || !methodOf(cur).available) sel.value = "springs";
    host.querySelector('[data-k="checking"]').hidden = !checking;
    const list = host.querySelector('[data-k="unavailable"]');
    const off = (methods || []).filter((m) => !m.available && m.status !== "checking");
    host.querySelector('[data-k="more-summary"]').textContent = off.length
      ? `${off.length} method${off.length === 1 ? "" : "s"} not available: how to set ${off.length === 1 ? "it" : "them"} up`
      : checking ? "Other methods" : "Every method is available";      // never claim "available" while still checking
    // the hint is prose; each command in it is its own chip, and a click copies just that command
    list.innerHTML = off.map((m) =>
      `<li class="clean-hint" data-method="${esc(m.id)}"><b>${esc(m.label)}</b>: ${esc(m.reason || m.status)}${m.hint
        ? `<br><span class="insp-hint">Set up: ${esc(m.hint)}</span>` : ""}${(m.commands || []).length
        ? `<span class="clean-cmds">${m.commands.map((c) => `<code class="clean-cmd" title="Click to copy">${esc(c)}</code>`).join("")}</span>` : ""}</li>`).join("");
  }

  function methodChanged(api) {
    const host = api.frame.params, m = host.querySelector('[data-k="method"]').value, k = kindOf(m), info = methodOf(m) || {};
    const d = (info.defaults || {}), lim = host.querySelector('[data-k="limit"]');
    lim.max = String(info.max_time_limit || (k === "springs" ? 120 : 600));
    lim.value = String(d.time_limit || (k === "springs" ? 10 : k === "gfnff" ? 60 : 120));
    host.querySelector('[data-k="fmax"]').placeholder = String(d.fmax || "");
    host.querySelector('[data-k="steps"]').placeholder = String(d.steps || "");
    host.querySelector('[data-k="run-row"]').hidden = k === "springs";
    host.querySelector('[data-k="springs-note"]').hidden = k !== "springs";
    host.querySelector('[data-k="adv"]').hidden = k === "springs";
    host.querySelector('[data-k="sym-first-row"]').hidden = k === "springs";
    host.querySelector('[data-k="unrelated"]').hidden = k !== "mlip";
    host.querySelector('[data-k="unrelated"]').textContent = info.unrelated_note ? `A third-party model; not part of, and unrelated to, this MACE tool` : "";
    host.querySelector('[data-k="uhf-note"]').hidden = k !== "gfnff";
    host.querySelector('[data-k="ml-charge-note"]').hidden = k !== "mlip";
    host.querySelector('[data-k="relax-note"]').textContent = RELAX_NOTE[k === "auto" ? "mlip" : k];
    discardProposal(api);
    lastResult = null;
    cancelRunning();
  }

  return openBuilder(H, {
    id: "clean", title: "Clean Geometry", route: "clean", results: [], defaultResult: "inplace", primary: "Accept", debounce: 400,
    render(host, api) {
      const meta = docMeta(api), hints = meta.hints || {}, deck = meta.deck || {};
      const q = hints.charge !== undefined && hints.charge !== null ? String(hints.charge) : (deck.charge_records && deck.charge_records.length ? "" : "0");
      host.innerHTML = `<p class="bld-note" data-k="springs-note">Spring model (rough pre-relaxation, not physics)</p>
        <label class="bld-row"><span>Method</span><select data-k="method" aria-label="Method"><option value="springs">Springs: rough, always available</option></select></label>
        <p class="bld-note" data-k="checking">Checking available methods…</p>
        <p class="bld-note insp-hint" data-k="unrelated" hidden></p>
        <details class="clean-more" data-k="more"><summary data-k="more-summary">Other methods</summary>
          <ul class="clean-unavailable" data-k="unavailable"></ul>
          <button type="button" class="tb-btn-text" data-k="recheck">Recheck</button></details>
        ${deck.geometry_edits && deck.geometry_edits.length ? `<p class="bld-note bld-warn" data-k="host-warning">This document is the host geometry of ${esc(deck.name || "its deck")} (${esc(deck.geometry_edits.join(", "))}): Clean acts on the host, not the structure CRYSTAL builds from the deck.</p>` : ""}
        <label class="bld-row"><span>Charge</span><input type="number" step="1" value="${esc(q)}" data-k="charge" aria-label="Charge"></label>
        <p class="bld-note insp-hint">Springs ignore the charge<span data-k="uhf-note" hidden>; the spin (uhf) is recorded only; GFN-FF ignores it</span><span data-k="ml-charge-note" hidden>; ML potentials take no charge input (they see a neutral cell)</span></p>
        <label class="bld-check"><input type="checkbox" data-k="sym" checked> Keep the space group</label>
        <div data-k="sym-first-row" hidden><label class="bld-check"><input type="checkbox" data-k="symfirst"> Symmetrize first</label>
          <label class="bld-row"><span>Tolerance (Å)</span><input type="number" min="0.001" step="0.01" value="0.1" data-k="symtol" aria-label="Symmetrize tolerance (Å)" disabled></label>
          <p class="bld-note insp-hint" data-k="symfirst-note" hidden>This can raise the space group; CRYSTAL keeps the raised group</p></div>
        <label class="bld-check"><input type="checkbox" data-k="cell"> Relax the cell</label>
        <p class="bld-note insp-hint" data-k="relax-note">${RELAX_NOTE.springs}</p>
        <label class="bld-row"><span>Time limit (s)</span><input type="number" min="0.5" max="120" step="0.5" value="10" data-k="limit" aria-label="Time limit (s)"></label>
        <div class="bld-row" data-k="run-row" hidden><span></span><span><button type="button" class="tb-btn-text primary" data-k="run">Run</button></span></div>
        <details data-k="adv" hidden><summary>Advanced</summary>
          <label class="bld-row"><span>fmax (eV/Å)</span><input type="number" min="0.0001" step="0.001" data-k="fmax" aria-label="fmax (eV/Å)"></label>
          <label class="bld-row"><span>Steps</span><input type="number" min="1" step="1" data-k="steps" aria-label="Steps"></label></details>
        <div class="bld-row" data-k="longer-row" hidden><span></span><span><button type="button" class="tb-btn-text" data-k="longer">Run Longer</button></span></div>`;
      host.querySelector('[data-k="method"]').addEventListener("change", (e) => {
        e.target.dataset.picked = "1";
        profile.remember("cleanMethod", e.target.value);
        methodChanged(api); api.changed();
      });
      host.querySelector('[data-k="recheck"]').addEventListener("click", () => { checking = true; fillSelect(api); loadMethods(api, true); });
      host.querySelector('[data-k="symfirst"]').addEventListener("change", (e) => {
        host.querySelector('[data-k="symtol"]').disabled = !e.target.checked;
        host.querySelector('[data-k="symfirst-note"]').hidden = !e.target.checked;
      });
      const start = () => { runRequested = true; cancelRunning(); api.frame.previewKey = null; api.frame.schedulePreview(true); };
      host.querySelector('[data-k="run"]').addEventListener("click", () => { api.changed(); start(); });
      host.querySelector('[data-k="longer"]').addEventListener("click", () => {
        const i = host.querySelector('[data-k="limit"]');
        i.value = String(Math.min(Number(i.max || 120), Number(i.value || 10) * 2));
        api.changed();
        if (kindOf(host.querySelector('[data-k="method"]').value) !== "springs") start();
      });
      host.addEventListener("click", (e) => {
        const c = e.target.closest(".clean-cmd");
        if (c && navigator.clipboard) navigator.clipboard.writeText(c.textContent).then(() => H.toast("Copied", "success")).catch(() => {});
      });
      api.frame.readout.addEventListener("click", (e) => {
        if (!e.target.closest('[data-k="copy-cite"]') || !prop || !prop.citation) return;
        const text = prop.citation;
        api.frame.dlg.dataset.copied = text;            // what was copied (also where the clipboard is blocked)
        if (navigator.clipboard) navigator.clipboard.writeText(text).then(() => H.toast("Citation copied", "success")).catch(() => {});
      });
      loadMethods(api);
    },
    params(api) {
      const host = api.frame.params, val = (k) => host.querySelector(`[data-k="${k}"]`);
      const method = val("method").value || "springs", kind = kindOf(method);
      const t = Number(val("limit").value);
      if (!(t > 0)) return { error: "The time limit is a number of seconds", blocked: true };
      // springs: exactly the request Clean made before the other methods existed
      const p = { keep_symmetry: val("sym").checked, relax_cell: val("cell").checked, time_limit: t };
      if (kind !== "springs") {
        const qs = val("charge").value.trim();
        p.method = method;
        val("run").disabled = true;
        if (qs === "") {
          const deck = docMeta(api).deck || {};
          if (deck.charge_records && deck.charge_records.length)
            return { params: p, error: `Enter the charge: the deck has ${deck.charge_records.join("/")}`, blocked: true, readout: "" };
        } else {
          if (!/^-?\d+$/.test(qs)) return { params: p, error: "The charge is a whole number", blocked: true };
          p.charge = Number(qs);
        }
        if (val("symfirst").checked) p.symmetrize_tol = Number(val("symtol").value);
        if (val("fmax").value) p.fmax = Number(val("fmax").value);
        if (val("steps").value) p.max_steps = Number(val("steps").value);
        val("run").disabled = false;
      }
      if (msg) return { params: p, error: msg, blocked: true, readout: "" };
      let ro;
      if (progress !== null) ro = `<p>Cleaning… <b>${Math.round(progress * 100)}%</b>${kind !== "springs" ? ` <span class="insp-hint">${esc(lastResult && lastResult.message || "")}</span>` : ""}</p>`;
      else if (prop) {
        const r = prop, lines = [];
        if (r.ran) lines.push(`<p data-k="ran" class="clean-ran">${esc(r.ran)}</p>`);
        if (r.attempts && r.attempts.length > 1) lines.push(`<ul class="clean-attempts" data-k="attempts">${r.attempts.map((a) =>
          `<li>${esc(a.method)}: ${esc(a.status)}${a.reason ? ` (${esc(a.reason)})` : ""}</li>`).join("")}</ul>`);
        if (r.warning) lines.push(`<p class="warn">${esc(r.warning)}</p>`);
        lines.push(`<p data-k="summary">${esc(r.summary)}</p>`);
        for (const w of r.warnings || []) lines.push(`<p class="warn" data-k="warning">${esc(w)}</p>`);
        if (r.starting_geometry_note) lines.push(`<p class="warn" data-k="opt-next">${esc(r.starting_geometry_note)}</p>`);
        if (r.citation) lines.push(`<p class="insp-hint" data-k="citation">Cite: ${esc(r.citation.split("\n")[0])}… <button type="button" class="tb-btn-text" data-k="copy-cite">Copy citation</button></p>`);
        ro = lines.join("");
      } else ro = kind === "springs" ? "<p>…</p>" : "<p class=\"insp-hint\">Press Run to start (GFN-FF and ML runs take a minute or more)</p>";
      return { params: p, n: api.structure().numbers.length, readout: ro, blocked: !prop };
    },
    async preview(api, p) {
      const ed = api.ed;
      if (!ed) return null;
      const kind = kindOf(p.params.method || "springs");
      if (kind !== "springs" && !runRequested) return null;      // GFN-FF/ML/Auto: only on Run
      runRequested = false;
      discardProposal(api);
      msg = "";
      progress = 0;
      api.frame.params.querySelector('[data-k="longer-row"]').hidden = true;
      if (!ed.srv.session) await api.E.editing.ensureSession(ed);
      await waitIdle(ed);
      const body = { session: ed.srv.session, base_rev: ed.srv.rev, ...p.params };
      const mine = ++runSeq;
      let r;
      try {
        const onUpdate = (t) => {
          lastResult = t;
          if (t.state === "running" || t.state === "queued") { progress = t.progress || 0; api.changed(); }
        };
        const resp = await H.api("clean", { body, withStatus: true });
        r = resp.data;
        if (resp.status === 202 && r && r.task) {
          if (closedFlag || mine !== runSeq) { cancelTask(r.task.id); return null; }   // closed or replaced meanwhile
          taskId = r.task.id;
          r = await S.awaitTask(r.task, onUpdate);
        } else if (r && r.task) S.taskEvent(r.task);            // answered at once: the Tasks panel still sees it
      } catch (e) {
        progress = null;
        if (e.state === "canceled") { api.changed(); return null; }
        throw e;
      } finally { if (mine === runSeq) { progress = null; taskId = null; } }
      if (mine !== runSeq || closedFlag) return null;           // replaced by a newer Run, or the dialog closed
      if (r.nothing_to_clean) {
        msg = r.message.charAt(0).toUpperCase() + r.message.slice(1);
        api.changed();
        return { clear: true };
      }
      prop = r;
      api.frame.params.querySelector('[data-k="longer-row"]').hidden = !(r.report.timed_out ||
        (r.attempts || []).some((a) => /time limit/.test(a.reason || "")));
      api.changed();
      return r;
    },
    drawPreview(r, api, core) {
      if (r.clear || !r.proposal) return { kind: "clean" };
      const s = api.structure(), vec = r.displacements || [];
      const max = r.comparison ? r.comparison.max_displacement : 0;
      const scale = max > 0 ? Math.max(1, Math.min(200, Math.round(1.5 / max))) : 1;
      const p = { kind: "clean", dimStructure: true, opacity: 1, arrows: [] };
      if (vec.length === s.numbers.length) {
        p.atoms = { numbers: s.numbers, symbols: s.symbols, cart: s.cart.map((c, i) => [c[0] + vec[i][0], c[1] + vec[i][1], c[2] + vec[i][2]]) };
        p.arrows = s.cart.map((c, i) => (Math.hypot(...vec[i]) > 1e-4 ? [c, [c[0] + vec[i][0] * scale, c[1] + vec[i][1] * scale, c[2] + vec[i][2] * scale]] : null)).filter(Boolean);
      }
      return p;
    },
    async go(api) {
      const ed = api.ed, frame = api.frame;
      if (!prop) return;
      const pid = prop.proposal_id;
      clearTimeout(statusTimer);
      frame.teardown();                         // editing comes back for this one step
      frame.closed = true;
      try { frame.dlg.close(); } catch (e) { /* closed */ }
      try {
        const res = await api.E.editing.request(ed, "clean_accept", "edit/clean_accept", () => ({ proposal_id: pid }), { changes: true });
        if (res && res.doc_meta) ed.docMeta = res.doc_meta;
        prop = null;
      } catch (e) {
        if (e.status === 404 || /session|proposal/.test(e.message)) H.toast("The preview was lost when the server let go of this document. Run Clean again", "warning");
        else H.toast(`Clean Geometry: ${e.message}`, "error");
      }
    },
    onDiscard(api) { closedFlag = true; clearTimeout(statusTimer); cancelRunning(); discardProposal(api); },
    // every way the dialog closes (the panel's own close too): a running GFN-FF/ML job stops with it
    onClose() { closedFlag = true; clearTimeout(statusTimer); cancelRunning(); },
  });
}

async function waitIdle(e, ms = 20000) {
  const end = performance.now() + ms;
  while (e && (e.pending > 0 || e.waiting) && performance.now() < end) await new Promise((r) => setTimeout(r, 50));
}
