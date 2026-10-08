/* Drafts of modified documents (spec §2.6, plan T11): while a document is
 * modified, its structure goes into this browser's localStorage every
 * Settings › Autosave drafts seconds (0 turns it off), when the browser is
 * idle, and at once when the page is hidden or closed and when the
 * connection to the server is lost. Nothing is written into the served
 * folder. The key holds the served folder's hash (from the server, not the
 * token, so a restart finds it) and the file's path or the untitled
 * document's id:  mace.gui.draft.<dir_hash>.<path | untitled-…>
 * localStorage belongs to the page's origin, so a draft written under one
 * port is not offered under another. A draft over 2 MB is skipped (one
 * toast). The shell reads the drafts back at boot (shell.js offerDrafts). */

export const DRAFT_PREFIX = "mace.gui.draft.";
const MAX_BYTES = 2 * 1024 * 1024;

function b64(bytes) {
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(s);
}
export function unb64(text) {
  const s = atob(text), out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
  return out;
}

export function createDrafts(E, S) {
  let timer = 0, tooBig = new Set();
  const keyOf = (ed) => {
    if (!ed.path && !ed.draftId) ed.draftId = `untitled-${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
    return `${DRAFT_PREFIX}${S.dirHash()}.${ed.path || ed.draftId}`;
  };
  function encode(ed) {
    const s = ed.structure, n = s.numbers.length;
    const frac = new Float64Array(3 * n);
    for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) frac[3 * i + k] = s.frac[i][k];
    return JSON.stringify({
      v: 1, t: Date.now(), name: ed.name, path: ed.path || null, frame: ed.frame ?? -1, diskStamp: ed.diskStamp || null,
      cell: s.cell, z: b64(new Uint8Array(new Uint16Array(s.numbers).buffer)), frac: b64(new Uint8Array(frac.buffer)),
      symMode: ed.symMode, symprec: ed.symprec, measurements: ed.measurements,
      bondOverrides: [...ed.bondOverrides], selectionSets: ed.selectionSets,
      // Phase 4: a built or dropped document's provenance and hints (old drafts have none)
      meta: ed.docMeta || (ed.structure && ed.structure.doc_meta) || null, saveName: ed.doc.saveName || null,
    });
  }
  /** Write (or clear) one document's draft; returns what happened. */
  function write(ed) {
    let key;
    try { key = keyOf(ed); } catch (e) { return "error"; }
    try {
      if (!ed.dirty || ed.closed) { localStorage.removeItem(key); return "clear"; }
      const text = encode(ed);
      if (text.length > MAX_BYTES) {
        localStorage.removeItem(key);
        if (!tooBig.has(key)) { tooBig.add(key); S.toast(`${ed.name} is too large for a draft (over 2 MB): save it to keep the changes`, "info"); }
        return "big";
      }
      localStorage.setItem(key, text);
      return "write";
    } catch (e) {
      return "error";                                     // private mode, quota
    }
  }
  /** Every open structure document, now. */
  function flush() {
    const out = {};
    for (const ed of S.editorDocs()) out[ed.path || ed.name] = write(ed);
    return out;
  }
  function clear(ed) { try { localStorage.removeItem(keyOf(ed)); } catch (e) { /* private */ } }
  /** The timer: every autosaveDrafts seconds, when the browser is idle. */
  function schedule() {
    clearTimeout(timer);
    const s = Number(S.setting("autosaveDrafts"));
    if (!(s > 0)) return;
    timer = setTimeout(() => {
      const go = () => { if (S.editorDocs().some((ed) => ed.dirty)) flush(); schedule(); };
      if (window.requestIdleCallback) requestIdleCallback(go, { timeout: 2000 }); else go();
    }, s * 1000);
  }
  addEventListener("pagehide", flush);
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "hidden") flush(); });
  schedule();
  return { flush, write, clear, schedule, keyOf };
}

/** A draft back as a structure for edit/from_structure. */
export function decode(d) {
  const z = Array.from(new Uint16Array(unb64(d.z).buffer)), f = new Float64Array(unb64(d.frac).buffer), frac = [];
  for (let i = 0; i < z.length; i++) frac.push([f[3 * i], f[3 * i + 1], f[3 * i + 2]]);
  return { numbers: z, frac, cell: d.cell };
}
