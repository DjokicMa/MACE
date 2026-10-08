/* Saving a structure document (spec §2.6, plan D1, T11). One path for
 * every save: /api/structure/save through the document's request queue.
 *   - a .cif saves in place (every atom, P1 CIF), after a one-time
 *     confirmation per file ("Overwrite MgO.cif? The original will be kept
 *     as MgO.cif.bak"), unless Settings › Confirm before overwriting is off;
 *     the .bak follows Keep a .bak and is never replaced once it exists;
 *   - a .out, a .d12 or an untitled document goes through Save As
 *     (<stem>_edited.cif by default);
 *   - every in-place save sends the file's stamp from when it was opened:
 *     if the file changed on disk since, the user picks Overwrite, Save As…
 *     or Cancel (also with the confirmation turned off).
 * After a save the document is clean, takes the new name after a Save As,
 * the explorer and the recent files learn about it, and its draft goes. */

export function createSaving(E, S) {
  const askedKey = (ed) => `mace.gui.saveConfirmed.${S.dirHash()}.${ed.path}`;
  const asked = (ed) => { try { return localStorage.getItem(askedKey(ed)) === "1"; } catch (e) { return false; } };
  const remember = (ed) => { try { localStorage.setItem(askedKey(ed), "1"); } catch (e) { /* private */ } };
  const stem = (p) => (p || "").split("/").pop().replace(/\.[^.]+$/, "");
  const folder = (p) => (p && p.includes("/") ? p.slice(0, p.lastIndexOf("/") + 1) : "");

  /** The request: {saved, rev, file, bak, bak_created} or {conflict: {...}} or null. */
  async function send(ed, body) {
    if (ed.doc.held && !ed.srv.session) {
      // a view-only built structure: no session, the server's held copy is saved
      try { return await window.MACE.api("structure/save", { body: { ...body, source: { held: ed.doc.held } } }); }
      catch (e) { if (e.status === 409) return { conflict: e.data }; throw e; }
    }
    return E.editing.request(ed, "save", "structure/save", () => body, { payloadKey: false, passConflicts: true });
  }
  function saved(ed, r, path, asNew) {
    ed.savedRev = r.rev;
    ed.diskStamp = r.file;
    if (asNew) {
      ed.path = path;
      ed.name = path.split("/").pop();
      ed.doc.path = path;
      ed.doc.title = ed.name;
      ed.doc.fileKind = path.toLowerCase().endsWith(".cif") ? "cif" : ed.doc.fileKind;
      ed.doc.untitled = false;
      ed.doc.saveName = null;
      ed.frame = -1;
    }
    S.afterSave(ed, path, r);
    const bak = r.bak && r.bak_created ? ` (the original is kept as ${r.bak.split("/").pop()})` : "";
    const h = r.history;
    if (h && h.skipped) S.toast(`Saved ${path.split("/").pop()}${bak}. ${h.skipped}`, "info");
    else if (h && h.id && S.undoSave) {
      // the previous version is in File History: Undo for 30 s, and the history itself
      S.toast(`Saved ${path.split("/").pop()}${bak}`, "success",
        { label: "Undo", ms: 30000, run: () => S.undoSave(ed, path, h, r.file), extra: { label: "History", run: () => S.fileHistory(path) } });
    } else S.toast(`Saved ${path.split("/").pop()}${bak}`, "success");
    E.stateChanged(ed);
  }

  /** Save: in place for a .cif, else Save As. */
  async function save(ed) {
    const p = ed.path || "";
    if (!p.toLowerCase().endsWith(".cif") || ed.doc.untitled) return saveAs(ed);
    if (S.setting("confirmOverwrite") !== false && !asked(ed)) {
      const ok = await S.confirmOverwrite(ed);
      if (!ok) return null;
      if (ok === "remember") remember(ed);
    }
    return inPlace(ed, ed.diskStamp);
  }
  async function inPlace(ed, expected) {
    const r = await send(ed, { path: ed.path, format: "cif", cif_mode: "p1", overwrite: true,
      backup: S.setting("keepBackup") !== false, expected: expected || undefined });
    if (!r) return null;
    if (r.conflict && r.conflict.changed_on_disk) {
      const pick = await S.changedOnDisk(ed);
      if (pick === "overwrite") return inPlace(ed, { mtime_ns: r.conflict.mtime_ns, size: r.conflict.size });
      if (pick === "saveas") return saveAs(ed);
      return null;
    }
    if (r.conflict) { S.toast(r.conflict.error || "Not saved", "error"); return null; }
    saved(ed, r, ed.path, false);
    return r;
  }
  /** Save As: the dialog picks a folder, a name and a format. */
  async function saveAs(ed, preset = {}) {
    // a built or dropped document is saved under its own name (MgO_2x2x2, MgO_dropped)
    const def = ed.doc.saveName ? ed.doc.saveName
      : `${folder(ed.path)}${stem(ed.path || ed.name) || "structure"}${ed.path && ed.path.toLowerCase().endsWith(".cif") && !ed.doc.untitled ? "" : "_edited"}`;
    const choice = await S.saveAsDialog(ed, def, preset);
    if (!choice) return null;
    return saveTo(ed, choice, false);
  }
  async function saveTo(ed, choice, overwrite) {
    const own = choice.path === ed.path;
    const r = await send(ed, { path: choice.path, format: choice.format, cif_mode: choice.cifMode || "p1", overwrite: overwrite || own,
      backup: S.setting("keepBackup") !== false, expected: own ? ed.diskStamp || undefined : undefined });
    if (!r) return null;
    if (r.conflict && r.conflict.needs_overwrite) {
      if (!(await S.confirmReplace(choice.path))) return null;
      return saveTo(ed, choice, true);
    }
    if (r.conflict && r.conflict.changed_on_disk) {
      const pick = await S.changedOnDisk(ed);
      if (pick === "overwrite") { ed.diskStamp = { mtime_ns: r.conflict.mtime_ns, size: r.conflict.size }; return saveTo(ed, choice, true); }
      if (pick === "saveas") return saveAs(ed);
      return null;
    }
    if (r.conflict && r.conflict.protected) { S.toast(r.conflict.error, "error"); return null; }
    if (r.conflict) { S.toast(r.conflict.error || "Not saved", "error"); return null; }
    saved(ed, r, choice.path, true);
    return r;
  }
  /** Save All: every modified document in turn, stopping at the first that needs Save As. */
  async function saveAll(eds) {
    for (const ed of eds) {
      if (!ed.dirty) continue;
      const p = ed.path || "";
      if (!p.toLowerCase().endsWith(".cif") || ed.doc.untitled) { S.activateEd(ed); return saveAs(ed); }
      if (!(await save(ed))) return null;
    }
    return true;
  }
  return { save, saveAs, saveAll, saveTo };
}
