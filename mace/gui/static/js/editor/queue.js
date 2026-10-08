/* One serial queue of server requests per document (plan A3a). The browser
 * may use several connections and the server answers on several threads,
 * so order is kept here: at most one request of a document is on its way,
 * the next leaves only when the answer to the one before has been adopted.
 * Every request carries the revision it was made against (base_rev); the
 * server refuses a mismatch (409) rather than apply anything out of order.
 *
 * Nudges that wait with the same merge key travel as one request carrying
 * their sum and their number. A failed request drops the ones after it.
 * Two failures are told apart: 403 (the server restarted with a new token:
 * nothing can be recovered here, nothing is retried) and 404 (the server let
 * go of the session to save memory: it is rebuilt from the structure this
 * page holds, and the request is sent again). */

/** A name edit/from_structure accepts (letters, digits, _ . -): "MgO.cif · step 3" -> "MgO.cif_step_3". */
export const safeName = (n) => String(n || "").replace(/[^A-Za-z0-9_.-]+/g, "_").replace(/^[^A-Za-z0-9]+/, "").slice(0, 60) || "structure";

export class EditQueue {
  constructor(ed, hooks) {
    this.ed = ed;
    this.hooks = hooks;
    this.items = [];
    this.inFlight = null;
    this.stopped = false;
    this.seq = 0;
    this.log = [];                       // every request sent, for tests: {id, kind, t0, t1, status}
  }
  get size() { return this.items.length + (this.inFlight ? 1 : 0); }
  get idle() { return !this.inFlight && !this.items.length; }

  /** Queue a request: {kind, route, make(ed) -> body, predicted, merge, count, onDone, onConfirm, passConflicts}. */
  push(item) {
    if (this.stopped) return Promise.resolve(null);
    const last = this.items[this.items.length - 1];
    if (item.merge && last && last.merge === item.merge && last.group === item.group && last.combine) {
      last.combine(item);                // the waiting nudge carries this one too
      this.changed();
      return last.promise;
    }
    item.id = ++this.seq;
    item.promise = new Promise((resolve) => { item.resolve = resolve; });
    this.items.push(item);
    this.changed();
    this.pump();
    return item.promise;
  }
  changed() { this.ed.pending = this.size; if (this.hooks.changed) this.hooks.changed(this.ed); }

  async pump() {
    if (this.inFlight || this.stopped || !this.items.length) return;
    const item = this.items.shift();
    this.inFlight = item;
    this.changed();
    let result = null;
    try {
      result = await this.send(item);
    } finally {
      this.inFlight = null;
      item.resolve(result);
      this.changed();
      if (this.idle && this.hooks.drained) this.hooks.drained(this.ed);
      this.pump();
    }
  }

  async post(route, body, item) {
    const headers = { "X-Requested-With": "mace-gui", "Content-Type": "application/json",
      ...(window.__maceQueueHeaders ? window.__maceQueueHeaders(item, body) || {} : {}) };
    const entry = { id: item ? item.id : 0, kind: item ? item.kind : route, route, t0: performance.now() };
    this.log.push(entry);
    let res, data;
    try {
      res = await fetch("api/" + route, { method: "POST", credentials: "same-origin", headers, body: JSON.stringify(body) });
    } catch (e) {
      entry.t1 = performance.now(); entry.status = 0;
      return { status: 0, data: { error: "the server did not answer (" + e.message + ")" } };
    }
    try { data = await res.json(); } catch (e) { data = { error: res.statusText }; }
    entry.t1 = performance.now(); entry.status = res.status;
    return { status: res.status, data };
  }

  async send(item) {
    const ed = this.ed;
    for (let attempt = 0; attempt < 3; attempt++) {
      const body = item.make(ed);
      const { status, data } = await this.post(item.route, body, item);
      if (status === 200 && data && data.needs_confirm) {
        if (data.session && !ed.srv.session) this.hooks.opened(ed, data);   // the open block made the session
        const more = item.onConfirm ? await item.onConfirm(data) : null;
        if (!more) { this.hooks.canceled(ed, item); return null; }
        item.extra = { ...(item.extra || {}), ...more };
        attempt--;                       // a confirmation is not a failure
        continue;
      }
      if (status === 200) { this.hooks.adopt(ed, data, item); return data; }
      if (status === 403) { this.stop(); this.hooks.lost(ed, data); return null; }
      if (status === 409 && data && data.conflict === "rev") { this.drop(); this.hooks.conflict(ed, data); return null; }
      if ((status === 404 && ed.srv.session) || (status === 409 && data && data.changed_on_disk && !ed.srv.session)) {
        // the session is gone (evicted), or the file changed on disk before the
        // first edit: a session from the structure on screen, then this request again
        const ok = await this.recreate(status === 404 ? "evicted" : "changed");
        if (ok) continue;
        return null;
      }
      // Save: "already exists" and "changed on disk" are questions for the user, not failures
      if (status === 409 && item.passConflicts && data) return { conflict: data };
      this.drop();
      this.hooks.failed(ed, item, data && data.error ? data.error : `HTTP ${status}`);
      return null;
    }
    this.drop();
    this.hooks.failed(ed, item, "the server kept refusing this edit");
    return null;
  }

  async recreate(why) {
    const ed = this.ed;
    const s = ed.srv.structure;
    const body = { name: safeName(ed.name), cell: s.cell, numbers: s.numbers.slice(), frac: s.frac,
      locked: ed.symMode === "sym", symprec: ed.symprec, label: `Restored ${ed.name}` };
    const { status, data } = await this.post("edit/from_structure", body, { id: 0, kind: "recreate" });
    if (status !== 200) {
      if (status === 403) { this.stop(); this.hooks.lost(ed, data); return false; }
      this.drop();
      this.hooks.failed(ed, null, data && data.error ? data.error : `HTTP ${status}`);
      return false;
    }
    this.hooks.recreated(ed, data, why);
    return true;
  }

  /** Drop every waiting request (after a failure or a conflict). */
  drop() {
    const dropped = this.items.splice(0);
    if (dropped.length) console.info("mace: dropped queued edits", dropped.map((x) => x.kind));
    dropped.forEach((x) => x.resolve(null));
    this.changed();
  }
  stop() { this.stopped = true; this.drop(); }
}
