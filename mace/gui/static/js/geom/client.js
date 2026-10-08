/* The page's side of the geometry worker: one module worker for the whole
 * page (spec §9.1: the browser adds exactly one), started on first use.
 * job(kind, args, transfer) resolves with the result, or with
 * {superseded: true} when a newer job of the same kind replaced it. */
let worker = null, nextId = 1;
const pending = new Map();
export const stats = { started: 0, jobs: 0, superseded: 0 };

function ensure() {
  if (worker) return worker;
  worker = new Worker(new URL("./geom.worker.js", import.meta.url), { type: "module", name: "mace-geom" });
  stats.started++;
  worker.onmessage = (e) => {
    const m = e.data, p = pending.get(m.id);
    if (!p) return;
    pending.delete(m.id);
    if (m.superseded) { stats.superseded++; p.resolve({ superseded: true }); }
    else if (m.error) p.reject(new Error(m.error));
    else p.resolve({ ...m.result, ms: m.ms });
  };
  worker.onerror = (e) => {
    for (const p of pending.values()) p.reject(new Error(e.message || "geometry worker failed"));
    pending.clear();
  };
  return worker;
}

export function job(kind, args, transfer = []) {
  const w = ensure(), id = nextId++;
  stats.jobs++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    w.postMessage({ id, kind, args }, transfer);
  });
}

export const running = () => !!worker;
