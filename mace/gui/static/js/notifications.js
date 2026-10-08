/* The notification history (plan E13, PLAN_viewer O8): every toast is
 * recorded here, newest first, for the session (not a preference: never
 * persisted). Documented API for Copy Diagnostics and the connection's
 * diagnostics ring:
 *   recent(n)      -> the newest n entries {t, kind, text, action?, count} (text only, no DOM)
 *   subscribe(fn)  -> fn(entry) for every new or repeated entry; returns an unsubscribe function
 *   clear()        -> empties the history
 * A repeat of the newest entry's text and kind within 10 s raises its count
 * instead of adding a row. */
const MAX = 500;
const entries = [];
const subs = new Set();
let unread = 0;

export function record(text, kind = "info", action = null) {
  const now = Date.now();
  const top = entries[0];
  let e;
  if (top && top.text === text && top.kind === kind && now - top.last < 10000) {
    top.count += 1; top.last = now; e = top;
  } else {
    e = { t: now, last: now, kind, text: String(text), action: action && action.label ? action.label : undefined, count: 1 };
    entries.unshift(e);
    if (entries.length > MAX) entries.length = MAX;
  }
  unread += 1;
  for (const fn of subs) { try { fn({ ...e }); } catch (err) { /* a subscriber never breaks a toast */ } }
  return e;
}
export function recent(n = 50) {
  return entries.slice(0, Math.max(0, n)).map(({ t, kind, text, action, count }) =>
    (action ? { t, kind, text, action, count } : { t, kind, text, count }));
}
export function subscribe(fn) { subs.add(fn); return () => subs.delete(fn); }
export function clear() { entries.length = 0; unread = 0; for (const fn of subs) { try { fn(null); } catch (e) { /* ignore */ } } }
export function unreadCount() { return unread; }
export function markRead() { unread = 0; for (const fn of subs) { try { fn(null); } catch (e) { /* ignore */ } } }
export function all() { return entries.slice(); }
