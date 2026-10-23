/* The Inspector's two levels of detail and its Inspector / Setup tabs (1.1.6).
 *
 * Basic shows the controls most runs need; Advanced adds the niche ones. A
 * control, row or whole section that is niche carries data-tier="advanced"
 * (untagged means Basic); any module that builds Inspector content tags its
 * own. In Basic, the container's data-insp-tier="basic" hides every tagged
 * element (panelctx.css), so content that is rebuilt or arrives later (a
 * lazily loaded panel, a mode picked in the table) follows without a call.
 * The choice is kept per view kind (Settings "inspector.tier": {kind: tier}):
 * the structure view can show Advanced while the mode view stays Basic.
 * Settings › Appearance › Inspector detail (DETAIL) can instead hold every view
 * at Basic or at Advanced; it wins over the table and the server's default.
 *
 * A section whose advanced controls are hidden ends with a "N more in
 * Advanced" button, so nothing is out of reach from Basic.
 *
 * The tab strip on top of the Inspector column switches between the
 * Inspector and Calculation Setup (which fills the same column): before
 * 1.1.6 the only way back from Setup was to press Setup again. */
import * as settings from "./settings.js";

export const KEY = "inspector.tier";
export const DETAIL = "inspector.detail";   // "remember" (the table), "basic" or "advanced" for every view
export const TIERS = ["basic", "advanced"];
/* the view kinds that keep their own choice; anything else shares "other" */
const KINDS = new Set(["structure", "trajectory", "freq", "volume", "plot", "runplot", "mech", "db", "editor", "text"]);
export const kindKey = (kind) => (KINDS.has(kind) ? kind : "other");
/* whole Inspector sections (shell.js section keys) that are niche: the classic editing forms */
export const ADVANCED_SECTIONS = new Set(["classic"]);

function table() {
  const v = settings.get(KEY);
  return v && typeof v === "object" && !Array.isArray(v) ? v : {};
}
/* where the user has not chosen: Basic, unless the server says otherwise (MACE_GUI_INSPECTOR_TIER) */
let fallback = "basic";
export function setDefaultTier(t) {
  fallback = TIERS.includes(t) ? t : "basic";
  document.querySelectorAll("#inspector").forEach((h) => { if (h.__tiers) h.__tiers.apply(); });
}
/* "basic" / "advanced" when Inspector detail holds every view there, else null */
export function always() { const d = settings.get(DETAIL); return TIERS.includes(d) ? d : null; }
/* In an Always mode a pick (the switch, View › Advanced Inspector, "N more in Advanced") shows the
 * other tier for the view on screen only: showing another view drops it, and neither the setting
 * nor the per-kind table changes, so the chosen mode is what every next view gets. */
let pick = null;   // {view, tier}
/** "basic" or "advanced" for this view kind (view: the document on screen, for an Always mode's pick). */
export function tierOf(kind, view) {
  const a = always();
  if (a) {
    if (pick && pick.view !== (view ?? null)) pick = null;
    return pick ? pick.tier : a;
  }
  const t = table()[kindKey(kind)];
  return TIERS.includes(t) ? t : fallback;
}
/** Keep tier for this view kind (Settings, so it follows the profile like any other setting);
 * in an Always mode, only for this view until another one shows. */
export function setTier(kind, tier, view) {
  if (!TIERS.includes(tier)) throw new Error(`no inspector tier ${tier}`);
  const a = always();
  if (a) { pick = tier === a ? null : { view: view ?? null, tier }; return; }
  const t = { ...table(), [kindKey(kind)]: tier };
  settings.set(KEY, t);
}

/* An advanced element that is not inside another advanced one (what one click would show). */
function topAdvanced(root) {
  return [...root.querySelectorAll('[data-tier="advanced"]')].filter((el) => {
    const up = el.parentElement && el.parentElement.closest('[data-tier="advanced"]');
    return !up || !root.contains(up);
  });
}
const CONTROL = "input:not([type=hidden]), select, textarea, button:not(.ovr-dot):not(.tier-more)";
/* what an advanced element holds: its form controls, or one row of values; a hint counts none */
const controlsIn = (el) => (el.matches(CONTROL) ? 1 : el.matches("p, .insp-hint") ? 0 : Math.max(1, el.querySelectorAll(CONTROL).length));
/** How many settings and rows Basic hides in root. */
export function hiddenCount(root) { return root ? topAdvanced(root).reduce((n, el) => n + controlsIn(el), 0) : 0; }

const ESC = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

/* The "N more in Advanced" button at the end of each section that hides some (Basic only).
 * Called on every change of the container; it touches the page only when the count changed. */
function syncMore(host, basic, onAdvanced) {
  host.querySelectorAll(".insp-section").forEach((sec) => {
    const body = sec.querySelector(":scope > .insp-body");
    if (!body) return;
    let more = body.querySelector(":scope > .tier-more");
    const n = basic && sec.dataset.tier !== "advanced" ? hiddenCount(body) : 0;
    if (!n) { if (more) more.remove(); return; }
    const text = `${n} more in Advanced`;
    if (more && more.dataset.n === String(n) && more === body.lastElementChild) return;
    if (!more) {
      more = document.createElement("button");
      more.type = "button";
      more.className = "tier-more";
      more.addEventListener("click", () => onAdvanced(sec.dataset.key));
    }
    more.dataset.n = String(n);
    more.textContent = text;
    more.title = "Show the Advanced Inspector: the niche settings of this view";
    body.appendChild(more);
  });
}

/**
 * Wire the tiers into an Inspector container (once): host gets data-insp-tier,
 * kept in step with kind() and the setting; the "more" buttons follow content changes.
 * deps: { kind: () => view kind, view: () => the document on screen, refresh: () => void (after a change of tier) }
 */
export function attach(host, deps) {
  if (!host || host.__tiers) return host && host.__tiers;
  let frame = 0, busy = false;
  const view = () => (deps.view ? deps.view() : undefined);
  const toAdvanced = (key) => {
    setTier(deps.kind(), "advanced", view());
    apply();
    if (deps.refresh) deps.refresh();
    const sec = key ? host.querySelector(`.insp-section[data-key="${key}"]`) : null;
    if (sec) requestAnimationFrame(() => sec.scrollIntoView({ block: "nearest" }));
  };
  function apply() {
    const tier = tierOf(deps.kind(), view());
    if (host.dataset.inspTier !== tier) host.dataset.inspTier = tier;
    // the sections the shell builds that are niche as a whole
    host.querySelectorAll(".insp-section").forEach((sec) => {
      if (ADVANCED_SECTIONS.has(sec.dataset.key) && sec.dataset.tier !== "advanced") sec.dataset.tier = "advanced";
    });
    busy = true;
    try { syncMore(host, tier === "basic", toAdvanced); } finally { busy = false; }
    host.querySelectorAll(".insp-tier [data-tier-pick]").forEach((b) => {
      b.setAttribute("aria-checked", String(b.dataset.tierPick === tier));
    });
    const seg = host.querySelector(".insp-tier");
    if (seg) {
      const n = hiddenCount(host);
      seg.title = tier === "basic" ? (n > 0 ? `Basic: ${n} niche ${n === 1 ? "setting is" : "settings are"} hidden here` : "Basic: every setting of this view shows")
        : "Advanced: every setting, the niche ones too";
      const a = always();
      if (a) seg.title += a === tier ? ` (Settings › Appearance › Inspector detail: Always ${a === "basic" ? "Basic" : "Advanced"})`
        : " (this view only, until another view shows; Settings › Appearance › Inspector detail keeps the rest)";
    }
  }
  const schedule = () => { if (busy) return; cancelAnimationFrame(frame); frame = requestAnimationFrame(apply); };
  new MutationObserver(schedule).observe(host, { childList: true, subtree: true, attributes: true, attributeFilter: ["data-tier"] });
  settings.onChange && settings.onChange((k) => {
    if (k === DETAIL) pick = null;
    if (k === KEY || k === DETAIL) schedule();
  });
  host.addEventListener("click", (e) => {
    const b = e.target.closest(".insp-tier [data-tier-pick]");
    if (!b) return;
    setTier(deps.kind(), b.dataset.tierPick, view());
    apply();
    if (deps.refresh) deps.refresh();
  });
  host.__tiers = { apply, toAdvanced };
  apply();
  return host.__tiers;
}

/**
 * The tab strip on top of the Inspector column: Inspector | Calculation Setup,
 * and (Inspector only) the Basic / Advanced switch.
 * o: { setupOpen, setupWhy: "" or why Setup cannot open here, tier, icon, onInspector(), onSetup() }
 */
export function tabStrip(o) {
  const icon = o.icon || (() => "");
  const el = document.createElement("div");
  el.className = "insp-tabs";
  const why = o.setupWhy || "";
  el.innerHTML = `<span class="insp-tabs-list" role="tablist" aria-label="Inspector column">
      <button type="button" role="tab" class="insp-tab" data-insp-tab="inspector" aria-selected="${!o.setupOpen}"
        data-tip="The Inspector: what you see, and how it is drawn" data-tip-cmd="view.inspectorTab">${icon("panel-right")}<span>Inspector</span></button>
      <button type="button" role="tab" class="insp-tab" data-insp-tab="setup" aria-selected="${!!o.setupOpen}"${why && !o.setupOpen ? ` aria-disabled="true" data-reason="${ESC(why)}"` : ""}
        data-tip="${ESC(why && !o.setupOpen ? `Calculation Setup: ${why}` : "Calculation Setup: the deck for this structure or run")}" data-tip-cmd="calc.setup">${icon("flask-conical")}<span>Setup</span></button></span>` +
    (o.setupOpen ? "" : `<span class="segment insp-tier" role="radiogroup" aria-label="Inspector detail">
      ${TIERS.map((t) => `<button type="button" role="radio" data-tier-pick="${t}" aria-checked="${o.tier === t}">${t === "basic" ? "Basic" : "Advanced"}</button>`).join("")}</span>`);
  el.addEventListener("click", (e) => {
    const t = e.target.closest("[data-insp-tab]");
    if (!t) return;
    if (t.dataset.inspTab === "inspector") { if (o.setupOpen) o.onInspector(); }
    else if (!o.setupOpen && t.getAttribute("aria-disabled") !== "true") o.onSetup();
  });
  // arrow keys move between the two tabs (WAI-ARIA tabs pattern)
  el.addEventListener("keydown", (e) => {
    const tabs = [...el.querySelectorAll("[data-insp-tab]")], i = tabs.indexOf(document.activeElement);
    if (i < 0 || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) return;
    e.preventDefault();
    const j = e.key === "Home" ? 0 : e.key === "End" ? tabs.length - 1 : (i + (e.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
    tabs[j].focus();
  });
  return el;
}
