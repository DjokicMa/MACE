/* Runs before first paint (a classic script in <head>, since the page's CSP
 * allows no inline script): apply the stored theme, density, UI scale and
 * motion so the page does not flash, and start the performance clock. */
(function () {
  var s = {}, pending = [];
  try { s = JSON.parse(localStorage.getItem("mace.gui.settings.v1") || "{}") || {}; } catch (e) { s = {}; }
  try { pending = JSON.parse(localStorage.getItem("mace.gui.settings.pending") || "[]") || []; } catch (e) { pending = []; }
  var root = document.documentElement;
  // The server put the profile's values on <html> (a new port, a fresh browser
  // paint them at once). This browser's own value wins only for a key whose
  // change has not reached the server yet; an older server sends none, and
  // this browser's values apply as before.
  var server = root.getAttribute("data-prefs") === "server";
  var mine = function (k) { return !server || pending.indexOf(k) >= 0; };
  if (mine("theme")) {
    if (s.theme === "light" || s.theme === "dark") root.setAttribute("data-theme", s.theme);
    else if (server) root.removeAttribute("data-theme");
  }
  if (mine("density")) {
    if (s.density === "compact") root.setAttribute("data-density", "compact"); else if (server) root.removeAttribute("data-density");
  }
  if (mine("reduceMotion")) {
    root.removeAttribute("data-motion");
    if (s.reduceMotion === "on") root.setAttribute("data-motion", "reduce");
    if (s.reduceMotion === "off") root.setAttribute("data-motion", "full");
  }
  if (mine("uiScale")) {
    var scale = Number(s.uiScale);
    root.style.setProperty("--ui-scale", String((scale >= 80 && scale <= 150 ? scale : 100) / 100));
  }
  // what the first frame is painted with (the tests read it once the page is up)
  window.__maceFirstPaint = { theme: root.getAttribute("data-theme") || "system", density: root.getAttribute("data-density") || "",
    scale: root.style.getPropertyValue("--ui-scale").trim() || "1", source: server ? "server" : "browser" };
  try { performance.mark("mace:boot"); } catch (e) { /* old browser */ }
  // every console.error and uncaught error is kept, so tests can insist on none
  var errs = window.__maceErrors = [];
  var orig = console.error;
  console.error = function () {
    try { errs.push(Array.prototype.map.call(arguments, String).join(" ")); } catch (e) { /* ignore */ }
    return orig.apply(console, arguments);
  };
}());
