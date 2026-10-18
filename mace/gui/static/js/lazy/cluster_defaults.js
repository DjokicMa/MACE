/* Settings › Advanced: cluster defaults (1.1.6). The site settings MACE writes into job
 * scripts, as `mace site` shows them (every value with where it comes from), and edits
 * made by running the CLI itself: `mace site set KEY VALUE`, `mace site unset KEY` and
 * `mace site --init` (site_routes.py). Its words are the answer; nothing is written while
 * MACE GUI is open to the network. */
let H = null, host = null, data = null, last = null, busy = false;

export async function mount(el, handle) {
  H = handle; host = el;
  host.addEventListener("click", onClick);
  host.addEventListener("keydown", (e) => { if (e.key === "Enter" && e.target.id === "cd-value") { e.preventDefault(); edit("set"); } });
  await load();
}

async function load() {
  try { data = await H.api("site/settings"); } catch (e) { data = { error: e.message }; }
  render();
}

function render() {
  if (!host || !host.isConnected) return;
  const { esc, icon } = H;
  if (!data) { host.innerHTML = `<p class="muted">Loading…</p>`; return; }
  if (data.error) { host.innerHTML = `<p class="set-note" role="alert">${esc(data.error)}</p>`; return; }
  const off = !!data.network;
  const key = (host.querySelector("#cd-key") || {}).value || "", val = (host.querySelector("#cd-value") || {}).value || "";
  const dis = off || busy ? " disabled" : "";
  host.innerHTML = `<p class="set-folder-file">What MACE writes into every job script it makes, here and in the CLI.
      ${data.exists ? `Your site settings: <code>${esc(data.path || "")}</code>.` : `There is no <code>${esc(data.path || "site.toml")}</code> yet: every value is built in.`}
      Each change runs <code>mace site</code> itself.</p>` +
    (off ? `<p class="set-note" role="note">Changing the cluster defaults is turned off while MACE GUI is open to the network. They can still be read here; change them with <code>mace site set</code> in a shell.</p>` : "") +
    `<h4 class="set-sub" id="cd-now">Now in effect <span class="muted small">(<code>${esc(data.command)}</code>; the origin of each value is in brackets)</span></h4>
    <pre class="cd-text mono" tabindex="0" aria-labelledby="cd-now" data-part="site-text">${esc(data.text || "")}</pre>
    <h4 class="set-sub">Change a setting</h4>
    <div class="cd-form" role="group" aria-label="Change a site setting">
      <label class="f" for="cd-key">Key<input class="field mono" id="cd-key" list="cd-keys" value="${esc(key)}" placeholder="resources.ntasks" autocomplete="off" spellcheck="false"${dis}></label>
      <datalist id="cd-keys">${(data.keys || []).map((k) => `<option value="${esc(k)}"></option>`).join("")}</datalist>
      <label class="f" for="cd-value">Value<input class="field mono" id="cd-value" value="${esc(val)}" placeholder="16" autocomplete="off" spellcheck="false"${dis}></label>
      <div class="cd-acts"><button type="button" class="tb-btn-text primary" data-cd="set"${dis}>${icon("check")}<span>Set</span></button>
        <button type="button" class="tb-btn-text" data-cd="unset"${dis}>${icon("rotate-ccw")}<span>Unset</span></button></div>
    </div>
    <p class="muted small">Set writes <code>mace site set KEY VALUE</code> into your site.toml (created when absent, its comments kept); Unset removes the line.</p>
    ${data.exists ? "" : `<div class="set-actions"><button type="button" class="tb-btn-text" data-cd="init"${dis}>${icon("plus")}<span>Create site.toml</span></button>
      <span class="muted small">Runs <code>mace site --init</code>: every setting commented out, ready to edit.</span></div>`}
    <div class="cd-result" aria-live="polite" data-part="site-result">${last ? resultHtml(last) : ""}</div>`;
}

function resultHtml(r) {
  const { esc } = H;
  return `<p class="${r.ok ? "cd-ok" : "cd-bad"}"><code>${esc(r.command || "")}</code>: ${r.ok ? "done" : `the CLI refused (exit ${esc(String(r.code))})`}</p>` +
    (r.output ? `<pre class="cd-out mono">${esc(r.output)}</pre>` : "");
}

async function edit(action) {
  if (busy || !data || data.network) return;
  const key = (host.querySelector("#cd-key") || {}).value || "", value = (host.querySelector("#cd-value") || {}).value || "";
  busy = true; render();
  try {
    const r = await H.api("site/edit", { body: { action, key: key.trim(), value: value.trim(), root_gen: data.root_gen } });
    last = r;
    data = r.settings || data;
    H.toast(r.ok ? `${r.command}: done` : `${r.command}: refused`, r.ok ? "success" : "error");
  } catch (e) {
    last = { ok: false, command: action === "init" ? "mace site --init" : `mace site ${action} ${key.trim()}`, code: "-", output: e.message };
    H.toast(e.message, "error");
  } finally {
    busy = false;
    render();
  }
}

function onClick(e) {
  const b = e.target.closest("[data-cd]");
  if (b && !b.disabled) edit(b.dataset.cd);
}
