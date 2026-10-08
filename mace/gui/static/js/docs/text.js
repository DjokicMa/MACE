/* "Open as Text" for any size of file (spec §8.4.6, plan B8): a virtual list
 * of lines that fetches 256 KB windows as it scrolls. The server's sparse
 * line index (where every 4096th line starts) lets a jump to any line cost
 * one request; Find in File streams the file on the server, as a task.
 * At most ~80 line rows are in the page, whatever the file's size. */
const LH = 18, OVERSCAN = 12, MAX_PX = 8e6, MAX_BLOCKS = 48;

export function createTextView(host, H) {
  const { api, q, esc, highlight, apiTask, toast } = H;
  host.innerHTML = `<div class="tv" role="document">
    <div class="tv-bar"><input class="tv-find" type="search" placeholder="Find in file, or :line" aria-label="Find in file, or a line number after a colon">
      <span class="tv-count muted small" aria-live="polite"></span><div class="tv-hits" role="listbox" hidden></div>
      <span class="tv-live" hidden><span class="live-chip" role="status" aria-live="polite"></span>
        <label class="tog"><input type="checkbox" class="tv-follow" checked><span>Follow (tail)</span></label></span></div>
    <div class="tv-replaced" role="alert" hidden><span>This output was replaced (a new run started)</span> <button class="tb-btn-text" data-act="reload">Reload</button></div>
    <div class="tv-scroll" tabindex="0" aria-label="File text"><div class="tv-space"></div><div class="tv-rows" role="presentation"></div></div></div>`;
  const scroller = host.querySelector(".tv-scroll"), space = host.querySelector(".tv-space"), rows = host.querySelector(".tv-rows");
  const findBox = host.querySelector(".tv-find"), count = host.querySelector(".tv-count"), hits = host.querySelector(".tv-hits");
  let doc = null, frame = 0, mark = null;
  const stats = { requests: 0, bytes: 0 };

  function total() { return doc ? (doc.lines || doc.estLines || 0) + (doc.follow ? doc.follow.tail.length : 0) : 0; }
  function scaled() { return total() * LH > MAX_PX; }
  function firstLine() {
    const vis = Math.ceil(scroller.clientHeight / LH);
    if (!scaled()) return Math.floor(scroller.scrollTop / LH);
    const maxTop = Math.max(1, space.offsetHeight - scroller.clientHeight);
    return Math.round((scroller.scrollTop / maxTop) * Math.max(0, total() - vis));
  }
  function topFor(line) {
    if (!scaled()) return line * LH;
    const vis = Math.ceil(scroller.clientHeight / LH);
    const maxTop = Math.max(1, space.offsetHeight - scroller.clientHeight);
    return (line / Math.max(1, total() - vis)) * maxTop;
  }

  /* blocks of STEP lines, filled by windows */
  function block(b) {
    let bl = doc.blocks.get(b);
    if (!bl) { bl = { lines: [], next: null, loading: null, done: false }; doc.blocks.set(b, bl); }
    doc.blocks.delete(b); doc.blocks.set(b, bl);                 // most recently used last
    while (doc.blocks.size > MAX_BLOCKS) doc.blocks.delete(doc.blocks.keys().next().value);
    return bl;
  }
  function blockStart(b) {
    if (b === 0) return 0;
    return doc.offsets && b < doc.offsets.length ? doc.offsets[b] : null;
  }
  function load(b) {
    const bl = block(b);
    if (bl.done || bl.loading) return;
    const off = bl.next !== null ? bl.next : blockStart(b);
    if (off === null) return;                                     // the line index is not here yet
    const d = doc;
    stats.requests++;
    bl.loading = api("text?" + q({ path: d.srcPath || d.path, offset: off })).then((w) => {
      if (doc !== d) return;
      stats.bytes += w.text.length;
      const lines = w.text.split("\n");
      if (lines.length && lines[lines.length - 1] === "" && !w.cut_line) lines.pop();
      const want = d.step || 4096;
      for (const ln of lines) { if (bl.lines.length >= want) break; bl.lines.push(ln); }
      bl.next = w.end;
      bl.done = bl.lines.length >= want || w.eof;
      if (w.eof && !d.lines) { d.estLines = (b * want) + bl.lines.length; size(); }
      if (b === 0 && !d.lines && !w.eof) { d.estLines = Math.max(d.estLines || 0, Math.round(d.size / Math.max(1, w.end) * bl.lines.length)); size(); }
    }).catch((e) => toast(e.message, "error")).finally(() => { bl.loading = null; paint(); });
  }
  function lineText(n) {
    if (doc.follow && doc.lines && n >= doc.lines) return doc.follow.tail[n - doc.lines] ?? null;
    const step = doc.step || 4096, b = Math.floor(n / step), k = n - b * step;
    const bl = block(b);
    if (k < bl.lines.length) return bl.lines[k];
    load(b);
    return null;
  }
  function size() { space.style.height = `${Math.min(MAX_PX, Math.max(1, total()) * LH)}px`; }
  function paint() {
    frame = 0;
    if (!doc) return;
    const first = Math.max(0, firstLine() - OVERSCAN), n = Math.ceil(scroller.clientHeight / LH) + OVERSCAN * 2;
    const last = Math.min(total(), first + n);
    const gut = String(Math.max(1, total())).length;
    let html = "";
    for (let i = first; i < last; i++) {
      const t = lineText(i);
      const body = t === null ? '<span class="tv-wait">…</span>' : (doc.hl ? highlight(t) : esc(t));
      html += `<div class="tv-line${mark === i ? " hit" : ""}" data-line="${i}"><span class="tv-ln">${String(i + 1).padStart(gut, " ")}</span><span class="tv-t">${body}</span></div>`;
    }
    rows.innerHTML = html;
    rows.style.transform = `translateY(${topFor(first) + (scaled() ? 0 : 0)}px)`;
    window.__maceTextRows = Math.max(window.__maceTextRows || 0, rows.childElementCount);
  }
  scroller.addEventListener("scroll", () => { if (!frame) frame = requestAnimationFrame(paint); }, { passive: true });
  new ResizeObserver(() => { if (!frame) frame = requestAnimationFrame(paint); }).observe(scroller);

  function jump(line) {
    if (!doc) return;
    line = Math.max(0, Math.min(total() - 1, line));
    mark = line;
    scroller.scrollTop = Math.max(0, topFor(line) - scroller.clientHeight / 3);
    paint();
  }
  async function findInFile(text) {
    if (!doc || !text) return;
    const m = /^:(\d+)$/.exec(text.trim());
    if (m) { jump(Number(m[1]) - 1); return; }
    count.textContent = "Searching…";
    try {
      const r = await apiTask("text/find", { path: doc.srcPath || doc.path, query: text, doc: doc.id });
      count.textContent = r.matches.length ? `${r.matches.length}${r.more ? "+" : ""} line${r.matches.length === 1 ? "" : "s"}` : "No match";
      hits.innerHTML = r.matches.slice(0, 200).map((h, k) => `<button role="option" data-hit="${h.line}" ${k === 0 ? 'aria-selected="true"' : ""}>` +
        `<span class="tv-ln">${h.line + 1}</span><span>${esc(h.text.slice(0, 160))}</span></button>`).join("");
      hits.hidden = !r.matches.length;
      if (r.matches.length) jump(r.matches[0].line);
      doc.lastFind = r;
    } catch (e) { count.textContent = ""; if (e.state !== "canceled") toast(e.message, "error"); }
  }
  findBox.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); findInFile(findBox.value); }
    if (e.key === "Escape") { hits.hidden = true; scroller.focus(); }
  });
  hits.addEventListener("click", (e) => { const b = e.target.closest("[data-hit]"); if (b) { jump(Number(b.dataset.hit)); hits.hidden = true; } });
  scroller.addEventListener("keydown", (e) => {
    const page = Math.floor(scroller.clientHeight / LH) - 1, cur = firstLine();
    const to = { PageDown: cur + page, PageUp: cur - page, Home: 0, End: total() }[e.key];
    if (to !== undefined && (e.key.startsWith("Page") || e.ctrlKey || e.metaKey)) { e.preventDefault(); jump(to); }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "f") { e.preventDefault(); findBox.focus(); findBox.select(); }
  });

  /* ------------------------------------------------ Follow (PLAN_analysis C18.5)
   * A file with no terminal marker: the whole lines written after the last
   * read (text?since=, with the file's snapshot) every 15 s while live, every
   * 60 s while it looks stalled, until CRYSTAL's end marker appears. It
   * scrolls to the end on new lines unless the user scrolled up. */
  const live = host.querySelector(".tv-live"), chip = host.querySelector(".tv-live .live-chip");
  const followBox = host.querySelector(".tv-follow"), replacedBar = host.querySelector(".tv-replaced");
  const ago = (s) => (s < 90 ? `${Math.round(s)} s ago` : s < 5400 ? `${Math.round(s / 60)} min ago` : `${(s / 3600).toFixed(1)} h ago`);
  const atEnd = () => scroller.scrollTop + scroller.clientHeight >= scroller.scrollHeight - LH * 2;
  function showChip(f) {
    live.hidden = !f;
    if (!f) return;
    chip.className = "live-chip" + (f.finished ? " done" : f.live ? " on" : " stalled");
    chip.textContent = f.finished ? "Finished" : !followBox.checked ? "Follow off"
      : f.paused ? "Paused - scroll to the end to follow" : f.live ? `Live · updated ${ago(f.age_s || 0)}` : `No end marker; last write ${ago(f.age_s || 0)}`;
  }
  function followSchedule(d, ms) {
    clearTimeout(d.follow.timer);
    d.follow.timer = setTimeout(() => followPoll(d), ms ?? (d.follow.live ? 15000 : 60000) * (window.__maceLiveScale || 1));
  }
  async function followPoll(d) {
    const f = d.follow;
    if (!f || doc !== d || f.finished) return;
    if (document.hidden) { followSchedule(d); return; }
    try {
      const r = await api("text?" + q({ path: d.srcPath || d.path, since: f.end, snap: f.snap }));
      if (doc !== d) return;
      if (r.replaced) { replacedBar.hidden = false; f.finished = true; showChip(null); return; }
      const wasEnd = atEnd();
      if (r.text) {
        const lines = r.text.split("\n");
        if (lines[lines.length - 1] === "") lines.pop();
        f.tail.push(...lines);
        size();
      }
      f.end = r.end; f.snap = r.snap; f.live = r.live; f.finished = r.finished; f.age_s = r.age_s; f.polls = (f.polls || 0) + 1;
      f.paused = followBox.checked && !wasEnd && !!r.text;
      if (r.text && followBox.checked && wasEnd) { scroller.scrollTop = scroller.scrollHeight; }
      paint();
      showChip(f);
      if (!f.finished) followSchedule(d, r.more ? 300 : undefined);
    } catch (e) { followSchedule(d); }
  }
  followBox.addEventListener("change", () => { if (doc && doc.follow) { showChip(doc.follow); if (followBox.checked) { scroller.scrollTop = scroller.scrollHeight; } } });
  scroller.addEventListener("scroll", () => { if (doc && doc.follow && doc.follow.paused && atEnd()) { doc.follow.paused = false; showChip(doc.follow); } }, { passive: true });
  replacedBar.addEventListener("click", (e) => {
    if (!e.target.closest("[data-act=reload]") || !doc) return;
    const d = doc;
    replacedBar.hidden = true;
    if (d.follow) clearTimeout(d.follow.timer);
    Object.assign(d, { lines: null, offsets: null, blocks: new Map(), follow: null, linesLoading: null, estLines: 0, scroll: 0 });
    api_open(d);
  });
  let api_open = null;

  return {
    async open(d) {
      api_open = (x) => this.open(x);
      doc = d;
      mark = null;
      if (!d.blocks) d.blocks = new Map();
      hits.hidden = true; count.textContent = ""; findBox.value = "";
      size();
      scroller.scrollTop = d.scroll || 0;
      performance.mark("text:open:start");
      load(0);
      replacedBar.hidden = true;
      showChip(d.follow || null);
      if (!d.lines && !d.linesLoading) {
        d.linesLoading = api("text/lines?" + q({ path: d.srcPath || d.path })).then((r) => {
          stats.requests++;
          d.lines = r.lines; d.offsets = r.offsets; d.step = r.step; d.size = r.size;
          if (!r.finished && (d.srcPath || d.path)) {
            // still being written: show whole lines only, and follow the rest
            d.lines = r.complete_lines;
            d.follow = { end: r.complete_end, snap: r.snap, tail: [], live: r.live, age_s: r.age_s, finished: false, timer: 0 };
            if (doc === d) { showChip(d.follow); followSchedule(d); }
          }
          if (doc === d) { size(); paint(); }
        }).catch(() => {});
      } else if (d.follow && !d.follow.finished) followSchedule(d, 0);
      paint();
    },
    close() { if (doc) { doc.scroll = scroller.scrollTop; if (doc.follow) clearTimeout(doc.follow.timer); } doc = null; },
    follow: () => (doc && doc.follow ? { ...doc.follow, tail: doc.follow.tail.length, timer: undefined } : null),
    jump, findInFile, stats: () => ({ ...stats, rows: rows.childElementCount, maxRows: window.__maceTextRows || 0 }),
    lineAt: (n) => (doc ? lineText(n) : null), total,
  };
}
