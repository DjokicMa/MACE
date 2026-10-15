"""Where a plot's legends, colorbars and annotation boxes sit, checked on the
drawn figure: never over the data, a tick label, an axis title, each other, a
subtitle, or past the figure's edge.

One rule for every plot MACE draws:

* plotly (the GUI's plots, and the HTML/PNG/SVG the CLI's cube and FREQ
  engines write): ``PLOTLY_BOXES`` (JavaScript, run in a browser on a drawn
  graph div) returns the boxes, ``plotly_problems`` judges them;
* matplotlib (``mace plotting --band/--dos/--ir/--raman``):
  ``mpl_problems(fig)`` judges a saved figure.

An annotation anchored to the data on both axes (the corner labels of a 2D
map) labels the data and may sit on it; every other annotation is a box that
must stay out of the plot areas.
"""

# The boxes of a drawn plotly graph div, in viewport pixels. arguments[0] is the
# graph div (or a selector), arguments[1] an optional subtitle element (or selector).
PLOTLY_BOXES = r"""
const pick = (x) => (typeof x === 'string' ? document.querySelector(x) : x);
const p = pick(arguments[0]), sub = arguments.length > 1 ? pick(arguments[1]) : null;
if (!p || !p._fullLayout) return null;
const L = p._fullLayout, svg = p.querySelector('.main-svg'), fr = svg.getBoundingClientRect(), z = fr.width / L.width;
const box = (b) => [b.left, b.top, b.right, b.bottom];
const real = (b) => b[2] - b[0] > 0.5 && b[3] - b[1] > 0.5;
const plots = [];
for (const k of Object.keys(L._plots || {})) {
  const sp = L._plots[k], xa = sp.xaxis, ya = sp.yaxis;
  if (!xa || !ya || !xa._length || !ya._length || xa.visible === false || ya.visible === false) continue;
  if (xa.overlaying || ya.overlaying) continue;
  plots.push({id: k, locked: !!(xa.scaleanchor || ya.scaleanchor), box: [fr.left + z * xa._offset, fr.top + z * ya._offset, fr.left + z * (xa._offset + xa._length), fr.top + z * (ya._offset + ya._length)]});
}
for (const k of Object.keys(L).filter((k) => /^scene\d*$/.test(k))) {
  const d = L[k].domain, s = L._size;
  plots.push({id: k, box: [fr.left + z * (s.l + d.x[0] * s.w), fr.top + z * (s.t + (1 - d.y[1]) * s.h), fr.left + z * (s.l + d.x[1] * s.w), fr.top + z * (s.t + (1 - d.y[0]) * s.h)]});
}
const legends = [...p.querySelectorAll('.infolayer > g')].filter((g) => /^legend\d*$/.test(g.getAttribute('class')))
  .map((g) => ({kind: 'legend', id: g.getAttribute('class'), box: box((g.querySelector('rect.bg') || g).getBoundingClientRect()),
    texts: [...g.querySelectorAll('text[class^=legend][class$=text]')].map((t) => t.textContent)}))
  .filter((l) => real(l.box) && l.texts.length);
const colorbars = [...p.querySelectorAll('.infolayer g.colorbar')].map((g, i) => ({kind: 'colorbar', id: 'colorbar' + i,
  box: box(g.getBoundingClientRect()), texts: []})).filter((c) => real(c.box));
const anns = [...p.querySelectorAll('.infolayer .annotation')].map((g) => {
  const i = Number(g.getAttribute('data-index')), a = (L.annotations || [])[i] || {};
  const dataRef = (r) => r && !/paper|domain/.test(r);
  const t = g.querySelector('.annotation-text-g') || g;
  return {kind: 'annotation', id: 'annotation' + i, box: box(t.getBoundingClientRect()), texts: [String(a.text || '')],
          data: dataRef(a.xref || 'x') && dataRef(a.yref || 'y')};
}).filter((a) => real(a.box) && a.texts[0].replace(/<[^>]*>/g, '').trim());
const labels = [...p.querySelectorAll('.cartesianlayer g[class$=tick] > text, .cartesianlayer g[class$=tick2] > text')]
  .concat([...p.querySelectorAll('.infolayer g[class^=g-][class$=title] text')])
  .filter((t) => t.textContent.replace(/[\s ​]/g, '') && getComputedStyle(t).display !== 'none' && t.getAttribute('opacity') !== '0')
  .map((t) => ({text: t.textContent, box: box(t.getBoundingClientRect())})).filter((l) => real(l.box));
const controls = [...p.querySelectorAll('.menulayer > g, .slider-container')].map((g) => box(g.getBoundingClientRect())).filter(real);
let note = null;
if (sub) {
  const nb = sub.getBoundingClientRect(), lh = parseFloat(getComputedStyle(sub).lineHeight) || 1.4 * parseFloat(getComputedStyle(sub).fontSize);
  note = {box: box(nb), lines: Math.round(nb.height / lh), text: sub.textContent};
}
return {fig: box(fr), plots, boxes: [...legends, ...colorbars, ...anns], labels, controls, note};
"""


def hit(a, b, pad=0.0):
    """Two [left, top, right, bottom] boxes overlap by more than pad pixels."""
    return a[0] < b[2] - pad and b[0] < a[2] - pad and a[1] < b[3] - pad and b[1] < a[3] - pad


def _name(b):
    t = ", ".join(x for x in b["texts"] if x)[:60]
    return f"{b['kind']} {t!r}" if t else b["kind"]


def plotly_problems(st, pad=1.5, min_frac=0.2):
    """What PLOTLY_BOXES found out of place, as readable strings ([] = all clear)."""
    if not st:
        return ["no figure"]
    out = []
    boxes = st["boxes"]
    for b in boxes:
        if not b.get("data"):
            for pl in st["plots"]:
                if hit(b["box"], pl["box"], pad):
                    out.append(f"{_name(b)} {b['box']} over plot area {pl['id']} {pl['box']}")
        for lab in st["labels"]:
            if hit(b["box"], lab["box"], pad):
                out.append(f"{_name(b)} over label {lab['text']!r}")
        for c in st["controls"]:
            if hit(b["box"], c, pad):
                out.append(f"{_name(b)} over a control (buttons/slider)")
        if st.get("note") and hit(b["box"], st["note"]["box"]):
            out.append(f"{_name(b)} over the subtitle")
        f = st["fig"]
        if b["box"][0] < f[0] - 1 or b["box"][2] > f[2] + 1 or b["box"][1] < f[1] - 1 or b["box"][3] > f[3] + 1:
            out.append(f"{_name(b)} {b['box']} cut by the figure edge {f}")
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            if hit(a["box"], b["box"], pad):
                out.append(f"{_name(a)} and {_name(b)} overlap")
    # the margins a legend takes leave the data most of the figure: not a sliver
    if st["plots"] and not any(p.get("locked") for p in st["plots"]):      # equal axes (a map, a slice) fill one side
        f = st["fig"]
        w = max(p["box"][2] for p in st["plots"]) - min(p["box"][0] for p in st["plots"])
        h = max(p["box"][3] for p in st["plots"]) - min(p["box"][1] for p in st["plots"])
        # (a plot locked to equal axes, a map or a slice, fills one dimension only)
        fw, fh = w / max(1, f[2] - f[0]), h / max(1, f[3] - f[1])
        if max(fw, fh) < 0.5 or min(fw, fh) < min_frac:
            out.append(f"the plot areas take {w:.0f} x {h:.0f} px of a {f[2] - f[0]:.0f} x {f[3] - f[1]:.0f} px figure")
    return out


def mpl_problems(fig, pad=0.5):
    """The same rule on a matplotlib figure: its legends and boxed texts against every
    axes' data area, tick labels, axis labels and titles, each other and the figure edge."""
    if not hasattr(fig.canvas, "get_renderer"):
        # a Figure made without pyplot has a bare FigureCanvasBase, which cannot render
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        FigureCanvasAgg(fig)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    axes = [ax for ax in fig.axes if ax.get_visible()]
    items = []
    for ax in axes:
        lg = ax.get_legend()
        if lg is not None and lg.get_visible():
            items.append(("legend " + ",".join(t.get_text() for t in lg.get_texts())[:50], lg.get_window_extent(r), False))
        for t in ax.texts:
            if t.get_visible() and t.get_bbox_patch() is not None and t.get_text().strip():
                # a boxed note: placed in data coordinates on both axes it labels the data
                data = t.get_transform() == ax.transData
                items.append((f"text box {t.get_text()[:40]!r}", t.get_window_extent(r), data))
    for lg in fig.legends:
        items.append(("figure legend", lg.get_window_extent(r), False))
    labels = []
    for ax in axes:
        for t in [*ax.get_xticklabels(), *ax.get_yticklabels(), ax.xaxis.label, ax.yaxis.label, ax.title]:
            if t.get_visible() and t.get_text().strip():
                labels.append((t.get_text(), t.get_window_extent(r)))
    out = []
    fb = fig.bbox

    def over(a, b):
        return a.x0 < b.x1 - pad and b.x0 < a.x1 - pad and a.y0 < b.y1 - pad and b.y0 < a.y1 - pad

    for name, bb, data in items:
        if not data:
            for ax in axes:
                if over(bb, ax.get_window_extent(r)):
                    out.append(f"{name} over the axes {ax.get_title() or ax.get_ylabel()!r}")
        for txt, lb in labels:
            if over(bb, lb):
                out.append(f"{name} over label {txt!r}")
        if bb.x0 < fb.x0 - 0.5 or bb.x1 > fb.x1 + 0.5 or bb.y0 < fb.y0 - 0.5 or bb.y1 > fb.y1 + 0.5:
            out.append(f"{name} cut by the figure edge")
    for i, (n1, b1, _) in enumerate(items):
        for n2, b2, _ in items[i + 1:]:
            if over(b1, b2):
                out.append(f"{n1} and {n2} overlap")
    return out
