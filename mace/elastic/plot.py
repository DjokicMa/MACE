"""``mace analyze elastic FILE --plot NAME``: the plots of an ELASTIC, EOS, QHA or
CPHF run as png/svg/pdf, drawn with matplotlib from ``views.figure`` (the
arrays the GUI's results document draws with plotly), so Copy CLI Command
gives what the screen shows.

Output: ``<stem>.<NAME>.<fmt>`` beside the output (or in ``-o``); a
directional plot adds its shape (``<stem>.youngs_surface.png``).
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import views as V

FORMATS = ("png", "svg", "pdf")


def mathtext(text: str) -> str:
    """Plot text (plotly's <sub>, <sup>) as matplotlib mathtext: E<sub>min</sub> -> E$_\\mathregular{min}$."""
    t = re.sub(r"<sub>(.*?)</sub>", lambda m: "$_{\\mathregular{" + m.group(1) + "}}$", str(text or ""))
    t = re.sub(r"<sup>(.*?)</sup>", lambda m: "$^{\\mathregular{" + m.group(1) + "}}$", t)
    return re.sub(r"<[^>]+>", "", t).replace("$$", "")


def plot_names(doc: Dict[str, Any]) -> List[str]:
    return [v["key"] for v in doc.get("views") or []]


def render(fig: Dict[str, Any], out_prefix: str, formats=("png",), style: Optional[str] = None,
           dpi: int = 300) -> List[str]:
    """Draw one figure (views.figure) into ``out_prefix.<fmt>``; the files written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mace.plotting import presets
    sid = presets.resolve_id(style or "default", "band")
    with matplotlib.rc_context(presets.mpl_rc(sid, "band")):
        if fig.get("shape") == "surface":
            f = _surface(plt, fig)
        elif fig.get("shape") == "planes":
            f = _planes(plt, fig, presets, sid)
        else:
            f = _lines(plt, fig, presets, sid)
        paths = []
        for fmt in formats:
            p = f"{out_prefix}.{fmt}"
            f.savefig(p, dpi=dpi)
            paths.append(p)
        plt.close(f)
    return paths


def _surface(plt, fig):
    import numpy as np
    from matplotlib import cm, colors
    f = plt.figure(figsize=(6.4, 5.6))
    ax = f.add_subplot(projection="3d")
    env = fig.get("envelopes")
    if env:
        # ELATE's style: the minimum over m opaque green, the maximum translucent blue, negative values red
        r = 0.0
        for key, color, alpha in (("max", "#3b6fb6", 0.22), ("min", "#2e9b4f", 1.0)):
            s = env[key]
            X, Y, Z, Vv = (np.array(s[k], dtype=float) for k in ("x", "y", "z", "value"))
            fc = np.empty(Vv.shape + (4,))
            fc[...] = colors.to_rgba(color, alpha)
            fc[Vv < 0] = colors.to_rgba("#c8382e", alpha)
            ax.plot_surface(X, Y, Z, facecolors=fc, rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)
            r = max(r, float(np.nanmax(np.abs(Vv))))
        from matplotlib.patches import Patch
        handles = [Patch(color="#2e9b4f", label=mathtext(f"{fig['symbol']}<sub>min</sub>")),
                   Patch(color="#3b6fb6", alpha=0.35, label=mathtext(f"{fig['symbol']}<sub>max</sub>"))]
        if fig.get("negative"):
            handles.append(Patch(color="#c8382e", label=mathtext(f"{fig['symbol']} < 0")))
        ax.legend(handles=handles, loc="upper left", fontsize="small", frameon=False)
    else:
        s = fig["surface"]
        X, Y, Z, Vv = (np.array(s[k], dtype=float) for k in ("x", "y", "z", "value"))
        lo, hi = float(np.nanmin(Vv)), float(np.nanmax(Vv))
        neg = fig.get("negative") and lo < 0
        if neg:                                   # a signed quantity: diverging around zero
            m = max(abs(lo), abs(hi)) or 1e-12
            norm = colors.TwoSlopeNorm(vmin=-m, vcenter=0.0, vmax=m)
        else:
            norm = colors.Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1e-12)
        name = "RdBu" if neg else "viridis"
        cmap = cm.get_cmap(name) if hasattr(cm, "get_cmap") else plt.colormaps[name]
        ax.plot_surface(X, Y, Z, facecolors=cmap(norm(Vv)), rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        m.set_array(Vv)
        f.colorbar(m, ax=ax, shrink=0.65, pad=0.08, label=mathtext(fig["label"]))
        r = float(np.nanmax(np.abs(Vv)))
    r = r or 1.0
    for set_lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        set_lim(-r, r)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x"), ax.set_ylabel("y"), ax.set_zlabel("z")
    ax.set_title(mathtext(_subtitle(fig)), fontsize="medium")
    f.tight_layout()
    return f


def _planes(plt, fig, presets, sid):
    import numpy as np
    cuts = fig["planes"]
    f, axes = plt.subplots(1, len(cuts), subplot_kw={"projection": "polar"}, figsize=(max(3.6 * len(cuts) + 0.4, 6.4), 4.4),
                           squeeze=False)
    curves = lambda c: c.get("curves") or [{"name": fig["symbol"], "r": c["r"]}]       # noqa: E731
    rmax = max(abs(x) for c in cuts for k in curves(c) for x in k["r"] if x is not None) * 1.05 or 1.0
    env = fig["view"] in ("shear", "poisson")
    for ax, c in zip(axes[0], cuts):
        th = np.radians(c["theta_deg"])
        for k, cv in enumerate(curves(c)):
            r = np.array([math.nan if x is None else x for x in cv["r"]], dtype=float)
            if env and "curves" in c:
                color = {"min": "#2e9b4f", "max": "#3b6fb6"}[cv["key"]]
                dash = "solid" if cv["key"] == "min" else "dash"
            elif fig["view"] == "piezo":
                color, dash = "#2f6db5", "solid"          # positive blue, negative red (as the surface)
            else:
                st = presets.series_style(sid, "band", k % 3 if fig["view"] == "sound" else k)
                color = st["color"] if len(curves(c)) > 1 else presets.series_style(sid, "band", cuts.index(c))["color"]
                dash = cv.get("dash") or "solid"
            label = mathtext(cv["name"]) if len(curves(c)) > 1 else None
            ax.plot(th, np.abs(r), color=color, linewidth=1.6 if dash == "solid" else 1.2,
                    linestyle=presets.mpl_dash(dash), label=label)
            if (r < 0).any():                   # a negative value (beta, nu, d_L): drawn red, |value| from the center
                ax.plot(th, np.where(r < 0, np.abs(r), np.nan), color="#c8382e", linewidth=1.6,
                        linestyle=presets.mpl_dash(dash))
        ax.set_ylim(0, rmax)
        ax.set_rlabel_position(67.5)               # the radial labels off the 0-degree axis label
        title = f"{c['plane']} plane" if len(c["plane"]) == 2 else f"plane {c['plane']}: 0° {c['axes'][0]}, 90° {c['axes'][1]}"
        ax.set_title(title, fontsize="medium", pad=24)
        ax.tick_params(labelsize="small")
    if "curves" in cuts[0]:
        axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), fontsize="small", frameon=False)
    f.suptitle(mathtext(f"{fig['title']}: {_subtitle(fig)}"), fontsize="medium")
    f.tight_layout()
    return f


def _lines(plt, fig, presets, sid):
    f, ax = plt.subplots(figsize=(7.2, 4.4))
    k = 0
    for s in fig["series"]:
        xs = s["x"]
        ys = [math.nan if v is None else v for v in s["y"]]
        if s.get("role") == "points":
            if s.get("hollow"):                   # the computed volumes of an F(V) family
                ax.plot(xs, ys, linestyle="none", marker="o", markerfacecolor="none", color="0.45", markersize=5,
                        label=mathtext(s["name"]), zorder=4)
            else:
                ax.plot(xs, ys, linestyle="-" if s.get("line") else "none", linewidth=0.9, marker="o", color="0.15",
                        markersize=5, label=mathtext(s["name"]), zorder=5)
            continue
        if s.get("key") == "reference":           # the enthalpy plot's zero: this run
            ax.plot(xs, ys, color="0.45", linewidth=1.2, linestyle=":", label=mathtext(s["name"]))
            continue
        st = presets.series_style(sid, "band", k)
        k += 1
        dash = s.get("dash") or st["dash"]
        ax.plot(xs, ys, color=st["color"], linewidth=1.5, linestyle=presets.mpl_dash(dash), label=mathtext(s["name"]),
                marker="o" if s.get("marker") else None, markersize=4)
    ax.set_xlabel(mathtext(fig["x_label"]))
    ax.set_ylabel(mathtext(fig["y_label"]))
    if fig.get("exponent"):
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    if fig.get("view") == "pv" or fig.get("zero_line"):
        ax.axhline(0.0, color="0.6", linewidth=0.8)
    for x in fig.get("markers") or []:            # transition pressures
        ax.axvline(x, color="0.35", linewidth=0.9, linestyle=":")
        ax.annotate(mathtext(f"P<sub>t</sub> = {x:.2f} GPa"), (x, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -12),
                    textcoords="offset points", fontsize="small", color="0.25")
    ax.grid(True, alpha=0.3)
    if len(fig["series"]) > 1:
        ax.legend(fontsize="small", frameon=False)
    ax.set_title(mathtext(fig.get("title", "")), fontsize="medium")
    f.tight_layout()
    return f


def _subtitle(fig: Dict[str, Any]) -> str:
    ex = fig.get("extrema") or {}
    u = f" ({fig['unit']})" if fig.get("unit") else ""
    if ex.get("isotropic"):
        return f"isotropic, {ex['max']['value']:g}{' ' + fig['unit'] if fig.get('unit') else ''}"
    if ex and ex.get("min"):
        if fig.get("view") == "piezo":
            mx = ex["max"]
            return f"largest |{fig['symbol']}| {abs(mx['value']):.4g} along {mx['label']}{u}"
        fmt = lambda e: f"{e['value']:.4g} {e['label']}" + (f" / {e['label2']}" if e.get("label2") else "")   # noqa: E731
        return f"min {fmt(ex['min'])}, max {fmt(ex['max'])}{u}".replace("-", "−")
    if fig.get("view") == "sound":
        return "phase (solid) and group (dotted) velocities (km/s)"
    return fig.get("title", "")


def plot_file(path: Path, names: List[str], shape: str = "surface", out_dir: Optional[str] = None,
              formats=("png",), style: Optional[str] = None, compare=None, fit: Optional[str] = None) -> List[str]:
    """Every requested plot of one output ("all": each plot it has); the files written. ``compare``:
    [(label, analyze() of another EOS run)] for the enthalpy plot."""
    text = Path(path).read_text(errors="replace")
    doc = V.document(text, Path(path))
    have = plot_names(doc)
    wanted = have if "all" in names else names
    if not have:
        raise ValueError(f"{Path(path).name}: no plots in this output"
                         + (f" (a {doc['kind']} run: its results are tables only)" if doc.get("kind") else ""))
    for name in wanted:                         # every name checked before anything is written
        if name not in have:
            raise ValueError(f"{Path(path).name}: this output has no {V.plain(V.PLOTS[name][1])} plot (it has {', '.join(have)})")
    dest = Path(out_dir) if out_dir else Path(path).parent
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for name in wanted:
        fig = V.figure(doc["data"], name, text, shape, compare, fit, Path(path).name)
        tag = f"{name}_{fig['shape']}" if fig.get("shape") else name
        written += render(fig, str(dest / f"{Path(path).stem}.{tag}"), formats, style)
    return written
