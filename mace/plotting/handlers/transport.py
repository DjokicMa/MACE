"""BOLTZTRA transport plots for mace plotting (``--transport``).

Any of a run's ``.SEEBECK/.SIGMA/.SIGMAS/.KAPPA/.TDF.dat`` files (any case) draws
any quantity the run's files beside it allow: the raw ones, and PF, κₑ, ZT, the
Lorenz number, σ/τ and PF/τ derived from them on the run's own (T, μ) grid
(``--quantity``). The numbers and the curves are mace.utils.transport_plot's,
which the GUI's transport plot draws too (``traces``), so Publication Figure and
Copy CLI Command give what the screen shows.

Output: matplotlib png/svg/pdf (default png and svg), ``<stem>.<QUANTITY>.<fmt>``
beside the data (or in ``-o``).

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import outcome
from ..prompts import configure_output_formats, select_option
from ..registry import PlotKind, PlotterEntry, register
from mace.utils import plot_labels as L

_IMAGE_FORMATS = ("png", "svg", "pdf")
PATTERNS = [f"*.{q}.{ext}" for q in ("SEEBECK", "SIGMA", "SIGMAS", "KAPPA", "TDF") for ext in ("dat", "DAT")]


def _formats(args) -> List[str]:
    fmt = getattr(args, "format", None)
    if fmt in _IMAGE_FORMATS:
        return [fmt]
    if fmt == "html":
        print("  Transport plots are drawn as png/svg/pdf; html is drawn as png.")
        return ["png"]
    return ["png", "svg"]


def configure_transport(interactive: bool = True) -> Dict[str, Any]:
    config: Dict[str, Any] = {"quantity": None, "component": "avg", "temperature": None, "x": "mu",
                              "kappa_lattice": None, "tau_fs": None, "formats": ["png", "svg"]}
    if not interactive:
        return config
    from mace.utils import transport_plot as T
    print("\n" + "-" * 50)
    print("  TRANSPORT (BOLTZTRA) PLOT CONFIGURATION")
    print("-" * 50)
    names = ["each file's own"] + list(T.CLI_NAMES)
    pick = select_option("Which quantity?", [n if i == 0 else f"{n}: {L.plain(T.QUANTITIES[T.CLI_NAMES[n]]['label'])}"
                                             for i, n in enumerate(names)], default=1)
    config["quantity"] = None if pick == 1 else names[pick - 1]
    if config["quantity"] == "zt":
        kl = input("  Lattice thermal conductivity κ_L in W/(m·K), or a T,kappa_L CSV "
                   "(blank: ZTₑ with κ_L = 0, an upper bound): ").strip()
        config["kappa_lattice"] = kl or None
    config["formats"] = configure_output_formats(interactive=True)
    return config


def _config_from_args(args) -> Dict[str, Any]:
    cfg = configure_transport(interactive=False)
    cfg.update({"quantity": getattr(args, "quantity", None), "component": getattr(args, "component", None) or "avg",
                "temperature": getattr(args, "temperature", None), "x": getattr(args, "x_axis", None) or "mu",
                "kappa_lattice": getattr(args, "kappa_lattice", None), "tau_fs": getattr(args, "tau_fs", None),
                "style": getattr(args, "style", None) or "default", "formats": _formats(args),
                "width": getattr(args, "width", None), "height": getattr(args, "height", None),
                "dpi": getattr(args, "dpi", None)})
    return cfg


def _jobs(files: List[str], quantity: Optional[str]):
    """(file, quantity) per figure: with --quantity one figure per run (the first of its files given),
    else one per file, its own quantity."""
    from mace.utils import transport_plot as T
    seen, jobs = set(), []
    for f in files:
        p = Path(f)
        try:
            stem, own = T.split_name(p)
        except T.TransportError as e:
            print(f"  {e}")
            outcome.failed(p.name, "not a BOLTZTRA .dat file")
            continue
        key = (str(p.parent.resolve()), stem, quantity or own)
        if key in seen:
            continue
        seen.add(key)
        jobs.append((p, quantity or own))
    return jobs


def plot_transport(files: List[str], config: Dict[str, Any], output_dir: Optional[str] = None) -> List[str]:
    from mace.utils import transport_plot as T
    from mace.utils import transport_zt as Z
    out: List[str] = []
    for path, q in _jobs(files, config.get("quantity")):
        try:
            from mace.utils import transport_mesh
            data = transport_mesh.add_note(T.quantity_data(path, q, config.get("kappa_lattice"), config.get("tau_fs")),
                                           path)
            curves = T.traces(data, config.get("component"), config.get("temperature"), config.get("x") or "mu")
        except (T.TransportError, Z.KappaLatticeError) as e:
            print(f"\n  {path.name}: {e}")
            outcome.failed(path.name, str(e))
            continue
        dest = Path(output_dir) if output_dir else path.parent
        dest.mkdir(parents=True, exist_ok=True)
        # ZTₑ (no κ_L) apart from ZT with κ_L
        tag = ("ZT_E" if data["quantity"] == "zt" and data["upper_bound"]
               else T.CLI_OF[data["quantity"]].upper().replace("-", "_")) + _suffix(data, config)
        print(f"\n  {L.plain(data['short'])} ({T.CLI_OF[data['quantity']]}) of {data['stem']}")
        for line in _summary(data):
            print(f"    {line}")
        new = render(data, curves, str(dest / f"{data['stem']}.{tag}"), config)
        for p in new:
            print(f"    Generated: {Path(p).name}")
        out += new
    return out


def _suffix(data: Dict[str, Any], config: Dict[str, Any]) -> str:
    """The figure's options in its name (a second figure of the run does not overwrite the first):
    _300K (one temperature) or _allT, _xx/_all (a component), _carriers (the carrier axis)."""
    out = ""
    if data["blocks"][0]["T"] is not None:
        t = config.get("temperature")
        out += "_allT" if str(t).lower() == "all" else f"_{(float(t) if t not in (None, '') else data['blocks'][0]['T']):g}K"
    if (config.get("component") or "avg") != "avg":
        out += f"_{config['component']}"
    if (config.get("x") or "mu") == "carriers":
        out += "_carriers"
    return out


def _summary(data: Dict[str, Any]) -> List[str]:
    lines = []
    if data.get("tau_fs") is not None:
        lines.append(f"τ = {data['tau_fs']:g} fs ({data['tau_source']})")
    if data["quantity"] == "zt":
        if data["upper_bound"]:
            lines.append("no κ_L: ZTₑ (κ_L = 0), an upper bound (--kappa-lattice for the total)")
        else:
            lines.append(L.plain(data["kappa_l_label"]))
        for key, what in (("zt_e_max", "ZTₑ peak"), ("zt_total_max", "ZT peak")):
            m = data.get(key)
            if m:
                lines.append(f"{what}: {m['value']:.4g} at {m['temperature_k']:g} K, μ {m['mu_ev']:.3f} eV")
    if data.get("refused"):
        lines.append(data["refused"])
    lines += data.get("notes") or []
    return lines


def render(data: Dict[str, Any], curves: List[Dict[str, Any]], out_prefix: str, config: Dict[str, Any]) -> List[str]:
    """Draw ``curves`` (transport_plot.traces) of ``data``; the files written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mace.utils import transport_plot as T
    from .. import presets
    sid = presets.resolve_id(config.get("style") or "default", "band")
    if not curves:
        print("    Nothing to draw: no point of this quantity counts at this temperature.")
        outcome.failed(Path(out_prefix).name, "no points to draw")
        return []
    with matplotlib.rc_context(presets.mpl_rc(sid, "band")):
        w, h = config.get("width") or 7.9, config.get("height") or 4.4
        fig, ax = plt.subplots(figsize=(w, h))
        lw = 1.5 * presets.get(sid, "band")["line"]
        for c in curves:
            ys = [math.nan if v is None else v for v in c["y"]]
            if c["role"] == "reference":
                ax.plot(c["x"], ys, color="0.45", linewidth=1.0, linestyle=presets.mpl_dash("dot"), label=L.mathtext(c["name"]))
                continue
            st = presets.series_style(sid, "band", c["color"] or 0)
            dash = c["dash"] if c["dash"] != "solid" else st["dash"]
            ax.plot(c["x"], ys, color=st["color"], linewidth=lw, linestyle=presets.mpl_dash(dash), label=L.mathtext(c["name"]))
        x = config.get("x") or "mu"
        # the plot markup of the GUI's figure (plot_labels) as mathtext
        ax.set_xlabel(L.mathtext(T.x_title(data, x)))
        ax.set_ylabel(L.mathtext(T.y_title(data)))
        if x == "carriers":
            ax.set_xscale("log")
        yr = T.y_range(data, curves)
        if yr:
            ax.set_ylim(*yr)
        elif data["quantity"] == "zt":
            ax.set_ylim(bottom=0)
        blocks = T.pick_blocks(data, config.get("temperature"))
        if len(blocks) == 1 and blocks[0]["T"] is not None:
            ax.set_title(f"T = {blocks[0]['T']:g} K", fontsize="medium")
        if presets.get(sid, "band")["grid"] != "none":
            ax.grid(True, alpha=0.3)
        if len(curves) > 1:
            from ..legend_layout import legend_outside
            fig.tight_layout()
            legend_outside(fig, ax)
        else:
            fig.tight_layout()
        paths = []
        for fmt in config.get("formats") or ["png", "svg"]:
            p = f"{out_prefix}.{fmt}"
            fig.savefig(p, dpi=config.get("dpi") or 300)
            paths.append(p)
        plt.close(fig)
    return paths


register(PlotterEntry(
    kind=PlotKind.TRANSPORT,
    flag="transport",
    label="transport (BOLTZTRA)",
    handler=plot_transport,
    configure=configure_transport,
    config_from_args=_config_from_args,
    patterns=PATTERNS,
    accepts_energy_range=False,
    progress_tmpl="\n  Plotting {n} transport file(s)...",
    not_found_msg="  No BOLTZTRA transport (.SEEBECK/.SIGMA/.SIGMAS/.KAPPA/.TDF.dat) files found.",
    menu_tmpl="Plot transport (BOLTZTRA) data ({n} files)",
    discovery_label="Transport (BOLTZTRA)",
))
