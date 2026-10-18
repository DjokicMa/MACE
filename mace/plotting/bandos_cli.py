"""mace plotting's band, DOS and band + DOS files drawn with plotly (plotly_bands.py: the GUI's
figure), and the choice between that and the matplotlib plotters (Plotting/ipBANDS_V2.py,
ipDOS_V2.py).

The backend: ``--backend``, else a folder's ``mace-folder.toml [plots] backend``, else a site's
``site.toml [plots] backend``, else ``MACE_PLOT_BACKEND``, else plotly when plotly and kaleido
import and a test export works (an HPCC's plain python3 has no plotly: matplotlib, said once).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import outcome
from . import presets as P

BACKENDS = ("plotly", "matplotlib")
FIGURE_FORMATS = ("png", "svg", "pdf", "html")
DEFAULT_FORMATS = ["png", "svg"]
_NOTED = set()


def _note(text: str) -> None:
    if text not in _NOTED:
        _NOTED.add(text)
        print(f"  Note: {text}")


def site_backend() -> Optional[str]:
    try:
        from mace.submission import site
        tables, _path, _text = site.load_tables()
        v = (tables.get("plots") or {}).get("backend")
        return v if v in BACKENDS else None
    except Exception:
        return None


def choose_backend(requested: Optional[str] = None) -> str:
    """plotly or matplotlib for this run (see the module's docstring); a fallback is said once."""
    want = requested or site_backend() or (os.environ.get("MACE_PLOT_BACKEND") or "").strip().lower() or None
    if want == "matplotlib":
        return "matplotlib"
    from . import plotly_bands
    ok, why = plotly_bands.available()
    if ok:
        return "plotly"
    _note(f"plotly/kaleido cannot export here, so this figure is drawn with matplotlib (--backend matplotlib). {why}")
    return "matplotlib"


def formats_of(config: Dict[str, Any], backend: str) -> List[str]:
    fmts = [f for f in (config.get("formats") or DEFAULT_FORMATS) if f in FIGURE_FORMATS]
    if backend != "plotly" and "html" in fmts:
        fmts = [f for f in fmts if f != "html"] or ["png"]
    return fmts or list(DEFAULT_FORMATS)


def look_of_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """The style (presets.STYLE_OPTIONS names) of a band/DOS config (legacy.py's keys)."""
    look = dict(P.STYLE_DEFAULTS)
    for name, key in (("style", "style"), ("spin_colors", "spin_colors"), ("spin_up_color", "spin_up_color"),
                      ("spin_down_color", "spin_down_color"), ("band_color", "band_color"), ("spin_down_style", "spin_style"),
                      ("dos_colors", "dos_colors"), ("dos_spin", "spin_dos"), ("dos_total", "dos_total"),
                      ("dos_smooth", "dos_smooth")):
        v = config.get(key)
        if v is not None:
            look[name] = v
    return look


def spin_pair(config: Dict[str, Any]) -> Dict[str, str]:
    """The spin pair the config draws (a light figure): the matplotlib plotters take it as colors."""
    look = look_of_config(config)
    return P.spin_colors(look["style"], "band", "light", look["spin_colors"], look["spin_up_color"], look["spin_down_color"])


def _opts(config: Dict[str, Any]) -> Dict[str, Any]:
    from . import plotly_bands as PB
    return PB.opts_for(config.get("e_lower", -6.0), config.get("e_upper", 6.0), config.get("fermi_shift", 0.0),
                       config.get("units") or "eV")


def _out_dir(data_file: Path, output_dir) -> Path:
    if output_dir in (None, ""):
        return data_file.parent
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _size(kind: str, config: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """--width/--height (inches) as the figure's layout px (96 to the inch); None: the GUI pane's size."""
    from . import plotly_bands as PB
    if not (config.get("width") or config.get("height")):
        return None
    w, h = PB.SIZES[kind]
    return (int(round(config["width"] * 96)) if config.get("width") else w,
            int(round(config["height"] * 96)) if config.get("height") else h)


def _write(fig: Dict[str, Any], out_dir: Path, stem: str, fmts: List[str], dpi: Optional[int] = None) -> List[str]:
    from . import plotly_bands as PB
    made = []
    for fmt in fmts:
        target = out_dir / f"{stem}.{fmt}"
        if fmt == "html":
            import plotly.io as pio
            pio.write_html(fig, str(target), include_plotlyjs=True, full_html=True, validate=False,
                           config={"displaylogo": False})
        else:
            PB.write(fig, target, fmt, dpi)
        made.append(str(target))
        print(f"    Generated: {target.name}")
    return made


def _band(path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    from mace.gui import results
    from . import plotly_bands as PB
    # every band in the requested window (the Fermi shift moves it), no cap (the GUI's 400 is a browser cost)
    shift = float(config.get("fermi_shift") or 0.0)
    lo, hi = sorted((float(config.get("e_lower", -6)), float(config.get("e_upper", 6))))
    band = results.band_data(path, config.get("energy_reference") or "auto", window=(lo + shift, hi + shift))
    if config.get("band_path"):
        import sys
        plotting = Path(__file__).resolve().parents[2] / "Plotting"
        if str(plotting) not in sys.path:
            sys.path.insert(0, str(plotting))
        import ipBANDS_V2
        edit = ipBANDS_V2.load_band_path(config["band_path"])
        edited = PB.edited_band(band, edit)
        if edited is band:
            raise ValueError("--band-path does not fit these band data (its .d3 path labels)")
        band = edited
        print("  k-path as edited: " + " ".join(ipBANDS_V2._band_path_module().path_tokens(edit)))
    return band


def _dos(path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    from mace.gui import results
    return results.dos_data(path, config.get("energy_reference") or "auto")


def _say_spin(band=None, dos=None) -> None:
    if band is not None:
        state = "spin-polarized" if band.get("spin_polarized") else \
            "spin-polarized, α = β (non-magnetic): one set drawn" if band.get("spin_identical") else "closed shell"
        print(f"  {len(band['up'])} bands, {state}")
    if dos is not None:
        names = ", ".join(p["label"] for p in dos["projections"])
        state = "spin-polarized" if dos.get("spin_polarized") else \
            "spin-polarized, α = β (non-magnetic): total α + β drawn" if dos.get("spin_summed") else "closed shell"
        print(f"  DOS: {state}; projections {names}")


def band_stem(path: Path) -> str:
    """The name ipBANDS_V2 gives a band figure: fe_band.BAND.DAT -> fe (fe.BANDS.png)."""
    name = re.sub(r"[._]band\.band\.dat$", "", path.name, flags=re.IGNORECASE)
    return name.replace(".BAND.DAT", "").replace("_band.dat", "")


def dos_stem(path: Path) -> str:
    return path.name.replace("_doss.DOSS.DAT", "").replace(".DOSS.DAT", "").replace("_doss.dat", "")


def material_of(band_path: Path) -> str:
    from mace.gui import results
    stem = results._stem(band_path.name, results.BAND_SUFFIXES)
    return results._material(stem, "_band")


def plot(kind: str, files: List[str], config: Dict[str, Any], output_dir=None) -> List[str]:
    """Draw each file with plotly: kind "band" ({stem}.BANDS.fmt), "dos" ({material}.DOSS.fmt) or
    "bandos" (a band file and its DOS, {material}.BAND_DOS.fmt)."""
    from . import plotly_bands as PB
    fmts = formats_of(config, "plotly")
    look = look_of_config(config)
    o = _opts(config)
    if config.get("projection_given"):
        _note("--projection is for --backend matplotlib; the plotly figure draws --dos-colors' projections (as the GUI)")
    made: List[str] = []
    for f in files:
        path = Path(f).resolve()
        print(f"\n  Plotting: {path.name}")
        try:
            if kind == "band":
                band = _band(path, config)
                _say_spin(band=band)
                fig = PB.figure(band, None, look, o, size=_size("band", config))
                made += _write(fig, _out_dir(path, output_dir), f"{band_stem(path)}.BANDS", fmts, config.get("dpi"))
            elif kind == "dos":
                dos = _dos(path, config)
                _say_spin(dos=dos)
                fig = PB.figure(None, dos, look, o, size=_size("dos", config))
                made += _write(fig, _out_dir(path, output_dir), f"{dos_stem(path)}.DOSS", fmts, config.get("dpi"))
            else:
                band_file, dos_file = pair_of(path)
                if band_file is None or dos_file is None:
                    print(f"    No {'DOS' if band_file else 'band structure'} of the same run beside it "
                          f"(a band + DOS figure needs both): skipped")
                    outcome.failed(path.name, "no band or DOS partner of the same run")
                    continue
                print(f"    with {(dos_file if band_file == path else band_file).name}")
                band, dos = _band(band_file, config), _dos(dos_file, config)
                _say_spin(band, dos)
                fig = PB.figure(band, dos, look, o, size=_size("bandos", config))
                made += _write(fig, _out_dir(path, output_dir), f"{material_of(band_file)}.BAND_DOS", fmts, config.get("dpi"))
        except Exception as e:      # one bad file does not stop the others (the run exits non-zero, outcome)
            print(f"    Error: {e}")
            outcome.failed(path.name, str(e))
    return made


def pair_of(path: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """(band file, DOS file) of the run of ``path`` (either of them)."""
    from mace.gui import results
    name = path.name
    if name.endswith(results.BAND_SUFFIXES):
        return path, results.partner(path, "band")
    if name.endswith(results.DOS_SUFFIXES):
        return results.partner(path, "dos"), path
    return None, None


def has_partner(path: str) -> bool:
    try:
        b, d = pair_of(Path(path))
        return b is not None and d is not None
    except Exception:
        return False
