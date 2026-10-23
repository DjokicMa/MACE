"""Transport (BOLTZTRA) plot data: every quantity one run's .dat files give, raw and derived.

One reader for the GUI's transport plot (mace/gui/results.py) and ``mace plotting
--transport`` (mace/plotting/handlers/transport.py), so both draw the same numbers.

The run's files are ``<stem>.SEEBECK.dat``, ``.SIGMA.dat``, ``.SIGMAS.dat``,
``.KAPPA.dat`` and ``.TDF.dat`` (in any case: MACE's own runs write ``.DAT``).
Opening any one of them finds the others beside it by the stem, as the cube
viewer finds a density's POT cube.

Raw quantities (CRYSTAL23 manual, BOLTZTRA, Eqs. 14.1-14.4; the files are SI):

* S (SEEBECK.dat, V/K, drawn in µV/K), σ (SIGMA.dat, S/m), σS (SIGMAS.dat,
  A/(m·K): CRYSTAL writes −σ·S, the sign of Eq. 14.2, as SIGMA.dat × SEEBECK.dat
  shows at every point of every run checked), κ₀ (KAPPA.dat, W/(m·K): the
  manual's κₑ of Eq. 14.3 is the zero-field coefficient) and the TDF against
  energy. An open-shell run's alpha and beta blocks are summed (the extractor's
  ``_parse_transport_dat_tensor``; the TDF the same way here).

Derived, on one run's (T, μ) grid (files on different grids are refused, not
interpolated):

* PF = S²σ                       µW/(cm·K²)
* κₑ = κ₀ − TσS²                 W/(m·K), the electronic thermal conductivity
* ZT = S²σT/(κₑ + κ_L)           with the user's lattice κ_L (W/(m·K), a value or
  a T,kappa_L CSV: mace.utils.transport_zt); without it ZTₑ (κ_L = 0), an
  upper bound
* L = κₑ/(σT)                    10⁻⁸ W·Ω/K², beside the Sommerfeld L₀ = π²k_B²/(3e²)
* σ/τ, PF/τ                      10¹⁸ 1/(Ω·m·s), 10¹¹ W/(m·K²·s), with τ the run's
  RELAXTIM (from its .out, else its .d3; CRYSTAL's default is 10 fs)

As ``mace analyze`` does, a derived point counts where |σ| is at least 10⁻³ of
the component's peak |σ| (BoltzTraP's determinant spikes), and for κₑ, ZTₑ
and L κ₀ is nonzero and κₑ above the files' precision (|κ₀ − TσS²| ≥ 2×10⁻⁶ κ₀:
transport_zt.kappa_e_state; below it κₑ is rounding noise of the 8-digit files,
a gap counted in the notes). The total ZT keeps such a point where κ_L is at
least 100 times that uncertainty. ZT goes through ``transport_zt.zt_points``, so the average's
peak is the database's ``transport_zt_*_max_avg`` to the last digit. σ and κ₀
are at the run's τ; a τ the user gives rescales them by τ/τ_run for PF, κₑ and
the total ZT (S, ZTₑ and L do not depend on τ).

Components: the file's diagonal (xx, yy, zz; a slab's file xx, yy; a polymer's
xx) and "avg", their average (the database's ``*_avg``; a derived average is
computed from the averaged S, σ and κ₀). The carrier axis is N, the carriers per
cell of the run's SIGMA.dat (alpha + beta), positive for holes and negative for
electrons, over the cell volume (cm⁻³) for a 3D run: the .out's ``CELL VOLUME (A.U.)``, else the
block header's V(cm^3), which is that volume in bohr³ × 10⁻²⁴ (so converted from bohr³).
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from mace.utils import plot_labels as L
from mace.utils import transport_zt as Z

FILE_KINDS = ("SEEBECK", "SIGMA", "SIGMAS", "KAPPA", "TDF")
NAME_RE = re.compile(r"^(?P<stem>.+)\.(?P<q>SEEBECK|SIGMAS|SIGMA|KAPPA|TDF)\.dat$", re.I)

# Sommerfeld's Lorenz number pi^2 k_B^2 / (3 e^2), W Ohm / K^2 (CODATA 2018 k_B, e)
L0 = math.pi ** 2 / 3.0 * (1.380649e-23 / 1.602176634e-19) ** 2
# a bohr in cm (CODATA 2018): CRYSTAL's "V(cm^3)" block headers hold the cell volume in bohr^3 times 1e-24
# (diamond: CELL VOLUME (A.U.) 75.147743 and V(cm^3) = 7.515E-23, while the cell is 11.136 A^3)
from mace.constants import BOHR_TO_ANGSTROM  # noqa: E402
BOHR_CM = BOHR_TO_ANGSTROM * 1e-8
_CELL_VOLUME = re.compile(r"CELL VOLUME \(A\.U\.\)\s+([0-9.]+(?:[EeDd][-+]?\d+)?)")
TAU_DEFAULT_FS = 10.0          # CRYSTAL23 manual, BOLTZTRA RELAXTIM: "default 10. fs"
SIGMA_FLOOR = 1e-3             # the extractor's rule (zt_points, _pf_peak)
_SEEBECK_SIGMA_KAPPA = ("SEEBECK", "SIGMA", "KAPPA")

# key: label, short (the Quantity menu), unit (unit_name: its plot_labels.UNITS name, the CSV's spelling), factor
# from SI, the run's files it needs, whether it needs the run's relaxation time, the CLI's --quantity name, and
# the x axis. The labels are plot markup (mace.utils.plot_labels; plain() for the terminal)
_K0, _KE, _KL, _S2 = L.SYMBOLS["kappa_0"], L.SYMBOLS["kappa_e"], L.SYMBOLS["kappa_L"], L.SYMBOLS["S2"]
_M = L.MINUS
QUANTITIES: Dict[str, Dict[str, Any]] = {
    "SEEBECK": {"label": "Seebeck coefficient S", "short": "S", "unit_name": "uV/K", "factor": 1e6, "needs": ("SEEBECK",)},
    "SIGMA": {"label": "Electrical conductivity σ", "short": "σ", "unit_name": "S/m", "factor": 1.0, "needs": ("SIGMA",)},
    "SIGMAS": {"label": f"σS (SIGMAS.dat: {_M}σ·S, CRYSTAL's sign)", "short": "σS", "unit_name": "A/(m K)", "factor": 1.0,
               "needs": ("SIGMAS",)},
    "KAPPA": {"label": f"Zero-field thermal coefficient {_K0} (KAPPA.dat, the manual's {_KE}, Eq. 14.3)", "short": _K0,
              "unit_name": "W/(m K)", "factor": 1.0, "needs": ("KAPPA",)},
    "TDF": {"label": "Transport distribution function", "short": "TDF", "unit_name": "1/hbar2 eV fs/A", "factor": 1.0,
            "needs": ("TDF",), "x": "energy"},
    "pf": {"label": f"Power factor {_S2}σ", "short": "PF", "unit_name": "uW/(cm K2)", "factor": 1e4,
           "needs": ("SEEBECK", "SIGMA")},
    "kappa_e": {"label": f"Electronic thermal conductivity {_KE} = {_K0} {_M} Tσ{_S2}", "short": _KE, "unit_name": "W/(m K)",
                "factor": 1.0, "needs": _SEEBECK_SIGMA_KAPPA},
    "zt": {"label": f"Figure of merit ZT = {_S2}σT/({_KE} + {_KL})", "short": "ZT", "unit_name": "", "factor": 1.0,
           "needs": _SEEBECK_SIGMA_KAPPA},
    "lorenz": {"label": f"Lorenz number L = {_KE}/(σT)", "short": "L", "unit_name": "1e-8 W Ohm/K2", "factor": 1e8,
               "needs": _SEEBECK_SIGMA_KAPPA},
    "sigma_tau": {"label": "σ/τ", "short": "σ/τ", "unit_name": "1e18 1/(Ohm m s)", "factor": 1e-18, "needs": ("SIGMA",),
                  "tau": True},
    "pf_tau": {"label": f"PF/τ = {_S2}σ/τ", "short": "PF/τ", "unit_name": "1e11 W/(m K2 s)", "factor": 1e-11,
               "needs": ("SEEBECK", "SIGMA"), "tau": True},
}
for _q in QUANTITIES.values():
    _q["unit"] = L.unit(_q["unit_name"])
RAW = FILE_KINDS
DERIVED = ("pf", "kappa_e", "zt", "lorenz", "sigma_tau", "pf_tau")
# mace plotting --quantity NAME
CLI_NAMES = {"seebeck": "SEEBECK", "sigma": "SIGMA", "sigmas": "SIGMAS", "kappa": "KAPPA", "tdf": "TDF",
             "pf": "pf", "kappa-e": "kappa_e", "zt": "zt", "lorenz": "lorenz", "sigma-tau": "sigma_tau",
             "pf-tau": "pf_tau"}
CLI_OF = {v: k for k, v in CLI_NAMES.items()}
COMPONENTS = ("avg", "xx", "yy", "zz", "all")
X_AXES = ("mu", "carriers")


class TransportError(ValueError):
    """A quantity MACE cannot draw from these files, with the reason in the message."""


def quantity_key(name: str) -> str:
    """A quantity's key from a CLI name (zt, kappa-e, seebeck), a key (kappa_e) or a file token (SEEBECK)."""
    n = str(name or "").strip()
    if n in QUANTITIES:
        return n
    k = CLI_NAMES.get(n.lower()) or CLI_NAMES.get(n.lower().replace("_", "-"))
    if k is None:
        raise TransportError(f"unknown transport quantity {name!r}: one of {', '.join(CLI_NAMES)}")
    return k


def split_name(path: Path) -> Tuple[str, str]:
    """(stem, file kind) of a transport .dat file's name; raises for another file."""
    m = NAME_RE.match(Path(path).name)
    if not m:
        raise TransportError(f"{Path(path).name} is not a BOLTZTRA .SEEBECK/.SIGMA/.SIGMAS/.KAPPA/.TDF.dat file")
    return m.group("stem"), m.group("q").upper()


def run_files(path: Path, inside: Optional[Callable[[Path], bool]] = None) -> Dict[str, Path]:
    """The run's transport files beside ``path`` (it included), by kind: the same stem, any case.
    ``inside(path) -> bool``: the files it may read (the GUI's served folder)."""
    path = Path(path)
    stem, kind = split_name(path)
    ok = inside or (lambda f: True)
    found: Dict[str, Path] = {kind: path}
    try:
        names = sorted(p.name for p in path.parent.iterdir())
    except OSError:
        names = []
    for name in names:
        m = NAME_RE.match(name)
        if not m or m.group("stem") != stem:
            continue
        q = m.group("q").upper()
        cand = path.parent / name
        if q not in found and cand.is_file() and ok(cand):
            found[q] = cand
    return {q: found[q] for q in FILE_KINDS if q in found}


def beside(path: Path, stem: str, suffix: str, ok) -> Optional[Path]:
    """``<stem><suffix>`` beside ``path``, in any case of the suffix."""
    exact = path.with_name(stem + suffix)
    if exact.is_file():
        return exact if ok(exact) else None
    try:
        for c in sorted(path.parent.iterdir()):
            if c.name.lower() == (stem + suffix).lower() and c.is_file() and ok(c):
                return c
    except OSError:
        pass
    return None


def find_out(path: Path, stem: str, inside=None) -> Optional[Path]:
    """The BOLTZTRA .out of a run's .dat files: ``<stem>.out``, else the one transport .out beside them."""
    from mace.utils.calc_detection import is_transport_output
    ok = inside or (lambda f: True)
    exact = path.with_name(f"{stem}.out")
    if exact.is_file():
        return exact if ok(exact) else None
    cands = []
    for c in sorted(path.parent.glob("*.out")):
        if not ok(c):
            continue
        try:
            with c.open(errors="ignore") as fh:
                if is_transport_output(fh.read(2_000_000)):
                    cands.append(c)
        except OSError:
            continue
    return cands[0] if len(cands) == 1 else None


def d3_relaxation_time(text: str) -> Optional[float]:
    """τ (fs) a .d3's BOLTZTRA block sets: its RELAXTIM value, CRYSTAL's 10 fs without one, None without BOLTZTRA."""
    lines = [ln.strip() for ln in (text or "").splitlines()]
    try:
        start = next(i for i, ln in enumerate(lines) if ln.upper().startswith("BOLTZTRA"))
    except StopIteration:
        return None
    for i in range(start + 1, len(lines)):
        word = lines[i].upper()
        if word.startswith("END"):
            break
        if word.startswith("RELAXTIM"):
            for nxt in lines[i + 1:]:
                if nxt:
                    try:
                        t = float(nxt.split()[0].replace("D", "E").replace("d", "e"))
                    except ValueError:
                        return None
                    return t if math.isfinite(t) and t > 0 else None
            return None
    return TAU_DEFAULT_FS


def relaxation_time(path: Path, inside=None) -> Dict[str, Any]:
    """The run's relaxation time: {"tau_fs", "source", "out"} from its .out's ``RELAXATION TIME (fs)``
    line, else its .d3's RELAXTIM (CRYSTAL's 10 fs when the BOLTZTRA block has none); tau_fs None
    with the reason in "source" when neither is beside the data."""
    path = Path(path)
    stem, _ = split_name(path)
    ok = inside or (lambda f: True)
    out = find_out(path, stem, inside)
    if out is not None:
        tau = Z.read_relaxation_time_fs(out.read_text(errors="ignore"))
        if tau is not None:
            return {"tau_fs": tau, "source": f"RELAXTIM, {out.name}", "out": out}
    d3 = beside(path, stem, ".d3", ok)
    if d3 is not None:
        text = d3.read_text(errors="ignore")
        tau = d3_relaxation_time(text)
        if tau is not None:
            has = re.search(r"^\s*RELAXTIM", text, re.M | re.I)
            return {"tau_fs": tau, "out": out,
                    "source": f"RELAXTIM, {d3.name}" if has else f"CRYSTAL's default (no RELAXTIM in {d3.name})"}
    return {"tau_fs": None, "out": out,
            "source": f"no {stem}.out or {stem}.d3 beside the data says which relaxation time (RELAXTIM) the run used"}


# ---------------------------------------------------------------- reading
_CACHE: Dict[Tuple[str, int, int], Any] = {}
_CACHE_MAX = 24
_HEAD_V = re.compile(r"T\(K\)\s*=\s*([-+0-9.Ee]+).*V\(cm\^3\)\s*=\s*([-+0-9.Ee]+)")


def _cached(path: Path, what: str, fn):
    try:
        st = path.stat()
        key = (f"{path.resolve()}|{what}", st.st_mtime_ns, st.st_size)
    except OSError:
        return fn()
    if key not in _CACHE:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = fn()
    return _CACHE[key]


def _volumes(path: Path) -> Dict[float, float]:
    """{T: V (cm^3)} of the block headers ``# T(K) = 300.000 Npoints = 711 V(cm^3) = 1.123E-21``: the first
    header of each temperature (an open-shell file's beta headers print V 10^-24 times too small, the
    conversion to cm^3 applied twice, in every open-shell run checked)."""
    vols = {}
    with path.open(errors="ignore") as fh:
        for line in fh:
            if line.startswith("#") and "V(cm^3)" in line:
                m = _HEAD_V.search(line)
                if m:
                    try:
                        vols.setdefault(round(float(m.group(1)), 3), float(m.group(2)))
                    except ValueError:
                        pass
    return vols


def read_tensor(path: Path) -> Dict[str, Any]:
    """A (T, μ) file as {"axes", "unit", "blocks": [{T, mu, n, diag: {axis: [...]}}], "volumes"}:
    the extractor's rows (an open-shell file's alpha and beta blocks summed), grouped by
    temperature in the file's order. SI values."""
    path = Path(path)

    def read():
        from mace.utils.property_extractor import CrystalPropertyExtractor
        unit, rows, axes = CrystalPropertyExtractor._parse_transport_dat_tensor(path, with_axes=True)
        if not rows:
            raise TransportError(f"{path.name}: no data rows")
        blocks: Dict[float, Dict[str, Any]] = {}
        for T, mu, n, diag in rows:
            b = blocks.setdefault(T, {"T": T, "mu": [], "n": [], "diag": {a: [] for a in axes}})
            b["mu"].append(mu)
            b["n"].append(n)
            for a, v in zip(axes, diag):
                b["diag"][a].append(v)
        return {"axes": tuple(axes), "unit": unit, "blocks": [blocks[T] for T in sorted(blocks)],
                "volumes": _volumes(path)}
    return _cached(path, "tensor", read)


def read_tdf(path: Path) -> Dict[str, Any]:
    """TDF.dat as {"axes", "energy", "diag": {axis: [...]}}: the diagonal by the header's column names
    (TDF_xx ... TDF_zz; a slab's xx, yy; a polymer's xx); an open-shell file's alpha and beta blocks
    summed per energy, as σ (Eq. 14.4 is a sum over bands). Raises with the reason for a file with
    no rows (CRYSTAL can write the header alone)."""
    path = Path(path)

    def read():
        from mace.utils.property_extractor import CrystalPropertyExtractor
        columns = None
        idx = axes = None
        blocks: List[Dict[float, List[float]]] = []
        for line in path.read_text(errors="ignore").splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                if "energy(ev)" in s.lower():
                    columns = s.lstrip("#").split()[1:]
                elif re.match(r"^#\s*[01]\s*\[", s):
                    blocks.append({})
                continue
            try:
                v = [float(x) for x in s.split()]
            except ValueError:
                continue
            if len(v) < 2 or not all(math.isfinite(x) for x in v):
                continue
            if idx is None:
                idx, axes = CrystalPropertyExtractor._diagonal_layout(columns, len(v) - 1)
            if not blocks:
                blocks.append({})
            blocks[-1][round(v[0], 6)] = [v[1 + i] for i in idx]
        blocks = [b for b in blocks if b]
        if not blocks:
            raise TransportError(f"{path.name} has a header and no data rows: CRYSTAL wrote no transport "
                                 f"distribution function to it")
        energies = list(blocks[0])
        total = {E: list(vals) for E, vals in blocks[0].items()}
        for b in blocks[1:]:          # beta (open shell): summed point by point
            for E, vals in b.items():
                if E in total:
                    total[E] = [x + y for x, y in zip(total[E], vals)]
                else:
                    total[E] = list(vals)
                    energies.append(E)
        energies.sort()
        return {"axes": tuple(axes), "energy": energies, "spin_blocks": len(blocks),
                "diag": {a: [total[E][i] for E in energies] for i, a in enumerate(axes)}}
    return _cached(path, "tdf", read)


def tdf_has_rows(path: Path) -> bool:
    with Path(path).open(errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if s and not s.startswith("#"):
                return True
    return False


# ---------------------------------------------------------------- availability
def availability(files: Dict[str, Path], tau: Dict[str, Any], opened: Path) -> Dict[str, Optional[str]]:
    """{quantity: None (it can be drawn) or why not} for every quantity, from the files beside
    ``opened`` (no grids compared here: a mismatch is refused when the quantity is drawn)."""
    out: Dict[str, Optional[str]] = {}
    for q, spec in QUANTITIES.items():
        missing = [k for k in spec["needs"] if k not in files]
        if missing:
            out[q] = (f"needs the run's {' and '.join(k + '.dat' for k in missing)}, which is not beside "
                      f"{Path(opened).name}")
        elif q == "TDF" and not tdf_has_rows(files["TDF"]):
            out[q] = f"{files['TDF'].name} has a header and no data rows (CRYSTAL wrote none)"
        elif spec.get("tau") and tau.get("tau_fs") is None:
            out[q] = f"needs the run's relaxation time: {tau['source']}"
        else:
            out[q] = None
    return out


# ---------------------------------------------------------------- grids
def _grid(t: Dict[str, Any]) -> List[Tuple[float, Tuple[float, ...]]]:
    return [(round(b["T"], 3), tuple(round(m, 4) for m in b["mu"])) for b in t["blocks"]]


def _grid_text(g) -> str:
    Ts = ", ".join(f"{T:g}" for T, _ in g)
    mus = [m for _, ms in g for m in ms]
    n = len(g[0][1]) if g else 0
    return f"T = {Ts} K, μ from {min(mus):g} to {max(mus):g} eV ({n} points per temperature)" if mus else "no points"


def check_grids(tensors: Dict[str, Dict[str, Any]], files: Dict[str, Path]) -> None:
    """Refuses files of one quantity that are not on the same (T, μ) grid (MACE does not interpolate)."""
    kinds = list(tensors)
    ref = _grid(tensors[kinds[0]])
    for k in kinds[1:]:
        g = _grid(tensors[k])
        if g != ref:
            raise TransportError(
                f"{files[kinds[0]].name} and {files[k].name} are not on the same chemical potential and "
                f"temperature grid ({_grid_text(ref)}; {_grid_text(g)}): they are not from one BOLTZTRA run, "
                f"and MACE does not interpolate between them")


# ---------------------------------------------------------------- the data of one quantity
def _avg_axes(axes: Sequence[str], periodic: Optional[Sequence[str]]) -> List[str]:
    sel = [a for a in axes if periodic is None or a in periodic]
    return sel or list(axes)


def avg_label(axes: Sequence[str]) -> str:
    return axes[0] if len(axes) == 1 else f"({' + '.join(axes)}) / {len(axes)}"


def _rows(t: Dict[str, Any], comp: str, sel: Sequence[str]) -> List[Tuple[float, float, float, float]]:
    """Rows (T, μ, N, value) of one component ("avg": the average over ``sel``), as zt_points takes them."""
    rows = []
    for b in t["blocks"]:
        for i, (mu, n) in enumerate(zip(b["mu"], b["n"])):
            v = (sum(b["diag"][a][i] for a in sel) / len(sel)) if comp == "avg" else b["diag"][comp][i]
            rows.append((b["T"], mu, n, v))
    return rows


def _by_t(rows) -> Dict[float, List[float]]:
    out: Dict[float, List[float]] = {}
    for T, _mu, _n, v in rows:
        out.setdefault(T, []).append(v)
    return out


def _floor(sig_rows) -> float:
    vals = [abs(v) for *_, v in sig_rows if v != 0.0]
    return (max(vals) if vals else 0.0) * SIGMA_FLOOR


def kappa_e_counts(s_rows, g_rows, k_rows) -> Tuple[int, int, Optional[float]]:
    """(points whose κₑ is below the files' precision, points where κₑ is computed: |σ| above the floor
    and κ₀ nonzero, the median Lorenz number L/L₀ of the points whose κₑ counts), by
    transport_zt.kappa_e_state."""
    floor = _floor(g_rows)
    S, G, K = _by_t(s_rows), _by_t(g_rows), _by_t(k_rows)
    masked = counted = 0
    ratios = []
    for T in S:
        for s, g, k in zip(S[T], G[T], K[T]):
            if floor <= 0.0 or abs(g) < floor or not k:
                continue
            counted += 1
            ke = abs(k) - T * s * s * abs(g)
            state = Z.kappa_e_state(k, ke)
            masked += state == "masked"
            if state == "ok" and T > 0:
                ratios.append(ke / (abs(g) * T) / L0)
    ratios.sort()
    return masked, counted, (ratios[len(ratios) // 2] if ratios else None)


def _derived_series(q: str, s_rows, g_rows, k_rows, kl, scale: float, tau_run_fs: Optional[float]):
    """{T: values} of a derived quantity in SI (None where a point does not count), and for zt with a
    κ_L the ZTₑ bound too: (values, bound or None)."""
    if q == "zt":
        r = Z.zt_points(s_rows, g_rows, k_rows, kl, scale)
        vals = {T: (r["blocks"][T]["zt_total"] if kl is not None else r["blocks"][T]["zt_e"]) for T in r["temperatures"]}
        bound = {T: r["blocks"][T]["zt_e"] for T in r["temperatures"]} if kl is not None else None
        refused = {T: r["blocks"][T]["refused"] for T in r["temperatures"] if r["blocks"][T]["refused"]}
        return vals, bound, r, refused
    if q == "sigma_tau":
        tau_s = tau_run_fs * 1e-15
        return {T: [v / tau_s for v in vs] for T, vs in _by_t(g_rows).items()}, None, None, {}
    floor = _floor(g_rows)
    S = _by_t(s_rows)
    G = _by_t(g_rows)
    K = _by_t(k_rows) if k_rows is not None else None
    out: Dict[float, List[Optional[float]]] = {}
    for T in S:
        vals = []
        for i, s in enumerate(S[T]):
            g = abs(G[T][i])
            if floor <= 0.0 or g < floor:
                vals.append(None)
                continue
            pf = s * s * g
            if q == "pf":
                vals.append(scale * pf)
            elif q == "pf_tau":
                vals.append(pf / (tau_run_fs * 1e-15))
            else:
                k = abs(K[T][i])
                ke = k - T * pf
                # κₑ below the files' precision (rounding noise) or negative: a gap (transport_zt.kappa_e_state)
                if not k or Z.kappa_e_state(k, ke) != "ok":
                    vals.append(None)
                elif q == "kappa_e":
                    vals.append(scale * ke)
                else:   # lorenz: τ cancels
                    vals.append(ke / (g * T) if T > 0 else None)
        out[T] = vals
    return out, None, None, {}


def quantity_data(path: Path, quantity: Optional[str] = None, kappa_l=None, tau_fs=None,
                  inside: Optional[Callable[[Path], bool]] = None) -> Dict[str, Any]:
    """Everything a transport plot draws for ``quantity`` (default: the opened file's own) of the run
    ``path`` belongs to, in the units of QUANTITIES:

    {"name", "stem", "quantity", "file_quantity", "label", "short", "unit", "x" ("mu" or "energy"),
     "axes", "components" (the file's diagonal), "avg_label", "dim" (3, 2 a slab, 1 a polymer),
     "blocks": [{"T", "mu", "carriers" (N per cell), "volume_cm3", "avg", "xx", ... and for ZT with
     κ_L "bound": {component: ZTₑ}}], "siblings": {kind: file name}, "available": {quantity: None or
     why not}, "tau_fs", "tau_run_fs", "tau_source", "kappa_l_label", "kappa_l_source", "upper_bound",
     "reference" (L₀ for L), "refused", "notes", "zt_e_max", "zt_total_max", "header"}.

    Raises TransportError (a missing file, files on different grids, no relaxation time for σ/τ)
    and transport_zt.KappaLatticeError (a κ_L or τ MACE refuses)."""
    path = Path(path)
    stem, own = split_name(path)
    q = quantity_key(quantity) if quantity else own
    spec = QUANTITIES[q]
    files = run_files(path, inside)
    tau = relaxation_time(path, inside)
    avail = availability(files, tau, path)
    siblings = {k: p.name for k, p in files.items()}
    base = {"name": path.name, "stem": stem, "quantity": q, "file_quantity": own, "label": spec["label"],
            "short": spec["short"], "unit": spec["unit"], "unit_name": spec["unit_name"], "x": spec.get("x", "mu"), "siblings": siblings,
            "available": avail, "tau_run_fs": tau["tau_fs"], "tau_source": tau["source"], "tau_fs": None,
            "kappa_l_label": None, "kappa_l_source": None, "upper_bound": False, "reference": None,
            "refused": None, "notes": [], "zt_e_max": None, "zt_total_max": None, "header": _header(path)}
    if avail[q] is not None and not (q == "TDF" and "TDF" in files):
        raise TransportError(f"{spec['short']}: {avail[q]}")
    if q == "TDF":
        t = read_tdf(files["TDF"])
        axes = list(t["axes"])
        avg = [sum(t["diag"][a][i] for a in axes) / len(axes) for i in range(len(t["energy"]))]
        block = {"T": None, "mu": t["energy"], "carriers": None, "volume_cm3": None, "avg": avg,
                 **{a: t["diag"][a] for a in axes}}
        notes = ["the alpha and beta blocks summed"] if t["spin_blocks"] > 1 else []
        return {**base, "axes": axes, "components": axes, "avg_label": avg_label(axes), "dim": len(axes),
                "blocks": [block], "notes": notes}

    out = tau["out"]
    from mace.utils.property_extractor import CrystalPropertyExtractor
    periodic = CrystalPropertyExtractor.periodic_axes(out.read_text(errors="ignore")) if out is not None else None
    need = spec["needs"]
    tensors = {k: read_tensor(files[k]) for k in need}
    if len(tensors) > 1:
        check_grids(tensors, files)
    first = tensors[need[0]]
    axes = list(first["axes"])
    for k in need[1:]:
        axes = [a for a in axes if a in tensors[k]["axes"]]
    sel = _avg_axes(axes, periodic)
    # the carrier axis: the run's SIGMA.dat (alpha + beta) when it is on this grid, else this file's own
    carriers_src = first
    if "SIGMA" in files and need[0] != "SIGMA":
        try:
            sig = read_tensor(files["SIGMA"])
            if _grid(sig) == _grid(first):
                carriers_src = sig
        except TransportError:
            pass
    vols = carriers_src["volumes"] or first["volumes"]
    cell = _CELL_VOLUME.search(out.read_text(errors="ignore")) if out is not None else None
    cell_cm3 = float(cell.group(1).replace("D", "E")) * BOHR_CM ** 3 if cell else None
    blocks = []
    for b, cb in zip(first["blocks"], carriers_src["blocks"]):
        head = vols.get(round(b["T"], 3))
        blocks.append({"T": b["T"], "mu": list(b["mu"]), "carriers": list(cb["n"]),
                       "volume_cm3": cell_cm3 or (head * (BOHR_CM * 1e8) ** 3 if head else None)})
    factor = spec["factor"]
    comps = ["avg"] + axes

    if q in RAW:
        for blk, b in zip(blocks, first["blocks"]):
            for c in comps:
                vals = ([sum(b["diag"][a][i] for a in sel) / len(sel) for i in range(len(b["mu"]))] if c == "avg"
                        else b["diag"][c])
                blk[c] = [v * factor for v in vals]
        return {**base, "axes": axes, "components": axes, "avg_label": avg_label(sel), "dim": len(axes),
                "blocks": blocks}

    # derived: σ and κ₀ at the run's τ, or rescaled to a τ given (PF, κₑ, total ZT)
    if q != "zt" or kappa_l is None or (isinstance(kappa_l, str) and not kappa_l.strip()):
        kl = None
    else:
        kl = kappa_l if isinstance(kappa_l, Z.KappaLattice) else Z.parse_kappa_lattice(kappa_l)
    tau_given = Z.check_tau_fs(tau_fs)
    scale = 1.0
    tau_used = tau["tau_fs"]
    refused = None
    if tau_given is not None and q in ("pf", "kappa_e", "zt"):
        if tau["tau_fs"] is None:
            raise TransportError(f"a relaxation time rescales σ and κ₀ by τ/τ_run, and {tau['source']}")
        tau_used, scale = tau_given, tau_given / tau["tau_fs"]
    if q == "zt" and kl is not None and tau["tau_fs"] is None:
        refused = (f"no total ZT: {tau['source']}, and SIGMA.dat and KAPPA.dat are at that time (κ_L carries "
                   f"no τ); ZTₑ (κ_L = 0) is drawn")
        kl_used = None
    else:
        kl_used = kl if q == "zt" else None
    zt_r = None
    refused_t: Dict[float, str] = {}
    masked = counted = 0
    l_ratio = None
    for c in comps:
        s_rows = _rows(tensors["SEEBECK"], c, sel) if "SEEBECK" in tensors else None
        g_rows = _rows(tensors["SIGMA"], c, sel)
        k_rows = _rows(tensors["KAPPA"], c, sel) if "KAPPA" in tensors else None
        vals, bound, r, ref_t = _derived_series(q, s_rows, g_rows, k_rows, kl_used, scale, tau["tau_fs"])
        if c == "avg":
            zt_r, refused_t = r, ref_t
            if q in ("kappa_e", "zt", "lorenz"):
                masked, counted, l_ratio = kappa_e_counts(s_rows, g_rows, k_rows)
        for blk in blocks:
            v = vals.get(blk["T"]) or [None] * len(blk["mu"])
            blk[c] = [None if x is None else x * factor for x in v]
            if bound is not None:
                blk.setdefault("bound", {})[c] = bound.get(blk["T"])
    res = {**base, "axes": axes, "components": axes, "avg_label": avg_label(sel), "dim": len(axes),
           "blocks": blocks}
    notes = [f"points with |σ| below {SIGMA_FLOOR:g} of its peak are left out, as mace analyze"] if q != "sigma_tau" else []
    if q in ("kappa_e", "zt", "lorenz") and masked:
        # κₑ = κ₀ − TσS² within the 8-digit files' rounding of 0 (transport_zt.KAPPA_E_EPS)
        what = ("ZTₑ left out; the total kept where κ_L is at least 100 times that uncertainty"
                if q == "zt" and kl_used is not None else "left out")
        notes.append(f"κₑ below the files' precision at {masked} points ({what}): there κ₀ and TσS² agree "
                     f"to within {Z.KAPPA_E_EPS:g} of κ₀")
        res["n_masked"] = masked
    if q in ("kappa_e", "zt", "lorenz") and len(axes) < 3 and l_ratio is not None and l_ratio < 0.1:
        # a slab's or polymer's KAPPA.dat (the example suite's graphene and polyacetylene): κ₀ is nearly TσS²
        # everywhere, so κₑ is a small difference even where it is above the files' precision
        notes.append(f"in this {'slab' if len(axes) == 2 else 'polymer'} run κ₀ nearly equals TσS² at every point: "
                     f"κₑ is below the files' precision at {masked} of {counted} points, and elsewhere L is "
                     f"{l_ratio:.2g} L₀ (median), far below Sommerfeld's L₀ for electrons, so κₑ, L and ZTₑ "
                     f"of this run are not reliable")
    if q in ("sigma_tau", "pf_tau"):
        notes.append(f"τ = {tau['tau_fs']:g} fs ({tau['source']})")
        res["tau_fs"] = tau["tau_fs"]
    elif q in ("pf", "kappa_e", "zt"):
        res["tau_fs"] = tau_used
        if tau_given is not None:
            notes.append(f"σ and κ₀ rescaled from the run's {tau['tau_fs']:g} fs to τ = {tau_given:g} fs")
    if q == "lorenz":
        res["reference"] = {"value": L0 * factor, "label": f"Sommerfeld {L.SYMBOLS['L_0']} = {L0 * 1e8:.3g} × {L.unit('1e-8 W Ohm/K2')}"}
    if q == "zt":
        res["upper_bound"] = kl_used is None
        if kl is not None:
            res["kappa_l_label"], res["kappa_l_source"] = kl.label(), kl.source()
        if zt_r is not None:
            res["zt_e_max"] = zt_r["zt_e_max"]
            res["zt_total_max"] = zt_r["zt_total_max"] if kl_used is not None else None
        if refused is None and kl_used is not None:
            missing = Z.uncovered(kl_used, [b["T"] for b in blocks])
            if missing:
                refused = (f"the κ_L table {Path(kl_used.path).name} covers {kl_used.range_text()}, and the run "
                           f"has {', '.join(f'{t:g} K' for t in missing)}: no total ZT there (MACE does not "
                           f"extrapolate)")
                res["zt_total_max"] = None
        for blk in blocks:
            blk["refused"] = refused_t.get(blk["T"])
    res["refused"] = refused
    res["notes"] = notes
    return res


def _header(path: Path) -> Optional[List[str]]:
    with Path(path).open(errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("#") and ("Mu(eV)" in s or "Energy(eV)" in s):
                return s.lstrip("#").split()
            if s and not s.startswith("#"):
                break
    return None


# ---------------------------------------------------------------- what a figure draws (GUI and CLI)
def comp_label(data: Dict[str, Any], c: str) -> str:
    return data["avg_label"] if c == "avg" else c


def pick_blocks(data: Dict[str, Any], temperature=None) -> List[Dict[str, Any]]:
    """The blocks drawn: one temperature (None: the first, as the GUI opens), or "all"."""
    blocks = data["blocks"]
    if temperature in (None, "") or blocks[0]["T"] is None:
        return blocks[:1]
    if str(temperature).lower() == "all":
        return blocks
    try:
        T = float(temperature)
    except (TypeError, ValueError):
        raise TransportError(f"the temperature is a number of K or 'all'; got {temperature!r}")
    for b in blocks:
        if abs(b["T"] - T) < 1e-3:
            return [b]
    have = ", ".join(f"{b['T']:g}" for b in blocks)
    raise TransportError(f"no {T:g} K block: the run has {have} K")


def pick_components(data: Dict[str, Any], component: Optional[str], temperature=None) -> List[str]:
    c = component or "avg"
    if c not in COMPONENTS:
        raise TransportError(f"the component is one of {', '.join(COMPONENTS)}")
    if c == "all":
        if str(temperature).lower() == "all":
            raise TransportError("all temperatures are drawn for one component: pick avg, xx, yy or zz")
        return list(data["components"])
    if c != "avg" and c not in data["components"]:
        raise TransportError(f"this run's files have {', '.join(data['components'])} only, no {c}")
    return [c]


# the axis titles and the ZT legend entries, in plot markup (plot_labels): js/docs/plot.js trXTitle, trYTitle
ZT_E_KEY = f"{L.SYMBOLS['ZT_e']} ({L.SYMBOLS['kappa_L']} = 0)"


def x_title(data: Dict[str, Any], x: str = "mu") -> str:
    if data["x"] == "energy":
        return L.with_unit("Energy", "eV")
    if x == "carriers":
        return L.with_unit("Carrier concentration |n|", "cm-3") if data.get("dim") == 3 else "Carriers per cell |N|"
    return L.with_unit("Chemical potential μ", "eV")


def y_title(data: Dict[str, Any]) -> str:
    if data["quantity"] == "zt":
        return "Figure of merit ZT" if not data["upper_bound"] else f"{L.SYMBOLS['ZT_e']} ({L.SYMBOLS['kappa_L']} = 0, an upper bound)"
    return L.with_unit(data["label"], data["unit"])


def y_range(data: Dict[str, Any], curves: List[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    """ZT with κ_L: the y axis spans the total ZT, so the dashed ZTₑ bound (no lattice term: it runs up where κₑ
    is tiny, in the gap) does not flatten it; None otherwise (the plot's own range). js/docs/plot.js trYRange."""
    if data["quantity"] != "zt" or data["upper_bound"]:
        return None
    ys = [v for c in curves if c["role"] == "data" for v in c["y"] if v is not None and math.isfinite(v)]
    return (0.0, 1.08 * max(ys)) if ys and max(ys) > 0 else None


def traces(data: Dict[str, Any], component: Optional[str] = None, temperature=None, x: str = "mu") -> List[Dict[str, Any]]:
    """The curves a transport figure draws, in order: [{"name", "x", "y", "color" (series index),
    "dash" (solid, dash, dot), "role" (data, bound, reference)}]; None in y is a gap. js/docs/plot.js
    transportTraces draws the same curves (tests/gui/test_transport_quantities_e2e.py)."""
    if x not in X_AXES:
        raise TransportError(f"the x axis is one of {', '.join(X_AXES)}")
    if x == "carriers" and data["x"] == "energy":
        raise TransportError("the TDF is against energy: it has no carrier axis")
    blocks = pick_blocks(data, temperature)
    comps = pick_components(data, component, temperature)
    many_t = len(blocks) > 1
    zt_total = data["quantity"] == "zt" and not data["upper_bound"]
    out: List[Dict[str, Any]] = []
    k = 0
    for b in blocks:
        for c in comps:
            # one curve (a component at a temperature): the legend names what it is, not the component
            base = f"{b['T']:g} K" if many_t else comp_label(data, c) if len(comps) > 1 else None
            series = [(b[c], "data", (f"ZT ({data['kappa_l_label']}, τ = {data['tau_fs']:g} fs)" if zt_total else None))]
            if zt_total and b.get("bound") and b["bound"].get(c) is not None:
                series.append((b["bound"][c], "bound", ZT_E_KEY))
            for y, role, what in series:
                if zt_total and role == "data" and b.get("refused"):
                    continue
                name = (f"{base} · {what}" if base else what) if what else (base or comp_label(data, c))
                dash = "dash" if role == "bound" else "solid"
                if x == "carriers":
                    vol = b.get("volume_cm3") if data.get("dim") == 3 else None
                    for sign, kind, d in ((1, "p-type", dash), (-1, "n-type", "dot" if role == "bound" else "dash")):
                        xs, ys = [], []
                        for n, v in zip(b["carriers"], y):
                            if n * sign > 0:
                                xs.append(abs(n) / vol if vol else abs(n))
                                ys.append(v)
                        if xs:
                            out.append({"name": f"{name}, {kind}", "x": xs, "y": ys, "color": k, "dash": d, "role": role})
                else:
                    out.append({"name": name, "x": list(b["mu"]), "y": list(y), "color": k, "dash": dash, "role": role})
            k += 1
    ref = data.get("reference")
    if ref and out:
        xs = [v for t in out for v in t["x"]]
        out.append({"name": ref["label"], "x": [min(xs), max(xs)], "y": [ref["value"], ref["value"]], "color": None,
                    "dash": "dot", "role": "reference"})
    return out
