"""Total thermoelectric figure of merit of a BOLTZTRA run with a lattice kappa.

ZT_total = S^2 * sigma * T / (kappa_e + kappa_L)

* S (SEEBECK.dat, V/K), sigma (SIGMA.dat, 1/(Ohm m)) and kappa0 (KAPPA.dat,
  W/(m K), the zero-field K coefficient) are the diagonal averages the
  database extractor reads (``CrystalPropertyExtractor._parse_transport_dat``);
  kappa_e = kappa0 - T*sigma*S^2, as for the electronic ZT_e. For an
  open-shell run sigma and kappa0 are the alpha + beta totals (CRYSTAL23
  manual p. 315: separate alpha and beta blocks; SEEBECK.dat is not spin
  dependent).
* kappa_L is the user's lattice thermal conductivity in W/(m K): one value,
  or a table kappa_L(T) (a CSV of ``T,kappa_L`` rows) interpolated linearly
  inside its temperature range and never extrapolated.

The relaxation time. CRYSTAL23's transport distribution function (manual
Eq. 14.4) carries the constant relaxation time tau of RELAXTIM (in fs,
default 10 fs), and SIGMA.dat and KAPPA.dat are written "in SI units"
(1/Ohm/m and W/m/K): they are sigma and kappa0 at that tau, not per unit
tau. Both scale linearly with tau, so ZT_e = S^2 sigma T / kappa_e does not
depend on it, but ZT_total does: kappa_L carries no tau. ZT_total is
therefore evaluated at the run's tau, read from the ``RELAXATION TIME (fs)``
line of its .out, or at a tau the user gives, with sigma and kappa0 scaled
by tau / tau_run (S does not depend on tau). With no .out tau the total is
refused rather than guessed.

The precision of kappa_e. The .dat files print 8 significant digits, so each
value carries a relative rounding error of at most 5e-8: kappa0 5e-8, S^2 1e-7,
sigma 5e-8, and kappa_e = kappa0 - T sigma S^2 is uncertain by up to about
2e-7 * kappa0 where T sigma S^2 ~ kappa0. In the runs checked (diamond,
graphene, polyacetylene) every negative kappa_e, which is noise by definition,
is within 1.35e-7 * kappa0. A kappa_e below KAPPA_E_EPS = 2e-6 * kappa0 (10 times
that bound, so a kept kappa_e is good to 10%) is cancellation noise: ZT_e and
kappa_e are not computed there (diamond's ZT_e reached 1.6e9 at such a point).
The total ZT does not need kappa_e to be known well where kappa_L dominates: at a
masked point it is S^2 sigma T / kappa_L (kappa_e taken as its value, at least 0)
when the uncertainty of kappa_e, KAPPA_E_EPS * kappa0, is at most 1% of kappa_L,
and left out otherwise.

The CLI (``mace analyze --extract-properties --kappa-lattice``) and the GUI's
transport plot both compute through :func:`zt_points`, so they give the
same numbers.
"""
from __future__ import annotations

import csv
import json
import hashlib
import io
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

UNIT = "W/(m·K)"
# temperatures in the .dat files carry 3 decimals; a table end within this of
# a run temperature covers it
_T_TOL = 1e-6

# kappa_e below this fraction of kappa0 is the files' rounding noise (see the module docstring)
KAPPA_E_EPS = 2e-6
# at such a point the total ZT is kept while the kappa_e uncertainty is at most this fraction of kappa_L
KAPPA_L_SHARE = 0.01

_RELAX_RE = re.compile(r"RELAXATION TIME \(fs\)\s*:?\s*([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)")


class KappaLatticeError(ValueError):
    """A lattice kappa (or tau) MACE refuses, with the reason in the message."""


@dataclass
class KappaLattice:
    """A lattice thermal conductivity in W/(m K): one value or a table of T."""

    kind: str                                  # "constant" or "table"
    value: Optional[float] = None              # constant
    temps: List[float] = field(default_factory=list)    # table, ascending
    values: List[float] = field(default_factory=list)
    path: Optional[str] = None
    sha256: Optional[str] = None

    @property
    def t_min(self) -> Optional[float]:
        return self.temps[0] if self.temps else None

    @property
    def t_max(self) -> Optional[float]:
        return self.temps[-1] if self.temps else None

    def covers(self, T: float) -> bool:
        if self.kind == "constant":
            return True
        return self.temps[0] - _T_TOL <= T <= self.temps[-1] + _T_TOL

    def range_text(self) -> str:
        if self.kind == "constant":
            return "any temperature"
        if len(self.temps) == 1:
            return f"{_g(self.temps[0])} K only"
        return f"{_g(self.temps[0])}-{_g(self.temps[-1])} K"

    def at(self, T: float) -> float:
        """kappa_L at T; a table is interpolated linearly, never extrapolated."""
        if self.kind == "constant":
            return float(self.value)
        if not self.covers(T):
            raise KappaLatticeError(
                f"the kappa_L table {Path(self.path).name if self.path else ''} covers "
                f"{self.range_text()}; {_g(T)} K is outside it and MACE does not "
                f"extrapolate. Add rows that reach {_g(T)} K, or give one kappa_L value.")
        ts, vs = self.temps, self.values
        if T <= ts[0]:
            return vs[0]
        if T >= ts[-1]:
            return vs[-1]
        for i in range(1, len(ts)):
            if T <= ts[i]:
                t0, t1, v0, v1 = ts[i - 1], ts[i], vs[i - 1], vs[i]
                return v0 + (v1 - v0) * (T - t0) / (t1 - t0)
        return vs[-1]   # unreachable

    def source(self) -> str:
        """What the database records as the kappa_L source."""
        if self.kind == "constant":
            return f"constant {_g(self.value)} {UNIT}"
        n = len(self.temps)
        return (f"table {self.path} (sha256 {self.sha256[:12]}, {n} point{'s' if n != 1 else ''}, "
                f"{self.range_text()}, linear, no extrapolation; {UNIT})")

    def label(self) -> str:
        """A short legend text, in plot markup (mace.utils.plot_labels; its plain() for the terminal)."""
        from mace.utils import plot_labels as L
        if self.kind == "constant":
            return f"{L.SYMBOLS['kappa_L']} = {_g(self.value)} {L.unit('W/(m K)')}"
        return f"{L.SYMBOLS['kappa_L']}(T) from {Path(self.path).name}"


def _g(x: float) -> str:
    return f"{x:.6g}"


def _positive(x: float, what: str) -> float:
    if not math.isfinite(x) or x <= 0.0:
        raise KappaLatticeError(f"{what} must be a positive number; got {x!r}")
    return float(x)


def parse_kappa_lattice(spec, base: Optional[Path] = None) -> KappaLattice:
    """A ``--kappa-lattice`` value: a number (W/(m K)) or a CSV file of T,kappa_L.

    The CSV: optional ``#`` comment lines, an optional header line (``T,kappa_L``),
    then one ``T,kappa_L`` row per temperature (K, W/(m K)); at least one row,
    every T and kappa_L positive, no temperature twice. Rows are sorted by T.
    """
    if isinstance(spec, (int, float)):
        return KappaLattice("constant", value=_positive(float(spec), "kappa_L"))
    text = str(spec).strip()
    if not text:
        raise KappaLatticeError("kappa_L is empty: give a value in W/(m·K) or a T,kappa_L CSV file")
    try:
        number = float(text)
    except ValueError:
        number = None
    if number is not None:
        return KappaLattice("constant", value=_positive(number, "kappa_L"))
    p = Path(text).expanduser()
    if base is not None and not p.is_absolute():
        p = Path(base) / p
    if not p.is_file():
        raise KappaLatticeError(f"kappa_L {text!r} is neither a number nor a file")
    raw = p.read_bytes()
    return _parse_table(raw.decode("utf-8-sig", errors="replace"), str(p.resolve()),
                        hashlib.sha256(raw).hexdigest())


def _parse_table(text: str, path: str, sha: str) -> KappaLattice:
    name = Path(path).name
    rows: List[Tuple[float, float]] = []
    header_seen = False
    for lineno, rec in enumerate(csv.reader(io.StringIO(text)), start=1):
        cells = [c.strip() for c in rec]
        if not cells or not any(cells) or cells[0].startswith("#"):
            continue
        cells = [c for c in cells if c != ""]
        try:
            vals = [float(c) for c in cells]
        except ValueError:
            if not rows and not header_seen:
                header_seen = True     # "T,kappa_L"
                continue
            raise KappaLatticeError(f"{name} line {lineno}: {','.join(cells)!r} is not two numbers T,kappa_L")
        if len(vals) != 2:
            raise KappaLatticeError(f"{name} line {lineno}: expected two columns T,kappa_L, found {len(vals)}")
        T, k = vals
        if not (math.isfinite(T) and T > 0.0):
            raise KappaLatticeError(f"{name} line {lineno}: the temperature must be positive (K); got {T!r}")
        if not (math.isfinite(k) and k > 0.0):
            raise KappaLatticeError(f"{name} line {lineno}: kappa_L must be positive (W/(m·K)); got {k!r}")
        rows.append((T, k))
    if not rows:
        raise KappaLatticeError(f"{name} has no T,kappa_L rows")
    rows.sort()
    for (t0, _), (t1, _) in zip(rows, rows[1:]):
        if abs(t1 - t0) <= _T_TOL:
            raise KappaLatticeError(f"{name}: the temperature {_g(t0)} K appears twice")
    return KappaLattice("table", temps=[r[0] for r in rows], values=[r[1] for r in rows], path=path, sha256=sha)


# where a tau read from a ``mace tau --save`` file came from: {tau_fs: text}
_TAU_PROVENANCE: Dict[float, str] = {}


def tau_from_file(path: str) -> Tuple[float, str]:
    """(tau_fs, provenance) of a JSON file written by ``mace tau ... --save``
    (mace.transport: deformation-potential theory, or tau from a mobility)."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise KappaLatticeError(f"{path}: not a mace tau result ({exc})")
    if not isinstance(data, dict) or "tau_fs" not in data:
        raise KappaLatticeError(f"{path}: no tau_fs in it (write it with mace tau ... --save)")
    tau = _positive(float(data["tau_fs"]), "the relaxation time (fs)")
    what = data.get("method", "mace tau")
    extra = " ".join(x for x in (data.get("carrier"), data.get("axis")) if x)
    return tau, f"{Path(path).name}: {what}{' (' + extra + ')' if extra else ''}"


def check_tau_fs(tau) -> Optional[float]:
    """A relaxation time in fs (None stays None), or the tau of a ``mace tau
    --save`` JSON file (its provenance kept for resolve_tau)."""
    if tau is None or (isinstance(tau, str) and not tau.strip()):
        return None
    if isinstance(tau, str) and tau.strip().lower().endswith(".json"):
        t, prov = tau_from_file(tau.strip())
        _TAU_PROVENANCE[t] = prov
        return t
    try:
        t = float(tau)
    except (TypeError, ValueError):
        raise KappaLatticeError(f"the relaxation time must be a number of fs; got {tau!r}")
    return _positive(t, "the relaxation time (fs)")


def read_relaxation_time_fs(out_text: str) -> Optional[float]:
    """tau of a BOLTZTRA run, from its .out's ``RELAXATION TIME (fs):`` line."""
    m = _RELAX_RE.search(out_text or "")
    if not m:
        return None
    try:
        t = float(m.group(1).replace("D", "E").replace("d", "e"))
    except ValueError:
        return None
    return t if math.isfinite(t) and t > 0.0 else None


def resolve_tau(tau_run_fs: Optional[float], tau_fs: Optional[float], out_name: str = "the .out"):
    """(tau used, scale = tau / tau_run, how tau was chosen); raises when tau_run is unknown."""
    if tau_run_fs is None:
        raise KappaLatticeError(
            f"{out_name} does not say which relaxation time (RELAXTIM) BOLTZTRA used, and SIGMA.dat and "
            f"KAPPA.dat are at that time: the total ZT depends on it (kappa_L carries no tau), so MACE "
            f"does not compute it")
    if tau_fs is None:
        return tau_run_fs, 1.0, "the run's RELAXTIM"
    how = _TAU_PROVENANCE.get(tau_fs, "given")
    return tau_fs, tau_fs / tau_run_fs, (f"{how}; the run used {_g(tau_run_fs)} fs, so σ and κ₀ are "
                                         f"scaled by {_g(tau_fs / tau_run_fs)}")


def kappa_e_state(kappa0: float, kappa_e: float) -> str:
    """"ok" (kappa_e is known to 10% or better), "masked" (|kappa_e| below the files' precision,
    KAPPA_E_EPS * |kappa0|) or "negative" (kappa_e <= 0 beyond it: outside the model)."""
    tol = KAPPA_E_EPS * abs(kappa0)
    if kappa_e >= tol:
        return "ok"
    return "masked" if kappa_e > -tol else "negative"


def _lut(rows) -> Dict[Tuple[float, float], float]:
    """(T, mu) -> value. The rows are the extractor's, an open-shell file's
    alpha and beta blocks already summed (``_parse_transport_dat``); a point
    that comes twice (spin blocks not combined) is refused rather than one
    block silently overwriting the other."""
    lut: Dict[Tuple[float, float], float] = {}
    for (T, mu, _nc, v) in rows:
        key = (round(T, 3), round(mu, 4))
        if key in lut:
            raise ValueError(f"the point T = {T:g} K, mu = {mu:g} eV comes twice (spin blocks not combined?)")
        lut[key] = v
    return lut


def zt_points(s_rows: Sequence, sig_rows: Sequence, kap_rows: Sequence,
              kappa_l: Optional[KappaLattice] = None, scale: float = 1.0) -> Dict:
    """ZT_e and ZT_total at every (T, mu) of SEEBECK.dat, with the extractor's rules.

    A point counts where |sigma| is at least 1e-3 of the run's peak |sigma|
    (BoltzTraP's determinant spikes are excluded), KAPPA.dat has a nonzero
    kappa0 and kappa_e = kappa0 - T sigma S^2 is above the files' precision
    (kappa_e_state "ok"). At a point whose kappa_e is below it ("masked") ZT_e
    is None and the total is S^2 sigma T / (max(kappa_e, 0) + kappa_L) while
    KAPPA_E_EPS * kappa0 is at most KAPPA_L_SHARE of kappa_L (None otherwise).
    ``scale`` multiplies sigma and kappa0 (tau / tau_run) for ZT_total; ZT_e
    does not depend on it.

    Returns ``{"temperatures": [...], "blocks": {T: {"mu", "zt_e", "zt_total",
    "kappa_l", "refused", "n_masked"}}, "zt_e_max", "zt_total_max", "n_masked"}``: a list entry is None
    where the point does not count; a block whose temperature the kappa_L
    table does not cover has ``zt_total`` None and the reason in ``refused``.
    A peak is ``{"value", "temperature_k", "mu_ev", "kappa0", "kappa_e",
    "kappa_l"}`` (the first point of the highest value) or None.
    """
    sig_lut = _lut(sig_rows)
    kap_lut = _lut(kap_rows)
    sig_abs = [abs(v) for v in sig_lut.values() if v != 0.0]
    sig_floor = (max(sig_abs) if sig_abs else 0.0) * 1e-3
    blocks: Dict[float, Dict] = {}
    peak_e = peak_t = None
    for (T, mu, _nc, s_val) in s_rows:
        b = blocks.get(T)
        if b is None:
            b = blocks[T] = {"mu": [], "zt_e": [], "zt_total": [] if kappa_l is not None else None,
                             "kappa_l": None, "refused": None, "n_masked": 0}
            if kappa_l is not None:
                try:
                    b["kappa_l"] = kappa_l.at(T)
                except KappaLatticeError as e:
                    b["refused"] = str(e)
        b["mu"].append(mu)
        key = (round(T, 3), round(mu, 4))
        g = sig_lut.get(key)
        k = kap_lut.get(key)
        zt_e = zt_t = None
        kappa_e = None
        if (g is not None and sig_floor > 0.0 and abs(g) >= sig_floor and k and abs(k) > 0.0):
            pf_t = s_val * s_val * abs(g) * T
            kappa_e = abs(k) - pf_t
            state = kappa_e_state(k, kappa_e)
            if state == "masked":
                # kappa_e is rounding noise: no ZT_e; the total where kappa_L dominates that noise
                b["n_masked"] += 1
                kl = b["kappa_l"]
                if kl is not None and scale * KAPPA_E_EPS * abs(k) <= KAPPA_L_SHARE * kl:
                    ke = scale * max(kappa_e, 0.0)
                    zt_t = scale * pf_t / (ke + kl)
                    if peak_t is None or zt_t > peak_t["value"]:
                        peak_t = {"value": zt_t, "temperature_k": T, "mu_ev": mu, "kappa0": scale * abs(k),
                                  "kappa_e": ke, "kappa_l": kl}
            if state == "ok":
                zt_e = pf_t / kappa_e
                if peak_e is None or zt_e > peak_e["value"]:
                    peak_e = {"value": zt_e, "temperature_k": T, "mu_ev": mu, "kappa0": abs(k),
                              "kappa_e": kappa_e, "kappa_l": None}
                if b["kappa_l"] is not None:
                    kl = b["kappa_l"]
                    zt_t = scale * pf_t / (scale * kappa_e + kl)
                    if peak_t is None or zt_t > peak_t["value"]:
                        peak_t = {"value": zt_t, "temperature_k": T, "mu_ev": mu, "kappa0": scale * abs(k),
                                  "kappa_e": scale * kappa_e, "kappa_l": kl}
        b["zt_e"].append(zt_e)
        if b["zt_total"] is not None:
            b["zt_total"].append(zt_t)
    for b in blocks.values():
        if b["refused"]:
            b["zt_total"] = None
    return {"temperatures": sorted(blocks), "blocks": blocks, "zt_e_max": peak_e, "zt_total_max": peak_t,
            "n_masked": sum(b["n_masked"] for b in blocks.values())}


def uncovered(kappa_l: KappaLattice, temperatures) -> List[float]:
    """The run temperatures a kappa_L table does not cover."""
    return [T for T in temperatures if not kappa_l.covers(T)]
