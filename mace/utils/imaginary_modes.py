"""Imaginary vibrational modes of a FREQ output: one rule for the GUI's run
analysis (mace/gui/runinfo.py) and the property extractor's
``num_imaginary_frequencies``."""
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

NOISE_CM = 20.0

_MOLECULE = re.compile(r'DIMENSIONALITY OF THE SYSTEM\s+0\b')


def is_molecule(text: str) -> bool:
    """The output is of a molecule (dimensionality 0)."""
    return bool(_MOLECULE.search(text))


def is_linear(points: Sequence[Tuple[float, float, float]]) -> bool:
    """The atoms (Cartesian) lie on a line: 2 atoms, or the second singular
    value of the centered coordinates is negligible."""
    pts = list(points)
    if len(pts) <= 2:
        return len(pts) == 2
    try:
        import numpy as np
        a = np.asarray(pts, float)
        a = a - a.mean(axis=0)
        s = np.linalg.svd(a, compute_uv=False)
        return bool(s[1] < 1e-3 * max(s[0], 1e-12))
    except Exception:
        return False


def imaginary(freqs: List[float], molecule: bool, linear: bool,
              noise_cm: float = NOISE_CM) -> Optional[Dict[str, Any]]:
    """{severity, n, lowest} for the negative modes left after dropping the
    expected zero modes (3 for a periodic system, 6 for a molecule, 5 for a
    linear one), or None. ``freqs`` has one entry per mode (degenerate rows
    expanded). Below ``noise_cm`` in size: info (usually numerical noise),
    else a warning, whose ``n`` counts only the modes beyond the noise."""
    if not freqs:
        return None
    drop = (5 if linear else 6) if molecule else 3
    order = sorted(range(len(freqs)), key=lambda i: abs(freqs[i]))
    rest = [freqs[i] for i in order[drop:]]
    neg = sorted(f for f in rest if f < 0)
    if not neg:
        return None
    lowest = neg[0]
    real = [f for f in neg if abs(f) >= noise_cm]
    if real:
        return {"severity": "warning", "n": len(real), "lowest": lowest}
    return {"severity": "info", "n": len(neg), "lowest": lowest}


def count_imaginary(freqs: List[float], molecule: bool, linear: bool,
                    noise_cm: float = NOISE_CM) -> int:
    """The number of imaginary modes beyond the noise (0 for none or noise only)."""
    im = imaginary(freqs, molecule, linear, noise_cm)
    return im["n"] if im and im["severity"] == "warning" else 0
