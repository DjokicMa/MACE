"""The text of every figure MACE draws: one entry per symbol and unit, written in plotly's markup.

mace/gui/static/js/labels.js is its twin (the GUI and mace plotting label a figure alike;
tests/test_plot_labels.py holds the two to the same tables). It lives in mace.utils so the readers the GUI
shares (transport_plot) import it without the plotting package.

The markup: <sub>, <sup> and <br>, a true minus (U+2212), and μ, Å, ° as characters. plotly draws it as
it is (its HTML, PNG and SVG files alike); ``mathtext`` turns a label into matplotlib's (the sub- and
superscripts as $_\\mathregular{F}$, in the text font, so a PDF/SVG reads as the plotly figure), and
``plain`` into text for where nothing draws markup (CSV/TSV headers, the terminal, file names):
E − E_F, cm⁻¹, κₑ.

Units go in parentheses after the quantity, "Name (unit)" (``with_unit``). A compound unit is written
with negative powers (W m⁻¹ K⁻¹); the energy units keep their menu names (eV, kJ/mol), and a DOS
counts states per energy unit (states/eV).
"""

from __future__ import annotations

import html
import re
from typing import Optional

MINUS = "−"

# a symbol with its sub- or superscript
SYMBOLS = {
    "E_F": "E<sub>F</sub>",
    "E_vac": "E<sub>vac</sub>",
    "E_ref": "E<sub>ref</sub>",
    "E_min": "E<sub>min</sub>",
    "E_final": "E<sub>final</sub>",
    "E_g": "E<sub>g</sub>",
    "kappa_0": "κ<sub>0</sub>",
    "kappa_e": "κ<sub>e</sub>",
    "kappa_L": "κ<sub>L</sub>",
    "ZT_e": "ZT<sub>e</sub>",
    "L_0": "L<sub>0</sub>",
    "S2": "S<sup>2</sup>",
    "C_V": "C<sub>V</sub>",
    "e_1": "e<sub>1</sub>",
    "e_2": "e<sub>2</sub>",
    "log10": "log<sub>10</sub>",
    "psi2": "|ψ|<sup>2</sup>",
}

# a unit by its plain name (the menus' and the files' spelling) -> the label drawn
UNITS = {
    "eV": "eV", "meV": "meV", "Ha": "Ha", "kJ/mol": "kJ/mol", "kcal/mol": "kcal/mol", "THz": "THz",
    "cm-1": f"cm<sup>{MINUS}1</sup>",
    "cm-3": f"cm<sup>{MINUS}3</sup>",
    "A": "Å",
    "A3": "Å<sup>3</sup>",
    "A-1": f"Å<sup>{MINUS}1</sup>",
    "deg": "°",
    "K": "K",
    "fs": "fs",
    "nm-2": f"nm<sup>{MINUS}2</sup>",
    "km/mol": f"km mol<sup>{MINUS}1</sup>",
    "J/(K mol)": f"J K<sup>{MINUS}1</sup> mol<sup>{MINUS}1</sup>",
    "e/bohr3": f"e bohr<sup>{MINUS}3</sup>",
    "bohr-3": f"bohr<sup>{MINUS}3</sup>",
    "bohr-3/2": f"bohr<sup>{MINUS}3/2</sup>",
    "Ha/e": f"Ha e<sup>{MINUS}1</sup>",
    "uV/K": f"µV K<sup>{MINUS}1</sup>",
    "S/m": f"S m<sup>{MINUS}1</sup>",
    "A/(m K)": f"A m<sup>{MINUS}1</sup> K<sup>{MINUS}1</sup>",
    "W/(m K)": f"W m<sup>{MINUS}1</sup> K<sup>{MINUS}1</sup>",
    "uW/(cm K2)": f"µW cm<sup>{MINUS}1</sup> K<sup>{MINUS}2</sup>",
    "1e-8 W Ohm/K2": f"10<sup>{MINUS}8</sup> W Ω K<sup>{MINUS}2</sup>",
    "1e18 1/(Ohm m s)": f"10<sup>18</sup> Ω<sup>{MINUS}1</sup> m<sup>{MINUS}1</sup> s<sup>{MINUS}1</sup>",
    "1e11 W/(m K2 s)": f"10<sup>11</sup> W m<sup>{MINUS}1</sup> K<sup>{MINUS}2</sup> s<sup>{MINUS}1</sup>",
    "1/hbar2 eV fs/A": f"ħ<sup>{MINUS}2</sup> eV fs Å<sup>{MINUS}1</sup>",
}


def unit(name: Optional[str]) -> str:
    """A unit's label from its plain name ("cm-1" -> cm<sup>−1</sup>); another text as it is."""
    return UNITS.get(name or "", name or "")


def with_unit(name: str, u: Optional[str]) -> str:
    """"Name (unit)", the unit by its plain name or as markup; the name alone without one."""
    return f"{name} ({unit(u)})" if u else name


def shift_text(shift: float) -> str:
    """' + 0.50 eV' for a Fermi shift (two decimals, half up), '' without one (bandfig.js shiftText)."""
    if not shift:
        return ""
    from decimal import ROUND_HALF_UP, Decimal
    q = Decimal(abs(shift)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f" {'+' if shift > 0 else MINUS} {q} eV"


def energy_title(ref: Optional[str], shift: float = 0.0) -> str:
    """The energy axis of a band or DOS plot: E − E_F, E − E_vac on the vacuum scale, and with a Fermi
    shift s, which moves the data by −s, E − (E_F + s)."""
    zero = SYMBOLS["E_vac"] if ref == "vacuum" else SYMBOLS["E_F"]
    return f"E {MINUS} ({zero}{shift_text(shift)})" if shift else f"E {MINUS} {zero}"


def fermi_key(shift: float = 0.0) -> str:
    """The dotted Fermi line's legend entry: E_F, or E_F + s."""
    return f"{SYMBOLS['E_F']}{shift_text(shift)}"


# ---------------------------------------------------------------- the markup for other targets
_TAG = re.compile(r"<(sub|sup)>(.*?)</\1>", re.S)
_SUB = dict(zip("0123456789+-−=()e", "₀₁₂₃₄₅₆₇₈₉₊₋₋₌₍₎ₑ"))
_SUP = dict(zip("0123456789+-−=()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁻⁼⁽⁾"))


def _strip(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]*>", "", s))


def plain(s) -> str:
    """A label as text (CSV/TSV headers, the terminal, file names): a sub- or superscript in Unicode
    where it has one (κₑ, L₀, cm⁻¹, Å³), else E_F, E_min, x^(−3/2); <br> a space, other tags dropped."""
    s = re.sub(r"<br\s*/?>", " ", str(s if s is not None else ""))

    def one(m):
        tag, body = m.group(1), _strip(m.group(2))
        table = _SUB if tag == "sub" else _SUP
        if body and all(c in table for c in body):
            return "".join(table[c] for c in body)
        mark = "_" if tag == "sub" else "^"
        return f"{mark}{body}" if len(body) <= 1 or tag == "sub" else f"{mark}({body})"
    return _strip(_TAG.sub(one, s))


def mathtext(s) -> str:
    """A label for matplotlib: the sub- and superscripts as mathtext in the text font
    (E$_{\\mathregular{F}}$), <br> a new line, other tags dropped. A label without markup is as it is
    (one that is mathtext already, $E_g$, too)."""
    s = re.sub(r"<br\s*/?>", "\n", str(s if s is not None else ""))

    def one(m):
        body = _strip(m.group(2)).replace(" ", r"\ ").replace("{", r"\{").replace("}", r"\}")
        return f"${'_' if m.group(1) == 'sub' else '^'}{{\\mathregular{{{body}}}}}$"
    out = _TAG.sub(one, s)
    return html.unescape(re.sub(r"</?(b|i|span)\b[^>]*>", "", out))


def has_markup(s) -> bool:
    return bool(re.search(r"</?(sub|sup|br)\b", str(s or "")))
