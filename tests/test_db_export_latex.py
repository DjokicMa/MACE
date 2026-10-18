"""The LaTeX export compiles: special characters are escaped once.

_latex_escape replaced the characters in turn with the backslash last, so the
backslashes it had just inserted were escaped again ("1_dia" became
"1\\textbackslash{}_dia"), and the table did not compile.
"""
import shutil
import subprocess

import pytest

from mace.utils.db_export_formats import ExportFormatter

ROWS = [
    {"material_id": "1_dia", "formula": "C", "space_group": 227, "band_gap": 5.43,
     "total_energy_au": -75.8, "optimization_converged": True},
    {"material_id": "TiPbO3_mp-19845_sg221 & 50% #1 {x} ~^ $a\\b", "formula": "PbTiO3",
     "space_group": 221, "band_gap": 1.9, "total_energy_au": -1234.5,
     "optimization_converged": False},
]


def _latex():
    return ExportFormatter().render(ROWS, "latex", metadata={"export_type": "all_materials"}).decode()


def test_latex_escape_is_single_pass():
    e = ExportFormatter()._latex_escape
    assert e("1_dia") == r"1\_dia"
    assert e(r"a&b%c$d#e{f}g~h^i\j") == \
        r"a\&b\%c\$d\#e\{f\}g\textasciitilde{}h\^{}i\textbackslash{}j"
    text = _latex()
    assert r"\textbackslash{}_" not in text
    assert r"1\_dia" in text


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="no pdflatex")
def test_latex_export_compiles(tmp_path):
    doc = tmp_path / "t.tex"
    doc.write_text("\\documentclass{article}\n\\begin{document}\n" + _latex()
                   + "\n\\end{document}\n")
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", doc.name],
                       cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout[-2000:]
    assert (tmp_path / "t.pdf").is_file()
