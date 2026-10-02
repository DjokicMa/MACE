"""The user docs state the conventions the code assumes.

* docs/defects/README.md: the POTC alignment is applied as q x (-dV); which
  sign convention of V0 that assumes, and that it is unverified, was not said.
* DOCUMENTATION.md (phonopy): phonopy's standard masses against CRYSTAL's
  isotopic ones shift frequencies (about 1 cm-1 at 540 cm-1 for Si, measured);
  nothing mentioned masses.
"""
from conftest import REPO_ROOT


def _text(path):
    return " ".join((REPO_ROOT / path).read_text().split())


def test_defects_readme_states_the_alignment_sign_convention():
    text = _text("docs/defects/README.md")
    assert "Sign convention (assumed, not verified)" in text
    assert "positive* test charge" in text and "q x (-dV)" in text


def test_phonopy_docs_explain_the_mass_offset():
    text = _text("DOCUMENTATION.md")
    assert "Masses differ from CRYSTAL's" in text
    assert "about 1 cm-1 at 540 cm-1 for Si" in text
