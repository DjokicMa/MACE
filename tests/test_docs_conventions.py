"""The user docs state the conventions the code assumes.

* docs/defects/README.md: the POTC alignment is applied as q x (-dV); which
  sign convention of V0 that assumes, and that it is unverified, was not said.
* DOCUMENTATION.md (phonopy): phonopy's standard masses against CRYSTAL's
  isotopic ones shift frequencies (about 1 cm-1 at 540 cm-1 for Si, measured);
  nothing mentioned masses.
* INSTALLATION.md (HPC): the module install steps fail on a login gateway
  ("module(s) could not be loaded: Python/3.11.3-GCCcore-12.3.0" on an MSU
  gateway node, measured; the same load works on a development node).
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


def test_stuttgart_library_says_its_records_are_unverified():
    import basis_library
    text = " ".join(basis_library.load_family("stuttgart").description.split())
    assert "the other 117 are unverified" in text
    assert "every deck path refuses them" in text


def test_phonopy_docs_say_the_decks_are_opt2d12s():
    text = " ".join(_text("DOCUMENTATION.md").split())
    assert "the set has no phonopy `TEMPLATE`: rewrite a set with opt2d12" in text
    assert "`BIPOSIZE`/`EXCHSIZE 110000000`" in text


def test_installation_says_to_load_job_modules_off_the_gateway():
    text = _text("INSTALLATION.md")
    assert "Run these on a development or compute node" in text
    assert "login gateway may not serve software modules" in text
