"""One formula helper (mace.utils.formula) for the CLI and the GUI (CLI/GUI
parity, 1.1.6). The GUI's structure view (reduced formula and Z, the Hill
cell formula), its builders (ase's "metal" order), the Database view's
reduced formula, the CLI's materials.db formula and the ion-fragment Hill
formula all come from it.

tests/data/formula_golden.json holds what each of the former copies gave
(generated from the 1.1.6 code before they were joined): every element
alone, common compounds and random mixtures, plus the material id, formula
and cell formula the CLI writes for real corpus outputs and CIFs. Nothing
may change."""
import json
from collections import Counter
from pathlib import Path

import pytest

from conftest import REPO_ROOT, TEST_DATA

GOLDEN = json.loads((REPO_ROOT / "tests" / "data" / "formula_golden.json").read_text())


def _symbols(numbers):
    from mace.utils.formula import ELEMENTS
    return [ELEMENTS[n - 1] for n in numbers]


def test_the_helper_gives_every_former_format():
    from mace.utils import formula as F
    for c in GOLDEN["cases"]:
        counts = Counter(_symbols(c["numbers"]))
        assert list(F.reduced_formula(counts)) == c["gui_reduced"], c
        assert F.format_formula(counts, "hill-ase") == c["gui_hill"], c
        assert F.format_formula(counts, "hill") == c["ion_fragment_hill"], c
        assert F.format_formula(counts, "metal") == c["builders_metal"], c
        assert F.format_formula(counts, "mace") == c["cli_format"], c
        parsed = F.parse_formula(c["cli_format"])
        assert (list(F.reduced_formula(parsed)) if parsed else None) == c["db_text_reduced"], c


def test_the_former_entry_points_are_unchanged():
    pytest.importorskip("ase")
    from mace.gui import builders, structures
    from mace.utils import formula_extractor as fe
    from mace.utils.ion_fragments import formula_string
    for c in GOLDEN["cases"]:
        n = c["numbers"]
        assert list(structures.reduced_formula(n)) == c["gui_reduced"]
        assert structures._formula(n) == c["gui_hill"]
        assert builders._formula(n) == c["builders_metal"]
        assert builders._plane_name(c["builders_metal"]) == c["plane_name"]
        assert (list(fe._formulas(n, False) or []) or None) == c["cli_formulas"]
        assert (list(fe._formulas(n, True) or []) or None) == c["cli_formulas_mol"]
        assert fe._format_formula(Counter(_symbols(n))) == c["cli_format"]
        assert formula_string(dict(Counter(_symbols(n)))) == c["ion_fragment_hill"]
        assert (list(structures.reduce_formula_text(c["cli_format"]) or []) or None) == c["db_text_reduced"]


def test_no_module_keeps_its_own_formula_tables():
    """The tables live in mace.utils.formula only: the GUI and the CLI import them."""
    from mace.utils import formula as F
    from mace.utils import formula_extractor as fe
    assert fe._NONMETAL_ORDER is F.NONMETAL_ORDER
    for mod in ("mace/gui/structures.py", "mace/gui/builders.py", "mace/utils/formula_extractor.py",
                "mace/utils/ion_fragments.py"):
        text = (REPO_ROOT / mod).read_text()
        assert "_PAULING = " not in text and "from ase.formula import" not in text, mod
        assert "def _reduce(" not in text, mod


def test_material_ids_and_formulas_of_real_outputs_are_unchanged():
    pytest.importorskip("ase", reason="the formulas come from the structure reader, which needs ase")
    if not TEST_DATA.is_dir():
        pytest.skip("test/ data corpus not present (gitignored)")
    from mace.database.materials import create_material_id_from_file
    from mace.utils import formula_extractor as fe
    for c in GOLDEN["corpus"]:
        p = TEST_DATA / c["file"]
        if not p.is_file():
            continue
        assert create_material_id_from_file(str(p)) == c["material_id"], c["file"]
        if p.suffix == ".cif":
            assert fe.extract_formula_from_cif(p) == c["formula"], c["file"]
            continue
        d12 = p.with_suffix(".d12")
        info = fe.material_info(output_file=p, d12_file=d12 if d12.exists() else None)
        got = {k: info.get(k) for k in ("formula", "cell_formula", "dimensionality")}
        assert got == {k: c[k] for k in ("formula", "cell_formula", "dimensionality")}, c["file"]
