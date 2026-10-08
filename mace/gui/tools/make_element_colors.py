"""Regenerate mace/gui/static/element_colors.json (optional; needs pyyaml).

    python mace/gui/tools/make_element_colors.py path/to/ElementColorSchemes.yaml

The source is pymatgen's src/pymatgen/vis/ElementColorSchemes.yaml (MIT
License, The Materials Project), which carries the VESTA and Jmol element
colors. The committed JSON was made from pymatgen commit
9100860d7d938560610bcfadd04923b53756548e (2024-06-26).
"""
import json
import sys
from pathlib import Path

PYMATGEN_COMMIT = "9100860d7d938560610bcfadd04923b53756548e"
OUT = Path(__file__).resolve().parents[1] / "static" / "element_colors.json"
FALLBACK = "#ff1493"   # unknown element: deep pink, as VESTA and Jmol do


def build(yaml_path):
    import yaml
    from ase.data import chemical_symbols

    src = yaml.safe_load(Path(yaml_path).read_text())
    schemes = {}
    for name in ("VESTA", "Jmol"):
        table = src[name]
        schemes[name.lower()] = {sym: "#%02x%02x%02x" % tuple(table[sym])
                                 for sym in chemical_symbols[1:104] if sym in table}
    return {"source": "pymatgen src/pymatgen/vis/ElementColorSchemes.yaml "
                      f"(commit {PYMATGEN_COMMIT}), MIT License, The Materials Project",
            "default": "vesta", "fallback": FALLBACK, "schemes": schemes}


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(sys.argv[1]), indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT}")
