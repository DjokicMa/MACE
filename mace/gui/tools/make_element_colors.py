"""Regenerate mace/gui/static/element_colors.json (optional; needs pyyaml).

    python mace/gui/tools/make_element_colors.py path/to/ElementColorSchemes.yaml

The source is pymatgen's src/pymatgen/vis/ElementColorSchemes.yaml (MIT
License, The Materials Project), which carries the VESTA and Jmol element
colors. The committed JSON was made from pymatgen commit
9100860d7d938560610bcfadd04923b53756548e (2024-06-26).

The "cvd" (colorblind-safe) scheme is MACE's own (MIT, as MACE), made by
cvd_scheme() below: each family of elements takes one hue of the Okabe-Ito
palette (Okabe and Ito, "Color Universal Design", 2002/2008; a published
recommendation of eight colors, free to use), and the periods step its
lightness, so neighbors differ in lightness, which every kind of color
vision keeps. The common elements are set one by one.
"""
import json
import sys
from pathlib import Path

PYMATGEN_COMMIT = "9100860d7d938560610bcfadd04923b53756548e"
OUT = Path(__file__).resolve().parents[1] / "static" / "element_colors.json"
FALLBACK = "#ff1493"   # unknown element: deep pink, as VESTA and Jmol do


SOURCE = ("pymatgen src/pymatgen/vis/ElementColorSchemes.yaml "
          f"(commit {PYMATGEN_COMMIT}), MIT License, The Materials Project (vesta, jmol); "
          "cvd: MACE's own, from the Okabe-Ito colorblind-safe hues; legacy: MACE's own pre-1.1.6 "
          "mace plotting colors (see element_colors.LICENSE.txt)")
LABELS = {"vesta": "VESTA (MACE default)", "jmol": "Jmol (CPK)", "cvd": "Colorblind safe",
          "legacy": "Legacy (mace plotting before 1.1.6)"}

# Plotting/plottingCIFs.py's ELEMENT_COLORS before 1.1.6 (MACE's own, MIT); every other
# element was drawn #808080. `mace plotting --palette legacy` draws each plot's own old table.
LEGACY_TABLE = {"H": "#ffffff", "C": "#808080", "N": "#0000ff", "O": "#ff0000", "F": "#ffff00", "Cl": "#00ff00",
                "Br": "#a52a2a", "I": "#940094", "Fe": "#ffa500", "Au": "#ffd700", "Ag": "#c0c0c0", "Cu": "#ff7f50",
                "Si": "#daa520"}


def legacy_scheme():
    """The legacy preset (Z = 1..103): the old structure plotter's colors, gray elsewhere."""
    return {s: LEGACY_TABLE.get(s, "#808080") for s in cvd_scheme()}


def _ramp(a, b, n):
    """n colors from a to b (inclusive), linear in sRGB."""
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    if n == 1:
        return [a]
    return ["#%02x%02x%02x" % tuple(round(x + (y - x) * k / (n - 1)) for x, y in zip(ca, cb)) for k in range(n)]


def cvd_scheme():
    """MACE's colorblind-safe element colors (Z = 1..103)."""
    fam = {
        # alkali metals: reddish purple (Okabe-Ito #cc79a7), light to dark
        "Li Na K Rb Cs Fr": ("#f0c4dd", "#8c4a78"),
        # alkaline earths: orange (#e69f00)
        "Be Mg Ca Sr Ba Ra": ("#ffe0a3", "#9c6b00"),
        # 3d, 4d, 5d transition metals: sky blue (#56b4e9) to blue (#0072b2), each row its own lightness
        "Sc Ti V Cr Mn Fe Co Ni Cu Zn": ("#a8d8f5", "#2a7ab0"),
        "Y Zr Nb Mo Tc Ru Rh Pd Ag Cd": ("#9cc4e0", "#1f5482"),
        "Hf Ta W Re Os Ir Pt Au Hg": ("#b8b8d6", "#505078"),
        # post-transition metals: neutral to slate
        "Al Ga In Sn Tl Pb Bi Po": ("#c2c2c2", "#4a4a6e"),
        # metalloids
        "B Si Ge As Sb Te": ("#f4a582", "#8a6d3b"),
        # halogens: bluish green (#009e73)
        "F Cl Br I At": ("#9ce0c8", "#00553e"),
        # noble gases: pale cyan
        "He Ne Ar Kr Xe Rn": ("#e0f3fa", "#6aa8c0"),
        # lanthanides and actinides: green-teal and violet ramps
        "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu": ("#c6ead6", "#3f8c6c"),
        "Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr": ("#d9c6ea", "#5c3f8c"),
    }
    out = {}
    for syms, (a, b) in fam.items():
        s = syms.split()
        out.update(zip(s, _ramp(a, b, len(s))))
    # the common elements, one by one (Okabe-Ito where it has the hue)
    out.update({"H": "#e8e8e8", "C": "#404040", "N": "#0072b2", "O": "#d55e00", "P": "#e69f00",
                "S": "#f0e442", "Se": "#b8a800", "Si": "#56b4e9", "Au": "#d4a017", "Ag": "#c8c8c8",
                "Pt": "#9a9ab0", "Cu": "#c27a4a", "Fe": "#7a4f9e", "Pb": "#4a4a6e", "Ti": "#78c0e8"})
    return dict(sorted(out.items()))


def build(yaml_path):
    import yaml
    from ase.data import chemical_symbols

    src = yaml.safe_load(Path(yaml_path).read_text())
    schemes = {}
    for name in ("VESTA", "Jmol"):
        table = src[name]
        schemes[name.lower()] = {sym: "#%02x%02x%02x" % tuple(table[sym])
                                 for sym in chemical_symbols[1:104] if sym in table}
    schemes["cvd"] = cvd_scheme()
    schemes["legacy"] = legacy_scheme()
    return {"source": SOURCE, "default": "vesta", "fallback": FALLBACK, "schemes": schemes, "labels": LABELS}


def add_cvd(path=OUT):
    """Add (or remake) the cvd scheme and the labels in the committed JSON, without the YAML."""
    data = json.loads(Path(path).read_text())
    data["schemes"]["cvd"] = cvd_scheme()
    data["schemes"]["legacy"] = legacy_scheme()
    data["labels"] = LABELS
    data["source"] = SOURCE
    Path(path).write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    if sys.argv[1:] == ["--cvd-only"]:
        add_cvd()
    else:
        OUT.write_text(json.dumps(build(sys.argv[1]), indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT}")
