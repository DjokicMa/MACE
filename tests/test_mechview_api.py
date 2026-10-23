"""The results document of ELASTIC, EOS, QHA and CPHF outputs (1.1.6, mace/elastic/views.py,
GET /api/elastic/document and elastic/figure) and its CLI twin, mace analyze elastic --plot.

On the real HPCC runs of tests/data/hpcc_116/mechanics (README there): the document's numbers are
exactly what `mace analyze elastic --json` prints; the directional Young's modulus agrees with
CRYSTAL's own compliance; the four EOS curves agree with the P(V) and E(V) tables CRYSTAL prints
for each fit; the settings are read from the output and its deck. No real slab ELASTCON run or
CPHF DYNAMIC run exists in the repository: those two are checked on blocks written in CRYSTAL's
format (the parser's own 2D and frequency handling).
"""
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO_ROOT
from mace.elastic import analysis as A
from mace.elastic import views as V
from mace.gui import cache as gui_cache
from mace.gui import mechanics_routes as MR

MECH = REPO_ROOT / "tests" / "data" / "hpcc_116" / "mechanics"
MACE_CLI = REPO_ROOT / "mace_cli"
RUNS = ("dia_elastcon", "dia_elastcon_nuchess", "bn_elapiezo", "dia_elapiezo", "dia_photoela", "dia_eos", "dia_qha", "dia_cphf")
SLAB = """ SLAB CALCULATION
 PRIMITIVE CELL - CENTRING CODE 1/0 VOLUME=     5.240000 - DENSITY  0.000 g/cm^3
         A              B              C           ALPHA      BETA       GAMMA
     2.46000000     2.46000000   500.00000000    90.000000  90.000000 120.000000

 SYMMETRIZED ELASTIC CONSTANTS FOR HEXAGONAL     CASE, IN HARTREE

 |     1.1000    0.1900    0.0000 |
 |               1.1000    0.0000 |
 |                         0.4550 |

"""


class _St:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.parse_cache = gui_cache.ParseCache()

    def resolve(self, rel, must_exist=True):
        t = (self.root / rel).resolve()
        if t != self.root and self.root not in t.parents:
            raise PermissionError("outside the served directory")
        if must_exist and not t.exists():
            raise FileNotFoundError(rel)
        return t

    def rel(self, p):
        return Path(p).resolve().relative_to(self.root).as_posix()


class _Handler:
    def __init__(self, root):
        self.state = _St(root)
        self.headers = {}

    def _json(self, obj, status=200):
        return status, obj

    def _error(self, status, msg):
        return status, {"error": msg}


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    root = tmp_path_factory.mktemp("mechview_api")
    for f in MECH.iterdir():
        shutil.copy(f, root / f.name)
    (root / "graphene_elastcon.out").write_text(SLAB)
    return _Handler(root)


def _get(h, route, **q):
    return MR.handle(h, "GET", route, {k: [v] for k, v in q.items()})


def _cli_json(name):
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    r = subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "analyze", "elastic", str(MECH / f"{name}.out"), "--json"],
                       capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("name", RUNS)
def test_the_document_holds_what_mace_elastic_analyze_prints(served, name):
    code, body = _get(served, "elastic/document", path=f"{name}.out")
    assert code == 200, body
    cli = _cli_json(name)
    cli.pop("file")
    assert json.loads(json.dumps(body["data"])) == cli
    assert body["kind"] == {"dia_eos": "EOS", "dia_qha": "QHA", "dia_cphf": "CPHF"}.get(name, "ELASTIC")
    assert body["cli"] == f"mace analyze elastic {name}.out" and body["report"] == A.report(A.analyze_file(MECH / f"{name}.out"), f"{name}.out")


def test_elastcon_settings_born_criteria_and_directional_moduli(served):
    _, body = _get(served, "elastic/document", path="dia_elastcon.out")
    s = {r["label"]: (r["value"], r["source"]) for r in body["settings"]}
    assert s["Module"] == ("ELASTCON", "d12") and s["Options"][0] == "NUCHESS HESSNUM2 PRINT"
    assert s["NUMDERIV (points per strain)"] == ("3", "out") and s["STEPSIZE (strain step)"] == ("0.01", "out")
    assert s["Functional"][0] == "B3LYP-D3" and s["Periodicity"][0] == "3D crystal"
    born = {b["criterion"]: b for b in body["born"]}
    # diamond, cubic: C11 1150.388, C12 122.664, C44 618.258 GPa
    assert born["C11 − C12 > 0"]["value"] == pytest.approx(1150.388 - 122.664) and born["C44 > 0"]["value"] == pytest.approx(618.258)
    assert all(b["ok"] for b in body["born"])
    assert [v["key"] for v in body["views"]] == ["youngs", "compressibility", "shear", "poisson", "sound"]
    E = body["directional"]["youngs"]
    # E[100] = 1/S11 of CRYSTAL's printed compliance (0.8875 TPa^-1); for a cubic crystal
    # 1/E[111] = S11 - 2 (S11 - S12 - S44/2) / 3 (Nye 1985)
    s11, s12, s44 = 0.8875e-3, -0.0855e-3, 1.6174e-3
    assert E["min"]["label"] == "[1 0 0]" and E["min"]["value"] == pytest.approx(1 / s11, rel=2e-4)
    assert E["max"]["label"] == "[1 1 1]" and E["max"]["value"] == pytest.approx(1 / (s11 - 2 * (s11 - s12 - s44 / 2) / 3), rel=1e-3)
    beta = body["directional"]["compressibility"]
    # a cubic crystal's linear compressibility is isotropic: 1/(3 K) with K = 465.24 GPa
    assert beta["isotropic"] and beta["max"]["value"] == pytest.approx(1000 / (3 * 465.24), rel=1e-4)


def test_directional_figures_and_their_cli_line(served):
    code, f = _get(served, "elastic/figure", path="dia_elastcon.out", view="youngs", shape="surface")
    assert code == 200 and f["shape"] == "surface" and f["unit"] == "GPa"
    v = [x for row in f["surface"]["value"] for x in row]
    assert min(v) == pytest.approx(1126.7, abs=0.2) and max(v) <= 1285.4 + 0.1
    assert f["cli"] == "mace analyze elastic dia_elastcon.out --plot youngs --shape surface"
    assert f["table"]["columns"] == ["n_x", "n_y", "n_z", "E (GPa)"]
    code, f = _get(served, "elastic/figure", path="dia_elastcon.out", view="youngs", shape="planes")
    assert [c["plane"] for c in f["planes"]] == ["xy", "xz", "yz"] and f["cli"].endswith("--shape planes")
    xy = f["planes"][0]
    assert xy["r"][0] == pytest.approx(1126.7, abs=0.1) and xy["r"][45] == pytest.approx(max(xy["r"]), abs=0.1)   # [110] in the xy plane
    # a bad view or shape is a 400, a plot the run does not have names the ones it has
    assert _get(served, "elastic/figure", path="dia_elastcon.out", view="nope")[0] == 400
    assert _get(served, "elastic/figure", path="dia_elastcon.out", view="youngs", shape="cube")[0] == 400
    code, body = _get(served, "elastic/figure", path="dia_eos.out", view="youngs")
    assert code == 400 and "it has eos, pv" in body["error"]
    assert _get(served, "elastic/document", path=".")[0] == 400


def test_shear_poisson_sound_and_piezo_figures_and_the_epsilon_of_the_coupling(served):
    """1.1.6 (mechmore): G(n, m) and nu(n, m) as ELATE's min/max envelopes or plane cuts, the Christoffel
    velocities, the longitudinal piezoelectric coefficient, and k^2 with a given permittivity."""
    _, body = _get(served, "elastic/document", path="bn_elapiezo.out")
    assert [v["key"] for v in body["views"]] == ["youngs", "compressibility", "shear", "poisson", "sound", "piezo"]
    assert body["directional"]["shear"]["min"]["label2"] and body["data"]["derived"]["piezo"]["coupling"] == {"available": False}
    code, f = _get(served, "elastic/figure", path="dia_elastcon.out", view="shear", shape="surface")
    assert code == 200 and set(f["envelopes"]) == {"min", "max"} and f["cli"].endswith("--plot shear --shape surface")
    assert f["table"]["columns"] == ["n_x", "n_y", "n_z", "G_min (GPa)", "G_max (GPa)"]
    gmin = min(x for row in f["envelopes"]["min"]["value"] for x in row)
    assert gmin == pytest.approx((1150.388 - 122.664) / 2, rel=1e-3)       # diamond: (C11 - C12)/2 near <110>
    _, f = _get(served, "elastic/figure", path="dia_elastcon.out", view="poisson", shape="planes")
    assert [k["key"] for k in f["planes"][0]["curves"]] == ["min", "max"] and f["table"]["columns"] == ["plane", "angle (deg)", "ν_min", "ν_max"]
    _, f = _get(served, "elastic/figure", path="dia_elastcon.out", view="sound", shape="surface")
    assert f["shape"] == "planes" and [k["key"] for k in f["planes"][0]["curves"]][:3] == ["qL", "qS1", "qS2"]
    assert f["planes"][0]["curves"][0]["r"][0] == pytest.approx(17.929, abs=2e-3)     # CRYSTAL's Vp along [100]
    code, body = _get(served, "elastic/document", path="bn_elapiezo.out", epsilon="6.8")
    cp = body["data"]["derived"]["piezo"]["coupling"]
    assert code == 200 and cp["available"] and cp["epsilon_source"] == "given"
    assert body["cli"] == "mace analyze elastic bn_elapiezo.out --epsilon 6.8"
    assert body["data"] == json.loads(json.dumps(A.analyze_file(MECH / "bn_elapiezo.out", "6.8")))
    assert _get(served, "elastic/document", path="bn_elapiezo.out", epsilon="1 2")[0] == 400
    code, f = _get(served, "elastic/figure", path="bn_elapiezo.out", view="piezo", shape="planes")
    assert code == 200 and f["cli"].endswith("--plot piezo --shape planes") and f["planes"][0]["plane"] == "(1 −1 0)"


def test_the_enthalpy_route_compares_another_phase(served):
    """1.1.6 (mechmore): elastic/figure?view=enthalpy&compare=REL (Add phase… in the EOS document)."""
    from eos_synthetic import eos_output
    root = served.state.root
    (root / "c_hp_eos.out").write_text(eos_output(10.80, -76.21522625, 480.0, 3.7))
    (root / "bn_eos.out").write_text(eos_output(11.8, -79.5, 380.0, 3.8, atoms=(("5", "B"), ("7", "N"))))
    _, body = _get(served, "elastic/document", path="dia_eos.out")
    assert [v["key"] for v in body["views"]] == ["eos", "pv", "bp", "vp", "residuals", "enthalpy"]
    assert body["data"]["formula"] == "C" and body["data"]["eos"]["quality"]["birch_murnaghan"]["quality"] == "good"
    code, f = MR.handle(served, "GET", "elastic/figure", {"path": ["dia_eos.out"], "view": ["enthalpy"], "compare": ["c_hp_eos.out"]})
    assert code == 200 and f["compare"] == ["c_hp_eos.out"] and f["transitions"][0]["crossings"][0]["pressure_gpa"] == pytest.approx(26.9, abs=0.1)
    assert f["cli"] == "mace analyze elastic dia_eos.out --plot enthalpy --compare c_hp_eos.out"
    code, f = MR.handle(served, "GET", "elastic/figure", {"path": ["dia_eos.out"], "view": ["enthalpy"], "compare": ["c_hp_eos.out"], "fit": ["vinet"]})
    assert code == 200 and f["fit"] == "vinet" and f["cli"].endswith("--fit vinet")
    code, body = MR.handle(served, "GET", "elastic/figure", {"path": ["dia_eos.out"], "view": ["enthalpy"], "compare": ["bn_eos.out"]})
    assert code == 400 and "differ in composition" in body["error"]
    assert MR.handle(served, "GET", "elastic/figure", {"path": ["dia_eos.out"], "view": ["enthalpy"], "compare": ["../x.out"]})[0] in (400, 403, 404)
    assert MR.handle(served, "GET", "elastic/figure", {"path": ["dia_eos.out"], "view": ["pv"], "compare": ["c_hp_eos.out"]})[0] == 400


def _crystal_eos_tables(text):
    names = {"MURNAGHAN 1944": "murnaghan", "BIRCH-MURNAGHAN 1947": "birch_murnaghan",
             "POIRIER-TARANTOLA 1998": "poirier_tarantola", "VINET 1987": "vinet"}
    out = {}
    for m in re.finditer(r"THERMODYNAMIC FUNCTIONS OBTAINED AT 0 K WITH EOS: (.+?)\s*\n", text):
        rows = []
        for line in text[m.end():].splitlines()[6:40]:
            v = line.split()
            if len(v) == 5:
                rows.append([float(x) for x in v])
            elif rows:
                break
        out[names[m.group(1).strip()]] = rows
    return out


def test_the_eos_curves_are_crystal_s_own_tables(served):
    """E(V) and P(V) of each fit, from its V0, E0, B0, B' as CRYSTAL prints them (2-4 decimals), against
    the table CRYSTAL prints for that fit (PRINT): 25 volumes evenly over the computed ones."""
    text = (MECH / "dia_eos.out").read_text()
    d = A.analyze(text)
    pts, fits = d["eos"]["points"], d["eos"]["fits"]
    tables = _crystal_eos_tables(text)
    assert sorted(tables) == sorted(fits)
    for name, rows in tables.items():
        n = len(rows)
        vs = [pts[0][0] + (pts[-1][0] - pts[0][0]) * k / (n - 1) for k in range(n)]
        for v, r in zip(vs, rows):
            assert abs(V.eos_pressure(name, v, fits[name]) - r[1]) < 0.02, (name, v)
            assert abs(V.eos_energy(name, v, fits[name]) - r[2]) < 5e-7, (name, v)
    code, f = _get(served, "elastic/figure", path="dia_eos.out", view="eos")
    names = [s["name"] for s in f["series"]]
    assert names == ["Murnaghan 1944", "Birch-Murnaghan 1947", "Poirier-Tarantola 1998", "Vinet 1987", "computed"]
    comp = f["series"][-1]
    assert min(comp["y"]) == 0.0 and comp["x"] == [p[0] for p in pts]
    _, f = _get(served, "elastic/figure", path="dia_eos.out", view="pv")
    bm = next(s for s in f["series"] if s["name"] == "Birch-Murnaghan 1947")
    # P = 0 at V0
    k = min(range(len(bm["x"])), key=lambda i: abs(bm["x"][i] - fits["birch_murnaghan"]["volume_ang3"]))
    assert abs(bm["y"][k]) < 0.6
    _, body = _get(served, "elastic/document", path="dia_eos.out")
    s = {r["label"]: r["value"] for r in body["settings"]}
    assert s["Volume range"].startswith("0.920 – 1.080") and s["Volumes"].startswith("7 (10.2449")


def test_qha_views_and_settings(served):
    _, body = _get(served, "elastic/document", path="dia_qha.out")
    assert [v["key"] for v in body["views"]] == ["volume", "vratio", "expansion", "bulk", "heat"]
    s = {r["label"]: r["value"] for r in body["settings"]}
    assert s["Temperatures"] == "10 – 600 K, 60 points" and s["Volumes"] == "4"
    q = body["data"]["qha"]
    _, f = _get(served, "elastic/figure", path="dia_qha.out", view="expansion")
    assert f["series"][0]["y"] == q["alpha_v_per_k"]
    assert [x["name"] for x in f["series"]] == ["α<sub>V</sub>", "α<sub>a</sub> = α<sub>b</sub> = α<sub>c</sub>"]          # cubic: one linear curve
    _, f = _get(served, "elastic/figure", path="dia_qha.out", view="bulk")
    assert f["series"][0]["y"] == q["bulk_modulus_t_gpa"] and f["series"][1]["y"] == q["bulk_modulus_s_gpa"]
    _, f = _get(served, "elastic/figure", path="dia_qha.out", view="heat")
    # the repository's QHA output is trimmed to its QHA tables: Cp - Cv only (the full run adds each volume's Cv)
    assert [x["name"] for x in f["series"]] == ["C<sub>p</sub> − C<sub>v</sub>"]


def test_harmonic_heat_capacity_per_volume():
    """The per-volume blocks of a QHA run, in CRYSTAL's format (the full run's are 12,000 lines)."""
    block = """ PRIMITIVE CELL - CENTRING CODE 5/0 VOLUME=    10.801674 - DENSITY  3.690 g/cm^3
 HARMONIC VIBRATIONAL CONTRIBUTIONS TO THERMODYNAMIC FUNCTIONS AT GIVEN
 THERMODYNAMIC FUNCTIONS WITH VIBRATIONAL CONTRIBUTIONS

 AT (T =  100.00 K, P =   0.10132500E+00 MPA):
 HEAT CAPACITY :       0.000000000674       0.000000018351           0.00000177
 THERMODYNAMIC FUNCTIONS WITH VIBRATIONAL CONTRIBUTIONS

 AT (T =  200.00 K, P =   0.10132500E+00 MPA):
 HEAT CAPACITY :       0.000000000674       0.000000018351           0.50000000
"""
    other = block.replace("10.801674", "11.476637").replace("0.50000000", "0.40000000")
    got = V.harmonic_heat_capacity(block + other)
    assert [b["volume_ang3"] for b in got] == [10.801674, 11.476637]
    assert got[0]["temperature_k"] == [100.0, 200.0] and got[0]["cv_j_mol_k"] == [0.00000177, 0.5]
    assert got[1]["cv_j_mol_k"][1] == 0.4


def test_cphf_tensors_and_a_dynamic_run(served):
    _, body = _get(served, "elastic/document", path="dia_cphf.out")
    c = body["cphf"]
    assert c["epsilon"][0][0] == 5.387 and c["alpha"][2][2] == 26.2346 and c["epsilon"][0][1] == 0.0 and c["n_frequencies"] == 1
    assert [v["key"] for v in body["views"]] == ["indicatrix"] and {r["label"]: r["value"] for r in body["settings"]}["Frequency"] == "static (ω = 0)"
    # DYNAMIC: rows of several frequencies in the same table (CRYSTAL's FR.(eV) column)
    head = (MECH / "dia_cphf.out").read_text()
    i = head.index(" POLARIZABILITY (ALPHA), DIELECTRIC")
    j = head.index(" DIELECTRIC TENSOR", i)
    rows = []
    for k, ev in enumerate((0.0, 1.0, 2.0)):
        for comp, val in (("XX", 5.387), ("XY", 0.0), ("XZ", 0.0), ("YY", 5.387), ("YZ", 0.0), ("ZZ", 5.387)):
            e = val + 0.1 * k if val else 0.0
            rows.append(f"  {ev:5.3f}   {comp}      {26.2346 + k:8.4f}       0.0000       {e:6.4f}       {0.01 * k if val else 0.0:6.4f}       {max(e - 1, 0):6.4f}")
    dyn = head[:i] + " POLARIZABILITY (ALPHA), DIELECTRIC (EPSILON) AND FIRST-ORDER ELECTRIC\n SUSCEPTIBILITY (CHI(1)) TENSORS (INDEPENDENT COMPONENTS)\n\n" \
        "  FR.(eV) COMP.   ALPHA(Re,Im)           EPSILON(Re,Im)                CHI(1)\n" + "\n".join(rows) + "\n\n" + head[j:]
    doc = V.document(dyn)
    assert [v["key"] for v in doc["views"]] == ["epsilon", "refractive", "absorption", "indicatrix"] and doc["cphf"]["n_frequencies"] == 3
    f = V.figure(doc["data"], "epsilon", dyn)
    assert [s["name"] for s in f["series"]] == [f"{p} ε<sub>{c}</sub>" for c in ("xx", "yy", "zz") for p in ("Re", "Im")]
    assert f["series"][0]["x"] == [0.0, 1.0, 2.0] and f["series"][0]["y"] == [5.387, 5.487, 5.587]


def test_a_slab_s_in_plane_moduli(served):
    _, body = _get(served, "elastic/document", path="graphene_elastcon.out")
    assert body["kind"] == "ELASTIC" and body["data"]["dimensionality"] == 2
    assert [b["criterion"] for b in body["born"]] == ["C11 > 0", "C11 C22 − C12² > 0", "C66 > 0"]
    E = body["directional"]["youngs"]
    m = body["data"]["moduli"]
    # C66 = (C11 - C12)/2: an isotropic layer, E(theta) = E_x everywhere, in N/m
    assert E["unit"] == "N/m" and E["isotropic"] and E["max"]["value"] == pytest.approx(m["youngs_modulus_x"], rel=1e-4)
    assert body["directional"]["compressibility"]["unit"] == "m/N"
    code, f = _get(served, "elastic/figure", path="graphene_elastcon.out", view="youngs", shape="surface")
    assert code == 200 and f["shape"] == "planes" and [c["plane"] for c in f["planes"]] == ["xy"]       # a slab: its own plane
    assert f["cli"] == "mace analyze elastic graphene_elastcon.out --plot youngs"


def test_an_output_still_running_has_no_results(served):
    code, body = _get(served, "elastic/document", path="dia_cphf_running.out")
    assert code == 200 and body["empty"] and body["kind"] is None and body["views"] == []


def test_mace_elastic_analyze_plot_writes_the_figures(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "MPLBACKEND": "Agg"}

    def run(*args):
        return subprocess.run([sys.executable, str(MACE_CLI), "--no-banner", "analyze", "elastic", *args],
                              capture_output=True, text=True, env=env, timeout=300)
    r = run(str(MECH / "dia_elastcon.out"), "--plot", "youngs", "--shape", "planes", "--plot", "compressibility",
            "-o", str(tmp_path), "--format", "svg", "--format", "png")
    assert r.returncode == 0, r.stderr
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["dia_elastcon.compressibility_planes.png", "dia_elastcon.compressibility_planes.svg",
                     "dia_elastcon.youngs_planes.png", "dia_elastcon.youngs_planes.svg"]
    assert "Generated:" in r.stdout and "1150.388" in r.stdout          # the report, then the figures
    r = run(str(MECH / "dia_qha.out"), "--plot", "all", "-o", str(tmp_path / "qha"))
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in (tmp_path / "qha").iterdir()) == [f"dia_qha.{k}.png" for k in ("bulk", "expansion", "heat", "volume", "vratio")]
    assert (tmp_path / "qha" / "dia_qha.volume.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    # a plot the run does not have: an error naming the ones it has, nothing written
    r = run(str(MECH / "dia_eos.out"), "--plot", "youngs", "-o", str(tmp_path / "eos"))
    assert r.returncode == 1 and "it has eos, pv" in r.stderr and not (tmp_path / "eos").exists()
    r = run(str(MECH / "dia_eos.out"), "--plot", "nope")
    assert r.returncode == 2 and "invalid choice" in r.stderr
