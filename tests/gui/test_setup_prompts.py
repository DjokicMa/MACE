"""Setup answers only the questions it knows (answer_driver.py).

Calculation Setup runs the real CLI (cif2d12 / opt2d12 / opt2d3) and
answers its questions by rule. A question no rule names used to get Enter,
the CLI's default, silently: a reworded prompt then turned a GUI answer into
a default without notice. Now such a question gets Enter only when it is on
Setup's list of questions where Enter is the intended answer
(setup.known_questions); any other question stops the run and names it.

The flows test runs every Setup route through the driver (each mode and
calculation type: the defaults, the survey of every question, and answers
that open further questions) and asserts each question was matched by a
rule or by the list: a changed prompt in the CLI fails here first.
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import cli_prompts as CP
import guiharness as H
import prompt_coverage as PC

pytest.importorskip("numpy")

sys.path.insert(0, str(H.REPO_ROOT))
from mace.gui import clirun  # noqa: E402
from mace.gui import setup as S  # noqa: E402

DRIVER = H.REPO_ROOT / "mace" / "gui" / "answer_driver.py"

FAKE_CLI = """
import sys
from pathlib import Path
a = input("Use the usual thing? [Y/n] ")
b = input("A question nobody has seen before? [y/N] ")
Path(sys.argv[1]).write_text(repr((a, b)))
"""


def _drive(tmp_path, spec):
    cli = tmp_path / "fake_cli.py"
    cli.write_text(FAKE_CLI)
    rules, tr, done = tmp_path / "rules.json", tmp_path / "tr.json", tmp_path / "done.txt"
    rules.write_text(json.dumps(spec))
    p = subprocess.run([sys.executable, str(DRIVER), str(rules), str(tr), str(cli), str(done)],
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)
    return p, json.loads(tr.read_text()), done


def test_an_unknown_question_stops_the_run_and_is_named(tmp_path):
    p, tr, done = _drive(tmp_path, {"rules": [], "known": [r"Use the usual thing\?"]})
    assert p.returncode == 3, (p.returncode, p.stdout, p.stderr)
    assert not done.exists(), "the unknown question was answered"
    assert "A question nobody has seen before?" in (tr["error"] or "")
    first, second = tr["questions"]
    assert first["answer"] == "" and first["known"] == 0 and first["rule"] is None
    assert second["answer"] is None


def test_known_and_ruled_questions_run_through(tmp_path):
    p, tr, done = _drive(tmp_path, {"rules": [{"match": r"nobody has seen", "answer": "y"}],
                                    "known": [r"Use the usual thing\?"]})
    assert p.returncode == 0, (p.stdout, p.stderr)
    assert done.read_text() == repr(("", "y"))
    assert tr["error"] is None


def test_no_list_means_every_question_needs_a_rule(tmp_path):
    p, tr, done = _drive(tmp_path, {"rules": []})
    assert p.returncode == 3 and not done.exists()
    assert "Use the usual thing?" in tr["error"]


def test_setup_reports_an_unknown_question(tmp_path, monkeypatch):
    """run_setup turns the driver's stop into a SetupError naming the question."""
    src = tmp_path / "src"
    shutil.copytree(H.EXAMPLES / "outputs", src)
    real = S.known_questions
    monkeypatch.setattr(S, "known_questions", lambda mode: [k for k in real(mode) if "MAXCYCLE" not in k])
    req = S.Request({"answers": {}, "patches": {}, "preview": False})
    with pytest.raises(S.SetupError, match=r"mace opt2d12 did not ask what Setup expected: "
                                           r"a question Setup does not know \(not answered\): .*MAXCYCLE \[60\]"):
        S.run_setup(req, "opt2d12", "SP", tmp_path / "w", src / "Si_sp.out", "Si_sp.out",
                    S.source_key("opt2d12", src / "Si_sp.out"), lambda k: None, lambda k, v, n: None, None)


# ------------------------------------------------------------ every question in the CLI sources (no CLI run)
# cli_prompts reads every prompt call in the sources of the CLIs Setup runs; each must match a
# Setup regex or be listed, with its reason, in prompt_coverage.UNANSWERED. A new or reworded
# question fails here even on a route the sweeps below do not run.
MIN_PROMPTS = {"cif2d12": 200, "opt2d12": 200, "opt2d3": 70}


def _setup_regexes(mode):
    pats = list(S.known_questions(mode))
    for f in S.FIELDS:
        if mode in f.modes:
            pats += [r["match"] for r in f.rules(mode, "PBE0" if f.id == "functional" else "x")]
    return [re.compile(p, re.I | re.S) for p in pats]


def _unanswered_prompts():
    """(mode, prompt) for each prompt no Setup regex matches."""
    out = []
    for mode in CP.MODE_FILES:
        regs = _setup_regexes(mode)
        for p in CP.all_prompts(mode):
            if p.prompt is not None and any(r.search(p.text) or r.search(p.prompt) for r in regs):
                continue
            out.append((mode, p))
    return out


def _entry_for(p):
    name = Path(p.file).name
    return next((k for k in PC.UNANSWERED if k[0] == name and k[1] in p.text), None)


def test_the_extractor_finds_the_questions():
    for mode, files in CP.MODE_FILES.items():
        ps = CP.all_prompts(mode)
        assert len(ps) >= MIN_PROMPTS[mode], (mode, len(ps))
        for rel in files:
            if not rel.endswith("d12_constants.py") or mode != "opt2d3":
                assert any(p.file == rel for p in ps), f"no question found in {rel}"


def test_every_cli_question_is_answered_or_listed():
    missing = [f"{mode}: {p.file}:{p.line} {p.text.strip()[-160:]!r}"
               for mode, p in _unanswered_prompts() if _entry_for(p) is None]
    assert not missing, ("questions no Setup regex matches and prompt_coverage.UNANSWERED does not list "
                         "(add a pattern to setup.KNOWN_QUESTIONS if Setup can reach it and Enter keeps "
                         "a sensible default, else list it with its reason):\n" + "\n".join(missing))


def test_every_listed_question_still_exists_unanswered():
    used = {_entry_for(p) for _, p in _unanswered_prompts()}
    stale = [k for k in PC.UNANSWERED if k not in used]
    assert not stale, f"listed but no longer an unanswered question in the sources: {stale}"
    assert {v[0] for v in PC.UNANSWERED.values()} <= {PC.NOT_REACHABLE, PC.NO_DEFAULT, PC.AFTER_A_STOP}


# ------------------------------------------------------------ every Setup route, every question matched
def _flows():
    out = []
    for mode in S.MODES:
        for calc in S.CALC_TYPES[mode]:
            out.append((mode, calc, "defaults", None))
            out.append((mode, calc, "survey", None))
    # the calculation type is cif2d12's own question (run_setup answers it)
    for calc in ("SP", "FREQ"):
        out.append(("cif2d12", calc, "answers", {"calc_type": calc}))
    # answers that open further questions
    more = [
        ("cif2d12", "OPT", {"functional": "PBE0"}), ("cif2d12", "OPT", {"basis_set": "POB-DZVP-REV2"}),
        ("cif2d12", "OPT", {"basis_type": "EXTERNAL"}), ("cif2d12", "OPT", {"spin": False}),
        ("cif2d12", "OPT", {"smearing": True, "smear_width": "0.005", "fmixing": "30"}),
        ("cif2d12", "OPT", {"method": "HF"}), ("cif2d12", "OPT", {"dimensionality": "SLAB"}),
        ("cif2d12", "OPT", {"dimensionality": "MOLECULE"}), ("cif2d12", "FREQ", {"calc_type": "FREQ", "ir": True}),
        ("opt2d12", "OPT", {"functional": "HSE06"}), ("opt2d12", "SP", {"fmixing": "40"}),
        ("opt2d12", "SP", {"spin": False}), ("opt2d12", "FREQ", {"numderiv": "Two displacements per atom"}),
        ("opt2d3", "BAND", {"band_format": "K-point vectors"}), ("opt2d3", "BAND", {"band_path": "Template selection"}),
        ("opt2d3", "BAND", {"band_range": "Around Fermi level"}),
        ("opt2d3", "BAND", {"band_range": "Custom range"}),
        ("opt2d3", "DOSS", {"doss_projection": "Total DOS only"}), ("opt2d3", "DOSS", {"doss_range": "Energy window around Fermi level"}),
        ("opt2d3", "DOSS", {"doss_projection": "Projected DOS on specific atoms"}),
        ("cif2d12", "OPT", {"opt_convergence": "Custom"}), ("cif2d12", "OPT", {"tolerance": "Custom"}),
        ("opt2d12", "SP", {"basis_keep": False}),
        # one functional of each category (VBH: the LDA question, "Select LDA/LSD functional")
        ("cif2d12", "OPT", {"functional": "VBH"}), ("cif2d12", "OPT", {"functional": "PBE"}),
        ("cif2d12", "OPT", {"functional": "r2SCAN"}), ("cif2d12", "OPT", {"functional": "HSE3C"}),
        ("opt2d12", "SP", {"functional": "VBH"}),
    ]
    # every FREQ template opens its own questions
    for t in ("IR spectrum", "Raman spectrum", "IR + Raman spectra", "Thermodynamic properties",
              "Phonon band structure", "Phonon density of states", "Custom settings"):
        more.append(("cif2d12", "FREQ", {"calc_type": "FREQ", "freq_template": t}))
    more.append(("opt2d12", "FREQ", {"freq_template": "Custom settings"}))
    # FREQ answers in "Other questions" that open each template's own customizations
    fq = {"calc_type": "FREQ"}
    more += [
        ("cif2d12", "FREQ", {**fq, "freq_template": "IR spectrum", "q:Customize spectral range?": "y"}),
        ("cif2d12", "FREQ", {**fq, "freq_template": "Thermodynamic properties", "q:Customize temperature range?": "y"}),
        # the band path's crystal system and template ("Select template (1-3)" is not the FREQ template menu)
        ("cif2d12", "FREQ", {**fq, "freq_template": "Phonon band structure", "q:Customize supercell size?": "y",
                             "q:Select method (1-4) [1]": "2"}),
        ("cif2d12", "FREQ", {**fq, "freq_template": "Phonon band structure", "q:Select format (1-4) [1]": "2",
                             "q:Customize k-point density?": "y"}),
        ("cif2d12", "FREQ", {**fq, "freq_template": "Phonon density of states",
                             "q:2: INS - Inelastic neutron scattering spectrum": "2", "q:Customize supercell size?": "y"}),
        # custom settings: IR by CPHF (its iterations and tolerance) or Wannier (RELOCAL), Raman,
        # thermodynamics, restart
        ("opt2d12", "FREQ", {"freq_template": "Custom settings", "ir": True, "q:Calculate Raman intensities?": "y",
                             "q:Calculate thermodynamic properties?": "y", "q:Configure restart capability?": "y"}),
        ("opt2d12", "FREQ", {"freq_template": "Custom settings", "ir": True, "q:Select method (1-3) [3]": "2"}),
        # custom settings: IR and Raman spectra (IRSPEC/RAMSPEC) and their settings, RAMANEXP
        ("opt2d12", "FREQ", {"freq_template": "Custom settings", "ir": True, "q:Calculate Raman intensities?": "y",
                             "q:Customize spectral range?": "y", "q:Select broadening (1-2)": "2",
                             "q:Use RAMANEXP for experimental conditions?": "y"}),
        ("cif2d12", "FREQ", {**fq, "freq_template": "Custom settings", "ir": True}),
        # custom settings: phonon dispersion (supercell, INTERPHESS, bands from a template, DOS, INS)
        ("opt2d12", "FREQ", {"freq_template": "Custom settings", "q:Select mode (1-2)": "2",
                             "q:Select supercell method (1-3)": "2", "q:Select path method (1-4)": "2",
                             "q:Calculate INS (Inelastic Neutron Scattering) spectrum?": "y"}),
        ("opt2d12", "FREQ", {"freq_template": "Custom settings", "q:Select mode (1-2)": "2",
                             "q:High-symmetry points only (quick check)": "2"}),
    ]
    out += [(m, c, "answers", a) for m, c, a in more]
    return out


FLOWS = _flows()
SRC = {"cif2d12": "structures/MgO.cif", "opt2d12": "outputs/Si_sp.out", "opt2d3": "outputs/Si_sp.out"}


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    """One staged input per mode and calculation type (the CLI only reads it)."""
    tmp = tmp_path_factory.mktemp("prompts")
    cache = {}

    def get(mode, calc):
        if (mode, calc) not in cache:
            d = tmp / f"src_{mode}_{calc}".replace("+", "_")
            shutil.copytree(H.EXAMPLES / Path(SRC[mode]).parent, d)
            if mode == "opt2d3":
                (d / "Si_sp.f9").write_text("wavefunction placeholder\n")
            work = tmp / f"w_{mode}_{calc}".replace("+", "_")
            st = S.stage(mode, calc, work, d / Path(SRC[mode]).name, Path(SRC[mode]).name, None, None)
            cache[(mode, calc)] = (st, work)
        return cache[(mode, calc)]
    return get


@pytest.mark.parametrize("mode, calc, kind, answers", FLOWS,
                         ids=[f"{m}-{c}-{k}" + (("-" + "-".join(f"{a}={v}" for a, v in x.items())) if x else "")
                              for m, c, k, x in FLOWS])
def test_every_question_of_every_setup_route_is_matched(staged, mode, calc, kind, answers):
    if mode == "cif2d12":
        pytest.importorskip("ase")
    st, work = staged(mode, calc)
    rules = {"defaults": lambda: list(S.DEFAULT_RULES[mode]), "survey": lambda: S.survey_rules(mode),
             "answers": lambda: S.answer_rules(mode, answers)}[kind]()
    tag = kind + "_" + "_".join(sorted(answers or {})) + str(abs(hash(json.dumps(answers, sort_keys=True))) % 10**6)
    r = S.ask(st, work, rules, tag)
    assert r["error"] is None, r["error"]
    assert r["questions"], r["log"][-2000:]
    unmatched = [S.Question(q).last for q in r["questions"] if q.get("rule") is None and q.get("known") is None]
    assert not unmatched, f"answered with a bare Enter: {unmatched}"
    if kind == "defaults":
        assert r["deck"] is not None, r["log"][-2000:]
    if "functional" in (answers or {}):
        fq = [q for q in r["questions"] if S.FIELD["functional"].re.search(q["question"])]
        assert fq and all(q.get("rule") is not None for q in fq), "the functional's question was not answered"


# ------------------------------------------------------------ other inputs: other questions
# The CLI asks more on some inputs: a CIF spglib cannot reduce (P1, a slab or molecule in a
# box) asks "Use the 'reduced' structure anyway?"; an output with a Fermi energy asks
# TRANSPORT's chemical potential relative to it. Each runs through every Setup route.
_CIF_HEAD = ("data_{name}\n_cell_length_a {a}\n_cell_length_b {b}\n_cell_length_c {c}\n"
             "_cell_angle_alpha {al}\n_cell_angle_beta {be}\n_cell_angle_gamma {ga}\n"
             "_symmetry_space_group_name_H-M 'P 1'\n_symmetry_Int_Tables_number 1\n\n"
             "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n_atom_site_fract_y\n"
             "_atom_site_fract_z\n_atom_site_occupancy\n")
CIFS = {
    # no symmetry at all: spglib's asymmetric unit is every atom
    "p1.cif": (dict(a=4.91, b=5.37, c=6.02, al=83.0, be=97.0, ga=101.0),
               [("Si", .02, .11, .07), ("O", .31, .19, .42), ("O", .58, .73, .26), ("Mg", .77, .44, .81)]),
    # a buckled BN sheet in 20 Å of vacuum
    "slab.cif": (dict(a=2.51, b=2.51, c=20.0, al=90.0, be=90.0, ga=120.0),
                 [("B", 1 / 3, 2 / 3, .50), ("N", 2 / 3, 1 / 3, .51)]),
    # a water molecule in a box
    "molecule.cif": (dict(a=12.0, b=12.0, c=12.0, al=90.0, be=90.0, ga=90.0),
                     [("O", .50, .50, .50), ("H", .5797, .50, .5497), ("H", .4203, .50, .5497)]),
}


def _cif_text(name):
    cell, atoms = CIFS[name]
    rows = "".join(f"{el}{i + 1} {el} {x:.6f} {y:.6f} {z:.6f} 1.0\n" for i, (el, x, y, z) in enumerate(atoms))
    return _CIF_HEAD.format(name=name[:-4], **cell) + rows


def _other_flows():
    out = []
    for cif in CIFS:
        for calc in S.CALC_TYPES["cif2d12"]:
            out += [("cif2d12", calc, k, cif) for k in ("defaults", "survey")]
    # graphene_slab_sp.out: a slab with a Fermi energy (a conducting state)
    for mode in ("opt2d12", "opt2d3"):
        for calc in S.CALC_TYPES[mode]:
            out += [(mode, calc, k, "outputs/graphene_slab_sp.out") for k in ("defaults", "survey")]
    # corpus outputs (skipped without the test/ corpus): spin-polarized, a molecule
    for pat in ("SP/1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.out",
                "SP/EC_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized_sp_HSESOL3C_optimized.out"):
        for mode in ("opt2d12", "opt2d3"):
            for calc in S.CALC_TYPES[mode]:
                out += [(mode, calc, k, "corpus:" + pat) for k in ("defaults", "survey")]
    return out


OTHER = _other_flows()


def _stage_other(tmp, mode, calc, src):
    d = tmp / "src"
    d.mkdir()
    if src in CIFS:
        p = d / src
        p.write_text(_cif_text(src))
    else:
        if src.startswith("corpus:"):
            from conftest import find_data
            real = find_data(src[len("corpus:"):])
        else:
            real = H.EXAMPLES / src
        p = d / real.name
        shutil.copy2(real, p)
        d12 = real.with_suffix(".d12")
        if d12.exists():
            shutil.copy2(d12, d / d12.name)
        if mode == "opt2d3":
            (d / (p.stem + ".f9")).write_text("wavefunction placeholder\n")
    work = tmp / "work"
    return S.stage(mode, calc, work, p, p.name, None, None), work


@pytest.mark.parametrize("mode, calc, kind, src", OTHER, ids=[f"{m}-{c}-{k}-{Path(s).name[:24]}" for m, c, k, s in OTHER])
def test_every_question_on_other_inputs_is_matched(tmp_path, mode, calc, kind, src):
    if mode == "cif2d12":
        pytest.importorskip("ase")
        pytest.importorskip("spglib")
    st, work = _stage_other(tmp_path, mode, calc, src)
    rules = list(S.DEFAULT_RULES[mode]) if kind == "defaults" else S.survey_rules(mode)
    r = S.ask(st, work, rules, kind)
    assert r["error"] is None, r["error"]
    if "MOLECULE" in src and calc in ("BAND", "DOSS", "TRANSPORT"):
        # opt2d3 refuses these for a molecule before asking anything; Setup shows its words
        assert r["deck"] is None and "requires a periodic structure" in S.refusal(r["log"]), r["log"][-2000:]
        return
    assert r["questions"], r["log"][-2000:]
    unmatched = [S.Question(q).last for q in r["questions"] if q.get("rule") is None and q.get("known") is None]
    assert not unmatched, f"answered with a bare Enter: {unmatched}"
    if kind == "defaults":
        assert r["deck"] is not None, r["log"][-2000:]


def test_the_p1_cif_asks_the_reduced_structure_question(tmp_path):
    """The P1 case above reaches the question it is there for (not a quieter route)."""
    pytest.importorskip("ase")
    pytest.importorskip("spglib")
    st, work = _stage_other(tmp_path, "cif2d12", "SP", "p1.cif")
    r = S.ask(st, work, list(S.DEFAULT_RULES["cif2d12"]), "p1")
    assert any("'reduced' structure anyway" in q["question"] for q in r["questions"]), r["log"][-2000:]


def test_transport_on_an_output_with_a_fermi_energy_asks_relative_to_it(tmp_path):
    st, work = _stage_other(tmp_path, "opt2d3", "TRANSPORT", "outputs/graphene_slab_sp.out")
    r = S.ask(st, work, [], "fermi")
    assert any("μ (eV below Fermi)" in q["question"] for q in r["questions"]), r["log"][-2000:]


def test_a_question_with_nothing_to_keep_stops_the_run(staged):
    """opt2d3's custom band labels ask "Path:" with no default: Enter would answer nothing, so
    Setup stops there and names the question (it used to send Enter)."""
    st, work = staged("opt2d3", "BAND")
    r = S.ask(st, work, S.answer_rules("opt2d3", {"band_path": "Custom labels"}), "custom_labels")
    assert r["error"] and "Path:" in r["error"], r["error"]
    assert r["questions"][-1]["answer"] is None


def test_potc_points_keep_their_count_and_stop_at_the_coordinates(staged):
    """POTENTIAL as POTC at custom points: "Number of points:" keeps its default (1); the point
    itself has none, so Setup stops at "Point 1 (x y z)" and names it."""
    st, work = staged("opt2d3", "POTENTIAL")
    r = S.ask(st, work, S.answer_rules("opt2d3", {"q:2: POTC - Potential at specific points": "2",
                                                    "q:Select option (1-3) [1]": "2"}), "potc_points")
    asked = [S.Question(q).last for q in r["questions"]]
    assert any(a.startswith("Number of points:") for a in asked), asked
    assert r["error"] and "Point 1 (x y z)" in r["error"], r["error"]


def test_a_parent_functional_mace_cannot_name_is_asked_again(tmp_path):
    """opt2d12 on a parent whose functional MACE cannot identify (B2PLYP): with any setting
    answered, the CLI asks for another functional; Enter takes its default, HSE06."""
    d = tmp_path / "src"
    shutil.copytree(H.EXAMPLES / "outputs", d)
    deck = d / "Si_sp.d12"
    deck.write_text("\n".join("B2PLYP" if ln.strip() == "PBE0" else ln for ln in deck.read_text().splitlines()) + "\n")
    st = S.stage("opt2d12", "SP", tmp_path / "w", d / "Si_sp.out", "Si_sp.out", None, None)
    r = S.ask(st, tmp_path / "w", S.answer_rules("opt2d12", {"spin": False}), "b2plyp")
    assert r["error"] is None, r["error"]
    asked = [q for q in r["questions"] if "Enter another functional" in q["question"]]
    assert asked and asked[0]["known"] is not None, [S.Question(q).last for q in r["questions"]]
    assert "HSE06" in r["deck"]


def test_a_cif_the_basis_set_lacks_an_element_for_is_refused_by_the_cli(tmp_path):
    """POB-TZVP-REV2 has no La: convert asks "Continue anyway? [y/N]"; Enter is No and
    no deck is written (Setup shows the CLI's refusal)."""
    pytest.importorskip("ase")
    pytest.importorskip("spglib")
    d = tmp_path / "src"
    d.mkdir()
    cif = d / "lao.cif"
    cif.write_text(_CIF_HEAD.format(name="lao", a=5.1, b=5.1, c=5.1, al=90.0, be=90.0, ga=90.0)
                   + "La1 La 0.000000 0.000000 0.000000 1.0\nO1 O 0.500000 0.500000 0.500000 1.0\n")
    st = S.stage("cif2d12", "SP", tmp_path / "w", cif, cif.name, None, None)
    r = S.ask(st, tmp_path / "w", list(S.DEFAULT_RULES["cif2d12"]), "lao")
    assert r["error"] is None, r["error"]
    asked = [q for q in r["questions"] if "Missing elements" in q["question"]]
    assert asked and asked[0]["known"] is not None and asked[0]["answer"] == "", r["log"][-2000:]
    assert r["deck"] is None
