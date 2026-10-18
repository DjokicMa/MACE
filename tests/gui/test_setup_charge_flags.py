"""Charged systems in Calculation Setup (1.1.6 parity A1): the CLI's own
charged-system switches of convert (cif2d12) and opt2d12 (--ion-charges,
--net-charge, --site-states, --chemod, --fragment-bond-scale, --spinlock,
--spinlock-cycles), passed as switches only, on the command Setup runs and
on the one it prints. A deck Setup makes with them is byte for byte the deck
the CLI typed by hand makes with the same switches and answers, on real
corpus parents (skipped without the test/ corpus); a meaning the CLI refuses
is refused in its own words."""
import re
import shlex
import shutil
import sys
from pathlib import Path

import pytest

import guiharness as H
import p4util as U
import setup_routes as R

pytest.importorskip("numpy")

sys.path.insert(0, str(H.REPO_ROOT))
from mace.gui import setup as S  # noqa: E402

CORPUS = H.REPO_ROOT / "test" / "OPT"
MOL = CORPUS / "1LiFSI-1DEC-conf3_MOLECULE_OPT_symm_HSESOL3C_SOLDEF2MSVP_opt_HSESOL3C_optimized"


# ---------------------------------------------------------------- the switches themselves
@pytest.mark.parametrize("module", ["NewCifToD12.py", "CRYSTALOptToD12.py"])
def test_every_switch_is_the_clis(module):
    """Each switch Setup passes is one of the command's own, with the same choices."""
    src = (H.REPO_ROOT / "Crystal_d12" / module).read_text()
    for key, switch, kind, choices, _, _ in S.CHARGE_FLAGS:
        m = re.search(r'charged\.add_argument\(\s*"' + re.escape(switch) + r'"([^)]*)\)', src)
        assert m, f"{module} has no {switch}"
        if choices:
            assert re.search(r"choices=\[" + ", ".join(f'"{c}"' for c in choices) + r"\]", m.group(1)), (switch, m.group(1))
        if kind == "int":
            assert "type=int" in m.group(1), switch


def test_request_takes_each_switch_in_its_shape():
    req = S.Request({"flags": {"ion_charges": "auto", "net_charge": "-1", "chemod": "yes", "fragment_bond_scale": 1.2,
                               "spinlock": 1, "spinlock_cycles": "30",
                               "site_states": ' [{"atom": 1, "shell_charges": [2, 0]}] '}})
    assert req.flags == {"ion_charges": "auto", "net_charge": -1, "chemod": "yes", "fragment_bond_scale": 1.2,
                         "spinlock": 1, "spinlock_cycles": 30, "site_states": '[{"atom": 1, "shell_charges": [2, 0]}]'}
    # the command line order is the CLI's group order, the values as typed
    assert S.flag_args(req.flags, "opt2d12") == [
        "--ion-charges", "auto", "--net-charge", "-1", "--site-states", '[{"atom": 1, "shell_charges": [2, 0]}]',
        "--chemod", "yes", "--fragment-bond-scale", "1.2", "--spinlock", "1", "--spinlock-cycles", "30"]
    assert S.flag_args(req.flags, "opt2d3") == []          # properties take no charge switches
    # blank fields are not sent; a net charge of 0 is
    assert S.Request({"flags": {"net_charge": 0, "chemod": "", "spinlock": None}}).flags == {"net_charge": 0}
    assert S.Request({"flags": {"net_charge": 0.5}}).flags == {"net_charge": 0.5}


@pytest.mark.parametrize("flags,match", [
    ({"charge": 1}, "flags are the CLI's charged-system switches: --ion-charges"),
    ({"ion_charges": "yes"}, "--ion-charges is one of auto"),
    ({"chemod": "maybe"}, "--chemod is one of auto, yes, no"),
    ({"net_charge": "one"}, "--net-charge is a number"),
    ({"net_charge": True}, "--net-charge is a number"),
    ({"spinlock": 1.5}, "--spinlock is a whole number"),
    ({"site_states": "/etc/passwd"}, "Setup does not read it from a file"),
    ({"site_states": "states.json"}, "Setup does not read it from a file"),
])
def test_bad_shapes_are_named(flags, match):
    with pytest.raises(S.SetupError, match=re.escape(match)):
        S.Request({"flags": flags})


def test_batch_takes_every_switch_but_site_states():
    assert "site_states" not in S.BATCH_CHARGE_FLAGS
    with pytest.raises(S.SetupError, match="--site-states"):
        S.charge_flags({"site_states": "[]"}, S.BATCH_CHARGE_FLAGS)
    assert S.charge_flags({"ion_charges": "auto"}, S.BATCH_CHARGE_FLAGS) == {"ion_charges": "auto"}


def test_schema_lists_the_switches():
    sch = S.schema()
    assert [f["switch"] for f in sch["charge_flags"]] == [f[1] for f in S.CHARGE_FLAGS]
    assert sch["charge_modes"] == ["cif2d12", "opt2d12"]


# ---------------------------------------------------------------- real corpus parents
def _cache():
    store = {}
    import json
    return (lambda k: store.get(json.dumps(k, default=str)),
            lambda k, v, n: store.__setitem__(json.dumps(k, default=str), v))


def _need_corpus():
    if not MOL.with_suffix(".out").is_file():
        pytest.skip("test/ corpus not present (gitignored)")


def test_cli_refusal_is_shown_in_its_own_words(tmp_path):
    """Site states without a net charge: opt2d12's own refusal, no deck."""
    _need_corpus()
    src = tmp_path / "src"
    src.mkdir()
    for ext in (".out", ".d12"):
        shutil.copyfile(MOL.with_suffix(ext), src / f"mol{ext}")
    get, put = _cache()
    req = S.Request({"answers": {}, "flags": {"site_states": '[{"atom": 1, "shell_charges": [2, 0, 0, 0, 0]}]'}})
    r = S.run_setup(req, "opt2d12", "SP", tmp_path / "w", src / "mol.out", "mol.out", S.source_key("opt2d12", src / "mol.out"),
                    get, put, None, schema_info=S.schema())
    assert not r.get("deck")
    told = (r.get("refused") or "") + " ".join(p["message"] for p in r.get("problems", []))
    assert "give the net charge too" in told, told


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    _need_corpus()
    tmp = tmp_path_factory.mktemp("charge_flags")
    root = tmp / "root"
    (root / "mol").mkdir(parents=True)
    for ext in (".out", ".d12"):
        shutil.copyfile(MOL.with_suffix(ext), root / "mol" / f"mol_opt{ext}")
    (root / "cif").mkdir()
    shutil.copyfile(MOL.with_suffix(".cif"), root / "cif" / "mol.cif")
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1")
    yield root, base
    proc.terminate()
    proc.wait(timeout=15)


def _decks(folder: Path):
    return {p.name for p in folder.iterdir() if p.suffix.lower() == ".d12"} if folder.is_dir() else set()


CASES = [
    ("mol/mol_opt.out", "opt2d12", "SP", R.YES, {"ion_charges": "auto", "chemod": "yes"}),
    ("mol/mol_opt.out", "opt2d12", "SP", R.YES, {"ion_charges": "auto", "fragment_bond_scale": 1.25, "chemod": "auto"}),
    ("mol/mol_opt.out", "opt2d12", "SP", R.YES, {"net_charge": 1, "spinlock": 1, "spinlock_cycles": 40}),
    ("mol/mol_opt.out", "opt2d12", "SP", R.YES, {"net_charge": 0, "site_states": '[{"atom": 1, "shell_charges": [2, 2, 0, 0, 4, 0, 0]}]'}),
    ("cif/mol.cif", "cif2d12", "OPT", R.ENTER, {"ion_charges": "auto", "chemod": "yes"}),
]


@pytest.mark.parametrize("src,mode,calc,manual,flags", CASES, ids=[f"{c[1]}-{'-'.join(c[4])}" for c in CASES])
def test_setup_deck_equals_the_cli_typed_by_hand(served, tmp_path, src, mode, calc, manual, flags):
    if mode == "cif2d12":
        pytest.importorskip("ase", reason="mace convert reads CIF files with ase")
    root, base = served
    body = {"source": {"path": src}, "calc_type": calc, "answers": {}, "patches": {}, "flags": flags}
    st, _, r = U.post(base, "api/setup/run", {"preview": True, "wait": 300, **body}, timeout=400)
    assert st == 200, r
    assert r["deck"], r.get("refused") or r.get("log")
    want = S.flag_args(S.charge_flags(flags), mode)
    assert shlex.split(r["command"])[-len(want):] == want, r["command"]
    assert r.get("control_same") is not False, "the interactive run and the printed command differ"
    # Save: the command it prints carries the same switches, and run as printed it writes the same deck
    st, _, s = U.post(base, "api/setup/save", {**body, "json_text": r["json_text"], "overwrite": True})
    assert st == 200, s
    argv = shlex.split(s["command"])
    assert argv[-len(want):] == want, s["command"]
    g = tmp_path / "g"
    shutil.copytree(root, g)
    out = g / (argv[argv.index("--output_dir") + 1] if "--output_dir" in argv else str(Path(src).parent))
    before = _decks(out)
    R.run_printed(argv, g)
    new = sorted(_decks(out) - before)
    assert new, s["command"]
    gui_deck = (out / new[0]).read_text()
    assert gui_deck == r["deck"]
    # the manual CLI: the same switches typed, the same answers
    m = tmp_path / "m"
    if mode == "cif2d12":
        (m / "cif").mkdir(parents=True)
        shutil.copy2(root / src, m / "cif" / Path(src).name)
        args = ["convert", "--cif_dir", "cif", "--output_dir", "m_out"]
    else:
        shutil.copytree((root / src).parent, m / "in")
        args = ["opt2d12", "--out-file", f"in/{Path(src).name}", "--calc-type", calc, "--output-dir", "m_out"]
    res = R.run_cli_manual(args + want, manual, m)
    made = sorted(_decks(m / "m_out"))
    assert res.returncode == 0 and made, (res.stdout + res.stderr)[-3000:]
    assert new[0] == made[0]
    assert gui_deck == (m / "m_out" / made[0]).read_text()
    if flags.get("ion_charges") or flags.get("site_states"):
        assert "CHEMOD" in gui_deck.split() or flags.get("chemod") == "no"
    if flags.get("spinlock"):
        assert "SPINLOCK" in gui_deck.split()
