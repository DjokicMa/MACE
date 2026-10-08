"""File kinds by content (plan E1) and molecule outputs (plan E2), on the
real corpus.

Every output and log of test/ is copied, one run at a time with its sibling
.d3/.d12, into tmp under a neutral name, classified, and deleted. The
expected answers come from an oracle written here that does not use the
calc_detection / completion_checker regexes: a FREQ output has a parsed mode
table, an OPT output optimization points (or the first-cycle banner), a
properties output "CRYSTAL - PROPERTIES" in its first 200 lines. Zero
misclassifications. Skipped without the gitignored corpus."""
import re
import shutil
import subprocess
import sys
import importlib.util
from pathlib import Path

import pytest

import guiharness as H

CORPUS = H.REPO_ROOT / "test"
PROPS_DIRS = {"BAND": "BAND", "DOSS": "DOSS", "ECH3POT3": "CHARGE+POTENTIAL", "TRANSPORT": "TRANSPORT"}
needs_corpus = pytest.mark.skipif(not (CORPUS / "FREQ").is_dir(), reason="test/ corpus not present (gitignored)")

_MODE_ROW = re.compile(r"^\s*\d+-\s*\d+\s+-?\d\.\d+E[-+]\d+\s+-?\d+\.\d+")


FROZEN = Path(__file__).resolve().parent / "frozen"

def oracle(text: str, folder: str) -> tuple:
    """(role, calc) from the text alone, by rules independent of the CLI's detectors."""
    head200 = "\n".join(text.splitlines()[:200])
    if "CRYSTAL - PROPERTIES" in head200:
        return "properties_out", PROPS_DIRS.get(folder)
    if "EEEEEEEEEE STARTING" not in text:
        return "joblog", None
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if "MODES" in ln and "EIGV" in ln and "FREQUENCIES" in ln and "IRREP" in ln:
            rows = [x for x in lines[i + 2:i + 12] if _MODE_ROW.match(x)]
            if rows:
                return "crystal_out", "FREQ"
    # an optimization prints its points, or its control block before the first one (a run that died early)
    if re.search(r"OPTIMIZATION - POINT\s+\d", text) or "SATISFIED AFTER THE FIRST OPTIMIZATION CYCLE" in text \
            or "OPTIMIZATION CONTROL" in text:
        return "crystal_out", "OPT"
    return "crystal_out", "SP"


def _runs(folder: str):
    d = CORPUS / folder
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix == ".out")


def _copy_run(src: Path, dst: Path, with_d3: bool = True) -> Path:
    """The output and its sibling .d3/.d12 (same stem), under a neutral stem."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir()
    out = dst / "run.out"
    shutil.copyfile(src, out)
    for ext in ((".d3", ".d12") if with_d3 else (".d12",)):
        sib = src.with_suffix(ext)
        if sib.is_file():
            shutil.copyfile(sib, dst / f"run{ext}")
    return out


@needs_corpus
@pytest.mark.parametrize("folder", ["FREQ", "OPT", "SP", "BAND", "DOSS", "ECH3POT3", "TRANSPORT", "FAILED_QA"])
def test_every_corpus_output_classified_by_content(folder, tmp_path):
    from mace.gui import filekind
    wrong, seen = [], 0
    for src in _runs(folder):
        text = src.read_text(errors="replace")
        want = oracle(text, folder)
        for with_d3 in (True, False):              # the second pass: content alone gives the same answer
            out = _copy_run(src, tmp_path / "run", with_d3)
            got = filekind.refine(out)
            if (got["role"], got["calc"] if got["role"] != "joblog" else None) != want:
                wrong.append((src.name[-60:], with_d3, want, (got["role"], got["calc"])))
            shutil.rmtree(tmp_path / "run")
        seen += 1
    assert seen, f"no outputs in {folder}"
    assert not wrong, wrong[:10]


@needs_corpus
def test_corpus_counts_match_the_plan(tmp_path):
    """161 FREQ outputs are FREQ, 118 OPT outputs OPT (the plan's census)."""
    from mace.gui import filekind
    for folder, calc, n in (("FREQ", "FREQ", 161), ("OPT", "OPT", 118)):
        got = [filekind.refine(p)["calc"] for p in _runs(folder)]
        assert len(got) == n and set(got) == {calc}, (folder, len(got), set(got))


@needs_corpus
def test_every_job_log_is_a_job_log(tmp_path):
    """All SLURM logs (.o), the ipBANDS_<id>.out stub and its .err: job logs, never geometry."""
    from mace.gui import filekind
    logs = sorted(p for p in CORPUS.rglob("*") if p.is_file() and (p.suffix in (".o", ".err") or
                                                                  (p.name.startswith("ipBANDS_") and p.suffix == ".out")))
    assert len([p for p in logs if p.suffix == ".o"]) == 551
    bad = []
    for k, src in enumerate(logs):
        dst = tmp_path / f"job-{100000 + k}{src.suffix}"
        shutil.copyfile(src, dst)
        if filekind.refine(dst)["role"] != "joblog":
            bad.append(src.name[-50:])
        dst.unlink()
    assert not bad, bad[:10]


@needs_corpus
def test_content_wins_over_the_name(tmp_path):
    """A CRYSTAL output saved as job-123.o is a CRYSTAL output; an SCF output beside
    a .d3 of the same stem keeps its own (content) type, not the .d3's."""
    from mace.gui import filekind
    src = _runs("OPT")[0]
    o = tmp_path / "job-123.o"
    shutil.copyfile(src, o)
    assert filekind.refine(o)["role"] == "crystal_out"
    sp = next(p for p in _runs("SP") if oracle(p.read_text(errors="replace"), "SP")[1] == "SP")
    shutil.copyfile(sp, tmp_path / "pair.out")
    d3 = next((CORPUS / "DOSS").glob("*.d3"))
    shutil.copyfile(d3, tmp_path / "pair.d3")
    r = filekind.refine(tmp_path / "pair.out")
    assert (r["role"], r["calc"]) == ("crystal_out", "SP")


def _old_classify():
    """server.classify as it was at the plan's base, frozen in tests/gui/frozen."""
    body = (FROZEN / "gui_classify.base.txt").read_text()
    ns = {}
    exec(compile(body, "old_classify", "exec"), ns)
    return ns["classify"]


def test_kind_by_name_keeps_every_old_kind():
    """Every name the explorer listed before keeps its kind (.out and .o stay "out")."""
    from mace.gui import filekind
    old = _old_classify()
    names = ["x.cif", "X.CIF", "a.d12", "a.d3", "r.BAND.DAT", "r_band.dat", "r.band.band.dat", "r.DOSS.DAT", "r_doss.dat",
             "x_IRSPEC.DAT", "x_RAMSPEC.DAT", "x.out", "x-123.o", "slurm-123.out", "materials.db", "a.json", "p.png",
             "p.svg", "p.JPG", "p.gif", "p.html", "p.htm"]
    if CORPUS.is_dir():
        names += sorted({p.name for p in CORPUS.rglob("*") if p.is_file()})
    changed = [(n, old(n), filekind.kind_by_name(n)) for n in names if old(n) and old(n) != filekind.kind_by_name(n)]
    assert not changed, changed[:10]


def test_new_kinds_by_name_and_cube_header(tmp_path):
    """Kinds hidden before; a .DAT named like a cube is a cube only with a cube header."""
    from mace.gui import filekind
    K = filekind.kind_by_name
    assert [K(n) for n in ("a_DENS.CUBE", "b.cube", "DENS_CUBE.DAT", "POT_CUBE.DAT", "SPIN_CUBE.DAT", "x_WAN_CUBE.DAT",
                           "y.CUBE.DAT")] == ["cube"] * 7
    assert [K(n) for n in ("m.xyz", "m.extxyz", "POSCAR", "CONTCAR", "c.vasp", "m.mol", "m.sdf", "p.pdb", "g.gen", "fort.34",
                           "r.f25", "fort.25", "r.SEEBECK.dat", "r.SIGMAS.dat", "r.KAPPA.dat", "r.TDF.dat", "FREQINFO.DAT",
                           "x.f9", "x.f98", "fort.9", "job.sh", "run.log", "j.e", "j.err", "j.o12345", "j.e12345")] == \
        ["xyz", "xyz", "poscar", "poscar", "poscar", "mol", "mol", "pdb", "gen", "fort34", "f25", "f25", "transport",
         "transport", "transport", "transport", "crystaldata", "wavefunction", "wavefunction", "wavefunction", "script",
         "log", "log", "log", "log", "log"]
    assert filekind.cube_type("a_DENS.CUBE") == "density" and filekind.cube_type("POT_CUBE.DAT") == "potential"
    assert filekind.cube_type("x_SPIN.CUBE") == "spin" and filekind.cube_type("t_WAN_CUBE.DAT") == "orbital"
    good = tmp_path / "DENS_CUBE.DAT"
    good.write_text("cube\ncomment\n 1 0.0 0.0 0.0\n 2 0.5 0.0 0.0\n 2 0.0 0.5 0.0\n 2 0.0 0.0 0.5\n 1 1.0 0.0 0.0 0.0\n"
                    " 1.0 2.0 3.0 4.0 5.0 6.0\n 7.0 8.0\n")
    bad = tmp_path / "x_CUBE.DAT"
    bad.write_text("not a cube at all\njust text\nhello world\n\n\n\n\n")
    assert filekind.cube_header_ok(good) and not filekind.cube_header_ok(bad)


def test_committed_fixtures_classify(tmp_path):
    """CI coverage without the corpus: the shipped example outputs are CRYSTAL outputs
    (SP, OPT); a synthetic SLURM log beside one is a job log."""
    from mace.gui import filekind
    out = H.EXAMPLES / "outputs"
    r = filekind.refine(out / "Si_sp.out")
    assert (r["role"], r["calc"], r["dim"]) == ("crystal_out", "SP", 3)
    r = filekind.refine(out / "Ag2Br3_opt_unfinished.out")
    assert (r["role"], r["calc"]) == ("crystal_out", "OPT")
    r = filekind.refine(out / "graphene_slab_sp.out")
    assert (r["role"], r["dim"]) == ("crystal_out", 2)
    log = tmp_path / "mol-123456.o"
    log.write_text("submit directory: \n/scratch/someone/run\nFound queue manager at: /opt/mace/queue/manager.py\n\n"
                   "=== Queue Manager Callback (completion) - 2026-01-01 00:00:00 ===\n=== Callback Complete ===\n")
    assert filekind.refine(log)["role"] == "joblog"


# ---------------------------------------------------------------- plan E2: molecules
@needs_corpus
@pytest.mark.parametrize("folder", ["FREQ", "OPT", "SP"])
def test_molecule_outputs_have_frames(folder, tmp_path):
    """Every 0D output gives >= 1 frame, dim 0, and the atoms CRYSTAL printed: the same
    as d12_parsers.CrystalOutputParser (and, for FREQ, Crystal23FreqParser) to 1e-6 A."""
    pytest.importorskip("ase")          # the frames are built with ase's cell geometry
    np = pytest.importorskip("numpy")
    from mace.gui import outindex
    from d12_parsers import CrystalOutputParser
    n0 = 0
    bad = []
    for src in _runs(folder):
        text = src.read_text(errors="replace")
        if not re.search(r"DIMENSIONALITY OF THE SYSTEM\s+0\s*$", text, re.M):
            continue
        n0 += 1
        out = _copy_run(src, tmp_path / "run")
        idx = outindex.build(out)
        if idx.n_steps < 1 or idx.dim != 0:
            bad.append((src.name[-40:], idx.n_steps, idx.dim))
            continue
        last = idx.frame(-1)["structure"]
        ref = CrystalOutputParser(str(out)).parse()["coordinates"]
        ref_xyz = np.array([[float(a["x"]), float(a["y"]), float(a["z"])] for a in ref])
        if len(ref_xyz) != len(last["numbers"]) or np.abs(np.asarray(last["cart"]) - ref_xyz).max() > 1e-6:
            bad.append((src.name[-40:], "coords differ from CrystalOutputParser"))
        if folder == "FREQ" and idx.freq:
            from mace.plotting.engines.vibmode_viewer import Crystal23FreqParser
            p = Crystal23FreqParser(str(out))
            fx = np.array([a[1:] for a in p.atoms], float)
            if np.abs(np.asarray(last["cart"]) - fx).max() > 1e-6:
                bad.append((src.name[-40:], "coords differ from Crystal23FreqParser"))
        if folder == "OPT":
            # CRYSTAL prints only the initial and the final table for a molecule (plan F15)
            if idx.labels not in (["initial", "final"], ["initial"]):
                bad.append((src.name[-40:], idx.labels))
        shutil.rmtree(tmp_path / "run")
    assert n0 > 50, n0
    assert not bad, bad[:10]


def _frozen(module: str, tmp_path: Path):
    """mace/gui/<module>.py as it was at the plan's base, frozen in tests/gui/frozen."""
    return (FROZEN / f"gui_{module}.base.txt").read_text()


@needs_corpus
def test_periodic_outputs_unchanged_against_the_frozen_index(tmp_path):
    """For every 3D/2D output, frames, labels, energies and dim are identical to the
    index at the base commit (the oracle is the frozen implementation, from git)."""
    pytest.importorskip("ase")          # the frames are built with ase's cell geometry
    np = pytest.importorskip("numpy")
    import types
    from mace.gui import outindex, structures
    pkg = types.ModuleType("frozen_gui")
    pkg.__path__ = []
    sys.modules["frozen_gui"] = pkg
    mods = {}
    for name in ("structures", "outindex"):
        src = _frozen(name, tmp_path)
        f = tmp_path / f"frozen_{name}.py"
        f.write_text(src.replace("from . import structures as S", "import frozen_gui.structures as S"))
        spec = importlib.util.spec_from_file_location(f"frozen_gui.{name}", f)
        m = importlib.util.module_from_spec(spec)
        sys.modules[f"frozen_gui.{name}"] = m
        spec.loader.exec_module(m)
        mods[name] = m
    try:
        bad, n = [], 0
        for folder in ("FREQ", "OPT", "SP", "FAILED_QA"):
            for src in _runs(folder):
                old = mods["outindex"].build(src)
                if not old.n_steps:
                    continue                          # the molecules: nothing to compare (they had no frames)
                new = outindex.build(src)
                n += 1
                if (old.dim, old.labels, old.energies(), old.n_steps) != (new.dim, new.labels, new.energies(), new.n_steps):
                    bad.append(src.name[-40:])
                    continue
                for k in (0, -1):
                    a, b = old.frame(k)["structure"], new.frame(k)["structure"]
                    if np.abs(np.asarray(a["cart"]) - np.asarray(b["cart"])).max() > 0:
                        bad.append((src.name[-40:], k))
        assert n > 100 and not bad, (n, bad[:10])
    finally:
        for k in [k for k in sys.modules if k.startswith("frozen_gui")]:
            del sys.modules[k]


@needs_corpus
def test_cut_molecule_table_is_dropped(tmp_path):
    """A molecule OPT written in slices: a cut table is left out, never a half
    molecule; the frames grow as the file does (a prefix of the final frames)."""
    pytest.importorskip("ase")          # the frames are built with ase's cell geometry
    from mace.gui import outindex
    src = next(p for p in _runs("OPT") if re.search(r"DIMENSIONALITY OF THE SYSTEM\s+0\s*$", p.read_text(errors="replace"), re.M))
    data = src.read_bytes()
    full = outindex.build(src)
    assert full.labels == ["initial", "final"]
    starts = [m.start() for m in re.finditer(rb"DIMENSIONALITY OF THE SYSTEM\s+0", data)]
    out = tmp_path / "grow.out"
    prev = []
    for cut in (starts[0] + 700, starts[0] + 4000, starts[-1] + 700, len(data)):
        out.write_bytes(data[:cut])
        idx = outindex.build(out)
        labs = idx.labels
        assert labs == full.labels[:len(labs)], (cut, labs)
        assert len(labs) >= len(prev)
        for k in range(idx.n_steps):
            assert len(idx.frame(k)["structure"]["numbers"]) == len(full.frame(0)["structure"]["numbers"])
        prev = labs
    assert prev == full.labels


@needs_corpus
def test_job_logs_link_to_their_output(tmp_path):
    """Every corpus log (copied as an empty file with its name, beside empty
    copies of its folder's outputs) links to the output whose stem starts its
    name, and find_slurm_job_id on that output names one of the logs linked to it."""
    from mace.gui import joblink
    from mace.completion_checker import find_slurm_job_id
    total = 0
    for folder in ("FREQ", "OPT", "SP", "BAND", "DOSS", "ECH3POT3"):
        d = tmp_path / folder
        d.mkdir()
        names = [p.name for p in (CORPUS / folder).iterdir() if p.suffix in (".o", ".out")]
        for n in names:
            (d / n).write_bytes(b"")
        linked = {}
        for n in names:
            if not n.endswith(".o"):
                continue
            r = joblink.link(tmp_path, d / n)
            total += 1
            stem_out = [o for o in names if o.endswith(".out") and n.startswith(o[:-4]) and n[len(o) - 4] in "-._"]
            want = max(stem_out, key=len) if stem_out else None
            got = r["calc"].split("/")[-1] if r["calc"] else None
            assert got == want, (n, got, want)
            if got:
                linked.setdefault(got, set()).add(r["job_id"])
        for out, ids in linked.items():
            assert find_slurm_job_id(d / out) in ids, (out, ids)
        shutil.rmtree(d)
    assert total == 551
