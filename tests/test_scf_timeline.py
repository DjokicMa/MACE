"""The SCF convergence timeline (out_convergence.scf_timeline), the shared data
of the run view's SCF card, Run Analysis › SCF and `mace analyze --convergence
FILE --scf`. Real outputs from the gitignored test/ corpus (skipped without
it): a spin-polarized SP, a spin-locked OPT, a conducting SP, a FREQ with two
SCFs; the spin lock's release and TOO MANY CYCLES, which the corpus lacks, on
small synthetic outputs; a growing output read as the GUI reads a live one."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "test"
sys.path[:0] = [p for p in (str(REPO), str(REPO / "mace")) if p not in sys.path]

from mace.utils import out_convergence as oc  # noqa: E402

SP = CORPUS / "SP" / "1_dia_opt_rev1_sp_B3LYP-D3-D3_optimized.out"
OPT = CORPUS / "OPT" / "1_dia_opt_rev1.out"
COND = CORPUS / "SP" / ("Ti19O30_mp-760075_sg1_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3"
                        "_optimized_sp_B3LYP-D3-D3_optimized.out")
FREQ = CORPUS / "FREQ" / ("Ag2Cl1_sym_CRYSTAL_OPT_symm_PBE-D3_full.basis.triplezeta_opt_B3LYP-D3-D3_optimized"
                          "_freq_B3LYP-D3-D3_optimized.out")


def _copy(tmp_path, src):
    if not src.is_file():
        pytest.skip("test/ corpus not present (gitignored)")
    dst = tmp_path / src.name
    shutil.copyfile(src, dst)
    return dst


def _env():
    return {k: v for k, v in os.environ.items() if k not in ("COLORTERM", "TERM", "NO_COLOR", "MACE_HOME", "PYTHONPATH")}


def _kinds(block):
    return {e["kind"]: e["cyc"] for e in block["events"]}


def test_spin_polarized_sp(tmp_path):
    tl = oc.scf_timeline(oc.parse(_copy(tmp_path, SP)))
    assert len(tl["blocks"]) == 1
    b = tl["blocks"][0]
    assert b["label"] == "SCF" and b["ended"] == "energy" and not b["running"]
    assert [c["cyc"] for c in b["cycles"]] == list(range(9))
    assert b["cycles"][2]["detot"] == -3.17e-03 and b["cycles"][2]["etot"] == -76.19401797198
    assert tl["spin"] and b["cycles"][1]["spin"] == 0.0                   # SUMMED SPIN DENSITY per cycle
    assert _kinds(b) == {"fmix": 1, "diis": 2}                            # MIX 30 % after cycle 1, DIIS from cycle 2
    assert b["events"][0]["text"] == "Fock/KS mixing 30%"
    assert tl["toldee"]["value"] == 1e-9 and tl["toldee"]["source"] == "out"
    assert tl["settings"]["levshift_disabled"] and tl["settings"]["fmixing"]["value"] == 30


def test_opt_first_point_and_cycles_per_point(tmp_path):
    tl = oc.scf_timeline(oc.parse(_copy(tmp_path, OPT)))
    assert [b["label"] for b in tl["blocks"]] == ["POINT 1"]
    assert tl["moved_to_scfout_log"] is True
    assert [(p["point"], p["scf_cycles"], p["detail"]) for p in tl["points"]] == [(1, 6, True), (2, 5, False), (3, 5, False), (4, 3, False)]
    b = tl["blocks"][0]
    # SPINLOCK for 50 cycles, on through the whole 7-cycle SCF: locked, never released
    assert tl["settings"]["spinlock"]["cycles"] == 50
    assert "spinlock_release" not in _kinds(b)
    assert _kinds(b) == {"fmix": 1, "diis": 2}


def test_conducting_state_is_marked(tmp_path):
    tl = oc.scf_timeline(oc.parse(_copy(tmp_path, COND)))
    b = oc.scf_block(tl)
    assert "conducting" in _kinds(b)
    assert next(e for e in b["events"] if e["kind"] == "conducting")["text"] == "possibly conducting"


def test_freq_blocks_and_step_choice(tmp_path):
    tl = oc.scf_timeline(oc.parse(_copy(tmp_path, FREQ)))
    assert len(tl["blocks"]) == 2
    assert oc.scf_block(tl)["index"] == 1 and oc.scf_block(tl, "latest")["index"] == 1
    assert oc.scf_block(tl, "first")["index"] == 0 and oc.scf_block(tl, "1")["index"] == 0
    assert oc.scf_block(tl, 3) is None and oc.scf_block({"blocks": []}) is None


SYNTH = """ ALPHA-BETA ELECTRONS LOCKED TO   2 FOR   3 SCF CYCLES
 CONVERGENCE ON ENERGY        10**- 7
 CYC   0 ETOT(AU) -1.000000000000E+01 DETOT -1.00E+01 tst  0.00E+00 PX  1.00E+00
 SPIN LOCKING: NO ENERGY GAP COMPUTED
 SUMMED SPIN DENSITY           2.00000000
 CYC   1 ETOT(AU) -1.050000000000E+01 DETOT -5.00E-01 tst  0.00E+00 PX  1.00E+00
 DIIS TEST: 0.25649E-01 AT SCF   CYCLE   1 - MIX   50 %
 SPIN LOCKING: NO ENERGY GAP COMPUTED
 SUMMED SPIN DENSITY           2.00000000
 CYC   2 ETOT(AU) -1.051000000000E+01 DETOT -1.00E-02 tst  0.00E+00 PX  1.00E+00
 DIIS TEST: 0.86867E+01 AT SCF   CYCLE   2 - DIIS ACTIVE - HISTORY:     2 CYCLES
 SPIN LOCKING: NO ENERGY GAP COMPUTED
 SUMMED SPIN DENSITY           2.00000000
 CYC   3 ETOT(AU) -1.052000000000E+01 DETOT -1.00E-02 tst  0.00E+00 PX  1.00E+00
 SUMMED SPIN DENSITY           1.60000000
 CYC   4 ETOT(AU) -1.051990000000E+01 DETOT  1.00E-04 tst  0.00E+00 PX  1.00E+00
 SUMMED SPIN DENSITY           1.50000000
"""


def test_spin_lock_release_and_too_many_cycles(tmp_path):
    p = tmp_path / "synth.out"
    p.write_text(SYNTH + " == SCF ENDED - TOO MANY CYCLES   E(AU) -1.0519900000000E+01 CYCLES   4\n")
    tl = oc.scf_timeline(oc.parse(p))
    b = tl["blocks"][0]
    assert _kinds(b) == {"fmix": 1, "diis": 2, "spinlock_release": 3, "too_many_cycles": 4}
    assert [c["spin"] for c in b["cycles"]] == [2.0, 2.0, 2.0, 1.6, 1.5]
    assert b["ended"] == "too_many_cycles"
    assert tl["settings"]["spinlock"] == {"value": 2, "cycles": 3, "line": 1}
    tsv = oc.scf_tsv(b, "meV").splitlines()
    assert tsv[0] == "cycle\tE_Ha\tdE_meV\tspin\tevents"
    row = tsv[4].split("\t")
    assert row[0] == "3" and row[3] == "1.6" and row[4] == "spin lock released"
    assert abs(float(row[2]) - (-1.0e-2 * oc.HARTREE_EV * 1000)) < 1e-9
    assert tsv[5].split("\t")[4] == "TOO MANY CYCLES"


def test_growing_output_is_read_as_it_grows(tmp_path):
    """The GUI reads a live output to its last complete line (runinfo.convergence): the
    running SCF gains cycles, is marked running, and ends when SCF ENDED arrives."""
    p = _copy(tmp_path, SP)
    full = p.read_bytes()
    marks = [full.find(f" CYC {k:>3} ETOT".encode()) for k in (2, 5)] + [full.find(b"== SCF ENDED") + 200]
    seen = []
    for cut in marks:
        q = tmp_path / "live.out"
        q.write_bytes(full[:cut + 30])                     # a cut inside a line: that line is left out
        tl = oc.scf_timeline(oc.parse(q, size=q.stat().st_size, complete_lines_only=True))
        b = tl["blocks"][-1]
        seen.append((len(b["cycles"]), b["running"], b["ended"]))
    assert seen[0][0] < seen[1][0] < seen[2][0] == 9
    assert seen[0][1] and seen[1][1] and seen[0][2] is None
    assert seen[2] == (9, False, "energy")


def test_cli_scf_matches_the_shared_function(tmp_path):
    p = _copy(tmp_path, FREQ)
    res = oc.parse(p)
    tl = oc.scf_timeline(res)
    cli = [sys.executable, str(REPO / "mace_cli"), "--no-banner", "analyze", "--convergence", str(p), "--scf"]
    r = subprocess.run(cli, capture_output=True, text=True, env=_env(), timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout == oc.render_scf_text(tl, oc.scf_block(tl), "Ha") + "\n"
    assert oc.scf_tsv(oc.scf_block(tl), "Ha") in r.stdout + "\n"
    r = subprocess.run(cli + ["--step", "first", "--unit", "meV"], capture_output=True, text=True, env=_env(), timeout=120)
    assert r.returncode == 0 and oc.scf_tsv(oc.scf_block(tl, "first"), "meV").rstrip("\n") in r.stdout
    r = subprocess.run(cli + ["--json"], capture_output=True, text=True, env=_env(), timeout=120)
    j = json.loads(r.stdout)
    assert j["selected"] == 1 and j["blocks"] == json.loads(json.dumps(tl["blocks"]))
    r = subprocess.run(cli + ["--step", "nope"], capture_output=True, text=True, env=_env(), timeout=120)
    assert r.returncode == 2 and "--step" in r.stderr
    h = subprocess.run([sys.executable, str(REPO / "mace_cli"), "analyze", "--help"], capture_output=True, text=True,
                       env=_env(), timeout=120)
    assert "--scf" in h.stdout and "--step" in h.stdout


# ------------------------------------------------------------------ SCFOUT.LOG (an optimization's later points)
def scfout_block(energies, final, dft="DFT"):
    """One SCF as CRYSTAL23 writes it to SCFOUT.LOG (the layout of a real one from an
    HPCC OPT: no POINT line, CYC lines, SCF ENDED, the TOTAL ENERGY line without "DE (AU)")."""
    out = [" DFT PARAMETERS", "", " CRYSTAL - SCF - TYPE OF CALCULATION :  UNRESTRICTED OPEN SHELL",
           " RESTART FROM A PREVIOUS DENSITY MATRIX - DEP ACTIVE"]
    prev = None
    for k, e in enumerate(energies):
        de = e if prev is None else e - prev
        out.append(f" CYC {k:>3} ETOT(AU) {e: .12E} DETOT {de: .2E} tst  0.00E+00 PX  1.00E+00")
        out.append(" TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT DIIS        TELAPSE       16.40 TCPU       13.56")
        if k == 1:
            out.append(f" DIIS TEST: 0.38986E-05 AT SCF   CYCLE   {k} - MIX   30 %")
        elif k > 1:
            out.append(f" DIIS TEST: 0.38986E-05 AT SCF   CYCLE   {k} - DIIS ACTIVE - HISTORY:     {k} CYCLES")
        out.append(" SUMMED SPIN DENSITY           0.00000000")
        prev = e
    n = len(energies) - 1
    out.append(f" == SCF ENDED - CONVERGENCE ON ENERGY      E(AU) {final: .13E} CYCLES {n:>3}")
    out.append(f" TOTAL ENERGY({dft})(AU)({n:>3}) {final: .13E} DE 3.1E-09 tester 9.9E-12")
    return "\n".join(out) + "\n"


def _dia_with_scfout(tmp_path, order=(2, 3, 4), energies=True):
    p = _copy(tmp_path, OPT)
    pts = {q["point"]: q for q in oc.parse(p)["points"]}
    text = ""
    for n in order:
        e = pts[n]["energy_scf_au"] if energies else -1.0 - n
        cyc = [e + 10 ** -(k + 2) for k in range(pts[n]["scf_cycles"])] + [e]
        text += scfout_block(cyc, e)
    (tmp_path / (p.stem + ".SCFOUT.LOG")).write_text(text)
    return p


def test_scfout_log_gives_every_point_its_scf(tmp_path):
    p = _dia_with_scfout(tmp_path, order=(3, 2, 4))             # matched by energy, whatever their order
    res = oc.parse(p)
    assert res["scf"]["scfout"] == p.stem + ".SCFOUT.LOG"
    tl = oc.scf_timeline(res)
    assert [b["label"] for b in tl["blocks"]] == ["POINT 1", "POINT 2", "POINT 3", "POINT 4"]
    assert all(p_["detail"] for p_ in tl["points"])
    assert [len(b["cycles"]) for b in tl["blocks"]] == [7, 6, 6, 4]
    assert tl["blocks"][1]["source"] == p.stem + ".SCFOUT.LOG" and tl["blocks"][0]["source"] is None
    b4 = oc.scf_block(tl)                                         # Latest: the last point
    assert b4["point"] == 4 and b4["ended"] == "energy"
    assert {e["kind"] for e in b4["events"]} == {"fmix", "diis"}
    assert oc.scf_block(tl, "point:3")["cycles"][-1]["etot"] == pytest.approx(-76.217215565143, abs=1e-9)


def test_scfout_log_in_order_when_energies_do_not_match_and_stale_is_ignored(tmp_path):
    p = _dia_with_scfout(tmp_path, energies=False)
    tl = oc.scf_timeline(oc.parse(p))
    assert [b["point"] for b in tl["blocks"]] == [1, 2, 3, 4]       # the file's order: points 2, 3, 4
    log = tmp_path / (p.stem + ".SCFOUT.LOG")
    os.utime(log, (p.stat().st_mtime - 3600, p.stat().st_mtime - 3600))   # an earlier run's
    res = oc.parse(p)
    assert res["scf"]["scfout"] is None and len(oc.scf_timeline(res)["blocks"]) == 1


def test_cli_step_reads_the_scfout_log(tmp_path):
    p = _dia_with_scfout(tmp_path)
    cli = [sys.executable, str(REPO / "mace_cli"), "--no-banner", "analyze", "--convergence", str(p), "--scf"]
    r = subprocess.run(cli + ["--step", "point:3"], capture_output=True, text=True, env=_env(), timeout=120)
    assert r.returncode == 0, r.stderr
    tl = oc.scf_timeline(oc.parse(p))
    assert r.stdout.startswith("SCF of POINT 3 (6 cycles, converged)")
    assert oc.scf_tsv(oc.scf_block(tl, "point:3"), "Ha").rstrip("\n") in r.stdout
    assert "SCF cycles per point" in r.stdout
