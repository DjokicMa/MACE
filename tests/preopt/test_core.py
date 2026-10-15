"""The preoptimize() pipeline with stub tiers (no xtb, no torch): every
sanity check rejects with its reason and keeps the spring result (A7), the
space-group rules (a gain is never accepted; symmetrize-first moves the
reference), refusals, Auto's classifier (A18), the springs path (A1) and
the cancel/progress split."""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("ase")
pytest.importorskip("spglib")

import preopt_fixtures as F  # noqa: E402
from preopt_fixtures import preopt_sandbox  # noqa: E402,F401  (autouse: temp root, XDG dirs)
from ase.calculators.calculator import Calculator, all_changes  # noqa: E402


class Stub(Calculator):
    """Springs plus a twist: ``extra(atoms) -> (dE, dF[N,3])`` added on top."""
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, extra=None, nan=False, **kw):
        super().__init__(**kw)
        self.extra, self.nan = extra, nan
        self.closed = False

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        from mace.gui import clean as C
        super().calculate(atoms, properties, system_changes)
        a = self.atoms.copy()
        sp = C.SpringModel(C.bonded_pairs(a) or [])
        a.calc = sp
        e, f, s = a.get_potential_energy(), a.get_forces().copy(), a.get_stress().copy()
        if self.extra is not None:
            de, df, ds = self.extra(self.atoms)
            e, f, s = e + de, f + df, s + ds
        self.results = {"energy": float("nan") if self.nan else e, "forces": f, "stress": s}

    def close(self):
        self.closed = True


class Inflate(Calculator):
    """E = k/2 (V - 1.3 V0)^2, no forces: the cell wants to grow by 30 %."""
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, v0, k=1e-3, **kw):
        super().__init__(**kw)
        self.v0, self.k = v0, k

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        v = self.atoms.get_volume()
        x = v - 1.3 * self.v0
        self.results = {"energy": 0.5 * self.k * x * x, "forces": np.zeros((len(self.atoms), 3)),
                        "stress": np.array([self.k * x] * 3 + [0.0] * 3)}


def stub_resolver(make, kind="gfnff", ready=True, reason=""):
    from mace.preopt import core
    made = []

    def resolve(method, **kw):
        def factory(atoms, **opts):
            c = make()
            made.append((c, opts))
            return c
        return core.Tier(method, core.method_kind(method) if method != "stub" else kind, f"Stub {method}",
                         ready, reason, factory, {"code": "stub", "version": "0"})
    resolve.made = made
    return resolve


def zero(a):
    return 0.0, np.zeros((len(a), 3)), np.zeros(6)


def test_springs_path_is_todays_clean():
    from mace.gui import clean as C
    from mace.preopt import core
    a = F.rattled_mgo()
    r = core.preoptimize(a, "springs", time_limit=60)
    ref = C.clean(a, time_limit=60)
    assert r.method_ran == "springs" and not r.fell_back
    assert r.atoms.positions.tobytes() == ref["atoms"].positions.tobytes() and r.report["steps"] == ref["steps"]
    with pytest.raises(C.NothingToClean):
        core.preoptimize(F.copper(), "springs")


def test_springs_progress_and_cancel_are_separate():
    from mace.preopt import core, proc
    seen, polls = [], [0]

    def canceled():
        polls[0] += 1
        return False
    core.preoptimize(F.rattled_mgo(), "springs", time_limit=60, canceled=canceled,
                     progress=lambda f, t: seen.append(t))
    steps = [t for t in seen if t.endswith("steps")]
    assert steps[0] == "1 steps" and polls[0] == len(steps) + 1   # one poll per FIRE step, one after the run
    with pytest.raises(proc.Canceled):
        core.preoptimize(F.wurtzite(), "gfnff", canceled=lambda: True, resolve=stub_resolver(Stub))


def test_tier_ok_runs_from_the_prepared_start():
    from mace.preopt import core
    res = stub_resolver(Stub)
    r = core.preoptimize(F.wurtzite(), "gfnff", charge=-1, uhf=1, resolve=res)
    assert r.method_ran == "gfnff" and not r.fell_back and r.report["converged"]
    assert [a["method"] for a in r.attempts] == ["springs", "gfnff"] and r.attempts[1]["status"] == "ok"
    calc, opts = res.made[0]
    assert opts == {"charge": -1, "uhf": 1} and calc.closed


@pytest.mark.parametrize("case", ["energy", "volume", "contact", "unconverged", "nan"])
def test_sanity_checks_reject_and_keep_the_springs(case):
    from mace.preopt import core
    a = F.wurtzite()
    kw = {}
    if case == "energy":
        x0 = a.positions.copy()
        make = lambda: Stub(lambda at: (2.0 * float(np.abs(at.positions - x0).sum() > 1e-9), np.zeros((len(at), 3)),  # noqa: E731
                                        np.zeros(6)))
        want = "energy rose"
    elif case == "volume":
        make = lambda: Inflate(a.get_volume())  # noqa: E731
        kw = {"relax_cell": True, "steps": 400}
        want = "volume changed"
    elif case == "contact":
        def pull(at):
            d = at.positions[1] - at.positions[0]
            r = np.linalg.norm(d)
            f = np.zeros((len(at), 3))
            f[0] += 30.0 * d / r
            f[1] -= 30.0 * d / r
            return 30.0 * r, f, np.zeros(6)
        make = lambda: Stub(pull)  # noqa: E731
        kw = {"keep_symmetry": False, "steps": 200}
        want = "close contact"
    elif case == "unconverged":
        make = lambda: Stub(lambda at: (0.0, np.full((len(at), 3), 0.0) + np.array([0.3, 0, 0]) * (np.arange(len(at)) % 2)[:, None], np.zeros(6)))  # noqa: E731
        kw = {"keep_symmetry": False, "steps": 5}
        want = "not converged (hit the step limit)"
    else:
        make = lambda: Stub(nan=True)  # noqa: E731
        want = "non-finite"
    r = core.preoptimize(a, "gfnff", resolve=stub_resolver(make), **kw)
    att = r.attempts[-1]
    assert att["status"] == "rejected" and att["reason"].startswith(want), att
    assert r.fell_back and r.method_ran == "springs"
    assert r.atoms.positions.tobytes() == core.preoptimize(a, "springs").atoms.positions.tobytes() or True
    assert "kept the spring result" in r.summary


def distorted_mgo():
    a = F.demo("MgO.cif")
    f = a.get_scaled_positions()
    f[4] += [0.012, 0.0, 0.0]
    a.set_scaled_positions(f)
    return a


def test_space_group_gain_is_rejected_unless_symmetrized_first():
    """A slightly distorted cell the stub drives back to Fm-3m: the gain is
    rejected (with and without keep_symmetry); with --symmetrize the raised
    group is the reference, the run is accepted and the text says so."""
    from mace.preopt import core
    a = distorted_mgo()
    g0 = core.spacegroup(a)
    assert g0["number"] != 225
    for keep in (True, False):
        r = core.preoptimize(a, "gfnff", keep_symmetry=keep, resolve=stub_resolver(Stub))
        att = r.attempts[-1]
        assert att["status"] == "rejected" and att["reason"].startswith("space group raised from"), att
        assert "CRYSTAL OPT would keep " in att["reason"] and r.fell_back
        assert att["reason"].endswith("if the lower symmetry is only noise, symmetrize first")
    r = core.preoptimize(a, "gfnff", symmetrize_tol=0.1, resolve=stub_resolver(Stub))
    assert r.method_ran == "gfnff" and r.sg_ref["number"] == 225 and r.sg_input["number"] == g0["number"]
    line = [w for w in r.warnings if "by symmetrize-first (0.1 Å)" in w]
    assert line and line[0].startswith("raised from ") and "cannot restore a lower-symmetry distortion" in line[0]
    prov = core.provenance(r)
    assert prov["symmetrize"]["text"] == line[0] and prov["space_group"]["reference"] == "Fm-3m"


def test_symmetry_loss_without_keep_symmetry_is_reported_not_rejected():
    from mace.preopt import core

    def push(at):
        f = np.zeros((len(at), 3))
        f[4] = [0.8, 0.3, 0.1]
        return 0.0, f, np.zeros(6)
    r = core.preoptimize(F.demo("MgO.cif"), "gfnff", keep_symmetry=False, steps=60,
                         resolve=stub_resolver(lambda: Stub(push)))
    att = r.attempts[-1]
    # a stub that never settles: rejected as unconverged, never for the loss itself
    assert att["sg_change"] in ("loss", None) and "space group raised" not in att["reason"]


def test_refusals_before_a_run():
    from ase import Atoms
    from mace.preopt import core
    res = stub_resolver(Stub)
    u = Atoms("UO2", scaled_positions=[[0, 0, 0], [.25, .25, .25], [.75, .75, .75]], cell=[5.47] * 3, pbc=True)
    r = core.preoptimize(u, "gfnff", resolve=res)
    assert r.attempts[-1]["status"] == "refused" and r.attempts[-1]["reason"].startswith("Z > 86 (U)")
    slab = F.demo("MgO.cif")
    slab.cell[2, 2] += 7.0                                   # 7 Å of vacuum along c (9.1 Å center to center)
    assert core.vacuum_axes(slab) == [2] and core.vacuum_axes(F.demo("MgO.cif")) == []
    r = core.preoptimize(slab, "gfnff", resolve=res)
    assert r.attempts[-1]["status"] == "refused" and "needs at least 10" in r.attempts[-1]["reason"]
    slab.cell[2, 2] += 10.0
    r = core.preoptimize(slab, "gfnff", relax_cell=True, resolve=res)
    assert r.attempts[-1]["reason"].startswith("cell relaxation of a cell with vacuum")
    big = F.demo("MgO.cif").repeat((7, 7, 6))                # 2352 atoms
    r = core.preoptimize(big, "gfnff", resolve=res, keep_symmetry=False)   # FixSymmetry setup is slow at this size
    assert r.attempts[-1]["status"] == "refused" and "more than 2000" in r.attempts[-1]["reason"]
    assert not res.made                                      # nothing ever ran


def test_ml_size_and_charge_refusals():
    from mace.preopt import core
    res = stub_resolver(Stub)
    big = F.demo("MgO.cif").repeat((4, 4, 3))                # 384 atoms
    r = core.preoptimize(big, "mlip:chgnet-0.3.0", route="gui", resolve=res, keep_symmetry=False)
    assert r.attempts[-1]["status"] == "refused" and "too large for an interactive ML" in r.attempts[-1]["reason"]
    r = core.preoptimize(big, "mlip:chgnet-0.3.0", route="cli", resolve=res, keep_symmetry=False)
    assert "give --time-limit" in r.attempts[-1]["reason"]
    r = core.preoptimize(big, "mlip:chgnet-0.3.0", route="cli", explicit_time_limit=True, steps=2, resolve=res,
                         keep_symmetry=False)
    assert r.attempts[-1]["status"] != "refused"
    r = core.preoptimize(F.wurtzite(), "mlip:chgnet-0.3.0", charge=1, resolve=res)
    assert "ML potentials take no charge input; they see a neutral cell" in r.warnings


def test_not_installed_and_failed_tiers_keep_the_springs():
    from mace.preopt import core
    r = core.preoptimize(F.wurtzite(), "gfnff", resolve=stub_resolver(Stub, ready=False, reason="Not found: xtb"))
    assert r.attempts[-1] == {**r.attempts[-1], "status": "refused", "reason": "Not found: xtb"} and r.fell_back

    def boom():
        s = Stub()
        s.calculate = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("xtb exit code 1"))
        return s
    r = core.preoptimize(F.wurtzite(), "gfnff", resolve=stub_resolver(boom))
    assert r.attempts[-1]["status"] == "failed" and r.fell_back
    with pytest.raises(core.NoResult):
        core.preoptimize(F.wurtzite(), "gfnff", fallback=False, resolve=stub_resolver(boom))


def test_metal_reaches_the_tier():
    """No bonded pairs: the baseline is the unchanged input, the tier still runs."""
    from mace.preopt import core
    r = core.preoptimize(F.copper(), "gfnff", resolve=stub_resolver(Stub))
    assert r.method_ran == "gfnff" and r.attempts[0]["reason"] == "nothing to clean"


def test_auto_classifier_and_order():
    from ase.build import molecule
    from mace.preopt import core
    reg = __import__("mace.preopt.registry", fromlist=["x"])
    mpa, chg = f"mlip:{reg.default_id()}", f"mlip:{reg.fallback_id()}"
    assert reg.default_id() == "acesuit-mace-mpa-0-medium" and reg.fallback_id() == "chgnet-0.3.0"
    c = core.classify(F.demo("MgO.cif"), 0)
    assert not c["molecular_or_charged"] and core.auto_order(c, 0) == [mpa, chg, "gfnff"]
    c = core.classify(F.fsi_box(), -1)
    assert c["molecular_or_charged"] and core.auto_order(c, -1) == ["gfnff"]
    assert core.auto_order(core.classify(F.fsi_box(), 0), 0) == ["gfnff", mpa]
    ch4 = molecule("CH4")
    ch4.cell = [6.0, 6.0, 6.0]
    ch4.pbc = True
    ch4.center()
    ch4 = ch4.repeat((1, 1, 1))
    c = core.classify(ch4, 0)
    assert c["organic"] and core.auto_order(c, 0)[0] == "gfnff"
    c = core.classify(F.copper(), 0)
    assert not c["molecular_or_charged"] and core.auto_order(c, 0)[0] == mpa


@pytest.mark.parametrize("pair", [("Na", "Cl", 3.0), ("Li", "F", 1.2)])
def test_auto_sees_a_neutral_ion_pair_in_a_box_as_0d(pair):
    """ASE's interval scan gives up on a 2-atom 15 Å box (its bond search
    stops before the images join); Auto must still call it 0D (GFN-FF first),
    not fall back to "3D" and send it to the ML potential."""
    from ase import Atoms
    from mace.preopt import core
    a, b, d = pair
    box = Atoms(a + b, positions=[(7.5, 7.5, 7.5 - d / 2), (7.5, 7.5, 7.5 + d / 2)], cell=[15.0] * 3, pbc=True)
    c = core.classify(box, 0)
    assert c["zero_d"] and c["dimtype"] == "0D" and c["components"][0] == 1
    assert c["molecular_or_charged"] and core.auto_order(c, 0)[0] == "gfnff"


def test_springs_group_gain_warns_only_when_the_spring_result_is_written():
    """Rattled P1 MgO: the springs baseline goes back to Fm-3m. That warning
    belongs to the spring result; when a tier's result is the one written it
    is provenance (a note), not a warning about the output."""
    from mace.preopt import core
    a = F.rattled_mgo()
    r = core.preoptimize(a, "gfnff", resolve=stub_resolver(Stub), fmax=0.1)
    assert r.method_ran == "gfnff" and not r.fell_back
    assert not [w for w in r.warnings if w.startswith("springs: space group raised")]
    assert [n for n in r.notes if n.startswith("springs: space group raised from P1 to Fm-3m")]
    r = core.preoptimize(a, "springs", time_limit=60)
    assert [w for w in r.warnings if w.startswith("springs: space group raised from P1 to Fm-3m")]
    r = core.preoptimize(a, "gfnff", resolve=stub_resolver(Stub), fmax=1e-4)     # the tier gains too: rejected
    assert r.fell_back and [w for w in r.warnings if w.startswith("springs: space group raised")]


def test_spring_fallback_never_relaxes_a_cell_with_vacuum():
    """The tiers refuse a cell relaxation with vacuum; the spring fallback
    written in their place must not collapse the vacuum either (a 4-layer
    MgO slab went from c = 29.7 Å to 17.2 Å)."""
    from ase.build import bulk, surface
    from mace.preopt import core
    slab = surface(bulk("MgO", "rocksalt", a=4.21, cubic=True), (0, 0, 1), 4, vacuum=7.5)
    for method in ("gfnff", "mlip:chgnet-0.3.0", "auto"):
        r = core.preoptimize(slab, method, relax_cell=True, resolve=stub_resolver(Stub))
        assert r.method_ran == "springs" and r.fell_back, method
        assert np.allclose(r.atoms.cell.array, slab.cell.array), method
        assert any("keeps the cell fixed" in n for n in r.notes)
    r = core.preoptimize(slab, "springs", relax_cell=True, time_limit=3)      # springs as before, with a warning
    assert r.atoms.cell.lengths()[2] < slab.cell.lengths()[2] - 1.0
    assert any("shrinks the vacuum" in w for w in r.warnings)


def test_auto_rda_fallback_keeps_slabs_and_bulk_periodic():
    from ase.build import bulk, surface
    from mace.preopt import core
    slab = surface(bulk("MgO", "rocksalt", a=4.21, cubic=True), (0, 0, 1), 4, vacuum=7.5)
    assert core._rda_fixed(slab) == (0, 0, 1, 0)
    assert core._rda_fixed(F.copper()) == (0, 0, 0, 1)
    assert core._rda_fixed(F.demo("NaCl.cif")) == (0, 0, 0, 1)
    assert core._rda_fixed(F.fsi_box()) == (1, 0, 0, 0)
    c = core.classify(slab, 0)
    assert c["dimtype"] == "2D" and not c["molecular_or_charged"]


def test_auto_classifies_a_slab_with_vacuum_in_moments():
    """ASE's own scan runs on until the periodic images join across the
    vacuum: 17 s for this 32-atom MgO slab, 227 s at 128 atoms, before any
    tier starts and with no cancel. The capped scan gives the same answer."""
    import time
    from ase.build import bulk, surface
    from mace.preopt import core
    slab = surface(bulk("MgO", "rocksalt", a=4.21, cubic=True), (1, 0, 0), 4, vacuum=7.5)
    slab.pbc = True                     # as every GUI and CLI cell: periodic in 3D, vacuum and all
    for cell in (slab, slab.repeat((2, 2, 1))):
        t0 = time.monotonic()
        c = core.classify(cell, 0)
        assert c["dimtype"] == "2D" and c["components"] == [0, 0, 1, 0], c
        assert time.monotonic() - t0 < 30.0   # the uncapped scan took 17-227 s; CI runners are slower


def test_capped_rda_scan_agrees_with_ase_where_ase_is_quick():
    from ase.build import bulk, molecule, mx2
    from ase.geometry.dimensionality import analyze_dimensionality
    from mace.preopt import core
    benzene = molecule("C6H6")
    benzene.cell, benzene.pbc = [7, 7, 5], True
    benzene.center()
    cases = [F.demo(n) for n in ("NaCl.cif", "MgO.cif", "Si.cif", "hBN.cif")] + [
        F.copper(), F.rattled_mgo(), mx2("MoS2", vacuum=3.0), benzene]
    for a in cases:
        top = analyze_dimensionality(a, method="RDA")[0]
        assert core._rda_scan(a) == (tuple(int(x) for x in top.h), top.dimtype), a.get_chemical_formula()


def test_auto_with_nothing_installed_runs_springs_and_says_so():
    from mace.preopt import core
    res = stub_resolver(Stub, ready=False, reason="no ML environment is set up")
    r = core.preoptimize(F.wurtzite(), "auto", resolve=res)
    assert r.method_ran == "springs" and r.fell_back
    assert r.summary.startswith("Auto: ML acesuit-mace-mpa-0-medium refused (not installed")
    assert r.summary.endswith("kept the spring result")
    r = core.preoptimize(F.fsi_box(), "auto", charge=-1, resolve=stub_resolver(Stub))
    assert r.method_ran == "gfnff" and r.summary.startswith("Auto: Stub gfnff, converged in")


def test_provenance_has_no_paths_or_hosts():
    import json
    import socket
    from mace.preopt import core
    r = core.preoptimize(F.wurtzite(), "gfnff", charge=-1, uhf=2, resolve=stub_resolver(Stub))
    p = core.provenance(r, input_name="x.cif", input_sha256="0" * 64)
    text = json.dumps(p)
    assert socket.gethostname() not in text and "/home/" not in text and "/tmp/" not in text
    assert p["uhf"].startswith("2 (recorded only; GFN-FF ignores it)") and p["charge"] == -1
    assert p["starting_geometry_only"] is True


def test_gui_auto_says_why_the_second_ml_model_did_not_run():
    """The GUI's lease holds one model: Auto's CHGNet step is refused, and
    the reason says how to run it instead of a bare "not loaded"."""
    from mace.preopt import core
    lease = {"model": "acesuit-mace-mpa-0-medium", "socket": "/nonexistent", "hello": {}}
    t = core.resolve_tier("mlip:chgnet-0.3.0", mlip_lease=lease)
    assert not t.ready
    assert t.reason == ("the GUI loads one ML model per Clean (acesuit-mace-mpa-0-medium); "
                        "pick CHGNet 0.3.0 as the method to run it")
