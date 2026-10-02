"""mace.defects.analysis on the numbers of the HPCC defect runs (FACTS.md).

Outputs are synthetic excerpts carrying the measured energies, electron counts,
Fermi levels and band edges of the MgO V_O (3D) and hBN V_N (2D) runs, so the
expected results are the measured ones: C = 0.444 Ha, eps(2+/+) ~ 1.9 and
eps(+/0) ~ 3.6 eV for MgO; eps(+/0) ~ 3.8 and eps(0/-) ~ 6.6 eV for hBN.
"""
import json

import numpy as np
import pytest

from _defects_fixtures import fake_output, hbn_host, mgo_host

from mace.constants import HARTREE_TO_EV
from mace.defects import DefectError
from mace.defects import analysis as an
from mace.defects import deck_edit as de

MGO_222 = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]

# name: (E, N Mulliken, N header, EFERMI, VBM, CBM, spin, spheropole)  - measured
MGO = {
    "pris": (-8810.8448478966, 640.0, 640, None, -0.21811894, 0.048263216, None, 143.57133471018),
    "q0": (-8735.4571902973, 632.0, 632, None, -0.10369777, 0.062925781, None, 141.2229682454),
    "qp1": (-8735.8147109285, 631.0, 632, None, None, None, 1.0, 140.00301427787),
    "qp2": (-8736.1092241143, 630.0, 632, -0.13170263, -0.2072842, -0.13183017, None, 139.02970834809),
    "calm05": (-8735.6261310083, 631.5, 632, -0.11251191, None, None, None, 140.63210845155),
    "calm10": (-8735.7907575909, 631.0, 632, -0.11906862, None, None, None, 140.07458129098),
    "calm15": (-8735.9514835622, 630.5, 632, -0.12496889, None, None, None, 139.5491708026),
}

HBN = {  # n: {tag: (E, N)}, VBM/CBM of the pristine cells
    3: {"pris": (-716.94545379578, 108.0), "q0": (-661.88091141289, 101.0),
        "qp1": (-661.8521298489, 100.0), "qm1": (-661.9697598078, 102.0)},
    4: {"pris": (-1274.5696957433, 192.0), "q0": (-1219.505210973, 185.0),
        "qp1": (-1219.4563206217, 184.0), "qm1": (-1219.5718053708, 186.0)},
    5: {"pris": (-1991.5151498674, 300.0), "q0": (-1936.4506962909, 293.0),
        "qp1": (-1936.3898464017, 292.0), "qm1": (-1936.5050874288, 294.0)},
    6: {"pris": (-2867.7818159384, 432.0), "q0": (-2812.7173718096, 425.0),
        "qp1": (-2812.6484145211, 424.0), "qm1": (-2812.7637798436, 426.0)},
}
HBN_VBM = {3: (-0.24753863, 0.0062249184), 4: (-0.24753873, 0.0062248458),
           5: (-0.24753876, 0.0062247808), 6: (-0.24753858, 0.0062249712)}


def _mgo_out(tag, **over):
    e, n, h, ef, vbm, cbm, spin, sph = MGO[tag]
    kw = dict(energy=e, electrons=n, header=h, efermi=ef, vbm=vbm, cbm=cbm, spin=spin, sph=sph)
    kw.update(over)
    return fake_output(**kw)


def _manifest(host_text, site, charges, keyword="SUPERCON", sizes=((MGO_222, None, None),), name="mgo",
              **opts):
    """The parts of a build manifest the analysis reads, from the deck builder itself."""
    host, _ = de.classify_host(host_text)
    supercells = []
    for matrix, size, pristine_n in sizes:
        nm = f"{name}{size}" if size else name
        mult = de.supercell_multiplicity(host, keyword, matrix)
        electrons = de.supercell_electrons(host, mult, pristine_n)
        decks, _, _ = de.build_supercell_set(host, site, keyword, matrix,
                                             de.SetOptions(charges=list(charges), **opts), electrons, nm)
        supercells.append({"pristine_job": f"{nm}_pris", "size": size,
                           "jobs": [d.manifest() for d in decks]})
    return {"dimensionality": host.dimensionality,
            "defect": {"kind": site.kind, "element": site.element, "new_element": site.new_element,
                       "cart": None, "delta_electrons": site.delta_z},
            "site": {"label": site.label, "source": "label"},
            "calibration": {"on": opts.get("calibrate_on", "defect")} if opts.get("calibrate") else None,
            "supercells": supercells}


def _mgo_set(charges=(0, 1, 2), calibrate=False, host_text=None, outs=None, skip=()):
    """(manifest, outputs) of an MgO V_O set with the measured numbers."""
    manifest = _manifest(host_text or mgo_host(), de.DefectSite("vacancy", "O", label=1), charges,
                         **({"calibrate": [0.5, 1.0, 1.5]} if calibrate else {}))
    outs = outs or {}
    texts = {}
    for sc in manifest["supercells"]:
        for job in sc["jobs"]:
            tag = job["name"].split("_", 1)[1]
            text = outs.get(tag) or (_mgo_out(tag) if tag in MGO else None)
            if text and tag not in skip:
                texts[job["name"]] = text
    return manifest, texts


def _report(manifest_texts, **kw):
    manifest, texts = manifest_texts
    return an.analyze(manifest, texts, **kw)


def _level(r, name):
    return next(x["eV_above_VBM"] for x in r["levels"] if x["transition"] == name)


def test_mgo_calibrated_levels():
    d = _mgo_set(calibrate=True)
    r = _report(d)
    (cal,) = r["calibration"]
    assert cal["C"] == pytest.approx(0.444, abs=0.002)
    assert cal["q_range"] == [0.0, 2.0] and not cal["single_central"]
    for v in cal["values"]:
        assert v["spheropole_crosscheck"] == pytest.approx(v["C"], abs=0.002)
    assert _level(r, "+1/0") == pytest.approx(3.57, abs=0.05)
    assert _level(r, "+2/+1") == pytest.approx(1.86, abs=0.05)
    assert r["image_charge_correction"] == "none"
    assert "uncorrected for image charges" in r["labels"]
    assert any("unrelaxed" in x for x in r["labels"])
    assert r["pristine"]["vbm_band"] == 320 and r["pristine"]["gap_ev"] == pytest.approx(7.249, abs=1e-3)


def test_uncalibrated_3d_reports_only_the_neutral_state():
    d = _mgo_set()
    r = _report(d)
    assert r["levels"] == []
    raw = (MGO["q0"][0] - MGO["qp1"][0] - MGO["pris"][4]) * HARTREE_TO_EV
    assert raw == pytest.approx(15.66, abs=0.01)          # the impossible raw level
    assert "15.6" not in json.dumps(r)
    reported = {s["q"]: s for s in r["states"]}
    assert "formation_energy_rel_q0_ev_at_vbm" in reported[0.0]
    assert "not reported" in reported[1.0]["status"] and "formation_energy_rel_q0_ev_at_vbm" not in reported[1.0]
    assert any("not reported without --calibrate" in n for n in r["notes"])
    assert r["image_charge_correction"] == "none" and "uncorrected for image charges" in r["labels"]


def test_charges_outside_the_calibrated_range_are_not_reported():
    outs = {"qp3": _mgo_out("qp2", energy=-8736.2, electrons=629.0, spin=1.0)}
    d = _mgo_set(charges=(0, 1, 2, 3), calibrate=True, outs=outs)
    r = _report(d)
    states = {s["q"]: s for s in r["states"]}
    assert "not calibrated" in states[3.0]["status"]
    assert all("+3" not in x["transition"] for x in r["levels"])


def test_spin_polarised_endpoints_fall_back_to_the_single_central_value():
    spin_host = mgo_host().replace("DFT\nPBE0", "DFT\nSPIN\nPBE0")
    outs = {"q0": _mgo_out("q0", spin=0.5), "qp2": _mgo_out("qp2", spin=0.5)}
    d = _mgo_set(calibrate=True, host_text=spin_host, outs=outs)
    r = _report(d)
    (cal,) = r["calibration"]
    assert cal["single_central"] and len(cal["values"]) == 1 and cal["values"][0]["doping"] == -1.0
    assert cal["C"] == pytest.approx(0.444, abs=0.002)
    assert cal["q_range"] == [0.5, 1.5]
    assert any("single central value" in n for n in r["notes"])
    states = {s["q"]: s for s in r["states"]}
    assert "C_ha" in states[1.0] and "not calibrated" in states[2.0]["status"]


def test_closed_shell_unrestricted_endpoints_are_used():
    spin_host = mgo_host().replace("DFT\nPBE0", "DFT\nSPIN\nPBE0")
    outs = {"q0": _mgo_out("q0", spin=0.0), "qp2": _mgo_out("qp2", spin=0.001)}
    d = _mgo_set(calibrate=True, host_text=spin_host, outs=outs)
    (cal,) = _report(d)["calibration"]
    assert len(cal["values"]) == 3 and cal["q_range"] == [0.0, 2.0]


def test_doping_check_spin_check_and_gap_flag():
    outs = {"qp2": _mgo_out("qp2", electrons=631.0),            # DOPING silently not applied
            "qp1": _mgo_out("qp1", spin=0.3, cycles=45)}          # lock released and drifted
    d = _mgo_set(outs=outs)
    d[0]["spinlock_cycles"] = 30
    jobs = {j["name"]: j for j in _report(d)["jobs"]}
    assert jobs["mgo_qp2"]["status"] == "excluded" and "DOPING not applied" in jobs["mgo_qp2"]["reason"]
    assert any("drifted after SPINLOCK was released" in w for w in jobs["mgo_qp1"]["warnings"])
    # measured: N_O q=0 converged in 15 cycles, still locked, with a Mulliken spin of 0.84
    d[1]["mgo_qp1"] = _mgo_out("qp1", spin=0.843544, cycles=15)
    (w,) = {j["name"]: j for j in _report(d)["jobs"]}["mgo_qp1"]["warnings"]
    assert "lock still on" in w and "drifted" not in w
    levels = an.transition_levels({0.0: 0.0, 1.0: -9.0}, 7.25)
    assert levels[0]["eV_above_VBM"] == pytest.approx(9.0) and "outside the gap" in levels[0]["flag"]


def test_unconverged_jobs_are_excluded_with_their_reason():
    outs = {"qp1": _mgo_out("qp1", converged=False)}
    d = _mgo_set(charges=(0, 1), outs=outs)
    jobs = {j["name"]: j for j in _report(d)["jobs"]}
    assert jobs["mgo_qp1"]["status"] == "excluded" and "TOO MANY CYCLES" in jobs["mgo_qp1"]["reason"]


def test_chemical_potentials_need_units_and_give_absolute_energies():
    assert an.parse_energy("-75.1Ha") == -75.1
    assert an.parse_energy("-2043.6 eV") == pytest.approx(-2043.6 / HARTREE_TO_EV)
    with pytest.raises(DefectError, match="needs a unit"):
        an.parse_energy("-75.1")
    assert an.parse_mu(["O=-75.0Ha"]) == {"O": -75.0}
    d = _mgo_set(calibrate=True)
    r = _report(d, mu=an.parse_mu(["O=-75.0Ha"]))
    s0 = next(s for s in r["states"] if s["q"] == 0)
    expected = (MGO["q0"][0] - MGO["pris"][0] - 75.0) * HARTREE_TO_EV
    assert s0["formation_energy_ev_at_vbm"] == pytest.approx(expected)
    assert any("same method and basis" in n for n in r["notes"])
    with pytest.raises(DefectError, match="missing O"):
        _report(d, mu={"Mg": -200.0})


def test_vbm_override():
    d = _mgo_set(calibrate=True)
    base = _level(_report(d), "+1/0")
    shifted = _report(d, vbm=an.parse_energy("-5.835eV"))
    assert _level(shifted, "+1/0") == pytest.approx(base - (-5.835 - MGO["pris"][4] * HARTREE_TO_EV), abs=1e-6)


def test_alignment_sign_and_application():
    from mace.defects.structure import parse_geometry
    from _defects_fixtures import DATA
    geom = parse_geometry((DATA / "mgo_pris_geometry.out").read_text())
    pts = [(a.label, a.cart / 0.52917721067) for a in geom.atoms]
    pris = "".join(f"{i:3d} {x:.4f} {y:.4f} {z:.4f} -1.00000E+00 0 0 0\n" for i, (x, y, z) in pts)
    dv = 0.01   # au, defect cell potential higher everywhere
    dfc = "".join(f"{i:3d} {x:.4f} {y:.4f} {z:.4f} {-1.0 + dv:.5E} 0 0 0\n" for i, (x, y, z) in pts[1:])
    res = an.potential_alignment(geom, geom.atom(1).cart, pris, dfc)
    assert res["dV_ev"] == pytest.approx(dv * HARTREE_TO_EV, rel=1e-4) and res["std_ev"] < 1e-6
    assert an.alignment_term(1, res["dV_ev"]) == pytest.approx(-dv * HARTREE_TO_EV, rel=1e-4)
    assert an.alignment_term(-1, 0.2) == pytest.approx(0.2)
    # through analyze: reported always, applied only when asked
    d = _mgo_set(calibrate=True)
    d[1]["mgo_pris"] = _mgo_out("pris") + (DATA / "mgo_pris_geometry.out").read_text()
    potc = {"mgo_pris": pris, "mgo_qp1": dfc}
    plain = _report(d, potc=potc)
    applied = _report(d, potc=potc, apply_alignment=True)
    assert plain["alignment"]["+1"]["dV_ev"] == pytest.approx(dv * HARTREE_TO_EV, rel=1e-4)
    shift = _level(applied, "+1/0") - _level(plain, "+1/0")
    assert shift == pytest.approx(dv * HARTREE_TO_EV, rel=1e-3)   # E_f(+1) lowered by dV


# --- 2D ---------------------------------------------------------------------------

def _hbn_set(energies=None):
    energies = energies or HBN
    sizes = [([[n, 0], [0, n]], n, 12 * n * n) for n in sorted(energies)]
    manifest = _manifest(hbn_host(), de.DefectSite("vacancy", "N", label=1), (-1, 0, 1), keyword="SUPERCEL",
                         sizes=sizes, name="hbn")
    texts = {}
    a = 2.504
    for n, tags in energies.items():
        lat = [[a * n * np.sqrt(3) / 2, -a * n / 2, 0.0], [0.0, a * n, 0.0], [0.0, 0.0, 500.0]]
        vbm, cbm = HBN_VBM[n]
        texts[f"hbn{n}_pris"] = fake_output(energy=tags["pris"][0], electrons=tags["pris"][1],
                                            header=int(tags["pris"][1]), vbm=vbm, cbm=cbm, lattice=lat,
                                            atoms=[(7, "N", 0.0, 0.0, 0.0)])
        for tag in ("q0", "qp1", "qm1"):
            e, ne = tags[tag]
            texts[f"hbn{n}_{tag}"] = fake_output(energy=e, electrons=ne, header=101,
                                                 spin=1.0 if tag == "q0" else None)
    return manifest, texts


def test_hbn_extrapolated_levels():
    r = _report(_hbn_set())
    assert _level(r, "+1/0") == pytest.approx(3.8, abs=0.15)
    assert _level(r, "0/-1") == pytest.approx(6.6, abs=0.15)
    fit = r["fits"]["+1"]
    assert max(abs(x) for x in fit["residuals_ev"]) <= 0.02
    assert r["image_charge_correction"].startswith("A + B/L")
    assert any("no vacuum level" in n for n in r["notes"])
    assert r["pristine"]["vbm_spread_ev"] < 1e-5
    assert r["pristine"]["L_angstrom"][0] == pytest.approx(np.sqrt(3 * 3 * 2.504 ** 2 * np.sqrt(3) / 2))


def test_exact_synthetic_fit():
    A, B = 0.1, -0.6     # Ha, Ha*Angstrom
    L = [7.5, 10.0, 12.5, 15.0]
    y = [A + B / x for x in L]
    f = an.fit_inverse_size(L, y)
    assert f["A"] == pytest.approx(A) and f["B"] == pytest.approx(B)
    assert max(abs(r) for r in f["residuals"]) < 1e-12
    two = an.fit_inverse_size(L[:2], y[:2])
    assert two["A"] == pytest.approx(A)


def test_two_sizes_warn():
    energies = {n: HBN[n] for n in (4, 6)}
    r = _report(_hbn_set(energies))
    assert "warning" in r["fits"]["+1"] and "2 sizes" in r["fits"]["+1"]["warning"]
