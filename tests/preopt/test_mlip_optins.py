"""The 1.1.6 ML-model additions and their gates, without torch or network:
MACE-OFF (Academic Software License) as an acknowledged opt-in, Meta's UMA
(gated weights fetched with the user's own Hugging Face access, its own
environment, task heads), OrbMol-v2 (charge and spin from the structure's
real settings, molecules and periodic 3D cells only, no stress), and the
installer's environment and Python rules for them."""
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

import preopt_fixtures as F
from preopt_fixtures import preopt_sandbox  # noqa: E402,F401  (autouse: temp root, XDG dirs)
from test_mlip_registry_fetch import server, _Handler  # noqa: F401  (a local HTTP server)

CORPUS = Path("/mnt/iscsi/UsefulScripts/Codebase/reorganization/test")


def _off(base, body):
    from mace.preopt import registry
    m = copy.deepcopy(registry.get("acesuit-mace-off24-medium", scope=None))
    m["weights"].update(url=f"{base}/off.model", sha256=hashlib.sha256(body).hexdigest(), size_bytes=len(body),
                        license_text_url=f"{base}/ASL")
    m["code"]["license_text_url"] = f"{base}/CODE"
    return m


# --- MACE-OFF: academic, non-commercial use only, as an acknowledged opt-in ----------------------

def test_academic_fetch_needs_the_acknowledgment_and_records_it_with_a_date(server):
    from mace.preopt import detect, fetch
    base, files = server
    body = os.urandom(50_000)
    files.update({"/off.model": body, "/ASL": b"ASL text\n", "/CODE": b"MIT\n"})
    m = _off(base, body)
    out = []
    assert fetch.fetch(m, yes=True, out=out.append, isatty=False) == 2            # --yes alone is not enough
    assert "--accept-academic-license" in "".join(out) and detect.weights_status(m)["status"] == "needs_ack"
    assert not (detect.weights_root() / m["id"]).exists()
    out = []
    assert fetch.fetch(m, yes=True, out=out.append, isatty=True, accept_academic=True, ask=lambda q: "academic") == 2
    assert "Not confirmed" in "".join(out)                                        # a terminal: the word, exactly
    asked = []
    rc = fetch.fetch(m, yes=False, out=out.append, isatty=True, accept_academic=True,
                     ask=lambda q: (asked.append(q), "ACADEMIC")[1])
    assert rc == 0 and "academic, non-commercial work only" in asked[0]
    rec = json.loads((detect.weights_root() / m["id"] / "ACADEMIC.json").read_text())
    assert rec["id"] == m["id"] and rec["license"] == "ASL" and rec["how"] == "typed"
    assert rec["acknowledged_utc"].endswith("Z") and len(rec["acknowledged_utc"]) == 20
    assert detect.academic_ack(m)["how"] == "typed" and detect.weights_status(m)["status"] == "ready"
    # a script: the flag and --yes, recorded as such; an existing file is not downloaded again
    (detect.weights_root() / m["id"] / "ACADEMIC.json").unlink()
    assert detect.weights_status(m)["status"] == "needs_ack"
    files.pop("/off.model")
    out = []
    assert fetch.fetch(m, yes=True, out=out.append, isatty=False, accept_academic=True) == 0
    assert "already there" in "".join(out) and detect.academic_ack(m)["how"] == "flag"


def test_academic_results_are_tagged_and_listed_only_after_the_opt_in(monkeypatch):
    from mace.calculators import mlip, registry as creg
    from mace.preopt import detect, fetch, marker, registry
    m = registry.get("acesuit-mace-off24-medium", scope=None)
    assert registry.tag(m) == "MACE-OFF24 medium (ACEsuit mace-torch) (academic license)"
    assert registry.provenance_info(m)["label"] == "ML potential: MACE-OFF24 medium (ACEsuit mace-torch) (academic license)"
    assert m["id"] not in [x.id for x in creg.get("mlip").methods()]
    fetch.record_academic(m, "gui")
    meths = {x.id: x for x in creg.get("mlip").methods()}
    assert meths[m["id"]].label.endswith("(academic license)") and meths[m["id"]].extra["academic"]
    prov = creg.get("mlip").provenance(m["id"])
    assert prov["academic"] is True and "MACE-OFF (academic license)" in prov["license_note"]
    assert prov["academic_acknowledged_utc"]
    assert marker.line("mlip:x", m["id"], academic=True).endswith("license=academic")
    assert marker.parse(marker.line("mlip:x", m["id"], academic=True) + "\n")["license"] == "academic"
    assert registry.default_id() != m["id"] and m["id"] not in registry.ids() and m["id"] not in registry.ids("calc")


# --- UMA: gated weights, the user's own access, its own environment, task heads ---------------

def _uma():
    from mace.preopt import registry
    return registry.get("meta-uma-s-1p1", scope=None)


def test_gated_fetch_never_accepts_the_gate_and_says_where_to_ask(tmp_path, monkeypatch):
    from mace.preopt import detect, fetch
    m = _uma()
    fake = F.make_fake_python(detect.default_env_dir("fairchem"))
    assert detect.configured_python(m) == str(fake)
    calls = []
    out = []

    def denied(argv):
        calls.append(argv)
        return 'MACE_HF {"error": "gated", "kind": "GatedRepoError", "message": "401 Client Error"}\n'
    rc = fetch.fetch(m, yes=True, out=out.append, isatty=False, hf_runner=denied)
    text = "".join(out)
    assert rc == 1 and "https://huggingface.co/facebook/UMA" in text and "hf auth login" in text
    assert "never accepts them for you and never stores your token" in text
    assert calls[0][0] == str(fake) and calls[0][4:9] == ["facebook/UMA", "checkpoints/uma-s-1p1.pt",
                                                          m["weights"]["hf_revision"], "fairchem", "uma-s-1p1"]
    assert detect.weights_status(m)["status"] == "needs_fetch"
    # the user has access: the cached file is verified and the accept recorded (no token anywhere)
    blob = tmp_path / "hf" / "uma-s-1p1.pt"
    blob.parent.mkdir()
    blob.write_bytes(b"x" * 1000)
    small = copy.deepcopy(m)
    small["weights"].update(size_bytes=1000, sha256=hashlib.sha256(b"x" * 1000).hexdigest())
    rc = fetch.fetch(small, yes=True, out=out.append, isatty=False, consent_how="gui",
                     hf_runner=lambda argv: f'MACE_HF {json.dumps({"path": str(blob)})}\n')
    assert rc == 0
    rec = json.loads((detect.record_dir(small) / "ACCEPTED.json").read_text())
    assert rec["path"] == str(blob) and rec["how"] == "gui" and rec["token_stored"] is False
    assert "FAIR Chemistry License v1" in rec["terms"] and rec["accepted_utc"].endswith("Z")
    assert not any("token" in k.lower() and k != "token_stored" for k in rec)
    assert detect.weights_status(small)["status"] == "ready"
    blob.write_bytes(b"y" * 1000)                                    # the cache changed under it
    rec["sha256"] = "0" * 64
    (detect.record_dir(small) / "ACCEPTED.json").write_text(json.dumps(rec))
    assert detect.weights_status(small)["status"] == "bad_hash"


def test_uma_task_head_follows_the_dimensionality():
    pytest.importorskip("ase")
    from ase import Atoms
    from ase.build import bulk, molecule
    from mace.calculators import chargespin as CS
    m = _uma()
    assert CS.default_task(m, bulk("Si")) == "omat" and CS.default_task(m, molecule("H2O")) == "omol"
    slab = bulk("Si").repeat((1, 1, 2))
    slab.pbc = [True, True, False]
    assert CS.default_task(m, slab) == "omat"
    assert CS.needs(m, "omol") and not CS.needs(m, "omat")
    assert set(m["tasks"]) == {"omat", "omol", "oc20", "odac", "omc"}


def test_the_installer_keeps_fairchem_apart_and_pins_its_torch(tmp_path, monkeypatch):
    from mace.preopt import mlip_install as MI
    base, fenv = F.make_fake_base_python(tmp_path / "fakepy", tmp_path / "pip.log")
    for k, v in fenv.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(MI, "gpu_present", lambda: False)
    with pytest.raises(MI.InstallError) as exc:
        MI.make_plan(backends=["mace-torch", "fairchem-core"], python=str(base))
    assert "separate environments" in str(exc.value)
    plan = MI.make_plan(backends=["fairchem-core"], python=str(base))
    assert plan["env"] == "fairchem" and plan["dir"].endswith("mace-tool/mlip/venv-fairchem")
    assert plan["torch_spec"] == "torch==2.13.0" and "'torch==2.13.0' --index-url" in plan["commands"][3]
    assert plan["commands"][-1].endswith("--env fairchem")
    MI.install(plan, log=lambda s: None)
    from mace.preopt import detect
    assert detect.python_source("fairchem")[0] == str(Path(plan["dir"]) / "bin" / "python")
    assert detect.python_source() == (None, "")                     # the default environment is untouched
    calls = [json.loads(x) for x in (tmp_path / "pip.log").read_text().splitlines()]
    assert calls[1]["argv"][:2] == ["install", "torch==2.13.0"]
    old = tmp_path / "py311"
    old.write_text("#!/bin/sh\necho 3 11 3\n")
    old.chmod(0o755)
    plan = MI.make_plan(backends=["orb-models"], python=str(old))
    assert plan["problems"] and "3.12 or newer" in plan["problems"][0]


# --- OrbMol-v2 and UMA omol: the charge and spin from the structure's real settings -----------

def test_charge_and_spin_precedence_defaults_and_warnings():
    pytest.importorskip("ase")
    from ase.build import molecule
    from mace.calculators import chargespin as CS
    w = molecule("H2O")
    r = CS.resolve(w)
    assert (r["charge"], r["spin"]) == (0, 1) and r["charge_source"].startswith("assumed") and not r["warnings"]
    oh = molecule("OH")
    r = CS.resolve(oh)
    assert (r["charge"], r["spin"]) == (0, 2) and "odd number of electrons (9)" in r["warnings"][0]
    r = CS.resolve(oh, charge=-1)
    assert (r["charge"], r["spin"], r["charge_source"]) == (-1, 1, "given")
    with pytest.raises(ValueError) as e:
        CS.resolve(oh, multiplicity=1)
    assert "impossible with 9 electrons" in str(e.value)
    r = CS.resolve(w, file_info={"charge": 0, "uhf": 2, "what": "w.d12"})
    assert (r["spin"], r["spin_source"]) == (3, "from w.d12") and "open-shell" in r["warnings"][0]
    r = CS.resolve(w, uhf=0, file_info={"charge": 0, "uhf": 2, "what": "w.d12"})
    assert r["spin"] == 1 and r["spin_source"].startswith("given")


DECK = """water
MOLECULE
1
3
8 0.0 0.0 0.119
1 0.0 0.763 -0.477
1 0.0 -0.763 -0.477
BASISSET
POB-TZVP-REV2
DFT
B3LYP
ENDDFT
SPINLOCK
2 30
END
"""


def test_charge_and_spin_from_a_deck(tmp_path):
    from mace.calculators import chargespin as CS
    p = tmp_path / "w.d12"
    p.write_text(DECK)
    fi = CS.from_file(p)
    assert fi["charge"] == 0 and fi["uhf"] == 2
    p.write_text(DECK.replace("SPINLOCK\n2 30\n", "CHARGED\n"))
    fi = CS.from_file(p)
    assert fi["charge"] is None and fi["uhf"] is None and "states no net charge" in fi["notes"][0]
    assert CS.from_file(tmp_path / "x.cif") is None


@pytest.mark.skipif(not CORPUS.is_dir(), reason="the real CRYSTAL corpus (test/) is not here")
def test_charge_and_spin_from_real_outputs():
    pytest.importorskip("ase")
    from mace.calculators import chargespin as CS
    locked = CORPUS / "OPT" / "1_dia_opt_rev1.out"
    fi = CS.from_file(locked)
    assert fi["uhf"] == 0 and fi["charge"] == 0, fi          # ALPHA-BETA ELECTRONS LOCKED TO 0; 12 electrons, C2
    mol = next(iter(sorted((CORPUS / "BAND").glob("Suflolane_MOLECULE*optimized.out"))), None)
    if mol is not None:
        fi = CS.from_file(mol)
        assert fi["charge"] == 0 or fi["notes"], fi


def test_the_model_takes_charge_spin_and_task_per_call():
    pytest.importorskip("ase")
    from mace.preopt.mlip_client import MlipCalculator
    from ase.build import molecule
    sent = []

    class T:
        def request(self, msg, timeout, canceled):
            sent.append(msg)
            return {"ok": True, "energy": -1.0, "forces": [[0, 0, 0]] * len(msg["numbers"])}
    c = MlipCalculator(T())
    c.task, c.charge_spin = "omol", (-1, 2)
    a = molecule("H2O")
    a.calc = c
    a.get_potential_energy()
    assert sent[0]["task"] == "omol" and (sent[0]["charge"], sent[0]["spin"]) == (-1, 2)
    c2 = MlipCalculator(T())
    a.calc = c2
    a.get_potential_energy()
    assert "charge" not in sent[1] and "task" not in sent[1]                # other models: unchanged messages


def test_orbmol_runs_molecules_and_bulk_not_slabs_and_never_relaxes_a_cell(monkeypatch):
    pytest.importorskip("ase")
    from ase.build import bulk, molecule
    from mace.calculators import base, mlip, registry as creg
    from mace.preopt import detect, registry
    m = registry.get("orbmol-v2", scope=None)
    assert m["dims"] == [0, 3] and m["stress"] is False and m["charge_spin"] == ["*"]
    assert m["weights"]["sha256"] == "3dbef70b5fdc9124392181b0d8686c79f1c02bf5fefcb878a4eaa51d03e4f5e8"
    assert m["weights"]["license"] == m["code"]["license"] == "Apache-2.0" and not m.get("academic")
    be = creg.get("mlip")
    slab = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((1, 1, 2))
    slab.pbc = [True, True, False]
    with pytest.raises(base.CalcError) as e:
        be.calculator(slab, base.CalcSpec("mlip", method="orbmol-v2", task="sp", device="cpu"))
    assert "molecules (0D) and periodic 3D cells" in str(e.value) and "slabs (2D)" in str(e.value)
    with pytest.raises(base.CalcError) as e:
        be.calculator(bulk("NaCl", "rocksalt", a=5.64), base.CalcSpec("mlip", method="orbmol-v2", task="opt",
                                                                      relax_cell=True, device="cpu"))
    assert "predicts no stress" in str(e.value)
    with pytest.raises(base.CalcError) as e:                  # an impossible multiplicity is refused before loading
        be.calculator(molecule("OH"), base.CalcSpec("mlip", method="orbmol-v2", task="sp", device="cpu",
                                                    uhf=0, uhf_set=True))
    assert "impossible with 9 electrons" in str(e.value)
    # auto never picks it: the preopt tiers do not even see it
    assert registry.get("orbmol-v2") is None and "orbmol-v2" not in registry.ids()


def test_structio_names_the_source_file(tmp_path):
    pytest.importorskip("ase")
    from mace.calculators import structio
    p = tmp_path / "w.d12"
    p.write_text(DECK)
    a, _, _ = structio.read(p)
    assert a.info["source_path"] == str(p.resolve()) and list(a.pbc) == [False, False, False]
