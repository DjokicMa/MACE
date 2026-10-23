"""ML interatomic potentials: MACE-MP (mace-torch), ORB-v3, MatterSim, CHGNet.

The models are the calc-scope entries of ``mace/preopt/mlip_registry.json``
(licenses pinned and checked in code, every one allowing commercial use;
weights sha256-verified).
They run exactly as Clean's ML tier runs them: in the ML environment's own
interpreter (``mace mlip setup``), through ``mace/preopt/mlip_worker.py``
and ``mlip_client``, so this MACE never imports torch, and ACEsuit's
``mace`` package (mace-torch, the MACE-MP potentials) never
meets this MACE tool, which is unrelated to it, in one interpreter.

Device: ``auto`` (the default) gives the worker one GPU, GPU 0 unless
CUDA_VISIBLE_DEVICES already says otherwise, and the worker takes it when
torch sees it, else the CPU; ``cpu`` and ``cuda:N`` as in ``mace preopt``.
The worker may use ``gpu_mem_fraction`` of that GPU's memory (default 0.5).

A model file the user already has (``mace calc --model-file``, e.g. their own
MACE-OFF23) runs through the same worker. MACE never lists, downloads, copies
or defaults to such a file: ``user_model`` describes it for this one run, read
where it is, and its license is the user's to follow.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from . import base

FAMILY_LABELS = {
    "MACE-MP": "MACE-MP (ACEsuit mace-torch)",
    "MACE-OFF (ACEsuit mace-torch)": "MACE-OFF (ACEsuit mace-torch, academic license)",
    "UMA (Meta fairchem)": "UMA (Meta fairchem)",
    "OrbMol": "OrbMol (orb-models)",
    "ORB-v3": "ORB-v3 (orb-models)",
    "MatterSim": "MatterSim (mattersim)",
    "CHGNet": "CHGNet (chgnet)",
}
_BACKEND_FAMILY = {"mace_mp": "MACE-MP", "orb": "ORB-v3", "mattersim": "MatterSim", "chgnet": "CHGNet",
                   "fairchem": "UMA (Meta fairchem)"}
GPU_MEM_FRACTION = 0.5
# --model-backend -> (worker backend, package, the package's code license); packages that load a local file
USER_BACKENDS = {"mace-torch": ("mace_mp", "mace-torch", "MIT"), "mattersim": ("mattersim", "mattersim", "MIT")}
USER_LICENSE = "user-supplied: its license terms are the user's responsibility"
_USER: Dict[str, Dict[str, Any]] = {}


def user_model(path, backend: str) -> Dict[str, Any]:
    """A model file the user supplies, as a registry-shaped entry for this run
    only (id ``user:<file name>``). The file is read where it is: never
    copied, downloaded or listed; its sha256 is recorded in the result."""
    from pathlib import Path
    from mace.preopt import detect
    if backend not in USER_BACKENDS:
        raise base.CalcError(f"--model-backend is one of {', '.join(USER_BACKENDS)}, not {backend!r}", kind="usage")
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise base.CalcError(f"--model-file {path}: no such file", kind="usage")
    worker, package, code_license = USER_BACKENDS[backend]
    mid = "user:" + p.name
    _USER[mid] = {"id": mid, "label": f"user-supplied model {p.name}", "backend": worker, "family": "user-supplied",
                  "code": {"package": package, "license": code_license, "pin": package},
                  "weights": {"license": USER_LICENSE, "user_file": str(p), "sha256": detect.sha256_file(p)},
                  "defaults": {"dtype": "float64"} if worker == "mace_mp" else {},
                  "unrelated_note": "a model file supplied by the user, not one MACE offers"}
    return _USER[mid]


def family(m: Dict[str, Any]) -> str:
    return m.get("family") or _BACKEND_FAMILY.get(m.get("backend"), m.get("backend") or "?")


def models() -> List[Dict[str, Any]]:
    """The calc-scope models; an academic-only one only once the user has
    acknowledged academic use for it (it is never offered before)."""
    from mace.preopt import detect, registry
    return [m for m in registry.load(scope="calc", include_academic=True)
            if not m.get("academic") or detect.academic_ack(m) is not None]


def model(mid: str) -> Optional[Dict[str, Any]]:
    if mid in _USER:
        return _USER[mid]
    from mace.preopt import registry
    return registry.get(mid, scope="calc")


class MLIPBackend(base.Backend):
    name = "mlip"
    label = "ML potentials (MACE-MP via mace-torch, ORB-v3, OrbMol-v2, MatterSim, CHGNet, UMA)"
    family = "mlip"
    license = "per model (see each model's code and weights licenses)"
    capabilities = base.Capabilities(frozenset({"sp", "opt", "freq", "md"}), frozenset({0, 1, 2, 3}),
                                     frozenset({"energy", "forces", "stress"}))

    def methods(self) -> List[base.Method]:
        from mace.preopt import registry
        out = []
        for m in models():
            w = m["weights"]
            info = registry.provenance_info(m)
            out.append(base.Method(
                m["id"], registry.tag(m), default=bool(m.get("default")), note=m.get("unrelated_note", ""),
                license=f"code {m['code']['license']}, weights {w['license']}",
                citation=list(m.get("citation") or []),
                extra={"family": family(m), "family_label": FAMILY_LABELS.get(family(m), family(m)),
                       "limits": m.get("limits"), "training_data": info["training_data"],
                       "data_citation": info["data_citation"], "package": m["code"]["package"],
                       "pin": m["code"]["pin"],
                       "size_mb": round(w["size_bytes"] / 1e6, 1) if "size_bytes" in w else 0.0,
                       "elements": m.get("elements"), "academic": bool(m.get("academic")),
                       "gated": bool(m.get("gated")), "tasks": m.get("tasks"), "task_default": m.get("task_default"),
                       "charge_spin": m.get("charge_spin") or [], "dims": m.get("dims"), "gpu_dims": m.get("gpu_dims"),
                       "role": m.get("role")}))
        return out

    def method(self, mid: Optional[str]) -> base.Method:
        from mace.preopt import registry
        if mid in registry.REMOVED:
            raise base.CalcError(registry.REMOVED[mid], kind="usage")
        m = registry.get(mid, scope="calc") if mid else None
        if m is not None and m.get("academic"):
            from mace.preopt import detect
            if detect.academic_ack(m) is None:   # never run, nor listed, before the user acknowledged academic use
                raise base.CalcError(detect.weights_status(m)["reason"], kind="usage")
        if mid in _USER:
            m = _USER[mid]
            return base.Method(mid, m["label"], license=f"code {m['code']['license']}, weights {USER_LICENSE}",
                               note=m["unrelated_note"], extra={"family": "user-supplied",
                                                                "model_file": m["weights"]["user_file"]})
        return super().method(mid)

    def status(self, method: Optional[str] = None, recheck: bool = False, options=None) -> Dict[str, Any]:
        from mace.preopt import detect, registry
        m = model(method or self.default_method())
        if m is None:
            return {"available": False, "status": "unknown", "reason": registry.unknown(method, scope="calc"),
                    "hint": ""}
        try:
            st = detect.mlip_status(m, recheck=recheck)
        except Exception as exc:
            st = {"status": "import_failed", "reason": str(exc), "hint": detect.setup_hint(m)}
        ok = st["status"] == "ready"
        return {"available": ok, "status": st["status"], "reason": st.get("reason", ""),
                "hint": st.get("hint", "") if not ok else "",
                "commands": detect.hint_commands(st.get("hint", "")) if not ok else [],
                "version": (st.get("versions") or {}).get(m["code"]["package"]),
                "versions": st.get("versions") or {}, "python": st.get("python")}

    def calculator(self, atoms, spec: base.CalcSpec, canceled: Optional[Callable[[], bool]] = None):
        from mace.preopt import mlip_client, proc
        m = model(self.method(spec.method).id)
        o = spec.options or {}
        allowed = m.get("elements")
        if allowed:
            from ase.data import chemical_symbols
            bad = sorted({int(z) for z in atoms.numbers} - set(allowed))
            if bad:
                raise base.CalcError(f"{m['label']} has no parameters for "
                                     + ", ".join(chemical_symbols[z] for z in bad)
                                     + " (it covers " + ", ".join(chemical_symbols[z] for z in allowed) + ")",
                                     kind="elements")
        try:
            _, wdev, _ = mlip_client.device_mapping(spec.device)
        except ValueError as exc:
            raise base.CalcError(str(exc), kind="usage")
        dims = int(sum(bool(x) for x in atoms.pbc))
        ok_dims = m.get("dims")
        if ok_dims is not None and dims not in ok_dims:
            names = {0: "molecules (0D)", 1: "polymers (1D)", 2: "slabs (2D)", 3: "periodic 3D cells"}
            raise base.CalcError(f"{m['label']} runs " + " and ".join(names[d] for d in sorted(ok_dims))
                                 + f"; this structure is one of the {names[dims]}" + (f" ({m['limits']})" if m.get("limits") else ""),
                                 kind="usage")
        if dims in (m.get("gpu_dims") or []):
            import shutil
            if wdev == "cpu" or (spec.device == "auto" and not shutil.which("nvidia-smi")):
                raise base.CalcError(f"{m['label']} on a periodic cell needs a GPU (Particle Mesh Ewald through "
                                     "nvalchemiops); run it on a GPU node (--device cuda:0), or use it for molecules",
                                     kind="usage")
        if m.get("stress") is False and (spec.relax_cell or (spec.task == "md" and spec.ensemble == "npt")):
            raise base.CalcError(f"{m['label']} predicts no stress: no cell relaxation or NPT (positions only, NVT, NVE)",
                                 kind="usage")
        task = None
        if m.get("tasks"):
            from . import chargespin as CS
            task = o.get("task_head") or CS.default_task(m, atoms)
            if task not in m["tasks"]:
                raise base.CalcError(f"{m['label']}: the task head is one of {', '.join(m['tasks'])}, not {task!r}",
                                     kind="usage")
        cs = None
        from . import chargespin as CS
        if CS.needs(m, task):
            try:
                cs = CS.resolve(atoms, charge=spec.charge if spec.charge_set else None,
                                uhf=spec.uhf if spec.uhf_set else None,
                                file_info=CS.from_file(atoms.info.get("source_path")))
            except ValueError as exc:
                raise base.CalcError(f"{m['label']}: {exc}", kind="usage")
        threads = int(spec.threads or proc.default_threads())
        frac = o.get("gpu_mem_fraction")
        try:
            calc = mlip_client.make_calculator(
                m, threads=threads, device=spec.device, lease=o.get("_lease"), canceled=canceled,
                elements={int(z) for z in atoms.numbers}, dtype=o.get("dtype"),
                gpu_mem_fraction=float(frac) if frac not in (None, "") else GPU_MEM_FRACTION)
        except mlip_client.MlipError as exc:
            raise base.CalcError(f"{m['label']}: {exc.kind}: {exc}", kind=exc.kind)
        except proc.Canceled:
            raise base.Canceled("canceled while the ML model loaded")
        from mace.preopt import registry
        calc.task = task
        if cs is not None:
            calc.charge_spin = (cs["charge"], cs["spin"])
        hello = calc.info.get("hello") or {}
        calc.mace_info = {"model": m["id"], "model_label": registry.tag(m), "family": FAMILY_LABELS.get(family(m)),
                          "device_text": calc.info.get("device") or mlip_client.device_mapping(spec.device)[2],
                          "device": hello.get("device"), "versions": hello.get("versions"),
                          "load_s": hello.get("load_s"), "threads": threads,
                          "weights_sha256": m["weights"].get("sha256"),
                          "limits": m.get("limits"), "unrelated_note": m.get("unrelated_note")}
        if m.get("academic"):
            calc.mace_info["license_note"] = "MACE-OFF (academic license): academic, non-commercial use only"
            calc.mace_info["academic"] = True
        if task:
            calc.mace_info["task_head"] = task
        if cs is not None:
            calc.mace_info.update(charge=cs["charge"], spin_multiplicity=cs["spin"], charge_source=cs["charge_source"],
                                  spin_source=cs["spin_source"], charge_spin_warnings=cs["warnings"])
        from mace.preopt import lowdim
        if lowdim.molecule_as_is(m, atoms):
            return calc                       # a molecule as a molecule (pbc off): mace-torch, ORB, OrbMol-v2, UMA
        if lowdim.is_low_dim(atoms):
            # a slab or polymer with its pbc, or a molecule for CHGNet or MatterSim (they always build a 3D
            # crystal): the model sees a copy boxed with lowdim.VACUUM Å along the open axes
            calc.mace_info["boxed_vacuum_A"] = lowdim.VACUUM
            return lowdim.Boxed(calc)
        return calc

    @staticmethod
    def report(calc) -> Dict[str, Any]:
        out = dict(getattr(calc, "mace_info", {}) or {})
        out["worker_calls"] = getattr(calc, "calls", None)
        return out

    def provenance(self, method: Optional[str] = None) -> Dict[str, Any]:
        from mace.preopt import registry
        p = super().provenance(method)
        m = model(self.method(method).id)
        info = registry.provenance_info(m)
        p.update(model=m["id"], model_label=registry.tag(m), family=FAMILY_LABELS.get(family(m)),
                 code=info["code"], code_pin=info["version"], code_license=info["code_license"],
                 weights_license=info["weights_license"], weights_sha256=info["weights_sha256"],
                 training_data=info["training_data"], data_citation=info["data_citation"],
                 limits=info["limits"], unrelated_note=info["unrelated_note"])
        if m["id"] in _USER:
            p.update(user_supplied=True, model_file=m["weights"]["user_file"])
        if m.get("academic"):
            ack = __import__("mace.preopt.detect", fromlist=["x"]).academic_ack(m) or {}
            p.update(academic=True, license_note="MACE-OFF (academic license): academic, non-commercial use only",
                     academic_acknowledged_utc=ack.get("acknowledged_utc"))
        return p
