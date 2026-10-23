"""Routes of Calculation Setup > 4 ELASTIC ... 7 CPHF (elastic/generate), of
Analysis > Mechanical Properties Results (ELASTCON & co., EOS) and of the
Relaxation Time dialog (transport tau).

GET  elastic/modes                     the modules (QHA included), their manual pages, sub-keywords
                                       and defaults (d12_calc_elastic.MODES)
GET  elastic/analyze?path=REL          the tensors, moduli and EOS fits of an output
                                       (``mace analyze elastic --json``) + its text report
GET  elastic/document?path=REL[&epsilon=E]
                                       the results document of an ELASTIC, EOS, QHA or CPHF output
                                       (mace.elastic.views.document): analyze() plus the run's kind,
                                       its key settings, its plots and the CLI line; epsilon: the
                                       permittivity of the piezoelectric coupling (--epsilon)
GET  elastic/figure?path=REL&view=K&shape=surface|planes[&compare=REL ...][&fit=F]
                                       one plot's arrays (views.figure; mace analyze elastic --plot K);
                                       compare: other phases' EOS outputs for view=enthalpy (--compare)
POST elastic/generate {parent, d12?, settings, output_dir?, root_gen}
                                       -> task: mace opt2d12 --calc-type ELASTIC|EOS|QHA|CPHF
                                       --non-interactive --elastic-settings JSON
POST tau/estimate {dim, m_star, m_other?, e1, c, temperature?} or {mobility, m_star}
                                       tau with its provenance (``mace tau estimate``)
POST tau/strain {parent, axis, strains, output_dir?, force?, root_gen}
                                       the strained SP decks (``mace tau strain``)
GET  tau/series?dir=REL                the strain series (<stem>_dp_<axis>_*.out) in a folder
POST tau/fit {outputs, carrier, m_star? (a number, or "auto"/empty), band_dat?, dos_file?, m_other?,
              c_from?, e1?, temperature?, save?, save_db?, root_gen?}
                                       E1 and C from the outputs, then tau (``mace tau fit``);
                                       with auto, "masses" says which masses were used and from where;
                                       save_db: Save to Database (mace/properties relaxation_time_dpt,
                                       stored with the unstrained run: "saved_db")
"""
from __future__ import annotations

import json
import math
import os
import re
import shlex
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List

from . import clirun
from . import tasks as gui_tasks

GET_ROUTES = {"elastic/modes", "elastic/analyze", "elastic/document", "elastic/figure", "tau/series"}
POST_ROUTES = {"elastic/generate", "tau/estimate", "tau/strain", "tau/fit"}

_REPO = Path(__file__).resolve().parents[2]


def _q(query, name: str, default: Any = None):
    return (query.get(name) or [default])[0]


def _gen(handler, body) -> None:
    want = body.get("root_gen", handler.headers.get("X-Mace-Root-Gen"))
    if want is None:
        raise ValueError("root_gen is required (reload the page)")
    if str(want) != str(getattr(handler.state, "root_gen", 0)):
        from .submit import SubmitConflict
        raise SubmitConflict("The folder changed; reload")


def _elastic_module():
    cd = str(_REPO / "Crystal_d12")
    if cd not in sys.path:
        sys.path.insert(0, cd)
    import d12_calc_elastic
    return d12_calc_elastic


def modes() -> Dict[str, Any]:
    E = _elastic_module()
    return {"modes": [{"mode": k, "title": v["title"], "what": v["what"], "manual": v["manual"],
                       "step": v["step"], "dims": list(v["dims"]), "pcrystal_only": v["pcrystal_only"],
                       "keys": list(v["keys"])} for k, v in E.MODES.items()],
            "defaults": {k: (list(v) if isinstance(v, tuple) else v) for k, v in E.DEFAULTS.items()},
            "module_toldee": E.MODULE_TOLDEE}


def analyze(path: Path) -> Dict[str, Any]:
    from mace.elastic import analysis as A
    data = A.analyze_file(path)
    return {"path": str(path.name), "data": data, "report": A.report(data, path.name), "empty": not data}


def _document(st, path: Path, epsilon: str = "") -> Dict[str, Any]:
    """views.document of an output, through the server's parse cache (keyed by the file's stamp
    and the permittivity given for the piezoelectric coupling)."""
    from . import cache as gui_cache
    from mace.elastic import views as V
    key = gui_cache.file_key(path, ("elasticdoc", 2, epsilon))
    hit = st.parse_cache.get(key)
    if hit is None:
        text = path.read_text(errors="replace")
        hit = (V.document(text, path, epsilon or None), text)
        st.parse_cache.put(key, hit, 2 * len(text) + 65536)
    return hit


def _epsilon(value) -> str:
    """The permittivity of the query as the CLI's --epsilon: one, three or nine numbers (else a 400)."""
    text = " ".join(str(value or "").replace(",", " ").split())
    if not text:
        return ""
    from mace.elastic import tensor as T
    T.epsilon_tensor(text)                       # ValueError: the route answers 400
    return text


def document(st, path: Path, epsilon: str = "") -> Dict[str, Any]:
    eps = _epsilon(epsilon)
    doc, _ = _document(st, path, eps)
    # run in the output's folder, as Copy CLI Command of the other plots
    cli = f"mace analyze elastic {shlex.quote(path.name)}" + (f" --epsilon {shlex.quote(eps)}" if eps else "")
    return {**doc, "path": st.rel(path), "cli": cli}


def figure(st, path: Path, view: str, shape: str, compare: List[str] = (), fit: str = "") -> Dict[str, Any]:
    """One plot's arrays; ``compare``: other EOS outputs (paths under the root) for the enthalpy plot."""
    from mace.elastic import views as V
    doc, text = _document(st, path)
    others = []
    for rel in compare or []:
        q = st.resolve(rel)
        if not q.is_file():
            raise ValueError(f"{rel} is not a file (choose another phase's EOS output)")
        others.append((q, _document(st, q)[0]["data"]))
    if others and view != "enthalpy":
        raise ValueError("compare is the enthalpy plot's (another phase's EOS output)")
    fig = V.figure(doc["data"], view, text, shape or "surface", [(q.name, d) for q, d in others], fit or None, path.name)
    cli = f"mace analyze elastic {shlex.quote(path.name)} --plot {view}"
    if view in V.SHAPED and fig.get("dim") == 3:
        cli += f" --shape {fig['shape']}"
    for q, _ in others:
        # the CLI runs in this output's folder: the other phase relative to it
        cli += f" --compare {shlex.quote(os.path.relpath(q, path.parent))}"
    if fit:
        cli += f" --fit {shlex.quote(fit)}"
    return {**fig, "path": st.rel(path), "cli": cli, "table": V.table(fig),
            "compare": [st.rel(q) for q, _ in others]}


def _shown(argv: List[str], root: Path) -> str:
    out = []
    for a in argv[1:]:
        if os.path.isabs(a):
            try:
                a = os.path.relpath(a, root)
            except ValueError:
                pass
        out.append(shlex.quote(a))
    return "mace " + " ".join(out)


def generate_argv(st, body) -> (List[str], Path):
    parent = st.resolve(str(body.get("parent") or ""))
    if parent.suffix.lower() != ".out":
        raise ValueError("the parent is the CRYSTAL output (.out) of the optimized structure")
    settings = body.get("settings") or {}
    if not isinstance(settings, dict):
        raise ValueError("settings is an object of the module's sub-keywords")
    E = _elastic_module()
    mode = str(settings.get("mode") or "ELASTCON").upper()
    calc = mode if mode in E.OWN_TYPES else "ELASTIC"
    norm = E.normalize(settings, calc)              # ElasticSettingsError is a ValueError: 400
    argv = ["--no-banner", "opt2d12", "--out-file", str(parent), "--calc-type", calc, "--non-interactive",
            "--elastic-settings", json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in norm.items()})]
    if body.get("d12"):
        argv += ["--d12-file", str(st.resolve(str(body["d12"])))]
    if body.get("output_dir"):
        od = st.out_dir(parent.parent / str(body["output_dir"]))
        argv += ["--output-dir", str(od)]
    return argv, parent.parent


def estimate(body) -> Dict[str, Any]:
    from mace.transport import dpt

    def num(k, required=True):
        v = body.get(k)
        if v in (None, ""):
            if required:
                raise ValueError(f"{k} is required")
            return None
        return float(v)
    if body.get("mobility") not in (None, ""):
        res = dpt.estimate_from_mobility(num("mobility"), num("m_star"),
                                         {"mobility": str(body.get("mobility_source") or "given"),
                                          "m_star": str(body.get("m_star_source") or "given")})
    else:
        dim = int(body.get("dim") or 3)
        res = dpt.estimate(dim, num("m_star"), num("e1"), num("c"), num("temperature", False) or 300.0,
                           num("m_other", False),
                           {k: str(body.get(f"{k.lower()}_source") or "given") for k in ("m_star", "E1", "C")})
    return {"result": res, "report": dpt.report(res)}


def series(st, base: Path) -> Dict[str, Any]:
    """The strain series in ``base``: {stem_axis: [rel .out ...]} of <stem>_dp_<axis>_*.out."""
    from .filekind import rel_inside
    groups: Dict[str, List[str]] = {}
    for p in sorted(base.glob("*_dp_*_*pct.out")):
        m = re.match(r"(.+_dp_[xyz])_[pm][\d_]+pct\.out$", p.name)
        rel = rel_inside(st, p) if m else None       # a symlink out of the folder is not listed
        if rel is not None:
            groups.setdefault(m.group(1), []).append(rel)
    return {"dir": st.rel(base), "series": [{"name": k, "outputs": v} for k, v in groups.items()]}


def fit(st, body) -> Dict[str, Any]:
    from mace.transport import dpt, strain
    from mace.transport import cli as tcli
    outs = body.get("outputs") or []
    if not isinstance(outs, list) or len(outs) < 3:
        raise ValueError("outputs: the strain series (the unstrained run and at least two strained ones)")
    paths = [st.resolve(str(o)) for o in outs]
    for o, p in zip(outs, paths):
        if not p.is_file():
            raise ValueError(f"{o} is not a file (the outputs are the strained runs' .out files)")
    carrier = str(body.get("carrier") or "")
    points = [strain.read_point(p) for p in paths]
    ft = strain.fit(points, carrier)
    sources = {"E1": f"linear fit of the {'CBM' if carrier == 'electron' else 'VBM'} over {len(ft['strains'])} "
                     f"strained SP runs ({ft['axis']})",
               "C": "quadratic fit of the total energy over the same runs (clamped ions)"}
    e1, c = ft["E1_eV"], ft["C"]
    if body.get("e1") not in (None, ""):
        e1, sources["E1"] = float(body["e1"]), "given"
    if body.get("c_from"):
        if ft["dimensionality"] != 3:
            raise ValueError("an ELASTCON C is read for 3D runs (GPa)")
        cf = st.resolve(str(body["c_from"]))
        from mace.elastic.analysis import analyze_file
        el = analyze_file(cf).get("elastic")
        if not el or not str(el.get("unit", "")).upper().startswith("GPA"):
            raise ValueError(f"{cf.name} has no 3D elastic constants in GPa")
        k = "xyz".index(ft["axis"])
        c, sources["C"] = float(el["matrix"][k][k]), f"C_{ft['axis'] * 2} of {cf.name} (ELASTCON)"
    # m_star: a number, or "auto"/empty: from band_dat/dos_file or the BAND/DOSS outputs beside the series
    band = st.resolve(str(body["band_dat"])) if body.get("band_dat") else None
    dos = st.resolve(str(body["dos_file"])) if body.get("dos_file") else None
    m, md, msrc, am = tcli.resolve_masses(body.get("m_star"), carrier, ft["axis"], ft["dimensionality"], paths,
                                          band, dos, inside=st.root)
    sources.update({k: ("given" if v.startswith("given") else v) for k, v in msrc.items()})
    m_other = float(body["m_other"]) if body.get("m_other") not in (None, "") else None
    res = dpt.estimate(ft["dimensionality"], m, e1, c, float(body.get("temperature") or 300.0), m_other,
                       sources, carrier=carrier, axis=ft["axis"], m_dos=md)
    res["fit"] = ft
    out = {"result": res, "report": dpt.report(res)}
    if am:
        from mace.transport import mass
        res["masses"] = am
        out["masses"] = {"m_star": am["m_star"], "m_dos": am["m_dos"], "sources": am["sources"],
                         "warnings": am["warnings"], "report": mass.report(am)}
    if body.get("save"):
        dest = paths[0].parent / f"tau_{carrier}_{ft['axis']}.json"
        st.check_write(dest)
        dest.write_text(json.dumps(res, indent=2) + "\n")
        out["saved"] = st.rel(dest)
    if body.get("save_db"):
        # Save to Database: the registry's interactive relaxation_time_dpt, with the
        # unstrained run (the series' reference; the strained runs are its references)
        from mace.properties import registry
        from . import db as gui_db, dbwriters, properties_routes
        run = next((p for p, pt in zip(paths, points) if not pt.get("axis") or not pt.get("strain")), paths[0])
        try:
            vals = registry.compute_interactive("relaxation_time_dpt", run, {"result": res},
                                                {"strained_runs": [p for p in paths if p != run]})
            out["saved_db"] = properties_routes.save_interactive(st, run, "relaxation_time_dpt", vals, "Relaxation time")
        except (ValueError, PermissionError, registry.Skip, gui_db.DbBusy, dbwriters.WriterBusy) as exc:
            # the fit stands; the save says why it did not happen
            out["saved_db"] = {"error": str(exc)}
        except sqlite3.Error as exc:
            # a locked, read-only or damaged database (or its migration) fails the save, never the fit
            out["saved_db"] = {"error": f"The database could not be written ({exc}); the fit is kept, "
                                        f"Save to Database again when it is free"}
    return out


def get(handler, route: str, query):
    st = handler.state
    if route == "elastic/modes":
        return handler._json(modes())
    if route == "elastic/analyze":
        p = st.resolve(_q(query, "path", ""))
        if not p.is_file():
            raise ValueError(f"{_q(query, 'path', '')} is not a file (choose a CRYSTAL output)")
        return handler._json(analyze(p))
    if route in ("elastic/document", "elastic/figure"):
        p = st.resolve(_q(query, "path", ""))
        if not p.is_file():
            raise ValueError(f"{_q(query, 'path', '')} is not a file (choose a CRYSTAL output)")
        if route == "elastic/document":
            return handler._json(document(st, p, str(_q(query, "epsilon", "") or "")))
        return handler._json(figure(st, p, str(_q(query, "view", "")), str(_q(query, "shape", "") or "surface"),
                                    [str(c) for c in (query.get("compare") or []) if c], str(_q(query, "fit", "") or "")))
    if route == "tau/series":
        base = st.resolve(_q(query, "dir", "") or ".")
        return handler._json(series(st, base.parent if base.is_file() else base))
    return handler._error(404, f"unknown API route {route}")


def post(handler, route: str, body):
    st = handler.state
    if route == "tau/estimate":
        return handler._json(estimate(body))
    if route == "tau/fit":
        if body.get("save") or body.get("save_db"):
            _gen(handler, body)
        return handler._json(fit(st, body))
    _gen(handler, body)
    if route == "tau/strain":
        from mace.transport import strain
        parent = st.resolve(str(body.get("parent") or ""))
        if parent.suffix.lower() != ".d12":
            raise ValueError("the parent is the SP deck (.d12) of the optimized structure")
        raw = body.get("strains") or "-0.01,-0.005,0.005,0.01"
        try:
            strains = [float(x) for x in (raw.split(",") if isinstance(raw, str) else raw) if str(x).strip()]
        except (TypeError, ValueError):
            raise ValueError("strains are numbers, comma-separated (for example -0.01,-0.005,0.005,0.01)")
        if not all(math.isfinite(x) for x in strains):
            raise ValueError("strains are finite numbers")
        od = st.out_dir(parent.parent / str(body["output_dir"]) if body.get("output_dir") else parent.parent)
        written = strain.write_strain_set(parent, str(body.get("axis") or "x"), strains, od, bool(body.get("force")),
                                          check=st.check_write)
        return handler._json({"written": [st.rel(p) for p in written],
                              "command": f"mace tau strain {shlex.quote(st.rel(parent))} --axis {body.get('axis') or 'x'} "
                                         f"--strains {','.join(str(s) for s in strains)}"})
    if route == "elastic/generate":
        from .server import cli_task
        argv, cwd = generate_argv(st, body)

        def job(ctx):
            if ctx is not None:
                ctx.progress(0.1, _shown(argv, st.root))
            r = clirun.run(clirun.mace_argv(*argv), cwd, ctx, timeout=300)
            log = r.log.replace(str(st.root) + os.sep, "")
            m = re.search(r"Successfully created (\S+\.d12)", log)
            deck = None
            if m:
                cand = (cwd / m.group(1)) if not os.path.isabs(m.group(1)) else Path(m.group(1))
                if not cand.is_file() and body.get("output_dir"):
                    cand = cwd / str(body["output_dir"]) / Path(m.group(1)).name
                deck = st.rel(cand) if cand.is_file() else None
            return {"code": r.returncode, "log": log[-20000:], "command": _shown(argv, st.root), "deck": deck}
        t = cli_task(st, "elastic", "Mechanical properties deck (mace opt2d12)", job,
                     doc="elastic", client=handler.headers.get("X-Mace-Client"), supersede=("elastic", "generate"))
        return gui_tasks.answer(handler, st.tasks, t, gui_tasks.clamp_wait(body.get("wait", 5)))
    return handler._error(404, f"unknown API route {route}")


def handle(handler, method: str, route: str, payload):
    from .submit import SubmitConflict
    from mace.transport.dpt import TauError
    try:
        if method == "GET" and route in GET_ROUTES:
            return get(handler, route, payload)
        if method == "POST" and route in POST_ROUTES:
            return post(handler, route, payload or {})
        return handler._error(404, f"unknown API route {route}")
    except SubmitConflict as exc:
        return handler._json({"error": str(exc), "conflict": True}, 409)
    except PermissionError as exc:
        return handler._error(403, str(exc))
    except FileNotFoundError as exc:
        return handler._error(404, str(exc))
    except (ValueError, TauError) as exc:
        return handler._error(400, str(exc))
