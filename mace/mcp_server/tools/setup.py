"""Deck tools: templates, D12/D3 deck previews and writes, preflight.

A preview is the GUI Setup's preview (mace.gui.setup.stage + run_printed):
the real CLI (``mace convert`` / ``opt2d12`` / ``opt2d3``) runs on a scratch
copy with the options file, and its deck comes back byte for byte. A write
runs the same command on the real files with the output folder given;
replacing an existing deck needs confirm (the old one goes to File History).
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .. import resources

D3_CALCS = ["BAND", "DOSS", "TRANSPORT", "CHARGE", "POTENTIAL", "CHARGE+POTENTIAL", "ORBITALS"]
DECK_TEXT_MAX = 200_000
_CHARGED = {"type": "object", "description": (
    "charged systems (D12 only), the CLI's switches: ion_charges ('auto'), net_charge, site_states, chemod "
    "('auto'|'yes'|'no'), fragment_bond_scale, spinlock, spinlock_cycles")}


def register(reg) -> None:
    reg.tool("setup_templates", (
        "The shipped option templates: D12 options for new calculations and follow-ups (OPT, SP, FREQ, 3c "
        "composites, slabs, molecules...) and D3 configurations (BAND, DOSS, TRANSPORT, CHARGE...). Read one in "
        "full as the resource mace://templates/<d12|d3>/<name>."),
        {"properties": {"kind": {"type": "string", "enum": ["d12", "d3", "all"], "default": "all"},
                        "full": {"type": "boolean", "default": False,
                                 "description": "include each template's JSON"}}},
        title="Deck templates", group="setup")(setup_templates)

    d12_props = {"structure": {"type": "string", "description": (
                     "a .cif (new calculation, `mace convert`), or a CRYSTAL .out or .d12 (follow-up, `mace opt2d12`)")},
                 "template": {"type": "string", "description": "a D12 template name (setup_templates)"},
                 "options": {"type": "object", "description": "a full D12 options object (instead of template)"},
                 "charged": _CHARGED}
    reg.tool("setup_d12_preview", (
        "Preview the CRYSTAL .d12 deck MACE writes for a structure with a template or options object (the "
        "real CLI on a scratch copy; nothing is written). Defaults follow the parent run for follow-ups."),
        {"properties": d12_props, "required": ["structure"]},
        title="Preview a D12 deck", group="setup")(setup_d12_preview)
    reg.tool("setup_d12_write", (
        "Write the .d12 deck(s) for a structure into output_dir (default: the structure's folder) with the "
        "real CLI. Replacing an existing deck needs confirm."),
        {"properties": {**d12_props, "output_dir": {"type": "string"}}, "required": ["structure"]},
        title="Write a D12 deck", group="setup", writes=True, confirm="on_overwrite")(setup_d12_write)

    d3_props = {"output_file": {"type": "string", "description": "the finished CRYSTAL .out (its .f9 beside it)"},
                "calc_type": {"type": "string", "enum": D3_CALCS,
                              "description": "default: the template's calculation type"},
                "template": {"type": "string", "description": "a D3 template name (setup_templates)"},
                "config": {"type": "object", "description": "a full D3 configuration object (instead of template)"}}
    reg.tool("setup_d3_preview", (
        "Preview the .d3 properties deck (BAND, DOSS, TRANSPORT, CHARGE, POTENTIAL, ORBITALS) for a finished "
        "run with a template or configuration (the real `mace opt2d3` on a scratch copy; nothing is written)."),
        {"properties": d3_props, "required": ["output_file"]},
        title="Preview a D3 deck", group="setup")(setup_d3_preview)
    reg.tool("setup_d3_write", (
        "Write the .d3 deck for a finished run into output_dir (default: the run's folder) with `mace opt2d3`, "
        "which also copies the wavefunction the deck reads. Replacing an existing deck needs confirm."),
        {"properties": {**d3_props, "output_dir": {"type": "string"}}, "required": ["output_file"]},
        title="Write a D3 deck", group="setup", writes=True, confirm="on_overwrite")(setup_d3_write)

    reg.tool("deck_preflight", (
        "Check .d12 decks with CRYSTAL's TESTPDIM (`mace preflight`: a copy of each deck is read by a serial "
        "CRYSTAL in seconds; the deck is never changed) before spending a job. static_only skips CRYSTAL."),
        {"properties": {"paths": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 200},
                        "static_only": {"type": "boolean", "default": False},
                        "recursive": {"type": "boolean", "default": False},
                        "timeout": {"type": "integer", "minimum": 10, "maximum": 1800, "default": 120,
                                    "description": "seconds per deck"}},
         "required": ["paths"]},
        title="Preflight decks", group="setup")(deck_preflight)


# ---------------------------------------------------------------- helpers
def setup_templates(ctx, args):
    kinds = ["d12", "d3"] if (args.get("kind") or "all") == "all" else [args["kind"]]
    out: Dict[str, Any] = {}
    for kind in kinds:
        rows = []
        for p in resources.template_files(kind):
            row = resources.template_summary(p)
            row["uri"] = f"mace://templates/{kind}/{p.stem}"
            if args.get("full"):
                try:
                    row["json"] = json.loads(p.read_text())
                except (OSError, json.JSONDecodeError):
                    pass
            rows.append(row)
        out[kind] = rows
    return out


def _options(kind: str, args: Dict[str, Any], obj_key: str) -> Tuple[str, Dict[str, Any], str]:
    """(file text, parsed, where it came from) of the template or the object given."""
    if args.get("template") and args.get(obj_key) is not None:
        raise ValueError(f"give template or {obj_key}, not both")
    if args.get("template"):
        p = resources.template_path(kind, str(args["template"]))
        text = p.read_text()
        return text, json.loads(text), f"template {p.stem}"
    if args.get(obj_key) is not None:
        data = args[obj_key]
        return json.dumps(data, indent=2) + "\n", data, obj_key
    names = ", ".join(p.stem for p in resources.template_files(kind))
    raise ValueError(f"give a template ({names}) or {obj_key}")


def _calc_of(data: Dict[str, Any]) -> Optional[str]:
    conf = data.get("configuration", data) if isinstance(data, dict) else {}
    calc = conf.get("calculation_type") if isinstance(conf, dict) else None
    calc = calc or (data.get("calculation_type") if isinstance(data, dict) else None)
    return str(calc).upper() if calc else None


def _rel(ctx, p: Path) -> str:
    return p.relative_to(ctx.root).as_posix() if ctx.inside(p) else p.name


def _charge_args(args, mode: str) -> List[str]:
    raw = args.get("charged")
    if not raw:
        return []
    from mace.gui import setup as gsetup
    if mode not in gsetup.CHARGE_MODES:
        raise ValueError("charged-system switches are for D12 decks only")
    return gsetup.flag_args(gsetup.charge_flags(raw), mode)


def _prepare(ctx, args, d3: bool):
    from mace.gui import decks, setup as gsetup
    key = "output_file" if d3 else "structure"
    p = ctx.path(args[key], kind="file")
    if d3:
        if p.suffix.lower() not in gsetup.OUT_SUFFIXES:
            raise ValueError(f"{p.name}: a D3 deck is made from a finished run's .out")
        text, data, source = _options("d3", args, "config")
        calc = (args.get("calc_type") or _calc_of(data) or "").upper()
        if not calc:
            raise ValueError("give calc_type (the configuration does not name one)")
        mode = "opt2d3"
        calc = gsetup.check_calc(mode, calc)
        extra: List[str] = []
    else:
        mode = gsetup.mode_for(str(p), None)
        text, data, source = _options("d12", args, "options")
        problems = decks.validate(mode, data)
        if problems:
            raise ValueError("; ".join(problems))
        calc = _calc_of(data) or gsetup.DEFAULT_CALC[mode]
        if calc not in gsetup.CALC_TYPES[mode]:
            calc = gsetup.DEFAULT_CALC[mode]
        extra = _charge_args(args, mode)
    return p, mode, calc, text, source, extra


def _preview(ctx, p: Path, mode: str, calc: str, text: str, extra: List[str]) -> Dict[str, Any]:
    from mace.gui import setup as gsetup
    work = Path(tempfile.mkdtemp(prefix="mace_mcp_setup_"))
    try:
        ctx.progress(0.1, "staging")
        st = gsetup.stage(mode, calc, work, p, _rel(ctx, p))
        ctx.check()
        ctx.progress(0.3, f"mace {mode}")
        res = gsetup.run_printed(st, text, ctx=ctx, extra_args=extra)
        if res["code"] != 0 or not res["decks"]:
            raise ValueError("the CLI wrote no deck:\n" + gsetup.refusal(res["log"]))
        return {"decks": res["decks"], "log": res["log"][-4000:], "notes": list(st.notes)}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _cut(text: str) -> str:
    return text if len(text) <= DECK_TEXT_MAX else text[:DECK_TEXT_MAX] + "\n... (cut)"


def _preview_tool(ctx, args, d3: bool):
    p, mode, calc, text, source, extra = _prepare(ctx, args, d3)
    pv = _preview(ctx, p, mode, calc, text, extra)
    return {"input": ctx.show(p), "mode": mode, "calc_type": calc, "options_from": source,
            "decks": [{"name": d["name"], "text": _cut(d["text"])} for d in pv["decks"]],
            "notes": pv["notes"], "log": pv["log"],
            "next": f"setup_{'d3' if d3 else 'd12'}_write with the same arguments writes it"}


def setup_d12_preview(ctx, args):
    return _preview_tool(ctx, args, d3=False)


def setup_d3_preview(ctx, args):
    return _preview_tool(ctx, args, d3=True)


def _stamps(d: Path) -> Dict[str, Tuple[int, int]]:
    if not d.is_dir():
        return {}
    out = {}
    for x in d.iterdir():
        try:
            if x.is_file():
                st = x.stat()
                out[x.name] = (st.st_mtime_ns, st.st_size)
        except OSError:
            pass
    return out


def _write_tool(ctx, args, d3: bool):
    if ctx.config.read_only:
        raise ValueError("the server runs with --read-only")
    from mace.gui import setup as gsetup
    p, mode, calc, text, source, extra = _prepare(ctx, args, d3)
    outdir = ctx.path(args["output_dir"], must_exist=False, kind="dir") if args.get("output_dir") else p.parent
    pv = _preview(ctx, p, mode, calc, text, extra)
    names = [d["name"] for d in pv["decks"]]
    clash = [outdir / n for n in names if (outdir / n).exists()]
    if clash:
        from mace.gui import saving
        for c in clash:
            why = saving.protected_reason(c)
            if why:
                raise ValueError(why)
        ctx.require_confirm("replace " + ", ".join(ctx.show(c) for c in clash),
                            {"existing": [ctx.show(c) for c in clash],
                             "new_decks": [{"name": d["name"], "text": _cut(d["text"])} for d in pv["decks"]]})
        from mace.gui import history
        if history.root() is None:
            history.configure(ctx.root)
        history.snapshot(clash, "Replaced by mace mcp " + ("setup_d3_write" if d3 else "setup_d12_write"),
                         route="mcp")
    outdir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="mace_mcp_write_"))
    try:
        opts = work / "options.json"
        opts.write_text(text)
        if mode == "cif2d12":
            cif_dir = work / "cif"
            cif_dir.mkdir()
            shutil.copyfile(p, cif_dir / p.name)
            argv = ["convert", "--batch", "--options_file", str(opts), "--cif_dir", str(cif_dir),
                    "--output_dir", str(outdir)]
        elif mode == "opt2d12":
            argv = ["opt2d12", gsetup.source_flag(str(p)), str(p), "--config-file", str(opts), "--yes",
                    "--output-dir", str(outdir)]
        else:
            argv = ["opt2d3", "-i", str(p), "-t", calc, "--config-file", str(opts), "--output-dir", str(outdir)]
        argv += extra
        before = _stamps(outdir)
        ctx.progress(0.7, f"mace {argv[0]}")
        r = ctx.run_cli(argv, cwd=p.parent, timeout=600)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    after = _stamps(outdir)
    written = sorted(n for n, s in after.items() if before.get(n) != s)
    decks = [n for n in written if n.lower().endswith((".d12", ".d3"))]
    if r["code"] != 0 or not decks:
        raise ValueError("the CLI wrote no deck:\n" + gsetup.refusal(r["stdout"] + r["stderr"]))
    from mace.gui import history
    if history.root() is None:
        history.configure(ctx.root)
    for n in written:
        if n not in before:
            try:
                history.record_created(outdir / n, "Written by mace mcp", route="mcp")
            except Exception:
                pass
    shown = [str(a) for a in r["argv"]]
    return {"input": ctx.show(p), "mode": mode, "calc_type": calc, "options_from": source,
            "output_dir": ctx.show(outdir), "written": [ctx.show(outdir / n) for n in written],
            "decks": [{"path": ctx.show(outdir / n), "text": _cut((outdir / n).read_text(errors="replace"))}
                      for n in decks],
            "replaced": [ctx.show(c) for c in clash], "command": " ".join(shown),
            "next": "deck_preflight on the .d12 deck(s), then job_submit"}


def setup_d12_write(ctx, args):
    return _write_tool(ctx, args, d3=False)


def setup_d3_write(ctx, args):
    return _write_tool(ctx, args, d3=True)


def deck_preflight(ctx, args):
    paths = [ctx.path(x) for x in args["paths"]]
    timeout = int(args.get("timeout") or 120)
    argv = ["preflight", *[str(p) for p in paths], "-t", str(timeout)]
    if args.get("static_only"):
        argv.append("--static-only")
    if args.get("recursive"):
        argv.append("-r")
    n = sum(1 for p in paths if p.is_file()) or len(paths) * 10
    r = ctx.run_cli(argv, cwd=ctx.root, timeout=min(4 * 3600, n * (timeout + 30) + 60))
    meaning = {0: "every deck CRYSTAL saw was accepted", 1: "at least one deck was REJECTED by CRYSTAL",
               2: "at least one deck could not be checked (or no decks / no CRYSTAL found)"}
    text = (r["stdout"] + ("\n" + r["stderr"] if r["stderr"].strip() else "")).strip()
    return {"exit_code": r["code"], "verdict": meaning.get(r["code"], "unexpected exit code"),
            "ok": r["code"] == 0, "report": text[-20000:], "command": " ".join(r["argv"])}
