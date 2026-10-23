"""The tool registry: what a tool is, how its arguments are checked, and
``plugin()``, which turns a plain typed function into a tool in one line.

A tool is a name, a description, a JSON Schema for its arguments and a
handler ``handler(ctx, args) -> JSON-able``. Three flags decide what the
server lets a client do with it:

* ``writes``  - it creates or changes files, submits or cancels jobs: hidden
  when the server runs with ``--read-only``;
* ``confirm`` - ``"always"``: nothing happens unless the call carries
  ``"confirm": true`` (a submit, a cancel); without it the tool answers with
  what it would do. ``"on_overwrite"``: the handler asks for it only when it
  would replace or delete a file (``ctx.require_confirm``). Both put a
  ``confirm`` boolean in the schema;
* ``open_world`` - it talks to something outside this machine's files (SLURM).

Tools of other MACE areas plug in through ``plugins.py``: one
``plugin("name", "module:function")`` line each. The function's signature is
the schema (type hints and defaults; ``Path`` arguments are resolved inside
the served folder), its docstring the description, its return value the
result. The module is imported only when the tool list is built, and a
module or function that is missing (an optional area not installed) leaves
the tool out with a note on stderr instead of stopping the server.
"""
from __future__ import annotations

import collections.abc as cabc
import importlib
import inspect
import re
import sys
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

CONFIRM_MODES = ("never", "always", "on_overwrite")
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")

CONFIRM_SCHEMA = {
    "type": "boolean",
    "default": False,
    "description": ("Must be true for the action to happen. Without it the tool only says what it would do; "
                    "ask the user before setting it."),
}


class ArgumentError(ValueError):
    """The call's arguments do not fit the tool's schema (the client can correct them)."""


@dataclass
class Tool:
    name: str
    description: str
    schema: Dict[str, Any]
    handler: Callable[..., Any]
    title: str = ""
    group: str = "general"
    writes: bool = False
    confirm: str = "never"
    open_world: bool = False
    idempotent: bool = False
    previews: bool = False                # confirm="always": the handler itself answers an unconfirmed call
    origin: str = "mace"                  # "mace" (built in) or "plugin:<module:function>"

    def __post_init__(self):
        if not NAME_RE.match(self.name):
            raise ValueError(f"tool name {self.name!r}: lowercase letters, digits and _ (2-64 characters)")
        if self.confirm not in CONFIRM_MODES:
            raise ValueError(f"tool {self.name}: confirm is one of {', '.join(CONFIRM_MODES)}")
        schema = dict(self.schema or {})
        schema.setdefault("type", "object")
        props = dict(schema.get("properties") or {})
        if self.confirm != "never":
            props["confirm"] = dict(CONFIRM_SCHEMA)
        schema["properties"] = props
        schema.setdefault("additionalProperties", False)
        self.schema = schema

    def listing(self) -> Dict[str, Any]:
        """The tool as tools/list shows it."""
        out = {"name": self.name, "description": self.description, "inputSchema": self.schema,
               "annotations": {"readOnlyHint": not self.writes,
                               "destructiveHint": bool(self.writes and self.confirm != "never"),
                               "idempotentHint": bool(self.idempotent or not self.writes),
                               "openWorldHint": bool(self.open_world)}}
        if self.title:
            out["title"] = self.title
            out["annotations"]["title"] = self.title
        return out


class Registry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self.skipped: List[Tuple[str, str]] = []        # (name, why) of plugins left out

    def add(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"two tools named {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def tool(self, name: str, description: str, schema: Optional[Dict[str, Any]] = None, **flags):
        """Decorator: ``@reg.tool("name", "what it does", {...schema...}, writes=True)``."""
        def deco(fn):
            self.add(Tool(name=name, description=description, schema=schema or {}, handler=fn, **flags))
            return fn
        return deco

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def tools(self, read_only: bool = False) -> List[Tool]:
        return [t for t in self._tools.values() if not (read_only and t.writes)]

    def names(self) -> List[str]:
        return list(self._tools)


# ---------------------------------------------------------------- argument checking
_TYPES = {"string": str, "boolean": bool, "object": dict, "array": (list, tuple)}


def _type_ok(value: Any, kind: str) -> bool:
    if kind == "null":
        return value is None
    if kind == "integer":
        return (isinstance(value, int) and not isinstance(value, bool)) or (
            isinstance(value, float) and value.is_integer())
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    py = _TYPES.get(kind)
    return True if py is None else isinstance(value, py)


def check(schema: Dict[str, Any], value: Any, where: str = "arguments") -> Any:
    """``value`` checked against the subset of JSON Schema the tools use
    (type, enum, minimum/maximum, minItems/maxItems, items, properties,
    required, additionalProperties); integers given as 3.0 come back as 3.
    Raises ArgumentError with the first problem."""
    kinds = schema.get("type")
    if kinds is not None:
        kinds = [kinds] if isinstance(kinds, str) else list(kinds)
        if not any(_type_ok(value, k) for k in kinds):
            raise ArgumentError(f"{where}: expected {' or '.join(kinds)}, got {type(value).__name__}")
        if "integer" in kinds and isinstance(value, float) and not ("number" in kinds):
            value = int(value)
    if "enum" in schema and value not in schema["enum"]:
        raise ArgumentError(f"{where}: one of {', '.join(map(str, schema['enum']))}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ArgumentError(f"{where}: at least {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise ArgumentError(f"{where}: at most {schema['maximum']}")
    if isinstance(value, str) and "maxLength" in schema and len(value) > schema["maxLength"]:
        raise ArgumentError(f"{where}: at most {schema['maxLength']} characters")
    if isinstance(value, (list, tuple)):
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise ArgumentError(f"{where}: at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ArgumentError(f"{where}: at most {schema['maxItems']} items")
        items = schema.get("items")
        if isinstance(items, dict):
            value = [check(items, v, f"{where}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, dict) and ("properties" in schema or "required" in schema):
        props = schema.get("properties") or {}
        for req in schema.get("required") or ():
            if req not in value:
                raise ArgumentError(f"{where}: {req} is required")
        extra = [k for k in value if k not in props]
        if extra and schema.get("additionalProperties") is False:
            raise ArgumentError(f"{where}: unknown argument{'s' if len(extra) > 1 else ''} {', '.join(sorted(extra))}; "
                                f"known: {', '.join(props) or 'none'}")
        value = {k: (check(props[k], v, k if where == 'arguments' else f"{where}.{k}") if k in props else v)
                 for k, v in value.items()}
    return value


# ---------------------------------------------------------------- plugins (signature -> schema)
@dataclass
class PluginSpec:
    name: str
    target: str                           # "package.module:function"
    writes: bool = False
    confirm: str = "never"
    open_world: bool = False
    description: Optional[str] = None
    title: str = ""
    group: str = "plugins"
    schema: Optional[Dict[str, Any]] = None
    kind: str = "function"                # "function", or "builder" (a structure transform, see plugin())


BUILDER_ARRAYS = ("cell", "frac", "numbers", "pbc")


def plugin(name: str, target: str, *, writes: bool = False, confirm: str = "never", open_world: bool = False,
           description: Optional[str] = None, title: str = "", group: str = "plugins",
           schema: Optional[Dict[str, Any]] = None, kind: str = "function") -> PluginSpec:
    """One plugins.py line: the tool ``name`` that calls ``target`` ("module:function").

    ``kind="builder"``: ``target`` is a structure transform in the style of
    mace.gui.builders - ``fn(ctx=..., cell=..., frac=..., numbers=..., pbc=..., **params)``
    returning ``builders.result(...)``. The tool then takes ``path`` (any
    structure file MACE reads), ``frame``, ``output`` and the function's other
    parameters, and writes the new structure (new file, or confirm to replace)."""
    if kind not in ("function", "builder"):
        raise ValueError("kind is 'function' or 'builder'")
    if kind == "builder":
        writes = True
        confirm = "on_overwrite" if confirm == "never" else confirm
        group = "structure" if group == "plugins" else group
    return PluginSpec(name, target, writes, confirm, open_world, description, title, group, schema, kind)


def _unwrap_optional(tp):
    origin = typing.get_origin(tp)
    if origin is typing.Union:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return tp, False


def type_schema(tp) -> Dict[str, Any]:
    """The JSON Schema of one type hint (the plain types, containers, Literal, Path)."""
    tp, _ = _unwrap_optional(tp)
    if tp is inspect.Parameter.empty or tp is Any:
        return {}
    if tp is bool:
        return {"type": "boolean"}
    if tp is int:
        return {"type": "integer"}
    if tp is float:
        return {"type": "number"}
    if tp is str:
        return {"type": "string"}
    if isinstance(tp, type) and issubclass(tp, Path):
        return {"type": "string", "description": "a path inside the served folder"}
    origin = typing.get_origin(tp)
    if origin is typing.Literal:
        vals = list(typing.get_args(tp))
        out: Dict[str, Any] = {"enum": vals}
        kinds = {type(v) for v in vals}
        if kinds == {str}:
            out["type"] = "string"
        elif kinds <= {int}:
            out["type"] = "integer"
        return out
    if origin in (list, tuple, set, frozenset, cabc.Sequence, cabc.Iterable, cabc.Collection) or tp in (list, tuple):
        args = [a for a in typing.get_args(tp) if a is not Ellipsis]
        out = {"type": "array"}
        if args:
            out["items"] = type_schema(args[0])
        return out
    if origin in (dict, cabc.Mapping) or tp is dict:
        return {"type": "object"}
    return {}


def _is_path(tp) -> bool:
    tp, _ = _unwrap_optional(tp)
    if isinstance(tp, type) and issubclass(tp, Path):
        return True
    args = typing.get_args(tp)
    return bool(args) and all(isinstance(a, type) and issubclass(a, Path) for a in args if a is not Ellipsis)


_ARG_LINE = re.compile(r"^\s{2,}(\*{0,2}[A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^)]*\))?\s*:\s*(.+)$")


def doc_parts(fn) -> Tuple[str, Dict[str, str]]:
    """(description, {argument: text}) from a Google-style docstring."""
    doc = inspect.getdoc(fn) or ""
    head, args, in_args, last = [], {}, False, None
    for line in doc.splitlines():
        s = line.strip()
        if s in ("Args:", "Arguments:", "Parameters:"):
            in_args = True
            continue
        if in_args:
            if s in ("Returns:", "Raises:", "Yields:", "Examples:", "Notes:"):
                in_args = False
                continue
            m = _ARG_LINE.match("    " + line if not line.startswith("  ") else line)
            if m:
                last = m.group(1).lstrip("*")
                args[last] = m.group(2).strip()
            elif s and last:
                args[last] += " " + s
            continue
        head.append(line)
    # a paragraph wrapped over several lines is one line of the description
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", "\n".join(head).strip())
    return text, args


def function_tool(spec: PluginSpec, fn: Callable) -> Tool:
    """The Tool of a plugin function: its typed parameters are the schema;
    a parameter named ``ctx`` gets the call's context; ``Path`` parameters
    are resolved inside the served folder before the call."""
    marks = getattr(fn, "__mcp__", None)
    if isinstance(marks, dict):
        # the function's own flags ({"writes": True, "confirm": "always", ...}) for a line that sets none
        for key in ("writes", "confirm", "open_world", "title", "description", "kind"):
            if key in marks and getattr(spec, key) == getattr(PluginSpec, key, None):
                setattr(spec, key, marks[key])
        if spec.kind == "builder":
            spec.writes = True
            spec.confirm = "on_overwrite" if spec.confirm == "never" else spec.confirm
    sig = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}
    desc, arg_docs = doc_parts(fn)
    props: Dict[str, Any] = {}
    required: List[str] = []
    paths: List[str] = []
    wants_ctx = False
    builder = spec.kind == "builder"
    if builder:
        props.update({"path": {"type": "string", "description": "the structure file (any format MACE reads)"},
                      "frame": {"type": "integer", "default": -1, "description": "for a CRYSTAL .out: the frame"},
                      "output": {"type": "string", "description": "file to write (.cif, .xyz, .vasp); default: "
                                                                  "<input stem>_<tool name>.cif beside the input"}})
        required.append("path")
    for pname, p in sig.parameters.items():
        if pname == "ctx":
            wants_ctx = True
            continue
        if builder and pname in BUILDER_ARRAYS:
            continue
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if pname == "confirm":
            continue                       # the registry's own argument
        tp = hints.get(pname, p.annotation)
        s = dict(type_schema(tp))
        if pname in arg_docs:
            s["description"] = arg_docs[pname]
        if p.default is not p.empty:
            if p.default is None or isinstance(p.default, (str, int, float, bool)):
                if p.default is not None:
                    s["default"] = p.default
            elif isinstance(p.default, (list, tuple)) and all(isinstance(x, (str, int, float, bool)) for x in p.default):
                s["default"] = list(p.default)
        else:
            required.append(pname)
        if _is_path(tp):
            paths.append(pname)
        props[pname] = s
    schema = spec.schema or {"type": "object", "properties": props, "required": required}
    path_args = tuple(paths)

    takes_any = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())

    def build_handler(ctx, args):
        from .tools import structure as st
        call = dict(args)
        p, d = st.read(ctx, call.pop("path"), call.pop("frame", -1))
        output = call.pop("output", None)
        for k in path_args:
            if call.get(k) is not None:
                call[k] = ctx.path(call[k], must_exist=False)
        if wants_ctx or takes_any:
            call["ctx"] = ctx
        built = fn(cell=d["cell"], frac=d["frac"], numbers=d["numbers"], pbc=tuple(bool(x) for x in d["pbc"]), **call)
        return st.write_built(ctx, p, built, spec.name, output, f"{spec.name} of {p.name} (mace mcp)")

    def handler(ctx, args):
        call = dict(args)
        for k in path_args:
            v = call.get(k)
            if v is None:
                continue
            if isinstance(v, (list, tuple)):
                call[k] = [ctx.path(x, must_exist=False) for x in v]
            else:
                call[k] = ctx.path(v, must_exist=False)
        if spec.confirm == "on_overwrite" and "confirm" in sig.parameters:
            call["confirm"] = ctx.confirmed      # "always" is checked by the server before the call
        if wants_ctx:
            call["ctx"] = ctx
        return fn(**call)

    return Tool(name=spec.name, description=(spec.description or desc or spec.target).strip(), schema=schema,
                handler=build_handler if builder else handler, title=spec.title, group=spec.group,
                writes=spec.writes, confirm=spec.confirm, open_world=spec.open_world, origin=f"plugin:{spec.target}")


def load_plugins(reg: Registry, specs: Iterable[PluginSpec], log=None) -> None:
    """Add each plugin whose module and function import; the others go to
    ``reg.skipped`` with the reason (and to ``log``, stderr by default)."""
    log = log or (lambda msg: sys.stderr.write(msg + "\n"))
    for spec in specs:
        mod_name, _, attr = spec.target.partition(":")
        try:
            if not attr:
                raise ValueError("target must be 'module:function'")
            fn = getattr(importlib.import_module(mod_name), attr)
            reg.add(function_tool(spec, fn))
        except Exception as exc:                 # an optional area that is not there: no tool, no crash
            why = f"{type(exc).__name__}: {exc}"
            reg.skipped.append((spec.name, why))
            log(f"mace mcp: tool {spec.name} ({spec.target}) not loaded - {why}")
