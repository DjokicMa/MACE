"""Tools from other MACE areas: one ``plugin(...)`` line each.

``plugin(name, "package.module:function", writes=..., confirm=...)``

* the function's typed parameters are the tool's arguments (str, int, float,
  bool, lists, dicts, Literal[...] choices, Optional[...] for optional ones;
  defaults are shown to the client). ``Path`` parameters are resolved inside
  the served folder before the call; a parameter named ``ctx`` receives the
  call's context (``ctx.progress(fraction, message)``, ``ctx.check()`` to
  stop when the client cancels, ``ctx.write_file(path, bytes)`` for a safe
  write, ``ctx.run_cli([...])`` for a ``mace`` command);
* the docstring's first part is the description, its ``Args:`` lines the
  argument descriptions;
* the return value (dicts, lists, numbers, strings, numpy arrays) is the
  result;
* ``writes=True`` for anything that writes files (hidden with --read-only);
  ``confirm="always"`` for anything that submits jobs or deletes files: the
  call then needs ``"confirm": true``. ``confirm="on_overwrite"`` adds the
  argument and leaves the check to the function (it takes ``confirm: bool``
  or calls ``ctx.require_confirm``).

A line whose module or function does not exist (an optional area not
installed) is skipped with a note on stderr; the other tools still load.
List what loaded with ``mace mcp --list-tools``.
"""
from __future__ import annotations

from .registry import plugin  # noqa: F401  (used by the lines below)

PLUGINS = (
    # 1.1.6: vacuum supercells of slabs, polymers and molecules (mace supercell --vacuum)
    plugin("structure_vacuum", "mace.mcp_server.tools.wrappers:structure_vacuum", writes=True,
           confirm="on_overwrite", title="Vacuum supercell", group="structure"),
    # 1.1.6: structure analysis (mace analyze), read-only
    plugin("analysis_porosity", "mace.mcp_server.tools.wrappers:analysis_porosity", title="Porosity",
           group="analysis"),
    plugin("analysis_pxrd", "mace.mcp_server.tools.wrappers:analysis_pxrd", title="Powder XRD", group="analysis"),
    plugin("analysis_topology", "mace.mcp_server.tools.wrappers:analysis_topology", title="Topology",
           group="analysis"),
    # 1.1.6: force fields, charge models, tight binding and ML potentials (mace calc)
    plugin("calc_backends", "mace.mcp_server.tools.wrappers:calc_backends", title="Calculators", group="calc"),
    plugin("calc_run", "mace.mcp_server.tools.wrappers:calc_run", writes=True, confirm="on_overwrite",
           title="Run a calculator", group="calc"),
)
