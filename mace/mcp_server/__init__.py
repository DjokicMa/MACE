"""``mace mcp``: MACE's tools over the Model Context Protocol (MCP).

An MCP client (an agent, an editor, a notebook kernel) starts ``mace mcp``
as a child process and talks JSON-RPC 2.0 to it over stdin/stdout. The
server answers ``initialize``, ``tools/list``, ``tools/call``,
``resources/list`` and ``resources/read``; every tool wraps a function or
command MACE already has (structure reading, supercells, Clean/preopt, deck
generation, preflight, SLURM submission and status, output parsing,
materials.db queries), so what a client gets is what the CLI and the GUI
give a person. User guide: docs/mcp.md.

Layout:

* ``registry``  - the Tool record, argument checking, and ``plugin()``:
  one line in ``plugins.py`` turns a plain typed function into a tool;
* ``context``   - the served folder (paths never leave it unless allowed),
  confirmations, safe writes, JSON conversion, running the CLI;
* ``server``    - the method dispatcher shared by both transports;
* ``stdio``     - the built-in transport (standard library only);
* ``sdk``       - the transport through the official ``mcp`` package, used
  when it is installed (``--transport auto``);
* ``resources`` - the option templates and the guide as MCP resources;
* ``tools/``    - the tools, one module per area;
* ``cli``       - ``mace mcp`` itself.

Nothing here is imported at ``mace`` startup, and the package itself needs
only the standard library: numpy, ase and the rest are imported by a tool
when it runs.

The package is ``mace.mcp_server``, not ``mace.mcp``: ``mace_cli`` puts the
``mace/`` folder itself on ``sys.path`` (as with ``mace/queue`` and the
standard library's ``queue``), so a folder named ``mcp`` there would be
imported in place of the official ``mcp`` package.
"""

PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
"""The protocol versions the built-in transport speaks, newest first."""
