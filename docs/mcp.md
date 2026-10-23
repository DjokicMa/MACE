# MACE over MCP (`mace mcp`)

`mace mcp` serves MACE's tools to any client that speaks the
[Model Context Protocol](https://modelcontextprotocol.io) (MCP): agent frameworks,
editors and notebooks that can drive local tools. The client starts `mace mcp` as a
child process and talks JSON-RPC 2.0 to it over stdin/stdout. Every tool wraps
something MACE already does - the GUI's structure readers and builders, `mace preopt`,
`mace convert` / `opt2d12` / `opt2d3`, `mace preflight`, `mace submit`,
`mace cluster`, the completion checker, the property extractor and the materials.db
queries - so a client gets exactly what the CLI and the GUI give you.

Nothing extra has to be installed: MACE has its own small MCP transport. When the
official `mcp` Python package is installed (`pip install mcp`, MIT license), `mace mcp`
uses it instead, which brings the newest protocol versions with it.

## Start it

```bash
mace mcp --check                     # a self-test: initialize, list the tools, exit
mace mcp --list-tools                # what each tool does
mace mcp --config --root ~/calcs     # the JSON block for a client's server list
```

`--config` prints a block like this (absolute paths, so it works from clients that
start with a minimal `PATH`):

```json
{
  "mcpServers": {
    "mace": {
      "command": "/path/to/python",
      "args": ["/path/to/MACE/mace_cli", "mcp", "--root", "/home/you/calcs"]
    }
  }
}
```

Add it to your client's MCP server configuration (most clients use this
`mcpServers` layout). The client starts and stops the server; you never run plain
`mace mcp` in a terminal yourself (it waits for JSON on stdin).

### On a cluster, from your own computer

MCP over stdio works through SSH, so a client on your laptop can drive MACE on the
cluster:

```json
{
  "mcpServers": {
    "mace-hpcc": {
      "command": "ssh",
      "args": ["-T", "hpcc", "bash -lc 'mace mcp --root $HOME/calcs'"]
    }
  }
}
```

`bash -lc` loads your login environment (modules, `PATH`). Nothing in your shell
start-up files may print to stdout for a non-interactive shell: stdout carries the
protocol. Run `ssh -T hpcc "bash -lc 'mace mcp --check'"` once to see that it starts.

## Options

| Option | Meaning |
|---|---|
| `--root DIR` | The folder the tools work in (default: the current folder). Paths are relative to it, and a path outside it (`..`, an absolute path, a symbolic link pointing out) is refused. |
| `--read-only` | Offer only the tools that read: no file writes, no submit, no cancel. |
| `--allow-outside-root` | Accept paths outside `--root`. |
| `--transport auto\|builtin\|sdk` | `auto` (default): the official `mcp` package when installed, else the built-in transport. |
| `--workers N` | Built-in transport: tool calls that run at once (default 2). |
| `--list-tools [--json]`, `--config`, `--check` | For you, not for a client: list, print a client block, self-test. |
| `--debug` | Tracebacks on stderr. |

The server logs on stderr only (clients usually show it in their MCP log).

## Safety

* **Confirmations.** `job_submit` and `job_cancel` do nothing unless the call carries
  `"confirm": true`. Without it they answer with exactly what they would do (the decks,
  the `mace submit` command lines, the outputs a resubmission would overwrite, the job
  ids) and mark the answer as not done, so the client has to ask you first. Tools that
  write files create new files directly but need `"confirm": true` to replace an
  existing one.
* **File History.** A file replaced through `mace mcp` (and an output a submission
  overwrites) is kept in the GUI's File History first, so you can restore it from the
  GUI's File History (with the GUI serving the same folder) the same way as a GUI save.
* **Never written:** CRYSTAL outputs, job logs, wavefunctions, databases, MACE job
  scripts, and the decks of queued or running jobs - the same list the GUI protects.
* **The served folder.** Every path stays inside `--root` unless you start the server
  with `--allow-outside-root`. `--read-only` removes every tool that writes.
* Job cancels go through the GUI's checks: only your own queued or running jobs, never
  a whole job array (its tasks one by one), never the GUI's own job.

## Tools

| Group | Tool | What it does |
|---|---|---|
| files | `files_list` | Files of a folder with MACE's kind for each (cif, d12, d3, out, band, dos, cube, db, ...). |
| structure | `structure_inspect` | Any structure MACE reads (CIF, D12, a frame of a CRYSTAL `.out`, XYZ, POSCAR, fort.34, MOL, PDB, GEN): formula, lattice, dimensionality, space group, sites, a vacuum/dimensionality hint. |
| | `structure_supercell` | `[na, nb, nc]` or a 3x3 matrix; slabs and polymers repeat along their periodic directions only. |
| | `structure_standardize` | spglib primitive or conventional standard cell, Niggli cell, or symmetrize (3D crystals). |
| | `structure_slab` | A slab along (hkl) with layers, vacuum, termination; `terminations_only` lists the terminations. |
| | `structure_clean` | `mace preopt` (springs, GFN-FF, xTB tiers, `mlip:<id>`, auto): `<stem>_preopt.cif` and its provenance. A starting geometry, not a result. |
| | `mlip_models` | The ML potentials preopt can use, their licenses and readiness (`mace mlip list`). |
| | `structure_vacuum` | `mace supercell --vacuum`: a slab, polymer or molecule repeated along its non-periodic directions too, spaced by a vacuum mode (bbox, file, crystal, open or a gap in Å). |
| setup | `setup_templates` | The shipped D12 option templates and D3 configurations. |
| | `setup_d12_preview` / `setup_d12_write` | The `.d12` deck for a CIF (new calculation) or a finished `.out` / a `.d12` (follow-up, defaults from the parent) with a template or an options object; charged-system switches included. The preview runs the real CLI on a scratch copy; the write runs it on your files. |
| | `setup_d3_preview` / `setup_d3_write` | The `.d3` deck (BAND, DOSS, TRANSPORT, CHARGE, POTENTIAL, CHARGE+POTENTIAL, ORBITALS) for a finished run; the write copies the wavefunction as `mace opt2d3` does. |
| | `deck_preflight` | `mace preflight`: CRYSTAL's TESTPDIM on a copy of each deck (`static_only` without CRYSTAL). |
| jobs | `job_submit` | `mace submit` for decks of one folder or one folder (`track`, `progress`, `walltime`, `array`, `max_running`, ...). Needs confirm. |
| | `job_status` | Your SLURM jobs, or one job's state, progress, ETA and walltime risk. |
| | `job_cancel` | `scancel` after the GUI's checks. Needs confirm. |
| | `workflow_status` | `mace workflow --list` / `--status ID`. |
| | `cluster_info` | `mace cluster overview / nodes / quota / limits / reservations`. |
| results | `outputs_scan` | Finished, failed (and why) and running outputs in a folder, as `mace completion` sorts them. |
| | `output_summary` | One run: status, stop reason, optimization steps and convergence, energy, band gap, timing. |
| | `output_properties` | Everything the property extractor parses from an output (nothing is written to a database). |
| | `db_summary`, `db_search`, `db_material` | Read-only materials.db queries: counts, filtered pages, one material with its calculations and properties. |
| analysis | `analysis_porosity`, `analysis_pxrd`, `analysis_topology` | `mace analyze`: pore surface area, void fraction and pore sizes; a simulated powder pattern; the net and its RCSR match. Read-only. |
| calc | `calc_backends` | `mace calc list`: the force fields, charge models, tight-binding methods and ML potentials, and whether each is installed. |
| | `calc_run` | `mace calc`: SP, OPT, FREQ, MD or charges on a structure; writes the report and the task's files (confirm to replace them; the old ones go to File History). `options` takes listed `mace calc` options by full name (`--fmax=0.02`, `--relax-cell`); `--model-file`, `--model-license-ok` and `--skf-dir` are yours to give on the command line, never the client's. |

A typical session: `files_list` -> `structure_inspect` -> (`structure_clean` /
`structure_supercell`) -> `setup_templates` -> `setup_d12_preview` ->
`setup_d12_write` -> `deck_preflight` -> `job_submit` (you confirm) -> `job_status` ->
`output_summary` -> `setup_d3_write` -> ... -> `db_search`.

Long calls report progress when the client asks for it, and a client's cancel stops a
running MACE command (its process group) and drops the answer.

## Resources

| URI | Content |
|---|---|
| `mace://templates/d12/<name>` | A D12 options template (`Crystal_d12/example_configs`). |
| `mace://templates/d3/<name>` | A D3 configuration (`Crystal_d3/example_configs`). |
| `mace://docs/mcp` | This page. |

## Protocol support

The built-in transport speaks MCP protocol versions 2024-11-05, 2025-03-26,
2025-06-18 and 2025-11-25 (newline-delimited JSON-RPC 2.0 on stdio; `initialize`,
`ping`, `tools/list`, `tools/call`, `resources/list`, `resources/read`,
`resources/templates/list`, progress notifications, cancellation, batches). Tool
results carry `structuredContent` from 2025-06-18 on. With the official `mcp` package
(1.x or 2.x) the protocol side is that package's.

## Adding tools (for developers)

A MACE module adds a tool with one line in `mace/mcp_server/plugins.py`:

```python
PLUGINS = (
    plugin("porosity", "mace.analysis.porosity:mcp_porosity"),
    plugin("calc_opt", "mace.calculators.mcp:optimize", writes=True, confirm="on_overwrite"),
)
```

The function's signature is the tool's schema: typed parameters (`str`, `int`,
`float`, `bool`, lists, dicts, `Literal[...]` for choices, `Optional[...]` or a default
for optional ones); `Path` parameters are resolved inside the served folder; a
parameter named `ctx` receives the call's context (`ctx.progress(fraction, message)`,
`ctx.canceled` / `ctx.check()`, `ctx.write_file(path, bytes)` for a safe write,
`ctx.run_cli([...])` for a `mace` command). The docstring is the description (its
`Args:` lines describe the arguments) and the return value (dicts, lists, numbers,
numpy arrays) is the result. `writes=True` hides the tool in `--read-only`;
`confirm="always"` makes every call need `"confirm": true`. A function can carry the
same flags itself as `fn.__mcp__ = {"writes": True, "confirm": "always"}`.

A line whose module or function is missing (an optional part not installed) is
skipped with a note on stderr; the server still starts. Site-specific tools can be
added without editing MACE: `MACE_MCP_PLUGINS="mytools:check_quota,run=mytools:run_it"`
(comma-separated `module:function` or `name=module:function`, importable from the
server's `PYTHONPATH`).

## Troubleshooting

* `mace mcp --check` prints the server version, the protocol, the tool count and which
  transport `auto` picks.
* A client that reports "invalid JSON" on start: something printed to stdout before
  the server started (a shell start-up file over SSH, a wrapper script). `mace mcp`
  itself never writes anything but protocol messages to stdout.
* Tools that need SLURM answer `"slurm": false` with a note on a machine without it;
  `job_submit` refuses there.
