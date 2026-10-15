"""Site settings: what MACE writes into the job scripts it generates on this cluster.

MACE's job-script templates (``submitcrystal23.sh``, ``submit_prop.sh``) were
written for one cluster (MSU's HPCC): its SLURM account, its CRYSTAL module
and its Python modules. This module is the one Python home of those values
(``DEFAULTS``) and lets a user on any cluster replace them, without editing
the templates:

* ``~/.config/mace/site.toml`` (``$XDG_CONFIG_HOME``; ``MACE_SITE_CONFIG=<path>``
  replaces it, ``MACE_SITE_CONFIG=none`` ignores it), keys ``[slurm] account,
  partition``, ``[modules] crystal, crystal_minimum, crystal_maximum,
  python``, ``[queue] max_jobs, reserve`` and ``[local_run]
  gateway_message, module_message``;
* the environment: ``MACE_SLURM_ACCOUNT``, ``MACE_SLURM_PARTITION`` and
  ``MACE_CRYSTAL_MODULE`` (an exact pin, comma-separated for several
  modules; ``none`` keeps the template's line). Precedence: env > site.toml >
  built-in.

With no setting of the user's own, MACE checks the built-in values against
the cluster it runs on (``resolve``), so it works on any SLURM cluster and
the cluster the built-ins come from keeps byte-identical scripts:

* account: kept when the user is a member of it (``sacctmgr``, cached a day),
  and where that cannot be checked only on MSU HPCC or on a host with no
  SLURM at all (a workstation, CI); otherwise the ``#SBATCH -A`` line is
  dropped and SLURM charges the user's default account (said once);
* Python modules: kept when they load here (an Lmod/Tcl probe, cached) or
  when there is no module system; otherwise the job runs its completion
  step with the Python that runs MACE (``sys.executable``, said once);
* queue limits of the completion callback (``--max-jobs 950 --reserve 50``,
  a 1000-job MaxSubmitJobs): from the user's association
  (``scontrol show assoc_mgr``, cached a day) when SLURM reports one, else
  the built-in numbers; ``[queue] max_jobs, reserve`` set them.

Beyond these, every other part of the job script has an optional key
(``[resources]`` job size, ``[mpi]`` launch, ``[crystal]`` bin_dir and
executable names, ``[modules] purge``, ``[slurm] qos/constraint/extra``,
``[paths] scratch``, ``[local_run] max_time``), each changing only its own
lines and only when set. Group files (``/etc/mace/site.toml``,
``$XDG_CONFIG_DIRS/mace/site.toml`` or ``MACE_SITE_GROUP_CONFIG``) sit below
the user's file; precedence per key: env > site.toml > group > built-in
(``config_layers``). A group file others can modify is refused; a settings
file this Python cannot read (no tomllib) stops generation. Values that
reach an executed line of the job script are allow-listed data (no shell
metacharacters), in ``load()`` and again in a workflow plan's replay.

The CRYSTAL module is resolved by ``crystal_module``: a pin as given; inside
a SLURM job (a completion callback) never a probe, but the module the job
loaded; without a module system (a workstation, CI) the preferred module
unchecked; else the preferred module when it loads and provides Pcrystal and
Pproperties, and only otherwise the best CRYSTAL module of the same major
version (never another major unless allowed by crystal_maximum/minimum).

``customize_generator_text`` rewrites only the default literals of a
template (matched after ``rstrip()``), so with no settings and a module that
loads (or no module system) every generated script keeps its bytes.
``generator_for`` applies it to MACE's own shipped templates only; copies
(a workflow's ``workflow_scripts/``, a user's ``<job>.sh``) run as written.

Standard library plus ``mace.utils.envmodules`` only: the generators also run
on a cluster's module Python.
"""
from __future__ import annotations

import json
import os
import re
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from mace.utils import envmodules
except ImportError:          # pragma: no cover - imported as a bare module next to mace/utils
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("envmodules", Path(__file__).resolve().parents[1] / "utils" / "envmodules.py")
    envmodules = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(envmodules)
try:
    from mace.workflow.common.constants import PROPERTIES_MEMORY, PROPERTIES_NTASKS
except ImportError:          # pragma: no cover - imported as a bare module next to mace/workflow
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("mace_constants", Path(__file__).resolve().parents[1] / "workflow" / "common" / "constants.py")
    _constants = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_constants)
    PROPERTIES_NTASKS = _constants.PROPERTIES_NTASKS
    PROPERTIES_MEMORY = _constants.PROPERTIES_MEMORY

SUBMISSION_DIR = Path(__file__).resolve().parent
TEMPLATES = {"scf": SUBMISSION_DIR / "submitcrystal23.sh", "prop": SUBMISSION_DIR / "submit_prop.sh"}
BINARIES = ("Pcrystal", "Pproperties")

# The built-in values: MSU HPCC's, which the templates carry. The only
# Python home of these literals (the templates hold them as text).
DEFAULTS: Dict[str, Any] = {
    "account": "mendoza_q",
    "partition": None,
    "crystal": ["CRYSTAL/23-intel-2023a"],
    "python": ["Python/3.11.3-GCCcore-12.3.0", "Python-bundle-PyPI/2023.06-GCCcore-12.3.0"],
    "gateway_message": ("This is a login node ({reason}): CRYSTAL cannot run here. Log in to an interactive "
                        "or development node first and rerun mace run."),
    "module_message": ("the {module} module cannot be loaded on {host}; run mace run on a node where it loads "
                       "(an interactive or development node)"),
    # the completion callback's queue limits: MaxSubmitJobs 1000 minus a reserve
    "max_jobs": 950,
    "reserve": 50,
    "max_submit": 10,
}
# The built-in messages name MSU's development nodes only where the host is
# verifiably one of MSU HPCC's (msu_hpcc), as an example.
MSU_EXAMPLES = {"gateway_message": " On MSU HPCC: ssh dev-amd20, dev-amd24 or dev-intel18.",
                "module_message": " (on MSU HPCC: ssh dev-amd20)"}
# MSU HPCC's own signals: its profile script (it sets $SCRATCH), its domain
# and its node names (development nodes, compute node families).
MSU_PROFILE = "/etc/profile.d/hpcc.sh"
MSU_HOST_RE = re.compile(r"^(dev-(amd|intel)\d+|(amr|nvf|agg|agx|nfh|neh|nel|acm|nal|nif|skl|nvl)-\d+)$")
# the template lines the rewrite looks for (compared after rstrip())
ACCOUNT_LINES = {"scf": "echo '#SBATCH -A mendoza_q # or general' >> $1.sh",
                 "prop": "echo '#SBATCH -A mendoza_q' >> $1.sh"}
CRYSTAL_LINE = "module load CRYSTAL/23-intel-2023a"
PYTHON_LINES = ("module load Python/3.11.3-GCCcore-12.3.0", "module load Python-bundle-PyPI/2023.06-GCCcore-12.3.0")
QUEUE_TEXT = "--max-jobs 950 --reserve 50 --max-submit 10"
PYTHON_FUNCTION = 'python() {{ "{exe}" "$@"; }}'
PYTHON_COMMENT = ("# MACE: the built-in Python modules do not load here; the completion step runs the Python "
                  "that runs MACE")
SAFE_EXE_RE = re.compile(r"^/[A-Za-z0-9._+/@:,=-]{1,400}$")
ROOT_FROM_PATH_LINE = ('[ -n "$EBROOTCRYSTAL" ] || export EBROOTCRYSTAL="$(dirname "$(dirname "$(command -v '
                       'Pcrystal)")")"')
# the templates' default walltime lines (time=..., wall=...) and their seconds
WALLTIME_LINES = {"scf": ("time=7", "wall=-00:00:00", 7 * 86400), "prop": ("time=2", "wall=:00:00", 2 * 3600)}
SLURM_CACHE_TTL_S = 24 * 3600
SLURM_TIMEOUT_S = 10

ACCOUNT_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
PARTITION_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}(,[A-Za-z0-9._-]{1,64}){0,15}$")
MODULE_RE = re.compile(r"^[A-Za-z0-9._+/-]{1,128}$")
MAJOR_RE = re.compile(r"^[0-9]{1,3}$")
PLACEHOLDERS = {"reason", "module", "host"}
KNOWN = {"slurm": {"account", "partition", "qos", "constraint", "extra"},
         "modules": {"crystal", "crystal_minimum", "crystal_maximum", "python", "purge"},
         "queue": {"max_jobs", "reserve"},
         "local_run": {"gateway_message", "module_message", "max_time"},
         "resources": {"ntasks", "nodes", "cpus_per_task", "mem", "mem_per_cpu", "properties"},
         "mpi": {"launcher", "srun_mpi", "env"},
         "paths": {"scratch"},
         "crystal": {"bin_dir", "scf", "properties"}}
RESOURCE_KEYS = ("ntasks", "nodes", "cpus_per_task", "mem", "mem_per_cpu")
# the tables of the same file that `mace cluster` reads (mace/cluster/config.py),
# and [gui] (gateway: the login host `mace gui` names in its tunnel hint)
FOREIGN_TABLES = {"cluster", "quota", "scan", "devnodes", "gui"}
INVALID_ACCOUNT_TEXT = "Invalid account or account/partition combination"

# ------------------------------------------------------------------ the layers
# Group settings (a cluster's admins, a lab): /etc/mace/site.toml, then
# <dir>/mace/site.toml for each $XDG_CONFIG_DIRS entry (the first wins), or
# the files named in MACE_SITE_GROUP_CONFIG instead. The user's own file and
# the environment override them per key.
SYSTEM_FILE = Path("/etc/mace/site.toml")

# ------------------------------------------------------------------ the values
# Every value lands in a single-quoted `echo '...' >> $1.sh` block of a
# template; the ones on an executed line of the job script use charsets with
# no shell metacharacters, so bash can run nothing from them.
INT_LIMITS = {"ntasks": (1, 4096), "nodes": (1, 64), "cpus_per_task": (1, 256)}
MEM_RE = re.compile(r"^[0-9]{1,7}[KMGT]?$")
QOS_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
CONSTRAINT_RE = re.compile(r"^[A-Za-z0-9._&|\[\]*:,!()-]{1,200}$")
EXTRA_RE = re.compile(r"^--([a-z][a-z-]{1,30})(=[A-Za-z0-9._:,@%+\[\]-]{1,200})?$")
# directives MACE writes itself (and the key that sets each)
EXTRA_REFUSED = {"ntasks": "[resources] ntasks", "nodes": "[resources] nodes",
                 "cpus-per-task": "[resources] cpus_per_task", "mem": "[resources] mem",
                 "mem-per-cpu": "[resources] mem_per_cpu", "time": "mace submit --walltime",
                 "account": "[slurm] account", "partition": "[slurm] partition", "qos": "[slurm] qos",
                 "constraint": "[slurm] constraint", "job-name": "(MACE names the job after the deck)",
                 "output": "(MACE names the output after the deck)", "error": "(MACE writes one output file)",
                 "array": "mace submit --array", "exclude": "mace submit --exclude",
                 "chdir": "(MACE runs the job in the deck's folder)"}
MAX_EXTRA = 16
LAUNCHERS = ("mpirun", "srun")
SRUN_MPI = ("pmix", "pmi2", "pmix_v3", "pmix_v4", "none")
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
ENV_VALUE_RE = re.compile(r"^[A-Za-z0-9_./:,=+@%^-]{0,300}$")
MAX_ENV = 16
# never set from site settings: they change how the job script, bash or
# MACE itself run (a superset of mace/cluster/config.py's list)
REFUSED_ENV_PREFIXES = ("LD_", "DYLD_", "PYTHON", "PERL5", "RUBY", "BASH_", "MACE_", "SLURM_")
REFUSED_ENV_NAMES = {"PATH", "BASH_ENV", "ENV", "IFS", "SHELLOPTS", "BASHOPTS", "CDPATH", "GLOBIGNORE",
                     "POSIXLY_CORRECT", "PS4", "PROMPT_COMMAND", "HOME", "USER",
                     "LOGNAME", "SHELL", "SCRATCH", "TMPDIR", "DIR", "JOB", "scratch", "MODULEPATH"}
BIN_DIR_RE = re.compile(r"^/[A-Za-z0-9._+@/-]{1,300}$")
EXE_RE = re.compile(r"^[A-Za-z0-9._+-]{1,64}$")
SCRATCH_RE = re.compile(r"^(/|\$HOME/|\$\{HOME\}/)[A-Za-z0-9._+@/${}-]{0,250}$")
# a bare $NAME token ends where a shell name ends: "$USERNAME" is not "$USER"
SCRATCH_TOKEN_RE = re.compile(r"\$\{(?:SLURM_JOB_ID|USER|HOME)\}|\$(?:SLURM_JOB_ID|USER|HOME)(?![A-Za-z0-9_])")
# the templates' job-size lines (compared after rstrip()) and their numbers
RESOURCE_LINES = {
    "scf": {"cpus_per_task": "echo '#SBATCH --cpus-per-task=1' >> $1.sh",
            "ntasks": "echo '#SBATCH --ntasks=32' >> $1.sh",
            "nodes": "echo '#SBATCH -N 1' >> $1.sh",
            "memory": "echo '#SBATCH --mem-per-cpu=5G' >> $1.sh"},
    "prop": {"cpus_per_task": "echo '#SBATCH --cpus-per-task=1' >> $1.sh",
             "ntasks": f"echo '#SBATCH --ntasks={PROPERTIES_NTASKS}' >> $1.sh",
             "nodes": "echo '#SBATCH -N 1' >> $1.sh",
             "memory": f"echo '#SBATCH --mem={PROPERTIES_MEMORY}' >> $1.sh"}}
TEMPLATE_RESOURCES = {"scf": {"ntasks": 32, "nodes": 1, "cpus_per_task": 1, "mem_form": "mem_per_cpu", "mem": "5G"},
                      "prop": {"ntasks": PROPERTIES_NTASKS, "nodes": 1, "cpus_per_task": 1, "mem_form": "mem", "mem": PROPERTIES_MEMORY}}
# how the templates start CRYSTAL
LAUNCH_LINES = {k: ('I_MPI_HYDRA_BOOTSTRAP="ssh" mpirun -n "$SLURM_NTASKS" "$EBROOTCRYSTAL/bin/' + exe
                    + '" 2>&1 >& "$DIR/${JOB}.out"') for k, exe in (("scf", "Pcrystal"), ("prop", "Pproperties"))}
# the same line as earlier templates wrote it (a template copied into a
# workflow folder by an earlier MACE still has it): unquoted before the 1.1.5
# release, and before 1.1.5 with an unquoted output path too
_UNQUOTED_LAUNCH = {k: l.replace('"$SLURM_NTASKS"', "$SLURM_NTASKS").replace('"$EBROOTCRYSTAL/bin/' + exe + '"',
                                                                               "$EBROOTCRYSTAL/bin/" + exe)
                    for (k, l), exe in zip(LAUNCH_LINES.items(), ("Pcrystal", "Pproperties"))}
LEGACY_LAUNCH_LINES = {k: l.replace('"$DIR/${JOB}.out"', "$DIR/${JOB}.out") for k, l in _UNQUOTED_LAUNCH.items()}
OLDER_LAUNCH_LINES = {k: (_UNQUOTED_LAUNCH[k], LEGACY_LAUNCH_LINES[k]) for k in LAUNCH_LINES}
PURGE_LINE = "module purge"
PURGE_COMMENT = "# MACE: module purge skipped ([modules] purge = false)"
# the scratch guard's /mnt/scratch candidate (the site's own goes before it)
SCRATCH_LINES = ('if [ -n "$MACE_USER" ] && [ -d "/mnt/scratch/$MACE_USER" ]; then',
                 '  MACE_SCRATCH_TRY+=("$(readlink -f "/mnt/scratch/$MACE_USER")$MACE_SCRATCH_SUFFIX")')

INIT_TEXT = """\
# MACE site settings: what MACE writes into the job scripts it generates on this cluster.
# Every line is commented out: MACE uses its built-in values, checked against this cluster
# (an account you are not in is left out, Python modules that do not load are replaced by
# the Python that runs MACE), until you uncomment a line and set your own value.
# Check the result with: mace site
# MACE submits jobs with SLURM only (PBS, Torque and LSF: MACE 1.2 or later).
# Before your first real job: mace site check
[slurm]
# account   = "my_account"                 # #SBATCH -A ...  (find yours with: sacctmgr show assoc user=$USER format=account)
# partition = ""                           # #SBATCH -p ...  ("" = none; "a,b" = a list SLURM picks from)
# qos        = "normal"                    # #SBATCH --qos=...
# constraint = "epyc"                      # #SBATCH --constraint=...  (node features: sinfo -o "%P %f")
# extra      = ["--exclusive"]             # more #SBATCH options, e.g. "--reservation=x", "--clusters=c2"
[resources]
# ntasks        = 32                       # #SBATCH --ntasks=...   (MPI ranks; built-in: 32, properties @PROP_NTASKS@)
# nodes         = 1                        # #SBATCH -N ...         (above 1 needs [paths] scratch shared by all nodes)
# cpus_per_task = 1                        # #SBATCH --cpus-per-task=...
# mem_per_cpu   = "5G"                     # #SBATCH --mem-per-cpu=...  } set one of the two
# mem           = "250G"                   # #SBATCH --mem=...          } (built-in: 5G per CPU, properties @PROP_MEM@)
[resources.properties]
# ntasks = @PROP_NTASKS_PADDED@# the same keys, for Pproperties jobs only
# mem    = "@PROP_MEM@"
[mpi]
# launcher = "mpirun"                      # or "srun" (srun reads the task count from SLURM)
# srun_mpi = "pmix"                        # srun --mpi=...  (pmix, pmi2, pmix_v3, pmix_v4, none)
# env = ["OMPI_MCA_btl=^openib"]           # exported before CRYSTAL starts (NAME=value, no spaces)
[modules]
# crystal = "CRYSTAL/23-foss-2023a"        # tried first; or a list loaded in order for Lmod hierarchies:
# crystal = ["foss/2023a", "CRYSTAL/23-foss-2023a"]
# crystal_minimum = "23"                   # if crystal does not load here, MACE uses the best CRYSTAL module
# crystal_maximum = "23"                   # between these major versions (default: the major of crystal)
# python = ["Python/3.11"]                 # loaded for the completion step; [] = load none
# purge = false                            # keep the modules loaded at login (Cray PE); default true
[crystal]
# bin_dir    = "/opt/crystal23/bin"        # CRYSTAL without a module: Pcrystal and Pproperties are here
# scf        = "Pcrystal"                  # or "MPPcrystal"
# properties = "Pproperties"
[paths]
# scratch = "/scratch/$USER"               # tried after $SCRATCH; only a directory every compute node shares
[queue]
# max_jobs = 950                           # jobs MACE keeps queued (default: your MaxSubmitJobs minus the reserve)
# reserve  = 50                            # queue slots MACE leaves free for you
[local_run]
# max_time = "4h"                          # longest mace run per deck (default: the node's ulimit -t; 1h55m on MSU HPCC)
# gateway_message = "This is a login node ({reason}): CRYSTAL cannot run here. Log in to a compute or dev node first and rerun mace run."
# module_message  = "the {module} module cannot be loaded on {host}; run mace run on a node where it loads"
""".replace("@PROP_NTASKS_PADDED@", f"{PROPERTIES_NTASKS:<32}").replace("@PROP_NTASKS@", str(PROPERTIES_NTASKS)).replace("@PROP_MEM@", PROPERTIES_MEMORY)


class SiteConfigError(ValueError):
    """An invalid site setting: nothing may be generated."""


@dataclass
class Site:
    # None: no #SBATCH -A line (SLURM's default account), set by resolve()
    account: Optional[str] = DEFAULTS["account"]
    partition: Optional[str] = None
    crystal: List[str] = field(default_factory=lambda: list(DEFAULTS["crystal"]))
    crystal_minimum: Optional[int] = None
    crystal_maximum: Optional[int] = None
    python: List[str] = field(default_factory=lambda: list(DEFAULTS["python"]))
    gateway_message: str = DEFAULTS["gateway_message"]
    module_message: str = DEFAULTS["module_message"]
    max_jobs: int = DEFAULTS["max_jobs"]
    reserve: int = DEFAULTS["reserve"]
    path: Optional[Path] = None
    exists: bool = False
    sources: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    pin: Optional[List[str]] = None           # MACE_CRYSTAL_MODULE, validated
    pin_none: bool = False                    # MACE_CRYSTAL_MODULE=none
    # set by resolve(): the Python the completion step runs when the Python
    # modules do not load here, and what was checked (for mace site)
    python_executable: Optional[str] = None
    walltime_cap: Optional[int] = None        # seconds: the partition's MaxTime, when below a template's
    cap_partition: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    checks: Dict[str, Any] = field(default_factory=dict)
    # where each set value comes from (a path, an env name), for mace site
    origins: Dict[str, str] = field(default_factory=dict)
    group_paths: List[Path] = field(default_factory=list)
    group_owners: Dict[str, str] = field(default_factory=dict)
    # [resources]: "" = both job kinds, "properties" = Pproperties jobs only;
    # only the keys someone set
    res_raw: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {"": {}, "properties": {}})
    # [mpi]
    launcher: str = "mpirun"
    srun_mpi: Optional[str] = None
    mpi_env: List[str] = field(default_factory=list)
    # [paths] scratch (None: the template's candidates only)
    scratch: Optional[str] = None
    # [crystal]
    crystal_bin_dir: Optional[str] = None
    crystal_exe: Dict[str, str] = field(default_factory=lambda: {"scf": "Pcrystal", "prop": "Pproperties"})
    # [slurm] qos, constraint, extra
    qos: Optional[str] = None
    constraint: Optional[str] = None
    extra: List[str] = field(default_factory=list)
    # [modules] purge
    module_purge: bool = True
    # [local_run] max_time (seconds)
    run_max_time: Optional[int] = None

    def is_set(self, key: str) -> bool:
        """The value is the user's (site.toml, a group file, the environment)."""
        return self.sources.get(key, "built-in") != "built-in"

    def effective_resources(self, kind: str) -> Dict[str, Any]:
        """The job size of ``kind`` ('scf' / 'prop'): the template's numbers
        with the keys someone set. ``mem_form`` is 'mem' or 'mem_per_cpu'."""
        out = dict(TEMPLATE_RESOURCES[kind])
        layers = [self.res_raw.get("", {})] + ([self.res_raw.get("properties", {})] if kind == "prop" else [])
        for layer in layers:
            for k in ("ntasks", "nodes", "cpus_per_task"):
                if k in layer:
                    out[k] = layer[k]
            if "mem" in layer:
                out["mem_form"], out["mem"] = "mem", layer["mem"]
            elif "mem_per_cpu" in layer:
                out["mem_form"], out["mem"] = "mem_per_cpu", layer["mem_per_cpu"]
        return out

    def resources_set(self, kind: str) -> Dict[str, Any]:
        """Only the job-size keys someone set for ``kind`` (memory as mem_form/mem)."""
        eff = self.effective_resources(kind)
        layers = [self.res_raw.get("", {})] + ([self.res_raw.get("properties", {})] if kind == "prop" else [])
        keys = set().union(*[set(l) for l in layers])
        out = {k: eff[k] for k in ("ntasks", "nodes", "cpus_per_task") if k in keys}
        if keys & {"mem", "mem_per_cpu"}:
            out["mem_form"], out["mem"] = eff["mem_form"], eff["mem"]
        return out

    def binaries(self) -> Tuple[str, str]:
        return self.crystal_exe["scf"], self.crystal_exe["prop"]

    @property
    def window(self) -> Tuple[int, int]:
        # the preferred module's major; a name without a version (a site's own
        # module name) falls back to the built-in module's major
        major = (envmodules.version_key(self.crystal[-1]) or envmodules.version_key(DEFAULTS["crystal"][-1]) or (0,))[0]
        lo = self.crystal_minimum if self.crystal_minimum is not None else major
        hi = self.crystal_maximum if self.crystal_maximum is not None else major
        return lo, hi

    def message(self, key: str, env: Optional[Dict[str, str]] = None, **values) -> str:
        text = getattr(self, key)
        if self.sources.get(key, "built-in") == "built-in" and key in MSU_EXAMPLES and msu_hpcc(env):
            text += MSU_EXAMPLES[key]
        return text.format(**{k: values.get(k, "") for k in PLACEHOLDERS})

    def resource_defaults(self) -> Dict[str, Dict[str, Any]]:
        """``workflow.common.constants.SLURM_RESOURCE_DEFAULTS`` with this site's
        account and the [resources] keys someone set (property calc types take
        the properties level), in the site's memory form."""
        from mace.workflow.common.constants import SLURM_RESOURCE_DEFAULTS
        out = {}
        for k, v in SLURM_RESOURCE_DEFAULTS.items():
            kind = "prop" if k in ("BAND", "DOSS", "TRANSPORT", "CHARGE+POTENTIAL") else "scf"
            out[k] = overlay_resources({**v, "account": self.account}, self, kind)
        return out

    @property
    def queue_text(self) -> str:
        """The completion callback's queue options for these limits."""
        return f"--max-jobs {self.max_jobs} --reserve {self.reserve} --max-submit {min(DEFAULTS['max_submit'], self.max_jobs)}"


# ------------------------------------------------------------------ config
def config_path(env: Optional[Dict[str, str]] = None) -> Optional[Path]:
    """The site.toml MACE reads (None for MACE_SITE_CONFIG=none)."""
    env = os.environ if env is None else env
    raw = (env.get("MACE_SITE_CONFIG") or "").strip()
    if raw.lower() == "none":
        return None
    if raw:
        return Path(raw).expanduser()
    base = env.get("XDG_CONFIG_HOME") or os.path.join(env.get("HOME") or os.path.expanduser("~"), ".config")
    return Path(base) / "mace" / "site.toml"


def _bad(source: str, what: str, value: Any) -> SiteConfigError:
    return SiteConfigError(f"{source} is not a valid {what}: {value!r}; nothing was generated")


def is_crystal_module(name: str) -> bool:
    """A CRYSTAL module name: ``CRYSTAL/...`` as EasyBuild names it, or
    ``crystal/...`` (any case) as many hand-written modulefiles do."""
    return str(name).lower().startswith("crystal/")


def _check_modules(value: Any, source: str, *, crystal: bool) -> List[str]:
    mods = [value] if isinstance(value, str) else value
    if not isinstance(mods, list) or not all(isinstance(m, str) for m in mods):
        raise _bad(source, "module list (a name or a list of names)", value)
    limit = 6 if crystal else 10
    if (crystal and not 1 <= len(mods) <= limit) or len(mods) > limit:
        raise _bad(source, f"module list (1 to {limit} names)" if crystal else f"module list (at most {limit})", value)
    for m in mods:
        if not MODULE_RE.fullmatch(m):
            raise _bad(source, "module name (letters, digits, . _ + / - only)", m)
    if crystal and not is_crystal_module(mods[-1]):
        raise _bad(source, "CRYSTAL module list (the last entry must start with CRYSTAL/ or crystal/)", value)
    return list(mods)


def _check_message(value: Any, source: str) -> str:
    if not isinstance(value, str) or "'" in value or "\n" in value or len(value) > 400:
        raise _bad(source, "message (one line, no single quote)", value)
    names = set(re.findall(r"\{([^{}]*)\}", value))
    if names - PLACEHOLDERS:
        raise _bad(source, "message (placeholders {reason}, {module} and {host} only)", value)
    return value


def config_layers(env: Optional[Dict[str, str]] = None) -> List[Tuple[str, Path, bool]]:
    """The settings files MACE reads, lowest precedence first, as (tag,
    path, explicit): group files ('group': /etc/mace/site.toml, then
    $XDG_CONFIG_DIRS/mace/site.toml with the first directory winning, or
    instead the files named in MACE_SITE_GROUP_CONFIG, a colon list whose
    first entry wins; 'none' drops them), then the user's own ('site.toml').
    MACE_SITE_CONFIG=none drops every file. A file need not exist."""
    env = os.environ if env is None else env
    if (env.get("MACE_SITE_CONFIG") or "").strip().lower() == "none":
        return []
    layers: List[Tuple[str, Path, bool]] = []
    named = (env.get("MACE_SITE_GROUP_CONFIG") or "").strip()
    if named.lower() == "none":
        pass
    elif named:
        for p in reversed([p for p in named.split(":") if p.strip()]):
            layers.append(("group", Path(p.strip()).expanduser(), True))
    else:
        layers.append(("group", SYSTEM_FILE, False))
        dirs = [d for d in (env.get("XDG_CONFIG_DIRS") or "/etc/xdg").split(":") if d.strip()]
        for d in reversed(dirs):
            layers.append(("group", Path(d.strip()) / "mace" / "site.toml", False))
    user = config_path(env)
    if user is not None:
        layers.append(("site.toml", user, True))
    return layers


def _owner(uid: int) -> str:
    try:
        import pwd
        return pwd.getpwuid(uid).pw_name
    except (ImportError, KeyError):
        return str(uid)


def _check_permissions(site: Site, tag: str, path: Path, explicit: bool, st=None, dst=None) -> None:
    """A group file is code-adjacent input that every member runs: refuse it
    when someone other than its owner can change it, or when it sits in an
    implicit location and belongs to neither root nor this user. The user's
    own file only gets a warning (umask 002 is common on clusters)."""
    st = path.stat() if st is None else st
    if dst is None:
        # a symlink: the folder of the link and the folder of the file it
        # names (whoever can write either can swap the settings)
        real = path.resolve()
        dirs = [(path.parent, path.parent.stat())]
        if real.parent != path.parent:
            dirs.append((real.parent, real.parent.stat()))
    else:
        dirs = [(path.parent, dst)]
    me = os.getuid() if hasattr(os, "getuid") else -1
    mode = st.st_mode & 0o7777
    if tag != "group":
        if st.st_mode & 0o022:
            site.warnings.append(f"{path} is writable by group or others (mode {mode:04o}); "
                                 f"chmod 644 {path} so only you can change your settings")
        return
    owner = _owner(st.st_uid)
    fix = f"chmod 644 {path}; chmod 755 {path.parent}"
    if st.st_mode & 0o022:
        raise SiteConfigError(f"{path}: refused: the group settings file is writable by group or others "
                              f"(mode {mode:04o}, owner {owner}); its owner should run: {fix}")
    for folder, dst in dirs:
        if dst.st_mode & 0o022:
            sticky_root = bool(dst.st_mode & 0o1000) and dst.st_uid == 0 and st.st_uid in (0, me)
            if not sticky_root:
                raise SiteConfigError(f"{path}: refused: its directory {folder} is writable by group or "
                                      f"others (mode {dst.st_mode & 0o7777:04o}); its owner should run: "
                                      f"chmod 644 {path}; chmod 755 {folder}")
    if not explicit and st.st_uid not in (0, me):
        raise SiteConfigError(f"{path}: refused: a group settings file in a shared location must belong to "
                              f"root or to you (owner {owner}); name a file you trust with "
                              "MACE_SITE_GROUP_CONFIG")
    site.group_owners[str(path)] = owner


def _read_toml(path: Path) -> Dict[str, Any]:
    try:
        import tomllib
    except ImportError:          # Python < 3.11: never fall back to the built-in values
        raise SiteConfigError(f"{path}: MACE needs Python 3.11 or newer to read site settings (this is "
                              f"Python {sys.version_info.major}.{sys.version_info.minor}); nothing was generated")
    try:
        return tomllib.loads(path.read_text())
    except OSError as exc:
        raise SiteConfigError(f"{path}: cannot read it: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        line = re.search(r"line (\d+)", str(exc))
        raise SiteConfigError(f"{path}:{line.group(1) if line else '?'}: {exc}") from exc


def load(env: Optional[Dict[str, str]] = None) -> Site:
    """The site settings: built-in, then group files, then site.toml, then
    the environment (per key). Raises SiteConfigError (with the source and
    the value) on any invalid value, on a group file others can change, and
    when a settings file exists but this Python cannot read TOML; unknown
    keys only give a warning."""
    env = os.environ if env is None else env
    site = Site()
    for k in ("account", "partition", "crystal", "python", "gateway_message", "module_message",
              "crystal_minimum", "crystal_maximum", "max_jobs", "reserve"):
        site.sources[k] = "built-in"
    site.path = config_path(env)
    for tag, path, explicit in config_layers(env):
        try:
            present = path.is_file()
        except OSError:
            present = False
        if not present:
            if tag == "group" and explicit:
                site.warnings.append(f"MACE_SITE_GROUP_CONFIG names {path}, which does not exist")
            continue
        _check_permissions(site, tag, path, explicit)
        data = _read_toml(path)
        if tag == "site.toml":
            site.exists = True
        else:
            site.group_paths.append(path)
        _apply_toml(site, data, path, tag)
    _apply_env(site, env)
    if site.srun_mpi and site.launcher != "srun":
        site.warnings.append(f"[mpi] srun_mpi = \"{site.srun_mpi}\" is ignored: it applies only with "
                             f"[mpi] launcher = \"srun\" (the launcher is {site.launcher})")
    return site


def load_tables(env: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, Any], Optional[Path], str]:
    """(every table of every settings file merged per key, lowest layer
    first; the user's site.toml; its text) for the tables other parts of
    MACE own (``mace cluster``: [cluster] [quota] [scan] [devnodes])."""
    env = os.environ if env is None else env
    merged: Dict[str, Any] = {}
    user_path, user_text = config_path(env), ""
    checked = Site()
    for tag, path, explicit in config_layers(env):
        try:
            present = path.is_file()
        except OSError:
            present = False
        if not present:
            continue
        # a group file others can change is refused here as in load()
        _check_permissions(checked, tag, path, explicit)
        data = _read_toml(path)
        if tag == "site.toml":
            user_text = path.read_text()
        _merge(merged, data)
    return merged, user_path, user_text


def _merge(into: Dict[str, Any], data: Dict[str, Any]) -> None:
    for k, v in data.items():
        if isinstance(v, dict) and isinstance(into.get(k), dict):
            _merge(into[k], v)
        else:
            into[k] = json.loads(json.dumps(v, default=str)) if isinstance(v, (dict, list)) else v


def _set(site: Site, key: str, value: Any, tag: str, origin: str) -> None:
    if hasattr(site, key) and key not in ("res_raw",):
        setattr(site, key, value)
    site.sources[key] = tag
    site.origins[key] = origin


# ------------------------------------------------------------------ validators (load and plan replay)
def _valid_int(value: Any, source: str, key: str) -> int:
    lo, hi = INT_LIMITS[key]
    if isinstance(value, str) and re.fullmatch(r"[0-9]{1,5}", value):
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise _bad(source, f"number ({lo} to {hi})", value)
    return value


def _valid_mem(value: Any, source: str) -> str:
    if not isinstance(value, str) or not MEM_RE.fullmatch(value):
        raise _bad(source, "memory size (a number with K, M, G or T: \"3900M\", \"250G\")", value)
    return value


def _valid_qos(value: Any, source: str) -> str:
    if not isinstance(value, str) or not QOS_RE.fullmatch(value):
        raise _bad(source, "QOS (letters, digits, . _ - only)", value)
    return value


def _valid_constraint(value: Any, source: str) -> str:
    if not isinstance(value, str) or not CONSTRAINT_RE.fullmatch(value):
        raise _bad(source, "constraint (node features with & | , [ ] * ! ( ) : only)", value)
    return value


def _valid_extra(value: Any, source: str) -> List[str]:
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list) or not all(isinstance(i, str) for i in items) or len(items) > MAX_EXTRA:
        raise _bad(source, f"list of #SBATCH options (at most {MAX_EXTRA})", value)
    for item in items:
        m = EXTRA_RE.fullmatch(item)
        if not m:
            raise _bad(source, "#SBATCH option (--name or --name=value; letters, digits, . _ : , @ % + [ ] - "
                               "in the value; no spaces or quotes)", item)
        # sbatch also takes an unambiguous prefix of a long option (--part=x)
        hit = m.group(1) if m.group(1) in EXTRA_REFUSED else next(
            (r for r in EXTRA_REFUSED if r.startswith(m.group(1))), None)
        if hit:
            raise SiteConfigError(f"{source}: MACE writes --{hit} itself; use "
                                  f"{EXTRA_REFUSED[hit]} instead (not {item!r}); nothing was generated")
    return list(items)


def _valid_launcher(value: Any, source: str) -> str:
    if value not in LAUNCHERS:
        if isinstance(value, str) and ("{" in value or " " in value):
            raise SiteConfigError(f"{source} is not a valid launcher: {value!r}; custom launchers are planned "
                                  "for MACE 1.2; use mpirun or srun; nothing was generated")
        raise _bad(source, "launcher (mpirun or srun)", value)
    return value


def _valid_srun_mpi(value: Any, source: str) -> str:
    if value not in SRUN_MPI:
        raise _bad(source, "srun MPI type (" + ", ".join(SRUN_MPI) + ")", value)
    return value


def env_name_refused(name: str) -> bool:
    return name in REFUSED_ENV_NAMES or name.startswith(REFUSED_ENV_PREFIXES)


def _valid_env(value: Any, source: str) -> List[str]:
    if not isinstance(value, list) or not all(isinstance(i, str) for i in value) or len(value) > MAX_ENV:
        raise _bad(source, f"list of NAME=value (at most {MAX_ENV})", value)
    for item in value:
        name, eq, val = item.partition("=")
        if not eq or not ENV_NAME_RE.fullmatch(name):
            raise _bad(source, "NAME=value (a name of letters, digits and _)", item)
        if env_name_refused(name):
            raise SiteConfigError(f"{source}: {name} cannot be set from site settings (it changes how the "
                                  f"job script or MACE itself runs): {item!r}; nothing was generated")
        if not ENV_VALUE_RE.fullmatch(val):
            raise _bad(source, "value (letters, digits and . _ / : , = + @ % ^ - only; no spaces, quotes "
                               "or shell characters)", item)
    return list(value)


def _no_dotdot(path: str) -> bool:
    return ".." not in path.split("/")


def _valid_bin_dir(value: Any, source: str) -> str:
    if not isinstance(value, str) or not BIN_DIR_RE.fullmatch(value) or not _no_dotdot(value):
        raise _bad(source, "directory (an absolute path of letters, digits and . _ + @ / -; no ..)", value)
    return value.rstrip("/") or "/"


def _valid_exe(value: Any, source: str) -> str:
    if not isinstance(value, str) or not EXE_RE.fullmatch(value) or value in (".", ".."):
        raise _bad(source, "program name (letters, digits and . _ + - only)", value)
    return value


def _valid_scratch(value: Any, source: str) -> str:
    if not isinstance(value, str) or not SCRATCH_RE.fullmatch(value) or not _no_dotdot(value):
        raise _bad(source, "directory (/... or $HOME/...; $USER, $HOME and $SLURM_JOB_ID allowed; no ..)", value)
    rest = SCRATCH_TOKEN_RE.sub("", value)
    if any(c in rest for c in "${}"):
        raise _bad(source, "directory (only $USER, $HOME and $SLURM_JOB_ID may be used)", value)
    return value.rstrip("/") or "/"


def _valid_bool(value: Any, source: str) -> bool:
    if isinstance(value, str) and value.strip().lower() in ("1", "true", "yes", "0", "false", "no"):
        return value.strip().lower() in ("1", "true", "yes")
    if not isinstance(value, bool):
        raise _bad(source, "true or false", value)
    return value


def parse_duration(value: Any, source: str) -> int:
    """'90s', '30m', '4h', 'H:MM:SS' or a number of seconds."""
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    t = value.strip().lower() if isinstance(value, str) else ""
    m = re.fullmatch(r"(\d{1,7})\s*([smh]?)", t)
    if m:
        secs = int(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600}[m.group(2)]
    else:
        m = re.fullmatch(r"(\d{1,4}):([0-5]?\d):([0-5]?\d)", t)
        secs = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) if m else 0
    if secs <= 0:
        raise _bad(source, "time (90s, 30m, 4h or H:MM:SS)", value)
    return secs


def _apply_resources(site: Site, table: Dict[str, Any], level: str, src: str, tag: str, origin: str) -> None:
    name = "[resources]" if not level else "[resources.properties]"
    if "mem" in table and "mem_per_cpu" in table:
        raise SiteConfigError(f"{src}: {name} sets mem and mem_per_cpu; set mem or mem_per_cpu, not both; "
                              "nothing was generated")
    raw = site.res_raw.setdefault(level, {})
    for k in ("ntasks", "nodes", "cpus_per_task"):
        if k in table:
            raw[k] = _valid_int(table[k], f"{src}: {name} {k}", k)
            site.sources[f"resources.{level + '.' if level else ''}{k}"] = tag
            site.origins[f"resources.{level + '.' if level else ''}{k}"] = origin
    for k, other in (("mem", "mem_per_cpu"), ("mem_per_cpu", "mem")):
        if k in table:
            raw[k] = _valid_mem(table[k], f"{src}: {name} {k}")
            raw.pop(other, None)            # one memory form per level: the higher layer's
            key = f"resources.{level + '.' if level else ''}"
            site.sources.pop(key + other, None)
            site.origins.pop(key + other, None)
            site.sources[key + k], site.origins[key + k] = tag, origin


def _apply_toml(site: Site, data: Dict[str, Any], path: Path, tag: str = "site.toml") -> None:
    unknown = []
    for section, keys in data.items():
        if section in FOREIGN_TABLES:
            continue
        if section not in KNOWN or not isinstance(keys, dict):
            unknown.append(section)
            continue
        for k in keys:
            if k not in KNOWN[section]:
                unknown.append(f"{section}.{k}")
            elif section == "resources" and k == "properties":
                if not isinstance(keys[k], dict):
                    unknown.append("resources.properties")
                    continue
                for kk in keys[k]:
                    if kk not in RESOURCE_KEYS:
                        unknown.append(f"resources.properties.{kk}")
    if unknown:
        site.warnings.append(f"{path}: unknown keys ignored: {', '.join(sorted(unknown))}")

    def table(name):
        return data.get(name) if isinstance(data.get(name), dict) else {}
    slurm, mods, local = table("slurm"), table("modules"), table("local_run")
    src = f"{path}"
    origin = str(path)
    if "account" in slurm:
        v = slurm["account"]
        if not isinstance(v, str) or not ACCOUNT_RE.fullmatch(v):
            raise _bad(f"{src}: [slurm] account", "account (letters, digits, . _ - only)", v)
        _set(site, "account", v, tag, origin)
    if "partition" in slurm:
        v = slurm["partition"]
        if not isinstance(v, str) or (v and not PARTITION_RE.fullmatch(v)):
            raise _bad(f"{src}: [slurm] partition", "partition (letters, digits, . _ -, comma-separated)", v)
        _set(site, "partition", v or None, tag, origin)
    if "qos" in slurm:
        v = slurm["qos"]
        _set(site, "qos", _valid_qos(v, f"{src}: [slurm] qos") if v != "" else None, tag, origin)
    if "constraint" in slurm:
        v = slurm["constraint"]
        _set(site, "constraint", _valid_constraint(v, f"{src}: [slurm] constraint") if v != "" else None,
             tag, origin)
    if "extra" in slurm:
        _set(site, "extra", _valid_extra(slurm["extra"], f"{src}: [slurm] extra"), tag, origin)
    if "crystal" in mods:
        _set(site, "crystal", _check_modules(mods["crystal"], f"{src}: [modules] crystal", crystal=True), tag, origin)
    for key in ("crystal_minimum", "crystal_maximum"):
        if key in mods:
            v = mods[key]
            v = str(v) if isinstance(v, int) and not isinstance(v, bool) else v
            if not isinstance(v, str) or not MAJOR_RE.fullmatch(v):
                raise _bad(f"{src}: [modules] {key}", "major version (1 to 3 digits)", mods[key])
            _set(site, key, int(v), tag, origin)
    if site.crystal_minimum is not None and site.crystal_maximum is not None \
            and site.crystal_minimum > site.crystal_maximum:
        raise SiteConfigError(f"{src}: [modules] crystal_minimum ({site.crystal_minimum}) is above crystal_maximum "
                              f"({site.crystal_maximum}); nothing was generated")
    if "python" in mods:
        _set(site, "python", _check_modules(mods["python"], f"{src}: [modules] python", crystal=False), tag, origin)
    if "purge" in mods:
        _set(site, "module_purge", _valid_bool(mods["purge"], f"{src}: [modules] purge"), tag, origin)
    queue = table("queue")
    for key, low in (("max_jobs", 1), ("reserve", 0)):
        if key in queue:
            v = queue[key]
            if isinstance(v, bool) or not isinstance(v, int) or not low <= v <= 100000:
                raise _bad(f"{src}: [queue] {key}", f"number ({low} to 100000)", v)
            _set(site, key, v, tag, origin)
    for key in ("gateway_message", "module_message"):
        if key in local:
            _set(site, key, _check_message(local[key], f"{src}: [local_run] {key}"), tag, origin)
    if "max_time" in local:
        _set(site, "run_max_time", parse_duration(local["max_time"], f"{src}: [local_run] max_time"), tag, origin)
    res = table("resources")
    if res:
        _apply_resources(site, {k: v for k, v in res.items() if k != "properties"}, "", src, tag, origin)
        if isinstance(res.get("properties"), dict):
            _apply_resources(site, res["properties"], "properties", src, tag, origin)
    mpi = table("mpi")
    if "launcher" in mpi:
        _set(site, "launcher", _valid_launcher(mpi["launcher"], f"{src}: [mpi] launcher"), tag, origin)
    if "srun_mpi" in mpi:
        _set(site, "srun_mpi", _valid_srun_mpi(mpi["srun_mpi"], f"{src}: [mpi] srun_mpi"), tag, origin)
    if "env" in mpi:
        _set(site, "mpi_env", _valid_env(mpi["env"], f"{src}: [mpi] env"), tag, origin)
    paths = table("paths")
    if "scratch" in paths:
        v = paths["scratch"]
        _set(site, "scratch", _valid_scratch(v, f"{src}: [paths] scratch") if v != "" else None, tag, origin)
    crys = table("crystal")
    if "bin_dir" in crys:
        v = crys["bin_dir"]
        _set(site, "crystal_bin_dir", _valid_bin_dir(v, f"{src}: [crystal] bin_dir") if v != "" else None,
             tag, origin)
    for key, kind in (("scf", "scf"), ("properties", "prop")):
        if key in crys:
            site.crystal_exe[kind] = _valid_exe(crys[key], f"{src}: [crystal] {key}")
            site.sources[f"crystal_exe.{kind}"], site.origins[f"crystal_exe.{kind}"] = tag, origin


ENV_RESOURCES = (("MACE_SLURM_NTASKS", "ntasks"), ("MACE_SLURM_NODES", "nodes"),
                 ("MACE_SLURM_CPUS_PER_TASK", "cpus_per_task"), ("MACE_SLURM_MEM_PER_CPU", "mem_per_cpu"),
                 ("MACE_SLURM_MEM", "mem"))


def _apply_env(site: Site, env: Dict[str, str]) -> None:
    """The environment wins over every file (MACE_SLURM_*, MACE_MPI_LAUNCHER, ...)."""
    def get(name):
        v = env.get(name)
        return v if v is not None and v != "" else None
    acct = get("MACE_SLURM_ACCOUNT")
    if acct is not None:
        if not ACCOUNT_RE.fullmatch(acct):
            raise SiteConfigError(f"MACE_SLURM_ACCOUNT is not a valid account (letters, digits, . _ - only): "
                                  f"{acct!r}; nothing was generated")
        _set(site, "account", acct, "env MACE_SLURM_ACCOUNT", "MACE_SLURM_ACCOUNT")
    part = get("MACE_SLURM_PARTITION")
    if part is not None:
        if not PARTITION_RE.fullmatch(part):
            raise SiteConfigError(f"MACE_SLURM_PARTITION is not a valid partition (letters, digits, . _ -, "
                                  f"comma-separated): {part!r}; nothing was generated")
        _set(site, "partition", part, "env MACE_SLURM_PARTITION", "MACE_SLURM_PARTITION")
    pin = get("MACE_CRYSTAL_MODULE")
    if pin is not None:
        if pin.strip().lower() == "none":
            site.pin_none = True
        else:
            mods = pin.split(",")
            if not all(MODULE_RE.fullmatch(m) for m in mods) or not 1 <= len(mods) <= 6:
                raise SiteConfigError(f"MACE_CRYSTAL_MODULE is not a valid module name (letters, digits, "
                                      f". _ + / - only, comma-separated): {pin!r}; nothing was generated")
            site.pin = mods
    simple = (("MACE_SLURM_QOS", "qos", _valid_qos), ("MACE_SLURM_CONSTRAINT", "constraint", _valid_constraint),
              ("MACE_MPI_LAUNCHER", "launcher", _valid_launcher), ("MACE_SCRATCH_BASE", "scratch", _valid_scratch),
              ("MACE_CRYSTAL_BIN_DIR", "crystal_bin_dir", _valid_bin_dir),
              ("MACE_MODULE_PURGE", "module_purge", _valid_bool))
    for name, key, check in simple:
        v = get(name)
        if v is not None:
            _set(site, key, check(v, name), f"env {name}", name)
    extra = get("MACE_SLURM_EXTRA")
    if extra is not None:
        items = [i.strip() for i in re.split(r",(?=\s*--)", extra) if i.strip()]
        _set(site, "extra", _valid_extra(items, "MACE_SLURM_EXTRA"), "env MACE_SLURM_EXTRA", "MACE_SLURM_EXTRA")
    t = get("MACE_RUN_MAX_TIME")
    if t is not None:
        _set(site, "run_max_time", parse_duration(t, "MACE_RUN_MAX_TIME"), "env MACE_RUN_MAX_TIME", "MACE_RUN_MAX_TIME")
    for name, key in ENV_RESOURCES:
        v = get(name)
        if v is None:
            continue
        value = _valid_mem(v, name) if key.startswith("mem") else _valid_int(v, name, key)
        drop = {"mem": "mem_per_cpu", "mem_per_cpu": "mem"}.get(key)
        for level in ("", "properties"):         # both job kinds
            raw = site.res_raw.setdefault(level, {})
            prefix = f"resources.{level + '.' if level else ''}"
            if level:
                raw.pop(key, None)
                site.sources.pop(prefix + key, None)
                site.origins.pop(prefix + key, None)
            else:
                raw[key] = value
                site.sources[prefix + key], site.origins[prefix + key] = f"env {name}", name
            if drop:
                raw.pop(drop, None)
                site.sources.pop(prefix + drop, None)
                site.origins.pop(prefix + drop, None)


def overlay_resources(resources: Dict[str, Any], site: "Site", kind: str) -> Dict[str, Any]:
    """An editor's resource dict (``-i``, the workflow planner) with the
    [resources] keys someone set: ntasks, nodes, cpus_per_task and the
    memory in the site's form (``memory_per_cpu`` or ``memory``, never both,
    plus ``mem_form``). Unchanged when nothing is set."""
    out = dict(resources)
    s = site.resources_set(kind)
    for k in ("ntasks", "nodes", "cpus_per_task"):
        if k in s:
            out[k] = s[k]
    if "mem_form" in s:
        out.pop("memory", None)
        out.pop("memory_per_cpu", None)
        out["memory" if s["mem_form"] == "mem" else "memory_per_cpu"] = s["mem"]
        out["mem_form"] = s["mem_form"]
    return out


def memory_line(line: str, resources: Dict[str, Any]) -> Optional[str]:
    """The memory directive an editor writes for a template line that holds
    either memory form: the form ``resources`` carries when the site chose one
    (``mem_form``), else None (the editor's own rule: the line's form)."""
    s = line.rstrip()
    if not (s.startswith("echo '#SBATCH --mem=") or s.startswith("echo '#SBATCH --mem-per-cpu=")):
        return None
    form = resources.get("mem_form")
    if form == "mem" and resources.get("memory"):
        return f"echo '#SBATCH --mem={resources['memory']}' >> $1.sh"
    if form == "mem_per_cpu" and resources.get("memory_per_cpu"):
        return f"echo '#SBATCH --mem-per-cpu={resources['memory_per_cpu']}' >> $1.sh"
    return None



# ------------------------------------------------------------------ CRYSTAL module
@dataclass
class Resolution:
    modules: List[str]
    source: str
    note: str = ""
    rejected: List[Tuple[str, str]] = field(default_factory=list)
    root: Optional[str] = None
    root_from_path: bool = False
    probed_at: Optional[float] = None
    other_major: Optional[str] = None

    @property
    def module(self) -> str:
        # no module: CRYSTAL from [crystal] bin_dir
        return self.modules[-1] if self.modules else ""

    def record(self) -> Dict[str, Any]:
        return {"modules": list(self.modules), "module": self.module, "source": self.source, "note": self.note,
                "rejected": [list(r) for r in self.rejected], "root_from_path": self.root_from_path,
                "other_major": self.other_major}


_MEMO: Dict[str, Resolution] = {}


def _hostname() -> str:
    return socket.gethostname().split(".")[0] or "this host"


# ------------------------------------------------------------------ this cluster
def host_names(env: Optional[Dict[str, str]] = None) -> Tuple[str, str]:
    """(short, fully qualified) name of this host; ``MACE_HOSTNAME`` replaces
    both (a test, or a host whose name does not say where it is)."""
    env = os.environ if env is None else env
    fake = (env.get("MACE_HOSTNAME") or "").strip().lower()
    if fake:
        return fake.split(".")[0], fake
    try:
        fqdn = socket.getfqdn().lower()
    except OSError:
        fqdn = ""
    short = socket.gethostname().lower()
    return short.split(".")[0], (fqdn if "." in fqdn else short)


def msu_hpcc(env: Optional[Dict[str, str]] = None) -> bool:
    """True only where this is verifiably MSU HPCC: its domain, one of its
    node names, or its profile script. Gates the MSU examples in messages and
    the MSU node-exclusion choices."""
    short, fqdn = host_names(env)
    if fqdn.endswith(".msu.edu") or MSU_HOST_RE.match(short):
        return True
    env = os.environ if env is None else env
    return not env.get("MACE_HOSTNAME") and os.path.exists(MSU_PROFILE)


def slurm_here(tool: str = "scontrol", env: Optional[Dict[str, str]] = None) -> bool:
    import shutil
    env = os.environ if env is None else env
    return shutil.which(tool, path=env.get("PATH")) is not None


# what says SLURM is here (any of its client commands)
SLURM_CLIENTS = ("sbatch", "squeue", "scontrol", "sacctmgr")
# how a job script without an account line is described (mace site, the
# warning, the GUI's Submit preview)
NO_ACCOUNT_TEXT = ("no account line (SLURM's default account); set yours with MACE_SLURM_ACCOUNT "
                   "or site.toml")


def keeps_builtin_account(env: Optional[Dict[str, str]] = None) -> bool:
    """Where membership of a built-in (MSU HPCC) account cannot be checked,
    it is still written: on MSU HPCC, and on a host with no SLURM at all (a
    workstation, CI: its scripts are MSU HPCC's). Any other host with SLURM
    gets no account line (SLURM's default account)."""
    return msu_hpcc(env) or not any(slurm_here(t, env) for t in SLURM_CLIENTS)


def msu_choices(env: Optional[Dict[str, str]] = None) -> bool:
    """The MSU HPCC node-exclusion choices (AMD20 nodes, ...) apply: on MSU
    HPCC, and where there is no SLURM to ask (a workstation keeps them)."""
    return not slurm_here("scontrol", env) or msu_hpcc(env)


def _user(env: Dict[str, str]) -> str:
    user = env.get("USER") or env.get("LOGNAME") or ""
    if not user:
        try:
            import getpass
            user = getpass.getuser()
        except Exception:
            user = ""
    return user


_SLURM_MEMO: Dict[str, Any] = {}


def _slurm_cached(key: str, env: Dict[str, str]) -> Optional[dict]:
    e = envmodules._read_cache(env).get(key)
    if isinstance(e, dict) and 0 <= time.time() - float(e.get("t", 0)) <= SLURM_CACHE_TTL_S:
        return e
    return None


def _slurm_run(argv: List[str], env: Dict[str, str]):
    import subprocess
    try:
        return subprocess.run(argv, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              timeout=SLURM_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return None


def slurm_associations(env: Optional[Dict[str, str]] = None, refresh: bool = False) -> Optional[Dict[str, Any]]:
    """{accounts, default}: this user's SLURM accounts and default account
    (``sacctmgr``, short timeout, cached a day), or None when they cannot be
    read here (no sacctmgr, no accounting, nothing recorded for the user)."""
    import shutil
    env = dict(os.environ if env is None else env)
    user = _user(env)
    exe = shutil.which("sacctmgr", path=env.get("PATH"))
    key = json.dumps(["SLURM-assoc", user, exe or ""])
    if not refresh:
        if key in _SLURM_MEMO:
            return _SLURM_MEMO[key]
        hit = _slurm_cached(key, env)
        if hit and hit.get("accounts"):
            _SLURM_MEMO[key] = {"accounts": list(hit["accounts"]), "default": hit.get("default")}
            return _SLURM_MEMO[key]
    accounts = None
    try:
        from mace.cluster import limits as _limits       # Area F, when present
        accounts = _limits.associations()
    except Exception:
        accounts = None
    if accounts is None and exe and user:
        r = _slurm_run([exe, "-nP", "show", "assoc", f"user={user}", "format=account"], env)
        if r is not None and r.returncode == 0:
            accounts = [l.strip().split("|")[0] for l in r.stdout.splitlines() if l.strip().split("|")[0]]
    if not accounts:
        _SLURM_MEMO[key] = None
        return None
    default = None
    if exe and user:
        r = _slurm_run([exe, "-nP", "show", "user", user, "format=defaultaccount"], env)
        if r is not None and r.returncode == 0:
            first = [l.strip().split("|")[0] for l in r.stdout.splitlines() if l.strip()]
            default = first[0] if first and ACCOUNT_RE.fullmatch(first[0]) else None
    out = {"accounts": sorted(set(accounts)), "default": default}
    envmodules.cache_put(key, {**out, "name": "SLURM"}, env)
    _SLURM_MEMO[key] = out
    return out


def _parse_submit_limit(text: str, user: str, account: Optional[str]) -> Optional[int]:
    try:
        from mace.submission.job_array import parse_submit_slots
    except ImportError:          # pragma: no cover - imported as a bare module
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location("mace_job_array", SUBMISSION_DIR / "job_array.py")
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        parse_submit_slots = mod.parse_submit_slots
    slots = parse_submit_slots(text, user, account)
    return slots[0] if slots and slots[0] > 0 else None


def submit_limit(account: Optional[str], env: Optional[Dict[str, str]] = None, refresh: bool = False) -> Optional[int]:
    """This user's MaxSubmitJobs for ``account`` (its default association
    when None) from ``scontrol show assoc_mgr`` (cached a day), or None."""
    import shutil
    env = dict(os.environ if env is None else env)
    user = _user(env)
    exe = shutil.which("scontrol", path=env.get("PATH"))
    key = json.dumps(["SLURM-maxsubmit", user, account or "", exe or ""])
    if not refresh:
        if key in _SLURM_MEMO:
            return _SLURM_MEMO[key]
        hit = _slurm_cached(key, env)
        if hit and isinstance(hit.get("limit"), int):
            _SLURM_MEMO[key] = hit["limit"]
            return hit["limit"]
    limit = None
    if exe and user:
        argv = [exe, "show", "assoc_mgr", f"users={user}", "flags=assoc"]
        if account:
            argv.insert(4, f"accounts={account}")
        r = _slurm_run(argv, env)
        if r is not None and r.returncode == 0:
            limit = _parse_submit_limit(r.stdout, user, account)
    if limit is not None:
        envmodules.cache_put(key, {"limit": limit, "name": "SLURM"}, env)
    _SLURM_MEMO[key] = limit
    return limit


def resolve_account(site: Site, env: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], str, str]:
    """(account or None for no -A line, status, note). A user's own account
    (env, site.toml) is used as given. The built-in one is kept for a member
    of it, and where membership cannot be checked only on MSU HPCC or where
    there is no SLURM (keeps_builtin_account); otherwise it is left out, so
    SLURM charges the user's default account."""
    env = dict(os.environ if env is None else env)
    if site.sources.get("account", "built-in") != "built-in" or site.account != DEFAULTS["account"]:
        return site.account, "set", ""
    if env.get("SLURM_JOB_ACCOUNT") == DEFAULTS["account"]:     # a callback in a job charged to it
        return site.account, "member", ""
    assoc = slurm_associations(env)
    if assoc is None:
        if keeps_builtin_account(env):
            return site.account, "not checked", ""
        where = config_path(env) or "~/.config/mace/site.toml"
        return None, "unreadable", (f"MACE: SLURM accounting cannot be read here, so job scripts get "
                                    f"{NO_ACCOUNT_TEXT} ([slurm] account in {where}).")
    if site.account in assoc["accounts"]:
        return site.account, "member", ""
    dflt = assoc.get("default")
    where = config_path(env) or "~/.config/mace/site.toml"
    note = (f"MACE: no SLURM account set; job scripts carry no #SBATCH -A, so SLURM charges your default "
            f"account{f' ({dflt})' if dflt else ''}. To choose one, set [slurm] account in {where} "
            "or MACE_SLURM_ACCOUNT.")
    return None, "default", note


def _python_cache_key(mods: List[str], env: Dict[str, str]) -> str:
    return envmodules.cache_key("PYTHON", mods, env.get("BASH_ENV") or "", env=env)


def resolve_python(site: Site, env: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], str, str]:
    """(executable or None, status, note): None keeps the Python module lines
    (a user's own list, or the built-in ones where they load or where there
    is no module system); otherwise the Python that runs MACE, which the job
    script then calls for its completion step."""
    env = dict(os.environ if env is None else env)
    mods = list(site.python)
    if site.sources.get("python", "built-in") != "built-in" or mods != DEFAULTS["python"]:
        return None, "set", ""
    if envmodules.module_system(env) is None:
        return None, "unchecked", ""
    if msu_hpcc(env):
        # their own cluster: they load on its compute and dev nodes, even
        # where this node (the gateway) serves no modules
        return None, "MSU HPCC", ""
    if env.get("SLURM_JOB_ID"):                  # a callback: what the job itself loaded, never a probe
        loaded = [m for m in (env.get("LOADEDMODULES") or "").split(":") if m]
        if all(m in loaded for m in mods):
            return None, "inherited", ""
        ok = False
    else:
        key = _python_cache_key(mods, env)
        hit = _slurm_cached(key, env) if key not in _SLURM_MEMO else {"ok": _SLURM_MEMO[key]}
        if hit and isinstance(hit.get("ok"), bool):
            ok = hit["ok"]
        else:
            ok = envmodules.load_modules_env(mods, env, purge=True) is not None
            envmodules.cache_put(key, {"ok": ok, "name": "PYTHON"}, env)
        _SLURM_MEMO[key] = ok
        if ok:
            return None, "loads here", ""
    exe = os.path.abspath(sys.executable or "")
    if not SAFE_EXE_RE.fullmatch(exe):
        return None, "kept", (f"MACE: the built-in Python modules do not load here and the Python that runs MACE "
                              f"({exe}) cannot be written into a job script; set [modules] python in site.toml")
    note = (f"MACE: the built-in Python modules do not load here; the job scripts run their completion step "
            f"with the Python that runs MACE ({exe}). To load your cluster's Python module instead, "
            "set [modules] python in site.toml.")
    return exe, "this Python", note


def queue_from_limit(limit: int) -> Tuple[int, int]:
    """(max_jobs, reserve) for a MaxSubmitJobs: a 5% reserve (at least one)."""
    reserve = max(1, limit // 20)
    return limit - reserve, reserve


def resolve_queue(site: Site, account: Optional[str], env: Optional[Dict[str, str]] = None) -> Tuple[int, int, str]:
    """(max_jobs, reserve, source) of the completion callback: the user's
    own settings (site.toml, a group file), else the user's MaxSubmitJobs
    (minus a 5% reserve), else the built-in."""
    if site.is_set("max_jobs") or site.is_set("reserve"):
        return site.max_jobs, site.reserve, ("site.toml" if "site.toml" in (site.sources.get("max_jobs"),
                                                                           site.sources.get("reserve")) else "group")
    limit = submit_limit(account, env)
    if limit is None or limit < 2:
        return site.max_jobs, site.reserve, "built-in"
    mj, rs = queue_from_limit(limit)
    return mj, rs, f"MaxSubmitJobs {limit}"


def clamp_queue(max_jobs: int, reserve: int, account: Optional[str], env: Optional[Dict[str, str]] = None
                ) -> Tuple[int, int, str]:
    """(max_jobs, reserve, note): lowered to fit this user's MaxSubmitJobs when
    max_jobs + reserve would exceed it; never raised. Unknown limit: unchanged."""
    try:
        limit = submit_limit(account, env)
    except Exception:
        limit = None
    if limit is None or limit < 2 or max_jobs + reserve <= limit:
        return max_jobs, reserve, ""
    mj, rs = queue_from_limit(limit)
    return mj, rs, (f"MACE: your SLURM MaxSubmitJobs is {limit}; the workflow keeps at most {mj} jobs queued "
                    f"with {rs} slots in reserve (not {max_jobs} and {reserve}).")


def parse_walltime(value: str) -> Optional[int]:
    """SLURM time string to seconds; None for UNLIMITED or unparseable."""
    value = (value or "").strip()
    if not value or value.upper() in ("UNLIMITED", "INFINITE", "NONE", "N/A"):
        return None
    days, _, rest = value.rpartition("-") if "-" in value else ("0", "", value)
    try:
        nums = [int(p) for p in rest.split(":")]
        days_n = int(days)
    except ValueError:
        return None
    if "-" in value:
        nums += [0] * (3 - len(nums))
        h, m, sec = nums[:3]
    elif len(nums) == 1:
        h, m, sec = 0, nums[0], 0
    elif len(nums) == 2:
        h, m, sec = 0, nums[0], nums[1]
    else:
        h, m, sec = nums[:3]
    return days_n * 86400 + h * 3600 + m * 60 + sec


def partition_limit(partition: Optional[str], env: Optional[Dict[str, str]] = None,
                    refresh: bool = False) -> Tuple[Optional[int], Optional[str]]:
    """(MaxTime in seconds, partition name) of ``partition`` (a comma list:
    the longest of them) or, for None, of the cluster's default partition,
    from ``scontrol show partition`` (cached a day); (None, None) when SLURM
    does not say or sets no limit."""
    import shutil
    env = dict(os.environ if env is None else env)
    exe = shutil.which("scontrol", path=env.get("PATH"))
    key = json.dumps(["SLURM-maxtime", partition or "", exe or ""])
    if not refresh:
        if key in _SLURM_MEMO:
            return tuple(_SLURM_MEMO[key])
        hit = _slurm_cached(key, env)
        if hit and isinstance(hit.get("limit"), int):
            _SLURM_MEMO[key] = (hit["limit"], hit.get("partition"))
            return hit["limit"], hit.get("partition")
    found: Tuple[Optional[int], Optional[str]] = (None, None)
    if exe:
        r = _slurm_run([exe, "show", "partition", "-a", "-o"], env)
        if r is not None and r.returncode == 0:
            parts = {}
            default = None
            for line in r.stdout.splitlines():
                name = re.search(r"\bPartitionName=(\S+)", line)
                if not name:
                    continue
                mt = re.search(r"\bMaxTime=(\S+)", line)
                parts[name.group(1)] = parse_walltime(mt.group(1)) if mt else None
                if re.search(r"\bDefault=YES\b", line):
                    default = name.group(1)
            wanted = partition.split(",") if partition else ([default] if default else [])
            limits = [(parts.get(p), p) for p in wanted if p in parts]
            if limits and all(l is not None for l, _ in limits):
                found = max(limits, key=lambda t: t[0])
    if found[0] is not None:
        envmodules.cache_put(key, {"limit": found[0], "partition": found[1], "name": "SLURM"}, env)
    _SLURM_MEMO[key] = found
    return found


def walltime_text(seconds: int) -> str:
    d, rem = divmod(int(seconds), 86400)
    h, rem = divmod(rem, 3600)
    m, sec = divmod(rem, 60)
    return f"{d}-{h:02d}:{m:02d}:{sec:02d}"


def resolve(site: Site, env: Optional[Dict[str, str]] = None, out=print) -> Site:
    """``site`` with the built-in values checked against this cluster
    (account, Python modules, queue limits); each change is said once."""
    env = dict(os.environ if env is None else env)
    acct, a_status, a_note = resolve_account(site, env)
    site.account = acct
    exe, p_status, p_note = resolve_python(site, env)
    site.python_executable = exe
    mj, rs, q_source = resolve_queue(site, acct, env)
    site.max_jobs, site.reserve = mj, rs
    # (not on MSU HPCC: a submit plugin routes its jobs by walltime, so its
    # default partition's MaxTime is not the limit - recovery/slurm_limits.py)
    cap, part = (None, None) if msu_hpcc(env) else partition_limit(site.partition, env)
    w_note = ""
    if cap is not None and cap < WALLTIME_LINES["scf"][2]:
        site.walltime_cap, site.cap_partition = cap, part
        w_note = (f"MACE: partition {part} allows at most {walltime_text(cap)}; job scripts whose default "
                  f"walltime is longer ask for {walltime_text(cap)}.")
    site.checks = {"account": a_status, "python": p_status, "queue": q_source,
                   "walltime_cap": walltime_text(cap) if cap is not None else None}
    site.notes = [n for n in (a_note, p_note, w_note) if n]
    for n in site.notes:
        say_once(n, out)
    return site


def default_walltime(kind: str, default: str, env: Optional[Dict[str, str]] = None, out=print) -> str:
    """``default`` (the template's walltime for ``kind``), or the partition's
    MaxTime when that is shorter - what the template's own -t line becomes
    (customize_generator_text), for the job scripts `mace submit -i`,
    --walltime and --exclude-* write with a literal -t line."""
    try:
        site = load(env)
    except SiteConfigError:
        return default
    resolve(site, env, lambda t: out(f"  {t}"))
    want = parse_walltime(default)
    if site.walltime_cap and want and site.walltime_cap < want:
        return walltime_text(site.walltime_cap)
    return default


def walltime_over_limit(value: str, env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """A warning when a walltime the user gave is longer than the partition
    allows (it is written as given: SLURM decides), else None."""
    secs = parse_walltime(value or "")
    if secs is None or msu_hpcc(env):     # MSU HPCC routes jobs by walltime (see resolve)
        return None
    try:
        site = load(env)
    except SiteConfigError:
        return None
    cap, part = partition_limit(site.partition, env)
    if cap is None or secs <= cap:
        return None
    return (f"Warning: walltime {value} is longer than partition {part} allows ({walltime_text(cap)}); "
            f"written as given - SLURM may reject the job or leave it pending.")


def _inherited(site: Site, env: Dict[str, str], parent_module: Optional[str]) -> Resolution:
    job = env.get("SLURM_JOB_ID", "")
    loaded = [m for m in (env.get("LOADEDMODULES") or "").split(":") if m]
    crys = [i for i, m in enumerate(loaded) if is_crystal_module(m)]
    if crys:
        i = crys[-1]
        before = [m for m in loaded[:i] if m in site.crystal[:-1]]
        return Resolution(before + [loaded[i]], f"inherited from job {job}")
    if parent_module and MODULE_RE.fullmatch(parent_module):
        return Resolution([m for m in site.crystal[:-1]] + [parent_module], "recorded by the parent calculation")
    return Resolution(list(site.crystal), "inside a SLURM job: not checked")


def crystal_module(env: Optional[Dict[str, str]] = None, site: Optional[Site] = None,
                   parent_module: Optional[str] = None, refresh: bool = False) -> Resolution:
    """The CRYSTAL module (list) the job scripts load, and why (see the module docstring)."""
    env = dict(os.environ if env is None else env)
    site = load(env) if site is None else site
    if site.crystal_bin_dir and not site.pin and not site.is_set("crystal"):
        # CRYSTAL from a directory, no module: nothing to probe
        return Resolution([], "bin_dir", f"CRYSTAL from {site.crystal_bin_dir} ([crystal] bin_dir)")
    binaries = site.binaries()
    if env.get("SLURM_JOB_ID") and not site.pin:          # a completion callback: never a probe
        if site.pin_none:
            return Resolution(list(DEFAULTS["crystal"]), "MACE_CRYSTAL_MODULE=none")
        return _inherited(site, env, parent_module)
    if site.pin:
        return Resolution(list(site.pin), "MACE_CRYSTAL_MODULE (pinned, not checked)")
    if site.pin_none:
        return Resolution(list(DEFAULTS["crystal"]), "MACE_CRYSTAL_MODULE=none")
    preferred = list(site.crystal)
    pref_source = {"site.toml": "site.toml", "group": "group"}.get(site.sources.get("crystal", ""), "default")
    if envmodules.module_system(env) is None:
        return Resolution(preferred, "unchecked")
    lo, hi = site.window
    memo_key = envmodules.cache_key("CRYSTAL", preferred, lo, hi, *([] if binaries == BINARIES else list(binaries)),
                                    env=env)
    if not refresh and memo_key in _MEMO:
        return _MEMO[memo_key]
    hit = None if refresh else envmodules.cache_get(memo_key, env, binaries)
    if hit:
        res = Resolution(list(hit["modules"]), hit["source"], hit.get("note", ""), [tuple(r) for r in hit.get("rejected", [])],
                         hit.get("root"), bool(hit.get("root_from_path")), hit.get("probed_at"))
        _MEMO[memo_key] = res
        return res
    host = _hostname()
    now = time.time()
    ok, root, from_path, why = envmodules.qualify(preferred, binaries, env)
    rejected: List[Tuple[str, str]] = []
    if ok:
        res = Resolution(preferred, pref_source, "", [], root, from_path, now)
    else:
        rejected.append((preferred[-1], why))
        # CRYSTAL/..., else a lower-case crystal/... (hand-written modulefiles)
        found = envmodules.candidates("CRYSTAL", env) or envmodules.avail_listing("crystal", env)
        names = [n for n in found if n != preferred[-1]]
        inside = [n for n in names if lo <= (envmodules.version_key(n) or (-1,))[0] <= hi]
        outside = [n for n in names if n not in inside]
        for n in outside:
            major = (envmodules.version_key(n) or ("?",))[0]
            rejected.append((n, f"major {major} outside {lo}..{hi}: set [modules] crystal_maximum to allow it"))
        res = None
        for n in envmodules.rank_candidates(inside, preferred[-1]):
            mods = preferred[:-1] + [n]
            q_ok, q_root, q_from, q_why = envmodules.qualify(mods, binaries, env)
            if q_ok:
                res = Resolution(mods, "discovered",
                                 f"{preferred[-1]} does not load on {host}; using {n} (best CRYSTAL "
                                 f"{lo if lo == hi else f'{lo}-{hi}'}.x module here with {binaries[0]} and {binaries[1]})",
                                 rejected, q_root, q_from, now)
                break
            rejected.append((n, q_why))
        if res is None:
            other = None
            for n in envmodules.rank_candidates(outside, preferred[-1])[:1]:
                if envmodules.qualify(preferred[:-1] + [n], binaries, env)[0]:
                    other = n
            note = (f"No CRYSTAL {lo if lo == hi else f'{lo}-{hi}'}.x module loads on {host} (this may be a login "
                    f"node); job scripts keep {preferred[-1]}")
            if other:
                note += (f". {other} loads here; MACE does not switch major versions on its own: set "
                         f"[modules] crystal_maximum = {envmodules.version_key(other)[0]} or [modules] crystal to use it")
            res = Resolution(preferred, "not found here", note, rejected, None, False, now, other)
    if res.source in ("default", "site.toml", "group", "discovered"):
        envmodules.cache_put(memo_key, {"modules": res.modules, "source": res.source, "note": res.note,
                                        "rejected": [list(r) for r in res.rejected], "root": res.root,
                                        "root_from_path": res.root_from_path, "probed_at": res.probed_at,
                                        "name": "CRYSTAL"}, env)
    _MEMO[memo_key] = res
    return res


def forget_crystal_cache(module: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> int:
    """Drop the cached CRYSTAL resolutions (those that chose ``module``, or all); a file edit only."""
    _MEMO.clear()
    return envmodules.cache_forget(lambda k, v: isinstance(v, dict) and v.get("name") == "CRYSTAL"
                                   and (module is None or module in (v.get("modules") or [])), env)


MISSING_BINARY_RE = re.compile(r"bin/(Pcrystal|Pproperties): No such file or directory")


def missing_binary_re(site: Optional[Site] = None) -> "re.Pattern":
    """What a job's .out says when its CRYSTAL executable is missing, for this
    site's executable names and [crystal] bin_dir."""
    if site is None or (site.binaries() == BINARIES and not site.crystal_bin_dir):
        return MISSING_BINARY_RE
    where = "bin/" if not site.crystal_bin_dir else f"(?:bin/|{re.escape(site.crystal_bin_dir)}/)"
    names = "|".join(re.escape(n) for n in site.binaries())
    return re.compile(f"{where}({names}): No such file or directory")


# ------------------------------------------------------------------ rewrite
def launch_block(kind: str, site: Site) -> Optional[List[str]]:
    """The lines that start CRYSTAL in a ``kind`` job script for this site's
    [mpi], [crystal] and cpus_per_task, or None for the template's own line.
    Under srun, `mace run` (no SLURM step to srun into) keeps mpirun."""
    cpt = site.resources_set(kind).get("cpus_per_task", 1)
    name = site.crystal_exe[kind]
    exe = f"{site.crystal_bin_dir}/{name}" if site.crystal_bin_dir else f"$EBROOTCRYSTAL/bin/{name}"
    default_exe = f"$EBROOTCRYSTAL/bin/{dict(scf=BINARIES[0], prop=BINARIES[1])[kind]}"
    if site.launcher == "mpirun" and not site.mpi_env and cpt <= 1 and exe == default_exe:
        return None
    out = [LAUNCH_COMMENT]
    out += [f"export {item}" for item in site.mpi_env]
    if cpt > 1:
        out.append("export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK")
    mpirun = f'I_MPI_HYDRA_BOOTSTRAP="ssh" mpirun -n "$SLURM_NTASKS" "{exe}" 2>&1 >& "$DIR/${{JOB}}.out"'
    if site.launcher == "mpirun":
        out.append(mpirun)
    else:
        srun = ("srun" + (f" --mpi={site.srun_mpi}" if site.srun_mpi else "")
                + (' -c "$SLURM_CPUS_PER_TASK"' if cpt > 1 else "") + f' "{exe}" 2>&1 >& "$DIR/${{JOB}}.out"')
        out += ['if [ -n "$MACE_LOCAL_RUN" ] && [ -z "$SLURM_JOB_ID" ]; then', "  " + mpirun,
                "else", "  " + srun, "fi"]
    return out


LAUNCH_COMMENT = "# MACE: CRYSTAL started as the site settings say ([mpi], [crystal])"


def _scratch_shell(value: str) -> str:
    """[paths] scratch as the job script writes it: $USER is $MACE_USER there."""
    return value.replace("${USER}", "${MACE_USER}").replace("$USER", "$MACE_USER")


def customize_generator_text(text: str, kind: str, site: Site, resolution: Optional[Resolution],
                             owned: Tuple[str, ...] = ()) -> Tuple[str, List[str], List[str]]:
    """(text, changes, notes): the template ``text`` with this site's values
    in place of the default literals. Idempotent; a default literal that is
    missing (an edited template) is left alone with a note. ``owned`` names
    the job-size lines an editor wrote itself (``ntasks``, ``memory``): they
    are left as written."""
    lines = text.split("\n")
    changes: List[str] = []
    notes: List[str] = []
    # account
    acct_line = ACCOUNT_LINES.get(kind)
    dropped_at = None
    if site.account is None and acct_line:
        # no account: SLURM charges the user's default one (resolve_account)
        hits = [i for i, l in enumerate(lines) if l.rstrip() == acct_line]
        for i in reversed(hits):
            del lines[i]
        if hits:
            dropped_at = hits[0]
            changes.append("no account (the SLURM default account)")
    elif site.account != DEFAULTS["account"] and acct_line:
        hits = [i for i, l in enumerate(lines) if l.rstrip() == acct_line]
        if hits:
            for i in hits:
                l = lines[i]
                trail = l[len(l.rstrip()):]
                lines[i] = f"echo '#SBATCH -A {site.account}' >> $1.sh" + trail
            changes.append(f"account {site.account}")
        elif not any(l.rstrip() == f"echo '#SBATCH -A {site.account}' >> $1.sh" for l in lines):
            notes.append("the #SBATCH -A line was not found; left as written")
    # partition
    if site.partition:
        p_line = f"echo '#SBATCH -p {site.partition}' >> $1.sh"
        a_idx = next((i for i, l in enumerate(lines) if l.startswith("echo '#SBATCH -A ")), None)
        if any(l.rstrip() == p_line for l in lines):
            pass
        elif a_idx is not None:
            lines.insert(a_idx + 1, p_line)
            changes.append(f"partition {site.partition}")
        elif dropped_at is not None:
            lines.insert(dropped_at, p_line)
            changes.append(f"partition {site.partition}")
        else:
            notes.append("no #SBATCH -A line to put the partition after; left as written")
    # [slurm] qos, constraint, extra: right after -p (or -A, or where -A was)
    wanted = ([f"echo '#SBATCH --qos={site.qos}' >> $1.sh"] if site.qos else []) \
        + ([f"echo '#SBATCH --constraint={site.constraint}' >> $1.sh"] if site.constraint else []) \
        + [f"echo '#SBATCH {item}' >> $1.sh" for item in site.extra]
    missing = [w for w in wanted if not any(l.rstrip() == w for l in lines)]
    if missing:
        anchor = None
        for i, l in enumerate(lines):
            if l.startswith("echo '#SBATCH -p ") or l.startswith("echo '#SBATCH -A ") or l.rstrip() in wanted:
                anchor = i + 1
        if anchor is None and dropped_at is not None:
            anchor = dropped_at
        if anchor is None:           # no -A line (an editor dropped it): where it would be, after --ntasks
            anchor = next((i + 1 for i, l in enumerate(lines) if l.startswith("echo '#SBATCH --ntasks=")), None)
        if anchor is None:
            notes.append("no #SBATCH -A line to put qos/constraint/extra after; left as written")
        else:
            for w in missing:
                lines.insert(anchor, w)
                anchor += 1
            changes.append("directives " + " ".join(w.split("'")[1][len("#SBATCH "):] for w in missing))
    # [resources]: the job size (the lines an editor wrote are its own)
    rset = site.resources_set(kind)
    lit = RESOURCE_LINES.get(kind, {})
    done = []
    for key, fmt in (("cpus_per_task", "echo '#SBATCH --cpus-per-task={}' >> $1.sh"),
                     ("ntasks", "echo '#SBATCH --ntasks={}' >> $1.sh"), ("nodes", "echo '#SBATCH -N {}' >> $1.sh")):
        if key in rset and key not in owned and key in lit:
            for i, l in enumerate(lines):
                if l.rstrip() == lit[key] and fmt.format(rset[key]) != lit[key]:
                    lines[i] = fmt.format(rset[key])
                    done.append(fmt.format(rset[key]).split("'")[1][len("#SBATCH "):])
    if "mem_form" in rset and "memory" not in owned and "memory" in lit:
        flag = "--mem" if rset["mem_form"] == "mem" else "--mem-per-cpu"
        new_line = f"echo '#SBATCH {flag}={rset['mem']}' >> $1.sh"
        for i, l in enumerate(lines):
            if l.rstrip() == lit["memory"] and new_line != lit["memory"]:
                lines[i] = new_line
                done.append(f"{flag}={rset['mem']}")
    if done:
        changes.append("resources " + " ".join(done))
    # [modules] purge = false
    if not site.module_purge:
        hits = [i for i, l in enumerate(lines) if l.rstrip() == PURGE_LINE]
        for i in hits:
            lines[i] = PURGE_COMMENT
        if hits:
            changes.append("no module purge")
    # [crystal] bin_dir without a module: no CRYSTAL module line
    if site.crystal_bin_dir and resolution is not None and not resolution.modules:
        hits = [i for i, l in enumerate(lines) if l.rstrip() == CRYSTAL_LINE]
        new = [f"# MACE: CRYSTAL from {site.crystal_bin_dir} ([crystal] bin_dir)"]
        if site.crystal_bin_dir.endswith("/bin") and len(site.crystal_bin_dir) > 4:
            new.append(f'export EBROOTCRYSTAL="{site.crystal_bin_dir[:-4]}"')
        for i in reversed(hits):
            lines[i:i + 1] = new
        if hits:
            changes.append(f"CRYSTAL from {site.crystal_bin_dir}")
    # CRYSTAL module
    if resolution is not None and resolution.modules and (
            resolution.modules != DEFAULTS["crystal"] or resolution.root_from_path
            or (resolution.source == "discovered" and resolution.note)):
        comment = "# MACE: " + resolution.note.replace("'", "") if resolution.source == "discovered" and resolution.note else None
        # a line already rewritten (its comment right above it) is left alone: rewriting is idempotent
        hits = [i for i, l in enumerate(lines) if l.rstrip() == CRYSTAL_LINE and not (i and comment and lines[i - 1] == comment)]
        if hits:
            new = []
            if comment:
                new.append(comment)
            new += [f"module load {m}" for m in resolution.modules]
            if resolution.root_from_path:
                new.append(ROOT_FROM_PATH_LINE)
            for i in reversed(hits):
                lines[i:i + 1] = new
            changes.append("CRYSTAL module " + " ".join(resolution.modules))
        elif not any(l.rstrip() == f"module load {resolution.module}" for l in lines) \
                and resolution.modules != DEFAULTS["crystal"]:
            notes.append(f"the {CRYSTAL_LINE} line was not found; left as written")
    # Python modules
    if site.python != DEFAULTS["python"]:
        idx = [i for i, l in enumerate(lines) if l.rstrip() in PYTHON_LINES]
        if idx:
            first = idx[0]
            for i in reversed(idx):
                del lines[i]
            for j, m in enumerate(site.python):
                lines.insert(first + j, f"module load {m}")
            changes.append("Python modules " + (" ".join(site.python) or "(none)"))
        elif not all(any(l.rstrip() == f"module load {m}" for l in lines) for m in site.python):
            notes.append("the Python module lines were not found; left as written")
    elif site.python_executable and SAFE_EXE_RE.fullmatch(site.python_executable):
        # the modules do not load here: the completion step's `python` is
        # the Python that runs MACE (resolve_python)
        func = PYTHON_FUNCTION.format(exe=site.python_executable)
        idx = [i for i, l in enumerate(lines) if l.rstrip() in PYTHON_LINES]
        if idx:
            first = idx[0]
            for i in reversed(idx):
                del lines[i]
            lines[first:first] = [PYTHON_COMMENT, func]
            changes.append(f"Python {site.python_executable}")
        elif func not in lines:
            notes.append("the Python module lines were not found; left as written")
    # [mpi] / [crystal] / cpus_per_task: how CRYSTAL is started
    block = launch_block(kind, site) if kind in LAUNCH_LINES else None
    if block and LAUNCH_COMMENT not in lines:
        hits = [i for i, l in enumerate(lines) if l.rstrip() in (LAUNCH_LINES[kind], *OLDER_LAUNCH_LINES[kind])]
        for i in reversed(hits):
            lines[i:i + 1] = block
        if hits:
            changes.append(f"CRYSTAL launch {site.launcher}" + (f" --mpi={site.srun_mpi}" if site.launcher == "srun"
                                                                and site.srun_mpi else ""))
        else:
            notes.append("the CRYSTAL launch line was not found; left as written")
    # [paths] scratch: a candidate right after $SCRATCH
    if site.scratch:
        v = _scratch_shell(site.scratch)
        guard = [f'if [ -d "{v}" ]; then', f'  MACE_SCRATCH_TRY+=("$(readlink -f "{v}")$MACE_SCRATCH_SUFFIX")', "fi"]
        for i in range(len(lines) - 1):
            if lines[i].rstrip() == SCRATCH_LINES[0] and lines[i + 1].rstrip() == SCRATCH_LINES[1]:
                if lines[max(0, i - 3):i] != guard:
                    lines[i:i] = guard
                    changes.append(f"scratch {site.scratch}")
                break
    # the default walltime, when the partition allows less
    wl = WALLTIME_LINES.get(kind)
    if site.walltime_cap and wl and site.walltime_cap < wl[2]:
        for i in range(len(lines) - 1):
            if lines[i].rstrip() == wl[0] and lines[i + 1].rstrip() == wl[1]:
                d, rest = walltime_text(site.walltime_cap).split("-", 1)
                lines[i], lines[i + 1] = f"time={d}", f"wall=-{rest}"
                changes.append(f"walltime {walltime_text(site.walltime_cap)}")
                break
    # the completion callback's queue limits
    queue = site.queue_text
    if queue != QUEUE_TEXT:
        hit = False
        for i, l in enumerate(lines):
            if QUEUE_TEXT in l:
                lines[i] = l.replace(QUEUE_TEXT, queue)
                hit = True
        if hit:
            changes.append(f"queue limits {queue}")
    return "\n".join(lines), changes, notes


def is_shipped(path) -> Optional[str]:
    """'scf' / 'prop' when ``path`` is MACE's own template, else None."""
    try:
        real = Path(path).resolve()
    except OSError:
        return None
    for kind, t in TEMPLATES.items():
        if real == t.resolve():
            return kind
    return None


_SAID: set = set()


def say_once(text: str, out=print) -> None:
    """Print a site notice once per process (a batch generates many scripts)."""
    if text and text not in _SAID:
        _SAID.add(text)
        out(text)


def generator_for(path, kind: Optional[str] = None, work_dir=None, env: Optional[Dict[str, str]] = None,
                  parent_module: Optional[str] = None, out=print):
    """(generator path, cleanup, resolution): ``path`` itself when it is not
    a shipped template or nothing changes; otherwise a rewritten copy
    ``.<stem>_site.<pid>.sh`` (mode 0700) in ``work_dir``, removed by
    ``cleanup()``. Raises SiteConfigError on invalid settings."""
    path = Path(path)
    shipped = is_shipped(path)
    if shipped is None:
        return path, (lambda: None), None
    kind = kind or shipped
    env = dict(os.environ if env is None else env)
    site = load(env)
    res = crystal_module(env, site, parent_module)
    resolve(site, env, lambda t: out(f"  {t}"))
    text = path.read_text()
    new, changes, notes = customize_generator_text(text, kind, site, res)
    if res.source == "discovered" and res.note:
        say_once(f"  MACE: {res.note}", out)
    if new == text:
        return path, (lambda: None), res
    work = Path(work_dir) if work_dir else Path.cwd()
    target = work / f".{path.stem}_site.{os.getpid()}.sh"
    fd = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o700)
    with os.fdopen(fd, "w") as fh:
        fh.write(new)

    def cleanup():
        try:
            target.unlink()
        except OSError:
            pass
    return target, cleanup, res


def customize_file_in_place(path, kind: str = "scf", env: Optional[Dict[str, str]] = None, out=print,
                            owned: Tuple[str, ...] = ("ntasks", "memory")):
    """Apply the site values to a temporary generator MACE wrote itself in
    this run (the ``-i``/walltime ``submitcrystal23_custom_<pid>.sh`` in its
    private temp folder): its location, and so its install pin, stay as
    they are. The ``owned`` lines
    (cores and memory) are the editor's: it already wrote the site's default
    or the user's answer there. Returns the resolution."""
    env = dict(os.environ if env is None else env)
    site = load(env)
    res = crystal_module(env, site)
    resolve(site, env, lambda t: out(f"  {t}"))
    p = Path(path)
    text = p.read_text()
    new, changes, notes = customize_generator_text(text, kind, site, res, owned=owned)
    if res.source == "discovered" and res.note:
        say_once(f"  MACE: {res.note}", out)
    if new != text:
        p.write_text(new)
    return res


def invalid_account_hint(text: str, env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """The one-line hint for sbatch's invalid-account error, else None."""
    if INVALID_ACCOUNT_TEXT.lower() not in (text or "").lower():
        return None
    path = config_path(env) or "~/.config/mace/site.toml"
    return (f"Set your cluster account in {path} ([slurm] account) or with MACE_SLURM_ACCOUNT; "
            "`mace site` shows what MACE uses.")


SBATCH_HINTS = (
    (("you must specify an account", "no default account"),
     "SLURM needs an account here: set [slurm] account in {path} or MACE_SLURM_ACCOUNT; `mace site` shows what "
     "MACE uses."),
    (("invalid qos specification", "job violates accounting/qos policy"),
     "Set the QOS your cluster expects with [slurm] qos in {path} (or MACE_SLURM_QOS); `mace site check` tests "
     "it."),
    (("requested node configuration is not available", "memory required by task is not available"),
     "The job asks for more than a node of this partition has: set [resources] ntasks and mem_per_cpu (or mem) "
     "in {path}; `mace site check` shows what fits."),
    (("invalid feature specification",),
     "Set [slurm] constraint in {path} to a feature the partition's nodes have (sinfo -o \"%P %f\")."),
    (("requested time limit is invalid",),
     "The walltime is above the partition's limit: use mace submit --walltime, or set [slurm] partition in "
     "{path} to one that allows it."),
)


def sbatch_hint(text: str, env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """The one-line hint for an sbatch rejection MACE's settings can fix, else None."""
    hint = invalid_account_hint(text, env)
    if hint:
        return hint
    t = (text or "").lower()
    if "maxsubmit" in t:              # a submit-limit refusal: the queue manager waits, no setting is wrong
        return None
    path = config_path(env) or "~/.config/mace/site.toml"
    for needles, hint in SBATCH_HINTS:
        if any(n in t for n in needles):
            return hint.format(path=path)
    return None


def account(env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """The effective account (None: no -A line, SLURM's default account),
    the built-in one when the settings are invalid."""
    try:
        site = load(env)
    except SiteConfigError:
        return DEFAULTS["account"]
    return resolve_account(site, env)[0]


def account_label(value: Optional[str]) -> str:
    """How an account is shown: None is the SLURM default account."""
    return value if value else "(your SLURM default account)"


def account_usable(name: str, env: Optional[Dict[str, str]] = None) -> bool:
    """``name`` (a built-in MSU HPCC account) may be written into a job
    script: the user is a member of it, or membership cannot be checked and
    the built-in account is kept here (keeps_builtin_account)."""
    assoc = slurm_associations(env)
    if assoc is None:
        return keeps_builtin_account(env)
    return name in assoc["accounts"]


def safe_load(env: Optional[Dict[str, str]] = None) -> Site:
    try:
        return load(env)
    except SiteConfigError:
        return Site()


def describe(env: Optional[Dict[str, str]] = None, check_account: bool = False, refresh: bool = False) -> Dict[str, Any]:
    """What ``mace site`` and the GUI show: each value as the job scripts get
    it (checked against this cluster, ``resolve``) and where it comes from."""
    env = dict(os.environ if env is None else env)
    out: Dict[str, Any] = {"errors": [], "warnings": []}
    try:
        site = load(env)
    except SiteConfigError as exc:
        out["errors"].append(str(exc))
        site = Site()
        site.path = config_path(env)
        site.exists = bool(site.path and site.path.is_file())
        site.sources = {k: "built-in" for k in ("account", "partition", "crystal", "python",
                                                 "gateway_message", "module_message", "max_jobs", "reserve")}
    out["warnings"] += site.warnings
    if refresh:
        forget_crystal_cache(env=env)
        forget_slurm_cache(env)
    configured = site.account
    resolve(site, env, out=lambda t: None)
    out["config"] = {"path": str(site.path) if site.path else None, "exists": site.exists,
                     "group": [{"path": str(g), "owner": site.group_owners.get(str(g))} for g in site.group_paths]}
    a_src = site.sources.get("account", "built-in")
    if site.account is None:
        a_src = ("built-in account left out: SLURM accounting cannot be read here"
                 if site.checks.get("account") == "unreadable"
                 else f"built-in {configured} left out: you are not a member")
    out["account"] = {"value": site.account, "source": a_src, "check": site.checks.get("account")}
    out["account"]["origin"] = site.origins.get("account")
    out["partition"] = {"value": site.partition, "source": site.sources.get("partition", "built-in"),
                        "origin": site.origins.get("partition")}
    out["python"] = {"value": site.python, "source": site.sources.get("python", "built-in"),
                     "executable": site.python_executable, "check": site.checks.get("python")}
    out["queue"] = {"max_jobs": site.max_jobs, "reserve": site.reserve, "source": site.checks.get("queue")}
    out["walltime_cap"] = ({"value": walltime_text(site.walltime_cap), "partition": site.cap_partition}
                           if site.walltime_cap else None)
    out["notes"] = list(site.notes)
    out["cluster"] = {"msu_hpcc": msu_hpcc(env), "host": host_names(env)[1]}
    out["messages"] = {k: {"value": site.message(k, env, reason="{reason}", module="{module}", host="{host}"),
                           "source": site.sources.get(k, "built-in")}
                       for k in ("gateway_message", "module_message")}
    res = crystal_module(env, site, refresh=refresh)
    out["crystal"] = {**res.record(), "preferred": site.crystal, "window": list(site.window),
                      "preferred_source": site.sources.get("crystal", "built-in"),
                      "module_system": envmodules.module_system(env)}
    out["account_check"] = check_account_membership(site.account, env) if check_account else {"status": "not checked"}
    out.update(describe_g(site, env, refresh=refresh))
    out["warnings"] += out.pop("g_warnings")
    try:
        from mace.cluster import config as _cc          # Area F's monitoring settings, when present
        out["monitoring_config"] = str(_cc.config_path())
    except Exception:
        out["monitoring_config"] = None
    return out


def _source(site: Site, *keys: str) -> Tuple[str, Optional[str]]:
    """(source, origin) of the first of ``keys`` someone set, else built-in."""
    for k in keys:
        if site.sources.get(k, "built-in") != "built-in":
            return site.sources[k], site.origins.get(k)
    return "built-in", None


def mem_mb(value: str) -> Optional[int]:
    m = MEM_RE.fullmatch(value or "")
    if not m:
        return None
    n = int(value.rstrip("KMGT"))
    unit = value[-1] if value[-1] in "KMGT" else "M"
    return {"K": max(1, n // 1024), "M": n, "G": n * 1024, "T": n * 1024 * 1024}[unit]


def node_shapes(partition: Optional[str], env: Optional[Dict[str, str]] = None,
                refresh: bool = False) -> Dict[str, Dict[str, Any]]:
    """{partition: {cores, max_cores, threads, mem_mb, max_mem_mb, nodes,
    features, gres}} from ``sinfo -N`` (cores = sockets x cores per socket,
    the physical ones; cached a day); {} without sinfo or an answer."""
    import shutil
    from collections import Counter
    env = dict(os.environ if env is None else env)
    exe = shutil.which("sinfo", path=env.get("PATH"))
    if not exe:
        return {}
    key = json.dumps(["SLURM-shape", partition or "", exe])
    if not refresh:
        if key in _SLURM_MEMO:
            return _SLURM_MEMO[key]
        hit = _slurm_cached(key, env)
        if hit and isinstance(hit.get("shapes"), dict):
            _SLURM_MEMO[key] = hit["shapes"]
            return hit["shapes"]
    argv = [exe, "-h", "-N"] + (["-p", partition] if partition else []) + ["-o", "%n|%T|%X|%Y|%Z|%c|%m|%f|%G|%P"]
    r = _slurm_run(argv, env)
    shapes: Dict[str, Dict[str, Any]] = {}
    if r is not None and r.returncode == 0:
        per: Dict[str, List[Tuple[int, int, int, str, str]]] = {}
        seen = set()
        for line in r.stdout.splitlines():
            f = line.strip().split("|")
            if len(f) < 10:
                continue
            try:
                cores = int(f[2]) * int(f[3])
                threads = int(f[4])
                mem = int(re.sub(r"[^0-9]", "", f[6]) or 0)
            except ValueError:
                continue
            part = f[9].rstrip("*")
            if (f[0], part) in seen:
                continue
            seen.add((f[0], part))
            per.setdefault(part, []).append((cores, threads, mem, f[7], f[8]))
        for part, rows in per.items():
            (cores, threads, mem), _ = Counter((c, t, m) for c, t, m, _, _ in rows).most_common(1)[0]
            feats = sorted({x for *_, fe, _ in rows for x in fe.split(",") if x and x != "(null)"})
            gres = sorted({g for *_, g in rows if g and g != "(null)"})
            shapes[part] = {"cores": cores, "max_cores": max(c for c, *_ in rows), "threads": threads,
                            "mem_mb": mem, "max_mem_mb": max(m for _, _, m, _, _ in rows), "nodes": len(rows),
                            "features": feats, "gres": gres}
        envmodules.cache_put(key, {"shapes": shapes, "name": "SLURM"}, env)
    _SLURM_MEMO[key] = shapes
    return shapes


def cached_node_cores(env: Optional[Dict[str, str]] = None) -> Optional[int]:
    """The most physical cores of a node in any cached ``node_shapes`` answer
    (a file read, never a SLURM call), or None."""
    best = None
    try:
        entries = envmodules._read_cache(os.environ if env is None else env)
    except Exception:
        return None
    for v in entries.values():
        if isinstance(v, dict) and isinstance(v.get("shapes"), dict):
            for sh in v["shapes"].values():
                if isinstance(sh, dict) and isinstance(sh.get("max_cores"), int):
                    best = max(best or 0, sh["max_cores"])
    return best


def default_partition(env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """The cluster's default partition (scontrol), or None."""
    import shutil
    env = dict(os.environ if env is None else env)
    exe = shutil.which("scontrol", path=env.get("PATH"))
    if not exe:
        return None
    key = json.dumps(["SLURM-default-partition", exe])
    if key in _SLURM_MEMO:
        return _SLURM_MEMO[key]
    found = None
    r = _slurm_run([exe, "show", "partition", "-a", "-o"], env)
    if r is not None and r.returncode == 0:
        for line in r.stdout.splitlines():
            if re.search(r"\bDefault=YES\b", line):
                m = re.search(r"\bPartitionName=(\S+)", line)
                if m:
                    found = m.group(1)
                    break
    _SLURM_MEMO[key] = found
    return found


def fit(site: Site, kind: str, env: Optional[Dict[str, str]] = None, refresh: bool = False) -> Optional[Dict[str, Any]]:
    """Does the ``kind`` job size fit a node of the partition it goes to?
    {ok, partition, node_cores, node_mem, request, suggest}, or None when
    SLURM does not say (no sinfo, an unknown partition)."""
    eff = site.effective_resources(kind)
    part = (site.partition or "").split(",")[0] or default_partition(env)
    if not part:
        return None
    shape = node_shapes(part, env, refresh).get(part)
    if not shape:
        return None
    per_node = -(-int(eff["ntasks"]) * int(eff["cpus_per_task"]) // max(1, int(eff["nodes"])))
    mb = mem_mb(eff["mem"]) or 0
    # --mem-per-cpu is per allocated CPU: every task's cpus_per_task CPUs count
    need_mem = mb * per_node if eff["mem_form"] == "mem_per_cpu" else mb
    ok = per_node <= shape["max_cores"] and (not shape["max_mem_mb"] or need_mem <= shape["max_mem_mb"])
    cores = shape["cores"] or 1
    return {"ok": ok, "partition": part, "node_cores": shape["max_cores"], "node_mem": shape["max_mem_mb"],
            "common_cores": shape["cores"], "common_mem": shape["mem_mb"],
            "request": {"cores_per_node": per_node, "mem_mb_per_node": need_mem},
            "suggest": {"ntasks": max(1, cores // max(1, int(eff["cpus_per_task"]))),
                        "mem_per_cpu": f"{int(0.95 * shape['mem_mb'] / cores)}M"}}


def describe_g(site: Site, env: Dict[str, str], refresh: bool = False, check_fit: bool = True) -> Dict[str, Any]:
    """The [resources] [mpi] [paths] [crystal] [slurm] qos/constraint/extra
    [modules] purge and [local_run] max_time part of describe()."""
    out: Dict[str, Any] = {}
    res = {}
    for kind in ("scf", "prop"):
        keys = [f"resources.{k}" for k in RESOURCE_KEYS]
        if kind == "prop":
            keys = [f"resources.properties.{k}" for k in RESOURCE_KEYS] + keys
        src, origin = _source(site, *keys)
        eff = site.effective_resources(kind)
        res[kind] = {"value": eff, "set": site.resources_set(kind), "source": src, "origin": origin,
                     "fit": fit(site, kind, env, refresh) if check_fit and slurm_here("sinfo", env) else None}
    out["resources"] = res
    src, origin = _source(site, "launcher", "srun_mpi", "mpi_env")
    out["mpi"] = {"launcher": site.launcher, "srun_mpi": site.srun_mpi, "env": list(site.mpi_env),
                  "source": src, "origin": origin}
    src, origin = _source(site, "scratch")
    out["scratch"] = {"value": site.scratch, "order": scratch_order(site), "source": src, "origin": origin}
    src, origin = _source(site, "crystal_bin_dir", "crystal_exe.scf", "crystal_exe.prop")
    out["crystal_install"] = {"bin_dir": site.crystal_bin_dir, "scf": site.crystal_exe["scf"],
                              "properties": site.crystal_exe["prop"], "source": src, "origin": origin}
    out["slurm_extra"] = {k: {"value": v, "source": _source(site, k)[0], "origin": _source(site, k)[1]}
                          for k, v in (("qos", site.qos), ("constraint", site.constraint), ("extra", list(site.extra)))}
    src, origin = _source(site, "module_purge")
    out["modules_purge"] = {"value": site.module_purge, "source": src, "origin": origin}
    src, origin = _source(site, "run_max_time")
    out["local_run"] = {"max_time": site.run_max_time, "source": src, "origin": origin,
                        "msu_hpcc": msu_hpcc(env)}
    warns = []
    if any(site.effective_resources(k)["nodes"] > 1 for k in ("scf", "prop")):
        warns.append(MULTI_NODE_WARNING)
    out["g_warnings"] = warns
    return out


MULTI_NODE_WARNING = ("nodes > 1: the job needs a scratch directory shared by all nodes ([paths] scratch) and an "
                      "MPI launch that spans nodes; $TMPDIR is node-local")


def scratch_order(site: Site) -> List[str]:
    """Where a job script looks for its scratch directory, in order."""
    return ["$SCRATCH"] + ([site.scratch] if site.scratch else []) + ["/mnt/scratch/$USER", "the submit directory",
                                                                      "$TMPDIR"]


def forget_slurm_cache(env: Optional[Dict[str, str]] = None) -> int:
    """Drop the cached SLURM answers and Python module checks (mace site --refresh)."""
    _SLURM_MEMO.clear()
    return envmodules.cache_forget(lambda k, v: isinstance(v, dict) and v.get("name") in ("SLURM", "PYTHON"), env)


def check_account_membership(acct: Optional[str], env: Optional[Dict[str, str]] = None,
                             refresh: bool = False) -> Dict[str, Any]:
    """{status: member|not_member|default|unreadable|not checked, accounts}:
    the account against the user's SLURM associations (sacctmgr, cached a
    day); status default when no account is written (SLURM charges the
    default one), unreadable when none is written because the associations
    cannot be read here."""
    assoc = slurm_associations(env, refresh=refresh)
    if not assoc:                     # no SLURM accounting here, or none recorded for this user
        return {"status": "unreadable" if acct is None else "not checked", "accounts": []}
    if acct is None:
        return {"status": "default", "accounts": assoc["accounts"], "default": assoc.get("default")}
    return {"status": "member" if acct in assoc["accounts"] else "not_member", "accounts": assoc["accounts"]}


def init_file(env: Optional[Dict[str, str]] = None) -> Tuple[bool, Path]:
    """Write the all-commented site.toml (mode 0644) when it is absent: (written, path)."""
    path = config_path(env)
    if path is None:
        raise SiteConfigError("MACE_SITE_CONFIG=none: there is no site.toml to create")
    if path.exists():
        return False, path
    path.parent.mkdir(parents=True, exist_ok=True)
    text = INIT_TEXT + detected_text(env)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    return True, path


def detected_text(env: Optional[Dict[str, str]] = None) -> str:
    """What `mace site --init` found on this cluster, every line a comment
    (offered, never applied): the node shape of the default partition and a
    scratch directory that looks shared. Empty where SLURM does not answer."""
    env = dict(os.environ if env is None else env)
    out = []
    if slurm_here("sinfo", env):
        part = default_partition(env)
        shape = node_shapes(part, env).get(part) if part else None
        if shape:
            cores = shape["cores"] or 1
            out.append(f"# partition {part}: {shape['nodes']} nodes of {shape['cores']} cores and "
                       f"{shape['mem_mb'] // 1024}G (sinfo); a whole node would be:")
            out.append(f"# [resources] ntasks = {cores}, mem_per_cpu = \"{int(0.95 * shape['mem_mb'] / cores)}M\"")
    try:
        from mace.submission.site_check import scratch_suggestion
        hint = scratch_suggestion(env)
    except Exception:
        hint = None
    if hint:
        out.append(f'# scratch = "{hint}"   # found here; confirm with your admins that it is shared by all nodes')
    if not out:
        return ""
    return "\n# Detected on this cluster (suggestions only; MACE never applies them):\n" + "\n".join(out) + "\n"


# ------------------------------------------------------------------ workflows
def plan_resources(env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """The site values a workflow plan stores once, at plan time (on the
    login or dev node): account (None: no -A line), CRYSTAL module (resolved
    here), and the partition, Python modules or Python executable and queue
    limits when they are not the built-in ones. Every later step and callback
    uses these stored values."""
    site = load(env)
    res = crystal_module(env, site)
    resolve(site, env)
    out: Dict[str, Any] = {"account": site.account}
    if res.modules:                   # none: CRYSTAL from [crystal] bin_dir
        out["module"] = res.module
    if len(res.modules) > 1:
        out["modules"] = list(res.modules)
    if res.root_from_path:
        out["module_root_from_path"] = True
    if site.partition:
        out["partition"] = site.partition
    if site.python != DEFAULTS["python"]:
        out["python_modules"] = list(site.python)
    if site.python_executable:
        out["python_executable"] = site.python_executable
    if site.queue_text != QUEUE_TEXT:
        out["queue_limits"] = [site.max_jobs, site.reserve]
    out.update(plan_site_keys(site))
    return out


def plan_site_keys(site: Site) -> Dict[str, Any]:
    """The [resources] (lines no workflow editor writes), [mpi], [paths],
    [crystal], [modules] purge and [slurm] qos/constraint/extra values a plan
    stores; only those that differ from the built-in ones."""
    out: Dict[str, Any] = {}
    sr = {}
    for kind in ("scf", "prop"):
        rs = {k: v for k, v in site.resources_set(kind).items() if k in ("nodes", "cpus_per_task")}
        if rs:
            sr[kind] = rs
    if sr:
        out["site_resources"] = sr
    if site.launcher != "mpirun" or site.mpi_env or site.srun_mpi:
        out["mpi"] = {"launcher": site.launcher, "srun_mpi": site.srun_mpi, "env": list(site.mpi_env)}
    if site.scratch:
        out["scratch"] = site.scratch
    if site.crystal_bin_dir:
        out["crystal_bin_dir"] = site.crystal_bin_dir
    if site.binaries() != BINARIES:
        out["crystal_exe"] = dict(site.crystal_exe)
    if not site.module_purge:
        out["module_purge"] = False
    if site.qos:
        out["slurm_qos"] = site.qos
    if site.constraint:
        out["slurm_constraint"] = site.constraint
    if site.extra:
        out["slurm_extra"] = list(site.extra)
    return out


def _replay_site_keys(site: Site, resources: Dict[str, Any]) -> None:
    """The plan's new site keys, validated as load() validates them: an
    invalid stored value (a plan file edited by hand) generates nothing."""
    src = "workflow plan:"
    sr = resources.get("site_resources")
    if sr is not None:
        if not isinstance(sr, dict):
            raise _bad(f"{src} site_resources", "table", sr)
        for kind, level in (("scf", ""), ("prop", "properties")):
            table = sr.get(kind) or {}
            if not isinstance(table, dict):
                raise _bad(f"{src} site_resources.{kind}", "table", table)
            for k, v in table.items():
                if k not in ("nodes", "cpus_per_task"):
                    raise _bad(f"{src} site_resources.{kind}", "key (nodes, cpus_per_task)", k)
                site.res_raw.setdefault(level, {})[k] = _valid_int(v, f"{src} site_resources.{kind}.{k}", k)
        # the scf numbers are the top level; prop's own override them, and an
        # scf-only key must not reach Pproperties jobs
        for k in list(site.res_raw[""]):
            if k not in site.res_raw["properties"]:
                site.res_raw["properties"][k] = TEMPLATE_RESOURCES["prop"][k]
    mpi = resources.get("mpi")
    if mpi is not None:
        if not isinstance(mpi, dict):
            raise _bad(f"{src} mpi", "table", mpi)
        site.launcher = _valid_launcher(mpi.get("launcher", "mpirun"), f"{src} mpi.launcher")
        if mpi.get("srun_mpi") is not None:
            site.srun_mpi = _valid_srun_mpi(mpi["srun_mpi"], f"{src} mpi.srun_mpi")
        env_items = mpi.get("env") or []
        for i, item in enumerate(env_items if isinstance(env_items, list) else [env_items]):
            _valid_env([item], f"{src} mpi.env[{i}]")
        site.mpi_env = _valid_env(env_items, f"{src} mpi.env")
    if resources.get("scratch") is not None:
        site.scratch = _valid_scratch(resources["scratch"], f"{src} scratch")
    if resources.get("crystal_bin_dir") is not None:
        site.crystal_bin_dir = _valid_bin_dir(resources["crystal_bin_dir"], f"{src} crystal_bin_dir")
    exe = resources.get("crystal_exe")
    if exe is not None:
        if not isinstance(exe, dict):
            raise _bad(f"{src} crystal_exe", "table", exe)
        for kind in ("scf", "prop"):
            if kind in exe:
                site.crystal_exe[kind] = _valid_exe(exe[kind], f"{src} crystal_exe.{kind}")
    if resources.get("module_purge") is not None:
        site.module_purge = _valid_bool(resources["module_purge"], f"{src} module_purge")
    if resources.get("slurm_qos") is not None:
        site.qos = _valid_qos(resources["slurm_qos"], f"{src} slurm_qos")
    if resources.get("slurm_constraint") is not None:
        site.constraint = _valid_constraint(resources["slurm_constraint"], f"{src} slurm_constraint")
    if resources.get("slurm_extra") is not None:
        site.extra = _valid_extra(resources["slurm_extra"], f"{src} slurm_extra")


def customize_workflow_text(text: str, resources: Optional[Dict[str, Any]] = None) -> str:
    """A workflow script built from a shipped template, with the partition,
    CRYSTAL module and Python modules its plan stored (``plan_resources``).
    The account is the plan's own (already written by the caller). A plan
    without these keys (made before site settings existed, e.g. by 1.1.4)
    gives the text unchanged."""
    resources = resources or {}
    site = Site()                     # built-in values: only what the plan stored changes anything
    part = resources.get("partition")
    if isinstance(part, str) and PARTITION_RE.fullmatch(part):
        site.partition = part
    py = resources.get("python_modules")
    if isinstance(py, list) and all(isinstance(m, str) and MODULE_RE.fullmatch(m) for m in py) and len(py) <= 10:
        site.python = list(py)
    exe = resources.get("python_executable")
    if isinstance(exe, str) and SAFE_EXE_RE.fullmatch(exe):
        site.python_executable = exe
    q = resources.get("queue_limits")
    if isinstance(q, list) and len(q) == 2 and all(isinstance(n, int) and not isinstance(n, bool) for n in q) \
            and q[0] >= 1 and q[1] >= 0:
        site.max_jobs, site.reserve = q
    res = None
    mods = resources.get("modules")
    module = resources.get("module")
    if isinstance(mods, list) and mods and all(isinstance(m, str) and MODULE_RE.fullmatch(m) for m in mods):
        res = Resolution(list(mods), "plan", root_from_path=bool(resources.get("module_root_from_path")))
    elif isinstance(module, str) and MODULE_RE.fullmatch(module):
        res = Resolution([module], "plan", root_from_path=bool(resources.get("module_root_from_path")))
    kind = "prop" if "Pproperties" in text else "scf"
    # the keys added in 1.1.5 (G): validated, an invalid one raises
    _replay_site_keys(site, resources)
    if site.crystal_bin_dir and res is None:
        res = Resolution([], "plan")
    new, _, _ = customize_generator_text(text, kind, site, res)
    return new
