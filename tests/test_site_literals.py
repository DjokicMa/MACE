"""Site literals live in one place (PLAN_submit D18/D34, shared with PLAN_F F14-b).

MSU HPCC's SLURM account and CRYSTAL module may appear only in the files of
ALLOWED, each with its reason; MACE's job scripts get them through
mace/submission/site.py. The new GUI submit/jobs/batch code and its tests
name no site at all. Other areas add their own patterns (storage system
names) to SITE_PATTERNS and their files to GENERIC_FILES."""
import re
import subprocess

import pytest

from conftest import REPO_ROOT

LITERALS = re.compile(r"mendoza_q|CRYSTAL/23-intel-2023a")
ALLOWED = {
    "mace/submission/site.py": "DEFAULTS: the one Python home of the built-in site values",
    "mace/submission/job_script_age.py": "the headers older MACE versions wrote, to recognize their unchanged job scripts",
    "mace/submission/submitcrystal23.sh": "the shipped template, never edited (site.py rewrites a copy)",
    "mace/submission/submit_prop.sh": "the shipped template, never edited (site.py rewrites a copy)",
    "mace/workflow/common/constants.py": "import-compatible constants; readers go through site.py",
    "mace/submission/local_run.py": "import-compatible CRYSTAL_MODULE; the probe uses site.py",
    "mace/submission/preflight.py": "import-compatible CRYSTAL_MODULE; the probe uses site.py",
    "mace/utils/node_exclusion.py": "partition default of the node-exclusion helper (out of scope, 1.1.6)",
    "mace/recovery/slurm_limits.py": "comments recording measured MSU partition behavior",
    "mace/recovery/recovery.py": "a comment recording measured MSU behavior",
    "mace/recovery/screened_hybrid.py": "docstring: the CRYSTAL build the numbers were measured with",
    "mace/config/recovery_config.yaml": "a comment recording measured MSU behavior",
    "Crystal_d12/ion_charges.py": "provenance: the CRYSTAL build the tables were measured with",
    "Crystal_d12/d12_constants.py": "provenance comments: the CRYSTAL build that was measured",
    "Crystal_d12/soc_ecp.py": "provenance comments: the CRYSTAL build the SOC checks were measured with",
    "mace/utils/envmodules.py": "docstring examples of module version parsing",
    "mace_cli": "help text of mace run / mace preflight naming the built-in module",
}
ALLOWED_PREFIXES = {
    "code/OldSLURMTemplates/": "historical templates, not used by MACE",
    "tests/": "test fixtures and golden files copied from git",
}
DOC_SUFFIXES = (".md",)          # docs name MSU as the example the built-ins come from

SITE_PATTERNS = re.compile(r"mendoza|\bmsu\b|djokicma|amd20|dev-amd|dev-intel|CRYSTAL/23", re.I)
GENERIC_FILES = ["mace/gui/submit.py", "mace/gui/jobs.py", "mace/gui/batch.py", "mace/gui/slurm_preview_shim.py",
                 "mace/gui/static/js/lazy/submit_dialog.js", "mace/gui/static/js/batch/panel.js",
                 "mace/gui/static/js/docs/jobs.js",
                 "tests/gui/test_submit_unit.py", "tests/gui/test_submit_api.py", "tests/gui/test_jobs_api.py",
                 "tests/gui/test_batch_api.py", "tests/gui/test_submit_env.py", "tests/gui/test_submit_batch_e2e.py",
                 "tests/gui/test_local_run_stop.py"]


def _tracked():
    r = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files", "-co", "--exclude-standard"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("not a git checkout")
    return [l for l in r.stdout.splitlines() if l and (REPO_ROOT / l).is_file()]


def test_site_literals_only_in_allowed_files():
    bad = []
    for rel in _tracked():
        if rel in ALLOWED or rel.endswith(DOC_SUFFIXES) or any(rel.startswith(p) for p in ALLOWED_PREFIXES):
            continue
        try:
            text = (REPO_ROOT / rel).read_text(errors="ignore")
        except OSError:
            continue
        if LITERALS.search(text):
            bad.append(rel)
    assert not bad, f"site literals outside site.py's allowlist: {bad}"


def test_planner_names_no_site_value():
    text = (REPO_ROOT / "mace" / "workflow" / "planner.py").read_text()
    assert not LITERALS.search(text)


def test_generic_gui_files_name_no_site():
    bad = []
    for rel in GENERIC_FILES:
        p = REPO_ROOT / rel
        if p.is_file():
            for i, line in enumerate(p.read_text().splitlines(), 1):
                if SITE_PATTERNS.search(line):
                    bad.append(f"{rel}:{i}: {line.strip()[:100]}")
    assert not bad, bad


def test_rewritten_literals_are_in_each_template_once():
    """Every line site.py rewrites is present exactly once in each template
    (a template edit that drops one would silently disable its key)."""
    from mace.submission import site
    for kind, path in site.TEMPLATES.items():
        lines = [l.rstrip() for l in path.read_text().split("\n")]
        wanted = list(site.RESOURCE_LINES[kind].values()) + [site.LAUNCH_LINES[kind], site.PURGE_LINE,
                                                               site.CRYSTAL_LINE, *site.SCRATCH_LINES]
        for w in wanted:
            n = lines.count(w)
            assert n == 1, (kind, w, n)


# One developer's or one machine's paths: never in code that runs for users.
# (`DjokicMa` in the GitHub URL is the project's home, not a path, and this
# match is case-sensitive on the NetID.)
MACHINE_PATHS = re.compile(r"/mnt/iscsi|/home/marcus|/home/djokic|djokicma|/mnt/research/")
MACHINE_PATHS_ALLOWED = {
    "mace/workflow/executor.py": "MSU group basis-set folder, tried only if it exists, after "
                                 "the bundled sets and MACE_*_BASIS_PATH",
}


def test_no_machine_specific_paths_in_runtime_code():
    # committed files only: a developer's untracked scratch is not shipped
    r = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files"], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("not a git checkout")
    bad = []
    for rel in [l for l in r.stdout.splitlines() if l and (REPO_ROOT / l).is_file()]:
        if (rel in MACHINE_PATHS_ALLOWED or rel.startswith("tests/") or rel.endswith(DOC_SUFFIXES)
                or rel in ("CITATION.cff", ".gitignore")):
            continue
        try:
            text = (REPO_ROOT / rel).read_text(errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if MACHINE_PATHS.search(line):
                bad.append(f"{rel}:{i}: {line.strip()[:100]}")
    assert not bad, f"machine-specific paths in runtime code: {bad}"
