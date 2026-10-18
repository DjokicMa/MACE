#!/usr/bin/env python3
"""
This script finds all .d3 files in the current directory and submits them for processing
by running submit_prop.sh with a parameter of 100 for each file.
"""
import os, sys, math
import re
import subprocess
import linecache
import shutil
import itertools
from pathlib import Path


def _budget():
    """mace.queue.budget (the user-wide queue budget)."""
    try:
        from mace.queue import budget
    except ImportError:          # pragma: no cover - run without the mace package
        return _NoBudget
    return budget


class _NoBudget:
    """offer_reserve() without the mace package (a bare script run)."""
    @staticmethod
    def offer_reserve():
        return [], []


def _budget_sbatch(argv):
    """sbatch from the current directory through the queue budget; a job
    that does not fit is listed for the reserve question (offer_reserve)."""
    b = _budget()
    if b is _NoBudget:
        from types import SimpleNamespace
        r = subprocess.run(argv, capture_output=True, text=True)
        m = re.search(r"Submitted batch job (\d+)", r.stdout or "")
        if r.returncode == 0:
            _note_submission()
        return SimpleNamespace(held=False, submitted=r.returncode == 0, job_id=m.group(1) if m else None,
                               message=(r.stderr or "").strip())
    out = b.sbatch(argv, os.getcwd(), quiet=True)
    if out.held:
        b.note_unsubmitted(argv[-1], os.getcwd(), argv)
    return out


def _note_submission():
    """Count a successful sbatch (mace.queue.slurm_ids: the completion
    callback never re-runs after a submission)."""
    try:
        from mace.queue.slurm_ids import note_submission
    except ImportError:          # run as a bare script without the mace package
        return
    note_submission()

try:
    from mace.submission.crystal import extract_slurm_job_id
except ImportError:
    def extract_slurm_job_id(output):
        """Parse the SLURM job id ('Submitted batch job N') from sbatch stdout."""
        if not output:
            return None
        match = re.search(r'Submitted batch job (\d+)', output)
        return match.group(1) if match else None

# Try to import node exclusion manager
try:
    from mace.utils.node_exclusion import NodeExclusionManager
    NODE_EXCLUSION_AVAILABLE = True
except ImportError:
    NODE_EXCLUSION_AVAILABLE = False

def get_default_amd20_exclusion():
    """Get default AMD20 node exclusion string (amr + nvf): MSU HPCC's
    node type, so only where its choices apply (site.msu_choices)."""
    if not NODE_EXCLUSION_AVAILABLE or not _site().msu_choices():
        return None
    try:
        manager = NodeExclusionManager()
        amd20_nodes = manager.get_amd20_nodes()
        if amd20_nodes:
            return manager.create_exclude_string(amd20_nodes)
    except Exception as e:
        print(f"Warning: Could not query AMD20 nodes: {e}")
    return None

def _site():
    """mace/submission/site.py (by path when run without the mace package)."""
    try:
        from mace.submission import site
    except ImportError:          # run as a bare script from mace/submission/
        import importlib.util
        spec = importlib.util.spec_from_file_location("mace_site", Path(__file__).with_name("site.py"))
        site = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(site)
    return site


def _site_account():
    """The SLURM account of this site (mace/submission/site.py): MACE_SLURM_ACCOUNT,
    ~/.config/mace/site.toml, else the built-in one."""
    return _site().account()


def _site_check():
    """Stop before anything is generated when a site setting is invalid."""
    site = _site()
    try:
        site.load()
    except site.SiteConfigError as exc:
        print(f"Error: {exc}")
        sys.exit(2)


def _site_generator(script_to_use, custom_script, data_folder, calc_type=None, auto=None):
    """The generator with this site's account, partition and modules
    (mace/submission/site.py). MACE's own temporary generator (the -i /
    walltime copy in a private temp folder, see temp_generator) is customized
    in place, so its install pin is unchanged; the shipped template gets a rewritten copy in the job
    folder, removed at exit, only when something differs. Copies of a
    template are used as written. ``calc_type``: the decks' type, when it
    has a [resources.<TYPE>] table of its own; ``auto``: their size-aware
    suggestion ([resources] auto)."""
    site = _site()
    try:
        if custom_script:
            site.customize_file_in_place(custom_script, "prop", calc_type=calc_type, auto=auto)
            return script_to_use
        path, cleanup, _ = site.generator_for(script_to_use, "prop", data_folder, calc_type=calc_type, auto=auto)
    except site.SiteConfigError as exc:
        print(f"Error: {exc}")
        sys.exit(2)
    import atexit
    atexit.register(cleanup)
    return path


_HINTS_SAID = set()


def _site_hint(stderr):
    """One line, once per run, after an sbatch rejection a site setting can
    fix (account, QOS, job size, constraint, time limit)."""
    hint = _site().sbatch_hint(stderr or "")
    if hint and hint not in _HINTS_SAID:
        _HINTS_SAID.add(hint)
        print(f"    {hint}")


def _site_resources(base, kind, calc_type=None, auto=None):
    """``base`` with the job size and walltime of the site settings
    ([resources], [resources.<TYPE>] for ``calc_type``, the size-aware
    suggestion ``auto`` below them), which the prompts then offer as their
    defaults; unchanged without settings."""
    site = _site()
    try:
        return site.overlay_resources(base, site.load_resolved(), kind, calc_type, auto)
    except site.SiteConfigError:
        return base


def _memory_label(resources):
    """(prompt label, key) of the memory form ``resources`` carries."""
    if 'memory' in resources and 'memory_per_cpu' not in resources:
        return ("Memory per node (total)" if resources.get('mem_form') else "Total memory"), 'memory'
    return "Memory per CPU", 'memory_per_cpu'


def _max_cores(default_ntasks):
    """The largest core count the prompt accepts: 128, or more when the site's
    job size or the cached node shape of this cluster says so (no SLURM call)."""
    return max(128, int(default_ntasks), _site().cached_node_cores() or 0)


def get_default_d3_resources(calc_type=None, auto=None):
    """Get default SLURM resources for D3 calculations (with the site's
    [resources] job size and walltime, when set; [resources.<TYPE>] too for
    ``calc_type``)"""
    return _site_resources({
        'ntasks': _site().PROPERTIES_NTASKS,
        'memory': _site().PROPERTIES_MEMORY,
        'walltime': _site().default_walltime("prop", '2:00:00'),   # the partition's MaxTime when shorter
        'account': _site_account(),
        'node_exclusion': get_default_amd20_exclusion()  # Default: exclude AMD20 nodes
    }, "prop", calc_type, auto)

def get_safe_integer_input(prompt, default, min_val=1, max_val=128):
    """Safely get integer input with validation"""
    while True:
        user_input = input(prompt).strip()
        if not user_input:
            return default
        try:
            value = int(user_input)
            if min_val <= value <= max_val:
                return value
            else:
                print(f"Please enter a value between {min_val} and {max_val}")
        except ValueError:
            print("Please enter a valid integer")

def get_safe_memory_input(prompt, default):
    """Safely get memory input with validation"""
    while True:
        user_input = input(prompt).strip()
        if not user_input:
            return default
        # Simple validation for memory format (e.g., 80G, 1024M)
        if re.match(r'^\d+[GM]B?$', user_input.upper()):
            return user_input.upper().rstrip('B')  # Remove trailing B if present
        else:
            print("Please enter memory in format like '80G' or '1024M'")

def get_safe_walltime_input(prompt, default):
    """Safely get walltime input with validation"""
    while True:
        user_input = input(prompt).strip()
        if not user_input:
            return default
        # Simple validation for walltime format (e.g., 2:00:00, 12:00:00, 1-12:00:00)
        if re.match(r'^(\d+-\d{1,2}:\d{2}:\d{2}|\d{1,3}:\d{2}:\d{2}|\d{1,2}:\d{2})$', user_input):
            return user_input
        else:
            print("Please enter walltime in format like '2:00:00', '12:00:00', or '1-12:00:00'")

def configure_interactive_resources(calc_type=None, decks=(), auto=None):
    """Interactively configure SLURM resources for D3 calculations
    (``calc_type``: the decks' type, when it has site settings of its own;
    ``auto``: their size-aware suggestion)"""
    print("\n" + "="*60)
    print("INTERACTIVE SLURM RESOURCE CONFIGURATION")
    print("="*60)

    default_resources = get_default_d3_resources(calc_type, auto)

    if auto:
        names = ", ".join(decks[:3]) + (", ..." if len(decks) > 3 else "")
        print(f"{calc_type} decks ({len(decks)}: {names}); defaults suggested from their size ([resources] auto):")
        print(f"  {auto['reason']}")
    elif calc_type:
        names = ", ".join(decks[:3]) + (", ..." if len(decks) > 3 else "")
        print(f"{calc_type} decks ({len(decks)}: {names}); defaults from [resources.{calc_type}]:")
    else:
        print("Default SLURM resources for CRYSTAL D3 property calculations:")
    print(f"  • Cores (ntasks): {default_resources['ntasks']}")
    mem_label, mem_key = _memory_label(default_resources)
    print(f"  • {'Total Memory' if mem_label == 'Total memory' else mem_label}: {default_resources[mem_key]}")
    print(f"  • Walltime: {default_resources['walltime']}")
    print(f"  • Account: {_site().account_label(default_resources['account'])}")

    # Show node exclusion default
    if default_resources.get('node_exclusion'):
        print(f"  • Node Exclusion: AMD20 nodes (amr + nvf) [ENABLED by default]")
    else:
        print(f"  • Node Exclusion: None")

    # Calculate per-CPU memory for display
    mem_str = default_resources.get('memory', '').upper()
    if mem_str.endswith('G'):
        mem_val = int(mem_str.rstrip('G'))
        per_cpu_gb = mem_val // default_resources['ntasks']
        print(f"  • Memory per CPU: ~{per_cpu_gb}G ({default_resources['memory']} ÷ {default_resources['ntasks']} cores)")

    print("\n📊 D3 Resource Notes:")
    print("  • BAND calculations: Memory-intensive, especially for large k-point meshes")
    print("  • DOSS calculations: Moderate memory, time depends on energy range/resolution")
    print("  • TRANSPORT calculations: High memory for Boltzmann transport equations")
    print("  • CHARGE+POTENTIAL: High memory for charge density and potential maps")
    print("  • ORBITALS: As a NEWK on its mesh (Γ or IS 2); writes one Molden file per k point")
    if default_resources.get('node_exclusion') or _site().msu_choices():
        print("\n⚠️  AMD20 node exclusion is ENABLED by default (recommended for CRYSTAL23)")

    modify = input("\nCustomize resources? (y/n) [n]: ").strip().lower()
    if modify not in ['y', 'yes']:
        return default_resources

    print("\nCustomizing SLURM resources:")
    resources = {}

    # Cores
    resources['ntasks'] = get_safe_integer_input(
        f"  Cores (ntasks) [{default_resources['ntasks']}]: ",
        default_resources['ntasks'], 1, _max_cores(default_resources['ntasks'])
    )

    # Total Memory
    resources[mem_key] = get_safe_memory_input(
        f"  {mem_label} [{default_resources[mem_key]}]: ",
        default_resources[mem_key]
    )
    if default_resources.get('mem_form'):
        resources['mem_form'] = default_resources['mem_form']

    # Walltime
    resources['walltime'] = get_safe_walltime_input(
        f"  Walltime [{default_resources['walltime']}]: ",
        default_resources['walltime']
    )

    # Account
    new_account = input(f"  Account [{_site().account_label(default_resources['account'])}]: ").strip()
    resources['account'] = new_account if new_account else default_resources['account']

    # Node exclusion - default is AMD20, allow customization
    if NODE_EXCLUSION_AVAILABLE:
        print("\n⚠️  Node Exclusion (AMD20 nodes excluded by default)" if _site().msu_choices()
              else "\nNode Exclusion (none by default)")
        exclude_choice = input("Change node exclusion? (y/n) [n]: ").strip().lower()
        if exclude_choice in ['y', 'yes']:
            resources['node_exclusion'] = prompt_node_exclusion()
        else:
            # Keep the default AMD20 exclusion
            resources['node_exclusion'] = default_resources.get('node_exclusion')
    else:
        resources['node_exclusion'] = None

    print(f"\nFinal resource configuration:")
    print(f"  • Cores: {resources['ntasks']}")
    print(f"  • {'Total Memory' if mem_label == 'Total memory' else mem_label}: {resources[mem_key]}")
    print(f"  • Walltime: {resources['walltime']}")
    print(f"  • Account: {_site().account_label(resources['account'])}")
    if resources.get('node_exclusion'):
        print(f"  • Node Exclusion: {resources['node_exclusion']}")

    return resources

def prompt_node_exclusion():
    """Prompt user for node exclusion configuration - delegates to NodeExclusionManager"""
    if not NODE_EXCLUSION_AVAILABLE:
        return None

    node_manager = NodeExclusionManager()
    # Use the manager's interactive method which has all the updated logic
    return node_manager.interactive_node_exclusion()

def create_custom_slurm_script(script_path, resources, tag=""):
    """Create a customized SLURM script with user-specified resources"""

    # Read the original script
    with open(script_path, 'r') as f:
        content = f.read()

    # Apply resource customizations
    lines = content.split('\n')
    modified_lines = []

    # Track if we need to add node exclusion
    last_sbatch_line = -1

    for i, line in enumerate(lines):
        # Modify resource directives
        if line.startswith("echo '#SBATCH --ntasks="):
            modified_lines.append(f"echo '#SBATCH --ntasks={resources['ntasks']}' >> $1.sh")
            last_sbatch_line = len(modified_lines) - 1
        elif line.startswith("echo '#SBATCH -t "):
            modified_lines.append(f"echo '#SBATCH -t {resources['walltime']}' >> $1.sh")
            last_sbatch_line = len(modified_lines) - 1
        elif line.startswith("echo '#SBATCH --mem=") or line.startswith("echo '#SBATCH --mem-per-cpu="):
            # the site's memory form when it set one (site.memory_line), else this line's
            site_line = _site().memory_line(line, resources)
            if site_line:
                modified_lines.append(site_line)
            elif line.startswith("echo '#SBATCH --mem="):
                modified_lines.append(f"echo '#SBATCH --mem={resources['memory']}' >> $1.sh")
            else:
                modified_lines.append(line)
            last_sbatch_line = len(modified_lines) - 1
        elif line.startswith("echo '#SBATCH -A "):
            # None: no -A line, SLURM charges the default account (site.py)
            if resources.get('account'):
                modified_lines.append(f"echo '#SBATCH -A {resources['account']}' >> $1.sh")
            last_sbatch_line = len(modified_lines) - 1
        elif line.startswith("echo '#SBATCH"):
            modified_lines.append(line)
            last_sbatch_line = len(modified_lines) - 1
        else:
            modified_lines.append(line)

    # Add node exclusion if specified
    if resources.get('node_exclusion') and last_sbatch_line >= 0:
        exclude_line = f"echo '#SBATCH --exclude={resources['node_exclusion']}' >> $1.sh"
        modified_lines.insert(last_sbatch_line + 1, exclude_line)

    # The temporary generator goes to a private folder in the user's temp
    # folder, not beside the template: an install can be read-only (a group
    # or site-wide copy), and two runs must not share a file. It still pins
    # the job to this install (temp_generator).
    return _temp_generator().write(script_path, '\n'.join(modified_lines),
                                   f"submit_prop_custom_{os.getpid()}{tag}.sh")


def _temp_generator():
    try:
        from mace.submission import temp_generator
    except ImportError:  # run as a script from mace/submission/
        import temp_generator
    return temp_generator


def remove_custom_slurm_script(path):
    """Remove a generator create_custom_slurm_script wrote (and its folder);
    says so when it cannot."""
    _temp_generator().remove(path)

def wavefunction_problem(data_folder, submit_name):
    """'missing' or 'empty' when <name>.f9, the wavefunction a properties job
    reads (its job script stages it as fort.9), is not usable; else None."""
    f9 = Path(data_folder) / f"{submit_name}.f9"
    try:
        return None if f9.stat().st_size > 0 else 'empty'
    except OSError:
        return 'missing'


def _refuse_without_wavefunction(data_folder, submit_name):
    """Say why a .d3 is not submitted when its .f9 is missing or empty, and
    return True; False when it can be submitted."""
    problem = wavefunction_problem(data_folder, submit_name)
    if not problem:
        return False
    print(f"  Error: {submit_name}.f9 is {problem} in {data_folder} - a properties job reads the "
          f"wavefunction of its SCF run from it. Copy that run's .f9 next to {submit_name}.d3 as "
          f"{submit_name}.f9 and submit again. Not submitted.")
    return True


def _refuse_orbitals_problem(data_folder, submit_name):
    """Say why an ORBITALS deck is not submitted when it is known to fail
    (ICAR and ILOC on one line, ILOC not 0, a two-component parent; as
    `mace preflight` checks it) and return True; False otherwise."""
    try:
        from mace.submission.preflight import is_orbitals_deck, orbitals_deck_problem, read_deck
    except ImportError:  # run as a script from mace/submission/
        from preflight import is_orbitals_deck, orbitals_deck_problem, read_deck
    deck = Path(data_folder) / f"{submit_name}.d3"
    try:
        text = read_deck(deck)
    except OSError:
        return False
    if not is_orbitals_deck(text):
        return False
    problem = orbitals_deck_problem(deck, text)
    if not problem:
        return False
    print(f"  Error: {submit_name}.d3: {problem}. Not submitted.")
    return True


def _job_name_helpers():
    """(unusable_job_name, unusable_name_message) of job_script_age."""
    try:
        from mace.submission.job_script_age import unusable_job_name, unusable_name_message
    except ImportError:  # run as a script from mace/submission/
        from job_script_age import unusable_job_name, unusable_name_message
    return unusable_job_name, unusable_name_message


def _refuse_unusable_name(file_name):
    """Say why a deck whose name a job script cannot use is skipped (True)."""
    unusable, message = _job_name_helpers()
    if unusable(file_name.rsplit(".", 1)[0]):
        print(message(file_name))
        return True
    return False


def _reuse_conflicts(data_folder, names, overwrite_sh, resources_given, **kw):
    try:
        from mace.submission.job_script_age import reuse_conflicts
    except ImportError:  # run as a script from mace/submission/
        from job_script_age import reuse_conflicts
    return reuse_conflicts(data_folder, names, overwrite_sh, resources_given, **kw)


def _conflict_message(conflicts, options):
    try:
        from mace.submission.job_script_age import conflict_message
    except ImportError:  # run as a script from mace/submission/
        from job_script_age import conflict_message
    return conflict_message(conflicts, options)


def _remove_custom_scripts(paths):
    """Remove the temporary -i / walltime generators of this run."""
    for custom_script in paths:
        remove_custom_slurm_script(custom_script)


def check_existing_sh_file(data_folder, submit_name):
    """Check if corresponding .sh file already exists"""
    return Path(data_folder) / f"{submit_name}.sh"

def _warn_if_pre_guard(script_path):
    """Say so when the reused script was written by a MACE older than 1.1.3
    (no scratch guard, no GUESSP staging); it is left as it is."""
    try:
        from mace.submission.job_script_age import predates_scratch_guard, reused_script_warning
    except ImportError:  # run as a script from mace/submission/
        from job_script_age import predates_scratch_guard, reused_script_warning
    try:
        text = Path(script_path).read_text(errors='ignore')
    except OSError:
        return
    if predates_scratch_guard(text):
        print(reused_script_warning(Path(script_path).name, properties=True))


def generate_or_use_script(script_to_use, submit_name, data_folder, overwrite_sh,
                           env=None, outcome=None, regenerate=False):
    """Generate .sh file or use existing one based on flags.

    env: the generator's environment (job arrays run it with an sbatch that
    refuses to submit); outcome: a dict that receives the generator's result;
    regenerate: write an existing <name>.sh again (MACE wrote it and nobody
    changed it, and resource options were given).
    """
    existing_sh = check_existing_sh_file(data_folder, submit_name)

    if existing_sh.exists() and not overwrite_sh and not regenerate:
        print(f"  Using existing script: {existing_sh.name}")
        _warn_if_pre_guard(existing_sh)
        return existing_sh, False  # (script_path, was_generated)
    else:
        if existing_sh.exists() and overwrite_sh:
            print(f"  Overwriting existing script: {existing_sh.name}")
        elif existing_sh.exists():
            print(f"  Writing {existing_sh.name} again with the requested resources (MACE wrote it; unchanged since)")

        # Generate new script by running the submission script generator.
        # The generator only writes <name>.sh; actual submission happens in
        # main() (the trailing sbatch in the template is commented out).
        result = subprocess.run(
            ['bash', str(script_to_use), submit_name, '100'],
            capture_output=True, text=True, env=env
        )
        if outcome is not None:
            outcome['result'] = result

        if result.returncode == 0 and existing_sh.exists():
            return existing_sh, True  # (script_path, was_generated)
        else:
            if result.returncode != 0 and result.stderr:
                print(f"  Script generator error: {result.stderr.strip()}")
            return None, False

def main():
    """Main function to submit D3 property files"""
    # Check for flags
    interactive_mode = '--interactive' in sys.argv
    if interactive_mode:
        sys.argv.remove('--interactive')

    nosubmit_mode = '--nosubmit' in sys.argv
    if nosubmit_mode:
        sys.argv.remove('--nosubmit')

    overwrite_sh = '--overwrite-sh' in sys.argv
    if overwrite_sh:
        sys.argv.remove('--overwrite-sh')

    # Check for node exclusion argument
    exclude_string = None
    if '--exclude-nodes' in sys.argv:
        idx = sys.argv.index('--exclude-nodes')
        if idx + 1 < len(sys.argv):
            exclude_string = sys.argv[idx + 1]
            sys.argv.pop(idx)  # Remove --exclude-nodes
            sys.argv.pop(idx)  # Remove the value

    # Walltime override. Must be consumed here even though the D3 default is
    # already short: leftover flags in sys.argv are otherwise taken for input
    # filenames by the directory-mode branch below.
    walltime_override = None
    if '--walltime' in sys.argv:
        idx = sys.argv.index('--walltime')
        if idx + 1 < len(sys.argv):
            walltime_override = sys.argv[idx + 1]
            sys.argv.pop(idx)
            sys.argv.pop(idx)

    # Job arrays (mace submit --array [--max-running N]): the decks' job
    # scripts are written as always, then submitted as SLURM job arrays.
    array_mode = '--array' in sys.argv
    if array_mode:
        sys.argv.remove('--array')
    max_running = None
    if '--max-running' in sys.argv:
        idx = sys.argv.index('--max-running')
        if idx + 1 < len(sys.argv):
            max_running = int(sys.argv[idx + 1])
            sys.argv.pop(idx)
            sys.argv.pop(idx)

    _site_check()

    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    submit_prop_script = script_dir / "submit_prop.sh"

    # Check if submit_prop.sh exists
    if not submit_prop_script.exists():
        print(f"Error: submit_prop.sh not found at {submit_prop_script}")
        sys.exit(1)

    # Get target from command line or use current directory
    if len(sys.argv) > 1:
        target = sys.argv[1]
        # Absolute: the loop below changes into data_folder and then looks
        # for <data_folder>/<job>.sh, which a relative path would resolve
        # against the wrong directory (as in crystal.py).
        if os.path.isfile(target) and target.endswith('.d3'):
            # Single file submission
            data_folder = os.path.dirname(os.path.abspath(target))
            data_files = [os.path.basename(target)]
        elif os.path.isdir(target):
            # Directory submission
            data_folder = os.path.abspath(target)
            data_files = os.listdir(data_folder)
        else:
            print(f"Error: {target} is not a valid D3 file or directory")
            sys.exit(1)
    else:
        # No argument provided, use current directory
        data_folder = os.getcwd()
        data_files = os.listdir(data_folder)

    # The decks' folder's mace-folder.toml, not the working folder's: the
    # completion callback and follow-ups run in data_folder and read that one.
    _site().use_folder(data_folder)
    _site().say_warnings()

    # Count D3 files
    d3_files = [f for f in data_files if f.endswith(".d3")]
    if not d3_files:
        print(f"No D3 files found in {data_folder}")
        return

    # Resource options go into the job scripts MACE writes: an existing script
    # it wrote and nobody changed is written again with them, and one that was
    # edited is never replaced - reusing it would silently drop the options.
    given = [o for o, on in (("-i", interactive_mode), ("--walltime", walltime_override),
                             ("node exclusion", exclude_string)) if on]
    regenerate, conflicts = _reuse_conflicts(data_folder, [f.rsplit(".", 1)[0] for f in d3_files],
                                             overwrite_sh, bool(given), kind="prop",
                                             walltime=walltime_override, exclude=exclude_string,
                                             interactive=bool(interactive_mode))
    if conflicts:
        print(_conflict_message(conflicts, " / ".join(given)))
        return 1

    # One generator per group of decks: all decks in one group, unless the
    # site settings have [resources.<TYPE>] tables (BAND, DOSS, TRANSPORT,
    # CHARGE+POTENTIAL, ORBITALS), whose decks then get a generator (and -i prompts)
    # of their own, or size-aware suggestions ([resources] auto): one group
    # per suggested job size.
    groups = _site().deck_groups(data_folder, d3_files, "prop")
    refusal = _site().piped_groups_refusal(groups) if interactive_mode else None
    if refusal:
        print(f"Error: {refusal}")
        return 1
    custom_scripts = []
    generators = {}
    for number, (calc_type, auto, names) in enumerate(groups):
        # Configure resources if interactive mode
        custom_script = None
        tag = "_" + calc_type.lower().replace("+", "_") if calc_type else ""
        if auto:
            tag += f"_{number}"
            if not interactive_mode:
                _site().say_suggestion(auto, names)
        if interactive_mode:
            resources = configure_interactive_resources(calc_type, names, auto)
            # If exclude_string was provided via command line, use it
            if exclude_string and not resources.get('node_exclusion'):
                resources['node_exclusion'] = exclude_string
            if walltime_override:
                resources['walltime'] = walltime_override
            custom_script = create_custom_slurm_script(submit_prop_script, resources, tag)
        elif exclude_string or walltime_override:
            # Non-interactive mode but exclusion and/or walltime specified
            resources = get_default_d3_resources(calc_type, auto)
            if exclude_string:
                resources['node_exclusion'] = exclude_string
            if walltime_override:
                resources['walltime'] = walltime_override
            custom_script = create_custom_slurm_script(submit_prop_script, resources, tag)
        # a walltime given on the command line or typed at the prompt is
        # written as given; say so when the partition allows less
        if custom_script is not None:
            warning = _site().walltime_over_limit(resources.get('walltime'))
            if warning:
                print(f"  {warning}")
        if custom_script:
            custom_scripts.append(custom_script)
        # Use custom script if created, otherwise use original
        script_to_use = custom_script if custom_script else submit_prop_script
        script_to_use = _site_generator(script_to_use, custom_script, data_folder, calc_type, auto)
        for name in names:
            generators[name] = generators[name.split(".d3")[0]] = script_to_use

    if nosubmit_mode:
        print(f"Found {len(d3_files)} D3 file(s) to generate scripts for")
    else:
        print(f"Found {len(d3_files)} D3 property file(s) to submit")

    if array_mode:
        from mace.submission.job_array import submit_untracked_arrays
        names = [f.split(".d3")[0] for f in d3_files if not _refuse_unusable_name(f)]
        refused = len(d3_files) - len(names)
        if not nosubmit_mode:
            # never queue a properties job without the wavefunction it reads
            kept = [n for n in names if not _refuse_without_wavefunction(data_folder, n)
                    and not _refuse_orbitals_problem(data_folder, n)]
            refused, names = refused + len(names) - len(kept), kept
        try:
            ids = submit_untracked_arrays(
                names, data_folder,
                lambda name, env=None, outcome=None: generate_or_use_script(
                    generators[name], name, data_folder, overwrite_sh, env=env, outcome=outcome,
                    regenerate=name in regenerate),
                nosubmit=nosubmit_mode, max_running=max_running, label="d3")
        finally:
            _remove_custom_scripts(custom_scripts)
        if not nosubmit_mode:
            more, _left = _budget().offer_reserve()
            ids = list(ids or []) + [j for j in more if j]
        if ids:
            print(f"\nSubmitted {len(ids)} job(s): {', '.join(ids)}")
        if nosubmit_mode:
            failed = refused + sum(1 for n in names if not (Path(data_folder) / f"{n}.sh").exists())
        else:
            failed = refused + max(0, len(names) - len(ids))
        return _finish(nosubmit_mode, failed, len(names) + refused)

    submitted_job_ids = []
    held_names = []
    failed = 0
    try:
        # Process each D3 file
        for file_name in d3_files:
            submit_name = file_name.split(".d3")[0]

            if nosubmit_mode:
                print(f"Generating script for: {file_name}")
            else:
                print(f"Processing: {file_name}")
            if _refuse_unusable_name(file_name):
                failed += 1
                continue
            # never queue a properties job without the wavefunction it reads
            if not nosubmit_mode and _refuse_without_wavefunction(data_folder, submit_name):
                failed += 1
                continue
            # nor an ORBITALS deck known to fail (mace preflight's check)
            if not nosubmit_mode and _refuse_orbitals_problem(data_folder, submit_name):
                failed += 1
                continue

            # Change to the directory containing the D3 file
            original_dir = os.getcwd()
            os.chdir(data_folder)

            # Generate or use existing script
            sh_script, was_generated = generate_or_use_script(
                generators[file_name], submit_name, data_folder, overwrite_sh,
                regenerate=submit_name in regenerate
            )

            if sh_script:
                if nosubmit_mode:
                    if was_generated:
                        print(f"  ✓ Generated script: {sh_script.name}")
                    else:
                        print(f"  ✓ Using existing script: {sh_script.name}")
                else:
                    # Submit the script
                    if sh_script.exists():
                        print(f"  Submitting: {sh_script.name}")
                        # through MACE's user-wide queue budget (mace/queue/budget.py)
                        submit_result = _budget_sbatch(['sbatch', sh_script.name])   # _note_submission() inside
                        if submit_result.held:
                            held_names.append(sh_script.name)
                            print(f"  Not submitted now: {sh_script.name} ({submit_result.message})")
                        elif not submit_result.submitted:
                            failed += 1
                            print(f"  Warning: Failed to submit {sh_script.name}")
                            if submit_result.message:
                                print(f"    {submit_result.message}")
                            _site_hint(submit_result.message)
                        else:
                            job_id = submit_result.job_id
                            if job_id:
                                submitted_job_ids.append(job_id)
                                print(f"  ✓ Submitted successfully (job {job_id})")
                            else:
                                print(f"  ✓ Submitted successfully")
                    else:
                        failed += 1
                        print(f"  Error: Script {sh_script.name} not found")
            else:
                failed += 1
                print(f"  Error: Failed to generate/find script for {file_name}")

            # Return to original directory
            os.chdir(original_dir)

        if held_names and not nosubmit_mode:
            # untracked: no row to hold them in; the reserve only after a yes
            more, left = _budget().offer_reserve()
            submitted_job_ids.extend(j for j in more if j)
            failed += len(left)
        if submitted_job_ids and not nosubmit_mode:
            print(f"\nSubmitted {len(submitted_job_ids)} job(s): "
                  f"{', '.join(submitted_job_ids)}")

    finally:
        # Cleanup: Remove temporary custom script if created
        _remove_custom_scripts(custom_scripts)

    return _finish(nosubmit_mode, failed, len(d3_files))


def _finish(nosubmit_mode, failed, total):
    """The closing line of a run, and its exit status: 1 when any deck was
    not submitted (or, with --nosubmit, got no job script)."""
    if nosubmit_mode:
        if failed:
            print(f"\nScript generation incomplete: {failed} of {total} job script(s) not written (see above).")
            return 1
        print(f"\nScript generation complete. Scripts are ready for submission.")
        return 0
    if not failed:
        print(f"\nSubmission complete. Use 'mace monitor' to track job status.")
        return 0
    if failed >= total:
        print(f"\nSubmission failed: none of the {total} job(s) was submitted (see above).")
    else:
        print(f"\nSubmission incomplete: {failed} of {total} job(s) not submitted (see above). "
              f"Use 'mace monitor' to track the others.")
    return 1

if __name__ == "__main__":
    sys.exit(main())
