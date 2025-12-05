#!/usr/bin/env python3
"""
This script finds all .d3 files in the current directory and submits them for processing
by running submit_prop.sh with a parameter of 100 for each file.
"""
import os, sys, math
import re
import linecache
import shutil
import itertools
from pathlib import Path

# Try to import node exclusion manager
try:
    from mace.utils.node_exclusion import NodeExclusionManager
    NODE_EXCLUSION_AVAILABLE = True
except ImportError:
    NODE_EXCLUSION_AVAILABLE = False

def get_default_amd20_exclusion():
    """Get default AMD20 node exclusion string (amr + nvf)"""
    if not NODE_EXCLUSION_AVAILABLE:
        return None
    try:
        manager = NodeExclusionManager()
        amd20_nodes = manager.get_amd20_nodes()
        if amd20_nodes:
            return manager.create_exclude_string(amd20_nodes)
    except Exception as e:
        print(f"Warning: Could not query AMD20 nodes: {e}")
    return None

def get_default_d3_resources():
    """Get default SLURM resources for D3 calculations"""
    return {
        'ntasks': 28,
        'memory': '80G',
        'walltime': '2:00:00',
        'account': 'mendoza_q',
        'node_exclusion': get_default_amd20_exclusion()  # Default: exclude AMD20 nodes
    }

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
        if re.match(r'^\d+(-\d{2}:\d{2}:\d{2}|\d{2}:\d{2}:\d{2}|\d{2}:\d{2})$', user_input):
            return user_input
        else:
            print("Please enter walltime in format like '2:00:00', '12:00:00', or '1-12:00:00'")

def configure_interactive_resources():
    """Interactively configure SLURM resources for D3 calculations"""
    print("\n" + "="*60)
    print("INTERACTIVE SLURM RESOURCE CONFIGURATION")
    print("="*60)

    default_resources = get_default_d3_resources()

    print("Default SLURM resources for CRYSTAL D3 property calculations:")
    print(f"  • Cores (ntasks): {default_resources['ntasks']}")
    print(f"  • Total Memory: {default_resources['memory']}")
    print(f"  • Walltime: {default_resources['walltime']}")
    print(f"  • Account: {default_resources['account']}")

    # Show node exclusion default
    if default_resources.get('node_exclusion'):
        print(f"  • Node Exclusion: AMD20 nodes (amr + nvf) [ENABLED by default]")
    else:
        print(f"  • Node Exclusion: None")

    # Calculate per-CPU memory for display
    mem_str = default_resources['memory'].upper()
    if mem_str.endswith('G'):
        mem_val = int(mem_str.rstrip('G'))
        per_cpu_gb = mem_val // default_resources['ntasks']
        print(f"  • Memory per CPU: ~{per_cpu_gb}G ({default_resources['memory']} ÷ {default_resources['ntasks']} cores)")

    print("\n📊 D3 Resource Notes:")
    print("  • BAND calculations: Memory-intensive, especially for large k-point meshes")
    print("  • DOSS calculations: Moderate memory, time depends on energy range/resolution")
    print("  • TRANSPORT calculations: High memory for Boltzmann transport equations")
    print("  • CHARGE+POTENTIAL: High memory for charge density and potential maps")
    print("\n⚠️  AMD20 node exclusion is ENABLED by default (recommended for CRYSTAL23)")

    modify = input("\nCustomize resources? (y/n) [n]: ").strip().lower()
    if modify not in ['y', 'yes']:
        return default_resources

    print("\nCustomizing SLURM resources:")
    resources = {}

    # Cores
    resources['ntasks'] = get_safe_integer_input(
        f"  Cores (ntasks) [{default_resources['ntasks']}]: ",
        default_resources['ntasks'], 1, 128
    )

    # Total Memory
    resources['memory'] = get_safe_memory_input(
        f"  Total memory [{default_resources['memory']}]: ",
        default_resources['memory']
    )

    # Walltime
    resources['walltime'] = get_safe_walltime_input(
        f"  Walltime [{default_resources['walltime']}]: ",
        default_resources['walltime']
    )

    # Account
    new_account = input(f"  Account [{default_resources['account']}]: ").strip()
    resources['account'] = new_account if new_account else default_resources['account']

    # Node exclusion - default is AMD20, allow customization
    if NODE_EXCLUSION_AVAILABLE:
        print("\n⚠️  Node Exclusion (AMD20 nodes excluded by default)")
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
    print(f"  • Total Memory: {resources['memory']}")
    print(f"  • Walltime: {resources['walltime']}")
    print(f"  • Account: {resources['account']}")
    if resources.get('node_exclusion'):
        print(f"  • Node Exclusion: {resources['node_exclusion']}")

    return resources

def prompt_node_exclusion():
    """Prompt user for node exclusion configuration"""
    if not NODE_EXCLUSION_AVAILABLE:
        return None

    node_manager = NodeExclusionManager()

    print("\n  " + "="*60)
    print("  SLURM Node Exclusion Configuration")
    print("  " + "="*60)

    print("\n  Select node exclusion option:")
    print("  1) No exclusions (use all available nodes)")
    print("  2) Exclude all AMD20 nodes (amr + nvf types) [RECOMMENDED]")
    print("  3) Exclude Mendoza group nodes (agg-[011-012], amr-[163,178-179])")
    print("  4) Exclude all nodes of a specific type (amr, nvf, agg, etc.)")
    print("  5) Custom node exclusion list")

    choice = input("\n  Enter choice [1-5] (default: 2): ").strip() or "2"

    if choice == "1":
        print("  No node exclusions will be applied.")
        return None

    elif choice == "2":
        # Exclude AMD20 nodes
        print("  Querying SLURM for all AMD20 nodes (amr + nvf)...")
        amd20_nodes = node_manager.get_amd20_nodes()
        if amd20_nodes:
            amr_count = sum(1 for n in amd20_nodes if n.startswith('amr-'))
            nvf_count = sum(1 for n in amd20_nodes if n.startswith('nvf-'))
            print(f"  Found {len(amd20_nodes)} AMD20 nodes:")
            print(f"    - amr (AMD EPYC 7452): {amr_count} nodes")
            print(f"    - nvf (AMD EPYC 7452 + V100): {nvf_count} nodes")
            exclude_str = node_manager.create_exclude_string(amd20_nodes)
            print(f"  Exclude string: {exclude_str}")
            return exclude_str
        else:
            print("  Warning: No AMD20 nodes found")
            return None

    elif choice == "3":
        exclude_str = node_manager.create_exclude_string(
            node_manager.MENDOZA_NODES
        )
        print(f"  Excluding Mendoza nodes: {exclude_str}")
        return exclude_str

    elif choice == "4":
        return exclude_by_type_prompt(node_manager)

    elif choice == "5":
        return custom_exclusion_prompt(node_manager)

    else:
        print("  Invalid choice. No exclusions will be applied.")
        return None

def exclude_by_type_prompt(node_manager):
    """Handle exclusion of all nodes by type"""
    print("\n  Available node types:")
    for i, node_type in enumerate(node_manager.KNOWN_NODE_TYPES, 1):
        print(f"  {i}) {node_type}")
    print(f"  {len(node_manager.KNOWN_NODE_TYPES) + 1}) Enter custom type")

    choice = input(
        f"\n  Select node type [1-{len(node_manager.KNOWN_NODE_TYPES) + 1}]: "
    ).strip()

    try:
        choice_num = int(choice)
        if 1 <= choice_num <= len(node_manager.KNOWN_NODE_TYPES):
            node_type = node_manager.KNOWN_NODE_TYPES[choice_num - 1]
        elif choice_num == len(node_manager.KNOWN_NODE_TYPES) + 1:
            node_type = input("  Enter custom node type prefix: ").strip()
        else:
            print("  Invalid choice.")
            return None
    except ValueError:
        print("  Invalid input.")
        return None

    print(f"\n  Querying SLURM for all '{node_type}' nodes...")
    nodes = node_manager.query_nodes_by_type(node_type)

    if not nodes:
        print(f"  No nodes found for type '{node_type}'")
        return None

    print(f"  Found {len(nodes)} nodes: {nodes[0]} to {nodes[-1]}")

    confirm = input(
        f"\n  Exclude all {len(nodes)} '{node_type}' nodes? [y/N]: "
    ).strip().lower()

    if confirm == 'y':
        exclude_str = node_manager.create_exclude_string(nodes)
        print(f"  Exclude string: {exclude_str}")
        return exclude_str
    else:
        print("  Exclusion cancelled.")
        return None

def custom_exclusion_prompt(node_manager):
    """Handle custom node exclusion list"""
    print("\n  Enter nodes to exclude (comma-separated).")
    print("  Examples:")
    print("    - Single nodes: amr-042,amr-050,nvf-123")
    print("    - With ranges: amr-[042-050],nvf-[100-110]")

    custom_input = input("\n  Nodes to exclude: ").strip()

    if not custom_input:
        print("  No exclusions specified.")
        return None

    # Check if already in SLURM format
    if '[' in custom_input and ']' in custom_input:
        print(f"  Using provided exclude string: {custom_input}")
        return custom_input

    # Parse comma-separated node names
    nodes = [n.strip() for n in custom_input.split(',')]

    # Group by prefix
    node_groups = {}
    for node in nodes:
        match = re.match(r'^([a-z]+)-(\d+)$', node)
        if match:
            prefix = match.group(1)
            if prefix not in node_groups:
                node_groups[prefix] = []
            node_groups[prefix].append(node)
        else:
            print(f"  Warning: Invalid node format '{node}', skipping")

    # Create exclude strings for each type
    exclude_dict = {}
    for prefix, prefix_nodes in node_groups.items():
        exclude_dict[prefix] = prefix_nodes

    exclude_str = node_manager.create_multi_type_exclude_string(exclude_dict)
    print(f"  Compact exclude string: {exclude_str}")

    return exclude_str

def create_custom_slurm_script(script_path, resources):
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
        elif line.startswith("echo '#SBATCH --mem="):
            modified_lines.append(f"echo '#SBATCH --mem={resources['memory']}' >> $1.sh")
            last_sbatch_line = len(modified_lines) - 1
        elif line.startswith("echo '#SBATCH -A "):
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

    # Create temporary customized script
    custom_script_path = script_path.parent / f"submit_prop_custom_{os.getpid()}.sh"

    with open(custom_script_path, 'w') as f:
        f.write('\n'.join(modified_lines))

    # Make executable
    custom_script_path.chmod(0o755)

    return custom_script_path

def check_existing_sh_file(data_folder, submit_name):
    """Check if corresponding .sh file already exists"""
    return Path(data_folder) / f"{submit_name}.sh"

def generate_or_use_script(script_to_use, submit_name, data_folder, overwrite_sh):
    """Generate .sh file or use existing one based on flags"""
    existing_sh = check_existing_sh_file(data_folder, submit_name)

    if existing_sh.exists() and not overwrite_sh:
        print(f"  Using existing script: {existing_sh.name}")
        return existing_sh, False  # (script_path, was_generated)
    else:
        if existing_sh.exists() and overwrite_sh:
            print(f"  Overwriting existing script: {existing_sh.name}")

        # Generate new script by running the submission script generator
        cmd = f"{script_to_use} {submit_name} 100"
        result = os.system(cmd)

        if result == 0 and existing_sh.exists():
            return existing_sh, True  # (script_path, was_generated)
        else:
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

    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    submit_prop_script = script_dir / "submit_prop.sh"

    # Check if submit_prop.sh exists
    if not submit_prop_script.exists():
        print(f"Error: submit_prop.sh not found at {submit_prop_script}")
        sys.exit(1)

    # Configure resources if interactive mode
    custom_script = None
    if interactive_mode:
        resources = configure_interactive_resources()
        # If exclude_string was provided via command line, use it
        if exclude_string and not resources.get('node_exclusion'):
            resources['node_exclusion'] = exclude_string
        custom_script = create_custom_slurm_script(submit_prop_script, resources)
    elif exclude_string:
        # Non-interactive mode but exclusion specified
        resources = get_default_d3_resources()
        resources['node_exclusion'] = exclude_string
        custom_script = create_custom_slurm_script(submit_prop_script, resources)

    # Use custom script if created, otherwise use original
    script_to_use = custom_script if custom_script else submit_prop_script

    # Get target from command line or use current directory
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if os.path.isfile(target) and target.endswith('.d3'):
            # Single file submission
            data_folder = os.path.dirname(target) or os.getcwd()
            data_files = [os.path.basename(target)]
        elif os.path.isdir(target):
            # Directory submission
            data_folder = target
            data_files = os.listdir(data_folder)
        else:
            print(f"Error: {target} is not a valid D3 file or directory")
            sys.exit(1)
    else:
        # No argument provided, use current directory
        data_folder = os.getcwd()
        data_files = os.listdir(data_folder)

    # Count D3 files
    d3_files = [f for f in data_files if f.endswith(".d3")]
    if not d3_files:
        print(f"No D3 files found in {data_folder}")
        return

    if nosubmit_mode:
        print(f"Found {len(d3_files)} D3 file(s) to generate scripts for")
    else:
        print(f"Found {len(d3_files)} D3 property file(s) to submit")

    try:
        # Process each D3 file
        for file_name in d3_files:
            submit_name = file_name.split(".d3")[0]

            if nosubmit_mode:
                print(f"Generating script for: {file_name}")
            else:
                print(f"Processing: {file_name}")

            # Change to the directory containing the D3 file
            original_dir = os.getcwd()
            os.chdir(data_folder)

            # Generate or use existing script
            sh_script, was_generated = generate_or_use_script(
                script_to_use, submit_name, data_folder, overwrite_sh
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
                        submit_result = os.system(f"sbatch {sh_script.name}")
                        if submit_result != 0:
                            print(f"  Warning: Failed to submit {sh_script.name}")
                        else:
                            print(f"  ✓ Submitted successfully")
                    else:
                        print(f"  Error: Script {sh_script.name} not found")
            else:
                print(f"  Error: Failed to generate/find script for {file_name}")

            # Return to original directory
            os.chdir(original_dir)

    finally:
        # Cleanup: Remove temporary custom script if created
        if custom_script and custom_script.exists():
            try:
                custom_script.unlink()
            except Exception as e:
                print(f"Warning: Could not remove temporary script {custom_script}: {e}")

    if nosubmit_mode:
        print(f"\nScript generation complete. Scripts are ready for submission.")
    else:
        print(f"\nSubmission complete. Use 'mace monitor' to track job status.")

if __name__ == "__main__":
    main()
