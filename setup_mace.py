#!/usr/bin/env python3
"""
Setup script for MACE (Mendoza Automated CRYSTAL Engine)
Configures environment after cloning the repository

Developed by: Marcus Djokic (Primary Developer)
Contributors: Daniel Maldonado Lopez, Brandon Lewis, William Comaskey
Advisor: Prof. Jose Luis Mendoza-Cortes
Institution: Michigan State University, Mendoza Group
"""

import os
import sys
import subprocess
from pathlib import Path
import platform
import argparse

class MACESetup:
    def __init__(self, shell=None):
        # Detect repository location
        self.script_dir = Path(__file__).resolve().parent
        self.repo_root = self.script_dir
        
        # Verify we're in the correct location by checking for key files
        required_files = [
            self.repo_root / 'mace' / 'enhanced_queue_manager.py',
            self.repo_root / 'Crystal_d12' / 'NewCifToD12.py',
            self.repo_root / 'Crystal_d3' / 'CRYSTALOptToD3.py',
            self.repo_root / 'mace_cli'
        ]
        
        missing_files = [f for f in required_files if not f.exists()]
        if missing_files:
            print("Error: This script must be run from the MACE repository root")
            print("Missing required files:")
            for f in missing_files:
                print(f"  - {f.relative_to(self.repo_root) if f.is_relative_to(self.repo_root) else f}")
            sys.exit(1)
        
        self.shell = shell if shell else self.detect_shell()
        self.shell_rc = self.get_shell_rc()
        
    def detect_shell(self):
        """Detect user's shell or ask interactively"""
        # First try to detect from environment
        detected_shell = os.environ.get('SHELL', '/bin/bash')
        detected_name = Path(detected_shell).name
        
        # Ask user to confirm or choose
        print(f"\nDetected shell: {detected_name}")
        print("Which shell configuration would you like to update?")
        print("1) zsh (.zshrc)")
        print("2) bash (.bashrc)")
        print("3) both")
        print(f"4) use detected ({detected_name})")
        
        while True:
            choice = input("\nEnter choice (1-4) [4]: ").strip() or '4'
            
            if choice == '1':
                return 'zsh'
            elif choice == '2':
                return 'bash'
            elif choice == '3':
                return 'both'
            elif choice == '4':
                return detected_name
            else:
                print("Invalid choice. Please enter 1-4.")
    
    def get_shell_rc(self):
        """Get shell configuration file(s)"""
        home = Path.home()
        
        if self.shell == 'both':
            # Return both config files
            files = []
            files.append(home / '.zshrc')
            if platform.system() == 'Darwin' and (home / '.bash_profile').exists():
                files.append(home / '.bash_profile')
            else:
                files.append(home / '.bashrc')
            return files
        elif self.shell == 'zsh':
            return home / '.zshrc'
        elif self.shell == 'bash':
            # Check for .bashrc vs .bash_profile on macOS
            if platform.system() == 'Darwin':
                if (home / '.bash_profile').exists():
                    return home / '.bash_profile'
            return home / '.bashrc'
        else:
            print(f"Warning: Unknown shell '{self.shell}', assuming bash")
            return home / '.bashrc'
    
    def check_existing_setup(self):
        """Check if MACE_HOME is already set"""
        current = os.environ.get('MACE_HOME')
        if current:
            print(f"\n⚠️  MACE_HOME is already set to: {current}")
            if current != str(self.repo_root):
                print(f"   This differs from current location: {self.repo_root}")
                response = input("\nUpdate to new location? [y/N]: ")
                return response.lower() == 'y'
            else:
                print("   ✓ Already pointing to this repository")
                return False
        return True
    
    def update_shell_config(self, add_to_path=True):
        """Update shell configuration file(s)"""
        export_line = f'export MACE_HOME="{self.repo_root}"'
        pythonpath_line = 'export PYTHONPATH="$MACE_HOME:$PYTHONPATH"'
        # Add all script directories to PATH
        # Using new structure - reorganized paths
        path_lines = [
            'export PATH="$MACE_HOME:$PATH"',  # Add root for mace.py and mace_cli
            'export PATH="$MACE_HOME/mace/submission:$PATH"',
            'export PATH="$MACE_HOME/mace/queue:$PATH"',
            'export PATH="$MACE_HOME/mace/database:$PATH"',
            'export PATH="$MACE_HOME/mace/recovery:$PATH"',
            'export PATH="$MACE_HOME/mace/utils:$PATH"',
            'export PATH="$MACE_HOME/Crystal_d12:$PATH"',
            'export PATH="$MACE_HOME/Crystal_d3:$PATH"',
            # Legacy paths for scripts not yet migrated
            'export PATH="$MACE_HOME/code/Check_Scripts:$PATH"',
            'export PATH="$MACE_HOME/code/Plotting_Scripts:$PATH"',
            'export PATH="$MACE_HOME/code/Post_Processing_Scripts:$PATH"',
            'export PATH="$MACE_HOME/code/Band_Alignment:$PATH"'
        ]
        
        # Create alias for mace command
        alias_line = 'alias mace="mace_cli"'
        
        # Handle multiple shell configs
        shell_files = self.shell_rc if isinstance(self.shell_rc, list) else [self.shell_rc]
        updated_any = False
        
        for shell_file in shell_files:
            # Read existing config
            if shell_file.exists():
                with open(shell_file, 'r') as f:
                    content = f.read()
            else:
                content = ""
            
            # Check if already configured
            lines_to_add = []
            
            if export_line not in content:
                lines_to_add.append(export_line)
            
            # Add PYTHONPATH
            if pythonpath_line not in content:
                lines_to_add.append(pythonpath_line)
            
            if add_to_path:
                for path_line in path_lines:
                    if path_line not in content:
                        lines_to_add.append(path_line)
                
                # Add alias for mace command
                if alias_line not in content:
                    lines_to_add.append(alias_line)
            
            if not lines_to_add:
                print(f"\n✓ {shell_file.name} already configured")
                continue
            
            # Add configuration
            with open(shell_file, 'a') as f:
                f.write("\n# MACE (Mendoza Automated CRYSTAL Engine) Configuration\n")
                for line in lines_to_add:
                    f.write(f"{line}\n")
            
            print(f"\n✓ Updated {shell_file.name} with:")
            for line in lines_to_add:
                print(f"  {line}")
            
            updated_any = True
        
        return updated_any
    
    def create_activation_script(self):
        """Create an activation script for temporary use"""
        activate_path = self.repo_root / 'activate_mace.sh'
        
        content = f'''#!/bin/bash
# MACE (Mendoza Automated CRYSTAL Engine) Activation Script
# Source this file to temporarily set up MACE environment
#
# Developed by: Marcus Djokic (Primary Developer)
# Contributors: Daniel Maldonado Lopez, Brandon Lewis, William Comaskey
# Mendoza Group, Michigan State University

export MACE_HOME="{self.repo_root}"

# Add MACE scripts to PATH - using reorganized structure
export PATH="$MACE_HOME:$PATH"  # For mace.py and mace_cli
export PATH="$MACE_HOME/mace/submission:$PATH"
export PATH="$MACE_HOME/mace/queue:$PATH"
export PATH="$MACE_HOME/mace/database:$PATH"
export PATH="$MACE_HOME/mace/recovery:$PATH"
export PATH="$MACE_HOME/mace/utils:$PATH"
export PATH="$MACE_HOME/Crystal_d12:$PATH"
export PATH="$MACE_HOME/Crystal_d3:$PATH"

# Legacy paths for scripts not yet migrated
export PATH="$MACE_HOME/code/Check_Scripts:$PATH"
export PATH="$MACE_HOME/code/Plotting_Scripts:$PATH"
export PATH="$MACE_HOME/code/Post_Processing_Scripts:$PATH"
export PATH="$MACE_HOME/code/Band_Alignment:$PATH"

# Add convenient alias
alias mace="mace_cli"

echo "MACE (Mendoza Automated CRYSTAL Engine) environment activated!"
echo "  MACE_HOME: $MACE_HOME"
echo "  Added to PATH:"
echo "    - mace/ (MACE core modules)"
echo "    - Crystal_d12 (D12 creation tools)"
echo "    - Crystal_d3 (D3 creation tools)"
echo "    - Legacy scripts (Check, Plotting, Post-Processing)"
'''
        
        with open(activate_path, 'w') as f:
            f.write(content)
        
        activate_path.chmod(0o755)
        print(f"\n✓ Created activation script: {activate_path}")
        print("  Use: source activate_mace.sh")
    
    def update_slurm_templates(self):
        """Update SLURM script templates with detected paths"""
        templates_updated = 0
        
        # List of SLURM script templates that might need updating
        slurm_scripts = [
            'code/Job_Scripts/submitcrystal23.sh',
            'code/Job_Scripts/submit_prop.sh'
        ]
        
        for script_path in slurm_scripts:
            full_path = self.repo_root / script_path
            if not full_path.exists():
                continue
            
            try:
                with open(full_path, 'r') as f:
                    content = f.read()
                
                # Check if it needs updating
                if 'MACE_HOME=/mnt/iscsi/UsefulScripts' in content or 'CRYSTAL_TOOLS_HOME=/mnt/iscsi/UsefulScripts' in content:
                    # Replace hardcoded path with dynamic detection
                    new_content = content.replace(
                        'export CRYSTAL_TOOLS_HOME=/mnt/iscsi/UsefulScripts/Codebase/reorganization',
                        f'# Auto-configured by setup script\nexport MACE_HOME={self.repo_root}'
                    ).replace(
                        'export MACE_HOME=/mnt/iscsi/UsefulScripts/Codebase/reorganization',
                        f'# Auto-configured by setup script\nexport MACE_HOME={self.repo_root}'
                    )
                    
                    with open(full_path, 'w') as f:
                        f.write(new_content)
                    
                    templates_updated += 1
                    print(f"  ✓ Updated {script_path}")
            except Exception as e:
                print(f"  ⚠️  Could not update {script_path}: {e}")
        
        if templates_updated > 0:
            print(f"\n✓ Updated {templates_updated} SLURM templates")
        
        return templates_updated
    
    def check_dependencies(self):
        """Check if required Python packages are installed"""
        required = ['numpy', 'matplotlib', 'ase', 'spglib', 'PyPDF2', 'yaml', 'pandas']
        missing = []
        
        print("\nChecking Python dependencies...")
        
        for package in required:
            try:
                __import__(package)
                print(f"  ✓ {package}")
            except ImportError:
                missing.append(package)
                print(f"  ✗ {package} (missing)")
        
        if missing:
            print(f"\nMissing packages: {', '.join(missing)}")
            print("Install with: pip install " + " ".join(missing))
            return False
        
        return True
    
    def create_example_commands(self):
        """Create example command script"""
        examples_path = self.repo_root / 'mace_examples.sh'
        
        content = f'''#!/bin/bash
# MACE (Mendoza Automated CRYSTAL Engine) Example Commands
# After sourcing your shell config or activate_mace.sh

echo "MACE (Mendoza Automated CRYSTAL Engine) Example Commands"
echo "=============================="
echo ""
echo "D12 Creation (Crystal_d12):"
echo "  # Convert CIF files to D12"
echo "  cd /path/to/cif/files"
echo "  NewCifToD12.py  # or python $MACE_HOME/code/Crystal_d12/NewCifToD12.py"
echo ""
echo "  # Create D12 from optimized structure"
echo "  CRYSTALOptToD12.py material.out"
echo ""
echo "D3 Creation (Crystal_d3):"
echo "  # Create property calculations"
echo "  CRYSTALOptToD3.py material.out --calc-type BAND"
echo "  CRYSTALOptToD3.py material.out --calc-type DOSS"
echo "  CRYSTALOptToD3.py material.out --calc-type TRANSPORT"
echo ""
echo "Job Management (Job_Scripts):"
echo "  # Run comprehensive workflow"
echo "  run_workflow.py --interactive"
echo ""
echo "  # Submit jobs"
echo "  submitcrystal23.sh jobname"
echo "  submit_prop.sh property_calc"
echo ""
echo "  # Monitor calculations"
echo "  enhanced_queue_manager.py --status"
echo "  material_monitor.py --action dashboard"
echo ""
echo "Analysis (Post_Processing_Scripts):"
echo "  # Extract properties from outputs"
echo "  grab_properties.py"
echo ""
echo "Plotting (Plotting_Scripts):"
echo "  # Plot band structures"
echo "  autoBands.py material.BAND"
echo "  ipBANDS_V2.py material.BAND  # Interactive plotting"
echo ""
echo "  # Plot DOS"
echo "  ipDOS_V2.py material.DOSS    # Interactive plotting"
echo ""
echo "  # Plot phonon bands"
echo "  autoPhononBands.py material.f25"
echo ""
echo "  # Create overview PDF"
echo "  OverviewPDF.py"
echo ""
echo "Current environment:"
echo "  MACE_HOME=$MACE_HOME"
echo "  All script directories are in PATH"
'''
        
        with open(examples_path, 'w') as f:
            f.write(content)
        
        examples_path.chmod(0o755)
        print(f"\n✓ Created examples script: {examples_path}")
    
    def run_setup(self, args):
        """Run the complete setup process"""
        # Try to import and show banner
        try:
            sys.path.insert(0, str(self.repo_root / 'code' / 'Job_Scripts'))
            from mace_banner import print_banner
            print_banner('banner')  # Use main banner style
        except:
            print("MACE (Mendoza Automated CRYSTAL Engine) Setup")
            print("=============================================")
        
        print(f"\nRepository location: {self.repo_root}")
        print(f"Detected shell: {self.shell}")
        shell_file_str = ', '.join(str(f) for f in self.shell_rc) if isinstance(self.shell_rc, list) else str(self.shell_rc)
        print(f"Shell config file: {shell_file_str}")
        
        # Check existing setup
        if not args.force and not self.check_existing_setup():
            print("\nSetup already complete. Use --force to reconfigure.")
            return
        
        # Update shell configuration
        if not args.no_shell:
            updated = self.update_shell_config(add_to_path=args.add_to_path)
            if updated:
                if isinstance(self.shell_rc, list):
                    print(f"\n⚠  Run 'source ~/.zshrc' or 'source ~/.bashrc' to apply changes")
                else:
                    print(f"\n⚠  Run 'source {self.shell_rc}' to apply changes")
                print("  or start a new terminal session")
        
        # Create activation script
        self.create_activation_script()
        
        # Update SLURM templates if requested
        if args.update_templates:
            self.update_slurm_templates()
        
        # Check dependencies
        if not args.skip_deps:
            self.check_dependencies()
        
        # Create examples
        self.create_example_commands()
        
        # Final instructions
        print("\n" + "="*50)
        print("Setup Complete!")
        print("="*50)
        print("\nTo start using MACE:")
        print(f"1. Source your shell config: source {self.shell_rc}")
        print("   OR")
        print("   Source the activation script: source activate_mace.sh")
        print("\n2. Verify setup: echo $MACE_HOME")
        print("\n3. See examples: ./mace_examples.sh")
        
        if args.add_to_path:
            print("\n4. You can now run MACE commands from anywhere:")
            print("   mace workflow --interactive       # Interactive workflow planning")
            print("   mace submit file.d12              # Submit calculations")
            print("   mace monitor --dashboard          # Monitor progress")
            print("   mace analyze --extract-properties # Extract properties")
            print("   mace convert --from-cif *.cif     # Convert CIF files")
            print("\n   Or use scripts directly:")
            print("   NewCifToD12.py                    # D12 creation")
            print("   CRYSTALOptToD3.py file.out        # D3 creation")
            print("   run_workflow.py --interactive     # Workflow management")
            print("   enhanced_queue_manager.py --help  # Queue management")

def main():
    parser = argparse.ArgumentParser(
        description='Setup MACE (Mendoza Automated CRYSTAL Engine) environment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic setup
  python setup_crystal_tools.py
  
  # Add scripts to PATH
  python setup_crystal_tools.py --add-to-path
  
  # Update SLURM templates with current location
  python setup_crystal_tools.py --update-templates
  
  # Force reconfiguration
  python setup_crystal_tools.py --force
"""
    )
    
    parser.add_argument('--add-to-path', action='store_true',
                        help='Add Crystal Tools scripts to PATH')
    parser.add_argument('--update-templates', action='store_true',
                        help='Update SLURM script templates with current location')
    parser.add_argument('--no-shell', action='store_true',
                        help='Skip shell configuration file updates')
    parser.add_argument('--skip-deps', action='store_true',
                        help='Skip dependency checking')
    parser.add_argument('--force', action='store_true',
                        help='Force reconfiguration even if already set up')
    parser.add_argument('--shell', choices=['bash', 'zsh', 'both'],
                        help='Specify shell to configure (default: auto-detect and ask)')
    
    args = parser.parse_args()
    
    setup = MACESetup(shell=args.shell)
    setup.run_setup(args)

if __name__ == '__main__':
    main()