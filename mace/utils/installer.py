#!/usr/bin/env python3
"""
Proof of concept for crystal-tools installation mechanism.
This demonstrates how scripts would be deployed in an installation-based system.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import List, Dict

# Scripts that must be copied to working directory
REQUIRED_LOCAL_SCRIPTS = [
    "crystal_property_extractor.py",
    "crystal_queue_manager.py", 
    "enhanced_queue_manager.py",
    "error_recovery.py",
    "formula_extractor.py",
    "input_settings_extractor.py",
    "material_database.py",
    "populate_completed_jobs.py",
    "queue_lock_manager.py",
    "workflow_engine.py",
]

# Additional files needed
REQUIRED_CONFIG_FILES = [
    "recovery_config.yaml",
]

# Command-line entry points
COMMAND_MAPPINGS = {
    "crystal-workflow": "run_workflow.py",
    "crystal-monitor": "monitor_workflow.py",
    "crystal-analyze": "show_properties.py",
    "crystal-check": "check_workflows.py",
    "crystal-db-status": "database_status_report.py",
    "crystal-queue": "enhanced_queue_manager.py",  # When called as command
}

class CrystalToolsInstaller:
    """Manages installation and deployment of crystal-tools"""
    
    def __init__(self, install_dir: Path = None):
        """Initialize installer with installation directory"""
        if install_dir:
            self.install_dir = install_dir
        else:
            # Try to get from environment
            self.install_dir = Path(os.environ.get('MACE_HOME', 
                                                  Path(__file__).parent))
        
        self.bin_dir = self.install_dir / "bin"
        self.lib_dir = self.install_dir / "lib" / "mace"
        self.share_dir = self.install_dir / "share" / "mace"
        
    def check_installation(self) -> bool:
        """Check if crystal-tools is properly installed"""
        print(f"Checking installation at: {self.install_dir}")
        
        issues = []
        
        # Check directories
        for dir_name, dir_path in [
            ("bin", self.bin_dir),
            ("lib", self.lib_dir),
            ("share", self.share_dir),
        ]:
            if not dir_path.exists():
                issues.append(f"Missing {dir_name} directory: {dir_path}")
        
        # Check command scripts
        if self.bin_dir.exists():
            for cmd in COMMAND_MAPPINGS:
                cmd_path = self.bin_dir / cmd
                if not cmd_path.exists():
                    issues.append(f"Missing command: {cmd}")
        
        # Check required scripts
        if self.lib_dir.exists():
            for script in REQUIRED_LOCAL_SCRIPTS:
                script_path = self.lib_dir / "required" / script
                if not script_path.exists():
                    issues.append(f"Missing required script: {script}")
        
        if issues:
            print("\n❌ Installation issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("✅ Installation appears complete")
            return True
    
    def deploy_to_directory(self, target_dir: Path, minimal: bool = True) -> None:
        """Deploy required scripts to a working directory"""
        target_dir = Path(target_dir).resolve()
        print(f"\nDeploying crystal-tools to: {target_dir}")
        
        if not target_dir.exists():
            target_dir.mkdir(parents=True)
            print(f"Created directory: {target_dir}")
        
        deployed = []
        failed = []
        
        # Deploy required scripts
        print("\nDeploying required scripts:")
        for script in REQUIRED_LOCAL_SCRIPTS:
            src = self.lib_dir / "required" / script
            dst = target_dir / script
            
            # For this proof of concept, use current directory as source
            if not src.exists():
                src = Path(__file__).parent / script
            
            if src.exists():
                try:
                    shutil.copy2(src, dst)
                    deployed.append(script)
                    print(f"  ✓ {script}")
                except Exception as e:
                    failed.append((script, str(e)))
                    print(f"  ✗ {script}: {e}")
            else:
                failed.append((script, "Source not found"))
                print(f"  ✗ {script}: Source not found")
        
        # Deploy config files
        print("\nDeploying configuration files:")
        for config in REQUIRED_CONFIG_FILES:
            src = self.share_dir / "configs" / config
            dst = target_dir / config
            
            # For proof of concept, use current directory
            if not src.exists():
                src = Path(__file__).parent / config
            
            if src.exists():
                try:
                    shutil.copy2(src, dst)
                    deployed.append(config)
                    print(f"  ✓ {config}")
                except Exception as e:
                    failed.append((config, str(e)))
                    print(f"  ✗ {config}: {e}")
        
        # Create necessary directories
        print("\nCreating working directories:")
        for dir_name in ["workflow_outputs", "workflow_configs", ".queue_locks"]:
            dir_path = target_dir / dir_name
            dir_path.mkdir(exist_ok=True)
            print(f"  ✓ {dir_name}/")
        
        # Create convenience symlinks (optional)
        if not minimal:
            print("\nCreating convenience symlinks:")
            for cmd, script in COMMAND_MAPPINGS.items():
                if (self.bin_dir / cmd).exists():
                    link = target_dir / script
                    if not link.exists():
                        link.symlink_to(self.bin_dir / cmd)
                        print(f"  ✓ {script} → {cmd}")
        
        # Summary
        print(f"\nDeployment summary:")
        print(f"  Successfully deployed: {len(deployed)} files")
        if failed:
            print(f"  Failed: {len(failed)} files")
            for name, reason in failed:
                print(f"    - {name}: {reason}")
        
        # Create initialization marker
        marker = target_dir / ".mace_workspace"
        with open(marker, 'w') as f:
            f.write(f"MACE Workspace\n")
            f.write(f"Deployed from: {self.install_dir}\n")
            f.write(f"Deployed on: {Path.cwd()}\n")
        
        print(f"\n✅ Workspace ready at: {target_dir}")
    
    def create_wrapper_scripts(self) -> None:
        """Create wrapper scripts in bin/ directory"""
        print(f"\nCreating wrapper scripts in {self.bin_dir}")
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        
        wrapper_template = '''#!/usr/bin/env python3
"""
Crystal-tools wrapper script for {command}
Auto-generated - do not edit directly
"""
import sys
import os
from pathlib import Path

# Add mace to Python path
mace_home = Path(os.environ.get('MACE_HOME', '{install_dir}'))
sys.path.insert(0, str(mace_home / 'lib'))

# Import and run the target script
script_path = mace_home / 'lib' / 'mace' / '{script}'
if script_path.exists():
    # For proof of concept, just import from current directory
    import {module}
    sys.exit({module}.main() if hasattr({module}, 'main') else 0)
else:
    # Fallback to current directory
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import {module}
    sys.exit({module}.main() if hasattr({module}, 'main') else 0)
'''
        
        for cmd, script in COMMAND_MAPPINGS.items():
            wrapper_path = self.bin_dir / cmd
            module = script.replace('.py', '')
            
            content = wrapper_template.format(
                command=cmd,
                install_dir=self.install_dir,
                script=script,
                module=module
            )
            
            with open(wrapper_path, 'w') as f:
                f.write(content)
            
            # Make executable
            wrapper_path.chmod(0o755)
            print(f"  ✓ Created {cmd}")
    
    def show_usage(self) -> None:
        """Show how to use the installed tools"""
        print("\n" + "="*60)
        print("Crystal-Tools Installation Complete!")
        print("="*60)
        
        print("\n1. Add to your shell configuration (.bashrc or .zshrc):")
        print(f"   export MACE_HOME={self.install_dir}")
        print(f"   export PATH=$MACE_HOME/bin:$PATH")
        
        print("\n2. Initialize a new workspace:")
        print("   cd /path/to/your/project")
        print("   crystal-tools-deploy")
        
        print("\n3. Use the tools:")
        print("   crystal-workflow --interactive")
        print("   crystal-monitor")
        print("   crystal-analyze output.out")
        print("   crystal-check")
        print("   crystal-db-status")
        
        print("\n4. For existing projects with all scripts:")
        print("   Just use the global commands - they'll find local scripts")

def main():
    """Main entry point for installer"""
    parser = argparse.ArgumentParser(
        description="Crystal-Tools Installation Manager"
    )
    parser.add_argument('action', choices=['check', 'deploy', 'install', 'usage'],
                       help='Action to perform')
    parser.add_argument('--target', '-t', type=Path,
                       help='Target directory for deployment')
    parser.add_argument('--minimal', '-m', action='store_true',
                       help='Minimal deployment (no symlinks)')
    parser.add_argument('--install-dir', '-i', type=Path,
                       help='Installation directory')
    
    args = parser.parse_args()
    
    installer = CrystalToolsInstaller(args.install_dir)
    
    if args.action == 'check':
        installer.check_installation()
    elif args.action == 'deploy':
        if not args.target:
            args.target = Path.cwd()
        installer.deploy_to_directory(args.target, args.minimal)
    elif args.action == 'install':
        installer.create_wrapper_scripts()
        installer.show_usage()
    elif args.action == 'usage':
        installer.show_usage()

if __name__ == "__main__":
    main()