#!/usr/bin/env python3
"""
Analyze which scripts must be in the working directory vs installable.
This helps plan the installation-based refactoring.
"""

import os
import ast
import sys
from pathlib import Path
from typing import Set, Dict, List

class ImportAnalyzer(ast.NodeVisitor):
    """Analyze imports in Python files"""
    
    def __init__(self):
        self.imports = set()
        self.from_imports = set()
        
    def visit_Import(self, node):
        for alias in node.names:
            self.imports.add(alias.name)
            
    def visit_ImportFrom(self, node):
        if node.module:
            self.from_imports.add(node.module)
        for alias in node.names:
            self.from_imports.add(f"{node.module}.{alias.name}" if node.module else alias.name)

def analyze_script_imports(script_path: Path) -> Set[str]:
    """Extract all imports from a Python script"""
    try:
        with open(script_path, 'r') as f:
            tree = ast.parse(f.read())
        analyzer = ImportAnalyzer()
        analyzer.visit(tree)
        return analyzer.imports | analyzer.from_imports
    except Exception as e:
        print(f"  Error analyzing {script_path.name}: {e}")
        return set()

def find_local_imports(imports: Set[str], local_scripts: Set[str]) -> Set[str]:
    """Find which imports are local scripts"""
    local_imports = set()
    for imp in imports:
        # Check if import matches a local script
        base_name = imp.split('.')[0]
        if base_name in local_scripts:
            local_imports.add(base_name)
        # Also check with .py extension
        if f"{base_name}.py" in local_scripts:
            local_imports.add(base_name)
    return local_imports

def main():
    print("Analyzing script dependencies for installation planning...")
    print("=" * 60)
    
    # Get all Python scripts in Job_Scripts
    job_scripts_dir = Path(__file__).parent
    python_scripts = list(job_scripts_dir.glob("*.py"))
    script_names = {s.stem for s in python_scripts}
    
    # Analyze each script
    dependencies = {}
    reverse_deps = {}  # What depends on each script
    
    for script in python_scripts:
        if script.name == __file__:
            continue
            
        imports = analyze_script_imports(script)
        local_deps = find_local_imports(imports, script_names)
        dependencies[script.stem] = local_deps
        
        # Build reverse dependencies
        for dep in local_deps:
            if dep not in reverse_deps:
                reverse_deps[dep] = set()
            reverse_deps[dep].add(script.stem)
    
    # Analyze SLURM scripts for references
    print("\n1. SLURM Script Dependencies:")
    print("-" * 50)
    slurm_scripts = ["submitcrystal23.sh", "submit_prop.sh"]
    slurm_deps = set()
    
    for slurm_script in slurm_scripts:
        slurm_path = job_scripts_dir / slurm_script
        if slurm_path.exists():
            with open(slurm_path, 'r') as f:
                content = f.read()
                # Look for Python script references
                for script in script_names:
                    if f"{script}.py" in content:
                        slurm_deps.add(script)
                        print(f"  {slurm_script} references {script}.py")
    
    # Identify core dependencies
    print("\n2. Core Dependencies (imported by many scripts):")
    print("-" * 50)
    core_deps = []
    for script, importers in reverse_deps.items():
        if len(importers) >= 3:  # Imported by 3 or more scripts
            core_deps.append((script, len(importers)))
            print(f"  {script}: imported by {len(importers)} scripts")
            if len(importers) <= 5:  # Show details for moderately used scripts
                print(f"    Importers: {', '.join(sorted(importers))}")
    
    # Scripts that must be in working directory
    print("\n3. Scripts Required in Working Directory:")
    print("-" * 50)
    required_local = set()
    
    # Add SLURM dependencies
    required_local.update(slurm_deps)
    print(f"  From SLURM callbacks: {', '.join(sorted(slurm_deps))}")
    
    # Add scripts that are imported by required scripts
    for script in list(required_local):
        if script in dependencies:
            deps = dependencies[script]
            required_local.update(deps)
            if deps:
                print(f"  {script} requires: {', '.join(sorted(deps))}")
    
    # Scripts that can be installed globally
    print("\n4. Scripts Safe to Install Globally:")
    print("-" * 50)
    installable = script_names - required_local - {Path(__file__).stem}
    main_entry_points = [
        "run_workflow",
        "monitor_workflow",
        "show_properties",
        "check_workflows",
        "database_status_report",
    ]
    
    print("  Entry points (can be made into commands):")
    for script in sorted(installable):
        if script in main_entry_points:
            print(f"    - {script} → crystal-{script.replace('_', '-')}")
    
    print("\n  Other installable scripts:")
    for script in sorted(installable):
        if script not in main_entry_points and not script.startswith("test_"):
            print(f"    - {script}")
    
    # Summary
    print("\n5. Summary:")
    print("-" * 50)
    print(f"  Total scripts: {len(script_names)}")
    print(f"  Required in working directory: {len(required_local)}")
    print(f"  Can be installed globally: {len(installable)}")
    print(f"  Percentage installable: {len(installable)/len(script_names)*100:.1f}%")
    
    # Minimal working directory contents
    print("\n6. Minimal Working Directory Contents:")
    print("-" * 50)
    print("  Essential scripts that must be copied:")
    for script in sorted(required_local):
        print(f"    - {script}.py")
    
    print("\n  Additional files needed:")
    print("    - recovery_config.yaml")
    print("    - materials.db (created on first run)")
    print("    - .queue_locks/ (created as needed)")

if __name__ == "__main__":
    main()