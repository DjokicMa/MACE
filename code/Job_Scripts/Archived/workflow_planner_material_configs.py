"""
Enhanced workflow planner module for creating per-material expert configurations
"""

def create_per_material_expert_configs(self, calc_type: str, step_num: int) -> Dict[str, Any]:
    """
    Create individual expert configuration files for each material/structure.
    This ensures that each material preserves its own symmetry and settings.
    
    Args:
        calc_type: Calculation type (e.g., OPT2, OPT3, SP2, FREQ)
        step_num: Step number in workflow
        
    Returns:
        Dictionary with configuration details and file mappings
    """
    print(f"\n    Expert {calc_type} Setup:")
    print(f"    Creating per-material configurations to preserve individual symmetries")
    
    # Find all D12 files that will be processed
    d12_files = []
    
    # Check multiple locations for D12 files
    search_dirs = [
        self.work_dir,
        self.work_dir / "workflow_inputs" / f"step_{step_num-1:03d}_*",
        self.work_dir / "workflow_outputs" / "*" / f"step_{step_num-1:03d}_*" / "*",
    ]
    
    for search_pattern in search_dirs:
        if isinstance(search_pattern, Path):
            if search_pattern.exists() and search_pattern.is_dir():
                d12_files.extend(search_pattern.glob("*.d12"))
        else:
            # It's a glob pattern
            for path in Path(self.work_dir).glob(str(search_pattern).replace(str(self.work_dir) + "/", "")):
                if path.is_dir():
                    d12_files.extend(path.glob("*.d12"))
    
    if not d12_files:
        print("    No D12 files found for expert configuration")
        print("    Will create a general template configuration")
        return self._create_general_expert_config(calc_type, step_num)
    
    print(f"    Found {len(d12_files)} D12 files to configure")
    
    # Ask if user wants to configure each individually or use same settings
    config_mode = input("    Configure each material individually (i) or use same settings for all (s)? [i/s]: ").strip().lower()
    
    if config_mode == 's':
        # Use first D12 as template for all
        return self._create_shared_expert_config(calc_type, step_num, d12_files)
    else:
        # Create individual configs
        return self._create_individual_expert_configs(calc_type, step_num, d12_files)


def _create_individual_expert_configs(self, calc_type: str, step_num: int, 
                                     d12_files: List[Path]) -> Dict[str, Any]:
    """Create individual expert config for each D12 file"""
    
    # Create config directory
    config_dir = self.work_dir / "workflow_configs" / f"expert_{calc_type.lower()}_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy required scripts
    self._copy_required_scripts_for_expert_mode()
    
    material_configs = {}
    
    for i, d12_file in enumerate(d12_files):
        material_name = self.extract_core_material_name(d12_file)
        print(f"\n    Configuring {material_name} ({i+1}/{len(d12_files)})")
        
        # Create temporary directory for this material
        temp_dir = self.temp_dir / f"expert_config_{calc_type.lower()}_{material_name}"
        temp_dir.mkdir(exist_ok=True)
        
        # Copy the real D12 and create sample files
        sample_out = temp_dir / "sample.out"
        sample_d12 = temp_dir / "sample.d12"
        
        # Copy real D12 with single atom modification
        self._create_sample_from_real_d12(d12_file, sample_d12)
        
        # Create minimal output file
        self._create_sample_output(sample_out)
        
        # Config file for this specific material
        config_file = config_dir / f"{material_name}_{calc_type.lower()}_expert_config.json"
        
        # Run CRYSTALOptToD12.py for this material
        print(f"      Launching CRYSTALOptToD12.py for {material_name}")
        
        cmd = [
            sys.executable, str(self.get_crystal_opt_script_path()),
            "--out-file", str(sample_out),
            "--d12-file", str(sample_d12),
            "--output-dir", str(temp_dir),
            "--save-options",
            "--options-file", str(config_file)
        ]
        
        try:
            result = subprocess.run(cmd, cwd=str(self.work_dir))
            
            if result.returncode == 0 and config_file.exists():
                # Load and enhance the config
                with open(config_file, 'r') as f:
                    saved_config = json.load(f)
                
                # Add material-specific metadata
                saved_config['material_name'] = material_name
                saved_config['source_d12'] = str(d12_file)
                saved_config['workflow_calc_type'] = calc_type
                
                # Save enhanced config
                with open(config_file, 'w') as f:
                    json.dump(saved_config, f, indent=2)
                
                material_configs[material_name] = {
                    "config_file": str(config_file),
                    "crystal_opt_config": saved_config
                }
                
                print(f"      ✅ Configuration saved for {material_name}")
            else:
                print(f"      ❌ Configuration failed for {material_name}")
                
        except Exception as e:
            print(f"      ❌ Error configuring {material_name}: {e}")
    
    return {
        "expert_mode": True,
        "per_material_configs": True,
        "config_directory": str(config_dir),
        "material_configs": material_configs,
        "step_num": step_num,
        "workflow_calc_type": calc_type
    }


def _create_sample_from_real_d12(self, source_d12: Path, sample_d12: Path):
    """Create a sample D12 from a real one, with single atom at origin"""
    try:
        with open(source_d12, 'r') as f:
            lines = f.readlines()
        
        with open(sample_d12, 'w') as f:
            # Write new title
            f.write("Sample D12 for configuration\n")
            
            # Copy structure section up to atom count
            i = 1
            while i < len(lines):
                line = lines[i].strip()
                f.write(lines[i])
                
                # After cell parameters, modify atom section
                try:
                    parts = line.split()
                    if len(parts) >= 6:
                        # Try to parse as floats - this is cell parameters
                        [float(p) for p in parts[:6]]
                        
                        # Next line should be atom count
                        if i + 1 < len(lines):
                            atom_count = int(lines[i + 1].strip())
                            f.write("1\n")  # Write 1 atom
                            
                            # Get first atom info for element
                            if i + 2 < len(lines):
                                atom_parts = lines[i + 2].strip().split()
                                if atom_parts:
                                    atom_num = atom_parts[0]
                                    # Write single atom at origin
                                    f.write(f"{atom_num} 0.0 0.0 0.0\n")
                            
                            # Skip original atom section
                            i += 2 + atom_count
                            
                            # Copy rest of file
                            while i < len(lines):
                                f.write(lines[i])
                                i += 1
                            break
                except (ValueError, IndexError):
                    pass
                
                i += 1
                
    except Exception as e:
        print(f"        Warning: Could not process D12 file: {e}")
        # Fall back to minimal template
        with open(sample_d12, 'w') as f:
            f.write("Sample D12 for configuration\n")
            f.write("CRYSTAL\n")
            f.write("0 0 0\n")
            f.write("1\n")
            f.write("5.0 5.0 5.0 90.0 90.0 90.0\n")
            f.write("1\n")
            f.write("6 0.0 0.0 0.0\n")
            f.write("END\n")