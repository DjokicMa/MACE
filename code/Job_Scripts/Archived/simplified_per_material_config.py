"""
Simplified approach for per-material expert configurations
"""
import json
from pathlib import Path
from typing import Dict, Any, List

def create_per_material_configs_simple(calc_type: str, base_config: Dict[str, Any], 
                                     d12_files: List[Path], config_dir: Path) -> Dict[str, Any]:
    """
    Create per-material configs by modifying only symmetry fields from a base config.
    
    This is much more efficient than running CRYSTALOptToD12.py for each material.
    """
    material_configs = {}
    
    for d12_file in d12_files:
        # Extract material name
        material_name = extract_material_name(d12_file)
        
        # Extract symmetry settings from this D12
        symmetry_settings = extract_symmetry_from_d12(d12_file)
        
        # Create material-specific config by updating base config
        material_config = base_config.copy()
        material_config.update({
            'spacegroup': symmetry_settings['spacegroup'],
            'origin_setting': symmetry_settings['origin_setting'],
            'dimensionality': symmetry_settings['dimensionality'],
            # Set write_only_unique based on space group
            'write_only_unique': symmetry_settings['spacegroup'] != 1  # True for non-P1
        })
        
        # Save material-specific config
        config_file = config_dir / f"{material_name}_{calc_type.lower()}_expert_config.json"
        with open(config_file, 'w') as f:
            json.dump(material_config, f, indent=2)
        
        material_configs[material_name] = {
            "config_file": str(config_file),
            "source_d12": str(d12_file)
        }
        
    return material_configs


def extract_symmetry_from_d12(d12_file: Path) -> Dict[str, Any]:
    """Extract only symmetry-related settings from a D12 file"""
    settings = {
        'spacegroup': 1,
        'origin_setting': '0 0 0',
        'dimensionality': 'CRYSTAL'
    }
    
    try:
        with open(d12_file, 'r') as f:
            lines = f.readlines()
        
        # Skip title line
        for i in range(1, len(lines)):
            line = lines[i].strip()
            
            if line in ['CRYSTAL', 'SLAB', 'POLYMER', 'MOLECULE']:
                settings['dimensionality'] = line
                
                if line == 'CRYSTAL' and i + 2 < len(lines):
                    settings['origin_setting'] = lines[i + 1].strip()
                    try:
                        settings['spacegroup'] = int(lines[i + 2].strip())
                    except ValueError:
                        pass
                break
                
    except Exception as e:
        print(f"Warning: Could not extract symmetry from {d12_file}: {e}")
        
    return settings