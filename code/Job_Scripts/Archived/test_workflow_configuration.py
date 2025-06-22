#!/usr/bin/env python3
"""
Test workflow configuration application
"""

import json
from pathlib import Path

# Example workflow plan with configurations
example_plan = {
    "workflow_sequence": ["OPT", "SP", "BAND", "DOSS", "OPT2", "OPT3", "SP2", "BAND2", "DOSS2", "FREQ"],
    "step_configurations": {
        "OPT_1": {
            "source": "cif_conversion",
            "calculation_type": "OPT",
            "optimization_settings": {"TOLDEG": 0.00003, "MAXCYCLE": 800}
        },
        "SP_2": {
            "source": "CRYSTALOptToD12.py", 
            "calculation_type": "SP",
            "inherit_settings": True
        },
        "BAND_3": {
            "source": "create_band_d3.py",
            "requires_wavefunction": True
        },
        "DOSS_4": {
            "source": "alldos.py",
            "requires_wavefunction": True
        },
        "OPT2_5": {
            "source": "CRYSTALOptToD12.py",
            "calculation_type": "OPT",  # Will be passed as OPT to CRYSTALOptToD12.py
            "inherit_settings": True,
            "optimization_settings": {"TOLDEG": 0.00001, "MAXCYCLE": 1000}
        },
        "OPT3_6": {
            "source": "CRYSTALOptToD12.py",
            "calculation_type": "OPT",
            "inherit_settings": True,
            "optimization_settings": {"TOLDEG": 0.000003, "MAXCYCLE": 1200}
        },
        "SP2_7": {
            "source": "CRYSTALOptToD12.py",
            "calculation_type": "SP",
            "inherit_settings": True
        },
        "BAND2_8": {
            "source": "create_band_d3.py",
            "requires_wavefunction": True
        },
        "DOSS2_9": {
            "source": "alldos.py",
            "requires_wavefunction": True
        },
        "FREQ_10": {
            "source": "CRYSTALOptToD12.py",
            "calculation_type": "FREQ",
            "inherit_base_settings": True,
            "frequency_settings": {
                "mode": "FREQCALC",
                "intensities": True,
                "raman": False,
                "custom_tolerances": {
                    "TOLINTEG": "12 12 12 12 24",
                    "TOLDEE": 12
                }
            }
        }
    }
}

print("Example Workflow Configuration")
print("=" * 60)
print(f"Workflow sequence: {' → '.join(example_plan['workflow_sequence'])}")
print("\nStep configurations:")

for step_key, config in example_plan['step_configurations'].items():
    print(f"\n{step_key}:")
    print(f"  Source: {config.get('source')}")
    print(f"  Type: {config.get('calculation_type', 'N/A')}")
    
    if 'optimization_settings' in config:
        print(f"  Optimization settings: {config['optimization_settings']}")
    
    if 'frequency_settings' in config:
        print(f"  Frequency settings:")
        for k, v in config['frequency_settings'].items():
            print(f"    {k}: {v}")

print("\n" + "=" * 60)
print("Key Points:")
print("1. Each step has its own configuration")
print("2. OPT2/OPT3 are passed as 'OPT' to CRYSTALOptToD12.py") 
print("3. SP2 will be generated from OPT3 (most recent optimization)")
print("4. BAND2/DOSS2 will use SP2's wavefunction")
print("5. FREQ will be generated from OPT3 with custom settings")
print("6. File names will be clean: mat_opt, mat_sp, mat_opt2, etc.")

# Show how configurations are applied
print("\n" + "=" * 60)
print("Configuration Application Example:")
print("\nWhen FREQ is generated from OPT3:")
print("1. Workflow executor finds FREQ_10 configuration")
print("2. Creates temporary JSON with:")
temp_config = {
    "calculation_type": "FREQ",
    "keep_current_settings": True,
    "frequency_settings": {
        "mode": "FREQCALC",
        "intensities": True,
        "raman": False,
        "custom_tolerances": {
            "TOLINTEG": "12 12 12 12 24",
            "TOLDEE": 12
        }
    },
    "custom_tolerances": {
        "TOLINTEG": "12 12 12 12 24",
        "TOLDEE": 12
    }
}
print(json.dumps(temp_config, indent=2))
print("\n3. Passes this to CRYSTALOptToD12.py with OPT3's output file")
print("4. FREQ calculation is generated with these exact settings")