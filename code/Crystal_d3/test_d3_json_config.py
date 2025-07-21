#!/usr/bin/env python3
"""
Test script to demonstrate D3 JSON configuration functionality
"""

from d3_config import (save_d3_config, load_d3_config, validate_d3_config,
                      print_d3_config_summary, get_default_d3_config,
                      list_available_d3_configs)

# Example 1: Create and save a BAND configuration
print("=== Example 1: Creating BAND configuration ===")
band_config = {
    "calculation_type": "BAND",
    "n_points": 200,
    "bands": "auto",
    "path": ["G", "X", "M", "G"],
    "labels": "auto",
    "shrink": [12, 12]
}

# Save configuration
save_d3_config(band_config, "example_band_config.json")

# Example 2: Create and save a DOSS configuration
print("\n=== Example 2: Creating DOSS configuration ===")
doss_config = {
    "calculation_type": "DOSS",
    "projection_type": 3,  # Element and orbital projections
    "energy_range": "window",
    "energy_window": (-0.3677, 0.7354),  # -10 to 20 eV in Ha
    "n_points": 2000,
    "print_integrated": True,
    "output_format": 0,
    "projections": []
}

# Save configuration
save_d3_config(doss_config, "example_doss_config.json")

# Example 3: Load and validate configurations
print("\n=== Example 3: Loading and validating configurations ===")

# Load BAND config
loaded_band = load_d3_config("example_band_config.json")
if loaded_band:
    print_d3_config_summary(loaded_band)
    is_valid, errors = validate_d3_config(loaded_band)
    print(f"Validation: {'PASSED' if is_valid else 'FAILED'}")
    if errors:
        for error in errors:
            print(f"  - {error}")

# Example 4: List available configurations
print("\n=== Example 4: Available D3 configurations ===")
configs = list_available_d3_configs()
if configs:
    print("Found configurations:")
    for config in configs:
        print(f"  - {config}")

# Example 5: Get default configurations
print("\n=== Example 5: Default configurations ===")
for calc_type in ["BAND", "DOSS", "CHARGE", "POTENTIAL"]:
    default = get_default_d3_config(calc_type)
    print(f"\n{calc_type} default:")
    print(f"  {default}")

print("\n=== Testing complete ===")
print("\nUsage examples:")
print("1. Generate D3 with saved config:")
print("   python CRYSTALOptToD3.py --input file.out --config-file example_band_config.json")
print("\n2. Save configuration after interactive setup:")
print("   python CRYSTALOptToD3.py --input file.out --calc-type DOSS --save-config")
print("\n3. Use saved config for batch processing:")
print("   python CRYSTALOptToD3.py --batch --config-file example_doss_config.json")
print("\n4. List available configs:")
print("   python CRYSTALOptToD3.py --list-configs")