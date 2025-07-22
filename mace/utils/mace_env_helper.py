#!/usr/bin/env python3
"""
MACE Environment Helper
Handles environment variable transitions and compatibility
"""

import os
import sys
from pathlib import Path

def get_mace_home():
    """
    Get MACE installation directory
    
    Priority:
    1. MACE_HOME environment variable
    2. Auto-detection from script location
    """
    # Check environment variable
    mace_home = os.environ.get('MACE_HOME')
    if mace_home:
        return Path(mace_home)
    
    # Try to auto-detect
    current_file = Path(__file__).resolve()
    if current_file.parent.name == 'Job_Scripts' and current_file.parent.parent.name == 'code':
        return current_file.parent.parent.parent
    
    return None

def setup_mace_paths():
    """Add MACE directories to Python path"""
    mace_home = get_mace_home()
    if not mace_home:
        raise EnvironmentError(
            "MACE installation not found. Please set MACE_HOME"
        )
    
    # Add all necessary paths
    paths_to_add = [
        mace_home / 'code' / 'Job_Scripts',
        mace_home / 'code' / 'Crystal_d12',
        mace_home / 'code' / 'Crystal_d3',
        mace_home / 'code' / 'Check_Scripts',
        mace_home / 'code' / 'Post_Processing_Scripts',
        mace_home / 'code' / 'Plotting_Scripts'
    ]
    
    for path in paths_to_add:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    
    return mace_home

def get_mace_config():
    """Get MACE configuration from environment"""
    mace_home = get_mace_home()
    
    config = {
        'MACE_HOME': str(mace_home) if mace_home else None,
        'MACE_SCRATCH': os.environ.get('MACE_SCRATCH', 
                                       os.path.join(os.environ.get('SCRATCH', '/tmp'), 'mace')),
        'MACE_DATABASE': os.environ.get('MACE_DATABASE',
                                       os.path.expanduser('~/.mace/mace.db')),
        'MACE_LOG_LEVEL': os.environ.get('MACE_LOG_LEVEL', 'INFO'),
        'MACE_MAX_JOBS': int(os.environ.get('MACE_MAX_JOBS', '200')),
        'MACE_BANNER': os.environ.get('MACE_BANNER', 'compact')
    }
    
    return config

def export_bash_config():
    """Generate bash export commands for MACE setup"""
    mace_home = get_mace_home()
    if not mace_home:
        return ""
    
    exports = f"""
# MACE Environment Configuration
export MACE_HOME="{mace_home}"

# Optional MACE settings
export MACE_SCRATCH="${{MACE_SCRATCH:-$SCRATCH/mace}}"
export MACE_DATABASE="${{MACE_DATABASE:-$HOME/.mace/mace.db}}"
export MACE_LOG_LEVEL="${{MACE_LOG_LEVEL:-INFO}}"
export MACE_MAX_JOBS="${{MACE_MAX_JOBS:-200}}"
export MACE_BANNER="${{MACE_BANNER:-compact}}"

# Add MACE to PATH
export PATH="$MACE_HOME/mace:$PATH"
export PATH="$MACE_HOME/Crystal_d12:$PATH"
export PATH="$MACE_HOME/Crystal_d3:$PATH"
export PATH="$MACE_HOME/mace/submission:$PATH"
export PATH="$MACE_HOME/mace/utils:$PATH"
"""
    return exports

if __name__ == '__main__':
    # Test environment detection
    print("MACE Environment Status:")
    print("-" * 40)
    
    mace_home = get_mace_home()
    if mace_home:
        print(f"✓ MACE Home: {mace_home}")
    else:
        print("✗ MACE Home: Not found")
    
    print("\nEnvironment Variables:")
    for var in ['MACE_HOME', 'MACE_SCRATCH', 'MACE_DATABASE']:
        value = os.environ.get(var)
        if value:
            print(f"  {var}: {value}")
        else:
            print(f"  {var}: <not set>")
    
    print("\nConfiguration:")
    config = get_mace_config()
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    if '--export' in sys.argv:
        print("\n" + "="*40)
        print("Add to your shell configuration:")
        print("="*40)
        print(export_bash_config())