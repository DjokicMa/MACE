#!/usr/bin/env python3
"""
Alternative run_workflow.py with animated startup
"""

import os
import sys
from pathlib import Path

def run_with_animation():
    """Run workflow with MACE animation"""
    try:
        from mace.utils.animation import mace_startup_animation
        # Show quick animation
        mace_startup_animation('quick')
    except:
        pass
    
    # Now run the normal workflow
    from run_mace import main
    main()

if __name__ == '__main__':
    run_with_animation()