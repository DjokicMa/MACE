#!/usr/bin/env python3
"""
Wrapper script for backward compatibility.
Calls the new run_mace module which handles workflow planning.
"""

import sys
from run_mace import main

if __name__ == "__main__":
    main()