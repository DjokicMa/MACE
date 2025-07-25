#!/usr/bin/env python3
"""
Wrapper script for backward compatibility with SLURM scripts.
Calls the new queue.manager module.
"""

import sys
from mace.queue.manager import main

if __name__ == "__main__":
    main()