#!/usr/bin/env python3
"""
Wrapper script for backward compatibility.
Calls the new queue.monitor module.

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group
"""

import sys
from mace.queue.monitor import main

if __name__ == "__main__":
    main()