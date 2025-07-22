#!/usr/bin/env python3
"""
Quick MACE animation for workflow startup
"""

import time
import sys

def quick_mace_animation():
    """Faster version of MACE assembly animation"""
    frames = [
        "",
        "███╗   ███╗",
        "███╗   ███╗ █████╗",
        "███╗   ███╗ █████╗  ██████╗███████╗\n████╗ ████║██╔══██╗██╔════╝██╔════╝",
        """███╗   ███╗ █████╗  ██████╗███████╗
████╗ ████║██╔══██╗██╔════╝██╔════╝
██╔████╔██║███████║██║     █████╗  
██║╚██╔╝██║██╔══██║██║     ██╔══╝  
██║ ╚═╝ ██║██║  ██║╚██████╗███████╗
╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝╚══════╝"""
    ]
    
    for i, frame in enumerate(frames):
        print("\033[2J\033[H")  # Clear screen and move cursor to top
        print(frame)
        if i < len(frames) - 1:
            time.sleep(0.15)  # Faster animation
    
    # Show subtitle
    print("\nMendoza Automated CRYSTAL Engine")
    time.sleep(0.5)

if __name__ == "__main__":
    quick_mace_animation()