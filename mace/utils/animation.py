#!/usr/bin/env python3
"""
MACE Animated Banner
Fun animations for MACE startup
"""

import time
import sys
import random
from typing import List

def print_with_delay(text: str, delay: float = 0.015):
    """Print text character by character with delay"""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def animate_mace_assembly(version="1.0.0"):
    """Animate MACE logo being assembled"""
    frames = [
        """
      █                           
        """,
        """
      ███╗                        
        """,
        """
      ███╗   ███╗                 
      ████╗ ████║                 
        """,
        """
      ███╗   ███╗ █████╗          
      ████╗ ████║██╔══██╗         
      ██╔████╔██║███████║         
        """,
        """
      ███╗   ███╗ █████╗  ██████╗ 
      ████╗ ████║██╔══██╗██╔════╝ 
      ██╔████╔██║███████║██║      
      ██║╚██╔╝██║██╔══██║██║      
        """,
        """
      ███╗   ███╗ █████╗  ██████╗███████╗
      ████╗ ████║██╔══██╗██╔════╝██╔════╝
      ██╔████╔██║███████║██║     █████╗  
      ██║╚██╔╝██║██╔══██║██║     ██╔══╝  
      ██║ ╚═╝ ██║██║  ██║╚██████╗███████╗
      ╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝╚══════╝
        """
    ]
    
    for frame in frames:
        print("\033[2J\033[H")  # Clear screen and move cursor to top
        print(frame)
        time.sleep(0.1)
    
    # Center the subtitle exactly under MACE (banner is 36 chars wide, starts at position 6 now)
    print("\n       Mendoza Automated CRYSTAL Engine")
    # Add version centered below
    version_text = f"Version {version}"
    padding = " " * ((44 - len(version_text)) // 2)  # Center based on banner width
    print(f"{padding}{version_text}")
    time.sleep(0.3)

def animate_crystal_structure():
    """Show animated crystal lattice"""
    crystal_frames = [
        """
           ◆
        """,
        """
         ◆─◆
        """,
        """
         ◆─◆
         │ │
         ◆─◆
        """,
        """
         ◆─◆─◆
         │╲│╱│
         ◆─◆─◆
         │╱│╲│
         ◆─◆─◆
        """
    ]
    
    print("\n    Building crystal lattice...")
    for frame in crystal_frames:
        print("\033[5A\033[J")  # Move up 5 lines and clear below
        print(frame)
        time.sleep(0.15)

def loading_bar(duration: float = 1.0, label: str = "Initializing MACE"):
    """Show a loading bar"""
    width = 40
    print(f"\n   {label}:")
    for i in range(width + 1):
        progress = i / width
        bar = "█" * i + "░" * (width - i)
        percentage = progress * 100
        sys.stdout.write(f"\r   [{bar}] {percentage:.0f}%")
        sys.stdout.flush()
        time.sleep(duration / width)
    print()  # Single newline instead of double

def sparkle_effect():
    """Random sparkle effect"""
    sparkles = ["✨", "⚡", "💎", "🔬", "⚛️", "🧪"]
    positions = [(random.randint(0, 70), random.randint(0, 5)) for _ in range(10)]
    
    for _ in range(20):
        print("\033[2J\033[H")  # Clear screen
        for x, y in positions:
            print(f"\033[{y};{x}H{random.choice(sparkles)}", end="")
        time.sleep(0.05)
        # Update positions
        positions = [(random.randint(0, 70), random.randint(0, 5)) for _ in range(10)]

def team_credits_animation():
    """Special team credits animation"""
    print("\033[2J\033[H")  # Clear screen
    
    contributors = [
        ("Marcus Djokic", "Primary Developer (PhD Student)", "🚀"),
        ("Daniel Maldonado Lopez", "Contributor (PhD Student)", "💻"),
        ("Brandon Lewis", "Contributor (Undergraduate)", "🔧"),
        ("Dr. William Comaskey", "Contributor (Postdoc)", "🔬"),
        ("Prof. J.L. Mendoza-Cortes", "Principal Investigator", "🎓")
    ]
    
    print_with_delay("═" * 60, 0.002)
    print_with_delay("        MACE Development Team", 0.01)
    print_with_delay("        Michigan State University", 0.01)
    print_with_delay("═" * 60, 0.002)
    print()
    
    for name, role, emoji in contributors:
        time.sleep(0.05)
        print(f"  {emoji}  ", end="")
        print_with_delay(f"{name} - {role}", 0.008)
    
    print()
    print_with_delay("═" * 60, 0.002)
    print("\n✨ Thank you for using MACE! ✨")

def mace_startup_animation(style: str = "full"):
    """Run MACE startup animation"""
    if style == "full":
        animate_mace_assembly()
        loading_bar(1.5, "Loading CRYSTAL Engine")
        print("\n✓ Ready for materials science!")
        print("\nDeveloped by the Mendoza Group at MSU")
        print("Contributors: M. Djokic, D. Maldonado Lopez, B. Lewis, W. Comaskey")
    elif style == "quick":
        from mace_banner import print_banner
        print_banner('compact')
        loading_bar(0.5, "MACE")
    elif style == "crystal":
        animate_crystal_structure()
        print("\n✨ Crystal structure ready for calculations!")
    elif style == "sparkle":
        sparkle_effect()
        from mace_banner import print_banner
        print_banner('alt')
    elif style == "team":
        # Special team credits animation
        team_credits_animation()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MACE Animation Demo")
    parser.add_argument("--style", choices=["full", "quick", "crystal", "sparkle", "team"], 
                        default="full", help="Animation style")
    args = parser.parse_args()
    
    mace_startup_animation(args.style)