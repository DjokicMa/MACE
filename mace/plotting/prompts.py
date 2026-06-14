"""Interactive prompt helpers for mace plotting (MACE style).

Extracted verbatim from ``main.py`` so handler modules can reuse them without
importing ``main`` (which would create an import cycle). Behavior unchanged.

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group
"""
from typing import List


def yes_no_prompt(prompt: str, default: str = "yes") -> bool:
    """Get yes/no response from user with validation."""
    default_char = "Y/n" if default.lower() == "yes" else "y/N"

    while True:
        response = input(f"{prompt} [{default_char}]: ").strip().lower()

        if not response:
            return default.lower() == "yes"

        if response in ["y", "yes", "true", "1"]:
            return True
        elif response in ["n", "no", "false", "0"]:
            return False
        else:
            print(f"  Invalid response '{response}'. Please enter yes/no (y/n).")


def select_option(prompt: str, options: List[str], default: int = 1) -> int:
    """Display numbered options and get user selection."""
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        marker = " *" if i == default else ""
        print(f"  {i}. {option}{marker}")

    while True:
        response = input(f"\nSelect option [1-{len(options)}] (default: {default}): ").strip()

        if not response:
            return default

        try:
            choice = int(response)
            if 1 <= choice <= len(options):
                return choice
        except ValueError:
            pass

        print(f"  Invalid selection. Please enter 1-{len(options)}.")


def get_float_input(prompt: str, default: float) -> float:
    """Get a float value from user with default."""
    while True:
        response = input(f"{prompt} [{default}]: ").strip()

        if not response:
            return default

        try:
            return float(response)
        except ValueError:
            print(f"  Invalid number. Please enter a numeric value.")


def get_string_input(prompt: str, default: str = "") -> str:
    """Get a string value from user with optional default."""
    if default:
        response = input(f"{prompt} [{default}]: ").strip()
        return response if response else default
    else:
        return input(f"{prompt}: ").strip()


def configure_output_formats(interactive: bool = True, default_formats: List[str] = None) -> List[str]:
    """
    Configure output file formats.

    Args:
        interactive: Whether to prompt user
        default_formats: Default formats if not interactive

    Returns:
        List of format strings (e.g., ['svg', 'png'])
    """
    if default_formats is None:
        default_formats = ['png', 'svg']

    if not interactive:
        return default_formats

    print("\n  Output format options:")
    print("    1. PNG + SVG (default)")
    print("    2. SVG only (scalable)")
    print("    3. PNG only (600 DPI)")
    print("    4. PDF only (vector)")
    print("    5. All formats (PNG + SVG + PDF)")

    choice = get_string_input("  Select format(s) [1-5]", "1")

    format_map = {
        '1': ['png', 'svg'],
        '2': ['svg'],
        '3': ['png'],
        '4': ['pdf'],
        '5': ['png', 'svg', 'pdf'],
    }

    return format_map.get(choice, ['png', 'svg'])
