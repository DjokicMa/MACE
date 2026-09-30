# Copyright (c) 2025 Computational Materials Science Team (lcao2wannier, author William Comaskey).
# MIT License, see LICENSE in this directory.
# Vendored into MACE from lcao2wannier 1.0.0; local changes are listed in VENDORED.md.
"""Run the canonical command with ``python -m lcao2wannier``."""

from .cli import cli_main


if __name__ == "__main__":
    cli_main()
