#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D12 From Configuration - Unified interface for JSON-based D12 generation
============================================================
This script provides a unified interface for generating D12 files from
JSON configuration files, automatically selecting the appropriate tool
(NewCifToD12.py or CRYSTALOptToD12.py) based on input file type.

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group

Usage:
    # From CIF file
    python d12_from_config.py --config standard_dft_opt.json structure.cif
    
    # From CRYSTAL output
    python d12_from_config.py --config high_accuracy_sp.json optimized.out
    
    # Batch processing
    python d12_from_config.py --config metallic_system.json --batch *.cif
    
    # List available configurations
    python d12_from_config.py --list-configs
"""

import os
import sys
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from d12_config import (
        list_available_configs, load_d12_config, 
        get_default_d12_configs, print_config_summary
    )
except ImportError:
    print("Error: Could not import d12_config module")
    sys.exit(1)


def detect_file_type(filename: str) -> str:
    """
    Detect whether file is CIF or CRYSTAL output.
    
    Args:
        filename: Input file path
        
    Returns:
        str: 'cif', 'crystal_output', or 'unknown'
    """
    if not os.path.exists(filename):
        return 'unknown'
    
    # Check extension first
    ext = Path(filename).suffix.lower()
    if ext == '.cif':
        return 'cif'
    elif ext in ['.out', '.output', '.log']:
        # Verify it's a CRYSTAL output
        try:
            with open(filename, 'r') as f:
                content = f.read(5000)  # Read first 5KB
                # The banner names CRYSTAL; MPI start-up lines can push
                # everything else (EEEE..., ETOT) past the first 5 KB
                if 'CRYSTAL' in content:
                    return 'crystal_output'
        except:
            pass
    
    # Try to detect by content
    try:
        with open(filename, 'r') as f:
            content = f.read(1000)  # Read first 1KB
            if 'data_' in content and '_cell_length_a' in content:
                return 'cif'
            elif 'CRYSTAL' in content:
                return 'crystal_output'
    except:
        pass
    
    return 'unknown'


def process_file_with_config(
    input_file: str, 
    config_file: str,
    file_type: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Process a single file with given configuration.
    
    Args:
        input_file: Input file path
        config_file: Configuration file path
        file_type: Override file type detection
        
    Returns:
        tuple: (success, message)
    """
    # Detect file type if not specified
    if file_type is None:
        file_type = detect_file_type(input_file)
    
    if file_type == 'unknown':
        return False, f"Could not determine file type for {input_file}"
    
    # Determine which script to use
    if file_type == 'cif':
        script = 'NewCifToD12.py'
        config_arg = '--options_file'
    else:  # crystal_output
        script = 'CRYSTALOptToD12.py'
        config_arg = '--config-file'
    
    # Build command. NewCifToD12.py converts a directory, so the CIF is
    # staged alone in a temporary one and the deck is written next to it;
    # neither tool takes the input as a positional argument, and neither may
    # stop at a prompt here.
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script)
    config_file = resolve_config_path(config_file)
    input_file = os.path.abspath(input_file)
    stage = None
    if file_type == 'cif':
        stage = tempfile.mkdtemp(prefix="d12_from_config_")
        shutil.copy(input_file, stage)
        cmd = [sys.executable, script_path, '--batch', config_arg, config_file,
               '--cif_dir', stage, '--output_dir', os.path.dirname(input_file)]
    else:
        cmd = [sys.executable, script_path, config_arg, config_file,
               '--out-file', input_file, '--non-interactive']
        d12 = os.path.splitext(input_file)[0] + '.d12'
        if os.path.exists(d12):
            cmd += ['--d12-file', d12]

    # Execute. Success is a deck on disk, not only a zero exit status: a
    # converter that refuses every input must not read as "Successfully".
    out_dir = os.path.dirname(input_file)
    before = _decks_in(out_dir)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                stdin=subprocess.DEVNULL,
                                cwd=out_dir)
        written = sorted(name for name, stamp in _decks_in(out_dir).items()
                         if before.get(name) != stamp)
        if result.returncode == 0 and written:
            return True, (f"Successfully processed {input_file} -> "
                          f"{', '.join(written)}")
        detail = (result.stderr.strip() or result.stdout.strip())[-2000:]
        if result.returncode == 0:
            return False, (f"No D12 file was written for {input_file}:\n"
                           f"{detail}")
        return False, f"Error processing {input_file}:\n{detail}"
    except Exception as e:
        return False, f"Failed to run {script}: {e}"
    finally:
        if stage:
            shutil.rmtree(stage, ignore_errors=True)


def _decks_in(directory: str) -> dict:
    """{name: (mtime_ns, size)} of the .d12 files in directory."""
    decks = {}
    for path in Path(directory).glob("*.d12"):
        try:
            st = path.stat()
        except OSError:
            continue
        decks[path.name] = (st.st_mtime_ns, st.st_size)
    return decks


def resolve_config_path(config_file: str) -> str:
    """The configuration file's path, looked up where load_d12_config looks."""
    candidates = [config_file]
    if not os.path.isabs(config_file):
        here = os.path.dirname(os.path.abspath(__file__))
        candidates += [os.path.join(os.getcwd(), 'd12_configs', config_file),
                       os.path.join(here, 'example_configs', config_file),
                       os.path.join(here, 'd12_configs', config_file)]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    return config_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate D12 files from JSON configurations"
    )
    
    # Configuration options
    parser.add_argument(
        '--config', '--config-file',
        help='JSON configuration file to use'
    )
    
    parser.add_argument(
        '--list-configs',
        action='store_true',
        help='List available configuration files'
    )
    
    parser.add_argument(
        '--show-config',
        help='Display details of a configuration file'
    )
    
    # Input files
    parser.add_argument(
        'input_files',
        nargs='*',
        help='Input files (CIF or CRYSTAL output)'
    )
    
    # Batch processing
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process all matching files in current directory'
    )
    
    parser.add_argument(
        '--file-type',
        choices=['cif', 'crystal_output'],
        help='Override automatic file type detection'
    )
    
    parser.add_argument(
        '--pattern',
        default='*.cif',
        help='File pattern for batch mode (default: *.cif)'
    )
    
    args = parser.parse_args()
    
    # List configurations
    if args.list_configs:
        print("\nAvailable D12 Configurations:")
        print("=" * 60)
        configs = list_available_configs()
        
        # Group by calculation type
        by_type = {}
        for cfg in configs:
            calc_type = cfg['calc_type']
            if calc_type not in by_type:
                by_type[calc_type] = []
            by_type[calc_type].append(cfg)
        
        # Display grouped
        for calc_type in sorted(by_type.keys()):
            print(f"\n{calc_type} Calculations:")
            for cfg in by_type[calc_type]:
                print(f"  {cfg['name']:<25} - {cfg['description']}")
        
        print("\nBuilt-in Templates:")
        for name, config in get_default_d12_configs().items():
            print(f"  {name:<25} - {config['description']}")
        
        return
    
    # Show configuration details
    if args.show_config:
        try:
            # Try to load from file first
            try:
                config = load_d12_config(args.show_config)
            except:
                # Try as template name
                configs = get_default_d12_configs()
                if args.show_config in configs:
                    config = configs[args.show_config]
                else:
                    print(f"Error: Configuration '{args.show_config}' not found")
                    return
            
            print_config_summary(config)
        except Exception as e:
            print(f"Error loading configuration: {e}")
        return
    
    # Process files
    if not args.config:
        parser.error("--config is required for file processing")
    
    # Verify config exists
    try:
        config = load_d12_config(args.config)
        print(f"Using configuration: {config.get('name', args.config)}")
        print(f"Description: {config.get('description', 'No description')}")
        print()
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1)
    
    # Collect input files
    input_files = []
    
    if args.batch:
        # Batch mode - find files matching pattern
        from glob import glob
        input_files = glob(args.pattern)
        if not input_files:
            print(f"No files found matching pattern: {args.pattern}")
            sys.exit(1)
        print(f"Found {len(input_files)} files to process")
    else:
        # Use provided files
        if not args.input_files:
            parser.error("No input files specified")
        input_files = args.input_files
    
    # Process each file
    success_count = 0
    for input_file in input_files:
        print(f"\nProcessing: {input_file}")
        success, message = process_file_with_config(
            input_file, args.config, args.file_type
        )
        if success:
            success_count += 1
            print(f"  ✓ {message}")
        else:
            print(f"  ✗ {message}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Processed {success_count}/{len(input_files)} files successfully")
    
    if success_count < len(input_files):
        sys.exit(1)


if __name__ == "__main__":
    main()