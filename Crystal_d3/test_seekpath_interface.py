#!/usr/bin/env python3
"""
Test Suite for SeeKPath Interface Module

This script tests the seekpath_interface.py module to verify:
1. Cell parameter to vector conversion
2. Structure parsing from CRYSTAL output
3. SeeKPath integration
4. Format conversion to MACE format
5. Discontinuity detection

Usage:
    python test_seekpath_interface.py                    # Run all unit tests
    python test_seekpath_interface.py --file output.out  # Test with specific file
    python test_seekpath_interface.py --compare          # Compare with old implementation

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group
"""

import numpy as np
import sys
import argparse
from pathlib import Path

# Import the module to test
try:
    from seekpath_interface import (
        cell_params_to_vectors,
        calculate_minimum_shrink,
        convert_seekpath_label,
        SEEKPATH_AVAILABLE,
        get_accurate_bandpath,
        parse_crystal_structure,
        print_bandpath_info
    )
    MODULE_AVAILABLE = True
except ImportError as e:
    print(f"Could not import seekpath_interface: {e}")
    MODULE_AVAILABLE = False


class TestResult:
    """Simple test result tracker."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, name):
        self.passed += 1
        print(f"  [PASS] {name}")

    def fail(self, name, msg=""):
        self.failed += 1
        self.errors.append((name, msg))
        print(f"  [FAIL] {name}: {msg}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*50}")
        print(f"Results: {self.passed}/{total} tests passed")
        if self.errors:
            print("\nFailed tests:")
            for name, msg in self.errors:
                print(f"  - {name}: {msg}")
        return self.failed == 0


# =============================================================================
# Unit Tests
# =============================================================================

def test_cell_params_to_vectors(results: TestResult):
    """Test cell parameter to vector conversion."""
    print("\n--- Testing cell_params_to_vectors ---")

    # Test 1: Simple cubic (a=b=c, all angles 90)
    cell = cell_params_to_vectors(5.0, 5.0, 5.0, 90.0, 90.0, 90.0)
    expected = np.array([
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0]
    ])
    if np.allclose(cell, expected, atol=1e-10):
        results.ok("Cubic cell")
    else:
        results.fail("Cubic cell", f"Got:\n{cell}")

    # Test 2: Tetragonal (a=b≠c, all angles 90)
    cell = cell_params_to_vectors(4.0, 4.0, 6.0, 90.0, 90.0, 90.0)
    if abs(cell[0, 0] - 4.0) < 1e-10 and abs(cell[2, 2] - 6.0) < 1e-10:
        results.ok("Tetragonal cell")
    else:
        results.fail("Tetragonal cell", f"Got:\n{cell}")

    # Test 3: Hexagonal (gamma=120)
    cell = cell_params_to_vectors(3.0, 3.0, 5.0, 90.0, 90.0, 120.0)
    # b should be at 120 degrees from a
    expected_bx = 3.0 * np.cos(np.radians(120))
    expected_by = 3.0 * np.sin(np.radians(120))
    if abs(cell[1, 0] - expected_bx) < 1e-10 and abs(cell[1, 1] - expected_by) < 1e-10:
        results.ok("Hexagonal cell")
    else:
        results.fail("Hexagonal cell", f"Got:\n{cell}")

    # Test 4: Monoclinic (beta ≠ 90)
    cell = cell_params_to_vectors(5.0, 6.0, 7.0, 90.0, 110.0, 90.0)
    # c should have non-zero x component
    if abs(cell[2, 0]) > 0.1:  # Should be c*cos(beta) ≈ -2.39
        results.ok("Monoclinic cell")
    else:
        results.fail("Monoclinic cell", f"c_x should be non-zero, got:\n{cell}")


def test_label_conversion(results: TestResult):
    """Test label conversion."""
    print("\n--- Testing label conversion ---")

    # Test GAMMA -> G
    if convert_seekpath_label('GAMMA') == 'G':
        results.ok("GAMMA -> G")
    else:
        results.fail("GAMMA -> G", f"Got: {convert_seekpath_label('GAMMA')}")

    # Test passthrough
    if convert_seekpath_label('X') == 'X':
        results.ok("X passthrough")
    else:
        results.fail("X passthrough", f"Got: {convert_seekpath_label('X')}")

    # Test subscript labels
    if convert_seekpath_label('SIGMA_0') == 'SIGMA_0':
        results.ok("SIGMA_0 passthrough")
    else:
        results.fail("SIGMA_0 passthrough", f"Got: {convert_seekpath_label('SIGMA_0')}")


def test_minimum_shrink(results: TestResult):
    """Test minimum shrink factor calculation."""
    print("\n--- Testing minimum shrink calculation ---")

    # Test 1: Simple fractions (1/2, 1/4)
    coords = {
        'G': [0.0, 0.0, 0.0],
        'X': [0.5, 0.0, 0.5],
        'L': [0.5, 0.5, 0.5]
    }
    shrink = calculate_minimum_shrink(coords)
    if shrink >= 2 and shrink % 2 == 0:
        results.ok("Simple fractions (1/2)")
    else:
        results.fail("Simple fractions", f"Got shrink={shrink}")

    # Test 2: Thirds (1/3)
    coords = {
        'G': [0.0, 0.0, 0.0],
        'K': [1/3, 1/3, 0.0]
    }
    shrink = calculate_minimum_shrink(coords)
    if shrink >= 6:  # Need at least 6 to represent 1/3
        results.ok("Thirds (1/3)")
    else:
        results.fail("Thirds", f"Got shrink={shrink}, need >=6")

    # Test 3: Complex fractions (3/8)
    coords = {
        'K': [0.375, 0.375, 0.75]  # 3/8, 3/8, 3/4
    }
    shrink = calculate_minimum_shrink(coords)
    if shrink >= 8:
        results.ok("Eighths (3/8)")
    else:
        results.fail("Eighths", f"Got shrink={shrink}, need >=8")


def test_discontinuity_detection(results: TestResult):
    """Test that discontinuities are properly detected."""
    print("\n--- Testing discontinuity detection ---")

    if not SEEKPATH_AVAILABLE:
        results.fail("Discontinuity detection", "seekpath not available")
        return

    # Create a mock seekpath result with known discontinuities
    # Path: G-X-U | K-G-L (discontinuity between U and K)
    mock_result = {
        'point_coords': {
            'GAMMA': [0.0, 0.0, 0.0],
            'X': [0.5, 0.0, 0.5],
            'U': [0.625, 0.25, 0.625],
            'K': [0.375, 0.375, 0.75],
            'L': [0.5, 0.5, 0.5]
        },
        'path': [
            ('GAMMA', 'X'),
            ('X', 'U'),
            ('K', 'GAMMA'),  # Discontinuity: U ≠ K
            ('GAMMA', 'L')
        ],
        'has_inversion_symmetry': True,
        'bravais_lattice': 'cF',
        'bravais_lattice_extended': 'cF1'
    }

    from seekpath_interface import convert_to_mace_format
    segments, labels, kpath_info = convert_to_mace_format(mock_result, shrink_factor=8)

    # Should have exactly 1 discontinuity
    n_discontinuities = labels.count('|')
    if n_discontinuities == 1:
        results.ok("Discontinuity count")
    else:
        results.fail("Discontinuity count", f"Expected 1, got {n_discontinuities}")

    # Labels should be: G-X-U-|-K-G-L
    expected_label_count = 6  # G, X, U, K, G, L (not counting |)
    actual_label_count = len([l for l in labels if l != '|'])
    if actual_label_count == expected_label_count:
        results.ok("Label count")
    else:
        results.fail("Label count", f"Expected {expected_label_count}, got {actual_label_count}")

    # Check segment count
    if len(segments) == 4:
        results.ok("Segment count")
    else:
        results.fail("Segment count", f"Expected 4, got {len(segments)}")


def test_seekpath_available(results: TestResult):
    """Test if seekpath library is available."""
    print("\n--- Testing seekpath availability ---")

    if SEEKPATH_AVAILABLE:
        results.ok("seekpath library imported")

        # Test basic seekpath call
        try:
            import seekpath
            # Simple cubic structure
            cell = np.eye(3) * 5.0
            positions = np.array([[0.0, 0.0, 0.0]])
            numbers = [14]  # Silicon

            result = seekpath.get_path((cell, positions, numbers))

            if 'path' in result and 'point_coords' in result:
                results.ok("seekpath.get_path() works")
            else:
                results.fail("seekpath.get_path()", "Missing expected keys")

        except Exception as e:
            results.fail("seekpath.get_path()", str(e))
    else:
        results.fail("seekpath library", "Not installed (pip install seekpath)")


# =============================================================================
# Integration Tests
# =============================================================================

def test_with_file(output_file: str, results: TestResult):
    """Test with an actual CRYSTAL output file."""
    print(f"\n--- Testing with file: {Path(output_file).name} ---")

    if not Path(output_file).exists():
        results.fail("File exists", f"Not found: {output_file}")
        return

    # Test structure parsing
    try:
        structure = parse_crystal_structure(output_file)
        if structure is not None:
            results.ok(f"Structure parsing ({len(structure['numbers'])} atoms)")
        else:
            results.fail("Structure parsing", "Returned None")
            return
    except Exception as e:
        results.fail("Structure parsing", str(e))
        return

    # Test full band path generation
    if not SEEKPATH_AVAILABLE:
        results.fail("Band path generation", "seekpath not available")
        return

    try:
        segments, labels, kpath_info = get_accurate_bandpath(output_file)

        results.ok(f"Band path generation ({len(segments)} segments)")

        # Verify basic properties
        if len(segments) > 0:
            results.ok("Non-empty segments")
        else:
            results.fail("Non-empty segments", "Got 0 segments")

        if len([l for l in labels if l != '|']) > 0:
            results.ok("Non-empty labels")
        else:
            results.fail("Non-empty labels", "Got 0 labels")

        print(f"\n  Bravais: {kpath_info['bravais_lattice_extended']}")
        print(f"  Inversion: {kpath_info['has_inversion']}")
        print(f"  Path: {'-'.join(labels)}")

    except Exception as e:
        results.fail("Band path generation", str(e))
        import traceback
        traceback.print_exc()


def compare_implementations(output_file: str):
    """Compare new seekpath implementation with old static dictionary."""
    print(f"\n{'='*60}")
    print("Comparing implementations")
    print('='*60)

    if not Path(output_file).exists():
        print(f"Error: File not found: {output_file}")
        return

    if not SEEKPATH_AVAILABLE:
        print("Error: seekpath not available for comparison")
        return

    # Get new implementation result
    print("\n--- New Implementation (seekpath library) ---")
    try:
        segments_new, labels_new, kpath_info_new = get_accurate_bandpath(output_file)
        print(f"  Bravais: {kpath_info_new['bravais_lattice_extended']}")
        print(f"  Segments: {len(segments_new)}")
        print(f"  Labels: {'-'.join(labels_new)}")
    except Exception as e:
        print(f"  Error: {e}")
        return

    # Get old implementation result
    print("\n--- Old Implementation (static dictionary) ---")
    try:
        # Import old implementation
        from d3_kpoints import get_seekpath_full_kpath, get_seekpath_labels

        # Need to get space group and lattice type from output
        structure = parse_crystal_structure(output_file)

        # Parse space group from file
        with open(output_file, 'r') as f:
            content = f.read()

        import re
        sg_match = re.search(r'SPACE GROUP.*?NUMBER:\s*(\d+)', content)
        if sg_match:
            space_group = int(sg_match.group(1))
        else:
            space_group = 1

        # Get lattice type
        sg_symbol_match = re.search(r'SPACE GROUP[^:]*:\s*([A-Z])', content)
        lattice_type = sg_symbol_match.group(1) if sg_symbol_match else 'P'

        segments_old, kpath_info_old = get_seekpath_full_kpath(space_group, lattice_type, output_file)
        labels_old = get_seekpath_labels(space_group, lattice_type, output_file)

        print(f"  Extended Bravais: {kpath_info_old.get('extended_bravais', 'unknown')}")
        print(f"  Segments: {len(segments_old)}")
        print(f"  Labels: {'-'.join(labels_old)}")

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Compare
    print("\n--- Comparison ---")

    if len(segments_new) == len(segments_old):
        print(f"  Segment count: MATCH ({len(segments_new)})")
    else:
        print(f"  Segment count: DIFFER (new={len(segments_new)}, old={len(segments_old)})")

    new_label_str = '-'.join(labels_new)
    old_label_str = '-'.join(labels_old)
    if new_label_str == old_label_str:
        print("  Labels: MATCH")
    else:
        print("  Labels: DIFFER")
        print(f"    New: {new_label_str}")
        print(f"    Old: {old_label_str}")


# =============================================================================
# Main
# =============================================================================

def run_all_tests():
    """Run all unit tests."""
    if not MODULE_AVAILABLE:
        print("Cannot run tests: module import failed")
        return False

    results = TestResult()

    print("\n" + "="*50)
    print("Running seekpath_interface.py test suite")
    print("="*50)

    # Unit tests
    test_cell_params_to_vectors(results)
    test_label_conversion(results)
    test_minimum_shrink(results)
    test_seekpath_available(results)
    test_discontinuity_detection(results)

    return results.summary()


def main():
    parser = argparse.ArgumentParser(description="Test seekpath_interface.py module")
    parser.add_argument('--file', '-f', type=str, help="Test with specific CRYSTAL output file")
    parser.add_argument('--compare', '-c', action='store_true',
                        help="Compare with old implementation (requires --file)")
    parser.add_argument('--info', '-i', action='store_true',
                        help="Print detailed band path info for file")

    args = parser.parse_args()

    if args.info and args.file:
        print_bandpath_info(args.file)
        return

    if args.compare:
        if not args.file:
            print("Error: --compare requires --file")
            sys.exit(1)
        compare_implementations(args.file)
        return

    # Run unit tests
    success = run_all_tests()

    # If file specified, also run integration tests
    if args.file:
        results = TestResult()
        test_with_file(args.file, results)
        results.summary()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
