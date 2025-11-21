#!/usr/bin/env python3
"""
Diagnose CUBE file subtraction issues - check grid alignment and interpolation
"""

import numpy as np
import sys

def read_cube_header_and_data(filename):
    """Read CUBE file and return header info + data"""
    with open(filename, 'r') as f:
        # Skip title lines
        title1 = f.readline()
        title2 = f.readline()

        # Read atom count and origin
        line = f.readline().split()
        natoms = int(line[0])
        origin = np.array([float(x) for x in line[1:4]])

        # Read grid dimensions and voxel vectors
        nvoxels = []
        voxel_vectors = []
        for i in range(3):
            line = f.readline().split()
            nvoxels.append(int(line[0]))
            voxel_vectors.append([float(x) for x in line[1:4]])
        nvoxels = np.array(nvoxels)
        voxel_vectors = np.array(voxel_vectors)

        # Read atoms
        atoms = []
        for i in range(abs(natoms)):
            line = f.readline().split()
            atoms.append([int(line[0]), float(line[1]),
                         float(line[2]), float(line[3]), float(line[4])])

        # Read data
        data = []
        for line in f:
            data.extend([float(x) for x in line.split()])

        data = np.array(data).reshape(nvoxels[0], nvoxels[1], nvoxels[2])

    return {
        'natoms': natoms,
        'origin': origin,
        'nvoxels': nvoxels,
        'voxel_vectors': voxel_vectors,
        'atoms': atoms,
        'data': data,
        'title1': title1,
        'title2': title2
    }

def diagnose_grid_alignment(cube1, cube2, name1="File 1", name2="File 2"):
    """Diagnose alignment issues between two CUBE files"""
    print("="*70)
    print(f"GRID ALIGNMENT DIAGNOSIS: {name1} vs {name2}")
    print("="*70)

    # Check origins
    print(f"\n1. ORIGINS (Bohr):")
    print(f"   {name1}: {cube1['origin']}")
    print(f"   {name2}: {cube2['origin']}")
    origin_diff = cube1['origin'] - cube2['origin']
    print(f"   Difference: {origin_diff}")
    if np.allclose(origin_diff, 0, atol=1e-6):
        print(f"   ✓ Origins match!")
    else:
        print(f"   ✗ WARNING: Origins differ by {np.linalg.norm(origin_diff):.6e} Bohr")

    # Check voxel vectors
    print(f"\n2. VOXEL VECTORS (Bohr):")
    for i, axis in enumerate(['X', 'Y', 'Z']):
        print(f"   {axis}-axis:")
        print(f"     {name1}: {cube1['voxel_vectors'][i]}")
        print(f"     {name2}: {cube2['voxel_vectors'][i]}")
        vox_diff = cube1['voxel_vectors'][i] - cube2['voxel_vectors'][i]
        if np.allclose(vox_diff, 0, atol=1e-6):
            print(f"     ✓ Vectors match!")
        else:
            print(f"     ✗ WARNING: Vectors differ by {vox_diff}")

    # Check grid dimensions
    print(f"\n3. GRID DIMENSIONS:")
    print(f"   {name1}: {cube1['nvoxels']} = {np.prod(cube1['nvoxels']):,} voxels")
    print(f"   {name2}: {cube2['nvoxels']} = {np.prod(cube2['nvoxels']):,} voxels")

    for i, axis in enumerate(['X', 'Y', 'Z']):
        if cube1['nvoxels'][i] != cube2['nvoxels'][i]:
            print(f"   ✗ WARNING: {axis}-axis dimensions differ!")
            print(f"      {name1}: {cube1['nvoxels'][i]} points")
            print(f"      {name2}: {cube2['nvoxels'][i]} points")

            # Calculate real-space extent
            extent1 = cube1['nvoxels'][i] * np.linalg.norm(cube1['voxel_vectors'][i])
            extent2 = cube2['nvoxels'][i] * np.linalg.norm(cube2['voxel_vectors'][i])
            print(f"      {name1} extent: {extent1:.3f} Bohr")
            print(f"      {name2} extent: {extent2:.3f} Bohr")
            print(f"      Extent ratio: {extent1/extent2:.3f}")
        else:
            print(f"   ✓ {axis}-axis dimensions match!")

    # Check data ranges
    print(f"\n4. DATA RANGES:")
    print(f"   {name1}: [{cube1['data'].min():.6e}, {cube1['data'].max():.6e}]")
    print(f"   {name2}: [{cube2['data'].min():.6e}, {cube2['data'].max():.6e}]")

    # Check if grids can be compared directly
    print(f"\n5. DIRECT COMPARISON POSSIBLE?")
    can_subtract = (
        np.allclose(cube1['origin'], cube2['origin'], atol=1e-6) and
        np.allclose(cube1['voxel_vectors'], cube2['voxel_vectors'], atol=1e-6) and
        np.array_equal(cube1['nvoxels'], cube2['nvoxels'])
    )

    if can_subtract:
        print(f"   ✓ YES - Grids are identical, direct subtraction is valid")
    else:
        print(f"   ✗ NO - Grids differ, interpolation required!")
        print(f"\n   INTERPOLATION REQUIREMENTS:")
        if not np.allclose(cube1['origin'], cube2['origin'], atol=1e-6):
            print(f"     • Origin alignment needed")
        if not np.allclose(cube1['voxel_vectors'], cube2['voxel_vectors'], atol=1e-6):
            print(f"     • Voxel vector transformation needed")
        if not np.array_equal(cube1['nvoxels'], cube2['nvoxels']):
            print(f"     • Grid interpolation needed")

    # Check atom positions (to see if molecule is in same location)
    print(f"\n6. ATOM POSITIONS:")
    print(f"   {name1}: {abs(cube1['natoms'])} atoms")
    print(f"   {name2}: {abs(cube2['natoms'])} atoms")

    if abs(cube2['natoms']) > 0 and abs(cube1['natoms']) > 0:
        # Show first few atoms from each
        print(f"\n   First 3 atoms from {name2} (molecule):")
        for i, atom in enumerate(cube2['atoms'][:3]):
            z_num, charge, x, y, z = atom
            print(f"     Atom {i+1}: Z={int(z_num)} pos=({x:.4f}, {y:.4f}, {z:.4f}) Bohr")

        print(f"\n   First 3 atoms from {name1} (hybrid):")
        for i, atom in enumerate(cube1['atoms'][:3]):
            z_num, charge, x, y, z = atom
            print(f"     Atom {i+1}: Z={int(z_num)} pos=({x:.4f}, {y:.4f}, {z:.4f}) Bohr")

    return can_subtract


def check_z_axis_mapping(cube_hybrid, cube_molecule):
    """Check how the molecule's Z-grid maps onto the hybrid's Z-grid"""
    print("\n" + "="*70)
    print("Z-AXIS MAPPING ANALYSIS")
    print("="*70)

    # Get Z-axis info
    nz_hybrid = cube_hybrid['nvoxels'][2]
    nz_mol = cube_molecule['nvoxels'][2]

    vz_hybrid = np.linalg.norm(cube_hybrid['voxel_vectors'][2])
    vz_mol = np.linalg.norm(cube_molecule['voxel_vectors'][2])

    # Total Z extent
    z_extent_hybrid = nz_hybrid * vz_hybrid
    z_extent_mol = nz_mol * vz_mol

    print(f"\nZ-axis grid points:")
    print(f"  Hybrid:   {nz_hybrid} points × {vz_hybrid:.6f} Bohr/voxel = {z_extent_hybrid:.3f} Bohr total")
    print(f"  Molecule: {nz_mol} points × {vz_mol:.6f} Bohr/voxel = {z_extent_mol:.3f} Bohr total")

    # Check if voxel sizes match
    if np.isclose(vz_hybrid, vz_mol, rtol=1e-4):
        print(f"\n✓ Voxel sizes match ({vz_hybrid:.6f} Bohr)")
        print(f"\nGrid point correspondence:")
        print(f"  Molecule's {nz_mol} points should map to hybrid's points 0 to {nz_mol-1}")
        print(f"  Hybrid has {nz_hybrid - nz_mol} additional points beyond molecule's extent")

        # Check what fraction of hybrid's Z-axis is covered by molecule
        coverage = nz_mol / nz_hybrid * 100
        print(f"\nMolecule covers {coverage:.1f}% of hybrid's Z-axis")

        return True, nz_mol
    else:
        print(f"\n✗ Voxel sizes differ!")
        print(f"  Ratio: {vz_hybrid/vz_mol:.6f}")
        print(f"  Interpolation required!")
        return False, None


def analyze_subtraction_result(cube_hybrid, cube_molecule, cube_diff, name_diff="Difference"):
    """Analyze the subtraction result"""
    print("\n" + "="*70)
    print(f"SUBTRACTION RESULT ANALYSIS")
    print("="*70)

    print(f"\nData ranges:")
    print(f"  Hybrid:     [{cube_hybrid['data'].min():.6e}, {cube_hybrid['data'].max():.6e}]")
    print(f"  Molecule:   [{cube_molecule['data'].min():.6e}, {cube_molecule['data'].max():.6e}]")
    print(f"  Difference: [{cube_diff['data'].min():.6e}, {cube_diff['data'].max():.6e}]")

    # Check if difference makes sense
    expected_max = cube_hybrid['data'].max()
    expected_min = cube_hybrid['data'].min() - cube_molecule['data'].max()

    print(f"\nExpected difference range: [{expected_min:.6e}, {expected_max:.6e}]")

    # Calculate statistics
    diff_data = cube_diff['data']
    print(f"\nDifference statistics:")
    print(f"  Mean: {diff_data.mean():.6e}")
    print(f"  Std:  {diff_data.std():.6e}")
    print(f"  Median: {np.median(diff_data):.6e}")

    # Check how much actually changed
    # If subtraction worked, we should see significant differences
    similarity = np.sum(np.abs(cube_diff['data'] - cube_hybrid['data'])) / np.sum(np.abs(cube_hybrid['data']))
    print(f"\nRelative change: {similarity*100:.2f}%")

    if similarity < 0.01:
        print(f"  ✗ WARNING: Less than 1% change - subtraction may have failed!")
    elif similarity < 0.1:
        print(f"  ⚠ Only {similarity*100:.1f}% change - check if this is expected")
    else:
        print(f"  ✓ Significant change detected")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python diagnose_cube_subtraction.py hybrid.CUBE molecule.CUBE [difference.CUBE]")
        sys.exit(1)

    hybrid_file = sys.argv[1]
    molecule_file = sys.argv[2]
    diff_file = sys.argv[3] if len(sys.argv) > 3 else None

    print("Reading CUBE files...")
    cube_hybrid = read_cube_header_and_data(hybrid_file)
    cube_molecule = read_cube_header_and_data(molecule_file)

    # Diagnose alignment
    can_subtract = diagnose_grid_alignment(cube_hybrid, cube_molecule,
                                          name1="Hybrid", name2="Molecule")

    # Check Z-axis mapping
    z_match, nz_overlap = check_z_axis_mapping(cube_hybrid, cube_molecule)

    # If difference file provided, analyze it
    if diff_file:
        cube_diff = read_cube_header_and_data(diff_file)
        analyze_subtraction_result(cube_hybrid, cube_molecule, cube_diff)

    print("\n" + "="*70)
    print("DIAGNOSIS COMPLETE")
    print("="*70)
