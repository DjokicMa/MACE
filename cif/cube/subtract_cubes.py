#!/usr/bin/env python3
"""
CUBE file subtraction with proper grid alignment and interpolation

Handles cases where:
- Origins differ
- Voxel sizes differ slightly
- Grid dimensions differ
- Grids cover different spatial regions

Usage: python subtract_cubes.py large.CUBE small.CUBE output.CUBE
"""

import numpy as np
import sys


def read_cube_file(filename):
    """Read CUBE file and return all data"""
    with open(filename, 'r') as f:
        # Read title lines
        title1 = f.readline().strip()
        title2 = f.readline().strip()

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
        'title1': title1,
        'title2': title2,
        'natoms': natoms,
        'origin': origin,
        'nvoxels': nvoxels,
        'voxel_vectors': voxel_vectors,
        'atoms': atoms,
        'data': data
    }


def write_cube_file(filename, cube_data):
    """Write CUBE file"""
    with open(filename, 'w') as f:
        # Write title lines
        f.write(f"{cube_data['title1']}\n")
        f.write(f"{cube_data['title2']}\n")

        # Write atom count and origin
        f.write(f"{cube_data['natoms']:5d} {cube_data['origin'][0]:11.6f} "
                f"{cube_data['origin'][1]:11.6f} {cube_data['origin'][2]:11.6f}\n")

        # Write grid dimensions and voxel vectors
        for i in range(3):
            f.write(f"{cube_data['nvoxels'][i]:5d} "
                   f"{cube_data['voxel_vectors'][i][0]:11.6f} "
                   f"{cube_data['voxel_vectors'][i][1]:11.6f} "
                   f"{cube_data['voxel_vectors'][i][2]:11.6f}\n")

        # Write atoms
        for atom in cube_data['atoms']:
            f.write(f"{int(atom[0]):5d} {atom[1]:11.6f} {atom[2]:11.6f} "
                   f"{atom[3]:11.6f} {atom[4]:11.6f}\n")

        # Write data
        data = cube_data['data'].flatten()
        for i in range(0, len(data), 6):
            chunk = data[i:i+6]
            f.write(' '.join(f"{val:12.5e}" for val in chunk) + '\n')


def create_coordinate_grids(origin, voxel_vectors, nvoxels):
    """
    Create coordinate arrays for a CUBE grid

    Returns:
        x, y, z: 1D coordinate arrays in Bohr
    """
    x = origin[0] + np.arange(nvoxels[0]) * voxel_vectors[0][0]
    y = origin[1] + np.arange(nvoxels[1]) * voxel_vectors[1][1]
    z = origin[2] + np.arange(nvoxels[2]) * voxel_vectors[2][2]

    return x, y, z


def trilinear_interpolate(data, x_coords, y_coords, z_coords, x_query, y_query, z_query):
    """
    Perform trilinear interpolation manually (no scipy needed)

    Args:
        data: 3D array with source data
        x_coords, y_coords, z_coords: 1D arrays of coordinates for source grid
        x_query, y_query, z_query: Query coordinates

    Returns:
        Interpolated value (or 0 if outside bounds)
    """
    # Check if query point is within bounds
    if (x_query < x_coords[0] or x_query > x_coords[-1] or
        y_query < y_coords[0] or y_query > y_coords[-1] or
        z_query < z_coords[0] or z_query > z_coords[-1]):
        return 0.0

    # Find surrounding grid points
    i = np.searchsorted(x_coords, x_query) - 1
    j = np.searchsorted(y_coords, y_query) - 1
    k = np.searchsorted(z_coords, z_query) - 1

    # Handle edge cases
    i = max(0, min(i, len(x_coords) - 2))
    j = max(0, min(j, len(y_coords) - 2))
    k = max(0, min(k, len(z_coords) - 2))

    # Calculate interpolation weights
    xd = (x_query - x_coords[i]) / (x_coords[i+1] - x_coords[i]) if x_coords[i+1] != x_coords[i] else 0.0
    yd = (y_query - y_coords[j]) / (y_coords[j+1] - y_coords[j]) if y_coords[j+1] != y_coords[j] else 0.0
    zd = (z_query - z_coords[k]) / (z_coords[k+1] - z_coords[k]) if z_coords[k+1] != z_coords[k] else 0.0

    # Trilinear interpolation
    c00 = data[i, j, k] * (1 - xd) + data[i+1, j, k] * xd
    c01 = data[i, j, k+1] * (1 - xd) + data[i+1, j, k+1] * xd
    c10 = data[i, j+1, k] * (1 - xd) + data[i+1, j+1, k] * xd
    c11 = data[i, j+1, k+1] * (1 - xd) + data[i+1, j+1, k+1] * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    c = c0 * (1 - zd) + c1 * zd

    return c


def interpolate_cube_onto_grid(cube_small, target_origin, target_voxel_vectors, target_nvoxels):
    """
    Interpolate a smaller CUBE file onto a target grid

    This handles:
    - Different origins
    - Different voxel spacings
    - Different grid dimensions

    Returns:
        interpolated_data: 3D array on target grid
        mask: Boolean array indicating where small cube has data
    """
    print("\nInterpolation details:")
    print(f"  Source grid: {cube_small['nvoxels']}")
    print(f"  Target grid: {target_nvoxels}")
    print(f"  Source origin: {cube_small['origin']}")
    print(f"  Target origin: {target_origin}")

    # Create coordinate grids
    x_small, y_small, z_small = create_coordinate_grids(
        cube_small['origin'],
        cube_small['voxel_vectors'],
        cube_small['nvoxels']
    )

    x_target, y_target, z_target = create_coordinate_grids(
        target_origin,
        target_voxel_vectors,
        target_nvoxels
    )

    print(f"\n  Source Z range: [{z_small[0]:.3f}, {z_small[-1]:.3f}] Bohr")
    print(f"  Target Z range: [{z_target[0]:.3f}, {z_target[-1]:.3f}] Bohr")

    # Initialize output arrays
    interpolated_data = np.zeros(target_nvoxels)
    mask = np.zeros(target_nvoxels, dtype=bool)

    # Calculate bounds for masking
    x_min, x_max = x_small.min(), x_small.max()
    y_min, y_max = y_small.min(), y_small.max()
    z_min, z_max = z_small.min(), z_small.max()

    # Interpolate
    print("\n  Interpolating... ", end='', flush=True)
    total_points = np.prod(target_nvoxels)
    points_processed = 0
    update_interval = total_points // 20  # Update every 5%

    for i in range(target_nvoxels[0]):
        for j in range(target_nvoxels[1]):
            for k in range(target_nvoxels[2]):
                x = x_target[i]
                y = y_target[j]
                z = z_target[k]

                # Check if point is within small cube bounds
                if (x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max):
                    mask[i, j, k] = True
                    interpolated_data[i, j, k] = trilinear_interpolate(
                        cube_small['data'], x_small, y_small, z_small, x, y, z
                    )

                points_processed += 1
                if points_processed % update_interval == 0:
                    progress = points_processed / total_points * 100
                    print(f"\r  Interpolating... {progress:.0f}%", end='', flush=True)

    print("\r  Interpolating... done!           ")

    points_in_region = np.sum(mask)
    points_total = np.prod(target_nvoxels)
    coverage = points_in_region / points_total * 100

    print(f"\n  Coverage: {points_in_region:,} / {points_total:,} points ({coverage:.1f}%)")
    print(f"  Interpolated data range: [{interpolated_data.min():.6e}, {interpolated_data.max():.6e}]")

    return interpolated_data, mask


def subtract_cubes(cube_large, cube_small, verbose=True):
    """
    Subtract cube_small from cube_large

    Returns result on cube_large's grid
    """
    if verbose:
        print("\n" + "="*70)
        print("CUBE SUBTRACTION WITH GRID ALIGNMENT")
        print("="*70)
        print(f"\nLarge cube: {cube_large['nvoxels']} grid")
        print(f"Small cube: {cube_small['nvoxels']} grid")

    # Interpolate small cube onto large cube's grid
    interpolated_small, mask = interpolate_cube_onto_grid(
        cube_small,
        cube_large['origin'],
        cube_large['voxel_vectors'],
        cube_large['nvoxels']
    )

    # Perform subtraction
    result_data = cube_large['data'] - interpolated_small

    if verbose:
        print("\n" + "-"*70)
        print("SUBTRACTION RESULTS:")
        print("-"*70)
        print(f"  Large data range:   [{cube_large['data'].min():.6e}, {cube_large['data'].max():.6e}]")
        print(f"  Small data range:   [{cube_small['data'].min():.6e}, {cube_small['data'].max():.6e}]")
        print(f"  Interpolated range: [{interpolated_small.min():.6e}, {interpolated_small.max():.6e}]")
        print(f"  Result range:       [{result_data.min():.6e}, {result_data.max():.6e}]")

        # Calculate how much changed
        diff = cube_large['data'] - result_data
        significant_change = np.sum(np.abs(diff) > 1e-6)
        total_points = np.prod(cube_large['nvoxels'])

        print(f"\n  Points changed: {significant_change:,} / {total_points:,} "
              f"({significant_change/total_points*100:.1f}%)")

        # Show mean values in subtraction region
        if np.any(mask):
            print(f"\n  In subtraction region:")
            print(f"    Large mean:   {cube_large['data'][mask].mean():.6e}")
            print(f"    Small mean:   {interpolated_small[mask].mean():.6e}")
            print(f"    Result mean:  {result_data[mask].mean():.6e}")

    # Create result cube (using large cube's metadata)
    result_cube = {
        'title1': f"SUBTRACTED: {cube_large['title1'].strip()}",
        'title2': f"           - {cube_small['title1'].strip()}",
        'natoms': cube_large['natoms'],
        'origin': cube_large['origin'].copy(),
        'nvoxels': cube_large['nvoxels'].copy(),
        'voxel_vectors': cube_large['voxel_vectors'].copy(),
        'atoms': cube_large['atoms'].copy(),
        'data': result_data
    }

    return result_cube


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python subtract_cubes.py large.CUBE small.CUBE output.CUBE")
        print("\nSubtracts small.CUBE from large.CUBE and writes result to output.CUBE")
        print("Handles different origins, voxel sizes, and grid dimensions via interpolation.")
        sys.exit(1)

    large_file = sys.argv[1]
    small_file = sys.argv[2]
    output_file = sys.argv[3]

    print(f"Reading large CUBE file: {large_file}")
    cube_large = read_cube_file(large_file)

    print(f"Reading small CUBE file: {small_file}")
    cube_small = read_cube_file(small_file)

    # Perform subtraction
    result_cube = subtract_cubes(cube_large, cube_small, verbose=True)

    # Write result
    print(f"\nWriting result to: {output_file}")
    write_cube_file(output_file, result_cube)

    print("\n" + "="*70)
    print("SUBTRACTION COMPLETE!")
    print("="*70)
    print(f"\nYou can now visualize the result:")
    print(f"  python crystal_cubeviz_plotly.py {output_file} --iso-browse")
