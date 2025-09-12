# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import math
import os
import glob
import itertools
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import re

def parse_concatenated_floats(s):
    """
    Parse a string that may contain concatenated scientific notation numbers.
    E.g., '-4.41239E-10-6.33884E-10' -> ['-4.41239E-10', '-6.33884E-10']
    """
    import re
    # Pattern to match scientific notation numbers including the sign
    # This matches patterns like: -1.23456E-10 or 1.23456E+10 or -1.23456E10
    pattern = r'[+-]?\d+\.?\d*[Ee][+-]?\d+'
    matches = re.findall(pattern, s)
    return matches

def getdataf25(fort25):
    data = []
    with open(fort25) as f:
        for i,line in enumerate(f):
            if i == 0:
                # Parse the first line to get nx, ny, and pixel increments
                parts = line.split()
                nx = int(parts[1])
                ny = int(parts[2])
                dx_angstrom = float(parts[3])  # increment in x (Angstrom)
                dy_angstrom = float(parts[4])  # increment in y (Angstrom)
                # Convert from Angstrom to nanometers (1 Å = 0.1 nm)
                dx_nm = dx_angstrom * 0.1
                dy_nm = dy_angstrom * 0.1
                nlines = math.ceil(nx*ny/6)
            if i > 2 and i <= nlines+2:
                # First try normal splitting
                parts = line.split()
                parsed_parts = []
                for part in parts:
                    try:
                        # Try to convert directly
                        float(part)
                        parsed_parts.append(part)
                    except ValueError:
                        # If it fails, it might be concatenated numbers
                        separated = parse_concatenated_floats(part)
                        if separated:
                            parsed_parts.extend(separated)
                        else:
                            # If parsing fails, keep the original
                            parsed_parts.append(part)
                data.extend(parsed_parts)
    
    # Convert all parsed values to float
    data_vect = []
    for x in data:
        try:
            data_vect.append(float(x))
        except ValueError:
            print(f"Warning: Could not convert '{x}' to float, skipping...")
            continue
    mean = np.mean(data_vect)
    std = np.std(data_vect)
    min = mean-std
    max = mean+std
    # max = 0
    # min = 1E-4
    # max = np.max(data_vect)
    # min = np.min(data_vect)
    return(data_vect, nx, ny, max, min, dx_nm, dy_nm)

def formatf25(material, data_vect, nx):
    data_matrix = [data_vect[i:i+nx] for i in range(0,len(data_vect),nx)]
    data_matrix = np.matrix(data_matrix)
    with open(material+'_matrix.txt','w+') as f:
        for line in data_matrix:
            np.savetxt(f, line, fmt='%.10f')
    return(data_matrix)

def plot_f25(data, nx, ny, dx_nm, dy_nm, vmin=None, vmax=None, dpi=200, save=False, save_name=None):
    fig, ax = plt.subplots(1, dpi=dpi, figsize=(8, 6))
    color_SnSe2 = "copper"#(0, 110/255, 174/255)
    
    # Calculate the extent of the image in nanometers
    # extent = [left, right, bottom, top]
    x_extent_nm = nx * dx_nm
    y_extent_nm = ny * dy_nm
    extent = [0, x_extent_nm, 0, y_extent_nm]
    
    vmin = min#0
    vmax = max#max#1E-4#np.max(data)#0.003
    
    # Use extent parameter to set the proper scale
    im = ax.imshow(np.asarray(data), 
                   vmin=vmin, 
                   vmax=vmax, 
                   cmap=color_SnSe2,
                   extent=extent,
                   origin='lower',  # Put origin at bottom-left
                   aspect='equal')  # Keep aspect ratio equal
    
    ax.set_xlabel('x (nm)', fontsize=12)
    ax.set_ylabel('y (nm)', fontsize=12)
    ax.set_title(save_name, fontsize=14)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(r'Electron/Bohr$^3$', fontsize=12)
    
    # Add text showing the dimensions
    textstr = f'Size: {x_extent_nm:.2f} × {y_extent_nm:.2f} nm²'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_name is None:        
        save_name = time.strftime("%Y%m%d-%H%M%S")
    
    #fig.savefig(save_name + ".pdf", dpi=dpi,format='pdf')
    fig.savefig(save_name + ".png", dpi=dpi,format='png')
    plt.close(fig)
    
# Main execution
DIR = (os.getcwd()+'/')
pathlist = glob.glob(DIR+'*.f25')
nDIR = len(DIR)
ntype = len('.f25')

for path in pathlist:
    material = str(path[nDIR:-ntype])
    print(f"Processing: {material}")
    data_vect, nx, ny, max, min, dx_nm, dy_nm = getdataf25(str(path))
    print(f"  Grid: {nx}×{ny} pixels")
    print(f"  Pixel size: {dx_nm:.4f}×{dy_nm:.4f} nm")
    print(f"  Total size: {nx*dx_nm:.2f}×{ny*dy_nm:.2f} nm²")
    data_matrix = formatf25(material, data_vect, nx)
    plot_f25(data_matrix, nx, ny, dx_nm, dy_nm, vmin=min, vmax=max, dpi=200, save=False, save_name=material)
