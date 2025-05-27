#!/usr/bin/python3
"""
Band Structure Plotting Script for CRYSTAL17/23 Output Files

Overview:
This script plots electronic band structures from CRYSTAL17 or CRYSTAL23 output files.
It supports both spin-polarized and non-spin-polarized calculations, and can plot bands
with respect to vacuum level when appropriate POTC files are available.

The script reads the following file types:
- .BAND.DAT: Contains band energies for each k-point
- .d3: Contains k-point paths and labels in the Brillouin zone
- _POTC.POTC.dat (optional): Electrostatic potential data for vacuum alignment
- _POTC.out (optional): Contains Fermi energy information

Usage:
python ipBANDS_V2.py [E_lower_limit] [E_upper_limit]

Arguments:
[E_lower_limit]: Lower energy limit for the plot (in eV)
[E_upper_limit]: Upper energy limit for the plot (in eV)

Examples:
1. Plot bands within +/-5 eV around Fermi level:
   python ipBANDS_V2.py -5 5

2. Focus on a specific energy range:
   python ipBANDS_V2.py 0 10

Important Considerations:
- Make sure all required input files are in the same directory as the script
- The material name is derived from the filename (e.g., "material" from "material_BAND.BAND.DAT")
- If using vacuum level alignment, ensure both _POTC.POTC.dat and _POTC.out exist
- Update FIGDIR path if you want output figures in a different directory

Requirements:
- Python 3.x with numpy and matplotlib installed
- CRYSTAL17 or CRYSTAL23 generated input files

Output:
The script generates band structure plots as PNG and SVG files named material.BANDS.png/.svg
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator, MultipleLocator
import matplotlib as mpl
import matplotlib.ticker as tck
import glob
from matplotlib import rc
import math as m
from matplotlib.cm import get_cmap
from os.path import exists

# Set up matplotlib style for better visualization
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.usetex': True,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

def read_kpoint_path(d3_file):
    """Read k-point path labels from .d3 file, handling discontinuous paths"""
    try:
        with open(d3_file, 'r') as f:
            # Skip first three lines (header)
            for _ in range(3):
                next(f)
            
            # Read lines until END
            lines = []
            for line in f:
                if line.strip() == 'END':
                    break
                points = line.strip().split()
                if len(points) == 2:
                    lines.append(points)

        # Process the path segments
        path_labels = []
        for i in range(len(lines)):
            current_line = lines[i]
            
            # Add first point of current line
            if i == 0 or current_line[0] != lines[i-1][1]:
                # If this is a discontinuity (current start ≠ previous end)
                if i > 0:
                    # Add separator between previous end and current start
                    path_labels[-1] = f"{path_labels[-1]}|{current_line[0]}"
                else:
                    path_labels.append(current_line[0])
            
            # Add second point of current line
            path_labels.append(current_line[1])

        # Replace 'G' with 'Γ' in labels
        path_labels = [x if '|' not in x else '|'.join(p.replace('G', r'$\Gamma$') for p in x.split('|')) 
                      for x in path_labels]
        path_labels = [x.replace('G', r'$\Gamma$') if '|' not in x else x for x in path_labels]
        
        return path_labels
    except Exception as e:
        print(f"Error reading k-point path from {d3_file}: {str(e)}")
        return None

def ipBANDS(material, E_l, E_u):
    file2 = material+'_BAND.BAND.DAT'
    file3 = material+'_BAND.d3'
    file4 = material+'_POTC.POTC.DAT'
    file5 = material+'_POTC.out'

    # Read k-point labels from .d3 file
    labels = read_kpoint_path(file3)
    if labels is None:
        print(f"Warning: Could not read k-point path from {file3}")
        # Fallback to default labels
        labels = ['M','G','K','A','G','L','H','G']
    print(f"Labels {labels} will be used from {file3}")
    
    #If POTC file exists, open file and look electrostatic potential at inf and Efermi
    if exists(file4) and exists(file5):
        z = []; V = [];
        EF = 0
        with open(file5) as f5:
            for i,line in enumerate(f5):
                if 'FERMI ENERGY' in line:
                    EF = float(line.split()[-1])
        with open(file4) as f:
            for i,line in enumerate(f):
                if line.startswith('#') or line.startswith('@'):
                    continue
                else:
                    z.append(float(line.split()[0]))
                    V.append(float(line.split()[1]))
        maxV = -(V[0]-EF)*27.2114
    else:
        maxV = 0

    ef = 0.
    l_labels = 999
    l_label = 999
    x_labels = []

    with open(file2) as fb:
        # Initiate variables
        header = fb.readline()
        N = int(header.split()[2])
        M = int(header.split()[4])
        E = np.zeros(N)
        Ebeta = np.zeros(N)
        BANDS = np.zeros((N,M))
        BANDSbeta = np.zeros((N,M))
        i = 0
        ib = 0
        alpha_beta_counter = 0
        
        #This "if" is to correct crystal formatting when you have over 1000 elements per line
        if M > 1000:
            for l,line in enumerate(fb):
                if line.startswith("@ XAXIS TICK SPEC"):
                    n_labels = int(line.split()[-1])
                    l_label = l
                    l_labels = l+1
                if l == l_labels:
                    x_labels.append(float(line.split()[-1]))
                    l_labels += 2
                    if l_labels > l_label+2*n_labels-1:
                        l_labels=1e30
                if line.startswith("# EFERMI (HARTREE)"):
                    ef = float(line.split()[-1])*27.2114
                    alpha_beta_counter += 1
                if line.startswith("#") or line.startswith("@"):
                    continue
                if alpha_beta_counter == 0:
                    data = line.split()
                    while len(data) != M+1:
                        data = np.concatenate([data,next(fb).split()])
                    E[i] = float(data[0])
                    for j in range(0,M):
                        BANDS[i,j] = (float(data[j+1])*27.2114)+maxV
                    i = i+1
                if alpha_beta_counter == 1:
                    data = line.split()
                    while len(data) != M+1:
                        data = np.concatenate([data,next(fb).split()])
                    Ebeta[ib] = float(data[0])
                    for j in range(0,M):
                        BANDSbeta[ib,j] = (float(data[j+1])*27.2114)+maxV
                    ib = ib+1
                if alpha_beta_counter == 2:
                    break
        else:
            for l,line in enumerate(fb):
                if line.startswith("@ XAXIS TICK SPEC"):
                    n_labels = int(line.split()[-1])
                    l_label = l
                    l_labels = l+1
                if l == l_labels:
                    x_labels.append(float(line.split()[-1]))
                    l_labels += 2
                    if l_labels > l_label+2*n_labels-1:
                        l_labels=1e30
                if line.startswith("# EFERMI (HARTREE)"):
                    ef = float(line.split()[-1])*27.2114
                    alpha_beta_counter += 1
                if line.startswith("#") or line.startswith("@"):
                    continue
                if alpha_beta_counter == 0:
                    data = line.split()
                    E[i] = float(data[0])
                    for j in range(0,M):
                        BANDS[i,j] = (float(data[j+1])*27.2114)+maxV
                    i = i+1
                if alpha_beta_counter == 1:
                    data = line.split()
                    Ebeta[ib] = float(data[0])
                    for j in range(0,M):
                        BANDSbeta[ib,j] = (float(data[j+1])*27.2114)+maxV
                    ib = ib+1
                if alpha_beta_counter == 2:
                    break

    # After reading all the data, print dimensions for debugging
    print(f"E array shape: {E.shape}")
    print(f"BANDS array shape: {BANDS.shape}")
    print(f"Data range - X: {E[0]} to {E[-1]}, Y: {np.min(BANDS)} to {np.max(BANDS)}")
    
    # Create figure with fixed aspect ratio
    fig = plt.figure(figsize=(4, 8))  # Reduced width from 6 to 4
    ax = fig.add_subplot(111)
    
    # Set title with larger font
    ax.set_title('Band Structure', pad=10, fontsize=18, weight='bold')

    # Define colors for spin up/down
    spin_up_color = '#fa26a0' #pink  #'#FF3366'    # Vivid pink-red
    spin_down_color = '#2ff3e0' #cyan    #'#3399FF'  # Bright blue
    
    # Plot bands with increased line width for better visibility
    for j in range(BANDS.shape[1]):
        ax.plot(E, BANDS[:,j], linewidth=2.0, color=spin_up_color, zorder=2)  # Increased from 1.8
        ax.plot(Ebeta, BANDSbeta[:,j], linewidth=2.0, linestyle='--', color=spin_down_color, 
                dashes=(5, 2), zorder=2)
    
    # Add single legend entries using dummy lines
    dummy_line_up = ax.plot([], [], linewidth=2.0, color=spin_up_color, label='Spin ↑', zorder=2)
    dummy_line_down = ax.plot([], [], linewidth=2.0, linestyle='--', color=spin_down_color, 
                              label='Spin ↓', dashes=(5, 2), zorder=2)
    
    # Plot Fermi level with increased visibility
    fermi_line = plt.axhline(maxV, color="black", linestyle='-.', lw=1.5, alpha=0.8, label='E$_F$', zorder=1)

    
    # Add vertical lines at high-symmetry points
    for label in x_labels:
        plt.axvline(label, color="gray", lw=1.0, alpha=0.4, zorder=0)

    # Set plot limits and labels
    ax.set(ylim=(E_l+maxV, E_u+maxV), xlim=(x_labels[0], x_labels[-1]))
    
    # Customize y-axis with larger ticks
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    
    # Label formatting with increased font sizes
    if exists(file4):
        ax.set_ylabel(r"Energy w.r.t. Vacuum (eV)", fontsize=16, weight='bold')
    else:
        ax.set_ylabel(r"$E-E_F$ (eV)", fontsize=16, weight='bold')
    
    ax.set_xlabel(r"Wave Vector", fontsize=16, weight='bold')
    
    # Customize ticks with larger fonts
    plt.xticks(x_labels[0:len(labels)], labels, fontsize=14)
    plt.yticks(fontsize=14)
    
    # Enhance tick parameters
    ax.tick_params(which='major', length=8, width=1.2)  # Increased size
    ax.tick_params(which='minor', length=5, width=1.0)  # Increased size
    
    # # Enhanced legend with background box
    # legend = ax.legend(loc='upper right', 
    #                   framealpha=0.85,  # More opaque background
    #                   edgecolor='gray',
    #                   fancybox=True, 
    #                   fontsize=12,  # Increased font size
    #                   bbox_to_anchor=(0.98, 0.98),  # Slight adjustment of position
    #                   bbox_transform=ax.transAxes,
    #                   borderpad=0.8,  # Increased padding inside legend box
    #                   handlelength=2.5,  # Longer legend lines
    #                   handletextpad=0.6,  # Space between line and text
    #                   borderaxespad=0.5)  # Padding between legend and plot
    
    # # Add a white background to legend with black edge
    # legend.get_frame().set_facecolor('white')
    # legend.get_frame().set_linewidth(1.0)
    
    # Create the legend with only these specific items
    legend = plt.legend(
        [dummy_line_up[0], dummy_line_down[0], fermi_line],
        ['Spin ↑', 'Spin ↓', 'E$_F$'],  # Explicit labels for the lines
        loc='upper right',
        frameon=True
    )

    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')
    legend.get_frame().set_linewidth(1.0)
    #legend.set_title('Legend Title', prop={'size':'15'})

    
    # Layout adjustments
    try:
        plt.tight_layout()
    except Warning:
        plt.subplots_adjust(left=0.20, right=0.95, top=0.95, bottom=0.1)  # Adjusted left margin for larger labels
    
    # Save figures
    fig.savefig(FIGDIR + material+'.BANDS.png', format='png', dpi=600, 
                bbox_inches='tight', pad_inches=0.1)
    fig.savefig(FIGDIR + material+'.BANDS.svg', format='svg', 
                bbox_inches='tight', pad_inches=0.1)
    
    plt.close('all')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python ipBANDS_V2.py [E lower limit] [E upper limit]")
        sys.exit(1)
        
    E_l = float(sys.argv[1])
    E_u = float(sys.argv[2])

    # Define directories
    DIR = os.getcwd() + '/'
    FIGDIR = DIR

    # Get list of band structure files
    pathlist = glob.glob(DIR+'*_BAND.BAND.DAT')
    nDIR = len(DIR)
    ntype = len("_BAND.BAND.DAT")

    # Process all band structure files
    for path in pathlist:
        path_in_str = str(path)
        material = path_in_str[nDIR:-ntype]
        if material == "":
            break
        print(f"Processing {material}...")
        ipBANDS(material, E_l, E_u)
