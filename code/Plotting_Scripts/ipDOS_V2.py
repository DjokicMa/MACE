#!/usr/bin/python3
"""
Density of States (DOS) Plotting Script for CRYSTAL17/23 Output Files

Overview:
This script plots the density of states (DOS) from CRYSTAL17 or CRYSTAL23 output files.
It supports both spin-polarized and non-spin-polarized calculations, and can plot DOS
with respect to vacuum level when appropriate POTC files are available.

The script reads the following file types:
- .DOSS.DAT: Contains DOS data for each energy point
- .d3: Contains orbital information and labels
- _POTC.POTC.dat (optional): Electrostatic potential data for vacuum alignment
- _POTC.out (optional): Contains Fermi energy information

Usage:
python ipDOS_V2.py [E lower limit] [E upper limit] [projection_type]

Arguments:
[E lower limit]: Lower energy limit for the plot (in eV)
[E upper limit]: Upper energy limit for the plot (in eV)
[projection_type]:
  'total': Plot only total DOS
  'orbital': Plot orbital-projected DOS
  'both': Combine both total and orbital projections in one plot

Examples:
1. Plot total DOS within +/-5 eV around Fermi level:
   python ipDOS_V2.py -5 5 total

2. Focus on specific energy range with orbital projections:
   python ipDOS_V2.py 0 10 orbital

3. Combine both total and orbital DOS in a single plot:
   python ipDOS_V2.py -7 7 both

Important Considerations:
- Make sure all required input files are in the same directory as the script
- The material name is derived from the filename (e.g., "material" from "material_DOSS.DAT")
- If using vacuum level alignment, ensure both _POTC.POTC.dat and _POTC.out exist
- Update FIGDIR path if you want output figures in a different directory

Requirements:
- Python 3.x with numpy and matplotlib installed
- CRYSTAL17 or CRYSTAL23 generated input files

Output:
The script generates DOS plots as PNG and SVG files named material.DOSS.png/.svg

Notes:
- For 'both' projection type, the total DOS will be shown as filled areas while orbital 
  projections will appear as lines
- Use smaller energy ranges for better visualization of specific features
- Adjust limits around the Fermi level (e.g., -5 to 5) for more detailed analysis
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
import re

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

# Read the user input
if len(sys.argv) != 4:
    print("Usage: python ipDOS_V2.py [E lower limit] [E upper limit] [projection_type]")
    print("projection_type options: 'total' or 'orbital' or 'both'")
    sys.exit(1)

E_l = float(sys.argv[1])
E_u = float(sys.argv[2])
proj_type = sys.argv[3].lower()

if proj_type not in ['total', 'orbital', 'both']:
    print("Error: projection_type must be either 'total' or 'orbital'")
    sys.exit(1)

def ipDOS(material, E_l, E_u, proj_type):
    file = material+'_DOSS.DOSS.DAT'
    file1 = material+'_DOSS.d3'
    file4 = material+'_POTC.POTC.DAT'
    file5 = material+'_POTC.out'

    # We define necessary lists and variables
    v = []
    labels = ['Energy (eV)']
    n = 9999
    num = 9999

    # Grabs the labels from the .d3 file and puts them in a list
    with open(file1) as F:
        for i, line in enumerate(F):
            if line.startswith('DOSS'):
                n = i
            if i == n+1:
                l = line.split()
                num = int(l[0])
            if i >= n+3 and i <= n+3+num:
                v.append(str(line[max(line.find('#'),0):].strip()))
            if line.startswith('END'):
                break

    # Process labels and determine search terms based on projection type
    search = []
    for i in v:
        j = i.replace(' all', ' ')
        j = j.replace(' S', ' (s)')
        j = j.replace(' P', ' (p)')
        j = j.replace(' D', ' (d)')
        j = j.replace(' F', ' (f)')
        j = j.replace('END', 'Total')
        l = j.replace('#', '')
        if len(l) > 1:
            if l[0].isupper() and l[1].isupper():
                j = list(l)
                j[1] = j[1].lower()
                l = ''.join(j)
        labels.append(l)
        
        if proj_type == 'total':
            #print(i)
            if any(x in i for x in [' all']):  # Original label contains ' all'
                #print(l)
                search.append(l)
        elif proj_type == 'both':
            if any(x in i for x in [' all',' S', ' P', ' D', ' F']):  # Original label contains 'both'
                #print(l)
                search.append(l)
        else:
            if any(x in i for x in [' S', ' P', ' D', ' F']):  # Original label contains 'orbital' type
                #print(l)
                search.append(l)
    #print(labels)
    #print(search)
    # Rest of the original file processing code...
    if exists(file4) and exists(file5):
        z = []
        V = []
        EF = 0
        with open(file5) as f5:
            for i, line in enumerate(f5):
                if 'FERMI ENERGY' in line:
                    EF = float(line.split()[-1])
        with open(file4) as f:
            for i, line in enumerate(f):
                if line.startswith('#') or line.startswith('@'):
                    continue
                else:
                    z.append(float(line.split()[0]))
                    V.append(float(line.split()[1]))
        maxV = -(V[0]-EF)*27.2114
    else:
        maxV = 0

    # Data processing
    data_vect = [[] for n in range(len(labels))]
    ef = 0

    with open(file) as f:
        if len(labels) >= 17:
            for i, line in enumerate(f):
                if line.startswith("# EFERMI (HARTREE)"):
                    ef = float(line.split()[-1])*27.2114
                if line.startswith("#") or line.startswith("@"):
                    continue
                if line.startswith("&"):
                    continue
                data = line.split()
                ne = next(f)
                ne = ne.split()
                if len(data) >= 17 and len(ne) < 17:
                    data = np.concatenate([data, ne])
                data_vect[0].append(float(data[0])*27.2114+maxV)
                for j in range(1, len(labels)):
                    data_vect[j].append(float(data[j]))
        else:
            for i, line in enumerate(f):
                if line.startswith("# EFERMI (HARTREE)"):
                    ef = float(line.split()[-1])*27.2114
                if line.startswith("#") or line.startswith("@"):
                    continue
                if line.startswith("&"):
                    continue
                data = line.split()
                data_vect[0].append(float(data[0])*27.2114+maxV)
                for j in range(1, len(labels)):
                    data[j] = float(data[j])/27.2114
                    data_vect[j].append(float(data[j]))

    # Create plot with original formatting
    fig = plt.figure(figsize=(4, 8))
    ax = fig.add_subplot(111)
    ax.set_title('Density of States ', pad=10, fontsize=18, weight='bold')
    
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    
    if exists(file4):
        ax.set_ylabel(r"Energy w.r.t. Vacuum (eV)", fontsize=16, weight='bold')
    else:
        ax.set_ylabel(r"$E-E_F$ (eV)", fontsize=16, weight='bold')

    
    # Set x-axis limits and add proper ticks
    xlimit = np.max([np.abs(1.0*np.min(data_vect[-1])), np.abs(1.0*np.max(data_vect[-1]))])
    ax.set(xlim=(-xlimit, xlimit), ylim=(E_l+maxV, E_u+maxV))
    
    # New x-axis formatting
    ax.xaxis.set_major_locator(MultipleLocator(xlimit/2))
    ax.xaxis.set_minor_locator(MultipleLocator(xlimit/4))
    
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    
    ax.set_prop_cycle(color=['#e56997', '#bd97cb', '#fbc740', '#bcece0'])
    ax.set_xlabel(r"DOS (states eV$^{-1}$ U.C.$^{-1}$)", fontsize=16, weight='bold')
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    ax.tick_params(which='major', length=8, width=1.2)
    ax.tick_params(which='minor', length=5, width=1.0)

    # Plot the data using original styling
    for i in range(1, len(data_vect)):
        if (labels[i].find("Total") == -1):
            continue
        else:
            ax.fill_betweenx(data_vect[0], data_vect[i], label=labels[i], alpha=0.8, color='gainsboro')

    for proj in search:
        for i in range(1, len(data_vect)):
            label_clean = labels[i]
            part_clean = proj
            #print(f"Checking {label_clean} against {part_clean}")
            if label_clean == part_clean:
                ax.plot(data_vect[i], data_vect[0], label=labels[i], linewidth=1.35, alpha=0.7)
                #print(f"Match found: {labels[i]}")

    fermi_line = plt.axhline(maxV, color="black", linestyle='-.', lw=1.5, alpha=0.8, label='E$_F$', zorder=1)    
    plt.axvline(0, color="black", lw=1.5, alpha=0.8, zorder=3)

    legend = plt.legend(loc='upper left', frameon=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')
    legend.get_frame().set_linewidth(1.0)
    
    try:
        plt.tight_layout()
    except Warning:
        plt.subplots_adjust(left=0.20, right=0.95, top=0.95, bottom=0.1)
    
    fig.savefig(FIGDIR + material+'.DOSS.png', format='png', dpi=600, 
                bbox_inches='tight', pad_inches=0.1)
    fig.savefig(FIGDIR + material+'.DOSS.svg', format='svg', 
                bbox_inches='tight', pad_inches=0.1)
    
    plt.close('all')

if __name__ == "__main__":
        
    # Directory setup
    DIR = os.getcwd() + '/'
    FIGDIR = DIR
    
    # Process all files
    pathlist = glob.glob(DIR + '*_DOSS.DOSS.DAT')
    nDIR = len(DIR)
    ntype = len("_DOSS.DOSS.DAT")
    
    for path in pathlist:
        path_in_str = str(path)
        material = path_in_str[nDIR:-ntype]
        if material == "":
            break
        print(f"Processing {material}...")
        ipDOS(material, E_l, E_u, proj_type)
