"""
This script plots the BANDS from CRYSTAL17/23 .dat and .d3 files.
Also recognizes _POTC.POTC.dat and _POTC.out files to plot bands wrt vacuum.
Enhanced version that automatically extracts band path labels from .d3 files.

Usage:
python ipBANDS.py [E lower limit] [E upper limit]

Example:
python ipBANDS.py -5 5
^ This plots bands in a range of (+-)5 eV around the fermi level
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
import matplotlib as mpl
import glob
from matplotlib import rc
import math as m
from matplotlib.cm import get_cmap
from os.path import exists
import re

mpl.rcParams.update(mpl.rcParamsDefault)

E_l = float(sys.argv[1])
E_u = float(sys.argv[2])

def parse_d3_file(d3_file):
    """
    Parse the .d3 file to extract band path segments.
    Returns tuple of (path_segments, labels) or None.
    """
    if not exists(d3_file):
        print(f"Warning: {d3_file} not found")
        return None
    
    try:
        with open(d3_file, 'r') as f:
            lines = f.readlines()
        
        # Debug: print first few lines to understand format
        #print(f"\nFirst 20 lines of {d3_file}:")
        #for i, line in enumerate(lines[:20]):
            #print(f"{i}: {line.rstrip()}")
        
        # Find the BAND section
        band_start = -1
        for i, line in enumerate(lines):
            if line.strip().upper().startswith('BAND'):
                band_start = i
                #print(f"Found BAND section at line {i}")
                break
        
        if band_start == -1:
            print("No BAND section found")
            return None
        
        # Parse from the BAND section
        current_line = band_start + 1
        path_segments = []
        
        # Skip until we find the band path definitions
        while current_line < len(lines):
            line = lines[current_line].strip()
            
            # Skip empty lines
            if not line:
                current_line += 1
                continue
                
            # Stop at END
            if line.upper() == 'END':
                break
            
            # Try different parsing strategies
            parts = line.split()
            
            # Strategy 1: Look for lines with 6 numbers followed by labels
            if len(parts) >= 6:
                try:
                    # Check if first 6 are numbers (k-points)
                    nums = [float(x) for x in parts[:6]]
                    
                    # If there are more parts, they might be labels
                    if len(parts) > 6:
                        remaining = ' '.join(parts[6:])
                        
                        # Look for arrow notation (e.g., "GAMMA -> X")
                        if '->' in remaining:
                            label_parts = remaining.split('->')
                            if len(label_parts) == 2:
                                start_label = label_parts[0].strip()
                                end_label = label_parts[1].strip()
                                if start_label and end_label:
                                    path_segments.append((start_label, end_label))
                                    print(f"Found path segment: {start_label} -> {end_label}")
                        
                        # Look for "to" notation
                        elif ' to ' in remaining.lower():
                            label_parts = remaining.lower().split(' to ')
                            if len(label_parts) == 2:
                                start_label = label_parts[0].strip().upper()
                                end_label = label_parts[1].strip().upper()
                                if start_label and end_label:
                                    path_segments.append((start_label, end_label))
                                    print(f"Found path segment: {start_label} to {end_label}")
                    
                except ValueError:
                    pass
            
            # Strategy 2: Simple label pairs (e.g., "GAMMA X" or "A G")
            elif len(parts) == 2 and all(is_valid_symmetry_label(p) for p in parts):
                path_segments.append((parts[0], parts[1]))
                print(f"Found simple path: {parts[0]} {parts[1]}")
            
            current_line += 1
        
        # Convert path segments to ordered list of labels
        labels = []
        if path_segments:
            print(f"All path segments: {path_segments}")
            
            # Build the path with discontinuity detection
            labels.append(path_segments[0][0])  # Start with first point
            
            for i, (start, end) in enumerate(path_segments):
                if i == 0:
                    labels.append(end)
                else:
                    # Check if this segment connects to where we left off
                    if labels[-1] != start:
                        # Discontinuity detected!
                        # Combine the last label with the new start label
                        last_label = labels[-1]
                        combined_label = f"{last_label}|{start}"
                        labels[-1] = combined_label
                    
                    # Add the end point
                    labels.append(end)
            
            #print(f"Built path with discontinuities: {labels}")
        
        # Clean and standardize labels (but preserve combined labels)
        if labels:
            cleaned_labels = []
            for label in labels:
                if '|' in label:
                    # Handle combined labels
                    parts = label.split('|')
                    cleaned_parts = [clean_label(p) for p in parts]
                    cleaned_labels.append('|'.join(cleaned_parts))
                else:
                    cleaned_labels.append(clean_label(label))
            labels = cleaned_labels
            #print(f"Final extracted labels: {labels}")
        
        # Return both segments and labels
        return (path_segments, labels) if labels else None
    
    except Exception as e:
        print(f"Error parsing {d3_file}: {e}")
        import traceback
        traceback.print_exc()
        return None

def is_valid_symmetry_label(label):
    """
    Check if a label looks like a valid high-symmetry point label.
    """
    if not label or len(label) > 10:
        return False
    
    # Remove common non-label strings
    if label.lower() in ['to', 'from', '->', '<-', '|', '#', '@']:
        return False
    
    # Common high-symmetry point labels
    common_labels = {
        'G', 'GAMMA', 'GAM', 'Γ',
        'X', 'Y', 'Z', 'M', 'K', 'H',
        'A', 'L', 'W', 'R', 'S', 'T', 'U', 'V',
        'D', 'P', 'Q', 'N', 'F', 'E', 'C', 'B',
        'DELTA', 'SIGMA', 'LAMBDA'
    }
    
    label_upper = label.upper()
    
    # Direct match
    if label_upper in common_labels:
        return True
    
    # Subscripted labels (X1, M2, etc.)
    if len(label) >= 2:
        base = label_upper[0]
        if base in common_labels and label[1:].replace('_', '').isdigit():
            return True
    
    # Prime notation (X', M'', etc.)
    prime_count = label.count("'")
    if prime_count > 0:
        base = label.replace("'", "")
        return is_valid_symmetry_label(base)
    
    # Single letters are usually valid
    if len(label) == 1 and label.isalpha():
        return True
    
    return False

def clean_label(label):
    """
    Clean and standardize high-symmetry point labels.
    """
    label = label.upper().strip()
    
    # Handle underscores for subscripts (R_2 -> R2)
    label = label.replace('_', '')
    
    # Convert common variations
    label_map = {
        'GAMMA': 'G',
        'GAM': 'G',
        'Γ': 'G',
        #'DELTA': 'D',
        #'SIGMA': 'S',
        #'LAMBDA': 'L'
    }
    
    # Check if the base (without numbers/primes) is in the map
    base_label = label.rstrip("0123456789'")
    if base_label in label_map:
        return label_map[base_label] + label[len(base_label):]
    
    return label

def format_label_for_plot(label):
    """
    Format label for matplotlib plotting with proper subscripts and primes.
    Handles combined labels (e.g., "X|Y") for discontinuous paths.
    """
    if not label:
        return ""
    
    # Handle combined labels (discontinuities)
    if '|' in label:
        parts = label.split('|')
        formatted_parts = [format_single_label(p) for p in parts]
        return '|'.join(formatted_parts)
    else:
        return format_single_label(label)

def format_single_label(label):
    """
    Format a single label (not combined) for matplotlib.
    """
    # Extract base, subscript, and prime parts
    match = re.match(r"([A-Z]+)(\d*)('*)", label)
    if match:
        base, subscript, prime = match.groups()
        
        # Convert base to Greek if needed
        greek_map = {
            'G': r'\Gamma',
            #'D': r'\Delta',
            #'S': r'\Sigma',
            #'L': r'\Lambda'
        }
        
        base_formatted = greek_map.get(base, base)
        
        # Build the formatted label
        result = base_formatted
        if subscript:
            result = f"{result}_{{{subscript}}}"
        if prime:
            result = f"{result}{prime}"
        
        return f"${result}$"
    
    return label

def get_manual_labels():
    """
    Prompt user to manually input labels.
    """
    print("\nAutomatic label detection failed or no .d3 file found.")
    print("Please enter the high-symmetry point labels manually.")
    print("Enter labels separated by spaces (e.g., G M K G A L H A L M H K):")
    print("Use 'G' for Gamma point.")
    
    user_input = input("Labels: ").strip()
    if not user_input:
        return ["", ""]
    
    labels = user_input.split()
    return [clean_label(label) for label in labels]

def ipBANDS(material, E_l, E_u):
    file2 = material + '_BAND.BAND.dat'
    file3 = material + '_BAND.d3'
    file4 = material + '_POTC.POTC.dat'
    file5 = material + '_POTC.out'

    # If POTC file exists, get vacuum reference
    if exists(file4) and exists(file5):
        z = []
        V = []
        EF = 0
        with open(file5) as f5:
            for line in f5:
                if 'FERMI ENERGY' in line:
                    EF = float(line.split()[-1])
        with open(file4) as f:
            for line in f:
                if line.startswith('#') or line.startswith('@'):
                    continue
                else:
                    z.append(float(line.split()[0]))
                    V.append(float(line.split()[1]))
        maxV = -(V[0] - EF) * 27.2114
    else:
        maxV = 0

    ef = 0.
    l_labels = 999
    l_label = 999
    x_labels = []
    str_labels = []  # Store string labels from .dat file if present

    with open(file2) as fb:
        # Read header
        header = fb.readline()
        N = int(header.split()[2])
        M = int(header.split()[4])
        E = np.zeros(N)
        Ebeta = np.zeros(N)
        BANDS = np.zeros((N, M))
        BANDSbeta = np.zeros((N, M))
        i = 0
        ib = 0
        alpha_beta_counter = 0
        
        # Read the file
        for l, line in enumerate(fb):
            if line.startswith("@ XAXIS TICK SPEC"):
                n_labels = int(line.split()[-1])
                l_label = l
                l_labels = l + 1
            
            # Read x-axis tick positions
            if l == l_labels:
                x_labels.append(float(line.split()[-1]))
                l_labels += 2
                if l_labels > l_label + 2 * n_labels - 1:
                    l_labels = 1e30
            
            # Read x-axis tick labels (if present)
            if l == l_labels - 1 and l > l_label:
                # This line might contain the label string
                label_line = line.strip()
                if label_line and not label_line.startswith('@'):
                    str_labels.append(label_line)
            
            if line.startswith("# EFERMI (HARTREE)"):
                ef = float(line.split()[-1]) * 27.2114
                alpha_beta_counter += 1
            
            if line.startswith("#") or line.startswith("@"):
                continue
            
            # Read band data
            if alpha_beta_counter == 0:  # Alpha bands
                data = line.split()
                if M > 1000 and len(data) != M + 1:
                    while len(data) != M + 1:
                        data = np.concatenate([data, next(fb).split()])
                E[i] = float(data[0])
                for j in range(0, M):
                    BANDS[i, j] = (float(data[j + 1]) * 27.2114) + maxV
                i = i + 1
            
            elif alpha_beta_counter == 1:  # Beta bands
                data = line.split()
                if M > 1000 and len(data) != M + 1:
                    while len(data) != M + 1:
                        data = np.concatenate([data, next(fb).split()])
                Ebeta[ib] = float(data[0])
                for j in range(0, M):
                    BANDSbeta[ib, j] = (float(data[j + 1]) * 27.2114) + maxV
                ib = ib + 1
            
            elif alpha_beta_counter == 2:
                break

    # Try to get labels from .d3 file
    d3_result = parse_d3_file(file3)
    
    if d3_result:
        path_segments, auto_labels = d3_result
        #print(f"Using labels from .d3 file: {auto_labels}")
        labels = auto_labels
        
    elif str_labels:
        #print(f"Using labels from .dat file: {str_labels}")
        labels = [clean_label(label) for label in str_labels]
    else:
        print(f"No labels found automatically for {material}")
        labels = get_manual_labels()
    
    # Check if we have duplicate x_labels (spin-polarized calculations)
    # This happens when CRYSTAL outputs the path twice (for alpha and beta spins)
    half_length = len(x_labels) // 2
    if len(x_labels) > 1 and len(x_labels) % 2 == 0:
        # Check if the second half is a duplicate of the first half
        first_half = x_labels[:half_length]
        second_half = x_labels[half_length:]
        
        if first_half == second_half:
            #print(f"Detected duplicated x-positions (spin-polarized). Using first half only.")
            x_labels = first_half
    
    # Ensure correct length
    if len(labels) < len(x_labels):
        labels.extend([""] * (len(x_labels) - len(labels)))
    elif len(labels) > len(x_labels):
        labels = labels[:len(x_labels)]
    
    # Format labels for plotting
    plot_labels = [format_label_for_plot(label) for label in labels]
    
    print(f"Final labels: {labels}")
    print(f"X-axis positions: {x_labels}")
    print(f"Number of labels: {len(labels)}, Number of x positions: {len(x_labels)}")

    # Create figure
    fig = plt.figure(figsize=(10, 5), dpi=100)
    ax = fig.add_subplot(111)

    # Set limits and plot bands
    ax.set(ylim=(E_l + maxV, E_u + maxV), xlim=(x_labels[0], x_labels[-1]))
    ax.plot(E, BANDS, linewidth=1.8, color="#f9665e")
    ax.plot(Ebeta, BANDSbeta, linewidth=1.8, linestyle='--', color="#45b6fe")
    plt.axhline(maxV, color="black", linestyle='--', lw=1.5, alpha=1)
    
    # Vertical lines at high-symmetry points
    for label in x_labels:
        plt.axvline(label, color="black", lw=1, alpha=0.5)

    # Set axis labels
    if exists(file4):
        ax.set_ylabel(r"Energy w.r.t. vac. (eV)", size=18)
    else:
        ax.set_ylabel(r"$E-E_f$ (eV)", size=20)

    # Set x-axis ticks and labels
    plt.xticks(x_labels, plot_labels, size=14)
    plt.yticks(size=18)

    # Legend (dummy lines for labels)
    plt.axvline(x_labels[0] - 100, linewidth=1.8, color="#f9665e", label='Spin up')
    plt.axvline(x_labels[0] - 100, linewidth=1.8, linestyle='--', color="#45b6fe", label='Spin down')

    # Position legend outside plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.75))

    # Save figure
    fig.tight_layout()
    plt.tight_layout()
    
    fig.savefig(FIGDIR + material + '.BANDS.svg', format='svg', dpi=300)
    fig.savefig(FIGDIR + material + '.BANDS.png', format='png', dpi=300)
    plt.close('all')

# Main execution
DIR = os.getcwd() + '/'
FIGDIR = DIR

pathlist = glob.glob(DIR + '*_BAND.BAND.dat')
nDIR = len(DIR)
ntype = len("_BAND.BAND.dat")

for path in pathlist:
    path_in_str = str(path)
    material = path_in_str[nDIR:-ntype]
    if material == "":
        break
    print(f"\nProcessing: {material}")
    ipBANDS(material, E_l, E_u)