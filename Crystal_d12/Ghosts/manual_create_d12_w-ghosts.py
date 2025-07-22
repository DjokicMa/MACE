"""
This script adds ghost atom layers above and below slab structures in CRYSTAL .d12 files, using a user-specified spacing.
It generates a modified geometry file with inserted ghost atoms for surface calculations.

Author: Daniel Maldonado Lopez
Institution: Michigan State University, Mendoza Group
"""

from pathlib import Path
import warnings
import numpy as np
import glob
import math
import matplotlib
matplotlib.use('tkagg') # <-- THIS MAKES IT FAST!
import matplotlib.pyplot as plt
import sys
import csv
import os

DIR    = os.getcwd() + '/'
pathlist = glob.glob(DIR+'*_slab.d12')
nDIR     = len(DIR)
ntype    = len("_slab.out")
LAT = ' LATTICE PARAMETERS  (ANGSTROMS AND DEGREES) - PRIMITIVE CELL'
CART   = " CARTESIAN COORDINATES - PRIMITIVE CELL"
#################################################################################

for path in pathlist:
    path_in_str = str(path)
    material = path_in_str[nDIR:-ntype]
    if material == "":break
    print(material)
    space = float(input("Insert distance between ghost atoms and slab (Angstoms): "))
    SLABd12  = material+"_slab.d12"
    new_SLABd12 = material+'_'+str(space)+"A_ghosts_slab.d12"
    # if os.path.exists(DIR+SLAB) and os.path.exists(DIR+BULK):
    #     #:::::::::::::::   BULK ANALYSIS   ::::::::::::::::::::::
    #     with open(DIR+BULK,'r') as blk:
    #         el = np.zeros(99)
    #         x  = np.zeros(99)
    #         y  = np.zeros(99)
    #         z  = np.zeros(99)
    #         z_c  = np.zeros(99)
    #         z_new = np.zeros(99)
    #
    #         id  = 9999
    #         md  = 9999
    #         g = 9999
    #         atoms    = 99
    #         d      = 0.
    #         k      = 0
    #         switch   = 0
    #         E_blk    = 999999
    #         IAD_blk  = 999999
    #         state_blk    = "Null"
    #         Ebg_blk    = 9999999
    #         GAPTYPE_blk  = "Null"
    #         for i, line in enumerate(blk):
    #           # GET CARTESIAN POSTIONS FOR ATOMS
    #           if line.startswith(" NUMBER OF IRREDUCIBLE"):
    #             atoms = int(line.split()[-1])
    #           if line.startswith(" CARTESIAN COORDINATES - PRIMITIVE CELL"):
    #               md = i
    #           if " PROCESS" in line:
    #             md+=1
    #             continue
    #           if md+4 <= i and i <= md+atoms+3:
    #              el[i-md-4] = float(line.split()[-5])
    #              if el[i-md-4] >200:el[i-md-11] -=200
    #              x[i-md-4] = float(line.split()[-3])
    #              y[i-md-4] = float(line.split()[-2])
    #              z[i-md-4] = float(line.split()[-1])
    #           # GET THE INFORMATION ON THE PRIMITIVE CELL (ANG. and DEG.)
    #           if LAT in line:
    #             id = i
    #           if i == id+2:
    #             # FIND THE OPTIMIZED PRIMITIVE CELL
    #             cell    = line.split()
    #             A     = float(cell[0])
    #             B     = float(cell[1])
    #             C     = float(cell[2])
    #             alpha   = float(cell[3])
    #             beta    = float(cell[4])
    #             gamma   = float(cell[5])
    #             V     = float(cell[6])
    #             ang_alpha = math.radians(alpha)
    #             ang_beta = math.radians(beta)
    #             ang_gamma = math.radians(gamma)
    #           if line.startswith(" ATOMS IN THE ASYMMETRIC UNIT"):
    #             g = i
    #           if i>=g+3 and i<=g+3+atoms-1:
    #             z_c[i-g-3] = float(line.split()[6])
    #             if z[i-g-3] < 0.:
    #                 z_new[i-g-3] = z[i-g-3] + 1.0
    #             else:
    #                 z_new[i-g-3] = z[i-g-3]
        #interlayer spacings:
        # if (1+min(z_c)-max(z_c)) > 0.001:
        #     space = ((1+min(z_c))-max(z_c))*(C/math.sin(ang_gamma))*math.sqrt(1-math.cos(ang_alpha)**2-math.cos(ang_beta)**2-math.cos(ang_gamma)**2+2*math.cos(ang_alpha)*math.cos(ang_beta)*math.cos(ang_gamma))
        # elif (1+min(z_c)-max(z_c)) < 0.001:
        #     space = ((1-max(z_new)+min(z_new)))*(C/math.sin(ang_gamma))*math.sqrt(1-math.cos(ang_alpha)**2-math.cos(ang_beta)**2-math.cos(ang_gamma)**2+2*math.cos(ang_alpha)*math.cos(ang_beta)*math.cos(ang_gamma))
        #
        # blk.close()
    lines_in_d12 = []
    end_index = []
    counter = 999999999
    with open(DIR+SLABd12,'r') as f:
        for i, line in enumerate(f):
            if "END" in line:
                end_index.append(i) #Keeps track of the "END"s this will allow us to insert lines later
            if line.startswith('99 0'):
                idx_ghostatoms = int(i) + 1 #counter for idx_ghostatoms starts after "99 0"
            if i == 4:
                num_atoms = int(line)
                counter = i + 1
                el = np.zeros(num_atoms)
                x = np.zeros(num_atoms)
                y = np.zeros(num_atoms)
                z = np.zeros(num_atoms)
            if counter <= i and i <= counter+num_atoms-1:
                el[i-counter] = int(line.split()[0])
                #if el[i-counter] >200:el[i-counter]-=200 #check if this is needed
                x[i-counter] = float(line.split()[1])
                y[i-counter] = float(line.split()[2])
                z[i-counter] = float(line.split()[3])
            lines_in_d12.append(line)
    f.close()
    idx = min(end_index) # "idx" tracks the line where the first "END" was

    # z_high and z_low designate where we will put the atoms in the z-direction
    # I'm using the interlayer bulk distance (called "space" in this script) as the separation distance
    z_high = [x+(max(z)-min(z))+space for x in z]
    z_low = [x-(max(z)-min(z))-space for x in z]

    # We initiate a bunch of variables
    el_ghosts = [*el,*el]
    el_ghosts = [int(x) for x in el_ghosts]
    x_ghosts = [*x,*x]
    y_ghosts = [*y,*y]
    z_ghosts = [*z_high,*z_low]
    atom_index = []

    # Append the ghost atom indexing into "atom_idx"
    # This will tell crystal which atoms to turn into ghosts
    # Only works with P1 symmetry
    for i in range(len(x_ghosts)):
        atom_index.append(str(num_atoms+1+i))
    atom_idx = ' '.join(atom_index)
    atom_idx = atom_idx + '\n'

    # Insert keywords in the geomtry input block.
    # We are keeping track of lines with idx and idx_ghostatoms.
    lines_in_d12.insert(idx,'FRACTION\n')
    idx += 1
    idx_ghostatoms += 1
    lines_in_d12.insert(idx,'ATOMINSE\n')
    idx += 1
    idx_ghostatoms += 1
    lines_in_d12.insert(idx,str(num_atoms*2)+'\n')
    idx += 1
    idx_ghostatoms += 1
    # Insert new atoms in the geometry input block
    # We use the same basis set for all of the new atoms
    # We will arbitrarily number these atoms "101"
    for i in range(len(x_ghosts)):
        lines_in_d12.insert(idx+i,str(101)+' '+str(x_ghosts[i])+' '+str(y_ghosts[i])+' '+str(z_ghosts[i])+'\n')
        idx_ghostatoms +=1

    # Insert the basis set for ghosts (101)
    lines_in_d12.insert(idx_ghostatoms-1,'101 1\n')
    idx_ghostatoms += 1
    lines_in_d12.insert(idx_ghostatoms-1,'0  1  1  0.0  1.0\n')
    idx_ghostatoms += 1
    lines_in_d12.insert(idx_ghostatoms-1,'       0.150    1.0       1.0\n')
    idx_ghostatoms += 1

    #Make new inserted atoms into ghosts
    lines_in_d12.insert(idx_ghostatoms,'GHOSTS\n')
    idx_ghostatoms += 1
    lines_in_d12.insert(idx_ghostatoms,str(num_atoms*2)+ '\n') #We are using num_atoms*2 because we are putting a ghost layer above and below
    idx_ghostatoms += 1
    lines_in_d12.insert(idx_ghostatoms,atom_idx)
    #print(lines_in_d12)
    print(space)
    f = open(DIR+new_SLABd12,'w+')
    for items in lines_in_d12:
        f.writelines(items)
    f.close()
