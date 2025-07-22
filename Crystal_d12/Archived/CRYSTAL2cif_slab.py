#!/usr/bin/env python

"""
This is a simple script to convert the final geometry of CRYSTAL09 output into cif file
Usage: CRYSTAL2cif.py [CRYSTAL09 output] [cif file name] [c value; default=20]

Author: Daniel Maldonado Lopez
Institution: Michigan State University, Mendoza Group
"""
import os, sys, math
import re

cryout = open(str(sys.argv[1]))
cif  = open(str(sys.argv[2]),'w')

for line in cryout:
  #if line.find(" PRIMITIVE CELL - ",0,20) == 0:
  if re.match(r"^ PRIMITIVE CELL", line):
    cryout.next()
    [a,b,c,alpha,beta,gamma] = [float(i) for i in cryout.next().split()]
    if len(sys.argv) > 3:
      c = float(sys.argv[3])
    else:
      c = 40.0
    cryout.next()
    natoms = int(cryout.next().split()[-1])
    cryout.next()
    cryout.next()
    atomindex = [0]*natoms
    atomname = ["N/A"]*natoms
    atomx = [1.0]*natoms
    atomy = [1.0]*natoms
    atomz = [1.0]*natoms
    for i in range(natoms):
      atom = cryout.next().split()
      atomindex[i] = int(atom[0])
      atomname_i = str(atom[3])
      if len(atomname_i) == 1:
          atomname[i] = atomname_i
      elif len(atomname_i) == 2:
          ABCs = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' 
          abcs = 'abcdefghijklmnopqrstuvwxyz'
          lowercase = {ABCs[i]:abcs[i] for i in range(len(ABCs))}
          atomname[i] = atomname_i[0] + lowercase[atomname_i[1]]
      atomx[i] = float(atom[4])
      atomy[i] = float(atom[5])
      atomz[i] = float(atom[6])/c
    #uncomment this if want initial structure
    #break

cryout.close()

print >> cif, "data_%s"%(str(sys.argv[1]))
print >> cif, ""
print >> cif, "_cell_length_a                         %.6f"%(a)
print >> cif, "_cell_length_b                         %.6f"%(b)
print >> cif, "_cell_length_c                         %.6f"%(c)
print >> cif, "_cell_angle_alpha                      %.6f"%(alpha)
print >> cif, "_cell_angle_beta                       %.6f"%(beta)
print >> cif, "_cell_angle_gamma                      %.6f"%(gamma)
print >> cif, "_symmetry_space_group_name_H-M         'P 1'"
print >> cif, "_symmetry_Int_Tables_number            1"
print >> cif, ""
print >> cif, "loop_"
print >> cif, "_symmetry_equiv_pos_as_xyz"
print >> cif, "   'x, y, z'"
print >> cif, ""
print >> cif, "loop_"
print >> cif, "   _atom_site_label"
print >> cif, "   _atom_site_type_symbol"
print >> cif, "   _atom_site_fract_x"
print >> cif, "   _atom_site_fract_y"
print >> cif, "   _atom_site_fract_z"
for i in range(natoms):
  print >> cif, "%2s%03d  %2s  %9.6f  %9.6f  %9.6f"%(atomname[i],atomindex[i],atomname[i],atomx[i],atomy[i],atomz[i])
#print >> cif, "#END"

cif.close()

