#!/usr/bin/env python

"""
This script will take the optimized geometry of CRYSTAL14/CRYSTAL09 output and edit the input .d12 file
with the new atom coordinates & unit cell parameter, then save it as a new d12 file named case_new.d12, 
where case is your job name. 
Requirement: to use this script, please have both CRYSTAL14 output file (case.out) and CRYSTAL14 input file(case.d12)
Usage: get_optimize.py 
Kevin Lucht
Modified by Wangwei Lan, Marcus Djokic, and Danny Maldonado 
"""

import os, sys, math
import re
import linecache


"""RETURNS THE PRIMITIVE CELL VALUES"""
def primitive_cell(filename):
   #open output file
   cryout = open(filename, 'r')
   primitive_cell = [] #takes cell parameters
   for line in cryout:
      #if line.find(" FINAL OPTIMIZED GEOMETRY - ", 0,len(line)) == 0:
      #exit loop and loop through final optimized geometry values
      if re.match(r"^ FINAL OPTIMIZED GEOMETRY", line):
         break
   #Begin to read at FINAL OPTIMIZED GEOMETRY      
   for line in cryout:
      #if there is a blank line, exit
      if not line.strip():
         continue
      if re.match(r"^ PRIMITIVE CELL", line):
         line=next(cryout)
      if re.search(r"           ALPHA      BETA       GAMMA", line):
         value= next(cryout).split()
         primitive_cell=value
         break
   cryout.close()
   return primitive_cell

""" RETURNS THE CONVENTIONAL UNIT CELL LATTICE PARAMETER"""
def conventional_cells(filename):
   #open output file
   cryout = open(str(filename), 'r')
   conventional_cell=[] #store the structure information
   tmp=""
   isconventional = False
   for line in cryout:
      if re.match(r"^ FINAL OPTIMIZED GEOMETRY", line):
         break
   #Begin to read at FINAL OPTIMIZED GEOMETRY      
   for line in cryout:
         #if there is a blank line, exit
      if not line.strip():
         continue
      if re.match(r"^ CRYSTALLOGRAPHIC CELL", line):
         line=next(cryout)
         if re.search(r"           ALPHA      BETA       GAMMA", line):
            value= next(cryout).split()
            conventional_cell=value
            isconventional = True
            break

   if not isconventional:
         conventional_cell=primitive_cell(filename)

   cryout.close()
   return conventional_cell
      
""" RETRUN SPACEGROUP OF THIS COMPOUND"""
def get_spacegroup(filename):
   cryout = open(filename,'r')
   spacegroup=0
   for line in cryout:
      if re.search("CRYSTAL",line):
         line=next(cryout)
         value=next(cryout).split()
         spacegroup=value[0]
#         spacegroup.append(value[0])
         break
   return(spacegroup)
   


"""RETURNS A LIST OF THE COORDINATES OF EACH ATOM"""
def copy_coordinates(filename):
   opt_geom_total = [] #store atom information
   atom_num = []       #atom species
   x_coord = []        #atom coordinates
   y_coord = []
   z_coord = [] 
   isconventional = False

   #if this is conventional unit cell get conventional unit cell parameter
   with open(filename,'r') as cryout:
      #Begin to read at FINAL OPTIMIZED GEOMETRY
      for line in cryout:
         if re.match(r"^ FINAL OPTIMIZED GEOMETRY", line):
            break
      for line in cryout:
         if re.search(r"COORDINATES IN THE CRYSTALLOGRAPHIC CELL", line):
            isconventional = True
            for line in cryout:
               string_list = line.split()
               if len(string_list) < 6:
                  pass
               elif string_list[1] == "T":
                  atom_num.append(string_list[2])
                  x_coord.append(string_list[4])
                  y_coord.append(string_list[5])
                  z_coord.append(string_list[6])
                  

   # if this is primitive unit cell, get primitive unit cell
   if not isconventional:
      with open(filename,'r') as cryout:
      #Begin to read at FINAL OPTIMIZED GEOMETRY
         for line in cryout:
            if re.match(r" FINAL OPTIMIZED GEOMETRY", line):
               break
         for line in cryout:
            if re.search(r"ATOMS IN THE ASYMMETRIC UNIT", line):
               for line in cryout:
                  string_list = line.split()
                  if len(string_list) < 6:
                     pass
                  elif string_list[1] == "T":
                     atom_num.append(string_list[2])
                     x_coord.append(string_list[4])
                     y_coord.append(string_list[5])
                     z_coord.append(string_list[6])

   #put all atom information together
   opt_geom_total.append(atom_num)
   opt_geom_total.append(x_coord)
   opt_geom_total.append(y_coord)
   opt_geom_total.append(z_coord)
   cryout.close()
   return opt_geom_total
   
"""FINDS THE ATOM NUMBER OF MN, AND RETURNS A STRING TO PLACE INTO ATOMSPIN"""
# def atomspin(atom_num):
#    list =[]
#    ATOMSPIN = ""
#    Mn = 0
#    for i in range(len(atom_num)):
#       if atom_num[i] == "25" or atom_num[i] == "225":
#          Mn +=1
#          ATOMSPIN = ATOMSPIN + str(i + 1) + " " + "1" + " "
#    list.append(str(Mn))
#    list.append(ATOMSPIN)
#    return list 
   
"""CREATES A NEW D12 WITH THE EXISTING D12"""
def modify_d12(existingd12, newd12, opt_geom_total, conventional_cell, spacegroupNM):

   for line in existingd12:
      #brings you to the number of atoms in the cell of the existing d12
      if re.match(r"^CRYSTAL", line):
         newd12.write(line)
         newd12.write(next(existingd12))
         newd12.write(next(existingd12))
         next((existingd12))
         break
      else:
         newd12.write(line)

   #skip the existing coordinates
   skip = int(next(existingd12))
   for i in range(skip):
      next(existingd12)

   # prints the unit cell parameter, according to the spacegroup the compound has
   if int(spacegroupNM) < 3:
      newd12.write("%1s %8s %8s %8s %8s %8s\n"%(conventional_cell[0],conventional_cell[1],conventional_cell[2],conventional_cell[3],conventional_cell[4],conventional_cell[5]))
   elif int(spacegroupNM) >= 3 and int(spacegroupNM) <= 15:
      for i in (3,5):
         if conventional_cell[i] != "90.000000":
            newd12.write("%1s  %19s %19s %19s\n"%(conventional_cell[0],conventional_cell[1],conventional_cell[2],conventional_cell[i]))
         break
   elif int(spacegroupNM) >= 16 and int(spacegroupNM) <= 74:
      newd12.write("%1s %19s %19s\n"%(conventional_cell[0], conventional_cell[1], conventional_cell[2]))
   elif int(spacegroupNM) >= 75 and int(spacegroupNM) <= 142:
      newd12.write("%1s %19s\n"%(conventional_cell[0],conventional_cell[2]))
   elif int(spacegroupNM) >= 143 and int(spacegroupNM) <= 167:
      newd12.write("%1s %19s\n"%(conventional_cell[0],conventional_cell[2]))
   elif int(spacegroupNM) >= 168 and int(spacegroupNM) <= 194:
      newd12.write("%1s %19s\n"%(conventional_cell[0],conventional_cell[2]))
   elif int(spacegroupNM) >= 195 and int(spacegroupNM) <= 230:
      newd12.write("%1s\n"%(conventional_cell[0]))
      
   total_atoms = len(opt_geom_total[0])
   # write the number of atoms
   newd12.write(str(total_atoms) +"\n")
   #loop prints the new coordinates into the new d12
   for i in range(0, total_atoms):
      newd12.write("%1s  %19s  %19s  %19s\n"%(opt_geom_total[0][i], opt_geom_total[1][i], opt_geom_total[2][i], opt_geom_total[3][i]))
   #Search through parameters, skips until first end
   #Writes the first end
   for line in existingd12:
      if re.search(r"OPTGEOM", line):
         pass
      elif re.search(r"MAXCYCLE", line):
         next(existingd12)
         pass
      elif re.search(r"MAXTRADIUS", line):
         next(existingd12)
         next(existingd12)
         pass
      elif re.search(r"SUPERCEL", line):
         next(existingd12)
         next(existingd12)
         next(existingd12)
         next(existingd12)
         pass
      #Crude fix, we don't want to remove the SCF MAXCYCLE 
      elif re.search(r"EXCHSIZE", line):
         newd12.write(line)
         newd12.write(next(existingd12))  # this writes the value like "110000000\n"
         newd12.write("MAXCYCLE\n")
         newd12.write("800\n")
         pass
      elif re.search(r"SLABCUT", line):
         next(existingd12)
         next(existingd12) 
         pass
      elif re.search(r"FULLOPTG", line):
         pass
      elif re.search(r"ENDOPT", line):
         pass
      # elif re.search(r"ATOMSPIN", line):
      #    ATOMSPIN = atomspin(opt_geom_total[0])
      #    newd12.write("ATOMSPIN\n")
      #    next(existingd12)
      #    next(existingd12)
      #    pass
         # for item in ATOMSPIN:
         #    newd12.write(item+"\n")
      elif re.search(r"ENDGEOM",line):
         pass
      else:
         newd12.write(line)
   existingd12.close()
   newd12.close()


if __name__ == "__main__":
   
   #
   data_files = os.listdir((os.getcwd()))
   for file_name in data_files:
     if ".out" in file_name: #fix this
       submit_name = file_name.split(".out")[0]
       outputfile = str(submit_name)+".out"
       existingd12 = open(str(submit_name)+".d12", 'r')
       newd12  = open(str(submit_name)+"_optimized.d12",'w')

       print(outputfile)
       # get conventional unit cell parameter
       conventional_cell = conventional_cells(outputfile)
    
       #gets the coordinates of the atoms
    
       opt_geom_total = copy_coordinates(outputfile)
    
       #get spacegroup of this compound
       spacegroupNM=get_spacegroup(str(submit_name)+".d12")
    
       #creates a new d12 from the existing d12
       if spacegroupNM == "P" or spacegroupNM == "C" or spacegroupNM == "A" or spacegroupNM == "F" or spacegroupNM == "I":
          print ("The spacegroup is written in HM form, please change it!!")
       else:
          modify_d12(existingd12, newd12, opt_geom_total, conventional_cell, spacegroupNM) 
    
       existingd12.close()
       newd12.close()
