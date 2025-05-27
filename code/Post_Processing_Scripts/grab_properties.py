from pathlib import Path
import warnings
import numpy as np
import glob
import math
import matplotlib
matplotlib.use('tkagg') # <-- THIS MAKES IT FAST!
import matplotlib.pyplot as plt
import csv
import os


def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

DIR      = "/home/daniel/Desktop/Work/2DMat/get_properties_singlepoint/out/"
pathlist = glob.glob(DIR+'*_slab.out')
nDIR     = len(DIR)
ntype    = len("_bulk.out")
DATA     = [[],[],[]]
#FIN    = " FINAL OPTIMIZED GEOMETRY"                   # switch on
LAT    = " LATTICE PARAMETERS  (ANGSTROMS AND DEGREES) - PRIMITIVE CELL"
CART   = " CARTESIAN COORDINATES - PRIMITIVE CELL"
ENERGY = " * OPT END - CONVERGED * E(AU):"
INDGAP = " INDIRECT ENERGY BAND GAP:"
DIRGAP = " DIRECT ENERGY BAND GAP:"
HeV    = 27.2114
#################################################################################

with open("HSE06_PROPERTIES.csv","w") as out:
  writer = csv.writer(out)
  #               table = [material,Ebg_slb,GAPTYPE_slb,Ebg_blk,GAPTYPE_blk,E_slb,E_blk,Eb,IAD_slb,IAD_blk,state_slb,state_blk,dens,mass,atoms,space]
  writer.writerow(["Material","Eg (slab)","Type","Eg (bulk)","Type","E (slab)","E (bulk)","Eb","IAD (slab)","IAD (bulk)","State (slab)","State (bulk)","Dens (bulk)","Mass","Atoms","Vacuum"])
  for path in pathlist:
            # because path is object not string
      path_in_str = str(path)
      material = path_in_str[nDIR:-ntype]
      if material == "":break
      #print(material)

      FERMI    = "   FERMI ENERGY:"
      DONE     = " * OPT END - CONVERGED"
      GEOM     = " FINAL OPTIMIZED GEOMETRY"
      COND     = " POSSIBLY CONDUCTING STATE"

      SLAB     = material+"_slab.out"
      BULK     = material+"_bulk.out"

      if os.path.exists(DIR+SLAB) and os.path.exists(DIR+BULK):
        print(material)
        with open(DIR+SLAB,'r') as slb, open(DIR+BULK,'r') as blk:
          #:::::::::::::::   SLAB ANALYSIS   ::::::::::::::::::::::
          el = np.zeros(99)
          x  = np.zeros(99)
          y  = np.zeros(99)
          z  = np.zeros(99)

          id  = 9999
          md  = 9999
          atoms    = 99
          k        = 0
          switch   = 0
          E_slb    = 999999
          IAD_slb  = 999999
          state_slb    = "Null"
          Ebg_slb      = 999999
          GAPTYPE_slb  = "Null"
          for i, line in enumerate(slb):
            # GET CARTESIAN POSTIONS FOR ATOMS
            if line.startswith(" NUMBER OF IRREDUCIBLE"):
              atoms = int(line.split()[-1])
            if line.startswith(CART):
              md    = i 
            if " PROCESS" in line: 
              md+=1
              continue
            if md+4 <= i and i <= md+atoms+3:
               el[i-md-5] = float(line.split()[-5])
               if el[i-md-5] >200:el[i-md-11] -=200
               x[i-md-5] = float(line.split()[-3])
               y[i-md-5] = float(line.split()[-2])
               z[i-md-5] = float(line.split()[-1])

            # GET THE INFORMATION ON THE PRIMITIVE CELL (ANG. and DEG.)
            if LAT in line:
              id = i  
            if i == id+2:
                # FIND THE OPTIMIZED PRIMITIVE CELL
                cell    = line.split()
                A       = float(cell[0])
                B       = float(cell[1])
                C       = float(cell[2])
                alpha   = float(cell[3])
                beta    = float(cell[4])
                gamma   = float(cell[5])
                V       = float(cell[6])

            if line.startswith(COND):
              state_slb = "COND"
              Ebg_slb   = 0.


            # TOTAL ENERGY OF THE SLAB
            if "ETOT(AU)" in line:
              E_slb = float(line.split()[3])
              #print("ETOT(AU)",E_slb)


            # BANDGAP FOR SLAB
            if line.startswith(INDGAP) or line.startswith(DIRGAP):
              Ebg_slb = float(line.split()[-2])
              if line.startswith(INDGAP):
                GAPTYPE_slb = "INDIRECT"
              if line.startswith(DIRGAP):
                GAPTYPE_slb = "DIRECT"
              #print(line)

            if Ebg_slb <9. and Ebg_slb >0.:     state_slb = "SEMI"
            if Ebg_slb >9. and Ebg_slb < 10000: state_slb = "INSU"
            if Ebg_slb >100: continue
            N   = float(nCr(atoms,2))
            d        = 0.
            for i in range(0,atoms):
               for j in range(i,atoms):
                  d += np.sqrt((x[i]-x[j])**2.+(y[i]-y[j])**2.+(z[i]-z[j])**2)
            IAD_slb  = d/N
            mass     = sum(el)
            el = np.zeros(20)


          #:::::::::::::::   BULK ANALYSIS   ::::::::::::::::::::::
          el = np.zeros(99)
          x  = np.zeros(99)
          y  = np.zeros(99)
          z  = np.zeros(99)
          z_c  = np.zeros(99)
          z_new = np.zeros(99)

          id  = 9999
          md  = 9999
          g = 9999         
          atoms    = 99
          d        = 0.
          k        = 0
          switch   = 0
          E_blk    = 999999
          IAD_blk  = 999999
          state_blk    = "Null"
          Ebg_blk      = 9999999
          GAPTYPE_blk  = "Null"
          for i, line in enumerate(blk):
            # GET CARTESIAN POSTIONS FOR ATOMS
            if line.startswith(" NUMBER OF IRREDUCIBLE"):
              atoms = int(line.split()[-1])
            if line.startswith(CART):
              md    = i 
            if " PROCESS" in line: 
              md+=1
              continue
            if md+4 <= i and i <= md+atoms+3:
               el[i-md-5] = float(line.split()[-5])
               if el[i-md-5] >200:el[i-md-11] -=200
               x[i-md-5] = float(line.split()[-3])
               y[i-md-5] = float(line.split()[-2])
               z[i-md-5] = float(line.split()[-1])
            # GET THE INFORMATION ON THE PRIMITIVE CELL (ANG. and DEG.)
            if LAT in line:
              id = i  
            if i == id+2:
                # FIND THE OPTIMIZED PRIMITIVE CELL
                cell    = line.split()
                A       = float(cell[0])
                B       = float(cell[1])
                C       = float(cell[2])
                alpha   = float(cell[3])
                beta    = float(cell[4])
                gamma   = float(cell[5])
                V       = float(cell[6])
                ang_alpha = math.radians(alpha)
                ang_beta = math.radians(beta)
                ang_gamma = math.radians(gamma)

            if line.startswith(" ATOMS IN THE ASYMMETRIC UNIT"):
                g = i
            if i>=g+3 and i<=g+3+atoms-1:
                z_c[i-g-3] = float(line.split()[6])
                if z[i-g-3] < 0.:
                    z_new[i-g-3] = z[i-g-3] + 1.0
                else:
                    z_new[i-g-3] = z[i-g-3]

            if line.startswith(COND):
              state_blk = "COND"
              Ebg_blk   = 0.


            # TOTAL ENERGY OF THE BULK
            if "ETOT(AU)" in line:
              E_blk = float(line.split()[3])
              #print("ETOT(AU)",E_blk)


            # BANDGAP FOR BULK
            if line.startswith(INDGAP) or line.startswith(DIRGAP):
              Ebg_blk = float(line.split()[-2])
              if line.startswith(INDGAP):
                GAPTYPE_blk = "INDIRECT"
              if line.startswith(DIRGAP):
                GAPTYPE_blk = "DIRECT"
              #print(line)

          if Ebg_blk <9. and Ebg_blk >0.:     state_blk = "SEMI"
          if Ebg_blk >9. and Ebg_blk < 10000: state_blk = "INSU"
          if Ebg_blk >100: continue
          N   = float(nCr(atoms,2))
          d   = 0.
          for i in range(0,atoms):
             for j in range(i,atoms):
                d += np.sqrt((x[i]-x[j])**2.+(y[i]-y[j])**2.+(z[i]-z[j])**2)
          IAD_blk  = d/N
          mass     = sum(el)

          if (1+min(z_c)-max(z_c)) > 0.001:
            space = ((1+min(z_c))-max(z_c))*(C/math.sin(ang_gamma))*math.sqrt(1-math.cos(ang_alpha)**2-math.cos(ang_beta)**2-math.cos(ang_gamma)**2+2*math.cos(ang_alpha)*math.cos(ang_beta)*math.cos(ang_gamma))
          elif (1+min(z_c)-max(z_c)) < 0.001:
            space = ((1-max(z_new)+min(z_new)))*(C/math.sin(ang_gamma))*math.sqrt(1-math.cos(ang_alpha)**2-math.cos(ang_beta)**2-math.cos(ang_gamma)**2+2*math.cos(ang_alpha)*math.cos(ang_beta)*math.cos(ang_gamma))
          Eb     = (E_blk-E_slb)*HeV
          dens   = float(mass)/V
          table = [material,Ebg_slb,GAPTYPE_slb,Ebg_blk,GAPTYPE_blk,E_slb,E_blk,Eb,IAD_slb,IAD_blk,state_slb,state_blk,dens,mass,atoms,space]
          writer.writerow(table)
      else: continue
out.close()
