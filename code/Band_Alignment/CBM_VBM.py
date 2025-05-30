"""
This script parses CRYSTAL output files to extract band gap data, VBM/CBM values (for alpha and beta spins),
and generates a CSV summary (CBM_VBM.csv) containing these electronic properties.
"""


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

DIR      = os.getcwd()+'/'
pathlist = glob.glob(DIR+'*.out')
nDIR     = len(DIR)
ntype    = len(".out")
DATA     = [[],[],[]]
INDGAP = " INDIRECT ENERGY BAND GAP:"
DIRGAP = " DIRECT ENERGY BAND GAP:"
ALPHA  = "    ALPHA      ELECTRONS"
BETA   = "    BETA       ELECTRONS"
VBM    = " TOP OF VALENCE BANDS"
COND   = " POSSIBLY CONDUCTING STATE"
EF     = ""
HeV    = 27.2114
#################################################################################
with open("CBM_VBM.csv","w") as out:
  writer = csv.writer(out)
  writer.writerow(["Material","Eg_alpha (eV)","Type","Eg_beta (eV)","Type","VBM_alpha (eV)","CBM_alpha (eV)","VBM_beta (eV)","CBM_beta (eV)","Total VBM (eV)","Total CBM (eV)","Total Type (eV)"])

  for path in pathlist:
      path_in_str = str(path)
      material = path_in_str[nDIR:-ntype]
      if material == "":break
      SLAB     = material+".out"
      if os.path.exists(DIR+SLAB):
        print(material)

        with open(DIR+SLAB,'r') as slb:
          #:::::::::::::::   SLAB ANALYSIS   ::::::::::::::::::::::
          state_slb     = "Null"
          Ebg_slb_alpha = 999999
          Ebg_slb_beta  = 999999
          GAPTYPE_slb_alpha   = "Null"
          GAPTYPE_slb_beta   = "Null"
          #alpha_counter = 0
          #beta_counter = 0
          VBM_alpha =    99999999
          CBM_alpha_eV = 99999999
          VBM_beta =     99999999
          CBM_beta_eV =  99999999
          VBM_alpha_eV = 99999999
          VBM_beta_eV =  99999999
          alpha_line =   99999999
          beta_line =    99999999

          for i, line in enumerate(slb):

            # BANDGAP AND BAND EDGES
            if line.startswith(ALPHA):
              alpha_line = i
              alpha_counter = 0

            if line.startswith(BETA):
              beta_line = i
              beta_counter = 0

          slb.seek(0) #this is important, returns to beginning of file
          for i, line in enumerate(slb):
            if line.startswith(COND):
              Ebg_slb_alpha   = 0.
              Ebg_slb_beta    = 0.
              GAPTYPE_slb_beta = "COND"
              GAPTYPE_slb_alpha = "COND"
              VBM_alpha_eV = 0
              VBM_beta_eV = 0
              CBM_alpha_eV = 0
              CBM_beta_eV = 0
              total_VBM = 9999999
              total_CBM = 9999999

            if i > alpha_line and i < beta_line:
              if line.startswith(INDGAP) or line.startswith(DIRGAP):
                Ebg_slb_alpha = float(line.split()[-2])
                if line.startswith(INDGAP):
                  GAPTYPE_slb_alpha = "INDIRECT"
                if line.startswith(DIRGAP):
                  GAPTYPE_slb_alpha = "DIRECT"
              if line.startswith(VBM) and alpha_counter==0:
                alpha_counter += 1
                VBM_alpha = float(line.split()[-2])
              if line.startswith(VBM) and alpha_counter > 0 and VBM_alpha < float(line.split()[-2]):
                alpha_counter += 1
                VBM_alpha = float(line.split()[-2])
              VBM_alpha_eV = VBM_alpha*HeV
              CBM_alpha_eV = (VBM_alpha*HeV)+Ebg_slb_alpha

            if i > beta_line:
              if line.startswith(INDGAP) or line.startswith(DIRGAP):
                Ebg_slb_beta = float(line.split()[-2])
                if line.startswith(INDGAP):
                  GAPTYPE_slb_beta = "INDIRECT"
                if line.startswith(DIRGAP):
                  GAPTYPE_slb_beta = "DIRECT"
              if line.startswith(VBM) and beta_counter==0:
                beta_counter += 1
                VBM_beta = float(line.split()[-2])
              if line.startswith(VBM) and beta_counter > 0 and VBM_beta < float(line.split()[-2]):
                beta_counter += 1
                VBM_beta = float(line.split()[-2])
              VBM_beta_eV = VBM_beta*HeV
              CBM_beta_eV = (VBM_beta*HeV)+Ebg_slb_beta



            if Ebg_slb_alpha > 100 or Ebg_slb_beta > 100: continue
            
            if Ebg_slb_alpha < Ebg_slb_beta:
              total_Ebg = Ebg_slb_alpha
              total_VBM = VBM_alpha_eV
              total_CBM = CBM_alpha_eV
              total_type = GAPTYPE_slb_alpha

            if Ebg_slb_alpha == Ebg_slb_beta:
              total_Ebg = Ebg_slb_alpha
              total_VBM = VBM_alpha_eV
              total_CBM = CBM_alpha_eV
              total_type = GAPTYPE_slb_alpha

            if Ebg_slb_beta < Ebg_slb_alpha:
              total_Ebg = Ebg_slb_beta
              total_VBM = VBM_beta_eV
              total_CBM = CBM_beta_eV
              total_type = GAPTYPE_slb_beta

          table = [material,Ebg_slb_alpha,GAPTYPE_slb_alpha,Ebg_slb_beta,GAPTYPE_slb_beta,VBM_alpha_eV,CBM_alpha_eV,VBM_beta_eV,CBM_beta_eV,total_VBM,total_CBM,total_type]
          writer.writerow(table)
      else: continue
out.close()
