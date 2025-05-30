"""
This script extracts work function and electrostatic potential data from CRYSTAL POTC output files,
computes relevant values (e.g., WF max/min, EPOT top/bot/avg, EF), and saves the results into WF.csv.
"""

import os
import sys
import glob
import csv
from csv import writer
from os.path import exists
import numpy as np

def getWF(material):
    file4 = material+'_POTC.POTC.dat'
    file5 = material+'_POTC.out'

    if exists(file4) and exists(file5):
        z = []; V  = [];
        EF=0
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
        
        Vtop = V[0]*27.2114
        Vbot = V[-1]*27.2114
        Vmax = np.max([Vtop,Vbot])
        Vmin = np.min([Vtop,Vbot])
        Vavg = (Vtop+Vbot)/2
        WF0 = (V[0]-EF)*27.2114
        WF1 = (V[-1]-EF)*27.2114
        WFmax = np.max([WF0,WF1])
        WFmin = np.min([WF0,WF1])
        EF_eV = EF*27.2114

        list_data = [material,Vtop,Vbot,WF0,WF1,WFmax,WFmin,Vmax,Vmin,Vavg,EF_eV]
        writer.writerow(list_data)

with open("WF.csv","w") as out:
    writer = csv.writer(out)
    writer.writerow(["Material","EPOT top (eV)","EPOT bot (eV)","WF top (eV)", "WF bot (eV)","WFmax (eV)","WFmin (eV)","EPOTmax (eV)","EPOTmin (eV)","EPOTavg (eV)","EFermi (eV)"])
    
    DIR = (os.getcwd()+'/')
    pathlist = glob.glob(DIR+'*_POTC.POTC.dat')
    nDIR = len(DIR)
    ntype = len("_POTC.POTC.dat")

    for path in pathlist:
        path_in_str = str(path)
        material = path_in_str[nDIR:-ntype]
        if material == "":break
        print('Material '+material)
        getWF(material)
out.close()
