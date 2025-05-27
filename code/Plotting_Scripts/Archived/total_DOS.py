"""
This script plots the total DOS from CRYSTAL .dat and .d3 files.
​Also recognizes _POTC.POTC.dat and _POTC.out files to plot DOS wrt vacuum.

Usage:
python total_DOS.py [E lower limit] [E upper limit]

Example:
python total_DOS.py -5 5
^ This plots DOS in a range of (+-)5 eV around the fermi level
"""
#LOAD EVERYTHING
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from os.path import exists
import csv

#Read the user input
E_l = float(sys.argv[1])
E_u = float(sys.argv[2])

def ipDOS(material,E_l,E_u):
    file  = material+'_DOSS.DOSS.dat'
    file1 = material+'_DOSS.d3'
    file4 = material+'_ghosts_POTC.POTC.dat'
    file5 = material+'_ghosts_POTC.out'

    #We define necessary lists and variables
    v=[]; labels=['Energy (eV)']; n=9999; num=9999; 

    #Grabs the labels from the .d3 file and puts them in a list to be used by the rest of our code
    with open(file1) as F:
        for i,line in enumerate(F):
            if line.startswith('DOSS'):
                n=i
            if i == n+1:
                l = line.split()
                num = int(l[0])
            if i>=n+2 and i<=n+2+num:
                v.append(str(line[max(line.find('#'),0):].strip()))
            if line.startswith('END'):
                break

    #We change some of the label characters for formatting purposes
    for i in v:
        j = i.replace(' all','  ')
        j = j.replace(' S','_s')
        j = j.replace(' P','_p')
        j = j.replace(' D','_d')
        j = j.replace(' F','_f')
        j = j.replace('END','Total DOS')
        l = j.replace('#','')
        if len(l)>1 and l[0].isupper() and l[1].isupper():
            j = list(l)
            j[1]=j[1].lower()
            l = ''.join(j)
        labels.append(l)

    # User input for projections
    search = []

    #If POTC file exists, open file and look for electrostatic potential at inf and Efermi
    if exists(file4) and exists(file5):
        z = []; V  = []; 
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

    #Defines an empty data vector to put our data inside it. Also defines an empty variable for the fermi energy value
    data_vect=[[] for n in range(len(labels))]
    ef=0

    #Loops through the data and puts it in data_vect for plotting, also finds the fermi energy and puts it in ef
    #with open(new_file) as f:
    with open(file) as f:
        if len(labels) >= 17:
            for i,line in enumerate(f):
                if line.startswith("# EFERMI (HARTREE)"):
                    ef = float(line.split()[-1])*27.2114
                if line.startswith("#") or line.startswith("@"):
                    continue
                if line.startswith("&"):
                    continue#break
                data = line.split()
                ne = next(f)
                ne = ne.split()
                if len(data) >= 17 and len(ne) < 17:
                    data = np.concatenate([data,ne])
                data_vect[0].append(float(data[0])*27.2114+maxV)
                for j in range(1,len(labels)):
                    data_vect[j].append(float(data[j]))
        else:
            for i,line in enumerate(f):
                if line.startswith("# EFERMI (HARTREE)"):
                    ef = float(line.split()[-1])*27.2114
                if line.startswith("#") or line.startswith("@"):
                    continue
                if line.startswith("&"):
                    continue#break
                data = line.split()
                data_vect[0].append(float(data[0])*27.2114+maxV)
                for j in range(1,len(labels)):
                    data_vect[j].append(float(data[j]))

    #Grab data only in the range we want to plot
    plot_vect = [[] for n in range(len(labels))]
    for i in range(1,len(plot_vect)):
        for j in range(0,len(data_vect[0])):
            if data_vect[0][j] >= E_l+maxV and data_vect[0][j] <= E_u+maxV:
                plot_vect[i].append(data_vect[i][j])

    for j in range(0,len(data_vect[0])):
        if data_vect[0][j] >= E_l+maxV and data_vect[0][j] <= E_u+maxV:
            plot_vect[0].append(data_vect[0][j])
    
    E = []
    DOS = []
    for i in range(len(plot_vect[0])):
        E.append(plot_vect[0][i])
        DOS.append(plot_vect[-1][i])
    E_a = E[:int(len(E)/2)]
    #E_b = E[int(len(E)/2):]
    DOS_a = DOS[:int(len(E)/2)]
    DOS_b = DOS[int(len(E)/2):]
    total_DOS = [abs(DOS_a[i])+np.abs(DOS_b[i]) for i in range(len(DOS_a))]
    with open('DOS_'+material+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Energy (eV)','Total DOS (a.u.)'])
        for i in range(len(E_a)):
            writer.writerow([E_a[i],total_DOS[i]])
   
    #Define figure and axes
    fig = plt.figure(figsize=(4,5))
    ax  = fig.add_subplot(111)

    #Title, xlabel and ylabel
    ax.set_xlabel("DOS",size=18)
    if exists(file4):
        ax.set_ylabel(r"Energy w.r.t. vac. (eV)",size=18)
    else:
        ax.set_ylabel(r"$E-E_f$ (eV)",size=20)

    #set x,y limits and remove ticks for the x-axis (they're arbitrary anyway)
    xlimit = np.max(np.max(total_DOS))
    ax.set(xlim=(0,xlimit),ylim=(E_l+maxV,E_u+maxV))
    ax.set_xticks([])
    plt.yticks(size=18)

    #plot
    ax.fill_betweenx(E_a,total_DOS,color='darkgrey')
    ax.plot(total_DOS,E_a,color='black',alpha=0.3,linewidth=1)
    plt.axhline(maxV,color="black",linestyle='--',lw =1.5,alpha=1)
    plt.axvline(0,color='black',lw=1.5)

    #tight layout & show the plot
    fig.tight_layout()
    plt.tight_layout()
    #plt.show()

    #Save the plot
    fig.savefig(FIGDIR + material+'_total_DOS.png', format='png', dpi=300)


#This is the directory where the files will be saved. Make sure to change this accordingly.
DIR = (os.getcwd()+'/')
FIGDIR = DIR
#Necessary variables to execute the for loop
pathlist = glob.glob(DIR+'*_DOSS.DOSS.dat')
nDIR = len(DIR)
ntype = len("_DOSS.DOSS.dat")

for path in pathlist:
    path_in_str = str(path)
    material = path_in_str[nDIR:-ntype]
    if material == "":break
    print(material)
    ipDOS(material,E_l,E_u)
