"""
This script plots the DOS from CRYSTAL17 .dat and .d3 files.
​Also recognizes _POTC.POTC.dat and _POTC.out files to plot DOS wrt vacuum.

Usage:
python ipDOS.py [E lower limit] [E upper limit]

Example:
python ipDOS.py -5 5
^ This plots DOS in a range of (+-)5 eV around the fermi level

After this, the program will prompt you to plot any projections
"""
#LOAD EVERYTHING
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from os.path import exists

#Read the user input
E_l = float(sys.argv[1])
E_u = float(sys.argv[2])

def ipDOS(material,E_l,E_u):
    file  = material+'_DOSS.DOSS.dat'
    file1 = material+'_DOSS.d3'
    file4 = material+'_POTC.POTC.dat'
    file5 = material+'_POTC.out'

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
        j = i.replace(' all','_all')
        j = j.replace(' S','_s')
        j = j.replace(' P','_p')
        j = j.replace(' D','_d')
        j = j.replace(' F','_f')
        j = j.replace('END','Total DOS')
        l = j.replace('#','')
        if l[0].isupper() and l[1].isupper():
            j = list(l)
            j[1]=j[1].lower()
            l = ''.join(j)
        labels.append(l)

    # User input for projections
    search = []
    print('Available projections:')
    print(labels[1:len(labels)-1])
    num = input("Enter how many projections you want to plot: ")
    print('Enter string for projection: ')
    for i in range(int(num)):
        n = input("Projection "+str(i+1)+': ')
        search.append(str(n))

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
    data_vect=[[] for n in range(len(labels))]; ef=0;

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

    #Different color pallets
    color_pal = ['#ff0000','#000080','#006400','#8b4513','#00ced1','#ffa500','#2f4f4f','#ffff00','#00ff00','#0000ff','#d8bfd8','#ff00ff','#1e90ff','#ff1493','#98fb98']
    color_spinel = ['slategrey','peru','red'] #Zn, Fe, O

    #Define figure and axes
    fig = plt.figure(figsize=(4.5,8),dpi=100)
    ax  = fig.add_subplot(111)

    #Title, xlabel and ylabel
    TITLE = ("Density of States "+str(sys.argv[1]))
    #ax.set_title(TITLE)
    ax.set_xlabel("DOS",size=18)
    if exists(file4):
        ax.set_ylabel(r"Energy w.r.t. vac. (ev)",size=18)
    else:
        ax.set_ylabel(r"$E-E_f$ (ev)",size=20)

    #set x,y limits and remove ticks for the x-axis (they're arbitrary anyway)
    #ax.set(xlim=(1.1*np.min(plot_vect[-1]),1.1*np.max(plot_vect[-1])),ylim=(E_l+maxV,E_u+maxV))
    xlimit = np.max([np.abs(1.1*np.min(plot_vect[-1])),np.abs(1.1*np.max(plot_vect[-1]))])
    ax.set(xlim=(-xlimit,xlimit),ylim=(E_l+maxV,E_u+maxV))
    ax.set_xticks([])
    plt.yticks(size=18)
    ax.set_prop_cycle(color=color_pal)

    #for loop to plot all the projections
    for i in range(1,len(data_vect)):
        if (labels[i].find("Total") == -1):
            continue
        else:
            ax.fill_betweenx(data_vect[0],data_vect[i],label=labels[i],alpha=0.8,color='gainsboro')
            ax.plot(data_vect[i],data_vect[0],color='gainsboro',linewidth=1.5)

    for n in range(len(search)):
        for i in range(1,len(data_vect)):
            if (labels[i].find(search[n]) == -1):
                continue
            else:
                #ax.fill_betweenx(data_vect[0],data_vect[i],label=labels[i], alpha=0.8)
                ax.plot(data_vect[i],data_vect[0],label=labels[i],linewidth=1.35,alpha=0.7)

    plt.axhline(maxV,color="black",linestyle='--',lw =1.5,alpha=1)
    plt.axvline(0,color='black',lw=1.5)

    #Legend specs
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    legend=ax.legend(loc='center left', bbox_to_anchor=(1, 0.25))
    #legend.get_frame().set_facecolor('White')
    #legend.get_frame().set_edgecolor('None')

    #tight layout & show the plot
    fig.tight_layout()
    plt.tight_layout()
    plt.show()
    plt.close('all')

    #Save the plot
    fig.savefig(FIGDIR + material+'.DOSS.svg', format='svg', dpi=300)
    fig.savefig(FIGDIR + material+'.DOSS.png', format='png', dpi=300)


#This is the directory where the files will be saved. Make sure to change this accordingly.
DIR = (os.getcwd()+'/')
FIGDIR = DIR
#FIGDIR = "/home/daniel/Dropbox/Papers/2020_2D_Genome/Successful-Calculations/Chalcogenides-Danny/DOSS/figures/"
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
