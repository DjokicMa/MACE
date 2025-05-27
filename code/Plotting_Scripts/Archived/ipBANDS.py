"""
This script plots the BANDS from CRYSTAL17 .dat and .d3 files.
Also recognizes _POTC.POTC.dat and _POTC.out files to plot bands wrt vacuum.

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
from matplotlib.ticker import ScalarFormatter,AutoMinorLocator
import matplotlib as mpl
import glob
from matplotlib import rc
import math as m
from matplotlib.cm import get_cmap
from os.path import exists

mpl.rcParams.update(mpl.rcParamsDefault)

E_l = float(sys.argv[1])
E_u = float(sys.argv[2])

def ipBANDS(material,E_l,E_u):
    file2 = material+'_BAND.BAND.dat'
    file3 = material+'_BAND.d3'
    file4 = material+'_POTC.POTC.dat'
    file5 = material+'_POTC.out'

    #If POTC file exists, open file and look electrostatic potential at inf and Efermi
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
        maxV = -(V[0]-EF)*27.2114
    else:
        maxV = 0

    ef     = 0.
    l_labels = 999
    l_label  = 999
    x_labels = []

    with open(file2) as fb:
        # Initiate variables
        header = fb.readline()
        N = int(header.split()[2])
        M = int(header.split()[4])
        E     = np.zeros(N)
        Ebeta = np.zeros(N)
        BANDS = np.zeros((N,M))
        BANDSbeta = np.zeros((N,M))
        i = 0
        ib = 0
        alpha_beta_counter = 0
        #This "if" is to correct crystal formatting when you have over 1000 elements per line
        if M > 1000:
            for l,line in enumerate(fb):
                if line.startswith("@ XAXIS TICK SPEC"): # Get the x-axis ticks
                    n_labels = int(line.split()[-1])
                    l_label  = l
                    l_labels = l+1
                if l == l_labels:
                    x_labels.append(float(line.split()[-1]))
                    l_labels +=2
                    if l_labels > l_label+2*n_labels-1:
                        l_labels=1e30
                if line.startswith("# EFERMI (HARTREE)"): # fermi energy
                    ef = float(line.split()[-1])*27.2114
                    alpha_beta_counter =+ 1
                if line.startswith("#") or line.startswith("@"):
                    continue
                if alpha_beta_counter == 0: # This is to get alpha electron bands
                    data = line.split()
                    while len(data) != M+1:
                        data = np.concatenate([data,next(fb).split()])
                    E[i] = float(data[0])
                    for j in range(0,M):
                        BANDS[i,j] = (float(data[j+1])*27.2114)+maxV
                    i = i+1
                if alpha_beta_counter == 1: # This is to get beta electron bands
                    data = line.split()
                    while len(data) != M+1:
                        data = np.concatenate([data,next(fb).split()])
                    Ebeta[ib] = float(data[0])
                    for j in range(0,M):
                        BANDSbeta[ib,j] = (float(data[j+1])*27.2114)+maxV
                    ib = ib+1
                if alpha_beta_counter == 2:
                    break
        # Same logic repeats, but this time we don't have to worry about formatting
        else:
            for l,line in enumerate(fb):
                if line.startswith("@ XAXIS TICK SPEC"):
                    n_labels = int(line.split()[-1])
                    l_label  = l
                    l_labels = l+1
                if l == l_labels:
                    x_labels.append(float(line.split()[-1]))
                    l_labels +=2
                    if l_labels > l_label+2*n_labels-1:
                        l_labels=1e30
                if line.startswith("# EFERMI (HARTREE)"):
                    ef = float(line.split()[-1])*27.2114
                    alpha_beta_counter =+ 1
                if line.startswith("#") or line.startswith("@"):
                    continue
                if alpha_beta_counter == 0:
                    data = line.split()
                    E[i] = float(data[0])
                    for j in range(0,M):
                        BANDS[i,j] = (float(data[j+1])*27.2114)+maxV
                    i = i+1
                if alpha_beta_counter == 1:
                    data = line.split()
                    Ebeta[ib] = float(data[0])
                    for j in range(0,M):
                        BANDSbeta[ib,j] = (float(data[j+1])*27.2114)+maxV
                    ib = ib+1
                if alpha_beta_counter == 2:
                    break
    # Set the x-axis labels (depends on how you set up the d3)
    labels = ['G','M','K','G']
    #labels = ['G','X','U|K','G','L','W','X']
    # Uncomment if you are plotting the general P1 symmetry path
    # if x_labels[3]-x_labels[2] < 0.01:
    #     labels = ["G","X|Y","","G|R$_2$","G","T$_2$|U$_2$","G","V$_2$|$\Gamma$","X'|Y'","","G|R$_2$'","G","T$_2$'|U$_2$'","G","V$_2$'"]
    # else:
    #     labels = ["G","X|Y","G","Z|R$_2$","G","T$_2$|U$_2$","G","V$_2$|$\Gamma$","X'|Y'","G","Z'|R$_2$'","G","T$_2$'|U$_2$'","G","V$_2$'"]

    # Replaces "G" with latex style gamma in the x-labels
    for i in range(0,len(labels)):
        if labels[i] == 'G':
            labels[i] = r"$\Gamma$"

    #Make figure (you can change fig size if necessary)
    fig = plt.figure(figsize=(5,5), dpi=100)
    ax  = fig.add_subplot(111)

    #Uncomment next two lines if you want title
    #TITLE = ("Band Structure "+material)
    #ax.set_title(TITLE, x=0.5, y=1.05, size=14)

    #Limits, plot bands, plot E_F, and plot path labels
    ax.set(ylim=(E_l+maxV,E_u+maxV),xlim=(x_labels[0],x_labels[-1]))
    ax.plot(E,BANDS,linewidth=1.8,color="#f9665e")
    ax.plot(Ebeta,BANDSbeta,linewidth=1.8,linestyle='--',color="#45b6fe")
    plt.axhline(maxV,color="black",linestyle='--',lw =1.5,alpha=1)
    for label in x_labels:
        plt.axvline(label, color ="black",lw =1,alpha=0.5)

    #If there is a POTC file, change y label to "wrt vacuum"
    if exists(file4):
        ax.set_ylabel(r"Energy w.r.t. vac.(eV)",size=18)
    else:
        ax.set_ylabel(r"$E-E_f$ (eV)",size=20)

    #ticks
    plt.xticks(x_labels[0:len(labels)],labels,size=14)
    plt.yticks(size=18)

    #These are here just for the labels
    plt.axvline(x_labels[0]-100,linewidth=1.8,color="#f9665e",label='Spin up')
    plt.axvline(x_labels[0]-100,linewidth=1.8,linestyle='--',color="#45b6fe",label='Spin down')

    # These lines set the position of the "spin up" and "spin down" legend (outside of the plot)
    # they make the plot 20% smaller in the x direction to make space for the legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.75))

    # tight layout so everything shows correctly and plot
    fig.tight_layout()
    plt.tight_layout()
    plt.show()

    # Save figure as png and svg
    fig.savefig(FIGDIR + material+'.BANDS.svg', format='svg', dpi=300)
    fig.savefig(FIGDIR + material+'.BANDS.png', format='png', dpi=300)
    plt.close('all')

#Define directories where files are, and where figures will be saved
DIR = (os.getcwd()+'/')
#FIGDIR = "/home/daniel/Dropbox/Papers/2020_2D_Genome/Successful-Calculations/Chalcogenides-Danny/BANDS/figures/"
FIGDIR = DIR

#Necessary variables to execute the for loop
pathlist = glob.glob(DIR+'*_BAND.BAND.dat')
nDIR = len(DIR)
ntype = len("_BAND.BAND.dat")

# Loops over all BANDS.d3 files in a folder
for path in pathlist:
    path_in_str = str(path)
    material = path_in_str[nDIR:-ntype]
    if material == "":break
    print(material)
    ipBANDS(material,E_l,E_u)
