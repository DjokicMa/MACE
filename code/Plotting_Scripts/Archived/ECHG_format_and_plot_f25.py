# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import math
import os
import glob
import itertools
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time

def getdataf25(fort25):
    data = []
    with open(fort25) as f:
        for i,line in enumerate(f):
            if i == 0:
                nx = int(line.split()[1])
                ny = int(line.split()[2])
                nlines = math.ceil(nx*ny/6)
            if i > 2 and i <= nlines+2:
                data.append(line.split())
    data_vect = list(itertools.chain.from_iterable(data))
    data_vect = [float(x) for x in data_vect]
    #max = np.max(data_vect)
    mean = np.mean(data_vect)
    std = np.std(data_vect)
    min = mean-std
    max = mean+std
    return(data_vect,nx,ny,max,min)

def formatf25(material,data_vect,nx):
    data_matrix = [data_vect[i:i+nx] for i in range(0,len(data_vect),nx)]
    data_matrix = np.matrix(data_matrix)
    with open(material+'_matrix.txt','w+') as f:
        for line in data_matrix:
            np.savetxt(f, line, fmt='%.10f')
    return(data_matrix)

def plot_f25(data, vmin=None, vmax=None, dpi=200, save=False, save_name=None):
    fig, ax = plt.subplots(1, dpi=dpi)
    im = ax.imshow(np.asarray(data),vmin=vmin, vmax=vmax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(save_name)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label=r'Electron/Bohr$^3$')
    #plt.show()
    
    if save_name is None:        
        save_name = time.strftime("%Y%m%d-%H%M%S")
    
    #fig.savefig(save_name + ".pdf", dpi=dpi,format='pdf')
    fig.savefig(save_name + ".png", dpi=dpi,format='png')
    
DIR = (os.getcwd()+'/')
pathlist = glob.glob(DIR+'*.f25')
nDIR = len(DIR)
ntype = len('.f25')

for path in pathlist:
    material = str(path[nDIR:-ntype])
    print(material)
    data_vect,nx,ny,max,min = getdataf25(str(path))
    data_matrix = formatf25(material,data_vect,nx)
    plot_f25(data_matrix, vmin=min, vmax=max, dpi=200, save=False, save_name=material)
