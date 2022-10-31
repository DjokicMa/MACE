#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 14:17:47 2022

@author: marcus
"""
import glob
import os

HeV    = 27.2114
# converts Ha to eV

#################################################################################

#determines the minimum of the cond band in eV and top of valence band in eV
def get_bands(output_content):
  index_alpha = 0
  index_beta = 0
  index_direct = 0
  index_cond = 0 
  alpha_cond_band =[]
  alpha_val_band =[]
  beta_cond_band = []
  beta_val_band = []
  for line in output_content:
    if "ALPHA      ELECTRONS" in line:
      index_alpha = output_content.index(line)
    if "BETA       ELECTRONS" in line:
      index_beta = output_content.index(line)
    if line.startswith(" DIRECT ENERGY BAND GAP"):
      index_direct = output_content.index(line)
    if "POSSIBLY CONDUCTING STATE" in line:
      index_cond = output_content.index(line)
      
  if index_direct > index_alpha and index_direct > index_beta and index_direct > index_cond:
    for index in range(index_direct-4,index_direct+1):
      if "TOP OF VALENCE" in output_content[index]:
        unclean_alpha = [x for x in output_content[index].split(" ") if  x != "" ]
        alpha_cond_band.append(unclean_alpha[10].split(';')[0])
      if "BOTTOM OF VIRTUAL" in output_content[index]:
        unclean_alpha = [x for x in output_content[index].split(" ") if  x != "" ]
        alpha_val_band.append(unclean_alpha[10].split(';')[0])
    beta_cond_band = alpha_cond_band
    beta_val_band = alpha_val_band
  elif index_cond > index_alpha and index_cond > index_beta and index_cond > index_direct:
    unclean_alpha = [x for x in output_content[index_cond].split(" ") if  x != "" ]
    alpha_val_band.append(unclean_alpha[5].split(';')[0])
    alpha_cond_band = alpha_val_band
    beta_cond_band = alpha_cond_band
    beta_val_band = alpha_val_band
  else:
    for index in range(index_alpha+2,index_alpha+7):
      if "TOP OF VALENCE" in output_content[index]:
        unclean_alpha = [x for x in output_content[index].split(" ") if  x != "" ]
        alpha_cond_band.append(unclean_alpha[10].split(';')[0])
      if "BOTTOM OF VIRTUAL" in output_content[index]:
        unclean_alpha = [x for x in output_content[index].split(" ") if  x != "" ]
        alpha_val_band.append(unclean_alpha[10].split(';')[0])
    for index in range(index_beta+2,index_beta+7):
      if "TOP OF VALENCE" in output_content[index]:
        unclean_beta = [x for x in output_content[index].split(" ") if  x != "" ]
        beta_cond_band.append(unclean_beta[10].split(';')[0])
      if "BOTTOM OF VIRTUAL" in output_content[index]:
        unclean_beta = [x for x in output_content[index].split(" ") if  x != "" ]
        beta_val_band.append(unclean_beta[10].split(';')[0])

  max_alpha_c = alpha_cond_band[0]
  min_alpha_v = alpha_val_band[0]
  max_beta_c = beta_cond_band[0]
  min_beta_v = beta_val_band[0]
  if len(alpha_cond_band) == 1:
    max_alpha_c = alpha_cond_band[0]
  else:
    for shell in alpha_cond_band:
      if float(shell) > float(max_alpha_c):
        max_alpha_c = shell
        
  if len(alpha_val_band) == 1:
    min_alpha_v = alpha_val_band[0]
  else:
    for shell in alpha_val_band:
      if float(shell) < float(min_alpha_v):
        min_alpha_v = shell

  if len(beta_cond_band) == 1:
    max_beta_c = beta_cond_band[0]
  else:
    for shell in beta_cond_band:
      if float(shell) > float(max_beta_c):
        max_beta_c = shell

  if len(beta_val_band) == 1:
    min_beta_v = beta_val_band[0]
  else:
    for shell in beta_val_band:
      if float(shell) < float(min_beta_v):
        min_beta_v = shell
  
  if float(max_alpha_c) > float(max_beta_c):
    max_c = max_beta_c
  else:
    max_c = max_alpha_c
  if float(min_alpha_v) > float(min_beta_v):
    min_v = min_alpha_v
  else:
    min_v = min_beta_v
  return(str(float(max_c)*HeV),str(float(min_v)*HeV))

#################################################################################

# writes the Electronic Transport D3 for PProperty 
def write_d3(d3_file, val, cond):
    
    d3_file.write("NEWK\n")
    d3_file.write("48 48\n")
    d3_file.write("1 0\n")	 	
    d3_file.write("BOLTZTRA\n")
    d3_file.write("TDFRANGE\n")
    d3_file.write(str(val) + " " + str(cond) + " 0.001\n") # integration resolution
    d3_file.write("MURANGE\n")	
    d3_file.write(str(val) + " " + str(cond) + " 0.01\n") # plot resolution
    d3_file.write("TRANGE\n")	
    d3_file.write("300. 700. 200.\n")
    d3_file.write("RELAXTIM\n")	
    d3_file.write("22\n")	
    d3_file.write("END\n")		 	
    d3_file.write("END\n")	

#################################################################################

DIR = (os.getcwd()+'/')
#Necessary variables to execute the for loop
pathlist = glob.glob(DIR+'*.out')
nDIR = len(DIR)
ntype = len(".out")

for path in pathlist:
    path_in_str = str(path)
    material = path_in_str[nDIR:-ntype]
    if material == "":break
    output_content = []
    
    print(material)
    with open(path, 'r') as f:
       for line in f:
          if "SCF ENDED" in line:
             break
          else:
             output_content.append(line)
             
    max_c, min_v  = get_bands(output_content)
    print("  Bottom of Valence Band: " + min_v + " eV\n  Top of Conduction Band: " + max_c + " eV")
    cond = round(float(max_c),1) + 1
    val = round(float(min_v),1) - 1
    # murange needs to be 0.5 to 1 eV higher than cond and lower than valence band
    
    pathname = path_in_str[0:nDIR]
    f9_file_name = pathname + material + ".f9"
    new_f9 = pathname + material + "_TRANSPORT.f9"
    #makes new f9 for the PProperty calculation
    os.popen(f"cp {f9_file_name} {new_f9}")
    d3_file_name = pathname + material + "_TRANSPORT.d3"
    with open(d3_file_name, 'w+') as d3_file:
      write_d3(d3_file, val, cond)
