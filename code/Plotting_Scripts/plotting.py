#!/usr/bin/env python3
# Author: Kevin Lucht
# Editor: William Comaskey 03/23/2022
import os, sys, math
import re
import linecache
import shutil
import itertools

def sym(input_lines, output_lines):
  if(input_lines[1]=="CRYSTAL"): sym_num = int(input_lines[3])
  elif(input_lines[1]=="SLAB"):  sym_num = int(input_lines[2])
  else:
    print("ERROR! - UNSUPPORTED INPUT TYPE")
    sym_num  = 1
  sym_name = ""
  orbital_num = 0
  for line in output_lines:
    if "SPACE GROUP" in line:
      #print(line)
      clean_line = []
      split_line = line.split(" ")
      for string in split_line:
        if string != "":
          clean_line.append(string)
      sym_name = clean_line[4]
    if "NUMBER OF AO" in line:
      clean_line = []
      split_line = line.split(" ")
      for string in split_line:
        if string != "":
          clean_line.append(string)
      orbital_num = int(clean_line[3])
      break
  return(sym_num, sym_name, orbital_num)

def band_d3(sym_num, sym_name, orbital_num, d3_file, output_name,kpts=2000):
  d3_file.write("BAND\n")
  d3_file.write(output_name + "\n")
  d3_lines = []
  if sym_num == 1:
    d3_file.write("9 16 {} 1 ".format(kpts) + str(orbital_num) + " 1 0\n")
    d3_file.write("8 8 -8  0 8 0\n") #M Y
    d3_file.write("0 8 0   0 0 0\n") #Y G
    d3_file.write("0 0 0   0 0 8\n") #G Z
    d3_file.write("0 0 8   4 4 4\n") #Z N
    d3_file.write("4 4 4   8 8 8\n") #N R
    d3_file.write("8 8 8   0 0 0\n") #R G
    d3_file.write("0 0 0   8 0 0\n") #G X
    d3_file.write("8 0 0   8 0 8\n") #X L
    d3_file.write("8 0 8   0 0 0\n") #L G
    d3_file.write("END")
  elif sym_num == 2:
      with open("library/d3_input/Triclinic.d3", 'r+') as dat_file:
          d3_file.write(str(len(d3_lines)-1) + " 0 {} 1 ".format(kpts) + str(orbital_num) + " 1 0\n")
          for line in dat_file.readlines():
            d3_file.write(line)
  elif sym_num >= 3 and sym_num < 16:
    if sym_name == "P":
      with open("library/d3_input/Monoclinic_Simple.d3", 'r+') as dat_file:
          d3_file.write(str(len(d3_lines)-1) + " 0 {} 1 ".format(kpts) + str(orbital_num) + " 1 0\n")
          for line in dat_file.readlines():
            d3_file.write(line)
    if sym_name == "C":
      with open("library/d3_input/Monoclinic_AC.d3", 'r+') as dat_file:
          d3_file.write(str(len(d3_lines)-1) + " 0 {} 1 ".format(kpts) + str(orbital_num) + " 1 0\n")
          for line in dat_file.readlines():
            d3_file.write(line)
  elif sym_num >= 16 and sym_num < 75:
    if sym_name == "P":
      with open("library/d3_input/Orthorombic_Simple.d3", 'r+') as dat_file:
          d3_file.write(str(len(d3_lines)-1) + " 0 {} 1 ".format(kpts) + str(orbital_num) + " 1 0\n")
          for line in dat_file.readlines():
            d3_file.write(line)
    if sym_name == "C":
      with open("library/d3_input/Orthorombic_AB.d3", 'r+') as dat_file:
          d3_file.write(str(len(d3_lines)-1) + " 0 {} 1 ".format(kpts) + str(orbital_num) + " 1 0\n")
          for line in dat_file.readlines():
            d3_file.write(line)
    if sym_name == "F":
      with open("library/d3_input/Orthorombic_FC.d3", 'r+') as dat_file:
          d3_file.write(str(len(d3_lines)-1) + " 0 {} 1 ".format(kpts) + str(orbital_num) + " 1 0\n")
          for line in dat_file.readlines():
            d3_file.write(line)
    if sym_name == "I":
      with open("library/d3_input/Orthorombic_BC.d3", 'r+') as dat_file:
          d3_file.write(str(len(d3_lines)-1) + " 0 {} 1 ".format(kpts) + str(orbital_num) + " 1 0\n")
          for line in dat_file.readlines():
            d3_file.write(line)
    if sym_name == "A":
      with open("library/d3_input/Orthorombic_AB.d3", 'r+') as dat_file:
          d3_file.write(str(len(d3_lines)-1) + " 0 {} 1 ".format(kpts) + str(orbital_num) + " 1 0\n")
          for line in dat_file.readlines():
            d3_file.write(line)
  elif sym_num >= 75  and sym_num < 143:
    if sym_name == "I":
      d3_file.write("4 16 {} 1 ".format(kpts) + str(orbital_num) + " 1 0\n")
      d3_file.write("8 8 -8  0 0 0\n")
      d3_file.write("0 0 0   8 8 8\n")
      d3_file.write("8 8 8   0 0 8\n")
      d3_file.write("0 0 8   0 0 0\n")
      d3_file.write("END")
    if sym_name == "P":
      with open("library/d3_input/Tetragonal_Simple.d3", 'r+') as dat_file:
          d3_file.write(str(len(d3_lines)-1) + " 0 {} 1 ".format(kpts) + str(orbital_num) + " 1 0\n")
          for line in dat_file.readlines():
            d3_file.write(line)
  elif sym_num >= 143 and sym_num < 168:
    if sym_name == "P":
      with open("library/d3_input/Hexagonal.d3", 'r+') as dat_file:
          d3_file.write(str(len(d3_lines)-1) + " 0 {} 1 ".format(kpts) + str(orbital_num) + " 1 0\n")
          for line in dat_file.readlines():
            d3_file.write(line)
    if sym_name == "R":
      with open("library/d3_input/Rhombohedral.d3", 'r+') as dat_file:
          d3_file.write(str(len(d3_lines)-1) + " 0 {} 1 ".format(kpts) + str(orbital_num) + " 1 0\n")
          for line in dat_file.readlines():
            d3_file.write(line)
  elif sym_num >= 168 and sym_num < 195:
      with open("library/d3_input/Hexagonal.d3", 'r+') as dat_file:
          d3_file.write(str(len(d3_lines)-1) + " 0 {} 1 ".format(kpts) + str(orbital_num) + " 1 0\n")
          for line in dat_file.readlines():
            d3_file.write(line)
  elif sym_num >= 195:
    if sym_name == "P":
      with open("library/d3_input/Cubic_Simple.d3", 'r+') as dat_file:
          d3_file.write(str(len(d3_lines)-1) + " 0 {} 1 ".format(kpts) + str(orbital_num) + " 1 0\n")
          for line in dat_file.readlines():
            d3_file.write(line)
    if sym_name == "F":
      with open("library/d3_input/Cubic_FC.d3", 'r+') as dat_file:
          d3_file.write(str(len(d3_lines)-1) + " 0 {} 1 ".format(kpts) + str(orbital_num) + " 1 0\n")
          for line in dat_file.readlines():
            d3_file.write(line)
    if sym_name == "I":
      with open("library/d3_input/Cubic_BC.d3", 'r+') as dat_file:
          d3_file.write(str(len(d3_lines)-1) + " 0 {} 1 ".format(kpts) + str(orbital_num) + " 1 0\n")
          for line in dat_file.readlines():
            d3_file.write(line)

"""Reads each file given by data_folder and loops through to create d3"""

def write_band_d3(file_name,data_folder=""):
  if ".d12" in file_name:
    input_file_name = os.path.join(data_folder, file_name)
    input_file = open(input_file_name, 'r+')
    input_lines = []
    for line in input_file.readlines():
      if '\n' in line:
       clean_line = line.replace('\n','')
       input_lines.append(clean_line)
      else:
       input_lines.append(line)
    output_name = file_name.replace(".d12",".out")
    output_file_name = os.path.join(data_folder,output_name)
    output_file = open(output_file_name, 'r+')
    output_lines = []
    for line in output_file.readlines():
      if '\n' in line:
       clean_line = line.replace('\n','')
       output_lines.append(clean_line)
      else:
       output_lines.append(line)
    d3_name = file_name.replace(".d12","_BAND.d3")
    d3_file_name = os.path.join(data_folder,d3_name)
    d3_file = open(d3_file_name, 'w+')

    # Move Relevant Files to
    f9_old = file_name.replace(".d12",".f9")
    f9_new = file_name.replace(".d12","_BAND.f9")
    os.system("mv " + data_folder +"/" + f9_old + " " + data_folder + "/" + f9_new)
    f20_old = file_name.replace(".d12",".f20")
    f20_new = file_name.replace(".d12","_BAND.f20")
    os.system("mv " + data_folder +"/" + f20_old + " " + data_folder + "/" + f9_new)

    sym_num, sym_name, orbital_num = sym(input_lines, output_lines)
    band_d3(sym_num, sym_name, orbital_num, d3_file,output_name)
    #os.system("rm " + data_folder +"/" + file_name + " " + data_folder + "/" + output_name)

def all_band_d3(directory=""):
    data_files = os.listdir(directory)
    for file_name in data_files:
        write_band_d3(file_name,directory)
