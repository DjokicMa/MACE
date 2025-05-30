#!/usr/bin/env python3
"""
This script generates band structure input files (*.d3) for CRYSTAL based on symmetry info extracted from .d12 and .out files.
It selects the appropriate high-symmetry path template and renames the corresponding .f9 file for band structure calculations.
"""

import os, sys, math
import re
import linecache
import shutil
import itertools
#This directory is where the d3_input folder is saved
dir='/mnt/home/djokicma/bin'

def sym(input_lines, output_lines):
  if input_lines[1] == 'CRYSTAL':
    sym_num = int(input_lines[3])
  elif input_lines[1] == 'SLAB':
    sym_num = int(input_lines[2])
  sym_name = ""
  orbital_num = 0
  for line in output_lines:
    if "SPACE GROUP" in line:
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
  return sym_num, sym_name, orbital_num

def writed3(sym_num, sym_name, orbital_num, d3_file, output_name):
  d3_file.write("NEWK\n48 48\nBAND\n")
  d3_file.write(output_name + "\n")
  d3_lines = []
  #if sym_num == 1:
    ##dat_file = open("/gpfs/home/kevin_lucht/CarbonAllotropes/file_scripts/d3/Triclinic.d3', 'r+')
    ##for line in dat_file.readlines():
    ##  d3_lines.append(line)
    ##d3_file.write(str(len(d3_lines)-1) + " 0 1000 1 " + str(orbital_num) + " 1 0\n")
    ##for line in d3_lines:
    ##  d3_file.write(line)
    #d3_file.write("9 16 1000 1 " + str(orbital_num) + " 1 0\n")
    #d3_file.write("8 8 -8  0 8 0\n")
    #d3_file.write("0 8 0   0 0 0\n")
    #d3_file.write("0 0 0   0 0 8\n")
    #d3_file.write("0 0 8   4 4 4\n")
    #d3_file.write("4 4 4   8 8 8\n")
    #d3_file.write("8 8 8   0 0 0\n")
    #d3_file.write("0 0 0   8 0 0\n")
    #d3_file.write("8 0 0   8 0 8\n")
    #d3_file.write("8 0 8   0 0 0\n")
    #d3_file.write("END")
#M Y
#Y G
#G Z
#Z N
#N R
#R G
#G X
#X L
#L G
  if sym_num <= 2:
    #d3_file.write("9 16 1000 1 " + str(orbital_num) + " 1 0\n")
    dat_file = open(dir+'/d3_input/Triclinic-Explicit.d3', 'r+')
    for line in dat_file.readlines():
        d3_lines.append(line)
    d3_file.write(str(len(d3_lines)-1) + " 2 1000 1 " + str(orbital_num) + " 1 0\n")
    for line in d3_lines:
        d3_file.write(line)
  elif sym_num >= 3 and sym_num < 16:
    if sym_name == "P":
      dat_file = open(dir+'/d3_input/Monoclinic_Simple.d3', 'r+')
      for line in dat_file.readlines():
        d3_lines.append(line)
      d3_file.write(str(len(d3_lines)-1) + " 0 1000 1 " + str(orbital_num) + " 1 0\n")
      for line in d3_lines:
        d3_file.write(line)
    if sym_name == "C":
      dat_file = open(dir+'/d3_input/Monoclinic_AC.d3', 'r+')
      for line in dat_file.readlines():
        d3_lines.append(line)
      d3_file.write(str(len(d3_lines)-1) + " 0 1000 1 " + str(orbital_num) + " 1 0\n")
      for line in d3_lines:
        d3_file.write(line)
  elif sym_num >= 16 and sym_num < 75:
    if sym_name == "P":
      dat_file = open(dir+'/d3_input/Orthorombic_Simple.d3', 'r+')
      for line in dat_file.readlines():
        d3_lines.append(line)
      d3_file.write(str(len(d3_lines)-1) + " 0 1000 1 " + str(orbital_num) + " 1 0\n")
      for line in d3_lines:
        d3_file.write(line)
    if sym_name == "C":
      dat_file = open(dir+'/d3_input/Orthorombic_AB.d3', 'r+')
      for line in dat_file.readlines():
        d3_lines.append(line)
      d3_file.write(str(len(d3_lines)-1) + " 0 1000 1 " + str(orbital_num) + " 1 0\n")
      for line in d3_lines:
        d3_file.write(line)
    if sym_name == "F":
      dat_file = open(dir+'/d3_input/Orthorombic_FC.d3', 'r+')
      for line in dat_file.readlines():
        d3_lines.append(line)
      d3_file.write(str(len(d3_lines)-1) + " 0 1000 1 " + str(orbital_num) + " 1 0\n")
      for line in d3_lines:
        d3_file.write(line)
    if sym_name == "I":
      dat_file = open(dir+'/d3_input/Orthorombic_BC.d3', 'r+')
      for line in dat_file.readlines():
        d3_lines.append(line)
      d3_file.write(str(len(d3_lines)-1) + " 0 1000 1 " + str(orbital_num) + " 1 0\n")
      for line in d3_lines:
        d3_file.write(line)
    if sym_name == "A":
      dat_file = open(dir+'/d3_input/Orthorombic_AB.d3', 'r+')
      for line in dat_file.readlines():
        d3_lines.append(line)
      d3_file.write(str(len(d3_lines)-1) + " 0 1000 1 " + str(orbital_num) + " 1 0\n")
      for line in d3_lines:
        d3_file.write(line)
  elif sym_num >= 75  and sym_num < 143:
    if sym_name == "I":
      dat_file = open(dir+'/d3_input/Tetragonal_BC.d3', 'r+')
      for line in dat_file.readlines():
        d3_lines.append(line)
      d3_file.write(str(len(d3_lines)-1) + " 0 1000 1 " + str(orbital_num) + " 1 0\n")
      for line in d3_lines:
        d3_file.write(line)
#      d3_file.write("4 16 1000 1 " + str(orbital_num) + " 1 0\n")
#      d3_file.write("8 8 -8  0 0 0\n")
#      d3_file.write("0 0 0   8 8 8\n")
#      d3_file.write("8 8 8   0 0 8\n")
#      d3_file.write("0 0 8   0 0 0\n")
#      d3_file.write("END")
    if sym_name == "P":
      dat_file = open(dir+'/d3_input/Tetragonal_Simple.d3', 'r+')
      for line in dat_file.readlines():
        d3_lines.append(line)
      d3_file.write(str(len(d3_lines)-1) + " 0 1000 1 " + str(orbital_num) + " 1 0\n")
      for line in d3_lines:
        d3_file.write(line)
  elif sym_num >= 143 and sym_num < 168:
    if sym_name == "P":
      dat_file = open(dir+'/d3_input/Hexagonal.d3', 'r+')
      for line in dat_file.readlines():
        d3_lines.append(line)
      d3_file.write(str(len(d3_lines)-1) + " 0 1000 1 " + str(orbital_num) + " 1 0\n")
      for line in d3_lines:
        d3_file.write(line)
    if sym_name == "R":
      dat_file = open(dir+'/d3_input/Rhombohedral.d3', 'r+')
      for line in dat_file.readlines():
        d3_lines.append(line)
      d3_file.write(str(len(d3_lines)-1) + " 0 1000 1 " + str(orbital_num) + " 1 0\n")
      for line in d3_lines:
        d3_file.write(line)
  elif sym_num >= 168 and sym_num < 195:
    dat_file = open(dir+'/d3_input/Hexagonal.d3', 'r+')
    for line in dat_file.readlines():
      d3_lines.append(line)
    d3_file.write(str(len(d3_lines)-1) + " 0 1000 1 " + str(orbital_num) + " 1 0\n")
    for line in d3_lines:
      d3_file.write(line)
  elif sym_num >= 195:
    if sym_name == "P":
      dat_file = open(dir+'/d3_input/Cubic_Simple.d3', 'r+')
      for line in dat_file.readlines():
        d3_lines.append(line)
      d3_file.write(str(len(d3_lines)-1) + " 0 1000 1 " + str(orbital_num) + " 1 0\n")
      for line in d3_lines:
        d3_file.write(line)
    if sym_name == "F":
      dat_file = open(dir+'/d3_input/Cubic_FC.d3', 'r+')
      for line in dat_file.readlines():
        d3_lines.append(line)
      d3_file.write(str(len(d3_lines)-1) + " 0 1000 1 " + str(orbital_num) + " 1 0\n")
      for line in d3_lines:
        d3_file.write(line)
    if sym_name == "I":
      dat_file = open(dir+'/d3_input/Cubic_BC.d3', 'r+')
      for line in dat_file.readlines():
        d3_lines.append(line)
      d3_file.write(str(len(d3_lines)-1) + " 0 1000 1 " + str(orbital_num) + " 1 0\n")
      for line in d3_lines:
        d3_file.write(line)

"""Change the data_folder depending on where your files are"""
data_folder = os.getcwd()
#data_folder = r'/mnt/home/maldo103/2DHUB_2021/MX-MX2_Calchogenides/Cu2Te_BANDS'
data_files = os.listdir(data_folder)

"""Reads each file given by data_folder and loops through to create d3"""
for file_name in data_files:
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
    f9_old = file_name.replace(".d12",".f9")
    f9_new = file_name.replace(".d12","_BAND.f9")
    os.system("mv " + data_folder +"/" + f9_old + " " + data_folder + "/" + f9_new)
    sym_num, sym_name, orbital_num = sym(input_lines, output_lines)
    writed3(sym_num, sym_name, orbital_num, d3_file,output_name)
    #os.system("rm " + data_folder +"/" + file_name + " " + data_folder + "/" + output_name)
