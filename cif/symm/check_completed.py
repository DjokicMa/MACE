#!/usr/bin/env python3

"""
Please run updatelists.py before this, and then make a completed and errored directories
This script will move all completed runs into the completed folder
"""
import os, sys, math
import re
import pandas as pd

direct = os.path.dirname(os.path.realpath(__file__))
print(direct)
completed = str(direct)+'/completed'
print(completed)
data_files = pd.read_csv('completelist.csv', delimiter=',')
print(data_files)

for row in data_files.itertuples():
  print(row.data_files)

  name = str(row.data_files)
  print(name)
  sh_file = str('/'+name+'.sh')
  print(sh_file)
  out_file = str('/'+name+'.out')
  print(out_file)
  d12_file = str('/'+name+'.d12')
  print(d12_file)
  f9_file = str('/'+name+'.f9')  
  print(f9_file)

  os.replace(os.getcwd() + sh_file, completed + sh_file)
  os.replace(os.getcwd() + out_file, completed + out_file)
  os.replace(os.getcwd() + d12_file, completed + d12_file)
  os.replace(os.getcwd() + f9_file, completed + f9_file)
