#!/usr/bin/env python3
"""
This script finds all .d3 files in the current directory and submits them for processing 
by running submit_prop_17.sh with a parameter of 100 for each file.
"""
import os, sys, math
import re
import linecache
import shutil
import itertools

"""Change the data_folder depending on where your files are"""
data_folder = os.getcwd() 
data_files = os.listdir(data_folder)

"""Reads each file given by data_folder and loops through to find the average bond length"""
for file_name in data_files:
  if ".d3" in file_name:
    submit_name = file_name.split(".d3")[0]
    os.system(data_folder + "/submit_prop_17.sh " + submit_name + " 100")
