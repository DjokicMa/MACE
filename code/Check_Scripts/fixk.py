#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 09:50:36 2022

@author: marcus
"""

import os
import pandas as pd

file_to_search = os.getcwd()

File = []
counter = 0
for rootdir, dirs, files in os.walk(file_to_search):
    for f in files:
        if ".d12" in f:
            with open(rootdir + '/' + f,'r') as fi:
                l = list(fi)
            
            with open(rootdir + '/' + f,'w') as output:
                for lines in l:
                    if counter > 0:
                        counter += 1
                    if "SHRINK" in lines:
                        counter += 1
                    if counter == 3:
                        print(lines)
                        
                        klow = min(list(map(int, lines.split())))
                        newk = ' ' + str(klow) + ' ' + str(klow) + ' ' + str(klow) + '\n'
                        
                        output.write(newk)
                        counter = 0
                        
                    else: 
                        output.write(lines)
                        
