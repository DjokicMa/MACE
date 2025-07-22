# -*- coding: utf-8 -*-


import ase.io
import numpy as np
import math as m
import glob

class Element:
    H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P = list(range(1, 16))
    S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn = list(range(16, 31))
    Ga, Ge, As, Se, Br, Kr, Rb, Sr, Y, Zr, Nb, Mo, Tc, Ru = list(range(31, 45))
    Rh, Pd, Ag, Cd, In, Sn, Sb, Te, I, Xe, Cs, Ba, La, Ce = list(range(45, 59))
    Pr, Nd, Pm, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu, Hf = list(range(59, 73))
    Ta, W, Re, Os, Ir, Pt, Au, Hg, Tl, Pb, Bi, Po, At, Rn = list(range(73, 87))
    Fr, Ra, Ac, Th, Pa, U, Np, Pu, Am, Cm, Bk, Cf, Es, Fm = list(range(87, 101))
    Md, No, Lr, Rf, Db, Sg, Bh, Hs, Mt, Ds, Rg, Cn, Uut = list(range(101, 114))
    Fl, Uup, Lv, Uus, Uuo = list(range(114, 119))

def basis(num, basisset):
    if basisset == "DZ":
        dir_bas = "/home/marcus/Downloads/COF_Cifs/helpscripts/code/full.basis.doublezeta/" #Change This Directory to Double Zeta Basis Set Directory
    elif basisset == "TZ":
        dir_bas = "/home/marcus/Downloads/COF_Cifs/helpscripts/code/full.basis.triplezeta/" #Change This Directory to Triple Zeta Basis Set Directory
    else: print("ERROR Improper Basis Set")
    f       = open(dir_bas+str(num))
    bs      = f.read() 
    f.close()
    return(bs)

def unique(list):
    x = []
    for a in list:
        if a not in x:
            x.append(a)
    x.sort()
    return(x)

def CIF2D12(material,struc,path,opt,basisset):
    
    with open(DIR+material,'r') as x:
        contents = x.readlines()
    
    # counters for parsing
    sym_counter = 0
    atom_counter = 0
    a_counter = 0
    b_counter = 0
    c_counter = 0 
    alpha_counter = 0
    beta_counter = 0
    gamma_counter = 0
    atom_list = []
    # parses cif for info
    for line in contents:
        for i in line.split():
            # get lattice parameters
            if i == '_cell_length_a':
                a_counter = 1
            elif a_counter == 1:
                a = float(i)
                a_counter =0
            if i == '_cell_length_b':
                b_counter = 1
            elif b_counter == 1:
                b = float(i)
                b_counter =0
            if i == '_cell_length_c':
                c_counter = 1
            elif c_counter == 1:
                c = float(i)
                c_counter =0
            # get unit cell angle
            if i == '_cell_angle_alpha':
                alpha_counter = 1
            elif alpha_counter == 1:
                alpha = float(i)
                alpha_counter =0
            if i == '_cell_angle_beta':
                beta_counter = 1
            elif beta_counter == 1:
                beta = float(i)
                beta_counter =0
            if i == '_cell_angle_gamma':
                gamma_counter = 1
            elif gamma_counter == 1:
                gamma = float(i)
                gamma_counter =0
            # get symmetry
            if i == '_symmetry_Int_Tables_number':
                sym_counter = 1
            elif sym_counter == 1:
                spacegroup = i
                sym_counter = 0
            # get all atom info
            if i == '_atom_site_occupancy':
                atom_counter = 1
            elif i == 'loop_':
                atom_counter = 0
            elif atom_counter == 1:
                atom_list.append(i)
                
    # Specify only requires Lattic Parameters for Space Group
    if int(spacegroup) >= 1 and int(spacegroup) <= 2: #Triclinic
        UC = "%-8.6f   %-8.6f  %-8.6f  %-6.4f  %-6.4f  %-6.4f #a,b,c,alpha,beta,gamma Triclinic"%(a,b,c,alpha,beta,gamma)
    elif int(spacegroup) >= 3 and int(spacegroup) <= 15: #Monoclinic
        UC = "%-8.6f   %-8.6f  %-8.6f  %-6.4f #a,b,c,beta Monoclinic alpha = gamma = 90"%(a,b,c,beta)
    elif int(spacegroup) >= 16 and int(spacegroup) <= 74: #Orthorombic
        UC = "%-8.6f   %-8.6f  %-8.6f #a,b,c Orthorombic alpha = beta = gamma = 90"%(a,b,c)
    elif int(spacegroup) >= 75 and int(spacegroup) <= 142: #Tetragonal
        UC = "%-8.6f   %-8.6f #a=b,c Tetragonal alpha = beta = gamma = 90"%(a,c)
    elif int(spacegroup) >= 143 and int(spacegroup) <= 167: #Trigonal
        UC = "%-8.6f   %-8.6f #a=b,c Trigonal alpha = beta = 90, gamma = 120"%(a,c)
    elif int(spacegroup) >= 168 and int(spacegroup) <= 194: #Hexagonal
        UC = "%-8.6f   %-8.6f #a=b,c Hexagonal alpha = beta = 90, gamma = 120"%(a,c)
    elif int(spacegroup) >= 195 and int(spacegroup) <= 230: #cubic
        UC = "%-8.6f #a=b=c cubic alpha = beta = gamma = 90 "%(a)
    
    
    # parse atom info for fraction unit cell coordinates + name
    true_index = 0
    index = 0
    atom_name = []
    h = []
    k = []
    l = []
    for i in atom_list:
        if index == 1:
            atom_name.append(i)
        if index == 2:
            h.append(float(i))
        if index == 3:
            k.append(float(i))
        if index == 4:
            l.append(float(i))
        
        true_index += 1
        index += 1 
        if index == 8:
            index = 0

    
    # convert name to atomic number
    an =[] 
    for i in atom_name:
        atom = getattr(Element,i)
        an.append(int(atom))
   
         
    title  = material[:-4]
    output = title+"_"+struc+"_"+opt+"_"+basisset+".d12"
    with open(output,'w') as f: 

        ATOMS = len(an)
        
        print(title,file=f)
        if struc == "SLAB":
            print("SLAB",file=f)
            print(str(sg_2d),file=f)

            print("%-8.6f   %-8.6f  %-6.4f"%(a,b,gamma),file=f)
            print(str(ATOMS),file=f)
        if struc == "BULK":
            print("CRYSTAL\n0 0 0",file=f)
            #print(str(sg_bk),file=f)
            print(spacegroup,file=f)
            print(UC,file=f)
            print(str(ATOMS),file=f)

        ECPs    = [37,38,39,40,41,42] #full.basis        
        for i in range(0,ATOMS):
            if (an[i] in ECPs) or (an[i] > 43):
                # Use ELECTRON CORE POTENTIALS 
                an[i] += 200

            if struc == "BULK":
                print("%-3d %-8.6f  %-8.6f  %-9.6f  Biso    1.000000    %s "%(an[i],h[i],k[i],l[i],atom_name[i]),file=f)
            if struc == "SLAB":
                if zi>2.*c: zi = zi - 3.*c
                print("%-3d %-8.6f  %-8.6f  %-9.6f  Biso    1.000000    %s "%(an[i],h[i],k[i],z[i],name[i]),file=f)

        if opt   == "SP":
            OPT  = "END"
        elif opt   == "OPT":	
            OPT  = "OPT\nCVOLOPT\nMAXCYCLE\n800\nENDOPT\nEND"
        elif opt == "OPTGEOM":
            OPT  = "OPTGEOM\nFULLOPTG\nMAXCYCLE\n1600\nENDOPT\nEND"
            
        print(OPT,file=f)
        #WRITE INPUT DECK
        ## BASIS SETS
        ### DETERMINE UNIQUE ELEMENTS
        ele = unique(an)
        ### Include the Basis Sets for Each element
        for i in ele:
            if struc == "BULK":
                print(basis(i,basisset),end='',file=f)      
            elif struc == "SLAB":    
                print(basis(i,basisset),end='',file=f)
            else: print("ERROR Improper Structure type input")
            
        ks      = [2,3,5,6,10,15,30]
        FM      = 80 # FM Mixing

        ka = kb = kc =1
        for k in ks:
            if k*a > 40. and k*a<75. and ka ==1: ka = k
            if k*b > 40. and k*b<75. and kb ==1: kb = k
            if k*c > 40. and k*c<75. and kc ==1: kc = k
            #if int(spacegroup)!= 1:
            #   ka = kb = kc = 20
        
                
        if ka ==0 or kb == 0 or kc == 0: print("ERROR:",ka,kb,kc)
        k_max = max([ka,kb,kc])
        #ka = k_max
        #kb = k_max
        #kc = k_max
        nShrink = k_max*2
        if struc == "BULK":
            TAIL    = "99 0\nEND\nDFT\nSPIN\nPBE-D3\nXLGRID\nEND\nTOLINTEG\n7 7 7 7 14\nTOLDEE\n7\nMEMOPRT\nSHRINK\n0 %d\n %d %d %d\nSCFDIR\nSAVEWF\nSAVEPRED\nBIPOSIZE\n110000000\nEXCHSIZE\n110000000\nMAXCYCLE\n1600\nFMIXING\n%d\nDIIS\nHISTDIIS\n100\nPPAN\nEND"%(nShrink,ka,kb,kc,FM)    
        if struc == "SLAB":    
            TAIL    = "99 0\nEND\nDFT\nSPIN\nHSE06-D3\nXLGRID\nEND\nTOLINTEG\n9 9 9 9 18\nTOLDEE\n7\nSHRINK\n0 %d\n %d %d 1\nSCFDIR\nBIPOSIZE\n110000000\nEXCHSIZE\n110000000\nMAXCYCLE\n800\nFMIXING\n%d\nDIIS\nPPAN\nEND"%(nShrink,ka,kb,FM)    
        print(TAIL,file=f)
    return 
DIR     = "/home/marcus/Documents/PORMAKE data/Large/" # Change This Directory to CIF Directory
pathlist = glob.glob(DIR+'*.cif')
nDIR     = len(DIR)
ntype    = len(".cif")

while True:
    try:
        options1 = int(input(' Enter 0 for Single Point Energy, Enter 1 for OPT cVol, or Enter 2 for OPTGEOM: \n'))
        options2 = int(input('Enter 3 for SLAB or Enter 4 for BULK: \n'))
        options3 = int(input('Enter 5 for Double Zeta Basis Set or Enter 6 for Triple Zeta Basis Set: \n'))
        break
    
    except ValueError:
        print('Invalid Input. Try again.')

material = ""
for path in pathlist:
    # because path is object not string
    option1 = options1
    option2 = options2
    option3 = options3
    path_in_str = str(path)
    material = path_in_str[nDIR:]
    if material   == "":break
    if option1 == 0: 
        option1  = "SP"
    elif option1 == 1:
        option1  = "OPT"
    elif option1 == 2: 
        option1  = "OPTGEOM"
    else: 
        print('Invalid Input for Option 1. Try again.')
        break
    if option2 == 3:
        option2  = "SLAB"
    elif option2 == 4:
        option2  = "BULK"
    else: 
        print('Invalid Input for Option 2. Try again.')
        break
    if option3 == 5:
        option3  = "DZ"
    elif option3 == 6:
        option3  = "TZ"
    else: 
        print('Invalid Input for Option 3. Try again.')
        break
    CIF2D12(material,option2,DIR,option1,option3)
    
