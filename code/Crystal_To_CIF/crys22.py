from ase.build import bulk
from ase import Atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.optimize import QuasiNewton
from ase.build import fcc111, add_adsorbate
from ase.spacegroup.symmetrize import *
from ase.spacegroup.xtal import crystal
from ase.spacegroup import Spacegroup
from ase.spacegroup import get_spacegroup
from ase.io import read, write
import ast
import glob
import numpy as np
from numpy import sqrt
from numpy import unique
from scipy.spatial.distance import pdist,cdist
import unittest

def read_Output(file):
    ifin = 0
    with open(file) as f:
        FINAL   = "FINAL OPTIMIZED GEOMETRY"
        INPUT   = "GEOMETRY FOR WAVE FUNCTION"
        SLAB    = " * TWO DIMENSIONAL SLAB PARALLEL TO THE SELECTED PLANE"
        SLABCUT = False

        if SLAB in f.read():
            SLABCUT = True
            print("TRUE")
            f.seek(0)

        if FINAL in f.read():
            f.seek(0)
            for i, line in enumerate(f.readlines()):
                if FINAL in line:
                    ifin = i
        else:
            f.seek(0)
            for i, line in enumerate(f.readlines()):
                if INPUT in line:
                    ifin = i

        ilat = ifin +6
        inum = ifin +8
        ipos = ifin +11

        f.seek(0)

        data = f.readlines()

        a = float(data[ilat].split()[0])
        b = float(data[ilat].split()[1])
        c = float(data[ilat].split()[2])
        alpha = float(data[ilat].split()[3])
        beta  = float(data[ilat].split()[4])
        gamma = float(data[ilat].split()[5])

        atoms = int(data[inum].split()[-1])
        ecps  = []
        ele   = []
        fract = np.zeros((atoms,3))
        for j in range(ipos,ipos+atoms):
            k = j-ipos
            fract[k,0] = float(data[j].split()[-3])
            fract[k,1] = float(data[j].split()[-2])
            if SLABCUT == True:
                fract[k,2] = float(data[j].split()[-1])/500.
            else:
                fract[k,2] = float(data[j].split()[-1])
            ecps.append(int(data[j].split()[-5]))
            ele.append(data[j].split()[-4].title())

    output = crystal(ele, fract, spacegroup =1, cell=[a,b,c,alpha,beta,gamma])

    return(output)

class Crystal22():
    #Accepts an ASE-Python Crystal type object, material name and parameter file
    # NEED TO ACCOUNT FOR SPACEGROUPS WHEN WRITING UNIT CELL PARAMETERS
    ## - Also for d3 files
    ## - Also for Plotting Scripts
    ##
    ## Add Getters and Setters/ property decorators
    ##  - unitcell
    ##  - material
    ##  - ele
    ##  - fractional_coord
    ##  - crystal
    ######################################################################################
    def __init__(self,crystal,material='0',form='CRYSTAL',param='OPT'):
        self.crystal    = crystal
        self.material   = material
        self.sg         = get_spacegroup(self.crystal).no
        self.form       = form
        if self.form == 'SLAB':
            self.sg = 1
        # '/Users/williamcomaskey/Desktop/CRYS22/opt.param'
        paramlib = "/Users/williamcomaskey/Desktop/Projects/crys22/library/parameters/"

        if param in ['OPT','opt','Opt']:
            fparam = paramlib+"opt.param"
            with open(fparam) as f:
                data = f.read()
            self.param = ast.literal_eval(data)

        elif param in ['1C','1c']:
            fparam = paramlib+"1c.param"
            with open(fparam) as f:
                data = f.read()
            self.param = ast.literal_eval(data)

        elif param in ['2C','2c']:
            fparam = paramlib+"2c.param"
            with open(fparam) as f:
                data = f.read()
            self.param = ast.literal_eval(data)

        elif param in ['SOC','soc','SC','sc']:
            fparam = paramlib+"soc.param"
            with open(fparam) as f:
                data = f.read()
            self.param = ast.literal_eval(data)

        else:
            print("Invalid Parameter Input File")

    def set_crystal(self,crystal):
        self.crystal    = crystal
        self.sg         = get_spacegroup(self.crystal).no
        self.material   = self.crystal.get_chemical_formula()


    def read2Crystal(self,filename,fmt=""):
        if fmt =="":
            self.crystal = read(filename)
        else:
            self.crystal = read(filename,format=fmt)
        self.sg         = get_spacegroup(self.crystal).no

    def Header(self):
        if self.material == '0':
            #List out Elements and Make Label
            self.material = self.crystal.get_chemical_formula()

        #Make Header From Title
        if self.form == 'SLAB':
            d12_Header = "{}\nSLAB\n{}\n".format(self.material,self.sg)
        else:
            d12_Header = "{}\nCRYSTAL\n0 0 0\n{}\n".format(self.material,self.sg)
        return(d12_Header)

    def Optimize(self):
        #THIS SHOULD BE DONE LAST AFTER EVERYTHING ELSE
        d12_Optimize = ""
        if self.param == '0':
            d12_Optimize = "OPTGEOM\nFULLOPTG\nMAXCYCLE\n800\nENDOPT\nEND\n"
        else:
            #Use Parameters from Input File
            if self.param["SLABCUT"][0] == 1:
                d12_Optimize +="SLABCUT"+"\n"
                d12_Optimize += self.param["SLABCUT"][1]+"\n"
                d12_Optimize += self.param["SLABCUT"][2]+"\n"

            if self.param["Optimize"][0] == 1:
                for i in range(1,len(self.param["Optimize"])):
                    d12_Optimize +=self.param["Optimize"][i]+"\n"

            if self.param["MAXCYCLE"][0] == 1:
                d12_Optimize += "MAXCYCLE"

            if self.param["MAXTRADIUS"][0] ==1:
                d12_Optimize += "MAXTRADIUS"+"\n"
                d12_Optimize += str(self.param["MAXTRADIUS"][1])+"\n"
            if (self.param["Symmetry"] ==0) or (self.param["TWOCOMP"] == 1) or (self.param["SOC"] == 1):
                d12_Optimize += "SYMMREMO"+"\n"
            d12_Optimize += "END"+"\n"
        return(d12_Optimize)


    def Check_Sym(self):
        lattice = ""

        try:
            sym = get_spacegroup(self.crystal).symbol[0]
            sg  = get_spacegroup(self.crystal).no
        except ValueError:
            print("Error Improper Input! Use ASE Atoms/Crystal Object")
        lattice_sub =""
        if sg < 3:
                lattice = "Triclinic"
        elif sg >= 3 and sg < 16:
            if sym == "P":
                lattice     = "Monoclinic"
                lattice_sub = "Simple"
            if sym == "C":
                lattice     = "Monoclinic"
                lattice_sub = "FC"
        elif sg >= 16 and sg < 75:
            if sym == "P":
                lattice = "Orthorombic"
                lattice_sub = "FC"
            if sym == "C":
                lattice = "Orthorombic"
                lattice_sub ="AB"
            if sym == "F":
                lattice = "Orthorombic"
                lattice_sub ="FC"
            if sym == "I":
                lattice = "Orthorombic"
                lattice_sub ="BC"
            if sym == "A":
                lattice = "Orthorombic"
                lattice_sub ="AB"
        elif sg >= 75  and sg < 143:
            if sym == "I":
                lattice = "Tetragonal"
                lattice_sub ="BC"
            if sym == "P":
                lattice = "Tetragonal"
                lattice_sub = "Simple"
        elif sg >= 143 and sg < 168:
            if sym == "P":
                lattice = "Hexagonal"
            if sym == "R":
                lattice = "Rhombohedral"
        elif sg >= 168 and sg < 195:
            lattice = "Hexagonal"
        elif sg >= 195:
            if sym == "P":
                lattice = "Cubic"
                lattice_sub ="Simple"
            if sym == "F":
                lattice = "Cubic"
                lattice_sub ="FC"
            if sym == "I":
                lattice = "Cubic"
                lattice_sub ="BC"
        #if lattice_sub == "":
        #    crys_system = "{}".format(lattice)
        #else:
        #    crys_system = "{}_{}".format(lattice,lattice_sub)
        return(lattice,lattice_sub)

    def minimalunitcell(self):
        """
        When Writing .d12 input files the minimal set of lattice parameters must be used
        when utilizing spacegroups.
          alpha = >bc
          beta  = >ac
          gamma = >ab
        """


        lattice,lattice_sub = self.Check_Sym()

        cell = self.crystal.cell.cellpar()
        a = cell[0]
        b = cell[1]
        c = cell[2]

        alpha = cell[3]
        beta  = cell[4]
        gamma = cell[5]


        if   lattice == "Cubic":
            unit_cell = "%-8.6f"%(a)

        elif lattice == "Hexagonal":
            unit_cell = "%-8.6f\t%-8.6f"%(a,c)

        elif lattice == "Hexagonal":
            unit_cell = "%-8.6f\t%-8.6f"%(a,c)
        elif lattice == "Rhombohedral":
            unit_cell = "%-8.6f\t%-8.6f"%(a,alpha)

        elif lattice == "Tetragonal":
            unit_cell = "%-8.6f\t%-8.6f"%(a,c)

        elif lattice == "Orthorhombic":
            unit_cell = "%-8.6f\t%-8.6f\t%-8.6f"%(a,b,c)

        elif lattice == "Monoclinic":
            # Find Unique lattice constant from the first 3
            ## - Find List Location
            ## - Determine if a,b,c
            ## - Set for below
            id = np.where(unique([a,b,c]))[0][0]
            #print("ID",id)
            if id==0: #a unique
                unit_cell = "%-8.6f\t%-8.6f\t%-8.6f\t%-8.6f"%(a,b,c,beta)
            elif id==1: #b unique
                unit_cell = "%-8.6f\t%-8.6f\t%-8.6\t%-8.6ff"%(a,b,c,gamma)
            elif id==2: #c unique
                unit_cell = "%-8.6f\t%-8.6f\t%-8.6f\t%-8.6f"%(a,b,c,alpha)
        elif lattice == "Triclinic":
            unit_cell = "%-8.6f\t%-8.6f\t%-8.6f\t%-8.6f\t%-8.6f\t%-8.6f"%(a,b,c,alpha,beta,gamma)
        else:
            unit_cell = "%-8.6f\t%-8.6f\t%-8.6f\t%-8.6f\t%-8.6f\t%-8.6f"%(a,b,c,alpha,beta,gamma)
        return(unit_cell)

    def Cell(self):

        fract = self.crystal.get_scaled_positions()
        ans   = self.crystal.get_atomic_numbers()
        ECPS  = [26,34,35,37,38,39,40]
        self.ecps = []

        cell = self.crystal.cell.cellpar()
        a = cell[0]
        b = cell[1]
        c = cell[2]

        alpha = cell[3]
        beta  = cell[4]
        gamma = cell[5]

        for an in ans:
            if an in ECPS or an > 1:
                self.ecps.append(an+200)
            else:
                self.ecps.append(an)
        d12_Cell = ""
        #d12_Cell += str(self.sg)+"\n"
        if self.sg != 1:
            d12_Cell += self.minimalunitcell()
        else:
            if self.form == 'SLAB':
                d12_Cell += "%-8.6f\t%-8.6f\t%-8.6f"%(a,b,gamma)
            else:
                for i in range(0,len(cell)):
                     d12_Cell += "{0:0.8f}  ".format(cell[i])

        d12_Cell += "\n"

        d12_Cell += str(len(fract))+"\n"
        for i in range(0,len(fract)):
            d12_Cell += "{}  ".format(self.ecps[i])
            for j in range(0,len(fract[i])):
                d12_Cell +=  "{0:0.8f}   ".format(fract[i,j])
            d12_Cell += "\n"
        return(d12_Cell)

    def Basis(self):
        #LOGIC BLOCK FOR ATOMIC NUMBERS BASED ON BASIS SET LIBRARY
        #
        ##########################################################
        #Directory Where Basis Sets are Stored
        lib_basis = "/Users/williamcomaskey/Desktop/Projects/crys22/library/basis/"
        basis     = self.param["BASIS"]
        #TRIM TO UNIQUE ATOMIC NUMBERS
        atomic_num = unique(self.ecps)
        bs =""
        for i in range(0,len(atomic_num)):
            num     = atomic_num[i]
            basis_i = "{0:s}{1:s}/{2:d}".format(lib_basis,basis,num)
            f       = open(basis_i)
            bs      += f.read()
            f.seek(0)
            if f.read().endswith('\n'): pass
            else: bs      +='\n'
            f.close()
        bs +="99 0\nEND\n"
        return(bs)

    def DFT(self):
        # ADD LINES PF TEXT FOR TWO COMP
        # - SOC: If 2C is turned on then we must change the dictionary to turn off symmetry, optimizations, Range separated Hybrids etc
        #        We must also create a SOPSEUD file
        # - SPIN: If TWOCOMP or SOC are selcted SPIN is turned off
        # EXTRA VALUES WE NORMALLY USE SUCH AS PRECISION
        if self.param == '0':
            #Use Default Parameters set here
            a,b,c,alpha,beta,gamma = self.crystal.cell.cellpar()
            ks      = [2,3,5,6,10,15]
            ka = kb = kc =1
            for k in ks:
                if k*a > L and k*a<R and ka ==1: ka = k
                if k*b > L and k*b<R and kb ==1: kb = k
                if k*c > L and k*c<R and kc ==1: kc = k
                if self.sg != 1:
                    ka = kb = kc = 15
            d12_DFT = "DFT\nSPIN\nPBE0-D3\nXXLGRID\nEND\nTOLINTEG\n9 9 9 9 18\nTOLDEE\n7\nSHRINK\n0 30\n %d %d %d\nSCFDIR\nBIPOSIZE\n110000000\nEXCHSIZE\n110000000\nMAXCYCLE\n800\nFMIXING\n%d\nANDERSON\nPPAN\nEND"%(ka,kb,kc,FM)

        else:
            d12_DFT = ""
            if self.param["TWOCOMP"] == 1:
                d12_DFT +="TWOCOMPON"+"\n"
                d12_DFT +="GUESSPNOSO"+"\n"

            if self.param["SOC"]     == 1:
                d12_DFT +="SOC"+"\n"

            if (self.param["SOC"] == 1) or (self.param["TWOCOMP"] == 1):
                d12_DFT +="ENDTWO"+"\n"

            d12_DFT +="DFT"+"\n"
            if (self.param["SPIN"] == 1) and (self.param["SOC"] != 1) or (self.param["TWOCOMP"] != 1):
                d12_DFT += "SPIN"+"\n"

            # Check which functional to be used i.e B3LYP,PBE0,HSE06, etc.
            # - HSE06 (Range Separated Hybrid)can not be used if the TWOCOMP code is selected
            if self.param["Functional"]:
                if self.param["Dispersion"]:
                    d12_DFT += self.param["Functional"]+"-"+self.param["Dispersion"]+"\n"
                else:
                    d12_DFT += self.param["Functional"]+"\n"

            # Check Size of Grid default in Crystal is XLGRID
            if self.param["Grid"]:
                d12_DFT += self.param["Grid"]+"\n"
            d12_DFT +="END"+"\n"
            if self.param["K RES"]:
                # Shrink Factor

                if self.param["K RES"] =='LOW':
                    ISP    = 30
                    L      = 40.
                    R      = 60.
                    ks     = [2,3,5,6,10,15]
                if self.param["K RES"] =='MED':
                    ISP    = 60
                    L      = 60.
                    R      = 100.
                    ks     = [2,3,5,6,10,15,20,30]
                if self.param["K RES"] =='HIGH':
                    ISP    = 100
                    L      = 100.
                    R      = 450.
                    ks     = [1, 2, 4, 5, 10, 20, 25, 50]
            else:
                ISP    = 30
                L      = 40.
                R      = 60.
                ks     = [2,3,5,6,10,15]

            a,b,c,alpha,beta,gamma = self.crystal.cell.cellpar()
            IS = IS1 = IS2 = IS3 = 1
            if self.sg == 1:
                for k in ks:
                    if k*a > L and k*a<R and IS1 == 1: IS1 = k
                    if k*b > L and k*b<R and IS2 == 1: IS2 = k
                    if k*c > L and k*c<R and IS3 == 1: IS3 = k
                d12_DFT +="TOLINTEG\n9 9 9 9 18\nTOLDEE\n9\nSHRINK\n0 {0:d}\n {1:d} {2:d} {3:d}\n".format(ISP,IS1,IS2,IS3)

            elif self.sg != 1:
                x = min(a,b,c)
                for k in ks:
                    if k*x > L and k*x<R and IS == 1: IS = k
                ISP =2*IS
                d12_DFT +="TOLINTEG\n9 9 9 9 18\nTOLDEE\n9\nSHRINK\n{0:d} {1:d}\n".format(IS,ISP)


            d12_DFT +="SCFDIR\nMAXCYCLE\n800\n"
            if self.param["Fock-Mixing"]:
                FM = self.param["Fock-Mixing"]
            else:
                FM = 50
            d12_DFT += "FMIXING\n{0:d}".format(FM)
            d12_DFT += "\nANDERSON\nPPAN\nEND"

        return(d12_DFT)

    def D12(self):
        #GLUE TOGETHER EVERYTHING
        d12_Header   = self.Header()
        d12_Cell     = self.Cell()
        d12_Optimize = self.Optimize()
        d12_Basis    = self.Basis()
        d12_DFT      = self.DFT()
        d12_string   = d12_Header+d12_Cell+d12_Optimize+d12_Basis+d12_DFT
        return(d12_string)

    def Write(d12_string, file):
        with open(file, 'w') as f:
            f.write(d12_string)
    def Crystal2Cif(self,file,fmt="cif"):
        a,b,c,alpha,beta,gamma = self.crystal.cell.cellpar()
        if c > 499.0:
            pass
        else:
            self.crystal.write(file,format=fmt)

def write_Output2D12(file, in_type = "OPT",out_type="1C",form='CRYSTAL'):
    outfile = file.replace(".out",".d12")
    outfile = outfile.replace(in_type,out_type)
    output  = read_Output(file)
    crysout = Crystal22(output,form = form, param =out_type)

    with open(outfile,'w') as f:
        f.write(crysout.D12())
