# This script converts crystal output files to CIF using python3
# Usage: python crystal_to_cif_python3.py
# Author: Daniel Maldonado

import re, os, glob, subprocess

#Remove PROCESS lines that might be interfering with file order
subprocess.run('sed -i "/^ PROCESS/d" *.out',shell=True, check=True)

def crystal17_to_cif(input_file, output_file):
    with open(input_file, 'r') as f:
        cell_counter = atom_counter = num_atoms = 0
        atom_data = []
        for i,line in enumerate(f):
            # Extract cell parameters
            if line.startswith(" PRIMITIVE CELL"):
                cell_counter = i
            if cell_counter > 0 and i == cell_counter+2:
                cell_params = line
            # Extract number of atoms
            if line.startswith(' ATOMS IN THE ASYMMETRIC UNIT'):
                num_atoms = int(line.split()[-1])
                atom_counter = i
            # Extract atomic positions
            if i >= atom_counter+3 and i < atom_counter+3+num_atoms and i>3:
                atom_data.append([line.split()[3].capitalize(), line.split()[4],line.split()[5],line.split()[6]]) 
            # Determine if bulk or slab calculation
            if line.startswith(' CRYSTAL CALCULATION'):
                calc_type = 'bulk'
            if line.startswith(' SLAB CALCULATION'):
                calc_type = 'slab'
            if line.startswith(' * TWO DIMENSIONAL SLAB'):
                calc_type = 'slab'
    f.close()
    atom_data = atom_data[-num_atoms:]
    a,b,c,alpha,beta,gamma = cell_params.split()

    # Write CIF file
    with open(output_file, 'w') as f:
        f.write('data_'+str(input_file)+'\n')
        f.write('\n')
        f.write('_cell_length_a {}\n'.format(a))
        f.write('_cell_length_b {}\n'.format(b))
        if calc_type == 'bulk':
            f.write('_cell_length_c {}\n'.format(c))
            f.write('_cell_angle_alpha {}\n'.format(alpha))
            f.write('_cell_angle_beta {}\n'.format(beta))
            f.write('_cell_angle_gamma {}\n'.format(gamma))
            f.write("_symmetry_space_group_name_H-M         'P 1'\n")
            f.write("_symmetry_Int_Tables_number            1\n")
            f.write('\n')
            f.write('loop_\n')
            f.write('_symmetry_equiv_pos_as_xyz\n')
            f.write("   'x, y, z'\n")
            f.write('\n')
            f.write('loop_\n')
            f.write('_atom_site_label\n')
            f.write('_atom_site_type_symbol\n')
            f.write('_atom_site_fract_x\n')
            f.write('_atom_site_fract_y\n')
            f.write('_atom_site_fract_z\n')
            counter = 0
            for atom in atom_data:
                counter += 1
                f.write('{}{:03d} {} {} {} {}\n'.format(atom[0], counter, atom[0], atom[1], atom[2], atom[3]))
        if calc_type == 'slab':
            f.write('_cell_length_c {}\n'.format(40))
            f.write('_cell_angle_alpha {}\n'.format(alpha))
            f.write('_cell_angle_beta {}\n'.format(beta))
            f.write('_cell_angle_gamma {}\n'.format(gamma))
            f.write("_symmetry_space_group_name_H-M         'P 1'\n")
            f.write("_symmetry_Int_Tables_number            1\n")
            f.write('\n')
            f.write('loop_\n')
            f.write('_symmetry_equiv_pos_as_xyz\n')
            f.write("   'x, y, z'\n")
            f.write('\n')
            f.write('loop_\n')
            f.write('_atom_site_label\n')
            f.write('_atom_site_type_symbol\n')
            f.write('_atom_site_fract_x\n')
            f.write('_atom_site_fract_y\n')
            f.write('_atom_site_fract_z\n')
            counter = 0
            for atom in atom_data:
                counter += 1
                f.write('{}{:03d} {} {} {} {}\n'.format(atom[0], counter, atom[0], atom[1], atom[2], str(float(atom[3])/40)))

DIR = (os.getcwd()+'/')
nDIR = len(DIR)
ntype = len(".out")
pathlist = glob.glob(DIR+'*.out')

for path in pathlist:
    path_in_str = str(path)
    material = path_in_str[nDIR:-ntype]
    if material == "":break
    crystal17_to_cif(material+'.out',material+'.cif')
