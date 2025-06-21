#!/usr/bin/env python
import os, sys, math
import re
import linecache
from collections import OrderedDict

"""
This script is meant to get all the shells from all atoms and separate the shells in terms of S,P,D,SP, and F shells. Note that this script shells for each element but does not pick individual atoms
"""


"""Finds the shells in the output file and then sends them as a list to be organized"""


def get_unclean_shells(output_lines):
    index_shells = 0
    for line in output_lines:
        if "LOCAL ATOMIC FUNCTIONS BASIS SET" in line:
            index_shells = output_lines.index(line)
            break
    unclean_shells = []
    for index in range(index_shells + 4, len(output_lines)):
        if "PROCESS" in output_lines[index]:
            continue
        if (
            "*******" in output_lines[index]
            or " INFORMATION " in output_lines[index]
            or output_lines[index] == "\n"
        ):
            break
        else:
            unclean_shells.append(output_lines[index].split("\n")[0])
    data_list = []
    for line in unclean_shells:
        data_list.append([x for x in line.split(" ") if x is not ""])
    return data_list


"""Finds the line with the total number of orbitals and returns it"""


def sym(input_lines, output_lines):
    orbital_num = 0
    for line in output_lines:
        if "NUMBER OF AO" in line:
            clean_line = []
            split_line = line.split(" ")
            for string in split_line:
                if string != "":
                    clean_line.append(string)
            orbital_num = int(clean_line[3])
            break
    return orbital_num


"""Finds the atoms and their index and returns them"""


def shell_data(data_list):
    # Compiles a list of all the atoms in order in the output
    atoms = []
    # Compiles all unique atoms in order to generate list for shells
    atom_list = []
    atoms_index = []
    for line in data_list:
        if len(line) == 5:
            atoms.append(line[1])
            if line[1] not in atom_list:
                atom_and_index = []
                atom_index = data_list.index(line)
                atom_list.append(line[1])
                atom_and_index.append(line[1])
                atom_and_index.append(atom_index)
                atoms_index.append(atom_and_index)
    return atoms, atoms_index


"""Separates the shells and cycles through to index them"""


def number_shells(atoms_index, data_list):
    S, P, SP, D, F = 0, 0, 0, 0, 0
    atoms_shells = {}
    for atom in atoms_index:
        for i in range(int(atom[1]) + 1, len(data_list)):
            if len(data_list[i]) >= 5:
                break
            elif "S" in data_list[i][-1]:
                S += 1
            elif "SP" in data_list[i][-1]:
                if len(data_list[i]) == 2:
                    SP_shells = data_list[i][0].split("-")
                    for j in range(int(SP_shells[0]), int(SP_shells[1]) + 1):
                        SP += 1
                elif len(data_list[i]) == 3:
                    for j in range(
                        int(data_list[i][0].split("-")[0]), int(data_list[i][1]) + 1
                    ):
                        SP += 1
            elif "P" in data_list[i][-1]:
                if len(data_list[i]) == 2:
                    P_shells = data_list[i][0].split("-")
                    for j in range(int(P_shells[0]), int(P_shells[1]) + 1):
                        P += 1
                elif len(data_list[i]) == 3:
                    for j in range(
                        int(data_list[i][0].split("-")[0]), int(data_list[i][1]) + 1
                    ):
                        P += 1
            elif "D" in data_list[i][-1]:
                if len(data_list[i]) == 2:
                    D_shells = data_list[i][0].split("-")
                    for j in range(int(D_shells[0]), int(D_shells[1]) + 1):
                        D += 1
                elif len(data_list[i]) == 3:
                    for j in range(
                        int(data_list[i][0].split("-")[0]), int(data_list[i][1]) + 1
                    ):
                        D += 1
            elif "F" in data_list[i][-1]:
                if len(data_list[i]) == 2:
                    F_shells = data_list[i][0].split("-")
                    for j in range(int(F_shells[0]), int(F_shells[1]) + 1):
                        F += 1
                elif len(data_list[i]) == 3:
                    for j in range(
                        int(data_list[i][0].split("-")[0]), int(data_list[i][1]) + 1
                    ):
                        F += 1
        atoms_shells[atom[0]] = {"S": S, "SP": SP, "P": P, "D": D, "F": F}
        S, P, SP, D, F = 0, 0, 0, 0, 0
    return atoms_shells


"""Takes the shells and the number of each type of shell and associates them with the indexes in the output """


def get_shells(atoms, atoms_shells):
    AOindex = 1
    total_shells = {}
    for element in atoms_shells:
        total_shells[element] = {"S": [], "P": [], "D": [], "F": [], "SP": []}
    for atom in atoms:
        for shell in range(0, int(atoms_shells[atom]["S"])):
            total_shells[atom]["S"].append(AOindex)
            AOindex += 1
        for shell in range(0, int(atoms_shells[atom]["SP"])):
            total_shells[atom]["SP"].append(AOindex)
            AOindex += 1
        for shell in range(0, int(atoms_shells[atom]["P"])):
            total_shells[atom]["P"].append(AOindex)
            AOindex += 1
        for shell in range(0, int(atoms_shells[atom]["D"])):
            total_shells[atom]["D"].append(AOindex)
            AOindex += 1
        for shell in range(0, int(atoms_shells[atom]["F"])):
            total_shells[atom]["F"].append(AOindex)
            AOindex += 1
    return total_shells


"""Writes the d3 file with the shells written in the total_shells list"""


def write_d3(d3_file, total_shells, orbital_num, newk):
    # ERROR WITH THE WRITING OF A FILE
    DOSS_lines = []
    for atom in total_shells:
        atom_total_lines = " "
        total_shell_number = 0
        S_line, P_line, SP_line, D_line, F_line = "", "", "", "", ""
        for shell in total_shells[atom]["S"]:
            S_line = S_line + " " + str(shell) + " "
        for shell in total_shells[atom]["SP"]:
            SP_line = SP_line + " " + str(shell) + " "
        for shell in total_shells[atom]["P"]:
            P_line = P_line + " " + str(shell) + " "
        for shell in total_shells[atom]["D"]:
            D_line = D_line + " " + str(shell) + " "
        for shell in total_shells[atom]["F"]:
            F_line = F_line + " " + str(shell) + " "
        if len(total_shells[atom]["S"]) != 0:
            DOSS_lines.append(
                str(len(total_shells[atom]["S"])) + " " + S_line + " #" + atom + " S \n"
            )
            atom_total_lines = atom_total_lines + " " + S_line
            total_shell_number = total_shell_number + len(total_shells[atom]["S"])
        if len(total_shells[atom]["SP"]) != 0:
            DOSS_lines.append(
                str(len(total_shells[atom]["SP"]))
                + " "
                + SP_line
                + " #"
                + atom
                + " SP \n"
            )
            atom_total_lines = atom_total_lines + " " + SP_line
            total_shell_number = total_shell_number + len(total_shells[atom]["SP"])
        if len(total_shells[atom]["P"]) != 0:
            DOSS_lines.append(
                str(len(total_shells[atom]["P"])) + " " + P_line + " #" + atom + " P \n"
            )
            atom_total_lines = atom_total_lines + " " + P_line
            total_shell_number = total_shell_number + len(total_shells[atom]["P"])
        if len(total_shells[atom]["D"]) != 0:
            DOSS_lines.append(
                str(len(total_shells[atom]["D"])) + " " + D_line + " #" + atom + " D \n"
            )
            atom_total_lines = atom_total_lines + " " + D_line
            total_shell_number = total_shell_number + len(total_shells[atom]["D"])
        if len(total_shells[atom]["F"]) != 0:
            DOSS_lines.append(
                str(len(total_shells[atom]["F"])) + " " + F_line + " #" + atom + " F \n"
            )
            atom_total_lines = atom_total_lines + " " + F_line
            total_shell_number = total_shell_number + len(total_shells[atom]["F"])
        DOSS_lines.append(
            str(total_shell_number) + " " + atom_total_lines + " #" + atom + " all \n"
        )
    DOSS_lines.append("END\n")
    # write d3 files
    input_line = (
        str(len(DOSS_lines) - 1)
        + " "
        + "10000 -1 "
        + "-1"
        + " "
        + "1 14"
        + " "
        + "0\n-1 2\n"
    )
    d3_file.write("NEWK\n")
    if len(newk) == 0:
        d3_file.write("48 48\n")
    else:
        for lines in newk:
            d3_file.write(lines)
    # d3_file.write("48 48\n")
    d3_file.write("1 0\n")
    d3_file.write("DOSS\n")
    d3_file.write(input_line)
    for line in DOSS_lines:
        d3_file.write(line)


""" NEED TO REMOVE LINES FROM DOSS_lines WHICH HAS 0 len (just split line and if == '0' dont print)"""


"""BEGIN SCRIPT"""
"""-----------------------------------------------"""
"""Change the data_folder depending on where your files are"""
data_folder = os.getcwd()
data_files = os.listdir(data_folder)

for file_name in data_files:
    if ".d12" in file_name:
        input_file_name = os.path.join(data_folder, file_name)
        input_file = open(input_file_name, "r+")
        input_lines = []
        newk = []
        newk_counter = 0
        for line in input_file.readlines():
            if "\n" in line:
                clean_line = line.replace("\n", "")
                input_lines.append(clean_line)
            else:
                input_lines.append(line)
            if newk_counter == 1:
                if "0" in line:
                    newk.append(line)
                else:
                    newk_counter = 0
                    newk.append(line)
            elif "SHRINK" in line:
                newk_counter = 1
                newk = []
        output_name = file_name.replace(".d12", ".out")
        output_file_name = os.path.join(data_folder, output_name)
        output_file = open(output_file_name, "r+")
        output_lines = []
        for line in output_file.readlines():
            if "\n" in line:
                clean_line = line.replace("\n", "")
                output_lines.append(clean_line)
            else:
                output_lines.append(line)
        d3_name = file_name.replace(".d12", "_DOSS.d3")
        d3_file_name = os.path.join(data_folder, d3_name)
        d3_file = open(d3_file_name, "w+")
        # Finds all shells, uncleaned for processing
        data_list = get_unclean_shells(output_lines)
        # Finds the bands which contribute to band gap, saves alpha and beta bands
        # Note that these are lists as the bands could potentially be different
        orbital_num = sym(input_lines, output_lines)

        # Returns list of all the atoms, and a list of the unique atoms and their index
        atoms, atoms_index = shell_data(data_list)
        atoms_shells = number_shells(atoms_index, data_list)
        total_shells = get_shells(atoms, atoms_shells)
        write_d3(d3_file, total_shells, orbital_num, newk)
    if ".f9" in file_name:
        input_file_name = os.path.join(data_folder, file_name)
        os.rename(input_file_name, os.path.splitext(input_file_name)[0] + "_DOSS.f9")
