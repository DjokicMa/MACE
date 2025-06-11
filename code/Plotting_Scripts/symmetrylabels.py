def crystal_system(sgno,sgnm):
    if sgno == 1:
        crysys == "Triclinic"

    elif sgno == 2:
        crysys = "Triclinic"

    elif sgno >= 3 and sgno < 16:

        if sgnm == "P":
          crysys = "Monoclinic_Simple"

        if sgnm == "C":
          crysys = "Monoclinic_AC"

    elif sgno >= 16 and sgno < 75:
        if sgnm == "P":
          crysys = "Orthorhombic_Simple"

        if sgnm == "C":
          crysys = "Orthorhombic_AB"

        if sgnm == "F":
          crysys = "Orthorhombic_FC"

        if sgnm == "I":
          crysys = "Orthorhombic_BC"

        if sgnm == "A":
          crysys = "Orthorhombic_AB"

    elif sgno >= 75  and sgno < 143:
        if sgnm == "I":
          crysys = "Tetragonal"

        if sgnm == "P":
          crysys = "Tetragonal_Simple"

    elif sgno >= 143 and sgno < 168:
        if sgnm == "P":
           crysys = "Hexagonal"

        if sgnm == "R":
          crysys = "Rhombohedral"

    elif sgno >= 168 and sgno < 195:
        crysys = "Hexagonal"

    elif sgno >= 195:
        if sgnm == "P":
           crysys = "Cubic_Simple"
        if sgnm == "F":
           crysys = "Cubic_FC"
        if sgnm == "I":
           crysys = "Cubic_BC"
    return(crysys)


def get_band_labels(crysys):
    if crysys == "Cubic_FC":
       labels = ["M",r"$\Gamma$","R","X",r"$\Gamma$"]

    elif crysys == "Cubic_BC":
       labels = ["H",r"$\Gamma$","P","N",r"$\Gamma$"]

    elif crysys == "Cubic_Simple":
       labels = ["M",r"$\Gamma$","R","X",r"$\Gamma$"]

    elif crysys == "Hexagonal":
       labels = ["M",r"$\Gamma$","K","A",r"$\Gamma$","L","H",r"$\Gamma$"]

    elif crysys == "Monoclinic_AC":
       labels = ["A",r"$\Gamma$","Y","M",r"$\Gamma$"]

    elif crysys == "Monoclinic_Simple":
       labels = ["A",r"$\Gamma$","B","C",r"$\Gamma$","D","E",r"$\Gamma$","Y","Z",r"$\Gamma$"]

    elif crysys == "Orthorhombic_AB":
       labels = ["S",r"$\Gamma$","T","R",r"$\Gamma$","Y","Z",r"$\Gamma$"]

    elif crysys == "Orthorhombic_BC":
       labels = ["S",r"$\Gamma$","T","R",r"$\Gamma$","X","W",r"$\Gamma$"]

    elif crysys == "Orthorhombic_FC":
       labels = ["Z",r"$\Gamma$","Y","T",r"$\Gamma$"]

    elif crysys == "Orthorhombic_Simple":
       labels = ["S",r"$\Gamma$","T","U",r"$\Gamma$","R","X",r"$\Gamma$","Y","Z",r"$\Gamma$"]

    elif crysys == "Rhombohedral":
       labels = ["T",r"$\Gamma$","F","L",r"$\Gamma$"]

    elif crysys == "Tetragonal":
       labels = ["M",r"$\Gamma$","P","X",r"$\Gamma$"]

    elif crysys == "Tetragonal_Simple":
       labels = ["M",r"$\Gamma$","R","A",r"$\Gamma$","X","Z",r"$\Gamma$"]

    elif crysys == "Triclinic":
       labels = ["V","Y",r"$\Gamma$","Z","T","R",r"$\Gamma$","X","U",r"$\Gamma$"]
    else:
       labels = ["V","Y",r"$\Gamma$","Z","T","R",r"$\Gamma$","X","U",r"$\Gamma$"]
    return(labels)
