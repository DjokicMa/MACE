"""The CLI questions Setup has no answer rule for, and why.

Every question cli_prompts.py finds in the sources of the CLIs Setup runs must either
match a Setup regex (setup.known_questions(mode) or a field's rules) or be listed here,
keyed by (file name, a stable piece of the question's text as cli_prompts renders it,
f-string fields as "X"), with one of these reasons:

* NOT_REACHABLE_FROM_SETUP: Setup's runs never get there (a flag Setup always passes,
  an answer Setup never gives, a function nothing on Setup's routes calls, a question
  only asked at a terminal: Setup's stdin is never one);
* NO_DEFAULT_SETUP_STOPS: reachable, but Enter keeps nothing (a path, a point, an
  atom): the run stops there and names the question (setup.py, after KNOWN_QUESTIONS);
* AFTER_A_STOP: reachable only by answering a NO_DEFAULT_SETUP_STOPS question first.

test_setup_prompts.py checks the sources against Setup's regexes and this list.
"""
NOT_REACHABLE = "NOT_REACHABLE_FROM_SETUP"
NO_DEFAULT = "NO_DEFAULT_SETUP_STOPS"
AFTER_A_STOP = "AFTER_A_STOP"

UNANSWERED = {
    # ---------------------------------------------------------------- not reachable from Setup
    # cif2d12 (NewCifToD12.py)
    ("NewCifToD12.py", "Select method (default"): (NOT_REACHABLE, "select_method(): not called (the menus are d12_interactive's)"),
    ("NewCifToD12.py", "Select X (default"): (NOT_REACHABLE, "select_dft_functional(): not called"),
    # opt2d12 (CRYSTALOptToD12.py): Setup passes one --out-file and --calc-type, no --config-file, and stdin is not a terminal
    ("CRYSTALOptToD12.py", "Apply these settings from config file?"): (NOT_REACHABLE, "--config-file only, and only at a terminal"),
    ("CRYSTALOptToD12.py", "Enter CRYSTAL output file path"): (NOT_REACHABLE, "several/no --out-file only (Setup passes one)"),
    ("CRYSTALOptToD12.py", "Apply to all X files?"): (NOT_REACHABLE, "several files with --config-file only"),
    ("CRYSTALOptToD12.py", "Use shared settings for all files?"): (NOT_REACHABLE, "several files only"),
    ("CRYSTALOptToD12.py", "is not empty. Write the set into it?"): (NOT_REACHABLE, "PHONOPY into a set folder that has files: Setup's run writes into a fresh --output-dir, and its saved command passes --yes (refused with a message, never asked)"),
    ("CRYSTALOptToD12.py", "SMEAR width in hartree for this SOC deck"): (NOT_REACHABLE, "SOC deck from a metallic-looking parent: asked only at a terminal"),
    # d12_interactive.py
    ("d12_interactive.py", "TOLINTEG [X]"): (NOT_REACHABLE, "configure_tolerances_with_defaults(): not called"),
    ("d12_interactive.py", "TOLDEE [X]"): (NOT_REACHABLE, "configure_tolerances_with_defaults(): not called"),
    ("d12_interactive.py", "Enter temperature in Kelvin"): (NOT_REACHABLE, "smearing width option 2: the smear_width field always chooses Custom value"),
    ("d12_interactive.py", "Use these exact settings for the new calculation?"): (NOT_REACHABLE, "cif2d12: opt2d12's follow-up only (a gate there)"),
    ("d12_interactive.py", "Keep the basis set from the original D12?"): (NOT_REACHABLE, "cif2d12: opt2d12's follow-up only (a field there)"),
    ("d12_interactive.py", "Use these extracted settings as baseline for shared configuration?"): (NOT_REACHABLE, "shared mode: several files only"),
    ("d12_interactive.py", "Change optimization settings?"): (NOT_REACHABLE, "shared mode: several files only"),
    ("d12_interactive.py", "Change method/functional?"): (NOT_REACHABLE, "shared mode: several files only"),
    ("d12_interactive.py", "Change basis set?"): (NOT_REACHABLE, "shared mode: several files only"),
    ("d12_interactive.py", "Change DFT grid?"): (NOT_REACHABLE, "shared mode: several files only"),
    ("d12_interactive.py", "Change tolerances?"): (NOT_REACHABLE, "shared mode: several files only"),
    ("d12_interactive.py", "Change advanced electronic and convergence settings?"): (NOT_REACHABLE, "shared mode: several files only"),
    ("d12_interactive.py", "Use these current advanced settings?"): (NOT_REACHABLE, "show_current=True: no caller passes it"),
    ("d12_interactive.py", "Use these default advanced settings?"): (NOT_REACHABLE, "opt2d12: get_calculation_options_new() is cif2d12's (a gate there)"),
    ("d12_interactive.py", "Enter another functional, 'm' to choose"): (NOT_REACHABLE, "cif2d12: no parent functional to replace (opt2d12 knows it)"),
    ("d12_interactive.py", "Select calculation type (default"): (NOT_REACHABLE, "opt2d12: Setup passes --calc-type (cif2d12: a field)"),
    ("d12_interactive.py", "Do you want to use these default settings?"): (NOT_REACHABLE, "opt2d12: cif2d12's questions (a gate there)"),
    ("d12_interactive.py", "Select system dimensionality (default"): (NOT_REACHABLE, "opt2d12: cif2d12's questions (a field there)"),
    ("d12_interactive.py", "Use these default symmetry settings?"): (NOT_REACHABLE, "opt2d12: shared mode only (cif2d12: a gate)"),
    ("d12_interactive.py", "Select symmetry handling (default"): (NOT_REACHABLE, "opt2d12: shared mode only (cif2d12: a field)"),
    ("d12_interactive.py", "Atom writing options (default"): (NOT_REACHABLE, "opt2d12: shared mode only (cif2d12: known)"),
    ("d12_interactive.py", "Validate symmetry operations?"): (NOT_REACHABLE, "opt2d12: shared mode only (cif2d12: known)"),
    ("d12_interactive.py", "Trigonal axes preference (default"): (NOT_REACHABLE, "opt2d12: shared mode only (cif2d12: known)"),
    ("d12_interactive.py", "Select origin choice (default"): (NOT_REACHABLE, "opt2d12: shared mode only (cif2d12: known)"),
    # interactive_d12_configuration() and what only it calls: not called by either CLI
    ("d12_interactive.py", "Configure print options?"): (NOT_REACHABLE, "interactive_d12_configuration(): not called"),
    ("d12_interactive.py", "Enter choices (e.g., 1 3 5)"): (NOT_REACHABLE, "configure_print_options(): only from interactive_d12_configuration()"),
    ("d12_interactive.py", "Add ghost atoms?"): (NOT_REACHABLE, "interactive_d12_configuration(): not called"),
    ("d12_interactive.py", "Apply external pressure?"): (NOT_REACHABLE, "interactive_d12_configuration(): not called"),
    ("d12_interactive.py", "Save these options to file?"): (NOT_REACHABLE, "interactive_d12_configuration(): not called"),
    ("d12_interactive.py", "Enter filename (default: d12_options.json)"): (NOT_REACHABLE, "interactive_d12_configuration(): not called"),
    ("d12_interactive.py", "Ghost atom configuration"): (NOT_REACHABLE, "configure_ghost_atoms(): only from interactive_d12_configuration()"),
    ("d12_interactive.py", "Distance above surface"): (NOT_REACHABLE, "configure_ghost_atoms(): only from interactive_d12_configuration()"),
    ("d12_interactive.py", "Position (x y z)"): (NOT_REACHABLE, "configure_ghost_atoms(): only from interactive_d12_configuration()"),
    ("d12_interactive.py", "Atom numbers to convert"): (NOT_REACHABLE, "configure_ghost_atoms(): only from interactive_d12_configuration()"),
    ("d12_interactive.py", "Pressure in GPa"): (NOT_REACHABLE, "configure_external_pressure(): only from interactive_d12_configuration()"),
    # d12_calc_basic.py
    ("d12_calc_basic.py", "Use tight convergence for SP calculation?"): (NOT_REACHABLE, "configure_single_point(): imported, not called"),
    ("d12_calc_basic.py", "Continue with geometry optimization?"): (NOT_REACHABLE, "is_already_optimized=True: no caller passes it"),
    # d12_constants.py: superseded by d12_interactive's *_with_defaults wrappers
    ("d12_constants.py", "Select SCF mixing scheme"): (NOT_REACHABLE, "configure_scf_settings(): not called"),
    ("d12_constants.py", "Shift value in Hartree"): (NOT_REACHABLE, "configure_scf_settings(): not called"),
    ("d12_constants.py", "Lock cycles [20]"): (NOT_REACHABLE, "configure_scf_settings(): not called"),
    ("d12_constants.py", "Use SMEAR (Fermi smearing"): (NOT_REACHABLE, "configure_scf_settings(): not called"),
    ("d12_constants.py", "SMEAR value in Hartree"): (NOT_REACHABLE, "configure_scf_settings(): not called"),
    ("d12_constants.py", "Select D3 variant"): (NOT_REACHABLE, "configure_dispersion(): not called"),
    ("d12_constants.py", "Enable Fermi smearing?"): (NOT_REACHABLE, "configure_smearing(): not called"),
    ("d12_constants.py", "Smearing width [0.005]"): (NOT_REACHABLE, "configure_smearing(): not called"),
    # opt2d3 (CRYSTALOptToD3.py): Setup passes -i (one .out), -t and --save-config
    ("CRYSTALOptToD3.py", "Select type (1-7)"): (NOT_REACHABLE, "no -t only (Setup passes it)"),
    ("CRYSTALOptToD3.py", "Use same configuration for all files?"): (NOT_REACHABLE, "a directory of outputs only"),
    ("CRYSTALOptToD3.py", "Save this configuration for future use?"): (NOT_REACHABLE, "--save-config answers it / directory only"),
    ("CRYSTALOptToD3.py", "Enter CRYSTAL output file path"): (NOT_REACHABLE, "no -i only (Setup passes it)"),

    # ---------------------------------------------------------------- reachable, no default: Setup stops
    ("NewCifToD12.py", "Please enter the space group number"): (NO_DEFAULT, "a CIF naming no space group"),
    ("d12_interactive.py", "needs one record"): (NO_DEFAULT, "opt2d12: LSRSH-PBE typed as the replacement functional"),
    ("d12_constants.py", "Basis set directory path"): (NO_DEFAULT, "external basis set 3: Custom path"),
    ("d12_calc_freq.py", "Path: "): (NO_DEFAULT, "FREQ band path 3: Custom labels (Enter falls back to G X M G, whatever the cell)"),
    ("d12_calc_freq.py", "Number of path segments"): (NO_DEFAULT, "FREQ band path 4: Fractional coordinates"),
    ("d12_calc_freq.py", "(3 integers)"): (NO_DEFAULT, "FREQ dispersion supercell 3: transformation matrix rows"),
    ("d12_calc_freq.py", "Tensor values"): (NO_DEFAULT, "FREQ dispersion, polar: the dielectric tensor"),
    ("d12_calc_freq.py", "Select method (1-4): "): (NO_DEFAULT, "FREQ anharmonic method: no default offered"),
    ("d12_calc_freq.py", "H/D atom to displace"): (NO_DEFAULT, "ANHARM's H atom (Enter falls back to atom 1)"),
    ("CRYSTALOptToD3.py", "min (bohr)"): (NO_DEFAULT, "CHARGE/POTENTIAL RANGE for non-periodic directions"),
    ("CRYSTALOptToD3.py", "max (bohr)"): (NO_DEFAULT, "CHARGE/POTENTIAL RANGE for non-periodic directions"),
    ("CRYSTALOptToD3.py", "Point X (x y z)"): (NO_DEFAULT, "ECHG map plane points, POTC custom points"),
    ("d3_interactive.py", "Path: "): (NO_DEFAULT, "BAND path 3: Custom labels (Enter falls back to GAMMA X M GAMMA)"),
    ("d3_interactive.py", "Number of path segments"): (NO_DEFAULT, "BAND path 4: coordinates/mixed segments"),
    ("d3_interactive.py", "Enter atom numbers (e.g., 1 3 5)"): (NO_DEFAULT, "DOSS atom selection 2 (Enter falls back to atom 1)"),
    ("d3_interactive.py", "Enter projection specification"): (NO_DEFAULT, "DOSS manual projections"),
    ("d3_interactive.py", "Enter atom numbers: "): (NO_DEFAULT, "POTC at chosen atoms (Calculate at all atoms? No)"),
    ("d3_interactive.py", "Minimum μ (eV, absolute)"): (NO_DEFAULT, "TRANSPORT from an output with no trustworthy E_F "
                                                               "(a spin-locked SCF with no BAND or DOSS run beside it): "
                                                               "absolute chemical potentials, never a guessed -2..+2 eV"),
    ("d3_interactive.py", "Maximum μ (eV, absolute)"): (AFTER_A_STOP, "after Minimum μ (eV, absolute)"),

    # ---------------------------------------------------------------- reachable only past a stop
    ("d12_constants.py", "Try again? [Y/n]"): (AFTER_A_STOP, "after the custom basis directory path"),
    ("d12_constants.py", "Use this path anyway?"): (AFTER_A_STOP, "after the custom basis directory path"),
    ("d12_calc_freq.py", "Use fractional coordinates"): (AFTER_A_STOP, "after Number of path segments"),
    ("d12_calc_freq.py", "Enter label pair"): (AFTER_A_STOP, "after Number of path segments"),
    ("d12_calc_freq.py", "Start point (3 values)"): (AFTER_A_STOP, "after Number of path segments"),
    ("d12_calc_freq.py", "End point (3 values)"): (AFTER_A_STOP, "after Number of path segments"),
    ("d12_calc_freq.py", "Number of modes to scan"): (AFTER_A_STOP, "after the anharmonic method"),
    ("d12_calc_freq.py", "Mode number X"): (AFTER_A_STOP, "after the anharmonic method"),
    ("d12_calc_freq.py", "VSCF level"): (AFTER_A_STOP, "after the anharmonic method"),
    ("d12_calc_freq.py", "VCI level"): (AFTER_A_STOP, "after the anharmonic method"),
    ("d12_calc_freq.py", "Maximum quanta sum"): (AFTER_A_STOP, "after the anharmonic method"),
    ("d12_calc_freq.py", "Number of isotopic substitutions"): (AFTER_A_STOP, "after ANHARM's H atom"),
    ("d12_calc_freq.py", "Atom label for substitution"): (AFTER_A_STOP, "after ANHARM's H atom"),
    ("d12_calc_freq.py", "New mass (amu)"): (AFTER_A_STOP, "after ANHARM's H atom"),
    ("d3_interactive.py", "Type (L)abels or (C)oordinates"): (AFTER_A_STOP, "after Number of path segments"),
    ("d3_interactive.py", "Enter two labels"): (AFTER_A_STOP, "after Number of path segments"),
    ("d3_interactive.py", "Six numbers (x1 y1 z1 x2 y2 z2)"): (AFTER_A_STOP, "after Number of path segments"),
    ("d3_interactive.py", "Segment X (e.g., 'G X')"): (AFTER_A_STOP, "after Number of path segments"),
    ("d3_interactive.py", "Shrink factor for k-points [X]"): (AFTER_A_STOP, "after Number of path segments"),
    ("d3_interactive.py", "fractional coordinates (will be scaled by shrink factor)"): (AFTER_A_STOP, "after Number of path segments"),
}
