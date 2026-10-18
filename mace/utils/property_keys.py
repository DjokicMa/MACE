"""The database's property keys: one registry of every name the property
extractor writes, with its category, unit, value type and meaning.

``mace analyze --extract-properties`` files each row under ``category`` and
labels it with ``unit`` from here (it used to guess both from substrings of
the name). A name the extractor writes that is missing here is a bug that
tests/test_property_keys.py catches.

Renamed and merged keys (MACE 1.1.6) stay readable under their old names
for one release: ``ALIASES`` maps each old name to the one it became, and
the read paths (exports, ``mace database`` queries and filters, the GUI's
Database view) look both up through :func:`query_names` and
:func:`normalize_rows`. Values of a database built before 1.1.6 keep their
old names until the outputs are extracted again; those rows are shown under
the new names. The old names are dropped in MACE 1.1.7.

Standard library only: the GUI imports it without the mace.database package.
"""
from __future__ import annotations

import sys
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence

#: how long the old names stay readable
ALIASES_UNTIL = "1.1.7"


class Key(NamedTuple):
    category: str
    unit: str
    type: str          # float, int, bool, text, json
    description: str


CATEGORIES = (
    'lattice', 'structural', 'crystallographic', 'energy', 'electronic',
    'band_structure', 'density_of_states', 'population_analysis', 'optimization',
    'frequency', 'thermodynamics', 'transport', 'charge_potential', 'orbitals',
    'scf_settings', 'computational',
)

_AXES = ('xx', 'yy', 'zz')


def _three(prefix: str, category: str, what: str, au: bool = True, ev: bool = True,
           kj: bool = False) -> Dict[str, Key]:
    out = {}
    if au:
        out[f'{prefix}_au'] = Key(category, 'Hartree', 'float', what)
    if ev:
        out[f'{prefix}_ev'] = Key(category, 'eV', 'float', what)
    if kj:
        out[f'{prefix}_kj_mol'] = Key(category, 'kJ/mol', 'float', what)
    return out


def _build() -> Dict[str, Key]:
    k: Dict[str, Key] = {}

    # ---------------------------------------------------------------- lattice
    cell = {'a': 'Å', 'b': 'Å', 'c': 'Å', 'alpha': 'degrees', 'beta': 'degrees', 'gamma': 'degrees'}
    for pre, what in (('primitive', 'primitive cell'), ('crystallographic', 'conventional cell')):
        for p, unit in cell.items():
            k[f'{pre}_{p}'] = Key('lattice', unit, 'float',
                                  f'{what} {p} of the final geometry (the input geometry when the run does '
                                  f'not optimize); none along a non-periodic direction')
            k[f'initial_{pre}_{p}'] = Key('lattice', unit, 'float', f'{what} {p} of the input geometry')
    k['primitive_cell_volume'] = Key('lattice', 'Å³', 'float', 'primitive cell volume (3D)')
    k['initial_primitive_cell_volume'] = Key('lattice', 'Å³', 'float', 'primitive cell volume of the input geometry')
    k['primitive_cell_area'] = Key('lattice', 'Å²', 'float', 'area of the 2D cell (slab)')
    k['initial_primitive_cell_area'] = Key('lattice', 'Å²', 'float', 'slab cell area of the input geometry')
    k['primitive_cell_length'] = Key('lattice', 'Å', 'float', 'length of the 1D cell (polymer)')
    k['initial_primitive_cell_length'] = Key('lattice', 'Å', 'float', 'polymer cell length of the input geometry')
    k['crystallographic_cell_volume'] = Key('lattice', 'Å³', 'float', 'conventional cell volume')

    # ------------------------------------------------------------- structural
    k['density'] = Key('structural', 'g/cm³', 'float', 'density of the final geometry')
    k['initial_density'] = Key('structural', 'g/cm³', 'float', 'density of the input geometry')
    k['atomic_positions'] = Key('structural', 'coordinates', 'json',
                                'atoms of the final geometry (see atomic_position_axes for the axes)')
    k['initial_atomic_positions'] = Key('structural', 'coordinates', 'json', 'atoms of the input geometry')
    k['atomic_position_axes'] = Key('structural', 'category', 'text',
                                    'the axes of atomic_positions when they are not all fractional, '
                                    'as CRYSTAL prints them (e.g. "X/A Y/B Z(ANGSTROM)")')
    k['atoms_in_unit_cell'] = Key('structural', 'dimensionless', 'int', 'atoms in the primitive cell')
    k['initial_atoms_count'] = Key('structural', 'dimensionless', 'int', 'atoms of the input geometry')
    for pre, what in (('', 'the first'), ('final_', 'the last')):
        k[f'{pre}neighbor_analysis'] = Key('structural', 'JSON', 'json',
                                           f'neighbors of each atom, from {what} NEIGHBORS table')
        k[f'{pre}min_bond_distance_ang'] = Key('structural', 'Å', 'float', f'shortest neighbor distance ({what} table)')
        k[f'{pre}max_bond_distance_ang'] = Key('structural', 'Å', 'float', f'longest neighbor distance ({what} table)')
        k[f'{pre}max_coordination_number'] = Key('structural', 'dimensionless', 'int',
                                                 f'largest neighbor count of a shell ({what} table)')
        k[f'{pre}total_coordination_shells'] = Key('structural', 'dimensionless', 'int',
                                                   f'neighbor shells listed ({what} table)')

    # ------------------------------------------------------- crystallographic
    k['space_group_number'] = Key('crystallographic', 'number', 'int',
                                  'space group (for a slab: the corresponding space group CRYSTAL prints); '
                                  'none for a molecule')
    k['layer_group_number'] = Key('crystallographic', 'number', 'int',
                                  'layer group of a slab ("TWO-SIDED PLANE GROUP N." of the output)')
    k['layer_group_symbol'] = Key('crystallographic', 'category', 'text', 'layer group symbol of a slab')
    k['crystal_system'] = Key('crystallographic', 'category', 'text', 'crystal system')
    k['centering_code'] = Key('crystallographic', 'code', 'text', 'centering code (CENTRING CODE)')  # british-ok: CRYSTAL's word

    # ----------------------------------------------------------------- energy
    k.update(_three('total_energy', 'energy', 'TOTAL ENERGY(DFT): the SCF energy without D3 and gCP'))
    k.update(_three('total_energy_plus_d3', 'energy', 'total energy + D3 (without gCP)'))
    k.update(_three('total_energy_corrected', 'energy', 'total energy + D3 + gCP: the energy to compare'))
    k.update(_three('d3_dispersion_energy', 'energy', 'D3 dispersion correction'))
    k.update(_three('gcp_energy', 'energy', 'gCP correction (3c methods)'))
    for comp, what in (('kinetic', 'kinetic energy'), ('electron_electron', 'total E-E'),
                       ('electron_nuclear', 'total E-N + N-E'), ('nuclear_nuclear', 'total N-N'),
                       ('exchange', 'exchange energy'), ('correlation', 'correlation energy')):
        k.update(_three(f'{comp}_energy', 'energy', f'{what} of the last SCF ("ENERGIES IN A.U.")'))

    # ------------------------------------------------------------- electronic
    k['band_gap'] = Key('electronic', 'eV', 'float',
                        'the gap of the last SCF: the fundamental gap across both spin channels of an '
                        'unrestricted run; 0 for a run that ends conducting')
    k['direct_band_gap'] = Key('electronic', 'eV', 'float', 'band_gap, when it is direct')
    k['indirect_band_gap'] = Key('electronic', 'eV', 'float', 'band_gap, when it is indirect')
    k['band_gap_type'] = Key('electronic', 'category', 'text', 'direct or indirect (of band_gap)')
    k['alpha_band_gap'] = Key('electronic', 'eV', 'float', 'alpha-channel gap of CRYSTAL\'s summary')
    k['beta_band_gap'] = Key('electronic', 'eV', 'float', 'beta-channel gap of CRYSTAL\'s summary')
    k['spin_polarized'] = Key('electronic', 'boolean', 'bool',
                              'an unrestricted (spin-polarized) run, whatever its net spin')
    k['fermi_energy_au'] = Key('electronic', 'Hartree', 'float',
                               'Fermi energy: the last SCF\'s, else the one a BAND or DOSS output prints')
    k['electronic_classification'] = Key('electronic', 'category', 'text',
                                         'metal (gap < 0.01 eV), semiconductor (up to 4 eV) or insulator '
                                         '(above 4 eV), from band_gap, else from the band structure; '
                                         'semimetal needs a band structure and a DOS')
    k['classification_gap_source'] = Key('electronic', 'category', 'text',
                                         'the gap electronic_classification is from: scf (band_gap) or '
                                         'band_structure (band_fundamental_gap_ev)')
    k['magnetic_classification'] = Key('electronic', 'category', 'text',
                                       'spin_polarized (an unrestricted run with a net or atomic spin) '
                                       'or non_magnetic')

    # --------------------------------------------------------- band structure
    k['has_band_structure'] = Key('band_structure', 'boolean', 'bool', 'a BAND output')
    k['total_kpoints'] = Key('band_structure', 'dimensionless', 'int', 'k points along the path (.out)')
    k['band_start'] = Key('band_structure', 'dimensionless', 'int',
                          'first band written, when it is not band 1')
    k['band_end'] = Key('band_structure', 'dimensionless', 'int', 'last band written')
    k['total_bands'] = Key('band_structure', 'dimensionless', 'int', 'bands written')
    k['band_dat_exists'] = Key('band_structure', 'boolean', 'bool', 'the BAND.DAT file is there')
    k['band_dat_size'] = Key('band_structure', 'bytes', 'int', 'BAND.DAT size')
    k['band_dat_num_kpoints'] = Key('band_structure', 'dimensionless', 'int', 'k points per spin in BAND.DAT')
    k['band_dat_num_bands'] = Key('band_structure', 'dimensionless', 'int', 'bands in BAND.DAT')
    k['band_dat_num_spins'] = Key('band_structure', 'dimensionless', 'int', 'spin blocks in BAND.DAT')
    k['band_dat_num_panels'] = Key('band_structure', 'dimensionless', 'int', 'path segments in BAND.DAT')
    k['band_dat_kpath_labels'] = Key('band_structure', 'JSON', 'json', 'BAND.DAT path labels')
    k['band_dat_metallic'] = Key('band_structure', 'boolean', 'bool', 'bands cross E_F (BAND.DAT)')
    k['band_dat_vbm_ev'] = Key('band_structure', 'eV', 'float', 'valence band maximum, from E_F (BAND.DAT)')
    k['band_dat_cbm_ev'] = Key('band_structure', 'eV', 'float', 'conduction band minimum, from E_F (BAND.DAT)')
    k['band_structure'] = Key('band_structure', 'JSON', 'json',
                              'k path, labels, band energies at the high-symmetry points, band edges')
    k['band_fundamental_gap_ev'] = Key('band_structure', 'eV', 'float',
                                       'the gap along the band path: CBM - VBM over every k point and spin')
    k['band_direct_gap_ev'] = Key('band_structure', 'eV', 'float', 'smallest vertical gap along the path')
    k['band_gap_is_direct'] = Key('band_structure', 'boolean', 'bool', 'VBM and CBM at one k point')
    k['band_vbm_kpoint_index'] = Key('band_structure', 'dimensionless', 'int', 'VBM k point (0-based, per spin)')
    k['band_cbm_kpoint_index'] = Key('band_structure', 'dimensionless', 'int', 'CBM k point (0-based, per spin)')
    k['band_kpath_label_string'] = Key('band_structure', 'category', 'text', 'path labels, joined by -')
    k['band_structure_k_points'] = Key('band_structure', 'dimensionless', 'int',
                                       'BAND.DAT rows read by the band analysis (all spins)')
    k['band_structure_k_range_max'] = Key('band_structure', 'bohr⁻¹', 'float', 'length of the k path')
    k['band_file_analyzed'] = Key('band_structure', 'category', 'text', 'the BAND.DAT the band analysis read')
    for carrier in ('electron', 'hole', 'average'):
        k[f'real_{carrier}_effective_mass'] = Key(
            'band_structure', 'm_e', 'float',
            f'{carrier} effective mass from a quadratic fit at the band edge'
            + (' (geometric mean of electron and hole)' if carrier == 'average' else ''))
    k['effective_mass_calculation_method'] = Key('band_structure', 'category', 'text',
                                                 'how the effective masses were found')

    # ------------------------------------------------------ density of states
    k['has_dos'] = Key('density_of_states', 'boolean', 'bool', 'a DOSS output')
    k['doss_dat_exists'] = Key('density_of_states', 'boolean', 'bool', 'the DOSS.DAT file is there')
    k['doss_dat_size'] = Key('density_of_states', 'bytes', 'int', 'DOSS.DAT size')
    k['doss_dat_num_energy_points'] = Key('density_of_states', 'dimensionless', 'int', 'energy points per spin')
    k['doss_dat_num_spins'] = Key('density_of_states', 'dimensionless', 'int', 'spin blocks in DOSS.DAT')
    k['doss_dat_energy_min'] = Key('density_of_states', 'Hartree', 'float', 'lowest DOSS.DAT energy, from E_F')
    k['doss_dat_energy_max'] = Key('density_of_states', 'Hartree', 'float', 'highest DOSS.DAT energy, from E_F')
    k['dos_structure'] = Key('density_of_states', 'JSON', 'json',
                             'downsampled total DOS per spin, band edges, projection weights')
    k['dos_at_fermi_total'] = Key('density_of_states', 'states/eV', 'float', 'DOS at E_F, both spins (per cell)')
    k['dos_at_fermi_alpha'] = Key('density_of_states', 'states/eV', 'float', 'DOS at E_F, alpha channel')
    k['dos_at_fermi_beta'] = Key('density_of_states', 'states/eV', 'float', 'DOS at E_F, beta channel')
    k['dos_band_gap_from_dos_ev'] = Key('density_of_states', 'eV', 'float',
                                        'the empty DOS window around E_F (an estimate, within the grid step)')
    k['dos_is_metallic'] = Key('density_of_states', 'boolean', 'bool', 'states at E_F (DOSS.DAT)')
    k['dos_integrated_states_to_fermi'] = Key('density_of_states', 'electrons', 'float',
                                              'states per cell integrated up to E_F')
    k['dos_num_projections'] = Key('density_of_states', 'dimensionless', 'int', 'projected DOS columns')
    k['dos_threshold'] = Key('density_of_states', 'states/(Hartree·cell)', 'float',
                             'the DOS at E_F above which the band analysis calls a gapless system a metal')
    k['dos_mean'] = Key('density_of_states', 'states/(Hartree·cell)', 'float', 'mean of the nonzero DOS')
    k['doss_file_analyzed'] = Key('density_of_states', 'category', 'text', 'the DOSS.DAT the band analysis read')

    # ---------------------------------------------------- population analysis
    for spin, what in (('alpha_plus_beta', 'alpha+beta (charge)'), ('alpha_minus_beta', 'alpha-beta (spin)')):
        k[f'mulliken_{spin}'] = Key('population_analysis', 'electrons', 'json',
                                    f'Mulliken {what} populations per atom and orbital (mulliken_population '
                                    f'per atom: the electron count, not the net charge)')
        k[f'overlap_population_{spin}'] = Key('population_analysis', 'JSON', 'json',
                                              f'Mulliken {what} overlap populations')
    k['mulliken_population'] = Key('population_analysis', 'electrons', 'json',
                                   'Mulliken populations of a closed-shell run')
    k['overlap_population'] = Key('population_analysis', 'JSON', 'json', 'overlap populations of a closed-shell run')
    k['mulliken_valence_charges'] = Key('population_analysis', 'electrons', 'json',
                                        'valence charge of each pseudopotential element, by Z')
    for name, what in (('atomic_charges', 'net atomic charges'), ('bonding_analysis', 'bonds from overlap populations'),
                       ('coordination_environments', 'coordination of each atom'),
                       ('magnetic_info', 'spin moments'), ('chemical_summary', 'charge and bond summary')):
        k[f'processed_{name}'] = Key('population_analysis', 'JSON', 'json', f'{what} (population analysis)')
    k['electron_count'] = Key('population_analysis', 'JSON', 'json', 'electron count of a charged run')
    k['fragment_charges'] = Key('population_analysis', 'JSON', 'json',
                                'Mulliken charge and spin of each molecular fragment')

    # ------------------------------------------------------------ optimization
    k['optimization_converged'] = Key('optimization', 'boolean', 'bool', 'OPT END - CONVERGED')
    k['optimization_cycles'] = Key('optimization', 'cycles', 'int', 'energy and gradient calculations')
    k['final_gradient_norm'] = Key('optimization', 'Hartree/Bohr', 'float', 'last gradient norm')

    # --------------------------------------------------------------- frequency
    k['has_frequency_data'] = Key('frequency', 'boolean', 'bool', 'a FREQ output')
    k['vibrational_frequencies'] = Key('frequency', 'cm⁻¹', 'json', 'modes: frequency, irrep, IR/Raman activity')
    k['num_vibrational_modes'] = Key('frequency', 'dimensionless', 'int', 'printed mode rows')
    k['num_vibrational_modes_total'] = Key('frequency', 'dimensionless', 'int', 'modes, degenerate ones counted')
    k['num_imaginary_frequencies'] = Key('frequency', 'dimensionless', 'int',
                                         'imaginary modes beyond 20 cm⁻¹, rigid-body modes left out')
    k['lowest_frequency_cm'] = Key('frequency', 'cm⁻¹', 'float',
                                   'lowest vibration, the 3 (molecule: 6, linear: 5) rigid-body modes '
                                   'closest to zero left out; negative for an imaginary mode')
    k['highest_frequency_cm'] = Key('frequency', 'cm⁻¹', 'float', 'highest vibration')
    k['vibrational_temperatures'] = Key('frequency', 'K', 'json', 'vibrational temperature of each mode')
    k['has_force_constants'] = Key('frequency', 'boolean', 'bool', 'a force-constant matrix is printed')
    k['num_displaced_calculations'] = Key('frequency', 'dimensionless', 'int', 'displaced geometries')
    k['symmetry_allowed_dof'] = Key('frequency', 'dimensionless', 'int', 'symmetry-allowed internal degrees of freedom')
    k['has_normal_mode_displacements'] = Key('frequency', 'boolean', 'bool', 'normal modes are printed')
    k['raman_intensities'] = Key('frequency', 'JSON', 'json', 'averaged isotropic Raman intensities')
    k['num_raman_modes'] = Key('frequency', 'dimensionless', 'int', 'modes in the Raman table')

    # ----------------------------------------------------------- thermodynamics
    k['thermodynamic_temperature_k'] = Key('thermodynamics', 'K', 'float', 'temperature of the thermodynamics')
    k['thermodynamic_pressure_mpa'] = Key('thermodynamics', 'MPa', 'float', 'pressure of the thermodynamics')
    for pre, what in (('electronic_energy', 'EL, the electronic energy of the FREQ run'),
                      ('zero_point_energy', 'E0, the zero-point energy'),
                      ('thermal_energy', 'ET, the thermal energy'), ('pv_term', 'PV'),
                      ('entropy_term', 'TS'), ('free_energy_correction', 'ET+PV-TS'),
                      ('gibbs_free_energy', 'G = EL+E0+ET+PV-TS'), ('enthalpy', 'H = G + TS')):
        k.update(_three(pre, 'thermodynamics', what, kj=True))
    for pre, what in (('entropy', 'entropy S'), ('heat_capacity', 'heat capacity C')):
        k[f'{pre}_mhartree_cell_k'] = Key('thermodynamics', 'mHartree/(cell·K)', 'float', what)
        k[f'{pre}_mev_cell_k'] = Key('thermodynamics', 'meV/(cell·K)', 'float', what)
        k[f'{pre}_j_mol_k'] = Key('thermodynamics', 'J/(mol·K)', 'float', what)

    # ---------------------------------------------------------------- transport
    k['has_transport'] = Key('transport', 'boolean', 'bool', 'a TRANSPORT (BOLTZTRA) output')
    k['transport'] = Key('transport', 'JSON', 'json', 'transport summary: peaks, their T and mu, units, files')
    k['transport_data_available'] = Key('transport', 'boolean', 'bool', 'usable SEEBECK.dat data')
    k['transport_n_temperatures'] = Key('transport', 'dimensionless', 'int', 'temperatures in SEEBECK.dat')
    k['transport_n_data_points'] = Key('transport', 'dimensionless', 'int', 'rows in SEEBECK.dat')
    k['transport_seebeck_max_uv_per_k'] = Key('transport', 'μV/K', 'float',
                                              'peak |S| of the averaged tensor diagonal')
    k['transport_carrier_type'] = Key('transport', 'category', 'text', 'p-type or n-type at the |S| peak')
    k['transport_tensor_average'] = Key('transport', 'category', 'text',
                                        'how the *_avg values average the tensors: (xx+yy+zz)/3 of each of '
                                        'S, sigma and kappa0, or (xx+yy)/2 in plane for a slab')
    k['transport_power_factor_max_avg'] = Key('transport', 'W/(m·K²)', 'float',
                                              'peak PF = S²σ of the averaged tensors (an average, not an axis)')
    k['transport_zt_electronic_max_avg'] = Key('transport', 'dimensionless', 'float',
                                               'peak ZTₑ of the averaged tensors (an average, not an axis; '
                                               'electrons only, no lattice κ)')
    k['transport_kappa0_at_zt_max'] = Key('transport', 'W/(m·K)', 'float',
                                          'κ₀ (KAPPA.dat K coefficient) at the transport_zt_electronic_max_avg peak')
    k['transport_kappa_electronic_at_zt_max'] = Key('transport', 'W/(m·K)', 'float',
                                                    'κₑ = κ₀ − TσS² at the transport_zt_electronic_max_avg peak')
    for ax in _AXES:
        k[f'transport_power_factor_max_{ax}'] = Key('transport', 'W/(m·K²)', 'float',
                                                    f'peak PF = S_{ax}²σ_{ax} along {ax}')
        k[f'transport_zt_electronic_max_{ax}'] = Key('transport', 'dimensionless', 'float',
                                                     f'peak ZTₑ along {ax} (electrons only, no lattice κ)')
        k[f'transport_zt_total_max_{ax}'] = Key('transport', 'dimensionless', 'float',
                                                f'peak total ZT along {ax} with the κ_L given')
    k['transport_power_factor_best_axis'] = Key('transport', 'category', 'text',
                                                'the axis of the highest PF (in plane for a slab)')
    k['transport_zt_electronic_best_axis'] = Key('transport', 'category', 'text',
                                                 'the axis of the highest ZTₑ (in plane for a slab)')
    k['transport_zt_total_best_axis'] = Key('transport', 'category', 'text',
                                            'the axis of the highest total ZT (in plane for a slab)')
    k['transport_zt_total'] = Key('transport', 'JSON', 'json', 'total ZT details: formula, κ_L, τ, peaks per axis')
    k['transport_zt_total_max_avg'] = Key('transport', 'dimensionless', 'float',
                                          'peak total ZT = S²σT/(κₑ + κ_L) of the averaged tensors, with the κ_L '
                                          'given to --kappa-lattice')
    k['transport_zt_total_max_temperature_k'] = Key('transport', 'K', 'float', 'T of transport_zt_total_max_avg')
    k['transport_zt_total_max_mu_ev'] = Key('transport', 'eV', 'float', 'mu of transport_zt_total_max_avg')
    k['transport_zt_total_kappa_lattice_at_max'] = Key('transport', 'W/(m·K)', 'float',
                                                       'κ_L at the transport_zt_total_max_avg peak')
    k['transport_zt_total_kappa_electronic_at_max'] = Key('transport', 'W/(m·K)', 'float',
                                                          'κₑ at the transport_zt_total_max_avg peak (at τ)')
    k['transport_zt_total_tau_fs'] = Key('transport', 'fs', 'float', 'the relaxation time the total ZT is at')
    k['transport_zt_total_lattice_source'] = Key('transport', 'category', 'text', 'where κ_L came from')
    k['transport_zt_total_tau_source'] = Key('transport', 'category', 'text', 'how τ was chosen')
    k['transport_zt_total_inputs'] = Key('transport', 'category', 'text',
                                         'fingerprint of the .out and .dat files the total ZT is from')

    # --------------------------------------------------------- charge+potential
    k['charge_potential'] = Key('charge_potential', 'JSON', 'json', 'ECH3/POT3 grid metadata and files')
    k['chargepot_has_ech3'] = Key('charge_potential', 'boolean', 'bool', 'a charge density grid (ECH3)')
    k['chargepot_has_pot3'] = Key('charge_potential', 'boolean', 'bool', 'a potential grid (POT3)')
    k['chargepot_grid_dims'] = Key('charge_potential', 'category', 'text', 'grid points along each axis')
    k['chargepot_grid_points'] = Key('charge_potential', 'dimensionless', 'int', 'grid points')
    k['chargepot_computed_properties'] = Key('charge_potential', 'category', 'text', 'what the grids hold')
    k['chargepot_grid_files'] = Key('charge_potential', 'category', 'text', 'the .CUBE files')
    k['chargepot_num_grid_files'] = Key('charge_potential', 'dimensionless', 'int', 'number of .CUBE files')
    k['chargepot_scf_total_energy_au'] = Key('charge_potential', 'Hartree', 'float', 'SCF energy the grids are of')

    # ----------------------------------------------------------------- orbitals
    k['orbitals'] = Key('orbitals', 'JSON', 'json', 'the ORBITALS index: files, k points, levels, HOMO/LUMO')
    for name, unit, typ, what in (
            ('orbitals_num_files', 'dimensionless', 'int', 'Molden files'),
            ('orbitals_total_bytes', 'bytes', 'int', 'size of the Molden files'),
            ('orbitals_files', 'category', 'text', 'the Molden files'),
            ('orbitals_num_kpoints', 'dimensionless', 'int', 'k points written'),
            ('orbitals_usable_kpoints', 'dimensionless', 'int', 'k points MACE can draw'),
            ('orbitals_nbf', 'dimensionless', 'int', 'basis functions'),
            ('orbitals_gap_ev', 'eV', 'float', 'gap of the .out\'s band edges'),
            ('orbitals_dimensionality', 'dimensionless', 'int', 'dimensionality of the run'),
            ('orbitals_kpoints', 'category', 'text', 'k points, their coordinates and kind'),
            ('orbitals_self_check', 'category', 'text', 'the worst Molden self-check'),
            ('orbitals_homo_band', 'dimensionless', 'int', 'HOMO level'),
            ('orbitals_homo_energy_au', 'Hartree', 'float', 'HOMO energy'),
            ('orbitals_lumo_band', 'dimensionless', 'int', 'level above the HOMO at its k point and spin'),
            ('orbitals_lumo_energy_au', 'Hartree', 'float', 'energy of orbitals_lumo_band'),
            ('orbitals_spin_polarized', 'boolean', 'bool', 'the Molden files hold alpha and beta orbitals')):
        k[name] = Key('orbitals', unit, typ, what)

    # ------------------------------------------------------------ scf settings
    for name, unit, typ, what in (
            ('scf_biposize', 'dimensionless', 'int', 'BIPOSIZE'), ('scf_exchsize', 'dimensionless', 'int', 'EXCHSIZE'),
            ('scf_maxcycle', 'cycles', 'int', 'MAXCYCLE'), ('scf_ppan', 'boolean', 'bool', 'PPAN'),
            ('scf_toldee', 'dimensionless', 'int', 'TOLDEE exponent'),
            ('scf_tolinteg', 'code', 'text', 'TOLINTEG T1-T5 exponents when the deck sets TOLINTEG'),
            ('scf_diis', 'boolean', 'bool', 'DIIS for SCF active'), ('scf_scfdir', 'boolean', 'bool', 'SCFDIR'),
            ('scf_savewf', 'boolean', 'bool', 'SAVEWF'), ('scf_savepred', 'boolean', 'bool', 'SAVEPRED'),
            ('scf_two_component', 'boolean', 'bool', 'a two-component (TWOCOMPON) SCF'),
            ('scf_levshift_enabled', 'boolean', 'bool', 'level shifter on'),
            ('scf_shrink_k1', 'dimensionless', 'int', 'Monkhorst-Pack shrink, 1st'),
            ('scf_shrink_k2', 'dimensionless', 'int', 'Monkhorst-Pack shrink, 2nd'),
            ('scf_shrink_k3', 'dimensionless', 'int', 'Monkhorst-Pack shrink, 3rd'),
            ('scf_gilat_shrink', 'dimensionless', 'int', 'Gilat net shrink'),
            ('scf_fmixing_percentage', 'dimensionless', 'int', 'FMIXING, %'),
            ('scf_anderson_factor', 'dimensionless', 'float', 'Anderson mixing factor'),
            ('scf_anderson_enabled', 'boolean', 'bool', 'Anderson mixing'),
            ('scf_diis_start_cycle', 'cycles', 'int', 'first DIIS cycle'),
            ('scf_diis_subspace_size', 'dimensionless', 'int', 'DIIS subspace'),
            ('scf_broyden_factor', 'dimensionless', 'float', 'Broyden mixing factor'),
            ('scf_broyden_enabled', 'boolean', 'bool', 'Broyden mixing'),
            ('scf_levshift_factor', 'dimensionless', 'float', 'level shift'),
            ('scf_levshift_cycles', 'cycles', 'int', 'level shift cycles'),
            ('scf_ilasize', 'dimensionless', 'int', 'ILASIZE'), ('scf_intgpack', 'dimensionless', 'int', 'INTGPACK'),
            ('scf_madelimit', 'dimensionless', 'int', 'MADELIMIT'), ('scf_poleordr', 'dimensionless', 'int', 'POLEORDR'),
            ('scf_exchange_permutation', 'boolean', 'bool', 'EXCHPERM'),
            ('scf_bipolarization', 'boolean', 'bool', 'BIPOLARIZ'),
            ('scf_dft_grid_points', 'dimensionless', 'int', 'DFT grid points'),
            ('scf_grid_pruning', 'boolean', 'bool', 'grid pruning')):
        k[name] = Key('scf_settings', unit, typ, what)

    # ------------------------------------------------------------ computational
    k['calculation_type'] = Key('computational', 'category', 'text',
                                'the run\'s type, as calculations.calc_type names it: OPT, SP, FREQ, BAND, '
                                'DOSS, TRANSPORT, CHARGE+POTENTIAL or ORBITALS')
    k['cpu_time_total_s'] = Key('computational', 'seconds', 'float',
                                'TOTAL CPU TIME: the CPU time of all processes')
    k['wall_time_s'] = Key('computational', 'seconds', 'float', 'wall time: the last TELAPSE of the output')
    k['scf_cycles'] = Key('computational', 'cycles', 'int', 'cycles of the last SCF')
    k['advanced_analysis_note'] = Key('computational', 'category', 'text',
                                      'why a BAND.DAT/DOSS.DAT was not analyzed')
    k['advanced_analysis_error'] = Key('computational', 'category', 'text', 'the band analysis failed')
    return k


KEYS: Dict[str, Key] = _build()

#: large JSON values kept in the property_blobs side table (MaterialDatabase),
#: not in the properties table
BLOB_KEYS = frozenset(n for n in KEYS if n.startswith(('neighbor_analysis', 'final_neighbor_analysis',
                                                       'overlap_population', 'processed_')))

# Old name -> the name it became (MACE 1.1.6). Several old names of one new
# one are listed in the order an old database's rows are taken in.
_RENAMES: Dict[str, Sequence[str]] = {
    'band_gap': ('classification_band_gap',),
    'band_fundamental_gap_ev': ('band_dat_band_gap_ev', 'band_indirect_gap_ev', 'band_gap_advanced_ev'),
    'dos_band_gap_from_dos_ev': ('doss_dat_band_gap_ev',),
    'dos_at_fermi_total': ('dos_at_fermi_both_spins_ev', 'dos_total_at_fermi'),
    'dos_is_metallic': ('doss_dat_metallic',),
    'spin_polarized': ('is_spin_polarized', 'chargepot_spin_polarized'),
    'electronic_classification': ('electronic_classification_advanced',),
    'fermi_energy_au': ('fermi_energy', 'fermi_energy_band', 'fermi_energy_dos'),
    'scf_diis': ('scf_diis_enabled',),
    'transport_zt_electronic_max_avg': ('transport_zt_electronic_max', 'transport_zt_max'),
    'transport_power_factor_max_avg': ('transport_power_factor_max',),
    'transport_zt_total_max_avg': ('transport_zt_total_max',),
    'cpu_time_total_s': ('total_cpu_time',),
    'atomic_position_axes': ('final_atomic_position_axes',),
    'atoms_in_unit_cell': ('final_atoms_count',),
}
for _p in ('primitive', 'crystallographic'):
    for _x in ('a', 'b', 'c', 'alpha', 'beta', 'gamma'):
        _RENAMES[f'{_p}_{_x}'] = (f'final_{_p}_{_x}',)
for _x in ('primitive_cell_volume', 'primitive_cell_area', 'primitive_cell_length',
           'crystallographic_cell_volume', 'density', 'atomic_positions'):
    _RENAMES[_x] = (f'final_{_x}',)

ALIASES: Dict[str, str] = {old: new for new, olds in _RENAMES.items() for old in olds}

#: names 1.1.5 wrote and 1.1.6 no longer writes, with no replacement (a
#: re-extraction removes their rows)
REMOVED = frozenset({
    'has_dielectric_data', 'dielectric_extraction_note', 'dos_analysis_gcrit_factor',
    'band_structure_k_range_min', 'advanced_analysis_available', 'advanced_analysis_method',
    'magnetic_type', 'conductivity_type', 'conductivity_type_advanced', 'conductivity_estimate_advanced',
    'insulator_type', 'semiconductor_type', 'is_semimetal_advanced', 'band_gap_advanced_ha',
    'real_electron_mobility', 'real_hole_mobility', 'dos_at_fermi_level', 'doss_dat_at_fermi',
})

_warned: set = set()


def key(name: str) -> Optional[Key]:
    """The registry entry of ``name`` (an old name: of the one it became)."""
    return KEYS.get(name) or KEYS.get(ALIASES.get(name, ''))


def category(name: str) -> str:
    k = KEYS.get(name)
    return k.category if k else 'other'


def unit(name: str) -> str:
    k = KEYS.get(name)
    return k.unit if k else 'dimensionless'


def canonical(name: str, warn: bool = True, out=None) -> str:
    """The current name of ``name``; an old one is noted once per process
    (on stderr, or through ``out``)."""
    new = ALIASES.get(name)
    if new is None:
        return name
    if warn and name not in _warned:
        _warned.add(name)
        msg = (f"note: the property {name!r} is called {new!r} since MACE 1.1.6; "
               f"the old name is read until MACE {ALIASES_UNTIL}")
        (out or (lambda m: print(m, file=sys.stderr)))(msg)
    return new


def query_names(name: str, warn: bool = True) -> List[str]:
    """The stored names a query for ``name`` (current or old) reads: the
    current name first, then its old names (rows of a database built before
    1.1.6)."""
    new = canonical(name, warn=warn)
    return [new, *_RENAMES.get(new, ())]


def normalize_rows(rows: Iterable, name_key: str = 'property_name',
                   scope: Sequence[str] = ('material_id', 'calc_id')) -> list:
    """Rows (dicts) with old names shown under their current names. A row
    under an old name is dropped where its scope (material and
    calculation) also has a row under the current name, or under an old
    name listed before it."""
    rows = [dict(r) for r in rows]
    present = {}
    for i, r in enumerate(rows):
        name = r.get(name_key)
        new = ALIASES.get(name, name)
        rank = 0 if new == name else 1 + list(_RENAMES[new]).index(name)
        sc = tuple(r.get(s) for s in scope) + (new,)
        if sc not in present or rank < present[sc][0]:
            present[sc] = (rank, i)
    keep = {i for _, i in present.values()}
    out = []
    for i, r in enumerate(rows):
        if i not in keep:
            continue
        name = r.get(name_key)
        if name in ALIASES:
            r[name_key] = ALIASES[name]
        out.append(r)
    return out


# The one classification scheme (electronic_classification): a gap below
# METAL_GAP_EV is none (CRYSTAL prints gaps to 1e-4 eV, BAND.DAT energies to
# 1e-5 Ha), up to INSULATOR_GAP_EV a semiconductor (GaN, ZnO, SiC), above it
# an insulator (diamond, oxides of the main group). 1.1.5 had two schemes:
# 2.0 eV for the .out's gap and 4.08 eV (0.15 Ha) for the band analysis.
METAL_GAP_EV = 0.01
INSULATOR_GAP_EV = 4.0


def classify_gap(gap_ev: float) -> str:
    """metal, semiconductor or insulator for a gap in eV."""
    if gap_ev < METAL_GAP_EV:
        return 'metal'
    return 'semiconductor' if gap_ev <= INSULATOR_GAP_EV else 'insulator'


# units.py's names for the registry's units (for UnitConverter)
CONVERTER_UNITS = {
    'Hartree': 'hartree', 'eV': 'ev', 'kJ/mol': 'kj/mol', 'Å': 'angstrom', 'degrees': 'degree',
    'K': 'k', 'MPa': 'mpa', 'cm⁻¹': 'cm^-1', 'Å³': 'angstrom^3', 'Å²': 'angstrom^2',
}
