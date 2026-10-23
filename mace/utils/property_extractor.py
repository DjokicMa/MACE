#!/usr/bin/env python3
"""
CRYSTAL Property Extractor
==========================
Comprehensive property extraction from CRYSTAL output files to populate the properties table.

Extracts:
- Structural properties: lattice parameters, atomic positions, cell volumes, densities
- Electronic properties: band gaps (direct/indirect, alpha/beta), total energies, D3 corrections
- Population analysis: Mulliken charges, overlap populations (alpha+beta, alpha-beta)
- Geometry optimization: initial vs final geometries, convergence information
- Crystallographic information: primitive vs crystallographic cells, space groups

Usage:
  python crystal_property_extractor.py --output-file file.out [--db-path materials.db]
  python crystal_property_extractor.py --scan-directory /path/to/outputs [--db-path materials.db]
"""

import os
import sys
import re
import math
import sqlite3
import time
import argparse
import contextlib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from mace.constants import HARTREE_TO_EV
from mace.utils import db_build
from mace.utils.fast_text import ci_findall, ci_search
from mace.utils.calc_detection import (
    is_band_output, is_optimization_output, is_transport_output,
    is_two_component_output, mechanics_output_type, mechanics_reference_text, is_cphf_output)

# Import MACE components
try:
    from mace.database.materials import MaterialDatabase
except ImportError as e:
    print(f"Error importing MaterialDatabase: {e}")
    sys.exit(1)


# The real electron count of a run (DOPING does not change the header's count)
from mace.utils.ion_fragments import mulliken_electron_count  # noqa: E402

from mace.utils import property_keys  # noqa: E402

_DIR_NAMES: Dict[str, Tuple[int, List[str]]] = {}


def _atom_population(atom: Dict[str, Any]):
    """A Mulliken atom row's population (named mulliken_charge before 1.1.6)."""
    value = atom.get('mulliken_population')
    return atom.get('mulliken_charge') if value is None else value


def _sorted_names(directory: Path) -> List[str]:
    """The entry names of ``directory`` in ``sorted(directory.iterdir())``
    order (entries of one directory sort by name), cached while the
    directory's mtime stays the same: every BAND/DOSS output of a folder
    looks for its .DAT file in the same listing. An entry added or removed
    changes the mtime, so a long-lived process never sees a stale list."""
    key = str(directory)
    try:
        mtime = os.stat(directory).st_mtime_ns
    except OSError:
        mtime = None
    hit = _DIR_NAMES.get(key)
    if mtime is not None and hit is not None and hit[0] == mtime:
        return hit[1]
    names = [p.name for p in sorted(directory.iterdir())]
    # a listing whose mtime is recent is not kept: on a file system with a
    # coarse mtime a file could still be added within the same tick
    if mtime is not None and time.time_ns() - mtime > 2_000_000_000:
        if len(_DIR_NAMES) > 64:
            _DIR_NAMES.clear()
        _DIR_NAMES[key] = (mtime, names)
    return names


# Lower-case literals each INFORMATION pattern of _extract_scf_settings holds
# verbatim: a file lacking one cannot match, so the search is skipped there
# (fast_text.ci_search).
_SCF_INFO_LITERALS = {
    'scf_biposize': ('information ', 'biposize', 'coulomb bipolar buffer set to'),
    'scf_exchsize': ('information ', 'exchsize', 'exchange bipolar buffer size set to'),
    'scf_maxcycle': ('information ', 'maxcycle', 'max number of scf cycles set to'),
    'scf_ppan': ('information ', 'ppan', 'mulliken population analysis'),
    'scf_toldee': ('information ', 'toldee', 'scf tol on total energy set to'),
    'scf_tolinteg': ('information ', 'tolinteg', 'coulomb and exchange series tolerances'),
    'scf_diis': ('information ', 'diis', 'diis for scf active'),
    'scf_scfdir': ('information ', 'scfdir', 'direct scf'),
    'scf_savewf': ('information ', 'savewf'),
    'scf_savepred': ('information ', 'savepred'),
}


from mace.utils.fermi_level import scf_parent_of_band_output as _scf_parent_of_band_output  # noqa: E402


def band_occupied_count(content: str, output_file: Optional[Path] = None) -> Optional[int]:
    """Occupied bands (closed shell) for a BAND .out, or None when not known.

    The real electron count is the Mulliken one, from the BAND .out itself or
    from its SCF parent's .out beside it; the properties header is used only
    when neither has one. An odd or fractional count has no closed-shell HOMO
    index, so None is returned and the BAND gap stays unset rather than wrong.
    """
    count = mulliken_electron_count(content)
    if count is None and output_file is not None:
        parent = _scf_parent_of_band_output(Path(output_file))
        if parent is not None:
            try:
                count = mulliken_electron_count(parent.read_text(errors="ignore"))
            except OSError:
                count = None
    if count is None:
        header = re.findall(r'N\.?\s*OF ELECTRONS PER CELL\s*(\d+)', content)
        if not header:
            return None
        count = float(header[-1])
    nearest = round(count)
    if abs(count - nearest) > 1e-3 or nearest % 2:
        return None
    return nearest // 2


def _property_tables(conn) -> List[str]:
    """The tables property rows live in: properties, and property_blobs (the
    large JSON values) where the database has it (1.1.6 on)."""
    has = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'property_blobs'").fetchone()
    return ['properties', 'property_blobs'] if has else ['properties']


def _derived_keys() -> List[str]:
    """Every key a derived property writes, and the provenance row: what a
    run that computes none of them (--no-derived, a failure) keeps."""
    try:
        from mace.properties import registry
        registry.load()
        return sorted({o.name for p in registry.properties(None) for o in p.outputs} | {registry.PROVENANCE_KEY})
    except Exception:
        return []


def _kept_outputs(registry, report: Dict[str, Any]) -> List[str]:
    """The outputs of the properties that failed or were not computed
    (``kept``): their stored rows stay."""
    keep = []
    for name, entry in report.items():
        prop = registry.get(name)
        if prop is not None and (entry.get('status') == 'error' or entry.get('kept')):
            keep += [o.name for o in prop.outputs]
    return keep


def _reference_choices(registry, report: Dict[str, Any], refs: Dict[str, List[Any]]) -> Optional[Dict[str, Any]]:
    """The given references to save with the run (_save_properties): those
    of a property that stored a value, and the ones given as "auto"
    (forgotten); None when there are none."""
    used, forget = {}, {}
    for name, entry in report.items():
        prop = registry.get(name)
        if prop is None or not prop.references:
            continue
        for ref in prop.references:
            if any(v == registry.AUTO for v in refs.get(ref.name, [])):
                forget.setdefault(name, []).append(ref.name)
            recs = (entry.get('references') or {}).get(ref.name) or []
            if entry.get('status') == 'stored' and recs and all(r.get('via') == 'given' for r in recs):
                used.setdefault(name, {})[ref.name] = [{**r, 'file': str(Path(r['file']).resolve())}
                                                       if r.get('file') else r for r in recs]
    if not (used or forget):
        return None
    choices = {'used': used, 'forget': forget}
    return choices


class CrystalPropertyExtractor:
    """Extract comprehensive properties from CRYSTAL output files."""
    
    def __init__(self, db_path: str = "materials.db", enable_tracking: bool = True,
                 ase_db_path: Optional[str] = None):
        # The database is created lazily (on first real use) so pure parsing —
        # e.g. extracting properties from a single .out with enable_tracking=False
        # — never touches or creates materials.db. Pass enable_tracking=False for
        # offline single-file extraction.
        self.db_path = db_path
        self.enable_tracking = enable_tracking
        self.ase_db_path = ase_db_path      # None: MaterialDatabase's own default
        self._db = None
        self.properties = []

    @property
    def db(self):
        """Lazily-opened MaterialDatabase, or None when tracking is disabled."""
        if self._db is None and self.enable_tracking and self.db_path:
            if self.ase_db_path:
                self._db = MaterialDatabase(self.db_path, ase_db_path=self.ase_db_path)
            else:
                self._db = MaterialDatabase(self.db_path)
        return self._db
        
    def extract_all_properties(self, output_file: Path, material_id: str = None, calc_id: str = None) -> Dict[str, Any]:
        """Extract all properties from a CRYSTAL output file."""
        print(f"🔍 Extracting properties from: {output_file.name}")
        
        if not output_file.exists():
            print(f"❌ Output file not found: {output_file}")
            return {}
        
        try:
            with open(output_file, 'r') as f:
                content = f.read()
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return {}
        
        # Auto-detect material_id and calc_id if not provided. A tracked
        # calculation files its rows under its own material (the one the
        # Database view lists), not under an id guessed from the file name;
        # an untracked output keeps the file-name id.
        if not calc_id:
            calc_id = self._find_calc_id_for_output(output_file)
        if not material_id and calc_id:
            material_id = self._material_of_calc(calc_id)
        if not material_id:
            material_id = self._extract_material_id_from_filename(output_file)
        
        properties = {}

        # An ELASTIC, EOS or QHA run optimizes strained cells or volumes, each
        # to its own SCF, FINAL OPTIMIZED GEOMETRY and (QHA) frequencies. The
        # structure, energies and populations are its reference cell's: read
        # from the part before the first strained cell. Its type is its own
        # (completion_checker); timings, settings and the module's results
        # are read from the whole output.
        full_content = content
        mechanics = mechanics_output_type(content)
        if mechanics:
            content = mechanics_reference_text(content)

        # Extract different property categories
        properties.update(self._extract_structural_properties(content))
        properties.update(self._extract_electronic_properties(content))
        properties.update(self._extract_population_analysis(content))
        properties.update(self._extract_energy_properties(content))
        properties.update(self._extract_geometry_optimization(content))
        if mechanics:
            properties['calculation_type'] = mechanics
        elif is_cphf_output(content):
            properties['calculation_type'] = 'CPHF'      # an SP with the CPHF/CPKS block (not FREQ INTCPHF)
        properties.update(self._extract_crystallographic_info(content))
        if 'space_group_number' not in properties:
            # CRYSTAL23 prints the symbol ("SPACE GROUP (CENTROSYMMETRIC) : F D 3 M"), not a
            # number: the formula extractor's validated symbol table gives it
            try:
                from mace.utils.formula_extractor import extract_space_group_from_output
                sg = extract_space_group_from_output(Path(output_file), content) if 'SPACE GROUP' in content else None
                if sg:
                    properties['space_group_number'] = int(sg)
            except Exception:
                pass
        properties.update(self._extract_neighbor_information(content))
        properties.update(self._extract_computational_properties(full_content))
        properties.update(self._extract_band_structure_properties(content, output_file))
        properties.update(self._extract_dos_properties(content, output_file))
        if not mechanics:
            properties.update(self._extract_frequency_properties(content))
            properties.update(self._extract_raman_intensities(content))
        properties.update(self._extract_transport_properties(content, output_file))
        # ELASTCON / ELAPIEZO / PIEZOCON / PIEZOCP / PHOTOELA / EOS tensors and
        # moduli (mace.elastic.analysis), and a two-component SCF's
        # magnetization; both self-skipping
        properties.update(self._extract_mechanical_properties(full_content))
        properties.update(self._extract_two_component_properties(full_content))
        # CHARGE+POTENTIAL (CRYSTAL ECH3/POT3): grid metadata + references to the
        # generated .CUBE grid files. Additive and self-skipping -- returns {} when
        # the .out has no ECH3/POT3 markers, so non-charge calcs are untouched.
        charge_potential = self._extract_charge_potential_properties(content, output_file)
        properties.update(charge_potential)
        if charge_potential and properties.get('calculation_type') in (None, 'SP'):
            # the database's calculation type of an ECH3/POT3 run
            properties['calculation_type'] = 'CHARGE+POTENTIAL'
        # ORBITALS (Molden orbital export): an index of the .molden files, k points,
        # levels and HOMO/LUMO; no grids and no coefficients. Self-skipping.
        orbitals = self._extract_orbitals_properties(content, output_file)
        properties.update(orbitals)
        if orbitals:
            properties['calculation_type'] = 'ORBITALS'

        # Extract SCF settings from output
        properties.update(self._extract_scf_settings(full_content))

        # A molecule has no space group (CRYSTAL prints P 1 for one)
        if self._dimensionality(content) == 0:
            properties.pop('space_group_number', None)

        # One Fermi energy: the last SCF's (EFERMI, full precision), else the
        # FERMI ENERGY a BAND or DOSS output prints (3 decimals)
        band_fermi = properties.pop('_fermi_band', None)
        dos_fermi = properties.pop('_fermi_dos', None)
        if 'fermi_energy_au' not in properties:
            fermi = band_fermi if band_fermi is not None else dos_fermi
            if fermi is not None:
                # CRYSTAL's -1.0 placeholder (an SCF that ended spin-locked): the band data's
                # derived Fermi level, the DOSS.DAT's, or none (mace.utils.fermi_level)
                from mace.utils import fermi_level
                fl = fermi_level.run_fermi(output_file, fermi)
                if fl.value_au is not None:
                    properties['fermi_energy_au'] = fl.value_au
                if fl.note:
                    properties['fermi_energy_note'] = fl.note

        # Advanced electronic analysis using BAND/DOSS data if available
        properties.update(self._extract_advanced_band_dos_analysis(output_file, properties, full_content))

        # One spin flag: an unrestricted run by any sign of one (spin-channel
        # gaps or band edges, the alpha-beta Mulliken table, the SPIN cube of
        # an ECH3 run, CRYSTAL's UNRESTRICTED OPEN SHELL line)
        properties['spin_polarized'] = bool(
            properties.get('spin_polarized') or properties.get('_mulliken_spin')
            or properties.pop('_chargepot_spin', False) or 'UNRESTRICTED OPEN SHELL' in full_content)

        # Add electronic classification based on band gap
        properties.update(self._classify_electronic_properties(properties, content))
        for internal in ('_mulliken_spin', '_band_dos_classification'):
            properties.pop(internal, None)
        
        # Add metadata
        properties['_metadata'] = {
            'material_id': material_id,
            'calc_id': calc_id,
            'output_file': str(output_file),
            'extracted_at': datetime.now().isoformat(),
            'extractor_version': '1.0'
        }
        
        # Charged runs (DOPING / CHEMOD / CHARGED): electron count and the
        # Mulliken charge and spin of each molecular fragment. Self-skipping:
        # nothing is added for a neutral run unless asked for.
        try:
            properties.update(self._extract_fragment_charges(content))
        except Exception as e:
            print(f"Warning: Could not compute fragment charges: {e}")

        # Process population analysis data if available. Package-qualified
        # import first: the bare name only resolves when mace/utils is on
        # sys.path directly, so in every production context the ImportError
        # was silently swallowed and population processing never ran.
        try:
            try:
                from mace.utils.population_analysis_processor import PopulationAnalysisProcessor
            except ImportError:
                from population_analysis_processor import PopulationAnalysisProcessor
            processor = PopulationAnalysisProcessor()
            
            # Check if we have population analysis data to process
            pop_data = {}
            for key, value in properties.items():
                if 'mulliken' in key or 'overlap' in key:
                    pop_data[key] = value
            
            if pop_data:
                processed_pop = processor.process_population_data(pop_data)
                
                # Add processed data as new properties
                for key, value in processed_pop.items():
                    if key != 'error':
                        properties[f'processed_{key}'] = value
                        
        except ImportError:
            pass  # Population processor not available
        except Exception as e:
            print(f"Warning: Error processing population analysis: {e}")

        # Derived properties (mace/properties): computed from what was read
        # above, the run's sibling files and reference runs; added after it,
        # nothing above changes. extraction_provenance says which ran.
        if self.derived_enabled():
            self._add_derived_properties(output_file, full_content, properties)
        elif self.enable_tracking:
            # --no-derived / MACE_DERIVED=0: none computed, the stored ones stay
            properties['_keep_rows'] = _derived_keys()

        return properties

    # Derived properties after the extraction (mace.properties.registry);
    # False reads the output alone (the registry's own reference loading),
    # and `--no-derived` / MACE_DERIVED=0 (a stored derived row then stays).
    run_derived = True
    # names of derived properties not computed (`--skip-property NAME`, also
    # MACE_SKIP_PROPERTIES=a,b); their stored rows stay
    skip_derived = frozenset()
    # a bulk build (a folder): the properties that read other runs wait until
    # every run's own rows are stored (refresh_reference_properties), so their
    # values do not depend on the order the folder is read in
    defer_references = False
    # {reference name: file | calc_id | RunRef} for derived properties that
    # need other runs (`--reference NAME=FILE|CALC_ID`)
    references = None
    # a database to look reference runs up in when this extraction writes to
    # none (the GUI's values of a run not in a database; read only)
    reference_db = None

    def derived_enabled(self) -> bool:
        """Whether derived properties are computed (run_derived, and MACE_DERIVED is not 0)."""
        return bool(self.run_derived) and os.environ.get('MACE_DERIVED', '1').strip().lower() not in (
            '0', 'false', 'no', 'off')

    def skipped_properties(self) -> set:
        """skip_derived and MACE_SKIP_PROPERTIES (comma-separated)."""
        env = {n.strip() for n in os.environ.get('MACE_SKIP_PROPERTIES', '').split(',') if n.strip()}
        return set(self.skip_derived or ()) | env

    def _given_references(self, registry) -> Dict[str, List[Any]]:
        refs = {}
        for name, given in (self.references or {}).items():
            # each a file, a calc_id of the database, N*either (a coefficient) or "auto"
            refs[name] = [registry.given_ref(v) if isinstance(v, str) else v
                          for v in (given if isinstance(given, (list, tuple)) else [given])]
        return refs

    def _ro_opener(self, opened: List[Any]):
        def ro_conn():
            # read-only, and only when a property needs reference runs
            conn = None
            if self.enable_tracking and self.db_path and Path(self.db_path).is_file():
                conn = db_build.open_ro(self.db_path)
            elif self.reference_db and Path(self.reference_db).is_file():
                conn = db_build.open_ro(self.reference_db)
            if conn is not None:
                opened.append(conn)
            return conn
        return ro_conn

    def _add_derived_properties(self, output_file: Path, content: str, properties: Dict[str, Any]) -> None:
        try:
            from mace.properties import registry
        except Exception as e:
            print(f"Warning: derived properties not available: {e}")
            if self.enable_tracking:
                properties['_keep_rows'] = _derived_keys()
            return

        opened = []
        refs = self._given_references(registry)
        meta = properties.get('_metadata') or {}
        skip = {n: "turned off (--skip-property, MACE_SKIP_PROPERTIES)" for n in self.skipped_properties()}
        if self.defer_references and meta.get('calc_id'):
            for prop in registry.reference_properties():
                skip.setdefault(prop.name, "computed after the build's runs are stored (it reads other runs)")
        try:
            values, report = registry.run_derived(
                output_file, content, {k: v for k, v in properties.items() if not k.startswith('_')},
                calc_type=properties.get('calculation_type'), references=refs, conn=self._ro_opener(opened),
                dimensionality=self._dimensionality(content), calc_id=meta.get('calc_id'),
                material_id=meta.get('material_id'), skip=skip)
        except Exception as e:
            # nothing was computed: the stored derived rows stay
            print(f"Warning: derived properties failed: {e}")
            if self.enable_tracking:
                properties['_keep_rows'] = _derived_keys()
            return
        finally:
            for conn in opened:
                conn.close()
        for name, entry in report.items():
            if entry.get('status') == 'error':
                print(f"Warning: derived property {name} failed: {entry.get('reason')}")
        properties.update(values)
        if report:
            properties['extraction_provenance'] = registry.provenance(report, output_file)
        # the stored rows of a property that failed, or was not computed this
        # time, stay (a failure never deletes an earlier value)
        keep = _kept_outputs(registry, report)
        if keep and self.enable_tracking:
            properties['_keep_rows'] = keep
        choices = _reference_choices(registry, report, refs)
        if choices:
            properties['_reference_choices'] = choices

    def refresh_reference_properties(self, runs) -> Dict[str, int]:
        """The derived properties that read other runs
        (registry.reference_properties: reference energies, ...) of ``runs``
        ([(output_file, calc_id)] of tracked runs), computed from what the
        database holds now and stored: a bulk build's second pass, after
        every run's own rows are stored, so a lookup's answer does not depend
        on the order the runs were read in. A saved choice (reference_runs)
        and ``references`` win over the lookup, as in an extraction. A
        property that fails keeps its stored rows. Returns counts."""
        counts = {'runs': 0, 'changed': 0, 'errors': 0}
        if not self.enable_tracking or self.db is None or not self.derived_enabled():
            return counts
        try:
            from mace.properties import registry
            props = [p for p in registry.reference_properties() if p.name not in self.skipped_properties()]
        except Exception as e:
            print(f"Warning: derived properties not available: {e}")
            return counts
        if not props or not Path(self.db_path).is_file():
            return counts
        refs = self._given_references(registry)
        derived = set(_derived_keys())
        conn = db_build.open_ro(self.db_path)
        if conn is None:
            return counts
        try:
            for output_file, calc_id in runs:
                if not calc_id or not Path(output_file).is_file():
                    continue
                row = conn.execute("SELECT material_id, calc_type FROM calculations WHERE calc_id = ?",
                                   (calc_id,)).fetchone()
                if not row:
                    continue
                material_id = row[0]
                stored = registry.stored_values(conn, calc_id)
                calc_type = stored.get('calculation_type') or row[1]
                todo = [p.name for p in props if p.applies_to(calc_type)]
                if not todo:
                    continue
                try:
                    content = Path(output_file).read_text(errors='replace')
                except OSError:
                    continue
                # what the extractor read (as an extraction gives it), not the derived rows
                own = {k: v for k, v in stored.items() if k not in derived}
                values, report = registry.run_derived(
                    Path(output_file), content, own, calc_type=calc_type, references=refs, conn=lambda: conn,
                    only=todo, dimensionality=self._dimensionality(content), calc_id=calc_id,
                    material_id=material_id)
                counts['runs'] += 1
                before = {o.name: stored.get(o.name) for p in props if p.name in report for o in p.outputs}
                after = dict(before)
                for name, entry in report.items():
                    if entry.get('status') == 'error':
                        counts['errors'] += 1
                        print(f"Warning: derived property {name} of {Path(output_file).name} failed: "
                              f"{entry.get('reason')}")
                    elif not entry.get('kept'):
                        for o in registry.get(name).outputs:
                            after[o.name] = values.get(o.name)
                if after != before:
                    counts['changed'] += 1
                self._store_reference_values(registry, material_id, calc_id, Path(output_file), values, report,
                                             stored.get(registry.PROVENANCE_KEY))
                choices = _reference_choices(registry, report, refs)
                if choices:
                    self._save_reference_choices(material_id, calc_id, choices)
        finally:
            conn.close()
        return counts

    def _store_reference_values(self, registry, material_id: str, calc_id: str, output_file: Path,
                                values: Dict[str, Any], report: Dict[str, Any], provenance) -> None:
        """refresh_reference_properties' rows: each property's outputs (the
        ones it no longer writes removed, a failed one's kept) and its entry
        of the run's extraction_provenance, in one transaction."""
        now = datetime.now().isoformat()
        transaction = getattr(self.db, 'transaction', None)
        try:
            with (transaction() if transaction is not None else contextlib.nullcontext()):
                for name, entry in report.items():
                    if entry.get('status') == 'error' or entry.get('kept'):
                        continue
                    outs = [o.name for o in registry.get(name).outputs]
                    stale = [o for o in outs if o not in values]
                    if stale:
                        with self.db._get_connection() as conn:
                            for table in _property_tables(conn):
                                conn.execute(
                                    f"DELETE FROM {table} WHERE material_id = ? AND calc_id = ? "
                                    "AND extractor_script = 'crystal_property_extractor.py' "
                                    f"AND property_name IN ({','.join('?' * len(stale))})",
                                    (material_id, calc_id, *stale))
                    for o in outs:
                        if o in values:
                            self._store_row(material_id, calc_id, o, values[o], now)
                prov = provenance if isinstance(provenance, dict) else registry.provenance({}, output_file)
                prov.setdefault('derived', {}).update(
                    registry.provenance(report, output_file)['derived'])
                self._store_row(material_id, calc_id, registry.PROVENANCE_KEY, prov, now)
        except sqlite3.Error as e:
            print(f"⚠️  Error saving the reference properties of {output_file.name}: {e}; none of them were written")

    # Fragment charges for every run, not only charged ones (opt-in; also
    # MACE_FRAGMENT_CHARGES=1 or `mace analyze --extract-properties DIR --fragments`).
    extract_fragments = False

    # Total ZT of TRANSPORT runs (opt-in: `--kappa-lattice VALUE|FILE`
    # [--tau-fs FS]): a transport_zt.KappaLattice and an optional relaxation
    # time in fs (None: the run's RELAXTIM). Without kappa_lattice no
    # transport_zt_total* key is written.
    kappa_lattice = None
    tau_fs = None

    # names written that the key registry does not know (warned once)
    _unregistered: set = set()

    def _extract_fragment_charges(self, content: str) -> Dict[str, Any]:
        """electron_count and fragment_charges of a charged run (ion_fragments.fragment_charges)."""
        from mace.utils.ion_fragments import fragment_charges
        force = self.extract_fragments or os.environ.get("MACE_FRAGMENT_CHARGES") == "1"
        return fragment_charges(content, force=force)

    # non-periodic directions of a slab (c) and a polymer (b, c): CRYSTAL
    # prints a formal 500 A there, which is no lattice parameter
    _NON_PERIODIC = {2: ('c',), 1: ('b', 'c')}

    _PROPERTIES_CELL = re.compile(r'LATTICE PARAMETERS\s+\(ANGSTROMS? AND DEGREES\)\s+-\s+PRIMITIVE CELL\s*\n'
                                  r'\s*A\s+B\s+C[^\n]*\n\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)')

    @staticmethod
    def _dimensionality(content: str) -> Optional[int]:
        """The run's dimensionality: CRYSTAL's DIMENSIONALITY line, else (a
        properties run - BAND, DOSS, TRANSPORT - prints none) its primitive
        cell, where a non-periodic direction is the formal 500 A."""
        dim = re.search(r'DIMENSIONALITY OF THE SYSTEM\s+(\d)', content)
        if dim:
            return int(dim.group(1))
        cell = CrystalPropertyExtractor._PROPERTIES_CELL.search(content)
        if not cell:
            return None
        b, c = (float(cell.group(i)) for i in (2, 3))
        return 1 if b == 500.0 and c == 500.0 else 2 if c == 500.0 else 3

    def _extract_structural_properties(self, content: str) -> Dict[str, Any]:
        """Lattice and geometry. The plain names are the final geometry's (an
        OPT's optimized one, else the input geometry), initial_* the input
        geometry's."""
        props = {}
        
        # Extract both initial and final geometries
        initial_geometry = self._extract_initial_geometry(content)
        if initial_geometry:
            for key, value in initial_geometry.items():
                # keys already named initial_* keep their name (no initial_initial_*)
                props[key if key.startswith('initial_') else f'initial_{key}'] = value
        
        final_geometry = self._extract_final_geometry(content)
        if final_geometry:
            props.update(final_geometry)
        elif initial_geometry:
            props.update({k: v for k, v in initial_geometry.items() if not k.startswith('initial_')})

        # CRYSTAL heads the cell measure VOLUME whatever the dimensionality:
        # for a slab it is the area of the 2D cell (A^2), for a polymer the
        # length of its 1D cell (A). primitive_cell_volume is kept for 3D.
        dim = self._dimensionality(content)
        measure = {2: 'area', 1: 'length'}.get(dim)
        if measure:
            for key in [k for k in props if k.endswith('primitive_cell_volume')]:
                props[key[:-len('volume')] + measure] = props.pop(key)
            # the final cell prints no measure: from its parameters (area
            # a*b*sin(gamma) of a slab, length a of a polymer)
            if final_geometry and 'primitive_a' in final_geometry and f'primitive_cell_{measure}' not in props:
                import math
                if measure == 'area' and {'primitive_b', 'primitive_gamma'} <= set(final_geometry):
                    final = (final_geometry['primitive_a'] * final_geometry['primitive_b']
                             * math.sin(math.radians(final_geometry['primitive_gamma'])))
                else:
                    final = final_geometry['primitive_a'] if measure == 'length' else None
                if final is not None:
                    props[f'primitive_cell_{measure}'] = round(final, 6)
        for axis in self._NON_PERIODIC.get(dim, ()):
            for pre in ('', 'initial_'):
                for cell in ('primitive', 'crystallographic'):
                    props.pop(f'{pre}{cell}_{axis}', None)
        
        # Number of atoms
        atoms_match = re.search(r'ATOMS IN THE UNIT CELL:\s*(\d+)', content)
        if atoms_match:
            props['atoms_in_unit_cell'] = int(atoms_match.group(1))
        
        return props
    
    def _extract_initial_geometry(self, content: str) -> Dict[str, Any]:
        """Extract initial geometry from beginning of output file."""
        props = {}
        
        # Look for initial lattice parameters (before optimization)
        # This appears early in the file in the geometry setup section
        initial_primitive_match = re.search(
            r'LATTICE PARAMETERS\s+\(ANGSTROMS AND DEGREES\)\s+-\s+PRIMITIVE CELL\s*\n'
            r'\s*A\s+B\s+C\s+ALPHA\s+BETA\s+GAMMA\s+VOLUME\s*\n'
            r'\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)',
            content
        )
        
        if initial_primitive_match:
            props.update({
                'primitive_a': float(initial_primitive_match.group(1)),
                'primitive_b': float(initial_primitive_match.group(2)),
                'primitive_c': float(initial_primitive_match.group(3)),
                'primitive_alpha': float(initial_primitive_match.group(4)),
                'primitive_beta': float(initial_primitive_match.group(5)),
                'primitive_gamma': float(initial_primitive_match.group(6)),
                'primitive_cell_volume': float(initial_primitive_match.group(7))
            })
        else:
            # Alternative pattern for early geometry information
            alt_primitive_match = re.search(
                r'PRIMITIVE CELL.*?VOLUME=\s*([\d.]+).*?DENSITY\s*([\d.]+)\s*g/cm\^3.*?'
                r'A\s+B\s+C\s+ALPHA\s+BETA\s+GAMMA\s*\n\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)',
                content[:5000], re.DOTALL  # Look only in first part of file
            )
            
            if alt_primitive_match:
                props.update({
                    'primitive_cell_volume': float(alt_primitive_match.group(1)),
                    'density': float(alt_primitive_match.group(2)),
                    'primitive_a': float(alt_primitive_match.group(3)),
                    'primitive_b': float(alt_primitive_match.group(4)),
                    'primitive_c': float(alt_primitive_match.group(5)),
                    'primitive_alpha': float(alt_primitive_match.group(6)),
                    'primitive_beta': float(alt_primitive_match.group(7)),
                    'primitive_gamma': float(alt_primitive_match.group(8))
                })
        
        # Initial crystallographic cell
        initial_crystal_match = re.search(
            r'LATTICE PARAMETERS\s+\(ANGSTROMS AND DEGREES\)\s+-\s+CONVENTIONAL CELL\s*\n'
            r'\s*A\s+B\s+C\s+ALPHA\s+BETA\s+GAMMA\s*\n'
            r'\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)',
            content[:5000]  # Look in first part of file
        )
        
        if initial_crystal_match:
            props.update({
                'crystallographic_a': float(initial_crystal_match.group(1)),
                'crystallographic_b': float(initial_crystal_match.group(2)),
                'crystallographic_c': float(initial_crystal_match.group(3)),
                'crystallographic_alpha': float(initial_crystal_match.group(4)),
                'crystallographic_beta': float(initial_crystal_match.group(5)),
                'crystallographic_gamma': float(initial_crystal_match.group(6))
            })
        
        # Extract initial atomic positions
        initial_positions = self._extract_atomic_positions(content, search_final=False)
        if initial_positions:
            props['initial_atomic_positions'] = initial_positions
            props['initial_atoms_count'] = len(initial_positions)
        
        return props
    
    def _extract_final_geometry(self, content: str) -> Dict[str, Any]:
        """Extract final geometry (for optimization calculations)."""
        props = {}
        
        # Look for final optimized geometry
        if 'FINAL OPTIMIZED GEOMETRY' in content or 'OPT END - CONVERGED' in content:
            # Final primitive cell
            final_primitive_match = re.search(
                r'FINAL OPTIMIZED GEOMETRY.*?PRIMITIVE CELL.*?VOLUME=\s*([\d.]+).*?DENSITY\s*([\d.]+)\s*g/cm\^3.*?'
                r'A\s+B\s+C\s+ALPHA\s+BETA\s+GAMMA\s*\n\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)',
                content, re.DOTALL
            )
            
            if final_primitive_match:
                props.update({
                    'primitive_cell_volume': float(final_primitive_match.group(1)),
                    'density': float(final_primitive_match.group(2)),
                    'primitive_a': float(final_primitive_match.group(3)),
                    'primitive_b': float(final_primitive_match.group(4)),
                    'primitive_c': float(final_primitive_match.group(5)),
                    'primitive_alpha': float(final_primitive_match.group(6)),
                    'primitive_beta': float(final_primitive_match.group(7)),
                    'primitive_gamma': float(final_primitive_match.group(8))
                })
            else:
                # A slab (or polymer) prints no VOLUME/DENSITY on its final
                # PRIMITIVE CELL line (the non-periodic parameter is formally
                # 500); read the cell within the final geometry's own block
                block = re.search(r'FINAL OPTIMIZED GEOMETRY.*?(?=ATOMS IN THE ASYMMETRIC UNIT|\Z)',
                                  content, re.DOTALL)
                cell = block and re.search(
                    r'PRIMITIVE CELL([^\n]*)\n\s*A\s+B\s+C\s+ALPHA\s+BETA\s+GAMMA\s*\n'
                    r'\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',
                    block.group(0))
                if cell:
                    props.update({name: float(cell.group(i)) for i, name in enumerate(
                        ('primitive_a', 'primitive_b', 'primitive_c', 'primitive_alpha',
                         'primitive_beta', 'primitive_gamma'), start=2)})
                    volume = re.search(r'VOLUME=\s*([\d.]+)', cell.group(1))
                    density = re.search(r'DENSITY\s*([\d.]+)\s*g/cm\^3', cell.group(1))
                    if volume:
                        props['primitive_cell_volume'] = float(volume.group(1))
                    if density:
                        props['density'] = float(density.group(1))
            
            # Final crystallographic cell
            final_crystal_match = re.search(
                r'FINAL OPTIMIZED GEOMETRY.*?CRYSTALLOGRAPHIC CELL \(VOLUME=\s*([\d.]+)\).*?'
                r'A\s+B\s+C\s+ALPHA\s+BETA\s+GAMMA\s*\n\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)',
                content, re.DOTALL
            )
            
            if final_crystal_match:
                props.update({
                    'crystallographic_cell_volume': float(final_crystal_match.group(1)),
                    'crystallographic_a': float(final_crystal_match.group(2)),
                    'crystallographic_b': float(final_crystal_match.group(3)),
                    'crystallographic_c': float(final_crystal_match.group(4)),
                    'crystallographic_alpha': float(final_crystal_match.group(5)),
                    'crystallographic_beta': float(final_crystal_match.group(6)),
                    'crystallographic_gamma': float(final_crystal_match.group(7))
                })
            
            # Extract final atomic positions
            final_positions = self._extract_atomic_positions(content, search_final=True)
            if final_positions:
                # x/y/z are fractional (X/A Y/B Z/C) for a 3D cell; a slab,
                # polymer or molecule has Cartesian columns, named here as
                # CRYSTAL prints them (e.g. "X/A Y/B Z(ANGSTROM)")
                axes = re.search(r'FINAL OPTIMIZED GEOMETRY.*?ATOM\s+(X\S*)\s+(Y\S*)\s+(Z\S*)\s*\n',
                                 content, re.DOTALL)
                if axes and any('ANGSTROM' in g for g in axes.groups()):
                    props['atomic_position_axes'] = ' '.join(axes.groups())
                props['atomic_positions'] = final_positions
        
        return props
    
    _GAP_LINE = re.compile(r'(INDIRECT|DIRECT) ENERGY BAND GAP:\s*([\d.]+)\s*eV')
    _SPIN_MARK = re.compile(r'^\s*(ALPHA|BETA)\s+ELECTRONS\s*$')
    _EDGE_LINE = re.compile(r'(TOP|BOTTOM) OF (VALENCE|VIRTUAL) BANDS\s*-\s*BAND\s+(\d+);\s*K\s+(\d+);'
                            r'\s*EIG\s+(\S+)\s*AU')
    _OPT_POINT = re.compile(r'OPTIMIZATION - POINT\s+\d+')

    @classmethod
    def _after_last_opt_point(cls, content: str) -> str:
        """The output from the last geometry-optimization point on (all of it
        when there is none).

        CRYSTAL prints an OPT's SCF in full (gap blocks, EFERMI lines, the
        ``+++ ENERGIES IN A.U. +++`` components) only at the first point of
        a run; later points print only the energy, and the summary after OPT
        END describes the final geometry. What comes before the last point
        is an earlier geometry's, never the final one's."""
        last = None
        for last in cls._OPT_POINT.finditer(content):
            pass
        return content[last.start():] if last else content

    @classmethod
    def final_band_gaps(cls, content: str) -> Dict[str, Any]:
        """The band gaps of the last SCF in the output (the converged one).

        Every SCF cycle prints, per spin channel (``ALPHA``/``BETA ELECTRONS``
        in a spin-polarized run), either ``INSULATING STATE`` with a
        ``DIRECT`` or ``INDIRECT ENERGY BAND GAP`` line or a ``(POSSIBLY)
        CONDUCTING STATE``. Only the last block of each channel describes the
        result: an earlier cycle's gap (cycle 0 of an SP is the guess) is not
        a property of the material.

        Returns ``{channel: (kind, value)}`` for channel ``None`` (closed
        shell), ``"ALPHA"`` and ``"BETA"``; kind is ``"direct"``,
        ``"indirect"`` or ``"conducting"`` (value 0.0)."""
        return {chan: (b['kind'], b['value'])
                for chan, b in cls._final_scf_blocks(content).items()}

    @classmethod
    def _final_scf_blocks(cls, content: str) -> Dict[Any, Dict[str, Any]]:
        """final_band_gaps with each record's band edges.

        ``{channel: {'kind', 'value', 'tops', 'bottoms'}}``; ``tops`` and
        ``bottoms`` are ``(eig_au, k)`` of the block's ``TOP OF VALENCE
        BANDS`` and ``BOTTOM OF VIRTUAL BANDS`` lines (CRYSTAL prints more
        than one when the direct gap is at another k than the indirect one,
        and a BOTTOM line can follow the gap line). In a multi-point OPT the
        blocks are point 1's, so there are none for the final geometry."""
        content = cls._after_last_opt_point(content)
        final: Dict[Any, Dict[str, Any]] = {}
        chan = None
        edges: Dict[str, list] = {'tops': [], 'bottoms': []}
        collecting = False    # inside a channel's block, where edge lines go
        for line in content.splitlines():
            m = cls._SPIN_MARK.match(line)
            if m:
                chan = m.group(1)
                edges, collecting = {'tops': [], 'bottoms': []}, True
                continue
            if 'INSULATING STATE' in line:
                edges, collecting = {'tops': [], 'bottoms': []}, True
                continue
            m = cls._EDGE_LINE.search(line)
            if m:
                if collecting:
                    key = 'tops' if m.group(1) == 'TOP' else 'bottoms'
                    edges[key].append((float(m.group(5)), int(m.group(4))))
                continue
            if 'CONDUCTING STATE' in line:
                rec = ('conducting', 0.0)
            else:
                m = cls._GAP_LINE.search(line)
                if not m:
                    if line.strip():
                        collecting = False
                    continue
                rec = (m.group(1).lower(), float(m.group(2)))
            # BOTTOM lines after the gap line still belong to this record
            block = {'kind': rec[0], 'value': rec[1], **edges}
            # a cycle's records: ALPHA then BETA, or one for both channels
            # (closed shell, or a spin run whose Fermi level is common)
            if chan == 'BETA':
                final['BETA'] = block
            else:
                final = {chan: block}
            chan = None
        return final

    @classmethod
    def _fundamental_gap(cls, blocks: Dict[Any, Dict[str, Any]]):
        """The fundamental gap of the last SCF of a spin-polarized run.

        min(CBM over channels) - max(VBM over channels), in eV: a channel's
        own gap is only an upper bound, the valence-band top can be in one
        channel and the conduction-band bottom in the other. Returns
        ``(kind, value)`` -- kind ``"direct"`` when VBM and CBM are at the
        same k point (k decides direct/indirect even when they are in
        different channels), ``"conducting"`` (0.0) when the edges overlap --
        or None when the blocks are not two gapped channels with their band
        edges (closed shell, conducting, or edge lines missing). When VBM and
        CBM are in the same channel the gap is that channel's printed one."""
        if set(blocks) != {'ALPHA', 'BETA'}:
            return None
        if any(b['kind'] not in ('direct', 'indirect') or not b['tops'] or not b['bottoms']
               for b in blocks.values()):
            return None
        vbm_chan, (vbm, vbm_k) = max(((c, max(b['tops'])) for c, b in blocks.items()),
                                     key=lambda cv: cv[1][0])
        cbm_chan, (cbm, cbm_k) = min(((c, min(b['bottoms'])) for c, b in blocks.items()),
                                     key=lambda cv: cv[1][0])
        if vbm_chan == cbm_chan:
            b = blocks[vbm_chan]
            return (b['kind'], b['value'])
        gap = round((cbm - vbm) * HARTREE_TO_EV, 4)
        if gap <= 0:
            return ('conducting', 0.0)
        return ('direct' if vbm_k == cbm_k else 'indirect', gap)

    def _extract_electronic_properties(self, content: str) -> Dict[str, Any]:
        """Extract electronic properties."""
        return self._band_gap_properties(content)

    def _band_gap_properties(self, content: str) -> Dict[str, Any]:
        """band_gap, direct/indirect_band_gap, band_gap_type, alpha/beta_band_gap
        and spin_polarized, all from the last SCF (see final_band_gaps)."""
        props: Dict[str, Any] = {}

        # CRYSTAL's own summary after the run (when it prints one): the last one
        alpha_summary = re.findall(r'ALPHA BAND GAP:\s*([\d.]+)\s*eV', content)
        beta_summary = re.findall(r'BETA BAND GAP:\s*([\d.]+)\s*eV', content)
        blocks = self._final_scf_blocks(content)
        final = {chan: (b['kind'], b['value']) for chan, b in blocks.items()}
        # spin-polarized: the fundamental gap across both channels, not the
        # smaller channel gap (an upper bound); None without the band edges
        fundamental = self._fundamental_gap(blocks)

        if alpha_summary and beta_summary:
            # Spin-polarized calculation
            props.update({
                'alpha_band_gap': float(alpha_summary[-1]),
                'beta_band_gap': float(beta_summary[-1]),
                'spin_polarized': True
            })
            # the material's gap: the fundamental gap of the last SCF when its
            # band edges are printed for the final geometry, else the smaller
            # channel gap of the summary (a multi-point OPT prints no edges
            # after point 1)
            props['band_gap'] = (fundamental[1] if fundamental is not None
                                 else min(props['alpha_band_gap'], props['beta_band_gap']))
        else:
            plain = re.findall(r'^\s*BAND GAP:\s*([\d.]+)\s*eV', content, re.M)
            if plain:
                props['band_gap'] = float(plain[-1])
                props['spin_polarized'] = False
            elif final and self._GAP_LINE.search(content):
                # the last SCF: its fundamental gap (spin-polarized), else the
                # smaller gap of its spin channels (0 when it ended conducting
                # after gapped cycles), not the last line of an earlier cycle
                props['band_gap'] = (fundamental[1] if fundamental is not None
                                     else min(v for _, v in final.values()))
                # an unrestricted run whose last cycle is conducting prints one
                # common-EFERMI block, not ALPHA and BETA ones
                props['spin_polarized'] = ({'ALPHA', 'BETA'} <= set(blocks)
                                           or 'UNRESTRICTED OPEN SHELL' in content)
            elif final and all(kind == 'conducting' for kind, _ in final.values()):
                # conducting in every cycle (no gap line at all): 0, as for
                # a run that ended conducting and a multi-point OPT ending
                # conducting. One record serves both spin channels of an
                # unrestricted run here (a common Fermi level).
                props['band_gap'] = 0.0
                props['spin_polarized'] = ({'ALPHA', 'BETA'} <= set(blocks)
                                           or 'UNRESTRICTED OPEN SHELL' in content)
            elif (not final and self._GAP_LINE.search(content)
                  and re.search(r'OPT END\b.*?^\s*FERMI ENERGY:', content, re.S | re.M)):
                # a multi-point OPT (its SCF blocks are point 1's) whose final
                # geometry is conducting: CRYSTAL prints the FERMI ENERGY
                # instead of a BAND GAP after OPT END. 0 as for a run that
                # ended conducting after gapped cycles.
                props['band_gap'] = 0.0
                props['spin_polarized'] = bool(
                    re.search(r'^\s*BETA\s+ELECTRONS\s*$', content, re.M)
                    or 'UNRESTRICTED OPEN SHELL' in content)

        # Direct vs indirect: the kind of the last SCF's gap (the fundamental
        # gap of a spin-polarized run, by the k points of its VBM and CBM, else
        # the smaller of the two spin channels), never a gap of an earlier
        # cycle. Taking the last DIRECT line and the last INDIRECT line
        # separately stored cycle 0's INDIRECT gap of an SP whose converged gap
        # is direct.
        insulating = [kv for kv in final.values() if kv[0] in ('direct', 'indirect')]
        if fundamental is not None:
            if fundamental[0] != 'conducting':
                props[f'{fundamental[0]}_band_gap'] = fundamental[1]
                props['band_gap_type'] = fundamental[0]
        elif insulating and len(insulating) == len(final):
            kind, value = min(insulating, key=lambda kv: kv[1])
            props[f'{kind}_band_gap'] = value
            props['band_gap_type'] = kind
        return props

    def _final_gap_value(self, content: str) -> Optional[float]:
        """The material's gap after the last SCF (as band_gap reports it)."""
        p = self._band_gap_properties(content)
        if 'band_gap' in p:
            return p['band_gap']
        # with CRYSTAL's ALPHA/BETA summary: the fundamental gap of the last
        # SCF (as direct/indirect_band_gap report it), else the smaller channel
        fundamental = self._fundamental_gap(self._final_scf_blocks(content))
        if fundamental is not None:
            return fundamental[1]
        gaps = [p[k] for k in ('alpha_band_gap', 'beta_band_gap') if k in p]
        return min(gaps) if gaps else None

    def _extract_energy_properties(self, content: str) -> Dict[str, Any]:
        """Extract energy-related properties."""
        props = {}
        
        # Total energy (final value from DFT calculation)
        energy_matches = re.findall(r'TOTAL ENERGY\(DFT\)\(AU\)\s*\(\s*\d+\)\s*([-\d.E+]+)', content)
        if energy_matches:
            props['total_energy_au'] = float(energy_matches[-1])
            props['total_energy_ev'] = float(energy_matches[-1]) * HARTREE_TO_EV  # Hartree to eV
        
        # Alternative energy extraction patterns
        if not energy_matches:
            # From SCF convergence
            scf_energy_match = re.search(r'== SCF ENDED - CONVERGENCE ON ENERGY\s+E\(AU\)\s*([-\d.E+]+)', content)
            if scf_energy_match:
                props['total_energy_au'] = float(scf_energy_match.group(1))
                props['total_energy_ev'] = float(scf_energy_match.group(1)) * HARTREE_TO_EV
            else:
                # From final optimization energy
                final_energy_match = re.search(r'E\(AU\):\s*([-\d.E+]+)', content)
                if final_energy_match:
                    props['total_energy_au'] = float(final_energy_match.group(1))
                    props['total_energy_ev'] = float(final_energy_match.group(1)) * HARTREE_TO_EV
        
        # D3 dispersion correction — take the LAST occurrence; the first is
        # printed for the initial geometry and is inconsistent with the
        # final total energy (which is taken last)
        d3_matches = re.findall(r'D3 DISPERSION ENERGY \(AU\)\s*([-\d.E+]+)', content)
        d3_energy = None
        if d3_matches:
            d3_energy = float(d3_matches[-1])
            props['d3_dispersion_energy_au'] = d3_energy
            props['d3_dispersion_energy_ev'] = d3_energy * HARTREE_TO_EV

            # Total energy + dispersion (D3 only). Kept under its historical
            # name for back-compat; for 3C methods this is NOT the fully
            # corrected total (those also add gCP — see total_energy_corrected_au).
            total_plus_disp_matches = re.findall(r'TOTAL ENERGY \+ DISP \(AU\)\s*([-\d.E+]+)', content)
            if total_plus_disp_matches:
                props['total_energy_plus_d3_au'] = float(total_plus_disp_matches[-1])
                props['total_energy_plus_d3_ev'] = float(total_plus_disp_matches[-1]) * HARTREE_TO_EV
            elif 'total_energy_au' in props:
                # Calculate if not explicitly given
                props['total_energy_plus_d3_au'] = props['total_energy_au'] + d3_energy
                props['total_energy_plus_d3_ev'] = props['total_energy_plus_d3_au'] * HARTREE_TO_EV

        # gCP (geometrical counterpoise) correction — printed by 3C methods
        # (e.g. HSESOL3C) alongside D3. Omitting it silently understates the
        # corrected total by the gCP term (~0.17 AU ≈ 108 kcal/mol).
        gcp_matches = re.findall(r'GCP ENERGY \(AU\)\s*([-\d.E+]+)', content)
        gcp_energy = None
        if gcp_matches:
            gcp_energy = float(gcp_matches[-1])
            props['gcp_energy_au'] = gcp_energy
            props['gcp_energy_ev'] = gcp_energy * HARTREE_TO_EV

        # From the second point of an OPT on (and from the first point of a
        # restarted one), CRYSTAL's TOTAL ENERGY(DFT) line already includes D3
        # and gCP: that point's DE is measured from the previous point's
        # corrected energy, and OPT END's E(AU) equals the last such line.
        # Adding the D3/gCP printed after OPT END to it counted them twice
        # (4LG 3c OPT by -0.094 AU). Then OPT END's energy is the corrected
        # total and the plain DFT energy is it minus the final corrections. A
        # one-point OPT's DFT line is the plain energy (OPT END = it + D3 +
        # gCP), and is left as is.
        #
        # An OPT stopped before OPT END (wall time, killed) after its second
        # point: its last DFT line is the corrected total in the same way.
        # The plain DFT energy is then that minus the last D3/gCP printed
        # (point 1's: none is printed for the later points). Stopped inside
        # point 2's SCF, the last DFT line is point 1's plain energy.
        opt_end = re.findall(r'OPT END\b[^\n]*?E\(AU\):\s*([-\d.E+]+)', content)
        corrections = (d3_energy or 0.0) + (gcp_energy or 0.0)
        unfinished_opt = False
        if not opt_end and energy_matches and 'OPT END' not in content:
            points = [m.start() for m in self._OPT_POINT.finditer(content)]
            last_dft = content.rfind('TOTAL ENERGY(DFT)(AU)')
            # the last DFT line is of point 2 or later (not point 1's, as
            # when the run was stopped inside point 2's SCF)
            unfinished_opt = len(points) >= 2 and last_dft > points[1]
        if energy_matches and abs(corrections) > 1e-6 and (
                (opt_end and abs(float(opt_end[-1]) - float(energy_matches[-1])) < 1e-6)
                or unfinished_opt):
            end = float(opt_end[-1]) if opt_end else float(energy_matches[-1])
            props['total_energy_au'] = end - corrections
            props['total_energy_ev'] = props['total_energy_au'] * HARTREE_TO_EV
            if d3_energy is not None:
                props['total_energy_plus_d3_au'] = end - (gcp_energy or 0.0)
                props['total_energy_plus_d3_ev'] = props['total_energy_plus_d3_au'] * HARTREE_TO_EV
            props['total_energy_corrected_au'] = end
            props['total_energy_corrected_ev'] = end * HARTREE_TO_EV
        # Fully corrected total energy = total + D3 + gCP, built from the
        # FINAL-geometry components (total_energy_au, d3_energy, gcp_energy are
        # all taken as last occurrences = the converged geometry). We do NOT
        # prefer CRYSTAL's printed "TOTAL ENERGY + DISP + GCP" line: in an OPT
        # that combined line is emitted only for the INITIAL geometry and is
        # never re-printed after OPT END, so trusting it yields a stale total
        # (4LG AA OPT: by 1.2e-4 AU). For single-geometry SP runs the
        # computed sum matches the printed line to ~1e-10. Only fall back to the
        # printed line when the components themselves are unavailable.
        elif 'total_energy_au' in props and (d3_energy is not None or gcp_energy is not None):
            corrected = props['total_energy_au'] + (d3_energy or 0.0) + (gcp_energy or 0.0)
            props['total_energy_corrected_au'] = corrected
            props['total_energy_corrected_ev'] = corrected * HARTREE_TO_EV
        else:
            corrected_line = (re.findall(r'TOTAL ENERGY \+ DISP \+ GCP \(AU\)\s*([-\d.E+]+)', content)
                              or re.findall(r'TOTAL ENERGY \+ DISP \(AU\)\s*([-\d.E+]+)', content))
            if corrected_line:
                props['total_energy_corrected_au'] = float(corrected_line[-1])
                props['total_energy_corrected_ev'] = float(corrected_line[-1]) * HARTREE_TO_EV

        # A FREQ run restarted from a wavefunction prints no SCF energy: its
        # displacement table's CENTRAL POINT energy is the corrected total
        # (equal to the corrected total, D3/gCP included, wherever both are
        # printed). The plain DFT energy and the corrections are not printed.
        if 'total_energy_au' not in props and 'total_energy_corrected_au' not in props:
            central = re.search(r'^\s*CENTRAL POINT\s+(-?\d+\.\d+E[+-]\d+)', content, re.M)
            if central:
                props['total_energy_corrected_au'] = float(central.group(1))
                props['total_energy_corrected_ev'] = float(central.group(1)) * HARTREE_TO_EV

        # Extract energy components if available
        energy_components = self._extract_energy_components(content)
        props.update(energy_components)
        
        return props
    
    def _extract_energy_components(self, content: str) -> Dict[str, Any]:
        """Extract detailed energy components."""
        components = {}
        
        # Look for energy breakdown section (an OPT prints it at point 1 only:
        # in a multi-point OPT it is not the final geometry's)
        content = self._after_last_opt_point(content)
        if '+++ ENERGIES IN A.U. +++' in content:
            component_patterns = {
                'kinetic_energy_au': r'KINETIC ENERGY\s*([-\d.E+]+)',
                'electron_electron_energy_au': r'TOTAL E-E\s*([-\d.E+]+)',
                'electron_nuclear_energy_au': r'TOTAL E-N \+ N-E\s*([-\d.E+]+)',
                'nuclear_nuclear_energy_au': r'TOTAL N-N\s*([-\d.E+]+)',
                'exchange_energy_au': r'EXCHANGE ENERGY:\s*([-\d.E+]+)',
                'correlation_energy_au': r'CORRELATION ENERGY:\s*([-\d.E+]+)'
            }
            
            for prop_name, pattern in component_patterns.items():
                match = re.search(pattern, content)
                if match:
                    au_value = float(match.group(1))
                    components[prop_name] = au_value
                    # Also add eV version
                    ev_name = prop_name.replace('_au', '_ev')
                    components[ev_name] = au_value * HARTREE_TO_EV
        
        return components
    
    def _extract_population_analysis(self, content: str) -> Dict[str, Any]:
        """Extract Mulliken population analysis."""
        props = {}
        
        # Check for spin-resolved sections (both alpha+beta and alpha-beta)
        has_alpha_beta = "ALPHA+BETA ELECTRONS" in content
        has_alpha_minus_beta = "ALPHA-BETA ELECTRONS" in content
        is_spin_polarized = has_alpha_beta and has_alpha_minus_beta
        
        # Extract alpha+beta populations when available (even if not fully spin-polarized)
        if has_alpha_beta:
            alpha_beta_mulliken = self._extract_mulliken_section(content, "ALPHA+BETA ELECTRONS")
            alpha_beta_overlap = self._extract_overlap_populations(content, "ALPHA+BETA ELECTRONS")
            
            if alpha_beta_mulliken:
                props['mulliken_alpha_plus_beta'] = alpha_beta_mulliken
            if alpha_beta_overlap:
                props['overlap_population_alpha_plus_beta'] = alpha_beta_overlap
        
        # Extract alpha-beta populations when available
        if has_alpha_minus_beta:
            alpha_minus_beta_mulliken = self._extract_mulliken_section(content, "ALPHA-BETA ELECTRONS")
            alpha_minus_beta_overlap = self._extract_overlap_populations(content, "ALPHA-BETA ELECTRONS")
            
            if alpha_minus_beta_mulliken:
                props['mulliken_alpha_minus_beta'] = alpha_minus_beta_mulliken
            if alpha_minus_beta_overlap:
                props['overlap_population_alpha_minus_beta'] = alpha_minus_beta_overlap
        
        # Always try to extract general populations as fallback
        # This ensures we get overlap data even if spin-resolved extraction fails
        general_mulliken = self._extract_general_mulliken_section(content)
        if general_mulliken and not has_alpha_beta:  # Only use general if no alpha+beta
            props['mulliken_population'] = general_mulliken
            
        # Extract general overlap populations as fallback
        general_overlap = self._extract_general_overlap_populations(content)
        if general_overlap and not has_alpha_beta:  # Only use general if no alpha+beta
            props['overlap_population'] = general_overlap
        
        # both spin tables printed: one sign of an unrestricted run, merged
        # into spin_polarized by extract_all_properties (not stored itself)
        props['_mulliken_spin'] = is_spin_polarized

        # Valence charges of pseudopotential atoms: their Mulliken populations
        # count valence electrons only (see population_analysis_processor).
        if any(k.startswith('mulliken_') for k in props):
            try:
                from mace.utils.population_analysis_processor import valence_charges
            except ImportError:
                from population_analysis_processor import valence_charges
            valence = valence_charges(content)
            if valence:
                props['mulliken_valence_charges'] = {str(z): v for z, v in valence.items()}

        return props
    
    def _extract_geometry_optimization(self, content: str) -> Dict[str, Any]:
        """Extract geometry optimization information."""
        props = {}
        
        # Check if this was an optimization. From lines CRYSTAL prints, not a
        # bare 'OPTGEOM' search: the echoed title of an SP/FREQ/BAND/DOSS run
        # named "..._BULK_OPTGEOM_..." holds that word too.
        if is_optimization_output(content):
            props['calculation_type'] = 'OPT'
            
            # Convergence information. Runs that converge at the first point
            # print "OPT END - CONVERGED" without the convergence-tests line,
            # so treat either as converged.
            converged_match = re.search(r'CONVERGENCE TESTS SATISFIED AFTER\s+(\d+)\s+ENERGY AND GRADIENT CALCULATIONS', content)
            opt_end_match = re.search(r'OPT END - CONVERGED.*?POINTS\s+(\d+)', content)
            if converged_match:
                props['optimization_cycles'] = int(converged_match.group(1))
                props['optimization_converged'] = True
            elif opt_end_match:
                props['optimization_cycles'] = int(opt_end_match.group(1))
                props['optimization_converged'] = True
            else:
                props['optimization_converged'] = False

            # Final gradient — last occurrence; the first GRADIENT NORM line
            # is the initial geometry's
            grad_norm_matches = re.findall(r'GRADIENT NORM\s+([\d.E+-]+)', content)
            if grad_norm_matches:
                props['final_gradient_norm'] = float(grad_norm_matches[-1])
        
        # Check for single point calculation: CRYSTAL23 prints neither
        # phrase for an SP, but its SCF header (a FREQ run's is overridden
        # by the frequency extraction)
        elif ('SINGLE POINT CALCULATION' in content or 'SCFDIR' in content
              or 'CRYSTAL - SCF - TYPE OF CALCULATION' in content):
            props['calculation_type'] = 'SP'
        
        return props
    
    def _extract_crystallographic_info(self, content: str) -> Dict[str, Any]:
        """Extract crystallographic information."""
        props = {}
        
        # Space group information
        space_group_match = re.search(r'SPACE GROUP NUMBER\s+(\d+)', content)
        if space_group_match:
            props['space_group_number'] = int(space_group_match.group(1))
        
        # Crystal system
        if 'CUBIC' in content:
            props['crystal_system'] = 'cubic'
        elif 'TETRAGONAL' in content:
            props['crystal_system'] = 'tetragonal'
        elif 'ORTHORHOMBIC' in content:
            props['crystal_system'] = 'orthorhombic'
        elif 'HEXAGONAL' in content:
            props['crystal_system'] = 'hexagonal'
        elif 'TRIGONAL' in content:
            props['crystal_system'] = 'trigonal'
        elif 'MONOCLINIC' in content:
            props['crystal_system'] = 'monoclinic'
        elif 'TRICLINIC' in content:
            props['crystal_system'] = 'triclinic'
        
        # Centering code
        centering_match = re.search(r'CENTRING CODE\s+([\d/]+)', content)
        if centering_match:
            props['centering_code'] = centering_match.group(1)

        # A slab's layer group: CRYSTAL prints it as the two-sided plane
        # group ("TWO-SIDED PLANE GROUP N.  1          :  P 1")
        layer = re.search(r'^\s*TWO-SIDED PLANE GROUP N\.\s*(\d+)\s*:\s*(\S.*?)\s*$', content, re.M)
        if layer:
            props['layer_group_number'] = int(layer.group(1))
            props['layer_group_symbol'] = layer.group(2)
        
        return props
    
    def _extract_neighbor_information(self, content: str) -> Dict[str, Any]:
        """Extract neighbor information from CRYSTAL output.

        OPT outputs print the neighbor table twice: once for the initial
        geometry and again after the final optimized geometry. The first
        table keeps the original property names; when a later table exists,
        it is stored under final_* property names.
        """
        props = {}

        neighbor_section_matches = list(re.finditer(
            r'NEIGHBORS OF THE NON-EQUIVALENT ATOMS.*?N = NUMBER OF NEIGHBORS AT DISTANCE R\s*\n\s*ATOM\s+N\s+R/ANG\s+R/AU\s+NEIGHBORS.*?\n(.*?)(?=\n\s*SYMMETRY|\n\s*TTTT|\n\s*MMMM|\n\s*[A-Z]{3,}|$)',
            content, re.DOTALL
        ))

        if not neighbor_section_matches:
            return props

        props.update(self._parse_neighbor_section(neighbor_section_matches[0].group(1).strip()))

        if len(neighbor_section_matches) > 1:
            final_props = self._parse_neighbor_section(neighbor_section_matches[-1].group(1).strip())
            props.update({f'final_{name}': value for name, value in final_props.items()})

        return props

    def _parse_neighbor_section(self, neighbor_content: str) -> Dict[str, Any]:
        """Parse a single neighbor analysis table into property values."""
        props = {}
        neighbors_data = []
        coordination_numbers = []
        bond_distances = []
        
        # Parse each line for neighbor information
        # Pattern: ATOM_NUM ELEMENT N_NEIGHBORS DISTANCE_ANG DISTANCE_AU NEIGHBOR_LIST
        lines = neighbor_content.split('\n')
        current_atom = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line starts with atom number and element.
            # Shell header rows carry decimal R/ANG and R/AU values; wrapped
            # continuation rows hold only integer cell indices, so requiring
            # decimal points avoids misreading 5-token continuation lines
            # (e.g. "2 C    0 0 1") as new shells with bogus distances.
            parts = line.split()
            if (len(parts) >= 5 and parts[0].isdigit() and parts[1].isalpha()
                    and '.' in parts[3] and '.' in parts[4]):
                try:
                    atom_num = int(parts[0])
                    element = parts[1]
                    n_neighbors = int(parts[2])
                    dist_ang = float(parts[3])
                    dist_au = float(parts[4])
                    
                    # Find or create atom entry
                    current_atom = None
                    for atom in neighbors_data:
                        if atom['atom_number'] == atom_num:
                            current_atom = atom
                            break
                    
                    if current_atom is None:
                        current_atom = {
                            'atom_number': atom_num,
                            'element': element,
                            'neighbor_shells': []
                        }
                        neighbors_data.append(current_atom)
                    
                    # Add this neighbor shell
                    neighbor_info = ' '.join(parts[5:]) if len(parts) > 5 else ''
                    neighbors = self._parse_neighbor_list(neighbor_info)
                    
                    shell = {
                        'n_neighbors': n_neighbors,
                        'distance_ang': dist_ang,
                        'distance_au': dist_au,
                        'neighbors': neighbors
                    }
                    current_atom['neighbor_shells'].append(shell)
                    
                    # Track statistics
                    coordination_numbers.append(n_neighbors)
                    if dist_ang > 0:
                        bond_distances.append(dist_ang)
                        
                except (ValueError, IndexError):
                    continue
            
            elif current_atom is not None and line and not line.startswith(' ' * 50):
                # This might be a continuation line with more neighbors
                if current_atom['neighbor_shells']:
                    last_shell = current_atom['neighbor_shells'][-1]
                    neighbors = self._parse_neighbor_list(line)
                    last_shell['neighbors'].extend(neighbors)
        
        if neighbors_data:
            props['neighbor_analysis'] = neighbors_data
            
            # Extract summary statistics
            if coordination_numbers:
                props['max_coordination_number'] = max(coordination_numbers)
                props['total_coordination_shells'] = len(coordination_numbers)
            
            if bond_distances:
                props['min_bond_distance_ang'] = min(bond_distances)
                props['max_bond_distance_ang'] = max(bond_distances)
        
        return props
    
    def _parse_neighbor_list(self, neighbor_text: str) -> List[Dict]:
        """Parse neighbor list from text like '2 C 0 0 0 2 C 1 0 0'."""
        neighbors = []
        parts = neighbor_text.split()
        
        # Process in groups of 5: atom_num element i j k
        i = 0
        while i + 4 < len(parts):
            try:
                atom_num = int(parts[i])
                element = parts[i + 1]
                cell_i = int(parts[i + 2])
                cell_j = int(parts[i + 3])
                cell_k = int(parts[i + 4])
                
                neighbors.append({
                    'neighbor_atom_number': atom_num,
                    'neighbor_element': element,
                    'cell_indices': [cell_i, cell_j, cell_k]
                })
                
                i += 5
            except (ValueError, IndexError):
                i += 1
                continue
        
        return neighbors
    
    def _extract_atomic_positions(self, content: str, search_final: bool = True) -> List[Dict]:
        """Extract atomic positions from geometry."""
        positions = []
        
        if search_final:
            # Look for final atomic positions
            # Stop at TRANSFORMATION too: capturing through to "T = ATOM"
            # (printed only after the crystallographic table) spanned both
            # the primitive and crystallographic tables, doubling every atom
            final_geom_match = re.search(
                r'FINAL OPTIMIZED GEOMETRY.*?ATOM\s+X/A\s+Y/B\s+Z/C\s*\n\s*\*+\s*\n(.*?)\n\s*(?:T = ATOM|TRANSFORMATION)',
                content, re.DOTALL
            )
            
            if final_geom_match:
                atom_lines = final_geom_match.group(1).strip().split('\n')
                positions = self._parse_atomic_position_lines(atom_lines)
            else:
                # A slab (X/A Y/B Z(ANGSTROM)), polymer (X/A Y(ANGSTROM)
                # Z(ANGSTROM)) or molecule (all in ANGSTROM): the table's
                # header names each column's unit (atomic_position_axes)
                final_geom_match = re.search(
                    r'FINAL OPTIMIZED GEOMETRY.*?ATOM\s+(?:X/A|X\(ANGSTROM\))\s+(?:Y/B|Y\(ANGSTROM\))'
                    r'\s+(?:Z/C|Z\(ANGSTROM\))\s*\n\s*\*+\s*\n(.*?)\n\s*(?:T = ATOM|TRANSFORMATION)',
                    content, re.DOTALL
                )
                if final_geom_match:
                    atom_lines = final_geom_match.group(1).strip().split('\n')
                    positions = self._parse_atomic_position_lines(atom_lines)
        
        if not positions:
            # Look for initial/general atomic positions
            geom_patterns = [
                r'GEOMETRY FOR WAVE FUNCTION.*?ATOM\s+X/A\s+Y/B\s+Z/C\s*\n\s*\*+\s*\n(.*?)\n\s*(?:T = ATOM|TRANSFORMATION)',
                r'ATOMS IN THE ASYMMETRIC UNIT.*?ATOM\s+X/A\s+Y/B\s+Z/C\s*\n\s*\*+\s*\n(.*?)\n\s*(?:T = ATOM|TRANSFORMATION)',
                r'ATOM\s+X/A\s+Y/B\s+Z/C\s*\n\s*\*+\s*\n(.*?)\n\s*(?:T = ATOM|TRANSFORMATION)'
            ]
            
            for pattern in geom_patterns:
                geom_match = re.search(pattern, content, re.DOTALL)
                if geom_match:
                    atom_lines = geom_match.group(1).strip().split('\n')
                    positions = self._parse_atomic_position_lines(atom_lines)
                    if positions:
                        break
        
        return positions
    
    def _parse_atomic_position_lines(self, atom_lines: List[str]) -> List[Dict]:
        """Parse atomic position lines into structured data."""
        positions = []
        
        for line in atom_lines:
            line = line.strip()
            if not line or line.startswith('*'):
                continue
            
            parts = line.split()
            if len(parts) >= 7:  # Need at least 7 parts for CRYSTAL format
                try:
                    # CRYSTAL format: ATOM_NUM ATOM_TYPE ATOMIC_NUM ELEMENT X Y Z
                    positions.append({
                        'atom_number': int(parts[0]),
                        'atom_type': parts[1],
                        'atomic_number': int(parts[2]),
                        'element': parts[3],
                        'x': float(parts[4]),
                        'y': float(parts[5]),
                        'z': float(parts[6])
                    })
                except (ValueError, IndexError):
                    continue
            elif len(parts) >= 6:  # Fallback for simpler format
                try:
                    positions.append({
                        'atom_number': int(parts[0]),
                        'atom_type': parts[1],
                        'element': parts[2],
                        'x': float(parts[3]),
                        'y': float(parts[4]),
                        'z': float(parts[5])
                    })
                except (ValueError, IndexError):
                    continue
        
        return positions
    
    def _extract_mulliken_section(self, content: str, section_type: str) -> Dict[str, Any]:
        """Extract Mulliken population analysis for a specific section."""
        # The section header line is printed IMMEDIATELY above the Mulliken
        # header; a loose `.*?` join matched the SPINLOCK echo line
        # ("ALPHA-BETA ELECTRONS LOCKED TO ...") and then skipped forward to
        # the alpha+beta table, storing full populations as spin densities.
        # The capture is bounded so it cannot run into the other spin section
        # or the overlap tables (which doubled atom counts), and the LAST
        # PPAN print is used (earlier ones belong to earlier SCF/geometries).
        escaped_section = re.escape(section_type)
        pattern = (f'{escaped_section}\\s*\\n\\s*MULLIKEN POPULATION ANALYSIS[^\\n]*?'
                   f'NO. OF ELECTRONS\\s+([\\d.-]+)(.*?)'
                   f'(?=ALPHA\\+BETA ELECTRONS|ALPHA-BETA ELECTRONS|'
                   f'OVERLAP POPULATION|MMMMM|TTTTTT|$)')
        matches = list(re.finditer(pattern, content, re.DOTALL))

        if not matches:
            return None

        match = matches[-1]
        total_electrons = float(match.group(1))
        section_content = match.group(2)
        
        # The section holds two tables with identical atom rows (A.O.
        # POPULATION followed by SHELL POPULATION) — parse only the first
        # so atoms aren't double-counted
        section_content = re.split(r'ATOM\s+Z\s+CHARGE\s+SHELL\s+POPULATION', section_content)[0]

        atoms = self._mulliken_ao_rows(section_content)
        
        return {
            'total_electrons': total_electrons,
            'atoms': atoms
        }
    
    # An A.O. population row ("   1 O    8  8.632  1.242 ...") and the
    # indented lines continuing it (8 populations per line)
    _MULLIKEN_ROW = re.compile(r'^\s*(\d+)\s+([A-Za-z]{1,2})\s+(\d+)\s+(-?\d+\.\d+)(.*)$')
    _MULLIKEN_CONT = re.compile(r'^\s+-?\d+\.\d+(?:\s*-?\d+\.\d+)*\s*$')
    _POPULATION = re.compile(r'-?\d+\.\d+')

    @classmethod
    def _mulliken_ao_rows(cls, table: str) -> List[Dict[str, Any]]:
        """The atom rows of a Mulliken A.O. POPULATION table, each with all
        its populations (continuation lines appended)."""
        atoms = []
        for line in table.splitlines():
            row = cls._MULLIKEN_ROW.match(line)
            if row:
                atoms.append({
                    'atom_number': int(row.group(1)),
                    'element': row.group(2),
                    'atomic_number': int(row.group(3)),
                    # the atom's electron population (CRYSTAL's CHARGE
                    # column), not its net charge
                    'mulliken_population': float(row.group(4)),
                    'orbital_populations': [float(x) for x in cls._POPULATION.findall(row.group(5))],
                })
            elif atoms and cls._MULLIKEN_CONT.match(line):
                atoms[-1]['orbital_populations'].extend(
                    float(x) for x in cls._POPULATION.findall(line))
            elif atoms and line.strip():
                break  # end of the table
        return atoms

    def _extract_overlap_populations(self, content: str, section_type: str) -> List[Dict]:
        """Extract overlap population data for spin-resolved sections."""
        # First, check if we're in a spin-resolved section
        if section_type in content:
            # Find the section and then look for overlap population after it
            section_start = content.find(section_type)
            remaining_content = content[section_start:]
            
            # Look for overlap population section in the remaining content
            overlap_match = re.search(r'OVERLAP POPULATION CONDENSED TO ATOMS(.*?)(?=EIGENVECTORS|MMMMM|TTTTTT|ALPHA|BETA|$)', 
                                    remaining_content, re.DOTALL)
        else:
            # Fallback to general search
            overlap_match = re.search(r'OVERLAP POPULATION CONDENSED TO ATOMS(.*?)(?=EIGENVECTORS|MMMMM|TTTTTT|$)', 
                                    content, re.DOTALL)
        
        if not overlap_match:
            return []
        
        overlap_content = overlap_match.group(1)
        overlaps = []
        
        # Parse overlap data with improved pattern
        # Look for ATOM A lines followed by neighbor data
        atom_pattern = r'ATOM A\s+(\d+)\s+(\w+)\s+ATOM B.*?\n(.*?)(?=ATOM A|\Z)'
        atom_matches = re.finditer(atom_pattern, overlap_content, re.DOTALL)
        
        for atom_match in atom_matches:
            atom_a_num = int(atom_match.group(1))
            atom_a_element = atom_match.group(2)
            neighbor_section = atom_match.group(3)
            
            # Parse neighbor interactions with more flexible parsing
            neighbors = []
            neighbor_lines = neighbor_section.strip().split('\n')
            
            for line in neighbor_lines:
                line = line.strip()
                if not line or 'ATOM' in line or 'CELL' in line:
                    continue
                
                # Look for neighbor line format: ATOM_NUM ELEMENT (CELL_I CELL_J CELL_K) DIST_AU DIST_ANG OVERLAP
                # or format: ATOM_NUM ELEMENT CELL_I CELL_J CELL_K DIST_AU DIST_ANG OVERLAP
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        # Handle format with parentheses: 2 C ( 0 0 0) 2.922 1.546 0.293
                        if '(' in line and ')' in line:
                            cell_match = re.search(r'\(\s*([-\d]+)\s+([-\d]+)\s+([-\d]+)\s*\)', line)
                            if cell_match:
                                atom_b_num = int(parts[0])
                                atom_b_element = parts[1]
                                cell_i = int(cell_match.group(1))
                                cell_j = int(cell_match.group(2))
                                cell_k = int(cell_match.group(3))
                                
                                # Find numeric values after the parentheses
                                after_paren = line.split(')')[1].strip().split()
                                if len(after_paren) >= 3:
                                    distance_au = float(after_paren[0])
                                    distance_ang = float(after_paren[1])
                                    overlap_pop = float(after_paren[2])
                                    
                                    neighbors.append({
                                        'atom_b_number': atom_b_num,
                                        'atom_b_element': atom_b_element,
                                        'cell_indices': [cell_i, cell_j, cell_k],
                                        'distance_au': distance_au,
                                        'distance_ang': distance_ang,
                                        'overlap_population': overlap_pop
                                    })
                        
                        # Handle format without parentheses: ATOM_NUM ELEMENT CELL_I CELL_J CELL_K DIST_AU DIST_ANG OVERLAP
                        elif len(parts) >= 8:
                            neighbors.append({
                                'atom_b_number': int(parts[0]),
                                'atom_b_element': parts[1],
                                'cell_indices': [int(parts[2]), int(parts[3]), int(parts[4])],
                                'distance_au': float(parts[5]),
                                'distance_ang': float(parts[6]),
                                'overlap_population': float(parts[7])
                            })
                    except (ValueError, IndexError):
                        continue
            
            if neighbors:  # Only add atoms that have neighbor data
                overlaps.append({
                    'atom_a_number': atom_a_num,
                    'atom_a_element': atom_a_element,
                    'neighbors': neighbors
                })
        
        return overlaps
    
    def _extract_general_mulliken_section(self, content: str) -> Dict[str, Any]:
        """Extract general Mulliken population analysis for non-spin-polarized calculations.

        The LAST PPAN print is used (earlier ones belong to earlier SCF runs or
        geometries), and each atom keeps all its A.O. populations: CRYSTAL
        prints them 8 per line and continues on indented lines, which a
        stripped line-by-line parse dropped (as _extract_mulliken_section).
        """
        pattern = r'MULLIKEN POPULATION ANALYSIS - NO\. OF ELECTRONS\s+([\d.-]+)(.*?)(?=OVERLAP POPULATION|EIGENVECTORS|$)'
        matches = list(re.finditer(pattern, content, re.DOTALL))
        if not matches:
            return None
        match = matches[-1]

        total_electrons = float(match.group(1))
        section_content = match.group(2)

        # The A.O. POPULATION table only (the SHELL POPULATION table that
        # follows repeats the atom rows)
        atoms = []
        ao_match = re.search(r'ATOM\s+Z\s+CHARGE\s+A\.O\.\s+POPULATION(.*?)(?=ATOM\s+Z\s+CHARGE\s+SHELL|$)',
                             section_content, re.DOTALL)
        if ao_match:
            atoms = self._mulliken_ao_rows(ao_match.group(1))

        return {
            'total_electrons': total_electrons,
            'atoms': atoms
        }
    
    def _extract_general_overlap_populations(self, content: str) -> List[Dict]:
        """Extract general overlap populations for non-spin-polarized calculations."""
        overlaps = []
        
        # Look for the overlap population section
        pattern = r'OVERLAP POPULATION CONDENSED TO ATOMS.*?ATOM A\s+(\d+)\s+(\w+)\s+ATOM B.*?\n((?:.*?\([^)]*\).*?\n)+)'
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for match in matches:
            atom_a_num = int(match.group(1))
            atom_a_element = match.group(2)
            overlap_data = match.group(3)
            
            # Parse neighbor interactions
            neighbors = []
            overlap_lines = overlap_data.strip().split('\n')
            
            for line in overlap_lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Look for pattern: ATOM_NUM ELEMENT (CELL) DISTANCE_AU DISTANCE_ANG OVERLAP
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        # Extract cell indices from parentheses
                        cell_match = re.search(r'\(\s*([\d-]+)\s+([\d-]+)\s+([\d-]+)\s*\)', line)
                        if cell_match:
                            atom_b_num = int(parts[0])
                            atom_b_element = parts[1]
                            cell_i = int(cell_match.group(1))
                            cell_j = int(cell_match.group(2))
                            cell_k = int(cell_match.group(3))
                            
                            # Find distances and overlap in remaining parts
                            numeric_parts = [p for p in parts[2:] if p.replace('.', '').replace('-', '').isdigit()]
                            if len(numeric_parts) >= 3:
                                distance_au = float(numeric_parts[0])
                                distance_ang = float(numeric_parts[1])
                                overlap_pop = float(numeric_parts[2])
                                
                                neighbors.append({
                                    'atom_b_number': atom_b_num,
                                    'atom_b_element': atom_b_element,
                                    'cell_indices': [cell_i, cell_j, cell_k],
                                    'distance_au': distance_au,
                                    'distance_ang': distance_ang,
                                    'overlap_population': overlap_pop
                                })
                    except (ValueError, IndexError):
                        continue
            
            if neighbors:
                overlaps.append({
                    'atom_a_number': atom_a_num,
                    'atom_a_element': atom_a_element,
                    'neighbors': neighbors
                })
        
        return overlaps
    
    def _extract_material_id_from_filename(self, output_file: Path) -> str:
        """Extract material ID from filename."""
        return db_build.material_id_from_filename(output_file)
    
    def _material_of_calc(self, calc_id: str) -> Optional[str]:
        """The material_id of a tracked calculation, or None."""
        if not self.enable_tracking or self.db is None:
            return None
        try:
            calc = self.db.get_calculation(calc_id)
        except Exception:
            return None
        return (calc or {}).get('material_id') or None

    def _migrate_calc_rows(self, material_id: str, calc_id: str) -> int:
        """Rows this extractor filed for ``calc_id`` under another material id
        (the file-name id earlier versions used) move to ``material_id`` by
        UPDATE; nothing is deleted. A row whose property name the target
        already has for this calculation stays where it is (moving it would
        make a duplicate)."""
        try:
            with self.db._get_connection() as conn:
                rows = conn.execute(
                    "SELECT DISTINCT material_id FROM properties WHERE calc_id = ? AND material_id != ? "
                    "AND extractor_script = 'crystal_property_extractor.py'", (calc_id, material_id)).fetchall()
                moved = 0
                for table in _property_tables(conn):
                    # OR IGNORE: a name two earlier ids both hold moves once
                    # (the unique index of the 1.1.6 schema)
                    moved += conn.execute(
                        f"UPDATE OR IGNORE {table} SET material_id = ? WHERE calc_id = ? AND material_id != ? "
                        "AND extractor_script = 'crystal_property_extractor.py' AND property_name NOT IN "
                        f"(SELECT property_name FROM {table} WHERE material_id = ? AND calc_id = ?)",
                        (material_id, calc_id, material_id, material_id, calc_id)).rowcount
        except Exception as e:
            print(f"⚠️  Error moving earlier properties of {calc_id}: {e}")
            return 0
        if moved:
            old = ", ".join(r[0] for r in rows)
            print(f"   ↪ moved {moved} rows of {calc_id} from {old} to {material_id}")
        return moved

    def _fill_placeholder_material(self, material_id: str, properties: Dict[str, Any],
                                   output_file: Path, calc_id: Optional[str]) -> None:
        """A material created as a placeholder (formula 'Unknown', by
        populate_completed_jobs) gets its formula, dimensionality and space
        group from this run: formula_extractor.material_info, which reads the
        cell the output prints (the deck, expanded by its space group, only
        when the output has no geometry). The cell formula goes into
        metadata_json.

        A formula MACE itself took from the deck's atom lines (the asymmetric
        unit: TiPbO3 stored as TiPbO2) is corrected the same way, and so is
        the dimensionality CRYSTAL every scan used to write for a molecule,
        polymer or slab. A formula or space group set any other way is never
        overwritten, so a second run changes nothing."""
        try:
            with self.db._get_connection() as conn:
                row = conn.execute("SELECT formula, space_group, dimensionality, metadata_json "
                                   "FROM materials WHERE material_id = ?", (material_id,)).fetchone()
        except Exception:
            return
        if not row:
            return
        old_formula, old_sg, old_dim, old_meta = row[0], row[1], row[2], row[3]
        try:
            try:
                from mace.utils.formula_extractor import (extract_formula_from_d12,
                                                          extract_space_group_from_output, material_info)
            except ImportError:
                from formula_extractor import extract_formula_from_d12, extract_space_group_from_output, material_info
            d12 = None
            calc = self.db.get_calculation(calc_id) if calc_id else None
            for cand in ((calc or {}).get('input_file'), Path(output_file).with_suffix('.d12')):
                if cand and Path(cand).suffix == '.d12' and Path(cand).is_file():
                    d12 = Path(cand)
                    break
            info = material_info(output_file=output_file, d12_file=d12)
            formula = info.get('formula')
            if not formula and d12 is not None:
                formula = extract_formula_from_d12(d12)     # the deck's atom lines: the last resort
            sg = extract_space_group_from_output(output_file)
        except Exception as e:
            print(f"⚠️  Could not read the formula for {material_id}: {e}")
            return
        sg = sg or properties.get('space_group_number')
        fields, values = [], []
        placeholder = old_formula == 'Unknown'
        # the asymmetric-unit formula an earlier scan took from this deck
        from_deck = (not placeholder and d12 is not None and info.get('source') is not None
                     and old_formula == extract_formula_from_d12(d12))
        if formula and formula != old_formula and (placeholder or from_deck):
            fields.append('formula = ?')
            values.append(formula)
        if sg and old_sg != sg and (old_sg is None or (placeholder and old_sg == 1)):
            fields.append('space_group = ?')
            values.append(int(sg))
        dim = info.get('dimensionality')
        if dim and dim != old_dim and old_dim in (None, '', 'CRYSTAL'):
            fields.append('dimensionality = ?')
            values.append(dim)
        if info.get('cell_formula') and (placeholder or from_deck or formula == old_formula):
            try:
                meta = json.loads(old_meta) if old_meta else {}
            except (TypeError, ValueError):
                meta = None                       # not MACE's JSON: left as it is
            if isinstance(meta, dict) and (meta.get('cell_formula') != info['cell_formula']
                                           or meta.get('formula_source') != info['source']):
                meta['cell_formula'] = info['cell_formula']
                meta['formula_source'] = info['source']
                fields.append('metadata_json = ?')
                values.append(json.dumps(meta))
        if not fields:
            return
        try:
            with self.db._get_connection() as conn:
                conn.execute(f"UPDATE materials SET {', '.join(fields)}, updated_at = ? WHERE material_id = ?",
                             (*values, datetime.now().isoformat(), material_id))
            shown = [f"formula {formula}" if 'formula = ?' in fields else None,
                     f"space group {sg}" if 'space_group = ?' in fields else None,
                     f"dimensionality {dim}" if 'dimensionality = ?' in fields else None]
            if any(shown):
                print(f"   📝 {material_id}: {', '.join(x for x in shown if x)}")
        except Exception as e:
            print(f"⚠️  Error updating material {material_id}: {e}")

    def _find_calc_id_for_output(self, output_file: Path) -> Optional[str]:
        """Find the calculation ID associated with this output file."""
        if not self.enable_tracking or self.db is None:
            return None  # offline / untracked extraction: no DB to consult
        # The database holds the paths the submission recorded (resolved, as
        # a rule); a scan passes them as found ('LiEC.out' for a scan of '.').
        # Compare every spelling, or no row ever matched and each extraction
        # filed its properties a second time with calc_id NULL.
        # An in-place directory holds several calculations: the one whose
        # input deck has this output's name, and the directory alone only
        # when it holds just one (db_build.calc_for_output; the dry run of
        # --register/--skip-unchanged uses the same lookup read-only).
        try:
            # and a path the database records relative to its own folder (examples/gui/materials.db)
            base = Path(self.db.db_path).parent if getattr(self.db, "db_path", None) else None
            with self.db._get_connection() as conn:
                return db_build.calc_for_output(conn, Path(output_file), base)
        except Exception:
            return None
    
    def save_properties_to_database(self, properties: Dict[str, Any]) -> int:
        """Save extracted properties to the database."""
        if not self.enable_tracking or self.db is None:
            return 0  # tracking disabled: nothing to persist
        if not properties or '_metadata' not in properties:
            return 0
        # One connection and one commit for the whole output: every
        # _get_connection below is a savepoint of this transaction (a row
        # that fails is undone alone, as before), not a connect/commit/close
        # per row, which left the build waiting on fsync.
        transaction = getattr(self.db, 'transaction', None)
        if transaction is None:
            return self._save_properties(properties)
        try:
            with transaction():
                return self._save_properties(properties)
        except sqlite3.Error as e:
            # the database stayed locked or failed at BEGIN/COMMIT (row errors
            # are reported row by row inside); any other error propagates as
            # before, and in either case none of the output's rows are kept
            name = Path(str(properties['_metadata'].get('output_file') or '')).name
            print(f"⚠️  Error saving the properties of {name}: {e}; none of them were written")
            return 0

    def _save_properties(self, properties: Dict[str, Any]) -> int:
        """save_properties_to_database, inside its transaction."""
        metadata = properties['_metadata']
        material_id = metadata['material_id']
        calc_id = metadata['calc_id']
        try:
            # the derived properties' keys (their category, unit, table)
            from mace.properties import registry
            registry.load()
        except Exception:
            pass
        
        saved_count = 0

        # Re-extraction replaces this extractor's rows for the calculation:
        # rows an older extractor wrote that this one no longer produces
        # (e.g. 1.1.1's band fields on DOSS runs) would otherwise stay.
        # Untracked rows (calc_id NULL) are shared by every output of the
        # material, so they cannot be attributed and are left alone.
        if calc_id is not None:
            # rows earlier versions filed under a file-name id move to the
            # calculation's material (UPDATE, never DELETE)
            self._migrate_calc_rows(material_id, calc_id)
            produced = [name for name in properties if not name.startswith('_')]
            # the rows of a derived property that failed or was not computed
            # this time (--skip-property, --no-derived, a bulk build's later
            # pass) stay: an error never deletes a stored value
            produced += [n for n in properties.get('_keep_rows') or () if n not in produced]
            # a TRANSPORT run's total ZT came from a kappa_L the user gave an
            # earlier extraction (--kappa-lattice): one without it keeps those
            # rows while the run is unchanged (_settle_earlier_total)
            keep_total = self._settle_earlier_total(properties, material_id, calc_id)
            try:
                with self.db._get_connection() as conn:
                    for table in _property_tables(conn):
                        conn.execute(
                            f"DELETE FROM {table} WHERE material_id = ? AND calc_id = ? "
                            "AND extractor_script = 'crystal_property_extractor.py' "
                            f"AND property_name NOT IN ({','.join('?' * len(produced))})"
                            + (" AND property_name NOT GLOB 'transport_zt_total*'" if keep_total else ""),
                            (material_id, calc_id, *produced))
            except Exception as e:
                print(f"⚠️  Error removing stale properties for {calc_id}: {e}")
        else:
            # untracked rows are upserted, so an earlier total this extraction
            # does not write again is settled here
            self._settle_earlier_total(properties, material_id, calc_id)

        # Save each property
        for prop_name, prop_value in properties.items():
            if prop_name.startswith('_'):
                continue  # Skip metadata
            if self._store_row(material_id, calc_id, prop_name, prop_value, metadata['extracted_at']):
                saved_count += 1

        # the --reference choices of the derived properties that stored a
        # value (and those given as "auto"): the run's reference_runs row,
        # which a re-extraction keeps and uses (registry.CHOICES_KEY)
        if calc_id is not None and properties.get('_reference_choices'):
            self._save_reference_choices(material_id, calc_id, properties['_reference_choices'])
        elif properties.get('_reference_choices'):
            print(f"⚠️  {Path(metadata['output_file']).name} has no calculation record: its --reference choices "
                  "are used now but not saved for later extractions (register it: --register)")

        if calc_id is not None and saved_count:
            self._fill_placeholder_material(material_id, properties, Path(metadata['output_file']), calc_id)

        return saved_count

    def _store_row(self, material_id: str, calc_id, prop_name: str, prop_value, extracted_at: str) -> bool:
        """One row of this extractor (properties, or property_blobs for a
        large JSON value): replaced when the run has it. True when written."""
        # Determine property category
        category = self._categorize_property(prop_name)
        if not property_keys.known(prop_name) and prop_name not in self._unregistered:
            # every name written is in the registry (tests/test_property_keys.py)
            self._unregistered.add(prop_name)
            print(f"⚠️  {prop_name} is not in mace/utils/property_keys.py: stored under 'other'")
        
        # Handle complex values (convert to JSON)
        if isinstance(prop_value, (dict, list)):
            value_text = json.dumps(prop_value)
            value_numeric = None
        elif isinstance(prop_value, (int, float)):
            value_numeric = float(prop_value)
            value_text = str(prop_value)
        else:
            value_numeric = None
            value_text = str(prop_value)
        
        # Determine units
        unit = self._get_property_unit(prop_name)
        
        try:
            # Check if property already exists. NULL-safe calc_id match:
            # untracked extractions have calc_id=None, and `calc_id = ?`
            # never matches NULL, so every re-extraction inserted a full
            # duplicate set of property rows.
            with self.db._get_connection() as conn:
                key = (material_id, prop_name, calc_id, calc_id)
                where = ("material_id = ? AND property_name = ? "
                         "AND (calc_id = ? OR (calc_id IS NULL AND ? IS NULL))")
                if property_keys.is_blob(prop_name) and 'property_blobs' in _property_tables(conn):
                    # a large JSON value: the property_blobs table (a
                    # copy an earlier version left in properties goes)
                    conn.execute(f"DELETE FROM properties WHERE {where}", key)
                    existing = conn.execute(f"SELECT blob_id FROM property_blobs WHERE {where}", key).fetchone()
                    if existing:
                        conn.execute("""
                            UPDATE property_blobs
                            SET property_value_text = ?, property_unit = ?, property_category = ?,
                                extracted_at = ?, extractor_script = ?
                            WHERE blob_id = ?
                        """, (value_text, unit, category, extracted_at,
                              'crystal_property_extractor.py', existing[0]))
                    else:
                        conn.execute("""
                            INSERT INTO property_blobs
                            (material_id, calc_id, property_category, property_name,
                             property_value_text, property_unit, extracted_at, extractor_script)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (material_id, calc_id, category, prop_name, value_text, unit,
                              extracted_at, 'crystal_property_extractor.py'))
                    return True
                cursor = conn.execute(f"SELECT property_id FROM properties WHERE {where}", key)
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing property (unit and category too:
                    # a database built by an earlier version gets this
                    # version's labels)
                    conn.execute("""
                        UPDATE properties 
                        SET property_value = ?, property_value_text = ?, property_unit = ?,
                            property_category = ?, extracted_at = ?, extractor_script = ?
                        WHERE property_id = ?
                    """, (value_numeric, value_text, unit, category,
                          extracted_at, 'crystal_property_extractor.py',
                          existing[0]))
                else:
                    # Insert new property
                    conn.execute("""
                        INSERT INTO properties 
                        (material_id, calc_id, property_category, property_name, 
                         property_value, property_value_text, property_unit, 
                         extracted_at, extractor_script)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (material_id, calc_id, category, prop_name,
                          value_numeric, value_text, unit,
                          extracted_at, 'crystal_property_extractor.py'))
                
                return True
                
        except Exception as e:
            print(f"⚠️  Error saving property {prop_name}: {e}")
            return False

    def _save_reference_choices(self, material_id: str, calc_id: str, choices: Dict[str, Any]) -> None:
        try:
            from mace.properties import reference, registry
            used = choices.get('used') or {}
            with self.db._get_connection() as conn:
                current = registry.saved_choices(conn, calc_id)
                # a file the database tracks is saved as its calc_id (it survives a move)
                for roles in used.values():
                    for recs in roles.values():
                        for r in recs:
                            if r.get('file'):
                                cid = reference.tracked_calc(conn, Path(r['file']))
                                if cid:
                                    r['calc_id'] = cid
                                    r.pop('file')
            merged = registry.merge_choices(current, used, choices.get('forget') or {})
            if merged == current:
                return
            if merged:
                prop = registry.get(registry.CHOICES_KEY)
                self.db.store_material_property(material_id, registry.CHOICES_KEY, merged,
                                                prop.category if prop else 'computational', calc_id=calc_id,
                                                property_unit='JSON', extractor_script=registry.REFERENCE_SCRIPT)
            else:
                with self.db._get_connection() as conn:
                    conn.execute("DELETE FROM properties WHERE calc_id = ? AND property_name = ?",
                                 (calc_id, registry.CHOICES_KEY))
        except Exception as e:
            print(f"⚠️  Error saving the reference choices of {calc_id}: {e}")

    def _settle_earlier_total(self, properties: Dict[str, Any], material_id: str, calc_id) -> bool:
        """Whether a TRANSPORT run's earlier ``transport_zt_total*`` rows stay.

        This extraction wrote a total: the earlier rows are replaced (False,
        the tracked DELETE removes any it does not write again). It was given
        a kappa_L and refused it (a table that misses a run temperature, no
        RELAXTIM): the earlier rows are removed - "no total ZT stored" is
        then true. It had no kappa_L: the rows stay only while the run's .out
        and .dat files are the ones they were computed from (their
        fingerprint, ``transport_zt_total_inputs``); otherwise they are
        removed, and the user told. Not a TRANSPORT run: nothing to settle."""
        state = properties.get('_transport_zt')
        if state is None:
            return False
        if any(n.startswith('transport_zt_total') for n in properties):
            return False
        where = ("material_id = ? AND (calc_id = ? OR (calc_id IS NULL AND ? IS NULL)) "
                 "AND extractor_script = 'crystal_property_extractor.py' "
                 "AND property_name GLOB 'transport_zt_total*'")
        args = (material_id, calc_id, calc_id)
        try:
            with self.db._get_connection() as conn:
                old = {n: (v, t) for n, v, t in conn.execute(
                    f"SELECT property_name, property_value, property_value_text FROM properties WHERE {where}",
                    args)}
                if not old:
                    return True
                inputs = (old.get('transport_zt_total_inputs') or (None, None))[1]
                if not state.get('refused') and state.get('inputs') and inputs == state['inputs']:
                    return True
                conn.execute(f"DELETE FROM properties WHERE {where}", args)
        except Exception as e:
            print(f"⚠️  Error settling the earlier total ZT rows of {material_id}: {e}")
            return True
        name = Path((properties.get('_metadata') or {}).get('output_file') or material_id).name
        value = (old.get('transport_zt_total_max_avg') or old.get('transport_zt_total_max') or (None, None))[0]
        source = (old.get('transport_zt_total_lattice_source') or (None, None))[1]
        earlier = (f"the earlier total ZT ({value:.4g}, κL {source})" if value is not None and source
                   else "the earlier total ZT")
        if state.get('refused'):
            print(f"❌ {name}: {earlier} is removed from the database, as this extraction stores none")
        else:
            print(f"⚠️  {name}: {earlier} is removed from the database: the run's output has changed since "
                  "it was computed (or MACE cannot tell); extract again with --kappa-lattice for a new one")
        return False

    def remove_stale_untracked_rows(self, material_id: str, before: str) -> int:
        """Delete this extractor's untracked (calc_id NULL) rows of a material
        written before ``before`` (an ISO timestamp): rows a re-extraction of
        all of the material's outputs did not write again. The caller decides
        that every output of the material was re-extracted."""
        if not self.enable_tracking or self.db is None:
            return 0
        try:
            with self.db._get_connection() as conn:
                # the total ZT rows of a TRANSPORT run extracted again in this
                # run without --kappa-lattice stay (see save_properties_to_database)
                removed = conn.execute(
                    "DELETE FROM properties WHERE material_id = ? AND calc_id IS NULL "
                    "AND extractor_script = 'crystal_property_extractor.py' AND extracted_at < ? "
                    "AND NOT (property_name GLOB 'transport_zt_total*' AND EXISTS ("
                    "SELECT 1 FROM properties p2 WHERE p2.material_id = ? AND p2.calc_id IS NULL "
                    "AND p2.property_name = 'transport_data_available' AND p2.extracted_at >= ?))",
                    (material_id, before, material_id, before)).rowcount
                if 'property_blobs' in _property_tables(conn):
                    removed += conn.execute(
                        "DELETE FROM property_blobs WHERE material_id = ? AND calc_id IS NULL "
                        "AND extractor_script = 'crystal_property_extractor.py' AND extracted_at < ?",
                        (material_id, before)).rowcount
                return removed
        except Exception as e:
            print(f"⚠️  Error removing stale properties for {material_id}: {e}")
            return 0

    def _extract_computational_properties(self, content: str) -> Dict[str, Any]:
        """Extract computational performance and timing properties."""
        props = {}
        
        # CPU time of all processes (TOTAL CPU TIME), and the wall time:
        # the last TELAPSE CRYSTAL prints (its END line in a finished run)
        cpu_time_match = ci_search(r'TOTAL CPU TIME\s*[=:]\s*([\d.]+)', content, ('total cpu time',))
        if cpu_time_match:
            props['cpu_time_total_s'] = float(cpu_time_match.group(1))
        telapse = content.rfind('TELAPSE')
        if telapse >= 0:
            wall = re.match(r'TELAPSE\s+([\d.]+)', content[telapse:])
            if wall:
                props['wall_time_s'] = float(wall.group(1))
        
        # Fermi energy extraction - including SP calculation conducting state format
        fermi_patterns = [
            (r'FERMI ENERGY\s*[=:]\s*([-\d.]+(?:[Ee][+-]?\d+)?)', ('fermi energy',)),
            (r'FERMI LEVEL\s*[=:]\s*([-\d.]+(?:[Ee][+-]?\d+)?)', ('fermi level',)),
            (r'CHEMICAL POTENTIAL\s*[=:]\s*([-\d.]+(?:[Ee][+-]?\d+)?)', ('chemical potential',)),
            (r'EFERMI\(AU\)\s+([-\d.E+\-]+)', ('efermi(au)',)),  # SP calculation conducting state: "EFERMI(AU) -9.9423732E-02"
            (r'POSSIBLY CONDUCTING STATE.*?EFERMI\(AU\)\s+([-\d.E+\-]+)',  # Full conducting state pattern
             ('possibly conducting state', 'efermi(au)')),
        ]
        
        # A multi-point OPT prints EFERMI lines at point 1 only; its final
        # Fermi energy is the FERMI ENERGY line after OPT END, if any
        final_part = self._after_last_opt_point(content)
        for pattern, literals in fermi_patterns:
            # Last occurrence: conducting-state EFERMI lines are printed per
            # SCF iteration and only the final one is converged
            fermi_matches = ci_findall(pattern, final_part, literals)
            if fermi_matches:
                props['fermi_energy_au'] = float(fermi_matches[-1])
                break
        
        # SCF cycles of the last SCF: CRYSTAL23 prints '== SCF ENDED ...
        # CYCLES n' after an SP's SCF (and an OPT's first point), and the
        # cycles in 'TOTAL ENERGY(DFT)(AU)( n)' for every SCF, the only
        # record of an OPT's later points ('***' above 999: none)
        cycle_lines = list(re.finditer(
            r'== SCF ENDED[^\n]*?CYCLES\s+(\d+)|TOTAL ENERGY\((?:DFT|HF)\)\(AU\)\(\s*(\d+)\)', content))
        if cycle_lines:
            last = cycle_lines[-1]
            props['scf_cycles'] = int(last.group(1) or last.group(2))
        else:
            scf_match = ci_search(r'SCF FIELD CONVERGENCE IN\s*(\d+)\s*CYCLES', content, ('scf field convergence in', 'cycles'))
            if scf_match:
                props['scf_cycles'] = int(scf_match.group(1))
        
        return props
    
    def _parse_band_dat_kpath_nodes(self, band_dat_file: Path):
        """Parse the high-symmetry node positions from a CRYSTAL BAND.DAT header.

        The ``# <kindex> <label>`` lines in the ``# NPANEL`` block give the
        cumulative (1-based) k-point index of each high-symmetry node along the
        path, plus a (usually placeholder, e.g. "A") label. ``DatFileProcessor``
        keeps only the labels, so re-scan the header here to also recover the
        node *indices* (needed to read band energies AT the high-symmetry
        points) and the explicit ``# EFERMI (HARTREE)`` value.

        Returns ``(node_indices, node_labels, efermi_hartree)`` using native
        Python ints/floats/strs. Scans only ``#``/``@`` header lines and stops
        at the first numeric data row, so it is cheap even for large files.
        """
        node_indices = []
        node_labels = []
        efermi_ha = None
        try:
            import io
            from mace.utils.dat_text import read_text
            with io.StringIO(read_text(band_dat_file)) as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    if line.startswith('@'):
                        continue  # xmgrace directive
                    if line.startswith('#'):
                        upper = line.upper()
                        if 'EFERMI' in upper and efermi_ha is None:
                            m = re.search(r'(-?\d+\.\d+(?:[Ee][+-]?\d+)?)', line)
                            if m:
                                efermi_ha = float(m.group(1))
                            continue
                        tok = line.lstrip('#').split()
                        # Panel node line: "<int> <label>" (skip NKPT/NPANEL header
                        # lines, which have a keyword in tok[0]).
                        if (len(tok) >= 2 and tok[0].isdigit()
                                and not tok[1].isdigit()
                                and 'NKPT' not in tok and 'NPANEL' not in tok):
                            node_indices.append(int(tok[0]))
                            node_labels.append(tok[1])
                        continue
                    # Numeric data row. The node block sits in the header, but the
                    # "# EFERMI (HARTREE)" comment is written between the ALPHA and
                    # BETA data blocks (not in the header). Once we have the node
                    # indices, keep skipping data rows cheaply until we hit that
                    # comment (or EOF) so EFERMI is captured without re-reading the
                    # full BETA block.
                    if node_indices and efermi_ha is not None:
                        break
                    continue
        except Exception:
            return [], [], None
        return node_indices, node_labels, efermi_ha

    def _kpath_labels_from_d3(self, output_file: Path, num_nodes: int):
        """Resolve the REAL high-symmetry labels (G/X/L/...) from the band ``.d3``.

        CRYSTAL writes placeholder labels ("A") into BAND.DAT, but the input deck
        (``<stem>.d3``) carries the SeeKPath path string on its title line, e.g.
        ``... SeeKPath (w.I) - GAMMA-X-U-|-K-GAMMA-L-W-X``. A ``|`` marks a path
        discontinuity (the two endpoints it joins coincide as ONE node), so the
        node count after collapsing ``|`` matches the BAND.DAT panel-node count.
        Returns a list of ``num_nodes`` label strings, or ``[]`` if unavailable.
        """
        d3_file = output_file.parent / f"{output_file.stem}.d3"
        if not d3_file.exists():
            return []
        try:
            lines = d3_file.read_text(errors='ignore').splitlines()
        except Exception:
            return []
        title = ''
        for ln in lines[:5]:
            if 'SeeKPath' in ln or 'SEEKPATH' in ln.upper() or 'Band Structure' in ln:
                title = ln
                break
        if not title:
            return []
        m = re.search(r'SeeKPath[^-]*-\s*(.+)$', title)
        path_str = (m.group(1) if m else '').strip()
        if not path_str:
            return []
        raw = path_str.split('-')
        labels = []
        i = 0
        while i < len(raw):
            lab = raw[i].strip()
            if lab == '|':
                # Discontinuity: merge the next endpoint into the previous node.
                if labels and i + 1 < len(raw):
                    labels[-1] = labels[-1] + '|' + raw[i + 1].strip()
                    i += 2
                    continue
                i += 1
                continue
            if lab:
                labels.append(lab)
            i += 1
        if num_nodes and len(labels) == num_nodes:
            return labels
        return labels  # best-effort even on count mismatch (caller may zip-truncate)

    def _build_band_structure_summary(self, dat_info, content, output_file,
                                      band_dat_file, num_occ, band_start=1):
        """Build a compact, JSON-serializable band-structure summary.

        Per the storage analysis this stores the scientifically useful structure
        WITHOUT dumping the full ~1000k-point x ~100-band eigenvalue matrix into
        the properties table: the k-path with real high-symmetry labels, the band
        energies AT the high-symmetry points, direct vs indirect gap with their
        k-locations, band/k-point counts, and the Fermi energy. The raw BAND.DAT
        stays on disk and is re-parsed on demand. All values are native
        float/int/str/bool so they JSON-serialize cleanly.
        """
        eigenvalues = dat_info.get('eigenvalues') or []
        if not eigenvalues:
            return None, {}
        nk = int(dat_info.get('num_k_points') or 0)
        nb = int(dat_info.get('num_bands') or (len(eigenvalues[0]) if eigenvalues else 0))
        ns = int(dat_info.get('num_spins') or 1)
        k_abscissa = dat_info.get('k_points') or []

        node_indices, _placeholder_labels, efermi_ha = self._parse_band_dat_kpath_nodes(band_dat_file)
        if efermi_ha is None:
            # Fall back to the Fermi energy parsed from the .out (CRYSTAL writes
            # the same value, in Hartree, on the band .out's FERMI ENERGY line).
            m = re.search(r'FERMI ENERGY\s+(-?\d+\.\d+(?:[Ee][+-]?\d+)?)', content)
            if m:
                try:
                    efermi_ha = float(m.group(1))
                except ValueError:
                    efermi_ha = None
        real_labels = self._kpath_labels_from_d3(output_file, len(node_indices))
        # Prefer real .d3 labels; fall back to BAND.DAT placeholders.
        if real_labels and len(real_labels) == len(node_indices):
            labels = real_labels
        elif dat_info.get('k_path_labels'):
            labels = list(dat_info['k_path_labels'])
        else:
            labels = list(_placeholder_labels)

        # Restrict to the ALPHA (first-spin) k-block for the per-k geometry; the
        # node indices are 1-based cumulative positions within one spin block.
        alpha = eigenvalues[:nk] if nk else eigenvalues

        summary = {
            'source_file': band_dat_file.name,
            'num_kpoints': nk,
            'num_bands': nb,
            'num_spins': ns,
            'num_panels': int(dat_info.get('num_panels') or 0),
            'fermi_energy_hartree': float(efermi_ha) if efermi_ha is not None else None,
            'energy_units': 'eV',
            'energy_reference': 'fermi',  # BAND.DAT eigenvalues are E - E_Fermi
            'kpath_node_indices': [int(i) for i in node_indices],
            'kpath_node_distances': [
                float(k_abscissa[i - 1]) for i in node_indices
                if 0 < i <= len(k_abscissa)
            ],
            'kpath_labels': [str(l) for l in labels],
        }

        H = HARTREE_TO_EV
        # Band energies (Fermi-referenced eV) at each high-symmetry node. Only the
        # node rows are stored (a few KB), not the full matrix.
        hs_bands = []
        for ni in node_indices:
            if 0 < ni <= len(alpha):
                hs_bands.append([float(e) * H for e in alpha[ni - 1]])
            else:
                hs_bands.append([])
        summary['high_symmetry_band_energies_ev'] = hs_bands

        # Direct vs indirect gap from the occupied-band index (referencing-
        # independent, same convention as DatFileProcessor._analyze_band_structure).
        #
        # Two correctness invariants, matching the legacy scalar analyzer:
        #   (1) Indices are over the UNFILTERED rows: ragged rows are skipped
        #       in place but never shift the surviving rows' indices, so the
        #       reported k-point index stays a valid index into the per-k
        #       geometry (k_abscissa / node_indices), which is defined over one
        #       spin block of `nk` k-points (kidx = global_row % nk).
        #   (2) The fundamental gap considers ALL spin channels (VBM = max
        #       occupied over every k AND every spin; CBM = min virtual over
        #       every k AND every spin), so it agrees with the legacy
        #       BAND.DAT gap for spin-polarized systems too — not just
        #       the first spin block.
        extra_scalars = {}
        # BAND.DAT column n holds band band_start + n - 1: the HOMO column is
        # the absolute occupied-band count less the bands not written
        shift = max(int(band_start or 1), 1) - 1
        if num_occ and num_occ - shift >= 1 and eigenvalues:
            n = num_occ - shift
            kdiv = nk if nk else len(eigenvalues)

            def _label_for(kidx):
                # Map a 0-based per-spin-block k index to the high-symmetry
                # label if it sits exactly on a node, else None.
                for lbl, nidx in zip(labels, node_indices):
                    if nidx - 1 == kidx:
                        return str(lbl)
                return None

            vbm_e = None
            vbm_global = None   # global row index of the VBM (max occupied)
            cbm_e = None
            cbm_global = None   # global row index of the CBM (min virtual)
            direct = None       # smallest per-(k,spin) HOMO-LUMO gap (Hartree)
            for gi, row in enumerate(eigenvalues):
                # Skip ragged rows in place but KEEP the true global index.
                if len(row) < n + 1:
                    continue
                homo_e = row[n - 1]
                lumo_e = row[n]
                if vbm_e is None or homo_e > vbm_e:
                    vbm_e = homo_e
                    vbm_global = gi
                if cbm_e is None or lumo_e < cbm_e:
                    cbm_e = lumo_e
                    cbm_global = gi
                d = lumo_e - homo_e
                if direct is None or d < direct:
                    direct = d

            if vbm_e is not None and cbm_e is not None:
                # Per-spin-block k indices (valid into k_abscissa/node_indices).
                vbm_k = (vbm_global % kdiv) if kdiv else vbm_global
                cbm_k = (cbm_global % kdiv) if kdiv else cbm_global

                indirect = (cbm_e - vbm_e) * H
                if indirect < 0:
                    indirect = 0.0  # band overlap along the path -> metallic
                direct_ev = direct * H
                if direct_ev < 0:
                    direct_ev = 0.0
                is_direct = bool(vbm_k == cbm_k)

                summary['direct_gap_ev'] = float(direct_ev)
                summary['indirect_gap_ev'] = float(indirect)
                summary['fundamental_gap_ev'] = float(indirect)  # min over all k+spin
                summary['is_direct_gap'] = is_direct
                summary['vbm'] = {
                    'energy_ev': float(vbm_e * H),
                    'kpoint_index': int(vbm_k),
                    'kpath_distance': (float(k_abscissa[vbm_k]) if vbm_k < len(k_abscissa) else None),
                    'band_index': int(n + shift),
                    'kpoint_label': _label_for(vbm_k),
                }
                summary['cbm'] = {
                    'energy_ev': float(cbm_e * H),
                    'kpoint_index': int(cbm_k),
                    'kpath_distance': (float(k_abscissa[cbm_k]) if cbm_k < len(k_abscissa) else None),
                    'band_index': int(n + shift + 1),
                    'kpoint_label': _label_for(cbm_k),
                }
                summary['occupied_bands'] = int(n + shift)

                # Queryable scalar rows (numeric/bool) — distinct from the JSON blob.
                extra_scalars['band_direct_gap_ev'] = float(direct_ev)
                extra_scalars['band_fundamental_gap_ev'] = float(indirect)
                extra_scalars['band_gap_is_direct'] = is_direct
                extra_scalars['band_vbm_kpoint_index'] = int(vbm_k)
                extra_scalars['band_cbm_kpoint_index'] = int(cbm_k)

        if labels:
            extra_scalars['band_kpath_label_string'] = '-'.join(str(l) for l in labels)

        return summary, extra_scalars

    def _extract_band_structure_properties(self, content: str, output_file: Path) -> Dict[str, Any]:
        """Extract band structure specific properties from BAND calculations."""
        props = {}
        
        # Check if this is a BAND calculation. Not "FROM BAND": a DOSS run
        # prints "FROM BAND n TO BAND m" for its projected band range, which
        # tagged every DOSS output has_band_structure with the band index as
        # vbm_energy.
        if not is_band_output(content):
            return props
            
        # Extract k-point information
        kpoint_match = ci_search(r'TOTAL OF\s*(\d+)\s*K-POINTS ALONG THE PATH', content, ('total of', 'k-points along the path'))
        if kpoint_match:
            props['total_kpoints'] = int(kpoint_match.group(1))
        
        # Extract band range information
        band_range_match = ci_search(r'FROM BAND\s*(\d+)\s*TO BAND\s*(\d+)', content, ('from band', 'to band'))
        band_start = 1
        if band_range_match:
            band_start = int(band_range_match.group(1))
            if band_start != 1:  # stored only when it is not the first band
                props['band_start'] = band_start
            props['band_end'] = int(band_range_match.group(2))
            props['total_bands'] = int(band_range_match.group(2)) - int(band_range_match.group(1)) + 1
            
        # Extract Fermi energy from BAND calculation
        fermi_patterns = [
            (r'FERMI ENERGY\s*[=:]\s*([-\d.]+(?:[Ee][+-]?\d+)?)', ('fermi energy',)),
            (r'FERMI LEVEL\s*[=:]\s*([-\d.]+(?:[Ee][+-]?\d+)?)', ('fermi level',)),
            (r'EF\s*[=:]\s*([-\d.]+(?:[Ee][+-]?\d+)?)', ('ef',)),
            (r'FERMI ENERGY\s+([-\d.E+\-]+)', ('fermi energy',)),  # CRYSTAL format: "FERMI ENERGY -0.123E+00"
        ]
        
        for pattern, literals in fermi_patterns:
            fermi_matches = ci_findall(pattern, content, literals)
            if fermi_matches:
                # the run's fermi_energy_au when its SCF prints none
                props['_fermi_band'] = float(fermi_matches[-1])
                break
                
        # Check for BAND.DAT file and extract information. CRYSTAL/the workflow
        # writes "<stem>.BAND.DAT" (e.g. "..._band.BAND.DAT"), not a bare
        # "BAND.DAT", so the fixed-name lookup never matched real files and the
        # band analysis stayed inert. Mirror the DOSS alt-name fallback.
        band_dat_default = output_file.parent / "BAND.DAT"
        band_dat_alt = output_file.parent / f"{output_file.stem}.BAND.DAT"
        band_dat_file = None
        if band_dat_default.exists():
            band_dat_file = band_dat_default
        elif band_dat_alt.exists():
            band_dat_file = band_dat_alt

        if band_dat_file:
            props['band_dat_exists'] = True
            props['band_dat_size'] = band_dat_file.stat().st_size
            
            # Try to extract information from DAT file using existing processor
            try:
                try:
                    from mace.utils.dat_file_processor import DatFileProcessor
                except ImportError:
                    from dat_file_processor import DatFileProcessor
                processor = DatFileProcessor()
                # CRYSTAL BAND.DAT eigenvalues are Fermi-referenced, so the gap is
                # computed by band index. Determine the occupied-band count (HOMO
                # index): prefer the explicit "TOP OF VALENCE BANDS - BAND N"
                # line, else fall back to (electrons per cell)//2 — exact for
                # closed-shell / non-magnetic systems. The count is the Mulliken
                # one (band_occupied_count): the header is the count before
                # DOPING. Without it the BAND-derived gap is left unset rather
                # than wrong.
                num_occ = None
                nocc_match = re.findall(r'TOP OF VALENCE BANDS\s*-\s*BAND\s*(\d+)', content)
                if nocc_match:
                    num_occ = int(nocc_match[-1])
                else:
                    num_occ = band_occupied_count(content, output_file)
                # the BAND.DAT's columns start at band_start: its HOMO column
                # is num_occ less the bands below it (none left: unknown)
                num_occ_dat = (num_occ - (band_start - 1)
                               if num_occ and num_occ - (band_start - 1) >= 1 else None)
                dat_info = processor.process_band_dat_file(
                    band_dat_file, fermi_energy=props.get('_fermi_band'),
                    num_occupied_bands=num_occ_dat)
                if dat_info and 'error' not in dat_info:
                    # Merge only compact scalars — never the raw eigenvalue/k-point
                    # arrays, which would be JSON-dumped into the properties table.
                    props['band_dat_num_kpoints'] = dat_info.get('num_k_points')
                    props['band_dat_num_bands'] = dat_info.get('num_bands')
                    props['band_dat_num_spins'] = dat_info.get('num_spins')
                    props['band_dat_num_panels'] = dat_info.get('num_panels')
                    if dat_info.get('k_path_labels'):
                        props['band_dat_kpath_labels'] = dat_info['k_path_labels']
                    bep = dat_info.get('electronic_properties', {})
                    if bep.get('metallic') is not None:
                        props['band_dat_metallic'] = bep['metallic']
                    if bep.get('valence_band_maximum_ev') is not None:
                        props['band_dat_vbm_ev'] = bep['valence_band_maximum_ev']
                    if bep.get('conduction_band_minimum_ev') is not None:
                        props['band_dat_cbm_ev'] = bep['conduction_band_minimum_ev']

                    # LAYERED ADDITION: store the actual band STRUCTURE (k-path +
                    # high-symmetry-point band energies + direct/indirect gap with
                    # k-locations) as a compact JSON summary, plus queryable scalar
                    # rows. Per the storage analysis we do NOT dump the full
                    # per-k eigenvalue matrix (MBs); the raw BAND.DAT stays on disk
                    # and is re-parsed on demand. Failures here never affect the
                    # scalars above.
                    try:
                        summary, extra_scalars = self._build_band_structure_summary(
                            dat_info, content, output_file, band_dat_file, num_occ, band_start)
                        if summary:
                            props['band_structure'] = summary
                        for k, v in (extra_scalars or {}).items():
                            if v is not None:
                                props[k] = v
                    except Exception as e:
                        print(f"Warning: Could not build band-structure summary: {e}")
            except ImportError:
                pass  # DAT file processor not available
            except Exception as e:
                print(f"Warning: Could not process BAND.DAT file: {e}")
        
        # Mark this as a band structure calculation
        if props:
            props['calculation_type'] = 'BAND'
            props['has_band_structure'] = True
            
        return props
    
    def _build_dos_summary(self, dat_info, content, output_file, doss_dat_file):
        """Build a compact, JSON-serializable density-of-states summary.

        Mirrors ``_build_band_structure_summary``: this stores the scientifically
        useful DOS structure WITHOUT dumping the full ~1000-pt x N-projection x
        2-spin matrix (which is 96 KB - 2 MB of JSON for real slabs) into the
        properties table. The raw ``DOSS.DAT`` stays on disk (referenced via
        ``source_file`` + the already-stored ``doss_dat_*`` scalars) and is
        re-parsed on demand. We persist:

          * the DOS @ Fermi PER SPIN channel (alpha + beta, not just the first),
          * the gap-from-DOS / band-edge energies (window straddling E_F),
          * the integrated number of states up to E_Fermi (occupied-state count),
          * a DOWNSAMPLED total-DOS curve per spin (<= 200 pts/spin, ~few KB) in
            eV so the shape can be replotted without re-reading the big file,
          * the projection count + per-projection integrated weight (a single
            scalar per projection — the atom/orbital decomposition magnitude —
            not the full per-energy projected curves).

        UNITS. CRYSTAL DOSS.DAT energies are Fermi-referenced Hartree (E -
        E_Fermi, Fermi at 0) and the DOS magnitudes are in states/HARTREE/cell.
        We convert the energy axis to eV (``* H``) AND the DOS magnitudes to
        per-eV (``/ H``) so the stored curve, the DOS@Fermi values and the
        projection weights are all self-consistent with the ``states/eV/cell``
        label. ``dos_at_fermi_total`` (the database row) is the TOTAL DOS(E_F):
        the sum of the spin channels of a spin-polarized DOSS.DAT (each block
        holds one spin) and the single block of a closed-shell one;
        ``dos_at_fermi_<spin>`` keep the channels. (Before 1.1.6 the same
        value was also stored per Hartree, as doss_dat_at_fermi and
        dos_at_fermi_level.)

        For spin-polarized runs the energy grid is written once per spin block
        (concatenated in ``energy_points``); the beta block's DOS values are
        sign-negated for mirror plotting, so we take magnitudes for the physical
        DOS. Returns ``(summary, extra_scalars)`` and ``(None, {})`` when there
        is no curve to summarize. All values are native float/int/str/bool/list
        so they JSON-serialize cleanly.
        """
        energy_points = dat_info.get('energy_points') or []
        total_dos = dat_info.get('total_dos') or []
        if not energy_points or not total_dos:
            return None, {}
        # Guard ragged parser output (energy/DOS must stay aligned for slicing).
        if len(energy_points) != len(total_dos):
            return None, {}

        H = HARTREE_TO_EV
        nspin = int(dat_info.get('num_spins') or 1)
        if nspin < 1:
            nspin = 1
        nproj = int(dat_info.get('num_proj') or 0)
        # NEPTS = points PER spin block. Prefer the header value but never let it
        # run past the actual array; the grid is repeated once per spin block.
        nepts_hdr = int(dat_info.get('num_energy_points') or 0)
        total_len = len(energy_points)
        if nepts_hdr > 0 and nepts_hdr * nspin == total_len:
            nepts = nepts_hdr
        else:
            # Fall back to an even split; if it doesn't divide cleanly, treat the
            # whole thing as one block (never desync the per-block slices).
            if nspin > 1 and total_len % nspin == 0:
                nepts = total_len // nspin
            else:
                nepts = total_len
                nspin = 1

        da = dat_info.get('dos_analysis') or {}

        # Genuine projection count: the CRYSTAL header NPROJ counts the TOTAL DOS
        # column as a "projection", and the parser also exposes a projected_dos
        # column that is byte-identical to the total DOS. Neither is a real
        # atom/orbital projection, so we count only the projected_dos columns
        # that are NOT identical to the total DOS (computed below alongside the
        # per-projection weights). ``num_proj`` is then == len(weights).
        proj_all = dat_info.get('projected_dos') or {}
        total_dos_full = total_dos
        genuine_proj = {
            name: vals for name, vals in proj_all.items()
            if not (isinstance(vals, list) and vals == total_dos_full)
        }
        n_genuine_proj = len(genuine_proj)

        summary = {
            'source_file': Path(doss_dat_file).name,
            'num_energy_points': int(nepts),
            'num_proj': int(n_genuine_proj),
            'num_spins': int(nspin),
            'energy_units': 'eV',
            'energy_reference': 'fermi',  # DOSS.DAT energies are E - E_Fermi
            # DOS magnitudes are converted from states/HARTREE/cell to per-eV
            # (divided by HARTREE_TO_EV) so they match the eV energy axis below.
            'dos_units': 'states/eV/cell',
            'spin_polarized': bool(nspin >= 2),
        }

        erange = dat_info.get('energy_range')
        if erange and len(erange) == 2:
            summary['energy_min_ev'] = float(erange[0]) * H
            summary['energy_max_ev'] = float(erange[1]) * H

        # Absolute Fermi energy (Hartree): footer "# EFERMI (HARTREE) <val>",
        # printed once per spin block. Parse from the raw .DAT (it is NOT on the
        # energy axis, which is already Fermi-shifted). Best-effort only.
        efermi_ha = None
        try:
            from mace.utils.dat_text import read_text
            dat_text = read_text(doss_dat_file)
            ef_matches = re.findall(
                r'EFERMI\s*\(HARTREE\)\s*(-?\d+\.?\d*(?:[Ee][+-]?\d+)?)', dat_text)
            if ef_matches:
                efermi_ha = float(ef_matches[-1])
        except Exception:
            efermi_ha = None
        summary['fermi_energy_hartree'] = (
            float(efermi_ha) if efermi_ha is not None else None)

        # --- Per-spin DOS @ Fermi (value at first grid point AT or ABOVE E=0) ---
        # Same convention as DatFileProcessor._analyze_density_of_states, but per
        # spin block so the alpha and beta Fermi-level DOS are both reported
        # (never silently collapsing to the first channel).
        def _dos_at_fermi(es, ds):
            at_or_above = [(e, d) for e, d in zip(es, ds) if e >= 0]
            if at_or_above:
                return abs(min(at_or_above, key=lambda ed: ed[0])[1])
            if not es:
                return None
            ci = min(range(len(es)), key=lambda i: abs(es[i]))
            return abs(ds[ci])

        spin_labels = ['alpha', 'beta'] if nspin >= 2 else ['total']
        dos_at_fermi_by_spin = {}      # states/eV/cell, per spin channel
        dos_at_fermi_per_channel = None  # states/eV/cell, single channel (legacy)
        downsampled = {}
        # Energy grid (eV) for the first block; identical across blocks.
        block0_es_ha = energy_points[0:nepts]
        for s in range(nspin):
            lo = s * nepts
            hi = lo + nepts
            es_ha = energy_points[lo:hi]
            # DOS magnitudes converted states/Ha -> states/eV (divide by H) so the
            # stored values match the 'states/eV/cell' label and the eV axis.
            ds_block = [abs(d) / H for d in total_dos[lo:hi]]
            if not es_ha:
                continue
            label = spin_labels[s] if s < len(spin_labels) else f'spin_{s}'
            daf = _dos_at_fermi(es_ha, ds_block)  # already per-eV
            daf_val = float(daf) if daf is not None else None
            dos_at_fermi_by_spin[label] = daf_val
            # The first block's channel, kept in the JSON summary as
            # dos_at_fermi_per_channel_ev beside the both-spins sum below.
            if s == 0:
                dos_at_fermi_per_channel = daf_val
            # Downsample the (E in eV, DOS in states/eV) curve to <= 200 points.
            # Stride = ceil(n/200) guarantees len(range(0,n,stride)) <= 200
            # (n//200 could yield 201 points: an off-by-one).
            n = len(es_ha)
            stride = max(1, -(-n // 200))  # ceil division
            curve = [[float(es_ha[i]) * H, float(ds_block[i])]
                     for i in range(0, n, stride)]
            downsampled[label] = curve
        summary['dos_at_fermi_by_spin'] = dos_at_fermi_by_spin
        summary['total_dos_curve_ev'] = downsampled

        # Two clearly-named Fermi-level DOS aggregates in the JSON summary:
        #   * dos_at_fermi_per_channel_ev: the first block's channel,
        #   * dos_at_fermi_both_spins_ev:  alpha + beta (the spin-summed total).
        finite_daf = [v for v in dos_at_fermi_by_spin.values() if v is not None]
        both_spins = float(sum(finite_daf)) if finite_daf else None
        summary['dos_at_fermi_per_channel_ev'] = dos_at_fermi_per_channel
        summary['dos_at_fermi_both_spins_ev'] = both_spins
        # ``total_dos_at_fermi`` is the TOTAL: both spin channels of a
        # spin-polarized DOSS (each block of an unrestricted DOSS.DAT holds one
        # spin), the single block of a closed-shell one. It held the alpha
        # channel only, about 2/3 of Fe's DOS(E_F).
        total_dos_at_fermi = both_spins
        summary['total_dos_at_fermi'] = total_dos_at_fermi

        # Gap-from-DOS / band edges: the zero-DOS window straddling E_F. Recompute
        # the edges here (the analyzer keeps only the width) using the first block
        # (insulator gap edges are spin-degenerate for the systems CRYSTAL writes
        # this way; abs() makes the beta block identical). Consistent-by-design
        # with doss_dat_band_gap_ev which the analyzer derived the same way.
        band_gap_ev = da.get('band_gap_ev')
        summary['band_gap_ev'] = (
            float(band_gap_ev) if band_gap_ev is not None else None)
        summary['metallic'] = (
            bool(da['metallic']) if da.get('metallic') is not None else None)

        vbm_ev = cbm_ev = None
        if block0_es_ha:
            es0 = [e for e in block0_es_ha]
            ds0 = [abs(d) for d in total_dos[0:nepts]]
            pairs = sorted(zip(es0, ds0), key=lambda p: p[0])
            es = [e for e, _ in pairs]
            ds = [d for _, d in pairs]
            span = (es[-1] - es[0]) if len(es) >= 2 else 0.0
            if span > 0:
                margin = 0.02 * span
                interior = [d for e, d in zip(es, ds)
                            if es[0] + margin <= e <= es[-1] - margin] or ds
                ordered = sorted(interior)
                scale = ordered[int(0.95 * (len(ordered) - 1))] if ordered else 0.0
                if scale > 0:
                    near_zero = 0.001 * scale
                    i, n = 0, len(ds)
                    best = None  # (lo_e, hi_e) of the gap window nearest E_F
                    while i < n:
                        if ds[i] < near_zero:
                            j = i
                            while j + 1 < n and ds[j + 1] < near_zero:
                                j += 1
                            lo_e, hi_e = es[i], es[j]
                            dist = (0.0 if lo_e <= 0 <= hi_e
                                    else min(abs(lo_e), abs(hi_e)))
                            if dist <= 0.01 and (best is None or
                                                 (hi_e - lo_e) > (best[1] - best[0])):
                                best = (lo_e, hi_e)
                            i = j + 1
                        else:
                            i += 1
                    if best is not None:
                        vbm_ev = float(best[0]) * H  # top of valence (gap onset)
                        cbm_ev = float(best[1]) * H  # bottom of conduction
        summary['vbm_ev'] = vbm_ev
        summary['cbm_ev'] = cbm_ev

        # numpy is NOT imported at module scope here, so use a self-contained
        # trapezoid (the integrand magnitudes are tiny per-point, no overflow).
        def _trapz(ys, xs):
            if len(ys) != len(xs) or len(xs) < 2:
                return None
            acc = 0.0
            for k in range(1, len(xs)):
                acc += (xs[k] - xs[k - 1]) * (ys[k] + ys[k - 1]) * 0.5
            return acc

        # Integrated states up to E_Fermi == approx number of OCCUPIED states
        # (electrons) per cell. The legacy ``total_states`` integrated the SIGNED
        # alpha(+)/beta(-) concatenated curve over the whole Hartree axis, giving
        # ~0 (physically meaningless), so we DROP that and instead integrate the
        # |DOS| magnitude (states/eV, summed over spins) on the eV axis from the
        # band bottom up to E_Fermi (E <= 0). Units: states/cell (eV cancels).
        integrated_states_to_fermi = None
        if block0_es_ha:
            es_ev_occ = []   # eV energies <= E_Fermi, ascending
            dos_occ = []     # summed-over-spin |DOS| (states/eV) at those energies
            for i in range(nepts):
                e_ha = energy_points[i]
                if e_ha > 0:
                    continue
                e_ev = float(e_ha) * H
                d_sum = 0.0
                for s in range(nspin):
                    idx = s * nepts + i
                    if idx < len(total_dos):
                        d_sum += abs(float(total_dos[idx])) / H
                es_ev_occ.append(e_ev)
                dos_occ.append(d_sum)
            if len(es_ev_occ) >= 2:
                order = sorted(range(len(es_ev_occ)), key=lambda k: es_ev_occ[k])
                xs = [es_ev_occ[k] for k in order]
                ys = [dos_occ[k] for k in order]
                w = _trapz(ys, xs)
                if w is not None:
                    integrated_states_to_fermi = float(w)
        summary['dos_integrated_states_to_fermi'] = integrated_states_to_fermi

        peaks = da.get('peak_positions') or []
        summary['peak_positions_ev'] = [float(p) * H for p in peaks]

        # Per-projection integrated weight (atom/orbital decomposition magnitude):
        # the trapezoid integral of |projection|/H over the first block, in
        # states/cell (states/eV * eV). A single scalar per projection — NOT the
        # per-energy curves. We integrate only the GENUINE projections (the column
        # byte-identical to the TOTAL DOS is excluded — it is not a projection),
        # so len(proj_weights) == num_proj. Keep the neutral projected_dos_N
        # indices (atom/orbital identity is not available from the .DAT).
        proj_weights = {}
        if block0_es_ha and genuine_proj:
            es0 = block0_es_ha
            es0_ev = [float(e) * H for e in es0]
            for name, vals in genuine_proj.items():
                if not isinstance(vals, list) or len(vals) < nepts:
                    continue
                block = [abs(v) / H for v in vals[0:nepts]]  # states/eV
                if len(block) == len(es0_ev) and len(es0_ev) >= 2:
                    w = _trapz(block, es0_ev)  # states/cell
                    if w is not None:
                        proj_weights[str(name)] = float(w)
        if proj_weights:
            summary['projection_integrated_weights'] = proj_weights

        # --- Queryable scalar rows (numeric/bool) — distinct from the JSON blob.
        extra_scalars = {}
        # DOS@Fermi of both spins (states/eV); the per-spin values follow.
        if total_dos_at_fermi is not None:
            extra_scalars['dos_at_fermi_total'] = total_dos_at_fermi
        for label, val in dos_at_fermi_by_spin.items():
            if val is not None and label in ('alpha', 'beta'):
                extra_scalars[f'dos_at_fermi_{label}'] = float(val)
        if band_gap_ev is not None:
            extra_scalars['dos_band_gap_from_dos_ev'] = float(band_gap_ev)
        if da.get('metallic') is not None:
            extra_scalars['dos_is_metallic'] = bool(da['metallic'])
        # Meaningful integrated occupied-state count (replaces the meaningless ~0
        # signed-trapezoid ``dos_total_states`` that was dropped from the blob).
        if integrated_states_to_fermi is not None:
            extra_scalars['dos_integrated_states_to_fermi'] = (
                float(integrated_states_to_fermi))
        # Genuine projection count == len(projection_integrated_weights); the
        # total-equals column and the header's TOTAL slot are not counted.
        extra_scalars['dos_num_projections'] = int(n_genuine_proj)

        return summary, extra_scalars

    def _extract_dos_properties(self, content: str, output_file: Path) -> Dict[str, Any]:
        """Extract density of states specific properties from DOSS calculations."""
        props = {}
        
        # Check if this is a DOSS calculation
        if 'DENSITY OF STATES' not in content.upper() and 'DOSS' not in content.upper():
            return props
            
        # Extract Fermi energy from DOSS calculation
        fermi_patterns = [
            (r'FERMI ENERGY\s*[=:]\s*([-\d.]+(?:[Ee][+-]?\d+)?)', ('fermi energy',)),
            (r'FERMI LEVEL\s*[=:]\s*([-\d.]+(?:[Ee][+-]?\d+)?)', ('fermi level',)),
            (r'EF\s*[=:]\s*([-\d.]+(?:[Ee][+-]?\d+)?)', ('ef',)),
            (r'FERMI ENERGY\s+([-\d.E+\-]+)', ('fermi energy',)),  # CRYSTAL format: "FERMI ENERGY -0.123E+00"
        ]
        
        for pattern, literals in fermi_patterns:
            fermi_matches = ci_findall(pattern, content, literals)
            if fermi_matches:
                # the run's fermi_energy_au when its SCF prints none
                props['_fermi_dos'] = float(fermi_matches[-1])
                break
                
        # Check for DOSS.DAT file and extract information
        doss_dat_file = output_file.parent / "DOSS.DAT"
        alt_doss_dat = output_file.parent / f"{output_file.stem}.DOSS.DAT"
        
        dat_file = None
        if doss_dat_file.exists():
            dat_file = doss_dat_file
        elif alt_doss_dat.exists():
            dat_file = alt_doss_dat
            
        if dat_file:
            props['doss_dat_exists'] = True
            props['doss_dat_size'] = dat_file.stat().st_size
            
            # Try to extract information from DAT file using existing processor
            try:
                try:
                    from mace.utils.dat_file_processor import DatFileProcessor
                except ImportError:
                    from dat_file_processor import DatFileProcessor
                processor = DatFileProcessor()
                dat_info = processor.process_doss_dat_file(dat_file)
                if dat_info and 'error' not in dat_info:
                    # Merge only compact scalars — never the raw DOS arrays.
                    props['doss_dat_num_energy_points'] = dat_info.get('num_energy_points')
                    props['doss_dat_num_spins'] = dat_info.get('num_spins')
                    erange = dat_info.get('energy_range')
                    if erange:
                        props['doss_dat_energy_min'] = erange[0]
                        props['doss_dat_energy_max'] = erange[1]

                    # LAYERED ADDITION: store the actual DENSITY OF STATES (the
                    # per-spin DOS @ Fermi, gap/band edges from the DOS, integrated
                    # states, projection decomposition weights, and a downsampled
                    # total-DOS curve) as a compact JSON summary, plus queryable
                    # scalar rows. Per the storage analysis we do NOT dump the full
                    # ~1000-pt x N-projection x 2-spin matrix; the raw DOSS.DAT
                    # stays on disk and is re-parsed on demand. Failures here never
                    # disturb the scalars merged above.
                    try:
                        summary, extra_scalars = self._build_dos_summary(
                            dat_info, content, output_file, dat_file)
                        if summary:
                            # Key prefixed 'dos_' so _categorize_property routes
                            # it to 'density_of_states' (a bare 'density_of_states'
                            # key would be caught by the earlier 'density' ->
                            # 'structural' branch).
                            props['dos_structure'] = summary
                        for k, v in (extra_scalars or {}).items():
                            if v is not None:
                                props[k] = v
                    except Exception as e:
                        print(f"Warning: Could not build density-of-states summary: {e}")
            except ImportError:
                pass  # DAT file processor not available
            except Exception as e:
                print(f"Warning: Could not process DOSS.DAT file: {e}")
            # a band-index DOS that stops below E_F + 15 eV (mace.utils.doss_reach: the plot's note too)
            from mace.utils import doss_reach
            reach = doss_reach.cached_note(dat_file)
            if reach:
                props['doss_reach_note'] = reach
                print(f"Note: {output_file.name}: {reach}")
        
        # Mark this as a DOS calculation
        if props:
            props['calculation_type'] = 'DOSS'
            props['has_dos'] = True
            
        return props
    
    _NUM_INTEGRATED = re.compile(r'NUMERICALLY INTEGRATED DENSITY\s+(-?[\d.]+)\s+(-?[\d.]+)')
    _SUMMED_SPIN = re.compile(r'SUMMED SPIN DENSITY\s+(-?[\d.]+)')
    # |alpha - beta| electrons above this is a net spin (smeared metals and
    # SCF noise leave ~1e-2 and less)
    NET_SPIN_TOL = 0.05

    def _is_magnetic(self, properties: Dict[str, Any], content: Optional[str]) -> bool:
        """Whether the run is spin-polarized with a spin: unrestricted
        (UNRESTRICTED OPEN SHELL / spin_polarized) and with a net spin (alpha
        - beta electrons of the last SCF: SUMMED SPIN DENSITY, or the
        alpha/beta columns of NUMERICALLY INTEGRATED DENSITY, or the alpha-beta
        Mulliken total) or atomic spin populations (antiferromagnetic). An
        unrestricted run that prints no spin at all (a FREQ restarted from a
        wavefunction, no SCF printed) keeps the earlier rule, the Mulliken
        spin tables."""
        mulliken_spin = bool(properties.get('_mulliken_spin', False))
        if content is None:  # no output text: as before, by the Mulliken print
            return mulliken_spin
        unrestricted = ('UNRESTRICTED OPEN SHELL' in content
                        or bool(properties.get('spin_polarized'))
                        or mulliken_spin)
        if not unrestricted:
            return False
        net = None
        summed = self._SUMMED_SPIN.findall(content)
        if summed:
            net = float(summed[-1])
        else:
            integrated = self._NUM_INTEGRATED.findall(content)
            if integrated:
                net = float(integrated[-1][0]) - float(integrated[-1][1])
        spin_table = properties.get('mulliken_alpha_minus_beta')
        if isinstance(spin_table, dict):
            if net is None and spin_table.get('total_electrons') is not None:
                net = float(spin_table['total_electrons'])
            atoms = spin_table.get('atoms') or []
            if any(abs(_atom_population(a) or 0.0) >= self.NET_SPIN_TOL for a in atoms):
                return True
        if net is None:
            return mulliken_spin
        return abs(net) >= self.NET_SPIN_TOL

    def _classify_electronic_properties(self, properties: Dict[str, Any],
                                        content: Optional[str] = None) -> Dict[str, Any]:
        """electronic_classification (one scheme, property_keys.classify_gap:
        metal below 0.01 eV, semiconductor up to 4 eV, insulator above) from
        band_gap, else from the band structure's gap; and
        magnetic_classification."""
        classification = {}
        candidates = dict(properties)
        if candidates.get('band_gap') is None:
            # both spin channels without a band_gap: the smaller one, as
            # _final_gap_value takes it
            try:
                candidates['band_gap'] = min(float(properties[k]) for k in ('alpha_band_gap', 'beta_band_gap'))
            except (KeyError, TypeError, ValueError):
                pass
        for gap_prop, source in (('band_gap', 'scf'), ('band_fundamental_gap_ev', 'band_structure')):
            try:
                gap = float(candidates[gap_prop])
            except (KeyError, TypeError, ValueError):
                continue
            if gap < 0:
                continue
            kind = property_keys.classify_gap(gap)
            # a gapless band structure with a DOS beside it: metal or
            # semimetal by the DOS at E_F (the band analysis)
            hint = properties.get('_band_dos_classification')
            if source == 'band_structure' and kind == 'metal' and hint in ('metal', 'semimetal'):
                kind = hint
            classification.update({'electronic_classification': kind,
                                   'classification_gap_source': source})
            break

        # Magnetic classification: 'spin_polarized' for an unrestricted run
        # with a spin (_is_magnetic), else 'non_magnetic'
        if self._is_magnetic(properties, content):
            classification['magnetic_classification'] = 'spin_polarized'
        else:
            classification['magnetic_classification'] = 'non_magnetic'
        return classification
    
    def _extract_raman_intensities(self, content: str) -> Dict[str, Any]:
        """Raman intensities of a FREQ run with RAMAN/RAMSPEC: CRYSTAL's
        "AVERAGED ISOTROPIC INTENSITIES" table (I_tot, I_par, I_perp per mode,
        normalized to the most intense peak). Self-skipping: {} without it."""
        head = content.find('AVERAGED ISOTROPIC INTENSITIES')
        if head < 0:
            return {}
        rows = []
        row = re.compile(r'^\s*(\d+)-\s*(\d+)\s+(-?\d+\.\d+)\s+\(\s*(\S+)\s*\)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$')
        started = False
        for line in content[head:head + 400000].splitlines()[1:]:
            m = row.match(line)
            if m:
                started = True
                rows.append({'mode_start': int(m.group(1)), 'mode_end': int(m.group(2)),
                             'frequency_cm': float(m.group(3)), 'irrep': m.group(4),
                             'i_tot': float(m.group(5)), 'i_par': float(m.group(6)), 'i_perp': float(m.group(7))})
            elif started and line.strip() and not line.strip().startswith('-'):
                break
        return {'raman_intensities': rows, 'num_raman_modes': len(rows)} if rows else {}

    def _extract_frequency_properties(self, content: str) -> Dict[str, Any]:
        """Extract frequency calculation properties."""
        props = {}
        
        # Check if this is a FREQ calculation
        if 'FORCE CONSTANT MATRIX' not in content and 'VIBRATIONAL' not in content:
            return props
            
        # Mark as frequency calculation
        props['calculation_type'] = 'FREQ'
        props['has_frequency_data'] = True
        
        # Extract vibrational frequencies
        # Note: real CRYSTAL output prints the units line as "(CM**-1)" (no
        # inner parens), so the anchor must tolerate both "(CM**-1)" and the
        # older "(CM**(-1))" form.
        freq_section = re.search(
            # the table ends at the next of these, else at the next dispersion k point or the end of the
            # file: a polymer's phonon-band output has no other terminator after its 99 k-point tables, and
            # the search then rescanned the rest of the file from every one of them (minutes per file)
            r'MODES\s+EIGV\s+FREQUENCIES\s+IRREP.*?\n\s*\(HARTREE\*\*2\)\s*\(CM\*\*-?\(?-?1\)?\)\s*\(THZ\).*?\n(.*?)(?=NORMAL MODES|VIBRATIONAL TEMPERATURES|<RAMAN>|\*{5,}|DISPERSION K POINT|\Z)',
            content, re.DOTALL
        )
        
        if freq_section:
            freq_data = []
            freq_lines = freq_section.group(1).strip().split('\n')
            
            for line in freq_lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Parse frequency data lines. Real CRYSTAL output prints the
                # mode range space-separated, e.g. "1-   1" or "4-   6", so the
                # range and the three numeric columns (EIGV, CM**-1, THZ) must
                # be parsed from the line directly rather than from split()[0..3]
                # (a naive split puts "1-" in parts[0] and shifts every column).
                # Format: MODE_RANGE EIGV FREQ_CM FREQ_THZ (IRREP) IR_ACTIVE (INTENSITY) RAMAN_ACTIVE
                row_match = re.match(
                    r'(\d+)-\s*(\d+)\s+([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)',
                    line
                )
                if row_match:
                    try:
                        mode_start = int(row_match.group(1))
                        mode_end = int(row_match.group(2))
                        eigv = float(row_match.group(3))
                        freq_cm = float(row_match.group(4))
                        freq_thz = float(row_match.group(5))

                        # Parse the symmetry/activity tail. Real layout after
                        # the three numeric columns is:
                        #   (IRREP)   IR_FLAG (INTENSITY)   RAMAN_FLAG
                        # e.g. "(A  )   A (     0.93)   A" where IR_FLAG /
                        # RAMAN_FLAG are single A(ctive)/I(nactive) letters and
                        # the bracketed value is the integrated IR intensity in
                        # KM/MOL. Default to None/0.0 when the tail is absent
                        # (files without IR/Raman columns must not crash).
                        irrep = None
                        ir_active = False
                        raman_active = False
                        intensity = 0.0
                        tail_match = re.search(
                            r'\(\s*([^)]*?)\s*\)\s*([AI])\s*\(\s*([-\d.E+]+)\s*\)\s*([AI])',
                            line[row_match.end():]
                        )
                        if tail_match:
                            irrep = tail_match.group(1).strip()
                            ir_active = tail_match.group(2) == 'A'
                            try:
                                intensity = float(tail_match.group(3))
                            except ValueError:
                                intensity = 0.0
                            raman_active = tail_match.group(4) == 'A'
                        else:
                            # Fallback: at least capture the IRREP label if the
                            # full activity tail does not match this row's shape.
                            irrep_match = re.search(r'\(([^)]+)\)', line[row_match.end():])
                            if irrep_match:
                                irrep = irrep_match.group(1).strip()

                        freq_data.append({
                            'mode_start': mode_start,
                            'mode_end': mode_end,
                            'eigenvalue': eigv,
                            'frequency_cm': freq_cm,
                            'frequency_thz': freq_thz,
                            'irrep': irrep,
                            'ir_active': ir_active,
                            'raman_active': raman_active,
                            'ir_intensity': intensity
                        })
                    except (ValueError, IndexError):
                        continue
            
            if freq_data:
                props['vibrational_frequencies'] = freq_data
                # Number of printed mode rows. Periodic outputs group degenerate
                # modes into one row (e.g. "4-   6"); num_vibrational_modes is
                # the number of parsed rows so it matches len(vibrational_frequencies).
                props['num_vibrational_modes'] = len(freq_data)
                # Degeneracy-expanded count (sums each row's mode range) — this
                # is the 3N-consistent number consumers should use for mode
                # bookkeeping; the row count above is kept for back-compat.
                props['num_vibrational_modes_total'] = sum(
                    (f['mode_end'] - f['mode_start'] + 1) for f in freq_data
                )

                # Imaginary modes are flagged by a NEGATIVE frequency in CRYSTAL
                # output (no trailing 'i'). Counted as the GUI's run analysis
                # counts them (one shared rule): without the 3/5/6 modes
                # closest to zero (rigid-body motion), each mode of a
                # degenerate row, and only beyond the 20 cm-1 noise. This is
                # always set when modes were parsed.
                try:
                    from mace.utils.imaginary_modes import count_imaginary, is_linear, is_molecule
                except ImportError:
                    from imaginary_modes import count_imaginary, is_linear, is_molecule
                molecule = is_molecule(content)
                linear = molecule and is_linear(
                    [(a['x'], a['y'], a['z']) for a in self._extract_atomic_positions(content)])
                props['num_imaginary_frequencies'] = count_imaginary(
                    [f['frequency_cm'] for f in freq_data
                     for _ in range(f['mode_end'] - f['mode_start'] + 1)],
                    molecule, linear)

                # Lowest / highest vibration, without the rigid-body modes:
                # the 3 (molecule 6, linear molecule 5) modes closest to zero,
                # as num_imaginary_frequencies leaves them out. The lowest is
                # negative for an imaginary mode. (Before 1.1.6 the lowest
                # positive mode above 0.1 cm-1 was stored, usually a
                # rigid-body mode of a few cm-1.)
                from mace.utils.imaginary_modes import vibrations
                modes = vibrations([f['frequency_cm'] for f in freq_data
                                    for _ in range(f['mode_end'] - f['mode_start'] + 1)],
                                   molecule, linear)
                if modes:
                    props['lowest_frequency_cm'] = min(modes)
                    props['highest_frequency_cm'] = max(modes)
        
        # Extract vibrational temperatures
        vib_temp_match = re.search(r'VIBRATIONAL TEMPERATURES \(K\).*?\n\s*TO MODES\s*\n(.*?)(?=\*{5,}|\n\n)', content, re.DOTALL)
        if vib_temp_match:
            temps = []
            temp_lines = vib_temp_match.group(1).strip().split('\n')
            for line in temp_lines:
                # Extract temperatures and mode info
                temp_values = re.findall(r'([\d.]+)\s*\[\s*(\d+);([^]]+)\]', line)
                for temp, mode, irrep in temp_values:
                    temps.append({
                        'temperature_k': float(temp),
                        'mode_number': int(mode),
                        'irrep': irrep.strip()
                    })
            if temps:
                props['vibrational_temperatures'] = temps
        
        # Extract thermodynamic properties (includes electronic_energy EL,
        # Gibbs, and enthalpy H = Gibbs + TS — see _extract_thermodynamic_properties)
        thermo_props = self._extract_thermodynamic_properties(content)
        props.update(thermo_props)

        # Extract zero-point energy
        zpe_match = re.search(r'E0\s*:\s*([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)', content)
        if zpe_match:
            props['zero_point_energy_au'] = float(zpe_match.group(1))
            props['zero_point_energy_ev'] = float(zpe_match.group(2))
            props['zero_point_energy_kj_mol'] = float(zpe_match.group(3))
        
        # Extract force constants information
        force_const_match = re.search(r'FORCE CONSTANT MATRIX.*?MAX ABS\(DGRAD\).*?\n(.*?)(?=GCALCO|\n\n)', content, re.DOTALL)
        if force_const_match:
            props['has_force_constants'] = True
            
            # Count number of displaced calculations
            displaced_atoms = len(re.findall(r'^\s*\d+\s+\w+\s+D[XYZ]', force_const_match.group(1), re.MULTILINE))
            if displaced_atoms > 0:
                props['num_displaced_calculations'] = displaced_atoms
        
        # Extract symmetry-allowed directions
        symm_dirs_match = re.search(r'SYMMETRY ALLOWED INTERNAL DEGREE\(S\) OF FREEDOM:\s*(\d+)', content)
        if symm_dirs_match:
            props['symmetry_allowed_dof'] = int(symm_dirs_match.group(1))
        else:
            no_symm_match = re.search(r'THERE ARE NO SYMMETRY ALLOWED DIRECTIONS', content)
            if no_symm_match:
                props['symmetry_allowed_dof'] = 0
        
        # Extract normal mode displacements if available
        normal_modes_match = re.search(r'NORMAL MODES NORMALIZED TO CLASSICAL AMPLITUDES.*?\n(.*?)(?=\*{5,}|VIBRATIONAL)', content, re.DOTALL)
        if normal_modes_match:
            props['has_normal_mode_displacements'] = True

        return props

    # ------------------------------------------------------------------
    # Thermoelectric / electronic transport (CRYSTAL BOLTZTRA / BoltzTraP)
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_transport_dat(dat_file: Path, axes=None):
        """Parse one CRYSTAL/BoltzTraP transport ``.dat`` file.

        These files store a tensor property as a function of chemical potential
        ``Mu(eV)`` and temperature ``T(K)``.  The per-temperature block markers
        (``# T(K) = 300.000 Npoints = ...``) are *commented*, so the temperature
        is taken from column 2 of every data row (which carries the same value
        and is therefore robust to the comment style).  Layout::

            # 0 [<property> in SI units, i.e. in <unit>]
            # Mu(eV) T(K) N(#carriers) <tensor components...>
            # T(K) = 300.000 Npoints = 711 V(cm^3) = 1.123E-21
              -3.400  300.000  1.351E+00  <xx> <xy> ... <zz>

        An open-shell run (CRYSTAL23 manual, BOLTZTRA, p. 315) writes SIGMA,
        SIGMAS and KAPPA as an alpha block and then a beta block (``# 1
        [Alpha ...]`` / ``# 1 [Beta ...]``); the rows returned are the
        totals, alpha + beta (see ``_parse_transport_dat_tensor``).

        The off-diagonal components are ~0 for these cubic-ish cells; we keep the
        isotropic average of the diagonal (xx+yy+zz)/3 as the scalar property.

        Returns ``(unit_str_or_None, rows)`` where each row is the native tuple
        ``(T, mu, n_carriers, diag_avg)``.  Never raises: returns ``(None, [])``
        on a missing/unreadable/malformed file.
        """
        unit, rows, file_axes = CrystalPropertyExtractor._transport_tensor_rows(dat_file)
        return unit, CrystalPropertyExtractor._average_rows(rows, file_axes, axes)

    @staticmethod
    def periodic_axes(content: Optional[str]) -> Tuple[str, ...]:
        """The tensor axes along which a run is periodic: xx, yy (a slab),
        xx (a polymer), all three otherwise."""
        dim = CrystalPropertyExtractor._dimensionality(content or '')
        return ('xx', 'yy', 'zz')[:dim] if dim in (1, 2) else ('xx', 'yy', 'zz')

    @staticmethod
    def _average_rows(rows, file_axes, axes=None):
        """Rows ``(T, mu, n, value)``: the diagonal averaged over ``axes``
        (None: every diagonal element of the file)."""
        idx = [i for i, a in enumerate(file_axes) if axes is None or a in axes]
        if not idx:
            idx = list(range(len(file_axes))) or [0]
        return [(T, mu, n, sum(diag[i] for i in idx) / len(idx)) for (T, mu, n, diag) in rows
                if len(diag) > max(idx)]

    @staticmethod
    def _axis_rows(rows, file_axes, axis):
        """Rows ``(T, mu, n, value)`` of one diagonal element."""
        i = list(file_axes).index(axis)
        return [(T, mu, n, diag[i]) for (T, mu, n, diag) in rows if len(diag) > i]

    @staticmethod
    def _pf_peak(s_rows, sig_rows):
        """``(peak PF = S^2 |sigma|, (T, mu))`` over the points whose |sigma|
        is at least 1e-3 of the peak |sigma| (the extractor's rule)."""
        sig_lut = {(round(T, 3), round(mu, 4)): v for (T, mu, _nc, v) in sig_rows}
        sig_abs = [abs(v) for v in sig_lut.values() if v != 0.0]
        floor = (max(sig_abs) if sig_abs else 0.0) * 1e-3
        peak, loc = 0.0, None
        for (T, mu, _nc, s_val) in s_rows:
            g = sig_lut.get((round(T, 3), round(mu, 4)))
            if floor <= 0.0 or g is None or abs(g) < floor:
                continue
            pf = s_val * s_val * abs(g)
            if pf > peak:
                peak, loc = pf, (T, mu)
        return peak, loc

    # "# 0 [...]" closed shell; "# 1 [Alpha ...]" / "# 1 [Beta ...]" open shell
    _TRANSPORT_HEAD_RE = re.compile(r'^#\s*([01])\s*\[\s*(.*?)\]?\s*$')

    @staticmethod
    def _read_transport_blocks(dat_file: Path):
        """``(unit, descriptor, {spin: rows})`` of a transport ``.dat`` file,
        each row ``(T, mu, n_carriers, diag)`` with ``diag`` the tensor's
        diagonal; ``spin`` is None for a closed-shell (or spin-independent)
        block, else "alpha" / "beta" from the block's header."""
        try:
            if dat_file is None or not dat_file.exists():
                return None, None, {}, ()
            text = dat_file.read_text(errors="ignore")
        except OSError:
            return None, None, {}, ()

        unit = None
        descriptor = None
        spin = None
        columns = None   # the tensor columns of the "# Mu(eV) T(K) N(#carriers) ..." header
        layout = None    # (diagonal indices, their axes), from the first data row
        blocks: Dict[Any, List] = {}
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith('#'):
                if columns is None and 'mu(ev)' in s.lower():
                    columns = s.lstrip('#').split()[3:]
                head = CrystalPropertyExtractor._TRANSPORT_HEAD_RE.match(s)
                if head and 'mu(ev)' not in s.lower() and not s.lower().startswith('# t(k)'):
                    desc = head.group(2).strip()
                    word = desc.split()[0].lower() if desc.split() else ""
                    spin = word if head.group(1) == '1' and word in ('alpha', 'beta') else None
                    if spin:
                        desc = desc.split(None, 1)[1] if len(desc.split()) > 1 else ""
                    if descriptor is None:
                        descriptor = desc
                # Units descriptor: "# 0 [Seebeck coefficient ... i.e. in V/K]".
                # Skip the column-header comment ("# Mu(eV) T(K) ...").
                if unit is None and 'units' in s.lower() and 'mu(ev)' not in s.lower():
                    m = re.search(r'\[(.*)\]', s)
                    if m:
                        desc = m.group(1).strip()
                        # The descriptor is verbose ("Seebeck coefficient in SI
                        # units, i.e. in V/K"); keep just the clean trailing unit
                        # ("... in V/K" -> "V/K"). Fall back to the full descriptor
                        # if no "in <unit>" tail is present.
                        um = re.search(r'\bin\s+(\S+)\s*$', desc)
                        unit = um.group(1) if um else desc
                continue
            parts = s.split()
            if len(parts) < 4:
                continue
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            if not all(math.isfinite(v) for v in vals):
                continue
            mu, T, n_carr = vals[0], vals[1], vals[2]
            tensor = vals[3:]
            if layout is None or len(tensor) != layout[2]:
                layout = CrystalPropertyExtractor._diagonal_layout(columns, len(tensor)) + (len(tensor),)
            diag = tuple(tensor[i] for i in layout[0])
            blocks.setdefault(spin, []).append((T, mu, n_carr, diag))
        return unit, descriptor, blocks, (layout[1] if layout else ())

    @staticmethod
    def _diagonal_layout(columns, n: int):
        """``(indices, axes)`` of a transport tensor's diagonal among its
        ``n`` columns: by the column names of the file's header (S_xx ...
        S_zz, sigma_xx ... sigma_zz), else by the count - 9 a full 3x3
        (SEEBECK), 6 a symmetric one (SIGMA, KAPPA), and for a slab (2D) 4
        (S_xx S_xy S_yx S_yy) or 3 (sigma_xx sigma_xy sigma_yy, the manual's
        example, p. 315)."""
        if columns and len(columns) == n:
            found = [(i, c.rsplit('_', 1)[-1].lower()) for i, c in enumerate(columns)]
            diag = [(i, a) for i, a in found if a in ('xx', 'yy', 'zz')]
            if diag:
                return tuple(i for i, _ in diag), tuple(a for _, a in diag)
        by_count = {9: ((0, 4, 8), ('xx', 'yy', 'zz')), 6: ((0, 3, 5), ('xx', 'yy', 'zz')),
                    4: ((0, 3), ('xx', 'yy')), 3: ((0, 2), ('xx', 'yy'))}
        return by_count.get(n, ((0,), ('xx',)))

    @staticmethod
    def _parse_transport_dat_tensor(dat_file: Path, with_axes: bool = False):
        """``(unit, rows)`` of a transport ``.dat`` file, each row
        ``(T, mu, n_carriers, diag)``: the spin total of an open-shell file.

        CRYSTAL23 manual, BOLTZTRA (p. 315): an open-shell run writes SIGMA,
        SIGMAS and KAPPA as an alpha block then a beta block, and "the Seebeck
        coefficient, SEEBECK.DAT file, is not spin dependent" (one block).
        The two spin channels conduct in parallel and Eqs. 14.1-14.3 are sums
        over bands, so sigma, sigma*S, kappa0 and the carrier count of the
        run are alpha + beta. The rows used to be keyed on (T, mu) alone, so
        the beta block overwrote the alpha one.

        A SEEBECK file with per-spin blocks (not the documented layout) is
        combined the only consistent way, S = (sigma_a S_a + sigma_b S_b) /
        (sigma_a + sigma_b) per diagonal component, with the per-spin sigma of
        the SIGMA.dat beside it; without one it gives no rows (with a
        warning) rather than a guess.
        """
        unit, rows, axes = CrystalPropertyExtractor._transport_tensor_rows(dat_file)
        return (unit, rows, axes) if with_axes else (unit, rows)

    @staticmethod
    def _transport_tensor_rows(dat_file: Path):
        """``(unit, rows, axes)``: _parse_transport_dat_tensor's rows and the
        axes of their diagonal (('xx', 'yy', 'zz'); ('xx', 'yy') for a slab's
        file)."""
        unit, descriptor, blocks, axes = CrystalPropertyExtractor._read_transport_blocks(dat_file)
        spins = [k for k in blocks if k is not None]
        if not spins:
            return unit, blocks.get(None, []), axes
        if set(spins) != {'alpha', 'beta'} or None in blocks:
            print(f"Warning: {dat_file.name}: spin blocks {sorted(map(str, blocks))} are not one alpha "
                  f"and one beta block; its data are not used")
            return unit, [], axes
        alpha, beta = blocks['alpha'], blocks['beta']

        def key(r):
            return (round(r[0], 3), round(r[1], 4))

        # the file's kind by its name, else by its header: SIGMAS.dat's header ("electrical
        # conductivity * Seebeck coefficient") names the Seebeck coefficient too, and an open-shell
        # SIGMAS.dat read as a per-spin SEEBECK.dat gave no rows
        named = re.search(r'\.(seebeck|sigmas|sigma|kappa)\.dat$', dat_file.name.lower())
        seebeck = (named.group(1) == 'seebeck' if named
                   else (descriptor or '').lower().startswith('seebeck'))
        if not seebeck:
            # additive: alpha + beta, component by component
            b_lut = {key(r): r for r in beta}
            rows = []
            for r in alpha:
                b = b_lut.pop(key(r), None)
                if b is None:
                    rows.append(r)
                    continue
                rows.append((r[0], r[1], r[2] + b[2],
                             tuple(x + y for x, y in zip(r[3], b[3]))))
            rows.extend(b_lut.values())
            return unit, rows, axes

        name = dat_file.name
        idx = name.lower().rfind('.seebeck.dat')
        sig_file = None
        if idx >= 0:
            for suffix in ('.SIGMA.dat', '.sigma.dat', '.SIGMA.DAT'):
                cand = dat_file.with_name(name[:idx] + suffix)
                if cand.exists():
                    sig_file = cand
                    break
        _, _, sig_blocks, _ = CrystalPropertyExtractor._read_transport_blocks(sig_file)
        if set(sig_blocks) != {'alpha', 'beta'}:
            print(f"Warning: {dat_file.name} has per-spin Seebeck blocks and no per-spin SIGMA.dat beside "
                  f"it to weight them with; its data are not used")
            return unit, [], axes
        sa = {key(r): r[3] for r in sig_blocks['alpha']}
        sb = {key(r): r[3] for r in sig_blocks['beta']}
        b_lut = {key(r): r for r in beta}
        rows = []
        for r in alpha:
            k = key(r)
            b = b_lut.get(k)
            ga, gb = sa.get(k), sb.get(k)
            if b is None or ga is None or gb is None or len(ga) != len(r[3]):
                continue
            diag = []
            for s_a, s_b, g_a, g_b in zip(r[3], b[3], ga, gb):
                w = g_a + g_b
                diag.append((g_a * s_a + g_b * s_b) / w if w != 0.0 else 0.5 * (s_a + s_b))
            rows.append((r[0], r[1], r[2] + b[2], tuple(diag)))
        return unit, rows, axes

    def _extract_transport_properties(self, content: str, output_file: Path) -> Dict[str, Any]:
        """Extract thermoelectric / electronic transport properties.

        Detects a CRYSTAL "THERMOELECTRIC AND ELECTRONIC TRANSPORT PROPERTIES
        CALCULATION" (BOLTZTRA) run and, when its companion BoltzTraP ``.dat``
        files are present next to the ``.out`` (``*.SEEBECK.dat`` / ``*.SIGMA.dat``
        / ``*.KAPPA.dat``), stores:

          * a compact JSON ``transport`` summary (temperatures covered, mu range,
            peak |Seebeck| with its (T, mu) and carrier type, peak power factor,
            peak electronic ZT, units, and the source data files referenced), and
          * flat queryable scalar rows (``transport_seebeck_max_uv_per_k``,
            ``transport_power_factor_max``, ``transport_zt_electronic_max``
            (alias ``transport_zt_max``),
            ``transport_n_temperatures``).

        Mirrors ``_build_band_structure_summary``: a compact payload + flat
        scalars, native python types, no schema change (the dict/list is
        ``json.dumps``-ed into ``property_value_text``).  Only peaks + a reference
        to the ``.dat`` are stored — never the full multi-thousand-row tables.
        Failures never propagate (no transport key emitted on absent/malformed
        data).

        Physical guards (these BoltzTraP outputs are noisy):
          * BoltzTraP sets S to a huge value where "THE DETERMINANT OF THE
            CONDUCTIVITY MATRIX IS VERY SMALL"; we only evaluate operating points
            whose electrical conductivity is within 1e-3 of the peak |sigma| so
            those determinant-singularity spikes are excluded.
          * Power factor PF = S^2 * sigma; electronic figure of merit
            ZT_e = PF * T / kappa_e (electronic kappa only -- no lattice term, so
            it is an upper bound the summary labels ``zt_electronic``).
          * KAPPA.dat (the manual's kappa_e, Eq. 14.3) is the zero-field coefficient
            kappa0 = (1/T) int(-df/dE)(E-mu)^2 Xi ("K coefficient" in its
            header), and the electronic kappa_e = kappa0 - T*sigma*S^2 (zero current). So
            ZT_e = x / (1 - x) with x = PF*T/kappa0 (PF*T/kappa0 alone is
            below 1 always). A point with x >= 1 (kappa_e <= 0, outside the
            model) is skipped with a warning, and so is one whose kappa_e is
            below the files' 8-digit precision (|kappa_e| < 2e-6 * kappa0,
            transport_zt.kappa_e_state: rounding noise, where ZT_e ran up to
            1e9); kappa0 and kappa_e at the peak
            are stored as transport_kappa0_at_zt_max and
            transport_kappa_electronic_at_zt_max (W/m/K). All from the
            diagonal averages of the tensors.
        """
        props: Dict[str, Any] = {}

        # Only act on a real transport (BOLTZTRA) run -- judged from lines the
        # properties program prints, not from the echoed title.
        if not is_transport_output(content):
            return props

        # What save_properties_to_database needs to settle the run's earlier
        # total ZT rows (never stored itself: a "_" key): the fingerprint of
        # the files a total would be computed from, and whether a kappa_L
        # given to this extraction was refused.
        zt_state = props['_transport_zt'] = {'inputs': None, 'refused': False}
        try:
            props['calculation_type'] = 'TRANSPORT'
            props['has_transport'] = True

            # Temperature / chemical-potential grid from the .out header (if present).
            def _grab(label):
                m = re.search(label + r'\s*[:=]?\s*(-?\d+\.\d+)', content)
                return float(m.group(1)) if m else None

            t_min = _grab(r'MIN TEMPERATURE VALUE \(K\)')
            t_max = _grab(r'MAX TEMPERATURE VALUE \(K\)')
            mu_min_hdr = _grab(r'MIN CHEMICAL POTENTIAL VALUE \(eV\)')
            mu_max_hdr = _grab(r'MAX CHEMICAL POTENTIAL VALUE \(eV\)')

            # Locate companion BoltzTraP .dat files (case-insensitive stem match).
            out_dir = output_file.parent
            stem = output_file.stem

            def _find(suffix):
                # Prefer the exact-stem companion; fall back to a companion
                # whose stem starts at a separator boundary (same rule as the
                # CUBE sidecar match) so a sibling material's data in a shared
                # directory is never silently attributed to this one.
                # MACE's own runs write .DAT (x.SEEBECK.DAT): every case of the extension
                variants = (suffix, suffix.lower(), suffix[:-len('dat')] + 'DAT')
                for v in variants:
                    exact = out_dir / f"{stem}.{v}"
                    if exact.exists():
                        return exact
                cands = sorted({c for v in variants for c in out_dir.glob(f"*.{v}")})
                for c in cands:
                    c_stem = c.name[:-(len(suffix) + 1)]
                    if (c_stem == stem or c_stem.startswith(stem + '_')
                            or stem.startswith(c_stem + '_')):
                        return c
                # Unambiguous single companion: accept it (common layout —
                # one material per calc dir with differently-styled names).
                return cands[0] if len(cands) == 1 else None

            seebeck_file = _find('SEEBECK.dat')
            sigma_file = _find('SIGMA.dat')
            kappa_file = _find('KAPPA.dat')

            if seebeck_file is None:
                # No data to summarize beyond the calc-type marker.
                props['transport_data_available'] = False
                return props

            # the tensors' diagonals, and the axes the run is periodic along:
            # the *_avg values average those (a slab: in plane), and each
            # axis gets its own PF and ZT (B8: S̄²σ̄ of averaged tensors is
            # no physical quantity of an anisotropic cell)
            s_unit, s_t, s_axes = self._transport_tensor_rows(seebeck_file)
            _, sig_t, sig_axes = self._transport_tensor_rows(sigma_file)
            _, kap_t, kap_axes = self._transport_tensor_rows(kappa_file)
            axes = [a for a in self.periodic_axes(content) if a in s_axes
                    and (not sig_t or a in sig_axes) and (not kap_t or a in kap_axes)] or list(s_axes)
            s_rows = self._average_rows(s_t, s_axes, axes)
            sig_rows = self._average_rows(sig_t, sig_axes, axes) if sig_t else []
            kap_rows = self._average_rows(kap_t, kap_axes, axes) if kap_t else []
            tensor_average = (('(' + '+'.join(axes) + f')/{len(axes)}' if len(axes) > 1 else axes[0])
                              + {2: ', in plane (a slab)', 1: ' (a polymer)'}.get(len(axes), '')
                              + ' of the diagonal of each of S, sigma and kappa0') if axes else None
            per_axis_rows = {a: (self._axis_rows(s_t, s_axes, a),
                                 self._axis_rows(sig_t, sig_axes, a) if sig_t else [],
                                 self._axis_rows(kap_t, kap_axes, a) if kap_t else [])
                             for a in axes}

            if not s_rows:
                props['transport_data_available'] = False
                return props
            zt_state['inputs'] = self._transport_inputs_fingerprint(
                output_file, seebeck_file, sigma_file, kappa_file)

            # The first header's "# 0" (closed shell) or "# 1" (open shell:
            # SIGMA/SIGMAS/KAPPA hold alpha and beta blocks, summed above).
            spin_channels = 'closed shell'
            for f in (sigma_file, kappa_file, seebeck_file):
                if f is None:
                    continue
                try:
                    with open(f, errors='ignore') as fh:
                        first = fh.readline().strip()
                except OSError:
                    continue
                if re.match(r'^#\s*1\s*\[', first):
                    spin_channels = 'alpha+beta'
                break

            # (T, mu) -> property lookups (round mu to align grids across files).
            sig_lut = {(round(T, 3), round(mu, 4)): v
                       for (T, mu, _nc, v) in sig_rows}
            kap_lut = {(round(T, 3), round(mu, 4)): v
                       for (T, mu, _nc, v) in kap_rows}

            sig_abs = [abs(v) for v in sig_lut.values() if v != 0.0]
            sig_max = max(sig_abs) if sig_abs else 0.0
            sig_floor = sig_max * 1e-3  # exclude determinant-singularity points
            # SEEBECK-only runs (no SIGMA.dat): the conductivity gate would
            # otherwise skip every row and store misleading zeros for a valid
            # Seebeck dataset. Compute Seebeck statistics unconditionally and
            # leave only the sigma-dependent PF/ZT unavailable.
            have_sigma = bool(sig_lut)

            temperatures = sorted({round(r[0], 3) for r in s_rows})
            mus = [r[1] for r in s_rows]
            mu_lo = min(mus) if mus else None
            mu_hi = max(mus) if mus else None

            peak_s = 0.0           # signed Seebeck (V/K) at peak |S|
            peak_s_loc = None      # (T, mu)
            peak_s_ncarr = None
            peak_pf = 0.0          # W/m/K^2
            peak_pf_loc = None
            peak_zt = 0.0          # dimensionless (electronic only)
            peak_zt_loc = None
            peak_zt_kappas = None  # (kappa0, kappa_e) at the ZT peak, W/m/K
            n_kappa_skipped = 0    # points with kappa_e <= 0 (x >= 1)
            n_kappa_masked = 0     # points with kappa_e below the files' precision
            n_usable = 0
            from mace.utils.transport_zt import kappa_e_state

            for (T, mu, n_carr, s_val) in s_rows:
                key = (round(T, 3), round(mu, 4))
                g = sig_lut.get(key)
                if have_sigma:
                    # Require a physically meaningful conductivity at this point.
                    if sig_floor <= 0.0 or g is None or abs(g) < sig_floor:
                        continue
                n_usable += 1
                if abs(s_val) > abs(peak_s):
                    peak_s = s_val
                    peak_s_loc = (T, mu)
                    peak_s_ncarr = n_carr
                if g is not None:
                    pf = s_val * s_val * abs(g)
                    if pf > peak_pf:
                        peak_pf = pf
                        peak_pf_loc = (T, mu)
                    k = kap_lut.get(key)   # kappa0 (K coefficient), not kappa_e
                    if k and abs(k) > 0.0:
                        kappa_e = abs(k) - pf * T
                        state = kappa_e_state(k, kappa_e)
                        if state != "ok":
                            if state == "masked":
                                n_kappa_masked += 1
                            else:
                                n_kappa_skipped += 1
                            continue
                        zt = pf * T / kappa_e
                        if zt > peak_zt:
                            peak_zt = zt
                            peak_zt_loc = (T, mu)
                            peak_zt_kappas = (abs(k), kappa_e)
            if n_kappa_skipped:
                print(f"Warning: {output_file.name}: {n_kappa_skipped} transport points with "
                      f"T*sigma*S^2 >= kappa0 (kappa_e <= 0) left out of ZT")
            if n_kappa_masked:
                print(f"Note: {output_file.name}: kappa_e below the files' precision at {n_kappa_masked} "
                      f"transport points (|kappa0 - T*sigma*S^2| < 2e-6 * kappa0) left out of ZT_e")

            has_usable = abs(peak_s) > 0.0
            # Carrier-type convention: N(#carriers) > 0 -> holes (p-type),
            # N(#carriers) < 0 -> electrons (n-type), matching the sign that
            # accompanies the dominant Seebeck operating point.
            carrier_type = None
            if peak_s_ncarr is not None and peak_s_ncarr != 0.0:
                carrier_type = 'p-type' if peak_s_ncarr > 0 else 'n-type'

            def _loc(loc):
                return {'temperature_k': float(loc[0]),
                        'mu_ev': float(loc[1])} if loc else None

            summary = {
                'source_files': [f.name for f in
                                 (seebeck_file, sigma_file, kappa_file) if f],
                'seebeck_source_file': seebeck_file.name,
                'temperatures_k': [float(t) for t in temperatures],
                'n_temperatures': int(len(temperatures)),
                'temperature_min_k': (float(t_min) if t_min is not None
                                      else (float(min(temperatures))
                                            if temperatures else None)),
                'temperature_max_k': (float(t_max) if t_max is not None
                                      else (float(max(temperatures))
                                            if temperatures else None)),
                'mu_min_ev': float(mu_min_hdr) if mu_min_hdr is not None
                             else (float(mu_lo) if mu_lo is not None else None),
                'mu_max_ev': float(mu_max_hdr) if mu_max_hdr is not None
                             else (float(mu_hi) if mu_hi is not None else None),
                'n_data_points': int(len(s_rows)),
                'n_usable_points': int(n_usable),
                'has_usable_data': bool(has_usable),
                # False => SEEBECK-only run: PF/ZT peaks below are not computed.
                'sigma_data_available': bool(have_sigma),
                'carrier_type': carrier_type,
                # open shell: sigma, kappa0 and the carriers are alpha + beta
                'spin_channels': spin_channels,
                'tensor_reduction': (f'the *_max values average the diagonal: {tensor_average}; '
                                     'per_axis: each diagonal element alone; off-diagonal components '
                                     'discarded'),
                'axes': list(axes),
                'units': {
                    'seebeck': s_unit or 'V/K',
                    'electrical_conductivity': '1/Ohm/m',
                    'kappa0': 'W/m/K (KAPPA.dat zero-field K coefficient)',
                    'kappa_electronic': 'W/m/K (kappa0 - T*sigma*S^2)',
                    'power_factor': 'W/m/K^2',
                    'zt': ('dimensionless; ZT_e: electrons only, no lattice kappa, '
                           'an upper bound that diverges as T*sigma*S^2 -> kappa0'),
                },
                'seebeck_max_signed_v_per_k': float(peak_s),
                'seebeck_max_abs_uv_per_k': float(abs(peak_s) * 1.0e6),
                'seebeck_max_at': _loc(peak_s_loc),
                'power_factor_max_w_per_m_k2': (float(peak_pf) if have_sigma
                                                else None),
                'power_factor_max_at': _loc(peak_pf_loc),
                'zt_electronic_max': (float(peak_zt) if peak_zt_loc else None),
                'zt_electronic_max_at': _loc(peak_zt_loc),
                'kappa0_at_zt_max_w_per_m_k': (float(peak_zt_kappas[0])
                                               if peak_zt_kappas else None),
                'kappa_electronic_at_zt_max_w_per_m_k': (float(peak_zt_kappas[1])
                                                         if peak_zt_kappas else None),
                'zt_points_skipped_kappa_e_nonpositive': int(n_kappa_skipped),
                'zt_points_kappa_e_below_precision': int(n_kappa_masked),
            }
            # each axis: the peak PF and ZT_e of its diagonal element alone
            per_axis = {}
            if have_sigma:
                from mace.utils import transport_zt as Z
                for a, (s_a, g_a, k_a) in per_axis_rows.items():
                    pf_a, pf_loc_a = self._pf_peak(s_a, g_a)
                    entry = {'power_factor_max_w_per_m_k2': float(pf_a) if pf_loc_a else None,
                             'power_factor_max_at': _loc(pf_loc_a)}
                    peak_a = Z.zt_points(s_a, g_a, k_a)['zt_e_max'] if k_a else None
                    if peak_a:
                        entry.update({
                            'zt_electronic_max': float(peak_a['value']),
                            'zt_electronic_max_at': _loc((peak_a['temperature_k'], peak_a['mu_ev'])),
                            'kappa0_at_zt_max_w_per_m_k': float(peak_a['kappa0']),
                            'kappa_electronic_at_zt_max_w_per_m_k': float(peak_a['kappa_e'])})
                    per_axis[a] = entry
                summary['per_axis'] = per_axis
            # a k mesh too coarse for transport (mace.utils.transport_mesh: the plot's note too)
            from mace.utils import transport_mesh
            mesh_note = transport_mesh.cached_note(seebeck_file)
            summary['mesh_note'] = mesh_note
            if mesh_note:
                props['transport_mesh_note'] = mesh_note
                print(f"Note: {output_file.name}: {mesh_note}")
            props['transport'] = summary

            # Flat, numerically-queryable scalar rows.
            props['transport_n_temperatures'] = int(len(temperatures))
            props['transport_n_data_points'] = int(len(s_rows))
            props['transport_data_available'] = bool(has_usable)
            if has_usable:
                props['transport_seebeck_max_uv_per_k'] = float(abs(peak_s) * 1.0e6)
                if have_sigma:
                    # PF/ZT need conductivity; for SEEBECK-only runs storing
                    # 0.0 here would be the same misleading-zero bug.
                    props['transport_power_factor_max_avg'] = float(peak_pf)
                    if peak_zt_loc:   # none without a usable KAPPA.dat
                        # electrons only (no lattice kappa): an upper bound on ZT
                        props['transport_zt_electronic_max_avg'] = float(peak_zt)
                        props['transport_kappa0_at_zt_max'] = float(peak_zt_kappas[0])
                        props['transport_kappa_electronic_at_zt_max'] = float(peak_zt_kappas[1])
                props['transport_tensor_average'] = tensor_average
                for quantity, field in (('power_factor', 'power_factor_max_w_per_m_k2'),
                                        ('zt_electronic', 'zt_electronic_max')):
                    values = {a: e[field] for a, e in per_axis.items() if e.get(field) is not None}
                    for a, v in values.items():
                        props[f'transport_{quantity}_max_{a}'] = v
                    if values:
                        props[f'transport_{quantity}_best_axis'] = max(values, key=values.get)
                if carrier_type:
                    props['transport_carrier_type'] = carrier_type
                if have_sigma and peak_zt_loc and self.kappa_lattice is not None:
                    total = self._transport_zt_total(content, output_file, s_rows, sig_rows,
                                                     kap_rows, temperatures, per_axis_rows)
                    if total:
                        total['transport_zt_total_inputs'] = zt_state['inputs']
                    props.update(total)

        except Exception as e:  # never let transport extraction break the run
            print(f"Warning: Could not extract transport properties: {e}")

        if self.kappa_lattice is not None and 'transport_zt_total_max_avg' not in props:
            zt_state['refused'] = True
        return props

    @staticmethod
    def _transport_inputs_fingerprint(output_file, *dat_files) -> Optional[str]:
        """The run's .out and .dat files as the database build's change
        detection sees them (size and the sha1 of the first and last 64 KiB,
        db_build.fingerprint; an mtime alone never decides), in one string."""
        parts = []
        for f in (output_file, *dat_files):
            if f is None:
                continue
            try:
                size, _mtime, sha = db_build.fingerprint(Path(f))
            except OSError:
                return None
            parts.append(f"{Path(f).name}:{size}:{sha}")
        return ";".join(parts)

    def _transport_zt_total(self, content, output_file, s_rows, sig_rows, kap_rows, temperatures,
                            per_axis_rows=None) -> Dict[str, Any]:
        """The total ZT = S^2 sigma T / (kappa_e + kappa_L) of a TRANSPORT run
        (``self.kappa_lattice``, ``self.tau_fs``), as ``transport_zt_total*``
        keys; nothing, with the reason printed, when MACE refuses it (a
        kappa_L table that misses a run temperature, no RELAXTIM tau in the
        .out). See :mod:`mace.utils.transport_zt` for the tau argument."""
        from mace.utils import transport_zt as Z
        kl = self.kappa_lattice
        try:
            missing = Z.uncovered(kl, temperatures)
            if missing:
                raise Z.KappaLatticeError(
                    f"the kappa_L table {Path(kl.path).name} covers {kl.range_text()}, and the run has "
                    f"{', '.join(f'{t:g} K' for t in missing)}: MACE does not extrapolate")
            tau_run = Z.read_relaxation_time_fs(content)
            tau, scale, tau_source = Z.resolve_tau(tau_run, self.tau_fs, output_file.name)
        except Z.KappaLatticeError as e:
            print(f"❌ {output_file.name}: no total ZT stored: {e}")
            return {}
        r = Z.zt_points(s_rows, sig_rows, kap_rows, kl, scale)
        peak = r["zt_total_max"]
        if peak is None:
            return {}
        detail = {
            'formula': ('S^2*sigma*T/(kappa_e + kappa_L), kappa_e = kappa0 - T*sigma*S^2; '
                        'sigma and kappa0 at tau (scaled by tau/tau_run)'),
            'zt_total_max': float(peak['value']),
            'zt_total_max_at': {'temperature_k': float(peak['temperature_k']), 'mu_ev': float(peak['mu_ev'])},
            'kappa_lattice_source': kl.source(),
            'kappa_lattice_at_max_w_per_m_k': float(peak['kappa_l']),
            'kappa0_at_max_w_per_m_k': float(peak['kappa0']),
            'kappa_electronic_at_max_w_per_m_k': float(peak['kappa_e']),
            'tau_fs': float(tau),
            'tau_run_fs': float(tau_run),
            'tau_source': tau_source,
        }
        # each axis (``per_axis_rows``: axis -> its S, sigma, kappa0 rows)
        axis_peaks = {}
        for a, (s_a, g_a, k_a) in (per_axis_rows or {}).items():
            if not (g_a and k_a):
                continue
            pk = Z.zt_points(s_a, g_a, k_a, kl, scale)["zt_total_max"]
            if pk:
                axis_peaks[a] = pk
        if axis_peaks:
            detail['per_axis'] = {a: {'zt_total_max': float(pk['value']),
                                      'zt_total_max_at': {'temperature_k': float(pk['temperature_k']),
                                                          'mu_ev': float(pk['mu_ev'])},
                                      'kappa_lattice_at_max_w_per_m_k': float(pk['kappa_l']),
                                      'kappa_electronic_at_max_w_per_m_k': float(pk['kappa_e'])}
                                  for a, pk in axis_peaks.items()}
        per_axis_keys = {f'transport_zt_total_max_{a}': float(pk['value']) for a, pk in axis_peaks.items()}
        if axis_peaks:
            per_axis_keys['transport_zt_total_best_axis'] = max(axis_peaks, key=lambda a: axis_peaks[a]['value'])
        return {
            **per_axis_keys,
            'transport_zt_total': detail,
            'transport_zt_total_max_avg': float(peak['value']),
            'transport_zt_total_max_temperature_k': float(peak['temperature_k']),
            'transport_zt_total_max_mu_ev': float(peak['mu_ev']),
            'transport_zt_total_kappa_lattice_at_max': float(peak['kappa_l']),
            'transport_zt_total_kappa_electronic_at_max': float(peak['kappa_e']),
            'transport_zt_total_tau_fs': float(tau),
            'transport_zt_total_lattice_source': kl.source(),
            'transport_zt_total_tau_source': tau_source,
        }

    def _extract_scf_settings(self, content: str) -> Dict[str, Any]:
        """Extract SCF and advanced electronic settings from output file."""
        props = {}
        
        # Extract SCF parameters reported in output
        # INFORMATION lines pattern - these show actual settings used
        info_patterns = {
            'scf_biposize': r'INFORMATION \*+\s*BIPOSIZE\s*\*+.*?COULOMB BIPOLAR BUFFER SET TO\s+(\d+)',
            'scf_exchsize': r'INFORMATION \*+\s*EXCHSIZE\s*\*+.*?EXCHANGE BIPOLAR BUFFER SIZE SET TO\s+(\d+)',
            'scf_maxcycle': r'INFORMATION \*+\s*MAXCYCLE\s*\*+.*?MAX NUMBER OF SCF CYCLES SET TO\s+(\d+)',
            'scf_ppan': r'INFORMATION \*+\s*PPAN\s*\*+.*?MULLIKEN POPULATION ANALYSIS',
            'scf_toldee': r'INFORMATION \*+\s*TOLDEE\s*\*+.*?SCF TOL ON TOTAL ENERGY SET TO\s+(\d+)',
            'scf_tolinteg': r'INFORMATION \*+\s*TOLINTEG\s*\*+.*?COULOMB AND EXCHANGE SERIES TOLERANCES',
            'scf_diis': r'INFORMATION \*+\s*DIIS\s*\*+.*?DIIS FOR SCF ACTIVE',
            'scf_scfdir': r'INFORMATION \*+\s*SCFDIR\s*\*+.*?DIRECT SCF',
            'scf_savewf': r'INFORMATION \*+\s*SAVEWF\s*\*+',
            'scf_savepred': r'INFORMATION \*+\s*SAVEPRED\s*\*+'
        }
        
        for param, pattern in info_patterns.items():
            match = ci_search(pattern, content, _SCF_INFO_LITERALS[param], re.IGNORECASE | re.DOTALL)
            if match:
                if param in ['scf_ppan', 'scf_diis', 'scf_scfdir', 'scf_savewf', 'scf_savepred']:
                    props[param] = True
                elif param in ['scf_biposize', 'scf_exchsize', 'scf_maxcycle', 'scf_toldee']:
                    props[param] = int(match.group(1))
                elif param == 'scf_tolinteg':
                    # The five exponents CRYSTAL used ("8 8 8 9 24"), from the
                    # T1-T5 lines of the header; None when the header is cut.
                    tols = re.findall(r'\(T([1-5])\)\s+10\*\*\s+(-?\d+)', content)
                    first = {}
                    for t, v in tols:
                        first.setdefault(t, str(abs(int(v))))
                    if len(first) == 5:
                        props[param] = ' '.join(first[str(i)] for i in range(1, 6))
                else:
                    props[param] = True
        
        # A two-component (TWOCOMPON) SCF, from the lines CRYSTAL prints for
        # it (calc_detection). Stored only for such a run; the calculation
        # type stays SP.
        if is_two_component_output(content):
            props['scf_two_component'] = True

        # Check for level shifter status
        if 'LEVEL SHIFTER DISABLED' in content:
            props['scf_levshift_enabled'] = False
        elif 'LEVEL SHIFTER ACTIVE' in content or 'LEVSHIFT' in content:
            props['scf_levshift_enabled'] = True
            
        # Extract shrink factors from output
        shrink_match = re.search(r'SHRINK\.\s*FACT\.\(MONKH\.\)\s+(\d+)\s+(\d+)\s+(\d+)', content)
        if shrink_match:
            props['scf_shrink_k1'] = int(shrink_match.group(1))
            props['scf_shrink_k2'] = int(shrink_match.group(2))
            props['scf_shrink_k3'] = int(shrink_match.group(3))
        
        # Extract Gilat shrink factor
        gilat_match = re.search(r'SHRINKING FACTOR\(GILAT NET\)\s+(\d+)', content)
        if gilat_match:
            props['scf_gilat_shrink'] = int(gilat_match.group(1))
        
        # FMIXING percentage: CRYSTAL23 prints 'FOCK/KS MATRIX MIXING SET TO n %'
        fmixing = re.findall(r'FOCK/KS MATRIX MIXING SET TO\s+(\d+)\s*%', content)
        fmixing_match = None if fmixing else re.search(r'FMIXING\s*[:=]\s*(\d+)', content)
        if fmixing:
            props['scf_fmixing_percentage'] = int(fmixing[-1])
        elif fmixing_match:
            props['scf_fmixing_percentage'] = int(fmixing_match.group(1))
        
        # Anderson mixing
        anderson_match = re.search(r'ANDERSON MIXING.*?FACTOR\s*[:=]\s*([\d.]+)', content)
        if anderson_match:
            props['scf_anderson_factor'] = float(anderson_match.group(1))
            props['scf_anderson_enabled'] = True
        
        # DIIS mixing parameters (DIIS itself: scf_diis)
        if 'DIIS FOR SCF ACTIVE' in content:
            # Extract DIIS parameters
            diis_start_match = re.search(r'DIIS STARTING FROM CYCLE\s*(\d+)', content)
            if diis_start_match:
                props['scf_diis_start_cycle'] = int(diis_start_match.group(1))
                
            diis_size_match = re.search(r'DIIS SUBSPACE SIZE\s*[:=]\s*(\d+)', content)
            if diis_size_match:
                props['scf_diis_subspace_size'] = int(diis_size_match.group(1))
        
        # Broyden mixing
        broyden_match = re.search(r'BROYDEN MIXING.*?FACTOR\s*[:=]\s*([\d.]+)', content)
        if broyden_match:
            props['scf_broyden_factor'] = float(broyden_match.group(1))
            props['scf_broyden_enabled'] = True
        
        # Level shifter details
        levshift_match = re.search(r'LEVSHIFT\s*[:=]\s*([\d.]+)\s*CYCLES\s*[:=]\s*(\d+)', content)
        if levshift_match:
            props['scf_levshift_factor'] = float(levshift_match.group(1))
            props['scf_levshift_cycles'] = int(levshift_match.group(2))
        
        # Extract integration pack parameters
        intgpack_patterns = {
            'scf_ilasize': r'ILASIZE\s*[:=]\s*(\d+)',
            'scf_intgpack': r'INTGPACK\s*[:=]\s*(\d+)',
            'scf_madelimit': r'MADELIMIT\s*[:=]\s*(\d+)',
            'scf_poleordr': r'POLEORDR\s*[:=]\s*(\d+)'
        }
        
        for param, pattern in intgpack_patterns.items():
            match = ci_search(pattern, content, (param[len('scf_'):],))
            if match:
                props[param] = int(match.group(1))
        
        # Extract exchange parameters
        if 'EXCHPERM' in content:
            props['scf_exchange_permutation'] = True
            
        if 'BIPOLARIZ' in content:
            props['scf_bipolarization'] = True
        
        # Extract DFT grid information
        grid_match = re.search(r'DFT GRID.*?(\d+)\s+POINTS', content)
        if grid_match:
            props['scf_dft_grid_points'] = int(grid_match.group(1))
            
        # Extract pruning information
        if 'PRUNING' in content:
            props['scf_grid_pruning'] = True
        
        return props
    
    def _extract_mechanical_properties(self, content: str) -> Dict[str, Any]:
        """Elastic, piezoelectric, photoelastic tensors and EOS fits (CRYSTAL23
        manual ch. 12-13), with the Voigt-Reuss-Hill moduli of a 3D run or the
        in-plane moduli of a slab. {} for a run without them."""
        props: Dict[str, Any] = {}
        if not any(k in content for k in ("ELASTIC CONSTANTS", "PIEZOELECTRIC", "POCKELS TENSOR",
                                          "FITTING USING ALL POINTS", "THERMAL PROPERTIES FROM HELMHOLTZ",
                                          "DIELECTRIC (EPSILON)")):
            return props
        try:
            from mace.elastic.analysis import analyze
            data = analyze(content)
        except Exception as e:
            print(f"Warning: could not read the elastic/EOS results: {e}")
            return props
        el = data.get("elastic")
        if el:
            props['elastic_tensor'] = {"matrix": el["matrix"], "unit": el["unit"], "system": el.get("system")}
        if data.get("compliance"):
            props['elastic_compliance_tensor'] = data["compliance"]
        if data.get("seismic"):
            props['elastic_seismic_velocities'] = data["seismic"]
        mod = data.get("moduli") or {}
        if "bulk_modulus_hill_gpa" in mod:
            props['bulk_modulus_voigt_gpa'] = mod["bulk_modulus_voigt_gpa"]
            props['bulk_modulus_reuss_gpa'] = mod["bulk_modulus_reuss_gpa"]
            props['bulk_modulus_hill_gpa'] = mod["bulk_modulus_hill_gpa"]
            props['shear_modulus_voigt_gpa'] = mod["shear_modulus_voigt_gpa"]
            props['shear_modulus_reuss_gpa'] = mod["shear_modulus_reuss_gpa"]
            props['shear_modulus_hill_gpa'] = mod["shear_modulus_hill_gpa"]
            props['elastic_min_eigenvalue_gpa'] = mod["min_eigenvalue_gpa"]
        # an unstable C (not positive definite) keeps its tensor, VRH averages and eigenvalues; the
        # moduli derived from them are unphysical and not stored (mace.elastic.properties does the same)
        if "bulk_modulus_hill_gpa" in mod and mod.get("born_stable") is not False:
            if "youngs_modulus_gpa" in mod:
                props['youngs_modulus_gpa'] = mod["youngs_modulus_gpa"]
                props['poisson_ratio'] = mod["poisson_ratio"]
            if "universal_anisotropy" in mod:
                props['elastic_anisotropy_universal'] = mod["universal_anisotropy"]
            if "debye_temperature_k" in mod:
                props['debye_temperature_elastic_k'] = mod["debye_temperature_k"]
                props['sound_velocity_mean_km_s'] = mod["sound_velocity_mean_km_s"]
        if "born_stable" in mod:
            props['elastic_born_stable'] = mod["born_stable"]
        if "layer_modulus" in mod:
            props['elastic_2d_moduli'] = mod
        if data.get("piezo_e"):
            props['piezoelectric_e_tensor'] = data["piezo_e"]
        if data.get("piezo_d"):
            props['piezoelectric_d_tensor'] = data["piezo_d"]
        if data.get("pockels"):
            props['photoelastic_tensor'] = data["pockels"]
        if data.get("piezo_optic"):
            props['piezo_optic_tensor'] = data["piezo_optic"]
        eos = data.get("eos") or {}
        # the Å³ / GPa keys hold a 3D run's values; a slab's or polymer's QHA prints an area or a length
        # and a modulus per length or area (its tables, with their units, are kept whole)
        flat = data.get("dimensionality") in (None, 3)
        if eos.get("fits") and not flat:
            props['eos_fits'] = eos["fits"]
        elif eos.get("fits"):
            props['eos_fits'] = eos["fits"]
            bm = eos["fits"].get("birch_murnaghan") or next(iter(eos["fits"].values()))
            props['eos_bulk_modulus_gpa'] = bm["bulk_modulus_gpa"]
            props['eos_bulk_modulus_prime'] = bm["bulk_modulus_prime"]
            props['eos_equilibrium_volume_ang3'] = bm["volume_ang3"]
        if eos.get("points"):
            props['eos_points'] = eos["points"]
        diel = data.get("dielectric")
        if diel:
            props['dielectric_cphf'] = diel
            eps = diel.get("principal_epsilon")
            if eps:
                props['dielectric_constant_electronic'] = sum(eps) / len(eps)
            ri = diel.get("refractive_indices")
            if ri:
                props['refractive_index_electronic'] = sum(ri) / len(ri)
        if data.get("beta"):
            props['hyperpolarizability_beta'] = data["beta"]
        if data.get("gamma"):
            props['hyperpolarizability_gamma'] = data["gamma"]
        qha = data.get("qha")
        if qha:
            props['qha_thermal_properties'] = qha
            temps = qha["temperature_k"]
            i300 = min(range(len(temps)), key=lambda i: abs(temps[i] - 300.0))
            if abs(temps[i300] - 300.0) <= 15.0 and flat:
                props['qha_volume_300k_ang3'] = qha["volume_ang3"][i300]
                props['qha_bulk_modulus_300k_gpa'] = qha["bulk_modulus_t_gpa"][i300]
                props['qha_thermal_expansion_300k_per_k'] = qha["alpha_v_per_k"][i300]
        return props

    _TWOC_MAG = re.compile(r"TOTAL ([XYZ])-COMP MAGNETIZATION\s+([-+]?\d+\.\d+(?:[EeDd][-+]?\d+)?)")

    def _extract_two_component_properties(self, content: str) -> Dict[str, Any]:
        """A two-component (TWOCOMPON) SCF (scf_two_component is
        _extract_scf_settings'): the final total and per-atom magnetization vectors it prints each cycle ("TOTAL X-COMP
        MAGNETIZATION" + "ATOMIC MX :" rows; HPCC CRYSTAL23 outputs). {} for a
        one-component run."""
        props: Dict[str, Any] = {}
        if "TWO-COMPONENT SCF" not in content:
            return props
        last = content.rfind("TOTAL X-COMP MAGNETIZATION")
        if last < 0:
            return props
        block = content[last:last + 200000]
        total: Dict[str, float] = {}
        atoms: Dict[str, List[float]] = {}
        lines = block.splitlines()
        i = 0
        while i < len(lines):
            m = self._TWOC_MAG.search(lines[i])
            if m:
                axis = m.group(1)
                if axis in total:
                    break                         # the next cycle's block
                total[axis] = float(m.group(2).replace("D", "E"))
                vals: List[float] = []
                if i + 1 < len(lines) and lines[i + 1].strip().startswith(f"ATOMIC M{axis}"):
                    j = i + 2
                    while j < len(lines) and re.fullmatch(r"(\s*[-+]?\d+\.\d+)+\s*", lines[j] or "x"):
                        vals += [float(x) for x in lines[j].split()]
                        j += 1
                    i = j - 1
                atoms[axis] = vals
                if len(total) == 3:
                    break
            i += 1
        if len(total) == 3:
            mx, my, mz = total["X"], total["Y"], total["Z"]
            props['two_component_magnetization'] = [mx, my, mz]
            props['two_component_magnetization_norm'] = (mx * mx + my * my + mz * mz) ** 0.5
            n = min(len(atoms.get(a, [])) for a in "XYZ")
            if n:
                props['two_component_atomic_magnetization'] = [
                    [atoms["X"][k], atoms["Y"][k], atoms["Z"][k]] for k in range(n)]
        return props

    def _extract_thermodynamic_properties(self, content: str) -> Dict[str, Any]:
        """Extract thermodynamic properties from FREQ calculations."""
        props = {}
        
        # Temperature and pressure conditions
        conditions_match = re.search(r'AT \(T =\s*([\d.]+)\s*K,\s*P =\s*([\d.E+-]+)\s*MPA\)', content)
        if conditions_match:
            props['thermodynamic_temperature_k'] = float(conditions_match.group(1))
            props['thermodynamic_pressure_mpa'] = float(conditions_match.group(2))
        
        # Electronic energy (EL) — the baseline total energy printed at the top
        # of the harmonic thermodynamic block; H and G are built on top of it.
        el_match = re.search(
            r'^\s*EL\s*:\s*([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)',
            content, re.MULTILINE
        )
        if el_match:
            props['electronic_energy_au'] = float(el_match.group(1))
            props['electronic_energy_ev'] = float(el_match.group(2))
            props['electronic_energy_kj_mol'] = float(el_match.group(3))

        # Extract thermodynamic functions. CRYSTAL labels the ET/PV/TS block
        # "...WITH VIBRATIONAL CONTRIBUTIONS" for periodic systems but
        # "...TAKING INTO ACCOUNT MOLECULAR ..." for molecules — match both,
        # else molecular FREQ runs lose Gibbs/ET/PV/TS entirely.
        thermo_section = re.search(
            r'THERMODYNAMIC FUNCTIONS '
            r'(?:WITH VIBRATIONAL CONTRIBUTIONS|TAKING INTO ACCOUNT MOLECULAR).*?'
            r'AU/CELL\s+EV/CELL\s+KJ/MOL\s*\n(.*?)(?=OTHER THERMODYNAMIC|\*{5,})',
            content, re.DOTALL
        )
        
        if thermo_section:
            thermo_content = thermo_section.group(1)
            
            # ET (thermal energy)
            et_match = re.search(r'ET\s*:\s*([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)', thermo_content)
            if et_match:
                props['thermal_energy_au'] = float(et_match.group(1))
                props['thermal_energy_ev'] = float(et_match.group(2))
                props['thermal_energy_kj_mol'] = float(et_match.group(3))
            
            # PV
            pv_match = re.search(r'PV\s*:\s*([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)', thermo_content)
            if pv_match:
                props['pv_term_au'] = float(pv_match.group(1))
                props['pv_term_ev'] = float(pv_match.group(2))
                props['pv_term_kj_mol'] = float(pv_match.group(3))
            
            # TS (entropy term)
            ts_match = re.search(r'TS\s*:\s*([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)', thermo_content)
            if ts_match:
                props['entropy_term_au'] = float(ts_match.group(1))
                props['entropy_term_ev'] = float(ts_match.group(2))
                props['entropy_term_kj_mol'] = float(ts_match.group(3))
            
            # ET+PV-TS (free energy correction)
            correction_match = re.search(r'ET\+PV-TS\s*:\s*([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)', thermo_content)
            if correction_match:
                props['free_energy_correction_au'] = float(correction_match.group(1))
                props['free_energy_correction_ev'] = float(correction_match.group(2))
                props['free_energy_correction_kj_mol'] = float(correction_match.group(3))
            
            # Total free energy (EL+E0+ET+PV-TS)
            total_match = re.search(r'EL\+E0\+ET\+PV-TS:\s*([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)', thermo_content)
            if total_match:
                props['gibbs_free_energy_au'] = float(total_match.group(1))
                props['gibbs_free_energy_ev'] = float(total_match.group(2))
                props['gibbs_free_energy_kj_mol'] = float(total_match.group(3))

            # Enthalpy H = EL + E0 + ET + PV = Gibbs + TS. Built from CRYSTAL's
            # own printed columns (per unit) so it carries no conversion drift.
            if 'gibbs_free_energy_au' in props and 'entropy_term_au' in props:
                props['enthalpy_au'] = props['gibbs_free_energy_au'] + props['entropy_term_au']
                props['enthalpy_ev'] = props['gibbs_free_energy_ev'] + props['entropy_term_ev']
                props['enthalpy_kj_mol'] = props['gibbs_free_energy_kj_mol'] + props['entropy_term_kj_mol']
        
        # Extract entropy and heat capacity
        other_thermo = re.search(
            r'OTHER THERMODYNAMIC FUNCTIONS:.*?'
            r'mHARTREE/\(CELL\*K\)\s+mEV/\(CELL\*K\)\s+J/\(MOL\*K\)\s*\n(.*?)(?=\*{5,}|TTTT)',
            content, re.DOTALL
        )
        
        if other_thermo:
            other_content = other_thermo.group(1)
            
            # Entropy
            entropy_match = re.search(r'ENTROPY\s*:\s*([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)', other_content)
            if entropy_match:
                props['entropy_mhartree_cell_k'] = float(entropy_match.group(1))
                props['entropy_mev_cell_k'] = float(entropy_match.group(2))
                props['entropy_j_mol_k'] = float(entropy_match.group(3))
            
            # Heat capacity
            heat_cap_match = re.search(r'HEAT CAPACITY\s*:\s*([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)', other_content)
            if heat_cap_match:
                props['heat_capacity_mhartree_cell_k'] = float(heat_cap_match.group(1))
                props['heat_capacity_mev_cell_k'] = float(heat_cap_match.group(2))
                props['heat_capacity_j_mol_k'] = float(heat_cap_match.group(3))
        
        return props
    
    def _categorize_property(self, prop_name: str) -> str:
        """The property's category (property_keys.KEYS; 'other' for a name
        the registry does not know)."""
        return property_keys.category(prop_name)

    def _get_property_unit(self, prop_name: str) -> str:
        """The property's unit (property_keys.KEYS). By the module: callers
        also use it unbound (self None)."""
        return property_keys.unit(prop_name)

    def _extract_charge_potential_properties(self, content: str, output_file: Path) -> Dict[str, Any]:
        """Extract CHARGE+POTENTIAL (CRYSTAL ECH3/POT3) metadata.

        A CRYSTAL ECH3 (3D charge density) / POT3 (3D electrostatic potential)
        job stores its real numerical payload -- the 3D grids -- in BINARY/large
        sidecar files (``*_DENS.CUBE``, ``*_POT.CUBE``, ``*_SPIN.CUBE``), NOT in
        the .out text. A text extractor must NOT parse those grids. So, mirroring
        ``_build_band_structure_summary``, this stores ONLY what the .out text
        honestly exposes plus *references* to the grid files on disk:

          * confirmation that ECH3 / POT3 ran (the ``ECH3``/``POT3`` timing banners),
          * the grid divisions (``3D DENSITY, GRID IS nx*ny*nz``) and derived
            number of grid points,
          * the real-space bounding box (``X/Y/Z COORDINATE RANGE IS lo TO hi``),
          * multipolar-expansion order / spin-polarization flag when printed,
          * the SCF total energy / Fermi energy already on the page,
          * the names of the generated ``*.CUBE`` grid files in the same directory
            (so the mace plotting cube viewer, which globs ``*.CUBE``, can find them).

        Returns a compact JSON-serializable ``charge_potential`` summary plus a few
        flat queryable scalar/string rows. All values are native float/int/str/bool
        so they JSON-serialize cleanly. Returns ``{}`` (no keys) when the file has
        no ECH3/POT3 markers, so non-charge calculations are completely untouched.
        Any internal failure degrades gracefully -- it never raises.
        """
        try:
            has_ech3 = 'ECH3' in content and re.search(r'\bECH3\s+(?:START|END)\b', content) is not None
            has_pot3 = 'POT3' in content and re.search(r'\bPOT3\s+(?:START|END)\b', content) is not None
            if not (has_ech3 or has_pot3):
                return {}  # not a charge/potential calc -> contribute nothing

            summary: Dict[str, Any] = {
                'source': 'ECH3/POT3 (.out text + .CUBE grid sidecars)',
                'has_charge_density': bool(has_ech3),
                'has_electrostatic_potential': bool(has_pot3),
            }

            # --- Grid divisions: "3D DENSITY, GRID IS nx*ny*nz" (space-padded
            # numbers like "100* 58* 86" appear, so tolerate inner whitespace).
            # ECH3 and POT3 print the same grid; take the first occurrence.
            grid_dims = None
            num_points = None
            gm = ci_search(
                r'3D\s+\w+,\s*GRID\s+IS\s+(\d+)\s*\*\s*(\d+)\s*\*\s*(\d+)',
                content, ('3d', 'grid', 'is'))
            if gm:
                grid_dims = [int(gm.group(1)), int(gm.group(2)), int(gm.group(3))]
                num_points = int(grid_dims[0] * grid_dims[1] * grid_dims[2])
                summary['grid_divisions'] = grid_dims
                summary['num_grid_points'] = num_points

            # --- Real-space bounding box of the grid (Bohr), if printed.
            box = {}
            for axis, lo, hi in re.findall(
                    r'([XYZ])\s+COORDINATE RANGE IS\s+(-?\d+\.\d+)\s+TO\s+(-?\d+\.\d+)',
                    content):
                box[axis.lower()] = [float(lo), float(hi)]
            if box:
                summary['coordinate_range'] = box

            # --- Multipolar expansion order (POT3 section), if printed.
            mm = re.search(r'MULTIPOLAR EXPANSION UP TO L=\s*(\d+)', content)
            if mm:
                summary['multipole_expansion_l'] = int(mm.group(1))

            # --- Spin-polarization flag.
            spin_polarized = 'SPIN POLARIZED SYSTEM' in content
            summary['spin_polarized'] = bool(spin_polarized)

            # --- Scalar metadata already on the page: SCF total energy / Fermi.
            em = re.search(r'TOTAL ENERGY\s+(-?\d+\.\d+E[+-]?\d+)', content)
            scf_energy = None
            if em:
                try:
                    scf_energy = float(em.group(1))
                    summary['scf_total_energy_au'] = scf_energy
                except ValueError:
                    pass
            fm = re.search(r'FERMI ENERGY\s+(-?\d+\.\d+E?[+-]?\d*)', content)
            fermi_energy = None
            if fm:
                try:
                    fermi_energy = float(fm.group(1))
                except ValueError:
                    pass
                else:
                    # CRYSTAL's -1.0 placeholder (an SCF that ended spin-locked) is no Fermi level
                    from mace.utils import fermi_level
                    fl = fermi_level.run_fermi(output_file, fermi_energy)
                    fermi_energy = fl.value_au
                    if fermi_energy is not None:
                        summary['fermi_energy_au'] = fermi_energy
                    if fl.note:
                        summary['fermi_energy_note'] = fl.note

            # --- Which 3D properties were computed (provenance list).
            computed = []
            if has_ech3:
                computed.append('charge_density')
                if spin_polarized:
                    computed.append('spin_density')
            if has_pot3:
                computed.append('electrostatic_potential')
            summary['computed_properties'] = computed

            # --- Reference the generated grid/cube files on disk so the mace
            # plotting cube viewer (which globs *.CUBE in the .out's directory)
            # can find them. We only RECORD names that actually exist -- never
            # parse the binary grids. The match is scoped to this job's stem.
            grid_files = []
            try:
                output_dir = output_file.parent
                stem = output_file.stem  # e.g. ..._charge+potential
                for cube in sorted(output_dir.glob('*.CUBE')):
                    # Require a separator boundary after the stem so a sibling
                    # material whose stem is a PREFIX of this one (e.g. "mat_1"
                    # vs "mat_12_DENS.CUBE") is not mis-attributed. CRYSTAL names
                    # the sidecars "<stem>_DENS/_POT/_SPIN.CUBE".
                    if cube.stem == stem or cube.stem.startswith(stem + '_'):
                        grid_files.append(cube.name)
            except OSError:
                grid_files = []
            if grid_files:
                summary['grid_files'] = grid_files

            props: Dict[str, Any] = {'charge_potential': summary}

            # --- Flat, queryable scalar/string rows (distinct from the JSON blob).
            props['chargepot_has_ech3'] = bool(has_ech3)
            props['chargepot_has_pot3'] = bool(has_pot3)
            # merged into spin_polarized by extract_all_properties
            props['_chargepot_spin'] = bool(spin_polarized)
            if grid_dims is not None:
                props['chargepot_grid_dims'] = 'x'.join(str(d) for d in grid_dims)
            if num_points is not None:
                props['chargepot_grid_points'] = int(num_points)
            if computed:
                props['chargepot_computed_properties'] = ','.join(computed)
            if grid_files:
                props['chargepot_grid_files'] = ','.join(grid_files)
                props['chargepot_num_grid_files'] = int(len(grid_files))
            if scf_energy is not None:
                props['chargepot_scf_total_energy_au'] = float(scf_energy)

            return props
        except Exception as e:
            print(f"Warning: Could not extract charge/potential properties: {e}")
            return {}

    # Molden files read for the levels/self-check of the index; bigger runs are
    # indexed from the .out and file names only
    ORBITALS_INDEX_MAX_BYTES = 20 * 1024 * 1024
    ORBITALS_CHECK_MAX_NBF = 600
    ORBITALS_LEVEL_WINDOW = 8

    def _extract_orbitals_properties(self, content: str, output_file: Path) -> Dict[str, Any]:
        """Index of a CRYSTAL ORBITALS run (the Molden files it wrote).

        Stores what a later viewer needs to find and choose orbitals: the file
        names and sizes, the k list (fractional k, real/complex, whether MACE can
        draw it), and per k point and spin the HOMO/LUMO and the levels around
        them (Hartree), plus the self-check result. Grids and coefficients stay
        in the files. Returns {} when the .out has no ORBITALS section.
        """
        if 'ORBITALS' not in content:
            return {}
        try:
            from mace.orbitals.outfile import has_orbitals_section, parse_out_text
            if not has_orbitals_section(content):
                return {}
            from mace.orbitals.dataset import OrbitalRun, HARTREE_EV
            out = parse_out_text(content, str(output_file))
            folder = output_file.parent
            files = [folder / n for n in out.files if (folder / n).is_file()]
            summary: Dict[str, Any] = {
                'source': 'ORBITALS (.out text + .molden sidecars)',
                'files_listed': len(out.files),
                'files': [{'name': f.name, 'bytes': f.stat().st_size} for f in files],
                'missing_files': [n for n in out.files if not (folder / n).is_file()],
                'nao': out.nao,
                'electrons': out.electrons,
                'kpoints': [{'index': k.index, 'frac': [float(x) for x in k.frac], 'real': k.real}
                            for k in out.kpoints],
            }
            if out.errors:
                summary['errors'] = out.errors[:5]
            props: Dict[str, Any] = {'orbitals': summary}
            props['orbitals_num_files'] = len(files)
            props['orbitals_total_bytes'] = int(sum(f['bytes'] for f in summary['files']))
            if files:
                props['orbitals_files'] = ','.join(f.name for f in files)
            props['orbitals_num_kpoints'] = len(out.kpoints) if out.kpoints else (1 if files else 0)
            if out.nao is not None:
                props['orbitals_nbf'] = int(out.nao)
            if out.vbm and out.cbm:
                props['orbitals_gap_ev'] = (out.cbm[2] - out.vbm[2]) * HARTREE_EV
            if not files:
                return props
            run = OrbitalRun(files, out)
            props['orbitals_dimensionality'] = run.dimensionality()
            summary['dimensionality'] = run.dimensionality()
            kinfo = []
            statuses = []
            homo_best = None
            for e in run.kpoints():
                rec = e.to_dict()
                rec['files'] = sorted(p.name for p in e.files.values())
                if not e.usable:
                    kinfo.append(rec)
                    continue
                big = sum(p.stat().st_size for p in e.files.values()) > self.ORBITALS_INDEX_MAX_BYTES
                if big:
                    rec['status'] = 'unchecked'
                    kinfo.append(rec)
                    continue
                kd = run.load(e.index, check=(out.nao or 0) <= self.ORBITALS_CHECK_MAX_NBF)
                rec['status'] = kd.status
                statuses.append(kd.status)
                rec['spins'] = []
                for sp in kd.spins:
                    lv = kd.levels(sp)
                    h = kd.homo(sp)
                    lo = max(1, (h or 1) - self.ORBITALS_LEVEL_WINDOW)
                    hi = min(len(lv), (h or 0) + self.ORBITALS_LEVEL_WINDOW + 1)
                    rec['spins'].append({
                        'spin': sp, 'homo': h, 'n_levels': len(lv),
                        'levels_from': lo,
                        'energies_au': [lv[i - 1].energy for i in range(lo, hi + 1)],
                        'occupations': [lv[i - 1].occupation for i in range(lo, hi + 1)],
                        'degenerate_groups': [lv[i - 1].group for i in range(lo, hi + 1)],
                    })
                    if h is not None and (homo_best is None or lv[h - 1].energy > homo_best[1]):
                        lumo = lv[h].energy if h < len(lv) else None
                        homo_best = (h, lv[h - 1].energy, lumo, sp)
                kinfo.append(rec)
            summary['kpoints'] = kinfo
            props['orbitals_kpoints'] = ';'.join(
                f"{k['index']}:{','.join(f'{x:g}' for x in k['frac'])}:{'R' if k['real'] else 'C'}"
                for k in kinfo)
            props['orbitals_usable_kpoints'] = sum(1 for k in kinfo if k.get('usable'))
            if statuses:
                order = {'fail': 3, 'warn': 2, 'unchecked': 1, 'ok': 0}
                props['orbitals_self_check'] = max(statuses, key=lambda s: order.get(s, 1))
            if homo_best:
                props['orbitals_homo_band'] = int(homo_best[0])
                props['orbitals_homo_energy_au'] = float(homo_best[1])
                if homo_best[2] is not None:
                    props['orbitals_lumo_band'] = int(homo_best[0] + 1)
                    props['orbitals_lumo_energy_au'] = float(homo_best[2])
            props['orbitals_spin_polarized'] = any(
                s['spin'] != 'ALPHA+BETA' for k in kinfo for s in k.get('spins', []))
            return props
        except Exception as e:
            print(f"Warning: Could not index ORBITALS files: {e}")
            return {}

    @staticmethod
    def _dat_file_for(output_file: Path, kind: str):
        """The ``<name>.<kind>.DAT`` file that belongs to ``output_file``.

        Its own run's file, ``<stem>.<kind>.DAT``, comes first; otherwise the
        BAND/DOSS run of the same parent, ``<base>_<kind>.<kind>.DAT`` or
        ``<base>.<kind>.DAT``, where ``<base>`` is the stem without a
        ``_band``/``_doss`` suffix. Files of other stems belong to other
        calculations and are never used. Returns ``(path, [])``, or
        ``(None, names)`` when two files of the same rank match.
        """
        stem = output_file.stem
        base = stem
        for suffix in ("_band", "_doss"):
            if base.lower().endswith(suffix):
                base = base[: -len(suffix)]
                break
        tail = f".{kind}.DAT".lower()
        siblings = {f"{base}_{kind}".lower(), base.lower()}
        own, sibling = [], []
        parent = output_file.parent
        for name in _sorted_names(parent):
            if not name.lower().endswith(tail):
                continue
            path = parent / name
            if not path.is_file():
                continue
            dat_stem = name[: -len(tail)]
            if dat_stem == stem:
                own.append(path)
            elif dat_stem.lower() in siblings:
                sibling.append(path)
        for rank in (own, sibling):
            if len(rank) == 1:
                return rank[0], []
            if len(rank) > 1:
                return None, [p.name for p in rank]
        return None, []

    @staticmethod
    def _parent_dos_file(output_file: Path) -> Optional[Path]:
        """The DOSS.DAT that decides metal or semimetal of a gapless BAND run:
        ``_dat_file_for`` (the run's folder), else the DOSS run of the same
        parent in a folder beside the run's (``step_006_DOSS/<base>_doss.DOSS.DAT``
        of a workflow, ``DOSS/`` beside ``BAND/``): exactly one such file."""
        path, ambiguous = CrystalPropertyExtractor._dat_file_for(Path(output_file), "DOSS")
        if path is not None or ambiguous:
            return path
        output_file = Path(output_file)
        base = output_file.stem
        for suffix in ("_band", "_doss"):
            if base.lower().endswith(suffix):
                base = base[: -len(suffix)]
                break
        names = (f"{base}_doss.DOSS.DAT", f"{base}.DOSS.DAT")
        own = output_file.parent
        try:
            dirs = [d for d in own.parent.iterdir() if d.is_dir() and d != own][:2000]
        except OSError:
            return None
        hits = [d / n for d in dirs for n in names if (d / n).is_file()]
        return hits[0] if len(hits) == 1 else None

    def _extract_advanced_band_dos_analysis(self, output_file: Path, properties: Dict[str, Any],
                                            content: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract advanced electronic properties using BAND/DOSS data analysis.
        
        The band gap and, for a gapless band structure, metal or semimetal by
        N(E_F) per atom (the rule and the DOSS.DAT of band_path_gap_type);
        the quick effective-mass estimate real_*_effective_mass.
        """
        advanced_props = {}
        
        try:
            try:
                from mace.utils.advanced_electronic_analyzer import AdvancedElectronicAnalyzer
            except ImportError:
                from advanced_electronic_analyzer import AdvancedElectronicAnalyzer
            
            # This output's own BAND.DAT / DOSS.DAT, matched by stem: a
            # directory can hold other calculations' files.
            band_file, band_ambiguous = self._dat_file_for(output_file, "BAND")
            doss_file, doss_ambiguous = self._dat_file_for(output_file, "DOSS")
            if band_file and not doss_file and not doss_ambiguous:
                doss_file = self._parent_dos_file(output_file)
            notes = [f"ambiguous {kind}.DAT files skipped: {', '.join(names)}"
                     for kind, names in (("BAND", band_ambiguous), ("DOSS", doss_ambiguous))
                     if names]
            if notes:
                advanced_props['advanced_analysis_note'] = '; '.join(notes)
            
            if not band_file and not doss_file:
                return advanced_props  # nothing to analyze
            
            # Perform advanced analysis
            from mace.utils.advanced_electronic_analyzer import atoms_per_cell
            if content is None and properties.get('atoms_in_unit_cell') is None:
                try:
                    content = Path(output_file).read_text(errors='replace')
                except OSError:
                    content = ''
            analyzer = AdvancedElectronicAnalyzer()
            analysis_results = analyzer.analyze_material(band_file, doss_file,
                                                         atoms_per_cell(properties, content))
            
            # metal or semimetal of a gapless band structure, decided by
            # N(E_F) per atom (_classify_electronic_properties; not stored itself)
            if band_file and doss_file and analysis_results.get('classified_by_dos'):
                advanced_props['_band_dos_classification'] = analysis_results['electronic_classification']
            
            # The quick effective-mass estimate (one band along the path at
            # the edge); the accurate m_c / m_d are effective_mass_*_mc / _md
            if analysis_results.get('has_real_effective_mass'):
                if analysis_results.get('electron_effective_mass') is not None:
                    advanced_props['real_electron_effective_mass'] = analysis_results['electron_effective_mass']
                if analysis_results.get('hole_effective_mass') is not None:
                    advanced_props['real_hole_effective_mass'] = analysis_results['hole_effective_mass']
                if analysis_results.get('average_effective_mass') is not None:
                    advanced_props['real_average_effective_mass'] = analysis_results['average_effective_mass']
                
                advanced_props['effective_mass_calculation_method'] = analysis_results.get('calculation_method', 'unknown')
            
            # the metal/semimetal criterion of the DOS analysis
            if analysis_results.get('dos_data_available'):
                if analysis_results.get('dos_threshold') is not None:
                    advanced_props['dos_threshold'] = analysis_results['dos_threshold']
                advanced_props['dos_mean'] = analysis_results.get('dos_mean', 0.0)
            
            # Band structure analysis results
            if analysis_results.get('band_data_available'):
                advanced_props['band_structure_k_points'] = analysis_results.get('band_k_points', 0)
                if 'band_structure_range' in analysis_results:
                    advanced_props['band_structure_k_range_max'] = analysis_results['band_structure_range'][1]
            
            # The band-path gap by occupation when the BAND.DAT gap by band
            # index is unknown (no occupied-band count in the .out)
            if (analysis_results.get('gap_ev') is not None
                    and properties.get('band_fundamental_gap_ev') is None and band_file):
                advanced_props['band_fundamental_gap_ev'] = analysis_results['gap_ev']
            
            # Analysis metadata
            files_analyzed = analysis_results.get('files_analyzed', {})
            if files_analyzed.get('band_file'):
                advanced_props['band_file_analyzed'] = files_analyzed['band_file']
            if files_analyzed.get('doss_file'):
                advanced_props['doss_file_analyzed'] = files_analyzed['doss_file']
                
            print(f"      🚀 Advanced analysis: {analysis_results.get('electronic_classification') or 'no'} classification")
            if analysis_results.get('has_real_effective_mass'):
                print(f"      📊 Effective mass, quick estimate (one band along the path): real_*_effective_mass; "
                      f"the accurate m_c / m_d are effective_mass_*")
            
        except ImportError:
            advanced_props['advanced_analysis_note'] = 'AdvancedElectronicAnalyzer not available'
        except Exception as e:
            advanced_props['advanced_analysis_error'] = str(e)
            print(f"      ⚠️  Advanced analysis failed: {e}")
        
        return advanced_props


def transport_axis_lines(properties: Dict[str, Any]) -> List[str]:
    """What `mace analyze --extract-properties` prints for a TRANSPORT run:
    the peak PF, ZT_e and (with --kappa-lattice) total ZT along each axis,
    the best axis, and the value of the averaged tensors beside them."""
    lines = []
    for quantity, label, fmt in (('power_factor', 'PF (W/(m·K²))', '.4g'),
                                 ('zt_electronic', 'ZTₑ (no lattice κ)', '.4g'),
                                 ('zt_total', 'total ZT', '.4g')):
        per = {a: properties.get(f'transport_{quantity}_max_{a}') for a in ('xx', 'yy', 'zz')}
        per = {a: v for a, v in per.items() if v is not None}
        avg = properties.get(f'transport_{quantity}_max_avg')
        if not per and avg is None:
            continue
        parts = [f"{a} {format(v, fmt)}" for a, v in per.items()]
        best = properties.get(f'transport_{quantity}_best_axis')
        text = f"{label} peak: " + ", ".join(parts)
        if best:
            text += f" (best {best})"
        if avg is not None:
            text += f"; of the averaged tensors {format(avg, fmt)}"
        lines.append(text)
    return lines


def _output_calc_type(output_file: Path) -> str:
    """The calculation type of a finished .out, as ``mace check`` sorts it."""
    from mace.completion_checker import determine_completed_subtype
    with open(output_file, 'r', errors='ignore') as f:
        lines = f.readlines()
    has_opt_end = any("OPT END" in line for line in lines)
    return determine_completed_subtype(Path(output_file), lines, has_opt_end=has_opt_end)


def _output_matches_calc_type(output_file: Path, wanted: str) -> bool:
    """True when ``output_file`` is a ``wanted`` (OPT, SP2, DOSS, ...) run.

    Replaces a file-name substring test ("opt" is in every chained MACE name,
    so --filter-type OPT kept SP, FREQ and properties outputs) followed by a
    phrase test on the first 5 KB ("SINGLE POINT", "OPTGEOM", ... which CRYSTAL
    does not print there), which together dropped every output.
    """
    base = re.sub(r'\d+$', '', wanted.strip().upper())
    if base == 'DOS':
        base = 'DOSS'
    try:
        return _output_calc_type(Path(output_file)) == base
    except OSError:
        return False


def _material_count(db, ids=None) -> int:
    """Rows of the materials table (of ``ids`` only, when given)."""
    with db._get_connection() as conn:
        if ids is None:
            return int(conn.execute("SELECT count(*) FROM materials").fetchone()[0])
        ids = sorted(ids)
        return sum(int(conn.execute(f"SELECT count(*) FROM materials WHERE material_id IN "
                                    f"({','.join('?' * len(ids[k:k + 500]))})", ids[k:k + 500]).fetchone()[0])
                   for k in range(0, len(ids), 500))


def _ensure_material(db, material_id: str, output_file: Path) -> None:
    """A materials row for ``material_id`` when it has none: formula and
    dimensionality from the output (formula_extractor.material_info)."""
    if db.get_material(material_id):
        return
    try:
        from mace.utils.formula_extractor import material_info, extract_space_group_from_output
        d12 = Path(output_file).with_suffix('.d12')
        info = material_info(output_file=output_file, d12_file=d12 if d12.is_file() else None)
        sg = extract_space_group_from_output(output_file) or 1
    except Exception:
        info, sg = {}, 1
    meta = ({"cell_formula": info["cell_formula"], "formula_source": info["source"]}
            if info.get("cell_formula") else None)
    db.create_material(material_id=material_id, formula=info.get("formula") or "Unknown", space_group=sg,
                       dimensionality=info.get("dimensionality") or "CRYSTAL", source_type="output",
                       source_file=str(output_file), metadata=meta)
    print(f"   📝 Created material {material_id} for an output with no calculation record")


def _parse_references(items) -> Dict[str, List[str]]:
    """--reference NAME=FILE|CALC_ID (repeatable; one name may be given
    several times: the runs of a reference with many, such as the elements).
    NAME is a reference a derived property takes (registry.reference_roles)."""
    refs: Dict[str, List[str]] = {}
    roles = None
    for item in items or []:
        name, sep, value = str(item).partition('=')
        if not sep or not name.strip() or not value.strip():
            print(f"❌ --reference {item!r}: NAME=FILE or NAME=CALC_ID", file=sys.stderr)
            sys.exit(2)
        if roles is None:
            try:
                from mace.properties import registry
                roles = registry.reference_roles()
            except Exception:
                roles = {}
        if roles and name.strip() not in roles:
            print(f"❌ --reference {item!r}: no derived property takes a reference named {name.strip()!r}; "
                  f"the names are {', '.join(sorted(roles))}", file=sys.stderr)
            sys.exit(2)
        refs.setdefault(name.strip(), []).append(value.strip())
    return refs


def _parse_skips(names) -> set:
    """--skip-property NAME (repeatable, or comma-separated): registered derived properties."""
    from mace.properties import registry
    known = {p.name for p in registry.properties(None, interactive=False)}
    out = set()
    for item in names or []:
        for name in str(item).split(','):
            name = name.strip()
            if not name:
                continue
            if name not in known:
                print(f"❌ --skip-property {name!r}: not a derived property (mace analyze --list-properties "
                      "--derived lists them)", file=sys.stderr)
                sys.exit(2)
            out.add(name)
    return out


def _json_line(stream, obj) -> None:
    stream.write(json.dumps(obj) + "\n")
    stream.flush()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Extract properties from CRYSTAL output files")
    parser.add_argument("--output-file", help="Single output file to process")
    parser.add_argument("--scan-directory", help="Directory to scan for output files")
    parser.add_argument("--db-path", default="materials.db", help="Path to materials database")
    parser.add_argument("--material-id", help="Override material ID (or filter by material ID when scanning)")
    parser.add_argument("--calc-id", help="Override calculation ID")
    parser.add_argument("--calc-type", help="Filter by calculation type (OPT, SP, FREQ, BAND, DOSS, etc.)")
    parser.add_argument("--filter-material", help="Only extract properties for specific material ID")
    parser.add_argument("--filter-type", help="Only extract properties from specific calculation type")
    parser.add_argument("--fragments", action="store_true",
                        help="Per-fragment Mulliken charges for every run (charged runs get them anyway)")
    parser.add_argument("--reference", action="append", default=[], metavar="NAME=FILE|CALC_ID",
                        help="A reference run a derived property needs (mace/properties: the elements, the "
                             "isolated atoms, a bulk, a clean surface and an adsorbate, the reactants and "
                             "products): its output file or calc_id, N*FILE for a coefficient, auto for the "
                             "database's choice again; repeat for several. Saved with the run when the "
                             "property is stored")
    parser.add_argument("--no-derived", action="store_true",
                        help="Compute no derived property (mace/properties: reference energies, Bader charges, "
                             "...); their stored rows stay. Also MACE_DERIVED=0")
    parser.add_argument("--skip-property", action="append", default=[], metavar="NAME",
                        help="Do not compute this derived property (repeatable); its stored rows stay. Also "
                             "MACE_SKIP_PROPERTIES=NAME,NAME")
    # Building a database from a folder (all opt-in; without them a run is unchanged)
    # On by default for a folder (--scan-directory): without the calculation
    # records, `mace database` lists none of the materials and an export is
    # empty. --no-register keeps the untracked extraction.
    parser.add_argument("--register", dest="register", action="store_true", default=None,
                        help="First record the folder's completed calculations (materials and calculations "
                             "rows); the default with --scan-directory")
    parser.add_argument("--no-register", dest="register", action="store_false",
                        help="Do not record calculations: store the properties of untracked outputs only")
    parser.add_argument("--register-output-only", action="store_true",
                        help="With --output-file --register: record that output only, not the other "
                             "completed outputs of its folder and the folders below it")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write nothing: list each output as new, changed, unchanged, not tracked, "
                             "unreadable or incomplete")
    parser.add_argument("--json", action="store_true", help="With --dry-run: one JSON document")
    parser.add_argument("--skip-unchanged", action="store_true",
                        help="Extract only new and changed outputs of tracked calculations")
    parser.add_argument("--backup", action="store_true",
                        help="Back up the database into <db dir>/backups before the first write")
    parser.add_argument("--keep-backups", type=int, default=None,
                        help="With --backup: keep the newest N backups (1-100, default 3)")
    parser.add_argument("--progress-json", action="store_true",
                        help="Machine-readable progress: one JSON line per file on stdout, the rest on stderr")
    # Total ZT of TRANSPORT runs (opt-in; without it only the electronic ZT_e is stored)
    parser.add_argument("--kappa-lattice", metavar="VALUE|FILE",
                        help="Lattice thermal conductivity in W/(m·K) for the total ZT = S²σT/(κe + κL) of "
                             "TRANSPORT runs: one value, or a CSV of T,kappa_L rows (linear interpolation, "
                             "no extrapolation)")
    parser.add_argument("--tau-fs", metavar="FS|TAU.json",
                        help="With --kappa-lattice: the relaxation time in fs to evaluate the total ZT at "
                             "(default: the run's RELAXTIM; σ and κ0 are scaled by tau/tau_run), or the "
                             "JSON a mace tau ... --save wrote (its method is recorded as the tau source)")
    
    args = parser.parse_args()
    
    if not args.output_file and not args.scan_directory:
        print("❌ Please specify either --output-file or --scan-directory")
        sys.exit(1)
    if args.register is None:
        args.register = bool(args.scan_directory) and not args.output_file
    if args.register_output_only and not (args.output_file and args.register):
        print("❌ --register-output-only goes with --output-file and --register", file=sys.stderr)
        sys.exit(2)
    if args.keep_backups is not None and not args.backup:
        print("❌ --keep-backups needs --backup", file=sys.stderr)
        sys.exit(2)
    keep = 3 if args.keep_backups is None else args.keep_backups
    if not 1 <= keep <= 100:
        print("❌ --keep-backups is a number from 1 to 100", file=sys.stderr)
        sys.exit(2)
    if args.json and not args.dry_run:
        print("❌ --json goes with --dry-run (use --progress-json for a run)", file=sys.stderr)
        sys.exit(2)
    kappa_lattice = tau_fs = None
    if args.tau_fs is not None and args.kappa_lattice is None:
        print("❌ --tau-fs goes with --kappa-lattice (the electronic ZT does not depend on tau)", file=sys.stderr)
        sys.exit(2)
    if args.kappa_lattice is not None:
        from mace.utils import transport_zt
        try:
            kappa_lattice = transport_zt.parse_kappa_lattice(args.kappa_lattice)
            tau_fs = transport_zt.check_tau_fs(args.tau_fs)
        except transport_zt.KappaLatticeError as e:
            print(f"❌ --kappa-lattice/--tau-fs: {e}", file=sys.stderr)
            sys.exit(2)
    args.kappa_lattice_parsed, args.tau_fs_parsed = kappa_lattice, tau_fs
    if args.skip_unchanged and not args.register and not db_build.has_tracked(args.db_path):
        print("❌ --skip-unchanged needs tracked calculations; add --register", file=sys.stderr)
        sys.exit(2)

    # JSON on stdout (--progress-json, --dry-run --json): everything else,
    # including what the helpers print, goes to stderr
    import contextlib
    machine = args.progress_json or (args.dry_run and args.json)
    real_stdout = sys.stdout
    with (contextlib.redirect_stdout(sys.stderr) if machine else contextlib.nullcontext()):
        _run(args, keep, real_stdout)


def _run(args, keep: int, real_stdout) -> None:
    db_dir = Path(args.db_path).resolve().parent
    building = args.register or args.backup or args.skip_unchanged or args.dry_run or args.progress_json
    # structures.db beside materials.db (not in the current directory) when
    # building from a folder; a plain run keeps MaterialDatabase's default
    ase_db = str(db_dir / "structures.db") if building else None
    output_files = []
    seen_outputs = []  # every output the run could see, extracted or filtered out
    if args.output_file:
        output_files = [Path(args.output_file)]
        seen_outputs = list(output_files[0].parent.glob("*.out"))
    elif args.scan_directory:
        scan_dir = Path(args.scan_directory)
        output_files = list(scan_dir.rglob("*.out"))
        seen_outputs = list(output_files)
        
        # Apply filters if specified
        if args.filter_material or args.filter_type:
            filtered_files = []
            for f in output_files:
                # Check material ID filter
                if args.filter_material:
                    if args.filter_material not in str(f):
                        continue
                
                # Check calculation type filter
                if args.filter_type and not _output_matches_calc_type(f, args.filter_type):
                    continue
                
                filtered_files.append(f)
            
            print(f"🔍 Found {len(output_files)} output files, {len(filtered_files)} match filters")
            output_files = filtered_files
        else:
            print(f"🔍 Found {len(output_files)} output files to process")
    else:
        print(f"🔍 Found {len(output_files)} output files to process")

    scan_root = Path(args.scan_directory) if args.scan_directory else Path(args.output_file).parent

    if args.dry_run:
        rows = db_build.classify(output_files, args.db_path, register=args.register, scan_dir=scan_root)
        doc = {"db": str(Path(args.db_path).resolve()), "db_exists": Path(args.db_path).is_file(),
               "register": args.register, "files": rows, "counts": db_build.counts(rows)}
        if args.json:
            _json_line(real_stdout, doc)
        else:
            where = "updates" if doc["db_exists"] else "would create"
            print(f"Dry run ({where} {doc['db']}); nothing is written.")
            print(db_build.table(rows))
        return

    # --backup: before the first write; a backup that fails stops the run
    backup_made = None
    if args.backup:
        db_file = Path(args.db_path)
        if db_file.is_file():
            try:
                from mace.database import dbsafety
            except ImportError:
                from database import dbsafety
            try:
                b = dbsafety.snapshot_backup(db_file, db_dir / "backups", "materials_backup")
                print(f"💾 Backup: {b}")
                if (db_dir / "structures.db").is_file():
                    sb = dbsafety.snapshot_backup(db_dir / "structures.db", db_dir / "backups", "structures_backup")
                    print(f"💾 Backup: {sb}")
                pruned = dbsafety.prune_backups(db_dir / "backups", keep=keep)
                backup_made = str(b)
                if args.progress_json:
                    _json_line(real_stdout, {"backup": backup_made, "pruned": pruned})
            except Exception as e:
                print(f"❌ The backup failed ({e}); nothing was written.", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"No database at {db_file} yet: nothing to back up.")

    extractor = CrystalPropertyExtractor(args.db_path, ase_db_path=ase_db)
    extractor.extract_fragments = args.fragments
    extractor.references = _parse_references(args.reference)
    extractor.run_derived = not args.no_derived
    extractor.skip_derived = frozenset(_parse_skips(args.skip_property))
    # a folder: the properties that read other runs are computed once every
    # run's own rows are stored (the second pass below)
    extractor.defer_references = bool(args.scan_directory)
    extractor.kappa_lattice = args.kappa_lattice_parsed
    extractor.tau_fs = args.tau_fs_parsed
    if extractor.kappa_lattice is not None:
        print(f"κL for the total ZT of TRANSPORT runs: {extractor.kappa_lattice.source()}"
              + (f"; tau {extractor.tau_fs:g} fs" if extractor.tau_fs is not None else "; tau: each run's RELAXTIM"))

    materials_before = _material_count(extractor.db) if building else None
    if args.register:
        from mace.database.populate_completed_jobs import scan_for_completed_calculations, populate_database
        # absolute paths, as `mace submit` records them: a path relative to the
        # shell's folder is found neither from the database's folder nor by the GUI
        only = [Path(args.output_file).resolve()] if getattr(args, "register_output_only", False) else None
        calcs = scan_for_completed_calculations(scan_root.resolve(), files=only)
        added = populate_database(calcs, extractor.db) if calcs else 0
        print(f"📋 Registered {added} new calculation(s) from {only[0].name if only else scan_root}")

    classes = {}
    reasons = {}
    if building:
        for r in db_build.classify(output_files, args.db_path, register=False, scan_dir=scan_root,
                                   registering=bool(args.register)):
            classes[str(r["file"])] = r["class"]
            if r.get("reason"):
                reasons[str(r["file"])] = r["reason"]
    # a build that stopped before its end (a CPU-time limit, a kill, a cancel) left its marker: with
    # --skip-unchanged this one continues it (what it committed is "unchanged" now) and says so
    resumed = None
    if building:
        previous = db_build.begin_build(args.db_path, scan_root, len(output_files))
        if previous is not None:
            if args.skip_unchanged:
                already = sum(1 for f in output_files if classes.get(str(f)) == "unchanged")
                resumed = {"already": already, "total": len(output_files)}
                print(f"↻ {db_build.resumed_message(already, len(output_files))} "
                      "(the last update of this database stopped before its end)")
                if args.progress_json:
                    _json_line(real_stdout, {"resumed": resumed})
            else:
                print("ℹ️  The last update of this database stopped before its end; this run extracts every "
                      "output again (--skip-unchanged continues where it stopped).")
    records = []   # one per file, as --progress-json writes them: the summary counts these
    
    total_properties = 0
    files_processed = 0
    materials_processed = set()
    # material -> the TRANSPORT outputs given the one --kappa-lattice's total ZT
    zt_materials: Dict[str, List[str]] = {}
    # Untracked outputs (no calculation record) share one set of rows per
    # material, so a row cannot name the output that wrote it. Rows of such
    # a material that this run did not write again are removed after the
    # run, when it re-extracted every output of the material it could see
    # (the scanned tree, or the directory of a single --output-file). Not
    # with --material-id: outputs given one material by hand can be anywhere.
    run_started = datetime.now().isoformat()
    untracked = {}
    n_files = len(output_files)
    summary = {"extracted": 0, "skipped": 0, "failed": 0, "by_class": {}}
    
    # the tracked runs whose reference properties the second pass computes:
    # every run of the folder, an unchanged one (--skip-unchanged) too
    second_pass = []
    for i, output_file in enumerate(output_files, start=1):
        cls = classes.get(str(output_file))
        if cls:
            summary["by_class"][cls] = summary["by_class"].get(cls, 0) + 1
        if args.skip_unchanged and cls == "unchanged" and extractor.defer_references:
            second_pass.append((output_file, extractor._find_calc_id_for_output(output_file)))
        if (args.skip_unchanged and cls == "unchanged") or cls == db_build.SKIPPED:
            # a build files nothing for a failed run no calculation records,
            # or for the _temp copy of a run's output: their rows had no material
            summary["skipped"] += 1
            rec = {"i": i, "n": n_files, "file": str(output_file), "status": "skipped",
                   "properties": 0, "class": cls}
            if reasons.get(str(output_file)):
                rec["reason"] = reasons[str(output_file)]
            records.append(rec)
            if cls == db_build.SKIPPED:
                # db_build's reason (e.g. "a copy of X.out")
                print(f"\n⏭️  Skipping {output_file}: "
                      f"{reasons.get(str(output_file)) or 'not a registered calculation output'}")
            if args.progress_json:
                _json_line(real_stdout, rec)
            continue
        print(f"\n📊 Processing: {output_file}")
        
        # --calc-type filters like --filter-type (a scanned directory was
        # already filtered by --filter-type above)
        wanted = args.calc_type or (args.filter_type if args.output_file else None)
        if wanted and not _output_matches_calc_type(output_file, wanted):
            print(f"   ⏭️  Skipping - not a {wanted.upper()} calculation")
            summary["skipped"] += 1
            rec = {"i": i, "n": n_files, "file": str(output_file), "status": "skipped",
                   "properties": 0, "class": cls}
            records.append(rec)
            if args.progress_json:
                _json_line(real_stdout, rec)
            continue
        
        properties = extractor.extract_all_properties(
            output_file, 
            material_id=args.material_id,
            calc_id=args.calc_id
        )
        status, saved_count = "no properties", 0
        meta = (properties or {}).get('_metadata') or {}
        if building and properties and meta.get('calc_id') is None and meta.get('material_id'):
            # an untracked output (Register off): its rows need a material
            # the Database view can open (no calculation row is written)
            _ensure_material(extractor.db, meta['material_id'], Path(output_file))
        
        if properties:
            # Extract material ID from properties if available
            material = (properties.get('_metadata') or {}).get('material_id')
            if material:
                materials_processed.add(material)
            
            saved_count = extractor.save_properties_to_database(properties)
            total_properties += saved_count
            files_processed += 1
            if saved_count and 'transport_zt_total_max_avg' in properties:
                zt_materials.setdefault(material or Path(output_file).stem, []).append(Path(output_file).name)
            meta = properties.get('_metadata', {})
            if extractor.defer_references and meta.get('calc_id') and saved_count:
                second_pass.append((output_file, meta['calc_id']))
            if meta.get('calc_id') is None and saved_count and not args.material_id:
                untracked.setdefault(meta['material_id'], set()).add(Path(output_file).resolve())
            if building and meta.get('calc_id') and saved_count:
                try:
                    db_build.record_extracted(extractor.db, meta['calc_id'], Path(output_file))
                except Exception as e:
                    print(f"   ⚠️  Could not record the extraction of {output_file}: {e}")
            status = "extracted" if saved_count else "failed"
            print(f"   ✅ Extracted and saved {saved_count} properties")
            for line in transport_axis_lines(properties):
                print(f"   {line}")
            if properties.get("fragment_charges"):
                from mace.utils.ion_fragments import format_fragment_charges
                for line in format_fragment_charges(properties):
                    print(f"   {line}")
        else:
            print(f"   ⚠️  No properties extracted")
        summary["extracted" if status == "extracted" else "failed"] += 1
        rec = {"i": i, "n": n_files, "file": str(output_file), "status": status,
               "properties": saved_count, "class": cls, "material": meta.get('material_id')}
        records.append(rec)
        if args.progress_json:
            _json_line(real_stdout, rec)
    
    # Second pass: the derived properties that read other runs (reference
    # energies), now that every run's own rows are stored: the same values
    # whatever order the folder was read in, and a lookup whose answer
    # changed (a new or better reference run) is updated
    if second_pass:
        print(f"\n🔗 Reference properties of {len(second_pass)} run(s) (they read the other runs)")
        done2 = extractor.refresh_reference_properties(second_pass)
        print(f"   {done2['runs']} run(s) computed, {done2['changed']} changed"
              + (f", {done2['errors']} error(s) (their stored values kept)" if done2['errors'] else ""))

    # The untracked clean-up needs every output of the material re-extracted:
    # with --skip-unchanged the extracted set is not complete
    for material_id, extracted in ({} if args.skip_unchanged else untracked).items():
        siblings = {f.resolve() for f in seen_outputs
                    if extractor._extract_material_id_from_filename(f) == material_id}
        if not siblings <= extracted:
            continue  # an output of the material was left out: its rows may be among them
        removed = extractor.remove_stale_untracked_rows(material_id, run_started)
        if removed:
            print(f"   🧹 {material_id}: removed {removed} stale properties an earlier "
                  f"extraction stored and this one no longer produces")

    if len(zt_materials) > 1:
        # κL is a property of the material: one value or table for several
        # materials is rarely what the user meant (the source is stored, but
        # nothing else would say so)
        print(f"\n⚠️  One κL ({extractor.kappa_lattice.source()}) was used for the total ZT of "
              f"{len(zt_materials)} materials: " + ", ".join(sorted(zt_materials))
              + ". κL is a property of each material; extract each one with its own --kappa-lattice "
              "unless they share it.")

    print(f"\n🎉 Processing complete!")
    print(f"   Files processed: {files_processed}")
    print(f"   Total properties extracted: {total_properties}")
    print(f"   Materials processed: {len(materials_processed)}")
    n_untracked = sum(len(v) for v in untracked.values())
    if n_untracked:
        # (--no-register, or outputs the registration does not record)
        print(f"   ⚠️  {n_untracked} output(s) have no calculation record: they share one set of rows "
              "per material (a later output's value replaces an earlier one)"
              + ("" if building else ", and `mace database` lists no material for them")
              + ". Run with --register to record them as calculations.")
    
    if args.filter_material or args.filter_type or args.calc_type:
        print(f"\n📌 Filters applied:")
        if args.filter_material:
            print(f"   Material ID: {args.filter_material}")
        if args.filter_type or args.calc_type:
            print(f"   Calculation type: {args.filter_type or args.calc_type}")
    if building:
        counted = db_build.summarize(records)
        written = {r["material"] for r in records if r.get("status") == "extracted" and r.get("material")}
        done = {"done": True, "files": n_files, "files_processed": files_processed,
                "properties": total_properties, "materials": _material_count(extractor.db, written),
                "materials_new": max(0, _material_count(extractor.db) - materials_before),
                "backup": backup_made, **summary,
                "new": counted["new"], "changed": counted["changed"],
                "reextracted": counted["reextracted"],
                "failed_outputs": len(counted["failed"]), "skipped_outputs": counted["skipped"]}
        if resumed is not None:
            done["resumed"] = resumed
        done["message"] = db_build.summary_message(done)
        db_build.end_build(args.db_path)
        print(f"   {done['message']}")
        if args.progress_json:
            _json_line(real_stdout, done)


if __name__ == "__main__":
    main()