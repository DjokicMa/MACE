# MACE - Mendoza Automated CRYSTAL Engine

<p align="center">
  <pre>
      ███╗   ███╗ █████╗  ██████╗███████╗
      ████╗ ████║██╔══██╗██╔════╝██╔════╝
      ██╔████╔██║███████║██║     █████╗  
      ██║╚██╔╝██║██╔══██║██║     ██╔══╝  
      ██║ ╚═╝ ██║██║  ██║╚██████╗███████╗
      ╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝╚══════╝
  </pre>
</p>

<p align="center">
  <strong>Comprehensive automation toolkit for CRYSTAL quantum chemistry workflows</strong>
</p>

---

## Overview

MACE (Mendoza Automated CRYSTAL Engine) is a powerful automation framework designed to streamline quantum chemistry calculations using the CRYSTAL software package. It provides end-to-end workflow management, from structure preparation to property analysis, with robust error handling and HPC integration.

**Developed by**: Marcus Djokic (Primary Developer)  
**Contributors**: Daniel Maldonado Lopez, Brandon Lewis, William Comaskey  
**PI**: Prof. Jose Luis Mendoza-Cortes  
**Institution**: Michigan State University, Mendoza Group

## Key Features

### 🚀 Workflow Automation
- **Interactive workflow planning** with customizable calculation sequences
- **Automated progression** from geometry optimization to property calculations
- **Material tracking database** for comprehensive calculation history
- **Error recovery** with configurable retry strategies

### 💎 CRYSTAL Integration
- **CIF to D12 conversion** with full customization options
- **Property calculations** including band structures, DOS, transport, and charge analysis
- **Phonon calculations** with automated band structure plotting
- **Basis set management** for different calculation types

### 🖥️ HPC Optimization
- **SLURM integration** with intelligent resource allocation
- **Queue management** with concurrent job limiting
- **Scratch space optimization** for large calculations
- **Multi-node parallelization** support

### 📊 Analysis Tools
- **Automated property extraction** from output files
- **Publication-quality plots** for band structures and DOS
- **Comprehensive CSV reports** for material properties
- **Electronic structure classification** (metal/semiconductor/insulator)

## Quick Start

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd reorganization

# Run interactive setup
python setup_mace.py --add-to-path

# Apply configuration
source ~/.zshrc  # or ~/.bashrc
```

See [INSTALLATION.md](../INSTALLATION.md) for detailed instructions.

### Basic Usage

#### Getting Help
```bash
# General help
mace --help

# Command-specific help (shows MACE wrapper options)
mace workflow --help
mace submit --help
mace monitor --help
mace analyze --help

# For conversion/generation commands, these pass through to the underlying scripts
mace convert --help    # Shows NewCifToD12.py help
mace opt2d12 --help   # Shows CRYSTALOptToD12.py help
mace opt2d3 --help    # Shows CRYSTALOptToD3.py help
```

#### 1. Interactive Workflow Planning
```bash
mace workflow --interactive
# Or with full path:
python ../mace_cli workflow --interactive
```
This launches an interactive session to:
- Select input files (CIFs or existing D12s)
- Choose calculation workflow (e.g., OPT → SP → BAND → DOS)
- Configure SLURM resources
- Set up material tracking

#### 2. Quick Workflow Execution
```bash
# Process CIF files with full electronic structure workflow
mace workflow --quick-start --cif-dir ./cifs --workflow full_electronic

# Run optimization only
mace workflow --quick-start --d12-dir ./d12s --workflow basic_opt
```

#### 3. Submit Individual Calculations
```bash
# Submit CRYSTAL calculations
mace submit calculation.d12
mace submit property.d3

# Or use direct scripts (if in PATH)
submitcrystal23.sh calculation.d12
submit_prop.sh property.d3
```

#### 4. Monitor Progress
```bash
# Real-time monitoring dashboard
mace monitor --dashboard

# Check queue status
mace monitor --status

# View material properties
mace database --action properties --material-id diamond

# NEW: Filter materials by properties
mace database --action query --filter "band_gap > 3.0"
mace database --action query --filter "total_energy < -1000" --filter "band_gap > 2" --logic AND
```

## Workflow Templates

MACE includes predefined workflow templates:

- **basic_opt**: Geometry optimization only
- **opt_sp**: OPT → Single point calculation
- **full_electronic**: OPT → SP → BAND → DOS (recommended)
- **transport_analysis**: OPT → SP → TRANSPORT
- **charge_analysis**: OPT → SP → CHARGE+POTENTIAL
- **complete**: OPT → SP → BAND → DOS → FREQ

## Database Query and Analysis Features

### Property-Based Material Filtering

MACE now includes powerful property-based filtering to find materials matching specific criteria:

#### Basic Usage
```bash
# Find materials with large band gaps
mace database --action query --filter "band_gap > 3.0"

# Find materials with low total energy
mace database --action query --filter "total_energy < -1000"

# Combine multiple filters with AND logic (default)
mace database --action query --filter "band_gap > 2" --filter "band_gap < 5"

# Use OR logic for filters
mace database --action query --filter "band_gap > 5.0" --filter "total_energy < -1000" --logic OR
```

#### Supported Operators
- `>`, `>=`, `<`, `<=` - Numeric comparisons
- `==`, `!=` - Equality comparisons (works for both numbers and strings)
- `=` - Alias for `==`

#### Filter Examples
```bash
# Materials with specific space group
mace database --action query --filter "space_group == 227"

# Exclude specific formulas
mace database --action query --filter "formula != C"

# Complex queries
mace database --action query --filter "band_gap > 2.0" --filter "atoms_in_unit_cell < 10" --filter "total_energy < -500"
```

#### Advanced SQL-like Filtering
MACE now supports advanced SQL-like syntax for complex queries using parentheses, logical operators, and special functions:

```bash
# Parentheses and logical operators
mace database --action query --filter "(band_gap > 3 AND space_group = 227) OR total_energy < -1000"

# LIKE operator for pattern matching
mace database --action query --filter "formula LIKE 'C%'"           # Formulas starting with C
mace database --action query --filter "formula LIKE '%O%'"          # Formulas containing O
mace database --action query --filter "formula NOT LIKE '%H%'"      # Formulas without H

# IN operator for value lists
mace database --action query --filter "space_group IN (225, 227, 229)"
mace database --action query --filter "conductivity_type IN ('insulator', 'semiconductor')"

# IS NULL/IS NOT NULL checks
mace database --action query --filter "transport_seebeck_300k IS NOT NULL"
mace database --action query --filter "band_gap IS NULL"

# Complex combinations
mace database --action query --filter "(formula LIKE 'C%' AND band_gap > 2) OR (space_group IN (227, 229) AND total_energy < -500)"
mace database --action query --filter "(band_gap > 2 AND band_gap < 5) AND formula NOT LIKE '%O%'"
```

Note: Advanced filtering is automatically detected when using parentheses, AND/OR keywords, or SQL-style operators (=, LIKE, IN, IS).
Regular filtering with --logic OR/AND is still supported for simple queries.

```

### Property Statistics and Visualization

#### View All Properties
```bash
# Show property statistics across all materials
mace database --action properties

# Filter by category
mace database --action properties --category electronic
mace database --action properties --category structural

# Compact view
mace database --action properties --compact

# Show distribution data
mace database --action properties --json
```

#### Material-Specific Properties
```bash
# View all properties for a material
mace database --action properties --material-id diamond

# Filter by category
mace database --action properties --material-id diamond --category electronic

# Filter by calculation type
mace database --action properties --material-id diamond --from-calc OPT
```

### Available Property Categories
- **structural** - Atomic positions, density, cell volume
- **lattice** - Lattice parameters (a, b, c, α, β, γ)
- **electronic** - Band gaps, energies, electronic structure
- **electronic_classification** - Conductivity type, magnetic properties
- **optimization** - Convergence info, gradients, cycles
- **computational** - CPU time, calculation settings
- **thermodynamic** - Free energies, entropy, heat capacity
- **vibrational** - Frequencies, phonon properties

### Multi-Format Data Export

MACE supports exporting materials and properties data in multiple formats:

#### Export Formats
- **CSV** - Standard comma-separated values
- **JSON** - Structured data with metadata
- **Excel** - Formatted .xlsx with multiple sheets
- **LaTeX** - Publication-ready tables
- **HTML** - Interactive web tables

#### Basic Export
```bash
# Export all materials to Excel
mace database --action export --format excel --output materials.xlsx

# Export to JSON with metadata
mace database --action export --format json --output data.json

# Export filtered materials
mace database --action export --format csv --filter "band_gap > 3" --output high_gap.csv
```

#### Advanced Export Options
```bash
# Export only properties data
mace database --action export --properties-only --format excel

# Include specific properties with materials
mace database --action export --include-property band_gap --include-property total_energy

# Include structure data (normally excluded)
mace database --action export --include-structures --format json

# Combine filters with export
mace database --action export --format latex --filter "space_group == 227" --filter "band_gap > 2"
```

#### Export Features
- **Automatic formatting** - Excel exports include styled headers and auto-sized columns
- **Metadata inclusion** - JSON and Excel formats include export metadata
- **Publication-ready** - LaTeX format generates tables ready for inclusion in papers
- **Web-friendly** - HTML format includes CSS styling and hover effects
- **Smart defaults** - Auto-generated filenames with timestamps
- **Filter support** - Export only materials matching property criteria

### Material Property Comparison

Compare properties across multiple materials to identify trends and relationships:

#### Basic Comparison
```bash
# Compare specific materials
mace database --action compare --materials "1_dia,2_dia2,3_dia3"

# Compare specific properties only
mace database --action compare --materials "1_dia,3_dia3" --properties "band_gap,total_energy"

# Output as JSON for further analysis
mace database --action compare --materials "1_dia,2_dia2" --output-format json
```

#### Comparison Features
- **Side-by-side display** - Properties shown in easy-to-read table format
- **Automatic statistics** - Min/max values and relative differences calculated
- **Common property detection** - Identifies properties present in all materials
- **Largest differences** - Highlights properties with biggest variations
- **Formula and space group** - Material metadata included in comparison
- **Export formats** - Table (default), JSON, or raw dictionary

#### Example Output
```
Property                  | mat_1        | mat_2       | mat_3      
-------------------------------------------------------------------
Formula                   | C            | C2          | C3         
Space Group               | 227          | 227         | 225        
-------------------------------------------------------------------
band_gap                  | 6.0010 eV    | 6.8348 eV   | 5.2341 eV  
total_energy              | -76.21 Ha    | -152.43 Ha  | -228.64 Ha 

=== Summary ===
Common properties: 2

Largest relative differences:
  total_energy: 100.00% difference
    Min: -228.64 (mat_3)
    Max: -76.21 (mat_1)
```

### Missing Data Analysis

Identify missing properties across materials to guide future calculations:

#### Basic Usage
```bash
# Analyze all materials for missing properties
mace database --action missing

# Analyze specific materials
mace database --action missing --material-ids "1_dia,2_dia2,3_dia3"

# Check for specific properties
mace database --action missing --target-properties "band_gap,fermi_energy,phonon_frequencies"

# Get detailed report
mace database --action missing --detail-level detailed

# Export as JSON for programmatic analysis
mace database --action missing --output-format json
```

#### Analysis Features
- **Property completeness scoring** - Percentage of expected properties present
- **Calculation coverage** - Which calculation types have been run
- **Smart recommendations** - Suggests calculations to obtain missing properties
- **Property dependencies** - Understands which calculations produce which properties
- **Batch analysis** - Process entire material database or subsets

#### Report Levels
- **summary** - Overview statistics and key findings
- **detailed** - Includes top recommendations and property coverage
- **full** - Complete material-by-material breakdown

#### Example Output
```
=== Missing Data Analysis Report ===
Materials analyzed: 50
Average completeness: 67.3%

=== Key Findings ===

[HIGH] Most commonly missing properties across 50 materials:
  - transport_seebeck_300k: missing in 45 materials (90.0%)
  - phonon_frequencies: missing in 40 materials (80.0%)
  - band_n_kpoints: missing in 35 materials (70.0%)

[MEDIUM] Materials with less than 50% property completeness (5 found):
  - mat_123: 25.0% complete
  - mat_456: 30.0% complete

=== Calculation Coverage ===
OPT            100.0% (50/50)
SP              80.0% (40/50)
BAND            30.0% (15/50)
TRANSPORT       10.0% (5/50)
```

### Property Correlation Analysis

Discover relationships between material properties through statistical correlation analysis:

#### Basic Usage
```bash
# Analyze all property correlations
mace database --action correlate

# Analyze specific property pairs
mace database --action correlate --properties "band_gap,total_energy;density,band_gap"

# Set minimum sample requirement
mace database --action correlate --min-samples 5

# Get results in different formats
mace database --action correlate --output-format json
mace database --action correlate --top-n 50  # Show top 50 correlations
```

#### Features
- **Pearson correlation coefficient** - Measures linear relationships (-1 to 1)
- **R-squared values** - Explains variance in the relationship
- **Linear regression** - Provides slope and intercept for predictions
- **Automatic property pairing** - Analyzes all numeric property combinations
- **Strong correlation detection** - Highlights |r| > 0.7 relationships

#### Example Output
```
=== Property Correlation Analysis ===
Materials analyzed: 50
Properties analyzed: 25
Property pairs analyzed: 300

=== Strong Correlations (|r| > 0.7) ===
Found 5 strong correlations:
  band_gap vs conductivity: r = -0.856 (R² = 0.733, n = 45)
  density vs bulk_modulus: r = 0.812 (R² = 0.659, n = 38)

Strongest positive correlation:
  a_lattice vs c_lattice: r = 0.923 (R² = 0.852)
Strongest negative correlation:
  band_gap vs conductivity: r = -0.856 (R² = 0.733)
```

### Property Distribution Analysis

Analyze property distributions to understand data spread, identify outliers, and visualize histograms:

#### Basic Usage
```bash
# Analyze all property distributions
mace database --action distribution

# Analyze specific properties
mace database --action distribution --properties "band_gap,density,total_energy"

# Customize histogram bins
mace database --action distribution --bins 20

# Get results in different formats
mace database --action distribution --output-format json
mace database --action distribution --top-n 20  # Show top 20 properties
```

#### Features
- **Statistical analysis** - Min, max, mean, median, standard deviation
- **Histogram generation** - Visualize data distribution with customizable bins
- **Percentile calculation** - 25th, 50th, 75th, 90th, 95th, 99th percentiles
- **Outlier detection** - Identifies values beyond 1.5 * IQR
- **Categorical analysis** - Mode, frequency counts for non-numeric properties

#### Example Output
```
=== Property Distribution Analysis ===
Properties analyzed: 50
Materials included: 200

=== Numeric Properties ===
Property                Count    Min         Max         Mean       Std Dev
---------------------------------------------------------------------------
band_gap                  185    0.000       8.521       3.245       2.134
  Histogram: ▂▅█▇▆▃▂▁
  Outliers: 3 values beyond [0.234, 7.891]
density                   200    1.234      12.456       5.678       2.345
  Histogram: ▁▃▆█▇▅▂▁
total_energy             200   -5432.1     -123.4    -2345.6     1234.5
  Histogram: ▁▂▄▇█▆▃▁

=== Categorical Properties ===
Property                Count  Unique    Mode              Frequency
--------------------------------------------------------------------
crystal_system            200       7    cubic                45.0%
  Values: cubic(90), hexagonal(40), tetragonal(30), ...
conductivity_type         185       3    insulator            65.0%
  Values: insulator(120), semiconductor(50), metal(15)
```

### Workflow Progress Tracking

Track the completion status of multi-step workflows (OPT→SP→BAND→DOS) to monitor progress and identify bottlenecks:

#### Basic Usage
```bash
# Track default workflow progress for all materials
mace database --action workflow

# Track specific workflow
mace database --action workflow --workflow full_electronic

# Track specific materials
mace database --action workflow --material-ids "1_dia,2_dia2,3_dia3" --workflow opt_sp

# Use custom calculation sequence
mace database --action workflow --sequence "OPT,SP,TRANSPORT"

# Show detailed per-material progress
mace database --action workflow --workflow full_electronic --detailed

# Export progress data
mace database --action workflow --output-format csv > progress.csv
```

#### Predefined Workflows
- **basic_opt**: OPT only
- **opt_sp**: OPT → SP
- **full_electronic**: OPT → SP → BAND → DOSS
- **transport_analysis**: OPT → SP → TRANSPORT
- **charge_analysis**: OPT → SP → CHARGE+POTENTIAL
- **combined_analysis**: OPT → SP → BAND → DOSS → TRANSPORT
- **complete**: OPT → SP → BAND → DOSS → FREQ

#### Features
- **Progress visualization** - Histogram showing completion distribution
- **Step-by-step tracking** - Individual calculation status and runtime
- **Bottleneck identification** - Find slowest and most error-prone steps
- **Workflow insights** - Automated detection of stuck calculations
- **Dependency tracking** - Shows blocked calculations

#### Example Output
```
=== Workflow Progress Report ===
Workflow: full_electronic
Sequence: OPT → SP → BAND → DOSS

=== Summary ===
Total materials: 50
Completed: 15 (30.0%)
In progress: 20
Not started: 10
Failed: 5
Average completion: 47.5%

=== Progress Distribution ===
  0- 25%: ████████████ 12
 25- 50%: ████████████████████ 20
 50- 75%: ████████ 8
 75-100%: ██████████ 10

=== Insights ===
[PERFORMANCE] Slowest step: BAND (avg 24.5 hours)
[RELIABILITY] Most failures: DOSS (8 failures)
[WARNING] 3 materials have jobs running > 7 days
```

### Unit Conversion System

Convert property values between different units commonly used in materials science:

#### Basic Usage
```bash
# Show all available units for properties
mace database --action properties --all-units

# Convert units during property display
mace database --action properties --material-id 1_dia --units "band_gap:eV,total_energy:kcal/mol"

# Multiple conversions
mace database --action properties --units "a_lattice:angstrom,pressure:gpa,phonon_frequencies:cm^-1"
```

#### Supported Unit Types
- **Energy**: hartree, eV, kcal/mol, kJ/mol, cm⁻¹, meV, THz
- **Length**: bohr, angstrom, nm, pm
- **Pressure**: hartree/bohr³, GPa, mbar, kbar, Pa, atm
- **Temperature**: K, °C, °F
- **Frequency**: THz, cm⁻¹, meV, GHz, MHz
- **Angle**: rad, degree

#### Example
```bash
# Original display (stored in Hartree)
Total Energy.......................... -76.213456 hartree

# With unit conversion
mace database --action properties --units "total_energy:eV"
Total Energy.......................... -2074.123456 eV
```

## Comprehensive Directory Structure

### Overview
```
mace/
├── run_mace.py               # Main workflow manager interface
├── run_workflow.py           # Primary workflow execution script
├── enhanced_queue_manager.py # Enhanced SLURM queue management
├── material_monitor.py       # Real-time monitoring dashboard
│
├── config/                   # Configuration management
├── database/                 # Material tracking database
├── queue/                    # Job queue management
├── recovery/                 # Error detection and recovery
├── submission/               # Job submission scripts
├── utils/                    # Utility functions and tools
└── workflow/                 # Workflow planning and execution
```

### Detailed Component Breakdown

#### **Root Level Scripts**

- **`run_mace.py`** - Main entry point for MACE workflow system
  - Interactive workflow planning mode
  - Quick-start templates for common workflows
  - Workflow execution and monitoring
  - Example: `python run_mace.py --interactive`

- **`run_workflow.py`** - Direct workflow execution wrapper
  - Simplified interface to run_mace.py
  - Used by mace_cli for workflow command

- **`enhanced_queue_manager.py`** - Advanced queue management
  - Material tracking integration
  - Early failure detection
  - Automated workflow progression
  - Resource optimization

- **`material_monitor.py`** - Real-time monitoring dashboard
  - Live calculation status
  - Property extraction monitoring
  - Database health checks

#### **1. config/** - Configuration Management

Stores system configuration files for error recovery and workflow settings.

**Files:**
- **`recovery_config.yaml`** - Error recovery strategies
  ```yaml
  SHRINK_ERROR:
    detection: "SHRINK FACTOR"
    fix: increase_shrink
    max_attempts: 3
  MEMORY_ERROR:
    detection: "MEMORY ALLOCATION"
    fix: increase_memory
    escalation: reduce_cores
  ```

#### **2. database/** - Material Tracking System

SQLite-based tracking system with ASE integration for complete calculation provenance.

**Core Components:**
- **`materials.py`** - Main database interface
  - Thread-safe material and calculation tracking
  - Property storage and retrieval
  - ASE structure integration
  - **NEW: Property-based filtering capabilities**
  - Example usage:
    ```python
    db = MaterialDatabase()
    db.add_material("diamond", structure)
    db.update_calculation_status(calc_id, "completed")
    
    # NEW: Filter materials by properties
    materials = db.filter_materials_by_properties(
        ["band_gap > 3.0", "total_energy < -1000"],
        logic="AND"
    )
    ```

- **`query/`** - Advanced query and filtering module (**NEW**)
  - **`filters.py`** - Property range filtering system
    - Support for numeric and string comparisons
    - Operators: >, >=, <, <=, ==, !=
    - AND/OR logic for combining filters
    - Example:
      ```python
      from mace.database.query import PropertyFilter
      filter = PropertyFilter()
      filter.add_filter("band_gap", ">", 3.0)
      filter.add_filter("space_group", "==", 227)
      ```

- **`create_fresh_database.py`** - Database initialization
  - Creates schema with proper indices
  - Sets up trigger functions
  - Initializes workflow templates

- **`database_status_report.py`** - Generate reports
  - Material statistics
  - Calculation success rates
  - Performance metrics

#### **3. queue/** - Job Queue Management

Sophisticated SLURM integration with intelligent job scheduling.

**Key Scripts:**
- **`manager.py`** - Enhanced queue manager class
  - Job submission with throttling
  - Resource allocation optimization
  - Callback handling for job completion
  - Integration with material database

- **`monitor.py`** - Queue monitoring utilities
  - Real-time job status tracking
  - Resource utilization analysis
  - Failure pattern detection

- **`queue_lock_manager.py`** - Concurrency control
  - Prevents race conditions
  - Manages callback throttling
  - Ensures atomic operations

#### **4. recovery/** - Error Detection and Recovery

Automated error handling with configurable recovery strategies.

**Components:**
- **`detector.py`** - Error pattern detection
  - Parses CRYSTAL output files
  - Identifies common error patterns
  - Classifies error severity

- **`recovery.py`** - Recovery engine
  - Applies fixes based on error type
  - Integrates with fixk.py and updatelists2.py
  - Automatic job resubmission
  - Example recovery flow:
    ```python
    error = detector.detect_error(output_file)
    fix = recovery.get_fix_strategy(error)
    recovery.apply_fix(input_file, fix)
    recovery.resubmit_job(job_id)
    ```

#### **5. submission/** - Job Submission

SLURM script generation and job submission utilities.

**Scripts:**
- **`crystal.py`** - Submit CRYSTAL calculations
  - Handles .d12 input files
  - Resource allocation
  - Scratch directory setup

- **`properties.py`** - Submit property calculations
  - Handles .d3 property files
  - Manages wavefunction dependencies
  - Optimized for memory-intensive calculations

- **`submitcrystal23.sh`** - CRYSTAL23 submission script
  - Module loading
  - Environment setup
  - Callback integration

- **`submit_prop.sh`** - Property submission script
  - Specialized for BAND/DOSS/TRANSPORT
  - Higher memory allocation
  - Wavefunction handling

#### **6. utils/** - Utility Functions

Comprehensive toolkit for property extraction and analysis.

**Property Extraction:**
- **`property_extractor.py`** - Extract all properties
  - Band gaps (direct/indirect)
  - Total energies
  - Structural parameters
  - Electronic properties

- **`formula_extractor.py`** - Chemical information
  - Extract molecular formula
  - Determine space group
  - Count atoms and species

- **`settings_extractor.py`** - Settings extraction
  - Parse D12/D3 input files
  - Extract calculation parameters
  - Store configuration history

**Analysis Tools:**
- **`advanced_electronic_analyzer.py`** - Electronic analysis
  - Band structure analysis
  - DOS integration
  - Fermi level determination

- **`population_analysis_processor.py`** - Population analysis
  - Mulliken charges
  - Orbital populations
  - Charge density analysis

**Display and Visualization:**
- **`banner.py`** - MACE banner display
- **`animation.py`** - Progress animations
- **`show_properties.py`** - Display extracted properties

#### **7. workflow/** - Workflow Management

Complete workflow planning and execution system.

**Core Components:**
- **`planner.py`** - Interactive workflow planner
  - CIF/D12 input selection
  - Template-based workflow design
  - Resource planning
  - Configuration persistence

- **`executor.py`** - Workflow execution engine
  - Dependency management
  - Error handling
  - Progress tracking
  - File organization

- **`engine.py`** - Workflow automation
  - Automatic progression (OPT→SP→Properties)
  - State management
  - Recovery integration

- **`monitor_workflow.py`** - Workflow monitoring
  - Real-time status updates
  - Progress visualization
  - Performance metrics

**Subdirectory:**
- **`common/`** - Shared components
  - **`constants.py`** - Workflow constants
    - Calculation type definitions
    - Resource defaults
    - Template configurations

### Integration Points

1. **Material Database** ↔ **Queue Manager**
   - Automatic calculation tracking
   - Status updates on job completion

2. **Error Recovery** ↔ **Submission**
   - Automatic resubmission of fixed jobs
   - Resource adjustment based on errors

3. **Workflow Engine** ↔ **All Components**
   - Orchestrates entire calculation pipeline
   - Manages dependencies and data flow

4. **Property Extractor** ↔ **Database**
   - Stores extracted properties
   - Enables property queries

### Usage Patterns

**Simple Calculation:**
```bash
mace submit calculation.d12
```

**Complete Workflow:**
```bash
mace workflow --interactive
# Select CIFs → Choose workflow → Configure resources → Execute
```

**Monitoring:**
```bash
mace monitor --dashboard
# Real-time view of all calculations
```

**Analysis:**
```bash
mace analyze --extract-properties output_directory/
# Extract and store all properties
```

## Documentation

- [INSTALLATION.md](INSTALLATION.md) - Installation and setup guide
- [CLAUDE.md](../CLAUDE.md) - Comprehensive technical documentation
- [Examples](examples/) - Tutorial notebooks and example workflows

## Advanced Features

### Material Database
Track all calculations with full provenance:
```python
from mace.database.materials import MaterialDatabase
db = MaterialDatabase()
properties = db.get_material_properties("diamond")
```

### Custom Workflows
Create complex calculation sequences:
```python
from mace.workflow.planner import WorkflowPlanner
planner = WorkflowPlanner()
planner.add_calculation("OPT", resources={"cores": 32, "walltime": "7-00:00:00"})
planner.add_calculation("BAND", dependencies=["OPT"])
```

### Error Recovery
Automatic error detection and fixing:
```yaml
# recovery_config.yaml
SHRINK_ERROR:
  detection: "SHRINK FACTOR"
  fix: increase_shrink
  max_attempts: 3
```

## Contributing

We welcome contributions! Please see our contributing guidelines for:
- Code style conventions
- Testing requirements
- Pull request process

## Support

- **Issues**: Report bugs via GitHub Issues
- **Questions**: Contact the development team
- **Documentation**: See [CLAUDE.md](../CLAUDE.md) for comprehensive docs

## License

[License information to be added]

## Citation

If you use MACE in your research, please cite:
```
[Citation to be added]
```

---

<p align="center">
  <strong>MACE - Making CRYSTAL calculations accessible and automated</strong>
</p>