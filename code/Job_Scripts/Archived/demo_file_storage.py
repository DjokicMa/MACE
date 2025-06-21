#!/usr/bin/env python3
"""
File Storage System Demonstration
=================================
Demonstrate the comprehensive file storage system for CRYSTAL calculations.

This script shows how the system:
1. Stores D12/D3 input files with complete settings extraction
2. Preserves calculation provenance with checksums
3. Archives all calculation files (input, output, binary, scripts)
4. Extracts and stores calculation parameters
5. Enables file retrieval and integrity verification

Usage:
  python demo_file_storage.py --demo-storage
  python demo_file_storage.py --show-capabilities
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

# Add script directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

try:
    from file_storage_manager import FileStorageManager
    from material_database import MaterialDatabase
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)


def create_demo_calculation_files(demo_dir: Path) -> str:
    """Create demo calculation files for testing."""
    print("🔧 Creating demo calculation files...")
    
    # Create a realistic D12 file
    d12_content = """DIAMOND BULK
CRYSTAL
0 0 0
227
5.67
1
6 0.0 0.0 0.0
END
DFT
EXCHANGE
B3LYP
CORRELAT
LYP
NONLOCAL
6-31G**
0 0 6 2.0 1.0
0 2 2 8.0 1.0
END
SHRINK
8 8
TOLINTEG
8 8 8 8 16
TOLDEE
8
SCFDIR
MAXCYCLE
50
FMIXING
90
END
"""
    
    # Create output file with properties
    out_content = """
CRYSTAL17 - QUANTUM MECHANICAL CALCULATIONS ON CRYSTALLINE SYSTEMS
 
 COMPUTATIONAL CONDITIONS
 SPACE GROUP NUMBER  227 - "F D -3 M:1"

 LATTICE PARAMETERS (ANGSTROMS AND DEGREES) - PRIMITIVE CELL
        A              B              C           ALPHA      BETA       GAMMA      VOLUME
     2.52000000     2.52000000     2.52000000    60.000000  60.000000  60.000000     11.32
  
 PRIMITIVE CELL - CENTRING CODE 1/0 VOLUME=     11.32 - DENSITY  3.515 g/cm^3
          A              B              C           ALPHA      BETA       GAMMA 
       2.52000000     2.52000000     2.52000000    60.000000  60.000000  60.000000

 ATOMS IN THE UNIT CELL: 2

 ATOM              X/A                 Y/B                 Z/C    
 *******************************************************************************
   1 T   6 C      0.000000000000E+00  0.000000000000E+00  0.000000000000E+00
   2 T   6 C      2.500000000000E-01  2.500000000000E-01  2.500000000000E-01

 TOTAL ENERGY(DFT)(AU)(     5) -7.5738502621E+01
 D3 DISPERSION ENERGY (AU)   -2.4567E-03
 TOTAL ENERGY + DISP (AU)    -7.5741059291E+01

 DIRECT ENERGY BAND GAP:   5.48 eV
 INDIRECT ENERGY BAND GAP: 5.48 eV

 MULLIKEN POPULATION ANALYSIS - ALPHA+BETA ELECTRONS
 NO. OF ELECTRONS     8.000000

  ATOM    CHARGE          ATOM ELECTRONS
     1     6.0000           4.0000
     2     6.0000           4.0000

 MULLIKEN POPULATION ANALYSIS - ALPHA-BETA ELECTRONS
 NO. OF ELECTRONS     0.000000

  ATOM    CHARGE          ATOM ELECTRONS  
     1     0.0000           0.0000
     2     0.0000           0.0000
"""
    
    # Create SLURM script
    slurm_content = """#!/bin/bash
#SBATCH --job-name=diamond_opt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --memory=5G
#SBATCH --time=7-00:00:00
#SBATCH --account=mendoza_q

module load intel/2021.4.0
module load mpi/intel/2021.4.0

cd $SLURM_SUBMIT_DIR
mpirun crystal < diamond.d12 > diamond.out
"""
    
    # Create demo files
    demo_dir.mkdir(exist_ok=True)
    
    material_id = "demo_diamond"
    
    (demo_dir / f"{material_id}.d12").write_text(d12_content)
    (demo_dir / f"{material_id}.out").write_text(out_content)
    (demo_dir / f"{material_id}.sh").write_text(slurm_content)
    
    # Create some binary files (simulated)
    (demo_dir / "fort.9").write_bytes(b"CRYSTAL_WAVEFUNCTION_BINARY_DATA" * 100)
    (demo_dir / "fort.25").write_bytes(b"CRYSTAL_PHONON_BINARY_DATA" * 50)
    
    print(f"   ✅ Created demo files in {demo_dir}")
    return material_id


def demonstrate_file_storage():
    """Demonstrate complete file storage capabilities."""
    print("🚀 CRYSTAL File Storage System Demonstration")
    print("=" * 60)
    
    # Create temporary demo directory
    with tempfile.TemporaryDirectory() as temp_dir:
        demo_dir = Path(temp_dir) / "demo_calculation"
        storage_root = Path(temp_dir) / "storage"
        
        # Create demo files
        material_id = create_demo_calculation_files(demo_dir)
        
        # Initialize file storage system
        print("\n📁 Initializing File Storage System")
        storage_manager = FileStorageManager(
            db_path=str(Path(temp_dir) / "demo.db"),
            storage_root=str(storage_root)
        )
        
        # Demonstrate file storage
        print(f"\n📦 Storing Calculation Files")
        calc_id = f"calc_{material_id}_opt_001"
        calc_type = "OPT"
        
        storage_info = storage_manager.store_calculation_files(
            calc_id=calc_id,
            material_id=material_id,
            calc_type=calc_type,
            source_directory=demo_dir,
            preserve_original=True
        )
        
        # Show storage results
        print(f"\n📊 Storage Results:")
        print(f"   Files stored: {len(storage_info['files_stored'])}")
        print(f"   Settings extracted: {len(storage_info['settings_extracted'])}")
        print(f"   Storage directory: {storage_info['storage_directory']}")
        
        # Show stored files
        print(f"\n📋 Stored Files:")
        for filename, file_info in storage_info['files_stored'].items():
            category = file_info['category']
            importance = file_info['importance']
            size = file_info['size']
            print(f"   {category:10} | {importance:8} | {filename:20} | {size:8} bytes")
        
        # Show extracted settings
        print(f"\n⚙️  Extracted Settings:")
        for filename, settings in storage_info['settings_extracted'].items():
            print(f"   📄 {filename}:")
            if 'crystal_keywords' in settings:
                keywords = ', '.join(settings['crystal_keywords'])
                print(f"      CRYSTAL Keywords: {keywords}")
            
            if 'calculation_parameters' in settings:
                params = settings['calculation_parameters']
                for param, value in params.items():
                    print(f"      {param}: {value}")
        
        # Demonstrate file listing
        print(f"\n📂 Database File Records:")
        stored_files = storage_manager.list_stored_files(calc_id)
        for file_info in stored_files:
            print(f"   {file_info['file_type']:10} | {file_info['file_name']:20} | {file_info['file_size']:8} bytes")
        
        # Demonstrate integrity verification
        print(f"\n🔍 File Integrity Check:")
        integrity = storage_manager.verify_file_integrity(calc_id)
        for filename, is_valid in integrity.items():
            status = "✅ Valid" if is_valid else "❌ Corrupted"
            print(f"   {status:10} | {filename}")
        
        # Demonstrate file retrieval
        print(f"\n📂 File Retrieval Demonstration:")
        retrieval_dir = Path(temp_dir) / "retrieved_files"
        success = storage_manager.retrieve_calculation_files(calc_id, retrieval_dir)
        
        if success:
            retrieved_files = list(retrieval_dir.glob("*"))
            print(f"   ✅ Retrieved {len(retrieved_files)} files:")
            for file_path in retrieved_files:
                print(f"      📄 {file_path.name}")
        
        # Demonstrate settings retrieval
        print(f"\n⚙️  Settings Retrieval:")
        settings = storage_manager.get_calculation_settings(calc_id)
        if settings:
            print(f"   ✅ Retrieved settings for {len(settings)} files")
            print(json.dumps(settings, indent=2)[:500] + "..." if len(str(settings)) > 500 else json.dumps(settings, indent=2))
        
        print(f"\n🎉 File Storage Demonstration Complete!")
        print(f"\nKey Capabilities Demonstrated:")
        print(f"✅ Complete file preservation with checksums")
        print(f"✅ D12/D3 input settings extraction") 
        print(f"✅ Binary file storage (fort.9, fort.25)")
        print(f"✅ Database integration with file tracking")
        print(f"✅ File retrieval and integrity verification")
        print(f"✅ Calculation provenance preservation")


def show_file_storage_capabilities():
    """Show comprehensive file storage capabilities."""
    print("📋 CRYSTAL File Storage System Capabilities")
    print("=" * 60)
    
    print("\n🎯 File Types Supported:")
    file_types = {
        'Input Files': ['.d12', '.d3', '.input'],
        'Output Files': ['.out', '.output', '.log'],
        'Binary Files': ['.f9', '.f25', '.fort.9', '.fort.25', '.wf', '.prop'],
        'Property Files': ['.BAND', '.DOSS', '.OPTC', '.ELPH'],
        'Script Files': ['.sh', '.slurm', '.job'],
        'Plot Files': ['.png', '.pdf', '.eps', '.svg'],
        'Data Files': ['.csv', '.json', '.yaml', '.xml'],
        'Config Files': ['.conf', '.cfg', '.ini', '.param']
    }
    
    for category, extensions in file_types.items():
        print(f"   {category:15} | {', '.join(extensions)}")
    
    print("\n⚙️  Settings Extraction Features:")
    print("   ✅ CRYSTAL keywords (OPTGEOM, DFT, EXCHANGE, etc.)")
    print("   ✅ Calculation parameters (SHRINK, TOLINTEG, TOLDEE)")
    print("   ✅ Basis set information (internal/external)")
    print("   ✅ Geometry optimization settings")
    print("   ✅ Exchange-correlation functionals")
    print("   ✅ SCF convergence parameters")
    
    print("\n📦 Storage Features:")
    print("   ✅ Complete file preservation with original names")
    print("   ✅ SHA256 checksums for integrity verification")
    print("   ✅ Organized directory structure by calculation")
    print("   ✅ Metadata storage with timestamps")
    print("   ✅ Database integration for tracking")
    print("   ✅ Configurable preservation vs. archival")
    
    print("\n🔍 Retrieval Features:")
    print("   ✅ Full calculation reconstruction")
    print("   ✅ Individual file access")
    print("   ✅ Settings query and analysis")
    print("   ✅ File integrity verification")
    print("   ✅ Calculation provenance tracking")
    
    print("\n🗄️  Database Integration:")
    print("   ✅ File records in materials database")
    print("   ✅ Settings stored in calculations table")
    print("   ✅ File type and importance classification")
    print("   ✅ Checksum verification tracking")
    print("   ✅ Storage timestamp recording")
    
    print("\n🔄 Workflow Integration:")
    print("   ✅ Automatic storage on job completion")
    print("   ✅ Enhanced queue manager integration")
    print("   ✅ Property extraction coordination")
    print("   ✅ Error recovery file preservation")
    
    print("\n💡 Use Cases:")
    print("   📊 Complete calculation provenance")
    print("   🔬 Reproducing calculations exactly")
    print("   📈 Parameter analysis across materials")
    print("   🔍 Debugging failed calculations")
    print("   📚 Building calculation databases")
    print("   🔄 Workflow restart and recovery")


def main():
    """Main demonstration function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CRYSTAL File Storage System Demo")
    parser.add_argument("--demo-storage", action="store_true", help="Run file storage demonstration")
    parser.add_argument("--show-capabilities", action="store_true", help="Show system capabilities")
    
    args = parser.parse_args()
    
    if args.demo_storage:
        demonstrate_file_storage()
    elif args.show_capabilities:
        show_file_storage_capabilities()
    else:
        print("❌ Please specify --demo-storage or --show-capabilities")
        print("\nAvailable options:")
        print("  --demo-storage      : Run complete demonstration with example files")
        print("  --show-capabilities : Show detailed system capabilities")


if __name__ == "__main__":
    main()