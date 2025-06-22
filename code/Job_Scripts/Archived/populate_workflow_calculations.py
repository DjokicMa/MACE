#!/usr/bin/env python3
"""
Populate Database with Completed Workflow Calculations
======================================================
Scan workflow output directories and populate the materials database with
completed calculations to enable calculation type tracking.
"""

import sys
from pathlib import Path
import re

# Add script directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir / "code" / "Job_Scripts"))

from material_database import MaterialDatabase, create_material_id_from_file
from formula_extractor import extract_formula_and_space_group

def populate_workflow_calculations(workflow_dir="workflow_outputs", db_path="materials.db"):
    """
    Populate database with completed workflow calculations.
    
    Args:
        workflow_dir: Directory containing workflow outputs
        db_path: Path to materials database
    """
    db = MaterialDatabase(db_path)
    workflow_path = Path(workflow_dir)
    
    if not workflow_path.exists():
        print(f"❌ Workflow directory not found: {workflow_dir}")
        return
    
    print(f"🔍 Scanning workflow outputs in: {workflow_path}")
    print("=" * 60)
    
    materials_created = 0
    calculations_added = 0
    
    # Find all workflow directories
    for workflow_id_dir in workflow_path.glob("workflow_*"):
        if not workflow_id_dir.is_dir():
            continue
            
        print(f"\n📁 Processing workflow: {workflow_id_dir.name}")
        
        # Process each step directory
        for step_dir in sorted(workflow_id_dir.glob("step_*")):
            if not step_dir.is_dir():
                continue
                
            # Extract step info
            step_match = re.match(r'step_(\d+)_([A-Z]+)', step_dir.name)
            if not step_match:
                continue
                
            step_num = int(step_match.group(1))
            calc_type = step_match.group(2)
            
            print(f"  📂 Step {step_num}: {calc_type}")
            
            # Process each material calculation in this step
            for material_dir in step_dir.glob("*"):
                if not material_dir.is_dir():
                    continue
                    
                material_name = material_dir.name
                
                # Look for output files
                output_files = list(material_dir.glob("*.out"))
                if not output_files:
                    continue
                    
                output_file = output_files[0]  # Take first output file
                
                # Extract material ID from directory name
                material_id = create_material_id_from_file(material_name)
                
                print(f"    🔬 Material: {material_id} ({calc_type})")
                
                # Check if material exists, create if not
                existing_material = db.get_material(material_id)
                if not existing_material:
                    # Try to extract formula and space group
                    try:
                        # Look for input file to extract formula
                        input_files = list(material_dir.glob("*.d12")) + list(material_dir.glob("*.d3"))
                        formula = "Unknown"
                        space_group = None
                        
                        if input_files:
                            input_file = input_files[0]
                            try:
                                formula, space_group = extract_formula_and_space_group(str(input_file), str(output_file))
                            except Exception as e:
                                print(f"      ⚠️ Could not extract formula: {e}")
                        
                        # Create material
                        db.create_material(
                            material_id=material_id,
                            formula=formula,
                            space_group=space_group,
                            source_type="workflow",
                            source_file=str(output_file),
                            metadata={"workflow_id": workflow_id_dir.name}
                        )
                        materials_created += 1
                        print(f"      ✅ Created material: {formula}")
                        
                    except Exception as e:
                        print(f"      ❌ Error creating material: {e}")
                        continue
                
                # Create calculation record
                try:
                    calc_id = f"{material_id}_{calc_type}_{workflow_id_dir.name}_{step_num:03d}"
                    
                    # Determine status based on file existence and content
                    status = "completed" if output_file.exists() else "failed"
                    
                    # Check if calculation already exists
                    existing_calc = db.get_calculation(calc_id)
                    if existing_calc:
                        print(f"      ⏭️ Calculation already exists: {calc_id}")
                        continue
                    
                    # Create calculation
                    db.create_calculation(
                        material_id=material_id,
                        calc_type=calc_type,
                        input_file=str(input_files[0]) if input_files else None,
                        work_dir=str(material_dir),
                        settings={"workflow_step": step_num, "workflow_id": workflow_id_dir.name}
                    )
                    
                    # Update with completion status
                    db.update_calculation_status(
                        calc_id=calc_id,
                        status=status,
                        output_file=str(output_file) if output_file.exists() else None
                    )
                    
                    calculations_added += 1
                    print(f"      ✅ Added calculation: {calc_type} - {status}")
                    
                except Exception as e:
                    print(f"      ❌ Error creating calculation: {e}")
                    continue
    
    print(f"\n📊 Population Summary:")
    print(f"   Materials created: {materials_created}")
    print(f"   Calculations added: {calculations_added}")
    
    # Show updated database stats
    stats = db.get_database_stats()
    print(f"\n📈 Database Statistics:")
    print(f"   Total materials: {stats['total_materials']}")
    print(f"   Calculations by type: {stats.get('calculations_by_type', {})}")
    print(f"   Calculations by status: {stats.get('calculations_by_status', {})}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Populate database with workflow calculations")
    parser.add_argument("--workflow-dir", default="workflow_outputs", 
                       help="Directory containing workflow outputs")
    parser.add_argument("--db-path", default="materials.db",
                       help="Path to materials database")
    
    args = parser.parse_args()
    
    populate_workflow_calculations(args.workflow_dir, args.db_path)