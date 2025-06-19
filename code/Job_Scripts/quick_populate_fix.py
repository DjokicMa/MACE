#!/usr/bin/env python3
"""
Quick fix for populate_completed_jobs.py parameters
"""

def fix_populate_file():
    """Fix the parameter names in populate_completed_jobs.py"""
    
    try:
        with open('populate_completed_jobs.py', 'r') as f:
            content = f.read()
        
        # Fix the parameter name
        content = content.replace('structure_file=', 'source_file=')
        
        # Also fix the method call to include all required parameters correctly
        old_call = '''db.create_material(
                    material_id=calc['material_id'],
                    formula="Unknown",  # Will be updated later
                    source_file=calc['input_file'],
                    metadata={'auto_populated': True, 'source': 'populate_completed_jobs.py'}
                )'''
        
        new_call = '''db.create_material(
                    material_id=calc['material_id'],
                    formula="Unknown",
                    source_file=calc['input_file'],
                    source_type='auto_detected',
                    metadata={'auto_populated': True, 'source': 'populate_completed_jobs.py'}
                )'''
        
        content = content.replace(old_call, new_call)
        
        with open('populate_completed_jobs.py', 'w') as f:
            f.write(content)
        
        print("✓ Fixed populate_completed_jobs.py parameter names")
        return True
        
    except Exception as e:
        print(f"Error fixing file: {e}")
        return False

if __name__ == "__main__":
    fix_populate_file()
    print("Now try: python enhanced_queue_manager.py --callback-mode completion")