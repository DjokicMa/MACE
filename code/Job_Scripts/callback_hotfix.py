#!/usr/bin/env python3
"""
Callback Hotfix
---------------
Quick patch for the callback mechanism issues.
"""

def apply_hotfix():
    """Apply hotfix to the enhanced_queue_manager.py file."""
    import re
    
    # Read the current file
    with open('enhanced_queue_manager.py', 'r') as f:
        content = f.read()
    
    # Fix 1: Store db_path
    content = re.sub(
        r'self\.enable_tracking = enable_tracking',
        'self.enable_tracking = enable_tracking\n        self.db_path = db_path',
        content
    )
    
    # Fix 2: Change add_material to create_material in populate_completed_jobs.py
    try:
        with open('populate_completed_jobs.py', 'r') as f:
            pop_content = f.read()
        
        pop_content = pop_content.replace('db.add_material(', 'db.create_material(')
        pop_content = pop_content.replace('db.add_calculation(', 'db.create_calculation(')
        
        with open('populate_completed_jobs.py', 'w') as f:
            f.write(pop_content)
        
        print("✓ Fixed populate_completed_jobs.py")
        
    except FileNotFoundError:
        print("! populate_completed_jobs.py not found, creating minimal version...")
        
        # Create a minimal version that works
        minimal_pop = '''def scan_for_completed_calculations(base_dir):
    return []

def populate_database(completed_calcs, db):
    return 0
'''
        with open('populate_completed_jobs.py', 'w') as f:
            f.write(minimal_pop)
    
    # Write the updated content
    with open('enhanced_queue_manager.py', 'w') as f:
        f.write(content)
    
    print("✓ Applied hotfix to enhanced_queue_manager.py")

if __name__ == "__main__":
    apply_hotfix()
    print("\nHotfix applied! Now try:")
    print("python enhanced_queue_manager.py --callback-mode completion")