"""
Property History and Versioning
================================
Track changes to material properties over time.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import sqlite3
from contextlib import contextmanager


class PropertyHistory:
    """Manages property history and versioning."""
    
    def __init__(self, db_path: str = 'materials.db'):
        """
        Initialize property history manager.
        
        Args:
            db_path: Path to the materials database
        """
        self.db_path = db_path
        self._ensure_history_tables()
        
    @contextmanager
    def _get_connection(self):
        """Get database connection context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
            
    def _ensure_history_tables(self):
        """Create history tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Property history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS property_history (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    material_id TEXT NOT NULL,
                    property_name TEXT NOT NULL,
                    property_value TEXT,
                    property_unit TEXT,
                    property_category TEXT,
                    calc_id TEXT,
                    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    changed_by TEXT,
                    change_reason TEXT,
                    old_value TEXT,
                    version INTEGER DEFAULT 1,
                    FOREIGN KEY (material_id) REFERENCES materials(material_id)
                )
            """)
            
            # Create indices for efficient queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_prop_history_material 
                ON property_history(material_id, property_name)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_prop_history_time 
                ON property_history(changed_at)
            """)
            
            # Material version tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS material_versions (
                    version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    material_id TEXT NOT NULL,
                    version_number INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_by TEXT,
                    version_notes TEXT,
                    properties_snapshot TEXT,  -- JSON snapshot of all properties
                    FOREIGN KEY (material_id) REFERENCES materials(material_id),
                    UNIQUE(material_id, version_number)
                )
            """)
            
            conn.commit()
            
    def record_property_change(self, material_id: str, property_name: str,
                             new_value: Any, new_unit: Optional[str] = None,
                             calc_id: Optional[str] = None,
                             changed_by: Optional[str] = None,
                             change_reason: Optional[str] = None) -> int:
        """
        Record a property change in history.
        
        Args:
            material_id: Material identifier
            property_name: Name of the property
            new_value: New property value
            new_unit: Unit of the new value
            calc_id: Calculation ID if from calculation
            changed_by: User/system that made the change
            change_reason: Reason for the change
            
        Returns:
            History record ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get current value if exists
            cursor.execute("""
                SELECT property_value, property_unit 
                FROM properties 
                WHERE material_id = ? AND property_name = ?
                ORDER BY property_id DESC LIMIT 1
            """, (material_id, property_name))
            
            current = cursor.fetchone()
            old_value = None
            
            if current:
                old_value = current['property_value']
                
            # Get current version number
            cursor.execute("""
                SELECT MAX(version) as max_version 
                FROM property_history 
                WHERE material_id = ? AND property_name = ?
            """, (material_id, property_name))
            
            result = cursor.fetchone()
            version = (result['max_version'] or 0) + 1
            
            # Insert history record
            cursor.execute("""
                INSERT INTO property_history 
                (material_id, property_name, property_value, property_unit, 
                 calc_id, changed_by, change_reason, old_value, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (material_id, property_name, str(new_value), new_unit,
                  calc_id, changed_by, change_reason, old_value, version))
                  
            history_id = cursor.lastrowid
            conn.commit()
            
            return history_id
            
    def get_property_history(self, material_id: str, property_name: str,
                           limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get history of a specific property.
        
        Args:
            material_id: Material identifier
            property_name: Property name
            limit: Maximum number of records to return
            
        Returns:
            List of history records
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT * FROM property_history
                WHERE material_id = ? AND property_name = ?
                ORDER BY changed_at DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
                
            cursor.execute(query, (material_id, property_name))
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    'history_id': row['history_id'],
                    'property_value': row['property_value'],
                    'property_unit': row['property_unit'],
                    'changed_at': row['changed_at'],
                    'changed_by': row['changed_by'],
                    'change_reason': row['change_reason'],
                    'old_value': row['old_value'],
                    'version': row['version'],
                    'calc_id': row['calc_id']
                })
                
            return history
            
    def get_material_history(self, material_id: str,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all property changes for a material.
        
        Args:
            material_id: Material identifier
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            
        Returns:
            List of all property changes
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT * FROM property_history
                WHERE material_id = ?
            """
            
            params = [material_id]
            
            if start_date:
                query += " AND changed_at >= ?"
                params.append(start_date)
                
            if end_date:
                query += " AND changed_at <= ?"
                params.append(end_date)
                
            query += " ORDER BY changed_at DESC"
            
            cursor.execute(query, params)
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    'property_name': row['property_name'],
                    'property_value': row['property_value'],
                    'property_unit': row['property_unit'],
                    'changed_at': row['changed_at'],
                    'changed_by': row['changed_by'],
                    'change_reason': row['change_reason'],
                    'old_value': row['old_value'],
                    'version': row['version']
                })
                
            return history
            
    def create_material_snapshot(self, material_id: str,
                               created_by: Optional[str] = None,
                               version_notes: Optional[str] = None) -> int:
        """
        Create a versioned snapshot of all material properties.
        
        Args:
            material_id: Material identifier
            created_by: User creating the snapshot
            version_notes: Notes about this version
            
        Returns:
            Version number
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get all current properties
            cursor.execute("""
                SELECT property_name, property_value, property_unit, 
                       property_category, calc_id
                FROM properties
                WHERE material_id = ?
            """, (material_id,))
            
            properties = {}
            for row in cursor.fetchall():
                properties[row['property_name']] = {
                    'value': row['property_value'],
                    'unit': row['property_unit'],
                    'category': row['property_category'],
                    'calc_id': row['calc_id']
                }
                
            # Get current version number
            cursor.execute("""
                SELECT MAX(version_number) as max_version
                FROM material_versions
                WHERE material_id = ?
            """, (material_id,))
            
            result = cursor.fetchone()
            version_number = (result['max_version'] or 0) + 1
            
            # Create snapshot
            cursor.execute("""
                INSERT INTO material_versions
                (material_id, version_number, created_by, version_notes, properties_snapshot)
                VALUES (?, ?, ?, ?, ?)
            """, (material_id, version_number, created_by, version_notes,
                  json.dumps(properties)))
                  
            conn.commit()
            
            return version_number
            
    def get_material_versions(self, material_id: str) -> List[Dict[str, Any]]:
        """
        Get all versions of a material.
        
        Args:
            material_id: Material identifier
            
        Returns:
            List of versions
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT version_id, version_number, created_at, created_by, version_notes
                FROM material_versions
                WHERE material_id = ?
                ORDER BY version_number DESC
            """, (material_id,))
            
            versions = []
            for row in cursor.fetchall():
                versions.append({
                    'version_id': row['version_id'],
                    'version_number': row['version_number'],
                    'created_at': row['created_at'],
                    'created_by': row['created_by'],
                    'version_notes': row['version_notes']
                })
                
            return versions
            
    def get_version_snapshot(self, material_id: str, version_number: int) -> Dict[str, Any]:
        """
        Get property snapshot for a specific version.
        
        Args:
            material_id: Material identifier
            version_number: Version number
            
        Returns:
            Property snapshot
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT properties_snapshot, created_at, created_by, version_notes
                FROM material_versions
                WHERE material_id = ? AND version_number = ?
            """, (material_id, version_number))
            
            row = cursor.fetchone()
            
            if row:
                return {
                    'version_number': version_number,
                    'created_at': row['created_at'],
                    'created_by': row['created_by'],
                    'version_notes': row['version_notes'],
                    'properties': json.loads(row['properties_snapshot'])
                }
                
            return None
            
    def compare_versions(self, material_id: str, version1: int, version2: int) -> Dict[str, Any]:
        """
        Compare two versions of a material.
        
        Args:
            material_id: Material identifier
            version1: First version number
            version2: Second version number
            
        Returns:
            Comparison results
        """
        snapshot1 = self.get_version_snapshot(material_id, version1)
        snapshot2 = self.get_version_snapshot(material_id, version2)
        
        if not snapshot1 or not snapshot2:
            return {'error': 'One or both versions not found'}
            
        props1 = snapshot1['properties']
        props2 = snapshot2['properties']
        
        # Find differences
        added = []
        removed = []
        changed = []
        unchanged = []
        
        all_props = set(props1.keys()) | set(props2.keys())
        
        for prop in all_props:
            if prop in props1 and prop not in props2:
                removed.append({
                    'property': prop,
                    'old_value': props1[prop]['value'],
                    'old_unit': props1[prop].get('unit')
                })
            elif prop not in props1 and prop in props2:
                added.append({
                    'property': prop,
                    'new_value': props2[prop]['value'],
                    'new_unit': props2[prop].get('unit')
                })
            elif prop in props1 and prop in props2:
                if props1[prop]['value'] != props2[prop]['value']:
                    changed.append({
                        'property': prop,
                        'old_value': props1[prop]['value'],
                        'new_value': props2[prop]['value'],
                        'old_unit': props1[prop].get('unit'),
                        'new_unit': props2[prop].get('unit')
                    })
                else:
                    unchanged.append(prop)
                    
        return {
            'material_id': material_id,
            'version1': {
                'number': version1,
                'created_at': snapshot1['created_at']
            },
            'version2': {
                'number': version2,
                'created_at': snapshot2['created_at']
            },
            'added': added,
            'removed': removed,
            'changed': changed,
            'unchanged_count': len(unchanged),
            'total_changes': len(added) + len(removed) + len(changed)
        }
        
    def get_recent_changes(self, limit: int = 100,
                         material_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get recent property changes across all materials.
        
        Args:
            limit: Maximum number of changes to return
            material_id: Filter by specific material
            
        Returns:
            List of recent changes
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT h.*, m.formula, m.space_group
                FROM property_history h
                LEFT JOIN materials m ON h.material_id = m.material_id
            """
            
            params = []
            if material_id:
                query += " WHERE h.material_id = ?"
                params.append(material_id)
                
            query += " ORDER BY h.changed_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            
            changes = []
            for row in cursor.fetchall():
                changes.append({
                    'material_id': row['material_id'],
                    'formula': row['formula'],
                    'property_name': row['property_name'],
                    'property_value': row['property_value'],
                    'old_value': row['old_value'],
                    'changed_at': row['changed_at'],
                    'changed_by': row['changed_by'],
                    'change_reason': row['change_reason']
                })
                
            return changes
            
    def rollback_property(self, material_id: str, property_name: str,
                        version: int, reason: str) -> bool:
        """
        Rollback a property to a previous version.
        
        Args:
            material_id: Material identifier
            property_name: Property to rollback
            version: Version number to rollback to
            reason: Reason for rollback
            
        Returns:
            Success status
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get the historical value
            cursor.execute("""
                SELECT property_value, property_unit
                FROM property_history
                WHERE material_id = ? AND property_name = ? AND version = ?
            """, (material_id, property_name, version))
            
            row = cursor.fetchone()
            
            if not row:
                return False
                
            # Record the rollback
            self.record_property_change(
                material_id, property_name,
                row['property_value'], row['property_unit'],
                changed_by='system',
                change_reason=f"Rollback to version {version}: {reason}"
            )
            
            # Update current property
            cursor.execute("""
                UPDATE properties
                SET property_value = ?, property_unit = ?
                WHERE material_id = ? AND property_name = ?
            """, (row['property_value'], row['property_unit'],
                  material_id, property_name))
                  
            conn.commit()
            
            return True
            
    def format_history_report(self, material_id: str,
                            property_name: Optional[str] = None) -> str:
        """
        Format property history as readable report.
        
        Args:
            material_id: Material identifier
            property_name: Specific property (None for all)
            
        Returns:
            Formatted report
        """
        lines = []
        
        if property_name:
            lines.append(f"=== Property History: {property_name} ===")
            lines.append(f"Material: {material_id}")
            lines.append("")
            
            history = self.get_property_history(material_id, property_name)
            
            if not history:
                lines.append("No history found.")
            else:
                for i, record in enumerate(history):
                    lines.append(f"Version {record['version']} - {record['changed_at']}")
                    lines.append(f"  Value: {record['property_value']} {record.get('property_unit', '')}")
                    
                    if record['old_value']:
                        lines.append(f"  Previous: {record['old_value']}")
                        
                    if record['changed_by']:
                        lines.append(f"  Changed by: {record['changed_by']}")
                        
                    if record['change_reason']:
                        lines.append(f"  Reason: {record['change_reason']}")
                        
                    if i < len(history) - 1:
                        lines.append("")
        else:
            lines.append(f"=== Material History: {material_id} ===")
            lines.append("")
            
            history = self.get_material_history(material_id)
            
            if not history:
                lines.append("No history found.")
            else:
                # Group by date
                by_date = {}
                for record in history:
                    date = record['changed_at'].split('T')[0]
                    if date not in by_date:
                        by_date[date] = []
                    by_date[date].append(record)
                    
                for date, records in sorted(by_date.items(), reverse=True):
                    lines.append(f"\n{date}:")
                    for record in records:
                        value_str = f"{record['property_value']} {record.get('property_unit', '')}".strip()
                        if record['old_value']:
                            value_str += f" (was {record['old_value']})"
                            
                        lines.append(f"  {record['property_name']}: {value_str}")
                        
                        if record['change_reason']:
                            lines.append(f"    Reason: {record['change_reason']}")
                            
        return "\n".join(lines)