"""
Isolated pandas utilities to avoid PyArrow conflicts
"""

# Store pandas instance to ensure single import
_pd = None

def get_pandas():
    """Get pandas instance, ensuring it's only imported once"""
    global _pd
    if _pd is None:
        try:
            import pandas as pd
            _pd = pd
        except Exception:
            _pd = False
    return _pd

def pandas_available():
    """Check if pandas is available"""
    pd = get_pandas()
    return pd is not False

def read_csv(filepath, **kwargs):
    """Read CSV file using pandas if available, otherwise fallback"""
    pd = get_pandas()
    if pd:
        return pd.read_csv(filepath, **kwargs)
    else:
        # Fallback to basic CSV reading
        import csv
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)

def create_dataframe(data):
    """Create DataFrame if pandas available"""
    pd = get_pandas()
    if pd:
        return pd.DataFrame(data)
    else:
        return data  # Return raw data if pandas not available

def to_datetime(series):
    """Convert to datetime if pandas available"""
    pd = get_pandas()
    if pd:
        return pd.to_datetime(series)
    else:
        return series