"""One read of a CRYSTAL .DAT file per extraction.

The property extractor looks at a BAND.DAT three times (the k-path header,
DatFileProcessor and AdvancedElectronicAnalyzer). They share the text read
here: the last few files are kept, keyed by path, size and modification time,
so a rewritten file is read again.
"""
import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=2)
def _text(path: str, size: int, mtime_ns: int) -> str:
    with open(path, 'r') as fh:
        return fh.read()


def read_text(path) -> str:
    """The text of ``path`` (raises OSError as open() does)."""
    p = os.fspath(Path(path))
    st = os.stat(p)
    return _text(os.path.abspath(p), st.st_size, st.st_mtime_ns)
