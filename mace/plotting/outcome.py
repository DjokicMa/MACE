"""The figures a `mace plotting` run could not draw.

Each plotter goes on to the next file when one fails (one bad file does not stop the batch), and says
so where it happens. It also records the failure here, so `mace plotting` can exit non-zero when any
requested figure was not written (1.1.6 review F4-3: it exited 0 after a file error).
"""
from typing import List, Tuple

_failed: List[Tuple[str, str]] = []


def reset() -> None:
    _failed.clear()


def failed(name: str, why: str = "") -> None:
    """Record that the figure of ``name`` (a file name) was not drawn, and why."""
    _failed.append((str(name), str(why)))


def failures() -> List[Tuple[str, str]]:
    return list(_failed)
