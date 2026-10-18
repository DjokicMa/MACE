"""The GUI's side of the element color palettes (View › Element Colors…).

The file, its format and the reading rules are in
mace/utils/element_palettes.py, which ``mace plotting`` reads too, so the
CLI draws atoms in the palette saved here. Writes go through prefstore
(the lock and read-back check gui.json uses), so two servers saving at once
keep both changes; the file is mode 0600 like the rest of the profile.
"""
from __future__ import annotations

from typing import Any, Dict

from . import prefstore
from ..utils import element_palettes as EP


def view() -> Dict[str, Any]:
    """What the page needs: the file's path, the default and every user palette."""
    doc = EP.read_user()
    return {"path": str(EP.config_path()), "rev": doc["rev"], "default": doc["default"],
            "palettes": doc["palettes"]}


def change(op: Dict[str, Any]) -> Dict[str, Any]:
    """Apply one change (EP.apply_op); ValueError for a bad request."""
    EP.apply_op(EP.read_user(), op)       # checked first: a bad request never writes
    made: Dict[str, Any] = {}

    def step(cur):
        new, m = EP.apply_op(cur, op)
        made.clear()
        made.update(m)
        return new

    prefstore.update(EP.config_path(), step, lambda back: True, EP.MAX_BYTES)
    return {**view(), **made}
