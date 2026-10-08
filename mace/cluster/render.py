"""Plain-text tables for `mace cluster` (standard library only)."""
from __future__ import annotations

from typing import Any, List, Optional, Sequence


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]], right: Sequence[int] = ()) -> str:
    cells = [[str(h) for h in headers]] + [["" if c is None else str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    out = []
    for r in cells:
        parts = []
        for i, c in enumerate(r):
            parts.append(c.rjust(widths[i]) if i in right else c.ljust(widths[i]))
        out.append("  ".join(parts).rstrip())
    return "\n".join(out)


def num(n: Optional[float]) -> str:
    return "-" if n is None else f"{int(n):,}"


def gib(gb: Optional[float]) -> str:
    if gb is None:
        return "-"
    return f"{gb / 1024:.1f} TiB" if gb >= 1024 else f"{gb:.1f} GiB"


def gpus(free: dict, total: dict) -> str:
    if not total:
        return "-"
    return ", ".join(f"{t} {free.get(t, 0)}/{n}" for t, n in sorted(total.items()))


def age(seconds: Optional[float]) -> str:
    if seconds is None:
        return "never"
    s = int(seconds)
    return f"{s // 60}:{s % 60:02d} ago"


def pct(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:g} %"


def bar(p: Optional[float], width: int = 20) -> str:
    if p is None:
        return " " * width
    n = max(0, min(width, int(round(p / 100.0 * width))))
    return "#" * n + "." * (width - n)


def lines(*parts: Optional[str]) -> List[str]:
    return [p for p in parts if p]
