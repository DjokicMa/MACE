"""Shared helpers for the mace.defects tests (no test_ prefix: not collected).

The decks under tests/defects_data/ are the ones run on HPCC for the defect
facts in round 6 (MgO V_O, hBN V_N; PBE0/POB-TZVP-REV2).
"""
import difflib
from pathlib import Path

DATA = Path(__file__).resolve().parent / "defects_data"


def deck(name: str) -> str:
    return (DATA / f"{name}.d12").read_text()


def strip_records(text: str, keyword: str, n_records: int) -> str:
    """Remove a keyword and the n records that follow it."""
    lines = text.splitlines()
    i = [l.strip() for l in lines].index(keyword)
    return "\n".join(lines[:i] + lines[i + 1 + n_records:]) + "\n"


def mgo_host() -> str:
    """The measured pristine MgO deck without its SUPERCON: a primitive host."""
    return strip_records(deck("mgo_pris"), "SUPERCON", 3)


def hbn_host() -> str:
    return strip_records(deck("hbn3_pris"), "SUPERCEL", 2)


def same_but_title(a: str, b: str) -> bool:
    return a.splitlines()[1:] == b.splitlines()[1:] and a.endswith("\n") == b.endswith("\n")


def edits(host: str, built: str):
    """(inserted lines, deleted lines, replaced pairs) going from host to built, title excluded."""
    h = host.splitlines()[1:]
    b = built.splitlines()[1:]
    ins, dele, rep = [], [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=h, b=b, autojunk=False).get_opcodes():
        if op == "insert":
            ins += b[j1:j2]
        elif op == "delete":
            dele += h[i1:i2]
        elif op == "replace":
            rep.append((h[i1:i2], b[j1:j2]))
    return ins, dele, rep

