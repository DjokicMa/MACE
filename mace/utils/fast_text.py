"""Faster whole-file regex searches with identical results.

A case-insensitive ``re.search`` over a multi-megabyte CRYSTAL output cannot
use the regex engine's literal fast path, so each one scans the whole file
character by character. Most of those patterns hold a plain literal that any
match must contain; when the text has no non-ASCII character, the lower-case
copy of the text contains that literal (lower-cased) exactly when such a match
is possible, so a substring test settles most searches. Only when the literal
is present (or the text is not ASCII, where ``re.IGNORECASE`` folds characters
such as U+017F or U+212A to ASCII letters) does the original regex run, on the
original text, from the first place the pattern's leading literal occurs (no
match can start before it), so every result is the one ``re.search`` gives.
"""

import re
from typing import List, Optional

_cache = ()                   # ((text, lower copy or False), ...): the last two texts, swapped atomically


def lower_ascii(text: str):
    """``text.lower()`` when ``text`` is ASCII, else None (cached for the
    last two texts: the extractors search the same file content, and a part
    of it, many times)."""
    global _cache
    cache = _cache
    for t, low in cache:
        if t is text:
            return low or None
    low = text.lower() if text.isascii() else False
    _cache = ((text, low),) + cache[:1]
    return low or None


def _start(text: str, literals) -> Optional[int]:
    """Where a match can first start: the first occurrence of the leading
    literal (``literals[0]``) in the lower-case ASCII copy; -1 when a literal
    is missing (no match is possible); 0 when the text is not ASCII."""
    low = lower_ascii(text)
    if low is None:
        return 0
    for lit in literals[1:]:
        if lit not in low:
            return -1
    return low.find(literals[0])


def ci_search(pattern, text: str, literals, flags: int = re.IGNORECASE) -> Optional[re.Match]:
    """``re.search(pattern, text, flags)``. ``flags`` holds IGNORECASE;
    ``literals`` are lower-case texts the pattern matches verbatim, and the
    pattern starts with the first one (no anchor or lookbehind before it), so
    no match can begin before that literal's first occurrence."""
    pos = _start(text, literals)
    if pos < 0:
        return None
    return re.compile(pattern, flags).search(text, pos)


def ci_findall(pattern, text: str, literals, flags: int = re.IGNORECASE) -> List:
    """``re.findall(pattern, text, flags)``, under the contract of ci_search."""
    pos = _start(text, literals)
    if pos < 0:
        return []
    return re.compile(pattern, flags).findall(text, pos)
