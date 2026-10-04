#!/usr/bin/env python3
"""Fetch and cut basis records from the CRYSTAL basis set library pages
(https://www.crystal.unito.it/Basis_Sets/<element>.html).

Each element page lists its basis sets as an anchor (``<a name="Rb_POB_TZVP_rev2">``)
followed by a ``<pre>`` block: the record as CRYSTAL reads it, then free text
(authors, journal, DOI). This module finds a named set on a page, cuts the
record off by the CRYSTAL23 basis-set grammar, and keeps the trailing text as
the reference - never as part of the record.

Grammar (CRYSTAL23 user's manual, "2.2.1 Standard route", p.25, and
"Effective core pseudo-potentials - ECP", p.84):

    NAT NSHL
    [if NAT > 200: the ECP]
        PSN keyword: HAYWLC HAYWSC BARTHE DURAND COLUSC COLULC COLUSH STUTSC
                     STUTLC STUTSH (internal libraries, no further records), or
        INPUT:  ZNUC M M0 M1 M2 M3 M4, then M+M0+M1+M2+M3+M4 records
                "ALFKL CGKL NKL"
        INPSOC: PSONAME, SOSCALE, then ZNUC M M0..M4 and M+M0+..+M4 records
                "ALFKL CGKL CSOGKL NKL"
    NSHL shells: "ITYB LAT NG CHE SCAL", then - only when ITYB = 0 - NG records
                 "EXP COE1" (plus COE2 when LAT = 1, an SP shell).

Standard library only, so it runs anywhere; pages are cached on disk.
"""
import datetime
import hashlib
import html
import os
import re
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basis_library import (  # noqa: E402  (the grammar lives with the loader)
    GrammarError, cut_record, describe_record, numeric_tokens, same_numbers)

BASE_URL = "https://www.crystal.unito.it/Basis_Sets/"
SITE_LABEL = "CRYSTAL basis set library (crystal.unito.it)"

# Page names as the site spells them. Alternates are tried in order and the
# one that answered is recorded with the record.
ELEMENT_PAGES = {
    1: ["hydrogen"], 2: ["helium"], 3: ["lithium"], 4: ["beryllium"], 5: ["boron"],
    6: ["carbon"], 7: ["nitrogen"], 8: ["oxygen"], 9: ["fluorine"], 10: ["neon"],
    11: ["sodium"], 12: ["magnesium"], 13: ["aluminium", "aluminum"], 14: ["silicon"],  # british-ok: UniTO page name
    15: ["phosphorus"], 16: ["sulphur", "sulfur"], 17: ["chlorine"], 18: ["argon"],  # british-ok: UniTO page name
    19: ["potassium"], 20: ["calcium"], 21: ["scandium"], 22: ["titanium"],
    23: ["vanadium"], 24: ["chromium"], 25: ["manganese"], 26: ["iron"], 27: ["cobalt"],
    28: ["nickel"], 29: ["copper"], 30: ["zinc"], 31: ["gallium"], 32: ["germanium"],
    33: ["arsenic"], 34: ["selenium"], 35: ["bromine"], 36: ["krypton"],
    37: ["rubidium"], 38: ["strontium"], 39: ["yttrium"], 40: ["zirconium"],
    41: ["niobium"], 42: ["molibdenum", "molybdenum"], 43: ["technecium", "technetium"],
    44: ["ruthenium"], 45: ["rhodium"], 46: ["palladium"], 47: ["silver"],
    48: ["cadmium"], 49: ["indium"], 50: ["tin"], 51: ["antimony"], 52: ["tellurium"],
    53: ["iodine"], 54: ["xenon"], 55: ["caesium", "cesium"], 56: ["barium"],  # british-ok: UniTO page name
    57: ["lanthanum"], 58: ["cerium"], 59: ["praseodymium"], 60: ["neodymium"],
    61: ["promethium"], 62: ["samarium"], 63: ["europium"], 64: ["gadolinium"],
    65: ["terbium"], 66: ["dysprosium"], 67: ["holmium"], 68: ["erbium"],
    69: ["thulium"], 70: ["ytterbium"], 71: ["lutetium"], 72: ["hafnium"],
    73: ["tantalum"], 74: ["tungsten"], 75: ["rhenium"], 76: ["osmium"],
    77: ["iridium"], 78: ["platinum"], 79: ["gold"], 80: ["mercury"], 81: ["thallium"],
    82: ["lead"], 83: ["bismuth"], 84: ["polonium"], 85: ["astatine"], 86: ["radon"],
    87: ["francium"], 88: ["radium"], 89: ["actinium"], 90: ["thorium"],
    91: ["protactinium"], 92: ["uranium"], 93: ["neptunium"], 94: ["plutonium"],
    95: ["americium"], 96: ["curium"], 97: ["berkelium"], 98: ["californium"],
    99: ["einsteinium"],
}

SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
    "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es").split()

DOI_RE = re.compile(r"(?:DOI:?\s*|doi\.org/)(10\.\d{4,9}/[^\s\"',;<>]+)", re.IGNORECASE)


def symbol(z):
    return SYMBOLS[z - 1]


# --- fetching -------------------------------------------------------------

class Page:
    def __init__(self, z, url, text, retrieved):
        self.z = z
        self.url = url
        self.text = text
        self.retrieved = retrieved
        self.sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self.sections = parse_sections(text)


def fetch_page(z, cache_dir, timeout=30, opener=None):
    """The element's page, from cache_dir or the site. Returns a Page or
    raises LookupError naming every URL tried."""
    os.makedirs(cache_dir, exist_ok=True)
    for name in ELEMENT_PAGES[z]:
        cached = os.path.join(cache_dir, name + ".html")
        stamp = cached + ".retrieved"
        if os.path.isfile(cached) and os.path.isfile(stamp):
            with open(cached, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            with open(stamp) as fh:
                retrieved = fh.read().strip()
            return Page(z, BASE_URL + name + ".html", text, retrieved)
    tried = []
    for name in ELEMENT_PAGES[z]:
        url = BASE_URL + name + ".html"
        cached = os.path.join(cache_dir, name + ".html")
        stamp = cached + ".retrieved"
        try:
            with (opener or urllib.request.urlopen)(url, timeout=timeout) as resp:
                if getattr(resp, "status", 200) != 200:
                    tried.append(f"{url} (HTTP {resp.status})")
                    continue
                text = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            tried.append(f"{url} (HTTP {exc.code})")
            continue
        except (urllib.error.URLError, OSError) as exc:
            tried.append(f"{url} ({exc})")
            continue
        if "<pre" not in text.lower():
            tried.append(f"{url} (no basis set blocks on the page)")
            continue
        retrieved = datetime.date.today().isoformat()
        with open(cached, "w", encoding="utf-8") as fh:
            fh.write(text)
        with open(stamp, "w") as fh:
            fh.write(retrieved + "\n")
        return Page(z, url, text, retrieved)
    raise LookupError(f"no page for Z={z}: " + "; ".join(tried))


# --- page structure -------------------------------------------------------

_ANCHOR_RE = re.compile(r'<a\s+name\s*=\s*"([^"]+)"', re.IGNORECASE)
_PRE_RE = re.compile(r"<pre[^>]*>(.*?)</pre>", re.IGNORECASE | re.DOTALL)


def parse_sections(text):
    """{anchor name: text of the first <pre> block after it}."""
    sections = {}
    anchors = [(m.start(), m.group(1)) for m in _ANCHOR_RE.finditer(text)]
    for i, (pos, name) in enumerate(anchors):
        end = anchors[i + 1][0] if i + 1 < len(anchors) else len(text)
        pre = _PRE_RE.search(text, pos, end)
        if pre:
            raw = re.sub(r"<[^>]+>", "", pre.group(1))
            sections[name] = html.unescape(raw).replace("\r\n", "\n").replace("\r", "\n")
    return sections


def find_section(page, set_name):
    """(anchor as printed, text) of a set on the page, matching the anchor
    case-insensitively (the site writes both X_POB_TZVP_rev2 and
    X_pob_TZVP_rev2). None when the page has no such set."""
    wanted = set_name.lower()
    for name, text in page.sections.items():
        if name.lower() == wanted:
            return name, text
    return None


# --- grammar cutter -------------------------------------------------------

def dois(trailing):
    """DOIs exactly as printed on the page, in page order."""
    seen = []
    for m in DOI_RE.finditer(trailing):
        doi = m.group(1).rstrip(".")
        if doi not in seen:
            seen.append(doi)
    return seen


def get_set(page, set_name):
    """Everything the library stores for one set on one page, or raises
    LookupError / GrammarError with the reason."""
    found = find_section(page, set_name)
    if found is None:
        raise LookupError(f"{page.url} has no set named {set_name}")
    anchor, text = found
    record, trailing = cut_record(text)
    label, nshl, core = describe_record(record)
    return {
        "anchor": anchor, "record": record, "reference": trailing,
        "dois": dois(trailing), "label": label, "nshl": nshl, "core": core,
        "url": f"{page.url}#{anchor}", "retrieved": page.retrieved,
        "page_sha256": page.sha256,
    }


def ascii_text(text):
    """One line of ASCII for a library header: whitespace collapsed, dashes
    and quotes folded, anything else non-ASCII dropped."""
    import unicodedata

    folded = (text.replace("–", "-").replace("—", "-")
              .replace("‘", "'").replace("’", "'")
              .replace("“", '"').replace("”", '"'))
    folded = unicodedata.normalize("NFKD", folded).encode("ascii", "ignore").decode("ascii")
    return " ".join(folded.split())


def check_ecp_set(z, found):
    """Why a fetched ECP set cannot stand in for MACE's record of element z,
    or None when it can. MACE writes 200 + Z in the geometry for every
    element in ECP_ELEMENTS_EXTERNAL, so the record's label must be 200 + Z
    and the record must carry an ECP."""
    if found["label"] != 200 + z:
        return f"label {found['label']} is not {200 + z} (200 + Z, what MACE writes in the geometry)"
    if found["core"] == "all-electron":
        return "record has no ECP"
    try:
        found["record"].encode("ascii")
    except UnicodeEncodeError:
        return "record is not ASCII"
    return None
