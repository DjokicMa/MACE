#!/usr/bin/env python3
"""Fetch the official pob-TZVP-rev2-SOC basis sets from the CRYSTAL basis
set library (https://www.crystal.unito.it/Basis_Sets/<element>.html) into
pob-tzvp-rev2-soc.lib, the file soc_ecp.load_soc_basis reads.

Each element page lists its basis sets as an anchor (<a name="Bi_pob_TZVP_rev2_SOC">)
followed by a <pre> block: the record as CRYSTAL reads it, then free text
(authors, journal, DOI, "Note: To be used with CRYSTAL23"). The record is cut
off by the basis-set grammar (manual p. 25, and p. 84 for the ECP): "NAT NSHL",
the ECP keyword STUTSC (CRYSTAL's internal Stuttgart spin-orbit library), then
NSHL shells. The trailing text is kept as the reference, never as part of the
record.

The library file uses the "MACE basis library v1" format of the basis-library
consolidation (one file per family, metadata lines starting "#%"), so that
loader can read it as it stands. TODO: once that loader is merged, fold this
into its fetch tool (fetch_unito.py) and drop this script.

    python fetch_soc_basis.py [--cache DIR] [--out FILE]

Standard library only (urllib).
"""
import argparse
import datetime
import html
import os
import re
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from basis_file_check import basis_block_problem  # noqa: E402

BASE_URL = "https://www.crystal.unito.it/Basis_Sets/"
FAMILY = "pob-tzvp-rev2-soc"
SET_SUFFIX = "_pob_TZVP_rev2_SOC"
LIBRARY = os.path.join(HERE, FAMILY + ".lib")

# Elements whose pages carry an X_pob_TZVP_rev2_SOC set (FACTS: Rb-I, Cs, Ba,
# Hf-Bi), with the page names the site uses.
PAGES = {
    37: "rubidium", 38: "strontium", 39: "yttrium", 40: "zirconium", 41: "niobium",
    42: "molibdenum", 43: "technecium", 44: "ruthenium", 45: "rhodium", 46: "palladium",
    47: "silver", 48: "cadmium", 49: "indium", 50: "tin", 51: "antimony",
    52: "tellurium", 53: "iodine", 55: "caesium", 56: "barium", 72: "hafnium",
    73: "tantalum", 74: "tungsten", 75: "rhenium", 76: "osmium", 77: "iridium",
    78: "platinum", 79: "gold", 80: "mercury", 81: "thallium", 82: "lead",
    83: "bismuth",
}
ALTERNATE_PAGES = {42: "molybdenum", 43: "technetium", 55: "cesium"}
SYMBOLS = {37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru",
           45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn", 51: "Sb", 52: "Te",
           53: "I", 55: "Cs", 56: "Ba", 72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os",
           77: "Ir", 78: "Pt", 79: "Au", 80: "Hg", 81: "Tl", 82: "Pb", 83: "Bi"}

_ANCHOR_RE = re.compile(r'<a\s+name\s*=\s*"([^"]+)"', re.IGNORECASE)
_PRE_RE = re.compile(r"<pre[^>]*>(.*?)</pre>", re.IGNORECASE | re.DOTALL)
DOI_RE = re.compile(r"(?:DOI:?\s*|doi\.org/)(10\.\d{4,9}/[^\s\"',;<>]+)", re.IGNORECASE)


def sections(page: str) -> dict:
    """{anchor: text of the first <pre> block after it, HTML removed}."""
    found = {}
    anchors = [(m.start(), m.group(1)) for m in _ANCHOR_RE.finditer(page)]
    for i, (pos, name) in enumerate(anchors):
        end = anchors[i + 1][0] if i + 1 < len(anchors) else len(page)
        pre = _PRE_RE.search(page, pos, end)
        if pre:
            raw = re.sub(r"<[^>]+>", "", pre.group(1))
            found[name] = html.unescape(raw).replace("\r\n", "\n").replace("\r", "\n")
    return found


def cut_record(text: str):
    """(record lines, trailing text) of a <pre> block: "NAT NSHL", one ECP
    keyword line, NSHL shells ('ITYB LAT NG CHE SCAL' and, for ITYB 0, NG
    primitive lines). Raises ValueError when the block does not follow it."""
    lines = text.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    head = lines[0].split()
    nat, nshl = int(head[0]), int(head[1])
    if not 200 < nat < 300 or lines[1].strip().upper() != "STUTSC":
        raise ValueError(f"expected 'NAT NSHL' with 200 < NAT < 300 and STUTSC, got "
                         f"{lines[0].strip()!r} / {lines[1].strip()!r}")
    i = 2
    for _ in range(nshl):
        while not lines[i].strip():
            i += 1
        ityb, _lat, ng = (int(x) for x in lines[i].split()[:3])
        i += 1 + (ng if ityb == 0 else 0)
    record = [ln.rstrip() for ln in lines[:i] if ln.strip()]
    problem = basis_block_problem("\n".join(record))
    if problem:
        raise ValueError(problem)
    return record, "\n".join(lines[i:]).strip()


def get_set(page: str, z: int):
    """(anchor, record lines, trailing text) of element z's SOC set on a page."""
    wanted = (SYMBOLS[z] + SET_SUFFIX).lower()
    for name, text in sections(page).items():
        if name.lower() == wanted:
            record, trailing = cut_record(text)
            return name, record, trailing
    raise LookupError(f"no {SYMBOLS[z]}{SET_SUFFIX} on the page")


def fetch(z: int, cache: str, timeout: int = 30):
    """(url, page text) for element z, from the cache or the site."""
    names = [PAGES[z]] + ([ALTERNATE_PAGES[z]] if z in ALTERNATE_PAGES else [])
    tried = []
    for name in names:
        cached = os.path.join(cache, name + ".html")
        if os.path.isfile(cached):
            with open(cached, encoding="utf-8", errors="replace") as fh:
                return BASE_URL + name + ".html", fh.read()
    for name in names:
        url = BASE_URL + name + ".html"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError) as exc:
            tried.append(f"{url} ({exc})")
            continue
        os.makedirs(cache, exist_ok=True)
        with open(os.path.join(cache, name + ".html"), "w", encoding="utf-8") as fh:
            fh.write(text)
        return url, text
    raise LookupError("; ".join(tried))


def _ascii(text: str) -> str:
    text = text.replace("–", "-").replace("—", "-").replace("’", "'")
    return " ".join(text.encode("ascii", "replace").decode("ascii").split())


def library_text(entries, missing, retrieved: str) -> str:
    """The pob-tzvp-rev2-soc.lib text for {z: (url, anchor, record, trailing)}."""
    out = ["#%MACE-BASIS-LIBRARY 1", f"#%family {FAMILY}",
           "#%description Official X_pob_TZVP_rev2_SOC basis sets of the CRYSTAL basis set "
           "library: CRYSTAL's internal",
           "#%description STUTSC spin-orbit ECP with the valence basis published for it. "
           "\"To be used with CRYSTAL23\".",
           "#%description Opt-in for SOC decks only (\"soc_basis\": \"pob-tzvp-rev2-soc\", "
           "Crystal_d12/soc_ecp.py).",
           f"#%description Fetched by basis_sets/fetch_soc_basis.py on {retrieved}."]
    for z in sorted(entries):
        url, anchor, record, trailing = entries[z]
        body = "\n".join(record) + "\n"
        out.append(f"#%record {200 + z} bytes={len(body.encode('ascii'))}")
        out.append(f"#%  source {anchor}")
        for line in trailing.split("\n"):
            line = _ascii(line)
            if line and not line.lower().startswith("note:"):
                out.append(f"#%  reference {line}")
            elif line:
                out.append(f"#%  note {line[5:].strip()}")
        for doi in dict.fromkeys(d.rstrip(".") for d in DOI_RE.findall(trailing)):
            out.append(f"#%  doi {doi}")
        out.append(f"#%  url {url}#{anchor}")
        out.append(f"#%  retrieved {retrieved}")
        out.append(body)        # the record, then the separating newline
    for z, why in sorted(missing.items()):
        out.append(f"#%missing {200 + z} {_ascii(why)}")
    return "\n".join(out) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache", default=os.path.join(HERE, ".unito_cache"))
    parser.add_argument("--out", default=LIBRARY)
    args = parser.parse_args(argv)
    entries, missing = {}, {}
    for z in sorted(PAGES):
        try:
            url, page = fetch(z, args.cache)
            anchor, record, trailing = get_set(page, z)
            entries[z] = (url, anchor, record, trailing)
            print(f"{SYMBOLS[z]}: {anchor} ({len(record)} lines)")
        except (LookupError, ValueError) as exc:
            missing[z] = f"{SYMBOLS[z]}: not fetched ({exc})"
            print(f"{SYMBOLS[z]}: MISSING - {exc}")
    text = library_text(entries, missing, datetime.date.today().isoformat())
    with open(args.out, "w", encoding="ascii") as fh:
        fh.write(text)
    print(f"wrote {args.out}: {len(entries)} records, {len(missing)} missing")


if __name__ == "__main__":
    main()
