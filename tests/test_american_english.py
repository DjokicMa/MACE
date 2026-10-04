"""MACE's prose uses American spelling, and this keeps it that way.

The scan reads the shipped documentation (README files, INSTALLATION,
DOCUMENTATION, CHANGELOG, docs/, example configs, CITATION) whole, and the
Python sources token by token: string literals (f-string text included),
docstrings and comments. Identifiers are not prose and are not scanned.

A British spelling may stay where changing it would break something:
- a word in capitals (CRYSTAL and SLURM print "CANCELLED", "CENTRE", ...),
- a word inside a URL (UniTO names its basis-set pages after British element
  names) or joined to an identifier by "_" (saved-config keys such as
  ``unrecognised_functional``),
- a line marked ``british-ok: <reason>``,
- an entry in ALLOWED below (stored values that cannot carry a marker).
"""

import io
import os
import re
import tokenize
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

_STEMS = (r"optimis|normalis|initialis|minimis|maximis|organis|recognis|summaris|standardis|"
          r"symmetris|polaris|localis|delocalis|parametris|characteris|visualis|serialis|"
          r"customis|prioritis|specialis|utilis|equalis|stabilis|neutralis|categoris|finalis|realis|"
          r"randomis|synchronis|generalis|discretis|regularis|harmonis|crystallis|ionis|"
          r"parenthesis|idealis|emphasis|apologis|criticis|itemis|hypothesis|vectoris|tokenis|"
          r"sanitis|capitalis|memoris|authoris|digitis|mobilis|economis|rationalis|personalis|"
          r"sensitis|scrutinis|energis|magnetis|metallis|oxidis|polymeris|hybridis|modernis|"
          r"centralis|materialis|vaporis|theoris|fertilis|sterilis|homogenis|legitimis|"
          r"internationalis|compartmentalis|amortis")
BRITISH = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    r"(?:un|re|de|non|pre)?(?:" + _STEMS + r")(?:e|ed|es|ing|ation|ations|er|ers|able)"
    r"|(?:re)?(?:analys|catalys|paralys|hydrolys|electrolys|dialys)(?:e|ed|ing|er|ers)"
    r"|(?:un)?(?:colour|behaviour|neighbour|favour|honour|flavour|harbour|humour|labour|vapour|"
    r"rumour|endeavour|armour|odour|vigour|tumour|fervour|savour|parlour)[a-z]*"
    r"|(?:kilo|centi|milli|micro|nano|pico)?(?:metre|litre)s?|fibres?|theatres?|lustre"
    r"|centre|centres|centred|centring|off-centre"
    r"|(?:un|re|mis)?(?:labelled|labelling)|modelling|modelled|cancelled|cancelling"
    r"|travelled|travelling|signalling|signalled|levelled|levelling|tunnelled|tunnelling"
    r"|fuelled|fuelling|channelled|channelling|totalled|totalling|dialled|dialling|counselled"
    r"|grey|greys|greyed|greyscale|whilst|amongst|artefacts?|catalogues?|judgements?|licences?"
    r"|acknowledgements?|programmes?|defence|ageing|learnt|fulfil|fulfils|sulphur[a-z]*|caesium"
    r"|aluminium|sceptical|speciality|manoeuvr[a-z]*|cheques?"
    r")(?![A-Za-z0-9_])",
    re.IGNORECASE)

PROSE_SUFFIXES = {".md", ".sh", ".yaml", ".yml", ".cff", ".json", ".txt"}
SCAN_ROOTS = ["mace", "Crystal_d12", "Crystal_d3", "Plotting", "code", "docs", "tests", ".github"]
TOP_FILES = ["README.md", "INSTALLATION.md", "DOCUMENTATION.md", "CHANGELOG.md", "AUTHORSHIP.md",
             "CITATION.cff", "mace_cli", "mace_config.py", "setup_mace.py", "mace_examples.sh",
             "activate_mace.sh"]
# Real program output and third-party data: not MACE's prose.
SKIP_DIRS = {"__pycache__", ".git", "data", "fixtures"}
SKIP_FILES = {Path("tests/test_american_english.py")}
SKIP_PROSE_UNDER = [Path("Crystal_d12/basis_sets")]   # UniTO-named .lib files and their notes

# (file, text on the same line): stored values a marker comment cannot sit next to.
ALLOWED = [
    # the calculations table's schema text lists the stored status values
    ("mace/database/materials.py", "completed, failed, cancelled"),
    # the 1.1.5 changelog entry names that stored value
    ("CHANGELOG.md", "`cancelled`, the saved-config keys"),
]


def _files():
    for name in TOP_FILES:
        p = ROOT / name
        if p.is_file():
            yield p
    for root in SCAN_ROOTS:
        base = ROOT / root
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
            for fn in sorted(filenames):
                p = Path(dirpath) / fn
                rel = p.relative_to(ROOT)
                if rel in SKIP_FILES:
                    continue
                if p.suffix == ".py":
                    yield p
                elif p.suffix in PROSE_SUFFIXES and not any(
                        rel.is_relative_to(s) for s in SKIP_PROSE_UNDER):
                    yield p


def _in_url(line, col):
    start = max(line.rfind(c, 0, col) for c in " \t\"'(<[") + 1
    end = min([i for i in (line.find(c, col) for c in " \t\"')>]") if i >= 0] or [len(line)])
    chunk = line[start:end]
    return "://" in chunk or "www." in chunk


def _hits_in(text, first_line, lines, rel):
    out = []
    for m in BRITISH.finditer(text):
        w = m.group(0)
        if w.isupper():
            continue
        ln = first_line + text.count("\n", 0, m.start())
        line = lines[ln - 1] if 0 < ln <= len(lines) else ""
        col = line.find(w)
        if "british-ok" in line or (col >= 0 and _in_url(line, col)):
            continue
        if any(rel == a and s in line for a, s in ALLOWED):
            continue
        out.append(f"{rel}:{ln}: {w}")
    return out


_PY_KINDS = {tokenize.STRING, tokenize.COMMENT}
if getattr(tokenize, "FSTRING_MIDDLE", None) is not None:   # Python 3.12+ splits f-strings
    _PY_KINDS.add(tokenize.FSTRING_MIDDLE)


def scan_source(src, rel, python):
    lines = src.splitlines()
    if not python:
        return _hits_in(src, 1, lines, rel)
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in _PY_KINDS:
            out.extend(_hits_in(tok.string, tok.start[0], lines, rel))
    return out


def test_scanner_finds_british_words_in_strings_f_strings_and_comments():
    src = ('x = "the colour of it"  # optimised\n'
           'y = f"was {x} analysed"\n'
           'z = "OPT END CANCELLED"  # optimisation_type is a key\n'
           'u = "https://crystal.unito.it/sulphur.html"\n'
           'v = "a centre"  # british-ok: test\n')
    hits = scan_source(src, "t.py", python=True)
    assert [h.split(": ")[1] for h in hits] == ["colour", "optimised", "analysed"]


def test_scanner_reads_prose_files_whole_and_leaves_analyses_alone():
    hits = scan_source("Two analyses, one behaviour.\nCentred.\n", "t.md", python=False)
    assert hits == ["t.md:1: behaviour", "t.md:2: Centred"]


def test_scanner_covers_less_common_british_forms():
    src = ("an unparenthesised row, idealised ring, vapour, 5 nanometres,\n"
           "acknowledgements, catalysed, tunnelling, caesium\n")
    hits = [h.split(": ")[1] for h in scan_source(src, "t.md", python=False)]
    assert hits == ["unparenthesised", "idealised", "vapour", "nanometres",
                    "acknowledgements", "catalysed", "tunnelling", "caesium"]
    # American and neutral words stay quiet
    assert scan_source("parenthesis, parameters, analyses, ideal, program, parenthesized\n",
                       "t.md", python=False) == []


def test_mace_prose_uses_american_spelling():
    files = list(_files())
    assert len(files) > 100, "the scan found too few files to mean anything"
    hits = []
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        try:
            src = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        python = p.suffix == ".py" or p.name == "mace_cli"
        try:
            hits.extend(scan_source(src, rel, python))
        except (tokenize.TokenError, SyntaxError) as exc:   # pragma: no cover
            pytest.fail(f"{rel}: cannot tokenize: {exc}")
    assert not hits, "British spelling in MACE prose (use American, or mark british-ok):\n" + "\n".join(hits)
