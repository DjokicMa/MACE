"""Bundled external basis sets: one library file per family.

MACE ships its external basis sets in Crystal_d12/basis_sets/<family>.lib.
Each file holds every element record of one family together with where the
record came from. The format ("MACE basis library v1") is plain ASCII:

    #%MACE-BASIS-LIBRARY 1
    #%family full.basis.triplezeta
    #%description <one line of text; the line may repeat>
    #%record 237 bytes=1043
    #%  source Rb_POB_TZVP_2018
    #%  reference J. Laun, D. V. Oliveira, T. Bredow, J. Comput. Chem. 39 (2018) 1285
    #%  doi 10.1002/jcc.25195
    #%  provenance <how the record was identified>
    <exactly 1043 bytes: the record as CRYSTAL reads it>
    <one newline>
    #%missing 43 <why this family has no record for Z=43>

A record key is the file name the old per-element directories used: Z for an
all-electron record, 200 + Z for an ECP record (plus odd names such as
stuttgart's "231.SDF"). The body is stored length-prefixed and returned byte
for byte, so a record without a final newline stays without one. Lines
starting "#%  " after a record line are metadata (field, space, value); a
field may repeat. "#%missing" lines name elements the family deliberately
does not carry; asking for one raises MissingRecord with the reason.

Only paths that ARE a bundled family directory (Crystal_d12/basis_sets/<family>,
with or without a trailing slash) are served from the library. Any other path -
a user's own basis directory - is still read from disk by the callers.

Standard library only.
"""
import functools
import os

FORMAT_HEADER = "#%MACE-BASIS-LIBRARY"
FORMAT_VERSION = "1"

BASIS_SETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "basis_sets")

# Families a deck can name as its external basis directory.
FAMILIES = (
    "full.basis.triplezeta",
    "full.basis.doublezeta",
    "stuttgart",
    # Opt-in: full.basis.triplezeta with X_POB_TZVP_rev2 for Rb-I and
    # X_pob_TZVP_rev2 for La-Lu (see the file's description).
    "full.basis.triplezeta.rev2022",
)

# Data-only families: loadable by name, never mapped from a basis path, so
# they cannot be picked up as an ordinary external basis of a deck.
# pob-tzvp-rev2-soc: X_pob_TZVP_rev2_SOC, valence bases re-fitted for CRYSTAL's
# internal STUTSC library (fully relativistic, manual p.84) - for two-component
# SOC decks only (manual pp.169-175); read with soc_basis_record().
SOC_FAMILY = "pob-tzvp-rev2-soc"
DATA_ONLY_FAMILIES = (SOC_FAMILY,)


class LibraryFormatError(ValueError):
    """A basis library file does not follow the MACE basis library format."""


class MissingRecord(KeyError):
    """The family deliberately has no record for this element."""


class BasisFamily:
    """One parsed library file."""

    def __init__(self, name, description, records, provenance, missing):
        self.name = name
        self.description = description
        self._records = records
        self._provenance = provenance
        self.missing = missing

    def keys(self):
        return list(self._records)

    def __contains__(self, key):
        return str(key) in self._records

    def record_bytes(self, key):
        key = str(key)
        if key not in self._records:
            if key in self.missing:
                raise MissingRecord(f"{self.name}: no record {key}: {self.missing[key]}")
            raise KeyError(f"{self.name}: no record {key}")
        return self._records[key]

    def record(self, key):
        """The record text exactly as stored."""
        return self.record_bytes(key).decode("ascii")

    def provenance(self, key):
        """[(field, value), ...] for one record, in file order."""
        self.record_bytes(key)
        return list(self._provenance[str(key)])

    def field(self, key, name):
        """Values of one metadata field of a record (fields may repeat)."""
        return [v for f, v in self.provenance(key) if f == name]


def _parse(data, path):
    """Parse the bytes of a library file into a BasisFamily."""
    where = os.path.basename(path)

    def line_at(pos):
        end = data.find(b"\n", pos)
        if end < 0:
            raise LibraryFormatError(f"{where}: line at byte {pos} has no newline")
        try:
            return data[pos:end].decode("ascii"), end + 1
        except UnicodeDecodeError:
            raise LibraryFormatError(f"{where}: non-ASCII header at byte {pos}") from None

    first, pos = line_at(0)
    parts = first.split()
    if len(parts) != 2 or parts[0] != FORMAT_HEADER:
        raise LibraryFormatError(f"{where}: not a MACE basis library (first line {first!r})")
    if parts[1] != FORMAT_VERSION:
        raise LibraryFormatError(f"{where}: unknown library version {parts[1]!r}")

    name = None
    description = []
    records, provenance, missing = {}, {}, {}
    while pos < len(data):
        line, nxt = line_at(pos)
        if not line.strip():
            pos = nxt
            continue
        if line.startswith("#%family "):
            name = line[len("#%family "):].strip()
            pos = nxt
        elif line.startswith("#%description"):
            description.append(line[len("#%description"):].strip())
            pos = nxt
        elif line.startswith("#%missing "):
            bits = line[len("#%missing "):].strip().split(None, 1)
            if not bits:
                raise LibraryFormatError(f"{where}: empty #%missing line")
            missing[bits[0]] = bits[1] if len(bits) > 1 else "not available"
            pos = nxt
        elif line.startswith("#%record "):
            bits = line[len("#%record "):].split()
            if len(bits) != 2 or not bits[1].startswith("bytes="):
                raise LibraryFormatError(f"{where}: bad record line {line!r}")
            key = bits[0]
            try:
                size = int(bits[1][len("bytes="):])
            except ValueError:
                raise LibraryFormatError(f"{where}: bad length in {line!r}") from None
            if key in records:
                raise LibraryFormatError(f"{where}: duplicate record {key}")
            pos = nxt
            meta = []
            while pos < len(data) and data.startswith(b"#%  ", pos):
                mline, pos = line_at(pos)
                field_value = mline[4:].split(" ", 1)
                meta.append((field_value[0], field_value[1] if len(field_value) > 1 else ""))
            body = data[pos:pos + size]
            if len(body) != size or data[pos + size:pos + size + 1] != b"\n":
                raise LibraryFormatError(
                    f"{where}: record {key} is not {size} bytes followed by a newline")
            try:
                body.decode("ascii")
            except UnicodeDecodeError:
                raise LibraryFormatError(f"{where}: record {key} is not ASCII") from None
            records[key] = body
            provenance[key] = meta
            pos = pos + size + 1
        else:
            raise LibraryFormatError(f"{where}: unexpected line {line!r}")
    if name is None:
        raise LibraryFormatError(f"{where}: no #%family line")
    overlap = set(records) & set(missing)
    if overlap:
        raise LibraryFormatError(f"{where}: records also listed as missing: {sorted(overlap)}")
    return BasisFamily(name, "\n".join(description), records, provenance, missing)


def library_file(family):
    return os.path.join(BASIS_SETS_DIR, family + ".lib")


def load_file(path):
    """Parse a library file at an arbitrary path (not cached)."""
    with open(path, "rb") as fh:
        return _parse(fh.read(), path)


@functools.lru_cache(maxsize=None)
def load_family(family):
    """The parsed library of a bundled family (cached)."""
    if family not in FAMILIES and family not in DATA_ONLY_FAMILIES:
        raise KeyError(f"no bundled basis family {family!r}")
    parsed = load_file(library_file(family))
    if parsed.name != family:
        raise LibraryFormatError(
            f"{os.path.basename(library_file(family))}: names family {parsed.name!r}")
    return parsed


def _normalise(path):
    return os.path.abspath(os.path.expanduser(str(path))).rstrip("/")


def _bundled_roots():
    """Directories whose <family> subpath names a bundled family: this
    checkout's basis_sets, and the one mace_config points at when MACE_HOME
    is set (mace_config builds DEFAULT_TZ_PATH from MACE_HOME)."""
    roots = [BASIS_SETS_DIR]
    home = os.environ.get("MACE_HOME")
    if home:
        roots.append(os.path.join(home, "Crystal_d12", "basis_sets"))
    return roots


def _sibling_library(norm):
    """<path>.lib when <path> is named after a bundled family, is not itself a
    directory, and has a library file beside it - the layout of every MACE
    checkout. This keeps absolute paths into a checkout working for scripts
    copied elsewhere (mace/utils/copy_dependencies.py)."""
    if os.path.basename(norm) not in FAMILIES or os.path.isdir(norm):
        return None
    lib = norm + ".lib"
    return lib if os.path.isfile(lib) else None


def _match(path):
    """(family, library file) that serves a basis path, or None."""
    if not path:
        return None
    norm = _normalise(path)
    real = os.path.realpath(norm)
    for family in FAMILIES:
        for root in _bundled_roots():
            candidate = _normalise(os.path.join(root, family))
            if norm == candidate or real == os.path.realpath(candidate):
                # Serve the library beside the matched root: $MACE_HOME's for
                # a $MACE_HOME path. This module's own basis_sets/ is only the
                # fallback - it is absent when copy_dependencies copied the
                # scripts into a job directory.
                for lib in (os.path.join(root, family + ".lib"), library_file(family)):
                    if os.path.isfile(lib):
                        return family, lib
                return family, library_file(family)
    sibling = _sibling_library(norm)
    if sibling:
        return os.path.basename(norm), sibling
    return None


@functools.lru_cache(maxsize=None)
def _load_library_file(path):
    return load_file(path)


def _library(match):
    family, path = match
    if _normalise(path) == _normalise(library_file(family)):
        return load_family(family)
    lib = _load_library_file(path)
    if lib.name != family:
        raise LibraryFormatError(f"{path}: names family {lib.name!r}")
    return lib


def bundled_family_for(path):
    """The bundled family a basis path names, or None.

    Only an exact match counts: the path, normalised (~ expanded, made
    absolute, trailing slash dropped; symlinks resolved as a second try),
    must be <basis_sets>/<family> for one of FAMILIES - in this checkout, in
    $MACE_HOME, or any directory-less <family> path with <family>.lib beside
    it. An existing directory of that name is never redirected. Data-only
    families are never mapped."""
    match = _match(path)
    return match[0] if match else None


def path_exists(path):
    """Whether a basis directory is usable: a bundled family or a directory."""
    if bundled_family_for(path):
        return True
    return bool(path) and os.path.isdir(path)


def record_exists(directory, key):
    """Whether <directory>/<key> exists, for a bundled family or on disk."""
    match = _match(directory)
    if match:
        return str(key) in _library(match)
    return os.path.exists(os.path.join(directory, str(key)))


def read_record(directory, key):
    """Text of <directory>/<key>: from the library for a bundled family, else
    from disk. Raises FileNotFoundError like open() when there is none - for a
    bundled family's #%missing element the message carries the reason.

    Every deck writer reads its basis records here, so a record CRYSTAL could
    not read as it stands (basis_file_check: a placeholder for the shell
    count, "X" for a charge, fewer shells than announced - stuttgart 204,
    219, 220, 227, 294, say) raises BasisFileError, a ValueError naming it,
    and no deck is written with it."""
    match = _match(directory)
    if match:
        lib = _library(match)
        if str(key) in lib:
            return _readable(os.path.join(directory, str(key)), lib.record(key), key)
        reason = lib.missing.get(str(key))
        raise FileNotFoundError(
            os.path.join(directory, str(key))
            + (f" (not in {match[0]}: {reason})" if reason else ""))
    path = os.path.join(directory, str(key))
    with open(path, "r") as fh:
        return _readable(path, fh.read(), key)


def _readable(path, text, key):
    from basis_file_check import check_basis_file
    if text:
        number = str(key).split(".")[0]
        check_basis_file(path, text, int(number) if number.isdigit() else 0)
    return text


def iter_records(family):
    """(key, bytes) for every record of a family, in file order."""
    lib = load_family(family)
    for key in lib.keys():
        yield key, lib.record_bytes(key)


def soc_basis_elements():
    """Atomic numbers with an X_pob_TZVP_rev2_SOC record."""
    return sorted(int(k) - 200 for k in load_family(SOC_FAMILY).keys())


def soc_basis_record(z):
    """The X_pob_TZVP_rev2_SOC record of element z (label 200 + Z, STUTSC
    ECP), exactly as crystal.unito.it prints it. For two-component SOC decks
    only: STUTSC is the fully relativistic Stuttgart-Cologne library. Raises
    MissingRecord (with the reason) or KeyError for an element without one."""
    return load_family(SOC_FAMILY).record(str(200 + int(z)))


def shell_count(family, key):
    """NSHL of a record: the second number on its first line. CHEMOD lists
    one charge per shell, so this is what a carried CHEMOD must match."""
    first = load_family(family).record(key).split("\n", 1)[0].split()
    return int(first[1])


# --- CRYSTAL23 basis-set grammar -------------------------------------------
#
# CRYSTAL23 user's manual, "2.2.1 Standard route" (p.25) and "Effective core
# pseudo-potentials - ECP" (p.84):
#
#     NAT NSHL
#     [if NAT > 200: the ECP]
#         PSN keyword: HAYWLC HAYWSC BARTHE DURAND COLUSC COLULC COLUSH STUTSC
#                      STUTLC STUTSH (internal libraries, no further records), or
#         INPUT:  ZNUC M M0 M1 M2 M3 M4, then M+M0+M1+M2+M3+M4 records
#                 "ALFKL CGKL NKL"
#         INPSOC: PSONAME, SOSCALE, then ZNUC M M0..M4 and M+M0+..+M4 records
#                 "ALFKL CGKL CSOGKL NKL"
#     NSHL shells: "ITYB LAT NG CHE SCAL", then - only when ITYB = 0 - NG records
#                  "EXP COE1" (plus COE2 when LAT = 1, an SP shell).

ECP_KEYWORDS = {"HAYWLC", "HAYWSC", "BARTHE", "DURAND", "COLUSC", "COLULC", "COLUSH",
                "STUTSC", "STUTLC", "STUTSH"}


class GrammarError(ValueError):
    """Text does not start with a basis record CRYSTAL can read."""


def _numbers(line, count, what):
    parts = line.split()
    if len(parts) != count:
        raise GrammarError(f"{what}: expected {count} numbers, got {line!r}")
    try:
        return [float(p.replace("D", "E").replace("d", "e")) for p in parts]
    except ValueError:
        raise GrammarError(f"{what}: not numeric: {line!r}") from None


def cut_record(text):
    """Split text that starts with a basis record into (record, trailing text).

    The record is the exact lines CRYSTAL reads, each newline-terminated; the
    trailing text (authors, journal, DOI, notes) is returned stripped."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()                     # the newline ending the last line
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i == len(lines):
        raise GrammarError("empty block")
    start = i

    def take():
        nonlocal i
        if i >= len(lines):
            raise GrammarError("record ends early")
        line = lines[i]
        i += 1
        return line

    head = take().split()
    if len(head) != 2 or not all(t.isdigit() for t in head):
        raise GrammarError(f"first line is not 'NAT NSHL': {lines[start]!r}")
    nat, nshl = int(head[0]), int(head[1])
    if nat > 200:
        psn = take().strip()
        key = psn.upper()
        if key in ECP_KEYWORDS:
            pass
        elif key in ("INPUT", "INPSOC"):
            if key == "INPSOC":
                take()                               # PSONAME
                _numbers(take(), 1, "SOSCALE")
            counts = _numbers(take(), 7, "ZNUC M M0 M1 M2 M3 M4")
            per_line = 4 if key == "INPSOC" else 3
            for _ in range(int(sum(counts[1:]))):
                _numbers(take(), per_line, "ECP term")
        else:
            raise GrammarError(f"unknown pseudo-potential keyword {psn!r}")
    for _ in range(nshl):
        shell = _numbers(take(), 5, "ITYB LAT NG CHE SCAL")
        ityb, lat, ng = int(shell[0]), int(shell[1]), int(shell[2])
        if ityb == 0:
            for _ in range(ng):
                _numbers(take(), 3 if lat == 1 else 2, "primitive")
    record = "".join(line.rstrip("\n") + "\n" for line in lines[start:i])
    trailing = "\n".join(lines[i:]).strip()
    return record, trailing


def describe_record(record):
    """(label, nshl, core) of a cut record: core is 'ZNUC=<x> (INPUT ECP)',
    the keyword library name, or 'all-electron'."""
    lines = record.split("\n")
    nat, nshl = (int(t) for t in lines[0].split())
    if nat <= 200:
        return nat, nshl, "all-electron"
    psn = lines[1].strip().upper()
    if psn == "INPUT":
        return nat, nshl, f"ZNUC={lines[2].split()[0]} (INPUT ECP)"
    if psn == "INPSOC":
        return nat, nshl, f"ZNUC={lines[4].split()[0]} (INPSOC ECP)"
    return nat, nshl, f"{psn} (internal ECP library)"


def numeric_tokens(record):
    """The record as a list of per-line token lists, numbers as floats, so two
    records compare equal at printed precision whatever their spacing."""
    out = []
    for line in record.split("\n"):
        if not line.strip():
            continue
        row = []
        for tok in line.split():
            try:
                row.append(float(tok.replace("D", "E").replace("d", "e")))
            except ValueError:
                row.append(tok.upper())
        out.append(row)
    return out


def same_numbers(a, b):
    return numeric_tokens(a) == numeric_tokens(b)

