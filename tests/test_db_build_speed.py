"""The database build's speed-ups store exactly what the slower code did.

* save_properties_to_database writes an output's rows in one connection and
  one transaction (it used to connect, commit and close once per row), and an
  error leaves none of that output's rows behind.
* The faster searches and parsers (fast_text, classify, the Mulliken count,
  the transport detector, the space group and dimensionality readers, the
  .DAT parsers, the directory listing cache) give the results of the code
  they replace, checked on real outputs from the test/ corpus.
"""

import re
import sqlite3

import pytest

from conftest import TEST_DATA, corpus_guard


def _corpus(pattern: str, limit: int = None):
    corpus_guard(pattern)
    if not TEST_DATA.is_dir():
        pytest.skip("test/ data corpus not present (gitignored, ~12GB)")
    files = sorted(TEST_DATA.glob(pattern))
    if not files:
        pytest.skip(f"no test/ file matching {pattern!r}")
    if limit and len(files) > limit:
        step = len(files) / limit                 # spread over the folder
        files = [files[int(i * step)] for i in range(limit)]
    return files


# ---------------------------------------------------------------- transaction

class _CountingConnection(sqlite3.Connection):
    opened = 0
    commits = 0

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        type(self).opened += 1

    def commit(self):
        type(self).commits += 1
        return super().commit()


@pytest.fixture
def counting(monkeypatch):
    real = sqlite3.connect

    def connect(*a, **kw):
        kw.setdefault('factory', _CountingConnection)
        return real(*a, **kw)

    monkeypatch.setattr(sqlite3, 'connect', connect)
    _CountingConnection.opened = _CountingConnection.commits = 0
    return _CountingConnection


def _extractor(tmp_path):
    from mace.utils.property_extractor import CrystalPropertyExtractor
    ex = CrystalPropertyExtractor(db_path=str(tmp_path / "materials.db"))
    ex.db  # created now, not inside the counted save
    return ex


def _props(n=12, calc_id=None):
    props = {f'test_prop_{i}': float(i) / 3 for i in range(n)}
    props['test_text'] = 'abc'
    props['test_list'] = [1, 2]
    props['_metadata'] = {'material_id': 'mat1', 'calc_id': calc_id,
                          'extracted_at': '2026-01-01T00:00:00', 'output_file': 'x/mat1.out'}
    return props


def _rows(tmp_path):
    with sqlite3.connect(str(tmp_path / "materials.db")) as conn:
        return conn.execute("SELECT property_name, property_value, property_value_text, calc_id "
                            "FROM properties ORDER BY property_name").fetchall()


@pytest.mark.parametrize("calc_id", [None, "calc-1"])
def test_one_connection_and_one_commit_per_output(tmp_path, counting, calc_id):
    ex = _extractor(tmp_path)
    counting.opened = counting.commits = 0
    props = _props(calc_id=calc_id)
    assert ex.save_properties_to_database(props) == 14
    assert (counting.opened, counting.commits) == (1, 1)
    assert len(_rows(tmp_path)) == 14

    # extracting again updates the same rows, again in one transaction
    counting.opened = counting.commits = 0
    props['test_prop_0'] = 7.0
    assert ex.save_properties_to_database(props) == 14
    assert (counting.opened, counting.commits) == (1, 1)
    rows = _rows(tmp_path)
    assert len(rows) == 14
    assert ('test_prop_0', 7.0, '7.0', calc_id) in rows


def test_an_error_leaves_none_of_the_outputs_rows(tmp_path, monkeypatch):
    ex = _extractor(tmp_path)
    real = ex._get_property_unit
    calls = []

    def unit(name):
        calls.append(name)
        if len(calls) == 5:
            raise RuntimeError("boom")
        return real(name)

    monkeypatch.setattr(ex, '_get_property_unit', unit)
    with pytest.raises(RuntimeError):
        ex.save_properties_to_database(_props())
    assert len(calls) == 5
    assert _rows(tmp_path) == []          # the four rows before it are rolled back


def test_a_failing_statement_is_undone_alone(tmp_path):
    """Inside transaction(), each _get_connection is a savepoint: a block that
    raises loses only its own writes, as its own connection's rollback did."""
    ex = _extractor(tmp_path)
    db = ex.db
    sql = ("INSERT INTO properties (material_id, property_category, property_name, extracted_at) "
           "VALUES ('m', 'c', ?, 't')")
    with db.transaction():
        with db._get_connection() as conn:
            conn.execute(sql, ('kept',))
        with pytest.raises(sqlite3.OperationalError):
            with db._get_connection() as conn:
                conn.execute(sql, ('undone',))
                conn.execute("INSERT INTO no_such_table VALUES (1)")
        with db._get_connection() as conn:
            conn.execute(sql, ('kept too',))
    assert [r[0] for r in _rows(tmp_path)] == ['kept', 'kept too']


def test_transaction_rolls_back_on_error(tmp_path):
    ex = _extractor(tmp_path)
    db = ex.db
    with pytest.raises(ValueError):
        with db.transaction():
            with db._get_connection() as conn:
                conn.execute("INSERT INTO properties (material_id, property_category, property_name, "
                             "extracted_at) VALUES ('m', 'c', 'p', 't')")
            raise ValueError
    assert _rows(tmp_path) == []
    # and the database takes writes again afterwards
    with db._get_connection() as conn:
        conn.execute("INSERT INTO properties (material_id, property_category, property_name, "
                     "extracted_at) VALUES ('m', 'c', 'p', 't')")
    assert len(_rows(tmp_path)) == 1


# ------------------------------------------------------------- fast searches

def test_ci_search_falls_back_on_non_ascii_text():
    from mace.utils.fast_text import ci_findall, ci_search
    # re.IGNORECASE folds U+017F (long s) to "s": no ASCII prefilter may skip it
    text = "xſPACE GROUP NUMBER 12\nspace group number 13\n"
    pat = r'SPACE GROUP NUMBER\s+(\d+)'
    assert ci_search(pat, text, ('space group number',)).span() == re.search(pat, text, re.I).span()
    assert ci_findall(pat, text, ('space group number',)) == re.findall(pat, text, re.I)
    assert ci_search(pat, "nothing here", ('space group number',)) is None
    assert ci_findall(pat, "Space Group Number 4 space group number 5", ('space group number',)) == ['4', '5']


_OUT_SAMPLE = 60


@pytest.fixture(scope="module")
def outputs():
    return [(p, p.read_text(errors='ignore')) for p in _corpus("**/*.out", limit=_OUT_SAMPLE)]


def _prefiltered_patterns():
    """(pattern, literals, flags) of every ci_search/ci_findall site: direct
    calls, (pattern, literals) tables, and the dict-held patterns."""
    import ast
    import inspect
    from mace.utils import formula_extractor, property_extractor

    found = set()
    for mod in (property_extractor, formula_extractor):
        for node in ast.walk(ast.parse(inspect.getsource(mod))):
            pair = None
            if isinstance(node, ast.Call) and getattr(node.func, 'id', None) in ('ci_search', 'ci_findall'):
                pair = node.args[0], node.args[2]
            elif isinstance(node, ast.Tuple) and len(node.elts) == 2:
                pair = node.elts
            if pair is None:
                continue
            try:
                pattern, literals = ast.literal_eval(pair[0]), ast.literal_eval(pair[1])
            except ValueError:
                continue
            if isinstance(pattern, str) and isinstance(literals, tuple) and literals \
                    and all(isinstance(x, str) for x in literals):
                found.add((pattern, literals, re.IGNORECASE))
    src = inspect.getsource(property_extractor)
    for name, lits in property_extractor._SCF_INFO_LITERALS.items():
        found.add((re.search(rf"'{name}': r'([^']*)'", src).group(1), lits, re.IGNORECASE | re.DOTALL))
    for key in ('ilasize', 'intgpack', 'madelimit', 'poleordr'):
        found.add((re.search(rf"'scf_{key}': r'([^']*)'", src).group(1), (key,), re.IGNORECASE))
    return sorted(found, key=str)


def test_property_extractor_searches_match_re(outputs):
    """Every ci_search/ci_findall of the extractors returns what the
    re.search/re.findall with re.IGNORECASE it replaced returns."""
    from mace.utils import fast_text
    patterns = _prefiltered_patterns()
    assert len(patterns) >= 25   # 1.1.6 removed the dead ones (memory, CPU time, DOS range, ...)
    for pattern, literals, flags in patterns:
        assert all(lit == lit.lower() for lit in literals), pattern
        assert re.match(re.escape(literals[0]), pattern.replace('\\', ''), re.I), pattern   # leads the pattern
    for path, text in outputs:
        for pattern, literals, flags in patterns:
            got, want = fast_text.ci_search(pattern, text, literals, flags), re.search(pattern, text, flags)
            assert (got and (got.span(), got.groups())) == (want and (want.span(), want.groups())), \
                (path.name, pattern)
            assert fast_text.ci_findall(pattern, text, literals, flags) == re.findall(pattern, text, flags), \
                (path.name, pattern)


def test_space_group_from_content_matches_reading_the_file(outputs):
    from mace.utils.formula_extractor import extract_space_group_from_output
    for path, _ in outputs:
        content = open(path).read()
        assert extract_space_group_from_output(path, content) == extract_space_group_from_output(path)


def test_detectors_match_their_regex(outputs):
    from mace.phonopy_bridge.parent import output_dimensionality
    from mace.utils import calc_detection, crystal_errors, ion_fragments

    def old_count(text):
        count = None
        for m in ion_fragments._MULLIKEN_COUNT.finditer(text):
            if m.group(1) == "ALPHA-BETA":
                continue
            try:
                count = float(m.group(2))
            except ValueError:
                continue
        return count

    def old_classify(text):
        lines = text.splitlines()
        lowered = [l.lower() for l in lines]
        for cls in crystal_errors.ERROR_CLASSES:
            for pattern in cls.patterns:
                for raw, low in zip(lines, lowered):
                    if pattern.lower() in low:
                        return cls, pattern, raw.strip()
        return None

    def old_dimensionality(text):
        for marker, dim in ((" CRYSTAL CALCULATION", 3), (" SLAB CALCULATION", 2),
                            (" POLYMER CALCULATION", 1), (" MOLECULAR CALCULATION", 0),
                            (" HELIX CALCULATION", 1)):
            if re.search(r"^" + marker + r"\s*$", text, re.M):
                return dim
        m = re.search(r"^\s*([0-3])D - ", text, re.M)
        return int(m.group(1)) if m else None

    texts = list(outputs) + [(p, p.read_text(errors='ignore')) for p in _corpus("TRANSPORT/*.out")]
    for path, text in texts:
        assert ion_fragments.mulliken_electron_count(text) == old_count(text), path.name
        assert calc_detection.is_transport_output(text) == \
            (calc_detection._TRANSPORT_OUTPUT_LINE.search(text) is not None), path.name
        m = crystal_errors.classify(text)
        assert (m and (m.cls, m.pattern, m.line)) == old_classify(text), path.name
        assert output_dimensionality(text) == old_dimensionality(text), path.name
    assert any(calc_detection.is_transport_output(t) for _, t in texts)


def test_mulliken_windows_on_crafted_layouts():
    """Edge layouts of the windowed search: two prints on one line, the
    ALPHA+BETA line right before, a phrase without a count, the phrase on
    the first line."""
    from mace.utils import ion_fragments
    rx = ion_fragments._MULLIKEN_COUNT
    texts = [
        " MULLIKEN POPULATION ANALYSIS - NO. OF ELECTRONS   10.0 MULLIKEN POPULATION ANALYSIS NO. OF ELECTRONS 12.0\n",
        "\n ALPHA+BETA ELECTRONS\n MULLIKEN POPULATION ANALYSIS - NO. OF ELECTRONS  8.0\n",
        " ALPHA-BETA ELECTRONS\n MULLIKEN POPULATION ANALYSIS - NO. OF ELECTRONS  2.0\nx",
        " MULLIKEN POPULATION ANALYSIS\n MULLIKEN POPULATION ANALYSIS NO. OF ELECTRONS 3",
        "MULLIKEN POPULATION ANALYSIS NO. OF ELECTRONS 4\n ALPHA+BETA ELECTRONS\n",
    ]
    for text in texts:
        assert [m.span() for m in ion_fragments._mulliken_matches(text)] == \
            [m.span() for m in rx.finditer(text)], text


# ------------------------------------------------------------- .DAT parsers

def _old_doss_records(tokens, width):
    energy, total, proj = [], [], {}
    for i in range(0, len(tokens), width):
        rec = tokens[i:i + width]
        energy.append(rec[0])
        total.append(rec[-1])
        for j, val in enumerate(rec[1:-1], start=1):
            proj.setdefault(f'projected_dos_{j}', []).append(val)
    return energy, total, proj


def test_doss_dat_parse_matches_the_record_loop():
    from mace.utils.dat_file_processor import DatFileProcessor
    proc = DatFileProcessor()
    for path in _corpus("DOSS/*.DOSS.DAT", limit=12):
        content = path.read_text()
        res = proc._parse_doss_dat_content(content)
        tokens = []
        for raw in content.split('\n'):
            line = raw.strip()
            if not line or line[0] in '#@':
                continue
            try:
                tokens.extend(float(x) for x in line.split())
            except ValueError:
                continue
        width = len(res['projected_dos']) + 2
        if res['energy_points']:
            energy, total, proj = _old_doss_records(tokens, width)
            assert res['energy_points'] == energy
            assert res['total_dos'] == total
            assert res['projected_dos'] == proj
            assert list(res['projected_dos']) == list(proj)


def test_band_dat_parse_matches_float_per_token():
    from mace.utils.dat_file_processor import DatFileProcessor
    proc = DatFileProcessor()
    for path in _corpus("BAND/*.BAND.DAT", limit=8):
        content = path.read_text()
        res = proc._parse_band_dat_content(content)
        rows = []
        for raw in content.split('\n'):
            line = raw.strip()
            if not line or line[0] in '#@':
                continue
            try:
                rows.append([float(x) for x in line.split()])
            except ValueError:
                continue
        if res['eigenvalues'] and all(len(r) == len(rows[0]) for r in rows):
            assert res['k_points'] == [r[0] for r in rows]
            assert res['eigenvalues'] == [r[1:] for r in rows]


# ----------------------------------------------------- directory listing cache

def test_sorted_names_is_sorted_iterdir_and_never_stale(tmp_path, monkeypatch):
    import os
    import time
    from mace.utils import property_extractor as pe
    for d in {p.parent for p in _corpus("*/*.DAT", limit=4)}:
        assert pe._sorted_names(d) == [p.name for p in sorted(d.iterdir())]
        assert pe._sorted_names(d) == [p.name for p in sorted(d.iterdir())]   # from the cache
    (tmp_path / "b.BAND.DAT").write_text("")
    (tmp_path / "a.out").write_text("")
    old = time.time_ns() - 10 ** 10
    os.utime(tmp_path, ns=(old, old))
    assert pe._sorted_names(tmp_path) == ["a.out", "b.BAND.DAT"]
    (tmp_path / "a.BAND.DAT").write_text("")            # changes the folder's mtime
    assert pe._sorted_names(tmp_path) == ["a.BAND.DAT", "a.out", "b.BAND.DAT"]
