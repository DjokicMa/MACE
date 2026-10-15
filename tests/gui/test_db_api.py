"""materials.db paged on the server (plan T7, B6; spec 8.4.7): keyset pages that
are stable under ties, every filter, injection-safe search and sort, the file
never written, and the export the CLI's own columns."""
import csv
import hashlib
import io
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U
import dbgen


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dbapi")
    root = H.examples_copy(tmp)
    dbgen.make(root / "big.db", 2500, calcs_per=3, props_per=6)
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1")
    yield root, base, tmp
    proc.terminate()
    proc.wait(timeout=15)


def pages(base, path, **params):
    out, after = [], None
    for _ in range(200):
        q = {"path": path, **params}
        if after:
            q["after"] = after
        st, _, r = U.req(base, "api/db/page?" + "&".join(f"{k}={__import__('urllib.parse').parse.quote(str(v))}" for k, v in q.items()))
        assert st == 200, r
        out += r["rows"]
        after = r["next"]
        if not after:
            return out, r
    raise AssertionError("too many pages")


def test_pages_cover_every_row_once_in_order(served):
    root, base, _ = served
    rows, last = pages(base, "big.db")
    ids = [r["material_id"] for r in rows]
    assert len(ids) == len(set(ids)) == last["total"] == 2500
    assert ids == sorted(ids)


@pytest.mark.parametrize("sort", ["formula", "space_group", "dimensionality", "updated_at", "n_calcs"])
@pytest.mark.parametrize("direction", ["asc", "desc"])
def test_paging_is_stable_under_ties(served, sort, direction):
    """Formula has a dozen values for 2500 rows: keyset on (value, material_id) loses nothing, repeats nothing."""
    root, base, _ = served
    rows, last = pages(base, "big.db", sort=sort, dir=direction)
    ids = [r["material_id"] for r in rows]
    assert len(ids) == len(set(ids)) == 2500
    key = lambda r: ((r[sort] if r[sort] is not None else (-1 if sort == "space_group" else "")), r["material_id"])  # noqa: E731
    want = sorted(rows, key=key, reverse=direction == "desc")
    assert [r["material_id"] for r in rows] == [r["material_id"] for r in want]


def test_a_page_is_small(served):
    root, base, _ = served
    import urllib.request
    req = urllib.request.Request(base + "api/db/page?path=big.db", headers={"X-Mace-Token": H.TOKEN, "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req) as r:
        body = r.read()
    assert len(body) <= 30 * 1024, len(body)


def test_jump_to_a_row_by_index(served):
    root, base, _ = served
    st, _, r = U.req(base, "api/db/page?path=big.db&index=2400&sort=material_id")
    assert r["rows"][0]["material_id"] == "mat002400"
    st, _, r = U.req(base, "api/db/page?path=big.db&index=2490&sort=n_calcs")
    assert len(r["rows"]) == 10


@pytest.mark.parametrize("query, check", [
    ("q=mat00012", lambda r: r["material_id"].startswith("mat00012")),
    ("q=Mg4", lambda r: "Mg4" in r["formula"]),
    ("dim=SLAB,POLYMER", lambda r: r["dimensionality"] in ("SLAB", "POLYMER")),
    ("sg_min=200&sg_max=225", lambda r: 200 <= (r["space_group"] or 0) <= 225),
    ("calc_type=BAND", None), ("calc_status=failed", None), ("prop=band_gap", None),
])
def test_every_filter(served, query, check):
    root, base, _ = served
    st, _, r = U.req(base, f"api/db/page?path=big.db&{query}&limit=200")
    assert st == 200 and r["rows"], r
    if check:
        assert all(check(x) for x in r["rows"])
    st, _, c = U.req(base, f"api/db/count?path=big.db&{query}")
    assert c["count"] == r["total"] and 0 < c["count"] <= 2500
    # the filters agree with the database itself
    import sqlite3
    conn = sqlite3.connect(f"file:{root / 'big.db'}?mode=ro", uri=True)
    if "calc_type=BAND" == query:
        n = conn.execute("SELECT COUNT(DISTINCT material_id) FROM calculations WHERE calc_type='BAND'").fetchone()[0]
        assert c["count"] == n
    if "prop=band_gap" == query:
        n = conn.execute("SELECT COUNT(DISTINCT material_id) FROM properties WHERE property_name='band_gap'").fetchone()[0]
        assert c["count"] == n
    conn.close()


@pytest.mark.parametrize("q", ["' OR 1=1 --", "%", "_", "mat%' UNION SELECT * FROM properties --", "\\", '"; DROP TABLE materials; --'])
def test_search_is_a_parameter_with_escaped_wildcards(served, q):
    root, base, _ = served
    import urllib.parse
    st, _, r = U.req(base, "api/db/page?path=big.db&q=" + urllib.parse.quote(q))
    assert st == 200 and r["total"] == 0, (q, r["total"])


@pytest.mark.parametrize("sort", ["material_id; DROP TABLE materials", "formula --", "x", "(SELECT 1)"])
def test_sort_is_a_whitelisted_column(served, sort):
    root, base, _ = served
    import urllib.parse
    st, _, r = U.req(base, "api/db/page?path=big.db&sort=" + urllib.parse.quote(sort))
    assert st == 400


def test_reads_never_change_the_file(served):
    root, base, _ = served
    before = hashlib.sha256((root / "big.db").read_bytes()).hexdigest()
    for path in ("api/db/stats?path=big.db", "api/db/props?path=big.db", "api/db/page?path=big.db&sort=n_calcs&dir=desc",
                 "api/db/count?path=big.db&q=mat", "api/db/material?path=big.db&id=mat000007",
                 "api/db/export?path=big.db&format=csv&dim=SLAB"):
        assert U.req(base, path)[0] == 200, path
    assert hashlib.sha256((root / "big.db").read_bytes()).hexdigest() == before
    assert not (root / "big.db-wal").exists() and not (root / "structures.db").exists()


def test_every_query_is_paged(served):
    root, base, _ = served
    U.req(base, "api/db/page?path=big.db&sort=formula&q=Mg")
    st, _, log = U.req(base, "api/test/dblog")
    # the table's queries (an export streams on purpose; counts are one number)
    sel = [s for s in log["log"] if s.lstrip().upper().startswith("SELECT") and " FROM MATERIALS M" in s.upper()
           and "COUNT(" not in s.upper() and not s.lstrip().upper().startswith("SELECT M.*") and "DISTINCT P.PROPERTY_NAME" not in s.upper()
           and "METADATA_JSON IS NOT NULL" not in s.upper()]
    assert sel and all("LIMIT" in s.upper() for s in sel), sel[:3]


def test_export_has_the_clis_columns(served, tmp_path):
    """The CSV header equals `mace database --action export --format csv` on a copy of the demo db."""
    root, base, _ = served
    d = tmp_path / "cli"
    d.mkdir()
    shutil.copy2(H.EXAMPLES / "materials.db", d / "materials.db")
    env = {**H.env(tmp_path / "xdg"), "NO_COLOR": "1", "MACE_NO_BANNER": "1"}
    r = subprocess.run([sys.executable, str(H.MACE_CLI), "--no-banner", "database", "--action", "export", "--format", "csv",
                        "--output", "out.csv"], cwd=str(d), capture_output=True, text=True, env=env, timeout=300, stdin=subprocess.DEVNULL)
    cli_csv = next(d.glob("*.csv"), None)
    if cli_csv is None:
        pytest.skip("mace database export wrote no CSV here: " + (r.stdout + r.stderr)[-400:])
    cli_header = next(csv.reader(io.StringIO(cli_csv.read_text())))
    st, h, body = U.req(base, "api/db/export?path=materials.db&format=csv")
    text = body.decode() if isinstance(body, bytes) else body
    ours = list(csv.reader(io.StringIO(text)))
    assert ours[0] == cli_header
    import sqlite3
    n = sqlite3.connect(f"file:{H.EXAMPLES / 'materials.db'}?mode=ro", uri=True).execute("SELECT COUNT(*) FROM materials").fetchone()[0]
    assert len(ours) - 1 == n                      # every demo material
    assert "attachment" in h["Content-Disposition"]
    st, h, body = U.req(base, "api/db/export?path=big.db&format=json&dim=SLAB")
    data = body if isinstance(body, dict) else json.loads(body)
    st, _, c = U.req(base, "api/db/count?path=big.db&dim=SLAB")
    assert len(data["data"]) == c["count"] == data["record_count"]


def _parity_db(path):
    """The demo database plus the property rows the two exports used to
    disagree on: a text value, one name written by two calculations, and a
    structure field the CLI leaves out."""
    import sqlite3
    shutil.copy2(H.EXAMPLES / "materials.db", path)
    c = sqlite3.connect(str(path))
    with c:
        mid = c.execute("SELECT material_id FROM materials ORDER BY material_id LIMIT 1").fetchone()[0]
        cols = "material_id, calc_id, property_category, property_name, property_value, property_value_text, extracted_at"
        rows = [(mid, None, "electronic", "electronic_classification", None, "semiconductor", "2026-01-01"),
                (mid, "c1", "energy", "parity_energy", -1.5, None, "2026-01-01"),
                (mid, "c2", "energy", "parity_energy", -2.5, None, "2026-01-02"),
                (mid, None, "structural", "atomic_positions", None, "[[0, 0, 0]]", "2026-01-01")]
        c.executemany(f"INSERT INTO properties ({cols}) VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    c.close()
    return mid


def test_export_values_equal_the_clis(served, tmp_path):
    """Every cell of the GUI's CSV equals the CLI's (a text-valued property
    was empty in the GUI's: it read property_value only)."""
    root, base, _ = served
    mid = _parity_db(root / "parity.db")
    d = tmp_path / "cli"
    d.mkdir()
    _parity_db(d / "materials.db")
    env = {**H.env(tmp_path / "xdg"), "NO_COLOR": "1", "MACE_NO_BANNER": "1"}
    r = subprocess.run([sys.executable, str(H.MACE_CLI), "--no-banner", "database", "--action", "export", "--format", "csv",
                        "--output", "out.csv"], cwd=str(d), capture_output=True, text=True, env=env, timeout=300, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    cli = list(csv.DictReader(io.StringIO((d / "out.csv").read_text())))
    st, h, body = U.req(base, "api/db/export?path=parity.db&format=csv")
    ours = list(csv.DictReader(io.StringIO(body.decode() if isinstance(body, bytes) else body)))
    assert list(ours[0].keys()) == list(cli[0].keys())
    by_id = lambda rows: {x["material_id"]: x for x in rows}      # noqa: E731
    assert by_id(ours) == by_id(cli)
    assert by_id(ours)[mid]["electronic_classification"] == "semiconductor"
    assert "atomic_positions" not in ours[0]


def test_material_files_resolve_inside_the_folder(served):
    root, base, _ = served
    st, _, r = U.req(base, "api/db/material?path=materials.db&id=Ag2Br3")
    c = r["calculations"][0]
    assert c["output_rel"] == "outputs/Ag2Br3_opt_unfinished.out" and c["input_rel"] == "outputs/Ag2Br3_opt_unfinished.d12"
    st, _, r = U.req(base, "api/db/material?path=big.db&id=mat000001")
    assert all(c["output_rel"] is None for c in r["calculations"])         # not in this folder


def test_material_files_are_found_after_the_project_folder_moved(served):
    """A database copied or moved with its project keeps the paths of the old
    place: a file is found under the database's folder by the end of its
    recorded path (and through the recorded work_dir for a bare name)."""
    import sqlite3
    root, base, _ = served
    moved = root / "moved.db"
    shutil.copy2(H.EXAMPLES / "materials.db", moved)
    c = sqlite3.connect(str(moved))
    with c:
        old = "/nonexistent/old/place/project"
        c.execute("UPDATE calculations SET output_file = ? || '/' || output_file, "
                  "input_file = 'Ag2Br3_opt_unfinished.d12', work_dir = ? || '/outputs' "
                  "WHERE material_id = 'Ag2Br3'", (old, old))
        row = c.execute("SELECT output_file FROM calculations WHERE material_id = 'Ag2Br3'").fetchone()
    c.close()
    assert row[0].startswith("/nonexistent/")
    st, _, r = U.req(base, "api/db/material?path=moved.db&id=Ag2Br3")
    calc = r["calculations"][0]
    assert calc["output_rel"] == "outputs/Ag2Br3_opt_unfinished.out", calc
    assert calc["input_rel"] == "outputs/Ag2Br3_opt_unfinished.d12", calc


def test_export_file_name_is_safe_in_the_header(served):
    """A database whose name has quotes and spaces still gives one clean Content-Disposition."""
    root, base, _ = served
    odd = root / 'my "best" db.db'
    shutil.copy2(H.EXAMPLES / "materials.db", odd)
    from urllib.parse import quote
    st, h, _ = U.req(base, "api/db/export?format=csv&path=" + quote(odd.name))
    assert st == 200
    assert h["Content-Disposition"] == 'attachment; filename="my_best_db_export.csv"'
