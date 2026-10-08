"""Open as Text through windows (plan T5, spec 8.4.6): at most 256 KB per
request, on line boundaries; a sparse line index; Find in File as a task;
a big file's first view costs one window, and the viewer keeps ~80 rows."""
import json
import os
import time
from pathlib import Path

import pytest

import guiharness as H
import p4util as U

N_BIG = 34 * 1024 * 1024


def _gen(path: Path, size: int, marker="LAST-LINE-MARKER"):
    """A text file of about ``size`` bytes: the demo .out repeated (public data), a marker on the last line."""
    unit = (H.EXAMPLES / "outputs" / "Ag2Br3_opt_unfinished.out").read_bytes()
    with open(path, "wb") as fh:
        n = 0
        while n + len(unit) < size:
            fh.write(unit)
            n += len(unit)
        fh.write(f"the end {marker}\n".encode())
    return path


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    if not U.tmp_free_ok(1):
        pytest.skip("needs 1 GB free in /tmp")
    tmp = tmp_path_factory.mktemp("textwin")
    root = H.examples_copy(tmp)
    _gen(root / "outputs" / "big_text.out", N_BIG)
    _gen(root / "outputs" / "ten_mb.out", 10 * 1024 * 1024)
    (root / "outputs" / "blob.out").write_bytes(b"CRYSTAL\0\0\0" + os.urandom(2000))
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1")
    yield root, base, tmp
    proc.terminate()
    proc.wait(timeout=15)
    for n in ("big_text.out", "ten_mb.out"):
        (root / "outputs" / n).unlink(missing_ok=True)


def test_windows_are_line_aligned_and_capped(served):
    root, base, _ = served
    raw = (root / "outputs" / "big_text.out").read_bytes()
    st, h, w = U.req(base, "api/text?path=outputs/big_text.out&offset=1000001&limit=99999999")
    assert st == 200
    assert w["offset"] <= 1000001 and raw[w["offset"] - 1:w["offset"]] == b"\n"
    assert w["end"] - w["offset"] <= 256 * 1024 and raw[w["end"] - 1:w["end"]] == b"\n"
    assert w["text"].encode() == raw[w["offset"]:w["end"]]
    assert w["size"] == len(raw) and not w["eof"]
    st, _, w0 = U.req(base, "api/text?path=outputs/Si_sp.out")             # a small file: whole, as before
    assert w0["eof"] and w0["text"] == (root / "outputs" / "Si_sp.out").read_text()


def test_first_view_of_a_34_mb_output_is_one_window(served):
    root, base, _ = served
    import urllib.request
    total = 0
    for path in ("api/text?path=outputs/big_text.out", "api/text/lines?path=outputs/big_text.out"):
        req = urllib.request.Request(base + path, headers={"X-Mace-Token": H.TOKEN, "Accept-Encoding": "gzip"})
        with urllib.request.urlopen(req, timeout=60) as r:
            total += len(r.read())
    assert total <= 300 * 1024, total


def test_line_index_matches_the_file(served):
    root, base, _ = served
    raw = (root / "outputs" / "ten_mb.out").read_bytes()
    st, _, idx = U.req(base, "api/text/lines?path=outputs/ten_mb.out")
    assert st == 200 and idx["lines"] == raw.count(b"\n") and idx["step"] == 4096
    for k, off in enumerate(idx["offsets"]):
        assert raw[off - 1:off] == b"\n" or off == 0
    # every 4096th line starts where the index says
    lines = raw.split(b"\n")
    off = sum(len(x) + 1 for x in lines[:4096 * 3])
    assert idx["offsets"][3] == off


def test_find_in_file_streams_and_reports_lines(served):
    root, base, _ = served
    st, _, r = U.post(base, "api/text/find", {"path": "outputs/ten_mb.out", "query": "LAST-LINE-MARKER"})
    if st == 202:
        tid = r["task"]["id"]
        end = time.time() + 60
        while time.time() < end:
            st, _, r = U.req(base, f"api/tasks/{tid}/result")
            if st == 200:
                break
            time.sleep(0.1)
    assert st == 200 and len(r["matches"]) == 1
    raw = (root / "outputs" / "ten_mb.out").read_bytes()
    assert r["matches"][0]["line"] == raw.count(b"\n") - 1
    assert raw[r["matches"][0]["offset"]:].startswith(b"the end LAST-LINE-MARKER")


def test_binary_file_is_refused(served):
    root, base, _ = served
    st, _, r = U.req(base, "api/text?path=outputs/blob.out")
    assert st == 415 and "binary" in r["error"]


@pytest.mark.gui
def test_viewer_scrolls_to_the_end_and_finds_the_marker(served):
    pytest.importorskip("selenium.webdriver")
    root, base, tmp = served
    drv = H.firefox(1280, 800)
    try:
        H.load(drv, base)
        drv.execute_async_script("const done = arguments[1]; window.__mace.openAsText(arguments[0]).then(() => done(1))",
                                 "outputs/ten_mb.out")
        H.wait(drv, "const v = window.__mace.textView; return v && v.total() > 100000", 30)
        H.wait(drv, "return document.querySelectorAll('#text-view .tv-line .tv-wait').length === 0", 20)
        drv.execute_script("document.querySelector('#text-view .tv-scroll').focus()")
        from selenium.webdriver.common.keys import Keys
        U.keys(drv, Keys.CONTROL, Keys.END)
        H.wait(drv, "return [...document.querySelectorAll('#text-view .tv-line .tv-t')].some((x) => x.textContent.includes('LAST-LINE-MARKER'))", 20)
        rows = drv.execute_script("return window.__mace.textView.stats()")
        assert rows["maxRows"] <= 80, rows
        drv.execute_script("const f = document.querySelector('#text-view .tv-find'); f.value = 'LAST-LINE-MARKER';"
                           "f.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', bubbles: true}))")
        H.wait(drv, "return document.querySelector('#text-view .tv-count').textContent.includes('1 line')", 30)
        hit = drv.execute_script("const h = document.querySelector('#text-view .tv-line.hit'); return h && h.textContent")
        assert hit and "LAST-LINE-MARKER" in hit
        U.shot(drv, "text_windowed_end")
        assert not H.errors(drv), H.errors(drv)
    finally:
        drv.quit()
