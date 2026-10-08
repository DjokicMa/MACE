"""The right editor group's page (`?pane=side`, groups.js): only this server's own
page may frame it; the main page still may not be framed at all."""
import guiharness as H
import p4util as U


def test_pane_page_may_be_framed_by_the_same_origin_only(tmp_path):
    root = H.examples_copy(tmp_path)
    proc, base, _ = H.start_server(root, tmp_path / "xdg", MACE_GUI_NO_WARMUP="1")
    try:
        st, hdr, _ = U.req(base, "?pane=side")
        assert st == 200 and "frame-ancestors 'self'" in hdr["Content-Security-Policy"]
        st, hdr, _ = U.req(base, "")
        assert st == 200 and "frame-ancestors 'none'" in hdr["Content-Security-Policy"]
        st, hdr, _ = U.req(base, "?pane=other")
        assert "frame-ancestors 'none'" in hdr["Content-Security-Policy"]
        st, hdr, _ = U.req(base, "?pane=side", token=False)          # signed out: the deny page, never framed
        assert st in (401, 403) and "frame-ancestors 'none'" in hdr.get("Content-Security-Policy", "frame-ancestors 'none'")
    finally:
        proc.terminate()
        proc.wait(30)
