"""Files read beside an opened one stay inside the served folder (1.1.6
review): a `<stem>.SCFOUT.LOG` beside an optimization and the `.out`/`.d3`
beside a BAND.DAT for the Brillouin zone view are used only when they lie in
the folder; a symbolic link to a file outside it is ignored, as the
transport-zt route and the 1.1.5 siblings already do. On the real `mace gui`
subprocess; the corpus OPT and BAND runs (test/, read only, copied)."""
import os
import shutil
import sys
import urllib.parse

import pytest

import guiharness as H
import p4util as U

sys.path[:0] = [p for p in (str(H.REPO_ROOT / "tests"),) if p not in sys.path]
import test_scf_timeline as T  # noqa: E402

BAND = H.REPO_ROOT / "test" / "BAND"


def _get(base, route, **q):
    return U.req(base, f"api/{route}?" + urllib.parse.urlencode(q))


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    if not T.OPT.is_file():
        pytest.skip("test/ corpus not present (gitignored)")
    tmp = tmp_path_factory.mktemp("siblings")
    outside = tmp / "outside"
    outside.mkdir()
    p = T._dia_with_scfout(outside)                     # outside/<stem>.out and its SCFOUT.LOG
    root = tmp / "root"
    (root / "in").mkdir(parents=True)
    (root / "link").mkdir()
    # beside it, a real copy: read
    shutil.copyfile(p, root / "in" / p.name)
    shutil.copyfile(p.with_name(p.stem + ".SCFOUT.LOG"), root / "in" / (p.stem + ".SCFOUT.LOG"))
    # beside it, a link out of the folder: not read
    shutil.copyfile(p, root / "link" / p.name)
    os.symlink(p.with_name(p.stem + ".SCFOUT.LOG"), root / "link" / (p.stem + ".SCFOUT.LOG"))
    band = sorted(BAND.glob("*.BAND.DAT"))[:1] if BAND.is_dir() else []
    bz_rel = None
    if band:
        b = band[0]
        stem = b.name[: -len(".BAND.DAT")]
        from mace.gui import results
        _, bz = results._d3_tools()
        out = bz.output_for(str(b))
        if out is not None:
            for d in ("bin", "bout"):
                (root / d).mkdir()
                shutil.copyfile(b, root / d / b.name)
            shutil.copyfile(out, root / "bin" / out.name)
            shutil.copyfile(out, outside / out.name)
            os.symlink(outside / out.name, root / "bout" / out.name)
            bz_rel = b.name
    old = os.stat(T.OPT).st_mtime - 3600                 # finished files, not ones still being written
    for f in [*root.rglob("*"), *outside.rglob("*")]:
        if f.is_file() and not f.is_symlink():
            os.utime(f, (old, old))
    proc, base, _ = H.start_server(root, tmp / "xdg")
    try:
        yield base, p.name, bz_rel
    finally:
        proc.terminate()
        proc.wait(timeout=15)


def test_scfout_log_only_inside_the_folder(srv):
    base, name, _ = srv
    st, _, res = _get(base, "out/convergence", path=f"in/{name}")
    assert st == 200 and res["scf"]["scfout"] == name[:-4] + ".SCFOUT.LOG"
    st, _, res = _get(base, "out/convergence", path=f"link/{name}")
    assert st == 200, res
    assert res["scf"]["scfout"] is None                 # the link out of the folder is not read


def test_bz_cell_only_from_an_output_inside_the_folder(srv):
    pytest.importorskip("scipy", reason="the Brillouin zone is built with scipy")
    base, _, bz_rel = srv
    if bz_rel is None:
        pytest.skip("no corpus BAND run with its output")
    st, _, res = _get(base, "bz", path=f"bin/{bz_rel}")
    assert st == 200, res
    st, _, res = _get(base, "bz", path=f"bout/{bz_rel}")
    assert st == 404, res


def test_the_parser_takes_a_sibling_check(tmp_path):
    from mace.utils import out_convergence as oc
    p = T._dia_with_scfout(tmp_path)
    assert oc.parse(p)["scf"]["scfout"] is not None
    assert oc.parse(p, sibling_ok=lambda f: False)["scf"]["scfout"] is None
    assert oc.scfout_path(p, lambda f: False) is None and oc.scfout_path(p) is not None
