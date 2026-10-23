"""/api/parse through the real `mace gui` (plan T9, B8): a dropped file or
pasted text is sniffed and read without ever being written into --dir, its
temporary copy in the server's cache is gone when the answer arrives, a
body over 50 MB is refused before it is read, a plain XYZ gets a box only
when asked, and unreadable text says why. Fixtures are the demo CIFs
rewritten by ase into tmp."""
import http.client
import io
import warnings
from urllib.parse import quote, urlsplit

import pytest

import guiharness as H
import p4util as U

ase_io = pytest.importorskip("ase.io")
pytest.importorskip("numpy")


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("parse")
    root, xdg = H.examples_copy(tmp), tmp / "xdg"
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_NO_WARMUP="1", MACE_GUI_TEST_HOOKS="1")
    yield base, root
    proc.terminate()
    proc.wait(timeout=15)


def parse(base, name, data: bytes, box=True, label=None):
    h = {"Content-Type": "application/octet-stream", "X-File-Name": quote(name)}
    if label:
        h["X-Mace-Label"] = quote(label)
    return U.req(base, f"api/parse?box={1 if box else 0}", raw=data, headers=h)


def as_text(name, fmt, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = ase_io.read(str(H.EXAMPLES / "structures" / name))
        buf = io.BytesIO() if fmt == "cif" else io.StringIO()
        ase_io.write(buf, a, format=fmt, **kw)
    v = buf.getvalue()
    return a, v if isinstance(v, bytes) else v.encode()


def test_parse_never_writes_dir_or_leaves_temp(served):
    base, root = served
    before = U.tree_md5(root)
    files0 = U.req(base, "api/files")[2]
    data = (H.EXAMPLES / "structures" / "NaCl.cif").read_bytes()
    st, _, r = parse(base, "NaCl_copy.cif", data)
    assert st == 200 and r["session"] and r["n"] == 8 and r["format"] == "cif", r
    assert r["session"]["doc_meta"]["provenance"] == {"parent": "NaCl_copy.cif", "op": "drop", "label": "Dropped NaCl_copy.cif"}
    assert U.tree_md5(root) == before
    assert U.req(base, "api/files")[2] == files0
    assert U.req(base, "api/test/parse_tmp")[2]["files"] == []


def test_parse_413_over_50mb(served):
    """The Content-Length alone refuses it: the body is never read (a
    generator body; nothing is built in memory)."""
    base = served[0]
    u = urlsplit(base)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=30)
    c.putrequest("POST", u.path + "api/parse")
    for k, v in (("X-Mace-Token", H.TOKEN), ("X-Requested-With", "mace-gui"), ("Content-Type", "application/octet-stream"),
                 ("X-File-Name", "big.xyz"), ("Content-Length", str(51 * 2**20))):
        c.putheader(k, v)
    c.endheaders()
    try:
        c.send(b"1\n\nH 0 0 0\n" + b" " * 65536)
    except (BrokenPipeError, ConnectionResetError):
        pass
    resp = c.getresponse()
    assert resp.status == 413 and b"over 50 MB" in resp.read()
    c.close()
    assert U.req(base, "api/test/parse_tmp")[2]["files"] == []


@pytest.mark.parametrize("fmt,name,ext", [("vasp", "POSCAR", "poscar"), ("xyz", "x.txt", "xyz"),
                                          ("extxyz", "x.txt", "extxyz"), ("cif", "pasted", "cif")])
def test_sniff_formats(served, fmt, name, ext):
    """By content when the name doesn't tell (pasted text), by name when it does."""
    base = served[0]
    a, data = as_text("MgO.cif", fmt)
    st, _, r = parse(base, name, data)
    assert st == 200 and r["format"] == ext and r["n"] == len(a), r
    if ext != "xyz":
        assert r["session"]["symmetry"]["number"] == 225


def test_sniff_unit():
    from mace.gui import structures as S
    assert S.sniff("MgO.cif", b"") == "cif" and S.sniff("CONTCAR", b"") == "poscar" and S.sniff("POSCAR_relaxed", b"") == "poscar"
    assert S.sniff("a.d12", b"") == "d12" and S.sniff("run.out", b"") == "out" and S.sniff("x.extxyz", b"") == "extxyz"
    assert S.sniff("clip", b"data_x\n_cell_length_a 4\n") == "cif"
    assert S.sniff("clip", b"2\nLattice=\"4 0 0 0 4 0 0 0 4\" Properties=species:S:1:pos:R:3\nH 0 0 0\nH 0 0 1\n") == "extxyz"
    assert S.sniff("clip", b"hello world\n") is None


def test_parse_xyz_box_option(served):
    """A plain XYZ: with the box it's an editable 3D cell (10 A of vacuum
    between images); without, a molecule (1.1.6: editable too, in P1)."""
    base = served[0]
    data = b"3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n"
    st, _, r = parse(base, "water.xyz", data, box=True)
    assert st == 200 and r["session"], r
    cell = r["session"]["cell"]
    assert all(cell[k][k] >= 10.0 for k in range(3)), cell
    st, _, r = parse(base, "water.xyz", data, box=False)
    assert st == 200 and r.get("session") and not r.get("held"), r
    assert r["session"]["pbc"] == [False, False, False] and r["session"]["dimensionality"] == "MOLECULE"


def test_parse_errors_shown(served):
    base = served[0]
    st, _, r = parse(base, "broken.cif", b"data_x\n_cell_length_a 4\n_cell_length_b oops\nloop_\n_atom_site_label\n")
    assert st == 400 and r["error"].startswith("Couldn't read broken.cif as CIF: ") and "Iteration" not in r["error"], r
    st, _, r = parse(base, "notes", b"just some words\nnothing here\n")
    assert st == 400 and r.get("unknown_format") and "can't tell what format notes is" in r["error"], r
    assert parse(base, "empty.cif", b"")[0] == 400
    assert U.req(base, "api/test/parse_tmp")[2]["files"] == []


def test_parse_requires_token_host_xrw(served):
    base = served[0]
    h = {"Content-Type": "application/octet-stream", "X-File-Name": "a.xyz"}
    assert U.req(base, "api/parse", raw=b"1\n\nH 0 0 0\n", headers=h, token=False)[0] == 403
    assert U.req(base, "api/parse", raw=b"1\n\nH 0 0 0\n", headers={**h, "X-Requested-With": "x"})[0] == 403
