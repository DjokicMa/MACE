"""One place makes 3D viewers (polish V3): `$3Dmol.createViewer` is called
only inside app.js viewer(), which probes WebGL first and returns null
without it; every other module goes through viewer()."""
import re

from conftest import REPO_ROOT

STATIC = REPO_ROOT / "mace" / "gui" / "static"


def test_create_viewer_only_in_viewer():
    hits = []
    for p in sorted(STATIC.rglob("*.js")):
        rel = p.relative_to(STATIC).as_posix()
        if rel.startswith("vendor/"):
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if "createViewer(" in line:
                hits.append((rel, i))
    assert len(hits) == 1 and hits[0][0] == "app.js", hits
    app = (STATIC / "app.js").read_text()
    body = app[app.index("function viewer(id) {"):]
    body = body[:body.index("\n}\n")]
    assert "createViewer(" in body and "webgl().ok" in body
