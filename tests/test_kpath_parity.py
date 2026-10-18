"""The k-path edits are the same in the GUI (mace/gui/static/js/kpath.js) and in
opt2d3 (Crystal_d3/d3_band_path.py): each sequence runs through both, the JSON must match."""
import json
import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "Crystal_d3"))
import d3_band_path as kp  # noqa: E402

JS = REPO_ROOT / "mace" / "gui" / "static" / "js" / "kpath.js"
GRAPHITE = ["GAMMA", "M", "K", "GAMMA", "A", "L", "H", "A", "|", "L", "M", "|", "H", "K"]
SLIP = ["GAMMA", "Y", "F_0", "|", "DELTA_0", "GAMMA", "Z", "B_0", "|", "G_0", "T", "Y", "|", "GAMMA", "S", "R", "Z", "T"]
FCC_K = [[0, 0, 0, .5, 0, .5], [.5, 0, .5, .625, .25, .625], [.375, .375, .75, 0, 0, 0], [0, 0, 0, .5, .5, .5]]
FCC = ["GAMMA", "X", "U", "|", "K", "GAMMA", "L"]

CASES = [
    (GRAPHITE, None, []),
    (GRAPHITE, None, [["toggle_hidden", 2], ["reverse", 0], ["move_piece", 1, 0], ["remove_point", 5, "to", False]]),
    (GRAPHITE, None, [["jumps_first"], ["move", 0, 4], ["remove", 3]]),
    (GRAPHITE, None, [["continuous_only"], ["reverse", 6], ["remove_point", 2, "to", True]]),
    (GRAPHITE, None, [["jumps_last"], ["toggle_hidden", 8], ["toggle_hidden", 8]]),
    (SLIP, None, [["move_piece", 3, 1], ["remove_point", 0, "from", False], ["jumps_first"]]),
    (FCC, FCC_K, [["add_point", "P", [0.25, 0.25, 0.25]], ["reverse", 4], ["remove_point", 0, "to", True]]),
    (FCC, FCC_K, [["add_point", "Q", [0.1, 0.2, 0.3], "GAMMA", [0, 0, 0]], ["move", 4, 0], ["continuous_only"]]),
]


def _node(cases):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    script = f"""
import * as K from {json.dumps(JS.as_uri())};
const cases = {json.dumps(cases)};
const out = cases.map(([t, ks, ops]) => {{
  const r = K.applyOps(K.fromTokens(t, ks), ops);
  const v = K.validate(r, K.fromTokens(t, ks));
  return {{segs: r, tokens: K.pathTokens(r), pieces: K.pieces(r), errors: v.errors.length, warnings: v.warnings.length,
           same: K.samePath(r, K.fromTokens(t, ks))}};
}});
console.log(JSON.stringify(out));
"""
    p = subprocess.run([node, "--input-type=module", "-e", script], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


def test_python_and_browser_edits_agree():
    got = _node(CASES)
    for (tokens, ks, ops), js in zip(CASES, got):
        segs = kp.apply_ops(kp.from_tokens(tokens, ks), ops)
        errors, warnings = kp.validate(segs, kp.from_tokens(tokens, ks))
        assert js["segs"] == json.loads(json.dumps(segs)), ops
        assert js["tokens"] == kp.path_tokens(segs) and js["pieces"] == kp.pieces(segs), ops
        assert (js["errors"], js["warnings"]) == (len(errors), len(warnings)), ops
        assert js["same"] == kp.same_path(segs, kp.from_tokens(tokens, ks)), ops


def test_labels_agree():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    labels = ["Γ", "G", "gamma", "F<sub>0</sub>", "$Δ_{0}$", "X'", "Σ₁", "K"]
    script = f"import * as K from {json.dumps(JS.as_uri())}; console.log(JSON.stringify({json.dumps(labels)}.map(K.canonical)))"
    p = subprocess.run([node, "--input-type=module", "-e", script], capture_output=True, text=True, timeout=60)
    assert json.loads(p.stdout) == [kp.canonical_label(x) for x in labels]


def test_slab_and_polymer_rules_agree():
    """A point with a non-zero non-periodic k component (manual p. 309, note 3)."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    base = kp.from_tokens(["GAMMA", "M"], [[0, 0, 0, .5, 0, 0]])
    cases = [(kp.add_point(base, "Q", k), dim) for k in ([.25, .25, .5], [.25, .25, 0], [.25, 0, 0])
             for dim in (3, 2, 1)]
    script = f"""
import * as K from {json.dumps(JS.as_uri())};
const base = {json.dumps(base)};
console.log(JSON.stringify({json.dumps(cases)}.map(([s, d]) => K.validate(s, base, false, d).errors)));
"""
    p = subprocess.run([node, "--input-type=module", "-e", script], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    got = json.loads(p.stdout)
    assert any(got)
    for (segs, dim), js in zip(cases, got):
        assert js == kp.validate(segs, base, dimensionality=dim)[0], (segs, dim)
