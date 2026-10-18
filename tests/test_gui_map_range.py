"""Bug (g): the color range of a 2D map is robust, so nuclear spikes do not take the whole scale.

The GUI's fort.25 maps (js/maprange.js robustRange) and mace plotting's cube slices
(crystal_cubeviz_plotly.robust_range / slice_range) use one rule: the 1st to 99th percentile,
symmetric about 0 for signed data; --clip and --color-range (the CLI) and Plot options' Color range
(the GUI) override it.
"""
import json
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import REPO_ROOT

JS = REPO_ROOT / "mace" / "gui" / "static" / "js"


@pytest.fixture
def E():
    """The cube engine: it imports plotly, which the CI test jobs do not install."""
    return pytest.importorskip("mace.plotting.engines.crystal_cubeviz_plotly", reason="the cube engine imports plotly")


def _spiky(signed=False, n=60, seed=3):
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0 if signed else 0.05, 0.02, size=(n, n))
    if not signed:
        z = np.abs(z)
    z[n // 2, n // 2] = 500.0               # a nucleus
    if signed:
        z[3, 3] = -40.0
    return z


def test_robust_range_ignores_a_spike(E):
    z = _spiky()
    lo, hi = E.robust_range(z)
    assert hi < 1.0 and lo >= 0
    full = E.robust_range(z, full=True)
    assert full[1] == 500.0
    s = E.robust_range(_spiky(signed=True), signed=True)
    assert s[0] == -s[1] and s[1] < 1.0


def test_slice_range_density_log_signed_and_override(E):
    cube = SimpleNamespace(data_type="density", is_diverging_data=lambda: False)
    z = _spiky()
    lo, hi = E.slice_range(cube, z, log_scale=True)
    assert hi < np.log10(500) - 1                 # the spike is not the top of the scale
    spin = SimpleNamespace(data_type="spin", is_diverging_data=lambda: True)
    a, b = E.slice_range(spin, _spiky(signed=True), log_scale=False)
    assert a == -b and b < 1.0
    E._COLOR_RANGE = (-2.0, 3.0)
    try:
        assert E.slice_range(spin, _spiky(signed=True), log_scale=False) == (-2.0, 3.0)
    finally:
        E._COLOR_RANGE = None
    assert E.parse_color_range("-1e-3,2e-3") == (-1e-3, 2e-3)
    with pytest.raises(ValueError):
        E.parse_color_range("3,1")


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_gui_and_cli_take_the_same_range(E, tmp_path):
    cases = [(_spiky().tolist(), False, False), (_spiky(signed=True).tolist(), True, False),
             (_spiky(seed=7).tolist(), False, True), ([[1.0, 1.0], [1.0, 1.0]], False, False)]
    script = tmp_path / "run.mjs"
    script.write_text(f"import {{ robustRange }} from {json.dumps((JS / 'maprange.js').as_uri())};\n"
                      f"const cases = {json.dumps(cases)};\n"
                      "console.log(JSON.stringify(cases.map(([z, signed, full]) => robustRange(z, { signed, full }))));\n")
    got = json.loads(subprocess.run(["node", str(script)], capture_output=True, text=True, check=True, timeout=60).stdout)
    for (z, signed, full), g in zip(cases, got):
        assert g == pytest.approx(list(E.robust_range(z, signed=signed, full=full)), rel=1e-12, abs=1e-15)


def test_cli_passes_clip_and_color_range():
    from mace.plotting.main import create_parser
    args = create_parser().parse_args(["--cube", "x.CUBE", "--view", "slice", "--color-range=-0.01,0.01"])
    assert args.clip == 99.0 and args.color_range == "-0.01,0.01"
