"""Phase 3 (TODO #2): content-sniff discovery in ``mace.plotting.detect``.

FREQ ``.out`` files share the ``.out`` extension with hundreds of ordinary
SCF / OPT / BAND runs, so glob alone cannot tell them apart. Discovery must gate
on the phonon signature AND a real ``NORMAL MODES NORMALIZED`` block (the C1
eigenvector gate from the integration plan): an aborted freq run that printed the
banner but produced no modes classifies UNKNOWN, not FREQ.

Cube (``*.CUBE``) and spectra (``*IRSPEC.DAT`` / ``*RAMSPEC.DAT``) disambiguate
by extension / suffix and stay glob-only — they are exercised in the handler
tests; here we prove the generic sniff machinery + the FREQ gate against the real
corpus.
"""
from pathlib import Path

import pytest

from conftest import find_data

from mace.plotting import detect
from mace.plotting.registry import PlotKind, PlotterEntry, REGISTRY, register


def _noop_handler(files, config, out):
    return []


def _noop_configure(interactive):
    return {}


@pytest.fixture
def temp_registry():
    """Snapshot + restore REGISTRY so test registrations don't leak into the
    legacy band/DOS/structure entries other tests rely on."""
    saved = dict(REGISTRY)
    try:
        yield
    finally:
        REGISTRY.clear()
        REGISTRY.update(saved)


# --------------------------------------------------------------------------- #
# is_freq_output gate (real corpus)
# --------------------------------------------------------------------------- #

def test_is_freq_output_true_for_real_freq():
    f = find_data("FREQ/*.out", "NORMAL MODES NORMALIZED")
    assert detect.is_freq_output(str(f)) is True


def test_is_freq_output_false_for_band_scf_out():
    # *band.out are band single-points: no phonon signature.
    f = find_data("BAND/*band.out")
    assert detect.is_freq_output(str(f)) is False


def test_is_freq_output_false_for_missing_file(tmp_path):
    assert detect.is_freq_output(str(tmp_path / "nope.out")) is False


def test_is_freq_output_requires_modes_block(tmp_path):
    # Signature present but no normal-modes block => aborted run => not FREQ.
    f = tmp_path / "aborted.out"
    f.write_text(
        "CALCULATION OF PHONON FREQUENCIES AT THE GAMMA POINT\n"
        "... the run crashed before printing modes ...\n"
    )
    assert detect.is_freq_output(str(f)) is False


# --------------------------------------------------------------------------- #
# generic sniff-aware discovery / classification
# --------------------------------------------------------------------------- #

def test_discover_gates_out_files_by_sniff(temp_registry, tmp_path):
    """A FREQ .out and a non-FREQ .out in one dir: only the FREQ one is found."""
    freq = find_data("FREQ/*.out", "NORMAL MODES NORMALIZED")
    band = find_data("BAND/*band.out")
    (tmp_path / "a_freq.out").write_text(Path(freq).read_text(errors="replace"))
    (tmp_path / "b_band.out").write_text(Path(band).read_text(errors="replace"))

    register(PlotterEntry(
        kind=PlotKind.FREQ, flag="freq", label="FREQ",
        handler=_noop_handler, configure=_noop_configure,
        patterns=["*.out", "*.OUT"], sniff=detect.is_freq_output,
    ))

    found = detect.discover(str(tmp_path))
    names = {Path(p).name for p in found.get(PlotKind.FREQ, [])}
    assert names == {"a_freq.out"}


def test_classify_file_dual_entry_requires_sniff(temp_registry):
    register(PlotterEntry(
        kind=PlotKind.FREQ, flag="freq", label="FREQ",
        handler=_noop_handler, configure=_noop_configure,
        patterns=["*.out", "*.OUT"], sniff=detect.is_freq_output,
    ))
    freq = find_data("FREQ/*.out", "NORMAL MODES NORMALIZED")
    band = find_data("BAND/*band.out")
    # pattern matches both, but only the FREQ one passes the content gate
    assert detect.classify_file(str(freq)) == PlotKind.FREQ
    assert detect.classify_file(str(band)) is None
