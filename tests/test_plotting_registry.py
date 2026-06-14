"""Phase 0 (mace plotting integration): the registry spine.

New code — written test-first. The registry lets later phases (cube / FREQ /
spectra) add a visualization with a single ``register()`` call instead of
editing discovery, the interactive menu, and main() dispatch by hand.
"""
import pytest

from mace.plotting.registry import PlotKind, PlotterEntry, register, REGISTRY


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Snapshot/restore the module-global REGISTRY so tests don't leak."""
    snapshot = dict(REGISTRY)
    yield
    REGISTRY.clear()
    REGISTRY.update(snapshot)


def _entry(kind):
    return PlotterEntry(
        kind=kind,
        flag=kind.value,
        label=kind.value,
        handler=lambda files, cfg, out: list(files),
        configure=lambda interactive: {},
        patterns=["*.x"],
    )


def test_plotkind_has_the_three_legacy_kinds():
    assert {PlotKind.BAND, PlotKind.DOS, PlotKind.STRUCTURE} <= set(PlotKind)


def test_register_then_lookup_returns_same_entry():
    REGISTRY.clear()
    e = _entry(PlotKind.BAND)
    register(e)
    assert REGISTRY[PlotKind.BAND] is e


def test_registration_order_is_preserved():
    REGISTRY.clear()
    register(_entry(PlotKind.BAND))
    register(_entry(PlotKind.DOS))
    register(_entry(PlotKind.STRUCTURE))
    assert list(REGISTRY.keys()) == [PlotKind.BAND, PlotKind.DOS, PlotKind.STRUCTURE]


def test_entry_exposes_required_fields():
    e = _entry(PlotKind.DOS)
    for field_name in ("kind", "flag", "label", "handler", "configure", "patterns"):
        assert hasattr(e, field_name), f"PlotterEntry missing {field_name!r}"


def test_handler_is_callable_with_three_args():
    e = _entry(PlotKind.BAND)
    assert e.handler(["a", "b"], {}, ".") == ["a", "b"]
