"""The Software tab's rows (mace.cluster.software), asked of Submit's
site.crystal_module() and the module discovery; both are stand-ins here (the
real ones come with other parts of MACE 1.1.5)."""
import sys
import types

from mace.cluster import software

FAKE_SITE = '''
class _Res:
    def __init__(self, refresh):
        self.refresh = refresh
    def record(self):
        return {"modules": ["CRYSTAL/23-intel-2023a"], "module": "CRYSTAL/23-intel-2023a", "source": "discovered",
                "note": "CRYSTAL/23 does not load on host1; using CRYSTAL/23-intel-2023a", "rejected": [["CRYSTAL/23", "no Pcrystal"]],
                "root_from_path": False, "other_major": None, "refresh": self.refresh}
def crystal_module(env=None, site=None, parent_module=None, refresh=False):
    return _Res(refresh)
'''

FAKE_ENVMODULES = '''
CALLS = []
def module_system(env=None):
    return "lmod"
def discover(name, binary, minimum=(), *, preferred=None, env=None, use_cache=True):
    CALLS.append((name, binary, tuple(minimum), use_cache))
    return {"module": "xtb/6.7.1-gfbf-2024a", "cached": False, "available": ["xtb/6.7.1-gfbf-2024a", "xtb/6.6.1"],
            "reason": ""}
'''


def _install(monkeypatch, envmodules=True, site=True):
    if envmodules:
        m = types.ModuleType("mace.utils.envmodules")
        exec(FAKE_ENVMODULES, m.__dict__)
        monkeypatch.setitem(sys.modules, "mace.utils.envmodules", m)
    else:
        monkeypatch.setitem(sys.modules, "mace.utils.envmodules", None)
    if site:
        s = types.ModuleType("mace.submission.site")
        exec(FAKE_SITE, s.__dict__)
        monkeypatch.setitem(sys.modules, "mace.submission.site", s)
    else:
        monkeypatch.setitem(sys.modules, "mace.submission.site", None)


def test_rows_from_both_helpers(monkeypatch):
    _install(monkeypatch)
    assert software.module_system() == "lmod" and software.available()
    r = software.rows(refresh=True)
    assert r["module_system"] == "lmod"
    cr, xtb = r["rows"]
    assert cr["code"] == "CRYSTAL" and cr["modules"] == ["CRYSTAL/23-intel-2023a"] and cr["refresh"] is True
    assert cr["rejected"] == [["CRYSTAL/23", "no Pcrystal"]]
    assert xtb["module"] == "xtb/6.7.1-gfbf-2024a" and xtb["source"] == "discovered"
    from mace.preopt.detect import XTB_MIN
    assert sys.modules["mace.utils.envmodules"].CALLS == [("xtb", "xtb", tuple(XTB_MIN), False)]


def test_hidden_without_a_module_system_or_helpers(monkeypatch):
    _install(monkeypatch, envmodules=False)
    assert software.module_system() is None and not software.available()
    assert software.rows() == {"module_system": None, "rows": []}


def test_a_failing_probe_is_a_row_not_an_error(monkeypatch):
    _install(monkeypatch)

    def boom(**kw):
        raise RuntimeError("module avail timed out")
    monkeypatch.setattr(sys.modules["mace.submission.site"], "crystal_module", boom)
    r = software.rows()
    assert r["rows"][0] == {"code": "CRYSTAL", "binaries": "Pcrystal, Pproperties",
                            "error": "RuntimeError: module avail timed out"}
