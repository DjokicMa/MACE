"""The GUI profile in gui.json, version 2 (polish P1), through the real
`mace gui`: settings, ui, memory and recent sections, per-key patches,
the one-time v1 copy, unknown sections of a newer file kept, two servers
writing at once, the section checks, 0600, the token; and the lock
fallback and read-back retry of prefstore.py as units."""
import json
import threading

import pytest

import guiharness as H
from test_shell_api import req


@pytest.fixture
def served(tmp_path):
    root, xdg = H.examples_copy(tmp_path), tmp_path / "xdg"
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_NO_WARMUP="1")
    yield base, root, xdg
    proc.terminate()
    proc.wait(timeout=15)


def profile(base, all=False):
    st, _, body = req(base, "api/profile" + ("?all=1" if all else ""))
    assert st == 200, body
    return json.loads(body)


def patch(base, body):
    return req(base, "api/profile", data=body)


def test_empty_profile_and_round_trip(served):
    base, root, xdg = served
    p = profile(base)
    assert p["rev"] == 0 and p["settings"] == {} and p["ui"] == {} and p["memory"] == {}
    st, _, body = patch(base, {"settings": {"set": {"theme": "dark"}}, "ui": {"set": {"layout": {"wExplorer": 300}}},
                               "memory": {"set": {"cleanMethod": "springs"}},
                               "recent": {"files": {p["dir_hash"]: {"root": str(root), "items": [
                                   {"path": "structures/MgO.cif", "t": 1, "pinned": True}]}}}})
    r = json.loads(body)
    assert st == 200 and r["rev"] == 1 and not r["rejected"], r
    assert r["settings"]["theme"] == "dark" and r["ui"]["layout"]["wExplorer"] == 300
    assert r["memory"]["cleanMethod"] == "springs" and r["recent"]["files"][p["dir_hash"]]["items"][0]["pinned"]
    f = xdg / "mace" / "gui.json"
    assert (f.stat().st_mode & 0o777) == 0o600 and json.loads(f.read_text())["version"] == 2
    # unset and a rejected settings key (named, the stored value kept)
    r = json.loads(patch(base, {"settings": {"set": {"depthCue": 9}, "unset": ["theme"]}})[2])
    assert "settings.depthCue" in r["rejected"] and "theme" not in r["settings"]
    # the old settings route still stores, and keeps the other sections
    assert req(base, "api/settings", data={"settings": {"uiScale": 110}})[0] == 200
    p = profile(base)
    assert p["settings"] == {"uiScale": 110} and p["ui"]["layout"]["wExplorer"] == 300


def test_v1_file_is_copied_once_then_v2(tmp_path):
    root, xdg = H.examples_copy(tmp_path), tmp_path / "xdg"
    (xdg / "mace").mkdir(parents=True)
    v1 = json.dumps({"version": 1, "settings": {"theme": "dark", "uiScale": 120}}, indent=2) + "\n"
    (xdg / "mace" / "gui.json").write_text(v1)
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_NO_WARMUP="1")
    try:
        assert profile(base)["settings"] == {"theme": "dark", "uiScale": 120}
        assert profile(base)["recent"] == {"files": {}, "folders": []}     # never null before the first write
        assert patch(base, {"ui": {"set": {"x": 1}}})[0] == 200
        assert (xdg / "mace" / "gui.json.v1.bak").read_text() == v1
        stored = json.loads((xdg / "mace" / "gui.json").read_text())
        assert stored["version"] == 2 and stored["settings"] == {"theme": "dark", "uiScale": 120}
        assert patch(base, {"ui": {"set": {"x": 2}}})[0] == 200
        assert (xdg / "mace" / "gui.json.v1.bak").read_text() == v1          # never replaced
    finally:
        proc.terminate()
        proc.wait(timeout=15)


def test_newer_file_keeps_unknown_sections(tmp_path):
    root, xdg = H.examples_copy(tmp_path), tmp_path / "xdg"
    (xdg / "mace").mkdir(parents=True)
    (xdg / "mace" / "gui.json").write_text(json.dumps({"version": 3, "rev": 7, "settings": {"theme": "light"},
                                                       "future": {"keep": [1, 2]}}))
    proc, base, _ = H.start_server(root, xdg, MACE_GUI_NO_WARMUP="1")
    try:
        assert patch(base, {"memory": {"set": {"a": 1}}})[0] == 200
        stored = json.loads((xdg / "mace" / "gui.json").read_text())
        assert stored["future"] == {"keep": [1, 2]} and stored["version"] == 3 and stored["rev"] == 8
    finally:
        proc.terminate()
        proc.wait(timeout=15)


def test_section_checks(served):
    base, root, _ = served
    deep = {"a": {"b": {"c": {"d": 1}}}}
    st, _, body = patch(base, {"ui": {"set": {"deep": deep}}})
    assert st == 400 and b"ui.deep" in body
    st, _, body = patch(base, {"memory": {"set": {"big": "x" * 1500, "big2": ["y" * 1500] * 30}}})
    assert st == 400 and b"memory" in body
    h = profile(base)["dir_hash"]
    st, _, body = patch(base, {"recent": {"files": {h: {"root": str(root), "items": [{"path": "../x", "t": 1, "pinned": False}]}}}})
    assert st == 400 and b"recent" in body
    assert patch(base, {"recent": {"folders": ["relative/dir"]}})[0] == 400
    assert req(base, "api/profile", token=False)[0] == 403
    assert req(base, "api/profile", data={"ui": {"set": {}}}, token=False)[0] == 403


def test_two_servers_patch_different_keys(tmp_path):
    root, xdg = H.examples_copy(tmp_path), tmp_path / "xdg"
    a, base_a, _ = H.start_server(root, xdg, MACE_GUI_NO_WARMUP="1")
    b, base_b, _ = H.start_server(root, xdg, MACE_GUI_NO_WARMUP="1")
    errors = []

    def run(base, tag):
        for k in range(50):
            st, _, body = patch(base, {"memory": {"set": {f"{tag}{k}": k}}})
            if st != 200:
                errors.append((st, body))

    try:
        ts = [threading.Thread(target=run, args=(base_a, "a")), threading.Thread(target=run, args=(base_b, "b"))]
        [t.start() for t in ts]
        [t.join() for t in ts]
        assert not errors, errors[:3]
        stored = json.loads((xdg / "mace" / "gui.json").read_text())
        assert len(stored["memory"]) == 100 and stored["rev"] == 100
    finally:
        for p in (a, b):
            p.terminate()
            p.wait(timeout=15)


# ---------------------------------------------------------------- prefstore units
@pytest.fixture
def store(tmp_path, monkeypatch):
    from mace.gui import prefstore
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    prefstore._probe_cache.clear()
    return prefstore


def test_lock_fallback_when_lockf_raises(store, monkeypatch, tmp_path):
    import fcntl
    from mace.gui import settings

    def broken(*a, **k):
        raise OSError(37, "No locks available")
    monkeypatch.setattr(fcntl, "lockf", broken)
    p, r = settings.patch({"memory": {"set": {"k": 1}}})
    assert p["memory"] == {"k": 1}
    assert store._probe_cache.get(str(settings.path().parent)) is False


def test_lock_probe_forced_off(store, tmp_path):
    from mace.gui import settings
    settings.path().parent.mkdir(parents=True)
    store._probe_cache[str(settings.path().parent)] = False
    before = store.stats["unlocked_writes"]
    p, _ = settings.patch({"ui": {"set": {"a": 1}}})
    assert p["ui"] == {"a": 1} and store.stats["unlocked_writes"] == before + 1


def test_lock_probe_on_a_local_disk(store, tmp_path):
    d = tmp_path / "probe"
    d.mkdir()
    assert store.locks_work(d) is True


def test_read_back_reapplies_a_lost_patch(store, monkeypatch):
    """A lock that only held on this host: another server's write lands
    between our replace and our read-back; the patch is merged again."""
    from mace.gui import settings
    settings.patch({"memory": {"set": {"other": 0}}})
    fired = []

    def other_server(path):
        if fired:
            return
        fired.append(1)
        data = json.loads(path.read_text())
        data["memory"] = {"other": 1}             # their key; ours is gone
        data["rev"] += 1
        path.write_text(json.dumps(data))
    monkeypatch.setattr(store, "_after_replace", other_server)
    before = store.stats["retries"]
    p, _ = settings.patch({"memory": {"set": {"mine": 2}}})
    assert p["memory"] == {"other": 1, "mine": 2} and store.stats["retries"] == before + 1


def test_one_server_saving_a_setting_and_a_remembered_choice_at_once_keeps_both(served):
    """The request threads of one server: a settings save and a profile patch
    arriving together (Settings and a remembered choice right after a page
    load) both land. fcntl locks are per process, so prefstore also orders
    the threads; before that the later write dropped the other's change."""
    base, root, xdg = served
    errors = []

    def save(i):
        st, _, body = req(base, "api/settings", data={"settings": {"theme": "dark" if i % 2 else "light",
                                                                   "labelSize": 10 + i % 5}})
        if st != 200:
            errors.append(body)

    def remember(i):
        st, _, body = patch(base, {"memory": {"set": {f"k{i}": i}}})
        if st != 200:
            errors.append(body)
    for i in range(25):
        ts = [threading.Thread(target=save, args=(i,)), threading.Thread(target=remember, args=(i,))]
        [t.start() for t in ts]
        [t.join() for t in ts]
    assert not errors, errors
    p = profile(base)
    assert p["memory"] == {f"k{i}": i for i in range(25)}, p["memory"]
