"""Research spaces and the HPCC quota tool, against output captured on MSU's
HPCC (tests/fixtures/hpcc: userquota.py plain and --verbose, plain quota)."""
import os
from pathlib import Path

import pytest


from mace.cluster import config, quotas, research

FIX = Path(__file__).resolve().parent / "fixtures" / "hpcc"


@pytest.fixture(autouse=True)
def own_cache(tmp_path, monkeypatch):
    """Nothing reads or writes the real ~/.cache/mace or ~/.config/mace."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    monkeypatch.setenv("MACE_SITE_CONFIG", "none")
GROUPS = ["mendozacortes_group", "chemeng", "g09", "2D_carbon_allotrope", "3DCarbonAllotropes", "PFGP",
          "Dual_ion_batteries", "common-data"]
NAMES = ["2D_carbon_allotrope", "3DCarbonAllotropes", "Dual_ion_batteries", "PFGP", "PbSeCuCl_phase_convex_hull",
         "Quaternary_phase_convex_hull", "Semiconductor_genome", "Ternary_phase_convex_hull", "mendozacortes_group"]


class Cfg:
    """The two lookups research.py makes of the site settings."""

    def __init__(self, roots, paths=()):
        self.v = {"research_roots": list(roots), "research_paths": list(paths)}

    def get(self, table, key):
        assert table == "quota"
        return self.v[key]


@pytest.fixture
def hpcc(tmp_path):
    """A /mnt/research look-alike: symlinks into a shared tree, like HPCC's."""
    shared = tmp_path / "gpfs" / "rs"
    res = tmp_path / "mnt" / "research"
    shared.mkdir(parents=True)
    res.mkdir(parents=True)
    for n in NAMES + ["common-data", "somebody_elses"]:
        (shared / n).mkdir()
        (res / n).symlink_to(shared / n)
    (shared / "somebody_elses").chmod(0o000)
    yield res, shared
    (shared / "somebody_elses").chmod(0o755)


def real_table():
    return quotas.table_parser((FIX / "userquota_raw.txt").read_text())


def test_real_hpcc_table_parses():
    rows = {r["label"]: r for r in real_table()}
    assert len(rows) == 11 and "home (VAST)" in rows and "scratch" in rows
    m = rows["mendozacortes_group"]
    assert m["space_pct"] == 60.7 and m["files_pct"] == 80.2
    assert m["space_limit_b"] == 2 * 1024 ** 4 and m["files_used"] == 1_700_000
    assert rows["home (VAST)"]["space_limit_b"] == 100 * 1024 ** 3
    assert quotas.levels(m, 80, 90) == "warning"          # files 80.2 %
    assert quotas.levels(rows["PFGP"], 80, 90) == "ok"


def test_plain_quota_wrapped_line_parses():
    """HPCC's long file system name wraps the numbers to the next line."""
    rows = quotas.quota_parser((FIX / "quota_plain.txt").read_text())
    assert len(rows) == 1
    r = rows[0]
    assert r["space_used_b"] == 52162208 * 1024 and r["space_limit_b"] == 104857600 * 1024
    assert r["files_used"] == 402865 and r["files_limit"] == 1_000_000


def test_annotate_labels_home_scratch_research(hpcc, tmp_path):
    res, shared = hpcc
    rows = real_table()
    research.annotate(rows, Cfg([res]), scratch=[str(tmp_path)], groups=GROUPS)
    by = {r["label"]: r for r in rows}
    assert by["home (VAST)"]["space"] == "home"
    assert by["scratch"]["space"] == "scratch" and by["scratch"]["path"] == str(tmp_path)
    m = by["mendozacortes_group"]
    assert m["space"] == "research" and m["path"] == str(res / "mendozacortes_group")
    assert m["real"] == os.path.realpath(shared / "mendozacortes_group")
    # a quota row for a group you are not listed in still resolves when the folder is there
    assert by["PbSeCuCl_phase_convex_hull"]["space"] == "research"


def test_spaces_lists_group_folders_without_a_quota_row(hpcc):
    res, _ = hpcc
    rows = real_table()
    for r in rows:
        r["level"] = quotas.levels(r, 80, 90)
    research.annotate(rows, Cfg([res]), groups=GROUPS)
    sp = {s["name"]: s for s in research.spaces(rows, Cfg([res]), groups=GROUPS)}
    assert sp["mendozacortes_group"]["has_quota"] and sp["mendozacortes_group"]["files_pct"] == 80.2
    assert sp["mendozacortes_group"]["level"] == "warning"
    assert not sp["common-data"]["has_quota"] and sp["common-data"]["space_pct"] is None
    assert "somebody_elses" not in sp and "chemeng" not in sp and "g09" not in sp   # no folder / not entered


def test_verbose_paths_become_names(hpcc, tmp_path):
    """userquota.py --verbose prints the real folders (/mnt/ufs18/rs/NAME)."""
    res, shared = hpcc
    text = (FIX / "userquota_verbose_raw.txt").read_text()
    for n in NAMES:
        text = text.replace("/mnt/ufs18/rs-048/" + n, str(shared / n)).replace("/mnt/ufs18/rs-035/" + n, str(shared / n)) \
            .replace("/mnt/ufs18/rs-038/" + n, str(shared / n)).replace("/mnt/ufs18/rs-045/" + n, str(shared / n)) \
            .replace("/mnt/ufs18/rs/" + n, str(shared / n))
    rows = quotas.table_parser(text)
    research.annotate(rows, Cfg([res]), groups=GROUPS)
    by = {r["label"]: r for r in rows}
    assert by["PFGP"]["space"] == "research" and by["PFGP"]["space_pct"] == 74.9


def test_no_research_root_changes_nothing(tmp_path):
    rows = real_table()
    research.annotate(rows, Cfg([tmp_path / "missing"]), groups=GROUPS)
    assert {r["space"] for r in rows} <= {"home", "scratch", ""}
    assert research.spaces(rows, Cfg([tmp_path / "missing"]), groups=GROUPS) == []


def test_extra_paths(tmp_path):
    d = tmp_path / "proj"
    d.mkdir()
    sp = research.usable_spaces(Cfg([tmp_path / "none"], [str(d)]), groups=[])
    assert [s["name"] for s in sp] == ["proj"]


def test_config_keys(tmp_path, monkeypatch):
    site = tmp_path / "site.toml"
    site.write_text('[quota]\nresearch_roots = ["/proj/research"]\nresearch_paths = ["~/shared"]\n')
    monkeypatch.setenv("MACE_SITE_CONFIG", str(site))
    cfg = config.load()
    assert cfg.get("quota", "research_roots") == ["/proj/research"]
    assert cfg.get("quota", "research_paths") == ["~/shared"]
    assert not cfg.errors
    site.write_text('[quota]\nresearch_roots = "oops"\n')
    cfg = config.load()
    assert cfg.errors and cfg.get("quota", "research_roots") == ["/mnt/research"]


def test_cli_text_shows_type_folder_and_quota_percent(hpcc, tmp_path):
    from mace.cluster import cli
    res, _ = hpcc
    rows = real_table()
    for r in rows:
        r["level"] = quotas.levels(r, 80, 90)
        r.setdefault("source", "site tool")
    research.annotate(rows, Cfg([res]), groups=GROUPS)
    v = {"rows": rows, "warn_percent": 80, "critical_percent": 90, "warnings": [], "notes": [],
         "research": research.spaces(rows, Cfg([res]), groups=GROUPS), "sources": ["site tool"]}
    txt = cli.quota_text(v)
    assert "Quota %" in txt and "Research" in txt and str(res / "PFGP") in txt
    assert str(res / "common-data") in txt                  # listed: no quota row


def test_roots_include_research_spaces(hpcc, tmp_path, monkeypatch):
    from mace.gui import roots
    res, shared = hpcc
    site = tmp_path / "site.toml"
    site.write_text(f'[quota]\nresearch_roots = ["{res}"]\n')
    monkeypatch.setenv("MACE_SITE_CONFIG", str(site))
    monkeypatch.setattr(research, "group_names", lambda: GROUPS)
    home = tmp_path / "home"
    home.mkdir()
    env = {"HOME": str(home), "USER": "u"}
    got = roots.allowed_roots(None, env)
    assert Path(os.path.realpath(shared / "mendozacortes_group")) in got
    assert Path(os.path.realpath(shared / "somebody_elses")) not in got
    label = roots.root_label(Path(os.path.realpath(shared / "PFGP")), env)
    assert label == "Research: PFGP"
    # an explicit --root list replaces the defaults, research spaces included
    assert roots.allowed_roots([str(home)], env) == [Path(os.path.realpath(home))]


def test_eta_with_a_few_seconds_low_end_reads_as_under():
    """The 1.1.6 preview showed 'about 0 s - 5 min left'."""
    from mace.cluster import progress, totals
    assert progress._about(0, 300) == "under 5 min"
    assert progress._about(3, 300) == "under 5 min"
    assert progress._about(120, 600) == "about 2 min - 10 min"
    assert progress._about(5, 8) == "about 5 s - 8 s"          # both low: unchanged
    assert totals._about(None, 300).startswith("about ")
