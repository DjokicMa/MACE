"""F-D12: Area F's site.toml tables, their ranges and their error lines."""
from cluster_fixtures import own_cache  # noqa: F401  (autouse)
from mace.cluster import config


def _load(tmp_path, text):
    p = tmp_path / "site.toml"
    p.write_text(text)
    return config.load({"MACE_SITE_CONFIG": str(p)})


def test_builtin_without_file(tmp_path):
    cfg = config.load({"MACE_SITE_CONFIG": "none"})
    assert cfg.get("quota", "warn_percent") == 80 and cfg.get("quota", "critical_percent") == 90
    assert cfg.get("scan", "entries_per_second") == 2000 and cfg.get("scan", "max_entries") == 1_200_000
    assert cfg.errors == [] and cfg.path is None
    cfg = config.load({"MACE_SITE_CONFIG": str(tmp_path / "absent.toml")})
    assert cfg.errors == [] and cfg.get("devnodes", "hosts") == []


def test_values_and_sources(tmp_path):
    cfg = _load(tmp_path, '[quota]\nwarn_percent = 85\n\n[devnodes]\nhosts = ["dev-node-1", "dev-node-2"]\n')
    assert cfg.get("quota", "warn_percent") == 85
    assert cfg.sources["quota.warn_percent"] == "site.toml"
    assert cfg.sources["quota.critical_percent"] == "built-in"
    assert cfg.get("devnodes", "hosts") == ["dev-node-1", "dev-node-2"]


def test_errors_name_line_key_and_value(tmp_path):
    cfg = _load(tmp_path, "[quota]\nwarn_percent = 120\n")
    assert cfg.errors == ["site.toml:2: [quota] warn_percent = 120: must be a whole number from 50 to 99"]
    assert cfg.get("quota", "warn_percent") == 80


def test_critical_must_exceed_warn(tmp_path):
    cfg = _load(tmp_path, "[quota]\nwarn_percent = 90\ncritical_percent = 85\n")
    assert any("critical_percent = 85 must be above warn_percent = 90" in e for e in cfg.errors)
    assert (cfg.get("quota", "warn_percent"), cfg.get("quota", "critical_percent")) == (80, 90)


def test_scan_caps_must_agree(tmp_path):
    cfg = _load(tmp_path, "[scan]\nentries_per_second = 1000\nmax_seconds = 60\nmax_entries = 100000\n")
    err = " ".join(cfg.errors)
    assert "max_entries" in err and "entries_per_second x max_seconds" in err
    assert cfg.get("scan", "max_entries") == 60000


def test_alias_env_allow_refuses_code_changing_names(tmp_path):
    cfg = _load(tmp_path, '[quota]\nalias_env_allow = ["PYTHONPATH", "LC_ALL"]\n')
    assert any("PYTHONPATH" in e and "may never set" in e for e in cfg.errors)
    assert cfg.get("quota", "alias_env_allow") == []


def test_unknown_key_warns_and_bad_regex(tmp_path):
    cfg = _load(tmp_path, '[cluster]\ngroup_feature_regex = "(["\nnope = 1\n')
    assert any("nope is not a known key" in w for w in cfg.warnings)
    assert any("not a valid regular expression" in e for e in cfg.errors)
    assert cfg.get("cluster", "group_feature_regex") == config.DEFAULT_GROUP_REGEX
