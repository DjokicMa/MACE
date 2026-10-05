"""The compact banner shows the installed version, not a hard-coded one."""
from mace import __version__
from mace.utils.banner import get_mace_banner_compact


def test_compact_banner_shows_the_current_version():
    banner = get_mace_banner_compact()
    assert f"v{__version__}" in banner
    assert "v1.0\n" not in banner
