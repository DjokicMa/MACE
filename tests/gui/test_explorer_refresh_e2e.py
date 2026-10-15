"""The Explorer shows files that appear in the folder without a reload
(Firefox and Chromium): a CIF copied in from outside the GUI and a job's
.out written into a new subfolder are listed within a few seconds, and one
that is deleted goes. The server watches the folders' modification times
(dirwatch.py, a few stat calls per check), not the files, and tells the page
with a files-changed event; the page then lists the folder once."""
import shutil
import time

import pytest

import guiharness as H

pytestmark = [pytest.mark.gui, pytest.mark.guismoke]
pytest.importorskip("selenium.webdriver")


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("explorer_refresh")
    root = H.examples_copy(tmp)
    proc, base, _ = H.start_server(root, tmp / "xdg", MACE_GUI_TEST_HOOKS="1", MACE_GUI_NO_WARMUP="1")
    yield root, base
    proc.terminate()
    proc.wait(timeout=30)


def _listed(drv, path):
    return drv.execute_script("return !!document.querySelector(`#file-list .file[data-path=\"${arguments[0]}\"]`)", path)


@pytest.mark.parametrize("browser", ["firefox", "chromium"])
def test_new_files_appear_without_a_reload(served, browser):
    root, base = served
    try:
        drv = H.browser(browser, width=1280, height=800)
    except Exception as e:                       # noqa: BLE001
        if browser == "chromium":
            pytest.skip(f"no Chromium: {e}")
        raise
    cif, out = f"structures/new_{browser}.cif", f"runs_{browser}/job1/job1.out"
    try:
        H.load(drv, base)
        H.wait(drv, "return !!document.querySelector('#file-list .file[data-path=\"structures/MgO.cif\"]')", 30)
        drv.execute_script("window.__stay = 1")
        assert not _listed(drv, cif)
        # outside the GUI: a copied CIF, and a job writing its output into a new folder
        shutil.copy2(root / "structures" / "MgO.cif", root / cif)
        (root / out).parent.mkdir(parents=True)
        (root / out).write_text(" CRYSTAL23 output (placeholder)\n")
        t0 = time.monotonic()
        H.wait(drv, f"return !!document.querySelector('#file-list .file[data-path=\"{cif}\"]')", 10)
        H.wait(drv, f"return !!document.querySelector('#file-list .file[data-path=\"{out}\"]')", 10)
        assert time.monotonic() - t0 < 10
        assert drv.execute_script("return window.__stay") == 1, "the page reloaded"
        (root / cif).unlink()
        H.wait(drv, f"return !document.querySelector('#file-list .file[data-path=\"{cif}\"]')", 10)
    except BaseException:
        H.save_failure(drv, f"explorer_refresh_{browser}")
        raise
    finally:
        drv.quit()
        (root / cif).unlink(missing_ok=True)
        shutil.rmtree(root / f"runs_{browser}", ignore_errors=True)
