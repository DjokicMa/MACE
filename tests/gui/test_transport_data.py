"""TRANSPORT plots' data (plan E8): for the four real BOLTZTRA runs, the
average of the tensor diagonal per temperature equals the database
extractor's own ``_parse_transport_dat`` rows (the GUI reads with it), and
the components are that average's parts. Corpus files read in place (read
only); skipped without the corpus."""
from pathlib import Path

import pytest

import guiharness as H

np = pytest.importorskip("numpy")
CORPUS = H.REPO_ROOT / "test" / "TRANSPORT"


@pytest.mark.skipif(not CORPUS.is_dir(), reason="test/ corpus not present (gitignored)")
def test_transport_arrays_are_the_extractors():
    from mace.gui import results
    from mace.utils.property_extractor import CrystalPropertyExtractor
    files = sorted(CORPUS.glob("*.SEEBECK.dat")) + sorted(CORPUS.glob("*.SIGMA.dat")) + sorted(CORPUS.glob("*.KAPPA.dat"))
    assert len({f.name.split(".")[0] for f in files}) == 4
    for f in files:
        d = results.transport_data(f)
        _, rows = CrystalPropertyExtractor._parse_transport_dat(f)
        factor = results.TRANSPORT_QUANTITIES[d["quantity"]][2]
        for b in d["blocks"]:
            want = [(mu, avg * factor) for T, mu, n, avg in rows if T == b["T"]]
            assert [(m, a) for m, a in zip(b["mu"], b["avg"])] == want
            got = (np.asarray(b["xx"]) + np.asarray(b["yy"]) + np.asarray(b["zz"])) / 3
            assert np.allclose(got, b["avg"], rtol=1e-12, atol=1e-30)
        assert d["siblings"]["SEEBECK"] and d["quantity"] in d["siblings"]
