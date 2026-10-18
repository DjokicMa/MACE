"""Size-aware job sizes in the GUI ([resources] auto): the Submit preview
writes the suggested size (the CLI does it: GUI route == CLI route), lists
the reason per deck, equals the real run, and GET /api/submit/suggest answers
what `mace site suggest --json` answers (one function, site.suggest_decks)."""
import json
import subprocess
import sys
from urllib.parse import urlencode

import submit_util as SU
from submit_util import SP_D12

OUT_HEAD = """ CRYSTAL CALCULATION
 NUMBER OF SYMMETRY OPERATORS         :    4
 N. OF ATOMS PER CELL       120  COULOMB OVERLAP TOL         (T1) 10**   -7
 NUMBER OF AO              1800  EXCHANGE OVERLAP TOL        (T3) 10**   -7
 SHRINK. FACT.(MONKH.)    4  4  4  NUMBER OF K POINTS IN THE IBZ     24
 CYC   0 ETOT(AU) -7.685930554264E+01 DETOT -7.69E+01 tst  0.00E+00 PX  1.00E+00
"""


def test_preview_writes_and_explains_the_suggestion(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("suggest")
    cfg = tmp / "site.toml"
    cfg.write_text("[resources]\nauto = true\n")
    s = SU.Served(tmp, MACE_SITE_CONFIG=str(cfg), MACE_SITE_GROUP_CONFIG="none")
    try:
        d = s.decks("p/decks", ["small_sp"], SP_D12)
        (d / "big_opt.out").write_text(OUT_HEAD + " CYC   1 ...\n" + "x" * 5000 + "\n")
        (d / "big_opt_sp.d12").write_text(SP_D12)
        st, prev = s.preview(["p/decks"])
        assert st == 200 and prev["code"] == 0, prev.get("cli_log")
        res = {r["deck"]: r for r in prev["resources"]}
        assert str(res["big_opt_sp"]["ntasks"]) == "32" and str(res["small_sp"]["ntasks"]) == "1"
        why = {g["deck"]: g for g in prev["suggestions"]}
        big = why["p/decks/big_opt_sp.d12"]
        assert big["auto"] and big["suggestion"]["reason"].startswith("~1,800 AOs, 24 k-points -> 32 cores")
        assert big["applies"]["ntasks"] == 32 and why["p/decks/small_sp.d12"]["applies"]["ntasks"] == 1
        # the route answers the same as the CLI
        q = urlencode([("path", "p/decks/big_opt_sp.d12"), ("path", "p/decks/small_sp.d12")])
        st, _, body = s.req(f"api/submit/suggest?{q}")
        assert st == 200, body
        env = dict(SU.fake_env(s.fake, tmp))
        env.update(MACE_SITE_CONFIG=str(cfg), MACE_SITE_GROUP_CONFIG="none")
        import os
        env = {**{k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "MACE_HOME")}, **env}
        r = subprocess.run([sys.executable, str(SU.MACE_CLI), "site", "suggest", "--json", "big_opt_sp.d12",
                            "small_sp.d12"], cwd=d, env=env, capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stdout + r.stderr
        cli = json.loads(r.stdout)
        for c, g in zip(cli, body["suggestions"]):
            assert (c["suggestion"], c["applies"], c["auto"]) == (g["suggestion"], g["applies"], g["auto"])
        assert body["suggestions"] == [why["p/decks/big_opt_sp.d12"], why["p/decks/small_sp.d12"]]
        # a bad path is refused
        st, _, _ = s.req("api/submit/suggest?path=p/decks/big_opt.out")
        assert st == 400
        # preview == the real run
        st, run = s.run(prev["preview_id"])
        assert st == 200 and run["code"] == 0, run
        assert run["script_drift"] == [], run["script_drift"]
        sh = (d / "big_opt_sp.sh").read_text()
        assert "#SBATCH --ntasks=32" in sh
    finally:
        s.stop()
