"""mace/gui/redact.py: a table of inputs and outputs for every rule."""
import pytest

from mace.gui.redact import Redactor, decode

TOKEN = "Zq9TokenAbCdEfGhIjKlMnOp"
NONCE = "nOnCe0123456789abcdefg"


@pytest.fixture
def r():
    return Redactor(secrets=[TOKEN, "SessionIdXyZ0123456789ab", "LoginCodeQwErTy0987"], nonce=NONCE,
                    locations={"/home/zq user/proj": "<folder>", "/home/zq user": "~", "/mnt/scratch/zqx": "<scratch>"},
                    names=["Zq9Fake_system.out", "runs/Zq9Fake_system.d12", "mydata"],
                    users=["zqxuser", "al"], hosts=["node-17.cluster.example.edu"])


@pytest.mark.parametrize("raw,want", [
    ("/home/zq user/proj/sub/x y.out", "<folder>/sub/x <file>.out"),
    ("open /home/zq user/notes", "open <path>"),
    ("/mnt/scratch/zqx/run1/a.d12", "<scratch>/run1/<file>.d12"),
    ("C:\\Users\\zq\\Desktop\\a.cif", "<path>"),
    ("\\\\server\\share\\x", "<path>"),
    ("cannot read /opt/other/place/file", "cannot read <path>"),
    ("file:///tmp/a b", "<path> b"),
    ("GET /api/text?path=sub%2FZq9Fake_system.out 404", "GET /api/text?path=<path> 404"),
    ("path=sub%252FZq9Fake_system.out", "path=<path>"),
    ("x?q=a+b+c", "x?q=a b c"),
    ("at 17:01:47 done", "at 17:01:47 done"),
    ("relative sub/dir/X.out here", "relative <path> here"),
    ("bare X_opt.out", "bare <file>.out"),
    ("stem Zq9Fake_system failed", "stem <name> failed"),
    ("folder mydata listed", "folder <folder-name> listed"),
    (f"?token={TOKEN}&x=1", "?token=<redacted>&x=1"),
    ("login=LoginCodeQwErTy0987", "login=<redacted>"),
    (f"http://127.0.0.1:8765/s/{NONCE}/api/info", "http://<ip>:8765/s/<session>/api/info"),
    ("zqxuser@node-17.cluster.example.edu", "<email>"),
    ("ssh zqxuser@node-17", "ssh <user>@<host>"),
    ("on node-17.cluster.example.edu now", "on <host> now"),
    ("from 10.1.2.3 and fe80::1:2:3", "from <ip> and <ip>"),
    ("mail me at a.b@example.org", "mail me at <email>"),
    ("job 12345678 failed; slurm-87654321.out", "job <job> failed; <file>.out"),
    ("SLURM_JOB_ID=4455667", "SLURM_JOB_ID=<job>"),
    ("secret aGVsbG8gd29ybGQgdGhpcyBpcyBsb25n", "secret <secret>"),
    ("hash 9f86d081884c7d659a2feaa0c55ad015", "hash <secret>"),
    ("metal and al", "metal and <user>"),
    ("shell.js:120 TypeError", "shell.js:120 TypeError"),
    ("app.js:3", "app.js:3"),
    ("Python 3.12.2 MACE 1.1.5", "Python 3.12.2 MACE 1.1.5"),
    ("GET /api/folders 403 12ms", "GET /api/folders 403 12ms"),
    ("Content-Type application/json", "Content-Type application/json"),
    # a browser's user agent stays readable (which browser and version), its versions are not IP addresses
    ("Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
     "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"),
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 "
     "Safari/537.36 Edg/120.0.0.0", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
     "Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"),
    ("data/Si/100 at http://10.0.0.1/", "<path> at http://<ip>/"),
])
def test_rules(r, raw, want):
    assert r.text(raw) == want


def test_decode_two_passes():
    assert decode("a%252Fb") == "a/b" and decode("x?q=a+b") == "x?q=a b"


def test_final_check_counts_leftovers(r):
    text, n = r.final(f"raw {TOKEN} and Zq9Fake_system.out")
    assert TOKEN not in text and "Zq9Fake_system" not in text and n == 2


def test_settings_values_kept_only_when_safe():
    from mace.gui.diagnostics import settings_summary
    got = settings_summary({"theme": "dark", "uiScale": 110, "showCell": False, "slurm.account": "zqxacct",
                            "defaultDb": "runs/x.db", "partition": "zqxpart", "explorerExclude": ["*.bak"],
                            "gpu_device": "0"})
    assert got == {"theme": "dark", "uiScale": 110, "showCell": False, "slurm.account": "<text>",
                   "defaultDb": "<text>", "partition": "<text>", "explorerExclude": "<list of 1>", "gpu_device": "0"}
