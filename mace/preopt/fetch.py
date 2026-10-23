"""Download a model's weights, only when the user asks (``mace mlip fetch``).

Before a byte is downloaded the user sees the model, that it is unrelated to
this MACE, the code and weights licenses with their links, the size, URL,
sha256, the training data and the citation, and answers ``Download N MB?``
(``--yes`` is required when stdin is not a terminal). The file goes to
``<file>.part`` and is renamed only after its size and sha256 match the
registry. Next to it: ``LICENSE.txt`` (the license text fetched from the
pinned raw URL when reachable, else its name and URL, said so) and
``ACCEPTED.json`` ({id, license, url, sha256, size, mtime_ns, accepted_utc};
no user or host name). ``MACE_MLIP_NO_DOWNLOAD=1`` refuses.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Optional

TIMEOUT = 60.0
CHUNK = 1 << 20
UA = "mace-tool-mlip-fetch"


def describe(model: Dict[str, Any]) -> str:
    w, c = model["weights"], model["code"]
    lines = [f"Model:        {model['label']}  (id {model['id']})",
             f"              {model.get('unrelated_note', 'a third-party model, unrelated to this MACE tool')}",
             f"Code:         {c['package']} ({c['pin']}), license {c['license']}: {c.get('license_text_url', '')}",
             f"Weights:      license {w['license']}: {w.get('license_text_url', '')}"]
    if "url" in w:
        lines += [f"Size:         {w['size_bytes'] / 1e6:.1f} MB ({w['size_bytes']} bytes)",
                  f"URL:          {w['url']}", f"sha256:       {w['sha256']}"]
    elif model.get("gated"):
        lines += [f"Size:         {int(w['size_bytes']) / 1e9:.2f} GB ({w['size_bytes']} bytes)",
                  f"Hugging Face: {w['hf_repo']}/{w['hf_file']} at {w['hf_revision']} (gated: {model['gated']['page']})",
                  f"sha256:       {w['sha256']}"]
    if model.get("academic"):
        lines.insert(1, "              ACADEMIC, NON-COMMERCIAL USE ONLY (opt-in; results are tagged \"(academic license)\")")
    lines.append("Training data: " + "; ".join(model.get("training_data") or []))
    for cit in model.get("citation") or []:
        lines.append("Cite:         " + cit["text"] + (f" (doi:{cit['doi']})" if cit.get("doi") else ""))
    for cit in model.get("data_citation") or []:
        lines.append("Data:         " + cit["text"] + (f" (doi:{cit['doi']})" if cit.get("doi") else ""))
    return "\n".join(lines)


def _open(url: str, timeout: float):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=timeout)


def _download(url: str, part: Path, size: int, out: Callable[[str], None], timeout: float) -> str:
    h = hashlib.sha256()
    n = 0
    last = 0.0
    with _open(url, timeout) as r, open(part, "wb") as fh:
        while True:
            b = r.read(CHUNK)
            if not b:
                break
            n += len(b)
            if n > size:                     # never let a wrong answer fill the disk
                raise ValueError(f"the server sent more than the {size} bytes the registry pins")
            fh.write(b)
            h.update(b)
            now = time.monotonic()
            if now - last > 0.5:
                last = now
                out(f"\r  {n / 1e6:6.1f} / {size / 1e6:.1f} MB")
    out(f"\r  {n / 1e6:6.1f} / {size / 1e6:.1f} MB\n")
    if n != size:
        raise ValueError(f"got {n} bytes, expected {size}")
    return h.hexdigest()


def _license_text(model: Dict[str, Any], timeout: float) -> str:
    parts = []
    for what, block in (("Weights", model["weights"]), ("Code", model["code"])):
        url = block.get("license_text_url")
        head = f"{what} license: {block['license']}\nSource: {url}\n"
        text = None
        if url:
            try:
                with _open(url, timeout) as r:
                    text = r.read(1 << 20).decode("utf-8", errors="replace")
            except Exception:
                text = None
        parts.append(head + ("\n" + text.strip() + "\n" if text else
                             "\n(The license text could not be fetched; read it at the URL above.)\n"))
    return f"{model['label']} ({model['id']})\n\n" + "\n" + ("\n" + "-" * 72 + "\n\n").join(parts)


ACADEMIC_WORD = "ACADEMIC"
ACADEMIC_STATEMENT = ("I will use this model for academic, non-commercial work only, under its license "
                      "({license}); MACE tags every result made with it \"(academic license)\".")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _academic_consent(model, *, accept_academic: bool, yes: bool, ask, tty: bool, out) -> Optional[str]:
    """How the user acknowledged academic use (``typed``, ``flag``, ``gui``), or None."""
    lic = model["weights"]["license"]
    out(f"\n{model['label']} is for ACADEMIC, NON-COMMERCIAL USE ONLY: its weights are under the {lic} "
        f"({model['weights'].get('license_text_url', '')}). It is never a default, never chosen by auto, and "
        "every result made with it is tagged \"(academic license)\".\n")
    if not accept_academic:
        out("Not fetched: pass --accept-academic-license to acknowledge academic, non-commercial use "
            "(in the GUI: tick the acknowledgment under the license text).\n")
        return None
    if tty:
        answer = (ask or input)(f"Type {ACADEMIC_WORD} to confirm: {ACADEMIC_STATEMENT.format(license=lic)}\n> ").strip()
        if answer != ACADEMIC_WORD:
            out("Not confirmed: nothing was downloaded.\n")
            return None
        return "typed"
    if not yes:
        out("stdin is not a terminal: pass --yes with --accept-academic-license.\n")
        return None
    return "flag"


def record_academic(model, how: str, d: Optional[Path] = None) -> Path:
    from . import detect
    d = Path(d) if d is not None else detect.record_dir(model)
    d.mkdir(parents=True, exist_ok=True)
    rec = {"id": model["id"], "license": model["weights"]["license"], "how": how, "acknowledged_utc": _now(),
           "statement": ACADEMIC_STATEMENT.format(license=model["weights"]["license"])}
    (d / "ACADEMIC.json").write_text(json.dumps(rec, indent=2))
    return d / "ACADEMIC.json"


HF_SCRIPT = r"""
import json, sys
repo, name, rev, backend, variant = sys.argv[1:6]
try:
    from huggingface_hub import hf_hub_download
except ImportError as exc:
    print("MACE_HF " + json.dumps({"error": "no_hub", "message": str(exc)})); sys.exit(0)
try:
    if backend == "fairchem":
        # fairchem's own cache (~/.cache/fairchem): the checkpoint and its reference energies, where the
        # worker's get_predict_unit finds them offline
        from fairchem.core.calculate.pretrained_mlip import resolve_checkpoint
        path = resolve_checkpoint(variant).path
    else:
        path = hf_hub_download(repo_id=repo, filename=name, revision=rev)
    print("MACE_HF " + json.dumps({"path": str(path)}))
except Exception as exc:
    kind = type(exc).__name__
    msg = str(exc).splitlines()[0][:400] if str(exc) else kind
    gated = kind in ("GatedRepoError", "RepositoryNotFoundError") or "401" in msg or "403" in msg or "gated" in msg.lower()
    print("MACE_HF " + json.dumps({"error": "gated" if gated else "failed", "kind": kind, "message": msg}))
"""


def _fetch_gated(model, *, yes: bool, ask, tty: bool, out, accepted_how: Optional[str] = None,
                 runner=None) -> int:
    """Gated weights (UMA): show the terms, record the accept, then download
    with the user's own Hugging Face access (``hf auth login`` or HF_TOKEN) in
    the model's environment. MACE never accepts the gate for the user and
    never stores a token."""
    import subprocess
    from . import detect
    w, g = model["weights"], model["gated"]
    out(f"\nTerms: {g.get('terms', w['license'])}\n  {w.get('license_text_url', '')}\n"
        + (f"  Acceptable Use Policy: {g['aup_url']}\n" if g.get("aup_url") else "")
        + f"The weights are gated on Hugging Face: accept the terms at {g['page']} with your own account first, "
          "then log in where MACE runs (hf auth login, or set HF_TOKEN). MACE never accepts them for you and "
          "never stores your token.\n")
    if not yes:
        if not tty:
            out("stdin is not a terminal: pass --yes to accept the terms and download.\n")
            return 2
        if (ask or input)(f"Accept the {w['license']} and its Acceptable Use Policy, and download "
                          f"{int(w['size_bytes']) / 1e9:.1f} GB? [y/N] ").strip().lower() not in ("y", "yes"):
            out("Nothing was downloaded.\n")
            return 2
    py = detect.configured_python(model)
    if not py:
        out(f"No {detect.env_name(model)} environment: mace mlip install --backend {model['code']['package']} first.\n")
        return 1
    out(f"Downloading {g.get('repo', w['hf_repo'])}/{w['hf_file']} (revision {w['hf_revision'][:12]}) through "
        "huggingface_hub (its own cache)...\n")
    if runner is None:
        def runner(argv):
            r = subprocess.run(argv, env=detect.interpreter_env(argv[0]), capture_output=True, text=True, timeout=7200,
                               stdin=subprocess.DEVNULL)
            return r.stdout
    text = runner([py, "-I", "-c", HF_SCRIPT, w["hf_repo"], w["hf_file"], w["hf_revision"], model["backend"],
                   str(model.get("variant") or "")])
    line = next((ln for ln in (text or "").splitlines() if ln.startswith("MACE_HF ")), None)
    res = json.loads(line[8:]) if line else {"error": "failed", "message": "no answer from huggingface_hub"}
    if res.get("error") == "gated":
        out(f"No access: your Hugging Face login has not been granted {w['hf_repo']} ({res.get('message')}). "
            f"Request access at {g['page']}, then log in (hf auth login) or set HF_TOKEN, and fetch again.\n")
        return 1
    if res.get("error"):
        out(f"The download failed: {res.get('message')}\n")
        return 1
    path = res["path"]
    size = os.path.getsize(path)
    digest = sha256_of(path)
    if size != int(w["size_bytes"]) or digest != w["sha256"]:
        out(f"The cached file is not the registry's (size {size}, sha256 {digest[:12]}...): nothing was recorded.\n")
        return 1
    d = detect.record_dir(model)
    d.mkdir(parents=True, exist_ok=True)
    rec = {"id": model["id"], "license": w["license"], "terms": g.get("terms"), "page": g["page"],
           "aup_url": g.get("aup_url"), "how": accepted_how or ("flag" if yes else "typed"), "accepted_utc": _now(),
           "path": path, "sha256": digest, "size": size, "token_stored": False}
    (d / "ACCEPTED.json").write_text(json.dumps(rec, indent=2))
    out(f"Saved in the Hugging Face cache: {path} (sha256 verified); the accept is recorded in {d / 'ACCEPTED.json'}.\n")
    return 0


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def fetch(model: Dict[str, Any], *, yes: bool = False, directory: Optional[str] = None,
          out: Callable[[str], None] = None, ask: Callable[[str], str] = None, isatty: Optional[bool] = None,
          timeout: float = TIMEOUT, accept_academic: bool = False, academic_how: Optional[str] = None,
          hf_runner=None, consent_how: Optional[str] = None) -> int:
    """0 done (or nothing to fetch), 1 failed, 2 refused (no consent).
    An academic-only model needs ``accept_academic`` (and, at a terminal, a
    typed confirmation; ``academic_how="gui"`` is the GUI's checkbox), and the
    acknowledgment is recorded with its date (ACADEMIC.json)."""
    out = out or (lambda s: (sys.stdout.write(s), sys.stdout.flush()))
    w = model["weights"]
    tty = sys.stdin.isatty() if isatty is None else isatty
    if "url" not in w and not model.get("gated"):
        out(f"{model['label']}: weights ship with the {model['code']['package']} package; nothing to fetch.\n")
        return 0
    if os.environ.get("MACE_MLIP_NO_DOWNLOAD") == "1":
        out("MACE_MLIP_NO_DOWNLOAD=1 is set: nothing is downloaded. Fetch the weights on a machine (or node) "
            "with network access, without that variable.\n")
        return 1
    from . import detect
    out("ML potentials are third-party models, unrelated to this MACE tool.\n\n" + describe(model) + "\n")
    how = None
    if model.get("academic"):
        how = academic_how if (academic_how and accept_academic) else _academic_consent(
            model, accept_academic=accept_academic, yes=yes, ask=ask, tty=tty, out=out)
        if how is None:
            return 2
        yes = True                                       # the acknowledgment is the consent
    if model.get("gated"):
        return _fetch_gated(model, yes=yes, ask=ask, tty=tty, out=out, runner=hf_runner, accepted_how=consent_how)
    root = Path(directory).expanduser() if directory else detect.weights_root()
    target = root / model["id"] / w["filename"]
    out(f"\nDestination:  {target}\n")
    if not yes:
        if not tty:
            out("stdin is not a terminal: pass --yes to accept the license and download.\n")
            return 2
        answer = (ask or input)(f"Download {w['size_bytes'] / 1e6:.0f} MB? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            out("Nothing was downloaded.\n")
            return 2
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == int(w["size_bytes"]) and sha256_of(target) == w["sha256"]:
        out("The weights are already there (sha256 verified): nothing to download.\n")
        if how:
            out(f"Acknowledgment recorded in {record_academic(model, how, target.parent)}.\n")
        return 0
    part = target.with_name(target.name + ".part")
    digest, err, used = None, None, None
    for url in [w["url"]] + ([w["mirror_url"]] if w.get("mirror_url") else []):
        out(f"Downloading {url}\n")
        try:
            digest = _download(url, part, int(w["size_bytes"]), out, timeout)
            used = url
            break
        except Exception as exc:
            err = exc
            part.unlink(missing_ok=True)
            out(f"  failed: {exc}\n")
    if digest is None:
        out(f"The download failed: {err}\n")
        return 1
    if digest != w["sha256"]:
        part.unlink(missing_ok=True)
        out(f"sha256 mismatch: got {digest}, the registry pins {w['sha256']}. Nothing was kept.\n")
        return 1
    os.replace(part, target)
    (target.parent / "LICENSE.txt").write_text(_license_text(model, timeout))
    st = target.stat()
    rec = {"id": model["id"], "license": w["license"], "url": used, "sha256": digest, "size": st.st_size,
           "mtime_ns": st.st_mtime_ns, "accepted_utc": _now()}
    (target.parent / "ACCEPTED.json").write_text(json.dumps(rec, indent=2))
    if how:
        out(f"Acknowledgment of academic, non-commercial use recorded in {record_academic(model, how, target.parent)}.\n")
    out(f"Saved {target} (sha256 verified), with LICENSE.txt and ACCEPTED.json.\n")
    return 0
