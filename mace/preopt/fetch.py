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


def fetch(model: Dict[str, Any], *, yes: bool = False, directory: Optional[str] = None,
          out: Callable[[str], None] = None, ask: Callable[[str], str] = None, isatty: Optional[bool] = None,
          timeout: float = TIMEOUT) -> int:
    """0 done (or nothing to fetch), 1 failed, 2 refused (no consent)."""
    out = out or (lambda s: (sys.stdout.write(s), sys.stdout.flush()))
    w = model["weights"]
    if "url" not in w:
        out(f"{model['label']}: weights ship with the {model['code']['package']} package; nothing to fetch.\n")
        return 0
    if os.environ.get("MACE_MLIP_NO_DOWNLOAD") == "1":
        out("MACE_MLIP_NO_DOWNLOAD=1 is set: nothing is downloaded. Fetch the weights on a machine (or node) "
            "with network access, without that variable.\n")
        return 1
    from . import detect
    root = Path(directory).expanduser() if directory else detect.weights_root()
    target = root / model["id"] / w["filename"]
    out("ML potentials are third-party models, unrelated to this MACE tool.\n\n" + describe(model) + "\n\n")
    out(f"Destination:  {target}\n")
    if not yes:
        tty = sys.stdin.isatty() if isatty is None else isatty
        if not tty:
            out("stdin is not a terminal: pass --yes to accept the license and download.\n")
            return 2
        answer = (ask or input)(f"Download {w['size_bytes'] / 1e6:.0f} MB? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            out("Nothing was downloaded.\n")
            return 2
    target.parent.mkdir(parents=True, exist_ok=True)
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
           "mtime_ns": st.st_mtime_ns,
           "accepted_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    (target.parent / "ACCEPTED.json").write_text(json.dumps(rec, indent=2))
    out(f"Saved {target} (sha256 verified), with LICENSE.txt and ACCEPTED.json.\n")
    return 0
