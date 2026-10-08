"""Redaction for Copy Diagnostics (plan PLAN_connect D20): no paths, no file
or folder names, no tokens, user or host names may leave in a bundle.

Applied to every field and then once more to the whole text, in this order:

0. decode first: percent-encoding (two passes, so ``%252F`` goes too) and
   ``+`` in query strings, so ``path=sub%2FX_opt.out`` is seen as a name;
1. exact secrets: the token, session ids, sign-in codes, the session path
   nonce (``/s/<nonce>/`` becomes ``/s/<session>/``);
2. known locations, longest first: the served folder (``<folder>``), the
   cache folder, the scratch roots (``<scratch>``), the home folder (``~``),
   the working folder, the XDG folders, the temporary folder;
3. names known to the session (files and folders listed, opened or
   switched to): a file becomes ``<file>.ext``, a folder ``<folder-name>``,
   a bare stem ``<name>``; matched as whole tokens; names shorter than 3
   characters and MACE's own static file names are exempt;
4. any token ending in a CRYSTAL/MACE/structure extension: ``<file>.ext``;
5. patterns: absolute and relative paths, ``file://`` URLs, ``token=`` and
   ``login=`` values, the user name, host names, IP addresses, e-mail
   addresses, SLURM job ids, long base64url or hex runs;
6. a final check for every exact secret, location and known name (counted).

Standard library only.
"""
from __future__ import annotations

import getpass
import os
import re
import socket
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import unquote

EXTENSIONS = (".out", ".d12", ".d3", ".f9", ".f98", ".gui", ".fort.34", ".cif", ".xyz", ".vasp", ".POSCAR", ".json",
              ".yaml", ".yml", ".db", ".h5", ".BAND.DAT", ".DOSS.DAT", ".cube", ".CUBE", ".xsf", ".log", ".sh",
              ".err", ".DAT", ".dat", ".txt", ".csv", ".bak", ".tmp")
_EXT_RE = re.compile(r"(?<![\w.<>-])([\w.-]+?)((?:\.BAND\.DAT|\.DOSS\.DAT|\.fort\.34|\.[oe]\d+|"
                     + "|".join(re.escape(e) for e in sorted(set(EXTENSIONS) - {".BAND.DAT", ".DOSS.DAT", ".fort.34"},
                                                                key=len, reverse=True)) + r"))(?![\w-])")
# a browser's user agent is made of product/version tokens ("Firefox/128.0"): kept, so a report says which browser
UA_PRODUCT_RE = re.compile(r"(?:Mozilla|Gecko|Firefox|FxiOS|AppleWebKit|KHTML|Chrome|Chromium|CriOS|HeadlessChrome|Safari|"
                           r"Version|Mobile|Edg|EdgA|EdgiOS|Edge|OPR|Opera|Electron|Code|Trident|Presto|SamsungBrowser|"
                           r"YaBrowser|Vivaldi)/\d+(?:\.\d+)*")
_UA_BEFORE_RE = re.compile(UA_PRODUCT_RE.pattern.split("/")[0] + r"/$")
SAFE_PREFIXES = ("api/", "/api/", "static/", "js/", "css/", "text/", "application/", "image/", "/s/<session>/")
HEADER = ("MACE GUI diagnostics. Paths, file and folder names, tokens, user and host names were removed; read it "
          "before you share it.")
_WORD = r"[\w.\-]"


def _static_names() -> Set[str]:
    out: Set[str] = set()
    static = Path(__file__).resolve().parent / "static"
    try:
        for p in static.rglob("*"):
            if p.is_file():
                out.add(p.name)
    except OSError:
        pass
    for p in Path(__file__).resolve().parent.glob("*.py"):
        out.add(p.name)
    return out


PRODUCT_NAMES = _static_names()


def decode(text: str) -> str:
    """Percent-decoding (two passes) and + -> space inside query strings."""
    def q(m):
        return m.group(0).replace("+", " ")
    text = re.sub(r"[?&][^\s]*", q, text)
    for _ in range(2):
        if not re.search(r"%[0-9A-Fa-f]{2}", text):
            break
        text = re.sub(r"(?:%[0-9A-Fa-f]{2})+", lambda m: unquote(m.group(0)), text)
    return text


class Redactor:
    def __init__(self, secrets: Iterable[str] = (), nonce: str = "", locations: Optional[Dict[str, str]] = None,
                 names: Iterable[str] = (), users: Iterable[str] = (), hosts: Iterable[str] = ()):
        self.secrets = sorted({s for s in secrets if s and len(s) >= 6}, key=len, reverse=True)
        self.nonce = nonce
        locs = {}
        for path, label in (locations or {}).items():
            if path and len(path) > 1:
                locs[path.rstrip("/") or path] = label
        self.locations = sorted(locs.items(), key=lambda kv: len(kv[0]), reverse=True)
        self.names = self._names(names)
        self.users = sorted({u for u in users if u and len(u) >= 2}, key=len, reverse=True)
        hs = set()
        for h in hosts:
            if h:
                hs.add(h)
                hs.add(h.split(".")[0])
        self.hosts = sorted({h for h in hs if len(h) >= 3 and h not in ("localhost",)}, key=len, reverse=True)
        self.extra_hits = 0

    @staticmethod
    def _names(names: Iterable[str]) -> List[Tuple[str, str]]:
        """(needle, replacement) for every known file/folder name and stem, longest first."""
        out: Dict[str, str] = {}
        for n in names:
            n = (n or "").strip().strip("/")
            if not n:
                continue
            for part in n.split("/"):
                if len(part) < 3 or part in PRODUCT_NAMES or part.startswith("<"):
                    continue
                m = _EXT_RE.fullmatch(part)
                if m:
                    out.setdefault(part, f"<file>{m.group(2)}")
                    stem = m.group(1)
                    if len(stem) >= 3:
                        out.setdefault(stem, "<name>")
                elif "." in part and not part.startswith("."):
                    stem, ext = part.rsplit(".", 1)
                    out.setdefault(part, f"<file>.{ext}")
                    if len(stem) >= 3:
                        out.setdefault(stem, "<name>")
                else:
                    out.setdefault(part, "<folder-name>")
        return sorted(out.items(), key=lambda kv: len(kv[0]), reverse=True)

    # -- the rules ------------------------------------------------------------------
    def text(self, text: str) -> str:
        if not text:
            return text or ""
        t = decode(str(text))
        # 1. exact secrets and the session path
        if self.nonce:
            t = t.replace(f"/s/{self.nonce}/", "/s/<session>/").replace(self.nonce, "<session>")
        for s in self.secrets:
            t = t.replace(s, "<secret>")
        t = re.sub(r"\b(token|login|code|sid)=([^\s&\"']+)", r"\1=<redacted>", t)
        # 2. known locations
        for path, label in self.locations:
            t = t.replace(path, label)
        t = re.sub(r"file://\S+", "<path>", t)
        # 3. known names (whole tokens)
        for needle, rep in self.names:
            t = re.sub(rf"(?<![\w.-]){re.escape(needle)}(?![\w-])", rep, t)
        # 4. any name with a known extension
        t = _EXT_RE.sub(lambda m: m.group(0) if m.group(0) in PRODUCT_NAMES or m.group(1) in ("<file>",)
                        else f"<file>{m.group(2)}", t)
        # 5. patterns
        t = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", "<email>", t)
        t = re.sub(r"(?<![\w/<])(?:[A-Za-z]:\\|\\\\)[^\s'\"]*", "<path>", t)            # Windows, UNC
        t = re.sub(r"(?:(?<=\s)|(?<=^)|(?<=[=('\"]))~?/(?!api/|s/<session>)[^\s'\":,)]+", self._abs, t, flags=re.M)
        t = re.sub(r"(?<![\w/<.~:-])[\w.-]+(?:/[\w.<>-]+)+", self._rel, t)
        for u in self.users:
            t = re.sub(rf"(?<![\w-]){re.escape(u)}(?![\w-])", "<user>", t)
        for h in self.hosts:
            t = re.sub(rf"(?<![\w-]){re.escape(h)}(?![\w-])", "<host>", t)
        # an IPv4 address, but not a browser version (Chrome/120.0.0.0)
        t = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
                   lambda m: m.group(0) if _UA_BEFORE_RE.search(m.string, 0, m.start()) else "<ip>", t)
        # IPv6: with a "::" or at least five groups (a clock time 17:01:47 is not one)
        t = re.sub(r"(?<![\w:])(?=[0-9a-fA-F:]*::|(?:[0-9a-fA-F]{1,4}:){4})(?:[0-9a-fA-F]{0,4}:){2,7}"
                   r"[0-9a-fA-F]{0,4}(?![\w:])", "<ip>", t)
        t = re.sub(r"(?i)\b(job(?:\s*id)?[\s:=#]+|slurm[_-]job[_-]id[=:\s]+|slurm-)(\d{4,})", r"\1<job>", t)
        t = re.sub(r"(?<![\w-])[A-Za-z0-9_-]{20,}(?![\w-])", self._long, t)
        return t

    @staticmethod
    def _abs(m) -> str:
        s = m.group(0)
        return s if s.startswith(("/api/", "/s/<session>")) else "<path>"

    @staticmethod
    def _rel(m) -> str:
        s = m.group(0)
        if s.startswith(SAFE_PREFIXES) or s.split("/")[-1] in PRODUCT_NAMES or "<" in s.split("/")[0]:
            return s
        if re.fullmatch(r"\d+(?:/\d+)+", s):                  # a date or a fraction
            return s
        if UA_PRODUCT_RE.fullmatch(s):                        # Firefox/128.0 in a user agent
            return s
        return "<path>"

    @staticmethod
    def _long(m) -> str:
        s = m.group(0)
        # a long word made of letters only (a word in a message) stays; mixed runs are secrets
        if s.isalpha() or re.fullmatch(r"[a-z_]+", s):
            return s
        if not re.search(r"\d", s) and not re.search(r"[A-Z].*[a-z].*[A-Z]", s):
            return s
        return "<secret>"

    def final(self, text: str) -> Tuple[str, int]:
        """Rule 6: look for every exact secret, location and known name once more."""
        n = 0
        needles = list(self.secrets) + ([self.nonce] if self.nonce else []) + [p for p, _ in self.locations] + \
            [nm for nm, _ in self.names]
        for s in needles:
            if s and s in text:
                n += text.count(s)
                text = text.replace(s, "<removed>")
        return text, n


def default_redactor(token: str = "", nonce: str = "", root: Optional[str] = None, extra_secrets: Sequence[str] = (),
                     names: Iterable[str] = (), cache: Optional[str] = None,
                     scratch: Sequence[str] = ()) -> Redactor:
    """A Redactor for this process: its user, host, home, cwd, XDG and temp folders."""
    import tempfile
    locs: Dict[str, str] = {}
    if root:
        locs[str(root)] = "<folder>"
        locs[os.path.realpath(str(root))] = "<folder>"
    if cache:
        locs[str(cache)] = "<cache>"
    for s in scratch:
        locs[str(s)] = "<scratch>"
    home = os.environ.get("HOME") or os.path.expanduser("~")
    for h in (home, os.path.realpath(home)):
        locs.setdefault(h, "~")
    try:
        locs.setdefault(os.getcwd(), "<cwd>")
    except OSError:
        pass
    for k in ("XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_RUNTIME_DIR", "TMPDIR"):
        v = os.environ.get(k)
        if v:
            locs.setdefault(v, f"<{k.lower().replace('_home', '').replace('xdg_', '')}>")
    locs.setdefault(tempfile.gettempdir(), "<tmp>")
    users = set()
    for u in (os.environ.get("USER"), os.environ.get("LOGNAME")):
        if u:
            users.add(u)
    try:
        users.add(getpass.getuser())
    except Exception:
        pass
    hosts = set()
    for h in (socket.gethostname(), socket.getfqdn()):
        if h:
            hosts.add(h)
    return Redactor(secrets=[token, *extra_secrets], nonce=nonce, locations=locs, names=names, users=users,
                    hosts=hosts)
