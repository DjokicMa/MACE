"""Finding ssh when it is not on PATH (Windows: a PATH that PowerShell or
VS Code's terminal lost, or a Git for Windows install nobody added to it).

``mace gui connect`` needs an OpenSSH client. On Windows 10 and 11 that is
``C:\\Windows\\System32\\OpenSSH\\ssh.exe``, which is on PATH unless something
overwrote the PATH. ``find_ssh()`` looks on PATH first, then in the places an
OpenSSH client is installed:

1. ``%SystemRoot%\\System32\\OpenSSH`` (and ``Sysnative`` for a 32-bit Python);
2. ``%ProgramFiles%\\OpenSSH`` (the Win32-OpenSSH installer);
3. Git for Windows: ``usr\\bin`` of ``%ProgramFiles%\\Git``,
   ``%ProgramFiles(x86)%\\Git``, ``%LOCALAPPDATA%\\Programs\\Git``;
4. the ``remote.SSH.path`` setting of VS Code (the ssh its Remote-SSH uses).

``ensure_ssh()`` puts the folder of the ssh it found at the front of this
process's PATH, so every ``ssh`` the program starts (and the ``ssh-keygen``
and ``ssh-add`` beside it) runs, and says in one line what it did. When no
ssh exists it returns the exact steps to install one. Standard library only.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Tuple

SYSTEM_DIR = r"C:\Windows\System32\OpenSSH"
INSTALL_WINDOWS = ("Windows 10 and 11 include the OpenSSH client as an optional feature. In an administrator "
                   "PowerShell run: Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0 (or Settings > "
                   "Apps > Optional features > Add a feature > OpenSSH Client), then open a new terminal and run "
                   "ssh -V. Git for Windows (git-scm.com) also brings an ssh.")
INSTALL_POSIX = "Install the OpenSSH client (the openssh-client or openssh package)."


@dataclass
class SshFound:
    path: str                  # the ssh executable
    source: str                # "PATH", "Windows OpenSSH", "Git for Windows", "Win32-OpenSSH", "VS Code setting"
    on_path: bool              # the shell would have found it by itself

    @property
    def folder(self) -> str:
        import ntpath
        return ntpath.dirname(self.path) if "\\" in self.path else os.path.dirname(self.path)


def _env(env: Optional[Mapping[str, str]]) -> Mapping[str, str]:
    return os.environ if env is None else env


def _getenv(env: Mapping[str, str], name: str) -> str:
    """A Windows environment name is case-insensitive; a dict is not."""
    if name in env:
        return env[name]
    low = name.lower()
    for k, v in env.items():
        if k.lower() == low:
            return v
    return ""


def vscode_ssh_path(env: Optional[Mapping[str, str]] = None, read: Callable[[str], str] = None) -> Optional[str]:
    """``remote.SSH.path`` from VS Code's user settings (%APPDATA%\\Code\\User\\settings.json;
    the Insiders and VSCodium folders too): the ssh Remote-SSH runs. The file may hold comments
    and trailing commas (JSONC), so the one key is read by pattern."""
    e = _env(env)
    appdata = _getenv(e, "APPDATA")
    roots = [appdata] if appdata else []        # Windows only: no settings are read elsewhere
    rd = read or (lambda p: open(p, encoding="utf-8", errors="replace").read())
    for root in roots:
        for app in ("Code", "Code - Insiders", "VSCodium"):
            p = os.path.join(root, app, "User", "settings.json")
            try:
                text = rd(p)
            except OSError:
                continue
            m = re.search(r'"remote\.SSH\.path"\s*:\s*("(?:[^"\\]|\\.)*")', text)
            if m:
                try:
                    val = json.loads(m.group(1))
                except ValueError:
                    continue
                if val:
                    return val
    return None


def candidates(env: Optional[Mapping[str, str]] = None, exe: str = "ssh.exe"
               ) -> List[Tuple[str, str]]:
    """(source, path) pairs to try, in order, on Windows."""
    e = _env(env)
    root = _getenv(e, "SystemRoot") or _getenv(e, "windir") or r"C:\Windows"
    pf = _getenv(e, "ProgramFiles") or r"C:\Program Files"
    pf86 = _getenv(e, "ProgramFiles(x86)")
    local = _getenv(e, "LOCALAPPDATA")
    w = lambda *parts: "\\".join(p.rstrip("\\") for p in parts[:-1]) + "\\" + parts[-1]
    out: List[Tuple[str, str]] = [
        ("Windows OpenSSH", w(root, "System32", "OpenSSH", exe)),
        ("Windows OpenSSH", w(root, "Sysnative", "OpenSSH", exe)),
        ("Win32-OpenSSH", w(pf, "OpenSSH", exe)),
        ("Git for Windows", w(pf, "Git", "usr", "bin", exe)),
    ]
    if pf86:
        out.append(("Git for Windows", w(pf86, "Git", "usr", "bin", exe)))
    if local:
        out.append(("Git for Windows", w(local, "Programs", "Git", "usr", "bin", exe)))
    vs = vscode_ssh_path(e)
    if vs:
        out.insert(0, ("VS Code setting", vs))
    return out


def find_ssh(env: Optional[Mapping[str, str]] = None, which: Callable[[str], Optional[str]] = shutil.which,
             isfile: Callable[[str], bool] = os.path.isfile, windows: Optional[bool] = None
             ) -> Optional[SshFound]:
    """The ssh to use, or None. On PATH wins everywhere; the install places are tried on Windows only."""
    e = _env(env)
    win = (os.name == "nt") if windows is None else windows
    path_env = _getenv(e, "PATH")
    on_path = which("ssh", path=path_env) if _accepts_path(which) else which("ssh")
    if on_path:
        return SshFound(on_path, "PATH", True)
    if not win:
        return None
    for source, p in candidates(e):
        if isfile(p):
            return SshFound(p, source, False)
    return None


def _accepts_path(which) -> bool:
    return which is shutil.which


def prepend_to_path(folder: str, env: Optional[Dict[str, str]] = None) -> None:
    """Put `folder` first in PATH (in this process's environment when env is None)."""
    target = os.environ if env is None else env
    key = next((k for k in target if k.lower() == "path"), "PATH")
    cur = target.get(key, "")
    sep = ";" if "\\" in folder else os.pathsep         # a Windows folder: Windows' separator, whatever runs this
    if folder.lower().rstrip("\\") in [p.lower().rstrip("\\") for p in cur.split(sep)]:
        return
    target[key] = folder + (sep + cur if cur else "")


def missing_message(windows: Optional[bool] = None) -> str:
    win = (os.name == "nt") if windows is None else windows
    return "mace gui connect: no ssh command found. " + (INSTALL_WINDOWS if win else INSTALL_POSIX)


def found_note(f: SshFound) -> str:
    """One line for a user whose PATH did not have ssh."""
    why = {"Windows OpenSSH": "Windows' own OpenSSH client", "Win32-OpenSSH": "the Win32-OpenSSH client",
           "Git for Windows": "the ssh of Git for Windows", "VS Code setting": "the ssh VS Code uses "
           "(remote.SSH.path)"}.get(f.source, f.source)
    line = f"ssh is not on your PATH; using {f.path} ({why}) for this run."
    if f.source == "Git for Windows":
        line += (" It keeps its own agent: ssh-add in Git Bash, not the Windows OpenSSH agent. Windows' own client "
                 "is the supported one (Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0).")
    else:
        line += f" To fix PATH for good, add {f.folder} to your user PATH (Settings > System > About > Advanced system settings > Environment Variables)."
    return line


def ensure_ssh(say: Callable[[str], None] = print, env: Optional[Dict[str, str]] = None,
               which: Callable[[str], Optional[str]] = shutil.which,
               isfile: Callable[[str], bool] = os.path.isfile, windows: Optional[bool] = None
               ) -> Tuple[Optional[SshFound], Optional[str]]:
    """(found, None) with its folder first in PATH, or (None, message) when there is no ssh."""
    f = find_ssh(env, which=which, isfile=isfile, windows=windows)
    if f is None:
        return None, missing_message(windows)
    if not f.on_path:
        prepend_to_path(f.folder, env)
        say(found_note(f))
    return f, None


if __name__ == "__main__":  # pragma: no cover
    got, why = ensure_ssh()
    print(got.path if got else why, file=sys.stderr if not got else sys.stdout)
