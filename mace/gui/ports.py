"""Ports for ``mace gui`` on a computer several people share.

On your own computer ``mace gui`` keeps its port, 8765 (your browser keeps its
drafts per port). On a shared node (an ssh session or a batch job: two people
in the same group on the same development node) 8765 would belong to whoever
started first, so the default there is a port of your own:

    20000 + (your user id mod 10000)       (Windows: from a hash of the user name)

the same number on every node and every day, so your drafts and your
``ssh -L`` line stay the same. If another process already holds it (a user
whose id ends the same way, or another program), the next free port above
it is used and the line says so; a port you ask for with ``--port`` or
``MACE_GUI_PORT`` is used exactly, and reported busy if it is.

``MACE_GUI_SHARED=1`` (or ``0``) overrides the shared-node guess.
Standard library only.
"""
from __future__ import annotations

import os
import socket
import zlib
from typing import Callable, Mapping, Optional, Tuple

LOCAL_DEFAULT = 8765
USER_PORT_LOW = 20000
USER_PORT_SPAN = 10000          # 20000-29999: below Linux's ephemeral range, none of a browser's blocked ports
PROBE_TRIES = 50


def _env(env: Optional[Mapping[str, str]]) -> Mapping[str, str]:
    return os.environ if env is None else env


def user_slot(user: Optional[str] = None, uid: Optional[int] = None) -> int:
    """0..9999 from your user id (or, without ids, from the user name)."""
    if uid is None and hasattr(os, "getuid"):
        uid = os.getuid()
    if uid is not None:
        return int(uid) % USER_PORT_SPAN
    if user is None:
        import getpass
        user = getpass.getuser()
    return zlib.crc32(user.encode("utf-8")) % USER_PORT_SPAN


def user_port(user: Optional[str] = None, uid: Optional[int] = None) -> int:
    return USER_PORT_LOW + user_slot(user, uid)


def shared_node(env: Optional[Mapping[str, str]] = None) -> bool:
    """A computer other people log in to: an ssh session or a batch job.
    MACE_GUI_SHARED=1/0 decides when set."""
    e = _env(env)
    forced = (e.get("MACE_GUI_SHARED") or "").strip().lower()
    if forced in ("1", "true", "yes", "on"):
        return True
    if forced in ("0", "false", "no", "off"):
        return False
    return bool(e.get("SSH_CONNECTION") or e.get("SSH_CLIENT") or e.get("SLURM_JOB_ID"))


def default_port(env: Optional[Mapping[str, str]] = None, user: Optional[str] = None,
                 uid: Optional[int] = None) -> Tuple[int, str]:
    """(port, "user" or "local"): the port `mace gui` starts from when you name none."""
    if shared_node(env):
        return user_port(user, uid), "user"
    return LOCAL_DEFAULT, "local"


def is_free(port: int, host: str = "127.0.0.1") -> bool:
    """Whether `host`:`port` can be bound now (the same test the server's own bind makes)."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as s:
        try:
            if os.name != "nt":
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
        except OSError:
            return False
    return True


def is_free_loopback(port: int, host: str = "127.0.0.1") -> bool:
    """is_free for the server's address and, where the computer has IPv6, for ::1 too (the server listens on both)."""
    if not is_free(port, host):
        return False
    if host in ("127.0.0.1", "localhost") and socket.has_ipv6:
        try:
            probe = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        except OSError:
            return True
        probe.close()
        return is_free(port, "::1") or not _v6_loopback_usable()
    return True


def _v6_loopback_usable() -> bool:
    """Whether ::1 can be bound at all here (a container may have IPv6 without a loopback address)."""
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
        return True
    except OSError:
        return False


def pick(start: int, host: str = "127.0.0.1", tries: int = PROBE_TRIES,
         free: Callable[[int, str], bool] = is_free, skip=()) -> Optional[int]:
    """`start`, or the next free port above it (not in `skip`), within `tries` steps; None if all are busy."""
    for p in range(start, min(start + tries, 65535) + 1):
        if p in skip:
            continue
        if free(p, host):
            return p
    return None


def forward_lines(port: int, node: str, user: str, gateway: Optional[str]) -> list:
    """The exact lines that forward `port` of `node` to the same port on your computer: ssh through the
    gateway (bash, zsh, PowerShell, cmd and MobaXterm take the same line)."""
    gw = gateway or "LOGIN_HOST"
    if "@" not in gw:
        gw = f"{user}@{gw}"
    return [f"ssh -N -J {gw} -L {port}:127.0.0.1:{port} {user}@{node}",
            f"then open http://127.0.0.1:{port}/ on your computer (the sign-in link carries the rest)"]
