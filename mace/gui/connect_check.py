"""``mace gui connect --check``: read-only diagnostics of the ssh setup a
connection profile needs, with one OK, WARN, FAIL (or INFO, SKIP) line per
finding and, under each problem, one exact line that fixes it.

1. the ssh command and its version (Windows: OpenSSH 8.1+ for -J, and
   Windows' own OpenSSH rather than another ssh earlier on PATH);
2. ``ssh -G <gateway>``: user, host name, the key files that exist,
   connection sharing, ProxyJump, an email-like or unresolvable gateway;
3. ``ssh -G <node>`` against the gateway: what ``node_key_options`` adds for
   the jump hop, and whether a plain ``ssh -J`` would ask for the node's
   password (the node gets none of the gateway's key files and the agent
   lacks them);
4. the agent: ``ssh-add -l`` (fingerprints only), the gateway's key files
   matched by their ``.pub`` fingerprints (``ssh-keygen -lf``); Windows: the
   ssh-agent service (when it is disabled, ``ssh-keygen -y -P ""`` asks
   whether each key has a passphrase: it opens the private key, and only its
   exit code is kept);
5. permissions of ~/.ssh, its config and the key files (Windows: a hint);
6. live probes, skipped with ``--no-net``: ``ssh -o BatchMode=yes <gateway>
   true``, then the jump hop and the nested hop, each running only ``true``
   (a live ControlMaster connection is reused; a gateway that wants a
   password or 2FA code stops the probes);
7. the verdict: which hop works, and the exact ``mace gui connect`` line.

``--fix-config`` prints (never writes) a suggested ~/.ssh/config block from
``ssh -G``. Nothing here reads or stores a passphrase or a private key.

The message texts live in MESSAGES; the standalone scripts
(scripts/mace-gui-connect.sh ``--check``, .ps1 ``-Check``) print the same
words (tests/gui/test_connect_check.py). Standard library only.
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import connect as C
from . import sshfind

PROBE_TIMEOUT_S = 45
PROBE_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
DEFAULT_KEY_NAMES = ("id_rsa", "id_ecdsa", "id_ecdsa_sk", "id_ed25519", "id_ed25519_sk", "id_xmss", "id_dsa")
WIN_SSH = r"C:\Windows\System32\OpenSSH"

# key: (platforms, text). platforms: "all", "posix" or "win" (the scripts that
# must carry the text: bash for all/posix, PowerShell for all/win). Texts use
# {placeholders} and no double quote, dollar sign or backtick, so that each
# script holds them verbatim in its own strings.
MESSAGES: Dict[str, Tuple[str, str]] = {
    "header": ("all", "Checking the ssh setup for {what} (read-only; nothing is changed)"),
    "ssh-ok": ("all", "ssh: {version}"),
    "ssh-missing": ("all", "no ssh command on PATH: connect needs the OpenSSH client"),
    "ssh-missing-fix": ("posix", "install the OpenSSH client (the openssh-client or openssh package)"),
    "ssh-missing-fix-win": ("win", "Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0 (in an "
                                   "administrator PowerShell)"),
    "ssh-off-path": ("win", "ssh is not on your PATH, but {path} exists ({source}); this check and mace gui connect "
                            "use it"),
    "ssh-off-path-fix": ("win", "add {folder} to your user PATH (Settings > System > About > Advanced system "
                                "settings > Environment Variables), then open a new terminal"),
    "ssh-missing-where": ("win", "looked in System32 OpenSSH, Program Files OpenSSH, Git for Windows (usr bin) and "
                                 "VS Code's remote.SSH.path"),
    "ssh-old": ("win", "Windows OpenSSH {version} is older than 8.1, whose -J (the jump hop) is unreliable"),
    "ssh-old-fix": ("win", "update the Windows OpenSSH client, or use the nested hop"),
    "ssh-not-system": ("win", "this ssh is {path}, not Windows' own OpenSSH ({system}): it may use another agent "
                              "and config"),
    "ssh-not-system-fix": ("win", "put {system} first in PATH (Get-Command ssh shows which ssh runs)"),
    "g-failed": ("all", "ssh -G {host} printed nothing: ssh cannot read its settings for {host}"),
    "g-failed-fix": ("all", "ssh -G {host} (shows ssh's own error, often a typo in ~/.ssh/config)"),
    "gw-ok": ("all", "{role} {host}: user {user}, HostName {hostname}, port {port}"),
    "gw-proxyjump": ("all", "{host} is reached through ProxyJump {jump} (from your ssh config)"),
    "email": ("all", "the gateway {host} looks like an email address. It should be user@login-host (your "
                     "cluster's login node) or a Host alias from ~/.ssh/config"),
    "email-fix": ("all", "{cmd} --gateway <user@login-host> --save"),
    "unresolvable": ("all", "{host} matches no Host in ~/.ssh/config and does not resolve (DNS)"),
    "unresolvable-fix": ("all", "use your cluster's login host (e.g. login.example.edu), or add a Host block "
                                "with its HostName to ~/.ssh/config"),
    "keys-ok": ("all", "key files ssh offers {host}: {files}"),
    "keys-none": ("all", "none of the key files ssh offers {host} exist; only an agent's keys can log in"),
    "keys-none-fix": ("all", "ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_cluster, then put its .pub in "
                             "~/.ssh/authorized_keys on the cluster (docs/ssh-setup.md)"),
    "cm-on": ("posix", "connection sharing is on for {host} (ControlMaster {value})"),
    "cm-win": ("win", "ControlMaster is set for {host}, which Windows OpenSSH does not support"),
    "cm-win-fix": ("win", "remove the ControlMaster, ControlPath and ControlPersist lines for {host} from "
                          "~/.ssh/config"),
    "node-ok-g": ("all", "node {node}: user {user}, key files {files}"),
    "node-adds": ("all", "for the jump hop, the connection also gives {node}: {opts}"),
    "node-prompt": ("all", "{node} gets none of {files} from your ssh config, and the agent does not hold "
                           "them, so a plain ssh -J {gateway} {node} would ask for the "
                           "password of {node}"),
    "node-prompt-fix": ("all", "the connection adds them for you (above); for plain ssh too, add a Host "
                                 "block for the node (--fix-config prints one) or run ssh-add {file}"),
    "node-ok": ("all", "the jump hop offers {node} your gateway key ({how})"),
    "node-nested": ("all", "nested hop: the gateway logs in to {node} with the keys it has there; your keys "
                           "here are not used for that hop"),
    "agent-ok": ("all", "agent: {count} key(s): {fps}"),
    "agent-empty": ("all", "the ssh agent holds no keys"),
    "agent-empty-fix": ("all", "ssh-add {key}"),
    "agent-none": ("all", "cannot reach an ssh agent ({why})"),
    "agent-none-fix": ("posix", 'eval "$(ssh-agent)" && ssh-add {key}'),
    "agent-none-fix-win": ("win", "Start-Service ssh-agent, then ssh-add {key}"),
    "agent-disabled": ("win", "the OpenSSH Authentication Agent service (ssh-agent) is disabled"),
    "agent-disabled-fix": ("win", "in an administrator PowerShell: Get-Service ssh-agent | Set-Service "
                                  "-StartupType Automatic; Start-Service ssh-agent"),
    "agent-stopped": ("win", "the OpenSSH Authentication Agent service (ssh-agent) is {status} (startup type "
                             "{start})"),
    "agent-stopped-fix": ("win", "Start-Service ssh-agent"),
    "agent-service-missing": ("win", "no OpenSSH Authentication Agent service (ssh-agent) on this computer"),
    "agent-service-missing-fix": ("win", "add the OpenSSH Client in Settings, Optional features"),
    "key-in-agent": ("all", "{file} ({fp}) is in the agent"),
    "key-not-in-agent": ("all", "{file} ({fp}) is not in the agent: with a passphrase, ssh must ask for it on "
                                "every hop, and a connection may have no terminal to ask on"),
    "key-not-in-agent-fix": ("all", "ssh-add {file}"),
    "key-no-pub": ("all", "{file}.pub is missing, so the check cannot compare {file} with the agent"),
    "key-no-pub-fix": ("all", "ssh-keygen -y -f {file} > {pub}"),
    "perm-ok": ("posix", "permissions: ~/.ssh, its config and your key files are private"),
    "perm-dir": ("posix", "~/.ssh is mode {mode}; ssh wants 700"),
    "perm-dir-fix": ("posix", "chmod 700 ~/.ssh"),
    "perm-file": ("posix", "{file} is mode {mode}; ssh wants 600 and refuses a key or config others can use"),
    "perm-file-fix": ("posix", "chmod 600 {file}"),
    "perm-win": ("win", "if ssh says UNPROTECTED PRIVATE KEY FILE or Bad permissions, make each key yours alone"),
    "perm-win-fix": ("win", 'icacls <key file> /inheritance:r /grant:r "$($env:USERNAME):(R)"'),
    "probe-skip": ("all", "live probes skipped: no ssh login was tried"),
    # mace gui connect itself (connect.py), and the scripts' connections
    "behind-jump": ("all", "{what} did not answer through {via}: it may be down or not accept logins (try another "
                           "node, or {check}). Or the login took too long: ssh waited {wait}, and that includes "
                           "logging in to {via} (ssh calls the node UNKNOWN port 65535 then). If typing the "
                           "passphrase or approving 2FA took longer, load your key into the agent once with ssh-add "
                           "{key}, so that ssh asks for the passphrase once instead of on each hop, or allow more "
                           "time with {opt}."),
    # node_probe (connect.py and the bash script; the PowerShell script has no probe)
    "node-down": ("posix", "the node {node} did not answer through the gateway {gateway}: it may be down or not accept "
                         "logins (its ssh sent nothing within {probe} s when the gateway tried it). Try another node, "
                         "or {check}."),
    "waiting-node": ("all", "(waiting for {node} through the gateway {gateway}, up to {wait}; if {node} is down, "
                            "press Ctrl+C and pick another node)"),
    "passphrase-hops": ("all", "{key} has a passphrase and no ssh agent holds it, so ssh asks for it {times}. To "
                               "type it once per session, first run: {cmd}"),
    "cm-live": ("posix", "a shared connection to {host} is open: the probes reuse it"),
    "probe-ok": ("all", "ssh {host} true: logged in without a prompt"),
    "unreachable": ("all", "cannot reach host {host} ({why})"),
    "node-silent": ("all", "{node} did not answer through the gateway {gateway} in time (ssh calls it UNKNOWN port "
                           "65535): it may be down or not accept logins"),
    "node-silent-fix": ("all", "pick another node ({flag}), then check again"),
    "unreachable-fix": ("all", "check the name, and your network or VPN (ssh -v {host} true shows where it "
                               "stops)"),
    "hostkey": ("all", "ssh does not know the host key of {host} yet, or it changed"),
    "hostkey-fix": ("all", "ssh {host} true (once, in this terminal, to check and accept the key)"),
    "twofa": ("all", "{host} asks for a 2FA code or a password, which the check cannot type; the hop probes "
                     "stop here"),
    "twofa-fix": ("posix", "turn on connection sharing (ControlMaster, see --fix-config), log in once with ssh "
                           "{host}, then check again; or skip the probes with --no-net"),
    "twofa-fix-win": ("win", "skip the probes with -NoNet (Windows ssh has no connection sharing); connecting "
                             "shows the prompt"),
    "probe-key": ("all", "{host} refused your keys (Permission denied: {methods})"),
    "probe-key-fix": ("all", "ssh-add your key, and check that its .pub is in ~/.ssh/authorized_keys on the "
                             "cluster (docs/ssh-setup.md)"),
    "probe-fail": ("all", "ssh {host} true failed (exit {code}): {line}"),
    "probe-fail-fix": ("all", "ssh -v {host} true"),
    "hop-jump-ok": ("all", "jump hop: ssh -J {gateway} {node} true works"),
    "hop-jump-refused": ("all", "jump hop: {node} refused your key from this computer (a connection would ask "
                                "for the password of {node})"),
    "hop-jump-refused-fix": ("all", "use the nested hop, or give {node} your key "
                                    "(docs/gui.md#mace-gui-connect-the-node-asks-for-a-password)"),
    "hop-jump-fail": ("all", "jump hop: ssh -J {gateway} {node} true failed (exit {code}): {line}"),
    "hop-nested-ok": ("all", "nested hop: the gateway logs in to {node} by itself"),
    "hop-nested-fail": ("all", "nested hop: the gateway could not log in to {node} (exit {code}): {line}"),
    "hop-nested-fail-fix": ("all", "log in to the gateway and run ssh {node} true there: the cluster may need a "
                                   "key for logins between its own hosts"),
    "verdict-ok": ("all", "Verdict: the {hop} hop works. Connect with: {cmd}"),
    "verdict-nested": ("all", "Verdict: only the nested hop works. Connect with: {cmd}"),
    "verdict-gateway": ("all", "Verdict: MACE GUI runs on the gateway itself. Connect with: {cmd}"),
    "verdict-untested": ("all", "Verdict (without the probes): no problem found. Connect with: {cmd}"),
    "verdict-fail": ("all", "Verdict: fix the FAIL lines above, then check again."),
    "fixcfg-head": ("all", "# A suggested ~/.ssh/config block for this profile, from ssh -G (printed only: "
                           "nothing was written). Review it, then add it to ~/.ssh/config."),
    "fixcfg-merge": ("all", "# You already have a Host {alias} block: merge these lines into it."),
    "fixcfg-nokey": ("all", "# No key file exists yet: make one first, ssh-keygen -t ed25519 -f {key}"),
    "fixcfg-then": ("all", "# Then connect with: {cmd}"),
}


_PLAIN = re.compile(r"[A-Za-z0-9_@%+=:,./-]+")


def shell_path(path: str, windows: bool) -> str:
    """A path as one argument of a fix line's shell: PowerShell's single quotes on Windows, POSIX sh's
    elsewhere (a key under C:\\Users\\Jane Doe, or ~/my keys/id). A leading ~/ stays outside the quotes."""
    p = str(path)
    head, rest = (p[:2], p[2:]) if p.startswith(("~/", "~\\")) else ("", p)
    if not rest or _PLAIN.fullmatch(rest.replace("\\", "/") if windows else rest):
        return head + rest
    return head + ("'" + rest.replace("'", "''") + "'" if windows else shlex.quote(rest))


def msg(_key: str, **kw: Any) -> str:
    return MESSAGES[_key][1].format(**kw)


def system_name() -> str:
    return "windows" if os.name == "nt" else "darwin" if sys.platform == "darwin" else "linux"


def _expand(path: str, home: Path) -> Path:
    if path == "~":
        return home
    if path.startswith("~/") or path.startswith("~\\"):
        return home / path[2:]
    return Path(path)


def _fp_of_ssh_add(line: str) -> Optional[str]:
    """``256 SHA256:abc... comment (ED25519)`` -> ``SHA256:abc... (ED25519)``: the
    comment (often an email address or a path) is left out."""
    parts = line.split()
    if len(parts) < 2 or not parts[0].isdigit():
        return None
    kind = parts[-1] if parts[-1].startswith("(") else ""
    return f"{parts[1]} {kind}".strip()


def denied_methods(stderr_lines: Sequence[str]) -> Optional[List[str]]:
    for ln in stderr_lines:
        m = re.search(r"Permission denied \(([^)]*)\)", ln)
        if m:
            return [x.strip() for x in m.group(1).split(",") if x.strip()]
    return None


class Report:
    def __init__(self, out: Callable[[str], None] = print):
        self.out = out
        self.levels: List[str] = []

    def add(self, level: str, text: str, fix: Optional[str] = None) -> None:
        self.levels.append(level)
        self.out(f"{level:<5} {text}")
        if fix:
            self.out(f"      fix: {fix}")

    @property
    def failed(self) -> bool:
        return "FAIL" in self.levels


class Checker:
    """One run of the checks for a profile (a connect profile dict). Every
    outside call goes through ``runner`` (subprocess.run's signature),
    ``which`` and ``resolver``, so tests can fake them."""

    def __init__(self, profile: Dict[str, Any], name: Optional[str] = None, *, no_net: bool = False,
                 runner: Callable = subprocess.run, which: Callable = shutil.which,
                 resolver: Callable = socket.getaddrinfo, system: Optional[str] = None,
                 home: Optional[Path] = None, env: Optional[Dict[str, str]] = None,
                 out: Callable[[str], None] = print):
        self.prof = profile
        self.name = name
        self.no_net = no_net
        self.runner = runner
        self.which = which
        self.resolver = resolver
        self.system = system or system_name()
        self.win = self.system == "windows"
        self.home = home or Path(os.path.expanduser("~"))
        self.env = dict(os.environ if env is None else env)
        self.r = Report(out)
        self.out = out
        self.gateway = str(profile.get("gateway") or "")
        self.node = str(profile.get("node") or "")
        self.first = self.gateway or self.node
        self.hop = str(profile.get("hop") or "auto")
        self.jump = str(profile.get("jump") or "")
        self._g: Dict[str, Dict[str, List[str]]] = {}
        self.agent_deferred: Optional[Tuple[str, str]] = None
        self._agent: Optional[List[str]] = None
        self._fp: Dict[str, Optional[str]] = {}

    # ---------------------------------------------------------------- helpers
    def run(self, argv: List[str], timeout: float = 15, env: Optional[Dict[str, str]] = None):
        kw: Dict[str, Any] = {"capture_output": True, "text": True, "timeout": timeout,
                              "stdin": subprocess.DEVNULL}
        if env is not None:
            kw["env"] = env
        if os.name == "posix" and self.runner is subprocess.run:
            kw["start_new_session"] = True       # no terminal: ssh cannot prompt, it fails instead
        try:
            return self.runner(argv, **kw)
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(argv, None, "", f"timed out after {int(timeout)} s")
        except (OSError, subprocess.SubprocessError) as exc:
            return subprocess.CompletedProcess(argv, None, "", str(exc))

    def g(self, host: str) -> Dict[str, List[str]]:
        if host not in self._g:
            self._g[host] = C.ssh_g(host, self.runner)
        return self._g[host]

    def existing(self, files: Sequence[str]) -> List[str]:
        return [f for f in files if _expand(f, self.home).is_file()]

    def fingerprint(self, f: str) -> Optional[str]:
        """The SHA256 fingerprint of f's public half (f.pub), or None."""
        if f not in self._fp:
            pub = _expand(f, self.home).with_name(_expand(f, self.home).name + ".pub")
            fp = None
            if pub.is_file():
                r = self.run(["ssh-keygen", "-lf", str(pub)])
                parts = (r.stdout or "").split()
                fp = parts[1] if r.returncode == 0 and len(parts) > 1 else None
            self._fp[f] = fp
        return self._fp[f]

    def cmd(self, hop: Optional[str] = None, save: bool = False) -> str:
        argv = ["mace", "gui", "connect"]
        if self.name:
            argv.append(self.name)
        else:
            argv += ["--gateway", self.gateway] if self.gateway else []
            argv += ["--node", self.node] if self.node else []
            argv += ["--jump", self.jump] if self.jump else []
        if hop:
            argv += ["--hop", hop]
        if save:
            argv.append("--save")
        return shlex.join(argv)

    def gateway_keys(self) -> List[str]:
        return self.existing(self.g(self.first).get("identityfile", [])) if self.first else []

    # ---------------------------------------------------------------- the checks
    def check_ssh(self) -> bool:
        path = self.which("ssh")
        if not path and self.win:
            found = sshfind.find_ssh(self.env, which=self.which, windows=True)
            if found:
                path = found.path
                self.r.add("WARN", msg("ssh-off-path", path=found.path, source=found.source),
                           msg("ssh-off-path-fix", folder=found.folder))
                sshfind.prepend_to_path(found.folder)
        if not path:
            self.r.add("FAIL", msg("ssh-missing"), msg("ssh-missing-fix-win" if self.win else "ssh-missing-fix"))
            if self.win:
                self.r.add("INFO", msg("ssh-missing-where"))
            return False
        v = self.run(["ssh", "-V"])
        version = ((v.stderr or "") + (v.stdout or "")).strip().splitlines()
        version_text = version[0] if version else "unknown version"
        self.r.add("OK", msg("ssh-ok", version=f"{version_text} ({path})"))
        if self.win:
            if C.windows_ssh_too_old(version_text):
                self.r.add("WARN", msg("ssh-old", version=version_text), msg("ssh-old-fix"))
            if not str(path).lower().startswith(WIN_SSH.lower()):
                self.r.add("WARN", msg("ssh-not-system", path=path, system=WIN_SSH + "\\ssh.exe"),
                           msg("ssh-not-system-fix", system=WIN_SSH))
        return True

    def check_gateway(self) -> bool:
        host = self.first
        g = self.g(host)
        if not g:
            self.r.add("FAIL", msg("g-failed", host=host), msg("g-failed-fix", host=host))
            return False
        role = "gateway" if self.gateway else "node"
        self.r.add("OK", msg("gw-ok", role=role, host=host, user=(g.get("user") or ["?"])[0],
                             hostname=(g.get("hostname") or [host])[0], port=(g.get("port") or ["22"])[0]))
        pj = (g.get("proxyjump") or ["none"])[0]
        if pj.lower() != "none":
            self.r.add("INFO", msg("gw-proxyjump", host=host, jump=pj))
        hosts = C.ssh_config_hosts(self.home / ".ssh" / "config")
        if self.gateway and C.gateway_looks_like_email(self.gateway, hosts):
            base = shlex.join(["mace", "gui", "connect"] + ([self.name] if self.name else []))
            self.r.add("WARN", msg("email", host=self.gateway), msg("email-fix", cmd=base))
        if self.gateway and not self.no_net:
            bad = C.gateway_unresolvable(self.gateway, self.jump, self.runner, self.resolver, hosts)
            if bad:
                self.r.add("WARN", msg("unresolvable", host=bad), msg("unresolvable-fix"))
        files = self.existing(g.get("identityfile", []))
        if files:
            extra = " (IdentitiesOnly yes)" if (g.get("identitiesonly") or [""])[0] == "yes" else ""
            self.r.add("OK", msg("keys-ok", host=host, files=", ".join(files)) + extra)
        else:
            self.r.add("WARN", msg("keys-none", host=host), msg("keys-none-fix"))
        cm = (g.get("controlmaster") or ["false"])[0].lower()
        if cm not in ("false", "no"):
            if self.win:
                self.r.add("WARN", msg("cm-win", host=host), msg("cm-win-fix", host=host))
            else:
                self.r.add("OK", msg("cm-on", host=host, value=cm))
        return True

    def check_node(self) -> None:
        if not (self.node and self.gateway):
            return
        if self.hop == "nested":
            self.r.add("INFO", msg("node-nested", node=self.node))
            return
        unode = C.effective_node_user(self.gateway, self.runner) if "@" not in self.node else None
        target = C.node_target(self.prof, unode)
        n = self.g(target)
        nfiles = self.existing(n.get("identityfile", []))
        self.r.add("INFO", msg("node-ok-g", node=self.node, user=(n.get("user") or ["?"])[0],
                               files=", ".join(nfiles) or "none that exist"))
        adds = C.node_key_options(self.gateway, target, self.runner)
        if adds:
            self.r.add("INFO", msg("node-adds", node=self.node, opts=" ".join(adds)))
        gfiles = self.gateway_keys()
        missing = [f for f in gfiles if f not in n.get("identityfile", [])]
        if not gfiles:
            return
        if len(missing) < len(gfiles):
            self.r.add("OK", msg("node-ok", node=self.node, how="from your ssh config"))
            return
        agent = set(self.agent_fps() or [])
        held = [f for f in missing if self.fingerprint(f) and self.fingerprint(f) in agent]
        if held:
            self.r.add("OK", msg("node-ok", node=self.node, how=f"the agent holds {held[0]}"))
            return
        self.r.add("WARN", msg("node-prompt", gateway=self.gateway, node=self.node, files=", ".join(missing)),
                   msg("node-prompt-fix", file=shell_path(missing[0], self.win)))

    def agent_env(self) -> Dict[str, str]:
        env = dict(self.env)
        ia = (self.g(self.first).get("identityagent") or ["none"])[0] if self.first else "none"
        if ia.lower() not in ("none", "ssh_auth_sock") and not ia.startswith("$"):
            env["SSH_AUTH_SOCK"] = str(_expand(ia, self.home))
        return env

    def agent_fps(self) -> Optional[List[str]]:
        """The agent's key fingerprints; None when no agent answers."""
        if self._agent is None:
            r = self.run(["ssh-add", "-l"], env=self.agent_env())
            self._agent_rc = r.returncode
            self._agent_err = ((r.stderr or "") + (r.stdout or "")).strip()
            if r.returncode == 0:
                self._agent = [fp.split()[0] for fp in (_fp_of_ssh_add(ln) for ln in r.stdout.splitlines()) if fp]
                self._agent_lines = [fp for fp in (_fp_of_ssh_add(ln) for ln in r.stdout.splitlines()) if fp]
            else:
                self._agent = []
                self._agent_lines = []
        return self._agent if self._agent_rc in (0, 1) else None

    def keys_unencrypted(self, keys: List[str]) -> bool:
        """Every key file ssh offers exists and has no passphrase (ssh-keygen -y with an empty one succeeds;
        it prints only the public key)."""
        found = [f for f in keys if _expand(f, self.home).is_file()]
        if not found:
            return False
        return all(self.run(["ssh-keygen", "-y", "-P", "", "-f", str(_expand(f, self.home))]).returncode == 0
                   for f in found)

    def check_agent(self) -> None:
        keys = self.gateway_keys()
        key = keys[0] if keys else "~/.ssh/id_ed25519_cluster"      # as the other fix lines name it
        if self.win:
            r = self.run(["powershell", "-NoProfile", "-Command",
                          "$s = Get-Service ssh-agent -ErrorAction SilentlyContinue; "
                          "if ($s) { '' + $s.Status + ' ' + $s.StartType }"])
            words = (r.stdout or "").split()
            if not words:
                self.r.add("WARN", msg("agent-service-missing"), msg("agent-service-missing-fix"))
                return
            status, start = words[0], (words[1] if len(words) > 1 else "?")
            if start.lower() == "disabled":
                # ssh needs no agent for a key without a passphrase, and none when a live probe logs in: then
                # this is only a WARN. A passphrase key whose probes cannot run (--no-net) stays a FAIL; with
                # probes it is a WARN now and a FAIL after them if no login worked (agent_deferred).
                if self.keys_unencrypted(keys):
                    self.r.add("WARN", msg("agent-disabled"), msg("agent-disabled-fix"))
                elif self.no_net:
                    self.r.add("FAIL", msg("agent-disabled"), msg("agent-disabled-fix"))
                else:
                    self.agent_deferred = (msg("agent-disabled"), msg("agent-disabled-fix"))
                    self.r.add("WARN", msg("agent-disabled"), msg("agent-disabled-fix"))
                return
            if status.lower() != "running":
                self.r.add("WARN", msg("agent-stopped", status=status, start=start), msg("agent-stopped-fix"))
                return
        fps = self.agent_fps()
        if fps is None:
            env = self.agent_env()
            why = (self._agent_err.splitlines() or ["no answer"])[0]
            if not self.win and not env.get("SSH_AUTH_SOCK"):
                why = "SSH_AUTH_SOCK is not set in this terminal"
            self.r.add("WARN", msg("agent-none", why=why),
                       msg("agent-none-fix-win" if self.win else "agent-none-fix", key=shell_path(key, self.win)))
            return
        if not fps:
            self.r.add("WARN", msg("agent-empty"), msg("agent-empty-fix", key=shell_path(key, self.win)))
        else:
            self.r.add("OK", msg("agent-ok", count=len(fps), fps=", ".join(self._agent_lines)))
        for f in keys:
            fp = self.fingerprint(f)
            if fp is None:
                self.r.add("INFO", msg("key-no-pub", file=f), msg("key-no-pub-fix", file=shell_path(f, self.win),
                                                                           pub=shell_path(f + ".pub", self.win)))
            elif fp in fps:
                self.r.add("OK", msg("key-in-agent", file=f, fp=fp))
            else:
                self.r.add("WARN", msg("key-not-in-agent", file=f, fp=fp), msg("key-not-in-agent-fix", file=shell_path(f, self.win)))

    def check_permissions(self) -> None:
        if self.win:
            self.r.add("INFO", msg("perm-win"), msg("perm-win-fix"))
            return
        d = self.home / ".ssh"
        if not d.is_dir():
            return
        found = False
        mode = d.stat().st_mode & 0o777
        if mode & 0o077:
            self.r.add("WARN", msg("perm-dir", mode=f"{mode:o}"), msg("perm-dir-fix"))
            found = True
        files = ["~/.ssh/config"] + [f for f in self.gateway_keys() if f != "~/.ssh/config"]
        if self.node and self.gateway:
            files += [f for f in self.existing(self.g(self.node).get("identityfile", [])) if f not in files]
        for f in files:
            p = _expand(f, self.home)
            if not p.is_file():
                continue
            m = p.stat().st_mode & 0o777
            if m & 0o077:
                self.r.add("FAIL" if f != "~/.ssh/config" or m & 0o022 else "WARN",
                           msg("perm-file", file=f, mode=f"{m:o}"), msg("perm-file-fix", file=shell_path(f, self.win)))
                found = True
        if not found:
            self.r.add("OK", msg("perm-ok"))

    # ---------------------------------------------------------------- probes
    def classify(self, r, host: str, hop: str = "") -> Tuple[str, str, Optional[str]]:
        """(level, text, fix) for a failed probe of host (hop: "" for the
        gateway, "jump" for the node through it)."""
        lines = [ln for ln in (r.stderr or "").splitlines() if ln.strip()]
        gone = C.unreachable(lines, self.node if hop else "")
        if gone and hop and gone[0] in ("UNKNOWN", "the host"):
            return "FAIL", msg("node-silent", node=self.node, gateway=self.gateway), msg("node-silent-fix", flag="--node")
        if gone:
            return "FAIL", msg("unreachable", host=gone[0], why=gone[1]), msg("unreachable-fix", host=gone[0])
        if any("Host key verification failed" in ln or "IDENTIFICATION HAS CHANGED" in ln for ln in lines):
            return "FAIL", msg("hostkey", host=host), msg("hostkey-fix", host=host)
        last = lines[-1] if lines else "no message"
        if hop:
            if C.node_auth_refused(lines, self.node, self.gateway):
                return "FAIL", msg("hop-jump-refused", node=self.node), msg("hop-jump-refused-fix", node=self.node)
            return "FAIL", msg("hop-jump-fail", gateway=self.gateway, node=self.node, code=r.returncode,
                               line=last), None
        methods = denied_methods(lines)
        if methods is not None:
            if "publickey" not in methods:
                return "WARN", msg("twofa", host=host), msg("twofa-fix-win" if self.win else "twofa-fix", host=host)
            return "FAIL", msg("probe-key", host=host, methods=",".join(methods)), msg("probe-key-fix")
        return "FAIL", msg("probe-fail", host=host, code=r.returncode, line=last), msg("probe-fail-fix", host=host)

    def probes(self) -> Dict[str, Optional[bool]]:
        """The gateway, then both hops (whatever the profile's hop: the verdict
        names the one that works), each running only `true`."""
        res: Dict[str, Optional[bool]] = {"gateway": None, "jump": None, "nested": None}
        if self.no_net:
            self.r.add("SKIP", msg("probe-skip"))
            return res
        host = self.first
        jumps = ["-J", self.jump] if self.jump else []
        cm = (self.g(host).get("controlmaster") or ["false"])[0].lower()
        if not self.win and cm not in ("false", "no"):
            if self.run(["ssh", "-O", "check"] + jumps + [host]).returncode == 0:
                self.r.add("OK", msg("cm-live", host=host))
        # the profile's ssh options, where the connection passes them: the gateway's ssh of the
        # nested hop (and without a hop), the node's ssh of the jump hop (-J's gateway leg gets none)
        extra = [str(x) for x in self.prof.get("ssh_options") or []]
        gw_extra = extra if not (self.node and self.gateway) or self.hop == "nested" else []
        r = self.run(["ssh"] + jumps + PROBE_OPTS + gw_extra + [host, "true"], timeout=PROBE_TIMEOUT_S)
        res["gateway"] = r.returncode == 0
        if r.returncode == 0:
            self.r.add("OK", msg("probe-ok", host=host))
        else:
            self.r.add(*self.classify(r, host))
            return res
        if not (self.node and self.gateway):
            return res
        unode = C.effective_node_user(self.gateway, self.runner) if "@" not in self.node else None
        target = C.node_target(self.prof, unode)
        keys = C.node_key_options(self.gateway, target, self.runner)
        r = self.run(["ssh", "-J", ",".join(([self.jump] if self.jump else []) + [self.gateway])] + PROBE_OPTS
                     + keys + extra + [target, "true"], timeout=PROBE_TIMEOUT_S)
        res["jump"] = r.returncode == 0
        if r.returncode == 0:
            self.r.add("OK", msg("hop-jump-ok", gateway=self.gateway, node=self.node))
        else:
            self.r.add(*self.classify(r, self.node, hop="jump"))
        inner = shlex.join(["ssh"] + PROBE_OPTS + [self.node, "true"])
        r = self.run(["ssh"] + jumps + PROBE_OPTS + extra + [self.gateway, inner], timeout=PROBE_TIMEOUT_S)
        res["nested"] = r.returncode == 0
        if r.returncode == 0:
            self.r.add("OK", msg("hop-nested-ok", node=self.node))
        else:
            last = ([ln for ln in (r.stderr or "").splitlines() if ln.strip()] or ["no message"])[-1]
            self.r.add("FAIL", msg("hop-nested-fail", node=self.node, code=r.returncode, line=last),
                       msg("hop-nested-fail-fix", node=self.node))
        return res

    def verdict(self, res: Dict[str, Optional[bool]]) -> int:
        """Print the verdict; the exit code (0: a way to connect works or
        nothing failed; 1: a FAIL line stands). A hop that fails while the
        other one works is not a failure of the setup."""
        hop_fails = sum(1 for k in ("jump", "nested") if res.get(k) is False)
        others = self.r.levels.count("FAIL") - hop_fails
        if others == 0 and (res.get("jump") or res.get("nested")):
            if self.hop == "nested" and res.get("nested"):
                self.out(msg("verdict-ok", hop="nested", cmd=self.cmd()))
            elif res.get("jump"):
                hop = "jump" if self.hop == "nested" else None
                self.out(msg("verdict-ok", hop="jump", cmd=self.cmd(hop, save=bool(hop))))
            else:
                self.out(msg("verdict-nested", cmd=self.cmd("nested", save=True)))
            return 0
        if self.r.failed:
            self.out(msg("verdict-fail"))
            return 1
        if not (self.node and self.gateway) and res.get("gateway"):
            self.out(msg("verdict-gateway", cmd=self.cmd()))
        else:
            self.out(msg("verdict-untested", cmd=self.cmd()))
        return 0

    def run_all(self) -> int:
        what = (f"{self.name}: " if self.name else "") + (
            f"{self.gateway} -> {self.node}" if self.node and self.gateway else self.first)
        self.out(msg("header", what=what))
        if not self.check_ssh():            # no ssh client: exit 2, as a connect without one does
            self.out(msg("verdict-fail"))
            return 2
        if not self.check_gateway():
            self.out(msg("verdict-fail"))
            return 1
        self.check_node()
        self.check_agent()
        self.check_permissions()
        res = self.probes()
        if self.agent_deferred and not (res.get("jump") or res.get("nested")
                                        or (not (self.node and self.gateway) and res.get("gateway"))):
            self.r.add("FAIL", *self.agent_deferred)      # the logins failed, and no agent can hold the key
        return self.verdict(res)

    # ---------------------------------------------------------------- --fix-config
    def fix_config(self) -> List[str]:
        host = self.first
        g = self.g(host) if host else {}
        hosts = C.ssh_config_hosts(self.home / ".ssh" / "config")
        user = (g.get("user") or [""])[0]
        hostname = (g.get("hostname") or [host.rsplit("@", 1)[-1]])[0]
        if "@" in host or not host:
            alias = suggest_alias(hostname)
        else:
            alias = host
        keys = self.gateway_keys()
        custom = [f for f in keys if Path(f).name not in DEFAULT_KEY_NAMES]
        key = (custom or keys or ["~/.ssh/id_ed25519_cluster"])[0]
        lines = [msg("fixcfg-head")]
        if alias in hosts:
            lines.append(msg("fixcfg-merge", alias=alias))
        if not keys:
            lines.append(msg("fixcfg-nokey", key=key))
        lines += [f"Host {alias}", f"    HostName {hostname}"]
        if user:
            lines.append(f"    User {user}")
        pj = self.jump or (g.get("proxyjump") or ["none"])[0]
        if pj.lower() != "none":
            lines.append(f"    ProxyJump {pj}")
        lines += [f"    IdentityFile {key}", "    IdentitiesOnly yes", "    AddKeysToAgent yes"]
        if self.system == "darwin":
            lines.append("    UseKeychain yes")
        if not self.win:
            lines += ["    ControlMaster auto", "    ControlPath ~/.ssh/cm-%C", "    ControlPersist 30m"]
        lines.append("    ServerAliveInterval 30")
        if self.node and self.gateway:
            node = self.node.rsplit("@", 1)[-1]
            lines += ["", f"Host {node}"]
            if user:
                lines.append(f"    User {user}")
            lines += [f"    IdentityFile {key}", "    IdentitiesOnly yes", f"    ProxyJump {alias}"]
        argv = ["mace", "gui", "connect"] + ([self.name] if self.name else []) + ["--gateway", alias]
        if self.node:
            argv += ["--node", self.node.rsplit("@", 1)[-1]]
        lines.append(msg("fixcfg-then", cmd=shlex.join(argv + ["--save"])))
        return lines


def suggest_alias(hostname: str) -> str:
    """A short Host alias from a login host name: login.example.edu -> example,
    hpcc.example.edu -> hpcc."""
    labels = [x for x in hostname.lower().split(".") if x]
    if not labels:
        return "cluster"
    if labels[0] in ("login", "ssh", "gateway", "gw", "bastion") and len(labels) > 2:
        return labels[1]
    return labels[0] if not labels[0].replace("-", "").isdigit() else "cluster"


def unloaded_key(profile: Dict[str, Any], **kw: Any) -> Optional[Tuple[str, bool]]:
    """(key file, an agent answers) when a key file ssh offers the first host
    has a passphrase and the agent holds none of the keys offered there: ssh
    then asks for the passphrase on every hop (twice with the jump hop: the
    gateway's ssh and the node's each open the key). None when no key file
    exists, the agent holds one, or none has a passphrase. No network: ssh -G,
    ssh-add -l, ssh-keygen -lf, and ssh-keygen -y -P "" (its exit code only)."""
    ck = Checker(profile, no_net=True, out=lambda _t: None, **kw)
    if not ck.first:
        return None
    keys = ck.gateway_keys()
    if not keys:
        return None
    fps = ck.agent_fps()
    if fps and any(ck.fingerprint(f) in fps for f in keys if ck.fingerprint(f)):
        return None
    for f in keys:
        if not ck.keys_unencrypted([f]):
            return f, fps is not None
    return None


def passphrase_hint(key: str, agent: bool, twice: bool = False, windows: Optional[bool] = None) -> str:
    """The line that suggests ssh-add for unloaded_key's key, before connecting."""
    win = (system_name() == "windows") if windows is None else windows
    arg = shell_path(key, win)            # one argument of the shell shown (a space in a Windows user folder)
    if agent:
        cmd = f"ssh-add {arg}"
    elif win:
        cmd = f"Start-Service ssh-agent, then ssh-add {arg}"
    else:
        cmd = f'eval "$(ssh-agent)" && ssh-add {arg}'
    return msg("passphrase-hops", key=key, times="for the gateway and again for the node" if twice
               else "on each connection", cmd=cmd)


def main(profile: Dict[str, Any], name: Optional[str], *, check: bool, no_net: bool, fix_config: bool,
         **kw: Any) -> int:
    ck = Checker(profile, name, no_net=no_net, **kw)
    code = 0
    if check:
        code = ck.run_all()
    if fix_config:
        if check:
            ck.out("")
        for ln in ck.fix_config():
            ck.out(ln)
    return code
