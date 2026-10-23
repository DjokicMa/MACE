# SSH keys for `mace gui connect`

`mace gui connect` (and the standalone `mace-gui-connect.sh` and
`mace-gui-connect.ps1`) runs your computer's own `ssh`. It uses whatever key,
agent and `~/.ssh/config` that `ssh` uses, and it never asks for, reads or
stores a passphrase or a password. This page sets that up once, so that one
command opens the GUI:

1. [Make a key with a passphrase](#1-make-a-key-with-a-passphrase) and put its
   public half on the cluster.
2. [Load it into an agent](#2-load-the-key-into-an-agent), so you type the
   passphrase once per session, not on every hop.
3. [Write a `~/.ssh/config`](#3-the-ssh-config) for the login host (the
   *gateway*) and the dev nodes.
4. [Use it with `mace gui connect`](#4-use-it-with-mace-gui-connect).
5. [Check your setup](#5-check-your-setup) with `mace gui connect --check`.

The same key and config serve every way of opening the GUI from your
computer (the scripts, MobaXterm, VS Code, a plain tunnel); [Choose your
setup](gui.md#choose-your-setup) in docs/gui.md picks the way and its steps.

The examples use placeholders: `login.example.edu` (your cluster's login
host), `dev-node-01` (a dev node), `netid` (your cluster user name) and
`cluster` (the alias you pick). Your cluster's documentation names the real
ones; if it says how to register keys (some clusters use a web portal), do
what it says.

**Rules that keep your account safe**

- The **private** key (`id_ed25519_cluster`, no `.pub`) never leaves your
  computer: do not email it, copy it to the cluster or share it. Only the
  `.pub` file goes to the cluster.
- Give the key a passphrase. Never write the passphrase (or your cluster
  password) into a file, a script, a profile or a `--ssh-option`.
- One key per computer. When a laptop is lost, remove its line from
  `~/.ssh/authorized_keys` on the cluster.

## 1. Make a key with a passphrase

**Linux, macOS, WSL, MobaXterm's local terminal**

```
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_cluster -C "netid laptop"
ssh-copy-id -i ~/.ssh/id_ed25519_cluster.pub netid@login.example.edu
```

`ssh-keygen` asks for the passphrase twice. `ssh-copy-id` logs in once with
your cluster password (and 2FA) and appends the public key to
`~/.ssh/authorized_keys` on the cluster. Without `ssh-copy-id`, this does the
same:

```
cat ~/.ssh/id_ed25519_cluster.pub | ssh netid@login.example.edu "umask 077; mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

**Windows (PowerShell)**

Windows 10 and 11 include the OpenSSH client (`ssh`, `ssh-keygen`,
`ssh-add`). There is no `ssh-copy-id`; the second line does its job (`tr`
drops the Windows line endings):

```
ssh-keygen -t ed25519 -f $env:USERPROFILE\.ssh\id_ed25519_cluster -C "netid laptop"
Get-Content $env:USERPROFILE\.ssh\id_ed25519_cluster.pub | ssh netid@login.example.edu "umask 077; mkdir -p ~/.ssh && tr -d '\r' >> ~/.ssh/authorized_keys"
```

Most clusters share your home folder between the login host and the dev
nodes, so this one `authorized_keys` covers the nodes too. Check it with
`ssh -i ~/.ssh/id_ed25519_cluster netid@login.example.edu`: it should ask for
the key's passphrase (and 2FA, if your cluster uses it), not your password.

## 2. Load the key into an agent

An agent holds the unlocked key in memory, so `ssh` (and `mace gui connect`,
which may have no terminal to ask on) can use it without asking again. Check
what the agent holds with `ssh-add -l` ("The agent has no identities" means
it runs but is empty; "Could not open a connection to your authentication
agent" means none is running).

**Linux.** Most desktop sessions start an agent at login. Add the key for
the day (`-t` sets how long the agent keeps it):

```
ssh-add -t 10h ~/.ssh/id_ed25519_cluster
```

With no agent (a plain console, a server): `eval "$(ssh-agent)"` first, in the
same terminal. `AddKeysToAgent yes` in the config (below) adds the key the
first time `ssh` uses it.

**macOS.** The built-in agent can keep the passphrase in your login Keychain,
so you do not type it again after a restart:

```
ssh-add --apple-use-keychain ~/.ssh/id_ed25519_cluster
```

and add `UseKeychain yes` and `AddKeysToAgent yes` to the config (below).
(`--apple-use-keychain` is the newer name of `-K`.)

**Windows.** The **OpenSSH Authentication Agent** service is disabled by
default. Turn it on once, in PowerShell **run as administrator**:

```
Get-Service ssh-agent | Set-Service -StartupType Automatic
Start-Service ssh-agent
```

Then, in a normal PowerShell:

```
ssh-add $env:USERPROFILE\.ssh\id_ed25519_cluster
```

The Windows agent keeps the key across restarts (stored for your Windows
account), so this is once, not once per session: anyone signed in to your
Windows account can then use it without the passphrase. Remove it with
`ssh-add -d <file>` when you no longer want that.

**MobaXterm.** MobaXterm has its own agent, **MobAgent** (Settings ›
Configuration › SSH: tick "Use internal SSH agent MobAgent", add the key with
`+`, restart MobaXterm). It asks for the passphrase at startup, and
`ssh-add -l` in MobaXterm's **local terminal** lists the key, so
`bash mace-gui-connect.sh ...` run there uses it. MobAgent and the Windows
agent are separate: a key in one is not in the other, and MobaXterm's local
terminal runs its own `ssh` with its own home folder (check `echo $HOME`).
If you also run `mace gui connect` or `mace-gui-connect.ps1` from PowerShell,
the Windows route above is the simpler one to keep.

**VS Code.** Remote-SSH runs your computer's `ssh` (Windows: the built-in
OpenSSH) with the same agent and config, so the steps above cover it. Check
with `ssh-add -l` in a *local* VS Code terminal; restart VS Code after
starting the agent. With 2FA, turn on `remote.SSH.showLoginTerminal` to see
the prompt. The GUI then opens in your local browser either way: run
`mace gui` in the remote window's terminal and Ctrl+click the link (VS Code
forwards the port), or run `mace gui connect` in a local terminal
([docs/gui.md](gui.md#vs-code-remote-ssh)).

## 3. The ssh config

One file for every client: `~/.ssh/config` (Windows:
`C:\Users\<you>\.ssh\config`; MobaXterm's local terminal: `~/.ssh/config`
under its own home). Make it, or add to it:

```
# The login host (the gateway)
Host cluster
    HostName login.example.edu
    User netid
    IdentityFile ~/.ssh/id_ed25519_cluster
    IdentitiesOnly yes
    AddKeysToAgent yes

# The dev nodes, reached through the gateway
Host dev-*
    User netid
    ProxyJump cluster
    IdentityFile ~/.ssh/id_ed25519_cluster
    IdentitiesOnly yes
    AddKeysToAgent yes
```

- `Host dev-*` is a pattern: write the names your cluster's dev nodes use.
  The pattern must not match the gateway itself.
- Naming the key for the nodes too matters: with a jump (`ssh -J`), the node
  hop logs in from *your computer* with the settings for the node's name, not
  the gateway's ([why](gui.md#mace-gui-connect-the-node-asks-for-a-password)).
- `IdentitiesOnly yes` offers only this key, so a server that allows few
  attempts does not refuse you after a pile of other keys.
- Now `ssh cluster` and `ssh dev-node-01` both work, with one passphrase.

**macOS**: add this line inside `Host cluster` and `Host dev-*` (and
`IgnoreUnknown UseKeychain` at the very top of the file if the same config is
also used on Linux or Windows):

```
    UseKeychain yes
```

**2FA once, then reuse it (Linux, macOS, WSL).** With 2FA on the gateway,
each new connection asks again. Connection sharing reuses one approved
connection for later ones, including the jump of `mace gui connect`. Add to
`Host cluster`:

```
    ControlMaster auto
    ControlPath ~/.ssh/cm-%C
    ControlPersist 4h
```

The first `ssh cluster` asks for 2FA; for 4 hours after the last connection
closes, new ones do not. `ssh -O exit cluster` closes it. Anyone who can
use your account on this computer can use that open connection, so keep it to
your own computer. **Windows' OpenSSH does not support connection sharing**
(leave these lines out there): on Windows, the agent saves the passphrase,
but 2FA asks on each new connection.

## 4. Use it with `mace gui connect`

| Your setup | Command |
|---|---|
| The config above | `mace gui connect --gateway cluster --node dev-node-01 --dir '~/runs'` |
| Nodes with their own `Host` block (`ProxyJump` inside) | `mace gui connect --node dev-node-01 --dir '~/runs'` (no gateway: ssh's config makes the jump) |
| No config | `mace gui connect --gateway netid@login.example.edu --node dev-node-01 --dir '~/runs' --ssh-option=-oIdentityFile=~/.ssh/id_ed25519_cluster` |
| Windows | the same flags on `mace-gui-connect.ps1` (`-Gateway cluster -Node dev-node-01 -Dir '~/runs'`) |

- **`--gateway`** is the `Host` alias (`cluster`) or `netid@login-host`.
  `mace gui connect` reads the alias's `User` with `ssh -G` and logs in to the
  node as the same user. `--save` keeps the flags in a profile (no secrets go
  into it).
- **`--hop jump`** (`ssh -J`) logs in to the node from your computer: your key
  and agent cover both hops. `mace gui connect` also passes the gateway's
  `IdentityFile`, `IdentityAgent` and `IdentitiesOnly` to the node hop.
- **`--hop nested`** makes the gateway log in to the node itself. That inner
  ssh runs with `BatchMode=yes`: it cannot ask for anything, so it works only
  when the gateway can reach the node on its own (the cluster's host-based
  logins, or a key in your cluster home). Do not copy your private key to the
  cluster for this; use `--hop jump` instead.
- **`--hop auto`** (the default) tries `jump`, then `nested` once.
- **`--ssh-option`** adds one ssh argument (repeatable). With the jump hop it
  applies to the node (the gateway keeps its `Host` settings); with the
  nested hop, to the gateway. Write it as one word, `--ssh-option=-oName=value`.
- **Quote `~`** for paths on the cluster (`--dir '~/runs'`), or your own shell
  turns it into your local home (MACE rewrites that case, and says so).
  When the node really has that same absolute path under your local home
  (a shared file system mounted at the same place), start it with `//`:
  `--dir //home/you/runs` is sent as `/home/you/runs`, never rewritten. The
  connect scripts (`mace-gui-connect.sh`, `.ps1`) read `//` the same way.
  `~/.ssh/...` inside `--ssh-option=-oIdentityFile=~/.ssh/...` is fine: it is
  not at the start of a word, and ssh reads it as a local path.
- **`--dry-run`** prints the exact ssh command without connecting.

Test the pieces first, in the same terminal: `ssh cluster true` and
`ssh dev-node-01 true` should finish without asking for a password.
`ssh -G dev-node-01 | grep -i -e identityfile -e proxyjump` shows the key and
jump ssh uses for the node.

## 5. Check your setup

`mace gui connect --check` checks each piece above and says, for each
problem, the one command that fixes it. It is read-only: it changes no file,
and it reads the `.pub` file of a key to compare fingerprints with the agent.
It opens the private half only on Windows when the ssh-agent service is
disabled: `ssh-keygen -y -P ""` tells whether the key has a passphrase, and
only its exit code is kept.

```
mace gui connect --gateway cluster --node dev-node-01 --check
mace gui connect myprofile --check          # a saved profile
mace gui connect myprofile --check --no-net # no ssh logins, no DNS lookups
mace gui connect myprofile --fix-config     # print a suggested ~/.ssh/config block
```

What it looks at, in order:

1. **ssh**: that `ssh` is on `PATH`, and its version. On Windows, OpenSSH
   8.1 or newer (for `-J`) and Windows' own `C:\Windows\System32\OpenSSH`,
   not another `ssh` earlier on `PATH`.
2. **The gateway** (`ssh -G`): the user, host name and port ssh uses, which
   key files exist, `ProxyJump`, connection sharing (a warning on Windows,
   which does not support it), and a gateway that looks like an email
   address or does not resolve.
3. **The node**: which key files ssh's config gives the node, what
   `mace gui connect` adds for the jump hop, and whether a plain
   `ssh -J cluster dev-node-01` would ask for the node's password (the node
   gets none of the gateway's keys, and the agent does not hold them).
4. **The agent**: `ssh-add -l` (fingerprints only, no comments), and whether
   the gateway's key is loaded. On Windows, the ssh-agent service: when it
   is disabled, the fix is the two administrator commands from step 2.
5. **Permissions**: `~/.ssh` 700, the config and keys 600 (Windows: an
   `icacls` hint).
6. **Live probes** (skipped with `--no-net`): `ssh -o BatchMode=yes cluster
   true`, then the jump hop and the nested hop, each running only `true`. A
   gateway that asks for a password or a 2FA code cannot be probed this way;
   the check says so and stops there. On Linux and macOS, open a shared
   connection first (`ssh cluster true` with the `ControlMaster` lines of
   step 3), and the probes reuse it.
7. **The verdict**: which hop works, and the exact `mace gui connect` line.

Example, `--check --no-net` for a profile whose node asks for a password (the
key is named only in `Host cluster`, and no agent holds it):

```
Checking the ssh setup for myprofile: cluster -> dev-node-01 (read-only; nothing is changed)
OK    ssh: OpenSSH_9.6p1, OpenSSL 3.0.13 (/usr/bin/ssh)
OK    gateway cluster: user netid, HostName login.example.edu, port 22
OK    key files ssh offers cluster: ~/.ssh/id_ed25519_cluster (IdentitiesOnly yes)
INFO  node dev-node-01: user netid, key files none that exist
INFO  for the jump hop, the connection also gives dev-node-01: -o IdentityFile=~/.ssh/id_ed25519_cluster -o IdentitiesOnly=yes
WARN  dev-node-01 gets none of ~/.ssh/id_ed25519_cluster from your ssh config, and the agent does not hold them, so a plain ssh -J cluster dev-node-01 would ask for the password of dev-node-01
      fix: the connection adds them for you (above); for plain ssh too, add a Host block for the node (--fix-config prints one) or run ssh-add ~/.ssh/id_ed25519_cluster
WARN  cannot reach an ssh agent (SSH_AUTH_SOCK is not set in this terminal)
      fix: eval "$(ssh-agent)" && ssh-add ~/.ssh/id_ed25519_cluster
OK    permissions: ~/.ssh, its config and your key files are private
SKIP  live probes skipped: no ssh login was tried
Verdict (without the probes): no problem found. Connect with: mace gui connect myprofile
```

Without `--no-net`, the probes then show whether the node accepts the key
over the jump hop. When it does not (some clusters accept logins on their
nodes only from the login node), the lines end like this:

```
OK    ssh cluster true: logged in without a prompt
FAIL  jump hop: dev-node-01 refused your key from this computer (a connection would ask for the password of dev-node-01)
      fix: use the nested hop, or give dev-node-01 your key (docs/gui.md#mace-gui-connect-the-node-asks-for-a-password)
OK    nested hop: the gateway logs in to dev-node-01 by itself
Verdict: only the nested hop works. Connect with: mace gui connect myprofile --hop nested --save
```

`--fix-config` prints a `Host` block for the gateway and one for the node,
filled in from `ssh -G` (`ControlMaster` lines except on Windows,
`UseKeychain` only on macOS). It never writes `~/.ssh/config`: read it, then
paste what you want.

The standalone scripts check the same way, with the same messages:
`mace-gui-connect.sh --check [--no-net] [--fix-config]` and
`mace-gui-connect.ps1 -Check [-NoNet] [-FixConfig]`.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Not sure which piece is wrong | `mace gui connect --check` (step 5) tests each one and prints the fix. |
| "cannot reach host … (timeout)", "(DNS: …)" or "(connection refused)" | The name is wrong, or the network (or VPN) does not reach the cluster. `mace gui connect` gives up after 20 seconds (`ConnectTimeout`; a `ConnectTimeout` in your `~/.ssh/config` for that host, or an `--ssh-option=-oConnectTimeout=60`, wins). `ssh -v login.example.edu true` shows where it stops. |
| The node asks for a password (the prompt names `netid@dev-node-01`) | The node hop is not offered your key. Add the node pattern to the config (step 3), or load the key into the agent (step 2). See [gui.md](gui.md#mace-gui-connect-the-node-asks-for-a-password). |
| "Permission denied (publickey)" | The cluster does not know this key, or ssh offered another one. Check `authorized_keys` on the cluster (step 1), and that `IdentityFile` names the right file. `ssh -v cluster true` lists the keys it offers. |
| "ssh could not log in with your key" from `mace gui connect` | The key has a passphrase and no agent holds it; ssh has no terminal to ask on. Load it (step 2), then run `mace gui connect` again. |
| Windows: "Error connecting to agent" or "unable to start ssh-agent service" | The OpenSSH Authentication Agent service is disabled or stopped. `Get-Service ssh-agent` shows it; turn it on as in step 2 (administrator PowerShell, once). |
| Windows: `ssh-add -l` lists the key but `ssh` still asks | Another `ssh` comes first on `PATH` (Git for Windows, MobaXterm) with its own agent. `Get-Command ssh` should show `C:\Windows\System32\OpenSSH\ssh.exe`. |
| "Too many authentication failures" or the wrong key is used | ssh offered every key in the agent first. Add `IdentitiesOnly yes` with the right `IdentityFile` (step 3). |
| 2FA asks every time | Linux, macOS, WSL: connection sharing (step 3). Windows: not available; WSL's ssh or one long `mace gui connect` session are the ways around it. |
| The passphrase is asked every time | No agent, or it is empty: `ssh-add -l`. macOS: `UseKeychain yes` and `AddKeysToAgent yes`, and `ssh-add --apple-use-keychain` once. Linux: `AddKeysToAgent yes`. Windows: the agent service (step 2). |
| "UNPROTECTED PRIVATE KEY FILE" or "bad permissions" | Linux, macOS: `chmod 700 ~/.ssh` and `chmod 600 ~/.ssh/config ~/.ssh/id_ed25519_cluster`. Windows (PowerShell): `icacls $env:USERPROFILE\.ssh\id_ed25519_cluster /inheritance:r /grant:r "$($env:USERNAME):(R)"` leaves only you with access. |
| "Bad configuration option: usekeychain" | `UseKeychain` is macOS only. Put `IgnoreUnknown UseKeychain` at the top of the config, or remove the line on other systems. |
| `ssh -J` fails on Windows | `-J` needs OpenSSH 8.1 or newer (`ssh -V`); update the Windows OpenSSH client, or use `--hop nested`. |
