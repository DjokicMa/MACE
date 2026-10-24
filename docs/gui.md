# MACE GUI (`mace gui`)

The MACE GUI shows a folder of calculations in **your own web browser**: its
structures with their symmetry, optimizations step by step, band, DOS, IR,
Raman and FREQ plots, and `materials.db`. You can edit and build structures,
compare them, and set up the next calculation, saving the same JSON settings
file the CLI reads. A small server started by `mace gui` reads the folder; the
page draws everything in the browser, so nothing is installed on your laptop
and the 3D views use your laptop's GPU, even when MACE runs on a cluster node.

The MACE GUI is new in MACE 1.1.5; a few menu items are still to come and say
so ("Not in this version yet").

It writes into the served folder only what you save there (a structure, a
CIF, a settings JSON, Calculation Setup's files and the decks you ask it to
write) and, when you ask, a `materials.db` built from the folder's outputs. It
never writes over a calculation's outputs, and keeps the previous version of
every file it replaces; [What the GUI writes into your
folder](#what-the-gui-writes-into-your-folder) lists everything, and where.

**New here?** Read [Where MACE runs](#where-mace-runs), then find your row in
[Choose your setup](#choose-your-setup) and follow that one section.

## Contents

**Getting started**

- [Where MACE runs](#where-mace-runs) (the cluster, your computer, and why)
- [Choose your setup](#choose-your-setup): the table that sends you to your section
- On your computer only: [Run it locally](#run-it-locally)
- On the cluster, viewed on your computer:
  [Run it on a cluster and view it on your computer](#run-it-on-a-cluster-and-view-it-on-your-computer)
  - [First run from your computer](#first-run-from-your-computer) (what every cluster setup needs first)
  - [mace gui connect](#mace-gui-connect-linux-macos-wsl-windows) (MACE on your computer too)
  - [Only MACE on the cluster](#only-mace-on-the-cluster) (the standalone scripts):
    [Linux, macOS and Git Bash](#linux-macos-and-git-bash-mace-gui-connectsh),
    [Windows (PowerShell)](#windows-powershell), [MobaXterm](#mobaxterm)
  - [VS Code Remote-SSH](#vs-code-remote-ssh)
  - [Windows and VS Code, step by step](#windows-and-vs-code-step-by-step) (ssh not found, a page that keeps loading)
  - [Several people on one node](#several-people-on-one-node) (ports, sessions)
  - [Plain ssh tunnel (any OS)](#plain-ssh-tunnel-any-os)
  - [Common steps for every cluster setup](#common-steps-for-every-cluster-setup):
    [host keys and 2FA](#first-use-of-a-node-with---hop-jump),
    [jump hosts](#jump-hosts-and-nodes-that-accept-logins-only-from-the-login-node),
    [its own window](#mace-gui-in-its-own-window),
    [stopping the server](#sign-out-and-stop-the-server),
    [updating MACE](#updating-mace-from-the-gui)

**Using the GUI**

- [The window](#the-window)
- [Keys, the command search, settings and context menus](#keys-the-command-search-settings-and-context-menus)
- [What it shows](#what-it-shows)
- [Editing](#editing)
- [The classic editor panel](#the-classic-editor-panel)
- [Building structures](#building-structures)
- [Files](#files)
- [Calculation Setup](#calculation-setup)
- [Submit and Jobs](#submit-and-jobs)
- [Batch Setup](#batch-setup)
- [The Database document](#the-database-document)
  - [The Properties tab](#the-properties-tab)
- [Analyzing a run](#analyzing-a-run)
- [The Cluster view](#the-cluster-view)
- [Workflows](#workflows)
- [Plots: band, DOS, IR, Raman and FREQ modes](#plots-band-dos-ir-raman-and-freq-modes)
- [Volumes (cube files)](#volumes-cube-files)
- [Orbitals (ORBITALS .molden files)](#orbitals-orbitals-molden-files)
- [Molecules, slabs and polymers](#molecules-slabs-and-polymers)
- [Plot Gallery](#plot-gallery)
- [Compare](#compare)
- [Two editor groups side by side](#two-editor-groups-side-by-side)
- [Display range, boundary and rendering](#display-range-boundary-and-rendering)
- [Coordination and polyhedra](#coordination-and-polyhedra)
- [SVG export](#svg-export)

**Reference**

- [Long sessions and SLURM jobs](#long-sessions-and-slurm-jobs)
- [`mace gui` options and sign-in links](#mace-gui-options-and-sign-in-links)
- [Security](#security)
- [Dependencies](#dependencies)
- [Example folder (`examples/gui/`)](#example-folder-examplesgui) and [example suite (`examples/suite/`)](#example-suite-examplessuite)
- [Limitations](#limitations)
- [For developers](#for-developers)

**Troubleshooting**

- [Troubleshooting](#troubleshooting): [Reporting a problem](#reporting-a-problem), [mace gui connect: key passphrases and the gateway](#mace-gui-connect-key-passphrases-and-the-gateway), [mace gui connect: the node asks for a password](#mace-gui-connect-the-node-asks-for-a-password), [mace gui connect --check](#mace-gui-connect---check), [Messages](#messages)

## Where MACE runs

MACE has two places it can run, and they do different jobs.
[INSTALLATION.md, "Where to install"](../INSTALLATION.md#where-to-install)
explains them in full; in short:

- **On the cluster (required for full use).** CRYSTAL runs there, so SLURM
  submission, workflows, job tracking, the completion callbacks that start
  each next step, and recovery all need MACE installed on the cluster.
  `mace gui` running there can do all of that for you from the browser.
- **On your own computer, at minimum one of these**, to see the cluster's
  GUI: nothing but ssh and a browser (the [standalone connect
  script](#only-mace-on-the-cluster), no Python needed); [VS Code
  Remote-SSH](#vs-code-remote-ssh); a [plain ssh tunnel](#plain-ssh-tunnel-any-os)
  you open by hand; or a local MACE install, for [`mace gui
  connect`](#mace-gui-connect-linux-macos-wsl-windows) (saved profiles,
  `--check`, `mace update --remote`). Or skip the GUI: `ssh` to the cluster
  and use the `mace` command there.
- **A local install alone (no cluster)** is for getting things ready and
  looking at results: building structures, writing decks and settings,
  plotting and browsing outputs you copied from the cluster, and the
  database. It cannot submit jobs or run workflows, and it runs CRYSTAL only
  if CRYSTAL is installed on your computer too. You copy the inputs it
  writes to the cluster yourself.

## Choose your setup

**First, pick where MACE GUI runs.**

- **On the cluster, viewed on your computer (recommended).** The server runs
  on a cluster node, next to your data, and your browser (or VS Code) shows
  it through an ssh tunnel. Everything works: Submit, the Jobs panel, the
  Cluster view, workflows, recovery and database builds of the cluster's
  folders. Needs MACE on the cluster.
- **On your computer only.** The server and the browser are both on your
  computer. Use it to set things up (structures, Calculation Setup, decks)
  and to look at outputs you copied over (plots, Run Analysis, the
  database). It does not submit jobs: copy the decks to the cluster (scp,
  MobaXterm's file browser, VS Code's Explorer) and submit them there.

**Then find your row.** Each row links to one section with numbered steps
from zero.

| Where it runs | Your computer | How you connect | You view it in | Needs on your computer | Go to |
|---|---|---|---|---|---|
| Cluster | Linux, macOS, WSL | `mace gui connect` | your browser | a copy of MACE, Python 3, ssh | [mace gui connect](#mace-gui-connect-linux-macos-wsl-windows) |
| Cluster | Windows (PowerShell) | `mace gui connect` with Python | your browser | a copy of MACE, Python 3, Windows' ssh (8.1+) | [mace gui connect](#mace-gui-connect-linux-macos-wsl-windows), "On Windows" |
| Cluster | Linux, macOS, Git Bash on Windows | standalone `mace-gui-connect.sh` | your browser | ssh and bash only | [Linux, macOS and Git Bash](#linux-macos-and-git-bash-mace-gui-connectsh) |
| Cluster | Windows (PowerShell) | standalone `mace-gui-connect.ps1` | your browser | Windows' ssh (8.1+) only | [Windows (PowerShell)](#windows-powershell) |
| Cluster | Windows with MobaXterm | `mace-gui-connect.sh` in its local terminal, or its tunnel tool | your browser | MobaXterm | [MobaXterm](#mobaxterm) |
| Cluster | any | VS Code Remote-SSH | VS Code's built-in browser or your browser | VS Code with Remote-SSH | [VS Code Remote-SSH](#vs-code-remote-ssh) |
| Cluster | any | a plain `ssh -L` tunnel (PuTTY on Windows) | your browser | ssh | [Plain ssh tunnel](#plain-ssh-tunnel-any-os) |
| Your computer | Linux, macOS, WSL | none: `mace gui` | your browser (or VS Code's) | a full MACE install | [Run it locally](#run-it-locally) |

Not sure? On the cluster, viewed on your computer, is the one to pick. With
VS Code already set up for the cluster, its row has the fewest steps. With
nothing installed, the standalone script for your OS needs no Python. With
Python on your computer, `mace gui connect` saves a profile, so every later
run is one command.

**Words used below**

- **Gateway**: your cluster's login host, the one you `ssh` to first
  (`me@hpcc.example.edu` in the examples). It is shared by everyone, so the
  GUI does not run there.
- **Dev node**: a development node behind the gateway where you may run
  programs (`dev-node-01` in the examples). MACE GUI runs here.
- **Tunnel** (port forwarding): ssh carries a port on your computer
  (`127.0.0.1:8765`) to the server's port on the node. The server listens
  only on the node's own loopback (`127.0.0.1`, and `::1` for tools that say
  `localhost`), so the tunnel must end at that node.
- **Sign-in link**: the `http://127.0.0.1:.../?login=...` link `mace gui`
  prints. Each link works once; `mace gui login-url` prints another.
- **Profile**: the gateway, node, folder and setup lines `mace gui connect`
  saves in `connect.toml`, so you do not type them again.

## Run it locally

*On your computer only: set things up and look at copied outputs; no jobs.*

1. **Install MACE** on your computer with its packages:
   [INSTALLATION.md, "Local install"](../INSTALLATION.md#local-install-optional).
2. **Start it** in the folder you want to see:

   ```
   cd /path/to/your/calculations
   mace gui                       # or: mace gui --dir examples/gui --open
   ```

   It prints something like:

   ```
   MACE GUI serving /home/you/calcs
     http://127.0.0.1:8765/s/Xq3…/?login=k8F…
     (the link signs in once; mace gui login-url prints another)
   Token (for the sign-in form): 9vQ…
   Press Ctrl+C to stop.
   ```

3. **Open the link** (or start with `--open`, which opens MACE GUI in its
   own window; see [MACE GUI in its own window](#mace-gui-in-its-own-window)). In
   a VS Code terminal, Ctrl+click it; VS Code may open it in its built-in
   browser (see [VS Code Remote-SSH](#vs-code-remote-ssh) to change that).
4. **Every day:** the same `mace gui` command. The port stays the same each
   time (8765 on your own computer; on a shared node a port of your own,
   see [Several people on one node](#several-people-on-one-node)), so your
   browser drafts come back.
5. **Stop it:** File › Sign Out and Stop Server…, or Ctrl+C in the terminal.
   Closing the browser tab does not stop it.
6. **Send the work to the cluster:** copy the decks and settings files the
   GUI wrote to the cluster (scp, rsync, MobaXterm's file browser or VS
   Code's Explorer) and submit them there (`mace submit`).

Pitfalls:

- **No jobs here.** Submit and the Jobs and Cluster views need SLURM, which
  your computer does not have. Running CRYSTAL here works only when CRYSTAL
  and `mpirun` are installed on your computer (`mace run`; see
  [INSTALLATION.md](../INSTALLATION.md#where-to-install)).
- **A used link** shows the sign-in page: run `mace gui login-url` for a new
  one, or paste the token. All options, the sign-in link and the token:
  [`mace gui` options and sign-in links](#mace-gui-options-and-sign-in-links).

To try it on the example folder shipped with MACE:

```
mace gui --dir examples/gui --open
```

## Run it on a cluster and view it on your computer

*Recommended: MACE GUI runs on a cluster node and can do everything for you
(submit, track, recover, run workflows), while you use it from your own
computer.*

The server runs where the data is (a cluster's dev node); the browser runs on
your computer; an ssh tunnel joins them. Start with [First run from your
computer](#first-run-from-your-computer), then follow the one section your
row of [Choose your setup](#choose-your-setup) names. The manual tunnel, with
two terminals and no script, is [Plain ssh tunnel](#plain-ssh-tunnel-any-os).

### First run from your computer

Do these once, whatever way you connect.

1. **MACE on the cluster.** Install it there, on a dev node, and check that
   `mace --help` works in a fresh login:
   [INSTALLATION.md, "Cluster install"](../INSTALLATION.md#cluster-install-required-for-full-use).
   Note the line that makes `mace` work (for example `module load Python`)
   and where MACE is (for example `~/MACE`); the connect methods ask for both.
2. **A browser** with WebGL2 on your computer: a current Firefox, Chrome,
   Edge or Safari.
3. **An OpenSSH client** on your computer. It comes with macOS and Linux,
   and with Windows 10 and 11 as `ssh` in PowerShell (`ssh -V` shows its
   version; the jump through the gateway needs 8.1 or newer). MobaXterm and
   Git Bash bring their own.
4. **Log in by hand once:** `ssh me@hpcc.example.edu`, then from there
   `ssh dev-node-01`. This answers the gateway's host-key question and shows
   whether your cluster asks for 2FA (for example Duo).
5. **An ssh key (recommended):** [docs/ssh-setup.md](ssh-setup.md) sets up a
   key with a passphrase, an agent (Linux, macOS Keychain, the Windows
   OpenSSH agent, MobaXterm's MobAgent, VS Code) and an `~/.ssh/config` for
   the gateway and the dev nodes, so you type the passphrase (and, on Linux
   and macOS, 2FA) once. Without a key, ssh asks for your password, and with
   the jump hop it asks for the gateway's and the node's.

Then go to your section. The first connection through the gateway asks the
host-key questions again for the node: [First use of a node with --hop
jump](#first-use-of-a-node-with---hop-jump).

### mace gui connect (Linux, macOS, WSL, Windows)

*For: any OS, with a copy of MACE on your computer. One command starts the
server on the node, opens the tunnel and your browser, and stops it all
again. It saves profiles, checks your ssh setup (`--check`), and `mace update
--remote` updates the cluster's MACE from here.*

1. **Get MACE on your computer:** `git clone
   https://github.com/DjokicMa/MACE.git` (or download and unpack a release).
   `mace gui connect` needs only Python 3 and ssh, none of MACE's scientific
   packages, so no `pip install` is needed for it.
2. **Check your ssh setup** (it changes nothing):
   `mace gui connect --gateway me@hpcc.example.edu --node dev-node-01 --check`.
   Fix each `FAIL` line with the `fix:` line under it
   ([mace gui connect --check](#mace-gui-connect---check)).
3. **First connection:**
   `mace gui connect --gateway me@hpcc.example.edu --node dev-node-01 --dir ~/runs`.
   Run with no flags, it asks for each value instead. Add `--setup 'module
   load Python'` (the line that makes `mace` work on the node) and `--mace
   ~/MACE/mace_cli` when `mace` is not on the node's PATH. Answer the
   host-key questions and the 2FA prompts
   ([First use of a node](#first-use-of-a-node-with---hop-jump)). Your
   browser opens signed in.
4. **Save it:** add `--save` to that line (the first run with no flags saves
   by itself). From then on, **every day** is `mace gui connect` (or `mace
   gui connect <profile>`). Press Enter in its terminal for a new sign-in
   link.
5. **Stop it:** Ctrl+C in the terminal of `mace gui connect`, or File › Sign
   Out and Stop Server… in the browser. Closing the tab does not stop it.

Pitfalls: the node asks for a password ([the node asks for a
password](#mace-gui-connect-the-node-asks-for-a-password)); a passphrase or
an email address as the gateway ([key passphrases and the
gateway](#mace-gui-connect-key-passphrases-and-the-gateway)); a host that
does not answer gives up after 20 seconds ([other
messages](#mace-gui-connect---check)); a node that accepts logins only from
the login node ([Jump hosts](#jump-hosts-and-nodes-that-accept-logins-only-from-the-login-node)).

**On Windows** (PowerShell, Windows' own `ssh`; OpenSSH 8.1 or newer for
`-J`, see `ssh -V`) run it with the Python you have, from a copy of MACE's
folder: `python <MACE>\mace_cli gui connect --gateway ...`. Anaconda and
Miniconda install no `py` launcher, so use `python` there (from the Anaconda
Prompt, or with Anaconda's `python.exe` on PATH); with the python.org
installer `py <MACE>\mace_cli gui connect ...` works too. The flags, the
profile and the messages are the same as below.

- **The first run** with no profile asks for the gateway, a jump host (blank
  for none), the node (blank runs on the gateway itself, which is shared: a
  dev node is better), the folder, setup lines (e.g. `module load Python`),
  the `mace` command on the node and a profile name, and saves them in
  `~/.config/mace/connect.toml` (Windows: `%APPDATA%\mace\connect.toml`), with
  no passwords. On Linux and macOS only you can read it (mode 0600); on
  Windows it has the access rights of your AppData folder, which only you
  (and the computer's administrators) can read. Later runs need only `mace
  gui connect` (the default profile) or `mace gui connect <name>`.
- **Flags override the profile** for one run: `--gateway`, `--jump`,
  `--node`, `--hop`, `--dir`, `--mace`, `--setup`, `--conda`,
  `--local-port`, `--ssh-option` (repeatable), `--root` (repeatable),
  `--no-open`. `--browser app|tab` picks MACE GUI's own window or a browser
  tab for one run ([its own window](#mace-gui-in-its-own-window); not saved
  in the profile). `--save [NAME]` writes them back; `--list` and
  `--forget NAME` manage the profiles; `--dry-run` prints the exact ssh
  command and exits (see [dry-run quoting](#dry-run-quoting) for which
  shell each printed form is for).
- **Paths on the node under `~`**: your shell expands an unquoted `~`
  before `mace gui connect` sees it, so `--mace ~/mace/mace_cli` arrives as
  `/home/you/mace/mace_cli`, a path on *this* computer. When the node's home
  is elsewhere (`/mnt/home/netid`), that path is not there. So a `--mace`,
  `--dir`, `--root` or `--conda` value (or a saved profile value) under this
  computer's home folder is sent as `~/...`, which the node expands to its
  own home, and a note says so: "Note: treated /home/you/mace/mace_cli as
  ~/mace/mace_cli on the remote ...". Quote the `~` (`--mace '~/mace/mace_cli'`)
  to pass it as is. `--setup` is shell code you quote, so it is left alone.
  The standalone scripts do the same.
- **The ~/.ssh/config way** works too: a `Host hpcc` block for the gateway
  (with `User`), and `--gateway hpcc`. `mace gui connect` asks `ssh -G` which
  user that alias logs in as, and uses the same user on the node.
- **Ctrl+C** stops the server on the node and its cache folder. When GUI
  tasks are still running (a local CRYSTAL run, a database build, a Clean),
  it asks first: "2 tasks are running on dev-node-01 (...). Stop them too?
  [y/N]". No (the default) leaves the server running only for them.
- **A dropped connection** (a closed lid, a network change) stops the server
  when no task runs. With tasks running it keeps them running and stops after
  the last one ends (plus 10 minutes with nobody reconnected; 24 hours at
  most). Run `mace gui connect` again with the same folder: it says
  "Reattached to the server still running on dev-node-01 (1 task running)."
- **Enter** in the terminal of `mace gui connect` opens a new sign-in link
  (each link works once).
- **Your local port** is `--local-port` (default 8765), so your browser's
  drafts come back on the next run; when it is busy, the next free one is used
  and it says so.
- **On Windows, closing or killing** `mace gui connect` also ends its `ssh`
  (MACE puts ssh in a Windows job object that ends with it), so no stray
  `ssh.exe` keeps the tunnel open. The PowerShell script does not do that:
  if its PowerShell is killed (Task Manager) instead of stopped with Ctrl+C,
  a leftover `ssh.exe` may keep the tunnel open; end it in Task Manager.

#### Dry-run quoting

`mace gui connect --dry-run` prints each ssh command as one line in POSIX
shell quoting (with `'"'"'` for a quote inside quotes): paste it into bash,
zsh, Git Bash, MobaXterm's terminal or WSL. On Windows it also prints a
second line, `ssh --% ...`, to paste into PowerShell (5.1 or 7): `--%` makes
PowerShell hand the rest of the line to `ssh.exe` unchanged, quoted the way
`ssh.exe` reads its command line. Neither line is for `cmd.exe`, whose
quoting differs. The scripts' `--dry-run` (`-DryRun`) print one argument per
line, and `mace-gui-connect.ps1 -DryRun` adds the same `ssh --% ...` line.

### Only MACE on the cluster

You do not need MACE on your own computer: the standalone scripts
`mace-gui-connect.sh` (bash: Linux, macOS, Git Bash, MobaXterm's local
terminal) and `mace-gui-connect.ps1` (Windows PowerShell) need only ssh, and
no Python. They take the same flags as `mace gui connect` and read the same
`connect.toml` profiles, but they cannot save a profile (`--save` needs
`mace gui connect`), and they have no `--slurm`. So either type the flags each
time (keep the line in a small text file), or make a copy with your settings
filled in, on the cluster:

```
mace gui connect --gateway me@hpcc.example.edu --node dev-node-01 --dir ~/runs --export-script bash -o mace-gui-connect.sh
```

(`--export-script powershell -o mace-gui-connect.ps1` for Windows), and copy
that file to your computer. `mace gui` on a node prints this line for you,
with the node and folder filled in.

Or download them from a release tag (not `main`), with the tag's checksums,
and check them before running. Each release is tagged `vX.Y.Z-preview` first
and `vX.Y.Z` when it is final; set `TAG` to one that exists on the
[tags page](https://github.com/DjokicMa/MACE/tags).

<!-- RELEASE-TAG: the TAG= and $Tag = lines below name the release; set both at release time -->
```
TAG=v1.1.6
curl -fsSLO https://raw.githubusercontent.com/DjokicMa/MACE/$TAG/scripts/mace-gui-connect.sh
curl -fsSLO https://raw.githubusercontent.com/DjokicMa/MACE/$TAG/scripts/SHA256SUMS
sha256sum -c --ignore-missing SHA256SUMS     # macOS: shasum -a 256 -c --ignore-missing SHA256SUMS
chmod +x mace-gui-connect.sh
./mace-gui-connect.sh --gateway me@hpcc.example.edu --node dev-node-01 --update-remote release
```

Windows (PowerShell; `curl.exe` comes with Windows 10 and 11):

```
$Tag = "v1.1.6"
curl.exe -fsSLO "https://raw.githubusercontent.com/DjokicMa/MACE/$Tag/scripts/mace-gui-connect.ps1"
curl.exe -fsSLO "https://raw.githubusercontent.com/DjokicMa/MACE/$Tag/scripts/SHA256SUMS"
(Get-FileHash .\mace-gui-connect.ps1).Hash.ToLower()   # must equal its line in SHA256SUMS
Unblock-File .\mace-gui-connect.ps1
powershell -ExecutionPolicy Bypass -File .\mace-gui-connect.ps1 -Gateway me@hpcc.example.edu -Node dev-node-01 -UpdateRemote release
```

`--update-remote [release|preview|latest]` (`-UpdateRemote`) runs
`mace update` in the cluster's MACE checkout before the GUI starts (it shows
what changes and asks, in the same window); on connecting the scripts print
the cluster's MACE version. Updating needs MACE 1.1.6 or newer there; update
an older one once by hand (`git -C <MACE folder> pull`). With MACE on your
computer, `mace update --remote` does the same:
[docs/update.md](update.md).

### Linux, macOS and Git Bash (mace-gui-connect.sh)

*For: Linux, macOS, or Windows with Git Bash, with nothing installed but ssh
and bash (macOS's bash 3.2 is enough). In MobaXterm, see
[MobaXterm](#mobaxterm).*

1. **Get the script** and check it ([Only MACE on the
   cluster](#only-mace-on-the-cluster)), then `chmod +x mace-gui-connect.sh`.
2. **Check your ssh setup** (read-only):
   `./mace-gui-connect.sh --gateway me@hpcc.example.edu --node dev-node-01 --check`.
3. **First connection:**
   `./mace-gui-connect.sh --gateway me@hpcc.example.edu --node dev-node-01 --dir ~/runs`,
   plus `--setup 'module load Python'` and `--mace ~/MACE/mace_cli` when
   `mace` is not on the node's PATH. Answer the host-key questions and the
   2FA prompts ([First use of a node](#first-use-of-a-node-with---hop-jump)).
   Your browser opens signed in.
4. **Every day:** the same line (or `./mace-gui-connect.sh` alone, with a
   copy made by `--export-script`, or a profile `mace gui connect --save`
   wrote). `--update-remote release` updates the cluster's MACE first.
5. **Stop it:** Ctrl+C in that terminal, or File › Sign Out and Stop
   Server… in the browser. Closing the tab does not stop it.

Pitfalls: the same as for `mace gui connect` ([Troubleshooting](#troubleshooting));
in Git Bash, the script reads `%APPDATA%\mace\connect.toml` (the profile
Windows' `mace gui connect --save` writes) when its own home has none, and
says which file it read.

### Windows (PowerShell)

*For: Windows 10 or 11 with nothing installed (Windows PowerShell 5.1, which
comes with Windows, is enough; PowerShell 7 works too).*

1. **Check ssh:** `ssh -V` in PowerShell must say OpenSSH 8.1 or newer (the
   script warns when it is older). If PowerShell does not know `ssh`, the
   script looks for it in `System32\OpenSSH`, `Program Files\OpenSSH`, Git
   for Windows and VS Code's `remote.SSH.path` itself and says which it uses;
   with none it prints the install command ([Windows and VS Code, step by
   step](#windows-and-vs-code-step-by-step)). No Python is needed.
2. **Get the script** and check it: the PowerShell lines in [Only MACE on
   the cluster](#only-mace-on-the-cluster), or a copy with your settings
   from `--export-script powershell` on the cluster. Run `Unblock-File
   .\mace-gui-connect.ps1` once.
3. **Check your ssh setup** (read-only):
   `powershell -ExecutionPolicy Bypass -File .\mace-gui-connect.ps1 -Gateway me@hpcc.example.edu -Node dev-node-01 -Check`.
4. **First connection:** the line below. Answer the host-key questions and
   the 2FA prompts ([First use of a node](#first-use-of-a-node-with---hop-jump)).
   Windows' ssh has no connection sharing, so 2FA asks on each connection.
5. **Every day:** the same line. `-UpdateRemote release` updates the
   cluster's MACE first.
6. **Stop it:** Ctrl+C in that PowerShell window, or File › Sign Out and
   Stop Server… in the browser. Closing the tab does not stop it.

Windows 10 1809 and later ship `ssh`; `-J` (the jump through the gateway)
needs OpenSSH 8.1 or newer: check with `ssh -V` (the script warns when it is
older). Get the script on your computer in one of two ways: run
`mace gui connect --export-script powershell -o mace-gui-connect.ps1` on the
cluster and copy the file (your profile is filled in), or take
`scripts/mace-gui-connect.ps1` from MACE's repository. Downloaded files are
blocked by Windows: run `Unblock-File .\mace-gui-connect.ps1` once. Then:

```
powershell -ExecutionPolicy Bypass -File .\mace-gui-connect.ps1 -Gateway me@hpcc.example.edu -Node dev-node-01 -Dir ~/runs
```

The parameters match the flags of `mace gui connect` (`-Jump`, `-Hop`,
`-Mace`, `-Setup`, `-Conda`, `-LocalPort`, `-SshOption`, `-Root`, `-NoOpen`,
`-Browser app|tab`, `-DryRun`), and it reads the same `connect.toml`. Two differences: it has
no `--slurm`, and `-Hop auto` (the default of a saved profile) works as
`jump`: it does not retry with `nested` by itself when the node refuses the
login (it says to pass `-Hop nested`). For either, run `mace gui connect`
itself (`python <MACE>\mace_cli gui connect`, above). If no password or 2FA
prompt appears and the connection ends, set up an ssh key or agent, or use
`-Manual`, which prints the two-terminal commands of the manual tunnel.

`-SshOption` takes several words (`-SshOption "-o","SetEnv=A=B"`). From a
PowerShell session call the script with `&` (`& .\mace-gui-connect.ps1 ...`).
With `powershell -File`, Windows PowerShell 5.1 hands the list over as one
string (`-o,SetEnv=A=B`); the script splits it back, keeping commas inside a
value (`HostKeyAlgorithms=a,b`). Options you always want belong in the
profile's `ssh_options` (`connect.toml`).

`-Check` exits 0 when a way to connect works, 1 when a FAIL line stands, and 2
when there is no ssh client at all (the same exit code as connecting without
one; `mace gui connect --check` and the bash script agree). A disabled
ssh-agent service is a warning when your key has no passphrase or the live
probes log in; it is a failure only when the probes cannot log in without an
agent.

The script behaves the same in Windows PowerShell 5.1 and in PowerShell 7:
it builds ssh's command line itself (not through PowerShell's native-argument
passing, which differs between the two for empty strings and quotes), writes
no files, and uses no PowerShell 7 syntax. `-Check` says which ssh it found.

The `Unblock-File` and `-ExecutionPolicy Bypass` steps are for Windows'
default execution policy (Restricted or RemoteSigned), and have not been
tested on such a computer yet. `Get-ExecutionPolicy` shows yours: where it is
already Bypass or Unrestricted, neither step is needed. Where your
organization sets the policy, `-ExecutionPolicy Bypass` may be refused; run
`mace gui connect` with Python then. With
PuTTY instead, set the tunnel under Connection › SSH › Tunnels (source port
26118, destination `127.0.0.1:26118`, where 26118 stands for the port `mace
gui` printed on the node) and connect to the node through the gateway.

### MobaXterm

*For: Windows with MobaXterm, and nothing else installed. MobaXterm keeps
its own home folder and its own `~/.ssh`, separate from Windows' `ssh`,
PowerShell and VS Code.*

1. **Get `mace-gui-connect.sh`** ([Only MACE on the
   cluster](#only-mace-on-the-cluster)) into MobaXterm's home folder
   (drive C: is `/drives/c` there).
2. **Keys:** set MobaXterm up with a key of its own, or point it at your
   Windows key (the **Keys** item below), or use MobAgent
   ([docs/ssh-setup.md](ssh-setup.md#2-load-the-key-into-an-agent)).
3. **First connection,** in MobaXterm's **local terminal** (not a session on
   the cluster):
   `bash mace-gui-connect.sh --gateway me@hpcc.example.edu --node dev-node-01 --dir ~/runs`.
   ssh asks the host-key questions here even when Windows' `ssh` already
   trusts both hosts. Your browser opens signed in.
4. **Every day:** the same line; `--update-remote release` updates the
   cluster's MACE first.
5. **Stop it:** Ctrl+C in the local terminal, or File › Sign Out and Stop
   Server… in the browser. Closing the tab does not stop it.

The details, and the two other ways (a MobaXterm session plus an ssh line,
or the MobaSSHTunnel tool):

(a) Open MobaXterm's **local terminal** and run `mace-gui-connect.sh` (bash
comes with it): `bash mace-gui-connect.sh --gateway me@hpcc.example.edu --node
dev-node-01 --dir ~/runs`. Get the script with `mace gui connect
--export-script bash -o mace-gui-connect.sh` on the cluster, or from MACE's
`scripts/` folder.

- **Profiles:** the script reads `~/.config/mace/connect.toml` in
  MobaXterm's own home folder (its "persistent home directory", set under
  Settings › Configuration › General). When that file is missing, it reads
  `%APPDATA%\mace\connect.toml`, the file Windows' `mace gui connect --save`
  writes, and says which file it read. Git Bash and Cygwin do the same.
- **Keys:** MobaXterm keeps its own `~/.ssh` (keys, `config`,
  `known_hosts`) in that home folder, separate from `C:\Users\<you>\.ssh`,
  which Windows' `ssh`, PowerShell and VS Code use. A key, a Host alias or a
  trusted host key in one is not seen by the other. So, the first time from
  MobaXterm:
  - ssh asks the **host-key questions again**, for the gateway and (with the
    jump hop) for the node, even when Windows' `ssh` already trusts both
    (see [First use of a node with --hop jump](#first-use-of-a-node-with---hop-jump));
  - with the **jump hop**, the node checks a key from MobaXterm's `~/.ssh`
    (or its agent): without one there, ssh asks for the **node's password**
    (and the gateway's), even if Windows' `ssh` logs in with a key.

  To use your Windows key in MobaXterm, either set MobaXterm's persistent
  home directory (Settings › Configuration › General; MobaXterm's
  documentation: "Choose a folder that MobaXterm will use as its HOME
  directory ... so you can store your SSH keys in it") to your Windows user
  folder, `C:\Users\<you>`, so that its `~/.ssh` is that `.ssh`; or copy
  the key into MobaXterm's `~/.ssh` (MobaXterm shows drive C: as
  `/drives/c`) and make it private, since ssh refuses a private key that
  others can read:

  ```
  mkdir -p ~/.ssh
  cp /drives/c/Users/YOU/.ssh/id_ed25519 /drives/c/Users/YOU/.ssh/id_ed25519.pub ~/.ssh/
  chmod 600 ~/.ssh/id_ed25519
  ```

  (replace YOU with your Windows user name). Or name the key for one run:
  `--ssh-option=-oIdentityFile=/drives/c/Users/<you>/.ssh/id_ed25519`.
- **"stty: standard input: Inappropriate ioctl for device"** before the
  link: a login profile on the node ran `stty` with no terminal (every hop
  runs `bash -l` without one). It is harmless; `mace gui connect` and
  `mace-gui-connect.sh` no longer show it before the server is ready, and
  `mace-gui-connect.ps1` (which leaves ssh's messages on the console) may.
- **"channel 2: open failed: connect failed: Connection refused"** before
  the link (mostly with the jump hop): the tunnel's local port listens as
  soon as ssh connects, and something connected to it before the server on
  the node was listening, typically a browser tab left open from an earlier
  session, which keeps retrying. Harmless; `mace gui connect` and
  `mace-gui-connect.sh` hide it until the tunnel answers, and show it when
  the tunnel never does. `mace-gui-connect.ps1` may show it.

(b) Or, in a normal MobaXterm session on `dev-node-01`, run `mace gui
--dir ~/runs` and paste the `ssh -N -J ... -L ...` line it prints into
MobaXterm's **local terminal** (leave it running), then open the link it
printed. Use that line as printed: it ends at `127.0.0.1` on the node.

**MobaSSHTunnel** (Tools › MobaSSHTunnel) needs **two** tunnels for this,
because `mace gui` listens only on the node's own `127.0.0.1`: a single
tunnel with SSH server the gateway and remote server `dev-node-01` asks the
gateway to connect to `dev-node-01:<port>`, which the server never sees (the
browser then keeps loading and nothing appears). Chain them, as MobaXterm's
own guide does for a host behind a jump host
([Mobatek: SSH tunnels and port forwarding](https://blog.mobatek.net/post/ssh-tunnels-and-port-forwarding/)):

1. A local tunnel with forwarded port `2222`, SSH server the gateway (your
   user, port 22), remote server `dev-node-01`, remote port `22`.
2. A second local tunnel with forwarded port `<port>`, SSH server
   `localhost` port `2222` (your user: this logs in to `dev-node-01` through
   the first tunnel), remote server `127.0.0.1`, remote port `<port>`.

Start both, run `mace gui --dir ~/runs --port <port>` on `dev-node-01`, and
open the link it prints. A dev node is shared; since 1.1.6 `mace gui` on a
node uses a port of your own by default (see [Several people on one
node](#several-people-on-one-node)): use the number it prints as
`<port>`. To choose it yourself, any port from 8000 to 60999 works (`mace
gui` says "cannot listen" when it is taken; then pick another); keep it in
`~/.bashrc` on the cluster as `export MACE_GUI_PORT=<port>`, so it stays the
same and your browser drafts come back. `mace gui connect` and the scripts
avoid this by choosing a random free port on the node themselves.

**A browser that keeps loading** (no page, no error) means the tunnel does
not reach the server: it ends on the wrong node, or at the node's name
instead of its `127.0.0.1`. Check it against the line `mace gui` prints.

### VS Code Remote-SSH

*For: any OS with VS Code and its Remote-SSH extension. VS Code forwards the
port itself, so you need no script; the GUI opens in VS Code's built-in
browser or in your own browser.*

1. **Install** VS Code and the Remote-SSH extension, and add the ssh config
   below.
2. **Connect:** "Remote-SSH: Connect to Host..." `<dev-node>`. The first
   connection asks the host-key question in VS Code's window; for a 2FA
   prompt turn on `remote.SSH.showLoginTerminal`
   ([docs/ssh-setup.md](ssh-setup.md#2-load-the-key-into-an-agent)).
3. **Start it** in VS Code's integrated terminal (on the node):
   `mace gui --dir ~/runs`.
4. **Open it:** Ctrl+click the sign-in link it prints. It opens in VS Code's
   built-in browser or in your own browser (the setting below picks).
5. **Every day:** connect, run `mace gui`, Ctrl+click.
6. **Stop it:** File › Sign Out and Stop Server… in MACE GUI, or Ctrl+C in
   the terminal that runs `mace gui`. Closing the tab does not stop it.

Add the gateway and the node to your ssh config (Windows:
`C:\Users\<you>\.ssh\config`; Linux and macOS: `~/.ssh/config`):

```
Host hpcc
    HostName hpcc.example.edu
    User <user>
Host <dev-node>
    ProxyJump hpcc
    User <user>
```

where `<dev-node>` is the dev node you use (for example `dev-node-01`). On
Windows, VS Code uses Windows' own `ssh` (its `ProxyJump` needs OpenSSH 8.1
or newer: `ssh -V`) and that `C:\Users\<you>\.ssh` folder, not MobaXterm's;
the first connection to the node asks the host-key question in VS Code's
window. Then
"Remote-SSH: Connect to Host..." `<dev-node>`, and in VS Code's
integrated terminal run `mace gui --dir ~/runs`. **Ctrl+click the sign-in link
it prints**: VS Code forwards the port and rewrites the link (a different
local port is fine). The Ports tab's "Open in Browser" opens the address
without the link, which shows the sign-in page; use the link from the
terminal (or `mace gui login-url`). `mace gui --open` uses VS Code's own
`$BROWSER`, so it opens on your computer.

- **Ctrl+click may open VS Code's integrated browser** (an editor tab)
  instead of your own browser: recent VS Code versions open `localhost` and
  `127.0.0.1` links there by default. To open them in your default browser,
  turn off the setting `workbench.browser.openLocalhostLinks` (Settings,
  search "open localhost links"); `workbench.externalBrowser` picks another
  browser than the system default. A link the integrated browser opened is
  used up (each works once): run `mace gui login-url` in the terminal for a
  new one, then copy it into your browser or Ctrl+click it.
- **Closing the browser tab does not stop the server.** It keeps running on
  the node until you choose File › Sign Out and Stop Server… in MACE GUI
  ([Sign out and stop the server](#sign-out-and-stop-the-server)) or press
  Ctrl+C in the VS Code terminal that runs `mace gui`.

### Plain ssh tunnel (any OS)

*For: any OS with an ssh client and nothing else. On Windows, PowerShell's
`ssh` works the same way; with PuTTY, see the end of [Windows
(PowerShell)](#windows-powershell).*

The manual way, with two terminals: the server on the node, the tunnel on
your computer.

1. Log in to a **dev node** (or to a compute node in an `salloc`), load your
   cluster's Python 3.11 or newer (e.g. `module load Python`) and start the
   GUI:

   ```
   ssh hpcc
   ssh dev-node-01
   module load Python
   cd /mnt/scratch/$USER/my_calcs
   mace gui
   ```

   In an ssh session it names the exact node and prints the tunnel lines:

   ```
   This server runs on dev-node-01. The tunnel must go to this exact node.
   From your computer, with MACE installed there:
     mace gui connect --gateway you@LOGIN_HOST --node dev-node-01 --dir /mnt/scratch/you/my_calcs
   Without MACE there (MobaXterm's local terminal, Git Bash, PowerShell): make the standalone script here, copy
   it to your computer and run it there (...):
     mace gui connect --gateway you@LOGIN_HOST --node dev-node-01 --dir /mnt/scratch/you/my_calcs --export-script bash -o mace-gui-connect.sh
   Or with ssh alone (leave it running), then open the URL above:
     ssh -N -J you@LOGIN_HOST -L 26118:127.0.0.1:26118 you@dev-node-01
   ...
   (replace LOGIN_HOST with your cluster's login host; ...)
   ```

2. On **your computer**, in a second terminal, run that `ssh -N -J ... -L ...`
   line (it prints nothing and keeps running; that is expected), then open the
   `http://127.0.0.1:26118/s/.../?login=...` link in your browser. 26118 here
   stands for the port `mace gui` printed: on a shared node each user gets a
   port of their own (see "Several people on one node" below). The tunnel
   must go to the node `mace gui` names: another dev node behind the same
   gateway does not see the server.

If you have an `~/.ssh/config` entry for the gateway (for example `Host hpcc`),
`ssh -N -J hpcc -L 26118:127.0.0.1:26118 dev-node-01` does the same thing. If port
26118 is already in use on your computer, change the first number and use that
port in the link. The hint prints `LOGIN_HOST` (a word you can paste and then
replace; `<...>` would be a redirect in bash) unless it knows your cluster's
login host: from `--gateway`, `export MACE_GUI_GATEWAY=login.example.edu` (in
`~/.bashrc`), or `gateway = "login.example.edu"` in the `[gui]` table of
`site.toml` (where a cluster's administrators can set it for everyone, in
`/etc/mace/site.toml`). The script line is for a computer without MACE:
`--export-script` writes the standalone script with these settings filled in;
copy it over and run `bash mace-gui-connect.sh` (or use `--export-script
powershell -o mace-gui-connect.ps1` and run `.\mace-gui-connect.ps1`). On a shared node pick a personal port so you do not collide
with other users and your drafts stay on one port: `export MACE_GUI_PORT=8xxx`
in `~/.bashrc`.

**Stop it:** File › Sign Out and Stop Server… in the browser, or Ctrl+C in
the terminal that runs `mace gui`; then Ctrl+C the tunnel. Closing the tab
does not stop the server.

### Several people on one node

A development node is shared: you and the others in your group can all run
`mace gui` on it at the same time without a port collision.

- **Your own port.** On your own computer `mace gui` uses 8765. In an ssh
  session or a batch job (a shared node) it uses `20000 + (your user id mod
  10000)`, for example 26118 for user id 846118: the same number on every
  node and every day, so your drafts and your `ssh -L` line stay the same, and
  two people get different numbers. If another program already holds yours
  (a user id that ends the same way), it takes the next free port and says
  so. A port you name with `--port` or `MACE_GUI_PORT` is used exactly as
  given. `MACE_GUI_SHARED=1` (or `0`) overrides the shared-node guess.
- **The exact forwarding line.** `mace gui` prints it for the port it chose,
  for example `ssh -N -J you@LOGIN_HOST -L 26118:127.0.0.1:26118 you@dev-node-01`
  (`LOGIN_HOST` is your cluster's login host; set `--gateway`,
  `MACE_GUI_GATEWAY` or `[gui] gateway` in site.toml to fill it in). Windows,
  macOS and Linux use the same line.
- **Already running?** Every server records itself (host, port, folder, never
  the token) in `~/.cache/mace/gui/sessions/` while it runs. Starting `mace
  gui` for a folder you already serve on this node does not start a second
  server: it says so, prints a new sign-in link and the forwarding line, and
  exits. `mace gui --new-session` starts another anyway. For a different
  folder it starts a second server on the next port and lists the others.
  `mace gui --sessions` lists all of yours, also those on other nodes (a
  server on another node shows while it is alive; ask for its link there).
- **It stops with its session.** Started over plain `ssh node "mace gui"`
  (no terminal, so ssh sends no hangup), a server used to keep running after
  you closed the connection and held its port and memory on the shared node.
  Now it stops when that connection ends (its standard input closes), and its
  registry record goes with it. A terminal session stops it as before (Ctrl+C
  or hangup). To keep a server running after you log out, start it with
  `nohup mace gui --dir ~/runs > mace-gui.log 2>&1 &`, or add `--keep`
  (also `MACE_GUI_KEEP=1`); `mace gui --slurm` is the batch job version.
  Servers started by `mace gui connect` and `--slurm` follow their own rules.
- **`mace gui connect` and `--slurm`** pick a random free port on the node
  themselves, so they never collide either.

### Windows and VS Code, step by step

*For: a Windows 10 or 11 computer with VS Code and nothing else installed
(no Python, no Git). Everything below works in Windows PowerShell 5.1 and in
PowerShell 7.*

1. **ssh.** Open PowerShell and run `ssh -V`. It must print `OpenSSH_for_Windows_8.1`
   or newer. If PowerShell says *ssh is not recognized*, the PATH of that
   window lost Windows' OpenSSH folder. Check `Test-Path
   "$env:SystemRoot\System32\OpenSSH\ssh.exe"`: when it prints `True`, run
   ssh by its full path (`& "$env:SystemRoot\System32\OpenSSH\ssh.exe" -V`) and
   add `C:\Windows\System32\OpenSSH` to your user PATH (Settings › System ›
   About › Advanced system settings › Environment Variables › Path › Edit).
   When it prints `False`, install the client in an administrator PowerShell:
   `Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0`, then open a new
   PowerShell. `mace gui connect` and `mace-gui-connect.ps1` look in
   `System32\OpenSSH`, `Program Files\OpenSSH`, Git for Windows (`usr\bin`) and
   VS Code's `remote.SSH.path` setting themselves, put the folder they find
   first in PATH for their own run and tell you what they used. A Git for
   Windows ssh has its own agent (`ssh-add` in Git Bash): Windows' own client
   is the supported one.
2. **Your key and the config.** Follow [docs/ssh-setup.md](ssh-setup.md) once,
   then add the `Host` blocks of [VS Code Remote-SSH](#vs-code-remote-ssh) to
   `C:\Users\<you>\.ssh\config`. Test: `ssh -o BatchMode=yes hpcc hostname`
   prints the gateway's name without asking anything.
3. **Connect VS Code** to the dev node (F1 › Remote-SSH: Connect to Host…) and
   open a terminal there (Ctrl+`).
4. **Start MACE GUI** in that terminal: `mace gui --dir ~/runs`. It prints the
   sign-in link for your own port. VS Code forwards the port by itself.
5. **Open it:** Ctrl+click the link. If it opens in VS Code's own browser tab
   and you prefer your browser, turn off `workbench.browser.openLocalhostLinks`.
   Each link works once; `mace gui login-url` prints another.
6. **Without Python or MACE on your computer** (the cluster has MACE, your
   computer only has PowerShell): use the standalone script instead of
   steps 3 to 5, see [Windows (PowerShell)](#windows-powershell). Windows'
   `python` can be the Microsoft Store stub that prints *Python was not
   found*: that is not a Python, and the script does not use one.

**The page keeps loading although the port is forwarded.** Check in this
order:

1. *Is the tunnel reaching the server?* In PowerShell on your computer run
   `curl.exe -s -o NUL -w "%{http_code}" --max-time 10 http://127.0.0.1:<port>/`
   (the port of your link). **403 or 404** within a second means the tunnel
   and the server are fine (the sign-in page answers; the problem is the
   link or the browser). **000** means the tunnel ends nowhere.
2. *The server must listen on both loopback addresses.* Before 1.1.6 it
   listened on `127.0.0.1` only. VS Code's port forwarding runs on Node and
   reaches the node through the name `localhost`, which most nodes resolve to
   `::1` first: VS Code accepted your browser's connection and had nothing to
   connect to, so the page loaded forever. MACE 1.1.6 also listens on `::1`
   (still only on the node itself; no other computer can reach it). On the
   node, `ss -ltn | grep <port>` shows both `127.0.0.1` and `[::1]` when MACE
   is current; update the cluster's MACE (`mace update`) if only the first
   is there. A tunnel of your own must end at `127.0.0.1` on the node,
   never at the node's name (`mace gui` prints the right line).
3. *The port is already taken on your computer.* `netstat -ano | findstr :<port>`
   shows who holds it (an old `ssh -L`, an earlier VS Code forward). Close
   that one, or forward another port (change the first number of `-L`).
4. *Use the link from the terminal,* not the Ports tab's *Open in Browser*:
   that opens the address without the sign-in link and shows the sign-in
   page, which looks like nothing happened when the page is blank.
5. *Do not bind MACE to `0.0.0.0` to "fix" it.* That makes the token and your
   data readable by everyone on the node's network. The ssh tunnel is the
   supported path; `--insecure-network` exists only for a network you trust.

## Common steps for every cluster setup

### First use of a node with --hop jump

With `--hop jump` (and the first try of `--hop auto`) your computer reaches
the node through the gateway with `ssh -J`, and checks **two** host keys
itself, against your own `known_hosts` (Windows:
`C:\Users\<you>\.ssh\known_hosts`): the **gateway's** first, then the
**node's**. The gateway leg runs as a separate ssh process that does not get
the command line's `-o` options, so it always uses your real `known_hosts`.
The first time, ssh asks in the terminal, once per host it has not seen:

```
The authenticity of host 'dev-node-01 (<no hostip for proxy command>)' can't be established.
ED25519 key fingerprint is SHA256:....
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

Compare the fingerprint with the one your cluster publishes, and type `yes`.
`mace gui connect` waits for the answer, and after 15 s says "(waiting:
approve the 2FA prompt if there is one. If ssh asks whether to continue
connecting (the host-key yes/no question), check the fingerprint and answer
yes ...)". When the question could not be answered (or was answered no),
ssh says "Host key verification failed." and MACE prints the exact line to
run once, then connect again. When ssh's message names the gateway, that is
`ssh me@hpcc.example.edu true`; when it names the node, `ssh -J
me@hpcc.example.edu dev-node-01 true`; when it names neither (or with
`mace-gui-connect.ps1`, which leaves ssh's messages on the console), both, in
this order:

```
ssh me@hpcc.example.edu true
ssh -J me@hpcc.example.edu dev-node-01 true
```

Or use `--hop nested`, where the gateway checks the node's key with its own
list of known hosts (the gateway's own key is still checked on your
computer). MACE never turns host-key checking off or accepts a new
key for you, and `--hop auto` does not switch to `nested` after a host-key
failure: trusting a key is your decision. A key that has *changed* gets a
different message: ask your administrators before you remove the old key.

### Jump hosts and nodes that accept logins only from the login node

- `--jump bastion.example.edu` (comma separated for several) reaches the
  gateway through a bastion: `ssh -J bastion,gateway node`.
- `--hop jump` (one ssh path from your computer to the node, through `-J`)
  needs the node to accept your own login (key, agent, password). If the
  node asks for 2FA too, you see two prompts.
- `--hop nested` makes the gateway log in to the node itself
  (`ssh gateway "ssh node ..."`), for clusters whose nodes accept logins only
  from the login node. The inner login cannot ask anything (it runs with
  BatchMode): it works with host-based logins or keys on the gateway. The
  forwarded port then also sits on the gateway's own `127.0.0.1`, where other
  users of the login node could reach it; the session token still guards
  every request there, as on the node.
- `--hop auto` (the default) tries `jump`; when the node refuses the login
  through the gateway, it tries `nested` once and says "connecting through the
  gateway instead (hop = nested). Save it with --save so this is not tried
  twice." (A 2FA gateway then asks twice on that first run.)

### MACE GUI in its own window

`mace gui --open`, `mace gui connect` (and its Enter to open again) and the
standalone scripts open MACE GUI in a window of its own, without tabs or an
address bar: the app mode of a Chromium-family browser, the first one of
Brave, Chrome, Chromium, Edge and Vivaldi that is installed (Windows: in
Program Files or `%LOCALAPPDATA%`; macOS: in `/Applications` or
`~/Applications`; Linux and others: on `PATH`). The window is titled MACE
(with the folder's name), shows the MACE icon, and its title bar takes the
theme's colors. It runs in a browser profile of its own,
`~/.cache/mace/gui/app-profile` (Windows:
`%LOCALAPPDATA%\mace\gui\app-profile`), so it does not mix with your
browsing: its own taskbar entry (on Linux the window class `MACE`), and your
everyday browser's extensions, tabs and sign-ins stay out of it. The terminal
says "Opened MACE GUI in its own Brave window (--browser tab opens a browser
tab instead)."

- **A browser tab instead:** `--browser tab` (`-Browser tab` in PowerShell),
  or `open = "tab"` in the `[gui]` table of `~/.config/mace/site.toml` for
  good (`open = "app"` sets it back). A `$BROWSER` you set (VS Code's
  terminal sets one) also means a tab, unless you pass `--browser app`.
  Without any of those browsers (Firefox has no app mode) MACE opens a tab
  of your default browser.
- **Install MACE:** the page is an installable web app (a manifest named
  MACE with its icons), so a Chromium-family browser's menu offers to
  install it as an app (Install MACE, or Install page as app). The
  installed app opens the session it was installed from; after the server
  restarts it shows the sign-in page, so open the new link (or `mace gui
  --open`) as usual. MACE does not need the install: `--open` already gives
  the app window.
- **macOS:** the window is the browser's (Dock icon included); the page's
  icon and title are MACE's.

### Sign out and stop the server

**File › Sign Out and Stop Server…** (also in the command search, "quit" or
"logout", and as Stop Server… in the connection dialog of the status bar)
ends your session and stops MACE GUI on the server, cleanly: its temporary
folders and state files go, the port is free again, and every open tab shows
"MACE GUI has stopped" with the command that starts it again. Modified
documents ask first (Save All, or Don't Save, which keeps them as drafts in
this browser). The dialog lists the GUI tasks still running; stopping cancels
them, while **Stop When Tasks Finish** keeps the server (and your session)
until they end, with a banner in every tab and **Cancel Stop** to undo it.
Your SLURM jobs keep running; only MACE GUI stops. "Also remove the drafts and
recent files for this folder" is for a shared computer. Through `mace gui
connect`, its terminal then says "The server on dev-node-01 was stopped from
the browser." and exits.

### Updating MACE from the GUI

MACE GUI looks for a newer MACE with `mace update --check --json` 20 s after
it starts and then every 6 hours (Settings › Server: **Check for MACE
updates**, **Update channel** release, preview or latest, release by
default, and the interval). Offline, the check fails quietly. **Help › Check
for Updates…** looks at once, whatever the setting.

An available update shows a quiet **MACE 1.1.7 available** in the status
bar. Its dialog shows the installed and the new version, the CHANGELOG
sections in between, and what to know first:

- `mace update`'s refusal, word for word: local changes (with the `git -C
  <MACE folder> stash` line to copy), local commits, or a checkout you cannot
  write (a read-only or someone else's install: ask its maintainer);
- your queued or running SLURM jobs: their completion callbacks run this
  install's code. Updating while they run is generally safe for a
  fast-forward, but workflows in progress continue their next steps with the
  new code;
- GUI tasks still running (the restart would cancel them);
- a changed requirements.txt, with the pip command for the Python that runs
  MACE. **Also run this pip command during the update** is off unless you
  tick it; on a system or module Python, install them yourself (or set up
  your own environment, docs/update.md).

The choices:

- **Update Now and Restart** (**Update When Tasks Finish** while GUI tasks
  run): MACE GUI runs `mace update --yes --no-pip` (`--pip` when ticked),
  checks that the new code starts (every module compiles and every GUI
  module imports), and restarts the server in the same
  process: the same port, session path and sign-in, so the page reloads by
  itself with the new version. A local `mace gui` restarts in its terminal
  (a new sign-in link is printed there too); through `mace gui connect` or
  the connect scripts the connection and the tunnel stay, with no new browser
  tab, and sessions reattached to a lingering server stay attached;
  `mace gui --slurm` restarts inside the same job and stays up as long as
  its login-node session does. If the new code does
  not start (often: new requirements not installed yet), the old server keeps
  running and the dialog says what to fix; then stop it and start `mace gui`
  (or `mace gui connect`) again.
- **Update When I Sign Out**: **File › Sign Out and Stop Server…** says so,
  and the update runs after the server has stopped (no code is replaced under
  a running server), from a small helper; under `mace gui --slurm` it runs in
  the job before the job ends. The next start shows the result.
- **Not Now** reminds you in the next session; **Skip This Version** stays
  quiet until a newer one appears.

A server started with `--insecure-network` never updates from the browser:
anyone with the token could replace the code. Run `mace update` in a
terminal there.

### Why port forwarding and not X11

`ssh -X` with a GUI program draws every frame on the cluster and sends the
pixels to you. The HPCC dev nodes only have software OpenGL (llvmpipe), so a
3D viewer gets a few frames per second at best. With port forwarding, only a
few KB of JSON per structure cross the network, and your browser draws the 3D
view on your laptop's GPU. X11 still works as a fallback if you cannot open a
tunnel: run `firefox` on the dev node (if it is installed there) over
`ssh -X` and open the URL in it. It is slow, but it works.

## The window

The page is laid out like a desktop structure editor:

- a **menu bar** (File, Edit, View, Select, Build, Tools, Calculation,
  Analysis, Help) with a command search (**F1** or **Ctrl+Alt+K**; ⌥⌘K on a Mac), the
  theme switch and the settings;
- a **toolbar** with the tools, the view toggles (cell, bonds, labels,
  boundary atoms, axes, projection), view along a/b/c, the display range,
  the P1/Sym switch, Clean and Calculation Setup;
- the **Explorer** on the left (open documents, the folder grouped into
  outputs, results, structures and this folder; long names lose their middle,
  and the tooltip shows the full path). Its sections (Open Editors, Files,
  Materials.db, Recent) fold from their headings. Drag the line between two
  sections to resize them, or focus it and use the arrow keys; double-click
  it to reset. Drag a heading, or press Alt+Up or Alt+Down on it, to move a
  section. Right-click a heading for Move Up, Move Down, the sections to
  show and Reset Sidebar Layout (also under View › Explorer Sections);
- **document tabs** in the middle: each file you open is a tab. A `.cif`,
  `.d12` or single-point `.out` opens as a structure, an OPT `.out` as a
  trajectory, a BAND/DOSS file as a plot and `materials.db` as a database;
- the **Inspector** on the right (selection, structure, cell, symmetry and
  display for a structure; the selected material for a database), which the
  **Calculation Setup** panel (the old Deck tab) replaces while it is open.
  Click a section's heading to open or shut it. The two buttons at the top
  right of the Inspector (when it has more than one section) and of
  Calculation Setup open or shut every section at once (View › Inspector
  Sections › Expand All, **Ctrl+Alt+]**, and Collapse All, **Ctrl+Alt+[**);
  **Shift-click** on a heading does the same (Alt-click too, but many Linux
  desktops, KDE and Xfce among them, take Alt-drag to move the window).
  Which sections are open is remembered. The × at the end of the
  Inspector's header hides the whole Inspector (View › Inspector brings it
  back).
  In a window narrower than 1100 px it is a drawer over the viewport, shut
  until you ask for it (View › Inspector, **Ctrl+Alt+B**);
- a **bottom panel** with Atoms, Measurements, History, Timeline (the OPT
  energy chart and the cell, bond and displacement tables), Modes,
  Coordination, Tasks, Jobs and Problems (see
  [What the side panels show](#what-the-side-panels-show));
- a **status bar**: the connection to the server, the selection, the space
  group, the cell parameters, the atom under the pointer and whether the
  document is modified.

Every menu item, toolbar button, tooltip and shortcut comes from one
command list, so they always agree; **?** shows every shortcut and
**F1** (or Ctrl+Alt+K) finds any command by name. Items that are not built yet are shown
disabled, with "Coming soon". Single-key shortcuts (`1`, `V`, `?`) never fire
while you type in a field. **Ctrl+,** opens the settings; they are kept in
`~/.config/mace/gui.json` (or `$XDG_CONFIG_HOME/mace/gui.json`), outside the
served folder, with the rest of your profile (see
[Settings and your profile](#settings-and-your-profile)).

**Zen Mode** (View › Zen Mode, **Ctrl+Alt+Z**) hides the menu bar, toolbar,
activity bar and status bar. To leave it, click **Exit Zen Mode** at the top
right (it fades after a few seconds and comes back when you move the pointer
to the top), press **Esc** (when no dialog, menu, field, tool or selection
wants it first), use the viewport's right-click menu, or press Ctrl+Alt+Z
again. A reload keeps Zen Mode, with the button shown.

The browser tab reads "*document* — *folder* — MACE"; the menu bar shows the
folder's path, shortened: `~` for your home folder, `$SCRATCH/…`, `$WORK/…` or
`$PSCRATCH/…` for a folder under those (when your cluster sets them), with
the full path as its tooltip.

The page stays light over an ssh tunnel: it starts with one request, the 3D
viewer loads after the first paint and plotly only when a plot opens, every
asset is cached under a versioned URL (a reload sends about 8 KB), and a
heartbeat over Server-Sent Events drives the status bar's **Connected**. If
the tunnel drops, a banner shows the `ssh -L` line to run again. The 3D
views paint only when something changes, at most once per frame, and the
server's caches stay under `MACE_GUI_CACHE_MB` (512 MB by default).

### What the side panels show

The Inspector and the bottom panel always refer to the document in focus in
the center (with the editor split, the focused group's document), and each
names it in its header, for example **Inspector · NaCl.cif** and
**Atoms · NaCl.cif**. Nothing an earlier document showed stays behind.

| Panel | Follows | What it shows |
|---|---|---|
| Inspector | the document | its own sections (Selection, Nudge, Structure, Cell and Symmetry for a structure; Plot Options for a plot; Mode for normal modes; the volume's controls; the selected material); a plot, spectrum, volume or properties file adds its run's Structure and Cell, read-only |
| Atoms, Coordination | the document | the structure, or the run's frame on screen; for a plot, spectrum, volume, `.d3` or job log, the structure of the run it came from, read-only |
| Measurements | the document | a structure's or a run's measurements |
| History | the document | a structure's undo steps (and its Calculation Setup history) |
| Timeline | the document | an optimization run's energy chart and its cell, bond and displacement tables |
| Modes | the document | a FREQ output's normal modes |
| Properties | the document's run | what materials.db holds for the run the document came from (an output, or a plot, spectrum, volume or deck it wrote), or the same values computed on the fly |
| Tasks, Problems | every document | the session's tasks, and the problems of every open output and deck |
| Jobs | the folder | the SLURM jobs of the served folder |

When a panel has nothing for the document in focus, it shows a short note
instead, such as "No timeline: mgo_doss.DOSS.DAT is a DOS plot. The
Timeline tab applies to optimization runs.", often with a button such as
**Open mgo_doss.out**. Its tab is dimmed but stays where it is and can
still be clicked. Screen readers read the note, and Tab reaches it.

A band or DOS plot, an IR or Raman spectrum, a volume (cube), a 2D map, a
properties deck (`.d3`) or a job log has no atoms of its own. For these,
the Atoms table, Coordination and the Inspector show the structure of the
run the file came from, read-only and labeled with its source, for example
**Atoms · from mgo_doss.out (a DOS plot mgo_doss.DOSS.DAT)**. There is no
Nudge or Edit Cell, and the table refuses edits; **Open mgo_doss.out** opens
that run so you can edit it. MACE finds the run by name, nearest first:

1. the run of the same name beside the file (the FREQ run of
   `water_freq_IRSPEC.DAT`, the run of a job log `water_opt-123456.o`);
2. the material's SCF run (`<material>_sp.out`, `_opt.out` or `.out`) in
   the same folder, the folder above, or a folder beside it (a workflow's
   steps);
3. the properties run that wrote the file (its cell and atoms);
4. the material's `.d12`;
5. for a file with one of CRYSTAL's own names (`DENS_CUBE.DAT`, `fort.25`),
   the one output in its folder;
6. for a cube, the atoms in its header.

When two runs fit equally well, for example `Si_sp.out` and `Si_opt.out`
beside `Si_doss.d3`, the panel says so, offers both, and shows the one you
pick. A DOS plot's projections are listed with the atoms each one covers
("Projections: Mg (s, p, d, all) on atom 1").

When you switch back to a document, its panels are as you left them: the
scroll position, the selected atoms, and which sections are open.

## Keys, the command search, settings and context menus

### Keyboard shortcuts and how to change them

Every shortcut belongs to a command, and the menus, toolbar tooltips,
context menus, the command search and the shortcut sheet all read the keys
from that one list. **?** (or **Ctrl+/**) shows the sheet in four columns:
Global, File and Edit, Select and Tools, and View. Its **Print** button
prints it as one landscape page.

To change a key, open **Settings › Keyboard** (or **Edit keybindings…** on
the sheet). Click a binding and press the new keys; **+** adds a second
key. The change applies at once, everywhere the key is shown.

- **Esc** cancels the recording and **Backspace** removes the binding. To
  bind Esc or Backspace themselves, use **Enter as text…** and type `Esc` or
  `Backspace` (any chord, such as `Mod+Shift+K`, can be typed there).
- If another command already uses the key, the editor names it and asks
  before it takes the key over ("Ctrl+D is also Duplicate. Replace it?").
  **Pick another key** keeps recording.
- Enter and Space on the recorder's buttons press them. To bind Enter or
  Space, press it with the recorder panel itself focused, or type it.
- **Reset** on a row, **Reset all keybindings**, and **Export / Import
  keymap.json** are on the same page. Only your changes are saved: a list of
  `{"command": id, "keys": [...], "remove": true?}` entries, in Settings'
  `keymap`.

`Mod` means Ctrl on Linux and Windows and ⌘ on a Mac. Keys are matched by
their position on the keyboard (`event.code`), so they work with any
layout.

**Keys the browser owns.** These are refused, with the reason, because a
page cannot reliably use them: Ctrl+N, Ctrl+T, Ctrl+W, Ctrl+Shift+N/T/W,
Ctrl+Tab, Ctrl+L, Ctrl+Q, Ctrl+Shift+P, Ctrl+PageUp/PageDown, Ctrl+1..9, F5,
F11 and F12. That is why New Structure is **Alt+N**, Close Tab **Alt+W**,
Set Coordinates **Ctrl+Shift+U** (Ctrl+Shift+T reopens a closed browser tab)
and Focus Viewport **Ctrl+Alt+1** (Ctrl+1 is browser tab 1).

**Keys the editor warns about** (you can still use them):

- developer-tools keys that some browsers take first: Ctrl+Shift+I, J, C, K,
  E and M;
- keys that leave the page or switch tabs: Alt+Left and Alt+Right (Back,
  Forward), Alt+Home, Alt+1..9 (tab switching on Linux) and Backspace (Back
  in some browsers). While a command is bound to one of these, MACE keeps
  the key from the browser even when the command cannot run (it says why in
  a toast). So Alt+Left with the viewport unfocused does not go Back.
  Unbound keys stay the browser's.

Leaving or reloading the page with a modified document asks first.

**Menus from the keyboard.** Alt+F, Alt+V, Alt+S, Alt+T, Alt+C, Alt+A and
Alt+H open File, View, Select, Tools, Calculation, Analysis and Help. Edit
and Build have no Alt key by default: Firefox takes Alt+E and Alt+B for its
own Edit and Bookmarks menus.

**Keys Firefox keeps.** Measured with real key presses in a visible Firefox
(`tests/gui/test_browser_keys_xvfb.py`), Firefox acts on Alt+E and Alt+B
(its menus), Ctrl+K (web search), Ctrl+Shift+J (browser console),
Ctrl+Shift+M (responsive design mode), Ctrl+Shift+G (find previous) and
Ctrl+Shift+A (add-ons) before the page sees them, so MACE doesn't use them:
the command search is F1 or Ctrl+Alt+K, Select by Element Ctrl+Alt+E, Select
Bonded Neighbors Ctrl+Alt+N, Save Settings JSON Ctrl+Alt+J, Clear
Measurements Ctrl+Alt+M, Clean Geometry Ctrl+Alt+G and Select None Esc or
Ctrl+Alt+A. The keymap editor refuses those keys with the reason. F12 always
opens the developer tools. **F10** focuses the menu
bar, and the arrow keys then reach every menu. The mnemonics are listed
under Settings › Keyboard › Show: Menu mnemonics, and can be moved or
removed like any key.

**Viewer and timeline keys:** the arrow keys
turn the view by 15°, or by the angle snap's step when one is on (Shift: 1°),
**0** snaps the view to the nearest axis, and **+**/**−** zoom while the viewport has
focus; **,** and **.** step through an optimization (Shift: first and last
step), **Space** plays it, and **Ctrl+Alt+O** switches overlay and side by
side; **Ctrl+G** jumps to an atom; **Shift+F10** opens the context menu of
the focused row or tab. If a browser takes one of these keys despite MACE,
each has a `Mod+Alt+…` alternative (the atom styles: `Ctrl+Alt+1..4`).

### The command search (F1, Ctrl+Alt+K)

Type any part of a command's name; the first character picks a mode:

| Prefix | Searches | Also |
|---|---|---|
| (none) or `>` | commands, recently used first | **Enter** runs, **Ctrl+Enter** runs and keeps the search open |
| `#` | files in the folder | **Ctrl+P**, File › Open… |
| `%` | open documents | **Ctrl+E** |
| `@` | atoms of the open structure (`@O5`) | **Ctrl+G**; centers the view on the atom |
| `:` | a step of the open optimization (`:12`) | |
| `=` | a calculator: `= 0.5 0 0.5` gives Cartesian Å in the open cell, `= c 2.1 0 2.1` gives fractional | **Enter** copies the result |

Every command is listed, including the ones not built yet (with "Coming
soon"); Settings › Advanced › Show command ids adds each command's id.

### Settings (Ctrl+,)

Fourteen sections: Appearance, Viewport, Units and numbers, Plots, Editor,
Keyboard, Files and data, MACE defaults, Server, ML potentials, Cluster, This
folder, Advanced and Advanced: cluster defaults, with a search over all of them.
Changes apply at once; a **Reset** link shows next to anything you changed.
Rows marked "Not used in this version yet" are kept for a later version.

- **Appearance** › **Theme**: a card per theme (see "Themes and element
  colors" below); pointing at one previews it, a click keeps it. **Plot
  background** and **Plot style** are below it (see "Plot style" below).
- **Viewport** settings redraw the 3D views live with the camera kept: color
  scheme (VESTA, the default, Jmol, Colorblind safe, or one of your own
  palettes; **Edit Element Colors…** opens the editor), atom scale, bond radius, bond
  tolerance (drawn bonds only; the bond table keeps 1.2), bonds between
  metals, labels and their size, background, cell color, the axis triad,
  the cell and the boundary atoms; and (1.1.6) every surface's **Mesh
  quality** (Draft, Standard, High, Publication: see [Mesh
  quality](#mesh-quality)) and the pore surfaces' style, shading and walls.
- **MACE defaults** show what the CLI does when you leave a value blank
  (for example "CLI default: HSE06"), read from the CLI's own code. A blank
  GUI value means "use the CLI default".
- **ML potentials** (also **Calculation › ML Potentials…** and the **ML
  Potentials…** button in Clean and the Calculators) shows the ML
  environment MACE uses and where it came from, and each model with its code
  and weights license and status. **Set Up ML Environment…** runs `mace mlip
  install`'s own installer as a task: the dialog shows the location (default
  `~/.local/share/mace-tool/mlip/venv`), the packages with their licenses
  (mace-torch, CHGNet, ORB-v3, MatterSim), a CUDA option, and before you
  press Start the Python it builds from, the size (about 1.5 GB and 46,000
  files for mace-torch and CHGNet), the free space and, where `quota`
  answers, the space and file quota. The default model's weights download
  afterward only when you tick both the download and the license. The log
  streams live; **Cancel** stops pip and removes a folder the run created.
  When it is done, Clean and the Calculators see the new environment without
  a restart. **Use an Existing Python…** probes an interpreter and records it
  (`mace mlip setup --python`); **Download Weights…** shows a model's license,
  size, training data and citation and downloads only after you accept the
  license. A GUI listening on the network installs only into the default place
  and records no interpreter; on a host where the environment cannot start
  (a cluster gateway without its Python module) setting it up is disabled
  with the reason. UMA (fairchem-core) installs into its own environment:
  ticking it unticks the others and moves the location to `venv-fairchem`.
  Its weights are gated: **Download Weights…** shows the FAIR Chemistry
  License and its Acceptable Use Policy and links the Hugging Face page where
  you request access with your own account; MACE downloads with your login
  (`hf auth login` or HF_TOKEN) and never stores a token. MACE-OFF is listed
  as **Academic, non-commercial use only**: **Acknowledge and Download…**
  shows its Academic Software License and downloads only after you tick that
  you will use it for academic, non-commercial work; the date is recorded,
  and only then does it appear in the Calculators and Clean, its results
  labeled "(academic license)".
- **Advanced** exports, imports and resets the settings (the file is a
  browser download; nothing is written into the served folder).
- **Advanced: cluster defaults** shows the site settings MACE writes into
  every job script as `mace site` prints them, with where each value comes
  from (built-in, site.toml, a group file or `env MACE_...`). **Set** and
  **Unset** run `mace site set KEY VALUE` and `mace site unset KEY` (the key
  box lists every key, for example `resources.ntasks`,
  `resources.OPT.walltime` or `slurm.account`), and **Create site.toml**
  runs `mace site --init`. The CLI's answer shows below, a refusal in its own
  words. These are your own site settings
  (`~/.config/mace/site.toml`, or `MACE_SITE_CONFIG`), the same file the
  CLI uses. They are read-only while MACE GUI listens on the network, and an
  edit from a page whose folder has changed asks you to reload.

The server checks each value's type and range when it saves; a bad value is
named in a toast and goes back to its previous value, while the others are
saved.

### Themes and element colors

**Themes.** Settings › Appearance, View › Theme and **Color Theme…** in the
command search show the 18 themes in three groups:

- **Light & Dark**: System (the default: light or dark, as your computer is
  set), Light, Dark, **Sepia** (an e-reader's page: brown text on warm tan
  paper, a little softer than black on white, with warm accents only (rust,
  ochre, olive, plum: no blue), for long reading sessions; it was called
  Paper, and "paper" still finds it)
  and **True Black** (pure black surfaces for OLED screens);
- **Accessibility**: **High Contrast Light** and **High Contrast Dark**
  (black on white or white on black, text at 7:1, WCAG AAA, strong control
  borders, plot series and axes at 3:1 or more; the dark one has a yellow
  focus ring) and **Colorblind Safe** (the Okabe–Ito hues for the accent,
  status colors, plot series, the axis triad and Compare's two tones, so
  red–green color vision deficiency loses none of them);
- **Editor themes**, after popular editor color schemes, a family's dark
  and light flavors side by side: Catppuccin Mocha and Catppuccin Latte,
  Dracula, Gruvbox Dark, Nord, One Dark, Solarized Dark and Solarized Light,
  Tokyo Night and Tokyo Night Storm. Each takes its surfaces, accents,
  selection, borders, status colors, file badges, plot colors and 3D
  background from its published palette (all MIT licensed; the sources are
  noted in `css/tokens.css`); the layout and fonts stay MACE's. Where a
  palette color fell short of the contrast below, it is lightened or
  darkened in its own hue, and the 3D background is a shade of the theme
  that keeps the common element colors visible.

Every theme is a full set of design tokens (`css/tokens.css`).
`tests/gui/test_tokens_contrast.py` checks every one for WCAG 2.1 contrast:
body, muted and disabled text, links and status colors at 4.5:1 on every
surface (the page, panels, menu hover, a selected row), text on buttons,
banners, badges, the tooltip and toasts at 4.5:1, the focus ring and
control borders at 3:1, plot text, frame, grid and series on the plot
background, and the axis triad, cell and selection on the 3D background (the
high-contrast themes at AAA). Plots and the 3D views' backgrounds follow the
theme. Point at a card in Settings to preview it; in Color Theme… the arrow
keys preview each theme, Enter keeps it and Esc goes back. The choice is
part of your profile (gui.json) and is on the page's first paint in any
browser. Ctrl+Alt+T still cycles System, Light and Dark.

**Dark interface, light plots (or the other way round).** Settings ›
Appearance › **Plot background** is Follow theme (the default), Light or
Dark. Light or Dark gives the band, DOS, run (SCF, optimization), spectrum,
volume-slice and Gallery plots and the FREQ stick spectrum a surface of
their own with matching text, grid, axes, Fermi and zero lines, legend and
series colors, in any theme; PNG and SVG exports from the GUI match what is
shown (with Follow theme they keep a see-through background, as before).
`mace plotting` figures are made by the CLI and do not depend on the GUI
theme.

**Plot style.** Settings › Appearance › **Plot style** sets the look of every
plot, and **Band & DOS style**, **Spectra style**, **Maps & volumes style** and
**SCF & run charts style** below it can each take a preset of their own ("Same
as Plot style" follows it). The presets (`mace/plotting/presets.py`, the same
for `mace plotting --style`):

| Preset | What it draws |
|---|---|
| Default | The theme's series colors and grid |
| Publication | A figure for a paper in any theme and on any Plot background: Okabe–Ito colors after a first series in black (white on a dark plot), black axes, frame and outside ticks, larger tick labels, heavier lines, no decorative grid (the band structure keeps its lines at the high-symmetry k-points); spin colors black/vermilion |
| VESTA-like | Maps and volumes only: VESTA's rainbow section map with black contour lines, yellow (+) and cyan (−) isosurfaces. Other plots draw Default |
| Presentation | Large text, thick lines and frame: readable from the back of a room |
| Print grayscale | Black and grays, the series of spectra and run charts told apart by line style and markers; spin down gray dashed |
| Colorblind safe | Okabe–Ito colors, a blue / vermilion spin pair of its own, cividis maps |

The band, DOS, run (SCF, optimization), IR/Raman and transport plots, volume
slices and isosurfaces, 2D maps, the Gallery and the FREQ stick spectrum all
follow their family's preset. PNG and SVG exports from the GUI are the figure
as shown, and Export › Publication Figure… passes the same style, DOS projections
and spin styling to `mace plotting`. A folder's `mace-folder.toml` can set
`[plots] style`, `spin_colors`, `spin_up_color`, `spin_down_color`,
`band_color`, `band_style`, `spin_down_style`, `dos_colors`, `dos_spin`, `dos_total` and `dos_smooth` for the GUI and the
CLI: the same names as the plot's Style controls and `mace plotting`'s flags
(`--style`, `--spin-colors`, `--spin-up-color`, `--spin-down-color`,
`--band-color`, `--band-style`, `--spin-down-style`/`--spin-down-solid`, `--dos-colors`, `--dos-spin`,
`--no-dos-total`/`--dos-total`, `--dos-smooth`), and `[plots] potential_colors`
(`chemistry` or `physics`, `--potential-colors`; the GUI's Settings › Viewport ›
Potential colors). Two more `[plots]` keys are for `mace plotting` only:
`energy_reference` (`--energy-reference`: `auto`, `fermi` or `vacuum`) and
`backend` (`--backend`: `plotly` or `matplotlib`). For the backend,
`--backend` wins, then the folder's file, then `site.toml`'s `[plots] backend`,
then the `MACE_PLOT_BACKEND` environment variable, so a `site.toml` value
overrides the variable.

**DOS projections.** Settings › Plots › **DOS projections** (and a DOS plot's own Plot
options › DOS projections; `mace plotting --dos-colors`): **Auto** (by orbital for one element, element × orbital
for two or three, by element for more), **Total only**, **By element** (each
element in its 3D color, Viewport › Color scheme, darkened where too light to
read), **By orbital** (s, p, d, f in four hues, the same for every element;
elements by line style) or **Element × orbital** (shades of each element's
color, s strongest, s/p/d/f also by line style). **DOS total** draws the total
beside them. The legend names projections "Mg s", "O p", "Mg".

**The DOS layout.** A DOS draws the 2 to 4 projections that carry the most
DOS in the energy window (each at least 4% of it); the others are listed,
unchecked, in Plot options › Projections (a click on a projection's legend
swatch hides it too). With DOS projections Auto they take colors that keep clear of
the spin pair. A spin-polarized DOS is mirrored: spin up on the right, spin
down on the left, "↑ up" and "down ↓" over each half, magnitudes on both
sides, each total lightly filled in its spin's color and each projection in
one color on both sides, dashed on the spin down side. A closed-shell DOS is
one-sided: the total as a light fill under a line in the color pair's first color (the band color), the
projections on top. The legend has two titled groups: **Spin** (neutral keys,
spin up solid and spin down dashed, or "α = β (closed shell)") and
**Projections** (color swatches). A band plot alone has it on one line above
the frame, a DOS alone inside the plot where it covers no curve (else right of
it), band + DOS right of the plot. E<sub>F</sub> is dotted (a dash means spin
down). Energies the DOS run did not compute are shaded and, on a DOS alone,
labeled "outside the DOS window (no data)". **DOS smoothing** (Settings ›
Plots, a plot's Style section and Style ▾; `mace plotting --dos-smooth EV`,
`[plots] dos_smooth`) applies a Gaussian of σ = 0.05 eV (or the σ given) to
the drawn curves only, and the plot says so; it is off by default.

**Colors and spin.** Settings › Plots › **Color pair** picks the pair the bands
and the DOS share: spin up / spin down of a spin-polarized run, and a closed-shell
run's bands and DOS total (below). One pair for a light plot and one for a dark
plot, the same in `mace plotting --spin-colors`:

| Color pair | Light plot (up / down) | Dark plot (up / down) |
|---|---|---|
| Auto | the plot style's: Blue/red in Default and Presentation, Black/vermilion in Publication; Print grayscale and Colorblind safe keep their own pairs | |
| MACE pink/cyan | `#e0218a` / `#00a0a4` | `#fa26a0` / `#2ff3e0` |
| Blue/red | `#1f4e9c` / `#c0392b` | `#7aa7f0` / `#f08a7c` |
| Black/vermilion | `#000000` / `#d55e00` | `#ffffff` / `#e8803a` |
| Custom | **First color** and **Second color** | |

A color set in First/Second color also applies with Auto. **Band style**
(Settings › Plots, a plot's Style section, `mace plotting --band-style`,
`[plots] band_style`) draws a run that is not spin-polarized (closed shell, or
α = β): **Single**, one solid line in the pair's first color, so blue bands
beside a blue DOS total by default (**Band color** overrides it: Settings ›
Plots, the Style section, `--band-color`); **Overlay**, that line with a dashed
line in the second color on top, the look of a spin-polarized plot; **Ink**, the
neutral ink color (the plot's text color, or the Publication, Print grayscale and
Colorblind safe style's own). **Auto** (the default) is
Single when the bands have a gap at E<sub>F</sub> and Overlay for a metal (a band
reaching 0.01 eV above and below E<sub>F</sub>). The one-sided DOS total takes the
band color (Ink: ink). The legend key reads "α = β (closed shell)". A spin-polarized
run always draws spin up solid in the first color and spin down dashed in the second,
whatever the Band style. **Spin down line**
is dashed by default in every scheme, so the two spins read apart without
color (dotted, dash-dot or solid; `--spin-down-solid` in the CLI), and **Spin DOS**: Auto draws a
spin-polarized run mirrored (spin down on the left) and a closed-shell run
one-sided, as `mace plotting` does; Mirrored or Overlaid (both spins on the
right). An unrestricted run whose α and β DOS are equal (non-magnetic; MACE's
spin-polarized default) is drawn one-sided as the total, α + β; Mirrored draws
one spin on each side. Spin down is drawn over spin up, so a weakly polarized run reads in
both colors; the subtitle says how far apart α and β are. With both spins
drawn, the DOS projections are all solid and told apart by color (each
element's orbitals by a shade of its color, the largest strongest), so a dash
always means spin down.

**A plot's own style.** Every plot's Plot options has a **Style** section with
**Plot style**. A band or DOS plot adds **Colors** (**Color pair**, with two
color pickers for Custom: it drives both halves) and its halves' groups: **Bands**
(**Band style**, **Band color** while a closed-shell run is drawn Single, and
**β dashed**) and **DOS** (**DOS projections**, **DOS spin layout**, **DOS total**,
**DOS smoothing** and its σ). A band + DOS plot has both groups, whether you
opened the BAND.DAT or its DOSS.DAT; a BAND.DAT alone has no DOS group, and a
DOSS.DAT alone no Bands group (its total's color and β line are in DOS). Each starts at "Settings (…)", which follows Settings and names
what that is; a choice applies to this plot only, redraws at once and is
undone with Mod+Z like every plot option. **Reset Plot Options** goes back to
Settings. **Make This the Default** writes the plot's style, both halves, into
Settings (and the plot follows Settings again). The **Style ▾** menu in the
plot's toolbar, next to Export, has the same controls in the same groups one
click away (a color opens the color picker), **Follow Settings**, and
**Copy CLI Command**.

**Background** (in Style, and Settings › Plots) is Theme, White, Black or a Custom color, as the 3D view's; text and grid turn light or dark to match, PNG/SVG exports follow it, `mace plotting` keeps white.

**Copy CLI Command** (Style ▾ and Export ▾) copies the `mace plotting` command
that draws the plot on screen: `--band`, `--dos`, or `--band-dos` when the
plot shows both, with the file, `--e-lower`/`--e-upper`, `--fermi-shift`,
`--energy-reference=vacuum` with a POTC run, `--units` other than eV and the
style flags that are not the defaults; an edited k-path comes first as a
`band_path.json` written by the command, then `--band-path band_path.json`.
When the folder has a `mace-folder.toml`, the command also names the values
equal to `mace plotting`'s defaults (`--style=default`, `--spin-colors=auto`, `--band-style=auto`,
`--fermi-shift=0`, `--units=eV`, `--energy-reference=fermi`, `--dos-total`,
...), because the folder file's `[plots]` and `[units]` values would replace
any default left out. Run in the data's folder it draws the same figure.

*White figures* for a paper or slides: Plot style **Publication** with Plot
background **Light** gives black-on-white plots in any theme; for the 3D
view add Settings › Viewport › Lighting preset **Publication** (white
background, outlines, ambient occlusion). For the 3D view, Settings › Viewport › **Background** (or Inspector ›
Display) White or Black keeps it light or dark in any theme: when it is the
other lightness from the theme, its axis triad, labels, measurement chips,
cell lines and selection switch to that lightness's colors, so they stay
legible (a background of the theme's own lightness changes nothing).

**Atom sizes.** Settings › Viewport › **Radii** picks the radius every view
draws an atom at, times **Atom scale**: **Atomic** (the default: VESTA's atomic
radii, so Atom scale 0.4 is VESTA's ball and stick, 40 % of the radius: Na
0.76, Ti 0.59, Fe 0.50, O 0.30 Å), **Covalent** (Cordero 2008), **Ionic**
(Shannon, six-coordinate) or **van der Waals** (the sizes drawn before 1.1.6:
0.28 × the van der Waals radius, every transition metal the same size).

**Element colors.** View › Element Colors… (also Settings › Viewport ›
Color scheme › Edit Element Colors…) shows the periodic table in the
palette being edited. Click an element, then pick its color or type it as
`#rrggbb`; every 3D view (structures, the editor, FREQ modes, slabs,
volumes, Compare and both editor groups), the periodic-table picker, the
slab preview and exported images draw the change at once. **Reset to …**
gives an element back its preset color; atomic radii are not affected.

- Presets: **VESTA (MACE default)**, **Jmol (CPK)**, **Colorblind safe**
  (MACE's own scheme, from the Okabe–Ito hues; families of elements share a
  hue and step in lightness, and the common elements are set one by one) and
  **Legacy** (the colors `mace plotting` drew structures in before 1.1.6;
  as the default, the CLI draws every plot in its own old colors).
- **Save as New** stores the palette under a name; **Save**, **Rename** and
  **Delete** manage your own; **Set as Default** makes it the palette of
  every view, in every browser, and of `mace plotting`.
- **Export…** downloads the palette as JSON (the changed elements and every
  element's color); **Import…** adds such a file as a new palette.

Your palettes and the default are kept in
`~/.config/mace/element_colors.json` (`$XDG_CONFIG_HOME/mace/`), beside
gui.json, so the CLI reads them too:

```json
{"version": 1, "rev": 3, "default": "user:My oxides",
 "palettes": {"My oxides": {"base": "vesta", "colors": {"O": "#d55e00"}}}}
```

`default` is a preset (`vesta`, `jmol`, `cvd`, `legacy`) or `user:<name>`; a palette
stores its base preset and only the elements it changes. A hand-edited file
with a bad entry loses only that entry. `mace plotting` draws atoms in the
default palette (VESTA when none is saved, as in the GUI), or in
`--palette NAME` (`docs/cli.md`).

### Element legends

**File › Export Element Legend…** (also in the 3D view's right-click menu)
makes a legend of the structure's elements, for figures. The Explorer's
right-click menu has it for the selected files or a folder (the union of
their elements), Batch Setup has an **Element Legend…** button for the
selected files, and **File › Export Element Legend of All Open
Documents…** takes every open structure. The dialog previews the legend
as you change it: the palette (the default, any other, or the one being
edited in Element Colors), the order (atomic number, first appearance,
alphabetical), vertical, horizontal or grid layout, circle or square
swatches and their size, the font size, symbol, full name or both, dark
or light text (from the theme; Plot style Publication gives black on white) and the
background. Export writes SVG (each element is its own group with real
text: editable in Inkscape, Illustrator or PowerPoint), PNG at a DPI, or
a TSV/CSV/JSON table (symbol, name, Z, hex color, RGB). The server makes
the file with the same function as `mace plotting --legend`, so the two
give the same file for the same choices.

**Export Image…** has an **Element legend** option that puts the legend
in a corner of the PNG or SVG (in the SVG as its own editable group).

### Settings and your profile

Everything you choose in the GUI is kept on the machine that runs the
server, in `~/.config/mace/gui.json` (or `$XDG_CONFIG_HOME/mace/gui.json`),
outside the served folder: the Settings you changed, the layout (panel
widths, the bottom tab, collapsed Inspector sections, the Explorer's
sections, database column widths, Setup's sections, each folder's split into
two editor groups), remembered choices (the
last Export Image choices, the New Structure form, the Clean method, the
Submit dialog's last options, the last commands of the command search, recent
elements), the Recent files of each folder and the Recent folders of File ›
Open Folder. Each document's own view options (its display range, boundary atoms,
rendering) are kept next to it, in `gui-docprefs.json`.

So everything comes back when the local port changes (a new ssh tunnel, VS
Code's port forwarding), in another browser, and on another computer that
shares your home folder, except:

| Kept in this browser only | Why |
|---|---|
| Unsaved drafts of edited structures and of Calculation Setup | they are work in progress, offered again after a reload in the same browser |
| Zen Mode and the narrow-window Inspector drawer | a small screen should not open another computer in Zen |
| The "overwrite this file?" question, asked once per file | a safety prompt |
| A plot's own options and its undo history, Run Analysis's and the energy card's options, Follow the run | per browser for now |

A change made just before you close the tab is still saved (it is sent as
the page closes). The browser keeps a copy (localStorage) so the page starts
at once; the server's copy wins when they differ, unless a change made here
has not reached the server yet. The page's theme, density and UI scale come
with the page itself, so a new port does not flash the defaults. Two servers
of your account (two folders, or two cluster nodes sharing your home) can
save at the same time: each change is merged per key under a file lock, and
the file is read back to catch a lock that did not work across nodes. The
first save of this version keeps a copy of the older file as
`gui.json.v1.bak`.

Settings › Advanced:

- **Export settings…** downloads your profile as `mace-gui-profile-<date>.json`.
  Choose what goes in: Settings, Layout and panels, Remembered choices (on by
  default), and Recent files and folders and Per-document view options (off
  by default: they hold paths on this computer; leave them off when sharing
  the file).
- **Import settings…** reads such a file (or a settings file from an earlier
  version), says what it will replace, and merges it after you confirm; a
  value the server refuses is named and the others are applied.
- **Reset all…** asks which parts to reset (Settings and keybindings, Layout
  and Remembered choices by default) and first saves a copy of your profile
  next to it (`gui.json.before-reset-<time>.json`; the newest 3 are kept).

These cover the GUI profile only; the profiles of `mace gui connect` are
never exported, imported or reset here.

### Folder settings (mace-folder.toml)

A project folder can carry its own defaults, for everyone who works in it:
a `mace-folder.toml` in the folder (or in a parent folder below your home).
While `mace gui` serves that folder, its values override your own defaults;
the CLI commands run inside it read the same file for the settings they
have (below). The file is commented TOML, holds data only and is safe to
keep in git or share with your group. Settings > Save to this folder writes
it, and so does `mace site set --folder [DIR] KEY VALUE` from the shell
(docs/cli.md, "Folder settings from the shell"), with the same checks:

```toml
[viewer]                      # theme-independent display options
color_scheme = "vesta"        # or "jmol", "cvd", "legacy" (a preset)
atom_scale = 0.4
bond_tolerance = 1.15
show_cell = true
projection = "orthographic"   # or "perspective"
[units]
energy = "meV"                # eV, meV, Ha, kJ/mol, kcal/mol
[plots]
energy_window = [-6.0, 6.0]   # band/DOS, eV; also mace plotting --e-lower/--e-upper
fermi_shift = 0.0
log_delta_e = true
energy_reference = "auto"     # or "fermi", "vacuum": mace plotting --energy-reference only
backend = "plotly"            # or "matplotlib": mace plotting --backend only
[setup]
functional = "HSE06"
basis_set = "POB-TZVP-REV2"
opt2d12_config = "presets/sp.json"    # mace opt2d12 --config-file, when none is given
opt2d3_config = "presets/band.json"   # mace opt2d3 --config-file, when none is given
cif2d12_options = "presets/cif.json"  # mace convert --batch --options_file, when none is given
[database]
columns = ["material_id", "formula", "space_group", "status", "updated_at"]
[submit]                      # within what site.toml's [folders] table allows
ntasks = 64
walltime = "2-00:00:00"
[submit.properties]
ntasks = 16
walltime = "4:00:00"
```

**Which value wins**, per setting: a change you make in this GUI session or
a command-line option > the folder's file > your own defaults
(`gui.json`; `site.toml` for cluster resources) > MACE's built-in values.
For cluster resources the folder's `[submit]` values sit among site.toml's
`[resources]` tables, not above all of them: the environment
(`MACE_SLURM_*`) and a `[resources.<TYPE>]` table still win over the
folder; the folder's `[submit.properties]` wins over `[resources.properties]`,
but its plain `[submit]` values do not (its walltime covers properties jobs
only when neither sets one); a folder's walltime is never above the
partition's MaxTime (docs/cli.md, "Per key, the first of these wins").
The nearest `mace-folder.toml` is used (files are not merged);
`MACE_FOLDER_SETTINGS=none` turns folder settings off and
`MACE_FOLDER_SETTINGS=/path/mace-folder.toml` names one.

**In Settings**, once the folder has a file, each value that a folder can
hold shows where it comes from: *Built-in*, *Mine*, *This folder* or *This
session*. A change you make to a value the folder sets stays in this
session (it is not written to your `gui.json` or to the folder's file)
until you choose:

- **Save to this folder** writes the value into the folder's
  `mace-folder.toml` (it creates the file, commented, the first time). It
  shows next to any such setting when you point at it.
- **Save as my default** keeps the value as your own default. Here the
  folder's value still applies; elsewhere yours does.
- **Reset to my default** removes the folder's value from its file (or
  forgets the session's change), so your own default applies.

**Calculation Setup** starts a deck made from a CIF with Settings'
Functional and Basis set (from this session, the folder's file or your own
settings, in that order) and says where its default came from; a
follow-up deck keeps its parent's method. Blank means the CLI's own default.

Settings › **This folder** shows the file, anything in it MACE ignored and
why, the GUI values it sets and its command-line values. The previous
version of the file goes to File History before each save. A file in a
parent of the served folder is read but not written (open its folder to
change it), and so is a file MACE refuses (below).

**On the command line**, run inside the folder:

| Command | Reads |
|---|---|
| `mace plotting` | `[plots] energy_window` as `--e-lower`/`--e-upper` when they are not given, and as the energy range its interactive questions offer; `fermi_shift`, `energy_reference`, `backend`, `style`, `dos_colors`, `dos_spin`, `dos_total`, `spin_colors`, `spin_up_color`, `spin_down_color`, `spin_down_style` and `potential_colors` as the flags of the same names (`dos_total = false` is `--no-dos-total`) when they are not given; `[units] energy` as `--units` |
| `mace convert` (cif2d12) | `[setup] functional` and `basis_set` as the defaults of a deck made from a CIF (a name with `-D3`, such as `PBE0-D3`, turns D3 on; without it D3 follows the built-in default) |
| `mace opt2d12` | the same `functional`, only where the parent's functional cannot be used (instead of HSE06); a follow-up otherwise keeps its parent's method |
| `mace opt2d12`, `mace opt2d3` | `[setup] opt2d12_config` / `opt2d3_config` as `--config-file` when none is given (and no `--save-options` / `--save-config`). GUI Calculation Setup does not start from these presets (its command names its own file); it says so above its form |
| `mace convert --batch` | `[setup] cif2d12_options` as `--options_file` when none is given (never with `--save_options`) |
| `mace submit`, GUI Submit | `[submit]` job size and walltime per job kind (CRYSTAL; properties), and a partition, within `site.toml`'s `[folders]` rules; `--walltime`, `-i` answers and `MACE_SLURM_*` still win |
| `mace workflow` plans | the same `[submit]` job size and walltime, stored in the plan (a folder's walltime is not scaled by calculation type) |

Each command says once which value it took from which file. Cluster
resources stay the site's: by default a folder may set `ntasks` and
`walltime`, and may choose a partition only from `[folders] partitions`
in `site.toml`; `allow`, `max_ntasks` and `max_walltime` narrow that
(`mace site` shows what a folder set). A value outside these rules is
ignored with a warning.

**Safety.** The file can never run anything: it has a fixed set of keys,
each checked for its type and range, and a key MACE does not know or a bad
value is ignored with a warning. Paths in it are relative to its folder and
must stay inside it (no absolute paths, no `..`, no symbolic link out). A
file that everyone can change, one in a folder everyone can write to (a
scratch root), a symbolic link, or invalid TOML is ignored as a whole.
Without a `mace-folder.toml`, MACE behaves exactly as before.

### Context menus

Right-click (or press the Menu key or Shift+F10):

- **an atom**: Select Equivalents, Select Element, Select Neighbors, Replace
  Element…, Set Coordinates…, Delete, Measure from Here, Center View Here,
  Copy Coordinates (Fractional / Cartesian, of the image you clicked), Properties;
- **empty space** in the viewport: Add Atom Here, Paste, Select All, View
  Along, Reset View, Display Range, Export Image…;
- **a file** in the Explorer: Open, Open to the Side, Open as Text (a
  read-only text tab), Open as Structure (an ELASTIC, EOS, QHA or CPHF
  output, which Open shows as its results), Compare With Active, Plot, Copy
  Path, Reveal in Folder, Set Up Calculation;
- **a tab**: Close, Close Others, Close to the Right, Copy Path, Reveal in
  Explorer.

Each entry is a command, so it shows its key and can be run from the
command search too. Esc closes the menu and puts focus back on the file,
tab or view it was opened for.

### How much of the machine it uses

`mace gui` shares the machine politely (cluster dev nodes are shared, and a
workstation may run other jobs on its GPUs):

| Resource | Default | Setting | Environment variable |
|---|---|---|---|
| CPU workers (the process pool and the server's own symmetry threads, counted together) | a quarter of the usable cores, 1 to 8, within SLURM's allocation; nice 10 | Advanced › CPU workers (`max_workers`, 0 = auto) | `MACE_GUI_MAX_WORKERS` |
| Cache memory | 512 MB, at most a quarter of RAM, at least 64 MB | Advanced › Cache limit (`cache_mb`) | `MACE_GUI_CACHE_MB` |
| GPU for MACE's own GPU features (the server uses none today) | GPU 0; None keeps everything on the CPU | Advanced › GPU (`gpu_device`) | `MACE_GUI_GPU` |

**The process pool and the Tasks panel.** Builds over 2000 atoms, Clean,
the symmetry sweep, defect searches and the explorer's file details run in
worker processes: started with the first such job, at most the CPU-workers
number of them, at nice 10 with one BLAS thread each, and stopped after 60 s
idle (about 0.3 s to start one again, about 100 MB each). They share one CPU
budget with the server's own symmetry threads, so the two never add up to
more than the cap. A job answers at once when it takes under 0.3 s;
otherwise it becomes a task: the dialog shows its progress, and closing the
dialog hands it to the **Tasks** panel (bottom), where it can be canceled
(it stops within 2.5 s), retried, or its error copied. Opened files, built
structures that are too big to edit and the server's cache share the one
cache-memory budget above.

An environment variable wins over the setting. A setting saved on a bigger
host (say `max_workers: 64` on a cluster node) is kept as it is and limited
to what the current host has when it is used; Settings › Advanced shows
both numbers.

**Which GPU the browser draws on.** The browser, not the server, does the
3D drawing, and the operating system picks its GPU: by default GPU 0, the
one that usually drives the display. On a Linux workstation with a second
NVIDIA GPU, `mace gui --open --gpu 1` (for example) starts the browser with
PRIME render offload on GPU 1 (GPU *n* ≥ 1 is the X provider
`NVIDIA-G`*n−1*, see `xrandr --listproviders`). It only affects a browser
that `mace gui` starts; a browser that is already running keeps its GPU. On a shared node, do not raise the CPU
workers above the default.

The headless browser used by the tests and benchmarks always draws in
software (Mesa's llvmpipe) and touches no GPU.

## What it shows

| Tab | What you get | MACE code it uses |
|---|---|---|
| **Structure** | A 3D view of a `.cif`, a `.d12` or a CRYSTAL `.out`. You get the unit cell box, the atoms on the cell faces (as VESTA draws them), a 2×2×2 supercell, bonds, atom labels, and VESTA element colors (Jmol colors as an option). The camera fits the cell and every atom, also in narrow panes. A SLAB's box spans the slab itself; **Vacuum** pads it with 4 Å on each side. Hovering over an atom shows its fractional coordinates and Wyckoff letter. Alongside the view are the reduced formula with Z, the space group (spglib), the lattice parameters and an atom table. | ase (CIF), `d12_parsers.CrystalInputParser`, `seekpath_interface._geometry_blocks` |
| **Editor** | Edit a 3D structure, or build a new cell from lattice parameters (optionally a space group plus the asymmetric unit). See [Editing](#editing) below. | ase, spglib |
| **Compare** | A VESTA-style before and after, for an OPT `.out` (frame 0 against the selected frame), for an output against its input (**Analysis › Compare with Input**), for any two structures (**Tools › Compare Two Structures**), and for Clean. Four looks, cycled by Ctrl+Alt+O: overlay (the initial as a ghost), two-tone, side by side (linked cameras), and a displacement map colored by \|Δr\|. Tables of Δcell, Δbond, displacements and Δsymmetry in the Timeline panel. See [Compare](#compare). | `_geometry_blocks`, spglib, scipy (optional) |
| **Plots** | Band structures, DOS, IR and Raman spectra as documents with their own options, and FREQ normal modes animated in the viewport. See [Plots](#plots-band-dos-ir-raman-and-freq-modes). | `Plotting/ipBANDS_V2.py`, `Plotting/ipDOS_V2.py`, `spectra_api.read_spectrum`, `vibmode_viewer.Crystal23FreqParser` |
| **Database** | A read-only document for a `materials.db`: sorted, filtered and paged on the server, so 100,000 materials scroll like ten. See [The Database document](#the-database-document). | the MACE schema, through a read-only sqlite connection |
| **Calculation Setup** | The CLI's own questions as a form, the deck the printed command writes, and Run Preflight. See [Calculation Setup](#calculation-setup). | `mace convert`, `mace opt2d12`, `mace opt2d3`, `mace preflight` |

### The GUI → JSON → CLI loop

Calculation Setup has no deck writer of its own. Every value it shows comes
from the CLI: its defaults, its menus and its questions. Previewing runs the
printed command in a scratch mirror of the folder and shows the deck the CLI
writes. **Save Settings JSON** then writes that same JSON, so the printed
command (for example

```
mace convert --batch --options_file gui_options.json --cif_dir structures --output_dir decks/
```

) writes the deck you previewed, byte for byte. `tests/gui/test_setup_routes.py`
checks this against the manual, interactive route of each command for the
matrix of calculation types.

### Formulas

The formula is the **reduced formula** with the elements in ascending Pauling
electronegativity, as pymatgen writes it: MgO, LiF, SiO2, GaAs, BaTiO3, CsPbI3,
NaHCO3. Z, the number of formula units in the cell, is shown beside it
(conventional MgO: Z = 4; diamond: C, Z = 8). `materials.db` keeps the cell's
full formula in Hill order.

## Editing

A CIF, a 3D `.d12` or the structure of an `.out` opens as a structure
document, and it is edited where it is shown: there is no separate editor
mode. The editor's code loads with the first structure document, so an idle
page or a reload fetches none of it. Editing needs a cell of at most 5000
atoms; a slab, polymer or molecule is edited in P1 (see [Slabs, polymers and
molecules](#slabs-polymers-and-molecules-116)). Larger cells and runs say why
their editing commands are off (a run's step can be edited with **Edit This
Step**, below).

### The pointer

The pointer works the same way in every tool, as in OVITO, Avogadro and
Blender:

- the wheel zooms (scrolling up zooms in, as in VESTA; the run and
  comparison views scroll the same way, and Settings › Viewport › Invert
  scroll zoom flips it everywhere) toward Settings › Viewport › **Zoom
  toward**: **View center** (the default: the middle of the view, which is
  the rotation center, stays put, so a turn after zooming is still about the
  same point) or **Mouse pointer** (the point under the pointer stays under
  it, as in map apps; the rotation center moves with the view). Both apply
  to every 3D view, live, and are also in the toolbar's View Along ▾ menu. A
  middle-drag pans, a right-drag
  orbits, and Alt+left-drag orbits from any tool (for trackpads);
- a right-click without dragging opens the atom menu (on an atom) or the
  viewport menu (on empty space);
- the left button belongs to the tool. With **Select** (V) a click picks an
  atom, Shift+click adds, Ctrl+click toggles, a drag draws a box, a
  double-click selects the symmetry orbit and a triple-click the bonded
  fragment. **Lasso** (L) draws a free polygon; **Orbit** (O), **Pan** (H) and
  **Zoom** (Z) turn the left button into that motion.

**Rotation.** Settings › Viewport › **Rotation** and **Angle snap** (also
the toolbar's **View Along ▾** menu and View › Rotation and Zoom)
set how a drag turns every 3D view, the structure view and the 3Dmol run,
mode and volume views alike, live:

- **Free** (the default) is the trackball: a drag turns about the screen axis
  across it, so a few diagonal drags also roll the structure.
- **Turntable**: a horizontal drag turns about the
  structure's up (whatever pointed up on screen when you started, kept until
  something else rolls the view, such as an in-plane turn or View Along), a
  vertical drag about the screen's horizontal axis. Up never leaves the
  screen's vertical, so the structure stays upright however long you drag.
  The arrow keys Left and Right turn about the same up.
- **Angle snap** Off, 15°, 30°, 45° or 90°: a drag's horizontal and vertical
  turns are summed from where it began and applied in whole steps (a drag
  of half a step jumps one step), with either rotation mode; the arrow keys
  then turn by one step instead of 15°.
- **Snap View to Nearest Axis** (**0**, View menu, the viewport's empty-space
  menu, the View Along ▾ menu) turns the view to look along the nearest of a,
  b, c, a\*, b\*, c\* (a periodic structure) and the Cartesian x, y, z,
  either sign, with the nearest of those across it as up; the View Along
  button then names it. Use it after a free drag to square the view up
  again; View Along and Rotate 90° are unchanged.

Selected atoms get an amber ring and a translucent shell, and the atom under
the pointer a blue ring (the halo). The rings are drawn over the 3D view
after every frame, so they stay sharp and follow the camera; they are left
out of exported images. The status bar shows the atom
under the pointer with its Wyckoff letter and coordinates (Shift for
Cartesian).

Large cells are drawn at a lower level of detail, and a badge in the corner
says so: up to 2000 displayed atoms everything is drawn (LOD 1); up to
20,000 bonds become lines and labels show on hover only (LOD 2); up to
200,000 bonds are left out unless you turn them on (LOD 3); beyond that the
drawing is simplified (LOD 4). Settings › Viewport › Quality halves (Low) or
doubles (High) these limits. Bonds are found in a web worker
on a cell-list grid (10,648 atoms in a few tens of milliseconds), and box and
lasso selection run there too.

### Tools and keys

- **Move** (G): with a selection, G makes it follow the pointer in the
  screen plane, Blender style; a drag on a selected atom does the same. While
  moving, X, Y or Z locks a Cartesian axis and Shift+X/Y/Z a cell axis, typed
  digits give the offset in Å, Ctrl snaps to 0.05 of a cell axis, Enter or a
  click applies, and Esc or a right-click cancels. These keys only mean this
  while a move is on; the shortcut sheet lists them as their own layer.
- **Add Atom** (A): a click adds the current element on the plane through the
  cell's center facing you; Shift+click on an atom adds one bonded to it, away
  from its neighbors. E (or the element field in the tool chip) picks the
  element; while that field has focus, letters are typed there, not run as
  shortcuts. The viewport menu's **Add Atom Here** adds one where you clicked.
- **Replace** (E): a click replaces the atom; a drag paints, and the whole
  stroke is one undo step. With a selection, E opens the periodic table.
- **Measure** (M): click 2, 3 or 4 atoms for a distance, an angle or a
  dihedral; Shift+click chains distances. Ctrl+M measures the selection.
  Measurements are listed in the Measurements panel (copyable as TSV) and
  follow the atoms through edits.
- **Bond** (B): click two atoms to add or remove a bond from the drawing only.
- **Delete** (Delete, Backspace), **Set Coordinates…** (Ctrl+Shift+U, with
  expressions such as `1/3`), **Edit Cell…** (Ctrl+Shift+L, with a dashed
  preview of the new cell), the Alt+arrow nudges (Shift: ten times the step),
  the selection commands (Ctrl+Alt+E by element, Alt+Q equivalents, Alt+R within a
  distance, Ctrl+Alt+N bonded neighbors, Ctrl+] / Ctrl+[ grow and shrink, Alt+I by
  index such as `1-4, 9, 12-`, by fractional coordinate such as `z > 0.5`,
  and named selection sets).
- The views: 1, 2, 3 look along a, b, c and Shift+1/2/3 along a\*, b\*, c\*;
  Alt+U along any [u v w]; 5 switches orthographic and perspective; F frames
  the selection. The axis triad in the corner shows the cell axes.
- The **Atoms** panel lists every atom (only the rows in view are in the
  page, so 8000 atoms scroll smoothly), follows the selection both ways,
  edits a cell in place (Enter, Tab) and takes a pasted column of numbers as
  one Set Coordinates step. The **History** panel (Ctrl+Shift+H) lists the
  undo steps from "Opened MgO.cif"; a click goes back to any of them, and the
  later rows stay gray and redoable until the next edit.

### Sym and P1

Every document is in one of two modes, shown in the status bar and the
inspector:

- **Sym** (the default for a structure with more than one symmetry
  operation): the space group's operations are kept, and an edit acts on the
  whole orbit. Replacing Mg in MgO replaces all four Mg, deleting an atom
  deletes its equivalents, and moving one moves every image. A move off a
  special position offers **Keep** (the move projected onto what the site
  allows) or **Allow** (the orbit grows and the composition changes).
- **P1**: every atom on its own. The status bar still shows the space group
  of the structure as it is, in amber with "↓ from Fm-3m" when it dropped
  below the group the document had.

Switching from P1 back to Sym on a lowered structure asks whether to
**Symmetrize** first (spglib, one undo step) or lock the lower group. The
live space group during a drag comes from `/api/symmetry`: one request at a
time, 150 ms after the last change, and spglib runs on at most the workers
the [resource limits](#how-much-of-the-machine-it-uses) allow.

### Saving

- **Save** (Ctrl+S) writes a CIF in place, with every atom (P1 CIF). The
  first time for each file it asks "Overwrite MgO.cif? The original will be
  kept as MgO.cif.bak", with **Don't ask again for this file**
  (Settings › Editor › Confirm before overwriting turns the question off).
- The backup is `MgO.cif.bak`, next to the file. It is made only if there is
  none, and never replaced, so it is always the oldest original MACE saw.
  Settings › Editor › Keep a .bak turns it off.
- If the file changed on disk since it was opened (another program, `git
  checkout`), Save says so and offers **Overwrite**, **Save As…** or
  **Cancel**. This check runs even with the confirmation turned off.
- A `.out`, a `.d12`, a step of a run and a new structure always go through
  **Save As…** (Ctrl+Shift+S), which suggests `<name>_edited.cif`. It picks a
  folder inside the served directory (nowhere else) and a format: CIF with
  every atom, a symmetrized CIF (spglib's standard cell), POSCAR or extended
  XYZ (Settings › Files and data › Save format sets the first choice).
- **Save All** (Ctrl+Alt+S) saves every modified document and stops at the
  first that needs Save As. **Revert File** asks, then reopens the file and
  drops the unsaved changes and the undo history.
- The classic panel's **Save CIF** button opens the same Save As, with its
  name and CIF kind filled in.

A modified document has a dot on its tab, "● " in the window title,
"Modified" in the status bar and an accent on the Save button. Closing it or
the page asks first.

### Drafts

While a document is modified, its structure is kept as a draft in the
browser's local storage every Settings › Editor › Autosave drafts seconds
(10 by default; 0 turns it off), and at once when the page is hidden or
closed or the connection to the server is lost. Nothing is written into the
served directory. Opening the page again offers "Restore unsaved changes to
MgO.cif from 14:02? **Restore** / **Discard**". Drafts over 2 MB are skipped
with a note.

The browser keeps drafts per origin, that is per `http://127.0.0.1:<port>`:
a draft written under one port (or, over a tunnel, one local forwarded port)
is not offered under another. `mace gui` uses the same port every time
unless `--port` says otherwise, and it prints a reminder when it does.

### When the server goes away

- **The server restarted** (Ctrl+C and start again, a new token): every
  request is refused, so the page stops sending, keeps your edits in the tab,
  saves the drafts at once and shows "Lost the connection to the MACE server".
  Reopen the URL the server printed: the drafts are offered there. A fixed
  token (`MACE_GUI_TOKEN`, or `--token`) keeps the same URL across restarts,
  so the page carries on instead.
- **The server let go of a document's history** to stay within its memory
  budget: the page rebuilds the session from the structure it shows and
  says "This document's undo history was released to save memory"; the
  structure is kept and the earlier undo steps are gone.

### Memory

The undo history, the open sessions and the cache of page files share one
budget, Settings › Advanced › Cache limit (`cache_mb`, 512 MB by default, or
`MACE_GUI_CACHE_MB`): the history may use at most half of it and the page
files a quarter. Each document keeps up to Settings › Editor › Undo steps
(200) steps; one step of a 5000-atom structure takes about 125 KB. When the
budget is full the oldest undo steps of the least recently used documents go
first. Closing a tab lets the server drop its session at once.

### Export Image and New Structure

- **Export Image…** (Ctrl+Shift+X) saves a PNG of the view at the viewport's
  size, 2×, 4×, 1920×1080, 3000×3000 or a custom size (at most 8192 pixels a
  side, less if the graphics can't draw that much), with a transparent
  background, the axis triad, the measurement labels and the cell as options.
  The browser downloads it as `<name>_<view>.png` (`MgO_a.png` after viewing
  along a). **Copy Image to Clipboard** (Ctrl+Shift+C) copies the view, or
  says why the browser refused and offers Export Image.
  **Physical size** sets a printed width (inches or cm) at a resolution (72,
  150, 300, 600 or a custom DPI, Settings › Export resolution, 150 by default):
  3.5 in at 300 DPI is 1050 pixels wide. Every exported PNG says its DPI (a
  pHYs chunk), so it keeps its printed size in a document. SVG export has
  **Shading**: Flat (plain colors, the default) or Shaded (one radial gradient
  per element); fog and ambient occlusion are raster only. Plots have the same
  dialog under **Image…** (their PNG and SVG buttons still download at once).
- **New Structure…** (Alt+N) builds a cell from a b c α β γ, a space group by
  number or Hermann-Mauguin symbol (`225` or `F m -3 m`) with its setting,
  and the asymmetric unit, one "El x y z" line per site. It opens as an
  untitled document; Save goes to Save As.
- **Edit This Step** (in the Timeline panel and the command search) opens the
  step of a run on screen as a structure document.

### The classic panel

The first editor is still in the inspector under "More editing (classic)"
([below](#the-classic-editor-panel)): typed coordinates, supercell,
primitive cell, bond display limits, Clean and Preview deck. It acts on the
active document through the same requests, history and save path. The Build
menu's dialogs (supercell, slab, defects, transform, Clean) do the same with
more options.

## The classic editor panel

Open a structure with **Edit this structure** on the Structure tab (3D
crystals only), or expand **New cell from lattice parameters**. Every edit
runs on a session on the server, which keeps the full undo/redo history
(**Undo**/**Redo**, Ctrl+Z/Ctrl+Y).

- **Select** atoms by clicking them; Shift+click adds to the selection, Esc
  clears it.
- **Add** an atom at typed fractional or Cartesian coordinates (**From
  selection** fills in the center of the selected atoms). Type the element;
  the field suggests every symbol.
- **Delete**, **Replace** (with the typed element) and **Move** the selection.
  The arrow keys nudge it along x and y and PgUp/PgDn along z, by the step you
  set. They do nothing while a text field has focus.
- **Cell**: change a, b, c, α, β, γ, with or without scaling the atoms; make a
  supercell; reduce to the primitive cell; **Symmetrize** (spglib refine at the
  tolerance you set). The space group is recomputed after every edit.
- **Bond display** limits a pair's bonds to a length you choose. This changes
  the drawing only.
- **Symmetry-aware** editing captures the space group operations when you
  switch it on. An edit then acts on the whole orbit: adding an atom adds all
  its symmetry images, and deleting or replacing an atom does the same to all
  its equivalents. Moving an atom moves each equivalent atom by the image of the
  displacement. If a move takes an atom off a special position, its orbit grows
  (4a → 32f in Fm-3m) and the composition changes, so the editor asks first.
  Supercell, primitive, symmetrize, a cell change and Clean can change the
  operations (stretching cubic MgO to a ≠ b ≠ c leaves Fmmm), so the group is
  captured again afterwards, and the status line says so.
- **Save CIF** opens Save As (above) with the name and the CIF kind (a
  symmetrized CIF: spglib's refined cell, space group and asymmetric unit; or a
  P1 CIF with every atom) filled in, in the document's folder. It then selects
  that file in the Deck tab. **Preview deck** runs cif2d12 on the edited
  structure with the Deck tab's JSON, using the CIF name you typed, so the
  preview is the deck the saved CIF gives. A P1 CIF gives a P1 deck unless the
  JSON's `symmetry_handling` is SPGLIB.

`tests/test_gui_editor.py` saves symmetrized CIFs of MgO (225), Si (227), hBN
(194), the R-3c Ag2Br3 cell (hexagonal setting) and a P21/c cell, writes their
decks with cif2d12, expands each deck again, and checks the space group,
volume and neighbor shells against the edited structure.

### Clean geometry

**Clean** is a rough pre-relaxation for tidying hand-placed atoms. It is not
physics, and its result is a starting geometry for CRYSTAL, not a relaxed
structure. It is a periodic spring model in ase (`mace/gui/clean.py`):

- the bonds are the ones the viewer draws, by the same rule
  (`structures.bond_pairs`, the server's copy of the browser's bond rule):
  pairs closer than 1.15 × the sum of their bond radii and farther apart than
  0.1 Å. Metal-metal pairs are bonded in a structure of metals only (a metal or
  an alloy), or with **Bonds between metals** on. The document's own rule from
  **Build › Calculate Bonds…** (tolerance, shortest bond, pair rules) applies
  too. A bond stretched past the tolerance is not drawn and is not held either:
  raise the tolerance in Calculate Bonds… first, or pick another **Bonds held**
  in the panel (remembered for the next Clean, the classic Clean too):
  **As shown** (the default, the rule above), **Lenient** (1.2 × the sum of the
  bond radii at the default criteria, metal-metal pairs again in a structure of
  metals only: it also holds bonds stretched a little past what the view draws,
  as Clean did before 1.1.6) or **None** (no bond: close atoms only push apart,
  which also runs where the others have nothing to clean). `mace preopt
  --bonds shown|lenient|none` is the same choice. Each bond relaxes toward the
  sum of the covalent radii;
- other pairs closer than 1.8 × the sum of the radii repel softly. Bond springs
  alone leave zero-energy motions (6 in conventional Si, 9 in conventional MgO
  or NaCl, by the rank of the rigidity matrix), and the repulsion removes them;
- FIRE moves the atoms; **keep the space group** adds ase's `FixSymmetry`.
  **Relax the cell** is off by default: covalent radii underestimate ionic
  bonds (Na-Cl 2.68 Å against 2.82 Å measured), so an ionic cell shrinks by
  about 5 %. The cell responds to the bond springs only.

**Build › Clean Geometry…** (Ctrl+Alt+G, and the classic panel's **Clean**)
opens a panel that runs Clean in the process pool and shows a **preview**:
the cleaned structure as a ghost, an arrow per moved atom, and a summary
(maximum and RMS shift, the space group after). Nothing changes until
**Accept**, which is one History entry "Clean Geometry"; **Discard** (or Esc,
or Ctrl+Z while the preview shows) adds nothing. While a preview shows, the
document can't be edited (the status bar says why); switching tabs keeps the
proposal. The time limit stops it with a "not converged" note, and Cancel
stops a long one within 2.5 s. A lone atom, or a structure whose bond rule
bonds no pair, has no spring bonds, and Clean says there is nothing to clean. The tests check that distorted MgO, Si
and NaCl cells return to their bond lengths, keep their space group with
FixSymmetry, and that an ideal cell does not move.

**Method** picks how Clean moves the atoms: **Springs** (the default, the model
above, with the live preview), **GFN-FF** (the external `xtb` program, 6.7 or
newer), an **ML potential** (MACE-MPA-0 from ACEsuit's mace-torch, a
third-party model unrelated to this MACE, or CHGNet 0.3.0, run in their own
Python environment) or **Auto**. A method that is not installed is grayed out
with the commands that set it up; **Recheck** looks again, and **ML
Potentials…** opens Settings › ML potentials, where the ML environment is
installed and weights are downloaded when you ask. GFN-FF, ML and Auto take a minute or
more, so they do not preview on every change: **Run** starts one (a second Run
replaces it; closing the dialog cancels it), and Run Longer goes up to 600 s.
The dialog also has the **Charge** (prefilled from the document; required for a
document opened from a CHARGED/CHEMOD/DOPING deck), **Symmetrize first** (off;
it can raise the space group, which CRYSTAL then keeps) and, under Advanced,
fmax and steps. The readout says which method ran, why another was refused or
rejected (the spring result is then kept), that the result is a starting
geometry for a CRYSTAL OPT, and offers **Copy citation**. A document from a
deck whose records edit its geometry (SUPERCEL, ATOMREMO, ...) is the host
geometry: Clean offers the springs only and says so. The same pipeline is the
CLI's `mace preopt`; see [preopt.md](preopt.md) for the methods, their limits,
licenses and citations.
Any of them would be an optional ase calculator.

## Building structures

The **Build** menu (and the Inspector's Cell section, the atom menu, and the
classic panel's Make/Primitive/Symmetrize buttons) opens a dialog with a live
preview in the viewport. Each dialog has a **Result** control: **New tab**
opens the result as a new, unsaved document named for what it is
(`MgO_2x2x2`, `MgO_001_5L`, `MgO_2x2x2_V_O_4b`) that remembers where it came
from; **This structure** applies it to the open document as one History entry.
The choice is remembered per dialog. A slab is always a new tab (it never
replaces its bulk).

- **Supercell…** (Ctrl+Alt+P): n1 × n2 × n3, or a 3 × 3 integer matrix with
  its determinant (entries within ±30), and presets (2×2×2, √2×√2×1 R45°, and
  √3×√3 R30° for hexagonal cells). The preview draws the new cell dashed.
- **Slab from Bulk…** (Ctrl+Alt+H): (hkl), the number of **atomic layers**
  (planes of atoms within 0.15 Å along the surface normal; ase's own `layers`
  counts surface cells), the termination (told apart by the composition of the
  top plane), and the vacuum, which is the total gap between periodic images.
  Terminations are named by the top plane's reduced formula ("O-terminated",
  "MgO-terminated"). While the dialog is open the slab preview covers the
  viewport, at its standard view with the vacuum running up, framed in the part
  the dialog leaves free; the bulk's own view is unchanged. The vacuum only matters
  for a 3D periodic cell: CRYSTAL's SLAB keyword (2D) ignores it, and the
  dialog says so.
- **Point Defect…**: vacancy, substitution, interstitial and antisite, on a
  site picked in the list (or from an atom's context menu), in a supercell
  searched so the defect's periodic images are at least Settings › Defect
  image distance apart (10 Å by default). Interstitials are found on a grid
  and snapped to their site symmetry, with Wyckoff letters. "One structure per
  site" makes one document per inequivalent site. A charge is kept with the
  document as a hint for Setup.
- **Transform Cell…**: primitive, conventional, or Niggli-reduced.
- **Symmetrize…**: the space group found at each of 13 tolerances from 10⁻⁵
  to 0.1 Å, and spglib's refined structure at the one picked, with arrows
  showing how far each atom moves. The result is the group's conventional
  cell: on a supercell that is smaller (MgO 2×2×2 becomes the 8-atom cell),
  and the dialog warns before you accept.

A preview that reaches past the view (a supercell's ghost atoms, a defect's
supercell outline) zooms the view out until it fits beside the dialog; Cancel
puts the view back, unless you moved the camera meanwhile.

**Limits.** In place stops at 5000 atoms (the editing limit). A new tab over
5000 atoms opens view-only, with the reason; over 20,000 atoms its space
group is computed on request (**Compute now**, a task). Builds stop at
250,000 atoms and refuse before building. Saving a supercell as a
**symmetrized** CIF is refused: spglib's refined cell would be the space
group's smaller cell, so Save As says to save P1, which keeps every atom.

## Files

- **Drag and drop.** Drop CIF, POSCAR/CONTCAR, XYZ, extXYZ, `.d12` or `.out`
  files anywhere on the window (up to 8 at a time, 50 MB each). Each opens as
  an unsaved document with its own name ("MgO.cif (dropped)" when the folder
  already has that name), marked "Not saved in the folder yet"; the server
  reads it from a temporary file it deletes at once, and nothing is written
  into the folder until you Save As. Folders, other extensions and bigger
  files are refused with a message.
- **Import from Clipboard** (Ctrl+Shift+V, or File menu): the text of a
  POSCAR, CIF, XYZ or extXYZ, with its format detected as you type ("Looks
  like POSCAR, 8 atoms") and, for a plain XYZ, a box with 10 Å of vacuum or a
  molecule.
- **Copy, Paste, Cut and Duplicate.** Ctrl+C copies the selected atoms;
  Ctrl+V pastes them at the same fractional coordinates as one History entry
  ("Paste 3 Atoms") and selects them. In Sym mode each pasted atom brings its
  orbit, and more than three times the copied count asks first. Ctrl+X is
  Copy plus one Delete; Ctrl+D duplicates the selection one step to the right
  and grabs the copies (Esc leaves them there). Other text pasted into the
  window opens Import from Clipboard.
- **Export Structure** (File menu) downloads CIF (P1 or symmetrized), POSCAR,
  XYZ or extXYZ to this computer (over a tunnel: the laptop); **Save As**
  writes into the served folder. Export never writes into the folder.
- **Recent files** are kept per folder: File › Open Recent, the welcome
  screen and the explorer's collapsible RECENT section. Right-click one to
  open, pin (pinned files stay on top and don't count toward the limit),
  remove it, or copy its path. A file that is gone shows "Missing". Clear
  Recently Opened can be undone for 8 s. Built and dropped documents join the
  list when they are saved.
- **Changed on disk.** When the window regains focus and an open file changed,
  a banner offers Reload or Ignore.
- **Big outputs.** A CRYSTAL `.out` is indexed once (byte offsets of each
  geometry block, the energies and the OPT status, read by the same block
  parser); the Optimization view opens from a summary (energies, labels, the
  first and last steps) and loads other steps as you scrub. A 34 MB run opens
  in under 2.5 s and sends under 300 KB. The explorer shows each structure
  file's formula and atom count, read in the pool once the page is idle.

### Open Folder

**File › Open Folder…** (also in the command search, and the folder button in
the Explorer's header; no default key) serves another folder in this window
without restarting the server. Pick a root (Home, Scratch), then a folder, or
paste a path, and press Open; the window reloads into the new folder. Modified
documents ask first (Save All, Don't Save keeps them as drafts in this
browser, Cancel); drafts and recent files are kept per folder, so going back
offers them again. Other tabs of the same server follow with a notice. While
GUI tasks run, Open Folder waits ("Cancel or wait for the running tasks
first").

- **Which folders.** The roots are your home folder and your scratch folder:
  `$SCRATCH`, `$WORK` or `$PSCRATCH` when set, else the first of
  `/scratch/$USER`, `/mnt/scratch/$USER`, `/global/scratch/$USER` or
  `/work/$USER` that exists and is yours. `--root DIR` (repeatable) or
  `MACE_GUI_ROOTS` (a `:`-separated list) replace them; `--root none` (or
  `MACE_GUI_ROOTS=` set to nothing) turns Open Folder off. The folder you
  started with (`--dir`) is always allowed. A symbolic link that leads out of
  the roots is refused.
- **Never hidden or credential folders.** A folder with a hidden part below
  the root (`~/.config`, `~/proj/.git`) is refused: "Hidden folders cannot be
  opened here (they hold settings and credentials). Start mace gui with --dir
  to serve one on purpose." So is a folder holding `.ssh`, `.gnupg`, `.aws`,
  `.kube`, `.docker`, `.config/gh`, `.netrc`, `.git-credentials` or `.pgpass`
  (your home folder itself): "Open a project folder inside your home, not your
  home folder itself." Even when you serve your home folder with `--dir`, MACE
  GUI never reads those credential files or the other secret files listed in
  [Security](#security), and it never writes into a hidden folder.

### What each file opens as

The explorer lists every kind of file the CLI reads or writes, and decides
what an output or a log is from its content, not its name: the first 256 KB
and the last 64 KB are read in the pool (once per file version), and the
calculation type comes from the CLI's own `determine_completed_subtype`.
Each row's chip shows it (FREQ, OPT, SP, BAND, DOSS, CHG+POT, TRANSP, log, ...),
and the **Logs & data** filter gathers job logs, scripts, restart data and
wave functions.

| File | Opens as |
|---|---|
| CIF, `.d12` (3D) | a structure document |
| `.xyz`, `.extxyz`, POSCAR/CONTCAR/`.vasp`, `.mol`, `.sdf`, `.pdb`, `.gen`, `fort.34` | a structure (a file with no cell is a molecule, drawn with no cell, view only) |
| CRYSTAL `.out` (crystal, Pcrystal), also when named `.o` or `.log` | OPT: the optimization trajectory; SP: the structure; FREQ: **Normal modes** (Settings › Open FREQ outputs as: Structure brings back the structure); molecules too (their Angstrom geometry tables) |
| a CRYSTAL `.out` that stopped before printing a geometry | its text, with a Problems row saying so |
| properties `.out` (Pproperties: BAND, DOSS, CHARGE+POTENTIAL, TRANSPORT, ORBITALS) | a properties document: the run's type and status, its data files (each opens its plot or volume), and the structure from the SCF output beside it |
| SLURM job logs (`<job>-<id>.o`, `<job>.o<id>`, array logs `<tag>-<A>_<T>.o`, `slurm-<id>.out`, `.e`, `.err`) | read-only text with a header naming the job and its calculation (**Open**); a `.e`/`.err` adds its first line to Problems. A log is never parsed for geometry |
| `.d3` | text |
| `*.BAND.DAT`, `*.DOSS.DAT`, `*_IRSPEC.DAT`, `*_RAMSPEC.DAT` | plots |
| `.molden` (CRYSTAL23 ORBITALS: `<job>_orb.molden`, `<job>_orb_<n>_K<ijk>_real.molden`, `_complex.molden`) | the Volume document with the orbital picker (below); the files are never changed |
| cube files: `*_DENS.CUBE`, `*_POT.CUBE`, `*_SPIN.CUBE`, `.cube`, and CRYSTAL's own names `DENS_CUBE.DAT`, `POT_CUBE.DAT`, `SPIN_CUBE.DAT`, `<input>_WAN_CUBE.DAT` | the Volume document (below) |
| `fort.25` / `.f25` | a 2D map (ECHG charge density, POTM potential: the MAPN block) opens as a contour plot of the A-B-C plane with its corners labeled, a log scale for a positive map, and the spin density as a second map; the colors span the 1st to 99th percentile of the map (symmetric about zero for a signed map), so the spikes at the nuclei do not take the whole scale (Plot options › **Color range**: Auto, Full min to max, or Custom from/to); a POTM or CLAS potential is drawn in Settings › Viewport › Potential colors (chemistry, the default: red = negative); each map is named by the properties output of the same name beside it (CRYSTAL's fort.25 does not say whether a block is a density or a potential), else "2D map"; a BAND or DOSS fort.25 plots like the `BAND.DAT` or `DOSS.DAT` of its run (see Plot viewers, 1.1.6, below) |
| BOLTZTRA `*.SEEBECK.dat`, `*.SIGMA.dat`, `*.SIGMAS.dat`, `*.KAPPA.dat`, `*.TDF.dat` (`.DAT` too) | a transport plot of any quantity the run's files beside it give |
| `FREQINFO.DAT`, `BORN.DAT`, `HESSOPT.DAT`, `OPTINFO.DAT`, `SCFOUT.LOG`, ...; `.f9`, `.f98` | a note on what the file is (MACE uses wave functions for GUESSP restarts and properties runs), with **Reveal the run's output**; never read past 4 KB |
| `.sh` job scripts | text |

Ctrl-click (or Shift-click a range of) outputs to pick several without
opening them; right-click › **Compare Convergence** plots each run's energy
per atom at every printed geometry, above the lowest final energy of the
set, with each run's OPT END line in the legend.

### Notifications

Information and success notices close after Settings › Info notices stay (6 s
by default; a notice under the pointer or the keyboard stays). Warnings and
errors stay until you close them (the ×, Esc on the notice, or Dismiss All).
The same notice shown again counts up ("×3") instead of stacking, and more
than three fold into "+N more". Nothing is lost: every notice of the session
is in **Problems › All Notifications** (the bell in the status bar shows how
many are new), and warnings and errors also appear under Problems.

## Calculation Setup

**Calculation › Calculation Setup…** (Ctrl+Shift+R) opens the settings of the
structure in front: `mace convert` (cif2d12) for a CIF or a built structure,
`mace opt2d12` for a follow-up from a CRYSTAL `.out` or a `.d12` deck, and
`mace opt2d3` for BAND, DOSS and the other properties runs. The calculation
types come in two groups, in rows of three, each numbered as its CLI menu
numbers them: **Regular Calculations** (opt2d12's "Select calculation type"
menu: 1 SP, 2 OPT, 3 FREQ, 4 ELASTIC, 5 EOS, 6 QHA, 7 CPHF, 8 Phonopy; see [Phonons, mechanical
properties and the relaxation
time](#phonons-mechanical-properties-and-the-relaxation-time-116)) above
**Properties Calculations** (opt2d3's menu: 1 BAND, 2 DOSS, 3 TRANSPORT, 4
CHARGE, 5 POTENTIAL, 6 CHARGE+POTENTIAL, 7 ORBITALS); the arrow keys move
through them. A type the file cannot have is grayed, with the reason as its
tooltip: everything but SP, OPT and FREQ needs a finished OPT or SP output
(not a CIF, a deck, a FREQ run or an optimization without `OPT END -
CONVERGED`), as the workflow's step prerequisites say. **Calculation › Regular
Calculations** and **Properties Calculations** list the same types, numbers
and reasons, and open Setup at the one you pick. Right-click a CIF, an `.out`
or a `.d12` in the Explorer: **Set Up Calculation**; or open the file (Ctrl+O)
with Setup open.

- **No CIF needed: start from a finished run or a deck.** Step 1 names the
  parent and the command: an `.out` is `mace opt2d12 --out-file` (the run's
  geometry, the settings of the `.d12` beside it) or `mace opt2d3 -i`; a
  `.d12` is `mace opt2d12 --d12-file` (the deck's own geometry records and
  settings; the new deck is named `<deck>_<calc>_<functional>.d12`). The
  fields say "from parent .out" or "from parent .d12". The preview, the
  saved settings and the written deck are the CLI's own, byte for byte.
  - A deck offers SP, OPT and FREQ only: BAND, DOSS and the other properties
    read a finished run's wavefunction, so set them up from its `.out`.
  - SLAB, POLYMER and MOLECULE outputs keep their dimensionality. The GUI
    draws no SLAB, POLYMER or MOLECULE deck (it names the run's `.out`
    instead), so Set Up Calculation on such a deck says the same; Batch
    Setup's *Decks* filter takes them (`mace opt2d12 --d12-file` writes them).
  - A deck with no geometry of its own (EXTERNAL or DLVINPUT read fort.34),
    or one that edits its geometry after the atoms (SUPERCEL, ATOMSUBS,
    SLABCUT, ...), shows the CLI's refusal: set up from its run's `.out`,
    which holds the structure CRYSTAL built.
  - An unfinished or unconverged optimization: the deck is the one the CLI
    writes (the run's starting geometry, as `mace opt2d12` does) and step 1
    shows the CLI's warning under the parent's name.
  - An edited `.d12` (unsaved changes) is set up as a new structure
    (cif2d12), as before.

- **The CLI is the source of the defaults.** The form lists the CLI's own
  questions, its menus and its defaults, read from the CLI at the moment you
  open Setup. A follow-up starts from the parent's settings (functional, basis,
  SPIN, tolerances), as `opt2d12` does on the command line, and its fields say
  "from parent .out" (or ".d12"); opt2d3's fields start from opt2d3's own
  defaults and say "default". TRANSPORT, CHARGE, POTENTIAL and
  CHARGE+POTENTIAL list every question opt2d3 asks under "Other questions"
  (a menu by its heading, such as "Charge density type"); a question asked
  in both CHARGE+POTENTIAL blocks is a row for each, named after its block
  ("(charge density)", "(electrostatic potential)"), and a change applies to
  that block only. Each changed field says what it was and has a Reset. Fields the CLI
  ignores for a choice are read-only and say why ("auto from the cell: SHRINK
  12 24" for cif2d12's mesh).
- **Preview While Editing** (`setupAutoPreview`, on by default) previews after
  each change; Ctrl+Enter previews by hand. A newer edit cancels the older
  preview. **Run Preflight on Preview** (`runPreflightOnPreview`, off by
  default) adds the TESTPDIM check to each preview.
- **opt2d3 and the wavefunction.** A properties deck needs the parent's `.f9`.
  The preview stages an empty stand-in with the CLI's chosen name (it never
  copies a big `.f9`) and Problems says so; the printed command copies the
  real one when you run it. Without an `.f9`, Problems shows the CLI's
  refusal.
- **seekpath.** Band paths come from SeeK-path, as on the command line; there is
  no static path unless you pick the CLI's own static option.
- **A dirty `.out`.** opt2d12 reads the geometry from the output file, not your
  edits: Setup asks whether to use the unedited parent or to Save As CIF and
  set up a new calculation. The same applies to a trajectory frame that is not
  the last.
- **P1 CIFs.** When a CIF is P1 but spglib finds more symmetry at 1e-3 Å (a
  structure built in the GUI and saved with every atom), Problems says so and
  offers **Save Symmetrized Copy…**. cif2d12 keeps P1 otherwise; nothing changes
  silently.
- **Saving.** *Own folder* writes the JSON and the command in a new folder,
  *In place* writes them beside the input, and *Write decks now* also runs the
  printed command (after checking the disk space; on a tmpfs it warns about
  RAM).
- **Undo.** Setup has its own history (the History panel's **Settings** chip);
  Reset All Settings goes back to the CLI's defaults in one undo step.
- **Copy CLI Command** (Ctrl+Alt+C) copies the printed command.
- **Submit…** (the sidebar, under Preview) writes the deck these settings
  make (Save with *Write decks now*, under Save's own rules) unless Setup
  already wrote it, then opens the [Submit dialog](#submit-and-jobs) on it:
  nothing is submitted until the dialog's own Submit. While MACE GUI listens
  on the network it says that submitting is off and writes nothing.
- **Batch Prepare + Submit…** opens [Batch Setup](#batch-setup) on the file's
  folder with the same kind of file (CIFs, runs needing this follow-up, or
  decks), this calculation, this file as the representative and the fields
  you changed as the settings for all files; its step 4 writes the decks and
  opens Submit on them. A hand-edited settings file and the k-point mesh
  stay in Setup.
- **Spin-orbit coupling (SOC).** For a single point (SP) from a CIF, an
  `.out` or a `.d12`, Settings has a **Spin-orbit coupling** section.
  Switching it on makes the deck a two-component SCF with SOC (TWOCOMPON).
  Choose the ECP route: INPSOC (the default), CRYSTAL's internal STUTSC
  (grayed out, with the CLI's reason, for Hf, whose internal STUTSC is not
  the published potential) or the official pob-TZVP-rev2-SOC basis. For a
  follow-up from a run, choose the starting guess: from scratch, GUESSPNOSO
  from a spin-polarized parent run without symmetry, or a GUESSPSO restart
  from a two-component parent. Only the guesses the parent allows are
  offered, each other one with the CLI's reason (a `.d12` parent has no
  wavefunction, so it starts from scratch). A **SMEAR** width for metals is
  optional (empty = none; MACE never adds SMEAR by itself): **Use 0.005**
  fills in the width fcc Au converged with on HPCC, and is suggested more
  strongly when the CLI finds the parent looks metallic.
  These choices are the keys `"soc"`, `"soc_ecp"`, `"soc_basis"`,
  `"soc_guess"` and `"smear"` of the CLI's own settings file, so the deck is
  the one `mace convert` or `mace opt2d12` writes from that file, byte for
  byte (see [SOC decks](cli.md#spin-orbit-coupling-soc-decks)). Each choice
  has the same origin badge and **Reset** as the other rows. The section
  shows the CLI's SOC warnings (all-electron atoms that lose their
  spin-orbit splitting, with the measured InAs numbers; f elements; metals
  and the k-mesh), its refusal word for word, and the lines saying what the
  CLI changed in the deck. With **Charged system** switches too, the CLI
  refuses the deck when they would add CHARGED, DOPING or SPINLOCK (CRYSTAL's
  two-component SCF does not list them), and the section shows its reason.
  A SOC deck from a CIF needs an external basis
  (for example `full.basis.triplezeta`) and a functional the two-component
  SCF runs (PBE0, PBE, B3LYP and others); with the defaults (HSE06 and the
  internal POB-TZVP-REV2) the CLI refuses and says why. For OPT, FREQ and
  properties runs the section is grayed out with the CLI's reason (CRYSTAL23
  has no optimization or frequencies in a two-component run, manual p. 166;
  properties of a SOC run are limited to what manual sec. 6.4 lists). A
  follow-up of a SOC run is a SOC deck too, as `opt2d12` writes it ("from
  parent"). With a GUESSPNOSO or GUESSPSO restart, **Write decks now** also
  copies the parent's `.f9` as `<deck>.f20`: it needs room for the copy and
  never replaces a different `.f20`.
- **Run Preflight** without a serial `crystal` on the machine runs MACE's
  structure check alone and says so; a failure there reads "Structure check
  failed (CRYSTAL not run)", never "REJECTED", which is kept for CRYSTAL's own
  TESTPDIM rejection.
- **Charged system** (cif2d12 and opt2d12; shut until you give a switch):
  the charged-system switches of `mace convert` and `mace opt2d12`.
  **Find ions** is `--ion-charges auto`, then **Net charge**
  (`--net-charge`), **Site states** (`--site-states`, the JSON list itself,
  for example `[{"atom": 1, "shell_charges": [2, 0, 0, 0, 0]}]`; Setup does
  not read it from a file), **CHEMOD** (`--chemod auto|yes|no`), **Fragment
  bond scale** (`--fragment-bond-scale`, with Find ions only), **SPINLOCK**
  (`--spinlock`) and **SPINLOCK cycles** (`--spinlock-cycles`). Each row
  names its switch. Setup passes the switches as they are, on the command
  it runs and on the command it prints and saves, so the deck is the one the
  CLI writes when you type the same switches. The CLI checks what they mean
  and refuses in its own words, for example "site_states set the starting
  guess only; give the net charge too" or a SPINLOCK that does not fit the
  electron count. The refusal shows in Problems and no deck is written.
  opt2d3 (BAND, DOSS and the other properties) takes none of these switches.
  The defect builder's **Apply** for its charge hint sets the net charge
  here.

## Submit and Jobs

**Calculation › Submit…** (or right-click a `.d12`/`.d3`, or a folder header
in the Explorer: **Submit Folder…**) submits decks with `mace submit`, and
**Run on This Node…** runs them here with `mace run` (a dev node, no SLURM).
Neither has a default shortcut, and nothing is submitted until you click the
dialog's own **Submit** button: the dialog opens with **Cancel** focused, and
Enter never submits.

- **The dialog runs the real command first, in a copy of the folder.** It
  shows the exact command a shell user would type (`cd <folder> && mace
  submit ...`, with `echo N |` for a folder of D12 and D3 decks), the
  resources read from the job scripts (cores, memory, walltime, account,
  partition, `-N`, exclusions; a reused `<deck>.sh` says so), every file it
  will write, and what the jobs then write (`<deck>.out`, `<deck>.f9`). In
  the preview, read-only SLURM queries (node types for `--exclude-type`, the
  job-array limit, MaxSubmitJobs) reach your cluster; every state change
  (`sbatch`, `scancel`, `scontrol release`) is only recorded. Without SLURM
  the preview says it used default cluster limits.
- **Options** map to `mace submit` flags: *Submit as job arrays* and *Max
  running at once* (`--array --max-running N`, a folder only: Batch Setup
  collects decks into one), *Track in materials.db* (`--track`, enables
  `mace status`), *Then run the rest of a workflow* (`--progress TEMPLATE`;
  only templates that contain your decks' calculation type, never
  `interactive` or `custom`; each deck then runs in its workflow's step
  folder, so a finished output beside it is not overwritten but starts its
  next step, and *Regenerate job scripts* and *Overwrite finished results*
  do not apply), *Walltime* (`--walltime`, checked by the CLI),
  *Exclude* none, these nodes or a node type (`--exclude-nodes` /
  `--exclude-type`), and *Regenerate job scripts* (`--overwrite-sh`). Up to
  20 decks the preview updates as you change them; above that, **Update
  preview**.
- **Conflicts need their own tick.** A finished `<deck>.out` or a `<deck>.f9`
  the job would replace: *Overwrite finished results*. A deck already queued
  (your squeue, or a submitted/pending/running row of the folder's
  databases): *Submit again although ... already queued*. An account that is
  not one of your SLURM accounts: *Submit with account ... anyway*. The
  server checks the ticks again, and refuses a submit whose preview is used,
  older than 15 minutes, or whose decks, job scripts, outputs, queue, site
  settings or CRYSTAL module changed since (409, "preview again"; the dialog
  then offers **Update preview**). While a submit runs, the dialog cannot be
  closed.
- **Site** shows the SLURM account and the CRYSTAL module the job scripts
  load, and where each comes from (see [Site settings](cli.md#site-settings-mace-site-sitetoml)),
  then one line each for the job size (Resources), the MPI launch, the
  scratch order, QOS, constraint and the CRYSTAL executables, with the
  source of each value (the values `mace site --json` prints). **Check** runs
  `mace site check` on the machine the server runs on, with the environment
  a submit gets, and lists each item with its fix; it asks `sbatch
  --test-only` and submits nothing. The GUI never edits `site.toml`.
- **The real run** is the same argv, in the same folder, with the environment
  `mace gui` was started with: no `nice`, no thread limits, none of the GUI's
  own variables, so `sbatch` gives the job what a shell gives it. Inside an
  allocation (`salloc`), the allocation's `SLURM_*` variables are left out.
  A submit cannot be stopped halfway ("Submitting... N s"); stopping the
  server waits for it (a second Ctrl+C stops waiting; the submit finishes on
  its own). Afterwards the job scripts are compared with the preview's; a
  difference (the cluster changed in between) is reported.
- **Run on This Node** previews with `mace run --dry-run` on the selected
  decks; where it cannot run (a login node, inside a SLURM job, no CRYSTAL)
  the reason is `mace run`'s own. The run is a task in Tasks; Cancel sends
  one SIGTERM and waits for `mace run` to stop its job (up to 25 s) before
  anything is killed. The run stops when MACE GUI stops.
- **Submitting, canceling and recovering jobs are off while MACE GUI listens
  on the network** (`--insecure-network`).

**Jobs** (bottom panel, **View › Jobs**, or the status bar's "Jobs: 2 running
· 1 pending") lists your own SLURM jobs: *This folder* (their working folder
is inside the served folder) or *All my jobs*, job arrays grouped as "Array
A: 1 running, 2 pending". Each row has **Open Output** (the job's
`<deck>.out`), **Show Log** and **Cancel**; *Finished (last 2 days)* lists
ended jobs from `sacct`. Above the table a line shows your queue budget
("Queue budget: 120 of 940 in use (limit 1040, reserve 100) · 15 held"), and
*Held* lists the jobs MACE holds for a free slot (calculation, material,
reason, since when, folder; *All my jobs* lists those of every workflow);
they are not in SLURM yet, so they have no job id. The `mace-drain` job,
when MACE runs one, is one of your SLURM jobs (see `mace queue`). It asks SLURM (`squeue`) every
`cluster.pollSeconds` (Settings › Cluster › Job poll interval, 60 s, 30 to
300) while it is shown or while a job this session submitted is pending or
running, through the same cached answer as the Cluster view and `mace
cluster` (Refresh asks again at most once per 15 s); a job that leaves the
queue gives one toast with its `sacct` state (and a Problems entry unless it
completed), once, however many views noticed it.
**Cancel** runs `scancel <id>` after a confirmation that names the command,
for your own single jobs and array tasks only: a whole array (whether MACE
submitted it or not), a task range, another user's job and the job MACE GUI
runs in are refused with the reason.
**Recover…** (a failed tracked job under *Finished*) and **Recover Failed…**
(every failed tracked calculation in the folder) run `mace recover`, the CLI
itself: first `mace recover --dry-run`, whose text (the table of error class,
proposed fix and retries, the diff of every file a fix would write and the
`sbatch` command) the confirmation shows with the command it will run; then,
once confirmed, that command (`mace recover ... --yes`). Jobs that ended more
than `MACE_RECOVER_MAX_AGE_DAYS` (default 3) days ago get a second question
(`--include-stale`). Like Submit and Cancel, Recover is off while MACE GUI
listens on the network.

**Runtime estimate.** Once the preview is in, the dialog shows `mace
cluster estimate DECK --cores N` for the first three `.d12` decks: the
median and the 10 to 90 % range from similar finished runs in the deck's
folder, or why it cannot say (no similar runs, an unknown core count, a
history too scattered to trust). **Details** shows the CLI's own text and
the command. The estimate is advisory: it changes nothing in the job and
never holds the **Submit** button back. While MACE GUI listens on the
network it says it cannot estimate.

## Batch Setup

**Calculation › Batch Setup…** (or right-click a folder header: **Batch
Setup…**) opens a "Batch: <folder>" tab that sets up many decks at once, in
four steps:

1. **Pick files:** the folder's *CIF files*, *Completed runs needing a
   follow-up* (SP, OPT, FREQ, BAND, DOSS, TRANSPORT, CHARGE, POTENTIAL,
   ORBITALS),
   *Decks (.d12) as parents* (SP, OPT or FREQ through `mace opt2d12
   --d12-file`; an EXTERNAL deck or one that edits its geometry says why
   not), or *Failed runs*, optionally with subfolders. A file that is not offered
   says why: the follow-up is already beside it, in an earlier
   `batch_*/batch.json`, in the folder's or a workflow's database, or next
   in a workflow plan; FREQ needs a converged OPT; properties need the
   wavefunction; a name with spaces or shell characters cannot be submitted.
   Failed runs show what can be done; a tracked one shows the `mace recover`
   command (it lists every failed tracked calculation of that database with
   its fix and, once confirmed, applies the fixes and resubmits them; Jobs ›
   **Recover Failed…** does the same from the GUI).
2. **Settings for all files:** the Setup form of one representative file.
   Only the fields you change are applied, to each file on its own through
   the CLI, so every follow-up keeps its own parent's functional, basis and
   settings otherwise. For CIFs and SCF follow-ups, **Charged system** takes
   Calculation Setup's charged-system switches except `--site-states`, which
   names atoms of one structure. Each command of the batch carries them, as
   `commands.sh` and `batch.json` show, and the CLI checks each file on its
   own.
   For an SP batch, the **Spin-orbit coupling** choices of Calculation Setup
   (SOC on, ECP route, starting guess, SMEAR) apply to every file. They are
   written into each file's settings file as the same keys, so `bash
   commands.sh` re-creates the SOC decks; Preview shows the CLI's refusal
   of a file (for example STUTSC for Hf) and its SOC warnings. STUTSC is
   named "refused for Hf in 1 of 3 files", with the CLI's reason, when a
   selected file holds an element whose internal STUTSC the CLI refuses
   (grayed out when every file does). A file's **Override…** has the SOC
   switch and ECP route for that file alone (INPSOC for the Hf file of a
   STUTSC batch, say), also for a file the CLI refused. **Batch
   Prepare + Submit…** in Calculation Setup brings the charged-system
   switches and the SOC choices along.
3. **Preview:** each file's deck, how many lines differ from its default,
   **Diff**, and **Override...** for one file only. Files the CLI refuses are
   shown with its words and left out.
4. **Write and submit:** a folder `batch_<calc>_<date>/` (name it as you
   like) with `options/` (the CLI's settings files, one per CIF group or per
   parent, and `template.json`), `inputs/` (cif2d12's CIF groups), `decks/`,
   `commands.sh` and `batch.json`. `bash commands.sh` in that folder's parent
   re-creates every deck with the same commands (`mace convert --batch`,
   `mace opt2d12 --out-file ... --yes` or `--d12-file ... --yes`, `mace
   opt2d3 ...`). **Submit...** opens the Submit
   dialog on `batch_*/decks`, with job arrays ticked for two or more decks; it
   is submitted from the folder that holds the batch, so tracked rows join
   the parents' `materials.db` there (`mace status` in that folder lists
   them; a `--progress` workflow keeps its records in
   `batch_*/decks/.mace_context_<id>/`, so run `mace status` there for it).

## The Database document

Opening a `materials.db` (or **Analysis › Open Materials Database**) shows a
table that the server sorts, filters and pages: the page holds only the rows
on screen, so 100,000 materials scroll at the display's frame rate (the bench
measures 13 ms to the first page and at most 45 row nodes). The filter row
searches material ids and formulas and narrows by dimensionality, calculation
type and status, a space-group range, and a property the material has. Selecting a row shows its calculations and properties in
the inspector, with links that open the input and output files; a property
that several of the material's calculations have names its calculation, as
in `cpu_time_total_s (OPT)`. **Export
Table** (CSV, JSON, Excel, LaTeX or HTML) writes the filtered, sorted rows
with the same code as `mace database --action export`: for the same rows the
file is the CLI's, byte for byte apart from the export time; a JSON, Excel,
LaTeX or HTML export names the filters in its metadata. The table keyboard: arrows, Page Up/Down, Home/End over the whole
result, Enter to open.

The Database view reads the database; Create or Update Database from a
Folder writes to it, with a backup first. The view never waits behind a job:
while a completion callback records a result, the status line says "Database
busy (a job is recording its result) - retrying" and keeps the rows it shows;
a database left mid-write by a killed job says so until the next MACE command
that writes it repairs it. A database in a folder the server cannot write is
read as a snapshot, with a banner. While the view is visible it looks at the
file every 30 s (Settings › Files and data › Refresh the Database view: Off, 15, 30 or 60 s) and
reloads the page in place, keeping the sort, the filters and the selection
("Updated 4 s ago · 3 new calculations").

### Building a database from a folder

**Create or Update Database…** reads a folder's CRYSTAL outputs into a
`materials.db`. It is on the Database view's toolbar (**Update from
Folder…**), on the empty Database page (**Create Database from Folder…**), on
a folder's right-click menu in the Explorer (**Create or Update Database from
This Folder…**), in Analysis and in the command search. It runs the same
command a terminal would, shown at the bottom of the dialog:

```bash
mace analyze --extract-properties runs --db-path runs/materials.db --register --skip-unchanged --backup --keep-backups 3
```

- **Folder** and **Database**: the database is the open one, else the
  folder's `materials.db`, else the workflow's own database (one
  `.mace_context_*` folder), else a new `<folder>/materials.db`. The dialog
  says "Will update runs/materials.db (1,240 property rows)" or "Will create
  runs/materials.db". A database outside the served folder is not written
  from the GUI: the dialog shows the command to run in a terminal instead.
- **Register calculations (recommended)** (`--register`) records the
  folder's completed calculations first, so the Database view lists their
  materials; turned off, the command carries `--no-register` (the
  terminal's default for a folder is to register). **Only new and changed files** (`--skip-unchanged`) extracts
  only what changed since the last build (size or content, never the file
  time alone); after an update that stopped before its end (canceled, or a
  CPU-time limit) it continues where that one stopped and says "Resumed: N
  of M already done". **Back up the database first** (`--backup`) is always on for an
  existing database: a consistent copy goes to `backups/` beside it, and the
  last 3 are kept (`--keep-backups`); the dialog lists them and says which one
  the update would prune.
- **Preview** is the command's `--dry-run --json`: every output is New,
  Changed, Unchanged, Not tracked (always re-extracted), Unreadable,
  Incomplete or Skipped (a failed or unfinished run no calculation records,
  or MACE's `X_temp.out` copy of the `X.out` beside it: not extracted); click
  a chip to list those files.
- **Run**: a task with progress, the current file, the elapsed time and
  Cancel. Extraction commits file by file, so a canceled build keeps what it
  wrote and names the backup. When it ends: "Database updated: 12 new, 3
  changed, 0 failed outputs; 9 materials (4 new); 2 skipped · Open
  Database" (the CLI's last line is the same sentence); outputs that gave no
  properties are listed under Problems.
- While a build runs, an open Database view of that database shows
  "Updating… (build in progress)" and reloads when it ends. A second build or
  a tracked submit of the same database from the GUI waits for it.

Nothing is deleted from the database. Re-extracting a calculation replaces its own earlier property values; rows filed under an old material id for that calculation are moved, not removed. A backup is made first.

### The Properties tab

The bottom panel's **Properties** tab shows everything the database holds for
the run of the document in focus: an output, or a plot, spectrum, volume or
deck that run wrote (a `diamond_band.BAND.DAT` shows `diamond_band.out`'s
rows). The header names the run, its calculation type, its calculation id and
the database. Rows are grouped (Energy, Electronic, Band structure, Effective
masses, Thermodynamics, ...), each with its label, value, unit, database key
and source: the file, the extractor or the GUI's Save to Database, a derived
property's name and version; the tooltip adds when it was extracted and the
files it was computed from. **All runs of** *material* lists every
calculation of the material, grouped by calculation type then group. A JSON
value (a curve, a table) shows its size; **View** opens it in full. Below
the table, each derived property of the run says whether it was stored, or
why it was skipped ("one temperature printed", "no BAND file beside the
run"). The filter matches labels, keys and groups.

- **Which database**: the open Database document's, when it has the run;
  else the nearest `materials.db` from the run's folder up to the served
  folder; else the folder's workflow database.
- **Not in the database yet**: the run's values computed now from the run
  (the extractor and the derived properties, nothing stored), and
  **Extract**, which runs the command shown (`mace analyze
  --extract-properties --output-file RUN --db-path DB --register
  --register-output-only --backup`) as a task: the run itself is registered
  (never the rest of its folder), the database backed up, and only
  this run's rows are written. **Extract Again** does the same for a run
  already in it (a derived property newer than its rows says so).
- **No database**: the computed values, and **Create Database…** (Create or
  Update Database from a Folder).
- **Copy** (tab-separated label, value, unit, key, source), **CSV** and
  **JSON** (with the provenance) export the rows shown, labels in plain text
  (`m_c (holes)`), each value in the unit shown with the stored value and unit
  beside it.
- **Units**: the preset at the top (**As stored**, **Hartree atomic units**,
  **eV/Å (solid state)**, **kJ/mol (chemistry)**) sets every row's unit, and
  **Normalize** shows the per-cell values (total energies, the FREQ
  thermodynamics, the cell volume, the DOS at E<sub>F</sub>, the total
  magnetization) per atom or per formula unit; intensive values are not
  divided. Click a row's **Unit** cell for a drop-down of that row's units
  (a standard drop-down: Tab to it, then the arrow keys): only that row
  changes, and its unit stays until the preset changes again (Normalize
  rescales it and keeps its unit). Energies show CRYSTAL's printed precision
  (10 decimals in Hartree, the same resolution in other units). A quantity
  the database holds in several units (E<sub>D3</sub> in Hartree and eV) is
  one row (its other keys show as `+1` beside the key). The conversions use
  the CODATA 2018 constants of `mace/constants.py`; `energy_per_atom_ev` is
  labeled E<sub>total</sub>/atom (without D3 and gCP). Display only: the
  database is not changed. The choice is remembered in this browser. See
  [docs/properties.md](properties.md#units).

Values the GUI computes elsewhere go through the same registry, so they are in
the database after the next extraction: the run card's energy per atom,
functional, basis set, CRYSTAL version and MPI processes, a BAND run's
automatic effective masses (the Relaxation Time dialog's), and a FREQ run's
thermodynamics at every temperature. A result you steer (the Relaxation Time
dialog's fit) is stored only when you click its **Save to Database**: it is
never written behind your back, and a re-extraction keeps it. What the
database stores and what the GUI still computes only for the screen is
listed in [docs/properties.md](properties.md) (`mace analyze --list-properties --gui`).

## Analyzing a run

Opening an optimization or single-point `.out` shows the run in the 3D view
with the Timeline (the energy card and the frame slider) in the bottom panel,
and a **Run** card first in the inspector:

- **How the run stopped**, in one line with an icon (green: converged or a
  normal end; amber: CRYSTAL stopped with a warning; red: failed). The words
  are those of `mace status`/`mace check`, refined by how the optimization
  ended: "Converged: all four criteria met at point 4 (CRYSTAL23 manual
  p.188)", "Stopped after the first point: the gradients were already below
  threshold", "Stopped with a warning: gradients converged and the energy
  change was below TOLDEE, but the displacements were not (manual p.208)",
  and so on. **Show in Output** opens the output at that line.
- The CRYSTAL version, calculation type, dimensionality, space group, atoms,
  functional, basis, SHRINK, TOLINTEG, spin, MPI processes, start and end,
  wall and CPU time, file size, SCF cycles per point and the final energy in
  eV and eV/atom: what the database would store for this output.
- **Copy CLI** copies the two terminal commands that print the same:

  ```bash
  mace analyze --convergence runs/mat1_opt.out      # add --json for the numbers
  mace check runs/mat1_opt.out
  ```

**The energy card** counts CRYSTAL's optimization points (the x axis is the
point number as the `.out` prints it) and draws point 1 on the same energy
surface as the others: CRYSTAL prints point 1's `TOTAL ENERGY(DFT)` without
its D3/gCP correction and compares later points with the corrected value, so
the card uses the corrected one (the energy table's tooltip shows both). Its
row has Log |ΔE|, Relative to (First or Final), Units, Segments (for a run
with FINALRUN restarts), Copy TSV and **Open as Plot**. **Segments:
Continuous** (the default, Settings › Plots › FINALRUN segments) draws every
point as one curve relative to the run's first or final point, with a dotted
line at each FINALRUN restart; **Separate** takes each FINALRUN segment
relative to its own first or final point. Copy TSV (point, segment, frame,
E in Ha, ΔE) and Open as Plot use the same choice, and so does the terminal:

```bash
mace analyze --convergence runs/mat1_opt.out --energy                       # continuous, relative to the final point, meV
mace analyze --convergence runs/mat1_opt.out --energy --segments separate --relative-to first --unit eV
```

**The SCF card** under it draws one SCF cycle by cycle: |ΔE| per cycle on a
log axis against the run's TOLDEE (dashed; where it came from in its
tooltip), or with Log |ΔE| off each cycle's energy minus the last cycle's.
Cycles where the energy went up are red; dotted lines mark the notable
cycles CRYSTAL reports after them: Fock/KS mixing, the cycle DIIS took over,
the cycle a SPINLOCK was released, a "possibly conducting" state and a
quasi-degeneracy warning, and TOO MANY CYCLES. **Step** (when the output
prints more than one SCF per cycle, e.g. a FREQ) picks it; the default,
Latest, follows a running job as the run view refreshes. An optimization's
`.out` prints the cycles of its first point only: CRYSTAL writes the rest
to `SCFOUT.LOG` in the job's scratch directory, which a MACE job copies
back next to the output as `<job>.SCFOUT.LOG` (only the one this run
wrote). With it there, Step lists every point (each SCF matched to its
point by the energy it ended with) and Latest is the last point; `mace
analyze --convergence FILE --scf --step point:N` reads it too. Units, Copy TSV (cycle, E in Ha, ΔE, the spin density of a
spin-polarized run, the events) and **Open as Plot** (Run Analysis › SCF)
work as on the energy card. A single-point output (which opens as a
structure, without the run view) shows the same card as the inspector's
first section, **SCF**, above its Run card; a FREQ output shows it under
Mode. Both read an output still being written again every 15 s. The run
card has an **SCF** button that opens Run Analysis › SCF. The terminal prints
the same:

```bash
mace analyze --convergence runs/mat1_sp.out --scf --unit meV        # --step first|N|point:N, --json
```

**Run Analysis** (Open as Plot, or Analysis › Run) is a plot document with
five tabs:

| Tab | Shows |
|---|---|
| Energy | E − E_ref per point (default: the final energy), one curve across the FINALRUN restarts, each marked by a dotted line. A restart re-references the integrals: **Segments: Separate** (Plot Options) gives each segment its own reference instead. **Log Y** shows CRYSTAL's printed energy change per point (`DE`, red when it went up) and each point's distance to the final energy (the run's, or its segment's when Separate), with the TOLDEE line. |
| Convergence | The maximum and RMS gradient and displacement per point on a log axis, each with its threshold (dashed) and where the threshold came from (the `.out`, the `.d12`, or the CRYSTAL23 default and its manual page). Filled markers met their criterion. Gradients and displacements are in CRYSTAL's internal coordinates for cell optimizations ("given without units", as CRYSTAL says). |
| SCF | SCF cycles per point, and the chosen SCF cycle by cycle (the SCF card's plot: \|ΔE\| against TOLDEE, or E minus the last cycle's, with the notable cycles marked; Step picks the SCF when the `.out` prints several). CRYSTAL prints the cycles of an optimization's first point only and moves the rest to `SCFOUT.LOG`, which stays in the job's scratch directory. |
| Cell | a, b, c, α, β, γ and the volume per point, absolute or in % of the first. |
| Tracked | Distances and angles followed through every frame (below). |

A click on a point selects that frame everywhere (the slider, the energy
card, the 3D view); the selected frame is a vertical line on every tab.
**Compare Runs…** overlays a second run; with different atom counts the
energies default to per atom, and absolute energies are offered only for the
same composition.

**The Atoms, Coordination and Measurements panels** of a run show the frame
on screen, read-only, under "Frame 3 of 4 · POINT 4 (read-only). Use Edit This
Step to change it." Moving the slider updates them. In Measurements, pick 2
to 4 atoms in the Atoms table (Ctrl+click) and **Measure Selected**, or type
the atom numbers; **Track Along the Run** adds the distance or angle to Run
Analysis › Tracked, followed across cell faces (an atom that leaves through one
face and comes back through the other does not jump).

**Problems** lists what the output says, with links to the line: how the run
stopped, SCF that did not converge, FINALRUN restarts (information when the
run converged), CRYSTAL's own WARNING lines (counted once each), tolerances
that differ from CRYSTAL's defaults, and imaginary frequencies of a FREQ run
beyond numerical noise (Settings › Plots › Imaginary-mode noise level, 20 cm⁻¹ by
default; smaller ones are listed as information), with **Show Modes**.

### Runs and files that are still being written

An output a job is still writing is shown as **live**, never as broken: MACE
reads it up to a snapshot and ignores a half-written last line, so a number
is never read half-printed. The run card says "Live · updated 40 s ago" and,
with **Follow the run** on, new points appear every 15 s without resetting
the zoom or the selected frame. A run counts as live while it has no end
marker and SLURM says the job is running, or (when MACE cannot ask SLURM) it
changed in the last 10 minutes (Settings › Files and data › Live window). A
run that looks stalled is still checked every 60 s until CRYSTAL's end marker
appears ("Finished"). The text view of a live output has **Follow (tail)**;
an output that a resubmitted job started over says "This output was replaced
(a new run started) · Reload". A BAND.DAT, DOSS.DAT or cube a job is still
copying opens with **Retry** instead of an error, and Calculation Setup waits
for a running job's wavefunction (`.f9`) before it writes decks that need it.

## The activity bar

The strip of buttons at the far left opens the views: **Explorer** (shows or
hides the side bar), **Database**, **Search files** (Go to File), **Tasks**,
**Jobs** (the bottom panel's Jobs tab, as View › Jobs), **Workflows** (the
Workflows view, as View › Workflows) and, on a cluster, **Cluster**; Settings
sits at the bottom. Each button names its view and shortcut on hover and is
reachable with Tab (Enter or Space opens it).

## The Cluster view

On a cluster (where `sinfo` or a quota tool exists) the activity bar has a
**Cluster** button (also **View › Cluster** and the command search; Settings ›
Cluster › Cluster view turns it on or off). It opens one document with five
tabs (six where a module system exists), all read-only and served from the
same rate-limited cache as [`mace cluster`](cli.md#mace-cluster):

- **Overview:** your running and pending jobs against your limits, cores in
  use against the QOS cap, fairshare, the fullest storage space, upcoming
  maintenance, and your jobs with a **Details** button each.
- **Nodes:** nodes by CPU type or partition (idle, mixed, allocated, down or
  drained, free cores, memory and GPUs) with a bar per row; **Show all
  partitions** adds the ones your accounts cannot use.
- **Will It Start?:** cores, nodes, memory, GPUs, constraint, partition and
  time; **Find Free Nodes** shows where it fits now, **Predict Start** asks
  SLURM (`sbatch --test-only`, nothing is submitted) and shows the predicted
  start in your browser's time zone. A job script of the folder can be
  predicted instead of a request.
- **Storage:** a Space and a Files bar per quota, amber from 80 % and red
  from 90 % (the `[quota]` levels of `~/.config/mace/site.toml`). **Find What
  Uses My Files…** runs the throttled scan of `mace cluster files` as a
  background task (with Cancel) and lists the busiest folders and cleanup
  commands to copy; nothing is deleted.
- **Limits:** your associations, current use and fairshare, and what the
  next submission of N jobs x C cores would run into.
- **Software** (only where `module` exists): the CRYSTAL module the job
  scripts MACE writes would load (the same answer as `mace site`, with the
  modules it passed over and why) and the xtb module Clean would load. The
  first look runs the module command once per code, which can take up to a
  minute on a busy login node; **Refresh** looks again.

The strip at the top says how old the data is; **Refresh** asks again, at
most every 15 s (the button counts down). When storage reaches the warning
level, a chip in the status bar ("scratch files 91 %") opens the Storage tab.

### Job details

**Details** on a job opens a Job document: state and reason, node, elapsed
time against the limit, a walltime warning with what MACE will do if the job
times out, progress from the live output (an optimization's four criteria
against their thresholds and "Point 14 of about 20-24"; a frequency run's
displacements), the time left as nested levels (the total, the stages of the
run and the loop running inside the current one, as in
[`mace cluster job`](cli.md#one-job-progress-eta-and-walltime)) with a "Why
this estimate" disclosure (what each level is based on and any threshold that
differs from CRYSTAL23's default; it stays open while the document
refreshes), cores, CPU and memory efficiency in plain numbers ("CPU
efficiency 62% (used about 20 of 32 cores on average)", "Memory peak 9.4 GiB
of 64 GiB requested (15%)", a short hint when clearly over-requested, the
measured values behind a toggle; a figure SLURM did not record says why; under
the CPU line one note: MPI libraries such as Intel MPI keep waiting ranks busy,
so CPU efficiency near 100% does not show that every core did useful work, and
comparing run times at different core counts finds the useful size),
**Show Live Use** for a running job (one `sstat` over its running steps:
`sacct` has no CPU time or peak memory until a step ends), for an array task the
array's tasks in the queue (each with **Details**), and the job's files. A file
inside the served folder opens here; one in another folder of yours (your
home, your scratch) gets "Open in a New Window Rooted at …" with the command
to copy; anything else is only named ("Outside the folders this GUI may
open"). While the document is visible it refreshes with the job poll
(Settings › Cluster › Job poll interval, 30 to 300 s, default 60).

**Notify me when my jobs finish** (Settings › Cluster, off by default) shows
a browser notification when one of your jobs leaves the queue; clicking it
opens the job. The browser asks for permission first; it works on
`localhost` (the tunnel), not on a plain `http` address of another host.

## Workflows

**View › Workflows** (also **Calculation › Workflows** and the command search:
"workflow") lists the workflows under the folder: every plan in a
`workflow_configs/` folder here or up to two levels down, whether it was
started from the command line (`mace submit --progress`, `mace workflow
--execute`, `mace submit --progress-plan`) or written by MACE 1.1.4 or 1.1.5.
It shows the same states as `mace workflow --status ID`, and its buttons run
the CLI's own commands (see **Actions** below).

Pick a workflow on the left. The grid has one row per material and one
column per step; each cell is the step's state in plain words:

| State | Meaning |
|---|---|
| Not started | Waits for the step it is built from |
| Starting | The step it is built from finished; its job's completion step submits this one |
| Not submitted | The step it is built from finished a while ago (30 min) and this one was not submitted: the completion step did not run (**Process Now** continues it) |
| Held | Waiting for a free slot of your queue budget (`mace queue`): the budget is full, the workflow's own *Keep at most N* is reached, or SLURM refused it at your submit limit (the reason shows on hover). MACE submits it as slots free up |
| On hold | The plan has automatic progression off |
| Paused / Canceled | **Pause** / **Cancel Workflow…** (`mace workflow --pause` / `--cancel`); **Resume** lifts either |
| Queued / Running | Submitted; SLURM has not started it yet / is running it. A running step shows its time left and a progress bar, from the same estimate as the Job details |
| Finishing | The job left the queue; its completion step has not recorded the result yet |
| Completed / Failed | Finished and recorded / the job failed |
| Being recovered | The failure was fixed and the job resubmitted (the cell counts the attempts) |
| Blocked | The step it is built from failed or was canceled |

Select a cell for its job id, run time, progress, attempts and error, and
buttons that open its output, input deck, job script and **Job Details**.
A TAU step's cell counts its strained single points (for example 3/5). Its
details show them finished, failed and running. After the fit, a
**Relaxation time** card shows:
- the RELAXTIM TRANSPORT uses, or "fell back, <reason>";
- the carrier and axes, and why they were chosen;
- E1 and C with their R², m_c and m_d (hover for where each came from) and
  τ, for every carrier and axis;
- the warnings (low confidence, failed runs);
- **Open tau_summary.json**.

The TRANSPORT cell's details show the τ it ran with. While the TAU step
runs, TRANSPORT shows **Starting** ("waiting for its relaxation time").
**What the states mean** under the grid repeats this table.

All your workflows and manual submissions share one queue budget (your SLURM
submit limit minus a reserve, `[queue]` in site.toml; see `mace queue` in
docs/cli.md). A job that does not fit is held, not lost: the list shows a
**Held N** count beside each workflow that has some, the counts above the
grid show **Held** (the reasons on hover), and a note says how many are held
and why. Held jobs start after each job's completion step, from **Process
Now**, and while the GUI is open (at most every two minutes it submits what
fits, the way `mace queue drain` does).

The view follows your jobs: each poll of your jobs (Settings › Cluster › Job
poll interval) that changes a step updates it at once, and while it is shown it
refreshes on the same interval so the time left moves on. Where SLURM cannot
be asked, queued and running states are the database's. A job MACE submitted
in the last two minutes shows as Queued even when the last answer of `squeue`
predates it.

#### Actions

Each button runs one CLI command in the workflow's folder, the command you
would type; the GUI changes nothing itself.

| Button | Runs | What happens |
|---|---|---|
| **Pause** | `mace workflow --pause ID` | Completions are still recorded; no next step and no recovery job is submitted. Queued and running jobs go on. |
| **Resume** | `mace workflow --resume ID` | Lifts a pause or a cancel and submits the steps held meanwhile. |
| **Cancel Workflow…** | `mace workflow --cancel ID`, then `--yes` | First lists the workflow's own queued and running jobs (job id, step, material, state) and asks; **Cancel Workflow and N Jobs** stops progression and cancels exactly those jobs with `scancel`. If the list changed meanwhile (a job ended, a step was submitted), nothing is canceled and you are asked to look again. |
| **Recover…** (a Failed step) | `mace recover --db DB --calc-id ID --dry-run`, then `--yes` | Shows the error class, the fix and the diff of every file it writes; **Resubmit** applies it and resubmits that one step, which stays in the workflow (the cell counts the attempts). Off while the workflow is paused or canceled. |
| **Process Now** (a Not submitted step) | `mace workflow --process ID` | Submits the steps whose completion step did not run, the way the finished job's callback would have, for this workflow only. Off while it is paused or canceled (Resume does it). |
| **Clone…** | (reads only) | Opens New Workflow… with the same structures, steps, settings and resources (see below). |

Resume, Cancel, Recover and Process Now change SLURM's state, so they are
turned off while MACE GUI is open to the network; Pause still works.

**Clone…** fills the builder from what the workflow was started with:
for a workflow made in New Workflow…, exactly the builder's choices (saved
next to its spec as `builder_draft.json`); for one started from a spec on the
command line, the spec as launched (`workflow_configs/spec_<id>/spec.json`:
the methods, BAND/DOSS settings files and resources; anything the builder
cannot show is named in a note); for one started with `mace submit
--progress`, the plan's structures, steps and job sizes (each step's
calculation settings then start from MACE's defaults, and a note says so).
The name gets `-copy`. Clone is off, with the reason, when the structures
are no longer in the folder.

### New Workflow…

**Calculation › New Workflow…** (also **New Workflow…** in the Workflows
view, and **New Workflow…** on CIFs, decks or a folder in the Explorer)
builds a workflow without the command line. Nothing is submitted until you
click **Submit** in the last section; **Back** and **Next**, or the numbered
list on the left, move between the sections.

1. **Structures.** Pick CIF files (MACE writes each first deck) or `.d12`
   decks you already made (the first step runs them as they are), from one
   folder. Each file shows the name it gets in MACE. The workflow's files
   (plan, inputs, outputs) are written in that folder; every step, the
   first one too, runs in its own folder under
   `workflow_outputs/<id>/step_NNN_<TYPE>/<material>/`, so the decks,
   outputs and job scripts already in the folder are never changed. A deck
   whose own output finished already is not run again: that output starts
   its next step (a deck changed after its output was written runs again).
2. **Steps.** Pick a ready-made chain (OPT → SP → BAND → DOSS and the
   others), or add and remove steps; only the steps that can follow are
   offered (BAND needs an SP or OPT before it; OPT2 and SP2 repeat a step;
   ELASTIC, EOS, QHA and CPHF come after an OPT or SP, once each).
   Each step type is explained under the chain. The name is optional (it
   is shown in the Workflows view; it starts as the folder and the chain).
3. **Settings.** Each step's settings in Calculation Setup's form: the
   CLI's own questions, answered by the CLI.
   - A CIF start's first step shows the settings the workflow converts every
     CIF with. These are the workflow's own defaults (B3LYP-D3,
     POB-TZVP-REV2, ...), not `mace convert`'s; what you change is all that
     differs, and **Reset** goes back.
   - A later SP, OPT2 or FREQ starts from each structure's own result of the
     step it is built from and keeps its settings. Each row says "inherited
     from …" with the parent step's value (for a deck start, the first
     structure's; every structure keeps its own). Change a row to override it
     for this step only; **Reset** goes back to the inherited value. You can
     change the functional (3c methods get their basis) and D3, the basis
     set, the DFT grid, the SCF tolerances, MAXCYCLE, FMIXING, the mixing
     method, level shifting, SPINLOCK, smearing, spin polarization and the
     k-point mesh, OPT2's optimization settings and FREQ's frequency
     settings; the rest stays the parent's. A FREQ step left at the inherited MAXCYCLE gets at
     least 200 cycles; a value set on the step is written as set (under 200
     with a warning).
   - **Spin polarized** shows the step before's spin treatment ("inherited
     from OPT (step 1) (spin-polarized)"); spin-polarized stays the default.
     Switch it for one step, for example a closed-shell OPT to get a tough
     geometry to converge, then spin on for OPT2: the steps after OPT2
     inherit its spin. A step that turns spin on or off starts from a fresh
     SCF guess, not its parent's wavefunction, since CRYSTAL stops on a
     density matrix of the other spin treatment ("RESTART NOT POSSIBLE -
     OPEN-CLOSED SHELL INCOMPATIBILITY"). The row says so ("starts from a
     fresh guess: the parent's wavefunction has no spin"), the Defaults
     section warns, and the deck's title ends with "SPIN on (closed-shell
     parent), fresh SCF guess" (or "SPIN off ..."). Spin off also drops
     SPINLOCK; spin on writes SPIN alone, as MACE's own spin-polarized decks
     do (a SPINLOCK only where you set one). An SP step with spin-orbit
     coupling treats spin itself: its row stays inherited.
   - **k-point mesh (SHRINK)** shows the step before's SHRINK record
     ("inherited from OPT (step 1) (SHRINK 12 24)") and takes this step's
     own: `IS ISP` ("8 16"), per-axis factors `IS1 IS2 IS3` ("8 8 4"; a
     slab's third factor is 1, a polymer's second and third), or `auto`,
     MACE's mesh from each structure's cell (the field offers `auto` and the
     inherited mesh). Emptying the field or **Reset** goes back to the
     inherited mesh, and the steps after inherit the change. With `auto` the
     row says what the first structure's deck gets; factors the space group
     makes equivalent are raised to agree, and the row says so ("written as
     SHRINK 9 18: ..."). The SCF still restarts from the parent's density
     matrix (GUESSP): the k points are a computational condition the
     restart allows.
   - An SP step (a later one, or the first step of a CIF start) has a
     **Spin-orbit coupling** group: switch it on to make that step a
     two-component (SOC) single point, choose the ECP route (INPSOC, the
     default; STUTSC, named "refused for Hf" with the CLI's reason when a
     structure holds an element whose internal STUTSC the CLI refuses, and
     grayed out when every structure does; or the official
     pob-TZVP-rev2-SOC basis) and, for a
     metal, a SMEAR width (empty = none; 0.005 Ha converged the HPCC fcc Au
     runs). The choices become that step's `overrides` in the spec, the keys
     `opt2d12` reads, and the form shows the deck `opt2d12` writes with
     them, with the CLI's SOC warnings or its refusal. A SOC deck needs an
     external basis set (explicit basis input): on a later SP step whose
     basis is internal (the workflow's default POB-TZVP-REV2), switching SOC
     on sets the step's Basis set to the matching external library
     (TZVP-REV2, the same shells; DZVP-REV2 for POB-DZVP-REV2) and, when every
     ECP atom of the structures has one, the route to the official
     pob-TZVP-rev2-SOC basis set; the group says what it picked and names an
     element without a SOC basis set (its ECPs get INPSOC). While SOC is on,
     the internal sets and the functionals without a two-component form
     (HSE06, for example; PBE0 and B3LYP have one) are grayed out, and an
     inherited one is named with the row to change. Switching SOC off again
     gives the step back its parent's basis. A SOC step here always
     starts from scratch; a GUESSPNOSO or GUESSPSO restart needs a parent
     known to be spin-polarized without symmetry or two-component, so set
     `soc_guess` in a spec by hand. OPT, FREQ and properties steps show the
     group grayed out with CRYSTAL's reason (no optimization or frequencies
     in a two-component run, manual p. 166), and a step that cannot start
     from a SOC run (OPT, FREQ, TRANSPORT or CHARGE+POTENTIAL after it) is
     flagged in the check.
   - BAND, DOSS, TRANSPORT, CHARGE+POTENTIAL and ORBITALS show opt2d3's form, worked
     out on a stand-in of the parent output (no step has run yet); the same
     settings apply to every structure. A step left as it is runs with the
     workflow's defaults for it; a changed one runs opt2d3's settings file
     from the form, so its deck is the one `mace opt2d3 --config-file` writes
     with that file on the same parent (for example a TRANSPORT temperature
     range, or CHARGE+POTENTIAL's potential at the atoms, POTC).
   - TRANSPORT also has a **Relaxation time** group. It sets RELAXTIM, the τ
     that σ and κ scale with (the Seebeck coefficient does not depend on τ):
     - **Default (10 fs)**: CRYSTAL's default.
     - **Fixed**: a value you type, in fs.
     - **From deformation potentials**: adds a **TAU** step right before
       TRANSPORT. Choosing another option removes that step again.
   - The **TAU** step's settings say how many strained single points each
     structure gets. The count comes from the parent deck's symmetry: 5 for
     diamond (one axis). Its options are:
     - **Carrier**: Auto means electrons, because MACE knows no doping.
     - **Axes**: Auto takes them from the symmetry.
     - **Strains**.
     - **m\***: Auto takes it from the workflow's BAND and DOSS.
     - **Fallback τ**.
     - **Hold TRANSPORT** when τ falls back or the fit is poor.

     Without BAND or DOSS in the chain, a note offers **Add BAND and DOSS**.
     The strained runs get the SP's resources with 1.5× its time limit
     (Resources).
   - **ELASTIC**, **EOS**, **QHA** and **CPHF** steps write `mace opt2d12`'s
     module deck for each structure from its own result of the OPT before
     the step (the SP in a chain without one): its geometry, functional,
     basis set and SCF settings, with SCF TOLDEE at least 8 (QHA 10) and
     MAXCYCLE at least 200. Their form has Calculation Setup's module fields
     (ELASTIC also the module: ELASTCON, ELAPIEZO, PIEZOCON, PIEZOCP or
     PHOTOELA), each marked "CRYSTAL default" until set, and shows the deck
     `opt2d12` writes on a stand-in of the parent, or the CLI's reason for
     writing none (EOS is 3D only, for example). The fields become
     the step's `elastic_settings` in the spec. Nothing is built from these
     steps; each is one CRYSTAL job (Resources: 7 days, 7G per core for
     ELASTIC and QHA, 5G for EOS and CPHF). Finite-displacement phonons
     (phonopy) are not a workflow step: use Calculation Setup's 8 Phonopy
     on the finished OPT output.
4. **Resources.** One row per step: cores, memory (per job or per core),
   time limit, partition, account and nodes to avoid. Each value says where
   it comes from: MACE default, site setting (`~/.config/mace/site.toml`, or
   a group file), environment variable, or "set here" for a value you typed
   (with **Reset**). A time limit longer than the partition allows is
   flagged. **Account** is exactly what each job script gets, as the CLI
   works it out (`mace workflow --spec SPEC --json` names it per step): your
   own (`[slurm] account` or `MACE_SLURM_ACCOUNT`), MACE's built-in one
   only where you are a member of it, or empty with "no account line" when
   the job script has no `#SBATCH -A` and SLURM charges your default
   account. Where your SLURM accounts cannot be read and MACE keeps its
   built-in account on a cluster other than MSU HPCC, the cell says so: type
   your account there or set it in your site settings. Below the table: submit the first step as a job array (and how
   many tasks run at once), and how many of the workflow's jobs may be
   queued.
5. **Review and submit.** What Submit does, in a sentence; the commands it
   runs, to copy; each structure's first deck and job script, exactly; the
   later steps of the first structure, **projected**: written by the engine
   from a stand-in parent output, so the structure and anything read from
   the output are placeholders while the settings and resources are the
   ones the workflow uses; the workflow spec, the plan, the job-script
   templates and every file written.

How it runs: the GUI writes only a workflow spec,
`<folder>/workflow_configs/gui_specs/<name>_<time>/spec.json` (and the
BAND/DOSS settings files beside it), then runs the CLI in the folder,
exactly as you would:

```
cd <folder> && mace workflow --spec workflow_configs/gui_specs/<name>_<time>/spec.json --write-plan
mace workflow --execute workflow_configs/workflow_plan_<time>.json                # CIFs
mace submit --progress-plan workflow_configs/workflow_plan_<time>.json <decks>    # decks
```

The review runs these same commands in a staged copy of the folder, with
SLURM's commands that change something recorded instead of run (queries
reach the cluster), and Submit runs them for real; the job scripts are then
compared with the review's and a difference is reported under Problems.
Submit needs a fresh review (15 minutes, used once) of unchanged structures
and site settings; one launch at a time runs in a folder; structures that
still have queued, held or running steps in a workflow in that folder are refused
(wait for them, or cancel that workflow first); and Submit is turned off
while MACE GUI is open to the network. After Submit the new workflow opens
in the Workflows view.

## Plots: band, DOS, IR, Raman and FREQ modes

Plots are documents with their own options (with their own undo) in the
inspector. plotly loads from the installed package the first time a plot
opens, once; nothing loads from another site.

| Document | Opens from | Options |
|---|---|---|
| Band structure | `*.BAND.DAT`, Analysis › Plot Band Structure | energy window, Fermi shift (or drag the Fermi line), line width, spin (both, α, β) |
| DOS | `*.DOSS.DAT`, Analysis › Plot Density of States | energy window, Fermi shift, line width, spin, which projections are shown |
| IR spectrum | `*_IRSPEC.DAT`, Analysis › IR Spectrum | wavenumber window, normalize to the highest peak, **average** the conformers beside it (as `mace plotting --spectra --average`); the main peaks are labeled |
| Raman spectrum | `*_RAMSPEC.DAT`, Analysis › Raman Spectrum | as IR, plus the component: total, parallel, perpendicular or one of the six tensor components, or all |
| Vibrational modes | a FREQ `.out`, Analysis › Vibrational Modes | the mode table (Modes panel), amplitude, speed, Arrow length, Play, Export Mode as XYZ |
| IR/Raman from the modes | Modes › IR or Raman (or Analysis › IR/Raman Spectrum) on a FREQ output with intensities but no SPEC.DAT | peak width (FWHM), line shape (IR Lorentzian or Gaussian; Raman Lorentzian share η) |
| Transport | any of a BOLTZTRA run's `.SEEBECK`, `.SIGMA`, `.SIGMAS`, `.KAPPA`, `.TDF` `.dat` files (`.DAT` too) | **Quantity**: the run's files and what they give (power factor, κₑ, ZT, Lorenz number, σ/τ, PF/τ); temperature (one, or all); tensor component (average of the diagonal, xx, yy, zz, or all three); against μ or the carrier concentration; for ZT the lattice κ_L, for PF, κₑ and ZT the relaxation time τ; line width; see below |
| Convergence | Explorer › Compare Convergence | line width, logarithmic axis |

**Transport.** Open any of a BOLTZTRA run's files (SEEBECK, SIGMA, SIGMAS,
KAPPA or TDF). As a density volume offers the potential cube of its run, the
plot finds the run's other files beside it by name (any case: MACE's runs write
`.DAT`). Its **Quantity** choice lists them all in the same document:

- **From the files:** S, σ, σS, κ₀ and TDF. SIGMAS.dat is −σ·S as CRYSTAL
  writes it. KAPPA.dat is the zero-field κ₀: the manual's κₑ (Eq. 14.3) is
  that coefficient. The TDF is drawn against energy.
- **Derived from them** on the run's own (T, μ) grid:
  - PF = S²σ in µW/(cm·K²);
  - κₑ = κ₀ − TσS²;
  - ZT;
  - the Lorenz number L = κₑ/(σT) in 10⁻⁸ W·Ω/K², with Sommerfeld's
    L₀ = 2.44 as a dotted line;
  - σ/τ and PF/τ, per second at the run's relaxation time.

A quantity the files beside the one you opened do not allow is grayed. Its
tooltip (and a click) says why, for example that it needs the run's
KAPPA.dat. Files that are not on one μ/T grid are refused, not interpolated,
and the panel says so.

- **Temperature:** one, or **All temperatures** (one curve each, for one
  component).
- **Component:** the average of the diagonal, xx, yy, zz, or all three.
- **Against:** μ, or the carrier concentration |n| in cm⁻³ on a log axis, with
  p- and n-type drawn apart. The carriers come from SIGMA.dat (alpha + beta)
  and the cell volume from the run's `.out`; a slab or polymer is drawn per
  cell.
- **Spin:** for an open-shell run, the alpha and beta blocks of σ, σS, κ₀ and
  the TDF are summed.

**ZT.** Without a lattice κ_L, the curve is ZTₑ = S²σT/κₑ (κ_L = 0). It has
no lattice term, so the axis title calls it an upper bound. Type the lattice
thermal conductivity into **Lattice κ_L** for ZT = S²σT/(κₑ + κ_L), drawn solid
with ZTₑ dashed. The y axis spans the total, because ZTₑ runs up where κₑ is
tiny. Where κₑ is below the precision of the files (|κ₀ − TσS²| < 2×10⁻⁶ κ₀:
they print 8 digits), it is rounding noise. κₑ, L and ZTₑ have a gap there,
and the panel says how many points. The total ZT keeps such a point only where
κ_L is at least 100 times that uncertainty. For a slab or polymer run whose κ₀
nearly equals TσS² everywhere, the panel says that its κₑ, L and ZTₑ are not
reliable. Give one value in W/(m·K), or the name of a CSV file with `T,kappa_L`
rows: beside the run (`kappa.csv` next to it), else a path from the served
folder. MACE interpolates the table linearly and never extrapolates: a
temperature outside the table has no total curve, and the panel says why.

SIGMA.dat and KAPPA.dat are σ and κ₀ at the run's relaxation time. MACE reads
it (RELAXTIM) from the run's `.out`, else from its `.d3`, which without
RELAXTIM means CRYSTAL's 10 fs; the panel says which. ZTₑ does not depend on
it, but the total does, because κ_L has no τ. **Relaxation time τ** is blank
by default, which uses the run's τ; a value rescales σ and κ₀ by τ/τ_run for
PF, κₑ and the total. With neither file beside the data, MACE does not know τ:
it draws no total, and σ/τ and PF/τ are grayed.

The panel lists both ZT peaks. The numbers are the ones
`mace analyze --extract-properties --kappa-lattice` stores
(`transport_zt_total_max`), computed by the same code. Export › CSV writes the
quantity per μ and T with the carriers. **Publication Figure** and **Copy CLI
Command** carry the quantity, temperature, component, axis, κ_L and τ to
`mace plotting --transport` ([CLI reference](cli.md#transport-plots-mace-plotting---transport)),
which draws the same curves.

The FREQ document animates one normal mode at a time in the viewport, at most
30 frames a second and never while the tab is hidden. The arrows follow their
atoms while the mode plays (one set per animation frame), and the Arrows box
turns them on or off without stopping the animation. The header's
**Structure** button opens the output's structure, and **IR** / **Raman**
open its spectrum. With Settings › Reduce
motion on it shows a still view: the atoms at rest, each with an arrow from its
surface along the mode, and a Play button. Arrows are turned into the drawn
cell's frame (for a slab, by matching a, b and their normal); when CRYSTAL's
frame cannot be matched the GUI says so.
The modes come from the output's FREQ section only: the output index records
where it is, and a worker parses that range with `mace plotting`'s own parser,
so a large output is never read whole. The table lists frequency, irrep, IR
and Raman activity; imaginary modes are flagged.

The rest of the band and DOS notes:

- Energies are in eV relative to the Fermi level. When a POTC run of the same
  material sits beside the data (`<material>_POTC.POTC.DAT` and
  `<material>_POTC.out`), the plotters put everything on the vacuum scale. The
  axis then reads E − E<sub>vac</sub>, and the dotted Fermi line sits at that
  shift instead of 0.
- CRYSTAL writes two spin blocks even for a closed-shell run. When the beta
  block equals the alpha block (bands) or its negative (DOS), the run is
  closed-shell and one set is drawn, with no spin legend. A spin-polarized run
  draws α solid and β dotted.
- k-point labels come from the `.d3` path. At a jump in the path (`U|K`) the
  line breaks, so no band is drawn across it.
- **Inspector › Brillouin zone** (every tier) shows a band plot's zone with
  the path the bands were computed along, its high-symmetry points labeled:
  a slab's 2D zone and a polymer's 1D zone, from the run's `.out` beside the
  file. Under it are the path facts: the segments with their fractional k,
  the ISS (shrinking factor), the number of k points and the file they come
  from. They are read from the run's `.d3`, else a fort.25's segment records
  or a BAND.DAT's header. A phonon fort.25 takes its ISS from the FREQCALC
  BANDS of the `.d12` beside it. Without an `.out` the path is still listed
  and the zone says why it is not drawn. An edited path (the k-path editor,
  Advanced) is drawn there too; a point under the pointer in the zone is
  marked in the editor's strip and on the plot.
- A band plot and a DOS plot of one run are drawn together when the other half
  sits beside the file. When it is in another folder (a workflow's `BAND/` and
  `DOSS/` folders), **Add DOS…** on a band plot or **Add Band…** on a DOS plot
  (next to Log Y) lists the folder's DOS or band files, the same material and
  the nearest folders first, and draws the band + DOS figure. The pair is
  checked from the runs' outputs: a different formula is refused, and a
  different periodic cell or a Fermi level more than 0.05 eV apart is a warning.
  Both halves use one energy zero: the vacuum scale only when both have their
  POTC files. When the two runs' Fermi levels differ, the DOS is moved onto the
  band run's E<sub>F</sub>, and the chip above the plot says by how much. The
  plot remembers the pairing, and undo, Reset Plot Options and **Remove DOS** /
  **Remove Band** undo it. Copy CLI Command and Publication Figure pass it
  as `mace plotting --band-dos FILE --partner OTHER`, with OTHER relative to
  FILE's folder.
- A phonon run's plots pair the same way. A phonon dispersion (the fort.25 of
  a `PHONON_BAND` run) has **Add Phonon DOS…** and a phonon DOS (the fort.25
  of a `PHONON_DOS` run) **Add Phonon Bands…**. The picker lists only the
  phonon runs' files, and an electronic plot's Add DOS… / Add Band… only the
  electronic ones. The figure is the dispersion on the left and the DOS on the
  right, on one frequency axis in the plot's unit (Units ▾: cm⁻¹, THz or
  meV). Both halves share zero frequency: there is no Fermi level and nothing
  is shifted. An imaginary (negative) mode keeps the axis below zero for both
  panels. The pair is checked as the electronic one: a different formula is
  refused, and so is a different supercell (SCELPHONO, from the runs' outputs
  or their `.d12`); an electronic half is never paired with a phonon half. The
  chip above the plot names the other file, and its tooltip the supercell and
  the alignment. **Remove Phonon DOS** / **Remove Phonon Bands** undo it, and
  Copy CLI Command and Publication Figure pass it as `mace plotting
  --phonon-band-dos FILE --partner OTHER` (`<material>.PHONON_BAND_DOS.svg`).
- Without a `.d3`, the DOS columns are counted from the DOSS.DAT header
  (`NPROJ`).
- Only bands within 25 eV of the Fermi level are sent, at most 400 per spin;
  the plot says when bands were left out.
- `api/plot` still makes `mace plotting`'s static PNG/SVG, for scripts.
- **Export › Publication Figure…** is `mace plotting --band` / `--dos`, or
  `--band-dos` when the plot shows the band structure and its DOS
  (`--phonon-band-dos` for a phonon dispersion and its DOS), run on
  the data. `mace plotting` draws it with plotly from the same figure code as
  the plot on screen (`--backend plotly`, the default where plotly and kaleido
  can export), at a white Light background, with an `E_F` legend entry for
  the dotted line (the screen has the note line instead) and the legends
  fitted by plotly's margins. It uses what the plot on screen uses: the plot's
  style (Plot style, color pair, band style, β line, DOS projections, DOS spin layout, total,
  unit), the energy window
  (`--e-lower`/`--e-upper`), the Fermi shift, dragged or typed
  (`--fermi-shift`), and with a POTC run the vacuum scale
  (`--energy-reference vacuum`; the POTC files are copied along with the
  data). The command it shows draws the same figure when run in the data's
  folder. `api/plot` takes the same as `emin`, `emax`, `shift` and
  `reference` (`auto`, `fermi` or `vacuum`), `kind=bandos` for the band + DOS
  figure (with `partner`, the other half from another folder), `unit`,
  `log_dos`, and the style by its option names (`style`, `spin_colors`,
  `spin_up_color`, `spin_down_color`, `band_color`, `spin_down_style`, `dos_colors`,
  `dos_spin`, `dos_total`, `dos_smooth`).

The IR and Raman notes:

- Every IR- or Raman-active mode is a marker on the curve: hover names it
  ("Mode 37 · 1234.5 cm⁻¹ · B2u · IR 123.4 km/mol") and a click opens Normal
  modes at that mode (a click on the curve picks the nearest active mode within
  half the line width). The FREQ output beside the SPEC.DAT is read for its
  modes when it was not read yet. A click on the Modes panel's stick spectrum
  selects that mode too.
- A spectrum broadened from the modes uses CRYSTAL's own width: IRSPEC and
  RAMSPEC's `DAMPFAC` is a full width at half maximum, 8 cm⁻¹ unless the run's
  `.d12` sets it, so the curve has the width of a SPEC.DAT beside it. Each line
  has unit area times its intensity, as CRYSTAL's curves. The HTML viewer of
  `mace plotting --freq` uses γ = 8 cm⁻¹ as a *half* width (FWHM 16 cm⁻¹), so
  its peaks look twice as wide; the CLI is unchanged.
- A degenerate set (a row such as `4-   6` in CRYSTAL's mode table) is one
  peak: CRYSTAL's IR intensity already includes the degeneracy
  (INTENS_n = ... d_n (Z_x² + Z_y² + Z_z²), as the output prints), so the
  GUI does not add it once per mode. The HTML viewer of `mace plotting --freq`
  still sums every mode of the set (a triply degenerate peak three times too
  tall); the CLI is unchanged.
- Raman intensities are relative, normalized by CRYSTAL to a maximum of 1000;
  from the modes the powder total is drawn (the SPEC.DAT plot offers every
  column).
- A FREQ output with no IR or Raman intensities says "This output has no IR or
  Raman intensities (run FREQCALC with INTENS)".

### Plot options

Every plot (band, DOS, IR, Raman, FREQ modes, the run card and Run Analysis)
has the same toolbar: **Log Y**, **Units** (eV, meV, Ha, kJ/mol, kcal/mol for
energies; cm⁻¹, meV, THz for frequencies), **Export** (PNG, SVG, Image… at a size and
DPI, CSV, Copy as TSV, Publication Figure…, Copy CLI Command), **Style ▾** (the plot's own style,
above) and **Plot Options**: Log X, Relative to, Per (cell,
atom, formula unit), the axis ranges, the line width and Reset Zoom. A log
axis of values that are zero or negative plots |value| and says so. Each plot
document remembers its own choices; new plots start from Settings › Units and
numbers › Energy unit and Settings › Plots. Copy as TSV copies exactly what is
plotted (units, normalization, reference and range), with a header row.

Log Y is a DOS axis on band and DOS plots: band energies are signed, so a band
plot alone has no Log Y, and on a band + DOS plot the button reads **Log DOS**
and makes only the DOS axis logarithmic (the energy axis stays linear). A DOS
plot alone keeps **Log Y**. Publication Figure and Copy CLI Command pass it as
`mace plotting --log-dos`.


## Volumes (cube files)

A cube opens as a **Volume** document: isosurfaces drawn in the page by 3Dmol
from a grid the server cut to Settings › Volume quality (Low 0.25 M, Medium
1 M, High 2 M points; the chip says "100×100×792 → 50×50×396 shown"). The
server reads the cube with `mace plotting --cube`'s own reader (`CubeFile`),
crops the vacuum as it does, and sends the voxel vectors as they are, so a
skewed (hexagonal) cell's surface sits on its atoms.

- **Surface**: level (a log slider for a density; ± for a signed volume), the
  + and − lobes (spin up and down, an orbital's two phases), opacity, style,
  mesh quality, atoms, cell. The first level is the middle of the CLI's own list of
  isovalues, so the first view matches `mace plotting --cube`.
- **Style** (1.1.6): Solid (at the opacity: below 1 it is translucent),
  Wireframe, Solid + edges, or Points (the surface's vertices as dots); in the
  Advanced tier, Smooth or Flat shading and Both sides or Front only. A
  potential-colored surface keeps its colors in every style.
- **Mesh quality** (1.1.6, the same presets as the pore surfaces: see
  [Mesh quality](#mesh-quality)): Draft sends at most 0.25 M points, Standard
  the grid of Settings › Volume quality (as before), High 2 M, and Publication
  up to 8 M, refining a grid smaller than that twice by cubic interpolation
  (a denser, smoother surface; the values in between are interpolated, never
  new data) with more smoothing passes. A new quality reads the grid again for
  that document only; **Advanced** holds the numbers (grid points, refine,
  smoothing passes). An orbital's grid follows the picker's Quality; a
  difference keeps the grid it was computed on (its surfaces' smoothing and
  style still change).
- **The toolbar's display options** work in a volume as in a structure
  (and the Inspector's Cell box is the same setting as **Show Unit Cell**):
  **Display Range** (the supercell: 2×2×2, Cycle Display Range, Custom), **Show
  Unit Cell**, **Show Bonds**, **Labels** (Element, Index, Element+Index;
  Wyckoff is drawn as Element+Index, a cube has no symmetry), **Boundary
  Atoms**, **Show Axes**, **View Along** a, b, c (and a\*, b\*, c\*), Reset
  View, the 90° turns and Orthographic / Perspective. A supercell draws every
  cell's atoms and, for a cube, the isosurface again in each cell (the grid
  is periodic: the same grid, moved by whole lattice vectors; surfaces are
  drawn for the first 27 cells of a larger range, and the atoms and cell for
  all, with a message). The range is per document, like a structure's, and a
  periodic axis only: a slab's vacuum axis stays one cell. A volume of a
  molecule has no cell, so Display Range, Boundary Atoms and View Along are
  grayed out with that reason; bonds and labels still work, and Show Axes has
  nothing to draw.
  The atoms keep the volume's own look (the atom style options are for
  structures).
- **Slice**: a lattice plane (a, b or c) at full resolution as a contour map
  under the 3D view, resampled on Cartesian in-plane axes for a skewed cell,
  drawn as a plane in 3D; the arrow keys step it.
- **Statistics** come from the full grid: min, max and, for a density, the
  *integrated density on this grid*. With effective core potentials the core
  electrons are not in the density, so that number is the valence count; it is
  left out when the grid does not cover the cell (the density is cut off at a
  face, or a periodic grid spans less than the cell), and a periodic grid that
  repeats its first plane as its last counts it once.
- **Color by Potential…** (the header, or Analysis › Color Density by
  Potential) colors a density's surface by a potential cube of the same grid
  (the same stem's `_POT.CUBE` is offered first; a cube of another cell is
  refused with the reason). The automatic range is the CLI's: the potential
  sampled on the isosurface as `plot_dual_cube_isosurface` builds it (where
  scikit-image is installed; otherwise at the density's grid points within 2 %
  of the level), then ±max(|5th percentile of the negative values|, 95th of the
  positive ones). The colors are the CLI's too, in Settings › Viewport ›
  **Potential colors**: Chemistry (the default, as GaussView and VMD: red =
  negative, electron-rich; blue = positive, electron-poor) or Physics (red =
  positive). The same setting colors a potential cube's isosurfaces (+ blue,
  − red in chemistry colors, as spin) and slices, and POTM/CLAS 2D maps.
  The inspector's **Red = negative (chemistry colors)** box overrides it for
  one volume; a custom range and Reset Range are in the inspector.
- **Difference…** (Analysis › Volume Difference) makes A − B (− C ...) with any
  coefficients and opens the result as an untitled volume; **Save Result As…**
  writes it as a new `.CUBE` with the CLI's `write_cube_file` and never
  replaces a file. Grids of different sizes or origins on the same cell are
  aligned by the exact affine map (each grid point sampled linearly in the
  other grid's own voxel coordinates): `mace plotting`'s `align_cube_grids`
  interpolates one point at a time in Python (hours for a 100 × 100 × 792 slab)
  and finds a point's index by projecting onto each voxel direction, which
  shears a skewed cell. Grids of one shape go through `perform_cube_arithmetic`
  itself. A slice of the result is a plane of the difference (read again from
  its operands), and Save Result As writes a temporary file that is linked to
  the name only if no file of that name exists.
- **Which CRYSTAL keywords give which grids**: ECH3 → `_DENS.CUBE` (charge
  density; with spin, also `_SPIN.CUBE`), POT3 → `_POT.CUBE`, properties LOCALWF
  with PRINTPLO (IPLOT < 0 and ICUBE > 0) → `<input>_WAN_CUBE.DAT`, the Wannier
  functions on a 3D grid, shown as a signed volume at ±0.2 (the level the
  manual's own example uses); ECHG and POTM → 2D maps (fort.25 MAPN, opened as
  contour plots). A WAN cube is recognized and drawn, but no real one has been
  tested yet.
- **Bonded minus superposed atoms** (the deformation density): run ECH3 on the
  SCF wave function, then PATO + ECH3 with the same grid records (PATO builds
  the density from superposed atomic densities), and take SCF − PATO with
  Difference. opt2d3 does not write a PATO deck yet.
- **Render with mace plotting** runs `mace plotting --cube` on the file, for the
  CLI's own figure (a difference made in the GUI has no file, so it is not
  offered there).
- A density is drawn with one surface: round-off values such as −1e-15 on its
  grid do not make it signed. A lobe with no surface at the current level is
  not drawn.

## Orbitals (ORBITALS .molden files)

An ORBITALS step (Calculation Setup, Batch Setup or the workflow builder,
like BAND or DOSS; see docs/cli.md, mace plotting › Orbitals, for the deck
and the k-point rules) leaves CRYSTAL23's Molden files beside the job.
Opening any of them (or **Open Orbitals** on the properties document) shows
the run in a Volume document with an orbital picker on the left:

- **k points** from the ORBITALS `.out`, with an R (real) or C (complex)
  badge. Complex k points of slabs and crystals are grayed out: CRYSTAL23's
  coefficients there fail the orthonormality check, and the tooltip says so.
  A polymer's complex k points work.
- **Spin tabs** (Alpha, Beta) for a spin-polarized run. CRYSTAL23 writes no
  valid β coefficients for a spin-polarized run whose α and β are the same
  (a closed shell with MACE's spin-polarized default, SPINLOCK: the example
  suite's diamond, h-BN, polyacetylene and EC): its β block fails the
  orthonormality check. When every β level has the energy and occupation of
  its α partner and the `.out` reports no net spin, β shows the α orbitals
  (checked again) and the picker says so ("β = α: …"); a run whose spins
  differ draws CRYSTAL's own β (O<sub>2</sub>, bcc Fe), and a β that fails
  there is not drawn. `mace plotting --orbitals` prints the same line.
- **The level diagram**, highest level on top, spaced by energy (eV from the
  HOMO; from E<sub>F</sub> for a metal, with "E<sub>F</sub> ± n" tags).
  Occupied levels are filled, the HOMO and LUMO are marked, and degenerate
  sets are bracketed with a ×n badge. A closed-shell level reads "2 e (closed
  shell)" (periodic files list 1.0 per band). Use Up/Down, Page Up/Down,
  Home/End to move and Enter or Space to show a level.
- **View**: Re ψ (+ and − lobes in the theme's --vol-plus/--vol-minus
  colors, the default), Im ψ (complex k only) or \|ψ\|². **Sum the set**
  (for a degenerate level) shows the summed \|ψ\|² of the set, never a
  signed sum: a single member of a degenerate set is one arbitrary mixing,
  and the inspector says so.
- **Cells** repeats a periodic orbital along its periodic axes (default 4 for
  a polymer, 2×2 for a slab, 2×2×2 for a crystal, at most 4 per axis) so lobes
  are not cut at the cell edge; the atoms, bonds and the box are drawn. It is
  the toolbar's **Display Range** (View › Display Range, Cycle Display Range,
  Shift+R), not a second control: the three fields show the range, and
  typing in them or choosing a range in the toolbar gives the same new grid.
  A range above 4 along an axis is brought back to 4 (the server draws that
  many); a non-periodic axis stays at 1. Each orbital grid is computed again
  for a new range, so the view is busy for a moment.
  **Quality** is Preview, Normal or Fine (0.40, 0.25, 0.15 bohr; budgets of
  0.3 M, 1 M, 2 M points), following Settings › Volume quality at first.
- The **Level** slider starts at the isovalue that encloses about 90% of
  \|ψ\|²; **Default Level** goes back to it.
- Each file is checked when it opens (orthonormality under the analytic
  overlap); a file that fails is not drawn.

Grids are computed in the GUI's worker pool and kept only in its cache (quick
previews). **Save Cube** writes the shown orbital as a Gaussian cube in
`<job>_orbitals/` beside the job (Normal or Fine quality), reusing a cube
already there; it warns when that folder holds many or large cubes (HPCC
counts files against a 1M-file quota). Slices and Color by Potential are cube
tools and are not offered on an orbital.

## Molecules, slabs and polymers

- A file with no cell (`.xyz`, `.mol`, `.sdf`, `.pdb`) opens as a molecule:
  it is edited in P1, and Inspector › Display › Vacuum picks the box drawn
  around it (see [Slabs, polymers and molecules](#slabs-polymers-and-molecules-116)).
- A periodic cell that holds a molecule, a slab or a polymer in vacuum (at
  least Settings › Vacuum gap, 8 Å, along one, two or three of its vectors)
  says so under the structure's badges ("Looks like a slab (vacuum 19.6 Å along
  c)"); a molecular crystal with no vacuum does not.
- Calculation Setup asks the CLI's dimensionality question (CRYSTAL, SLAB,
  POLYMER, MOLECULE). A structure with no 3D cell is answered for (MOLECULE,
  SLAB or POLYMER from its periodic directions). The CLI's coordinate writer is
  right only for orthogonal, correctly oriented cells, so the GUI first moves
  the structure rigidly (every distance kept): a molecule into an orthogonal
  cube (its largest extent + 2 × the vacuum gap), a slab with its two periodic
  vectors as a and b and c perpendicular to them (vacuum along z), a polymer
  along x with b and c orthogonal. A molecule or chain stored wrapped across
  the faces of its box (as a CIF keeps it) is made whole first (lattice
  translations only). The deck is the CLI's (`mace convert` through
  the answer driver) on that cell, so the GUI deck is the CLI's for the same
  oriented CIF. A cell whose vacuum is not along its vectors is refused with the
  reason. Save for such a source asks for a CIF first, as for any unsaved
  structure.

## Plot Gallery

Many outputs' plots of one kind at once: **Analysis › Plot Gallery…**, or
**Plot Gallery** in the Explorer's menu of a folder or of several picked
files. The same grid as `mace plotting --gallery KIND FILES/FOLDER`.

- **Set**: a folder (searched with its subfolders), the Explorer's selection
  (Ctrl-click; a file of another kind brings its run's file of this kind, so
  picking outputs and asking for band structures works), a Batch Setup folder
  (`batch_*`: its decks and the files it was made from) or an open database's
  materials, with the Database view's filters (their files below the
  database's folder).
- **Plot**: Band structure, Density of states, Band structure + DOS (each
  BAND file with the DOSS file of its run in a second panel), IR spectrum,
  Raman spectrum, Optimization energy (E − E(final) along each run), SCF
  convergence (|ΔE| per cycle of the last SCF) or Structure (thumbnails seen
  along c).
- **Columns** (1-8), **Axes** shared (every pane the same ranges) or
  independent, and for band and DOS panes the **Energy** window around the
  Fermi level (-6 to 6 eV by default, the same for every pane).
- **One at a Time** shows one pane large and interactive: Left and Right
  arrows (or Page Up/Down) move through the set and wrap at the ends, Home
  and End go to the first and the last.
- The **caption table** under the grid: number, material id, formula, band
  gap and energy (eV) from the `materials.db` beside the files (or up to
  three folders above), when it has them. Click a row to go to its pane;
  double-click a pane to open its file as a plot of its own.
- **SVG** and **PNG** export the grid as one figure: the server runs `mace
  plotting --gallery` in its own cache folder (never the browsed one) and the
  file downloads. **Captions CSV** downloads the table; **CLI** copies the
  command.
- A set is capped at 200 panes, with a message naming how many there were.
  Panes are read and drawn as they come near the view and dropped when they
  leave it, so a large set stays light.

## Compare

One document for every before and after:

- **Analysis › Optimization Trajectory**: frame 0 against the selected frame,
  with the frame slider.
- **Analysis › Compare with Input**: an output against the structure it
  started from: the `.d12` of the same name, else the parent CIF its
  provenance names, else its first frame. A centered `.d12` (R, F, I) that
  CRYSTAL ran in its primitive cell is compared through the output's first
  frame, which is that deck as CRYSTAL read it (checked: same formula, volume
  per atom and space group).
- **Tools › Compare Two Structures**, or **Compare With Active** on a file's
  context menu in the explorer.

The frames of one run (every OPT frame, and Clean) are paired by index. Two
files are paired by position: each atom with the nearest atom of its element
under the minimum image, measured in fractional coordinates so two files in
different Cartesian orientations still pair (the Hungarian assignment when
scipy is installed, else greedy). Two files can list the same species in the
same order and still order the atoms of one element differently, so the
files' own order is kept only when it pairs the atoms as well as any other;
otherwise a banner says "Atoms matched by position (max mismatch 0.012 Å)".
Above 3000 atoms of one element the files' order is used as it is (the
distance matrix would take hundreds of MB). Different counts or species are
refused with the reason ("8 atoms vs 64: compare a supercell with Build ›
Supercell first").

The looks (Ctrl+Alt+O cycles them): **Overlay** (the initial structure as a
ghost; 35% opacity in light mode, 60% and lightened in dark mode so it stays
visible on the dark viewport), **Two-tone** (initial and final each in one
tone, element shading kept, with a legend), **Side by side** (two linked
views), and **Displacement map** (the final structure colored by \|Δr\| on a
viridis-like scale, with a legend in Å or pm and scaled arrows).

**Δr** (shown when the cell changed) picks the displacement the arrows, the
map, the max/rms |Δr| and the Displacements table use: **With cell change**
(the default then) is r_final − r_initial in Å with the cell's change
included, each atom at the periodic image nearest its start and the mean
shift removed, so it does not depend on the cell origin; **Relative to the
cell** is the change of fractional coordinates in the final cell, which is
zero for atoms that symmetry pins while only the cell relaxes (AB graphite
along c). When no atom moved, the legend says so and lists the cell's change.
The arrow note reads e.g. "arrows ×11, 1 Å drawn = 0.0909 Å moved"; Arrow ×
starts where the largest arrow is 1.5 Å long.

The Timeline panel's sub-tabs hold the energy chart, Δcell, Δbond,
displacements and Δsymmetry ("Fm-3m (225) → Fm-3m (225)"). Click a header to
sort; **Copy TSV** copies the shown table; **Analysis › Export Table** writes it
as CSV or JSON. Analysis › Energy Along the Run, Bond Length Table and
Displacement Table open their sub-tab.

## Two editor groups side by side

Any two documents can sit side by side: an IR spectrum beside its structure,
two structures, two outputs, two volumes, two runs' plots, a database beside
a plot. Every document kind works in either group, with all its controls, and
either group can edit.

- **View › Editor Groups › Split Editor Right** (**Ctrl+\\**): a second group
  to the right with the active document (moved, so a file is edited in one
  place); with only one document open, the right group starts empty.
- **Open to the Side** on a file's context menu in the Explorer, **Move to
  Other Group** on a tab's context menu, **View › Editor Groups › Move
  Document to Other Group** (**Ctrl+Shift+\\**), or drag a tab onto the right
  group (with one group, onto the right half of the editor) to split.
  A document with unsaved changes moves after Save; one without a file (a
  comparison, a new structure) stays where it is.
- The focused group has a blue line over its tabs; click into a group, or
  **Focus Other Group** (**Ctrl+Alt+\\**), to focus it. Menu and keyboard
  commands that act on a document (Edit, Select, Build, Analysis, View Along,
  Save, Export Image, ...) go to the focused group, and the Explorer opens files
  into it. The status bar says "Right group" while the right group has the
  focus. The Inspector follows the focused group: it shows the right group's
  document (its plot options, volume surface, mode, cell and symmetry) while
  that group has the focus. Each group has its own bottom panel (atoms,
  timeline, modes, ...) under it. In a window narrower than 1100 px, where the
  Inspector is a drawer, the drawer shows the left group's document.
- **Link Cameras** (View › Editor Groups): two 3D views (structures, runs,
  volumes, modes) turn together; each keeps its own zoom and center.
- **Combined analyses** read the other group: a peak clicked in an IR or Raman
  spectrum opens its mode in the other group; **Color Density by Potential**
  takes the other group's volume as the potential without asking, and
  **Volume Difference** starts with the two groups' volumes; **Analysis ›
  Compare the Two Groups' Structures** compares them (as Compare With Active).
- Drag the line between the groups to resize them (or focus it and use the
  arrow keys). The split, its width, Link Cameras and the right group's
  documents are kept per folder in your profile and come back after a reload
  or on another port. **Close Group** closes the right group (it asks first
  when a document there has unsaved changes).

## Display range, boundary and rendering

Two Inspector groups, under **Display**, change the active structure document
only (they say "This document"); Settings › Viewport holds the defaults that
every document starts with, and a document follows a changed default for
every value it has not changed itself. **Display** (Cell, Bonds, Labels, the
style and colors) stays the same for all documents ("All documents").

**Range & boundary**:

- **Range**: from and to, in fractions of a, b and c (−5 to 10, the end at
  least 0.05 above the start). **Unit Cell** is 0 to 1, **Cell + 10%** is −0.1
  to 1.1 on each periodic axis, and **2×2×2** and **3×3×3** repeat the cell.
  An axis that is not periodic (a slab's c, a molecule) says so and ignores
  the range. The estimate below says how many atoms that draws; a range of
  more than 200 000 atoms is refused. View › Display Range › Custom… (and the
  toolbar's range menu) takes the same from and to values.
  A custom range of more than one cell along a non-periodic axis repeats it at the document's Vacuum mode (Inspector › Display › Vacuum; Settings › Viewport › **Vacuum spacing** is the default; display only): **Bounding box** (the atoms + 1 Å each side), **Open** (no repeats), **File-defined** (a CRYSTAL output's 500 Å, an extended XYZ's or a cube's box), **CRYSTAL 500 Å** or **Custom** (the atoms + **Custom vacuum gap**, 10 Å); a molecule repeats along x, y and z.
- **Boundary Atoms**: also draw the atoms on the faces, edges and corners of
  the range (Shift+A).
- **Bonded Atoms Outside**: **Bonded Neighbors** adds the atoms bonded to the
  drawn ones (one shell); **Whole Molecules** completes every discrete
  molecule that crosses the range. A chain, layer or 3D network is never
  grown: the group says the structure is an extended network (with its
  dimension, from the bonds MACE draws) and only completes discrete molecules.
  At most 50 000 atoms are drawn.
- **Polyhedra** (Shift+P), **Atom Size** and **Bond Radius** moved to
  Rendering › Style (below).

**Rendering** has two dropdowns that combine freely (Space-filling with
Glossy dark, Polyhedral with Publication): **Style** sets how atoms are drawn
and colored, **Lighting** sets the light and the background. Each applies only
its own values. Each also has **Custom**, your own look: changing one of its
values shows Custom and keeps that look; choosing Custom later (after another
preset, or in another document) brings it back. Before you have edited any
value, Custom starts from the current view and says so.

**Style**:

| Style | What it draws |
|---|---|
| Default | Ball and stick with the Display section's atom style, Settings' color scheme and radii (the look of earlier versions) |
| VESTA-like | VESTA's ball and stick whatever Settings choose: VESTA's atomic radii at 40 %, VESTA's colors, 0.12 Å bonds |
| Space-filling | van der Waals spheres, no bonds |
| Wireframe / sticks | Thin sticks only (0.06 Å), for big cells, frameworks and supercells |
| Grayscale print | Each element color as the gray of the same lightness, with outlines whatever the lighting |
| Polyhedral | Coordination polyhedra around the cations, translucent in the cation's color, with smaller balls and sticks. A center is a cation with 4 to 12 anion ligands (bonded neighbors more electronegative, Pauling): the Ti–O octahedra in TiPbO3, Mg in MgO, S in a sulfate; a perovskite's A site stays a ball. Each polyhedron is drawn once, around the center inside the drawn cells (Inspector › Rendering › Polyhedra sets the centers, opacity, edges and center atom) |
| Magnetic (spin-colored) | Each atom by its Mulliken spin moment on a diverging scale centered at 0: blue for excess spin down, white for none, red for excess spin up, full color at the largest moment. A small legend in the corner gives the scale in μB |
| Custom | Your own style |

Magnetic reads the moments from the final "ALPHA-BETA ELECTRONS" Mulliken
analysis of the run: the output itself, or for a CIF, deck or cube the output
of the same name beside it. The moments must be for the drawn atoms (same
count and elements). When there is no such output, the run is closed-shell
(every moment zero) or the counts differ, Magnetic is grayed out and its
tooltip, and the note under the dropdown, say why. Polyhedral is grayed out
in the optimization, mode and volume views (polyhedra are drawn in the
structure view only). **Polyhedra** (Shift+P), **Atom Size** and **Bond
Radius** are part of the style: changing them makes it Custom.

**Lighting**:

| Lighting | Look |
|---|---|
| Default | One front light, no ambient light, subtle depth cueing, the theme's background (the look of earlier versions) |
| Flat | Soft, even light without depth cueing: colors read true |
| Publication | White background, thin outline, ambient light and ambient occlusion, for papers |
| Presentation | Black background, stronger light and depth cueing, for projectors |
| Glossy dark | Black background, shiny atoms (a specular highlight) lit from the upper left, strong depth cueing, for slides |
| Custom | Your own lighting |

Grayscale print is a style, not a lighting: it changes the colors, and goes
with any lighting (Publication for a white page).

The lighting values:

- **Depth Cueing**: how much the farthest atoms fade toward the background
  (15% by default; 0 is off).
- **Light Strength**, **Light Direction** (Front, Upper Left, Top; an
  off-axis light raises Ambient Light to 15% so the shaded side stays
  visible) and **Ambient Light** (light from all directions).
- **Shininess**: a glossy highlight on every atom (Glossy dark).
- **Ambient Occlusion** (darkens creases between atoms; off on very large
  structures), **Outline** with its width and color (Theme: the theme's
  text color, or black or white, whichever stands out on a white, black or
  custom background), and **Background**.

A dot after a label means the document has its own value; click it to use
the default again. **Reset** does that for the whole group (Style and
Lighting each have their own), and **Save as Default** makes the document's
values the defaults. Settings › Viewport shows both defaults, **Style preset**
and **Lighting preset**; choosing one there also sets the values it stands
for. A document saved before the Style dropdown keeps its look: its Rendering
preset is its Lighting. These options only
change what you see: decks, CIF files and the settings JSON never contain
them. Export Image (PNG) draws the document's look; Export SVG draws flat
colors (no lighting, depth cueing or outline).

In VESTA terms: VESTA's default boundary is 0 to 1 on each axis with the
atoms on the boundary included (here: Unit Cell with Boundary Atoms on, the
default); −0.1 to 1.1 is a common choice; "Search molecules" is Whole
Molecules.

The 3D views also keep every drawn atom inside the clipping planes after a
range change, a pan or a zoom.

## Coordination and polyhedra

**Tools › Coordination Analysis** opens the Coordination panel: for the
selected atoms, or for each symmetry-unique site (with its Wyckoff letter)
when nothing is selected, the coordination number, Hoppe's effective
coordination number (ECoN), the nearest and mean bond lengths, the neighbors
by element, and the neighbor shells. The inspector shows the same for one
selected atom ("CN 6 (O×6, 2.106 Å)").

Two atoms are bonded under Settings' bond tolerance. Metal-metal pairs count
only for an atom with no metal-nonmetal bond, so Mg in MgO has its six O and
copper its twelve Cu.

**View › Show Polyhedra** (Shift+P) draws the coordination polyhedra as VESTA
does: around the cations coordinated by anions (O, N, F, Cl, S, ...), faces
translucent in the center's color, thin edges a shade darker. A perovskite's
A site (a cation whose polyhedra share faces with smaller ones of another
element) stays a ball, as do carbon and intermetallics. While they are on, a
center's sticks to its anions are not drawn, the center atom is small and the
corner anions smaller. Every other atom and bond keeps the Atom style (Ball and
Stick, Sticks, Wireframe, Space Filling): the A site's bonds, the bonds between
neighboring polyhedra (S-S in pyrite, Mg-Si in an olivine), a carbonate's C-O.
Each polyhedron is drawn once, around the center inside the drawn cells; the
center's copies on the far faces of the range draw none and are left out, and
so is an anion at no drawn corner unless it has a bond to a drawn atom, while
a corner outside the cells gets its atom, in the Atom style. Inspector › Rendering › Polyhedra (under the
checkbox): **Centers** (the candidate elements of this structure; checked ones
are drawn), **Face Opacity**, **Edges** and **Center Atom** (Small, Hidden,
Full Size), per document; Settings › Viewport holds the defaults. The geometry
worker builds them. Above 2000 drawn atoms they stay off and the viewport
banner says why. PNG export draws them as the view does; SVG export draws the
faces toward the viewer as flat translucent polygons and the edges as lines.

## SVG export

**Export Image…** offers SVG beside PNG: MACE's own orthographic projector
draws one circle per drawn atom (boundary atoms and the display range
included), the bonds as two half lines in their atoms' colors, and the cell,
painted back to front. The drawing lines up with the screen (it uses the
viewer's calibrated projection) and has no external references. Perspective
is refused ("SVG export is orthographic; press 5"). On a Compare document,
Export Image writes the comparison as SVG, with one group per pane when side
by side. 8000 atoms take about 0.2 s, written in chunks so the page never
blocks.

## Long sessions and SLURM jobs

For long or heavy sessions on a cluster.

### Long sessions on shared nodes

Some clusters limit every process on a dev node to a fixed amount of CPU time
(check with `ulimit -t`; many allow 120 minutes). The kernel kills a process
that reaches it. MACE GUI's server is mostly idle, but its helpers can get
there, so when such a limit is set:

- it says so when it starts ("This node limits each process to 120 CPU
  minutes. ...") and shows a "CPU 62 of 120 min" chip in the status bar from
  half of the limit;
- it restarts the ML worker (the model reloads, a few seconds) and starts
  fresh pool workers before they reach it; a pool worker the kernel kills
  anyway fails only its own task ("This node's CPU-time limit (120 min per
  process) stopped the task."), and the other tasks run again;
- at 80 % it warns, and at 97 % the server stops itself cleanly ("stopped
  before this node's CPU-time limit"); start it again for a fresh limit;
- a database update the limit stops keeps every file it already wrote and
  says so ("The database update reached this node's CPU-time limit after 40
  files; what it wrote is kept. ..."); start it again with "Only new and
  changed files" on (from a terminal, the same `mace analyze` command with
  `--skip-unchanged`) and it continues where it stopped, saying "Resumed: 40
  of 120 already done" in its progress and its summary.

Only per-process limits (`ulimit -t`) are detected; limits enforced in other
ways (cgroups, a site's monitoring daemon) are not. For long, heavy work,
submit jobs instead of running them on the dev node, or run MACE GUI itself
as a job (next section).

### Run it as a SLURM job

For long or heavy sessions (ML Clean, large outputs, a database build), MACE
GUI can run as a small batch job on a compute node instead of on a shared dev
node:

```bash
mace gui connect --slurm --gateway me@hpcc.example.edu --node dev-node-01 --dir ~/runs
                                  # from your computer
mace gui --slurm --dir ~/runs     # on a dev or login node of the cluster
```

What happens:

1. `sbatch` gets one job named `mace-gui` (`--parsable`, one argument per
   option) that runs `mace gui --managed --http-control --ready-file ...` in
   the folder (the server writes its address and token to that 0600 file, not
   to the job's log), with
   the environment you started it from (sbatch exports it, so a module or
   conda Python keeps working) and `--propagate=NONE` (a dev node's
   per-process CPU-time limit is not carried into the job). The job script is
   written to `~/.cache/mace/gui/slurm/` (mode 0700) and removed once
   submitted; the session token is never in it, in the job's environment or
   in its log.
2. While the job waits: "Waiting for a compute node (job 123, pending:
   Priority, expected start ...). Press Ctrl+C to give up." MACE asks SLURM
   every 30 seconds (through the same rate-limited cache as the Jobs panel
   and `mace cluster`). Ctrl+C, or a closed connection, cancels that job and
   nothing else, also when it has just started and MACE GUI is not ready
   yet.
3. When it runs, MACE forwards a port of the node you are on to the server
   on the job's node (`ssh -N -L ... <job node>`). This needs ssh to the node
   of your own running job, which most clusters allow (pam_slurm_adopt). If
   yours does not, the job is canceled and MACE says so; run MACE GUI on a
   dev node instead. A port that is busy on the node you are on (a shared
   login node) is not that: MACE takes another one (`mace gui connect` does
   too) and keeps the job.
4. The status bar shows "Job 123 · until 14:00". At 15 and 5 minutes before
   the job's time limit a banner warns; one minute before, the server stops
   cleanly. With nobody using it, it stops after 1 hour (Settings > Server
   changes this; an idle allocation spends your fairshare).
5. Ctrl+C in `mace gui connect` or `mace gui --slurm`, or File > Sign Out and
   Stop Server ("This also ends the job 123 ..."), stops the server; the job
   then ends and its allocation is released. Once its server answers, a job
   is never canceled from your side: its server stops and the job with it. If tasks are still
   running when the connection drops, the server keeps them, and running the
   same command again reattaches to that job instead of submitting another.

The request (flags of both commands; `mace gui connect` also keeps them in
its profile as `slurm = true`, `slurm_time`, `slurm_cpus`, `slurm_mem`,
`slurm_partition`, `slurm_account`, `slurm_qos`, `slurm_constraint`,
`slurm_gpus` and `slurm_args`):

| Option | Default | Notes |
|---|---|---|
| `--time` | 4:00:00 | the job's time limit |
| `--cpus` | 2 | one or two pool workers; the server itself is mostly idle |
| `--mem` | 4G | the server and one ML model; raise it for ML Clean or large files |
| `--partition`, `--account`, `--qos`, `--constraint` | your site settings, else SLURM's defaults | the `[slurm]` values of `mace site` that you or your group set |
| `--gpus` | none | e.g. `1` for ML Clean on a GPU; without it the GUI's GPU setting defaults to None (CPU) in the job |
| `--sbatch-arg` | | one more sbatch option as `--name=value` (repeatable), e.g. `--sbatch-arg=--reservation=r1` |

Smaller requests start sooner and cost less fairshare. `mace gui --slurm
--dry-run` prints the sbatch command and the job script without submitting.
The standalone scripts in `scripts/` do not support `--slurm`; use `mace gui
connect --slurm`.

## `mace gui` options and sign-in links

**The sign-in link** signs in **once**: the code in it becomes a session cookie
on the first load and the address bar drops it, so the same link from the
browser's history signs nobody in. For a new link run `mace gui login-url`
(`--port P` when several servers run; `--open` opens it), or paste the token
into the sign-in page. For a server started by `mace gui connect`, run `mace
gui login-url` on the computer where `mace gui connect` runs: the link uses
the forwarded port there (on the node it says so instead of printing a link). The `/s/…/` part is this session's own path; the
server answers nothing outside it.

- `--port P` (or `MACE_GUI_PORT=P`) picks the port; `--port 0` picks a free
  one for a one-off run. Browser drafts are kept per port, so if you rely on
  them use the same port every time (see [Drafts](#drafts)).
- `--open` opens a browser on this machine. On an ssh session with no display
  and no `$BROWSER` it opens nothing (a text browser would take over the
  terminal) and says so.
- `--verbose` (`-v`) logs each request on stderr (the token is never logged).
- `--root DIR` (repeatable, or `MACE_GUI_ROOTS`) sets the folders File ›
  Open Folder… may switch within (default: your home and scratch folders);
  `--root none` turns Open Folder off (see [Open Folder](#open-folder)).
- `--managed` is the mode `mace gui connect` uses over ssh: one
  machine-readable `MACE_GUI_READY` line on stdout, control lines on stdin,
  and the server stops when that connection closes and no task is running
  (see [First run from your computer](#first-run-from-your-computer)).
- When stdout is not a terminal (`nohup`, `sbatch`, `> gui.log`), neither the
  link nor the token is printed: the token goes to a state file only you can
  read, named in the output and removed when the server stops
  (`$XDG_RUNTIME_DIR/mace-gui-<port>.json`, or
  `~/.cache/mace/gui/url-<host>-<port>.json`, with the base URL, the token
  and the folder), and `mace gui login-url --port <port>` prints a sign-in
  link. `MACE_GUI_PRINT_TOKEN=1` prints the link and the token instead, for
  scripts. Scripts can also use the token in an `X-Mace-Token` header.
- `MACE_GUI_TOKEN` (or `--token`) sets the token instead of a random one per
  run. It must be at least 16 characters, each a letter, a digit or one of
  `. _ ~ -` (`python -c "import secrets; print(secrets.token_urlsafe(24))"`
  makes one); anything else is refused. `--token` on the command line is
  visible to other users in `ps`; `MACE_GUI_TOKEN` is not. A fixed token
  survives restarts (so does the session path derived from it).

## Security

A dev node is shared: any user logged in there can connect to its
`127.0.0.1`, so only the session token protects your session. The server:

- binds `127.0.0.1` only. Another address needs `--host` together with
  `--insecure-network`, because the token and your data then cross the
  network **unencrypted** (plain HTTP); the ssh tunnel is the supported way.
  With `--host 0.0.0.0` it accepts the node's own names and addresses as the
  Host and prints them. The test hooks (below) are never enabled then;
- requires a signed-in session or the token on **every** request: the token
  is random (192 bits) unless you set your own, which must be at least 16
  URL-safe characters (letters, digits, `. _ ~ -`; see
  [Run it locally](#run-it-locally)). The session path is derived from the
  token, so a log that shows the path is a reason to choose a long one. The
  token is never accepted in a URL or a cookie (see
  [Sign-in links, sessions and expiry](#sign-in-links-sessions-and-expiry));
  the session cookie (`HttpOnly; SameSite=Strict`) holds a random session id
  and its path is the session's own `/s/<nonce>/`, so the browser never sends
  it to another server it reaches on `127.0.0.1` or `localhost` (a Jupyter or
  TensorBoard tunnel, another user's app on the same node). The server
  answers nothing outside that path;
- does not print the token when stdout is not a terminal (see
  [Run it locally](#run-it-locally)); `--token` is visible in `ps`,
  `MACE_GUI_TOKEN` is not;
- refuses requests whose `Host` header is not the loopback name. This guards
  against DNS rebinding;
- accepts a POST only with an `X-Requested-With: mace-gui` header, which a
  page on another site cannot send without a CORS preflight that the server
  never answers;
- only serves files inside `--dir`. Symbolic links that lead outside it are
  not followed, and no save writes through a symbolic link;
- sends a Content-Security-Policy that allows scripts from the page's own
  origin only (no `eval`: neither 3Dmol.js nor the SVG plots need it), and
  serves HTML and SVG plots with a sandbox policy, so a plot cannot read the
  API or the cookie;
- drops a connection that sends no request within 30 s and serves at most 64
  connections at once; a connection that never sent a valid token is the
  first to go, so idle sockets of another user cannot lock you out;
- never enables its test hooks (`MACE_GUI_TEST_HOOKS=1`, for MACE's own
  tests) by accident: it says so on stderr when they are on, refuses them on
  a non-loopback address, and they never return the environment.

### Sign-in links, sessions and expiry

- The link `mace gui` prints works **once**: reopening it from the browser's
  history (or a chat app's link preview that opened it first) shows "This
  sign-in link was already used or has expired." with a sign-in form (a
  browser that is still signed in just shows MACE GUI). New links: `mace gui
  login-url`, or Enter in the terminal of `mace gui connect`.
- The token is for the sign-in form and for scripts (`X-Mace-Token` header).
  Wrong tokens and codes are answered after one second each, one at a time
  for the whole server, so guessing is slow; the right token never waits.
- A session ends after 12 hours without use (Settings › Server › Sign out
  after: 1, 4, 12, 24, 72 hours or Never; `MACE_GUI_SESSION_IDLE_HOURS`) and
  at the latest 7 days after sign-in (`session.maxDays`, 1 to 30;
  `MACE_GUI_SESSION_MAX_DAYS`). Only
  keyboard and mouse input count as use: a tab left open, its polls and its
  event stream do not. The page then shows **Signed Out**: your edits are kept
  as drafts in this browser, tasks keep running on the server, and pasting the
  token signs you in again.
- **Stop the server after … without use** (Settings › Server,
  `MACE_GUI_IDLE_STOP_HOURS`; off by default) stops the server after that many
  hours with no input from any tab and no task running. Every tab shows a
  warning 5 minutes before; any click or key keeps it running.
- A fixed token (`--token`, `MACE_GUI_TOKEN`) survives restarts. The state
  file holds the token (mode 0600, removed at exit), as the URL file did.

### Open Folder, mace gui connect and VS Code

- Open Folder stays inside the allowed roots (your home and scratch folders,
  or `--root`), refuses hidden and credential folders, and `--root none`
  turns it off ([Open Folder](#open-folder)). Nothing is written into a
  hidden folder.
- Credential and secret files are hidden for safety, wherever they are below
  the served folder and even when you serve your home folder: they are not
  listed, and opening, previewing or downloading one (also through a
  symbolic link) answers "<name> is hidden for safety (it may hold
  credentials or other secrets); MACE GUI never opens it". They are:
  - folders `.ssh/`, `.gnupg/`, `.aws/`, `.azure/`, `.kube/` (with
    `.kube/config`), `.docker/` (with `.docker/config.json`), `.config/gh/`
    and `.config/gcloud/`;
  - files `.netrc`, `.git-credentials`, `.pgpass`, `.pypirc`, `.npmrc`,
    `.vault-token`, `.env` and `.env.*` (`.env.local`, ...);
  - shell and interpreter histories: any dot name ending in `history`
    (`.bash_history`, `.zsh_history`, `.python_history`, `.psql_history`,
    `.Rhistory`, ...), `.histfile` and `.lesshst`;
  - ssh private keys anywhere: `id_rsa`, `id_dsa`, `id_ecdsa`, `id_ed25519`,
    `id_xmss` and names that continue them (`id_ed25519_work`); `id_*.pub`
    is public and opens.

  The list lives in one place (`mace/gui/roots.py`). Only the first group's
  folders and `.netrc`, `.git-credentials` and `.pgpass` make a folder count
  as a home folder for Open Folder; a project's own `.env` or `.npmrc` does
  not.
- `mace gui connect` gets the token only through the encrypted ssh channel
  (`MACE_GUI_READY` on the server's stdout) and opens the browser with a
  one-time link. While a dropped connection leaves the server running for its
  tasks, a 0600 state file in `~/.cache/mace/gui/` holds the token so that
  `mace gui connect` can reattach; it is removed when the server stops. With
  `--hop nested` the forwarded port also listens on the gateway's own
  `127.0.0.1`; the token guards it as on the node.
- In VS Code, the terminal's Ctrl+click goes through VS Code's own encrypted
  forwarding; the Ports tab address needs the sign-in link.

**Drafts and the shared origin.** Browser drafts are kept per origin, that
is per `http://127.0.0.1:<port>` in your browser. Any other page served at
that same address in the same browser can read them, so do not reuse your
GUI port for other tunnels.

### What the GUI writes into your folder

| What | Where | Notes |
|---|---|---|
| Save, Save As, Save All, the classic Save CIF | inside `--dir` (subfolders too) | a temporary file, then a rename; `<file>.bak` the first time a file is overwritten; never through a symbolic link |
| Save JSON (deck settings) | `--dir` | the same rules |
| Calculation Setup's settings file, its CIF copy (`setup/<name>/`) and, when you ask, the decks | inside `--dir` | the same rules; the decks are written by the CLI command it shows, into a new private folder first, then moved in (a deck already there keeps a `.bak`) |
| Batch Setup › Write | a new `batch_*/` folder inside `--dir`, nothing outside it | a replaced deck keeps a `.bak` |
| Submit | exactly the files `mace submit` writes (`<deck>.sh`, `.mace_arrays/`, `workflow_configs/`, `.mace_context_*`, `materials.db` rows; with a workflow, the decks' copies and job scripts in `workflow_outputs/`) | for a tracked submit, one backup of the folder's `materials.db` in `backups/` (the newest three are kept) and File History copies of the files the submit replaces; the jobs then write their `<deck>.out`, `<deck>.f9` and logs |
| Run on This Node | what `mace run` writes (`.mace_local/`, the outputs) | |
| New Workflow… › Submit | the workflow spec in `<folder>/workflow_configs/gui_specs/<name>_<time>/` (and the properties steps' settings files and the builder's `builder_draft.json` beside it), then exactly what `mace workflow --spec … --write-plan` and the launch command write (the plan, `workflow_configs/spec_<id>/`, `workflow_scripts/`, `.mace_context_<id>/`, `workflow_outputs/`, the first step's decks and job scripts) | a lock file `workflow_configs/.gui_launch.lock` while the launch runs; the review writes nothing in the folder (it runs in a staged copy) |
| Workflows view: Pause, Resume, Cancel, Recover, Process Now | what `mace workflow --pause/--resume/--cancel/--process ID` and `mace recover --calc-id ID` write (the control file `workflow_configs/workflow_control_<id>.json`, the next steps' or the recovered step's decks and job scripts, database rows) | the CLI runs in the workflow's folder; Recover's preview and Cancel's job list change nothing |
| Create or Update Database from a Folder | the database you choose inside `--dir` (its `-wal`/`-journal` while it is written, `structures.db` beside it) and `backups/` beside it | written by the `mace analyze --extract-properties` command the dialog shows; a consistent backup first (the last 3 kept); nothing is deleted from the database |
| File History: the previous version of every file a GUI write replaces | `~/.local/state/mace/gui/history/` (or `$XDG_STATE_HOME/mace/gui/history`, or `MACE_GUI_HISTORY_DIR`), never inside `--dir` | per folder and file; the last 20 versions of the past 30 days, 500 MB for inputs and 1000 MB for copies of outputs, never the newest copy of a file; files over 50 MB are not copied (their `.bak` stays). If the copy cannot be written, the GUI refuses the write |
| Settings | `~/.config/mace/gui.json` (or `$XDG_CONFIG_HOME/mace/gui.json`) | written atomically |
| The "built from" records of saved files (keyed by the saved bytes' hash) | `~/.cache/mace/gui/provenance/` (or under `$XDG_CACHE_HOME/mace/gui`, or `MACE_GUI_STATE_DIR`) | at most 2000 small files, oldest removed first; safe to delete |
| Clean's method check, GFN-FF and ML runs | `$XDG_CACHE_HOME/mace/preopt/` and `<tmp>/mace-preopt-<uid>/` | see [docs/preopt.md](preopt.md) |
| The Cluster view's cache (nothing in your folders) | `~/.cache/mace/cluster/<cluster>/` (or under `$XDG_CACHE_HOME`) | about 20 small files, mode 0600; shared with `mace cluster` |
| The state file (base URL, token, folder, for `mace gui login-url`; `mace gui connect` writes one for the forwarded port on your computer, and a server it starts writes one without the token on the node) | `$XDG_RUNTIME_DIR/mace-gui-<port>.json` or `~/.cache/mace/gui/url-<host>-<port>.json` | mode 0600, removed at exit |
| `mace gui connect` profiles | `~/.config/mace/connect.toml` (Windows: `%APPDATA%\mace\connect.toml`) | written only by the first run or `--save`; mode 0600, no passwords |
| A server kept for its tasks after the connection dropped | `~/.cache/mace/gui/managed-<host>-<port>.json` | mode 0600, removed when the server stops |
| The environment-module search (xtb) | `~/.cache/mace/modules.json` | 7 days; safe to delete |
| Plot renders, deck previews, Setup runs | `$TMPDIR/mace_gui_cache_*` and `$TMPDIR/mace_gui_deck_*` | removed when the server stops |

**Never written over:** calculation outputs (a CRYSTAL `.out` and SLURM job
logs), wavefunctions (`.f9`, `.f98`, `fort.9`, `fort.98`, `fort.20`), data
files (`*.DAT`, `fort.25`, cubes, `SCFOUT.LOG`), SQLite files, MACE's own job
scripts (recognized by content) and the deck of a job that is queued or
running. Save As to such a name is refused in the dialog with the reason.
Before any other replace, the current version goes to File History
(Explorer › Show File History…, or File: Show History in the command search:
Open, Compare with Current, Restore This Version); the save toast offers
**Undo** for 30 s. The `<file>.bak` beside a file is kept as before.

**materials.db and running jobs.** The Database view reads the database
through read-only connections and never waits behind a job. The GUI writes it
only through Create or Update Database (and the tracked submit), always after
a backup. On a local disk MACE keeps SQLite's WAL mode, as before; on a
network or cluster file system (NFS, GPFS, Lustre, BeeGFS, CIFS, …), where <!-- british-ok: Lustre is a file system -->
SQLite's WAL is not safe between nodes, every MACE command uses the rollback
journal instead, and on mounts without cluster-wide locks it also serializes
its writers with a lock file beside the database (it says so once). Set
`MACE_DB_JOURNAL=wal|delete` or `MACE_DB_LOCKING=trust|degraded` to override;
set them in `~/.bashrc` or not at all, so the jobs' completion callbacks use
the same mode as your shell.

After a crash or `kill -9`, the
`$TMPDIR/mace_gui_cache_*` folder and the state file can be left behind; both
are safe to delete. On HPCC `$TMPDIR` may be unset (then `/tmp`) or node-local.

## Dependencies

No new packages. The server uses only the Python standard library
(`http.server.ThreadingHTTPServer`); structures and the editor use ase and
spglib, and plots use plotly, all three from MACE's `requirements.txt`. If
one is missing (an install that never ran `pip install -r requirements.txt`
again), the feature that needs it answers with the `pip install` line
instead of failing, and the rest works. The process pool relies on two
internals of Python's `multiprocessing`; it is tested on Python 3.11 and
3.12, and on another version it falls back to the standard start method
with one warning.

The page uses:

- **3Dmol.js 2.5.5** (BSD-3-Clause), vendored with its license in
  `mace/gui/static/vendor/3dmol/`, beside `THIRD_PARTY_LICENSES.txt` for the
  MIT/zlib libraries its build bundles (iobuffer, netcdfjs, pako, upng-js);
- the VESTA and Jmol element colors in `mace/gui/static/element_colors.json`,
  taken from pymatgen's `ElementColorSchemes.yaml` (MIT License, The Materials
  Project / LBNL, commit 9100860d; license text in
  `element_colors.LICENSE.txt`; `mace/gui/tools/make_element_colors.py`
  regenerates the file from that YAML), and MACE's own colorblind-safe scheme
  in the same file (MIT, as MACE; made by that script from the hues of the
  Okabe–Ito palette, a published recommendation free to use; noted in
  `element_colors.LICENSE.txt`);
- the **Lucide 0.469** icons the shell uses, as one sprite
  `mace/gui/static/icons/lucide.svg` (ISC License, which covers the Feather
  portions under MIT; license text in `icons/LICENSE-lucide.txt`), next to
  MACE's own crystal glyphs in `icons/mace-icons.svg`;
- the colors of the editor themes in `mace/gui/static/css/tokens.css`
  (Catppuccin, Dracula, Gruvbox, Nord, One Dark, Solarized, Tokyo Night), as
  their projects publish them (each MIT License; notices and license text in
  `css/THEMES.LICENSE.txt`);
- the MACE logo (`mace/gui/static/brand/`, copies of [docs/logo](logo/README.md); MIT, this repository);
- **plotly.js**, served from the `plotly` Python package that MACE already requires
  (`plotly/package_data/plotly.min.js`).

The page loads nothing from a CDN, so it works on nodes without internet access.

## Example folder (`examples/gui/`)

`examples/gui/build_examples.py` rebuilds this folder. Nothing in it is
research data, and no number is made up for the GUI:

- `structures/`: symmetrized CIFs for textbook crystals (MgO, LiF, NaCl, Si,
  diamond, hBN), built from experimental lattice constants;
- `outputs/`: CRYSTAL outputs copied from MACE's own test files:
  - Si SP: the phonopy CRYSTAL example, BSD-3; see
    `tests/fixtures/phonopy_crystal/README.md`;
  - an hBN slab SP: a short CRYSTAL-style excerpt of the textbook hBN
    monolayer written for MACE's tests, labeled as synthetic in its first
    lines, there to show a 2D slab;
  - an **unfinished** Ag2Br3 FULLOPTG run (the R-3c mp-862982 cell, which the tests call `tqc_agbr`) with 6 geometries (from
    `tests/data/opt_restart`). It is there to show the before/after view;
- `materials.db`: written with `MaterialDatabase` from what those files
  contain. It has 5 materials with OPT/SP calculations and their statuses. Every
  number in it comes from the files above.

The Results tab needs `*.BAND.DAT` / `*.DOSS.DAT` files, which this folder
does not carry; they are in the example suite below.

### Example suite (`examples/suite/`)

```
mace examples                          # the table of what ships
mace gui --dir examples/suite --open
```

Real CRYSTAL23 inputs and outputs for five material archetypes (porous
framework, dense 3D crystal, 2D slab, 1D polymer, 0D molecule) across the
calculation types OPT, SP, FREQ, BAND, DOSS, TRANSPORT, CHARGE+POTENTIAL,
ORBITALS, PHONON_BAND, PHONON_DOS, ELASTIC, EOS, QHA and CPHF, laid out as
`<archetype>/<system>/<calculation type>/`. Diamond (3D), h-BN (2D),
polyacetylene (1D) and EC (0D) carry every type: OPT, then SP at its geometry,
then every other run from that SP, with the decks `mace opt2d12` and `mace
opt2d3` write; their n/a cells are only what CRYSTAL23 does not run for that
dimensionality, with the manual page in the manifest. Where the corpus has
no real output for a cell, the folder carries the input deck MACE writes and
`mace examples` shows the cell as a deck only; a calculation CRYSTAL does not
run for a system (a band structure for a molecule) is marked n/a, and a run
CRYSTAL stopped before finishing ships its real output as a failed run, with
the line CRYSTAL stopped on in the manifest. Large text
files are stored compressed and unpacked in place the first time the Explorer
lists the folder (or with `mace examples hydrate`); on a read-only install
`mace examples copy DEST` makes a writable copy. `MANIFEST.json` records every
file's origin and SHA-256. `examples/gui/build_examples.py --suite --corpus
PATH` rebuilds it.

## Limitations

- The editor edits a SLAB, POLYMER or MOLECULE structure in P1 only (Sym mode
  needs a 3D space group), and it refuses ghost and dummy atoms, which a CIF
  cannot carry.
- The Volume document has no orbital viewer of its own (a WAN cube is a signed
  volume) and no GIF export (the cube engine makes one:
  `python mace/plotting/engines/crystal_cubeviz_plotly.py FILE.CUBE --save-gif out.gif`,
  with kaleido and Pillow; `mace plotting` does not pass `--save-gif` on).
- Two editor groups at most (Split Editor Right); a tab dragged from the right
  group goes back with Move to Other Group (its tab menu or Ctrl+Shift+\\).
- Folder settings (`mace-folder.toml`) cover the theme-independent display,
  plot, Setup, database and Submit defaults listed above; the theme, layout
  and keybindings stay per user.
- A database update stopped by a dev node's CPU-time limit does not start
  again by itself; start it again with "Only new and changed files" and it
  continues where it stopped ("Resumed: N of M already done").
- Editing stops at 5000 atoms (viewing does not).
- The structure readers cover CIF, 3D `.d12` and CRYSTAL `.out`. SLAB, POLYMER
  and MOLECULE geometry is read from the `.out`. spglib symmetry is reported for
  3D cells only, because layer and rod groups come from CRYSTAL itself.
- The before/after arrows show r_final − r_initial with the cell's change
  included (**With cell change**, the default when the cell changed); **Relative
  to the cell** shows the change of fractional coordinates in the final cell
  instead, and the cell strain is listed separately in the cell table.
  Molecules are compared in Cartesian coordinates with rigid motion removed.
- The bond table counts two atoms as bonded by the viewer's rule at 1.2 × the
  sum of their bond radii (`structures.BOND_SCALE`); the drawn bonds and Clean
  use the document's bond tolerance (1.15 by default, which bonds the same
  pairs in every example structure). Very large cells (more than about
  4000 displayed atoms) are drawn without bonds.
- Setup previews ask the CLI each time the settings change (about 0.3-1.5 s
  locally, cached for settings it has seen). Run Preflight checks with
  CRYSTAL's TESTPDIM where a serial `crystal` is found (on HPCC through the
  CRYSTAL module); elsewhere it runs the structure check only and says so.
- FREQ normal modes are animated by the GUI itself, from the output's FREQ
  section. The HTML pages `mace plotting --freq` writes load plotly from a CDN
  and stay blank on nodes without internet access; the GUI needs neither.
- Spin-orbit coupling is for single points only (CRYSTAL23 has no
  optimization, frequencies or response properties in a two-component run,
  manual p. 166). In the workflow builder a SOC step starts from scratch; a
  GUESSPNOSO or GUESSPSO restart is set in a workflow spec by hand.
- MACE drives CRYSTAL only. Export Structure lists a Quantum ESPRESSO input
  but disables it ("Needs the Quantum ESPRESSO backend"); there is no Quantum
  ESPRESSO support yet.
- The page follows its theme and reduced motion, and every theme's colors
  meet WCAG 2.1 AA contrast (the two high-contrast themes AAA). The OS
  setting `prefers-contrast: more` does not pick a high-contrast theme by
  itself; choose one in Settings › Appearance.

## Troubleshooting

### Reporting a problem

**Help › Copy Diagnostics** copies a report for a bug: MACE and Python
versions, the OS, the browser and its WebGL, how the server runs, package
versions, recent errors, failed requests and the last server log lines.
Paths, file and folder names, free-text settings, tokens, user and host names
are removed; read it before you share it. **Help › Report a Problem…** shows
the same text to edit, copy or save (`mace-gui-diagnostics.txt`), with a link
to open an issue on GitHub. When the GUI does not start at all, `mace gui
diagnostics --dir <folder>` prints the server's part.

### mace gui connect: key passphrases and the gateway

- **"ssh could not log in with your key"**: ssh ended with "Permission denied
  (publickey)" or could not unlock a key. When your key has a passphrase,
  ssh may have no terminal to ask for it on (often so on Windows, or when run
  from a script). Load the key into an agent
  once per login session, then run `mace gui connect` again:
  `eval "$(ssh-agent)"` and `ssh-add` (Linux, macOS, MobaXterm), or on
  Windows start the **OpenSSH Authentication Agent** service (in Services,
  set its startup type to Manual and start it) and run `ssh-add`. A Host
  alias from `~/.ssh/config` (with its `HostName`, `User` and
  `IdentityFile`) works as the gateway too: `--gateway hpcc`.
- **"… has a passphrase and no ssh agent holds it, so ssh asks for it for
  the gateway and again for the node"**: with the jump hop two ssh
  processes open your key, the gateway's and the node's, and each asks for
  the passphrase. Run the `ssh-add` line it prints once per login session
  (Windows: `Start-Service ssh-agent` first) and connect asks for nothing.
- **"the node … did not answer through the gateway …: it may be down or not
  accept logins"**: with the jump hop, ssh reaches the node through the
  gateway and can only tell that the node's ssh did not answer in time
  (ssh's own words: "Connection timed out during banner exchange" and
  "Connection to UNKNOWN port 65535 timed out"). Most often the node is down
  or hangs: pick another dev node (`--node`, or `--save` it), or run
  `mace gui connect --check`. While it waits, connect says so: "(waiting for
  dev-node-01 through the gateway …, up to 120 s; if dev-node-01 is down,
  press Ctrl+C and pick another node)".
  - The other cause is a slow login: your login to the gateway (the
    passphrase, a 2FA approval) counts toward the node's `ConnectTimeout`.
    MACE gives the node 120 seconds when the gateway may ask for something
    (no `BatchMode yes` in its ssh config); load the key with `ssh-add` so
    the prompts go away, or allow more time with
    `--ssh-option=-oConnectTimeout=300`.
  - When the gateway lets you in without a prompt (the agent holds your key,
    or the key has no passphrase), connect also logs in to it a second time,
    beside the connection, and checks within 8 seconds whether the node's
    ssh answers (`bash` and `timeout` on the gateway). A node that does not
    then fails in about 10 seconds: "its ssh sent nothing within 8 s when
    the gateway tried it". mace-gui-connect.sh does the same;
    mace-gui-connect.ps1 waits out the timeout.
  - A gateway that does not answer at all is still reported within 20
    seconds: MACE opens a TCP connection to it first.
- **"the gateway … looks like an email address"**: the gateway is the login
  host you `ssh` to, `user@login-host` (e.g. `netid@login.example.edu`) or a
  Host alias from `~/.ssh/config`, not your email address
  (`netid@university.edu`). The warning shows when the part after `@` is a
  bare `name.tld` and no `Host` pattern in `~/.ssh/config` matches it; a real
  login host of that form gets the warning too, which a Host alias for it
  silences. MACE still connects. Fix the saved profile (in
  `~/.config/mace/connect.toml`, on Windows `%APPDATA%\mace\connect.toml`)
  with `mace gui connect [PROFILE] --gateway netid@login.example.edu --save`.
  The standalone scripts warn the same way.

### mace gui connect: the node asks for a password

With the jump hop (`ssh -J`, the default), ssh logs in to the node *from
your computer*, with the settings your `~/.ssh/config` has for the node's
name. A `Host hpcc` block with an `IdentityFile` covers the gateway only: the
node (`dev-node-01`) gets ssh's default key files (`~/.ssh/id_ed25519`,
`~/.ssh/id_rsa`, ...). When your cluster key is a different file and no agent
holds it, the node refuses the defaults and ssh asks for the node's password
(the prompt names the node, `netid@dev-node-01`).

`mace gui connect` (and the standalone scripts) now pass the gateway's
`IdentityFile` (and `IdentityAgent`, `IdentitiesOnly`) for the node hop too:
`--dry-run` shows them as `-o IdentityFile=...` after the `-o` options. If
the node still asks, any one of these fixes it:

- Name the key for the nodes too, in `~/.ssh/config` (the node names your
  cluster uses; a pattern works):

  ```
  Host hpcc hpcc.example.edu dev-*
      User netid
      IdentityFile ~/.ssh/id_ed25519_cluster
  ```

- Load the key into an agent (`eval "$(ssh-agent)"` and
  `ssh-add ~/.ssh/id_ed25519_cluster`): ssh offers an agent's keys to every
  host, both hops included.
- Pass it for this run: `--ssh-option=-oIdentityFile=~/.ssh/id_ed25519_cluster`
  (with the jump hop, `--ssh-option` values apply to the node; the gateway
  keeps its `Host` settings).
- `--hop nested`: the gateway logs in to the node itself, with the cluster's
  own keys there (no key of yours on the node hop).

Your key must also be in `~/.ssh/authorized_keys` in your home on the
cluster; most clusters share that home between the login and dev nodes.
Setting up keys, agents and the config from scratch, per platform:
[docs/ssh-setup.md](ssh-setup.md).

### mace gui connect --check

`mace gui connect --check` (with a profile, or the same flags you connect
with) checks the ssh setup one piece at a time and prints an `OK`, `WARN`,
`FAIL`, `INFO` or `SKIP` line for each, with one `fix:` line under each
problem:

```
mace gui connect --gateway me@hpcc.example.edu --node dev-node-01 --check
mace gui connect --gateway me@hpcc.example.edu --node dev-node-01 --check --no-net
mace gui connect --gateway me@hpcc.example.edu --node dev-node-01 --fix-config
```

It checks your `ssh` and its version, what `ssh -G` says for the gateway and
the node (key files, `ProxyJump`, connection sharing, and whether a plain
`ssh -J` would ask for the node's password), the agent (fingerprints only),
the permissions of `~/.ssh`, and then, unless `--no-net`, logs in to the
gateway and to the node by each hop running only `true` (with
`BatchMode=yes`: a gateway that wants a password or a 2FA code is reported,
not probed). The last line is the verdict: which hop works, and the
`mace gui connect` line to use. `--fix-config` prints a suggested
`~/.ssh/config` block and writes nothing. It reads no private key and asks
for nothing. The standalone scripts take `--check` (`-Check` in PowerShell)
too. Details and an example: [docs/ssh-setup.md](ssh-setup.md#5-check-your-setup).

Other messages from `mace gui connect`:

- **"cannot reach host … (timeout)"**, **"(DNS: the name does not
  resolve)"**, **"(connection refused)"**: ssh could not reach that host. ssh
  waits 20 seconds (`ConnectTimeout`; a `ConnectTimeout` in your
  `~/.ssh/config` for that host, or `--ssh-option=-oConnectTimeout=60`,
  replaces it, also for the nested hop's inner ssh). Through a jump host
  that may ask for a passphrase or 2FA the node gets 120 seconds, and the
  first jump host is checked within 20 before ssh starts ([key passphrases
  and the gateway](#mace-gui-connect-key-passphrases-and-the-gateway)).
  Check the name and your network or VPN.
- **"… matches no Host in ~/.ssh/config and does not resolve (DNS)"**: a
  warning before connecting, when the gateway is neither a `Host` alias nor a
  name your computer can resolve (usually a typo). MACE still tries.
- **"Warning: MACE on dev-node-01 is 1.1.5, older than 1.1.6 here"**: the
  page comes from the cluster's MACE; update that install (the message names
  its folder: `git -C <folder> pull`), then connect again.
- **"mace is not on dev-node-01's PATH there; using …/mace_cli"**: the default
  `mace` command was not found in the node's login shell, so MACE looked for
  `mace_cli` in `$MACE_HOME`, `~/mace` and `~/MACE` (and, for a `--mace`
  path that is not there, in that path's folder) and used the first one.
  Save it to skip the search: `mace gui connect --mace <that path> --save`.

### Messages

- **"3D view unavailable" ("3D view needs WebGL") where the structure should
  be:** this browser cannot draw 3D here (WebGL is off or not supported, which
  is common in remote desktops, or the browser gave the 3D view no WebGL
  context). Plots, text, the database and Setup still work; Export Image (PNG
  and SVG) and the tools that pick atoms in 3D say "Needs 3D (WebGL)". Turn on
  hardware acceleration in the browser's settings (Chrome and Edge: Settings ›
  System; Firefox: Settings › General › Performance), then reload the page. If
  that does not help, try another browser, or open MACE GUI from your own
  computer through an ssh tunnel. The status bar shows "No 3D" while this is
  so.
- **`kf.iconthemes: Icon theme "elementary" not found.`** (or other lines
  from the browser, such as GPU or sandbox notes): these come from the
  browser or your desktop's opener (this one is KDE's, harmless), not from
  MACE. MACE sends what the browser it starts prints to `browser.log` in
  its cache folder (`~/.cache/mace/gui/`, on Windows
  `%LOCALAPPDATA%\mace\gui\`), so they no longer show between MACE's own
  lines; a browser that was already running may still print to the terminal
  it was started from.
- **"cannot listen on 127.0.0.1:... Address already in use"** (on the
  node): this happens only for a port you named (`--port`, `MACE_GUI_PORT`);
  without one, `mace gui` takes your own port and the next free one if that
  is busy. Someone else, or another `mace gui` of yours, uses the port you
  named: leave `--port` off, or pick another; `--port 0` picks a free one for
  a one-off run. `mace gui --sessions` lists your own servers.
  (`mace gui connect` picks a random port on the node itself and tries again.)
- **`bind: Address already in use`** (from `ssh -L` on your computer): the
  port is taken there. Change the *first* number of the `-L` argument
  (`-L 9000:127.0.0.1:8xxx`) and use that port in the link (`mace gui
  connect` takes the next free one itself).
- **"This sign-in link was already used or has expired."**: each link works
  once (also after a link preview opened it). Run `mace gui login-url`, press
  Enter in `mace gui connect`, or paste the token into the form.
- **A page titled MACE that asks to sign in (403):** no session, or the server
  restarted with a new token. Open the link `mace gui` printed, or sign in
  with the token. In VS Code, the Ports tab opens this page: Ctrl+click the
  link in the terminal instead.
- **Signed Out:** the session was not used for the time set in Settings ›
  Server. Paste the token to continue; drafts and tasks are kept.
- **404 at `http://127.0.0.1:<port>/`:** the URL lacks its `/s/…/` session
  path. Open the link that `mace gui` printed.
- **421 "Misdirected request":** the page was opened by the node's name or
  real address. Use `http://127.0.0.1:<port>/…` through the tunnel (or
  `--allow-host`, with `--insecure-network`, if you really serve it on the
  network).
- **Wrong node:** a tunnel to another dev node behind the same gateway gives
  "connection refused" or "channel … open failed". `mace gui` names the node
  it runs on; the tunnel (or `--node`) must go to that exact node.
- **2FA or a host-key question:** `mace gui connect` waits for your answer
  and says "(waiting: approve the 2FA prompt if there is one. If ssh asks
  whether to continue connecting (the host-key yes/no question), check the
  fingerprint and answer yes ...)" after 15 s. With `--hop jump` a node that
  asks for 2FA too means a second prompt. See
  [First use of a node with --hop jump](#first-use-of-a-node-with---hop-jump).
- **"ssh could not verify dev-node-01's host key"**: this computer has never
  connected to that node. Run the `ssh -J ... true` line it prints once,
  check the fingerprint, answer yes, and connect again (or use `--hop
  nested`).
- **"The folder ... does not exist on dev-node-01"**: `--dir` names a folder
  the node does not have (the remote shell's `cd` failed). It is a path on
  the node; `~/...` is the node's home folder.
- **"mace: command not found"** (from `mace gui connect`): the node's login
  shell does not find `mace`. Pass `--mace <full path to mace_cli>` or a
  `--setup` line that puts it on PATH.
- **"The MACE on <node> is older than this connect"**: update MACE on the
  node, or use the manual tunnel.
- **"The node did not accept a login through the gateway"**: the node accepts
  logins only from the login node; `--hop nested` (auto does it once).
- **"… administratively prohibited"** (from ssh): your cluster turns off port
  forwarding on that host; ask your administrators.
- **A server still running on the node** after a dropped connection: it stays
  only while GUI tasks run, then stops by itself; `mace gui connect` again
  reattaches to it. As a last resort, `pkill -u $USER -f 'mace gui --managed'`
  on the node.
- **"stopped before this node's CPU-time limit"**: see
  [Long sessions on shared nodes](#long-sessions-on-shared-nodes).
- **A panel says it needs a package (503 with a `pip install` line):** run
  `pip install -r requirements.txt` for the Python that runs `mace` (on a
  cluster after the same `module load` lines).
- **A blank or gray 3D view:** WebGL2 is off in the browser, often in a remote
  desktop or a virtual machine. Check `about:support` (Firefox) or
  `chrome://gpu` (Chrome, Edge) for WebGL2, turn on hardware acceleration, and
  reload.
- **"Lost the connection to the MACE server":** the ssh tunnel closed or the
  server stopped (the end of an `salloc`, Ctrl+C). Your edits are kept in the
  tab and saved as drafts; start the tunnel or the server again and open the
  link it prints.
- **Leftovers after a crash:** see the table in [Security](#security): the
  `$TMPDIR/mace_gui_cache_*` folder and the state file are safe to delete.

## Setup and workflow changes in 1.1.6

- **One row for each choice.** Calculation Setup has one *Functional* row (the
  method, Hartree-Fock or DFT, and the functional's family follow from it; the
  Hartree-Fock variants are listed last) and one *Basis set* row (the internal
  sets and MACE's bundled external sets; for a follow-up the first choice is
  "Keep the parent's basis set"). A Hartree-Fock or 3c functional hides the D3
  and grid rows it cannot use. The settings file and the decks are the ones the
  CLI writes for the same choices.
- **Numbers.** A menu option shows the number the CLI gives it ("3 - Hybrid ...");
  a functional is numbered inside its family, as `mace opt2d12` lists it. The
  ready-made workflow chains carry the planner's template numbers.
- **Convergence levels show their values.** The SCF tolerances menu lists each
  level with what it sets ("2 - Tight (TOLINTEG: 8 8 8 9 24, TOLDEE: 9)"), the
  optimization's Convergence menu with TOLDEG, TOLDEX, TOLDEE and MAXCYCLE, and the
  row's hint names the chosen level's values. They come from the same table as the
  CLI's menus.
- **SeeK-path is the default BAND path of a 3D crystal.** The BAND section's *Path
  format* starts at "SeeK-path full paths" for a 3D output and at the high-symmetry
  labels for a slab or polymer, as Enter does in `mace opt2d3`. *Static k-path*
  (`--static-kpath`) writes the static table path instead, for a machine without
  the seekpath package. A workflow BAND step is shown for a stand-in output, so
  its preview carries the static table path; the real step computes SeeK-path's.
- **TRANSPORT: a k-point mesh row.** The TRANSPORT section's *k-point mesh* is
  `mace opt2d3`'s question for BOLTZTRA's NEWK mesh: Screening (the parent SCF's
  mesh), Standard (the default: 0.064 / 0.015 / 0.009 1/Å for 3D / 2D / 1D), Fine,
  a k spacing (*k spacing*) or a shrinking factor (*NEWK IS*, written as given);
  each preset names the IS it gives for the output in front, never below the parent's. The
  transport plot notes a run whose mesh looks too coarse (its IS below 60% of the
  Standard one, or a 300 K Lorenz number in the bands far from Sommerfeld's).
- **DOSS: a Resolution row.** The DOSS section's *Resolution* is `mace opt2d3`'s
  energy-resolution question: Screening (0.2 eV step), Standard (0.02 eV, the
  default), Fine (0.005 eV), Step (another step, in *Energy step*) or Points (a
  number of energy points, written as given, for screening many structures).
  Each preset names the points and the step it gives for the output in front;
  for all bands the span is estimated from the atoms, and the deck has at least
  1000 and at most 99997 points (CRYSTAL's DOSS.DAT limit). When a step over every
  band needs more than 5000 points, the Resolution row says so, and the deck
  covers -20 to +15 eV around E_F (*Window around E_F*: Yes by default, bounds in
  *Window below/above E_F*; No keeps every band), as `mace opt2d3` writes it on
  every route; without a trustworthy E_F (a spin-locked SCF with no BAND or DOSS
  run beside it) the deck keeps every band. With a BAND run of the SCF beside it
  the bounds move outward into band gaps (the deck preview's log says which). A
  1D material takes its bands by index instead (*Bands by index*: Yes by
  default, No keeps every band; *Conduction bands*: how many above the highest
  occupied band), since CRYSTAL needs a window's bounds in gaps. The DOS plot
  of a band-index run that stops below E_F+15 eV says so, and how to reach
  higher. A workflow DOSS step's form has the same rows,
  at Standard unless changed. A saved settings file keeps the number of points
  it names. The DOS plot draws the deck's points; *DOS smoothing* only smooths
  what is drawn.
- **Per-step settings in New Workflow.** A step after the first inherits its
  parent's settings for every structure. Its form changes what the CLI planner's
  customization levels change: the functional and D3, the basis set, the DFT
  grid, the SCF convergence and the rest of the SCF block (MAXCYCLE, FMIXING,
  mixing method, LEVSHIFT, SPINLOCK, smearing), spin polarization, the k-point
  mesh (SHRINK, or `auto`), OPT's type and convergence, and FREQ's template,
  NUMDERIV and IR. Each row shows the parent step's value as inherited until you
  change it; a step that turns spin on or off starts from a fresh SCF guess. The
  choices are written into the workflow spec as `basis_modifications`,
  `tolerance_modifications`, `grid_modifications`, `scf_modifications`,
  `kpoint_modifications`, `optimization_settings` and `frequency_settings`, the
  keys `mace workflow --spec` reads.
- **A folder of existing calculations.** A workflow started from decks copies
  each deck (with the `.f20`, `.ext` and, for a GUESSP restart, `.f9` its job
  script reads) into its step folder and writes a new job script there, so a
  deck's own edited `.sh` and its outputs stay as they are and nothing asks to
  overwrite them. A deck whose `.out` is a finished run of its type is not run
  again (unless the deck changed after it, or the `.out` is a symlink); the review lists it, and its next step is among the jobs Submit sends.
- **Start each next step by itself** (Resources) is the spec's
  `execution.auto_progression`; off, every finished step waits for *Process now*
  in the Workflows view. A chosen cap of 200 jobs is kept (a 1.1.5 plan's 200
  was only the default).

## For developers

### Measuring the editor's speed on a GPU

The default bench draws in software, so its orbit frame times are only
information. On a workstation with a second GPU, run the orbit headful on
that GPU once:

```bash
MACE_GUI_BENCH=1 MACE_GUI_BENCH_GPU1=1 nice -n 10 pytest -m guibench \
    tests/gui/bench/test_bench_editor.py -k orbit
```

It starts Firefox through PRIME offload on GPU1 (never GPU0, which drives the
display), checks that GPU1 has at least 1 GB free first, and fails if any of
its processes appears on GPU0.

### Tests

- `tests/test_gui_examples.py`: the server, its token and access rules, the
  readers, the before/after comparison and the GUI → JSON → CLI loop, through
  `mace_cli gui` in a subprocess.
- `tests/test_gui_editor.py`, `tests/test_gui_clean.py`: every edit operation,
  symmetry-aware edits, undo/redo, the CIF → cif2d12 round trip, the editor API,
  and Clean on distorted textbook cells.
- `tests/test_gui_results.py`, `tests/test_ipdos_readers.py`: the band/DOS
  arrays on synthetic files, and a check that lifting ipDOS's readers into
  functions left its plot byte-identical.
- `tests/test_gui_viewer.py`: formulas, element colors, slabs, and
  `node --check` on the page's own scripts.
- `tests/test_gui_browser.py`: the Structure, Results and Editor tabs in
  headless Firefox (selenium; skipped without Firefox or WebGL). With
  `MACE_GUI_SHOTS=<dir>` it also saves light and dark screenshots.
- `tests/test_american_english.py` (the GUI's front end, these docs, the
  logo and the example folder) and `tests/test_gui_american_english.py` (the
  example database): American English.
- `tests/test_gui_smoke_cli.py`, `tests/test_gui_writes.py`,
  `tests/test_gui_help.py`, `tests/test_gui_cli_startup.py`: the release
  gates through `mace_cli gui` (access rules, the session path and cookie
  scope, the token rules and the URL file, connection limits, SIGTERM and a
  hangup (also of a real terminal), writes that refuse symbolic links, the help, and that plain commands import no GUI
  code). `tests/gui/test_child_env.py` and `tests/gui/test_pool_internals.py`:
  the Python children's environment and the pool (no orphan after SIGTERM,
  Ctrl+C, a hangup or SIGKILL; a deck still written with one worker). `tests/gui/test_a11y_e2e.py` (axe-core, keyboard,
  reduced motion; needs `requirements-gui-test.txt`) and
  `tests/gui/test_tokens_contrast.py`: accessibility.
- `tests/gui/test_shell_e2e.py`: the window in headless Firefox: the menus
  against the command list (light and dark, 1280×800 and 1920×1080), every
  toolbar tooltip, the status bar, documents and the timeline, shortcuts,
  the command search, undo/redo and the modified marks, settings that survive
  a reload, themed scrollbars, unclipped labels and the same layout in both
  themes. Phase 2 adds the default keymap against the spec, user keymaps and
  the refused, risky and page-leaving keys, the menu mnemonics and F10, the
  palette's modes and speed, the Settings dialog (live viewport changes, the
  exact contents of gui.json, search, errors, the smallest window), the
  keymap editor, the shortcut sheet's columns and one-page print, a rebinding
  shown everywhere, the context menus, the leave prompt, and contrast and
  clipping in the dialogs. `tests/gui/test_shell_api.py` covers the settings
  file and its per-key checks, the resource limits, the CLI defaults, the
  event stream, the cached assets and the access rules of the new routes.
- `tests/gui/test_editor_e2e.py` and `tests/gui/test_editor_api.py`: the
  editing of Phase 3 in headless Firefox with real pointer actions (picking,
  box and lasso selection, the camera, every tool and its modal keys, Sym and
  P1, the inspector, the dialogs, measurements, the Atoms and History panels,
  saving with its confirmations and `.bak`, drafts across a reload, export at
  4× with transparency, New Structure and Edit This Step), and the server's
  side through the real `mace gui`: revisions, jumps, the one save path and
  the bounded `/api/symmetry`. `tests/gui/test_geom_core.py` runs the pure
  geometry module under node against numpy.
- `tests/gui/bench/` (opt-in, `MACE_GUI_BENCH=1`): the performance budgets
  (start time, first paint, the size of the page, tab switching, idle CPU,
  reload size, the heartbeat; and for Phase 2 the menu, palette and context
  menu open times, the palette filter on 500 commands and 2000 atoms, the
  tooltip, and that Settings and the shortcut sheet load on first open; for
  Phase 3 the worker's bond finding at 10k atoms, box selection at 8000, edit
  to paint at 1000, the symmetry requests during a drag and the server's
  threads and CPU meanwhile, ordered nudges and undos, the virtualized Atoms
  panel, hidden and freed viewers, and the server's memory with the cache
  filled; for Phase 4, `test_bench_files.py`: the batch file details on 2050
  files and their speed-up over one worker, Clean in the pool and four at
  once, cancel latencies, the parse cache, packed payload size, the 34 MB
  output's summary and step, the pool's workers, nice, threads, memory and
  idle shutdown, the mixed CPU load against the cap, and with
  `MACE_GUI_BENCH_BIG=1` the 85k-atom supercell) and
  the resources used: CPU workers, peak
  memory and the memory added on each GPU. It writes `bench_report.json`, into
  `$MACE_GUI_BENCH_OUT` if set; `MACE_GUI_BENCH_TUNNEL=<url>` repeats the
  browser rows over an ssh tunnel. Run it on its own and niced
  (`nice -n 10 pytest -m guibench tests/gui/bench`), never with `pytest -n`.
- Phase 4: `tests/gui/test_tasks_*.py` (the pool and tasks), `test_files_*.py`
  (cache, held documents, export, the banner, prefetch, the trajectory
  summary), `test_parse_api.py` and `test_drop_paste_e2e.py` (drop, import,
  paste), `test_recent_e2e.py`, `test_palette_rank_e2e.py`, `test_outindex.py`
  (the index against `read_out_frames` on every example output and a generated
  34 MB run), `test_builders_*.py`, `test_preview_lifecycle.py`,
  `test_builder_entry_e2e.py`, and `test_builder_decks.py`, which builds the
  same supercells, slab and vacancy with ase alone, converts them with the real
  `mace convert`, and checks the GUI's decks against them line for line.
- Phase 5: `test_setup_api.py`, `test_setup_routes.py` and `test_setup_e2e.py`
  (Setup's fields against the CLI's questions, and the matrix of calculation
  types where the GUI's saved settings give the manual, interactive route's
  deck byte for byte), `test_cli_tasks.py` (CLI runs as niced, cancelable
  process groups), `test_text_window.py`, `test_db_api.py` and
  `test_db_e2e.py` (a generated 10k-material database), `test_plot_e2e.py`,
  `test_freq_api.py` and `test_freq_e2e.py` (IR, Raman and FREQ modes on
  copies of `test/FREQ`, skipped without the corpus), `test_compare_api.py`
  and `test_compare_e2e.py` (100 shuffled MgO supercells matched exactly, the
  four looks, the dark ghost's contrast), `test_coordination_e2e.py`,
  `test_svg_export_e2e.py`, and `test_p5_route_security.py` (the token, Host,
  header and path rules of every new route, with a check that fails on a
  route it does not list). `bench/test_bench_analysis.py` measures the
  database, text, plotly, Setup, FREQ, polyhedra and SVG rows, and with
  `MACE_GUI_BENCH_TUNNEL` the same through ssh to an HPCC dev node, including
  Run Preflight with the serial `crystal` there.
- **Interpreters.** The GUI tests are run on Python 3.12 and 3.11, with ase,
  spglib, plotly and selenium. CI installs only `requirements-test.txt`, so
  the browser tests and those needing ase skip there; the browser tests need
  `pip install -r requirements-gui-test.txt` and Firefox.
- The headless Firefox of these tests draws WebGL in software (Mesa's
  llvmpipe), so they never use a GPU; a test checks the renderer.

## Slabs, polymers and molecules (1.1.6)

**Editing off 3D.** A slab, polymer or molecule (a CRYSTAL `.out`, a P1
SLAB/POLYMER/MOLECULE `.d12`, an extended XYZ with `pbc="T T F"`, an `.xyz`
opened without a box, or a built supercell) is now an editable document, like
a crystal: Move, Add, Delete, Replace, Bond, Measure, Edit Cell, Supercell and
Clean Geometry all work on it. Editing is P1 only (the symmetry-aware Sym mode
needs a 3D space group; its menu item says so). Along a non-periodic direction
nothing wraps and there are no periodic images: an atom moved past a slab's
top stays there, and the atoms at a slab's two faces are never bonded through
its vacuum. Edit Cell scales the atoms in the periodic directions only (the
non-periodic vector is the drawing box). Clean keeps no space group there
(the result says so) and **Relax the cell** relaxes only the periodic
directions; a molecule keeps its box. Saving as CIF or POSCAR, which have no
periodicity flags, puts the structure in a box with 15 Å of vacuum along each
non-periodic direction; extended XYZ keeps `pbc`. A SLAB/POLYMER deck in a
layer or rod group other than 1 is still drawn from the `.out` of its run
(MACE has no layer/rod group tables).

**Vacuum (Inspector › Display).** The old Vacuum checkbox (which padded only
the classic views) is a five-way choice for a structure with at least one
non-periodic direction, kept per document (Settings › Viewport › Vacuum spacing
is the default):

| Mode | The box along a non-periodic direction | Repeats (display range, supercell) |
|---|---|---|
| Bounding box (default) | the atoms' extent + 1 Å on each side: a small fake cell | that far apart |
| Open (infinite) | no walls: the cell outline lies in the atoms' mid-plane | none |
| File-defined | the file's own vacuum: a CRYSTAL output's formal 500 Å, an extended XYZ's or a cube's box | that far apart |
| CRYSTAL 500 Å | CRYSTAL's convention | 500 Å apart |
| Custom | the atoms' extent + the vacuum you type (0-1000 Å) | that far apart |

A display range or a supercell is 1 along a non-periodic direction unless you
ask for more there (a custom range such as 1×1×3, or Build › Supercell's n3);
then the structure repeats at the active mode's spacing. In Build › Supercell
the result stays non-periodic there (a slab of two layers with the vacuum
between them). A cube document defaults to File-defined (its isosurfaces fill
its box). The camera frames the atoms, not a 500 Å box. Display only: nothing
here changes a deck or a file. The command line has the same supercells:
`mace supercell INPUT N1 N2 N3 --vacuum bbox|file|crystal|open|GAP`.

The optimization view (Overlay, Two-tone, Side by side, Displacement map) and
the FREQ modes view draw the same box: each frame's box is its Vacuum mode
around that frame's atoms, and a display range repeats whole cells along the
periodic axes only. A molecule gets its box there too (none in Open mode).

**Bonds of heavy elements.** Bonds are found with the covalent radii (Cordero
2008, completed to Og with Pyykkö's single-bond radii), except for the heavy
cations whose ionic bonds are far longer than their covalent radius: Pd, Ag,
Cd, In, Pt, Au, Hg, Tl, Pb and Bi bond to an anion out to about the sum of the
Shannon ionic radii (Ag-Cl in AgCl, Pb-O in PbTiO3, Tl-Cl were not drawn
before). Metal-metal pairs still use the covalent radii (Settings › Viewport ›
Bonds between metals). They are bonded only with that setting on, or in a
structure of metals only (an alloy, an intermetallic), as `mace analyze bonds`
and Compare's bond table count them: before 1.1.6 such a structure drew no
bond at all. The van der Waals radius set fills the elements 3Dmol lacks
(Sc-Co, Y-Rh, La-Lu, Hf-Ir, the early actinides) with Alvarez's radii.

**Calculate Bonds… (Build menu, the viewport's menu).** The bond criteria of
the drawn structure, as Materials Studio's Calculate Bonds sets them. Two atoms
are bonded when the shortest bond < d < tolerance × (r<sub>i</sub> +
r<sub>j</sub>): **Tolerance** (0.8 to 1.6, default 1.15), **Shortest bond**
(default 0.1 Å; overlapping sites closer than it are not bonded), **Bonds
between metals in compounds**, and **Across cell boundaries** (drawn atoms
only, or with their bonded neighbors or whole molecules outside the range, as
Inspector › Range & boundary › Bonded atoms outside). **Element pairs** gives a
pair a rule of its own: a Min and/or Max length in Å (a Max bonds the pair up
to it, a metal pair too), or **Never**. The preview counts the bonds per cell
and between the drawn atoms, and lists each pair's count, shortest and longest
bond (**Rule…** starts a rule for that pair at its longest bond + 0.05 Å).
**Recalculate** (or Enter) gives this structure those criteria (kept with the
document, as its display range is); **Save as Default** writes them to
Settings › Viewport for every structure without its own; **Reset to Defaults**
makes this structure follow Settings again. The same rule draws the bonds,
finds Bonded atoms outside, Select › Bonded Neighbors and fragments, the
coordination numbers and the polyhedra (`geom/core.js bondRule`), and Clean
Geometry holds the bonds it makes. Bonds added or removed by hand with the Bond
tool (B) stay on top of the criteria; the dialog counts them and **Clear Hand
Edits** drops them. The server's tables (Compare's bond table, `mace analyze`)
use the same rule (`structures.bond_pairs`) at their own fixed criteria:
Compare at 1.2, `mace analyze` at its `--tolerance` (1.15 by default).

**Coordination polyhedra.** A polyhedron is drawn around a cation with 4 to 12
anion ligands: bonded neighbors more electronegative (Pauling) than it that are
nonmetals (H, C, N, O, F, P, S, Cl, As, Se, Br, Te, I) with no neighbor more
electronegative themselves: the cations of ionic and polar solids (TiO6 in
BaTiO3, ZnS4, SiO4 next to metal polyhedra, SO4, a MOF's metal nodes), never an
anion, an atom bonded to its own element (Si in Si), a C-H carbon or an
intermetallic. By default carbon and the large cations whose polyhedra share a
face with a smaller polyhedron of another element (BaO12 around TiO6, the A site
of a perovskite) are left out; Inspector › Rendering › Polyhedra › Centers draws
them. The rule lives in one place (`geom/core.js polyhedronCenters`, with
`structures.polyhedron_centers` its server twin for analyses).

**fort.34 handedness.** CRYSTAL23 writes a fort.34's operators as x' = R x + t
and lists every atom of the cell; read the other way round, an enantiomorphic
screw axis (3₁/3₂, 4₁/4₃, 6₁/6₅, 6₂/6₄) adds the mirror image's atoms, so MACE
uses the reading that generates fewer atoms. A file that lists only the
asymmetric unit closes under both readings with the same atom count: MACE picks
the one whose space group matches the file's group record; without that record
it keeps CRYSTAL23's reading and the structure's source carries a warning naming
both groups.

## Calculators (mace calc)

**Calculation › Calculators…** runs `mace calc` on the active structure: a
single point, an optimization, frequencies, molecular dynamics or partial
charges with UFF/UFF4MOF, EQeq, xTB (GFN2, GFN1, GFN-FF), DFTB+ or an ML
potential (MACE-MP from ACEsuit's mace-torch, which is unrelated to this MACE
tool; ORB-v3; MatterSim; CHGNet). The panel stays beside the
viewport; Run starts a job in the process pool (progress and Cancel in the
panel and the Tasks panel). The answer lists the energy, maximum force,
pressure, charges by element, frequencies with the harmonic thermochemistry,
or the MD summary, with warnings and the citation. An optimized or MD
structure opens as a new document; *Save files next to the document* writes
the CLI's files into the document's folder. Backends that are not installed
show the commands that set them up; **ML Potentials…** opens Settings › ML
potentials (nothing is installed or downloaded until you ask there). For
OrbMol-v2 and UMA's omol head the panel shows **Charge** and **Spin
multiplicity**: blank reads them from the deck or output the document came
from, else charge 0 and the lowest multiplicity, and the result says which
were used and where they came from. UMA shows its **Task head** (blank: omat
for solids, slabs and polymers, omol for molecules). A MACE-OFF result is
labeled "MACE-OFF (academic license)". ML models load once, in the server's
ML worker (the one Clean
uses), on GPU 0 by default (the Device field: Auto, CPU or a GPU). Niche
fields (optimizer, displacement, friction, seed, backend options) are
Advanced. See [calculators.md](calculators.md).

## Inspector detail, Setup tabs, display cells and normal-mode amplitude (1.1.6)

**Inspector | Setup tabs.** A tab strip sits on top of the Inspector column. Pick **Setup** to open
Calculation Setup and **Inspector** to go back. Before 1.1.6 the only way back was to press Setup
again. The palette's **Show Inspector** does the same.

**Basic and Advanced.** The strip's **Basic | Advanced** switch sets how much the Inspector shows.
Basic keeps the settings most runs need. Advanced adds the niche ones: fractional range bounds,
Cell + 10%, bonded atoms outside, the finer lighting (light strength and direction, ambient light,
shininess, ambient occlusion, outline), the symmetry tolerance, the Hall symbol and operation count,
Scale atoms with cell, the FREQ arrow length and XYZ export, and the classic editing forms. A
section that hides settings in Basic ends with an **N more in Advanced** button. The choice is kept
per kind of view (structure, optimization, modes, volume, plot, and so on) in Settings
(`inspector.tier`). View › **Advanced Inspector** toggles it. To start every view in Advanced, set
`MACE_GUI_INSPECTOR_TIER=advanced` in the server's environment.

**Inspector detail.** Settings › Appearance › **Inspector detail** picks how a view starts.
**Remember per view** (the default) keeps the choice per kind of view as above. **Always Basic**
and **Always Advanced** open every view in that tier. The Basic | Advanced switch, View ›
**Advanced Inspector** and **N more in Advanced** still work there, but only for the view on
screen: showing another view brings back the chosen tier. They leave the setting and the
per-view choices alone, so switching back to Remember per view restores those choices. The
setting wins over the per-view choices, and both win over `MACE_GUI_INSPECTOR_TIER`, which only
fills in for kinds of view you have not picked a tier for.

**Display cells (N × M × L).** Inspector › Range & boundary starts with a **Cells** picker: 1 to 10
cells along a, b and c. The **=** button links the periodic directions, and **Presets** offers
1×1×1, 2×2×2, 3×3×3, 2×2×1, 3×3×1 and 4×4×1. The same picker appears in the FREQ Mode section, which
replaces its old 2×2×2 checkbox, and under View › Display Range › **N × M × L…**. The optimization
view now repeats whole cells too, moving each image with its atom. Only the drawing changes. To
build a real supercell, use Build › Supercell…, which the picker's **Build…** button opens.

**Normal-mode amplitude.** CRYSTAL prints each mode as *NORMAL MODES NORMALIZED TO CLASSICAL
AMPLITUDES (IN BOHR)*. That is each atom's displacement at the classical turning point of the
vibrational ground state, x = √(ħ/ωμ) (CRYSTAL23 manual, SCANMODE). Inspector › Mode › Amplitude
offers two scales:

- **Physical** (the default): ×1 draws that true displacement, usually 0.01 to 0.2 Å. The
  **Exaggeration** slider (×1 to ×100) scales it.
- **Fit 0.5 Å**: draws every mode so its largest atom moves 0.5 Å × Amplitude. This is the drawing
  1.1.5 called "1×", now with an honest name.

The view's header always gives the scale (for example "physical amplitude" or "exaggerated ×10"),
and the note under the slider gives the true and the drawn displacement. The arrows have their own
**Arrow length** setting. Settings › Files › Normal-mode amplitude picks the default scale. In the
CLI, `mace plotting --freq FILE` draws the physical amplitude: `--amplitude N` exaggerates it,
`--normalize` fits the largest atom to 1 Å, and `--freq-supercell N1 N2 N3` repeats cells.
`mace animate FILE --mode N` draws the physical amplitude too (`--amplitude-scale normalized`
for the 0.5 Å fit), and File › Export Animation… from the file uses the view's scale.

**Activity bar.** Right-click the activity bar to hide a view, move it up or down, show hidden
views again, reset the bar or hide the whole bar. You can also drag a button onto another to
reorder them. View › **Activity Bar** brings a hidden bar back.

## Rotate and Move gizmos (1.1.6)

**Rotate tool (R, toolbar after Orbit).** Compass wheels sit on the structure view while the tool is on:
one wheel per Cartesian axis (x red, y green, z blue; the far half drawn faint) and an outer wheel in the
screen plane. What turns is the **target** in the panel at the bottom right: the selected atoms (about the
**pivot**, one undo step that names it, "Rotate 3 Atoms about Ti1") when the structure is editable and atoms
are selected, otherwise the view (about its center). It works in structure documents only: in an
optimization run, the Modes, volume and orbital views it is grayed out with the reason (drag to turn those
views, or View › Rotate 90°, Alt+arrows), and a Rotate tool left on goes back to Select there.

- Drag a wheel to turn about that axis only; the outer wheel turns in-plane, about the line of sight.
  Drag anywhere else for a free, out-of-plane turn (horizontal drag: about the screen's vertical, or the
  structure's up in Turntable mode; vertical drag: about the screen's horizontal).
- The readout shows each angle from the orientation the drag began at, both ways round
  (`+30.0° / −330.0°`): on the wheel, in the chip, and as compass dials in the panel.
- Snapping is to whole steps of that reference turn: Settings › Viewport › **Angle snap** (Off, 15°, 30°,
  45°, 90° or **Custom**, any angle up to 180° in **Custom angle**; the panel's **Snap** and **Step** set the
  same settings).
- Modifier keys while dragging: **Ctrl** (⌘ on a Mac) inverts the snap; **Shift** snaps every angle
  (≈88°, ≈43° → 90°, 45°); **Alt** keeps only the larger turn, snapped (→ 90°, 0°), and pressing **Alt again**
  (or **Tab**) switches to the other one (→ 0°, 45°); **Shift+Alt** turns in-plane only. Press Alt after the button goes down
  (Alt+drag from the start orbits, as in every tool). **Esc** or a right click cancels; letting go applies.
- **Pivot** (the panel): what the selected atoms turn about. **Selection centroid** (the default),
  **Center of mass** (standard atomic masses), both of the atoms made whole across the cell's faces;
  **Cell center**, the middle of the unit cell, (½, ½, ½), and for a slab, polymer or molecule the middle of
  the box drawn along its open directions (Inspector › Display › Vacuum); **Origin**; or **Picked atom**:
  click **Pick**, then click any atom (it need not be selected; the selection stays). The list then reads
  "Picked atom (Ti1)", the atom's symbol and number as Labels › Both show it; a picked atom outside the
  selection is taken at its image nearest the selection's centroid. A marker (a ring with a crosshair, a
  diamond for an atom) and its name show the pivot while the tool is on, and the wheels sit on it. Each
  document keeps its own pivot; **Settings › Viewport › Rotate pivot** is the default
  (`viewport.rotatePivot`). One selected atom about its own centroid would turn in place, so the panel says
  "One atom turns in place; pick another pivot". The view's turns and the Move tool are unchanged.
- The panel also turns by a typed angle about a typed axis: x, y, z, a, b, c, a*, b*, c*, the line of
  sight, the screen's axes, or a lattice direction [u v w] (a Cartesian vector for a molecule), through the
  same pivot. **↺ Apply** turns counterclockwise seen from the axis's tip, **↻** the other way.
- A click without a drag selects, as in the Select tool (after **Pick**: makes that atom the pivot; Pick
  again stops picking). **Keys** in the panel lists the modifiers; the
  panel's ▾ hides its fields (remembered).

**Move tool gizmo.** With atoms selected, the Move tool shows an AutoCAD-style 3D Move handle at the
selection's centroid: drag an **arrow** to move along that axis only, a **square** to move in that plane
only, the **center ring** freely in the screen plane. The arrows follow the panel's frame: **Cartesian**
(x, y, z) or **Cell** (a, b, c). Ctrl snaps the drag (0.05 of a cell axis in the Cell frame, 0.1 Å in the
Cartesian one). The panel takes an exact displacement, Δx Δy Δz in Å or Δa Δb Δc as fractions of the cell
axes: the atoms preview it as you type, **Apply** (or Enter) moves them in one undo step. G, the X/Y/Z axis
locks and typed offsets work as before.

**Orbit readout.** Any orbit drag (right drag, Alt+drag, the Orbit tool) shows in the chip the turn since
the drag began (horizontal and vertical with a snap or in Turntable mode), and Ctrl held during the drag
inverts the angle snap there too.

**Command line.** `mace transform INPUT --atoms LIST --rotate DEG --axis AXIS
[--about centroid|com|cell|origin|atom:N|x,y,z] [--translate DX DY DZ | --translate-frac DA DB DC] [-o OUTPUT]`
does the same turns and moves on a structure file (CIF, D12, outputs and the formats the GUI opens), with the
same axis names and the same pivots (`atom:N` is 1-based, as the panel's "Picked atom (Ti1)"; `cell` uses the
Bounding box along a slab's open directions); `mace transform --help` lists the options.

## Cube slices, print patterns and animations (1.1.6)

### Slicing a cube along any plane

The Volume document's **Slice** section (Inspector) shows where the slice is taken from:

- **Plane**: a grid plane along a, b or c (the grid's own points, exact values, as before), a
  **Miller plane (h k l)** (normal h a* + k b* + l c*), or the plane **normal to a direction
  [u v w]** (u a + v b + w c). The chips (1 0 0) ... (1 −1 0) pick the common planes.
- **Mini unit-cell preview**: the cell box seen from the 3D view's direction (it turns with the
  view), the plane cut through it and its normal. Drag the plane to move it along its normal;
  Shift-drag (or a right-drag) turns it; the wheel and the arrow keys step it (PageUp/PageDown
  tilt, Left/Right twist). In the 3D view itself, **Alt-drag** moves the plane and Alt+Shift-drag
  turns it without turning the camera.
- **Position** (0 to 1 across the box along the normal), and in the Advanced tab **Tilt**,
  **Twist** and **Untilt the Plane**. Turning a grid plane makes it the same Miller plane.
- While the plane moves, the 2D map is sampled from the grid the 3D view holds ("preview from the
  3D view's grid"); when it stops, the server's full-grid plane replaces it. The map's heading gives
  the plane, its position in Å, and for (h k l) the equation h x + k y + l z = t and d(hkl).

The box is the grid's: a periodic axis spans its period, a non-periodic one (a slab's c, a
molecule) its points. The same plane on the command line: `mace slice CUBE --hkl 1 1 0 --pos 0.4
[--tilt 10 --twist -5] -o plane.png` (or `.svg`, `.pdf`, `.csv`), `--uvw U V W`, or
`--axis c --index 40` for a grid plane.

The volume view's atoms now follow the Inspector's Display section (Atom size, Bond radius,
radii, colors, style), as the structure view's do.

### Contrast on any background

The slice plane, the isosurfaces, the cell and the Brillouin zone view draw in the 3D view's own
tokens, so a White background in a dark theme (or Black in a light one) keeps them legible; on a
custom background color a mark that would fall under 3:1 is moved toward black or white until it
stands out. The Brillouin zone view now fills its panel when it opens.

### Print patterns for black-and-white output

**Settings › Viewport › Print patterns** (or View: Cycle Print Patterns) puts slashes,
cross-hatching, stipples and other patterns on the atoms' visible surfaces: **Similar grays**
patterns only elements whose colors are too close in lightness to tell apart (useful with the
Grayscale print style), **Every element** gives each element its own. Export Image (PNG) has its
own Print patterns choice, and animations draw them too. The element legend (in an exported
image's corner, or Export Element Legend of the open structure) puts each element's pattern on its
swatch. `mace animate --grayscale --patterns auto` does the same headless.

### Export Animation

**File › Export Animation…** writes the moving view as a GIF (always) or WebM / MP4 (when the
server machine has `ffmpeg` on PATH, or `MACE_FFMPEG`; the dialog says when it does not):

- **This view**: the FREQ document's normal mode (one cycle, 24 frames; the still view's arrows are
  left out) or an optimization's steps, drawn by the view itself at the chosen size.
- **Rendered by MACE**: the document's file drawn on the server exactly as `mace animate` draws
  it: CRYSTAL FREQ and OPT outputs, and MD or NEB trajectories (.xyz, .extxyz, .traj via ASE).
  The bonds are the ones the view draws, by the same rule and the document's Calculate Bonds…
  criteria (`mace animate`: the default criteria): no Mg-Mg in MgO, a heavy cation's bonds.
- Frames per second, plays (forever, once, n times), size (the view's, presets or custom) and a
  scale factor, background (the view's, transparent for GIF/WebM, white or black), print patterns,
  and keep every n-th step.

On the command line: `mace animate run_freq.out --mode 7 [--format mp4 --fps 30]`,
`mace animate opt.out --every 2 --size 1280x720`, `mace animate md.extxyz --format webm
--background transparent`, `mace animate --check` (which formats this machine writes),
`mace animate run_freq.out --list-modes`.

## Structure Analysis (porosity, diffraction, topology)

Analysis › Structure Analysis opens a panel for the structure on screen, with
one tab per tool (each is also `mace analyze <tool>`; see
[Structure analysis](analysis.md) for the methods, their validation and their
limits). The panel floats over the Inspector at about a third of the window's
width (380 to 600 px); drag its left edge to make it wider (double-click the
edge for its own width again), or its header to move it. A moved panel stays
inside the window when the window gets smaller or the panel opens again.

- **Porosity:** pore sizes (Di, Df, Dif), accessible and non-accessible
  volumes and surface areas, density and channel dimensionality for an
  adsorbate (He, H2, H2O, CO2, Ar, N2, CH4 or any probe radius). Compute Pores
  draws what the probe can occupy in the 3D view: blue where it can reach,
  orange in closed pockets. The surfaces come from the same classification
  as the numbers, and the results name the structure, the method, the probe
  and the radii. Advanced: atomic radii (CCDC or UFF), grid spacing, and a
  Zeo++ run alongside when Zeo++ is installed.
- **Powder Pattern:** the simulated pattern and peak list for any X-ray
  source, compared with a measured pattern if you pick one; download the
  pattern (.xy) and the peaks (.csv).
- **Peaks & Fit:** peak search, indexing (from scratch, or with the open
  structure's cell), and Pawley or Rietveld fits against a measured pattern
  from this computer or the served folder, with the fit, difference curve and
  reflection ticks drawn.
- **Topology:** the underlying net, its RCSR symbol, coordination sequences,
  point symbols and the degree of interpenetration; Show Net in 3D draws the
  simplified net. RCSR's list of 3-periodic nets is bundled, and the result
  shows the RCSR citation; Import uses a newer 3dall.txt you download.
- **Polyhedra**, **Bonds & Angles**, **Morphology** (BFDH or Wulff, drawn at
  the cell center) and **Planes** (a Miller plane with a position slider, the
  bonds it cuts, and a cleavage ranking).

Each tool runs as a background task. The panel can be moved and stays open
while you work in the 3D view. A run goes on until you cancel it: switching
documents, switching tabs or closing the panel leaves it running. The tab's
status line shows its progress and a Cancel button, and the Tasks tab lists
it by tool and structure ("Porosity: SOD") with the same Cancel. Results belong to the document they were
computed for: switching documents shows that document's results (or none),
and pores, nets, shapes and planes are drawn only on the structure they were
computed for, and hidden after it is edited. Each document keeps its own
overlays: switch back and its pores are drawn again. Clear 3D Overlays removes
the overlays of the structure on screen.

### The Analysis tab

The bottom panel's **Analysis** tab (or All Results in the Structure Analysis
panel) lists every Structure Analysis result of the document in front, newest
first; **All open documents** lists every structure's. Each result says what
it was computed with: the sorbate (probe) and its radius, the radii set, the
grid spacing and the engine for porosity; the radiation and wavelength, 2θ
range and peak shape for a powder pattern; the simplification and bond
tolerance for topology; and so on. Its key numbers follow, then its own table:

- **Porosity:** one row per channel system and per closed pocket: accessible
  or closed, dimensionality (and the axes a channel runs along), the volume
  the probe can occupy, the surface it touches, the largest included sphere
  Di and its center. The footer gives the totals (they equal POAV, ASA, PONAV
  and NASA) and the averages. Click a row to draw that channel or pocket in
  the highlight color with the others dimmed and frame it in the 3D view;
  click it again to go back.
- **Powder pattern:** the peak list. **Peaks & Fit:** the peaks, indexing
  solutions or the fit's cell and R factors. **Topology:** each net's vertex
  classes, coordination sequences and point symbols. **Polyhedra:** every
  polyhedron. **Bonds & Angles**, **Morphology** (the facets) and **Planes**.

Actions: **Show in 3D / Hide in 3D**, **Run Again** (the same tool and
settings on the same structure, a background task), **Copy Command** (the
`mace analyze` line that gives the same numbers), **JSON** and **CSV** (with
the settings in the file), **Delete Mesh Data** (porosity: frees the
surfaces' memory, the numbers stay) and **Delete Result**. Results last for
the session of the page.

### 3D Overlays in the Inspector

Once Structure Analysis has drawn something, the Inspector of a structure has
a **3D Overlays** section: one row per overlay on that structure (pores, a
net, a crystal shape, Miller planes), each with its eye (show or hide) and
Delete (the overlay and its mesh data go; the result stays in the Analysis
tab). Click a row to edit it: for pores the accessible channels and the closed
pockets apart (each with its color), Color pockets individually, Solid or
Wireframe, the opacity, and every channel and pocket on its own (show it, or
highlight and frame it); for a net, a shape or planes their color, style and
opacity. Pores start from Settings › Viewport › Pore surfaces; what you change
here stays with that structure's overlay (in bold), and Reset takes it back to
the Settings. The controls under the Porosity results change the Settings.

Pores, Miller planes and a net belong to the cell, so they are drawn in each
cell of the display range (the N × M × L cells picker or a display-range
preset), as the atoms are; a slab or polymer has no copies along its open
axes. The surface is built once and each cell draws a copy moved by its
lattice vector, so a 3×3×3 range costs about what one cell does. The channels
are cut at the cell faces and join the next cell's (in every style), each
closed pocket is drawn whole once per cell (a cage on a cell corner around
that corner) in the same color in every cell, and a Miller plane lying on a
cell face is drawn once. A crystal shape is one particle and is not repeated.
Overlays are drawn in at most 64 cells; a larger range says so in the row's
editor.

Styles (1.1.6) for pores, a crystal shape and planes: **Solid** (at the
opacity: below 1 it is translucent), **Wireframe**, **Solid + edges** (the
triangle edges a shade darker over the surface) and **Points** (the vertices
as dots); for pores also **Smooth** or **Flat** shading and the walls drawn:
**Far + faint near** (each pore's far walls, and its near walls faintly, so the
atoms in front stay sharp; the default), **Far only** or **Both sides**.
Settings › Viewport › Pore surface style, shading and walls are the
defaults. Settings › Viewport › **Drop the faint near walls while dragging**
(off by default) hides the near walls while you turn the view, for fewer
triangles a frame, and brings them back when you let go.

### Mesh quality

Every surface the 3D view draws has a **Mesh quality** (1.1.6): pore surfaces,
volume isosurfaces and a net's spheres (a crystal shape and Miller planes are
flat polygons, exact at any quality). Settings › Viewport › Mesh quality is
the default for new results; the Inspector's 3D Overlays section (and the
Volume view's Surface section) changes one overlay, and only that one is made
again.

The pore surfaces have their own default, Settings › Viewport › **Pore surface
mesh quality** (Preview unless you choose another; Settings › Viewport › Mesh
quality does not change it): they are the heaviest surfaces the view draws,
two walls each, repeated over every cell of a supercell. **Preview** is a tier
below Draft for pore surfaces only (a volume has no Preview). Raise one
overlay's in the Inspector; that structure's overlay keeps your choice for its
next result, and Export Image's Publication option makes the picture's
surfaces fine whatever the view uses.

| Preset | For | Pore surfaces (render grid, refine, smoothing, triangle cap) | Volume grid |
|---|---|---|---|
| Preview | pore surfaces only: many cells, slow graphics (the pores' default) | 24, ×1, 1.4, 60 k | – |
| Draft | screening many materials | 40, ×1, 0.7, 150 k | 0.25 M points |
| Standard | the default (the surfaces before presets) | 72, ×1, 0.7, 600 k | Settings › Volume quality |
| High | a close look | 96, ×2, 0.8, 1.2 M | 2 M points |
| Publication | a figure | 128, ×3, 0.9, 2.4 M | 8 M points, refined ×2 |

The pores' render grid is at most that many points per cell axis and may be
finer than the analysis grid by the refine factor (the distance to the atoms
is exact there; the analysis numbers never change). A new quality is made by
the server without the analysis when it made that result (otherwise the
analysis runs again with the same numbers), in a background task; the
Inspector shows what the surfaces were made at ("Made at Standard: 48×48×48
grid, 72 k triangles") and, while a new mesh is coming, its estimated size.
Surfaces made before stay in memory, so going back is instant. **Advanced**
holds the numbers; a number typed there wins over the preset (within hard
limits: render grid 160, refine ×4, 3 M triangles; volumes 8 M points) and
Reset takes it back. On LTA (N₂, 0.25 Å grid) Draft gives 51 k triangles,
Standard 72 k, High 292 k in about 5 s and Publication 517 k in about 11 s.
On the example suite's sodalite (N₂, 0.25 Å grid) a cell is small enough that
the analysis grid, not the preset, limits Draft and Standard (both 33 k
triangles); Preview gives 15 k and Publication 289 k. Over a 3×3×3 range the
surfaces cost no extra build, but every copy is drawn each frame: in a browser
rendering in software, Preview adds about 0.08 s a frame to the atoms' 0.13 s,
Draft and Standard about 0.4 s and Publication about 2 s.

Export Image can **Re-mesh the pore surfaces at Publication quality for this
image**: the surfaces are made again densely for the picture only and the
view's own come back after it.

### Analyze Structures… (many files)

Select structure files in the Explorer (Ctrl+click, Shift+click for a range),
right-click and pick **Analyze Structures…** (also on a folder's menu, under
Analysis › Structure Analysis, and the panel's Batch… button). Pick the tool
(porosity, powder pattern, topology, polyhedra, bonds, morphology or planes)
and its settings, tick or untick files in the list, and **Run**. Each
structure is one background task in the Tasks tab (Cancel stops it), a few at
a time: Files at a time, 1 to 4 (Settings › Advanced › Analyze Structures:
files at once, 2 by default), and never more than the CPU budget. A porosity
batch keeps the 3D surfaces only when you tick Keep the 3D pore surfaces.

The bottom panel's Analysis tab opens on **Batches**: one row per structure
with its state and key numbers, the same columns as
`mace analyze <tool> FILE... --csv`. Export CSV, JSON and Copy Command (the
command line for the same files) are there, Cancel the Rest stops what has
not finished, and Open opens a file, whose result is then listed with it.

## Plot viewers (1.1.6)

- **Explorer chips.** Plot data is outlined: `BAND` and `DOS` on the `.BAND.DAT`/`.DOSS.DAT` files and on a fort.25
  (`*.f25`, `fort.25`) of BAND or DOSS blocks, `MAP` on a fort.25 map; the text output of a properties run (BAND,
  DOSS, TRANSPORT, CHARGE+POTENTIAL, ORBITALS) is a dashed `out` (log text only: its tooltip names the calculation,
  and the data file beside it plots).
- **fort.25 band and DOS plots.** A fort.25 opens as the plot of its run and draws what the `.DAT` draws (checked
  against the 76 BAND and 80 DOSS fort.25 of the test corpus). CRYSTAL appends to fort.25, so a file that holds
  several runs draws the last (Reference ▾ › Run picks another; `mace plotting --f25-run N`). A phonon fort.25
  (FREQCALC with BANDS or PDOS; read as phonon when the run's output beside it says so, else Reference ▾ ›
  fort.25 holds, `--f25-domain phonon`) plots frequencies in cm-1, THz or meV with no Fermi level. The phonon
  layout is the manual's (Appendix D); the corpus has no phonon fort.25 to check it against.
- **Toolbar menus of a band or DOS plot.** `Style ▾` (plot style, background, color pair), `Traces ▾` (band
  style, β line, DOS projections, spin layout, total, smoothing), `Axes & Ranges ▾` (energy window, axis ranges,
  line width, Reset Zoom), `Reference ▾` (Fermi shift, energy zero: Auto, Fermi level or Vacuum, the dotted Fermi
  line) and `Labels ▾` (k-point labels, the "up"/"down" words). Plot options in the Inspector has the same
  sections; niche rows are Advanced. CLI: `--energy-reference`, `--fermi-shift`, `--no-fermi-line`,
  `--no-k-labels`, `--no-spin-labels`, `--units cm-1|THz` (phonon).
- **Plot Gallery pages.** A set larger than a page is paged (Panes per page 25 to 500, default 200; the pager
  appears when there is more than one page); `mace plotting --gallery KIND FOLDER --max N --page P` draws the same page.

## Phonons, mechanical properties and the relaxation time (1.1.6)

Their decks are written in **Calculation Setup**, their runs are read under
**Analysis**. Each step runs the CLI command it shows (as a cancelable task in
the parent's folder), so the GUI and the terminal write the same files.

- **Setup › 4 ELASTIC, 5 EOS, 6 QHA, 7 CPHF** (`mace opt2d12 --calc-type
  ELASTIC|EOS|QHA|CPHF --elastic-settings JSON`): with a
  finished CRYSTAL `.out` in Setup, these are calculation types after 3 FREQ,
  numbered as the CLI's menu. Step 2 shows the sub-keywords of the module (4
  ELASTIC picks ELASTCON, ELAPIEZO, PIEZOCON, PIEZOCP or PHOTOELA, numbered as
  the CLI's "Mechanical property module" menu; only the keywords the manual
  lists for it); step 3 **Write Deck** writes the deck from the parent output,
  shows the command and the log, then **Open Deck**, **Submit…** and **Reveal
  in Explorer** (Setup stays on the output). The deck keeps the
  parent's functional, basis, spin and SCF settings. **Calculation › Regular
  Calculations › 4 ELASTIC** (to 7 CPHF) opens Setup there on the output in
  front; on a CIF, a deck or an unfinished run it is grayed, and `mace opt2d12`
  refuses such a parent with its reasons (as for 8 Phonopy).
- **Setup › 8 Phonopy** (`mace opt2d12 --calc-type PHONOPY`): the displaced
  supercells of a finished 3D OPT or SP output (supercell, displacement, ±
  displacements, CRY_SYM, SHRINK), written by opt2d12 with the parent's
  settings. **Generate Displaced Decks** writes the set (its folder is named as
  opt2d12 names a deck of the parent, `<stem>_ph222_<functional>_optimized`);
  **Submit the Jobs…** hands its folder to Submit with the walltime opt2d12
  suggested (the jobs are never tracked), and **Phonons with Phonopy…** opens
  the set under Analysis. **Calculation › Regular Calculations › 8 Finite-displacement
  phonons with phonopy (PHONOPY)** opens Setup there. phonopy must be installed where the server runs (`pip install
  'phonopy<5'`); Setup says so when it is not.
- **Analysis › Phonons with Phonopy…** (`mace analyze phonopy`): *1 · Forces*
  lists a set's jobs (not run, running, done, failed) and hands the folder to
  **Submit…**; *2 · Phonons* runs `mace analyze phonopy` (band path from
  SeeK-path or your own, DOS mesh, temperatures, NAC from a Γ FREQCALC with
  INTENS + INTCPHF) and plots the phonon bands, the DOS and the heat capacity,
  entropy and free energy, in THz or cm⁻¹ (the completion callback of the
  set's last job runs the same analysis with its defaults). Right-click a set
  folder or one of its files (`mace_phonopy.json`, `band.yaml`, ...) in the
  Explorer: **Phonons with Phonopy…** opens it there. **Open Setup ›
  Phonopy** goes to the deck step.
- **Analysis › Mechanical Properties Results…** (`mace analyze elastic`):
  reads a finished run (opened on one, it reads it at once): the stiffness
  and compliance tensors, Voigt/Reuss/Hill moduli with Born stability, sound
  velocities and Debye temperature, seismic velocities, piezoelectric,
  Pockels and piezo-optic tensors, the EOS fits with the E(V) points, and the
  QHA V(T) and thermal expansion. **Open Setup › ELASTIC** goes to the deck
  step; **Open as Document** opens the output's results document (below).
- **Results documents of ELASTIC, EOS, QHA and CPHF runs.** The Explorer
  marks these outputs `ELASTIC`, `EOS`, `QHA` and `CPHF` (outlined: a click
  opens their results, as a transport file opens as its plots and a cube as
  its volume). The document sits in the plot area with a strip of tabs:
  **Results** (the numbers of `mace analyze elastic`) and the run's plots.
  - *ELASTIC* (ELASTCON, ELAPIEZO, PIEZOCON, PIEZOCP, PHOTOELA): the bulk,
    shear and Young's moduli, Poisson's ratio, the Debye temperature and
    Born stability at the top; C and S; the Voigt/Reuss/Hill table; each
    Born condition of the crystal system with its value (Mouhat & Coudert,
    and every eigenvalue of C positive); sound velocities and CRYSTAL's
    Christoffel table; the directional extremes; the piezoelectric e and d,
    Pockels p and piezo-optic π tensors. **Young's modulus** and **Linear
    compressibility** draw E(n) and β(n) as a **3D Surface** (the distance
    from the center is the value along that direction; drag to turn it) or
    as **Planes** (polar cuts in xy, xz and yz). A slab shows its in-plane
    stiffness in N/m, E_x, E_y, ν_xy, G_xy, the layer modulus and its 2D
    Born conditions, and its in-plane polar plot.
    The **Directional** table gives the minimum and maximum of E(n), β(n),
    G(n, m) and ν(n, m) with their directions (G and ν depend on n and on m
    perpendicular to it: the extremes over m, with the m that gives each,
    as ELATE reports them; Gaillac, Pullumbi & Coudert 2016) and max/min,
    and flags negative linear compressibility and auxetic (ν < 0)
    directions. **Anisotropy**: the universal index A^U, Zener's A (cubic),
    Chung–Buessem A_C and the ratios E_max/E_min, G_max/G_min,
    ν_max/ν_min. **Mechanical character**: Pugh's ratio B/G (ductile above
    1.75), the Cauchy pressure C12 − C44 (cubic) or the Cauchy relations,
    and the Vickers hardness of Chen (2011) and Tian (2012), each with its
    formula and reference. **Thermal**: the Debye temperature with Clarke's
    and Cahill's minimum thermal conductivity. The sound velocities add the
    Christoffel extremes (phase and group) of the qL, qS1 and qS2 waves. An
    ELAPIEZO or PIEZOCON run shows the longitudinal piezoelectric
    coefficient d_L(n) (its largest value and direction, along x, y, z) and
    the coupling k² along x, y, z and its largest value: type the static,
    stress-free permittivity ε^T and **Apply** (without it, the run's own
    electronic ε∞ when it has one, an upper bound). More tabs: **Shear
    modulus** and **Poisson's ratio** (3D Surface: the minimum envelope
    green, the maximum a translucent blue shell, negative values red;
    Planes: the minimum and maximum curves in each plane), **Sound
    velocities** (phase velocities solid, group velocities dotted, in xy,
    xz and yz) and **Piezoelectric** (d_L(n) as a surface colored by its
    sign, or cuts; when it vanishes in the principal planes, as in a cubic
    −43m crystal, the cut through z and its largest direction). A slab
    adds its in-plane G(θ) and ν(θ) and the 2D anisotropy A^SU.
  - *EOS*: V0, B0, B′ and E0 (Birch-Murnaghan), the formula and Z, the four
    fits, their **Fit quality** (each fit's residual RMS and largest
    residual, rated by the RMS over the spread of the computed energies), the
    computed points; **E(V)** (relative to the lowest point, with the four
    fitted curves), **P(V)**, **B(P)** (B = −V dP/dV of each fit),
    **V/V0(P)**, **Residuals** (E_computed − E_fit of each point and fit) and
    **Enthalpy** (H = E + PV per formula unit). On the Enthalpy tab, **Add
    phase…** takes another EOS output of the same composition (type its path
    or pick it from the list of the folder's EOS runs): the plot becomes ΔH =
    H − H_ref of each added phase, per formula unit, with the transition
    pressure P_t marked where the curves cross (within the pressures both
    runs cover); the fit it uses is chosen beside the phases (Birch-Murnaghan
    by default), × removes a phase, and Copy CLI Command gives
    `mace analyze elastic A.out --plot enthalpy --compare B.out`. Phases of
    another composition are refused.
  - *QHA*: the values at 300 K and a table of the temperatures; **V(T)**,
    **V(T)/V0** (the volume and a, b, c relative to the lowest temperature),
    **Thermal expansion** (α_V and the linear α of a, b, c), **B(T)** (B_T,
    B_S) and **Heat capacity** (Cp − Cv, with each volume's harmonic Cv when
    the output prints them). When the output has each volume's harmonic
    thermodynamics (a complete QHA output), Results adds S, Cv, Cp and γ_th
    at 300 K and a table of F, S, Cv, Cp and γ_th along V(T) (each volume's
    F = EL + E0 + ET − TS, S and Cv interpolated to V(T); Cp = Cv + (Cp −
    Cv)), and the tabs **Grüneisen γ(T)** (CRYSTAL's mode γ and γ_th =
    α_V B_T V/Cv), **F(T)** and **S(T)**; Heat capacity then draws Cp and Cv
    along V(T). **F(V)** draws the free energy against volume at up to seven
    temperatures from CRYSTAL's PRESSURE-VOLUME-TEMPERATURE table (else a
    polynomial through the computed volumes), with each curve's minimum
    traced (the zero-pressure V(T)) and the computed volumes marked.
  - *CPHF*: ε∞ (principal values, or one value for an isotropic crystal),
    the refractive index, the ε∞, α and χ(1) tensors, and the non-zero β and
    γ components; a DYNAMIC run adds **ε(ω)**. **Principal axes of ε∞**: the
    eigenvalues and eigenvectors of the tensor, n = √ε along each, the
    optical class (isotropic, uniaxial, biaxial), the birefringence Δn, and
    a uniaxial crystal's n_o, n_e, sign and optic axis (a biaxial one's 2V);
    **Index ellipsoid** draws the indicatrix (semi-axes n along the principal
    axes, true proportions). A crystal without inversion (THIRD or FOURTH)
    shows **Second-harmonic generation**: d11, d22, d33, the largest
    longitudinal d_L(n) and |d(n)| with their directions and the powder
    averages ⟨d_L²⟩^½ and ⟨|d|²⟩^½ (d = χ(2)/2 in pm/V, Kleinman-symmetric,
    no phase matching), and the **SHG |d(n)|** tab (a surface, or cuts with
    d_L dashed). A DYNAMIC run adds **n(ω)** (n and, with DAMPING, κ of each
    diagonal component) and **Absorption** (α = 2ωκ/c, with DAMPING).

  - *FREQ with INTENS + INTCPHF*: the normal-mode view's **Dielectric**
    button (shown when the output prints Born charges) opens the output's
    dielectric results: the Born effective charge tensor of every atom with
    its Tr(Z*)/3 and the acoustic sum, ε∞, ε0 = ε∞ + ε_vib (CRYSTAL's sum of
    the modes' oscillator strengths) with their principal values, and the
    LO-TO splitting at Γ along x, y, z and the principal axes of ε∞. The
    runs print no LO frequencies: they are computed from the mode Born
    charges, frequencies and ε∞ the output prints (the non-analytic term),
    with the Lyddane–Sachs–Teller check beside each table. A crystal's
    principal axes and index ellipsoid are those of a CPHF run.

  Results that are not physical get a red **Warning** line at the top of
  Results (the report's `WARNING:` lines): a mechanically unstable C (its
  derived numbers are shown but not stored in the database), a dielectric
  tensor less symmetric than the space group allows (a *symmetry mismatch*
  badge on the principal axes; its principal n, 2V, optic sign and the LO-TO
  splitting are not stored), an unstable CPHF (CRYSTAL's DIIS test), and a
  diverged or incomplete β or γ table (no SHG, and never "centrosymmetric").
  A Chen hardness below zero shows as 0 with the formula's value. Directions
  are Cartesian (`x+y`, `x−z`), lattice directions (`[1 1 0]`) only in a cubic
  crystal; the LO-TO tables pair each LO with its own TO mode.

  The Inspector shows **Calculation** (the module, its options, NUMDERIV
  and STEPSIZE as the output prints them, the nuclear relaxation, the EOS
  volume range, the QHA temperatures and volumes, the CPHF response and
  frequency, the functional and basis; each row's tooltip says whether it
  came from the `.out`, the `.d12` beside it or CRYSTAL's default) above the
  **Run** card. The toolbar is the plots': **Copy Report** on Results,
  **Export ▾** with PNG, SVG, CSV, Copy as TSV and **Copy CLI Command**
  (`mace analyze elastic FILE --plot youngs --shape surface`, ...). An
  output with no results yet (still running) says so and offers Open as
  Text and Open as Structure; right-click › Open as Structure shows the
  geometry of any of them.
- **Analysis › Transport Relaxation Time…** (`mace tau`): *Estimate* computes
  τ and the mobility from m*, E1 and C (or from a mobility); *Strain Decks*
  writes the strained single points of an SP deck and hands them to Submit;
  *Strained Runs* finds a finished series in a folder, fits E1 and C, and
  reports τ with its provenance and the `RELAXTIM` line for a TRANSPORT deck.
  Leave m* empty for the automatic masses (`--m-star auto`): m* from the
  band curvature of the BAND run beside the series (or the BAND.DAT given),
  m_d from a DOSS (the DOSS field, or one beside the series). The fields then
  say where each came from ("from BAND: 0.297 mₑ (Γ→X)"), the mass report
  heads the result and its warnings join the fit's. **Save to Database**
  stores τ and the mobility (per carrier and strain axis) with the series'
  unstrained run in its folder's `materials.db`, where the Properties tab
  shows them (rows by `mace-gui`, which a re-extraction keeps); without a
  database it says so and the fit stays on screen.
