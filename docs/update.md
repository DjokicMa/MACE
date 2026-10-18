# Updating MACE: `mace update`

`mace update` moves a MACE git checkout to a newer version, here and (with
`--remote`) on your cluster. It never forces anything, never resets, and
never throws away your changes or commits: when it cannot move forward
cleanly, it says why and changes nothing.

```
mace update --check                      # what is available, nothing changes
mace update                              # the newest release (asks first)
mace update --channel preview --dry-run  # what a preview update would do
mace update --remote --remote-only       # the cluster's MACE (default connect profile)
mace update --remote hpcc --local --yes  # this computer and the cluster, no questions
```

## Channels

| Channel | Moves to |
|---|---|
| `release` (default) | the newest final tag, `vX.Y.Z` |
| `preview` | the newest `vX.Y.Z-preview` or `vX.Y.Z` tag, whichever is newer |
| `latest` | the head of `origin/main` (unreleased work) |

Each release is tagged `vX.Y.Z-preview` first and `vX.Y.Z` when it is final;
the final is newer than its own preview.

## What it does

1. Fetches branches and tags from `origin`. A tag that was moved there (a
   release retagged, e.g. the preview's `v1.1.5` became `v1.1.5-preview` and
   `v1.1.5` now names the final) is updated in your checkout too, with a
   note; plain `git fetch` keeps the stale tag. Only tags and `origin`'s
   branches change.
2. Refuses, changing nothing, when:
   - tracked files have local changes (they are listed; `git stash` keeps
     them aside, `git stash pop` brings them back after the update);
     untracked files (outputs, notes) do not count;
   - your checkout has commits the target lacks (not a fast-forward);
   - you cannot write the checkout (a read-only or shared install): ask
     whoever installed it, or use a clone of your own. `--check` still
     works there: it asks `origin` with `git ls-remote` and fetches nothing.

   A target older than your checkout (for example `main` after a `git pull`,
   ahead of the newest release) is not a failure: nothing to do, exit 0, and
   the hint `mace update --channel latest`.
3. Shows the current and the target version and the CHANGELOG sections in
   between, warns when SLURM jobs of yours are queued or running (their
   completion callbacks run the code in this checkout, so update between
   batches when you can) or a `mace gui` server runs, then asks. `--yes`
   answers for you; without a terminal, `--yes` is required. `--dry-run`
   stops before changing anything.
4. Moves the checkout:
   - `latest`: `git merge --ff-only origin/main` on `main` (switching to
     `main` first when needed);
   - a tag on your branch's upstream (the usual case, `main`):
     `git merge --ff-only <tag>`, so `git pull` keeps working afterwards;
   - otherwise (for example a checkout left on a tag): a new branch
     `mace-<tag>` at the tag. `git checkout main && git pull` returns to
     main later.
5. When `requirements.txt` changed, installs the new requirements or tells
   you how (see below). A changed `requirements-optional.txt` (packages of
   optional features) gets one line of its own; the developers' test
   requirement files are never mentioned.
6. Checks the result with `mace_cli --version`.

## New requirements: pip or your own environment

When the update changed `requirements.txt`, `mace update` lists the
packages added or changed, and names the Python that runs MACE
and what kind it is: a virtual environment, a conda environment (by name), or
a system or module Python. In a terminal it then asks:

```
  [1] run it now: /home/me/mace-env/bin/python -m pip install -r /home/me/MACE/requirements.txt
  [2] I'll do it myself, or set up my own environment
Install the new requirements? [1/2] (default 1)
```

The default is 1 in a virtual or conda environment and 2 for a system or
module Python. For those the command uses `pip install --user`, because
installing into the Python's own site-packages needs administrator rights and
can break other tools. Choice 2 prints the command and a line on making your
own venv or conda environment.

- Without a terminal, or with `--yes`, pip never runs: the command is
  printed.
- `--pip` runs it without asking (also with `--yes`); `--no-pip` prints it
  without asking. `--remote` passes either one to the cluster.
- If pip fails, the update stays: the code is updated, the packages are not.
  The message says so and repeats the command to rerun after fixing the
  error. `MACE_PYTHON` names another Python than the one running
  `mace update`.

## From MACE GUI

MACE GUI checks with `mace update --check --json` (at start, then every 6
hours; Settings › Server) and shows "MACE x.y.z available" in the status
bar. Its dialog offers **Update Now and Restart** (the GUI runs `mace update
--yes --no-pip`, or `--pip` when you tick the box, then restarts in place),
**Update When I Sign Out** (the update runs after the server stopped),
**Not Now** and **Skip This Version**. The refusals are the ones above,
word for word; a server started with `--insecure-network` never updates
from the browser. docs/gui.md (Updating MACE from the GUI) has the details.

`--check --json` prints one JSON object for `--channel` (this computer's
checkout only): `current` and `target` (version, ref, commits ahead),
`state` (`current`, `available`, `newer`, `diverged`), `changelog` (the
CHANGELOG lines in between), `refusal` and `refusal_kind` (`local_changes`,
`not_ff`, `not_writable`, `no_origin`, `fetch`, ...), `local_changes` and
`stash`, `requirements` (files, changed lines, `pip`, the kind of Python),
`busy` (your queued SLURM jobs) and `notes` (moved tags). It changes nothing
but `origin`'s refs, like `--check`, and exits 0 whenever it printed the
object.

## On the cluster: `--remote`

`--remote [PROFILE]` uses a `mace gui connect` profile (the default one when
no name is given) and runs the same procedure in the cluster's checkout over
ssh, on the login node (it shares your home with the nodes). The output
streams here. The cluster's checkout is the one its `mace` command runs from
(the profile's `--mace`), else `$MACE_HOME`, `~/mace` or `~/MACE`. When that
MACE has `mace update`, it runs; a cluster MACE without it (older than
1.1.6) works too: the update code is sent along with the command, compressed
(about 15 KB once base64-encoded), and runs with the cluster's `python3` from 3.6 on.

With a terminal, ssh is interactive, so a 2FA prompt works; without one, it
uses `BatchMode=yes`. `--remote` updates only the cluster; add `--local` to
update this computer too (`--remote-only` says the same as `--remote` alone).
`mace update --check --remote --local` reports both. When the update here
stops (a refusal, exit 2), the cluster is not updated, and `mace update` says
so.

## Without MACE on your computer

The standalone connect scripts update the cluster's MACE before starting the
GUI: `mace-gui-connect.sh --update-remote [release|preview|latest]` or
`mace-gui-connect.ps1 -UpdateRemote release`. They need MACE 1.1.6 or newer
on the cluster; update an older one once by hand
(`git -C <MACE folder> pull`). Downloading the scripts: see
[gui.md](gui.md#only-mace-on-the-cluster).
