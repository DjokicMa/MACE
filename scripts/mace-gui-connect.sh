#!/usr/bin/env bash
# mace-gui-connect.sh: start MACE GUI on a cluster node and open it in this
# computer's browser, with one ssh session. The standalone twin of
# `mace gui connect` for a computer without MACE (Linux, macOS, the local
# terminal of MobaXterm); needs bash 3.2+ and ssh. Same flags, same profiles
# file (~/.config/mace/connect.toml), same protocol.
#
#   mace-gui-connect.sh [PROFILE] [--gateway G] [--node N] [--dir D] ...
#   mace-gui-connect.sh --help
#
# Ctrl+C (or closing this window) stops the server on the node; GUI tasks
# that are still running keep it alive until they finish, and running this
# script again reattaches. See docs/gui.md (First run from your computer).

# >>> mace-gui-connect defaults >>>
D_GATEWAY=""
D_JUMP=""
D_NODE=""
D_HOP="jump"
D_DIR=""
D_MACE="mace"
D_SETUP=""
D_CONDA=""
D_LOCAL_PORT="8765"
D_SSH_OPTIONS=()
D_ROOTS=()
D_OPEN="1"
# <<< mace-gui-connect defaults <<<

set -u

usage() {
  cat <<'EOF'
Usage: mace-gui-connect.sh [PROFILE] [options]

Start MACE GUI on a cluster node and open it in this computer's browser.

Options:
  --gateway G        your cluster's login host (user@host or an ~/.ssh/config Host)
  --jump J           bastion host(s) before the gateway, comma separated
  --node N           the node to run on, e.g. a dev node (default: the gateway)
  --hop H            jump (ssh -J, default) or nested (the gateway logs in to the node)
  --dir D            folder to serve on the node (default: your home folder)
  --mace CMD         the mace command on the node (default: mace)
  --setup LINE       shell line(s) run first on the node, e.g. 'module load Python'
  --conda ENV        a conda environment to activate on the node
  --local-port P     port on this computer (default 8765; the next free one when busy)
  --ssh-option OPT   an extra ssh argument (repeatable)
  --root DIR         a folder File > Open Folder may switch to (repeatable)
  --no-open          print the URL; do not open a browser
  --browser B        app: MACE GUI in its own window of Brave, Chrome, Chromium, Edge
                     or Vivaldi (the default when one is installed); tab: a browser
                     tab (also the default when $BROWSER is set)
  --dry-run          print the ssh command (one argument per line) and exit
  --check            check ssh, its config, keys, agent and permissions, try each
                     hop with only `true`, and say which works (read-only)
  --no-net           with --check: no ssh logins or DNS lookups
  --fix-config       print a suggested ~/.ssh/config block (never written)
  --update-remote [release|preview|latest]
                     run mace update on the cluster first (default: release)
  -h, --help         this help

Values come from the flags, then the profile in ~/.config/mace/connect.toml
(written by mace gui connect --save), then the defaults in this script. In
MobaXterm, Cygwin or Git Bash, when that file is missing, the profile file
of Windows' mace gui connect (%APPDATA%\mace\connect.toml) is read instead.
--hop auto works as jump here (mace gui connect falls back to nested by
itself); --slurm is not supported (use mace gui connect --slurm).
EOF
}

# ------------------------------------------------------------------ options
PROFILE=""
F_GATEWAY="" F_JUMP="" F_NODE="" F_HOP="" F_DIR="" F_MACE="" F_SETUP="" F_CONDA="" F_LOCAL_PORT="" F_OPEN=""
F_BROWSER=""
F_SSH_OPTIONS=() F_ROOTS=()
HAVE_SSH_OPTIONS=0 HAVE_ROOTS=0
DRY_RUN=0 CHECK=0 NO_NET=0 FIX_CONFIG=0 UPDATE_REMOTE=""
need() { [ $# -ge 2 ] || { echo "mace-gui-connect: $1 needs a value" >&2; exit 2; }; }
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --gateway) need "$@"; F_GATEWAY="$2"; shift 2 ;;
    --jump) need "$@"; F_JUMP="$2"; shift 2 ;;
    --node) need "$@"; F_NODE="$2"; shift 2 ;;
    --hop) need "$@"; F_HOP="$2"; shift 2 ;;
    --dir) need "$@"; F_DIR="$2"; shift 2 ;;
    --mace) need "$@"; F_MACE="$2"; shift 2 ;;
    --setup) need "$@"; F_SETUP="$2"; shift 2 ;;
    --conda) need "$@"; F_CONDA="$2"; shift 2 ;;
    --local-port) need "$@"; F_LOCAL_PORT="$2"; shift 2 ;;
    --ssh-option) need "$@"; F_SSH_OPTIONS+=("$2"); HAVE_SSH_OPTIONS=1; shift 2 ;;
    --ssh-option=*) F_SSH_OPTIONS+=("${1#--ssh-option=}"); HAVE_SSH_OPTIONS=1; shift ;;
    --root) need "$@"; F_ROOTS+=("$2"); HAVE_ROOTS=1; shift 2 ;;
    --no-open) F_OPEN="0"; shift ;;
    --browser|--browser=*)
      if [ "$1" = --browser ]; then need "$@"; F_BROWSER="$2"; shift 2; else F_BROWSER="${1#--browser=}"; shift; fi
      case "$F_BROWSER" in app|tab) ;; *) echo "mace-gui-connect: --browser takes app or tab" >&2; exit 2 ;; esac ;;
    --dry-run) DRY_RUN=1; shift ;;
    --check) CHECK=1; shift ;;
    --no-net) NO_NET=1; shift ;;
    --fix-config) FIX_CONFIG=1; shift ;;
    --update-remote)
      case "${2:-}" in release|preview|latest) UPDATE_REMOTE="$2"; shift 2 ;; *) UPDATE_REMOTE=release; shift ;; esac ;;
    --slurm) echo "mace-gui-connect: --slurm is not supported by this script" >&2; exit 2 ;;
    -*) echo "mace-gui-connect: unknown option $1 (see --help)" >&2; exit 2 ;;
    *) PROFILE="$1"; shift ;;
  esac
done

# ------------------------------------------------------------------ profile (flat TOML subset)
P_GATEWAY="" P_JUMP="" P_NODE="" P_HOP="" P_DIR="" P_MACE="" P_SETUP="" P_CONDA="" P_LOCAL_PORT="" P_OPEN=""
P_SSH_OPTIONS=() P_ROOTS=()
P_HAVE_SSH_OPTIONS=0 P_HAVE_ROOTS=0
CONF="${XDG_CONFIG_HOME:-$HOME/.config}/mace/connect.toml"

winpath() {  # a Windows path (C:\Users\me) -> this shell's form of it (cygpath, else /cygdrive/c, /drives/c, /c)
  local p d rest
  if command -v cygpath >/dev/null 2>&1; then cygpath -u "$1"; return; fi
  p="$(printf '%s' "$1" | tr '\\' /)"
  case "$p" in
    [A-Za-z]:/*|[A-Za-z]:) d="$(printf '%s' "${p%%:*}" | tr '[:upper:]' '[:lower:]')"; rest="${p#?:}" ;;
    *) printf '%s' "$p"; return ;;
  esac
  for root in /cygdrive /drives ""; do
    if [ -d "$root/$d" ]; then printf '%s' "$root/$d$rest"; return; fi
  done
  printf '/cygdrive/%s%s' "$d" "$rest"
}

case "$(uname -s 2>/dev/null)" in
  CYGWIN*|MSYS*|MINGW*)
    # Git Bash (MSYS) rewrites arguments that look like paths (/c/x, //x) when
    # it starts a Windows program such as Windows' ssh.exe: pass them as is
    case "$(uname -s 2>/dev/null)" in MSYS*|MINGW*) export MSYS2_ARG_CONV_EXCL="*" ;; esac
    # MobaXterm and Cygwin have their own home folder: when it has no profile
    # file, read the one mace gui connect --save writes on Windows
    if [ ! -f "$CONF" ] && [ -n "${APPDATA:-}" ]; then
      WIN_CONF="$(winpath "$APPDATA")/mace/connect.toml"
      if [ -f "$WIN_CONF" ]; then
        echo "mace-gui-connect: no $CONF here; reading the profiles of Windows' mace gui connect in $WIN_CONF" >&2
        CONF="$WIN_CONF"
      fi
    fi ;;
esac

utf8() {  # $1 = a code point in hex -> U8 (its UTF-8 bytes)
  local cp fmt
  cp=$((16#$1))
  if [ "$cp" -lt 128 ]; then fmt="$(printf '\\%03o' "$cp")"
  elif [ "$cp" -lt 2048 ]; then fmt="$(printf '\\%03o\\%03o' $((192 | cp >> 6)) $((128 | (cp & 63))))"
  elif [ "$cp" -lt 65536 ]; then
    fmt="$(printf '\\%03o\\%03o\\%03o' $((224 | cp >> 12)) $((128 | ((cp >> 6) & 63))) $((128 | (cp & 63))))"
  else
    fmt="$(printf '\\%03o\\%03o\\%03o\\%03o' $((240 | cp >> 18)) $((128 | ((cp >> 12) & 63))) \
           $((128 | ((cp >> 6) & 63))) $((128 | (cp & 63))))"
  fi
  # shellcheck disable=SC2059  # fmt holds octal escapes only
  printf -v U8 "$fmt"
}

unbasic() {  # the inside of a TOML basic string -> UB, its value, in one pass as tomllib reads it
  # (replacing \\ and \t one after the other read "C:\\Users\\me\\tmp" as C:\Users\me<TAB>mp)
  local s="$1" out="" c hex n
  while [ -n "$s" ]; do
    case "$s" in
      \\*)
        c="${s:1:1}"
        case "$c" in
          b) out="$out"$'\b'; s="${s:2}" ;;
          t) out="$out"$'\t'; s="${s:2}" ;;
          n) out="$out"$'\n'; s="${s:2}" ;;
          f) out="$out"$'\f'; s="${s:2}" ;;
          r) out="$out"$'\r'; s="${s:2}" ;;
          u|U)
            if [ "$c" = u ]; then n=4; else n=8; fi
            hex="${s:2:$n}"
            case "$hex" in
              *[!0-9A-Fa-f]*) out="$out$c"; s="${s:2}" ;;
              *) if [ ${#hex} -eq "$n" ]; then utf8 "$hex"; out="$out$U8"; s="${s:$((n + 2))}"
                 else out="$out$c"; s="${s:2}"; fi ;;
            esac ;;
          "") out="$out\\"; s="" ;;
          *) out="$out$c"; s="${s:2}" ;;       # \" and \\ stand for the character
        esac ;;
      *) c="${s%%\\*}"; out="$out$c"; s="${s:${#c}}" ;;
    esac
  done
  UB="$out"    # not printed: $(...) would drop a trailing newline
}

str_list() {  # one-line array of basic strings -> one value per line
  printf '%s' "$1" | grep -o '"\([^"\\]\|\\.\)*"' | sed -e 's/^"//' -e 's/"$//'
}

read_profile() {
  [ -f "$CONF" ] || return 0
  local want="$1" section="" n=0 line key val default=""
  # first pass: the [defaults] profile
  if [ -z "$want" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
      line="${line%$'\r'}"     # a CRLF file (written on Windows)
      line="${line#"${line%%[![:space:]]*}"}"
      line="${line%"${line##*[![:space:]]}"}"
      case "$line" in
        "["*"]") section="${line#[}"; section="${section%]}" ;;
        profile*=*) if [ "$section" = "defaults" ]; then
                      val="${line#*=}"; val="${val#"${val%%[![:space:]]*}"}"
                      val="${val#\"}"; default="${val%\"*}"
                    fi ;;
      esac
    done < "$CONF"
    want="$default"
  fi
  [ -n "$want" ] || return 0
  section=""
  while IFS= read -r line || [ -n "$line" ]; do
    n=$((n + 1))
    line="${line%$'\r'}"       # a CRLF file (written on Windows)
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    case "$line" in
      ""|"#"*) continue ;;
      "["*"]") section="${line#[}"; section="${section%]}"; continue ;;
    esac
    [ "$section" = "profiles.$want" ] || continue
    key="${line%%=*}"; key="${key%"${key##*[![:space:]]}"}"
    val="${line#*=}"; val="${val#"${val%%[![:space:]]*}"}"
    case "$val" in
      \"*\") val="${val#\"}"; val="${val%\"}"; unbasic "$val"; val="$UB" ;;
      true) val="1" ;;
      false) val="0" ;;
      \[*\]) ;;
      [0-9]*) ;;
      *) echo "mace-gui-connect: connect.toml uses a form this script cannot read (line $n); pass the values as flags" >&2
         continue ;;
    esac
    case "$key" in
      gateway) P_GATEWAY="$val" ;;
      jump) P_JUMP="$val" ;;
      node) P_NODE="$val" ;;
      hop) P_HOP="$val" ;;
      dir) P_DIR="$val" ;;
      mace) P_MACE="$val" ;;
      setup) P_SETUP="$val" ;;
      conda) P_CONDA="$val" ;;
      local_port) P_LOCAL_PORT="$val" ;;
      open) P_OPEN="$val" ;;
      ssh_options|roots)
        case "$val" in
          \[*\]) ;;
          *) echo "mace-gui-connect: connect.toml uses a form this script cannot read (line $n); pass the values as flags" >&2
             continue ;;
        esac
        local item
        while IFS= read -r item; do
          [ -n "$item" ] || continue
          unbasic "$item"; item="$UB"
          if [ "$key" = ssh_options ]; then P_SSH_OPTIONS+=("$item"); else P_ROOTS+=("$item"); fi
        done <<EOF2
$(str_list "$val")
EOF2
        if [ "$key" = ssh_options ]; then P_HAVE_SSH_OPTIONS=1; else P_HAVE_ROOTS=1; fi ;;
    esac
  done < "$CONF"
}
read_profile "$PROFILE"

pick() {  # flag, else profile, else default
  if [ -n "$1" ]; then printf '%s' "$1"; elif [ -n "$2" ]; then printf '%s' "$2"; else printf '%s' "$3"; fi
}
GATEWAY="$(pick "$F_GATEWAY" "$P_GATEWAY" "$D_GATEWAY")"
JUMP="$(pick "$F_JUMP" "$P_JUMP" "$D_JUMP")"
NODE="$(pick "$F_NODE" "$P_NODE" "$D_NODE")"
HOP="$(pick "$F_HOP" "$P_HOP" "$D_HOP")"
DIR="$(pick "$F_DIR" "$P_DIR" "$D_DIR")"
MACE="$(pick "$F_MACE" "$P_MACE" "$D_MACE")"
SETUP="$(pick "$F_SETUP" "$P_SETUP" "$D_SETUP")"
CONDA="$(pick "$F_CONDA" "$P_CONDA" "$D_CONDA")"
LOCAL_PORT="$(pick "$F_LOCAL_PORT" "$P_LOCAL_PORT" "$D_LOCAL_PORT")"
OPEN="$(pick "$F_OPEN" "$P_OPEN" "$D_OPEN")"
# app (its own window) unless --browser tab, or a $BROWSER of yours without --browser (appwindow.py)
BROWSER_MODE="$F_BROWSER"
[ -n "$BROWSER_MODE" ] || { if [ -n "${BROWSER:-}" ]; then BROWSER_MODE=tab; else BROWSER_MODE=app; fi; }
SSH_OPTIONS=() ROOTS=()
if [ "$HAVE_SSH_OPTIONS" = 1 ]; then SSH_OPTIONS=("${F_SSH_OPTIONS[@]}")
elif [ "$P_HAVE_SSH_OPTIONS" = 1 ]; then SSH_OPTIONS=("${P_SSH_OPTIONS[@]}")
elif [ ${#D_SSH_OPTIONS[@]} -gt 0 ]; then SSH_OPTIONS=("${D_SSH_OPTIONS[@]}"); fi
if [ "$HAVE_ROOTS" = 1 ]; then ROOTS=("${F_ROOTS[@]}")
elif [ "$P_HAVE_ROOTS" = 1 ]; then ROOTS=("${P_ROOTS[@]}")
elif [ ${#D_ROOTS[@]} -gt 0 ]; then ROOTS=("${D_ROOTS[@]}"); fi
# A path under THIS computer's home folder is an unquoted ~ that your shell
# expanded here (--mace ~/mace/mace_cli arrives as /home/you/mace/mace_cli,
# which the node lacks when its home is elsewhere): sent as ~/... instead,
# with a note. The same rule as mace gui connect (localize_home_paths). A
# leading // (//home/you/x) is a literal path on the node, sent as /home/you/x.
MOVED=""
loc() {  # $1 = a path on the node -> LOC
  local h="${HOME%/}"
  LOC="$1"
  [ -n "$h" ] || return 0
  # shellcheck disable=SC2088  # a literal ~, expanded on the node
  case "$1" in
    "//"*) return 0 ;;  # a literal path on the node (sent as /...): never rewritten
    "$h") LOC="~" ;;
    "$h"/*) LOC="~/${1#"$h"/}" ;;
    *) return 0 ;;
  esac
  MOVED="${MOVED:+$MOVED, }$1 as $LOC"
  [ -n "${MOVED_FIRST:-}" ] || MOVED_FIRST="$LOC"
}
MOVED_FIRST=""
if [ -n "$DIR" ]; then loc "$DIR"; DIR="$LOC"; fi
if [ -n "$CONDA" ]; then loc "$CONDA"; CONDA="$LOC"; fi
case "$MACE" in
  *"${HOME%/}"*)
    set -f
    NEW_MACE=""
    for w in $MACE; do loc "$w"; NEW_MACE="${NEW_MACE:+$NEW_MACE }$LOC"; done
    set +f
    MACE="$NEW_MACE" ;;
esac
if [ ${#ROOTS[@]} -gt 0 ]; then
  for i in "${!ROOTS[@]}"; do loc "${ROOTS[$i]}"; ROOTS[i]="$LOC"; done
fi
if [ -n "$MOVED" ]; then
  echo "Note: treated $MOVED on the remote (your shell turned ~ into this computer's home folder; quote the ~, as '$MOVED_FIRST', to avoid this)." >&2
fi
HOP_AUTO=0
[ "$HOP" = auto ] && { HOP=jump; HOP_AUTO=1; }
case "$HOP" in jump|nested) ;; *) echo "mace-gui-connect: --hop must be jump or nested" >&2; exit 2 ;; esac
if [ -z "$GATEWAY" ] && [ -z "$NODE" ]; then
  echo "mace-gui-connect: pass --gateway <login host> (and usually --node <dev node> and --dir <folder>)," >&2
  echo "or save a profile with mace gui connect --save." >&2
  exit 2
fi

# A gateway such as user@university.edu is an email address, not a login host
# (user@login.university.edu): flagged when the part after @ is two labels
# (name.tld) and no ~/.ssh/config Host pattern matches it. The same heuristic
# as mace gui connect (gateway_looks_like_email in connect.py, which explains
# the tradeoff); a warning only.
email_like() {
  local gw host pat
  case "$1" in *@*) ;; *) return 1 ;; esac
  gw="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  host="${gw##*@}"
  printf '%s' "$host" | grep -Eq '^[a-z0-9-]+[.][a-z]{2,}$' || return 1
  [ -f "$HOME/.ssh/config" ] || return 0
  while IFS= read -r pat; do
    pat="${pat#\"}"; pat="${pat%\"}"
    case "$pat" in ""|"!"*) continue ;; esac
    # shellcheck disable=SC2254  # the Host pattern is a glob on purpose
    case "$host" in $pat) return 1 ;; esac
    # shellcheck disable=SC2254
    case "$gw" in $pat) return 1 ;; esac
  done <<EOF3
$(sed -n -E 's/^[[:space:]]*[Hh][Oo][Ss][Tt]([[:space:]]+|[[:space:]]*=[[:space:]]*)//p' "$HOME/.ssh/config" | tr '[:upper:]' '[:lower:]' | tr -s ' \t' '\n\n')
EOF3
  return 0
}
if [ -n "$GATEWAY" ] && [ "$CHECK$FIX_CONFIG" = 00 ] && email_like "$GATEWAY"; then
  echo "Warning: the gateway $GATEWAY looks like an email address. It should be user@login-host (your cluster's login node, e.g. netid@login.example.edu) or a Host alias from ~/.ssh/config. Connecting anyway; to fix the profile in $CONF, run: mace gui connect ${PROFILE:+$PROFILE }--gateway <user@login-host> --save (or pass --gateway to this script)." >&2
fi

# ------------------------------------------------------------------ the command (as mace gui connect builds it)
shq() {  # Python's shlex.quote
  case "$1" in
    "") printf "''" ;;
    *[!a-zA-Z0-9_@%+=:,./-]*) printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\"'\"'/g")" ;;
    *) printf '%s' "$1" ;;
  esac
}
home_path() {
  # shellcheck disable=SC2088  # a literal "~/" from the user, rewritten to "$HOME" on the node
  case "$1" in
    "~") printf '"$HOME"' ;;
    "~/"*) printf '"$HOME"/%s' "$(shq "${1#\~/}")" ;;
    "//"*) shq "${1#/}" ;;
    *) shq "$1" ;;
  esac
}

# When the mace command is not on the node's PATH (the default "mace", or a
# one-word path that is not there), the node's shell tries mace_cli in that
# path's folder, $MACE_HOME, ~/mace and ~/MACE, and prints MACE_GUI_USING
# <path> (mace_fallback in connect.py; the same shell code).
mace_fallback() {  # -> FIND
  local w own c cands t
  FIND=""
  set -f
  # shellcheck disable=SC2086  # the mace command's words
  set -- $MACE
  set +f
  [ $# -eq 1 ] || return 0
  w="$1"
  # shellcheck disable=SC2016  # shell code for the node, expanded there
  if [ "$w" = mace ]; then
    t='command -v "$M" >/dev/null 2>&1'
    cands='${MACE_HOME:+"$MACE_HOME"/mace_cli} "$HOME"/mace/mace_cli "$HOME"/MACE/mace_cli'
  else
    case "$w" in */mace_cli|*/) return 0 ;; */*) ;; *) return 0 ;; esac
    t='[ -x "$M" ]'
    own="$(home_path "${w%/*}/mace_cli")"
    cands="$own"
    for c in '${MACE_HOME:+"$MACE_HOME"/mace_cli}' '"$HOME"/mace/mace_cli' '"$HOME"/MACE/mace_cli'; do
      [ "$c" = "$own" ] || cands="$cands $c"
    done
  fi
  FIND="M=$(home_path "$w"); $t || for c in $cands; do if [ -x \"\$c\" ]; then M=\$c; echo \"MACE_GUI_USING \$c\"; break; fi; done"
}

remote_command() {  # $1 = remote port
  local parts="" script cmd w
  if [ -n "$SETUP" ]; then parts="${SETUP%;}"; fi
  if [ -n "$CONDA" ]; then parts="${parts:+$parts; }conda activate $(home_path "$CONDA")"; fi
  cmd="exec"
  [ -z "$NODE" ] && cmd="$cmd env MACE_GUI_MAX_WORKERS=2"
  mace_fallback
  if [ -n "$FIND" ]; then
    cmd="$cmd \"\$M\""
  else
    # the mace command may be several words (e.g. python3 ~/mace/mace_cli)
    set -f
    for w in $MACE; do cmd="$cmd $(home_path "$w")"; done
    set +f
  fi
  cmd="$cmd --no-banner gui --managed --port $1 --dir ."
  if [ ${#ROOTS[@]} -gt 0 ]; then
    for w in "${ROOTS[@]}"; do cmd="$cmd --root $(home_path "$w")"; done
  fi
  if [ -n "$FIND" ]; then
    script="cd -- $(home_path "${DIR:-~}") && { $FIND; $cmd; }"
  else
    script="cd -- $(home_path "${DIR:-~}") && $cmd"
  fi
  script="${parts:+$parts; }$script"
  printf 'exec bash -lc %s' "$(shq "$script")"
}

node_target() {  # the node, with the gateway's user when that differs from yours (ssh -G for an alias)
  local u=""
  case "$NODE" in *@*) printf '%s' "$NODE"; return ;; esac
  case "$GATEWAY" in
    *@*) u="${GATEWAY%%@*}" ;;
    *) u="$(ssh -G "$GATEWAY" 2>/dev/null | awk '$1 == "user" { print $2; exit }')" ;;
  esac
  if [ -n "$u" ] && [ "$u" != "${LOGNAME:-${USER:-$(id -un)}}" ]; then printf '%s@%s' "$u" "$NODE"; else printf '%s' "$NODE"; fi
}

g_val() {  # $1 = key, $2 = ssh -G output: the values of key, one per line
  printf '%s\n' "$2" | sed -n "s/^$1 //p"
}
node_keys() {  # $1 = the node target -> NODE_KEYS (node_key_options in connect.py)
  # With -J, ssh logs in to the node from this computer with the settings
  # ssh's config has for the node's name: the gateway's IdentityFile (and
  # IdentityAgent, IdentitiesOnly) are passed for the node hop too, or the
  # node gets only the default key files and asks for a password.
  local g n f a
  NODE_KEYS=()
  g="$(ssh -G "$GATEWAY" 2>/dev/null)"
  n="$(ssh -G "$1" 2>/dev/null)"
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    g_val identityfile "$n" | grep -qxF -- "$f" && continue
    # shellcheck disable=SC2088  # ssh -G prints a literal ~/
    case "$f" in "~/"*) [ -f "$HOME/${f#\~/}" ] || continue ;; *) [ -f "$f" ] || continue ;; esac
    NODE_KEYS+=(-o "IdentityFile=$f")
  done <<EOF4
$(g_val identityfile "$g")
EOF4
  a="$(g_val identityagent "$g" | head -n 1)"
  if [ -n "$a" ] && [ "$a" != "$(g_val identityagent "$n" | head -n 1)" ]; then NODE_KEYS+=(-o "IdentityAgent=$a"); fi
  if [ "$(g_val identitiesonly "$g" | head -n 1)" = yes ] && [ "$(g_val identitiesonly "$n" | head -n 1)" != yes ]; then
    NODE_KEYS+=(-o IdentitiesOnly=yes)
  fi
}

# The jump hosts before the destination: the -J list, else a ProxyJump (or ProxyCommand) your ssh
# config sets for it. With them, the destination's ConnectTimeout also covers the logins to them
# (a passphrase prompt, 2FA): PROMPT=1 unless each has BatchMode yes, and then 120 s, not 20
# (jump_chain, prompt_wait and PROMPT_TIMEOUT_S in connect.py)
chain_setup() {  # $1 = the destination -> CHAIN (one host per line, or proxycommand), PROMPT
  local g pj pc h bm
  CHAIN="" PROMPT=0
  if [ -n "$NODE" ] && [ -n "$GATEWAY" ] && [ "$HOP" != nested ]; then CHAIN="${JUMP:+$JUMP,}$GATEWAY"
  elif [ -n "$JUMP" ]; then CHAIN="$JUMP"
  else
    g="$(ssh -G "$1" 2>/dev/null </dev/null)"
    pj="$(g_val proxyjump "$g" | head -n 1)"; pc="$(g_val proxycommand "$g" | head -n 1)"
    if [ -n "$pj" ] && [ "$pj" != none ]; then CHAIN="$pj"
    elif [ -n "$pc" ] && [ "$pc" != none ]; then CHAIN=proxycommand; fi
  fi
  CHAIN="$(printf '%s\n' "$CHAIN" | tr ',' '\n' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' | grep -v '^$')"
  while IFS= read -r h; do
    [ -n "$h" ] || continue
    if [ "$h" = proxycommand ]; then PROMPT=1; break; fi
    bm="$(ssh -G "$(printf '%s' "$h" | sed 's/:[0-9][0-9]*$//')" 2>/dev/null </dev/null | awk '$1 == "batchmode" { print $2; exit }')"
    if [ "$bm" != yes ]; then PROMPT=1; break; fi
  done <<EOF6
$CHAIN
EOF6
  return 0
}
# The first jump host's HostName in SLOW when a TCP connection to it does not open within its
# ConnectTimeout (yours, else 20): -J's own ssh gets none of our options, so this keeps an
# unreachable gateway a fast failure under the 120 s (first_hop_timed_out in connect.py)
first_hop_timed_out() {
  local h spec port g hn t pid i
  SLOW=""
  h="$(printf '%s\n' "$CHAIN" | head -n 1)"
  [ -n "$h" ] && [ "$h" != proxycommand ] || return 1
  spec="$(printf '%s' "$h" | sed 's/:[0-9][0-9]*$//')"; port=""
  [ "$spec" != "$h" ] && port="${h##*:}"
  g="$(ssh -G "$spec" 2>/dev/null </dev/null)"
  [ -n "$g" ] || return 1
  case "$(g_val proxyjump "$g" | head -n 1)" in ""|none) ;; *) return 1 ;; esac
  case "$(g_val proxycommand "$g" | head -n 1)" in ""|none) ;; *) return 1 ;; esac
  hn="$(g_val hostname "$g" | head -n 1)"; hn="${hn:-${spec##*@}}"
  [ -n "$port" ] || port="$(g_val port "$g" | head -n 1)"
  case "$port" in ""|*[!0-9]*) port=22 ;; esac
  t="$(g_val connecttimeout "$g" | head -n 1)"
  case "$t" in ""|0|*[!0-9]*) t=20 ;; esac
  ( exec 5<>"/dev/tcp/$hn/$port" ) 2>/dev/null &
  pid=$!
  i=0
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$i" -ge $((t * 10)) ]; then kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; SLOW="$hn"; return 0; fi
    sleep 0.1; i=$((i + 1))
  done
  wait "$pid" 2>/dev/null
  return 1
}
# -o ConnectTimeout=20 (120 with PROMPT=1) unless an --ssh-option or your ssh config (ssh -G) sets
# one: a command-line -o would beat the config's own value (a slow VPN's). CT_VAL: yours, else 20,
# for the nested hop's inner ssh; WAIT_S: the one the connection gets (CONNECT_TIMEOUT_S,
# config_connect_timeout in connect.py)
ct_setup() {  # $1 = the host the options apply to -> CT_OPTS, CT_VAL, WAIT_S
  local v=""
  WAIT_S=20
  [ "${PROMPT:-0}" = 1 ] && WAIT_S=120
  CT_OPTS=(-o "ConnectTimeout=$WAIT_S") CT_VAL=20
  if [ ${#SSH_OPTIONS[@]} -gt 0 ]; then
    v="$(printf '%s ' "${SSH_OPTIONS[@]}" | tr '[:upper:]' '[:lower:]')"
    case "$v" in *connecttimeout*) CT_OPTS=() ;; esac
    v="$(printf '%s\n' "$v" | sed -n 's/.*connecttimeout[ =]*\([0-9][0-9]*\).*/\1/p')"
  fi
  if [ -z "$v" ] && [ -n "$1" ]; then
    v="$(ssh -G "$1" 2>/dev/null </dev/null | awk '$1 == "connecttimeout" && $2 ~ /^[0-9]+$/ && $2 > 0 { print $2; exit }')"
    [ -n "$v" ] && CT_OPTS=()
  fi
  [ -n "$v" ] && CT_VAL="$v" WAIT_S="$v"
  return 0
}
build_argv() {  # $1 local port, $2 remote port, $3 gateway-side port -> ARGV
  local lp="$1" rp="$2" gp="$3" rcmd jumps
  rcmd="$(remote_command "$rp")"
  ARGV=(ssh)
  jumps="$JUMP"
  # TRUST_CMD: the line that asks once to trust the host key(s) this session needs (host_key_trust_command)
  if [ -z "$NODE" ] || [ -z "$GATEWAY" ]; then
    chain_setup "${NODE:-$GATEWAY}"
    ct_setup "${NODE:-$GATEWAY}"
    TRUST_CMD="ssh ${jumps:+-J $jumps }${NODE:-$GATEWAY} true"
    [ -n "$jumps" ] && ARGV+=(-J "$jumps")
    ARGV+=(-o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4)
    [ ${#CT_OPTS[@]} -gt 0 ] && ARGV+=("${CT_OPTS[@]}")
    [ ${#SSH_OPTIONS[@]} -gt 0 ] && ARGV+=("${SSH_OPTIONS[@]}")
    ARGV+=(-L "127.0.0.1:$lp:127.0.0.1:$rp" "${NODE:-$GATEWAY}" "$rcmd")
    return
  fi
  if [ "$HOP" = nested ]; then
    TRUST_CMD="ssh ${jumps:+-J $jumps }$GATEWAY true"
    chain_setup "$GATEWAY"
    ct_setup "$GATEWAY"
    local inner
    inner="ssh -o BatchMode=yes -o ConnectTimeout=$CT_VAL -o ExitOnForwardFailure=yes -L 127.0.0.1:$gp:127.0.0.1:$rp $(shq "$NODE") $(shq "$rcmd")"
    [ -n "$jumps" ] && ARGV+=(-J "$jumps")
    ARGV+=(-o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4)
    [ ${#CT_OPTS[@]} -gt 0 ] && ARGV+=("${CT_OPTS[@]}")
    [ ${#SSH_OPTIONS[@]} -gt 0 ] && ARGV+=("${SSH_OPTIONS[@]}")
    ARGV+=(-L "127.0.0.1:$lp:127.0.0.1:$gp" "$GATEWAY" "$inner")
    return
  fi
  local target
  target="$(node_target)"
  node_keys "$target"
  chain_setup "$target"
  ct_setup "$target"
  TRUST_CMD="ssh -J ${jumps:+$jumps,}$GATEWAY $target true"
  ARGV+=(-J "${jumps:+$jumps,}$GATEWAY")
  ARGV+=(-o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4)
  [ ${#CT_OPTS[@]} -gt 0 ] && ARGV+=("${CT_OPTS[@]}")
  [ ${#NODE_KEYS[@]} -gt 0 ] && ARGV+=("${NODE_KEYS[@]}")
  [ ${#SSH_OPTIONS[@]} -gt 0 ] && ARGV+=("${SSH_OPTIONS[@]}")
  ARGV+=(-L "127.0.0.1:$lp:127.0.0.1:$rp" "$target" "$rcmd")
}

# ------------------------------------------------------------------ host names (gateway_unresolvable in connect.py)
config_hosts() {  # the Host patterns of ~/.ssh/config, one per line (ssh_config_hosts in connect.py)
  [ -f "$HOME/.ssh/config" ] || return 0
  sed -n -E 's/^[[:space:]]*[Hh][Oo][Ss][Tt]([[:space:]]+|[[:space:]]*=[[:space:]]*)//p' "$HOME/.ssh/config" \
    | tr -s ' \t' '\n\n' | sed -e 's/^"//' -e 's/"$//' | grep -v -e '^!' -e '^$'
}
host_in_config() {  # $1 host: 0 when a Host pattern matches it
  local h pat
  h="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  while IFS= read -r pat; do
    [ -n "$pat" ] || continue
    pat="$(printf '%s' "$pat" | tr '[:upper:]' '[:lower:]')"
    # shellcheck disable=SC2254  # the Host pattern is a glob on purpose
    case "$h" in $pat) return 0 ;; esac
  done <<EOF5
$(config_hosts)
EOF5
  return 1
}
resolves() {  # $1 host name: 0 when it resolves here (or nothing here can tell)
  if command -v getent >/dev/null 2>&1; then
    getent hosts "$1" >/dev/null 2>&1 || getent ahosts "$1" >/dev/null 2>&1; return
  fi
  if command -v dscacheutil >/dev/null 2>&1; then [ -n "$(dscacheutil -q host -a name "$1" 2>/dev/null)" ]; return; fi
  if command -v host >/dev/null 2>&1; then host "$1" >/dev/null 2>&1; return; fi
  return 0
}
gw_unresolvable() {  # -> UNRES: the gateway's host name when no Host matches it and it does not resolve
  local h g pj pc hn
  UNRES=""
  [ -n "$GATEWAY" ] && [ -z "$JUMP" ] || return 1
  h="${GATEWAY##*@}"
  host_in_config "$h" && return 1
  g="$(ssh -G "$GATEWAY" 2>/dev/null)"
  pj="$(g_val proxyjump "$g" | head -n 1)"
  pc="$(g_val proxycommand "$g" | head -n 1)"
  [ "${pj:-none}" = none ] && [ "${pc:-none}" = none ] || return 1
  hn="$(g_val hostname "$g" | head -n 1)"
  hn="${hn:-$h}"
  resolves "$hn" && return 1
  UNRES="$hn"
}

# ssh's words for a host it could not reach (unreachable in connect.py)
unreach() {  # $1 ssh's messages, $2 the node (for "open failed" through the gateway) -> UR_HOST, UR_WHY
  local l
  UR_HOST="" UR_WHY=""
  l="$(printf '%s\n' "$1" | grep -E 'Could not resolve hostname|connect to host [^ ]+ port [0-9]+: (Connection timed out|Operation timed out|Connection refused|No route to host|Network is unreachable)|Connection timed out during banner exchange|Connection to [^ ]+ port [0-9]+ timed out|open failed: (connect failed|administratively prohibited): ' | head -n 1)"
  [ -n "$l" ] || return 1
  case "$l" in
    *"Could not resolve hostname"*)
      UR_HOST="$(printf '%s' "$l" | sed -E 's/.*Could not resolve hostname ([^: ]+).*/\1/')"
      UR_WHY="DNS: the name does not resolve" ;;
    *"open failed:"*)
      [ -n "${2:-}" ] || return 1
      UR_HOST="${2##*@}"
      case "$(printf '%s' "${l#*open failed: }" | tr '[:upper:]' '[:lower:]')" in
        *"not known"*|*servname*|*resolve*) UR_WHY="DNS on the gateway: the name does not resolve" ;;
        *"timed out"*) UR_WHY="timeout" ;;
        *refused*) UR_WHY="connection refused" ;;
        *route*|*unreachable*) UR_WHY="no route" ;;
        *) UR_WHY="${l##*: }" ;;
      esac ;;
    *"banner exchange"*)
      UR_HOST="$(printf '%s\n' "$1" | sed -n -E 's/.*Connection to ([^ ]+) port.*/\1/p' | head -n 1)"
      UR_HOST="${UR_HOST:-the host}"; UR_WHY="timeout" ;;
    *"timed out"*)
      UR_HOST="$(printf '%s' "$l" | sed -E 's/.*(connect to host|Connection to) ([^ ]+) port.*/\2/')"; UR_WHY="timeout" ;;
    *refused*)
      UR_HOST="$(printf '%s' "$l" | sed -E 's/.*connect to host ([^ ]+) port.*/\1/')"; UR_WHY="connection refused" ;;
    *)
      UR_HOST="$(printf '%s' "$l" | sed -E 's/.*connect to host ([^ ]+) port.*/\1/')"; UR_WHY="no route" ;;
  esac
}

# ------------------------------------------------------------------ --check and --fix-config
# The same checks and words as mace gui connect --check (connect_check.py):
# read-only; the private half of a key is never opened.
CK_FAILS=0 CK_HOPFAILS=0
ck() {  # $1 level, $2 text, $3 the fix (optional)
  printf '%-5s %s\n' "$1" "$2"
  if [ -n "${3:-}" ]; then printf '      fix: %s\n' "$3"; fi
  if [ "$1" = FAIL ]; then CK_FAILS=$((CK_FAILS + 1)); fi
  return 0
}
xpath() {  # a path as ssh -G prints it (~/...) -> a path here
  # shellcheck disable=SC2088  # ssh -G prints a literal ~/
  case "$1" in "~") printf '%s' "$HOME" ;; "~/"*) printf '%s/%s' "$HOME" "${1#\~/}" ;; *) printf '%s' "$1" ;; esac
}
existing_files() {  # $1 ssh -G output: its identity files that exist, one per line
  local f
  g_val identityfile "$1" | while IFS= read -r f; do
    if [ -n "$f" ] && [ -f "$(xpath "$f")" ]; then printf '%s\n' "$f"; fi
  done
}
joined() { paste -s -d ',' - | sed 's/,/, /g'; }
fingerprint() {  # $1 key file: the SHA256 fingerprint of its .pub (the public half), or nothing
  local pub
  pub="$(xpath "$1").pub"
  [ -f "$pub" ] && ssh-keygen -lf "$pub" 2>/dev/null | awk '{ print $2; exit }'
}
fmode() { stat -c %a "$1" 2>/dev/null || stat -f %Lp "$1" 2>/dev/null; }
ck_cmd() {  # $1 the hop to name (optional): the line that connects with this script
  local c="$0"
  if [ -n "$PROFILE" ]; then c="$c $(shq "$PROFILE")"
  else
    [ -n "$GATEWAY" ] && c="$c --gateway $(shq "$GATEWAY")"
    [ -n "$NODE" ] && c="$c --node $(shq "$NODE")"
    [ -n "$JUMP" ] && c="$c --jump $(shq "$JUMP")"
  fi
  [ -n "${1:-}" ] && c="$c --hop $1"
  printf '%s' "$c"
}
agent_load() {  # -> AGENT_RC, AGENT_OUT: ssh-add -l (with the gateway's IdentityAgent, if any)
  local ia
  [ -n "${AGENT_DONE:-}" ] && return 0
  AGENT_DONE=1
  ia="$(g_val identityagent "$CK_G" | head -n 1)"
  case "$ia" in
    ""|none|SSH_AUTH_SOCK|'$'*) AGENT_SOCK="${SSH_AUTH_SOCK:-}"; AGENT_OUT="$(ssh-add -l 2>&1)"; AGENT_RC=$? ;;
    *) AGENT_SOCK="$(xpath "$ia")"; AGENT_OUT="$(SSH_AUTH_SOCK="$AGENT_SOCK" ssh-add -l 2>&1)"; AGENT_RC=$? ;;
  esac
  AGENT_FPS="$(printf '%s\n' "$AGENT_OUT" | awk '$1 ~ /^[0-9]+$/ { print $2 }')"
}
# A node that is down (or hangs) behind the gateway: through -J, ssh can tell that only when the
# node's ConnectTimeout runs out (120 s while a prompt may need it). Beside the connection, a second
# login to the gateway that cannot prompt (BatchMode: an agent's key or one without a passphrase;
# else it says nothing) asks whether the node's ssh sends its first bytes within 8 s; its last line,
# MACE_NODE_PROBE <code>, goes to $TMPD/probe (124: nothing came; node_probe in connect.py)
node_probe() {
  local g hn port ct pj=()
  g="$(ssh -G "$(node_target)" 2>/dev/null </dev/null)"
  hn="$(g_val hostname "$g" | head -n 1)"; hn="${hn:-${NODE##*@}}"
  port="$(g_val port "$g" | head -n 1)"
  case "$port" in ""|*[!0-9]*) port=22 ;; esac
  ct=(-o ConnectTimeout=20)
  case "$(ssh -G "$GATEWAY" 2>/dev/null </dev/null | awk '$1 == "connecttimeout" { print $2; exit }')" in
    ""|none|0) ;; *) ct=() ;;
  esac
  [ -n "$JUMP" ] && pj=(-J "$JUMP")
  printf '%s\n' "timeout 8 bash -c 'exec 3<>/dev/tcp/\"\$0\"/\"\$1\" && head -c 4 <&3' $(shq "$hn") $port" 'c=$?' 'echo' \
    'echo "MACE_NODE_PROBE $c"' \
    | ssh ${pj[@]+"${pj[@]}"} -T -o BatchMode=yes ${ct[@]+"${ct[@]}"} "$GATEWAY" "bash -s" > "$TMPD/probe.part" 2>/dev/null
  mv "$TMPD/probe.part" "$TMPD/probe" 2>/dev/null
}
# UNLOADED: a key file ssh offers the first host that has a passphrase while the agent holds none
# of the keys offered there, so ssh asks for it on every hop; AGENT_UP=1 when an agent answers
# (unloaded_key in connect_check.py). ssh-keygen -y -P "" only tells whether a key has one.
unloaded_key() {
  local f fp keys
  UNLOADED="" AGENT_UP=0
  CK_G="$(ssh -G "${GATEWAY:-$NODE}" 2>/dev/null </dev/null)"
  keys="$(existing_files "$CK_G")"
  [ -n "$keys" ] || return 1
  agent_load
  case "$AGENT_RC" in 0|1) AGENT_UP=1 ;; esac
  while IFS= read -r f; do
    fp="$(fingerprint "$f")"
    if [ -n "$fp" ] && printf '%s\n' "$AGENT_FPS" | grep -qxF -- "$fp"; then return 1; fi
  done <<EOF7
$keys
EOF7
  while IFS= read -r f; do
    if ! ssh-keygen -y -P "" -f "$(xpath "$f")" >/dev/null 2>&1 </dev/null; then UNLOADED="$f"; return 0; fi
  done <<EOF8
$keys
EOF8
  return 1
}
node_refused() {  # $1 ssh's messages: the node refused the login, not the gateway (node_auth_refused)
  local d nh gh
  case "$1" in *"Host key verification failed"*) return 1 ;; esac
  d="$(printf '%s\n' "$1" | grep 'Permission denied (' | cut -d: -f1)"
  [ -n "$d" ] || return 1
  nh="${NODE##*@}"; gh="${GATEWAY##*@}"
  if [ -n "$gh" ] && printf '%s\n' "$d" | grep -qF -- "$gh"; then return 1; fi
  printf '%s\n' "$d" | grep -qF -- "$nh"
}
ck_classify() {  # $1 host, $2 ssh's messages, $3 exit code, $4 "jump" for the node hop
  local last methods
  if unreach "$2" "${4:+$NODE}"; then
    if [ -n "${4:-}" ] && { [ "$UR_HOST" = UNKNOWN ] || [ "$UR_HOST" = "the host" ]; }; then
      ck FAIL "$NODE did not answer through the gateway $GATEWAY in time (ssh calls it UNKNOWN port 65535): it may be down or not accept logins" "pick another node (--node), then check again"
      return
    fi
    ck FAIL "cannot reach host $UR_HOST ($UR_WHY)" "check the name, and your network or VPN (ssh -v $UR_HOST true shows where it stops)"
    return
  fi
  case "$2" in
    *"Host key verification failed"*|*"IDENTIFICATION HAS CHANGED"*)
      ck FAIL "ssh does not know the host key of $1 yet, or it changed" "ssh $1 true (once, in this terminal, to check and accept the key)"
      return ;;
  esac
  last="$(printf '%s\n' "$2" | grep -v '^[[:space:]]*$' | tail -n 1)"
  last="${last:-no message}"
  if [ -n "${4:-}" ]; then
    if node_refused "$2"; then
      ck FAIL "jump hop: $NODE refused your key from this computer (a connection would ask for the password of $NODE)" "use the nested hop, or give $NODE your key (docs/gui.md#mace-gui-connect-the-node-asks-for-a-password)"
    else
      ck FAIL "jump hop: ssh -J $GATEWAY $NODE true failed (exit $3): $last"
    fi
    return
  fi
  if printf '%s\n' "$2" | grep -q 'Permission denied ('; then
    methods="$(printf '%s\n' "$2" | sed -n 's/.*Permission denied (\([^)]*\)).*/\1/p' | head -n 1)"
    case ",$methods," in
      *,publickey,*) ck FAIL "$1 refused your keys (Permission denied: $methods)" "ssh-add your key, and check that its .pub is in ~/.ssh/authorized_keys on the cluster (docs/ssh-setup.md)" ;;
      *) ck WARN "$1 asks for a 2FA code or a password, which the check cannot type; the hop probes stop here" "turn on connection sharing (ControlMaster, see --fix-config), log in once with ssh $1, then check again; or skip the probes with --no-net" ;;
    esac
    return
  fi
  ck FAIL "ssh $1 true failed (exit $3): $last" "ssh -v $1 true"
}

# shellcheck disable=SC2088  # ~/ paths printed for you, not expanded here
run_check() {
  local first what role u hn port pj gfiles nfiles n target missing held f fp key cm why extra err rc inner
  local res_gw="" res_jump="" res_nested="" others
  first="${GATEWAY:-$NODE}"
  if [ -n "$NODE" ] && [ -n "$GATEWAY" ]; then what="$GATEWAY -> $NODE"; else what="$first"; fi
  echo "Checking the ssh setup for ${PROFILE:+$PROFILE: }$what (read-only; nothing is changed)"
  # 1. ssh
  if ! command -v ssh >/dev/null 2>&1; then
    ck FAIL "no ssh command on PATH: connect needs the OpenSSH client" "install the OpenSSH client (the openssh-client or openssh package)"
    echo "Verdict: fix the FAIL lines above, then check again."
    return 2
  fi
  ck OK "ssh: $(ssh -V 2>&1 | head -n 1) ($(command -v ssh))"
  # 2. the gateway (ssh -G)
  CK_G="$(ssh -G "$first" 2>/dev/null)"
  if [ -z "$CK_G" ]; then
    ck FAIL "ssh -G $first printed nothing: ssh cannot read its settings for $first" "ssh -G $first (shows ssh's own error, often a typo in ~/.ssh/config)"
    echo "Verdict: fix the FAIL lines above, then check again."
    return 1
  fi
  if [ -n "$GATEWAY" ]; then role=gateway; else role=node; fi
  u="$(g_val user "$CK_G" | head -n 1)"; hn="$(g_val hostname "$CK_G" | head -n 1)"; port="$(g_val port "$CK_G" | head -n 1)"
  ck OK "$role $first: user ${u:-?}, HostName ${hn:-$first}, port ${port:-22}"
  pj="$(g_val proxyjump "$CK_G" | head -n 1)"
  if [ -n "$pj" ] && [ "$pj" != none ]; then ck INFO "$first is reached through ProxyJump $pj (from your ssh config)"; fi
  if [ -n "$GATEWAY" ] && email_like "$GATEWAY"; then
    ck WARN "the gateway $GATEWAY looks like an email address. It should be user@login-host (your cluster's login node) or a Host alias from ~/.ssh/config" "mace gui connect ${PROFILE:+$PROFILE }--gateway <user@login-host> --save"
  fi
  if [ -n "$GATEWAY" ] && [ "$NO_NET" != 1 ] && gw_unresolvable; then
    ck WARN "$UNRES matches no Host in ~/.ssh/config and does not resolve (DNS)" "use your cluster's login host (e.g. login.example.edu), or add a Host block with its HostName to ~/.ssh/config"
  fi
  gfiles="$(existing_files "$CK_G")"
  if [ -n "$gfiles" ]; then
    extra=""
    [ "$(g_val identitiesonly "$CK_G" | head -n 1)" = yes ] && extra=" (IdentitiesOnly yes)"
    ck OK "key files ssh offers $first: $(printf '%s\n' "$gfiles" | joined)$extra"
  else
    ck WARN "none of the key files ssh offers $first exist; only an agent's keys can log in" "ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_cluster, then put its .pub in ~/.ssh/authorized_keys on the cluster (docs/ssh-setup.md)"
  fi
  cm="$(g_val controlmaster "$CK_G" | head -n 1)"
  case "${cm:-false}" in false|no) ;; *) ck OK "connection sharing is on for $first (ControlMaster $cm)" ;; esac
  # 3. the node (ssh -G) against the gateway
  if [ -n "$NODE" ] && [ -n "$GATEWAY" ]; then
    if [ "$HOP" = nested ]; then
      ck INFO "nested hop: the gateway logs in to $NODE with the keys it has there; your keys here are not used for that hop"
    else
      target="$(node_target)"
      n="$(ssh -G "$target" 2>/dev/null)"
      nfiles="$(existing_files "$n")"
      u="$(g_val user "$n" | head -n 1)"
      ck INFO "node $NODE: user ${u:-?}, key files $(if [ -n "$nfiles" ]; then printf '%s\n' "$nfiles" | joined; else printf 'none that exist'; fi)"
      node_keys "$target"
      if [ ${#NODE_KEYS[@]} -gt 0 ]; then ck INFO "for the jump hop, the connection also gives $NODE: ${NODE_KEYS[*]}"; fi
      if [ -n "$gfiles" ]; then
        missing=""
        while IFS= read -r f; do
          g_val identityfile "$n" | grep -qxF -- "$f" || missing="${missing:+$missing
}$f"
        done <<EOF6
$gfiles
EOF6
        if [ "$(printf '%s\n' "$missing" | grep -c .)" -lt "$(printf '%s\n' "$gfiles" | grep -c .)" ]; then
          ck OK "the jump hop offers $NODE your gateway key (from your ssh config)"
        else
          agent_load
          held=""
          while IFS= read -r f; do
            fp="$(fingerprint "$f")"
            if [ -n "$fp" ] && [ "$AGENT_RC" = 0 ] && printf '%s\n' "$AGENT_FPS" | grep -qxF -- "$fp"; then held="$f"; break; fi
          done <<EOF7
$missing
EOF7
          if [ -n "$held" ]; then
            ck OK "the jump hop offers $NODE your gateway key (the agent holds $held)"
          else
            ck WARN "$NODE gets none of $(printf '%s\n' "$missing" | joined) from your ssh config, and the agent does not hold them, so a plain ssh -J $GATEWAY $NODE would ask for the password of $NODE" "the connection adds them for you (above); for plain ssh too, add a Host block for the node (--fix-config prints one) or run ssh-add $(printf '%s\n' "$missing" | head -n 1)"
          fi
        fi
      fi
    fi
  fi
  # 4. the agent: fingerprints only
  key="$(printf '%s\n' "$gfiles" | head -n 1)"
  key="${key:-~/.ssh/id_ed25519_cluster}"
  agent_load
  case "$AGENT_RC" in
    0) ck OK "agent: $(printf '%s\n' "$AGENT_FPS" | grep -c .) key(s): $(printf '%s\n' "$AGENT_OUT" | awk '$1 ~ /^[0-9]+$/ { t = ($NF ~ /^\(/) ? " " $NF : ""; print $2 t }' | joined)" ;;
    1) ck WARN "the ssh agent holds no keys" "ssh-add $key" ;;
    *)
      why="$(printf '%s\n' "$AGENT_OUT" | head -n 1)"
      [ -z "$AGENT_SOCK" ] && why="SSH_AUTH_SOCK is not set in this terminal"
      ck WARN "cannot reach an ssh agent (${why:-no answer})" "eval \"\$(ssh-agent)\" && ssh-add $key" ;;
  esac
  if [ "$AGENT_RC" = 0 ] || [ "$AGENT_RC" = 1 ]; then
    while IFS= read -r f; do
      [ -n "$f" ] || continue
      fp="$(fingerprint "$f")"
      if [ -z "$fp" ]; then
        ck INFO "$f.pub is missing, so the check cannot compare $f with the agent" "ssh-keygen -y -f $f > $f.pub"
      elif printf '%s\n' "$AGENT_FPS" | grep -qxF -- "$fp"; then
        ck OK "$f ($fp) is in the agent"
      else
        ck WARN "$f ($fp) is not in the agent: with a passphrase, ssh must ask for it on every hop, and a connection may have no terminal to ask on" "ssh-add $f"
      fi
    done <<EOF8
$gfiles
EOF8
  fi
  # 5. permissions
  if [ -d "$HOME/.ssh" ]; then
    local bad=0 m files
    m="$(fmode "$HOME/.ssh")"
    if [ -n "$m" ] && [ $((8#$m & 8#077)) -ne 0 ]; then ck WARN "~/.ssh is mode $m; ssh wants 700" "chmod 700 ~/.ssh"; bad=1; fi
    files="$(printf '%s\n%s\n%s\n' "~/.ssh/config" "$gfiles" "${nfiles:-}" | grep . | awk '!seen[$0]++')"
    while IFS= read -r f; do
      [ -f "$(xpath "$f")" ] || continue
      m="$(fmode "$(xpath "$f")")"
      [ -n "$m" ] && [ $((8#$m & 8#077)) -ne 0 ] || continue
      if [ "$f" = "~/.ssh/config" ] && [ $((8#$m & 8#022)) -eq 0 ]; then
        ck WARN "$f is mode $m; ssh wants 600 and refuses a key or config others can use" "chmod 600 $f"
      else
        ck FAIL "$f is mode $m; ssh wants 600 and refuses a key or config others can use" "chmod 600 $f"
      fi
      bad=1
    done <<EOF9
$files
EOF9
    [ "$bad" = 0 ] && ck OK "permissions: ~/.ssh, its config and your key files are private"
  fi
  # 6. live probes, each running only `true`
  if [ "$NO_NET" = 1 ]; then
    ck SKIP "live probes skipped: no ssh login was tried"
  else
    local J=() X=()
    [ -n "$JUMP" ] && J=(-J "$JUMP")
    if [ ${#SSH_OPTIONS[@]} -gt 0 ]; then X=("${SSH_OPTIONS[@]}"); fi
    case "${cm:-false}" in
      false|no) ;;
      *) if ssh -O check ${J[@]+"${J[@]}"} "$first" >/dev/null 2>&1 </dev/null; then ck OK "a shared connection to $first is open: the probes reuse it"; fi ;;
    esac
    # the profile's ssh options where the connection passes them: the gateway's ssh of the nested
    # hop (and without a hop), the node's ssh of the jump hop (-J's gateway leg gets none)
    if [ -n "$NODE" ] && [ -n "$GATEWAY" ] && [ "$HOP" != nested ]; then
      err="$(ssh ${J[@]+"${J[@]}"} -o BatchMode=yes -o ConnectTimeout=10 "$first" true 2>&1 >/dev/null </dev/null)"; rc=$?
    else
      err="$(ssh ${J[@]+"${J[@]}"} -o BatchMode=yes -o ConnectTimeout=10 ${X[@]+"${X[@]}"} "$first" true 2>&1 >/dev/null </dev/null)"; rc=$?
    fi
    if [ "$rc" = 0 ]; then
      ck OK "ssh $first true: logged in without a prompt"; res_gw=1
    else
      ck_classify "$first" "$err" "$rc"
    fi
    if [ -n "$res_gw" ] && [ -n "$NODE" ] && [ -n "$GATEWAY" ]; then
      target="$(node_target)"
      node_keys "$target"
      err="$(ssh -J "${JUMP:+$JUMP,}$GATEWAY" -o BatchMode=yes -o ConnectTimeout=10 ${NODE_KEYS[@]+"${NODE_KEYS[@]}"} ${X[@]+"${X[@]}"} "$target" true 2>&1 >/dev/null </dev/null)"; rc=$?
      if [ "$rc" = 0 ]; then
        ck OK "jump hop: ssh -J $GATEWAY $NODE true works"; res_jump=1
      else
        ck_classify "$NODE" "$err" "$rc" jump; CK_HOPFAILS=$((CK_HOPFAILS + 1))
      fi
      inner="ssh -o BatchMode=yes -o ConnectTimeout=10 $(shq "$NODE") true"
      err="$(ssh ${J[@]+"${J[@]}"} -o BatchMode=yes -o ConnectTimeout=10 ${X[@]+"${X[@]}"} "$GATEWAY" "$inner" 2>&1 >/dev/null </dev/null)"; rc=$?
      if [ "$rc" = 0 ]; then
        ck OK "nested hop: the gateway logs in to $NODE by itself"; res_nested=1
      else
        ck FAIL "nested hop: the gateway could not log in to $NODE (exit $rc): $(printf '%s\n' "$err" | grep -v '^[[:space:]]*$' | tail -n 1)" "log in to the gateway and run ssh $NODE true there: the cluster may need a key for logins between its own hosts"
        CK_HOPFAILS=$((CK_HOPFAILS + 1))
      fi
    fi
  fi
  # 7. the verdict
  others=$((CK_FAILS - CK_HOPFAILS))
  if [ "$others" = 0 ] && { [ -n "$res_jump" ] || [ -n "$res_nested" ]; }; then
    if [ "$HOP" = nested ] && [ -n "$res_nested" ]; then echo "Verdict: the nested hop works. Connect with: $(ck_cmd)"
    elif [ -n "$res_jump" ]; then
      if [ "$HOP" = nested ]; then echo "Verdict: the jump hop works. Connect with: $(ck_cmd jump)"
      else echo "Verdict: the jump hop works. Connect with: $(ck_cmd)"; fi
    else echo "Verdict: only the nested hop works. Connect with: $(ck_cmd nested)"; fi
    return 0
  fi
  if [ "$CK_FAILS" -gt 0 ]; then echo "Verdict: fix the FAIL lines above, then check again."; return 1; fi
  if [ -z "$NODE" ] || [ -z "$GATEWAY" ]; then
    if [ -n "$res_gw" ]; then echo "Verdict: MACE GUI runs on the gateway itself. Connect with: $(ck_cmd)"; return 0; fi
  fi
  echo "Verdict (without the probes): no problem found. Connect with: $(ck_cmd)"
  return 0
}

suggest_alias() {  # a short Host alias from a login host: login.example.edu -> example, hpcc.example.edu -> hpcc
  local h l1 rest
  h="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  l1="${h%%.*}"; rest="${h#*.}"
  case "$l1" in
    login|ssh|gateway|gw|bastion)
      if [ "$rest" != "$h" ] && [ "${rest#*.}" != "$rest" ]; then printf '%s' "${rest%%.*}"; return; fi ;;
  esac
  case "$l1" in ""|*[!0-9-]*) printf '%s' "${l1:-cluster}" ;; *) printf cluster ;; esac
}
# shellcheck disable=SC2088
fix_config() {  # print (never write) a suggested ~/.ssh/config block (Checker.fix_config in connect_check.py)
  local first g user hn alias keys key f pj c node
  first="${GATEWAY:-$NODE}"
  g="$(ssh -G "$first" 2>/dev/null)"
  user="$(g_val user "$g" | head -n 1)"
  hn="$(g_val hostname "$g" | head -n 1)"; hn="${hn:-${first##*@}}"
  case "$first" in *@*|"") alias="$(suggest_alias "$hn")" ;; *) alias="$first" ;; esac
  keys="$(existing_files "$g")"
  key=""
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    case "${f##*/}" in id_rsa|id_ecdsa|id_ecdsa_sk|id_ed25519|id_ed25519_sk|id_xmss|id_dsa) ;; *) key="$f"; break ;; esac
  done <<EOF10
$keys
EOF10
  [ -n "$key" ] || key="$(printf '%s\n' "$keys" | head -n 1)"
  echo "# A suggested ~/.ssh/config block for this profile, from ssh -G (printed only: nothing was written). Review it, then add it to ~/.ssh/config."
  if config_hosts | grep -qxF -- "$alias"; then echo "# You already have a Host $alias block: merge these lines into it."; fi
  if [ -z "$key" ]; then
    key="~/.ssh/id_ed25519_cluster"
    echo "# No key file exists yet: make one first, ssh-keygen -t ed25519 -f $key"
  fi
  printf 'Host %s\n    HostName %s\n' "$alias" "$hn"
  [ -n "$user" ] && printf '    User %s\n' "$user"
  pj="${JUMP:-$(g_val proxyjump "$g" | head -n 1)}"
  if [ -n "$pj" ] && [ "$pj" != none ]; then printf '    ProxyJump %s\n' "$pj"; fi
  printf '    IdentityFile %s\n    IdentitiesOnly yes\n    AddKeysToAgent yes\n' "$key"
  [ "$(uname 2>/dev/null)" = Darwin ] && echo "    UseKeychain yes"
  printf '    ControlMaster auto\n    ControlPath ~/.ssh/cm-%%C\n    ControlPersist 30m\n    ServerAliveInterval 30\n'
  if [ -n "$NODE" ] && [ -n "$GATEWAY" ]; then
    node="${NODE##*@}"
    printf '\nHost %s\n' "$node"
    [ -n "$user" ] && printf '    User %s\n' "$user"
    printf '    IdentityFile %s\n    IdentitiesOnly yes\n    ProxyJump %s\n' "$key" "$alias"
  fi
  c="$0"
  [ -n "$PROFILE" ] && c="$c $(shq "$PROFILE")"
  c="$c --gateway $(shq "$alias")"
  [ -n "$NODE" ] && c="$c --node $(shq "${NODE##*@}")"
  echo "# Then connect with: $c"
}

if [ "$CHECK" = 1 ] || [ "$FIX_CONFIG" = 1 ]; then
  RC=0
  if [ "$CHECK" = 1 ]; then run_check; RC=$?; fi
  if [ "$FIX_CONFIG" = 1 ]; then
    [ "$CHECK" = 1 ] && echo
    fix_config
  fi
  exit "$RC"
fi

if [ "$DRY_RUN" = 1 ]; then
  build_argv "$LOCAL_PORT" 40000 40001
  # the hop, as mace gui connect --dry-run labels it; then one argument per line
  if [ -n "$NODE" ] && [ -n "$GATEWAY" ]; then
    echo "# hop $HOP (the remote port is random on a real run)"
  else
    echo "# no hop (the remote port is random on a real run)"
  fi
  printf '%s\n' "${ARGV[@]}"
  exit 0
fi

# ------------------------------------------------------------------ run
port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
LP="$LOCAL_PORT"
while port_busy "$LP"; do
  LP=$((LP + 1))
  if [ "$LP" -gt $((LOCAL_PORT + 20)) ]; then echo "mace-gui-connect: no free local port near $LOCAL_PORT" >&2; exit 1; fi
done
[ "$LP" != "$LOCAL_PORT" ] && echo "Port $LOCAL_PORT is busy here; using $LP. Drafts saved under port $LOCAL_PORT won't be offered on $LP."
rand_port() { echo $(( (RANDOM * 32768 + RANDOM) % 41000 + 20000 )); }

TMPD="$(mktemp -d "${TMPDIR:-/tmp}/mace-gui-connect.XXXXXX")" || exit 1
chmod 700 "$TMPD"
SSH_PID=""
PINGER=""
NODE_PROBE=""
cleanup() {
  trap - INT TERM HUP EXIT
  [ -n "$NODE_PROBE" ] && kill "$NODE_PROBE" 2>/dev/null
  if [ -n "$SSH_PID" ]; then
    { echo detach >&4; } 2>/dev/null
    exec 4>&- 2>/dev/null
    for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$SSH_PID" 2>/dev/null || break; sleep 0.5; done
    HELPERS="$(pgrep -P "$SSH_PID" 2>/dev/null)"
    kill "$SSH_PID" 2>/dev/null
    # shellcheck disable=SC2086  # one pid per word
    [ -n "$HELPERS" ] && kill $HELPERS 2>/dev/null
  fi
  [ -n "$PINGER" ] && kill "$PINGER" 2>/dev/null
  rm -rf "$TMPD"
}
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM HUP
trap cleanup EXIT

# MACE GUI in its own window: a Chromium-family browser in app mode (--app), in a profile of its
# own under MACE GUI's cache folder; Brave, Chrome, Chromium, Edge, Vivaldi, the first one found
# (APP_BROWSERS in appwindow.py). Firefox has no app mode: then a browser tab.
app_browser() {  # -> APP_NAME, APP_EXE
  local spec name bundle winp root v c
  APP_NAME="" APP_EXE=""
  for spec in "Brave|Brave Browser|BraveSoftware/Brave-Browser/Application/brave.exe|brave-browser brave-browser-stable brave" \
              "Chrome|Google Chrome|Google/Chrome/Application/chrome.exe|google-chrome google-chrome-stable chrome" \
              "Chromium|Chromium|Chromium/Application/chrome.exe|chromium chromium-browser" \
              "Edge|Microsoft Edge|Microsoft/Edge/Application/msedge.exe|microsoft-edge microsoft-edge-stable msedge" \
              "Vivaldi|Vivaldi|Vivaldi/Application/vivaldi.exe|vivaldi vivaldi-stable"; do
    name="${spec%%|*}"; spec="${spec#*|}"; bundle="${spec%%|*}"; spec="${spec#*|}"; winp="${spec%%|*}"; spec="${spec#*|}"
    case "$(uname -s 2>/dev/null)" in
      Darwin)
        for root in /Applications "$HOME/Applications"; do
          if [ -x "$root/$bundle.app/Contents/MacOS/$bundle" ]; then
            APP_NAME="$name" APP_EXE="$root/$bundle.app/Contents/MacOS/$bundle"; return 0
          fi
        done ;;
      CYGWIN*|MSYS*|MINGW*)
        for v in ProgramFiles PROGRAMFILES 'ProgramFiles(x86)' 'PROGRAMFILES(X86)' LOCALAPPDATA; do
          root="$(printenv "$v" 2>/dev/null)"
          [ -n "$root" ] || continue
          c="$(winpath "$root")/$winp"
          if [ -f "$c" ]; then APP_NAME="$name" APP_EXE="$c"; return 0; fi
        done ;;
    esac
    for c in $spec; do
      if command -v "$c" >/dev/null 2>&1; then APP_NAME="$name" APP_EXE="$(command -v "$c")"; return 0; fi
    done
  done
  return 1
}
app_command() {  # $1 the URL -> APP_ARGV: the app window's command line (app_command in appwindow.py)
  local base sys
  sys="$(uname -s 2>/dev/null)"
  case "$sys" in
    CYGWIN*|MSYS*|MINGW*) base="$(winpath "${LOCALAPPDATA:-$HOME}")/mace/gui" ;;
    *) base="${XDG_CACHE_HOME:-$HOME/.cache}/mace/gui" ;;
  esac
  [ -n "${MACE_GUI_STATE_DIR:-}" ] && base="$MACE_GUI_STATE_DIR"
  mkdir -p "$base/app-profile" 2>/dev/null
  APP_LOG="$base/browser.log"
  APP_PROFILE="$base/app-profile"
  case "$sys" in CYGWIN*|MSYS*|MINGW*) command -v cygpath >/dev/null 2>&1 && APP_PROFILE="$(cygpath -w "$APP_PROFILE")" ;; esac
  APP_ARGV=("$APP_EXE" "--app=$1" "--user-data-dir=$APP_PROFILE" --no-first-run --no-default-browser-check)
  case "$sys" in Darwin|CYGWIN*|MSYS*|MINGW*) ;; *) APP_ARGV+=(--class=MACE) ;; esac
}
# what the browser or the desktop's opener prints (KDE's "kf.iconthemes: ..." lines) stays out of this terminal
open_browser() {
  if [ "$OPEN" = 0 ]; then return; fi
  if [ "$BROWSER_MODE" = app ] && app_browser; then
    app_command "$1"
    # its own session: closing this terminal does not close the window
    if command -v setsid >/dev/null 2>&1; then setsid "${APP_ARGV[@]}" >>"$APP_LOG" 2>&1 </dev/null &
    else nohup "${APP_ARGV[@]}" >>"$APP_LOG" 2>&1 </dev/null &
    fi
    if [ -z "${APP_TOLD:-}" ]; then
      echo "Opened MACE GUI in its own $APP_NAME window (--browser tab opens a browser tab instead)."
      APP_TOLD=1
    fi
    return
  fi
  if [ -n "${BROWSER:-}" ]; then "$BROWSER" "$1" >/dev/null 2>&1 &
  elif command -v open >/dev/null 2>&1 && [ "$(uname)" = Darwin ]; then open "$1" >/dev/null 2>&1
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$1" >/dev/null 2>&1 &
  elif command -v cygstart >/dev/null 2>&1; then cygstart "$1" >/dev/null 2>&1
  fi
}

json_field() {  # $1 field, $2 JSON line: a string value (or empty)
  printf '%s' "$2" | sed -n "s/.*\"$1\": *\"\([^\"]*\)\".*/\1/p"
}

http_ok() {  # $1 port, $2 path, $3 token: 0 when the server answers 200
  # read -t: a forward that accepts but does not answer yet never blocks the
  # script (it waited for good, and showed the URL only after a key press)
  (
    exec 5<>"/dev/tcp/127.0.0.1/$1" || exit 1
    printf 'GET %s HTTP/1.0\r\nHost: 127.0.0.1:%s\r\nX-Mace-Token: %s\r\n\r\n' "$2" "$1" "$3" >&5
    IFS= read -r -t 3 status <&5
    case "$status" in *" 200 "*) exit 0 ;; *) exit 1 ;; esac
  ) 2>/dev/null
}

# A node's login profile that runs stty without a terminal (bash -l with no
# tty, as every hop here runs it) prints this; it means nothing here. Dropped
# before READY only, where it is noise; later lines all pass.
# The local forward listens from the moment ssh connects: a connection that
# comes before the server listens (an old GUI tab retrying) makes ssh print
# "channel N: open failed: connect failed: Connection refused". Dropped until
# the tunnel answers ($TMPD/tunnel), shown after it (REFUSED_NOISE_RE in connect.py).
STTY_NOISE="Inappropriate ioctl for device"
err_filter() {  # ssh's stderr -> $TMPD/err (all of it) and this terminal (without the noise above)
  local l
  while IFS= read -r l || [ -n "$l" ]; do
    printf '%s\n' "$l" >> "$TMPD/err"
    case "$l" in
      "stty: "*"standard input"*": $STTY_NOISE"*) [ -e "$TMPD/ready" ] || continue ;;
      "channel "*": open failed: connect failed: Connection refused"*) [ -e "$TMPD/tunnel" ] || continue ;;
    esac
    printf '%s\n' "$l" >&2
  done
}

WHERE="${NODE:-$GATEWAY}"
# --update-remote: `mace update` in the cluster's MACE checkout first, on the
# gateway (it shares your home with the nodes), interactive with a terminal.
# Exit 3 from the cluster: its MACE has no update command (older than 1.1.6).
if [ -n "$UPDATE_REMOTE" ]; then
  mace_fallback
  if [ -n "$FIND" ]; then
    # shellcheck disable=SC2016  # shell code for the cluster, expanded there
    UM='"$M"'
  else
    UM=""
    set -f
    for w in $MACE; do UM="${UM:+$UM }$(home_path "$w")"; done
    set +f
  fi
  UARGS="--channel $UPDATE_REMOTE"
  UOPTS=(-o BatchMode=yes)
  if [ -t 0 ]; then UOPTS=(-t); else UARGS="$UARGS --yes"; fi
  UCMD="if $UM update --help 2>/dev/null | grep -q -e --remote-only; then $UM update $UARGS; else exit 3; fi"
  [ -n "$FIND" ] && UCMD="{ $FIND; } >/dev/null 2>&1; $UCMD"
  USCRIPT="cd -- \"\$HOME\" && $UCMD"
  [ -n "$CONDA" ] && USCRIPT="conda activate $(home_path "$CONDA"); $USCRIPT"
  [ -n "$SETUP" ] && USCRIPT="${SETUP%;}; $USCRIPT"
  UHOST="${GATEWAY:-$NODE}"
  echo "Updating MACE on $UHOST ($UPDATE_REMOTE) ..."
  UJ=()
  [ -n "$JUMP" ] && UJ=(-J "$JUMP")
  # the jump hosts on the way to $UHOST (the nested hop's): 120 s when one may prompt
  UHOP="$HOP"; HOP=nested
  chain_setup "$UHOST"
  HOP="$UHOP"
  ct_setup "$UHOST"
  URC=0
  if [ "$PROMPT" = 1 ] && first_hop_timed_out; then
    echo "mace-gui-connect: cannot reach host $SLOW (timeout): check the name, and your network or VPN." >&2
    URC=255
  else
    ssh ${UJ[@]+"${UJ[@]}"} "${UOPTS[@]}" ${CT_OPTS[@]+"${CT_OPTS[@]}"} ${SSH_OPTIONS[@]+"${SSH_OPTIONS[@]}"} "$UHOST" "exec bash -lc $(shq "$USCRIPT")" || URC=$?
  fi
  case "$URC" in
    0) ;;
    1) echo "mace-gui-connect: the MACE on $UHOST was not updated (you said no). Connecting anyway." >&2 ;;
    2) echo "mace-gui-connect: mace update on $UHOST stopped without updating (its reason is above). Connecting anyway." >&2 ;;
    3) echo "mace-gui-connect: the MACE on $UHOST is older than 1.1.6 and has no update command: update it once by hand (git -C <MACE folder> pull). Connecting anyway." >&2 ;;
    255) echo "mace-gui-connect: ssh to $UHOST failed (see above), so mace update did not run. Connecting anyway." >&2 ;;
    *) echo "mace-gui-connect: mace update did not finish on $UHOST (exit $URC; see above). Connecting anyway." >&2 ;;
  esac
fi
if gw_unresolvable; then
  echo "Warning: $UNRES matches no Host in ~/.ssh/config and does not resolve (DNS): check the gateway's name (your cluster's login host, e.g. login.example.edu). Connecting anyway." >&2
fi
# ssh's stdin: a FIFO this script writes to (ping, detach; its EOF stops the
# server). ssh's stdout: a regular file, polled below. Under MobaXterm
# (Cygwin), ssh's READY line written to a FIFO stdout reached this script
# only after a key press in the terminal; a file is always writable.
mkfifo "$TMPD/in"
: > "$TMPD/out"
RP="$(rand_port)"
GP="$(rand_port)"
build_argv "$LP" "$RP" "$GP"
# ssh waits 120 s for the jump hosts now: a first one that does not answer fails fast here
if [ "$PROMPT" = 1 ] && first_hop_timed_out; then
  echo "mace-gui-connect: cannot reach host $SLOW (timeout): check the name, and your network or VPN." >&2
  echo "mace-gui-connect: $0 --check (with the same profile or flags) tests each step of the connection." >&2
  exit 1
fi
UNLOADED=""
if unloaded_key; then
  if [ "$AGENT_UP" = 1 ]; then UCMD="ssh-add $UNLOADED"; else UCMD="eval \"\$(ssh-agent)\" && ssh-add $UNLOADED"; fi
  TIMES="on each connection"
  [ "$HOP" = jump ] && [ -n "$NODE" ] && [ -n "$GATEWAY" ] && TIMES="for the gateway and again for the node"
  echo "$UNLOADED has a passphrase and no ssh agent holds it, so ssh asks for it $TIMES. To type it once per session, first run: $UCMD"
fi
if [ -n "$NODE" ] && [ -n "$GATEWAY" ]; then echo "Connecting to $WHERE through $GATEWAY..."; else echo "Connecting to $WHERE..."; fi
echo "(answer ssh's questions here: a password, 2FA, or on a first connection the host-key yes/no question)"
JUMPED=0
if [ "$HOP" = jump ] && [ -n "$NODE" ] && [ -n "$GATEWAY" ]; then
  JUMPED=1
  echo "(waiting for $NODE through the gateway $GATEWAY, up to $WAIT_S s; if $NODE is down, press Ctrl+C and pick another node)"
fi
"${ARGV[@]}" < "$TMPD/in" > "$TMPD/out" 2> >(err_filter) &
SSH_PID=$!
if [ "$JUMPED" = 1 ]; then
  node_probe &
  NODE_PROBE=$!
fi
exec 4>"$TMPD/in"
STATE="wait"
handle_line() {  # one line of ssh's stdout; returns 1 when the session is over
  local line="$1"
  case "$line" in
    "MACE_GUI_READY "*)
      PREFIX="$(json_field prefix "$line")"
      TOKEN="$(json_field token "$line")"
      LOGIN="$(json_field login "$line")"
      BASE="http://127.0.0.1:$LP$PREFIX"
      T0=$SECONDS       # up to 20 s, as mace gui connect waits (READY_POLL_S)
      while [ $((SECONDS - T0)) -lt 20 ]; do
        if http_ok "$LP" "${PREFIX}api/info" "$TOKEN"; then : > "$TMPD/tunnel"; break; fi
        sleep 0.3
      done
      if [ ! -e "$TMPD/tunnel" ]; then   # quiet while waiting; part of the diagnosis now
        grep -E '^channel [0-9]+: open failed: connect failed: Connection refused' "$TMPD/err" 2>/dev/null | tail -n 3 >&2
      fi
      if [ -n "$LOGIN" ]; then
        URL="$(printf '%s' "$LOGIN" | sed "s#^http://127\.0\.0\.1:[0-9]*#http://127.0.0.1:$LP#")"
      else
        URL="$BASE?token=$TOKEN"
      fi
      RV="$(json_field version "$line")"
      [ -n "$RV" ] && echo "MACE on $WHERE: $RV"
      case "$line" in *'"reattached": true'*) echo "Reattached to the server still running on $WHERE." ;; esac
      open_browser "$URL"
      echo "MACE GUI on $WHERE: $URL"
      echo "Press Ctrl+C to stop the server on $WHERE."
      ( while sleep 60; do echo ping >&4 || exit 0; done ) &
      PINGER=$!
      : > "$TMPD/ready"
      STATE="ready" ;;
    "MACE_GUI_LOGIN "*)
      URL="$(json_field url "$line" | sed "s#^http://127\.0\.0\.1:[0-9]*#http://127.0.0.1:$LP#")"
      echo "New sign-in link: $URL"; open_browser "$URL" ;;
    "MACE_GUI_STOPPED "*)
      case "$line" in
        *'"browser"'*) echo "The server on $WHERE was stopped from the browser." ;;
        *'"cpu-limit"'*) echo "The server reached the node's CPU-time limit and stopped cleanly. Run this script again to start a fresh one." ;;
        *) echo "The server on $WHERE stopped." ;;
      esac
      STATE="stopped"; return 1 ;;
    "MACE_GUI_USING "*)
      echo "mace is not on $WHERE's PATH there; using ${line#MACE_GUI_USING }. To skip the search, pass --mace ${line#MACE_GUI_USING }." ;;
    "MACE_GUI_ERROR "*) echo "mace-gui-connect: the server reported: ${line#MACE_GUI_ERROR }" >&2; STATE="error"; return 1 ;;
    "") ;;
    *) echo "remote: $line" ;;
  esac
  return 0
}
# follow the file as ssh writes it: whole lines as they come, until ssh ends
exec 3< "$TMPD/out"
PARTIAL=""
while :; do
  if IFS= read -r CHUNK <&3; then
    handle_line "$PARTIAL$CHUNK" || break
    PARTIAL=""
    continue
  fi
  PARTIAL="$PARTIAL$CHUNK"
  if [ "$STATE" = wait ] && [ -f "$TMPD/probe" ] && grep -qx 'MACE_NODE_PROBE 124' "$TMPD/probe"; then
    echo "mace-gui-connect: the node $NODE did not answer through the gateway $GATEWAY: it may be down or not accept logins (its ssh sent nothing within 8 s when the gateway tried it). Try another node, or $0 --check." >&2
    # its -J helper (ssh -W) too: it would wait on for the node that does not answer
    HELPERS="$(pgrep -P "$SSH_PID" 2>/dev/null)"
    kill "$SSH_PID" 2>/dev/null
    # shellcheck disable=SC2086  # one pid per word
    [ -n "$HELPERS" ] && kill $HELPERS 2>/dev/null
    SSH_PID=""                # gone: cleanup has nothing to say to it (a write to its stdin would be a SIGPIPE)
    exec 4>&- 2>/dev/null
    exit 1
  fi
  if ! kill -0 "$SSH_PID" 2>/dev/null; then
    # ssh is gone: what it wrote last, then stop
    while IFS= read -r CHUNK <&3; do handle_line "$PARTIAL$CHUNK" || break 2; PARTIAL=""; done
    PARTIAL="$PARTIAL$CHUNK"
    [ -n "$PARTIAL" ] && handle_line "$PARTIAL"
    break
  fi
  sleep 0.2
done
exec 3<&-
wait "$SSH_PID" 2>/dev/null
CODE=$?
SSH_PID=""
case "$STATE" in
  stopped) exit 0 ;;
  ready) echo "The connection to $WHERE closed. If GUI tasks were running, the server keeps them running: run this script again to reattach."; exit 1 ;;
esac
ERR="$(cat "$TMPD/err" 2>/dev/null)"
key_failed() {  # no key got ssh in: the same test as mace gui connect (KEY_FAILURE_RE in connect.py)
  [ "$CODE" = 255 ] && printf '%s\n' "$ERR" | grep -Eq "Permission denied \\([^)]*publickey|Enter passphrase for key|read_passphrase: can't open|incorrect passphrase supplied|agent refused operation"
}
key_hint() {
  echo "mace-gui-connect: ssh could not log in with your key (see ssh's message above). If the key has a passphrase, ssh cannot ask for it here: load it into an agent first, with eval \"\$(ssh-agent)\" and ssh-add (on Windows: start the OpenSSH Authentication Agent service, then run ssh-add), then run this script again. A Host alias from ~/.ssh/config also works as the gateway." >&2
}
case "$ERR" in
  *"REMOTE HOST IDENTIFICATION HAS CHANGED"*)
    echo "mace-gui-connect: ssh refused a host key that has CHANGED since this computer last connected (see ssh's message above). That can mean an attack, or that the host was reinstalled: ask your administrators before you remove the old key (ssh-keygen -R <host>)." >&2
    exit 1 ;;
  *"Host key verification failed"*)
    # never trusted silently: the user checks the fingerprint and answers once. With -J the
    # gateway's key is checked here too, by its own ssh process (host_key_leg in connect.py)
    GW_TRUST="ssh ${JUMP:+-J $JUMP }$GATEWAY true"
    LEG=""
    if [ "$HOP" = jump ] && [ -n "$NODE" ] && [ -n "$GATEWAY" ]; then
      HK="$(printf '%s\n' "$ERR" | tr -d '\r' | sed -n -e "s/.*The authenticity of host '\[\{0,1\}\([^' ]*\)\]\{0,1\}\(:[0-9]*\)\{0,1\} (\([^)]*\))'.*/\1 \3/p" \
            -e 's/.*No [^ ]* host key is known for \[\{0,1\}\([^] ]*\)\]\{0,1\}\(:[0-9]*\)\{0,1\} and.*/\1 -/p' | head -n 1 | tr '[:upper:]' '[:lower:]')"
      HK_HOST="${HK%% *}"
      GW_NAME="$(ssh -G "$GATEWAY" 2>/dev/null | awk '$1 == "hostname" { print $2; exit }' | tr '[:upper:]' '[:lower:]')"
      NODE_LC="$(printf '%s' "${NODE##*@}" | tr '[:upper:]' '[:lower:]')"
      GW_LC="$(printf '%s' "${GATEWAY##*@}" | tr '[:upper:]' '[:lower:]')"
      if [ -n "$HK" ]; then
        if [ "$HK_HOST" = "$NODE_LC" ] || [ "${HK#* }" = "<no hostip for proxy command>" ]; then LEG=node
        elif [ "$HK_HOST" = "$GW_LC" ] || [ "$HK_HOST" = "$GW_NAME" ]; then LEG=gateway; fi
      fi
      [ -n "$LEG" ] || LEG=both
    fi
    case "$LEG" in
      gateway)
        echo "mace-gui-connect: ssh could not verify the gateway $GATEWAY's host key (see ssh's message above): this computer has not connected to it before, and nobody answered yes to ssh's question. Check the key, then trust it once: run" >&2
        echo "    $GW_TRUST" >&2 ;;
      both)
        echo "mace-gui-connect: ssh could not verify a host key (see ssh's message above): this computer has not connected to that host before, and nobody answered yes to ssh's question. With the jump hop this computer checks both keys itself, the gateway's first, then the node's. Check them, then trust them once: run" >&2
        echo "    $GW_TRUST" >&2
        echo "then" >&2
        echo "    $TRUST_CMD" >&2 ;;
      *)
        echo "mace-gui-connect: ssh could not verify the host key (see ssh's message above): this computer has not connected to that host before, and nobody answered yes to ssh's question. Check the key, then trust it once: run" >&2
        echo "    $TRUST_CMD" >&2 ;;
    esac
    echo "in this terminal, compare the fingerprint with your cluster's, answer yes, and run this script again." >&2
    if [ "$LEG" = node ] || [ "$LEG" = both ]; then
      echo "mace-gui-connect: or use --hop nested: the gateway then checks the node's key with its own list of known hosts." >&2
    fi
    exit 1 ;;
  *"cd: "*": No such file or directory"*|*"cd: can't cd to "*)
    FOLDER="$(printf '%s\n' "$ERR" | tr -d '\r' | sed -n -e "s/.*cd: can't cd to \(.*\)\$/\1/p" \
              -e 's/.*cd: \(.*\): No such file or directory$/\1/p' | head -n 1)"
    echo "mace-gui-connect: the folder $FOLDER does not exist on $WHERE (the remote shell could not cd to it): check --dir, a path on the node (~/... is the node's home folder)." >&2
    exit 2 ;;
  *"unrecognized arguments: --managed"*|*"invalid choice: 'gui'"*)
    echo "mace-gui-connect: the MACE on $WHERE is older than this script (it has no managed GUI). Update it, or use the manual tunnel (docs/gui.md)." >&2; exit 3 ;;
  *"administratively prohibited"*)
    echo "mace-gui-connect: your cluster turns off port forwarding on this host; ask your administrators." >&2; exit 1 ;;
  *"Permission denied ("*)
    if [ "$HOP" = jump ] && [ -n "$NODE" ] && [ -n "$GATEWAY" ]; then
      echo "mace-gui-connect: if your cluster's nodes accept logins only from the login node, add --hop nested." >&2
      [ "$HOP_AUTO" = 1 ] && echo "mace-gui-connect: (--hop auto works as jump in this script: it does not retry with nested by itself, as mace gui connect does.)" >&2
      echo "mace-gui-connect: with the jump hop, $NODE checks your key from this computer; if it refused it (or asked for a password), see docs/gui.md#mace-gui-connect-the-node-asks-for-a-password." >&2
    fi
    key_failed && key_hint
    exit 1 ;;
esac
key_failed && { key_hint; exit 1; }
if [ "$CODE" = 255 ] && unreach "$ERR" "$NODE" && { [ "$UR_HOST" = UNKNOWN ] || { [ "$UR_HOST" = "the host" ] && [ -n "$CHAIN" ]; }; }; then
  # ssh's name for a target reached through a jump host: it did not answer in time (behind_jump_message)
  if [ "$HOP" = jump ] && [ -n "$NODE" ] && [ -n "$GATEWAY" ]; then WHAT="the node $NODE" THROUGH="the gateway $GATEWAY"
  else VIA="$(printf '%s\n' "$CHAIN" | grep -vx proxycommand | joined)"; WHAT="${GATEWAY:-$NODE}" THROUGH="${VIA:-its proxy}"; fi
  echo "mace-gui-connect: $WHAT did not answer through $THROUGH: it may be down or not accept logins (try another node, or $0 --check). Or the login took too long: ssh waited $WAIT_S s, and that includes logging in to $THROUGH (ssh calls the node UNKNOWN port 65535 then). If typing the passphrase or approving 2FA took longer, load your key into the agent once with ssh-add ${UNLOADED:-<your key>}, so that ssh asks for the passphrase once instead of on each hop, or allow more time with --ssh-option=-oConnectTimeout=300." >&2
  exit 1
fi
if [ "$CODE" = 255 ] && unreach "$ERR" "$NODE"; then
  echo "mace-gui-connect: cannot reach host $UR_HOST ($UR_WHY): check the name, and your network or VPN." >&2
  echo "mace-gui-connect: $0 --check (with the same profile or flags) tests each step of the connection." >&2
  exit 1
fi
if [ "$CODE" = 127 ]; then
  echo "mace-gui-connect: the remote shell did not find mace: pass --mace <full path> or a --setup line." >&2
  mace_fallback
  [ -n "$FIND" ] && echo "mace-gui-connect: it also looked for mace_cli in \$MACE_HOME, ~/mace and ~/MACE there." >&2
  [ -n "$MOVED" ] && echo "mace-gui-connect: $MOVED came from this computer's home folder (your shell expands an unquoted ~ here): check that it exists on $WHERE. Quote the ~ ('~/...') so that your shell passes it as is." >&2
  exit 127
fi
echo "mace-gui-connect: the connection ended before MACE GUI was ready (exit code $CODE)." >&2
exit 1
