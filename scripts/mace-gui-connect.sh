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
  --dry-run          print the ssh command (one argument per line) and exit
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
F_SSH_OPTIONS=() F_ROOTS=()
HAVE_SSH_OPTIONS=0 HAVE_ROOTS=0
DRY_RUN=0
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
    --dry-run) DRY_RUN=1; shift ;;
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
if [ -n "$GATEWAY" ] && email_like "$GATEWAY"; then
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

remote_command() {  # $1 = remote port
  local parts="" script cmd w
  if [ -n "$SETUP" ]; then parts="${SETUP%;}"; fi
  if [ -n "$CONDA" ]; then parts="${parts:+$parts; }conda activate $(home_path "$CONDA")"; fi
  cmd="exec"
  [ -z "$NODE" ] && cmd="$cmd env MACE_GUI_MAX_WORKERS=2"
  # the mace command may be several words (e.g. python3 ~/mace/mace_cli)
  set -f
  for w in $MACE; do cmd="$cmd $(home_path "$w")"; done
  set +f
  cmd="$cmd --no-banner gui --managed --port $1 --dir ."
  if [ ${#ROOTS[@]} -gt 0 ]; then
    for w in "${ROOTS[@]}"; do cmd="$cmd --root $(home_path "$w")"; done
  fi
  script="cd -- $(home_path "${DIR:-~}") && $cmd"
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

build_argv() {  # $1 local port, $2 remote port, $3 gateway-side port -> ARGV
  local lp="$1" rp="$2" gp="$3" rcmd jumps
  rcmd="$(remote_command "$rp")"
  ARGV=(ssh)
  jumps="$JUMP"
  # TRUST_CMD: the line that asks once to trust the host key(s) this session needs (host_key_trust_command)
  if [ -z "$NODE" ] || [ -z "$GATEWAY" ]; then
    TRUST_CMD="ssh ${jumps:+-J $jumps }${NODE:-$GATEWAY} true"
    [ -n "$jumps" ] && ARGV+=(-J "$jumps")
    ARGV+=(-o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4)
    [ ${#SSH_OPTIONS[@]} -gt 0 ] && ARGV+=("${SSH_OPTIONS[@]}")
    ARGV+=(-L "127.0.0.1:$lp:127.0.0.1:$rp" "${NODE:-$GATEWAY}" "$rcmd")
    return
  fi
  if [ "$HOP" = nested ]; then
    TRUST_CMD="ssh ${jumps:+-J $jumps }$GATEWAY true"
    local inner
    inner="ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -L 127.0.0.1:$gp:127.0.0.1:$rp $(shq "$NODE") $(shq "$rcmd")"
    [ -n "$jumps" ] && ARGV+=(-J "$jumps")
    ARGV+=(-o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4)
    [ ${#SSH_OPTIONS[@]} -gt 0 ] && ARGV+=("${SSH_OPTIONS[@]}")
    ARGV+=(-L "127.0.0.1:$lp:127.0.0.1:$gp" "$GATEWAY" "$inner")
    return
  fi
  local target
  target="$(node_target)"
  node_keys "$target"
  TRUST_CMD="ssh -J ${jumps:+$jumps,}$GATEWAY $target true"
  ARGV+=(-J "${jumps:+$jumps,}$GATEWAY")
  ARGV+=(-o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4)
  [ ${#NODE_KEYS[@]} -gt 0 ] && ARGV+=("${NODE_KEYS[@]}")
  [ ${#SSH_OPTIONS[@]} -gt 0 ] && ARGV+=("${SSH_OPTIONS[@]}")
  ARGV+=(-L "127.0.0.1:$lp:127.0.0.1:$rp" "$target" "$rcmd")
}

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
cleanup() {
  trap - INT TERM HUP EXIT
  if [ -n "$SSH_PID" ]; then
    { echo detach >&4; } 2>/dev/null
    exec 4>&- 2>/dev/null
    for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$SSH_PID" 2>/dev/null || break; sleep 0.5; done
    kill "$SSH_PID" 2>/dev/null
  fi
  [ -n "$PINGER" ] && kill "$PINGER" 2>/dev/null
  rm -rf "$TMPD"
}
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM HUP
trap cleanup EXIT

open_browser() {
  if [ "$OPEN" = 0 ]; then return; fi
  if [ -n "${BROWSER:-}" ]; then "$BROWSER" "$1" >/dev/null 2>&1 &
  elif command -v open >/dev/null 2>&1 && [ "$(uname)" = Darwin ]; then open "$1"
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$1" >/dev/null 2>&1 &
  elif command -v cygstart >/dev/null 2>&1; then cygstart "$1"
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
# ssh's stdin: a FIFO this script writes to (ping, detach; its EOF stops the
# server). ssh's stdout: a regular file, polled below. Under MobaXterm
# (Cygwin), ssh's READY line written to a FIFO stdout reached this script
# only after a key press in the terminal; a file is always writable.
mkfifo "$TMPD/in"
: > "$TMPD/out"
RP="$(rand_port)"
GP="$(rand_port)"
build_argv "$LP" "$RP" "$GP"
if [ -n "$NODE" ] && [ -n "$GATEWAY" ]; then echo "Connecting to $WHERE through $GATEWAY..."; else echo "Connecting to $WHERE..."; fi
echo "(answer ssh's questions here: a password, 2FA, or on a first connection the host-key yes/no question)"
"${ARGV[@]}" < "$TMPD/in" > "$TMPD/out" 2> >(err_filter) &
SSH_PID=$!
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
if [ "$CODE" = 127 ]; then
  echo "mace-gui-connect: the remote shell did not find mace: pass --mace <full path> or a --setup line." >&2
  [ -n "$MOVED" ] && echo "mace-gui-connect: $MOVED came from this computer's home folder (your shell expands an unquoted ~ here): check that it exists on $WHERE. Quote the ~ ('~/...') so that your shell passes it as is." >&2
  exit 127
fi
echo "mace-gui-connect: the connection ended before MACE GUI was ready (exit code $CODE)." >&2
exit 1
