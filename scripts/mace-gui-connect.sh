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
(written by mace gui connect --save), then the defaults in this script.
This script does not support --hop auto or --slurm (mace gui connect does auto).
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

unbasic() {  # the inside of a TOML basic string -> its value
  printf '%s' "$1" | sed -e 's/\\"/"/g' -e 's/\\t/\t/g' -e 's/\\\\/\\/g'
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
      line="${line#"${line%%[![:space:]]*}"}"
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
      \"*\") val="${val#\"}"; val="${val%\"}"; val="$(unbasic "$val")" ;;
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
          item="$(unbasic "$item")"
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
[ "$HOP" = auto ] && HOP=jump
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
    *) shq "$1" ;;
  esac
}

remote_command() {  # $1 = remote port
  local parts="" script cmd w
  if [ -n "$SETUP" ]; then parts="${SETUP%;}"; fi
  if [ -n "$CONDA" ]; then parts="${parts:+$parts; }conda activate $(shq "$CONDA")"; fi
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

build_argv() {  # $1 local port, $2 remote port, $3 gateway-side port -> ARGV
  local lp="$1" rp="$2" gp="$3" rcmd jumps
  rcmd="$(remote_command "$rp")"
  ARGV=(ssh)
  jumps="$JUMP"
  if [ -z "$NODE" ] || [ -z "$GATEWAY" ]; then
    [ -n "$jumps" ] && ARGV+=(-J "$jumps")
    ARGV+=(-o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4)
    [ ${#SSH_OPTIONS[@]} -gt 0 ] && ARGV+=("${SSH_OPTIONS[@]}")
    ARGV+=(-L "127.0.0.1:$lp:127.0.0.1:$rp" "${NODE:-$GATEWAY}" "$rcmd")
    return
  fi
  if [ "$HOP" = nested ]; then
    local inner
    inner="ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -L 127.0.0.1:$gp:127.0.0.1:$rp $(shq "$NODE") $(shq "$rcmd")"
    [ -n "$jumps" ] && ARGV+=(-J "$jumps")
    ARGV+=(-o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4)
    [ ${#SSH_OPTIONS[@]} -gt 0 ] && ARGV+=("${SSH_OPTIONS[@]}")
    ARGV+=(-L "127.0.0.1:$lp:127.0.0.1:$gp" "$GATEWAY" "$inner")
    return
  fi
  ARGV+=(-J "${jumps:+$jumps,}$GATEWAY")
  ARGV+=(-o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4)
  [ ${#SSH_OPTIONS[@]} -gt 0 ] && ARGV+=("${SSH_OPTIONS[@]}")
  ARGV+=(-L "127.0.0.1:$lp:127.0.0.1:$rp" "$(node_target)" "$rcmd")
}

if [ "$DRY_RUN" = 1 ]; then
  build_argv "$LOCAL_PORT" 40000 40001
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
  (
    exec 5<>"/dev/tcp/127.0.0.1/$1" || exit 1
    printf 'GET %s HTTP/1.0\r\nHost: 127.0.0.1:%s\r\nX-Mace-Token: %s\r\n\r\n' "$2" "$1" "$3" >&5
    IFS= read -r status <&5
    case "$status" in *" 200 "*) exit 0 ;; *) exit 1 ;; esac
  ) 2>/dev/null
}

WHERE="${NODE:-$GATEWAY}"
mkfifo "$TMPD/in" "$TMPD/out"
RP="$(rand_port)"
GP="$(rand_port)"
build_argv "$LP" "$RP" "$GP"
if [ -n "$NODE" ] && [ -n "$GATEWAY" ]; then echo "Connecting to $WHERE through $GATEWAY..."; else echo "Connecting to $WHERE..."; fi
"${ARGV[@]}" < "$TMPD/in" > "$TMPD/out" 2> >(tee "$TMPD/err" >&2) &
SSH_PID=$!
exec 4>"$TMPD/in"
STATE="wait"
while IFS= read -r line; do
  case "$line" in
    "MACE_GUI_READY "*)
      PREFIX="$(json_field prefix "$line")"
      TOKEN="$(json_field token "$line")"
      LOGIN="$(json_field login "$line")"
      BASE="http://127.0.0.1:$LP$PREFIX"
      for _ in $(seq 1 60); do http_ok "$LP" "${PREFIX}api/info" "$TOKEN" && break; sleep 0.3; done
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
      STATE="stopped"; break ;;
    "MACE_GUI_ERROR "*) echo "mace-gui-connect: the server reported: ${line#MACE_GUI_ERROR }" >&2; STATE="error"; break ;;
    "") ;;
    *) echo "remote: $line" ;;
  esac
done < "$TMPD/out"
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
  *"unrecognized arguments: --managed"*|*"invalid choice: 'gui'"*)
    echo "mace-gui-connect: the MACE on $WHERE is older than this script (it has no managed GUI). Update it, or use the manual tunnel (docs/gui.md)." >&2; exit 3 ;;
  *"administratively prohibited"*)
    echo "mace-gui-connect: your cluster turns off port forwarding on this host; ask your administrators." >&2; exit 1 ;;
  *"Permission denied ("*)
    if [ "$HOP" = jump ] && [ -n "$NODE" ] && [ -n "$GATEWAY" ]; then
      echo "mace-gui-connect: if your cluster's nodes accept logins only from the login node, add --hop nested." >&2
    fi
    key_failed && key_hint
    exit 1 ;;
esac
key_failed && { key_hint; exit 1; }
[ "$CODE" = 127 ] && { echo "mace-gui-connect: the remote shell did not find mace: pass --mace <full path> or a --setup line." >&2; exit 127; }
echo "mace-gui-connect: the connection ended before MACE GUI was ready (exit code $CODE)." >&2
exit 1
