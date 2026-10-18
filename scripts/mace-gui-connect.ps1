<#
.SYNOPSIS
  Start MACE GUI on a cluster node and open it in this computer's browser, with one ssh session.

.DESCRIPTION
  The Windows (PowerShell 5.1+) twin of `mace gui connect`: same options, same profiles file
  (%APPDATA%\mace\connect.toml), same protocol. Needs the ssh built into Windows 10 1809+ (-J needs
  OpenSSH 8.1+; check with ssh -V). Ctrl+C stops the server on the node; GUI tasks that are still
  running keep it alive until they finish, and running this script again reattaches.
  -Manual prints the two-terminal manual commands instead of connecting.
  -Check checks ssh, its config, keys, the agent service and each hop (running only `true`) and
  says which works, read-only (-NoNet: no ssh logins or DNS lookups); -FixConfig prints a suggested
  ~/.ssh/config block (never written). The same checks and words as mace gui connect --check.
  -UpdateRemote release|preview|latest runs mace update on the cluster first (as --update-remote).

  Downloaded scripts are blocked by Windows: run Unblock-File .\mace-gui-connect.ps1 once, then
  powershell -ExecutionPolicy Bypass -File .\mace-gui-connect.ps1 -Gateway <user>@<gateway> -Node <node> -Dir <folder>

.EXAMPLE
  .\mace-gui-connect.ps1 -Gateway me@hpcc.example.edu -Node dev-node-01 -Dir ~/runs
#>
[CmdletBinding()]
param(
  [Parameter(Position = 0)][string]$ProfileName = "",
  [string]$Gateway = "",
  [string]$Jump = "",
  [string]$Node = "",
  [ValidateSet("", "auto", "jump", "nested")][string]$Hop = "",
  [string]$Dir = "",
  [string]$Mace = "",
  [string]$Setup = "",
  [string]$Conda = "",
  [int]$LocalPort = 0,
  [string[]]$SshOption = @(),
  [string[]]$Root = @(),
  [switch]$NoOpen,
  [switch]$DryRun,
  [switch]$Manual,
  [switch]$Check,
  [switch]$NoNet,
  [switch]$FixConfig,
  [ValidateSet("", "release", "preview", "latest")][string]$UpdateRemote = ""
)

# >>> mace-gui-connect defaults >>>
$D_GATEWAY = ""
$D_JUMP = ""
$D_NODE = ""
$D_HOP = "jump"
$D_DIR = ""
$D_MACE = "mace"
$D_SETUP = ""
$D_CONDA = ""
$D_LOCAL_PORT = 8765
$D_SSH_OPTIONS = @()
$D_ROOTS = @()
$D_OPEN = $true
# <<< mace-gui-connect defaults <<<

$ErrorActionPreference = "Stop"

function Fail([string]$msg, [int]$code) {
  # not Write-Error: under "Stop" it ends the script at once, with exit code 1 and a stack trace
  [Console]::Error.WriteLine("mace-gui-connect: $msg")
  exit $code
}

# Windows ssh takes one command line: quote each argument for CreateProcess
function Win-Quote([string]$s) {
  if ($s -notmatch '[\s"]' -and $s -ne "") { return $s }
  return '"' + ($s -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"'
}

# Every native program (ssh -V, ssh -G) runs through Invoke-Native, never as
# "& ssh ... 2>&1" or "2>$null": in Windows PowerShell 5.1 a redirected stderr
# line of a native command becomes an ErrorRecord (NativeCommandError), and
# with $ErrorActionPreference = "Stop" the first one ends the script
# (about_Redirection, about_Preference_Variables; PowerShell 7.2+ no longer
# does this). ssh -V prints its version on stderr, so every run failed. A
# Process with both streams read here keeps PowerShell's streams out of it.
function Invoke-Native([string]$exe, [string[]]$argList) {
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = $exe
  $psi.Arguments = (@($argList | ForEach-Object { Win-Quote $_ }) -join " ")
  $psi.UseShellExecute = $false
  $psi.CreateNoWindow = $true
  $psi.RedirectStandardInput = $true
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError = $true
  try { $p = [System.Diagnostics.Process]::Start($psi) } catch { return @{ Code = -1; Out = ""; Err = "$_" } }
  $p.StandardInput.Close()
  $errTask = $p.StandardError.ReadToEndAsync()     # both pipes drained: neither can fill up and block ssh
  $out = $p.StandardOutput.ReadToEnd()
  $p.WaitForExit()
  return @{ Code = $p.ExitCode; Out = $out; Err = $errTask.Result }
}

# ------------------------------------------------------------------ profile (flat TOML subset)
function Unquote-Basic([string]$s) {
  # One pass over the escapes, as Python's tomllib reads a basic string.
  # (Replacing \\ and \t one after the other read "C:\\Users\\me\\tmp" as
  # C:\Users\me<TAB>mp.)
  return [regex]::Replace($s, '\\(?:u([0-9A-Fa-f]{4})|U([0-9A-Fa-f]{8})|([\s\S]))', {
      param($m)
      if ($m.Groups[1].Success) { return [string][char][Convert]::ToInt32($m.Groups[1].Value, 16) }
      if ($m.Groups[2].Success) { return [char]::ConvertFromUtf32([Convert]::ToInt32($m.Groups[2].Value, 16)) }
      switch -CaseSensitive ($m.Groups[3].Value) {
        'b' { return [string][char]8 }
        't' { return "`t" }
        'n' { return "`n" }
        'f' { return [string][char]12 }
        'r' { return "`r" }
        default { return $m.Groups[3].Value }     # \" and \\ (and anything else) stand for the character
      }
    })
}

function Read-Profile([string]$want) {
  $conf = Join-Path $env:APPDATA "mace\connect.toml"
  $p = @{}
  if (-not (Test-Path $conf)) { return $p }
  $lines = Get-Content -LiteralPath $conf -Encoding UTF8     # mace gui connect writes UTF-8 (5.1 reads ANSI otherwise)
  $section = ""
  if (-not $want) {
    foreach ($raw in $lines) {
      $line = $raw.Trim()
      if ($line -match '^\[(.+)\]$') { $section = $Matches[1]; continue }
      if ($section -eq "defaults" -and $line -match '^profile\s*=\s*"(.*)"$') { $want = $Matches[1] }
    }
  }
  if (-not $want) { return $p }
  $section = ""
  $n = 0
  foreach ($raw in $lines) {
    $n++
    $line = $raw.Trim()
    if (-not $line -or $line.StartsWith("#")) { continue }
    if ($line -match '^\[(.+)\]$') { $section = $Matches[1]; continue }
    if ($section -ne "profiles.$want") { continue }
    if ($line -notmatch '^([A-Za-z0-9_-]+)\s*=\s*(.*)$') {
      Write-Warning "connect.toml uses a form this script cannot read (line $n); pass the values as flags"; continue
    }
    $key = $Matches[1]; $val = $Matches[2].Trim()
    if ($val -match '^"((?:[^"\\]|\\.)*)"$') { $p[$key] = Unquote-Basic $Matches[1] }
    elseif ($val -eq "true") { $p[$key] = $true }
    elseif ($val -eq "false") { $p[$key] = $false }
    elseif ($val -match '^-?\d+$') { $p[$key] = [int]$val }
    elseif ($val -match '^\[.*\]$') {
      $items = [regex]::Matches($val, '"((?:[^"\\]|\\.)*)"') | ForEach-Object { Unquote-Basic $_.Groups[1].Value }
      $p[$key] = @($items)
    }
    else { Write-Warning "connect.toml uses a form this script cannot read (line $n); pass the values as flags" }
  }
  return $p
}

$P = Read-Profile $ProfileName
function Pick($flag, $key, $default) {
  if ($flag) { return $flag }
  if ($P.ContainsKey($key) -and $P[$key] -ne "") { return $P[$key] }
  return $default
}
$GatewayV = Pick $Gateway "gateway" $D_GATEWAY
$JumpV = Pick $Jump "jump" $D_JUMP
$NodeV = Pick $Node "node" $D_NODE
$HopV = Pick $Hop "hop" $D_HOP
# hop auto: mace gui connect tries jump and falls back to nested when the node
# refuses the login, which it reads in ssh's messages. This script leaves those
# messages to the console (so ssh's prompts work), so auto is jump here, and a
# refusal says to pass -Hop nested.
$HopAuto = ($HopV -eq "auto")
if ($HopAuto) { $HopV = "jump" }
$DirV = Pick $Dir "dir" $D_DIR
$MaceV = Pick $Mace "mace" $D_MACE
$SetupV = Pick $Setup "setup" $D_SETUP
$CondaV = Pick $Conda "conda" $D_CONDA
$LocalPortV = [int](Pick $(if ($LocalPort -gt 0) { $LocalPort } else { $null }) "local_port" $D_LOCAL_PORT)
$SshOptions = @(if ($SshOption.Count) { $SshOption } elseif ($P.ContainsKey("ssh_options")) { $P["ssh_options"] } else { $D_SSH_OPTIONS })
$Roots = @(if ($Root.Count) { $Root } elseif ($P.ContainsKey("roots")) { $P["roots"] } else { $D_ROOTS })
$OpenV = if ($NoOpen) { $false } elseif ($P.ContainsKey("open")) { $P["open"] } else { $D_OPEN }
# A path under THIS computer's home folder is a ~ that a shell expanded here
# (-Mace ~/mace/mace_cli from bash or pwsh on Linux/macOS arrives as
# /home/you/mace/mace_cli, which the node lacks when its home is elsewhere):
# sent as ~/... instead, with a note. The same rule as mace gui connect
# (localize_home_paths). A leading // (//home/you/x) is a literal path on the
# node, sent as /home/you/x.
$script:Moved = @()
function Localize-Home([string]$s) {
  $h = "$HOME".TrimEnd('/', '\')
  if (-not $h -or -not $s) { return $s }
  if ($s.StartsWith("//")) { return $s }   # a literal path on the node (sent as /...): never rewritten
  foreach ($f in @($h, ($h -replace '\\', '/')) | Select-Object -Unique) {
    $new = $null
    if ($s -ceq $f) { $new = "~" }
    elseif ($s.Length -gt $f.Length -and $s.StartsWith($f, [System.StringComparison]::Ordinal) -and (@('/', '\') -contains [string]$s[$f.Length])) {
      $new = "~/" + ($s.Substring($f.Length + 1) -replace '\\', '/')
    }
    if ($new) { $script:Moved += "$s as $new"; return $new }
  }
  return $s
}
$DirV = Localize-Home $DirV
$CondaV = Localize-Home $CondaV
$MaceV = (@($MaceV -split '\s+' | Where-Object { $_ } | ForEach-Object { Localize-Home $_ }) -join " ")
$Roots = @($Roots | ForEach-Object { Localize-Home $_ })
if ($script:Moved.Count) {
  $first = ($script:Moved[0] -split ' as ')[-1]
  Write-Warning "Note: treated $($script:Moved -join ', ') on the remote (your shell turned ~ into this computer's home folder; quote the ~, as '$first', to avoid this)."
}
if (-not $GatewayV -and -not $NodeV) {
  Fail "Pass -Gateway <login host> (and usually -Node <dev node> and -Dir <folder>), or save a profile with mace gui connect --save." 2
}

# A gateway such as user@university.edu is an email address, not a login host
# (user@login.university.edu): flagged when the part after @ is two labels
# (name.tld) and no ~/.ssh/config Host pattern matches it. The same heuristic
# as mace gui connect (gateway_looks_like_email in connect.py, which explains
# the tradeoff); a warning only.
function Test-EmailLike([string]$gw) {
  if ($gw -notmatch '@([^@]+)$') { return $false }
  $h = $Matches[1].ToLower()
  if ($h -notmatch '^[a-z0-9-]+\.[a-z]{2,}$') { return $false }
  $cfg = Join-Path $env:USERPROFILE ".ssh\config"
  if (Test-Path -LiteralPath $cfg) {
    foreach ($l in Get-Content -LiteralPath $cfg) {
      if ($l -match '^\s*Host(?:\s+|\s*=\s*)(.+)$') {
        foreach ($pat in ($Matches[1].Trim() -split '\s+')) {
          $pat = $pat.Trim('"')
          if (-not $pat -or $pat.StartsWith("!")) { continue }
          if ($h -like $pat -or $gw -like $pat) { return $false }
        }
      }
    }
  }
  return $true
}
if ($GatewayV -and -not ($Check -or $FixConfig) -and (Test-EmailLike $GatewayV)) {
  $conf = Join-Path $env:APPDATA "mace\connect.toml"
  $pn = if ($ProfileName) { "$ProfileName " } else { "" }
  Write-Warning "The gateway $GatewayV looks like an email address. It should be user@login-host (your cluster's login node, e.g. netid@login.example.edu) or a Host alias from ~/.ssh/config. Connecting anyway; to fix the profile in $conf, run: mace gui connect $($pn)--gateway <user@login-host> --save (or pass -Gateway to this script)."
}

# ------------------------------------------------------------------ the command (as mace gui connect builds it)
function ShQuote([string]$s) {
  if ($s -eq "") { return "''" }
  if ($s -match '^[a-zA-Z0-9_@%+=:,./-]+$') { return $s }
  return "'" + ($s -replace "'", "'`"'`"'") + "'"
}
function HomePath([string]$s) {
  if ($s.StartsWith("//")) { return ShQuote $s.Substring(1) }
  if ($s -eq "~") { return '"$HOME"' }
  if ($s.StartsWith("~/")) { return '"$HOME"/' + (ShQuote $s.Substring(2)) }
  return ShQuote $s
}
# When the mace command is not on the node's PATH (the default "mace", or a
# one-word path that is not there), the node's shell tries mace_cli in that
# path's folder, $MACE_HOME, ~/mace and ~/MACE, and prints MACE_GUI_USING
# <path> (mace_fallback in connect.py; the same shell code).
function Mace-Fallback {
  $words = @($MaceV -split '\s+' | Where-Object { $_ })
  if ($words.Count -ne 1) { return "" }
  $w = $words[0]
  $fallbacks = @('${MACE_HOME:+"$MACE_HOME"/mace_cli}', '"$HOME"/mace/mace_cli', '"$HOME"/MACE/mace_cli')
  # the node's shell test, as connect.py writes it ([char]62 is ">": a stream redirection
  # written out here would read as PowerShell's own, which this script never uses)
  if ($w -ceq "mace") { $t = 'command -v "$M" >/dev/null 2' + [char]62 + '&1'; $cands = $fallbacks }
  elseif ($w.Contains("/") -and -not $w.EndsWith("/mace_cli") -and -not $w.EndsWith("/")) {
    $t = '[ -x "$M" ]'
    $own = HomePath ($w.Substring(0, $w.LastIndexOf("/")) + "/mace_cli")
    $cands = @($own) + @($fallbacks | Where-Object { $_ -cne $own })
  }
  else { return "" }
  return "M=" + (HomePath $w) + "; " + $t + " || for c in " + ($cands -join " ") + '; do if [ -x "$c" ]; then M=$c; echo "MACE_GUI_USING $c"; break; fi; done'
}
function Remote-Command([int]$rp) {
  $parts = @()
  if ($SetupV) { $parts += $SetupV.TrimEnd(";") }
  if ($CondaV) { $parts += "conda activate " + (HomePath $CondaV) }
  $cmd = @("exec")
  if (-not $NodeV) { $cmd += @("env", "MACE_GUI_MAX_WORKERS=2") }
  $find = Mace-Fallback
  if ($find) { $cmd += '"$M"' }
  else { foreach ($w in ($MaceV -split '\s+' | Where-Object { $_ })) { $cmd += (HomePath $w) } }
  $cmd += @("--no-banner", "gui", "--managed", "--port", "$rp", "--dir", ".")
  foreach ($r in $Roots) { $cmd += @("--root", (HomePath $r)) }
  $d = if ($DirV) { $DirV } else { "~" }
  if ($find) { $parts += "cd -- " + (HomePath $d) + " && { " + $find + "; " + ($cmd -join " ") + "; }" }
  else { $parts += "cd -- " + (HomePath $d) + " && " + ($cmd -join " ") }
  return "exec bash -lc " + (ShQuote ($parts -join "; "))
}
# With -J, ssh logs in to the node from this computer with the settings ssh's
# config has for the node's name: the gateway's IdentityFile (and
# IdentityAgent, IdentitiesOnly) are passed for the node hop too, or the node
# gets only the default key files and asks for a password (node_key_options
# in connect.py).
function Ssh-G([string]$h) {
  $m = @{}
  $r = Invoke-Native "ssh" @("-G", $h)
  foreach ($l in ($r.Out -split "`r?`n")) {
    if ($l -match '^(\S+) (.+)$') { $k = $Matches[1].ToLower(); if (-not $m.ContainsKey($k)) { $m[$k] = @() }; $m[$k] += $Matches[2] }
  }
  return $m
}
function Node-Keys([string]$target) {
  $g = Ssh-G $GatewayV; $n = Ssh-G $target
  $out = @()
  foreach ($f in @($g["identityfile"])) {
    if (-not $f -or @($n["identityfile"]) -contains $f) { continue }
    $p = if ($f.StartsWith("~/")) { Join-Path $HOME $f.Substring(2) } else { $f }
    if (Test-Path -LiteralPath $p -PathType Leaf) { $out += @("-o", "IdentityFile=$f") }
  }
  $ga = @($g["identityagent"])[0]
  if ($ga -and $ga -ne @($n["identityagent"])[0]) { $out += @("-o", "IdentityAgent=$ga") }
  if (@($g["identitiesonly"])[0] -eq "yes" -and @($n["identitiesonly"])[0] -ne "yes") { $out += @("-o", "IdentitiesOnly=yes") }
  return $out
}
# The line that asks once, in this window, to trust the host key(s) the
# session needs (host_key_trust_command in connect.py); set by Build-Argv.
$script:TrustCmd = ""
$script:GatewayTrustCmd = ""
# the node, with the gateway's user when that differs from yours (ssh -G for an alias)
function Node-Target {
  $target = $NodeV
  if ($NodeV -notmatch "@") {
    $u = ""
    if ($GatewayV -match "^([^@]+)@") { $u = $Matches[1] }
    else { $u = "$(((Invoke-Native "ssh" @("-G", $GatewayV)).Out -split "`r?`n" | Where-Object { $_ -match "^user " } | Select-Object -First 1))" -replace "^user ", "" }
    if ($u -and $u -ne $env:USERNAME) { $target = "$u@$NodeV" }
  }
  return $target
}
# -o ConnectTimeout=20 unless an -SshOption or your ssh config (ssh -G) sets one: a command-line -o
# would beat the config's own value (a slow VPN's). Val: yours, else 20, for the nested hop's inner
# ssh (CONNECT_TIMEOUT_S, config_connect_timeout in connect.py)
function Ct-Setup([string]$h) {
  $joined = ($SshOptions -join " ")
  $opts = @("-o", "ConnectTimeout=20")
  $v = ""
  if ($joined -match '(?i)connecttimeout') { $opts = @() }
  if ($joined -match '(?i)connecttimeout[ =]*(\d+)') { $v = $Matches[1] }
  if (-not $v -and $h) {
    $c = "" + @((Ssh-G $h)["connecttimeout"])[0]
    if ($c -match '^\d+$' -and [int]$c -gt 0) { $v = $c; $opts = @() }
  }
  return @{ Opts = $opts; Val = $(if ($v) { $v } else { "20" }) }
}
function Build-Argv([int]$lp, [int]$rp, [int]$gp) {
  $a = New-Object System.Collections.ArrayList
  $opts = @("-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4")
  $dest = if (-not $NodeV) { $GatewayV } elseif (-not $GatewayV) { $NodeV } elseif ($HopV -eq "nested") { $GatewayV } else { Node-Target }
  $ct = Ct-Setup $dest
  $opts += $ct.Opts
  $rcmd = Remote-Command $rp
  if (-not $NodeV -or -not $GatewayV) {
    if ($JumpV) { [void]$a.AddRange(@("-J", $JumpV)) }
    [void]$a.AddRange($opts); if ($SshOptions.Count) { [void]$a.AddRange($SshOptions) }
    $target = if ($NodeV) { $NodeV } else { $GatewayV }
    $script:TrustCmd = "ssh " + $(if ($JumpV) { "-J $JumpV " } else { "" }) + "$target true"
    [void]$a.AddRange(@("-L", "127.0.0.1:${lp}:127.0.0.1:${rp}", $target, $rcmd))
    return ,$a
  }
  if ($HopV -eq "nested") {
    $inner = "ssh -o BatchMode=yes -o ConnectTimeout=$($ct.Val) -o ExitOnForwardFailure=yes -L 127.0.0.1:${gp}:127.0.0.1:${rp} " + (ShQuote $NodeV) + " " + (ShQuote $rcmd)
    if ($JumpV) { [void]$a.AddRange(@("-J", $JumpV)) }
    [void]$a.AddRange($opts); if ($SshOptions.Count) { [void]$a.AddRange($SshOptions) }
    [void]$a.AddRange(@("-L", "127.0.0.1:${lp}:127.0.0.1:${gp}", $GatewayV, $inner))
    $script:TrustCmd = "ssh " + $(if ($JumpV) { "-J $JumpV " } else { "" }) + "$GatewayV true"
    return ,$a
  }
  $j = if ($JumpV) { "$JumpV,$GatewayV" } else { $GatewayV }
  $target = Node-Target
  [void]$a.AddRange(@("-J", $j))
  [void]$a.AddRange($opts)
  $keys = @(Node-Keys $target)
  if ($keys.Count) { [void]$a.AddRange($keys) }
  if ($SshOptions.Count) { [void]$a.AddRange($SshOptions) }
  [void]$a.AddRange(@("-L", "127.0.0.1:${lp}:127.0.0.1:${rp}", $target, $rcmd))
  $script:TrustCmd = "ssh -J $j $target true"
  # with -J the gateway's key is checked here too, by its own ssh process (the real known_hosts)
  $script:GatewayTrustCmd = "ssh " + $(if ($JumpV) { "-J $JumpV " } else { "" }) + "$GatewayV true"
  return ,$a
}

# ------------------------------------------------------------------ host names and unreachable hosts (connect.py)
function Config-Hosts {
  $cfg = Join-Path $HOME ".ssh\config"
  $out = @()
  if (Test-Path -LiteralPath $cfg) {
    foreach ($l in Get-Content -LiteralPath $cfg) {
      if ($l -match '^\s*Host(?:\s+|\s*=\s*)(.+)$') {
        foreach ($pat in ($Matches[1].Trim() -split '\s+')) { $pat = $pat.Trim('"'); if ($pat -and -not $pat.StartsWith("!")) { $out += $pat } }
      }
    }
  }
  return $out
}
# The gateway's host name when no ~/.ssh/config Host matches it and it does not
# resolve here (gateway_unresolvable in connect.py); "" otherwise.
function Gw-Unresolvable {
  if (-not $GatewayV -or $JumpV) { return "" }
  $h = ($GatewayV -split '@')[-1]
  foreach ($pat in @(Config-Hosts)) { if ($h.ToLower() -like $pat.ToLower()) { return "" } }
  $g = Ssh-G $GatewayV
  foreach ($k in @("proxyjump", "proxycommand")) { $v = @($g[$k])[0]; if ($v -and $v -ne "none") { return "" } }
  $hn = @($g["hostname"])[0]
  if (-not $hn) { $hn = $h }
  try { [void][System.Net.Dns]::GetHostAddresses($hn); return "" } catch { return $hn }
}
# ssh's words for a host it could not reach: @(host, why), or $null (unreachable in connect.py)
function Unreach([string[]]$lines, [string]$node) {
  foreach ($l in $lines) {
    if ($l -match 'Could not resolve hostname ([^:\s]+)') { return @($Matches[1], "DNS: the name does not resolve") }
    if ($l -match 'connect to host (\S+) port \d+: (?:Connection|Operation) timed out') { return @($Matches[1], "timeout") }
    if ($l -match 'Connection timed out during banner exchange|Connection to (\S+) port \d+ timed out') {
      $h = $Matches[1]
      if (-not $h) { foreach ($x in $lines) { if ($x -match 'Connection to (\S+) port') { $h = $Matches[1]; break } } }
      if (-not $h) { $h = "the host" }
      return @($h, "timeout")
    }
    if ($l -match 'connect to host (\S+) port \d+: Connection refused') { return @($Matches[1], "connection refused") }
    if ($l -match 'connect to host (\S+) port \d+: (?:No route to host|Network is unreachable)') { return @($Matches[1], "no route") }
    if ($node -and $l -match 'open failed: (?:connect failed|administratively prohibited): (.+)') {
      $t = $Matches[1].ToLower()
      $why = $t.Trim()
      if ($t -match 'not known|servname|resolve') { $why = "DNS on the gateway: the name does not resolve" }
      elseif ($t -match 'timed out') { $why = "timeout" }
      elseif ($t -match 'refused') { $why = "connection refused" }
      elseif ($t -match 'route|unreachable') { $why = "no route" }
      return @(($node -split '@')[-1], $why)
    }
  }
  return $null
}

# ------------------------------------------------------------------ -Check and -FixConfig
# The same checks and words as mace gui connect --check (connect_check.py):
# read-only; the private half of a key is never opened.
$OnWindows = ($env:OS -eq "Windows_NT")
$script:CkFails = 0
$script:CkHopFails = 0
function Ck([string]$level, [string]$text, [string]$fix = "") {
  Write-Output ("{0,-5} {1}" -f $level, $text)
  if ($fix) { Write-Output "      fix: $fix" }
  if ($level -eq "FAIL") { $script:CkFails++ }
}
function XPath([string]$f) {
  if ($f -eq "~") { return "$HOME" }
  if ($f.StartsWith("~/") -or $f.StartsWith("~\")) { return (Join-Path $HOME $f.Substring(2)) }
  return $f
}
function Existing-Files($m) {
  return @(@($m["identityfile"]) | Where-Object { $_ -and (Test-Path -LiteralPath (XPath $_) -PathType Leaf) })
}
# a native program's exit code and its output lines, stdout then stderr (through Invoke-Native)
function Run-Native([string]$exe, [string[]]$a) {
  $r = Invoke-Native $exe $a
  $o = @((("" + $r.Out).TrimEnd() -split "`r?`n") + (("" + $r.Err).TrimEnd() -split "`r?`n") | Where-Object { $_ -ne "" })
  return @{ Code = $r.Code; Out = $o }
}
function Fingerprint([string]$f) {
  $pub = (XPath $f) + ".pub"
  if (-not (Test-Path -LiteralPath $pub -PathType Leaf)) { return "" }
  $r = Run-Native "ssh-keygen" @("-lf", $pub)
  if ($r.Code -ne 0 -or -not $r.Out.Count) { return "" }
  return (("" + $r.Out[0]).Trim() -split '\s+')[1]
}
function Ck-Cmd([string]$hop = "") {
  $c = if ($PSCommandPath) { $PSCommandPath } else { ".\mace-gui-connect.ps1" }
  if ($ProfileName) { $c += " $ProfileName" }
  else {
    if ($GatewayV) { $c += " -Gateway $GatewayV" }
    if ($NodeV) { $c += " -Node $NodeV" }
    if ($JumpV) { $c += " -Jump $JumpV" }
  }
  if ($hop) { $c += " -Hop $hop" }
  return $c
}
function Node-Refused([string[]]$lines) {
  if (($lines -join "`n") -match "Host key verification failed") { return $false }
  $d = @($lines | Where-Object { $_ -match 'Permission denied \(' } | ForEach-Object { ($_ -split ':')[0] })
  if (-not $d.Count) { return $false }
  $nh = ($NodeV -split '@')[-1]; $gh = ($GatewayV -split '@')[-1]
  if ($gh -and @($d | Where-Object { $_.Contains($gh) }).Count) { return $false }
  return [bool](@($d | Where-Object { $_.Contains($nh) }).Count)
}
function Ck-Classify([string]$h, [string[]]$lines, $code, [string]$hop = "") {
  $lines = @($lines | Where-Object { "$_".Trim() })
  $u = Unreach $lines $(if ($hop) { $NodeV } else { "" })
  if ($u) { Ck FAIL "cannot reach host $($u[0]) ($($u[1]))" "check the name, and your network or VPN (ssh -v $($u[0]) true shows where it stops)"; return }
  if (($lines -join "`n") -match "Host key verification failed|IDENTIFICATION HAS CHANGED") {
    Ck FAIL "ssh does not know the host key of $h yet, or it changed" "ssh $h true (once, in this terminal, to check and accept the key)"; return
  }
  $last = if ($lines.Count) { $lines[-1] } else { "no message" }
  if ($hop) {
    if (Node-Refused $lines) { Ck FAIL "jump hop: $NodeV refused your key from this computer (a connection would ask for the password of $NodeV)" "use the nested hop, or give $NodeV your key (docs/gui.md#mace-gui-connect-the-node-asks-for-a-password)" }
    else { Ck FAIL "jump hop: ssh -J $GatewayV $NodeV true failed (exit $code): $last" }
    return
  }
  $den = @($lines | Where-Object { $_ -match 'Permission denied \(([^)]*)\)' } | Select-Object -First 1)
  if ($den.Count) {
    $null = $den[0] -match 'Permission denied \(([^)]*)\)'
    $methods = $Matches[1]
    if ((",$methods,") -match ',publickey,') { Ck FAIL "$h refused your keys (Permission denied: $methods)" "ssh-add your key, and check that its .pub is in ~/.ssh/authorized_keys on the cluster (docs/ssh-setup.md)" }
    else { Ck WARN "$h asks for a 2FA code or a password, which the check cannot type; the hop probes stop here" "skip the probes with -NoNet (Windows ssh has no connection sharing); connecting shows the prompt" }
    return
  }
  Ck FAIL "ssh $h true failed (exit $code): $last" "ssh -v $h true"
}

function Run-Check {
  $first = if ($GatewayV) { $GatewayV } else { $NodeV }
  $what = if ($NodeV -and $GatewayV) { "$GatewayV -> $NodeV" } else { $first }
  $pn = if ($ProfileName) { "${ProfileName}: " } else { "" }
  Write-Output "Checking the ssh setup for $pn$what (read-only; nothing is changed)"
  # 1. ssh
  $sshCmd = Get-Command ssh -ErrorAction SilentlyContinue
  if (-not $sshCmd) {
    Ck FAIL "no ssh command on PATH: connect needs the OpenSSH client" "Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0 (in an administrator PowerShell)"
    Write-Output "Verdict: fix the FAIL lines above, then check again."; $script:CkRc = 1; return
  }
  $sshPath = "" + $sshCmd.Source
  $ver = "" + @((Run-Native "ssh" @("-V")).Out)[0]
  Ck OK "ssh: $ver ($sshPath)"
  if ($OnWindows) {
    if ($ver -match 'OpenSSH_for_Windows_(\d+)\.(\d+)' -and ([int]$Matches[1] -lt 8 -or ([int]$Matches[1] -eq 8 -and [int]$Matches[2] -lt 1))) {
      Ck WARN "Windows OpenSSH $ver is older than 8.1, whose -J (the jump hop) is unreliable" "update the Windows OpenSSH client, or use the nested hop"
    }
    $sys = "C:\Windows\System32\OpenSSH"
    if (-not $sshPath.ToLower().StartsWith($sys.ToLower())) {
      Ck WARN "this ssh is $sshPath, not Windows' own OpenSSH ($sys\ssh.exe): it may use another agent and config" "put $sys first in PATH (Get-Command ssh shows which ssh runs)"
    }
  }
  # 2. the gateway (ssh -G)
  $g = Ssh-G $first
  if (-not $g.Count) {
    Ck FAIL "ssh -G $first printed nothing: ssh cannot read its settings for $first" "ssh -G $first (shows ssh's own error, often a typo in ~/.ssh/config)"
    Write-Output "Verdict: fix the FAIL lines above, then check again."; $script:CkRc = 1; return
  }
  $role = if ($GatewayV) { "gateway" } else { "node" }
  $u = @($g["user"])[0]; if (-not $u) { $u = "?" }
  $hn = @($g["hostname"])[0]; if (-not $hn) { $hn = $first }
  $port = @($g["port"])[0]; if (-not $port) { $port = "22" }
  Ck OK "$role ${first}: user $u, HostName $hn, port $port"
  $pj = @($g["proxyjump"])[0]
  if ($pj -and $pj -ne "none") { Ck INFO "$first is reached through ProxyJump $pj (from your ssh config)" }
  if ($GatewayV -and (Test-EmailLike $GatewayV)) {
    $pn2 = if ($ProfileName) { "$ProfileName " } else { "" }
    Ck WARN "the gateway $GatewayV looks like an email address. It should be user@login-host (your cluster's login node) or a Host alias from ~/.ssh/config" "mace gui connect $($pn2)--gateway <user@login-host> --save"
  }
  if ($GatewayV -and -not $NoNet) {
    $bad = Gw-Unresolvable
    if ($bad) { Ck WARN "$bad matches no Host in ~/.ssh/config and does not resolve (DNS)" "use your cluster's login host (e.g. login.example.edu), or add a Host block with its HostName to ~/.ssh/config" }
  }
  $gfiles = @(Existing-Files $g)
  if ($gfiles.Count) {
    $extra = if (@($g["identitiesonly"])[0] -eq "yes") { " (IdentitiesOnly yes)" } else { "" }
    Ck OK ("key files ssh offers ${first}: " + ($gfiles -join ", ") + $extra)
  }
  else { Ck WARN "none of the key files ssh offers $first exist; only an agent's keys can log in" "ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_cluster, then put its .pub in ~/.ssh/authorized_keys on the cluster (docs/ssh-setup.md)" }
  $cm = "" + @($g["controlmaster"])[0]
  $cmOn = $cm -and $cm -ne "false" -and $cm -ne "no"
  if ($cmOn -and $OnWindows) { Ck WARN "ControlMaster is set for $first, which Windows OpenSSH does not support" "remove the ControlMaster, ControlPath and ControlPersist lines for $first from ~/.ssh/config" }
  # the agent (ssh-add -l), read once
  $agentOk = $true
  if ($OnWindows) {
    $svc = Get-Service ssh-agent -ErrorAction SilentlyContinue
    if (-not $svc) { $agentOk = $false }
    elseif ("$($svc.StartType)" -eq "Disabled" -or "$($svc.Status)" -ne "Running") { $agentOk = $false }
  }
  $agentRun = Run-Native "ssh-add" @("-l")
  $agentFps = @($agentRun.Out | Where-Object { $_ -match '^\d+ \S+' } | ForEach-Object { ($_ -split '\s+')[1] })
  # 3. the node (ssh -G) against the gateway
  $nfiles = @()
  if ($NodeV -and $GatewayV) {
    if ($HopV -eq "nested") { Ck INFO "nested hop: the gateway logs in to $NodeV with the keys it has there; your keys here are not used for that hop" }
    else {
      $target = Node-Target
      $n = Ssh-G $target
      $nfiles = @(Existing-Files $n)
      $nu = @($n["user"])[0]; if (-not $nu) { $nu = "?" }
      $nf = if ($nfiles.Count) { $nfiles -join ", " } else { "none that exist" }
      Ck INFO "node ${NodeV}: user $nu, key files $nf"
      $keys = @(Node-Keys $target)
      if ($keys.Count) { Ck INFO ("for the jump hop, the connection also gives ${NodeV}: " + ($keys -join " ")) }
      if ($gfiles.Count) {
        $missing = @($gfiles | Where-Object { @($n["identityfile"]) -notcontains $_ })
        if ($missing.Count -lt $gfiles.Count) { Ck OK "the jump hop offers $NodeV your gateway key (from your ssh config)" }
        else {
          $held = ""
          if ($agentOk -and $agentRun.Code -eq 0) {
            foreach ($f in $missing) { $fp = Fingerprint $f; if ($fp -and $agentFps -contains $fp) { $held = $f; break } }
          }
          if ($held) { Ck OK "the jump hop offers $NodeV your gateway key (the agent holds $held)" }
          else { Ck WARN ("$NodeV gets none of " + ($missing -join ", ") + " from your ssh config, and the agent does not hold them, so a plain ssh -J $GatewayV $NodeV would ask for the password of $NodeV") "the connection adds them for you (above); for plain ssh too, add a Host block for the node (-FixConfig prints one) or run ssh-add $($missing[0])" }
        }
      }
    }
  }
  # 4. the agent: fingerprints only
  $key = if ($gfiles.Count) { $gfiles[0] } else { "~/.ssh/id_ed25519_cluster" }
  if ($OnWindows -and -not $svc) { Ck WARN "no OpenSSH Authentication Agent service (ssh-agent) on this computer" "add the OpenSSH Client in Settings, Optional features" }
  elseif ($OnWindows -and "$($svc.StartType)" -eq "Disabled") { Ck FAIL "the OpenSSH Authentication Agent service (ssh-agent) is disabled" "in an administrator PowerShell: Get-Service ssh-agent | Set-Service -StartupType Automatic; Start-Service ssh-agent" }
  elseif ($OnWindows -and "$($svc.Status)" -ne "Running") { Ck WARN "the OpenSSH Authentication Agent service (ssh-agent) is $($svc.Status) (startup type $($svc.StartType))" "Start-Service ssh-agent" }
  elseif ($agentRun.Code -eq 1) { Ck WARN "the ssh agent holds no keys" "ssh-add $key" }
  elseif ($agentRun.Code -ne 0) {
    $why = "" + @($agentRun.Out)[0]; if (-not $why) { $why = "no answer" }
    Ck WARN "cannot reach an ssh agent ($why)" "Start-Service ssh-agent, then ssh-add $key"
  }
  else {
    $shown = @($agentRun.Out | Where-Object { $_ -match '^\d+ \S+' } | ForEach-Object { $w = $_ -split '\s+'; if ($w[-1].StartsWith("(")) { "$($w[1]) $($w[-1])" } else { $w[1] } })
    Ck OK ("agent: $($shown.Count) key(s): " + ($shown -join ", "))
  }
  if ($agentOk -and ($agentRun.Code -eq 0 -or $agentRun.Code -eq 1)) {
    foreach ($f in $gfiles) {
      $fp = Fingerprint $f
      if (-not $fp) { Ck INFO "$f.pub is missing, so the check cannot compare $f with the agent" "ssh-keygen -y -f $f > $f.pub" }
      elseif ($agentFps -contains $fp) { Ck OK "$f ($fp) is in the agent" }
      else { Ck WARN "$f ($fp) is not in the agent: with a passphrase, ssh must ask for it on every hop, and a connection may have no terminal to ask on" "ssh-add $f" }
    }
  }
  # 5. permissions
  Ck INFO "if ssh says UNPROTECTED PRIVATE KEY FILE or Bad permissions, make each key yours alone" 'icacls <key file> /inheritance:r /grant:r "$($env:USERNAME):(R)"'
  # 6. live probes, each running only `true`
  $res = @{ gateway = $false; jump = $false; nested = $false }
  if ($NoNet) { Ck SKIP "live probes skipped: no ssh login was tried" }
  else {
    $J = @(); if ($JumpV) { $J = @("-J", $JumpV) }
    $PO = @("-o", "BatchMode=yes", "-o", "ConnectTimeout=10")
    # the profile's ssh options where the connection passes them: the gateway's ssh of the nested
    # hop (and without a hop), the node's ssh of the jump hop (-J's gateway leg gets none)
    $X = if ($NodeV -and $GatewayV -and $HopV -ne "nested") { @() } else { @($SshOptions) }
    $r = Run-Native "ssh" (@($J) + $PO + $X + @($first, "true"))
    if ($r.Code -eq 0) { Ck OK "ssh $first true: logged in without a prompt"; $res.gateway = $true }
    else { Ck-Classify $first $r.Out $r.Code }
    if ($res.gateway -and $NodeV -and $GatewayV) {
      $target = Node-Target
      $keys = @(Node-Keys $target)
      $jj = if ($JumpV) { "$JumpV,$GatewayV" } else { $GatewayV }
      $r = Run-Native "ssh" (@("-J", $jj) + $PO + $keys + @($SshOptions) + @($target, "true"))
      if ($r.Code -eq 0) { Ck OK "jump hop: ssh -J $GatewayV $NodeV true works"; $res.jump = $true }
      else { Ck-Classify $NodeV $r.Out $r.Code "jump"; $script:CkHopFails++ }
      $inner = "ssh -o BatchMode=yes -o ConnectTimeout=10 " + (ShQuote $NodeV) + " true"
      $r = Run-Native "ssh" (@($J) + $PO + @($SshOptions) + @($GatewayV, $inner))
      if ($r.Code -eq 0) { Ck OK "nested hop: the gateway logs in to $NodeV by itself"; $res.nested = $true }
      else {
        $lines = @($r.Out | Where-Object { "$_".Trim() })
        $last = if ($lines.Count) { $lines[-1] } else { "no message" }
        Ck FAIL "nested hop: the gateway could not log in to $NodeV (exit $($r.Code)): $last" "log in to the gateway and run ssh $NodeV true there: the cluster may need a key for logins between its own hosts"
        $script:CkHopFails++
      }
    }
  }
  # 7. the verdict
  $others = $script:CkFails - $script:CkHopFails
  if ($others -eq 0 -and ($res.jump -or $res.nested)) {
    if ($HopV -eq "nested" -and $res.nested) { Write-Output "Verdict: the nested hop works. Connect with: $(Ck-Cmd)" }
    elseif ($res.jump) {
      $hh = if ($HopV -eq "nested") { "jump" } else { "" }
      Write-Output "Verdict: the jump hop works. Connect with: $(Ck-Cmd $hh)"
    }
    else { Write-Output "Verdict: only the nested hop works. Connect with: $(Ck-Cmd 'nested')" }
    $script:CkRc = 0; return
  }
  if ($script:CkFails -gt 0) { Write-Output "Verdict: fix the FAIL lines above, then check again."; $script:CkRc = 1; return }
  if ((-not $NodeV -or -not $GatewayV) -and $res.gateway) { Write-Output "Verdict: MACE GUI runs on the gateway itself. Connect with: $(Ck-Cmd)"; $script:CkRc = 0; return }
  Write-Output "Verdict (without the probes): no problem found. Connect with: $(Ck-Cmd)"
  $script:CkRc = 0
}

function Suggest-Alias([string]$h) {
  $labels = @($h.ToLower().Split(".") | Where-Object { $_ })
  if (-not $labels.Count) { return "cluster" }
  if (@("login", "ssh", "gateway", "gw", "bastion") -contains $labels[0] -and $labels.Count -gt 2) { return $labels[1] }
  if ($labels[0] -match '^[0-9-]+$') { return "cluster" }
  return $labels[0]
}
function Fix-Config {
  $first = if ($GatewayV) { $GatewayV } else { $NodeV }
  $g = Ssh-G $first
  $user = @($g["user"])[0]
  $hn = @($g["hostname"])[0]; if (-not $hn) { $hn = ($first -split '@')[-1] }
  $alias = if (-not $first -or $first.Contains("@")) { Suggest-Alias $hn } else { $first }
  $keys = @(Existing-Files $g)
  $defaults = @("id_rsa", "id_ecdsa", "id_ecdsa_sk", "id_ed25519", "id_ed25519_sk", "id_xmss", "id_dsa")
  $custom = @($keys | Where-Object { $defaults -notcontains ($_ -split '[/\\]')[-1] })
  $key = if ($custom.Count) { $custom[0] } elseif ($keys.Count) { $keys[0] } else { "~/.ssh/id_ed25519_cluster" }
  Write-Output "# A suggested ~/.ssh/config block for this profile, from ssh -G (printed only: nothing was written). Review it, then add it to ~/.ssh/config."
  if (@(Config-Hosts) -contains $alias) { Write-Output "# You already have a Host $alias block: merge these lines into it." }
  if (-not $keys.Count) { Write-Output "# No key file exists yet: make one first, ssh-keygen -t ed25519 -f $key" }
  Write-Output "Host $alias"
  Write-Output "    HostName $hn"
  if ($user) { Write-Output "    User $user" }
  $pj = if ($JumpV) { $JumpV } else { @($g["proxyjump"])[0] }
  if ($pj -and $pj -ne "none") { Write-Output "    ProxyJump $pj" }
  Write-Output "    IdentityFile $key"
  Write-Output "    IdentitiesOnly yes"
  Write-Output "    AddKeysToAgent yes"
  if ($IsMacOS) { Write-Output "    UseKeychain yes" }
  if (-not $OnWindows) { Write-Output "    ControlMaster auto"; Write-Output "    ControlPath ~/.ssh/cm-%C"; Write-Output "    ControlPersist 30m" }
  Write-Output "    ServerAliveInterval 30"
  if ($NodeV -and $GatewayV) {
    $nn = ($NodeV -split '@')[-1]
    Write-Output ""
    Write-Output "Host $nn"
    if ($user) { Write-Output "    User $user" }
    Write-Output "    IdentityFile $key"
    Write-Output "    IdentitiesOnly yes"
    Write-Output "    ProxyJump $alias"
  }
  $c = if ($PSCommandPath) { $PSCommandPath } else { ".\mace-gui-connect.ps1" }
  if ($ProfileName) { $c += " $ProfileName" }
  $c += " -Gateway $alias"
  if ($NodeV) { $c += " -Node " + ($NodeV -split '@')[-1] }
  Write-Output "# Then connect with: $c"
}

if ($Check -or $FixConfig) {
  $rc = 0
  if ($Check) {
    $script:CkRc = 1
    Run-Check
    $rc = $script:CkRc
  }
  if ($FixConfig) {
    if ($Check) { Write-Output "" }
    Fix-Config
  }
  exit $rc
}

if ($DryRun) {
  # one argument per line, as mace-gui-connect.sh --dry-run prints them; then
  # the same command as one line to paste into PowerShell (5.1 or 7): --% hands
  # the rest to ssh.exe unchanged, quoted as this script passes it (not cmd.exe)
  $dry = Build-Argv $LocalPortV 40000 40001
  # the hop, as mace gui connect --dry-run labels it
  if ($NodeV -and $GatewayV) { Write-Output "# hop $HopV (the remote port is random on a real run)" }
  else { Write-Output "# no hop (the remote port is random on a real run)" }
  $dry | ForEach-Object { $_ }
  Write-Output "# PowerShell (paste as is; not for cmd.exe):"
  Write-Output ("ssh --% " + (@($dry | ForEach-Object { Win-Quote $_ }) -join " "))
  exit 0
}
# -UpdateRemote: `mace update` in the cluster's MACE checkout first, on the
# gateway (it shares your home with the nodes), interactive in this window
# (-t only with a console to answer on; else BatchMode and --yes). Exit 3
# from the cluster: its MACE has no update command (older than 1.1.6).
function Invoke-UpdateRemote([string]$after = "Connecting anyway.") {
  $find = Mace-Fallback
  $um = if ($find) { '"$M"' } else { (@($MaceV -split '\s+' | Where-Object { $_ }) | ForEach-Object { HomePath $_ }) -join " " }
  $interactive = -not [Console]::IsInputRedirected
  $uargs = "--channel $UpdateRemote" + $(if ($interactive) { "" } else { " --yes" })
  # [char]62 is ">": a stream redirection written out here would read as PowerShell's own
  $gt = [char]62
  $ucmd = "if $um update --help 2" + $gt + "/dev/null | grep -q -e --remote-only; then $um update $uargs; else exit 3; fi"
  if ($find) { $ucmd = "{ " + $find + "; } " + $gt + "/dev/null 2" + $gt + "&1; " + $ucmd }
  $uscript = 'cd -- "$HOME" && ' + $ucmd
  if ($CondaV) { $uscript = "conda activate " + (HomePath $CondaV) + "; " + $uscript }
  if ($SetupV) { $uscript = $SetupV.TrimEnd(";") + "; " + $uscript }
  $uhost = if ($GatewayV) { $GatewayV } else { $NodeV }
  Write-Output "Updating MACE on $uhost ($UpdateRemote) ..."
  $ua = @()
  if ($JumpV) { $ua += @("-J", $JumpV) }
  if ($interactive) { $ua += @("-t") } else { $ua += @("-o", "BatchMode=yes") }
  $ct = Ct-Setup $uhost
  $ua += @($ct.Opts) + @($SshOptions) + @($uhost, ("exec bash -lc " + (ShQuote $uscript)))
  # one command line, quoted for CreateProcess as for the connection below; the console stays this window's
  $upsi = New-Object System.Diagnostics.ProcessStartInfo
  $upsi.FileName = "ssh.exe"
  $upsi.Arguments = (($ua | ForEach-Object { Win-Quote $_ }) -join " ")
  $upsi.UseShellExecute = $false
  try { $uproc = [System.Diagnostics.Process]::Start($upsi) }
  catch { Write-Warning "ssh to $uhost could not start ($_), so mace update did not run. $after"; return }
  $uproc.WaitForExit()
  switch ($uproc.ExitCode) {
    0 { }
    1 { Write-Warning "the MACE on $uhost was not updated (you said no). $after" }
    2 { Write-Warning "mace update on $uhost stopped without updating (its reason is above). $after" }
    3 { Write-Warning "the MACE on $uhost is older than 1.1.6 and has no update command: update it once by hand (git -C <MACE folder> pull). $after" }
    255 { Write-Warning "ssh to $uhost failed (see above), so mace update did not run. $after" }
    default { Write-Warning "mace update did not finish on $uhost (exit $($uproc.ExitCode); see above). $after" }
  }
}

if ($Manual) {
  # -UpdateRemote first: the manual steps then start the updated MACE
  if ($UpdateRemote) { Invoke-UpdateRemote "The manual steps follow." }
  $u = if ($GatewayV -match "@") { $GatewayV } else { "$env:USERNAME@$GatewayV" }
  Write-Output "1. In one terminal, on the node (or through ssh -J $u $NodeV): mace gui --dir $DirV --port $LocalPortV"
  Write-Output "2. In another terminal here (leave it running): ssh -N -J $u -L ${LocalPortV}:127.0.0.1:${LocalPortV} $NodeV"
  Write-Output "3. Open the URL that mace gui printed (127.0.0.1:$LocalPortV/...)."
  exit 0
}

# ------------------------------------------------------------------ run
$sv = Invoke-Native "ssh" @("-V")
$ver = "$($sv.Err)$($sv.Out)"
if ($ver -match 'OpenSSH_for_Windows_(\d+)\.(\d+)' -and ([int]$Matches[1] -lt 8 -or ([int]$Matches[1] -eq 8 -and [int]$Matches[2] -lt 1)) -and ($JumpV -or ($NodeV -and $GatewayV -and $HopV -ne "nested"))) {
  Write-Warning "Your ssh is older than 8.1, whose -J is unreliable on Windows; update Windows OpenSSH or use -Hop nested."
}
function Port-Free([int]$p) {
  try { $l = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, $p); $l.Start(); $l.Stop(); return $true }
  catch { return $false }
}
$LP = $LocalPortV
while (-not (Port-Free $LP)) {
  $LP++
  if ($LP -gt $LocalPortV + 20) { Fail "No free local port near $LocalPortV" 1 }
}
if ($LP -ne $LocalPortV) { Write-Output "Port $LocalPortV is busy here; using $LP. Drafts saved under port $LocalPortV won't be offered on $LP." }
$rnd = New-Object System.Random
$RP = $rnd.Next(20000, 61000); $GP = $rnd.Next(20000, 61000)
$argv = Build-Argv $LP $RP $GP
$where = if ($NodeV) { $NodeV } else { $GatewayV }
$unres = Gw-Unresolvable
if ($unres) { Write-Warning "$unres matches no Host in ~/.ssh/config and does not resolve (DNS): check the gateway's name (your cluster's login host, e.g. login.example.edu). Connecting anyway." }
if ($UpdateRemote) { Invoke-UpdateRemote }
Write-Output "Connecting to $where... (answer ssh's questions in this window: a password, 2FA, or on a first connection the host-key yes/no question)"

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = "ssh.exe"
$psi.Arguments = (($argv | ForEach-Object { Win-Quote $_ }) -join " ")
$psi.UseShellExecute = $false
$psi.RedirectStandardInput = $true
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $false
$proc = [System.Diagnostics.Process]::Start($psi)
$state = "wait"
$timer = $null
try {
  while (-not $proc.StandardOutput.EndOfStream) {
    $line = $proc.StandardOutput.ReadLine()
    if ($line -like "MACE_GUI_READY *") {
      $j = $line.Substring(15) | ConvertFrom-Json
      $base = "http://127.0.0.1:$LP$($j.prefix)"
      for ($i = 0; $i -lt 60; $i++) {
        try { $r = Invoke-WebRequest -UseBasicParsing -Uri "$($base)api/info" -Headers @{ "X-Mace-Token" = $j.token } -TimeoutSec 3; if ($r.StatusCode -eq 200) { break } } catch { Start-Sleep -Milliseconds 300 }
      }
      $url = if ($j.login) { $j.login -replace '^http://127\.0\.0\.1:\d+', "http://127.0.0.1:$LP" } else { "$($base)?token=$($j.token)" }
      if ($j.version) { Write-Output "MACE on ${where}: $($j.version)" }
      if ($j.reattached) { Write-Output "Reattached to the server still running on $where." }
      if ($OpenV) { try { Start-Process $url } catch { Write-Warning "Could not open a browser; open the address below." } }
      Write-Output "MACE GUI on ${where}: $url"
      Write-Output "Press Ctrl+C to stop the server on $where."
      $timer = New-Object System.Timers.Timer 60000
      Register-ObjectEvent -InputObject $timer -EventName Elapsed -Action { try { $proc.StandardInput.WriteLine("ping") } catch {} } | Out-Null
      $timer.Start()
      $state = "ready"
    }
    elseif ($line -like "MACE_GUI_LOGIN *") {
      $j = $line.Substring(15) | ConvertFrom-Json
      $u = $j.url -replace '^http://127\.0\.0\.1:\d+', "http://127.0.0.1:$LP"
      Write-Output "New sign-in link: $u"; if ($OpenV) { try { Start-Process $u } catch {} }
    }
    elseif ($line -like "MACE_GUI_STOPPED *") {
      if ($line -match '"browser"') { Write-Output "The server on $where was stopped from the browser." }
      elseif ($line -match '"cpu-limit"') { Write-Output "The server reached the node's CPU-time limit and stopped cleanly. Run this script again to start a fresh one." }
      else { Write-Output "The server on $where stopped." }
      $state = "stopped"; break
    }
    elseif ($line -like "MACE_GUI_USING *") {
      $mp = $line.Substring(15)
      Write-Output "mace is not on ${where}'s PATH there; using $mp. To skip the search, pass -Mace $mp."
    }
    elseif ($line -like "MACE_GUI_ERROR *") { Write-Warning "The server reported: $($line.Substring(15))"; $state = "error"; break }
    elseif ($line) { Write-Output "remote: $line" }
  }
}
finally {
  if ($timer) { $timer.Stop() }
  if (-not $proc.HasExited) {
    try { $proc.StandardInput.WriteLine("detach"); $proc.StandardInput.Close() } catch {}
    if (-not $proc.WaitForExit(5000)) { try { $proc.Kill() } catch {} }
  }
}
if ($state -eq "stopped") { exit 0 }
if ($state -eq "ready") { Write-Output "The connection to $where closed. If GUI tasks were running, the server keeps them running: run this script again to reattach."; exit 1 }
# ssh's messages go straight to this window (so its prompts work); ssh's exit code 255 means ssh itself failed
$code = if ($proc.WaitForExit(5000)) { $proc.ExitCode } else { -1 }
if ($code -eq 127) {
  Write-Warning "The remote shell did not find mace: pass -Mace <full path> or a -Setup line."
  if (Mace-Fallback) { Write-Warning "It also looked for mace_cli in `$MACE_HOME, ~/mace and ~/MACE there." }
  if ($script:Moved.Count) { Write-Warning "$($script:Moved -join ', ') came from this computer's home folder (your shell expands an unquoted ~ here): check that it exists on $where. Quote the ~ ('~/...') so that your shell passes it as is." }
  exit 127
}
if ($code -eq 1) {
  Write-Warning "If ssh printed ""cd: ...: No such file or directory"" above, the folder $(if ($DirV) { $DirV } else { '~' }) does not exist on ${where}: check -Dir (a path on the node; ~/... is the node's home folder)."
}
if ($code -eq 255 -and $NodeV -and $GatewayV -and $HopV -eq "jump") {
  Write-Warning "With the jump hop, $NodeV checks your key from this computer. -Hop nested logs in from the gateway instead. If ssh said Permission denied for $NodeV, or asked for its password, see docs/gui.md#mace-gui-connect-the-node-asks-for-a-password"
  if ($HopAuto) { Write-Warning "(Hop auto works as jump in this script: it does not retry with -Hop nested by itself, as mace gui connect does.)" }
}
if ($code -eq 255 -and $script:TrustCmd) {
  # never trusted silently: the user checks the fingerprint and answers once
  # ssh's messages go straight to the console here, so the host that failed is not known: with -J both steps
  if ($script:GatewayTrustCmd) {
    Write-Warning "If ssh said ""Host key verification failed"": this computer has not connected to that host before. With the jump hop it checks both keys itself, the gateway's first, then the node's. Check them, then trust them once: run  $($script:GatewayTrustCmd)  first, then  $($script:TrustCmd)  in this window, compare each fingerprint with your cluster's, answer yes, and run this script again. Or use -Hop nested: the gateway then checks the node's key with its own list of known hosts."
  } else {
    Write-Warning "If ssh said ""Host key verification failed"": this computer has not connected to that host before. Check the key, then trust it once: run  $($script:TrustCmd)  in this window, compare the fingerprint with your cluster's, answer yes, and run this script again."
  }
}
if ($code -eq 255) {
  Write-Warning "If ssh said it cannot connect, timed out or could not resolve a host name: check that name, and your network or VPN. -Check (with the same profile or options) tests each step of the connection."
  Write-Warning ('If ssh said "Permission denied (publickey)" above, or could not read a passphrase: ssh could not log in with your key. If the key has a passphrase, ssh cannot ask for it here: load it into an agent first, with eval "$(ssh-agent)" and ssh-add (on Windows: start the OpenSSH Authentication Agent service, then run ssh-add), then run this script again. A Host alias from ~/.ssh/config also works as the gateway.')
}
Write-Warning "The connection ended before MACE GUI was ready. If no password or 2FA prompt appeared, set up an ssh key or agent (docs/gui.md#windows-powershell), or use -Manual."
exit 1
