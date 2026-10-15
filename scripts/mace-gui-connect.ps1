<#
.SYNOPSIS
  Start MACE GUI on a cluster node and open it in this computer's browser, with one ssh session.

.DESCRIPTION
  The Windows (PowerShell 5.1+) twin of `mace gui connect`: same options, same profiles file
  (%APPDATA%\mace\connect.toml), same protocol. Needs the ssh built into Windows 10 1809+ (-J needs
  OpenSSH 8.1+; check with ssh -V). Ctrl+C stops the server on the node; GUI tasks that are still
  running keep it alive until they finish, and running this script again reattaches.
  -Manual prints the two-terminal manual commands instead of connecting.

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
  [switch]$Manual
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
if ($GatewayV -and (Test-EmailLike $GatewayV)) {
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
function Remote-Command([int]$rp) {
  $parts = @()
  if ($SetupV) { $parts += $SetupV.TrimEnd(";") }
  if ($CondaV) { $parts += "conda activate " + (HomePath $CondaV) }
  $cmd = @("exec")
  if (-not $NodeV) { $cmd += @("env", "MACE_GUI_MAX_WORKERS=2") }
  foreach ($w in ($MaceV -split '\s+' | Where-Object { $_ })) { $cmd += (HomePath $w) }
  $cmd += @("--no-banner", "gui", "--managed", "--port", "$rp", "--dir", ".")
  foreach ($r in $Roots) { $cmd += @("--root", (HomePath $r)) }
  $d = if ($DirV) { $DirV } else { "~" }
  $parts += "cd -- " + (HomePath $d) + " && " + ($cmd -join " ")
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
function Build-Argv([int]$lp, [int]$rp, [int]$gp) {
  $a = New-Object System.Collections.ArrayList
  $opts = @("-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4")
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
    $inner = "ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -L 127.0.0.1:${gp}:127.0.0.1:${rp} " + (ShQuote $NodeV) + " " + (ShQuote $rcmd)
    if ($JumpV) { [void]$a.AddRange(@("-J", $JumpV)) }
    [void]$a.AddRange($opts); if ($SshOptions.Count) { [void]$a.AddRange($SshOptions) }
    [void]$a.AddRange(@("-L", "127.0.0.1:${lp}:127.0.0.1:${gp}", $GatewayV, $inner))
    $script:TrustCmd = "ssh " + $(if ($JumpV) { "-J $JumpV " } else { "" }) + "$GatewayV true"
    return ,$a
  }
  $j = if ($JumpV) { "$JumpV,$GatewayV" } else { $GatewayV }
  $target = $NodeV
  if ($NodeV -notmatch "@") {
    $u = ""
    if ($GatewayV -match "^([^@]+)@") { $u = $Matches[1] }
    else { $u = "$(((Invoke-Native "ssh" @("-G", $GatewayV)).Out -split "`r?`n" | Where-Object { $_ -match "^user " } | Select-Object -First 1))" -replace "^user ", "" }
    if ($u -and $u -ne $env:USERNAME) { $target = "$u@$NodeV" }
  }
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
if ($Manual) {
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
  Write-Warning ('If ssh said "Permission denied (publickey)" above, or could not read a passphrase: ssh could not log in with your key. If the key has a passphrase, ssh cannot ask for it here: load it into an agent first, with eval "$(ssh-agent)" and ssh-add (on Windows: start the OpenSSH Authentication Agent service, then run ssh-add), then run this script again. A Host alias from ~/.ssh/config also works as the gateway.')
}
Write-Warning "The connection ended before MACE GUI was ready. If no password or 2FA prompt appeared, set up an ssh key or agent (docs/gui.md#windows-powershell), or use -Manual."
exit 1
