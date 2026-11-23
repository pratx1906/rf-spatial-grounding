# Keep the laptop out of Modern Standby while paper GPU jobs run (2026-10-29: a screen-off at 20:45 put the
# machine to sleep for 1 h 50 min and hung the RL update). Holds ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED until
# the deadline or until runs/paper/keep_awake.stop exists. Start hidden:
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile','-File','scripts\keep_awake.ps1'
param([string]$Until = "2026-11-01 00:00")
Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$stop = Join-Path $PSScriptRoot "..\runs\paper\keep_awake.stop"
$end = [datetime]::Parse($Until)
$ES = [uint32]"0x80000003"   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
while ((Get-Date) -lt $end -and -not (Test-Path $stop)) {
    [void][W.P]::SetThreadExecutionState($ES)
    Start-Sleep -Seconds 60
}
[void][W.P]::SetThreadExecutionState([uint32]"0x80000000")
