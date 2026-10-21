<#
.SYNOPSIS
  Prepare this Windows machine to serve Retinal Screening on the hospital network.
  Run ONCE, in PowerShell opened "as Administrator", after deploy\setup.py:

      powershell -ExecutionPolicy Bypass -File deploy\install_windows.ps1
      powershell -ExecutionPolicy Bypass -File deploy\install_windows.ps1 -Uninstall

  It changes system settings, so read what it does first:
    1. Firewall: allows inbound TCP on the service port ONLY from the staff
       networks in deploy\server.env (DR_ALLOWED_NETWORKS), on the Domain and
       Private profiles. Nothing is opened on Public networks.
    2. Start on boot: a scheduled task runs deploy\start_server.bat at startup
       as SYSTEM, restarting it if it stops.
    3. Power (plugged in only): never sleep or hibernate, and closing the lid
       does nothing -- a laptop that sleeps takes the service down for every
       ward. Battery settings are not touched.
#>
param([switch]$Uninstall)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$RuleName = 'Retinal Screening (staff network)'
$TaskName = 'Retinal Screening server'

$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this from PowerShell opened as Administrator.'
}

if ($Uninstall) {
    Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host 'Removed the firewall rule and the start-on-boot task. Power settings were left as they are.'
    return
}

$envFile = Join-Path $Root 'deploy\server.env'
if (-not (Test-Path $envFile)) { throw "deploy\server.env not found. Run deploy\setup.py first." }
$cfg = @{}
foreach ($line in Get-Content $envFile) {
    $l = ($line -split '#', 2)[0].Trim()
    if ($l -match '^([^=]+)=(.*)$') { $cfg[$Matches[1].Trim()] = $Matches[2].Trim() }
}
$port = [int]$cfg['DR_PORT']
$networks = @($cfg['DR_ALLOWED_NETWORKS'] -split '[,;]' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($networks.Count -eq 0) {
    throw 'DR_ALLOWED_NETWORKS is empty in deploy\server.env. Set the staff network range first.'
}

# 1. Firewall
Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule
New-NetFirewallRule -DisplayName $RuleName -Direction Inbound -Protocol TCP -LocalPort $port `
    -RemoteAddress $networks -Action Allow -Profile Domain,Private | Out-Null
Write-Host "Firewall: TCP $port allowed from $($networks -join ', ') (Domain/Private profiles only)."

# 2. Start on boot
$bat = Join-Path $Root 'deploy\start_server.bat'
$action = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument "/c `"$bat`"" -WorkingDirectory $Root
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero)
$taskUser = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Principal $taskUser -Force | Out-Null
Write-Host "Start on boot: scheduled task '$TaskName' registered."

# 3. Power, on mains only
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0
powercfg /setactive SCHEME_CURRENT
Write-Host 'Power: on mains the laptop no longer sleeps, hibernates or reacts to the lid.'

Write-Host ''
Write-Host "Done. Start it now with:  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host 'Also recommended: BitLocker on this drive (the encryption key file lives on it),'
Write-Host 'and a DHCP reservation so the server keeps the same IP address.'
