# FREUID Docker Blocker Resolution - 2026-07-10

## Current Symptom

Docker Desktop is installed and the `desktop-linux` context exists, but the Linux engine does not become usable.

Observed checks:

```powershell
docker info --format '{{.ServerVersion}}'
```

Result: timeout or HTTP 500 from `dockerDesktopLinuxEngine`.

```powershell
wsl.exe -l -v
```

Result: `docker-desktop` is present but stopped.

```powershell
wsl.exe -d docker-desktop -- echo docker-desktop-wsl-ok
```

Result: WSL2 cannot start because virtualization / Virtual Machine Platform is unavailable. The diagnostic included:

```text
HCS_E_HYPERV_NOT_INSTALLED
```

## Required Fix

This needs a machine-level Windows/firmware fix, not a repo change.

1. Enable CPU virtualization in firmware/BIOS if it is disabled.
2. Enable Windows Virtual Machine Platform from an elevated PowerShell:

```powershell
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

3. Ensure the Windows hypervisor launches:

```powershell
bcdedit /set hypervisorlaunchtype auto
```

4. Reboot Windows.
5. After reboot, update/start WSL and Docker:

```powershell
wsl.exe --update
wsl.exe --status
wsl.exe -l -v
```

6. Start Docker Desktop and verify:

```powershell
docker info --format '{{.ServerVersion}}'
```

## FREUID Smoke Test After Fix

Once `docker info` returns a server version, run:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test_freuid_docker.py
```

Expected behavior:

- Stages five local public-test images.
- Builds `freuid-frozen-stack:local`.
- Runs the container with `--network none`.
- Verifies `/submissions/submission.csv` has `id,label`, the expected IDs, and finite scores in `[0, 1]`.

## Current Workaround

The frozen runtime artifacts, short report PDF, and final-package draft zip are already published on:

```text
https://github.com/DrStrangel0ve/ai-image-forensics-comparison/releases/tag/freuid-freeze-2026-07-10
```

The only unverified package item is local no-network Docker execution.
