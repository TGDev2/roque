# Testing From a Windows Host While Keeping Ubuntu VM Dependencies

This workflow keeps the full Python/OpenCV stack inside the Ubuntu VM and uses
only built-in PowerShell/.NET APIs on the Windows host.

The intended use case is a real chess.com game with the board theme fixed to
`green`, which is the default theme used by the remote runner.

## 1. Start the receiver in the Ubuntu VM

From the project root:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/remote_live.py --host 0.0.0.0 --port 8765
```

Useful options:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/remote_live.py \
  --host 0.0.0.0 \
  --port 8765 \
  --theme green \
  --active-color w
```

Notes:

- `green` is already the default theme.
- Use `--black-perspective` if the board is flipped.
- Use `--allow-uncertain` only if you want a FEN even when recognition confidence is low.

## 2. Make the VM reachable from Windows

The Windows host must be able to reach the VM on port `8765`.

Common working setups:

- bridged network
- host-only network
- explicit port-forwarding from the hypervisor to the VM

To find the VM IP from Ubuntu:

```bash
hostname -I
```

You can verify the server from Windows:

```powershell
Invoke-RestMethod http://<vm-ip>:8765/health
```

## 3. Run the capture script on Windows

The repository contains a PowerShell script at
`scripts/windows_capture_to_vm.ps1`.

Example using the full virtual desktop:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows_capture_to_vm.ps1 `
  -Uri http://<vm-ip>:8765/frame `
  -IntervalMs 750
```

Example using a cropped capture region:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows_capture_to_vm.ps1 `
  -Uri http://<vm-ip>:8765/frame `
  -Left 300 -Top 120 -Width 1400 -Height 1000 `
  -IntervalMs 500
```

Example selecting the capture region with the mouse:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows_capture_to_vm.ps1 `
  -Uri http://<vm-ip>:8765/frame `
  -SelectRegion `
  -IntervalMs 500
```

Notes:

- With no crop, the script captures the full virtual desktop.
- With `-PrimaryScreenOnly`, it captures only the main screen.
- With `-SelectRegion`, the script opens a one-time overlay: drag the mouse over the board area, release to confirm, `Esc` to cancel.
- `-PrimaryScreenOnly` can be combined with `-SelectRegion` to limit the overlay to the main screen only.
- If `-SelectRegion` is used, explicit `-Left/-Top/-Width/-Height` values are ignored.
- The Python side still runs board detection, so the crop does not need to be exact.
- A smaller crop reduces transfer size and latency.

## 4. Expected behaviour

For each uploaded frame, the VM returns JSON with:

- detection status
- selected theme
- uncertain squares, if any
- generated FEN when the position is valid

The VM logs only when the position changes or when the pipeline status changes.
The Windows script prints the latest status and the FEN when a new position is
detected.

## 5. Recommended setup for chess.com

- Keep the board theme on chess.com set to `green`.
- Keep the board large and fully visible.
- Avoid overlays or browser zoom changes while calibrating.
- Start with `-SelectRegion` or a cropped region around the browser window if full-screen capture is noisy.

## 6. Quick troubleshooting

- `board_not_found`: reduce the capture area around the browser window.
- `position_uncertain`: keep the `green` theme, increase board size, or wait for piece animation to finish.
- Windows cannot reach `/health`: adjust VM networking first; the processing stack is not the blocker.
