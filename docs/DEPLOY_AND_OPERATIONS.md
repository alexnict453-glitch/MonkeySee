# Deployment & Operations Runbook

## Target Boards

| Board | Host / SSH | SoC / OS / Kernel | Locked Clocks (`monkeysee-performance.service`) |
|---|---|---|---|
| **Orange Pi 5** (`funkyvision4`) | `10.8.46.204` (`orangepi`) | RK3588S, Ubuntu 22.04.5, `5.10.160-rockchip-rk3588`, `gcc-12` (`12.3.0`) | A55 `1800 MHz`, A76 `2400 MHz`, DMC `2112 MHz`, NPU `1000 MHz` *(pending Phase 12 libs/Python sync)* |
| **Orange Pi 5 Plus** (`orangepi5-plus`) | `fe80::c274:2bff:fef0:93e1%22` / dynamic IPv6 discovery via `opi_ssh.py` (`orangepi`) | RK3588, Armbian Bookworm, `6.1.115-vendor-rk35xx`, `gcc` `12.2.0` | A55 `1800 MHz`, A76 `2304/2352 MHz`, DMC `2400 MHz`, NPU `1000 MHz` |

> [!NOTE]
> **Sync status (`2026-11-26`):** Phase 12 native libraries (`libnapa_v061_a76_neon2_pgo.so`, `libv4l2_mpp.so`, `libnative_vision.so`) and Python native-pipeline updates are verified on **Orange Pi 5 Plus**; **Orange Pi 5** (`10.8.46.204`) has not yet been synced with the Phase 12 libraries or Python changes.

## Mandatory Hardware & OS Prerequisites

1. **USB 2.0 Root-Hub Isolation (`lsusb -t`):** Each `1280×800 @ 120 FPS` UVC MJPG camera reserves periodic `altsetting 6` bandwidth regardless of `uvcvideo quirks`. Connect `ATCam1`, `ATCam2`, and `ATCam4` to **separate `480M` host controllers** (e.g., on OPi5: `Bus 07`, `Bus 02`, `Bus 01`; on OPi5+: `Bus 05`, `Bus 03` white USB 2.0 port, `Bus 01`). Always resolve devices via `/dev/v4l/by-id/*` — never hardcoded `/dev/video*` numbers.
2. **Governors & VPU Permissions:** `monkeysee-performance.service` locks CPU, `dmc`, and `fdab0000.npu` governors to `performance`. `/etc/udev/rules.d/99-rockchip-vpu.rules` grants `video` group `0660` access to `/dev/mpp_service`, `/dev/rga`, and `/dev/dma_heap/*`.

## Safe Deployment Procedure

> [!CAUTION]
> **Never overwrite remote `/home/orangepi/MonkeySee/cal.json` or `/home/orangepi/MonkeySee/config.json`.** Every board stores camera-specific intrinsic calibration and tuned exposure/gain values (`Camera1`..`Camera4`).

1. **Back up and upload core pipeline files** (excludes `./venv` and `./__pycache__`):
   ```powershell
   python .\.agents\skills\monkeysee-rk3588-deploy\scripts\deploy_monkeysee.py `
     --host 10.8.46.204 --user orangepi --password <pass> `
     --backup-dir /home/orangepi/MonkeySee_backup_before_optimal
   ```
2. **Upload calibration & NT4 modules if modified** (`deploy_monkeysee.py` covers 8 core files; upload these 4 with `--lf` when deploying `optimal`):
   - `pipeline/ntables.py`
   - `calibration/calibrate_offline.py`
   - `calibration/charuco_calibrator.py`
   - `calibration/test_charuco_diagnostics.py`
3. **Validate remotely before starting `boot.sh`:**
   - Run `/home/orangepi/MonkeySee/venv/bin/python -m py_compile ...` on all deployed files.
   - Run `PYTHONPATH=/home/orangepi/MonkeySee /home/orangepi/MonkeySee/venv/bin/python -W error::RuntimeWarning /home/orangepi/MonkeySee/calibration/test_charuco_diagnostics.py --require-full`.
4. **Restart `boot.sh` detached & wait 20 s for camera warmup:**
   ```powershell
   python .\.agents\skills\opi-ssh-ops\scripts\opi_ssh.py --host 10.8.46.204 --user orangepi --password <pass> `
     start-daemon --sudo --pkill "boot\.sh|main\.py" --truncate-log `
     --cmd "/home/orangepi/MonkeySee/boot.sh" --log /home/orangepi/MonkeySee/crash.log
   ```
5. **Verify live operation:**
   ```powershell
   python .\.agents\skills\monkeysee-rk3588-deploy\scripts\verify_monkeysee.py `
     --host 10.8.46.204 --user orangepi --password <pass> --sample-gap 5 --cams 1,2,4
   ```
   Pass criteria: stable `boot.sh` + `python3.11 main.py` PIDs across T0/T1, `tracebacks=0`, `runtime_warnings=0`, `libv4l2_mpp.so` + `libnapa_v061_a76_neon2_pgo.so` loaded, all 3 cameras opened `1280x800 @ 120.0fps fourcc=MJPG buf=2`, `/api/cameras` ~`119–121 FPS`, valid `ffd8...ffd9` JPEGs on `/video_feed/{1,2,4}`, `/api/calibration/health` HTTP 200 on `:5801`, and thermals `< 65 °C`.

## Troubleshooting

| Symptom | Root Cause | Fix |
|---|---|---|
| Camera fails with `ENOSPC` (`rc=-3`) | Two `1280×800 @ 120 FPS` cameras share one `480M` USB hub | Move one camera to an unused `480M` USB controller; verify with `lsusb -t` |
| Camera caps at `~68 FPS` | `CAP_PROP_BUFFERSIZE=1` starves V4L2 ring requeue | Keep `DEFAULT_BUFFERSIZE = 2` in `camera/camerareader.py` |
| All 3 cams drop to `~90–108 FPS` (`sh16`) | Old `.so` missing `sh16` C/NEON/PGO optimizations or `dmc` governor on `ondemand` | Rebuild `/usr/local/lib/libnapa_v061_a76_neon2_pgo.so` per [NATIVE_LIBRARIES.md](NATIVE_LIBRARIES.md); confirm `dmc=performance` |
| Camera opens at `800×600` | Missing `Camera<id>` section in `config.json` | Ensure explicit `Camera1`/`Camera2`/`Camera4` sections exist in `config.json` |
| `RuntimeWarning` in logs | `np.ctypeslib.as_array` on packed C struct | Keep explicit `_NAPA_DET_DTYPE` + `np.frombuffer` in `localization/detection.py` |
| Branch push returns HTTP `403` | `origin` points to read-only upstream | Push via `safe_push.py` (uses user fork on feature branch; never push `master`/`main`) |
