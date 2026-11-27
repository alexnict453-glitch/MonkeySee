# MonkeySee

FRC Team 846 **2026 REBUILT** multi-camera vision system for Orange Pi 5 (`RK3588S`) and Orange Pi 5 Plus (`RK3588`). Runs up to 3 simultaneous `1280×800 @ 120 FPS` MJPG AprilTag cameras (`ATCam1`, `ATCam2`, `ATCam4`; `tag36h11`, valid IDs `1..32`, `TAG_H_in=6.5`) with Rockchip MPP hardware MJPEG decode (`libv4l2_mpp.so`), native AArch64 NEON NAPA detection (`libnapa_v061_a76_neon2_pgo.so`, `shadow_min_tag_px=16`; `498.67 agg fps` 3-cam concurrent MPP+NAPA `sh16 t=2` on OPi5+), NetworkTables 4 telemetry, ChArUco optical self-diagnostics, and a live Dash/Flask dashboard.

## Documentation Map

| Guide | Read When You Need To... |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | Trace the V4L2→MPP→NAPA→NT4 pipeline, module ownership, CPU pinning, and HTTP/NT4 endpoints |
| [Deploy & Operations](docs/DEPLOY_AND_OPERATIONS.md) | Deploy code safely without overwriting `cal.json`/`config.json`, verify live FPS, tunnel dashboards, or debug USB/FPS drops |
| [Native Libraries](docs/NATIVE_LIBRARIES.md) | Rebuild `libmpp_mjpeg.so`, `libv4l2_mpp.so`, or `libnapa_v061_a76_neon2_pgo.so`, check `sh16` C/NEON optimizations, or avoid known dead ends |
| [Calibration & Quality](docs/CALIBRATION_AND_QUALITY.md) | Audit `master`/`origin/beta` detection-quality invariants or use ChArUco calibration and optical health diagnostics |
| [Knowledge Graph Report](../graphify-out/GRAPH_REPORT.md) | Inspect Graphify communities, god nodes, and cross-module links (interactive chart: [graph.html](../graphify-out/graph.html)) |

## 30-Second Operator Cheatsheet

Run from the workspace root (`c:\Users\anicf\agy\MonkeySee`, using PowerShell 5.1 with `;` and `$LASTEXITCODE` checks):

```powershell
# 1. Run camera-less ChArUco + diagnostics test suite locally
python .\MonkeySee\calibration\test_charuco_diagnostics.py --require-full

# 2. Deploy core Python files to Orange Pi 5 (preserves cal.json & config.json)
python .\.agents\skills\monkeysee-rk3588-deploy\scripts\deploy_monkeysee.py --host 10.8.46.204 --user orangepi --password <pass> --restart

# 3. Verify live processes, native libs, /api/cameras, /video_feed/{1,2,4}, and thermals
python .\.agents\skills\monkeysee-rk3588-deploy\scripts\verify_monkeysee.py --host 10.8.46.204 --user orangepi --password <pass> --sample-gap 5 --cams 1,2,4

# 4. Forward MonkeyVision hub (5800) + per-camera dashboards (5801..5804)
python .\.agents\skills\opi-ssh-ops\scripts\opi_tunnel.py --host 10.8.46.204 --user orangepi --password <pass> --ports 5800,5801,5802,5803,5804
```
