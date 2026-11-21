"""Camera-less synthetic ChArUco calibration & optical self-diagnostics test suite."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import localization.undistort as undistort
from calibration.charuco_calibrator import CharucoCalibrator


def _euler_to_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]], dtype=np.float64)
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float64)
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def render_charuco_view(
    board_tex: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    img_size=(1280, 800),
    tx: float = 0.0,
    ty: float = 0.0,
    tz: float = 0.42,
    rx_deg: float = 0.0,
    ry_deg: float = 0.0,
    rz_deg: float = 0.0,
    board_phys_w: float = 0.27,
) -> np.ndarray:
    """Project a planar ChArUco texture into a pinhole+distortion camera view."""
    w, h = img_size
    bh, bw = board_tex.shape[:2]
    board_phys_h = board_phys_w * (bh / float(bw))

    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=-1).reshape(-1, 1, 2)
    undist = cv2.undistortPoints(pts, K, dist)
    rays = np.concatenate([undist.reshape(-1, 2), np.ones((w * h, 1), dtype=np.float32)], axis=1)

    R = _euler_to_matrix(rx_deg, ry_deg, rz_deg)
    t = np.array([tx, ty, tz], dtype=np.float64)
    normal = R[:, 2]
    denom = rays @ normal
    valid_s = np.where(np.abs(denom) > 1e-6, (normal @ t) / denom, -1.0)
    P_cam = rays * valid_s[:, None]
    local = (P_cam - t[None, :]) @ R[:, :2]

    u_b = (local[:, 0] / board_phys_w + 0.5) * (bw - 1)
    v_b = (local[:, 1] / board_phys_h + 0.5) * (bh - 1)
    invalid = (valid_s <= 0.05) | (u_b < 0) | (u_b >= bw) | (v_b < 0) | (v_b >= bh)
    map_x = np.where(invalid, -1.0, u_b).reshape(h, w).astype(np.float32)
    map_y = np.where(invalid, -1.0, v_b).reshape(h, w).astype(np.float32)

    img = cv2.remap(
        board_tex,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(115, 115, 115),
    )
    return img


def make_board_texture(sx=11, sy=8, dict_id=cv2.aruco.DICT_4X4_50, legacy=False, tex_w=1600):
    cal = CharucoCalibrator(1)
    cal.squares_x = sx
    cal.squares_y = sy
    cal.legacy_pattern = legacy
    cal.aruco_dict_id = dict_id
    cal._build_board_and_detectors()
    tex_h = int(round(tex_w * sy / float(sx)))
    return cal.generate_board_image(width=tex_w, height=tex_h, margin_px=40)


def generate_calibration_views(K_true, dist_true, tex, count=24):
    # Cover all 3x3 cells with varied tilts/distances
    poses = [
        (-0.18, -0.11, 0.50, 12, -14, 3),
        (0.00, -0.11, 0.48, 15, 3, -4),
        (0.18, -0.11, 0.50, 12, 16, 5),
        (-0.18, 0.00, 0.48, -4, -18, 8),
        (0.00, 0.00, 0.42, 2, 2, -2),
        (0.18, 0.00, 0.48, 5, 18, -7),
        (-0.18, 0.11, 0.50, -15, -14, -6),
        (0.00, 0.11, 0.48, -18, 2, 6),
        (0.18, 0.11, 0.50, -14, 15, 9),
        (-0.16, -0.10, 0.46, 22, -10, 14),
        (0.16, -0.10, 0.46, 20, 12, -12),
        (-0.16, 0.10, 0.46, -22, -10, -14),
        (0.16, 0.10, 0.46, -20, 12, 12),
        (-0.17, -0.09, 0.54, 10, -22, -5),
        (0.00, -0.10, 0.53, 24, 0, 4),
        (0.17, -0.09, 0.54, 10, 22, -5),
        (-0.17, 0.01, 0.52, -8, -24, 10),
        (0.00, 0.00, 0.50, -10, 10, 22),
        (0.17, 0.01, 0.52, 8, 24, -10),
        (-0.17, 0.10, 0.54, -18, -20, 4),
        (0.00, 0.10, 0.53, -24, -4, -8),
        (0.17, 0.10, 0.54, -18, 20, -4),
        (-0.06, 0.00, 0.39, 12, -16, 18),
        (0.06, 0.00, 0.39, -12, 16, -18),
    ]
    return [render_charuco_view(tex, K_true, dist_true, tx=tx, ty=ty, tz=tz, rx_deg=rx, ry_deg=ry, rz_deg=rz) for (tx, ty, tz, rx, ry, rz) in poses[:count]]


def issue_codes(report):
    return {i["code"] for i in report.get("issues", [])}


def run_all_tests(require_full: bool = False):
    os.chdir(REPO_ROOT)
    K_true = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 400.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_true = np.array([-0.18, 0.05, 0.0, 0.0, 0.0], dtype=np.float64)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        old_cal_path = undistort.CAL_PATH
        undistort.CAL_PATH = str(tmp / "cal.json")
        undistort._calibrations_loaded = False

        try:
            tex_default = make_board_texture(11, 8, cv2.aruco.DICT_4X4_50, legacy=False)
            views = generate_calibration_views(K_true, dist_true, tex_default, count=24)

            # 1. Clean sharp 24-view ChArUco calibration
            cal1 = CharucoCalibrator(1)
            for idx, v in enumerate(views):
                cc, ci, mc, mi, gray = cal1.detect_board(v)
                health = cal1.analyze_frame_health(gray, cc, ci, mc, mi)
                if idx == 0:
                    assert health["status"] == "OK", f"Clean sharp frame flagged: {health}"
                assert cal1.add_sample(gray, cc, ci, health=health), f"View {idx} rejected: {health}"

            res = cal1.calibrate()
            assert res is not None, f"Calibration failed: {cal1.status_message}"
            meta = res["meta"]
            k = meta["camera_matrix"]
            fx_err = abs(k[0][0] - 900.0) / 900.0
            fy_err = abs(k[1][1] - 900.0) / 900.0
            cx_err = abs(k[0][2] - 640.0)
            cy_err = abs(k[1][2] - 400.0)
            assert fx_err < 0.025 and fy_err < 0.025, f"Focal recovery error too high: fx={k[0][0]}, fy={k[1][1]}"
            assert cx_err < 20.0 and cy_err < 20.0, f"Center recovery error too high: cx={k[0][2]}, cy={k[1][2]}"
            assert meta["reprojection_error"] < 0.5, f"Reproj error too high: {meta['reprojection_error']}"
            diag1 = meta["diagnostics"]
            assert diag1["status"] == "OK" and not diag1["issues"], f"Expected clean diagnostics, got: {diag1}"
            assert cal1.save_calibration()
            print(f"[PASS 1] Clean ChArUco calibration: fx={k[0][0]:.1f} fy={k[1][1]:.1f} err={meta['reprojection_error']:.4f}px health={diag1['status']}")

            # 2. calib.io legacyPattern=True board auto-detection
            tex_legacy = make_board_texture(11, 8, cv2.aruco.DICT_4X4_50, legacy=True)
            cal2 = CharucoCalibrator(2)
            legacy_views = generate_calibration_views(K_true, dist_true, tex_legacy, count=18)
            for v in legacy_views:
                cc, ci, mc, mi, gray = cal2.detect_board(v)
                assert cal2.add_sample(gray, cc, ci), "legacyPattern view failed"
            assert cal2.legacy_pattern is True, "Failed to auto-detect legacyPattern=True"
            res2 = cal2.calibrate()
            assert res2 is not None and res2["meta"]["reprojection_error"] < 0.6
            print(f"[PASS 2] Auto-detected legacyPattern=True: err={res2['meta']['reprojection_error']:.4f}px")

            # 3. Rotated (8, 11) layout auto-detection
            tex_8x11 = make_board_texture(8, 11, cv2.aruco.DICT_4X4_50, legacy=False, tex_w=1200)
            view_8x11 = render_charuco_view(tex_8x11, K_true, dist_true, tz=0.44, board_phys_w=0.20)
            cal3 = CharucoCalibrator(3)
            cc3, ci3, _, _, _ = cal3.detect_board(view_8x11)
            assert ci3 is not None and len(ci3) >= 10
            assert (cal3.squares_x, cal3.squares_y) == (8, 11), f"Got {(cal3.squares_x, cal3.squares_y)}"
            print(f"[PASS 3] Auto-detected rotated 8x11 layout ({len(ci3)} corners)")

            sharp_view = render_charuco_view(tex_default, K_true, dist_true, tz=0.38, rx_deg=4, ry_deg=-5)

            # 4. Blurry / out-of-focus lens -> BLURRY_LENS ERROR & rejected from capture
            cal_diag = CharucoCalibrator(4)
            cal_diag.capture_cooldown = 0.0
            blurry_view = cv2.GaussianBlur(sharp_view, (0, 0), 3.5)
            rep_blur = cal_diag.diagnose_frame(blurry_view)
            assert rep_blur["status"] == "ERROR", f"Expected ERROR for blur: {rep_blur}"
            assert "BLURRY_LENS" in issue_codes(rep_blur), f"Missing BLURRY_LENS: {rep_blur}"
            _, good_flag, _ = cal_diag.process_frame(blurry_view, auto_capture=True)
            assert not good_flag and not cal_diag.try_auto_capture() and len(cal_diag.all_corners) == 0
            print(f"[PASS 4] Blurry lens self-reported BLURRY_LENS ({rep_blur['summary']}) and blocked sample")

            # 5. Tilted / uneven focus -> UNEVEN_FOCUS
            half_blur = sharp_view.copy()
            mid_x = sharp_view.shape[1] // 2
            blurred_half = cv2.GaussianBlur(sharp_view, (0, 0), 3.0)
            half_blur[:, mid_x:] = blurred_half[:, mid_x:]
            rep_uneven = cal_diag.diagnose_frame(half_blur)
            assert "UNEVEN_FOCUS" in issue_codes(rep_uneven), f"Missing UNEVEN_FOCUS: {rep_uneven}"
            print(f"[PASS 5] Tilted/uneven focus detected UNEVEN_FOCUS (uniformity={rep_uneven['metrics'].get('focus_uniformity')})")

            # 6. Smudged / low-contrast lens -> LOW_CONTRAST
            low_c = np.clip(sharp_view.astype(np.float32) * 0.16 + 110.0, 0, 255).astype(np.uint8)
            rep_lowc = cal_diag.diagnose_frame(low_c)
            assert "LOW_CONTRAST" in issue_codes(rep_lowc), f"Missing LOW_CONTRAST: {rep_lowc}"
            print(f"[PASS 6] Low contrast lens detected LOW_CONTRAST (contrast={rep_lowc['metrics'].get('contrast')})")

            # 7. Overexposed board -> OVEREXPOSED
            over = np.clip(sharp_view.astype(np.float32) * 1.6 + 120.0, 0, 255).astype(np.uint8)
            rep_over = cal_diag.diagnose_frame(over)
            assert "OVEREXPOSED" in issue_codes(rep_over), f"Missing OVEREXPOSED: {rep_over}"
            print(f"[PASS 7] Overexposed board detected OVEREXPOSED (clipped={rep_over['metrics'].get('clipped_white_frac')})")

            # 8. Underexposed board -> UNDEREXPOSED
            under = np.clip(sharp_view.astype(np.float32) * 0.14 + 4.0, 0, 255).astype(np.uint8)
            rep_under = cal_diag.diagnose_frame(under)
            assert "UNDEREXPOSED" in issue_codes(rep_under), f"Missing UNDEREXPOSED: {rep_under}"
            print(f"[PASS 8] Underexposed board detected UNDEREXPOSED (luma={rep_under['metrics'].get('mean_luma')})")

            # 9. Heavy sensor noise on blurry image -> BOTH HIGH_NOISE and BLURRY_LENS
            rng = np.random.default_rng(42)
            noisy_blur = np.clip(
                blurry_view.astype(np.float32) + rng.normal(0.0, 25.0, blurry_view.shape),
                0, 255,
            ).astype(np.uint8)
            rep_nb = cal_diag.diagnose_frame(noisy_blur)
            codes_nb = issue_codes(rep_nb)
            assert "HIGH_NOISE" in codes_nb and "BLURRY_LENS" in codes_nb, f"Expected both HIGH_NOISE & BLURRY_LENS: {rep_nb}"
            print(f"[PASS 9] Noisy+blurry image detected both HIGH_NOISE & BLURRY_LENS (noise={rep_nb['metrics'].get('noise_sigma')}, sharp={rep_nb['metrics'].get('edge_sharpness')})")

            # 10. Calibration drift vs previously stored cal.json
            undistort.store_calibration(1, {
                "meta": {
                    "resolution": {"width": 1280, "height": 800},
                    "camera_matrix": [[700.0, 0.0, 590.0], [0.0, 700.0, 400.0], [0.0, 0.0, 1.0]],
                    "dist_coeffs": [[0.0, 0.0, 0.0, 0.0, 0.0]],
                }
            })
            cal10 = CharucoCalibrator(1)
            for v in views:
                cc, ci, mc, mi, gray = cal10.detect_board(v)
                cal10.add_sample(gray, cc, ci)
            res10 = cal10.calibrate()
            assert res10 is not None
            codes10 = issue_codes(res10["meta"]["diagnostics"])
            assert "CALIBRATION_DRIFT" in codes10, f"Expected CALIBRATION_DRIFT, got: {res10['meta']['diagnostics']}"
            print(f"[PASS 10] Calibration drift detected CALIBRATION_DRIFT")

            # Bonus offline CLI --diagnose-only smoke check
            diag_dir = tmp / "cam4_1280x800_test"
            diag_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(diag_dir / "001_sharp.png"), sharp_view)
            cv2.imwrite(str(diag_dir / "002_blur.png"), blurry_view)
            cli = subprocess.run(
                [sys.executable, str(REPO_ROOT / "calibration" / "calibrate_offline.py"), str(diag_dir), "--diagnose-only", "--json", "--cal", str(tmp / "cal.json")],
                capture_output=True, text=True, check=False,
            )
            assert cli.returncode == 2, f"Expected exit code 2 for blurry frame, got {cli.returncode}: {cli.stdout} {cli.stderr}"
            cli_data = json.loads(cli.stdout)
            assert cli_data["status"] == "ERROR"
            print("[PASS CLI] calibrate_offline.py --diagnose-only flagged blurry PNG cleanly")

            # 11. Camera-less VisionManager + Hub /api/cameras + Dashboard /api/calibration/health
            try:
                from pipeline.htmlserver import CameraIndexServer, DashboardServer
                from pipeline.visionmanager import VisionManager
            except ImportError as exc:
                if require_full:
                    raise
                print(f"[SKIP 11] Skipping full Dash/NT4 server test on host without robot venv deps ({exc})")
            else:
                cfg_path = tmp / "config.json"
                with open(REPO_ROOT / "config.json", "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                cfg["network_tables"] = {"server": "127.0.0.1", "team_number": 846}
                for cam in cfg["cameras"]:
                    cam["device_id"] = f"NONEXISTENT_TEST_{cam['id']}"
                with open(cfg_path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f)

                old_start_thread = DashboardServer.start_server_thread
                old_hub_start = CameraIndexServer.start_server
                DashboardServer.start_server_thread = lambda self: None
                CameraIndexServer.start_server = lambda self: None
                vm = None
                try:
                    vm = VisionManager(str(cfg_path))
                    # Simulate a disconnected Pi camera read failure cleanly without opening webcam 0 on Windows
                    p1 = vm.get_pipeline(1)
                    p1.cam.fail_count = 1
                    p1.cam._opened = True
                    hub = CameraIndexServer(vm)
                    dash = DashboardServer(vm, 1)

                    hub_resp = hub.server.test_client().get("/api/cameras")
                    assert hub_resp.status_code == 200
                    hub_json = hub_resp.get_json()
                    cam1_entry = [c for c in hub_json["cameras"] if c["id"] == 1][0]
                    assert cam1_entry["health"]["status"] == "ERROR"
                    assert any(i["code"] == "CAMERA_DISCONNECTED" for i in cam1_entry["health"]["issues"])

                    cal_resp = dash.server.test_client().get("/api/calibration/health")
                    assert cal_resp.status_code == 200
                    cal_json = cal_resp.get_json()
                    assert cal_json["camera_id"] == 1
                    assert cal_json["health"]["status"] == "ERROR"
                    print("[PASS 11] Camera-less VisionManager + /api/cameras + /api/calibration/health self-reported CAMERA_DISCONNECTED")
                finally:
                    DashboardServer.start_server_thread = old_start_thread
                    CameraIndexServer.start_server = old_hub_start
                    if vm is not None:
                        vm.stop_all()

        finally:
            undistort.CAL_PATH = old_cal_path
            undistort._calibrations_loaded = False

    print("ALL CHARUCO + DIAGNOSTICS TESTS PASSED")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-full", action="store_true", help="Fail if Dash/Flask/ntcore are not installed")
    args = parser.parse_args()
    raise SystemExit(run_all_tests(require_full=args.require_full))
