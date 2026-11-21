import base64
import math
import os
import time
from threading import Lock
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from localization.undistort import calibration_for, store_calibration
from util.config import ConfigCategory
from util.logger import Logger

logger = Logger("CharucoCalibrator")

_cal_prefs = ConfigCategory("Calibration")

fix_principal_point = _cal_prefs.getIntConfig("fix_principal_point", 0)
zero_tangent_dist = _cal_prefs.getIntConfig("zero_tangent_dist", 0)
save_samples = _cal_prefs.getIntConfig("save_samples", 0)

SAMPLES_DIR = os.path.join("captures", "calibration")


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def _immerkaer_noise_sigma(gray: np.ndarray) -> float:
    """Fast robust noise std estimate using Immerkaer's 3x3 Laplacian mask."""
    h, w = gray.shape[:2]
    if h < 8 or w < 8:
        return 0.0
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    conv = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kernel)
    inner = np.abs(conv[1:-1, 1:-1])
    # Median-based scale is much less biased by crisp checkerboard edges than mean.
    return float(np.median(inner) * 1.4826 / 6.0)


def _edge_sharpness_score(gray: np.ndarray, contrast: float) -> Tuple[float, float, float]:
    """Return (laplacian_var, denoised_laplacian_var, norm_edge_sharpness)."""
    if gray.size < 64:
        return 0.0, 0.0, 0.0
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    smooth = cv2.GaussianBlur(gray, (5, 5), 1.0)
    denoised_lap_var = float(cv2.Laplacian(smooth, cv2.CV_64F).var())
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    # 95th percentile gradient captures edge steepness; divide by board contrast so
    # low-brightness sharp boards are not confused with out-of-focus blur.
    p95_grad = float(np.percentile(mag, 95))
    norm_sharp = p95_grad / max(contrast, 25.0)
    return lap_var, denoised_lap_var, norm_sharp


def _quadrant_sharpness_uniformity(gray_roi: np.ndarray) -> Optional[Tuple[float, List[float]]]:
    """Compare edge sharpness across 2x2 sub-quadrants of the ChArUco board ROI."""
    h, w = gray_roi.shape[:2]
    if h < 120 or w < 120:
        return None
    mh, mw = h // 2, w // 2
    quads = [
        gray_roi[:mh, :mw],
        gray_roi[:mh, mw:],
        gray_roi[mh:, :mw],
        gray_roi[mh:, mw:],
    ]
    scores: List[float] = []
    for q in quads:
        q_contrast = float(np.percentile(q, 95) - np.percentile(q, 5))
        if q_contrast < 25.0:
            return None
        _, _, ns = _edge_sharpness_score(q, q_contrast)
        scores.append(ns)
    max_s = max(scores)
    if max_s <= 1e-6:
        return None
    return float(min(scores) / max_s), scores


def _make_issue(
    code: str,
    severity: str,
    metric: str,
    value: float,
    threshold: float,
    message: str,
    recommendation: str,
) -> Dict:
    return {
        "code": code,
        "severity": severity,
        "metric": metric,
        "value": round(float(value), 4),
        "threshold": round(float(threshold), 4),
        "message": message,
        "recommendation": recommendation,
    }


def _severity_status(issues: List[Dict]) -> str:
    if any(i["severity"] == "ERROR" for i in issues):
        return "ERROR"
    if any(i["severity"] == "WARNING" for i in issues):
        return "WARNING"
    return "OK"


class CharucoCalibrator:
    """ChArUco camera calibrator with built-in blur & optical defect self-diagnostics.

    Compatible with both OpenCV 4.x (4.7-4.13) and OpenCV 5.0+ via CharucoDetector
    and cv2.calibrateCameraExtended.
    """

    SQUARES_X = 11
    SQUARES_Y = 8
    SQUARE_LENGTH = 0.0212
    MARKER_LENGTH = 0.0155466
    ARUCO_DICT = cv2.aruco.DICT_4X4_50

    MIN_SAMPLES = 15
    TARGET_SAMPLES = 54

    def __init__(
        self,
        camera_id: int,
        resolution: Tuple[int, int] = (1280, 800),
        on_health_update: Optional[Callable[[Dict], None]] = None,
    ):
        self.camera_id = camera_id
        self.resolution = resolution
        self.lock = Lock()
        self._on_health_update = on_health_update

        self.squares_x = self.SQUARES_X
        self.squares_y = self.SQUARES_Y
        self.square_length = self.SQUARE_LENGTH
        self.marker_length = self.MARKER_LENGTH
        self.aruco_dict_id = self.ARUCO_DICT
        self.legacy_pattern = False

        self.detector_params = cv2.aruco.DetectorParameters()
        if hasattr(self.detector_params, "minMarkerPerimeterRate"):
            self.detector_params.minMarkerPerimeterRate = 0.025
        if hasattr(self.detector_params, "maxErroneousBitsInBorderRate"):
            self.detector_params.maxErroneousBitsInBorderRate = 0.35
        if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
            self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            if hasattr(self.detector_params, "cornerRefinementWinSize"):
                self.detector_params.cornerRefinementWinSize = 5
            if hasattr(self.detector_params, "cornerRefinementMaxIterations"):
                self.detector_params.cornerRefinementMaxIterations = 30
            if hasattr(self.detector_params, "cornerRefinementMinAccuracy"):
                self.detector_params.cornerRefinementMinAccuracy = 0.05

        self._dict_candidates = [
            cv2.aruco.DICT_4X4_50,
            cv2.aruco.DICT_4X4_100,
            cv2.aruco.DICT_4X4_250,
            cv2.aruco.DICT_4X4_1000,
        ]
        self._layout_candidates = [(11, 8), (8, 11)]
        self._board_autodetected = False

        self.aruco_dict = None
        self.board = None
        self._charuco_detector = None
        self._aruco_detector = None
        self._build_board_and_detectors()

        self.all_corners: List[np.ndarray] = []
        self.all_ids: List[np.ndarray] = []
        self.sample_health: List[Dict] = []
        self.sample_dir: Optional[str] = None
        self.image_size: Optional[Tuple[int, int]] = None
        self.is_calibrating = False
        self.calibration_result: Optional[Dict] = None
        self.last_frame_annotated: Optional[np.ndarray] = None
        self.status_message = "Ready to calibrate (ChArUco 11x8 4x4)"
        self.progress = 0.0

        self.auto_capture_enabled = True
        self.last_capture_time = 0.0
        self.capture_cooldown = 0.3
        self.last_good_frame: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]] = None

        self.coverage_grid = np.zeros((3, 3), dtype=int)
        self.max_samples_per_cell = 6

        self.min_corners_live = 6
        self.min_corners_capture = 10

        # Optical & exposure health thresholds (calibrated for 8-bit grayscale ChArUco ROIs)
        self.blur_laplacian_var_min = 90.0
        self.blur_denoised_lap_min = 45.0
        self.edge_sharpness_error = 1.45
        self.edge_sharpness_warn = 2.00
        self.focus_uniformity_warn = 0.52
        self.focus_uniformity_error = 0.35
        self.min_contrast_warn = 65.0
        self.min_luma_warn = 35.0
        self.min_p90_luma_warn = 80.0
        self.max_clipped_white_frac = 0.12
        self.max_noise_sigma_warn = 11.5

        self.min_board_bbox_area_frac = 0.03
        self.min_board_span_frac = 0.15
        self.min_quadrants_covered = 1

        self.last_health_report: Dict = {
            "status": "UNKNOWN",
            "ok": True,
            "summary": "Show ChArUco board to run optical health check",
            "issues": [],
            "metrics": {},
            "timestamp": time.time(),
        }

    def _make_board(self, sx: int, sy: int, dict_id: int, legacy: bool):
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        board = cv2.aruco.CharucoBoard(
            (sx, sy), self.square_length, self.marker_length, aruco_dict
        )
        if hasattr(board, "setLegacyPattern"):
            board.setLegacyPattern(bool(legacy))
        return aruco_dict, board

    def _make_charuco_detector(self, board):
        if hasattr(cv2.aruco, "CharucoDetector"):
            charuco_params = cv2.aruco.CharucoParameters()
            if hasattr(charuco_params, "tryRefineMarkers"):
                charuco_params.tryRefineMarkers = True
            if hasattr(charuco_params, "minMarkers"):
                charuco_params.minMarkers = 1
            return cv2.aruco.CharucoDetector(board, charuco_params, self.detector_params)
        return None

    def _build_board_and_detectors(self) -> None:
        self.aruco_dict, self.board = self._make_board(
            self.squares_x, self.squares_y, self.aruco_dict_id, self.legacy_pattern
        )
        self._charuco_detector = self._make_charuco_detector(self.board)
        self._aruco_detector = (
            cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
            if hasattr(cv2.aruco, "ArucoDetector")
            else None
        )

    def generate_board_image(
        self, width: int = 1100, height: int = 800, margin_px: int = 20
    ) -> np.ndarray:
        if hasattr(self.board, "generateImage"):
            img = self.board.generateImage((width, height), marginSize=margin_px, borderBits=1)
        else:
            img = self.board.draw((width, height), marginSize=margin_px, borderBits=1)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def _detect_with_board(
        self, gray: np.ndarray, board, charuco_detector=None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple], Optional[np.ndarray]]:
        det = charuco_detector if charuco_detector is not None else self._make_charuco_detector(board)
        if det is not None:
            cc, ci, mc, mi = det.detectBoard(gray)
            if cc is not None and len(cc) > 0 and ci is not None and len(ci) > 0:
                cc = np.ascontiguousarray(cc, dtype=np.float32).reshape(-1, 1, 2)
                ci = np.ascontiguousarray(ci, dtype=np.int32).reshape(-1, 1)
                cc = cv2.cornerSubPix(
                    gray,
                    cc,
                    (5, 5),
                    (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                )
            return cc, ci, mc, mi

        # Legacy OpenCV < 4.7 fallback
        if hasattr(cv2.aruco, "ArucoDetector"):
            mc, mi, _ = cv2.aruco.ArucoDetector(board.getDictionary(), self.detector_params).detectMarkers(gray)
        else:
            mc, mi, _ = cv2.aruco.detectMarkers(gray, board.getDictionary(), parameters=self.detector_params)
        cc, ci = None, None
        if mi is not None and len(mi) > 0 and hasattr(cv2.aruco, "interpolateCornersCharuco"):
            ret, cc_tmp, ci_tmp = cv2.aruco.interpolateCornersCharuco(mc, mi, gray, board, minMarkers=1)
            if ret and ret > 0 and cc_tmp is not None and ci_tmp is not None:
                cc = np.ascontiguousarray(cc_tmp, dtype=np.float32).reshape(-1, 1, 2)
                ci = np.ascontiguousarray(ci_tmp, dtype=np.int32).reshape(-1, 1)
        return cc, ci, mc, mi

    def detect_board(
        self, frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple], Optional[np.ndarray], np.ndarray]:
        """Detect ChArUco corners + ArUco markers, auto-probing legacyPattern/layout/dict once."""
        gray_full = _to_gray(frame)
        cc, ci, mc, mi = self._detect_with_board(gray_full, self.board, self._charuco_detector)

        has_corners = cc is not None and ci is not None and len(ci) > 0
        has_markers = mi is not None and len(mi) > 0

        if not has_corners and not self._board_autodetected:
            if has_markers:
                for sx, sy in self._layout_candidates:
                    for legacy in (self.legacy_pattern, not self.legacy_pattern):
                        if (sx, sy) == (self.squares_x, self.squares_y) and legacy == self.legacy_pattern:
                            continue
                        try:
                            _, cand_board = self._make_board(sx, sy, self.aruco_dict_id, legacy)
                            cc2, ci2, mc2, mi2 = self._detect_with_board(gray_full, cand_board)
                            if cc2 is not None and ci2 is not None and len(ci2) >= 4:
                                self.squares_x, self.squares_y = sx, sy
                                self.legacy_pattern = legacy
                                self._board_autodetected = True
                                self._build_board_and_detectors()
                                logger.Log(
                                    f"Auto-detected ChArUco layout {sx}x{sy} (legacyPattern={legacy})"
                                )
                                return cc2, ci2, mc2, mi2, gray_full
                        except Exception:
                            continue
            else:
                for dict_id in self._dict_candidates:
                    if dict_id == self.aruco_dict_id:
                        continue
                    for legacy in (False, True):
                        try:
                            _, cand_board = self._make_board(self.squares_x, self.squares_y, dict_id, legacy)
                            cc2, ci2, mc2, mi2 = self._detect_with_board(gray_full, cand_board)
                            if cc2 is not None and ci2 is not None and len(ci2) >= 4:
                                self.aruco_dict_id = dict_id
                                self.legacy_pattern = legacy
                                self._board_autodetected = True
                                self._build_board_and_detectors()
                                logger.Log(
                                    f"Auto-detected ChArUco dict={dict_id} legacyPattern={legacy}"
                                )
                                return cc2, ci2, mc2, mi2, gray_full
                        except Exception:
                            continue
        elif has_corners and len(ci) >= 6:
            self._board_autodetected = True

        return cc, ci, mc, mi, gray_full

    def _board_roi_from_points(
        self, gray: np.ndarray, charuco_corners: Optional[np.ndarray], marker_corners
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        pts_list = []
        if charuco_corners is not None and len(charuco_corners) > 0:
            pts_list.append(charuco_corners.reshape(-1, 2))
        if marker_corners is not None and len(marker_corners) > 0:
            for mc in marker_corners:
                pts_list.append(np.asarray(mc, dtype=np.float32).reshape(-1, 2))
        if not pts_list:
            return None, None
        all_pts = np.vstack(pts_list)
        h, w = gray.shape[:2]
        x1 = max(0, int(np.floor(np.min(all_pts[:, 0]))))
        y1 = max(0, int(np.floor(np.min(all_pts[:, 1]))))
        x2 = min(w, int(np.ceil(np.max(all_pts[:, 0]))))
        y2 = min(h, int(np.ceil(np.max(all_pts[:, 1]))))
        if x2 - x1 < 20 or y2 - y1 < 20:
            return None, None
        return gray[y1:y2, x1:x2], (x1, y1, x2, y2)

    def analyze_frame_health(
        self,
        gray: np.ndarray,
        charuco_corners: Optional[np.ndarray],
        charuco_ids: Optional[np.ndarray],
        marker_corners,
        marker_ids: Optional[np.ndarray],
    ) -> Dict:
        """Evaluate camera focus/blur, exposure, contrast, noise, and uniformity on the ChArUco board."""
        n_markers = int(len(marker_ids)) if marker_ids is not None else 0
        n_corners = int(len(charuco_ids)) if charuco_ids is not None else 0
        roi, bbox = self._board_roi_from_points(gray, charuco_corners, marker_corners)

        if roi is None or (n_markers == 0 and n_corners == 0):
            mean_luma = float(np.mean(gray))
            clipped_white = float(np.mean(gray >= 250))
            contrast = float(np.percentile(gray, 95) - np.percentile(gray, 5))
            issues: List[Dict] = []
            if mean_luma < 12.0:
                issues.append(_make_issue(
                    "CAMERA_DARK_OR_COVERED", "ERROR", "mean_luma", mean_luma, 12.0,
                    f"Image is nearly black (mean luma {mean_luma:.1f})",
                    "Check lens cap, exposure setting, or camera sensor health.",
                ))
            elif clipped_white > 0.60:
                issues.append(_make_issue(
                    "CAMERA_SATURATED", "ERROR", "clipped_white_frac", clipped_white, 0.60,
                    f"Image is blown out ({clipped_white * 100:.0f}% white pixels)",
                    "Lower camera exposure/gain.",
                ))
            status = _severity_status(issues) if issues else "NO_BOARD"
            return {
                "status": status,
                "ok": len(issues) == 0,
                "board_detected": False,
                "summary": issues[0]["message"] if issues else "No ChArUco board detected",
                "issues": issues,
                "metrics": {
                    "markers": n_markers,
                    "corners": n_corners,
                    "mean_luma": round(mean_luma, 1),
                    "contrast": round(contrast, 1),
                    "clipped_white_frac": round(clipped_white, 4),
                    "blur_score": 0.0,
                },
                "timestamp": time.time(),
            }

        mean_luma = float(np.mean(roi))
        p5, p10, p90, p95 = [float(v) for v in np.percentile(roi, [5, 10, 90, 95])]
        contrast = p95 - p5
        clipped_white = float(np.mean(roi >= 252))
        noise_sigma = _immerkaer_noise_sigma(roi)
        lap_var, denoised_lap_var, edge_sharp = _edge_sharpness_score(roi, contrast)
        unif_res = _quadrant_sharpness_uniformity(roi)
        focus_uniformity = unif_res[0] if unif_res is not None else 1.0
        quad_scores = unif_res[1] if unif_res is not None else []

        issues = []

        # 1. Exposure / clipping / haze checks
        # Note: a sharp ChArUco board is ~35-42% white squares, so only flag overexposure
        # when saturation spills past white squares into dark markers/squares or washes blacks out.
        is_overexposed = (
            clipped_white > 0.48
            or (clipped_white > self.max_clipped_white_frac and (mean_luma > 165.0 or p10 > 80.0))
        )
        if is_overexposed:
            issues.append(_make_issue(
                "OVEREXPOSED", "WARNING" if clipped_white < 0.55 and p10 <= 110.0 else "ERROR",
                "clipped_white_frac", clipped_white, self.max_clipped_white_frac,
                f"Board overexposed ({clipped_white * 100:.1f}% clipped white pixels, mean={mean_luma:.0f})",
                "Reduce camera exposure or gain so white squares do not clip.",
            ))
        if mean_luma < self.min_luma_warn or p90 < self.min_p90_luma_warn:
            issues.append(_make_issue(
                "UNDEREXPOSED", "ERROR" if mean_luma < 20.0 else "WARNING",
                "mean_luma", mean_luma, self.min_luma_warn,
                f"Board too dark (mean={mean_luma:.1f}, p90={p90:.1f})",
                "Increase exposure/lighting or verify lens aperture.",
            ))
        elif contrast < self.min_contrast_warn:
            issues.append(_make_issue(
                "LOW_CONTRAST", "ERROR" if contrast < 40.0 else "WARNING",
                "contrast", contrast, self.min_contrast_warn,
                f"Low board contrast ({contrast:.1f}) - possible smudged/dirty lens or glare",
                "Clean the camera lens and eliminate direct glare on the ChArUco board.",
            ))

        # 2. Sensor grain / excessive gain check
        if noise_sigma > self.max_noise_sigma_warn:
            issues.append(_make_issue(
                "HIGH_NOISE", "WARNING" if noise_sigma < 22.0 else "ERROR",
                "noise_sigma", noise_sigma, self.max_noise_sigma_warn,
                f"High sensor noise (sigma={noise_sigma:.1f})",
                "Lower camera gain and increase lighting or exposure slightly.",
            ))

        # 3. Blur / focus checks (contrast-normalized edge sharpness + denoised Laplacian
        #    so high sensor grain cannot mask an out-of-focus lens)
        is_severe_blur = (
            edge_sharp < self.edge_sharpness_error
            or denoised_lap_var < self.blur_denoised_lap_min
        )
        is_mild_blur = (
            edge_sharp < self.edge_sharpness_warn
            or lap_var < self.blur_laplacian_var_min
        )
        if is_severe_blur:
            issues.append(_make_issue(
                "BLURRY_LENS", "ERROR", "edge_sharpness", edge_sharp, self.edge_sharpness_error,
                f"Camera is blurry / out of focus (sharpness={edge_sharp:.2f}, lap={denoised_lap_var:.0f})",
                "Refocus the lens on the Focus/Calibration tab until sharpness > 2.5 and tighten lock ring.",
            ))
        elif is_mild_blur:
            issues.append(_make_issue(
                "SOFT_FOCUS", "WARNING", "edge_sharpness", edge_sharp, self.edge_sharpness_warn,
                f"Camera focus is soft (sharpness={edge_sharp:.2f}, lap={denoised_lap_var:.0f})",
                "Fine-tune lens focus ring for sharper AprilTag edges at long range.",
            ))

        # 4. Uneven focus across board (tilted lens / decentered element)
        if unif_res is not None and not is_severe_blur:
            if focus_uniformity < self.focus_uniformity_error:
                issues.append(_make_issue(
                    "UNEVEN_FOCUS", "ERROR", "focus_uniformity", focus_uniformity, self.focus_uniformity_error,
                    f"Severe uneven focus across board (uniformity={focus_uniformity:.2f})",
                    "Check for tilted lens mount, loose M12 threads, or warped lens holder.",
                ))
            elif focus_uniformity < self.focus_uniformity_warn:
                issues.append(_make_issue(
                    "UNEVEN_FOCUS", "WARNING", "focus_uniformity", focus_uniformity, self.focus_uniformity_warn,
                    f"Uneven focus across board (uniformity={focus_uniformity:.2f})",
                    "Check lens seating and verify board is parallel to sensor.",
                ))

        # 5. Marker-to-corner drop diagnostic
        if n_markers >= 4 and n_corners < 4:
            issues.append(_make_issue(
                "CORNER_INTERPOLATION_FAILED", "ERROR", "corners", float(n_corners), 4.0,
                f"Detected {n_markers} ArUco markers but only {n_corners} ChArUco corners",
                "Board may be heavily blurred, motion-blurred, warped, or mismatched with 11x8 4x4 config.",
            ))

        status = _severity_status(issues)
        summary = (
            "Camera optical health OK"
            if not issues
            else "; ".join(i["message"] for i in issues[:2])
        )
        return {
            "status": status,
            "ok": status == "OK",
            "board_detected": True,
            "summary": summary,
            "issues": issues,
            "metrics": {
                "markers": n_markers,
                "corners": n_corners,
                "blur_score": round(denoised_lap_var, 1),
                "raw_laplacian_var": round(lap_var, 1),
                "edge_sharpness": round(edge_sharp, 3),
                "focus_uniformity": round(focus_uniformity, 3),
                "quadrant_sharpness": [round(s, 3) for s in quad_scores],
                "mean_luma": round(mean_luma, 1),
                "contrast": round(contrast, 1),
                "clipped_white_frac": round(clipped_white, 4),
                "noise_sigma": round(noise_sigma, 2),
                "bbox": list(bbox) if bbox else None,
            },
            "timestamp": time.time(),
        }

    def diagnose_frame(self, frame: np.ndarray) -> Dict:
        """One-shot ChArUco optical health check on a single frame (does not alter calibration samples)."""
        cc, ci, mc, mi, gray = self.detect_board(frame)
        report = self.analyze_frame_health(gray, cc, ci, mc, mi)
        with self.lock:
            self.last_health_report = report
        if self._on_health_update is not None and (report.get("board_detected") or not report.get("ok")):
            try:
                self._on_health_update(report)
            except Exception:
                pass
        return report

    def _get_board_position(
        self, corners: np.ndarray, frame_shape: Tuple[int, int]
    ) -> Tuple[Optional[int], Optional[int]]:
        if corners is None or len(corners) == 0:
            return None, None
        pts = corners.reshape(-1, 2)
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        h, w = frame_shape[:2]
        return max(0, min(2, int(cy / h * 3))), max(0, min(2, int(cx / w * 3)))

    def _draw_coverage_grid(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        cell_h, cell_w = h // 3, w // 3
        overlay = frame.copy()
        for row in range(3):
            for col in range(3):
                count = int(self.coverage_grid[row, col])
                x1, y1 = col * cell_w, row * cell_h
                x2, y2 = (col + 1) * cell_w, (row + 1) * cell_h
                if count == 0:
                    color = (0, 0, 200)
                elif count < self.max_samples_per_cell:
                    color = (0, 200, 200)
                else:
                    color = (0, 200, 0)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (200, 200, 200), 2)
                text = str(count)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                cv2.putText(
                    overlay, text,
                    (x1 + (cell_w - tw) // 2, y1 + (cell_h + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                )
        return cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)

    def _is_sample_acceptable(
        self,
        corners: Optional[np.ndarray],
        frame_shape: Tuple[int, int],
        required_min_points: int,
        health: Dict,
    ) -> Tuple[bool, str]:
        if corners is None or len(corners) < required_min_points:
            return False, f"Need {required_min_points}+ ChArUco corners"

        for issue in health.get("issues", []):
            if issue["severity"] == "ERROR":
                return False, issue["message"]
            if issue["code"] in ("BLURRY_LENS", "SOFT_FOCUS"):
                return False, issue["message"]

        h, w = frame_shape[:2]
        pts = corners.reshape(-1, 2)
        min_xy = np.min(pts, axis=0)
        max_xy = np.max(pts, axis=0)
        bw = float(max_xy[0] - min_xy[0])
        bh = float(max_xy[1] - min_xy[1])
        if bw <= 1.0 or bh <= 1.0:
            return False, "Bad geometry"
        if (bw * bh) / float(w * h) < self.min_board_bbox_area_frac:
            return False, "Board too small"
        if max(bw / w, bh / h) < self.min_board_span_frac:
            return False, "Board too small"
        cx, cy = w * 0.5, h * 0.5
        quads = {(0 if x < cx else 1, 0 if y < cy else 1) for x, y in pts}
        if len(quads) < self.min_quadrants_covered:
            return False, "Move board around"
        return True, "OK"

    def process_frame(
        self, frame: np.ndarray, downsample: bool = True, auto_capture: bool = True
    ) -> Tuple[Optional[np.ndarray], bool, str]:
        if frame is None:
            logger.Warn("process_frame called with None frame")
            return None, False, "No frame"
        if self.is_calibrating:
            return frame, False, self.status_message

        cc, ci, mc, mi, gray_full = self.detect_board(frame)
        health = self.analyze_frame_health(gray_full, cc, ci, mc, mi)

        annotated = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
        annotated = self._draw_coverage_grid(annotated)

        n_corners = int(len(ci)) if ci is not None else 0
        n_markers = int(len(mi)) if mi is not None else 0

        if mi is not None and len(mi) > 0:
            cv2.aruco.drawDetectedMarkers(annotated, mc, mi)
        if cc is not None and ci is not None and len(ci) > 0:
            if hasattr(cv2.aruco, "drawDetectedCornersCharuco"):
                cv2.aruco.drawDetectedCornersCharuco(annotated, cc, ci, (0, 255, 0))
            else:
                for pt in cc.reshape(-1, 2):
                    cv2.circle(annotated, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)

        is_good_sample = False
        should_capture = False
        status = health["summary"]

        if n_corners >= self.min_corners_live:
            ok, reason = self._is_sample_acceptable(cc, annotated.shape, self.min_corners_live, health)
            is_good_sample = ok
            if not ok:
                status = reason
            else:
                grid_row, grid_col = self._get_board_position(cc, annotated.shape)
                if grid_row is not None:
                    if self.coverage_grid[grid_row, grid_col] < self.max_samples_per_cell:
                        should_capture = True
                        status = f"Good sample ({n_corners} corners) - Cell [{grid_row},{grid_col}]"
                    else:
                        status = f"Move board (cell [{grid_row},{grid_col}] full)"
                else:
                    status = f"Good sample ({n_corners} corners)"
        elif n_markers > 0 and not health["issues"]:
            status = f"Detected {n_markers} markers, need {self.min_corners_live}+ ChArUco corners"

        h_status = health["status"]
        color = (0, 255, 0) if h_status == "OK" else ((0, 165, 255) if h_status == "WARNING" else (0, 0, 255))
        cv2.putText(annotated, status[:78], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(
            annotated,
            f"Samples: {len(self.all_corners)}/{self.TARGET_SAMPLES} | Corners: {n_corners} | Health: {h_status}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
        )
        m = health.get("metrics", {})
        if health.get("board_detected"):
            cv2.putText(
                annotated,
                f"Sharpness: {m.get('edge_sharpness', 0):.2f} | Lap: {m.get('blur_score', 0):.0f} | "
                f"Contrast: {m.get('contrast', 0):.0f} | Noise: {m.get('noise_sigma', 0):.1f}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
            )
        cv2.putText(
            annotated, f"Frame: {annotated.shape[1]}x{annotated.shape[0]}",
            (10, annotated.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

        with self.lock:
            self.last_frame_annotated = annotated
            self.last_health_report = health
            if self.is_calibrating or self.calibration_result is not None:
                return annotated, is_good_sample, status
            self.status_message = status
            self.progress = len(self.all_corners) / float(self.TARGET_SAMPLES)
            if (
                auto_capture
                and self.auto_capture_enabled
                and should_capture
                and len(self.all_corners) < self.TARGET_SAMPLES
            ):
                if time.time() - self.last_capture_time >= self.capture_cooldown:
                    if cc is not None and ci is not None and len(ci) >= self.min_corners_capture:
                        self.last_good_frame = (gray_full.copy(), cc.copy(), ci.copy(), dict(health))

        if self._on_health_update is not None and (health.get("board_detected") or not health.get("ok")):
            try:
                self._on_health_update(health)
            except Exception:
                pass

        return annotated, is_good_sample, status

    def add_sample(
        self,
        gray: np.ndarray,
        corners: np.ndarray,
        ids: np.ndarray,
        health: Optional[Dict] = None,
        save_to_disk: bool = False,
    ) -> bool:
        """Validate and append a ChArUco sample directly (used by auto-capture and offline CLI)."""
        if corners is None or ids is None or len(ids) < self.min_corners_capture:
            return False
        if health is None:
            health = self.analyze_frame_health(gray, corners, ids, None, None)
        ok, _ = self._is_sample_acceptable(corners, gray.shape, self.min_corners_capture, health)
        if not ok:
            return False

        if save_to_disk:
            self._save_sample(gray, len(self.all_corners) + 1)
        with self.lock:
            self.all_corners.append(corners)
            self.all_ids.append(ids)
            grid_row, grid_col = self._get_board_position(corners, gray.shape)
            sample_meta = dict(health.get("metrics", {}))
            sample_meta["grid"] = [grid_row, grid_col]
            self.sample_health.append(sample_meta)
            if self.image_size is None:
                self.image_size = (int(gray.shape[1]), int(gray.shape[0]))
            if grid_row is not None and grid_col is not None:
                self.coverage_grid[grid_row, grid_col] += 1
            self.last_capture_time = time.time()
            self.progress = len(self.all_corners) / float(self.TARGET_SAMPLES)
        return True

    def try_auto_capture(self) -> bool:
        with self.lock:
            if self.last_good_frame is None:
                return False
            gray, corners, ids, health = self.last_good_frame
            self.last_good_frame = None
        return self.add_sample(
            gray, corners, ids, health=health, save_to_disk=bool(save_samples.valueInt())
        )

    def _save_sample(self, gray: np.ndarray, index: int) -> None:
        if self.sample_dir is None:
            h, w = gray.shape[:2]
            self.sample_dir = os.path.join(
                SAMPLES_DIR, f"cam{self.camera_id}_{w}x{h}_{time.strftime('%Y%m%d_%H%M%S')}"
            )
            os.makedirs(self.sample_dir, exist_ok=True)
        path = os.path.join(self.sample_dir, f"{index:03d}.png")
        if not cv2.imwrite(path, gray):
            logger.Warn(f"Failed to save calibration sample {path}")

    def _matched_points(
        self, corners_list: List[np.ndarray], ids_list: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        obj_points: List[np.ndarray] = []
        img_points: List[np.ndarray] = []
        chess_corners = self.board.getChessboardCorners()
        for cc, ci in zip(corners_list, ids_list):
            if hasattr(self.board, "matchImagePoints"):
                obj_p, img_p = self.board.matchImagePoints(cc, ci)
            else:
                flat_ids = ci.reshape(-1).astype(int)
                obj_p = chess_corners[flat_ids].reshape(-1, 1, 3).astype(np.float32)
                img_p = cc.reshape(-1, 1, 2).astype(np.float32)
            if obj_p is not None and img_p is not None and len(obj_p) >= 6:
                obj_points.append(obj_p.astype(np.float32))
                img_points.append(img_p.astype(np.float32))
        return obj_points, img_points

    def _evaluate_calibration_health(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        mean_error: float,
        image_size: Tuple[int, int],
        sample_health: List[Dict],
        coverage_grid: np.ndarray,
        prev_cal: Optional[Tuple[np.ndarray, np.ndarray]],
    ) -> Dict:
        width, height = image_size
        fx = float(camera_matrix[0, 0])
        fy = float(camera_matrix[1, 1])
        cx = float(camera_matrix[0, 2])
        cy = float(camera_matrix[1, 2])
        fov_x = float(2.0 * np.degrees(np.arctan(width / (2.0 * fx))))
        fov_y = float(2.0 * np.degrees(np.arctan(height / (2.0 * fy))))

        issues: List[Dict] = []

        if mean_error > 1.5:
            issues.append(_make_issue(
                "HIGH_REPROJECTION_ERROR", "ERROR", "reprojection_error_px", mean_error, 1.5,
                f"High reprojection error ({mean_error:.3f} px)",
                "Recalibrate with a flat rigid ChArUco board, sharper focus, and slower motion.",
            ))
        elif mean_error > 0.75:
            issues.append(_make_issue(
                "HIGH_REPROJECTION_ERROR", "WARNING", "reprojection_error_px", mean_error, 0.75,
                f"Elevated reprojection error ({mean_error:.3f} px)",
                "Verify lens focus and board flatness; recapture more angled views.",
            ))

        dcx = abs(cx - width / 2.0) / float(width)
        dcy = abs(cy - height / 2.0) / float(height)
        max_pp_frac = max(dcx, dcy)
        if max_pp_frac > 0.12:
            issues.append(_make_issue(
                "PRINCIPAL_POINT_OFFSET", "ERROR", "principal_point_offset_frac", max_pp_frac, 0.12,
                f"Principal point severely off-center ({cx:.1f}, {cy:.1f} vs {width/2:.0f}, {height/2:.0f})",
                "Inspect lens/sensor alignment or capture wider corner coverage across all 9 grid cells.",
            ))
        elif max_pp_frac > 0.08:
            issues.append(_make_issue(
                "PRINCIPAL_POINT_OFFSET", "WARNING", "principal_point_offset_frac", max_pp_frac, 0.08,
                f"Principal point off-center ({cx:.1f}, {cy:.1f})",
                "Capture additional calibration views near all four frame corners.",
            ))

        aspect_skew = abs(fx - fy) / max(fx, fy, 1e-6)
        if aspect_skew > 0.08:
            issues.append(_make_issue(
                "ASPECT_RATIO_SKEW", "ERROR", "aspect_ratio_skew", aspect_skew, 0.08,
                f"Severe fx/fy skew ({fx:.1f} vs {fy:.1f}, {aspect_skew * 100:.1f}%)",
                "Check board square dimensions or image scaling distortion.",
            ))
        elif aspect_skew > 0.03:
            issues.append(_make_issue(
                "ASPECT_RATIO_SKEW", "WARNING", "aspect_ratio_skew", aspect_skew, 0.03,
                f"Moderate fx/fy skew ({fx:.1f} vs {fy:.1f}, {aspect_skew * 100:.1f}%)",
                "Include both tilted and rolled ChArUco views.",
            ))

        if fov_x > 120.0 or fov_y > 120.0 or fov_x < 30.0 or fov_y < 30.0:
            issues.append(_make_issue(
                "IMPLAUSIBLE_FOV", "ERROR", "fov_horizontal_deg", fov_x, 120.0,
                f"Implausible FOV (H={fov_x:.1f} deg, V={fov_y:.1f} deg)",
                "Verify SQUARES_X/Y, SQUARE_LENGTH, and MARKER_LENGTH match the physical ChArUco board.",
            ))

        fx_drift_pct = 0.0
        pp_drift_px = 0.0
        if prev_cal is not None:
            old_k, _ = prev_cal
            old_fx, old_fy = float(old_k[0, 0]), float(old_k[1, 1])
            old_cx, old_cy = float(old_k[0, 2]), float(old_k[1, 2])
            fx_drift_pct = max(
                abs(fx - old_fx) / max(old_fx, 1e-6),
                abs(fy - old_fy) / max(old_fy, 1e-6),
            )
            pp_drift_px = float(math.hypot(cx - old_cx, cy - old_cy))
            if fx_drift_pct > 0.03 or pp_drift_px > 15.0:
                issues.append(_make_issue(
                    "CALIBRATION_DRIFT", "WARNING", "focal_drift_frac", fx_drift_pct, 0.03,
                    f"Lens calibration drifted vs saved cal.json (focal {fx_drift_pct * 100:.1f}%, center {pp_drift_px:.1f} px)",
                    "Focus ring or camera mount may have shifted - check lock screw and save updated calibration.",
                ))

        sharp_vals = [float(s["edge_sharpness"]) for s in sample_health if "edge_sharpness" in s]
        blur_vals = [float(s["blur_score"]) for s in sample_health if "blur_score" in s]
        unif_vals = [float(s["focus_uniformity"]) for s in sample_health if "focus_uniformity" in s]
        med_sharp = float(np.median(sharp_vals)) if sharp_vals else 0.0
        med_blur = float(np.median(blur_vals)) if blur_vals else 0.0
        med_unif = float(np.median(unif_vals)) if unif_vals else 1.0

        if sharp_vals and med_sharp < self.edge_sharpness_warn:
            issues.append(_make_issue(
                "BLURRY_LENS", "ERROR" if med_sharp < self.edge_sharpness_error else "WARNING",
                "median_edge_sharpness", med_sharp, self.edge_sharpness_warn,
                f"Captured samples show soft/blurry focus (median sharpness={med_sharp:.2f})",
                "Refocus lens before competition.",
            ))

        cell_sharpness: Dict[Tuple[int, int], List[float]] = {}
        for s in sample_health:
            g = s.get("grid")
            if g and g[0] is not None and g[1] is not None and "edge_sharpness" in s:
                cell_sharpness.setdefault((int(g[0]), int(g[1])), []).append(float(s["edge_sharpness"]))
        spatial_ratio = med_unif
        if len(cell_sharpness) >= 3:
            cell_medians = [float(np.median(v)) for v in cell_sharpness.values()]
            if max(cell_medians) > 1e-6:
                spatial_ratio = min(spatial_ratio, float(min(cell_medians) / max(cell_medians)))
        if spatial_ratio < self.focus_uniformity_error:
            issues.append(_make_issue(
                "UNEVEN_FOCUS", "ERROR", "spatial_focus_ratio", spatial_ratio, self.focus_uniformity_error,
                f"Uneven focus across field of view (ratio={spatial_ratio:.2f}) - possible tilted lens",
                "Inspect lens mount seating and M12 threads.",
            ))
        elif spatial_ratio < self.focus_uniformity_warn:
            issues.append(_make_issue(
                "UNEVEN_FOCUS", "WARNING", "spatial_focus_ratio", spatial_ratio, self.focus_uniformity_warn,
                f"Noticeable corner/edge focus drop (ratio={spatial_ratio:.2f})",
                "Verify lens is seated squarely in its mount.",
            ))

        covered_cells = int(np.count_nonzero(coverage_grid))
        if covered_cells < 6:
            issues.append(_make_issue(
                "POOR_COVERAGE", "WARNING", "covered_cells", float(covered_cells), 6.0,
                f"Only {covered_cells}/9 grid cells covered during calibration",
                "Move the ChArUco board into all corners and edges for accurate lens distortion fit.",
            ))

        status = _severity_status(issues)
        summary = (
            f"Calibration & optical health OK ({mean_error:.3f} px error)"
            if not issues
            else "; ".join(i["message"] for i in issues[:2])
        )
        return {
            "status": status,
            "ok": status == "OK",
            "source": "charuco_calibration",
            "summary": summary,
            "issues": issues,
            "metrics": {
                "reprojection_error_px": round(mean_error, 4),
                "blur_score": round(med_blur, 1),
                "median_edge_sharpness": round(med_sharp, 3),
                "spatial_focus_ratio": round(spatial_ratio, 3),
                "principal_point_offset_frac": round(max_pp_frac, 4),
                "aspect_ratio_skew": round(aspect_skew, 4),
                "covered_cells": covered_cells,
                "focal_drift_frac": round(fx_drift_pct, 4),
                "principal_point_drift_px": round(pp_drift_px, 2),
                "fov_horizontal_deg": round(fov_x, 2),
                "fov_vertical_deg": round(fov_y, 2),
            },
            "timestamp": time.time(),
        }

    def calibrate(self) -> Optional[Dict]:
        with self.lock:
            if len(self.all_corners) < self.MIN_SAMPLES:
                logger.Warn(f"Not enough samples: {len(self.all_corners)}/{self.MIN_SAMPLES}")
                self.status_message = f"Need {self.MIN_SAMPLES - len(self.all_corners)} more samples"
                return None
            self.is_calibrating = True
            self.status_message = "Preparing ChArUco data... (0%)"
            self.progress = 0.0
            corners_list = list(self.all_corners)
            ids_list = list(self.all_ids)
            sample_health = list(self.sample_health)
            coverage_grid = self.coverage_grid.copy()
            image_size = self.image_size

        try:
            start_time = time.time()
            width, height = image_size
            prev_cal = calibration_for(self.camera_id, width, height)

            cal_flags = 0
            if hasattr(cv2, "CALIB_USE_INTRINSIC_GUESS"):
                cal_flags |= cv2.CALIB_USE_INTRINSIC_GUESS
            if fix_principal_point.valueInt() and hasattr(cv2, "CALIB_FIX_PRINCIPAL_POINT"):
                cal_flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
            if zero_tangent_dist.valueInt() and hasattr(cv2, "CALIB_ZERO_TANGENT_DIST"):
                cal_flags |= cv2.CALIB_ZERO_TANGENT_DIST

            camera_matrix_init = np.array(
                [[width, 0, width / 2.0], [0, width, height / 2.0], [0, 0, 1]],
                dtype=np.float64,
            )
            dist_coeffs_init = np.zeros(5, dtype=np.float64)

            obj_points, img_points = self._matched_points(corners_list, ids_list)
            if len(obj_points) < self.MIN_SAMPLES:
                raise ValueError(f"Only {len(obj_points)} valid ChArUco views after corner matching")

            with self.lock:
                self.progress = 0.25
                self.status_message = "Running ChArUco calibration... (25%)"

            ret, camera_matrix, dist_coeffs, *_, per_view = cv2.calibrateCameraExtended(
                obj_points,
                img_points,
                (width, height),
                camera_matrix_init,
                dist_coeffs_init,
                flags=cal_flags,
            )
            per_view = np.ravel(per_view)
            med_err = float(np.median(per_view))
            keep_idx = [i for i, e in enumerate(per_view) if e < max(2.0 * med_err, 0.25)]
            if self.MIN_SAMPLES <= len(keep_idx) < len(obj_points):
                logger.Log(
                    f"Dropped {len(obj_points) - len(keep_idx)} outlier views (> 2x median error), refitting"
                )
                with self.lock:
                    self.progress = 0.60
                    self.status_message = "Refitting without outlier views... (60%)"
                obj_points = [obj_points[i] for i in keep_idx]
                img_points = [img_points[i] for i in keep_idx]
                sample_health = [sample_health[i] for i in keep_idx if i < len(sample_health)]
                ret, camera_matrix, dist_coeffs, *_ = cv2.calibrateCamera(
                    obj_points,
                    img_points,
                    (width, height),
                    camera_matrix,
                    dist_coeffs,
                    flags=cal_flags,
                )

            mean_error = float(ret)
            if camera_matrix is None or np.any(np.isnan(camera_matrix)):
                raise ValueError("calibrateCamera returned invalid camera matrix")

            with self.lock:
                self.progress = 0.85
                self.status_message = "Running camera health diagnostics... (85%)"

            elapsed = time.time() - start_time
            fx = float(camera_matrix[0, 0])
            fy = float(camera_matrix[1, 1])
            fov_x_deg = float(2.0 * np.degrees(np.arctan(width / (2.0 * fx))))
            fov_y_deg = float(2.0 * np.degrees(np.arctan(height / (2.0 * fy))))
            tan_half_x = width / (2.0 * fx)
            tan_half_y = height / (2.0 * fy)
            fov_diag = float(2.0 * np.degrees(np.arctan(np.sqrt(tan_half_x**2 + tan_half_y**2))))

            diagnostics = self._evaluate_calibration_health(
                camera_matrix, dist_coeffs, mean_error, (width, height),
                sample_health, coverage_grid, prev_cal,
            )

            calibration_dict = {
                "meta": {
                    "board_type": "charuco",
                    "board_config": {
                        "squares_x": self.squares_x,
                        "squares_y": self.squares_y,
                        "square_length_m": self.square_length,
                        "marker_length_m": self.marker_length,
                        "legacy_pattern": self.legacy_pattern,
                    },
                    "resolution": {"width": width, "height": height},
                    "camera_matrix": camera_matrix.tolist(),
                    "dist_coeffs": dist_coeffs.tolist(),
                    "reprojection_error": mean_error,
                    "num_samples": len(obj_points),
                    "calibration_time": elapsed,
                    "fov": {
                        "horizontal": fov_x_deg,
                        "vertical": fov_y_deg,
                        "diagonal": fov_diag,
                    },
                    "diagnostics": diagnostics,
                }
            }
            calibration_dict["meta"]["distortion_viz_base64"] = self._generate_distortion_visualization(
                camera_matrix, dist_coeffs, (width, height)
            )

            with self.lock:
                self.calibration_result = calibration_dict
                self.last_health_report = diagnostics
                self.is_calibrating = False
                self.progress = 1.0
                self.status_message = (
                    f"Calibration complete ({mean_error:.4f} px, Health: {diagnostics['status']}). "
                    "Scroll down and press Save Calibration"
                )

            if self._on_health_update is not None:
                try:
                    self._on_health_update(diagnostics)
                except Exception:
                    pass

            logger.Log(
                f"ChArUco calibration complete in {elapsed:.2f}s: err={mean_error:.4f}px, "
                f"FOV={fov_x_deg:.1f}x{fov_y_deg:.1f} deg, health={diagnostics['status']}"
            )
            return calibration_dict

        except Exception as e:
            logger.Error(f"ChArUco calibration error: {e}")
            with self.lock:
                self.is_calibrating = False
                self.progress = 0.0
                self.status_message = f"Error: {e}"
            return None

    def save_calibration(self) -> bool:
        if self.calibration_result is None:
            logger.Error("No calibration result to save")
            return False
        try:
            store_calibration(self.camera_id, self.calibration_result)
            res = self.calibration_result["meta"]["resolution"]
            with self.lock:
                self.status_message = f"Saved {res['width']}x{res['height']} ChArUco calibration"
            logger.Log(f"ChArUco calibration saved for camera {self.camera_id}")
            return True
        except Exception as e:
            logger.Error(f"Failed to save calibration: {e}")
            return False

    def reset(self) -> None:
        with self.lock:
            self.all_corners.clear()
            self.all_ids.clear()
            self.sample_health.clear()
            self.sample_dir = None
            self.image_size = None
            self.calibration_result = None
            self.status_message = "Ready to calibrate (ChArUco 11x8 4x4)"
            self.progress = 0.0
            self.coverage_grid = np.zeros((3, 3), dtype=int)
            self.last_capture_time = 0.0
            self.last_good_frame = None

    def get_health_report(self) -> Dict:
        with self.lock:
            return dict(self.last_health_report)

    def get_status(self) -> Dict:
        with self.lock:
            return {
                "num_samples": len(self.all_corners),
                "target_samples": self.TARGET_SAMPLES,
                "min_samples": self.MIN_SAMPLES,
                "progress": self.progress,
                "status_message": self.status_message,
                "is_calibrating": self.is_calibrating,
                "is_ready": len(self.all_corners) >= self.MIN_SAMPLES,
                "calibration_complete": self.calibration_result is not None,
                "sample_dir": self.sample_dir,
                "health": dict(self.last_health_report),
            }

    def get_last_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.last_frame_annotated

    def _generate_distortion_visualization(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        image_size: Tuple[int, int],
    ) -> str:
        width, height = image_size
        grid_size = 20
        y_coords = np.linspace(0, height - 1, grid_size)
        x_coords = np.linspace(0, width - 1, grid_size)
        xx, yy = np.meshgrid(x_coords, y_coords)
        points = np.stack([xx.ravel(), yy.ravel()], axis=-1).reshape(-1, 1, 2).astype(np.float32)
        undistorted = cv2.undistortPoints(points, camera_matrix, dist_coeffs, P=camera_matrix)
        distortion = np.linalg.norm(undistorted - points, axis=2).reshape(grid_size, grid_size)
        fig_h, fig_w = 300, int(300 * width / height)
        viz = np.zeros((fig_h, fig_w, 3), dtype=np.uint8)
        max_d = float(np.max(distortion))
        distortion_norm = (distortion / max_d) if max_d > 0 else distortion
        cell_h, cell_w = fig_h // grid_size, fig_w // grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                d = float(distortion_norm[i, j])
                if d < 0.33:
                    r, g, b = 0, int(255 * d / 0.33), int(255 * (1 - d / 0.33))
                elif d < 0.66:
                    r, g, b = int(255 * (d - 0.33) / 0.33), 255, 0
                else:
                    r, g, b = 255, int(255 * (1 - (d - 0.66) / 0.34)), 0
                cv2.rectangle(viz, (j * cell_w, i * cell_h), ((j + 1) * cell_w, (i + 1) * cell_h), (b, g, r), -1)
        for i in range(grid_size + 1):
            cv2.line(viz, (0, i * cell_h), (fig_w, i * cell_h), (80, 80, 80), 1)
        for j in range(grid_size + 1):
            cv2.line(viz, (j * cell_w, 0), (j * cell_w, fig_h), (80, 80, 80), 1)
        cv2.putText(viz, "Lens Distortion Heatmap", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(viz, f"Max distortion: {max_d:.2f} px", (10, fig_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        _, buffer = cv2.imencode(".png", viz)
        return base64.b64encode(buffer).decode("utf-8")
