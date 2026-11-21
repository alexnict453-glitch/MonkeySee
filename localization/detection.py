import ctypes
import os
import platform
import re
import threading
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np
from cv2.typing import MatLike

import util.config as config_mod
from localization.undistort import calibration_for, has_opencv_calibration
from util.config import ConfigCategory, Config
from util.logger import Logger

logger = Logger("Detection")

pref_category = ConfigCategory("Detection")

min_thresh_win = pref_category.getIntConfig("min_thresh_win", 3)
max_thresh_win = pref_category.getIntConfig("max_thresh_win", 23)
thresh_step = pref_category.getIntConfig("thresh_step", 17)
persp_px_per_cell = pref_category.getIntConfig("persp_px_per_cell", 10)

corner_refine = pref_category.getIntConfig("corner_refine", 1)
corner_refine_win = pref_category.getIntConfig("corner_refine_win", 5)
corner_refine_iters = pref_category.getIntConfig("corner_refine_iters", 30)
corner_refine_eps = pref_category.getFloatConfig("corner_refine_eps", 0.01)

DEFAULT_NAPA_LIB_PATH = "/usr/local/lib/libnapa_v061_a76_neon2_pgo.so"
_FALLBACK_NAPA_LIB_PATH = "/home/orangepi/opi5plus_bench/napa_libs/libnapa_v061_a76_neon2_pgo.so"
_NAPA_LENS_NONE = 0
_NAPA_LENS_BROWN_CONRADY = 1
_REORDER_FLAT = [4, 5, 6, 7, 0, 1, 2, 3]
_DEFAULT_MAX_TAGS = 128

_napa_lib_lock = threading.Lock()
_napa_lib_loaded = False
_napa_lib_handle = None
_napa_cache_lock = threading.Lock()
_napa_cache: Dict[Tuple, "NativeNapaDetector"] = {}
_py_napa_cache: Dict[Tuple, object] = {}


class _NapaParams(ctypes.Structure):
    _fields_ = [
        ("min_tag_px", ctypes.c_int),
        ("max_hamming", ctypes.c_int),
        ("min_contrast", ctypes.c_int),
        ("min_border_cells", ctypes.c_int),
        ("sharpening_x100", ctypes.c_int),
        ("split_touching", ctypes.c_int),
    ]


class _NapaDetection(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_int),
        ("hamming", ctypes.c_int),
        ("decision_margin", ctypes.c_float),
        ("corners", ctypes.c_double * 8),
    ]


_NAPA_DET_DTYPE = np.dtype(
    {
        "names": ["id", "hamming", "decision_margin", "corners"],
        "formats": [np.int32, np.int32, np.float32, (np.float64, 8)],
        "offsets": [
            _NapaDetection.id.offset,
            _NapaDetection.hamming.offset,
            _NapaDetection.decision_margin.offset,
            _NapaDetection.corners.offset,
        ],
        "itemsize": ctypes.sizeof(_NapaDetection),
    }
)


class _NativeTag:
    __slots__ = ("tag_id", "hamming", "decision_margin", "corners")

    def __init__(self, tag_id: int, hamming: int, decision_margin: float, corners: np.ndarray):
        self.tag_id = tag_id
        self.hamming = hamming
        self.decision_margin = decision_margin
        self.corners = corners


def _load_napa_c_library():
    """Load libnapa_v061_a76_neon2_pgo.so with exact C ABI bindings, returning None if absent."""
    global _napa_lib_loaded, _napa_lib_handle

    custom_path = os.environ.get("MONKEYSEE_NAPA_LIB") or os.environ.get("NAPA_CORE_LIBRARY")
    if custom_path is None:
        with _napa_lib_lock:
            if _napa_lib_loaded:
                return _napa_lib_handle

    if platform.system() != "Linux" and custom_path is None:
        with _napa_lib_lock:
            _napa_lib_loaded = True
            _napa_lib_handle = None
        return None

    candidates = (
        [custom_path]
        if custom_path
        else [DEFAULT_NAPA_LIB_PATH, _FALLBACK_NAPA_LIB_PATH, "libnapa_v061_a76_neon2_pgo.so"]
    )
    loaded = None
    for path in candidates:
        if not path:
            continue
        if os.path.isabs(path) and not os.path.exists(path):
            continue
        try:
            lib = ctypes.CDLL(path)
            lib.napa_create_family.argtypes = [ctypes.c_char_p]
            lib.napa_create_family.restype = ctypes.c_void_p
            lib.napa_destroy.argtypes = [ctypes.c_void_p]
            lib.napa_destroy.restype = None
            lib.napa_set_threads.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.napa_set_threads.restype = ctypes.c_int
            if hasattr(lib, "napa_set_lens"):
                lib.napa_set_lens.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.c_int,
                ]
                lib.napa_set_lens.restype = ctypes.c_int
            lib.napa_detect.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(_NapaParams),
                ctypes.POINTER(_NapaDetection),
                ctypes.c_int,
            ]
            lib.napa_detect.restype = ctypes.c_int
            if hasattr(lib, "napa_detect_shadows_configured"):
                lib.napa_detect_shadows_configured.argtypes = [*lib.napa_detect.argtypes, ctypes.c_int]
                lib.napa_detect_shadows_configured.restype = ctypes.c_int
            if hasattr(lib, "napa_set_regrow_small_tags"):
                lib.napa_set_regrow_small_tags.argtypes = [ctypes.c_void_p, ctypes.c_int]
                lib.napa_set_regrow_small_tags.restype = ctypes.c_int
            if hasattr(lib, "napa_set_tag_shape"):
                lib.napa_set_tag_shape.argtypes = [ctypes.c_void_p, ctypes.c_int]
                lib.napa_set_tag_shape.restype = ctypes.c_int
            loaded = lib
            logger.Log(f"Loaded native NAPA C library: {path}")
            break
        except (OSError, AttributeError) as exc:
            logger.Warn(f"Failed to load native NAPA C library {path}: {exc}")

    if custom_path is None:
        with _napa_lib_lock:
            _napa_lib_loaded = True
            _napa_lib_handle = loaded
    return loaded


def _infer_camera_id(section: Optional[str]) -> Optional[int]:
    if not section:
        return None
    match = re.search(r"(\d+)$", section)
    return int(match.group(1)) if match else None


def _read_optional_config(section: Optional[str], keys: Tuple[str, ...], default, cast):
    """Read an optional config value from loaded_config without mutating config.json."""
    for cat in (section, "Detection"):
        if not cat:
            continue
        sec_dict = config_mod.loaded_config.get(cat)
        if not isinstance(sec_dict, dict):
            continue
        for key in keys:
            if key not in sec_dict:
                continue
            entry = sec_dict[key]
            raw = entry.get("value", entry) if isinstance(entry, dict) else entry
            try:
                return cast(raw)
            except (TypeError, ValueError):
                logger.Warn(f"Invalid config value for {cat}.{key}: {raw!r}, using {default}")
                return default
    return default


def _sanitize_shadow_min_tag_px(shadow_min_tag_px: int, min_tag_px: int) -> int:
    px = int(shadow_min_tag_px)
    if px <= 0:
        return 0
    return min(30000, max(int(min_tag_px), px))


def _as_bool(raw) -> bool:
    if isinstance(raw, str):
        val = raw.strip().lower()
        if val in ("1", "true", "yes", "on"):
            return True
        if val in ("0", "false", "no", "off"):
            return False
    return bool(int(raw))


def _resolve_napa_settings(section: Optional[str], default_min_tag_px: int = 0) -> dict:
    min_tag_px = _read_optional_config(section, ("napa_min_tag_px",), default_min_tag_px, int)
    threads = max(1, min(16, _read_optional_config(section, ("napa_threads",), 2, int)))
    recover_shadows = _read_optional_config(section, ("napa_recover_shadows", "recover_shadows"), True, _as_bool)
    raw_shadow_px = _read_optional_config(section, ("napa_shadow_min_tag_px", "shadow_min_tag_px"), 64, int)
    shadow_min_tag_px = _sanitize_shadow_min_tag_px(raw_shadow_px, max(1, min_tag_px))
    return {
        "min_tag_px": min_tag_px,
        "max_hamming": _read_optional_config(section, ("napa_max_hamming",), 2, int),
        "min_contrast": _read_optional_config(section, ("napa_min_contrast",), 12, int),
        "min_border_cells": _read_optional_config(section, ("napa_min_border_cells",), 22, int),
        "sharpening": _read_optional_config(section, ("napa_sharpening",), 0.5, float),
        "split_touching": _read_optional_config(section, ("napa_split_touching",), 0, int),
        "threads": threads,
        "recover_shadows": recover_shadows,
        "shadow_min_tag_px": shadow_min_tag_px,
        "apply_lens": bool(_read_optional_config(section, ("napa_apply_lens",), 1, int)),
    }


class NativeNapaDetector:
    """Thread-safe per-detector wrapper around libnapa_v061_a76_neon2_pgo.so."""

    def __init__(
        self,
        lib,
        min_tag_px: int = 16,
        max_hamming: int = 2,
        min_contrast: int = 12,
        min_border_cells: int = 22,
        sharpening: float = 0.5,
        split_touching: int = 0,
        threads: int = 2,
        recover_shadows: bool = True,
        shadow_min_tag_px: int = 64,
        camera_id: Optional[int] = None,
        apply_lens: bool = True,
        max_tags: int = _DEFAULT_MAX_TAGS,
    ):
        self._lib = lib
        self._lock = threading.Lock()
        self._det = lib.napa_create_family(b"tag36h11")
        if not self._det:
            raise RuntimeError("napa_create_family('tag36h11') returned NULL")

        self._has_lens = hasattr(lib, "napa_set_lens")
        self._has_shadows = hasattr(lib, "napa_detect_shadows_configured")
        self._camera_id = camera_id
        self._apply_lens = apply_lens
        self._lens_size: Optional[Tuple[int, int]] = None
        self._lens_k_ref = None
        self._lens_d_ref = None

        self._max_tags = max(1, int(max_tags))
        self._out_buf = (_NapaDetection * self._max_tags)()
        self._rows_view = np.frombuffer(self._out_buf, dtype=_NAPA_DET_DTYPE)

        self._recover_shadows = bool(recover_shadows) and self._has_shadows
        self._shadow_min_tag_px = _sanitize_shadow_min_tag_px(shadow_min_tag_px, min_tag_px)
        self._params = _NapaParams(
            min_tag_px=int(min_tag_px),
            max_hamming=int(max_hamming),
            min_contrast=int(min_contrast),
            min_border_cells=int(min_border_cells),
            sharpening_x100=int(round(float(sharpening) * 100.0)),
            split_touching=int(split_touching),
        )
        lib.napa_set_threads(self._det, max(1, min(16, int(threads))))

    def set_camera_id(self, camera_id: Optional[int]) -> None:
        with self._lock:
            if camera_id != self._camera_id:
                self._camera_id = camera_id
                self._lens_size = None
                self._lens_k_ref = None
                self._lens_d_ref = None

    def _sync_lens_locked(self, width: int, height: int) -> None:
        if not self._has_lens:
            return
        if not self._apply_lens or self._camera_id is None:
            if self._lens_size is not None or self._lens_k_ref is not None:
                self._lib.napa_set_lens(self._det, _NAPA_LENS_NONE, None, None, 0)
                self._lens_size = (width, height)
                self._lens_k_ref = None
                self._lens_d_ref = None
            return

        if has_opencv_calibration(self._camera_id, width, height):
            cal = calibration_for(self._camera_id, width, height)
            if cal is not None:
                cam_mat, dist_coeffs = cal
                if (width, height) != self._lens_size or cam_mat is not self._lens_k_ref or dist_coeffs is not self._lens_d_ref:
                    intrinsics = (ctypes.c_double * 5)(
                        float(cam_mat[0, 0]),
                        float(cam_mat[1, 1]),
                        float(cam_mat[0, 2]),
                        float(cam_mat[1, 2]),
                        0.0,
                    )
                    dflat = np.asarray(dist_coeffs, dtype=np.float64).ravel()
                    dist = (ctypes.c_double * 5)(*[float(dflat[i]) if i < dflat.size else 0.0 for i in range(5)])
                    rc = self._lib.napa_set_lens(self._det, _NAPA_LENS_BROWN_CONRADY, intrinsics, dist, 5)
                    if rc != 0:
                        logger.Warn(f"Camera {self._camera_id}: napa_set_lens failed (rc={rc}), disabling lens model")
                        self._lib.napa_set_lens(self._det, _NAPA_LENS_NONE, None, None, 0)
                    self._lens_size = (width, height)
                    self._lens_k_ref = cam_mat
                    self._lens_d_ref = dist_coeffs
                return

        if (width, height) != self._lens_size or self._lens_k_ref is not None:
            self._lib.napa_set_lens(self._det, _NAPA_LENS_NONE, None, None, 0)
            self._lens_size = (width, height)
            self._lens_k_ref = None
            self._lens_d_ref = None

    def _prepare_gray(self, image: MatLike) -> np.ndarray:
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.dtype != np.uint8 or image.strides[1] != 1 or image.strides[0] < image.shape[1]:
            image = np.ascontiguousarray(image, dtype=np.uint8)
        return image

    def _run_detect_locked(self, gray: np.ndarray) -> int:
        h, w = gray.shape[:2]
        stride = int(gray.strides[0])
        self._sync_lens_locked(w, h)
        if self._recover_shadows:
            cnt = self._lib.napa_detect_shadows_configured(
                self._det,
                gray.ctypes.data,
                w,
                h,
                stride,
                ctypes.byref(self._params),
                self._out_buf,
                self._max_tags,
                self._shadow_min_tag_px,
            )
            if cnt < 0:
                logger.Warn(f"napa_detect_shadows_configured returned {cnt}, retrying without shadow pass")
                cnt = self._lib.napa_detect(
                    self._det,
                    gray.ctypes.data,
                    w,
                    h,
                    stride,
                    ctypes.byref(self._params),
                    self._out_buf,
                    self._max_tags,
                )
        else:
            cnt = self._lib.napa_detect(
                self._det,
                gray.ctypes.data,
                w,
                h,
                stride,
                ctypes.byref(self._params),
                self._out_buf,
                self._max_tags,
            )
        if cnt < 0:
            logger.Warn(f"napa_detect failed with code {cnt}")
            return 0
        return cnt

    def detect_raw(self, image: MatLike):
        """Return (reordered_corners_list, ids_int32_or_None) directly without per-tag wrappers."""
        if image is None or image.size == 0:
            return [], None
        gray = self._prepare_gray(image)
        with self._lock:
            if not self._det:
                return [], None
            cnt = self._run_detect_locked(gray)
            if cnt <= 0:
                return [], None
            sub = self._rows_view[:cnt]
            pts = sub["corners"][:, _REORDER_FLAT].reshape(cnt, 1, 4, 2)
            corners = [pts[i] for i in range(cnt)]
            ids = sub["id"].astype(np.int32, copy=True)
            return corners, ids

    def detect(self, image: MatLike):
        """Return list of tag objects matching napa_vision.NapaDetector.detect()."""
        if image is None or image.size == 0:
            return []
        gray = self._prepare_gray(image)
        with self._lock:
            if not self._det:
                return []
            cnt = self._run_detect_locked(gray)
            if cnt <= 0:
                return []
            sub = self._rows_view[:cnt]
            tags = []
            for i in range(cnt):
                row = sub[i]
                tags.append(
                    _NativeTag(
                        tag_id=int(row["id"]),
                        hamming=int(row["hamming"]),
                        decision_margin=float(row["decision_margin"]),
                        corners=row["corners"].reshape(4, 2).copy(),
                    )
                )
            return tags

    def close(self) -> None:
        with self._lock:
            det = getattr(self, "_det", None)
            lib = getattr(self, "_lib", None)
            self._det = None
            if det is not None and lib is not None:
                try:
                    lib.napa_destroy(det)
                except Exception:
                    pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def GET_THRESH_STEP() -> int:
    return thresh_step.valueFloat()


def SET_THRESH_STEP(value: int):
    thresh_step.setInt(value)


def GET_THRESH_WIN() -> int:
    return max_thresh_win.valueFloat()


def SET_THRESH_WIN(value: int):
    max_thresh_win.setInt(value)


dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

parameters = cv2.aruco.DetectorParameters()

parameters.adaptiveThreshWinSizeMin = min_thresh_win.valueInt()
parameters.adaptiveThreshWinSizeMax = max_thresh_win.valueInt()
parameters.adaptiveThreshWinSizeStep = thresh_step.valueInt()
parameters.perspectiveRemovePixelPerCell = persp_px_per_cell.valueInt()

_detector = cv2.aruco.ArucoDetector(dictionary, parameters) if hasattr(cv2.aruco, "ArucoDetector") else None


def GET_CORNER_REFINE() -> int:
    return corner_refine.valueInt()


def SET_CORNER_REFINE(value: int):
    corner_refine.setInt(value)


def _refine_corners(corners, refine_image: MatLike):

    if refine_image is None or not corners:
        return corners

    if refine_image.ndim == 3:
        refine_image = cv2.cvtColor(refine_image, cv2.COLOR_BGR2GRAY)

    base_win = max(2, corner_refine_win.valueInt())
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        max(1, corner_refine_iters.valueInt()),
        max(1e-4, corner_refine_eps.valueFloat()),
    )

    refined = []
    for quad in corners:
        pts = np.asarray(quad, dtype=np.float32).reshape(-1, 1, 2)

        sides = [
            float(np.linalg.norm(pts[i, 0] - pts[(i + 1) % 4, 0]))
            for i in range(4)
        ]
        win = int(min(base_win, max(2, min(sides) / 8.0)))
        if win < 2:
            refined.append(quad)
            continue

        try:
            out = cv2.cornerSubPix(refine_image, pts.copy(), (win, win), (-1, -1), criteria)
            refined.append(out.reshape(np.asarray(quad).shape))
        except cv2.error as e:
            logger.Warn(f"cornerSubPix failed, using unrefined corners: {e}")
            refined.append(quad)

    return refined


def DETECT_TAGS(image: MatLike, refine_image: MatLike = None):
    global _detector
    if _detector is not None:
        corners, IDs, _ = _detector.detectMarkers(image)
    else:
        corners, IDs, _ = cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)

    if IDs is None:
        return corners, None

    if corner_refine.valueInt() > 0:
        corners = _refine_corners(corners, refine_image if refine_image is not None else image)

    corners = [np.roll(q, 2, axis=1) for q in corners]

    return corners, IDs.flatten()


def _build_native_napa(section: Optional[str], camera_id: Optional[int], cfg: dict) -> Optional[NativeNapaDetector]:
    lib = _load_napa_c_library()
    if lib is None:
        return None
    cam_id = camera_id if camera_id is not None else _infer_camera_id(section)
    try:
        return NativeNapaDetector(
            lib,
            min_tag_px=cfg["min_tag_px"],
            max_hamming=cfg["max_hamming"],
            min_contrast=cfg["min_contrast"],
            min_border_cells=cfg["min_border_cells"],
            sharpening=cfg["sharpening"],
            split_touching=cfg["split_touching"],
            threads=cfg["threads"],
            recover_shadows=cfg["recover_shadows"],
            shadow_min_tag_px=cfg["shadow_min_tag_px"],
            camera_id=cam_id,
            apply_lens=cfg["apply_lens"],
        )
    except Exception as exc:
        logger.Warn(f"{section}: NativeNapaDetector init failed ({exc}), falling back")
        return None


def MAKE_APRILTAG_DETECTOR(section: str, camera_id: Optional[int] = None) -> Optional[Callable]:
    settings = ConfigCategory(section)
    napa_min_tag_px = settings.getIntConfig("napa_min_tag_px", 0).valueInt()
    if napa_min_tag_px > 0:
        cfg = _resolve_napa_settings(section, default_min_tag_px=napa_min_tag_px)
        cfg["min_tag_px"] = napa_min_tag_px
        cfg["shadow_min_tag_px"] = _sanitize_shadow_min_tag_px(cfg["shadow_min_tag_px"], napa_min_tag_px)
        cam_id = camera_id if camera_id is not None else _infer_camera_id(section)
        native = _build_native_napa(section, cam_id, cfg)
        if native is not None:
            logger.Log(
                f"{section}: Native NEON2/PGO NAPA detector, min_tag_px={napa_min_tag_px}, "
                f"threads={cfg['threads']}, shadows={int(cfg['recover_shadows'])}, "
                f"shadow_min_tag_px={cfg['shadow_min_tag_px']}, camera_id={cam_id}"
            )
            return native.detect_raw

        try:
            from napa_vision import NapaDetector
        except ImportError:
            logger.Warn(f"{section}: napa_vision not installed, trying robotpy_apriltag")
        else:
            napa = NapaDetector(
                min_tag_px=napa_min_tag_px,
                max_hamming=cfg["max_hamming"],
                min_contrast=cfg["min_contrast"],
                sharpening=cfg["sharpening"],
            )
            logger.Log(f"{section}: NAPA detector, min_tag_px {napa_min_tag_px}")
            return lambda image: DETECT_NAPA(napa, image)

    decimate = settings.getIntConfig("apriltag_decimate", 0).valueInt()
    if decimate <= 0:
        return None
    try:
        import robotpy_apriltag
    except ImportError:
        logger.Warn(f"{section}: robotpy_apriltag not installed, using ArUco")
        return None
    detector = robotpy_apriltag.AprilTagDetector()
    detector.addFamily("tag36h11")
    config = detector.getConfig()
    config.quadDecimate = float(decimate)
    config.numThreads = 1
    detector.setConfig(config)
    quad = detector.getQuadThresholdParameters()
    quad.minClusterPixels = 5
    detector.setQuadThresholdParameters(quad)
    logger.Log(f"{section}: AprilTag detector, decimate {decimate}")
    return lambda image: DETECT_APRILTAGS(detector, image)


def _get_cached_napa_detector(config_key: Optional[str], camera_id: Optional[int]):
    section = config_key if config_key is not None else (f"AprilTag{camera_id}" if camera_id is not None else "Detection")
    cam_id = camera_id if camera_id is not None else _infer_camera_id(section)
    cfg = _resolve_napa_settings(section, default_min_tag_px=16)
    if cfg["min_tag_px"] <= 0:
        cfg["min_tag_px"] = 16
        cfg["shadow_min_tag_px"] = _sanitize_shadow_min_tag_px(cfg["shadow_min_tag_px"], 16)
    key = (
        section,
        cam_id,
        cfg["min_tag_px"],
        cfg["max_hamming"],
        cfg["min_contrast"],
        cfg["min_border_cells"],
        cfg["sharpening"],
        cfg["split_touching"],
        cfg["threads"],
        cfg["recover_shadows"],
        cfg["shadow_min_tag_px"],
        cfg["apply_lens"],
    )
    with _napa_cache_lock:
        det = _napa_cache.get(key)
        if det is not None:
            return det, True
        native = _build_native_napa(section, cam_id, cfg)
        if native is not None:
            _napa_cache[key] = native
            return native, True
        py_det = _py_napa_cache.get(key)
        if py_det is None:
            from napa_vision import NapaDetector
            py_det = NapaDetector(
                min_tag_px=cfg["min_tag_px"],
                max_hamming=cfg["max_hamming"],
                min_contrast=cfg["min_contrast"],
                sharpening=cfg["sharpening"],
            )
            _py_napa_cache[key] = py_det
        return py_det, False


def DETECT_NAPA(detector_or_image, image: Optional[MatLike] = None, config_key: Optional[str] = None, camera_id: Optional[int] = None):
    """Detect tags with NAPA. Supports both DETECT_NAPA(detector, image) and DETECT_NAPA(image, config_key=..., camera_id=...)."""
    if isinstance(detector_or_image, np.ndarray) and (image is None or isinstance(image, str)):
        if isinstance(image, str):
            if isinstance(config_key, int) and camera_id is None:
                camera_id = config_key
            config_key = image
        det, is_native = _get_cached_napa_detector(config_key, camera_id)
        if is_native:
            return det.detect_raw(detector_or_image)
        tags = det.detect(detector_or_image)
        if not tags:
            return [], None
        corners = [np.asarray(t.corners, dtype=np.float64)[[2, 3, 0, 1]].reshape(1, 4, 2) for t in tags]
        return corners, np.array([t.tag_id for t in tags], dtype=np.int32)

    detector = detector_or_image
    if camera_id is not None and hasattr(detector, "set_camera_id"):
        detector.set_camera_id(camera_id)
    if hasattr(detector, "detect_raw"):
        return detector.detect_raw(image)
    tags = detector.detect(image)
    if not tags:
        return [], None
    corners = [np.asarray(t.corners, dtype=np.float64)[[2, 3, 0, 1]].reshape(1, 4, 2) for t in tags]
    return corners, np.array([t.tag_id for t in tags], dtype=np.int32)


def DETECT_APRILTAGS(detector, image: MatLike):
    tags = [t for t in detector.detect(image) if t.getHamming() == 0]
    if not tags:
        return [], None
    corners = [
        np.array([[[t.getCorner(i).x - 0.5, t.getCorner(i).y - 0.5] for i in (3, 2, 1, 0)]], dtype=np.float32)
        for t in tags
    ]
    return corners, np.array([t.getId() for t in tags], dtype=np.int32)


def ANNOTATE_TAGS(image: MatLike, corners, IDs) -> MatLike:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    if IDs is not None and len(IDs) > 0:
        corners_f32 = [np.asarray(c, dtype=np.float32).reshape(1, 4, 2) for c in corners]
        image = cv2.aruco.drawDetectedMarkers(image, corners_f32, np.asarray(IDs, dtype=np.int32))
    return image
