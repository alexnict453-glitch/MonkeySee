from camera.camerareader import CameraReader
import cv2
import time
import inspect
import localization.detection
import localization.apriltag_solution
import localization.undistort
import pipeline.ntables
from time import monotonic_ns
from typing import List, Dict, Optional, Set, Tuple
import platform
import json
import os
import struct
from datetime import datetime
import threading
from threading import Event, Thread, Lock
from camera.preprocess import PROCESS_FRAME
from util.logger import Logger

logger = Logger("VisionManager")

if hasattr(cv2, "setNumThreads"):
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

CAPTURES_DIR = "captures"


def _detect_big_little_cpus() -> Tuple[Optional[Set[int]], Optional[Set[int]]]:
    """Detect heterogeneous big/little CPU sets on Linux (e.g., RK3588 A76 vs A55)."""
    if platform.system() != "Linux" or not hasattr(os, "sched_getaffinity"):
        return None, None
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except OSError:
        return None, None
    if len(allowed) < 2:
        return None, None

    scores: Dict[int, int] = {}
    for cpu in allowed:
        val = None
        for rel in ("cpu_capacity", "cpufreq/cpuinfo_max_freq", "cpufreq/scaling_max_freq"):
            path = f"/sys/devices/system/cpu/cpu{cpu}/{rel}"
            try:
                with open(path, "r") as f:
                    val = int(f.read().strip())
                break
            except (OSError, ValueError):
                continue
        if val is None:
            return None, None
        scores[cpu] = val

    min_score = min(scores.values())
    max_score = max(scores.values())
    if max_score <= min_score:
        return None, None

    big = {cpu for cpu, score in scores.items() if score > min_score}
    little = {cpu for cpu, score in scores.items() if score == min_score}
    return (big or None), (little or None)


_BIG_CPUS, _LITTLE_CPUS = _detect_big_little_cpus()


def _pin_current_thread(cpus: Optional[Set[int]]) -> bool:
    if not cpus or platform.system() != "Linux" or not hasattr(os, "sched_setaffinity"):
        return False
    try:
        os.sched_setaffinity(0, cpus)
        return True
    except OSError:
        return False


def pin_current_thread_big_cores() -> bool:
    """Pin calling Linux thread to high-performance cores (no-op elsewhere)."""
    return _pin_current_thread(_BIG_CPUS)


def pin_current_thread_little_cores() -> bool:
    """Pin calling Linux thread to efficiency cores (no-op elsewhere)."""
    return _pin_current_thread(_LITTLE_CPUS)


def _set_current_thread_nice(nice: int) -> bool:
    """Set Linux scheduling priority for the calling kernel thread (no-op elsewhere)."""
    if platform.system() != "Linux" or not hasattr(os, "setpriority"):
        return False
    tid = threading.get_native_id() if hasattr(threading, "get_native_id") else 0
    if tid <= 0:
        return False
    try:
        os.setpriority(os.PRIO_PROCESS, tid, int(nice))
        return True
    except OSError:
        return False


def _pipeline_detector_threads(cam_pipeline: "CameraPipeline") -> int:
    det_obj = getattr(getattr(cam_pipeline, "_apriltag", None), "__self__", None)
    if det_obj is not None:
        return int(getattr(det_obj, "threads", 1))
    return int(getattr(cam_pipeline, "_detector_threads", 1))


def _assign_pipeline_affinities(pipelines: Dict[int, "CameraPipeline"]) -> Dict[int, Optional[Set[int]]]:
    if not _BIG_CPUS:
        return {cid: None for cid in pipelines}
    big_list = sorted(_BIG_CPUS)
    active = [p for p in pipelines.values() if p.enabled]
    if 1 < len(active) <= len(big_list) and all(_pipeline_detector_threads(p) <= 1 for p in active):
        affinities: Dict[int, Optional[Set[int]]] = {cid: _BIG_CPUS for cid in pipelines}
        for idx, cam_pipeline in enumerate(active):
            affinities[cam_pipeline.camera_id] = {big_list[idx]}
        return affinities
    return {cid: _BIG_CPUS for cid in pipelines}


def _supports_kwarg(fn, name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def _open_writer(path: str, size, is_color: bool, raw: bool):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    if raw:
        writer = cv2.VideoWriter(path, cv2.CAP_FFMPEG, fourcc, 30.0, size, [cv2.VIDEOWRITER_PROP_RAW_VIDEO, 1])
        if writer.isOpened():
            return writer, True
    return cv2.VideoWriter(path, fourcc, 30.0, size, isColor=is_color), False


def _set_avi_fps(path: str, fps: float) -> None:
    with open(path, "r+b") as f:
        head = f.read(8192)
        avih, strh = head.find(b"avih"), head.find(b"strh")
        if avih < 0 or strh < 0:
            return
        f.seek(avih + 8)
        f.write(struct.pack("<I", round(1e6 / fps)))
        f.seek(strh + 28)
        f.write(struct.pack("<II", 1000, round(fps * 1000)))


class CameraPipeline:
    def __init__(self, camera_config: dict):
        self.camera_id = camera_config["id"]
        self.camera_name = camera_config["name"]
        self.device_id = camera_config["device_id"]
        self.pipeline_type = camera_config["pipeline"]
        self.enabled = camera_config["enabled"]

        use_high_res = self.pipeline_type == "apriltag"

        self.frame_count = 0
        self._rate_count = 0
        self.start_time = time.monotonic()
        self.fps_log_time = self.start_time
        self.fps_log_count = 0
        self.framerate = 0.0
        self.backend_fps = 0.0
        self.processing_latency = 0.0
        self.capture_latency = 0.0
        self._last_processed_at = 0.0
        self.frame: cv2.typing.MatLike = None
        self._tags = None
        self._frame_seq = 0
        self._overlay_seq = -1
        self._overlay = None
        self._preview_lock = Lock()
        self._preview_seq = -1
        self._preview_frame = None
        self._preview_cache: Dict[Tuple[int, int], bytes] = {}
        self.detections: List = []
        self.frame_num = 0
        self.lock = Lock()
        self.running = True
        self._enabled_event = Event()
        if self.enabled:
            self._enabled_event.set()

        self.focus_mode = False
        self.focus_score = 0.0
        self._optical_health: Optional[Dict] = None
        self._health_lock = Lock()
        self._last_health_pub_at = 0.0
        self._last_published_health_sig = None

        cal_saved = localization.undistort.load_calibration(self.camera_id)
        if isinstance(cal_saved, dict):
            saved_diag = cal_saved.get("meta", {}).get("diagnostics")
            if isinstance(saved_diag, dict):
                self._optical_health = dict(saved_diag)

        self._writer: Optional[cv2.VideoWriter] = None
        self._recording_path: Optional[str] = None
        self._recording_pending = False
        self._last_capture_message = ""

        prev_cpus = None
        if _BIG_CPUS and platform.system() == "Linux" and hasattr(os, "sched_getaffinity"):
            try:
                prev_cpus = os.sched_getaffinity(0)
                os.sched_setaffinity(0, _BIG_CPUS)
            except OSError:
                prev_cpus = None
        try:
            settings_category = f"Camera{self.camera_id}"
            if platform.system() == "Windows" or platform.system() == "Darwin":
                self.cam = CameraReader(self.device_id if self.camera_id == 0 else self.camera_id - 1, use_high_res, settings_category)
            else:
                self.cam = CameraReader(self.device_id, use_high_res, settings_category)

            self._frame_resolution: Tuple[int, int] = self.cam.get_resolution()
            self._cam_accepts_want_jpeg = _supports_kwarg(self.cam.get_frame, "want_jpeg")

            if self.pipeline_type == "apriltag":
                apriltag_section = camera_config.get("apriltag_config", f"AprilTag{self.camera_id}")
                self._apriltag_section = apriltag_section
                localization.apriltag_solution.SET_CAM(self.camera_id, apriltag_section)
                resolver = getattr(localization.detection, "_resolve_napa_settings", None)
                self._detector_threads = int(resolver(apriltag_section).get("threads", 2)) if callable(resolver) else 2
                self._apriltag = None
                self._apriltag_accepts_cam_id = False
                self._apriltag_initialized = False
                self.ntables = pipeline.ntables.AprilTagNTables(self.camera_id)
            else:
                from localization import gamepiece_solution
                from localization import visiony
                gamepiece_solution.SET_CAM(self.camera_id)
                self.ntables = pipeline.ntables.GamePieceNTables(self.camera_id)
                self._visiony = visiony
                self._gamepiece_solution = gamepiece_solution
                self._apriltag_section = None
                self._detector_threads = 1
                self._apriltag = None
                self._apriltag_accepts_cam_id = False
                self._apriltag_initialized = True
        finally:
            if prev_cpus is not None:
                try:
                    os.sched_setaffinity(0, prev_cpus)
                except OSError:
                    pass

    def _ensure_apriltag_ready(self) -> None:
        if self._apriltag_initialized or self.pipeline_type != "apriltag":
            return
        with self.lock:
            if self._apriltag_initialized or self.pipeline_type != "apriltag":
                return
            if _supports_kwarg(localization.detection.MAKE_APRILTAG_DETECTOR, "camera_id"):
                self._apriltag = localization.detection.MAKE_APRILTAG_DETECTOR(
                    self._apriltag_section, camera_id=self.camera_id
                )
            else:
                self._apriltag = localization.detection.MAKE_APRILTAG_DETECTOR(self._apriltag_section)
            self._apriltag_accepts_cam_id = (
                self._apriltag is not None and _supports_kwarg(self._apriltag, "camera_id")
            )
            self._apriltag_initialized = True

    def process_frame(self):
        if not self.running:
            return
        if not self.enabled:
            self._enabled_event.wait(0.25)
            return

        try:
            if not self.running:
                return
            if not self._apriltag_initialized:
                self._ensure_apriltag_ready()
            want_jpeg = self._writer is not None or self._recording_pending
            if self._cam_accepts_want_jpeg:
                frame, capture_ns = self.cam.get_frame(defer_auto_gain=True, want_jpeg=want_jpeg)
            else:
                frame, capture_ns = self.cam.get_frame(defer_auto_gain=True)
            cam_frame, jpeg = frame, self.cam.last_jpeg
            read_ns = monotonic_ns()

            if frame is None:
                self._maybe_publish_health()
                time.sleep(0.1)
                return

            frame_res = (int(frame.shape[1]), int(frame.shape[0]))
            self.frame_num += 1
            self.frame_num %= 500

            if self.focus_mode:
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame
                focus_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                with self.lock:
                    self.focus_score = focus_score
                    self.frame = frame
                    self._tags = None
                    self._frame_seq += 1
                    self._frame_resolution = frame_res
                time.sleep(0.03)
                return

            tags = None
            if self.pipeline_type == "apriltag":
                raw_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                if self._apriltag is not None:
                    frame = raw_gray
                    if self._apriltag_accepts_cam_id:
                        corners, ids = self._apriltag(raw_gray, camera_id=self.camera_id)
                    else:
                        corners, ids = self._apriltag(raw_gray)
                else:
                    frame = PROCESS_FRAME(raw_gray)
                    corners, ids = localization.detection.DETECT_TAGS(frame, refine_image=raw_gray)
                display_frame = frame
                tags = (corners, ids)
                detections = localization.apriltag_solution.CALCULATE_PARTIAL_SOLUTION(
                    self.camera_id, raw_gray, corners, ids
                )
            else:
                display_frame, rawDets = self._visiony.runPipeline(frame)
                detections = self._gamepiece_solution.CALCULATE_PARTIAL_SOLUTION(self.camera_id, frame, rawDets)

            done_ns = monotonic_ns()
            processing_latency = (done_ns - read_ns) / 1e9
            capture_latency = (done_ns - capture_ns) / 1e9

            if self.pipeline_type == "apriltag":
                self.ntables.updateFrameNum(self.frame_num)
            self.ntables.execute(detections, processing_latency, capture_ns)

            gain_frame = raw_gray if self.pipeline_type == "apriltag" else frame
            gain_corners = corners if (self.pipeline_type == "apriltag" and ids is not None) else None
            self.cam.update_auto_gain(gain_frame, gain_corners)
            self._maybe_publish_health()

            now_m = time.monotonic()
            with self.lock:
                if self._recording_pending and self._writer is None:
                    self._start_writer_locked(cam_frame, jpeg is not None)
                if self._writer is not None:
                    data = jpeg if self._writer_raw else cam_frame
                    if data is not None:
                        self._writer.write(data)
                        self._record_frames += 1
                self.frame = display_frame
                self._tags = tags
                self._frame_seq += 1
                self._frame_resolution = frame_res
                self.detections = detections
                self.processing_latency = processing_latency
                self.capture_latency = capture_latency
                self._last_processed_at = now_m

            self.frame_count += 1
            self._rate_count += 1
            self.fps_log_count += 1

            rate_elapsed = now_m - self.start_time
            if self._rate_count >= 30 or (self._rate_count >= 5 and rate_elapsed >= 0.5):
                if rate_elapsed > 0:
                    self.framerate = self._rate_count / rate_elapsed
                self.start_time = now_m
                self._rate_count = 0

            log_elapsed = now_m - self.fps_log_time
            if self.fps_log_count >= 100 or (self.fps_log_count >= 10 and log_elapsed >= 2.0):
                if log_elapsed > 0:
                    self.backend_fps = self.fps_log_count / log_elapsed
                    logger.Log(
                        f"Camera {self.camera_id} ({self.pipeline_type}): {self.backend_fps:.2f} FPS | "
                        f"Latency: {processing_latency*1000:.2f}ms (since capture {capture_latency*1000:.2f}ms) | "
                        f"Detections: {len(detections)}"
                    )
                self.fps_log_time = now_m
                self.fps_log_count = 0

        except Exception as e:
            logger.Warn(f"Error in camera {self.camera_id}: {e}")

    def _capture_stamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    def _start_writer_locked(self, frame, raw: bool = False) -> None:
        height, width = frame.shape[:2]
        path = os.path.join(
            CAPTURES_DIR,
            f"cam{self.camera_id}_{self._capture_stamp()}.avi",
        )
        writer, self._writer_raw = _open_writer(path, (width, height), frame.ndim == 3, raw)
        if not writer.isOpened():
            self._last_capture_message = f"Failed to open video writer: {path}"
            logger.Warn(self._last_capture_message)
            self._recording_pending = False
            return
        self._writer = writer
        self._record_frames = 0
        self._record_start = time.monotonic()
        self._recording_path = path
        self._recording_pending = False
        self._last_capture_message = f"Recording: {path}"
        logger.Log(self._last_capture_message)

    def save_snapshot(self) -> Optional[str]:
        frame = self.get_frame()
        if frame is None:
            self._last_capture_message = "No frame available"
            return None
        os.makedirs(CAPTURES_DIR, exist_ok=True)
        path = os.path.join(
            CAPTURES_DIR,
            f"cam{self.camera_id}_{self._capture_stamp()}.png",
        )
        if not cv2.imwrite(path, frame):
            self._last_capture_message = f"Failed to save snapshot: {path}"
            logger.Warn(self._last_capture_message)
            return None
        self._last_capture_message = f"Saved snapshot: {path}"
        logger.Log(self._last_capture_message)
        return path

    def start_recording(self) -> str:
        with self.lock:
            if self._writer is not None or self._recording_pending:
                return self._last_capture_message or "Already recording"
            os.makedirs(CAPTURES_DIR, exist_ok=True)
            self._recording_pending = True
            self._last_capture_message = "Recording starting..."
            return self._last_capture_message

    def stop_recording(self) -> str:
        with self.lock:
            if self._writer is None and not self._recording_pending:
                self._last_capture_message = "Not recording"
                return self._last_capture_message
            path = self._recording_path
            if self._writer is not None:
                self._writer.release()
                seconds = time.monotonic() - self._record_start
                fps = self._record_frames / seconds if seconds > 0 else 0.0
                if fps > 0:
                    _set_avi_fps(path, fps)
                path = f"{path} ({self._record_frames} frames, {seconds:.0f} s, {fps:.1f} fps)"
            self._writer = None
            self._recording_path = None
            self._recording_pending = False
            self._last_capture_message = f"Saved recording: {path}" if path else "Recording cancelled"
            logger.Log(self._last_capture_message)
            return self._last_capture_message

    def is_recording(self) -> bool:
        with self.lock:
            return self._writer is not None or self._recording_pending

    def get_capture_status(self) -> dict:
        with self.lock:
            return {
                "recording": self._writer is not None or self._recording_pending,
                "path": self._recording_path,
                "message": self._last_capture_message,
            }

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            if self._tags is None:
                return self.frame.copy()
            if self._overlay_seq != self._frame_seq:
                self._overlay = localization.detection.ANNOTATE_TAGS(self.frame, *self._tags)
                self._overlay_seq = self._frame_seq
            return self._overlay

    def update_optical_health(self, report: Dict) -> None:
        with self._health_lock:
            self._optical_health = dict(report) if report else None
        self._maybe_publish_health(force=True)

    def get_health_report(self) -> Dict:
        if not self.enabled:
            return {
                "status": "DISABLED",
                "ok": True,
                "summary": "Camera disabled",
                "issues": [],
                "metrics": {},
            }

        issues: List[Dict] = []
        cam_open = self.cam.is_open() if hasattr(self.cam, "is_open") else False
        last_at = self._last_processed_at
        stale = last_at > 0.0 and (time.monotonic() - last_at) > 1.5
        fail_count = int(getattr(self.cam, "fail_count", 0))
        opened_attempted = bool(getattr(self.cam, "_opened", False))
        if (opened_attempted and not cam_open) or fail_count > 0 or stale:
            issues.append({
                "code": "CAMERA_DISCONNECTED",
                "severity": "ERROR",
                "metric": "fail_count",
                "value": fail_count,
                "threshold": 0,
                "message": "Camera stream offline or not receiving frames",
                "recommendation": "Check USB cable, power, and camera port connection.",
            })

        res_w, res_h = self._frame_resolution
        if not localization.undistort.has_opencv_calibration(self.camera_id, int(res_w), int(res_h)):
            issues.append({
                "code": "NO_CALIBRATION",
                "severity": "WARNING",
                "metric": "calibrated",
                "value": 0,
                "threshold": 1,
                "message": f"No OpenCV calibration for {int(res_w)}x{int(res_h)}",
                "recommendation": "Run ChArUco calibration on the Calibration tab.",
            })

        if bool(getattr(self.cam, "_gain_stuck", False)):
            issues.append({
                "code": "EXPOSURE_STUCK",
                "severity": "WARNING",
                "metric": "gain",
                "value": float(getattr(self.cam, "_live_gain", -1)),
                "threshold": 0,
                "message": str(getattr(self.cam, "_exposure_status", "Auto gain cannot reach target")),
                "recommendation": "Check exposure/gain limits or lighting.",
            })

        if cam_open and not self.focus_mode and last_at > 0.0 and self.frame_count >= 30:
            fps = float(self.get_fps())
            if 0.0 < fps < 60.0:
                issues.append({
                    "code": "LOW_FPS",
                    "severity": "WARNING",
                    "metric": "fps",
                    "value": round(fps, 2),
                    "threshold": 60.0,
                    "message": f"Low stream rate ({fps:.1f} FPS < 60 FPS)",
                    "recommendation": "Check exposure time, USB controller bandwidth, and CPU governors.",
                })

        metrics: Dict = {}
        with self._health_lock:
            opt = dict(self._optical_health) if self._optical_health else None
        if opt is not None:
            metrics = dict(opt.get("metrics") or {})
            for issue in opt.get("issues") or []:
                issues.append(dict(issue))

        if any(i.get("severity") == "ERROR" for i in issues):
            status = "ERROR"
        elif any(i.get("severity") == "WARNING" for i in issues):
            status = "WARNING"
        else:
            status = "OK"

        summary = "Camera & optics OK" if not issues else "; ".join(i.get("message", "") for i in issues[:2])
        return {
            "status": status,
            "ok": status == "OK",
            "summary": summary,
            "issues": issues,
            "metrics": metrics,
        }

    def _maybe_publish_health(self, force: bool = False) -> None:
        now_m = time.monotonic()
        if not force and (now_m - self._last_health_pub_at) < 1.0:
            return
        self._last_health_pub_at = now_m
        if not hasattr(self.ntables, "publish_health"):
            return
        report = self.get_health_report()
        sig = (
            report["status"],
            tuple(i.get("code", "") for i in report["issues"]),
            report.get("metrics", {}).get("blur_score"),
        )
        if force or sig != self._last_published_health_sig:
            self._last_published_health_sig = sig
            try:
                self.ntables.publish_health(report)
            except Exception:
                pass

    def get_frame_seq(self) -> int:
        return self._frame_seq

    def get_preview_jpeg(self, quality: int, max_width: int):
        quality = max(1, min(100, int(quality)))
        max_width = max(1, int(max_width))
        with self.lock:
            frame, tags, seq = self.frame, self._tags, self._frame_seq
            if frame is None:
                return None, None
            with self._preview_lock:
                if self._preview_seq == seq:
                    cached = self._preview_cache.get((quality, max_width))
                    if cached is not None:
                        return self._preview_frame, cached
            # Copy inside self.lock only on a preview cache miss (<= STREAM_MAX_FPS)
            # so ring-buffered decoder frames cannot tear while resizing/encoding.
            frame_copy = frame.copy()

        scale = min(1.0, max_width / frame_copy.shape[1])
        if scale < 1.0:
            dst_w = max(1, int(round(frame_copy.shape[1] * scale)))
            dst_h = max(1, int(round(frame_copy.shape[0] * scale)))
            small = cv2.resize(frame_copy, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
        else:
            small = frame_copy

        if tags is not None:
            corners, ids = tags
            if ids is not None and len(ids) > 0:
                scaled_corners = [c * scale for c in corners] if scale < 1.0 else corners
                small = localization.detection.ANNOTATE_TAGS(small, scaled_corners, ids)

        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return None, None
        jpeg = buf.tobytes()

        with self._preview_lock:
            if self._preview_seq != seq:
                self._preview_cache.clear()
                self._preview_seq = seq
                self._preview_frame = frame_copy
            self._preview_cache[(quality, max_width)] = jpeg
            token_frame = self._preview_frame
        return token_frame, jpeg

    def get_detections(self):
        with self.lock:
            return list(self.detections)

    def get_detection_summary(self) -> List[dict]:
        items = []
        for det in self.get_detections():
            if hasattr(det, "getTag"):
                items.append({
                    "tag": int(det.getTag()),
                    "r": round(float(det.getR()), 2),
                    "theta": round(float(det.getTheta()), 2),
                })
            else:
                items.append({
                    "r": round(float(det.getR()), 2),
                    "theta": round(float(det.getTheta()), 2),
                    "on_top": bool(det.isOnTop()),
                })
        return items

    def _is_stale(self) -> bool:
        if not self.enabled:
            return True
        last = self._last_processed_at
        return last > 0.0 and (time.monotonic() - last) > 2.5

    def get_framerate(self):
        if self._is_stale():
            return 0.0
        return self.framerate

    def get_backend_fps(self):
        if self._is_stale():
            return 0.0
        return self.backend_fps

    def get_fps(self) -> float:
        backend = self.get_backend_fps()
        return backend if backend > 0.0 else self.get_framerate()

    def get_processing_latency(self):
        return self.processing_latency

    def get_capture_latency(self):
        return self.capture_latency

    def get_resolution(self) -> Tuple[int, int]:
        return self._frame_resolution

    def get_camera_id(self):
        return self.camera_id

    def get_camera_name(self):
        return self.camera_name

    def get_pipeline_type(self):
        return self.pipeline_type

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        if enabled:
            self._enabled_event.set()
        else:
            self._enabled_event.clear()

    def is_enabled(self):
        return self.enabled

    def stop(self):
        self.running = False
        self._enabled_event.set()
        try:
            with self.lock:
                if self._writer is not None:
                    self._writer.release()
                    self._writer = None
                    self._recording_pending = False
                if self.cam:
                    self.cam.release()
            det_obj = getattr(self._apriltag, "__self__", None)
            if det_obj is not None and hasattr(det_obj, "close"):
                det_obj.close()
        except Exception as e:
            logger.Warn(f"Error releasing camera {self.camera_id}: {e}")

class VisionManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        if "network_tables" in self.config and "server" in self.config["network_tables"]:
            server = self.config["network_tables"]["server"]
            pipeline.ntables.set_server(server)
            logger.Log(f"NetworkTables server set to: {server}")

        if "valid_apriltag_ids" in self.config:
            localization.apriltag_solution.SET_VALID_TAG_IDS(self.config["valid_apriltag_ids"])

        self.pipelines: Dict[int, CameraPipeline] = {}

        for camera_config in self.config["cameras"]:
            pipeline_instance = CameraPipeline(camera_config)
            self.pipelines[camera_config["id"]] = pipeline_instance

        self.threads: List[Thread] = []

    def execute(self):
        affinities = _assign_pipeline_affinities(self.pipelines)
        for camera_id, cam_pipeline in self.pipelines.items():
            thread = Thread(
                target=self._run_pipeline,
                args=(cam_pipeline, affinities.get(camera_id)),
                daemon=True,
            )
            thread.start()
            self.threads.append(thread)

        for thread in self.threads:
            thread.join()

    def stop_all(self):
        for cam_pipeline in list(self.pipelines.values()):
            cam_pipeline.stop()
        for thread in list(self.threads):
            thread.join(timeout=1.0)

    def _run_pipeline(self, cam_pipeline: CameraPipeline, core_affinity: Optional[Set[int]] = None):
        if core_affinity is not None:
            _pin_current_thread(core_affinity)
        else:
            pin_current_thread_big_cores()
        _set_current_thread_nice(-10)
        cam_pipeline._ensure_apriltag_ready()
        while cam_pipeline.running:
            try:
                cam_pipeline.process_frame()
            except Exception as e:
                print(f"Error in pipeline {cam_pipeline.camera_id}: {e}")
                time.sleep(0.1)

    def get_pipeline(self, camera_id: int) -> CameraPipeline:
        return self.pipelines.get(camera_id)

    def get_all_pipelines(self) -> Dict[int, CameraPipeline]:
        return self.pipelines

    def update_camera_pipeline(self, camera_id: int, pipeline_type: str):
        if camera_id in self.pipelines:
            old_pipeline = self.pipelines[camera_id]
            old_pipeline.stop()

            camera_config = None
            for cam in self.config["cameras"]:
                if cam["id"] == camera_id:
                    cam["pipeline"] = pipeline_type
                    camera_config = cam
                    break

            if camera_config:
                new_pipeline = CameraPipeline(camera_config)
                self.pipelines[camera_id] = new_pipeline
                affinity = _assign_pipeline_affinities(self.pipelines).get(camera_id)
                thread = Thread(target=self._run_pipeline, args=(new_pipeline, affinity), daemon=True)
                thread.start()
                self.threads.append(thread)

    def toggle_camera(self, camera_id: int, enabled: bool):
        if camera_id in self.pipelines:
            self.pipelines[camera_id].set_enabled(enabled)

    def reload_all_pipelines(self):
        try:
            logger.Log("Reloading all pipelines...")

            for cam_pipeline in list(self.pipelines.values()):
                try:
                    cam_pipeline.running = False
                    cam_pipeline._enabled_event.set()
                except Exception:
                    pass

            for thread in list(self.threads):
                try:
                    thread.join(timeout=1.0)
                except Exception:
                    pass

            for cam_pipeline in list(self.pipelines.values()):
                try:
                    if hasattr(cam_pipeline, 'cam') and cam_pipeline.cam:
                        cam_pipeline.cam.release()
                except Exception:
                    pass

            self.threads = []

            if platform.system() == "Windows":
                time.sleep(1.5)
            else:
                time.sleep(0.5)

            with open(self.config_path, 'r') as f:
                self.config = json.load(f)

            try:
                pipeline.ntables.reset_instance()
            except Exception:
                pass

            if "network_tables" in self.config and "server" in self.config["network_tables"]:
                try:
                    pipeline.ntables.set_server(self.config["network_tables"]["server"])
                except Exception:
                    pass

            if "valid_apriltag_ids" in self.config:
                try:
                    localization.apriltag_solution.SET_VALID_TAG_IDS(self.config["valid_apriltag_ids"])
                except Exception:
                    pass

            self.pipelines = {}
            for camera_config in self.config["cameras"]:
                try:
                    self.pipelines[camera_config["id"]] = CameraPipeline(camera_config)
                except Exception as e:
                    logger.Warn(f"Error creating pipeline for camera {camera_config.get('id', 'unknown')}: {e}")

            affinities = _assign_pipeline_affinities(self.pipelines)
            for cam_pipeline in self.pipelines.values():
                try:
                    thread = Thread(
                        target=self._run_pipeline,
                        args=(cam_pipeline, affinities.get(cam_pipeline.camera_id)),
                        daemon=True,
                    )
                    thread.start()
                    self.threads.append(thread)
                except Exception:
                    pass

            logger.Log("All pipelines reloaded successfully")

        except Exception as e:
            logger.Warn(f"Error during pipeline reload: {e}")
            import traceback
            logger.Warn(traceback.format_exc())


if __name__ == "__main__":
    import tempfile
    import numpy as np
    gray = np.random.randint(0, 255, (800, 1280), np.uint8)
    for raw in (True, False):
        path = os.path.join(tempfile.mkdtemp(), "check.avi")
        writer, used_raw = _open_writer(path, (1280, 800), False, raw)
        assert used_raw == raw, f"raw JPEG passthrough unavailable in OpenCV {cv2.__version__}"
        for _ in range(10):
            writer.write(cv2.imencode(".jpg", gray)[1].reshape(1, -1) if raw else gray)
        writer.release()
        _set_avi_fps(path, 57.5)
        cap = cv2.VideoCapture(path)
        frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        assert frames == 10 and abs(fps - 57.5) < 0.01, (raw, frames, fps)
    print("recorder ok")
