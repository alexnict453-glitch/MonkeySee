import ctypes
import os
import platform
import threading
from typing import Iterable, List, Optional, Set, Tuple

import numpy as np

from util.logger import Logger

logger = Logger("NativePipeline")

DEFAULT_NATIVE_VISION_LIB = "/usr/local/lib/libnative_vision.so"
_FALLBACK_NATIVE_VISION_LIB = "/home/orangepi/MonkeySee/camera/libnative_vision.so"
DEFAULT_NAPA_LIB = "/usr/local/lib/libnapa_v061_a76_neon2_pgo.so"

NV_MAX_TAGS = 64
NV_VALID_WORDS = 10


class NvPoseConfig(ctypes.Structure):
    _fields_ = [
        ("fx", ctypes.c_double),
        ("fy", ctypes.c_double),
        ("cx", ctypes.c_double),
        ("cy", ctypes.c_double),
        ("dist", ctypes.c_double * 5),
        ("yaw_deg", ctypes.c_double),
        ("pitch_deg", ctypes.c_double),
        ("roll_deg", ctypes.c_double),
        ("tag_height", ctypes.c_double),
        ("has_cal", ctypes.c_int),
        ("filter_valid_ids", ctypes.c_int),
        ("valid_bits", ctypes.c_uint64 * NV_VALID_WORDS),
    ]


class NvPoseOut(ctypes.Structure):
    _fields_ = [
        ("r_in", ctypes.c_double),
        ("theta_deg", ctypes.c_double),
        ("tag_id", ctypes.c_int),
    ]


class NvPipelineConfig(ctypes.Structure):
    _fields_ = [
        ("min_tag_px", ctypes.c_int),
        ("max_hamming", ctypes.c_int),
        ("min_contrast", ctypes.c_int),
        ("min_border_cells", ctypes.c_int),
        ("sharpening_x100", ctypes.c_int),
        ("split_touching", ctypes.c_int),
        ("napa_threads", ctypes.c_int),
        ("recover_shadows", ctypes.c_int),
        ("shadow_min_tag_px", ctypes.c_int),
        ("apply_lens", ctypes.c_int),
        ("async_vpu", ctypes.c_int),
        ("big_cpu_mask", ctypes.c_uint32),
        ("little_cpu_mask", ctypes.c_uint32),
        ("nice_level", ctypes.c_int),
    ]


class NvFrameResult(ctypes.Structure):
    _fields_ = [
        ("seq", ctypes.c_uint64),
        ("ring_slot", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("capture_ns", ctypes.c_int64),
        ("read_ns", ctypes.c_int64),
        ("done_ns", ctypes.c_int64),
        ("decode_ms", ctypes.c_double),
        ("detect_ms", ctypes.c_double),
        ("pose_ms", ctypes.c_double),
        ("n_raw_tags", ctypes.c_int),
        ("raw_ids", ctypes.c_int * NV_MAX_TAGS),
        ("raw_corners", ctypes.c_double * (NV_MAX_TAGS * 8)),
        ("n_poses", ctypes.c_int),
        ("poses", NvPoseOut * NV_MAX_TAGS),
        ("jpeg_len", ctypes.c_int),
        ("dropped_v4l2", ctypes.c_int),
    ]


_BINDINGS = (
    ("nv_solve_apriltags", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(NvPoseConfig), ctypes.POINTER(NvPoseOut), ctypes.c_int], ctypes.c_int),
    ("nv_cam_open", [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(NvPipelineConfig), ctypes.c_char_p], ctypes.c_void_p),
    ("nv_replay_open", [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(NvPipelineConfig), ctypes.c_char_p], ctypes.c_void_p),
    ("nv_cam_get_info", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
    ("nv_cam_get_ring_slot_ptr", [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_size_t)], ctypes.c_void_p),
    ("nv_cam_set_ctrl", [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int], ctypes.c_int),
    ("nv_cam_get_ctrl", [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
    ("nv_cam_set_want_jpeg", [ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
    ("nv_cam_update_lens", [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.c_int], None),
    ("nv_cam_update_pose_cfg", [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double], None),
    ("nv_cam_set_valid_ids", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int], None),
    ("nv_cam_start", [ctypes.c_void_p], ctypes.c_int),
    ("nv_cam_stop", [ctypes.c_void_p], None),
    ("nv_cam_close", [ctypes.c_void_p], None),
    ("nv_cam_wait_result", [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.POINTER(NvFrameResult), ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
)

_lib_lock = threading.Lock()
_lib_loaded = False
_lib_handle = None
_tl_pose = threading.local()


def _cpu_set_to_mask(cpus: Optional[Iterable[int]]) -> int:
    mask = 0
    if not cpus:
        return 0
    for c in cpus:
        if 0 <= int(c) < 32:
            mask |= 1 << int(c)
    return mask


def load_native_vision_lib(lib_path: Optional[str] = None) -> Optional[ctypes.CDLL]:
    """Load libnative_vision.so on Linux, returning None when unavailable."""
    global _lib_loaded, _lib_handle
    custom = lib_path or os.environ.get("MONKEYSEE_NATIVE_VISION_LIB")
    if custom is None:
        with _lib_lock:
            if _lib_loaded:
                return _lib_handle
        if platform.system() != "Linux":
            with _lib_lock:
                _lib_loaded, _lib_handle = True, None
            return None

    candidates = [custom] if custom else [DEFAULT_NATIVE_VISION_LIB, _FALLBACK_NATIVE_VISION_LIB, "libnative_vision.so"]
    loaded = None
    for path in candidates:
        if not path or (os.path.isabs(path) and not os.path.exists(path)):
            continue
        try:
            lib = ctypes.CDLL(path)
            for name, argtypes, restype in _BINDINGS:
                fn = getattr(lib, name)
                fn.argtypes = argtypes
                fn.restype = restype
            loaded = lib
            logger.Log(f"Loaded native C vision pipeline library: {path}")
            break
        except (OSError, AttributeError) as exc:
            logger.Warn(f"Failed to load native vision library {path}: {exc}")

    if custom is None:
        with _lib_lock:
            _lib_loaded, _lib_handle = True, loaded
    return loaded


def solve_apriltags_native(
    pts_flat: np.ndarray,
    ids_arr: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    tag_height: float,
) -> Optional[List[Tuple[float, float, int]]]:
    """Solve poses in C for pre-filtered MonkeySee-ordered float64 corners (k, 4, 2)."""
    lib = load_native_vision_lib()
    if lib is None:
        return None
    dflat = np.asarray(dist_coeffs, dtype=np.float64).ravel()
    if dflat.size > 5:
        return None

    k = int(ids_arr.shape[0])
    if k <= 0:
        return []

    pts_c = pts_flat if (pts_flat.dtype == np.float64 and pts_flat.flags.c_contiguous) else np.ascontiguousarray(pts_flat, dtype=np.float64)
    ids_c = ids_arr if (ids_arr.dtype == np.int32 and ids_arr.flags.c_contiguous) else np.ascontiguousarray(ids_arr, dtype=np.int32)

    cfg = getattr(_tl_pose, "cfg", None)
    if cfg is None:
        cfg = NvPoseConfig()
        out = (NvPoseOut * NV_MAX_TAGS)()
        _tl_pose.cfg = cfg
        _tl_pose.out = out
    else:
        out = _tl_pose.out

    if k > len(out):
        out = (NvPoseOut * k)()
        _tl_pose.out = out

    cfg.fx = float(camera_matrix[0, 0])
    cfg.fy = float(camera_matrix[1, 1])
    cfg.cx = float(camera_matrix[0, 2])
    cfg.cy = float(camera_matrix[1, 2])
    for i in range(5):
        cfg.dist[i] = float(dflat[i]) if i < dflat.size else 0.0
    cfg.yaw_deg = float(yaw_deg)
    cfg.pitch_deg = float(pitch_deg)
    cfg.roll_deg = float(roll_deg)
    cfg.tag_height = float(tag_height)
    cfg.has_cal = 1
    cfg.filter_valid_ids = 0

    n_poses = lib.nv_solve_apriltags(pts_c.ctypes.data, ids_c.ctypes.data, k, ctypes.byref(cfg), out, len(out))
    return [(float(out[i].r_in), float(out[i].theta_deg), int(out[i].tag_id)) for i in range(n_poses)]


class NativeVisionWorker:
    """Native C V4L2/MPP -> NAPA -> C Newton pose worker with 8-slot gray ring."""

    def __init__(
        self,
        dev_path: Optional[str],
        width: int,
        height: int,
        fps: int = 120,
        buffersize: int = 2,
        napa_cfg: Optional[dict] = None,
        big_cpus: Optional[Set[int]] = None,
        little_cpus: Optional[Set[int]] = None,
        nice_level: int = -10,
        async_vpu: bool = False,
        napa_lib_path: Optional[str] = None,
        replay_jpegs: Optional[List[bytes]] = None,
    ):
        self._lib = load_native_vision_lib()
        self._worker = None
        self._started = False
        self._lock = threading.Lock()
        self._replay_refs = None
        self._ring_bufs: List[np.ndarray] = []
        self._ring_2d: List[np.ndarray] = []
        self._last_seq = 0
        self._res = NvFrameResult()
        self._res_ref = ctypes.byref(self._res)
        self._ctrl_c = ctypes.c_int()
        self._ctrl_ref = ctypes.byref(self._ctrl_c)
        self._jpeg_buf: Optional[np.ndarray] = None
        self._want_jpeg = False
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.buffers = int(buffersize)
        self.ring_slots = 8

        if self._lib is None:
            return

        ncfg = napa_cfg or {}
        pcfg = NvPipelineConfig(
            min_tag_px=int(ncfg.get("min_tag_px", 16)),
            max_hamming=int(ncfg.get("max_hamming", 2)),
            min_contrast=int(ncfg.get("min_contrast", 12)),
            min_border_cells=int(ncfg.get("min_border_cells", 22)),
            sharpening_x100=int(round(float(ncfg.get("sharpening", 0.5)) * 100.0)),
            split_touching=int(ncfg.get("split_touching", 0)),
            napa_threads=int(ncfg.get("threads", 2)),
            recover_shadows=1 if ncfg.get("recover_shadows", True) else 0,
            shadow_min_tag_px=int(ncfg.get("shadow_min_tag_px", 16)),
            apply_lens=1 if ncfg.get("apply_lens", True) else 0,
            async_vpu=1 if async_vpu else 0,
            big_cpu_mask=_cpu_set_to_mask(big_cpus),
            little_cpu_mask=_cpu_set_to_mask(little_cpus),
            nice_level=int(nice_level),
        )
        napa_bytes = napa_lib_path.encode("utf-8") if napa_lib_path else None

        if replay_jpegs is not None:
            arr_refs = [np.frombuffer(bytes(j), dtype=np.uint8) for j in replay_jpegs]
            ptrs = (ctypes.c_void_p * len(arr_refs))(*[int(a.ctypes.data) for a in arr_refs])
            lens = (ctypes.c_size_t * len(arr_refs))(*[int(a.nbytes) for a in arr_refs])
            self._replay_refs = (arr_refs, ptrs, lens)
            handle = self._lib.nv_replay_open(ptrs, lens, len(arr_refs), self.width, self.height, ctypes.byref(pcfg), napa_bytes)
        elif dev_path:
            handle = self._lib.nv_cam_open(
                dev_path.encode("utf-8"), self.width, self.height, int(fps), int(buffersize), ctypes.byref(pcfg), napa_bytes
            )
        else:
            handle = None

        if not handle:
            self._replay_refs = None
            return

        self._worker = handle
        w_c, h_c, fps_c, buf_c, rs_c = ctypes.c_int(), ctypes.c_int(), ctypes.c_double(), ctypes.c_int(), ctypes.c_int()
        if self._lib.nv_cam_get_info(handle, ctypes.byref(w_c), ctypes.byref(h_c), ctypes.byref(fps_c), ctypes.byref(buf_c), ctypes.byref(rs_c)) == 0:
            self.width = max(1, int(w_c.value))
            self.height = max(1, int(h_c.value))
            self.fps = float(fps_c.value) if fps_c.value > 0 else float(fps)
            self.buffers = int(buf_c.value)
            self.ring_slots = max(1, int(rs_c.value))

        cap_c = ctypes.c_size_t()
        for slot in range(self.ring_slots):
            ptr = self._lib.nv_cam_get_ring_slot_ptr(handle, slot, ctypes.byref(cap_c))
            cap = int(cap_c.value)
            arr_t = ctypes.c_uint8 * cap
            buf = np.frombuffer(arr_t.from_address(ptr), dtype=np.uint8)
            self._ring_bufs.append(buf)
            self._ring_2d.append(buf[: self.width * self.height].reshape(self.height, self.width))

    @property
    def available(self) -> bool:
        return self._worker is not None

    def start(self) -> bool:
        with self._lock:
            if not self._worker or self._started:
                return self._started
            rc = self._lib.nv_cam_start(self._worker)
            self._started = rc == 0
            return self._started

    def stop(self) -> None:
        with self._lock:
            if self._worker and self._started:
                self._lib.nv_cam_stop(self._worker)
                self._started = False

    def set_ctrl(self, ctrl_id: int, value: int) -> bool:
        if not self._worker or not self._lib:
            return False
        return self._lib.nv_cam_set_ctrl(self._worker, int(ctrl_id), int(value)) == 0

    def get_ctrl(self, ctrl_id: int) -> Optional[int]:
        if not self._worker or not self._lib:
            return None
        if self._lib.nv_cam_get_ctrl(self._worker, int(ctrl_id), self._ctrl_ref) == 0:
            return int(self._ctrl_c.value)
        return None

    def set_want_jpeg(self, want: bool) -> None:
        want_b = bool(want)
        if want_b == self._want_jpeg or not self._worker or not self._lib:
            return
        if want_b and self._jpeg_buf is None:
            max_pkt = max(self.width * self.height * 2, 2 << 20)
            self._jpeg_buf = np.empty(max_pkt, dtype=np.uint8)
        if self._lib.nv_cam_set_want_jpeg(self._worker, 1 if want_b else 0) == 0:
            self._want_jpeg = want_b

    def update_lens(self, cal: Optional[Tuple[np.ndarray, np.ndarray]]) -> bool:
        """Update NAPA lens & pose intrinsics/distortion. Returns False if >5 coeffs (needs OpenCV fallback)."""
        if not self._worker or not self._lib:
            return False
        if cal is None:
            self._lib.nv_cam_update_lens(self._worker, 0.0, 0.0, 0.0, 0.0, None, 0)
            return True
        cam_mat, dist_coeffs = cal
        dflat = np.asarray(dist_coeffs, dtype=np.float64).ravel()
        if dflat.size > 5:
            self._lib.nv_cam_update_lens(self._worker, 0.0, 0.0, 0.0, 0.0, None, 0)
            return False
        d5 = (ctypes.c_double * 5)(*[float(dflat[i]) if i < dflat.size else 0.0 for i in range(5)])
        self._lib.nv_cam_update_lens(
            self._worker,
            float(cam_mat[0, 0]),
            float(cam_mat[1, 1]),
            float(cam_mat[0, 2]),
            float(cam_mat[1, 2]),
            d5,
            1,
        )
        return True

    def update_pose_cfg(self, yaw_deg: float, pitch_deg: float, roll_deg: float, tag_height: float) -> None:
        if self._worker and self._lib:
            self._lib.nv_cam_update_pose_cfg(
                self._worker, float(yaw_deg), float(pitch_deg), float(roll_deg), float(tag_height)
            )

    def set_valid_ids(self, valid_ids: Optional[Iterable[int]]) -> None:
        if not self._worker or not self._lib:
            return
        if valid_ids is None:
            self._lib.nv_cam_set_valid_ids(self._worker, None, -1)
            return
        ids_list = sorted(int(x) for x in valid_ids)
        arr = (ctypes.c_int * len(ids_list))(*ids_list)
        self._lib.nv_cam_set_valid_ids(self._worker, arr, len(ids_list))

    def wait_result(self, timeout_ms: int = 500, want_jpeg: bool = False):
        """Wait for next completed frame (releases GIL while waiting). Returns tuple or None."""
        if not self._worker or not self._lib:
            return None
        if want_jpeg != self._want_jpeg:
            self.set_want_jpeg(want_jpeg)

        jptr = int(self._jpeg_buf.ctypes.data) if (want_jpeg and self._jpeg_buf is not None) else 0
        jcap = int(self._jpeg_buf.nbytes) if jptr else 0
        rc = self._lib.nv_cam_wait_result(
            self._worker, self._last_seq, int(timeout_ms), self._res_ref, jptr, jcap
        )
        if rc != 1:
            return None

        res = self._res
        self._last_seq = int(res.seq)
        slot = int(res.ring_slot)
        w = int(res.width)
        h = int(res.height)
        if w == self.width and h == self.height:
            gray = self._ring_2d[slot]
        else:
            gray = self._ring_bufs[slot][: w * h].reshape(h, w)

        n_raw = int(res.n_raw_tags)
        if n_raw > 0:
            raw_pts = np.frombuffer(res.raw_corners, dtype=np.float64, count=n_raw * 8).reshape(n_raw, 1, 4, 2).copy()
            corners = [raw_pts[i] for i in range(n_raw)]
            ids_arr = np.frombuffer(res.raw_ids, dtype=np.int32, count=n_raw).copy()
        else:
            corners = []
            ids_arr = None

        n_poses = int(res.n_poses)
        poses = [(float(res.poses[i].r_in), float(res.poses[i].theta_deg), int(res.poses[i].tag_id)) for i in range(n_poses)]

        jpeg = None
        if want_jpeg and res.jpeg_len > 0 and self._jpeg_buf is not None:
            jpeg = self._jpeg_buf[: int(res.jpeg_len)].copy().reshape(1, -1)

        return (
            gray,
            int(res.capture_ns),
            int(res.read_ns),
            int(res.done_ns),
            corners,
            ids_arr,
            poses,
            jpeg,
        )

    def close(self) -> None:
        with self._lock:
            worker, lib = self._worker, self._lib
            self._worker = None
            self._started = False
            self._ring_2d = []
            self._ring_bufs = []
            self._jpeg_buf = None
            self._replay_refs = None
            if worker and lib:
                try:
                    lib.nv_cam_close(worker)
                except Exception:
                    pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
