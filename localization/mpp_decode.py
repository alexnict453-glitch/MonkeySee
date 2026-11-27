import ctypes
import os
import platform
import threading
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from util.logger import Logger

logger = Logger("MppDecode")

DEFAULT_MPP_LIB_PATH = "/usr/local/lib/libmpp_mjpeg.so"
_FALLBACK_MPP_LIB_PATH = "/home/orangepi/opi5plus_bench/libmpp_mjpeg.so"
DEFAULT_V4L2_MPP_LIB_PATH = "/usr/local/lib/libv4l2_mpp.so"
_FALLBACK_V4L2_MPP_LIB_PATH = "/home/orangepi/MonkeySee/camera/libv4l2_mpp.so"
_MPP_DEVICE_PATH = "/dev/mpp_service"
_RING_SLOTS = 4

_lib_cache_lock = threading.Lock()
_lib_cache: Dict[str, Optional[ctypes.CDLL]] = {}
_thread_local = threading.local()

_MPP_BINDINGS = (
    ("mpp_mjpeg_create", [ctypes.c_int, ctypes.c_int, ctypes.c_int], ctypes.c_void_p),
    ("mpp_mjpeg_destroy", [ctypes.c_void_p], None),
    (
        "mpp_mjpeg_decode_gray",
        [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ],
        ctypes.c_int,
    ),
)

_V4L2_MPP_BINDINGS = (
    ("v4l2_mpp_open", [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int], ctypes.c_void_p),
    ("v4l2_mpp_close", [ctypes.c_void_p], None),
    (
        "v4l2_mpp_get_info",
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)],
        ctypes.c_int,
    ),
    ("v4l2_mpp_set_ctrl", [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int], ctypes.c_int),
    ("v4l2_mpp_get_ctrl", [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
    (
        "v4l2_mpp_grab_gray",
        [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int64),
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ],
        ctypes.c_int,
    ),
)


def _load_c_lib(
    cache_key: str,
    env_var: str,
    defaults: Sequence[str],
    bindings: Sequence[Tuple[str, list, object]],
    label: str,
    lib_path: Optional[str] = None,
    warn_no_dev: bool = False,
) -> Optional[ctypes.CDLL]:
    custom = lib_path or os.environ.get(env_var)
    if custom is None:
        with _lib_cache_lock:
            if cache_key in _lib_cache:
                return _lib_cache[cache_key]
        if platform.system() != "Linux":
            with _lib_cache_lock:
                _lib_cache[cache_key] = None
            return None
        if not (os.path.exists(_MPP_DEVICE_PATH) and os.access(_MPP_DEVICE_PATH, os.R_OK | os.W_OK)):
            if warn_no_dev:
                logger.Warn(f"{_MPP_DEVICE_PATH} unavailable; using OpenCV imdecode fallback")
            with _lib_cache_lock:
                _lib_cache[cache_key] = None
            return None

    loaded = None
    for path in ([custom] if custom else defaults):
        if not path or (os.path.isabs(path) and not os.path.exists(path)):
            continue
        try:
            lib = ctypes.CDLL(path)
            for fn_name, argtypes, restype in bindings:
                fn = getattr(lib, fn_name)
                fn.argtypes = argtypes
                fn.restype = restype
            loaded = lib
            logger.Log(f"Loaded {label}: {path}")
            break
        except (OSError, AttributeError) as exc:
            logger.Warn(f"Failed to load {label} {path}: {exc}")

    if custom is None:
        with _lib_cache_lock:
            _lib_cache[cache_key] = loaded
    return loaded


def _alloc_ring(w: int, h: int) -> Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[int, int]]]:
    # Pre-touch pages at init and keep 4 ring slots (25 ms tear-free window at 120 FPS)
    # so MJPEG preview copies can happen safely outside the camera pipeline lock.
    cap = max(1, w * h)
    ring = [np.zeros(cap, dtype=np.uint8) for _ in range(_RING_SLOTS)]
    ring_2d = [buf.reshape(h, w) for buf in ring]
    ring_ptrs = [(int(buf.ctypes.data), int(buf.nbytes)) for buf in ring]
    return ring, ring_2d, ring_ptrs


def _as_uint8_view(raw) -> Tuple[np.ndarray, object]:
    if isinstance(raw, np.ndarray):
        arr = raw if (raw.dtype == np.uint8 and raw.flags["C_CONTIGUOUS"]) else np.ascontiguousarray(raw, dtype=np.uint8)
        return arr, arr
    return np.frombuffer(raw, dtype=np.uint8), raw


def _jpeg_dimensions(arr: np.ndarray) -> Optional[Tuple[int, int]]:
    """Parse JPEG SOF width and height without decoding."""
    view = memoryview(arr.ravel())
    n = len(view)
    if n < 4 or view[0] != 0xFF or view[1] != 0xD8:
        return None
    pos = 2
    while pos + 1 < n:
        while pos < n and view[pos] != 0xFF:
            pos += 1
        while pos < n and view[pos] == 0xFF:
            pos += 1
        if pos >= n:
            break
        marker = view[pos]
        pos += 1
        if marker in (0xD9, 0xDA):
            break
        if marker in (0x00, 0x01) or (0xD0 <= marker <= 0xD8):
            continue
        if pos + 2 > n:
            break
        seg_len = (view[pos] << 8) | view[pos + 1]
        if seg_len < 2 or pos + seg_len > n:
            break
        if marker in (0xC0, 0xC1, 0xC2, 0xC3):
            if seg_len >= 7:
                h = (view[pos + 3] << 8) | view[pos + 4]
                w = (view[pos + 5] << 8) | view[pos + 6]
                if w > 0 and h > 0:
                    return w, h
            return None
        pos += seg_len
    return None


class MppMjpegDecoder:
    """Rockchip MPP hardware MJPEG-to-grayscale decoder with 4-slot ring buffer and OpenCV fallback."""

    def __init__(self, max_w: int = 1280, max_h: int = 800, cachable: Union[bool, int] = True, lib_path: Optional[str] = None):
        self._max_w = int(max_w)
        self._max_h = int(max_h)
        self._cachable = 1 if cachable else 0
        self._lock = threading.Lock()
        self._lib = _load_c_lib(
            "mpp", "MONKEYSEE_MPP_LIB",
            (DEFAULT_MPP_LIB_PATH, _FALLBACK_MPP_LIB_PATH, "libmpp_mjpeg.so"),
            _MPP_BINDINGS, "Rockchip MPP MJPEG library", lib_path, warn_no_dev=True,
        )
        self._dec: Optional[int] = None
        self._ring: List[np.ndarray] = []
        self._ring_2d: List[np.ndarray] = []
        self._ring_ptrs: List[Tuple[int, int]] = []
        self._ring_idx = 0
        self._w_c, self._h_c, self._fmt_c = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        self._w_ref = ctypes.byref(self._w_c)
        self._h_ref = ctypes.byref(self._h_c)
        self._fmt_ref = ctypes.byref(self._fmt_c)

        if self._lib is not None:
            self._dec = self._lib.mpp_mjpeg_create(self._max_w, self._max_h, self._cachable)
            if not self._dec:
                logger.Warn("mpp_mjpeg_create failed; falling back to OpenCV imdecode")
                self._dec = None
            else:
                self._ring, self._ring_2d, self._ring_ptrs = _alloc_ring(self._max_w, self._max_h)

    @property
    def available(self) -> bool:
        return self._dec is not None

    is_hardware_active = available

    def _recreate_locked(self, max_w: int, max_h: int) -> bool:
        if self._lib is None:
            return False
        if self._dec is not None:
            self._lib.mpp_mjpeg_destroy(self._dec)
        self._max_w = max(self._max_w, int(max_w))
        self._max_h = max(self._max_h, int(max_h))
        self._dec = self._lib.mpp_mjpeg_create(self._max_w, self._max_h, self._cachable)
        if not self._dec:
            logger.Warn(f"mpp_mjpeg_create({self._max_w}, {self._max_h}) failed; using OpenCV fallback")
            self._dec = None
            self._ring, self._ring_2d, self._ring_ptrs = [], [], []
            return False
        self._ring, self._ring_2d, self._ring_ptrs = _alloc_ring(self._max_w, self._max_h)
        self._ring_idx = 0
        return True

    def _fallback_imdecode(self, arr: np.ndarray, out: Optional[np.ndarray]) -> Optional[np.ndarray]:
        img = cv2.imdecode(arr.ravel(), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        if out is not None and out.dtype == np.uint8 and out.flags["C_CONTIGUOUS"] and out.nbytes >= img.nbytes:
            h, w = img.shape[:2]
            dst = out.ravel()[: h * w].reshape(h, w)
            np.copyto(dst, img)
            return dst
        return img

    def _decode_with_mpp_locked(
        self, arr: np.ndarray, nbytes: int, out: Optional[np.ndarray], min_bytes: int = 0
    ) -> Optional[np.ndarray]:
        use_out = out is not None and out.dtype == np.uint8 and out.flags["C_CONTIGUOUS"] and out.nbytes >= min_bytes
        idx = self._ring_idx
        if use_out:
            target = out
            target_ptr, target_cap = int(out.ctypes.data), int(out.nbytes)
        else:
            target = self._ring[idx]
            target_ptr, target_cap = self._ring_ptrs[idx]
        rc = self._lib.mpp_mjpeg_decode_gray(
            self._dec, arr.ctypes.data, nbytes, target_ptr, target_cap,
            self._w_ref, self._h_ref, self._fmt_ref,
        )
        if rc == 0:
            w_out, h_out = self._w_c.value, self._h_c.value
            if w_out > 0 and h_out > 0 and target_cap >= w_out * h_out:
                if not use_out:
                    self._ring_idx = (idx + 1) % _RING_SLOTS
                    if w_out == self._max_w and h_out == self._max_h:
                        return self._ring_2d[idx]
                return target.ravel()[: w_out * h_out].reshape(h_out, w_out)
        return None

    def decode_gray(self, raw, out: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Decode raw MJPEG bytes/array to 2D uint8 grayscale, falling back to cv2.imdecode."""
        if raw is None:
            return None
        arr, _keepalive = _as_uint8_view(raw)
        nbytes = int(arr.nbytes)
        if nbytes == 0:
            return None

        with self._lock:
            if self._dec is not None:
                max_pkt = max(self._max_w * self._max_h * 2, 2 << 20)
                if nbytes <= max_pkt:
                    decoded = self._decode_with_mpp_locked(arr, nbytes, out)
                    if decoded is not None:
                        return decoded
                dims = _jpeg_dimensions(arr)
                if dims is not None and (dims[0] > self._max_w or dims[1] > self._max_h):
                    if self._recreate_locked(*dims):
                        max_pkt = max(self._max_w * self._max_h * 2, 2 << 20)
                        if nbytes <= max_pkt:
                            decoded = self._decode_with_mpp_locked(arr, nbytes, out, min_bytes=dims[0] * dims[1])
                            if decoded is not None:
                                return decoded

        return self._fallback_imdecode(arr, out)

    def close(self) -> None:
        lock = getattr(self, "_lock", None)
        if lock is None:
            return
        with lock:
            dec, lib = getattr(self, "_dec", None), getattr(self, "_lib", None)
            self._dec, self._ring, self._ring_2d, self._ring_ptrs = None, [], [], []
            if dec is not None and lib is not None:
                try:
                    lib.mpp_mjpeg_destroy(dec)
                except Exception:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def decode_mjpeg_gray(raw, decoder: Optional[MppMjpegDecoder] = None, out: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    if decoder is None:
        decoder = getattr(_thread_local, "decoder", None)
        if decoder is None:
            decoder = MppMjpegDecoder()
            _thread_local.decoder = decoder
    return decoder.decode_gray(raw, out=out)


class V4L2MppCamera:
    """Direct Linux V4L2 MMAP + Rockchip MPP MJPEG-to-grayscale capture with 4-slot ring buffer."""

    def __init__(
        self, dev_path: str, width: int = 1280, height: int = 800, fps: int = 120,
        buffersize: int = 2, cachable: Union[bool, int] = True, lib_path: Optional[str] = None,
    ):
        self._lock = threading.Lock()
        self._lib = _load_c_lib(
            "v4l2_mpp", "MONKEYSEE_V4L2_MPP_LIB",
            (DEFAULT_V4L2_MPP_LIB_PATH, _FALLBACK_V4L2_MPP_LIB_PATH, "libv4l2_mpp.so"),
            _V4L2_MPP_BINDINGS, "V4L2+MPP capture library", lib_path,
        )
        self._cam: Optional[int] = None
        self.width, self.height, self.fps, self.buffers = int(width), int(height), float(fps), int(buffersize)
        self._max_pkt = max(self.width * self.height * 2, 2 << 20)
        self._ring: List[np.ndarray] = []
        self._ring_2d: List[np.ndarray] = []
        self._ring_ptrs: List[Tuple[int, int]] = []
        self._ring_idx = 0
        self._jpeg_buf: Optional[np.ndarray] = None
        self._jpeg_ptr = 0
        self._jpeg_cap = 0
        self._w_c, self._h_c, self._ts_c = ctypes.c_int(), ctypes.c_int(), ctypes.c_int64()
        self._jlen_c, self._drop_c, self._ctrl_c = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        self._w_ref = ctypes.byref(self._w_c)
        self._h_ref = ctypes.byref(self._h_c)
        self._ts_ref = ctypes.byref(self._ts_c)
        self._jlen_ref = ctypes.byref(self._jlen_c)
        self._drop_ref = ctypes.byref(self._drop_c)
        self._ctrl_ref = ctypes.byref(self._ctrl_c)

        if self._lib is not None and dev_path:
            handle = self._lib.v4l2_mpp_open(
                dev_path.encode("utf-8"), int(width), int(height), int(fps), int(buffersize), 1 if cachable else 0
            )
            if handle:
                self._cam = handle
                w_c, h_c, fps_c, buf_c, pkt_c = ctypes.c_int(), ctypes.c_int(), ctypes.c_double(), ctypes.c_int(), ctypes.c_int()
                if self._lib.v4l2_mpp_get_info(
                    handle, ctypes.byref(w_c), ctypes.byref(h_c), ctypes.byref(fps_c), ctypes.byref(buf_c), ctypes.byref(pkt_c)
                ) == 0:
                    self.width = max(1, w_c.value)
                    self.height = max(1, h_c.value)
                    self.fps = fps_c.value if fps_c.value > 0 else float(fps)
                    self.buffers = max(1, buf_c.value)
                    self._max_pkt = max(pkt_c.value, 2 << 20)
                self._ring, self._ring_2d, self._ring_ptrs = _alloc_ring(self.width, self.height)

    @property
    def available(self) -> bool:
        return self._cam is not None

    def set_ctrl(self, ctrl_id: int, value: int) -> bool:
        with self._lock:
            if self._cam is None or self._lib is None:
                return False
            return self._lib.v4l2_mpp_set_ctrl(self._cam, int(ctrl_id), int(value)) == 0

    def get_ctrl(self, ctrl_id: int) -> Optional[int]:
        with self._lock:
            if self._cam is None or self._lib is None:
                return None
            if self._lib.v4l2_mpp_get_ctrl(self._cam, int(ctrl_id), self._ctrl_ref) == 0:
                return int(self._ctrl_c.value)
            return None

    def grab_gray(self, timeout_ms: int = 500, want_jpeg: bool = False) -> Tuple[Optional[np.ndarray], int, Optional[np.ndarray]]:
        with self._lock:
            if self._cam is None or self._lib is None or not self._ring:
                return None, 0, None

            idx = self._ring_idx
            target_ptr, target_cap = self._ring_ptrs[idx]
            jpeg_ptr, jpeg_cap = 0, 0
            if want_jpeg:
                if self._jpeg_buf is None or self._jpeg_buf.nbytes < self._max_pkt:
                    self._jpeg_buf = np.empty(self._max_pkt, dtype=np.uint8)
                    self._jpeg_ptr, self._jpeg_cap = int(self._jpeg_buf.ctypes.data), int(self._jpeg_buf.nbytes)
                jpeg_ptr, jpeg_cap = self._jpeg_ptr, self._jpeg_cap

            rc = self._lib.v4l2_mpp_grab_gray(
                self._cam, int(timeout_ms), target_ptr, target_cap,
                self._w_ref, self._h_ref, self._ts_ref,
                jpeg_ptr, jpeg_cap, self._jlen_ref, self._drop_ref,
            )

            jpeg_arr: Optional[np.ndarray] = None
            jlen = int(self._jlen_c.value)
            if want_jpeg and jlen > 0 and self._jpeg_buf is not None and jlen <= self._jpeg_cap:
                jpeg_arr = self._jpeg_buf[:jlen].copy().reshape(1, -1)

            if rc == 0:
                w_out, h_out = int(self._w_c.value), int(self._h_c.value)
                if w_out > 0 and h_out > 0 and target_cap >= w_out * h_out:
                    self._ring_idx = (idx + 1) % _RING_SLOTS
                    gray = self._ring_2d[idx] if (w_out == self.width and h_out == self.height) else self._ring[idx][: w_out * h_out].reshape(h_out, w_out)
                    return gray, int(self._ts_c.value), jpeg_arr

            if jpeg_arr is not None:
                fallback = cv2.imdecode(jpeg_arr.ravel(), cv2.IMREAD_GRAYSCALE)
                if fallback is not None:
                    return fallback, int(self._ts_c.value), jpeg_arr

            return None, 0, None

    def close(self) -> None:
        lock = getattr(self, "_lock", None)
        if lock is None:
            return
        with lock:
            cam, lib = getattr(self, "_cam", None), getattr(self, "_lib", None)
            self._cam, self._ring, self._ring_2d, self._ring_ptrs, self._jpeg_buf = None, [], [], [], None
            if cam is not None and lib is not None:
                try:
                    lib.v4l2_mpp_close(cam)
                except Exception:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
