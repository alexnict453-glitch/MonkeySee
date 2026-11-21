import ctypes
import os
import platform
import threading
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from util.logger import Logger

logger = Logger("MppDecode")

DEFAULT_MPP_LIB_PATH = "/usr/local/lib/libmpp_mjpeg.so"
_FALLBACK_MPP_LIB_PATH = "/home/orangepi/opi5plus_bench/libmpp_mjpeg.so"
DEFAULT_V4L2_MPP_LIB_PATH = "/usr/local/lib/libv4l2_mpp.so"
_FALLBACK_V4L2_MPP_LIB_PATH = "/home/orangepi/MonkeySee/camera/libv4l2_mpp.so"
_MPP_DEVICE_PATH = "/dev/mpp_service"
_RING_SLOTS = 3

_lib_lock = threading.Lock()
_lib_loaded = False
_lib_handle = None
_v4l2_lib_lock = threading.Lock()
_v4l2_lib_loaded = False
_v4l2_lib_handle = None
_thread_local = threading.local()


def _load_mpp_library(lib_path: Optional[str] = None):
    """Load libmpp_mjpeg.so with exact C ABI bindings, returning None if unavailable."""
    global _lib_loaded, _lib_handle

    custom_path = lib_path or os.environ.get("MONKEYSEE_MPP_LIB")
    if custom_path is None:
        with _lib_lock:
            if _lib_loaded:
                return _lib_handle

    if platform.system() != "Linux" and custom_path is None:
        with _lib_lock:
            _lib_loaded = True
            _lib_handle = None
        return None

    if custom_path is None:
        if not (os.path.exists(_MPP_DEVICE_PATH) and os.access(_MPP_DEVICE_PATH, os.R_OK | os.W_OK)):
            logger.Warn(f"{_MPP_DEVICE_PATH} unavailable; using OpenCV imdecode fallback")
            with _lib_lock:
                _lib_loaded = True
                _lib_handle = None
            return None

    candidates = [custom_path] if custom_path else [DEFAULT_MPP_LIB_PATH, _FALLBACK_MPP_LIB_PATH, "libmpp_mjpeg.so"]
    loaded = None
    for path in candidates:
        if not path:
            continue
        if os.path.isabs(path) and not os.path.exists(path):
            continue
        try:
            lib = ctypes.CDLL(path)
            lib.mpp_mjpeg_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
            lib.mpp_mjpeg_create.restype = ctypes.c_void_p
            lib.mpp_mjpeg_destroy.argtypes = [ctypes.c_void_p]
            lib.mpp_mjpeg_destroy.restype = None
            lib.mpp_mjpeg_decode_gray.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
            ]
            lib.mpp_mjpeg_decode_gray.restype = ctypes.c_int
            loaded = lib
            logger.Log(f"Loaded Rockchip MPP MJPEG library: {path}")
            break
        except OSError as exc:
            logger.Warn(f"Failed to load MPP library {path}: {exc}")

    if custom_path is None:
        with _lib_lock:
            _lib_loaded = True
            _lib_handle = loaded
    return loaded


def _as_uint8_view(raw) -> Tuple[np.ndarray, object]:
    """Return a contiguous uint8 numpy view plus keepalive reference."""
    if isinstance(raw, np.ndarray):
        arr = raw if (raw.dtype == np.uint8 and raw.flags["C_CONTIGUOUS"]) else np.ascontiguousarray(raw, dtype=np.uint8)
        return arr, arr
    arr = np.frombuffer(raw, dtype=np.uint8)
    return arr, raw


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
        if marker == 0xD9 or marker == 0xDA:
            break
        if marker == 0x00 or marker == 0x01 or (0xD0 <= marker <= 0xD8):
            continue
        if pos + 2 > n:
            break
        seg_len = (view[pos] << 8) | view[pos + 1]
        if seg_len < 2 or pos + seg_len > n:
            break
        if marker in (0xC0, 0xC1, 0xC2, 0xC3):
            if seg_len >= 7:
                height = (view[pos + 3] << 8) | view[pos + 4]
                width = (view[pos + 5] << 8) | view[pos + 6]
                if width > 0 and height > 0:
                    return width, height
            return None
        pos += seg_len
    return None


class MppMjpegDecoder:
    """Rockchip MPP hardware MJPEG-to-grayscale decoder with 3-slot ring buffer and OpenCV fallback."""

    def __init__(
        self,
        max_w: int = 1280,
        max_h: int = 800,
        cachable: Union[bool, int] = True,
        lib_path: Optional[str] = None,
    ):
        self._max_w = int(max_w)
        self._max_h = int(max_h)
        self._cachable = 1 if cachable else 0
        self._lock = threading.Lock()
        self._lib = _load_mpp_library(lib_path)
        self._dec: Optional[int] = None
        self._ring: List[np.ndarray] = []
        self._ring_2d: List[np.ndarray] = []
        self._ring_idx = 0
        self._w_c = ctypes.c_int()
        self._h_c = ctypes.c_int()
        self._fmt_c = ctypes.c_int()

        if self._lib is not None:
            self._dec = self._lib.mpp_mjpeg_create(self._max_w, self._max_h, self._cachable)
            if not self._dec:
                logger.Warn("mpp_mjpeg_create failed; falling back to OpenCV imdecode")
                self._dec = None
            else:
                self._alloc_ring_locked()

    @property
    def available(self) -> bool:
        return self._dec is not None

    @property
    def is_hardware_active(self) -> bool:
        return self._dec is not None

    def _alloc_ring_locked(self) -> None:
        cap = max(1, self._max_w * self._max_h)
        # Pre-touch pages at init so 120 FPS decode avoids per-frame mmap page faults.
        self._ring = [np.zeros(cap, dtype=np.uint8) for _ in range(_RING_SLOTS)]
        self._ring_2d = [buf.reshape(self._max_h, self._max_w) for buf in self._ring]
        self._ring_idx = 0

    def _recreate_locked(self, max_w: int, max_h: int) -> bool:
        if self._lib is None:
            return False
        if self._dec is not None:
            self._lib.mpp_mjpeg_destroy(self._dec)
            self._dec = None
        self._max_w = max(self._max_w, int(max_w))
        self._max_h = max(self._max_h, int(max_h))
        self._dec = self._lib.mpp_mjpeg_create(self._max_w, self._max_h, self._cachable)
        if not self._dec:
            logger.Warn(f"mpp_mjpeg_create({self._max_w}, {self._max_h}) failed; using OpenCV fallback")
            self._dec = None
            self._ring = []
            self._ring_2d = []
            return False
        self._alloc_ring_locked()
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
    ) -> Tuple[int, Optional[np.ndarray]]:
        target = None
        used_ring = False
        if out is not None and out.dtype == np.uint8 and out.flags["C_CONTIGUOUS"] and out.nbytes >= min_bytes:
            target = out
        if target is None:
            idx = self._ring_idx
            target = self._ring[idx]
            used_ring = True

        rc = self._lib.mpp_mjpeg_decode_gray(
            self._dec,
            arr.ctypes.data,
            nbytes,
            target.ctypes.data,
            int(target.nbytes),
            ctypes.byref(self._w_c),
            ctypes.byref(self._h_c),
            ctypes.byref(self._fmt_c),
        )
        if rc == 0:
            w_out = self._w_c.value
            h_out = self._h_c.value
            if w_out > 0 and h_out > 0 and target.nbytes >= w_out * h_out:
                if used_ring:
                    self._ring_idx = (idx + 1) % _RING_SLOTS
                    if w_out == self._max_w and h_out == self._max_h:
                        return 0, self._ring_2d[idx]
                return 0, target.ravel()[: w_out * h_out].reshape(h_out, w_out)
        return rc, None

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
                    rc, decoded = self._decode_with_mpp_locked(arr, nbytes, out)
                    if decoded is not None:
                        return decoded
                else:
                    rc = -1

                # Slow path: only inspect JPEG SOF headers if hardware decode failed or packet exceeded max_pkt.
                dims = _jpeg_dimensions(arr)
                if dims is not None:
                    w_hdr, h_hdr = dims
                    if w_hdr > self._max_w or h_hdr > self._max_h:
                        if self._recreate_locked(w_hdr, h_hdr):
                            max_pkt = max(self._max_w * self._max_h * 2, 2 << 20)
                            if nbytes <= max_pkt:
                                _, decoded = self._decode_with_mpp_locked(arr, nbytes, out, min_bytes=w_hdr * h_hdr)
                                if decoded is not None:
                                    return decoded

        return self._fallback_imdecode(arr, out)

    def close(self) -> None:
        lock = getattr(self, "_lock", None)
        if lock is None:
            return
        with lock:
            dec = getattr(self, "_dec", None)
            lib = getattr(self, "_lib", None)
            self._dec = None
            self._ring = []
            self._ring_2d = []
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
    """Decode MJPEG frame to grayscale using provided or thread-local MppMjpegDecoder."""
    if decoder is None:
        decoder = getattr(_thread_local, "decoder", None)
        if decoder is None:
            decoder = MppMjpegDecoder()
            _thread_local.decoder = decoder
    return decoder.decode_gray(raw, out=out)


def _load_v4l2_mpp_library(lib_path: Optional[str] = None):
    """Load libv4l2_mpp.so with exact C ABI bindings, returning None if unavailable."""
    global _v4l2_lib_loaded, _v4l2_lib_handle

    custom_path = lib_path or os.environ.get("MONKEYSEE_V4L2_MPP_LIB")
    if custom_path is None:
        with _v4l2_lib_lock:
            if _v4l2_lib_loaded:
                return _v4l2_lib_handle

    if platform.system() != "Linux" and custom_path is None:
        with _v4l2_lib_lock:
            _v4l2_lib_loaded = True
            _v4l2_lib_handle = None
        return None

    if custom_path is None:
        if not (os.path.exists(_MPP_DEVICE_PATH) and os.access(_MPP_DEVICE_PATH, os.R_OK | os.W_OK)):
            with _v4l2_lib_lock:
                _v4l2_lib_loaded = True
                _v4l2_lib_handle = None
            return None

    candidates = (
        [custom_path]
        if custom_path
        else [DEFAULT_V4L2_MPP_LIB_PATH, _FALLBACK_V4L2_MPP_LIB_PATH, "libv4l2_mpp.so"]
    )
    loaded = None
    for path in candidates:
        if not path:
            continue
        if os.path.isabs(path) and not os.path.exists(path):
            continue
        try:
            lib = ctypes.CDLL(path)
            lib.v4l2_mpp_open.argtypes = [
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.v4l2_mpp_open.restype = ctypes.c_void_p
            lib.v4l2_mpp_close.argtypes = [ctypes.c_void_p]
            lib.v4l2_mpp_close.restype = None
            lib.v4l2_mpp_get_info.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
            ]
            lib.v4l2_mpp_get_info.restype = ctypes.c_int
            lib.v4l2_mpp_set_ctrl.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int]
            lib.v4l2_mpp_set_ctrl.restype = ctypes.c_int
            lib.v4l2_mpp_get_ctrl.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_int),
            ]
            lib.v4l2_mpp_get_ctrl.restype = ctypes.c_int
            lib.v4l2_mpp_grab_gray.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int64),
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
            ]
            lib.v4l2_mpp_grab_gray.restype = ctypes.c_int
            loaded = lib
            logger.Log(f"Loaded V4L2+MPP capture library: {path}")
            break
        except (OSError, AttributeError) as exc:
            logger.Warn(f"Failed to load V4L2+MPP library {path}: {exc}")

    if custom_path is None:
        with _v4l2_lib_lock:
            _v4l2_lib_loaded = True
            _v4l2_lib_handle = loaded
    return loaded


class V4L2MppCamera:
    """Direct Linux V4L2 MMAP + Rockchip MPP MJPEG-to-grayscale capture with 3-slot ring buffer."""

    def __init__(
        self,
        dev_path: str,
        width: int = 1280,
        height: int = 800,
        fps: int = 120,
        buffersize: int = 2,
        cachable: Union[bool, int] = True,
        lib_path: Optional[str] = None,
    ):
        self._lock = threading.Lock()
        self._lib = _load_v4l2_mpp_library(lib_path)
        self._cam: Optional[int] = None
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.buffers = int(buffersize)
        self._max_pkt = max(self.width * self.height * 2, 2 << 20)
        self._ring: List[np.ndarray] = []
        self._ring_2d: List[np.ndarray] = []
        self._ring_idx = 0
        self._jpeg_buf: Optional[np.ndarray] = None

        self._w_c = ctypes.c_int()
        self._h_c = ctypes.c_int()
        self._ts_c = ctypes.c_int64()
        self._jlen_c = ctypes.c_int()
        self._drop_c = ctypes.c_int()
        self._ctrl_c = ctypes.c_int()

        if self._lib is not None and dev_path:
            handle = self._lib.v4l2_mpp_open(
                dev_path.encode("utf-8"),
                int(width),
                int(height),
                int(fps),
                int(buffersize),
                1 if cachable else 0,
            )
            if handle:
                self._cam = handle
                w_c = ctypes.c_int()
                h_c = ctypes.c_int()
                fps_c = ctypes.c_double()
                buf_c = ctypes.c_int()
                pkt_c = ctypes.c_int()
                if self._lib.v4l2_mpp_get_info(
                    handle,
                    ctypes.byref(w_c),
                    ctypes.byref(h_c),
                    ctypes.byref(fps_c),
                    ctypes.byref(buf_c),
                    ctypes.byref(pkt_c),
                ) == 0:
                    self.width = max(1, w_c.value)
                    self.height = max(1, h_c.value)
                    self.fps = fps_c.value if fps_c.value > 0 else float(fps)
                    self.buffers = max(1, buf_c.value)
                    self._max_pkt = max(pkt_c.value, 2 << 20)
                cap = max(1, self.width * self.height)
                # Pre-touch pages at init so 120 FPS capture avoids per-frame page faults.
                self._ring = [np.zeros(cap, dtype=np.uint8) for _ in range(_RING_SLOTS)]
                self._ring_2d = [buf.reshape(self.height, self.width) for buf in self._ring]

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
            if self._lib.v4l2_mpp_get_ctrl(self._cam, int(ctrl_id), ctypes.byref(self._ctrl_c)) == 0:
                return int(self._ctrl_c.value)
            return None

    def grab_gray(
        self, timeout_ms: int = 500, want_jpeg: bool = False
    ) -> Tuple[Optional[np.ndarray], int, Optional[np.ndarray]]:
        """Grab latest V4L2 MJPEG frame, decode via MPP into ring buffer, return (gray, ts_ns, opt_jpeg)."""
        with self._lock:
            if self._cam is None or self._lib is None or not self._ring:
                return None, 0, None

            idx = self._ring_idx
            target = self._ring[idx]
            jpeg_ptr = 0
            jpeg_cap = 0
            if want_jpeg:
                if self._jpeg_buf is None or self._jpeg_buf.nbytes < self._max_pkt:
                    self._jpeg_buf = np.empty(self._max_pkt, dtype=np.uint8)
                jpeg_ptr = int(self._jpeg_buf.ctypes.data)
                jpeg_cap = int(self._jpeg_buf.nbytes)

            rc = self._lib.v4l2_mpp_grab_gray(
                self._cam,
                int(timeout_ms),
                target.ctypes.data,
                int(target.nbytes),
                ctypes.byref(self._w_c),
                ctypes.byref(self._h_c),
                ctypes.byref(self._ts_c),
                jpeg_ptr,
                jpeg_cap,
                ctypes.byref(self._jlen_c),
                ctypes.byref(self._drop_c),
            )

            jpeg_arr: Optional[np.ndarray] = None
            jlen = int(self._jlen_c.value)
            if want_jpeg and jlen > 0 and self._jpeg_buf is not None and jlen <= self._jpeg_buf.nbytes:
                jpeg_arr = self._jpeg_buf[:jlen].copy().reshape(1, -1)

            if rc == 0:
                w_out = int(self._w_c.value)
                h_out = int(self._h_c.value)
                if w_out > 0 and h_out > 0 and target.nbytes >= w_out * h_out:
                    self._ring_idx = (idx + 1) % _RING_SLOTS
                    if w_out == self.width and h_out == self.height:
                        return self._ring_2d[idx], int(self._ts_c.value), jpeg_arr
                    return target[: w_out * h_out].reshape(h_out, w_out), int(self._ts_c.value), jpeg_arr

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
            cam = getattr(self, "_cam", None)
            lib = getattr(self, "_lib", None)
            self._cam = None
            self._ring = []
            self._ring_2d = []
            self._jpeg_buf = None
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
