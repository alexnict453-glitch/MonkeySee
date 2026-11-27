import math
import os
import sys
from typing import Dict, List, Set, Tuple

if __name__ == "__main__" and not __package__:
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)

import cv2
import numpy as np
from cv2.typing import MatLike
from localization.undistort import load_calibration, calibration_for
from util.config import ConfigCategory, Config
from util.logger import Logger

try:
    from localization.native_pipeline import solve_apriltags_native
except ImportError:
    solve_apriltags_native = None

logger = Logger("AprilTagSolution")

valid_tag_ids: Set[int] = set(range(1, 33))

CAM_PARAMS: Dict[int, Dict[str, Config]] = {}
_CAM_CFG_CACHE: Dict[int, Tuple[Dict[str, Config], Config, Config, Config, Config]] = {}
_TRIG_CACHE: Dict[int, Tuple[float, float, float, float, float, float, float, float]] = {}

UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 1e-10)
_UNDISTORT_ITER = getattr(cv2, "undistortPointsIter", None)

def SET_VALID_TAG_IDS(tag_ids: List[int]):
    global valid_tag_ids
    valid_tag_ids = set(tag_ids)
    logger.Log(f"Valid AprilTag IDs: {sorted(valid_tag_ids)}")

def SET_CAM(pipeline: int, config_key: str = None):
    section = config_key if config_key is not None else f"AprilTag{pipeline}"
    pref_category = ConfigCategory(section)
    CAM_PARAMS[pipeline] = {
        "CAM_MOUNT_H_deg": pref_category.getFloatConfig("CAM_MOUNT_H_deg", 0.0),
        "CAM_MOUNT_V_deg": pref_category.getFloatConfig("CAM_MOUNT_V_deg", 0.0),
        "CAM_MOUNT_ROLL_deg": pref_category.getFloatConfig("CAM_MOUNT_ROLL_deg", 0.0),
        "TAG_H_in": pref_category.getFloatConfig("TAG_H_in", 6.5),
    }
    logger.Log(f"Camera {pipeline} using config section {section}: "
               f"H={CAM_PARAMS[pipeline]['CAM_MOUNT_H_deg'].valueFloat()}, "
               f"V={CAM_PARAMS[pipeline]['CAM_MOUNT_V_deg'].valueFloat()}, "
               f"roll={CAM_PARAMS[pipeline]['CAM_MOUNT_ROLL_deg'].valueFloat()}, "
               f"TAG_H={CAM_PARAMS[pipeline]['TAG_H_in'].valueFloat()}")
    load_calibration(pipeline)

class Detection:
    __slots__ = ("r", "theta", "tag")

    def __init__(self, r_ground: float, theta_h: float, tag: int):
        self.r = r_ground
        self.theta = theta_h
        self.tag = tag

    def getTag(self) -> int:
        return self.tag

    def getR(self) -> float:
        return self.r

    def getTheta(self) -> float:
        return self.theta

    def __str__(self) -> str:
        return f"Tag {self.tag}: ({self.r:.2f}, {self.theta:.2f} deg)"

    def __repr__(self) -> str:
        return self.__str__()

def CALCULATE_PARTIAL_SOLUTION(camera_id: int, image: MatLike, all_corners, all_IDs) -> List[Detection]:
    result: List[Detection] = []

    if all_IDs is None or len(all_IDs) == 0 or all_corners is None:
        return result

    params = CAM_PARAMS.get(camera_id)
    if params is None:
        logger.Warn(f"No AprilTag params for camera {camera_id}; call SET_CAM first")
        return result

    height, width = image.shape[:2]
    calibration = calibration_for(camera_id, width, height)
    if calibration is None:
        return result
    camera_matrix, dist_coeffs = calibration

    if len(all_IDs) == 1 and len(all_corners) == 1:
        tid_int = int(all_IDs[0])
        if tid_int not in valid_tag_ids:
            return result
        c0 = all_corners[0]
        pts = (
            c0.reshape(4, 1, 2)
            if (isinstance(c0, np.ndarray) and c0.dtype == np.float64 and c0.flags.c_contiguous)
            else np.asarray(c0, dtype=np.float64).reshape(4, 1, 2)
        )
        valid_ids = (tid_int,)
        k = 1
    else:
        valid_pts = []
        valid_ids = []
        for corners, tID in zip(all_corners, all_IDs):
            tid_int = int(tID)
            if tid_int in valid_tag_ids:
                if isinstance(corners, np.ndarray) and corners.dtype == np.float64 and corners.flags.c_contiguous:
                    valid_pts.append(corners.reshape(4, 2))
                else:
                    valid_pts.append(np.asarray(corners, dtype=np.float64).reshape(4, 2))
                valid_ids.append(tid_int)

        k = len(valid_ids)
        if k == 0:
            return result
        pts = valid_pts[0].reshape(4, 1, 2) if k == 1 else np.concatenate(valid_pts, axis=0).reshape(k * 4, 1, 2)

    cfg = _CAM_CFG_CACHE.get(camera_id)
    if cfg is None or cfg[0] is not params:
        cfg = (
            params,
            params["CAM_MOUNT_H_deg"],
            params["CAM_MOUNT_V_deg"],
            params["CAM_MOUNT_ROLL_deg"],
            params["TAG_H_in"],
        )
        _CAM_CFG_CACHE[camera_id] = cfg
    _, h_cfg, v_cfg, r_cfg, th_cfg = cfg

    cam_angle_h = h_cfg.valueFloat()
    v_deg = v_cfg.valueFloat()
    roll_deg = r_cfg.valueFloat()
    tag_h = th_cfg.valueFloat()

    if solve_apriltags_native is not None:
        ids_np = np.asarray(valid_ids, dtype=np.int32)
        native_res = solve_apriltags_native(
            pts, ids_np, camera_matrix, dist_coeffs, cam_angle_h, v_deg, roll_deg, tag_h
        )
        if native_res is not None:
            return [Detection(r_g, th_g, tid) for r_g, th_g, tid in native_res]

    cached = _TRIG_CACHE.get(camera_id)
    if cached is not None and cached[0] == v_deg and cached[1] == roll_deg:
        _, _, cos_p, sin_p, cos_r, sin_r, has_roll, _ = cached
    else:
        pitch = math.radians(v_deg)
        roll = math.radians(roll_deg)
        cos_p, sin_p = math.cos(pitch), math.sin(pitch)
        cos_r, sin_r = math.cos(roll), math.sin(roll)
        has_roll = float(roll != 0.0)
        _TRIG_CACHE[camera_id] = (v_deg, roll_deg, cos_p, sin_p, cos_r, sin_r, has_roll, 0.0)

    if _UNDISTORT_ITER is not None:
        und = _UNDISTORT_ITER(pts, camera_matrix, dist_coeffs, None, None, UNDISTORT_CRITERIA)
    else:
        und = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, None, None, None, UNDISTORT_CRITERIA)

    for row, tID in zip(und.reshape(k, 8).tolist(), valid_ids):
        x0, y0, x1, y1, x2, y2, x3, y3 = row
        if has_roll:
            x0, y0 = x0 * cos_r - y0 * sin_r, x0 * sin_r + y0 * cos_r
            x1, y1 = x1 * cos_r - y1 * sin_r, x1 * sin_r + y1 * cos_r
            x2, y2 = x2 * cos_r - y2 * sin_r, x2 * sin_r + y2 * cos_r
            x3, y3 = x3 * cos_r - y3 * sin_r, x3 * sin_r + y3 * cos_r

        f0 = cos_p + y0 * sin_p
        f1 = cos_p + y1 * sin_p
        f2 = cos_p + y2 * sin_p
        f3 = cos_p + y3 * sin_p

        l_i = abs((sin_p - y0 * cos_p) / f0 - (sin_p - y3 * cos_p) / f3)
        r_i = abs((sin_p - y1 * cos_p) / f1 - (sin_p - y2 * cos_p) / f2)
        if min(l_i, r_i) < 1e-9:
            logger.Warn(f"Camera {camera_id}: tag {tID} has no height, check corner order and calibration")
            continue
        z_left = tag_h / l_i
        z_right = tag_h / r_i
        x_center = (z_left * (x0 / f0 + x3 / f3) + z_right * (x1 / f1 + x2 / f2)) * 0.25
        z_center = (z_left + z_right) * 0.5

        r_ground = math.hypot(x_center, z_center)
        theta = math.degrees(math.atan2(x_center, z_center)) + cam_angle_h
        result.append(Detection(r_ground, theta, tID))

    return result

if __name__ == "__main__":
    import localization.undistort as undistort

    class _Setting:
        def __init__(self, value):
            self.value = value
        def valueFloat(self):
            return self.value

    K = np.array([[910.3, 0.0, 630.6], [0.0, 914.6, 379.8], [0.0, 0.0, 1.0]])
    D = np.array([0.0282, 0.0837, -0.0035, 0.0006, -0.2264])
    undistort._calibrations_loaded = True
    undistort._camera_matrices[(99, 1280, 800)], undistort._dist_coeffs[(99, 1280, 800)] = K, D
    image = np.zeros((800, 1280), np.uint8)
    rng = np.random.default_rng(0)
    for pitch_deg, roll_deg in ((0.0, 0.0), (21.0, 0.0), (-10.0, 0.0), (23.0, -1.5), (0.0, 3.0)):
        CAM_PARAMS[99] = {"CAM_MOUNT_H_deg": _Setting(30.0), "CAM_MOUNT_V_deg": _Setting(pitch_deg), "CAM_MOUNT_ROLL_deg": _Setting(roll_deg), "TAG_H_in": _Setting(6.5)}
        p, q = math.radians(pitch_deg), math.radians(roll_deg)
        level = np.array([[1, 0, 0], [0, -math.cos(p), math.sin(p)], [0, math.sin(p), math.cos(p)]])
        to_cam = np.array([[math.cos(q), math.sin(q), 0], [-math.sin(q), math.cos(q), 0], [0, 0, 1]]) @ level
        batch_corners = []
        batch_ids = []
        checked = 0
        while checked < 300:
            r, bearing, yaw = rng.uniform(20, 300), rng.uniform(-33, 33), rng.uniform(-65, 65)
            b, a = math.radians(bearing), math.radians(bearing + yaw)
            center = np.array([r * math.sin(b), rng.uniform(-40, 80), r * math.cos(b)])
            across, up = 3.25 * np.array([math.cos(a), 0.0, -math.sin(a)]), np.array([0.0, 3.25, 0.0])
            cam = np.array([center - across + up, center + across + up, center + across - up, center - across - up]) @ to_cam.T
            if cam[:, 2].min() < 6:
                continue
            px = cv2.projectPoints(cam, np.zeros(3), np.zeros(3), K, D)[0].reshape(1, 4, 2)
            if (px < 0).any() or (px[..., 0] > 1279).any() or (px[..., 1] > 799).any():
                continue
            for order in ([0, 1, 2, 3], [2, 3, 0, 1]):
                found = CALCULATE_PARTIAL_SOLUTION(99, image, [px[:, order]], np.array([7]))
                assert len(found) == 1, found
                assert abs(found[0].getR() / r - 1) < 2e-4 and abs(found[0].getTheta() - bearing - 30.0) < 2e-3, (pitch_deg, roll_deg, r, bearing, yaw, found)
            if len(batch_corners) < 16:
                batch_corners.append(px)
                batch_ids.append(1 + (checked % 30))
            checked += 1
        # Multi-tag batched parity vs single-tag calls (with an invalid tag ID mixed in)
        mixed_corners = batch_corners + [batch_corners[0]]
        mixed_ids = np.array(batch_ids + [999], dtype=np.int32)
        batched = CALCULATE_PARTIAL_SOLUTION(99, image, mixed_corners, mixed_ids)
        assert len(batched) == len(batch_corners)
        for idx, (c, tid) in enumerate(zip(batch_corners, batch_ids)):
            single = CALCULATE_PARTIAL_SOLUTION(99, image, [c], np.array([tid], dtype=np.int32))
            assert len(single) == 1 and batched[idx].getTag() == tid
            dr_mm = abs(batched[idx].getR() - single[0].getR()) * 25.4
            dth = abs(batched[idx].getTheta() - single[0].getTheta())
            assert dr_mm < 1e-6 and dth < 1e-9, (dr_mm, dth)
    print("apriltag solution ok")
