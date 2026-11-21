import argparse
import glob
import json
import os
import re
import sys

_REAL_STDOUT = sys.stdout
if "--json" in sys.argv:
    sys.stdout = sys.stderr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import localization.undistort as undistort
from calibration.charuco_calibrator import CharucoCalibrator
from calibration.checkerboard_calibrator import CheckerboardCalibrator


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate or run ChArUco optical health diagnostics on saved camera images"
    )
    parser.add_argument("folder", help="captures/calibration/cam<id>_<w>x<h>_<time> folder")
    parser.add_argument("--cal", default="cal.json", help="cal.json to read/write calibration")
    parser.add_argument(
        "--mode",
        choices=("charuco", "checkerboard"),
        default="charuco",
        help="Calibration target type (default: charuco)",
    )
    parser.add_argument(
        "--diagnose-only",
        action="store_true",
        help="Run ChArUco blur/optical health self-check on images without saving calibration",
    )
    parser.add_argument("--json", action="store_true", help="Print diagnostic report as JSON")
    args = parser.parse_args()

    match = re.match(r"cam(\d+)_", os.path.basename(os.path.normpath(args.folder)))
    if not match:
        sys.exit(f"Folder name should start with cam<id>_ : {args.folder}")
    camera_id = int(match.group(1))

    undistort.CAL_PATH = args.cal
    paths = sorted(glob.glob(os.path.join(args.folder, "*.png")))
    if not paths:
        sys.exit(f"No PNG images found in {args.folder}")

    if args.mode == "checkerboard":
        calibrator = CheckerboardCalibrator(camera_id)
        for path in paths:
            frame = cv2.imread(path)
            found, corners, _ = calibrator.find_corners(frame)
            if not found or corners is None or len(corners) < calibrator.min_corners_capture:
                print(f"skipped {os.path.basename(path)}: checkerboard not found")
                continue
            calibrator.all_corners.append(corners)
            calibrator.image_size = (frame.shape[1], frame.shape[0])
    else:
        calibrator = CharucoCalibrator(camera_id)
        frame_reports = []
        for path in paths:
            frame = cv2.imread(path)
            if frame is None:
                continue
            cc, ci, mc, mi, gray = calibrator.detect_board(frame)
            health = calibrator.analyze_frame_health(gray, cc, ci, mc, mi)
            frame_reports.append({"file": os.path.basename(path), **health})
            if not args.diagnose_only:
                if not calibrator.add_sample(gray, cc, ci, health=health, save_to_disk=False):
                    print(f"skipped {os.path.basename(path)}: {health['summary']}")

        if args.diagnose_only:
            any_error = any(r["status"] == "ERROR" for r in frame_reports)
            any_warn = any(r["status"] == "WARNING" for r in frame_reports)
            overall = "ERROR" if any_error else ("WARNING" if any_warn else "OK")
            summary = {"camera_id": camera_id, "status": overall, "frames": frame_reports}
            if args.json:
                sys.stdout = _REAL_STDOUT
                print(json.dumps(summary, indent=2))
            else:
                print(f"Camera {camera_id} ChArUco Health: {overall} ({len(frame_reports)} frames)")
                for r in frame_reports:
                    print(f"  {r['file']}: {r['status']} - {r['summary']}")
            return 2 if any_error else 0

    print(f"Using {len(calibrator.all_corners)} of {len(paths)} images")
    result = calibrator.calibrate()
    if result is None:
        sys.exit(calibrator.status_message)

    meta = result["meta"]
    k = meta["camera_matrix"]
    print(
        f"Reprojection error {meta['reprojection_error']:.3f} px, "
        f"fx {k[0][0]:.1f} fy {k[1][1]:.1f} cx {k[0][2]:.1f} cy {k[1][2]:.1f}"
    )
    diag = meta.get("diagnostics")
    if diag:
        if args.json:
            sys.stdout = _REAL_STDOUT
            print(json.dumps(diag, indent=2))
            sys.stdout = sys.stderr
        else:
            print(f"Camera Health: {diag['status']} - {diag['summary']}")
            for issue in diag.get("issues", []):
                print(f"  [{issue['severity']}] {issue['code']}: {issue['message']} -> {issue['recommendation']}")
    if not calibrator.save_calibration():
        sys.exit("Failed to save calibration")
    print(
        f"Saved camera {camera_id} {meta['resolution']['width']}x{meta['resolution']['height']} "
        f"calibration to {args.cal}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
