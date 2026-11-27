# Detection Quality Parity & ChArUco Diagnostics

## `master` / `origin/beta` Quality-Parity Contract

Never sacrifice tag recall or capture accuracy for FPS. Both `localization/detection.py` and `config.json` must preserve these exact production settings:

| Parameter | Required Value | Source / Why |
|---|---|---|
| `min_tag_px` | `16` | Preserves distant/small `ATCam4` FRC tags (`24 px` drops `97%`) |
| `shadow_min_tag_px` | `16` (`recover_shadows=True`) | Recovers dim/shadowed tags missed by `sh64` / `shadows=off` |
| `max_hamming` | `2` | Standard `tag36h11` bit-error tolerance |
| `min_contrast` | `12` | Retains low-contrast tags under gym/arena lighting |
| `min_border_cells` | `22` | Matches `master` / `origin/beta` quad acceptance |
| `sharpening` | `0.5` | Matches `master` / `origin/beta` bit-sampling sharpening |
| `split_touching` | `0` | FRC tags are isolated; avoids `~8 ms` AprilGrid pass with `0` dropped tags |
| `napa_set_lens` | Enabled (`Brown-Conrady`) | Uses per-camera `cal.json` intrinsics/distortion directly inside NAPA |
| `napa_threads` | `2` | Best 3-camera throughput on A76 (`+11.5%` over `t=1`) |
| Auto-Gain / Exposure | `auto_exposure=1`, `gain_target=100`, `tag_white_target=190` | `AUTO_GAIN_EVERY=10`, `AUTO_GAIN_WINDOW=(0.5, 1.7)`, `TAG_WHITE_CLIPPED=250` |
| Ground Pose | `TAG_H_in=6.5`, IDs `1..32` | Batched `cv2.undistortPointsIter` (`< 1e-6 mm` vs per-tag loop) |

## ChArUco Calibration & Self-Diagnostics

- **Board & Compatibility (`calibration/charuco_calibrator.py`):** `CharucoCalibrator` targets an `11×8` ChArUco board (`SQUARE_LENGTH=0.0212 m`, `MARKER_LENGTH=0.0155466 m`, `DICT_4X4_50`, `MIN_SAMPLES=15`, `TARGET_SAMPLES=54`) and handles both OpenCV `4.x` and `5.0+` APIs, including `legacyPattern` auto-detection.
- **Automated Optical & Runtime Health Checks:**
  - `analyze_frame_health()` flags `BLURRY_LENS`, `UNEVEN_FOCUS`, `LOW_CONTRAST`, `OVEREXPOSED`, `UNDEREXPOSED`, and `HIGH_NOISE` (blocking bad calibration samples automatically).
  - Post-calibration and live runtime checks detect `CALIBRATION_DRIFT` and `CAMERA_DISCONNECTED`.
  - Surfaces via `http://<pi>:5801..5804/api/calibration/health` and NT4 `CameraHealthNTables` topics (`health_ok`, `health_status`, `health_issues`, `blur_score`).
- **Offline CLI & Synthetic Verification:**
  ```powershell
  # Diagnose a folder of saved PNGs without modifying cal.json
  python .\MonkeySee\calibration\calibrate_offline.py .\captures\calibration\cam1_1280x800_... --diagnose-only --json

  # Run full synthetic optical + runtime diagnostics suite (OpenCV 4.x/5.0)
  python -W error::RuntimeWarning .\MonkeySee\calibration\test_charuco_diagnostics.py --require-full
  ```
