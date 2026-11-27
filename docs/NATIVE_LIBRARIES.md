# Native RK3588 Libraries & Performance Baselines

## Installed `/usr/local/lib` Artifacts

| Library | Source | Purpose & Key Build Flags |
|---|---|---|
| `librockchip_mpp.so` | `nyanmisaka/mpp` (`jellyfin-mpp`) | Rockchip VPU/JPEG hardware decode (`-DRKPLATFORM=ON -DHAVE_DRM=ON`) |
| `libmpp_mjpeg.so` | `opi5plus_bench/mpp_mjpeg.c` | Cached/zero-copy HW MJPG → gray8 (`gcc-12 -shared -fPIC -O3 -mcpu=cortex-a76 -lrockchip_mpp`) |
| `libv4l2_mpp.so` | `camera/v4l2_mpp.c` | Direct V4L2 `MMAP` (`BUFFERSIZE=2`), latest-drain, immediate `VIDIOC_QBUF`, `wc_pkt_nodup_copy` (uncachable/WC `pkt_grp`, persistent `MppPacket`, cached `out_memcpy`; `-Wall -Wextra -Werror`) |
| `libnative_vision.so` | `camera/native_vision.c` | Fused C worker: V4L2 `MMAP`/replay + `wc_pkt_nodup_copy` MPP + `dlopen` `libnapa_v061_a76_neon2_pgo.so` + 2D Newton-Raphson Brown-Conrady undistortion & ground-plane pose (`-O3 -mcpu=cortex-a76 -fno-math-errno -fno-trapping-math -std=c99 -Wall -Wextra -Werror`) |
| `libnapa_v061_a76_neon2_pgo.so` | `napa-vision 0.6.1` + `napa_core_opt.c` | OPi5+ Phase 12 (`2026-11-26`): GCC `12.2.0` plain `-O3 -mcpu=cortex-a76 -fno-math-errno -fno-trapping-math -std=c99 -fvisibility=hidden -fno-wrapv -pthread`. Note: `py_mask[16]` scans 100% of full frames (`<= 2048` rows) dynamically from candidate boxes (zero ROI crop). OPi5 `funkyvision4` (not yet synced to Phase 12): GCC `12.3.0` 2-stage PGO |

## Phase 12 Zero-Loss `sh16` C/NEON & Direct MPP Optimizations (`2026-11-26`, OPi5+)

### NAPA v0.6.1 Pillars 1–5 + Swarm v2 Kept Wins (`0.000000000 px` / `0` mismatches vs. stock `libnapa_v061_a76_o3.so`)

1. **`dark_bits_quad`:** Register-expanded 4-row NEON thresholding directly from tile thresholds `tt[]`, skipping `expand4` writes on full 4-row tiles.
2. **`extrema_band_worker` + `comp_flush_pair`:** Row-pair runs split into even-`pair_y`-aligned row bands over active candidate `py_mask[16]` spans, 128-byte aligned `extrema_acc` (`lsl #7`), strict `>` tie-breaking merge, positive `P[i]` + 8-byte `packed_box` (`paired` split), `memset(det->slot, 0xFF)`, and `int16x8_t` NEON extrema (`lw, lh <= 10900`).
3. **Split `threshold_index_{ordinary,shadow,split}` (`NAPA_NOINLINE`):** Moved `masks/prefixes/flags` to `label_scratch`, register-cached `join_above`, and hoisted `prank` check in `label_row_pair`.
4. **`rough_pick` & `profile_fixed` / `fit_sides_impl`:** 15-diagonal NEON `(dot << 3) | lane` branchless min/max + 16-vertex `int16` cyclic refinement (`maxx/maxy <= 8000`); unrolled `K=11/15` in `profile_fixed` + on-demand `gj` gradient in `fit_sides_impl` `jl/jr` lobe walk (keeping scalar `K==11` `initial_plateau_peak` with `inv_p11`).
5. **`sample_points_impl`, `pair_diffs`/`floor_row_regions`, `quiet_zone_passes`:** Non-FMA NEON `vdivq_f64` overlap, `int64_t` offsets, `uint16_t` pixel-pair loads, and NEON histogram reduction.
6. **Compiler A/B Discovery (GCC `12.2.0`, `-mcpu=cortex-a76`):** On the split `threshold_index_*` code, plain `-O3` (`o3_nolto`, `3.1859 ms` `sh16 t=2`) beat both `-flto=4 -fprofile-use` (`3.2203 ms`) and non-LTO PGO (`3.2006 ms` `sh16 t=2`, `4.5918 ms` `sh16 t=1`).

### Direct V4L2 + Rockchip MPP (`wc_pkt_nodup_copy` in `camera/v4l2_mpp.c`)

- Uncachable/WC `pkt_grp` (`pkt_copy_sync`: `68.8 -> 10.7 us`), persistent `MppPacket` reuse, removed redundant user-space `mpp_buffer_sync_ro_partial_begin` (already performed by `mpp_get_frame`), and kept cached `out_memcpy` into caller buffers (skipping copy via DMA zero-copy `wc_pkt_zc` hurt NAPA L2/L3 warmth).
- **1-cam decode (`baseline_copy` → `wc_pkt_nodup_copy`):** `atcam4_real` `1617.9 -> 1514.3 us` (`mae=0.00946`, `maxdiff=1`), `rv_cam2` `1383.2 -> 1208.4 us` (`mae=0.00733`), `rv_cam4` `1631.0 -> 1392.4 us` (`mae=0.01069`).
- **3-cam concurrent MPP + NAPA `sh16 t=2`:** **`486.05 -> 498.67 agg fps`** (`+12.62 fps`; decode `2.214/1.963/2.046 -> 2.012/1.757/1.969 ms`).

## Verified Benchmarks & Parity Gates

| Platform & Mode | Results / Compute | Notes |
|---|---|---|
| **OPi5+ Phase 12 `nbench`** (`2026-11-26`, `frames.bin`, `440` frames, `582` `sh16` tags, `1800/2304/2352 MHz`, `dmc=2400 MHz`) | `sh16 t=1`: `4.9696 ms` (base) / `4.8659 ms` (`.pre_p12_bak` PGO) → **`4.6150 ms`** (`o3_nolto`; **`4.5918 ms`** `pgo_nolto`)<br>`sh16 t=2`: `3.4862 ms` / `3.3794 ms` → **`3.1859 ms`** (`-5.7%` vs `.pre_p12_bak` PGO, `-8.6%` vs base)<br>`off t=2`: `2.1149` → **`1.6327 ms`** (`-22.8%`, `572` tags)<br>`sh64 t=2`: `2.3734` → **`1.8444 ms`** (`-22.3%`, `573` tags) | `0.000000000 px`, `0` mismatches across all modes |
| **OPi5+ Phase 12 `bench_shadow16.py`** (`results_shadow16_opt_v4.json`, `--verify-gt`, `--compare-lib ...pre_p12_bak`) | Head-to-head `sh16 t=2`: `real_cam2` `2.565 → 2.411 ms`, `real_cam4` `4.053 → 3.875 ms`, `atcam4_real` `3.899 → 3.706 ms`, `ycbcr_synth` `2.795 → 2.609 ms`<br>`PROFILE sh16 t=2`: `real_videos_shadow` `WALL_MS=3.155 ms` (`192` tags), `atcam4_real` `WALL_MS=3.7299 ms` (`120` tags) | `VERIFY_GT`: `max_corner_diff_px=0.000000000`, `mismatches=0`; `COMPARE_LIB`: `sh16 max_diff=0.000000000`, `mismatches=0` |
| **OPi5+ Phase 12 Sustained Replay** (`sh16` / `sh64` / `off`, `t=2`) | 1-cam `sh16 t=2`: **`225.58 fps`** (`mixed`, `2.944 ms` detect), **`277.69 fps`** (`cam2`), **`188.19 fps`** (`cam4`)<br>3-cam sustained `t=2`: **`488.13 agg fps`** (`sh16`), **`728.09 agg fps`** (`sh64`), **`782.65 agg fps`** (`off`); 3-cam concurrent MPP + NAPA `sh16 t=2`: **`498.67 agg fps`** | `0` errors, max `55.5 °C` |
| **OPi5+ Native C Pipeline (`libnative_vision.so`)** (`2026-11-26`, `results_native_pipeline.json`, `1280×800`, `sh16`) | **3-cam pure C:** **`505.42 agg fps`** (`sync t=2`), `453.56` (`sync t=1` pinned), **`663.24 agg fps`** (`async_vpu=1 t=2`), `604.93` (`async t=1`)<br>**3-cam Python + `NativeVisionWorker`:** **`506.02 agg fps`** (`sync t=2`), **`625.90 agg fps`** (`async t=2`)<br>**1-cam C:** `cam2` `279.38` / **`422.98`**, `cam4` `186.85` / **`263.75`**, `atcam4` `185.80` / **`277.27 FPS`**<br>**C Newton pose (`nv_solve_apriltags`):** `3.84 / 4.97 / 9.56 us` ctypes (`1–4 us` in-C; `9.88–10.57×` vs `37.91–101.04 us` Py+cv2)<br>**3× 15 FPS A55 preview concurrent:** **`500.10 agg fps`** (`-1.0%`), preview `7.14 → 6.12 ms`<br>**2 MP (`1920×1080`) C scaling:** `307.95` (`sync t=2 sh16`) / **`400.37 agg fps`** (`async t=2 sh16`), `368.69` (`sync sh24`) | `0.000000000 px` gray parity; pose `max_dr_mm <= 3.07e-8`, `max_dtheta <= 4.19e-12`; `1800/2304/2352 MHz`, `dmc=2400 MHz`, max `52.7 °C` |
| **OPi5 `funkyvision4` Live** (Phase 11 `optimal` + pre-P12 `sh16` PGO, `2026-11-24`) | `ATCam1`: **120.51–120.81** / `3.83–4.19 ms`<br>`ATCam2`: **120.16–120.77** / `4.35–4.36 ms` (`[9, 10]`)<br>`ATCam4`: **119.83–120.39** / `5.15–5.17 ms` (`[9, 10]`) | Max `57.3 °C`, `tracebacks=0` (Phase 12 libs not yet synced to OPi5) |
| **OPi5 Offline `sh16`** (Phase 11, `t=2`, A76 `{4..7}`, `lens_on`) | `train_at1`: `3.659 ms` (`-36.7%`), `train_at2`: `3.837 ms` (`-35.2%`), `train_at4`: `4.147 ms` (`-40.1%`) | **432.88 agg FPS** (`cam2/4`: `2.399 / 4.048 ms`) |

**Mandatory Gates Before Promoting Any Native Build:**
- **MPP parity:** `rc=0`, `mae < 0.05`, `maxdiff <= 1` vs `cv2.imdecode` across real camera frames (`0.00724–0.01141` verified).
- **NAPA `sh16` parity:** `max_corner_diff_px == 0.000000000`, `mismatches == 0`, `t=1` vs `t=2` `0.000000000 px`, `0` dropped shadow tags, `0` false positives vs stock pip `napa_vision 0.6.1` (`lens_off` & `lens_on`).
- **Hang protection:** Wrap every remote C/pthread test with `timeout 120`.

## Known Dead Ends (Do Not Repeat)

- `CAP_PROP_BUFFERSIZE=1`: caps `1280×800` MJPG UVC capture at `~68 FPS`.
- `min_tag_px=24`: drops `97%` of live `ATCam4` tags (`68` vs `2,318` at `16 px`).
- `split_touching=2` in production: adds unnecessary `~8 ms` second AprilGrid pass (`split_touching=0` drops `0` FRC tags).
- `roi_y=160, h=480`: misses `100%` of low-slung `21°`-pitched `ATCam4` tags below `y=640`.
- `uvcvideo quirks=128`: does not bypass MJPG `altsetting 6` bandwidth reservation.
- Ad-hoc atomic `pool_drain` replacement: deadlocked at `0% CPU`; keep standard `pthread` mutex/condvar synchronization.
- `-flto=4` + PGO code layout on split `threshold_index_*` under GCC `12.2.0` (`3.2203 ms` vs `3.1859 ms` plain `-O3` on `sh16 t=2`).
- `pair_touch` / `pair_touch64` monotonic cursor (loses to O(1) `edges_before`).
- `-1` in `rank[]` / `bits_any` precheck.
- NEON `vmin_u16` `packed_box` cross-bank moves.
- NEON `initial_plateau_peak` (stack spills; keep scalar `K==11` with `inv_p11`).
- `rough_pick` scalar 4-vertex sweep.
- 4-slot DMA zero-copy `wc_pkt_zc` skipping `out_memcpy` (hurts NAPA L2/L3 cache warmth).
- `MPP_SET_DISABLE_THREAD` (`rc=-7`).
