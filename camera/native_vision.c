/*
 * Native C vision pipeline for RK3588 / Orange Pi 5:
 *   V4L2 MMAP (or MJPG replay) -> Rockchip MPP hardware MJPEG decode (WC packet,
 *   cache-warming memcpy into 8-slot ring) -> NAPA v0.6.1 AprilTag (dlopen) ->
 *   analytic 2D Newton-Raphson Brown-Conrady undistortion + ground-plane pose.
 *
 * Build on RK3588:
 *   gcc -shared -fPIC -O3 -mcpu=cortex-a76 -fno-math-errno -fno-trapping-math \
 *       -std=c99 -Wall -Wextra -Werror MonkeySee/camera/native_vision.c \
 *       -I/usr/local/include/rockchip -L/usr/local/lib \
 *       -lrockchip_mpp -lpthread -ldl -lm -o /usr/local/lib/libnative_vision.so
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <math.h>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#include <rockchip/mpp_buffer.h>
#include <rockchip/mpp_frame.h>
#include <rockchip/mpp_meta.h>
#include <rockchip/mpp_packet.h>
#include <rockchip/rk_mpi.h>
#include <rockchip/rk_mpi_cmd.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define NV_RING_SLOTS 8
#define NV_MAX_TAGS 64
#define NV_VALID_WORDS 10 /* covers tag36h11 IDs 0..639 */
#define NV_MAX_FRAME_AGE_NS 1000000000LL
#define NV_DEFAULT_NAPA_LIB "/usr/local/lib/libnapa_v061_a76_neon2_pgo.so"

/* ---- Exact NAPA C ABI ---- */

typedef struct {
    int min_tag_px;
    int max_hamming;
    int min_contrast;
    int min_border_cells;
    int sharpening_x100;
    int split_touching;
} NvNapaParams;

typedef struct {
    int id;
    int hamming;
    float decision_margin;
    double corners[8]; /* NAPA order: BL, BR, TR, TL */
} NvNapaDetection;

typedef void *(*nv_napa_create_fn)(const char *);
typedef void (*nv_napa_destroy_fn)(void *);
typedef int (*nv_napa_set_threads_fn)(void *, int);
typedef int (*nv_napa_set_lens_fn)(void *, int, const double *, const double *, int);
typedef int (*nv_napa_detect_fn)(
    void *, const uint8_t *, int, int, int, const NvNapaParams *, NvNapaDetection *, int
);
typedef int (*nv_napa_detect_sh_fn)(
    void *, const uint8_t *, int, int, int, const NvNapaParams *, NvNapaDetection *, int, int
);
typedef int (*nv_napa_set_mask_fn)(void *, uint64_t);

typedef struct {
    void *handle;
    nv_napa_create_fn create_family;
    nv_napa_destroy_fn destroy;
    nv_napa_set_threads_fn set_threads;
    nv_napa_set_lens_fn set_lens;
    nv_napa_detect_fn detect;
    nv_napa_detect_sh_fn detect_sh;
    nv_napa_set_mask_fn set_mask;
} NvNapaApi;

/* ---- Public Structs ---- */

typedef struct {
    double fx, fy, cx, cy;
    double dist[5]; /* k1, k2, p1, p2, k3 */
    double yaw_deg;
    double pitch_deg;
    double roll_deg;
    double tag_height;
    int has_cal;
    int filter_valid_ids;
    uint64_t valid_bits[NV_VALID_WORDS];
} NvPoseConfig;

typedef struct {
    double r_in;
    double theta_deg;
    int tag_id;
} NvPoseOut;

typedef struct {
    int min_tag_px;
    int max_hamming;
    int min_contrast;
    int min_border_cells;
    int sharpening_x100;
    int split_touching;
    int napa_threads;
    int recover_shadows;
    int shadow_min_tag_px;
    int apply_lens;
    int async_vpu;
    uint32_t big_cpu_mask;    /* bitmask of A76 cores (0 = inherit) */
    uint32_t little_cpu_mask; /* bitmask of A55 cores (0 = inherit) */
    int nice_level;
} NvPipelineConfig;

typedef struct {
    uint64_t seq;
    int ring_slot;
    int width;
    int height;
    int64_t capture_ns;
    int64_t read_ns;
    int64_t done_ns;
    double decode_ms;
    double detect_ms;
    double pose_ms;
    int n_raw_tags;
    int raw_ids[NV_MAX_TAGS];
    double raw_corners[NV_MAX_TAGS * 8]; /* MonkeySee TR, TL, BL, BR */
    int n_poses;
    NvPoseOut poses[NV_MAX_TAGS];
    int jpeg_len;
    int dropped_v4l2;
} NvFrameResult;

typedef struct {
    int frames_ok;
    int frames_err;
    int tags_total;
    int poses_total;
    double wall_s;
    double fps;
    double mean_total_ms;
    double p50_total_ms;
    double p95_total_ms;
    double max_total_ms;
    double mean_decode_ms;
    double mean_detect_ms;
    double mean_pose_ms;
} NvBenchStats;

/* ---- Time, Affinity & Sort Helpers ---- */

static inline int64_t monotonic_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

static inline double ns_to_ms(int64_t dt_ns) {
    return (double)dt_ns * 1e-6;
}

static inline int align_up_int(int v, int a) {
    return (v + a - 1) & ~(a - 1);
}

static void apply_thread_affinity(uint32_t cpu_mask, int nice_level) {
    if (cpu_mask != 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        for (int cpu = 0; cpu < 32; ++cpu) {
            if (cpu_mask & (1u << cpu))
                CPU_SET(cpu, &set);
        }
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
    if (nice_level != 0) {
        setpriority(PRIO_PROCESS, 0, nice_level);
    }
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

static double pct_sorted(const double *arr, int n, double p) {
    if (n <= 0) return 0.0;
    if (n == 1) return arr[0];
    double idx = p * (double)(n - 1);
    int lo = (int)idx;
    int hi = lo + 1;
    if (hi >= n) return arr[n - 1];
    double frac = idx - (double)lo;
    return arr[lo] * (1.0 - frac) + arr[hi] * frac;
}

static void summarize_latencies(
    double *samples, int n, double *mean_out, double *p50_out, double *p95_out, double *max_out
) {
    if (n <= 0 || !samples) {
        if (mean_out) *mean_out = 0.0;
        if (p50_out) *p50_out = 0.0;
        if (p95_out) *p95_out = 0.0;
        if (max_out) *max_out = 0.0;
        return;
    }
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += samples[i];
    qsort(samples, (size_t)n, sizeof(double), cmp_double);
    if (mean_out) *mean_out = sum / (double)n;
    if (p50_out) *p50_out = pct_sorted(samples, n, 0.50);
    if (p95_out) *p95_out = pct_sorted(samples, n, 0.95);
    if (max_out) *max_out = samples[n - 1];
}

/* ---- Shared NAPA Loader ---- */

static pthread_mutex_t g_napa_mu = PTHREAD_MUTEX_INITIALIZER;
static NvNapaApi g_napa_api;
static char g_napa_path[512];

static int load_napa_api(const char *lib_path, NvNapaApi *out) {
    const char *path = (lib_path && lib_path[0]) ? lib_path : NV_DEFAULT_NAPA_LIB;
    pthread_mutex_lock(&g_napa_mu);
    if (g_napa_api.handle && strcmp(g_napa_path, path) == 0) {
        *out = g_napa_api;
        pthread_mutex_unlock(&g_napa_mu);
        return 0;
    }

    void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        pthread_mutex_unlock(&g_napa_mu);
        return -1;
    }

    NvNapaApi api;
    memset(&api, 0, sizeof(api));
    api.handle = h;
    api.create_family = (nv_napa_create_fn)dlsym(h, "napa_create_family");
    api.destroy = (nv_napa_destroy_fn)dlsym(h, "napa_destroy");
    api.set_threads = (nv_napa_set_threads_fn)dlsym(h, "napa_set_threads");
    api.set_lens = (nv_napa_set_lens_fn)dlsym(h, "napa_set_lens");
    api.detect = (nv_napa_detect_fn)dlsym(h, "napa_detect");
    api.detect_sh = (nv_napa_detect_sh_fn)dlsym(h, "napa_detect_shadows_configured");
    api.set_mask = (nv_napa_set_mask_fn)dlsym(h, "napa_set_tag_id_mask");

    if (!api.create_family || !api.destroy || !api.set_threads || !api.detect) {
        dlclose(h);
        pthread_mutex_unlock(&g_napa_mu);
        return -2;
    }

    /* Cache primary library handle without closing prior one if a different path is requested later. */
    g_napa_api = api;
    snprintf(g_napa_path, sizeof(g_napa_path), "%s", path);
    *out = api;
    pthread_mutex_unlock(&g_napa_mu);
    return 0;
}

/* ---- Exact 2D Newton-Raphson Brown-Conrady Undistortion + Pose ---- */

static inline int is_valid_tag_id(const NvPoseConfig *cfg, int tag_id) {
    if (!cfg->filter_valid_ids)
        return 1;
    if (tag_id < 0 || tag_id >= NV_VALID_WORDS * 64)
        return 0;
    return (cfg->valid_bits[tag_id >> 6] & (1ULL << (tag_id & 63))) != 0;
}

static inline void undistort_point_fixed_fallback(
    double u, double v, const NvPoseConfig *cfg, double *ox, double *oy
) {
    double fx = cfg->fx, fy = cfg->fy, cx = cfg->cx, cy = cfg->cy;
    double k1 = cfg->dist[0], k2 = cfg->dist[1], p1 = cfg->dist[2], p2 = cfg->dist[3], k3 = cfg->dist[4];
    double x0 = (u - cx) / fx;
    double y0 = (v - cy) / fy;
    double x = x0, y = y0;

    for (int it = 0; it < 20; ++it) {
        double r2 = x * x + y * y;
        double icdist = 1.0 / (1.0 + ((k3 * r2 + k2) * r2 + k1) * r2);
        if (icdist < 0.0) {
            x = x0;
            y = y0;
            break;
        }
        double dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        double dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
        x = (x0 - dx) * icdist;
        y = (y0 - dy) * icdist;

        double r2p = x * x + y * y;
        double cdistp = 1.0 + ((k3 * r2p + k2) * r2p + k1) * r2p;
        double xd = x * cdistp + 2.0 * p1 * x * y + p2 * (r2p + 2.0 * x * x);
        double yd = y * cdistp + p1 * (r2p + 2.0 * y * y) + 2.0 * p2 * x * y;
        double ex = xd * fx + cx - u;
        double ey = yd * fy + cy - v;
        if (ex * ex + ey * ey < 1e-20)
            break;
    }
    *ox = x;
    *oy = y;
}

static inline void undistort_point_newton(
    double u, double v, const NvPoseConfig *cfg, double *ox, double *oy
) {
    double fx = cfg->fx, fy = cfg->fy;
    double k1 = cfg->dist[0], k2 = cfg->dist[1], p1 = cfg->dist[2], p2 = cfg->dist[3], k3 = cfg->dist[4];
    double x0 = (u - cfg->cx) / fx;
    double y0 = (v - cfg->cy) / fy;

    if (k1 == 0.0 && k2 == 0.0 && p1 == 0.0 && p2 == 0.0 && k3 == 0.0) {
        *ox = x0;
        *oy = y0;
        return;
    }

    double x = x0, y = y0;
    int converged = 0;
    for (int it = 0; it < 8; ++it) {
        double xx = x * x, yy = y * y, xy = x * y;
        double r2 = xx + yy;
        double c = 1.0 + ((k3 * r2 + k2) * r2 + k1) * r2;
        double fx_err = x * c + 2.0 * p1 * xy + p2 * (r2 + 2.0 * xx) - x0;
        double fy_err = y * c + p1 * (r2 + 2.0 * yy) + 2.0 * p2 * xy - y0;

        /* Normalized tol 1e-13 (~1e-10 px at fx~1000) before Jacobian step */
        if (fx_err * fx_err + fy_err * fy_err < 1e-26) {
            converged = 1;
            break;
        }

        double dr = 2.0 * (k1 + r2 * (2.0 * k2 + 3.0 * k3 * r2));
        double jxx = c + xx * dr + 2.0 * p1 * y + 6.0 * p2 * x;
        double jxy = xy * dr + 2.0 * p1 * x + 2.0 * p2 * y;
        double jyy = c + yy * dr + 6.0 * p1 * y + 2.0 * p2 * x;

        double det = jxx * jyy - jxy * jxy;
        if (fabs(det) < 1e-12)
            break;

        double inv_det = 1.0 / det;
        double dx = (jyy * fx_err - jxy * fy_err) * inv_det;
        double dy = (jxx * fy_err - jxy * fx_err) * inv_det;
        x -= dx;
        y -= dy;

        if (!isfinite(x) || !isfinite(y)) {
            converged = 0;
            break;
        }
        if (dx * dx + dy * dy < 1e-26) {
            converged = 1;
            break;
        }
    }

    if (converged && isfinite(x) && isfinite(y)) {
        *ox = x;
        *oy = y;
        return;
    }
    undistort_point_fixed_fallback(u, v, cfg, ox, oy);
}

int nv_solve_apriltags(
    const double *corners_flat,
    const int *ids,
    int n_tags,
    const NvPoseConfig *cfg,
    NvPoseOut *out_poses,
    int max_out
) {
    if (!corners_flat || !ids || !cfg || !out_poses || n_tags <= 0 || max_out <= 0)
        return 0;
    if (!cfg->has_cal || cfg->fx == 0.0 || cfg->fy == 0.0)
        return 0;

    double pitch = cfg->pitch_deg * (M_PI / 180.0);
    double roll = cfg->roll_deg * (M_PI / 180.0);
    double cos_p = cos(pitch), sin_p = sin(pitch);
    double cos_r = cos(roll), sin_r = sin(roll);
    int has_roll = (roll != 0.0);

    int kept = 0;
    for (int t = 0; t < n_tags && kept < max_out; ++t) {
        int tid = ids[t];
        if (!is_valid_tag_id(cfg, tid))
            continue;

        const double *c = corners_flat + (size_t)t * 8u;
        double ux[4], uy[4];
        for (int i = 0; i < 4; ++i) {
            undistort_point_newton(c[i * 2], c[i * 2 + 1], cfg, &ux[i], &uy[i]);
            if (has_roll) {
                double rx = ux[i] * cos_r - uy[i] * sin_r;
                double ry = ux[i] * sin_r + uy[i] * cos_r;
                ux[i] = rx;
                uy[i] = ry;
            }
        }

        double f0 = cos_p + uy[0] * sin_p;
        double f1 = cos_p + uy[1] * sin_p;
        double f2 = cos_p + uy[2] * sin_p;
        double f3 = cos_p + uy[3] * sin_p;

        double l_i = fabs((sin_p - uy[0] * cos_p) / f0 - (sin_p - uy[3] * cos_p) / f3);
        double r_i = fabs((sin_p - uy[1] * cos_p) / f1 - (sin_p - uy[2] * cos_p) / f2);
        if (fmin(l_i, r_i) < 1e-9)
            continue;

        double z_left = cfg->tag_height / l_i;
        double z_right = cfg->tag_height / r_i;
        double x_center = (z_left * (ux[0] / f0 + ux[3] / f3) + z_right * (ux[1] / f1 + ux[2] / f2)) * 0.25;
        double z_center = (z_left + z_right) * 0.5;

        out_poses[kept].r_in = hypot(x_center, z_center);
        out_poses[kept].theta_deg = atan2(x_center, z_center) * (180.0 / M_PI) + cfg->yaw_deg;
        out_poses[kept].tag_id = tid;
        kept++;
    }
    return kept;
}

/* ---- Rockchip MPP Hardware MJPEG Decoder (Phase 12 wc_pkt_nodup_copy) ---- */

typedef struct {
    MppCtx ctx;
    MppApi *mpi;
    MppDecCfg dec_cfg;
    MppBufferGroup frm_grp;
    MppBufferGroup pkt_grp;
    MppBuffer frm_buf;
    MppBuffer pkt_buf;
    void *pkt_ptr;
    MppPacket packet;
    MppFrame frame;
    size_t max_pkt_size;
    size_t max_frm_size;
    int max_w;
    int max_h;
} NvMppDec;

static void mpp_dec_destroy(NvMppDec *m) {
    if (!m) return;
    if (m->mpi && m->ctx)
        m->mpi->reset(m->ctx);
    if (m->packet)
        mpp_packet_deinit(&m->packet);
    if (m->frame)
        mpp_frame_deinit(&m->frame);
    if (m->frm_buf)
        mpp_buffer_put(m->frm_buf);
    if (m->pkt_buf)
        mpp_buffer_put(m->pkt_buf);
    if (m->frm_grp)
        mpp_buffer_group_put(m->frm_grp);
    if (m->pkt_grp)
        mpp_buffer_group_put(m->pkt_grp);
    if (m->dec_cfg)
        mpp_dec_cfg_deinit(m->dec_cfg);
    if (m->ctx)
        mpp_destroy(m->ctx);
    free(m);
}

static NvMppDec *mpp_dec_create(int max_w, int max_h) {
    if (max_w <= 0 || max_h <= 0)
        return NULL;
    NvMppDec *m = (NvMppDec *)calloc(1, sizeof(NvMppDec));
    if (!m) return NULL;
    m->max_w = max_w;
    m->max_h = max_h;

    int hor_stride = align_up_int(max_w, 64);
    int ver_stride = align_up_int(max_h, 64);
    m->max_frm_size = (size_t)hor_stride * (size_t)ver_stride * 4u;
    m->max_pkt_size = (size_t)max_w * (size_t)max_h * 2u;
    if (m->max_pkt_size < (2u << 20))
        m->max_pkt_size = (2u << 20);

    /* Uncachable WC pkt_grp + cachable frm_grp (Phase 12 wc_pkt_nodup_copy). */
    RK_U32 frm_type = MPP_BUFFER_TYPE_DMA_HEAP | MPP_BUFFER_FLAGS_CACHABLE;
    RK_U32 pkt_type = MPP_BUFFER_TYPE_DMA_HEAP;

    if (mpp_buffer_group_get_internal(&m->frm_grp, frm_type) != MPP_OK || !m->frm_grp) goto fail;
    if (mpp_buffer_group_get_internal(&m->pkt_grp, pkt_type) != MPP_OK || !m->pkt_grp) goto fail;
    if (mpp_buffer_get(m->frm_grp, &m->frm_buf, m->max_frm_size) != MPP_OK || !m->frm_buf) goto fail;
    if (mpp_buffer_get(m->pkt_grp, &m->pkt_buf, m->max_pkt_size) != MPP_OK || !m->pkt_buf) goto fail;

    m->pkt_ptr = mpp_buffer_get_ptr(m->pkt_buf);
    if (!m->pkt_ptr) goto fail;
    if (mpp_packet_init_with_buffer(&m->packet, m->pkt_buf) != MPP_OK || !m->packet) goto fail;
    if (mpp_frame_init(&m->frame) != MPP_OK || !m->frame) goto fail;
    mpp_frame_set_buffer(m->frame, m->frm_buf);

    if (mpp_create(&m->ctx, &m->mpi) != MPP_OK || !m->ctx || !m->mpi) goto fail;
    if (mpp_init(m->ctx, MPP_CTX_DEC, MPP_VIDEO_CodingMJPEG) != MPP_OK) goto fail;

    if (mpp_dec_cfg_init(&m->dec_cfg) == MPP_OK && m->dec_cfg) {
        if (m->mpi->control(m->ctx, MPP_DEC_GET_CFG, m->dec_cfg) == MPP_OK) {
            mpp_dec_cfg_set_u32(m->dec_cfg, "base:split_parse", 0);
            m->mpi->control(m->ctx, MPP_DEC_SET_CFG, m->dec_cfg);
        }
    }
    return m;

fail:
    mpp_dec_destroy(m);
    return NULL;
}

static int mpp_dec_pkt_to_gray(
    NvMppDec *m, size_t len, uint8_t *out_y, size_t out_cap, int *out_w, int *out_h
) {
    mpp_packet_set_data(m->packet, m->pkt_ptr);
    mpp_packet_set_size(m->packet, m->max_pkt_size);
    mpp_packet_set_pos(m->packet, m->pkt_ptr);
    mpp_packet_set_length(m->packet, len);

    MppMeta meta = mpp_packet_get_meta(m->packet);
    if (meta)
        mpp_meta_set_frame(meta, KEY_OUTPUT_FRAME, m->frame);

    if (m->mpi->decode_put_packet(m->ctx, m->packet) != MPP_OK)
        return -13;

    MppFrame frame_ret = NULL;
    MPP_RET ret = m->mpi->decode_get_frame(m->ctx, &frame_ret);
    if (ret != MPP_OK || !frame_ret) {
        if (frame_ret && frame_ret != m->frame)
            mpp_frame_deinit(&frame_ret);
        return -14;
    }
    if (mpp_frame_get_errinfo(frame_ret)) {
        if (frame_ret != m->frame)
            mpp_frame_deinit(&frame_ret);
        return -15;
    }

    MppBuffer out_buf = mpp_frame_get_buffer(frame_ret);
    if (!out_buf) {
        if (frame_ret != m->frame)
            mpp_frame_deinit(&frame_ret);
        return -16;
    }

    int w = (int)mpp_frame_get_width(frame_ret);
    int h = (int)mpp_frame_get_height(frame_ret);
    int stride = (int)mpp_frame_get_hor_stride(frame_ret);
    const uint8_t *y_ptr = (const uint8_t *)mpp_buffer_get_ptr(out_buf);
    if (frame_ret != m->frame)
        mpp_frame_deinit(&frame_ret);

    if (!y_ptr || w <= 0 || h <= 0 || stride < w)
        return -17;

    size_t y_bytes = (size_t)stride * (size_t)h;
    if (!out_y || out_cap < (size_t)w * (size_t)h) {
        if (y_bytes > 0)
            mpp_buffer_sync_ro_partial_end(out_buf, 0, (RK_U32)y_bytes);
        return -18;
    }

    /* Cached memcpy warms L2/L3 right before full-frame NAPA scan. */
    if (stride == w) {
        memcpy(out_y, y_ptr, (size_t)w * (size_t)h);
    } else {
        for (int r = 0; r < h; ++r)
            memcpy(out_y + (size_t)r * (size_t)w, y_ptr + (size_t)r * (size_t)stride, (size_t)w);
    }

    if (y_bytes > 0)
        mpp_buffer_sync_ro_partial_end(out_buf, 0, (RK_U32)y_bytes);

    *out_w = w;
    *out_h = h;
    return 0;
}

static int mpp_dec_decode_gray(
    NvMppDec *m, const uint8_t *src, size_t len, uint8_t *out_y, size_t out_cap, int *out_w, int *out_h
) {
    if (!m || !src || len == 0 || len > m->max_pkt_size)
        return -11;
    memcpy(m->pkt_ptr, src, len);
    return mpp_dec_pkt_to_gray(m, len, out_y, out_cap, out_w, out_h);
}

/* ---- V4L2 Helper ---- */

typedef struct {
    void *start;
    size_t length;
} NvV4L2Buf;

typedef struct {
    int fd;
    int streaming;
    int width;
    int height;
    double fps;
    NvV4L2Buf *bufs;
    unsigned int n_bufs;
} NvV4L2Cam;

static int xioctl(int fd, unsigned long req, void *arg) {
    int r;
    do {
        r = ioctl(fd, req, arg);
    } while (r == -1 && errno == EINTR);
    return r;
}

static void v4l2_close_dev(NvV4L2Cam *v) {
    if (!v) return;
    if (v->fd >= 0 && v->streaming) {
        enum v4l2_buf_type btype = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(v->fd, VIDIOC_STREAMOFF, &btype);
        v->streaming = 0;
    }
    if (v->bufs) {
        for (unsigned int i = 0; i < v->n_bufs; ++i) {
            if (v->bufs[i].start)
                munmap(v->bufs[i].start, v->bufs[i].length);
        }
        free(v->bufs);
        v->bufs = NULL;
        v->n_bufs = 0;
    }
    if (v->fd >= 0) {
        close(v->fd);
        v->fd = -1;
    }
}

static int v4l2_open_dev(NvV4L2Cam *v, const char *dev_path, int width, int height, int fps, int req_bufs) {
    memset(v, 0, sizeof(*v));
    v->fd = -1;
    if (!dev_path || width <= 0 || height <= 0)
        return -1;

    v->fd = open(dev_path, O_RDWR | O_NONBLOCK);
    if (v->fd < 0)
        goto fail;

    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = (uint32_t)width;
    fmt.fmt.pix.height = (uint32_t)height;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    fmt.fmt.pix.field = V4L2_FIELD_ANY;
    if (xioctl(v->fd, VIDIOC_S_FMT, &fmt) < 0)
        goto fail;
    if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_MJPEG)
        goto fail;

    v->width = (int)fmt.fmt.pix.width;
    v->height = (int)fmt.fmt.pix.height;
    if (v->width <= 0 || v->height <= 0)
        goto fail;

    if (fps > 0) {
        struct v4l2_streamparm sp;
        memset(&sp, 0, sizeof(sp));
        sp.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        sp.parm.capture.timeperframe.numerator = 1;
        sp.parm.capture.timeperframe.denominator = (uint32_t)fps;
        if (xioctl(v->fd, VIDIOC_S_PARM, &sp) == 0 &&
            sp.parm.capture.timeperframe.numerator > 0 &&
            sp.parm.capture.timeperframe.denominator > 0) {
            v->fps = (double)sp.parm.capture.timeperframe.denominator /
                     (double)sp.parm.capture.timeperframe.numerator;
        } else {
            v->fps = (double)fps;
        }
    }

    int target_bufs = (req_bufs > 0) ? req_bufs : 2;
    if (target_bufs > 8) target_bufs = 8;

    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = (uint32_t)target_bufs;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(v->fd, VIDIOC_REQBUFS, &req) < 0 || req.count < 1)
        goto fail;

    v->n_bufs = req.count;
    v->bufs = (NvV4L2Buf *)calloc(v->n_bufs, sizeof(NvV4L2Buf));
    if (!v->bufs)
        goto fail;

    for (unsigned int i = 0; i < v->n_bufs; ++i) {
        struct v4l2_buffer b;
        memset(&b, 0, sizeof(b));
        b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        b.memory = V4L2_MEMORY_MMAP;
        b.index = i;
        if (xioctl(v->fd, VIDIOC_QUERYBUF, &b) < 0)
            goto fail;
        v->bufs[i].length = b.length;
        v->bufs[i].start = mmap(NULL, b.length, PROT_READ | PROT_WRITE, MAP_SHARED, v->fd, b.m.offset);
        if (v->bufs[i].start == MAP_FAILED) {
            v->bufs[i].start = NULL;
            goto fail;
        }
        if (xioctl(v->fd, VIDIOC_QBUF, &b) < 0)
            goto fail;
    }

    enum v4l2_buf_type btype = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(v->fd, VIDIOC_STREAMON, &btype) < 0)
        goto fail;
    v->streaming = 1;
    return 0;

fail:
    v4l2_close_dev(v);
    return -1;
}

static int v4l2_grab_into_mpp(
    NvV4L2Cam *v,
    NvMppDec *mpp,
    int timeout_ms,
    size_t *out_len,
    int64_t *out_cap_ns,
    int *out_dropped,
    uint8_t *opt_jpeg_dst,
    size_t opt_jpeg_cap
) {
    *out_len = 0;
    *out_dropped = 0;
    int64_t deadline_ns = (timeout_ms > 0) ? (monotonic_now_ns() + (int64_t)timeout_ms * 1000000LL) : 0;
    struct v4l2_buffer vb;

    for (;;) {
        int wait_ms = timeout_ms;
        if (timeout_ms > 0) {
            int64_t rem_ns = deadline_ns - monotonic_now_ns();
            if (rem_ns <= 0)
                return -2;
            wait_ms = (int)((rem_ns + 999999LL) / 1000000LL);
        }

        struct pollfd pfd = { .fd = v->fd, .events = POLLIN, .revents = 0 };
        int pr = poll(&pfd, 1, wait_ms);
        if (pr < 0) {
            if (errno == EINTR) continue;
            return -3;
        }
        if (pr == 0) return -2;
        if (pfd.revents & (POLLERR | POLLNVAL | POLLHUP)) return -3;

        memset(&vb, 0, sizeof(vb));
        vb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        vb.memory = V4L2_MEMORY_MMAP;
        if (xioctl(v->fd, VIDIOC_DQBUF, &vb) < 0) {
            if (errno == EAGAIN) continue;
            return -4;
        }
        break;
    }

    int dropped = 0;
    for (unsigned int d = 0; d < v->n_bufs; ++d) {
        struct v4l2_buffer vb_next;
        memset(&vb_next, 0, sizeof(vb_next));
        vb_next.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        vb_next.memory = V4L2_MEMORY_MMAP;
        if (xioctl(v->fd, VIDIOC_DQBUF, &vb_next) == 0) {
            if (vb_next.index < v->n_bufs && vb_next.bytesused >= 4 && !(vb_next.flags & V4L2_BUF_FLAG_ERROR)) {
                xioctl(v->fd, VIDIOC_QBUF, &vb);
                vb = vb_next;
                dropped++;
            } else {
                xioctl(v->fd, VIDIOC_QBUF, &vb_next);
            }
        } else {
            break;
        }
    }
    *out_dropped = dropped;

    if (vb.index >= v->n_bufs || vb.bytesused < 4 || (vb.flags & V4L2_BUF_FLAG_ERROR) ||
        (size_t)vb.bytesused > v->bufs[vb.index].length) {
        xioctl(v->fd, VIDIOC_QBUF, &vb);
        return -5;
    }

    int64_t now_ns = monotonic_now_ns();
    int64_t ts_ns = now_ns;
    if ((vb.flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) == V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC) {
        int64_t cap_ns = (int64_t)vb.timestamp.tv_sec * 1000000000LL +
                         (int64_t)vb.timestamp.tv_usec * 1000LL;
        if (cap_ns > 0 && now_ns >= cap_ns && (now_ns - cap_ns) < NV_MAX_FRAME_AGE_NS)
            ts_ns = cap_ns;
    }
    *out_cap_ns = ts_ns;

    size_t jlen = (size_t)vb.bytesused;
    const uint8_t *src = (const uint8_t *)v->bufs[vb.index].start;
    if (jlen > mpp->max_pkt_size) {
        xioctl(v->fd, VIDIOC_QBUF, &vb);
        return -6;
    }

    memcpy(mpp->pkt_ptr, src, jlen);
    if (opt_jpeg_dst && opt_jpeg_cap >= jlen) {
        memcpy(opt_jpeg_dst, src, jlen);
        *out_len = jlen;
    } else {
        *out_len = 0;
    }
    if (xioctl(v->fd, VIDIOC_QBUF, &vb) < 0)
        return -8;

    return (int)jlen;
}

/* ---- Core Camera / Replay Worker ---- */

struct NvCamWorker {
    int is_replay;
    int max_w;
    int max_h;
    size_t slot_cap;

    NvNapaApi api;
    void *det;
    NvNapaParams napa_params;
    int napa_threads;
    int recover_shadows;
    int shadow_min_tag_px;
    int apply_lens;

    int async_vpu;
    uint32_t big_cpu_mask;
    uint32_t little_cpu_mask;
    int nice_level;

    NvPoseConfig pose_cfg;
    uint64_t lens_dirty_gen;
    uint64_t lens_applied_gen;

    NvV4L2Cam v4l2;
    pthread_mutex_t ctrl_mu;

    /* Replay source (owned pointers into caller-pinned buffers for replay duration) */
    const uint8_t **rep_jpegs;
    const size_t *rep_lens;
    int rep_count;
    int rep_idx;

    NvMppDec *mpp;
    uint8_t *ring[NV_RING_SLOTS];
    uint8_t *jpeg_ring[NV_RING_SLOTS];
    int jpeg_lens[NV_RING_SLOTS];
    size_t max_pkt_size;
    volatile int want_jpeg;

    /* Async VPU <-> NAPA 2-slot handoff */
    pthread_mutex_t hq_mu;
    pthread_cond_t hq_cv_ready;
    pthread_cond_t hq_cv_free;
    int hq_has_item;
    int hq_slot;
    int hq_w;
    int hq_h;
    int64_t hq_cap_ns;
    int64_t hq_read_ns;
    double hq_dec_ms;
    int hq_dropped;
    int hq_stop;

    /* Consumer -> Python latest published result */
    pthread_mutex_t res_mu;
    pthread_cond_t res_cv;
    NvFrameResult latest;
    int has_error;
    int last_err;

    volatile int running;
    int started;
    pthread_t prod_thr;
    pthread_t cons_thr;
    int prod_started;
    int cons_started;
};

typedef struct NvCamWorker NvCamWorker;

static void sync_worker_lens_locked(NvCamWorker *w) {
    if (w->lens_applied_gen == w->lens_dirty_gen)
        return;
    w->lens_applied_gen = w->lens_dirty_gen;
    if (!w->det || !w->api.set_lens)
        return;

    if (w->apply_lens && w->pose_cfg.has_cal) {
        double intr[5] = { w->pose_cfg.fx, w->pose_cfg.fy, w->pose_cfg.cx, w->pose_cfg.cy, 0.0 };
        if (w->api.set_lens(w->det, 1, intr, w->pose_cfg.dist, 5) != 0)
            w->api.set_lens(w->det, 0, NULL, NULL, 0);
    } else {
        w->api.set_lens(w->det, 0, NULL, NULL, 0);
    }
}

static int run_napa_and_pose_slot(
    NvCamWorker *w,
    int slot,
    int img_w,
    int img_h,
    int64_t cap_ns,
    int64_t read_ns,
    double dec_ms,
    int dropped,
    NvFrameResult *out_res
) {
    pthread_mutex_lock(&w->ctrl_mu);
    sync_worker_lens_locked(w);
    NvPoseConfig pose_cfg = w->pose_cfg;
    NvNapaParams params = w->napa_params;
    int rec_sh = w->recover_shadows;
    int sh_px = w->shadow_min_tag_px;
    pthread_mutex_unlock(&w->ctrl_mu);

    const uint8_t *gray = w->ring[slot];
    NvNapaDetection dets[NV_MAX_TAGS];

    int64_t t_det0 = monotonic_now_ns();
    int n_det;
    if (rec_sh && w->api.detect_sh) {
        n_det = w->api.detect_sh(w->det, gray, img_w, img_h, img_w, &params, dets, NV_MAX_TAGS, sh_px);
        if (n_det < 0)
            n_det = w->api.detect(w->det, gray, img_w, img_h, img_w, &params, dets, NV_MAX_TAGS);
    } else {
        n_det = w->api.detect(w->det, gray, img_w, img_h, img_w, &params, dets, NV_MAX_TAGS);
    }
    int64_t t_det1 = monotonic_now_ns();

    if (n_det < 0)
        n_det = 0;
    if (n_det > NV_MAX_TAGS)
        n_det = NV_MAX_TAGS;

    static const int remap[4] = {2, 3, 0, 1};
    memset(out_res, 0, sizeof(*out_res));
    out_res->ring_slot = slot;
    out_res->width = img_w;
    out_res->height = img_h;
    out_res->capture_ns = cap_ns;
    out_res->read_ns = read_ns;
    out_res->decode_ms = dec_ms;
    out_res->detect_ms = ns_to_ms(t_det1 - t_det0);
    out_res->n_raw_tags = n_det;
    out_res->dropped_v4l2 = dropped;
    out_res->jpeg_len = w->jpeg_lens[slot];

    for (int i = 0; i < n_det; ++i) {
        out_res->raw_ids[i] = dets[i].id;
        double *dst = out_res->raw_corners + (size_t)i * 8u;
        const double *src = dets[i].corners;
        for (int c = 0; c < 4; ++c) {
            int sc = remap[c];
            dst[c * 2 + 0] = src[sc * 2 + 0];
            dst[c * 2 + 1] = src[sc * 2 + 1];
        }
    }

    int64_t t_pose0 = monotonic_now_ns();
    out_res->n_poses = nv_solve_apriltags(
        out_res->raw_corners, out_res->raw_ids, n_det, &pose_cfg, out_res->poses, NV_MAX_TAGS
    );
    int64_t t_pose1 = monotonic_now_ns();
    out_res->done_ns = t_pose1;
    out_res->pose_ms = ns_to_ms(t_pose1 - t_pose0);
    return 0;
}

static void publish_frame_result(NvCamWorker *w, const NvFrameResult *res) {
    pthread_mutex_lock(&w->res_mu);
    uint64_t next_seq = w->latest.seq + 1u;
    w->latest = *res;
    w->latest.seq = next_seq;
    w->has_error = 0;
    pthread_cond_broadcast(&w->res_cv);
    pthread_mutex_unlock(&w->res_mu);
}

static int grab_and_decode_next(
    NvCamWorker *w,
    int slot,
    int *out_w,
    int *out_h,
    int64_t *out_cap_ns,
    int64_t *out_read_ns,
    double *out_dec_ms,
    int *out_dropped
) {
    int img_w = 0, img_h = 0, dropped = 0;
    int64_t cap_ns = 0;
    int64_t t0 = monotonic_now_ns();

    uint8_t *jpg_dst = NULL;
    size_t jpg_cap = 0;
    if (w->want_jpeg && w->jpeg_ring[slot]) {
        jpg_dst = w->jpeg_ring[slot];
        jpg_cap = w->max_pkt_size;
    }

    if (w->is_replay) {
        if (w->rep_count <= 0)
            return -10;
        int j = w->rep_idx;
        w->rep_idx = (j + 1) % w->rep_count;
        cap_ns = t0;
        const uint8_t *src = w->rep_jpegs[j];
        size_t len = w->rep_lens[j];
        if (jpg_dst && len <= jpg_cap) {
            memcpy(jpg_dst, src, len);
            w->jpeg_lens[slot] = (int)len;
        } else {
            w->jpeg_lens[slot] = 0;
        }
        int64_t t_d0 = monotonic_now_ns();
        int rc = mpp_dec_decode_gray(w->mpp, src, len, w->ring[slot], w->slot_cap, &img_w, &img_h);
        int64_t t_d1 = monotonic_now_ns();
        if (rc != 0)
            return rc;
        *out_dec_ms = ns_to_ms(t_d1 - t_d0);
        *out_read_ns = t_d1;
    } else {
        size_t opt_len = 0;
        int pkt_len = v4l2_grab_into_mpp(&w->v4l2, w->mpp, 500, &opt_len, &cap_ns, &dropped, jpg_dst, jpg_cap);
        if (pkt_len <= 0)
            return pkt_len;
        w->jpeg_lens[slot] = (int)opt_len;
        int64_t t_d0 = monotonic_now_ns();
        int rc = mpp_dec_pkt_to_gray(w->mpp, (size_t)pkt_len, w->ring[slot], w->slot_cap, &img_w, &img_h);
        int64_t t_d1 = monotonic_now_ns();
        if (rc != 0)
            return rc;
        *out_dec_ms = ns_to_ms(t_d1 - t_d0);
        *out_read_ns = t_d1;
    }

    *out_w = img_w;
    *out_h = img_h;
    *out_cap_ns = cap_ns;
    *out_dropped = dropped;
    return 0;
}

static void *worker_vpu_producer(void *arg) {
    NvCamWorker *w = (NvCamWorker *)arg;
    apply_thread_affinity(w->little_cpu_mask, 0);

    int slot = 0;
    while (w->running) {
        int img_w = 0, img_h = 0, dropped = 0;
        int64_t cap_ns = 0, read_ns = 0;
        double dec_ms = 0.0;

        int rc = grab_and_decode_next(w, slot, &img_w, &img_h, &cap_ns, &read_ns, &dec_ms, &dropped);
        if (rc != 0) {
            if (!w->running)
                break;
            if (rc == -2) /* poll timeout */
                continue;
            pthread_mutex_lock(&w->res_mu);
            w->has_error = 1;
            w->last_err = rc;
            pthread_cond_broadcast(&w->res_cv);
            pthread_mutex_unlock(&w->res_mu);
            usleep(2000);
            continue;
        }

        pthread_mutex_lock(&w->hq_mu);
        while (w->running && w->hq_has_item)
            pthread_cond_wait(&w->hq_cv_free, &w->hq_mu);
        if (!w->running) {
            pthread_mutex_unlock(&w->hq_mu);
            break;
        }
        w->hq_slot = slot;
        w->hq_w = img_w;
        w->hq_h = img_h;
        w->hq_cap_ns = cap_ns;
        w->hq_read_ns = read_ns;
        w->hq_dec_ms = dec_ms;
        w->hq_dropped = dropped;
        w->hq_has_item = 1;
        pthread_cond_signal(&w->hq_cv_ready);
        pthread_mutex_unlock(&w->hq_mu);

        slot = (slot + 1) & (NV_RING_SLOTS - 1);
    }

    pthread_mutex_lock(&w->hq_mu);
    w->hq_stop = 1;
    pthread_cond_broadcast(&w->hq_cv_ready);
    pthread_mutex_unlock(&w->hq_mu);
    return NULL;
}

static int ensure_worker_napa_created(NvCamWorker *w) {
    if (w->det)
        return 0;
    w->det = w->api.create_family("tag36h11");
    if (!w->det)
        return -1;
    w->api.set_threads(w->det, w->napa_threads);
    pthread_mutex_lock(&w->ctrl_mu);
    w->lens_applied_gen = UINT64_MAX;
    sync_worker_lens_locked(w);
    pthread_mutex_unlock(&w->ctrl_mu);
    return 0;
}

static void *worker_consumer_loop(void *arg) {
    NvCamWorker *w = (NvCamWorker *)arg;
    /* Set consumer affinity BEFORE creating NAPA so its thread pool inherits A76 cores. */
    apply_thread_affinity(w->big_cpu_mask, w->nice_level);
    if (ensure_worker_napa_created(w) != 0) {
        pthread_mutex_lock(&w->res_mu);
        w->has_error = 1;
        w->last_err = -20;
        w->running = 0;
        pthread_cond_broadcast(&w->res_cv);
        pthread_mutex_unlock(&w->res_mu);
        return NULL;
    }

    NvFrameResult res;
    if (!w->async_vpu) {
        int slot = 0;
        while (w->running) {
            int img_w = 0, img_h = 0, dropped = 0;
            int64_t cap_ns = 0, read_ns = 0;
            double dec_ms = 0.0;

            int rc = grab_and_decode_next(w, slot, &img_w, &img_h, &cap_ns, &read_ns, &dec_ms, &dropped);
            if (rc != 0) {
                if (!w->running)
                    break;
                if (rc == -2)
                    continue;
                pthread_mutex_lock(&w->res_mu);
                w->has_error = 1;
                w->last_err = rc;
                pthread_cond_broadcast(&w->res_cv);
                pthread_mutex_unlock(&w->res_mu);
                usleep(2000);
                continue;
            }
            run_napa_and_pose_slot(w, slot, img_w, img_h, cap_ns, read_ns, dec_ms, dropped, &res);
            publish_frame_result(w, &res);
            slot = (slot + 1) & (NV_RING_SLOTS - 1);
        }
    } else {
        while (w->running) {
            pthread_mutex_lock(&w->hq_mu);
            while (w->running && !w->hq_has_item && !w->hq_stop)
                pthread_cond_wait(&w->hq_cv_ready, &w->hq_mu);
            if (!w->running || (!w->hq_has_item && w->hq_stop)) {
                pthread_mutex_unlock(&w->hq_mu);
                break;
            }
            int slot = w->hq_slot;
            int img_w = w->hq_w;
            int img_h = w->hq_h;
            int64_t cap_ns = w->hq_cap_ns;
            int64_t read_ns = w->hq_read_ns;
            double dec_ms = w->hq_dec_ms;
            int dropped = w->hq_dropped;
            w->hq_has_item = 0;
            pthread_cond_signal(&w->hq_cv_free);
            pthread_mutex_unlock(&w->hq_mu);

            run_napa_and_pose_slot(w, slot, img_w, img_h, cap_ns, read_ns, dec_ms, dropped, &res);
            publish_frame_result(w, &res);
        }
    }
    return NULL;
}

static NvCamWorker *worker_alloc_base(
    int max_w, int max_h, const NvPipelineConfig *cfg, const char *napa_lib_path
) {
    if (max_w <= 0 || max_h <= 0 || !cfg)
        return NULL;

    NvCamWorker *w = (NvCamWorker *)calloc(1, sizeof(NvCamWorker));
    if (!w) return NULL;
    w->v4l2.fd = -1;
    w->max_w = max_w;
    w->max_h = max_h;
    w->slot_cap = (size_t)max_w * (size_t)max_h;

    if (load_napa_api(napa_lib_path, &w->api) != 0)
        goto fail;

    w->mpp = mpp_dec_create(max_w, max_h);
    if (!w->mpp)
        goto fail;
    w->max_pkt_size = w->mpp->max_pkt_size;

    for (int i = 0; i < NV_RING_SLOTS; ++i) {
        void *ptr = NULL;
        if (posix_memalign(&ptr, 64, w->slot_cap) != 0 || !ptr)
            goto fail;
        memset(ptr, 0, w->slot_cap); /* pre-touch pages at init */
        w->ring[i] = (uint8_t *)ptr;
    }

    w->napa_params.min_tag_px = cfg->min_tag_px > 0 ? cfg->min_tag_px : 16;
    w->napa_params.max_hamming = cfg->max_hamming;
    w->napa_params.min_contrast = cfg->min_contrast;
    w->napa_params.min_border_cells = cfg->min_border_cells;
    w->napa_params.sharpening_x100 = cfg->sharpening_x100;
    w->napa_params.split_touching = cfg->split_touching;
    w->napa_threads = (cfg->napa_threads >= 1 && cfg->napa_threads <= 16) ? cfg->napa_threads : 2;
    w->recover_shadows = cfg->recover_shadows ? 1 : 0;
    w->shadow_min_tag_px = cfg->shadow_min_tag_px;
    w->apply_lens = cfg->apply_lens ? 1 : 0;
    w->async_vpu = cfg->async_vpu ? 1 : 0;
    w->big_cpu_mask = cfg->big_cpu_mask;
    w->little_cpu_mask = cfg->little_cpu_mask;
    w->nice_level = cfg->nice_level;

    pthread_mutex_init(&w->ctrl_mu, NULL);
    pthread_mutex_init(&w->hq_mu, NULL);
    pthread_cond_init(&w->hq_cv_ready, NULL);
    pthread_cond_init(&w->hq_cv_free, NULL);
    pthread_mutex_init(&w->res_mu, NULL);
    pthread_cond_init(&w->res_cv, NULL);
    return w;

fail:
    for (int i = 0; i < NV_RING_SLOTS; ++i)
        free(w->ring[i]);
    mpp_dec_destroy(w->mpp);
    free(w);
    return NULL;
}

/* ---- Public Lifecycle & Controls ---- */

NvCamWorker *nv_cam_open(
    const char *dev_path,
    int width,
    int height,
    int fps,
    int req_bufs,
    const NvPipelineConfig *cfg,
    const char *napa_lib_path
) {
    NvCamWorker *w = worker_alloc_base(width, height, cfg, napa_lib_path);
    if (!w) return NULL;
    w->is_replay = 0;

    if (v4l2_open_dev(&w->v4l2, dev_path, width, height, fps, req_bufs) != 0) {
        for (int i = 0; i < NV_RING_SLOTS; ++i) free(w->ring[i]);
        mpp_dec_destroy(w->mpp);
        pthread_mutex_destroy(&w->ctrl_mu);
        pthread_mutex_destroy(&w->hq_mu);
        pthread_cond_destroy(&w->hq_cv_ready);
        pthread_cond_destroy(&w->hq_cv_free);
        pthread_mutex_destroy(&w->res_mu);
        pthread_cond_destroy(&w->res_cv);
        free(w);
        return NULL;
    }

    if (w->v4l2.width > w->max_w || w->v4l2.height > w->max_h) {
        /* Reallocate MPP & ring if camera negotiated larger resolution */
        int nw = w->v4l2.width, nh = w->v4l2.height;
        size_t ncap = (size_t)nw * (size_t)nh;
        NvMppDec *nmpp = mpp_dec_create(nw, nh);
        if (nmpp) {
            mpp_dec_destroy(w->mpp);
            w->mpp = nmpp;
            w->max_w = nw;
            w->max_h = nh;
            w->slot_cap = ncap;
            w->max_pkt_size = nmpp->max_pkt_size;
            for (int i = 0; i < NV_RING_SLOTS; ++i) {
                free(w->ring[i]);
                void *ptr = NULL;
                if (posix_memalign(&ptr, 64, ncap) == 0 && ptr) {
                    memset(ptr, 0, ncap);
                    w->ring[i] = (uint8_t *)ptr;
                } else {
                    w->ring[i] = NULL;
                }
            }
        }
    }
    return w;
}

NvCamWorker *nv_replay_open(
    const uint8_t **jpegs,
    const size_t *lens,
    int n_jpegs,
    int max_w,
    int max_h,
    const NvPipelineConfig *cfg,
    const char *napa_lib_path
) {
    if (!jpegs || !lens || n_jpegs <= 0)
        return NULL;
    NvCamWorker *w = worker_alloc_base(max_w, max_h, cfg, napa_lib_path);
    if (!w) return NULL;
    w->is_replay = 1;
    w->rep_jpegs = jpegs;
    w->rep_lens = lens;
    w->rep_count = n_jpegs;
    return w;
}

int nv_cam_get_info(
    NvCamWorker *w, int *out_w, int *out_h, double *out_fps, int *out_bufs, int *out_ring_slots
) {
    if (!w) return -1;
    if (out_w) *out_w = w->is_replay ? w->max_w : w->v4l2.width;
    if (out_h) *out_h = w->is_replay ? w->max_h : w->v4l2.height;
    if (out_fps) *out_fps = w->is_replay ? 0.0 : w->v4l2.fps;
    if (out_bufs) *out_bufs = w->is_replay ? 0 : (int)w->v4l2.n_bufs;
    if (out_ring_slots) *out_ring_slots = NV_RING_SLOTS;
    return 0;
}

uint8_t *nv_cam_get_ring_slot_ptr(NvCamWorker *w, int slot_idx, size_t *out_cap) {
    if (!w || slot_idx < 0 || slot_idx >= NV_RING_SLOTS)
        return NULL;
    if (out_cap) *out_cap = w->slot_cap;
    return w->ring[slot_idx];
}

int nv_cam_set_ctrl(NvCamWorker *w, uint32_t ctrl_id, int value) {
    if (!w || w->is_replay || w->v4l2.fd < 0)
        return -1;
    struct v4l2_control ctrl;
    memset(&ctrl, 0, sizeof(ctrl));
    ctrl.id = ctrl_id;
    ctrl.value = value;
    pthread_mutex_lock(&w->ctrl_mu);
    int rc = (xioctl(w->v4l2.fd, VIDIOC_S_CTRL, &ctrl) == 0) ? 0 : -1;
    pthread_mutex_unlock(&w->ctrl_mu);
    return rc;
}

int nv_cam_get_ctrl(NvCamWorker *w, uint32_t ctrl_id, int *out_value) {
    if (!w || w->is_replay || w->v4l2.fd < 0 || !out_value)
        return -1;
    struct v4l2_control ctrl;
    memset(&ctrl, 0, sizeof(ctrl));
    ctrl.id = ctrl_id;
    pthread_mutex_lock(&w->ctrl_mu);
    int rc = (xioctl(w->v4l2.fd, VIDIOC_G_CTRL, &ctrl) == 0) ? 0 : -1;
    pthread_mutex_unlock(&w->ctrl_mu);
    if (rc == 0)
        *out_value = ctrl.value;
    return rc;
}

int nv_cam_set_want_jpeg(NvCamWorker *w, int want_jpeg) {
    if (!w) return -1;
    pthread_mutex_lock(&w->ctrl_mu);
    if (want_jpeg) {
        for (int i = 0; i < NV_RING_SLOTS; ++i) {
            if (!w->jpeg_ring[i]) {
                w->jpeg_ring[i] = (uint8_t *)malloc(w->max_pkt_size);
                if (!w->jpeg_ring[i]) {
                    pthread_mutex_unlock(&w->ctrl_mu);
                    return -2;
                }
            }
        }
        w->want_jpeg = 1;
    } else {
        w->want_jpeg = 0;
    }
    pthread_mutex_unlock(&w->ctrl_mu);
    return 0;
}

void nv_cam_update_lens(
    NvCamWorker *w, double fx, double fy, double cx, double cy, const double *dist5, int has_cal
) {
    if (!w) return;
    pthread_mutex_lock(&w->ctrl_mu);
    w->pose_cfg.fx = fx;
    w->pose_cfg.fy = fy;
    w->pose_cfg.cx = cx;
    w->pose_cfg.cy = cy;
    if (dist5) {
        for (int i = 0; i < 5; ++i)
            w->pose_cfg.dist[i] = dist5[i];
    } else {
        memset(w->pose_cfg.dist, 0, sizeof(w->pose_cfg.dist));
    }
    w->pose_cfg.has_cal = has_cal ? 1 : 0;
    w->lens_dirty_gen++;
    pthread_mutex_unlock(&w->ctrl_mu);
}

void nv_cam_update_pose_cfg(
    NvCamWorker *w, double yaw_deg, double pitch_deg, double roll_deg, double tag_height
) {
    if (!w) return;
    pthread_mutex_lock(&w->ctrl_mu);
    w->pose_cfg.yaw_deg = yaw_deg;
    w->pose_cfg.pitch_deg = pitch_deg;
    w->pose_cfg.roll_deg = roll_deg;
    w->pose_cfg.tag_height = tag_height;
    pthread_mutex_unlock(&w->ctrl_mu);
}

void nv_cam_set_valid_ids(NvCamWorker *w, const int *ids, int count) {
    if (!w) return;
    pthread_mutex_lock(&w->ctrl_mu);
    memset(w->pose_cfg.valid_bits, 0, sizeof(w->pose_cfg.valid_bits));
    if (count < 0 || !ids) {
        w->pose_cfg.filter_valid_ids = 0;
    } else {
        w->pose_cfg.filter_valid_ids = 1;
        for (int i = 0; i < count; ++i) {
            int tid = ids[i];
            if (tid >= 0 && tid < NV_VALID_WORDS * 64)
                w->pose_cfg.valid_bits[tid >> 6] |= (1ULL << (tid & 63));
        }
    }
    pthread_mutex_unlock(&w->ctrl_mu);
}

int nv_cam_start(NvCamWorker *w) {
    if (!w || w->started)
        return -1;
    w->running = 1;
    w->hq_stop = 0;
    w->hq_has_item = 0;
    w->has_error = 0;

    if (pthread_create(&w->cons_thr, NULL, worker_consumer_loop, w) != 0) {
        w->running = 0;
        return -2;
    }
    w->cons_started = 1;

    if (w->async_vpu) {
        if (pthread_create(&w->prod_thr, NULL, worker_vpu_producer, w) != 0) {
            w->running = 0;
            pthread_mutex_lock(&w->hq_mu);
            w->hq_stop = 1;
            pthread_cond_broadcast(&w->hq_cv_ready);
            pthread_mutex_unlock(&w->hq_mu);
            pthread_join(w->cons_thr, NULL);
            w->cons_started = 0;
            return -3;
        }
        w->prod_started = 1;
    }

    w->started = 1;
    return 0;
}

void nv_cam_stop(NvCamWorker *w) {
    if (!w) return;
    w->running = 0;

    pthread_mutex_lock(&w->hq_mu);
    w->hq_stop = 1;
    pthread_cond_broadcast(&w->hq_cv_ready);
    pthread_cond_broadcast(&w->hq_cv_free);
    pthread_mutex_unlock(&w->hq_mu);

    pthread_mutex_lock(&w->res_mu);
    pthread_cond_broadcast(&w->res_cv);
    pthread_mutex_unlock(&w->res_mu);

    if (w->prod_started) {
        pthread_join(w->prod_thr, NULL);
        w->prod_started = 0;
    }
    if (w->cons_started) {
        pthread_join(w->cons_thr, NULL);
        w->cons_started = 0;
    }
    w->started = 0;
}

void nv_cam_close(NvCamWorker *w) {
    if (!w) return;
    nv_cam_stop(w);
    if (!w->is_replay)
        v4l2_close_dev(&w->v4l2);
    if (w->det && w->api.destroy) {
        w->api.destroy(w->det);
        w->det = NULL;
    }
    mpp_dec_destroy(w->mpp);
    w->mpp = NULL;
    for (int i = 0; i < NV_RING_SLOTS; ++i) {
        free(w->ring[i]);
        free(w->jpeg_ring[i]);
    }
    pthread_mutex_destroy(&w->ctrl_mu);
    pthread_mutex_destroy(&w->hq_mu);
    pthread_cond_destroy(&w->hq_cv_ready);
    pthread_cond_destroy(&w->hq_cv_free);
    pthread_mutex_destroy(&w->res_mu);
    pthread_cond_destroy(&w->res_cv);
    free(w);
}

/*
 * Wait until worker publishes seq > last_seq.
 * Returns:
 *    1  new result written to *out_res
 *    0  timeout or no new frame yet
 *   -1  worker error / stopped (code in out_res->n_raw_tags when negative or -last_err)
 */
int nv_cam_wait_result(
    NvCamWorker *w,
    uint64_t last_seq,
    int timeout_ms,
    NvFrameResult *out_res,
    uint8_t *opt_jpeg_out,
    int opt_jpeg_cap
) {
    if (!w || !out_res)
        return -1;

    struct timespec ts_abs;
    if (timeout_ms > 0) {
        clock_gettime(CLOCK_REALTIME, &ts_abs);
        ts_abs.tv_sec += timeout_ms / 1000;
        ts_abs.tv_nsec += (long)(timeout_ms % 1000) * 1000000L;
        if (ts_abs.tv_nsec >= 1000000000L) {
            ts_abs.tv_sec += 1;
            ts_abs.tv_nsec -= 1000000000L;
        }
    }

    pthread_mutex_lock(&w->res_mu);
    while (w->running && !w->has_error && w->latest.seq <= last_seq) {
        if (timeout_ms <= 0)
            break;
        int rc = pthread_cond_timedwait(&w->res_cv, &w->res_mu, &ts_abs);
        if (rc == ETIMEDOUT)
            break;
    }

    if (w->latest.seq > last_seq) {
        *out_res = w->latest;
        if (opt_jpeg_out && out_res->jpeg_len > 0 && out_res->jpeg_len <= opt_jpeg_cap) {
            const uint8_t *jsrc = w->jpeg_ring[out_res->ring_slot];
            if (jsrc)
                memcpy(opt_jpeg_out, jsrc, (size_t)out_res->jpeg_len);
            else
                out_res->jpeg_len = 0;
        } else {
            out_res->jpeg_len = 0;
        }
        pthread_mutex_unlock(&w->res_mu);
        return 1;
    }

    int err = w->has_error ? w->last_err : 0;
    pthread_mutex_unlock(&w->res_mu);
    if (err != 0)
        return err < 0 ? err : -1;
    return w->running ? 0 : -1;
}

/* ---- Deterministic Single-Step Replay API (for parity testing) ---- */

int nv_replay_step_sync(NvCamWorker *w, int frame_idx, NvFrameResult *out_res) {
    if (!w || !w->is_replay || !out_res || frame_idx < 0 || frame_idx >= w->rep_count)
        return -1;
    if (ensure_worker_napa_created(w) != 0)
        return -2;
    w->rep_idx = frame_idx;
    int img_w = 0, img_h = 0, dropped = 0;
    int64_t cap_ns = 0, read_ns = 0;
    double dec_ms = 0.0;
    int slot = frame_idx & (NV_RING_SLOTS - 1);
    int rc = grab_and_decode_next(w, slot, &img_w, &img_h, &cap_ns, &read_ns, &dec_ms, &dropped);
    if (rc != 0)
        return rc;
    return run_napa_and_pose_slot(w, slot, img_w, img_h, cap_ns, read_ns, dec_ms, dropped, out_res);
}

int nv_replay_detect_gray_sync(
    NvCamWorker *w, const uint8_t *gray, int width, int height, int stride, NvFrameResult *out_res
) {
    if (!w || !gray || !out_res || width <= 0 || height <= 0 || stride < width)
        return -1;
    if ((size_t)width * (size_t)height > w->slot_cap)
        return -2;
    if (ensure_worker_napa_created(w) != 0)
        return -3;
    if (stride == width) {
        memcpy(w->ring[0], gray, (size_t)width * (size_t)height);
    } else {
        for (int r = 0; r < height; ++r)
            memcpy(w->ring[0] + (size_t)r * (size_t)width, gray + (size_t)r * (size_t)stride, (size_t)width);
    }
    w->jpeg_lens[0] = 0;
    int64_t now_ns = monotonic_now_ns();
    return run_napa_and_pose_slot(w, 0, width, height, now_ns, now_ns, 0.0, 0, out_res);
}

/* ---- Pure C Multi-Camera Benchmark Harness ---- */

typedef struct {
    const uint8_t **jpegs;
    const size_t *lens;
    int n_jpegs;
    int max_w;
    int max_h;
    NvPipelineConfig cfg;
    NvPoseConfig pose_cfg;
    const char *napa_lib_path;
} NvBenchCamSpec;

typedef struct {
    const NvBenchCamSpec *spec;
    double duration_s;
    pthread_barrier_t *barrier;
    NvBenchStats stats;
    int rc;
} NvBenchThreadArg;

static void *bench_cam_thread(void *arg) {
    NvBenchThreadArg *ba = (NvBenchThreadArg *)arg;
    const NvBenchCamSpec *s = ba->spec;
    memset(&ba->stats, 0, sizeof(ba->stats));

    NvCamWorker *w = nv_replay_open(s->jpegs, s->lens, s->n_jpegs, s->max_w, s->max_h, &s->cfg, s->napa_lib_path);
    if (!w) {
        ba->rc = -1;
        pthread_barrier_wait(ba->barrier);
        return NULL;
    }

    pthread_mutex_lock(&w->ctrl_mu);
    w->pose_cfg = s->pose_cfg;
    w->lens_dirty_gen++;
    pthread_mutex_unlock(&w->ctrl_mu);

    const int cap = 65536;
    double *tot_s = (double *)malloc((size_t)cap * sizeof(double));
    if (!tot_s) {
        nv_cam_close(w);
        ba->rc = -2;
        pthread_barrier_wait(ba->barrier);
        return NULL;
    }

    if (nv_cam_start(w) != 0) {
        free(tot_s);
        nv_cam_close(w);
        ba->rc = -3;
        pthread_barrier_wait(ba->barrier);
        return NULL;
    }

    /* Warmup 10 frames before synchronizing across cameras */
    NvFrameResult res;
    uint64_t seq = 0;
    for (int i = 0; i < 10; ++i) {
        if (nv_cam_wait_result(w, seq, 2000, &res, NULL, 0) == 1)
            seq = res.seq;
    }

    pthread_barrier_wait(ba->barrier);

    int n_samp = 0;
    double sum_dec = 0.0, sum_det = 0.0, sum_pose = 0.0;
    int64_t t_start = monotonic_now_ns();
    int64_t t_end = t_start + (int64_t)(ba->duration_s * 1e9);

    while (monotonic_now_ns() < t_end) {
        int wr = nv_cam_wait_result(w, seq, 1000, &res, NULL, 0);
        if (wr == 1) {
            uint64_t missed = (res.seq > seq + 1u) ? (res.seq - seq - 1u) : 0u;
            (void)missed;
            seq = res.seq;
            ba->stats.frames_ok++;
            ba->stats.tags_total += res.n_raw_tags;
            ba->stats.poses_total += res.n_poses;
            sum_dec += res.decode_ms;
            sum_det += res.detect_ms;
            sum_pose += res.pose_ms;
            double tot_ms = res.decode_ms + res.detect_ms + res.pose_ms;
            if (n_samp < cap)
                tot_s[n_samp++] = tot_ms;
        } else if (wr < 0) {
            ba->stats.frames_err++;
        }
    }
    int64_t t_done = monotonic_now_ns();
    nv_cam_stop(w);

    ba->stats.wall_s = (double)(t_done - t_start) * 1e-9;
    ba->stats.fps = (ba->stats.wall_s > 0.0) ? ((double)ba->stats.frames_ok / ba->stats.wall_s) : 0.0;
    if (ba->stats.frames_ok > 0) {
        ba->stats.mean_decode_ms = sum_dec / (double)ba->stats.frames_ok;
        ba->stats.mean_detect_ms = sum_det / (double)ba->stats.frames_ok;
        ba->stats.mean_pose_ms = sum_pose / (double)ba->stats.frames_ok;
    }
    summarize_latencies(
        tot_s, n_samp,
        &ba->stats.mean_total_ms, &ba->stats.p50_total_ms,
        &ba->stats.p95_total_ms, &ba->stats.max_total_ms
    );

    free(tot_s);
    nv_cam_close(w);
    ba->rc = 0;
    return NULL;
}

int nv_bench_multi_cam(
    const NvBenchCamSpec *specs,
    int n_cams,
    double duration_s,
    NvBenchStats *out_stats
) {
    if (!specs || !out_stats || n_cams <= 0 || n_cams > 8 || duration_s <= 0.0)
        return -1;

    pthread_t thrs[8];
    NvBenchThreadArg args[8];
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, (unsigned)n_cams);

    for (int i = 0; i < n_cams; ++i) {
        args[i].spec = &specs[i];
        args[i].duration_s = duration_s;
        args[i].barrier = &barrier;
        args[i].rc = 0;
        if (pthread_create(&thrs[i], NULL, bench_cam_thread, &args[i]) != 0) {
            /* Should not happen for <=8 threads */
            pthread_barrier_destroy(&barrier);
            return -2;
        }
    }

    int err = 0;
    for (int i = 0; i < n_cams; ++i) {
        pthread_join(thrs[i], NULL);
        out_stats[i] = args[i].stats;
        if (args[i].rc != 0)
            err = args[i].rc;
    }
    pthread_barrier_destroy(&barrier);
    return err;
}
