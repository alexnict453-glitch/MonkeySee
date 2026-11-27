/*
 * Direct Linux V4L2 MMAP + Rockchip MPP hardware MJPEG grayscale capture library.
 *
 * Eliminates OpenCV VideoCapture copy/buffer-hold overhead by:
 *   1. Capturing MJPEG frames via V4L2 MMAP with non-blocking latest-frame drain
 *      so a 2-buffer ring never serves stale queued frames.
 *   2. Immediately returning the V4L2 MMAP buffer (VIDIOC_QBUF) after copying
 *      compressed bytes into MPP's DMA packet buffer, before VPU decode runs.
 *   3. Decoding MJPEG directly via Rockchip MPP (cachable=1, copy=1) into the
 *      caller's pre-allocated grayscale ring buffer and returning the kernel
 *      CLOCK_MONOTONIC v4l2_buffer timestamp in nanoseconds.
 *
 * Build on RK3588 / Orange Pi 5:
 *   gcc -shared -fPIC -O3 -mcpu=cortex-a76 -Wall -Wextra \
 *       camera/v4l2_mpp.c -I/usr/local/include/rockchip -L/usr/local/lib \
 *       -lrockchip_mpp -o /usr/local/lib/libv4l2_mpp.so
 */

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <rockchip/mpp_buffer.h>
#include <rockchip/mpp_frame.h>
#include <rockchip/mpp_meta.h>
#include <rockchip/mpp_packet.h>
#include <rockchip/rk_mpi.h>
#include <rockchip/rk_mpi_cmd.h>

#define V4L2_MPP_MAX_FRAME_AGE_NS 1000000000LL

typedef struct {
    void *start;
    size_t length;
} V4L2MmapBuf;

typedef struct {
    int fd;
    int streaming;
    int width;
    int height;
    double fps;
    V4L2MmapBuf *bufs;
    unsigned int n_bufs;

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
    int cachable;
    int max_w;
    int max_h;
} V4L2MppCam;

static inline int align_up_int(int v, int a) {
    return (v + a - 1) & ~(a - 1);
}

static inline int64_t monotonic_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

static int xioctl(int fd, unsigned long req, void *arg) {
    int r;
    do {
        r = ioctl(fd, req, arg);
    } while (r == -1 && errno == EINTR);
    return r;
}

static void mpp_cleanup(V4L2MppCam *cam) {
    if (!cam) return;
    if (cam->mpi && cam->ctx)
        cam->mpi->reset(cam->ctx);
    if (cam->packet) {
        mpp_packet_deinit(&cam->packet);
        cam->packet = NULL;
    }
    if (cam->frame) {
        mpp_frame_deinit(&cam->frame);
        cam->frame = NULL;
    }
    if (cam->frm_buf) {
        mpp_buffer_put(cam->frm_buf);
        cam->frm_buf = NULL;
    }
    if (cam->pkt_buf) {
        mpp_buffer_put(cam->pkt_buf);
        cam->pkt_buf = NULL;
    }
    if (cam->frm_grp) {
        mpp_buffer_group_put(cam->frm_grp);
        cam->frm_grp = NULL;
    }
    if (cam->pkt_grp) {
        mpp_buffer_group_put(cam->pkt_grp);
        cam->pkt_grp = NULL;
    }
    if (cam->dec_cfg) {
        mpp_dec_cfg_deinit(cam->dec_cfg);
        cam->dec_cfg = NULL;
    }
    if (cam->ctx) {
        mpp_destroy(cam->ctx);
        cam->ctx = NULL;
        cam->mpi = NULL;
    }
}

static int mpp_setup(V4L2MppCam *cam, int max_w, int max_h, int cachable) {
    cam->max_w = max_w;
    cam->max_h = max_h;
    cam->cachable = cachable ? 1 : 0;

    int hor_stride = align_up_int(max_w, 64);
    int ver_stride = align_up_int(max_h, 64);
    cam->max_frm_size = (size_t)hor_stride * (size_t)ver_stride * 4u;
    cam->max_pkt_size = (size_t)max_w * (size_t)max_h * 2u;
    if (cam->max_pkt_size < (2u << 20))
        cam->max_pkt_size = (2u << 20);

    RK_U32 frm_type = MPP_BUFFER_TYPE_DMA_HEAP;
    if (cam->cachable)
        frm_type |= MPP_BUFFER_FLAGS_CACHABLE;
    /* Packet DMA buffer is sequential CPU write-only: uncachable WC avoids 2 sync ioctls/frame. */
    RK_U32 pkt_type = MPP_BUFFER_TYPE_DMA_HEAP;

    if (mpp_buffer_group_get_internal(&cam->frm_grp, frm_type) != MPP_OK || !cam->frm_grp)
        goto fail;
    if (mpp_buffer_group_get_internal(&cam->pkt_grp, pkt_type) != MPP_OK || !cam->pkt_grp)
        goto fail;
    if (mpp_buffer_get(cam->frm_grp, &cam->frm_buf, cam->max_frm_size) != MPP_OK || !cam->frm_buf)
        goto fail;
    if (mpp_buffer_get(cam->pkt_grp, &cam->pkt_buf, cam->max_pkt_size) != MPP_OK || !cam->pkt_buf)
        goto fail;
    cam->pkt_ptr = mpp_buffer_get_ptr(cam->pkt_buf);
    if (!cam->pkt_ptr)
        goto fail;
    if (mpp_packet_init_with_buffer(&cam->packet, cam->pkt_buf) != MPP_OK || !cam->packet)
        goto fail;

    if (mpp_frame_init(&cam->frame) != MPP_OK || !cam->frame)
        goto fail;
    mpp_frame_set_buffer(cam->frame, cam->frm_buf);

    if (mpp_create(&cam->ctx, &cam->mpi) != MPP_OK || !cam->ctx || !cam->mpi)
        goto fail;
    if (mpp_init(cam->ctx, MPP_CTX_DEC, MPP_VIDEO_CodingMJPEG) != MPP_OK)
        goto fail;

    if (mpp_dec_cfg_init(&cam->dec_cfg) == MPP_OK && cam->dec_cfg) {
        if (cam->mpi->control(cam->ctx, MPP_DEC_GET_CFG, cam->dec_cfg) == MPP_OK) {
            mpp_dec_cfg_set_u32(cam->dec_cfg, "base:split_parse", 0);
            cam->mpi->control(cam->ctx, MPP_DEC_SET_CFG, cam->dec_cfg);
        }
    }
    return 0;

fail:
    mpp_cleanup(cam);
    return -1;
}

void v4l2_mpp_close(V4L2MppCam *cam) {
    if (!cam) return;
    if (cam->fd >= 0 && cam->streaming) {
        enum v4l2_buf_type btype = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(cam->fd, VIDIOC_STREAMOFF, &btype);
        cam->streaming = 0;
    }
    if (cam->bufs) {
        for (unsigned int i = 0; i < cam->n_bufs; ++i) {
            if (cam->bufs[i].start)
                munmap(cam->bufs[i].start, cam->bufs[i].length);
        }
        free(cam->bufs);
        cam->bufs = NULL;
        cam->n_bufs = 0;
    }
    if (cam->fd >= 0) {
        close(cam->fd);
        cam->fd = -1;
    }
    mpp_cleanup(cam);
    free(cam);
}

V4L2MppCam *v4l2_mpp_open(
    const char *dev_path, int width, int height, int fps, int req_bufs, int cachable
) {
    if (!dev_path || width <= 0 || height <= 0)
        return NULL;

    V4L2MppCam *cam = (V4L2MppCam *)calloc(1, sizeof(V4L2MppCam));
    if (!cam) return NULL;
    cam->fd = -1;

    cam->fd = open(dev_path, O_RDWR | O_NONBLOCK);
    if (cam->fd < 0)
        goto fail;

    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = (uint32_t)width;
    fmt.fmt.pix.height = (uint32_t)height;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    fmt.fmt.pix.field = V4L2_FIELD_ANY;
    if (xioctl(cam->fd, VIDIOC_S_FMT, &fmt) < 0)
        goto fail;
    if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_MJPEG)
        goto fail;

    cam->width = (int)fmt.fmt.pix.width;
    cam->height = (int)fmt.fmt.pix.height;
    if (cam->width <= 0 || cam->height <= 0)
        goto fail;

    if (fps > 0) {
        struct v4l2_streamparm sp;
        memset(&sp, 0, sizeof(sp));
        sp.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        sp.parm.capture.timeperframe.numerator = 1;
        sp.parm.capture.timeperframe.denominator = (uint32_t)fps;
        if (xioctl(cam->fd, VIDIOC_S_PARM, &sp) == 0 &&
            sp.parm.capture.timeperframe.numerator > 0 &&
            sp.parm.capture.timeperframe.denominator > 0) {
            cam->fps = (double)sp.parm.capture.timeperframe.denominator /
                       (double)sp.parm.capture.timeperframe.numerator;
        } else {
            cam->fps = (double)fps;
        }
    }

    int target_bufs = (req_bufs > 0) ? req_bufs : 2;
    if (target_bufs > 8) target_bufs = 8;

    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = (uint32_t)target_bufs;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(cam->fd, VIDIOC_REQBUFS, &req) < 0 || req.count < 1)
        goto fail;

    cam->n_bufs = req.count;
    cam->bufs = (V4L2MmapBuf *)calloc(cam->n_bufs, sizeof(V4L2MmapBuf));
    if (!cam->bufs)
        goto fail;

    for (unsigned int i = 0; i < cam->n_bufs; ++i) {
        struct v4l2_buffer b;
        memset(&b, 0, sizeof(b));
        b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        b.memory = V4L2_MEMORY_MMAP;
        b.index = i;
        if (xioctl(cam->fd, VIDIOC_QUERYBUF, &b) < 0)
            goto fail;
        cam->bufs[i].length = b.length;
        cam->bufs[i].start = mmap(NULL, b.length, PROT_READ | PROT_WRITE, MAP_SHARED, cam->fd, b.m.offset);
        if (cam->bufs[i].start == MAP_FAILED) {
            cam->bufs[i].start = NULL;
            goto fail;
        }
        if (xioctl(cam->fd, VIDIOC_QBUF, &b) < 0)
            goto fail;
    }

    if (mpp_setup(cam, cam->width, cam->height, cachable) != 0)
        goto fail;

    enum v4l2_buf_type btype = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(cam->fd, VIDIOC_STREAMON, &btype) < 0)
        goto fail;
    cam->streaming = 1;
    return cam;

fail:
    v4l2_mpp_close(cam);
    return NULL;
}

int v4l2_mpp_get_info(
    V4L2MppCam *cam, int *out_w, int *out_h, double *out_fps, int *out_bufs, int *out_max_pkt
) {
    if (!cam) return -1;
    if (out_w) *out_w = cam->width;
    if (out_h) *out_h = cam->height;
    if (out_fps) *out_fps = cam->fps;
    if (out_bufs) *out_bufs = (int)cam->n_bufs;
    if (out_max_pkt) *out_max_pkt = (int)cam->max_pkt_size;
    return 0;
}

int v4l2_mpp_set_ctrl(V4L2MppCam *cam, uint32_t ctrl_id, int value) {
    if (!cam || cam->fd < 0) return -1;
    struct v4l2_control ctrl;
    memset(&ctrl, 0, sizeof(ctrl));
    ctrl.id = ctrl_id;
    ctrl.value = value;
    return (xioctl(cam->fd, VIDIOC_S_CTRL, &ctrl) == 0) ? 0 : -1;
}

int v4l2_mpp_get_ctrl(V4L2MppCam *cam, uint32_t ctrl_id, int *out_value) {
    if (!cam || cam->fd < 0 || !out_value) return -1;
    struct v4l2_control ctrl;
    memset(&ctrl, 0, sizeof(ctrl));
    ctrl.id = ctrl_id;
    if (xioctl(cam->fd, VIDIOC_G_CTRL, &ctrl) < 0)
        return -1;
    *out_value = ctrl.value;
    return 0;
}

static int decode_pkt_to_gray(
    V4L2MppCam *cam, size_t len, uint8_t *out_y, int out_cap, int *out_w, int *out_h
) {
    void *pkt_ptr = cam->pkt_ptr;
    MppPacket packet = cam->packet;
    if (!pkt_ptr || !packet)
        return -11;

    mpp_packet_set_data(packet, pkt_ptr);
    mpp_packet_set_size(packet, cam->max_pkt_size);
    mpp_packet_set_pos(packet, pkt_ptr);
    mpp_packet_set_length(packet, len);

    MppMeta meta = mpp_packet_get_meta(packet);
    if (meta)
        mpp_meta_set_frame(meta, KEY_OUTPUT_FRAME, cam->frame);

    MPP_RET ret = cam->mpi->decode_put_packet(cam->ctx, packet);
    if (ret != MPP_OK)
        return -13;

    MppFrame frame_ret = NULL;
    ret = cam->mpi->decode_get_frame(cam->ctx, &frame_ret);

    if (ret != MPP_OK || !frame_ret) {
        if (frame_ret && frame_ret != cam->frame)
            mpp_frame_deinit(&frame_ret);
        return -14;
    }
    if (mpp_frame_get_errinfo(frame_ret)) {
        if (frame_ret != cam->frame)
            mpp_frame_deinit(&frame_ret);
        return -15;
    }

    MppBuffer out_buf = mpp_frame_get_buffer(frame_ret);
    if (!out_buf) {
        if (frame_ret != cam->frame)
            mpp_frame_deinit(&frame_ret);
        return -16;
    }

    int w = (int)mpp_frame_get_width(frame_ret);
    int h = (int)mpp_frame_get_height(frame_ret);
    int stride = (int)mpp_frame_get_hor_stride(frame_ret);
    uint8_t *y_ptr = (uint8_t *)mpp_buffer_get_ptr(out_buf);
    if (frame_ret != cam->frame)
        mpp_frame_deinit(&frame_ret);

    if (!y_ptr || w <= 0 || h <= 0 || stride < w)
        return -17;

    /* mpp_get_frame() already called mpp_buffer_sync_ro_begin(out_buf). */
    size_t y_bytes = (size_t)stride * (size_t)h;
    if (!out_y || out_cap < w * h) {
        if (cam->cachable && y_bytes > 0)
            mpp_buffer_sync_ro_partial_end(out_buf, 0, (RK_U32)y_bytes);
        return -18;
    }

    if (stride == w) {
        memcpy(out_y, y_ptr, (size_t)w * (size_t)h);
    } else {
        for (int r = 0; r < h; ++r)
            memcpy(out_y + (size_t)r * (size_t)w, y_ptr + (size_t)r * (size_t)stride, (size_t)w);
    }

    if (cam->cachable && y_bytes > 0)
        mpp_buffer_sync_ro_partial_end(out_buf, 0, (RK_U32)y_bytes);

    *out_w = w;
    *out_h = h;
    return 0;
}

int v4l2_mpp_grab_gray(
    V4L2MppCam *cam,
    int timeout_ms,
    uint8_t *out_y,
    int out_cap,
    int *out_w,
    int *out_h,
    int64_t *out_ts_ns,
    uint8_t *opt_jpeg_out,
    int opt_jpeg_cap,
    int *opt_jpeg_len,
    int *out_dropped
) {
    if (opt_jpeg_len) *opt_jpeg_len = 0;
    if (out_dropped) *out_dropped = 0;
    if (!cam || cam->fd < 0 || !cam->streaming || !out_y)
        return -1;

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

        struct pollfd pfd = { .fd = cam->fd, .events = POLLIN, .revents = 0 };
        int pr = poll(&pfd, 1, wait_ms);
        if (pr < 0) {
            if (errno == EINTR)
                continue;
            return -3;
        }
        if (pr == 0)
            return -2;
        if (pfd.revents & (POLLERR | POLLNVAL | POLLHUP))
            return -3;

        memset(&vb, 0, sizeof(vb));
        vb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        vb.memory = V4L2_MEMORY_MMAP;
        if (xioctl(cam->fd, VIDIOC_DQBUF, &vb) < 0) {
            if (errno == EAGAIN)
                continue;
            return -4;
        }
        break;
    }

    /* Non-blocking drain: if newer frames are already ready, keep the latest and re-queue older ones. */
    int dropped = 0;
    for (unsigned int d = 0; d < cam->n_bufs; ++d) {
        struct v4l2_buffer vb_next;
        memset(&vb_next, 0, sizeof(vb_next));
        vb_next.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        vb_next.memory = V4L2_MEMORY_MMAP;
        if (xioctl(cam->fd, VIDIOC_DQBUF, &vb_next) == 0) {
            if (vb_next.index < cam->n_bufs && vb_next.bytesused >= 4 && !(vb_next.flags & V4L2_BUF_FLAG_ERROR)) {
                xioctl(cam->fd, VIDIOC_QBUF, &vb);
                vb = vb_next;
                dropped++;
            } else {
                xioctl(cam->fd, VIDIOC_QBUF, &vb_next);
            }
        } else {
            break;
        }
    }
    if (out_dropped) *out_dropped = dropped;

    if (vb.index >= cam->n_bufs || vb.bytesused < 4 || (vb.flags & V4L2_BUF_FLAG_ERROR) ||
        (size_t)vb.bytesused > cam->bufs[vb.index].length) {
        xioctl(cam->fd, VIDIOC_QBUF, &vb);
        return -5;
    }

    int64_t now_ns = monotonic_now_ns();
    int64_t ts_ns = now_ns;
    if ((vb.flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) == V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC) {
        int64_t cap_ns = (int64_t)vb.timestamp.tv_sec * 1000000000LL +
                         (int64_t)vb.timestamp.tv_usec * 1000LL;
        if (cap_ns > 0 && now_ns >= cap_ns && (now_ns - cap_ns) < V4L2_MPP_MAX_FRAME_AGE_NS)
            ts_ns = cap_ns;
    }
    if (out_ts_ns) *out_ts_ns = ts_ns;

    const uint8_t *src = (const uint8_t *)cam->bufs[vb.index].start;
    size_t jlen = (size_t)vb.bytesused;
    if (jlen > cam->max_pkt_size) {
        xioctl(cam->fd, VIDIOC_QBUF, &vb);
        return -6;
    }

    void *pkt_ptr = cam->pkt_ptr;
    if (!pkt_ptr) {
        xioctl(cam->fd, VIDIOC_QBUF, &vb);
        return -7;
    }

    memcpy(pkt_ptr, src, jlen);

    if (opt_jpeg_out && opt_jpeg_cap >= (int)jlen) {
        memcpy(opt_jpeg_out, src, jlen);
        if (opt_jpeg_len) *opt_jpeg_len = (int)jlen;
    }

    /* Immediately return MMAP buffer to kernel before MPP decode runs. */
    if (xioctl(cam->fd, VIDIOC_QBUF, &vb) < 0)
        return -8;

    int w = 0, h = 0;
    int rc = decode_pkt_to_gray(cam, jlen, out_y, out_cap, &w, &h);
    if (rc != 0)
        return rc;

    if (out_w) *out_w = w;
    if (out_h) *out_h = h;
    return 0;
}
