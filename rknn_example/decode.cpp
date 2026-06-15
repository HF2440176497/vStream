/**
 * RTMP 硬解码 Demo (RKMPP)
 *
 * 功能：循环从 RTMP 拉取视频流，使用 Rockchip MPP 硬件解码器解码，
 *      并将每一帧转成 OpenCV cv::Mat (BGR) 供后续处理/显示。
 *
 * 用法: ./demo rtmp://server/live/stream
 *
 * 设计参考: example/decode.cpp
 */

#include <iostream>
#include <string>
#include <atomic>
#include <csignal>
#include <cstring>
#include <cerrno>
#include <chrono>
#include <thread>
#include <vector>
#include <functional>
#include <unistd.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/time.h>
#include <libavutil/hwcontext.h>
#include <libswscale/swscale.h>
}

#include <opencv2/opencv.hpp>

static std::atomic<bool> g_running{true};
static void onSignal(int) { g_running = false; }


class RtmpHardDecoder {
public:
    RtmpHardDecoder()  = default;
    ~RtmpHardDecoder() { release(); }

    RtmpHardDecoder(const RtmpHardDecoder&) = delete;
    RtmpHardDecoder& operator=(const RtmpHardDecoder&) = delete;

    /**
     * 打开 RTMP 源并初始化 RKMPP 硬解码器
     * @param url    RTMP 拉流地址 (如 rtmp://192.168.100.12/live/in)
     * @param timeoutMs 协议打开超时 (毫秒)
     * @return 成功返回 true
     */
    bool open(const std::string& url, int timeoutMs = 10000000) {
        // 1. 初始化 FFmpeg 全局
        avformat_network_init();

        // 2. 探测板端 DRM/RKMPP 设备节点 (用于 EFAULT 时的友好报错)
        precheckDeviceNodes();

        // 3. 设置 RTMP 超时 (libavformat 的 flv(rtmp) 协议选项)
        AVDictionary* openOpts = nullptr;
        av_dict_set(&openOpts, "rw_timeout", std::to_string(timeoutMs).c_str(), 0);
        av_dict_set(&openOpts, "stimeout",   std::to_string(timeoutMs).c_str(), 0);

        // 4. 打开输入 (RTMP 由 libavformat 的 flv 协议解析)
        int ret = avformat_open_input(&fmtCtx_, url.c_str(), nullptr, &openOpts);
        av_dict_free(&openOpts);
        if (ret < 0) {
            char err[128];
            av_strerror(ret, err, sizeof(err));
            std::cerr << "avformat_open_input 失败: " << err << std::endl;
            return false;
        }
        std::cout << "已连接: " << url << std::endl;

        // 5. 拉取流信息
        ret = avformat_find_stream_info(fmtCtx_, nullptr);
        if (ret < 0) {
            char err[128];
            av_strerror(ret, err, sizeof(err));
            std::cerr << "avformat_find_stream_info 失败: " << err << std::endl;
            return false;
        }

        // 6. 找视频流
        ret = av_find_best_stream(fmtCtx_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (ret < 0) {
            std::cerr << "找不到视频流" << std::endl;
            return false;
        }
        videoStreamIdx_ = ret;
        AVStream* vs = fmtCtx_->streams[videoStreamIdx_];

        std::cout << "视频流参数: " << vs->codecpar->width << "x" << vs->codecpar->height
            << ", codec=" << avcodec_get_name(vs->codecpar->codec_id)
            << ", fps=" << av_q2d(vs->avg_frame_rate) << std::endl;

        // 7. 选 RKMPP 硬解码器
        const AVCodec* decoder = pickHwDecoder(vs->codecpar->codec_id);
        if (!decoder) {
            std::cerr << "找不到对应的 RKMPP 解码器" << std::endl;
            return false;
        }
        std::cout << "使用硬解码器: " << decoder->name << std::endl;

        // 8. 分配解码器上下文
        decCtx_ = avcodec_alloc_context3(decoder);
        if (!decCtx_) { std::cerr << "avcodec_alloc_context3 失败" << std::endl; return false; }

        if (avcodec_parameters_to_context(decCtx_, vs->codecpar) < 0) {
            std::cerr << "avcodec_parameters_to_context 失败" << std::endl;
            return false;
        }

        // 9. 初始化 DRM 硬解设备 + 帧上下文
        if (!initHwDevice(decoder)) {
            std::cerr << "硬件设备初始化失败" << std::endl;
            return false;
        }

        // 10. 打开解码器
        AVDictionary* decOpts = nullptr;
        av_dict_set(&decOpts, "extra_hw_frames", "8", 0);
        ret = avcodec_open2(decCtx_, decoder, &decOpts);
        av_dict_free(&decOpts);
        if (ret < 0) {
            char err[128];
            av_strerror(ret, err, sizeof(err));
            std::cerr << "avcodec_open2 失败: " << err << std::endl;
            return false;
        }

        // 11. 准备包/帧/转换上下文
        pkt_   = av_packet_alloc();
        frame_ = av_frame_alloc();
        swFrame_ = av_frame_alloc();  // 硬解帧 (DRM_PRIME) -> 软件帧 (NV12) 的中转
        if (!pkt_ || !frame_ || !swFrame_) {
            std::cerr << "av_packet_alloc / av_frame_alloc 失败" << std::endl;
            return false;
        }

        // sws 上下文: NV12 -> BGR24
        swsCtx_ = sws_getContext(
            decCtx_->width, decCtx_->height, AV_PIX_FMT_NV12,
            decCtx_->width, decCtx_->height, AV_PIX_FMT_BGR24,
            SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!swsCtx_) { std::cerr << "sws_getContext 失败" << std::endl; return false; }

        // 预分配 BGR 目标缓冲区
        bgrBufSize_ = av_image_get_buffer_size(AV_PIX_FMT_BGR24,
                                               decCtx_->width, decCtx_->height, 1);
        bgrBuf_.resize(bgrBufSize_);

        width_  = decCtx_->width;
        height_ = decCtx_->height;
        return true;
    }

    /**
     * 主循环: 读包 -> 解码 -> 转 BGR -> 回调
     * @param onFrame 用户回调, 接收到一帧 BGR cv::Mat
     */
    void run(const std::function<void(const cv::Mat&)>& onFrame) {
        int64_t frameCnt = 0;
        auto t0 = std::chrono::steady_clock::now();

        while (g_running) {
            int ret = av_read_frame(fmtCtx_, pkt_);
            if (ret < 0) {
                if (ret == AVERROR_EOF) {
                    std::cout << "流结束 (EOF)" << std::endl;
                    break;
                }
                // 网络流中常见的瞬时错误, 短暂休眠后继续
                if (ret == AVERROR(EAGAIN)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    continue;
                }
                char err[128];
                av_strerror(ret, err, sizeof(err));
                std::cerr << "av_read_frame 错误: " << err << std::endl;
                break;
            }

            if (pkt_->stream_index != videoStreamIdx_) {
                av_packet_unref(pkt_);
                continue;
            }

            ret = avcodec_send_packet(decCtx_, pkt_);
            av_packet_unref(pkt_);
            if (ret < 0) {
                if (ret != AVERROR(EAGAIN)) {
                    char err[128];
                    av_strerror(ret, err, sizeof(err));
                    std::cerr << "avcodec_send_packet 错误: " << err << std::endl;
                }
                continue;
            }

            // 拉取所有可用的解码帧
            while (ret >= 0) {
                ret = avcodec_receive_frame(decCtx_, frame_);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                } else if (ret < 0) {
                    char err[128];
                    av_strerror(ret, err, sizeof(err));
                    std::cerr << "avcodec_receive_frame 错误: " << err << std::endl;
                    break;
                }

                cv::Mat bgr = hwFrameToBgrMat(frame_);
                if (!bgr.empty() && onFrame) {
                    onFrame(bgr);
                    ++frameCnt;
                }
                av_frame_unref(frame_);

                // 简单节流: 1s 打印一次 FPS
                auto now = std::chrono::steady_clock::now();
                auto el  = std::chrono::duration_cast<std::chrono::milliseconds>(now - t0).count();
                if (el >= 1000) {
                    double fps = frameCnt * 1000.0 / el;
                    std::cout << "已解码 " << frameCnt << " 帧, FPS=" << fps << std::endl;
                    frameCnt = 0;
                    t0 = now;
                }
            }
        }
    }

    void release() {
        if (swsCtx_)  { sws_freeContext(swsCtx_);  swsCtx_ = nullptr; }
        if (pkt_)     { av_packet_free(&pkt_);     pkt_     = nullptr; }
        if (frame_)   { av_frame_free(&frame_);     frame_   = nullptr; }
        if (swFrame_) { av_frame_free(&swFrame_);   swFrame_ = nullptr; }
        if (decCtx_)  { avcodec_free_context(&decCtx_); decCtx_ = nullptr; }
        if (fmtCtx_)  { avformat_close_input(&fmtCtx_); fmtCtx_ = nullptr; }
        if (hwDevCtx_){ av_buffer_unref(&hwDevCtx_);     hwDevCtx_ = nullptr; }
    }

    int width()  const { return width_;  }
    int height() const { return height_; }

private:

    const AVCodec* pickHwDecoder(AVCodecID id) {
        const char* hwName = nullptr;
        switch (id) {
            case AV_CODEC_ID_H264:  hwName = "h264_rkmpp";  break;
            case AV_CODEC_ID_HEVC:  hwName = "hevc_rkmpp";  break;
            case AV_CODEC_ID_VP9:   hwName = "vp9_rkmpp";   break;
            case AV_CODEC_ID_AV1:   hwName = "av1_rkmpp";   break;
            default: break;
        }
        if (hwName) {
            const AVCodec* c = avcodec_find_decoder_by_name(hwName);
            if (c) return c;
            std::cout << "硬解 " << hwName << " 不可用, 尝试软解" << std::endl;
        }
        return avcodec_find_decoder(id);
    }

    bool initHwDevice(const AVCodec* /*decoder*/) {
        // 一些 ffmpeg-rockchip 编译会把后端注册成 rkmpp,
        // 另一些只注册成 drm (走 libdrm + /dev/dri/...).
        // 用 av_hwdevice_find_type_by_name 运行时探测, 避免硬编码枚举值。
        // 两种都试一遍, 失败时把底层 errno 一起打出来, 方便定位 EFAULT 的根因。
        const char* devPath = pickDrmDevice();
        std::cout << "硬件设备路径: " << (devPath ? devPath : "(auto)") << std::endl;

        struct TryItem { AVHWDeviceType type; const char* name; };
        TryItem items[2];
        int n = 0;
        AVHWDeviceType t;
        t = av_hwdevice_find_type_by_name("rkmpp");
        if (t != AV_HWDEVICE_TYPE_NONE) items[n++] = {t, "rkmpp"};
        t = av_hwdevice_find_type_by_name("drm");
        if (t != AV_HWDEVICE_TYPE_NONE) items[n++] = {t, "drm"};

        if (n == 0) {
            std::cerr << "当前 ffmpeg-rockchip 既没注册 rkmpp 也没注册 drm 后端。" << std::endl
                      << "请用 ffmpeg -hwaccels 查看可用的硬件加速器。" << std::endl;
            return false;
        }

        for (int i = 0; i < n; ++i) {
            // 关键细节: rkmpp 后端内部自己 open("/dev/mpp_service"),
            // 那个 device 路径对它无效,传了反而误导; drm 后端才需要 DRM 节点。
            const char* devForType = (std::string(items[i].name) == "rkmpp")
                                     ? nullptr : devPath;
            int ret = av_hwdevice_ctx_create(&hwDevCtx_, items[i].type,
                                             devForType, nullptr, 0);
            if (ret == 0) {
                std::cout << "硬件设备创建成功: " << items[i].name
                          << " (" << av_hwdevice_get_type_name(items[i].type) << ")"
                          << "  device=" << (devForType ? devForType : "<rkmpp 自管>")
                          << std::endl;
                decCtx_->hw_device_ctx = av_buffer_ref(hwDevCtx_);
                decCtx_->pix_fmt       = AV_PIX_FMT_DRM_PRIME;
                decCtx_->get_format    = &RtmpHardDecoder::getHwFormat;
                return true;
            }
            char err[128];
            av_strerror(ret, err, sizeof(err));
            std::cerr << "  - 尝试 " << items[i].name << " 失败: " << err
                      << " (errno=" << errno << ": " << std::strerror(errno) << ")"
                      << std::endl;
        }
        std::cerr << "所有硬件设备类型都创建失败。" << std::endl
                  << "排查清单:\n"
                  << "  1) /dev/dri/card* / /dev/dri/renderD* 是否存在且可读 (ls -l /dev/dri/)\n"
                  << "  2) 内核模块 rockchip_mpp 等是否加载 (lsmod | grep mpp)\n"
                  << "  3) /dev/mpp_service 是否存在\n"
                  << "  4) 当前 ffmpeg-rockchip 是否带 rkmpp/drm 后端 (ffmpeg -hwaccels)\n"
                  << "  5) 进程权限 (root? 加 video / render 用户组?)" << std::endl;
        return false;
    }

    // 选一个可用的 DRM 设备节点, 没有就返回 nullptr 让 FFmpeg 自动探测
    static const char* pickDrmDevice() {
        static const char* kCandidates[] = {
            "/dev/dri/card0",
            "/dev/dri/card1",
            "/dev/dri/renderD128",
            "/dev/dri/renderD129",
            nullptr
        };
        for (int i = 0; kCandidates[i]; ++i) {
            if (access(kCandidates[i], R_OK | W_OK) == 0) {
                return kCandidates[i];
            }
        }
        return nullptr;
    }

    // 启动时打一遍设备节点状态, EFAULT 时一眼就能看出是缺节点还是缺权限
    static void precheckDeviceNodes() {
        std::cout << "--- 设备节点探测 ---" << std::endl;
        const char* paths[] = {
            "/dev/mpp_service",
            "/dev/dri/card0",
            "/dev/dri/card1",
            "/dev/dri/renderD128",
            "/dev/dri/renderD129",
        };
        for (auto p : paths) {
            if (access(p, F_OK) == 0) {
                bool rw = (access(p, R_OK | W_OK) == 0);
                std::cout << "  [OK] " << p << " (权限: " << (rw ? "rw" : "ro/无") << ")" << std::endl;
            } else {
                std::cout << "  [--] " << p << " (不存在)" << std::endl;
            }
        }
        std::cout << "--------------------" << std::endl;
    }

    static enum AVPixelFormat getHwFormat(AVCodecContext* /*ctx*/,
                                          const enum AVPixelFormat* pixFmts) {
        for (const enum AVPixelFormat* p = pixFmts; *p != AV_PIX_FMT_NONE; ++p) {
            if (*p == AV_PIX_FMT_DRM_PRIME) return *p;
        }
        return AV_PIX_FMT_NONE;
    }

    /**
     * 把 RKMPP 硬解帧 (DRM_PRIME) -> NV12 -> BGR24, 包装为 cv::Mat
     * 注意: 返回的 Mat 引用了内部 bgrBuf_, 回调里用完即丢, 下一帧会覆盖。
     */
    cv::Mat hwFrameToBgrMat(AVFrame* hwFrame) {
        if (hwFrame->format != AV_PIX_FMT_DRM_PRIME) {
            // 软件回退路径 (本 demo 主走硬解, 这里简单起见只警告)
            std::cout << "非 DRM_PRIME 帧 (format=" << hwFrame->format << "), 跳过" << std::endl;
            return {};
        }

        // 1) 硬解帧 -> 系统内存 (NV12)
        av_frame_unref(swFrame_);
        int ret = av_hwframe_transfer_data(swFrame_, hwFrame, 0);
        if (ret < 0) {
            char err[128];
            av_strerror(ret, err, sizeof(err));
            std::cerr << "av_hwframe_transfer_data 失败: " << err << std::endl;
            return {};
        }
        if (swFrame_->format != AV_PIX_FMT_NV12) {
            std::cout << "硬解后 sw format=" << swFrame_->format << " (非 NV12), 跳过" << std::endl;
            return {};
        }

        // 2) NV12 -> BGR24
        uint8_t* dst[1]      = { bgrBuf_.data() };
        int      dstStride[1]= { width_ * 3 };
        int h = sws_scale(swsCtx_,
                         swFrame_->data, swFrame_->linesize,
                         0, height_,
                         dst, dstStride);
        if (h != height_) {
            std::cout << "sws_scale 输出行数异常: " << h << " (期望 " << height_ << ")" << std::endl;
            return {};
        }

        // 3) 包装为 cv::Mat (不拷贝, 仅引用内部 bgrBuf_)
        return cv::Mat(height_, width_, CV_8UC3, bgrBuf_.data()).clone(); // 克隆一份防止被覆盖
    }

private:
    AVFormatContext* fmtCtx_     = nullptr;
    AVCodecContext*  decCtx_     = nullptr;
    AVBufferRef*     hwDevCtx_   = nullptr;
    AVPacket*        pkt_        = nullptr;
    AVFrame*         frame_      = nullptr;   // 硬解输出 (DRM_PRIME)
    AVFrame*         swFrame_    = nullptr;   // hw->sw 中转 (NV12)
    SwsContext*      swsCtx_     = nullptr;
    std::vector<uint8_t> bgrBuf_;
    int              bgrBufSize_ = 0;

    int videoStreamIdx_ = -1;
    int width_  = 0;
    int height_ = 0;
};

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::signal(SIGINT,  onSignal);
    std::signal(SIGTERM, onSignal);

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <rtmp_url>\n"
                  << "Example: " << argv[0] << " rtmp://192.168.100.12/live/in\n" << std::endl;
        return 1;
    }
    std::string url = argv[1];

    av_log_set_level(AV_LOG_DEBUG);

    RtmpHardDecoder dec;
    if (!dec.open(url)) {
        return 2;
    }
    std::cout << "开始解码 (" << dec.width() << "x" << dec.height() << "), 按 Ctrl+C 退出" << std::endl;

    dec.run([&](const cv::Mat& bgr) {
        // 1) 亮度均值
        cv::Scalar mean = cv::mean(bgr);
        std::cout << "frame " << bgr.cols << "x" << bgr.rows
            << "  BGR mean=(" << mean[0] << "," << mean[1] << "," << mean[2] << ")" << std::endl;
    });

    std::cout << "退出" << std::endl;
    return 0;
}
