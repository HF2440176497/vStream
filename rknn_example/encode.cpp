/**
 * ffmpeg-rockchip 稳定推流Demo
 * 
 * 功能：循环使用 Rockchip MPP 硬件编码器，将一张图片编码为 H.264 并通过 RTMP 推送。
 * 解决：通过主动请求并发送额外数据(SPS/PPS)，确保硬件编码器正常工作，并能被服务器正确解析。
 */

#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <cstring>

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

#define CHECK_AV_RET(expr, msg) \
    do { \
        int check_av_ret = (expr); \
        if (check_av_ret < 0) { \
            char errbuf[256]; \
            av_strerror(check_av_ret, errbuf, sizeof(errbuf)); \
            std::cerr << "[ERROR] " << msg << ": " << errbuf << std::endl; \
            return false; \
        } \
    } while(0)

struct PushConfig {
    int width = 960;
    int height = 640;
    int fps = 25;
    int gop_size = 50;
    int bitrate = 4000000;
    AVPixelFormat pix_fmt = AV_PIX_FMT_NV12;
    std::string codec_name = "h264_rkmpp";
};

class RtmpPusher {
public:
    RtmpPusher(const PushConfig& config) : config_(config) {}
    ~RtmpPusher() { cleanup(); }

    bool init(const std::string& output_url) {
        int ret = 0;
        output_url_ = output_url;

        // 1. 创建输出上下文 (RTMP)
        ret = avformat_alloc_output_context2(&fmt_ctx_, nullptr, "flv", output_url_.c_str());
        CHECK_AV_RET(ret, "创建输出上下文失败");

        // 2. 查找硬件编码器并创建编码器上下文
        const AVCodec* codec = avcodec_find_encoder_by_name(config_.codec_name.c_str());
        if (!codec) {
            std::cerr << "[ERROR] 找不到编码器: " << config_.codec_name << std::endl;
            return false;
        }
        std::cout << "[INFO] 使用编码器: " << config_.codec_name << std::endl;

        stream_ = avformat_new_stream(fmt_ctx_, nullptr);
        if (!stream_) {
            std::cerr << "[ERROR] avformat_new_stream 失败" << std::endl;
            return false;
        }
        stream_->time_base = (AVRational){1, config_.fps}; // 设置流的时基

        codec_ctx_ = avcodec_alloc_context3(codec);
        if (!codec_ctx_) {
            std::cerr << "[ERROR] 无法分配编码器上下文" << std::endl;
            return false;
        }

        // 3. 设置编码参数 (关键)
        codec_ctx_->width = config_.width;
        codec_ctx_->height = config_.height;
        codec_ctx_->time_base = stream_->time_base;
        codec_ctx_->framerate = (AVRational){config_.fps, 1};
        codec_ctx_->pix_fmt = config_.pix_fmt;
        codec_ctx_->bit_rate = config_.bitrate;
        codec_ctx_->gop_size = config_.gop_size;
        if (codec->id == AV_CODEC_ID_H264) {
            codec_ctx_->profile = FF_PROFILE_H264_MAIN;
        }
        // 关键：让编码器把 SPS/PPS 写入 extradata，
        // FLV 封装器在写 Header 时会用它生成 AVCDecoderConfigurationRecord。
        // 否则 SPS/PPS 会被编码器当作普通 NALU 跟在 I 帧前后发出去，
        // RTMP 服务器无法提前建立 H.264 解码上下文 → Broken pipe。
        codec_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

        // 4. 配置编码器选项
        AVDictionary* opts = nullptr;
        av_dict_set(&opts, "keyint", std::to_string(config_.gop_size).c_str(), 0);
        av_dict_set(&opts, "min-keyint", std::to_string(config_.gop_size / 2).c_str(), 0);
        // 重要：设置为0以阻止内部分析，确保在关键帧时发送SPS/PPS
        av_dict_set(&opts, "scenecut", "0", 0);
        // 推荐：为流媒体设置低延迟参数
        av_dict_set(&opts, "tune", "zerolatency", 0); 

        // 4.5 检测是否为硬件编码器，若是则建立硬件设备 / 帧上下文
        //     (h264_rkmpp / h264_nvenc / h264_vaapi 等都属于此类)
        is_hw_ = (getHwConfig(codec) != nullptr);
        std::cout << "[INFO] 编码器模式: " << (is_hw_ ? "硬件 (HW Device)" : "软件 (CPU)") << std::endl;
        if (is_hw_) {
            if (!setupHardwareContext(codec)) {
                return false;
            }
        }

        ret = avcodec_open2(codec_ctx_, codec, &opts);
        av_dict_free(&opts);
        CHECK_AV_RET(ret, "打开编码器失败");

        // 检查 SPS/PPS 是否被编码器填入 extradata，
        // 这一步是 RTMP/FLV 推流能否成功的关键。
        if (codec_ctx_->extradata && codec_ctx_->extradata_size > 0) {
            std::cout << "[INFO] SPS/PPS extradata: " << codec_ctx_->extradata_size
                      << " bytes (前16字节): ";
            for (int i = 0; i < std::min(codec_ctx_->extradata_size, 16); i++) {
                printf("%02x ", codec_ctx_->extradata[i]);
            }
            printf("\n");
        } else {
            std::cerr << "[WARN] extradata 为空！RKMPP 硬件编码器可能未生成 SPS/PPS，"
                      << "需在拿到首包后从 side_data 提取并手工填入 extradata" << std::endl;
        }

        // 5. 将编码器参数复制到输出流
        ret = avcodec_parameters_from_context(stream_->codecpar, codec_ctx_);
        CHECK_AV_RET(ret, "复制编码器参数失败");

        // 6. 打开网络IO
        if (!(fmt_ctx_->oformat->flags & AVFMT_NOFILE)) {
            ret = avio_open(&fmt_ctx_->pb, output_url_.c_str(), AVIO_FLAG_WRITE);
            CHECK_AV_RET(ret, "打开网络连接失败");
        }

        // 7. 写入文件头
        // ret = avformat_write_header(fmt_ctx_, nullptr);
        AVDictionary* mux_opts = nullptr;
        av_dict_set(&mux_opts, "flvflags", "no_duration_filesize", 0);
        ret = avformat_write_header(fmt_ctx_, &mux_opts);
        av_dict_free(&mux_opts);
        CHECK_AV_RET(ret, "写入文件头失败");

        // 8. 初始化图像转换器 (BGR -> NV12)
        sws_ctx_ = sws_getContext(
            config_.width, config_.height, AV_PIX_FMT_BGR24,
            config_.width, config_.height, config_.pix_fmt,
            SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!sws_ctx_) {
            std::cerr << "[ERROR] 创建图像转换上下文失败" << std::endl;
            return false;
        }

        // 9. 预分配编码帧并获取缓冲区
        //    - frame_    : CPU 侧 NV12 软件帧 (sws_scale 输出到此)
        //    - hw_frame_ : 硬件侧帧 (上传后送给硬件编码器)
        frame_ = av_frame_alloc();
        if (!frame_) {
            std::cerr << "[ERROR] 分配帧失败" << std::endl;
            return false;
        }
        frame_->format = config_.pix_fmt;
        frame_->width = config_.width;
        frame_->height = config_.height;
        ret = av_frame_get_buffer(frame_, 0);
        CHECK_AV_RET(ret, "分配帧缓冲区失败");

        if (is_hw_) {
            
            hw_frame_ = av_frame_alloc();
            if (!hw_frame_) {
                std::cerr << "[ERROR] 分配硬件帧失败" << std::endl;
                return false;
            }
            ret = av_hwframe_get_buffer(codec_ctx_->hw_frames_ctx, hw_frame_, 0);
            CHECK_AV_RET(ret, "获取硬件帧缓冲区失败");
        }

        pkt_ = av_packet_alloc();
        if (!pkt_) {
            std::cerr << "[ERROR] 分配包失败" << std::endl;
            return false;
        }

        start_time_ = av_gettime();
        frame_count_ = 0;
        std::cout << "[INFO] 推流器初始化完成，准备向 " << output_url_ << " 推流" << std::endl;
        return true;
    }

    /**
     * 编码并发送一帧 (核心逻辑)
     * @param bgr_data OpenCV Mat 的 data 指针
     * @param linesize 每行的字节数 (通常为 width * 3)
     */
    bool sendFrame(const uint8_t* bgr_data, int linesize) {
        if (!frame_ || !codec_ctx_) return false;
        int ret = 0;
        ret = av_frame_make_writable(frame_);
        if (ret < 0) {
            std::cerr << "[ERROR] 无法获取可写帧缓冲区" << std::endl;
            return false;
        }
        // BGR -> NV12 格式转换
        const uint8_t* src_data[1] = {bgr_data};
        int src_linesize[1] = {linesize};
        sws_scale(sws_ctx_, src_data, src_linesize, 0, config_.height,
                  frame_->data, frame_->linesize);
        frame_->pts = frame_count_++;
        AVFrame* send_frame = frame_;
        if (is_hw_) {
            ret = av_hwframe_transfer_data(hw_frame_, frame_, 0);
            CHECK_AV_RET(ret, "上传帧到硬件设备失败");
            hw_frame_->pts = frame_->pts;
            send_frame = hw_frame_;
        }
        // 发送帧到编码器
        ret = avcodec_send_frame(codec_ctx_, send_frame);
        if (ret < 0) {
            char errbuf[256];
            av_strerror(ret, errbuf, sizeof(errbuf));
            std::cerr << "[ERROR] 发送帧到编码器失败: " << errbuf << std::endl;
            return false;
        }
        // 循环从编码器中取出所有已编码的包
        while (ret >= 0) {
            ret = avcodec_receive_packet(codec_ctx_, pkt_);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                break;
            } else if (ret < 0) {
                std::cerr << "[ERROR] 编码失败" << std::endl;
                return false;
            }
            // 调整PTS到流的时基
            av_packet_rescale_ts(pkt_, codec_ctx_->time_base, stream_->time_base);
            pkt_->stream_index = stream_->index;

            std::cout << "[DEBUG] pkt: pts=" << pkt_->pts
                      << " dts=" << pkt_->dts
                      << " size=" << pkt_->size
                      << " key=" << ((pkt_->flags & AV_PKT_FLAG_KEY) ? "Y" : "N")
                      << std::endl;

            int64_t expected_pts_us = av_rescale_q(pkt_->pts, stream_->time_base, 
                                                    (AVRational){1, 1000000});
            int64_t now = av_gettime() - start_time_;
            if (expected_pts_us > now) {
                av_usleep(expected_pts_us - now);
            }
            ret = av_interleaved_write_frame(fmt_ctx_, pkt_);
            av_packet_unref(pkt_);
            CHECK_AV_RET(ret, "发送包失败");
        }
        return true;
    }

    bool sendEof() {
        if (!codec_ctx_) return false;
        if (!fmt_ctx_) return true;

        int ret = avcodec_send_frame(codec_ctx_, nullptr);
        if (ret < 0) return false;

        while (ret >= 0) {
            ret = avcodec_receive_packet(codec_ctx_, pkt_);
            if (ret == AVERROR_EOF) break;
            if (ret < 0) return false;

            av_packet_rescale_ts(pkt_, codec_ctx_->time_base, stream_->time_base);
            pkt_->stream_index = stream_->index;
            av_interleaved_write_frame(fmt_ctx_, pkt_);
            av_packet_unref(pkt_);
        }

        av_write_trailer(fmt_ctx_);
        std::cout << "[INFO] 流已正常结束" << std::endl;
        return true;
    }

private:
    void cleanup() {
        if (sws_ctx_) sws_freeContext(sws_ctx_);
        if (pkt_) av_packet_free(&pkt_);
        if (hw_frame_) av_frame_free(&hw_frame_);
        if (frame_) av_frame_free(&frame_);
        if (codec_ctx_) avcodec_free_context(&codec_ctx_);
        if (hw_device_ctx_) av_buffer_unref(&hw_device_ctx_);
        if (fmt_ctx_) {
            if (!(fmt_ctx_->oformat->flags & AVFMT_NOFILE) && fmt_ctx_->pb)
                avio_closep(&fmt_ctx_->pb);
            avformat_free_context(fmt_ctx_);
        }
    }

    const AVCodecHWConfig* getHwConfig(const AVCodec* codec) {
        for (int i = 0; ; i++) {
            const AVCodecHWConfig* cfg = avcodec_get_hw_config(codec, i);
            if (!cfg) break;
            if (cfg->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) {
                return cfg;
            }
        }
        return nullptr;
    }

    /**
     * 为硬件编码器建立 AVHWDeviceContext + AVHWFramesContext，
     * 并绑定到 codec_ctx_ 上。必须在 avcodec_open2 之前设置。
     */
    bool setupHardwareContext(const AVCodec* codec) {
        const AVCodecHWConfig* hw_cfg = getHwConfig(codec);
        if (!hw_cfg) {
            std::cerr << "[ERROR] 编码器 " << codec->name << " 未声明硬件设备上下文" << std::endl;
            return false;
        }

        const char* dev_name = av_hwdevice_get_type_name(hw_cfg->device_type);
        std::cout << "[INFO] 检测到硬件编码器，需使用设备类型: "
                  << (dev_name ? dev_name : "unknown") << std::endl;

        // 1. 创建硬件设备上下文
        int ret = av_hwdevice_ctx_create(&hw_device_ctx_, hw_cfg->device_type,
                                         nullptr, nullptr, 0);
        CHECK_AV_RET(ret, "创建硬件设备上下文失败");

        std::cout << "[INFO] 硬件设备上下文创建成功" << std::endl;

        // 2. 绑定到编码器上下文
        codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);

        // 3. 分配硬件帧上下文
        AVBufferRef* hw_frames_ref = av_hwframe_ctx_alloc(hw_device_ctx_);
        if (!hw_frames_ref) {
            std::cerr << "[ERROR] 分配硬件帧上下文失败" << std::endl;
            return false;
        }
        AVHWFramesContext* hw_frames_ctx = (AVHWFramesContext*)hw_frames_ref->data;
        hw_frames_ctx->format    = AV_PIX_FMT_DRM_PRIME;
        // hw_frames_ctx->format    = hw_cfg->pix_fmt;
        hw_frames_ctx->sw_format = config_.pix_fmt;
        hw_frames_ctx->width     = config_.width;
        hw_frames_ctx->height    = config_.height;
        hw_frames_ctx->initial_pool_size = 20;

        ret = av_hwframe_ctx_init(hw_frames_ref);
        CHECK_AV_RET(ret, "初始化硬件帧上下文失败");

        if (ret < 0) {
            av_buffer_unref(&hw_frames_ref);
            return false;
        }
        std::cout << "[INFO] 硬件帧上下文就绪: hw_format="
                  << av_get_pix_fmt_name(hw_frames_ctx->format)
                  << ", sw_format=" << av_get_pix_fmt_name(hw_frames_ctx->sw_format) << std::endl;

        codec_ctx_->hw_frames_ctx = av_buffer_ref(hw_frames_ref);
        av_buffer_unref(&hw_frames_ref);

        // 实际的硬件编码帧
        codec_ctx_->pix_fmt = hw_frames_ctx->format;

        return true;
    }

    PushConfig config_;
    AVCodecContext* codec_ctx_ = nullptr;
    AVFormatContext* fmt_ctx_ = nullptr;
    AVStream* stream_ = nullptr;
    AVFrame* frame_ = nullptr;          // CPU 侧 NV12 帧 (软件帧)
    AVFrame* hw_frame_ = nullptr;       // 硬件侧帧 (送往编码器)
    AVPacket* pkt_ = nullptr;
    SwsContext* sws_ctx_ = nullptr;
    AVBufferRef* hw_device_ctx_ = nullptr;  // 硬件设备上下文
    int64_t start_time_ = 0;
    int64_t frame_count_ = 0;
    std::string output_url_;
    bool is_hw_ = false;                // 当前编码器是否为硬件编码器
};

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "用法: " << argv[0] << " <图片路径> <RTMP地址> [推流时长(秒)]" << std::endl;
        std::cout << "示例: " << argv[0] << " test.jpg rtmp://192.168.100.12/live/stream 60" << std::endl;
        return 1;
    }

    av_log_set_level(AV_LOG_DEBUG);
    av_log_set_callback([](void* ptr, int level, const char* fmt, va_list vl) {
        char line[1024];
        vsnprintf(line, sizeof(line), fmt, vl);
        if (level <= AV_LOG_ERROR) {
            std::cerr << "[FFmpeg E] " << line;
        } else if (level <= AV_LOG_WARNING) {
            std::cerr << "[FFmpeg W] " << line;
        } else {
            std::cout << "[FFmpeg]   " << line;
        }
    });

    std::string image_path = argv[1];
    std::string output_url = argv[2];
    int duration_seconds = (argc > 3) ? std::stoi(argv[3]) : 0; // 0 表示无限循环

    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "[ERROR] 无法读取图片: " << image_path << std::endl;
        return 1;
    }

    PushConfig config;
    config.width = image.cols;
    config.height = image.rows;
    config.fps = 25;
    config.bitrate = 4000000; // 4 Mbps

    if (config.width % 2 != 0 || config.height % 2 != 0) {
        std::cerr << "[WARN] 图片宽高不是偶数，可能会导致编码失败。建议先缩放图片。" << std::endl;
    }

    std::cout << "[INFO] 图片分辨率: " << config.width << "x" << config.height << std::endl;
    std::cout << "[INFO] 目标帧率: " << config.fps << " fps" << std::endl;
    std::cout << "[INFO] 目标码率: " << config.bitrate / 1000 << " kbps" << std::endl;

    if (!image.isContinuous()) {
        image = image.clone();
    }

    RtmpPusher pusher(config);
    if (!pusher.init(output_url)) {
        std::cerr << "[ERROR] 推流器初始化失败" << std::endl;
        return 1;
    }

    int total_frames = 0;
    auto start_time = std::chrono::steady_clock::now();

    std::cout << "[INFO] 开始推流..." << std::endl;
    while (true) {
        if (!pusher.sendFrame(image.data, image.step[0])) {
            std::cerr << "[ERROR] 发送帧失败，停止推流" << std::endl;
            break;
        }
        total_frames++;
        if (total_frames % config.fps == 0) {
            std::cout << "[INFO] 已发送 " << total_frames << " 帧，当前帧率稳定在 " << config.fps << std::endl;
        }
        if (duration_seconds > 0) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            if (elapsed >= duration_seconds) {
                std::cout << "[INFO] 已达到预设推流时长 (" << duration_seconds << " 秒)，准备退出" << std::endl;
                break;
            }
        }
    }

    pusher.sendEof();
    std::cout << "[INFO] 推流结束，共发送 " << total_frames << " 帧" << std::endl;
    return 0;
}