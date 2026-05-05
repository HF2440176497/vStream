#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <string>
#include <atomic>
#include <csignal>

#include "util/cnstream_queue.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/hwcontext.h>
#include <libswscale/swscale.h>
}

using namespace cnstream;

std::atomic<bool> g_running{true};

void signal_handler(int) {
    g_running = false;
}


void producer(ThreadSafeQueue<cv::Mat>& queue,
              const std::string& image_path,
              int fps)
{
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "[Producer] Failed to load image: " << image_path << std::endl;
        queue.Stop();
        return;
    }
    std::cout << "[Producer] Image loaded: " << img.cols << "x" << img.rows << std::endl;

    auto frame_interval = std::chrono::milliseconds(1000 / fps);
    auto next_time = std::chrono::steady_clock::now();

    while (g_running.load()) {
        queue.Push(img.clone());
        next_time += frame_interval;
        auto now = std::chrono::steady_clock::now();
        if (next_time > now) {
            std::this_thread::sleep_until(next_time);
        } else {
            next_time = now + frame_interval;
        }
    }
    queue.Stop();
    std::cout << "[Producer] Stopped." << std::endl;
}


struct StreamContext {
    AVFormatContext* fmt_ctx      = nullptr;
    AVCodecContext*  codec_ctx    = nullptr;
    AVStream*        stream       = nullptr;
    AVBufferRef*     hw_device_ctx = nullptr;
    AVBufferRef*     hw_frames_ctx = nullptr;
    SwsContext*      sws_ctx      = nullptr;
    AVFrame*         sw_frame     = nullptr;
    AVFrame*         hw_frame     = nullptr;
    int64_t          frame_idx    = 0;
};

std::string get_format_from_url(const std::string& url) {
    if (url.find("rtmp://") == 0) return "flv";
    if (url.find("rtsp://") == 0) return "rtsp";
    if (url.find("http://") == 0 || url.find("https://") == 0) return "mpegts";
    return "flv";
}

bool init_stream(StreamContext& ctx,
                 int width, int height,
                 int fps, int bitrate_kbps,
                 const std::string& rtmp_url,
                 int device_id)
{

    AVOutputFormat* fmt = const_cast<AVOutputFormat*>(av_guess_format(get_format_from_url(rtmp_url).c_str(), NULL, NULL));
    if (!fmt) {
        std::cerr << "[Stream] Unknown format." << std::endl;
        return false;
    }

    int ret = avformat_alloc_output_context2(&ctx.fmt_ctx, nullptr, fmt->name, rtmp_url.c_str());
    if (ret < 0 || !ctx.fmt_ctx) {
        std::cerr << "[Stream] avformat_alloc_output_context2 failed." << std::endl;
        return false;
    }

    const AVCodec* codec = avcodec_find_encoder_by_name("h264_nvenc");
    if (!codec) {
        std::cerr << "[Stream] h264_nvenc encoder not found. Check FFmpeg build with --enable-nvenc." << std::endl;
        return false;
    }

    ctx.stream = avformat_new_stream(ctx.fmt_ctx, nullptr);
    if (!ctx.stream) {
        std::cerr << "[Stream] avformat_new_stream failed." << std::endl;
        return false;
    }

    ctx.codec_ctx = avcodec_alloc_context3(codec);
    if (!ctx.codec_ctx) {
        std::cerr << "[Stream] avcodec_alloc_context3 failed." << std::endl;
        return false;
    }

    ctx.codec_ctx->codec_id   = codec->id;
    ctx.codec_ctx->codec_type = AVMEDIA_TYPE_VIDEO;
    ctx.codec_ctx->width      = width;
    ctx.codec_ctx->height     = height;
    ctx.codec_ctx->time_base  = {1, fps};
    ctx.codec_ctx->framerate  = {fps, 1};
    ctx.codec_ctx->bit_rate   = bitrate_kbps * 1000;
    ctx.codec_ctx->gop_size   = fps;
    ctx.codec_ctx->max_b_frames = 1;

    ctx.stream->time_base = ctx.codec_ctx->time_base;

    ret = av_hwdevice_ctx_create(&ctx.hw_device_ctx, AV_HWDEVICE_TYPE_CUDA,
                                 std::to_string(device_id).c_str(), nullptr, 0);
    if (ret < 0) {
        std::cerr << "[Stream] av_hwdevice_ctx_create (CUDA) failed: " << ret << std::endl;
        return false;
    }

    ctx.hw_frames_ctx = av_hwframe_ctx_alloc(ctx.hw_device_ctx);
    if (!ctx.hw_frames_ctx) {
        std::cerr << "[Stream] av_hwframe_ctx_alloc failed." << std::endl;
        return false;
    }

    AVHWFramesContext* hw_frames = reinterpret_cast<AVHWFramesContext*>(ctx.hw_frames_ctx->data);
    hw_frames->format    = AV_PIX_FMT_CUDA;
    hw_frames->sw_format = AV_PIX_FMT_NV12;
    hw_frames->width     = width;
    hw_frames->height    = height;
    hw_frames->initial_pool_size = 20;

    ret = av_hwframe_ctx_init(ctx.hw_frames_ctx);
    if (ret < 0) {
        std::cerr << "[Stream] av_hwframe_ctx_init failed: " << ret << std::endl;
        return false;
    }

    ctx.codec_ctx->hw_device_ctx = av_buffer_ref(ctx.hw_device_ctx);
    ctx.codec_ctx->hw_frames_ctx = av_buffer_ref(ctx.hw_frames_ctx);
    ctx.codec_ctx->pix_fmt       = AV_PIX_FMT_CUDA;

    if (ctx.fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
        ctx.codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    ret = avcodec_open2(ctx.codec_ctx, codec, nullptr);
    if (ret < 0) {
        std::cerr << "[Stream] avcodec_open2 failed: " << ret << std::endl;
        return false;
    }

    ret = avcodec_parameters_from_context(ctx.stream->codecpar, ctx.codec_ctx);
    if (ret < 0) {
        std::cerr << "[Stream] avcodec_parameters_from_context failed." << std::endl;
        return false;
    }

    if (!(ctx.fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&ctx.fmt_ctx->pb, rtmp_url.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) {
            std::cerr << "[Stream] avio_open failed. Check RTMP URL." << std::endl;
            return false;
        }
    }

    ret = avformat_write_header(ctx.fmt_ctx, nullptr);
    if (ret < 0) {
        std::cerr << "[Stream] avformat_write_header failed." << std::endl;
        return false;
    }

    ctx.sws_ctx = sws_getCachedContext(
        ctx.sws_ctx,
        width, height, AV_PIX_FMT_BGR24,
        width, height, AV_PIX_FMT_NV12,
        SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!ctx.sws_ctx) {
        std::cerr << "[Stream] sws_getContext (BGR->NV12) failed." << std::endl;
        return false;
    }

    ctx.sw_frame = av_frame_alloc();
    ctx.sw_frame->format = AV_PIX_FMT_NV12;
    ctx.sw_frame->width  = width;
    ctx.sw_frame->height = height;
    ret = av_frame_get_buffer(ctx.sw_frame, 0);
    if (ret < 0) {
        std::cerr << "[Stream] av_frame_get_buffer (sw_frame) failed." << std::endl;
        return false;
    }

    ctx.hw_frame = av_frame_alloc();
    ret = av_hwframe_get_buffer(ctx.hw_frames_ctx, ctx.hw_frame, 0);
    if (ret < 0) {
        std::cerr << "[Stream] av_hwframe_get_buffer failed: " << ret << std::endl;
        return false;
    }

    std::cout << "[Stream] Initialized (NVENC GPU=" << device_id << "). RTMP: " << rtmp_url
              << " | Resolution: " << width << "x" << height
              << " | FPS: " << fps << " | Bitrate: " << bitrate_kbps << "kbps" << std::endl;
    return true;
}


bool send_frame(StreamContext& ctx, const cv::Mat& img) {
    cv::Mat img_pre;
    if (img.cols != ctx.codec_ctx->width || img.rows != ctx.codec_ctx->height) {
        cv::resize(img, img_pre, cv::Size(ctx.codec_ctx->width, ctx.codec_ctx->height));
    } else {
        img_pre = img;
    }

    cv::Mat bgr = img_pre.isContinuous() ? img_pre : img_pre.clone();

    const uint8_t* src_data[4] = { bgr.data, nullptr, nullptr, nullptr };
    int src_linesize[4] = { static_cast<int>(bgr.step[0]), 0, 0, 0 };

    int ret = av_frame_make_writable(ctx.sw_frame);
    if (ret < 0) {
        std::cerr << "[Stream] av_frame_make_writable (sw_frame) failed." << std::endl;
        return false;
    }

    sws_scale(ctx.sws_ctx,
              src_data, src_linesize,
              0, ctx.codec_ctx->height,
              ctx.sw_frame->data, ctx.sw_frame->linesize);

    ctx.sw_frame->pts = ctx.frame_idx;

    ret = av_frame_make_writable(ctx.hw_frame);
    if (ret < 0) {
        std::cerr << "[Stream] av_frame_make_writable (hw_frame) failed." << std::endl;
        return false;
    }

    ret = av_hwframe_transfer_data(ctx.hw_frame, ctx.sw_frame, 0);
    if (ret < 0) {
        std::cerr << "[Stream] av_hwframe_transfer_data failed: " << ret << std::endl;
        return false;
    }

    ctx.hw_frame->pts = ctx.frame_idx++;

    ret = avcodec_send_frame(ctx.codec_ctx, ctx.hw_frame);
    if (ret < 0) {
        std::cerr << "[Stream] avcodec_send_frame error: " << ret << std::endl;
        return false;
    }

    AVPacket* pkt = av_packet_alloc();
    while (true) {
        ret = avcodec_receive_packet(ctx.codec_ctx, pkt);

        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        }
        if (ret < 0) {
            std::cerr << "[Stream] avcodec_receive_packet error." << std::endl;
            av_packet_free(&pkt);
            return false;
        }
        av_packet_rescale_ts(pkt, ctx.codec_ctx->time_base, ctx.stream->time_base);
        pkt->stream_index = ctx.stream->index;
        ret = av_interleaved_write_frame(ctx.fmt_ctx, pkt);
        if (ret < 0) {
            std::cerr << "[Stream] av_interleaved_write_frame error." << std::endl;
            av_packet_free(&pkt);
            return false;
        }
    }
    av_packet_free(&pkt);
    return true;
}

void consumer(ThreadSafeQueue<cv::Mat>& queue, StreamContext& ctx) {
    cv::Mat frame;
    while (g_running.load()) {
        if (!queue.TryPop(frame)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        if (frame.empty()) continue;
        if (!send_frame(ctx, frame)) {
            std::cerr << "[Consumer] send_frame failed, stopping." << std::endl;
            break;
        }
    }
    std::cout << "[Consumer] Stopped." << std::endl;
}


void cleanup(StreamContext& ctx) {
    avcodec_send_frame(ctx.codec_ctx, nullptr);
    AVPacket* pkt = av_packet_alloc();
    while (avcodec_receive_packet(ctx.codec_ctx, pkt) == 0) {
        av_packet_rescale_ts(pkt, ctx.codec_ctx->time_base, ctx.stream->time_base);
        pkt->stream_index = ctx.stream->index;
        av_interleaved_write_frame(ctx.fmt_ctx, pkt);
    }
    av_packet_free(&pkt);

    av_write_trailer(ctx.fmt_ctx);

    if (ctx.hw_frame)    av_frame_free(&ctx.hw_frame);
    if (ctx.sw_frame)    av_frame_free(&ctx.sw_frame);
    if (ctx.sws_ctx)     sws_freeContext(ctx.sws_ctx);
    if (ctx.hw_frames_ctx) av_buffer_unref(&ctx.hw_frames_ctx);
    if (ctx.hw_device_ctx) av_buffer_unref(&ctx.hw_device_ctx);
    if (ctx.codec_ctx)   avcodec_free_context(&ctx.codec_ctx);
    if (ctx.fmt_ctx) {
        if (!(ctx.fmt_ctx->oformat->flags & AVFMT_NOFILE))
            avio_closep(&ctx.fmt_ctx->pb);
        avformat_free_context(ctx.fmt_ctx);
    }
    std::cout << "[Stream] Cleanup done." << std::endl;
}


int main(int argc, char* argv[]) {
    std::string image_path = (argc > 1) ? argv[1] : "image.png";
    std::string rtmp_url   = (argc > 2) ? argv[2] : "rtmp://127.0.0.1:1935/live/stream";
    int fps                = (argc > 3) ? std::stoi(argv[3]) : 25;
    int width              = (argc > 4) ? std::stoi(argv[4]) : 640;
    int height             = (argc > 5) ? std::stoi(argv[5]) : 480;
    int bitrate_kbps       = (argc > 6) ? std::stoi(argv[6]) : 800;
    int duration_sec       = (argc > 7) ? std::stoi(argv[7]) : 30;
    int device_id          = (argc > 8) ? std::stoi(argv[8]) : 0;

    std::cout << "Image:  " << image_path << std::endl;
    std::cout << "RTMP:   " << rtmp_url << std::endl;
    std::cout << "FPS:    " << fps << std::endl;
    std::cout << "Res:    " << width << "x" << height << std::endl;
    std::cout << "BR:     " << bitrate_kbps << "kbps" << std::endl;
    std::cout << "Time:   " << duration_sec << "s" << std::endl;
    std::cout << "GPU:    " << device_id << std::endl;

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    StreamContext ctx;
    if (!init_stream(ctx, width, height, fps, bitrate_kbps, rtmp_url, device_id)) {
        std::cerr << "Stream init failed!" << std::endl;
        return -1;
    }

    ThreadSafeQueue<cv::Mat> queue;
    std::thread consumer_thread(consumer, std::ref(queue), std::ref(ctx));
    std::thread producer_thread(producer, std::ref(queue), image_path, fps);

    std::this_thread::sleep_for(std::chrono::seconds(duration_sec));
    g_running = false;

    producer_thread.join();
    consumer_thread.join();
    cleanup(ctx);

    std::cout << "========== NVENC Demo Finished ==========" << std::endl;
    return 0;
}
