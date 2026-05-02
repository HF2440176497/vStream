
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
            next_time = now + frame_interval;  // 重置 next_time 用于下一次判断
        }
    }
    queue.Stop();
    std::cout << "[Producer] Stopped." << std::endl;
}


struct StreamContext {
    AVFormatContext* fmt_ctx   = nullptr;
    AVCodecContext*  codec_ctx = nullptr;
    AVStream*        stream    = nullptr;
    SwsContext*      sws_ctx   = nullptr;
    AVFrame*         sws_frame = nullptr;  // sws_scale 输出的YUV帧
    int64_t          frame_idx = 0;
};

std::string get_format_from_url(const std::string& url) {
    if (url.find("rtmp://") == 0) return "flv";
    if (url.find("rtsp://") == 0) return "rtsp";
    if (url.find("http://") == 0 || url.find("https://") == 0) return "mpegts";
    return "flv"; // 默认
}

bool init_stream(StreamContext& ctx,
                 int width, int height,
                 int fps, int bitrate_kbps,
                 const std::string& rtmp_url)
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

    const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!codec) {
        std::cerr << "[Stream] H.264 encoder not found." << std::endl;
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

    ctx.codec_ctx->codec_id = codec->id;
    ctx.codec_ctx->codec_type = AVMEDIA_TYPE_VIDEO;

    ctx.codec_ctx->width     = width;
    ctx.codec_ctx->height    = height;
    ctx.codec_ctx->time_base = {1, fps};          // 编码器时间基=帧率倒数
    ctx.codec_ctx->framerate = {fps, 1};          // 帧率
    ctx.codec_ctx->pix_fmt   = AV_PIX_FMT_YUV420P; // 像素格式
    ctx.codec_ctx->bit_rate  = bitrate_kbps * 1000; // 码率
    ctx.codec_ctx->gop_size  = fps;                // GOP大小
    ctx.codec_ctx->max_b_frames = 1;

    ctx.stream->time_base = ctx.codec_ctx->time_base;

    if (ctx.fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
        ctx.codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    ret = avcodec_open2(ctx.codec_ctx, codec, nullptr);
    if (ret < 0) {
        std::cerr << "[Stream] avcodec_open2 failed." << std::endl;
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

    // src: BGR24 -> dst: YUV420P
    ctx.sws_ctx = sws_getCachedContext(
        ctx.sws_ctx,
        width, height, AV_PIX_FMT_BGR24,
        width, height, AV_PIX_FMT_YUV420P,
        SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!ctx.sws_ctx) {
        std::cerr << "[Stream] sws_getContext failed." << std::endl;
        return false;
    }

    ctx.sws_frame = av_frame_alloc();
    ctx.sws_frame->format = AV_PIX_FMT_YUV420P;
    ctx.sws_frame->width  = width;
    ctx.sws_frame->height = height;
    ret = av_frame_get_buffer(ctx.sws_frame, 0);
    if (ret < 0) {
        std::cerr << "[Stream] av_frame_get_buffer failed." << std::endl;
        return false;
    }

    std::cout << "[Stream] Initialized. RTMP: " << rtmp_url
              << " | Resolution: " << width << "x" << height
              << " | FPS: " << fps << " | Bitrate: " << bitrate_kbps << "kbps" << std::endl;
    return true;
}

/**
 * av_frame_get_buffer 会自动分配 frame->buf，并且 frame->data 指向缓冲区
 * av_image_fill_arrays 不会产生新的缓冲区，只是指向给定的缓冲区
 */
bool send_frame(StreamContext& ctx, const cv::Mat& img) {
    AVFrame* bgr_frame = av_frame_alloc();
    bgr_frame->format = AV_PIX_FMT_BGR24;
    bgr_frame->width  = ctx.codec_ctx->width;
    bgr_frame->height = ctx.codec_ctx->height;
    int ret = av_frame_get_buffer(bgr_frame, 0);
    if (ret < 0) {
        av_frame_free(&bgr_frame);
        std::cerr << "[Stream] av_frame_get_buffer (bgr) failed." << std::endl;
        return false;
    }

    cv::Mat img_pre;
    if (img.cols != ctx.codec_ctx->width || img.rows != ctx.codec_ctx->height) {
        cv::resize(img, img_pre, cv::Size(ctx.codec_ctx->width, ctx.codec_ctx->height));
    } else {
        img_pre = img;
    }

    if (!img_pre.isContinuous()) {
        std::cerr << "[WARN] Image is not continuous." << std::endl;
    }
    cv::Mat bgr = img_pre.isContinuous() ? img_pre : img_pre.clone();

    const uint8_t* src_data[4] = { bgr.data, nullptr, nullptr, nullptr };
    int src_linesize[4] = { static_cast<int>(bgr.step[0]), 0, 0, 0 };

    ret = av_frame_make_writable(ctx.sws_frame);
    if (ret < 0) {
        std::cerr << "[Stream] av_frame_make_writable failed." << std::endl;
        return false;
    }

    // int sws_scale(struct SwsContext * c,
    //     const uint8_t *const 	srcSlice[],
    //     const int 	srcStride[],
    //     int 	srcSliceY,
    //     int 	srcSliceH,
    //     uint8_t *const 	dst[],
    //     const int 	dstStride[] 
    // )	

    sws_scale(ctx.sws_ctx,
              src_data, src_linesize,
              0, ctx.codec_ctx->height,
              ctx.sws_frame->data, ctx.sws_frame->linesize);

    // 将 cv::Mat 数据拷贝到 bgr_frame
    // int ret_ = av_frame_make_writable(bgr_frame);
    // if (ret_ < 0) { av_frame_free(&bgr_frame); return false; }

    // for (int y = 0; y < img.rows; y++) {
    //     memcpy(bgr_frame->data[0] + y * bgr_frame->linesize[0],
    //            img.ptr(y),
    //            bgr_frame->linesize[0]);
    // }

    // ret_ = av_frame_make_writable(ctx.sws_frame);
    // if (ret_ < 0) { av_frame_free(&bgr_frame); return false; }
    // sws_scale(ctx.sws_ctx,
    //           bgr_frame->data, bgr_frame->linesize,
    //           0, ctx.codec_ctx->height,
    //           ctx.sws_frame->data, ctx.sws_frame->linesize);
    // av_frame_free(&bgr_frame);

    ctx.sws_frame->pts = ctx.frame_idx++;
    ret = avcodec_send_frame(ctx.codec_ctx, ctx.sws_frame);
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
        // 转换时间基：编码器time_base -> 流time_base
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

    if (ctx.sws_frame) av_frame_free(&ctx.sws_frame);
    if (ctx.sws_ctx)   sws_freeContext(ctx.sws_ctx);
    if (ctx.codec_ctx) avcodec_free_context(&ctx.codec_ctx);
    if (ctx.fmt_ctx) {
        if (!(ctx.fmt_ctx->oformat->flags & AVFMT_NOFILE))
            avio_closep(&ctx.fmt_ctx->pb);
        avformat_free_context(ctx.fmt_ctx);
    }
    std::cout << "[Stream] Cleanup done." << std::endl;
}


int main(int argc, char* argv[]) {
    // 参数解析
    std::string image_path = (argc > 1) ? argv[1] : "image.png";
    std::string rtmp_url   = (argc > 2) ? argv[2] : "rtmp://127.0.0.1:1935/live/stream";
    int fps                = (argc > 3) ? std::stoi(argv[3]) : 25;
    int width              = (argc > 4) ? std::stoi(argv[4]) : 640;
    int height             = (argc > 5) ? std::stoi(argv[5]) : 480;
    int bitrate_kbps       = (argc > 6) ? std::stoi(argv[6]) : 800;
    int duration_sec       = (argc > 7) ? std::stoi(argv[7]) : 30;

    std::cout << "Image: " << image_path << std::endl;
    std::cout << "RTMP:  " << rtmp_url << std::endl;
    std::cout << "FPS:   " << fps << std::endl;
    std::cout << "Res:   " << width << "x" << height << std::endl;
    std::cout << "BR:    " << bitrate_kbps << "kbps" << std::endl;
    std::cout << "Time:  " << duration_sec << "s" << std::endl;

    // 信号处理
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // 初始化推流
    StreamContext ctx;
    if (!init_stream(ctx, width, height, fps, bitrate_kbps, rtmp_url)) {
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

    std::cout << "========== Demo Finished ==========" << std::endl;
    return 0;
}