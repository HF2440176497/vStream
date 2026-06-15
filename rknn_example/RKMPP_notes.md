# Rockchip 平台 RKMPP + RTMP 推流技术总结

> **适用芯片**：RK3566 / RK3568 / RK3588 等
> **场景**：OpenCV 读图 → BGR → NV12 → RKMPP 硬件编码 (H.264) → RTMP 推流
> **基于**：`/home/sasha/develop/TL3576/vision/ffmpeg-demo/demo/main2.cpp`

---

## 一、问题演进与根因

### 1.1 Broken pipe #1：CPU 帧直送硬件编码器

**症状**：`av_interleaved_write_frame` 第一次调用即返回 `Broken pipe`。

**根因**：`h264_rkmpp` 是硬件编码器（`AVCodecHWConfig` 含 `AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX`），要求输入帧位于硬件设备内存（DRM PRIME / DMA-BUF）。代码直接把 CPU 内存里的 NV12 送过去，编码器读到错误指针，产生空包/错包，服务器无法解析。

**修复**：按 CUDA/VAAPI 通用流程建立 `AVHWDeviceContext` + `AVHWFramesContext`，CPU 帧经 `av_hwframe_transfer_data` 上传后再送编码器。

```cpp
av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_RKMPP, ...);

hw_frames_ctx->format    = AV_PIX_FMT_DRM_PRIME;   // 硬件侧像素格式
hw_frames_ctx->sw_format = AV_PIX_FMT_NV12;        // CPU 侧像素格式
hw_frames_ctx->width     = config.width;
hw_frames_ctx->height    = config.height;
av_hwframe_ctx_init(hw_frames_ref);

codec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
codec_ctx->hw_frames_ctx = av_buffer_ref(hw_frames_ref);
codec_ctx->pix_fmt       = AV_PIX_FMT_DRM_PRIME;   // 硬件路径必须改成这个
```

发送时：
```cpp
av_hwframe_transfer_data(hw_frame, cpu_frame, 0);  // CPU NV12 → DRM_PRIME
avcodec_send_frame(codec_ctx, hw_frame);
```

### 1.2 Broken pipe #2：SPS/PPS 没进 FLV 头

**症状**：硬件路径正确，**仍在第一帧断开**。

**根因**：RTMP/FLV 协议要求 `onMetaData` 之后紧跟一个 `AVCDecoderConfigurationRecord`（封装 SPS/PPS），服务器据此建立 H.264 解码上下文。没有置 `AV_CODEC_FLAG_GLOBAL_HEADER` 时，`codec_ctx->extradata` 为空，FLV 头里该项为 0 字节，编码器把 SPS/PPS 当成普通 NALU 紧跟 I 帧发出，服务器拿到裸 NALU 无从解析 → 主动断开连接。

**修复**：`avcodec_open2` 之前置位，open 后立即校验：

```cpp
codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
avcodec_open2(codec_ctx, codec, &opts);

assert(codec_ctx->extradata_size > 0);  // 必备断言
```

### 1.3 调试黑洞：FFmpeg 日志被过滤

**症状**：明明设了 `av_log_set_level(AV_LOG_DEBUG)`，却看不到任何 `[FFmpeg]` 行。

**根因**：自定义回调里写了 `if (level <= AV_LOG_WARNING)`，把 INFO/VERBOSE/DEBUG 全部丢掉。

**修复**：按级别分流打印：

```cpp
if (level <= AV_LOG_ERROR)        std::cerr << "[FFmpeg E] ";
else if (level <= AV_LOG_WARNING) std::cerr << "[FFmpeg W] ";
else                              std::cout << "[FFmpeg]   ";
```

> **经验**：RKMPP 内部错误、pix_fmt 不匹配、MPP 返回值异常都只会写到 INFO/DEBUG，不放开永远看不到。

---

## 二、RKMPP 推流关键注意事项

### 2.1 系统依赖与权限

| 项目 | 说明 |
|------|------|
| 编码器名 | `h264_rkmpp` / `hevc_rkmpp` |
| 链接库 | `librockchip_mpp`、`librga`、`libdrm`、`libavcodec` |
| 内核节点 | `/dev/mpp_service`、`/dev/rga`、DRM 设备 (`/dev/dri/card0`) |
| 用户权限 | 运行用户需有上述设备的 rw 权限，否则 `av_hwdevice_ctx_create` 失败 |

### 2.2 硬件帧的 format 配对

| 字段 | 值 |
|------|------|
| `hw_frames_ctx->format`    | `AV_PIX_FMT_DRM_PRIME` |
| `hw_frames_ctx->sw_format` | `AV_PIX_FMT_NV12` |
| `codec_ctx->pix_fmt`       | `AV_PIX_FMT_DRM_PRIME`（硬件路径必须改） |
| `sws_getContext` 输出格式  | `AV_PIX_FMT_NV12`（即 sw_format） |

> 三者必须一致，否则 `avcodec_open2` 报 `Invalid argument`。

### 2.3 SPS/PPS 来源

1. **必须**在 open 前置 `AV_CODEC_FLAG_GLOBAL_HEADER`。
2. open 后立即检查 `codec_ctx->extradata_size > 0`。
3. 部分老版本 ffmpeg-rockchip 不写 extradata → 兜底逻辑：拿到首包后从 `pkt->side_data[AV_PKT_DATA_NEW_EXTRADATA]` 取出 SPS/PPS，复制到 `stream->codecpar->extradata` 并重发一帧，让 muxer 重新打包头。

### 2.4 时间戳

| 项 | 推荐 |
|------|------|
| `codec_ctx->time_base` | `{1, fps}`（如 `{1, 25}`）或 `{1, 90000}` |
| `stream->time_base` | 与 `codec_ctx` 一致，避免不必要 rescale |
| `frame->pts` 起点 | 0 或视频起始时间（毫秒） |
| 写出前 | **必做** `av_packet_rescale_ts(pkt, codec_tb, stream_tb)` |
| RTMP 限制 | 不能为 `AV_NOPTS_VALUE`，否则 server 端会丢包 |

> 帧率控制用 `av_usleep(expected_pts_us - now)` 把节流精确到目标 fps；首包 PTS≈0 时不要 sleep。

### 2.5 FLV / RTMP 封装选项

```cpp
av_dict_set(&mux_opts, "flvflags", "no_duration_filesize", 0);
```

- `no_duration_filesize`：关闭不断重写 `onMetaData` 里的 duration/filesize，避免 server 端做切片/HLS 时混乱。
- RTMP URL 形如 `rtmp://host/app/stream`，**多 client 推同一 stream key 会被抢占**。
- URL 中带特殊字符（`@`、空格）需要 URL-encode。

### 2.6 资源释放顺序

`cleanup()` 严格按以下顺序，否则 rkmpp 后端偶现 double free / 内核报错：

```
sws_ctx → pkt → hw_frame → frame → codec_ctx → hw_device_ctx → fmt_ctx
```

`avcodec_free_context` 必须在 `av_buffer_unref(&hw_device_ctx)` **之前**调用，否则 codec_ctx 还引用着 hw device。

---

## 三、启动验证清单

程序启动后必须看到以下三行，缺一不可：

```
[INFO] 检测到硬件编码器，需使用设备类型: rkmpp        ← 硬件路径建立成功
[INFO] SPS/PPS extradata: 30 bytes (前16字节): 01 64 00 ... ← SPS/PPS 已生成
[DEBUG] pkt: pts=0 dts=0 size=12345 key=Y                ← 首包是关键帧
```

- 缺失 #1 → 编码器名错或 `/dev/mpp_service` 权限不足。
- 缺失 #2 → 编码器版本过老，启用 2.3 节的 side_data 兜底。
- 缺失 #3 → 检查 `gop_size` 是否被设成 0，或添加 `forced-idr` 选项。

---

## 四、完整数据流

```
┌─────────────────────┐
│ OpenCV Mat (BGR)    │
└──────────┬──────────┘
           │ sws_scale (BGR → NV12)
           ▼
┌─────────────────────┐
│ CPU NV12 Frame      │  AV_PIX_FMT_NV12
└──────────┬──────────┘
           │ av_hwframe_transfer_data
           ▼
┌─────────────────────┐
│ DRM_PRIME Frame     │  AV_PIX_FMT_DRM_PRIME
└──────────┬──────────┘
           │ avcodec_send_frame
           ▼
┌─────────────────────┐
│ RKMPP Encoder       │  h264_rkmpp
└──────────┬──────────┘
           │ avcodec_receive_packet
           ▼
┌─────────────────────┐
│ H.264 AVPacket      │  extradata = SPS/PPS
└──────────┬──────────┘
           │ av_interleaved_write_frame
           ▼
       RTMP server
```

---

## 五、一句话总结

> **RKMPP 推流的两道坎**：
> 1. CPU 帧必须 `av_hwframe_transfer_data` 上传到 DRM_PRIME；
> 2. SPS/PPS 必须靠 `AV_CODEC_FLAG_GLOBAL_HEADER` 写进 extradata。
>
> 调试时务必把 `av_log` 回调按级别全放开，否则排错时一片漆黑。
