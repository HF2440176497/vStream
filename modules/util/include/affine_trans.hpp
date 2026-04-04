
#include <iostream>
#include <chrono>
#include <string>
#include <sstream>

#include <opencv2/opencv.hpp>

#include "cnstream_logging.hpp"

namespace cnstream {

struct AffineMat {
  float v0, v1, v2;
  float v3, v4, v5;

  AffineMat() : v0(1), v1(0), v2(0), v3(0), v4(1), v5(0) {}

  void from_cvmat(const cv::Mat& mat) {
    if (mat.rows != 2 || mat.cols != 3 || mat.type() != CV_32FC1) {
      LOGE(AFFINE) << "AffineMat: Input matrix must be 2x3 and CV_32FC1";
    }
    v0 = mat.ptr<float>(0)[0];
    v1 = mat.ptr<float>(0)[1];
    v2 = mat.ptr<float>(0)[2];
    v3 = mat.ptr<float>(1)[0];
    v4 = mat.ptr<float>(1)[1];
    v5 = mat.ptr<float>(1)[2];
  }

  void print(const std::string& title = "AffineMat") const {
    std::printf("%s:\n", title.c_str());
    std::printf("[%10.4f, %10.4f, %10.4f]\n", v0, v1, v2);
    std::printf("[%10.4f, %10.4f, %10.4f]\n", v3, v4, v5);
  }
};

/** 三通道排列顺序 */
enum class ChannelsArrange : int { RGB = 0, BGR = 1 };

enum class NormType : int { None = 0, MeanStd = 1, AlphaBeta = 2 };

struct Norm {
  float    mean[3];
  float    std[3];
  float    alpha, beta;
  NormType type = NormType::None;

  // out = (x * alpha - mean) / std
  static Norm mean_std(const float mean[3], const float std[3], float alpha = 1 / 255.0f) {
    Norm out;
    out.type = NormType::MeanStd;
    out.alpha = alpha;
    memcpy(out.mean, mean, sizeof(out.mean));
    memcpy(out.std, std, sizeof(out.std));
    return out;
  }

  // out = x * alpha + beta
  static Norm alpha_beta(float alpha, float beta = 0) {
    Norm out;
    out.type = NormType::AlphaBeta;
    out.alpha = alpha;
    out.beta = beta;
    return out;
  }
  static Norm None() {
    return Norm();
  }
};

class AffineTrans {
 public:
  AffineTrans() {}
  virtual ~AffineTrans() {}

 public:
  void compute(const std::tuple<int, int>& from, const std::tuple<int, int>& to) {
    auto src_w = std::get<0>(from);
    auto src_h = std::get<1>(from);

    auto dst_w = std::get<0>(to);
    auto dst_h = std::get<1>(to);

    float scale_x = (float)(dst_w) / (float)(src_w);
    float scale_y = (float)(dst_h) / (float)(src_h);
    float scale = std::min(scale_x, scale_y);

    cv::Mat src2dst = (cv::Mat_<float>(2, 3) << scale, 0.f, (-scale * src_w + dst_w + scale - 1) * 0.5,
        0.f, scale, (-scale * src_h + dst_h + scale - 1) * 0.5);
    cv::Mat dst2src = cv::Mat::zeros(2, 3, CV_32FC1);

    cv::invertAffineTransform(src2dst, dst2src);

    m_s2d.from_cvmat(src2dst);
    m_d2s.from_cvmat(dst2src);
  }

  AffineMat get_d2s() { return m_d2s; }
  AffineMat get_s2d() { return m_s2d; }

 public:
  AffineMat m_d2s;
  AffineMat m_s2d;
};

/**
 * 相当于 [x, y, 1] 乘以 [2*3] 矩阵，进行仿射变换
 */
inline void affine_project_cpu(const AffineMat& matrix, float x, float y, float* proj_x, float* proj_y) {
  *proj_x = matrix.v0 * x + matrix.v1 * y + matrix.v2;
  *proj_y = matrix.v3 * x + matrix.v4 * y + matrix.v5;
}


/**
 * @brief 利用仿射矩阵进行图像缩放：输入 HWC 格式，输出 CHW 格式
 * @param src 输入图像数据指针
 * @param src_w 输入图像宽度（像素）
 * @param src_h 输入图像高度（像素）
 * @param src_step 输入图像每行字节数（应传入 cv::Mat::step）
 * @param dst 输出图像数据指针, 是紧密排列的
 * @param dst_w 输出图像宽度
 * @param dst_h 输出图像高度
 * @param pad_value 填充值
 * @param matrix 仿射变换矩阵
 */
inline void resize_cpu(uint8_t* src, int src_w, int src_h, int src_step, float* dst, int dst_w, int dst_h, float pad_value,
                       const AffineMat& matrix) {
  int area = dst_w * dst_h;
  int channel_step = src_step / sizeof(uint8_t);  // 每行像素数（考虑填充）

  for (int dy = 0; dy < dst_h; ++dy) {
    for (int dx = 0; dx < dst_w; ++dx) {
      float src_x = 0;
      float src_y = 0;

      // 映射到原图像位置
      affine_project_cpu(matrix, dx, dy, &src_x, &src_y);

      float c0 = pad_value, c1 = pad_value, c2 = pad_value;

      if (!(src_x < -1 || src_x >= src_w || src_y < -1 || src_y >= src_h)) {
        int y_low = (int)std::floor(src_y);
        int x_low = (int)std::floor(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        uint8_t const_values[] = {(uint8_t)pad_value, (uint8_t)pad_value, (uint8_t)pad_value};

        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1.0f - ly;
        float hx = 1.0f - lx;
        float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

        uint8_t* v1 = const_values;
        uint8_t* v2 = const_values;
        uint8_t* v3 = const_values;
        uint8_t* v4 = const_values;

        // y 对应 高度 height，x 对应 宽度 width
        // 第一维度是 height，第二维度是 width
        // 使用 src_step 代替 src_w * 3 来处理可能的内存对齐填充
        if (y_low >= 0) {
          if (x_low >= 0) {
            v1 = src + y_low * src_step + x_low * 3;
          }
          if (x_high < src_w) {
            v2 = src + y_low * src_step + x_high * 3;
          }
        }
        if (y_high < src_h) {
          if (x_low >= 0) {
            v3 = src + y_high * src_step + x_low * 3;
          }
          if (x_high < src_w) {
            v4 = src + y_high * src_step + x_high * 3;
          }
        }

        c0 = std::floor(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
        c1 = std::floor(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
        c2 = std::floor(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
      }

      // note: 输出图像排列： CHW
      // 最后一个维度 dx 对应 width
      float* pdst_c0 = dst + dy * dst_w + dx;
      float* pdst_c1 = pdst_c0 + area;
      float* pdst_c2 = pdst_c1 + area;
      *pdst_c0 = c0;
      *pdst_c1 = c1;
      *pdst_c2 = c2;
    }
  }
}

/**
 * @brief 交换 CHW 格式的 R B 通道顺序
 * @param src 输入/输出数据指针（CHW 格式）
 * @param width 图像宽度
 * @param height 图像高度
 * @param channel_stride 每个通道的步长（字节数 / sizeof(float)）
 *        紧密排列时 channel_stride = width * height
 *        有内存填充时 channel_stride > width * height
 * @param order 表示输入数据的通道排列顺序
 */
inline void swap_channel_cpu(float* src, int width, int height, int channel_stride, ChannelsArrange order) {
  if (!src || width <= 0 || height <= 0 || channel_stride <= 0) {
    return;
  }

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int r_pos, b_pos;
      if (order == ChannelsArrange::RGB) {
        r_pos = y * width + x;                      // R 在第 0 通道
        b_pos = 2 * channel_stride + y * width + x; // B 在第 2 通道
      } else if (order == ChannelsArrange::BGR) {
        b_pos = y * width + x;                      // B 在第 0 通道
        r_pos = 2 * channel_stride + y * width + x; // R 在第 2 通道
      } else {
        return;
      }
      float temp = src[b_pos];
      src[b_pos] = src[r_pos];
      src[r_pos] = temp;
    }
  }
}


/**
 * @brief 对 CHW 格式的图片进行归一化，不改变通道顺序
 * @param src 输入/输出数据指针（CHW 格式）
 * @param width 图像宽度
 * @param height 图像高度
 * @param channel_stride 每个通道的步长（元素个数）
 * @param norm 归一化参数（支持 MeanStd 和 AlphaBeta 两种模式）
 * @param order 输入数据的通道排列顺序（RGB 或 BGR）
 */
inline void normalize_cpu(float* src, int width, int height, int channel_stride, const Norm& norm, ChannelsArrange order) {
  if (!src || width <= 0 || height <= 0 || channel_stride <= 0) {
    return;
  }

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float *src_r, *src_g, *src_b;
      if (order == ChannelsArrange::RGB) {
        src_r = src + y * width + x;
        src_g = src + channel_stride + y * width + x;
        src_b = src + 2 * channel_stride + y * width + x;
      } else if (order == ChannelsArrange::BGR) {
        src_b = src + y * width + x;
        src_g = src + channel_stride + y * width + x;
        src_r = src + 2 * channel_stride + y * width + x;
      } else {
        return;
      }

      float r = *src_r;
      float g = *src_g;
      float b = *src_b;

      if (norm.type == NormType::MeanStd) {
        r = (r * norm.alpha - norm.mean[0]) / norm.std[0];
        g = (g * norm.alpha - norm.mean[1]) / norm.std[1];
        b = (b * norm.alpha - norm.mean[2]) / norm.std[2];
      } else if (norm.type == NormType::AlphaBeta) {
        r = r * norm.alpha + norm.beta;
        g = g * norm.alpha + norm.beta;
        b = b * norm.alpha + norm.beta;
      }

      if (order == ChannelsArrange::RGB) {
        *src_r = r;
        *src_g = g;
        *src_b = b;
      } else if (order == ChannelsArrange::BGR) {
        *src_r = r;
        *src_g = g;
        *src_b = b;
      }
    }
  }
}


/**
 * @brief 保存 CHW 格式的图像为 opencv 格式 ( HWC BGR )
 * @param order 表示输入数据的通道排列顺序，RGB 时会处理为 BGR 
 * @param normalize 是否对图像进行归一化处理, 仅支持最大归一化范围 [0, 255] 范围
 * @note 仅用于输出调试，不代表模型输入，因为最小最大归一化导致了图像的范围变化
 */
inline void save_float_image_chw_cpu(float* src, int width, int height, const std::string& save_path, 
                                     ChannelsArrange order = ChannelsArrange::BGR, bool normalize = false) {
    if (!src || width <= 0 || height <= 0) {
        throw std::runtime_error("Invalid input parameters");
    }

    size_t data_size = width * height * 3 * sizeof(float);
    std::vector<float> h_src(width * height * 3);

    memcpy(h_src.data(), src, data_size);

    std::vector<float> h_src_hwc(width * height * 3);
    int target_c;
    for (int c = 0; c < 3; ++c) {
        if (order == ChannelsArrange::RGB) {
            target_c = 2 - c;
        } else if (order == ChannelsArrange::BGR) {
            target_c = c;
        } else { target_c = c; }
        for (int y = 0; y < height; ++y) {  // 行
            for (int x = 0; x < width; ++x) {  // 列
                // CHW: c * (H*W) + y * W + x
                // HWC: (y * W + x) * 3 + c
                h_src_hwc[(y * width + x) * 3 + target_c] = h_src[c * (height * width) + y * width + x];
            }
        }
    }
    cv::Mat img_float(height, width, CV_32FC3, h_src_hwc.data());
    cv::Mat img_uint8;

    if (normalize) {
        cv::normalize(img_float, img_float, 0, 255, cv::NORM_MINMAX);
        img_float.convertTo(img_uint8, CV_8UC3);
    } else {
        img_float.convertTo(img_uint8, CV_8UC3);
    }
    if (!cv::imwrite(save_path, img_uint8)) {
        throw std::runtime_error("Failed to save image: " + save_path);
    }
}

}
