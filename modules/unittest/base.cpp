/*************************************************************************
 * Copyright (C) [2019] by Cambricon, Inc. All rights reserved
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *************************************************************************/

#include "base.hpp"
#include "cnstream_logging.hpp"
#include "data_source_param.hpp"
#include "cnstream_frame_va.hpp"

extern int errno;

std::string random_field_str = "XXXXXX";

std::string GetExePath() {
  char path[PATH_MAX_LENGTH];
  // On Unix-like systems, /proc/self/exe is a symbolic link to the executable file
  // readlink() reads the symbolic link and get path
  int cnt = readlink("/proc/self/exe", path, PATH_MAX_LENGTH);
  if (cnt < 0 || cnt >= PATH_MAX_LENGTH) {
    return "";
  }
  for (int i = cnt - 1; i >= 0; --i) {
    if ('/' == path[i]) {
      path[i + 1] = '\0';
      break;
    }
  }
  std::string result(path);
  return result;
}

void CheckExePath(const std::string& path) {
  if (path.size() == 0) {
    LOGF_IF(COREUNITEST, 0 != errno) << std::string(strerror(errno));
    LOGF(COREUNITEST) << "length of exe path is larger than " << PATH_MAX_LENGTH;
  }
}

// Unix-like systems function
// 根据路径创建临时文件，在外需要手动维护关闭
std::pair<int, std::string> CreateTempFile(const std::string& filename_prefix) {
  char filename[PATH_MAX_LENGTH];
  if (filename_prefix.size() > PATH_MAX_LENGTH - random_field_str.size()) {
    LOGF(COREUNITEST) << "filename_prefix is too long, must be less than " << PATH_MAX_LENGTH - random_field_str.size() << std::endl;
  }
  strncpy(filename, filename_prefix.c_str(), filename_prefix.size());
  strncpy(filename + filename_prefix.size(), random_field_str.c_str(), random_field_str.size());
  int fd = mkstemp(filename);
  LOGF_IF(COREUNITEST, -1 == fd) << "Create temporary file for BuildPipelineByJSONFile test case failed! "
      << strerror(errno);
  return std::make_pair(fd, std::string(filename));
}


std::string readFile(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        LOGE(COREUNITEST) << "Open file " << filename << " failed";
        return "";
    }
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    return content;
}

namespace cnstream {

/**
 * @brief 创建一个测试用的 DecodeFrame
 * @param fmt 图像格式
 * @param width 图像宽度
 * @param height 图像高度
 * @return 返回一个指向测试用 DecodeFrame 的指针
 */
DecodeFrame* CreateTestDecodeFrame(DataFormat fmt, int width, int height) {
  DecodeFrame* frame = new DecodeFrame(height, width, fmt);
  frame->fmt = fmt;
  frame->width = width;
  frame->height = height;
  frame->device_id = -1;
  frame->planeNum = 0;

  for (int i = 0; i < CN_MAX_PLANES; ++i) {
    frame->plane[i] = nullptr;
    frame->stride[i] = 0;
  }

  const uint8_t R = 10, G = 128, B = 242;
  const uint8_t Y = 111;
  const uint8_t U = 190;
  const uint8_t V = 72;

  size_t frame_size = 0;
  if (fmt == DataFormat::PIXEL_FORMAT_BGR24 || 
      fmt == DataFormat::PIXEL_FORMAT_RGB24) {
    frame->planeNum = 1;
    frame_size = width * height * 3;
    frame->plane[0] = malloc(frame_size);
    uint8_t* data = static_cast<uint8_t*>(frame->plane[0]);
    for (int i = 0; i < width * height; ++i) {
      if (fmt == DataFormat::PIXEL_FORMAT_BGR24) {
        data[i * 3] = B;
        data[i * 3 + 1] = G;
        data[i * 3 + 2] = R;
      } else {
        data[i * 3] = R;
        data[i * 3 + 1] = G;
        data[i * 3 + 2] = B;
      }
    }
  } else if (fmt == DataFormat::PIXEL_FORMAT_YUV420_NV12 ||
             fmt == DataFormat::PIXEL_FORMAT_YUV420_NV21) {
    frame->planeNum = 2;
    frame_size = width * height * 3 / 2;
    frame->plane[0] = malloc(width * height);
    frame->plane[1] = malloc(width * height / 2);
    memset(frame->plane[0], Y, width * height);
    uint8_t* uv_data = static_cast<uint8_t*>(frame->plane[1]);
    for (size_t i = 0; i < static_cast<size_t>(width * height / 2); i += 2) {
      if (fmt == DataFormat::PIXEL_FORMAT_YUV420_NV12) {
        uv_data[i] = U;
        uv_data[i + 1] = V;
      } else {
        uv_data[i] = V;
        uv_data[i + 1] = U;
      }
    }
  }
  return frame;
}

// 辅助函数：清理测试用的DecodeFrame
void CleanupTestDecodeFrame(DecodeFrame* frame) {
  if (frame) {
    if (frame->plane[0]) free(frame->plane[0]);
    if (frame->plane[1]) free(frame->plane[1]);
    delete frame;
  }
}

}  // namespace cnstream
