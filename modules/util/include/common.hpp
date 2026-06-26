

#include <algorithm>
#include <cctype>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

namespace cnstream {

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i + 1 < vec.size()) os << ", ";
    }
    os << "]";
    return os;
}

namespace utils {

/**
 * @brief 获取文件名（不包含扩展名）
 */
inline std::string get_filename_without_ext(const std::string& file) {
  auto last_slash = file.find_last_of("/\\");
  auto filename = (last_slash == std::string::npos) 
                   ? file 
                   : file.substr(last_slash + 1);
  auto first_dot = filename.find_first_of(".");
  return (first_dot == std::string::npos) 
         ? filename 
         : filename.substr(0, first_dot);
}

inline std::vector<uint8_t> load_model(const std::string& file) {
  std::ifstream in(file, std::ios::in | std::ios::binary);
  if (!in.is_open()) {
    return {};
  }
  in.seekg(0, std::ios::end);
  size_t length = in.tellg();

  std::vector<uint8_t> data;
  if (length > 0) {
    in.seekg(0, std::ios::beg);
    data.resize(length);
    in.read((char*)&data[0], length);
  }
  in.close();
  return data;
}

/**
 * @brief 文件夹图像循环读取器
 *
 * 从指定文件夹中收集 png/jpg/jpeg/bmp/webp 图像，通过 read() 循环返回 cv::Mat。
 * 内部使用独立线程预取接下来若干张图像，避免一次性加载整个文件夹导致内存暴涨。
 */
class ImageFolderReader {
 public:
  /**
   * @brief 构造读取器并启动预取线程
   * @param folder 图像文件夹路径
   * @param prefetch_count 预取图像数量，默认 4；传 0 会按 1 处理
   */
  explicit ImageFolderReader(const std::string& folder, size_t prefetch_count = 4)
      : folder_(folder),
        prefetch_count_(prefetch_count <= 0 ? 1 : prefetch_count),
        running_(true),
        read_index_(0) {
    collect_image_files();
    max_buffer_size_ = std::min(prefetch_count_, image_files_.size());
    if (!image_files_.empty()) {
      producer_ = std::thread(&ImageFolderReader::produce, this);
    }
  }

  ~ImageFolderReader() { stop(); }

  ImageFolderReader(const ImageFolderReader&) = delete;
  ImageFolderReader& operator=(const ImageFolderReader&) = delete;

  /**
   * @brief 读取下一张图像
   * @return cv::Mat 成功返回图像，失败或已停止返回空 Mat
   */
  cv::Mat read() {
    if (image_files_.empty()) {
      return cv::Mat();
    }

    std::unique_lock<std::mutex> lock(queue_mutex_);
    cv_cond_.wait(lock, [this] { return !buffer_.empty() || !running_; });
    if (buffer_.empty()) {
      return cv::Mat();
    }

    cv::Mat mat = std::move(buffer_.front());
    buffer_.pop();
    read_index_ = (read_index_ + 1) % image_files_.size();
    lock.unlock();
    cv_cond_.notify_one();
    return mat;
  }

  bool empty() const { return image_files_.empty(); }

  size_t size() const { return image_files_.size(); }

  void stop() {
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      running_ = false;
    }
    cv_cond_.notify_all();
    if (producer_.joinable()) {
      producer_.join();
    }
  }

 private:
  static bool is_image_file(const std::filesystem::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext == ".png" || ext == ".jpg" || ext == ".jpeg" ||
           ext == ".bmp" || ext == ".webp";
  }

  void collect_image_files() {
    try {
      if (!std::filesystem::is_directory(folder_)) {
        return;
      }
      for (const auto& entry : std::filesystem::directory_iterator(folder_)) {
        if (entry.is_regular_file() && is_image_file(entry.path())) {
          image_files_.push_back(entry.path().string());
        }
      }
      std::sort(image_files_.begin(), image_files_.end());
    } catch (const std::filesystem::filesystem_error&) {
      image_files_.clear();
    }
  }

  void produce() {
    while (true) {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      cv_cond_.wait(lock, [this] { return buffer_.size() < max_buffer_size_ || !running_; });
      if (!running_) {
        break;
      }

      size_t load_index = (read_index_ + buffer_.size()) % image_files_.size();
      lock.unlock();

      const std::string& file = image_files_[load_index];
      cv::Mat mat = cv::imread(file, cv::IMREAD_COLOR);

      lock.lock();
      if (running_ && buffer_.size() < max_buffer_size_) {
        buffer_.push(std::move(mat));
        lock.unlock();
        cv_cond_.notify_one();
      }
    }
  }

  std::string folder_;
  size_t prefetch_count_;
  size_t max_buffer_size_;
  std::vector<std::string> image_files_;
  std::queue<cv::Mat> buffer_;
  std::mutex queue_mutex_;
  std::condition_variable cv_cond_;
  bool running_;
  size_t read_index_;
  std::thread producer_;
};

}  // namespace utils

}  // namespace cnstream