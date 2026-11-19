

#include "common_wrapper.hpp"

namespace py = pybind11;

namespace cnstream {

py::dtype GetNpDType(int depth) {
  switch (depth) {
    case CV_8U:  return py::dtype::of<uint8_t>();
    case CV_8S:  return py::dtype::of<int8_t>();
    case CV_16U: return py::dtype::of<uint16_t>();
    case CV_16S: return py::dtype::of<int16_t>();
    case CV_32S: return py::dtype::of<int32_t>();
    case CV_32F: return py::dtype::of<float>();
    case CV_64F: return py::dtype::of<double>();
    default:     throw std::invalid_argument("Not supported cv::Mat depth type");
  }
}

std::vector<std::size_t> GetMatShape(const cv::Mat& m) {
  if (m.channels() == 1) {
    return {static_cast<size_t>(m.rows), static_cast<size_t>(m.cols)};
  }
  return {static_cast<size_t>(m.rows), static_cast<size_t>(m.cols), static_cast<size_t>(m.channels())};
}

/**
 * @brief copy construct to hold cv::Mat
 */
py::capsule MakeCapsule(const cv::Mat& m) {
  return py::capsule(new cv::Mat(m), [](void *v) { delete reinterpret_cast<cv::Mat*>(v); });
}

/**
 * @brief Convert cv::Mat to python numpy array
 */
py::array MatToArray(const cv::Mat& m) {
  if (m.empty()) {
    return py::array();
  }
  if (!m.isContinuous()) {
    throw std::invalid_argument("only continuous cv::Mat is supported. Call mat.clone() or mat.reshape(1) first.");
  }
  std::vector<py::ssize_t> strides;
  if (m.channels() == 1) {
    strides = {static_cast<py::ssize_t>(m.step[0]), static_cast<py::ssize_t>(m.step[1])};
  } else {
    // elemSize1() is the size of each channel in bytes
    // for opencv, channel 是 HWC 的最后一个维度，可以使用 elemSize1 直接获取，不存在 step[2]
    strides = {static_cast<py::ssize_t>(m.step[0]), static_cast<py::ssize_t>(m.step[1]), static_cast<py::ssize_t>(m.elemSize1())};
  }
  return py::array(GetNpDType(m.depth()), GetMatShape(m), strides, m.data, MakeCapsule(m));
}

/**
 * @brief Convert python numpy array to cv::Mat
 */
cv::Mat ArrayToMat(py::array_t<uint8_t>& array) {
  py::buffer_info buf = array.request();
  if (buf.ndim != 2 && buf.ndim != 3) {
    throw std::runtime_error("number of dimensions must be 2 or 3");
  }
  if (buf.format != py::format_descriptor<uint8_t>::format()) {
    throw std::runtime_error("only uint8 format is supported");
  }
  int channels = (buf.ndim == 3) ? static_cast<int>(buf.shape[2]) : 1;
  if (channels != 1 && channels != 3) {
    throw std::runtime_error("only 1 or 3 channels are supported");
  }
  // 规范化为 C 连续内存：切片/翻转/非连续数组的步长在此被统一（按需拷贝），
  // 否则直接按 strides[0] 构造 cv::Mat 会行内错位，负步长还会越界
  py::array contig = py::array::ensure(array, py::array::c_style);
  if (!contig) {
    throw std::runtime_error("failed to make array C-contiguous");
  }
  py::buffer_info cbuf = contig.request();
  int rows = static_cast<int>(cbuf.shape[0]);  // height
  int cols = static_cast<int>(cbuf.shape[1]);  // width
  cv::Mat mat(rows, cols,
              (channels == 1) ? CV_8UC1 : CV_8UC3,
              static_cast<uint8_t*>(cbuf.ptr));
  return mat.clone();
}

}  // namespace cnstream