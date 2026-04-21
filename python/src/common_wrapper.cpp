

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <opencv2/opencv.hpp>

#include "common_wrapper.hpp"

namespace py = pybind11;

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

std::vector<std::size_t> GetMatShape(cv::Mat& m) {  // NOLINT
  if (m.channels() == 1) {
    return {static_cast<size_t>(m.rows), static_cast<size_t>(m.cols)};
  }
  return {static_cast<size_t>(m.rows), static_cast<size_t>(m.cols), static_cast<size_t>(m.channels())};
}


py::capsule MakeCapsule(cv::Mat& m) {
  return py::capsule(new cv::Mat(m), [](void *v) { delete reinterpret_cast<cv::Mat*>(v); });
}

/**
 * @brief Convert cv::Mat to python numpy array
 */
py::array MatToArray(cv::Mat& m) {
    if (!m.isContinuous()) {
        throw std::invalid_argument("Only continuous cv::Mat is supported. Call mat.clone() or mat.reshape(1) first.");
    }
    std::vector<py::ssize_t> strides;
    if (m.channels() == 1) {
        strides = {m.step[0], m.step[1]};
    } else {
        // elemSize1() is the size of each channel in bytes
        // for opencv, channel 是 HWC 的最后一个维度，可以使用 elemSize1 直接获取，不存在 step[2]
        strides = {m.step[0], m.step[1], static_cast<py::ssize_t>(m.elemSize1())};
    }
    return py::array(GetNpDType(m.depth()), GetMatShape(m), strides, m.data, MakeCapsule(m));
}

/**
 * @brief Convert python numpy array to cv::Mat
 */
cv::Mat ArrayToMat(py::array_t<uint8_t>& array) {
    py::buffer_info buf = array.request();
    if (buf.ndim != 2 && buf.ndim != 3) {
        throw std::runtime_error("Number of dimensions must be 2 or 3");
    }
    if (buf.format != py::format::format_descriptor<uint8_t>::format()) {
      throw std::runtime_error("Only uint8 format is supported");
    }
    int rows = static_cast<int>(buf.shape[0]);  // height
    int cols = static_cast<int>(buf.shape[1]);  // width
    int channels = (buf.ndim == 3) ? buf.shape[2] : 1;
    size_t step = static_cast<size_t>(buf.strides[0]);
    cv::Mat mat(rows, cols, 
                (channels == 1) ? CV_8UC1 : CV_8UC3, 
                static_cast<uint8_t*>(buf.ptr), step);
    return mat.clone();
}