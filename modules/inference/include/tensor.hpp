

#ifndef MODULES_INFERENCE_TENSOR_HPP_
#define MODULES_INFERENCE_TENSOR_HPP_

#include <string>
#include <vector>
#include <cstdint>
#include <sstream>
#include <ostream>
#include <memory>

#include "cnstream_logging.hpp"

#ifdef __CUDA_ARCH__
#include <cuda_fp16.h>
#endif

namespace cnstream {

/**
 * @brief Enumeration to specify data type of model input and output
 */
enum class DataType { 
  INVALID = -1,
  UINT8,
  INT8, 
  FLOAT16, 
  FLOAT32, 
  INT16, 
  INT32 
};

/**
 * @brief 仿照 nvinfer1::TensorFormat 定义
 */
enum class TensorFormat {
  INVALID = -1,
  LINEAR,
  CHW2,
  CHW4,
  CHW32,
  HWC8
};

inline const char* data_type_string(DataType dt) {
  switch(dt){
    case DataType::UINT8: return "UInt8";
    case DataType::INT8: return "Int8";
    case DataType::FLOAT16: return "Float16";
    case DataType::FLOAT32: return "Float32";
    case DataType::INT16: return "Int16";
    case DataType::INT32: return "Int32";
    default: return "Unknow";
  }
}

inline int data_type_size(DataType dt) {
  switch (dt) {
    case DataType::UINT8: return sizeof(uint8_t);
    case DataType::INT8: return sizeof(int8_t);
#ifdef __CUDA_ARCH__
    case DataType::FLOAT16: return sizeof(__half);
#endif
    case DataType::FLOAT32: return sizeof(float);
    case DataType::INT16: return sizeof(int16_t);
    case DataType::INT32: return sizeof(int32_t);
    default: {
      LOGI(TENSOR) << "Not support dtype: " << data_type_string(dt);
      return -1;
    }
  }
}

/**
 * @brief 封装张量的形状信息, 只支持 NCHW 格式
 */
class TensorShape {
 public:
  TensorShape() = default;
  TensorShape(const TensorShape& other) = default;
  TensorShape& operator=(const TensorShape& other) = default;

  TensorShape(int n, int c, int h, int w);
  TensorShape(int ndims, const int* dims);
  TensorShape(const std::vector<int>& dims);

  ~TensorShape() = default;

  int numel() const;  // 元素数量
  int DataCount() const { return numel(); }
  int ndims() const;
  int size(int index) const;
  int shape(int index) const;

  inline int N() const { return shape_[0]; }
  inline int C() const { return shape_[1]; }
  inline int H() const { return shape_[2]; }
  inline int W() const { return shape_[3]; }

  std::string ToString() const {
    if (shape_.empty()) {
      return "TensorShape()";
    }
    std::stringstream ss;
    ss << "TensorShape[";
    for (int i = 0; i < shape_.size(); ++i) {
      ss << shape_[i];
      if (i < shape_.size() - 1) {
        ss << " x ";
      }
    }
    ss << "]";
    return ss.str();
  }

  friend std::ostream& operator<<(std::ostream& os, const TensorShape& shape) {
    return os << shape.ToString();
  }

 private:
  std::vector<int> shape_;
};


/**
 * @brief 封装张量的信息 但是不负责管理生命周期
 * @note Tensor 应该是与平台 dev 无关的
 */
class Tensor {
 public:
  Tensor(const Tensor& other) = delete;
  Tensor& operator=(const Tensor& other) = delete;

  explicit Tensor(DataType dtype);
  explicit Tensor(DataType dtype, 
                  TensorFormat format,
                  std::shared_ptr<void*> data);
  explicit Tensor(int n, int c, int h, int w,
                  DataType dtype,
                  TensorFormat format,
                  std::shared_ptr<void*> data);
  explicit Tensor(int ndims, const int* dims, 
                  DataType dtype,
                  TensorFormat format,
                  std::shared_ptr<void*> data);
  explicit Tensor(const std::vector<int>& dims, 
                  DataType dtype,
                  TensorFormat format,
                  std::shared_ptr<void*> data);

  virtual ~Tensor();

  void release();

  int        numel() const;
  inline int ndims() const { return shape_.size(); }
  inline int size(int index) const { return shape_[index]; }
  inline int shape(int index) const { return shape_[index]; }

  int batch() const;
  int channel() const;
  int height() const;
  int width() const;

  inline DataType                   type() const { return dtype_; }
  inline const std::vector<int>&    dims() const { return shape_; }
  inline const std::vector<size_t>& strides() const { return strides_; }
  inline int                        bytes() const { return bytes_; }
  inline int                        bytes(int start_axis) const { return count(start_axis) * element_size(); }
  inline int                        element_size() const { return data_type_size(dtype_); }

  template <typename... _Args>
  int offset(int index, _Args... index_args) const {
    const int index_array[] = {index, index_args...};
    return offset_array(sizeof...(index_args) + 1, index_array);
  }

  int offset_array(const std::vector<int>& index) const;
  int offset_array(size_t size, const int* index_array) const;

  template <typename... _Args>
  Tensor& resize(int dim_size, _Args... dim_size_args) {
    const int dim_size_array[] = {dim_size, dim_size_args...};
    return resize(sizeof...(dim_size_args) + 1, dim_size_array);
  }

  Tensor& resize(std::initializer_list<int> dims) {
    return resize(dims.size(), dims.begin());  // const int*
  }

  Tensor& resize(int ndims, const int* dims);
  Tensor& resize(const std::vector<int>& dims);
  Tensor& resize_single_dim(int idim, int size);

  int count(int start_axis = 0) const;

  template <typename DType>
  inline const DType* data() const {
    return (DType*)(data_.get());
  }
  template <typename DType>
  inline DType* data() {
    return (DType*)(data_.get());
  }

  template <typename DType, typename... _Args>
  inline DType* data(int i, _Args&&... args) {
    return data<DType>() + offset(i, args...);
  }

  template <typename DType, typename... _Args>
  inline DType& at(int i, _Args&&... args) {
    return *(data<DType>() + offset(i, args...));
  }

  const char* shape_string() const { return shape_string_; }
  const char* descriptor() const;

 private:
  Tensor& compute_shape_string();
  Tensor& adjust_memory_by_update_dims_or_type();
  void    setup_data(std::shared_ptr<void*> data);

 private:
  TensorFormat               format_{TensorFormat::LINEAR};
  DataType                   dtype_{DataType::FLOAT32};
  std::vector<int>           shape_;
  std::vector<size_t>        strides_;
  size_t                     bytes_ = 0;

  char                       shape_string_[100];
  char                       descriptor_string_[100];
  std::shared_ptr<void*>     data_ = nullptr;
};

}  // namespace cnstream

#endif  // MODULES_INFERENCE_TENSOR_HPP_
