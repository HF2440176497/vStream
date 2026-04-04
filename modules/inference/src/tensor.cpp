

#include "tensor.hpp"
#include <cassert>

namespace cnstream {

TensorShape::TensorShape(int n, int c, int h, int w) : shape_{n, c, h, w} {}

TensorShape::TensorShape(int ndims, const int* dims) : shape_(dims, dims + ndims) {}

TensorShape::TensorShape(const std::vector<int>& dims) : shape_(dims) {}

int TensorShape::numel() const {
  int value = shape_.empty() ? 0 : 1;
  for (int i = 0; i < shape_.size(); ++i) {
    value *= shape_[i];
  }
  return value;
}

int TensorShape::ndims() const { return shape_.size(); }

int TensorShape::size(int index) const { return shape_[index]; }

int TensorShape::shape(int index) const { return shape_[index]; }

Tensor::Tensor(DataType dtype) {
  shape_string_[0] = 0;
  descriptor_string_[0] = 0;
  dtype_ = dtype;
}

Tensor::Tensor(DataType dtype, TensorFormat format, std::shared_ptr<void*> data) {
  shape_string_[0] = 0;
  descriptor_string_[0] = 0;
  dtype_ = dtype;
  format_ = format;
  setup_data(data);
}

Tensor::Tensor(int n, int c, int h, int w, DataType dtype, TensorFormat format, std::shared_ptr<void*> data) {
  this->dtype_ = dtype;
  this->format_ = format;
  descriptor_string_[0] = 0;
  setup_data(data);
  resize(n, c, h, w);  // resize(int ndims, const int* dims); dims = [n, c, h, w]
}

Tensor::Tensor(int ndims, const int* dims, DataType dtype, TensorFormat format, std::shared_ptr<void*> data) {
  this->dtype_ = dtype;
  this->format_ = format;
  descriptor_string_[0] = 0;
  setup_data(data);
  resize(ndims, dims);
}

Tensor::Tensor(const std::vector<int>& dims, DataType dtype, TensorFormat format, std::shared_ptr<void*> data) {
  this->dtype_ = dtype;
  this->format_ = format;
  descriptor_string_[0] = 0;
  setup_data(data);
  resize(dims);
}

Tensor::~Tensor() { release(); }

void Tensor::release() {
  shape_.clear();
  strides_.clear();
  bytes_ = 0;
  shape_string_[0] = 0;
  descriptor_string_[0] = 0;
}

int Tensor::batch() const { return shape_[0]; }

// 注意：除了 kLINEAR，其他类型不代表真实宽度、高度
int Tensor::channel() const {
  switch (format_) {
    case TensorFormat::LINEAR:
    case TensorFormat::CHW2:
    case TensorFormat::CHW4:
    case TensorFormat::CHW32:
      return shape_[1];
    case TensorFormat::HWC8:
      return shape_[3];
    default:
      LOGE(TENSOR) << "Unsupported tensor format";
  }
  return -1;
}

int Tensor::height() const {
  switch (format_) {
    case TensorFormat::LINEAR:
    case TensorFormat::CHW2:
    case TensorFormat::CHW4:
    case TensorFormat::CHW32:
      return shape_[2];
    case TensorFormat::HWC8:
      return shape_[1];
    default:
      LOGE(TENSOR) << "Unsupported tensor format";
  }
  return -1;
}

int Tensor::width() const {
  switch (format_) {
    case TensorFormat::LINEAR:
    case TensorFormat::CHW2:
    case TensorFormat::CHW4:
    case TensorFormat::CHW32:
      return shape_[3];
    case TensorFormat::HWC8:
      return shape_[2];
    default:
      LOGE(TENSOR) << "Unsupported tensor format";
  }
  return -1;
}

const char* Tensor::descriptor() const {
  char* descriptor_ptr = (char*)descriptor_string_;
  snprintf(descriptor_ptr, sizeof(descriptor_string_), "Tensor:%p, %s, %s, CUDA:%d", data_.get(),
           data_type_string(dtype_), shape_string_);
  return descriptor_ptr;
}

Tensor& Tensor::compute_shape_string() {
  shape_string_[0] = 0;

  char*  buffer = shape_string_;
  size_t buffer_size = sizeof(shape_string_);
  for (int i = 0; i < shape_.size(); ++i) {
    int size = 0;
    if (i < shape_.size() - 1)
      size = snprintf(buffer, buffer_size, "%d x ", shape_[i]);
    else
      size = snprintf(buffer, buffer_size, "%d", shape_[i]);

    buffer += size;
    buffer_size -= size;
  }
  return *this;
}

// no reason to trust
void Tensor::setup_data(std::shared_ptr<void*> data) { 
  data_ = data; 
}

/**
 * @brief 需要跳过多少个元素才能到达目标位置
 * @param index_array 索引数组
 * @return int 偏移量，元素数 而不是字节数
 */
int Tensor::offset_array(size_t size, const int* index_array) const {
  assert(size <= shape_.size());
  int value = 0;
  for (int i = 0; i < shape_.size(); ++i) {
    if (i < size) value += index_array[i];

    if (i + 1 < shape_.size()) value *= shape_[i + 1];
  }
  return value;
}

int Tensor::offset_array(const std::vector<int>& index_array) const {
  return offset_array(index_array.size(), index_array.data());
}

/**
 * 计算元素数量
 */
int Tensor::count(int start_axis) const {
  if (start_axis >= 0 && start_axis < shape_.size()) {
    int size = 1;
    for (int i = start_axis; i < shape_.size(); ++i) size *= shape_[i];
    return size;
  } else {
    return 0;
  }
}

Tensor& Tensor::resize(const std::vector<int>& dims) { 
  return resize(dims.size(), dims.data()); 
}

int Tensor::numel() const {
  int value = shape_.empty() ? 0 : 1;
  for (int i = 0; i < shape_.size(); ++i) {
    value *= shape_[i];
  }
  return value;
}

Tensor& Tensor::resize_single_dim(int idim, int size) {
  assert(idim >= 0 && idim < shape_.size());
  auto new_shape = shape_;
  new_shape[idim] = size;
  return resize(new_shape);
}

/**
 * @note strides_ 单位是 byte
 * @details dims -1 表示采用对应的原维度，此时要求 dims.size == shape_.size
 */
Tensor& Tensor::resize(int ndims, const int* dims) {
  std::vector<int> setup_dims(ndims);
  for (int i = 0; i < ndims; ++i) {
    int dim = dims[i];
    if (dim == -1) {
      assert(ndims == shape_.size());
      dim = shape_[i];
    }
    setup_dims[i] = dim;
  }
  this->shape_ = setup_dims;
  this->strides_.resize(setup_dims.size());

  size_t prev_size = element_size();
  size_t prev_shape = 1;
  for (int i = (int)strides_.size() - 1; i >= 0; --i) {
    if (i + 1 < strides_.size()) {
      prev_size = strides_[i + 1];
      prev_shape = shape_[i + 1];
    }
    strides_[i] = prev_size * prev_shape;
  }

  this->adjust_memory_by_update_dims_or_type();
  this->compute_shape_string();
  return *this;
}

/**
 * @brief 根据 dims 去更新自己的 bytes_ 为 needed_size
 */
Tensor& Tensor::adjust_memory_by_update_dims_or_type() {
  int needed_size = this->numel() * element_size();
  this->bytes_ = needed_size;
  return *this;
}

};  // namespace cnstream
