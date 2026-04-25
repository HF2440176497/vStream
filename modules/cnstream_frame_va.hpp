

#ifndef CNSTREAM_FRAME_VA_HPP_
#define CNSTREAM_FRAME_VA_HPP_


#include <memory>
#include <mutex>
#include <string>
#include <map>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>
#include <libyuv/convert.h>

#include "cnstream_common.hpp"
#include "cnstream_frame.hpp"

// ---- util
#include "memop.hpp"
#include "cnstream_syncmem.hpp"

namespace cnstream {


/**
 * @class DataFrame
 * @brief DataFrame is a class holding a data frame and the frame description.
 * @todo: 未来支持统一内存管理
 * 在外使用 shared_ptr 管理
 */
class DataFrame : public NonCopyable {
 public:
  DataFrame() {
    for (int i = 0; i < FRAME_MAX_PLANES; ++i) {
      data_[i] = nullptr;
    }
    for (int i = 0; i < FRAME_MAX_PLANES; ++i) {
      stride_[i] = 0;
    }
  };
  ~DataFrame() = default;

  int GetPlanes() const { return FormatPlanes(fmt_); }

  size_t GetPlaneBytes(int plane_idx) const;

  size_t GetBytes() const;

  void CopyToSyncMem(DecodeFrame* decode_frame);

  std::shared_ptr<MemOp> CreateMemOp();

  cv::Mat GetImage();

  bool HasImage() {
    std::lock_guard<std::mutex> lk(mtx_);
    if (mat_.empty()) return false;
    return true;
  }

 public:
  uint64_t GetFrameId() const { return frame_id_; }
  DataFormat GetFmt() const { return fmt_; }
  int GetWidth() const { return width_; }
  int GetHeight() const { return height_; }
  int GetStride(int plane_idx) const { return stride_[plane_idx]; }
  const DevContext& GetCtx() const { return ctx_; }

  std::unique_ptr<CNSyncedMemory> data_[FRAME_MAX_PLANES];
  std::unique_ptr<IDataDeallocator> deAllocator_ = nullptr;
  
#ifdef VSTREAM_UNIT_TEST
 public:
#else
 private:
#endif
  mutable std::mutex mtx_;  // protect mat_
  cv::Mat mat_;

  DevContext ctx_;
  uint64_t frame_id_ = -1;
  DataFormat fmt_ = DataFormat::INVALID;
  int width_ = 0;
  int height_ = 0;
  int stride_[FRAME_MAX_PLANES];
  
  friend class SourceRender;

#ifdef VSTREAM_UNIT_TEST
  std::map<std::string, uint64_t> frame_count_map_;  // just for test
#endif

};  // class DataFrame


/**
 * @struct InferBoundingBox
 *
 * @brief InferBoundingBox is a structure holding the bounding box information of a detected object in normalized
 * coordinates.
 */
struct InferBoundingBox {
  float x;  ///< The x-axis coordinate in the upper left corner of the bounding box.
  float y;  ///< The y-axis coordinate in the upper left corner of the bounding box.
  float w;  ///< The width of the bounding box.
  float h;  ///< The height of the bounding box.
};

/**
 * @struct InferAttr
 *
 * @brief InferAttr is a structure holding the classification properties of an object.
 */
typedef struct {
  int id = -1;      ///< The unique ID of the classification. The value -1 means invalid.
  int value = -1;   ///< The label value of the classification.
  float score = 0;  ///< The label score of the classification.
} InferAttr;

/**
 *  Defines an alias for std::vector<float>. InferFeature contains one kind of inference feature.
 */
using InferFeature = std::vector<float>;

/**
 * Defines an alias for std::vector<std::pair<std::string, std::vector<float>>>. InferFeatures contains all kinds of
 * features for one object.
 */
using InferFeatures = std::vector<std::pair<std::string, InferFeature>>;

/**
 * Defines an alias for std::vector<std::pair<std::string, std::string>>.
 */
using StringPairs = std::vector<std::pair<std::string, std::string>>;

/**
 * @class InferObject
 *
 * @brief InferObject is a class holding the information of an object.
 */
class InferObject {
 public:
  /**
   * @brief Constructs an instance storing inference results.
   *
   * @return No return value.
   */
  InferObject() = default;
  /**
   * @brief Constructs an instance.
   *
   * @return No return value.
   */
  ~InferObject() = default;
  std::string model_name;  ///< The name of the model.
  int id;                  ///< The ID of the classification (label value).
  std::string track_id;    ///< The tracking result.
  float score;             ///< The label score.
  InferBoundingBox bbox;   ///< The object normalized coordinates.
  Collection collection;   ///< User-defined structured information.

  /**
   * @brief Adds the key of an attribute to a specified object.
   *
   * @param[in] key The Key of the attribute you want to add to. See GetAttribute().
   * @param[in] value The value of the attribute.
   *
   * @return Returns true if the attribute has been added successfully. Returns false if the attribute
   *         already existed.
   *
   * @note This is a thread-safe function.
   */
  bool AddAttribute(const std::string& key, const InferAttr& value);

  /**
   * @brief Adds the key pairs of an attribute to a specified object.
   *
   * @param[in] attribute The attribute pair (key, value) to be added.
   *
   * @return Returns true if the attribute has been added successfully. Returns false if the attribute
   *         has already existed.
   *
   * @note This is a thread-safe function.
   */
  bool AddAttribute(const std::pair<std::string, InferAttr>& attribute);

  /**
   * @brief Gets an attribute by key.
   *
   * @param[in] key The key of an attribute you want to query. See AddAttribute().
   *
   * @return Returns the attribute key. If the attribute
   *         does not exist, InferAttr::id will be set to -1.
   *
   * @note This is a thread-safe function.
   */
  InferAttr GetAttribute(const std::string& key);

  /**
   * @brief Adds the key of the extended attribute to a specified object.
   *
   * @param[in] key The key of an attribute. You can get this attribute by key. See GetExtraAttribute().
   * @param[in] value The value of the attribute.
   *
   * @return Returns true if the attribute has been added successfully. Returns false if the attribute
   *         has already existed in the object.
   *
   * @note This is a thread-safe function.
   */
  bool AddExtraAttribute(const std::string& key, const std::string& value);

  /**
   * @brief Adds the key pairs of the extended attributes to a specified object.
   *
   * @param[in] attributes Attributes to be added.
   *
   * @return Returns true if the attribute has been added successfully. Returns false if the attribute
   *         has already existed.
   * @note This is a thread-safe function.
   */
  bool AddExtraAttributes(const std::vector<std::pair<std::string, std::string>>& attributes);

  /**
   * @brief Gets an extended attribute by key.
   *
   * @param[in] key The key of an identified attribute. See AddExtraAttribute().
   *
   * @return Returns the attribute that is identified by the key. If the attribute
   *         does not exist, returns NULL.
   *
   * @note This is a thread-safe function.
   */
  std::string GetExtraAttribute(const std::string& key);

  /**
   * @brief Removes an attribute by key.
   *
   * @param[in] key The key of an attribute you want to remove. See AddAttribute.
   *
   * @return Return true.
   *
   * @note This is a thread-safe function.
   */
  bool RemoveExtraAttribute(const std::string& key);

  /**
   * @brief Gets all extended attributes of an object.
   *
   * @return Returns all extended attributes.
   *
   * @note This is a thread-safe function.
   */
  StringPairs GetExtraAttributes();

  /**
   * @brief Adds the key of feature to a specified object.
   *
   * @param[in] key The Key of feature you want to add the feature to. See GetFeature.
   * @param[in] value The value of the feature.
   *
   * @return Returns true if the feature is added successfully. Returns false if the feature
   *         identified by the key already exists.
   *
   * @note This is a thread-safe function.
   */
  bool AddFeature(const std::string &key, const InferFeature &feature);

  /**
   * @brief Gets an feature by key.
   *
   * @param[in] key The key of an feature you want to query. See AddFeature.
   *
   * @return Return the feature of the key. If the feature identified by the key
   *         is not exists, CNInferFeature will be empty.
   *
   * @note This is a thread-safe function.
   */
  InferFeature GetFeature(const std::string &key);

  /**
   * @brief Gets the features of an object.
   *
   * @return Returns the features of an object.
   *
   * @note This is a thread-safe function.
   */
  InferFeatures GetFeatures();

 private:
  // InferAttr includes: id, value, score
  std::map<std::string, InferAttr> attributes_;
  std::map<std::string, std::string> extra_attributes_;
  std::map<std::string, InferFeature> features_;
  std::mutex attribute_mutex_;
  std::mutex feature_mutex_;
};

/*!
 * Defines an alias for the std::shared_ptr<InferObject>. InferObjectPtr now denotes a shared pointer of inference
 * objects.
 */
using InferObjectPtr = std::shared_ptr<InferObject>;

/**
 * @struct InferObjs
 *
 * @brief InferObjs is a structure holding inference results.
 */
struct InferObjs : public NonCopyable {
  std::vector<std::shared_ptr<InferObject>> objs_;  /// The objects storing inference results.
  std::mutex mutex_;   /// mutex of InferObjs
};

/**
 * @struct OneInferData
 *
 * @brief OneInferData is a structure holding the information of raw inference input & outputs.
 */
struct OneInferData {
  // infer input
  DataFormat input_fmt_;               /*!< The input image's pixel format.*/
  int input_width_;                      /*!< The input image's width.*/
  int input_height_;                     /*!< The input image's height. */
  std::shared_ptr<void> input_cpu_addr_; /*!< The input data's CPU address.*/
  size_t input_size_;                    /*!< The input data's size. */

  // infer output
  std::vector<std::shared_ptr<void>> output_cpu_addr_; /*!< The corresponding inference outputs to the input data. */
  std::vector<size_t> output_sizes_;                   /*!< The inference outputs' sizes.*/
  size_t output_num_;                                  /*!< The inference output count.*/
};

/**
 * @struct InferData
 *
 * @brief InferData is a structure holding a map between module name and OneInferData.
 */
struct InferData : public NonCopyable {
  std::map<std::string, std::vector<std::shared_ptr<OneInferData>>> datas_map_;
  /*!< The map between module name and OneInferData.*/
  std::mutex mutex_; /*!< Inference data mutex.*/
};


/**
 * @brief 全局函数：获取指定格式、指定平面索引、指定高度、指定步长下的平面字节数
 * 
 * @param fmt 数据格式
 * @param plane_idx 平面索引
 * @param height 高度
 * @param stride 步长数组，字节为单位
 * @return size_t 平面字节数
 */
inline size_t get_plane_bytes(DataFormat fmt, int plane_idx, int height, int stride[]) {
  if (plane_idx < 0 || plane_idx >= FormatPlanes(fmt)) return 0;
  switch (fmt) {
    case DataFormat::PIXEL_FORMAT_BGR24:
    case DataFormat::PIXEL_FORMAT_RGB24:
      return height * stride[0];
    case DataFormat::PIXEL_FORMAT_YUV420_NV12:
    case DataFormat::PIXEL_FORMAT_YUV420_NV21:
      if (0 == plane_idx)
        return height * stride[0];
      else if (1 == plane_idx)
        return std::ceil(1.0 * height * stride[1] / 2);
      else
        LOGF(FRAME) << "plane index wrong.";
    default:
      return 0;
  }
  return 0;
}

/*!
 * Defines an alias for the std::shared_ptr<DataFrame>.
 */
using DataFramePtr = std::shared_ptr<DataFrame>;
/*!
 * Defines an alias for the std::shared_ptr<InferObjs>.
 */
using InferObjsPtr = std::shared_ptr<InferObjs>;
/*!
 * Defines an alias for the std::vector<std::shared_ptr<InferObject>>.
 */
using ObjsVec = std::vector<std::shared_ptr<InferObject>>;
/*!
 * Defines an alias for the std::shared_ptr<InferData>.
 */
using InferDataPtr = std::shared_ptr<InferData>;

inline constexpr char kDataFrameTag[] = "DataFrame"; /*!< value type in FrameInfo::Collection : DataFramePtr. */
inline constexpr char kInferObjsTag[] = "InferObjs"; /*!< value type in FrameInfo::Collection : InferObjsPtr. */
inline constexpr char kInferDataTag[] = "InferData"; /*!< value type in FrameInfo::Collection : InferDataPtr. */


}  // namespace cnstream
#endif