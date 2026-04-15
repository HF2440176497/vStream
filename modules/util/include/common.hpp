

#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>

namespace cnstream {

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

}  // namespace utils

}  // namespace cnstream