

#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>

namespace cnstream {

namespace utils {

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