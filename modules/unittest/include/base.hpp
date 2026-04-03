
/**
 * @brief 单元测试程序相关全局入口
 *
 * @details 
*/

#ifndef MODULES_UNITEST_INCLUDE_TEST_BASE_HPP_
#define MODULES_UNITEST_INCLUDE_TEST_BASE_HPP_

#include <string>
#include <string.h>
#include <utility>
#include <cstdlib>
#include <unistd.h>
#include <cerrno>
#include <utility>
#include <vector>
#include <memory>

#include <iostream>
#include <fstream>
#include <gtest/gtest.h>
#include <glog/logging.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#define PATH_MAX_LENGTH 1024

/**
 * @brief 获取当前执行程序的路径
 * @return 返回当前执行程序的路径
 * @example /usr/bin/app return /usr/bin/
 **/
std::string GetExePath();
void CheckExePath(const std::string& path);

/**
 * @brief Creates a temp file.
 * @return Returns temp file informations.
 * Return value is a std::pair object, the first value stored temp file fd,
 * and the second value stored temp file name.
 * @note close(fd), unlink(filename) must be called when the temp file is uesd up.
 **/
std::pair<int, std::string> CreateTempFile(const std::string& filename_prefix);

/**
 * @brief 截取自 CNConfigBase::ParseByJSONFile
 * 读取 json 文件内容为 json 字符串
 */
std::string readFile(const char* filename);

DecodeFrame* CreateTestDecodeFrame(DataFormat fmt, int width, int height);

void CleanupTestDecodeFrame(DecodeFrame* frame);

#endif
