#ifndef CUDA_INSPECT_MEM_HPP_
#define CUDA_INSPECT_MEM_HPP_

#include <cuda_runtime.h>
#include <nvml.h>

#include <iostream>
#include <string>
#include <sstream>

#include <thread>
#include <chrono>

namespace cnstream {

/**
 * 使用 libnvidia-ml 库查询 GPU 信息
 * libnvidia-ml 是 NVIDIA 提供的管理 GPU 的库，是 nvidia-driver 提供的
 */

class CudaMemInspect {
 public:
  CudaMemInspect(int device_id = 0) : device_id_(device_id), initialized_(false) {
    nvmlReturn_t result = nvmlInit();
    if (result == NVML_SUCCESS) {
      result = nvmlDeviceGetHandleByIndex(device_id, &device_handle_);
      if (result == NVML_SUCCESS) {
        initialized_ = true;
      }
    }
  }

  ~CudaMemInspect() {
    if (initialized_) {
      nvmlShutdown();
    }
  }

  bool IsAvailable() const { return initialized_; }

  std::string GetDeviceName() const {
    if (!initialized_) return "N/A";
    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    nvmlReturn_t result = nvmlDeviceGetName(device_handle_, name, NVML_DEVICE_NAME_BUFFER_SIZE);
    if (result == NVML_SUCCESS) {
      return std::string(name);
    }
    return "Unknown";
  }

  std::string GetDriverVersion() const {
    if (!initialized_) return "N/A";
    char version[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
    nvmlReturn_t result = nvmlSystemGetNVMLVersion(version, NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE);
    if (result == NVML_SUCCESS) {
      return std::string(version);
    }
    return "Unknown";
  }

  std::string GetCUDAVersion() const {
    if (!initialized_) return "N/A";
    int cuda_version = 0;
    nvmlReturn_t result = nvmlSystemGetCudaDriverVersion(&cuda_version);
    if (result == NVML_SUCCESS) {
      int major = cuda_version / 1000;
      int minor = (cuda_version % 1000) / 10;
      return std::to_string(major) + "." + std::to_string(minor);
    }
    return "Unknown";
  }

  std::string GetMemoryInfo() const {
    if (!initialized_) return "N/A";
    nvmlMemory_t memory;
    nvmlReturn_t result = nvmlDeviceGetMemoryInfo(device_handle_, &memory);
    if (result != NVML_SUCCESS) {
      return "Failed to get memory info";
    }

    std::ostringstream oss;
    oss << "Total: " << FormatSize(memory.total) << ", "
        << "Used: " << FormatSize(memory.used) << ", "
        << "Free: " << FormatSize(memory.free);
    return oss.str();
  }

  float GetMemoryUsagePercent() const {
    if (!initialized_) return 0.0f;
    nvmlMemory_t memory;
    nvmlReturn_t result = nvmlDeviceGetMemoryInfo(device_handle_, &memory);
    if (result != NVML_SUCCESS || memory.total == 0) {
      return 0.0f;
    }
    return static_cast<float>(memory.used) * 100.0f / static_cast<float>(memory.total);
  }

  std::string GetUtilization() const {
    if (!initialized_) return "N/A";
    nvmlUtilization_t util;
    nvmlReturn_t result = nvmlDeviceGetUtilizationRates(device_handle_, &util);
    if (result != NVML_SUCCESS) {
      return "Failed to get utilization";
    }

    std::ostringstream oss;
    oss << "GPU: " << util.gpu << "%, "
        << "Memory: " << util.memory << "%";
    return oss.str();
  }

  int GetGPUUtilization() const {
    if (!initialized_) return 0;
    nvmlUtilization_t util;
    nvmlReturn_t result = nvmlDeviceGetUtilizationRates(device_handle_, &util);
    if (result != NVML_SUCCESS) {
      return 0;
    }
    return static_cast<int>(util.gpu);
  }

  int GetMemoryUtilization() const {
    if (!initialized_) return 0;
    nvmlUtilization_t util;
    nvmlReturn_t result = nvmlDeviceGetUtilizationRates(device_handle_, &util);
    if (result != NVML_SUCCESS) {
      return 0;
    }
    return static_cast<int>(util.memory);
  }

  std::string GetTemperature() const {
    if (!initialized_) return "N/A";
    unsigned int temp = 0;
    nvmlReturn_t result = nvmlDeviceGetTemperature(device_handle_, NVML_TEMPERATURE_GPU, &temp);
    if (result != NVML_SUCCESS) {
      return "Failed to get temperature";
    }
    return std::to_string(temp) + "C";
  }

  std::string GetPowerUsage() const {
    if (!initialized_) return "N/A";
    unsigned int power = 0;
    nvmlReturn_t result = nvmlDeviceGetPowerUsage(device_handle_, &power);
    if (result != NVML_SUCCESS) {
      return "Failed to get power usage";
    }
    return std::to_string(power / 1000) + "W";
  }

  std::string GetAllInfo() const {
    std::ostringstream oss;
    oss << "======================================\n"
        << "         GPU Information              \n"
        << "======================================\n"
        << "Device:          " << GetDeviceName() << "\n"
        << "Driver Version:  " << GetDriverVersion() << "\n"
        << "CUDA Version:    " << GetCUDAVersion() << "\n"
        << "----------------------------------------\n"
        << "Memory Info:     " << GetMemoryInfo() << "\n"
        << "Memory Usage:    " << GetMemoryUsagePercent() << "%\n"
        << "----------------------------------------\n"
        << "Utilization:     " << GetUtilization() << "\n"
        << "Temperature:     " << GetTemperature() << "\n"
        << "Power Usage:     " << GetPowerUsage() << "\n"
        << "======================================";
    return oss.str();
  }

  std::string GetBriefInfo() const {
    std::ostringstream oss;
    oss << "[GPU " << device_id_ << "] "
        << GetDeviceName() << " | "
        << "Mem: " << GetMemoryUsagePercent() << "% | "
        << "GPU: " << GetGPUUtilization() << "% | "
        << GetTemperature() << " | "
        << GetPowerUsage();
    return oss.str();
  }

 private:
  static std::string FormatSize(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_index = 0;
    double size = static_cast<double>(bytes);
    while (size >= 1024.0 && unit_index < 4) {
      size /= 1024.0;
      unit_index++;
    }
    std::ostringstream oss;
    oss.precision(2);
    oss << std::fixed << size << units[unit_index];
    return oss.str();
  }

  int device_id_;
  bool initialized_;
  nvmlDevice_t device_handle_;
};  // end CudaMemInspect


}  // namespace cnstream

#endif  // CUDA_INSPECT_MEM_HPP_
