cmake_minimum_required(VERSION 3.22)

if(NOT DEFINED VSTREAM_ROOT_PATH)
    set(VSTREAM_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..
        CACHE INTERNAL "Top-level directory of this project")
endif()

set(VSTREAM_MODULES_3RDPARTY_INCLUDE "")
set(VSTREAM_MODULES_3RDPARTY_LIBS "")
set(VSTREAM_MODULES_DEPENDENCIES "")

# ---[ gflags
include(${VSTREAM_ROOT_PATH}/cmake/FindGFlags.cmake)
if(GFLAGS_FOUND)
  message(STATUS "Using installed GFlags for modules unittest")
  list(APPEND VSTREAM_MODULES_3RDPARTY_INCLUDE ${GFLAGS_INCLUDE_DIRS})
  list(APPEND VSTREAM_MODULES_3RDPARTY_LIBS ${GFLAGS_LIBRARIES})
else()
  message(FATAL_ERROR "GFlags not found! Please ensure GFlags is installed and set GFLAGS_ROOT_DIR or CMAKE_PREFIX_PATH")
endif()

# ---[ glog
include(${VSTREAM_ROOT_PATH}/cmake/FindGlog.cmake)
if(GLOG_FOUND)
  message(STATUS "Using installed GLog for modules unittest")
  list(APPEND VSTREAM_MODULES_3RDPARTY_INCLUDE ${GLOG_INCLUDE_DIRS})
  list(APPEND VSTREAM_MODULES_3RDPARTY_LIBS ${GLOG_LIBRARIES})
else()
  message(FATAL_ERROR "GLog not found! Please ensure GLog is installed and set GLOG_ROOT_DIR or CMAKE_PREFIX_PATH")
endif()

# ---[ libyuv
if(DEFINED VSTREAM_HAS_BUILT_LIBYUV AND VSTREAM_HAS_BUILT_LIBYUV)
  if(TARGET yuv)  # 根据 libyuv 内的 cmake 文件确定
    list(APPEND VSTREAM_MODULES_3RDPARTY_LIBS yuv)
  elseif(TARGET yuv_shared)
    list(APPEND VSTREAM_MODULES_3RDPARTY_LIBS yuv_shared)
  else()
    message(FATAL_ERROR "libyuv not found. Please ensure libyuv is installed")
  endif()
  list(APPEND VSTREAM_MODULES_3RDPARTY_INCLUDE ${VSTREAM_ROOT_PATH}/3rdparty/libyuv/include)
else()
  message(FATAL_ERROR "libyuv has not built. Set BUILD_LIBYUV=ON in root CMakeLists.txt to build")
endif()

# ---[ FFmpeg
include(${VSTREAM_ROOT_PATH}/cmake/FindFFmpeg.cmake)
if(FFMPEG_FOUND)
  list(APPEND VSTREAM_MODULES_3RDPARTY_INCLUDE ${FFMPEG_INCLUDE_DIR})
  list(APPEND VSTREAM_MODULES_3RDPARTY_LIBS ${FFMPEG_LIBRARIES})
  message(STATUS "----- FFmpeg found Includes: ${FFMPEG_INCLUDE_DIR} Libraries: ${FFMPEG_LIBRARIES}")
else()
  message(FATAL_ERROR "FFmpeg not found! Please ensure FFmpeg is installed and set FFMPEG_ROOT_DIR")
endif()

# ---[ opencv
include(${VSTREAM_ROOT_PATH}/cmake/FindOpenCV.cmake)
if(OpenCV_FOUND)
  list(APPEND VSTREAM_MODULES_3RDPARTY_INCLUDE ${OpenCV_INCLUDE_DIRS})
  list(APPEND VSTREAM_MODULES_3RDPARTY_LIBS ${OpenCV_LIBS})
else()
  message(FATAL_ERROR "OpenCV not found. Please ensure OpenCV is installed")
endif()

# ---[ 平台专属依赖（RKNN / TensorRT / CUDA NPP 等）
# 由 cmake/platforms/<id>.cmake 声明，经 vstream_platform_configure() 查找，
# 通过 vstream_platform INTERFACE 目标传播到本目录的 target。

# ---[ nlohmann/json
list(APPEND VSTREAM_MODULES_3RDPARTY_INCLUDE ${VSTREAM_ROOT_PATH}/3rdparty/nlohmann-json/include)

# ---[ framework
include(${VSTREAM_ROOT_PATH}/cmake/HaveFrameworkTarget.cmake)
have_framework_target(${VSTREAM_ROOT_PATH})
if(HAVE_FRAMEWORK_TARGET)
  list(APPEND VSTREAM_MODULES_DEPENDENCIES vstream_core)
else()
  message(FATAL_ERROR "framework target not found")
endif()

if(VSTREAM_BUILD_TESTS)
  add_definitions(-DVSTREAM_UNIT_TEST)
endif()

option(VSTREAM_MODULES_BUILD_SOURCE "build module source" ON)
option(VSTREAM_MODULES_BUILD_SINK "build module sink" ON)
option(VSTREAM_MODULES_BUILD_INFERENCE "build module inference" ON)
option(VSTREAM_MODULES_BUILD_OSD "build module osd" ON)
option(VSTREAM_MODULES_BUILD_VA "build stream_va" ON)
option(VSTREAM_MODULES_BUILD_CONTRIB "build module_contrib (business-specific extension modules)" ON)

set(module_list "")
if(VSTREAM_MODULES_BUILD_SOURCE)
  list(APPEND module_list source)
endif()

if(VSTREAM_MODULES_BUILD_SINK)
  list(APPEND module_list sink)
endif()

if(VSTREAM_MODULES_BUILD_INFERENCE)
  list(APPEND module_list inference)
endif()

if(VSTREAM_MODULES_BUILD_OSD)
  list(APPEND module_list osd)
endif()

# 只保留真实存在的模块目录：否则 target_include_directories /
# install(DIRECTORY) 会指向不存在的路径，安装阶段报 "file INSTALL cannot find"
set(_existing_module_list "")
foreach(module ${module_list})
  if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${module}")
    list(APPEND _existing_module_list ${module})
  else()
    message(STATUS "modules: skip '${module}' (directory not found)")
  endif()
endforeach()
set(module_list ${_existing_module_list})

file(GLOB srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

vstream_platform_sources(util_src ${CMAKE_CURRENT_SOURCE_DIR}/util)
list(APPEND srcs ${util_src})

# 平台依赖（CUDA NPP / TensorRT / RKNN）由 vstream_platform 目标统一提供

# 递归添加模块的源文件: 要求每个 module 目录都是 include src 目录结构
foreach(module ${module_list})
  vstream_platform_sources(${module}_srcs ${CMAKE_CURRENT_SOURCE_DIR}/${module})
  list(APPEND srcs ${${module}_srcs})
endforeach()

# 前后处理模块 / filter 模块 / 输入派生模块
set(proc_common_srcs "")
file(GLOB proc_common_pre_src ${CMAKE_CURRENT_SOURCE_DIR}/proc/common/preproc/*.cpp)
file(GLOB proc_common_post_src ${CMAKE_CURRENT_SOURCE_DIR}/proc/common/postproc/*.cpp)
file(GLOB proc_common_filter_src ${CMAKE_CURRENT_SOURCE_DIR}/proc/common/filter/*.cpp)
file(GLOB proc_common_derive_src ${CMAKE_CURRENT_SOURCE_DIR}/proc/common/derive/*.cpp)
list(APPEND proc_common_srcs
  ${proc_common_pre_src} 
  ${proc_common_post_src} 
  ${proc_common_filter_src}
  ${proc_common_derive_src}
)

list(APPEND srcs ${proc_common_srcs})

# 业务相关性较强的扩展模块（默认编译，可通过 VSTREAM_MODULES_BUILD_CONTRIB=OFF 关闭）
if(VSTREAM_MODULES_BUILD_CONTRIB)
  # 与其它模块保持一致，走平台化收集（module_contrib/src/<platform>/ 也能被收进来）
  vstream_platform_sources(contrib_src ${CMAKE_CURRENT_SOURCE_DIR}/module_contrib)
  list(APPEND srcs ${contrib_src})
endif()

# 平台专属前后处理源码：proc/<sub>/{preproc,postproc}/*.cpp
foreach(_sub ${VSTREAM_PLATFORM_SRC_SUBDIRS})
  if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/proc/${_sub}")
    file(GLOB _proc_sub_srcs
      "${CMAKE_CURRENT_SOURCE_DIR}/proc/${_sub}/preproc/*.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/proc/${_sub}/postproc/*.cpp")
    list(APPEND srcs ${_proc_sub_srcs})
  endif()
endforeach()


if(VSTREAM_MODULES_BUILD_VA)
  add_library(vstream_va SHARED ${srcs})

  # include 时不区分 platform sub, 因此源码需要包含 cuda/ 等子目录
  target_include_directories(vstream_va 
   PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/util/include
   PRIVATE
    ${VSTREAM_MODULES_3RDPARTY_INCLUDE}
  )

  foreach(module ${module_list})
    target_include_directories(vstream_va 
     PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/${module}/include
    )
  endforeach()

  if(VSTREAM_MODULES_BUILD_CONTRIB)
    target_include_directories(vstream_va
     PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/module_contrib/include
    )
  endif()

  target_link_libraries(vstream_va
   PUBLIC
    ${VSTREAM_MODULES_DEPENDENCIES} 
   PRIVATE
    ${VSTREAM_MODULES_3RDPARTY_LIBS})  # vstream_core

  # 平台依赖（CUDA NPP / TensorRT / RKNN 等）与平台编译宏
  vstream_platform_link(vstream_va)
  vstream_platform_apply(vstream_va)

  message(STATUS "vstream_va enabled")
endif()

if(VSTREAM_BUILD_TESTS)
  add_subdirectory(unittest)
endif()


if(VSTREAM_MODULES_BUILD_VA)
  # 不声明 EXPORT：工程内没有配对的 install(EXPORT)，且 vstream_va 的
  # PUBLIC include 目录是源码树绝对路径，install(EXPORT) 会直接报错。
  install(TARGETS vstream_va
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )
endif()

# 各模块头文件分别安装到 include/vstream/modules/<module>
foreach(module ${module_list})
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${module}/include/
    DESTINATION include/vstream/modules/${module}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
  )
endforeach()

# module_contrib 头文件
if(VSTREAM_MODULES_BUILD_CONTRIB)
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module_contrib/include/
    DESTINATION include/vstream/modules/module_contrib
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
  )
endif()

# util 头文件
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/util/include/
  DESTINATION include/vstream/modules/util
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# libyuv 头文件
if(DEFINED VSTREAM_HAS_BUILT_LIBYUV AND VSTREAM_HAS_BUILT_LIBYUV)
  install(DIRECTORY ${VSTREAM_ROOT_PATH}/3rdparty/libyuv/include/
    DESTINATION include/vstream/libyuv
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
  )
endif()
