
cmake_minimum_required(VERSION 3.13)

if(NOT DEFINED VSTREAM_ROOT_PATH)
    set(VSTREAM_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..
        CACHE INTERNAL "Top-level directory of this project")
endif()

set(FRAMEWORK_3RDPARTY_LIBS "")
set(FRAMEWORK_DEPENDENCIES "")

# ---[ gflags
include(${VSTREAM_ROOT_PATH}/cmake/FindGFlags.cmake)
if(GFLAGS_FOUND)
  message(STATUS "Using installed GFlags for framework")
  include_directories(${GFLAGS_INCLUDE_DIRS})
  list(APPEND FRAMEWORK_3RDPARTY_LIBS ${GFLAGS_LIBRARIES})
else()
  message(FATAL_ERROR "GFlags not found! Please ensure GFlags is installed and set GFLAGS_ROOT_DIR or CMAKE_PREFIX_PATH")
endif()

# ---[ glog
include(${VSTREAM_ROOT_PATH}/cmake/FindGlog.cmake)
if(GLOG_FOUND)
  message(STATUS "Using installed GLog for framework")
  include_directories(${GLOG_INCLUDE_DIRS})
  list(APPEND FRAMEWORK_3RDPARTY_LIBS ${GLOG_LIBRARIES})
else()
  message(FATAL_ERROR "GLog not found! Please ensure GLog is installed and set GLOG_ROOT_DIR or CMAKE_PREFIX_PATH")
endif()

# ---[ nlohmann/json
include_directories(${VSTREAM_ROOT_PATH}/3rdparty/nlohmann-json/include)

# ---[ libyuv
if(DEFINED has_built_libyuv AND has_built_libyuv)
  if(TARGET yuv)  # 根据 libyuv 内的 cmake 文件确定
    list(APPEND FRAMEWORK_3RDPARTY_LIBS yuv)
    list(APPEND FRAMEWORK_DEPENDENCIES yuv)
    # message(STATUS "----- Using libyuv static library: yuv")
  elseif(TARGET yuv_shared)
    list(APPEND FRAMEWORK_3RDPARTY_LIBS yuv_shared)
    list(APPEND FRAMEWORK_DEPENDENCIES yuv_shared)
    # message(STATUS "----- Using libyuv shared library: yuv_shared")
  else()
    message(FATAL_ERROR "libyuv not found. Please ensure libyuv is installed")
  endif()
  include_directories(${VSTREAM_ROOT_PATH}/3rdparty/libyuv/include)
else()
  message(FATAL_ERROR "libyuv has not built. Set BUILD_LIBYUV=ON in root CMakeLists.txt to build")
endif()


# ---[ opencv
include(${VSTREAM_ROOT_PATH}/cmake/FindOpenCV.cmake)
if(OpenCV_FOUND)
  include_directories(${OpenCV_INCLUDE_DIRS})
  list(APPEND FRAMEWORK_3RDPARTY_LIBS ${OpenCV_LIBS})
else()
  message(FATAL_ERROR "OpenCV not found. Please ensure OpenCV is installed")
endif()

# ---[ source: framework
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/core/include)

if(build_tests)
  add_definitions(-DUNIT_TEST)
endif()

# ============================================================
#  分模块编译选项
# ============================================================
option(build_profiler "Build framework profiler module" ON)
option(build_core "Build framework core module" ON)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/core/src core_srcs)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/core/src/private private_srcs)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/core/src/profiler profiler_srcs)

# note: 前面已设置 include_directories，这里不需要 target_include_directories
if(build_profiler)
  add_library(vstream_framework_profiler OBJECT ${profiler_srcs})
  target_link_libraries(vstream_framework_profiler ${FRAMEWORK_3RDPARTY_LIBS} rt dl pthread)
  add_custom_target(profiler DEPENDS vstream_framework_profiler)
  message(STATUS "Profiler module enabled - targets: vstream_framework_profiler, profiler")
endif()

if(build_core)
  add_library(vstream_core SHARED ${core_srcs} ${private_srcs} ${profiler_srcs})
  target_link_libraries(vstream_core ${FRAMEWORK_3RDPARTY_LIBS} rt dl pthread)
  # set_target_properties(vstream_core PROPERTIES LINK_FLAGS_RELEASE -s)
  add_custom_target(core DEPENDS vstream_core)
  message(STATUS "Core module enabled - targets: vstream_core, core")
endif()

# unitest
if(build_tests)
  add_subdirectory(unittest)
endif()
