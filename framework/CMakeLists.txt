
cmake_minimum_required(VERSION 3.13)

if(NOT DEFINED VSTREAM_ROOT_PATH)
    set(VSTREAM_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..
        CACHE INTERNAL "Top-level directory of this project")
endif()

set(VSTREAM_FRAMEWORK_3RDPARTY_INCLUDE "")
set(VSTREAM_FRAMEWORK_3RDPARTY_LIBS "")

# ---[ gflags
include(${VSTREAM_ROOT_PATH}/cmake/FindGFlags.cmake)
if(GFLAGS_FOUND)
  message(STATUS "Using installed GFlags for framework")
  list(APPEND VSTREAM_FRAMEWORK_3RDPARTY_INCLUDE ${GFLAGS_INCLUDE_DIRS})
  list(APPEND VSTREAM_FRAMEWORK_3RDPARTY_LIBS ${GFLAGS_LIBRARIES})
else()
  message(FATAL_ERROR "GFlags not found! Please ensure GFlags is installed and set GFLAGS_ROOT_DIR or CMAKE_PREFIX_PATH")
endif()

# ---[ glog
include(${VSTREAM_ROOT_PATH}/cmake/FindGlog.cmake)
if(GLOG_FOUND)
  message(STATUS "Using installed GLog for framework")
  list(APPEND VSTREAM_FRAMEWORK_3RDPARTY_INCLUDE ${GLOG_INCLUDE_DIRS})
  list(APPEND VSTREAM_FRAMEWORK_3RDPARTY_LIBS ${GLOG_LIBRARIES})
else()
  message(FATAL_ERROR "GLog not found! Please ensure GLog is installed and set GLOG_ROOT_DIR or CMAKE_PREFIX_PATH")
endif()

# ---[ nlohmann/json
list(APPEND VSTREAM_FRAMEWORK_3RDPARTY_INCLUDE ${VSTREAM_ROOT_PATH}/3rdparty/nlohmann-json/include)

# ---[ libyuv
if(DEFINED VSTREAM_HAS_BUILT_LIBYUV AND VSTREAM_HAS_BUILT_LIBYUV)
  if(TARGET yuv)  # 根据 libyuv 内的 cmake 文件确定
    list(APPEND VSTREAM_FRAMEWORK_3RDPARTY_LIBS yuv)
  elseif(TARGET yuv_shared)
    list(APPEND VSTREAM_FRAMEWORK_3RDPARTY_LIBS yuv_shared)
  else()
    message(FATAL_ERROR "libyuv not found. Please ensure libyuv is installed")
  endif()
  list(APPEND VSTREAM_FRAMEWORK_3RDPARTY_INCLUDE ${VSTREAM_ROOT_PATH}/3rdparty/libyuv/include)
else()
  message(FATAL_ERROR "libyuv has not built. Set BUILD_LIBYUV=ON in root CMakeLists.txt to build")
endif()

# ---[ opencv
include(${VSTREAM_ROOT_PATH}/cmake/FindOpenCV.cmake)
if(OpenCV_FOUND)
  list(APPEND VSTREAM_FRAMEWORK_3RDPARTY_INCLUDE ${OpenCV_INCLUDE_DIRS})
  list(APPEND VSTREAM_FRAMEWORK_3RDPARTY_LIBS ${OpenCV_LIBS})
else()
  message(FATAL_ERROR "OpenCV not found. Please ensure OpenCV is installed")
endif()

if(VSTREAM_BUILD_TESTS)
  add_definitions(-DUNIT_TEST)
endif()

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/core/src core_srcs)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/core/src/private private_srcs)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/core/src/profiler profiler_srcs)

add_library(vstream_core SHARED ${core_srcs} ${private_srcs} ${profiler_srcs})

target_include_directories(vstream_core 
 PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/core/include
 PRIVATE
  ${VSTREAM_FRAMEWORK_3RDPARTY_INCLUDE}
)

target_link_libraries(vstream_core 
 PRIVATE
  ${VSTREAM_FRAMEWORK_3RDPARTY_LIBS}
)

# unitest
if(VSTREAM_BUILD_TESTS)
  add_subdirectory(unittest)
endif()
