cmake_minimum_required(VERSION 3.13)

if(NOT DEFINED VSTREAM_ROOT_PATH)
  set(VSTREAM_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..
    CACHE INTERNAL "Top-level directory of this project")
endif()

set(VSTREAM_TOOLS_3RDPARTY_INCLUDE "")
set(VSTREAM_TOOLS_3RDPARTY_LIBS "")
set(VSTREAM_TOOLS_DEPENDENCIES "")

# if (VSTREAM_MODULES_BUILD_VA)
#   include(${VSTREAM_ROOT_PATH}/cmake/HaveFrameworkTarget.cmake)
#   have_modules_target(${VSTREAM_ROOT_PATH})
#     include_directories(
#       ${VSTREAM_ROOT_PATH}/modules
#       ${VSTREAM_ROOT_PATH}/modules/util/include
#     )
#     if (module_list)
#       foreach(module ${module_list})
#         include_directories(${VSTREAM_ROOT_PATH}/modules/${module}/include)
#       endforeach()
#     endif()
#   if(HAVE_MODULES_TARGET)
#     list(APPEND VSTREAM_TOOLS_DEPENDENCIES vstream_va)
#   endif()
# endif()

# ---[ gflags
include(${VSTREAM_ROOT_PATH}/cmake/FindGFlags.cmake)
if(GFLAGS_FOUND)
  message(STATUS "Using installed GFlags for tools")
  list(APPEND VSTREAM_TOOLS_3RDPARTY_INCLUDE ${GFLAGS_INCLUDE_DIRS})
  list(APPEND VSTREAM_TOOLS_3RDPARTY_LIBS ${GFLAGS_LIBRARIES})
else()
  message(FATAL_ERROR "GFlags not found! Please ensure GFlags is installed and set GFLAGS_ROOT_DIR or CMAKE_PREFIX_PATH")
endif()

# ---[ glog
include(${VSTREAM_ROOT_PATH}/cmake/FindGlog.cmake)
if(GLOG_FOUND)
  message(STATUS "Using installed GLog for tools")
  list(APPEND VSTREAM_TOOLS_3RDPARTY_INCLUDE ${GLOG_INCLUDE_DIRS})
  list(APPEND VSTREAM_TOOLS_3RDPARTY_LIBS ${GLOG_LIBRARIES})
else()
  message(FATAL_ERROR "GLog not found! Please ensure GLog is installed and set GLOG_ROOT_DIR or CMAKE_PREFIX_PATH")
endif()

# ---[ nlohmann/json
include_directories(${VSTREAM_ROOT_PATH}/3rdparty/nlohmann-json/include)

if(VSTREAM_BUILD_TESTS)
  add_definitions(-DVSTREAM_UNIT_TEST)
endif()

# ---[ opencv
include(${VSTREAM_ROOT_PATH}/cmake/FindOpenCV.cmake)
if(OpenCV_FOUND)
  list(APPEND VSTREAM_TOOLS_3RDPARTY_INCLUDE ${OpenCV_INCLUDE_DIRS})
  list(APPEND VSTREAM_TOOLS_3RDPARTY_LIBS ${OpenCV_LIBS})
else()
  message(FATAL_ERROR "OpenCV not found. Please ensure OpenCV is installed")
endif()

# ---[ libyuv
if(DEFINED VSTREAM_HAS_BUILT_LIBYUV AND VSTREAM_HAS_BUILT_LIBYUV)
  if(TARGET yuv)
    list(APPEND VSTREAM_TOOLS_3RDPARTY_LIBS yuv)
  elseif(TARGET yuv_shared)
    list(APPEND VSTREAM_TOOLS_3RDPARTY_LIBS yuv_shared)
  else()
    message(FATAL_ERROR "libyuv not found. Please ensure libyuv is installed")
  endif()
  list(APPEND VSTREAM_TOOLS_3RDPARTY_INCLUDE ${VSTREAM_ROOT_PATH}/3rdparty/libyuv/include)
else()
  message(FATAL_ERROR "libyuv has not built. Set BUILD_LIBYUV=ON in root CMakeLists.txt to build")
endif()

# ---[ framework and modules
include(${VSTREAM_ROOT_PATH}/cmake/HaveFrameworkTarget.cmake)
have_framework_target(${VSTREAM_ROOT_PATH})
have_modules_target(${VSTREAM_ROOT_PATH})
if(HAVE_FRAMEWORK_TARGET AND HAVE_MODULES_TARGET)
  list(APPEND VSTREAM_TOOLS_DEPENDENCIES vstream_core vstream_va)
endif()

# ---[ Options
option(VSTREAM_BUILD_DEMO "build demo executable file" ON)

if(VSTREAM_BUILD_DEMO)
  add_subdirectory(demo)
endif()

